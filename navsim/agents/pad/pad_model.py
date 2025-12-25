from typing import Dict
import numpy as np
import torch
import torch.nn as nn
from .score_module.scorer import Scorer
from .traj_refiner import Traj_refiner
from .bevformer.image_encoder import ImgEncoder
from .bevformer.transformer_decoder import MLP
from navsim.agents.pad.resnet_encoder import Encoder_Resnet34
from navsim.agents.pad.geometry import calculate_birds_eye_view_parameters, VoxelsSumming, pose_vec2mat
from navsim.agents.pad.decoder import Decoder


def pack_sequence_dim(x):
    b, s = x.shape[:2]
    return x.view(b * s, *x.shape[2:])


def unpack_sequence_dim(x, b, s):
    return x.view(b, s, *x.shape[1:])


class PadModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self._config = config
        self.poses_num=config.num_poses
        self.state_size=3

        self._backbone = ImgEncoder(config)

        self.command_num=config.command_num

        self.hist_encoding = nn.Linear(11, config.tf_d_model)
        self.init_feature = nn.Embedding(self.poses_num * config.proposal_num, config.tf_d_model)

        ref_num=config.ref_num
        shared_refiner=Traj_refiner(config)
        self._trajectory_head=nn.ModuleList([shared_refiner for _ in range(ref_num) ] )

        self.scorer = Scorer(config)

        self.b2d=config.b2d

        ##########################################################################
        
        bev_resolution, bev_start_position, bev_dimension = calculate_birds_eye_view_parameters(
            self._config.LIFT_X_BOUND, self._config.LIFT_Y_BOUND, self._config.LIFT_Z_BOUND
        )
        self.bev_resolution = nn.Parameter(bev_resolution, requires_grad=False)
        self.bev_start_position = nn.Parameter(bev_start_position, requires_grad=False)
        self.bev_dimension = nn.Parameter(bev_dimension, requires_grad=False)
        
        self.encoder_downsample = self._config.MODEL_ENCODER_DOWNSAMPLE
        self.encoder_out_channels = self._config.MODEL_ENCODER_OUT_CHANNELS
        
        self.frustum = self.create_frustum()
        self.depth_channels, _, _, _ = self.frustum.shape
        self.discount = self._config.LIFT_DISCOUNT

        # Encoder for VeteranAD
        if self._config.USE_RESNET:
            self.encoder = Encoder_Resnet34(cfg=self._config, D=self.depth_channels)
        
        # Decoder for VeteranAD
        self.decoder = Decoder(
            in_channels=self._config.future_pred_in_channels,
            n_classes=self._config.num_bev_classes,
            config=config
        )
    
    def create_frustum(self):
        # Create grid in image plane
        h, w = self._config.IMAGE_FINAL_DIM
        if self._config.USE_RESNET:
            downsampled_h, downsampled_w = h // self.encoder_downsample , w // self.encoder_downsample
        else:
            downsampled_h, downsampled_w = h // self.encoder_downsample, w // self.encoder_downsample

        # Depth grid
        depth_grid = torch.arange(*self._config.LIFT_D_BOUND, dtype=torch.float)
        depth_grid = depth_grid.view(-1, 1, 1).expand(-1, downsampled_h, downsampled_w)
        n_depth_slices = depth_grid.shape[0]

        # x and y grids
        x_grid = torch.linspace(0, w - 1, downsampled_w, dtype=torch.float)
        x_grid = x_grid.view(1, 1, downsampled_w).expand(n_depth_slices, downsampled_h, downsampled_w)
        y_grid = torch.linspace(0, h - 1, downsampled_h, dtype=torch.float)
        y_grid = y_grid.view(1, downsampled_h, 1).expand(n_depth_slices, downsampled_h, downsampled_w)

        # Dimension (n_depth_slices, downsampled_h, downsampled_w, 3)
        # containing data points in the image: left-right, top-bottom, depth
        frustum = torch.stack((x_grid, y_grid, depth_grid), -1)
        return nn.Parameter(frustum, requires_grad=False)

    def forward(self, features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        ego_status: torch.Tensor = features["ego_status"][:,-1]  # torch.Size([64, 11])
        camera_feature: torch.Tensor = features["camera_feature"]  # torch.Size([64, 4, 3, 448, 768])

        batch_size = ego_status.shape[0]  # 64

        if self.b2d:
            ego_status[:,1:3]=0

        # Image Features
        image_feature, img_feats_resnet = self._backbone(camera_feature,img_metas=features)  # torch.Size([4, 336, 64, 256])

        # image = features['camera_feature_for_bev']  # torch.Size([4, 1, 3, 3, 432, 768])
        image = torch.zeros((batch_size, 1, 3, 3, 432, 768), device=camera_feature.device, dtype=camera_feature.dtype)
        
        intrinsics = features['intrinsics']  # torch.Size([4, 1, 3, 3, 3])
        extrinsics = features['extrinsics']  # torch.Size([4, 1, 3, 4, 4])
        future_egomotion = features['future_egomotion']  # torch.Size([4, 1, 1, 6])
        future_egomotion = future_egomotion.squeeze(-2)

        # Lifting features and project to bird's-eye view
        x, feature_for_attn = self.calculate_birds_eye_view_features(image, intrinsics, extrinsics, future_egomotion, img_feats_resnet)  # torch.Size([4, 1, 64, 128, 256])

        output={}

        # Ego Feature
        ego_feature = self.hist_encoding(ego_status)[:,None]  # torch.Size([64, 1, 256])

        # Initial BEV Proposal Queries
        bev_feature = ego_feature + self.init_feature.weight[None]  # torch.Size([64, 160, 256])

        # Proposal number N=64
        proposal_list = []

        # Iteration number K=4
        for i, refine in enumerate(self._trajectory_head):
            bev_feature, proposal_list = refine(bev_feature, proposal_list, image_feature)

        proposals=proposal_list[-1]  # torch.Size([64, 20, 8, 3])

        output["proposals"] = proposals
        output["proposal_list"] = proposal_list

        # BEV Proposal Features & Scorer
        pred_logit, pred_agents_states, pred_area_logit = self.scorer(proposals, bev_feature)

        output["pred_logit"]=pred_logit  
        # torch.Size([64, 20, 6]), Log-sim Scores

        output["pred_agents_states"]=pred_agents_states  
        # torch.Size([64, 20, 8, 5, 2, 9]), Collision Prediction
        # 2D positions of the four corners with corresponding validity labels, 2*4+1=9

        output["pred_area_logit"]=pred_area_logit  
        # torch.Size([64, 160, 10]), Passability Map
        # on-road and on-route probabilities for all proposals’ simulated states

        pdm_score=torch.sigmoid(pred_logit)[:,:,-1]

        # iPad 中引入的 BEV proposal queries 在功能上等价于并有效替代了 VeteranAD 中的 Position-guided image cross-attention 机制, 同时为模型提供了与 Anchored trajectories 相当的结构化先验信息
        # In iPad, the BEV proposal queries effectively replace the Position-guided image cross-attention used in VeteranAD, while providing structured priors comparable to Anchored trajectories

        ############### Add the decoder of VeteranAD ###############
        status_feature: torch.Tensor = features["status_feature"]  # torch.Size([4, 8])

        # Perceive decoder outputs
        decoder_output = self.decoder(x, status_feature, intrinsics, extrinsics, proposals, bev_feature)

        output["bev_semantic_map"] = decoder_output["bev_semantic_map"]
        output["agent_states"] = decoder_output["agent_states"]
        output["agent_labels"] = decoder_output["agent_labels"]
        output['poses_reg_list'] = decoder_output['poses_reg_list']
        proposals_final = decoder_output['poses_reg_list'][-1]  # torch.Size([B, 20, 8, 3])
        ############### End ###############
        
        token = torch.argmax(pdm_score, dim=1)
        trajectory = proposals_final[torch.arange(batch_size), token]

        output["trajectory"] = trajectory  # torch.Size([64, 8, 3])
        output["pdm_score"] = pdm_score  # torch.Size([64, 20])

        return output

    def get_geometry(self, intrinsics, extrinsics):
        """Calculate the (x, y, z) 3D position of the features.
        """
        rotation, translation = extrinsics[..., :3, :3], extrinsics[..., :3, 3]
        B, N, _ = translation.shape
        # Add batch, camera dimension, and a dummy dimension at the end
        points = self.frustum.unsqueeze(0).unsqueeze(0).unsqueeze(-1)

        # Camera to ego reference frame
        points = torch.cat((points[:, :, :, :, :, :2] * points[:, :, :, :, :, 2:3], points[:, :, :, :, :, 2:3]), 5)
        combined_transformation = rotation.matmul(torch.inverse(intrinsics))
        points = combined_transformation.view(B, N, 1, 1, 1, 3, 3).matmul(points).squeeze(-1)
        points += translation.view(B, N, 1, 1, 1, 3)

        # The 3 dimensions in the ego reference frame are: (forward, sides, height)
        return points
    
    def encoder_forward(self, x, img_feats_resnet):
        # batch, n_cameras, channels, height, width
        b, n, c, h, w = x.shape  # torch.Size([4, 3, 3, 432, 768])

        x = x.view(b * n, c, h, w)  # torch.Size([12, 3, 432, 768])
        x, depth, feature_for_attn = self.encoder(x, img_feats_resnet)  # torch.Size([12, 64, 54, 96])
        x = x.unsqueeze(2).repeat(1, 1, self.depth_channels, 1, 1)

        x = x.view(b, n, *x.shape[1:])
        x = x.permute(0, 1, 3, 4, 5, 2) # channel dimension

        return x, feature_for_attn  # torch.Size([4, 3, 48, 54, 96, 64])

    def projection_to_birds_eye_view(self, x, geometry, future_egomotion):
        """ Adapted from https://github.com/nv-tlabs/lift-splat-shoot/blob/master/src/models.py#L200"""
        # batch, s, n_cameras, depth, height, width, channels
        batch, s, n, d, h, w, c = x.shape  # torch.Size([4, 1, 3, 48, 54, 96, 64])
        output = torch.zeros(
            (batch, s, c, self.bev_dimension[0], self.bev_dimension[1]), dtype=torch.float, device=x.device
        )

        future_egomotion_mat = pose_vec2mat(future_egomotion)  # (3,3,4,4)
        rotation, translation = future_egomotion_mat[..., :3, :3], future_egomotion_mat[..., :3, 3]

        def voxel_to_pixel(geometry_b, x_b):
            # Mask out points that are outside the considered spatial extent.
            mask = (
                    (geometry_b[:, 0] >= 0)
                    & (geometry_b[:, 0] < self.bev_dimension[0])
                    & (geometry_b[:, 1] >= 0)
                    & (geometry_b[:, 1] < self.bev_dimension[1])
                    & (geometry_b[:, 2] >= 0)
                    & (geometry_b[:, 2] < self.bev_dimension[2])
            )
            x_b = x_b[mask]
            geometry_b = geometry_b[mask]

            # Sort tensors so that those within the same voxel are consecutives.
            ranks = (
                    geometry_b[:, 0] * (self.bev_dimension[1] * self.bev_dimension[2])
                    + geometry_b[:, 1] * (self.bev_dimension[2])
                    + geometry_b[:, 2]
            )
            ranks_indices = ranks.argsort()
            x_b, geometry_b, ranks = x_b[ranks_indices], geometry_b[ranks_indices], ranks[ranks_indices]

            # Project to bird's-eye view by summing voxels.
            x_b, geometry_b = VoxelsSumming.apply(x_b, geometry_b, ranks)
            return geometry_b, x_b

        # Number of 3D points
        N = n * d * h * w
        for b in range(batch):
            flow_b = x[b]
            flow_geo = geometry[b]

            #####  transform the 3D voxel to current frame  #####
            for t in range(s):
                if t != s - 1:
                    flow_geo_tmp = flow_geo[:t + 1]
                    rotation_b = rotation[b, t].view(1, 1, 1, 1, 1, 3, 3)
                    translation_b = translation[b, t].view(1, 1, 1, 1, 1, 3)
                    flow_geo_tmp = rotation_b.matmul(flow_geo_tmp.unsqueeze(-1)).squeeze(-1)
                    flow_geo_tmp += translation_b
                    flow_geo[:t + 1] = flow_geo_tmp

            bev_feature = torch.zeros((self.bev_dimension[2], self.bev_dimension[0], self.bev_dimension[1], c),
                                      device=flow_b.device)  # torch.Size([1, 128, 256, 64])

            for t in range(s):
                # flatten x
                x_b = flow_b[t].reshape(N, c)

                # Convert positions to integer indices
                geometry_b = (
                        (flow_geo[t] - (self.bev_start_position - self.bev_resolution / 2.0)) / self.bev_resolution)
                geometry_b = geometry_b.view(N, 3).long()
                geometry_b, x_b = voxel_to_pixel(geometry_b, x_b)

                tmp_bev_feature = torch.zeros((self.bev_dimension[2], self.bev_dimension[0], self.bev_dimension[1], c),
                                              device=flow_b.device)  # torch.Size([1, 128, 256, 64])
                tmp_bev_feature[geometry_b[:, 2], geometry_b[:, 0], geometry_b[:, 1]] = x_b

                bev_feature = bev_feature * self.discount + tmp_bev_feature  # self.discount = 0
                tmp_bev_feature = bev_feature.permute((0, 3, 1, 2))
                tmp_bev_feature = tmp_bev_feature.squeeze(0)
                output[b, t] = tmp_bev_feature

        return output

    def calculate_birds_eye_view_features(self, x, intrinsics, extrinsics, future_egomotion, img_feats_resnet):
        b, s, n, c, h, w = x.shape  # torch.Size([4, 1, 3, 3, 432, 768])
        # Reshape
        x = pack_sequence_dim(x)
        intrinsics = pack_sequence_dim(intrinsics)
        extrinsics = pack_sequence_dim(extrinsics)

        geometry = self.get_geometry(intrinsics, extrinsics)
        x, feature_for_attn = self.encoder_forward(x, img_feats_resnet)  # torch.Size([4, 3, 48, 54, 96, 64])
        x = unpack_sequence_dim(x, b, s)
        geometry = unpack_sequence_dim(geometry, b, s)

        x = self.projection_to_birds_eye_view(x, geometry, future_egomotion)
        return x, feature_for_attn  # torch.Size([4, 1, 64, 128, 256])
