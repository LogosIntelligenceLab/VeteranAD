from typing import Dict
import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from navsim.common.enums import StateSE2Index
from navsim.agents.pad.relative_transformer_qkv import SymmetricFusionTransformer


def linear_relu_ln(embed_dims, in_loops, out_loops, input_dims=None):
    if input_dims is None:
        input_dims = embed_dims
    layers = []
    for _ in range(out_loops):
        for _ in range(in_loops):
            layers.append(nn.Linear(input_dims, embed_dims))
            layers.append(nn.ReLU(inplace=True))
            input_dims = embed_dims
        layers.append(nn.LayerNorm(embed_dims))
    return layers


def gen_sineembed_for_position(pos_tensor, hidden_dim=256):
    """Mostly copy-paste from https://github.com/IDEA-opensource/DAB-DETR/
    """
    half_hidden_dim = hidden_dim // 2
    scale = 2 * math.pi
    dim_t = torch.arange(half_hidden_dim, dtype=torch.float32, device=pos_tensor.device)
    dim_t = 10000 ** (2 * (dim_t // 2) / half_hidden_dim)
    x_embed = pos_tensor[..., 0] * scale
    y_embed = pos_tensor[..., 1] * scale
    pos_x = x_embed[..., None] / dim_t
    pos_y = y_embed[..., None] / dim_t
    pos_x = torch.stack((pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()), dim=-1).flatten(-2)
    pos_y = torch.stack((pos_y[..., 0::2].sin(), pos_y[..., 1::2].cos()), dim=-1).flatten(-2)
    pos = torch.cat((pos_y, pos_x), dim=-1)
    return pos


def bias_init_with_prob(prior_prob):
    """initialize conv/fc bias value according to giving probablity."""
    bias_init = float(-np.log((1 - prior_prob) / prior_prob))
    return bias_init


def nerf_positional_encoding(
    tensor, num_encoding_functions=6, include_input=False, log_sampling=True
) -> torch.Tensor:
    r"""Apply positional encoding to the input.
    Args:
        tensor (torch.Tensor): Input tensor to be positionally encoded.
        encoding_size (optional, int): Number of encoding functions used to compute
            a positional encoding (default: 6).
        include_input (optional, bool): Whether or not to include the input in the
            positional encoding (default: True).
    Returns:
    (torch.Tensor): Positional encoding of the input tensor.
    """
    # TESTED
    # Trivially, the input tensor is added to the positional encoding.
    encoding = [tensor] if include_input else []
    frequency_bands = None
    if log_sampling:
        frequency_bands = 2.0 ** torch.linspace(
            0.0,
            num_encoding_functions - 1,
            num_encoding_functions,
            dtype=tensor.dtype,
            device=tensor.device,
        )
    else:
        frequency_bands = torch.linspace(
            2.0 ** 0.0,
            2.0 ** (num_encoding_functions - 1),
            num_encoding_functions,
            dtype=tensor.dtype,
            device=tensor.device,
        )

    for freq in frequency_bands:
        for func in [torch.sin, torch.cos]:
            encoding.append(func(tensor * freq))

    # Special case, for no positional encoding
    if len(encoding) == 1:
        return encoding[0]
    else:
        return torch.cat(encoding, dim=-1)
    

class MLN(nn.Module):
    def __init__(self, c_dim=256, f_dim=256):
        super().__init__()
        self.c_dim = c_dim
        self.f_dim = f_dim

        self.reduce = nn.Sequential(
            nn.Linear(c_dim, f_dim),
            nn.ReLU(),
        )
        self.gamma = nn.Linear(f_dim, f_dim)
        self.beta = nn.Linear(f_dim, f_dim)
        self.ln = nn.LayerNorm(f_dim, elementwise_affine=False)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.zeros_(self.gamma.weight)
        nn.init.zeros_(self.beta.weight)
        nn.init.ones_(self.gamma.bias)
        nn.init.zeros_(self.beta.bias)

    def forward(self, x, c):
        x = self.ln(x)
        c = self.reduce(c)
        gamma = self.gamma(c)
        beta = self.beta(c)
        out = gamma * x + beta
        return out
    

class GridSampleCrossBEVAttention(nn.Module):
    def __init__(self, embed_dims, num_heads, num_levels=1, in_bev_dims=64, num_points=8, config=None):
        super(GridSampleCrossBEVAttention, self).__init__()
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.num_levels = num_levels
        self.num_points = num_points
        self.config = config
        self.attention_weights = nn.Linear(embed_dims,1)
        self.output_proj = nn.Linear(embed_dims, embed_dims)
        self.dropout = nn.Dropout(0.1)

        self.value_proj = nn.Sequential(
            nn.Conv2d(in_bev_dims, 256, kernel_size=(3, 3), stride=(1, 1), padding=1,bias=True),
            nn.ReLU(inplace=True),
        )

        self.init_weight()

    def init_weight(self):

        nn.init.constant_(self.attention_weights.weight, 0)
        nn.init.constant_(self.attention_weights.bias, 0)

        nn.init.xavier_uniform_(self.output_proj.weight)
        nn.init.constant_(self.output_proj.bias, 0)

    def forward(self, queries, traj_points, bev_feature, spatial_shape, step_idx):
        """
        Args:
            queries: input features with shape of (bs, num_queries, embed_dims)
            traj_points: trajectory points with shape of (bs, num_queries, num_points, 2)
            bev_feature: bev features with shape of (bs, embed_dims, height, width)
            spatial_shapes: (height, width)

        """

        bs, num_queries, num_points, _ = traj_points.shape  # torch.Size([B, 20, 8, 2])
        
        # Normalize trajectory points to [-1, 1] range for grid_sample
        normalized_trajectory = traj_points.clone()
        normalized_trajectory[..., 0] = normalized_trajectory[..., 0] / self.config.lidar_max_y
        normalized_trajectory[..., 1] = normalized_trajectory[..., 1] / self.config.lidar_max_x

        normalized_trajectory = normalized_trajectory[..., [1, 0]]  # Swap x and y
        
        attention_weights = self.attention_weights(queries)
        attention_weights = attention_weights.view(bs, num_queries, 1).softmax(-1)  # torch.Size([B, 20, 8])

        value = self.value_proj(bev_feature)  # torch.Size([B, 256, 64, 64])
        grid = normalized_trajectory.view(bs, num_queries, num_points, 2)  # torch.Size([B, 20, 8, 2])
        
        grid = grid[:,:,step_idx]  # torch.Size([B, 20, 2])
        grid = grid.unsqueeze(2)  # torch.Size([B, 20, 1, 2])

        # Sample features
        sampled_features = torch.nn.functional.grid_sample(
            value, 
            grid, 
            mode='bilinear', 
            padding_mode='zeros', 
            align_corners=False
        )  # bs, C, num_queries, num_points torch.Size([B, 256, 20, 8])

        attention_weights = attention_weights.unsqueeze(1)  # torch.Size([B, 1, 20, 8])
        out = (attention_weights * sampled_features).squeeze(-1) 
        out = out.permute(0, 2, 1).contiguous()  # bs, num_queries, C torch.Size([B, 20, 256])
        out = self.output_proj(out)

        return self.dropout(out) + queries


class DiffMotionPlanningRefinementModule(nn.Module):
    def __init__(
        self,
        embed_dims=256,
        ego_fut_ts=8,
        ego_fut_mode=20,
        if_zeroinit_reg=True,
    ):
        super(DiffMotionPlanningRefinementModule, self).__init__()
        self.embed_dims = embed_dims
        self.ego_fut_ts = ego_fut_ts
        self.ego_fut_mode = ego_fut_mode
        # self.plan_cls_branch = nn.Sequential(
        #     *linear_relu_ln(embed_dims, 1, 2),
        #     nn.Linear(embed_dims, 1),
        # )
        self.plan_reg_branch = nn.Sequential(
            nn.Linear(embed_dims, embed_dims),
            nn.ReLU(),
            nn.Linear(embed_dims, embed_dims),
            nn.ReLU(),
            nn.Linear(embed_dims, 3),
        )
        self.if_zeroinit_reg = False

        self.init_weight()

    def init_weight(self):
        if self.if_zeroinit_reg:
            nn.init.constant_(self.plan_reg_branch[-1].weight, 0)
            nn.init.constant_(self.plan_reg_branch[-1].bias, 0)

        # bias_init = bias_init_with_prob(0.01)
        # nn.init.constant_(self.plan_cls_branch[-1].bias, bias_init)
    
    def forward(
        self,
        traj_feature,
        step_idx,
    ):
        bs, ego_fut_mode, _ = traj_feature.shape

        traj_feature = traj_feature.view(bs, ego_fut_mode,-1)
        
        plan_cls = None
        # if step_idx == self.ego_fut_ts - 1:
        #     plan_cls = self.plan_cls_branch(traj_feature).squeeze(-1)
        # else:
        #     plan_cls = None
        
        traj_delta = self.plan_reg_branch(traj_feature)
        plan_reg = traj_delta

        return plan_reg, plan_cls
    

class CustomTransformerDecoderLayer(nn.Module):
    def __init__(self, 
                 num_poses,
                 d_model,
                 d_ffn,
                 config,
                 ):
        super().__init__()

        self.dropout1 = nn.Dropout(0.1)

        self.cross_bev_attention = GridSampleCrossBEVAttention(
            config.tf_d_model,
            config.tf_num_head,
            num_points=num_poses,
            config=config,
            in_bev_dims=256,
        )

        self.cross_ego_attention = nn.MultiheadAttention(
            config.tf_d_model,
            config.tf_num_head,
            dropout=config.tf_dropout,
            batch_first=True,
        )

        self.ffn = nn.Sequential(
            nn.Linear(config.tf_d_model, config.tf_d_ffn),
            nn.ReLU(),
            nn.Linear(config.tf_d_ffn, config.tf_d_model),
        )

        self.norm2 = nn.LayerNorm(config.tf_d_model)
        self.norm3 = nn.LayerNorm(config.tf_d_model)
        
        self.task_decoder = DiffMotionPlanningRefinementModule(
            embed_dims=config.tf_d_model,
            ego_fut_ts=num_poses,
            ego_fut_mode=20,
        )

        self.rela_transformer = SymmetricFusionTransformer(
            d_model=config.tf_d_model,
            d_edge=config.tf_d_model,
            n_layer=1,
            dropout=config.tf_dropout,
        )

        self.rela_encode = nn.Sequential(
            nn.Linear(2, config.tf_d_model),
            nn.ReLU(),
            nn.Linear(config.tf_d_model, config.tf_d_model),
        )

        self.mln = MLN(c_dim=24, f_dim=config.tf_d_model)

    def forward(self, 
                agents, 
                traj_feature, 
                noisy_traj_points, 
                bev_feature, 
                bev_spatial_shape, 
                agents_query, 
                ego_query, 
                status_encoding,
                intrinsics,  # torch.Size([4, 1, 3, 3, 3])
                extrinsics,  # torch.Size([4, 1, 3, 4, 4])
                step_idx=None,
                ):
        
        current_point = noisy_traj_points[:, :, step_idx]
        current_point = nerf_positional_encoding(current_point)
        traj_feature = self.mln(traj_feature, current_point)

        ###### Position-guided BEV cross-attn ######
        traj_feature = self.cross_bev_attention(
            traj_feature, noisy_traj_points, bev_feature, bev_spatial_shape, step_idx)
        
        ###### Position-guided agent cross-attn ######
        agent_traj = agents['agent_states'][:, :, :2]  # torch.Size([B, 30, 2])
        ego_traj_anchor = noisy_traj_points[:, :, step_idx]  # torch.Size([B, 20, 2])
        relative_dist = ego_traj_anchor.unsqueeze(2) - agent_traj.unsqueeze(1)  # torch.Size([B, 20, 30, 2])
        relative_feat = self.rela_encode(relative_dist)
        traj_feature = self.rela_transformer(traj_feature, agents_query, agents_query, relative_feat)

        traj_feature = traj_feature + self.dropout1(
            self.cross_ego_attention(traj_feature, ego_query, ego_query)[0])
        traj_feature = self.norm2(traj_feature)
        
        traj_feature = self.norm3(self.ffn(traj_feature))
        
        poses_reg, poses_cls = self.task_decoder(traj_feature, step_idx) #bs,20,8,3; bs,20
        poses_reg[...,:2] = poses_reg[...,:2] + noisy_traj_points[:, :, step_idx]
        poses_reg[..., 2] = poses_reg[..., 2].tanh() * np.pi

        return poses_reg, poses_cls
    

def _get_clones(module, N):
    # FIXME: copy.deepcopy() is not defined on nn.module
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class CustomTransformerDecoder(nn.Module):
    def __init__(
        self, 
        decoder_layer, 
        num_layers,
        norm=None,
    ):
        super().__init__()
        torch._C._log_api_usage_once(f"torch.nn.modules.{self.__class__.__name__}")
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
    
    def forward(self, 
                agents,  
                traj_feature,  # torch.Size([B, 20, 256])
                noisy_traj_points,  # torch.Size([B, 20, 8, 2])
                bev_feature,  # torch.Size([B, 256, 64, 64])
                bev_spatial_shape,  # torch.Size([64, 64])
                agents_query,  # torch.Size([B, 30, 256])
                ego_query,  # torch.Size([B, 1, 256])
                status_encoding,  # torch.Size([B, 1, 256])
                intrinsics, 
                extrinsics):
        poses_reg_list = []
        poses_cls_list = []
        traj_points = noisy_traj_points

        total_step = noisy_traj_points.size(-2)
        
        for mod in self.layers:
            poses_reg = torch.zeros(
                (traj_points.size(0), traj_points.size(1), traj_points.size(2), 3), 
                device=traj_points.device
            )
            # poses_cls = torch.zeros(
            #     (traj_points.size(0), traj_points.size(1)), 
            #     device=traj_points.device
            # )
            for idx in range(total_step):
                poses_reg_step, poses_cls_step = mod(
                    agents, 
                    traj_feature, traj_points, 
                    bev_feature, bev_spatial_shape, 
                    agents_query, ego_query, status_encoding,
                    intrinsics, extrinsics,
                    step_idx=idx,
                )
                poses_reg[:,:,idx] = poses_reg_step
                # if poses_cls_step is not None:
                #     poses_cls = poses_cls_step
            poses_reg_list.append(poses_reg)  # torch.Size([B, 20, 8, 3])
            # poses_cls_list.append(poses_cls)  # torch.Size([B, 20])
            traj_points = poses_reg[...,:2].clone().detach()
        
        return poses_reg_list, poses_cls_list


class TrajectoryHead_WO_Diff_Autoregressive(nn.Module):
    """Trajectory prediction head."""

    def __init__(self, 
                 num_poses, 
                 d_ffn, 
                 d_model, 
                 plan_anchor_path,
                 config):
        """
        Initializes trajectory head.
        :param num_poses: number of (x,y,θ) poses to predict
        :param d_ffn: dimensionality of feed-forward network
        :param d_model: input dimensionality
        """
        super(TrajectoryHead_WO_Diff_Autoregressive, self).__init__()

        self._num_poses = num_poses
        self._d_model = d_model
        self._d_ffn = d_ffn
        self.diff_loss_weight = 2.0
        self.ego_fut_mode = 20

        # plan_anchor = np.load(plan_anchor_path)
        # self.plan_anchor = nn.Parameter(
        #     torch.tensor(plan_anchor, dtype=torch.float32),
        #     requires_grad=False,
        # )
        # self.plan_anchor_encoder = nn.Sequential(
        #     *linear_relu_ln(d_model, 1, 1,512),
        #     nn.Linear(d_model, d_model),
        # )

        diff_decoder_layer = CustomTransformerDecoderLayer(
            num_poses=num_poses,
            d_model=d_model,
            d_ffn=d_ffn,
            config=config,
        )

        # here to change the number of layers
        self.diff_decoder = CustomTransformerDecoder(diff_decoder_layer, 2)
    
    def forward(self, 
                agents, 
                ego_query, 
                agents_query, 
                bev_feature, 
                bev_spatial_shape, 
                status_encoding, 
                intrinsics, 
                extrinsics, 
                proposals, 
                traj_feature_bev) -> Dict[str, torch.Tensor]:
        """Torch module forward pass."""
        result = self.forward_train(
            agents, ego_query, agents_query, bev_feature, 
            bev_spatial_shape, status_encoding, 
            intrinsics, extrinsics, proposals, traj_feature_bev)
        return result

    def forward_train(self, 
                      agents, 
                      ego_query, 
                      agents_query, 
                      bev_feature, 
                      bev_spatial_shape, 
                      status_encoding,  
                      intrinsics, 
                      extrinsics,
                      proposals, 
                      traj_feature_bev) -> Dict[str, torch.Tensor]:
        
        # bs = ego_query.shape[0]
        # plan_anchor = self.plan_anchor.unsqueeze(0).repeat(bs,1,1,1)  # torch.Size([B, 20, 8, 2])
        # ego_fut_mode = plan_anchor.shape[1]  # 20
        # traj_pos_embed = gen_sineembed_for_position(
        #     plan_anchor,hidden_dim=64)  # torch.Size([B, 20, 8, 64])
        # traj_pos_embed = traj_pos_embed.flatten(-2)  # torch.Size([B, 20, 512])
        # traj_feature = self.plan_anchor_encoder(traj_pos_embed)  # torch.Size([B, 20, 256])
        # traj_feature = traj_feature.view(bs,ego_fut_mode,-1)  # torch.Size([B, 20, 256])

        plan_anchor = proposals[..., :2]  # torch.Size([B, 20, 8, 2])
        traj_feature =traj_feature_bev.reshape(proposals.size(0), proposals.size(1), proposals.size(2), -1).amax(-2)  # torch.Size([B, 20, 256])
        
        poses_reg_list, poses_cls_list = self.diff_decoder(
            agents, traj_feature, plan_anchor, bev_feature, 
            bev_spatial_shape, agents_query, ego_query, status_encoding, 
            intrinsics, extrinsics)
        
        return_dict = {
            "poses_reg_list": poses_reg_list,
            # "poses_cls_list":poses_cls_list,
            # "plan_anchor": plan_anchor,
        }

        return return_dict
