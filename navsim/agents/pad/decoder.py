import torch
import torch.nn as nn
import numpy as np
from torchvision.models.resnet import resnet18
from typing import Dict
import torch.nn.functional as F
from navsim.agents.pad.myself_head_autoregressive import TrajectoryHead_WO_Diff_Autoregressive
from navsim.common.enums import StateSE2Index
from navsim.agents.transfuser.transfuser_features import BoundingBox2DIndex


class UpsamplingAdd(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=2):
        super().__init__()
        self.upsample_layer = nn.Sequential(
            nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=False),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x, x_skip):
        x = self.upsample_layer(x)
        return x + x_skip
    

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


class Decoder(nn.Module):
    def __init__(self, in_channels, n_classes, config):
        super().__init__()
        self.cfg = config

        self.n_classes = n_classes
        backbone = resnet18(pretrained=False, zero_init_residual=True)

        self.first_conv = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = backbone.bn1
        self.relu = backbone.relu

        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3

        shared_out_channels = in_channels
        self.up3_skip = UpsamplingAdd(256, 128, scale_factor=2)
        self.up2_skip = UpsamplingAdd(128, 64, scale_factor=2)
        self.up1_skip = UpsamplingAdd(64, shared_out_channels, scale_factor=2)

        self.segmentation_head = nn.Sequential(
            nn.Conv2d(shared_out_channels, shared_out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(shared_out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(shared_out_channels, self.n_classes, kernel_size=1, padding=0),
        )

        self._keyval_embedding = nn.Embedding(self.cfg.bev_downscale_size**2 + 1, config.tf_d_model) 
        tf_decoder_layer = nn.TransformerDecoderLayer(
            d_model=config.tf_d_model,
            nhead=config.tf_num_head,
            dim_feedforward=config.tf_d_ffn,
            dropout=config.tf_dropout,
            batch_first=True,
        )
        self._tf_decoder = nn.TransformerDecoder(tf_decoder_layer, config.tf_num_layers)
        self._bev_downscale = nn.Sequential(
            nn.AdaptiveAvgPool2d((self.cfg.bev_downscale_size, self.cfg.bev_downscale_size)),
            nn.Conv2d(256, config.tf_d_model, kernel_size=1)
        )
        self.bev_proj = nn.Sequential(
            nn.AdaptiveAvgPool2d((64, 64)),
            nn.Conv2d(64, 64, kernel_size=1)
        )
        self.linear_proj = nn.Sequential(
            *linear_relu_ln(256, 1, 1, 320),
        )
        self._query_splits = [
            1,
            config.num_bounding_boxes,
        ]
        self._query_embedding = nn.Embedding(sum(self._query_splits), config.tf_d_model)
        self._status_encoding = nn.Linear(4 + 2 + 2, config.tf_d_model)
        self._agent_head = AgentHead(
            num_agents=config.num_bounding_boxes,
            d_ffn=config.tf_d_ffn,
            d_model=config.tf_d_model,
        )

        self._trajectory_head = TrajectoryHead_WO_Diff_Autoregressive(
            num_poses=config.trajectory_sampling.num_poses,
            d_ffn=config.tf_d_ffn,
            d_model=config.tf_d_model,
            plan_anchor_path=config.plan_anchor_path,
            config=config,
        )
            
    def forward(self, x, status_feature, intrinsics, extrinsics, proposals, traj_feature_bev):
        b, s, c, h, w = x.shape  # torch.Size([4, 1, 64, 128, 256])
        x = x.view(b * s, c, h, w)
        
        # (H, W)
        skip_x = {'1': x}

        # (H/2, W/2)
        x = self.first_conv(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.layer1(x)
        skip_x['2'] = x

        # (H/4 , W/4)
        x = self.layer2(x)
        skip_x['3'] = x

        # (H/8, W/8)
        x = self.layer3(x)  
        skip_x['4'] = x

        # First upsample to (H/4, W/4)
        x = self.up3_skip(x, skip_x['3'])

        # Second upsample to (H/2, W/2)
        x = self.up2_skip(x, skip_x['2'])

        # Third upsample to (H, W)
        x = self.up1_skip(x, skip_x['1']) 

        batch_size = status_feature.shape[0]
        bev_semantic_map = self.segmentation_head(x)
        output: Dict[str, torch.Tensor] = {"bev_semantic_map": bev_semantic_map}  # torch.Size([4, 7, 128, 256])

        bev_feature = self._bev_downscale(skip_x['4']).flatten(-2, -1) 
        bev_feature = bev_feature.permute(0, 2, 1)
        status_encoding = self._status_encoding(status_feature) 

        keyval = torch.concatenate([bev_feature, status_encoding[:, None]], dim=1) 
        keyval += self._keyval_embedding.weight[None, ...]

        bev_spatial_shape = torch.Size([64, 64])
        
        concat_cross_bev = keyval[:,:-1].permute(0,2,1).contiguous().view(batch_size, -1, 8, 8)
        concat_cross_bev = F.interpolate(concat_cross_bev, size=bev_spatial_shape, mode='bilinear', align_corners=False)

        cross_bev_feature = self.bev_proj(x)
        cross_bev_feature = torch.cat([concat_cross_bev, cross_bev_feature], dim=1)
        cross_bev_feature = self.linear_proj(cross_bev_feature.flatten(-2,-1).permute(0,2,1))
        cross_bev_feature = cross_bev_feature.permute(0,2,1).contiguous().view(batch_size, -1, bev_spatial_shape[0], bev_spatial_shape[1])
        
        query = self._query_embedding.weight[None, ...].repeat(batch_size, 1, 1)
        query_out = self._tf_decoder(query, keyval)

        trajectory_query, agents_query = query_out.split(self._query_splits, dim=1)

        agents = self._agent_head(agents_query)
        output.update(agents)

        trajectory = self._trajectory_head(agents, trajectory_query, agents_query, 
                                           cross_bev_feature, bev_spatial_shape,
                                           status_encoding[:, None],
                                           intrinsics, extrinsics, proposals, traj_feature_bev)
        
        output.update(trajectory)

        return output


class AgentHead(nn.Module):
    """Bounding box prediction head."""

    def __init__(
        self,
        num_agents: int,
        d_ffn: int,
        d_model: int,
    ):
        """
        Initializes prediction head.
        :param num_agents: maximum number of agents to predict
        :param d_ffn: dimensionality of feed-forward network
        :param d_model: input dimensionality
        """
        super(AgentHead, self).__init__()

        self._num_objects = num_agents
        self._d_model = d_model
        self._d_ffn = d_ffn

        self._mlp_states = nn.Sequential(
            nn.Linear(self._d_model, self._d_ffn),
            nn.ReLU(),
            nn.Linear(self._d_ffn, BoundingBox2DIndex.size()),
        )

        self._mlp_label = nn.Sequential(
            nn.Linear(self._d_model, 1),
        )

    def forward(self, agent_queries) -> Dict[str, torch.Tensor]:
        """Torch module forward pass."""

        agent_states = self._mlp_states(agent_queries)
        agent_states[..., BoundingBox2DIndex.POINT()] = agent_states[..., BoundingBox2DIndex.POINT()].tanh() * 32
        agent_states[..., BoundingBox2DIndex.HEADING] = agent_states[..., BoundingBox2DIndex.HEADING].tanh() * np.pi

        agent_labels = self._mlp_label(agent_queries).squeeze(dim=-1)

        return {"agent_states": agent_states, "agent_labels": agent_labels}
