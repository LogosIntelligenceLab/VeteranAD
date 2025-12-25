import torch
import numpy as np
import torch.nn as nn
from mmdet.models.necks.fpn import FPN

from .grid_mask import GridMask
import timm
import torch.nn.functional as F


class ImgEncoder(nn.Module):
    def __init__(self, config,num_feature_levels=2):
        super().__init__()
        self.embed_dims = config.tf_d_model
        self.num_feature_levels = num_feature_levels
        num_cams = 4

        self.num_cams = num_cams
        self.use_cams_embeds = True
        _num_levels_ = 1
        _dim_ = self.embed_dims

        self.use_lidar=False

        self.grid_mask = GridMask( True, True, rotate=1, offset=False, ratio=0.5, mode=1, prob=0.7)
        self.use_grid_mask = True

        self.img_backbone = timm.create_model( 
            "resnet34", 
            pretrained=False, 
            features_only=True,
        )
        state_dict = torch.load("ckpts/resnet34-333f7ec4.pth", map_location="cpu")
        self.img_backbone.load_state_dict(state_dict, strict=False)

        self.with_img_neck=True

        self.num_outs=1

        self.img_neck=FPN(
            in_channels=[64,128,256,512][-self.num_outs:],
            out_channels=_dim_,
            start_level=0,
            add_extra_convs='on_output',
            num_outs=self.num_outs,
            relu_before_extra_convs=True
        )
        self.level_embeds = nn.Parameter(torch.randn( self.num_feature_levels, self.embed_dims))
        self.cams_embeds = nn.Parameter(
            torch.randn([self.num_cams, self.embed_dims]))

    def forward(self,img,len_queue=None,**kwargs):
        B = img.size(0)
        if img is not None:
            B, N, C, H, W = img.size()
            img = img.reshape(B * N, C, H, W)
            if self.use_grid_mask:
                img = self.grid_mask(img)

            img_feats = self.img_backbone(img)
            img_feats_resnet = img_feats
            if isinstance(img_feats, dict):
                img_feats = list(img_feats.values())
        else:
            return None
        if self.with_img_neck:
            img_feats = self.img_neck(img_feats[-self.num_outs:])

        img_feats_reshaped = []
        for img_feat in img_feats:
            BN, C, H, W = img_feat.size()
            if len_queue is not None:
                img_feats_reshaped.append(img_feat.view(int(B / len_queue), len_queue, int(BN / B), C, H, W))
            else:
                img_feats_reshaped.append(img_feat.view(B, int(BN / B), C, H, W))

        feat_flatten = []
        spatial_shapes = []
        for lvl, feat in enumerate(img_feats_reshaped):
            bs, num_cam, c, h, w = feat.shape
            spatial_shape = (h, w)
            feat = feat.flatten(3).permute(1, 0, 3, 2)
            if self.use_cams_embeds:
                feat = feat + self.cams_embeds[:, None, None, :].to(feat.dtype)
            feat = feat + self.level_embeds[None,
                                            None, lvl:lvl + 1, :].to(feat.dtype)

            spatial_shape = torch.as_tensor(
                [spatial_shape], dtype=torch.long, device=feat.device)
            level_start_index = torch.cat((spatial_shape.new_zeros(
                (1,)), spatial_shape.prod(1).cumsum(0)[:-1]))

            feat = feat.permute(0, 2, 1, 3)

            spatial_shapes.append(spatial_shape)
            feat_flatten.append(feat)

        return (feat_flatten[-1], spatial_shapes[-1], level_start_index, kwargs), img_feats_resnet
