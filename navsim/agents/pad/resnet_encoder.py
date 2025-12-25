import os.path
import torch.nn as nn
import timm
import torch
import numpy as np
from navsim.agents.pad.convolutions import UpsamplingConcat, DeepLabHead


class Encoder_Resnet34(nn.Module):
    def __init__(self, cfg, D):
        super().__init__()
        self.cfg = cfg
        self.D = D
        self.C = cfg.MODEL_ENCODER_OUT_CHANNELS
        self.downsample = cfg.MODEL_ENCODER_DOWNSAMPLE

        # self.backbone = timm.create_model(cfg.MODEL_ENCODER_NAME, pretrained=False, features_only=True)
        # checkpoint = torch.load(os.path.join(os.getenv('PYTHONPATH'), "ckpts/resnet34-333f7ec4.pth"))
        # if "state_dict" in checkpoint:
        #     state_dict = checkpoint["state_dict"]
        # else:
        #     state_dict = checkpoint
        # state_dict = {k: v for k, v in state_dict.items() if not k.startswith("fc.")}
        # self.backbone.load_state_dict(state_dict, strict=False)
        # self.delete_unused_layers()

        self.reduction_channel = [0, 64, 64, 128, 256]

        index = np.log2(self.downsample).astype(np.int32)
        self.feature_layer_1 = DeepLabHead(self.reduction_channel[index+1], self.reduction_channel[index+1], hidden_channel=64)
        self.feature_layer_2 = UpsamplingConcat(self.reduction_channel[index+1] + self.reduction_channel[index], self.C)
    
    def delete_unused_layers(self):
        del self.backbone.layer4
    
    def get_features_depth(self, x, img_feats_resnet):
        # x = self.backbone(x)

        x_3 = img_feats_resnet[3]
        N_views = 4
        x_3 = x_3.view(-1, N_views, *x_3.shape[1:])[:, 1:].reshape(-1, *x_3.shape[1:])
        feature = self.feature_layer_1(x_3)  # torch.Size([12, 256, 27, 48])
        feature_for_attn = feature
        x_2 = img_feats_resnet[2]
        N_views = 4
        x_2 = x_2.view(-1, N_views, *x_2.shape[1:])[:, 1:].reshape(-1, *x_2.shape[1:])
        feature = self.feature_layer_2(feature, x_2)  # torch.Size([12, 64, 54, 96])

        return feature, None, feature_for_attn

    def forward(self, x, img_feats_resnet):
        feature, depth, feature_for_attn = self.get_features_depth(x, img_feats_resnet)  # get feature vector
        # if self.use_depth_distribution:
        #     depth_prob = depth.softmax(dim=1)
        #     feature = depth_prob.unsqueeze(1) * feature.unsqueeze(2)  # outer product depth and features
        # else:
        #     feature = feature.unsqueeze(2).repeat(1, 1, self.D, 1, 1)

        return feature, depth, feature_for_attn
