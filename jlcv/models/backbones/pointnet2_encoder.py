import torch
import torch.nn as nn
from jlcv.modules.base import build_conv_layer
from jlcv.modules.ext import furthest_point_sample, gather_points, group_points, knn, ball_points
from jlcv.models import MODELS

@MODELS.register_module()
class PointNet2Encoder(nn.Module):
    def __init__(self, 
                 channels=None, 
                 num_points=None,
                 num_sample = None,
                 radius = None,
                 conv='Conv2d',
                 norm='BN',
                 act='ReLU'):
        super().__init__()
        if radius is None:
            radius = [radius] * len(channels)

        assert len(channels) == len(num_points) == len(num_sample) == len(radius)
        self.num_points = num_points
        self.num_sample = num_sample
        self.radius = radius
        self.num_layers = len(channels)

        self.grouped_build_conv_layer = nn.ModuleList()
        for i, channel in enumerate(channels):
            num_layers = len(channel) - 1
            grouped_build_conv_layer = []
            for n in range(num_layers):
                if n == 0:
                    grouped_build_conv_layer.append(build_conv_layer(channel[n]+3, channel[n+1], 1, conv=conv, norm=norm, act=act))
                else:
                    grouped_build_conv_layer.append(build_conv_layer(channel[n], channel[n+1], 1, conv=conv, norm=norm, act=act))
            self.grouped_build_conv_layer.append(nn.Sequential(*grouped_build_conv_layer))

        
    def forward(self, points, feats=None):
        B, _, N = points.shape
        if feats is None:
            feats = points
        feats_list = []
        points_list = []
        grouped_index_list = []

        for i, grouped_build_conv_layer in enumerate(self.grouped_build_conv_layer):
            B, C, N = points.shape
            device = points.device
            num_points = self.num_points[i]
            num_sample = self.num_sample[i]
            radius = self.radius[i]

            if num_points < N:
                fps_index = furthest_point_sample(points.transpose(1, 2), num_points) # [B, npoint, C]
                proposal_points = gather_points(points, fps_index) # [B, C, nsample]
            else:
                proposal_points = points

            if radius is not None:
                grouped_index = ball_points(radius, num_sample, points, proposal_points) # [B, npoint, nsample]
            else:
                grouped_index = knn(num_sample, points, proposal_points).transpose(1, 2)
        
            grouped_points = group_points(points, grouped_index) # [B, C, npoint, nsample]
            grouped_points_norm = grouped_points - proposal_points[..., None] # [B, C, npoint, nsample]

            grouped_feats = group_points(feats, grouped_index)
            grouped_feats_cat = torch.cat([grouped_points_norm, grouped_feats], dim=1) # [B, C, npoint, nsample]

            grouped_feats_cat = grouped_build_conv_layer(grouped_feats_cat)
            proposal_feats = grouped_feats_cat.max(-1)[0]

            feats_list.append(proposal_feats)
            points_list.append(proposal_points)
            grouped_index_list.append(grouped_index)

            points = proposal_points
            feats = proposal_feats

        return points_list, feats_list, grouped_index_list
  