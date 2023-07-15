import torch
import torch.nn as nn
from jlcv.modules.base import build_conv_layer
from jlcv.modules.ext import furthest_point_sample, gather_points, group_points, knn

from jlcv.models import MODELS

@MODELS.register_module()
class DGCNN(nn.Module):
    def __init__(self, 
                 in_channels,
                 out_channels,
                 channels,
                 num_points=None,
                 num_sample=32,
                 conv='Conv2d',
                 norm = 'BN',
                 act = {'type': 'LeakyReLU', 'negative_slope': 0.2}
        ):
        super().__init__()
        self.num_points = num_points
        self.num_sample = num_sample
        self.edge_conv = nn.ModuleList()
        for channel in channels:
            self.edge_conv.append(
                build_conv_layer(in_channels*2, channel, 1, bias=False, norm=norm, conv=conv, act=act)
            )
            in_channels = channel
        
        self.mlp_out = build_conv_layer(sum(channels), out_channels, 1, bias=False, conv='Conv1d', norm=norm, act=act)

    def forward(self, feats, points):
        '''
        points: b, 3, n
        feats: b, c, n
        '''
        # if self.num_points is not None:
        #     fps_index = furthest_point_sample(points, 1024)
        #     new_points = gather_points(points, fps_index) # b, c, n
        # else:
        #     new_points = points
        grouped_index = knn(self.num_sample, points, points) # b, n, k

        feats_list = []

        for conv in self.edge_conv:
            grouped_feats = group_points(feats, grouped_index) # b, c, n, s
            feats = feats[..., None].repeat(1, 1,1, self.num_sample)
            grouped_feats = torch.cat([grouped_feats - feats, feats], dim=1)
            grouped_feats = conv(grouped_feats)     
            feats = grouped_feats.max(dim=-1)[0]          
            feats_list.append(feats)
        
        feature = torch.cat(feats_list, dim = 1)
        feature = self.mlp_out(feature)

        return feature