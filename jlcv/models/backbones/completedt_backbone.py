import torch
import torch.nn as nn
import copy
import torch.nn.functional as F

from jlcv.modules import build_conv_layer, build_linear_layer, build_sample_and_group_layer, build_transformer_layer
from jlcv.modules.ext import group_points, knn, ball_query, furthest_point_sample, gather_points

from jlcv.models import MODELS

@MODELS.register_module()
class CompleteDTBackbone(nn.Module):
    def __init__(self, 
                 channels, 
                 num_points_list,
                 num_sample_list,
                 conv_cfg='Conv1d', 
                 norm_cfg='BN', 
                 act_cfg='GELU'):
        super().__init__()

        self.num_points_list = num_points_list
        self.num_sample_list = num_sample_list
        self.num_pc = len(num_points_list)

        self.conv1 = build_conv_layer(conv_cfg, 3, channels[0], 1, norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.conv2 = build_conv_layer(conv_cfg, channels[0], channels[0], 1, norm_cfg=norm_cfg, act_cfg=act_cfg)

        for i in range(self.num_pc - 1):
            proj_layer = build_conv_layer(conv_cfg, channels[0], channels[i], 1, norm_cfg=norm_cfg, act_cfg=act_cfg)
            self.__setattr__(f'proj_layer_{i}', proj_layer)

            grouped_layer = build_conv_layer('Conv2d', channels[i]*2+6, channels[i+1], 1, norm_cfg=norm_cfg, act_cfg=act_cfg)
            self.__setattr__(f'grouped_layer_{i}', grouped_layer)

            res_layer = nn.Sequential(
                build_conv_layer('Conv2d', channels[i+1], channels[i+1], 1, norm_cfg=norm_cfg, act_cfg=act_cfg),
                build_conv_layer('Conv2d', channels[i+1], channels[i+1], 1, norm_cfg=norm_cfg)
            )
            self.__setattr__(f'res_layer_{i}', res_layer)

    def forward(self, points):
        feats = self.conv1(points)
        feats = self.conv2(points) + feats

        # fps
        points_list = [points]
        feats_list = [feats]
        for n in self.num_points_list:
            index = furthest_point_sample(points.transpose(1, 2), n)
            new_points = gather_points(points, index)
            new_feats = gather_points(feats, index)
            points_list.append(new_points)
            feats_list.append(new_feats)
        
        grouped_index_list = []
        for i in range(self.num_pc - 1):
            num_sample = self.num_sample_list[i]

            grouped_index = knn(num_sample, points_list[i], points_list[i+1]).transpose(1, 2)
            grouped_index_list.append(grouped_index)

            grouped_points = group_points(points_list[i], grouped_index)
            grouped_feats = group_points(feats_list[i], grouped_index)

            feat_proj = self.__getattr__(f'proj_layer_{i}')(feats_list[i+1])
            feat_proj = feat_proj[:,:,:,None].repeat(1, 1, 1, num_sample)

            grouped_feats = torch.cat([grouped_feats, 
                                       feat_proj,
                                       (grouped_points - points_list[i+1][:,:,:,None]).transpose(1, 2),
                                       points_list[i+1][:,:,:,None].repeat(1, 1, 1, num_sample).transpose(1, 2)], 1)
            
            grouped_feats = self.__getattr__(f'grouped_layer_{i}')(grouped_feats)
            grouped_feats = self.__getattr__(f'res_layer_{i}')(grouped_feats) + grouped_feats
            points_list[i+1] = torch.max(grouped_feats, -1)[0]
    
        return points_list, feats_list, grouped_index_list