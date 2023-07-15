import torch
import torch.nn as nn
import torch.nn.functional as F
from jlcv.models import MODELS
from jlcv.modules.modules import MODULES
from jlcv.modules.base import build_conv_layer, build_linear_layer
from jlcv.modules.ext import knn, three_nn, three_interpolate

@MODELS.register_module()
class QuerySegHead(nn.Module):
    def __init__(self,
                 channels,
                 QFP=None,
                 num_stage=None,
                 query_classifier=None,
                 points_classifier=None,
                 classifier=None,
                 conv='Conv1d',
                 norm='BN', 
                 act='GELU',
                 drop=0.4
                 ):
        super().__init__()
        self.QFP = MODULES.build(QFP, channels=channels)
        self.num_stage = num_stage
        self.norm = norm
        self.act = act
        self.drop = drop
        
        self.mlp = nn.ModuleList()
        for i in range(self.num_stage):
            self.mlp.append(nn.Sequential(
                build_conv_layer(channels//num_stage, channels, 1, conv=conv, norm=norm, act=act),
                build_conv_layer(channels, channels, 1, conv=conv, norm=norm, act=act),
            ))
        
        if query_classifier is not None:
            self.query_classifier = self.build_classifier(channels, **query_classifier)
        
        if points_classifier is not None:
            self.points_classifier = self.build_classifier(channels, **points_classifier)
            
        self.classifier = self.build_classifier(channels, **classifier)

    def build_classifier(self, in_channels, channels, layer_type):
        assert len(channels) == len(layer_type)
        num_layer = len(channels)
        layers = []
        for i in range(num_layer):
            if layer_type[i] == 'LBAD':
                layers.append(build_linear_layer(in_channels, channels[i], norm=self.norm, act=self.act, drop=self.drop))
            elif layer_type[i] == 'LBA':
                layers.append(build_linear_layer(in_channels, channels[i], norm=self.norm, act=self.act))
            elif layer_type[i] == 'L':
                layers.append(build_linear_layer(in_channels, channels[i]))
            in_channels = channels[i]
        return nn.Sequential(*layers)
    
    def forward(self, input_dict):
        original_points = input_dict['original_points']
        points = input_dict['points']
        points_index = input_dict['points_index']
        query = input_dict['query'] # b, 3, m
        query_feats = input_dict['query_feats']
        # query_feats_list = input_dict['query_feats_list']
        points_feats = input_dict['points_feats']
        points_feats_list = input_dict['points_feats_list']
        
        _, _, num_query = query_feats.shape
        _, _, num_feats = points_feats.shape
        batch_size, _, num_points = original_points.shape
        preds_dict = {}
        
        if self.query_classifier is not None:
            query_index = knn(1, original_points, query)[..., 0]
            query_feats_flatten = query_feats.transpose(1,2).reshape(batch_size*num_query, -1)
            query_preds = self.query_classifier(query_feats_flatten).reshape(batch_size, num_query, -1)
            
            preds_dict['query'] = {
                'preds': query_preds,
                'index': query_index
            }
        
        points_feats = self.QFP(points_feats, query_feats)
        if self.classifier is not None:
            feats_flatten = points_feats.transpose(1,2).reshape(batch_size*num_feats, -1)
            points_preds = self.points_classifier(feats_flatten).reshape(batch_size, num_feats, -1)
            preds_dict['points'] = {
                'preds': points_preds,
                'index': points_index
            }
        
        dist, interpolated_index = three_nn(original_points[:, :3, :].transpose(1,2), points.transpose(1,2))
        dist_recip = 1.0 / (dist + 1e-8)
        norm = torch.sum(dist_recip, dim=2, keepdim=True)
        interpolated_weight = dist_recip / norm    
        interpolated_feats = three_interpolate(points_feats, interpolated_index, interpolated_weight) 
        
        interpolated_feats_list = []
        for i in range(self.num_stage-1, -1, -1):
            tmp_interpolated_feats = three_interpolate(points_feats_list[i], interpolated_index, interpolated_weight) 
            interpolated_feats = self.mlp[i](tmp_interpolated_feats) + interpolated_feats
            interpolated_feats_list.append(interpolated_feats)
        
        interpolated_feats_flatten = interpolated_feats.transpose(1,2).reshape(batch_size*num_points, -1)
        preds = self.classifier(interpolated_feats_flatten).reshape(batch_size, num_points, -1)
        preds_dict['preds'] = {
                'preds': preds,
                'preds_logits': F.log_softmax(preds, -1)
            }
        return preds_dict
            
        
        