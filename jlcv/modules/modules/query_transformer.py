import torch
import torch.nn as nn
from ..ext import group_points
from ..base import build_conv_layer, build_linear_layer
from .transformer_decoder import TransformerDecoder
from . import MODULES

@MODULES.register_module()
class QureyTransformer(nn.Module):
    def __init__(self, 
                 channels,
                 query_attention,
                 feats_attention,
                 feed_forward_channels=None,
                 query_position_embedding=None,
                 feats_position_embedding=None,
                 conv='Conv1d',
                 norm='BN',
                 act='ReLU') -> None:
        super().__init__() 

        self.identity = build_conv_layer(channels, channels, 1, conv=conv, norm=norm, act=act)
        self.in_mlp = build_conv_layer(channels, channels, 1, conv=conv, norm=norm, act=act)
        self.query_attention = TransformerDecoder(channels, feed_forward_channels, query_attention, query_position_embedding)

        self.feats_attention = TransformerDecoder(channels, feed_forward_channels, feats_attention, feats_position_embedding)
        self.query_mlp = build_conv_layer(channels*2, channels, 1, conv=conv, norm=norm, act=act)
        self.global_mlp = build_conv_layer(channels*2, channels, 1, conv=conv, norm=norm, act=act)

    def forward(self, query_feats, feats, grouped_index_list):
        identity = self.identity(query_feats)
        batch_size, c, num_queries = query_feats.shape
        
        ################ query feats with local ################
        query_feats = self.in_mlp(query_feats)
        grouped_feats_list = []
        for grouped_index in grouped_index_list:
            grouped_feats = group_points(feats, grouped_index)
            grouped_feats_list.append(grouped_feats)
        query_feats_local = self.query_attention(query_feats, grouped_feats_list)

        ################ feats ################
        feats = self.feats_attention(feats, query_feats_local)

        ################ query feats with global ################
        query_feats_global = torch.cat([
            query_feats_local, 
            feats.max(-1)[0][..., None].repeat(1, 1, num_queries)
        ], 1)
        query_feats_global = self.global_mlp(query_feats_global)

        ################ aggregating query feats with local and global ################
        query_feats = torch.cat([query_feats_local, query_feats_global], 1)
        query_feats = self.query_mlp(query_feats)
        ################ query feats ################
        query_feats = query_feats + identity
        return query_feats, feats














        
        
     