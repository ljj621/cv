import copy
import torch
import torch.nn as nn
from . import MODULES
from ..attention import ATTENTION
from ..base import build_conv_layer, build_linear_layer, NORM, ACTIVATION
from ..ext import knn, group_points

@MODULES.register_module()
class TransformerEncoder(nn.Module):
    def __init__(self, 
                 attention,
                 channels,
                 feed_forward_channels=None,
                 position_embedding=None):
        super().__init__()
        self.channels=channels
        self.position_embedding = position_embedding
        if position_embedding is not None:
            self.position_embedding_module_on_local(position_embedding)
        
        self.attention = ATTENTION.build(attention)
        self.norm = NORM.build('LN', channels)

        self.feed_forward = nn.Sequential(
            build_linear_layer(channels, feed_forward_channels),
            ACTIVATION.build('ReLU'),
            build_linear_layer(channels, feed_forward_channels),
        )
        self.feed_forward_norm = NORM.build('LN', channels)

    def forward(self, x, x_pos=None):
        if self.position_embedding is not None:
            x = self.position_embedding_on_local(x, x_pos)
        
        x = self.attention(x) + x
        x = self.norm(x.transpose(1,2))

        x = self.feed_forward(x) + x
        x = self.feed_forward_norm(x).transpose(1,2)
        return x

    def position_embedding_module_on_local(self, position_embedding):
        pos_embedding = copy.deepcopy(position_embedding)
        self.pos_embedding = build_conv_layer(**pos_embedding, out_channels=self.channels)

        pos_embedding['in_channels'] = self.channels*2
        self.embedding1 = build_conv_layer(**pos_embedding, out_channels=self.channels)

        self.embedding2 = build_conv_layer(**pos_embedding, out_channels=self.channels)

    def position_embedding_on_local(self, x, x_pos):
        grouped_index = knn(self.num_sample, x_pos, x_pos)
        y_pos = group_points(x_pos, grouped_index)
        y = group_points(x, grouped_index)

        y_pos = self.pos_embedding(torch.cat([
            y_pos,
            x_pos[..., None] - y_pos], 1))

        y = self.embedding1(torch.cat([
            y, 
            x[..., None] - y], 1))

        x = self.embedding2(torch.cat([y, y_pos], 1)).max(-1)[0]
        return x


     
    
