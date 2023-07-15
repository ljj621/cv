import copy
import torch
import torch.nn as nn
from . import MODULES
from ..attention import ATTENTION
from ..base import NORM, ACTIVATION, build_conv_layer, build_linear_layer

@MODULES.register_module()
class TransformerDecoder(nn.Module):
    def __init__(self, 
                 channels,
                 feed_forward_channels=None,
                 attention=None,
                 position_embedding=None):
        super().__init__()
        self.channels=channels
        self.position_embedding = position_embedding
        if position_embedding is not None:
            self.position_embedding_module_on_local(position_embedding)

        self.kv_channels = attention.get('kv_channels', None)
        if 'channels' not in attention:
            attention['channels'] = channels
        self.attention = ATTENTION.build(attention)
        self.norm = NORM.build('LN', channels)

        self.feed_forward = nn.Sequential(
            build_linear_layer(channels, feed_forward_channels, act='ReLU'),
            build_linear_layer(feed_forward_channels, channels),
        )
        self.feed_forward_norm = NORM.build('LN', channels)

    def forward(self, x, y, x_pos=None, y_pos=None):
        if self.position_embedding is not None:
            x, y = self.position_embedding_on_local(x, y, x_pos, y_pos)
        
        x = self.attention(x, y) + x
        x = self.norm(x.transpose(1,2))

        x = self.feed_forward(x) + x
        x = self.feed_forward_norm(x).transpose(1,2)
        return x

    def position_embedding_module_on_local(self, position_embedding):
        # pos_embedding = copy.deepcopy(position_embedding)
        self.x_pos_embedding = build_conv_layer(position_embedding['channels'], self.channels, 1, conv='Conv2d')
        self.y_pos_embedding = build_conv_layer(position_embedding['channels'], self.channels, 1, conv='Conv2d')

        # pos_embedding['channels'] = self.channels*2
        self.y_embedding1 = build_conv_layer(self.channels*2, self.channels, 1, conv='Conv2d')
        self.y_embedding2 = build_conv_layer(self.channels*2, self.channels, 1, conv='Conv2d')

        self.x_embedding1 = build_conv_layer(self.channels*2, self.channels, 1, conv='Conv2d')
        self.x_embedding2 = build_conv_layer(self.channels*2, self.channels, 1, conv='Conv2d')
    
    def position_embedding_on_local(self, x, y, x_pos, y_pos):
        x_pos = self.x_pos_embedding(torch.cat([
            x_pos[..., None],
            x_pos[..., None] - y_pos], 1))
        x = self.x_embedding1(torch.cat([
            x[..., None], 
            x[..., None] - y], 1))
        x = self.x_embedding2(torch.cat([x, x_pos], 1))

        y_pos = self.y_pos_embedding(torch.cat([
            y_pos,
            x_pos[..., None] - y_pos], 1))
        y = self.y_embedding1(torch.cat([
            y, 
            x[..., None] - y], 1))
        y = self.y_embedding2(torch.cat([y, y_pos], 1))
        return x, y



    


     
    
