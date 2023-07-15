import torch
import torch.nn as nn
from .attention import SelfAttention
from ..base import build_linear_layer
from . import ATTENTION

@ATTENTION.register_module()
class DenseScaledAttention(nn.Module):
    def __init__(self, 
                 channels, 
                 num_heads=8, 
                 qkv_bias=False, 
                 attn_drop=0, 
                 proj_drop=0, 
                 local_v=None, 
                 scale_ratio=None):
        super().__init__()
        self.num_layers = len(scale_ratio)
        for i, sr in enumerate(scale_ratio):
            layer = SelfAttention(
                [channels + channels*i, channels*sr], 
                num_heads, 
                qkv_bias=qkv_bias, 
                attn_drop=attn_drop, 
                proj_drop=proj_drop, 
                local_v=local_v)
            self.add_module(f'layer_{i}', module=layer)
        
        self.feed_forward = build_linear_layer(channels*(i+2), channels)
    def forward(self, x):
        out = [x]
        for i in range(self.num_layers):
            feats = self.__getattr__(f'layer_{i}')(torch.cat(out, -1))
            out.append(feats)

        out = torch.cat(out, -1)
        out = self.feed_forward(out) + x
        return out

        
        





    
