import copy
import torch
import torch.nn as nn
from ..base import build_conv_layer, build_linear_layer
from . import ATTENTION

@ATTENTION.register_module()
class PointLocalScaleAttention(nn.Module):
    def __init__(self, 
                 channels,
                 num_heads=None,
                 qkv_bias=False,
                 attn_drop=0, 
                 proj_drop=0,
                 local_build_conv_layer=None, 
                 conv='Conv2d',
                 kv_channels=None,
                 pos_channels=None):
        super().__init__()
        self.kv_channels = kv_channels
        self.num_heads = num_heads
        self.head_channels = channels//num_heads
        self.num_scale = len(kv_channels)
        self.scales = self.head_channels**(-0.5)

        self.q = build_linear_layer(channels, channels, bias = qkv_bias)
        self.kv = nn.ModuleList()
        self.kv_scale = nn.ModuleList()
        self.local_v = nn.ModuleList() if local_build_conv_layer is not None else None
        self.attn_drop = nn.ModuleList()

        for i, c in enumerate(kv_channels):
            self.kv_scale.append(build_conv_layer(c, channels//self.num_scale, 1, conv=conv, norm='BN', act='ReLU'))
            self.kv.append(build_linear_layer(channels//self.num_scale, 2*channels//self.num_scale, bias=qkv_bias))
            if self.local_v is not None:
                self.local_v.append(build_conv_layer(channels//self.num_scale,  channels//self.num_scale, 1, groups=channels//self.num_scale, conv=conv))
            
            self.attn_drop.append(nn.Dropout(attn_drop))

        self.proj = nn.Sequential(build_linear_layer(channels, channels),nn.Dropout(proj_drop))
   
    def forward(self, x, y,  x_pos=None, y_pos=None):
        assert isinstance(y, list)
        if x_pos is not None:
            if x.shape[-1] > x_pos.shape[-1]:
                x, _x = x[..., :x_pos.shape[-1]], x[..., x_pos.shape[-1]:]
            else:
                _x = []
            x = x + self.x_pos_embedding(x_pos)
            x = torch.cat([x, *_x], -1)
          
        x = x.transpose(1, 2)

        B, N, C = x.shape
        q = self.q(x).reshape(B, N, 1, self.num_heads, -1).permute(0, 3, 1, 2, 4) # b, h, n, c
        q = torch.chunk(q, self.num_scale, 1)
        # outs = 0
        outs = []

        for i, _y in enumerate(y):
            _, _, _, K = _y.shape
            _q = q[i] # b, h, n, 1, c
            if y_pos is not None:
                _y = _y + self.y_pos_embedding[i](y_pos[i])
            _y = self.kv_scale[i](_y) # b, c, n, k
            kv = self.kv[i](_y.permute(0, 2, 3, 1))
            kv = kv.reshape(B, -1, K, self.num_heads//self.num_scale, self.head_channels, 2)
            kv=kv.permute(0, 3, 1, 2, 4, 5) # b, h, n, k, c, 2
            k, v = kv[..., 0], kv[..., 1] # b, h, n, k, c

            attn = (_q @ k.transpose(-2, -1)) * self.scales
            attn = self.attn_drop[i](attn.softmax(dim=-1))

            if self.local_v is not None:
                local_v = v.permute(0, 1, 4, 2, 3).reshape(B, C, -1)
                local_v = self.local_v(local_v)
                local_v = local_v.reshape(B, self.num_heads//self.num_scale, self.head_channels, -1, K)
                v = v + local_v.permute(0, 1, 3, 4, 2)

            
            out = (attn @ v).transpose(1, 2).reshape(B, N, -1)
            outs.append(out)
        
        outs = torch.cat(outs, dim=-1)
        outs = self.proj(outs).transpose(1, 2)
        return outs


     
    
