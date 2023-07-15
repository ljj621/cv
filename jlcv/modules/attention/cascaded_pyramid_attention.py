import copy
import torch
import torch.nn as nn
from ..base import build_conv_layer, build_linear_layer
from . import ATTENTION

@ATTENTION.register_module()
class CascadedPyramidAttention(nn.Module):
    def __init__(self, 
                 channels,
                 kv_channels,
                 num_heads=None,
                 qkv_bias=False,
                 attn_drop=0, 
                 proj_drop=0,
                 local_conv=None, 
                 conv_cfg='Conv1d',
                 norm='LN',
                 act='ReLU'
        ):
        super().__init__()
        self.kv_channels = kv_channels
        self.num_heads = num_heads
        self.head_channels = channels//num_heads
        self.scales = self.head_channels**(-0.5)
        self.q = build_linear_layer(channels, channels, bias = qkv_bias)

        self.local_conv = local_conv
        self.kv_head = []
        for i, c in enumerate(kv_channels):
            self.__setattr__(f'kv_{i}', build_linear_layer(c, 2*c, bias=qkv_bias))
            self.__setattr__(f'mlp_{i}', build_linear_layer(c, c, norm=norm, act=act))
            if local_conv is not None:
                local_v = build_conv_layer(c, c, 1, groups=c, conv=local_conv)
                self.__setattr__(f'local_v_{i}', local_v)
                
            self.__setattr__(f'attn_drop_{i}', nn.Dropout(attn_drop))
            self.kv_head.append(c//self.head_channels)

        self.proj = nn.Sequential(build_linear_layer(channels, channels),nn.Dropout(proj_drop))
        

    def forward(self, x, y_list):
        '''
        Args:
            x: [B, C N]
            y: [[B, C N1], [B, C N2],...]
            
        return:
            out: [B, C, N]
        '''
        x = x.transpose(1, 2)
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, self.head_channels).permute(0, 2, 1, 3) # b, h, n, c
        q_list = torch.split(q, self.kv_head, 1)
        
        outs = []
        for i, num_head in enumerate(self.kv_head):
            q = q_list[i] # b, h, n, c
            y = y_list[i].transpose(1,2)
            kv = self.__getattr__(f'kv_{i}')(y)
            kv = kv.reshape(B, y.shape[1], num_head, self.head_channels, 2)
            kv = kv.permute(4, 0, 2, 1, 3)
            k, v = kv[0], kv[1] # b, h, n, c

            attn = (q @ k.transpose(-2, -1)) * self.scales
            attn = self.__getattr__(f'attn_drop_{i}')(attn.softmax(dim=-1))
            
            if hasattr(self, f'local_v_{i}'):
                b, h, n, c = v.shape
                local_v = self.__getattr__(f'local_v_{i}')(v.transpose(1,2).reshape(b, h*c, n))
                local_v = local_v.reshape(b, h, c, n) # b, h, c, n
                v = v + local_v.transpose(2, 3)
            
            out = (attn @ v).transpose(1, 2).reshape(B, N, -1)
            out = self.__getattr__(f'mlp_{i}')(out)
            outs.append(out)

        outs = torch.cat(outs, dim=-1)
        out = self.proj(outs) + x
        return out.transpose(1,2)
               



        
        



