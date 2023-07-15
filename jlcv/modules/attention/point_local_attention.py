from .attention import CrossAttention
import torch.nn as nn
from . import ATTENTION

@ATTENTION.register_module()
class PointLocalAttention(CrossAttention):
    def __init__(self, 
                 channels, 
                 num_heads=8, 
                 qkv_bias=False, 
                 attn_drop=0, 
                 proj_drop=0, 
                 local_conv=None, 
                 conv_cfg='Conv1d'):
        super().__init__(channels, num_heads, qkv_bias, attn_drop, proj_drop, local_conv, conv_cfg)

   
    def forward(self, x, y):
        B, C, N = x.shape
        
        _, _, _, K = y.shape

        q = self.q(x.transpose(1,2)).reshape(B, N, 1, self.num_heads, self.head_channels).permute(0, 3, 1, 2, 4) # b, h, n, 1, c

        kv = self.kv(y.permute(0, 2, 3, 1)).reshape(B, -1, K, self.num_heads, self.head_channels, 2).permute(0, 3, 1, 2, 4, 5) # b, h, n, k, c, 2
        k, v = kv[..., 0], kv[..., 1] # b, h, n, k, c

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        if self.local_v is not None:
            local_v = v.permute(0, 1, 4, 2, 3).reshape(B, C, -1)
            local_v = self.local_v(local_v)
            local_v = local_v.reshape(B, self.num_heads, self.head_channels, -1, K)
            v = v + local_v.permute(0, 1, 3, 4, 2)

        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        out = self.proj(out).transpose(1,2) + x
        return out

            
    





    
