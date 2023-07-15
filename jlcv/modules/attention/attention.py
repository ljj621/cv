import torch.nn as nn
from ..base import build_conv_layer, build_linear_layer
from . import ATTENTION

@ATTENTION.register_module()
class CrossAttention(nn.Module):
    def __init__(self,
                 channels,
                 num_heads=8,
                 qkv_bias=False,
                 attn_drop=0,
                 proj_drop=0,
                 local_conv=None,
                 conv='Conv1d'):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.head_channels = channels // num_heads
        self.scale = self.head_channels**-0.5

        self.q = build_linear_layer(channels, channels, bias=qkv_bias)
        self.kv = build_linear_layer(channels, channels * 2, bias=qkv_bias)
        if local_conv is not None:
            self.local_v = build_conv_layer(channels, channels, 1, groups=channels, conv =local_conv)
        else:
            self.local_v = None

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Sequential(build_linear_layer(channels, channels),
                                  nn.Dropout(proj_drop))

    def forward(self, x, y):
        '''
        Args:
            x: [B, C, N] 
            y: [B, C, N]
        return:
            out: [B, C, N]
        '''
        B, C, N = x.shape
        M = y.shape[-1]

        q = self.q(x.transpose(1, 2)).reshape(B, N, self.num_heads,
                                              self.head_channels).permute(
                                                  0, 2, 1, 3)  # b, h, n, c
        kv = self.kv(y.transpose(1, 2)).reshape(B, M, self.num_heads,
                                                self.head_channels,
                                                2).permute(0, 2, 1, 3, 4)
        k, v = kv[..., 0], kv[..., 1]  # b, h, n, c
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = self.attn_drop(attn.softmax(dim=-1))

        if self.local_v is not None:
            local_v = v.transpose(2, 3).reshape(B, C, -1)  # b, h, c, n
            local_v = self.local_v(local_v)
            local_v = local_v.reshape(B, self.num_heads, self.head_channels,
                                      -1).transpose(2, 3)  # b, h, n, c
            v = v + local_v

        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        out = self.proj(out).transpose(1, 2)
        return out


@ATTENTION.register_module()
class SelfAttention(CrossAttention):

    def __init__(self,
                 channels,
                 num_heads=8,
                 qkv_bias=False,
                 attn_drop=0,
                 proj_drop=0,
                 local_conv=None,
                 conv='Conv1d'):
        super().__init__(channels, num_heads, qkv_bias, attn_drop, proj_drop,
                         local_conv, conv)

    def forward(self, x):
        return super().forward(x, x)