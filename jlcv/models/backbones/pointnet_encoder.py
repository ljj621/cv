import torch
import torch.nn as nn
from jlcv.modules.base import build_conv_layer
from jlcv.models import MODELS

@MODELS.register_module()
class PointNetEncoder(nn.Module):
    def __init__(self, 
                 channels=[3, 64, 128, 1024], 
                 conv='Conv1d', 
                 norm='BN', 
                 act='ReLU'):
        super().__init__()
        num_layers = len(channels) - 1
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            if i == num_layers - 1:
                act = None
            conv = build_conv_layer(channels[i], channels[i+1], 1, conv=conv,norm=norm, act=act)
            self.layers.append(conv)

    def forward(self, x):
        B, C, N = x.shape
        feats = []
        for layer in self.layers:
            x = layer(x)
            feats.append(x)
        global_feats = torch.max(x, 2)[0]
        return feats, global_feats
