import torch
import torch.nn as nn
import torch.nn.functional as F
from jlcv.modules import MODULES, Conv
from jlcv.modules.ext import spconv
from jlcv.models import MODELS
from jlcv.modules.modules import SparseEncoder

@MODELS.register_module()
class SECOND(nn.Module):
    def __init__(self,
                 data_preprocessor,
                 sparse_encoder,
                 out_channels = [128, 128, 256],
                 num_layers=[3, 5, 5],
                 layer_strides=[2, 2, 2],
                 conv= dict(type='Conv2d', bias=False),
                 norm=dict(type='BN', eps=1e-3, momentum=0.01),
                 act='ReLU'
        ):
        super().__init__()
        self.data_preprocessor = MODELS.build(data_preprocessor)
        self.sparse_encoder = SparseEncoder(**sparse_encoder)

        in_channels = sparse_encoder['out_channels']
        in_channels = [in_channels, *out_channels[:-1]]

        blocks = []
        for i, layer_num in enumerate(num_layers):
            block = [Conv(in_channels[i], out_channels[i], 3, 
                          stride=layer_strides[i], padding=1, 
                          conv=conv, norm=norm, act=act)]
            for j in range(layer_num):
                block.append(Conv(out_channels[i], out_channels[i], 3, padding=1, 
                                  conv=conv, norm=norm, act=act))

            block = nn.Sequential(*block)
            blocks.append(block)

        self.blocks = nn.ModuleList(blocks)
    
    def forward(self, points):
        sparse_points = self.data_preprocessor(points)

        sparse_feats, sparse_feats_list = self.sparse_encoder(sparse_points)

        dense_features = sparse_feats.dense()
        N, C, D, H, W = dense_features.shape
        dense_features = dense_features.view(N, C * D, H, W)

        outs = []
        for i in range(len(self.blocks)):
            x = self.blocks[i](x)
            outs.append(x)
        return outs





