import torch
import torch.nn as nn
import torch.nn.functional as F
from ..base import build_conv_layer
from ..ext import SparseSequential, SparseConvTensor
from . import MODULES

@MODULES.register_module()
class SparseEncoder(nn.Module):
    def __init__(self,
                 in_channels=4,
                 channels = [[16], 
                             [32, 32, 32], 
                             [64, 64, 64],
                             [64, 64, 64]],
                 padding = [[1], 
                            [1, 1, 1],
                            [1, 1, 1],
                            [1, 1, 1]],
                 out_channels = 128,
                 norm={'type': 'BN1d', 'momentum': 0.01},
                 act='GELU',
        ):
        super().__init__()
        self.norm = norm
        self.act = act
        self.conv_indentity = build_conv_layer(in_channels, out_channels, 3, padding=1, conv='SubMConv3d', norm=norm, act=act, indice_key='subm1')
        self.conv_input = build_conv_layer(in_channels, channels[0][0], 3, padding=1, conv='SubMConv3d', norm=norm, act=act, indice_key='subm1')

        in_channels = channels[0][0]
        self.conv_mlp = nn.ModuleList()
        self.num_stage = len(channels)
        for i, channel in enumerate(channels):
            layer = self.build_stage(in_channels, channel, padding[i], i+1)
            self.conv_mlp.append(SparseSequential(*layer))
            # self.conv_mlp.extend(layer)
            in_channels = channel[-1]
        # self.conv_mlp = SparseSequential(*conv_mlp)

        self.conv_out = build_conv_layer(in_channels, out_channels, 3, stride=[2, 1, 1], padding=1, conv='SparseConv3d', norm=norm, act=act, indice_key='spconv_down2')

    def build_stage(self, in_channels, channels, padding, indice_key):
        num_layer = len(channels)
        layers = []
        if num_layer == 1:
            layers.append(build_conv_layer(in_channels, channels[0], 3, padding=padding[0], conv='SubMConv3d', norm=self.norm, act=self.act, indice_key=f'subm{indice_key}'))
        else:
            for i, channel in enumerate(channels):
                if i == 0:
                    layers.append(build_conv_layer(in_channels, channel, 3, stride=2, padding=padding[i], conv='SparseConv3d', norm=self.norm, act=self.act, indice_key=f'spconv{indice_key}'))
                else:
                    layers.append(build_conv_layer(in_channels, channel, 3, padding=padding[i], conv='SubMConv3d', norm=self.norm, act=self.act, indice_key=f'subm{indice_key}'))
                in_channels = channel
        return layers

    def forward(self, input_dict):
        coors = input_dict['coors']
        voxels = input_dict['voxels']
        batch_size = input_dict['batch_size']
        sparse_shape = input_dict['sparse_shape']
        voxel_size = input_dict['voxel_size']
        pc_range = input_dict['point_cloud_range']

        sparse_points = SparseConvTensor(voxels, coors, sparse_shape, batch_size)

        
        sparse_feats = self.conv_input(sparse_points)
        # sparse_indentity = self.conv_indentity(sparse_points)

        for i, mlp in enumerate(self.conv_mlp):
            if i == 0:
                sparse_feats = mlp(sparse_feats)
            else:
                sparse_feats = self.encoder(sparse_feats, mlp)

        sparse_feats = self.conv_out(sparse_feats) 

        # output_dict = {}
        # output_dict['sparse_feats'] = sparse_feats
        # output_dict['voxel_size'] = voxel_size
        # output_dict['point_cloud_range'] = pc_range
        # output_dict['batch_size'] = batch_size

        return sparse_feats

    def encoder(self, sparse_feats, conv_mlp):
        for i, mlp in enumerate(conv_mlp):
            sparse_feats = mlp(sparse_feats)
            if i == 0:
                identity = sparse_feats
            
        sparse_feats.features = sparse_feats.features + identity.features
        return sparse_feats



    

