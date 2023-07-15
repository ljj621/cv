import torch
import torch.nn as nn
import torch.nn.functional as F
from ..base import build_conv_layer, build_linear_layer
from ..ext import SparseSequential, SparseConvTensor
from . import MODULES

@MODULES.register_module()
class PVEncoder(nn.Module):
    def __init__(self,
                 in_channels=4,
                 channels = [[32, 32, 32], 
                             [64, 64, 64],
                             [64, 64, 64]],
                 out_channels = 128,
                 norm='BN',
                 act='ReLU',) -> None:
        super().__init__()
        self.norm = norm
        self.act = act
        self.num_stage = len(channels)

        self.input_conv = build_conv_layer(in_channels*2, channels[0][0], 1, conv='Conv1d', norm=norm, act=act)

        in_channels = channels[0][0]
        self.sparse_layers = nn.ModuleList()
        for i in range(self.num_stage):
            sparse_layers = self.build_stage(in_channels, channels[i], i)
            self.sparse_layers.append(sparse_layers)
            in_channels = channels[i][-1]
        
        self.output_conv = build_conv_layer(
                    in_channels, out_channels, 3, stride=1, padding=1, 
                    bias=False, conv='SparseConv3d', norm=self.norm, act=self.act, indice_key=f'spconv_out'
                )

    
    def build_stage(self, in_channels, channels, indice_key):
        sparse_layers = []
        for i, channel in enumerate(channels):
            if i == 0:
                sparse_layers.append(build_conv_layer(
                    in_channels, channel, 3, stride=2, padding=1, 
                    bias=False, conv='SparseConv3d', norm=self.norm, act=self.act, indice_key=f'spconv{indice_key}'
                ))
            else:
                sparse_layers.append(build_conv_layer(
                    in_channels, channel, 3, stride=1, padding=1, 
                    bias=False, conv='SubMConv3d', norm=self.norm, act=self.act, indice_key=f'subm{indice_key}'
                ))
            in_channels = channels[i]
        sparse_layers = SparseSequential(*sparse_layers)
        return sparse_layers

    def forward(self, input_dict):
        points = input_dict['points']
        coors = input_dict['coors'].int()
        voxels = input_dict['voxels']
        num_voxels = input_dict['num_voxels']
        p2v_maps = input_dict['p2v_maps']
        batch_size = input_dict['batch_size']
        sparse_shape = input_dict['sparse_shape']
        voxel_size = input_dict['voxel_size'],
        pc_range = input_dict['pc_range']

        feats_list = []
        points_index_in_voxel_list = []
        points_in_voxel_list = []
        for i, n in enumerate(num_voxels):
            points_in_voxel, points_index_in_voxel = self.group_feats_in_voxel(points[i], p2v_maps[i], n, 20) # n, s, 3
            points_in_voxel_list.append(points_in_voxel)
        points_in_voxel = torch.cat(points_in_voxel_list, 0)
        feats = torch.cat([voxels[..., None, :] - points_in_voxel, points_in_voxel], -1) # n, s, 6
        feats = feats.transpose(1, 2) # n, 12, s

        feats = self.input_conv(feats) # n, c, s
        feats = feats.mean(-1) # n, c

        output_dict = {}
        sparse_feats = SparseConvTensor(feats, coors[:, [0, 3, 2, 1]], sparse_shape, batch_size)

        for i in range(self.num_stage):
            sparse_feats = self.sparse_layers[i](sparse_feats)
        
        sparse_feats = self.output_conv(sparse_feats)
        output_dict['sparse_feats'] = sparse_feats
        output_dict['voxel_size'] = voxel_size
        output_dict['pc_range'] = pc_range
        output_dict['output_scale'] = 2 ** (self.num_stage)
        output_dict['batch_size'] = batch_size

        return output_dict

    def group_feats_in_voxel(self, src, hash, num_voxel, num_sample=None):
        """
        Args:
            src: [c, n]
            hash: [n, ]
        """
        src = src.transpose(0, 1)
        src_in_tar_list = []
        src_index_in_tar_list = []
        src_index = torch.arange(0, src.shape[0]).to(src.device)
        for i in range(num_voxel):
            mask = hash == i
            src_in_tar = src[mask]
            src_index_in_tar = src_index[mask]
            src_index_in_tar_list.append(src_index_in_tar)
            if num_sample is not None:
                if len(src_in_tar) < num_sample:
                    aux = src_in_tar[0:1, ...].repeat(num_sample - len(src_in_tar), 1)
                    src_in_tar = torch.cat([src_in_tar, aux], 0)
            src_in_tar_list.append(src_in_tar[None])
        if num_sample is not None:
            return torch.cat(src_in_tar_list, 0), src_index_in_tar_list
        return src_in_tar_list, src_index_in_tar_list

            












    def get_feats_in_voxel(self, src_feats, src_coors, tar_feats, tar_coors, scale=2):
        """
        Args:
            src_feats: n, c
            src_coors: n, 4 batch_size, z, y, x
            tar_feats: m, c
            tar_coors: m, 4 (n > m)
        Returns:
            m, s, c
        """
        batch_size = src_coors[:, 0][-1]
        out_feats = []
        for i in batch_size:
            src_mask = src_coors[:, 0] == i
            src_hash = self.ravel_hash(src_coors[src_mask][:, 1:] // scale)
            src_features = src_feats[src_mask]

            tar_mask = tar_coors[:, 0] == i
            tar_hash = self.ravel_hash(tar_coors[tar_mask][:, 1:])
            tar_features = tar_feats[tar_mask]

            for i, t_h in enumerate(tar_hash):
                tf = tar_features[i] # c
                sf = src_features[src_hash == t_h] # s, c
                if len(sf) < 9:
                    sf = torch.cat([sf, sf[0][None].repeat(9-len(sf), 1)])
                sf = torch.cat([tf[None] - sf, sf], -1)
                out_feats.append(sf[None])
        out_feats = torch.cat(out_feats, 0) # m, s, c
        return out_feats.transpose(1, 2)



            
            
            
                







        
    


    def build_hash(self, x):
        assert x.ndim == 2, x.shape

        x = x - torch.min(x, axis=0)[0]
        x = x.long()
        xmax = torch.max(x, axis=0)[0] + 1

        h = torch.zeros(x.shape[0]).long().cuda()
        for k in range(x.shape[1] - 1):
            h += x[:, k]
            h *= xmax[k + 1]
        h += x[:, -1]
        return h


















