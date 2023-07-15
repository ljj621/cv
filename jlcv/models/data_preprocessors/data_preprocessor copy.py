import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from jlcv.models import MODELS
from jlcv.modules.ext import furthest_point_sample, gather_points


@MODELS.register_module()
class DataPreprocessor(nn.Module):
    def __init__(self,
                 data_type=['points'],
                 points_layer=None,
                 img_layer=None,
        ):
        super().__init__()
        self.data_type = data_type
        self.points_layer = points_layer
        if 'points' in data_type and points_layer is not None:
            self.points_pipelines = points_layer.get('pipelines', None)
            self.points_module = points_layer.get('points_module', None)
            if self.points_module is not None:
                self.points_module = MODELS.build(self.points_module)
            
            if 'voxelize' in points_layer:
                voxelize = points_layer['voxelize']
                self.num_points = voxelize.get('num_points', None)
                self.point_cloud_range = voxelize.get('point_cloud_range', None)
                self.voxel_size = voxelize.get('voxel_size', None)
                self.num_voxels = voxelize.get('num_voxels', None)
                self.max_num_points = voxelize.get('max_num_points', None)
    
    def forward(self, input_dict):
        output_dict = {}

        original_points = input_dict['points']
        original_points.to(torch.device('cuda'))
        data = original_points.data
        if original_points.batch_free():
            data = data[None]
        data = data.transpose(1, 2)
        seg_labels = original_points.seg_labels
        output_dict['original_points'] = data
        output_dict['original_seg_labels'] = seg_labels
        output_dict['batch_size'] = data.shape[0]

        fps_index = furthest_point_sample(data[..., 3:6, :], 1024)
        points = gather_points(data, fps_index) # b, c, n
        output_dict['points'] = points
        output_dict['index'] = fps_index
        output_dict['seg_labels'] = seg_labels.gather(1, fps_index.long())

        if self.points_module is not None:
            feats = self.points_module(points, points[:, 3:6,:]) # b, n, c
            output_dict['feats'] = feats
        else:
            feats = None
        
        voxel_dict = self.voxelize(points, feats)
        output_dict.update(voxel_dict)
        return output_dict

    def forward(self, input_dict):
        output_dict = {}

        original_points = input_dict['points']
        original_points.to(torch.device('cuda'))
        output_dict['original_points'] = original_points.data.transpose(1, 2)
        output_dict['original_seg_labels'] = original_points.seg_labels
        output_dict['batch_size'] = original_points.shape[0]

        points = original_points.clone()
        points.to(torch.device('cuda'))
        points.transform(self.points_pipelines)
        data = points.data.transpose(1, 2)
        output_dict['points'] = data
        output_dict['index'] = points.index
        output_dict['seg_labels'] = points.seg_labels

        if self.points_module is not None:
            feats = self.points_module(data, data[:, :3,:]) # b, n, c
            output_dict['feats'] = feats
        else:
            feats = None
        
        voxel_dict = self.voxelize(data, feats)
        output_dict.update(voxel_dict)
        return output_dict

            
    @torch.no_grad()
    def voxelize(self, points, feats=None):
        """
        Args:
            points: b, c, n
            feats: b, c, n
        """
        points = points.transpose(1,2)
        pc_range = points.new_tensor(self.point_cloud_range)
        voxel_size = points.new_tensor(self.voxel_size)
        grid_size = torch.round((pc_range[3:] - pc_range[:3]) / voxel_size).int()
        sparse_shape = grid_size[[2, 1, 0]]
        sparse_shape[0] = sparse_shape[0] + 1 # z, y, x

        if feats is not None:
            feats = feats.transpose(1, 2) # b, n, c

        output_dict = {
            'sparse_shape': sparse_shape.cpu().numpy(),
            'voxel_size': voxel_size,
            'pc_range': pc_range
        }

        voxels, coors = [], []
        for i, res in enumerate(points):
            res_points, res_aux = res[:, :3], res[:, 3:]
            in_range_flags = ((res_points[:, 0] > pc_range[0])
                          & (res_points[:, 1] > pc_range[1])
                          & (res_points[:, 2] > pc_range[2])
                          & (res_points[:, 0] < pc_range[3])
                          & (res_points[:, 1] < pc_range[4])
                          & (res_points[:, 2] < pc_range[5]))
            res_points = res_points[in_range_flags]
            res_points[:, 0] = res_points[:, 0] - pc_range[0]
            res_points[:, 1] = res_points[:, 1] - pc_range[1]
            res_points[:, 2] = res_points[:, 2] - pc_range[2]
            res_aux = res_aux[in_range_flags]
            res_coors = torch.round(res_points / voxel_size).int() # x, y, z
            if (len(res_coors) == 0):
                c = res_coors
            # res_coors -= res_coors.min(0)[0]

            hash_table = self.build_hash(res_coors)
            _, inds, point2voxel_map = np.unique(hash_table.cpu().numpy(), return_index=True, return_inverse=True)
            res_coors = res_coors[inds][:,[2, 1, 0]] # xyz -> zyx
            res_voxels = res_points[inds]
            
            if len(res_aux) > 0:
                res_aux = res_aux[inds]
                res_voxels = torch.cat([res_voxels, res_aux], -1)
            if feats is not None:
                res_feats = feats[i, ...][inds]
                res_voxels = torch.cat([res_voxels, res_feats], -1)
            res_coors = F.pad(res_coors, (1, 0), mode='constant', value=i)

            voxels.append(res_voxels)
            coors.append(res_coors)

        voxels = torch.cat(voxels, dim=0)
        coors = torch.cat(coors, dim=0)
        output_dict['coors'] = coors
        output_dict['voxels'] = voxels
        return output_dict

    def build_hash(self, x: torch.Tensor) -> torch.Tensor:
        """Get voxel coordinates hash for np.unique.

        Args:
            x (torch.Tensor): The voxel coordinates of points, Nx3.

        Returns:
            torch.Tensor: Voxels coordinates hash.
        """
        assert x.ndim == 2, x.shape

        x = x - torch.min(x, 0)[0]
        x = x.long()
        xmax = torch.max(x, 0)[0].long() + 1

        h = torch.zeros(x.shape[0]).long().cuda()
        for k in range(x.shape[1] - 1):
            h += x[:, k]
            h *= xmax[k + 1]
        h += x[:, -1]
        return h








