
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair
from jlcv.modules.ext import voxelization

class Voxelization(nn.Module):
    def __init__(self,
                 voxel_size,
                 point_cloud_range,
                 max_num_points,
                 max_voxels=20000):
        super(Voxelization, self).__init__()
        """
        Args:
            voxel_size (list): list [x, y, z] size of three dimension
            point_cloud_range (list):
                [x_min, y_min, z_min, x_max, y_max, z_max]
            max_num_points (int): max number of points per voxel
            max_voxels (tuple or int): max number of voxels in
                (training, testing) time
        """
        self.point_cloud_range = torch.tensor(point_cloud_range)
        self.voxel_size = torch.tensor(voxel_size)
        self.grid_size = torch.round((point_cloud_range[3:] - point_cloud_range[:3]) / voxel_size).int()
        self.sparse_shape = self.grid_size[[2, 1, 0]]
        self.sparse_shape[0] = self.sparse_shape[0] + 1 # z, y, x
        self.max_num_points = max_num_points
        if isinstance(max_voxels, tuple):
            self.max_voxels = max_voxels
        else:
            self.max_voxels = _pair(max_voxels)
    
    def voxel_layer(self, input):
        """
        Args:
            input: NC points
        """
        if self.training:
            max_voxels = self.max_voxels[0]
        else:
            max_voxels = self.max_voxels[1]

        return voxelization(input, self.voxel_size, self.point_cloud_range,
                            self.max_num_points, max_voxels)
    @torch.no_grad()
    def forward(self, points):
        voxel_dict = {}
        voxel_dict['point_cloud_range'] = self.point_cloud_range
        voxel_dict['voxel_size'] = self.voxel_size
        voxel_dict['sparse_shape'] = self.sparse_shape
        
        voxels, coors, num_points, voxel_centers = [], [], [], []
        for i, res in enumerate(points):
            res_voxels, res_coors, res_num_points = self.voxel_layer(res)
            res_voxel_centers = (
                res_coors[:, [2, 1, 0]] + 0.5) * self.voxel_size + self.point_cloud_range[0:3]
            res_coors = F.pad(res_coors, (1, 0), mode='constant', value=i)
            voxels.append(res_voxels)
            coors.append(res_coors)
            num_points.append(res_num_points)
            voxel_centers.append(res_voxel_centers)

        voxel_dict['voxels'] = torch.cat(voxels, dim=0)
        voxel_dict['coors'] = torch.cat(coors, dim=0).int()
        voxel_dict['num_points'] = torch.cat(num_points, dim=0)
        voxel_dict['voxel_centers'] = torch.cat(voxel_centers, dim=0)
        voxel_dict['batch_size'] = len(points)
        
        return voxel_dict

    def __repr__(self):
        tmpstr = self.__class__.__name__ + '('
        tmpstr += 'voxel_size=' + str(self.voxel_size)
        tmpstr += ', point_cloud_range=' + str(self.point_cloud_range)
        tmpstr += ', max_num_points=' + str(self.max_num_points)
        tmpstr += ', max_voxels=' + str(self.max_voxels)
        tmpstr += ')'
        return tmpstr