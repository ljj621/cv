import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from jlcv.models import MODELS


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
            
            # if 'voxelize' in points_layer:
            #     voxelize = points_layer['voxelize']
            #     self.voxel_size = voxelize.get('voxel_size', None)
            #     self.point_cloud_range = voxelize.get('point_cloud_range', None)
            #     self.max_num_points = voxelize.get('max_num_points', None)
            #     self.max_voxels = voxelize.get('max_voxels', None)
            #     self.voxel_layer = Voxelization(**voxelize)
    
    def forward(self, input_dict):
        output_dict = {}
        
        if 'points' in input_dict:
            if self.points_pipelines is None:
                points = input_dict['points']
                points.to(torch.device('cuda'))
                output_dict.update(points.to_dict())
                output_dict["points"] = output_dict.pop("data").transpose(1,2)
            else:
                original_points = input_dict['points']
                original_points.to(torch.device('cuda'))
                output_dict.update(original_points.to_dict(name='original'))
                output_dict["original_points"] = output_dict.pop("original_data").transpose(1,2)

                points = original_points.clone()
                points.transform(self.points_pipelines)
                output_dict.update(points.to_dict())
                output_dict["points"] = output_dict.pop("data").transpose(1,2)

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








