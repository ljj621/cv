import torch
import torch.nn as nn
import numpy as np
from ..instance import Instance
from .bbox import Bbox
# from jlcv.modules.ext import roiaware_pool3d

class LidarBbox(Bbox):
    def __init__(self, data):
        super().__init__(data)

    @property
    def corners(self):
        self.data = self.to_tensor(self.data)

        template = self.data.new_tensor((
            [1, 1, -1], [1, -1, -1], [-1, -1, -1], [-1, 1, -1],
            [1, 1, 1], [1, -1, 1], [-1, -1, 1], [-1, 1, 1],
        )) / 2

        corners = self.data[:, None, 3:6].repeat(1, 8, 1) * template[None, :, :]
        angle = self.data[:, 6]

        corners = rotate_along_z(corners.view(-1, 8, 3), angle).view(-1, 8, 3)
        corners += self.data[:, None, 0:3]
        return corners
    
    def points_in_bbox_cpu(self, points):
        """
        Args:
            points: (num_points, 3)
            boxes: [x, y, z, dx, dy, dz, heading], (x, y, z) is the box center, each box DO NOT overlaps
        Returns:
            index: (N, num_points)
        """
        data = torch.from_numpy(points[:, 0:3])
        bbox = torch.from_numpy(self.data)
        index = roiaware_pool3d.points_in_boxes_cpu(data, bbox)
        return index





        

