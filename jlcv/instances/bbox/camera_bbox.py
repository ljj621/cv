import torch
import torch.nn as nn
import numpy as np
from ..instance import Instance
from .lidar_bbox import LidarBbox

class CameraBbox(Instance):
    def __init__(self, data):
        super().__init__(data)

    def to_lidar(self, R0, Tr_velo2cam):
        loc = self.data[:, 3]
        dims = self.data[3, 6]
        rots = self.data[6]
        loc_hom = np.hstack((loc, np.ones((loc.shape[0], 1), dtype=np.float32)))

        if R0.shape == torch.Size([3, 3]):
            R0_hom = np.hstack((R0, np.zeros((3, 1), dtype=np.float32)))  # (3, 4)
            R0_hom = np.vstack((R0_hom, np.zeros((1, 4), dtype=np.float32)))  # (4, 4)
            R0_hom[3, 3] = 1

        elif R0.shape == torch.Size([4, 4]): R0_hom = R0

        if Tr_velo2cam.shape == torch.Size([3, 4]):
            Tr_velo2cam_hom = np.vstack((Tr_velo2cam, np.zeros((1, 4), dtype=np.float32)))  # (4, 4)
            Tr_velo2cam_hom[3, 3] = 1

        elif Tr_velo2cam.shape == torch.Size([4, 4]): Tr_velo2cam_hom = Tr_velo2cam
        
        loc_lidar = np.dot(loc_hom, np.linalg.inv(np.dot(R0_hom, Tr_velo2cam_hom).T))[:, :3]
        loc_lidar[:, 2] += dims[:, 1:2][:, 0] / 2
        gt_boxes_lidar = np.concatenate([loc_lidar, dims, -(np.pi / 2 + rots[..., np.newaxis])], axis=1)
        return LidarBbox(gt_boxes_lidar)


