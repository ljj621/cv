import torch
import torch.nn as nn
import numpy as np
import math
from ..instance import Instance
from typing import Iterator, Optional, Sequence, Union
from torch import Tensor

class Points(Instance):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.data = self.to_tensor(self.data).float()
    
    @property
    def coord(self) -> Tensor:
        """Coordinates of points in shape (N, 3)."""
        return self.to_tensor(self.data[..., :3])
    
    @coord.setter
    def coord(self, data: Union[Tensor, np.ndarray]) -> None:
        """Set the coordinates of each point."""
        data = self.to_tensor(data)
        self.data[..., :3] = data
    
    @property
    def color(self) -> Union[Tensor, None]:
        """Colors of points in shape (N, 3)."""        
        return self.data[..., 3:6] if self.shape[-1] > 3 else None

    @color.setter
    def color(self, data: Union[Tensor, np.ndarray]) -> None:
        """Set the color of each point."""
        data = self.to_tensor(data)
        self.data = torch.cat([self.coord, data], -1)
    
    # @property
    # def seg_labels(self) -> Tensor:
    #     """Segmentation labels of points in shape (B, N)."""
    #     if self.has_property('seg_labels'):
    #         return self.seg_labels
    #     else:
    #         return None
    
    # @seg_labels.setter
    # def seg_labels(self, data: Union[Tensor, np.ndarray]) -> None:
    #     """Set the segmentation label of each point."""
    #     data = self.to_tensor(data)
    #     self.seg_labels = data

    def shuffle(self) -> None:
        """Shuffle the points."""
        index = torch.randperm(self.shape[0], device=self.data.device)
        self.data = self.data[index]

        if self.has_property('seg_labels'):
            seg_labels = self.seg_labels[index]
            self.seg_labels = seg_labels

    def normalize(self):
        self.data = self.to_tensor(self.data)
        if not self.data.ndim == 3:
            self.data = self.data[None]
        batch_size = self.data.shape[0]
        centroid = self.data.mean(1)[:,None,:]
        self.data = self.data - centroid
        m = torch.sqrt(torch.sum(self.data**2, axis=-1))
        m = torch.max(m, -1)[0][:,None,None]
        self.data = self.data / m
        if batch_size == 1:
            self.data = self.data[0]
    
    def rotate(self, rotation: Union[np.ndarray, Tensor, float, None]=None, axis: str = 'z'):
        """Rotate points with the given rotation matrix or angle.

        Args:
            rotation (Tensor or np.ndarray or float or None): Rotation matrix or angle.
            axis (str): Axis to rotate at. Defaults to z.
        """
        coord = self.coord
        if self.batch_free():
            coord = coord[None]
        batch_shape = coord.shape

        if rotation is None:
            rotation = torch.Tensor(size=[batch_shape[0], ]).uniform_(0, 1) * 2 * math.pi
            rot_sin = torch.sin(rotation)
            rot_cos = torch.cos(rotation)
            ones = torch.ones_like(rot_cos)
            zeros = torch.zeros_like(rot_cos)

            if axis == 'z':
                rot_mat_T = torch.stack([
                    torch.stack([rot_cos, rot_sin, zeros]),
                    torch.stack([-rot_sin, rot_cos, zeros]),
                    torch.stack([zeros, zeros, ones])
                ])
            elif axis == 'x':
                rot_mat_T = torch.stack([
                    torch.stack([ones, zeros, zeros]),
                    torch.stack([zeros, rot_cos, rot_sin]),
                    torch.stack([zeros, -rot_sin, rot_cos])
                ])
            elif axis == 'y':
                rot_mat_T = torch.stack([
                    torch.stack([rot_cos, zeros, -rot_sin]),
                    torch.stack([zeros, ones, zeros]),
                    torch.stack([rot_sin, zeros, rot_cos])
                ])

        elif rotation.shape == [3, 3]:
            rot_mat_T = rotation
        

        # coord = coord @ rot_mat_T.to(self.device)
        coord = torch.einsum('aij,jka->aik', coord, rot_mat_T)
        coord = coord[0,...] if self.batch_free() else coord
        self.coord = coord
    
    def translate(self, translation: Union[Tensor, np.ndarray, list]=[-0.1, 0.1]):
        """Translate points with the translation.

        Args:
            trans_vector (Tensor or np.ndarray or list)
        """
        coord = self.coord
        if self.batch_free():
            coord = coord[None]
        batch_shape = coord.shape

        if isinstance(translation, list):
            trans_vector = torch.Tensor(size=[batch_shape[0], 1, 3]).uniform_(translation[0], translation[1])
        
        coord = coord + trans_vector.to(self.device)
        coord = coord[0,...] if self.batch_free() else coord
        self.coord = coord
    
    def scale(self, scale_factor: Union[Tensor, np.ndarray, list]=[0.8, 1.25]):
        """Scale the points with horizontal and vertical scaling factors.

        Args:
            scale_factor (float): Scale factors to scale the points.
        """
        coord = self.coord
        if self.batch_free():
            coord = coord[None]
        batch_shape = coord.shape

        if isinstance(scale_factor, list):
            scale_factor = torch.Tensor(size=[batch_shape[0], 1, 1]).uniform_(scale_factor[0], scale_factor[1])

        coord = coord * scale_factor.to(self.device)
        coord = coord[0,...] if self.batch_free() else coord
        self.coord = coord

    
    def noise(self, sigma=0.01, clip=0.05):
        if not self.data.ndim == 3:
            self.data = self.data[None]
        batch_size, n, c = self.data.shape
        device = self.data.device
        noises = torch.clip(
            sigma * torch.randn([batch_size, n, c], device=device),
            -clip, clip)
        self.data = self.data + noises
        if batch_size == 1:
            self.data = self.data[0]

    def dropout(self, max_dropout_ratio):
        if not self.data.ndim == 3:
            self.data = self.data[None]
        batch_size, n, c = self.data.shape
        device = self.data.device
     
        for b in range(batch_size):
            dropout_ratio = torch.rand(1)*max_dropout_ratio
            drop_mask = torch.rand(n)<=dropout_ratio
            if len(drop_mask)>0:
                self.data[b, drop_mask, :] = self.data[b,0, :].clone()
        if batch_size == 1:
            self.data = self.data[0]
        # self.add_meta('max_dropout_ratio', max_dropout_ratio)
    
    def furthest_point_sample(self, num_points):
        if self.batch_free():
            points = self.data[None]

        points = self.data.transpose(1, 2)
        from jlcv.modules.ext import furthest_point_sample, gather_points
        fps_index = furthest_point_sample(points[:,:3,:], num_points)
        fps_points = gather_points(points, fps_index) # b, c, m
        self.data = fps_points.transpose(1,2)
        self.set_property('index', fps_index)

        if self.has_property('seg_labels'):
            seg_labels = self.seg_labels.gather(1, fps_index.long())
            self.seg_labels = seg_labels
    
    def index_points(self, points, idx):
        """

        Input:
            points: input points data, [B, N, C]
            idx: sample index data, [B, S]
        Return:
            new_points:, indexed points data, [B, S, C]
        """
        device = points.device
        B = points.shape[0]
        view_shape = list(idx.shape)
        view_shape[1:] = [1] * (len(view_shape) - 1)
        repeat_shape = list(idx.shape)
        repeat_shape[0] = 1
        batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
        new_points = points[batch_indices, idx, :]
        return new_points
    
    def furthest_point_sample_cpu(self, num_points):
        if self.batch_free():
            points = self.data[None]
        device = points.device
        B, N, C = points.shape
        centroids = torch.zeros(B, num_points, dtype=torch.long).to(device)
        distance = torch.ones(B, N).to(device) * 1e10
        farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
        batch_indices = torch.arange(B, dtype=torch.long).to(device)
        for i in range(num_points):
            centroids[:, i] = farthest
            centroid = points[batch_indices, farthest, :].view(B, 1, 3)
            dist = torch.sum((points - centroid) ** 2, -1)
            mask = dist < distance
            distance[mask] = dist[mask]
            farthest = torch.max(distance, -1)[1]
        new_xyz = self.index_points(points, centroids)
        self.data = new_xyz[0, ...]
        
    



  



