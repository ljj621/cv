import torch
import torch.nn as nn
import numpy as np
from ..instance import Instance

class Bbox(Instance):
    def __init__(self, data, **kwargs):
        super().__init__(data)

        for k,v in kwargs.items():
            self.set_property(k, v)
    
    def in_hull(p, hull):
        """
        :param p: (N, K) test points
        :param hull: (M, K) M corners of a box
        :return (N) bool
        """
        try:
            if not isinstance(hull, Delaunay):
                hull = Delaunay(hull)
            flag = hull.find_simplex(p) >= 0
        except scipy.spatial.qhull.QhullError:
            print('Warning: not a hull %s' % str(hull))
            flag = np.zeros(p.shape[0], dtype=np.bool)

        return flag
    
    @property
    def corners(self):
        pass
    
    def rotate_along_z(self, data, angle):
        return super().rotate(data, angle, 'z')

    def normalize(self):
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
    
    def rotate(self, rotation=[0, 1]):
        if not self.data.ndim == 3:
            self.data = self.data[None]
        batch_size = self.data.shape[0]
        device = self.data.device
        if isinstance(rotation, list):
            rotate_factor = torch.Tensor(size=[batch_size, ]).uniform_(rotation[0], rotation[1]) * 2 * torch.pi
        self.add_meta('rotate_factor', rotate_factor)
        rotate_factor=rotate_factor.to(device)

        rot_sin = torch.sin(rotate_factor)
        rot_cos = torch.cos(rotate_factor)
        rotate_mat = torch.eye(3, device=device)[None].repeat(batch_size, 1, 1)
        rotate_mat[:, 0, 0] = rot_cos
        rotate_mat[:, 0, 1] = rot_sin
        rotate_mat[:, 1, 0] = -rot_sin
        rotate_mat[:, 1, 1] = rot_cos

        self.data[..., :3] = self.data[..., :3] @ rotate_mat

    def translate(self, translation=[-0.1, 0.1]):
        if not self.data.ndim == 3:
            self.data = self.data[None]
        batch_size = self.data.shape[0]
        device = self.data.device

        if isinstance(translation, list):
            translate_factor = torch.Tensor(size=[batch_size, 1, 3]).uniform_(translation[0], translation[1])
        self.add_meta('translate_factor', translate_factor)

        self.data[..., :3] = self.data[..., :3] + translate_factor.to(device)

        if batch_size == 1:
            self.data = self.data[0]
    
    def scale(self, scale_factor=[0.8, 1.25]):
        if not self.data.ndim == 3:
            self.data = self.data[None]
        batch_size = self.data.shape[0]
        device = self.data.device

        if isinstance(scale_factor, list):
            scale_factor = torch.Tensor(size=[batch_size, 1, 1]).uniform_(scale_factor[0], scale_factor[1])
        self.add_meta('scale_factor', scale_factor)
     
        self.data[..., :3] = self.data[..., :3] * scale_factor.to(device)

        if batch_size == 1:
            self.data = self.data[0]
    
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

    def shuffle(self):
        if not self.data.ndim == 3:
            self.data = self.data[None]
        batch_size, n, c = self.data.shape
        device = self.data.device

        index = torch.randint(0, n, [batch_size, n], device=device).long()
        self.data = self.data.gather(1, index[..., None].repeat(1, 1, c)).clone()
        if batch_size == 1:
            self.data = self.data[0]
        self.add_meta('shuffle', True)
    
    @staticmethod
    def voxels2pixels(cam_E, cam_k, vox_origin, voxel_size, img_W, img_H, scene_size):
        # Compute the x, y, z bounding of the scene in meter
        vol_bnds = np.zeros((3,2))
        vol_bnds[:,0] = vox_origin
        vol_bnds[:,1] = vox_origin + np.array(scene_size)

        # Compute the voxels centroids in lidar cooridnates
        vol_dim = np.ceil((vol_bnds[:,1]- vol_bnds[:,0])/ voxel_size).copy(order='C').astype(int)
        xv, yv, zv = np.meshgrid(
                range(vol_dim[0]),
                range(vol_dim[1]),
                range(vol_dim[2]),
                indexing='ij'
            )
        vox_coords = np.concatenate([
                xv.reshape(1,-1),
                yv.reshape(1,-1),
                zv.reshape(1,-1)
            ], axis=0).astype(int).T

        # Project voxels'centroid from lidar coordinates to camera coordinates
        cam_pts = fusion.TSDFVolume.vox2world(vox_origin, vox_coords, voxel_size)
        cam_pts = fusion.rigid_transform(cam_pts, cam_E)

        # Project camera coordinates to pixel positions
        projected_pix = fusion.TSDFVolume.cam2pix(cam_pts, cam_k)
        pix_x, pix_y = projected_pix[:, 0], projected_pix[:, 1]

        # Eliminate pixels outside view frustum
        pix_z = cam_pts[:, 2]
        fov_mask = np.logical_and(pix_x >= 0,
                    np.logical_and(pix_x < img_W,
                    np.logical_and(pix_y >= 0,
                    np.logical_and(pix_y < img_H,
                    pix_z > 0))))


        return projected_pix, fov_mask, pix_z

