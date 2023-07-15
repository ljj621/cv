# import cv2
import numpy as np
import torch
import open3d as o3d
from open3d import geometry

class Visualizer(object):
    def __init__(self):
        super(Visualizer, self).__init__()
        self.visualizer = o3d.visualization.Visualizer()
        self.visualizer.create_window()
        self.geometries = []
        
    def show(self, save_path=None):
        self.visualizer.run()

        if save_path is not None:
            self.visualizer.capture_screen_image(save_path)
    
    def add_coordinate_frame(self, size=1, origin=[0, 0, 0], name='coordinate_frame'):
        if name not in self.geometries:
            self.geometries.append('name')
            mesh_frame = geometry.TriangleMesh.create_coordinate_frame(size=size, origin=origin)
            self.visualizer.add_geometry(mesh_frame)

    def add_points(self, points, size=2, colors=[0.5, 0.5, 0.5], name='points'):
        if isinstance(points, torch.Tensor):
            points = points.cpu().numpy()
        if len(points.shape) == 3 and points.shape[0] == 1:
            points = points[0, :, :]
        assert len(points.shape) == 2 and points.shape[-1] >= 3
        N, C = points.shape

        if isinstance(colors, list):
            colors = np.array(colors)[None].repeat(N, 0)
        elif isinstance(colors, torch.Tensor):
            colors = colors.cpu().numpy()
        assert colors.shape == torch.Size([N, 3])

        self.visualizer.get_render_option().point_size = size
        
        if name not in self.geometries:
            pcd = geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points[:, :3])
            pcd.colors = o3d.utility.Vector3dVector(colors)
            self.__setattr__(name, pcd)
            self.visualizer.add_geometry(self.__getattribute__(name))
        else:
            pcd = self.__getattribute__(name)
            pcd.points = o3d.utility.Vector3dVector(points[:, :3])
            pcd.colors = o3d.utility.Vector3dVector(colors)
            self.__setattr__(name, pcd)
            self.visualizer.update_geometry(self.__getattribute__(name))

    
    def add_bboxes(self, bbox3d, bbox_color=[0., 1., 0.], points_in_box_color=[1., 0., 0.], rot_axis=2):
        if isinstance(bbox3d, torch.Tensor):
            bbox3d = bbox3d.cpu().numpy()
        bbox3d = bbox3d.copy()

        has_pcd = self.__getattribute__('pcd') is not None

        if has_pcd:
            points_colors = np.asarray(self.pcd.colors)

        in_box_color = np.array(points_in_box_color)
        for i in range(len(bbox3d)):
            center = bbox3d[i, 0:3]
            dim = bbox3d[i, 3:6]
            yaw = np.zeros(3)
            yaw[rot_axis] = -bbox3d[i, 6]
            rot_mat = geometry.get_rotation_matrix_from_xyz(yaw)

            center[rot_axis] += dim[rot_axis] / 2  # lidar bottom center to gravity center
           
            box3d = geometry.OrientedBoundingBox(center, rot_mat, dim)

            line_set = geometry.LineSet.create_from_oriented_bounding_box(box3d)
            line_set.paint_uniform_color(bbox_color)
            
            self.visualizer.add_geometry(line_set)

            # change the color of points which are in box
            if has_pcd:
                indices = box3d.get_point_indices_within_bounding_box(self.pcd.points)
                points_colors[indices] = in_box_color

        # update points colors
        if has_pcd:
            self.pcd.colors = o3d.utility.Vector3dVector(points_colors)
            self.visualizer.update_geometry(self.pcd)
