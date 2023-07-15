from .points import Points
# from .camera_points import CameraPoints

class LidarPoints(Points):
    def __init__(self, data, **kwargs):
        super().__init__(data, **kwargs)
    
    # def to_camera(self, R0, Tr_velo2cam):
    #     points = self.data[:, :3]
    #     points_hom = np.hstack((points, np.ones((points.shape[0], 1), dtype=np.float32)))
    #     points_camera = np.dot(points_hom, np.dot(Tr_velo2cam.T, R0.T))
    #     points_camera = np.cat([points_camera, self.data[:, 3:]], -1)
    #     return CameraPoints(points_camera)
    
    # def to_fov(self, R0, Tr_velo2cam, P, img_shape):
    #     points_camera = self.to_camera(R0, Tr_velo2cam)
    #     points_img, points_depth = points_camera.to_img(P)
    #     points_img, points_depth = points_img.data, points_depth.data

    #     val_flag_1 = np.logical_and(points_img[:, 0] >= 0, points_img[:, 0] < img_shape[1])
    #     val_flag_2 = np.logical_and(points_img[:, 1] >= 0, points_img[:, 1] < img_shape[0])
    #     val_flag_merge = np.logical_and(val_flag_1, val_flag_2)

    #     fov_flag = np.logical_and(val_flag_merge, points_depth >= 0)
    #     points_fov = self.data[fov_flag]
    #     return LidarPoints(points_fov)

