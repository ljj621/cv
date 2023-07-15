from jlcv.registry import Registry
PIPELINES = Registry('pipelines')

from .instance import Instance
from .points import Points
# from .camera_points import CameraPoints
# from .lidar_points import LidarPoints
# from .camera_bbox import CameraBbox
# from .lidar_bbox import LidarBbox
# from .calibration import Calibration

# __all__ = [
#     'Instance',
#     'PIPELINES', 'Points',  'CameraBbox', 'LidarBbox', 'CameraPoints', 'LidarPoints'
# ]