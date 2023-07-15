from jlcv.registry import Registry
CLASSIFIER = Registry('classifier')

from .mymodel import MyModel
# from .pointnet import PointNet
# from .pointnet2 import PointNet2

    
__all__ = [
    'MyModel',  'CLASSIFIER'
]