
from jlcv.registry import Registry
SEGMENTATION = Registry('segmentation')
from .mymodel import MyModel

__all__ = [
    'MyModel', 'SEGMENTATION'
]