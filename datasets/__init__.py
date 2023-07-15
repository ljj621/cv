from jlcv.registry import Registry
DATASETS = Registry('dataset')
from .dataloader import DataContainer
from .classify import ModelNetDataset
from .completion import Complete3DDataset
from .segmentation import ShapeNetPartDataset, S3DISDataset


__all__ = [
    'DATASETS', 
    'DataContainer',
    'ModelNetDataset',
    'Complete3DDataset', 
    'ShapeNetPartDataset', 'S3DISDataset'
]