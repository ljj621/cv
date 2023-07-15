from jlcv.registry import Registry
# SEGMENTATIONDATASETS = Registry('segmentation_dataset')
from torch.utils.data import Dataset
from .shapenet_part import ShapeNetPartDataset
from .s3dis_dataset import S3DISDataset

__all__ = [
    'SEGMENTATIONDATASETS', 'ShapeNetPartDataset', 'S3DISDataset'
]