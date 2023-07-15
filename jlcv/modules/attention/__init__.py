from jlcv.registry import Registry
ATTENTION = Registry('attention')

from .attention import SelfAttention, CrossAttention
from .point_local_attention import PointLocalAttention
from .point_local_scale_attention import PointLocalScaleAttention
from .dense_scaled_attention import DenseScaledAttention
from .cascaded_pyramid_attention import CascadedPyramidAttention

__all__ = [
    'ATTENTION',
    'CrossAttention', 'SelfAttention', 
    'PointLocalAttention', 'DenseScaledAttention', 'CascadedPyramidAttention','PointLocalScaleAttention'
]