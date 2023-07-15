from jlcv.registry import Registry

METRICS = Registry('metric')

from .f1_score import F1Score
from .accuracy import Accuracy
from .chamfer_3d_loss import Chamfer3DLoss
from .cls_loss import ClsLoss
from .smoth_cls_loss import SmoothClsLoss

__all__ = [
    'F1Score','METRICS'
]