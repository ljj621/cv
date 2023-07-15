from jlcv.registry import Registry
MODELS = Registry('models')
from .backbones import *
from .data_preprocessors import *
from .seg_heads import *