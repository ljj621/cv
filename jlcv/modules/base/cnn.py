import torch.nn as nn
from jlcv.registry import Registry

CONV = Registry('conv')
CONV.register_module('Conv1d', module=nn.Conv1d)
CONV.register_module('Conv2d', module=nn.Conv2d)
CONV.register_module('Conv3d', module=nn.Conv3d)
CONV.register_module('ConvTranspose2d', module=nn.ConvTranspose2d)
CONV.register_module('ConvTranspose3d', module=nn.ConvTranspose3d)
