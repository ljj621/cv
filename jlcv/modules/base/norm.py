import torch.nn as nn
from jlcv.registry import Registry

NORM = Registry('norm')
NORM.register_module('BN1d', module=nn.BatchNorm1d)
NORM.register_module('BN2d', module=nn.BatchNorm2d)
NORM.register_module('BN3d', module=nn.BatchNorm3d)
NORM.register_module('GN', module=nn.GroupNorm)
NORM.register_module('LN', module=nn.LayerNorm)
NORM.register_module('IN1d', module=nn.InstanceNorm1d)
NORM.register_module('IN2d', module=nn.InstanceNorm2d)
NORM.register_module('IN3d', module=nn.InstanceNorm3d)
