import torch.nn as nn
from jlcv.registry import Registry

ACTIVATION = Registry('activation')
ACTIVATION.register_module('ReLU', module=nn.ReLU)
ACTIVATION.register_module('LeakyReLU', module=nn.LeakyReLU)
ACTIVATION.register_module('ELU', module=nn.ELU)
ACTIVATION.register_module('GELU', module=nn.GELU)
ACTIVATION.register_module('Sigmoid', module=nn.Sigmoid)
ACTIVATION.register_module('Tanh', module=nn.Tanh)
