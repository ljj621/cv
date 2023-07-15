import torch.optim as optim
from jlcv.registry import Registry

OPTIMIZER = Registry('optimizer')

OPTIMIZER.register_module('AdamW', optim.AdamW)
OPTIMIZER.register_module('Adam', optim.Adam)
OPTIMIZER.register_module('SGD', optim.SGD)

