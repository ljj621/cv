from jlcv.registry import Registry
import torch.optim.lr_scheduler as lr_scheduler

SCHEDULER = Registry('lr_scheduler')

SCHEDULER.register_module('ExponentialLR', lr_scheduler.ExponentialLR)
SCHEDULER.register_module('CosineAnnealingLR', lr_scheduler.CosineAnnealingLR)
SCHEDULER.register_module('StepLR', lr_scheduler.StepLR)
SCHEDULER.register_module('CosineAnnealingWarmRestarts', lr_scheduler.CosineAnnealingWarmRestarts)