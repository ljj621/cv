from .optimizer import OPTIMIZER
from .scheduler import SCHEDULER


class Optimizer(object):
    @staticmethod
    def build(model, cfg):
        
        optimizer = OPTIMIZER.build(cfg.optimizer, filter(lambda p : p.requires_grad, model.parameters()))
        # optimizer = build_from_cfg(cfg.optimizer, OPTIMIZER, model.parameters())
        scheduler = SCHEDULER.build(cfg.scheduler, optimizer)
        return optimizer, scheduler

__all__ = [
    'Optimizer', 'OPTIMIZER', 
]