import torch
from .optimizer import OPTIMIZER

torch.optim.lr_scheduler.CyclicLR()

@OPTIMIZER.register_module()
class OneCyCleLR(object):
    def __init__(self,
                 optimizer,
                 base_lr,
                 lr_ratio,
                 warmup = 'exp',
                 warmup_iters=0,
                 warmup_ratio=0.1,
                 step_ratio_up=0.4) -> None:
        self.optimizer = optimizer
        self.warmup = warmup
        self.warmup_iters = warmup_iters
        self.warmup_ratio = warmup_ratio
        self.step_ratio_up = step_ratio_up

        self.base_lr = base_lr
        self.max_lr = base_lr * lr_ratio[0]
        self.min_lr = base_lr * lr_ratio[1]
    
    def update_lr(self, lr, cur_iters):
        if cur_iters < self.warmup_iters:
            cur_lr = self.warmup_lr(cur_iters)
        else:
            cur_lr = self.get_lr()
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
    
    def get_lr(self):
        return self.optimizer.state_dict()['param_groups'][0]['lr']
    
    def regular_lr(self, lr):
        if lr < self.max_lr:
            lr = lr * self.
    
    def warmup_lr(self, cur_iters):
        if self.warmup == 'constant':
            warmup_lr = [_lr * self.warmup_ratio for _lr in self.regular_lr]
        elif self.warmup == 'linear':
            k = (1 - cur_iters / self.warmup_iters) * (1 - self.warmup_ratio)
            warmup_lr = [_lr * (1 - k) for _lr in self.regular_lr]
        elif self.warmup == 'exp':
            k = self.warmup_ratio**(1 - cur_iters / self.warmup_iters)
            warmup_lr = [_lr * k for _lr in self.regular_lr]
        return warmup_lr
    