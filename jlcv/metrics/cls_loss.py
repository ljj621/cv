import torch.nn.functional as F
from . import METRICS

@METRICS.register_module()
class ClsLoss(object):
    def __call__(self, preds, targets, weight=None):
        loss = F.cross_entropy(preds, targets, weight)
        return loss


       