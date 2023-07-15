import torch
import torch.nn.functional as F
from . import METRICS


@METRICS.register_module()
class SmoothClsLoss(object):
    def __init__(self, smoothing_ratio=0.1) -> None:
        self.smoothing_ratio = smoothing_ratio
    
    def __call__(self, preds, targets):
        eps = self.smoothing_ratio
        num_classes = preds.shape[1]

        one_hot = F.one_hot(targets.long(), num_classes)
        one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (num_classes - 1)

        loss = -(one_hot * preds).sum(dim=1).mean()
        return loss


       