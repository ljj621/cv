import torch.nn.functional as F
import torch
from . import METRIC_LAYERS

@METRIC_LAYERS.register_module()
class SCELoss(object):
    def __init__(self, alpha=0.1, beta=1, A=1e-6) -> None:
        self.alpha = alpha
        self.beta = beta
        self.A = A
    def __call__(self, preds, targets, weights=None):
        """
        Args:
        preds: [N, C]
        targets: [N]
        """
        ce = F.cross_entropy(preds, targets)

        targets_one_hot = F.one_hot(targets, self.num_class).float().to(preds.device)
        targets_one_hot = torch.clamp(targets_one_hot, self.A, 1)
        preds_norm = F.softmax(preds, -1)
        preds_norm = torch.clamp(preds_norm, 1e-7, 1)

        rce = -torch.sum(preds_norm * torch.log(targets_one_hot), 1).mean()
        loss = ce * self.alpha + rce *self.beta
        return loss


       