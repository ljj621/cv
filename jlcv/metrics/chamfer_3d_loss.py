from jlcv.modules.ext import chamfer_dist_3d
import torch
from . import METRICS

@METRICS.register_module()
class Chamfer3DLoss(object):
    def __call__(self, preds, gt):
        dist1, dist2, _, _ = chamfer_dist_3d(preds, gt)
        loss_cd1 = (torch.sqrt(dist1).mean(1) + torch.sqrt(dist2).mean(1)).mean()
        loss_cd2 = dist1.mean() + dist2.mean()
        return loss_cd1, loss_cd2, dist1.mean(1)[...,None], dist2.mean(1)[...,None]

            

      




