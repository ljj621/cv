from . import METRICS
import numpy as np
from jlcv.average_meter import MetricMeter

@METRICS.register_module()
class Accuracy(object):
    def __init__(self) -> None:
        self.meter = MetricMeter()
    def __call__(self, preds, labels, category=None):
        """
        Args:
            preds: after log_softmax
        """
        preds_choice = preds.max(1)[1]
        instance_correct = preds_choice.eq(labels.data).cpu().sum().item()
        instance_accurcy = instance_correct / preds.shape[0]
        self.meter.update({'instance_accurcy': instance_accurcy})

        if category:
            for cat in np.unique(labels.cpu()):
                cat = cat.item()
                class_correct = preds_choice[labels == cat].eq(labels[labels == cat].long().data).cpu().sum().item()
                class_accurcy = class_correct / float(preds[labels == cat].shape[0])
                self.meter.update({f'class_{cat}_accurcy': class_accurcy})
            
            


        