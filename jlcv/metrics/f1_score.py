from . import METRICS
import numpy as np

@METRICS.register_module()
class F1Score(object):
    def __init__(self, threshold=0.0001) -> None:
        self.threshold = threshold
    def __call__(self, dist1, dist2):
        precision_1 = np.mean((dist1 < self.threshold), 1)
        precision_2 = np.mean((dist2 < self.threshold), 1)
        fscore = 2 * precision_1 * precision_2 / (precision_1 + precision_2)
        fscore[np.isnan(fscore)] = 0
        return fscore, precision_1, precision_2