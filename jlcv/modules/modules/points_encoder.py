import torch
import torch.nn as nn
import torch.nn.functional as F
from ..base import build_conv_layer, build_linear_layer
from . import MODULES
from ...models import MODELS

@MODULES.register_module()
class PointsEncoder(nn.Module):
    def __init__(self,
                 backbone: str,
                 config: dict) -> None:
        super().__init__()
        self.encoder = MODELS.build(backbone, **config)

    def forward(self, points):
        return self.encoder(points)













