import torch
import torch.nn as nn
from .registry import MODELS

@MODELS.register_module()
class MyModel(nn.Module):
    def __init__(self,
                 backbone=None) -> None:
        super().__init__()



    def forward(self, imgs, pix):
        b, c, h, w = imgs.shape





        