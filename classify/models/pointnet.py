import torch.nn as nn
import torch.nn.functional as F
from .base_model import BaseModel
from jlcv.modules import build_module, build_conv_layer, build_linear_layer
from jlcv.metrics import build_metric_layer
from . import CLASSIFIER

@CLASSIFIER.register_module()
class PointNet(BaseModel):
    def __init__(self, 
                 num_class, 
                 channels, 
                 backbone, 
                 loss_layer, 
                 norm_cfg='BN', 
                 act_cfg='ReLU', 
                 pretrained=None,train_meter=None,
                 test_meter=None
                 ):
        super().__init__(num_class,train_meter=train_meter,test_meter=test_meter)
        self.backbone = build_module(backbone)
        self.classifier = nn.Sequential(
            build_linear_layer(channels[-1], 512, norm_cfg=norm_cfg, act_cfg=act_cfg),
            build_linear_layer(512, 256, norm_cfg=norm_cfg, act_cfg=act_cfg),
            nn.Dropout(p=0.4),
            build_linear_layer(256, num_class)
        )

        self.loss_layer = build_metric_layer(loss_layer)
        self.init_weights(pretrained)

    def forward(self, batch):
        batch.fps(1024)
        points = batch.data.permute(0,2,1)
        _, global_feats = self.backbone(points)
        preds = self.classifier(global_feats)
        preds = F.log_softmax(preds, dim=1)
        return preds
