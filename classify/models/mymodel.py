import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import numpy as np
from jlcv.models import MODELS
from jlcv.modules.modules import MODULES
from jlcv.modules.base import build_conv_layer, build_linear_layer
from jlcv.metrics import METRICS
from .base_model import BaseModel
from . import CLASSIFIER

@CLASSIFIER.register_module()
class MyModel(BaseModel):
    def __init__(self, 
                 data_preprocessor,
                 num_class, 
                 backbone,
                 loss_layer, 
                 conv='Conv1d',
                 norm={'type': 'BN', 'momentum': 0.01}, 
                 act='GELU', 
                 checkpoints=None,
                 resume=False):
        super().__init__()
        self.num_class = num_class
        self.data_preprocessor = MODELS.build(data_preprocessor)
        ###################  query backbone  #################
        self.num_queries = backbone['query_generator']['num_queries']
        self.num_stage = backbone['query_encoder']['num_stage']
        self.backbone = MODELS.build(backbone)
        
        ###################  query prediction  #################
        channels = backbone['middle_encoder']['channels']
   
        self.classifier = nn.Sequential(
                build_linear_layer(channels*self.num_stage, 512, norm=norm, act=act),
                nn.Dropout(0.4),
                build_linear_layer(512, 256, norm=norm, act=act),
                nn.Dropout(0.4),
                build_linear_layer(256, num_class)
            )
        
        self.loss_layer = METRICS.build(loss_layer)
        self.load_checkpoints(checkpoints, resume)

    def forward(self, input_dict):
        data_dict = self.data_preprocessor(input_dict)
        
        results = self.backbone(data_dict)
        query_feats = results['query_feats']
        points_feats = results['points_feats']
        heatmap_preds = results['heatmap_preds']
        b, c, n = query_feats.shape
        query_feats = query_feats.mean(-1)
        points_feats = points_feats.max(-1)[0]
        preds = self.classifier(query_feats + points_feats)
        preds_dict = {}
        preds_dict['preds'] = preds
        preds_dict['preds_logits'] = F.log_softmax(preds,1)
        preds_dict['heatmap_preds'] = heatmap_preds
        return preds_dict
    
    def get_loss(self, preds_dict, labels):
        total_loss = 0
        loss_dict = {}
        for k, v in preds_dict.items():
            if k == 'preds_logits': continue
            loss = self.loss_layer(v, labels)
            total_loss = total_loss + loss
            loss_dict[k+'_loss'] = loss.item()

        total_loss.backward()
        loss_dict['total_loss'] = total_loss.item()
        return loss_dict

