import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import numpy as np
from jlcv.models import MODELS
from jlcv.modules.modules import MODULES
from jlcv.modules.base import build_conv_layer, build_linear_layer
from jlcv.metrics import METRICS
from jlcv.modules.ext import knn, group_points, three_nn, three_interpolate
from .base_model import BaseModel
from . import SEGMENTATION

@SEGMENTATION.register_module()
class MyModel(BaseModel):
    def __init__(self, 
                 data_preprocessor,
                 num_class, 
                 backbone,
                 seg_head,
                 loss_layer, 
                 conv='Conv1d',
                 norm={'type': 'BN', 'momentum': 0.01}, 
                 act='GELU', 
                 checkpoints=None,
                 resume=False
        ):
        super().__init__()
        self.num_class = num_class
        self.data_preprocessor = MODELS.build(data_preprocessor)
        ###################  query backbone  #################
        self.num_queries = backbone['query_generator']['num_queries']
        self.num_stage = backbone['query_encoder']['num_stage']
        self.backbone = MODELS.build(backbone)
        
        ###################  query prediction  #################
        channels = backbone['middle_encoder']['channels']
   
        self.seg_head = MODELS.build(seg_head, channels=channels*self.num_stage)
        
        self.loss_layer = METRICS.build(loss_layer)
        self.load_checkpoints(checkpoints, resume)

    def forward(self, input_dict):
        data_dict = self.data_preprocessor(input_dict)
        
        results = self.backbone(data_dict)
        results['original_points'] = data_dict['original_points']
        results['points'] = data_dict['points']
        results['points_index'] = data_dict['index']
        
        preds_dict = self.seg_head(results)
        preds_dict['heatmap_preds'] = results['heatmap_preds']
        return preds_dict
    
    def get_loss(self, preds_dict, seg_labels, labels, weights=None):
        loss_dict = {}
        
        seg_labels = seg_labels.long() # b, n
        if weights is not None:
            weights = torch.from_numpy(weights).float().cuda()
            
        total_loss = 0
        for k, p in preds_dict.items():
            if k == 'heatmap_preds':
                loss = self.loss_layer(p,labels.long(), weights)
                loss_dict[k] = loss.item()
            elif k == 'preds':
                loss = self.loss_layer(p['preds'].view(-1, self.num_class), seg_labels.view(-1), weights)
                loss_dict[k] = loss.item()
            else:
                preds, index = p['preds'], p['index']
                selected_labels = seg_labels.gather(1, index.long())
                loss = self.loss_layer(preds.view(-1, self.num_class), selected_labels.view(-1), weights)
                loss_dict[k] = loss.item()
                
            total_loss = total_loss + loss

        total_loss.backward()
        loss_dict['total_loss'] = total_loss.item()
        return loss_dict
