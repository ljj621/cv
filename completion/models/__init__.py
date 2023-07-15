import torch
import torch.nn as nn
from jlcv.registry import Registry, build_from_cfg
COMPLETION = Registry('completion')

from .mymodel import MyModel

class Completion:
    @classmethod
    def build(cls, cfg):
        model= build_from_cfg(cfg.model, COMPLETION)
        
        if cfg.has('pretrain'):
            cls.pretrained(model, cfg.pretrain)
        
        if cfg.has('resume'):
            cls.resumed(model, cfg.resume)
        return model
            
    @classmethod
    def pretrained(cls, model, pretrain):
        state = torch.load(pretrain)
        pretrained_state = state['model_state']
        model_state = model.state_dict()
        
        for k, v in model_state.items():
            if k in pretrained_state:
                if v.shape == pretrained_state[k].shape:
                    model_state[k] = pretrained_state[k]

        model.load_state_dict(model_state)
        model.best_metrics = state['best_metrics']*1000
        print('load pretrained ckpt')
        
    @classmethod
    def resumed(cls, model, resume):
        state = torch.load(resume)
        pretrained_state = state['model_state']
        model.load_state_dict(pretrained_state)
        model.best_metric = state['best_metric']
        print('load resumed ckpt')


__all__ = [
    'MyModel',  'Completion', 'COMPLETION'
]
    