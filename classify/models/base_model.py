import torch
import torch.nn as nn
import numpy as np
from jlcv.logger import Logger

class BaseModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.get_logger()
        
    def get_logger(self):
        self.logger = Logger.get("main.model")
        self.logger.info(f'Building model: {self.__class__.__name__}')


    def load_checkpoints(self, checkpoints, resume=False):
        if checkpoints is None:
            self.apply(self.__weights_init_normal)
            self.logger.info(f"No existing model, starting training from scratch...")
            metrics = 0
            self.set('metric', metrics)
        else:
            state = torch.load(checkpoints)
            pretrained_state = state['model_state']
            if not resume:
                model_state = self.state_dict()
                for k, v in model_state.items():
                    if k in pretrained_state:
                        if v.shape == pretrained_state[k].shape:
                            model_state[k] = pretrained_state[k]
                self.load_state_dict(model_state)
            else:
                self.load_state_dict(pretrained_state)
            # metrics = 0.5
            metrics = state['metrics']
            self.set('metric', metrics)
            self.logger.info(f"Successfully load checkpoint at '{checkpoints}'\n")
            self.logger.info(f'Current best loss {metrics}\n')
    
    def save_model(self, ckpt_path, metrics):
        state = {
                'model_state': self.state_dict(),
                'metrics': metrics
                }
        torch.save(state, ckpt_path)
    
    def set(self, *args):
        if len(args) == 1:
            assert isinstance(args[0], dict)
            for k, v in args[0].items():
                if not self.has(k):
                    setattr(self, k, v)
        elif len(args) == 2:
            assert isinstance(args[0], str)
            if not self.has(args[0]):
                setattr(self, args[0], args[1])

    def has(self, name):
        return hasattr(self, name)
        
    @staticmethod
    def __weights_init_normal(m):
        classname = m.__class__.__name__
        if classname.find("Conv2d") != -1:
            torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif classname.find("Conv1d") != -1:
            torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif classname.find("BatchNorm2d") != -1:
            torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
            torch.nn.init.constant_(m.bias.data, 0.0)
        elif classname.find("BatchNorm1d") != -1:
            torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
            torch.nn.init.constant_(m.bias.data, 0.0)
    