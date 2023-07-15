from .io import FileIO
import os
from easydict import EasyDict as edict
import time

class Config:
    def __init__(self, cfg):
        if isinstance(cfg, str):  
            self.__load_from_file(cfg)
            
        elif isinstance(cfg, dict):
            self.set(cfg)
    
        work_dir = os.path.join('work_dir', self.dataset_type, self.model.type)
        if self.has('version'):
            work_dir = work_dir + f'/{self.version}'
        self.set({"work_dir": work_dir})
        os.makedirs(work_dir, exist_ok=True)
    
    def dump(self, dump_path):
        FileIO.dump(self.data, dump_path)
    
    def __load_from_file(self, cfg_file):
        assert os.path.exists(cfg_file), f'File is not exist: "{cfg_file}"'
        cfg = FileIO.load(cfg_file)
        dataset = cfg.get('dataset', None)
        if isinstance(dataset, str):
            dataset = cfg.pop('dataset')
            dataset = FileIO.load(dataset)
            cfg.update(dataset)
        
        self.set(edict(cfg))
    
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
    

