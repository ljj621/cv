import torch
import numpy as np

class Instance(object):
    def __init__(self, **kwargs):
        super().__init__()
        self.properties = [] 
        for k,v in kwargs.items():
            self.set_property(k, v)

    def transform(self, pipelines=None):
        if pipelines is None: return
       
        for name, args in pipelines.items():
            assert hasattr(self, name), f'No member function {name} is available!'
            if args is None:
                getattr(self, name)()
            else:
                getattr(self, name)(args)
    
    def to(self, device):
        for p in self.properties:
            assert hasattr(self, p)
            if isinstance(getattr(self, p), torch.Tensor):
                setattr(self, p, getattr(self, p).to(device))
    
    def is_numpy(self, data):
        return isinstance(data, np.ndarray)
    
    def to_tensor(self, data):
        if isinstance(data, np.ndarray):
            data = torch.as_tensor(data, dtype=torch.float32)
        return data

    @property
    def shape(self):
        return self.data.shape

    @property
    def device(self):
        return self.data.device
    
    def batch_free(self):
        return len(self.shape) == 2

    def set_property(self, name, p):
        if p is not None:
            self.__setattr__(name, p)
            self.properties.append(name)
    
    def get_property(self, name):
        assert self.has_property(name), f'No member {name} is available!'
        return getattr(self, name)

    def has_property(self, name):
        return hasattr(self, name)
    
    def to_dict(self, name=None):
        output_dict = {}
        for p in self.properties:
            p_name = name + '_' + p if name is not None else p
            output_dict[p_name] = self.get_property(p)
        return output_dict
    
    def clone(self):
        ori_type = type(self)
        output_dict = self.to_dict()
        return ori_type(**output_dict)




    
