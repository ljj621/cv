from torch.utils.data import DataLoader 
from functools import partial
from torch.utils.data.dataloader import default_collate
from . import DATASETS
from jlcv.instances import Instance

class DataContainer:
    def __init__(self, dataset_cfg, dataloader_cfg) -> None:

        self.dataset = DATASETS.build(dataset_cfg)
        self.dataloader = DataLoader(self.dataset, 
                          **dataloader_cfg,
                          collate_fn=partial(self.collate))

    @staticmethod
    def collate_data(batch, name):
        stacked = {}
        points_list = [b[name] for b in batch]
        t = type(points_list[0])
        properties = points_list[0].properties
        for p in properties:
            data = [getattr(sample, p) for sample in points_list]
            if isinstance(data[0], list):
                stacked[p]=data
            elif isinstance(data[0], dict):
                stacked[p]=data
            else:
                stacked[p]=default_collate(data)
        stacked = t(**stacked)  
        return stacked
    
    @staticmethod
    def collate_metas(batch):
        return batch[0]['metas']

    @classmethod
    def collate(cls, batch):
        assert isinstance(batch[0], dict)
        data_dict = {}
        if 'points' in batch[0]:
            data_dict['points'] = cls.collate_data(batch, 'points')
        if 'metas' in batch[0]:
            data_dict['metas'] = cls.collate_metas(batch) 
        return data_dict

    def eval(self, input, labels, meters):
        return self.dataset.eval(input, labels, meters)
