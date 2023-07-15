from torch.utils.data import Dataset
from . import DATASETS

@DATASETS.register_module()
class BaseDataset(Dataset):
    def __init__(self, 
                 root, 
                 split_file, 
                 classes_file=None,
                 select_classes=None,
                 pipelines=None,
                 **kwargs):
        super().__init__()
        self.root = root
        self.split_file = split_file
        for k, v in kwargs.items():
            self.__setattr__(k, v)
        self.classes = self.get_classes(classes_file, select_classes)
        self.data_infos = self.get_data_infos(split_file)
        self.pipelines = pipelines

    def get_data_infos(self, split_file, **kwargs):
        pass
    
    def get_classes(self, classes_file, select_classes=None):
        pass
    
    def __len__(self):
        return len(self.data_infos)
    
    def __getitem__(self, index):
        pass











        







            






    


