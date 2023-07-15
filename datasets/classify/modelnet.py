import os
import numpy as np
from jlcv.io import FileIO

from datasets import DATASETS
from datasets.base_dataset import BaseDataset
from jlcv.instances import Points

@DATASETS.register_module()
class ModelNetDataset(BaseDataset):
    def __init__(self, 
                 root, 
                 split_file, 
                 classes_file=None, 
                 select_classes=None, 
                 pipelines=None):
        super().__init__(root, split_file, classes_file, select_classes, pipelines)
    
    def get_data_infos(self, split_file):
        h5_files = FileIO.load(os.path.join(self.root, split_file))
        data_infos = []
        data, label = [], []
        for h5_file in h5_files:
            file = FileIO.load(os.path.join(self.root, h5_file))
            data.append(file['data'])
            label.append(file['label'])
        data = np.concatenate(data, 0)
        label = np.concatenate(label, 0)

        data_infos = []
        for i in range(len(data)):
            data_infos.append({
                'data': data[i], 
                'label': label[i][0], 
                'class_name': self.classes[label[i][0]]
            })
        return data_infos
    
    def get_classes(self, classes_file, select_classes=None):
        classes_name = FileIO.load(os.path.join(self.root, classes_file))
        classes_index = np.arange(len(classes_name))

        if select_classes is not None:
            if isinstance(select_classes, str):
                select_classes = [select_classes]
            index = [i for i, name in enumerate(classes_name) if name in select_classes]
            select_index = [classes_index[i] for i in index]
            index_to_name = dict(zip(select_index, select_classes))
        else:
            index_to_name = dict(zip(classes_index, classes_name))

        return index_to_name

    def __getitem__(self, index):
        info = self.data_infos[index]
        points = Points(**info)
        points.transform(self.pipelines)
        data_dict = {}
        data_dict['points'] = points
        return data_dict
    
    def eval(self, preds_logits, labels, meters):
        preds_choice = preds_logits.max(1)[1]
        
        instance_correct = preds_choice.eq(labels.data).cpu().sum().item()
        instance_accurcy = instance_correct / preds_logits.shape[0]
        meters.update({'instance_accurcy': instance_accurcy})
       
        results = "Eval instance accurcy: {:.6f}\n".format(meters.instance_accurcy.avg)

        return results, meters.instance_accurcy.avg


    
    

