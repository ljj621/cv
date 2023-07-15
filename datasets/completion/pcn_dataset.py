import os
import numpy as np
from jlcv.io import FileIO
from .. import DATASETS
from ..base_dataset import BaseDataset
from datasets.instances import Points


@DATASETS.register_module()
class PCNDataset(BaseDataset):
    def __init__(self, 
                 root, 
                 split_file, 
                 classes_file=None, 
                 select_classes=None, 
                 pipeline=None, 
                 num_rander=8):
        super().__init__(root, split_file, classes_file, select_classes, pipeline)
        self.num_rander = num_rander
    
    def get_data_infos(self, split_file):
        data_infos = []
        dir_path = os.path.join(self.root, split_file.split('.')[0])
        infos = FileIO.load(os.path.join(self.root, split_file))

        for info in infos:
            classes_index = info.split('/')[0]
            if classes_index in self.classes:
                classes_name = self.classes[classes_index]
                if not self.test_mode:
                    gt_path = f'{dir_path}/gt/{info}.pcd'
                    for i in range(self.num_rander):
                        points_path = f'{dir_path}/partial/{info}/0{i}.pcd'
                        data_infos.append(dict(
                            classes_name=classes_name, classes_index=classes_index, gt_path=gt_path, points_path=points_path))
                else:
                    points_path = f'{dir_path}/partial/{info}/00.pcd' 
                    data_infos.append(dict(
                        classes_name=classes_name, classes_index=classes_index, points_path=points_path))
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
        points = Points(info['data'], info['label'])
        input_dict = dict(
            classes_name=info['classes_name'],
            classes_index=info['classes_index'],
            gt=info['gt_path'],
            points=info['points_path'],
            points_fields=['points', 'gt']
        )
        if self.pipelines is not None:
            input_dict = self.pipelines(input_dict)
        return input_dict



        

