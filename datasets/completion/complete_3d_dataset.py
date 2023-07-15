import os
import numpy as np
from jlcv.io import FileIO

from datasets import DATASETS
from datasets.base_dataset import BaseDataset
from jlcv.instances import Points
# import transforms3d


@DATASETS.register_module()
class Complete3DDataset(BaseDataset):
    def __init__(self, 
                 root, 
                 split_file, 
                 classes_file='synsetoffset2category.txt', 
                 select_classes=None, 
                 pipelines=None, 
                 **kwargs):
        super().__init__(root, split_file, classes_file, select_classes, pipelines, **kwargs)

    
    def get_classes(self, classes_file, select_classes=None):
        index_to_infos = {}
        contents = FileIO.load(os.path.join(self.root, classes_file))
        for c in contents:
            name, index, label = c
            index_to_infos[index] = {'class_name': name, 'label': int(label)}
        return index_to_infos
    
    def get_data_infos(self, split_file, **kwargs):
        data_infos = []

        files_list = FileIO.load(os.path.join(self.root, split_file))
        split = split_file.split('.')[0]
        
        for f in files_list:
            sample = {}
            index = f.split('/')[0]
            infos = self.classes[index]
            label = infos['label']
            class_name = infos['class_name']
            sample['label'] = label
            sample['class_name'] = class_name
            
            # partial = FileIO.load(os.path.join(self.root, split, 'partial', f + '.h5'))
            # sample['partial'] = np.array(partial['data'])
            sample['partial_path'] = os.path.join(self.root, split, 'partial', f + '.h5')

            if split == 'train' or split == 'val':
                sample['gt_path'] = os.path.join(self.root, split, 'gt', f + '.h5')

                # gt = FileIO.load(os.path.join(self.root, split, 'gt', f + '.h5'))
                # sample['gt'] = np.array(gt['data'])

            data_infos.append(sample)
        return data_infos

    def __getitem__(self, index):
        info = self.data_infos[index]
        partial = FileIO.load(info['partial_path'])['data']
        gt = FileIO.load(info['gt_path'])['data']

        if 'train' in self.split_file:
            choice = np.random.randint(0, 2048, (2048))
            partial = partial[choice]

            trfm_mat = transforms3d.zooms.zfdir2mat(1)
            trfm_mat_x = np.dot(transforms3d.zooms.zfdir2mat(-1, [1, 0, 0]), trfm_mat)
            trfm_mat_z = np.dot(transforms3d.zooms.zfdir2mat(-1, [0, 0, 1]), trfm_mat)
            rnd_value = np.random.uniform(0, 1)
            if rnd_value <= 0.25:
                trfm_mat = np.dot(trfm_mat_x, trfm_mat)
                trfm_mat = np.dot(trfm_mat_z, trfm_mat)
            elif rnd_value > 0.25 and rnd_value <= 0.5:    # lgtm [py/redundant-comparison]
                trfm_mat = np.dot(trfm_mat_x, trfm_mat)
            elif rnd_value > 0.5 and rnd_value <= 0.75:
                trfm_mat = np.dot(trfm_mat_z, trfm_mat)
            # trfm_mat = torch.from_numpy(trfm_mat).float()
            partial = partial @ trfm_mat.T
            gt = gt @ trfm_mat.T
        info['partial'] = partial
        info['gt'] = gt

        points = Points(**info)
        points.transform(self.pipelines)
        return points

