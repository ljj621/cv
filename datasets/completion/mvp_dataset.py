import torch
from .registry import DATASETS
from ..base_dataset import BaseDataset
import numpy as np
import open3d as o3d

@DATASETS.register_module()
class MVPDataset(BaseDataset):
    def __init__(self, 
                 root, 
                 split, 
                 classes='synsetoffset2category.txt', 
                 select_classes=None, 
                 pipeline=None, 
                 file_type='.pcd', 
                 test_mode=False):
        super().__init__(root, split, classes, select_classes, pipeline, file_type, test_mode)
    
    def get_points(self, file_path):
        pcd = o3d.io.read_point_cloud(file_path)
        points = np.array(pcd.points)
        return points



        

