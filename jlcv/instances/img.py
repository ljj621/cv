import torch
import numpy as np
import copy
import cv2
from jlcv.io import FileIO
from .instance import Instance

class Img(Instance):
    direction = {'x': 0, 'y': 1, 'xy':-1}
    def __init__(self, data, label=None):
        super().__init__(data)
        if label is not None:
            label = self.to_tensor(label).long()
            self.properties.append('label')
        self.label = label
    
    def load_file(self, filename):
        img = FileIO.load(filename, self.color_type)
        img = img.astype(np.float32)
        return copy.deepcopy(img)
    
    def flip(self, direction='x'):
        assert direction in self.direction.keys()
        self.add_meta('flip', direction)

        if isinstance(direction, str):
            direction = [self.direction[direction], -1]
        flip_ratio = [1 / (len(direction))] * len(direction)
        direction = np.random.choice(direction, p=flip_ratio)
        self.data = cv2.flip(self.data, direction)
    
    def resize(self, scale):
        h, w = self.data.shape[:2]
        # step 1. get scale_factor
        if isinstance(scale[0], int):
            new_h, new_w = scale
        elif isinstance(scale[0], float):
            new_h = int(h * float(scale[0]) + 0.5)
            new_w = int(w * float(scale[1]) + 0.5)
        else:
            img_scale_long = [max(s) for s in scale]
            img_scale_short = [min(s) for s in scale]
            long_edge = np.random.randint(
                min(img_scale_long),
                max(img_scale_long) + 1)
            short_edge = np.random.randint(
                min(img_scale_short),
                max(img_scale_short) + 1)
            long_edge = max([long_edge, short_edge])
            short_edge = min([long_edge, short_edge])
            scale_factor = min(long_edge / max(h, w),
                            short_edge / min(h, w))
            new_h = int(h * float(scale_factor) + 0.5)
            new_w = int(w * float(scale_factor) + 0.5)
            h_scale = new_h / h
            w_scale = new_w / w
        self.add_meta('resize', [h_scale, w_scale])
        
        self.data = cv2.resize(self.data, [new_w, new_h], interpolation=cv2.INTER_LINEAR)
    
    def crop(self, crop_scale=[512, 256], is_random=False):
        tw, th = crop_scale
        
        if isinstance(self.data, dict):
            for i, k, v in enumerate(self.data.items()):
                if i == 0:
                    w, h = self.data.size
                    if is_random:
                        w = np.random.randint(0, w - tw)
                        h = np.random.randint(0, h - th)
                    else:
                        w = w - tw
                        h = h - th
                
                self.data[k] = self.data[k][h: h + th, w: w + tw]
    
    # def normalize(self, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):

    def normalize(self):
        self.data = self.data / 255.
        



                

                

        
        





        
        







        


