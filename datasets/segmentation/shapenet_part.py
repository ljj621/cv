import os
import torch
import numpy as np
from jlcv.io import FileIO
from .. import DATASETS
from ..base_dataset import BaseDataset
from jlcv.instances import Points

seg_classes = {
    'Earphone': [16, 17, 18], 
    'Motorbike': [30, 31, 32, 33, 34, 35], 
    'Rocket': [41, 42, 43],
    'Car': [8, 9, 10, 11], 
    'Laptop': [28, 29], 
    'Cap': [6, 7], 
    'Skateboard': [44, 45, 46],
    'Mug': [36, 37], 
    'Guitar': [19, 20, 21], 
    'Bag': [4, 5], 
    'Lamp': [24, 25, 26, 27],
    'Table': [47, 48, 49], 
    'Airplane': [0, 1, 2, 3], 
    'Pistol': [38, 39, 40],
    'Chair': [12, 13, 14, 15], 
    'Knife': [22, 23]}

@DATASETS.register_module()
class ShapeNetPartDataset(BaseDataset):
    def __init__(self, 
                 root, 
                 split_file, 
                 classes_file=None, 
                 select_classes=None, 
                 pipelines=None):
        super().__init__(root, split_file, classes_file, select_classes, pipelines)
        self.num_class = 50

    def get_classes(self, classes_file='synsetoffset2category.txt', select_classes=None):
        class_info = {}
        pid_to_name = {}
        contents = FileIO.load(os.path.join(self.root, classes_file))
        for c in contents:
            c = c.split('\t')
            name, index = c
            tmp_pid = seg_classes[name]
            pid_to_name.update(dict(zip(tmp_pid, [name]*len(tmp_pid))))
            
        class_info['pid_to_name'] = pid_to_name
        return class_info
    
    def get_data_infos(self, split_file):
        file_list = FileIO.load(os.path.join(self.root, split_file))
        data_infos = []
        for f in file_list:
            data = FileIO.load(os.path.join(self.root, f))
            keys = list(data.keys()) # data, label, pid
            num = len(data[keys[0]])
            for i in range(num):
                sample = {}
                sample['data'] = data['data'][i]
                sample['label'] = data['label'][i,...][0]
                sample['seg_labels'] = data['pid'][i,...]
                # sample['class_name'] = self.classes[sample['label']]['class_name']
                # sample['seg_classes'] = self.classes[sample['label']]['seg_classes']
                data_infos.append(sample)
        return data_infos
    
    def __getitem__(self, index):
        info = self.data_infos[index]
        points = Points(**info)
        points.transform(self.pipelines)
        data_dict = {}
        data_dict['points'] = points
        return data_dict
    
    def eval(self, preds_logits, seg_labels, meters):
        preds_logits = preds_logits.cpu().data.numpy() # b, n, c
        seg_labels = seg_labels.long().cpu().data.numpy() # b, n
        
        # preds_choice = np.argmax(preds_logits, -1)
        batch_size, num_points = seg_labels.shape

        pid_to_name = self.classes['pid_to_name']
        cur_pred_val = np.zeros((batch_size, num_points)).astype(np.int32)
        
        for i in range(batch_size):
                cat = pid_to_name[seg_labels[i, 0]]
                logits = preds_logits[i, :, :]
                a = np.argmax(logits[:, seg_classes[cat]], 1) + seg_classes[cat][0]
                cur_pred_val[i, :] = np.argmax(logits[:, seg_classes[cat]], 1) + seg_classes[cat][0]
        
        meters.update({'eval_point_accuracy': {
                'val': np.sum(cur_pred_val == seg_labels),
                'count': batch_size*num_points
            }
        })
        results = "Eval point accuracy: {0:.6f}\n".format(meters.eval_point_accuracy.avg)
        
        # for i in range(self.num_class):
        #     meters.update({f'class_{i}_accuracy': {
        #         'val': np.sum((preds_choice == i) & (seg_labels == i)),
        #         'count': np.sum(seg_labels == i) + 1e-6
        #     }})
        # class_accuracy = np.array([getattr(meters, f'class_{i}_accuracy').avg for i in range(0, self.num_class)])
        # class_accuracy = np.mean(class_accuracy)
        # results += "Eval point avg class accuracy: {0:.6f}\n".format(class_accuracy)
    
        for i in range(batch_size):
            segp = cur_pred_val[i, :]
            segl = seg_labels[i, :]
            cat = self.classes['pid_to_name'][segl[0]]
            part_ious = [0.0 for _ in range(len(seg_classes[cat]))]
            
            for l in seg_classes[cat]:
             
                if (np.sum(segl == l) == 0) and (np.sum(segp == l) == 0):  
                    # part is not present, no prediction as well
                    part_ious[l - seg_classes[cat][0]] = 1.0
                else:
                    part_ious[l - seg_classes[cat][0]] = np.sum((segl == l) & (segp == l)) / float(np.sum((segl == l) | (segp == l)))
            # meters.update({f'class_{l}_IoU': np.mean(part_ious)})
            meters.update({'ins_iou': np.mean(part_ious)})
        
        # class_IoU = np.array([getattr(meters, f'class_{i}_IoU').avg for i in range(0, self.num_class)])
        # class_IoU = np.mean(class_IoU)
        # results += "Eval point avg class IoU: {0:.6f}\n".format(class_IoU)
        instance_IoU = getattr(meters,'ins_iou').avg
        results += "Eval point avg instance IoU: {0:.6f}\n".format(instance_IoU)
        
        # TODO 需要修改
        # results += "Eval point class:\n"
        # results += "{0:^5} {1:^5} {2:^5}\n".format("Class", "Accuracy", "IoU")
        # for i in range(self.num_class):
        #     accuracy = getattr(meters, f'class_{i}_accuracy').avg
        #     IoU = getattr(meters, f'class_{i}_IoU').avg
        #     results += "{0:^5} {1:^5} {2:^5}\n".format(i, accuracy, IoU)

        return results, instance_IoU


    
    


