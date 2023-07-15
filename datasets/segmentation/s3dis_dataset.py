import os
import numpy as np
from tqdm import tqdm
import torch
from jlcv.io import FileIO
from .. import DATASETS
from jlcv.instances import Points
from datasets.base_dataset import BaseDataset

@DATASETS.register_module()
class S3DISDataset(BaseDataset):
    def __init__(self, 
                 root='/home/lj/MyDisk/DATASET/s3dis', 
                 split_file='K-Cross-Validation_1_train.list',
                 classes_file=None,
                 select_classes=None,
                 pipelines=None,
                 num_point=4096,
                 block_size=1.0, 
                 sample_rate=1.0):
        super().__init__(root=root, split_file=split_file, classes_file=classes_file, select_classes=select_classes,pipelines=pipelines, num_point=num_point, block_size=block_size, sample_rate=sample_rate)

        self.num_class = 13
    
    
    def get_data_infos(self, split_file):
        data_infos = {}
        rooms = FileIO.load(os.path.join(self.root, split_file))
        
        label_weights = np.zeros(13)
        num_point_all = 0

        room_samples = {}
        for room_name in tqdm(rooms, total=len(rooms)):
            room_path = os.path.join(self.root, 'stanford_indoor3d', room_name)

            room_data = FileIO.load(room_path)
            data, seg_labels = room_data[:, 0:6], room_data[:, 6]  # xyzrgb, N*6; l, N
            tmp, _ = np.histogram(seg_labels, range(14))
            label_weights += tmp

            # coord_min, coord_max = np.amin(data, axis=0)[:3], np.amax(data, axis=0)[:3]
            room_name = room_name.split('.')[0]
            sample = {
                # 'room_name': room_name,
                'data': data,
                'seg_labels': seg_labels,
                'num_point': seg_labels.size,
                # 'coord_min': coord_min,
                # 'coord_max': coord_max,
            }
            room_samples[room_name] = sample
            num_point_all += seg_labels.size

        data_infos['room_samples'] = room_samples

        if self.sample_rate is not None:
            num_iter = int(num_point_all * self.sample_rate / self.num_point)
            room_idxs = []

            for room_name, sample in room_samples.items():
                sample_prob = sample['num_point'] / num_point_all
                room_idxs.extend([room_name] * int(round(sample_prob * num_iter)))

            data_infos['room_idxs'] = room_idxs
            data_infos['num_samples'] = len(room_idxs)
        else:
            data_infos['num_samples'] = len(room_samples)

        label_weights = label_weights.astype(np.float32)
        label_weights = label_weights / np.sum(label_weights)
        label_weights = np.power(np.amax(label_weights) / label_weights, 1 / 3.0)
        print(label_weights)
        data_infos['weights'] = np.array(label_weights)

        print("Totally {} samples in {}.".format(data_infos['num_samples'], split_file.split('.')[0]))
        return data_infos
    
    def __len__(self):
        return self.data_infos['num_samples']

    def __getitem__(self, index):
        room_idxs = self.data_infos['room_idxs']
        room_samples = self.data_infos['room_samples']
        
        room_name = room_idxs[index]
        sample = room_samples[room_name]
        data = sample['data']  # N * 6
        seg_labels = sample['seg_labels']   # N
        num_point = sample['num_point']
        # room_coord_max = sample['coord_max']

        center = data[np.random.choice(num_point)][:3]
        block_min = center - [self.block_size / 2.0, self.block_size / 2.0, 0]
        block_max = center + [self.block_size / 2.0, self.block_size / 2.0, 0]
        point_idxs = np.where((data[:, 0] >= block_min[0]) & (data[:, 0] <= block_max[0]) & (data[:, 1] >= block_min[1]) & (data[:, 1] <= block_max[1]))[0]

        selected_point_idxs = np.random.choice(point_idxs, self.num_point)
        current_labels = seg_labels[selected_point_idxs]

        # normalize
        selected_data = data[selected_point_idxs, :]  # num_point * 6
        points = selected_data[..., :3]
        colors = selected_data[..., 3:]
        
        points_min = np.amin(points, axis=0)[:3]
        points_max = np.amax(points, axis=0)[:3]
        points_norm = (points - points_min) / (points_max - points_min)
        
        current_data = np.zeros_like(selected_data)  # num_point * 6
        current_data[...,:3] = points_norm
        current_data[...,3:] = colors / 255.

        data_dict = {}
        points = Points(data=current_data, seg_labels=current_labels)
        points.transform(self.pipelines)
        data_dict['points'] = points
        data_dict['metas'] = {'weights': self.data_infos['weights']}
        return data_dict
    
    def get_single_scene(self, sample):
        points = sample['data']  # N * 6
        seg_labels = sample['seg_labels']   # N
        num_point = sample['num_point']
        coord_max = sample['coord_max']
        coord_min = sample['coord_min']
        labelweights = self.data_infos['weights'] 

        grid_x = int(np.ceil(float(coord_max[0] - coord_min[0] - self.block_size) / 0.5) + 1)
        grid_y = int(np.ceil(float(coord_max[1] - coord_min[1] - self.block_size) / 0.5) + 1)

        data_room, label_room, sample_weight, index_room = np.array([]), np.array([]), np.array([]),  np.array([])

        for index_y in range(0, grid_y):
            for index_x in range(0, grid_x):
                s_x = coord_min[0] + index_x * 0.5
                e_x = min(s_x + self.block_size, coord_max[0])
                s_x = e_x - self.block_size
                s_y = coord_min[1] + index_y * 0.5
                e_y = min(s_y + self.block_size, coord_max[1])
                s_y = e_y - self.block_size
                point_idxs = np.where(
                    (points[:, 0] >= s_x - 0.001) & (points[:, 0] <= e_x + 0.001) & (points[:, 1] >= s_y - 0.001) & (
                                points[:, 1] <= e_y + 0.001))[0]
                if point_idxs.size == 0:
                    continue
                num_batch = int(np.ceil(point_idxs.size / self.num_point))
                point_size = int(num_batch * self.num_point)
                replace = False if (point_size - point_idxs.size <= point_idxs.size) else True
                point_idxs_repeat = np.random.choice(point_idxs, point_size - point_idxs.size, replace=replace)
                point_idxs = np.concatenate((point_idxs, point_idxs_repeat))
                np.random.shuffle(point_idxs)
                data_batch = points[point_idxs, :]
                normlized_xyz = np.zeros((point_size, 3))
                normlized_xyz[:, 0] = data_batch[:, 0] / coord_max[0]
                normlized_xyz[:, 1] = data_batch[:, 1] / coord_max[1]
                normlized_xyz[:, 2] = data_batch[:, 2] / coord_max[2]
                data_batch[:, 0] = data_batch[:, 0] - (s_x + self.block_size / 2.0)
                data_batch[:, 1] = data_batch[:, 1] - (s_y + self.block_size / 2.0)
                data_batch[:, 3:6] /= 255.0
                data_batch = np.concatenate((data_batch, normlized_xyz), axis=1)
                label_batch = seg_labels[point_idxs].astype(int)
                batch_weight = labelweights[label_batch]

                data_room = np.vstack([data_room, data_batch]) if data_room.size else data_batch
                label_room = np.hstack([label_room, label_batch]) if label_room.size else label_batch
                sample_weight = np.hstack([sample_weight, batch_weight]) if label_room.size else batch_weight
                index_room = np.hstack([index_room, point_idxs]) if index_room.size else point_idxs
        data_room = data_room.reshape((-1, self.num_point, data_room.shape[1]))
        label_room = label_room.reshape((-1, self.num_point))
        sample_weight = sample_weight.reshape((-1, self.num_point))
        index_room = index_room.reshape((-1, self.num_point))
        points = Points(data=data_room, label_room=label_room, index_room=index_room, weights=sample_weight)
        # points.transform(self.pipelines)
        data_dict = {}
        data_dict['points'] = points
        data_dict['metas'] = {'weights': sample_weight}
        return data_dict
    
    def eval(self, preds_choice, seg_labels, meters):
        if isinstance(preds_choice, torch.Tensor):
            preds_choice = preds_choice.cpu().data.numpy() # b, n
        if isinstance(seg_labels, torch.Tensor):
            seg_labels = seg_labels.long().cpu().data.numpy() # b, n
        
        # preds_choice = np.argmax(preds_choice, -1)
        preds_choice = preds_choice.reshape(-1)
        seg_labels = seg_labels.reshape(-1)
        
        num_points = preds_choice.shape[0]
        meters.update({'eval_point_accuracy': {
            'val': np.sum(preds_choice == seg_labels),
            'count': num_points
        }})

        for i in range(self.num_class):
            seen_class = np.sum(preds_choice == i)
            correct_class = np.sum((preds_choice == i) & (seg_labels == i))
            iou_deno_class = np.sum(((preds_choice == i) | (seg_labels == i)))

            meters.update({f'class_{i}_IoU': {
                'val': correct_class,
                'count': iou_deno_class + 1e-6
            }})

            meters.update({f'class_{i}_accuracy': {
                'val': correct_class,
                'count': seen_class + 1e-6
            }})
        
        results = "Eval point accuracy: {0:.6f}\n".format(meters.eval_point_accuracy.avg)

        class_accuracy = np.array([getattr(meters, f'class_{i}_accuracy').avg for i in range(0, self.num_class)])
        class_accuracy = np.mean(class_accuracy)
        results += "Eval point avg class accuracy: {0:.6f}\n".format(class_accuracy)

        class_IoU = np.array([getattr(meters, f'class_{i}_IoU').avg for i in range(0, self.num_class)])
        class_IoU = np.mean(class_IoU)
        results += "Eval point avg class IoU: {0:.6f}\n".format(class_IoU)

        results += "Eval point class:\n"
        results += "{0:^5} {1:^5} {2:^5}\n".format("Class", "Accuracy", "IoU")
        for i in range(self.num_class):
            accuracy = getattr(meters, f'class_{i}_accuracy').avg
            IoU = getattr(meters, f'class_{i}_IoU').avg
            results += "{0:^5} {1:^5} {2:^5}\n".format(i, accuracy, IoU)

        return results, class_IoU

