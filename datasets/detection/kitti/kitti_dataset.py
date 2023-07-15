import os
import numpy as np
from jlcv.io import FileIO
import cv2

from datasets import DATASETS
from datasets.base_dataset import BaseDataset
from datasets.instances import Points, CameraBbox, LidarBbox, LidarPoints, Img

classes_to_label = {'Car': 1, 'Pedestrian': 2, 'Cyclist': 3, 'Van': 4}
@DATASETS.register_module()
class KittiDataset(BaseDataset):
    def __init__(self, 
                 root, 
                 split_file, 
                 classes_file=None, 
                 select_classes=None, 
                 pipelines=None,
                 point_cloud_range=None,
                 grid_size=None,
                 voxel_size=None,
                 modality=None,
                 ):
        super().__init__(root, split_file, classes_file, select_classes, pipelines)
        self.split = split
        self.point_cloud_range = np.array(point_cloud_range, dtype=np.float32)
        self.grid_size = grid_size
        self.voxel_size = voxel_size
        self.modality = modality

        if split != 'test':
            self.data_path = os.path.join(root, 'training')
        else:
            self.data_path = os.path.join(root, 'testing')

        split_path = os.path.join(root, 'ImageSets', f'{split}.txt')
        self.sample_index_list = FileIO.load(split_path)

    def get_data_infos(self, split_file):
        infos = FileIO.load(split_file)
        return infos

    
    def get_classes(self, classes_file=classes_to_label, select_classes=None):
        if isinstance(classes_file, dict): return classes_file
    
    def __getitem__(self, index):
        info = self.data_infos[index]
        sample_idx = info['point_cloud']['sample_index']
        img_shape = info['image']['image_shape']
        calib = info['calib']

        input_dict = {}
        if 'points' in self.modality:
            points_path = info['point_cloud']['path']
            points = self.load_points(points_path)
            input_dict['points'] = points
        
        if 'img' in self.modality:
            img_path = info['image']['path']
            img = self.load_img(img_path)
            img.normalize() # TODO 可以放在 pipe
        
        if 'depth' in self.modality:
            depth_path = info['depth']['path']
            depth = self.load_img(depth_path)
            depth.normalize()

        if 'annotations' in info:
            annos = info['annotations']
            # filter 'DontCare'
            keep_indices = [i for i, x in enumerate(annos['class_name']) if x != name]
            for key in annos.keys():
                annos[key] = annos[key][keep_indices]
            
            input_dict['gt_boxes'] = annos['gt_boxes_lidar']
            input_dict['class_name'] = annos['class_name']
            input_dict['gt_boxes2d'] = annos["bbox2d"]
            
            
    @staticmethod
    def preprocess(root, split, load_annotations=False, load_inside_points=False):
        print(f'-------------- Start to generate {split} data infos --------------')
        if split != 'test':
            data_path = os.path.join(root, 'training')
        else:
            data_path = os.path.join(root, 'testing')

        split_path = os.path.join(root, 'ImageSets', f'{split}.txt')
        sample_index_list = FileIO.load(split_path)
        sample_info_list = []
        for sample_index in sample_index_list:
            sample_info_list.append(KittiDataset.process_sample(data_path, sample_index, load_annotations, load_inside_points))

    @staticmethod
    def process_gt_database(root, split, info_path, used_classes=None):
        if split != 'test':
            data_path = os.path.join(root, 'training')
        else:
            data_path = os.path.join(root, 'testing')

        split_path = os.path.join(root, 'ImageSets', f'{split}.txt')
        sample_index_list = FileIO.load(split_path)

        db_save_dir = os.path.join(root, 'kitti_gt_database', split)
        os.makedirs(db_save_dir, exist_ok=True)
    
        db_info_save_path = os.path.join(root, 'kitti_gt_database', f'kitti_dbinfos_{split}.pkl')

        db_infos = {}

        infos = FileIO.load(info_path)

        for k in range(len(infos)):
            print('gt_database sample: %d/%d' % (k + 1, len(infos)))
            info = infos[k]
            sample_index = info['point_cloud']['sample_index']
            points = np.fromfile(info['point_cloud']['path'], dtype=np.float32).reshape(-1, 4)
            annos = info['annotations']
            names = annos['class_name']
            difficulty = annos['difficulty']
            bbox2d = annos['bbox2d']
            gt_boxes_lidar = annos['gt_boxes_lidar']
            index = annotations['index']

            points_in_bbox_index = gt_boxes_lidar.points_in_bbox_cpu(points).numpy()

            for i in index:
                filename = '%s_%s_%d.bin' % (sample_idx, names[i], i)
                filepath = db_save_dir / filename
                gt_points = points[point_indices[i] > 0]

                gt_points[:, :3] -= gt_boxes[i, :3]
                with open(filepath, 'w') as f:
                    gt_points.tofile(f)

                if (used_classes is None) or names[i] in used_classes:
                    db_path = str(filepath.relative_to(self.root_path))  # gt_database/xxxxx.bin
                    db_info = {'name': names[i], 'path': db_path, 'image_idx': sample_idx, 'gt_idx': i,
                               'box3d_lidar': gt_boxes[i], 'num_points_in_gt': gt_points.shape[0],
                               'difficulty': difficulty[i], 'bbox': bbox[i], 'score': annos['score'][i]}
                    if names[i] in db_infos:
                        db_infos[names[i]].append(db_info)
                    else:
                        db_infos[names[i]] = [db_info]

        for k, v in db_infos.items():
            print('Database %s: %d' % (k, len(v)))
        
        FileIO.dump(db_infos, db_info_save_path)
 
    @staticmethod
    def process_sample(data_path, sample_index, load_annotations=False, load_inside_points=False):
        info = {}
        points_path = os.path.join(data_path, 'velodyne', f'{sample_index}.txt')
        info['point_cloud'] = {'path': points_path, 'num_features': 4, 'sample_index': sample_index}
        # step 1. load img
        img_path = os.path.join(data_path, 'image_2', f'{sample_index}.png')
        img = cv2.imread(img_path)
        info['image'] = {'path': img_path, 'sample_index': sample_index, 'image_shape': img.shape[:2]}

        depth_path = os.path.join(data_path, 'depth_2', f'{sample_index}.png')
        depth = cv2.imread(depth_path)
        info['depth'] = {'path': depth_path, 'sample_index': sample_index, 'image_shape': depth.shape[:2]}

        # step 1. load calib
        calib_path = os.path.join(data_path, 'calib', f'{sample_index}.txt')
        calib = KittiDataset.load_calib(calib_path)
        info['calib'] = calib

        # step 2. load annotations
        if load_annotations:
            annotations_path = os.path.join(data_path, 'label_2', f'{sample_index}.txt')
            annotations = KittiDataset.load_annotations(annotations_path)
            gt_boxes_camera = annotations['gt_boxes_camera']
            gt_boxes_lidar = gt_boxes_camera(gt_boxes_camera).to_lidar(calib['RO'], calib['Tr_velo2cam'])
            annotations['gt_boxes_lidar'] = gt_boxes_lidar
            
            class_name = annotations['class_name']
            if load_inside_points:
                # step 2.1. load lidar points
                points_lidar = np.fromfile(points_path, dtype=np.float32).reshape(-1, 4)
                points_lidar = LidarPoints(points_lidar)

                # step 2.2. load lidar points within bboxes
                points_fov = points_lidar.to_fov(calib['RO'], calib['Tr_velo2cam'], calib['P2'], info['image']['image_shape'])
                corners_lidar = points_lidar.corners
                num_points_in_gt = -np.ones(num_gt, dtype=np.int32)
                
                for i, name in enumerate(class_name):
                    if name == 'DontCare':
                        flag = points_fov.in_hull(pts_fov.data[:, 0:3], corners_lidar[k])
                        num_points_in_gt[k] = flag.sum()
                annotations['num_points_in_gt'] = num_points_in_gt
            
            info['annotations'] = annotations
        return info
    
    @staticmethod
    def load_img(img_path):
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        return Img(img)
    
    @staticmethod
    def load_points(points_path):
        points = np.fromfile(points_path, dtype=np.float32).reshape(-1, 4)
        return LidarPoints(points)

    
    @staticmethod
    def load_calib(calib_path):
        contents = FileIO.load(calib_path)
        P2 = np.array(contents[2][1:], dtype=np.float32).reshape(3, 4)
        P3 = np.array(contents[3][1:], dtype=np.float32).reshape(3, 4)
        R0 = np.array(contents[4][1:], dtype=np.float32).reshape(3, 3)
        Tr_velo2cam = np.array(contents[5][1:], dtype=np.float32).reshape(3, 4)

        calib = {
            'P2': P2, 
            'P3': P3, 
            'R0': R0, 
            'Tr_velo2cam': Tr_velo2cam,
            'cu': P2[0, 2],
            'cv': P2[1, 2],
            'fu': P2[0, 0],
            'fv': P2[1, 1],
            'tx': P2[0, 3] / (-P2[0, 0]),
            'ty': P2[1, 3] / (-P2[1, 1])
        }
        return calib

    @staticmethod
    def load_annotations(annotations_path):
        sample_annotations_list = FileIO.load(annotations_path)
        
        anno_list = []
        for label in sample_annotations_list:
            class_name = label[0]
            class_label =  classes_to_label[class_name] if class_name in classes_to_label else -1
            truncation = float(label[1])
            occlusion = float(label[2])  # 0:fully visible 1:partly occluded 2:largely occluded 3:unknown
            alpha = float(label[3])
            bbox2d = np.array([float(label[4]), float(label[5]), float(label[6]), float(label[7])], dtype=np.float32).reshape(1, 4)
            dimensions = np.array([float(label[10]), float(label[8]), float(label[9])], dtype=np.float32).reshape(1, 3) # l h w
            location = np.array([float(label[11]), float(label[12]), float(label[13])], dtype=np.float32).reshape(1, 3)
            dis_to_cam = np.linalg.norm(location)
            rotation_y = float(label[14]).reshape(1, 1)
            score = float(label[15]) if label.__len__() == 16 else -1.0
            height = float(bbox2d[3]) - float(bbox2d[1]) + 1
            if height >= 40 and truncation <= 0.15 and occlusion <= 0:
                difficulty = 0  # Easy
            elif height >= 25 and truncation <= 0.3 and occlusion <= 1:
                difficulty = 1  # Moderate
            elif height >= 25 and truncation <= 0.5 and occlusion <= 2:
                difficulty = 2  # Hard
            else:
                difficulty = -1
            gt_boxes_camera = np.concatenate([location, dimensions, rotation_y], axis=1)

            anno = dict(
                class_name=class_name,
                class_label=class_label,
                truncation=truncation,
                occlusion=occlusion,
                alpha=alpha,
                bbox2d=bbox2d,
                dimensions=dimensions,
                location=location,
                dis_to_cam=dis_to_cam,
                rotation_y=rotation_y,
                score=score,
                difficulty=difficulty,
                gt_boxes_camera = CameraBbox(gt_boxes_camera),
            )
            anno_list.append(anno)

        annotations = {}
        for k in anno_list[0]:
            if k in ['bbox2d', 'location', 'dimensions']:
                annotations[k] = np.concatenate([anno[k] for anno in anno_list], 0)
            else:
                annotations[k] = np.array([anno[k] for anno in anno_list])
        return annotations
        
        
        

            
            
            




        


