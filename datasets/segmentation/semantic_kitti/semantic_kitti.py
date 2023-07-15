import torch 
import os
import glob
import numpy as np
import copy
from datasets.base_dataset import BaseDataset
from jlcv.io import FileIO
from tqdm import tqdm

splits = {
    "train": ["00", "01", "02", "03", "04", "05", "06", "07", "09", "10"],
    "val": ["08"],
    "test": ["11", "12", "13", "14", "15", "16", "17", "18", "19", "20", "21"],
}

class SemanticKITTIDataset(BaseDataset):
    def __init__(self, 
                 root, 
                 split_file, 
                 classes_file=None, 
                 select_classes=None, 
                 pipelines=None, 
                 frustum_size=8,
                 project_scale=2,
                 downscale=[1, 8],
                 preprocess_root=None,
                 remap_lut_path=None,):
        super().__init__()
        self.root = root
        self.preprocess_root = root if preprocess_root is None else preprocess_root
        self.remap_lut_path = remap_lut_path
        self.downscale = downscale

        self.frustum_size = frustum_size
        self.project_scale = project_scale
        self.output_scale = int(self.project_scale / 2)


    def get_data_infos(self, split_file, **kwargs):
        data_infos = []

        for sequence in split_file:
            calib = self.load_calib(
                os.path.join(self.root, "dataset", "sequences", sequence, "calib.txt")
            )
            preprocess_dir = os.path.join(self.preprocess_root, 'sequences', sequence, "preprocess")
            filenames = os.listdir(preprocess_dir)
            for filename in filenames:
                file_path = os.path.join(preprocess_dir, filename)
                info = FileIO.load(file_path)
                info['calib'] = calib
                data_infos.append(info)
        return data_infos

    def __getitem__(self, index):
        infos = self.data_infos[index]
        sequence = infos["sequence"]
        frame_id = infos["frame_id"]
        voxel_path = infos["voxel_path"]
        img_path = infos['img_path']
        label = infos['label']
        calib = infos['calib']

        scale_3ds = [self.output_scale, self.project_scale]
        P = calib["P2"][0:3, 0:3]
        T_velo_2_cam = calib["T_velo_2_cam"]
        proj_matrix = P @ T_velo_2_cam

        for scale_3d in scale_3ds:
            # compute the 3D-2D mapping
            projected_pix, fov_mask, pix_z = vox2pix(
                T_velo_2_cam,
                cam_k,
                self.vox_origin,
                self.voxel_size * scale_3d,
                self.img_W,
                self.img_H,
                self.scene_size,
            )
    

    def compute_local_frustums(projected_pix, pix_z, target, img_W, img_H, dataset, n_classes, size=4):
        H, W, D = target.shape
        ranges = [(i * 1.0/size, (i * 1.0 + 1)/size) for i in range(size)]
        local_frustum_masks = []
        local_frustum_class_dists = []
        pix_x, pix_y = projected_pix[:, 0], projected_pix[:, 1]
        for y in ranges:
            for x in ranges:
                start_x = x[0] * img_W
                end_x = x[1] * img_W
                start_y = y[0] * img_H
                end_y = y[1] * img_H
                local_frustum = compute_local_frustum(pix_x, pix_y, start_x, end_x, start_y, end_y, pix_z)
                if dataset == "NYU":
                    mask = (target != 255) & np.moveaxis(local_frustum.reshape(60, 60, 36), [0, 1, 2], [0, 2, 1])
                elif dataset == "kitti":
                    mask = (target != 255) & local_frustum.reshape(H, W, D)

                local_frustum_masks.append(mask)
                classes, cnts = np.unique(target[mask], return_counts=True)
                class_counts = np.zeros(n_classes)
                class_counts[classes.astype(int)] = cnts
                local_frustum_class_dists.append(class_counts)
        frustums_masks, frustums_class_dists = np.array(local_frustum_masks), np.array(local_frustum_class_dists)
        return frustums_masks, frustums_class_dists
        

        

        


        


    @staticmethod
    def load_calib(calib_path):
        calib_all = {}
        with open(calib_path, "r") as f:
            for line in f.readlines():
                if line == "\n":
                    break
                key, value = line.split(":", 1)
                calib_all[key] = np.array([float(x) for x in value.split()])

        # reshape matrices
        calib = {}
        # 3x4 projection matrix for left camera
        calib["P2"] = calib_all["P2"].reshape(3, 4)
        calib["Tr"] = np.identity(4)  # 4x4 matrix
        calib["Tr"][:3, :4] = calib_all["Tr"].reshape(3, 4)
        calib['proj_matrix'] = calib["P2"] @ calib["Tr"]
        return calib
     
    def preprocess(self, preprocess_label=False):
        ###################### Loading remap lut file ######################
        print("Loading remap lut file")
        remap_lut = self.load_remap_lut(self.remap_lut_path)
        for sequence in self.split_file:
            out_dir = os.path.join(self.preprocess_root, 'sequences', sequence, "preprocess")
            os.makedirs(out_dir, exist_ok=True)

            sequence_path = os.path.join(self.root, "dataset", "sequences", sequence)
            voxel_paths = sorted(glob.glob(os.path.join(sequence_path, "voxels", "*.bin")))
            num_sample = len(voxel_paths)
            print(f"The number of sequence {sequence} is {num_sample}")

            for i in range(num_sample):
                voxel_path = voxel_paths[i]
                frame_id, extension = os.path.splitext(os.path.basename(voxel_path))
                invalid_path = os.path.join(sequence_path, "voxels", f"{frame_id}.invalid")
                label_path = os.path.join(sequence_path, "voxels", f"{frame_id}.label")
                occluded_path = os.path.join(sequence_path, "voxels", f"{frame_id}.occluded")
                img_path = os.path.join(sequence_path, "image_2", frame_id + ".png")

                sample_dict = {}
                sample_dict['frame_id'] = frame_id
                sample_dict['sequence'] = sequence
                sample_dict['voxel_path'] = voxel_path
                sample_dict['invalid_path'] = invalid_path
                sample_dict['label_path'] = label_path
                sample_dict['occluded_path'] = occluded_path
                sample_dict['img_path'] = img_path

                if preprocess_label:
                    invalid = FileIO.load(invalid_path, dtype=np.uint8, do_unpack=True)
                    label = FileIO.load(label_path, dtype=np.uint16, do_unpack=False)
                    # Remap 20 classes semanticKITTI SSC
                    label = remap_lut[label].astype(np.float32)  
                    # Setting to unknown all voxels marked on invalid mask...
                    label[np.isclose(invalid, 1)] = 255  
                    label = label.reshape([256, 256, 32])
                    sample_dict['label'] = label

                    if self.downscale is not None:
                        for downscale in self.downscale:
                            label_downscale = self.downsample_label(label, (256, 256, 32), downscale)
                            sample_dict[f'label_{downscale}'] = label_downscale

                file_path = os.path.join(out_dir, f"{frame_id}.pkl")
                FileIO.dump(sample_dict, file_path)
                print("wrote to", file_path)


    @staticmethod
    def downsample_label(label, voxel_size=(240, 144, 240), downscale=4):
        if downscale == 1:
            return label
        downscale_size = (
            voxel_size[0] // downscale,
            voxel_size[1] // downscale,
            voxel_size[2] // downscale,
        )  # small size

        label_downscale = np.zeros(downscale_size, dtype=np.uint8)
        empty_t = 0.95 * downscale * downscale * downscale  # threshold
        s01 = downscale_size[0] * downscale_size[1]
        label_i = np.zeros((downscale, downscale, downscale), dtype=np.int32)

        for i in range(downscale_size[0] * downscale_size[1] * downscale_size[2]):
            z = int(i / s01)
            y = int((i - z * s01) / downscale_size[0])
            x = int(i - z * s01 - y * downscale_size[0])

            label_i[:, :, :] = label[
                x * downscale : (x + 1) * downscale, y * downscale : (y + 1) * downscale, z * downscale : (z + 1) * downscale
            ]
            label_bin = label_i.flatten()

            zero_count_0 = np.array(np.where(label_bin == 0)).size
            zero_count_255 = np.array(np.where(label_bin == 255)).size

            zero_count = zero_count_0 + zero_count_255
            if zero_count > empty_t:
                label_downscale[x, y, z] = 0 if zero_count_0 > zero_count_255 else 255
            else:
                label_i_s = label_bin[
                    np.where(np.logical_and(label_bin > 0, label_bin < 255))
                ]
                label_downscale[x, y, z] = np.argmax(np.bincount(label_i_s))
        return label_downscale



    @staticmethod
    def load_remap_lut(remap_lut_path):
        config = FileIO.load(remap_lut_path)
        # make lookup table for mapping
        maxkey = max(config['learning_map'].keys())

        # +100 hack making lut bigger just in case there are unknown labels
        remap_lut = np.zeros((maxkey + 100), dtype=np.int32)
        remap_lut[list(config['learning_map'].keys())] = list(config['learning_map'].values())

        # in completion we have to distinguish empty and invalid voxels.
        # Important: For voxels 0 correspondownscale to "empty" and not "unlabeled".
        remap_lut[remap_lut == 0] = 255  # map 0 to 'invalid'
        remap_lut[0] = 0  # only 'empty' stays 'empty'.
        return remap_lut
    
