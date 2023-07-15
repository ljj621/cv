import torch
import numpy as np
from tqdm import tqdm
def test(model, data_container, meters):
    dataloader = data_container.dataloader
    model.eval()
    with torch.no_grad():
        meters.reset()
        for i, test_data in tqdm(enumerate(dataloader), total=len(dataloader)):
            preds_dict = model(test_data)

            preds_choice = preds_dict['preds_choice']
            seg_labels = test_data['points'].seg_labels.long()
            info, class_IoU = data_container.eval(preds_choice, seg_labels, meters)
    return info, class_IoU


def voted(vote_label_pool, pred_label, point_idx, weight):
    if isinstance(pred_label, torch.Tensor):
        pred_label = pred_label.long().cpu().data.numpy() # b, n
    if isinstance(point_idx, torch.Tensor):
        point_idx = point_idx.long().cpu().data.numpy() # b, n
    if isinstance(weight, torch.Tensor):
        weight = weight.float().cpu().data.numpy() # b, n

    B = pred_label.shape[0]
    N = pred_label.shape[1]

    for b in range(B):
        for n in range(N):
            if weight[b, n] != 0 and not np.isinf(weight[b, n]):
                vote_label_pool[int(point_idx[b, n]), int(pred_label[b, n])] += 1
    return vote_label_pool


def test_whole_scene(model, data_container, meters, BATCH_SIZE, NUM_POINT):
    dataset = data_container.dataset
    data_infos = dataset.data_infos
    room_samples = data_infos['room_samples']
    
    model.eval()
    with torch.no_grad():
        for room_name, sample in room_samples.items():
            print(f'Eval room {room_name} ...')
            data_dict = dataset.get_single_scene(sample)
            scene_smpw = data_dict['metas']['weights']
            data = data_dict['points']
            data_type = type(data)
            scene_data = data.data
            scene_label = data.label_room
            scene_point_index = data.index_room

            num_blocks = scene_data.shape[0]
            s_batch_num = (num_blocks + BATCH_SIZE - 1) // BATCH_SIZE
            batch_data = np.zeros((BATCH_SIZE, NUM_POINT, 9))

            batch_label = np.zeros((BATCH_SIZE, NUM_POINT))
            batch_point_index = np.zeros((BATCH_SIZE, NUM_POINT))
            batch_smpw = np.zeros((BATCH_SIZE, NUM_POINT))

            vote_label_pool = np.zeros((sample['data'].shape[0], 13))

            for sbatch in tqdm(range(s_batch_num), total=s_batch_num):
                start_idx = sbatch * BATCH_SIZE
                end_idx = min((sbatch + 1) * BATCH_SIZE, num_blocks)
                real_batch_size = end_idx - start_idx
                batch_data[0:real_batch_size, ...] = scene_data[start_idx:end_idx, ...]
                batch_label[0:real_batch_size, ...] = scene_label[start_idx:end_idx, ...]
                batch_point_index[0:real_batch_size, ...] = scene_point_index[start_idx:end_idx, ...]
                batch_smpw[0:real_batch_size, ...] = scene_smpw[start_idx:end_idx, ...]
                batch_data[:, :, 3:6] /= 1.0

                torch_data = torch.Tensor(batch_data)
                torch_data = torch_data.float().cuda()
                torch_label = torch.Tensor(batch_label)
                torch_label = torch_label.long().cuda()

                torch_data = data_type(data=torch_data, seg_labels=torch_label)
                input_dict = {'points': torch_data, 'metas': {'weights': batch_smpw}}
                # torch_data = torch_data.transpose(2, 1)
                preds_dict = model(input_dict)

                preds_choice = preds_dict['preds_choice']
                vote_label_pool = voted(vote_label_pool, preds_choice, batch_point_index, batch_smpw)

            pred_label = np.argmax(vote_label_pool, 1)
            seg_labels = sample['seg_labels']
            info, class_IoU = data_container.eval(pred_label, seg_labels, meters)
    return info, class_IoU


