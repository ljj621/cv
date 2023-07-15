import os, torch, sys
ROOT_PATH = os.path.abspath('./')
sys.path.append(ROOT_PATH)
from datasets import DataContainer
from segmentation.models import SEGMENTATION
from jlcv.optimizer import Optimizer
from jlcv.config import Config
from jlcv.logger import Logger
from jlcv.average_meter import MetricMeter
from tqdm import tqdm
import numpy as np

seed = 111
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

if __name__ == '__main__':
    cfg_file = 'segmentation/configs/mymodel_sem_seg.yaml'
    config = Config(cfg_file)
    work_dir = config.work_dir
    
    logger = Logger(name='main', log_dir=config.work_dir)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    config.set('device', device)
    
    train_data_container = DataContainer(config.dataset.train, config.dataloader.train)
    test_data_container = DataContainer(config.dataset.test, config.dataloader.test)
    
    model = SEGMENTATION.build(config.model).to(device)

    best_accuracy = model.metric
    optimizer, scheduler = Optimizer.build(model, config)

    start_epoch, total_epoch = config.start_epoch, config.total_epoch

    logger.info('Start Training')
    train_meter = MetricMeter()
    test_meter = MetricMeter()
    num_iter = 0
    for n in range(start_epoch, total_epoch):
        train_meter.reset()
        logger.info(f'TRAIN [Epoch {n}]')
        for i, train_data in tqdm(enumerate(train_data_container.dataloader), total=len(train_data_container.dataloader)):
            model.train()
            optimizer.zero_grad()

            preds_dict = model(train_data)
            preds_choice = preds_dict['preds_choice']
            seg_labels = train_data['points'].seg_labels.long()
            loss_dict = model.get_loss(preds_dict, seg_labels)
            optimizer.step()

            info, class_IoU = train_data_container.eval(preds_choice, seg_labels, train_meter)

        log = f'TRAIN [Epoch {n}][{i+1}/{len(train_data_container.dataloader)}] ' + info
        logger.info(log)
        scheduler.step()

        
        ckpt_path = f'{work_dir}/train_model.pth'
        model.save_model(ckpt_path, class_IoU)
        logger.info(f'Saving checkpoints at {ckpt_path}')

        model.eval()
        with torch.no_grad():
            test_meter.reset()
            for i, test_data in tqdm(enumerate(test_data_container.dataloader), total=len(test_data_container.dataloader)):
                preds_dict = model(test_data)

                preds_choice = preds_dict['preds_choice']
                seg_labels = test_data['points'].seg_labels.long()
                info, class_IoU = test_data_container.eval(preds_choice, seg_labels, test_meter)
            
            log = f'TEST [Epoch {n}][{i+1}/{len(test_data_container.dataloader)}] ' + info
            logger.info(log)

            accuracy = class_IoU

            if best_accuracy <= accuracy:
                best_accuracy = accuracy
                ckpt_path = f'{work_dir}/model_best.pth'
                model.save_model(ckpt_path, accuracy)
                logger.info(f'Saving checkpoints at {ckpt_path}')
            logger.info('Best Test Instance Accuracy: {:.6f}'.format(best_accuracy))


        


            
