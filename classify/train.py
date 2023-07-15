import os, torch, sys
ROOT_PATH = os.path.abspath('./')
sys.path.append(ROOT_PATH)
from datasets import DataContainer
from classify.models import CLASSIFIER
from jlcv.optimizer import Optimizer
from jlcv.average_meter import MetricMeter
from jlcv.config import Config
from jlcv.logger import Logger
from tqdm import tqdm

seed = 111
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

if __name__ == '__main__':
    cfg_file = 'classify/configs/mymodel_modelnet40.yaml'
    config = Config(cfg_file)
    work_dir = config.work_dir
    
    logger = Logger(name='cls', log_dir=config.work_dir)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    train_data_container = DataContainer(config.dataset.train, config.dataloader.train)
    test_data_container = DataContainer(config.dataset.test, config.dataloader.test)
    
    model = CLASSIFIER.build(config.model).to(device)
    
    best_accuracy = model.metric
    optimizer, scheduler = Optimizer.build(model, config)

    start_epoch, total_epoch = config.start_epoch, config.total_epoch
    
    train_meter = MetricMeter()
    test_meter = MetricMeter()

    logger.info('Start Training')
    for epoch in range(start_epoch, total_epoch):
        train_meter.reset()
        logger.info(f'[Epoch {epoch}]')

        for i, train_data in tqdm(enumerate(train_data_container.dataloader), total=len(train_data_container.dataloader)):
            model.train()
            optimizer.zero_grad()
            
            preds_dict = model(train_data)
            preds_logits = preds_dict['preds_logits']
            labels = train_data['points'].label.long()
            
            loss_dict = model.get_loss(preds_dict, labels)
            optimizer.step()
            
            info, accuracy = train_data_container.eval(preds_logits, labels, train_meter)

        log = f'TRAIN [Epoch {epoch}][{i+1}/{len(train_data_container.dataloader)}] ' + info
        logger.info(log)
        scheduler.step()
        

        model.eval()
        test_meter.reset()
        with torch.no_grad():
            for i, test_data in tqdm(enumerate(test_data_container.dataloader), total=len(test_data_container.dataloader)):
                preds_dict = model(test_data)

                preds_logits = preds_dict['preds_logits']
                labels = test_data['points'].label.long()
                info, accuracy = test_data_container.eval(preds_logits, labels, test_meter)

            log = f'TEST [Epoch {epoch}][{i+1}/{len(test_data_container.dataloader)}] ' + info
            logger.info(log)

            ckpt_path = f'{work_dir}/model_last.pth'
            model.save_model(ckpt_path, best_accuracy)
            logger.info(f'Saving checkpoints at {ckpt_path}')

            if best_accuracy <= accuracy:
                best_accuracy = accuracy
                ckpt_path = f'{work_dir}/model_best.pth'
                model.save_model(ckpt_path, best_accuracy)
                logger.info(f'Saving checkpoints at {ckpt_path}')
            logger.info('Best Test Instance Accuracy: {:.6f}'.format(best_accuracy))
            








        


            

            












