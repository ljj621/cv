import torch
import torch.nn as nn
import torch.nn.functional as F
from jlcv.modules import build_module, build_conv_layer, build_linear_layer
from jlcv.loss import build_loss_layer


class CompleteDT(nn.Module):
    def __init__(self,
                 channels,  # 64, 128, 256
                 num_points,
                 input_module,
                 query_encoder,
                 query_decoder,
                 loss_layer,
                 conv_cfg='Conv1d',
                 norm_cfg='BN',
                 act_cfg='GELU') -> None:
        super().__init__()
        num_points_list = input_module['num_points_list']
        self.input_module = build_module(input_module, channels=channels)
   
        self.num_pc = len(num_points_list)
        
        # query encoder
        out_channels = [3*16384//n for n in num_points_list]
 
        for i, channel in enumerate(out_channels):
            mlp = build_conv_layer(conv_cfg, channels[0]*2**i, channel, 1)
            self.__setattr__(f'mlp_{i}', mlp)

        for i in range(self.num_pc-1):
            encoder = build_module(query_encoder, channels=[out_channels[i], out_channels[i+1]])
            self.__setattr__(f'query_encoder_{i}', encoder)

        self.global_layer = build_conv_layer(conv_cfg, out_channels[-1], 512, 1, norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.query_layer = nn.Sequential(
            build_linear_layer(512, 512, act_cfg=act_cfg),
            build_linear_layer(512, num_points_list[-1]*out_channels[-1], act_cfg=act_cfg),
        )

        # query decoder
        out_channels = [*out_channels, out_channels[-1]]
        num_points_list = [num_points, *num_points_list]

        for i in range(self.num_pc, 0, -1):
            decoder = build_module(query_decoder, channels=[out_channels[i], out_channels[i-1]], num_points=num_points_list[i-1] // num_points_list[i])
            self.__setattr__(f'query_decoder_{i}', decoder)

            if i == self.num_pc - 1:
                fusion_layer = build_conv_layer(conv_cfg, out_channels[i], 3, 1)
            else:
                fusion_layer = build_conv_layer(conv_cfg, out_channels[i-1], out_channels[i], 1, norm_cfg=norm_cfg, act_cfg=act_cfg)
            self.__setattr__(f'fusion_layer_{i}', fusion_layer)
        
        self.loss_layer = build_loss_layer(loss_layer)

    
    def forward(self, points):
        points = points.permute(0,2,1)
        points_list, feats_list, grouped_index_list= self.input_module(points)

        for i, feats in enumerate(feats_list):
            feats_list[i] = self.__getattr__(f'mlp_{i}')(feats)
        
        for i in range(self.num_pc-1):
            feats1 = feats_list[i]
            feats2 = feats_list[i+1]
            feats2 = self.__getattr__(f'query_encoder_{i}')(feats1, feats2, grouped_index_list[i])
            feats_list[i+1] = feats2
        
        global_feats = self.global_layer(feats_list[-1]).max(-1)[0]
        query_feats = self.query_layer(global_feats)

        # decoder
        query_feats_list = []
        preds_dict = []
        for i, feats in enumerate(feats_list):
            query_feats, pc = self.__getattr__(f'query_decoder_{i}')(query_feats, feats)
            query_feats_list.append(query_feats)
            preds_dict[i] = pc.transpose(1,2)

            b, c, n, s = query_feats.shape
            query_feats = query_feats.reshape(b, c, n*s)
            
        for i, query_feats in enumerate(query_feats_list):
            b, c, n, s = query_feats.shape
            query_feats = query_feats.reshape(b, c, -1)
            query_feats = self.__getattr__(f'fusion_layer_{i}')(query_feats)
            if i < len(query_feats_list) - 1:
                query_feats = query_feats_list[i+1] + query_feats[..., None]

        preds_dict['preds'] = {query_feats.transpose(1,2)}
        return preds_dict
    
    def get_loss(self, preds_dict, targets, alpha=None):
        if alpha is None:
            alpha = {k: 1 for k in preds_dict}
        total_loss = 0
        loss_dict = {}
        for k, v in preds_dict.items():
            loss_cd1, loss_cd2, ins_dist1, ins_dist2 = self.loss_layer(v, targets)
            total_loss = total_loss + loss_cd2 * alpha[k]
            loss_dict[f'{k}_loss'] = loss_cd2.item()

        total_loss.backward()
        loss_dict['total_loss'] = total_loss.item()
        return loss_dict
    
    def get_metirc(self, preds_dict, targets, category=None):
        metirc_dict = {}
        preds = preds_dict[f'preds']
        loss_cd1, loss_cd2, ins_dist1, ins_dist2 = self.loss_layer(preds, targets)
        metirc_dict['loss_cd1'] = loss_cd1.items()
        metirc_dict['loss_cd2'] = loss_cd2.items()

        if category is not None:
            ins_dist1 = ins_dist1.cpu().data.numpy()
            ins_dist2 = ins_dist2.cpu().data.numpy()
            metirc_dict['ins_dist1'] = ins_dist1
            metirc_dict['ins_dist2'] = ins_dist2
            from jlcv.metrics import F1Score
            f1_score,_,_= F1Score()(ins_dist1, ins_dist2)
            metirc_dict['f1_score'] = f1_score

            assert len(targets) == len(category)
            assert isinstance(category, dict)
            instance_metirc = {}
            for i, class_name, class_label in enumerate(category.items()):
                if class_name not in instance_metirc:
                    instance_metirc[class_name] = 0
                    instance_metirc[f'{class_name}_count'] = 0
                instance_metirc[class_name] += ins_dist1[i] + ins_dist2[i]
                instance_metirc[f'{class_name}_count'] += 1
            metirc_dict['instance'] = instance_metirc
        return metirc_dict
    
    def train_step(self, batch):
        targets = batch.targets
        points = batch.points.data
        preds_dict = self.forward(points)

        loss_dict = self.get_loss(preds_dict, targets)
        self.train_meter.update(loss_dict)

        metirc_dict = self.get_metirc(preds_dict, targets)
        self.train_meter.update(metirc_dict)

        log = self.train_meter.info()
        return log
    
    def test_step(self, batch):
        targets = batch.targets
        points = batch.points.data
        preds_dict = self.forward(points)

        metirc_dict = self.get_metirc(preds_dict, targets)
        
        _dict = {k: v for k,v in metirc_dict.items() if k != 'instance'}
        self.test_meter.update(_dict)

        log = self.test_meter.info()
        return log
    


        





        







       




        




        

        

        










        
