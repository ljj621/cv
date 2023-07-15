import torch
import torch.nn as nn
from jlcv.modules import build_module, build_conv_layer, build_linear_layer
from jlcv.modules.ext import furthest_point_sample, gather_points, group_points, knn
from jlcv.modules.tools.encoder import MyModuleModuleEncoder
from jlcv.modules.tools.decoder import MyModuleModuleDecoder
from jlcv.metrics import build_metric_layer
from . import COMPLETION
from .pcn import PCNEncoder

@COMPLETION.register_module()
class MyModel(nn.Module):
    def __init__(self,
                 channels,
                 num_dense,
                 num_points,
                 encoder,
                 neck,
                 decoder,
                 conv_cfg='Conv1d',
                 norm_cfg='BN',
                 act_cfg='GELU',
                 loss_layer=None) -> None:
        super().__init__()
        self.num_dense = num_dense
        self.num_points = num_points # [None, 512, 256, 128]

        # PCN ENCODER
        # self.pcn_encoder = PCNEncoder(1024)

        # UNET-TRANSFORMER
        # downsample
        in_channels = 3
        self.encoder = nn.ModuleList()
        for i, npoints in enumerate(num_points):
            self.encoder.append(
                MyModuleModuleEncoder(in_channels, channels[i], encoder, npoints)
            )
            in_channels = channels[i]
        
        channel = sum(channels)
        self.fc1 = nn.Sequential(
            build_linear_layer(channels[-1], channel, norm_cfg=norm_cfg, act_cfg=act_cfg),
            build_linear_layer(channel, channel, norm_cfg=norm_cfg, act_cfg=act_cfg)
        )
        self.fc2 = nn.Sequential(
            build_linear_layer(channels[-1], channel, norm_cfg=norm_cfg, act_cfg=act_cfg),
            build_linear_layer(channel, channel, norm_cfg=norm_cfg, act_cfg=act_cfg)
        )
        self.fc3 = nn.Sequential(
            build_linear_layer(channel*2, channel, norm_cfg=norm_cfg, act_cfg=act_cfg),
            build_linear_layer(channel, channel, norm_cfg=norm_cfg, act_cfg=act_cfg)
        )
        self.neck = build_module(neck, channels=channel)
        self.fc4 = build_linear_layer(channel, channels[-1], norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.points = nn.Sequential(
            build_linear_layer(channels[-1], channels[-1], norm_cfg=norm_cfg, act_cfg=act_cfg),
            build_linear_layer(channels[-1], 64, norm_cfg=norm_cfg, act_cfg=act_cfg),
            build_linear_layer(64, 3)
        )

        self.decoder = nn.ModuleList()
        for i in range(len(channels)-1, -1, -1):
        # for i, channel in enumerate(channels):
            # if i == len(channels)-1:
            #     grid_size = 1
            #     grid_x = torch.linspace(-0.05, 0.05, steps=grid_size).view(1, grid_size).expand(grid_size, grid_size).reshape(1, -1)
            #     grid_y = torch.linspace(-0.05, 0.05, steps=grid_size).view(grid_size, 1).expand(grid_size, grid_size).reshape(1, -1)
            #     grid = torch.cat([grid_x, grid_y], dim=0).view(1, 2, grid_size ** 2).cuda()

            # else:
            grid_size = 2
            grid_x = torch.linspace(-0.05, 0.05, steps=grid_size).view(1, grid_size).expand(grid_size, grid_size).reshape(1, -1)
            grid_y = torch.linspace(-0.05, 0.05, steps=grid_size).view(grid_size, 1).expand(grid_size, grid_size).reshape(1, -1)
            grid = torch.cat([grid_x, grid_y], dim=0).view(1, 2, grid_size ** 2).cuda()
            self.decoder.append(
                MyModuleModuleDecoder(channels[i], decoder, grid)
            )

        self.loss_layer = build_metric_layer(loss_layer)

    def forward(self, points):
        points = points.transpose(1, 2)
        B, _, N = points.shape
        feats = points
        preds_dict = {}

        feats_list = []
        for encoder in self.encoder:
            feats, points = encoder(feats, points)
            feats_list.append(feats)
        
        B, _, M = feats.shape
        global_feats = self.fc1(feats.max(-1)[0])
        feats = self.fc2(feats.transpose(1,2).reshape(B*M, -1)).reshape(B, M, -1)
        feats = torch.cat([feats, global_feats[..., None, :].repeat(1, M, 1)], -1).reshape(B*M, -1)
        feats = self.fc3(feats).reshape(B, M, -1).transpose(1,2)
        feats = self.neck(feats, feats_list)
        feats = self.fc4(feats.transpose(1,2).reshape(B*M, -1)).reshape(B, M, -1).transpose(1,2)
        points = self.points(feats.transpose(1,2).reshape(B*M, -1)).reshape(B, M, -1).transpose(1,2)
        preds_dict['coarse'] = points.transpose(1, 2)

        for i, decoder in enumerate(self.decoder):
            feats, points = decoder(feats, feats_list[1-i], points)
            preds_dict[f'{points.shape[-1]}'] = points.transpose(1, 2)
        return preds_dict
    
    def get_loss(self, preds_dict, targets, alpha=None):
        if alpha is None:
            alpha = {k: 1 for k in preds_dict}
            alpha['128'] = 10
            alpha['512'] = 2
        total_loss = 0
        loss_dict = {}
        for k, v in preds_dict.items():
            cd1, cd2, ins_dist1, ins_dist2 = self.loss_layer(v, targets)
            total_loss = total_loss + cd2 * alpha[k]
            loss_dict[f'cd2_{k}'] = cd2.item()*10000
            
        total_loss.backward()
        loss_dict['total_loss'] = total_loss.item()*10000
        return loss_dict
    
    def get_metirc(self, preds_dict, targets):
        metirc_dict = {}
        preds = preds_dict[f'{self.num_dense}']
        loss_cd1, loss_cd2, ins_dist1, ins_dist2 = self.loss_layer(preds, targets)
        metirc_dict['cd1'] = loss_cd1.item()*10000
        metirc_dict['cd2'] = loss_cd2.item()*10000

        ins_dist1 = ins_dist1.cpu().data.numpy()
        ins_dist2 = ins_dist2.cpu().data.numpy()
        # metirc_dict['ins_dist1'] = ins_dist1
        # metirc_dict['ins_dist2'] = ins_dist2
        from jlcv.metrics import F1Score
        f1_score,_,_= F1Score()(ins_dist1, ins_dist2)
        metirc_dict['f1_score'] = f1_score.mean()

        # if category is not None:
        #     assert len(targets) == len(category)
        #     assert isinstance(category, dict)
        #     instance_metirc = {}
        #     for i, class_name, class_label in enumerate(category.items()):
        #         if class_name not in instance_metirc:
        #             instance_metirc[class_name] = 0
        #             instance_metirc[f'{class_name}_count'] = 0
        #         instance_metirc[class_name] += ins_dist1[i] + ins_dist2[i]
        #         instance_metirc[f'{class_name}_count'] += 1
        #     metirc_dict['instance'] = instance_metirc
        return metirc_dict


    def train_step(self, batch):
        targets = batch.gt.float()
        points = batch.partial.float()
        preds_dict = self.forward(points)

        loss_dict = self.get_loss(preds_dict, targets)
        return loss_dict
    
    def test_step(self, batch):
        targets = batch.gt.float()
        points = batch.partial.float()
        preds_dict = self.forward(points)

        metirc_dict = self.get_metirc(preds_dict, targets)
        return metirc_dict

    def save_model(self, ckpt_path, metrics=0, best_metrics=0):
        state = {
                'model_state': self.state_dict(),
                'curent_metrics': metrics,
                'best_metrics': best_metrics
                }
        torch.save(state, ckpt_path)
    

    


        








        










