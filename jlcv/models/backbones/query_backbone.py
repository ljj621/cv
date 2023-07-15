import torch
import torch.nn as nn
import torch.nn.functional as F
from jlcv.modules.modules import QureyTransformer, MODULES
from jlcv.modules.ext import knn, ball_points, gather_points
from jlcv.modules.base import build_conv_layer, build_linear_layer
from jlcv.models import MODELS

@MODELS.register_module()
class QueryBackbone(nn.Module):
    def __init__(self, 
                 input_encoder=None,
                 middle_encoder=None,
                 query_generator=None,
                 query_encoder=None,
                 conv='Conv1d', 
                 norm='BN', 
                 act='GELU'):
        super().__init__()
        ###################  input encoder  #################
        self.input_encoder = MODELS.build(input_encoder)
        self.num_sample = [32, 64]

        ###################  middle encoder  #################
        num_layer = middle_encoder.get('num_layer', 1)
        in_channels = 256
        out_channels = middle_encoder['channels']
        self.middle_encoder = nn.ModuleList()
        layers = []
        for i in range(num_layer):
            layers.append(
                build_conv_layer(in_channels, out_channels, 1, conv=conv, norm=norm, act=act)
            )
            in_channels = out_channels
        self.middle_encoder = nn.Sequential(*layers)
        middle_channels = out_channels

        ###################  query generator  #################
        self.query_init = query_generator['init']
        if self.query_init in ['sparse_heatmap', 'heatmap']:
            self.num_queries = query_generator['num_queries']
            self.shape_class = query_generator['shape_class']
            heatmap_channels = query_generator['channels']
            self.heatmap_layer = nn.Sequential(
                build_linear_layer(middle_channels, heatmap_channels, bias=False, norm=norm, act=act),
                build_linear_layer(heatmap_channels, self.shape_class, bias=False),
            )
            
            if self.query_init == 'sparse_heatmap':
                self.voxelization = MODULES.build(query_generator['voxelization'])
                self.sparse_encoder = MODULES.build(query_generator['sparse_encoder'])
            
        ###################  query_centers module  #################
        self.num_stage = query_encoder.pop('num_stage')
        
        self.query_module = nn.ModuleList()
        for i in range(self.num_stage):
            self.query_module.append(QureyTransformer(**query_encoder, channels=middle_channels))

        aggregation_channels = middle_channels*self.num_stage
        self.query_aggregation = build_linear_layer(middle_channels*self.num_stage, aggregation_channels, norm=norm, act=act)
        self.aggregation = build_linear_layer(middle_channels*self.num_stage, aggregation_channels, norm=norm, act=act)
        
    def forward(self, input_dict):
        points = input_dict['points'] # b, c, n
        feats = self.input_encoder(points, points)
        b, c, n = feats.shape
        
        # feats = input_dict['feats']
        
        if self.query_init == 'heatmap':
            batch_size, c, n = feats.shape
            heatmap = self.heatmap_layer(feats.transpose(1,2).reshape(batch_size*n, c)).reshape(batch_size, n, self.shape_class)
            heatmap_preds = heatmap.mean(1) # b,40
            heatmap_labels = heatmap_preds.max(-1)[1]
            batch_indices = torch.arange(batch_size, dtype=torch.long).to(heatmap.device)
            query_heatmap = heatmap[batch_indices, :, heatmap_labels]
            query_index = torch.argsort(F.log_softmax(query_heatmap, -1), -1, descending=True)[:, :self.num_queries] # b, num_queries

            query = gather_points(points, query_index)
            query_feats = gather_points(feats, query_index)
            radius_list = [0.2, 0.4]
        
        elif self.query_init == 'sparse_heatmap':
            p = points[:, :3, :]# to [0, 1]
        
            voxel_dict = self.voxelization(torch.cat([p, feats]))
            batch_size = voxel_dict['batch_size']
            
            sparse_feats = self.sparse_encoder(voxel_dict)
            spatial_shape = sparse_feats.spatial_shape # z, y, x
            batch_features = sparse_feats.features # n, c
            batch_indices = sparse_feats.indices
            
            grid_size = spatial_shape[[2, 1, 0]]
            voxel_size = feats.new_tensor([1, 1, 1]) / grid_size.float()
            
            batch_ids = batch_indices[..., 0]
            batch_zyx = batch_indices[..., 1:]
            batch_xyz = batch_zyx[:, [2, 1, 0]]
            batch_grids = (batch_xyz + 0.5) * voxel_size[None]
     
            batch_heatmap = self.heatmap_layer(batch_features).view(-1) # n
            query_feats = torch.zeros([b, c, self.num_queries]).to(feats.device)
            query = torch.zeros([b, 3, self.num_queries]).to(feats.device)
            for i in range(b):
                mask = batch_ids==i
                features = batch_features[mask]
                heatmap = batch_heatmap[mask]
                grids = batch_grids[mask]
                
                heatmap, heatmap_index = torch.sort(heatmap, descending=True)
                # query_heatmap = heatmap[:self.num_queries] 
                query_index = heatmap_index[:self.num_queries] 
                query_feats[i, ...] = gather_points(features, query_index)
                query[i, ...] = gather_points(grids, query_index)
            
            # grid_size = spatial_shape[[2, 1, 0]]
            # voxel_size = grid_size.new_tensor([1, 1, 1]) / grid_size
            # grids = (dense_xyz+0.5) * voxel_size[None] + point_cloud_range[:3][None]
            
            
            # dense_feats = sparse_feats.dense()
            # B, C, D, H, W = dense_feats.shape  
            
            # point_cloud_range = voxel_dict['point_cloud_range']
            

            
            # dense_feats = sparse_feats.dense()
            # B, C, D, H, W = dense_feats.shape   
            # dense_feats = dense_feats.reshape(B, C, -1)
            # x, y, z = torch.meshgrid([
            #     torch.arange(0, W),
            #     torch.arange(0, H),
            #     torch.arange(0, D),
            # ])
            # x = x.reshape(-1, 1) # 1, m
            # y = y.reshape(-1, 1)
            # z = z.reshape(-1, 1)
            # dense_xyz = torch.cat([x, y, z], -1).cuda()
            # spatial_shape = feats.new_tensor(sparse_feats.spatial_shape)
            # grid_size = spatial_shape[[2, 1, 0]]
            # voxel_size = (point_cloud_range[3:] - point_cloud_range[:3]) / grid_size
            # dense_centers = (dense_xyz+0.5) * voxel_size[None] + point_cloud_range[:3][None]
            # dense_centers = dense_centers[None].repeat(B, 1, 1).transpose(1, 2)
            
            # heatmap = self.heatmap_layer(dense_feats.transpose(1, 2).reshape(-1, C)).reshape(B, D*H*W, -1).transpose(1,2) # b, 2, n
            # query_heatmap = F.softmax(heatmap.view(B,-1), -1)
            # query_index = torch.argsort(query_heatmap, -1, descending=True)[:, :self.num_queries] # b, num_queries
            # query_feats = gather_points(dense_feats, query_index)
            # query = gather_points(dense_centers, query_index)
            
            radius = torch.sqrt((voxel_size**2).sum())
            radius_list = [radius, 2*radius]
        
        feats = self.middle_encoder(feats)

        grouped_index_list = []
        for i, nsample in enumerate(self.num_sample):
            grouped_index = ball_points(radius_list[i], nsample, points[:,:3,:], query) # b, n, s
            grouped_index_list.append(grouped_index)

        feats_list = []
        query_feats_list = []
        
        for i, layer in enumerate(self.query_module):
            query_feats, feats = layer(query_feats, feats, grouped_index_list)
            query_feats_list.append(query_feats)
            feats_list.append(feats)
        
        query_feats = torch.cat(query_feats_list, 1)
        query_feats = self.query_aggregation(query_feats.transpose(1,2).reshape(batch_size*self.num_queries, -1))
        query_feats = query_feats.reshape(batch_size, self.num_queries, -1).transpose(1,2) 
        
        feats = torch.cat(feats_list, 1)
        num_feats = feats.shape[-1]
        feats = self.aggregation(feats.transpose(1,2).reshape(batch_size*num_feats, -1))
        feats = feats.reshape(batch_size, num_feats, -1).transpose(1,2) 

        results = {
            'query': query,
            'query_feats': query_feats,
            'query_feats_list': query_feats_list,
            'points': points[:,:3,:],
            'points_feats': feats,
            'points_feats_list': feats_list,
            'heatmap_preds': heatmap_preds
        }

        return results
    
        