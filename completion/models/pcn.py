import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
import torch.nn.functional as F
import math

def gen_grid_up(up_ratio, grid_size=0.2):
    sqrted = int(math.sqrt(up_ratio)) + 1
    for i in range(1, sqrted + 1).__reversed__():
        if (up_ratio % i) == 0:
            num_x = i
            num_y = up_ratio // i
            break

    grid_x = torch.linspace(-grid_size, grid_size, steps=num_x)
    grid_y = torch.linspace(-grid_size, grid_size, steps=num_y)

    x, y = torch.meshgrid(grid_x, grid_y)  # x, y shape: (2, 1)
    grid = torch.stack([x, y], dim=-1).view(-1, 2).transpose(0, 1).contiguous()
    return grid

class PCNEncoder(nn.Module):
    def __init__(self, output_size=1024):
        super(PCNEncoder, self).__init__()
        self.conv1 = nn.Conv1d(3, 128, 1)
        self.conv2 = nn.Conv1d(128, 256, 1)
        self.conv3 = nn.Conv1d(512, 512, 1)
        self.conv4 = nn.Conv1d(512, output_size, 1)

    def forward(self, x):
        batch_size, _, num_points = x.size()
        x = F.relu(self.conv1(x))
        x = self.conv2(x)
        global_feature, _ = torch.max(x, 2)
        x = torch.cat((x, global_feature.view(batch_size, -1, 1).repeat(1, 1, num_points).contiguous()), 1)
        x = F.relu(self.conv3(x))
        x = self.conv4(x)
        global_feature, _ = torch.max(x, 2)
        return global_feature.view(batch_size, -1)

class PCNDecoder(nn.Module):
    def __init__(self, num_coarse, num_fine, grid_size=0.2, grid_step=4):
        super(PCNDecoder, self).__init__()
        self.num_coarse = num_coarse
        self.num_fine = num_fine
        self.fc1 = nn.Linear(1024, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.fc3 = nn.Linear(1024, num_coarse * 3)

        grid_x = torch.linspace(-grid_size, grid_size, steps=grid_step)
        grid_y = torch.linspace(-grid_size, grid_size, steps=grid_step)
        x, y = torch.meshgrid(grid_x, grid_y)  # x, y shape: (2, 1)
        self.grid = torch.stack([x, y], dim=-1).view(-1, 2).transpose(0, 1).contiguous()

        self.conv1 = nn.Conv1d(2 + 3 + 1024, 512, 1)
        self.conv2 = nn.Conv1d(512, 512, 1)
        self.conv3 = nn.Conv1d(512, 3, 1)

    def forward(self, x):
        batch_size = x.size()[0]
        coarse = F.relu(self.fc1(x))
        coarse = F.relu(self.fc2(coarse))
        coarse = self.fc3(coarse).view(-1, 3, self.num_coarse)

        grid = self.grid.to(x.device).clone().detach()
        grid_feat = grid.unsqueeze(0).repeat(batch_size, 1, self.num_coarse).contiguous().cuda()

        point_feat = (
            (coarse.transpose(1, 2).contiguous()).unsqueeze(2).repeat(1, 1, self.scale, 1).view(-1, self.num_fine,3)).transpose(1,2).contiguous()

        global_feat = x.unsqueeze(2).repeat(1, 1, self.num_fine)

        feat = torch.cat((grid_feat, point_feat, global_feat), 1)

        center = ((coarse.transpose(1, 2).contiguous()).unsqueeze(2).repeat(1, 1, self.scale, 1).view(-1, self.num_fine,3)).transpose(1,2).contiguous()

        fine = self.conv3(F.relu(self.conv2(F.relu(self.conv1(feat))))) + center
        return coarse, fine


class PCN(nn.Module):
    def __init__(self, config, num_coarse=1024):
        super(PCN, self).__init__()

        self.num_coarse = num_coarse
        self.num_points = config.num_points
        self.scale = self.num_points // num_coarse
        self.cat_feature_num = 2 + 3 + 1024

        self.encoder = PCNEncoder()
        self.decoder = PCNDecoder(num_coarse, self.num_points, 0.05, self.scale)
        self.metric_loss = config.metric_loss

    def forward(self, x):
        feat = self.encoder(x)
        coarse_cloud, dense_cloud = self.decoder(feat)
        coarse_cloud = coarse_cloud.transpose(1, 2).contiguous()
        dense_cloud = dense_cloud.transpose(1, 2).contiguous()
        preds_dict = {'coarse': coarse_cloud, 'preds': dense_cloud}
        return preds_dict
    
    

  