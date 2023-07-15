import torch
import torch.nn as nn
import torch.nn.functional as F
from jlcv.modules import build_conv_layer
from jlcv.models import MODELS

@MODELS.register_module()
class DownSampleLayer(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 conv_cfg='Conv2d', 
                 norm_cfg='BN', 
                 act_cfg='ReLU') -> None:
        super().__init__()
        self.conv = nn.Sequential(
                build_conv_layer(conv_cfg, in_channels, out_channels, 3, 1, 1, norm_cfg=norm_cfg, act_cfg=act_cfg),
                build_conv_layer(conv_cfg, out_channels, out_channels, 3, 1, 1, norm_cfg=norm_cfg, act_cfg=act_cfg))
        self.downsample = build_conv_layer(conv_cfg, out_channels, out_channels, 3, 2, 1, norm_cfg=norm_cfg, act_cfg=act_cfg)
    def forward(self, x):
        x = self.conv(x)
        downsample = self.downsample(x)
        return x, downsample

class UpSampleLayer(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 conv_cfg='Conv2d', 
                 norm_cfg='BN', 
                 act_cfg='ReLU') -> None:
        super().__init__()
        self.layer = nn.Sequential(
                build_conv_layer(conv_cfg, in_channels, out_channels*2, 3, 1, 1, norm_cfg=norm_cfg, act_cfg=act_cfg),
                build_conv_layer(conv_cfg, out_channels*2, out_channels*2, 3, 1, 1, norm_cfg=norm_cfg, act_cfg=act_cfg),
                build_conv_layer('ConvTranspose2d', out_channels*2, out_channels, 3, 2, 1, norm_cfg=norm_cfg, act_cfg=act_cfg))
    def forward(self, x, feats):
        x = self.layer(x)
        x = torch.cat([x, feats], 1)
        return x


class Unet(nn.Module):
    def __init__(self, 
                 channels=[3, 64, 128, 256, 512], 
                 segmentation_classes = None,
                 conv_cfg='Conv2d', 
                 norm_cfg='BN', 
                 act_cfg='ReLU'):
        super().__init__()
        num_layers = len(channels) - 1
        self.num_layers = num_layers
    
        # downsample
        self.downsample = nn.ModuleList()
        for i in range(num_layers):
            self.downsample.append(DownSampleLayer(
                channels[i], 
                channels[i+1], 
                conv_cfg, 
                norm_cfg, 
                act_cfg))
        
        self.upsample = nn.ModuleList()
        for i in range(num_layers, -1, -1):
            if i == len(num_layers) - 1:
                in_channels = channels[i]
                out_channels = channels[i]
            else:
                in_channels = channels[i+1]
                out_channels = channels[i]
            self.upsample.append(UpSampleLayer(
                in_channels*2, 
                out_channels, 
                conv_cfg, 
                norm_cfg, 
                act_cfg))
        if segmentation_classes is not None:
            self.segmentation = nn.Sequential(
                build_conv_layer(conv_cfg, channels[1], channels[0], 3, 1, 1, norm_cfg=norm_cfg, act_cfg=act_cfg),
                build_conv_layer(conv_cfg, channels[1], channels[0], 3, 1, 1, norm_cfg=norm_cfg, act_cfg=act_cfg),
                build_conv_layer(conv_cfg, channels[0], segmentation_classes, 3, 1, 1),
                nn.Sigmoid())
        else:
            self.segmentation = None
    def forward(self, x):
        downsample = []
        feats = []
        for i, layer in enumerate(self.downsample):
            f, x = layer(x)
            feats.append(f)
            downsample.append(x)
            
        upsample = []
        for i, layer in enumerate(self.upsample):
            x = layer(x, feats[self.num_layers - i - 1])
            upsample.append(x)
        
        if self.segmentation is not None:
            x = self.segmentation(x)
        return x, downsample, upsample
        




        



            

       
            


