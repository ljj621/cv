import torch
import torch.nn as nn
import torch.nn.functional as F
from jlcv.ops.base import build_conv_layer

class BasicBlock(nn.Module):
    def __init__(self, 
                 in_channels, 
                 hidden_channels,
                 out_channels,
                 stride=1,
                 conv_cfg='Conv2d', 
                 norm_cfg='BN', 
                 act_cfg='ReLU'):
        super().__init__()
        self.stride = stride
        self.conv1 = build_conv_layer(conv_cfg, in_channels, hidden_channels, 3, stride=stride, padding=1, norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.conv2 = build_conv_layer(conv_cfg, hidden_channels, out_channels, 3, stride=1, padding=1, norm_cfg=norm_cfg)
        if stride > 1 or in_channels != out_channels:
            self.downsample = build_conv_layer(conv_cfg, in_channels, out_channels, 1, stride=stride,padding=1, norm_cfg=norm_cfg)
        else:
            self.downsample = None

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.conv2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = out + identity
        out = F.relu(out)
        return out

class Bottleneck(nn.Module):
    def __init__(self, 
                 in_channels, 
                 hidden_channels,
                 out_channels,
                 stride=1,
                 conv_cfg='Conv2d', 
                 norm_cfg='BN', 
                 act_cfg='ReLU'):
        super().__init__()
        self.conv1 = build_conv_layer(conv_cfg, in_channels, hidden_channels, 1, stride=1, padding=1, norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.conv2 = build_conv_layer(conv_cfg, hidden_channels, hidden_channels, 3, stride=stride, padding=1,  norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.conv3 = build_conv_layer(conv_cfg, hidden_channels, out_channels, 1, stride=1, padding=1, norm_cfg=norm_cfg)

        if stride > 1 or in_channels != out_channels:
            self.downsample = build_conv_layer(conv_cfg, in_channels, out_channels, 1, stride=stride,padding=1, norm_cfg=norm_cfg)
        else:
            self.downsample = None

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = out + identity
        out = F.relu(out)
        return out


@BACKBONE.register_module()
class ResNet(nn.Module):
    modules = {
        18: BasicBlock,
        34: BasicBlock,
        50: Bottleneck,
        101: Bottleneck,
        152: Bottleneck
    }
    num_stages = {
        18: [2, 2, 2, 2],
        34: [3, 4, 6, 3],
        50: [3, 4, 6, 3],
        101: [3, 4, 23, 3],
        152: [3, 8, 36, 3]
    }
    expansions = {
        18: 1,
        34: 1,
        50: 4,
        101: 4,
        152: 4
    }
    def __init__(self, 
                 depth=50,
                 channels=[64, 128, 256, 512], 
                 conv_cfg='Conv2d', 
                 norm_cfg='BN', 
                 act_cfg='ReLU'):
        super().__init__()
        self.conv1 = build_conv_layer(conv_cfg, 3, channels[0], 7, stride=2, padding=3, norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        module = self.modules[depth]
        strides = [1, 2, 2, 2]
        num_stage = self.num_stages[depth]
        expansion = self.expansions[depth]
        self.layers = nn.ModuleList()
        for i in num_stage:
            layers = []
            stride = strides[i]
            in_channels = channels[i]
            hidden_channels = channels[i]
            out_channels = channels[i] * expansion
            for n in range(i):
                if n > 0:
                    in_channels = out_channels
                    stride = 1
                layers.append(module(in_channels, 
                                     hidden_channels, 
                                     out_channels, 
                                     stride, 
                                     conv_cfg, 
                                     norm_cfg, 
                                     act_cfg))
            self.layers.append(nn.Sequential(*layers))

    def forward(self, x):
        B, C, N = x.shape
        x = self.conv1(x)
        x = self.maxpool(x)

        out = []
        for i, layer in enumerate(self.layers):
            x = layer(x)
            out.append(x)
        return out
