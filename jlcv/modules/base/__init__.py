import copy
import torch.nn as nn
from ..ext.spconv import SparseModule, SparseSequential
from .activation import ACTIVATION
from .cnn import CONV
from .norm import NORM

def build_conv_layer(
        in_channels,
        out_channels,
        kernel_size,
        conv='Conv1d',
        norm=None,
        act=None,
        **kwargs):
    conv_layer = CONV.build(conv, in_channels, out_channels, kernel_size, **kwargs)
    if norm is None and act is None:
        return conv_layer

    is_sparse = isinstance(conv_layer, SparseModule)
    layers = [conv_layer]
    
    if norm is not None:
        norm_dict = copy.deepcopy(norm)
        if isinstance(norm_dict, str):
            norm_dict = dict(type=norm_dict)
        if 'Sparse' in conv:
            norm_dict['type'] = 'BN1d'
        elif norm_dict['type'] in ['BN', 'IN']:
            norm_dict['type'] = norm_dict['type'] + conv[-2:]
        layer = NORM.build(norm_dict, out_channels)
        layers.append(layer)

    if act is not None:
        layer = ACTIVATION.build(act)
        layers.append(layer)
    
    if is_sparse:
        return SparseSequential(*layers)
    else:
        return nn.Sequential(*layers)

def build_linear_layer(in_channels, out_channels, norm=None, act=None, drop=None,**kwargs):
    linear_layer = nn.Linear(in_channels, out_channels, **kwargs)
    if norm is None and act is None:
        return linear_layer
    
    layers = [linear_layer]
    
    if norm is not None:
        norm_dict = copy.deepcopy(norm)
        if isinstance(norm_dict, str):
            norm_dict = dict(type=norm_dict)
        if norm_dict['type'] in ['BN', 'IN']:
            norm_dict['type'] = norm_dict['type'] + '1d'
            layer = NORM.build(norm_dict, out_channels)
            layers.append(layer)

    if act is not None:
        layer = ACTIVATION.build(act)
        layers.append(layer)
    
    if drop is not None:
        layers.append(nn.Dropout(drop))

    return nn.Sequential(*layers)

__all__ = [
    'build_conv_layer', 'build_linear_layer'
]
