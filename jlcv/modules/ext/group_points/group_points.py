import torch
from torch import nn as nn
from torch.autograd import Function
from typing import Tuple

from . import group_points_ext

class GroupPoints(Function):
    """Grouping Operation.

    Group feature with given index.
    """

    @staticmethod
    def forward(ctx, features: torch.Tensor,
                indices: torch.Tensor) -> torch.Tensor:
        """forward.

        Args:
            features (Tensor): (B, C, N) tensor of features to group.
            indices (Tensor): (B, npoint, nsample) the indicies of
                features to group with.

        Returns:
            Tensor: (B, C, npoint, nsample) Grouped features.
        """
        if not features.is_contiguous():
            features = features.contiguous()
        if not indices.is_contiguous():
            indices = indices.contiguous()
        
        indices = indices.int()

        B, nfeatures, nsample = indices.size()
        _, C, N = features.size()
        output = torch.cuda.FloatTensor(B, C, nfeatures, nsample)

        group_points_ext.forward(B, C, N, nfeatures, nsample, features,
                                 indices, output)

        ctx.for_backwards = (indices, N)
        return output

    @staticmethod
    def backward(ctx,
                 grad_out: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """backward.

        Args:
            grad_out (Tensor): (B, C, npoint, nsample) tensor of the gradients
                of the output from forward.

        Returns:
            Tensor: (B, C, N) gradient of the features.
        """
        idx, N = ctx.for_backwards

        B, C, npoint, nsample = grad_out.size()
        grad_features = torch.cuda.FloatTensor(B, C, N).zero_()

        grad_out_data = grad_out.data.contiguous()
        group_points_ext.backward(B, C, N, npoint, nsample, grad_out_data, idx,
                                  grad_features.data)
        return grad_features, None


group_points = GroupPoints.apply
