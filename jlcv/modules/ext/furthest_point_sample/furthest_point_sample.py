import torch
import torch.nn as nn
from torch.autograd import Function
from . import furthest_point_sample_ext

class FurthestPointSampling(Function):
    """Furthest Point Sampling.

    Uses iterative furthest point sampling to select a set of features whose
    corresponding points have the furthest distance.
    """

    @staticmethod
    def forward(ctx, points_xyz: torch.Tensor,
                num_points: int) -> torch.Tensor:
        """forward.

        Args:
            points_xyz (Tensor): (B, 3, N) where N > num_points.
            num_points (int): Number of points in the sampled set.

        Returns:
             Tensor: (B, num_points) indices of the sampled points.
        """
        
        points_xyz = points_xyz.transpose(1, 2)
        if not points_xyz.is_contiguous():
            points_xyz = points_xyz.contiguous()
            
        B, N = points_xyz.size()[:2]
        output = torch.cuda.IntTensor(B, num_points)
        temp = torch.cuda.FloatTensor(B, N).fill_(1e10)

        furthest_point_sample_ext.furthest_point_sampling_wrapper(
            B, N, num_points, points_xyz, temp, output)
        ctx.mark_non_differentiable(output)
        return output

    @staticmethod
    def backward(xyz, a=None):
        return None, None

furthest_point_sample = FurthestPointSampling.apply
