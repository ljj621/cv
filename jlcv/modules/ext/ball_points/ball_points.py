import torch
from torch.autograd import Function

from . import ball_points_ext


class BallQuery(Function):
    """Ball Query.
    Find nearby points in spherical space.
    """
    @staticmethod
    def forward(ctx, radius, num_sample: int,
                xyz: torch.Tensor, center_xyz: torch.Tensor) -> torch.Tensor:
        """forward.

        Args:
            radius (float | list): radius of the balls.
            num_sample (int): maximum number of features in the balls.
            xyz (Tensor): (B, 3, N) xyz coordinates of the features.
            center_xyz (Tensor): (B, 3, npoint) centers of the ball query.

        Returns:
            Tensor: (B, npoint, nsample) tensor with the indicies of
                the features that form the query balls.
        """
        xyz = xyz.transpose(1, 2)
        if not xyz.is_contiguous():
            xyz = xyz.contiguous()
            
        center_xyz = center_xyz.transpose(1, 2)
        if not center_xyz.is_contiguous():
            center_xyz = center_xyz.contiguous()
        
        if isinstance(radius, list):
            min_radius, max_radius = radius
        else: # float
            min_radius, max_radius = 0, radius
        
        B, N, _ = xyz.size()
        npoint = center_xyz.size(1)
        idx = torch.cuda.IntTensor(B, npoint, num_sample).zero_()
            
        ball_points_ext.ball_query_wrapper(B, N, npoint, min_radius, max_radius,
                                          num_sample, center_xyz, xyz, idx)
        ctx.mark_non_differentiable(idx)
        return idx

    @staticmethod
    def backward(ctx, a=None):
        return None, None, None, None


ball_points = BallQuery.apply