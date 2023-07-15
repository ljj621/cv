import torch
from torch.autograd import Function
from . import knn_ext


class KNN(Function):
    """KNN (CUDA).

    Find k-nearest points.
    """

    @staticmethod
    def forward(ctx,
                k: int,
                xyz: torch.Tensor,
                center_xyz: torch.Tensor) -> torch.Tensor:
        """forward.

        Args:
            k (int): number of nearest neighbors.
            xyz (Tensor): (B, 3, N) xyz coordinates of the features.
            center_xyz (Tensor): (B, 3, npoint) centers of the knn query.

        Returns:
            Tensor: (B, npoint, k) tensor with the indicies of
                the features that form k-nearest neighbours.
        """
        assert k > 0
        
        B, _, npoint = center_xyz.shape
        N = xyz.shape[2]
        center_xyz = center_xyz.contiguous() if not center_xyz.is_contiguous() else center_xyz
        xyz = xyz.contiguous() if not xyz.is_contiguous() else xyz

        center_xyz_device = center_xyz.get_device()
        assert center_xyz_device == xyz.get_device(), \
            'center_xyz and xyz should be put on the same device'
        if torch.cuda.current_device() != center_xyz_device:
            torch.cuda.set_device(center_xyz_device)

        idx = center_xyz.new_zeros((B, k, npoint)).long()

        for bi in range(B):
            knn_ext.knn_wrapper(xyz[bi], N, center_xyz[bi], npoint, idx[bi], k)

        ctx.mark_non_differentiable(idx)

        idx -= 1
        idx = idx.transpose(1, 2)

        return idx

    @staticmethod
    def backward(ctx, a=None):
        return None, None


knn = KNN.apply
