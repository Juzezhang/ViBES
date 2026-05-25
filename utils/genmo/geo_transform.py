"""Minimal geometry transforms used by GENMO render scripts."""
# Adapted from askhan1 GENMO utilities
# Simplified for local rendering utilities.
from __future__ import annotations

import torch
import torch.nn.functional as F

from utils.genmo.pylogger import Log


def apply_T_on_points(points: torch.Tensor, T: torch.Tensor) -> torch.Tensor:
    """
    Apply homogeneous transform to points.

    Args:
        points: (..., N, 3)
        T: (..., 4, 4)
    Returns:
        (..., N, 3)
    """
    return torch.einsum("...ki,...ji->...jk", T[..., :3, :3], points) + T[..., None, :3, 3]


def transform_mat(R: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """Build a 4x4 transform from rotation and translation."""
    bs = R.shape[0]
    T = torch.zeros((bs, 4, 4), device=R.device, dtype=R.dtype)
    T[:, :3, :3] = R
    T[:, :3, 3] = t
    T[:, 3, 3] = 1.0
    return T


def compute_T_ayfz2ay(joints: torch.Tensor, inverse: bool = False) -> torch.Tensor:
    """
    Args:
        joints: (B, J, 3), in the start-frame, ay-coordinate
    Returns:
        if inverse == False:
            T_ayfz2ay: (B, 4, 4)
        else:
            T_ay2ayfz: (B, 4, 4)
    """
    t_ayfz2ay = joints[:, 0, :].detach().clone()
    t_ayfz2ay[:, 1] = 0  # do not modify y

    # hips + shoulders define facing direction in xz plane
    RL_xz_h = joints[:, 1, [0, 2]] - joints[:, 2, [0, 2]]  # (B, 2)
    RL_xz_s = joints[:, 16, [0, 2]] - joints[:, 17, [0, 2]]  # (B, 2)
    RL_xz = RL_xz_h + RL_xz_s
    I_mask = RL_xz.pow(2).sum(-1) < 1e-4
    if I_mask.sum() > 0:
        Log.warn(f"{int(I_mask.sum())} samples can't decide the face direction")

    x_dir = torch.zeros_like(t_ayfz2ay)
    x_dir[:, [0, 2]] = F.normalize(RL_xz, 2, -1)
    y_dir = torch.zeros_like(x_dir)
    y_dir[..., 1] = 1
    z_dir = torch.cross(x_dir, y_dir, dim=-1)
    R_ayfz2ay = torch.stack([x_dir, y_dir, z_dir], dim=-1)
    R_ayfz2ay[I_mask] = torch.eye(3, device=R_ayfz2ay.device, dtype=R_ayfz2ay.dtype)

    if inverse:
        R_ay2ayfz = R_ayfz2ay.transpose(1, 2)
        t_ay2ayfz = -torch.einsum("bij,bi->bj", R_ayfz2ay, t_ayfz2ay)
        return transform_mat(R_ay2ayfz, t_ay2ayfz)
    return transform_mat(R_ayfz2ay, t_ayfz2ay)
