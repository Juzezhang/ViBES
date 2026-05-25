import torch
import torch.nn.functional as F
import numpy as np
from einops import rearrange
from scipy.ndimage._filters import _gaussian_kernel1d
from .rotation_conversions import axis_angle_to_matrix, matrix_to_axis_angle, rotation_6d_to_matrix, matrix_to_rotation_6d

def identity_mat(x=None, device="cpu", is_numpy=False):
    if x is not None:
        if isinstance(x, torch.Tensor):
            mat = torch.eye(4, device=device)
            mat = mat.repeat(x.shape[:-2] + (1, 1))
        elif isinstance(x, np.ndarray):
            mat = np.eye(4, dtype=np.float32)
            if x is not None:
                for _ in range(len(x.shape) - 2):
                    mat = mat[None]
            mat = np.tile(mat, x.shape[:-2] + (1, 1))
        else:
            raise ValueError
    else:
        if is_numpy:
            mat = np.eye(4, dtype=np.float32)
        else:
            mat = torch.eye(4, device=device)
    return mat

def get_TRS(rot_mat, pos):
    """
    Args:
        rot_mat: (*, 3, 3)
        pos: (*, 3)
    Returns:
        mat: (*, 4, 4)
    """
    mat = identity_mat(rot_mat, device=rot_mat.device)
    mat[..., :3, :3] = rot_mat
    mat[..., :3, 3] = pos
    return mat

def forward_kinematics(mat, parent):
    """
    Args:
        mat: (*, J, 4, 4)
        parent: list of J integers
    Returns:
        fk_mat: (*, J, 4, 4)
    """
    fk_mat = [None] * len(parent)
    for i, p in enumerate(parent):
        if p == -1:
            fk_mat[i] = mat[..., i, :, :]
        else:
            fk_mat[i] = torch.matmul(fk_mat[p], mat[..., i, :, :])
    return torch.stack(fk_mat, dim=-3)

def get_position(mat):
    return mat[..., :3, 3]

def get_rotation(mat):
    return mat[..., :3, :3]

def get_local_transl_vel(transl, global_orient):
    assert len(transl.shape) == len(global_orient.shape)
    global_orient_R = axis_angle_to_matrix(global_orient)
    transl_vel = transl[..., 1:, :] - transl[..., :-1, :]
    transl_vel = torch.cat([transl_vel, transl_vel[..., [-1], :]], dim=-2)
    local_transl_vel = torch.einsum("...lij,...li->...lj", global_orient_R, transl_vel)
    return local_transl_vel

def rollout_local_transl_vel(local_transl_vel, global_orient, transl_0=None):
    global_orient_R = axis_angle_to_matrix(global_orient)
    transl_vel = torch.einsum("...lij,...lj->...li", global_orient_R, local_transl_vel)
    if transl_0 is None:
        transl_0 = transl_vel[..., :1, :].clone().detach().zero_()
    transl_ = torch.cat([transl_0, transl_vel[..., :-1, :]], dim=-2)
    transl = torch.cumsum(transl_, dim=-2)
    return transl

def gaussian_smooth(x, sigma=3, dim=-1):
    kernel_smooth = _gaussian_kernel1d(sigma=sigma, order=0, radius=int(4 * sigma + 0.5))
    kernel_smooth = torch.from_numpy(kernel_smooth).float()[None, None].to(x)
    rad = kernel_smooth.size(-1) // 2
    x = x.transpose(dim, -1)
    x_shape = x.shape[:-1]
    x = rearrange(x, "... f -> (...) 1 f")
    x = F.pad(x[None], (rad, rad, 0, 0), mode="replicate")[0]
    x = F.conv1d(x, kernel_smooth)
    x = x.squeeze(1).reshape(*x_shape, -1)
    x = x.transpose(-1, dim)
    return x

def gaussian_augment(body_pose, std_angle=10.0, to_R=True):
    body_pose = body_pose.clone()
    if to_R:
        body_pose_R = axis_angle_to_matrix(body_pose)
    else:
        body_pose_R = body_pose
    shape = body_pose_R.shape[:-2]
    device = body_pose.device
    std_angle = torch.tensor(std_angle).to(device).reshape(-1)
    noise_angle = torch.randn(shape, device=device) * std_angle * torch.pi / 180
    noise_axis = torch.rand((*shape, 3), device=device)
    mask_ = torch.norm(noise_axis, dim=-1) < 1e-6
    noise_axis[mask_] = 1
    noise_axis = noise_axis / torch.norm(noise_axis, dim=-1, keepdim=True)
    noise_aa = noise_angle[..., None] * noise_axis
    noise_R = axis_angle_to_matrix(noise_aa)
    new_body_pose_R = torch.matmul(noise_R, body_pose_R)
    new_body_pose_r6d = matrix_to_rotation_6d(new_body_pose_R)
    new_body_pose_aa = matrix_to_axis_angle(new_body_pose_R)
    return new_body_pose_R, new_body_pose_r6d, new_body_pose_aa

def transform_mat(R: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    T = torch.eye(4).to(R.device).repeat(R.shape[:-2] + (1, 1))
    T[..., :3, :3] = R
    T[..., :3, 3] = t
    return T

def apply_T_on_points(points, T):
    if len(T.shape) == len(points.shape):
        return torch.einsum("...ij,...j->...i", T[..., :3, :3], points) + T[..., :3, 3]
    else:
        return torch.einsum("...ij,...nj->...ni", T[..., :3, :3], points) + T[..., :3, 3]
