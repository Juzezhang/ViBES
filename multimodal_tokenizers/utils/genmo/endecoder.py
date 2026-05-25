import torch
import torch.nn as nn
from .rotation_conversions import (
    rotation_6d_to_matrix,
    matrix_to_axis_angle,
    axis_angle_to_matrix,
    matrix_to_rotation_6d,
)
from .smplx_utils import make_smplx
from . import stats_compose
from . import vendor_utils as matrix
from .vendor_utils import get_local_transl_vel, rollout_local_transl_vel

class EnDecoder(nn.Module):
    def __init__(self, stats_name="DEFAULT_01", noise_pose_k=10):
        super().__init__()
        # Load mean, std
        if hasattr(stats_compose, stats_name):
            stats = getattr(stats_compose, stats_name)
        else:
            # Fallback or empty stats if not found
            stats = {"mean": [0.0], "std": [1.0]}

        self.register_buffer("mean", torch.tensor(stats.get("mean", [0.0])).float(), False)
        self.register_buffer("std", torch.tensor(stats.get("std", [1.0])).float(), False)

        # option
        self.noise_pose_k = noise_pose_k

        # smpl
        self.smplx_model = make_smplx("supermotion")
        parents = self.smplx_model.parents[:22]
        self.register_buffer("parents_tensor", parents, False)
        self.parents = parents.tolist()

    def fk_v2(self, body_pose, betas, global_orient=None, transl=None, get_intermediate=False):
        """
        Args:
            body_pose: (B, L, 63)
            betas: (B, L, 10)
            global_orient: (B, L, 3)
        Returns:
            joints: (B, L, 22, 3)
        """
        B, L = body_pose.shape[:2]
        if global_orient is None:
            global_orient = torch.zeros((B, L, 3), device=body_pose.device)
        aa = torch.cat([global_orient, body_pose], dim=-1).reshape(B, L, -1, 3)
        rotmat = axis_angle_to_matrix(aa)  # (B, L, 22, 3, 3)

        skeleton = self.smplx_model.get_skeleton(betas)[..., :22, :]  # (B, L, 22, 3)
        local_skeleton = skeleton - skeleton[:, :, self.parents_tensor]
        local_skeleton = torch.cat([skeleton[:, :, :1], local_skeleton[:, :, 1:]], dim=2)

        if transl is not None:
            local_skeleton[..., 0, :] += transl  # B, L, 22, 3

        mat = matrix.get_TRS(rotmat, local_skeleton)  # B, L, 22, 4, 4
        fk_mat = matrix.forward_kinematics(mat, self.parents)  # B, L, 22, 4, 4
        joints = matrix.get_position(fk_mat)  # B, L, 22, 3
        if not get_intermediate:
            return joints
        else:
            return joints, mat, fk_mat

    def decode(self, x_norm):
        """x_norm: (B, L, C)"""
        B, L, C = x_norm.shape
        x = (x_norm * self.std) + self.mean

        body_pose_r6d = x[:, :, :126]
        betas = x[:, :, 126:136]
        global_orient_r6d = x[:, :, 136:142]
        global_orient_gv_r6d = x[:, :, 142:148]
        local_transl_vel = x[:, :, 148:151]

        body_pose = matrix_to_axis_angle(rotation_6d_to_matrix(body_pose_r6d.reshape(B, L, -1, 6)))
        body_pose = body_pose.flatten(-2)
        global_orient_c = matrix_to_axis_angle(rotation_6d_to_matrix(global_orient_r6d))
        global_orient_gv = matrix_to_axis_angle(rotation_6d_to_matrix(global_orient_gv_r6d))

        output = {
            "body_pose": body_pose,
            "betas": betas,
            "global_orient": global_orient_c,
            "global_orient_gv": global_orient_gv,
            "local_transl_vel": local_transl_vel,
        }

        return output
