import torch
import torch.nn as nn
import torch.nn.functional as F
from .rotation_conversions import (
    matrix_to_rotation_6d,
    rotation_6d_to_matrix,
    axis_angle_to_matrix,
    matrix_to_axis_angle,
    matrix_to_quaternion,
)
from . import vendor_utils as matrix
from .endecoder import EnDecoder
from .vendor_utils import gaussian_smooth, transform_mat, apply_T_on_points

# Quaternion utilities (ported from SAMPA utils/common/quaternion.py)
from .quaternion import qbetween, qslerp, qinv, qmul, qrot

class CCD_IK:
    def __init__(
        self,
        local_mat,
        parent,
        target_ind,
        target_pos=None,
        target_rot=None,
        kinematic_chain=None,
        max_iter=2,
        threshold=0.001,
        pos_weight=1.0,
        rot_weight=0.0,
    ):
        if kinematic_chain is None:
            kinematic_chain = range(local_mat.shape[-3])
        global_mat = matrix.forward_kinematics(local_mat, parent)

        local_mat = local_mat.clone()
        local_mat = local_mat[..., kinematic_chain, :, :]
        local_mat[..., 0, :, :] = global_mat[..., kinematic_chain[0], :, :]

        parent = [i - 1 for i in range(len(kinematic_chain))]
        self.local_mat = local_mat
        self.global_mat = matrix.forward_kinematics(local_mat, parent)
        self.parent = parent

        self.target_ind = target_ind
        self.target_pos = target_pos
        if target_rot is not None:
            self.target_q = matrix_to_quaternion(target_rot)
        else:
            self.target_q = None

        self.threshold = threshold
        self.J_N = self.local_mat.shape[-3]
        self.target_N = len(target_ind)
        self.max_iter = max_iter
        self.pos_weight = pos_weight
        self.rot_weight = rot_weight

    def solve(self):
        for _ in range(self.max_iter):
            self.optimize(1)
        return self.local_mat

    def optimize(self, i):
        if i == self.J_N - 1:
            return
        pos = matrix.get_position(self.global_mat)[..., i, :]
        rot = matrix.get_rotation(self.global_mat)[..., i, :, :]
        quat = matrix_to_quaternion(rot)
        x_vec = torch.zeros((quat.shape[:-1] + (3,)), device=quat.device)
        x_vec[..., 0] = 1.0
        y_vec = torch.zeros((quat.shape[:-1] + (3,)), device=quat.device)
        y_vec[..., 1] = 1.0

        x_vec_sum = torch.zeros_like(x_vec)
        y_vec_sum = torch.zeros_like(y_vec)

        for target_i, j in enumerate(self.target_ind):
            if i >= j:
                continue
            end_pos = matrix.get_position(self.global_mat)[..., j, :]

            if self.target_pos is not None:
                solved_pos_target_quat = qslerp(
                    quat,
                    qmul(qbetween(end_pos - pos, self.target_pos[..., target_i, :] - pos), quat),
                    self.pos_weight,
                )
                x_vec_sum += qrot(solved_pos_target_quat, x_vec)
                y_vec_sum += qrot(solved_pos_target_quat, y_vec)

            if self.target_q is not None and self.rot_weight > 0:
                end_rot = matrix.get_rotation(self.global_mat)[..., j, :, :]
                end_quat = matrix_to_quaternion(end_rot)
                solved_q_target_quat = qslerp(
                    quat,
                    qmul(qmul(self.target_q[..., target_i, :], qinv(end_quat)), quat),
                    self.rot_weight,
                )
                x_vec_sum += qrot(solved_q_target_quat, x_vec) * self.rot_weight
                y_vec_sum += qrot(solved_q_target_quat, y_vec) * self.rot_weight

        # Update current joint i
        new_x = x_vec_sum / (x_vec_sum.norm(dim=-1, keepdim=True) + 1e-6)
        new_y = y_vec_sum / (y_vec_sum.norm(dim=-1, keepdim=True) + 1e-6)
        new_z = torch.cross(new_x, new_y, dim=-1)
        new_y = torch.cross(new_z, new_x, dim=-1)
        new_rot = torch.stack([new_x, new_y, new_z], dim=-1)

        parent_rot = matrix.get_rotation(self.global_mat)[..., self.parent[i], :, :]
        new_local_rot = torch.matmul(parent_rot.transpose(-1, -2), new_rot)
        self.local_mat[..., i, :3, :3] = new_local_rot

        # FK update child
        self.global_mat = matrix.forward_kinematics(self.local_mat, self.parent)
        self.optimize(i + 1)

def process_ik(outputs, endecoder):
    static_conf = outputs["static_conf_logits"].sigmoid()
    post_w_j3d, local_mat, post_w_mat = endecoder.fk_v2(**outputs["pred_smpl_params_global"], get_intermediate=True)

    joint_ids = [7, 10, 8, 11, 20, 21]
    post_target_j3d = post_w_j3d.clone()
    for i in range(1, post_w_j3d.size(1)):
        prev = post_target_j3d[:, i - 1, joint_ids]
        this = post_w_j3d[:, i, joint_ids]
        c_prev = static_conf[:, i - 1, :, None]
        post_target_j3d[:, i, joint_ids] = prev * c_prev + this * (1 - c_prev)

    global_rot = matrix.get_rotation(post_w_mat)
    parents = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19]
    left_leg_chain = [0, 1, 4, 7, 10]
    right_leg_chain = [0, 2, 5, 8, 11]
    left_hand_chain = [9, 13, 16, 18, 20]
    right_hand_chain = [9, 14, 17, 19, 21]

    def ik_solver_func(local_mat, target_pos, target_rot, target_ind, chain):
        local_mat = local_mat.clone()
        solver = CCD_IK(
            local_mat,
            parents,
            target_ind,
            target_pos,
            target_rot,
            kinematic_chain=chain,
            max_iter=2,
        )
        chain_local_mat = solver.solve()
        chain_rotmat = matrix.get_rotation(chain_local_mat)
        local_mat[:, :, chain[1:], :3, :3] = chain_rotmat[:, :, 1:]
        return local_mat

    local_mat = ik_solver_func(local_mat, post_target_j3d[:, :, [7, 10]], global_rot[:, :, [7, 10]], [3, 4], left_leg_chain)
    local_mat = ik_solver_func(local_mat, post_target_j3d[:, :, [8, 11]], global_rot[:, :, [8, 11]], [3, 4], right_leg_chain)
    local_mat = ik_solver_func(local_mat, post_target_j3d[:, :, [20]], global_rot[:, :, [20]], [4], left_hand_chain)
    local_mat = ik_solver_func(local_mat, post_target_j3d[:, :, [21]], global_rot[:, :, [21]], [4], right_hand_chain)

    body_pose = matrix_to_axis_angle(matrix.get_rotation(local_mat[:, :, 1:]))
    body_pose = body_pose.flatten(2)

    return body_pose

def pp_static_joint(outputs, endecoder: EnDecoder):
    # Simplified version of static joint postprocessing
    pred_w_j3d = endecoder.fk_v2(**outputs["pred_smpl_params_global"])
    L = pred_w_j3d.shape[1]
    joint_ids = [7, 10, 8, 11, 20, 21]
    pred_j3d_static = pred_w_j3d.clone()[:, :, joint_ids]

    pred_j_disp = pred_j3d_static[:, 1:] - pred_j3d_static[:, :-1]
    static_conf_logits = outputs["static_conf_logits"][:, :-1].clone()
    static_label_ = static_conf_logits > 0
    static_conf_logits = static_conf_logits.float() - (~static_label_ * 1e6)
    is_static = static_label_.sum(dim=-1) > 0

    pred_disp = pred_j_disp * static_conf_logits[..., None].softmax(dim=-2)
    pred_disp = pred_disp * is_static[..., None, None]
    pred_disp = pred_disp.sum(-2)

    pred_w_transl = outputs["pred_smpl_params_global"]["transl"].clone()
    pred_w_disp = pred_w_transl[:, 1:] - pred_w_transl[:, :-1]
    pred_w_disp_new = pred_w_disp - pred_disp
    post_w_transl = torch.cumsum(torch.cat([pred_w_transl[:, :1], pred_w_disp_new], dim=1), dim=1)

    post_w_transl[..., 0] = gaussian_smooth(post_w_transl[..., 0], dim=-1)
    post_w_transl[..., 2] = gaussian_smooth(post_w_transl[..., 2], dim=-1)

    post_w_j3d = pred_w_j3d - pred_w_transl.unsqueeze(-2) + post_w_transl.unsqueeze(-2)
    ground_y = post_w_j3d[..., 1].flatten(-2).min(dim=-1)[0]
    post_w_transl[..., 1] -= ground_y

    return post_w_transl
