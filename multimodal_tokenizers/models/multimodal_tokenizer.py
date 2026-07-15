import numpy as np
import torch
# import time
from multimodal_tokenizers.config import instantiate_from_config
from multimodal_tokenizers.losses.lom import VAELosses
from multimodal_tokenizers.optimizers.loss_factory import get_loss_func
from multimodal_tokenizers.utils.rotation_conversions import rotation_6d_to_matrix, rotation_6d_to_axis_angle, matrix_to_axis_angle, matrix_to_rotation_6d, axis_angle_to_6d
from multimodal_tokenizers.utils.other_tools import velocity2position, estimate_linear_velocity
from .base import BaseModel
from omegaconf import OmegaConf

from multimodal_tokenizers.data.mixed_dataset.data_tools import (
    joints_list, 
    JOINT_MASK_FACE,
    JOINT_MASK_UPPER,
    JOINT_MASK_HANDS,
    JOINT_MASK_LOWER,
    JOINT_MASK_FULL,
    BEAT_SMPLX_JOINTS,
    BEAT_SMPLX_FULL,
    BEAT_SMPLX_FACE,
    BEAT_SMPLX_UPPER,
    BEAT_SMPLX_HANDS,
    BEAT_SMPLX_LOWER,
    BEAT_SMPLX_UPPER_LOWER,
    JOINT_MASK_JOINTS_6D,
    JOINT_MASK_FACE_6D,
    JOINT_MASK_HANDS_6D,
    JOINT_MASK_LOWER_6D,
    JOINT_MASK_UPPER_6D,
    JOINT_MASK_UPPER_LOWER_6D
)
import torch.nn.functional as F
from smplx import FLAME
from smplx.lbs import blend_shapes, batch_rigid_transform
import smplx

class MultimodalTokenizer(BaseModel):
    """
    Multimodal tokenizer model for VQ/VAE training.
    For face-only configs it uses FLAME parameters:
    - 6 head pose parameters (global_orient)
    - 6 jaw pose parameters (jaw_pose)
    - 100 expression parameters (expression)
    Total dimension: 112
    """

    # GENMO joint indices for body split (21 body joints, 0-indexed)
    GENMO_LOWER_BODY_JOINTS = [0, 1, 3, 4, 6, 7, 9, 10]  # pelvis, hips, knees, ankles, feet
    GENMO_UPPER_BODY_JOINTS = [2, 5, 8, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]  # spine, neck, head, shoulders, elbows, wrists, collars

    def __init__(self,
                 cfg,
                 modality_setup=None,
                 modality_tokenizer=None,
                 **kwargs):

        self.save_hyperparameters(ignore='datamodule', logger=False)
        super().__init__()
        self.hparams.stage = cfg.TRAIN.STAGE
        self.hparams.Selected_part = cfg.Selected_part
        self.cfg = cfg
        self.hparams.metrics_dict = cfg.METRIC.TYPE
        self.configure_metrics()

        # Get audio and motion parameters from modality_setup or dataset.
        modality_params = {}
        if isinstance(modality_setup, dict):
            modality_params = modality_setup.get("params", {})
        self.motion_fps = modality_params.get("motion_fps", OmegaConf.select(cfg, "DATASET.pose_fps"))
        if self.motion_fps is None:
            self.motion_fps = 25
        self.motion_down = modality_params.get("motion_down", OmegaConf.select(cfg, "DATASET.unit_length"))
        if self.motion_down is None:
            self.motion_down = 1
        # Initialize SMPLX model for metrics/visualization.
        smplx_num_betas = OmegaConf.select(cfg, "DATASET.SMPLX_NUM_BETAS")
        if smplx_num_betas is None:
            smplx_num_betas = 10 if cfg.DATASET.motion_representation == "genmo" else 300
        self.smplx_num_betas = int(smplx_num_betas)
        self.lower_joint_indices = self._get_joint_indices(BEAT_SMPLX_LOWER)
        self.upper_joint_indices = self._get_joint_indices(BEAT_SMPLX_UPPER)
        self.upper_lower_joint_indices = self._get_joint_indices(BEAT_SMPLX_UPPER_LOWER)
        self.smplx = smplx.create(str(cfg.DATASET.SMPLX_MODEL_DIR),
            model_type='smplx',
            gender='NEUTRAL_2020',
            use_face_contour=False,
            num_betas=self.smplx_num_betas,
            num_expression_coeffs=100,
            ext='npz',
            use_pca=False,
            ).eval()
        self.train_batch_size = cfg.TRAIN.BATCH_SIZE
        self.max_batch_size_smplx = 1024 ### 64

        # self.flame = FLAME(
        #     model_path, 
        #     num_expression_coeffs=100, 
        #     ext='pkl', 
        #     batch_size=self.batch_size * 64
        # ).eval()
        
        # # Instantiate face tokenizer only
        # if modality_tokenizer:
        #     for tokenizer in modality_tokenizer:
        #         if 'face' in tokenizer:  # Only load the face tokenizer
        #             setattr(self, tokenizer, instantiate_from_config(modality_tokenizer[tokenizer]))

        # Instantiate modality tokenizer
        for tokenizer in modality_tokenizer:
            setattr(self, tokenizer, instantiate_from_config(modality_tokenizer[tokenizer]))
            if tokenizer == 'vae_face':
                self.flame = FLAME(
                    str(cfg.DATASET.FLAME_MODEL_DIR),
                    num_expression_coeffs=100, 
                    ext='pkl', 
                    batch_size=self.max_batch_size_smplx
                    ).eval()

        stage =  getattr(cfg.TRAIN, 'STAGE', None)
        # Initialize face loss for VAE/VQ stages
        if 'vae' in self.hparams.stage or 'vq' in self.hparams.stage:
            # self.face_loss = get_loss_func("FaceLoss")
            self.upper_loss = get_loss_func("UpperLoss")
            self.lower_loss = get_loss_func("LowerLoss")
            self.global_loss = get_loss_func("GlobalLoss")
            self.face_loss = get_loss_func("FaceLoss")
            self.hand_loss = get_loss_func("HandLoss")
            # LAMBDA_VELOCITY scales root translation-velocity supervision in GlobalLoss.
            self.global_loss.vel_weight = getattr(cfg.LOSS, "LAMBDA_VELOCITY", 1.0)
            # LAMBDA_GLOBAL scales reconstructed root position supervision in GlobalLoss.
            self.global_loss.pos_weight = getattr(cfg.LOSS, "LAMBDA_GLOBAL", 1.0)
            # LAMBDA_ROOT_ROT scales root rotation (pelvis) in lower-body geodesic loss.
            self.lower_loss.root_weight = getattr(cfg.LOSS, "LAMBDA_ROOT_ROT", 1.0)
            # Geodesic loss for rotation reconstruction (used for genmo rotations)
            self.rec_loss = get_loss_func("GeodesicLoss")
            # Instantiate the losses
            self._losses = torch.nn.ModuleDict({
                split: VAELosses(cfg, stage)
                for split in ["losses_train", "losses_test", "losses_val"]
            })
        
        
    def inverse_selection_tensor(self, filtered_t, selection_array, n):
        selection_array = torch.from_numpy(selection_array).to(filtered_t.device)
        original_shape_t = torch.zeros((n, 165)).to(filtered_t.device)
        selected_indices = torch.where(selection_array == 1)[0]
        for i in range(n):
            original_shape_t[i, selected_indices] = filtered_t[i]
        return original_shape_t

    def inverse_selection_tensor_full_body(self, filtered_t_face, filtered_t_hand, filtered_t_lower, filtered_t_upper, selection_array_face, selection_array_hand, selection_array_lower, selection_array_upper, n):
        selection_array_face = torch.from_numpy(selection_array_face).to(filtered_t_face.device)
        selection_array_hand = torch.from_numpy(selection_array_hand).to(filtered_t_hand.device)
        selection_array_lower = torch.from_numpy(selection_array_lower).to(filtered_t_lower.device)
        selection_array_upper = torch.from_numpy(selection_array_upper).to(filtered_t_upper.device)
        original_shape_t = torch.zeros((n, 165)).to(filtered_t_face.device)
        selected_indices_face = torch.where(selection_array_face == 1)[0]
        selected_indices_hand = torch.where(selection_array_hand == 1)[0]
        selected_indices_lower = torch.where(selection_array_lower == 1)[0]
        selected_indices_upper = torch.where(selection_array_upper == 1)[0]
        for i in range(n):
            original_shape_t[i, selected_indices_face] = filtered_t_face[i] 
            original_shape_t[i, selected_indices_hand] = filtered_t_hand[i]
            original_shape_t[i, selected_indices_lower] = filtered_t_lower[i]
            original_shape_t[i, selected_indices_upper] = filtered_t_upper[i]
        return original_shape_t

    def inverse_selection_tensor_full_body_6D(self, filtered_t_face, filtered_t_hand, filtered_t_lower, filtered_t_upper, selection_array_face, selection_array_hand, selection_array_lower, selection_array_upper, n):
        selection_array_face = torch.from_numpy(selection_array_face).to(filtered_t_face.device)
        selection_array_hand = torch.from_numpy(selection_array_hand).to(filtered_t_hand.device)
        selection_array_lower = torch.from_numpy(selection_array_lower).to(filtered_t_lower.device)
        selection_array_upper = torch.from_numpy(selection_array_upper).to(filtered_t_upper.device)
        original_shape_t = torch.zeros((n, 330), device=filtered_t_face.device, dtype=filtered_t_face.dtype)
        selected_indices_face = torch.where(selection_array_face == 1)[0]
        selected_indices_hand = torch.where(selection_array_hand == 1)[0]
        selected_indices_lower = torch.where(selection_array_lower == 1)[0]
        selected_indices_upper = torch.where(selection_array_upper == 1)[0]
        original_shape_t[:, selected_indices_face] = filtered_t_face
        original_shape_t[:, selected_indices_hand] = filtered_t_hand
        original_shape_t[:, selected_indices_lower] = filtered_t_lower
        original_shape_t[:, selected_indices_upper] = filtered_t_upper
        return original_shape_t
    
    def inverse_selection_tensor_full_body_6D_from_lower(self, filtered_t_lower, selection_array_lower, n):
        selection_array_lower = torch.from_numpy(selection_array_lower).to(filtered_t_lower.device)
        original_shape_t = torch.zeros((n, 330), device=filtered_t_lower.device, dtype=filtered_t_lower.dtype)
        selected_indices_lower = torch.where(selection_array_lower == 1)[0]
        original_shape_t[:, selected_indices_lower] = filtered_t_lower
        return original_shape_t
    
    def inverse_selection_tensor_full_body_6D_from_upper(self, filtered_t_upper, selection_array_upper, n):
        selection_array_upper = torch.from_numpy(selection_array_upper).to(filtered_t_upper.device)
        original_shape_t = torch.zeros((n, 330), device=filtered_t_upper.device, dtype=filtered_t_upper.dtype)
        selected_indices_upper = torch.where(selection_array_upper == 1)[0]
        original_shape_t[:, selected_indices_upper] = filtered_t_upper
        return original_shape_t
    
    def inverse_selection_tensor_full_body_6D_from_hand(self, filtered_t_hand, selection_array_hand, n):
        selection_array_hand = torch.from_numpy(selection_array_hand).to(filtered_t_hand.device)
        original_shape_t = torch.zeros((n, 330), device=filtered_t_hand.device, dtype=filtered_t_hand.dtype)
        selected_indices_hand = torch.where(selection_array_hand == 1)[0]
        original_shape_t[:, selected_indices_hand] = filtered_t_hand
        return original_shape_t

    def _recons_loss(self, rec, target):
        """Reconstruction loss helper for unified representations (e.g., GENMO)."""
        if self.cfg.DATASET.motion_representation == "genmo":
            return self._recons_genmo_loss(rec, target)

        loss_type = self.cfg.LOSS.ABLATION.RECONS_LOSS
        if loss_type == "l1":
            return F.l1_loss(rec, target)
        if loss_type == "l2":
            return F.mse_loss(rec, target)
        if loss_type == "l1_smooth":
            return F.smooth_l1_loss(rec, target)
        raise ValueError(f"Unsupported reconstruction loss: {loss_type}")

    def _recons_genmo_loss(self, rec, target):
        """
        GENMO reconstruction loss with configurable weights:
        - Geodesic loss on rotation components (body_pose + global_orient)
        - L1/L2/SmoothL1 on non-rotation components (betas + local_velocity)

        GENMO 145D layout:
        - [0:126]   → body_pose_r6d (21 joints × 6D rotation)
        - [126:136] → betas (10 shape parameters)
        - [136:142] → global_orient_r6d (root rotation, 6D)
        - [142:145] → local_transl_vel (local translation velocity)
        """
        # rec/target: [B, T, 145]
        # Extract rotation components
        rec_body = rec[..., :126].reshape(-1, 21, 6)
        tar_body = target[..., :126].reshape(-1, 21, 6)
        rec_global = rec[..., 136:142].reshape(-1, 6)
        tar_global = target[..., 136:142].reshape(-1, 6)

        # Convert 6D to rotation matrices for geodesic loss
        rec_body_R = rotation_6d_to_matrix(rec_body.reshape(-1, 6))
        tar_body_R = rotation_6d_to_matrix(tar_body.reshape(-1, 6))
        rec_global_R = rotation_6d_to_matrix(rec_global)
        tar_global_R = rotation_6d_to_matrix(tar_global)

        # Geodesic losses (measure actual rotation angle error)
        geo_body = self.rec_loss(rec_body_R, tar_body_R)
        geo_global = self.rec_loss(rec_global_R, tar_global_R)

        # Extract non-rotation components
        rec_betas = rec[..., 126:136]
        tar_betas = target[..., 126:136]
        rec_vel = rec[..., 142:145]
        tar_vel = target[..., 142:145]

        # Position loss derived from local velocity in world frame
        bsz, seq_len = rec.shape[:2]
        rec_global_R = rec_global_R.view(bsz, seq_len, 3, 3)
        tar_global_R = tar_global_R.view(bsz, seq_len, 3, 3)
        rec_world_vel = torch.einsum("btij,btj->bti", rec_global_R, rec_vel)
        tar_world_vel = torch.einsum("btij,btj->bti", tar_global_R, tar_vel)
        rec_pos = torch.zeros_like(rec_world_vel)
        tar_pos = torch.zeros_like(tar_world_vel)
        if seq_len > 1:
            rec_pos[:, 1:] = torch.cumsum(rec_world_vel[:, 1:], dim=1)
            tar_pos[:, 1:] = torch.cumsum(tar_world_vel[:, 1:], dim=1)

        # Non-rotation losses (direct reconstruction)
        loss_type = self.cfg.LOSS.ABLATION.RECONS_LOSS
        if loss_type == "l1":
            loss_fn = F.l1_loss
        elif loss_type == "l2":
            loss_fn = F.mse_loss
        elif loss_type == "l1_smooth":
            loss_fn = F.smooth_l1_loss
        else:
            raise ValueError(f"Unsupported reconstruction loss: {loss_type}")

        loss_betas = loss_fn(rec_betas, tar_betas)
        loss_vel = loss_fn(rec_vel, tar_vel)
        loss_pos = loss_fn(rec_pos, tar_pos)

        # Velocity/position smoothness (LoM-style supervision, adapted to local velocity)
        vel_v = loss_fn(rec_vel[:, 1:] - rec_vel[:, :-1], tar_vel[:, 1:] - tar_vel[:, :-1])
        vel_a = loss_fn(
            rec_vel[:, 2:] + rec_vel[:, :-2] - 2 * rec_vel[:, 1:-1],
            tar_vel[:, 2:] + tar_vel[:, :-2] - 2 * tar_vel[:, 1:-1],
        )
        pos_v = loss_fn(rec_pos[:, 1:] - rec_pos[:, :-1], tar_pos[:, 1:] - tar_pos[:, :-1])
        pos_a = loss_fn(
            rec_pos[:, 2:] + rec_pos[:, :-2] - 2 * rec_pos[:, 1:-1],
            tar_pos[:, 2:] + tar_pos[:, :-2] - 2 * tar_pos[:, 1:-1],
        )

        # Get configurable weights (with defaults)
        w_geo_body = getattr(self.cfg.LOSS, 'LAMBDA_GEO_BODY', 1.0)
        w_geo_global = getattr(self.cfg.LOSS, 'LAMBDA_GEO_GLOBAL', 0.2)
        w_betas = getattr(self.cfg.LOSS, 'LAMBDA_BETAS', 0.1)
        w_local_vel = getattr(self.cfg.LOSS, 'LAMBDA_LOCAL_VEL', 1.0)
        w_local_pos = getattr(self.cfg.LOSS, 'LAMBDA_LOCAL_POS', 1.0)

        total_loss = (
            w_geo_body * geo_body +
            w_geo_global * geo_global +
            w_betas * loss_betas +
            w_local_vel * (loss_vel + 5.0 * vel_v + 5.0 * vel_a) +
            w_local_pos * (loss_pos + 5.0 * pos_v + 5.0 * pos_a)
        )

        return total_loss

    def _integrate_local_velocity(self, local_vel, global_orient_6d, init_pos=None):
        """Integrate root local velocity into world positions using global orientation."""
        R = rotation_6d_to_matrix(global_orient_6d)  # (B, T, 3, 3)
        world_vel = torch.einsum("btij,btj->bti", R, local_vel)
        pos = torch.zeros_like(world_vel)
        if init_pos is None:
            pos[:, 0, :] = 0.0
        else:
            pos[:, 0, :] = init_pos
        if world_vel.shape[1] > 1:
            pos[:, 1:, :] = pos[:, 0:1, :] + torch.cumsum(world_vel[:, 1:, :], dim=1)
        return pos, world_vel

    def _batch_dataset_ids(self, batch, device):
        """Map batch['dataset_name'] (list of strings) -> LongTensor of dataset ids for VAEConvZero
        dataset-conditioning: beat2->0, amass->1, bones->2 (matches SAMPLING_RATIO). None if absent."""
        names = batch.get("dataset_name", None) if isinstance(batch, dict) else None
        if not names:
            return None
        ids = []
        for d in names:
            s = str(d).lower()
            ids.append(2 if "bones" in s else (0 if "beat2" in s else 1))  # default amass(1)
        return torch.as_tensor(ids, dtype=torch.long, device=device)

    def _betas_to_measurements(self, betas):
        """Deployable, model-agnostic shape feature: physical limb measurements (meters) from betas via
        SMPL-X T-pose FK. Translation scales with LEG LENGTH; unlike raw betas this is a basis-free physical
        quantity (a meter is a meter). Returns [B,3] = (leg_length, stature, hip_width), nominally centered.
        Uses the FULL native betas (e.g. BEAT2 300-D) so the reconstructed body — and thus the limb lengths —
        is correct; truncating to 10 would distort the shape."""
        import smplx as _smplx
        dev = betas.device
        if getattr(self, "_smplx_measure", None) is None:
            path = self.cfg.DATASET.SMPLX_MODEL_DIR
            self._smplx_measure = _smplx.create(
                path, model_type='smplx', gender='NEUTRAL_2020', use_face_contour=False,
                num_betas=300, num_expression_coeffs=100, ext='npz', use_pca=False).to(dev).eval()
        B = betas.shape[0]
        b300 = torch.zeros(B, 300, device=dev, dtype=torch.float32)
        n = min(300, betas.shape[1]); b300[:, :n] = betas[:, :n].float()   # full native betas
        z = lambda d: torch.zeros(B, d, device=dev, dtype=torch.float32)     # zero pose at batch B (T-pose)
        with torch.no_grad():
            J = self._smplx_measure(
                betas=b300, transl=z(3), expression=z(100), global_orient=z(3), body_pose=z(63),
                left_hand_pose=z(45), right_hand_pose=z(45), jaw_pose=z(3), leye_pose=z(3), reye_pose=z(3),
                return_verts=False, return_joints=True)['joints']   # [B, nJ, 3], T-pose
            L_hip, R_hip, L_knee, L_ankle, head, L_foot = J[:, 1], J[:, 2], J[:, 4], J[:, 7], J[:, 15], J[:, 10]
            leg = (L_hip - L_knee).norm(dim=-1) + (L_knee - L_ankle).norm(dim=-1)   # thigh + shank
            stature = (head[:, 1] - L_foot[:, 1]).abs()
            hipw = (L_hip - R_hip).norm(dim=-1)
            m = torch.stack([leg - 0.80, stature - 1.65, hipw - 0.19], dim=-1)       # meters, centered
        return m.detach()

    def _decode_genmo_to_smplx(self, motion_vector, trans=None):
        """
        Decode GENMO motion_vector (145D) to SMPL-X axis-angle pose + translation.
        This is used for evaluation-only SMPL-based metrics.
        """
        device = motion_vector.device
        bsz, seq_len, _ = motion_vector.shape
        motion_flat = motion_vector.reshape(bsz * seq_len, -1)

        body_r6d = motion_flat[:, :126].reshape(-1, 21, 6)
        betas_10 = motion_flat[:, 126:136]
        global_r6d = motion_flat[:, 136:142]
        local_vel = motion_flat[:, 142:145]

        body_aa = rotation_6d_to_axis_angle(body_r6d.reshape(-1, 6)).reshape(-1, 21, 3)
        global_aa = rotation_6d_to_axis_angle(global_r6d)

        R = rotation_6d_to_matrix(global_r6d)
        world_vel = torch.einsum("bij,bj->bi", R, local_vel).reshape(bsz, seq_len, 3)

        if trans is not None:
            trans0 = trans[:, 0]
        else:
            trans0 = torch.zeros((bsz, 3), device=device)

        trans_decoded = torch.zeros((bsz, seq_len, 3), device=device)
        trans_decoded[:, 0] = trans0
        if seq_len > 1:
            trans_decoded[:, 1:] = trans0.unsqueeze(1) + torch.cumsum(world_vel[:, 1:], dim=1)

        pose_aa = torch.zeros((bsz * seq_len, 55 * 3), device=device)
        pose_aa[:, :3] = global_aa
        pose_aa[:, 3:3 + 21 * 3] = body_aa.reshape(-1, 63)

        expr = torch.zeros((bsz * seq_len, 100), device=device)
        trans_flat = trans_decoded.reshape(bsz * seq_len, 3)

        return pose_aa, betas_10, expr, trans_flat

    def _get_joint_indices(self, subset_key):
        full_names = list(joints_list[BEAT_SMPLX_JOINTS].keys())
        subset_names = list(joints_list[subset_key].keys())
        return [full_names.index(name) for name in subset_names if name in full_names]

    def _get_genmo_split_cfg(self):
        include_betas = OmegaConf.select(self.cfg, "DATASET.GENMO_SPLIT_INCLUDE_BETAS")
        include_global = OmegaConf.select(self.cfg, "DATASET.GENMO_SPLIT_INCLUDE_GLOBAL_ORIENT")
        include_local_vel = OmegaConf.select(self.cfg, "DATASET.GENMO_SPLIT_INCLUDE_LOCAL_VEL")
        return {
            "include_betas": True if include_betas is None else bool(include_betas),
            "include_global": True if include_global is None else bool(include_global),
            "include_local_vel": True if include_local_vel is None else bool(include_local_vel),
        }

    @staticmethod
    def _genmo_joint_block_indices(joint_indices):
        out = []
        for joint_idx in joint_indices:
            out.extend(range(joint_idx * 6, (joint_idx + 1) * 6))
        return out

    def _get_genmo_split_indices(self, selected_part):
        if selected_part == "genmo_lower":
            joint_indices = self.GENMO_LOWER_BODY_JOINTS
            metric_joint_indices_22 = self.lower_joint_indices
        elif selected_part == "genmo_upper":
            joint_indices = self.GENMO_UPPER_BODY_JOINTS
            metric_joint_indices_22 = self.upper_joint_indices
        else:
            raise ValueError(f"Unsupported genmo split part: {selected_part}")

        cfg = self._get_genmo_split_cfg()
        indices = self._genmo_joint_block_indices(joint_indices)

        if cfg["include_betas"]:
            indices.extend(range(126, 136))
        if cfg["include_global"]:
            indices.extend(range(136, 142))
        if cfg["include_local_vel"]:
            indices.extend(range(142, 145))

        return indices, metric_joint_indices_22

    def _merge_genmo_split(self, rec_part, tar_full, split_indices):
        rec_full = tar_full.clone()
        idx = torch.as_tensor(split_indices, dtype=torch.long, device=tar_full.device)
        rec_full[..., idx] = rec_part
        return rec_full

    def _fk_joints(self, rot_mats, betas, joint_indices=None):
        """
        Fast FK-only joint computation (no pose blend shapes, no LBS).
        Uses SMPL-X rest joints shaped by betas, then applies kinematic chain.

        Args:
            rot_mats: (B, T, J, 3, 3) rotation matrices
            betas: (B, T, num_betas) or (B, num_betas)
            joint_indices: optional list/1D tensor of joint indices to subset
        Returns:
            joints: (B, T, J, 3)
        """
        if betas.dim() == 3:
            betas = betas[:, 0]
        betas = betas.to(rot_mats.device)

        # Shape the template and regress rest joints
        v_template = self.smplx.v_template.to(rot_mats.device)
        shapedirs = self.smplx.shapedirs.to(rot_mats.device)
        j_regressor = self.smplx.J_regressor.to(rot_mats.device)
        shapedirs = shapedirs[..., :betas.shape[1]]
        v_shaped = v_template + blend_shapes(betas, shapedirs)
        # J = J_regressor @ v_shaped (no mesh/LBS, single matmul)
        rest_joints = torch.einsum("jv,bvk->bjk", j_regressor, v_shaped)

        bsz, seq_len, num_joints = rot_mats.shape[:3]
        total_joints = rest_joints.shape[1]
        if joint_indices is not None:
            joint_indices = torch.as_tensor(joint_indices, device=rot_mats.device, dtype=torch.long)
            rest_joints = rest_joints[:, joint_indices]
            parents = self.smplx.parents[joint_indices].to(rot_mats.device).clone()
            remap = {int(old): i for i, old in enumerate(joint_indices.tolist())}
            for i in range(len(parents)):
                parent = int(parents[i])
                parents[i] = remap.get(parent, -1)
            if rot_mats.shape[2] != joint_indices.numel():
                if rot_mats.shape[2] == total_joints:
                    rot_mats = rot_mats[:, :, joint_indices]
                else:
                    raise ValueError(
                        f"FK joints: rot_mats has {rot_mats.shape[2]} joints but subset has "
                        f"{joint_indices.numel()}."
                    )
            num_joints = joint_indices.numel()
        else:
            if num_joints > total_joints:
                raise ValueError(
                    f"FK joints: rot_mats expects {num_joints} joints but SMPL-X provides {total_joints}."
                )
            if num_joints < total_joints:
                # Use the leading joints (e.g., 22 body joints for GENMO: pelvis + 21 body joints)
                rest_joints = rest_joints[:, :num_joints]
                parents = self.smplx.parents[:num_joints].to(rot_mats.device)
                # Ensure parent indices are valid within the subset
                if (parents[1:] >= num_joints).any():
                    raise ValueError(
                        "FK joints: subset parents contain indices outside the subset. "
                        "Provide full joint rotations or adjust joint ordering."
                    )
            else:
                parents = self.smplx.parents.to(rot_mats.device)

        rest_joints = rest_joints[:, None].expand(bsz, seq_len, num_joints, 3)

        rot_flat = rot_mats.reshape(bsz * seq_len, num_joints, 3, 3)
        joints_flat = rest_joints.reshape(bsz * seq_len, num_joints, 3)
        posed_joints, _ = batch_rigid_transform(rot_flat, joints_flat, parents, dtype=rot_mats.dtype)
        return posed_joints.reshape(bsz, seq_len, num_joints, 3)

    def forward(self, batch):
        raise NotImplementedError(
            "MultimodalTokenizer does not implement LM generation. "
            "Use VQ/VAE training or add a generation module."
        )

    # @torch.no_grad()
    def forward_flame(self, exps, jaw, shape, batch_size):

        ## Don't know why, but the J_regressor is on cpu by default
        if self.flame.J_regressor.device != exps.device:
            self.flame.to(exps.device)
        # bs, n = exps.shape[0], exps.shape[1]
        actual_length = exps.shape[0]
        if actual_length > batch_size:
            s, r = actual_length//batch_size, actual_length%batch_size
            output_joints = []
            output_vertices = []
            for div_idx in range(s):
                flame_output = self.flame(
                    global_orient=torch.zeros((batch_size, 3), device=exps.device),
                    expression=exps[div_idx*batch_size:(div_idx+1)*batch_size],
                    jaw_pose=jaw[div_idx*batch_size:(div_idx+1)*batch_size],
                    shape=shape[div_idx*batch_size:(div_idx+1)*batch_size],
                )
                output_joints.append(flame_output.joints)
                output_vertices.append(flame_output.vertices)
            if r != 0:
                exps_pad = torch.cat([exps[s*batch_size:s*batch_size+r], torch.zeros((batch_size-r, 100), device=exps.device)], dim=0)
                jaw_pad = torch.cat([jaw[s*batch_size:s*batch_size+r], torch.zeros((batch_size-r, 3), device=jaw.device)], dim=0)
                shape_pad = torch.cat([shape[s*batch_size:s*batch_size+r], torch.zeros((batch_size-r, 10), device=shape.device)], dim=0)
                flame_output = self.flame(
                    global_orient=torch.zeros((batch_size, 3), device=exps.device),
                    expression=exps_pad,
                    jaw_pose=jaw_pad,
                    shape=shape_pad,
                )
                output_joints.append(flame_output.joints[:r])
                output_vertices.append(flame_output.vertices[:r])
        else:
            output_joints = []
            output_vertices = []
            exps_pad = torch.cat([exps, torch.zeros((batch_size-actual_length, 100), device=exps.device)], dim=0)
            jaw_pad = torch.cat([jaw, torch.zeros((batch_size-actual_length, 3), device=jaw.device)], dim=0)
            shape_pad = torch.cat([shape, torch.zeros((batch_size-actual_length, 10), device=shape.device)], dim=0)
            flame_output = self.flame(
                global_orient=torch.zeros((batch_size, 3), device=exps.device),
                expression=exps_pad,
                jaw_pose=jaw_pad,
                shape=shape_pad,
            )
            output_joints.append(flame_output.joints)
            output_vertices.append(flame_output.vertices)

        output_joints = torch.cat(output_joints, dim=0)
        output_vertices = torch.cat(output_vertices, dim=0)

        return output_joints, output_vertices


    # @torch.no_grad()
    def forward_smplx(self, rec_pose, shape, expression, transl, batch_size):

        actual_length = rec_pose.shape[0]
        if actual_length > batch_size:
            s, r = actual_length//batch_size, actual_length%batch_size
            output_joints = []
            output_vertices = []
            for div_idx in range(s):
                output_rec = self.smplx(
                    betas=shape[div_idx*batch_size:(div_idx+1)*batch_size].reshape(batch_size, self.smplx_num_betas), 
                    transl=transl[div_idx*batch_size:(div_idx+1)*batch_size], 
                    expression =expression[div_idx*batch_size:(div_idx+1)*batch_size],
                    jaw_pose=rec_pose[div_idx*batch_size:(div_idx+1)*batch_size, 66:69], 
                    global_orient=rec_pose[div_idx*batch_size:(div_idx+1)*batch_size,:3], 
                    body_pose=rec_pose[div_idx*batch_size:(div_idx+1)*batch_size,3:21*3+3], 
                    left_hand_pose=rec_pose[div_idx*batch_size:(div_idx+1)*batch_size,25*3:40*3], 
                    right_hand_pose=rec_pose[div_idx*batch_size:(div_idx+1)*batch_size,40*3:55*3], 
                    return_verts=True,
                    return_joints=True,
                    leye_pose=rec_pose[div_idx*batch_size:(div_idx+1)*batch_size, 69:72], 
                    reye_pose=rec_pose[div_idx*batch_size:(div_idx+1)*batch_size, 72:75],
                    )
                output_joints.append(output_rec.joints[:, :22].reshape(batch_size, 22, 3).cpu())
                output_vertices.append(output_rec.vertices.reshape(batch_size, 10475, 3).cpu())
            if r != 0:
                output_rec = self.smplx(
                    betas=shape[s*batch_size:s*batch_size+r].reshape(r, self.smplx_num_betas), 
                    transl=transl[s*batch_size:s*batch_size+r], 
                    expression = torch.zeros((r, 100), device=rec_pose.device),
                    jaw_pose=rec_pose[s*batch_size:s*batch_size+r, 66:69], 
                    global_orient=rec_pose[s*batch_size:s*batch_size+r,:3], 
                    body_pose=rec_pose[s*batch_size:s*batch_size+r,3:21*3+3], 
                    left_hand_pose=rec_pose[s*batch_size:s*batch_size+r,25*3:40*3], 
                    right_hand_pose=rec_pose[s*batch_size:s*batch_size+r,40*3:55*3], 
                    return_verts=True,
                    return_joints=True,
                    leye_pose=rec_pose[s*batch_size:s*batch_size+r, 69:72], 
                    reye_pose=rec_pose[s*batch_size:s*batch_size+r, 72:75],
                    )
                output_joints.append(output_rec.joints[:, :22].reshape(r, 22, 3).cpu())
                output_vertices.append(output_rec.vertices.reshape(r, 10475, 3).cpu())
        else:
            output_joints = []
            output_vertices = []
            output_rec = self.smplx(
                betas=shape.reshape(actual_length, self.smplx_num_betas), 
                transl=transl, 
                expression=expression,
                jaw_pose=rec_pose[:, 66:69], 
                global_orient=rec_pose[:,:3], 
                body_pose=rec_pose[:,3:21*3+3], 
                left_hand_pose=rec_pose[:,25*3:40*3], 
                right_hand_pose=rec_pose[:,40*3:55*3], 
                return_verts=True,
                return_joints=True,
                leye_pose=rec_pose[:, 69:72], 
                reye_pose=rec_pose[:, 72:75],
                )
            output_joints.append(output_rec.joints[:, :22].reshape(actual_length, 22, 3).cpu())
            output_vertices.append(output_rec.vertices.reshape(actual_length, 10475, 3).cpu())

        output_joints = torch.cat(output_joints, dim=0)
        output_vertices = torch.cat(output_vertices, dim=0)

        return output_joints, output_vertices


    def train_vae_forward(self, batch):
        use_fk_joints = getattr(self.cfg.TRAIN, "FK_JOINTS", False)

        if self.hparams.Selected_part == 'lower_54' or self.hparams.Selected_part == 'lower':

            tar_beta, tar_lower = [
                batch[key] for key in ["shape", "lower"]
            ]
            lower_dim = self.cfg.model.params.modality_tokenizer.vae_lower.params.vae_test_dim

            net_out_lower = self.vae_lower(tar_lower[..., :lower_dim])
            tar_global = tar_lower.clone()
            tar_global[:, :, 54:57] *= 0.0
            # net_out_global = self.vae_global(tar_global)
            rec_lower = net_out_lower["rec_pose"]
            # rec_global = net_out_global["rec_pose"]
            bs = tar_lower.shape[0]
            n = min(tar_lower.shape[1], rec_lower.shape[1])
            rec_lower = rec_lower[:,:n]
            # rec_global = rec_global[:,:n]
            tar_lower = tar_lower[:,:n]
            # tar_pose = tar_pose[:,:n]
            tar_beta = tar_beta[:,:n]
            # tar_trans = tar_trans[:,:n]

            rec_pose = self.inverse_selection_tensor_full_body_6D_from_lower(rec_lower[:,:,:54].reshape(bs * n, 9 * 6), JOINT_MASK_LOWER_6D, n*bs)
            tar_pose = self.inverse_selection_tensor_full_body_6D_from_lower(tar_lower[:,:,:54].reshape(bs * n, 9 * 6), JOINT_MASK_LOWER_6D, n*bs)
            rec_pose_aa = rotation_6d_to_axis_angle(rec_pose.reshape(bs * n, 55, 6)).reshape(bs * n, 55 * 3)
            tar_pose_aa = rotation_6d_to_axis_angle(tar_pose.reshape(bs * n, 55, 6)).reshape(bs * n, 55 * 3)

            if self.cfg.TRAIN.LOSS_MESH:
                if use_fk_joints:
                    rec_rot = rotation_6d_to_matrix(rec_pose.reshape(bs * n, 55, 6)).reshape(bs, n, 55, 3, 3)
                    tar_rot = rotation_6d_to_matrix(tar_pose.reshape(bs * n, 55, 6)).reshape(bs, n, 55, 3, 3)
                    joints_rec = self._fk_joints(rec_rot, tar_beta).reshape(bs * n, -1, 3)
                    joints_tar = self._fk_joints(tar_rot, tar_beta).reshape(bs * n, -1, 3)
                    vertices_rec = {"vertices": joints_rec, "joints": joints_rec}
                    vertices_tar = {"vertices": joints_tar, "joints": joints_tar}
                else:
                    vertices_rec = self.smplx(
                        betas=tar_beta.reshape(bs*n, 300), 
                        transl=torch.zeros((bs*n, 3), device=rec_pose_aa.device), 
                        # expression=rec_exps.reshape(bs*n, 100),
                        expression = torch.zeros((bs*n, 100), device=rec_pose_aa.device),
                        jaw_pose=rec_pose_aa[:, 66:69], 
                        global_orient=rec_pose_aa[:,:3], 
                        body_pose=rec_pose_aa[:,3:21*3+3], 
                        left_hand_pose=rec_pose_aa[:,25*3:40*3], 
                        right_hand_pose=rec_pose_aa[:,40*3:55*3], 
                        return_verts=True,
                        return_joints=True,
                        leye_pose=tar_pose_aa[:, 69:72], 
                        reye_pose=tar_pose_aa[:, 72:75],
                        )
                    vertices_tar = self.smplx(
                        betas=tar_beta.reshape(bs*n, 300), 
                        transl=torch.zeros((bs*n, 3), device=rec_pose_aa.device), 
                        # expression=tar_exps.reshape(bs*n, 100), 
                        expression = torch.zeros((bs*n, 100), device=rec_pose_aa.device),
                        jaw_pose=tar_pose_aa[:, 66:69], 
                        global_orient=tar_pose_aa[:,:3], 
                        body_pose=tar_pose_aa[:,3:21*3+3], 
                        left_hand_pose=tar_pose_aa[:,25*3:40*3], 
                        right_hand_pose=tar_pose_aa[:,40*3:55*3], 
                        return_verts=True,
                        return_joints=True,
                        leye_pose=tar_pose_aa[:, 69:72], 
                        reye_pose=tar_pose_aa[:, 72:75],
                        )
            else:
                vertices_rec = None
                vertices_tar = None

            loss_lower = self.lower_loss(rec_lower, tar_lower, self.cfg.TRAIN.LOSS_6D, vertices_rec, vertices_tar, self.cfg.TRAIN.LOSS_MESH)
            betas_loss = torch.tensor(0.0, device=rec_lower.device)
            if rec_lower.shape[2] > 61:
                beta_dim = getattr(self.cfg.DATASET, "LOWER_BETA_DIM", 10)
                w_betas = getattr(self.cfg.LOSS, "LAMBDA_BETAS", 0.0)
                if w_betas > 0:
                    betas_rec = rec_lower[:, :, 61:61 + beta_dim]
                    betas_tar = tar_lower[:, :, 61:61 + beta_dim]
                    betas_loss = F.l1_loss(betas_rec, betas_tar)
                    loss_lower = loss_lower + w_betas * betas_loss
            # Get lengths if available, otherwise use full sequence length
            lengths = batch.get("motion_len", None)

            rs_set = {
                "tar_lower":tar_lower,
                "rec_lower":rec_lower,
                "tar_beta":tar_beta,
                # "tar_trans":tar_trans,
                "recons-lower_loss": loss_lower,
                "commit-lower_loss": net_out_lower["embedding_loss"],
                "betas_metric": betas_loss,
                "length": lengths  # Use the computed lengths
            }
            if "perplexity" in net_out_lower:
                rs_set["perplexity"] = net_out_lower["perplexity"]

        elif self.hparams.Selected_part == 'lower_global':

            tar_beta = batch["shape"]
            tar_lower = batch["lower"]
            tar_trans = batch.get("trans")

            lower_dim = self.cfg.model.params.modality_tokenizer.vae_lower.params.vae_test_dim

            net_out_lower = self.vae_lower(tar_lower[..., :lower_dim])
            rec_lower = net_out_lower["rec_pose"]
            bs = tar_lower.shape[0]
            n = min(tar_lower.shape[1], rec_lower.shape[1])
            rec_lower = rec_lower[:,:n]

            tar_lower = tar_lower[:,:n]
            tar_beta = tar_beta[:,:n]
            if tar_trans is not None:
                tar_trans = tar_trans[:,:n]

            # rec_lower[:, :, 54:57] now contains local velocity (GENMO-style)
            rec_local_vel = rec_lower[:, :, 54:57]
            tar_local_vel = tar_lower[:, :, 54:57]  # target local velocity

            # Get global_orient from lower (first 6D rotation is pelvis/global_orient)
            rec_global_orient_6d = rec_lower[:, :, :6]  # (bs, n, 6)
            tar_global_orient_6d = tar_lower[:, :, :6]

            if tar_trans is None:
                tar_trans, _ = self._integrate_local_velocity(tar_local_vel, tar_global_orient_6d)

            # Convert local velocity to world velocity and integrate
            rec_xyz_trans, rec_world_vel = self._integrate_local_velocity(
                rec_local_vel, rec_global_orient_6d, init_pos=tar_trans[:, 0, :]
            )

            rec_contact = rec_lower[:, :, 57:61]
            tar_contact = tar_lower[:, :, 57:61]

            rec_pose = self.inverse_selection_tensor_full_body_6D_from_lower(rec_lower[:,:,:54].reshape(bs * n, 9 * 6), JOINT_MASK_LOWER_6D, n*bs)
            tar_pose = self.inverse_selection_tensor_full_body_6D_from_lower(tar_lower[:,:,:54].reshape(bs * n, 9 * 6), JOINT_MASK_LOWER_6D, n*bs)
            rec_pose_aa = rotation_6d_to_axis_angle(rec_pose.reshape(bs * n, 55, 6)).reshape(bs * n, 55 * 3)
            tar_pose_aa = rotation_6d_to_axis_angle(tar_pose.reshape(bs * n, 55, 6)).reshape(bs * n, 55 * 3)

            if self.cfg.TRAIN.LOSS_MESH:
                if use_fk_joints:
                    rec_rot = rotation_6d_to_matrix(rec_pose.reshape(bs * n, 55, 6)).reshape(bs, n, 55, 3, 3)
                    tar_rot = rotation_6d_to_matrix(tar_pose.reshape(bs * n, 55, 6)).reshape(bs, n, 55, 3, 3)
                    joints_rec = self._fk_joints(rec_rot, tar_beta).reshape(bs * n, -1, 3)
                    joints_tar = self._fk_joints(tar_rot, tar_beta).reshape(bs * n, -1, 3)
                    vertices_rec = {"vertices": joints_rec, "joints": joints_rec}
                    vertices_tar = {"vertices": joints_tar, "joints": joints_tar}
                else:
                    vertices_rec = self.smplx(
                        betas=tar_beta.reshape(bs*n, 300), 
                        transl=torch.zeros((bs*n, 3), device=rec_pose_aa.device), 
                        # expression=rec_exps.reshape(bs*n, 100),
                        expression = torch.zeros((bs*n, 100), device=rec_pose_aa.device),
                        jaw_pose=rec_pose_aa[:, 66:69], 
                        global_orient=rec_pose_aa[:,:3], 
                        body_pose=rec_pose_aa[:,3:21*3+3], 
                        left_hand_pose=rec_pose_aa[:,25*3:40*3], 
                        right_hand_pose=rec_pose_aa[:,40*3:55*3], 
                        return_verts=True,
                        return_joints=True,
                        leye_pose=tar_pose_aa[:, 69:72], 
                        reye_pose=tar_pose_aa[:, 72:75],
                        )
                    vertices_tar = self.smplx(
                        betas=tar_beta.reshape(bs*n, 300), 
                        transl=torch.zeros((bs*n, 3), device=rec_pose_aa.device), 
                        # expression=tar_exps.reshape(bs*n, 100), 
                        expression = torch.zeros((bs*n, 100), device=rec_pose_aa.device),
                        jaw_pose=tar_pose_aa[:, 66:69], 
                        global_orient=tar_pose_aa[:,:3], 
                        body_pose=tar_pose_aa[:,3:21*3+3], 
                        left_hand_pose=tar_pose_aa[:,25*3:40*3], 
                        right_hand_pose=tar_pose_aa[:,40*3:55*3], 
                        return_verts=True,
                        return_joints=True,
                        leye_pose=tar_pose_aa[:, 69:72], 
                        reye_pose=tar_pose_aa[:, 72:75],
                        )
            else:
                vertices_rec = None
                vertices_tar = None

            loss_lower = self.lower_loss(rec_lower, tar_lower, self.cfg.TRAIN.LOSS_6D, vertices_rec, vertices_tar, self.cfg.TRAIN.LOSS_MESH)
            betas_loss = torch.tensor(0.0, device=rec_lower.device)
            if rec_lower.shape[2] > 61:
                beta_dim = getattr(self.cfg.DATASET, "LOWER_BETA_DIM", 10)
                w_betas = getattr(self.cfg.LOSS, "LAMBDA_BETAS", 0.0)
                if w_betas > 0:
                    betas_rec = rec_lower[:, :, 61:61 + beta_dim]
                    betas_tar = tar_lower[:, :, 61:61 + beta_dim]
                    betas_loss = F.l1_loss(betas_rec, betas_tar)
                    loss_lower = loss_lower + w_betas * betas_loss

            joint_pos_loss = torch.tensor(0.0, device=rec_lower.device)
            joint_vel_loss = torch.tensor(0.0, device=rec_lower.device)
            joint_acc_loss = torch.tensor(0.0, device=rec_lower.device)
            if use_fk_joints:
                w_joint_pos = getattr(self.cfg.LOSS, "LAMBDA_JOINT_POS", 0.0)
                w_joint_vel = getattr(self.cfg.LOSS, "LAMBDA_JOINT_VEL", 0.0)
                w_joint_acc = getattr(self.cfg.LOSS, "LAMBDA_JOINT_ACC", 0.0)
                if (w_joint_pos + w_joint_vel + w_joint_acc) > 0:
                    rec_rot = rotation_6d_to_matrix(rec_pose.reshape(bs * n, 55, 6)).reshape(bs, n, 55, 3, 3)
                    tar_rot = rotation_6d_to_matrix(tar_pose.reshape(bs * n, 55, 6)).reshape(bs, n, 55, 3, 3)
                    betas_fk = tar_beta[..., :10] if tar_beta.shape[-1] > 10 else tar_beta
                    joints_rec = self._fk_joints(rec_rot, betas_fk, joint_indices=self.lower_joint_indices)
                    joints_tar = self._fk_joints(tar_rot, betas_fk, joint_indices=self.lower_joint_indices)
                    joint_scale = getattr(self.cfg.LOSS, "JOINT_UNIT_SCALE", 1.0)
                    if joint_scale != 1.0:
                        joints_rec = joints_rec * joint_scale
                        joints_tar = joints_tar * joint_scale
                    if w_joint_pos > 0:
                        joint_pos_loss = F.mse_loss(joints_rec, joints_tar)
                    if w_joint_vel > 0 and n > 1:
                        joint_vel_loss = F.mse_loss(
                            joints_rec[:, 1:] - joints_rec[:, :-1],
                            joints_tar[:, 1:] - joints_tar[:, :-1],
                        )
                    if w_joint_acc > 0 and n > 2:
                        joint_acc_loss = F.mse_loss(
                            joints_rec[:, 2:] + joints_rec[:, :-2] - 2 * joints_rec[:, 1:-1],
                            joints_tar[:, 2:] + joints_tar[:, :-2] - 2 * joints_tar[:, 1:-1],
                        )
                    loss_lower = (
                        loss_lower
                        + w_joint_pos * joint_pos_loss
                        + w_joint_vel * joint_vel_loss
                        + w_joint_acc * joint_acc_loss
                    )
            loss_global = self.global_loss(
                rec_xyz_trans,
                rec_local_vel,
                tar_trans,
                tar_local_vel,
            )
            trans_vel_metric = F.l1_loss(rec_local_vel, tar_local_vel).detach()
            trans_pos_metric = F.l1_loss(rec_xyz_trans, tar_trans).detach()

            commit_loss = torch.tensor(0.0, device=tar_trans.device)

            # Get lengths if available, otherwise use full sequence length
            lengths = batch.get("motion_len", None)

            rs_set = {
                "tar_lower":tar_lower,
                "rec_lower":rec_lower,
                "tar_global":tar_trans,
                "rec_global":rec_xyz_trans,
                "tar_beta":tar_beta,
                # "tar_trans":tar_trans,
                "recons-lower_loss": loss_lower,
                "commit-lower_loss": net_out_lower["embedding_loss"],
                "recons-global_loss": loss_global,
                "commit-global_loss": commit_loss,
                "betas_metric": betas_loss,
                "joint_pos_metric": joint_pos_loss,
                "joint_vel_metric": joint_vel_loss,
                "joint_acc_metric": joint_acc_loss,
                "trans_vel_metric": trans_vel_metric,
                "trans_pos_metric": trans_pos_metric,
                "length": lengths  # Use the computed lengths
            }
            if "perplexity" in net_out_lower:
                rs_set["perplexity"] = net_out_lower["perplexity"]


        elif self.hparams.Selected_part == 'upper_lower_global':

            tar_beta = batch["shape"]
            tar_upper = batch["upper"]
            tar_lower = batch["lower"]
            tar_trans = batch.get("trans")

            upper_dim = int(JOINT_MASK_UPPER_6D.sum())
            combined_dim = self.cfg.model.params.modality_tokenizer.vae_lower.params.vae_test_dim
            lower_dim = combined_dim - upper_dim

            tar_upper = tar_upper[..., :upper_dim]
            tar_lower = tar_lower[..., :lower_dim]
            tar_combined = torch.cat([tar_upper, tar_lower], dim=-1)

            net_out_lower = self.vae_lower(tar_combined[..., :combined_dim])
            rec_combined = net_out_lower["rec_pose"]
            bs = tar_combined.shape[0]
            n = min(tar_combined.shape[1], rec_combined.shape[1])

            rec_combined = rec_combined[:, :n]
            tar_upper = tar_upper[:, :n]
            tar_lower = tar_lower[:, :n]
            tar_beta = tar_beta[:, :n]
            if tar_trans is not None:
                tar_trans = tar_trans[:, :n]

            rec_upper = rec_combined[:, :, :upper_dim]
            rec_lower = rec_combined[:, :, upper_dim:upper_dim + lower_dim]

            # rec_lower[:, :, 54:57] now contains local velocity (GENMO-style)
            rec_local_vel = rec_lower[:, :, 54:57]
            tar_local_vel = tar_lower[:, :, 54:57]  # target local velocity

            # Get global_orient from lower (first 6D rotation is pelvis/global_orient)
            rec_global_orient_6d = rec_lower[:, :, :6]  # (bs, n, 6)
            tar_global_orient_6d = tar_lower[:, :, :6]

            if tar_trans is None:
                tar_trans, _ = self._integrate_local_velocity(tar_local_vel, tar_global_orient_6d)

            # Convert local velocity to world velocity and integrate
            rec_xyz_trans, rec_world_vel = self._integrate_local_velocity(
                rec_local_vel, rec_global_orient_6d, init_pos=tar_trans[:, 0, :]
            )

            rec_contact = rec_lower[:, :, 57:61]
            tar_contact = tar_lower[:, :, 57:61]

            rec_pose_upper = self.inverse_selection_tensor_full_body_6D_from_upper(
                rec_upper.reshape(bs * n, 13 * 6), JOINT_MASK_UPPER_6D, n * bs
            )
            tar_pose_upper = self.inverse_selection_tensor_full_body_6D_from_upper(
                tar_upper.reshape(bs * n, 13 * 6), JOINT_MASK_UPPER_6D, n * bs
            )
            rec_pose_lower = self.inverse_selection_tensor_full_body_6D_from_lower(
                rec_lower[:, :, :54].reshape(bs * n, 9 * 6), JOINT_MASK_LOWER_6D, n * bs
            )
            tar_pose_lower = self.inverse_selection_tensor_full_body_6D_from_lower(
                tar_lower[:, :, :54].reshape(bs * n, 9 * 6), JOINT_MASK_LOWER_6D, n * bs
            )
            rec_pose = rec_pose_upper + rec_pose_lower
            tar_pose = tar_pose_upper + tar_pose_lower
            rec_pose_aa = rotation_6d_to_axis_angle(rec_pose.reshape(bs * n, 55, 6)).reshape(bs * n, 55 * 3)
            tar_pose_aa = rotation_6d_to_axis_angle(tar_pose.reshape(bs * n, 55, 6)).reshape(bs * n, 55 * 3)

            if self.cfg.TRAIN.LOSS_MESH:
                if use_fk_joints:
                    rec_rot = rotation_6d_to_matrix(rec_pose.reshape(bs * n, 55, 6)).reshape(bs, n, 55, 3, 3)
                    tar_rot = rotation_6d_to_matrix(tar_pose.reshape(bs * n, 55, 6)).reshape(bs, n, 55, 3, 3)
                    betas_fk = tar_beta[..., :10] if tar_beta.shape[-1] > 10 else tar_beta
                    joints_rec = self._fk_joints(rec_rot, betas_fk, joint_indices=self.upper_lower_joint_indices)
                    joints_tar = self._fk_joints(tar_rot, betas_fk, joint_indices=self.upper_lower_joint_indices)
                    vertices_rec = {"vertices": joints_rec, "joints": joints_rec}
                    vertices_tar = {"vertices": joints_tar, "joints": joints_tar}
                else:
                    vertices_rec = self.smplx(
                        betas=tar_beta.reshape(bs*n, 300),
                        transl=torch.zeros((bs*n, 3), device=rec_pose_aa.device),
                        expression=torch.zeros((bs*n, 100), device=rec_pose_aa.device),
                        jaw_pose=rec_pose_aa[:, 66:69],
                        global_orient=rec_pose_aa[:,:3],
                        body_pose=rec_pose_aa[:,3:21*3+3],
                        left_hand_pose=rec_pose_aa[:,25*3:40*3],
                        right_hand_pose=rec_pose_aa[:,40*3:55*3],
                        return_verts=True,
                        return_joints=True,
                        leye_pose=tar_pose_aa[:, 69:72],
                        reye_pose=tar_pose_aa[:, 72:75],
                        )
                    vertices_tar = self.smplx(
                        betas=tar_beta.reshape(bs*n, 300),
                        transl=torch.zeros((bs*n, 3), device=rec_pose_aa.device),
                        expression=torch.zeros((bs*n, 100), device=rec_pose_aa.device),
                        jaw_pose=tar_pose_aa[:, 66:69],
                        global_orient=tar_pose_aa[:,:3],
                        body_pose=tar_pose_aa[:,3:21*3+3],
                        left_hand_pose=tar_pose_aa[:,25*3:40*3],
                        right_hand_pose=tar_pose_aa[:,40*3:55*3],
                        return_verts=True,
                        return_joints=True,
                        leye_pose=tar_pose_aa[:, 69:72],
                        reye_pose=tar_pose_aa[:, 72:75],
                        )
            else:
                vertices_rec = None
                vertices_tar = None

            loss_upper = self.upper_loss(rec_upper, tar_upper, self.cfg.TRAIN.LOSS_6D, vertices_rec, vertices_tar, self.cfg.TRAIN.LOSS_MESH)
            loss_lower = self.lower_loss(rec_lower, tar_lower, self.cfg.TRAIN.LOSS_6D, vertices_rec, vertices_tar, self.cfg.TRAIN.LOSS_MESH)

            betas_loss = torch.tensor(0.0, device=rec_lower.device)
            if rec_lower.shape[2] > 61:
                beta_dim = getattr(self.cfg.DATASET, "LOWER_BETA_DIM", 10)
                w_betas = getattr(self.cfg.LOSS, "LAMBDA_BETAS", 0.0)
                if w_betas > 0:
                    betas_rec = rec_lower[:, :, 61:61 + beta_dim]
                    betas_tar = tar_lower[:, :, 61:61 + beta_dim]
                    betas_loss = F.l1_loss(betas_rec, betas_tar)
                    loss_lower = loss_lower + w_betas * betas_loss

            joint_pos_loss = torch.tensor(0.0, device=rec_lower.device)
            joint_vel_loss = torch.tensor(0.0, device=rec_lower.device)
            joint_acc_loss = torch.tensor(0.0, device=rec_lower.device)
            if use_fk_joints:
                w_joint_pos = getattr(self.cfg.LOSS, "LAMBDA_JOINT_POS", 0.0)
                w_joint_vel = getattr(self.cfg.LOSS, "LAMBDA_JOINT_VEL", 0.0)
                w_joint_acc = getattr(self.cfg.LOSS, "LAMBDA_JOINT_ACC", 0.0)
                if (w_joint_pos + w_joint_vel + w_joint_acc) > 0:
                    rec_rot = rotation_6d_to_matrix(rec_pose.reshape(bs * n, 55, 6)).reshape(bs, n, 55, 3, 3)
                    tar_rot = rotation_6d_to_matrix(tar_pose.reshape(bs * n, 55, 6)).reshape(bs, n, 55, 3, 3)
                    betas_fk = tar_beta[..., :10] if tar_beta.shape[-1] > 10 else tar_beta
                    joints_rec = self._fk_joints(rec_rot, betas_fk, joint_indices=self.upper_lower_joint_indices)
                    joints_tar = self._fk_joints(tar_rot, betas_fk, joint_indices=self.upper_lower_joint_indices)
                    joint_scale = getattr(self.cfg.LOSS, "JOINT_UNIT_SCALE", 1.0)
                    if joint_scale != 1.0:
                        joints_rec = joints_rec * joint_scale
                        joints_tar = joints_tar * joint_scale
                    if w_joint_pos > 0:
                        joint_pos_loss = F.mse_loss(joints_rec, joints_tar)
                    if w_joint_vel > 0 and n > 1:
                        joint_vel_loss = F.mse_loss(
                            joints_rec[:, 1:] - joints_rec[:, :-1],
                            joints_tar[:, 1:] - joints_tar[:, :-1],
                        )
                    if w_joint_acc > 0 and n > 2:
                        joint_acc_loss = F.mse_loss(
                            joints_rec[:, 2:] + joints_rec[:, :-2] - 2 * joints_rec[:, 1:-1],
                            joints_tar[:, 2:] + joints_tar[:, :-2] - 2 * joints_tar[:, 1:-1],
                        )
                    loss_lower = (
                        loss_lower
                        + w_joint_pos * joint_pos_loss
                        + w_joint_vel * joint_vel_loss
                        + w_joint_acc * joint_acc_loss
                    )

            loss_global = self.global_loss(
                rec_xyz_trans,
                rec_local_vel,
                tar_trans,
                tar_local_vel,
            )
            trans_vel_metric = F.l1_loss(rec_local_vel, tar_local_vel).detach()
            trans_pos_metric = F.l1_loss(rec_xyz_trans, tar_trans).detach()

            lengths = batch.get("motion_len", None)

            rs_set = {
                "tar_upper": tar_upper,
                "rec_upper": rec_upper,
                "tar_lower": tar_lower,
                "rec_lower": rec_lower,
                "tar_global": tar_trans,
                "rec_global": rec_xyz_trans,
                "tar_beta": tar_beta,
                "recons-upper_loss": loss_upper,
                "recons-lower_loss": loss_lower,
                "commit-lower_loss": net_out_lower["embedding_loss"],
                "recons-global_loss": loss_global,
                "betas_metric": betas_loss,
                "joint_pos_metric": joint_pos_loss,
                "joint_vel_metric": joint_vel_loss,
                "joint_acc_metric": joint_acc_loss,
                "trans_vel_metric": trans_vel_metric,
                "trans_pos_metric": trans_pos_metric,
                "length": lengths,
            }
            if "perplexity" in net_out_lower:
                rs_set["perplexity"] = net_out_lower["perplexity"]

        elif self.hparams.Selected_part == 'upper':

            tar_beta, tar_upper = [
                batch[key] for key in ["shape", "upper"]
            ]
            upper_dim = self.cfg.model.params.modality_tokenizer.vae_upper.params.vae_test_dim
            net_out_upper = self.vae_upper(tar_upper[..., :upper_dim])
            rec_upper = net_out_upper["rec_pose"]
            bs = tar_upper.shape[0]
            n = min(tar_upper.shape[1], rec_upper.shape[1])
            rec_upper = rec_upper[:,:n]
            tar_upper = tar_upper[:,:n]
            # tar_pose = tar_pose[:,:n]
            tar_beta = tar_beta[:,:n]
            # tar_trans = tar_trans[:,:n]

            rec_pose = self.inverse_selection_tensor_full_body_6D_from_upper(rec_upper.reshape(bs * n, 13 * 6), JOINT_MASK_UPPER_6D, n*bs)
            tar_pose = self.inverse_selection_tensor_full_body_6D_from_upper(tar_upper.reshape(bs * n, 13 * 6), JOINT_MASK_UPPER_6D, n*bs)
            rec_pose_aa = rotation_6d_to_axis_angle(rec_pose.reshape(bs * n, 55, 6)).reshape(bs * n, 55 * 3)
            tar_pose_aa = rotation_6d_to_axis_angle(tar_pose.reshape(bs * n, 55, 6)).reshape(bs * n, 55 * 3)

            if self.cfg.TRAIN.LOSS_MESH:
                if use_fk_joints:
                    rec_rot = rotation_6d_to_matrix(rec_pose.reshape(bs * n, 55, 6)).reshape(bs, n, 55, 3, 3)
                    tar_rot = rotation_6d_to_matrix(tar_pose.reshape(bs * n, 55, 6)).reshape(bs, n, 55, 3, 3)
                    joints_rec = self._fk_joints(rec_rot, tar_beta).reshape(bs * n, -1, 3)
                    joints_tar = self._fk_joints(tar_rot, tar_beta).reshape(bs * n, -1, 3)
                    vertices_rec = {"vertices": joints_rec, "joints": joints_rec}
                    vertices_tar = {"vertices": joints_tar, "joints": joints_tar}
                else:
                    vertices_rec = self.smplx(
                        betas=tar_beta.reshape(bs*n, 300), 
                        transl=torch.zeros((bs*n, 3), device=rec_pose_aa.device), 
                        expression=torch.zeros((bs*n, 100), device=rec_pose_aa.device), 
                        jaw_pose=rec_pose_aa[:, 66:69], 
                        global_orient=rec_pose_aa[:,:3], 
                        body_pose=rec_pose_aa[:,3:21*3+3], 
                        left_hand_pose=rec_pose_aa[:,25*3:40*3], 
                        right_hand_pose=rec_pose_aa[:,40*3:55*3], 
                        return_verts=True,
                        return_joints=True,
                        leye_pose=tar_pose_aa[:, 69:72], 
                        reye_pose=tar_pose_aa[:, 72:75],
                        )
                    vertices_tar = self.smplx(
                        betas=tar_beta.reshape(bs*n, 300), 
                        transl=torch.zeros((bs*n, 3), device=rec_pose_aa.device), 
                        expression=torch.zeros((bs*n, 100), device=rec_pose_aa.device), 
                        jaw_pose=tar_pose_aa[:, 66:69], 
                        global_orient=tar_pose_aa[:,:3], 
                        body_pose=tar_pose_aa[:,3:21*3+3], 
                        left_hand_pose=tar_pose_aa[:,25*3:40*3], 
                        right_hand_pose=tar_pose_aa[:,40*3:55*3], 
                        return_verts=True,
                        return_joints=True,
                        leye_pose=tar_pose_aa[:, 69:72], 
                        reye_pose=tar_pose_aa[:, 72:75],
                        )
            else:
                vertices_rec = None
                vertices_tar = None

            loss_upper = self.upper_loss(rec_upper, tar_upper, self.cfg.TRAIN.LOSS_6D, vertices_rec, vertices_tar, self.cfg.TRAIN.LOSS_MESH)

            # Get lengths if available, otherwise use full sequence length
            lengths = batch.get("motion_len", None)

            rs_set = {
                "tar_upper":tar_upper,
                "rec_upper":rec_upper,
                "tar_beta":tar_beta,
                # "tar_trans":tar_trans,
                "recons-upper_loss": loss_upper,
                "commit-upper_loss": net_out_upper["embedding_loss"],
                "length": lengths  # Use the computed lengths
            }


        elif self.hparams.Selected_part == 'hand':

            tar_beta, tar_hand = [
                batch[key] for key in ["shape", "hand"]
            ]
            hands_dim = self.cfg.model.params.modality_tokenizer.vae_hands.params.vae_test_dim
            net_out_hand = self.vae_hand(tar_hand[..., :hands_dim])
            rec_hand = net_out_hand["rec_pose"]
            bs = tar_hand.shape[0]
            n = min(tar_hand.shape[1], rec_hand.shape[1])
            rec_hand = rec_hand[:,:n]
            tar_hand = tar_hand[:,:n]
            tar_beta = tar_beta[:,:n]

            rec_pose = self.inverse_selection_tensor_full_body_6D_from_hand(rec_hand.reshape(bs * n, 30 * 6), JOINT_MASK_HAND_6D, n*bs)
            tar_pose = self.inverse_selection_tensor_full_body_6D_from_hand(tar_hand.reshape(bs * n, 30 * 6), JOINT_MASK_HAND_6D, n*bs)
            rec_pose_aa = rotation_6d_to_axis_angle(rec_pose.reshape(bs * n, 55, 6)).reshape(bs * n, 55 * 3)
            tar_pose_aa = rotation_6d_to_axis_angle(tar_pose.reshape(bs * n, 55, 6)).reshape(bs * n, 55 * 3)

            if self.cfg.TRAIN.LOSS_MESH:
                if use_fk_joints:
                    rec_rot = rotation_6d_to_matrix(rec_pose.reshape(bs * n, 55, 6)).reshape(bs, n, 55, 3, 3)
                    tar_rot = rotation_6d_to_matrix(tar_pose.reshape(bs * n, 55, 6)).reshape(bs, n, 55, 3, 3)
                    joints_rec = self._fk_joints(rec_rot, tar_beta).reshape(bs * n, -1, 3)
                    joints_tar = self._fk_joints(tar_rot, tar_beta).reshape(bs * n, -1, 3)
                    vertices_rec = {"vertices": joints_rec, "joints": joints_rec}
                    vertices_tar = {"vertices": joints_tar, "joints": joints_tar}
                else:
                    vertices_rec = self.smplx(
                        betas=tar_beta.reshape(bs*n, 300), 
                        transl=torch.zeros((bs*n, 3), device=rec_pose_aa.device), 
                        expression=torch.zeros((bs*n, 100), device=rec_pose_aa.device), 
                        jaw_pose=rec_pose_aa[:, 66:69], 
                        global_orient=rec_pose_aa[:,:3], 
                        body_pose=rec_pose_aa[:,3:21*3+3], 
                        left_hand_pose=rec_pose_aa[:,25*3:40*3], 
                        right_hand_pose=rec_pose_aa[:,40*3:55*3], 
                        return_verts=True,
                        return_joints=True,
                        leye_pose=tar_pose_aa[:, 69:72], 
                        reye_pose=tar_pose_aa[:, 72:75],
                        )
                    vertices_tar = self.smplx(
                        betas=tar_beta.reshape(bs*n, 300), 
                        transl=torch.zeros((bs*n, 3), device=rec_pose_aa.device), 
                        expression=torch.zeros((bs*n, 100), device=rec_pose_aa.device), 
                        jaw_pose=tar_pose_aa[:, 66:69], 
                        global_orient=tar_pose_aa[:,:3], 
                        body_pose=tar_pose_aa[:,3:21*3+3], 
                        left_hand_pose=tar_pose_aa[:,25*3:40*3], 
                        right_hand_pose=tar_pose_aa[:,40*3:55*3], 
                        return_verts=True,
                        return_joints=True,
                        leye_pose=tar_pose_aa[:, 69:72], 
                        reye_pose=tar_pose_aa[:, 72:75],
                        )
            else:
                vertices_rec = None
                vertices_tar = None

            loss_hand = self.hand_loss(rec_hand, tar_hand, self.cfg.TRAIN.LOSS_6D, vertices_rec, vertices_tar, self.cfg.TRAIN.LOSS_MESH)

            # Get lengths if available, otherwise use full sequence length
            lengths = batch.get("motion_len", None)

            rs_set = {
                "tar_hand":tar_hand,
                "rec_hand":rec_hand,
                # "tar_beta":tar_beta,
                "recons-hand_loss": loss_hand,
                "commit-hand_loss": net_out_hand["embedding_loss"],
                "length": lengths  # Use the computed lengths
            }

        elif self.hparams.Selected_part == 'face':

            tar_face = batch[ "face"]
            bs = tar_face.shape[0]
            if bs < self.train_batch_size:
                bs = self.train_batch_size
                tar_face = F.pad(tar_face, (0, 0, 0, 0, 0, self.train_batch_size - tar_face.shape[0]))
            # Process face data through the VQ-VAE
            net_out_face = self.vae_face(tar_face)
            rec_face = net_out_face["rec_pose"]
            n = tar_face.shape[1]
            n = min(tar_face.shape[1], rec_face.shape[1])
            rec_pose_jaw = rotation_6d_to_axis_angle(rec_face[:, :n , :6].reshape(-1, 6)).reshape(-1, 3)
            tar_pose_jaw = rotation_6d_to_axis_angle(tar_face[:, :n, :6].reshape(-1, 6)).reshape(-1, 3)

            if self.cfg.TRAIN.LOSS_MESH== True:
                # Run FLAME model
                rec_joints, rec_vertices = self.forward_flame(
                    exps=rec_face[:, :n, 6:].reshape(-1, 100),
                    jaw=rec_pose_jaw,
                    shape=torch.zeros((bs*n, 100), device=rec_face.device),
                    batch_size=self.max_batch_size_smplx,
                )
                tar_joints, tar_vertices = self.forward_flame(
                    exps=tar_face[:, :n, 6:].reshape(-1, 100),
                    jaw=tar_pose_jaw,
                    shape=torch.zeros((bs*n, 100), device=rec_face.device),
                    batch_size=self.max_batch_size_smplx,
                )
            else:
                rec_vertices = None
                tar_vertices = None

            # Apply face loss
            face_nonzero_mask = (tar_face.abs().sum(dim=-1).sum(dim=-1) > 0)
            if face_nonzero_mask.any():
                loss_face = self.face_loss(
                    rec_face[face_nonzero_mask], 
                    tar_face[face_nonzero_mask], 
                    self.cfg.TRAIN.LOSS_6D,
                    rec_vertices.reshape(bs, n, -1, 3)[face_nonzero_mask] if tar_vertices is not None else None, 
                    tar_vertices.reshape(bs, n, -1, 3)[face_nonzero_mask] if tar_vertices is not None else None, 
                    self.cfg.TRAIN.LOSS_MESH,
                )
            else:
                loss_face = torch.tensor(0.0, device=tar_face.device)
                
            # Extract embedding loss for monitoring
            commit_loss = net_out_face["embedding_loss"]
            # Get lengths if available, otherwise use full sequence length
            lengths = batch.get("motion_len", None)

            rs_set = {
                "tar_face":tar_face,
                "rec_face":rec_face,
                "recons-face_loss": loss_face,
                "commit-face_loss": commit_loss,
                "length": lengths  # Use the computed lengths
            }


        elif self.hparams.Selected_part == 'global':
            tar_beta = batch["shape"]
            tar_lower = batch["lower"]
            tar_trans = batch.get("trans")

            # Global orient is the pelvis rotation (first 6D) in lower representation.
            tar_global_orient_6d = tar_lower[:, :, :6]
            if tar_lower.shape[2] < 57:
                raise ValueError(
                    "Global VAE expects lower with local velocity at dims [54:57]. "
                    f"Got lower dim={tar_lower.shape[2]}."
                )
            tar_local_vel = tar_lower[:, :, 54:57]

            # Input to global VAE. Velocity channels (54:57) are ALWAYS zeroed (they are the target).
            # Foot contacts (57:61) are OPTIONALLY kept (DATASET.GLOBAL_FEED_CONTACTS): a deployable,
            # dataset-agnostic signal (computed from the pose) that separates planted stance
            # (net translation ~0) from gait (translation). This replaces per-dataset conditioning,
            # which is NOT deployable — a generated clip at inference has no dataset label.
            global_dim = self.cfg.model.params.modality_tokenizer.vae_global.params.vae_test_dim
            to_global = tar_lower[..., :global_dim].clone()
            feed_contacts = bool(getattr(self.cfg.DATASET, "GLOBAL_FEED_CONTACTS", False))
            if to_global.shape[2] > 54:
                to_global[:, :, 54:57] = 0.0
                if not (feed_contacts and to_global.shape[2] >= 61):
                    to_global[:, :, 57:] = 0.0
            # Unconditioned by default (deployable). GLOBAL_CONDITION re-enables the dataset token (legacy).
            use_cond = bool(getattr(self.cfg.DATASET, "GLOBAL_CONDITION", False))
            dsid = self._batch_dataset_ids(batch, to_global.device) if use_cond else None
            # Betas (shape) input — deployable: translation magnitude scales with leg length (GLOBAL_FEED_BETAS).
            betas_in = None
            if bool(getattr(self.cfg.DATASET, "GLOBAL_FEED_BETAS", False)):
                _b = tar_beta[:, 0, :] if tar_beta.dim() == 3 else tar_beta
                betas_in = _b[:, :10]   # raw-betas path (10-D)
                if bool(getattr(self.cfg.DATASET, "GLOBAL_MEASURE", False)):
                    m = self._betas_to_measurements(_b)   # FULL native betas -> limb measurements [B,3]
                    betas_in = torch.cat([m, _b[:, :10]], dim=-1) if bool(
                        getattr(self.cfg.DATASET, "GLOBAL_MEASURE_CAT", False)) else m   # cat -> [B,13]
            net_out_global = self.vae_global(to_global, dataset_id=dsid, betas=betas_in)
            rec_global = net_out_global["rec_pose"]
            rec_local_vel = rec_global[:, :, 54:57]

            if tar_trans is None:
                tar_trans, _ = self._integrate_local_velocity(tar_local_vel, tar_global_orient_6d)

            rec_xyz_trans, _ = self._integrate_local_velocity(
                rec_local_vel, tar_global_orient_6d, init_pos=tar_trans[:, 0, :]
            )

            loss_global = self.global_loss(
                rec_xyz_trans,
                rec_local_vel,
                tar_trans,
                tar_local_vel,
            )

            # V3: also regress foot contacts (gated by LOSS.LAMBDA_CONTACT > 0). The global VAE
            # then predicts both velocity (54:57) AND foot contacts (57:61, BCE). The contacts are
            # used at inference for foot-contact translation refinement (pp_static_joint). When
            # LAMBDA_CONTACT == 0 (V1/V2), this is a no-op and behaviour is unchanged.
            lambda_contact = float(getattr(self.cfg.LOSS, "LAMBDA_CONTACT", 0.0) or 0.0)
            if lambda_contact > 0 and tar_lower.shape[2] >= 61:
                contact_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                    rec_global[:, :, 57:61], tar_lower[:, :, 57:61]
                )
                loss_global = loss_global + lambda_contact * contact_loss

            # Stage B: drift/endpoint loss — directly penalize accumulated world-translation drift
            # (endpoint + full-trajectory position error), the dominant failure mode (drift ~1.7x per-frame).
            lambda_drift = float(getattr(self.cfg.LOSS, "LAMBDA_DRIFT", 0.0) or 0.0)
            if lambda_drift > 0:
                drift_loss = (torch.nn.functional.l1_loss(rec_xyz_trans[:, -1, :], tar_trans[:, -1, :])
                              + torch.nn.functional.l1_loss(rec_xyz_trans, tar_trans))
                loss_global = loss_global + lambda_drift * drift_loss

            # Extract embedding loss for monitoring
            commit_loss = torch.tensor(0.0, device=tar_trans.device)
            # Get lengths if available, otherwise use full sequence length
            lengths = batch.get("motion_len", None)

            rs_set = {
                "tar_global":tar_trans,
                "rec_global":rec_xyz_trans,
                "recons-global_loss": loss_global,
                "commit-global_loss": commit_loss,
                "length": lengths  # Use the computed lengths
            }
        elif self.hparams.Selected_part == 'full_rot':
            
            raise NotImplementedError("Not implemented")
        elif self.hparams.Selected_part in {'full_genmo', 'genmo_upper', 'genmo_lower'}:
            tar_motion = batch["motion_vector"]
            selected_part = self.hparams.Selected_part
            body_dim = int(self.cfg.model.params.modality_tokenizer.vae_body.params.vae_test_dim)

            if selected_part == "full_genmo":
                full_dim = tar_motion.shape[-1]
                if body_dim == full_dim:
                    tar_input = tar_motion
                elif body_dim == 135 and full_dim == 145:
                    # 135D GenmoFull (no betas): drop the betas block (126:136) ->
                    # body(126)+global(6)+vel(3). NOT [..., :135], which keeps betas
                    # and drops global_orient/local_vel.
                    tar_input = torch.cat([tar_motion[..., :126], tar_motion[..., 136:145]], dim=-1)
                else:
                    tar_input = tar_motion[..., :body_dim]
                split_indices = None
                metric_joint_indices = None
            else:
                split_indices, metric_joint_indices = self._get_genmo_split_indices(selected_part)
                if len(split_indices) != body_dim:
                    raise ValueError(
                        f"{selected_part} expects vae_body dim={len(split_indices)} from split spec, "
                        f"but config has vae_test_dim={body_dim}."
                    )
                tar_input = tar_motion[..., split_indices]

            net_out_body = self.vae_body(tar_input)
            rec_pred = net_out_body["rec_pose"]

            n = min(tar_motion.shape[1], rec_pred.shape[1])
            rec_pred = rec_pred[:, :n]
            tar_motion = tar_motion[:, :n]

            if selected_part == "full_genmo":
                if rec_pred.shape[-1] == 135 and tar_motion.shape[-1] == 145:
                    # 135D GenmoFull output has no betas: re-insert zero betas to
                    # restore the 145D layout so downstream loss/FK/decode can index
                    # betas(126:136)/global(136:142)/vel(142:145) consistently.
                    zeros_betas = torch.zeros(
                        *rec_pred.shape[:-1], 10, device=rec_pred.device, dtype=rec_pred.dtype
                    )
                    rec_motion = torch.cat(
                        [rec_pred[..., :126], zeros_betas, rec_pred[..., 126:135]], dim=-1
                    )
                else:
                    rec_motion = rec_pred
            else:
                rec_motion = self._merge_genmo_split(rec_pred, tar_motion, split_indices)

            recons_loss = self._recons_loss(rec_motion, tar_motion)
            commit_loss = net_out_body["embedding_loss"]
            joint_pos_loss = torch.tensor(0.0, device=tar_motion.device)
            joint_vel_loss = torch.tensor(0.0, device=tar_motion.device)
            joint_acc_loss = torch.tensor(0.0, device=tar_motion.device)

            if use_fk_joints:
                w_joint_pos = getattr(self.cfg.LOSS, "LAMBDA_JOINT_POS", 0.0)
                w_joint_vel = getattr(self.cfg.LOSS, "LAMBDA_JOINT_VEL", 0.0)
                w_joint_acc = getattr(self.cfg.LOSS, "LAMBDA_JOINT_ACC", 0.0)
                if (w_joint_pos + w_joint_vel + w_joint_acc) > 0:
                    bs = rec_motion.shape[0]
                    rec_body = rec_motion[..., :126].reshape(bs, n, 21, 6)
                    tar_body = tar_motion[..., :126].reshape(bs, n, 21, 6)
                    rec_global = rec_motion[..., 136:142].reshape(bs, n, 1, 6)
                    tar_global = tar_motion[..., 136:142].reshape(bs, n, 1, 6)

                    rec_rot6d = torch.cat([rec_global, rec_body], dim=2).reshape(-1, 6)
                    tar_rot6d = torch.cat([tar_global, tar_body], dim=2).reshape(-1, 6)
                    rec_rot = rotation_6d_to_matrix(rec_rot6d).reshape(bs, n, 22, 3, 3)
                    tar_rot = rotation_6d_to_matrix(tar_rot6d).reshape(bs, n, 22, 3, 3)
                    betas = tar_motion[..., 126:136]

                    joints_rec = self._fk_joints(rec_rot, betas)
                    joints_tar = self._fk_joints(tar_rot, betas)
                    if selected_part in {"genmo_upper", "genmo_lower"} and metric_joint_indices is not None:
                        joints_rec = joints_rec[:, :, metric_joint_indices]
                        joints_tar = joints_tar[:, :, metric_joint_indices]

                    joint_scale = getattr(self.cfg.LOSS, "JOINT_UNIT_SCALE", 1.0)
                    if joint_scale != 1.0:
                        joints_rec = joints_rec * joint_scale
                        joints_tar = joints_tar * joint_scale

                    if w_joint_pos > 0:
                        joint_pos_loss = F.mse_loss(joints_rec, joints_tar)
                    if w_joint_vel > 0 and n > 1:
                        joint_vel_loss = F.mse_loss(
                            joints_rec[:, 1:] - joints_rec[:, :-1],
                            joints_tar[:, 1:] - joints_tar[:, :-1],
                        )
                    if w_joint_acc > 0 and n > 2:
                        joint_acc_loss = F.mse_loss(
                            joints_rec[:, 2:] + joints_rec[:, :-2] - 2 * joints_rec[:, 1:-1],
                            joints_tar[:, 2:] + joints_tar[:, :-2] - 2 * joints_tar[:, 1:-1],
                        )

                    recons_loss = (
                        recons_loss
                        + w_joint_pos * joint_pos_loss
                        + w_joint_vel * joint_vel_loss
                        + w_joint_acc * joint_acc_loss
                    )

            lengths = batch.get("motion_len", None)
            perplexity = net_out_body.get("perplexity", torch.tensor(0.0, device=tar_motion.device))
            rs_set = {
                "tar_motion": tar_motion,
                "rec_motion": rec_motion,
                "recons_loss": recons_loss,
                "commit_loss": commit_loss,
                "joint_pos_metric": joint_pos_loss,
                "joint_vel_metric": joint_vel_loss,
                "joint_acc_metric": joint_acc_loss,
                "perplexity": perplexity,
                "length": lengths,
            }
        else:
            raise NotImplementedError("Not implemented")

        return rs_set

    @torch.no_grad()
    def val_vae_forward(self, batch, split="train"):

        if self.hparams.Selected_part == 'lower_54' or self.hparams.Selected_part == 'lower':
            self.vae_lower.eval()
            tar_beta, tar_lower = [
                batch[key] for key in ["shape", "lower"]
            ]
            lower_dim = self.cfg.model.params.modality_tokenizer.vae_lower.params.vae_test_dim
            net_out_lower = self.vae_lower(tar_lower[..., :lower_dim])
            rec_lower = net_out_lower["rec_pose"]
            bs = tar_lower.shape[0]
            n = min(tar_lower.shape[1], rec_lower.shape[1])
            rec_lower = rec_lower[:,:n]
            tar_lower = tar_lower[:,:n]
            tar_beta = tar_beta[:,:n].reshape(bs*n, 300)
            rec_pose = self.inverse_selection_tensor_full_body_6D_from_lower(rec_lower[:,:,:54].reshape(bs * n, 9 * 6), JOINT_MASK_LOWER_6D, n*bs)
            tar_pose = self.inverse_selection_tensor_full_body_6D_from_lower(tar_lower[:,:,:54].reshape(bs * n, 9 * 6), JOINT_MASK_LOWER_6D, n*bs)
            rec_pose_aa = rotation_6d_to_axis_angle(rec_pose.reshape(bs * n, 55, 6)).reshape(bs * n, 55 * 3)
            tar_pose_aa = rotation_6d_to_axis_angle(tar_pose.reshape(bs * n, 55, 6)).reshape(bs * n, 55 * 3)
            rec_joints, rec_vertices = self.forward_smplx(rec_pose_aa, tar_beta, torch.zeros((bs*n, 100), device=rec_pose_aa.device), torch.zeros((bs*n, 3), device=rec_pose_aa.device), batch_size=self.max_batch_size_smplx)
            tar_joints, tar_vertices = self.forward_smplx(tar_pose_aa, tar_beta, torch.zeros((bs*n, 100), device=tar_pose_aa.device), torch.zeros((bs*n, 3), device=tar_pose_aa.device), batch_size=self.max_batch_size_smplx)

            # Get lengths if available, otherwise use full sequence length
            lengths = batch.get("motion_len", None)
            rs_set = {
                "rec_joints": rec_joints.reshape(bs, n, 22, 3),
                "tar_joints": tar_joints.reshape(bs, n, 22, 3),
                "rec_vertices": rec_vertices.reshape(bs, n, 10475, 3),
                "tar_vertices": tar_vertices.reshape(bs, n, 10475, 3),
                "length": lengths  # Use the computed lengths
            }
            return rs_set
        
        elif self.hparams.Selected_part == 'lower_global':
            self.vae_lower.eval()
            tar_beta, tar_lower, tar_trans = [
                batch[key] for key in ["shape", "lower", "trans"]
            ]
            lower_dim = self.cfg.model.params.modality_tokenizer.vae_lower.params.vae_test_dim
            net_out_lower = self.vae_lower(tar_lower[..., :lower_dim])
            rec_lower = net_out_lower["rec_pose"]
            bs = tar_lower.shape[0]
            n = min(tar_lower.shape[1], rec_lower.shape[1])
            rec_lower = rec_lower[:,:n]
            tar_lower = tar_lower[:,:n]
            tar_beta = tar_beta[:,:n].reshape(bs*n, 300)
            if tar_trans is not None:
                tar_trans = tar_trans[:,:n]
            rec_pose = self.inverse_selection_tensor_full_body_6D_from_lower(rec_lower[:,:,:54].reshape(bs * n, 9 * 6), JOINT_MASK_LOWER_6D, n*bs)
            tar_pose = self.inverse_selection_tensor_full_body_6D_from_lower(tar_lower[:,:,:54].reshape(bs * n, 9 * 6), JOINT_MASK_LOWER_6D, n*bs)
            rec_pose_aa = rotation_6d_to_axis_angle(rec_pose.reshape(bs * n, 55, 6)).reshape(bs * n, 55 * 3)
            tar_pose_aa = rotation_6d_to_axis_angle(tar_pose.reshape(bs * n, 55, 6)).reshape(bs * n, 55 * 3)
            rec_joints, rec_vertices = self.forward_smplx(rec_pose_aa, tar_beta, torch.zeros((bs*n, 100), device=rec_pose_aa.device), torch.zeros((bs*n, 3), device=rec_pose_aa.device), batch_size=self.max_batch_size_smplx)
            tar_joints, tar_vertices = self.forward_smplx(tar_pose_aa, tar_beta, torch.zeros((bs*n, 100), device=tar_pose_aa.device), torch.zeros((bs*n, 3), device=tar_pose_aa.device), batch_size=self.max_batch_size_smplx)

            rec_local_vel = rec_lower[:, :, 54:57]
            tar_local_vel = tar_lower[:, :, 54:57]
            rec_global_orient_6d = rec_lower[:, :, :6]
            tar_global_orient_6d = tar_lower[:, :, :6]

            if tar_trans is None:
                tar_trans, _ = self._integrate_local_velocity(tar_local_vel, tar_global_orient_6d)

            rec_xyz_trans, _ = self._integrate_local_velocity(
                rec_local_vel, rec_global_orient_6d, init_pos=tar_trans[:, 0, :]
            )

            # Get lengths if available, otherwise use full sequence length
            lengths = batch.get("motion_len", None)
            rs_set = {
                "tar_trans": tar_trans,
                "rec_trans": rec_xyz_trans,
                "rec_joints": rec_joints.reshape(bs, n, 22, 3),
                "tar_joints": tar_joints.reshape(bs, n, 22, 3),
                "rec_vertices": rec_vertices.reshape(bs, n, 10475, 3),
                "tar_vertices": tar_vertices.reshape(bs, n, 10475, 3),
                "length": lengths  # Use the computed lengths
            }
            return rs_set
        
        elif self.hparams.Selected_part == 'upper_lower_global':
            self.vae_lower.eval()
            tar_beta = batch["shape"]
            tar_upper = batch["upper"]
            tar_lower = batch["lower"]
            tar_trans = batch.get("trans")
            upper_dim = int(JOINT_MASK_UPPER_6D.sum())
            combined_dim = self.cfg.model.params.modality_tokenizer.vae_lower.params.vae_test_dim
            lower_dim = combined_dim - upper_dim

            tar_upper = tar_upper[..., :upper_dim]
            tar_lower = tar_lower[..., :lower_dim]
            tar_combined = torch.cat([tar_upper, tar_lower], dim=-1)

            net_out_lower = self.vae_lower(tar_combined[..., :combined_dim])
            rec_combined = net_out_lower["rec_pose"]
            bs = tar_combined.shape[0]
            n = min(tar_combined.shape[1], rec_combined.shape[1])
            rec_combined = rec_combined[:, :n]
            tar_upper = tar_upper[:, :n]
            tar_lower = tar_lower[:, :n]
            tar_beta = tar_beta[:, :n].reshape(bs*n, 300)
            if tar_trans is not None:
                tar_trans = tar_trans[:, :n]
            rec_upper = rec_combined[:, :, :upper_dim]
            rec_lower = rec_combined[:, :, upper_dim:upper_dim + lower_dim]

            rec_pose_upper = self.inverse_selection_tensor_full_body_6D_from_upper(
                rec_upper.reshape(bs * n, 13 * 6), JOINT_MASK_UPPER_6D, n*bs
            )
            tar_pose_upper = self.inverse_selection_tensor_full_body_6D_from_upper(
                tar_upper.reshape(bs * n, 13 * 6), JOINT_MASK_UPPER_6D, n*bs
            )
            rec_pose_lower = self.inverse_selection_tensor_full_body_6D_from_lower(
                rec_lower[:, :, :54].reshape(bs * n, 9 * 6), JOINT_MASK_LOWER_6D, n*bs
            )
            tar_pose_lower = self.inverse_selection_tensor_full_body_6D_from_lower(
                tar_lower[:, :, :54].reshape(bs * n, 9 * 6), JOINT_MASK_LOWER_6D, n*bs
            )
            rec_pose = rec_pose_upper + rec_pose_lower
            tar_pose = tar_pose_upper + tar_pose_lower
            rec_pose_aa = rotation_6d_to_axis_angle(rec_pose.reshape(bs * n, 55, 6)).reshape(bs * n, 55 * 3)
            tar_pose_aa = rotation_6d_to_axis_angle(tar_pose.reshape(bs * n, 55, 6)).reshape(bs * n, 55 * 3)
            rec_joints, rec_vertices = self.forward_smplx(
                rec_pose_aa,
                tar_beta,
                torch.zeros((bs*n, 100), device=rec_pose_aa.device),
                torch.zeros((bs*n, 3), device=rec_pose_aa.device),
                batch_size=self.max_batch_size_smplx,
            )
            tar_joints, tar_vertices = self.forward_smplx(
                tar_pose_aa,
                tar_beta,
                torch.zeros((bs*n, 100), device=tar_pose_aa.device),
                torch.zeros((bs*n, 3), device=tar_pose_aa.device),
                batch_size=self.max_batch_size_smplx,
            )

            rec_local_vel = rec_lower[:, :, 54:57]
            tar_local_vel = tar_lower[:, :, 54:57]
            rec_global_orient_6d = rec_lower[:, :, :6]
            tar_global_orient_6d = tar_lower[:, :, :6]

            if tar_trans is None:
                tar_trans, _ = self._integrate_local_velocity(tar_local_vel, tar_global_orient_6d)

            rec_xyz_trans, _ = self._integrate_local_velocity(
                rec_local_vel, rec_global_orient_6d, init_pos=tar_trans[:, 0, :]
            )

            lengths = batch.get("motion_len", None)
            rs_set = {
                "tar_trans": tar_trans,
                "rec_trans": rec_xyz_trans,
                "rec_joints": rec_joints.reshape(bs, n, 22, 3),
                "tar_joints": tar_joints.reshape(bs, n, 22, 3),
                "rec_vertices": rec_vertices.reshape(bs, n, 10475, 3),
                "tar_vertices": tar_vertices.reshape(bs, n, 10475, 3),
                "length": lengths,
            }
            return rs_set

        elif self.hparams.Selected_part == 'upper':
            self.vae_upper.eval()
            tar_beta, tar_upper = [
                batch[key] for key in [ "shape", "upper"]
            ]
            upper_dim = self.cfg.model.params.modality_tokenizer.vae_upper.params.vae_test_dim
            net_out_upper = self.vae_upper(tar_upper[..., :upper_dim])
            rec_upper = net_out_upper["rec_pose"]
            bs = tar_upper.shape[0]
            n = min(tar_upper.shape[1], rec_upper.shape[1])
            rec_upper = rec_upper[:,:n]
            tar_upper = tar_upper[:,:n]
            tar_beta = tar_beta[:,:n].reshape(bs*n, 300)
            rec_pose = self.inverse_selection_tensor_full_body_6D_from_upper(rec_upper.reshape(bs * n, 13 * 6), JOINT_MASK_UPPER_6D, n*bs)
            tar_pose = self.inverse_selection_tensor_full_body_6D_from_upper(tar_upper.reshape(bs * n, 13 * 6), JOINT_MASK_UPPER_6D, n*bs)
            rec_pose_aa = rotation_6d_to_axis_angle(rec_pose.reshape(bs * n, 55, 6)).reshape(bs * n, 55 * 3)
            tar_pose_aa = rotation_6d_to_axis_angle(tar_pose.reshape(bs * n, 55, 6)).reshape(bs * n, 55 * 3)
            rec_joints, rec_vertices = self.forward_smplx(rec_pose_aa, tar_beta, torch.zeros((bs*n, 100), device=rec_pose_aa.device), torch.zeros((bs*n, 3), device=rec_pose_aa.device), batch_size=self.max_batch_size_smplx)
            tar_joints, tar_vertices = self.forward_smplx(tar_pose_aa, tar_beta, torch.zeros((bs*n, 100), device=tar_pose_aa.device), torch.zeros((bs*n, 3), device=tar_pose_aa.device), batch_size=self.max_batch_size_smplx)
            # Get lengths if available, otherwise use full sequence length
            lengths = batch.get("motion_len", None)
            rs_set = {
                "rec_joints": rec_joints.reshape(bs, n, 22, 3),
                "tar_joints": tar_joints.reshape(bs, n, 22, 3),
                "rec_vertices": rec_vertices.reshape(bs, n, 10475, 3),
                "tar_vertices": tar_vertices.reshape(bs, n, 10475, 3),
                "length": lengths  # Use the computed lengths
            }
            return rs_set

        elif self.hparams.Selected_part == 'face':
            self.vae_face.eval()
            tar_face = batch["face"]
            face_dim = self.cfg.model.params.modality_tokenizer.vae_face.params.vae_test_dim
            net_out_face = self.vae_face(tar_face[..., :face_dim])
            rec_face = net_out_face["rec_pose"]
            bs = tar_face.shape[0]
            n = min(tar_face.shape[1], rec_face.shape[1])
            rec_face = rec_face[:,:n]
            tar_face = tar_face[:,:n]
            
            # Extract the necessary parameters for visualization
            rec_exps = rec_face[:, :, 6:]  # Use all 100 dimensions for FLAME
            rec_jaw = rotation_6d_to_axis_angle(rec_face[:, :, :6].reshape(-1, 6)).reshape(-1, 3)
            tar_exps = tar_face[:, :, 6:]  # Use all 100 dimensions for FLAME
            tar_jaw = rotation_6d_to_axis_angle(tar_face[:, :, :6].reshape(-1, 6)).reshape(-1, 3)

            # Generate meshes using FLAME
            rec_joints, rec_vertices_face = self.forward_flame(exps=rec_exps.reshape(-1, 100), jaw=rec_jaw, shape=torch.zeros((bs*n, 10), device=rec_jaw.device), batch_size=self.max_batch_size_smplx)
            tar_joints, tar_vertices_face = self.forward_flame(exps=tar_exps.reshape(-1, 100), jaw=tar_jaw, shape=torch.zeros((bs*n, 10), device=tar_jaw.device), batch_size=self.max_batch_size_smplx)

            # Get lengths if available, otherwise use full sequence length
            lengths = batch.get("motion_len", None)
            rs_set = {
                "rec_joints_face": rec_joints.reshape(bs, n, 56, 3),
                "tar_joints_face": tar_joints.reshape(bs, n, 56, 3),
                "rec_vertices_face": rec_vertices_face.reshape(bs, n, 5023, 3),
                "tar_vertices_face": tar_vertices_face.reshape(bs, n, 5023, 3),
                "length": lengths  # Use the computed lengths
            }
            return rs_set

        elif self.hparams.Selected_part == 'global':

            tar_beta, tar_lower, tar_trans = [
                batch[key] for key in ["shape", "lower", "trans"]
            ]

            to_global = tar_lower.clone()
            to_global[:, :, 54:] = 0.0
            net_out_global = self.vae_global(to_global)
            rec_global = net_out_global["rec_pose"]
            rec_trans = rec_global[:, :, 54:57]
            rec_x_trans = velocity2position(rec_trans[:, :, 0:1], 1/self.motion_fps, tar_trans[:, 0, 0:1])
            rec_z_trans = velocity2position(rec_trans[:, :, 2:3], 1/self.motion_fps, tar_trans[:, 0, 2:3])
            rec_y_trans = rec_trans[:,:,1:2]
            rec_xyz_trans = torch.cat([rec_x_trans, rec_y_trans, rec_z_trans], dim=-1)
            # Get lengths if available, otherwise use full sequence length
            lengths = batch.get("motion_len", None)
            rs_set = {
                "tar_trans": tar_trans,
                "rec_trans": rec_xyz_trans,
                "length": lengths  # Use the computed lengths
            }
            return rs_set
        
        elif self.hparams.Selected_part in {'full_genmo', 'genmo_upper', 'genmo_lower'}:
            self.vae_body.eval()
            tar_motion = batch["motion_vector"]
            selected_part = self.hparams.Selected_part
            body_dim = int(self.cfg.model.params.modality_tokenizer.vae_body.params.vae_test_dim)

            if selected_part == "full_genmo":
                full_dim = tar_motion.shape[-1]
                if body_dim == full_dim:
                    tar_input = tar_motion
                elif body_dim == 135 and full_dim == 145:
                    # 135D GenmoFull (no betas): drop the betas block (126:136).
                    tar_input = torch.cat([tar_motion[..., :126], tar_motion[..., 136:145]], dim=-1)
                else:
                    tar_input = tar_motion[..., :body_dim]
                split_indices = None
            else:
                split_indices, _ = self._get_genmo_split_indices(selected_part)
                if len(split_indices) != body_dim:
                    raise ValueError(
                        f"{selected_part} expects vae_body dim={len(split_indices)} from split spec, "
                        f"but config has vae_test_dim={body_dim}."
                    )
                tar_input = tar_motion[..., split_indices]

            net_out_body = self.vae_body(tar_input)
            rec_pred = net_out_body["rec_pose"]

            n = min(tar_motion.shape[1], rec_pred.shape[1])
            rec_pred = rec_pred[:, :n]
            tar_motion = tar_motion[:, :n]

            if selected_part == "full_genmo":
                if rec_pred.shape[-1] == 135 and tar_motion.shape[-1] == 145:
                    # 135D GenmoFull output has no betas: re-insert zero betas to
                    # restore the 145D layout so downstream loss/FK/decode can index
                    # betas(126:136)/global(136:142)/vel(142:145) consistently.
                    zeros_betas = torch.zeros(
                        *rec_pred.shape[:-1], 10, device=rec_pred.device, dtype=rec_pred.dtype
                    )
                    rec_motion = torch.cat(
                        [rec_pred[..., :126], zeros_betas, rec_pred[..., 126:135]], dim=-1
                    )
                else:
                    rec_motion = rec_pred
            else:
                rec_motion = self._merge_genmo_split(rec_pred, tar_motion, split_indices)

            lengths = batch.get("motion_len", None)
            trans = batch.get("trans", None)
            if trans is not None:
                trans = trans[:, :n]

            rec_pose, rec_betas, rec_expr, rec_trans = self._decode_genmo_to_smplx(rec_motion, trans=trans)
            tar_pose, tar_betas, tar_expr, tar_trans = self._decode_genmo_to_smplx(tar_motion, trans=trans)

            rec_joints, rec_vertices = self.forward_smplx(
                rec_pose, rec_betas, rec_expr, rec_trans, batch_size=self.max_batch_size_smplx
            )
            tar_joints, tar_vertices = self.forward_smplx(
                tar_pose, tar_betas, tar_expr, tar_trans, batch_size=self.max_batch_size_smplx
            )

            rs_set = {
                "rec_motion": rec_motion,
                "tar_motion": tar_motion,
                "rec_joints": rec_joints.reshape(rec_motion.shape[0], n, 22, 3),
                "tar_joints": tar_joints.reshape(rec_motion.shape[0], n, 22, 3),
                "rec_vertices": rec_vertices.reshape(rec_motion.shape[0], n, 10475, 3),
                "tar_vertices": tar_vertices.reshape(rec_motion.shape[0], n, 10475, 3),
                "length": lengths,
            }
            return rs_set

        elif self.hparams.Selected_part == 'compositional':
            self.vae_upper.eval()
            tar_beta, tar_upper = [
                batch[key] for key in ["shape", "upper"]
            ]
            upper_dim = self.cfg.model.params.modality_tokenizer.vae_upper.params.vae_test_dim

            net_out_upper = self.vae_upper(tar_upper[..., :upper_dim])

            rec_upper = net_out_upper["rec_pose"]

            # Get the lengths if available, otherwise use full sequence length
            lengths = batch.get("motion_len", None)
            bs = tar_upper.shape[0]
            n = min(tar_upper.shape[1], rec_upper.shape[1])
            rec_upper = rec_upper[:,:n]
            tar_upper = tar_upper[:,:n]
            
            tar_beta = tar_beta[:,:n].reshape(bs*n, 300)
            rec_pose = self.inverse_selection_tensor_full_body_6D_from_upper(rec_upper.reshape(bs * n, 13 * 6), JOINT_MASK_UPPER_6D, n*bs)
            tar_pose = self.inverse_selection_tensor_full_body_6D_from_upper(tar_upper.reshape(bs * n, 13 * 6), JOINT_MASK_UPPER_6D, n*bs)
            rec_pose_aa = rotation_6d_to_axis_angle(rec_pose.reshape(bs * n, 55, 6)).reshape(bs * n, 55 * 3)
            tar_pose_aa = rotation_6d_to_axis_angle(tar_pose.reshape(bs * n, 55, 6)).reshape(bs * n, 55 * 3)
            rec_joints, rec_vertices = self.forward_smplx(rec_pose_aa, tar_beta, torch.zeros((bs*n, 100), device=rec_pose_aa.device), torch.zeros((bs*n, 3), device=rec_pose_aa.device), batch_size=self.max_batch_size_smplx)
            tar_joints, tar_vertices = self.forward_smplx(tar_pose_aa, tar_beta, torch.zeros((bs*n, 100), device=tar_pose_aa.device), torch.zeros((bs*n, 3), device=tar_pose_aa.device), batch_size=self.max_batch_size_smplx)
            # # Generate meshes using FLAME
            # rec_joints_face, rec_vertices_face = self.forward_flame(exps=rec_exps.reshape(-1, 100), jaw=rec_pose_jaw_aa, shape=torch.zeros((bs*n, 10), device=rec_pose_jaw_aa.device), batch_size=self.max_batch_size_smplx)
            # tar_joints_face, tar_vertices_face = self.forward_flame(exps=tar_exps.reshape(-1, 100), jaw=tar_pose_jaw_aa, shape=torch.zeros((bs*n, 10), device=tar_pose_jaw_aa.device), batch_size=self.max_batch_size_smplx)

            # Get lengths if available, otherwise use full sequence length
            lengths = batch.get("motion_len", None)
            rs_set = {
                "rec_joints": rec_joints.reshape(bs, n, 22, 3),
                "tar_joints": tar_joints.reshape(bs, n, 22, 3),
                "rec_vertices": rec_vertices.reshape(bs, n, 10475, 3),
                "tar_vertices": tar_vertices.reshape(bs, n, 10475, 3),
                # "rec_joints_face": rec_joints_face.reshape(bs, n, 56, 3),
                # "tar_joints_face": tar_joints_face.reshape(bs, n, 56, 3),
                # "rec_vertices_face": rec_vertices_face.reshape(bs, n, 5023, 3),
                # "tar_vertices_face": tar_vertices_face.reshape(bs, n, 5023, 3),
                "length": lengths  # Use the computed lengths
            }
            return rs_set
        
        elif self.hparams.Selected_part == 'compositional_full':
            self.vae_face.eval()
            self.vae_hand.eval()
            self.vae_upper.eval()
            self.vae_lower.eval()
            tar_beta, tar_lower, tar_face, tar_hand, tar_upper = [
                batch[key] for key in ["shape", "lower", "face", "hand", "upper"]
            ]
            lower_dim = self.cfg.model.params.modality_tokenizer.vae_lower.params.vae_test_dim
            face_dim = self.cfg.model.params.modality_tokenizer.vae_face.params.vae_test_dim
            hand_dim = self.cfg.model.params.modality_tokenizer.vae_hand.params.vae_test_dim
            upper_dim = self.cfg.model.params.modality_tokenizer.vae_upper.params.vae_test_dim

            net_out_face = self.vae_face(tar_face[..., :face_dim])
            net_out_hand = self.vae_hand(tar_hand[..., :hand_dim])
            net_out_upper = self.vae_upper(tar_upper[..., :upper_dim])
            net_out_lower = self.vae_lower(tar_lower[..., :lower_dim])

            rec_face = net_out_face["rec_pose"]
            rec_hand = net_out_hand["rec_pose"]
            rec_upper = net_out_upper["rec_pose"]
            rec_lower = net_out_lower["rec_pose"]
            rec_lower = rec_lower[..., :lower_dim]
            tar_lower = tar_lower[..., :lower_dim]

            # Get the lengths if available, otherwise use full sequence length
            lengths = batch.get("motion_len", None)
            bs = tar_lower.shape[0]
            n = min(tar_lower.shape[1], rec_lower.shape[1])
            rec_lower = rec_lower[:,:n]
            rec_face = rec_face[:,:n]
            rec_pose_jaw = rec_face[:, :, :6]
            rec_hand = rec_hand[:,:n]
            rec_upper = rec_upper[:,:n]
            tar_lower = tar_lower[:,:n]
            tar_face = tar_face[:,:n]
            tar_pose_jaw = tar_face[:, :, :6]
            rec_exps = rec_face[:, :, 6:]  # Use all 100 dimensions for FLAME
            rec_pose_jaw_aa = rotation_6d_to_axis_angle(rec_pose_jaw.reshape(-1, 6)).reshape(-1, 3)
            tar_exps = tar_face[:, :, 6:]  # Use all 100 dimensions for FLAME
            tar_pose_jaw_aa = rotation_6d_to_axis_angle(tar_pose_jaw.reshape(-1, 6)).reshape(-1, 3)
            tar_hand = tar_hand[:,:n]   
            tar_upper = tar_upper[:,:n]
            
            tar_beta = tar_beta[:,:n].reshape(bs*n, 300)
            rec_pose = self.inverse_selection_tensor_full_body_6D(rec_pose_jaw.reshape(bs * n, 6), rec_hand.reshape(bs * n, 30 * 6), rec_lower.reshape(bs * n, 9 * 6), rec_upper.reshape(bs * n, 13 * 6), JOINT_MASK_FACE_6D, JOINT_MASK_HAND_6D, JOINT_MASK_LOWER_6D, JOINT_MASK_UPPER_6D, n*bs)
            tar_pose = self.inverse_selection_tensor_full_body_6D(tar_pose_jaw.reshape(bs * n, 6), tar_hand.reshape(bs * n, 30 * 6), tar_lower.reshape(bs * n, 9 * 6), tar_upper.reshape(bs * n, 13 * 6), JOINT_MASK_FACE_6D, JOINT_MASK_HAND_6D, JOINT_MASK_LOWER_6D, JOINT_MASK_UPPER_6D, n*bs)
            rec_pose_aa = rotation_6d_to_axis_angle(rec_pose.reshape(bs * n, 55, 6)).reshape(bs * n, 55 * 3)
            tar_pose_aa = rotation_6d_to_axis_angle(tar_pose.reshape(bs * n, 55, 6)).reshape(bs * n, 55 * 3)
            rec_joints, rec_vertices = self.forward_smplx(rec_pose_aa, tar_beta, torch.zeros((bs*n, 100), device=rec_pose_aa.device), torch.zeros((bs*n, 3), device=rec_pose_aa.device), batch_size=self.max_batch_size_smplx)
            tar_joints, tar_vertices = self.forward_smplx(tar_pose_aa, tar_beta, torch.zeros((bs*n, 100), device=tar_pose_aa.device), torch.zeros((bs*n, 3), device=tar_pose_aa.device), batch_size=self.max_batch_size_smplx)
            # Generate meshes using FLAME
            rec_joints_face, rec_vertices_face = self.forward_flame(exps=rec_exps.reshape(-1, 100), jaw=rec_pose_jaw_aa, shape=torch.zeros((bs*n, 10), device=rec_pose_jaw_aa.device), batch_size=self.max_batch_size_smplx)
            tar_joints_face, tar_vertices_face = self.forward_flame(exps=tar_exps.reshape(-1, 100), jaw=tar_pose_jaw_aa, shape=torch.zeros((bs*n, 10), device=tar_pose_jaw_aa.device), batch_size=self.max_batch_size_smplx)

            # Get lengths if available, otherwise use full sequence length
            lengths = batch.get("motion_len", None)
            rs_set = {
                "rec_joints": rec_joints.reshape(bs, n, 22, 3),
                "tar_joints": tar_joints.reshape(bs, n, 22, 3),
                "rec_vertices": rec_vertices.reshape(bs, n, 10475, 3),
                "tar_vertices": tar_vertices.reshape(bs, n, 10475, 3),
                "rec_joints_face": rec_joints_face.reshape(bs, n, 56, 3),
                "tar_joints_face": tar_joints_face.reshape(bs, n, 56, 3),
                "rec_vertices_face": rec_vertices_face.reshape(bs, n, 5023, 3),
                "tar_vertices_face": tar_vertices_face.reshape(bs, n, 5023, 3),
                "length": lengths  # Use the computed lengths
            }
            return rs_set
        else:
            raise NotImplementedError("Not implemented")

        # """
        # Validation forward pass for the face tokenizer
        # """
        # self.vae_face.eval()

        # # Extract face data (assuming dataloader has already formatted it properly)
        # tar_pose, tar_beta, tar_face = [
        #     batch[key] for key in ["pose", "shape", "face"]
        # ]
        
        # # Process face data through the VQ-VAE
        # net_out_face = self.vae_face(tar_face)
        # rec_face = net_out_face["rec_pose"]
        
        # # Generate FLAME meshes for visualization and metrics
        # bs = tar_face.shape[0]
        # n = tar_face.shape[1]
        
        # # Extract the necessary parameters for visualization
        # rec_exps = rec_face[:, :, 12:]  # Use all 100 dimensions for FLAME
        # rec_jaw = rotation_6d_to_axis_angle(rec_face[:, :, 6:12].reshape(-1, 6)).reshape(-1, 3)
        # rec_global_orient = rotation_6d_to_axis_angle(rec_face[:, :, :6].reshape(-1, 6)).reshape(-1, 3)
        # tar_exps = tar_face[:, :, 12:]  # Use all 100 dimensions for FLAME
        # tar_jaw = rotation_6d_to_axis_angle(tar_face[:, :, 6:12].reshape(-1, 6)).reshape(-1, 3)
        # tar_global_orient = rotation_6d_to_axis_angle(tar_pose[:, :, :6].reshape(-1, 6)).reshape(-1, 3)
        

        # # Generate meshes using FLAME
        # vertices_rec = self.flame(
        #     global_orient=rec_global_orient,
        #     expression=rec_exps.reshape(-1, 100),
        #     jaw_pose=rec_jaw,
        #     shape=tar_beta.reshape(-1, 10)[:, :10],
        # )
        
        # vertices_tar = self.flame(
        #     global_orient=tar_global_orient,
        #     expression=tar_exps.reshape(-1, 100),
        #     jaw_pose=tar_jaw,
        #     shape=tar_beta.reshape(-1, 10)[:, :10],
        # )

        # # Get lengths if available
        # lengths = batch.get("motion_len", None)
        
        # # Return evaluation metrics
        # rs_set = {
        #     "rec_vertices": vertices_rec.vertices.reshape(bs, n, -1, 3),
        #     "tar_vertices": vertices_tar.vertices.reshape(bs, n, -1, 3),
        #     "rec_joints": vertices_rec.joints.reshape(bs, n, -1, 3),
        #     "tar_joints": vertices_tar.joints.reshape(bs, n, -1, 3),
        #     "length": lengths
        # }
        # return rs_set

    def allsplit_step(self, split, batch, batch_idx):
        loss = None
        # Determine which forward method to use based on stage
        if (self.hparams.stage == "vae" or self.hparams.stage == "vqvae") and split in ["train"]:
            rs_set = self.train_vae_forward(batch)
            loss = self._losses['losses_' + split].update(rs_set)
            
        # Compute the metrics
        if split in ["val", "test"]:
            if self.hparams.stage == "vae" or self.hparams.stage == "vqvae":
                if not self.hparams.metrics_dict:
                    return loss
                rs_set = self.val_vae_forward(batch, split)

                # Update metrics
                for metric in self.hparams.metrics_dict:
                    if metric == "RotationMetrics":
                        getattr(self.metrics, metric).update(
                            rec_joints=rs_set["rec_joints"],
                            tar_joints=rs_set["tar_joints"],
                            rec_vertices=rs_set["rec_vertices"],
                            tar_vertices=rs_set["tar_vertices"],
                            lengths=rs_set['length']
                        )
                    elif metric == "BodyMetrics":
                        getattr(self.metrics, metric).update(
                            rec_joints=rs_set["rec_joints"],
                            tar_joints=rs_set["tar_joints"],
                            rec_vertices=rs_set["rec_vertices"],
                            tar_vertices=rs_set["tar_vertices"],
                            lengths=rs_set['length']
                        )
                    elif metric == "GlobalMetrics":
                        getattr(self.metrics, metric).update(
                            rec_trans=rs_set["rec_trans"],
                            tar_trans=rs_set["tar_trans"],
                            lengths=rs_set['length']  # Pass the computed lengths
                        )

        return loss
