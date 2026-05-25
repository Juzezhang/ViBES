import os
import sys
import time
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch
from tqdm import tqdm
from einops import einsum
from omegaconf import OmegaConf

from multimodal_tokenizers.config import parse_args
from multimodal_tokenizers.data.build_data import build_data
from multimodal_tokenizers.models.build_model import build_model
from multimodal_tokenizers.utils.load_checkpoint import (
    load_pretrained_vae_body,
    load_pretrained_vae_compositional,
    load_pretrained_vae_upper,
)
from multimodal_tokenizers.utils.rotation_conversions import (
    axis_angle_to_matrix,
    rotation_6d_to_matrix,
    matrix_to_axis_angle,
    rotation_6d_to_axis_angle,
)
from multimodal_tokenizers.data.mixed_dataset.data_tools import (
    JOINT_MASK_FACE_6D,
    JOINT_MASK_HANDS_6D,
    JOINT_MASK_LOWER_6D,
    JOINT_MASK_UPPER_6D,
)

from pytorch3d.structures import Meshes
from pytorch3d.structures.meshes import join_meshes_as_scene
from pytorch3d.renderer import TexturesVertex, Materials
from pytorch3d.utils import ico_sphere

ROOT_DIR = Path(__file__).resolve().parents[1]
GVHMR_ASSET_DIR = ROOT_DIR / "model_files" / "gvhmr"

from utils.genmo.pylogger import Log
from utils.genmo.video_io_utils import save_video
from utils.genmo.vis.renderer import Renderer, get_global_cameras_static, get_ground_params_from_points
from utils.genmo.geo_transform import apply_T_on_points, compute_T_ayfz2ay
from utils.genmo.camera import create_camera_sensor
from utils.genmo.smplx_utils import make_smplx

# GENMO 145D joint layout: 21 body joints x 6D rotation = 126D, then betas(10), global_orient(6), local_vel(3)
# Lower body joint indices (within 21 body joints): 0,1,3,4,6,7,9,10 = 8 joints
GENMO_LOWER_JOINT_INDICES = [0, 1, 3, 4, 6, 7, 9, 10]
# Upper body joint indices: 2,5,8,11,12,13,14,15,16,17,18,19,20 = 13 joints
GENMO_UPPER_JOINT_INDICES = [2, 5, 8, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]

# Contact joint IDs in the SMPL joint regressor (ankle/foot joints)
CONTACT_JOINT_IDS = [7, 10, 8, 11]  # L_Ankle, L_Foot, R_Ankle, R_Foot
CONTACT_VEL_THRESHOLD = 0.15  # m/s


def insert_zero_betas(motion_135):
    """
    Insert zero betas (10D) at index 126 to convert 135D GenmoFull (no betas) to 145D.
    135D layout: body_r6d(126) + global_r6d(6) + local_vel(3)
    145D layout: body_r6d(126) + betas(10) + global_r6d(6) + local_vel(3)
    """
    seq_len = motion_135.shape[0]
    zeros = torch.zeros(seq_len, 10, device=motion_135.device, dtype=motion_135.dtype)
    return torch.cat([motion_135[:, :126], zeros, motion_135[:, 126:]], dim=-1)


def split_genmo_to_lower_upper(motion_145, include_global_orient=False):
    """
    Split a full 145D GENMO vector into lower and upper parts.
    Lower: 8 lower joints x 6D (48D) + betas(10D) [+ global_orient(6D)] + local_vel(3D)
      - include_global_orient=True:  67D (48+10+6+3)
      - include_global_orient=False: 61D (48+10+3)
    Upper (78D): 13 upper joints x 6D
    """
    seq_len = motion_145.shape[0]
    body_r6d = motion_145[:, :126].reshape(seq_len, 21, 6)
    betas = motion_145[:, 126:136]
    global_r6d = motion_145[:, 136:142]
    local_vel = motion_145[:, 142:145]

    lower_joints = body_r6d[:, GENMO_LOWER_JOINT_INDICES].reshape(seq_len, -1)  # (T, 48)
    upper_joints = body_r6d[:, GENMO_UPPER_JOINT_INDICES].reshape(seq_len, -1)  # (T, 78)

    if include_global_orient:
        lower = torch.cat([lower_joints, betas, global_r6d, local_vel], dim=-1)  # (T, 67)
    else:
        lower = torch.cat([lower_joints, betas, local_vel], dim=-1)  # (T, 61)
    return lower, upper_joints, global_r6d


def merge_lower_upper_to_genmo(lower, upper, global_r6d_gt, include_global_orient=False):
    """
    Merge lower and upper back into full 145D GENMO vector.
    Uses GT global_orient when not included in lower VAE.
    """
    seq_len = lower.shape[0]
    lower_joints = lower[:, :48].reshape(seq_len, 8, 6)
    betas = lower[:, 48:58]

    if include_global_orient:
        global_r6d = lower[:, 58:64]
        local_vel = lower[:, 64:67]
    else:
        global_r6d = global_r6d_gt  # Use GT global orient
        local_vel = lower[:, 58:61]

    upper_joints = upper.reshape(seq_len, 13, 6)

    body_r6d = torch.zeros(seq_len, 21, 6, device=lower.device, dtype=lower.dtype)
    for i, idx in enumerate(GENMO_LOWER_JOINT_INDICES):
        body_r6d[:, idx] = lower_joints[:, i]
    for i, idx in enumerate(GENMO_UPPER_JOINT_INDICES):
        body_r6d[:, idx] = upper_joints[:, i]

    return torch.cat([body_r6d.reshape(seq_len, 126), betas, global_r6d, local_vel], dim=-1)


def compute_contact_labels(joints_3d, fps=30.0, vel_thr=CONTACT_VEL_THRESHOLD):
    """
    Compute contact labels for ankle/foot joints based on velocity threshold.
    joints_3d: (L, J, 3) tensor
    Returns: (L, len(CONTACT_JOINT_IDS)) boolean tensor
    """
    diff = joints_3d[1:] - joints_3d[:-1]
    velocity = torch.norm(diff, dim=-1) * fps  # (L-1, J)
    velocity = torch.cat([velocity, velocity[-1:]], dim=0)  # (L, J)
    contact = velocity[:, CONTACT_JOINT_IDS] < vel_thr  # (L, 4)
    return contact


def decode_genmo_to_smplx(motion_vector):
    """
    Decode GENMO motion_vector (145D) to SMPL-X params.
    motion_vector: (T, 145)
    """
    seq_len = motion_vector.shape[0]

    body_r6d = motion_vector[:, :126].reshape(seq_len, 21, 6)
    betas = motion_vector[:, 126:136]
    global_r6d = motion_vector[:, 136:142]
    local_vel = motion_vector[:, 142:145]

    body_R = rotation_6d_to_matrix(body_r6d.reshape(-1, 6)).reshape(seq_len, 21, 3, 3)
    body_aa = matrix_to_axis_angle(body_R.reshape(-1, 3, 3)).reshape(seq_len, 63)
    global_R = rotation_6d_to_matrix(global_r6d)
    global_aa = matrix_to_axis_angle(global_R)

    world_vel = torch.einsum("tij,tj->ti", global_R, local_vel)
    trans = torch.cumsum(world_vel, dim=0)

    return body_aa, global_aa, trans, betas


def _normalize_vertices(verts_ay, J_regressor):
    v = verts_ay.clone()
    offset = einsum(J_regressor, v[0], "j v, v i -> j i")[0]
    offset[1] = v[:, :, 1].min()
    v -= offset
    T_rot = compute_T_ayfz2ay(einsum(J_regressor, v[[0]], "j v, l v i -> l j i"), inverse=True)
    return apply_T_on_points(v, T_rot)


def genmo_vertices_and_joints(motion_vector, smplx_model, smplx2smpl, J_regressor, max_seconds, fps):
    """
    Convert GENMO motion_vector to world-space vertices/joints for rendering.
    Uses the same pipeline as preprocess/render_genmo_converted_amass.py.
    """
    body_pose, global_orient, transl, betas = decode_genmo_to_smplx(motion_vector)

    total_frames = body_pose.shape[0]
    max_frames = int(max_seconds * fps)
    seq_len = min(total_frames, max_frames)

    body_pose = body_pose[:seq_len]
    global_orient = global_orient[:seq_len]
    transl = transl[:seq_len]
    betas = betas[:seq_len]

    betas_single = betas[0] if betas.ndim > 1 else betas
    smplx_params = {
        "body_pose": body_pose.cuda(),
        "global_orient": global_orient.cuda(),
        "transl": transl.cuda(),
        "betas": betas_single.unsqueeze(0).expand(seq_len, -1).cuda(),
    }

    smpl_out = smplx_model(**smplx_params)
    verts_ay = torch.stack([torch.matmul(smplx2smpl, v) for v in smpl_out.vertices])
    verts_glob = _normalize_vertices(verts_ay, J_regressor)
    joints_glob = einsum(J_regressor, verts_glob, "j v, l v i -> l j i")
    return verts_glob, joints_glob, seq_len


def smplx_vertices_and_joints_from_aa(pose_aa, transl, betas, smplx_model, smplx2smpl, J_regressor, max_seconds, fps):
    """
    Convert SMPL-X axis-angle pose + translation to world-space vertices/joints for rendering.
    Uses the same normalization as preprocess/render_genmo_converted_amass.py.
    pose_aa: (T, 55*3)
    transl: (T, 3)
    betas: (T, 10) or (T, 300)
    """
    total_frames = pose_aa.shape[0]
    max_frames = int(max_seconds * fps)
    seq_len = min(total_frames, max_frames)

    pose_aa = pose_aa[:seq_len]
    transl = transl[:seq_len]
    betas = betas[:seq_len]

    if betas.shape[1] > 10:
        betas = betas[:, :10]
    betas_single = betas[0] if betas.ndim > 1 else betas

    smplx_params = {
        "global_orient": pose_aa[:, :3].cuda(),
        "body_pose": pose_aa[:, 3:66].cuda(),
        "jaw_pose": pose_aa[:, 66:69].cuda(),
        "leye_pose": pose_aa[:, 69:72].cuda(),
        "reye_pose": pose_aa[:, 72:75].cuda(),
        "transl": transl.cuda(),
        "betas": betas_single.unsqueeze(0).expand(seq_len, -1).cuda(),
    }

    # Handle hand pose dimensions: GVHMR BodyModelSMPLX uses PCA (12 dims) by default.
    if hasattr(smplx_model, "hand_pose_dim"):
        hand_pose_dim = smplx_model.hand_pose_dim
    else:
        bm = getattr(smplx_model, "bm", None)
        if bm is not None and hasattr(bm, "use_pca"):
            hand_pose_dim = bm.num_pca_comps if bm.use_pca else 45
        else:
            use_pca = getattr(smplx_model, "use_pca", False)
            hand_pose_dim = smplx_model.num_pca_comps if use_pca else 45

    if hand_pose_dim == 45:
        smplx_params["left_hand_pose"] = pose_aa[:, 75:120].cuda()
        smplx_params["right_hand_pose"] = pose_aa[:, 120:165].cuda()

    smpl_out = smplx_model(**smplx_params)
    verts_ay = torch.stack([torch.matmul(smplx2smpl, v) for v in smpl_out.vertices])
    verts_glob = _normalize_vertices(verts_ay, J_regressor)
    joints_glob = einsum(J_regressor, verts_glob, "j v, l v i -> l j i")
    return verts_glob, joints_glob, seq_len


def _override_datasets_for_render(cfg):
    """Optionally override DATASET.datasets with preprocessed entries for rendering."""
    render_cfg = OmegaConf.select(cfg, "TEST.RENDER")
    if render_cfg is None:
        return None
    variant = OmegaConf.select(render_cfg, "PREPROCESSED_VARIANT")
    if not variant:
        return None
    if variant == "lower" and cfg.Selected_part in {"compositional", "upper_lower_global"}:
        raise ValueError(
            "PREPROCESSED_VARIANT=lower is incompatible with compositional rendering. "
            "Use PREPROCESSED_VARIANT=upper_lower."
        )
    variant_dirs = OmegaConf.select(cfg, f"DATASET.PREPROCESSED_DIRS.{variant}")
    if not variant_dirs:
        return None
    datasets_override = []
    for entry in cfg.DATASET.datasets:
        name = entry.get("name") if isinstance(entry, dict) else getattr(entry, "name", None)
        if name is None:
            continue
        pre_dir = variant_dirs.get(name)
        if not pre_dir:
            continue
        override = {"name": name, "preprocessed_dir": pre_dir}
        datasets_override.append(override)
    if not datasets_override:
        return None
    return datasets_override


def main():
    cfg = parse_args(phase="test")

    pl.seed_everything(cfg.SEED_VALUE)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    selected_part = cfg.Selected_part
    is_genmo = cfg.DATASET.motion_representation == "genmo" and selected_part == "full_genmo"
    is_genmo_lower = cfg.DATASET.get("motion_representation", "") == "genmo" and selected_part == "genmo_lower"
    genmo_full_include_betas = cfg.DATASET.get("GENMO_FULL_INCLUDE_BETAS", True)
    if not is_genmo and not is_genmo_lower and selected_part not in {"compositional", "upper_lower_global", "lower_global"}:
        raise ValueError(
            "render_genmo_reconstruction only supports full_genmo, genmo_lower, or compositional configs. "
            f"Got Selected_part={selected_part}."
        )

    render_cfg = getattr(cfg.TEST, "RENDER", None)
    use_face = bool(getattr(render_cfg, "USE_FACE", False)) if render_cfg is not None else False
    if not use_face:
        cfg.TEST.CHECKPOINTS_FACE = ""

    model = build_model(cfg)
    if is_genmo or is_genmo_lower:
        # Use phase="token" to load from cfg.TEST.CHECKPOINTS_BODY
        load_pretrained_vae_body(cfg, model, None, phase="token")
        if is_genmo_lower and hasattr(model, "vae_upper"):
            # Use phase="token" to load from cfg.TEST.CHECKPOINTS_UPPER
            load_pretrained_vae_upper(cfg, model, Log, phase="token")
    else:
        load_pretrained_vae_compositional(cfg, model, None, phase="test")
    model = model.to(device)
    model.eval()

    render_splits = ["test"]
    if render_cfg is not None:
        render_splits = list(getattr(render_cfg, "SPLITS", render_splits))
        max_samples_per_split = int(getattr(render_cfg, "NUM_SAMPLES_PER_SPLIT", 5))
        max_seconds = float(getattr(render_cfg, "MAX_SECONDS", 10.0))
        # Always derive translation from local velocity for reconstruction rendering.
        use_local_vel_trans = True
    else:
        max_samples_per_split = int(getattr(cfg.TEST, "NUM_SAMPLES", 5))
        max_seconds = float(getattr(cfg.TEST, "MAX_SECONDS", 10.0))
        use_local_vel_trans = True

    run_name = getattr(cfg, "NAME", "genmo_recon")
    base_folder = getattr(cfg, "FOLDER", "experiments")
    model_name = cfg.model.target.split(".")[-2]
    time_tag = getattr(cfg, "TIME", time.strftime("%Y-%m-%d-%H-%M-%S"))
    folder_exp = getattr(cfg, "FOLDER_EXP", os.path.join(base_folder, model_name, run_name, time_tag))
    output_dir = Path(folder_exp) / "genmo_recon_videos"

    Log.info("Initializing GENMO renderer...")
    smplx_model = make_smplx("supermotion").cuda()
    smplx2smpl = torch.load(GVHMR_ASSET_DIR / "smplx2smpl_sparse.pt").cuda()
    faces_smpl = make_smplx("smpl").faces
    J_regressor = torch.load(GVHMR_ASSET_DIR / "smpl_neutral_J_regressor.pt").cuda()

    if render_cfg is not None:
        width = int(getattr(render_cfg, "WIDTH", 1280))
        height = int(getattr(render_cfg, "HEIGHT", 720))
    else:
        width = int(getattr(cfg.TEST, "RENDER_WIDTH", 1280)) if hasattr(cfg, "TEST") else 1280
        height = int(getattr(cfg.TEST, "RENDER_HEIGHT", 720)) if hasattr(cfg, "TEST") else 720
    _, _, K = create_camera_sensor(width, height, 24)
    renderer = Renderer(width, height, device="cuda", faces=faces_smpl, K=K, bin_size=0)
    color = torch.tensor([0.69, 0.39, 0.96]).cuda()

    # Contact visualization resources
    sphere_mesh = ico_sphere(level=1, device=device)
    sphere_verts_template = sphere_mesh.verts_list()[0] * 0.05  # radius 0.05m
    sphere_faces = sphere_mesh.faces_list()[0]
    contact_color = torch.tensor([1.0, 1.0, 0.0]).to(device)  # yellow
    faces_t = torch.from_numpy(faces_smpl.astype(np.int64)).to(device)

    total_saved = 0
    for split in render_splits:
        if split == "train":
            cfg.TRAIN.SPLIT = "train"
        elif split == "val":
            cfg.EVAL.SPLIT = "val"
        else:
            cfg.TEST.SPLIT = "test"

        datasets_backup = cfg.DATASET.datasets
        datasets_override = _override_datasets_for_render(cfg)
        if datasets_override is not None:
            cfg.DATASET.datasets = datasets_override
        datamodule = build_data(cfg, phase="test")
        cfg.DATASET.datasets = datasets_backup

        if split == "train":
            dataloader = datamodule.train_dataloader()
        elif split == "val":
            dataloader = datamodule.val_dataloader()
        else:
            dataloader = datamodule.test_dataloader()

        split_output_dir = output_dir / split
        split_output_dir.mkdir(parents=True, exist_ok=True)

        saved = 0
        for batch in tqdm(dataloader, desc=f"Rendering GENMO recon [{split}]"):
            if saved >= max_samples_per_split:
                break

            fps = int(batch.get("fps", torch.tensor([cfg.DATASET.pose_fps]))[0])

            if is_genmo:
                motion = batch["motion_vector"][0].to(device)
                # Dataloader provides 145D: body_r6d(126) + betas(126:136)
                #   + global_r6d(136:142) + local_vel(142:145).
                # Normalize GT to 145D for rendering if it ever arrives as 135D.
                if motion.shape[-1] == 135:
                    motion = insert_zero_betas(motion)
                # Build the model input. The released GenmoFull model is 135D (no
                # betas): drop the betas block -> body(126)+global(6)+vel(3). The
                # 145D variant keeps everything. NOTE: do NOT use motion[:, :135] —
                # that keeps the betas and drops global_orient/local_vel, corrupting
                # the reconstructed root orientation and translation.
                if genmo_full_include_betas:
                    model_input = motion
                else:
                    model_input = torch.cat([motion[:, :126], motion[:, 136:145]], dim=-1)
                tokens = model.vae_body.map2index(model_input.unsqueeze(0))
                rec_motion_raw = model.vae_body.decode(tokens.int())[0]
                # If VAE outputs 135D (no betas), insert zero betas for decoding
                if rec_motion_raw.shape[-1] == 135:
                    rec_motion = insert_zero_betas(rec_motion_raw)
                else:
                    rec_motion = rec_motion_raw
                verts_gt, joints_gt, gt_len = genmo_vertices_and_joints(
                    motion, smplx_model, smplx2smpl, J_regressor, max_seconds, fps
                )
                verts_rec, joints_rec, rec_len = genmo_vertices_and_joints(
                    rec_motion, smplx_model, smplx2smpl, J_regressor, max_seconds, fps
                )
            elif is_genmo_lower:
                motion = batch["motion_vector"][0].to(device)
                # Check config flags for split dimensions
                include_go = getattr(cfg.DATASET, 'GENMO_SPLIT_INCLUDE_GLOBAL_ORIENT', False)

                # Split full 145D genmo into lower and upper
                gt_lower, gt_upper, gt_global_r6d = split_genmo_to_lower_upper(motion, include_global_orient=include_go)

                # Encode/decode lower through vae_body
                tokens_lower = model.vae_body.map2index(gt_lower.unsqueeze(0))
                rec_lower_raw = model.vae_body.decode(tokens_lower.int())[0]

                # Encode/decode upper through vae_upper (78D)
                tokens_upper = model.vae_upper.map2index(gt_upper.unsqueeze(0))
                rec_upper_raw = model.vae_upper.decode(tokens_upper.int())[0]

                # Merge back into full 145D (uses GT global_orient if not in lower)
                rec_motion = merge_lower_upper_to_genmo(rec_lower_raw, rec_upper_raw, gt_global_r6d, include_global_orient=include_go)

                verts_gt, joints_gt, gt_len = genmo_vertices_and_joints(
                    motion, smplx_model, smplx2smpl, J_regressor, max_seconds, fps
                )
                verts_rec, joints_rec, rec_len = genmo_vertices_and_joints(
                    rec_motion, smplx_model, smplx2smpl, J_regressor, max_seconds, fps
                )
            else:
                is_lower_only = selected_part in {"lower_global", "lower", "lower_54"}
                is_upper_lower_unified = selected_part == "upper_lower_global"
                if is_lower_only:
                    lower_key = "lower" if "lower" in batch else "lower_54"
                    tar_lower = batch[lower_key].to(device)
                    tar_beta = batch.get("shape")
                    if tar_beta is None:
                        tar_beta = torch.zeros((tar_lower.shape[0], tar_lower.shape[1], 10), device=device)
                    else:
                        tar_beta = tar_beta.to(device)
                    # translation is always derived from local velocity for recon rendering

                    bs, n = tar_lower.shape[:2]
                    lower_dim = cfg.model.params.modality_tokenizer.vae_lower.params.vae_test_dim
                    tar_lower = tar_lower[..., :lower_dim]
                    rec_lower = model.vae_lower.decode(
                        model.vae_lower.map2index(tar_lower).int()
                    )

                    upper_dim = int(JOINT_MASK_UPPER_6D.sum())
                    hand_dim = 180
                    face_dim = 112
                    identity_6d = torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0], device=device, dtype=tar_lower.dtype)

                    upper_identity = identity_6d.repeat(upper_dim // 6).view(1, 1, -1).expand(bs, n, -1)
                    tar_upper = upper_identity.clone()
                    rec_upper = upper_identity.clone()

                    hand_identity = identity_6d.repeat(hand_dim // 6).view(1, 1, -1).expand(bs, n, -1)
                    tar_hand = hand_identity.clone()
                    rec_hand = hand_identity.clone()

                    tar_face = torch.zeros((bs, n, face_dim), device=device, dtype=tar_lower.dtype)
                    rec_face = torch.zeros((bs, n, face_dim), device=device, dtype=tar_lower.dtype)
                    if face_dim >= 6:
                        tar_face[..., :6] = identity_6d
                        rec_face[..., :6] = identity_6d
                elif is_upper_lower_unified:
                    tar_combined = None
                    if "motion_vector" in batch:
                        tar_combined = batch["motion_vector"].to(device)
                    elif "upper" in batch and "lower" in batch:
                        tar_combined = torch.cat([batch["upper"], batch["lower"]], dim=-1).to(device)
                    elif "lower" in batch:
                        tar_combined = batch["lower"].to(device)
                    else:
                        raise KeyError("upper_lower_global expects motion_vector or upper+lower in batch.")

                    tar_beta = batch.get("shape")
                    if tar_beta is None:
                        tar_beta = torch.zeros((tar_combined.shape[0], tar_combined.shape[1], 10), device=device)
                    else:
                        tar_beta = tar_beta.to(device)

                    # translation is always derived from local velocity for recon rendering

                    bs, n = tar_combined.shape[:2]
                    upper_dim = int(JOINT_MASK_UPPER_6D.sum())
                    combined_dim = cfg.model.params.modality_tokenizer.vae_lower.params.vae_test_dim
                    lower_dim = combined_dim - upper_dim
                    tar_combined = tar_combined[..., :combined_dim]
                    rec_combined = model.vae_lower.decode(
                        model.vae_lower.map2index(tar_combined).int()
                    )

                    tar_upper = tar_combined[..., :upper_dim]
                    tar_lower = tar_combined[..., upper_dim:upper_dim + lower_dim]
                    rec_upper = rec_combined[..., :upper_dim]
                    rec_lower = rec_combined[..., upper_dim:upper_dim + lower_dim]

                    hand_dim = 180
                    face_dim = 112
                    identity_6d = torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0], device=device, dtype=tar_combined.dtype)

                    hand_joints = hand_dim // 6
                    hand_identity = identity_6d.repeat(hand_joints).view(1, 1, -1).expand(bs, n, -1)
                    tar_hand = hand_identity.clone()
                    rec_hand = hand_identity.clone()

                    tar_face = torch.zeros((bs, n, face_dim), device=device, dtype=tar_combined.dtype)
                    rec_face = torch.zeros((bs, n, face_dim), device=device, dtype=tar_combined.dtype)
                    if face_dim >= 6:
                        tar_face[..., :6] = identity_6d
                        rec_face[..., :6] = identity_6d
                else:
                    tar_lower = batch["lower"].to(device)
                    tar_upper = batch["upper"].to(device) if "upper" in batch else None
                    tar_hand = batch["hand"].to(device) if "hand" in batch else None
                    if use_face and "face" in batch:
                        tar_face = batch["face"].to(device)
                    elif use_face and "face_with_head" in batch:
                        tar_face = batch["face_with_head"][..., 6:].to(device)
                    else:
                        tar_face = None
                    tar_beta = batch.get("shape")
                    if tar_beta is None:
                        tar_beta = torch.zeros((tar_lower.shape[0], tar_lower.shape[1], 10), device=device)
                    else:
                        tar_beta = tar_beta.to(device)
                    # translation is always derived from local velocity for recon rendering

                    bs = tar_lower.shape[0]
                    n = tar_lower.shape[1]

                    lower_dim = cfg.model.params.modality_tokenizer.vae_lower.params.vae_test_dim
                    upper_dim = cfg.model.params.modality_tokenizer.vae_upper.params.vae_test_dim
                    hand_dim = cfg.model.params.modality_tokenizer.vae_hand.params.vae_test_dim
                    face_dim = cfg.model.params.modality_tokenizer.vae_face.params.vae_test_dim

                    if tar_upper is None:
                        tar_upper = torch.zeros((bs, n, upper_dim), device=device)
                    identity_6d = torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0], device=device, dtype=tar_lower.dtype)

                    if tar_hand is None:
                        hand_joints = hand_dim // 6
                        tar_hand = identity_6d.repeat(hand_joints).view(1, 1, -1).expand(bs, n, -1).clone()
                    if tar_face is None:
                        tar_face = torch.zeros((bs, n, face_dim), device=device, dtype=tar_lower.dtype)
                        if face_dim >= 6:
                            tar_face[..., :6] = identity_6d
                    tar_upper = tar_upper[:, :n]
                    tar_hand = tar_hand[:, :n]
                    tar_face = tar_face[:, :n]

                    if tar_face.shape[-1] < face_dim:
                        pad_size = face_dim - tar_face.shape[-1]
                        tar_face = torch.cat(
                            [tar_face, torch.zeros((bs, tar_face.shape[1], pad_size), device=device)], dim=-1
                        )

                    rec_lower = model.vae_lower.decode(model.vae_lower.map2index(tar_lower[..., :lower_dim]).int())
                    rec_upper = model.vae_upper.decode(model.vae_upper.map2index(tar_upper[..., :upper_dim]).int())
                    rec_hand = model.vae_hand.decode(model.vae_hand.map2index(tar_hand[..., :hand_dim]).int())
                    if use_face:
                        rec_face = model.vae_face.decode(model.vae_face.map2index(tar_face[..., :face_dim]).int())
                    else:
                        rec_face = torch.zeros((bs, n, face_dim), device=device, dtype=tar_lower.dtype)
                        if face_dim >= 6:
                            rec_face[..., :6] = identity_6d

                n = min(n, rec_lower.shape[1], rec_upper.shape[1], rec_hand.shape[1], rec_face.shape[1])
                rec_lower = rec_lower[:, :n]
                rec_upper = rec_upper[:, :n]
                rec_hand = rec_hand[:, :n]
                rec_face = rec_face[:, :n]
                tar_lower = tar_lower[:, :n]
                tar_upper = tar_upper[:, :n]
                tar_hand = tar_hand[:, :n]
                tar_face = tar_face[:, :n]
                tar_beta = tar_beta[:, :n]
                rec_pose_jaw = rec_face[:, :, :6]
                tar_pose_jaw = tar_face[:, :, :6]

                rec_pose = model.inverse_selection_tensor_full_body_6D(
                    rec_pose_jaw.reshape(bs * n, 6),
                    rec_hand[:, :, :180].reshape(bs * n, 30 * 6),
                    rec_lower[:, :, :54].reshape(bs * n, 9 * 6),
                    rec_upper[:, :, :78].reshape(bs * n, 13 * 6),
                    JOINT_MASK_FACE_6D,
                    JOINT_MASK_HANDS_6D,
                    JOINT_MASK_LOWER_6D,
                    JOINT_MASK_UPPER_6D,
                    n * bs,
                )
                tar_pose = model.inverse_selection_tensor_full_body_6D(
                    tar_pose_jaw.reshape(bs * n, 6),
                    tar_hand[:, :, :180].reshape(bs * n, 30 * 6),
                    tar_lower[:, :, :54].reshape(bs * n, 9 * 6),
                    tar_upper[:, :, :78].reshape(bs * n, 13 * 6),
                    JOINT_MASK_FACE_6D,
                    JOINT_MASK_HANDS_6D,
                    JOINT_MASK_LOWER_6D,
                    JOINT_MASK_UPPER_6D,
                    n * bs,
                )

                rec_pose_aa = rotation_6d_to_axis_angle(rec_pose.reshape(bs * n, 55, 6)).reshape(bs, n, 55 * 3)[0]
                tar_pose_aa = rotation_6d_to_axis_angle(tar_pose.reshape(bs * n, 55, 6)).reshape(bs, n, 55 * 3)[0]

                if lower_dim >= 61:
                    # Lower-body layout (GENMO-style):
                    # - lower[0:54]   : 9 joints * 6D rotation
                    # - lower[54:57]  : local translation velocity (root-local, meters/frame)
                    # - lower[57:61]  : foot contact (4 dims)
                    # - lower[61:..]  : optional extras (betas or dataset-specific)
                    rec_local_vel = rec_lower[0, :, 54:57]
                    tar_local_vel = tar_lower[0, :, 54:57]
                    rec_global_aa = rec_pose_aa[:, :3]
                    tar_global_aa = tar_pose_aa[:, :3]
                    rec_R = axis_angle_to_matrix(rec_global_aa)
                    tar_R = axis_angle_to_matrix(tar_global_aa)
                    rec_world_vel = torch.einsum("tij,tj->ti", rec_R, rec_local_vel)
                    tar_world_vel = torch.einsum("tij,tj->ti", tar_R, tar_local_vel)

                    # Always start from origin and integrate local velocity.
                    init_pos = torch.zeros((3,), device=device)

                    rec_xyz = torch.zeros_like(rec_world_vel)
                    rec_xyz[0] = init_pos
                    if n > 1:
                        rec_xyz[1:] = init_pos + torch.cumsum(rec_world_vel[1:], dim=0)

                    tar_xyz = torch.zeros_like(tar_world_vel)
                    tar_xyz[0] = init_pos
                    if n > 1:
                        tar_xyz[1:] = init_pos + torch.cumsum(tar_world_vel[1:], dim=0)
                else:
                    # No local-velocity channel available; keep translation at origin.
                    tar_xyz = torch.zeros((n, 3), device=device)
                    rec_xyz = torch.zeros((n, 3), device=device)

                betas = tar_beta[0, :n]

                verts_gt, joints_gt, gt_len = smplx_vertices_and_joints_from_aa(
                    tar_pose_aa, tar_xyz, betas, smplx_model, smplx2smpl, J_regressor, max_seconds, fps
                )
                verts_rec, joints_rec, rec_len = smplx_vertices_and_joints_from_aa(
                    rec_pose_aa, rec_xyz, betas, smplx_model, smplx2smpl, J_regressor, max_seconds, fps
                )

            seq_len = min(gt_len, rec_len)
            verts_gt = verts_gt[:seq_len]
            verts_rec = verts_rec[:seq_len]
            joints_gt = joints_gt[:seq_len]
            joints_rec = joints_rec[:seq_len]

            scale, cx, cz = get_ground_params_from_points(joints_gt[:, 0], verts_gt)
            renderer.set_ground(max(scale, 3) * 1.5, cx, cz)
            cam_R, cam_T, lights = get_global_cameras_static(verts_gt.cpu())

            # Compute contact labels for GT and reconstructed
            contact_gt = compute_contact_labels(joints_gt, fps=fps)
            contact_rec = compute_contact_labels(joints_rec, fps=fps)

            frames = []
            for i in range(seq_len):
                with torch.no_grad():
                    cams = renderer.create_camera(cam_R[i], cam_T[i])

                    # Build GT mesh with contact spheres
                    gt_body_mesh = Meshes(
                        verts=[verts_gt[i]], faces=[faces_t],
                        textures=TexturesVertex(verts_features=[color.expand(verts_gt.shape[1], -1)])
                    )
                    gt_meshes = [gt_body_mesh]
                    for j_idx, joint_id in enumerate(CONTACT_JOINT_IDS):
                        if contact_gt[i, j_idx]:
                            gt_meshes.append(Meshes(
                                verts=[sphere_verts_template + joints_gt[i, joint_id]],
                                faces=[sphere_faces],
                                textures=TexturesVertex(
                                    verts_features=[contact_color.expand(sphere_verts_template.shape[0], -1)]
                                )
                            ))

                    # Build reconstruction mesh with contact spheres
                    rec_body_mesh = Meshes(
                        verts=[verts_rec[i]], faces=[faces_t],
                        textures=TexturesVertex(verts_features=[color.expand(verts_rec.shape[1], -1)])
                    )
                    rec_meshes = [rec_body_mesh]
                    for j_idx, joint_id in enumerate(CONTACT_JOINT_IDS):
                        if contact_rec[i, j_idx]:
                            rec_meshes.append(Meshes(
                                verts=[sphere_verts_template + joints_rec[i, joint_id]],
                                faces=[sphere_faces],
                                textures=TexturesVertex(
                                    verts_features=[contact_color.expand(sphere_verts_template.shape[0], -1)]
                                )
                            ))

                    # Render GT scene
                    gv_gt, gf_gt, gc_gt = renderer.ground_geometry
                    gt_meshes.append(Meshes(
                        verts=[gv_gt], faces=[gf_gt],
                        textures=TexturesVertex(verts_features=[gc_gt[..., :3]])
                    ))
                    scene_gt = join_meshes_as_scene(gt_meshes)
                    img_gt = (renderer.renderer(
                        scene_gt, cameras=cams, lights=lights,
                        materials=Materials(device=device, shininess=0)
                    )[0, ..., :3].cpu().numpy() * 255).astype(np.uint8)

                    # Render reconstruction scene
                    gv_rec, gf_rec, gc_rec = renderer.ground_geometry
                    rec_meshes.append(Meshes(
                        verts=[gv_rec], faces=[gf_rec],
                        textures=TexturesVertex(verts_features=[gc_rec[..., :3]])
                    ))
                    scene_rec = join_meshes_as_scene(rec_meshes)
                    img_rec = (renderer.renderer(
                        scene_rec, cameras=cams, lights=lights,
                        materials=Materials(device=device, shininess=0)
                    )[0, ..., :3].cpu().numpy() * 255).astype(np.uint8)

                    frames.append(np.concatenate([img_gt, img_rec], axis=1))

            seq_name = batch["id_name"][0]
            dataset_name = batch.get("dataset_name", ["genmo"])[0]
            out_path = split_output_dir / f"{dataset_name}_{seq_name}.mp4"
            save_video(np.array(frames), str(out_path), fps=int(fps), crf=23)

            saved += 1
            total_saved += 1

    Log.info(f"Saved {total_saved} videos to {output_dir}")


if __name__ == "__main__":
    main()
