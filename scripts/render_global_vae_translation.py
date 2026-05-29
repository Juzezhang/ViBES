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
from multimodal_tokenizers.data.utils import conversation_collate
from multimodal_tokenizers.models.build_model import build_model
from multimodal_tokenizers.utils.load_checkpoint import _extract_state_dict
from multimodal_tokenizers.utils.rotation_conversions import rotation_6d_to_axis_angle
from multimodal_tokenizers.data.mixed_dataset.data_tools import JOINT_MASK_LOWER_6D

# GENMO rendering stack (aligned with preprocess/render_genmo_converted_amass.py)
ROOT_DIR = Path(__file__).resolve().parents[1]
GVHMR_ASSET_DIR = ROOT_DIR / "model_files" / "gvhmr"

from utils.genmo.pylogger import Log
from utils.genmo.video_io_utils import save_video
from utils.genmo.vis.renderer import Renderer, get_global_cameras_static, get_ground_params_from_points
from utils.genmo.geo_transform import apply_T_on_points, compute_T_ayfz2ay
from utils.genmo.camera import create_camera_sensor
from utils.genmo.smplx_utils import make_smplx


def _normalize_vertices(verts_ay, J_regressor):
    v = verts_ay.clone()
    offset = einsum(J_regressor, v[0], "j v, v i -> j i")[0]
    offset[1] = v[:, :, 1].min()
    v -= offset
    T_rot = compute_T_ayfz2ay(einsum(J_regressor, v[[0]], "j v, l v i -> l j i"), inverse=True)
    return apply_T_on_points(v, T_rot)


def smplx_vertices_from_aa(pose_aa, transl, betas, smplx_model, smplx2smpl):
    seq_len = pose_aa.shape[0]
    betas_single = betas[0] if betas.ndim > 1 else betas
    smplx_params = {
        "global_orient": pose_aa[:, :3].cuda(),
        "body_pose": pose_aa[:, 3:3 + 21 * 3].cuda(),
        "transl": transl.cuda(),
        "betas": betas_single.unsqueeze(0).expand(seq_len, -1).cuda(),
    }
    smpl_out = smplx_model(**smplx_params)
    verts_ay = torch.stack([torch.matmul(smplx2smpl, v) for v in smpl_out.vertices])
    return verts_ay


def render_pair(pose_aa, trans_gt, trans_rec, betas, smplx_model, smplx2smpl, J_regressor,
                renderer, max_seconds, fps, out_path):
    total_frames = pose_aa.shape[0]
    max_frames = int(max_seconds * fps)
    seq_len = min(total_frames, max_frames)

    pose_aa = pose_aa[:seq_len]
    trans_gt = trans_gt[:seq_len]
    trans_rec = trans_rec[:seq_len]

    verts_gt = smplx_vertices_from_aa(pose_aa, trans_gt, betas, smplx_model, smplx2smpl)
    verts_rec = smplx_vertices_from_aa(pose_aa, trans_rec, betas, smplx_model, smplx2smpl)

    verts_gt = _normalize_vertices(verts_gt, J_regressor).detach()
    verts_rec = _normalize_vertices(verts_rec, J_regressor).detach()

    joints_gt = einsum(J_regressor, verts_gt, "j v, l v i -> l j i")
    scale, cx, cz = get_ground_params_from_points(joints_gt[:, 0], verts_gt)
    renderer.set_ground(max(scale, 3) * 1.5, cx, cz)

    cam_R, cam_T, lights = get_global_cameras_static(verts_gt.cpu())
    color_gt = torch.tensor([0.35, 0.85, 0.55]).cuda()
    color_rec = torch.tensor([0.90, 0.45, 0.35]).cuda()

    frames = []
    for i in tqdm(range(seq_len), desc="Rendering", leave=False):
        cams = renderer.create_camera(cam_R[i], cam_T[i])
        img_gt = renderer.render_with_ground(verts_gt[[i]], color_gt[None], cams, lights)
        img_rec = renderer.render_with_ground(verts_rec[[i]], color_rec[None], cams, lights)
        if isinstance(img_gt, torch.Tensor):
            img_gt = img_gt.detach().cpu().numpy()
        if isinstance(img_rec, torch.Tensor):
            img_rec = img_rec.detach().cpu().numpy()
        frame = np.concatenate([img_gt, img_rec], axis=1)
        frames.append(frame)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_video(np.array(frames), str(out_path), fps=int(fps), crf=23)


def _load_vae_global(model, ckpt_path):
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state_dict = _extract_state_dict(checkpoint, ckpt_path)
    state_dict_global = {
        k.replace("vae_global.", ""): v
        for k, v in state_dict.items()
        if k.startswith("vae_global.")
    }
    if not state_dict_global:
        state_dict_global = state_dict
    model.vae_global.load_state_dict(state_dict_global, strict=True)


def main():
    cfg = parse_args(phase="test")
    if cfg.Selected_part != "global":
        raise ValueError("render_global_vae_translation expects Selected_part == 'global'")

    # Runtime overrides (env) — keep real/scrubbed paths out of the repo.
    if os.environ.get("VIBES_BEAT2_ROOT"):
        cfg.DATASET.BEAT2.ROOT = os.environ["VIBES_BEAT2_ROOT"]
    if os.environ.get("VIBES_AMASS_ROOT"):
        cfg.DATASET.AMASS.ROOT = os.environ["VIBES_AMASS_ROOT"]
    if os.environ.get("VIBES_EVAL_MODALITIES"):
        mods = [m.strip() for m in os.environ["VIBES_EVAL_MODALITIES"].split(",") if m.strip()]
        cfg.DATASET.MODALITIES = {m: ["lower"] for m in mods}
    if os.environ.get("VIBES_GLOBAL_CKPT"):
        cfg.TEST.CHECKPOINTS = os.environ["VIBES_GLOBAL_CKPT"]

    render_cfg = OmegaConf.select(cfg, "TEST.RENDER")
    splits = render_cfg.SPLITS if render_cfg and "SPLITS" in render_cfg else ["test"]
    num_samples = render_cfg.NUM_SAMPLES_PER_SPLIT if render_cfg and "NUM_SAMPLES_PER_SPLIT" in render_cfg else 5
    max_seconds = render_cfg.MAX_SECONDS if render_cfg and "MAX_SECONDS" in render_cfg else 10.0
    width = render_cfg.WIDTH if render_cfg and "WIDTH" in render_cfg else 1280
    height = render_cfg.HEIGHT if render_cfg and "HEIGHT" in render_cfg else 720

    ckpt_path = OmegaConf.select(cfg, "TEST.CHECKPOINTS")
    if ckpt_path is None:
        raise ValueError("Missing TEST.CHECKPOINTS for global VAE rendering.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(cfg).to(device)
    model.eval()
    _load_vae_global(model, ckpt_path)

    # Avoid heavy dataloader overhead when only sampling a few items.
    cfg.TRAIN.NUM_WORKERS = 0
    cfg.EVAL.NUM_WORKERS = 0
    cfg.TEST.NUM_WORKERS = 0
    datamodule = build_data(cfg)

    smplx_model = make_smplx("supermotion").cuda().eval()
    smplx2smpl = torch.load(GVHMR_ASSET_DIR / "smplx2smpl_sparse.pt").cuda()
    J_regressor = torch.load(GVHMR_ASSET_DIR / "smpl_neutral_J_regressor.pt").cuda()
    _, _, K = create_camera_sensor(width, height, 24)
    renderer = Renderer(width, height, device="cuda", faces=make_smplx("smpl").faces, K=K, bin_size=0)

    out_root = Path("experiments") / "global_vae_translation_check" / cfg.NAME / time.strftime("%Y-%m-%d-%H-%M-%S")

    for split in splits:
        if split == "train":
            cfg.TRAIN.SPLIT = "train"
            datamodule._train_dataset = None
            dataset = datamodule.train_dataset
        elif split == "val":
            cfg.EVAL.SPLIT = "val"
            datamodule._val_dataset = None
            dataset = datamodule.val_dataset
        else:
            cfg.TEST.SPLIT = "test"
            datamodule._test_dataset = None
            dataset = datamodule.test_dataset

        if len(dataset) == 0:
            continue

        rng = np.random.default_rng(1234)
        num_pick = min(num_samples, len(dataset))
        indices = rng.choice(len(dataset), size=num_pick, replace=False)

        saved = 0
        for idx in indices:
            sample = dataset[idx]
            batch = conversation_collate([sample])
            lower = batch["lower"].to(device)
            lengths = batch.get("motion_len", None)
            n = lower.shape[1]
            if lengths is not None:
                n = int(lengths[0])
            lower = lower[:, :n]

            # Build full-body pose from lower rotations (GT pose)
            bs = lower.shape[0]
            lower_rot6d = lower[:, :, :54].reshape(bs * n, 9 * 6)
            full_rot6d = model.inverse_selection_tensor_full_body_6D_from_lower(
                lower_rot6d, JOINT_MASK_LOWER_6D, bs * n
            )
            full_rot6d = full_rot6d.reshape(bs * n, 55, 6)
            # Ensure missing joints use identity rotation (so AA = 0)
            zero_mask = torch.isclose(full_rot6d.abs().sum(dim=-1), torch.zeros(1, device=full_rot6d.device))
            if zero_mask.any():
                identity_6d = torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0], device=full_rot6d.device)
                full_rot6d[zero_mask] = identity_6d
            pose_aa = rotation_6d_to_axis_angle(full_rot6d).reshape(bs, n, 55 * 3)[0]

            # Betas for SMPL-X (use shape if available, fallback to lower tail)
            if "shape" in batch:
                betas = batch["shape"][0, 0, :10].to(device)
            else:
                betas = lower[0, :, -10:]
                betas = betas[0].to(device)

            tar_local_vel = lower[:, :, 54:57]
            tar_global_orient_6d = lower[:, :, :6]
            tar_trans, _ = model._integrate_local_velocity(tar_local_vel, tar_global_orient_6d)

            # Reconstruct local velocity from global VAE
            global_dim = cfg.model.params.modality_tokenizer.vae_global.params.vae_test_dim
            to_global = lower[:, :, :global_dim].clone()
            if to_global.shape[2] > 54:
                to_global[:, :, 54:] = 0.0
            with torch.no_grad():
                rec_global = model.vae_global(to_global)["rec_pose"]
            rec_local_vel = rec_global[:, :, 54:57]
            rec_trans, _ = model._integrate_local_velocity(
                rec_local_vel, tar_global_orient_6d, init_pos=tar_trans[:, 0, :]
            )

            dataset_name = batch.get("dataset_name", ["dataset"])[0]
            sample_id = batch.get("id_name", ["sample"])[0]
            out_path = out_root / split / dataset_name / f"{sample_id}_gtpose_rectrans.mp4"

            Log.info(f"Rendering {dataset_name}/{sample_id} -> {out_path}")
            render_pair(
                pose_aa,
                tar_trans[0],
                rec_trans[0],
                betas,
                smplx_model,
                smplx2smpl,
                J_regressor,
                renderer,
                max_seconds,
                cfg.DATASET.pose_fps,
                out_path,
            )

            saved += 1
            if saved >= num_samples:
                break


if __name__ == "__main__":
    pl.seed_everything(1234)
    main()
