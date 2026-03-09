import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
from einops import einsum
from omegaconf import OmegaConf

from multimodal_tokenizers.config import parse_args
from multimodal_tokenizers.data.build_data import build_data
from multimodal_tokenizers.models.build_model import build_model
from multimodal_tokenizers.utils.load_checkpoint import _extract_state_dict
from multimodal_tokenizers.utils.rotation_conversions import rotation_6d_to_matrix, matrix_to_axis_angle

# GENMO rendering stack (aligned with preprocess/render_genmo_converted_amass.py)
ROOT_DIR = Path(__file__).resolve().parents[1]
GVHMR_ASSET_DIR = ROOT_DIR / "model_files" / "gvhmr"

from utils.genmo.pylogger import Log
from utils.genmo.video_io_utils import save_video
from utils.genmo.vis.renderer import Renderer, get_global_cameras_static, get_ground_params_from_points
from utils.genmo.geo_transform import apply_T_on_points, compute_T_ayfz2ay
from utils.genmo.camera import create_camera_sensor
from utils.genmo.smplx_utils import make_smplx


def decode_genmo_motion(motion_vector, init_trans=None):
    """
    Decode GENMO motion_vector (T,145) to SMPL-X body_pose/global_orient/translation.
    Translation is reconstructed by integrating local velocity rotated by global orient.
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
    trans = torch.zeros_like(world_vel)
    if init_trans is None:
        trans[0] = 0.0
    else:
        trans[0] = init_trans
    if seq_len > 1:
        trans[1:] = trans[0:1] + torch.cumsum(world_vel[1:], dim=0)

    return body_aa, global_aa, trans, betas


def _normalize_vertices(verts_ay, J_regressor):
    v = verts_ay.clone()
    offset = einsum(J_regressor, v[0], "j v, v i -> j i")[0]
    offset[1] = v[:, :, 1].min()
    v -= offset
    T_rot = compute_T_ayfz2ay(einsum(J_regressor, v[[0]], "j v, l v i -> l j i"), inverse=True)
    return apply_T_on_points(v, T_rot)


def _smplx_vertices(body_pose, global_orient, transl, betas, smplx_model, smplx2smpl):
    seq_len = body_pose.shape[0]
    betas_single = betas[0] if betas.ndim > 1 else betas
    smplx_params = {
        "body_pose": body_pose.cuda(),
        "global_orient": global_orient.cuda(),
        "transl": transl.cuda(),
        "betas": betas_single.unsqueeze(0).expand(seq_len, -1).cuda(),
    }
    smpl_out = smplx_model(**smplx_params)
    verts_ay = torch.stack([torch.matmul(smplx2smpl, v) for v in smpl_out.vertices])
    return verts_ay


def render_pair(body_gt, global_gt, trans_gt, body_rec, global_rec, trans_rec,
                betas, smplx_model, smplx2smpl, J_regressor, renderer,
                max_seconds, fps, out_path):
    total_frames = body_gt.shape[0]
    max_frames = int(max_seconds * fps)
    seq_len = min(total_frames, max_frames)

    body_gt = body_gt[:seq_len]
    global_gt = global_gt[:seq_len]
    trans_gt = trans_gt[:seq_len]
    body_rec = body_rec[:seq_len]
    global_rec = global_rec[:seq_len]
    trans_rec = trans_rec[:seq_len]

    verts_gt = _smplx_vertices(body_gt, global_gt, trans_gt, betas, smplx_model, smplx2smpl)
    verts_rec = _smplx_vertices(body_rec, global_rec, trans_rec, betas, smplx_model, smplx2smpl)

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


def _load_vae_body(cfg, model, ckpt_path):
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state_dict = _extract_state_dict(checkpoint, ckpt_path)
    state_dict_body = {
        k.replace("vae_body.", ""): v
        for k, v in state_dict.items()
        if k.startswith("vae_body.")
    }
    if not state_dict_body:
        state_dict_body = state_dict
    model.vae_body.load_state_dict(state_dict_body, strict=True)


def main():
    cfg = parse_args(phase="test")
    if cfg.DATASET.motion_representation != "genmo":
        raise ValueError("render_genmo_translation_check expects DATASET.motion_representation == 'genmo'")

    render_cfg = OmegaConf.select(cfg, "TEST.RENDER")
    splits = render_cfg.SPLITS if render_cfg and "SPLITS" in render_cfg else ["test"]
    num_samples = render_cfg.NUM_SAMPLES_PER_SPLIT if render_cfg and "NUM_SAMPLES_PER_SPLIT" in render_cfg else 5
    max_seconds = render_cfg.MAX_SECONDS if render_cfg and "MAX_SECONDS" in render_cfg else 10.0
    width = render_cfg.WIDTH if render_cfg and "WIDTH" in render_cfg else 1280
    height = render_cfg.HEIGHT if render_cfg and "HEIGHT" in render_cfg else 720

    ckpt_path = OmegaConf.select(cfg, "TEST.CHECKPOINTS_BODY")
    if ckpt_path is None:
        ckpt_path = OmegaConf.select(cfg, "TEST.CHECKPOINTS")
    if ckpt_path is None:
        raise ValueError("Missing TEST.CHECKPOINTS_BODY (or TEST.CHECKPOINTS) for rendering.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(cfg).to(device)
    model.eval()
    _load_vae_body(cfg, model, ckpt_path)

    datamodule = build_data(cfg)
    datamodule.setup("fit")
    datamodule.setup("test")

    smplx_model = make_smplx("supermotion").cuda().eval()
    smplx2smpl = torch.load(GVHMR_ASSET_DIR / "smplx2smpl_sparse.pt").cuda()
    J_regressor = torch.load(GVHMR_ASSET_DIR / "smpl_neutral_J_regressor.pt").cuda()
    _, _, K = create_camera_sensor(width, height, 24)
    renderer = Renderer(width, height, device="cuda", faces=make_smplx("smpl").faces, K=K, bin_size=0)

    out_root = Path("experiments") / "genmo_translation_check" / cfg.NAME / time.strftime("%Y-%m-%d-%H-%M-%S")

    for split in splits:
        if split == "train":
            loader = datamodule.train_dataloader()
        elif split == "val":
            loader = datamodule.val_dataloader()
        else:
            loader = datamodule.test_dataloader()

        saved = 0
        for batch in loader:
            motion = batch["motion_vector"].to(device)
            lengths = batch.get("motion_len", None)
            trans = batch.get("trans")
            if trans is not None:
                trans = trans.to(device)

            with torch.no_grad():
                body_dim = cfg.model.params.modality_tokenizer.vae_body.params.vae_test_dim
                net_out = model.vae_body(motion[..., :body_dim])
                rec_motion = net_out["rec_pose"]

            n = motion.shape[1]
            if lengths is not None:
                n = int(lengths[0])
            motion = motion[:, :n]
            rec_motion = rec_motion[:, :n]
            if trans is not None:
                trans = trans[:, :n]

            # Decode GT and recon into SMPL-X params
            pose_gt, betas_gt, _, trans_gt = model._decode_genmo_to_smplx(motion, trans=trans)
            pose_rec, _, _, trans_rec = model._decode_genmo_to_smplx(rec_motion, trans=trans)

            pose_gt = pose_gt.reshape(1, n, 55 * 3)[0]
            pose_rec = pose_rec.reshape(1, n, 55 * 3)[0]
            body_gt = pose_gt[:, 3:3 + 21 * 3]
            body_rec = pose_rec[:, 3:3 + 21 * 3]
            global_gt = pose_gt[:, :3]
            global_rec = pose_rec[:, :3]
            trans_gt = trans_gt.reshape(1, n, 3)[0]
            trans_rec = trans_rec.reshape(1, n, 3)[0]

            betas_gt = betas_gt.reshape(1, -1)

            dataset_name = batch.get("dataset_name", ["genmo"])[0]
            sample_id = batch.get("id_name", ["sample"])[0]
            out_path = out_root / split / dataset_name / f"{sample_id}_gt_rec.mp4"

            Log.info(f"Rendering {dataset_name}/{sample_id} -> {out_path}")
            render_pair(
                body_gt,
                global_gt,
                trans_gt,
                body_rec,
                global_rec,
                trans_rec,
                betas_gt[0],
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
    main()
