"""Render GT, LOM, GenmoFull, and Hybrid body motion reconstructions side-by-side.

Usage:
    CUDA_VISIBLE_DEVICES=2 python scripts/render_3way_comparison.py --num_samples 5 --output_dir output/3way_comparison
"""

import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from multimodal_tokenizers.archs.lom_vq import VQVAEConvZeroDSUS_PaperVersion as LOMVQ
from multimodal_tokenizers.archs.motiongpt_vq import MotionGPTVQVaeAdapter
from multimodal_tokenizers.utils.rotation_conversions import (
    rotation_6d_to_matrix,
    matrix_to_axis_angle,
    axis_angle_to_matrix,
    matrix_to_rotation_6d,
)

from pytorch3d.structures import Meshes
from pytorch3d.structures.meshes import join_meshes_as_scene
from pytorch3d.renderer import TexturesVertex, Materials

from einops import einsum

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT_DIR = Path(ROOT)
GVHMR_ASSET_DIR = ROOT_DIR / "model_files" / "gvhmr"

from utils.genmo.vis.renderer import (
    Renderer,
    get_global_cameras_static,
    get_ground_params_from_points,
)
from utils.genmo.geo_transform import apply_T_on_points, compute_T_ayfz2ay
from utils.genmo.camera import create_camera_sensor
from utils.genmo.smplx_utils import make_smplx
from utils.genmo.video_io_utils import save_video

# GENMO 21-joint indices (used by GenmoFull and Hybrid sections)
upper_idx = [2, 5, 8, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
lower_idx = [0, 1, 3, 4, 6, 7, 9, 10]

# SMPL-X 22-joint indices for LOM (joints 0-21 in SMPL-X body ordering)
lom_upper_idx = [3, 6, 9, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]
lom_lower_idx = [0, 1, 2, 4, 5, 7, 8, 10, 11]

# Colors for each model (RGB, 0-1 range)
COLOR_GT = [0.65, 0.65, 0.65]       # grey
COLOR_LOM = [0.30, 0.50, 0.85]      # blue
COLOR_GENMOFULL = [0.25, 0.72, 0.45] # green
COLOR_HYBRID = [0.90, 0.55, 0.20]   # orange

LABELS = ["GT", "LOM", "GenmoFull", "Hybrid"]
COLORS = [COLOR_GT, COLOR_LOM, COLOR_GENMOFULL, COLOR_HYBRID]


def parse_args():
    parser = argparse.ArgumentParser(description="Render 3-way body VQ-VAE comparison")
    parser.add_argument("--num_samples", type=int, default=5, help="Number of test samples to render")
    parser.add_argument("--output_dir", type=str, default="output/3way_comparison", help="Output directory")
    parser.add_argument("--max_frames", type=int, default=128, help="Max frames per sample")
    parser.add_argument("--fps", type=int, default=25, help="Video frame rate")
    parser.add_argument("--width", type=int, default=640, help="Width per panel")
    parser.add_argument("--height", type=int, default=480, help="Height per panel")
    parser.add_argument("--from_tokens", action="store_true",
                        help="Decode from saved token .npy files instead of re-encoding genmo npz")
    parser.add_argument("--dataset", type=str, default="AMASS",
                        choices=["AMASS", "AMASS_talking", "BEAT2"],
                        help="Dataset to visualize (default: AMASS)")
    return parser.parse_args()


def load_models(device):
    """Load all three body VQ-VAE models."""
    # LOM
    print("Loading LOM...")
    vae_upper = LOMVQ(vae_layer=3, code_num=256, vae_test_dim=78, codebook_size=256, vae_quantizer_lambda=1)
    vae_lower = LOMVQ(vae_layer=3, code_num=256, vae_test_dim=54, codebook_size=256, vae_quantizer_lambda=1)
    sd = torch.load(os.path.join(ROOT, 'model_files/pretrained_cpt/body/lom_vq.ckpt'), map_location='cpu', weights_only=False)['state_dict']
    for n, m in [('vae_upper', vae_upper), ('vae_lower', vae_lower)]:
        m.load_state_dict({k.replace(n + '.', ''): v for k, v in sd.items() if k.startswith(n + '.')}, strict=True)
        m.eval().to(device)

    # GenmoFull
    print("Loading GenmoFull...")
    vae_full = MotionGPTVQVaeAdapter(nfeats=135, code_dim=256, code_num=256, mu=0.99, down_t=2, stride_t=2, depth=3, dilation_growth_rate=3, width=512, output_emb_width=256)
    sd2 = torch.load(os.path.join(ROOT, 'model_files/pretrained_cpt/VQVAE_0320_GenmoFull/last.ckpt'), map_location='cpu', weights_only=False)['state_dict']
    vae_full.load_state_dict({k.replace('vae_body.', ''): v for k, v in sd2.items() if 'vae_body' in k}, strict=True)
    vae_full.eval().to(device)

    # Hybrid
    print("Loading Hybrid (NormalUpper+GenmoLower)...")
    vae_hybrid_upper = MotionGPTVQVaeAdapter(nfeats=78, code_dim=128, code_num=128, mu=0.95, down_t=2, stride_t=2, depth=3, dilation_growth_rate=3, width=256, output_emb_width=128)
    vae_hybrid_lower = MotionGPTVQVaeAdapter(nfeats=61, code_dim=128, code_num=256, mu=0.95, down_t=2, stride_t=2, depth=3, dilation_growth_rate=3, width=512, output_emb_width=128, norm='GN')
    sd3 = torch.load(os.path.join(ROOT, 'model_files/pretrained_cpt/VQVAE_0318_NormalUpper_GenmoLower/vqvae_normal_upper_last.ckpt'), map_location='cpu', weights_only=False)['state_dict']
    vae_hybrid_upper.load_state_dict({k.replace('vae_upper.', ''): v for k, v in sd3.items() if 'vae_upper' in k}, strict=True)
    vae_hybrid_upper.eval().to(device)
    sd4 = torch.load(os.path.join(ROOT, 'model_files/pretrained_cpt/VQVAE_0318_NormalUpper_GenmoLower/vqvar_genmo_lower_global_last.ckpt'), map_location='cpu', weights_only=False)['state_dict']
    vae_hybrid_lower.load_state_dict({k.replace('vae_lower.', ''): v for k, v in sd4.items() if 'vae_lower' in k}, strict=True)
    vae_hybrid_lower.eval().to(device)

    print("All models loaded.")
    models = {
        'lom_upper': vae_upper,
        'lom_lower': vae_lower,
        'full': vae_full,
        'hybrid_upper': vae_hybrid_upper,
        'hybrid_lower': vae_hybrid_lower,
    }
    return models


def reconstruct_genmofull(models, mv, T, device):
    """GenmoFull: encode/decode 135D (no betas), insert zero betas back to 145D."""
    m135 = torch.cat([mv[:, :126], mv[:, 136:145]], dim=-1)
    with torch.no_grad():
        rec135 = models['full'].decode(models['full'].map2index(m135.unsqueeze(0)).int())[0]
    rec145 = torch.cat([rec135[:, :126], torch.zeros(rec135.shape[0], 10, device=device), rec135[:, 126:]], dim=-1)
    return rec145


def reconstruct_lom(models, mv, T, device, npz_data=None):
    """LOM: uses SMPL-X rotation data (22 joints), NOT GENMO format.

    Loads body_pose + global_orient from npz_data, converts to 22-joint 6D,
    splits into upper(78D)/lower(54D) using SMPL-X joint indices,
    encodes/decodes through LOM VAEs, then reassembles into GENMO 145D
    for downstream rendering via genmo_to_verts.
    """
    betas = mv[:, 126:136]
    local_vel = mv[:, 142:145]

    # Load SMPL-X rotations from .npz
    bp_aa = torch.tensor(npz_data['body_pose'][:T], dtype=torch.float32, device=device).reshape(T, 21, 3)
    go_aa = torch.tensor(npz_data['global_orient'][:T], dtype=torch.float32, device=device).reshape(T, 1, 3)
    # Full 22-joint axis-angle: [global_orient, body_pose] -> joints 0-21
    full22_aa = torch.cat([go_aa, bp_aa], dim=1)  # (T, 22, 3)
    # Convert to 6D rotation
    full22_mat = axis_angle_to_matrix(full22_aa.reshape(-1, 3)).reshape(T, 22, 3, 3)
    full22_r6 = matrix_to_rotation_6d(full22_mat.reshape(-1, 3, 3)).reshape(T, 22, 6)

    # Split upper (13 joints) and lower (9 joints) using SMPL-X indices
    lom_ud = full22_r6[:, lom_upper_idx].reshape(T, -1)  # (T, 78)
    lom_ld = full22_r6[:, lom_lower_idx].reshape(T, -1)  # (T, 54)
    with torch.no_grad():
        ru = models['lom_upper'].decode(models['lom_upper'].map2index(lom_ud.unsqueeze(0)).int())[0]
        rl = models['lom_lower'].decode(models['lom_lower'].map2index(lom_ld.unsqueeze(0)).int())[0]
    Tl = min(T, ru.shape[0], rl.shape[0])

    # Reassemble 22-joint 6D from reconstructed upper/lower
    rec22_r6 = torch.zeros(Tl, 22, 6, device=device)
    for i, idx in enumerate(lom_upper_idx):
        rec22_r6[:, idx] = ru[:Tl, i * 6:(i + 1) * 6]
    for i, idx in enumerate(lom_lower_idx):
        rec22_r6[:, idx] = rl[:Tl, i * 6:(i + 1) * 6]

    # Convert reconstructed 22-joint 6D back to GENMO 145D for rendering
    # GENMO uses 21 body joints (SMPL-X joints 1-21, i.e., body_pose without pelvis)
    # Global orient = joint 0 (pelvis)
    rec_go_r6 = rec22_r6[:, 0]          # (Tl, 6) global_orient in 6D
    rec_bp_r6 = rec22_r6[:, 1:]         # (Tl, 21, 6) body_pose joints 1-21
    rec145 = torch.cat([
        rec_bp_r6.reshape(Tl, 126),     # 21 joints x 6D = 126
        betas[:Tl],                      # 10D betas
        rec_go_r6,                       # 6D global orient
        local_vel[:Tl],                  # 3D local velocity
    ], dim=-1)
    return rec145


def reconstruct_hybrid(models, mv, T, device, npz_data=None, smplx_model=None):
    """Hybrid: upper(78D) + lower_global(61D = 54 rotation + 3 vel + 4 contact)."""
    betas = mv[:, 126:136]
    local_vel = mv[:, 142:145]

    # Build SMPL-X 22-joint 6D
    bp_aa = torch.tensor(npz_data['body_pose'][:T], dtype=torch.float32, device=device).reshape(T, 21, 3)
    go_aa = torch.tensor(npz_data['global_orient'][:T], dtype=torch.float32, device=device).reshape(T, 1, 3)
    full22_aa = torch.cat([go_aa, bp_aa], dim=1)
    full22_mat = axis_angle_to_matrix(full22_aa.reshape(-1, 3)).reshape(T, 22, 3, 3)
    full22_r6 = matrix_to_rotation_6d(full22_mat.reshape(-1, 3, 3)).reshape(T, 22, 6)

    hud = full22_r6[:, lom_upper_idx].reshape(T, -1)  # (T, 78)
    lower_r6 = full22_r6[:, lom_lower_idx].reshape(T, -1)  # (T, 54)

    # Compute foot contact from FK
    import smplx as smplx_pkg
    with torch.no_grad():
        bm_fc = smplx_pkg.create(
            os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'model_files/smplx_models'),
            model_type='smplx', gender='neutral', batch_size=T, num_betas=10, use_pca=False, flat_hand_mean=True
        ).to(device)
        trans_t = torch.tensor(npz_data['trans'][:T], dtype=torch.float32, device=device)
        fk_out = bm_fc(global_orient=go_aa.reshape(T,3), body_pose=bp_aa.reshape(T,-1), betas=betas[:T], transl=trans_t)
        foot_joints = fk_out.joints[:, [7, 8, 10, 11], :]
    foot_vel = torch.zeros(T, 4, device=device)
    foot_vel[:-1] = (foot_joints[1:] - foot_joints[:-1]).norm(dim=-1)
    foot_contact = (foot_vel < 0.01).float()

    hld = torch.cat([lower_r6, local_vel, foot_contact], dim=-1)  # (T, 61)

    with torch.no_grad():
        hru = models['hybrid_upper'].decode(models['hybrid_upper'].map2index(hud.unsqueeze(0)).int())[0]
        hrl = models['hybrid_lower'].decode(models['hybrid_lower'].map2index(hld.unsqueeze(0)).int())[0]
    Th = min(T, hru.shape[0], hrl.shape[0])

    rec22_r6 = torch.zeros(Th, 22, 6, device=device)
    for i, idx in enumerate(lom_upper_idx):
        rec22_r6[:, idx] = hru[:Th, i * 6:(i + 1) * 6]
    rec_lower_9 = hrl[:Th, :54].reshape(Th, 9, 6)
    for i, idx in enumerate(lom_lower_idx):
        rec22_r6[:, idx] = rec_lower_9[:, i]
    rec_vel_h = hrl[:Th, 54:57]

    rec_go_r6 = rec22_r6[:, 0]
    rec_bp_r6 = rec22_r6[:, 1:]
    rec145 = torch.cat([rec_bp_r6.reshape(Th, 126), betas[:Th], rec_go_r6, rec_vel_h], dim=-1)
    return rec145


def decode_genmo_to_smplx(motion_vector):
    """Decode GENMO 145D motion_vector to SMPL-X params (body_aa, global_aa, transl, betas)."""
    T = motion_vector.shape[0]
    br6 = motion_vector[:, :126].reshape(T, 21, 6)
    betas = motion_vector[:, 126:136]
    gr6 = motion_vector[:, 136:142]
    local_vel = motion_vector[:, 142:145]

    bmat = rotation_6d_to_matrix(br6.reshape(-1, 6)).reshape(T, 21, 3, 3)
    baa = matrix_to_axis_angle(bmat.reshape(-1, 3, 3)).reshape(T, 63)
    gR = rotation_6d_to_matrix(gr6)
    gaa = matrix_to_axis_angle(gR)

    world_vel = torch.einsum("tij,tj->ti", gR, local_vel)
    transl = torch.cumsum(world_vel, dim=0)

    return baa, gaa, transl, betas


def _normalize_vertices(verts_ay, J_regressor):
    v = verts_ay.clone()
    offset = einsum(J_regressor, v[0], "j v, v i -> j i")[0]
    offset[1] = v[:, :, 1].min()
    v -= offset
    T_rot = compute_T_ayfz2ay(einsum(J_regressor, v[[0]], "j v, l v i -> l j i"), inverse=True)
    return apply_T_on_points(v, T_rot)


def genmo_to_verts(motion_vector, smplx_model, smplx2smpl, J_regressor, max_frames):
    """Convert GENMO 145D to world-space SMPL vertices for rendering."""
    body_aa, global_aa, transl, betas = decode_genmo_to_smplx(motion_vector)
    T = min(body_aa.shape[0], max_frames)
    body_aa = body_aa[:T]
    global_aa = global_aa[:T]
    transl = transl[:T]
    betas = betas[:T]

    betas_single = betas[0] if betas.ndim > 1 else betas
    smplx_params = {
        "body_pose": body_aa.cuda(),
        "global_orient": global_aa.cuda(),
        "transl": transl.cuda(),
        "betas": betas_single.unsqueeze(0).expand(T, -1).cuda(),
    }
    smpl_out = smplx_model(**smplx_params)
    verts_ay = torch.stack([torch.matmul(smplx2smpl, v) for v in smpl_out.vertices])
    verts_glob = _normalize_vertices(verts_ay, J_regressor)
    joints_glob = einsum(J_regressor, verts_glob, "j v, l v i -> l j i")
    return verts_glob, joints_glob, T


def add_label_to_frame(frame, label, position="top_center"):
    """Add text label on top of a rendered frame (numpy HWC uint8)."""
    h, w = frame.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.9
    thickness = 2
    (tw, th), baseline = cv2.getTextSize(label, font, font_scale, thickness)
    x = (w - tw) // 2
    y = th + 15
    # Draw background rectangle for readability
    cv2.rectangle(frame, (x - 5, y - th - 5), (x + tw + 5, y + baseline + 5), (255, 255, 255), -1)
    cv2.putText(frame, label, (x, y), font, font_scale, (0, 0, 0), thickness, cv2.LINE_AA)
    return frame


def reconstruct_from_fullbody_token(models, token_path, device):
    """Decode a saved fullbody_genmo token .npy to GENMO 145D."""
    tokens = torch.tensor(np.load(token_path), device=device)
    with torch.no_grad():
        rec135 = models['full'].decode(tokens.int())[0]
    return torch.cat([rec135[:, :126], torch.zeros(rec135.shape[0], 10, device=device), rec135[:, 126:]], dim=-1)


def reconstruct_from_lower_token(models, token_path, gt_mv, device):
    """Decode a saved lower_genmo token .npy, merge with GT upper + global_orient."""
    tokens = torch.tensor(np.load(token_path), device=device)
    with torch.no_grad():
        rec61 = models['hybrid_lower'].decode(tokens.int())[0]
    Tl = min(rec61.shape[0], gt_mv.shape[0])
    rec61 = rec61[:Tl]
    # Lower 61D: [0:54] 9 joints × 6D, [54:57] local_vel, [57:61] contact
    rec_lower_9 = rec61[:, :54].reshape(Tl, 9, 6)
    rec_vel = rec61[:, 54:57]
    # Use GT upper joints and global_orient
    gt_br6 = gt_mv[:Tl, :126].reshape(Tl, 21, 6)
    merged_br6 = gt_br6.clone()
    for i, idx in enumerate(lom_lower_idx):
        merged_br6[:, idx] = rec_lower_9[:, i]
    # GENMO 21-joint: joint 0 = pelvis (in body_r6d), global_orient is separate
    return torch.cat([merged_br6.reshape(Tl, 126), gt_mv[:Tl, 126:136], gt_mv[:Tl, 136:142], rec_vel], dim=-1)


def get_dataset_paths(dataset_name):
    """Return genmo_dir, token_dirs, test_seqs for a dataset."""
    if dataset_name == "AMASS":
        root = '/path/to/AMASS'
        genmo_dir = os.path.join(root, 'amass_genmo_25')
        token_root = os.path.join(root, 'TOKENS_AGENT_25')  # May not exist for AMASS
        with open(os.path.join(root, 'test.txt')) as f:
            seqs = [l.strip() for l in f if l.strip()]
    elif dataset_name == "AMASS_talking":
        root = '/path/to/AMASS_talking'
        genmo_dir = os.path.join(root, 'amass_genmo_25')
        token_root = os.path.join(root, 'TOKENS_AGENT_25')
        seqs = sorted([f.replace('.npz', '') for f in os.listdir(genmo_dir) if f.endswith('.npz')])
    elif dataset_name == "BEAT2":
        root = '/path/to/BEAT2/beat_english_v2.0.0'
        genmo_dir = os.path.join(root, 'beat2_genmo_25/smplxflame_25')
        token_root = os.path.join(root, 'TOKENS_AGENT_25')
        import csv
        seqs = []
        with open(os.path.join(root, 'train_test_split.csv')) as f:
            for row in csv.DictReader(f):
                if row['type'] == 'test':
                    seqs.append(row['id'])
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    return genmo_dir, token_root, seqs


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load all VQ-VAE models
    models = load_models(device)

    # Load SMPL-X rendering resources
    print("Initializing SMPL-X and renderer...")
    smplx_model = make_smplx("supermotion").cuda()
    smplx2smpl = torch.load(GVHMR_ASSET_DIR / "smplx2smpl_sparse.pt").cuda()
    faces_smpl = make_smplx("smpl").faces
    J_regressor = torch.load(GVHMR_ASSET_DIR / "smpl_neutral_J_regressor.pt").cuda()
    faces_t = torch.from_numpy(faces_smpl.astype(np.int64)).to(device)

    # Create one renderer per panel
    _, _, K = create_camera_sensor(args.width, args.height, 24)
    renderer = Renderer(args.width, args.height, device="cuda", faces=faces_smpl, K=K, bin_size=0)

    # Load test sequences based on dataset
    genmo_dir, token_root, seqs = get_dataset_paths(args.dataset)

    print(f"Dataset: {args.dataset}, {len(seqs)} test sequences. Rendering {args.num_samples} samples...")
    if args.from_tokens:
        print(f"  Mode: decode from saved tokens at {token_root}")

    saved = 0
    for seq_idx, sn in enumerate(seqs):
        if saved >= args.num_samples:
            break

        p = os.path.join(genmo_dir, sn + '.npz')
        if not os.path.exists(p):
            continue
        d = np.load(p, allow_pickle=True)
        if 'motion_vector' not in d:
            continue
        mv = torch.tensor(d['motion_vector'][:args.max_frames], dtype=torch.float32, device=device)
        T = mv.shape[0]
        if T < 32:
            continue

        print(f"  [{saved + 1}/{args.num_samples}] Processing sequence {sn} ({T} frames)...")

        # Reconstruct with each model
        with torch.no_grad():
            if args.from_tokens:
                fb_path = os.path.join(token_root, 'fullbody_genmo', sn + '.npy')
                lg_path = os.path.join(token_root, 'lower_genmo', sn + '.npy')
                rec_full = reconstruct_from_fullbody_token(models, fb_path, device) if os.path.exists(fb_path) else reconstruct_genmofull(models, mv, T, device)
                rec_lom = reconstruct_lom(models, mv, T, device, npz_data=d)
                rec_hyb = reconstruct_from_lower_token(models, lg_path, mv, device) if os.path.exists(lg_path) else reconstruct_hybrid(models, mv, T, device, npz_data=d)
            else:
                rec_full = reconstruct_genmofull(models, mv, T, device)
                rec_lom = reconstruct_lom(models, mv, T, device, npz_data=d)
                rec_hyb = reconstruct_hybrid(models, mv, T, device, npz_data=d)

        # Find common frame count across all reconstructions
        T_common = min(T, rec_full.shape[0], rec_lom.shape[0], rec_hyb.shape[0], args.max_frames)

        # Convert each to SMPL vertices
        all_verts = []
        all_joints = []
        motion_list = [mv[:T_common], rec_lom[:T_common], rec_full[:T_common], rec_hyb[:T_common]]
        for motion in motion_list:
            verts, joints, seq_len = genmo_to_verts(motion, smplx_model, smplx2smpl, J_regressor, T_common)
            all_verts.append(verts[:seq_len])
            all_joints.append(joints[:seq_len])

        # Use shortest across all
        T_render = min(v.shape[0] for v in all_verts)
        all_verts = [v[:T_render] for v in all_verts]
        all_joints = [j[:T_render] for j in all_joints]

        # Compute camera and ground from GT
        verts_gt = all_verts[0]
        joints_gt = all_joints[0]

        scale, cx, cz = get_ground_params_from_points(joints_gt[:, 0], verts_gt)
        renderer.set_ground(max(scale, 3) * 1.5, cx, cz)
        cam_R, cam_T, lights = get_global_cameras_static(verts_gt.cpu())

        # Render each frame: 4 panels side-by-side
        frames = []
        for fi in range(T_render):
            panel_images = []
            for mi in range(4):
                with torch.no_grad():
                    cams = renderer.create_camera(cam_R[fi], cam_T[fi])
                    color_t = torch.tensor(COLORS[mi], device=device, dtype=torch.float32)
                    body_mesh = Meshes(
                        verts=[all_verts[mi][fi]],
                        faces=[faces_t],
                        textures=TexturesVertex(verts_features=[color_t.expand(all_verts[mi].shape[1], -1)])
                    )
                    gv, gf, gc = renderer.ground_geometry
                    ground_mesh = Meshes(
                        verts=[gv], faces=[gf],
                        textures=TexturesVertex(verts_features=[gc[..., :3]])
                    )
                    scene = join_meshes_as_scene([body_mesh, ground_mesh])
                    img = (renderer.renderer(
                        scene, cameras=cams, lights=lights,
                        materials=Materials(device=device, shininess=0)
                    )[0, ..., :3].cpu().numpy() * 255).astype(np.uint8)

                    img = add_label_to_frame(img, LABELS[mi])
                    panel_images.append(img)

            # Concatenate 4 panels horizontally
            combined = np.concatenate(panel_images, axis=1)
            frames.append(combined)

        # Write video
        out_path = os.path.join(args.output_dir, f"comparison_{sn}.mp4")
        save_video(np.array(frames), out_path, fps=args.fps, crf=23)
        print(f"    Saved: {out_path}")
        saved += 1

    print(f"\nDone. Saved {saved} comparison videos to {args.output_dir}")


if __name__ == "__main__":
    main()
