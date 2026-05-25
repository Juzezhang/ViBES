"""Compare LOM VQ-VAE vs GenmoFull VQ-VAE on AMASS test set. Metrics: MPJPE, PA-MPJPE."""
import sys, os, torch, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from multimodal_tokenizers.archs.lom_vq import VQVAEConvZeroDSUS1_PaperVersion
from multimodal_tokenizers.archs.motiongpt_vq import MotionGPTVQVaeAdapter
from multimodal_tokenizers.utils.rotation_conversions import rotation_6d_to_matrix, matrix_to_axis_angle, axis_angle_to_matrix, matrix_to_rotation_6d
import smplx as smplx_pkg

device = 'cuda'
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Load LOM
print("Loading LOM...")
from multimodal_tokenizers.archs.lom_vq import VQVAEConvZeroDSUS_PaperVersion as LOMVQ
vae_upper = LOMVQ(vae_layer=3, code_num=256, vae_test_dim=78, codebook_size=256, vae_quantizer_lambda=1)
vae_lower = LOMVQ(vae_layer=3, code_num=256, vae_test_dim=54, codebook_size=256, vae_quantizer_lambda=1)
sd = torch.load(os.path.join(ROOT, 'model_files/pretrained_cpt/body/lom_vq.ckpt'), map_location='cpu', weights_only=False)['state_dict']
for n, m in [('vae_upper', vae_upper), ('vae_lower', vae_lower)]:
    m.load_state_dict({k.replace(n+'.', ''): v for k, v in sd.items() if k.startswith(n+'.')}, strict=True)
    m.eval().to(device)

# Load GenmoFull
print("Loading GenmoFull...")
vae_full = MotionGPTVQVaeAdapter(nfeats=135, code_dim=256, code_num=256, mu=0.99, down_t=2, stride_t=2, depth=3, dilation_growth_rate=3, width=512, output_emb_width=256)
sd2 = torch.load(os.path.join(ROOT, 'model_files/pretrained_cpt/VQVAE_0320_GenmoFull/last.ckpt'), map_location='cpu', weights_only=False)['state_dict']
vae_full.load_state_dict({k.replace('vae_body.', ''): v for k, v in sd2.items() if 'vae_body' in k}, strict=True)
vae_full.eval().to(device)

# Load Hybrid
print("Loading Hybrid (NormalUpper+GenmoLower)...")
vae_hybrid_upper = MotionGPTVQVaeAdapter(nfeats=78, code_dim=128, code_num=128, mu=0.95, down_t=2, stride_t=2, depth=3, dilation_growth_rate=3, width=256, output_emb_width=128)
vae_hybrid_lower = MotionGPTVQVaeAdapter(nfeats=61, code_dim=128, code_num=256, mu=0.95, down_t=2, stride_t=2, depth=3, dilation_growth_rate=3, width=512, output_emb_width=128, norm='GN')
sd3 = torch.load(os.path.join(ROOT, 'model_files/pretrained_cpt/VQVAE_0318_NormalUpper_GenmoLower/vqvae_normal_upper_last.ckpt'), map_location='cpu', weights_only=False)['state_dict']
vae_hybrid_upper.load_state_dict({k.replace('vae_upper.', ''): v for k, v in sd3.items() if 'vae_upper' in k}, strict=True)
vae_hybrid_upper.eval().to(device)
sd4 = torch.load(os.path.join(ROOT, 'model_files/pretrained_cpt/VQVAE_0318_NormalUpper_GenmoLower/vqvar_genmo_lower_global_last.ckpt'), map_location='cpu', weights_only=False)['state_dict']
vae_hybrid_lower.load_state_dict({k.replace('vae_lower.', ''): v for k, v in sd4.items() if 'vae_lower' in k}, strict=True)
vae_hybrid_lower.eval().to(device)
print("All models loaded")

def pa(pred, tgt):
    mu_p = pred.mean(0, keepdims=True); mu_t = tgt.mean(0, keepdims=True)
    p = pred - mu_p; t = tgt - mu_t; H = p.T @ t
    U, S, Vt = np.linalg.svd(H); R = Vt.T @ U.T
    if np.linalg.det(R) < 0: Vt[-1] *= -1; R = Vt.T @ U.T
    return (p @ R) + mu_t

# GENMO 21-joint indices (used by GenmoFull and Hybrid sections)
upper_idx = [2, 5, 8, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
lower_idx = [0, 1, 3, 4, 6, 7, 9, 10]

# SMPL-X 22-joint indices for LOM (joints 0-21 in SMPL-X body ordering)
# Upper: spine1, spine2, spine3, neck, left_collar, right_collar, head,
#         left_shoulder, right_shoulder, left_elbow, right_elbow, left_wrist, right_wrist
lom_upper_idx = [3, 6, 9, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]
# Lower: pelvis, left_hip, right_hip, left_knee, right_knee, left_ankle, right_ankle, left_foot, right_foot
lom_lower_idx = [0, 1, 2, 4, 5, 7, 8, 10, 11]
amass_root = '/path/to/AMASS'
genmo_dir = os.path.join(amass_root, 'amass_genmo_25')
with open(os.path.join(amass_root, 'test.txt')) as f:
    seqs = [l.strip() for l in f if l.strip()]

lom_m, lom_pa = [], []
full_m, full_pa = [], []
hyb_m, hyb_pa = [], []
cnt = 0

for sn in seqs:
    p = os.path.join(genmo_dir, sn + '.npz')
    if not os.path.exists(p): continue
    d = np.load(p, allow_pickle=True)
    if 'motion_vector' not in d: continue
    mv = torch.tensor(d['motion_vector'][:128], dtype=torch.float32, device=device)
    T = mv.shape[0]
    if T < 32: continue

    br6 = mv[:, :126].reshape(T, 21, 6)
    betas = mv[:, 126:136]; gr6 = mv[:, 136:142]

    # GT joints via SMPL-X FK
    bmat = rotation_6d_to_matrix(br6.reshape(-1, 6)).reshape(T, 21, 3, 3)
    baa = matrix_to_axis_angle(bmat.reshape(-1, 3, 3)).reshape(T, 21, 3)
    gaa = matrix_to_axis_angle(rotation_6d_to_matrix(gr6))
    bm = smplx_pkg.create(os.path.join(ROOT, 'model_files/smplx_models'), model_type='smplx', gender='neutral', batch_size=T, num_betas=10, use_pca=False, flat_hand_mean=True).to(device)
    with torch.no_grad():
        gt_j = bm(global_orient=gaa, body_pose=baa.reshape(T, -1), betas=betas).joints[:, :22].detach()
    gt_jr = gt_j - gt_j[:, 0:1]

    def eval_rec(rec145, label):
        Tr = rec145.shape[0]
        rb = rec145[:, :126].reshape(Tr, 21, 6)
        rmat = rotation_6d_to_matrix(rb.reshape(-1, 6)).reshape(Tr, 21, 3, 3)
        raa = matrix_to_axis_angle(rmat.reshape(-1, 3, 3)).reshape(Tr, 21, 3)
        rga = matrix_to_axis_angle(rotation_6d_to_matrix(rec145[:, 136:142]))
        if Tr != bm.batch_size:
            bm2 = smplx_pkg.create(os.path.join(ROOT, 'model_files/smplx_models'), model_type='smplx', gender='neutral', batch_size=Tr, num_betas=10, use_pca=False, flat_hand_mean=True).to(device)
        else:
            bm2 = bm
        with torch.no_grad():
            rj = bm2(global_orient=rga, body_pose=raa.reshape(Tr, -1), betas=rec145[:, 126:136]).joints[:, :22].detach()
        rjr = rj - rj[:, 0:1]
        mlen = min(T, Tr)
        mpjpe = (gt_jr[:mlen] - rjr[:mlen]).norm(dim=-1).mean().item() * 1000
        pampjpe = np.mean([np.linalg.norm(pa(rj[t].cpu().numpy(), gt_j[t].cpu().numpy()) - gt_j[t].cpu().numpy(), axis=-1).mean() * 1000 for t in range(mlen)])
        return mpjpe, pampjpe

    # GenmoFull
    m135 = torch.cat([mv[:, :126], mv[:, 136:145]], dim=-1)
    with torch.no_grad():
        rec135 = vae_full.decode(vae_full.map2index(m135.unsqueeze(0)).int())[0]
    rec145f = torch.cat([rec135[:, :126], torch.zeros(rec135.shape[0], 10, device=device), rec135[:, 126:]], dim=-1)
    mf, pf = eval_rec(rec145f, "Full")
    full_m.append(mf); full_pa.append(pf)

    # LOM — uses SMPL-X rotation data (22 joints), NOT GENMO format
    # Load body_pose (21 joints × 3 axis-angle) + global_orient (3) from .npz
    bp_aa = torch.tensor(d['body_pose'][:T], dtype=torch.float32, device=device).reshape(T, 21, 3)
    go_aa = torch.tensor(d['global_orient'][:T], dtype=torch.float32, device=device).reshape(T, 1, 3)
    # Full 22-joint axis-angle: [global_orient, body_pose] → joints 0-21
    full22_aa = torch.cat([go_aa, bp_aa], dim=1)  # (T, 22, 3)
    # Convert to 6D rotation: axis-angle → matrix → 6D
    full22_mat = axis_angle_to_matrix(full22_aa.reshape(-1, 3)).reshape(T, 22, 3, 3)
    full22_r6 = matrix_to_rotation_6d(full22_mat.reshape(-1, 3, 3)).reshape(T, 22, 6)
    # Split upper (13 joints) and lower (9 joints) using SMPL-X indices
    lom_ud = full22_r6[:, lom_upper_idx].reshape(T, -1)  # (T, 78)
    lom_ld = full22_r6[:, lom_lower_idx].reshape(T, -1)  # (T, 54)
    with torch.no_grad():
        ru = vae_upper.decode(vae_upper.map2index(lom_ud.unsqueeze(0)).int())[0]
        rl = vae_lower.decode(vae_lower.map2index(lom_ld.unsqueeze(0)).int())[0]
    Tl = min(T, ru.shape[0], rl.shape[0])
    # Reassemble 22-joint 6D from reconstructed upper/lower
    rec22_r6 = torch.zeros(Tl, 22, 6, device=device)
    for i, idx in enumerate(lom_upper_idx): rec22_r6[:, idx] = ru[:Tl, i*6:(i+1)*6]
    for i, idx in enumerate(lom_lower_idx): rec22_r6[:, idx] = rl[:Tl, i*6:(i+1)*6]
    # Convert back to axis-angle for SMPL-X FK
    rec22_mat = rotation_6d_to_matrix(rec22_r6.reshape(-1, 6)).reshape(Tl, 22, 3, 3)
    rec22_aa = matrix_to_axis_angle(rec22_mat.reshape(-1, 3, 3)).reshape(Tl, 22, 3)
    rec_go_aa = rec22_aa[:, 0]       # (Tl, 3) global_orient
    rec_bp_aa = rec22_aa[:, 1:]      # (Tl, 21, 3) body_pose
    # Run SMPL-X FK to get joint positions
    if Tl != bm.batch_size:
        bm_lom = smplx_pkg.create(os.path.join(ROOT, 'model_files/smplx_models'), model_type='smplx', gender='neutral', batch_size=Tl, num_betas=10, use_pca=False, flat_hand_mean=True).to(device)
    else:
        bm_lom = bm
    with torch.no_grad():
        lom_j = bm_lom(global_orient=rec_go_aa, body_pose=rec_bp_aa.reshape(Tl, -1), betas=betas[:Tl]).joints[:, :22].detach()
    lom_jr = lom_j - lom_j[:, 0:1]
    mlen = min(T, Tl)
    ml = (gt_jr[:mlen] - lom_jr[:mlen]).norm(dim=-1).mean().item() * 1000
    pl = np.mean([np.linalg.norm(pa(lom_j[t].cpu().numpy(), gt_j[t].cpu().numpy()) - gt_j[t].cpu().numpy(), axis=-1).mean() * 1000 for t in range(mlen)])
    lom_m.append(ml); lom_pa.append(pl)

    # Hybrid (NormalUpper 78D + lower_global 61D)
    # Upper: 13 SMPL-X joints [3,6,9,12,13,14,15,16,17,18,19,20,21] × 6D = 78D (same as LOM)
    # Lower 61D = 9 SMPL-X lower joints × 6D (54D) + local_vel (3D) + foot_contact (4D)
    #   [0:6]   = pelvis/global_orient 6D
    #   [6:54]  = 8 other lower joints × 6D
    #   [54:57] = local velocity 3D
    #   [57:61] = foot contact 4D (L_ankle, R_ankle, L_foot, R_foot)

    # Build SMPL-X 22-joint 6D (same as LOM)
    bp_aa_h = torch.tensor(d['body_pose'][:T], dtype=torch.float32, device=device).reshape(T, 21, 3)
    go_aa_h = torch.tensor(d['global_orient'][:T], dtype=torch.float32, device=device).reshape(T, 1, 3)
    full22_aa_h = torch.cat([go_aa_h, bp_aa_h], dim=1)
    full22_mat_h = axis_angle_to_matrix(full22_aa_h.reshape(-1, 3)).reshape(T, 22, 3, 3)
    full22_r6_h = matrix_to_rotation_6d(full22_mat_h.reshape(-1, 3, 3)).reshape(T, 22, 6)

    # Upper: same as LOM
    hud = full22_r6_h[:, lom_upper_idx].reshape(T, -1)  # (T, 78)

    # Lower: 9 joints in SMPL-X lower order [0,1,2,4,5,7,8,10,11] as 6D
    lower_r6 = full22_r6_h[:, lom_lower_idx].reshape(T, -1)  # (T, 54)
    local_vel = mv[:, 142:145]  # (T, 3)

    # Compute foot contact from FK joints velocity (threshold 0.01)
    with torch.no_grad():
        bm_fc = smplx_pkg.create(os.path.join(ROOT, 'model_files/smplx_models'), model_type='smplx',
                                  gender='neutral', batch_size=T, num_betas=10, use_pca=False, flat_hand_mean=True).to(device)
        fk_out = bm_fc(global_orient=go_aa_h.reshape(T,3), body_pose=bp_aa_h.reshape(T,-1), betas=betas[:T],
                        transl=torch.tensor(d['trans'][:T], dtype=torch.float32, device=device))
        foot_joints = fk_out.joints[:, [7, 8, 10, 11], :]  # (T, 4, 3)
    foot_vel = torch.zeros(T, 4, device=device)
    foot_vel[:-1] = (foot_joints[1:] - foot_joints[:-1]).norm(dim=-1)
    foot_contact = (foot_vel < 0.01).float()  # (T, 4)

    hld = torch.cat([lower_r6, local_vel, foot_contact], dim=-1)  # (T, 61)

    with torch.no_grad():
        hru = vae_hybrid_upper.decode(vae_hybrid_upper.map2index(hud.unsqueeze(0)).int())[0]
        hrl = vae_hybrid_lower.decode(vae_hybrid_lower.map2index(hld.unsqueeze(0)).int())[0]
    Th = min(T, hru.shape[0], hrl.shape[0])

    # Reassemble 22-joint 6D from upper + lower reconstruction
    rec22_r6_h = torch.zeros(Th, 22, 6, device=device)
    for i, idx in enumerate(lom_upper_idx): rec22_r6_h[:, idx] = hru[:Th, i*6:(i+1)*6]
    rec_lower_9 = hrl[:Th, :54].reshape(Th, 9, 6)
    for i, idx in enumerate(lom_lower_idx): rec22_r6_h[:, idx] = rec_lower_9[:, i]
    rec_vel_h = hrl[:Th, 54:57]

    # Convert 22-joint 6D → GENMO 145D for eval_rec
    rec_go_r6_h = rec22_r6_h[:, 0]
    rec_bp_r6_h = rec22_r6_h[:, 1:]
    rec145h = torch.cat([rec_bp_r6_h.reshape(Th, 126), betas[:Th], rec_go_r6_h, rec_vel_h], dim=-1)
    mh, ph = eval_rec(rec145h, "Hybrid")
    hyb_m.append(mh); hyb_pa.append(ph)

    cnt += 1
    if cnt % 10 == 0:
        print(f"  {cnt} done... LOM={np.mean(lom_m):.1f} Full={np.mean(full_m):.1f} Hybrid={np.mean(hyb_m):.1f}")
    if cnt >= 50: break

print(f"\nEvaluated {cnt} AMASS test samples (128 frames each)")
print(f"{'Model':<30} | {'MPJPE (mm)':>10} | {'PA-MPJPE (mm)':>13}")
print(f"{'-'*30}-+-{'-'*10}-+-{'-'*13}")
print(f"{'LOM (upper+lower)':<30} | {np.mean(lom_m):10.2f} | {np.mean(lom_pa):13.2f}")
print(f"{'Hybrid (Upper+GenmoLower)':<30} | {np.mean(hyb_m):10.2f} | {np.mean(hyb_pa):13.2f}")
print(f"{'GenmoFull (135D)':<30} | {np.mean(full_m):10.2f} | {np.mean(full_pa):13.2f}")
