#!/usr/bin/env python3
"""
Evaluate ViBES body model on Converse3D using Q2M evaluator v3.

Pipeline:
  1. For each question in Converse3D_eval test set, generate body motion via ViBES
  2. Convert generated motion to v3 format (22 joints × 6D + root velocity = 135-dim)
  3. Run balanced R-Precision, FID, Diversity, MM Dist using Q2M evaluator v3

Usage:
    # Generate + evaluate (single GPU)
    CUDA_VISIBLE_DEVICES=0 python scripts/eval_vibes_body_q2m.py \
        --vibes_checkpoint /path/to/body_checkpoint \
        --save_dir eval_results/body_v6

    # Multi-GPU generation + evaluation (single command)
    python scripts/eval_vibes_body_q2m.py \
        --vibes_checkpoint /path/to/body_checkpoint \
        --save_dir eval_results/body_v6 \
        --gpus 0,1,2,3

    # Evaluate only (from cached generations)
    python scripts/eval_vibes_body_q2m.py \
        --eval_only --save_dir eval_results/body_v6
"""
import os
import sys
import json
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

_script_dir = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(_script_dir, ".."))
if ROOT_DIR in sys.path:
    sys.path.remove(ROOT_DIR)
sys.path.insert(0, ROOT_DIR)

speech_related_path = os.path.join(ROOT_DIR, "speech_related")
cosyvoice_path = os.path.join(ROOT_DIR, "speech_related", "cosyvoice")
# Matcha-TTS is the cosyvoice/third_party submodule; speech_related/Matcha-TTS is an older local layout
matcha_path = os.path.join(ROOT_DIR, "speech_related", "cosyvoice", "third_party", "Matcha-TTS")
if not os.path.isdir(os.path.join(matcha_path, "matcha")):
    matcha_path = os.path.join(ROOT_DIR, "speech_related", "Matcha-TTS")
for p in [speech_related_path, cosyvoice_path, matcha_path]:
    if os.path.exists(p):
        if p in sys.path:
            sys.path.remove(p)
        sys.path.insert(1, p)

from omegaconf import OmegaConf
from torch.utils.data import Dataset, DataLoader

from multimodal_tokenizers.utils.rotation_conversions import (
    axis_angle_to_6d, axis_angle_to_matrix,
)
from multimodal_tokenizers.models.text_motion_clip import TextMotionCLIP
from multimodal_tokenizers.metrics.clip_eval import extract_embeddings
from multimodal_tokenizers.metrics.utils import (
    euclidean_distance_matrix_np,
    calculate_activation_statistics_np,
    calculate_frechet_distance_np,
    calculate_diversity_np,
)
from multimodal_tokenizers.data.converse3d_question_motion import (
    Converse3DQuestionMotionDataset,
    collate_fn,
)

# Import from eval_question_motion_stratified
from scripts.eval_question_motion_stratified import (
    balanced_r_precision,
    classify_dataset_from_jsonl,
)

# ============================================================================
# Constants
# ============================================================================
EVALUATOR_CHECKPOINT = os.path.join(ROOT_DIR, "model_files/pretrained_cpt/evaluator/question_motion_clip_v3_best.ckpt")
EVALUATOR_CONFIG = os.path.join(ROOT_DIR, "configs/evaluator/question_motion_clip_v3_body.yaml")
DATA_ROOT = "/path/to/Converse3D_eval"
GLM_BASE_PATH = "THUDM/glm-4-voice-9b"
VAE_CHECKPOINT_MAIN = os.path.join(ROOT_DIR, "model_files/pretrained_cpt/body/lom_vq.ckpt")
VAE_CHECKPOINT_GLOBAL = os.path.join(ROOT_DIR, "model_files/pretrained_cpt/VAE_Global_from_Lower54/last.ckpt")
TOKEN_OFFSET = 168736

MAX_LENGTH = 196
NUM_JOINTS_V3 = 22
FEATURE_DIM_V3 = 135  # 22*6 + 3
MOTION_FPS = 25


# ============================================================================
# Motion conversion: SMPLX axis-angle → v3 format
# ============================================================================

def convert_to_v3(poses, trans):
    """Convert SMPLX axis-angle poses + translation to v3 format (135-dim).

    Args:
        poses: (T, 165) axis-angle for 55 joints
        trans: (T, 3) world translation

    Returns:
        motion_v3: (T, 135) = 22 joints × 6D + root velocity
    """
    T = poses.shape[0]

    # 22 body joints → 6D rotation
    rot_body = poses[:, :66].reshape(T, 22, 3)  # first 22 joints
    rot_6d = axis_angle_to_6d(rot_body).reshape(T, 132)

    # Root-relative local velocity
    global_orient = poses[:, :3]
    velocity = torch.zeros_like(trans)
    velocity[1:] = trans[1:] - trans[:-1]
    velocity[0] = velocity[1] if T > 1 else velocity[0]
    R_inv = axis_angle_to_matrix(global_orient).transpose(-1, -2)
    local_vel = torch.einsum('tij,tj->ti', R_inv, velocity)

    return torch.cat([rot_6d, local_vel], dim=-1)  # (T, 135)


def pad_or_crop(motion, max_length):
    """Pad or crop motion to max_length frames."""
    T = motion.shape[0]
    if T >= max_length:
        return motion[:max_length], min(T, max_length)
    pad = torch.zeros(max_length - T, motion.shape[1], dtype=motion.dtype)
    return torch.cat([motion, pad], dim=0), T


# ============================================================================
# Generated motion dataset
# ============================================================================

class GeneratedMotionDataset(Dataset):
    """Dataset of generated motions for evaluation."""

    def __init__(self, save_dir, questions, segment_ids, max_length=196):
        self.save_dir = save_dir
        self.questions = questions
        self.segment_ids = segment_ids
        self.max_length = max_length

    def __len__(self):
        return len(self.questions)

    def __getitem__(self, idx):
        seg_id = self.segment_ids[idx]
        npz_path = os.path.join(self.save_dir, f"{seg_id}.npz")
        data = np.load(npz_path)
        poses = torch.from_numpy(data["poses"]).float()
        trans = torch.from_numpy(data["trans"]).float()

        motion_v3 = convert_to_v3(poses, trans)
        motion_padded, seq_len = pad_or_crop(motion_v3, self.max_length)

        return {
            "text": self.questions[idx],
            "motion": motion_padded,
            "length": seq_len,
        }


# ============================================================================
# Generation
# ============================================================================

def _load_vibes_model(vibes_checkpoint, glm_base_path, device):
    """Load ViBES from a full checkpoint or an Expert-1-only one (the released ViBES-Body-*)."""
    from transformers import AutoTokenizer, AutoConfig, AutoModel
    from transformers.modeling_utils import load_sharded_checkpoint
    sys.path.insert(0, os.path.join(ROOT_DIR, "training"))
    from expert_io import is_expert1_checkpoint, load_expert1_checkpoint

    tokenizer_path = os.path.join(ROOT_DIR, "vibes")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    config = AutoConfig.from_pretrained(tokenizer_path, trust_remote_code=True)
    model = AutoModel.from_config(
        config, trust_remote_code=True,
        torch_dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
        attn_implementation="flash_attention_2",
    ).to(device)
    if is_expert1_checkpoint(vibes_checkpoint):
        print(f"  Expert-1-only checkpoint; reconstructing Expert-0 from {glm_base_path}")
        load_expert1_checkpoint(model, vibes_checkpoint, glm_base_path)
    else:
        load_sharded_checkpoint(model, vibes_checkpoint)
    model.eval()
    return tokenizer, model


def _load_body_vaes(global_checkpoint, device):
    """Upper / lower / hand VQ-VAEs and the Global VAE (lower-body pose -> root velocity)."""
    from multimodal_tokenizers.archs.lom_vq import VQVAEConvZeroDSUS_PaperVersion, VAEConvZero

    main_ckpt = torch.load(VAE_CHECKPOINT_MAIN, map_location="cpu", weights_only=False)['state_dict']

    vae_upper = VQVAEConvZeroDSUS_PaperVersion(vae_layer=3, code_num=256, codebook_size=256, vae_quantizer_lambda=1, vae_test_dim=78)
    vae_upper.load_state_dict({k.replace('vae_upper.', ''): v for k, v in main_ckpt.items() if k.startswith('vae_upper.')}, strict=True)
    vae_upper.eval().to(device)

    vae_lower = VQVAEConvZeroDSUS_PaperVersion(vae_layer=3, code_num=256, codebook_size=256, vae_quantizer_lambda=1, vae_test_dim=54)
    vae_lower.load_state_dict({k.replace('vae_lower.', ''): v for k, v in main_ckpt.items() if k.startswith('vae_lower.')}, strict=True)
    vae_lower.eval().to(device)

    vae_hand = VQVAEConvZeroDSUS_PaperVersion(vae_layer=3, code_num=256, codebook_size=256, vae_quantizer_lambda=1, vae_test_dim=180)
    vae_hand.load_state_dict({k.replace('vae_hand.', ''): v for k, v in main_ckpt.items() if k.startswith('vae_hand.')}, strict=True)
    vae_hand.eval().to(device)

    vae_global = VAEConvZero(vae_layer=4, code_num=256, codebook_size=256, vae_quantizer_lambda=1, vae_test_dim=61)
    global_ckpt = torch.load(global_checkpoint, map_location="cpu", weights_only=False)
    global_sd = global_ckpt['state_dict'] if 'state_dict' in global_ckpt else global_ckpt
    missing, unexpected = vae_global.load_state_dict(
        {k.replace('vae_global.', ''): v for k, v in global_sd.items() if 'vae_global' in k}, strict=False)
    # Older Global VAEs predate the zero-init dataset-conditioning layers, which then load as a no-op
    assert not unexpected and all(k.startswith(('dataset_embedding', 'dataset_gamma')) for k in missing), \
        f"Global VAE mismatch: missing={missing}, unexpected={unexpected}"
    vae_global.eval().to(device)

    return vae_upper, vae_lower, vae_hand, vae_global


def generate_all_motions(vibes_checkpoint, save_dir, test_samples, device, num_gpus=1, gpu_id=0,
                         glm_base_path=GLM_BASE_PATH, global_checkpoint=VAE_CHECKPOINT_GLOBAL,
                         agent=False):
    """Generate body motion for each test question using ViBES.

    When num_gpus > 1, each GPU (or each --shard_id) processes a disjoint shard of samples.
    All shards write to the same save_dir; finished samples are skipped, so runs can resume.
    """
    os.makedirs(save_dir, exist_ok=True)

    # Shuffle deterministically then shard across GPUs
    import hashlib
    all_samples = sorted(test_samples, key=lambda s: hashlib.md5(s["segment_id"].encode()).hexdigest())
    if num_gpus > 1:
        shard = [s for i, s in enumerate(all_samples) if i % num_gpus == gpu_id]
        print(f"  GPU {gpu_id}/{num_gpus}: assigned {len(shard)}/{len(all_samples)} samples")
        test_samples = shard
    else:
        test_samples = all_samples

    # Check which samples already generated (resume support)
    all_tasks = [(s["segment_id"], s["question"]) for s in test_samples]
    existing = sum(1 for sid, _ in all_tasks if os.path.exists(os.path.join(save_dir, f"{sid}.npz")))
    print(f"  {existing}/{len(all_tasks)} already generated for this GPU")
    if existing == len(all_tasks):
        return

    print("  Loading ViBES model...")
    tokenizer, model = _load_vibes_model(vibes_checkpoint, glm_base_path, device)

    print(f"  Loading body VAEs (Global VAE: {global_checkpoint})...")
    vaes = _load_body_vaes(global_checkpoint, device)

    # Generate (check file existence per-iteration for multi-node support)
    failed = 0
    skipped = 0
    failed_ids = []
    for i, (seg_id, question) in enumerate(tqdm(all_tasks, desc="Generating")):
        if os.path.exists(os.path.join(save_dir, f"{seg_id}.npz")):
            skipped += 1
            continue
        try:
            success = _generate_single(seg_id, question, model, tokenizer, device, vaes, save_dir, agent)
            if not success:
                failed += 1
                failed_ids.append(seg_id)
        except Exception as e:
            print(f"  [{i}] {seg_id}: ERROR {e}")
            failed += 1
            failed_ids.append(seg_id)

    print(f"  Generation complete. Skipped: {skipped}, Failed: {failed}/{len(all_tasks)}")
    if failed_ids:
        print(f"  Failed IDs: {failed_ids[:20]}{'...' if len(failed_ids) > 20 else ''}")


def _generate_single(seg_id, question, model, tokenizer, device, vaes, save_dir, agent=False):
    """Generate and save a single sequence. Returns True on success, False on failure."""
    from utils.inference_utils import prepare_modality_masks, create_prompt
    from utils.token_utils import extract_modality_tokens_from_response
    from utils.tensor_utils import apply_body_token_offset

    prompt = create_prompt(question, agent=agent)
    inputs = tokenizer([prompt], return_tensors="pt").to(device)
    batch_size, seq_len = inputs.input_ids.shape
    modality_masks, position_encoding_indices = prepare_modality_masks(
        batch_size, seq_len, num_modalities=3, device=device
    )

    with torch.no_grad():
        output_ids, output_modality_masks = model.generate(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_new_tokens=1024,
            do_sample=True,
            temperature=0.1,
            top_p=0.1,
            modality_masks=modality_masks,
            use_cache=True,
            position_encoding_indices=position_encoding_indices,
            body_part="body",
        )

    output_ids = apply_body_token_offset(output_ids, output_modality_masks, TOKEN_OFFSET, modality_idx=2)
    full_response = tokenizer.decode(output_ids[0], skip_special_tokens=False)
    modality_tokens = extract_modality_tokens_from_response(full_response)

    upper_tokens = modality_tokens.get('upper', [])
    lower_tokens = modality_tokens.get('lower', [])
    hand_tokens = modality_tokens.get('hand', [])

    if not upper_tokens or not lower_tokens:
        print(f"  {seg_id}: empty tokens, skipping")
        return False

    _decode_and_save(seg_id, upper_tokens, lower_tokens, hand_tokens, vaes, save_dir, device)
    return True


def _decode_and_save(seg_id, upper_tokens, lower_tokens, hand_tokens, vaes, save_dir, device):
    """Decode body tokens to SMPL-X poses + Global-VAE translation and save them with the tokens."""
    from utils.tensor_utils import inverse_selection_tensor
    from multimodal_tokenizers.data.mixed_dataset.data_tools import (
        JOINT_MASK_UPPER, JOINT_MASK_LOWER, JOINT_MASK_HANDS,
    )
    from multimodal_tokenizers.utils.rotation_conversions import (
        rotation_6d_to_axis_angle, rotation_6d_to_matrix,
        matrix_to_axis_angle, matrix_to_rotation_6d,
    )

    LOWER_DIM = 54
    vae_upper, vae_lower, vae_hand, vae_global = vaes

    with torch.no_grad():
        rec_upper = vae_upper.decode(torch.tensor(upper_tokens, device=device).unsqueeze(0).int())
        rec_lower = vae_lower.decode(torch.tensor(lower_tokens, device=device).unsqueeze(0).int())
        rec_hand = vae_hand.decode(torch.tensor(hand_tokens, device=device).unsqueeze(0).int()) if len(hand_tokens) else None

    n = min(rec_upper.shape[1], rec_lower.shape[1])
    if rec_hand is not None:
        n = min(n, rec_hand.shape[1])

    pose_upper = rec_upper[0, :n].reshape(n, 13, 6)
    pose_upper_aa = rotation_6d_to_axis_angle(pose_upper).reshape(n, 39)
    pose_upper_full = inverse_selection_tensor(pose_upper_aa.to(device), JOINT_MASK_UPPER, n)

    pose_lower = rec_lower[0, :n, :LOWER_DIM].reshape(n, 9, 6)
    pose_lower_mat = rotation_6d_to_matrix(pose_lower)
    lower2global = matrix_to_rotation_6d(pose_lower_mat.clone()).reshape(1, n, 54)
    pose_lower_aa = matrix_to_axis_angle(pose_lower_mat).reshape(n, 27)
    pose_lower_full = inverse_selection_tensor(pose_lower_aa, JOINT_MASK_LOWER, n)

    # The evaluator only sees the 22 body joints, so the hands stay at rest here
    pose_hands = torch.zeros(n, 90, device=device)
    pose_hands_full = inverse_selection_tensor(pose_hands, JOINT_MASK_HANDS, n)

    rec_pose = pose_upper_full + pose_lower_full + pose_hands_full

    to_global = rec_lower[:, :n, :LOWER_DIM]
    to_global = F.pad(to_global, (0, 7))
    to_global[:, :, 54:57] = 0.0
    to_global[:, :, :LOWER_DIM] = lower2global
    with torch.no_grad():
        rec_global = vae_global(to_global)

    rec_trans_v = rec_global["rec_pose"][:, :, 54:57]
    global_orient_aa = rec_pose[:, :3]
    go_6d = axis_angle_to_6d(global_orient_aa.unsqueeze(1)).squeeze(1)
    R = rotation_6d_to_matrix(go_6d)
    world_vel = torch.einsum("tij,tj->ti", R, rec_trans_v[0])
    rec_trans = torch.cumsum(world_vel, dim=0)

    np.savez(
        os.path.join(save_dir, f"{seg_id}.npz"),
        poses=rec_pose.detach().cpu().numpy(),
        trans=rec_trans.detach().cpu().numpy(),
        mocap_frame_rate=MOTION_FPS,
        upper_tokens=np.asarray(upper_tokens, dtype=np.int64),
        lower_tokens=np.asarray(lower_tokens, dtype=np.int64),
        hand_tokens=np.asarray(hand_tokens, dtype=np.int64),
    )


def redecode_motions(src_dir, save_dir, global_checkpoint, device):
    """Decode the tokens saved in src_dir again, e.g. with another Global VAE, into save_dir."""
    files = sorted(f for f in os.listdir(src_dir) if f.endswith(".npz"))
    vaes = _load_body_vaes(global_checkpoint, device)
    os.makedirs(save_dir, exist_ok=True)
    for f in tqdm(files, desc="Re-decoding"):
        data = np.load(os.path.join(src_dir, f))
        if "upper_tokens" not in data:
            raise ValueError(f"{f} has no saved tokens; generate it with this script first")
        _decode_and_save(f[:-len(".npz")], data["upper_tokens"], data["lower_tokens"],
                         data["hand_tokens"], vaes, save_dir, device)
    print(f"  Re-decoded {len(files)} samples into {save_dir}")


def _worker_generate(vibes_checkpoint, save_dir, test_samples, gpu_id, num_gpus, rank,
                     glm_base_path, global_checkpoint, agent):
    """Worker process for multi-GPU generation."""
    device = torch.device(f"cuda:{gpu_id}")
    print(f"  [Worker {rank}] GPU {gpu_id}, generating on {device}")
    generate_all_motions(vibes_checkpoint, save_dir, test_samples, device,
                         num_gpus=num_gpus, gpu_id=rank, glm_base_path=glm_base_path,
                         global_checkpoint=global_checkpoint, agent=agent)
    print(f"  [Worker {rank}] Done.")



# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Evaluate ViBES body model on Converse3D with Q2M evaluator v3")
    parser.add_argument("--vibes_checkpoint", type=str, default="./ViBES-Body-Agent",
                        help="Body checkpoint: the released ViBES-Body-Agent (Expert-1-only) or a full checkpoint")
    parser.add_argument("--glm_base_path", type=str, default=GLM_BASE_PATH,
                        help="GLM-4-Voice base used to rebuild Expert-0 for Expert-1-only checkpoints")
    parser.add_argument("--global_checkpoint", type=str, default=VAE_CHECKPOINT_GLOBAL,
                        help="Global VAE that recovers root translation from the lower-body pose")
    parser.add_argument("--agent", action="store_true",
                        help="Use the agent system prompt / assistant prefix (create_prompt(agent=True))")
    parser.add_argument("--save_dir", type=str, default="eval_results/body_v6",
                        help="Directory to save/load generated motions")
    parser.add_argument("--eval_only", action="store_true",
                        help="Skip generation, evaluate from cached results")
    parser.add_argument("--redecode_from", type=str, default=None,
                        help="Decode the tokens saved in this directory again (with --global_checkpoint) "
                             "into --save_dir instead of generating, then evaluate")
    parser.add_argument("--num_shards", type=int, default=1,
                        help="Split generation into this many shards (e.g. one SLURM task each)")
    parser.add_argument("--shard_id", type=int, default=0,
                        help="Shard to generate when --num_shards > 1; exits without evaluating")
    parser.add_argument("--results_json", type=str, default=None,
                        help="Where to write the metrics (default: <save_dir>/q2m_results.json)")
    parser.add_argument("--data_root", type=str, default=DATA_ROOT)
    parser.add_argument("--evaluator_checkpoint", type=str, default=EVALUATOR_CHECKPOINT)
    parser.add_argument("--evaluator_config", type=str, default=EVALUATOR_CONFIG)
    parser.add_argument("--gpus", type=str, default="0",
                        help="Comma-separated GPU IDs for parallel generation (e.g. '0,1,2,3')")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    gpu_ids = [int(g) for g in args.gpus.split(",")]
    num_gpus = len(gpu_ids)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Load test questions
    print("\n=== Loading test set ===")
    jsonl_path = os.path.join(args.data_root, "test.jsonl")
    test_samples = []
    with open(jsonl_path) as f:
        for line in f:
            rec = json.loads(line.strip())
            if rec.get("question"):
                test_samples.append({
                    "segment_id": rec["segment_id"],
                    "question": rec["question"],
                    "source": rec.get("source", "unknown"),
                })
    print(f"  {len(test_samples)} test questions loaded")

    gen_kwargs = dict(glm_base_path=args.glm_base_path,
                      global_checkpoint=args.global_checkpoint, agent=args.agent)

    # Step 1: Generate motions
    if args.redecode_from:
        print(f"\n=== Re-decoding tokens from {args.redecode_from} ===")
        redecode_motions(args.redecode_from, args.save_dir, args.global_checkpoint,
                         torch.device(f"cuda:{gpu_ids[0]}"))
    elif args.num_shards > 1 and not args.eval_only:
        print(f"\n=== Generating shard {args.shard_id}/{args.num_shards} on GPU {gpu_ids[0]} ===")
        generate_all_motions(args.vibes_checkpoint, args.save_dir, test_samples,
                             torch.device(f"cuda:{gpu_ids[0]}"),
                             num_gpus=args.num_shards, gpu_id=args.shard_id, **gen_kwargs)
        print("  Shard done. Run with --eval_only once all shards have finished.")
        return
    elif not args.eval_only:
        if num_gpus > 1:
            print(f"\n=== Generating motions on {num_gpus} GPUs: {gpu_ids} ===")
            import multiprocessing as mp
            mp.set_start_method("spawn", force=True)
            processes = []
            for rank, gid in enumerate(gpu_ids):
                p = mp.Process(
                    target=_worker_generate,
                    args=(args.vibes_checkpoint, args.save_dir, test_samples,
                          gid, num_gpus, rank, args.glm_base_path,
                          args.global_checkpoint, args.agent),
                )
                p.start()
                processes.append(p)
            for p in processes:
                p.join()
            print("  All GPUs finished generation.")
        else:
            print(f"\n=== Generating motions on GPU {gpu_ids[0]} ===")
            device = torch.device(f"cuda:{gpu_ids[0]}")
            generate_all_motions(args.vibes_checkpoint, args.save_dir, test_samples, device,
                                 num_gpus=1, gpu_id=0, **gen_kwargs)
    else:
        print(f"\n=== Skipping generation (--eval_only), loading from {args.save_dir} ===")

    device = torch.device(f"cuda:{gpu_ids[0]}")

    # Check how many were generated
    generated = set(f.replace(".npz", "") for f in os.listdir(args.save_dir) if f.endswith(".npz"))
    valid_samples = [s for s in test_samples if s["segment_id"] in generated]
    print(f"  {len(valid_samples)}/{len(test_samples)} samples available for evaluation")

    if len(valid_samples) < 10:
        print("ERROR: too few generated samples for evaluation (need at least 10)")
        return
    if len(valid_samples) < len(test_samples):
        print(f"  WARNING: only {len(valid_samples)}/{len(test_samples)} samples generated, metrics are partial")

    # Step 2: Load Q2M evaluator
    print("\n=== Loading Q2M evaluator v3 ===")
    cfg = OmegaConf.load(args.evaluator_config)
    cfg.DATASET.data_root = args.data_root
    evaluator = TextMotionCLIP(cfg)
    ckpt = torch.load(args.evaluator_checkpoint, map_location="cpu", weights_only=False)
    evaluator.load_state_dict(ckpt.get("state_dict", ckpt), strict=True)
    evaluator.to(device).eval()

    # Step 3: Build dataloaders
    print("\n=== Building dataloaders ===")
    questions = [s["question"] for s in valid_samples]
    segment_ids = [s["segment_id"] for s in valid_samples]

    gen_dataset = GeneratedMotionDataset(args.save_dir, questions, segment_ids, max_length=MAX_LENGTH)
    gen_loader = DataLoader(gen_dataset, batch_size=64, shuffle=False, num_workers=4, collate_fn=collate_fn, pin_memory=True)

    # GT test loader (for FID reference)
    num_joints = getattr(cfg.DATASET, "num_joints", 22)
    gt_dataset = Converse3DQuestionMotionDataset(
        data_root=cfg.DATASET.data_root,
        motion_dir=cfg.DATASET.motion_dir,
        text_dir=cfg.DATASET.text_dir,
        split="test",
        max_length=cfg.DATASET.max_length,
        is_train=False,
        num_joints=num_joints,
    )
    gt_loader = DataLoader(gt_dataset, batch_size=64, shuffle=False, num_workers=4, collate_fn=collate_fn, pin_memory=True)

    # Step 4: Extract embeddings
    print("\n=== Extracting embeddings ===")
    gen_motion_embs, gen_text_embs = extract_embeddings(evaluator, gen_loader, device, normalize_embeddings=False)
    gt_motion_embs, gt_text_embs = extract_embeddings(evaluator, gt_loader, device, normalize_embeddings=False)
    print(f"  Generated: {gen_motion_embs.shape[0]} samples")
    print(f"  GT: {gt_motion_embs.shape[0]} samples")

    # Step 5: Classify question types
    source_map = classify_dataset_from_jsonl(args.data_root, split="test")
    question_types = []
    for s in valid_samples:
        src = source_map.get(s["segment_id"], "unknown")
        question_types.append("motion" if src == "amass" else "conversational")

    n_motion = sum(1 for t in question_types if t == "motion")
    n_conv = sum(1 for t in question_types if t == "conversational")
    print(f"  {n_motion} motion-descriptive, {n_conv} conversational")

    # Step 6: Compute metrics
    eval_cfg = cfg.EVAL_METRICS
    replication_times = eval_cfg.REPLICATION_TIMES
    R_size = eval_cfg.R_SIZE
    emb_scale = eval_cfg.EMB_SCALE
    diversity_pair_divisor = eval_cfg.DIVERSITY_PAIR_DIVISOR
    diversity_times = eval_cfg.DIVERSITY_TIMES

    # 6a. Balanced R-Precision (generated text ↔ generated motion)
    print(f"\n=== Balanced R-Precision ({replication_times} replications) ===")
    all_r = {f"top_{k}": [] for k in [1, 2, 3]}
    all_r_motion = {f"top_{k}": [] for k in [1, 2, 3]}
    all_r_conv = {f"top_{k}": [] for k in [1, 2, 3]}
    all_mm = []

    for rep in range(replication_times):
        r_prec, match_score, stats = balanced_r_precision(
            gen_text_embs, gen_motion_embs, question_types,
            R_size=R_size, top_k=3,
        )
        for k in [1, 2, 3]:
            all_r[f"top_{k}"].append(r_prec[f"top_{k}"])
            all_r_motion[f"top_{k}"].append(r_prec.get(f"top_{k}_motion", 0.0))
            all_r_conv[f"top_{k}"].append(r_prec.get(f"top_{k}_conv", 0.0))
        all_mm.append(match_score)

    ci = 1.96 / np.sqrt(replication_times)

    # 6b. FID (generated vs GT)
    gen_stats = calculate_activation_statistics_np(gen_motion_embs, emb_scale=emb_scale)
    gt_stats = calculate_activation_statistics_np(gt_motion_embs, emb_scale=emb_scale)
    fid = calculate_frechet_distance_np(gen_stats[0], gen_stats[1], gt_stats[0], gt_stats[1])

    # 6c. Diversity
    n_pairs = min(diversity_times, len(gen_motion_embs) - 1)
    diversity = calculate_diversity_np(gen_motion_embs, n_pairs, emb_scale=emb_scale, pair_divisor=diversity_pair_divisor)

    # Print results
    print("\n" + "=" * 80)
    print("  ViBES Body Model — Q2M Evaluator v3 Results")
    print("=" * 80)
    print(f"  Generations: {args.save_dir}")
    print(f"  Samples: {len(valid_samples)} ({n_motion} motion, {n_conv} conv)")

    print(f"\n  {'Metric':<25} {'Balanced':>20} {'Motion-only':>20} {'Conv-only':>20}")
    print("-" * 90)
    for k in [1, 2, 3]:
        bv = np.array(all_r[f"top_{k}"])
        mv = np.array(all_r_motion[f"top_{k}"])
        cv = np.array(all_r_conv[f"top_{k}"])
        print(f"  R-Prec Top-{k}             {bv.mean():.4f}+/-{bv.std()*ci:.4f}"
              f"       {mv.mean():.4f}+/-{mv.std()*ci:.4f}"
              f"       {cv.mean():.4f}+/-{cv.std()*ci:.4f}")

    mm = np.array(all_mm)
    print(f"  {'MM Dist':<25} {mm.mean():.4f}+/-{mm.std()*ci:.4f}")
    print(f"  {'FID':<25} {fid:.4f}")
    print(f"  {'Diversity':<25} {diversity:.4f}")

    results = {
        "save_dir": args.save_dir,
        "num_samples": len(valid_samples), "num_motion": n_motion, "num_conv": n_conv,
        "mm_dist": float(mm.mean()), "mm_dist_ci": float(mm.std() * ci),
        "fid": float(fid), "diversity": float(diversity),
    }
    for name, vals in (("balanced", all_r), ("motion", all_r_motion), ("conv", all_r_conv)):
        for k in [1, 2, 3]:
            v = np.array(vals[f"top_{k}"])
            results[f"r{k}_{name}"] = float(v.mean())
            results[f"r{k}_{name}_ci"] = float(v.std() * ci)
    results_json = args.results_json or os.path.join(args.save_dir, "q2m_results.json")
    with open(results_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Saved metrics to {results_json}")

    print(f"\n  GT Baseline (v3):")
    print(f"    Motion R-Prec Top-1: 0.5464, Top-3: 0.8162")
    print(f"    MM Dist: 2.34, FID: 0.00, Diversity: 11.05")
    print("=" * 80)


if __name__ == "__main__":
    main()
