"""
Stratified GT evaluation for Question2Motion CLIP evaluator.

Addresses the issue where general/conversational questions can match any
gesture equally well, contaminating R-Precision. Solution:

1. Classify questions as 'motion-descriptive' or 'conversational'
2. Construct R-Precision batches with at most 1 conversational sample each
   (remaining slots filled by motion-descriptive samples)
3. Leftover motion-descriptive samples form pure motion-descriptive batches
4. All samples used exactly once per replication

Usage:
    CUDA_VISIBLE_DEVICES=0 python scripts/eval_question_motion_stratified.py \
        --checkpoint model_files/pretrained_cpt/evaluator/question_motion_clip_best.ckpt \
        --data-root /path/to/Converse3D_eval
"""

import os
import sys
import argparse
import json
import numpy as np
import torch

_script_dir = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(_script_dir, ".."))
if ROOT_DIR in sys.path:
    sys.path.remove(ROOT_DIR)
sys.path.insert(0, ROOT_DIR)

from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from multimodal_tokenizers.models.text_motion_clip import TextMotionCLIP
from multimodal_tokenizers.data.converse3d_question_motion import (
    Converse3DQuestionMotionDataset,
    collate_fn,
)
from multimodal_tokenizers.metrics.clip_eval import (
    extract_embeddings,
    compute_r_precision_and_matching_score,
    calculate_top_k_np,
)
from multimodal_tokenizers.metrics.utils import (
    euclidean_distance_matrix_np,
    calculate_activation_statistics_np,
    calculate_frechet_distance_np,
    calculate_diversity_np,
)


# ============================================================================
# Question type classifier
# ============================================================================

def classify_dataset_from_jsonl(data_root, split="test"):
    """Classify questions by dataset source from JSONL metadata.

    - source='amass' → motion-descriptive (text describes specific motion)
    - source='beat2' → conversational (general gesture, any gesture is valid)

    Returns:
        types: list of 'motion' or 'conversational' for each sample
        source_map: dict mapping segment_id → source string
    """
    import json
    jsonl_file = os.path.join(data_root, f"{split}.jsonl")
    source_map = {}
    with open(jsonl_file) as f:
        for line in f:
            line = line.strip()
            if line:
                rec = json.loads(line)
                if rec.get("question"):
                    source_map[rec["segment_id"]] = rec.get("source", "unknown")
    return source_map


# ============================================================================
# Stratified R-Precision
# ============================================================================

def balanced_r_precision(
    text_embs,
    motion_embs,
    question_types,
    R_size=32,
    top_k=3,
):
    """
    Compute R-Precision with balanced batches and type-aware matching.

    Batch construction:
      Each batch has R_size/2 motion-descriptive + R_size/2 conversational.
      Leftover motion-descriptive samples form pure motion batches.

    Matching criteria:
      - Motion-descriptive: correct iff its exact paired motion is in top-k
        (strict 1-to-1, standard R-Precision)
      - Conversational: correct if ANY conversational gesture in the batch
        is in top-k (relaxed matching — any general gesture is acceptable)

    Returns:
        r_precision: dict with 'top_1', 'top_2', 'top_3' (overall + per-type)
        matching_score: float
        stats: dict with batch composition info
    """
    types = np.array(question_types)

    motion_indices = np.where(types == "motion")[0]
    conv_indices = np.where(types == "conversational")[0]

    np.random.shuffle(motion_indices)
    np.random.shuffle(conv_indices)

    n_motion = len(motion_indices)
    n_conv = len(conv_indices)
    half = R_size // 2

    # Balanced batches: limited by the smaller pool
    n_balanced = min(n_motion // half, n_conv // half)

    batches = []       # list of index arrays
    batch_types = []   # list of type arrays (parallel to batches)

    # Phase 1: balanced batches (half motion + half conv)
    for i in range(n_balanced):
        m_slice = motion_indices[i * half:(i + 1) * half]
        c_slice = conv_indices[i * half:(i + 1) * half]
        batch = np.concatenate([m_slice, c_slice])
        btypes = (["motion"] * half) + (["conversational"] * half)
        batches.append(batch)
        batch_types.append(btypes)

    motion_used = n_balanced * half
    conv_used = n_balanced * half

    # Phase 2: pure motion batches from remaining motion samples
    remaining_motion = motion_indices[motion_used:]
    n_pure = len(remaining_motion) // R_size
    for i in range(n_pure):
        batch = remaining_motion[i * R_size:(i + 1) * R_size]
        btypes = ["motion"] * R_size
        batches.append(batch)
        batch_types.append(btypes)

    # Type-aware R-Precision scoring
    matching_score_sum = 0.0
    top_k_counts = np.zeros(top_k)           # overall
    top_k_counts_motion = np.zeros(top_k)    # motion-descriptive only
    top_k_counts_conv = np.zeros(top_k)      # conversational only
    total_samples = 0
    total_motion = 0
    total_conv = 0

    for batch_idx, btypes in zip(batches, batch_types):
        batch_idx = list(batch_idx)
        batch_text = text_embs[batch_idx]
        batch_motion = motion_embs[batch_idx]
        B = len(batch_idx)

        dist_mat = euclidean_distance_matrix_np(batch_text, batch_motion)
        matching_score_sum += np.trace(dist_mat)

        sorted_indices = np.argsort(dist_mat, axis=1)  # (B, B)
        conv_positions = {j for j in range(B) if btypes[j] == "conversational"}

        for i in range(B):
            for k in range(top_k):
                top_k_set = set(sorted_indices[i, :k + 1].tolist())
                if btypes[i] == "motion":
                    correct = i in top_k_set
                    if correct:
                        top_k_counts[k] += 1
                        top_k_counts_motion[k] += 1
                else:
                    correct = bool(conv_positions & top_k_set)
                    if correct:
                        top_k_counts[k] += 1
                        top_k_counts_conv[k] += 1

        total_samples += B
        total_motion += sum(1 for t in btypes if t == "motion")
        total_conv += sum(1 for t in btypes if t == "conversational")

    if total_samples == 0:
        return {f"top_{k+1}": 0.0 for k in range(top_k)}, 0.0, {}

    r_precision = {}
    for k in range(top_k):
        r_precision[f"top_{k+1}"] = top_k_counts[k] / total_samples
        if total_motion > 0:
            r_precision[f"top_{k+1}_motion"] = top_k_counts_motion[k] / total_motion
        if total_conv > 0:
            r_precision[f"top_{k+1}_conv"] = top_k_counts_conv[k] / total_conv

    matching_score = matching_score_sum / total_samples

    stats = {
        "n_motion_descriptive": n_motion,
        "n_conversational": n_conv,
        "n_balanced_batches": n_balanced,
        "n_pure_motion_batches": n_pure,
        "n_motion_used": motion_used + n_pure * R_size,
        "n_conv_used": conv_used,
        "n_motion_leftover": len(remaining_motion) - n_pure * R_size,
        "n_conv_leftover": n_conv - conv_used,
        "total_samples_evaluated": total_samples,
        "total_motion_evaluated": total_motion,
        "total_conv_evaluated": total_conv,
    }

    return r_precision, matching_score, stats


# ============================================================================
# Main
# ============================================================================

def load_model(cfg, checkpoint_path, device):
    model = TextMotionCLIP(cfg)
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = ckpt.get("state_dict", ckpt)
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()
    return model


def main():
    parser = argparse.ArgumentParser(
        description="Stratified GT evaluation for QuestionMotionCLIP"
    )
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="Path to CLIP checkpoint"
    )
    parser.add_argument(
        "--data-root", type=str,
        default="/path/to/Converse3D_eval",
    )
    parser.add_argument(
        "--config", type=str,
        default=os.path.join(ROOT_DIR, "configs/evaluator/question_motion_clip.yaml"),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--show-classification", action="store_true",
        help="Print question classification examples",
    )
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load config
    cfg = OmegaConf.load(args.config)
    if args.data_root:
        cfg.DATASET.data_root = args.data_root

    # Load model
    print(f"Loading checkpoint: {args.checkpoint}")
    model = load_model(cfg, args.checkpoint, device)

    # Build test dataset and classify questions by data source
    num_joints = getattr(cfg.DATASET, "num_joints", 55)
    test_dataset = Converse3DQuestionMotionDataset(
        data_root=cfg.DATASET.data_root,
        motion_dir=cfg.DATASET.motion_dir,
        text_dir=cfg.DATASET.text_dir,
        split="test",
        max_length=cfg.DATASET.max_length,
        is_train=False,
        num_joints=num_joints,
    )
    source_map = classify_dataset_from_jsonl(cfg.DATASET.data_root, split="test")

    # Map each sample to question type via segment_id → source
    question_types = []
    for sample in test_dataset.samples:
        src = source_map.get(sample["segment_id"], "unknown")
        # amass = motion-descriptive, beat2 = conversational
        question_types.append("motion" if src == "amass" else "conversational")

    n_motion = sum(1 for t in question_types if t == "motion")
    n_conv = sum(1 for t in question_types if t == "conversational")
    print(f"\nQuestion classification (by source): "
          f"{n_motion} motion-descriptive (amass), {n_conv} conversational (beat2)")

    if args.show_classification:
        print("\n--- Motion-descriptive (amass) examples ---")
        shown = 0
        for i, t in enumerate(question_types):
            if t == "motion" and shown < 5:
                print(f"  [{i}] {test_dataset.samples[i]['question'][:100]}")
                shown += 1
        print("\n--- Conversational (beat2) examples ---")
        shown = 0
        for i, t in enumerate(question_types):
            if t == "conversational" and shown < 5:
                print(f"  [{i}] {test_dataset.samples[i]['question'][:100]}")
                shown += 1

    # Build dataloader
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    # Eval parameters
    eval_cfg = cfg.EVAL_METRICS
    replication_times = eval_cfg.REPLICATION_TIMES
    R_size = eval_cfg.R_SIZE
    diversity_times = eval_cfg.DIVERSITY_TIMES
    emb_scale = eval_cfg.EMB_SCALE
    diversity_pair_divisor = eval_cfg.DIVERSITY_PAIR_DIVISOR

    # 1. Extract embeddings (once)
    print("\nExtracting test embeddings...")
    test_motion_embs, test_text_embs = extract_embeddings(
        model, test_loader, device, normalize_embeddings=False
    )
    print(f"  Test: {test_motion_embs.shape[0]} samples, embed_dim={test_motion_embs.shape[1]}")

    # 2. Balanced R-Precision with type-aware matching (replicated)
    print(f"\nComputing balanced R-Precision ({replication_times} replications)...")
    all_r = {f"top_{k}": [] for k in [1, 2, 3]}
    all_r_motion = {f"top_{k}": [] for k in [1, 2, 3]}
    all_r_conv = {f"top_{k}": [] for k in [1, 2, 3]}
    all_matching_scores = []
    last_stats = None

    for rep in range(replication_times):
        r_prec, match_score, stats = balanced_r_precision(
            test_text_embs, test_motion_embs, question_types,
            R_size=R_size, top_k=3,
        )
        for k in [1, 2, 3]:
            all_r[f"top_{k}"].append(r_prec[f"top_{k}"])
            all_r_motion[f"top_{k}"].append(r_prec.get(f"top_{k}_motion", 0.0))
            all_r_conv[f"top_{k}"].append(r_prec.get(f"top_{k}_conv", 0.0))
        all_matching_scores.append(match_score)
        last_stats = stats

    ci_factor = 1.96 / np.sqrt(replication_times)

    # 3. Also compute standard (non-stratified) R-Precision for comparison
    print("Computing standard R-Precision for comparison...")
    all_r_standard = {f"top_{k}": [] for k in [1, 2, 3]}
    all_match_standard = []
    for rep in range(replication_times):
        indices = np.random.permutation(len(test_motion_embs))
        r_prec, match_score = compute_r_precision_and_matching_score(
            test_text_embs[indices], test_motion_embs[indices],
            top_k=3, R_size=R_size,
        )
        for k in [1, 2, 3]:
            all_r_standard[f"top_{k}"].append(r_prec[f"top_{k}"])
        all_match_standard.append(match_score)

    # 4. FID (GT: test vs test → ~0)
    test_stats = calculate_activation_statistics_np(test_motion_embs, emb_scale=emb_scale)
    fid = calculate_frechet_distance_np(
        test_stats[0], test_stats[1], test_stats[0], test_stats[1]
    )

    # 5. Diversity
    n_pairs = min(diversity_times, len(test_motion_embs) - 1)
    diversity = calculate_diversity_np(
        test_motion_embs, n_pairs,
        emb_scale=emb_scale, pair_divisor=diversity_pair_divisor,
    )

    # Print results
    print("\n" + "=" * 80)
    print("  QuestionMotionCLIP — Balanced GT Evaluation")
    print("=" * 80)
    print(f"  Checkpoint: {os.path.basename(args.checkpoint)}")
    print(f"  Test samples: {len(test_motion_embs)} "
          f"({n_motion} motion-descriptive, {n_conv} conversational)")
    print(f"  Batch composition: {last_stats['n_balanced_batches']} balanced "
          f"({R_size//2} motion + {R_size//2} conv each) + "
          f"{last_stats['n_pure_motion_batches']} pure motion")
    print(f"  Coverage: {last_stats['n_motion_used']}/{n_motion} motion, "
          f"{last_stats['n_conv_used']}/{n_conv} conv")
    print(f"  Matching: motion=strict 1-to-1, conv=any conv gesture in batch")

    print("\n" + "-" * 80)
    print(f"  {'Metric':<25} {'Balanced':>15} {'Motion-only':>15} {'Conv-only':>15} {'Standard':>15}")
    print("-" * 80)
    for k in [1, 2, 3]:
        bv = np.array(all_r[f"top_{k}"])
        mv = np.array(all_r_motion[f"top_{k}"])
        cv = np.array(all_r_conv[f"top_{k}"])
        sv = np.array(all_r_standard[f"top_{k}"])
        b_str = f"{bv.mean():.4f}+/-{bv.std()*ci_factor:.4f}"
        m_str = f"{mv.mean():.4f}+/-{mv.std()*ci_factor:.4f}"
        c_str = f"{cv.mean():.4f}+/-{cv.std()*ci_factor:.4f}"
        s_str = f"{sv.mean():.4f}+/-{sv.std()*ci_factor:.4f}"
        print(f"  R-Prec Top-{k}             {b_str:>15} {m_str:>15} {c_str:>15} {s_str:>15}")

    bm = np.array(all_matching_scores)
    sm = np.array(all_match_standard)
    print(f"  {'MM Dist':<25} {bm.mean():>10.4f}+/-{bm.std()*ci_factor:.4f}"
          f" {'':>15} {'':>15}"
          f" {sm.mean():>10.4f}+/-{sm.std()*ci_factor:.4f}")

    print(f"\n  FID:        {fid:.4f}")
    print(f"  Diversity:  {diversity:.4f}")
    print("=" * 80)


if __name__ == "__main__":
    main()
