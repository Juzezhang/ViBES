"""
GT evaluation for Question2Motion CLIP evaluator.

Computes baseline metrics on test data using a trained QuestionMotionCLIP checkpoint:
  - R-Precision (top-1/2/3) with 20 replications and 95% CI
  - MM Dist (matching score)
  - FID (test vs test, should be ~0 for GT baseline)
  - Diversity

Following the standard protocol from InterGen / text-to-motion:
  FID compares GT test motion embeddings against themselves (≈ 0),
  establishing the reference floor for generated motion evaluation.

Usage:
    CUDA_VISIBLE_DEVICES=0 python scripts/eval_question_motion_gt.py \
        --checkpoint /path/to/experiments/text_motion_clip/QuestionMotionCLIP/checkpoints/best-R3-epoch=59.ckpt \
        --data-root /path/to/Converse3D_eval
"""

import os
import sys
import argparse
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
)
from multimodal_tokenizers.metrics.utils import (
    calculate_activation_statistics_np,
    calculate_frechet_distance_np,
    calculate_diversity_np,
)


def load_model(cfg, checkpoint_path, device):
    model = TextMotionCLIP(cfg)
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = ckpt.get("state_dict", ckpt)
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()
    return model


def main():
    parser = argparse.ArgumentParser(description="GT evaluation for QuestionMotionCLIP")
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="Path to CLIP checkpoint"
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default="/path/to/Converse3D_eval",
        help="Converse3D data root",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=os.path.join(ROOT_DIR, "configs/evaluator/question_motion_clip.yaml"),
        help="Config YAML path",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
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

    # Build test dataloader
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
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    # Eval parameters from config
    eval_cfg = cfg.EVAL_METRICS
    replication_times = eval_cfg.REPLICATION_TIMES  # 20
    R_size = eval_cfg.R_SIZE  # 32
    diversity_times = eval_cfg.DIVERSITY_TIMES  # 300
    emb_scale = eval_cfg.EMB_SCALE  # 6.0
    diversity_pair_divisor = eval_cfg.DIVERSITY_PAIR_DIVISOR  # 2.0

    # 1. Extract test embeddings (once, reused for all metrics)
    print("Extracting test embeddings...")
    test_motion_embs, test_text_embs = extract_embeddings(
        model, test_loader, device, normalize_embeddings=False
    )
    print(f"  Test: {test_motion_embs.shape[0]} samples, embed_dim={test_motion_embs.shape[1]}")

    # 2. R-Precision + MM Dist (20 replications)
    print(f"Computing R-Precision and MM Dist ({replication_times} replications)...")
    all_r_precision = {f"top_{k}": [] for k in [1, 2, 3]}
    all_matching_scores = []

    for rep in range(replication_times):
        indices = np.random.permutation(len(test_motion_embs))
        shuffled_motion = test_motion_embs[indices]
        shuffled_text = test_text_embs[indices]

        r_prec, match_score = compute_r_precision_and_matching_score(
            shuffled_text, shuffled_motion, top_k=3, R_size=R_size
        )
        for k in [1, 2, 3]:
            all_r_precision[f"top_{k}"].append(r_prec[f"top_{k}"])
        all_matching_scores.append(match_score)

    ci_factor = 1.96 / np.sqrt(replication_times)

    r_precision_results = {}
    for k in [1, 2, 3]:
        vals = np.array(all_r_precision[f"top_{k}"])
        r_precision_results[f"R_top_{k}"] = vals.mean()
        r_precision_results[f"R_top_{k}_ci"] = vals.std() * ci_factor

    match_vals = np.array(all_matching_scores)
    mm_dist = match_vals.mean()
    mm_dist_ci = match_vals.std() * ci_factor

    # 3. FID (GT baseline: test vs test → ≈ 0)
    # Standard protocol (InterGen / text-to-motion): GT FID compares test motion
    # embeddings against themselves. When evaluating a generation model later,
    # FID = frechet_distance(generated_embs, gt_test_embs).
    print("Computing FID (GT: test vs test)...")
    n_test = len(test_motion_embs)
    test_stats = calculate_activation_statistics_np(test_motion_embs, emb_scale=emb_scale)
    fid = calculate_frechet_distance_np(
        test_stats[0], test_stats[1], test_stats[0], test_stats[1]
    )

    # 4. Diversity
    print("Computing Diversity...")
    n_pairs = min(diversity_times, len(test_motion_embs) - 1)
    diversity = calculate_diversity_np(
        test_motion_embs, n_pairs, emb_scale=emb_scale, pair_divisor=diversity_pair_divisor
    )

    # Print results
    print("\n" + "=" * 60)
    print("  QuestionMotionCLIP — GT Evaluation Results")
    print("=" * 60)
    print(f"  Checkpoint: {os.path.basename(args.checkpoint)}")
    print(f"  Test samples: {n_test}")
    print(f"  Replications: {replication_times}, R_size: {R_size}")
    print(f"  emb_scale: {emb_scale}, diversity_pair_divisor: {diversity_pair_divisor}")
    print("-" * 60)
    for k in [1, 2, 3]:
        mean = r_precision_results[f"R_top_{k}"]
        ci = r_precision_results[f"R_top_{k}_ci"]
        print(f"  R-Precision Top-{k}:   {mean:.4f} +/- {ci:.4f}")
    print(f"  MM Dist:              {mm_dist:.4f} +/- {mm_dist_ci:.4f}")
    print(f"  FID:                  {fid:.4f}")
    print(f"  Diversity:            {diversity:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
