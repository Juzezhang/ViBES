"""
Evaluation metrics for the Text-Motion CLIP evaluator.

Implements the standard text-to-motion evaluation protocol:
- R-Precision (top-1/2/3) with Euclidean distance, groups of R_SIZE, 20 replications
- Matching Score (avg Euclidean distance of correct pairs)
- FID (Frechet Inception Distance between test and train motion embeddings)
- Diversity (avg L2 distance between random pairs of motion embeddings)

References:
- text-to-motion (Guo et al.): evaluation protocol
- InterGen: CLIP evaluator architecture
"""

import numpy as np
import torch
import torch.nn.functional as F

from multimodal_tokenizers.metrics.utils import (
    euclidean_distance_matrix_np,
    calculate_activation_statistics_np,
    calculate_frechet_distance_np,
    calculate_diversity_np,
)


def calculate_top_k_np(mat, top_k):
    """
    Calculate top-k accuracy from a sorted index matrix.

    Args:
        mat: (N, N) matrix of sorted indices (from np.argsort on distance matrix)
        top_k: Maximum k to compute

    Returns:
        top_k_mat: (N, top_k) boolean matrix, where entry [i, k] is True if
                   the correct match for sample i appears in the top-(k+1) predictions
    """
    size = mat.shape[0]
    gt_mat = np.expand_dims(np.arange(size), 1).repeat(size, axis=1)
    bool_mat = mat == gt_mat
    correct_vec = False
    top_k_list = []
    for i in range(top_k):
        correct_vec = correct_vec | bool_mat[:, i]
        top_k_list.append(correct_vec[:, None])
    top_k_mat = np.concatenate(top_k_list, axis=1)
    return top_k_mat


@torch.no_grad()
def extract_embeddings(model, dataloader, device, normalize_embeddings=False):
    """
    Extract motion and text embeddings from a dataloader.

    Args:
        model: TextMotionCLIP model
        dataloader: DataLoader yielding batches with 'text', 'motion', 'length' keys
        device: torch device
        normalize_embeddings: If True, use forward() + unit-normalize (decouples from
            learnable latent_scale, making FID/Diversity comparable across training
            epochs). If False, use encode_motion/encode_text (includes latent_scale,
            matching InterGen's evaluation for final checkpoint comparison).

    Returns:
        motion_embs: numpy array (N, embed_dim)
        text_embs: numpy array (N, embed_dim)
    """
    model.eval()
    motion_embs_list = []
    text_embs_list = []

    for batch in dataloader:
        text = batch["text"]
        motion = batch["motion"].to(device)
        lengths = batch["length"].to(device)

        if normalize_embeddings:
            # Unit-normalize raw encoder outputs. Decouples from the learnable
            # latent_scale so FID/Diversity stay comparable across training epochs.
            B, T, _ = motion.shape
            mask = model.generate_src_mask(T, lengths)
            motion_emb, text_emb = model(text, motion, mask)
            motion_emb = F.normalize(motion_emb, dim=-1)
            text_emb = F.normalize(text_emb, dim=-1)
        else:
            # Use encode_motion/encode_text which include L2-norm * latent_scale,
            # matching InterGen's evaluation approach for final checkpoint eval.
            motion_emb = model.encode_motion(motion, lengths)
            text_emb = model.encode_text(text, device)

        motion_embs_list.append(motion_emb.cpu().numpy())
        text_embs_list.append(text_emb.cpu().numpy())

    motion_embs = np.concatenate(motion_embs_list, axis=0)
    text_embs = np.concatenate(text_embs_list, axis=0)
    return motion_embs, text_embs


def compute_r_precision_and_matching_score(text_embs, motion_embs, top_k=3, R_size=32):
    """
    Compute R-Precision and Matching Score for ONE replication.

    Standard protocol: split N samples into groups of R_size, compute Euclidean
    distance matrix per group, check if correct match is in top-k.

    Args:
        text_embs: (N, D) text embeddings
        motion_embs: (N, D) motion embeddings
        top_k: Maximum k for R-Precision
        R_size: Group size (typically 32)

    Returns:
        r_precision: dict with keys 'top_1', 'top_2', 'top_3' (float values)
        matching_score: float, average Euclidean distance of correct pairs
    """
    N = text_embs.shape[0]
    n_groups = N // R_size

    matching_score_sum = 0.0
    top_k_counts = np.zeros(top_k)
    total_samples = 0

    for i in range(n_groups):
        group_text = text_embs[i * R_size : (i + 1) * R_size]
        group_motion = motion_embs[i * R_size : (i + 1) * R_size]

        # Euclidean distance matrix: (R_size, R_size)
        dist_mat = euclidean_distance_matrix_np(group_text, group_motion)

        # Matching Score: average diagonal (correct pair distances)
        matching_score_sum += np.trace(dist_mat)

        # R-Precision: sort by distance, check if correct match is in top-k
        argsort_mat = np.argsort(dist_mat, axis=1)
        top_k_mat = calculate_top_k_np(argsort_mat, top_k)
        top_k_counts += top_k_mat.sum(axis=0)

        total_samples += R_size

    if total_samples == 0:
        return {f"top_{k+1}": 0.0 for k in range(top_k)}, 0.0

    r_precision = {}
    for k in range(top_k):
        r_precision[f"top_{k+1}"] = top_k_counts[k] / total_samples

    matching_score = matching_score_sum / total_samples

    return r_precision, matching_score


def evaluate_clip(
    model,
    test_dataloader,
    train_dataloader,
    device,
    replication_times=20,
    R_size=32,
    diversity_times=300,
    emb_scale=1.0,
    diversity_pair_divisor=1.0,
    normalize_embeddings=False,
):
    """
    Full evaluation entry point for Text-Motion CLIP.

    Runs the standard text-to-motion evaluation protocol:
    1. Extract test embeddings (once)
    2. Loop replications: shuffle -> R-Precision + Matching Score
    3. Extract train embeddings (once, for FID reference)
    4. Compute FID between test and train motion embeddings
    5. Compute Diversity on test motion embeddings
    6. Return results with means and 95% confidence intervals

    Args:
        model: TextMotionCLIP model
        test_dataloader: DataLoader for test split
        train_dataloader: DataLoader for train split
        device: torch device
        replication_times: Number of replications (default 20)
        R_size: Group size for R-Precision (default 32)
        diversity_times: Number of random pairs for Diversity (default 300)
        emb_scale: Scale factor applied before FID/Diversity statistics
        diversity_pair_divisor: Divisor applied to pair differences for Diversity
        normalize_embeddings: If True, use unit-normalized embeddings (for training
            monitoring). If False, use encode_motion/encode_text with latent_scale
            (for final eval, matching InterGen).

    Returns:
        dict with keys like 'R_precision_top_1', 'R_precision_top_1_conf',
        'Matching_score', 'Matching_score_conf', 'FID', 'Diversity'
    """
    model.eval()

    # 1. Extract test embeddings
    test_motion_embs, test_text_embs = extract_embeddings(
        model, test_dataloader, device, normalize_embeddings=normalize_embeddings
    )

    # 2. Replicated R-Precision and Matching Score
    all_r_precision = {f"top_{k}": [] for k in [1, 2, 3]}
    all_matching_scores = []

    for _ in range(replication_times):
        # Shuffle with consistent pairing
        indices = np.random.permutation(len(test_motion_embs))
        shuffled_motion = test_motion_embs[indices]
        shuffled_text = test_text_embs[indices]

        r_prec, match_score = compute_r_precision_and_matching_score(
            shuffled_text, shuffled_motion, top_k=3, R_size=R_size
        )
        for k in [1, 2, 3]:
            all_r_precision[f"top_{k}"].append(r_prec[f"top_{k}"])
        all_matching_scores.append(match_score)

    # 3. Extract train embeddings for FID
    train_motion_embs, _ = extract_embeddings(
        model, train_dataloader, device, normalize_embeddings=normalize_embeddings
    )

    # 4. FID
    test_stats = calculate_activation_statistics_np(
        test_motion_embs, emb_scale=emb_scale
    )
    train_stats = calculate_activation_statistics_np(
        train_motion_embs, emb_scale=emb_scale
    )
    fid = calculate_frechet_distance_np(
        test_stats[0], test_stats[1], train_stats[0], train_stats[1]
    )

    # 5. Diversity
    if len(test_motion_embs) > diversity_times:
        diversity = calculate_diversity_np(
            test_motion_embs,
            diversity_times,
            emb_scale=emb_scale,
            pair_divisor=diversity_pair_divisor,
        )
    else:
        diversity = calculate_diversity_np(
            test_motion_embs,
            len(test_motion_embs) - 1,
            emb_scale=emb_scale,
            pair_divisor=diversity_pair_divisor,
        )

    # 6. Compile results with means and 95% CI
    results = {}
    n_rep = replication_times
    ci_factor = 1.96 / np.sqrt(n_rep) if n_rep > 1 else 0.0

    for k in [1, 2, 3]:
        vals = np.array(all_r_precision[f"top_{k}"])
        results[f"R_precision_top_{k}"] = vals.mean()
        results[f"R_precision_top_{k}_conf"] = vals.std() * ci_factor

    match_vals = np.array(all_matching_scores)
    results["Matching_score"] = match_vals.mean()
    results["Matching_score_conf"] = match_vals.std() * ci_factor

    results["FID"] = float(fid)
    results["Diversity"] = float(diversity)

    return results
