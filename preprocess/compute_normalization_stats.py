"""
Compute normalization statistics (Mean and Std) for preprocessed motion data.

This script implements HumanML3D-style grouped normalization:
1. Compute per-dimension mean and std across all data
2. For each semantic group, replace per-dim std with the group's average std
3. Special handling for contact (no normalization) and betas (optional)

Supported formats:
- lower (71D): lower_pose_6d + local_transl_vel + foot_contact + betas
- upper_lower (149D): upper_pose_6d + lower_pose_6d + local_transl_vel + foot_contact + betas
- genmo (145D): body_pose_r6d + betas + global_orient_r6d + local_transl_vel

Usage:
    python compute_normalization_stats.py --format lower --data_dirs /path/to/data1 /path/to/data2
"""
import argparse
import json
import os
from datetime import datetime
from pathlib import Path

import numpy as np
from tqdm import tqdm


# =============================================================================
# Normalization Group Definitions
# =============================================================================

NORM_GROUPS = {
    'lower': {
        'total_dims': 71,
        'groups': {
            'lower_pose_6d': {
                'range': (0, 54),
                'dims': 54,
                'description': '9 lower joints × 6D rotation',
                'normalize': True,
            },
            'local_transl_vel': {
                'range': (54, 57),
                'dims': 3,
                'description': 'Local translation velocity',
                'normalize': True,
            },
            'foot_contact': {
                'range': (57, 61),
                'dims': 4,
                'description': 'Foot contact flags (0-1)',
                'normalize': False,  # Already 0-1, don't normalize
            },
            'betas': {
                'range': (61, 71),
                'dims': 10,
                'description': 'Shape parameters',
                'normalize': False,  # Fixed per sequence, don't normalize
            },
        },
    },
    'upper_lower': {
        'total_dims': 149,
        'groups': {
            'upper_pose_6d': {
                'range': (0, 78),
                'dims': 78,
                'description': '13 upper joints × 6D rotation',
                'normalize': True,
            },
            'lower_pose_6d': {
                'range': (78, 132),
                'dims': 54,
                'description': '9 lower joints × 6D rotation',
                'normalize': True,
            },
            'local_transl_vel': {
                'range': (132, 135),
                'dims': 3,
                'description': 'Local translation velocity',
                'normalize': True,
            },
            'foot_contact': {
                'range': (135, 139),
                'dims': 4,
                'description': 'Foot contact flags (0-1)',
                'normalize': False,
            },
            'betas': {
                'range': (139, 149),
                'dims': 10,
                'description': 'Shape parameters',
                'normalize': False,
            },
        },
    },
    'genmo': {
        'total_dims': 145,
        'groups': {
            'body_pose_r6d': {
                'range': (0, 126),
                'dims': 126,
                'description': '21 body joints × 6D rotation',
                'normalize': True,
            },
            'betas': {
                'range': (126, 136),
                'dims': 10,
                'description': 'Shape parameters',
                'normalize': False,
            },
            'global_orient_r6d': {
                'range': (136, 142),
                'dims': 6,
                'description': 'Global orientation 6D',
                'normalize': True,
            },
            'local_transl_vel': {
                'range': (142, 145),
                'dims': 3,
                'description': 'Local translation velocity',
                'normalize': True,
            },
        },
    },
}


# =============================================================================
# Data Loading
# =============================================================================

def load_all_motion_vectors(data_dirs, format_name, max_files=None):
    """
    Load all motion vectors from multiple directories.

    Args:
        data_dirs: List of directories containing .npz files
        max_files: Maximum total files to load (for testing)

    Returns:
        all_vectors: np.ndarray of shape (N_total_frames, D)
        stats: dict with loading statistics
    """
    all_vectors = []
    stats = {
        'total_files': 0,
        'total_frames': 0,
        'skipped_files': 0,
        'errors': [],
    }

    # Collect all npz files
    all_npz_files = []
    for data_dir in data_dirs:
        data_dir = Path(data_dir)
        if not data_dir.exists():
            print(f"Warning: Directory not found: {data_dir}")
            continue
        npz_files = list(data_dir.rglob('*.npz'))
        all_npz_files.extend(npz_files)
        print(f"Found {len(npz_files)} files in {data_dir}")

    if max_files is not None:
        all_npz_files = all_npz_files[:max_files]

    print(f"Total files to process: {len(all_npz_files)}")

    # Load motion vectors
    for npz_path in tqdm(all_npz_files, desc="Loading motion vectors"):
        try:
            data = np.load(npz_path, allow_pickle=True)
            motion = None
            if 'motion_vector' in data:
                motion = data['motion_vector']
            elif format_name == 'lower' and 'lower' in data:
                motion = data['lower']
            elif format_name == 'upper_lower' and 'upper' in data and 'lower' in data:
                motion = np.concatenate([data['upper'], data['lower']], axis=-1)

            if motion is None:
                stats['skipped_files'] += 1
                stats['errors'].append((str(npz_path), f"Missing data for format '{format_name}'"))
                continue
            all_vectors.append(motion)
            stats['total_files'] += 1
            stats['total_frames'] += motion.shape[0]

        except Exception as e:
            stats['skipped_files'] += 1
            stats['errors'].append((str(npz_path), str(e)))

    if not all_vectors:
        raise ValueError("No valid motion vectors found!")

    # Concatenate all vectors
    all_vectors = np.concatenate(all_vectors, axis=0)
    print(f"Loaded {stats['total_files']} files, {stats['total_frames']} frames")
    print(f"Motion vector shape: {all_vectors.shape}")

    return all_vectors, stats


# =============================================================================
# Normalization Computation
# =============================================================================

def compute_grouped_normalization(all_motion_vectors, format_name):
    """
    Compute grouped normalization statistics.

    Args:
        all_motion_vectors: np.ndarray of shape (N_frames, D)
        format_name: One of 'lower', 'upper_lower', 'genmo'

    Returns:
        Mean: np.ndarray of shape (D,)
        Std: np.ndarray of shape (D,)
        group_stats: dict with per-group statistics
    """
    if format_name not in NORM_GROUPS:
        raise ValueError(f"Unknown format: {format_name}. Expected one of {list(NORM_GROUPS.keys())}")

    config = NORM_GROUPS[format_name]
    D = config['total_dims']
    groups = config['groups']

    # Verify dimension
    if all_motion_vectors.shape[1] != D:
        raise ValueError(f"Expected {D} dimensions for format '{format_name}', got {all_motion_vectors.shape[1]}")

    # Step 1: Compute per-dimension mean and std
    Mean = all_motion_vectors.mean(axis=0)  # (D,)
    Std = all_motion_vectors.std(axis=0)    # (D,)

    print(f"\nRaw statistics:")
    print(f"  Mean range: [{Mean.min():.4f}, {Mean.max():.4f}]")
    print(f"  Std range: [{Std.min():.6f}, {Std.max():.4f}]")

    # Step 2: Apply grouped normalization
    group_stats = {}
    print(f"\nGroup-wise statistics for '{format_name}' format:")

    for group_name, group_info in groups.items():
        start, end = group_info['range']
        normalize = group_info['normalize']

        # Compute group statistics
        group_mean = Mean[start:end]
        group_std_raw = Std[start:end]
        group_std_avg = group_std_raw.mean()

        group_stats[group_name] = {
            'range': [start, end],
            'dims': group_info['dims'],
            'raw_std_min': float(group_std_raw.min()),
            'raw_std_max': float(group_std_raw.max()),
            'raw_std_mean': float(group_std_avg),
            'mean_min': float(group_mean.min()),
            'mean_max': float(group_mean.max()),
            'normalized': normalize,
        }

        if normalize:
            # Replace per-dim std with group average
            Std[start:end] = group_std_avg
            group_stats[group_name]['group_std'] = float(group_std_avg)
            print(f"  {group_name}: dims=[{start}:{end}], group_std={group_std_avg:.6f} (normalized)")
        else:
            # Don't normalize: set std=1.0, mean=0.0
            Std[start:end] = 1.0
            Mean[start:end] = 0.0
            group_stats[group_name]['group_std'] = 1.0
            print(f"  {group_name}: dims=[{start}:{end}], std=1.0 (not normalized)")

    # Step 3: Prevent division by zero
    min_std = 1e-8
    zero_std_count = (Std < min_std).sum()
    if zero_std_count > 0:
        print(f"\nWarning: {zero_std_count} dimensions have std < {min_std}, clipping to {min_std}")
    Std = np.clip(Std, a_min=min_std, a_max=None)

    return Mean, Std, group_stats


def verify_normalization(all_motion_vectors, Mean, Std, format_name):
    """Verify that normalization produces expected ranges."""
    normalized = (all_motion_vectors - Mean) / Std

    print(f"\nNormalized data statistics:")
    print(f"  Mean: {normalized.mean():.6f} (expected: ~0)")
    print(f"  Std: {normalized.std():.6f}")
    print(f"  Min: {normalized.min():.4f}")
    print(f"  Max: {normalized.max():.4f}")

    # Check each group
    config = NORM_GROUPS[format_name]
    for group_name, group_info in config['groups'].items():
        start, end = group_info['range']
        group_data = normalized[:, start:end]
        if group_info['normalize']:
            print(f"  {group_name}: mean={group_data.mean():.4f}, std={group_data.std():.4f}")


# =============================================================================
# Main
# =============================================================================

def parse_arguments():
    parser = argparse.ArgumentParser(description='Compute normalization statistics')
    parser.add_argument(
        "--format",
        type=str,
        required=True,
        choices=['lower', 'upper_lower', 'genmo'],
        help="Motion vector format",
    )
    parser.add_argument(
        "--data_dirs",
        type=str,
        nargs="+",
        required=True,
        help="Directories containing preprocessed .npz files",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for Mean.npy, Std.npy, and stats_info.json",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=25,
        help="Frame rate of the data",
    )
    parser.add_argument(
        "--max_files",
        type=int,
        default=None,
        help="Maximum files to load (for testing)",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify normalization by checking output statistics",
    )
    return parser.parse_args()


def main():
    args = parse_arguments()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load all motion vectors
    print(f"Loading motion vectors from: {args.data_dirs}")
    all_vectors, load_stats = load_all_motion_vectors(args.data_dirs, args.format, args.max_files)

    # Compute normalization statistics
    print(f"\nComputing normalization for format: {args.format}")
    Mean, Std, group_stats = compute_grouped_normalization(all_vectors, args.format)

    # Verify if requested
    if args.verify:
        verify_normalization(all_vectors, Mean, Std, args.format)

    # Save results
    mean_path = output_dir / 'Mean.npy'
    std_path = output_dir / 'Std.npy'
    info_path = output_dir / 'stats_info.json'

    np.save(mean_path, Mean.astype(np.float32))
    np.save(std_path, Std.astype(np.float32))

    # Save metadata
    stats_info = {
        'format': args.format,
        'total_dims': NORM_GROUPS[args.format]['total_dims'],
        'fps': args.fps,
        'total_frames': load_stats['total_frames'],
        'total_sequences': load_stats['total_files'],
        'data_dirs': [str(d) for d in args.data_dirs],
        'norm_groups': group_stats,
        'created_at': datetime.now().isoformat(),
    }

    with open(info_path, 'w') as f:
        json.dump(stats_info, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Saved normalization statistics:")
    print(f"  Mean: {mean_path}")
    print(f"  Std: {std_path}")
    print(f"  Info: {info_path}")

    # Print summary
    print(f"\nNormalization summary for '{args.format}':")
    for group_name, info in group_stats.items():
        norm_status = "normalized" if info['normalized'] else "not normalized"
        print(f"  {group_name}: [{info['range'][0]}:{info['range'][1]}] -> std={info['group_std']:.6f} ({norm_status})")


if __name__ == "__main__":
    main()
