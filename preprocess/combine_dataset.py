import os
from datasets import load_from_disk, concatenate_datasets
import argparse
from typing import List

def load_dataset_paths_from_file(file_path: str) -> List[str]:
    """Load dataset paths from a text file, one path per line."""
    with open(file_path, 'r') as f:
        paths = [line.strip() for line in f if line.strip()]
    return paths

def validate_dataset_paths(dataset_paths: List[str]) -> List[str]:
    """Validate that all dataset paths exist."""
    valid_paths = []
    for path in dataset_paths:
        if os.path.exists(path):
            valid_paths.append(path)
            print(f"✓ Found dataset: {path}")
        else:
            print(f"✗ Warning: Dataset path does not exist: {path}")

    if not valid_paths:
        raise ValueError("No valid dataset paths found!")

    return valid_paths

def main():
    parser = argparse.ArgumentParser(description='Combine multiple HuggingFace datasets')

    # Support multiple input methods
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--datasets', nargs='+',
                      help='Paths to HuggingFace datasets (space separated)')
    group.add_argument('--dataset_list', type=str,
                      help='Text file containing dataset paths (one per line)')

    parser.add_argument('--output_path', type=str, required=True,
                       help='Path to save the combined dataset')
    parser.add_argument('--validate', action='store_true',
                       help='Validate dataset paths before processing')

    args = parser.parse_args()

    # Get dataset paths from either command line or file
    if args.datasets:
        dataset_paths = args.datasets
        print(f"Using {len(dataset_paths)} datasets from command line")
    else:
        dataset_paths = load_dataset_paths_from_file(args.dataset_list)
        print(f"Loaded {len(dataset_paths)} datasets from file: {args.dataset_list}")

    # Validate paths if requested
    if args.validate:
        dataset_paths = validate_dataset_paths(dataset_paths)

    print(f"\nProcessing {len(dataset_paths)} datasets...")

    # Load all datasets
    datasets = []
    total_samples = 0

    for i, path in enumerate(dataset_paths, 1):
        try:
            print(f"Loading dataset {i}/{len(dataset_paths)}: {path}")
            ds = load_from_disk(path)
            datasets.append(ds)
            samples = len(ds)
            total_samples += samples
            print(f"  → Loaded {samples:,} samples")
        except Exception as e:
            print(f"  ✗ Error loading dataset {path}: {e}")
            continue

    if not datasets:
        raise ValueError("No datasets were successfully loaded!")

    print(f"\nCombining {len(datasets)} datasets with {total_samples:,} total samples...")

    # Concatenate all datasets
    combined = concatenate_datasets(datasets)

    print(f"Combined dataset contains {len(combined):,} samples")

    # Save to output path
    os.makedirs(args.output_path, exist_ok=True)
    print(f"Saving combined dataset to: {args.output_path}")
    combined.save_to_disk(args.output_path)

    print(f"✓ Successfully combined {len(datasets)} datasets with {len(combined):,} samples")
    print(f"✓ Saved to: {args.output_path}")

if __name__ == '__main__':
    main()