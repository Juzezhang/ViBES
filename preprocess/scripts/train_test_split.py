import os
import random
from pathlib import Path
import argparse

# Paths
VIDEO_DIR = '/simurgh/u/juze/datasets/CANDOR/original'
PROCESSED_DIR = '/simurgh/u/juze/datasets/CANDOR/FLAME_coeffs'
SAVE_DIR = '/simurgh/u/juze/datasets/CANDOR'
# Output filenames
SPLITS = ['train', 'val', 'test']
SPLIT_RATIOS = [0.95, 0.025, 0.025]

# Set random seed for reproducibility
RANDOM_SEED = 42
random.seed(RANDOM_SEED)

def get_video_names_from_dir(directory, exts):
    video_names = set()
    for ext in exts:
        for p in Path(directory).rglob(f'*{ext}'):
            video_names.add(p.stem)
    return video_names

def get_sequence_names_from_subdirs(directory):
    # Each subdirectory is a sequence
    return set([p.name for p in Path(directory).iterdir() if p.is_dir()])

def get_processed_names_from_dir(directory):
    return set([p.stem for p in Path(directory).rglob('*.npz')])

def get_processed_names_from_dir_candor(directory):
    return set([p.stem for p in Path(directory).iterdir()])

def split_list(names, ratios):
    names = list(names)
    random.shuffle(names)
    n = len(names)
    n_train = int(n * ratios[0])
    n_val = int(n * ratios[1])
    n_test = n - n_train - n_val
    train = names[:n_train]
    val = names[n_train:n_train+n_val]
    test = names[n_train+n_val:]
    return train, val, test

def write_split(names, filename):
    with open(filename, 'w') as f:
        for name in names:
            f.write(f'{name}\n')

def main():
    parser = argparse.ArgumentParser(description='Train/val/test split for YouTube or Candor datasets.')
    parser.add_argument('--mode', type=str, choices=['youtube', 'candor'], default='candor',
                        help="Dataset mode: 'youtube' (video files) or 'candor' (each subdir is a sequence)")
    parser.add_argument('--video_dir', type=str, default=VIDEO_DIR, help='Path to video directory or sequence root')
    parser.add_argument('--processed_dir', type=str, default=PROCESSED_DIR, help='Path to processed .npz directory')
    parser.add_argument('--save_dir', type=str, default=SAVE_DIR, help='Where to save split txt files')
    args = parser.parse_args()

    video_dir = args.video_dir
    processed_dir = args.processed_dir
    save_dir = args.save_dir
    mode = args.mode

    print(f'Using mode: {mode}')
    if mode == 'youtube':
        video_exts = ['.mp4', '.avi', '.mov', '.mkv']
        all_video_names = get_video_names_from_dir(video_dir, video_exts)
        print(f'Total video files in {video_dir}: {len(all_video_names)}')
        processed_names = get_processed_names_from_dir(processed_dir)
        print(f'Processed videos in {processed_dir}: {len(processed_names)}')
    else:
        all_video_names = get_sequence_names_from_subdirs(video_dir)
        print(f'Total sequence folders in {video_dir}: {len(all_video_names)}')
        processed_names = get_processed_names_from_dir_candor(processed_dir)
        print(f'Processed videos in {processed_dir}: {len(processed_names)}')


    unprocessed_names = all_video_names - processed_names
    print(f'Unprocessed videos: {len(unprocessed_names)}')

    # Split processed
    train_p, val_p, test_p = split_list(processed_names, SPLIT_RATIOS)
    write_split(train_p, os.path.join(save_dir, 'train_processed.txt'))
    write_split(val_p, os.path.join(save_dir, 'val_processed.txt'))
    write_split(test_p, os.path.join(save_dir, 'test_processed.txt'))
    print(f'Processed split: train={len(train_p)}, val={len(val_p)}, test={len(test_p)}')

    # Split unprocessed
    train_u, val_u, test_u = split_list(unprocessed_names, SPLIT_RATIOS)
    write_split(train_u, os.path.join(save_dir, 'train_unprocessed.txt'))
    write_split(val_u, os.path.join(save_dir, 'val_unprocessed.txt'))
    write_split(test_u, os.path.join(save_dir, 'test_unprocessed.txt'))
    print(f'Unprocessed split: train={len(train_u)}, val={len(val_u)}, test={len(test_u)}')

if __name__ == '__main__':
    main() 