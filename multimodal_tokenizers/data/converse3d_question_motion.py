"""
Converse3D Question-Motion Dataset and DataModule for CLIP evaluator training.

Loads paired questions and SMPLX motion sequences from Converse3D_eval,
converting axis-angle rotations to 6D representation for a 333-dim
per-frame feature vector (330 rotation 6D + 3 root velocity).
"""

import json
import os
import random
import warnings

import numpy as np
import torch
from torch.utils.data import Dataset

from multimodal_tokenizers.data import BASEDataModule
from multimodal_tokenizers.utils.rotation_conversions import (
    axis_angle_to_6d_np,
    axis_angle_to_matrix,
)


ALLOW_FRAME_MISMATCH = 2
SMPLX_TOTAL_JOINTS = 55
SMPLX_BODY_JOINTS = 22  # joints 0-21 (no face/hands)


def get_local_transl_vel(transl, global_orient):
    """Compute local translation velocity in root-relative coordinates.

    Args:
        transl: (T, 3) world translation
        global_orient: (T, 3) axis-angle global orientation

    Returns:
        (T, 3) local translation velocity as numpy array
    """
    transl = torch.as_tensor(transl, dtype=torch.float32)
    global_orient = torch.as_tensor(global_orient, dtype=torch.float32)

    if len(transl) == 0:
        return torch.zeros(0, 3).numpy()

    velocity = torch.zeros_like(transl)
    if len(transl) > 1:
        velocity[1:] = transl[1:] - transl[:-1]
        velocity[0] = velocity[1]

    R = axis_angle_to_matrix(global_orient)
    R_inv = R.transpose(-1, -2)
    local_vel = torch.einsum('lij,lj->li', R_inv, velocity)
    return local_vel.numpy()


class Converse3DQuestionMotionDataset(Dataset):
    """
    Dataset that loads paired question + motion from Converse3D_eval.

    Each sample consists of:
        - A question string from the JSONL split file
        - Motion features: 6D rotations + root velocity (3) per frame
        - Sequence length (before padding)

    Args:
        data_root: Root directory containing JSONL splits and motion_segments/
        motion_dir: Subdirectory name for .npz motion files (e.g. 'motion_segments')
        text_dir: Unused, kept for interface compatibility with AmassTextMotionDataset
        split: One of 'train', 'val', 'test'
        max_length: Maximum sequence length (pad/crop to this)
        is_train: Whether to use random cropping (default: True)
        num_joints: Number of joints to use (22=body only, 55=full SMPLX)
    """

    def __init__(
        self, data_root, motion_dir, text_dir, split, max_length=196,
        is_train=None, num_joints=SMPLX_TOTAL_JOINTS,
    ):
        super().__init__()
        self.data_root = data_root
        self.motion_path = os.path.join(data_root, motion_dir)
        self.max_length = max_length
        self.num_joints = num_joints
        self.feature_dim = num_joints * 6 + 3  # 6D rot per joint + 3 root vel
        self.is_train = True if is_train is None else bool(is_train)
        self._warned_fallback = False

        # Load JSONL split file
        jsonl_file = os.path.join(data_root, f"{split}.jsonl")
        records = []
        with open(jsonl_file, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))

        # Filter to records with non-null question and existing motion file
        self.samples = []
        for rec in records:
            if not rec.get("question"):
                continue
            segment_id = rec["segment_id"]
            motion_file = os.path.join(self.motion_path, split, f"{segment_id}.npz")
            if os.path.exists(motion_file):
                self.samples.append({
                    "segment_id": segment_id,
                    "question": rec["question"],
                    "motion_file": motion_file,
                })

        print(
            f"[Converse3DQuestionMotionDataset] split={split}: "
            f"{len(self.samples)}/{len(records)} valid pairs "
            f"(with both motion + question)"
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # --- Load motion ---
        data = np.load(sample["motion_file"], allow_pickle=True)

        if "poses" in data:
            rotations = data["poses"]  # (T, 165)
        else:
            if not self._warned_fallback:
                warnings.warn(
                    f"[Converse3DQuestionMotionDataset] 'poses' field missing in "
                    f"{sample['segment_id']}.npz, falling back to individual "
                    "rotation fields.",
                    stacklevel=2,
                )
                self._warned_fallback = True
            rotations = np.concatenate(
                [data["root_orient"], data["pose_body"],
                 data["pose_hand"], data["pose_jaw"],
                 data["pose_eye"]], axis=-1,
            )

        trans = data["trans"]  # (T_trans, 3)
        T_rot, T_trans = rotations.shape[0], trans.shape[0]

        if abs(T_rot - T_trans) > ALLOW_FRAME_MISMATCH:
            raise ValueError(
                f"Sample {sample['segment_id']}: frame mismatch "
                f"T_rot={T_rot} T_trans={T_trans}"
            )
        T = min(T_rot, T_trans)
        rotations = rotations[:T]
        trans = trans[:T]

        # Convert axis-angle to 6D rotation: (T, 165) -> (T, 55, 3) -> 6D
        rot_reshaped = rotations.reshape(T, 55, 3)
        rot_6d = axis_angle_to_6d_np(rot_reshaped)  # (T, 55, 6)

        # Select joints (22 body-only or all 55)
        rot_6d = rot_6d[:, :self.num_joints, :]  # (T, num_joints, 6)
        rot_6d = rot_6d.reshape(T, self.num_joints * 6)

        # Root velocity: world translation → local root-relative velocity (T, 3)
        global_orient = rotations[:, :3]  # first 3 dims = global orient axis-angle
        local_vel = get_local_transl_vel(trans, global_orient)  # (T, 3)

        # Concat: (T, feature_dim)
        motion = np.concatenate(
            [rot_6d, local_vel], axis=-1
        ).astype(np.float32)

        # Crop or pad to max_length
        seq_len = min(T, self.max_length)
        if T > self.max_length:
            if self.is_train:
                start = random.randint(0, T - self.max_length)
            else:
                start = 0
            motion = motion[start : start + self.max_length]
        elif T < self.max_length:
            pad = np.zeros((self.max_length - T, self.feature_dim), dtype=np.float32)
            motion = np.concatenate([motion, pad], axis=0)

        return {
            "text": sample["question"],
            "motion": torch.from_numpy(motion),  # (max_length, feature_dim)
            "length": seq_len,
        }


def collate_fn(batch):
    """Collate function: stack motions and lengths, keep texts as list."""
    texts = [b["text"] for b in batch]
    motions = torch.stack([b["motion"] for b in batch], dim=0)  # (B, T, 333)
    lengths = torch.tensor([b["length"] for b in batch], dtype=torch.long)  # (B,)
    return {"text": texts, "motion": motions, "length": lengths}


class Converse3DQuestionMotionDataModule(BASEDataModule):
    """
    PyTorch Lightning DataModule for Converse3D question-motion pairs.

    Args:
        cfg: OmegaConf config object
        phase: Training phase ('train', 'val', 'test')
    """

    def __init__(self, cfg, phase="train", **kwargs):
        super().__init__(collate_fn=collate_fn)
        self.cfg = cfg
        self.phase = phase

        self.data_root = cfg.DATASET.data_root
        self.motion_dir = cfg.DATASET.motion_dir
        self.text_dir = cfg.DATASET.text_dir
        self.max_length = cfg.DATASET.max_length
        self.num_joints = getattr(cfg.DATASET, "num_joints", SMPLX_TOTAL_JOINTS)

    def _make_dataset(self, split):
        eval_deterministic = getattr(self.cfg.DATASET, "EVAL_DETERMINISTIC", False)
        use_train_sampling = (split == self.cfg.TRAIN.SPLIT) or (not eval_deterministic)
        return Converse3DQuestionMotionDataset(
            data_root=self.data_root,
            motion_dir=self.motion_dir,
            text_dir=self.text_dir,
            split=split,
            max_length=self.max_length,
            is_train=use_train_sampling,
            num_joints=self.num_joints,
        )

    @property
    def train_dataset(self):
        if self._train_dataset is None:
            self._train_dataset = self._make_dataset(self.cfg.TRAIN.SPLIT)
        return self._train_dataset

    @property
    def val_dataset(self):
        if self._val_dataset is None:
            self._val_dataset = self._make_dataset(self.cfg.EVAL.SPLIT)
        return self._val_dataset

    @property
    def test_dataset(self):
        if self._test_dataset is None:
            self._test_dataset = self._make_dataset(self.cfg.TEST.SPLIT)
        return self._test_dataset

    def setup(self, stage=None):
        if stage in (None, "fit"):
            _ = self.train_dataset
            _ = self.val_dataset
        if stage in (None, "test"):
            _ = self.test_dataset
