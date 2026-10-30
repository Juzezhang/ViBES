#!/usr/bin/env python3
"""
Evaluate ViBES co-speech gesture generation on the BEAT2 speaker-2 benchmark (EMAGE protocol).

For every test conversation the motion tokens are masked out, the model regenerates them from
the conversation's text + audio (face, upper, lower and hand tokens, decoded with the EMAGE
speaker-2 VQ-VAEs at 30 fps), and the result is scored with the EMAGE metrics:

  FGD        Frechet distance between AESKConv latents of generated and GT 55-joint 6D poses
  BC         beat consistency between audio onsets and upper-body velocity minima
  Diversity  L1 diversity of the generated joints

In the paper's table (EMAGE convention, "FGD x10^-1 / BC x10^-1") FGD is the raw value and BC is
the raw value times 10; `paper_scale` prints the numbers in that form.

Usage:
    python -m evaluation.eval_beat2_cospeech \
        --checkpoint /path/to/beat2_speaker2_checkpoint \
        --vq_checkpoint /path/to/vq_emage_speaker_2.ckpt \
        --test_dataset /path/to/processed_beat2_dataset_tokenized_mot_spearker2_test/tokenized_dataset \
        --beat2_root /path/to/BEAT2/beat_english_v2.0.0 \
        --output_dir eval_results/beat2_cospeech
"""
import os
import sys

_script_dir = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(_script_dir, ".."))
if ROOT_DIR in sys.path:
    sys.path.remove(ROOT_DIR)
sys.path.insert(0, ROOT_DIR)

import argparse
import json
import re

import librosa
import numpy as np
import smplx
import torch
from datasets import load_from_disk
from omegaconf import OmegaConf
from tqdm import tqdm
from transformers import AutoConfig, AutoModel
from transformers.modeling_utils import load_sharded_checkpoint

sys.path.insert(0, os.path.join(ROOT_DIR, "training"))
from expert_io import is_expert1_checkpoint, load_expert1_checkpoint

from multimodal_tokenizers.archs.lom_vq import VQVAEConvZero
from multimodal_tokenizers.data.mixed_dataset.data_tools import (
    JOINT_MASK_UPPER, JOINT_MASK_LOWER, JOINT_MASK_HANDS,
)
from multimodal_tokenizers.metrics.co_speech import CoSpeechMetrics
from multimodal_tokenizers.utils.rotation_conversions import (
    axis_angle_to_6d, axis_angle_to_6d_np, rotation_6d_to_axis_angle,
    rotation_6d_to_matrix, matrix_to_axis_angle,
)
from utils.tensor_utils import inverse_selection_tensor

TOKEN_OFFSET = 168736       # first motion-token id in the text vocabulary
BOM_RELATIVE_ID = 1024      # begin_of_motion in the EMAGE-VQ (256 codes per part) vocabulary
POSE_FPS = 30
AUDIO_FPS = 16000


# ============================================================================
# Loading
# ============================================================================

def load_model(checkpoint, glm_base_path, device):
    """Build the model with the repo code and the checkpoint's own layer / vocab settings."""
    config = AutoConfig.from_pretrained(os.path.join(ROOT_DIR, "vibes"), trust_remote_code=True)
    ckpt_config = os.path.join(checkpoint, "config.json")
    if os.path.exists(ckpt_config):
        saved = json.load(open(ckpt_config))
        for key in ("num_layers", "t1_layer_limit", "mot_settings"):
            if key in saved:
                setattr(config, key, saved[key])
    model = AutoModel.from_config(
        config, trust_remote_code=True, torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    ).to(device)
    if is_expert1_checkpoint(checkpoint):
        load_expert1_checkpoint(model, checkpoint, glm_base_path)
    else:
        load_sharded_checkpoint(model, checkpoint)
    return model.eval()


def load_motion_vocab(checkpoint):
    """Map relative motion-token ids to (part, code) from the checkpoint's added tokens."""
    # The speaker-2 model uses the EMAGE-VQ vocabulary (256 codes per part), not the one in vibes/
    path = os.path.join(checkpoint, "added_tokens.json")
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found; the checkpoint must ship its own motion vocabulary")
    vocab = {}
    for name, idx in json.load(open(path)).items():
        m = re.fullmatch(r"<\|(face|upper|lower|hand)_(\d+)\|>", name)
        if m:
            vocab[idx - TOKEN_OFFSET] = (m.group(1), int(m.group(2)))
    return vocab


def load_emage_vqs(vq_checkpoint, device):
    """EMAGE speaker-2 VQ-VAEs: face (jaw 6D + 100 expressions), upper, lower, hand."""
    state_dict = torch.load(vq_checkpoint, map_location="cpu", weights_only=False)["state_dict"]
    vqs = {}
    for part, layers, dim in (("face", 2, 106), ("upper", 2, 78), ("lower", 4, 61), ("hand", 2, 180)):
        vq = VQVAEConvZero(vae_layer=layers, code_num=256, codebook_size=256,
                           vae_quantizer_lambda=1, vae_test_dim=dim)
        vq.load_state_dict({k.replace(f"vae_{part}.", ""): v for k, v in state_dict.items()
                            if k.startswith(f"vae_{part}.")}, strict=True)
        vqs[part] = vq.eval().to(device)
    return vqs


def build_metric_cfg(beat2_root):
    assets = OmegaConf.load(os.path.join(ROOT_DIR, "configs", "assets.yaml"))
    cfg = OmegaConf.create({
        "DATASET": {"BEAT2": {"ROOT": beat2_root}, "MODEL_FILES_DIR": os.path.join(ROOT_DIR, "model_files"),
                    "audio_fps": AUDIO_FPS, "pose_fps": POSE_FPS},
        "METRIC": {"CO_SPEECH": OmegaConf.to_container(assets.METRIC.CO_SPEECH), "DIST_SYNC_ON_STEP": True},
        "vq": {"emage": {"params": {"vae_test_len": assets.METRIC.CO_SPEECH.params.vae_test_len}}},
    })
    return cfg


# ============================================================================
# Generation + decoding
# ============================================================================

@torch.no_grad()
def generate_motion_tokens(model, record, vocab, device, temperature, top_p):
    """Regenerate every motion token of a test conversation from its text + audio."""
    input_ids = torch.tensor(record["input_ids"], device=device).unsqueeze(0)
    input_ids[input_ids >= TOKEN_OFFSET] -= TOKEN_OFFSET
    modality_masks = torch.tensor([[record["modality_masks_0"]], [record["modality_masks_1"]],
                                   [record["modality_masks_2"]]], dtype=torch.bool, device=device)
    position_encoding_indices = torch.tensor(record["position_encoding_indices"], device=device).unsqueeze(0)
    attention_mask = torch.tensor(record["attention_mask"], device=device).unsqueeze(0)
    input_ids[:, modality_masks[2, 0] & (input_ids[0] != BOM_RELATIVE_ID)] = 0

    gen_kwargs = dict(input_ids=input_ids, attention_mask=attention_mask, max_new_tokens=1024,
                      do_sample=True, temperature=temperature, top_p=top_p,
                      modality_masks=modality_masks, use_cache=True,
                      position_encoding_indices=position_encoding_indices)
    past_key_values = model.generate_first_expert_cache(**gen_kwargs)
    output_ids, output_masks = model.generate_second_expert(
        **gen_kwargs, past_key_values=past_key_values, body_part="face_body")

    motion = ~(output_masks[0][0] | output_masks[1][0])
    tokens = {"face": [], "upper": [], "lower": [], "hand": []}
    for rel in output_ids[0][motion].tolist():
        if rel in vocab:
            part, code = vocab[rel]
            tokens[part].append(code)
    return tokens


@torch.no_grad()
def decode_pose(tokens, vqs, device):
    """Motion tokens -> SMPL-X axis-angle poses (n, 165) and face expressions (n, 100)."""
    rec = {part: vqs[part].decode(torch.tensor(tokens[part], device=device).unsqueeze(0).int())
           for part in ("face", "upper", "lower", "hand")}
    n = min(r.shape[1] for r in rec.values())

    upper = rotation_6d_to_axis_angle(rec["upper"][0, :n].reshape(n, 13, 6)).reshape(n, 39)
    lower = matrix_to_axis_angle(rotation_6d_to_matrix(rec["lower"][0, :n, :54].reshape(n, 9, 6))).reshape(n, 27)
    # EMAGE protocol: hands stay at rest (the hand VQ is decoded only to align lengths)
    hands = torch.zeros(n, 90, device=device)
    pose = (inverse_selection_tensor(upper, JOINT_MASK_UPPER, n)
            + inverse_selection_tensor(lower, JOINT_MASK_LOWER, n)
            + inverse_selection_tensor(hands, JOINT_MASK_HANDS, n))
    pose[:, 66:69] = rotation_6d_to_axis_angle(rec["face"][0, :n, :6].float())
    return pose, rec["face"][0, :n, 6:].float()


def smplx_forward(body_model, pose, betas, expressions, face_only=False):
    """Zero-translation SMPL-X; with face_only, everything but jaw + expressions is zeroed."""
    n = pose.shape[0]
    keep = (lambda x: torch.zeros_like(x)) if face_only else (lambda x: x)
    return body_model(
        betas=betas, transl=torch.zeros(n, 3, device=pose.device), expression=expressions,
        jaw_pose=pose[:, 66:69], global_orient=keep(pose[:, :3]), body_pose=keep(pose[:, 3:66]),
        left_hand_pose=keep(pose[:, 75:120]), right_hand_pose=keep(pose[:, 120:165]),
        leye_pose=keep(pose[:, 69:72]), reye_pose=keep(pose[:, 72:75]),
        return_verts=True, return_joints=True,
    )


# ============================================================================
# Main
# ============================================================================

def evaluate(args):
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    model = None if args.from_dir else load_model(args.checkpoint, args.glm_base_path, device)
    vocab = None if args.from_dir else load_motion_vocab(args.checkpoint)
    vqs = load_emage_vqs(args.vq_checkpoint, device)
    body_model = smplx.create(os.path.join(ROOT_DIR, "model_files", "smplx_models"), model_type="smplx",
                              gender="NEUTRAL_2020", use_face_contour=False, num_betas=300,
                              num_expression_coeffs=100, ext="npz", use_pca=False).to(device).eval()
    metrics = CoSpeechMetrics(cfg=build_metric_cfg(args.beat2_root), dataname="beat2").to(device)

    dataset = load_from_disk(args.test_dataset)
    n_samples = min(args.num_samples, len(dataset)) if args.num_samples > 0 else len(dataset)
    for idx in tqdm(range(n_samples), desc="BEAT2 speaker-2"):
        record = dataset[idx]
        seq = record["conv_id"]
        gt = np.load(os.path.join(args.beat2_root, "smplxflame_30", f"{seq}.npz"), allow_pickle=True)
        audio, sr = librosa.load(os.path.join(args.beat2_root, "wave16k", f"{seq}.wav"))
        audio = librosa.resample(audio, orig_sr=sr, target_sr=AUDIO_FPS)

        cached = os.path.join(args.from_dir, f"{seq}.npz") if args.from_dir else None
        if cached:
            data = np.load(cached)
            tokens = {part: data[f"{part}_tokens"].tolist() for part in ("face", "upper", "lower", "hand")}
        else:
            tokens = generate_motion_tokens(model, record, vocab, device, args.temperature, args.top_p)
        rec_pose, rec_exps = decode_pose(tokens, vqs, device)

        n = min(rec_pose.shape[0], gt["poses"].shape[0])
        rec_pose, rec_exps = rec_pose[:n], rec_exps[:n]
        tar_pose = torch.from_numpy(gt["poses"][:n]).float().to(device)
        tar_exps = torch.from_numpy(gt["expressions"][:n]).float().to(device)
        betas = torch.from_numpy(np.tile(gt["betas"].reshape(1, 300), (n, 1))).float().to(device)

        with torch.no_grad():
            vertices_rec = smplx_forward(body_model, rec_pose, betas, torch.zeros_like(tar_exps))
            vertices_rec_face = smplx_forward(body_model, rec_pose, betas, rec_exps, face_only=True)
            vertices_tar_face = smplx_forward(body_model, tar_pose, betas, tar_exps, face_only=True)
        metrics.update(
            rec_pose=axis_angle_to_6d(rec_pose.reshape(n, 55, 3)).reshape(1, n, 330),
            tar_pose=torch.from_numpy(axis_angle_to_6d_np(gt["poses"][:n].reshape(n, 55, 3)).reshape(1, n, 330)).float().to(device),
            vertices_rec=vertices_rec, vertices_rec_face=vertices_rec_face, vertices_tar_face=vertices_tar_face,
            raw_audio=[audio], motion_lengths=torch.tensor([n], device=device),
        )
        np.savez(os.path.join(args.output_dir, f"{seq}.npz"), poses=rec_pose.cpu().numpy(),
                 expressions=rec_exps.cpu().numpy(), mocap_frame_rate=POSE_FPS,
                 **{f"{part}_tokens": np.asarray(t, dtype=np.int64) for part, t in tokens.items()})

    scores = {k: float(v) for k, v in metrics.compute(sanity_flag=False).items()}
    results = {
        "fgd": scores["fid_score"], "bc": scores["align_score"], "diversity": scores["l1div"],
        "paper_scale": {"FGD": scores["fid_score"], "BC": scores["align_score"] * 10,
                        "Diversity": scores["l1div"]},
        "face_l2": scores["l2_loss"], "face_lvel": scores["lvel_loss"],
        "num_sequences": n_samples, "seed": args.seed,
    }
    print(json.dumps(results, indent=2))
    with open(os.path.join(args.output_dir, "eval_results.json"), "w") as f:
        json.dump(results, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="BEAT2 speaker-2 ViBES checkpoint (full, or Expert-1-only)")
    parser.add_argument("--from_dir", type=str, default=None,
                        help="Score the tokens saved by an earlier run in this directory instead of generating")
    parser.add_argument("--glm_base_path", type=str, default="THUDM/glm-4-voice-9b",
                        help="GLM-4-Voice base used to rebuild Expert-0 for Expert-1-only checkpoints")
    parser.add_argument("--vq_checkpoint", type=str, required=True,
                        help="EMAGE speaker-2 VQ-VAEs (vq_emage_speaker_2.ckpt)")
    parser.add_argument("--test_dataset", type=str, required=True,
                        help="Tokenized BEAT2 speaker-2 test set (one sequence per conversation)")
    parser.add_argument("--beat2_root", type=str, required=True,
                        help="BEAT2 root with smplxflame_30/, wave16k/ and weights/ (EMAGE evaluator)")
    parser.add_argument("--output_dir", type=str, default="eval_results/beat2_cospeech")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--num_samples", type=int, default=0, help="0 = all test conversations")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if not args.checkpoint and not args.from_dir:
        parser.error("--checkpoint is required unless --from_dir is given")
    evaluate(args)


if __name__ == "__main__":
    main()
