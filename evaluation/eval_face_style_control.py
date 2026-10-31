#!/usr/bin/env python3
"""
Evaluate ViBES face model with style control (ARTalk-style evaluation).

Uses the same two-stage flow as conversational_agent/inference_glm_mot_trainingset_modalitynum_3_a2m.py:
1. Load complete sequence from test dataset (text + audio + face tokens)
2. Offset adjust + zero out face token positions
3. generate_first_expert_cache → KV cache (processes text+audio)
4. generate_second_expert(past_key_values=kv_cache) → generates face tokens
5. Compare with GT face tokens → decode to FLAME vertices → compute metrics

Usage:
    python -m evaluation.eval_face_style_control \
        --checkpoint /path/to/checkpoint \
        --test_dataset /path/to/processed_tfhp_tokenized_face_style_test/tokenized_dataset \
        --output_dir ./eval_results

The paper's TFHP numbers use one sequence per test session: build the test set with
preprocess/preprocess_hf_tfhp_face_style.py --split test --max_seq_length 10000000.
"""
import sys
import os

_script_dir = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(_script_dir, ".."))
if ROOT_DIR in sys.path:
    sys.path.remove(ROOT_DIR)
sys.path.insert(0, ROOT_DIR)

import argparse
import json
import torch
import numpy as np
from tqdm import tqdm
from datasets import load_from_disk
from transformers import AutoTokenizer, AutoConfig, AutoModel
from transformers.modeling_utils import load_sharded_checkpoint
from smplx import FLAME

sys.path.insert(0, os.path.join(ROOT_DIR, "training"))
from expert_io import is_expert1_checkpoint, load_expert1_checkpoint

from multimodal_tokenizers.archs.lom_vq import VQVAEConvZeroDSUS1_PaperVersion
from multimodal_tokenizers.utils.rotation_conversions import rotation_6d_to_axis_angle
from evaluation.face_metrics import calc_val_metrics

# ============================================================================
# Constants
# ============================================================================
FACE_TOKEN_OFFSET = 168736
FACE_FEATURE_DIM = 112
FLAME_NUM_EXPRESSIONS = 100
BOM_RELATIVE_ID = 1280  # begin_of_motion token in expert-2 vocab

VAE_CHECKPOINT_FACE = os.path.join(ROOT_DIR, 'model_files/pretrained_cpt/face/face.ckpt')
FLAME_MODEL_DIR = os.environ.get(
    "VIBES_FLAME_MODEL_DIR",
    os.path.join(ROOT_DIR, "model_files", "FLAME2020"),
)


# ============================================================================
# Helpers
# ============================================================================

def load_vae_face_model(device):
    vae = VQVAEConvZeroDSUS1_PaperVersion(
        vae_layer=3, code_num=512, codebook_size=512,
        vae_quantizer_lambda=1, vae_test_dim=FACE_FEATURE_DIM
    )
    sd = torch.load(VAE_CHECKPOINT_FACE, map_location="cpu", weights_only=False)
    sd = {k.replace('vae_face.', ''): v for k, v in sd.items() if 'vae_face' in k}
    vae.load_state_dict(sd, strict=True)
    vae.eval().to(device)
    return vae


def decode_tokens_to_verts(tokens, vae, device):
    """Decode face codebook indices → FLAME vertices."""
    if len(tokens) == 0:
        return None
    t = torch.tensor(tokens, device=device).unsqueeze(0)
    rec = vae.decode(t.int())
    n = rec.shape[1]
    head = rotation_6d_to_axis_angle(rec[0, :, :6])
    jaw = rotation_6d_to_axis_angle(rec[0, :, 6:12])
    exp = rec[0, :, 12:]
    all_v = []
    for i in range(0, n, 100):
        bs = min(100, n - i)
        fl = FLAME(FLAME_MODEL_DIR, num_expression_coeffs=FLAME_NUM_EXPRESSIONS,
                   ext='pkl', batch_size=bs).to(device)
        with torch.no_grad():
            out = fl(global_orient=torch.zeros(bs, 3, device=device),
                     expression=exp[i:i+bs],
                     jaw_pose=jaw[i:i+bs], shape=torch.zeros(bs, 100, device=device))
        all_v.append(out['vertices'].detach())
    return torch.cat(all_v, dim=0).unsqueeze(0)


def string_to_token(string_list):
    """Extract token IDs from token string representations like '<|face_123|>'."""
    return [int(val.split('_')[1].split('|')[0]) for val in string_list if isinstance(val, str)]


# ============================================================================
# Main Evaluation
# ============================================================================

def evaluate(args):
    device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load tokenizer
    tokenizer_path = args.vibes_config or os.path.join(ROOT_DIR, "vibes")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)

    # Load model
    print("Loading model...")
    config = AutoConfig.from_pretrained(tokenizer_path, trust_remote_code=True)
    config.num_layers = args.layer_num
    model = AutoModel.from_config(
        config, trust_remote_code=True,
        torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2",
    ).to(device)
    if is_expert1_checkpoint(args.checkpoint):
        print(f"Expert-1-only checkpoint; reconstructing Expert-0 from {args.glm_base_path}")
        load_expert1_checkpoint(model, args.checkpoint, args.glm_base_path)
    else:
        load_sharded_checkpoint(model, args.checkpoint)
    model.eval()

    # Load VAE
    vae = load_vae_face_model(device)

    # Load dataset
    print(f"Loading test dataset: {args.test_dataset}")
    dataset = load_from_disk(args.test_dataset)
    n_samples = min(args.num_samples, len(dataset)) if args.num_samples > 0 else len(dataset)
    print(f"Evaluating {n_samples}/{len(dataset)} samples")

    # Eval loop
    all_lve, all_mhd, all_fdd = [], [], []

    for idx in tqdm(range(n_samples), desc="Evaluating"):
        record = dataset[idx]

        try:
            # ---- Load complete sequence from dataset ----
            # (Same as conversational_agent/inference_glm_mot_trainingset_modalitynum_3_a2m.py)
            input_ids = torch.tensor(record['input_ids'], device=device).unsqueeze(0)

            # Offset adjustment (training collator does the same)
            input_ids[input_ids >= FACE_TOKEN_OFFSET] -= FACE_TOKEN_OFFSET

            # Extract GT face tokens BEFORE zeroing (labels have offset)
            gt_face = [record['labels'][i] - FACE_TOKEN_OFFSET
                       for i in range(len(record['labels']))
                       if record['modality_masks_2'][i] and record['labels'][i] != -100]

            if len(gt_face) < 10:
                continue

            # Build modality masks
            modality_masks = torch.tensor([
                [record['modality_masks_0']],
                [record['modality_masks_1']],
                [record['modality_masks_2']],
            ], dtype=torch.bool, device=device)

            position_encoding_indices = torch.tensor(
                record['position_encoding_indices'], device=device
            ).unsqueeze(0)

            attention_mask = torch.tensor(
                record['attention_mask'], device=device
            ).unsqueeze(0)

            # Zero out face token positions (keep begin_of_motion=1280)
            # Model will regenerate these via second expert
            mask_condition_2 = (
                torch.tensor(record['modality_masks_2'], device=device)
                & (input_ids[0] != BOM_RELATIVE_ID)
            )
            input_ids[:, mask_condition_2] = 0

            # ---- Stage 1: generate_first_expert_cache ----
            seq_len = input_ids.shape[1]
            valid_length = min(seq_len, args.max_seq_length)

            with torch.no_grad():
                past_key_values = model.generate_first_expert_cache(
                    input_ids=input_ids[:, :valid_length],
                    attention_mask=attention_mask[:, :valid_length],
                    modality_masks=modality_masks[:, :, :valid_length],
                    position_encoding_indices=position_encoding_indices[:, :valid_length],
                )

            # ---- Stage 2: generate_second_expert ----
            with torch.no_grad():
                output_ids, output_modality_masks = model.generate_second_expert(
                    input_ids=input_ids[:, :valid_length],
                    attention_mask=attention_mask[:, :valid_length],
                    max_new_tokens=1024,
                    do_sample=False,
                    temperature=0.0,
                    top_p=1.0,
                    modality_masks=modality_masks[:, :, :valid_length],
                    position_encoding_indices=position_encoding_indices[:, :valid_length],
                    past_key_values=past_key_values,
                    body_part="face",
                )

            # ---- Extract generated face tokens ----
            # Add offset back for decoding via tokenizer
            for i in range(output_ids.shape[0]):
                for j in range(output_ids.shape[1]):
                    if output_modality_masks[0][i, j] == False and output_modality_masks[1][i, j] == False:
                        output_ids[i, j] += FACE_TOKEN_OFFSET

            full_response = tokenizer.decode(output_ids[0], skip_special_tokens=False)
            response_split = full_response.split("<|")
            pred_face = string_to_token([s for s in response_split if "face_" in s])

            if len(pred_face) == 0:
                continue

            # ---- Compute metrics ----
            min_len = min(len(pred_face), len(gt_face))
            pred_verts = decode_tokens_to_verts(pred_face[:min_len], vae, device)
            gt_verts = decode_tokens_to_verts(gt_face[:min_len], vae, device)

            if pred_verts is None or gt_verts is None:
                continue

            mf = min(pred_verts.shape[1], gt_verts.shape[1])
            lve, mhd, fdd = calc_val_metrics(pred_verts[:, :mf], gt_verts[:, :mf])
            all_lve.append(lve.item())
            all_mhd.append(mhd.item())
            all_fdd.append(fdd.item())

            if (idx + 1) % 10 == 0:
                print(f"  [{idx+1}] LVE={np.nanmean(all_lve):.4f} MHD={np.nanmean(all_mhd):.4f} FDD={np.nanmean(all_fdd):.4f}")

        except Exception as e:
            print(f"  Sample {idx} failed: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Results
    results = {
        "LVE": float(np.nanmean(all_lve)) if all_lve else float('nan'),
        "MHD": float(np.nanmean(all_mhd)) if all_mhd else float('nan'),
        "FDD": float(np.nanmean(all_fdd)) if all_fdd else float('nan'),
        "num_evaluated": len(all_lve),
        "num_total": n_samples,
    }
    print("\n" + "=" * 50)
    print(f"LVE: {results['LVE']:.4f} mm")
    print(f"MHD: {results['MHD']:.4f} mm")
    print(f"FDD: {results['FDD']:.4f}")
    print(f"Evaluated: {results['num_evaluated']}/{results['num_total']}")
    print("=" * 50)

    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "eval_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved to {args.output_dir}/eval_results.json")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Face checkpoint: full, or Expert-1-only (marked with expert_checkpoint.json)")
    parser.add_argument("--glm_base_path", type=str, default="THUDM/glm-4-voice-9b",
                        help="GLM-4-Voice base used to rebuild Expert-0 for Expert-1-only checkpoints")
    parser.add_argument("--vibes_config", type=str, default=None,
                        help="MoME config+tokenizer dir (defaults to ./vibes, the 9B backbone). Set to vibes_0.5b "
                             "(with --glm_base_path pointing at ViBES-Audio and --layer_num 24) for the 0.5B variant.")
    parser.add_argument("--test_dataset", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./eval_results")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--layer_num", type=int, default=40)
    parser.add_argument("--num_samples", type=int, default=0, help="0 = all")
    parser.add_argument("--max_seq_length", type=int, default=2048)
    args = parser.parse_args()
    evaluate(args)


if __name__ == "__main__":
    main()
