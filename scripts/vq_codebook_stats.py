import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from tqdm import tqdm

from multimodal_tokenizers.config import get_module_config
from multimodal_tokenizers.data.build_data import build_data
from multimodal_tokenizers.models.build_model import build_model
from multimodal_tokenizers.utils.load_checkpoint import _extract_state_dict


DATASET_KEY_MAP = {
    "AMASS": "AMASS",
    "AMASS_talking": "AMASS_talking",
    "AMASS_P2P": "AMASS_P2P",
    "BEAT2": "BEAT2",
    "CANDOR": "CANDOR",
    "TFHP": "TFHP",
    "YouTube_Talking": "YouTube_Talking",
    "YouTube_Talking_Synthetic": "YouTube_Talking_Synthetic",
    "BEAT2_Genmo": "BEAT2_Genmo",
    "AMASS_Genmo": "AMASS_Genmo",
    "beat2_genmo": "BEAT2_Genmo",
    "amass_genmo": "AMASS_Genmo",
}


class RunningStats:
    def __init__(self):
        self.count = 0
        self.mean = 0.0
        self.m2 = 0.0
        self.min = None
        self.max = None

    def update(self, value):
        value = float(value)
        self.count += 1
        delta = value - self.mean
        self.mean += delta / self.count
        delta2 = value - self.mean
        self.m2 += delta * delta2
        self.min = value if self.min is None else min(self.min, value)
        self.max = value if self.max is None else max(self.max, value)

    def summary(self):
        if self.count < 2:
            std = 0.0
        else:
            std = (self.m2 / (self.count - 1)) ** 0.5
        return {
            "count": self.count,
            "mean": self.mean,
            "std": std,
            "min": self.min,
            "max": self.max,
        }


def load_cfg(cfg_path, cfg_assets_path):
    cfg_assets = OmegaConf.load(cfg_assets_path)
    cfg_base = OmegaConf.load(os.path.join(cfg_assets.CONFIG_FOLDER, "default.yaml"))
    cfg_exp = OmegaConf.merge(cfg_base, OmegaConf.load(cfg_path))
    if not cfg_exp.FULL_CONFIG:
        cfg_exp = get_module_config(cfg_exp, cfg_assets.CONFIG_FOLDER)
    cfg = OmegaConf.merge(cfg_exp, cfg_assets)
    return cfg


def parse_list_arg(raw):
    if not raw:
        return []
    items = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        items.append(item)
    return items


def normalize_modalities(value):
    if not value:
        return None
    normalized = set()
    for mod in value:
        mod_str = str(mod).lower()
        if mod_str in {"full_genmo", "body"}:
            normalized.add("body")
        elif mod_str in {"lower_54", "lower_global", "global"}:
            normalized.add("lower")
        else:
            normalized.add(mod_str)
    return normalized


def infer_parts(cfg, parts_override=None):
    if parts_override:
        return [p.lower() for p in parts_override]

    selected = str(getattr(cfg, "Selected_part", "")).lower()
    motion_rep = str(getattr(cfg.DATASET, "motion_representation", "")).lower()
    if motion_rep == "genmo" or selected == "full_genmo":
        return ["body"]
    if selected in {"face", "upper", "hand"}:
        return [selected]
    if selected in {"lower", "lower_54", "lower_global", "global"}:
        return ["lower"]
    if selected == "compositional":
        return ["face", "upper", "lower", "hand"]
    return ["face", "upper", "lower", "hand"]


def get_part_cfg(cfg, part):
    key_map = {
        "face": "vae_face",
        "upper": "vae_upper",
        "lower": "vae_lower",
        "hand": "vae_hand",
        "body": "vae_body",
    }
    key = key_map.get(part)
    if not key:
        return None
    return OmegaConf.select(cfg, f"model.params.modality_tokenizer.{key}")


def get_codebook_size(cfg, part):
    part_cfg = get_part_cfg(cfg, part)
    if not part_cfg:
        return None
    params = part_cfg.get("params", {})
    for candidate in ("codebook_size", "n_embed", "num_tokens", "nb_code"):
        if candidate in params and params[candidate] is not None:
            return int(params[candidate])
    if "code_num" in params:
        return int(params["code_num"])
    return None


def get_target_dim(cfg, part):
    part_cfg = get_part_cfg(cfg, part)
    if not part_cfg:
        return None
    params = part_cfg.get("params", {})
    if "vae_test_dim" in params and params["vae_test_dim"] is not None:
        return int(params["vae_test_dim"])
    return None


def match_dim(tensor, target_dim):
    if target_dim is None:
        return tensor
    current_dim = tensor.shape[-1]
    if current_dim == target_dim:
        return tensor
    if current_dim > target_dim:
        return tensor[..., :target_dim]
    pad_dim = target_dim - current_dim
    pad_shape = list(tensor.shape[:-1]) + [pad_dim]
    pad = torch.zeros(pad_shape, device=tensor.device, dtype=tensor.dtype)
    return torch.cat([tensor, pad], dim=-1)


def compute_usage_stats(counts, topk_ratios):
    total = int(counts.sum())
    if total == 0:
        return {
            "total_tokens": 0,
            "perplexity": 0.0,
            "used_codes": 0,
            "dead_codes": int(len(counts)),
            "usage_ratio": 0.0,
            "topk_coverage": {},
        }
    probs = counts / total
    probs_nonzero = probs[probs > 0]
    entropy = float(-np.sum(probs_nonzero * np.log(probs_nonzero)))
    perplexity = float(np.exp(entropy))
    used = int((counts > 0).sum())
    dead = int((counts == 0).sum())
    usage_ratio = used / len(counts)
    sorted_counts = np.sort(counts)[::-1]
    cdf = np.cumsum(sorted_counts) / total
    topk_coverage = {}
    for ratio in topk_ratios:
        if ratio <= 0:
            continue
        if ratio < 1:
            k = max(1, int(len(counts) * ratio))
            label = f"top_{int(ratio*100)}pct"
        else:
            k = min(len(counts), int(ratio))
            label = f"top_{k}"
        coverage = float(cdf[k - 1]) if k > 0 else 0.0
        topk_coverage[label] = coverage
    return {
        "total_tokens": total,
        "perplexity": perplexity,
        "used_codes": used,
        "dead_codes": dead,
        "usage_ratio": usage_ratio,
        "topk_coverage": topk_coverage,
    }


def update_counts(counts, indices):
    flat = indices.reshape(-1)
    if flat.size == 0:
        return 0
    binc = np.bincount(flat, minlength=counts.shape[0])
    counts += binc[:counts.shape[0]]
    return int(flat.size)


def load_checkpoint_into_part(model, part, ckpt_path):
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state_dict = _extract_state_dict(checkpoint, ckpt_path)
    prefix = f"vae_{part}."
    has_prefix = any(key.startswith(prefix) for key in state_dict)
    if has_prefix:
        part_state = {key.replace(prefix, ""): value for key, value in state_dict.items() if key.startswith(prefix)}
        getattr(model, f"vae_{part}").load_state_dict(part_state, strict=True)
    else:
        getattr(model, f"vae_{part}").load_state_dict(state_dict, strict=False)


def get_checkpoint_path(cfg, part):
    ckpt_map = {
        "face": OmegaConf.select(cfg, "TEST.CHECKPOINTS_FACE") or OmegaConf.select(cfg, "TRAIN.PRETRAINED_VQ_FACE"),
        "upper": OmegaConf.select(cfg, "TEST.CHECKPOINTS_UPPER") or OmegaConf.select(cfg, "TRAIN.PRETRAINED_VQ_UPPER"),
        "lower": OmegaConf.select(cfg, "TEST.CHECKPOINTS_LOWER") or OmegaConf.select(cfg, "TRAIN.PRETRAINED_VQ_LOWER"),
        "hand": OmegaConf.select(cfg, "TEST.CHECKPOINTS_HAND") or OmegaConf.select(cfg, "TRAIN.PRETRAINED_VQ_HAND"),
        "body": OmegaConf.select(cfg, "TEST.CHECKPOINTS_BODY") or OmegaConf.select(cfg, "TRAIN.PRETRAINED_VQ"),
    }
    return ckpt_map.get(part)


def collect_from_tokens(cfg, parts, topk_ratios, max_files, out_dir):
    dataset_mods = getattr(cfg.DATASET, "MODALITIES", None)
    results = {}
    per_dataset = {}
    code_path_global = getattr(cfg.DATASET, "code_path", None)

    for dataset_cfg in cfg.DATASET.datasets:
        dataset_name = dataset_cfg.get("name")
        dataset_key = DATASET_KEY_MAP.get(dataset_name)
        if dataset_key is None:
            continue
        dataset_root = OmegaConf.select(cfg, f"DATASET.{dataset_key}.ROOT")
        if not dataset_root:
            continue
        dataset_modalities = normalize_modalities(dataset_mods.get(dataset_key)) if dataset_mods else None
        dataset_code_path = dataset_cfg.get("code_path") or code_path_global
        if not dataset_code_path:
            continue

        per_dataset.setdefault(dataset_name, {})
        for part in parts:
            if dataset_modalities and part not in dataset_modalities:
                continue
            part_dir = Path(dataset_root) / dataset_code_path / part
            if not part_dir.exists():
                continue
            files = list(part_dir.rglob("*.npy"))
            if max_files > 0 and len(files) > max_files:
                rng = np.random.default_rng(0)
                files = list(rng.choice(files, size=max_files, replace=False))
            codebook_size = get_codebook_size(cfg, part)
            if codebook_size is None:
                continue
            counts = np.zeros(codebook_size, dtype=np.int64)
            total_tokens = 0
            for fpath in tqdm(files, desc=f"{dataset_name}:{part} tokens"):
                arr = np.load(fpath)
                total_tokens += update_counts(counts, arr.astype(np.int64))
            stats = compute_usage_stats(counts, topk_ratios)
            stats["codebook_size"] = codebook_size
            stats["dataset"] = dataset_name
            stats["part"] = part
            per_dataset[dataset_name][part] = stats

            if part not in results:
                results[part] = {"counts": np.zeros(codebook_size, dtype=np.int64), "total_tokens": 0}
            results[part]["counts"] += counts
            results[part]["total_tokens"] += total_tokens

            if out_dir:
                out_dir.mkdir(parents=True, exist_ok=True)
                np.save(out_dir / f"{dataset_name}_{part}_usage.npy", counts)

    summary = {}
    for part, data in results.items():
        stats = compute_usage_stats(data["counts"], topk_ratios)
        stats["codebook_size"] = int(len(data["counts"]))
        stats["dataset"] = "ALL"
        stats["part"] = part
        summary[part] = stats
        if out_dir:
            np.save(out_dir / f"ALL_{part}_usage.npy", data["counts"])

    return summary, per_dataset


def collect_from_model(cfg, parts, topk_ratios, split, max_batches, out_dir, ckpt_override=None):
    cfg.TRAIN.STAGE = "token"
    cfg.TRAIN.SPLIT = split
    cfg.EVAL.SPLIT = split
    cfg.TEST.SPLIT = split
    datasets_test = OmegaConf.select(cfg, "DATASET.datasets_test")
    if split in {"val", "test"} and datasets_test is not None:
        cfg.DATASET.datasets = datasets_test

    datamodule = build_data(cfg, phase="test")
    if split == "train":
        dataloader = datamodule.train_dataloader()
    elif split == "val":
        dataloader = datamodule.val_dataloader()
    else:
        dataloader = datamodule.test_dataloader()

    model = build_model(cfg)
    model.eval()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    part_models = {}
    for part in parts:
        ckpt_path = ckpt_override if ckpt_override and len(parts) == 1 else get_checkpoint_path(cfg, part)
        if not ckpt_path:
            raise ValueError(f"Missing checkpoint for part '{part}'. Set TEST.CHECKPOINTS_* or TRAIN.PRETRAINED_VQ_*.")
        load_checkpoint_into_part(model, part, ckpt_path)
        part_models[part] = getattr(model, f"vae_{part}")
        part_models[part].to(device).eval()

    counts_global = {}
    for part in parts:
        codebook_size = get_codebook_size(cfg, part)
        if codebook_size is None:
            raise ValueError(f"Missing codebook_size for part '{part}' in config.")
        counts_global[part] = np.zeros(codebook_size, dtype=np.int64)
    counts_per_dataset = {}
    loss_stats = {part: RunningStats() for part in parts}

    dataset_mods = getattr(cfg.DATASET, "MODALITIES", None)
    dataset_mods_norm = {}
    if dataset_mods:
        for key, value in dataset_mods.items():
            dataset_mods_norm[key] = normalize_modalities(value)

    def get_dataset_modalities(dataset_name):
        dataset_key = DATASET_KEY_MAP.get(dataset_name)
        if not dataset_key:
            return None
        return dataset_mods_norm.get(dataset_key)

    def get_input_tensor(batch, part):
        if part == "body":
            return batch["motion_vector"]
        if part == "face":
            if "face_with_head" in batch:
                data = batch["face_with_head"]
            else:
                data = batch["face"]
            return data
        if part == "upper":
            return batch["upper"]
        if part == "lower":
            if "lower_54" in batch:
                return batch["lower_54"]
            return batch["lower"]
        if part == "hand":
            return batch["hand"]
        return None

    for batch_idx, batch in enumerate(tqdm(dataloader, desc="Collecting codes")):
        if max_batches > 0 and batch_idx >= max_batches:
            break

        dataset_names = batch.get("dataset_name", [])
        if isinstance(dataset_names, str):
            dataset_names = [dataset_names]

        for part in parts:
            dataset_name = dataset_names[0] if dataset_names else "unknown"
            dataset_modalities = get_dataset_modalities(dataset_name)
            if dataset_modalities and part not in dataset_modalities:
                continue

            if part == "body":
                if "motion_vector" not in batch:
                    continue
            elif part == "face":
                if "face" not in batch and "face_with_head" not in batch:
                    continue
            else:
                if part not in batch:
                    continue
            inputs = get_input_tensor(batch, part)
            if inputs is None:
                continue
            inputs = inputs.to(device)
            inputs = match_dim(inputs, get_target_dim(cfg, part))

            with torch.no_grad():
                indices = part_models[part].map2index(inputs).detach().cpu().numpy().astype(np.int64)
                output = part_models[part](inputs)
                embed_loss = None
                if isinstance(output, dict):
                    embed_loss = output.get("embedding_loss")
                elif isinstance(output, (list, tuple)) and len(output) > 0:
                    embed_loss = output[0]
                if embed_loss is not None:
                    loss_stats[part].update(embed_loss.item())

            counts_global[part] = counts_global[part] + np.bincount(indices.reshape(-1), minlength=counts_global[part].shape[0])

            if dataset_name not in counts_per_dataset:
                counts_per_dataset[dataset_name] = {}
            if part not in counts_per_dataset[dataset_name]:
                counts_per_dataset[dataset_name][part] = np.zeros_like(counts_global[part])
            counts_per_dataset[dataset_name][part] += np.bincount(indices.reshape(-1), minlength=counts_global[part].shape[0])

    summary = {}
    per_dataset = {}
    for part, counts in counts_global.items():
        stats = compute_usage_stats(counts, topk_ratios)
        stats["codebook_size"] = int(len(counts))
        stats["dataset"] = "ALL"
        stats["part"] = part
        stats["embedding_loss"] = loss_stats[part].summary()
        summary[part] = stats
        if out_dir:
            out_dir.mkdir(parents=True, exist_ok=True)
            np.save(out_dir / f"ALL_{part}_usage.npy", counts)

    for dataset_name, part_counts in counts_per_dataset.items():
        per_dataset[dataset_name] = {}
        for part, counts in part_counts.items():
            stats = compute_usage_stats(counts, topk_ratios)
            stats["codebook_size"] = int(len(counts))
            stats["dataset"] = dataset_name
            stats["part"] = part
            stats["embedding_loss"] = loss_stats[part].summary()
            per_dataset[dataset_name][part] = stats
            if out_dir:
                np.save(out_dir / f"{dataset_name}_{part}_usage.npy", counts)

    return summary, per_dataset


def print_stats(stats):
    for part, info in stats.items():
        print(f"\n[{part}]")
        print(f"  total_tokens: {info['total_tokens']}")
        print(f"  codebook_size: {info['codebook_size']}")
        print(f"  perplexity: {info['perplexity']:.2f}")
        print(f"  used_codes: {info['used_codes']} ({info['usage_ratio']*100:.1f}%), dead_codes: {info['dead_codes']}")
        for label, value in info["topk_coverage"].items():
            print(f"  {label}_coverage: {value*100:.2f}%")
        if "embedding_loss" in info:
            loss_info = info["embedding_loss"]
            if loss_info and loss_info.get("count", 0) > 0:
                print(
                    f"  embedding_loss: mean={loss_info['mean']:.6f} "
                    f"std={loss_info['std']:.6f} min={loss_info['min']:.6f} max={loss_info['max']:.6f}"
                )


def main():
    parser = argparse.ArgumentParser(description="VQ codebook usage diagnostics")
    parser.add_argument("--cfg", type=str, required=True, help="experiment config yaml")
    parser.add_argument("--cfg_assets", type=str, default="./configs/assets.yaml", help="assets config yaml")
    parser.add_argument("--mode", choices=["model", "tokens"], default="model", help="read from model or saved tokens")
    parser.add_argument("--parts", type=str, default="", help="comma-separated parts (face,upper,lower,hand,body)")
    parser.add_argument("--split", type=str, default="train", help="dataset split for model mode")
    parser.add_argument("--max_batches", type=int, default=0, help="max batches to scan in model mode (0 = no limit)")
    parser.add_argument("--max_files", type=int, default=0, help="max token files per dataset/part in tokens mode")
    parser.add_argument("--topk", type=str, default="0.1", help="comma-separated top-k ratios or counts")
    parser.add_argument("--out_dir", type=str, default="", help="output directory for histograms")
    parser.add_argument("--ckpt", type=str, default="", help="override checkpoint path (single part only)")
    parser.add_argument("--debug", action="store_true", help="use debug dataset sizing")
    args = parser.parse_args()

    cfg = load_cfg(args.cfg, args.cfg_assets)
    cfg.DEBUG = bool(args.debug)
    parts = infer_parts(cfg, parse_list_arg(args.parts))
    topk_raw = parse_list_arg(args.topk)
    topk_ratios = [float(x) for x in topk_raw] if topk_raw else [0.1]

    out_dir = Path(args.out_dir) if args.out_dir else None
    run_tag = time.strftime("%Y-%m-%d-%H-%M-%S")
    if out_dir and out_dir.is_dir():
        out_dir = out_dir / f"vq_codebook_stats_{run_tag}"

    if args.mode == "tokens":
        summary, per_dataset = collect_from_tokens(cfg, parts, topk_ratios, args.max_files, out_dir)
    else:
        ckpt_override = args.ckpt if args.ckpt else None
        summary, per_dataset = collect_from_model(
            cfg, parts, topk_ratios, args.split, args.max_batches, out_dir, ckpt_override
        )

    print("\n=== Global stats ===")
    print_stats(summary)
    if per_dataset:
        print("\n=== Per-dataset stats ===")
        for dataset_name, stats in per_dataset.items():
            print(f"\nDataset: {dataset_name}")
            print_stats(stats)

    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_dir / "stats.json", "w") as f:
            json.dump({"summary": summary, "per_dataset": per_dataset}, f, indent=2)
        print(f"\nSaved stats to {out_dir}")


if __name__ == "__main__":
    main()
