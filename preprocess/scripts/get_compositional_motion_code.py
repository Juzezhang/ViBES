"""Unified compositional motion tokenizer.

Replaces the former per-dataset ``get_compositional_motion_code_<dataset>.py``
family (now archived under ``preprocess/mics/``). This single script is fully
config-driven:

* Datasets to process come from ``cfg.DATASET.datasets``.
* Which body parts to tokenize per dataset come from
  ``cfg.DATASET.MODALITIES[<dataset>]`` (subset of face / upper / lower / hand).
* Output goes to ``<cfg.DATASET[<dataset>].ROOT>/<code_path>/<part>/<seq>.npy``
  where ``code_path`` is the per-dataset override or the global
  ``cfg.DATASET.code_path``.

Because tokenization is just ``vae.map2index`` over whatever the dataloader
provides, the same script handles both the compositional/LOM format and the
GENMO format — the format is selected entirely by the config.

Visualization (``--visualize``): decode the tokens back and render a
reconstruction-vs-ground-truth video. Face datasets render a FLAME mesh;
body rendering (SMPL-X / GENMO) is added separately. ``--max_visualize N``
caps how many sequences are rendered.

Usage:
    python -m preprocess.scripts.get_compositional_motion_code --cfg <config.yaml>
    python -m preprocess.scripts.get_compositional_motion_code --cfg <config.yaml> --visualize --max_visualize 4
"""

import os
import numpy as np
import pytorch_lightning as pl
from pathlib import Path
from tqdm import tqdm
from loguru import logger

from multimodal_tokenizers.config import parse_args
from multimodal_tokenizers.data.build_data import build_data
from multimodal_tokenizers.models.build_model import build_model
from multimodal_tokenizers.utils.load_checkpoint import load_pretrained_vae_compositional
import torch

# modality -> (model attribute, batch key, last-dim slice or None)
# The lower VAE only consumes the first 54 dims (matching the legacy scripts).
MODALITY_SPECS = {
    "face": ("vae_face", "face_with_head", None),
    "upper": ("vae_upper", "upper", None),
    "lower": ("vae_lower", "lower", 54),
    "hand": ("vae_hand", "hand", None),
}

FLAME_MODEL_PATH = "./model_files/FLAME2020/"
FACE_RENDER_BATCH = 50


def _resolve_dataset_name(batch_name, config_names):
    """Map a dataloader ``dataset_name`` to a config dataset key (case/underscore-insensitive)."""
    key = str(batch_name).lower().replace("_", "")
    for name in config_names:
        if name.lower().replace("_", "") == key:
            return name
    return None


class FaceRenderer:
    """Renders a FLAME face sequence (recon | GT side-by-side) to a video tensor.

    The face vector layout is [0:6] head pose (6D), [6:12] jaw pose (6D),
    [12:] expression coefficients.
    """

    def __init__(self, device):
        self.device = device
        self._flame = None
        self._renderer = None
        self._bs = None
        self._n_exp = None

    def _ensure(self, batch_size, n_exp):
        from smplx import FLAME
        from multimodal_tokenizers.utils.renderer_utils import RenderMesh
        if self._flame is None or self._bs != batch_size or self._n_exp != n_exp:
            self._flame = FLAME(
                FLAME_MODEL_PATH, num_expression_coeffs=n_exp, ext="pkl",
                batch_size=batch_size,
            ).to(self.device)
            faces = torch.LongTensor(self._flame.faces.astype(np.int64))
            self._renderer = RenderMesh(image_size=256, faces=faces, scale=1.0)
            self._bs = batch_size
            self._n_exp = n_exp

    def _render(self, face_seq, n_exp):
        from multimodal_tokenizers.utils.rotation_conversions import rotation_6d_to_axis_angle
        head = rotation_6d_to_axis_angle(face_seq[:, :6])
        jaw = rotation_6d_to_axis_angle(face_seq[:, 6:12])
        exp = face_seq[:, 12:12 + n_exp]
        T = face_seq.shape[0]
        imgs = []
        for i in range(0, T, FACE_RENDER_BATCH):
            bs = min(FACE_RENDER_BATCH, T - i)
            self._ensure(bs, n_exp)
            with torch.no_grad():
                out = self._flame(
                    global_orient=head[i:i + bs],
                    expression=exp[i:i + bs],
                    jaw_pose=jaw[i:i + bs],
                    shape=torch.zeros(bs, n_exp, device=self.device),
                )
            verts = out["vertices"].detach()
            rendered = self._renderer(verts)[0] / 255.0  # (bs, C, H, W)
            imgs.append(rendered.cpu())
        return torch.cat(imgs, dim=0)  # (T, C, H, W)

    def render_comparison(self, rec_face, gt_face, out_path, fps=25, audio=None, sample_rate=16000):
        """rec_face / gt_face: (T, D) tensors. Writes recon|GT side-by-side mp4.

        If ``audio`` (1D float waveform at ``sample_rate``) is given, it is muxed
        in (e.g. the clip's driving speech). It is trimmed to the rendered length.
        """
        from multimodal_tokenizers.utils.utils_videos import write_video
        n_exp = min(rec_face.shape[-1] - 12, gt_face.shape[-1] - 12, 100)
        rec_imgs = self._render(rec_face, n_exp)   # (T, C, H, W)
        gt_imgs = self._render(gt_face, n_exp)
        T = min(rec_imgs.shape[0], gt_imgs.shape[0])
        combined = torch.cat([rec_imgs[:T], gt_imgs[:T]], dim=-1)  # (T, C, H, 2W)
        if audio is not None:
            # trim audio to the rendered clip length (T frames)
            max_samples = int(round(T * sample_rate / fps))
            audio = np.asarray(audio, dtype=np.float32).reshape(-1)[:max_samples]
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        write_video(combined * 255.0, out_path, fps, audio, sample_rate, "aac")


def main():
    cfg = parse_args(phase="test")
    cfg.TRAIN.STAGE = "token"
    cfg.TRAIN.BATCH_SIZE = 1
    visualize = bool(getattr(cfg, "VISUALIZE", False))
    max_visualize = getattr(cfg, "MAX_VISUALIZE", None)

    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    pl.seed_everything(cfg.SEED_VALUE)

    if cfg.ACCELERATOR == "gpu":
        os.environ["PYTHONWARNINGS"] = "ignore"
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

    datasets = build_data(cfg, phase="token")
    print("datasets module initialized")

    # Build generic per-dataset output dirs + modality lists from config.
    global_code_path = cfg.DATASET.get("code_path")
    modalities_cfg = cfg.DATASET.get("MODALITIES", {}) or {}
    output_dirs = {}
    dataset_modalities = {}
    for entry in cfg.DATASET.datasets:
        name = entry.get("name")
        code_path = entry.get("code_path") or global_code_path
        if code_path is None:
            raise ValueError(
                f"No code_path for dataset '{name}': set cfg.DATASET.code_path "
                f"or a per-dataset code_path."
            )
        root = cfg.DATASET[name].ROOT
        output_dirs[name] = os.path.join(root, code_path)
        os.makedirs(output_dirs[name], exist_ok=True)
        mods = modalities_cfg.get(name) or list(MODALITY_SPECS.keys())
        dataset_modalities[name] = [m for m in mods if m in MODALITY_SPECS]

    needed = set()
    for mods in dataset_modalities.values():
        needed.update(mods)
    if not needed:
        needed = set(MODALITY_SPECS)

    model = build_model(cfg)
    logger.info("model {} loaded".format(cfg.model.target))
    load_pretrained_vae_compositional(cfg, model, logger, phase="token")

    for m in needed:
        vae = getattr(model, MODALITY_SPECS[m][0])
        if cfg.ACCELERATOR == "gpu":
            vae.to(device)
        vae.eval()
    logger.info("model loaded; tokenizing modalities: {}".format(sorted(needed)))

    face_renderer = FaceRenderer(device) if visualize else None
    n_visualized = 0

    config_names = list(output_dirs.keys())
    n_saved = 0
    for batch in tqdm(datasets.token_dataloader(), desc="compositional motion tokenize"):
        seq_name = batch["id_name"][0]
        name = _resolve_dataset_name(batch["dataset_name"][0], config_names)
        if name is None:
            if len(config_names) == 1:
                name = config_names[0]
            else:
                logger.warning(
                    f"Unknown dataset_name '{batch['dataset_name'][0]}' for seq "
                    f"'{seq_name}' (configured: {config_names}); skipping."
                )
                continue

        out_dir = output_dirs[name]
        do_viz = visualize and (max_visualize is None or n_visualized < max_visualize)
        for m in dataset_modalities[name]:
            vae_attr, batch_key, last_dim = MODALITY_SPECS[m]
            if batch_key not in batch:
                continue
            data = batch[batch_key].to(device)
            if last_dim is not None:
                data = data[:, :, :last_dim]
            vae = getattr(model, vae_attr)
            tokens = vae.map2index(data)
            target_path = os.path.join(out_dir, m, seq_name + ".npy")
            Path(target_path).parent.mkdir(parents=True, exist_ok=True)
            np.save(target_path, tokens.to("cpu").numpy())

            if do_viz and m == "face":
                try:
                    rec = vae.decode(tokens.int())[0]  # (T, D)
                    gt = data[0]                        # (T, D)
                    # Resolve the clip's driving speech and mux it in. For TFHP the
                    # FLAME coef clip and audios/<id>/<video_name>.wav are 1:1 and
                    # duration-matched, so the recon|GT faces stay lip-synced.
                    audio_wav = None
                    try:
                        tfhp_root = getattr(getattr(cfg.DATASET, "TFHP", None), "ROOT", None)
                        if tfhp_root and name == "TFHP":
                            # TFHP audio mirrors the coef tree 1:1 and is duration-matched:
                            # audios/<seq_name>.wav (seq_name uses '/' separators, e.g.
                            # WDA_DougJones/000/051).
                            wav_path = os.path.join(str(tfhp_root), "audios", seq_name + ".wav")
                            if os.path.exists(wav_path):
                                import librosa
                                audio_wav, _ = librosa.load(wav_path, sr=16000, mono=True)
                    except Exception as ae:
                        logger.warning(f"audio load failed for {seq_name}: {ae}")
                    vid_path = os.path.join(
                        os.path.dirname(out_dir.rstrip("/")),
                        "reconstructed_vs_gt", "face", seq_name.replace("/", "_") + ".mp4",
                    )
                    face_renderer.render_comparison(rec, gt, vid_path,
                                                    fps=int(cfg.DATASET.get("pose_fps", 25)),
                                                    audio=audio_wav, sample_rate=16000)
                    logger.info(f"saved face viz: {vid_path}"
                                + ("" if audio_wav is None else " (with audio)"))
                except Exception as e:
                    logger.warning(f"face viz failed for {seq_name}: {e}")
        if do_viz:
            n_visualized += 1
        n_saved += 1

    print(f"Motion tokenization done: {n_saved} sequences -> {list(output_dirs.values())}")


if __name__ == "__main__":
    main()
