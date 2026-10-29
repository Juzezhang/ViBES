# Qwen-MoME T2M — frozen-LLM text-to-motion (ViBES AR-MoME)

Text-to-motion in the ViBES AR-MoME architecture on a **fully frozen Qwen3-1.7B backbone**: a narrow
trainable motion expert (d_mot 512, ~0.27B) joins the frozen text LLM only through shared
masked-interleaved attention (text attends text; motion attends text+motion), so the language
model is bit-exact preserved. Motion is tokenized with MotionGPT's public HumanML3D VQ-VAE
(512 codes, 4× temporal downsample, 263-D HumanML3D features).

## Results

HumanML3D test split (4384 clips), Guo et al. evaluator. All rows are measured with the same harness;
the Qwen-MoME rows average 20 replications (over the 4304 / 4332 clips that produced motion), the
reference rows 1–2:

| | R@1 ↑ | R@3 ↑ | FID ↓ | MM-Dist ↓ |
|---|---|---|---|---|
| Real motion (harness sanity check) | 0.522 | 0.809 | 0.000 | 2.948 |
| MotionGPT VQ-VAE reconstruction (tokenizer ceiling) | 0.497 | 0.789 | 0.098 | 3.071 |
| MotionGPT (official checkpoint) | 0.429 | 0.677 | 0.105 | 3.948 |
| **Qwen-MoME 1.7B, CFG 3** | **0.486** | **0.769** | 0.265 | **3.196** |
| Qwen-MoME 1.7B, CFG 2.5 | 0.489 | 0.766 | 0.301 | 3.208 |

Both Qwen-MoME rows sample with temperature 0.7 / top-p 0.9; CFG 3 is the default in `generate_t2m.py`.

## Quick start

`modeling_qwen_mome.py` needs a `transformers` release with Qwen3 (≥ 4.51). The main ViBES
environment pins 4.48.3 for GLM-4-Voice, so run this in a separate environment, e.g.
`pip install "transformers>=4.51" torch numpy scipy tqdm matplotlib imageio`.

```bash
# 1. Released motion expert (the frozen Qwen3-1.7B is pulled from Qwen/Qwen3-1.7B at load time)
huggingface-cli download JuzeZhang/ViBES-T2M vibes_t2m_qwen_mome_1p7b.pt --local-dir ./ViBES-T2M

# 2. MotionGPT VQ-VAE decoder + HumanML3D mean/std
huggingface-cli download OpenMotionLab/MotionGPT-base motiongpt_s3_h3d.tar --local-dir /path/to/MotionGPT/checkpoints/MotionGPT-base
#    mean.npy / std.npy: Comp_v6_KLD01/meta from MotionGPT's `bash prepare/download_t2m_evaluators.sh`

# 3. Generate
python -m qwen_mome.generate_t2m \
    --ckpt ./ViBES-T2M/vibes_t2m_qwen_mome_1p7b.pt \
    --vq_ckpt /path/to/MotionGPT/checkpoints/MotionGPT-base/motiongpt_s3_h3d.tar \
    --meta_dir /path/to/MotionGPT/deps/t2m/t2m/t2m/Comp_v6_KLD01/meta \
    --text "a person walks forward, then squats down" "the man runs backwards" \
    --out_dir ./results/t2m
```

Per caption `NN`, it writes `NN_motion.npy` (263-D HumanML3D features), `NN_joints.npy` (`[T, 22, 3]`, 20 fps)
and `NN.gif` (skeleton; skip with `--no_render`).

## Files

- `modeling_qwen_mome.py` — the model; `generate_motion(..., cfg_scale=...)` does sampling with
  classifier-free guidance (uncond = motion-only prefix)
- `generate_t2m.py` — caption → codes → 263-D motion → joints → gif
- `train_qwen_mome_t2m.py` — plain-PyTorch bf16 trainer on HumanML3D MotionGPT-512 tokens (`H3D_ROOT`).
  The released checkpoint: frozen Qwen3-1.7B, bs 16, lr 2e-4, caption-dropout 0.1 (needed for CFG),
  peak at ~10k steps. Select checkpoints by generation quality on the val split, not teacher-forced loss.
