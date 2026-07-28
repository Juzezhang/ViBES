# Qwen-MoME T2M — frozen-LLM text-to-motion (ViBES AR-MoME)

Text-to-motion in the ViBES AR-MoME architecture on a **fully frozen Qwen3 backbone**: a narrow
trainable motion expert (d_mot 512, ~0.27B) joins the frozen text LLM only through shared
masked-interleaved attention (text attends text; motion attends text+motion), so the language
model is bit-exact preserved.

HumanML3D test (Guo protocol, 20 reps): **R@1 0.489 / R@3 0.769 / FID 0.265** — T2M-GPT-level
quality with the conversational LLM intact.

- Weights + model card: https://huggingface.co/JuzeZhang/ViBES-T2M
- `modeling_qwen_mome.py` — model (needs transformers with Qwen3)
- `train_qwen_mome_t2m.py` — plain-PyTorch bf16 trainer (MotionGPT-512 tokens, caption-dropout 0.1
  for CFG; select checkpoints by generation quality on the val split, not teacher-forced loss)

Decode: sampling temp 0.7 / top-p 0.9 with CFG scale 2.5–3.
