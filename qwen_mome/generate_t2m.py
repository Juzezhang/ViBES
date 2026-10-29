#!/usr/bin/env python3
"""Text-to-motion with the released Qwen-MoME T2M checkpoint (HF: JuzeZhang/ViBES-T2M).

caption -> MotionGPT-512 codes (sampling + classifier-free guidance) -> 263-D HumanML3D features
(MotionGPT VQ-VAE decoder + Comp_v6_KLD01 mean/std) -> 22 joints -> skeleton gif.
The defaults are the decoding settings behind the reported HumanML3D numbers (t 0.7 / top-p 0.9 / CFG 3).

  python -m qwen_mome.generate_t2m --ckpt ViBES-T2M/vibes_t2m_qwen_mome_1p7b.pt \
      --vq_ckpt /path/to/MotionGPT/checkpoints/MotionGPT-base/motiongpt_s3_h3d.tar \
      --meta_dir /path/to/MotionGPT/deps/t2m/t2m/t2m/Comp_v6_KLD01/meta \
      --text "a person walks forward, then squats down" --out_dir ./results/t2m
"""
import os, sys, argparse
from collections import OrderedDict
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from qwen_mome.modeling_qwen_mome import QwenMoMEForT2M

# First phrasing of MotionGPT's Text-to-Motion instruction templates (the one the evaluation uses).
PROMPT_PREFIX = 'Give me a gesture that corresponds to '


def load_model(ckpt_path, device):
    ck = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    cfg = ck.get('cfg', {})
    base = cfg.get('base', 'Qwen/Qwen3-1.7B')
    model = QwenMoMEForT2M(base, d_mot=cfg.get('d_mot', 512), ffn_mot=cfg.get('ffn_mot', 4096),
                           mot_vocab=cfg.get('mot_vocab', 514))
    missing, unexpected = model.load_state_dict(ck['mot_state'], strict=False)
    # base.* (the frozen Qwen3) comes from the HF download, so only those keys may be missing
    assert not unexpected and all(k.startswith('base.') for k in missing), (unexpected, missing[:5])
    return model.to(device).eval(), base


def load_vq(vq_ckpt, device):
    from multimodal_tokenizers.archs.mgpt_vq import VQVae
    vae = VQVae(nfeats=263, quantizer='ema_reset', code_num=512, code_dim=512, output_emb_width=512,
                down_t=2, stride_t=2, width=512, depth=3, dilation_growth_rate=3, norm=None, activation='relu')
    sd = torch.load(vq_ckpt, map_location='cpu', weights_only=False)['state_dict']
    vd = OrderedDict()
    for k, v in sd.items():
        if 'motion_vae' in k: vd[k.replace('motion_vae.', '')] = v
        elif 'vae' in k:      vd[k.replace('vae.', '')] = v
    vae.load_state_dict(vd, strict=True)
    return vae.to(device).eval()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True, help='vibes_t2m_qwen_mome_1p7b.pt from JuzeZhang/ViBES-T2M')
    ap.add_argument('--vq_ckpt', required=True, help="MotionGPT's motiongpt_s3_h3d.tar")
    ap.add_argument('--meta_dir', required=True, help='dir with the Comp_v6_KLD01 mean.npy / std.npy')
    ap.add_argument('--text', nargs='+', required=True, help='one or more motion captions')
    ap.add_argument('--out_dir', default='./results/t2m')
    ap.add_argument('--cfg_scale', type=float, default=3.0)
    ap.add_argument('--temperature', type=float, default=0.7)
    ap.add_argument('--top_p', type=float, default=0.9)
    ap.add_argument('--greedy', action='store_true')
    ap.add_argument('--max_new', type=int, default=60, help='max motion codes (4 frames each at 20 fps)')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--no_render', action='store_true', help='skip the skeleton gif')
    ap.add_argument('--device', default='cuda')
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    torch.manual_seed(a.seed)

    from transformers import AutoTokenizer
    from utils.motion_process import recover_from_ric
    model, base = load_model(a.ckpt, a.device)
    tok = AutoTokenizer.from_pretrained(base)
    vae = load_vq(a.vq_ckpt, a.device)
    mean = torch.tensor(np.load(f'{a.meta_dir}/mean.npy')).float().to(a.device)
    std = torch.tensor(np.load(f'{a.meta_dir}/std.npy')).float().to(a.device)

    for i, caption in enumerate(a.text):
        # prefix and caption are tokenized separately, as in training/evaluation
        ids = tok(PROMPT_PREFIX, add_special_tokens=False)['input_ids'] \
            + tok(caption, add_special_tokens=False)['input_ids']
        with torch.no_grad():
            codes = model.generate_motion(torch.tensor([ids], device=a.device), max_new=a.max_new,
                                          greedy=a.greedy, temperature=a.temperature, top_p=a.top_p,
                                          cfg_scale=a.cfg_scale)
            if len(codes) < 2:
                print(f'[{i:02d}] {caption!r}: empty generation, skipped')
                continue
            motion = vae.decode(torch.tensor(codes, device=a.device).unsqueeze(0).int()) * std + mean
        motion = motion.squeeze(0).float().cpu()
        joints = recover_from_ric(motion.unsqueeze(0), 22)[0].numpy()
        stem = os.path.join(a.out_dir, f'{i:02d}')
        np.save(f'{stem}_motion.npy', motion.numpy())     # 263-D HumanML3D features (denormalized)
        np.save(f'{stem}_joints.npy', joints)             # [T, 22, 3]
        if not a.no_render:
            import multimodal_tokenizers.render.matplot.plot_3d_global as plot_3d
            plot_3d.draw_to_batch(joints[None], [caption], [f'{stem}.gif'])
        print(f'[{i:02d}] {caption!r}: {len(codes)} codes -> {joints.shape[0]} frames ({joints.shape[0] / 20:.1f}s) '
              f'-> {stem}_*', flush=True)


if __name__ == '__main__':
    main()
