#!/usr/bin/env python3
"""Tokenize HumanML3D with the converged full-body RVQ (rvq8_dt1_full) using the OFFSET VOCABULARY
scheme (方案 A) for ViBES T2M training.

WHY OFFSET: RVQ has N *separate* codebooks (verified: layers.{0..7}.codebook are all distinct,
1024x512 each). So raw index `k` means a DIFFERENT vector at each layer — a bare index is ambiguous.
Offset makes every (layer, code) pair a unique token id:

    token_id = layer * code_num + code_idx          # 0 .. N*code_num-1  (8*1024 = 8192)
    decode:   layer, code = divmod(token_id, code_num)

LAYOUT: interleaved / position-major — pos0[l0..l7], pos1[l0..l7], ... This keeps temporal order and
puts coarse->fine within each position (standard flattened-RQ pattern for AR LMs).

Output matches the existing TOKENS/ format: {seq_id}.npy holding an int64 array of token ids,
shape (1, L), consumable by preprocess_hf_h3d_text2motion.py (which maps each -> <|motion_k|>).
"""
import os, sys, argparse
import numpy as np
import torch

sys.path.insert(0, "/simurgh/u/juze/code/UniMoTok")
from multimodal_tokenizers.archs.motiongpt_vq import MotionGPTVQVae as VQVae

H3D = "/simurgh/u/juze/datasets/HumanML3D"
META = "/simurgh/u/juze/code/MotionGPT/deps/t2m/t2m/t2m/Comp_v6_KLD01/meta"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="/simurgh2/users/juze/vqvae_hpsearch/runs_final/rvq8_dt1_full/vqvae_it300000.pt")
    ap.add_argument("--out_dir", default="/simurgh2/users/juze/datasets/HumanML3D/TOKENS_RVQ8_DT1")
    ap.add_argument("--use_nq", type=int, default=0,
                    help="use only the first K RVQ layers (0 = all). The ckpt was trained with "
                         "quant_dropout=0.2 so prefix decoding is valid -> K=4 halves the sequence.")
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.out_dir, exist_ok=True)
    mean = torch.tensor(np.load(f"{META}/mean.npy"), dtype=torch.float32, device=dev)
    std = torch.tensor(np.load(f"{META}/std.npy"), dtype=torch.float32, device=dev)

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    hp = ck["hp"]
    model = VQVae(nfeats=263, quantizer=hp["quantizer"], code_num=hp["code_num"], code_dim=hp["code_dim"],
                  output_emb_width=hp["code_dim"], down_t=hp["down_t"], stride_t=hp["stride_t"],
                  width=hp["width"], depth=hp["depth"], dilation_growth_rate=hp["dilation"],
                  num_quantizers=hp["num_quantizers"], quant_dropout=hp.get("quant_dropout", 0.2))
    model.load_state_dict(ck["state_dict"])
    model = model.to(dev).eval()

    CODE_NUM = hp["code_num"]                       # 1024
    NQ_ALL = hp["num_quantizers"]                   # 8
    NQ = args.use_nq if args.use_nq > 0 else NQ_ALL
    assert 1 <= NQ <= NQ_ALL
    VOCAB = NQ * CODE_NUM
    print(f"[rvq-offset] ckpt @{ck.get('iter')} | code_num={CODE_NUM} nq_all={NQ_ALL} use_nq={NQ} "
          f"down_t={hp['down_t']} stride_t={hp['stride_t']} -> VOCAB={VOCAB} (mot_vocab_size = {VOCAB}+specials)",
          flush=True)

    names = []
    for split in ("train", "val", "test"):
        p = f"{H3D}/{split}.txt"
        if os.path.exists(p):
            names += [l.strip() for l in open(p) if l.strip()]
    names = sorted(set(names))
    print(f"[rvq-offset] {len(names)} sequences", flush=True)

    ok = skipped = 0
    lens = []
    for n in names:
        p = f"{H3D}/new_joint_vecs/{n}.npy"
        if not os.path.exists(p):
            skipped += 1
            continue
        m = np.load(p).astype(np.float32)
        if m.shape[0] < 8:
            skipped += 1
            continue
        x = torch.tensor(m, device=dev)
        xn = ((x - mean) / (std + 1e-8)).unsqueeze(0)
        with torch.no_grad():
            code, _ = model.encode(xn)              # (1, T', NQ_ALL) stacked RVQ indices
        code = code[0, :, :NQ].cpu().numpy().astype(np.int64)   # (T', NQ) — prefix of layers
        # OFFSET + interleave: pos0[l0..lNQ-1], pos1[...], ...
        offsets = (np.arange(NQ, dtype=np.int64) * CODE_NUM)[None, :]   # (1, NQ)
        toks = (code + offsets).reshape(-1)                              # (T'*NQ,)
        assert toks.min() >= 0 and toks.max() < VOCAB, f"{n}: token out of range"
        np.save(f"{args.out_dir}/{n}.npy", toks[None, :])                 # (1, L) — matches TOKENS/ format
        lens.append(len(toks)); ok += 1
        if ok % 2000 == 0:
            print(f"  {ok}/{len(names)}", flush=True)

    lens = np.array(lens)
    print(f"[rvq-offset] DONE {ok} written ({skipped} skipped) -> {args.out_dir}", flush=True)
    print(f"[rvq-offset] token-seq length: mean {lens.mean():.0f}  median {np.median(lens):.0f}  "
          f"p95 {np.percentile(lens,95):.0f}  max {lens.max()}", flush=True)
    print(f"[rvq-offset] SET --mot_vocab_size {VOCAB + 2}  (={VOCAB} codes + 2 specials, matching the "
          f"514 = 512+2 convention)", flush=True)


if __name__ == "__main__":
    main()
