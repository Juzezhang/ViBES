"""Train Qwen-MoME T2M (frozen Qwen3-0.6B + 0.27B motion expert) on HumanML3D MotionGPT-512 tokens.

Plain-torch bf16 loop (NO deepspeed — the ds-bf16 grad path NaNs on this model family; plain torch
is proven stable). Data built on the fly from H3D TOKENS/*.npy + texts (same source as h3d_t2m.py):
  text = Qwen-tokenized instruction+caption   motion = [BEGIN(512)] codes [END(513)]
val split + TF val_acc every eval_every; checkpoints saved for external greedy-probe R@1 evals
(TF loss decouples from generation quality — select checkpoints by generation, not eval_loss).
Run:  conda activate lom_release; python -m qwen_mome.train_qwen_mome_t2m --name mome_r1
"""
import os, sys, json, time, math, random, argparse
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from qwen_mome.modeling_qwen_mome import QwenMoMEForT2M, BEGIN, END

H3D = '/path/to/HumanML3D'
N_MOT = 512
TOKENS_DIR = 'TOKENS'


def read_ids(split):
    ids = [l.strip() for l in open(f'{H3D}/{split}.txt') if l.strip()]
    return [i for i in ids if os.path.exists(f'{H3D}/{TOKENS_DIR}/{i}.npy') and os.path.exists(f'{H3D}/texts/{i}.txt')]


def captions(idx):
    caps = [l.split('#')[0].strip() for l in open(f'{H3D}/texts/{idx}.txt')]
    return [c for c in caps if c] or ['a person moves.']


class T2MDataset(Dataset):
    def __init__(self, ids, tok, phrasings, max_mot=196, fixed=False, mot_vocab=514, cap_drop=0.0):
        self.ids, self.tok, self.ph, self.max_mot, self.fixed = ids, tok, phrasings, max_mot, fixed
        self.n_codes = mot_vocab - 2; self.BEGIN = mot_vocab - 2; self.END = mot_vocab - 1
        self.cap_drop = cap_drop

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        idx = self.ids[i]
        mot = np.load(f'{H3D}/{TOKENS_DIR}/{idx}.npy').reshape(-1).astype(np.int64)
        mot = np.clip(mot[:self.max_mot], 0, self.n_codes - 1)
        cap = captions(idx)[0] if self.fixed else random.choice(captions(idx))
        s = self.ph[0] if self.fixed else random.choice(self.ph)
        ph = '<Caption_Placeholder>'
        pre, post = s.split(ph, 1) if ph in s else (s, '')
        tids = (self.tok(pre, add_special_tokens=False)['input_ids'] if pre else []) \
             + self.tok(cap, add_special_tokens=False)['input_ids'] \
             + (self.tok(post, add_special_tokens=False)['input_ids'] if post else [])
        if self.cap_drop > 0 and (not self.fixed) and random.random() < self.cap_drop:
            tids = []                      # unconditional sample (CFG training)
        T = len(tids)
        ids = tids + [self.BEGIN] + mot.tolist() + [self.END]
        mm = [False] * T + [True] * (len(mot) + 2)
        lab = [-100] * (T + 1) + mot.tolist() + [self.END]   # supervise codes + END (predicted from BEGIN onward)
        return (torch.tensor(ids), torch.tensor(mm, dtype=torch.bool), torch.tensor(lab))


def collate(batch):
    ML = max(x[0].shape[0] for x in batch); B = len(batch)
    ids = torch.zeros(B, ML, dtype=torch.long)
    mm = torch.zeros(B, ML, dtype=torch.bool)
    lab = torch.full((B, ML), -100, dtype=torch.long)
    am = torch.zeros(B, ML, dtype=torch.long)
    for i, (x, m, l) in enumerate(batch):
        L = x.shape[0]
        ids[i, :L] = x; mm[i, :L] = m; lab[i, :L] = l; am[i, :L] = 1
    return ids, mm, lab, am


@torch.no_grad()
def evaluate(model, loader, dev, max_batches=40):
    model.eval(); tl = tc = tt = 0
    for bi, (ids, mm, lab, am) in enumerate(loader):
        if bi >= max_batches:
            break
        ids, mm, lab, am = ids.to(dev), mm.to(dev), lab.to(dev), am.to(dev)
        out = model(ids, mm, attention_mask=am, labels=lab)
        pred = out['logits'][:, :-1].argmax(-1); tgt = lab[:, 1:]; msk = tgt != -100
        tc += (pred[msk] == tgt[msk]).sum().item(); n = msk.sum().item()
        tt += n; tl += out['loss'].item() * n
    model.train()
    return tl / max(1, tt), tc / max(1, tt)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--name', required=True)
    p.add_argument('--base', default='Qwen/Qwen3-0.6B')
    p.add_argument('--d_mot', type=int, default=512)
    p.add_argument('--ffn_mot', type=int, default=4096)
    p.add_argument('--mot_vocab', type=int, default=514)
    p.add_argument('--tokens_dir', default='TOKENS')
    p.add_argument('--max_mot', type=int, default=196)
    p.add_argument('--cap_drop', type=float, default=0.0)
    p.add_argument('--batch_size', type=int, default=32)
    p.add_argument('--lr', type=float, default=2e-4)
    p.add_argument('--warmup', type=int, default=500)
    p.add_argument('--weight_decay', type=float, default=0.05)
    p.add_argument('--grad_clip', type=float, default=1.0)
    p.add_argument('--max_steps', type=int, default=30000)
    p.add_argument('--eval_every', type=int, default=500)
    p.add_argument('--log_every', type=int, default=25)
    p.add_argument('--out_root', default='/path/to/experiments/qwen_mome')
    p.add_argument('--seed', type=int, default=1234)
    p.add_argument('--smoke', action='store_true')
    p.add_argument('--resume_from', default='', help='load mot_state from a checkpoint and continue (fresh optimizer)')
    a = p.parse_args()
    dev = 'cuda'
    torch.manual_seed(a.seed); random.seed(a.seed); np.random.seed(a.seed)
    out = os.path.join(a.out_root, a.name); os.makedirs(out, exist_ok=True)
    logf = open(os.path.join(out, 'log.jsonl'), 'a')

    def log(d):
        logf.write(json.dumps({'t': round(time.time(), 1), **d}) + '\n'); logf.flush()
        print(f"[{a.name}] " + ' '.join(f"{k}={v}" for k, v in d.items()), flush=True)

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.base)
    global TOKENS_DIR, N_MOT_G
    model = QwenMoMEForT2M(a.base, d_mot=a.d_mot, ffn_mot=a.ffn_mot, mot_vocab=a.mot_vocab).to(dev)
    if a.resume_from:
        rck = torch.load(a.resume_from, map_location='cpu', weights_only=False)
        miss, unexp = model.load_state_dict(rck['mot_state'], strict=False)
        assert not unexp and all(k.startswith('base.') for k in miss)
        log({'event': 'resume', 'from': a.resume_from, 'prev_step': rck.get('step')})
    n_tr = sum(p.numel() for p in model.trainable_parameters())
    n_all = sum(p.numel() for p in model.parameters())
    log({'event': 'model', 'trainable_M': round(n_tr / 1e6, 1), 'total_M': round(n_all / 1e6, 1)})

    TOKENS_DIR = a.tokens_dir
    phr = json.load(open(f'{H3D}/template_instructions.json'))['Text-to-Motion']['caption']['input']
    train_ids, val_ids = read_ids('train'), read_ids('val')
    if not val_ids:                       # H3D has val.txt; fall back to a train holdout
        train_ids, val_ids = train_ids[:-1000], train_ids[-1000:]
    log({'event': 'data', 'train': len(train_ids), 'val': len(val_ids)})
    TOKENS_DIR = a.tokens_dir
    tl = DataLoader(T2MDataset(train_ids, tok, phr, max_mot=a.max_mot, mot_vocab=a.mot_vocab, cap_drop=a.cap_drop), batch_size=a.batch_size, shuffle=True,
                    drop_last=True, num_workers=2, collate_fn=collate, pin_memory=True)
    vl = DataLoader(T2MDataset(val_ids, tok, phr, fixed=True, max_mot=a.max_mot, mot_vocab=a.mot_vocab), batch_size=a.batch_size,
                    shuffle=False, num_workers=1, collate_fn=collate)
    if a.smoke:
        ids, mm, lab, am = next(iter(tl))
        out_ = model(ids.to(dev), mm.to(dev), attention_mask=am.to(dev), labels=lab.to(dev))
        out_['loss'].backward()
        gn = torch.nn.utils.clip_grad_norm_(model.trainable_parameters(), 1e9)
        log({'event': 'smoke', 'loss': round(out_['loss'].item(), 4), 'grad_norm': round(float(gn), 2),
             'expect_loss~': round(math.log(514), 2)})
        cds = model.generate_motion(ids[:1, :8].to(dev), max_new=8)
        log({'event': 'smoke_gen', 'codes': cds}); return

    decay = [p_ for p_ in model.trainable_parameters() if p_.ndim >= 2]
    nodecay = [p_ for p_ in model.trainable_parameters() if p_.ndim < 2]
    opt = torch.optim.AdamW([{'params': decay, 'weight_decay': a.weight_decay},
                             {'params': nodecay, 'weight_decay': 0.0}],
                            lr=a.lr, betas=(0.9, 0.95), eps=1e-8)

    def lr_at(s):
        if s < a.warmup:
            return a.lr * s / a.warmup
        pr = min(1.0, (s - a.warmup) / max(1, a.max_steps - a.warmup))
        return a.lr * (0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * pr)))

    step = 0; best = float('inf'); rl = rt = 0; t0 = time.time()
    model.train()
    while step < a.max_steps:
        for ids, mm, lab, am in tl:
            if step >= a.max_steps:
                break
            ids, mm, lab, am = ids.to(dev), mm.to(dev), lab.to(dev), am.to(dev)
            o = model(ids, mm, attention_mask=am, labels=lab)   # NB: don't shadow `out` (ckpt dir)
            o['loss'].backward()
            lr = lr_at(step)
            for g in opt.param_groups:
                g['lr'] = lr
            gn = torch.nn.utils.clip_grad_norm_(model.trainable_parameters(), a.grad_clip)
            opt.step(); opt.zero_grad(set_to_none=True); step += 1
            n = (lab != -100).sum().item(); rl += o['loss'].item() * n; rt += n
            if step % a.log_every == 0:
                log({'event': 'train', 'step': step, 'loss': round(rl / max(1, rt), 4),
                     'grad_norm': round(float(gn), 2), 'lr': round(lr, 6),
                     'sps': round(a.log_every / (time.time() - t0), 2)}); rl = rt = 0; t0 = time.time()
            if step % a.eval_every == 0:
                vloss, vacc = evaluate(model, vl, dev)
                log({'event': 'eval', 'step': step, 'val_loss': round(vloss, 4), 'val_acc': round(vacc, 4)})
                sd = {k: v for k, v in model.state_dict().items() if not k.startswith('base.')}
                ck = {'mot_state': sd, 'cfg': vars(a), 'step': step, 'val_loss': vloss, 'val_acc': vacc}
                torch.save(ck, f'{out}/last.pt')
                torch.save(ck, f'{out}/step_{step}.pt')       # keep all for generation probes
                if vloss < best:
                    best = vloss; torch.save(ck, f'{out}/best.pt')
    log({'event': 'done', 'best_val': round(best, 4)})


if __name__ == '__main__':
    main()
