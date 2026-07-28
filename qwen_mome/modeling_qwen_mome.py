"""Qwen-MoME: ViBES AR-MoME (2-expert masked-interleaved) on a frozen Qwen3 backbone, T2M-only.

Port of ChatGLMForConditionalGenerationMotExpertNum2's structure (see
the ViBES AR-MoME design) onto Qwen3-0.6B:
  - Expert-0 = the frozen Qwen3 (text): embeddings, per-layer attn/MLP, norms — untouched.
  - Expert-1 = NARROW trainable motion expert (ViBES-faithful d_mot=512, ffn_mot=4096):
    per-layer q/k/v/o_mot + q/k_norm_mot + norms_mot + SwiGLU mlp_mot, plus embed_mot(514),
    final norm_mot and untied lm_head_mot(514).
  - Experts meet ONLY in shared attention geometry (16Q/8KV heads, head_dim 128): per-expert
    QKV are scatter-merged into full-length tensors at original positions, RoPE (Qwen3 rotate_half,
    plain arange positions for T2M — NO fps interpolation) applied post-merge, one joint SDPA with
    the ViBES visibility rule: text attends causally to text only; motion attends causally to
    text+motion. Text never sees motion => freezing Expert-0 is exact.
  - Loss: motion stream only (labels -100 elsewhere), full-sequence shift (text is a contiguous
    prefix in T2M). Pad labels are -100 (fixes the ViBES pad-label-0 quirk).
Motion vocab: 514 raw expert ids (codes 0..511, begin=512, end=513).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM
from transformers.models.qwen3.modeling_qwen3 import Qwen3RMSNorm, apply_rotary_pos_emb, repeat_kv

MOT_VOCAB = 514
BEGIN, END = 512, 513


class MotMLP(nn.Module):  # SwiGLU, Qwen-style separate gate/up/down, no bias
    def __init__(self, d, ffn):
        super().__init__()
        self.gate_proj = nn.Linear(d, ffn, bias=False)
        self.up_proj = nn.Linear(d, ffn, bias=False)
        self.down_proj = nn.Linear(ffn, d, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class MotExpertLayer(nn.Module):
    """Trainable motion-side modules for one layer (mirrors Qwen3DecoderLayer shapes)."""
    def __init__(self, d_mot, ffn_mot, n_q, n_kv, head_dim, eps):
        super().__init__()
        self.q_proj = nn.Linear(d_mot, n_q * head_dim, bias=False)
        self.k_proj = nn.Linear(d_mot, n_kv * head_dim, bias=False)
        self.v_proj = nn.Linear(d_mot, n_kv * head_dim, bias=False)
        self.o_proj = nn.Linear(n_q * head_dim, d_mot, bias=False)
        self.q_norm = Qwen3RMSNorm(head_dim, eps=eps)
        self.k_norm = Qwen3RMSNorm(head_dim, eps=eps)
        self.input_layernorm = Qwen3RMSNorm(d_mot, eps=eps)
        self.post_attention_layernorm = Qwen3RMSNorm(d_mot, eps=eps)
        self.mlp = MotMLP(d_mot, ffn_mot)


class QwenMoMEForT2M(nn.Module):
    def __init__(self, base_repo="Qwen/Qwen3-0.6B", d_mot=512, ffn_mot=4096, dtype=torch.bfloat16, mot_vocab=MOT_VOCAB):
        super().__init__()
        base_lm = AutoModelForCausalLM.from_pretrained(base_repo, dtype=dtype)
        self.base = base_lm.model            # Qwen3Model (embed_tokens, layers, norm, rotary_emb)
        del base_lm                          # frozen head never used
        for p in self.base.parameters():
            p.requires_grad_(False)
        cfg = self.base.config
        self.cfg = cfg
        self.d_mot, self.head_dim = d_mot, cfg.head_dim
        self.n_q, self.n_kv = cfg.num_attention_heads, cfg.num_key_value_heads
        self.mot_vocab = mot_vocab; self.BEGIN = mot_vocab - 2; self.END = mot_vocab - 1
        self.embed_mot = nn.Embedding(mot_vocab, d_mot)
        self.mot_layers = nn.ModuleList(
            MotExpertLayer(d_mot, ffn_mot, self.n_q, self.n_kv, cfg.head_dim, cfg.rms_norm_eps)
            for _ in range(cfg.num_hidden_layers))
        self.norm_mot = Qwen3RMSNorm(d_mot, eps=cfg.rms_norm_eps)
        self.lm_head_mot = nn.Linear(d_mot, self.mot_vocab, bias=False)
        self._init_mot()
        self.mot_dtype = dtype
        for m in [self.embed_mot, self.mot_layers, self.norm_mot, self.lm_head_mot]:
            m.to(dtype)

    def _init_mot(self):
        for m in self.modules():
            if isinstance(m, nn.Linear) and m.weight.requires_grad:
                nn.init.normal_(m.weight, std=0.02)
        nn.init.normal_(self.embed_mot.weight, std=0.02)

    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]

    @staticmethod
    def _heads(x, n, hd):  # [B,S,n*hd] -> [B,n,S,hd]
        B, S, _ = x.shape
        return x.view(B, S, n, hd).transpose(1, 2)

    def _visibility(self, mot_mask, attn_pad):
        """[B,S] motion-position mask (+[B,S] padding 1/0) -> bool sdpa mask [B,1,S,S], True=attend.
        text q -> text k (causal); motion q -> text+motion k (causal)."""
        B, S = mot_mask.shape
        causal = torch.tril(torch.ones(S, S, dtype=torch.bool, device=mot_mask.device))
        q_text = ~mot_mask
        k_mot = mot_mask
        allowed = causal.unsqueeze(0) & ~(q_text.unsqueeze(2) & k_mot.unsqueeze(1))
        if attn_pad is not None:
            allowed = allowed & attn_pad.bool().unsqueeze(1)          # mask padded keys
        # never leave a fully-masked row (pad rows / empty-text CFG samples): allow self-attention.
        # Pad outputs are excluded from loss, but an all-masked SDPA row yields NaN that poisons
        # shared projection grads in backward.
        eye = torch.eye(S, dtype=torch.bool, device=mot_mask.device).unsqueeze(0)
        allowed = allowed | eye
        return allowed.unsqueeze(1)

    def forward(self, input_ids, modality_mot, attention_mask=None, position_ids=None, labels=None):
        """input_ids [B,S]: text ids where ~modality_mot, RAW motion ids (0..513) where modality_mot.
        modality_mot [B,S] bool. position_ids [B,S] (plain arange for T2M)."""
        B, S = input_ids.shape
        dev = input_ids.device
        if position_ids is None:
            position_ids = torch.arange(S, device=dev).unsqueeze(0).expand(B, S)
        mm = modality_mot.bool()
        # --- dual-stream embeddings (parked zeros at the other expert's positions) ---
        h_t = self.base.embed_tokens(torch.where(mm, torch.zeros_like(input_ids), input_ids))
        h_t = h_t.masked_fill(mm.unsqueeze(-1), 0)
        h_m = self.embed_mot(torch.where(mm, input_ids, torch.zeros_like(input_ids)))
        h_m = h_m.masked_fill(~mm.unsqueeze(-1), 0)
        cos, sin = self.base.rotary_emb(h_t, position_ids)
        vis = self._visibility(mm, attention_mask)
        mmE = mm.unsqueeze(-1)
        for base_l, mot_l in zip(self.base.layers, self.mot_layers):
            # -- attention --
            nt = base_l.input_layernorm(h_t)
            nm = mot_l.input_layernorm(h_m)
            a = base_l.self_attn
            q = self._heads(a.q_proj(nt), self.n_q, self.head_dim)
            k = self._heads(a.k_proj(nt), self.n_kv, self.head_dim)
            v = self._heads(a.v_proj(nt), self.n_kv, self.head_dim)
            q, k = a.q_norm(q), a.k_norm(k)
            qm = self._heads(mot_l.q_proj(nm), self.n_q, self.head_dim)
            km = self._heads(mot_l.k_proj(nm), self.n_kv, self.head_dim)
            vm = self._heads(mot_l.v_proj(nm), self.n_kv, self.head_dim)
            qm, km = mot_l.q_norm(qm), mot_l.k_norm(km)
            sel = mm.unsqueeze(1).unsqueeze(-1)                       # [B,1,S,1]
            q = torch.where(sel, qm, q)                               # scatter-merge at positions
            k = torch.where(sel, km, k)
            v = torch.where(sel, vm, v)
            q, k = apply_rotary_pos_emb(q, k, cos, sin)               # RoPE post-merge, abs positions
            k = repeat_kv(k, self.n_q // self.n_kv)
            v = repeat_kv(v, self.n_q // self.n_kv)
            ctx = F.scaled_dot_product_attention(q, k, v, attn_mask=vis)
            ctx = ctx.transpose(1, 2).reshape(B, S, self.n_q * self.head_dim)
            h_t = h_t + a.o_proj(ctx).masked_fill(mmE, 0)
            h_m = h_m + mot_l.o_proj(ctx).masked_fill(~mmE, 0)
            # -- MLP --
            h_t = h_t + base_l.mlp(base_l.post_attention_layernorm(h_t)).masked_fill(mmE, 0)
            h_m = h_m + mot_l.mlp(mot_l.post_attention_layernorm(h_m)).masked_fill(~mmE, 0)
        logits = self.lm_head_mot(self.norm_mot(h_m))                 # [B,S,514] (motion rows valid)
        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits[:, :-1].reshape(-1, self.mot_vocab).float(),
                                   labels[:, 1:].reshape(-1), ignore_index=-100)
        return {"loss": loss, "logits": logits}

    @torch.no_grad()
    def generate_motion(self, text_ids, max_new=60, greedy=True, temperature=1.0, top_p=0.9, cfg_scale=0.0):
        """text_ids [1,T] -> list of motion codes (0..511). Appends BEGIN then decodes to END.
        Full recompute per step (short seqs; correctness first)."""
        dev = text_ids.device
        ids = torch.cat([text_ids, torch.tensor([[self.BEGIN]], device=dev)], dim=1)
        mm = torch.zeros_like(ids, dtype=torch.bool)
        mm[0, -1] = True
        T = text_ids.shape[1]
        codes = []
        for _ in range(max_new):
            out = self.forward(ids, mm)
            logit = out["logits"][0, -1].float()
            if cfg_scale > 0:                      # classifier-free guidance: uncond = motion-only prefix
                u = self.forward(ids[:, T:], mm[:, T:])
                logit = (1 + cfg_scale) * logit - cfg_scale * u["logits"][0, -1].float()
            if greedy:
                nxt = int(logit.argmax())
            else:
                p = torch.softmax(logit / temperature, -1)
                sp, si = torch.sort(p, descending=True)
                keep = torch.cumsum(sp, -1) - sp < top_p
                p = torch.zeros_like(p).scatter(0, si[keep], sp[keep])
                nxt = int(torch.multinomial(p / p.sum(), 1))
            if nxt == self.END:
                break
            if nxt < self.mot_vocab - 2:
                codes.append(nxt)
            ids = torch.cat([ids, torch.tensor([[nxt]], device=dev)], dim=1)
            mm = torch.cat([mm, torch.ones(1, 1, dtype=torch.bool, device=dev)], dim=1)
        return codes
