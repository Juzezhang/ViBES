"""
MotionGPT VQVAE architecture (ported for ViBES).
Keeps MotionGPT model structure while providing ViBES-friendly interfaces.
"""
from typing import Union

import torch
import torch.nn as nn
import torch.nn.functional as F


class nonlinearity(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        # swish
        return x * torch.sigmoid(x)


class ResConv1DBlock(nn.Module):
    def __init__(self, n_in, n_state, dilation=1, activation="silu", norm=None):
        super().__init__()
        padding = dilation
        self.norm = norm
        if norm == "LN":
            self.norm1 = nn.LayerNorm(n_in)
            self.norm2 = nn.LayerNorm(n_in)
        elif norm == "GN":
            self.norm1 = nn.GroupNorm(num_groups=32, num_channels=n_in, eps=1e-6, affine=True)
            self.norm2 = nn.GroupNorm(num_groups=32, num_channels=n_in, eps=1e-6, affine=True)
        elif norm == "BN":
            self.norm1 = nn.BatchNorm1d(num_features=n_in, eps=1e-6, affine=True)
            self.norm2 = nn.BatchNorm1d(num_features=n_in, eps=1e-6, affine=True)
        else:
            self.norm1 = nn.Identity()
            self.norm2 = nn.Identity()

        if activation == "relu":
            self.activation1 = nn.ReLU()
            self.activation2 = nn.ReLU()
        elif activation == "silu":
            self.activation1 = nonlinearity()
            self.activation2 = nonlinearity()
        elif activation == "gelu":
            self.activation1 = nn.GELU()
            self.activation2 = nn.GELU()
        else:
            self.activation1 = nn.ReLU()
            self.activation2 = nn.ReLU()

        self.conv1 = nn.Conv1d(n_in, n_state, 3, 1, padding, dilation)
        self.conv2 = nn.Conv1d(n_state, n_in, 1, 1, 0)

    def forward(self, x):
        x_orig = x
        if self.norm == "LN":
            x = self.norm1(x.transpose(-2, -1))
            x = self.activation1(x.transpose(-2, -1))
        else:
            x = self.norm1(x)
            x = self.activation1(x)

        x = self.conv1(x)

        if self.norm == "LN":
            x = self.norm2(x.transpose(-2, -1))
            x = self.activation2(x.transpose(-2, -1))
        else:
            x = self.norm2(x)
            x = self.activation2(x)

        x = self.conv2(x)
        x = x + x_orig
        return x


class Resnet1D(nn.Module):
    def __init__(
        self,
        n_in,
        n_depth,
        dilation_growth_rate=1,
        reverse_dilation=True,
        activation="relu",
        norm=None,
    ):
        super().__init__()
        blocks = [
            ResConv1DBlock(
                n_in,
                n_in,
                dilation=dilation_growth_rate**depth,
                activation=activation,
                norm=norm,
            )
            for depth in range(n_depth)
        ]
        if reverse_dilation:
            blocks = blocks[::-1]
        self.model = nn.Sequential(*blocks)

    def forward(self, x):
        return self.model(x)


class QuantizeEMAReset(nn.Module):
    def __init__(self, nb_code, code_dim, mu):
        super().__init__()
        self.nb_code = nb_code
        self.code_dim = code_dim
        self.mu = mu
        self.reset_codebook()

    def reset_codebook(self):
        self.init = False
        self.code_sum = None
        self.code_count = None
        self.register_buffer("codebook", torch.zeros(self.nb_code, self.code_dim))

    def _tile(self, x):
        nb_code_x, code_dim = x.shape
        if nb_code_x < self.nb_code:
            n_repeats = (self.nb_code + nb_code_x - 1) // nb_code_x
            std = 0.01 / (code_dim**0.5)
            out = x.repeat(n_repeats, 1)
            out = out + torch.randn_like(out) * std
        else:
            out = x
        return out

    def init_codebook(self, x):
        out = self._tile(x)
        self.codebook = out[: self.nb_code]
        self.code_sum = self.codebook.clone()
        self.code_count = torch.ones(self.nb_code, device=self.codebook.device)
        self.init = True

    @torch.no_grad()
    def compute_perplexity(self, code_idx):
        code_onehot = torch.zeros(self.nb_code, code_idx.shape[0], device=code_idx.device)
        code_onehot.scatter_(0, code_idx.view(1, code_idx.shape[0]), 1)
        code_count = code_onehot.sum(dim=-1)
        prob = code_count / torch.sum(code_count)
        perplexity = torch.exp(-torch.sum(prob * torch.log(prob + 1e-7)))
        return perplexity

    @torch.no_grad()
    def update_codebook(self, x, code_idx):
        code_onehot = torch.zeros(self.nb_code, x.shape[0], device=x.device)
        code_onehot.scatter_(0, code_idx.view(1, x.shape[0]), 1)
        code_sum = torch.matmul(code_onehot, x)
        code_count = code_onehot.sum(dim=-1)

        out = self._tile(x)
        code_rand = out[: self.nb_code]

        self.code_sum = self.mu * self.code_sum + (1.0 - self.mu) * code_sum
        self.code_count = self.mu * self.code_count + (1.0 - self.mu) * code_count

        usage = (self.code_count.view(self.nb_code, 1) >= 1.0).float()
        code_update = self.code_sum.view(self.nb_code, self.code_dim) / self.code_count.view(
            self.nb_code, 1
        )
        self.codebook = usage * code_update + (1 - usage) * code_rand
        prob = code_count / torch.sum(code_count)
        perplexity = torch.exp(-torch.sum(prob * torch.log(prob + 1e-7)))
        return perplexity

    def preprocess(self, x):
        # NCT -> NTC -> [NT, C]
        x = x.permute(0, 2, 1).contiguous()
        x = x.view(-1, x.shape[-1])
        return x

    def quantize(self, x):
        k_w = self.codebook.t()
        distance = (
            torch.sum(x**2, dim=-1, keepdim=True)
            - 2 * torch.matmul(x, k_w)
            + torch.sum(k_w**2, dim=0, keepdim=True)
        )
        _, code_idx = torch.min(distance, dim=-1)
        return code_idx

    def dequantize(self, code_idx):
        x = F.embedding(code_idx, self.codebook)
        return x

    def forward(self, x):
        n, width, t = x.shape
        x = self.preprocess(x)
        if self.training and not self.init:
            self.init_codebook(x)
        code_idx = self.quantize(x)
        x_d = self.dequantize(code_idx)

        if self.training:
            perplexity = self.update_codebook(x, code_idx)
        else:
            perplexity = self.compute_perplexity(code_idx)

        commit_loss = F.mse_loss(x, x_d.detach())
        x_d = x + (x_d - x).detach()
        x_d = x_d.view(n, t, -1).permute(0, 2, 1).contiguous()
        return x_d, commit_loss, perplexity


class Quantizer(nn.Module):
    def __init__(self, n_e, e_dim, beta):
        super().__init__()
        self.e_dim = e_dim
        self.n_e = n_e
        self.beta = beta
        self.embedding = nn.Embedding(self.n_e, self.e_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)

    def preprocess(self, x):
        x = x.permute(0, 2, 1).contiguous()
        x = x.view(-1, x.shape[-1])
        return x

    def quantize(self, z):
        d = (
            torch.sum(z**2, dim=1, keepdim=True)
            + torch.sum(self.embedding.weight**2, dim=1)
            - 2 * torch.matmul(z, self.embedding.weight.t())
        )
        min_encoding_indices = torch.argmin(d, dim=1)
        return min_encoding_indices

    def dequantize(self, indices):
        index_flattened = indices.view(-1)
        z_q = self.embedding(index_flattened)
        z_q = z_q.view(indices.shape + (self.e_dim,)).contiguous()
        return z_q

    def forward(self, z):
        n, width, t = z.shape
        z = self.preprocess(z)
        assert z.shape[-1] == self.e_dim
        z_flattened = z.contiguous().view(-1, self.e_dim)
        d = (
            torch.sum(z_flattened**2, dim=1, keepdim=True)
            + torch.sum(self.embedding.weight**2, dim=1)
            - 2 * torch.matmul(z_flattened, self.embedding.weight.t())
        )
        min_encoding_indices = torch.argmin(d, dim=1)
        z_q = self.embedding(min_encoding_indices).view(z.shape)
        loss = torch.mean((z_q - z.detach()) ** 2) + self.beta * torch.mean(
            (z_q.detach() - z) ** 2
        )
        z_q = z + (z_q - z).detach()
        z_q = z_q.view(n, t, -1).permute(0, 2, 1).contiguous()
        min_encodings = F.one_hot(min_encoding_indices, self.n_e).type(z.dtype)
        e_mean = torch.mean(min_encodings, dim=0)
        perplexity = torch.exp(-torch.sum(e_mean * torch.log(e_mean + 1e-10)))
        return z_q, loss, perplexity


class QuantizeReset(nn.Module):
    def __init__(self, nb_code, code_dim):
        super().__init__()
        self.nb_code = nb_code
        self.code_dim = code_dim
        self.reset_codebook()
        self.codebook = nn.Parameter(torch.randn(nb_code, code_dim))

    def reset_codebook(self):
        self.init = False
        self.code_count = None

    def _tile(self, x):
        nb_code_x, code_dim = x.shape
        if nb_code_x < self.nb_code:
            n_repeats = (self.nb_code + nb_code_x - 1) // nb_code_x
            std = 0.01 / (code_dim**0.5)
            out = x.repeat(n_repeats, 1)
            out = out + torch.randn_like(out) * std
        else:
            out = x
        return out

    def init_codebook(self, x):
        out = self._tile(x)
        self.codebook = nn.Parameter(out[: self.nb_code])
        self.code_count = torch.ones(self.nb_code, device=self.codebook.device)
        self.init = True

    @torch.no_grad()
    def compute_perplexity(self, code_idx):
        code_onehot = torch.zeros(self.nb_code, code_idx.shape[0], device=code_idx.device)
        code_onehot.scatter_(0, code_idx.view(1, code_idx.shape[0]), 1)
        code_count = code_onehot.sum(dim=-1)
        prob = code_count / torch.sum(code_count)
        perplexity = torch.exp(-torch.sum(prob * torch.log(prob + 1e-7)))
        return perplexity

    @torch.no_grad()
    def update_codebook(self, x, code_idx):
        code_onehot = torch.zeros(self.nb_code, x.shape[0], device=x.device)
        code_onehot.scatter_(0, code_idx.view(1, x.shape[0]), 1)
        code_count = code_onehot.sum(dim=-1)
        out = self._tile(x)
        code_rand = out[: self.nb_code]
        self.code_count = code_count
        usage = (self.code_count.view(self.nb_code, 1) >= 1.0).float()
        self.codebook.data = usage * self.codebook.data + (1 - usage) * code_rand
        prob = code_count / torch.sum(code_count)
        perplexity = torch.exp(-torch.sum(prob * torch.log(prob + 1e-7)))
        return perplexity

    def preprocess(self, x):
        x = x.permute(0, 2, 1).contiguous()
        x = x.view(-1, x.shape[-1])
        return x

    def quantize(self, x):
        k_w = self.codebook.t()
        distance = (
            torch.sum(x**2, dim=-1, keepdim=True)
            - 2 * torch.matmul(x, k_w)
            + torch.sum(k_w**2, dim=0, keepdim=True)
        )
        _, code_idx = torch.min(distance, dim=-1)
        return code_idx

    def dequantize(self, code_idx):
        x = F.embedding(code_idx, self.codebook)
        return x

    def forward(self, x):
        n, width, t = x.shape
        x = self.preprocess(x)
        if self.training and not self.init:
            self.init_codebook(x)
        code_idx = self.quantize(x)
        x_d = self.dequantize(code_idx)
        if self.training:
            perplexity = self.update_codebook(x, code_idx)
        else:
            perplexity = self.compute_perplexity(code_idx)
        commit_loss = F.mse_loss(x, x_d.detach())
        x_d = x + (x_d - x).detach()
        x_d = x_d.view(n, t, -1).permute(0, 2, 1).contiguous()
        return x_d, commit_loss, perplexity


class QuantizeEMA(nn.Module):
    def __init__(self, nb_code, code_dim, mu):
        super().__init__()
        self.nb_code = nb_code
        self.code_dim = code_dim
        self.mu = mu
        self.reset_codebook()

    def reset_codebook(self):
        self.init = False
        self.code_sum = None
        self.code_count = None
        self.register_buffer("codebook", torch.zeros(self.nb_code, self.code_dim))

    def _tile(self, x):
        nb_code_x, code_dim = x.shape
        if nb_code_x < self.nb_code:
            n_repeats = (self.nb_code + nb_code_x - 1) // nb_code_x
            std = 0.01 / (code_dim**0.5)
            out = x.repeat(n_repeats, 1)
            out = out + torch.randn_like(out) * std
        else:
            out = x
        return out

    def init_codebook(self, x):
        out = self._tile(x)
        self.codebook = out[: self.nb_code]
        self.code_sum = self.codebook.clone()
        self.code_count = torch.ones(self.nb_code, device=self.codebook.device)
        self.init = True

    @torch.no_grad()
    def compute_perplexity(self, code_idx):
        code_onehot = torch.zeros(self.nb_code, code_idx.shape[0], device=code_idx.device)
        code_onehot.scatter_(0, code_idx.view(1, code_idx.shape[0]), 1)
        code_count = code_onehot.sum(dim=-1)
        prob = code_count / torch.sum(code_count)
        perplexity = torch.exp(-torch.sum(prob * torch.log(prob + 1e-7)))
        return perplexity

    @torch.no_grad()
    def update_codebook(self, x, code_idx):
        code_onehot = torch.zeros(self.nb_code, x.shape[0], device=x.device)
        code_onehot.scatter_(0, code_idx.view(1, x.shape[0]), 1)
        code_sum = torch.matmul(code_onehot, x)
        code_count = code_onehot.sum(dim=-1)
        self.code_sum = self.mu * self.code_sum + (1.0 - self.mu) * code_sum
        self.code_count = self.mu * self.code_count + (1.0 - self.mu) * code_count
        code_update = self.code_sum.view(self.nb_code, self.code_dim) / self.code_count.view(
            self.nb_code, 1
        )
        self.codebook = code_update
        prob = code_count / torch.sum(code_count)
        perplexity = torch.exp(-torch.sum(prob * torch.log(prob + 1e-7)))
        return perplexity

    def preprocess(self, x):
        x = x.permute(0, 2, 1).contiguous()
        x = x.view(-1, x.shape[-1])
        return x

    def quantize(self, x):
        k_w = self.codebook.t()
        distance = (
            torch.sum(x**2, dim=-1, keepdim=True)
            - 2 * torch.matmul(x, k_w)
            + torch.sum(k_w**2, dim=0, keepdim=True)
        )
        _, code_idx = torch.min(distance, dim=-1)
        return code_idx

    def dequantize(self, code_idx):
        x = F.embedding(code_idx, self.codebook)
        return x

    def forward(self, x):
        n, width, t = x.shape
        x = self.preprocess(x)
        if self.training and not self.init:
            self.init_codebook(x)
        code_idx = self.quantize(x)
        x_d = self.dequantize(code_idx)
        if self.training:
            perplexity = self.update_codebook(x, code_idx)
        else:
            perplexity = self.compute_perplexity(code_idx)
        commit_loss = F.mse_loss(x, x_d.detach())
        x_d = x + (x_d - x).detach()
        x_d = x_d.view(n, t, -1).permute(0, 2, 1).contiguous()
        return x_d, commit_loss, perplexity

class Encoder(nn.Module):
    def __init__(
        self,
        input_emb_width=3,
        output_emb_width=512,
        down_t=3,
        stride_t=2,
        width=512,
        depth=3,
        dilation_growth_rate=3,
        activation="relu",
        norm=None,
    ):
        super().__init__()
        blocks = []
        filter_t, pad_t = stride_t * 2, stride_t // 2
        blocks.append(nn.Conv1d(input_emb_width, width, 3, 1, 1))
        blocks.append(nn.ReLU())

        for _ in range(down_t):
            block = nn.Sequential(
                nn.Conv1d(width, width, filter_t, stride_t, pad_t),
                Resnet1D(
                    width,
                    depth,
                    dilation_growth_rate,
                    activation=activation,
                    norm=norm,
                ),
            )
            blocks.append(block)
        blocks.append(nn.Conv1d(width, output_emb_width, 3, 1, 1))
        self.model = nn.Sequential(*blocks)

    def forward(self, x):
        return self.model(x)


class Decoder(nn.Module):
    def __init__(
        self,
        input_emb_width=3,
        output_emb_width=512,
        down_t=3,
        stride_t=2,
        width=512,
        depth=3,
        dilation_growth_rate=3,
        activation="relu",
        norm=None,
    ):
        super().__init__()
        blocks = []
        blocks.append(nn.Conv1d(output_emb_width, width, 3, 1, 1))
        blocks.append(nn.ReLU())
        for _ in range(down_t):
            block = nn.Sequential(
                Resnet1D(
                    width,
                    depth,
                    dilation_growth_rate,
                    reverse_dilation=True,
                    activation=activation,
                    norm=norm,
                ),
                nn.Upsample(scale_factor=2, mode="nearest"),
                nn.Conv1d(width, width, 3, 1, 1),
            )
            blocks.append(block)
        blocks.append(nn.Conv1d(width, width, 3, 1, 1))
        blocks.append(nn.ReLU())
        blocks.append(nn.Conv1d(width, input_emb_width, 3, 1, 1))
        self.model = nn.Sequential(*blocks)

    def forward(self, x):
        return self.model(x)


class MotionGPTVQVae(nn.Module):
    def __init__(
        self,
        nfeats: int,
        quantizer: str = "ema_reset",
        code_num: int = 512,
        code_dim: int = 512,
        output_emb_width: int = 512,
        down_t: int = 3,
        stride_t: int = 2,
        width: int = 512,
        depth: int = 3,
        dilation_growth_rate: int = 3,
        norm: str | None = None,
        activation: str = "relu",
        mu: float = 0.99,  # EMA decay parameter
    ) -> None:
        super().__init__()
        self.code_dim = code_dim
        self.encoder = Encoder(
            nfeats,
            output_emb_width,
            down_t,
            stride_t,
            width,
            depth,
            dilation_growth_rate,
            activation=activation,
            norm=norm,
        )
        self.decoder = Decoder(
            nfeats,
            output_emb_width,
            down_t,
            stride_t,
            width,
            depth,
            dilation_growth_rate,
            activation=activation,
            norm=norm,
        )

        if quantizer == "ema_reset":
            self.quantizer = QuantizeEMAReset(code_num, code_dim, mu=mu)
        elif quantizer == "orig":
            self.quantizer = Quantizer(code_num, code_dim, beta=1.0)
        elif quantizer == "ema":
            self.quantizer = QuantizeEMA(code_num, code_dim, mu=mu)
        elif quantizer == "reset":
            self.quantizer = QuantizeReset(code_num, code_dim)
        else:
            self.quantizer = QuantizeEMAReset(code_num, code_dim, mu=mu)

    def preprocess(self, x):
        x = x.permute(0, 2, 1)
        return x

    def postprocess(self, x):
        x = x.permute(0, 2, 1)
        return x

    def forward(self, features: torch.Tensor):
        x_in = self.preprocess(features)
        x_encoder = self.encoder(x_in)
        x_quantized, loss, perplexity = self.quantizer(x_encoder)
        x_decoder = self.decoder(x_quantized)
        x_out = self.postprocess(x_decoder)
        x_q = self.postprocess(x_quantized)
        return x_out, loss, perplexity, x_q

    def encode(self, features: torch.Tensor) -> Union[torch.Tensor, None]:
        n, t, _ = features.shape
        x_in = self.preprocess(features)
        x_encoder = self.encoder(x_in)
        x_encoder = self.postprocess(x_encoder)
        x_encoder = x_encoder.contiguous().view(-1, x_encoder.shape[-1])
        code_idx = self.quantizer.quantize(x_encoder)
        code_idx = code_idx.view(n, -1)
        return code_idx, None

    def decode(self, z: torch.Tensor):
        x_d = self.quantizer.dequantize(z)
        x_d = x_d.view(1, -1, self.code_dim).permute(0, 2, 1).contiguous()
        x_decoder = self.decoder(x_d)
        x_out = self.postprocess(x_decoder)
        return x_out


class MotionGPTVQVaeAdapter(nn.Module):
    """
    Adapter to match ViBES tokenizer interface.
    Returns dict with rec_pose/embedding_loss/perplexity and map2index/decode helpers.
    """

    def __init__(self, **kwargs):
        super().__init__()
        # Allow configs to pass ViBES-style aliases without breaking MotionGPT init
        kwargs.pop("vae_test_dim", None)
        self.vqvae = MotionGPTVQVae(**kwargs)

    def forward(self, inputs: torch.Tensor):
        rec_pose, embedding_loss, perplexity, poses_feat = self.vqvae(inputs)
        return {
            "poses_feat": poses_feat,
            "embedding_loss": embedding_loss,
            "perplexity": perplexity,
            "rec_pose": rec_pose,
        }

    def map2index(self, inputs: torch.Tensor):
        code_idx, _ = self.vqvae.encode(inputs)
        return code_idx

    def map2latent(self, inputs: torch.Tensor):
        code_idx, _ = self.vqvae.encode(inputs)
        return self.vqvae.quantizer.dequantize(code_idx)

    def decode(self, index: torch.Tensor):
        # Support batched indices: (B, L)
        z_q = self.vqvae.quantizer.dequantize(index)
        z_q = z_q.permute(0, 2, 1).contiguous()
        x_decoder = self.vqvae.decoder(z_q)
        return self.vqvae.postprocess(x_decoder)
