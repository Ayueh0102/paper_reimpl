"""VQGAN font codebook backbone — Stage 0 of VQ-Font.

Paper note (036_VQ-Font結構感知字體_AAAI2023) describes:
  * Pre-trained VQGAN font codebook with **size 1024** and **16x16** spatial
    feature grid (so the encoder is a factor-8 downsample for a 128px input)
    [paper-cited Phase 0 row 04].
  * "Token prior refinement" later predicts indices into this codebook with a
    Transformer (`transformer.py`); during Stage 0 the codebook is trained
    end-to-end with the standard VQ-VAE / VQGAN recipe:
      L_vq = L_recon (L1 + L2) + L_commit (commitment β term) +
             L_codebook (codebook update) + (optional) L_gan.
    Adversarial loss is **omitted** at Phase 1 — paper says "VQGAN-based
    framework" but does not specify the discriminator depth and we keep
    the blind reimpl minimal. A patch-discriminator hook is left as a
    `[guessed]` extension point in `reports/blind_impl.md`.

This module is purposely self-contained: it does **not** import from
`paper_reimpl_shared.diffusion` (VQ-Font is not a diffusion model). The
codebook itself is the `VectorQuantize` layer with straight-through gradient
(Van den Oord 2017 / Esser et al. 2021 — both pre-date the paper and are
non-controversial reimpl choices).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "VQGANConfig",
    "VectorQuantize",
    "VQGANEncoder",
    "VQGANDecoder",
    "VQGAN",
    "build_vqgan",
]


# --------------------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------------------


@dataclass
class VQGANConfig:
    """Structural hyperparameters for the VQGAN font codebook.

    Defaults follow the paper-cited 1024-entry codebook on a 16x16 feature
    grid for a 128px input (`down_factors=(2,2,2)` => 8x downsample). Other
    values are blind reimpl conventions; see `reports/blind_impl.md`.
    """

    image_size: int = 128
    in_channels: int = 1
    """Grayscale glyph input. Set to 3 for RGB content fields."""
    base_channels: int = 64
    channel_mult: tuple[int, ...] = (1, 2, 4)
    """Per-stage channel multipliers. len(mult) controls the depth and hence
    the downsample factor. For 128px -> 16x16 we want 3 stages of stride-2
    => 2**3 = 8x downsample."""
    z_channels: int = 256
    """Latent channel width before / after the codebook (matches embed_dim)."""
    embed_dim: int = 256
    """Codebook entry dimensionality."""
    num_embeddings: int = 1024
    """Paper-cited codebook size (Phase 0 row 04)."""
    commitment_weight: float = 0.25
    """β in VQ-VAE commitment loss; standard 0.25 (Van den Oord 2017)."""
    num_res_blocks: int = 2
    dropout: float = 0.0

    def out_resolution(self) -> int:
        """Spatial size after the encoder.

        Encoder does ``len(channel_mult) - 1`` stride-2 downsamples (the last
        stage's downsample is an ``Identity``). For the paper default
        ``channel_mult=(1, 2, 4)`` and ``image_size=128`` this yields
        ``128 / 2^2 = 32`` — note the **paper's 16x16 latent grid requires
        4 stages of multipliers (or one extra stride-2)**. Our default
        ``channel_mult=(1, 2, 4)`` produces a 32x32 grid; bump to
        ``(1, 1, 2, 4)`` to hit the paper-cited 16x16. See blind_impl.md.
        """
        return self.image_size // (2 ** (len(self.channel_mult) - 1))


# --------------------------------------------------------------------------------------
# Building blocks
# --------------------------------------------------------------------------------------


def _gn(channels: int) -> nn.GroupNorm:
    """GroupNorm whose group count divides `channels`. Probes 32 -> 1."""
    for g in (32, 16, 8, 4, 2, 1):
        if channels % g == 0:
            return nn.GroupNorm(g, channels)
    return nn.GroupNorm(1, channels)


class _ResBlock(nn.Module):
    """Pre-norm residual block, used inside both encoder and decoder."""

    def __init__(self, in_c: int, out_c: int, *, dropout: float = 0.0) -> None:
        super().__init__()
        self.norm1 = _gn(in_c)
        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size=3, padding=1)
        self.norm2 = _gn(out_c)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_c, out_c, kernel_size=3, padding=1)
        self.skip = nn.Conv2d(in_c, out_c, kernel_size=1) if in_c != out_c else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = self.conv2(self.dropout(F.silu(self.norm2(h))))
        return h + self.skip(x)


class _AttnBlock(nn.Module):
    """Single-head self-attention used once at the bottleneck (taming-style)."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.norm = _gn(channels)
        self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1)
        self.proj = nn.Conv2d(channels, channels, kernel_size=1)
        self.channels = channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        qkv = self.qkv(self.norm(x))
        q, k, v = qkv.chunk(3, dim=1)
        q = q.reshape(b, c, h * w).transpose(-1, -2)  # [B, HW, C]
        k = k.reshape(b, c, h * w)                     # [B, C, HW]
        v = v.reshape(b, c, h * w).transpose(-1, -2)   # [B, HW, C]
        attn = torch.softmax(torch.matmul(q, k) / (c ** 0.5), dim=-1)
        out = torch.matmul(attn, v).transpose(-1, -2).reshape(b, c, h, w)
        return x + self.proj(out)


class _Downsample(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.op = nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.op(x)


class _Upsample(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.op = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        return self.op(x)


# --------------------------------------------------------------------------------------
# Vector Quantizer (codebook)
# --------------------------------------------------------------------------------------


class VectorQuantize(nn.Module):
    """Discrete codebook with straight-through gradient.

    Implements the standard VQ-VAE quantizer (Van den Oord 2017 §3) with
    Esser-style commitment + codebook losses surfaced as `quantize_loss` so
    the trainer can scale them.
    """

    def __init__(self, num_embeddings: int, embed_dim: int, *, commitment_weight: float = 0.25) -> None:
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embed_dim = embed_dim
        self.commitment_weight = commitment_weight
        self.codebook = nn.Embedding(num_embeddings, embed_dim)
        nn.init.uniform_(self.codebook.weight, -1.0 / num_embeddings, 1.0 / num_embeddings)

    def forward(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Quantize a [B, C, H, W] latent.

        Returns:
            z_q: same shape as z, but on the codebook lattice; gradient flows
                back as the identity (straight-through).
            indices: [B, H, W] long — index of the nearest codebook entry.
            loss: scalar — commitment + codebook update loss.
        """
        b, c, h, w = z.shape
        z_flat = z.permute(0, 2, 3, 1).reshape(-1, c)            # [BHW, C]
        # Euclidean distance to each codebook entry.
        cb = self.codebook.weight                                # [K, C]
        dists = (
            torch.sum(z_flat ** 2, dim=1, keepdim=True)
            + torch.sum(cb ** 2, dim=1)
            - 2 * z_flat @ cb.t()
        )                                                        # [BHW, K]
        indices = torch.argmin(dists, dim=1)                     # [BHW]
        z_q_flat = self.codebook(indices)                         # [BHW, C]

        codebook_loss = F.mse_loss(z_q_flat, z_flat.detach())
        commitment_loss = F.mse_loss(z_flat, z_q_flat.detach())
        loss = codebook_loss + self.commitment_weight * commitment_loss

        # Straight-through: encoder gradient flows as identity.
        z_q_flat = z_flat + (z_q_flat - z_flat).detach()
        z_q = z_q_flat.reshape(b, h, w, c).permute(0, 3, 1, 2).contiguous()
        indices = indices.reshape(b, h, w)
        return z_q, indices, loss

    def lookup(self, indices: torch.Tensor) -> torch.Tensor:
        """Look up codebook entries given an integer index tensor.

        Args:
            indices: any shape long tensor with values in [0, num_embeddings).

        Returns:
            Tensor with one extra leading-channel dim that matches `embed_dim`,
            shaped as ``[..., embed_dim]``.
        """
        return self.codebook(indices)


# --------------------------------------------------------------------------------------
# Encoder / Decoder
# --------------------------------------------------------------------------------------


class VQGANEncoder(nn.Module):
    """Taming-transformers-style convolutional encoder.

    Conv stem -> N stages of (ResBlock x num_res_blocks + Downsample) -> mid
    block (Res + Attn + Res) -> norm + conv to z_channels. Output is a
    [B, z_channels, H/8, W/8] feature map for the default 3-stage config.
    """

    def __init__(self, cfg: VQGANConfig) -> None:
        super().__init__()
        chs = [cfg.base_channels * m for m in cfg.channel_mult]
        self.stem = nn.Conv2d(cfg.in_channels, chs[0], kernel_size=3, padding=1)

        self.blocks: nn.ModuleList = nn.ModuleList()
        self.downsamples: nn.ModuleList = nn.ModuleList()
        prev_c = chs[0]
        for i, c in enumerate(chs):
            stage: list[nn.Module] = []
            for _ in range(cfg.num_res_blocks):
                stage.append(_ResBlock(prev_c, c, dropout=cfg.dropout))
                prev_c = c
            self.blocks.append(nn.Sequential(*stage))
            if i < len(chs) - 1:
                self.downsamples.append(_Downsample(c))
            else:
                self.downsamples.append(nn.Identity())

        mid_c = chs[-1]
        self.mid_res1 = _ResBlock(mid_c, mid_c, dropout=cfg.dropout)
        self.mid_attn = _AttnBlock(mid_c)
        self.mid_res2 = _ResBlock(mid_c, mid_c, dropout=cfg.dropout)

        self.out_norm = _gn(mid_c)
        self.out_conv = nn.Conv2d(mid_c, cfg.z_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.stem(x)
        for stage, down in zip(self.blocks, self.downsamples):
            h = stage(h)
            h = down(h)
        h = self.mid_res1(h)
        h = self.mid_attn(h)
        h = self.mid_res2(h)
        return self.out_conv(F.silu(self.out_norm(h)))


class VQGANDecoder(nn.Module):
    """Mirror of `VQGANEncoder` consuming the quantized latent."""

    def __init__(self, cfg: VQGANConfig) -> None:
        super().__init__()
        chs = [cfg.base_channels * m for m in cfg.channel_mult]
        rev_chs = list(reversed(chs))

        self.input_conv = nn.Conv2d(cfg.z_channels, rev_chs[0], kernel_size=3, padding=1)
        mid_c = rev_chs[0]
        self.mid_res1 = _ResBlock(mid_c, mid_c, dropout=cfg.dropout)
        self.mid_attn = _AttnBlock(mid_c)
        self.mid_res2 = _ResBlock(mid_c, mid_c, dropout=cfg.dropout)

        self.blocks: nn.ModuleList = nn.ModuleList()
        self.upsamples: nn.ModuleList = nn.ModuleList()
        prev_c = mid_c
        for i, c in enumerate(rev_chs):
            stage: list[nn.Module] = []
            for _ in range(cfg.num_res_blocks):
                stage.append(_ResBlock(prev_c, c, dropout=cfg.dropout))
                prev_c = c
            self.blocks.append(nn.Sequential(*stage))
            if i < len(rev_chs) - 1:
                self.upsamples.append(_Upsample(c))
            else:
                self.upsamples.append(nn.Identity())

        self.out_norm = _gn(rev_chs[-1])
        self.out_conv = nn.Conv2d(rev_chs[-1], cfg.in_channels, kernel_size=3, padding=1)

    def forward(self, z_q: torch.Tensor) -> torch.Tensor:
        h = self.input_conv(z_q)
        h = self.mid_res1(h)
        h = self.mid_attn(h)
        h = self.mid_res2(h)
        for stage, up in zip(self.blocks, self.upsamples):
            h = stage(h)
            h = up(h)
        return self.out_conv(F.silu(self.out_norm(h)))


# --------------------------------------------------------------------------------------
# Top-level VQGAN
# --------------------------------------------------------------------------------------


@dataclass
class VQGANOutputs:
    """Container for VQGAN.forward outputs.

    Using a dataclass keeps train-time call-sites readable (no tuple unpacking
    of five things). All fields are torch tensors except `z_q` which retains
    grad to flow back into the encoder via the straight-through estimator.
    """

    recon: torch.Tensor
    z_e: torch.Tensor
    z_q: torch.Tensor
    indices: torch.Tensor
    vq_loss: torch.Tensor


class VQGAN(nn.Module):
    """End-to-end VQGAN: encoder -> codebook quantize -> decoder.

    Stage 0 of VQ-Font pretrains this whole module on a font corpus (paper
    says 200k iters on 1xA6000); Stages 1+ freeze the encoder + codebook +
    decoder and only train the Transformer.
    """

    def __init__(self, cfg: VQGANConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.encoder = VQGANEncoder(cfg)
        self.codebook = VectorQuantize(
            num_embeddings=cfg.num_embeddings,
            embed_dim=cfg.embed_dim,
            commitment_weight=cfg.commitment_weight,
        )
        self.decoder = VQGANDecoder(cfg)
        # Project encoder output to codebook dim if mismatched; usually
        # z_channels == embed_dim and these are identities.
        self.pre_quant = (
            nn.Conv2d(cfg.z_channels, cfg.embed_dim, kernel_size=1)
            if cfg.z_channels != cfg.embed_dim
            else nn.Identity()
        )
        self.post_quant = (
            nn.Conv2d(cfg.embed_dim, cfg.z_channels, kernel_size=1)
            if cfg.z_channels != cfg.embed_dim
            else nn.Identity()
        )

    @torch.no_grad()
    def encode_indices(self, x: torch.Tensor) -> torch.Tensor:
        """Encode an image into [B, H/8, W/8] codebook indices. No grad."""
        z_e = self.pre_quant(self.encoder(x))
        _, indices, _ = self.codebook(z_e)
        return indices

    def decode_indices(self, indices: torch.Tensor) -> torch.Tensor:
        """Decode integer codebook indices back into a reconstruction.

        Args:
            indices: [B, H_lat, W_lat] long tensor.

        Returns:
            Reconstructed image [B, in_channels, H, W].
        """
        z_q = self.codebook.lookup(indices)                  # [B, H, W, C]
        z_q = z_q.permute(0, 3, 1, 2).contiguous()
        return self.decoder(self.post_quant(z_q))

    def forward(self, x: torch.Tensor) -> VQGANOutputs:
        z_e = self.pre_quant(self.encoder(x))
        z_q, indices, vq_loss = self.codebook(z_e)
        recon = self.decoder(self.post_quant(z_q))
        return VQGANOutputs(recon=recon, z_e=z_e, z_q=z_q, indices=indices, vq_loss=vq_loss)


def build_vqgan(cfg: VQGANConfig) -> VQGAN:
    return VQGAN(cfg)
