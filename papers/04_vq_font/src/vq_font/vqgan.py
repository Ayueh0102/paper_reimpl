"""VQGAN font codebook backbone — Stage 0 of VQ-Font.

Phase 2 (post-github-diff) note
-------------------------------
The official VQ-Font repo (`Yaomingshuai/VQ-Font`) does **not** use the
taming-transformers `Encoder/Decoder` despite declaring `taming.models.vqgan.VQModel`
in its YAML. Inside `taming/models/vqgan.py:29-30` the encoder/decoder are
overridden with:

    self.encoder = content_enc_builder(C_in=1, C=32, C_out=256)
    self.decoder = dec_builder(C=32, C_out=1, norm="in", out="tanh",
                               C_content=256)

i.e. a simple **InstanceNorm conv stack** `1 -> 32 -> 64 -> 128 -> 256`
with three stride-2 downsamples (no bottleneck attention, no residual
trunk in the encoder), and a mirror decoder with 3 ResBlocks at the
bottleneck plus 3 upsample stages back to 128 px. Both produce a
``[B, 256, 16, 16]`` latent that feeds the codebook of size 1024.

This module rewrites our blind-impl (taming-style residual + attn) to
match the official InstanceNorm conv stack. The codebook interface
(``VectorQuantize``) is preserved — only encoder/decoder shape changed.

The legacy `[guessed-because-paper-vague]` field ``num_res_blocks`` is
retained on ``VQGANConfig`` so existing checkpoints and tests keep
loading, but it now only governs the *decoder* bottleneck Res stack
(default 3 to match the official ``dec_builder``).

See ``reports/github_diff.md`` Special focus #1 for the line-level
diff and rationale.
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

    Defaults follow the official repo (``content_enc_builder`` /
    ``dec_builder`` in ``taming/models/vqgan.py:29-30``):

    * Encoder: ``1 -> C=32 -> 2C -> 4C -> 8C -> C_out=256`` with three
      stride-2 downsamples, ``InstanceNorm + ReLU`` activations.
    * Decoder: 3x ResBlock(8C=256) bottleneck + three upsample Conv
      stages back to ``in_channels`` with ``InstanceNorm`` + ``Tanh``
      output (we keep ``Tanh`` since training data lives in ``[-1, 1]``).

    ``channel_mult`` is no longer used to control encoder topology (the
    official builder hard-codes ``C, 2C, 4C, 8C``) but is kept for backward
    compatibility with ``out_resolution()`` callers.
    """

    image_size: int = 128
    in_channels: int = 1
    """Grayscale glyph input. Set to 3 for RGB content fields."""
    base_channels: int = 32
    """``C`` in the official builder. The encoder widens
    ``C -> 2C -> 4C -> 8C`` across three stride-2 stages, then ``8C -> C_out``."""
    channel_mult: tuple[int, ...] = (1, 1, 2, 4)
    """Retained for ``out_resolution()`` only — three stride-2 downsamples
    (``len(channel_mult) - 1 == 3``) give a 16x16 latent from 128px."""
    z_channels: int = 256
    """Latent channel width before / after the codebook (matches embed_dim)."""
    embed_dim: int = 256
    """Codebook entry dimensionality."""
    num_embeddings: int = 1024
    """Paper-cited codebook size (Phase 0 row 04)."""
    commitment_weight: float = 0.25
    """β in VQ-VAE commitment loss; standard 0.25 (Van den Oord 2017)."""
    num_res_blocks: int = 3
    """Number of ResBlocks at the decoder bottleneck. Official uses 3
    (``dec_builder`` lines 61-63). Encoder has no residual trunk."""
    dropout: float = 0.0

    def out_resolution(self) -> int:
        """Spatial size after the encoder.

        Encoder does ``len(channel_mult) - 1`` stride-2 downsamples (the last
        stage's downsample is an ``Identity``). For the paper-cited default
        ``channel_mult=(1, 1, 2, 4)`` and ``image_size=128`` this yields
        ``128 / 2^3 = 16``.
        """
        return self.image_size // (2 ** (len(self.channel_mult) - 1))


# --------------------------------------------------------------------------------------
# Building blocks (InstanceNorm conv stack, matching `content_enc_builder` /
# `dec_builder` in the official repo)
# --------------------------------------------------------------------------------------


class _ConvBlock(nn.Module):
    """Pre-activation Conv block: (InstanceNorm? -> ReLU -> Conv [-> Upsample]).

    Mirrors ``models/modules/blocks.py:ConvBlock``. Norm is skipped when
    ``in_channels == 1`` (the official builder does the same — see line 79
    of ``blocks.py``). ``upsample`` runs *before* the conv, which matches
    the official "pre-active" ordering.
    """

    def __init__(
        self,
        c_in: int,
        c_out: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        *,
        norm: str = "in",
        activ: str = "relu",
        upsample: bool = False,
    ) -> None:
        super().__init__()
        self.c_in = c_in
        self.upsample = upsample
        if norm == "in":
            self.norm: nn.Module = nn.InstanceNorm2d(c_in)
        elif norm == "none":
            self.norm = nn.Identity()
        else:
            raise ValueError(f"_ConvBlock: unsupported norm={norm!r}")
        if activ == "relu":
            self.activ: nn.Module = nn.ReLU(inplace=False)
        elif activ == "none":
            self.activ = nn.Identity()
        else:
            raise ValueError(f"_ConvBlock: unsupported activ={activ!r}")
        self.conv = nn.Conv2d(c_in, c_out, kernel_size, stride=stride, padding=padding)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Mirror official: skip norm when single-channel input.
        if self.c_in != 1:
            x = self.norm(x)
        x = self.activ(x)
        if self.upsample:
            x = F.interpolate(x, scale_factor=2, mode="nearest")
        return self.conv(x)


class _ResBlock(nn.Module):
    """Pre-active Res block built from two ``_ConvBlock``s + 1x1 skip.

    Matches ``ResBlock`` in ``models/modules/blocks.py:93`` (no upsample/
    downsample variant — used only at the decoder bottleneck where stride
    is 1). Activation is ReLU, norm is InstanceNorm.
    """

    def __init__(self, c_in: int, c_out: int) -> None:
        super().__init__()
        self.conv1 = _ConvBlock(c_in, c_out, kernel_size=3, stride=1, padding=1,
                                norm="in", activ="relu")
        self.conv2 = _ConvBlock(c_out, c_out, kernel_size=3, stride=1, padding=1,
                                norm="in", activ="relu")
        self.skip: nn.Module
        if c_in != c_out:
            self.skip = nn.Conv2d(c_in, c_out, kernel_size=1)
        else:
            self.skip = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv2(self.conv1(x))
        return out + self.skip(x)


# --------------------------------------------------------------------------------------
# Vector Quantizer (codebook)
# --------------------------------------------------------------------------------------


class VectorQuantize(nn.Module):
    """Discrete codebook with straight-through gradient.

    Implements the standard VQ-VAE quantizer (Van den Oord 2017 §3) with
    Esser-style commitment + codebook losses surfaced as ``vq_loss`` so the
    trainer can scale them. The codebook is trained by the explicit
    embedding loss below; there is no EMA update path. Mathematically
    identical to the ``VectorQuantizer2`` used in the official taming impl
    (β=0.25).
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

        # These two terms intentionally split gradients:
        # * codebook_loss updates the selected embedding rows.
        # * commitment_loss updates the encoder features.
        codebook_loss = F.mse_loss(z_q_flat, z_flat.detach())
        commitment_loss = F.mse_loss(z_flat, z_q_flat.detach())
        loss = codebook_loss + self.commitment_weight * commitment_loss

        # Straight-through: encoder gradient flows as identity.
        z_q_st = z_flat + (z_q_flat - z_flat).detach()
        z_q = z_q_st.reshape(b, h, w, c).permute(0, 3, 1, 2).contiguous()
        indices = indices.reshape(b, h, w)
        return z_q, indices, loss

    def lookup(self, indices: torch.Tensor) -> torch.Tensor:
        """Look up codebook entries given an integer index tensor.

        Args:
            indices: any shape long tensor with values in [0, num_embeddings).

        Returns:
            Tensor with one extra trailing-channel dim that matches `embed_dim`,
            shaped as ``[..., embed_dim]``.
        """
        return self.codebook(indices)


# --------------------------------------------------------------------------------------
# Encoder / Decoder (official InstanceNorm conv stack)
# --------------------------------------------------------------------------------------


class VQGANEncoder(nn.Module):
    """Encoder matching ``content_enc_builder`` in the official repo.

    Architecture (for default ``base_channels=32``, ``z_channels=256``,
    128px input):

        Conv(1, 32, 3/1)            # stem, no norm (single-channel input)
        Conv(32, 64, 3/2)           # 128 -> 64
        Conv(64, 128, 3/2)          # 64 -> 32
        Conv(128, 256, 3/2)         # 32 -> 16
        Conv(256, 256, 3/1)         # final projection to z_channels

    All non-stem blocks use ``InstanceNorm + ReLU + Conv``. Output is a
    ``[B, z_channels, 16, 16]`` feature map.
    """

    def __init__(self, cfg: VQGANConfig) -> None:
        super().__init__()
        c = cfg.base_channels
        self.stem = _ConvBlock(
            cfg.in_channels, c, kernel_size=3, stride=1, padding=1,
            norm="in", activ="relu",
        )
        self.down1 = _ConvBlock(c, c * 2, kernel_size=3, stride=2, padding=1,
                                 norm="in", activ="relu")
        self.down2 = _ConvBlock(c * 2, c * 4, kernel_size=3, stride=2, padding=1,
                                 norm="in", activ="relu")
        self.down3 = _ConvBlock(c * 4, c * 8, kernel_size=3, stride=2, padding=1,
                                 norm="in", activ="relu")
        self.out_conv = _ConvBlock(c * 8, cfg.z_channels, kernel_size=3, stride=1,
                                    padding=1, norm="in", activ="relu")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.stem(x)
        h = self.down1(h)
        h = self.down2(h)
        h = self.down3(h)
        return self.out_conv(h)


class VQGANDecoder(nn.Module):
    """Decoder matching ``dec_builder`` in the official repo.

    Architecture (default ``base_channels=32``, ``z_channels=256``):

        ResBlock(8C, 8C) x num_res_blocks   # bottleneck (default 3)
        ConvBlock(8C, 4C, upsample=True)    # 16 -> 32
        ConvBlock(4C, 2C, upsample=True)    # 32 -> 64
        ConvBlock(2C, C,  upsample=True)    # 64 -> 128
        Conv(C, in_channels, 3/1) + Tanh

    All blocks use ``InstanceNorm + ReLU``. We keep ``Tanh`` because our
    training images live in ``[-1, 1]`` (the official builder calls this
    out as ``out='tanh'``).
    """

    def __init__(self, cfg: VQGANConfig) -> None:
        super().__init__()
        c = cfg.base_channels
        z = cfg.z_channels
        # Input projection: z_channels -> 8C if they differ. Official has
        # z_channels == 8C (256 == 32*8) so usually Identity.
        if z != c * 8:
            self.in_proj: nn.Module = nn.Conv2d(z, c * 8, kernel_size=1)
        else:
            self.in_proj = nn.Identity()
        self.res_blocks = nn.ModuleList(
            [_ResBlock(c * 8, c * 8) for _ in range(cfg.num_res_blocks)]
        )
        self.up1 = _ConvBlock(c * 8, c * 4, kernel_size=3, stride=1, padding=1,
                               norm="in", activ="relu", upsample=True)
        self.up2 = _ConvBlock(c * 4, c * 2, kernel_size=3, stride=1, padding=1,
                               norm="in", activ="relu", upsample=True)
        self.up3 = _ConvBlock(c * 2, c, kernel_size=3, stride=1, padding=1,
                               norm="in", activ="relu", upsample=True)
        # NOTE: official keeps norm/activ for the final conv too (see
        # ``ConvBlk(C*1, C_out, 3, 1, 1)`` in ``dec_builder``). We mirror it.
        self.out_conv = _ConvBlock(c, cfg.in_channels, kernel_size=3, stride=1,
                                    padding=1, norm="in", activ="relu")

    def forward(self, z_q: torch.Tensor) -> torch.Tensor:
        h = self.in_proj(z_q)
        for blk in self.res_blocks:
            h = blk(h)
        h = self.up1(h)
        h = self.up2(h)
        h = self.up3(h)
        h = self.out_conv(h)
        return torch.tanh(h)


# --------------------------------------------------------------------------------------
# Top-level VQGAN
# --------------------------------------------------------------------------------------


@dataclass
class VQGANOutputs:
    """Container for VQGAN.forward outputs.

    All fields are torch tensors except ``z_q`` which retains grad to flow
    back into the encoder via the straight-through estimator.
    """

    recon: torch.Tensor
    z_e: torch.Tensor
    z_q: torch.Tensor
    indices: torch.Tensor
    vq_loss: torch.Tensor


class VQGAN(nn.Module):
    """End-to-end VQGAN: encoder -> codebook quantize -> decoder.

    Stage 0 of VQ-Font pretrains this whole module on a font corpus; Stage
    1+ partially freezes it (encoder + late decoder + codebook frozen, the
    three early decoder layers + ``post_quant_conv`` stay trainable — see
    ``model.VQFont._partial_freeze_vqgan``).
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
        # z_channels == embed_dim and these are identities. Names match the
        # official ``quant_conv`` / ``post_quant_conv`` so checkpoints port.
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
