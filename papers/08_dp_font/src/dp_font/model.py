"""DP-Font model — conditional DDPM U-Net with multi-attribute guidance.

Blind reimplementation. The paper note (Obsidian ``023_DP-Font書法擴散
PINN_IJCAI2024.md``) describes the architecture as "DDPM (UNet 主幹) + PINN
物理損失" with multi-attribute guidance and stroke-order sequence as a
fine-grained constraint. Specific channel widths / depths are NOT given —
they are tagged ``[guessed-because-paper-vague]`` in
``reports/blind_impl.md``.

Conditioning inputs accepted by ``DPFont.forward``:
  * ``x_t`` [B, 1, H, W] noisy glyph (grayscale)
  * ``timesteps`` [B] diffusion step indices
  * ``content`` [B, Cc, H, W] cached content fields (bitmap / skeleton / ...)
  * ``writer_id`` [B] writer/calligrapher categorical
  * ``script_id`` [B] script categorical (楷/行/草/...)
  * ``char_id`` [B] char categorical
  * ``ink_intensity`` [B] float in [0, 1]  (synthesized — see note)
  * ``font_size`` [B] float in [0, 1]  (synthesized — see note)
  * ``stroke_order`` [B, L] long, padding with -1; embedded with stroke
    type vocab + positional embedding then mean-pooled into the guidance
    vector. Stroke vocab size is configurable; padded entries (-1) are
    masked.
  * ``ref_images`` / ``ref_valid`` / ``style_family_id`` / ``unit_id`` —
    accepted for shared-API compatibility but ignored by DP-Font.

Why this matches the paper:
  * Multi-attribute guidance: the four categorical / scalar fields above are
    each embedded and summed into the time-mixed conditioning vector. AdaLN-
    style scale-shift modulates U-Net feature maps.
  * Stroke order constraint: stroke types are embedded as a sequence with a
    learnable positional embedding (so order matters), then mean-pooled
    over valid positions, and added to the guidance vector. The cross-
    attention path could be added later — Phase 1 takes the simpler vector
    pooling route to keep the conditioning shape stable across stages
    [guessed-because-paper-vague].
  * The U-Net is a vanilla pixel-space DDPM denoiser at 80 px (paper
    explicitly uses 80×80) with attention at the 10×10 stage.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------------


@dataclass
class DPFontConfig:
    """Structural hyperparameters for DP-Font.

    Defaults follow the paper note:
      * image_size 80  (paper: "80×80 input")
      * cosine β schedule, T=1000 (DDPM default; paper vague)
      * base_channels 64, channel_mult (1, 2, 2, 4)  -> 4 stages
        80 -> 40 -> 20 -> 10  (attn at 10)
    """

    image_size: int = 80
    in_channels: int = 1
    content_channels: int = 1
    """Cached content fields fused into the U-Net. Stage A starts with 1
    (bitmap). Stage B/C may raise to multi-channel (bitmap+skeleton+sdf+...).
    The PINN nib-motion term reads the ``skeleton`` channel preferentially —
    when content_channels >= 3 we assume the third channel is skeleton (see
    ``configs/data_*.yaml``)."""
    base_channels: int = 64
    channel_mult: tuple[int, ...] = (1, 2, 2, 4)
    attn_resolutions: tuple[int, ...] = (10,)
    num_res_blocks: int = 2
    time_embed_dim: int = 256
    cond_embed_dim: int = 256
    num_heads: int = 4
    dropout: float = 0.0

    # Multi-attribute vocab sizes (overridden by training script after the
    # dataset metadata is known).
    writer_vocab_size: int = 32
    script_vocab_size: int = 4
    char_vocab_size: int = 5000
    stroke_vocab_size: int = 36
    """Vocab size for the stroke-order sequence tokens. Public Chinese stroke
    inventories like Make-Me-a-Hanzi / cjklib use 32-36 atomic stroke types
    (橫/豎/撇/捺/折 + their compounds). We default to 36 [guessed-from-public-DB].
    A learnable [PAD] / [SOS] reserve the last two ids."""
    stroke_seq_len: int = 32
    """Max stroke-order sequence length we materialise. 32 covers most chars
    in the Liu Gongquan / Yan Zhenqing data (paper does not give a bound)."""

    # Scalar attributes (paper hints at "墨色濃淡 + 字體大小") — Phase 1 wires
    # the two scalars even if Ernantang ships no ink/size labels; we
    # synthesise them deterministically from the dataset (see ``dataset.py``).
    use_ink_intensity: bool = True
    use_font_size: bool = True

    @property
    def time_input_dim(self) -> int:
        return self.base_channels

    def stage_resolutions(self) -> list[int]:
        sizes = [self.image_size]
        for _ in range(len(self.channel_mult) - 1):
            sizes.append(sizes[-1] // 2)
        return sizes


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------


def _gn(channels: int) -> nn.GroupNorm:
    """GroupNorm with a group count that divides ``channels``."""
    for g in (32, 16, 8, 4, 2, 1):
        if channels % g == 0:
            return nn.GroupNorm(g, channels)
    return nn.GroupNorm(1, channels)


def timestep_embedding(timesteps: torch.Tensor, dim: int) -> torch.Tensor:
    """Sinusoidal timestep embedding (Vaswani 2017 / Ho DDPM 2020 §3.2)."""
    half = dim // 2
    device = timesteps.device
    freqs = torch.exp(
        -math.log(10_000.0)
        * torch.arange(half, device=device, dtype=torch.float32)
        / max(half - 1, 1)
    )
    args = timesteps.float().unsqueeze(1) * freqs.unsqueeze(0)
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    if dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb


# ----------------------------------------------------------------------------
# Multi-attribute guidance head
# ----------------------------------------------------------------------------


class MultiAttributeGuidance(nn.Module):
    """Build the multi-attribute conditioning vector.

    Paper note: "以多維屬性向量(書寫者 ID、筆觸濃淡、字體大小等)作為 diffusion
    的 condition". We embed each categorical attribute and concatenate two
    scalar attributes (ink_intensity, font_size) into a single MLP that
    produces a guidance vector summed with the time embedding (FiLM-style).

    All embedding tables include a learnable "null" id (id = vocab_size) so
    classifier-free guidance can drop any single attribute without breaking
    the linear path.
    """

    def __init__(self, cfg: DPFontConfig) -> None:
        super().__init__()
        self.cfg = cfg
        d = cfg.cond_embed_dim
        # +1 for the null token used during CFG dropout.
        self.writer_emb = nn.Embedding(cfg.writer_vocab_size + 1, d)
        self.script_emb = nn.Embedding(cfg.script_vocab_size + 1, d)
        self.char_emb = nn.Embedding(cfg.char_vocab_size + 1, d)
        self.stroke_emb = nn.Embedding(cfg.stroke_vocab_size + 2, d)
        # Position embedding for stroke order. +1 for [SOS] / no-stroke prefix.
        self.stroke_pos_emb = nn.Embedding(cfg.stroke_seq_len + 1, d)

        scalar_in = int(cfg.use_ink_intensity) + int(cfg.use_font_size)
        self.scalar_proj = (
            nn.Sequential(nn.Linear(scalar_in, d), nn.SiLU(inplace=True), nn.Linear(d, d))
            if scalar_in > 0
            else None
        )

        self.fuse = nn.Sequential(
            nn.Linear(d, d),
            nn.SiLU(inplace=True),
            nn.Linear(d, d),
        )

        # Sane init — keep guidance contribution modest at start.
        for module in (self.writer_emb, self.script_emb, self.char_emb, self.stroke_emb):
            nn.init.normal_(module.weight, std=0.02)
        nn.init.normal_(self.stroke_pos_emb.weight, std=0.02)

    @property
    def null_ids(self) -> dict[str, int]:
        return {
            "writer": self.cfg.writer_vocab_size,
            "script": self.cfg.script_vocab_size,
            "char": self.cfg.char_vocab_size,
        }

    def _clamp(self, ids: torch.Tensor | None, vocab: int, fallback: int) -> torch.Tensor | None:
        if ids is None:
            return None
        return ids.clamp(0, vocab).long()

    def forward(
        self,
        *,
        batch_size: int,
        device: torch.device,
        writer_id: torch.Tensor | None,
        script_id: torch.Tensor | None,
        char_id: torch.Tensor | None,
        stroke_order: torch.Tensor | None,
        ink_intensity: torch.Tensor | None,
        font_size: torch.Tensor | None,
    ) -> torch.Tensor:
        """Return [B, cond_embed_dim]."""
        nulls = self.null_ids
        w = self._clamp(writer_id, nulls["writer"], nulls["writer"])
        s = self._clamp(script_id, nulls["script"], nulls["script"])
        c = self._clamp(char_id, nulls["char"], nulls["char"])

        if w is None:
            w = torch.full((batch_size,), nulls["writer"], device=device, dtype=torch.long)
        if s is None:
            s = torch.full((batch_size,), nulls["script"], device=device, dtype=torch.long)
        if c is None:
            c = torch.full((batch_size,), nulls["char"], device=device, dtype=torch.long)

        h = self.writer_emb(w) + self.script_emb(s) + self.char_emb(c)

        # Stroke order: [B, L] long with -1 for padding. Substitute the
        # stroke-vocab null (id == stroke_vocab_size + 1) for padded slots and
        # mask them when pooling.
        if stroke_order is not None and stroke_order.numel() > 0:
            so = stroke_order.long()
            L = so.shape[1]
            null_id = self.cfg.stroke_vocab_size + 1
            valid = so >= 0
            so = torch.where(valid, so, torch.full_like(so, null_id))
            tok = self.stroke_emb(so)
            pos = self.stroke_pos_emb(torch.arange(L, device=device).clamp(max=self.cfg.stroke_seq_len))
            seq = tok + pos.unsqueeze(0)
            mask = valid.float().unsqueeze(-1)
            denom = mask.sum(dim=1).clamp_min(1.0)
            pooled = (seq * mask).sum(dim=1) / denom
            h = h + pooled

        if self.scalar_proj is not None:
            scalars: list[torch.Tensor] = []
            if self.cfg.use_ink_intensity:
                ii = (
                    ink_intensity.float()
                    if ink_intensity is not None
                    else torch.zeros(batch_size, device=device)
                )
                scalars.append(ii)
            if self.cfg.use_font_size:
                fs = (
                    font_size.float()
                    if font_size is not None
                    else torch.zeros(batch_size, device=device)
                )
                scalars.append(fs)
            scalar = torch.stack(scalars, dim=-1)
            h = h + self.scalar_proj(scalar)

        return self.fuse(h)


# ----------------------------------------------------------------------------
# U-Net building blocks
# ----------------------------------------------------------------------------


class _ConvBlock(nn.Module):
    def __init__(self, in_c: int, out_c: int, *, stride: int = 1) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, stride=stride, padding=1),
            _gn(out_c),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_c, out_c, kernel_size=3, stride=1, padding=1),
            _gn(out_c),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x)


class ContentEncoder(nn.Module):
    """Encode cached content fields into per-stage feature maps."""

    def __init__(self, *, in_channels: int, base_channels: int, channel_mult: Sequence[int]) -> None:
        super().__init__()
        chs = [base_channels * m for m in channel_mult]
        self.stem = nn.Conv2d(in_channels, chs[0], kernel_size=3, padding=1)
        self.blocks = nn.ModuleList()
        prev_c = chs[0]
        for i, c in enumerate(chs):
            stride = 2 if i > 0 else 1
            self.blocks.append(_ConvBlock(prev_c, c, stride=stride))
            prev_c = c

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        feats: list[torch.Tensor] = []
        h = self.stem(x)
        for block in self.blocks:
            h = block(h)
            feats.append(h)
        return feats


class ResBlock(nn.Module):
    """Residual block with FiLM-style timestep + multi-attribute modulation."""

    def __init__(
        self,
        in_c: int,
        out_c: int,
        *,
        time_embed_dim: int,
        cond_embed_dim: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.norm1 = _gn(in_c)
        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size=3, padding=1)
        self.time_proj = nn.Linear(time_embed_dim, out_c)
        self.cond_proj = nn.Linear(cond_embed_dim, out_c)
        self.norm2 = _gn(out_c)
        self.conv2 = nn.Conv2d(out_c, out_c, kernel_size=3, padding=1)
        self.dropout = nn.Dropout(dropout)
        self.skip = nn.Conv2d(in_c, out_c, kernel_size=1) if in_c != out_c else nn.Identity()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.time_proj(F.silu(t_emb))[:, :, None, None]
        h = h + self.cond_proj(F.silu(cond))[:, :, None, None]
        h = self.conv2(self.dropout(F.silu(self.norm2(h))))
        return h + self.skip(x)


class SelfAttn2D(nn.Module):
    """Standard multi-head self-attention over spatial tokens."""

    def __init__(self, channels: int, *, num_heads: int) -> None:
        super().__init__()
        assert channels % num_heads == 0
        self.num_heads = num_heads
        self.norm = _gn(channels)
        self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1)
        self.proj = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        qkv = self.qkv(self.norm(x))
        q, k, v = qkv.chunk(3, dim=1)
        q = q.reshape(b, self.num_heads, c // self.num_heads, h * w).transpose(-1, -2)
        k = k.reshape(b, self.num_heads, c // self.num_heads, h * w)
        v = v.reshape(b, self.num_heads, c // self.num_heads, h * w).transpose(-1, -2)
        attn = torch.softmax(torch.matmul(q, k) / math.sqrt(c // self.num_heads), dim=-1)
        out = torch.matmul(attn, v).transpose(-1, -2).reshape(b, c, h, w)
        return x + self.proj(out)


class Downsample(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.op = nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.op(x)


class Upsample(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.op = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        return self.op(x)


class ContentFuse(nn.Module):
    """Lightweight gated fusion of a content feature into the U-Net stream."""

    def __init__(self, content_channels: int, unet_channels: int) -> None:
        super().__init__()
        self.project = nn.Conv2d(content_channels, unet_channels, kernel_size=1)
        self.gate = nn.Conv2d(unet_channels, unet_channels, kernel_size=1)
        nn.init.zeros_(self.gate.weight)
        nn.init.zeros_(self.gate.bias)

    def forward(self, unet_feat: torch.Tensor, content_feat: torch.Tensor) -> torch.Tensor:
        if content_feat.shape[-2:] != unet_feat.shape[-2:]:
            content_feat = F.interpolate(
                content_feat, size=unet_feat.shape[-2:], mode="bilinear", align_corners=False
            )
        c = self.project(content_feat)
        g = torch.sigmoid(self.gate(c))
        return unet_feat + g * c


# ----------------------------------------------------------------------------
# U-Net
# ----------------------------------------------------------------------------


class DPFontUNet(nn.Module):
    """Pixel-space U-Net denoiser with multi-attribute FiLM conditioning."""

    def __init__(self, cfg: DPFontConfig) -> None:
        super().__init__()
        self.cfg = cfg
        channels = [cfg.base_channels * m for m in cfg.channel_mult]
        resolutions = cfg.stage_resolutions()
        self.stage_resolutions = resolutions

        self.time_mlp = nn.Sequential(
            nn.Linear(cfg.time_input_dim, cfg.time_embed_dim),
            nn.SiLU(inplace=True),
            nn.Linear(cfg.time_embed_dim, cfg.time_embed_dim),
        )

        self.input_conv = nn.Conv2d(cfg.in_channels, channels[0], kernel_size=3, padding=1)

        self.down_blocks: nn.ModuleList = nn.ModuleList()
        self.down_attn: nn.ModuleList = nn.ModuleList()
        self.down_content_fuse: nn.ModuleList = nn.ModuleList()
        self.downsamples: nn.ModuleList = nn.ModuleList()
        prev_c = channels[0]
        for i, c in enumerate(channels):
            stage = nn.ModuleList(
                [
                    ResBlock(
                        prev_c if r == 0 else c,
                        c,
                        time_embed_dim=cfg.time_embed_dim,
                        cond_embed_dim=cfg.cond_embed_dim,
                        dropout=cfg.dropout,
                    )
                    for r in range(cfg.num_res_blocks)
                ]
            )
            self.down_blocks.append(stage)
            if resolutions[i] in cfg.attn_resolutions:
                self.down_attn.append(SelfAttn2D(c, num_heads=cfg.num_heads))
            else:
                self.down_attn.append(nn.Identity())
            self.down_content_fuse.append(ContentFuse(content_channels=c, unet_channels=c))
            if i < len(channels) - 1:
                self.downsamples.append(Downsample(c))
            else:
                self.downsamples.append(nn.Identity())
            prev_c = c

        mid_c = channels[-1]
        self.mid_block1 = ResBlock(
            mid_c, mid_c,
            time_embed_dim=cfg.time_embed_dim,
            cond_embed_dim=cfg.cond_embed_dim,
            dropout=cfg.dropout,
        )
        self.mid_attn = SelfAttn2D(mid_c, num_heads=cfg.num_heads)
        self.mid_block2 = ResBlock(
            mid_c, mid_c,
            time_embed_dim=cfg.time_embed_dim,
            cond_embed_dim=cfg.cond_embed_dim,
            dropout=cfg.dropout,
        )

        self.up_blocks: nn.ModuleList = nn.ModuleList()
        self.up_attn: nn.ModuleList = nn.ModuleList()
        self.up_content_fuse: nn.ModuleList = nn.ModuleList()
        self.upsamples: nn.ModuleList = nn.ModuleList()
        reversed_channels = list(reversed(channels))
        reversed_resolutions = list(reversed(resolutions))

        prev_c = mid_c
        for i, c in enumerate(reversed_channels):
            stage = nn.ModuleList()
            for r in range(cfg.num_res_blocks + 1):
                in_c = prev_c + c if r == 0 else c
                stage.append(
                    ResBlock(
                        in_c, c,
                        time_embed_dim=cfg.time_embed_dim,
                        cond_embed_dim=cfg.cond_embed_dim,
                        dropout=cfg.dropout,
                    )
                )
                prev_c = c
            self.up_blocks.append(stage)
            if reversed_resolutions[i] in cfg.attn_resolutions:
                self.up_attn.append(SelfAttn2D(c, num_heads=cfg.num_heads))
            else:
                self.up_attn.append(nn.Identity())
            self.up_content_fuse.append(ContentFuse(content_channels=c, unet_channels=c))
            if i < len(reversed_channels) - 1:
                self.upsamples.append(Upsample(c))
            else:
                self.upsamples.append(nn.Identity())

        self.out_norm = _gn(channels[0])
        self.out_conv = nn.Conv2d(channels[0], cfg.in_channels, kernel_size=3, padding=1)
        # Non-zero init so the smoke gradient flows through every branch.
        nn.init.normal_(self.out_conv.weight, std=0.02)
        nn.init.zeros_(self.out_conv.bias)

    def forward(
        self,
        x: torch.Tensor,
        time_emb: torch.Tensor,
        cond: torch.Tensor,
        content_feats: list[torch.Tensor],
    ) -> torch.Tensor:
        h = self.input_conv(x)
        skips: list[torch.Tensor] = []

        for i, stage in enumerate(self.down_blocks):
            for block in stage:
                h = block(h, time_emb, cond)
            if i < len(content_feats):
                h = self.down_content_fuse[i](h, content_feats[i])
            if not isinstance(self.down_attn[i], nn.Identity):
                h = self.down_attn[i](h)
            skips.append(h)
            if not isinstance(self.downsamples[i], nn.Identity):
                h = self.downsamples[i](h)

        h = self.mid_block1(h, time_emb, cond)
        h = self.mid_attn(h)
        h = self.mid_block2(h, time_emb, cond)

        for i, stage in enumerate(self.up_blocks):
            skip = skips.pop()
            if h.shape[-2:] != skip.shape[-2:]:
                h = F.interpolate(h, size=skip.shape[-2:], mode="nearest")
            h = torch.cat([h, skip], dim=1)
            for block in stage:
                h = block(h, time_emb, cond)
            if i < len(content_feats):
                rev_idx = len(content_feats) - 1 - i
                h = self.up_content_fuse[i](h, content_feats[rev_idx])
            if not isinstance(self.up_attn[i], nn.Identity):
                h = self.up_attn[i](h)
            if not isinstance(self.upsamples[i], nn.Identity):
                h = self.upsamples[i](h)

        return self.out_conv(F.silu(self.out_norm(h)))


# ----------------------------------------------------------------------------
# Top-level DP-Font
# ----------------------------------------------------------------------------


class DPFont(nn.Module):
    """DP-Font conditional denoiser.

    forward signature matches the shared ``GaussianDiffusion`` contract.
    """

    def __init__(self, cfg: DPFontConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.content_encoder = ContentEncoder(
            in_channels=cfg.content_channels,
            base_channels=cfg.base_channels,
            channel_mult=cfg.channel_mult,
        )
        self.guidance = MultiAttributeGuidance(cfg)
        self.unet = DPFontUNet(cfg)

    def forward(
        self,
        x_t: torch.Tensor,
        timesteps: torch.Tensor,
        *,
        content: torch.Tensor,
        char_id: torch.Tensor | None = None,
        script_id: torch.Tensor | None = None,
        writer_id: torch.Tensor | None = None,
        style_family_id: torch.Tensor | None = None,
        unit_id: torch.Tensor | None = None,
        ref_images: torch.Tensor | None = None,
        ref_valid: torch.Tensor | None = None,
        stroke_order: torch.Tensor | None = None,
        ink_intensity: torch.Tensor | None = None,
        font_size: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Predict x0 or epsilon. ``style_family_id`` / ``unit_id`` /
        ``ref_images`` / ``ref_valid`` are accepted for shared-API
        compatibility but unused by DP-Font (paper conditions on writer +
        script + char + stroke-order, not reference images)."""
        del style_family_id, unit_id, ref_images, ref_valid  # explicitly ignored

        # Content path
        content_feats = self.content_encoder(content)

        # Multi-attribute guidance
        cond = self.guidance(
            batch_size=x_t.shape[0],
            device=x_t.device,
            writer_id=writer_id,
            script_id=script_id,
            char_id=char_id,
            stroke_order=stroke_order,
            ink_intensity=ink_intensity,
            font_size=font_size,
        )

        # Time embedding
        t_input = timestep_embedding(timesteps, self.cfg.time_input_dim).to(dtype=x_t.dtype)
        t_emb = self.unet.time_mlp(t_input)
        return self.unet(x_t, time_emb=t_emb, cond=cond, content_feats=content_feats)


def build_dp_font(cfg: DPFontConfig) -> DPFont:
    return DPFont(cfg)
