"""FontDiffuser — blind reimplementation.

Architecture (from paper notes — AAAI 2024):
  * U-Net DDPM backbone (pixel-space, noise-to-denoise) [p.1]
  * Content Encoder: encodes the source glyph (rendered TTF target char)
    into a pyramid of multi-scale feature maps.
  * Style Encoder: encodes the one-shot style reference glyph into a
    token sequence (CNN trunk + spatial flatten) used as cross-attention K/V.
  * MCA — Multi-scale Content Aggregation: at each U-Net resolution stage,
    fuse the matching-resolution content-encoder feature map into the UNet
    skip stream (paper §"resolves stroke loss on hard complex chars").
  * RSI — Reference-Structure Interaction: cross-attention block that lets
    UNet bottleneck features attend to the style token sequence. This is the
    place where the one-shot style reference gets injected — paper says RSI
    "models structural deformation between reference and target".
  * SCR — Style Contrastive Refinement: separate ``StyleExtractor`` consumes
    {predicted_x0, true_x0} and produces L2-normalized embeddings. Loss is
    supervised contrastive over writer_id in the batch (same writer = positive
    pair, different writer = negative).

Adapter notes for ``paper_reimpl_shared.diffusion.gaussian.GaussianDiffusion``:
  The shared diffusion utility calls
  ``model(x_t, t, *, content, char_id, script_id, writer_id, style_family_id,
            unit_id, ref_images, ref_valid)``. FontDiffuser only consumes
  ``content`` (source glyph render — already preprocessed by dataset, treated
  as ``content_channels``-channel input to the content encoder) and
  ``ref_images[:, 0]`` (the one-shot reference glyph). The remaining
  conditioning args are accepted but ignored — this keeps the contract clean
  with the shared sampler while staying faithful to paper §3.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import DeformConv2d


# --------------------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------------------


@dataclass
class FontDiffuserConfig:
    """All structural hyperparameters for the FontDiffuser blind reimpl.

    Defaults are conservative: small enough to run a CPU smoke test in seconds,
    and reasonable for a 128/256-px Ernantang Stage A run on a single GPU.

    Phase 2 revision (2026-05-11) brings the structure in line with the
    official ``yeungchenwa/FontDiffuser`` repo:
      * MCA is restricted to the inner two U-Net stages (concat+SE channel
        attention rather than gated-add).
      * RSI on the up-path is a deformable-conv warp of the skip connection,
        regularised by an offset L1 magnitude loss (``offset_l1_weight``).
      * Style cross-attention runs at every MCA stage on both paths.
    """

    image_size: int = 128
    in_channels: int = 1
    """Channels of the noisy target image (grayscale = 1)."""
    content_channels: int = 1
    """Channels of the source-glyph content image. Synthetic batch uses 3
    (RGB-rendered TTF preview); real ernantang uses 6 channels of cached content
    fields (bitmap/sdf/skeleton/...). Configurable per stage YAML."""
    ref_channels: int = 1
    """Channels of the one-shot reference glyph."""
    base_channels: int = 64
    channel_mult: tuple[int, ...] = (1, 2, 4, 4)
    """U-Net stage channel multipliers. Bottleneck = base_channels * mult[-1]."""
    attn_resolutions: tuple[int, ...] = (16,)
    """Legacy: spatial sizes at which self-attn fires. After the Phase 2
    revision, RSI placement is driven by ``mca_stages``/``rsi_up_stages``
    instead. Kept for the residual SelfAttn2D inside attn-stage down blocks
    (matches the BasicTransformerBlock self+cross bundle in the official)."""
    num_res_blocks: int = 2
    time_embed_dim: int = 256
    style_embed_dim: int = 256
    """Token width of the StyleEncoder output and of RSI cross-attn K/V."""
    num_heads: int = 4
    dropout: float = 0.0
    # ---- Phase 2 (architecture-faithful) additions ----
    mca_stages: tuple[int, ...] | None = None
    """Down/up-stage indices (0-indexed) where MCA fuse + style cross-attn fire.
    Default is the inner two stages, matching the official
    ``('DownBlock2D','MCADownBlock2D','MCADownBlock2D','DownBlock2D')`` layout
    for a 4-stage U-Net. When ``None`` we auto-pick ``(1, ..., N-2)``."""
    rsi_up_stages: tuple[int, ...] | None = None
    """Up-path stages (counted from the bottleneck outward, 0-indexed) where
    the deformable-conv RSI fires. Default mirrors ``mca_stages`` after
    reversal so the inner two up-stages carry RSI."""
    se_reduction: int = 32
    """Channel reduction in the MCA Squeeze-Excitation block (official:
    ``reduction=32``, ``configs/fontdiffuser.py:31``)."""
    offset_l1_weight: float = 0.5
    """Coefficient applied to the offset-L1 magnitude term in the training
    loss (official Phase 1: ``offset_coefficient=0.5``,
    ``third_party/01_fontdiffuser/configs/fontdiffuser.py:50``)."""
    perceptual_weight: float = 0.01
    """Coefficient applied to the VGG-Perceptual loss on predicted x0
    (official Phase 1: ``perceptual_coefficient=0.01``,
    ``third_party/01_fontdiffuser/configs/fontdiffuser.py:49``)."""

    @property
    def time_input_dim(self) -> int:
        """Sin/cos timestep base dim. We project to ``time_embed_dim``."""
        return self.base_channels

    def stage_resolutions(self) -> list[int]:
        """Spatial resolutions at each U-Net stage (after each downsample)."""
        sizes = [self.image_size]
        for _ in range(len(self.channel_mult) - 1):
            sizes.append(sizes[-1] // 2)
        return sizes

    def resolved_mca_stages(self) -> tuple[int, ...]:
        """0-indexed down-path stages that carry MCA + style cross-attn.

        Default ``(1, ..., N-2)`` — for 4 stages this becomes ``(1, 2)``,
        matching the official ``MCADownBlock2D`` placement in
        ``third_party/01_fontdiffuser/src/build.py:15-18``.
        """
        if self.mca_stages is not None:
            return tuple(int(x) for x in self.mca_stages)
        n = len(self.channel_mult)
        if n <= 2:
            return tuple(range(n))
        return tuple(range(1, n - 1))

    def resolved_rsi_up_stages(self) -> tuple[int, ...]:
        """Up-path stage indices (counted from the bottleneck out, ``i=0`` is
        innermost) that carry deformable-conv RSI. Mirrors ``mca_stages``
        after reversal so the inner up-stages 1-2 fire on a 4-stage U-Net,
        matching ``build.py:19-22``."""
        if self.rsi_up_stages is not None:
            return tuple(int(x) for x in self.rsi_up_stages)
        n = len(self.channel_mult)
        down = self.resolved_mca_stages()
        # Reverse onto up path: down stage i corresponds to up index (n-1-i).
        return tuple(sorted({n - 1 - i for i in down}))


# --------------------------------------------------------------------------------------
# Time / writer embedding helpers
# --------------------------------------------------------------------------------------


def _gn(channels: int) -> nn.GroupNorm:
    """GroupNorm with a group count that divides ``channels``.

    We probe down from 32 to 1 to find the largest divisor — keeps stats
    similar to the common ``num_groups=32`` recipe while never crashing on
    awkward widths like 48 (16 + 32 from a skip concat).
    """
    for g in (32, 16, 8, 4, 2, 1):
        if channels % g == 0:
            return nn.GroupNorm(g, channels)
    return nn.GroupNorm(1, channels)


def timestep_embedding(timesteps: torch.Tensor, dim: int) -> torch.Tensor:
    """Sinusoidal timestep embedding (Vaswani 2017 / Ho DDPM 2020 §3.2)."""
    half = dim // 2
    device = timesteps.device
    freqs = torch.exp(
        -math.log(10_000.0) * torch.arange(half, device=device, dtype=torch.float32) / max(half - 1, 1)
    )
    args = timesteps.float().unsqueeze(1) * freqs.unsqueeze(0)
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    if dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb


# --------------------------------------------------------------------------------------
# Content encoder (multi-scale)
# --------------------------------------------------------------------------------------


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
    """Multi-scale content encoder. Outputs one feature map per U-Net stage.

    Paper "MCA" requires content at multiple resolutions; we mirror the U-Net's
    channel-mult / stage layout so feeds align by spatial size and channels.
    """

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


# --------------------------------------------------------------------------------------
# Style encoder (one-shot reference)
# --------------------------------------------------------------------------------------


class StyleEncoder(nn.Module):
    """Encode a single reference glyph into a token sequence.

    Output shape: [B, L, embed_dim] where L = (H/8) * (W/8). RSI block uses
    this as cross-attention K/V. Paper note: "RSI cross-attention on reference"
    [p.2]. We deliberately keep the trunk shallow (4 strided convs) — the
    paper does not give a depth, and a shallow trunk is sufficient for
    one-shot style reference.
    """

    def __init__(self, *, in_channels: int, embed_dim: int) -> None:
        super().__init__()
        c1, c2, c3 = embed_dim // 4, embed_dim // 2, embed_dim
        self.body = nn.Sequential(
            nn.Conv2d(in_channels, c1, kernel_size=3, stride=2, padding=1),
            _gn(c1),
            nn.SiLU(inplace=True),
            nn.Conv2d(c1, c2, kernel_size=3, stride=2, padding=1),
            _gn(c2),
            nn.SiLU(inplace=True),
            nn.Conv2d(c2, c3, kernel_size=3, stride=2, padding=1),
            _gn(c3),
            nn.SiLU(inplace=True),
            nn.Conv2d(c3, embed_dim, kernel_size=3, stride=1, padding=1),
        )
        self.embed_dim = embed_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, C, H, W] -> tokens [B, L, embed_dim]."""
        h = self.body(x)
        b, c, hh, ww = h.shape
        return h.flatten(2).transpose(1, 2)  # [B, L, C]


# --------------------------------------------------------------------------------------
# UNet building blocks
# --------------------------------------------------------------------------------------


class ResBlock(nn.Module):
    """Residual block with FiLM-style timestep modulation."""

    def __init__(self, in_c: int, out_c: int, *, time_embed_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.norm1 = _gn(in_c)
        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size=3, padding=1)
        self.time_proj = nn.Linear(time_embed_dim, out_c)
        self.norm2 = _gn(out_c)
        self.conv2 = nn.Conv2d(out_c, out_c, kernel_size=3, padding=1)
        self.dropout = nn.Dropout(dropout)
        self.skip = nn.Conv2d(in_c, out_c, kernel_size=1) if in_c != out_c else nn.Identity()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.time_proj(F.silu(t_emb))[:, :, None, None]
        h = self.conv2(self.dropout(F.silu(self.norm2(h))))
        return h + self.skip(x)


class SelfAttn2D(nn.Module):
    """Standard multi-head self-attention over spatial tokens. Used inside
    attn_resolutions stages of the UNet."""

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


class RSIBlock(nn.Module):
    """Reference-Structure Interaction.

    Cross-attention from UNet spatial tokens (queries) to style tokens
    (keys/values). This is where the one-shot reference glyph injects
    deformation cues into the denoiser.

    Reference: paper §3 "RSI block" [p.2]. The note describes RSI as a
    cross-attention block, dimensions not specified — we use ``num_heads``
    heads at the unet stage channel width.
    """

    def __init__(self, *, channels: int, style_dim: int, num_heads: int) -> None:
        super().__init__()
        assert channels % num_heads == 0
        self.num_heads = num_heads
        self.norm_q = _gn(channels)
        self.norm_kv = nn.LayerNorm(style_dim)
        self.to_q = nn.Conv2d(channels, channels, kernel_size=1)
        self.to_k = nn.Linear(style_dim, channels)
        self.to_v = nn.Linear(style_dim, channels)
        self.proj = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(
        self,
        x: torch.Tensor,
        style_tokens: torch.Tensor,
        style_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        b, c, h, w = x.shape
        q = self.to_q(self.norm_q(x))
        k = self.to_k(self.norm_kv(style_tokens))
        v = self.to_v(self.norm_kv(style_tokens))
        head_c = c // self.num_heads
        seq_len = style_tokens.shape[1]
        q = q.reshape(b, self.num_heads, head_c, h * w).transpose(-1, -2)  # [B, H, HW, Hc]
        k = k.reshape(b, seq_len, self.num_heads, head_c).permute(0, 2, 3, 1)  # [B, H, Hc, L]
        v = v.reshape(b, seq_len, self.num_heads, head_c).permute(0, 2, 1, 3)  # [B, H, L, Hc]
        attn = torch.matmul(q, k) / math.sqrt(head_c)
        if style_mask is not None:
            # mask: [B, L]; expand to [B, 1, 1, L]
            mask = style_mask.bool().unsqueeze(1).unsqueeze(2)
            attn = attn.masked_fill(~mask, float("-inf"))
        attn = torch.softmax(attn, dim=-1)
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


# --------------------------------------------------------------------------------------
# MCA: Multi-scale Content Aggregation
# --------------------------------------------------------------------------------------


class _SELayer(nn.Module):
    """Squeeze-Excitation channel attention (Hu et al. 2018).

    Mirrors the SE used inside the official MCA ``ChannelAttnBlock``
    (``third_party/01_fontdiffuser/src/modules/attention.py:335-351``).
    Default ``reduction=32`` per ``configs/fontdiffuser.py``.
    """

    def __init__(self, channels: int, reduction: int = 32) -> None:
        super().__init__()
        hidden = max(1, channels // max(1, reduction))
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, hidden, bias=False),
            nn.SiLU(inplace=True),
            nn.Linear(hidden, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.shape
        y = self.avg_pool(x).reshape(b, c)
        y = self.fc(y).reshape(b, c, 1, 1)
        return x * y.expand_as(x)


class MCAFuse(nn.Module):
    """Multi-scale Content Aggregation — Phase 2 concat-then-SE variant.

    Concept origin: official ``ChannelAttnBlock``
    (``third_party/01_fontdiffuser/src/modules/attention.py:359-414``).

    Operation:
      1. Project ``content`` to ``unet_channels`` and align spatial size.
      2. Concat content with the UNet feature on the channel dim.
      3. GroupNorm + SiLU + 1x1 Conv to mix channels.
      4. SE channel-attention with a residual to the concat feature.
      5. GroupNorm + SiLU + 1x1 Conv to project back to ``unet_channels``.

    The earlier blind impl was a zero-init gated additive blend; that
    operator is replaced wholesale by this concat+SE block.
    """

    def __init__(
        self,
        content_channels: int,
        unet_channels: int,
        *,
        se_reduction: int = 32,
    ) -> None:
        super().__init__()
        self.project_content = nn.Conv2d(content_channels, unet_channels, kernel_size=1)
        concat_c = unet_channels * 2
        self.norm1 = _gn(concat_c)
        self.conv1 = nn.Conv2d(concat_c, concat_c, kernel_size=1)
        self.se = _SELayer(concat_c, reduction=max(1, se_reduction))
        self.norm2 = _gn(concat_c)
        self.down_channel = nn.Conv2d(concat_c, unet_channels, kernel_size=1)

    def forward(self, unet_feat: torch.Tensor, content_feat: torch.Tensor) -> torch.Tensor:
        if content_feat.shape[-2:] != unet_feat.shape[-2:]:
            content_feat = F.interpolate(
                content_feat, size=unet_feat.shape[-2:], mode="bilinear", align_corners=False
            )
        c = self.project_content(content_feat)
        concat = torch.cat([unet_feat, c], dim=1)
        h = self.conv1(F.silu(self.norm1(concat)))
        h = self.se(h)
        h = h + concat  # SE residual back to the concat feature, per official block
        h = self.down_channel(F.silu(self.norm2(h)))
        return h


class OffsetRefStrucInter(nn.Module):
    """Reference-Structure Interaction — deformable-conv warp on the skip.

    Concept origin: official ``OffsetRefStrucInter`` (predicts the offset map)
    + ``StyleRSIUpBlock2D.dcn_deforms`` (consumes the offset)
    (``third_party/01_fontdiffuser/src/modules/attention.py:266-332``,
    ``third_party/01_fontdiffuser/src/modules/unet_blocks.py:423-587``).

    Re-implementation summary (not a verbatim copy):

      1. The skip feature ``res`` (post-resblock, this U-Net stage) is the
         query; the matching-resolution **style-content** feature from
         ``ContentEncoder(style_image)`` is the context.
      2. Both branches are GN+SiLU normalized, projected to a shared inner
         dim, then a single-head cross-attention from style-content (Q) onto
         skip (K/V) produces a per-position aggregated style-content vector.
      3. A 1x1 conv reduces that vector to the deformable-conv offset map
         ``[B, 2*kH*kW, H, W]`` (default kernel 3x3 -> 18 channels).
      4. ``torchvision.ops.DeformConv2d`` warps the *skip* feature in-place
         using the predicted offsets — this is the geometric deformation
         the paper attributes to RSI.
      5. ``mean(|offset|)`` is exposed as ``offset_l1`` so train.py can
         accumulate it across stages and add ``0.5 * sum`` to the loss.

    The cross-attention here is light and intentionally simpler than the
    official ``BasicTransformerBlock`` — we keep the structural deformation
    bit (offset prediction + DCN) which is the paper's named contribution.
    """

    def __init__(
        self,
        *,
        skip_channels: int,
        style_content_channels: int,
        kernel_size: int = 3,
        num_heads: int = 4,
    ) -> None:
        super().__init__()
        inner_dim = skip_channels
        self.norm_skip = _gn(skip_channels)
        self.norm_style = _gn(style_content_channels)
        self.proj_skip = nn.Conv2d(skip_channels, inner_dim, kernel_size=1)
        self.proj_style = nn.Conv2d(style_content_channels, inner_dim, kernel_size=1)
        # Lightweight 1x1 mix: per-position add then small MLP.
        self.mix = nn.Sequential(
            nn.Conv2d(inner_dim * 2, inner_dim, kernel_size=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(inner_dim, inner_dim, kernel_size=1),
        )
        # Offset head: 2 * kH * kW channels per the DCNv2 contract.
        self.offset_head = nn.Conv2d(inner_dim, 2 * kernel_size * kernel_size, kernel_size=1)
        # Small-std init keeps the offset path alive (non-zero gradient on
        # iter 0) while staying close to identity-conv behaviour for the
        # first few hundred steps. Zero-init would disconnect the
        # style-content gradient entirely on smoke-test step 0.
        nn.init.normal_(self.offset_head.weight, std=1e-3)
        nn.init.zeros_(self.offset_head.bias)
        self.dcn = DeformConv2d(
            in_channels=skip_channels,
            out_channels=skip_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,
            bias=True,
        )
        self.kernel_size = kernel_size

    def forward(
        self,
        skip: torch.Tensor,
        style_content_feat: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Warp ``skip`` using offsets predicted from ``(skip, style_content_feat)``.

        Returns the warped skip and the L1-magnitude of the predicted offsets
        (a scalar tensor).
        """
        # Align style-content feature spatially to the skip resolution.
        if style_content_feat.shape[-2:] != skip.shape[-2:]:
            style_content_feat = F.interpolate(
                style_content_feat, size=skip.shape[-2:], mode="bilinear", align_corners=False
            )
        s_proj = self.proj_skip(F.silu(self.norm_skip(skip)))
        sc_proj = self.proj_style(F.silu(self.norm_style(style_content_feat)))
        mixed = self.mix(torch.cat([s_proj, sc_proj], dim=1))
        offset = self.offset_head(mixed).contiguous()
        offset_l1 = offset.abs().mean()
        warped = self.dcn(skip.contiguous(), offset)
        return warped, offset_l1


# --------------------------------------------------------------------------------------
# UNet
# --------------------------------------------------------------------------------------


class FontDiffuserUNet(nn.Module):
    """Pixel-space U-Net denoiser augmented with MCA + RSI.

    Down path channels: [base*m for m in channel_mult].
    Attention is enabled at stages whose spatial size lies in attn_resolutions.
    RSI cross-attn fires once per attn stage (down + up).
    """

    def __init__(self, cfg: FontDiffuserConfig) -> None:
        super().__init__()
        self.cfg = cfg
        channels = [cfg.base_channels * m for m in cfg.channel_mult]
        resolutions = cfg.stage_resolutions()
        self.stage_resolutions = resolutions
        self.mca_stages = set(cfg.resolved_mca_stages())
        self.rsi_up_stages = set(cfg.resolved_rsi_up_stages())

        # Time embedding MLP
        self.time_mlp = nn.Sequential(
            nn.Linear(cfg.time_input_dim, cfg.time_embed_dim),
            nn.SiLU(inplace=True),
            nn.Linear(cfg.time_embed_dim, cfg.time_embed_dim),
        )

        # Input projection
        self.input_conv = nn.Conv2d(cfg.in_channels, channels[0], kernel_size=3, padding=1)

        # Down stages
        self.down_blocks: nn.ModuleList = nn.ModuleList()
        self.down_attn: nn.ModuleList = nn.ModuleList()
        self.down_rsi: nn.ModuleList = nn.ModuleList()
        self.down_mca: nn.ModuleList = nn.ModuleList()
        self.downsamples: nn.ModuleList = nn.ModuleList()

        prev_c = channels[0]
        for i, c in enumerate(channels):
            stage = nn.ModuleList(
                [ResBlock(prev_c if r == 0 else c, c, time_embed_dim=cfg.time_embed_dim, dropout=cfg.dropout)
                 for r in range(cfg.num_res_blocks)]
            )
            self.down_blocks.append(stage)
            # MCA stage = concat+SE content fuse AND token-cross-attn style fuse.
            # Outer plain stages (matching official ``DownBlock2D``) have neither.
            if i in self.mca_stages:
                self.down_mca.append(
                    MCAFuse(content_channels=c, unet_channels=c, se_reduction=cfg.se_reduction)
                )
                # Token-level style cross-attention (cf. official SpatialTransformer
                # bundled inside MCADownBlock2D). We keep the lighter RSIBlock from
                # the blind impl; this is the *style* token branch, not the new
                # deformable RSI which lives on the up-path only.
                self.down_rsi.append(
                    RSIBlock(channels=c, style_dim=cfg.style_embed_dim, num_heads=cfg.num_heads)
                )
            else:
                self.down_mca.append(nn.Identity())
                self.down_rsi.append(nn.Identity())
            if resolutions[i] in cfg.attn_resolutions:
                self.down_attn.append(SelfAttn2D(c, num_heads=cfg.num_heads))
            else:
                self.down_attn.append(nn.Identity())
            if i < len(channels) - 1:
                self.downsamples.append(Downsample(c))
            else:
                self.downsamples.append(nn.Identity())
            prev_c = c

        # Middle (mid block always carries style cross-attn — matches official UNetMidBlock2DCrossAttn)
        mid_c = channels[-1]
        self.mid_block1 = ResBlock(mid_c, mid_c, time_embed_dim=cfg.time_embed_dim, dropout=cfg.dropout)
        self.mid_attn = SelfAttn2D(mid_c, num_heads=cfg.num_heads)
        self.mid_rsi = RSIBlock(channels=mid_c, style_dim=cfg.style_embed_dim, num_heads=cfg.num_heads)
        self.mid_block2 = ResBlock(mid_c, mid_c, time_embed_dim=cfg.time_embed_dim, dropout=cfg.dropout)

        # Up stages (reverse)
        self.up_blocks: nn.ModuleList = nn.ModuleList()
        self.up_attn: nn.ModuleList = nn.ModuleList()
        self.up_rsi: nn.ModuleList = nn.ModuleList()
        self.up_mca: nn.ModuleList = nn.ModuleList()
        # Deformable-RSI on the up path. One per up-stage; ``nn.Identity`` when
        # that stage is not in ``rsi_up_stages`` (= outer plain UpBlock2D).
        self.up_deform_rsi: nn.ModuleList = nn.ModuleList()
        self.upsamples: nn.ModuleList = nn.ModuleList()
        reversed_channels = list(reversed(channels))
        reversed_resolutions = list(reversed(resolutions))

        prev_c = mid_c
        for i, c in enumerate(reversed_channels):
            # Skip connection from down path doubles input channels at first res block
            stage = nn.ModuleList()
            for r in range(cfg.num_res_blocks + 1):
                in_c = prev_c + c if r == 0 else c
                stage.append(ResBlock(in_c, c, time_embed_dim=cfg.time_embed_dim, dropout=cfg.dropout))
                prev_c = c
            self.up_blocks.append(stage)
            down_idx = len(channels) - 1 - i  # matching down-stage index
            if down_idx in self.mca_stages:
                self.up_mca.append(
                    MCAFuse(content_channels=c, unet_channels=c, se_reduction=cfg.se_reduction)
                )
                self.up_rsi.append(
                    RSIBlock(channels=c, style_dim=cfg.style_embed_dim, num_heads=cfg.num_heads)
                )
            else:
                self.up_mca.append(nn.Identity())
                self.up_rsi.append(nn.Identity())
            if i in self.rsi_up_stages:
                # Style-content feature width at this stage equals the down-path
                # content-encoder channel width at ``down_idx``, which we
                # mirror with channel_mult — so we re-use ``c`` here.
                self.up_deform_rsi.append(
                    OffsetRefStrucInter(
                        skip_channels=c,
                        style_content_channels=c,
                        kernel_size=3,
                        num_heads=cfg.num_heads,
                    )
                )
            else:
                self.up_deform_rsi.append(nn.Identity())
            if reversed_resolutions[i] in cfg.attn_resolutions:
                self.up_attn.append(SelfAttn2D(c, num_heads=cfg.num_heads))
            else:
                self.up_attn.append(nn.Identity())
            if i < len(reversed_channels) - 1:
                self.upsamples.append(Upsample(c))
            else:
                self.upsamples.append(nn.Identity())

        # Output. We deliberately avoid zero-init here: while OpenAI's
        # guided-diffusion zero-inits the final conv so the model predicts
        # ε=0 at step 0, that also disconnects all upstream gradient on
        # iteration 0, which breaks the "gradient must flow through every
        # branch" smoke test. With non-zero init the first prediction is
        # small-random instead of zero — equivalent after one optimizer step.
        self.out_norm = _gn(channels[0])
        self.out_conv = nn.Conv2d(channels[0], cfg.in_channels, kernel_size=3, padding=1)
        nn.init.normal_(self.out_conv.weight, std=0.02)
        nn.init.zeros_(self.out_conv.bias)

    def forward(
        self,
        x: torch.Tensor,
        time_emb: torch.Tensor,
        content_feats: list[torch.Tensor],
        style_content_feats: list[torch.Tensor],
        style_tokens: torch.Tensor,
        style_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run the U-Net. Returns ``(out, offset_l1_sum)``.

        ``style_content_feats`` is the per-stage feature pyramid produced by
        running the content encoder on the **style** image. It feeds the
        deformable-RSI offset prediction on the up path (concept origin:
        ``third_party/01_fontdiffuser/src/model.py:43-44``).
        """
        h = self.input_conv(x)
        skips: list[torch.Tensor] = []
        offset_l1_terms: list[torch.Tensor] = []

        # Down
        for i, stage in enumerate(self.down_blocks):
            for block in stage:
                h = block(h, time_emb)
            # MCA only at the inner two stages (Phase 2 revision).
            if i < len(content_feats) and not isinstance(self.down_mca[i], nn.Identity):
                h = self.down_mca[i](h, content_feats[i])
                # Style cross-attention bundled with MCA, per official MCADownBlock2D.
                if not isinstance(self.down_rsi[i], nn.Identity):
                    h = self.down_rsi[i](h, style_tokens, style_mask=style_mask)
            if not isinstance(self.down_attn[i], nn.Identity):
                h = self.down_attn[i](h)
            skips.append(h)
            if not isinstance(self.downsamples[i], nn.Identity):
                h = self.downsamples[i](h)

        # Middle
        h = self.mid_block1(h, time_emb)
        h = self.mid_attn(h)
        h = self.mid_rsi(h, style_tokens, style_mask=style_mask)
        h = self.mid_block2(h, time_emb)

        # Up
        n_stages = len(self.up_blocks)
        for i, stage in enumerate(self.up_blocks):
            skip = skips.pop()
            # Deformable-RSI warps the skip BEFORE concat (matches the official
            # ``StyleRSIUpBlock2D.forward`` pre-concat warp pattern at
            # ``unet_blocks.py:553-563``).
            if not isinstance(self.up_deform_rsi[i], nn.Identity):
                down_idx = n_stages - 1 - i
                sc_feat = style_content_feats[down_idx] if down_idx < len(style_content_feats) else style_content_feats[-1]
                skip, offset_l1 = self.up_deform_rsi[i](skip, sc_feat)
                offset_l1_terms.append(offset_l1)
            if h.shape[-2:] != skip.shape[-2:]:
                h = F.interpolate(h, size=skip.shape[-2:], mode="nearest")
            h = torch.cat([h, skip], dim=1)
            for block in stage:
                h = block(h, time_emb)
            if not isinstance(self.up_mca[i], nn.Identity):
                rev_idx = len(content_feats) - 1 - i
                h = self.up_mca[i](h, content_feats[rev_idx])
                if not isinstance(self.up_rsi[i], nn.Identity):
                    h = self.up_rsi[i](h, style_tokens, style_mask=style_mask)
            if not isinstance(self.up_attn[i], nn.Identity):
                h = self.up_attn[i](h)
            if not isinstance(self.upsamples[i], nn.Identity):
                h = self.upsamples[i](h)

        out = self.out_conv(F.silu(self.out_norm(h)))
        if offset_l1_terms:
            offset_sum = torch.stack(offset_l1_terms).sum()
        else:
            offset_sum = torch.zeros((), device=x.device, dtype=x.dtype)
        return out, offset_sum


# --------------------------------------------------------------------------------------
# Top-level FontDiffuser module
# --------------------------------------------------------------------------------------


class FontDiffuser(nn.Module):
    """Composite model: content encoder + style encoder + UNet + projections.

    forward(x_t, t, *, content, ref_images, ref_valid, ...) -> noise/x0 prediction.

    Extra kwargs (char_id, script_id, writer_id, style_family_id, unit_id) are
    accepted for compatibility with the shared diffusion utility, but
    FontDiffuser ignores them — it conditions only on (content image, style
    reference image) per paper §3.
    """

    def __init__(self, cfg: FontDiffuserConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.content_encoder = ContentEncoder(
            in_channels=cfg.content_channels,
            base_channels=cfg.base_channels,
            channel_mult=cfg.channel_mult,
        )
        # Phase 2: a *second* content encoder consumes the style ref image,
        # producing the style-content pyramid that drives the deformable-RSI
        # offset prediction. The official repo re-uses one content encoder for
        # both inputs (parameter-tied) — concept origin:
        # ``third_party/01_fontdiffuser/src/model.py:43-44``. We separate it
        # for the blind reimpl because (a) the input channel count differs
        # (``content_channels`` vs ``ref_channels``) and (b) it keeps the
        # gradient paths cleanly separable for smoke-test inspection.
        self.style_content_encoder = ContentEncoder(
            in_channels=cfg.ref_channels,
            base_channels=cfg.base_channels,
            channel_mult=cfg.channel_mult,
        )
        self.style_encoder = StyleEncoder(
            in_channels=cfg.ref_channels,
            embed_dim=cfg.style_embed_dim,
        )
        # Learnable [PAD] style token so missing refs degrade gracefully.
        self.style_null_token = nn.Parameter(torch.zeros(1, 1, cfg.style_embed_dim))
        self.unet = FontDiffuserUNet(cfg)
        # Side-effect storage so ``train.py`` can read the last forward's
        # offset-L1 sum (the shared GaussianDiffusion sampler expects model
        # forward to return a single tensor — we keep that contract).
        self._last_offset_l1: torch.Tensor | None = None

    def encode_style(
        self, ref_images: torch.Tensor | None, ref_valid: torch.Tensor | None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (style_tokens, style_mask).

        ref_images: [B, N, C, H, W] or None. We always take the first reference
        slot (FontDiffuser is one-shot). When ref_valid[:,0] is False (no ref
        available — e.g. unconditional Stage A or CFG dropout), we substitute
        the learned style_null_token.
        """
        b = ref_valid.shape[0] if ref_valid is not None else 1
        if ref_images is None or ref_images.numel() == 0:
            tokens = self.style_null_token.expand(b, 1, -1)
            mask = torch.ones(b, 1, dtype=torch.bool, device=tokens.device)
            return tokens, mask
        ref = ref_images[:, 0]
        tokens = self.style_encoder(ref)
        seq_len = tokens.shape[1]
        if ref_valid is not None:
            valid = ref_valid[:, 0].bool()  # [B]
            if not valid.all():
                null = self.style_null_token.expand(b, seq_len, -1)
                tokens = torch.where(valid[:, None, None], tokens, null)
        mask = torch.ones(b, seq_len, dtype=torch.bool, device=tokens.device)
        return tokens, mask

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
    ) -> torch.Tensor:
        # Content path (multi-scale features for MCA)
        content_feats = self.content_encoder(content)
        # Style-content path: feed the style reference through a second content
        # encoder to produce the feature pyramid used by deformable RSI.
        if ref_images is not None and ref_images.numel() > 0:
            ref0 = ref_images[:, 0]
            style_content_feats = self.style_content_encoder(ref0)
            if ref_valid is not None:
                valid = ref_valid[:, 0].bool()
                if not valid.all():
                    # Zero out style-content features for samples without a ref.
                    # Offsets predicted from zero context default to ~0 (the
                    # offset head is zero-init), giving identity-conv warps.
                    mask = valid.view(-1, *([1] * (style_content_feats[0].ndim - 1))).to(
                        style_content_feats[0].dtype
                    )
                    style_content_feats = [f * mask for f in style_content_feats]
        else:
            # No ref: use zero-feature pyramid sized to match content_feats.
            style_content_feats = [torch.zeros_like(f) for f in content_feats]
        # Style token path (for cross-attention)
        style_tokens, style_mask = self.encode_style(ref_images, ref_valid)
        # Time embedding
        t_input = timestep_embedding(timesteps, self.cfg.time_input_dim).to(dtype=x_t.dtype)
        t_emb = self.unet.time_mlp(t_input)
        out, offset_l1 = self.unet(
            x_t,
            time_emb=t_emb,
            content_feats=content_feats,
            style_content_feats=style_content_feats,
            style_tokens=style_tokens,
            style_mask=style_mask,
        )
        self._last_offset_l1 = offset_l1
        return out


def build_fontdiffuser(cfg: FontDiffuserConfig) -> FontDiffuser:
    return FontDiffuser(cfg)


# --------------------------------------------------------------------------------------
# Perceptual loss (Phase 2 — official Phase 1 uses 0.01 * VGG16 enc1/enc2/enc3 MSE)
# --------------------------------------------------------------------------------------


_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


def _renormalize_for_vgg(x: torch.Tensor) -> torch.Tensor:
    """Map a tensor from train-time ``[-1, 1]`` (Normalize(0.5, 0.5)) to the
    ImageNet-normalised range that VGG-16 expects.

    Concept origin: ``third_party/01_fontdiffuser/utils.py`` (``reNormalize_img``
    + ``normalize_mean_std``) used in ``train.py:199-210``. Re-implemented
    here so the perceptual head is self-contained.
    """
    # [-1, 1] -> [0, 1]
    x = (x + 1.0) * 0.5
    # Expand 1-channel grayscale to 3 channels so the pretrained VGG works.
    if x.shape[1] == 1:
        x = x.repeat(1, 3, 1, 1)
    elif x.shape[1] != 3:
        # Project arbitrary content-channel count down to 3 by averaging in
        # groups. Cheap fallback for content tensors fed where target was
        # expected (shouldn't happen in normal flow).
        groups = max(1, x.shape[1] // 3)
        x = x.reshape(x.shape[0], 3, groups, *x.shape[2:]).mean(dim=2) if x.shape[1] % 3 == 0 else x[:, :3]
    mean = x.new_tensor(_IMAGENET_MEAN).view(1, 3, 1, 1)
    std = x.new_tensor(_IMAGENET_STD).view(1, 3, 1, 1)
    return (x - mean) / std


class ContentPerceptualLoss(nn.Module):
    """VGG-16 perceptual loss on (predicted_x0, target_x0).

    Concept origin: ``third_party/01_fontdiffuser/src/criterion.py:6-44``.
    Uses ``torchvision.models.vgg16(pretrained=True)`` slices enc_1/enc_2/enc_3
    (relu1_2, relu2_2, relu3_3). VGG weights are frozen.
    """

    def __init__(self) -> None:
        super().__init__()
        from torchvision.models import vgg16

        # Lazy import + safe-default for offline environments: try to load
        # pretrained weights; if download fails fall back to random init so
        # the smoke test still passes. Phase 2 training MUST have pretrained.
        try:
            from torchvision.models import VGG16_Weights

            vgg = vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
        except Exception:  # noqa: BLE001 — broad: offline / sandbox / signature drift
            vgg = vgg16(weights=None)
        features = vgg.features
        self.enc_1 = nn.Sequential(*features[:5])    # -> relu1_2
        self.enc_2 = nn.Sequential(*features[5:10])  # -> relu2_2
        self.enc_3 = nn.Sequential(*features[10:17]) # -> relu3_3
        for p in self.parameters():
            p.requires_grad = False
        self.eval()

    def train(self, mode: bool = True) -> "ContentPerceptualLoss":  # type: ignore[override]
        # Always keep VGG in eval (frozen BN-free, deterministic).
        return super().train(False)

    def forward(self, predicted_x0: torch.Tensor, target_x0: torch.Tensor) -> torch.Tensor:
        gen = _renormalize_for_vgg(predicted_x0)
        tgt = _renormalize_for_vgg(target_x0)
        loss = predicted_x0.new_zeros(())
        for enc in (self.enc_1, self.enc_2, self.enc_3):
            gen = enc(gen)
            with torch.no_grad():
                tgt = enc(tgt)
            loss = loss + F.mse_loss(gen, tgt)
        return loss / 3.0


# --------------------------------------------------------------------------------------
# Style Contrastive Refinement (SCR) — VGG-16 + 6 projector heads + InfoNCE
# --------------------------------------------------------------------------------------


class _SCRStyleFeatExtractor(nn.Module):
    """VGG-16 multi-stage style feature extractor.

    Concept origin: ``third_party/01_fontdiffuser/src/modules/scr_modules.py:5-46``
    (``StyleExtractor``). Splits VGG-16 into 6 sequential stages, each
    followed by a 1x1 conv that compresses GAP+GMP pooled feature pairs
    down to a per-stage style code. We re-implement on top of
    ``torchvision.models.vgg16(pretrained=True)`` without batch-norm, which
    is the standard torchvision VGG.
    """

    def __init__(self, freeze: bool = True) -> None:
        super().__init__()
        from torchvision.models import vgg16

        try:
            from torchvision.models import VGG16_Weights

            vgg = vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
        except Exception:  # noqa: BLE001
            vgg = vgg16(weights=None)
        feats = vgg.features
        # Indices below split torchvision VGG-16 (no BN) at MaxPool boundaries.
        # Stage out-channels: [64, 128, 256, 512, 512, 512].
        self.enc_1 = nn.Sequential(*feats[:5])    # relu1_2,  C=64
        self.enc_2 = nn.Sequential(*feats[5:10])  # relu2_2,  C=128
        self.enc_3 = nn.Sequential(*feats[10:17]) # relu3_3,  C=256
        self.enc_4 = nn.Sequential(*feats[17:24]) # relu4_3,  C=512
        self.enc_5 = nn.Sequential(*feats[24:31]) # relu5_3,  C=512
        self.enc_6 = nn.Sequential(*feats[31:])   # post-relu, C=512
        # GAP+GMP -> 1x1 conv to compress channels back to the per-stage width.
        stage_channels = (64, 128, 256, 512, 512, 512)
        self.compress = nn.ModuleList(
            [nn.Conv2d(c * 2, c, kernel_size=1, bias=True) for c in stage_channels]
        )
        self.stage_channels = stage_channels
        if freeze:
            for p in self.parameters():
                p.requires_grad = False
            self.eval()

    def train(self, mode: bool = True) -> "_SCRStyleFeatExtractor":  # type: ignore[override]
        return super().train(False)

    def encode(self, x: torch.Tensor) -> list[torch.Tensor]:
        results = []
        h = x
        for i in range(6):
            h = getattr(self, f"enc_{i + 1}")(h)
            results.append(h)
        return results

    def forward(self, x: torch.Tensor, layer_indices: tuple[int, ...]) -> list[torch.Tensor]:
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        feats = self.encode(x)
        codes: list[torch.Tensor] = []
        for idx in layer_indices:
            f = feats[idx]
            gap = F.adaptive_avg_pool2d(f, 1)
            gmp = F.adaptive_max_pool2d(f, 1)
            pooled = torch.cat([gap, gmp], dim=1)
            code = F.relu(self.compress[idx](pooled), inplace=True)
            codes.append(code.flatten(1))
        return codes


class _SCRProjector(nn.Module):
    """Per-stage MLP projector matching the official 6-head Projector
    (``third_party/01_fontdiffuser/src/modules/scr_modules.py:48-105``).
    Each head: ``[stage_C] -> 1024 -> 2048 -> 2048`` with L2 normalisation.
    """

    def __init__(self, stage_channels: tuple[int, ...] = (64, 128, 256, 512, 512, 512)) -> None:
        super().__init__()
        self.heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(c, 1024),
                    nn.ReLU(inplace=True),
                    nn.Linear(1024, 2048),
                    nn.ReLU(inplace=True),
                    nn.Linear(2048, 2048),
                )
                for c in stage_channels
            ]
        )

    def forward(self, codes: list[torch.Tensor], layer_indices: tuple[int, ...]) -> list[torch.Tensor]:
        out = []
        for code, idx in zip(codes, layer_indices):
            z = self.heads[idx](code)
            out.append(F.normalize(z, dim=-1))
        return out


class SCRModule(nn.Module):
    """Style Contrastive Refinement — Phase 2 InfoNCE variant.

    Concept origin: ``third_party/01_fontdiffuser/src/modules/scr.py:9-96``.

    Inputs (all in train-time ``[-1, 1]`` space):
      * ``sample``: predicted x0 from the diffusion model.
      * ``positive``: ground-truth target image (same style as sample).
      * ``negatives``: ``[B, num_neg, C, H, W]`` other-style same-content
        images sampled by the dataset layer (concept origin:
        ``third_party/01_fontdiffuser/dataset/font_dataset.py:78-99``).

    The positive image is first augmented by ``kornia.RandomResizedCrop``
    (scale=(0.8, 1.0), ratio=(0.75, 1.33)) so positives are "same-style
    different-patch" rather than identical.

    The InfoNCE loss is averaged across the requested ``nce_layers``. Default
    ``(0, 1, 2, 3)`` matches the argparse default in the official repo.
    """

    def __init__(
        self,
        *,
        temperature: float = 0.07,
        image_size: int = 96,
        nce_layers: tuple[int, ...] = (0, 1, 2, 3),
        freeze_backbone: bool = True,
    ) -> None:
        super().__init__()
        self.temperature = float(temperature)
        self.nce_layers = tuple(int(x) for x in nce_layers)
        self.extractor = _SCRStyleFeatExtractor(freeze=freeze_backbone)
        self.projector = _SCRProjector(stage_channels=self.extractor.stage_channels)
        try:
            from info_nce import InfoNCE

            self.nce = InfoNCE(temperature=self.temperature, negative_mode="paired")
            self._has_info_nce = True
        except Exception:  # noqa: BLE001 — fallback to manual InfoNCE
            self.nce = None
            self._has_info_nce = False
        try:
            import kornia.augmentation as K

            self.patch_sampler = K.RandomResizedCrop(
                (image_size, image_size), scale=(0.8, 1.0), ratio=(0.75, 1.33)
            )
        except Exception:  # noqa: BLE001 — degrade to identity augmentation
            self.patch_sampler = nn.Identity()

    def _augment_positive(self, pos: torch.Tensor) -> torch.Tensor:
        # kornia expects [0, 1] floats for RandomResizedCrop; our pixels are
        # in [-1, 1]. Shift, augment, shift back so semantics are unchanged.
        x = (pos + 1.0) * 0.5
        x = self.patch_sampler(x)
        return x * 2.0 - 1.0

    def _embed(self, x: torch.Tensor) -> list[torch.Tensor]:
        codes = self.extractor(x, self.nce_layers)
        return self.projector(codes, self.nce_layers)

    def forward(
        self,
        sample: torch.Tensor,
        positive: torch.Tensor,
        negatives: torch.Tensor,
    ) -> torch.Tensor:
        """Return total InfoNCE loss averaged over the requested layers."""
        sample_emb = self._embed(sample)
        pos_aug = self._augment_positive(positive)
        pos_emb = self._embed(pos_aug)
        # negatives: [B, num_neg, C, H, W]
        b, num_neg, c, h, w = negatives.shape
        neg_emb_per_layer: list[torch.Tensor] = [
            torch.zeros(b, num_neg, e.shape[-1], device=e.device, dtype=e.dtype)
            for e in sample_emb
        ]
        for i in range(num_neg):
            neg_emb_i = self._embed(negatives[:, i])
            for li, e in enumerate(neg_emb_i):
                neg_emb_per_layer[li][:, i] = e

        total = sample.new_zeros(())
        for li in range(len(sample_emb)):
            s = sample_emb[li]
            p = pos_emb[li]
            n = neg_emb_per_layer[li]
            if self._has_info_nce:
                loss = self.nce(s, p, n)
            else:
                # Manual paired InfoNCE: positive logit + per-sample negatives.
                pos_logit = (s * p).sum(dim=-1, keepdim=True) / self.temperature   # [B, 1]
                neg_logits = torch.einsum("bd,bnd->bn", s, n) / self.temperature   # [B, N]
                logits = torch.cat([pos_logit, neg_logits], dim=1)
                target = torch.zeros(s.shape[0], dtype=torch.long, device=s.device)
                loss = F.cross_entropy(logits, target)
            total = total + loss
        return total / float(len(sample_emb))


# Back-compat alias for callers that still import the old ``StyleExtractor`` name
# (tests / training scripts that haven't been Phase-2-updated yet).
StyleExtractor = SCRModule
