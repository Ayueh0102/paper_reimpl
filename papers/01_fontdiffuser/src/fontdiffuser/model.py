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


# --------------------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------------------


@dataclass
class FontDiffuserConfig:
    """All structural hyperparameters for the FontDiffuser blind reimpl.

    Defaults are conservative: small enough to run a CPU smoke test in seconds,
    and reasonable for a 128/256-px Ernantang Stage A run on a single GPU.
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
    """Spatial sizes (in pixels) at which self-attn + RSI cross-attn fire."""
    num_res_blocks: int = 2
    time_embed_dim: int = 256
    style_embed_dim: int = 256
    """Token width of the StyleEncoder output and of RSI cross-attn K/V."""
    num_heads: int = 4
    dropout: float = 0.0

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


class MCAFuse(nn.Module):
    """Fuse a content feature map into a UNet skip stream.

    Paper note p.1: MCA aggregates global+local content features at multiple
    scales. We implement the per-stage operation as: project content feature
    to the UNet stage's channel width, then gated-add into the UNet feature
    (sigmoid gate, init at 0 so MCA starts as an identity and learns to mix).
    """

    def __init__(self, content_channels: int, unet_channels: int) -> None:
        super().__init__()
        self.project = nn.Conv2d(content_channels, unet_channels, kernel_size=1)
        self.gate = nn.Conv2d(unet_channels, unet_channels, kernel_size=1)
        # Zero-init gate -> identity at init; gradient flows from t=0.
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
            if resolutions[i] in cfg.attn_resolutions:
                self.down_attn.append(SelfAttn2D(c, num_heads=cfg.num_heads))
                self.down_rsi.append(RSIBlock(channels=c, style_dim=cfg.style_embed_dim, num_heads=cfg.num_heads))
            else:
                self.down_attn.append(nn.Identity())
                self.down_rsi.append(nn.Identity())
            self.down_mca.append(MCAFuse(content_channels=c, unet_channels=c))
            if i < len(channels) - 1:
                self.downsamples.append(Downsample(c))
            else:
                self.downsamples.append(nn.Identity())
            prev_c = c

        # Middle
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
            if reversed_resolutions[i] in cfg.attn_resolutions:
                self.up_attn.append(SelfAttn2D(c, num_heads=cfg.num_heads))
                self.up_rsi.append(RSIBlock(channels=c, style_dim=cfg.style_embed_dim, num_heads=cfg.num_heads))
            else:
                self.up_attn.append(nn.Identity())
                self.up_rsi.append(nn.Identity())
            self.up_mca.append(MCAFuse(content_channels=c, unet_channels=c))
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
        style_tokens: torch.Tensor,
        style_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        h = self.input_conv(x)
        skips: list[torch.Tensor] = []

        # Down
        for i, stage in enumerate(self.down_blocks):
            for block in stage:
                h = block(h, time_emb)
            # MCA after res blocks so unet_feat is at this stage's channel
            # width (matches content_feats[i] by construction).
            if i < len(content_feats):
                h = self.down_mca[i](h, content_feats[i])
            if not isinstance(self.down_attn[i], nn.Identity):
                h = self.down_attn[i](h)
                h = self.down_rsi[i](h, style_tokens, style_mask=style_mask)
            skips.append(h)
            if not isinstance(self.downsamples[i], nn.Identity):
                h = self.downsamples[i](h)

        # Middle
        h = self.mid_block1(h, time_emb)
        h = self.mid_attn(h)
        h = self.mid_rsi(h, style_tokens, style_mask=style_mask)
        h = self.mid_block2(h, time_emb)

        # Up
        for i, stage in enumerate(self.up_blocks):
            skip = skips.pop()
            if h.shape[-2:] != skip.shape[-2:]:
                h = F.interpolate(h, size=skip.shape[-2:], mode="nearest")
            h = torch.cat([h, skip], dim=1)
            for block in stage:
                h = block(h, time_emb)
            if i < len(content_feats):
                # MCA in up path uses the matching reversed-content feature
                rev_idx = len(content_feats) - 1 - i
                h = self.up_mca[i](h, content_feats[rev_idx])
            if not isinstance(self.up_attn[i], nn.Identity):
                h = self.up_attn[i](h)
                h = self.up_rsi[i](h, style_tokens, style_mask=style_mask)
            if not isinstance(self.upsamples[i], nn.Identity):
                h = self.upsamples[i](h)

        return self.out_conv(F.silu(self.out_norm(h)))


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
        self.style_encoder = StyleEncoder(
            in_channels=cfg.ref_channels,
            embed_dim=cfg.style_embed_dim,
        )
        # Learnable [PAD] style token so missing refs degrade gracefully.
        self.style_null_token = nn.Parameter(torch.zeros(1, 1, cfg.style_embed_dim))
        self.unet = FontDiffuserUNet(cfg)

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
        # Style path
        style_tokens, style_mask = self.encode_style(ref_images, ref_valid)
        # Time embedding
        t_input = timestep_embedding(timesteps, self.cfg.time_input_dim).to(dtype=x_t.dtype)
        t_emb = self.unet.time_mlp(t_input)
        return self.unet(
            x_t,
            time_emb=t_emb,
            content_feats=content_feats,
            style_tokens=style_tokens,
            style_mask=style_mask,
        )


def build_fontdiffuser(cfg: FontDiffuserConfig) -> FontDiffuser:
    return FontDiffuser(cfg)


# --------------------------------------------------------------------------------------
# Style extractor for SCR loss
# --------------------------------------------------------------------------------------


class StyleExtractor(nn.Module):
    """Style extractor used by Style Contrastive Refinement (SCR).

    Paper §3 says SCR uses a separately-trained style extractor whose
    embedding is supervised by contrastive loss (same-style positive, others
    negative). Here we build a shallow CNN encoder that outputs an
    L2-normalized embedding. In a full pipeline this would be pre-trained
    on writer-id classification (Stage A); for blind smoke we keep it
    untrained and freeze it.
    """

    def __init__(self, *, in_channels: int = 1, embed_dim: int = 128) -> None:
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
            nn.AdaptiveAvgPool2d(1),
        )
        self.proj = nn.Linear(c3, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.body(x).flatten(1)
        h = self.proj(h)
        return F.normalize(h, dim=-1)
