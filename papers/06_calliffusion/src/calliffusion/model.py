"""Conditional DDPM U-Net for Calliffusion.

Architecture summary (per ``paper_notes/06.md`` §2):
- 4 down stages with channel widths [320, 640, 1280, 1280] (configurable)
- 1 mid block, 4 up stages mirroring the down side
- Each stage has ``num_res_blocks`` ResNet blocks + 1 self-attn + 1 cross-attn
- Time embedding: sin/cos PE → MLP → injected as bias in every ResBlock
- Cross-attention context comes from a text encoder (BERT or stub) and is a
  ``[B, L, hidden]`` tensor whose hidden dim is projected to each stage's
  channel count inside the cross-attn block.
- Predicts ε (noise), single grayscale channel (in/out channels configurable).
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# small building blocks
# ---------------------------------------------------------------------------


class SinusoidalTimeEmbedding(nn.Module):
    """Sin/cos time embedding then MLP to ``time_emb_dim``."""

    def __init__(self, *, base_dim: int = 320, time_emb_dim: int = 1280) -> None:
        super().__init__()
        if base_dim % 2 != 0:
            raise ValueError("base_dim must be even for sin/cos embedding")
        self.base_dim = base_dim
        self.mlp = nn.Sequential(
            nn.Linear(base_dim, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.base_dim // 2
        device = t.device
        freqs = torch.exp(
            -math.log(10_000.0) * torch.arange(half, device=device, dtype=torch.float32) / half
        )
        args = t.float()[:, None] * freqs[None, :]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        return self.mlp(emb)


class ResBlock(nn.Module):
    """GroupNorm-SiLU-Conv ResBlock with additive time-emb bias."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        time_emb_dim: int = 1280,
        groups: int = 32,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        groups_in = math.gcd(groups, in_channels) or 1
        groups_out = math.gcd(groups, out_channels) or 1
        self.norm1 = nn.GroupNorm(groups_in, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.time_mlp = nn.Linear(time_emb_dim, out_channels)
        self.norm2 = nn.GroupNorm(groups_out, out_channels)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        if in_channels != out_channels:
            self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.skip = nn.Identity()

    def forward(self, h: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        x = self.conv1(F.silu(self.norm1(h)))
        x = x + self.time_mlp(F.silu(t_emb))[:, :, None, None]
        x = self.conv2(self.dropout(F.silu(self.norm2(x))))
        return x + self.skip(h)


class SpatialSelfAttention(nn.Module):
    """Multi-head self-attention over (H*W) spatial tokens."""

    def __init__(self, channels: int, *, num_heads: int = 8) -> None:
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        groups = math.gcd(32, channels) or 1
        self.norm = nn.GroupNorm(groups, channels)
        self.to_qkv = nn.Linear(channels, channels * 3)
        self.to_out = nn.Linear(channels, channels)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        b, c, hh, ww = h.shape
        x = self.norm(h).reshape(b, c, hh * ww).transpose(1, 2)  # [B, N, C]
        qkv = self.to_qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        head_dim = c // self.num_heads
        # SDPA-friendly reshape
        q = q.reshape(b, hh * ww, self.num_heads, head_dim).transpose(1, 2)
        k = k.reshape(b, hh * ww, self.num_heads, head_dim).transpose(1, 2)
        v = v.reshape(b, hh * ww, self.num_heads, head_dim).transpose(1, 2)
        out = F.scaled_dot_product_attention(q, k, v)
        out = out.transpose(1, 2).reshape(b, hh * ww, c)
        out = self.to_out(out)
        return h + out.transpose(1, 2).reshape(b, c, hh, ww)


class SpatialCrossAttention(nn.Module):
    """Cross-attention: image queries, BERT context keys/values."""

    def __init__(self, channels: int, *, context_dim: int = 768, num_heads: int = 8) -> None:
        super().__init__()
        self.channels = channels
        self.context_dim = context_dim
        self.num_heads = num_heads
        groups = math.gcd(32, channels) or 1
        self.norm = nn.GroupNorm(groups, channels)
        # Names chosen so the LoRA matcher (``to_q/k/v/out``) works out of the
        # box at Stage C / one-shot transfer time.
        self.to_q = nn.Linear(channels, channels, bias=False)
        self.to_k = nn.Linear(context_dim, channels, bias=False)
        self.to_v = nn.Linear(context_dim, channels, bias=False)
        self.to_out = nn.Linear(channels, channels)

    def forward(
        self,
        h: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        b, c, hh, ww = h.shape
        x = self.norm(h).reshape(b, c, hh * ww).transpose(1, 2)  # [B, N, C]
        q = self.to_q(x)
        k = self.to_k(context)
        v = self.to_v(context)
        head_dim = c // self.num_heads
        q = q.reshape(b, hh * ww, self.num_heads, head_dim).transpose(1, 2)
        k = k.reshape(b, context.shape[1], self.num_heads, head_dim).transpose(1, 2)
        v = v.reshape(b, context.shape[1], self.num_heads, head_dim).transpose(1, 2)
        attn_mask = None
        if context_mask is not None:
            # mask: [B, L] with 1 for valid. Broadcast to [B, 1, 1, L].
            attn_mask = (context_mask == 0).reshape(b, 1, 1, context.shape[1])
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        out = out.transpose(1, 2).reshape(b, hh * ww, c)
        out = self.to_out(out)
        return h + out.transpose(1, 2).reshape(b, c, hh, ww)


class Downsample(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.op = nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.op(h)


class Upsample(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.op = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        h = F.interpolate(h, scale_factor=2.0, mode="nearest")
        return self.op(h)


# ---------------------------------------------------------------------------
# main model
# ---------------------------------------------------------------------------


@dataclass
class CalliffusionUNetConfig:
    image_size: int = 64
    in_channels: int = 1
    out_channels: int = 1
    base_channels: int = 320
    channel_mult: list[int] = field(default_factory=lambda: [1, 2, 4, 4])
    num_res_blocks: int = 2
    time_emb_dim: int = 1280
    context_dim: int = 768
    num_heads: int = 8
    dropout: float = 0.0

    @property
    def stage_channels(self) -> list[int]:
        return [self.base_channels * m for m in self.channel_mult]

    def validate(self) -> None:
        if self.image_size % (2 ** (len(self.channel_mult) - 1)) != 0:
            raise ValueError(
                f"image_size {self.image_size} must be divisible by 2^(num_stages-1) "
                f"= {2 ** (len(self.channel_mult) - 1)}"
            )
        if self.base_channels % self.num_heads != 0:
            raise ValueError(
                f"base_channels {self.base_channels} must be divisible by num_heads {self.num_heads}"
            )
        # Each stage_channels[i] = base_channels * channel_mult[i] flows into
        # SpatialSelfAttention / SpatialCrossAttention, which reshape on
        # ``head_dim = channels // num_heads``. A non-divisible stage width
        # would crash deep inside ``forward`` with a cryptic reshape error
        # rather than at startup.
        for mult in self.channel_mult:
            stage_ch = self.base_channels * int(mult)
            if stage_ch % self.num_heads != 0:
                raise ValueError(
                    f"stage channel {stage_ch} (base {self.base_channels} x mult {mult}) "
                    f"not divisible by num_heads {self.num_heads}"
                )


class CalliffusionUNet(nn.Module):
    """DDPM U-Net with BERT cross-attention at every stage."""

    def __init__(self, cfg: CalliffusionUNetConfig) -> None:
        super().__init__()
        cfg.validate()
        self.cfg = cfg
        ch = cfg.stage_channels

        self.time_embed = SinusoidalTimeEmbedding(
            base_dim=cfg.base_channels, time_emb_dim=cfg.time_emb_dim
        )

        self.conv_in = nn.Conv2d(cfg.in_channels, ch[0], kernel_size=3, padding=1)

        self.down_blocks = nn.ModuleList()
        self.down_attns = nn.ModuleList()
        self.down_crosses = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        in_ch = ch[0]
        skip_channels: list[int] = []
        for stage_idx, stage_ch in enumerate(ch):
            stage_res = nn.ModuleList()
            stage_attn = nn.ModuleList()
            stage_cross = nn.ModuleList()
            for _ in range(cfg.num_res_blocks):
                stage_res.append(
                    ResBlock(
                        in_ch,
                        stage_ch,
                        time_emb_dim=cfg.time_emb_dim,
                        dropout=cfg.dropout,
                    )
                )
                stage_attn.append(SpatialSelfAttention(stage_ch, num_heads=cfg.num_heads))
                stage_cross.append(
                    SpatialCrossAttention(
                        stage_ch, context_dim=cfg.context_dim, num_heads=cfg.num_heads
                    )
                )
                skip_channels.append(stage_ch)
                in_ch = stage_ch
            self.down_blocks.append(stage_res)
            self.down_attns.append(stage_attn)
            self.down_crosses.append(stage_cross)
            if stage_idx != len(ch) - 1:
                self.downsamples.append(Downsample(in_ch))
                skip_channels.append(in_ch)
            else:
                self.downsamples.append(nn.Identity())

        # Mid block: ResBlock → SelfAttn → CrossAttn → ResBlock
        self.mid_res1 = ResBlock(in_ch, in_ch, time_emb_dim=cfg.time_emb_dim, dropout=cfg.dropout)
        self.mid_attn = SpatialSelfAttention(in_ch, num_heads=cfg.num_heads)
        self.mid_cross = SpatialCrossAttention(
            in_ch, context_dim=cfg.context_dim, num_heads=cfg.num_heads
        )
        self.mid_res2 = ResBlock(in_ch, in_ch, time_emb_dim=cfg.time_emb_dim, dropout=cfg.dropout)

        # Up path mirrors down
        self.up_blocks = nn.ModuleList()
        self.up_attns = nn.ModuleList()
        self.up_crosses = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        for stage_idx, stage_ch in list(enumerate(ch))[::-1]:
            stage_res = nn.ModuleList()
            stage_attn = nn.ModuleList()
            stage_cross = nn.ModuleList()
            for _block_idx in range(cfg.num_res_blocks):
                skip_ch = skip_channels.pop()
                stage_res.append(
                    ResBlock(
                        in_ch + skip_ch,
                        stage_ch,
                        time_emb_dim=cfg.time_emb_dim,
                        dropout=cfg.dropout,
                    )
                )
                stage_attn.append(SpatialSelfAttention(stage_ch, num_heads=cfg.num_heads))
                stage_cross.append(
                    SpatialCrossAttention(
                        stage_ch, context_dim=cfg.context_dim, num_heads=cfg.num_heads
                    )
                )
                in_ch = stage_ch
            # Pop the downsample-induced skip if present.
            if stage_idx != 0:
                skip_ch = skip_channels.pop()
                # Fuse downsample skip into next ResBlock at the start of next iter.
                # Implemented as an extra ResBlock that consumes the skip then we
                # upsample. We reuse ResBlock by concatenating channels.
                stage_res.append(
                    ResBlock(
                        in_ch + skip_ch,
                        stage_ch,
                        time_emb_dim=cfg.time_emb_dim,
                        dropout=cfg.dropout,
                    )
                )
                stage_attn.append(SpatialSelfAttention(stage_ch, num_heads=cfg.num_heads))
                stage_cross.append(
                    SpatialCrossAttention(
                        stage_ch, context_dim=cfg.context_dim, num_heads=cfg.num_heads
                    )
                )
                in_ch = stage_ch
                self.upsamples.append(Upsample(in_ch))
            else:
                self.upsamples.append(nn.Identity())
            self.up_blocks.append(stage_res)
            self.up_attns.append(stage_attn)
            self.up_crosses.append(stage_cross)

        groups_out = math.gcd(32, ch[0]) or 1
        self.norm_out = nn.GroupNorm(groups_out, in_ch)
        self.conv_out = nn.Conv2d(in_ch, cfg.out_channels, kernel_size=3, padding=1)
        # Zero-init final conv so the model starts as a near-identity predictor
        # of zero noise — a common DDPM trick that stabilises early training.
        nn.init.zeros_(self.conv_out.weight)
        nn.init.zeros_(self.conv_out.bias)

    def forward(
        self,
        x_t: torch.Tensor,
        timesteps: torch.Tensor,
        *,
        context: torch.Tensor,
        context_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Predict ε given noisy image, timestep, and text context."""
        t_emb = self.time_embed(timesteps)
        h = self.conv_in(x_t)
        skips: list[torch.Tensor] = []

        for stage_idx, (res_blocks, attns, crosses) in enumerate(
            zip(self.down_blocks, self.down_attns, self.down_crosses, strict=True)
        ):
            for r, a, c in zip(res_blocks, attns, crosses, strict=True):
                h = r(h, t_emb)
                h = a(h)
                h = c(h, context, context_mask)
                skips.append(h)
            ds = self.downsamples[stage_idx]
            if not isinstance(ds, nn.Identity):
                h = ds(h)
                skips.append(h)

        h = self.mid_res1(h, t_emb)
        h = self.mid_attn(h)
        h = self.mid_cross(h, context, context_mask)
        h = self.mid_res2(h, t_emb)

        # Up path: blocks are stored bottom-up. For each up stage we first run
        # ``num_res_blocks`` blocks consuming a skip each, then (if present)
        # an extra block consuming the downsample skip, then upsample.
        for stage_idx, (res_blocks, attns, crosses) in enumerate(
            zip(self.up_blocks, self.up_attns, self.up_crosses, strict=True)
        ):
            for r, a, c in zip(res_blocks, attns, crosses, strict=True):
                skip = skips.pop()
                h = r(torch.cat([h, skip], dim=1), t_emb)
                h = a(h)
                h = c(h, context, context_mask)
            us = self.upsamples[stage_idx]
            if not isinstance(us, nn.Identity):
                h = us(h)

        h = F.silu(self.norm_out(h))
        return self.conv_out(h)

    # ------------------------------------------------------------------
    # convenience helpers
    # ------------------------------------------------------------------

    def trainable_parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def freeze_all(self) -> None:
        for p in self.parameters():
            p.requires_grad = False

    def unfreeze(self, predicate: Callable[[str], bool] = lambda name: True) -> None:
        for name, p in self.named_parameters():
            if predicate(name):
                p.requires_grad = True


def build_unet_from_yaml(model_cfg: dict) -> CalliffusionUNet:
    """Build the U-Net from a parsed model.yaml dictionary."""
    section = model_cfg.get("unet", model_cfg)
    cfg = CalliffusionUNetConfig(
        image_size=int(section.get("image_size", 64)),
        in_channels=int(section.get("in_channels", 1)),
        out_channels=int(section.get("out_channels", 1)),
        base_channels=int(section.get("base_channels", 320)),
        channel_mult=list(section.get("channel_mult", [1, 2, 4, 4])),
        num_res_blocks=int(section.get("num_res_blocks", 2)),
        time_emb_dim=int(section.get("time_emb_dim", 1280)),
        context_dim=int(section.get("context_dim", 768)),
        num_heads=int(section.get("num_heads", 8)),
        dropout=float(section.get("dropout", 0.0)),
    )
    return CalliffusionUNet(cfg)


def cross_attention_modules(model: nn.Module) -> Iterable[SpatialCrossAttention]:
    """Yield every cross-attention block (used for LoRA targeting docs)."""
    for sub in model.modules():
        if isinstance(sub, SpatialCrossAttention):
            yield sub
