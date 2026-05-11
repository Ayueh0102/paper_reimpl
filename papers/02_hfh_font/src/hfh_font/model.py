"""HFH-Font blind reimplementation — model definitions.

Components:
- ``TinyVAE``: 8× downsampling VAE encoder/decoder. Operates on grayscale glyphs
  and produces a low-res latent. Lightweight stand-in for whichever VAE the
  paper used (the note does not name one).
- ``ComponentEncoder``: ResNet-ish CNN over reference glyph stack. Emits
  ``K_comp`` tokens per reference. Output shape ``(B, N_refs * K_comp, d_ctx)``.
- ``LatentUNet``: standard residual U-Net in latent space with:
    * sinusoidal time embedding + AdaLN-Zero conditioning,
    * cross-attention against the component tokens at each mid block,
    * char / writer / script embeddings summed into the cond MLP.
- ``HFHFontModel``: composes the above and exposes ``forward`` /
  ``compute_loss`` / ``sample`` in a shape compatible with
  ``paper_reimpl_shared.diffusion.GaussianDiffusion``.

Every non-trivial choice is logged in ``reports/blind_impl.md`` as either
``[paper-cited]`` or ``[guessed]``.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

_logger = logging.getLogger(__name__)

__all__ = [
    "ModelConfig",
    "TinyVAE",
    "ComponentEncoder",
    "LatentUNet",
    "HFHFontModel",
    "build_model",
]


@dataclass
class ModelConfig:
    """Top-level config for ``HFHFontModel``. Mirrors ``configs/model.yaml``."""

    image_size: int = 128
    in_channels: int = 1
    content_channels: int = 3
    latent_channels: int = 4
    vae_down_factor: int = 8
    base_channels: int = 64
    channel_mult: tuple[int, ...] = (1, 2, 4)
    num_res_blocks: int = 2
    attention_resolutions: tuple[int, ...] = (4, 2)
    d_ctx: int = 256
    n_heads: int = 4
    char_vocab_size: int = 4096
    writer_vocab_size: int = 32
    script_vocab_size: int = 5
    n_refs: int = 4
    components_per_ref: int = 8
    dropout: float = 0.1
    diffusion_timesteps: int = 1000
    diffusion_target: Literal["x0", "epsilon"] = "x0"
    sr_enabled: bool = False
    sr_scale: int = 2

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ModelConfig:
        kwargs: dict[str, Any] = {}
        for field_name in cls.__dataclass_fields__:
            if field_name in data:
                value = data[field_name]
                if field_name in {"channel_mult", "attention_resolutions"} and value is not None:
                    value = tuple(value)
                kwargs[field_name] = value
        return cls(**kwargs)


# ---------------------------------------------------------------------------
# VAE
# ---------------------------------------------------------------------------


class _VAEResBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.norm1 = nn.GroupNorm(8, in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(8, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)
        self.skip = nn.Conv2d(in_ch, out_ch, kernel_size=1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.skip(x)


class TinyVAE(nn.Module):
    """Tiny conv VAE. Not the SOTA — a deterministic AE-style stand-in.

    ``forward`` returns a deterministic latent (mean only) so we can keep the
    smoke test free of KL-vs-recon tuning. The full KL path is exposed via
    ``encode_distribution`` and ``kl_loss`` for future Stage-A work but is
    *not* used by ``HFHFontModel.compute_loss`` until SDS distillation, per
    the paper note's claim that the VAE is frozen during diffusion training.
    """

    def __init__(self, in_channels: int = 1, base_channels: int = 32, latent_channels: int = 4, down_factor: int = 8) -> None:
        super().__init__()
        if down_factor not in (4, 8, 16):
            raise ValueError(f"down_factor must be one of 4/8/16, got {down_factor}")
        self.down_factor = down_factor
        self.latent_channels = latent_channels
        n_down = int(math.log2(down_factor))

        # Encoder
        enc_layers: list[nn.Module] = [nn.Conv2d(in_channels, base_channels, 3, padding=1)]
        ch = base_channels
        for _ in range(n_down):
            enc_layers.append(_VAEResBlock(ch, ch * 2))
            enc_layers.append(nn.Conv2d(ch * 2, ch * 2, 4, stride=2, padding=1))
            ch *= 2
        enc_layers.append(_VAEResBlock(ch, ch))
        enc_layers.append(nn.GroupNorm(8, ch))
        enc_layers.append(nn.SiLU())
        enc_layers.append(nn.Conv2d(ch, 2 * latent_channels, 3, padding=1))
        self.encoder = nn.Sequential(*enc_layers)

        # Decoder
        dec_layers: list[nn.Module] = [nn.Conv2d(latent_channels, ch, 3, padding=1)]
        for _ in range(n_down):
            dec_layers.append(_VAEResBlock(ch, ch))
            dec_layers.append(nn.ConvTranspose2d(ch, ch // 2, 4, stride=2, padding=1))
            ch //= 2
        dec_layers.append(_VAEResBlock(ch, ch))
        dec_layers.append(nn.GroupNorm(8, ch))
        dec_layers.append(nn.SiLU())
        dec_layers.append(nn.Conv2d(ch, in_channels, 3, padding=1))
        dec_layers.append(nn.Tanh())
        self.decoder = nn.Sequential(*dec_layers)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        h = self.encoder(x)
        mean, _ = torch.chunk(h, 2, dim=1)
        return mean

    def encode_distribution(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(x)
        mean, log_var = torch.chunk(h, 2, dim=1)
        log_var = log_var.clamp(-30.0, 20.0)
        return mean, log_var

    def reparameterize(self, mean: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * log_var)
        return mean + std * torch.randn_like(mean)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encode(x)
        return z, self.decode(z)


# ---------------------------------------------------------------------------
# Component encoder
# ---------------------------------------------------------------------------


class ComponentEncoder(nn.Module):
    """Reference-glyph encoder that emits ``components_per_ref`` tokens per ref.

    Architecture: stack of strided convs to a small spatial grid, then flatten
    and project to ``d_ctx``. The "component" interpretation is purely an
    abstraction — empirically the model can learn to use these tokens
    however cross-attention prefers.
    """

    def __init__(
        self,
        *,
        in_channels: int,
        base_channels: int,
        d_ctx: int,
        components_per_ref: int = 8,
    ) -> None:
        super().__init__()
        self.components_per_ref = components_per_ref
        # Three 2× downsamples → 16×16 for 128 input. Output flattened to 256 tokens.
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 3, padding=1),
            nn.GroupNorm(8, base_channels),
            nn.SiLU(),
        )
        ch = base_channels
        blocks: list[nn.Module] = []
        for mult in (1, 2, 4):
            blocks.append(_VAEResBlock(ch, base_channels * mult))
            blocks.append(nn.Conv2d(base_channels * mult, base_channels * mult, 4, stride=2, padding=1))
            ch = base_channels * mult
        self.backbone = nn.Sequential(*blocks)
        self.proj = nn.Conv2d(ch, d_ctx, 1)
        # Adaptive pool to ``components_per_ref`` tokens (a √K × √K grid).
        # Non-perfect-square requests are rounded to the nearest perfect square
        # and warned so the actual allocated token count is auditable.
        requested = components_per_ref
        side = int(round(requested ** 0.5))
        side = max(1, side)
        actual = side * side
        if actual != requested:
            _logger.warning(
                "ComponentEncoder: components_per_ref=%d is not a perfect square; "
                "rounding to %d (side=%d) for the √K × √K adaptive pool grid.",
                requested,
                actual,
                side,
            )
        self.k_side = side
        self.components_per_ref = actual
        self.pool = nn.AdaptiveAvgPool2d(side)

    def forward(self, refs: torch.Tensor, ref_valid: torch.Tensor | None = None) -> torch.Tensor:
        """``refs``: (B, N_refs, C, H, W). Returns ``(B, N_refs * K, d_ctx)``.

        Missing refs (``ref_valid[i,j] == False``) get zeroed token slots.
        """
        b, n, c, h, w = refs.shape
        x = refs.view(b * n, c, h, w)
        x = self.stem(x)
        x = self.backbone(x)
        x = self.proj(x)
        x = self.pool(x)  # (B*N, d_ctx, k_side, k_side)
        x = x.flatten(2).transpose(1, 2)  # (B*N, K, d_ctx)
        x = x.view(b, n, self.components_per_ref, -1)
        if ref_valid is not None:
            mask = ref_valid.to(dtype=x.dtype).view(b, n, 1, 1)
            x = x * mask
        return x.reshape(b, n * self.components_per_ref, -1)


# ---------------------------------------------------------------------------
# Latent U-Net
# ---------------------------------------------------------------------------


def _sinusoidal_time_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000.0) * torch.arange(half, dtype=torch.float32, device=t.device) / max(half - 1, 1)
    )
    args = t.float().unsqueeze(-1) * freqs.unsqueeze(0)
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb


class _AdaLNZero(nn.Module):
    """AdaLN-Zero modulation (DiT-style). Linear init at zero → identity init."""

    def __init__(self, channels: int, d_ctx: int) -> None:
        super().__init__()
        self.norm = nn.GroupNorm(8, channels, affine=False)
        self.proj = nn.Linear(d_ctx, 2 * channels)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        scale, shift = torch.chunk(self.proj(cond), 2, dim=-1)
        # x: (B, C, H, W); cond → (B, C)
        x = self.norm(x)
        return x * (1.0 + scale[:, :, None, None]) + shift[:, :, None, None]


class _ResBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, d_ctx: int, dropout: float) -> None:
        super().__init__()
        self.norm1 = _AdaLNZero(in_ch, d_ctx)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.norm2 = _AdaLNZero(out_ch, d_ctx)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x, cond)))
        h = self.conv2(self.dropout(F.silu(self.norm2(h, cond))))
        return h + self.skip(x)


class _CrossAttention(nn.Module):
    def __init__(self, channels: int, d_ctx: int, n_heads: int) -> None:
        super().__init__()
        if channels % n_heads != 0:
            raise ValueError(f"channels ({channels}) must be divisible by n_heads ({n_heads})")
        self.n_heads = n_heads
        self.head_dim = channels // n_heads
        self.norm = nn.GroupNorm(8, channels)
        self.to_q = nn.Conv2d(channels, channels, 1)
        self.to_kv = nn.Linear(d_ctx, 2 * channels)
        self.out_proj = nn.Conv2d(channels, channels, 1)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, x: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        if tokens.numel() == 0:
            return x
        q = self.to_q(self.norm(x)).view(b, self.n_heads, self.head_dim, h * w)
        q = q.transpose(-1, -2)  # (B, heads, HW, head_dim)
        kv = self.to_kv(tokens).view(b, tokens.shape[1], 2, self.n_heads, self.head_dim)
        k, v = kv[:, :, 0], kv[:, :, 1]
        k = k.transpose(1, 2)  # (B, heads, T, head_dim)
        v = v.transpose(1, 2)
        attn = (q @ k.transpose(-1, -2)) / math.sqrt(self.head_dim)
        attn = attn.softmax(dim=-1)
        out = attn @ v  # (B, heads, HW, head_dim)
        out = out.transpose(-1, -2).reshape(b, c, h, w)
        return x + self.out_proj(out)


class _Downsample(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class _Upsample(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(F.interpolate(x, scale_factor=2, mode="nearest"))


class LatentUNet(nn.Module):
    """Latent-space U-Net with AdaLN-Zero + cross-attention over component tokens."""

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        in_ch = cfg.latent_channels + cfg.content_channels  # content concatenated
        base = cfg.base_channels
        d_ctx = cfg.d_ctx

        # Time embedding
        self.time_mlp = nn.Sequential(
            nn.Linear(d_ctx, d_ctx * 4),
            nn.SiLU(),
            nn.Linear(d_ctx * 4, d_ctx),
        )

        # Conditioning embeddings (char / writer / script). All optional (None
        # ⇒ uncond branch). Built as ``+1`` embeddings to host a dropout slot.
        self.char_emb = nn.Embedding(cfg.char_vocab_size + 1, d_ctx)
        self.writer_emb = nn.Embedding(cfg.writer_vocab_size + 1, d_ctx)
        self.script_emb = nn.Embedding(cfg.script_vocab_size + 1, d_ctx)
        # ``+1`` index reserved for "dropped" / uncond.
        self.char_null_idx = cfg.char_vocab_size
        self.writer_null_idx = cfg.writer_vocab_size
        self.script_null_idx = cfg.script_vocab_size

        # Input stem
        self.input_proj = nn.Conv2d(in_ch, base, 3, padding=1)

        # Down path
        ch = base
        self.downs = nn.ModuleList()
        chs_for_skip = [base]
        for mult in cfg.channel_mult:
            target_ch = base * mult
            level_blocks: list[nn.Module] = []
            for _ in range(cfg.num_res_blocks):
                level_blocks.append(_ResBlock(ch, target_ch, d_ctx, cfg.dropout))
                ch = target_ch
                chs_for_skip.append(ch)
            self.downs.append(nn.ModuleList(level_blocks))
            if mult != cfg.channel_mult[-1]:
                self.downs.append(nn.ModuleList([_Downsample(ch)]))
                chs_for_skip.append(ch)
        # Mid
        self.mid_res1 = _ResBlock(ch, ch, d_ctx, cfg.dropout)
        self.mid_attn = _CrossAttention(ch, d_ctx, cfg.n_heads)
        self.mid_res2 = _ResBlock(ch, ch, d_ctx, cfg.dropout)

        # Up path
        self.ups = nn.ModuleList()
        for mult in reversed(cfg.channel_mult):
            target_ch = base * mult
            level_blocks_up: list[nn.Module] = []
            for _ in range(cfg.num_res_blocks + 1):
                skip_ch = chs_for_skip.pop()
                level_blocks_up.append(_ResBlock(ch + skip_ch, target_ch, d_ctx, cfg.dropout))
                ch = target_ch
            self.ups.append(nn.ModuleList(level_blocks_up))
            if mult != cfg.channel_mult[0]:
                self.ups.append(nn.ModuleList([_Upsample(ch)]))

        # Output
        self.out_norm = _AdaLNZero(ch, d_ctx)
        self.out_conv = nn.Conv2d(ch, cfg.latent_channels, 3, padding=1)
        nn.init.zeros_(self.out_conv.weight)
        nn.init.zeros_(self.out_conv.bias)

    # ------------------------------------------------------------------
    def _resolve_cond_id(
        self, value: torch.Tensor | None, vocab_size: int, null_idx: int, device: torch.device, batch: int
    ) -> torch.Tensor:
        if value is None:
            return torch.full((batch,), null_idx, dtype=torch.long, device=device)
        idx = value.long().to(device).clamp_max(vocab_size - 1)
        return idx

    def _build_cond(
        self,
        timesteps: torch.Tensor,
        char_id: torch.Tensor | None,
        writer_id: torch.Tensor | None,
        script_id: torch.Tensor | None,
    ) -> torch.Tensor:
        batch = timesteps.shape[0]
        device = timesteps.device
        d = self.cfg.d_ctx
        t_emb = self.time_mlp(_sinusoidal_time_embedding(timesteps, d))
        char_idx = self._resolve_cond_id(char_id, self.cfg.char_vocab_size, self.char_null_idx, device, batch)
        writer_idx = self._resolve_cond_id(writer_id, self.cfg.writer_vocab_size, self.writer_null_idx, device, batch)
        script_idx = self._resolve_cond_id(script_id, self.cfg.script_vocab_size, self.script_null_idx, device, batch)
        return t_emb + self.char_emb(char_idx) + self.writer_emb(writer_idx) + self.script_emb(script_idx)

    def forward(
        self,
        z_t: torch.Tensor,
        timesteps: torch.Tensor,
        *,
        content: torch.Tensor,
        ref_tokens: torch.Tensor,
        char_id: torch.Tensor | None = None,
        writer_id: torch.Tensor | None = None,
        script_id: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # Down-sample content to latent spatial size.
        if content.shape[-1] != z_t.shape[-1]:
            content = F.adaptive_avg_pool2d(content, z_t.shape[-2:])
        if content.shape[1] != self.cfg.content_channels:
            # Pad / truncate to the configured content channel count.
            target = self.cfg.content_channels
            if content.shape[1] > target:
                content = content[:, :target]
            else:
                pad = torch.zeros(
                    content.shape[0], target - content.shape[1], *content.shape[2:],
                    device=content.device, dtype=content.dtype,
                )
                content = torch.cat([content, pad], dim=1)
        x = torch.cat([z_t, content], dim=1)
        cond = self._build_cond(timesteps, char_id, writer_id, script_id)

        h = self.input_proj(x)
        skips: list[torch.Tensor] = [h]
        for level in self.downs:
            for block in level:
                if isinstance(block, _ResBlock):
                    h = block(h, cond)
                    skips.append(h)
                else:
                    h = block(h)
                    skips.append(h)

        h = self.mid_res1(h, cond)
        h = self.mid_attn(h, ref_tokens)
        h = self.mid_res2(h, cond)

        for level in self.ups:
            for block in level:
                if isinstance(block, _ResBlock):
                    skip = skips.pop()
                    h = block(torch.cat([h, skip], dim=1), cond)
                else:
                    h = block(h)

        h = F.silu(self.out_norm(h, cond))
        return self.out_conv(h)


# ---------------------------------------------------------------------------
# Style-guided SR
# ---------------------------------------------------------------------------


class StyleGuidedSR(nn.Module):
    """Tiny SR upsampler with reference-token cross-attention.

    Phase-1 deliverable: build + forward + L1 loss. Phase-2 may swap to a
    fuller arch matching the (unspecified) paper module.
    """

    def __init__(self, in_channels: int, base_channels: int, d_ctx: int, n_heads: int, scale: int) -> None:
        super().__init__()
        self.scale = scale
        self.stem = nn.Conv2d(in_channels, base_channels, 3, padding=1)
        self.res = _VAEResBlock(base_channels, base_channels)
        self.attn = _CrossAttention(base_channels, d_ctx, n_heads)
        self.up = nn.Sequential(
            nn.Conv2d(base_channels, base_channels * scale * scale, 3, padding=1),
            nn.PixelShuffle(scale),
            nn.Conv2d(base_channels, in_channels, 3, padding=1),
            nn.Tanh(),
        )

    def forward(self, lr: torch.Tensor, ref_tokens: torch.Tensor) -> torch.Tensor:
        h = self.stem(lr)
        h = self.res(h)
        h = self.attn(h, ref_tokens)
        return self.up(h)


# ---------------------------------------------------------------------------
# Top-level model
# ---------------------------------------------------------------------------


class HFHFontModel(nn.Module):
    """Top-level container exposing the training contract used by ``train.py``."""

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.vae = TinyVAE(
            in_channels=cfg.in_channels,
            base_channels=max(16, cfg.base_channels // 2),
            latent_channels=cfg.latent_channels,
            down_factor=cfg.vae_down_factor,
        )
        self.component_encoder = ComponentEncoder(
            in_channels=cfg.in_channels,
            base_channels=cfg.base_channels // 2 if cfg.base_channels > 16 else cfg.base_channels,
            d_ctx=cfg.d_ctx,
            components_per_ref=cfg.components_per_ref,
        )
        self.unet = LatentUNet(cfg)
        self.sr_module: StyleGuidedSR | None = None
        if cfg.sr_enabled:
            self.sr_module = StyleGuidedSR(
                in_channels=cfg.in_channels,
                base_channels=cfg.base_channels,
                d_ctx=cfg.d_ctx,
                n_heads=cfg.n_heads,
                scale=cfg.sr_scale,
            )

    # ------------------------------------------------------------------
    # VAE helpers
    # ------------------------------------------------------------------
    def encode_image(self, image: torch.Tensor) -> torch.Tensor:
        """Encode a (B, 1, H, W) glyph into latent (B, Cz, H/f, W/f)."""
        return self.vae.encode(image)

    def decode_latent(self, z: torch.Tensor) -> torch.Tensor:
        return self.vae.decode(z)

    # ------------------------------------------------------------------
    # Forward (shared.GaussianDiffusion signature)
    # ------------------------------------------------------------------
    def forward(
        self,
        z_t: torch.Tensor,
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
        """Signature compatible with ``shared.diffusion.GaussianDiffusion._model_pred``."""
        del style_family_id, unit_id  # currently unused — paper note doesn't name them
        if ref_images is None or ref_images.numel() == 0:
            # No refs → empty token tensor handled by _CrossAttention.
            ref_tokens = torch.empty(z_t.shape[0], 0, self.cfg.d_ctx, device=z_t.device, dtype=z_t.dtype)
        else:
            ref_tokens = self.component_encoder(ref_images, ref_valid)
        return self.unet(
            z_t,
            timesteps,
            content=content,
            ref_tokens=ref_tokens,
            char_id=char_id,
            writer_id=writer_id,
            script_id=script_id,
        )

    # ------------------------------------------------------------------
    # Loss
    # ------------------------------------------------------------------
    def compute_loss(
        self,
        batch: dict[str, Any],
        diffusion,  # paper_reimpl_shared.diffusion.GaussianDiffusion
        *,
        cfg_dropout: float = 0.1,
    ) -> dict[str, torch.Tensor]:
        """Single-stage denoising loss in latent space."""
        image = batch["image"]
        content = batch["content"]
        char_id = batch.get("char_id")
        writer_id = batch.get("writer_id")
        script_id = batch.get("script_id")
        ref_images = batch.get("ref_images")
        ref_valid = batch.get("ref_valid")

        with torch.no_grad():
            z0 = self.vae.encode(image)

        diff_batch = diffusion.sample_training_batch(z0)

        # Resize content to image-size (the VAE handles its own downsampling).
        if content.shape[-1] != image.shape[-1]:
            content = F.interpolate(content, size=image.shape[-2:], mode="bilinear", align_corners=False)

        # CFG dropout: independently drop each cond channel with prob `cfg_dropout`.
        if cfg_dropout > 0.0 and self.training:
            char_id = _maybe_drop(char_id, cfg_dropout, fill=self.unet.char_null_idx, device=z0.device)
            writer_id = _maybe_drop(writer_id, cfg_dropout, fill=self.unet.writer_null_idx, device=z0.device)
            script_id = _maybe_drop(script_id, cfg_dropout, fill=self.unet.script_null_idx, device=z0.device)
            if ref_images is not None and ref_images.numel() > 0:
                ref_images, ref_valid = _maybe_drop_refs(ref_images, ref_valid, cfg_dropout)

        pred = self.forward(
            diff_batch.x_t,
            diff_batch.timesteps,
            content=content,
            char_id=char_id,
            writer_id=writer_id,
            script_id=script_id,
            ref_images=ref_images,
            ref_valid=ref_valid,
        )
        l_simple = F.mse_loss(pred, diff_batch.target)
        return {"loss": l_simple, "l_simple": l_simple.detach()}

    # ------------------------------------------------------------------
    # SDS distillation loss (Stage C, placeholder)
    # ------------------------------------------------------------------
    def compute_sds_loss(
        self,
        batch: dict[str, Any],
        teacher: HFHFontModel,
        diffusion,
    ) -> dict[str, torch.Tensor]:
        """Naive SDS proxy.

        Student predicts x0 in one step; teacher provides x0 prediction at a
        random ``t`` after re-noising the student output. Loss is MSE between
        the two. This is a *placeholder* — the full SDS gradient form is in
        ``paper_notes/02.md §4.2`` and ``reports/blind_impl.md``.
        """
        image = batch["image"]
        content = batch["content"]
        char_id = batch.get("char_id")
        writer_id = batch.get("writer_id")
        script_id = batch.get("script_id")
        ref_images = batch.get("ref_images")
        ref_valid = batch.get("ref_valid")

        with torch.no_grad():
            z0 = self.vae.encode(image)

        if content.shape[-1] != image.shape[-1]:
            content = F.interpolate(content, size=image.shape[-2:], mode="bilinear", align_corners=False)

        # Student: pretend t = T-1, one-shot prediction from pure noise. The
        # raw model output's interpretation (x0 vs ε) depends on
        # ``diffusion.prediction_target``.
        t_max = torch.full((z0.shape[0],), diffusion.timesteps - 1, dtype=torch.long, device=z0.device)
        z_T = torch.randn_like(z0)
        student_pred = self.forward(
            z_T,
            t_max,
            content=content,
            char_id=char_id,
            writer_id=writer_id,
            script_id=script_id,
            ref_images=ref_images,
            ref_valid=ref_valid,
        )
        # Convert the student output to x0 so we can re-noise it for the
        # teacher pass — this is independent of the prediction target.
        student_x0 = (
            student_pred
            if diffusion.prediction_target == "x0"
            else diffusion.predict_x0(z_T, t_max, student_pred)
        )

        # Teacher: re-noise student x0 at random t, ask teacher to denoise.
        with torch.no_grad():
            t = torch.randint(0, diffusion.timesteps, (z0.shape[0],), device=z0.device)
            noise = torch.randn_like(student_x0)
            z_t = diffusion.q_sample(student_x0.detach(), t, noise)
            teacher_pred = teacher.forward(
                z_t,
                t,
                content=content,
                char_id=char_id,
                writer_id=writer_id,
                script_id=script_id,
                ref_images=ref_images,
                ref_valid=ref_valid,
            )

        # Compare apples-to-apples in the model's native prediction space:
        #   * if target == "x0",      MSE on x0
        #   * if target == "epsilon", MSE on ε
        # We re-noise student_x0 with the same (t, noise) and recover the
        # student's ε for the epsilon branch so the student loss is what the
        # *student* would emit if asked to denoise z_t. This keeps the
        # gradient flowing into student parameters via student_x0.
        if diffusion.prediction_target == "x0":
            loss = F.mse_loss(student_x0, teacher_pred)
        else:
            # ε = (z_t - sqrt(ᾱ_t) * x0) / sqrt(1 - ᾱ_t).
            # Reconstruct the student's ε for the SAME z_t / t pair the
            # teacher saw, so the comparison is in ε-space on both sides.
            sqrt_alpha = diffusion.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)
            sqrt_one_minus_alpha = diffusion.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)
            student_eps = (z_t - sqrt_alpha * student_x0) / sqrt_one_minus_alpha
            loss = F.mse_loss(student_eps, teacher_pred)

        return {"loss": loss, "l_sds": loss.detach()}

    # ------------------------------------------------------------------
    # SR loss
    # ------------------------------------------------------------------
    def compute_sr_loss(self, lr_image: torch.Tensor, hr_image: torch.Tensor, ref_tokens: torch.Tensor) -> dict[str, torch.Tensor]:
        if self.sr_module is None:
            raise RuntimeError("SR module disabled in config")
        hr_pred = self.sr_module(lr_image, ref_tokens)
        loss = F.l1_loss(hr_pred, hr_image)
        return {"loss": loss, "l_sr": loss.detach()}


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------


def _maybe_drop(value: torch.Tensor | None, p: float, *, fill: int, device: torch.device) -> torch.Tensor | None:
    if value is None:
        return None
    drop_mask = torch.rand(value.shape[0], device=device) < p
    if not drop_mask.any():
        return value
    out = value.clone()
    out[drop_mask] = fill
    return out


def _maybe_drop_refs(refs: torch.Tensor, ref_valid: torch.Tensor | None, p: float) -> tuple[torch.Tensor, torch.Tensor | None]:
    drop_mask = torch.rand(refs.shape[0], device=refs.device) < p
    if not drop_mask.any():
        return refs, ref_valid
    refs = refs.clone()
    refs[drop_mask] = 0.0
    if ref_valid is not None:
        ref_valid = ref_valid.clone()
        ref_valid[drop_mask] = False
    return refs, ref_valid


def build_model(cfg: ModelConfig | dict[str, Any]) -> HFHFontModel:
    if isinstance(cfg, dict):
        cfg = ModelConfig.from_dict(cfg)
    return HFHFontModel(cfg)
