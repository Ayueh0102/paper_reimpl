"""Moyun — blind reimplementation.

Architecture overview (from paper notes):
  * Latent diffusion backbone: input is a VAE-encoded latent (32x32x4 in the
    paper). For this blind reimpl we let the input be either a real VAE latent
    OR the raw pixel target — the model is agnostic to that choice and the
    Stage A YAML controls it. (When run with raw pixels we just treat the
    grayscale 1-channel image as the diffusion target; this lets us smoke-test
    without a pretrained VAE.)
  * **Vision Mamba (Mamba2)** replaces the U-Net (paper §3.3). Sequence of
    Mamba blocks operating on a patchified latent.
  * **TripleLabel** conditioning (paper §3.4): three INDEPENDENT learnable
    embedding tables, one each for:
      - calligrapher_id  (`writer_id`)
      - font / script id (楷/行/草/隸/篆 + ...)
      - character id     (Unicode)
    These are SUMMED ``e_total = e_writer + e_script + e_char``, then passed
    through ``MLP -> SiLU -> Linear -> chunk into (scale, shift)`` for each
    block (DiT-style adaLN-Zero modulation).
  * The diffusion timestep also goes through a sin/cos embedding + MLP and
    is ADDED to the e_total embedding before the per-block projection.

Interface contract with the shared diffusion utility
----------------------------------------------------
``paper_reimpl_shared.diffusion.gaussian.GaussianDiffusion`` calls models as::

    model(x_t, t, *, content, char_id, script_id, writer_id, style_family_id,
            unit_id, ref_images, ref_valid)

Moyun ignores ``content`` / ``ref_images`` / ``ref_valid`` / ``style_family_id``
/ ``unit_id`` — Moyun is **id-conditioned**, not image-conditioned. The
TripleLabel inputs we actually consume are::

    writer_id  -> calligrapher embedding
    script_id  -> font / script embedding
    char_id    -> character embedding

Each is either a ``LongTensor`` of shape ``(B,)`` OR ``None`` (CFG uncond
branch, in which case the embedding is replaced by a learnable [NULL]
embedding row at index 0).

Math primer for the SSM block lives in ``mamba_block.py``. Read that first.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn

from .mamba_block import VisionMambaBlock

__all__ = ["MoyunConfig", "Moyun", "build_moyun"]


# --------------------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------------------


@dataclass
class MoyunConfig:
    """All structural hyperparameters for the Moyun blind reimpl.

    Defaults are sized for a 24 GB single-GPU run at Ernantang scale. The
    paper-cited "patch=8 hidden=512 N=4" values are encoded as
    (``patch_size``, ``hidden_dim``, ``num_blocks``); we override for the CPU
    smoke test to keep the scan short.
    """

    # Image / latent geometry.
    image_size: int = 32
    """Side length of the (latent) input fed to the Mamba backbone. Paper uses
    32 for the VAE latent (4-ch); we use 32 for our smoke-test direct-pixel
    mode too, just with in_channels=1."""
    in_channels: int = 4
    """4 for VAE latent (paper). 1 for direct grayscale pixel mode (smoke /
    Stage A without a pretrained VAE)."""

    # Mamba backbone.
    patch_size: int = 8
    """Patchify stride. Paper §3.3 sets patch=8 over a 32x32 latent, giving a
    sequence of 16 tokens. For a 32x32 input that means token count L=16."""
    hidden_dim: int = 512  # paper §3.3
    num_blocks: int = 4  # paper §3.3 ("N=4")
    d_state: int = 16
    """SSM hidden state size per channel. Mamba1 default; paper does not
    specify. [guessed-because-paper-vague]."""
    d_conv: int = 3
    """Depthwise conv1d kernel for the SSM's input mixer. Mamba1 default;
    paper does not specify."""
    mlp_ratio: float = 4.0
    """FFN expansion ratio inside each VisionMamba block. DiT default."""
    bidirectional: bool = True
    """Whether each Mamba block runs forward + reversed scan and averages.
    Vision Mamba paper (Zhu et al. 2024) uses bidirectional; Moyun does not
    explicitly say, but unidirectional on a 2-D patch sequence is well known
    to underperform. [guessed-because-paper-vague]."""

    # TripleLabel vocab sizes (target = 二南堂 / Ernantang fine-tuning scale).
    writer_vocab: int = 24
    """Number of calligrapher IDs. Ernantang has 24."""
    script_vocab: int = 5
    """Number of font/script classes (paper enumerates 6 scripts: 楷/行/草/
    隸/篆/小篆; Ernantang has 4-5 — we use 5 as default with one [UNK])."""
    char_vocab: int = 4659
    """Number of distinct characters. Ernantang ≈ 4659."""

    # Conditioning MLP.
    cond_mlp_dim: Optional[int] = None
    """Hidden dim of the conditioning MLP. None -> use hidden_dim."""

    # Time embedding.
    time_embed_dim: Optional[int] = None
    """Width of the sin/cos timestep embedding before the MLP. None -> use
    hidden_dim."""

    # Misc.
    null_id_index: int = 0
    """Reserved index 0 inside each embedding table for the CFG uncond [NULL]
    token. So the *usable* id range is [1, vocab_size); during smoke we shift
    indices by +1 inside ``Moyun.forward``."""

    extra: dict = field(default_factory=dict)


# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------


def _sinusoidal_time_embed(timesteps: torch.Tensor, dim: int) -> torch.Tensor:
    """Sin/cos timestep embedding (Vaswani 2017 / Ho 2020 form)."""
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000.0)
        * torch.arange(0, half, dtype=torch.float32, device=timesteps.device)
        / max(1, half)
    )
    args = timesteps.float().unsqueeze(-1) * freqs.unsqueeze(0)  # (B, half)
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)  # (B, dim_even)
    if dim % 2 == 1:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
    return emb


def _resolve_id(ids: Optional[torch.Tensor], batch: int, device: torch.device) -> torch.Tensor:
    """Map ``ids`` to the embedding table input.

    ``None`` -> all rows point to ``null_id_index`` (CFG uncond branch).
    Tensor -> shifted by +1 so the [NULL] row (index 0) is always reserved.
    """
    if ids is None:
        return torch.zeros(batch, dtype=torch.long, device=device)
    return ids.long().to(device) + 1


# --------------------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------------------


class TripleLabelEmbedding(nn.Module):
    """Three INDEPENDENT trainable embeddings; summed to produce e_total.

    Critical correctness property (DL rubric, Moyun row): the three tables
    MUST be separate ``nn.Embedding`` modules with their own learnable
    weights. Sharing weights here would break the paper's promise of
    independent attribute control (paper §3.4: "three independent embeddings
    summed").

    We reserve index 0 of each table as the [NULL] token used by classifier-
    free guidance training. Real ids start at 1. ``Moyun.forward`` shifts
    user-provided ids by +1 before calling here.
    """

    def __init__(
        self,
        *,
        writer_vocab: int,
        script_vocab: int,
        char_vocab: int,
        hidden_dim: int,
    ) -> None:
        super().__init__()
        # +1 vocab slot each for [NULL] at index 0.
        self.writer = nn.Embedding(writer_vocab + 1, hidden_dim)
        self.script = nn.Embedding(script_vocab + 1, hidden_dim)
        self.char = nn.Embedding(char_vocab + 1, hidden_dim)
        # Zero-init the [NULL] rows so the CFG uncond branch starts as a
        # clean "no condition" signal. Other rows get default (N(0, 1)) init.
        with torch.no_grad():
            self.writer.weight[0].zero_()
            self.script.weight[0].zero_()
            self.char.weight[0].zero_()

    def forward(
        self,
        writer_ids: torch.Tensor,
        script_ids: torch.Tensor,
        char_ids: torch.Tensor,
    ) -> torch.Tensor:
        e_writer = self.writer(writer_ids)
        e_script = self.script(script_ids)
        e_char = self.char(char_ids)
        return e_writer + e_script + e_char


class PatchEmbed(nn.Module):
    """Convolutional patchify, mirroring ViT/DiT.

    Input: ``(B, C, H, W)`` -> Output: ``(B, L, hidden_dim)`` where
    ``L = (H/patch_size) * (W/patch_size)``.
    """

    def __init__(self, in_channels: int, hidden_dim: int, patch_size: int) -> None:
        super().__init__()
        self.proj = nn.Conv2d(in_channels, hidden_dim, kernel_size=patch_size, stride=patch_size)
        self.patch_size = patch_size

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int]]:
        if x.shape[-1] % self.patch_size != 0 or x.shape[-2] % self.patch_size != 0:
            raise ValueError(
                f"Input HxW={x.shape[-2]}x{x.shape[-1]} not divisible by patch_size={self.patch_size}"
            )
        h = self.proj(x)  # (B, hidden_dim, H/p, W/p)
        _, _, hh, ww = h.shape
        return h.flatten(2).transpose(1, 2), (hh, ww)  # (B, L, hidden_dim), grid


class PatchUnembed(nn.Module):
    """Inverse of PatchEmbed via transposed conv."""

    def __init__(self, out_channels: int, hidden_dim: int, patch_size: int) -> None:
        super().__init__()
        self.proj = nn.ConvTranspose2d(hidden_dim, out_channels, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor, grid_hw: tuple[int, int]) -> torch.Tensor:
        b, seq_len, c = x.shape
        hh, ww = grid_hw
        assert seq_len == hh * ww, f"L={seq_len} doesn't match grid {hh}x{ww}"
        h = x.transpose(1, 2).reshape(b, c, hh, ww)
        return self.proj(h)  # (B, out_channels, H, W)


class Moyun(nn.Module):
    """Vision Mamba + TripleLabel diffusion network.

    Forward signature matches ``GaussianDiffusion._model_pred`` so the same
    sampler can drive any of the 8 papers. Unused kwargs (content / ref_*) are
    accepted and silently dropped.
    """

    def __init__(self, cfg: MoyunConfig) -> None:
        super().__init__()
        self.cfg = cfg
        H = cfg.hidden_dim
        self.patch_embed = PatchEmbed(cfg.in_channels, H, cfg.patch_size)
        self.patch_unembed = PatchUnembed(cfg.in_channels, H, cfg.patch_size)

        # Learnable positional embedding over the token grid. Paper does not
        # specify; DiT uses 2-D sin/cos. We pick learnable for simplicity.
        max_tokens = max(1, (cfg.image_size // cfg.patch_size) ** 2)
        self.pos_embed = nn.Parameter(torch.zeros(1, max_tokens, H))
        nn.init.normal_(self.pos_embed, std=0.02)

        # Time embedding -> hidden_dim MLP.
        t_in = cfg.time_embed_dim or H
        self.time_in_dim = t_in
        self.time_mlp = nn.Sequential(
            nn.Linear(t_in, H),
            nn.SiLU(),
            nn.Linear(H, H),
        )

        # TripleLabel embedding.
        self.triple_label = TripleLabelEmbedding(
            writer_vocab=cfg.writer_vocab,
            script_vocab=cfg.script_vocab,
            char_vocab=cfg.char_vocab,
            hidden_dim=H,
        )

        # Per-block modulation MLP: e_total + t_emb -> 4 * H * num_blocks
        # (scale_ssm, shift_ssm, scale_ffn, shift_ffn) per block. Stacked into
        # one big linear so we get one matmul per forward.
        cond_dim = cfg.cond_mlp_dim or H
        self.cond_mlp = nn.Sequential(
            nn.Linear(H, cond_dim),
            nn.SiLU(),
            nn.Linear(cond_dim, 4 * H * cfg.num_blocks),
        )
        # Zero-init the last linear so each block starts as identity
        # (adaLN-Zero, DiT paper §3.2). Critical for stable training.
        with torch.no_grad():
            self.cond_mlp[-1].weight.zero_()
            if self.cond_mlp[-1].bias is not None:
                self.cond_mlp[-1].bias.zero_()

        # Vision Mamba stack.
        self.blocks = nn.ModuleList(
            [
                VisionMambaBlock(
                    H,
                    d_state=cfg.d_state,
                    d_conv=cfg.d_conv,
                    mlp_ratio=cfg.mlp_ratio,
                    bidirectional=cfg.bidirectional,
                )
                for _ in range(cfg.num_blocks)
            ]
        )

        # Final LayerNorm before patch-unembed.
        self.final_norm = nn.LayerNorm(H)

    def _modulation_for(self, e_total: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        """Combine TripleLabel embedding + time embedding -> per-block adaLN."""
        cond = e_total + t_emb  # (B, H)
        return self.cond_mlp(cond)  # (B, 4 * H * num_blocks)

    def forward(
        self,
        x_t: torch.Tensor,
        timesteps: torch.Tensor,
        *,
        # ---- TripleLabel inputs (the three things we actually consume) ----
        char_id: Optional[torch.Tensor] = None,
        script_id: Optional[torch.Tensor] = None,
        writer_id: Optional[torch.Tensor] = None,
        # ---- Shared-contract kwargs we IGNORE (kept for sampler compat) ----
        content: Optional[torch.Tensor] = None,
        style_family_id: Optional[torch.Tensor] = None,
        unit_id: Optional[torch.Tensor] = None,
        ref_images: Optional[torch.Tensor] = None,
        ref_valid: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        b, c, h, w = x_t.shape
        device = x_t.device

        # 1) Patchify.
        tokens, grid_hw = self.patch_embed(x_t)
        seq_len = tokens.shape[1]
        if self.pos_embed.shape[1] < seq_len:
            raise ValueError(
                f"pos_embed has {self.pos_embed.shape[1]} slots but got {seq_len} tokens; "
                f"increase image_size in config."
            )
        tokens = tokens + self.pos_embed[:, :seq_len]

        # 2) Conditioning: TripleLabel + time embedding -> adaLN modulation.
        writer_in = _resolve_id(writer_id, b, device)
        script_in = _resolve_id(script_id, b, device)
        char_in = _resolve_id(char_id, b, device)
        e_total = self.triple_label(writer_in, script_in, char_in)  # (B, H)
        t_emb_in = _sinusoidal_time_embed(timesteps, self.time_in_dim)
        t_emb = self.time_mlp(t_emb_in)  # (B, H)
        modulation = self._modulation_for(e_total, t_emb)  # (B, 4 * H * num_blocks)
        per_block = modulation.chunk(self.cfg.num_blocks, dim=-1)

        # 3) Mamba blocks.
        h_tok = tokens
        for block, mod in zip(self.blocks, per_block):
            h_tok = block(h_tok, mod)

        # 4) Final norm + patch unembed.
        h_tok = self.final_norm(h_tok)
        return self.patch_unembed(h_tok, grid_hw)


def build_moyun(cfg: MoyunConfig) -> Moyun:
    """Convenience constructor — mirrors ``build_fontdiffuser`` for symmetry."""
    return Moyun(cfg)
