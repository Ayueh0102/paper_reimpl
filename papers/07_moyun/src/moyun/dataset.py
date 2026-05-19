"""Moyun dataset adapters.

Moyun is **id-conditioned**, not image-conditioned. So the dataset only needs
to emit:

  * ``image``        — the target glyph image (or VAE latent), shape (C, H, W)
  * ``writer_id``    — calligrapher id (int)
  * ``script_id``    — font / script class id (int)
  * ``char_id``      — character id (int)

No ``content`` and no ``refs`` are required. We still emit them (zero tensors)
so the shared collate works unchanged and so the smoke test path stays
identical to the other papers.

We subclass ``paper_reimpl_shared.data.legacy.CalligraphyJsonlDataset`` to
inherit manifest parsing + id book-keeping for free. For synthetic / dry-run
we fall back to ``SyntheticCalligraphyDataset`` (also from shared).

The ``source=ttf`` branch is the latent-diffusion bridge: TTF pixels are
rendered on the fly via ``TTFCrossFontPairDataset`` and then encoded through
the 02_hfh_font ``TinyVAE`` to land in the 4ch 32×32 latent space the Moyun
model is built for.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

from paper_reimpl_shared.data.legacy import (
    CalligraphyJsonlDataset,
    SyntheticCalligraphyDataset,
)
from paper_reimpl_shared.data.manifest import BackendPaths

__all__ = ["build_dataset", "MoyunTripleLabelDataset", "MoyunTTFLatentDataset"]


class MoyunTripleLabelDataset(CalligraphyJsonlDataset):
    """Manifest-backed dataset that surfaces the three TripleLabel ids.

    Identical to the shared base — we keep the class name so paper-specific
    overrides (e.g. per-script balanced sampling) can be layered later
    without touching shared/.
    """

    pass


# ---------------------------------------------------------------------------
# TTF -> VAE latent bridge
# ---------------------------------------------------------------------------


def _import_tiny_vae():
    """Import 02_hfh_font's ``TinyVAE`` without making 02 a hard dep.

    We never import at module load (that would force the 07 venv to satisfy
    02's dep graph). Instead we add 02's ``src/`` directory to ``sys.path``
    on first request and import lazily.
    """
    here = Path(__file__).resolve()
    # papers/07_moyun/src/moyun/dataset.py -> papers/
    papers_root = here.parents[3]
    hfh_src = papers_root / "02_hfh_font" / "src"
    if str(hfh_src) not in sys.path:
        sys.path.insert(0, str(hfh_src))
    from hfh_font.model import TinyVAE  # type: ignore[import-not-found]

    return TinyVAE


class MoyunTTFLatentDataset(Dataset):
    """Render TTF glyphs on the fly + encode via 02_hfh_font's TinyVAE.

    The Moyun model expects latent-space inputs (4ch 32×32 by default), so a
    pixel-space TTF renderer is not sufficient on its own. We wrap
    ``TTFCrossFontPairDataset`` for the pixel side and run each glyph through
    the VAE encoder per ``__getitem__``. The VAE is loaded **once** at
    construction time, in eval mode, with ``requires_grad_(False)`` — Moyun
    treats the VAE as frozen (paper §3.5).

    If the VAE checkpoint is missing the loader prints a WARNING and falls
    back to a fresh-init TinyVAE. This is deliberate: we want
    ``--dry-run`` smokes to construct cleanly even before the 02 Stage-VAE
    has been trained. **Do not** run real training in this mode — the
    "latents" produced by a random VAE encode are garbage and the model will
    learn nothing meaningful.
    """

    def __init__(
        self,
        *,
        fonts_root: Path,
        image_size: int,
        vae_ckpt_path: Path | None,
        latent_resolution: int,
        latent_channels: int,
        vae_base_channels: int,
        vae_down_factor: int,
        font_ids: list[str] | None,
        font_size_ratio: float,
        length: int,
        n_refs: int,
        seed: int,
        cjk_start: int,
        cjk_end: int,
        char_cache_path: Path | None,
        script_categories: dict[str, str] | None,
        ensure_diff_source: bool,
        device: torch.device | str = "cpu",
        writer_vocab_cap: int | None = None,
        script_vocab_cap: int | None = None,
        char_vocab_cap: int | None = None,
    ) -> None:
        super().__init__()
        # Import the inner dataset lazily so the module remains importable
        # in stripped-down environments (the shared dataset only needs
        # PIL + numpy at import time, which are already in 07's venv, but
        # the explicit local import keeps the dependency graph readable).
        from paper_reimpl_shared.data.ttf_pair_dataset import TTFCrossFontPairDataset

        self.image_size = int(image_size)
        self.latent_resolution = int(latent_resolution)
        self.latent_channels = int(latent_channels)
        self.n_refs = int(n_refs)
        self.device = torch.device(device)
        # Vocab caps: id values that fall outside the model's embedding tables
        # cause an IndexError. We clamp into ``[0, cap)`` in __getitem__ so the
        # batch is always safe. ``None`` means "no clamp".
        self.writer_vocab_cap = writer_vocab_cap
        self.script_vocab_cap = script_vocab_cap
        self.char_vocab_cap = char_vocab_cap

        if self.image_size % self.latent_resolution != 0:
            raise ValueError(
                f"image_size={self.image_size} must be a multiple of "
                f"latent_resolution={self.latent_resolution} for the VAE's "
                f"power-of-2 downsampling."
            )
        expected_down = self.image_size // self.latent_resolution
        if expected_down != vae_down_factor:
            print(
                f"[moyun-ttf] WARNING: image_size/latent_resolution = "
                f"{expected_down} but vae_down_factor={vae_down_factor}. "
                f"Using {expected_down} from the geometry — adjust the yaml "
                f"to silence this."
            )
            vae_down_factor = expected_down

        self.inner = TTFCrossFontPairDataset(
            fonts_root=fonts_root,
            font_ids=font_ids,
            image_size=self.image_size,
            content_channels=1,
            font_size_ratio=font_size_ratio,
            length=length,
            ref_count=max(0, self.n_refs),
            seed=seed,
            ensure_diff_source=ensure_diff_source,
            cjk_start=cjk_start,
            cjk_end=cjk_end,
            char_cache_path=char_cache_path,
            script_categories=script_categories,
        )

        # Load TinyVAE (lazy import to avoid an unconditional 02 dep).
        TinyVAE = _import_tiny_vae()
        self.vae = TinyVAE(
            in_channels=1,
            base_channels=int(vae_base_channels),
            latent_channels=self.latent_channels,
            down_factor=int(vae_down_factor),
        )

        ckpt_loaded = False
        if vae_ckpt_path is not None and Path(vae_ckpt_path).exists():
            try:
                blob = torch.load(str(vae_ckpt_path), map_location="cpu", weights_only=False)
                state = blob.get("model", blob) if isinstance(blob, dict) else blob
                # ``pretrain_vae.py`` saves keys prefixed with ``vae.`` so the
                # blob can drop straight into the full HFH model. Strip the
                # prefix so it loads cleanly into a bare TinyVAE.
                stripped = {
                    (k[len("vae."):] if k.startswith("vae.") else k): v
                    for k, v in state.items()
                }
                missing, unexpected = self.vae.load_state_dict(stripped, strict=False)
                print(
                    f"[moyun-ttf] loaded VAE from {vae_ckpt_path} "
                    f"(missing={len(missing)} unexpected={len(unexpected)})"
                )
                ckpt_loaded = True
            except Exception as exc:  # pragma: no cover — fallback path
                print(
                    f"[moyun-ttf] WARNING: failed to load VAE ckpt at "
                    f"{vae_ckpt_path}: {exc!r}. Falling back to fresh-init."
                )
        else:
            where = vae_ckpt_path if vae_ckpt_path is not None else "<none>"
            print(
                f"[moyun-ttf] WARNING: VAE ckpt not found at {where}. "
                f"Using FRESH-INIT TinyVAE — fine for --dry-run plumbing, "
                f"BUT REAL TRAINING NEEDS A PRETRAINED VAE."
            )

        self._ckpt_loaded = ckpt_loaded
        self.vae.eval()
        for p in self.vae.parameters():
            p.requires_grad_(False)
        self.vae = self.vae.to(self.device)

    def __len__(self) -> int:
        return len(self.inner)

    @torch.no_grad()
    def _encode(self, image: torch.Tensor) -> torch.Tensor:
        # image: (1, H, W) in [-1, 1]; encode wants (B, 1, H, W).
        z = self.vae.encode(image.unsqueeze(0).to(self.device))
        return z.squeeze(0).cpu()

    @staticmethod
    def _clamp(value: int, cap: int | None) -> int:
        if cap is None or cap <= 0:
            return int(value)
        return int(value) % int(cap)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        item = self.inner[idx]
        # Replace the pixel ``image`` with its VAE latent. Moyun's forward
        # consumes the ``image`` key as the diffusion target (latent).
        latent = self._encode(item["image"])
        # Content / refs are ignored by Moyun but the shared collate still
        # expects ``content`` to stack. Emit a zero tensor of the latent shape
        # so the batch dict has uniform spatial geometry.
        zero_content = torch.zeros_like(latent[:1])  # (1, h_lat, w_lat)
        # ref_images are ignored too. Keep them as an empty list so the
        # collate's max_refs=0 path stays a no-op.
        writer_id = self._clamp(item["writer_id"], self.writer_vocab_cap)
        script_id = self._clamp(item["script_id"], self.script_vocab_cap)
        char_id = self._clamp(item["char_id"], self.char_vocab_cap)
        out = {
            "image": latent,
            "content": zero_content,
            "char_id": char_id,
            "script_id": script_id,
            "writer_id": writer_id,
            "style_family_id": writer_id,
            "unit_id": writer_id,
            "ref_images": [],
            "metadata": {
                **item.get("metadata", {}),
                "ttf_latent": True,
                "vae_ckpt_loaded": self._ckpt_loaded,
            },
        }
        return out


def _resolve_fonts_root(
    data_cfg: dict[str, Any],
    paths: BackendPaths,
) -> Path:
    """Resolve ``data_cfg.fonts_root`` against BackendPaths / PR_DATA_ROOT.

    Mirrors the pattern from 06_calliffusion so the same yaml ships on Mac
    (PR_DATA_ROOT=..../data), lab server, and vast.
    """
    cfg_value = data_cfg.get("fonts_root")
    if cfg_value:
        candidate = Path(str(cfg_value)).expanduser()
        if candidate.is_absolute():
            return candidate
        data_root = os.environ.get("PR_DATA_ROOT")
        if data_root:
            return Path(data_root).expanduser() / candidate
        if paths is not None:
            return Path(paths.ttf_root).parent / candidate
        raise ValueError(
            "relative data_cfg.fonts_root requires BackendPaths or PR_DATA_ROOT."
        )
    if paths is not None and Path(paths.ttf_root).parent.exists():
        return Path(paths.ttf_root).parent / "fonts_free"
    data_root = os.environ.get("PR_DATA_ROOT")
    if not data_root:
        raise ValueError(
            "07 source=ttf requires BackendPaths or PR_DATA_ROOT to resolve fonts_free."
        )
    return Path(data_root).expanduser() / "fonts_free"


def _resolve_vae_ckpt(data_cfg: dict[str, Any]) -> Path | None:
    """Resolve ``data_cfg.vae_ckpt_path`` against the 07_moyun paper root.

    ``None`` is allowed — the dataset will fall back to fresh-init with a
    loud WARNING (suitable for dry-run plumbing checks).
    """
    cfg_value = data_cfg.get("vae_ckpt_path")
    if not cfg_value:
        return None
    candidate = Path(str(cfg_value)).expanduser()
    if candidate.is_absolute():
        return candidate
    # 07_moyun paper root: papers/07_moyun/src/moyun/dataset.py -> papers/07_moyun
    paper_root = Path(__file__).resolve().parents[2]
    return paper_root / candidate


def build_dataset(
    *,
    args: argparse.Namespace,
    data_cfg: dict[str, Any],
    model_cfg: Any,
    paths: BackendPaths,
) -> Dataset:
    """Pick between synthetic, TTF latent, and manifest-backed dataset.

    Routing rules:
      1. ``--synthetic`` CLI flag → SyntheticCalligraphyDataset.
      2. ``data_cfg['source'] == 'synthetic'`` → SyntheticCalligraphyDataset.
      3. ``data_cfg['source'] == 'ttf'`` → MoyunTTFLatentDataset (VAE-encoded).
      4. Otherwise → MoyunTripleLabelDataset over ``paths.manifest_root``.
    """
    image_size = int(model_cfg.image_size)
    in_channels = int(model_cfg.in_channels)

    use_synthetic = bool(getattr(args, "synthetic", False))
    source = str(data_cfg.get("source", "manifest")).lower()

    if use_synthetic or source == "synthetic":
        return SyntheticCalligraphyDataset(
            length=int(data_cfg.get("synthetic_length", 16)),
            image_size=image_size,
            # Moyun ignores ``content`` but the shared dataset always emits it;
            # we keep content_channels small to limit memory.
            content_channels=int(data_cfg.get("content_channels", 1)),
            writer_vocab_size=int(data_cfg.get("writer_vocab_size", 4)),
            style_family_vocab_size=int(data_cfg.get("style_family_vocab_size", 4)),
            char_vocab_size=int(data_cfg.get("char_vocab_size", 64)),
            script_vocab_size=int(data_cfg.get("script_vocab_size", 5)),
            ref_count=0,
            seed=int(data_cfg.get("seed", 42)),
        )

    if source == "ttf":
        # Pixel render side: TTFCrossFontPairDataset renders at some pixel
        # resolution. The VAE downsamples by ``vae_down_factor`` so the
        # latent side = render / down. We require that latent side == the
        # model.image_size (the Moyun backbone expects 32x32 latents).
        render_size = int(data_cfg.get("render_image_size", 128))
        latent_channels = int(data_cfg.get("latent_channels", in_channels))
        fonts_root = _resolve_fonts_root(data_cfg, paths)
        vae_ckpt_path = _resolve_vae_ckpt(data_cfg)

        ratio = float(data_cfg.get("font_size_ratio", 0.85))
        cache_cfg = data_cfg.get("supported_chars_cache")
        if cache_cfg:
            cache_path = Path(str(cache_cfg)).expanduser()
            if not cache_path.is_absolute():
                cache_path = fonts_root.parent / cache_path
        else:
            cache_path = fonts_root / f".ttf_supported_chars_{render_size}px_{ratio}.json"

        return MoyunTTFLatentDataset(
            fonts_root=fonts_root,
            image_size=render_size,
            vae_ckpt_path=vae_ckpt_path,
            latent_resolution=image_size,
            latent_channels=latent_channels,
            vae_base_channels=int(data_cfg.get("vae_base_channels", 32)),
            vae_down_factor=int(data_cfg.get("vae_down_factor", max(1, render_size // image_size))),
            font_ids=data_cfg.get("font_ids"),
            font_size_ratio=ratio,
            length=int(data_cfg.get("ttf_epoch_length", 256)),
            n_refs=0,
            seed=int(data_cfg.get("seed", 42)),
            cjk_start=int(data_cfg.get("cjk_start", 0x4E00)),
            cjk_end=int(data_cfg.get("cjk_end", 0x9FFF)),
            char_cache_path=cache_path,
            script_categories=data_cfg.get("script_categories"),
            ensure_diff_source=bool(data_cfg.get("ensure_diff_source", True)),
            device=str(data_cfg.get("vae_device", "cpu")),
            writer_vocab_cap=int(getattr(model_cfg, "writer_vocab", 0)) or None,
            script_vocab_cap=int(getattr(model_cfg, "script_vocab", 0)) or None,
            char_vocab_cap=int(getattr(model_cfg, "char_vocab", 0)) or None,
        )

    manifest_name = data_cfg.get("manifest")
    if not manifest_name:
        raise ValueError(
            "data_cfg must contain `manifest: <file name>` when source != 'synthetic'"
        )
    manifest_path = paths.manifest_root / str(manifest_name)
    if not manifest_path.exists():
        raise FileNotFoundError(f"manifest missing: {manifest_path}")

    content_channels_list = list(data_cfg.get("content_channels", ["bitmap"]))
    # Moyun doesn't use content but we keep it loaded so the shared collate
    # has a uniform dict shape across papers.
    _ = in_channels  # silence unused-arg complaint; image_size is what matters
    return MoyunTripleLabelDataset(
        manifest_path,
        image_size=image_size,
        content_channels=content_channels_list,
        max_refs=0,
    )
