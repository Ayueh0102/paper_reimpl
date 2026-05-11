"""HFH-Font dataset helpers.

For Phase-1 we reuse the shared ``CalligraphyJsonlDataset`` /
``SyntheticCalligraphyDataset`` and just expose convenience builders that
respect the paper's reference-glyph requirement (``n_refs > 0``).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from paper_reimpl_shared.data.legacy import (
    CalligraphyJsonlDataset,
    SyntheticCalligraphyDataset,
    collate_calligraphy_batch,
)
from paper_reimpl_shared.data.manifest import BackendPaths, manifest_path
from paper_reimpl_shared.data.ttf_pair_dataset import TTFCrossFontPairDataset

__all__ = [
    "build_dataset",
    "build_collate",
    "ManifestNotFoundError",
]


class ManifestNotFoundError(FileNotFoundError):
    """Raised when the YAML-named manifest cannot be resolved on this backend."""


def build_dataset(
    *,
    data_cfg: dict[str, Any],
    backend: str,
    synthetic: bool,
    paths: BackendPaths,
    image_size: int,
    content_channels: list[str],
    n_refs: int,
) -> CalligraphyJsonlDataset | SyntheticCalligraphyDataset | TTFCrossFontPairDataset:
    if synthetic:
        return SyntheticCalligraphyDataset(
            length=int(data_cfg.get("synthetic_length", 16)),
            image_size=image_size,
            content_channels=len(content_channels),
            writer_vocab_size=int(data_cfg.get("synthetic_writer_vocab", 8)),
            style_family_vocab_size=int(data_cfg.get("synthetic_style_vocab", 8)),
            char_vocab_size=int(data_cfg.get("synthetic_char_vocab", 64)),
            script_vocab_size=int(data_cfg.get("synthetic_script_vocab", 4)),
            ref_count=n_refs,
            seed=int(data_cfg.get("synthetic_seed", 0xC0FFEE)),
        )

    source = str(data_cfg.get("source", "manifest")).lower()
    if source == "ttf":
        # Cross-font TTF pairs from data_snapshot/fonts_free (shared with 01).
        fonts_root_cfg = data_cfg.get("fonts_root")
        if fonts_root_cfg:
            fonts_root = Path(str(fonts_root_cfg))
        else:
            fonts_root = paths.ttf_root.parent / "fonts_free"
        cache_cfg = data_cfg.get("supported_chars_cache")
        ratio = float(data_cfg.get("font_size_ratio", 0.85))
        if cache_cfg:
            cache_path = Path(str(cache_cfg))
        else:
            cache_path = fonts_root / f".ttf_supported_chars_{image_size}px_{ratio}.json"
        # HFH expects multi-channel content (bitmap + sdf + skeleton). The
        # TTF dataset only renders bitmaps, so the result is broadcast to
        # len(content_channels) channels by the dataset's content_channels
        # arg. SDF / skeleton stand-ins won't be present — the model still
        # trains because TTF Stage A is pretraining without those signals.
        return TTFCrossFontPairDataset(
            fonts_root=fonts_root,
            font_ids=data_cfg.get("font_ids"),
            image_size=image_size,
            content_channels=len(content_channels),
            font_size_ratio=ratio,
            length=int(data_cfg.get("ttf_epoch_length", 10000)),
            ref_count=n_refs,
            seed=int(data_cfg.get("seed", 42)),
            ensure_diff_source=bool(data_cfg.get("ensure_diff_source", True)),
            cjk_start=int(data_cfg.get("cjk_start", 0x4E00)),
            cjk_end=int(data_cfg.get("cjk_end", 0x9FFF)),
            char_cache_path=cache_path,
            script_categories=data_cfg.get("script_categories"),
        )

    manifest_name = data_cfg.get("manifest")
    if not manifest_name:
        raise ValueError("data_cfg.manifest must be set when --synthetic is not used")
    try:
        manifest_full_path = manifest_path(manifest_name, backend=backend)  # type: ignore[arg-type]
    except FileNotFoundError as exc:
        raise ManifestNotFoundError(str(exc)) from exc
    return CalligraphyJsonlDataset(
        manifest_full_path,
        image_size=image_size,
        content_channels=content_channels,
        max_refs=n_refs,
    )


def build_collate(*, n_refs: int):
    """Return a callable suitable for ``DataLoader(collate_fn=...)``."""

    def _collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
        return collate_calligraphy_batch(batch, max_refs=n_refs)

    return _collate


def resolve_manifest_path(name: str, *, backend: str) -> Path:
    """Thin wrapper that re-exports ``manifest_path`` for orchestrator use."""
    return manifest_path(name, backend=backend)  # type: ignore[arg-type]
