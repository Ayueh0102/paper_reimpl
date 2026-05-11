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
) -> CalligraphyJsonlDataset | SyntheticCalligraphyDataset:
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
