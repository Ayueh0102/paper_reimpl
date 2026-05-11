"""FontDiffuser dataset adapters.

This module is intentionally thin — it reuses the shared
``CalligraphyJsonlDataset`` and ``SyntheticCalligraphyDataset`` from
``paper_reimpl_shared.data.legacy``. The only paper-specific concern is that
FontDiffuser conditions on a single style reference (one-shot), so we force
``max_refs=1`` when the YAML does not set it.

Stage-A (TTF pretraining) optionally falls back to a synthetic dataset so the
smoke test and dry-run paths do not require manifests on disk.
"""

from __future__ import annotations

from typing import Any

from torch.utils.data import Dataset

from paper_reimpl_shared.data.legacy import (
    CalligraphyJsonlDataset,
    SyntheticCalligraphyDataset,
)
from paper_reimpl_shared.data.manifest import BackendPaths

__all__ = ["build_dataset", "FontDiffuserPairDataset"]


class FontDiffuserPairDataset(CalligraphyJsonlDataset):
    """Manifest-backed (source, target, style_ref) triple dataset.

    Subclasses ``CalligraphyJsonlDataset`` so we get its retry logic and ID
    book-keeping for free. We override nothing today — the shared collate
    already produces ``ref_images`` of shape ``[B, N, C, H, W]`` and the
    FontDiffuser model takes ``ref_images[:, 0]`` as the one-shot reference.

    Kept as a named class so paper-specific overrides (e.g. excluding the
    query glyph from the ref pool, or sampling negatives for SCR) can be
    layered here in Stages B/C without touching legacy.
    """

    pass


def build_dataset(
    *,
    args,
    data_cfg: dict[str, Any],
    model_cfg,
    paths: BackendPaths,
) -> Dataset:
    """Choose between synthetic and manifest-backed datasets.

    Routing rules:
      1. ``--synthetic`` CLI flag wins → return SyntheticCalligraphyDataset.
      2. ``data_cfg['source'] == 'synthetic'`` (e.g. Stage A TTF pretrain where
         we want random glyph proxies for plumbing tests) → synthetic.
      3. Otherwise, resolve the manifest file path via
         ``paths.manifest_root / data_cfg['manifest']`` and return
         FontDiffuserPairDataset.
    """
    image_size = int(model_cfg.image_size)
    content_channels = int(model_cfg.content_channels)
    max_refs = int(data_cfg.get("max_refs", 1))

    use_synthetic = bool(getattr(args, "synthetic", False))
    source = str(data_cfg.get("source", "manifest")).lower()
    if use_synthetic or source == "synthetic":
        return SyntheticCalligraphyDataset(
            length=int(data_cfg.get("synthetic_length", 16)),
            image_size=image_size,
            content_channels=content_channels,
            writer_vocab_size=int(data_cfg.get("writer_vocab_size", 4)),
            style_family_vocab_size=int(data_cfg.get("style_family_vocab_size", 8)),
            char_vocab_size=int(data_cfg.get("char_vocab_size", 64)),
            script_vocab_size=int(data_cfg.get("script_vocab_size", 4)),
            ref_count=max_refs,
            seed=int(data_cfg.get("seed", 42)),
        )

    manifest_name = data_cfg.get("manifest")
    if not manifest_name:
        raise ValueError(
            "data_cfg must contain `manifest: <file name>` (e.g. "
            "a_main_clean_smoke_a0.jsonl) when source != 'synthetic'"
        )
    manifest_path = paths.manifest_root / str(manifest_name)
    if not manifest_path.exists():
        raise FileNotFoundError(f"manifest missing: {manifest_path}")

    content_channels_list = list(data_cfg.get("content_channels", ["bitmap"]))
    return FontDiffuserPairDataset(
        manifest_path,
        image_size=image_size,
        content_channels=content_channels_list,
        max_refs=max_refs,
    )
