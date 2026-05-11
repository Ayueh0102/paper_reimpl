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
from paper_reimpl_shared.data.ttf_pair_dataset import TTFCrossFontPairDataset

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

    if source == "ttf":
        # Resolve fonts_root via the same backend prefix used for manifests.
        # ``paths.ttf_root`` already points at ``ttf_renders`` (the legacy
        # pre-render cache). ``fonts_root`` should be a sibling
        # ``fonts_free`` under the same data_snapshot. The backend mapping
        # is documented in ``shared/data/manifest.py``; we resolve it
        # relative to that root rather than hardcoding paths here.
        fonts_root_cfg = data_cfg.get("fonts_root")
        if fonts_root_cfg:
            from pathlib import Path as _P
            fonts_root = _P(str(fonts_root_cfg))
        else:
            # ttf_root.parent is the data_snapshot / mother repo data root.
            fonts_root = paths.ttf_root.parent / "fonts_free"
        cache_path = None
        cache_cfg = data_cfg.get("supported_chars_cache")
        if cache_cfg:
            from pathlib import Path as _P
            cache_path = _P(str(cache_cfg))
        return TTFCrossFontPairDataset(
            fonts_root=fonts_root,
            font_ids=data_cfg.get("font_ids"),
            image_size=image_size,
            content_channels=content_channels,
            font_size_ratio=float(data_cfg.get("font_size_ratio", 0.85)),
            length=int(data_cfg.get("ttf_epoch_length", 10000)),
            ref_count=max_refs,
            seed=int(data_cfg.get("seed", 42)),
            ensure_diff_source=bool(data_cfg.get("ensure_diff_source", True)),
            cjk_start=int(data_cfg.get("cjk_start", 0x4E00)),
            cjk_end=int(data_cfg.get("cjk_end", 0x9FFF)),
            char_cache_path=cache_path,
            script_categories=data_cfg.get("script_categories"),
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
