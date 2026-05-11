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
"""

from __future__ import annotations

from typing import Any

from torch.utils.data import Dataset

from paper_reimpl_shared.data.legacy import (
    CalligraphyJsonlDataset,
    SyntheticCalligraphyDataset,
)
from paper_reimpl_shared.data.manifest import BackendPaths

__all__ = ["build_dataset", "MoyunTripleLabelDataset"]


class MoyunTripleLabelDataset(CalligraphyJsonlDataset):
    """Manifest-backed dataset that surfaces the three TripleLabel ids.

    Identical to the shared base — we keep the class name so paper-specific
    overrides (e.g. per-script balanced sampling) can be layered later
    without touching shared/.
    """

    pass


def build_dataset(
    *,
    args,
    data_cfg: dict[str, Any],
    model_cfg,
    paths: BackendPaths,
) -> Dataset:
    """Pick between synthetic and manifest-backed dataset.

    Routing rules:
      1. ``--synthetic`` CLI flag → SyntheticCalligraphyDataset.
      2. ``data_cfg['source'] == 'synthetic'`` → SyntheticCalligraphyDataset.
      3. Otherwise → MoyunTripleLabelDataset over ``paths.manifest_root``.
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
