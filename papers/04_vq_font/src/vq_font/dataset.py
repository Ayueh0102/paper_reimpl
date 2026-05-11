"""VQ-Font dataset adapters.

VQ-Font requires three artefacts per sample:
  1. ``image``  — the target glyph (the thing whose codebook indices the
     Transformer must predict).
  2. ``refs``   — R style reference glyphs (paper uses R=3).
  3. ``structure_id`` — the Chinese-character structure class (0..13) drawn
     from the target character's IDS lookup (``parse_structure`` in
     ``scripts/lookup_ids.py``). When the manifest does not ship a structure
     id we default to 0 (the 'unknown' sentinel) so smoke-test paths stay
     valid.

Stage 0 (VQGAN pretrain) only uses ``image`` — references are ignored.
Stages 1+ consume the full triple.

The actual loaders subclass the shared ``CalligraphyJsonlDataset``; the only
paper-specific extension is the ``structure_id`` field, which we sniff out
of either ``row['structure_id']`` (preferred — pre-computed during manifest
build) or ``row['structure']`` (string label, parsed at load time).
"""

from __future__ import annotations

from typing import Any

import torch
from torch.utils.data import Dataset

from paper_reimpl_shared.data.legacy import (
    CalligraphyJsonlDataset,
    SyntheticCalligraphyDataset,
)
from paper_reimpl_shared.data.manifest import BackendPaths

from .transformer import NUM_STRUCTURE_CLASSES

__all__ = ["build_dataset", "VQFontDataset", "VQFontSyntheticDataset"]


# Stable ordering of the 12 + atomic + unknown structure classes — keep this
# in sync with `scripts/lookup_ids.py::STRUCTURE_NAMES`.
STRUCTURE_NAME_TO_ID: dict[str, int] = {
    "unknown": 0,
    "atomic": 1,
    "left_right": 2,
    "top_bottom": 3,
    "left_mid_right": 4,
    "top_mid_bottom": 5,
    "surround_full": 6,
    "surround_open_bottom": 7,
    "surround_open_top": 8,
    "surround_open_right": 9,
    "surround_open_TR": 10,
    "surround_open_TL": 11,
    "surround_open_BR": 12,
    "overlap": 13,
}
assert len(STRUCTURE_NAME_TO_ID) == NUM_STRUCTURE_CLASSES, "structure id table size mismatch"


def _structure_id_from_row(row: dict[str, Any]) -> int:
    """Best-effort extraction of the structure id from a manifest row.

    Order of preference: explicit int id ``structure_id``, string label
    ``structure`` mapped via ``STRUCTURE_NAME_TO_ID``, otherwise 0 (unknown).
    """
    if "structure_id" in row:
        try:
            sid = int(row["structure_id"])
            if 0 <= sid < NUM_STRUCTURE_CLASSES:
                return sid
        except (TypeError, ValueError):
            pass
    label = row.get("structure")
    if isinstance(label, str):
        return STRUCTURE_NAME_TO_ID.get(label, 0)
    return 0


class VQFontDataset(CalligraphyJsonlDataset):
    """Manifest-backed dataset that adds ``structure_id`` to each sample."""

    def _fetch(self, index: int) -> dict[str, Any]:
        item = super()._fetch(index)
        item["structure_id"] = _structure_id_from_row(item["metadata"])
        return item


class VQFontSyntheticDataset(SyntheticCalligraphyDataset):
    """Synthetic dataset wrapper that exposes a deterministic structure id.

    For the smoke / dry-run paths we don't have IDS data, so we cycle the
    structure id deterministically across ``NUM_STRUCTURE_CLASSES`` classes.
    """

    def __getitem__(self, index: int) -> dict[str, Any]:
        item = super().__getitem__(index)
        item["structure_id"] = index % NUM_STRUCTURE_CLASSES
        return item


def vq_font_collate(batch: list[dict[str, Any]], *, max_refs: int = 3) -> dict[str, Any]:
    """Collate that pads refs and stacks structure_id alongside the regular keys.

    This is a stand-alone collate (not the shared one) because we need to
    surface ``structure_id`` as a long tensor. We still reuse the shared
    image/ref padding logic via a delegated call.
    """
    from paper_reimpl_shared.data.legacy import collate_calligraphy_batch

    base = collate_calligraphy_batch(batch, max_refs=max_refs)
    base["structure_id"] = torch.tensor(
        [int(item.get("structure_id", 0)) for item in batch], dtype=torch.long
    )
    return base


class VQFontCollate:
    """Picklable wrapper for `vq_font_collate` (DataLoader workers)."""

    def __init__(self, max_refs: int) -> None:
        self.max_refs = max_refs

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
        return vq_font_collate(batch, max_refs=self.max_refs)


def build_dataset(
    *,
    args,
    data_cfg: dict[str, Any],
    model_cfg,
    paths: BackendPaths,
) -> Dataset:
    """Choose between synthetic and manifest-backed datasets.

    The model_cfg passed in is the merged ``VQFontConfig``-shaped dataclass
    but we only need the image size from the VQGAN config, so we accept any
    object exposing ``vqgan.image_size`` or a plain ``image_size`` attr.
    """
    if hasattr(model_cfg, "vqgan"):
        image_size = int(model_cfg.vqgan.image_size)
    else:
        image_size = int(getattr(model_cfg, "image_size", 128))

    max_refs = int(data_cfg.get("max_refs", 3))
    use_synthetic = bool(getattr(args, "synthetic", False))
    source = str(data_cfg.get("source", "manifest")).lower()
    if use_synthetic or source == "synthetic":
        return VQFontSyntheticDataset(
            length=int(data_cfg.get("synthetic_length", 16)),
            image_size=image_size,
            content_channels=int(data_cfg.get("content_channels_n", 1)),
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
    return VQFontDataset(
        manifest_path,
        image_size=image_size,
        content_channels=content_channels_list,
        max_refs=max_refs,
    )
