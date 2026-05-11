"""VQ-Font dataset adapters.

VQ-Font requires three artefacts per sample:
  1. ``image``  — the target glyph (the thing whose codebook indices the
     Transformer must predict).
  2. ``refs``   — R style reference glyphs (paper uses R=3).
  3. ``structure_id`` — the Chinese-character structure class (0..13) drawn
     from the target character's IDS lookup (``parse_structure`` in
     ``scripts/lookup_ids.py``). When the manifest does not ship a structure
     id we fall back to ``parse_structure(get_ids(row['char']))``; only when
     IDS lookup is unavailable do we land on 0 (the 'unknown' sentinel).

Stage 0 (VQGAN pretrain) only uses ``image`` — references are ignored.
Stages 1+ consume the full triple.

The actual loaders subclass the shared ``CalligraphyJsonlDataset``; the only
paper-specific extension is the ``structure_id`` field, which we sniff out
of either ``row['structure_id']`` (preferred — pre-computed during manifest
build), ``row['structure']`` (string label), or by parsing IDS at load time
via the ``lookup_ids`` helper.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import logging
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
from torch.utils.data import Dataset

from paper_reimpl_shared.data.legacy import (
    CalligraphyJsonlDataset,
    SyntheticCalligraphyDataset,
)
from paper_reimpl_shared.data.manifest import BackendPaths

from .transformer import NUM_STRUCTURE_CLASSES

if TYPE_CHECKING:
    from .model import VQFontConfig
    from .vqgan import VQGANConfig

__all__ = ["build_dataset", "VQFontDataset", "VQFontSyntheticDataset"]

logger = logging.getLogger(__name__)


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
if len(STRUCTURE_NAME_TO_ID) != NUM_STRUCTURE_CLASSES:
    raise ValueError(
        f"structure id table size mismatch: "
        f"len(STRUCTURE_NAME_TO_ID)={len(STRUCTURE_NAME_TO_ID)} "
        f"!= NUM_STRUCTURE_CLASSES={NUM_STRUCTURE_CLASSES}"
    )


# Cached handle to (get_ids, parse_structure) — `None` once we've decided
# lookup is unavailable (so we don't re-warn for every sample).
_LOOKUP_IDS_CACHE: tuple[Any, Any] | None | bool = False  # `False` => not yet probed


def _load_lookup_ids() -> tuple[Any, Any] | None:
    """Best-effort import of `~/Char/datasets/ids/scripts/lookup_ids.py`.

    Returns ``(get_ids, parse_structure)`` callables on success, or
    ``None`` if the module is unavailable (e.g. running on a fresh PC
    without the IDS table). Logged at WARNING the first time it fails.
    """
    global _LOOKUP_IDS_CACHE
    if _LOOKUP_IDS_CACHE is not False:
        return _LOOKUP_IDS_CACHE  # type: ignore[return-value]
    # First try the regular import path (in case scripts/ is on sys.path).
    try:
        mod = importlib.import_module("lookup_ids")
        _LOOKUP_IDS_CACHE = (mod.get_ids, mod.parse_structure)
        return _LOOKUP_IDS_CACHE
    except ImportError:
        pass
    # Fallback: load directly from the well-known Char/datasets layout.
    candidate = Path.home() / "Char" / "datasets" / "ids" / "scripts" / "lookup_ids.py"
    if not candidate.exists():
        logger.warning(
            "vq_font/dataset: lookup_ids.py not importable and not at %s; "
            "structure_id fallback will default to 0 (unknown).",
            candidate,
        )
        _LOOKUP_IDS_CACHE = None
        return None
    spec = importlib.util.spec_from_file_location("lookup_ids", candidate)
    if spec is None or spec.loader is None:
        logger.warning("vq_font/dataset: failed to build import spec for %s", candidate)
        _LOOKUP_IDS_CACHE = None
        return None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["lookup_ids"] = mod
    try:
        spec.loader.exec_module(mod)
    except Exception:  # noqa: BLE001 — broad on purpose; missing TSV, etc.
        logger.warning(
            "vq_font/dataset: failed to load lookup_ids from %s; "
            "structure_id fallback will default to 0 (unknown).",
            candidate,
            exc_info=True,
        )
        _LOOKUP_IDS_CACHE = None
        return None
    _LOOKUP_IDS_CACHE = (mod.get_ids, mod.parse_structure)
    return _LOOKUP_IDS_CACHE


def _structure_id_from_row(row: dict[str, Any]) -> int:
    """Best-effort extraction of the structure id from a manifest row.

    Order of preference:
        1. explicit int id ``structure_id``;
        2. string label ``structure`` mapped via ``STRUCTURE_NAME_TO_ID``;
        3. ``parse_structure(get_ids(row['char']))`` via ``lookup_ids.py``;
        4. 0 (unknown).
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
    # Fallback: parse IDS for the target char so SSEM gradient doesn't
    # silently collapse to class 0 when manifests lack the structure field.
    char = row.get("char") or row.get("target_char")
    if isinstance(char, str) and char:
        helpers = _load_lookup_ids()
        if helpers is not None:
            get_ids, parse_structure = helpers
            try:
                ids_str = get_ids(char)
                struct_name = parse_structure(ids_str)
                return STRUCTURE_NAME_TO_ID.get(struct_name, 0)
            except Exception:  # noqa: BLE001 — lookup table issues, etc.
                logger.debug(
                    "vq_font/dataset: lookup_ids failed for char=%r", char, exc_info=True
                )
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
    args: argparse.Namespace,
    data_cfg: dict[str, Any],
    model_cfg: "VQFontConfig | VQGANConfig | Any",
    paths: BackendPaths,
) -> Dataset:
    """Choose between synthetic and manifest-backed datasets.

    ``model_cfg`` is either a ``VQFontConfig`` (Stage 1+ — exposes ``.vqgan``)
    or a bare ``VQGANConfig`` (Stage 0 — exposes ``.image_size`` directly).
    The duck-typed ``Any`` fallback exists so dry-run / smoke harnesses that
    pass simple namespaces continue to work.
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
