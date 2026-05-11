"""DP-Font dataset adapters.

Reuses the shared ``CalligraphyJsonlDataset`` / ``SyntheticCalligraphyDataset``
and adds two DP-Font-specific fields per sample:

  * ``stroke_order``: long tensor [stroke_seq_len] padded with -1.
  * ``ink_intensity`` / ``font_size``: float scalar in [0, 1].

These fields are NOT present in the Ernantang manifests today — the paper
demands them as multi-attribute guidance but provides no GT for our subset.
For the Phase 1 dry-run we **synthesise them deterministically** from the
row metadata (e.g. hash the char + writer id to pick stroke types). Stage
B/C will swap these in for real entries once we plumb the stroke-order
external DB; see ``reports/blind_impl.md`` decision #4.
"""

from __future__ import annotations

import argparse
import hashlib
from typing import Any

import torch
from torch.utils.data import Dataset

from paper_reimpl_shared.data.legacy import (
    CalligraphyJsonlDataset,
    SyntheticCalligraphyDataset,
    collate_calligraphy_batch,
)
from paper_reimpl_shared.data.manifest import BackendPaths


__all__ = [
    "build_dataset",
    "DPFontDataset",
    "DPFontSyntheticDataset",
    "collate_dp_font_batch",
    "synthesise_stroke_order",
    "synthesise_scalar_attribute",
]


# ---------------------------------------------------------------------------
# Stroke-order / scalar synthesis (placeholder until real DB plumbed)
# ---------------------------------------------------------------------------


def synthesise_stroke_order(
    *,
    seed_text: str,
    vocab_size: int,
    seq_len: int,
    min_len: int = 1,
) -> list[int]:
    """Deterministically synthesise a stroke-order sequence.

    Hashes ``seed_text`` (typically char + writer) to a stable random
    sequence in ``[0, vocab_size)``. Length is also drawn from the hash so
    different chars get different lengths. The first ``len`` entries are
    real stroke ids; remaining slots are -1 (padding).

    This is a *placeholder*. Stage B / C must replace it with a lookup into
    a real stroke-order DB (cjklib / Make-Me-a-Hanzi / 國家教育研究院筆順
    DB). The model interface is identical so swapping is mechanical.
    """
    if min_len < 0:
        raise ValueError(f"min_len must be non-negative, got {min_len}")
    if seq_len <= 0:
        raise ValueError(f"seq_len must be positive, got {seq_len}")
    if min_len > seq_len:
        raise ValueError(
            f"min_len={min_len} > seq_len={seq_len}: cannot fit synthesised "
            "sequence into the padded slot. Caller must either raise seq_len "
            "or clamp min_len."
        )
    digest = hashlib.sha256(seed_text.encode("utf-8")).digest()
    # First byte → length in [min_len, seq_len]
    length = min_len + int(digest[0]) % (seq_len - min_len + 1)
    out = [-1] * seq_len
    for i in range(length):
        out[i] = int(digest[(i + 1) % len(digest)]) % max(1, vocab_size)
    return out


def synthesise_scalar_attribute(seed_text: str, *, salt: str) -> float:
    """Deterministic scalar in [0, 1] from a seed text + salt."""
    h = hashlib.sha256(f"{salt}::{seed_text}".encode("utf-8")).digest()
    return int.from_bytes(h[:4], "big") / float(0xFFFFFFFF)


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------


class DPFontDataset(CalligraphyJsonlDataset):
    """Manifest-backed dataset that emits DP-Font's extra fields."""

    def __init__(
        self,
        manifest_path,
        *,
        image_size: int,
        content_channels: list[str],
        stroke_vocab_size: int,
        stroke_seq_len: int,
        max_refs: int = 0,
    ) -> None:
        super().__init__(
            manifest_path,
            image_size=image_size,
            content_channels=content_channels,
            max_refs=max_refs,
        )
        self.stroke_vocab_size = stroke_vocab_size
        self.stroke_seq_len = stroke_seq_len

    def __getitem__(self, index: int) -> dict[str, Any]:
        item = super().__getitem__(index)
        row = item.get("metadata", {})
        seed_text = f"{row.get('char', row.get('target_char', index))}::{row.get('writer', '')}"
        item["stroke_order"] = synthesise_stroke_order(
            seed_text=seed_text,
            vocab_size=self.stroke_vocab_size,
            seq_len=self.stroke_seq_len,
        )
        item["ink_intensity"] = synthesise_scalar_attribute(seed_text, salt="ink")
        item["font_size"] = synthesise_scalar_attribute(seed_text, salt="size")
        return item


class DPFontSyntheticDataset(SyntheticCalligraphyDataset):
    """Synthetic dataset variant that injects stroke-order + scalar fields."""

    def __init__(
        self,
        *,
        length: int,
        image_size: int,
        content_channels: int,
        writer_vocab_size: int,
        style_family_vocab_size: int,
        char_vocab_size: int,
        script_vocab_size: int,
        stroke_vocab_size: int,
        stroke_seq_len: int,
        ref_count: int = 0,
        seed: int = 42,
    ) -> None:
        super().__init__(
            length=length,
            image_size=image_size,
            content_channels=content_channels,
            writer_vocab_size=writer_vocab_size,
            style_family_vocab_size=style_family_vocab_size,
            char_vocab_size=char_vocab_size,
            script_vocab_size=script_vocab_size,
            ref_count=ref_count,
            seed=seed,
        )
        self.stroke_vocab_size = stroke_vocab_size
        self.stroke_seq_len = stroke_seq_len

    def __getitem__(self, index: int) -> dict[str, Any]:
        item = super().__getitem__(index)
        seed_text = f"synthetic::{index}"
        item["stroke_order"] = synthesise_stroke_order(
            seed_text=seed_text,
            vocab_size=self.stroke_vocab_size,
            seq_len=self.stroke_seq_len,
        )
        item["ink_intensity"] = synthesise_scalar_attribute(seed_text, salt="ink")
        item["font_size"] = synthesise_scalar_attribute(seed_text, salt="size")
        return item


# ---------------------------------------------------------------------------
# Collation
# ---------------------------------------------------------------------------


def collate_dp_font_batch(batch: list[dict[str, Any]], *, max_refs: int = 0) -> dict[str, Any]:
    """Like the shared collate but stacks DP-Font extras."""
    out = collate_calligraphy_batch(batch, max_refs=max_refs)
    stroke_order = torch.tensor(
        [list(item["stroke_order"]) for item in batch],
        dtype=torch.long,
    )
    ink = torch.tensor(
        [float(item["ink_intensity"]) for item in batch],
        dtype=torch.float32,
    )
    size = torch.tensor(
        [float(item["font_size"]) for item in batch],
        dtype=torch.float32,
    )
    out["stroke_order"] = stroke_order
    out["ink_intensity"] = ink
    out["font_size"] = size
    return out


class _DPFontCollate:
    """Picklable wrapper around ``collate_dp_font_batch``."""

    def __init__(self, max_refs: int) -> None:
        self.max_refs = max_refs

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
        return collate_dp_font_batch(batch, max_refs=self.max_refs)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


class _DPFontTTFAdapter(Dataset):
    """Wrap TTFCrossFontPairDataset and inject synthesised stroke_order,
    ink_intensity, font_size — DP-Font's collate requires these keys.

    Real values come from the Ernantang manifest in Stage B/C. For the TTF
    pretrain stage we use the same deterministic hash-based synthesis the
    SyntheticDataset uses, so each (char, font) pair gets a stable label.
    """

    def __init__(self, *, inner, stroke_vocab_size: int, stroke_seq_len: int) -> None:
        super().__init__()
        self.inner = inner
        self.stroke_vocab_size = int(stroke_vocab_size)
        self.stroke_seq_len = int(stroke_seq_len)

    def __len__(self) -> int:
        return len(self.inner)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        s = self.inner[idx]
        meta = s.get("metadata", {})
        char = meta.get("char", "")
        target_font = meta.get("target_font", "")
        seed_text = f"{char}::{target_font}::{idx}"
        stroke_ids = synthesise_stroke_order(
            seed_text=seed_text,
            vocab_size=self.stroke_vocab_size,
            seq_len=self.stroke_seq_len,
        )
        s = dict(s)  # shallow copy
        s["stroke_order"] = stroke_ids
        s["ink_intensity"] = float(synthesise_scalar_attribute(seed_text=seed_text, salt="ink"))
        s["font_size"] = float(synthesise_scalar_attribute(seed_text=seed_text, salt="size"))
        return s


def build_dataset(
    *,
    args: argparse.Namespace,
    data_cfg: dict[str, Any],
    model_cfg: Any,
    paths: BackendPaths,
) -> Dataset:
    """Choose between synthetic and manifest-backed datasets.

    Honors:
      * ``--synthetic`` CLI flag,
      * ``data_cfg.source == 'synthetic'``,
      * else loads manifest via ``paths.manifest_root / data_cfg.manifest``.
    """
    image_size = int(model_cfg.image_size)
    content_channels = int(model_cfg.content_channels)
    stroke_vocab_size = int(model_cfg.stroke_vocab_size)
    stroke_seq_len = int(model_cfg.stroke_seq_len)
    max_refs = int(data_cfg.get("max_refs", 0))

    use_synthetic = bool(getattr(args, "synthetic", False))
    source = str(data_cfg.get("source", "manifest")).lower()
    if source == "ttf" and not use_synthetic:
        from pathlib import Path as _P
        from paper_reimpl_shared.data.ttf_pair_dataset import TTFCrossFontPairDataset

        fonts_root_cfg = data_cfg.get("fonts_root")
        if fonts_root_cfg:
            fonts_root = _P(str(fonts_root_cfg))
        else:
            fonts_root = paths.ttf_root.parent / "fonts_free"
        cache_cfg = data_cfg.get("supported_chars_cache")
        ratio = float(data_cfg.get("font_size_ratio", 0.85))
        cache_path = _P(str(cache_cfg)) if cache_cfg else (
            fonts_root / f".ttf_supported_chars_{image_size}px_{ratio}.json"
        )
        inner = TTFCrossFontPairDataset(
            fonts_root=fonts_root,
            font_ids=data_cfg.get("font_ids"),
            image_size=image_size,
            content_channels=content_channels,
            font_size_ratio=ratio,
            length=int(data_cfg.get("ttf_epoch_length", 10000)),
            ref_count=max_refs,
            seed=int(data_cfg.get("seed", 42)),
            ensure_diff_source=bool(data_cfg.get("ensure_diff_source", True)),
            cjk_start=int(data_cfg.get("cjk_start", 0x4E00)),
            cjk_end=int(data_cfg.get("cjk_end", 0x9FFF)),
            char_cache_path=cache_path,
            script_categories=data_cfg.get("script_categories"),
        )
        # DP-Font's collate requires stroke_order / ink_intensity /
        # font_size on every item. Real Ernantang annotation for these is
        # a Stage B/C deliverable; for the TTF Stage A pretrain we
        # synthesise them deterministically from char+font (same helper
        # used by DPFontSyntheticDataset).
        return _DPFontTTFAdapter(
            inner=inner,
            stroke_vocab_size=stroke_vocab_size,
            stroke_seq_len=stroke_seq_len,
        )

    if use_synthetic or source == "synthetic":
        return DPFontSyntheticDataset(
            length=int(data_cfg.get("synthetic_length", 32)),
            image_size=image_size,
            content_channels=content_channels,
            writer_vocab_size=int(data_cfg.get("writer_vocab_size", 4)),
            style_family_vocab_size=int(data_cfg.get("style_family_vocab_size", 8)),
            char_vocab_size=int(data_cfg.get("char_vocab_size", 64)),
            script_vocab_size=int(data_cfg.get("script_vocab_size", 4)),
            stroke_vocab_size=stroke_vocab_size,
            stroke_seq_len=stroke_seq_len,
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
    return DPFontDataset(
        manifest_path,
        image_size=image_size,
        content_channels=content_channels_list,
        stroke_vocab_size=stroke_vocab_size,
        stroke_seq_len=stroke_seq_len,
        max_refs=max_refs,
    )
