"""IF-Font dataset adapter.

Thin wrapper around the shared ``CalligraphyJsonlDataset`` /
``SyntheticCalligraphyDataset`` plus IDS lookup. The only paper-specific
contract is that every emitted sample carries:

  * ``image``         : [C, H, W] target glyph in [-1, 1]
  * ``refs``          : [N, C, H, W] reference glyphs (≥1 for one-shot eval)
  * ``ids_string``    : str — IDS for the target char (may be empty for synthetic)
  * ``ids_token_ids`` : [L] long — pre-tokenised (added by collate)

The IDS string is looked up via a ``Callable[[str], str]`` injected at
``build_dataset`` time. For the smoke / dry-run path we synthesise a
deterministic IDS sequence from each row's index so the tokenizer/decoder
path is still exercised without depending on the external IDS TSV.

For real Stage B/C training, point ``ids_lookup_path`` in the data YAML at
``~/Char/datasets/ids/scripts/lookup_ids.py`` (default fall-through).
"""

from __future__ import annotations

import importlib.util
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import torch
from paper_reimpl_shared.data.legacy import (
    CalligraphyJsonlDataset,
    SyntheticCalligraphyDataset,
)
from paper_reimpl_shared.data.manifest import BackendPaths
from torch.utils.data import Dataset

from .ids import DEFAULT_IDC_CHARS, IDSTokenizer

__all__ = [
    "IFFontDataset",
    "build_dataset",
    "load_ids_lookup",
    "synthetic_ids_for_index",
]


# --------------------------------------------------------------------------------------
# IDS lookup loading
# --------------------------------------------------------------------------------------


def load_ids_lookup(path: str | Path | None) -> Callable[[str], str]:
    """Load the ``get_ids`` function from ``lookup_ids.py``.

    Args:
        path: absolute path to the ``lookup_ids.py`` file. ``None`` returns a
            no-op lookup (always returns empty string).
    """
    if not path:
        return lambda _ch: ""
    p = Path(path).expanduser()
    if not p.exists():
        # Soft failure: smoke / dry-run can still run.
        return lambda _ch: ""
    spec = importlib.util.spec_from_file_location("if_font_user_lookup_ids", p)
    if spec is None or spec.loader is None:
        return lambda _ch: ""
    module = importlib.util.module_from_spec(spec)
    sys.modules["if_font_user_lookup_ids"] = module
    spec.loader.exec_module(module)
    if not hasattr(module, "get_ids"):
        return lambda _ch: ""
    return module.get_ids  # type: ignore[no-any-return]


def synthetic_ids_for_index(index: int) -> str:
    """Deterministic stub IDS used by the synthetic / smoke path.

    Cycles through the 12 IDC chars and appends 1-2 placeholder leaf chars
    so the encoder still sees both structure tokens and leaf tokens.
    """
    idc = DEFAULT_IDC_CHARS[index % len(DEFAULT_IDC_CHARS)]
    leaf1 = chr(0x4E00 + (index * 7) % 1000)  # CJK leaf in range U+4E00..U+51E8
    leaf2 = chr(0x4E00 + (index * 13) % 1000)
    return f"{idc}{leaf1}{leaf2}"


# --------------------------------------------------------------------------------------
# Datasets
# --------------------------------------------------------------------------------------


class IFFontDataset(CalligraphyJsonlDataset):
    """Manifest-backed dataset that also carries an IDS string per row.

    Subclasses the shared ``CalligraphyJsonlDataset`` and overrides
    ``_fetch`` to attach the IDS string for the row's char.
    """

    def __init__(
        self,
        manifest_path: str | Path,
        *,
        image_size: int,
        content_channels: list[str],
        max_refs: int,
        ids_lookup: Callable[[str], str],
    ) -> None:
        super().__init__(
            manifest_path,
            image_size=image_size,
            content_channels=content_channels,
            max_refs=max_refs,
        )
        self.ids_lookup = ids_lookup

    def _row_char(self, row: dict[str, Any]) -> str:
        return str(row.get("char", row.get("target_char", row.get("char_id", ""))))

    def _fetch(self, index: int) -> dict[str, Any]:
        item = super()._fetch(index)
        char = self._row_char(self.rows[index])
        ids_str = self.ids_lookup(char) if char else ""
        item["ids_string"] = ids_str
        return item


class _SyntheticIFFontDataset(SyntheticCalligraphyDataset):
    """Synthetic dataset wrapper that adds a deterministic IDS string.

    Overrides ``__getitem__`` to keep the upstream synthetic image generation
    but append ``ids_string`` from ``synthetic_ids_for_index(index)``.
    """

    def __getitem__(self, index: int) -> dict[str, Any]:
        item = super().__getitem__(index)
        item["ids_string"] = synthetic_ids_for_index(index)
        return item


# --------------------------------------------------------------------------------------
# Collate
# --------------------------------------------------------------------------------------


class IFFontCollate:
    """Picklable collate function: stacks images + batch-encodes IDS.

    Constructor receives:
      * tokenizer : IDSTokenizer — should be pre-fitted and frozen before the
        DataLoader is constructed. Lazy collate-side fitting is supported only
        for the single-process / dry-run path (``num_workers=0``); under
        ``num_workers>0`` the fit happens inside a worker subprocess and the
        mutation never reaches the main-process tokenizer. The safe default
        path is: warm-pass the dataset, ``tokenizer.fit_from_strings(...)``,
        ``tokenizer.freeze()``, *then* construct this collate.
      * max_refs  : number of reference glyphs to stack.
      * ids_max_len : decoder context length budget for IDS.

    Lazy fit semantics:
      * When ``fit_on_first_call=True`` and the tokenizer is **not** frozen,
        the first batch grows the vocab from observed IDS strings.
      * When the tokenizer is frozen (``tokenizer.is_frozen``), the lazy fit
        becomes a no-op regardless of ``fit_on_first_call`` — unknown chars
        fall back to UNK at encode time, which is the multi-worker-safe path.
    """

    def __init__(
        self,
        *,
        tokenizer: IDSTokenizer,
        max_refs: int,
        ids_max_len: int,
        fit_on_first_call: bool = True,
    ) -> None:
        self.tokenizer = tokenizer
        self.max_refs = max_refs
        self.ids_max_len = ids_max_len
        self.fit_on_first_call = fit_on_first_call
        self._fitted = False

    def _maybe_fit(self, ids_strings: Sequence[str]) -> None:
        # Gate: a frozen tokenizer is the multi-worker-safe path. Any
        # collate-side mutation in a worker subprocess would not reach the
        # main process, so we refuse to grow vocab from inside collate.
        if self.tokenizer.is_frozen:
            self._fitted = True
            return
        if self._fitted or not self.fit_on_first_call:
            return
        # Single-process / dry-run only: grow the vocab from the first batch.
        # Caller is responsible for ensuring num_workers=0 in this mode.
        self.tokenizer.fit_from_strings(ids_strings)
        self._fitted = True

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
        # Stack images
        images = torch.stack([item["image"] for item in batch])
        char_ids = torch.tensor([item.get("char_id", 0) for item in batch], dtype=torch.long)
        script_ids = torch.tensor([item.get("script_id", 0) for item in batch], dtype=torch.long)
        writer_ids = torch.tensor([item.get("writer_id", 0) for item in batch], dtype=torch.long)

        # Refs
        ref_tensors: list[torch.Tensor] = []
        ref_valid_rows: list[torch.Tensor] = []
        for item in batch:
            refs = list(item.get("ref_images", []))[: self.max_refs]
            n_real = len(refs)
            while len(refs) < self.max_refs:
                refs.append(torch.zeros_like(images[0]))
            if self.max_refs:
                ref_tensors.append(torch.stack(refs))
            else:
                ref_tensors.append(torch.empty(0, *images.shape[1:]))
            valid = torch.zeros(self.max_refs, dtype=torch.bool)
            if n_real > 0:
                valid[:n_real] = True
            ref_valid_rows.append(valid)

        if self.max_refs:
            ref_images = torch.stack(ref_tensors)
        else:
            ref_images = torch.empty(images.shape[0], 0, *images.shape[1:])
        ref_valid = torch.stack(ref_valid_rows) if self.max_refs else torch.empty(
            images.shape[0], 0, dtype=torch.bool
        )

        ids_strings = [item.get("ids_string", "") for item in batch]
        self._maybe_fit(ids_strings)
        ids_token_ids, ids_attention_mask = self.tokenizer.batch_encode(
            ids_strings, max_len=self.ids_max_len
        )

        return {
            "image": images,
            "char_id": char_ids,
            "script_id": script_ids,
            "writer_id": writer_ids,
            "refs": ref_images,
            "ref_images": ref_images,
            "ref_valid": ref_valid,
            "ids_strings": ids_strings,
            "ids_token_ids": ids_token_ids,
            "ids_attention_mask": ids_attention_mask,
            "metadata": [item.get("metadata", {}) for item in batch],
        }


# --------------------------------------------------------------------------------------
# Builder
# --------------------------------------------------------------------------------------


def build_dataset(
    *,
    args,
    data_cfg: dict[str, Any],
    model_cfg,
    paths: BackendPaths,
    ids_lookup: Callable[[str], str] | None = None,
) -> Dataset:
    """Pick between synthetic and manifest-backed datasets.

    Routing:
      1. ``--synthetic`` flag OR ``data_cfg['source'] == 'synthetic'`` →
         ``_SyntheticIFFontDataset``.
      2. Otherwise → ``IFFontDataset`` consuming a manifest file under
         ``paths.manifest_root``.
    """
    image_size = int(model_cfg.image_size)
    content_channels_n = int(getattr(model_cfg, "in_channels", 1))
    max_refs = int(data_cfg.get("max_refs", 1))

    use_synthetic = bool(getattr(args, "synthetic", False))
    source = str(data_cfg.get("source", "manifest")).lower()

    if use_synthetic or source == "synthetic":
        return _SyntheticIFFontDataset(
            length=int(data_cfg.get("synthetic_length", 16)),
            image_size=image_size,
            content_channels=content_channels_n,
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
            "data_cfg must contain `manifest: <file name>` when source != 'synthetic'."
        )
    manifest_path = paths.manifest_root / str(manifest_name)
    if not manifest_path.exists():
        raise FileNotFoundError(f"manifest missing: {manifest_path}")

    content_channels_list = list(data_cfg.get("content_channels", ["bitmap"]))
    if ids_lookup is None:
        ids_lookup = load_ids_lookup(data_cfg.get("ids_lookup_path"))

    return IFFontDataset(
        manifest_path,
        image_size=image_size,
        content_channels=content_channels_list,
        max_refs=max_refs,
        ids_lookup=ids_lookup,
    )
