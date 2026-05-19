"""IF-Font dataset adapter — Phase 2 (RGB, IDS resolver, coverage similarity).

Changes from Phase 1:
  * Images are converted to RGB (3-channel) by replicating the grayscale
    channel — the frozen CompVis VQGAN expects RGB.
  * IDS source switched from CHISE → BabelStone + ids_iffont via
    ``IDSResolver``. Lookup callable returns the **resolved tuple** of leaf
    tokens (radical mode).
  * Each batch carries:
      - target_ids_tuples: list[tuple[str, ...]] — resolved IDS per target.
      - ref_ids_tuples:    list[list[tuple[str, ...]]] — resolved IDS per ref.
      - coverage_sim:      [B, N] float tensor — computed via
        ``IFFont.compute_coverage``.
      - font_id:           [B] long — used by `losses.sup_cl`.
  * The collate emits 3-channel images regardless of the underlying dataset's
    native channel count.
"""

from __future__ import annotations

import importlib.util
import random
import sys
import warnings
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

from .ids import DEFAULT_IDC_CHARS, IDSResolver, IDSTokenizer

__all__ = [
    "IFFontCollate",
    "IFFontDataset",
    "build_dataset",
    "load_ids_lookup",
    "synthetic_ids_for_index",
]


# --------------------------------------------------------------------------------------
# IDS lookup loading
# --------------------------------------------------------------------------------------


def load_ids_lookup(path: str | Path | None) -> Callable[[str], str]:
    """Load a `get_ids(char) -> str` function from a user-provided script.

    Kept for back-compat with the Phase-1 CHISE pathway; production runs use
    ``IDSResolver`` directly via ``IFFontDataset.ids_resolver``.
    """
    if not path:
        return lambda _ch: ""
    p = Path(path).expanduser()
    if not p.exists():
        return lambda _ch: ""
    if p.suffix.lower() in {".tsv", ".txt"}:
        mapping: dict[str, str] = {}
        with p.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split("\t")
                if len(parts) < 2:
                    parts = line.split(maxsplit=1)
                if len(parts) >= 2:
                    mapping[parts[0]] = parts[1]
        return lambda ch: mapping.get(ch, "")
    spec = importlib.util.spec_from_file_location("if_font_user_lookup_ids", p)
    if spec is None or spec.loader is None:
        return lambda _ch: ""
    module = importlib.util.module_from_spec(spec)
    sys.modules["if_font_user_lookup_ids"] = module
    try:
        spec.loader.exec_module(module)
    except Exception as exc:
        warnings.warn(f"failed to load IDS lookup script {p}: {exc}", stacklevel=2)
        return lambda _ch: ""
    if not hasattr(module, "get_ids"):
        return lambda _ch: ""
    return module.get_ids  # type: ignore[no-any-return]


def synthetic_ids_for_index(index: int) -> str:
    """Deterministic stub IDS used by the synthetic / smoke path."""
    idc = DEFAULT_IDC_CHARS[index % len(DEFAULT_IDC_CHARS)]
    leaf1 = chr(0x4E00 + (index * 7) % 1000)
    leaf2 = chr(0x4E00 + (index * 13) % 1000)
    return f"{idc}{leaf1}{leaf2}"


def _to_rgb(image: torch.Tensor, target_channels: int) -> torch.Tensor:
    """Convert [C, H, W] tensor to ``target_channels``.

    1→3: replicate channel. C→1: average. Same C: no-op. Otherwise raises.
    """
    c = image.shape[0]
    if c == target_channels:
        return image
    if c == 1 and target_channels == 3:
        return image.expand(3, *image.shape[1:]).contiguous()
    if c == 3 and target_channels == 1:
        return image.mean(dim=0, keepdim=True)
    raise ValueError(f"cannot convert image with {c} channels to {target_channels} channels")


# --------------------------------------------------------------------------------------
# Datasets
# --------------------------------------------------------------------------------------


class IFFontDataset(CalligraphyJsonlDataset):
    """Manifest-backed dataset with IDS resolver hookup."""

    def __init__(
        self,
        manifest_path: str | Path,
        *,
        image_size: int,
        content_channels: list[str],
        max_refs: int,
        ids_resolver: IDSResolver | None,
        ids_lookup: Callable[[str], str] | None = None,
    ) -> None:
        super().__init__(
            manifest_path,
            image_size=image_size,
            content_channels=content_channels,
            max_refs=max_refs,
        )
        self.ids_resolver = ids_resolver
        self.ids_lookup = ids_lookup or (lambda _ch: "")

    def _row_char(self, row: dict[str, Any]) -> str:
        return str(row.get("char", row.get("target_char", row.get("char_id", ""))))

    def _resolved(self, ch: str) -> tuple[str, ...]:
        if not ch:
            return ()
        if self.ids_resolver is not None:
            try:
                return self.ids_resolver.resolve(ch)
            except (KeyError, RecursionError):
                pass
        # Fallback: split the raw lookup string into characters.
        return tuple(self.ids_lookup(ch))

    def _fetch(self, index: int) -> dict[str, Any]:
        item = super()._fetch(index)
        ch = self._row_char(self.rows[index])
        resolved = self._resolved(ch)
        item["ids_string"] = "".join(resolved)
        item["ids_tokens"] = resolved
        item["target_char"] = ch
        # Resolve refs too (using the row's metadata if present; fall back to
        # synthetic placeholder).
        refs_meta = item.get("refs_meta", [])
        ref_chars = [str(r.get("char", "")) for r in refs_meta][: self.max_refs]
        item["ref_chars"] = ref_chars
        item["ref_ids_tokens"] = [self._resolved(c) for c in ref_chars]
        return item


class _SyntheticIFFontDataset(SyntheticCalligraphyDataset):
    """Synthetic dataset wrapper that adds a deterministic IDS sequence."""

    def __getitem__(self, index: int) -> dict[str, Any]:
        item = super().__getitem__(index)
        ids_str = synthetic_ids_for_index(index)
        item["ids_string"] = ids_str
        item["ids_tokens"] = tuple(ids_str)
        item["target_char"] = chr(0x4E00 + (index % 4096))
        n_refs = len(item.get("ref_images", []))
        item["ref_chars"] = [chr(0x4E00 + ((index + j + 1) % 4096)) for j in range(n_refs)]
        item["ref_ids_tokens"] = [
            tuple(synthetic_ids_for_index(index + j + 1)) for j in range(n_refs)
        ]
        return item


# --------------------------------------------------------------------------------------
# Collate
# --------------------------------------------------------------------------------------


def _compute_coverage_row(
    target: Sequence[str], refs: Sequence[Sequence[str]], idc_set: set[str]
) -> list[float]:
    """Inline copy of the coverage computation (avoid importing IFFont here)."""
    out: list[float] = []
    ti_max = len(target)
    if ti_max == 0:
        return [0.0] * len(refs)
    for source in refs:
        match_cnt = 0
        ti = 0
        while ti < ti_max:
            tc = target[ti]
            if tc not in idc_set:
                ti += 1
                continue
            si, si_max = 0, len(source)
            advanced = False
            while si < si_max:
                if source[si] != tc:
                    si += 1
                    continue
                ti2, si2 = ti, si
                while ti2 < ti_max and si2 < si_max and target[ti2] == source[si2]:
                    ti2 += 1
                    si2 += 1
                if ti2 == ti_max or target[ti2] in idc_set:
                    match_cnt += ti2 - ti
                    ti = ti2
                    advanced = True
                    break
                si += 1
            if not advanced:
                ti += 1
        out.append(match_cnt / ti_max)
    return out


class IFFontCollate:
    """Phase-2 collate: stacks RGB images, batches IDS, computes coverage_sim.

    Each emitted batch dict carries:
      * image:        [B, in_channels, H, W]
      * refs / ref_images: [B, N, in_channels, H, W]
      * ref_valid:    [B, N] bool
      * ids_token_ids: [B, ids_max_len] long  (PAD-right)
      * ids_attention_mask: [B, ids_max_len] bool
      * coverage_sim: [B, N] float — IDS-coverage scores per (target, ref).
      * font_id:      [B] long — alias for writer_id (used by sup_cl labels).
      * writer_id, script_id, char_id: passthrough convenience labels.
    """

    def __init__(
        self,
        *,
        tokenizer: IDSTokenizer,
        max_refs: int,
        ids_max_len: int,
        in_channels: int = 3,
        fit_on_first_call: bool = True,
    ) -> None:
        self.tokenizer = tokenizer
        self.max_refs = max_refs
        self.ids_max_len = ids_max_len
        self.in_channels = in_channels
        self.fit_on_first_call = fit_on_first_call
        self._fitted = False
        self._idc_set = set(DEFAULT_IDC_CHARS)

    def _maybe_fit(self, ids_strings: Sequence[str]) -> None:
        if self.tokenizer.is_frozen:
            self._fitted = True
            return
        if self._fitted or not self.fit_on_first_call:
            return
        self.tokenizer.fit_from_strings(ids_strings)
        self._fitted = True

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
        # ---- Images (force in_channels) ----
        images = torch.stack([_to_rgb(item["image"], self.in_channels) for item in batch])
        char_ids = torch.tensor([item.get("char_id", 0) for item in batch], dtype=torch.long)
        script_ids = torch.tensor([item.get("script_id", 0) for item in batch], dtype=torch.long)
        writer_ids = torch.tensor([item.get("writer_id", 0) for item in batch], dtype=torch.long)

        # ---- Refs ----
        ref_tensors: list[torch.Tensor] = []
        ref_valid_rows: list[torch.Tensor] = []
        ref_token_rows: list[list[tuple[str, ...]]] = []
        for item in batch:
            refs = [_to_rgb(r, self.in_channels) for r in item.get("ref_images", [])][
                : self.max_refs
            ]
            ref_toks = list(item.get("ref_ids_tokens", []))[: self.max_refs]
            n_real = len(refs)
            zero_img = torch.zeros(self.in_channels, *images.shape[-2:])
            while len(refs) < self.max_refs:
                refs.append(zero_img)
            while len(ref_toks) < self.max_refs:
                ref_toks.append(())
            if self.max_refs:
                ref_tensors.append(torch.stack(refs))
            else:
                ref_tensors.append(torch.empty(0, self.in_channels, *images.shape[-2:]))
            valid = torch.zeros(self.max_refs, dtype=torch.bool)
            if n_real > 0:
                valid[:n_real] = True
            ref_valid_rows.append(valid)
            ref_token_rows.append(ref_toks)

        if self.max_refs:
            ref_images = torch.stack(ref_tensors)
        else:
            ref_images = torch.empty(images.shape[0], 0, self.in_channels, *images.shape[-2:])
        ref_valid = (
            torch.stack(ref_valid_rows)
            if self.max_refs
            else torch.empty(images.shape[0], 0, dtype=torch.bool)
        )

        # ---- IDS ----
        ids_strings = [item.get("ids_string", "") for item in batch]
        target_token_rows: list[tuple[str, ...]] = [
            item.get("ids_tokens", tuple(item.get("ids_string", ""))) for item in batch
        ]
        self._maybe_fit(ids_strings)
        ids_token_ids, ids_attention_mask = self.tokenizer.batch_encode(
            target_token_rows, max_len=self.ids_max_len
        )

        # ---- Coverage similarity ----
        coverage_rows = [
            _compute_coverage_row(t, refs, self._idc_set)
            for t, refs in zip(target_token_rows, ref_token_rows, strict=False)
        ]
        if self.max_refs:
            coverage_sim = torch.tensor(coverage_rows, dtype=torch.float32)
        else:
            coverage_sim = torch.empty(images.shape[0], 0, dtype=torch.float32)

        return {
            "image": images,
            "char_id": char_ids,
            "script_id": script_ids,
            "writer_id": writer_ids,
            "font_id": writer_ids,  # alias for sup_cl labels
            "refs": ref_images,
            "ref_images": ref_images,
            "ref_valid": ref_valid,
            "ids_strings": ids_strings,
            "ids_token_ids": ids_token_ids,
            "ids_attention_mask": ids_attention_mask,
            "coverage_sim": coverage_sim,
            "target_ids_tokens": target_token_rows,
            "ref_ids_tokens": ref_token_rows,
            "metadata": [item.get("metadata", {}) for item in batch],
        }


# --------------------------------------------------------------------------------------
# Builder
# --------------------------------------------------------------------------------------


class _IFFontTTFAdapter(Dataset):
    """Wrap TTFCrossFontPairDataset and inject IF-Font's IDS keys per item.
    The shared collate (IFFontCollate) tokenizes ids_tokens to ids_token_ids
    and computes coverage_sim from ref_chars vs target_char.

    For TTF Stage A there are no real refs from a manifest, so we use the
    TTF-emitted ref glyphs but their underlying chars are all the same as
    the target (ensure_diff_source only enforces different *font*, not
    different *char*). Coverage sim is therefore mostly 1.0 — fine for
    pretrain plumbing.
    """

    def __init__(self, *, inner, ids_resolver, ids_lookup, max_refs: int) -> None:
        super().__init__()
        self.inner = inner
        self.ids_resolver = ids_resolver
        self.ids_lookup = ids_lookup or (lambda _ch: "")
        self.max_refs = int(max_refs)

    def __len__(self) -> int:
        return len(self.inner)

    def _resolved(self, ch: str) -> tuple[str, ...]:
        if not ch:
            return ()
        if self.ids_resolver is not None:
            try:
                return self.ids_resolver.resolve(ch)
            except (KeyError, RecursionError):
                pass
        return tuple(self.ids_lookup(ch))

    def __getitem__(self, idx: int) -> dict[str, Any]:
        s = self.inner[idx]
        meta = s.get("metadata", {})
        ch = str(meta.get("char", ""))
        resolved = self._resolved(ch)
        s = dict(s)
        s["ids_string"] = "".join(resolved)
        s["ids_tokens"] = resolved
        s["target_char"] = ch
        # Refs in TTF Stage A use the SAME target char (different font),
        # so their IDS resolves identically.
        ref_chars = [ch] * self.max_refs
        s["ref_chars"] = ref_chars
        s["ref_ids_tokens"] = [resolved for _ in ref_chars]
        return s


def build_dataset(
    *,
    args,
    data_cfg: dict[str, Any],
    model_cfg,
    paths: BackendPaths,
    ids_lookup: Callable[[str], str] | None = None,
    ids_resolver: IDSResolver | None = None,
) -> Dataset:
    image_size = int(model_cfg.image_size)
    content_channels_n = int(getattr(model_cfg, "in_channels", 3))
    max_refs = int(data_cfg.get("max_refs", 3))

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
            content_channels=content_channels_n,
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
        if ids_lookup is None:
            ids_lookup = load_ids_lookup(data_cfg.get("ids_lookup_path"))
        return _IFFontTTFAdapter(
            inner=inner,
            ids_resolver=ids_resolver,
            ids_lookup=ids_lookup,
            max_refs=max_refs,
        )

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
        ids_resolver=ids_resolver,
        ids_lookup=ids_lookup,
    )


# Silence unused-import warning when random is not used directly.
_ = random
