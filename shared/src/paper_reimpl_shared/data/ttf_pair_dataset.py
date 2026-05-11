"""On-the-fly TTF cross-font pair dataset for Stage A pretraining.

Renders cross-font (target, source, ref) triples directly from `.ttf` files
in ``data_snapshot/fonts_free/``. No pre-render step required.

Handles missing chars + tofu (.notdef) glyphs by rendering a guaranteed-missing
control char per font, hashing the bitmap, and excluding any candidate char
whose render hashes to the same value.

Yields FontDiffuser-style batches:
    image      : [1, H, W] target glyph (in [-1, 1])
    content    : [C, H, W] source glyph (broadcast to content_channels)
    ref_images : list of [1, H, W] reference glyphs
    char_id, writer_id, style_family_id, script_id : int label proxies
    metadata   : {'char', 'target_font', 'source_font', 'ref_fonts', ...}
"""

from __future__ import annotations

import functools
import hashlib
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from torch.utils.data import Dataset

__all__ = ["TTFCrossFontPairDataset", "render_glyph", "discover_supported_chars"]


# CJK Unified Ideographs Basic Block — covers the vast majority of Chinese
# characters used in practice (U+4E00..U+9FFF = 20992 codepoints).
CJK_BASIC_START = 0x4E00
CJK_BASIC_END = 0x9FFF  # inclusive
# Probe char used to fingerprint each font's .notdef glyph. U+0001 is a
# control char that no Chinese font has a real glyph for, so its render is
# always the tofu/notdef bitmap.
_TOFU_PROBE_CHAR = ""


def _ttf_path_for(fonts_root: Path, font_id: str) -> Path:
    """Find the .ttf file inside ``fonts_root/<font_id>/``."""
    sub = fonts_root / font_id
    if not sub.is_dir():
        raise FileNotFoundError(f"font subdir missing: {sub}")
    candidates = sorted(sub.glob("*.ttf")) + sorted(sub.glob("*.otf"))
    if not candidates:
        raise FileNotFoundError(f"no .ttf/.otf inside {sub}")
    return candidates[0]


@functools.lru_cache(maxsize=256)
def _cached_font(ttf_path_str: str, pt: int) -> ImageFont.FreeTypeFont:
    """Cache parsed TTF fonts so render_glyph doesn't re-parse the whole
    file on every call. Discovery alone calls this 13 fonts x 20k chars ~=
    270k times; without the cache, on Windows the TTF reload dominates
    runtime (~minutes vs. seconds with the cache)."""
    return ImageFont.truetype(ttf_path_str, size=pt)


def render_glyph(
    *,
    ttf_path: Path,
    char: str,
    image_size: int = 128,
    font_size_ratio: float = 0.85,
) -> np.ndarray:
    """Render a single glyph centered in a (image_size, image_size) canvas.

    Returns a uint8 array in [0, 255], white background (255), black ink (0).
    """
    pt = max(8, int(image_size * font_size_ratio))
    font = _cached_font(str(ttf_path), pt)
    img = Image.new("L", (image_size, image_size), color=255)
    draw = ImageDraw.Draw(img)
    # textbbox gives the inked bounding box for the glyph at origin (0, 0).
    # We center by computing the glyph's own bbox and offsetting accordingly.
    bbox = draw.textbbox((0, 0), char, font=font)
    glyph_w = bbox[2] - bbox[0]
    glyph_h = bbox[3] - bbox[1]
    if glyph_w <= 0 or glyph_h <= 0:
        # Empty / missing; render at origin so caller's tofu check catches it.
        draw.text((0, 0), char, fill=0, font=font)
    else:
        x = (image_size - glyph_w) // 2 - bbox[0]
        y = (image_size - glyph_h) // 2 - bbox[1]
        draw.text((x, y), char, fill=0, font=font)
    return np.asarray(img, dtype=np.uint8)


def _tofu_signature(
    *, ttf_path: Path, image_size: int, font_size_ratio: float
) -> bytes:
    """Stable hash of a font's .notdef bitmap, used to detect tofu chars."""
    arr = render_glyph(
        ttf_path=ttf_path,
        char=_TOFU_PROBE_CHAR,
        image_size=image_size,
        font_size_ratio=font_size_ratio,
    )
    return hashlib.sha1(arr.tobytes()).digest()


def discover_supported_chars(
    *,
    fonts_root: Path,
    font_ids: list[str],
    image_size: int = 128,
    font_size_ratio: float = 0.85,
    cjk_start: int = CJK_BASIC_START,
    cjk_end: int = CJK_BASIC_END,
    cache_path: Path | None = None,
) -> list[str]:
    """Return the intersection of CJK-Basic chars rendered by all given fonts
    *without* falling through to the .notdef tofu bitmap.

    Caches the result to ``cache_path`` as JSON so repeated dataset
    constructions are sub-second.
    """
    if cache_path is not None and cache_path.exists():
        with cache_path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        if (
            payload.get("font_ids") == sorted(font_ids)
            and payload.get("image_size") == image_size
            and payload.get("font_size_ratio") == font_size_ratio
            and payload.get("cjk_start") == cjk_start
            and payload.get("cjk_end") == cjk_end
        ):
            return list(payload["chars"])

    intersection: set[str] | None = None
    for font_id in font_ids:
        ttf_path = _ttf_path_for(fonts_root, font_id)
        tofu = _tofu_signature(
            ttf_path=ttf_path,
            image_size=image_size,
            font_size_ratio=font_size_ratio,
        )
        # Also reject the all-white empty render (some PIL versions render
        # missing glyphs as fully blank canvas).
        empty_arr = np.full((image_size, image_size), 255, dtype=np.uint8)
        empty_sig = hashlib.sha1(empty_arr.tobytes()).digest()

        supported: set[str] = set()
        for cp in range(cjk_start, cjk_end + 1):
            ch = chr(cp)
            arr = render_glyph(
                ttf_path=ttf_path,
                char=ch,
                image_size=image_size,
                font_size_ratio=font_size_ratio,
            )
            sig = hashlib.sha1(arr.tobytes()).digest()
            if sig != tofu and sig != empty_sig:
                supported.add(ch)
        intersection = supported if intersection is None else (intersection & supported)

    final = sorted(intersection or set())
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with cache_path.open("w", encoding="utf-8") as f:
            json.dump(
                {
                    "font_ids": sorted(font_ids),
                    "image_size": image_size,
                    "font_size_ratio": font_size_ratio,
                    "cjk_start": cjk_start,
                    "cjk_end": cjk_end,
                    "n_chars": len(final),
                    "chars": final,
                },
                f,
                ensure_ascii=False,
                indent=0,
            )
    return final


def _array_to_tensor(arr: np.ndarray) -> torch.Tensor:
    """uint8 [H, W] (white=255, ink=0) -> float [1, H, W] in [-1, 1] (ink=+1).

    The convention matches `paper_reimpl_shared.data.ttf_renders.load_glyph`:
    inked pixels are negative end of the range. We flip to ``2 * (1 - x/255) - 1``
    so that black ink => +1, white background => -1. Wait — that contradicts
    the existing convention. The existing ``load_glyph`` does
    ``arr / 127.5 - 1`` which maps:
      255 (white bg)  -> +1
      0   (black ink) -> -1
    So inked pixels are NEGATIVE. Match that convention here for consistency
    across the codebase.
    """
    f = arr.astype(np.float32) / 127.5 - 1.0
    return torch.from_numpy(f).unsqueeze(0)


class TTFCrossFontPairDataset(Dataset):
    """Cross-font triple dataset for Stage A TTF pretraining.

    Sampling protocol per ``__getitem__(idx)``:
        1. Seeded RNG from (base_seed, idx) for reproducibility.
        2. Pick a char from the supported set.
        3. Pick a target font.
        4. Pick a source font, optionally constrained != target.
        5. Pick ``ref_count`` reference fonts (may overlap target).
        6. Render target / source / refs and return as tensors.

    The dataset is "virtual": ``__len__`` returns ``length`` and ``idx`` is
    just a seed index, so we can train for arbitrary step counts without
    worrying about epoch boundaries.
    """

    def __init__(
        self,
        *,
        fonts_root: Path,
        font_ids: list[str] | None = None,
        image_size: int = 128,
        content_channels: int = 1,
        font_size_ratio: float = 0.85,
        length: int = 10000,
        ref_count: int = 1,
        seed: int = 42,
        ensure_diff_source: bool = True,
        cjk_start: int = CJK_BASIC_START,
        cjk_end: int = CJK_BASIC_END,
        char_cache_path: Path | None = None,
        script_categories: dict[str, str] | None = None,
    ) -> None:
        fonts_root = Path(fonts_root)
        if not fonts_root.is_dir():
            raise FileNotFoundError(f"fonts_root missing: {fonts_root}")
        if font_ids is None:
            font_ids = sorted([p.name for p in fonts_root.iterdir() if p.is_dir()])
        if len(font_ids) < 2:
            raise ValueError(f"need >=2 fonts for cross-font pairs; got {font_ids}")

        self.fonts_root = fonts_root
        self.font_ids = list(font_ids)
        self.image_size = int(image_size)
        self.content_channels = int(content_channels)
        self.font_size_ratio = float(font_size_ratio)
        self.length = int(length)
        self.ref_count = int(ref_count)
        self.base_seed = int(seed)
        self.ensure_diff_source = bool(ensure_diff_source)

        # Pre-resolve TTF paths so __getitem__ doesn't stat repeatedly.
        self._ttf_paths = {fid: _ttf_path_for(fonts_root, fid) for fid in self.font_ids}

        # Resolve script_id labels. If not provided, hash the font_id family
        # name (kai/xing/cao/hei/ming/decor) — caller may overwrite via the
        # script_categories arg.
        if script_categories is None:
            script_categories = {fid: "unk" for fid in self.font_ids}
        all_scripts = sorted({script_categories.get(fid, "unk") for fid in self.font_ids})
        self._script_id = {s: i for i, s in enumerate(all_scripts)}
        self._font_script = {fid: script_categories.get(fid, "unk") for fid in self.font_ids}

        # Discover supported chars (cached on disk).
        self.chars: list[str] = discover_supported_chars(
            fonts_root=fonts_root,
            font_ids=self.font_ids,
            image_size=self.image_size,
            font_size_ratio=self.font_size_ratio,
            cjk_start=cjk_start,
            cjk_end=cjk_end,
            cache_path=char_cache_path,
        )
        if not self.chars:
            raise RuntimeError(
                "TTFCrossFontPairDataset: no chars survived the cross-font tofu "
                "filter — check fonts_root contents."
            )

    def __len__(self) -> int:
        return self.length

    def metadata(self) -> dict[str, int]:
        return {
            "writer_vocab_size": len(self.font_ids),
            "style_family_vocab_size": len(self.font_ids),
            "char_vocab_size": len(self.chars),
            "script_vocab_size": len(self._script_id),
        }

    def __getitem__(self, idx: int) -> dict[str, Any]:
        # random.Random() only accepts int/str seeds, not tuples. Combine via
        # a simple LCG-style mix so different (base_seed, idx) pairs diverge.
        rng = random.Random(self.base_seed * 1_000_003 + int(idx))
        char = rng.choice(self.chars)
        target_font = rng.choice(self.font_ids)
        if self.ensure_diff_source and len(self.font_ids) > 1:
            source_pool = [f for f in self.font_ids if f != target_font]
            source_font = rng.choice(source_pool)
        else:
            source_font = rng.choice(self.font_ids)
        ref_fonts = [rng.choice(self.font_ids) for _ in range(self.ref_count)]

        image = _array_to_tensor(self._render(target_font, char))
        content_1ch = _array_to_tensor(self._render(source_font, char))
        if self.content_channels == 1:
            content = content_1ch
        else:
            content = content_1ch.repeat(self.content_channels, 1, 1)

        refs = [_array_to_tensor(self._render(f, char)) for f in ref_fonts]

        char_id = self.chars.index(char)
        target_writer_id = self.font_ids.index(target_font)
        script_id = self._script_id[self._font_script[target_font]]

        return {
            "image": image,
            "content": content,
            "char_id": char_id,
            "writer_id": target_writer_id,
            "style_family_id": target_writer_id,
            "unit_id": target_writer_id,
            "script_id": script_id,
            "ref_images": refs,
            "metadata": {
                "char": char,
                "target_font": target_font,
                "source_font": source_font,
                "ref_fonts": list(ref_fonts),
                "idx": idx,
            },
        }

    def _render(self, font_id: str, char: str) -> np.ndarray:
        return render_glyph(
            ttf_path=self._ttf_paths[font_id],
            char=char,
            image_size=self.image_size,
            font_size_ratio=self.font_size_ratio,
        )
