"""TTF glyph renderer / index.

Mother repo's `data/ttf_renders/` has 13 fonts; this module gives a clean
interface to enumerate glyphs and load them as grayscale tensors. Used by
Stage A pretraining where every paper reads cross-font style transfer pairs.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image


def list_fonts(ttf_root: Path) -> list[str]:
    """Return all font subdirectory names under the TTF render root."""
    return sorted([p.name for p in ttf_root.iterdir() if p.is_dir()])


def load_render_manifest(ttf_root: Path) -> dict:
    """Load MANIFEST_RENDER.json describing per-font glyph counts and scripts."""
    manifest = ttf_root / "MANIFEST_RENDER.json"
    if not manifest.exists():
        return {}
    with manifest.open() as f:
        return json.load(f)


def load_glyph(
    ttf_root: Path, *, font_name: str, char: str, image_size: int = 256
) -> torch.Tensor:
    """Load a rendered glyph as [1, H, W] tensor in [-1, 1].

    Args:
        ttf_root: root containing per-font subdirs.
        font_name: subdir name (e.g. 'lxgw_wenkai_regular').
        char: single character to render. Path is <ttf_root>/<font>/<unicode_hex>.png.
        image_size: target H/W. PIL resize with bicubic if not matching.

    Returns:
        [1, image_size, image_size] tensor in [-1, 1].
    """
    code = f"u{ord(char):04x}"
    path = ttf_root / font_name / f"{code}.png"
    if not path.exists():
        raise FileNotFoundError(f"Glyph not found: {path}")
    img = Image.open(path).convert("L")
    if img.size != (image_size, image_size):
        img = img.resize((image_size, image_size), Image.BICUBIC)
    arr = np.asarray(img, dtype=np.float32) / 127.5 - 1.0
    return torch.from_numpy(arr).unsqueeze(0)


def synthetic_glyph(image_size: int = 256) -> torch.Tensor:
    """Random glyph for smoke tests."""
    return torch.randn(1, image_size, image_size).clamp(-1, 1)
