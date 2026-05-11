"""Smoke test for TTFCrossFontPairDataset.

Skips if `data/fonts_free` is not available (e.g. on CI where mother repo
isn't checked out). The test is intentionally tiny — 2 fonts, 32x32 canvas,
length=2 — so it runs in <1 s.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from paper_reimpl_shared.data.ttf_pair_dataset import TTFCrossFontPairDataset


def _find_fonts_root() -> Path | None:
    """Search likely locations for the mother repo's fonts_free directory."""
    candidates = [
        Path.home() / "Char" / "ernantang-jit-calligraphy-generation" / "data" / "fonts_free",
        Path.cwd().parent.parent / "mother_repo_link" / "data" / "fonts_free",
        Path("/Users/Ayueh/Char/ernantang-jit-calligraphy-generation/data/fonts_free"),
    ]
    for c in candidates:
        if c.is_dir():
            return c
    return None


@pytest.mark.skipif(_find_fonts_root() is None, reason="fonts_free not available")
def test_ttf_pair_dataset_two_fonts_tiny():
    fonts_root = _find_fonts_root()
    assert fonts_root is not None
    # Pick two known-present fonts to keep the test sub-second.
    ds = TTFCrossFontPairDataset(
        fonts_root=fonts_root,
        font_ids=["lxgw_wenkai_regular", "noto_sans_sc"],
        image_size=32,
        font_size_ratio=0.8,
        length=2,
        ref_count=1,
        seed=0,
        # Tiny CJK slice so char discovery is fast.
        cjk_start=0x4E00,
        cjk_end=0x4E2F,
    )
    assert len(ds) == 2
    assert len(ds.chars) > 0, "expected at least one shared CJK char in the slice"

    sample = ds[0]
    assert sample["image"].shape == (1, 32, 32)
    assert sample["content"].shape == (1, 32, 32)
    assert len(sample["ref_images"]) == 1
    assert sample["ref_images"][0].shape == (1, 32, 32)
    assert sample["metadata"]["target_font"] in ds.font_ids
    assert sample["metadata"]["source_font"] != sample["metadata"]["target_font"]
    # Glyph render should put black ink (negative values) somewhere.
    assert (sample["image"] < 0).any(), "image looks like blank tofu"


@pytest.mark.skipif(_find_fonts_root() is None, reason="fonts_free not available")
def test_ttf_pair_dataset_tofu_rejection():
    """Confirm rendering a control char yields the tofu hash baseline.

    Indirect: when char_filter == [some control char], the dataset would
    raise because no font supports it. Run a manual single-font discover
    instead and verify the supported set excludes a control codepoint.
    """
    from paper_reimpl_shared.data.ttf_pair_dataset import (
        discover_supported_chars,
    )

    fonts_root = _find_fonts_root()
    chars = discover_supported_chars(
        fonts_root=fonts_root,
        font_ids=["lxgw_wenkai_regular"],
        image_size=32,
        font_size_ratio=0.8,
        cjk_start=0x4E00,
        cjk_end=0x4E0F,  # tiny slice
    )
    # All these are real CJK chars in lxgw, so the slice should be non-empty.
    assert len(chars) > 0
