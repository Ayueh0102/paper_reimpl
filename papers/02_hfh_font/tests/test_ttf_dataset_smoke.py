"""Smoke test for 02_hfh_font + TTFCrossFontPairDataset routing.

Validates the new `source: ttf` branch in dataset.py.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from hfh_font.dataset import build_collate, build_dataset
from paper_reimpl_shared.data.ttf_pair_dataset import TTFCrossFontPairDataset


def _find_fonts_root() -> Path | None:
    candidates = [
        Path.home() / "Char" / "ernantang-jit-calligraphy-generation" / "data" / "fonts_free",
        Path("/Users/Ayueh/Char/ernantang-jit-calligraphy-generation/data/fonts_free"),
    ]
    for c in candidates:
        if c.is_dir():
            return c
    return None


@pytest.mark.skipif(_find_fonts_root() is None, reason="fonts_free not available")
def test_02_dataset_ttf_routing_tiny():
    fonts_root = _find_fonts_root()
    data_cfg = {
        "source": "ttf",
        "fonts_root": str(fonts_root),
        "font_ids": ["lxgw_wenkai_regular", "noto_sans_sc"],
        "font_size_ratio": 0.85,
        "ttf_epoch_length": 4,
        "ensure_diff_source": True,
        "cjk_start": 0x4E00,
        "cjk_end": 0x4E2F,  # tiny slice
        "seed": 0,
    }
    # Minimal stand-in for BackendPaths: only attr the ttf branch reads is
    # paths.ttf_root, and we provide an absolute fonts_root override so it
    # is never consulted.
    class DummyPaths:
        ttf_root = fonts_root.parent / "ttf_renders"
        manifest_root = fonts_root.parent
        content_cache_root = fonts_root.parent

    ds = build_dataset(
        data_cfg=data_cfg,
        backend="mac_symlink",
        synthetic=False,
        paths=DummyPaths(),  # type: ignore[arg-type]
        image_size=32,
        content_channels=["bitmap", "sdf", "skeleton"],
        n_refs=4,
    )
    assert isinstance(ds, TTFCrossFontPairDataset)
    assert len(ds) == 4
    sample = ds[0]
    # HFH expects content broadcast to len(content_channels)=3 channels
    assert sample["image"].shape == (1, 32, 32)
    assert sample["content"].shape == (3, 32, 32)
    assert len(sample["ref_images"]) == 4
    for r in sample["ref_images"]:
        assert r.shape == (1, 32, 32)
    # End-to-end via the collate so the DataLoader path works too.
    collate = build_collate(n_refs=4)
    batch = collate([ds[0], ds[1]])
    assert batch["image"].shape == (2, 1, 32, 32)
    assert batch["content"].shape == (2, 3, 32, 32)
    assert batch["ref_images"].shape == (2, 4, 1, 32, 32)
