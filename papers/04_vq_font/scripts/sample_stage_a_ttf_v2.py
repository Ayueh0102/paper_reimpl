"""Eyeball sampler for 04 VQ-Font Stage A v2.

Loads ckpt, samples 12 (char, font-pair) examples from the TTF cross-font
corpus, runs ``sample_vq_font`` (frozen VQGAN + Transformer token prior),
and writes a 12x3 grid PNG (source | ref0 | generated).
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vq_font.dataset import _VQFontTTFAdapter
from vq_font.model import VQFontConfig, build_vq_font
from vq_font.sample import sample_vq_font
from paper_reimpl_shared.data.ttf_pair_dataset import TTFCrossFontPairDataset


def _denorm(t: torch.Tensor) -> np.ndarray:
    t = t.detach().cpu().clamp(-1, 1)
    if t.dim() == 3 and t.shape[0] == 3:
        t = t.mean(dim=0, keepdim=True)
    arr = ((t + 1.0) * 127.5).round().to(torch.uint8).squeeze(0).numpy()
    return arr


def _dict_to_cfg(cfg_dict):
    """Reconstruct nested dataclasses from torch.save'd dict."""
    if isinstance(cfg_dict, VQFontConfig):
        return cfg_dict
    return VQFontConfig(**cfg_dict)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True, type=Path)
    p.add_argument("--fonts-root", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--n", type=int, default=12)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--image-size", type=int, default=128)
    p.add_argument("--font-size-ratio", type=float, default=0.85)
    p.add_argument("--max-refs", type=int, default=3)
    args = p.parse_args()

    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    print(f"[04-sample] loading ckpt {args.ckpt}")
    blob = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = _dict_to_cfg(blob.get("cfg"))
    model = build_vq_font(cfg)
    state = blob["model"] if "model" in blob else blob
    miss, unexp = model.load_state_dict(state, strict=False)
    print(f"[04-sample] missing={len(miss)} unexpected={len(unexp)}")
    model.to(device).eval()

    print(f"[04-sample] loading TTF dataset from {args.fonts_root}")
    inner = TTFCrossFontPairDataset(
        fonts_root=args.fonts_root,
        image_size=args.image_size,
        content_channels=getattr(cfg, "in_channels", 1),
        font_size_ratio=args.font_size_ratio,
        length=10_000,
        ref_count=args.max_refs,
        seed=args.seed,
        ensure_diff_source=True,
    )
    ds = _VQFontTTFAdapter(inner=inner, max_refs=args.max_refs)
    print(f"[04-sample] dataset has {len(inner.font_ids)} fonts, {len(inner.chars)} chars")

    indices = random.sample(range(len(ds)), args.n)
    cells = []
    for ii, idx in enumerate(indices):
        s = ds[idx]
        initial = s["content"].unsqueeze(0).to(device)
        refs = torch.stack(s["ref_images"]).unsqueeze(0).to(device)
        structure_id = torch.tensor([s.get("structure_id", 0)], dtype=torch.long, device=device)
        target = sample_vq_font(
            model=model,
            initial_glyph=initial,
            ref_glyphs=refs,
            structure_id=structure_id,
            mode="argmax",
        )
        meta = s["metadata"]
        cells.append(
            (_denorm(s["content"]), _denorm(s["ref_images"][0]), _denorm(target[0]), meta)
        )
        print(f"[04-sample] {ii+1}/{args.n} char='{meta['char']}'")

    H = args.image_size
    sep = 4
    out_h = H * args.n + sep * (args.n - 1)
    out_w = H * 3 + sep * 2
    canvas = np.full((out_h, out_w), 255, dtype=np.uint8)
    for row, (src, ref, gen, _meta) in enumerate(cells):
        y = row * (H + sep)
        canvas[y : y + H, 0:H] = src
        canvas[y : y + H, H + sep : H + sep + H] = ref
        canvas[y : y + H, 2 * (H + sep) : 2 * (H + sep) + H] = gen

    args.output.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(canvas, mode="L").save(args.output)
    print(f"[04-sample] wrote {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
