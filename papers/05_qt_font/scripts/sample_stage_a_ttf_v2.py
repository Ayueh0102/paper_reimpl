"""Eyeball sampler for 05 QT-Font Stage A v2.

Loads ckpt + sparse-attn / depth=6 model, samples 12 char/font pairs from
TTF, runs ``sample_image`` (D3PM uniform reverse over quadtree labels),
maps {bg, contour, skeleton} -> {-1, 0, +1}, and writes a 12x3 grid PNG.
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

from qt_font.model import D3PMUniform, QTFontConfig, build_qt_font
from qt_font.sample import sample_image
from paper_reimpl_shared.data.ttf_pair_dataset import TTFCrossFontPairDataset


def _denorm(t: torch.Tensor) -> np.ndarray:
    t = t.detach().cpu().clamp(-1, 1)
    if t.dim() == 3 and t.shape[0] >= 1:
        t = t[0:1]
    arr = ((t + 1.0) * 127.5).round().to(torch.uint8).squeeze(0).numpy()
    return arr


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True, type=Path)
    p.add_argument("--fonts-root", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--n", type=int, default=12)
    p.add_argument("--device", default="cuda:1")
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--max-refs", type=int, default=1)
    args = p.parse_args()

    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    print(f"[05-sample] loading ckpt {args.ckpt}")
    blob = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg_dict = blob.get("cfg")
    cfg = cfg_dict if isinstance(cfg_dict, QTFontConfig) else QTFontConfig(**cfg_dict)
    model = build_qt_font(cfg)
    state = blob["model"] if "model" in blob else blob
    miss, unexp = model.load_state_dict(state, strict=False)
    print(f"[05-sample] missing={len(miss)} unexpected={len(unexp)}")
    model.to(device).eval()

    diffusion = D3PMUniform(
        n_states=cfg.n_states,
        timesteps=cfg.timesteps,
        schedule=cfg.schedule,
        beta_start=0.02,
        beta_end=1.0,
    ).to(device)

    image_size = 1 << cfg.depth
    print(f"[05-sample] loading TTF dataset (image_size={image_size})")
    inner = TTFCrossFontPairDataset(
        fonts_root=args.fonts_root,
        image_size=image_size,
        content_channels=cfg.content_channels,
        font_size_ratio=0.85,
        length=10_000,
        ref_count=args.max_refs,
        seed=args.seed,
        ensure_diff_source=True,
    )
    print(f"[05-sample] dataset has {len(inner.font_ids)} fonts, {len(inner.chars)} chars")

    indices = random.sample(range(len(inner)), args.n)
    cells = []
    for ii, idx in enumerate(indices):
        s = inner[idx]
        content = s["content"].unsqueeze(0).to(device)
        refs = torch.stack(s["ref_images"]).unsqueeze(0).to(device)
        gen = sample_image(
            model, diffusion,
            batch_size=1,
            content=content,
            refs=refs,
        )
        meta = s["metadata"]
        cells.append((_denorm(s["content"]), _denorm(s["ref_images"][0]), _denorm(gen[0]), meta))
        print(f"[05-sample] {ii+1}/{args.n} char='{meta['char']}'")

    H = image_size
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
    print(f"[05-sample] wrote {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
