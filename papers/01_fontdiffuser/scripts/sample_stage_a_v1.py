"""Eyeball sampler for Stage A v1 — produce a (source | ref | generated) grid PNG.

Loads the Stage A v1 ckpt, renders 12 source / style-ref pairs from the cached
TTF cross-font corpus, runs ``sample_ddim``, and writes a 12×3 grid showing
(source content | style reference | generated target).

Usage (from papers/01_fontdiffuser/):
    uv run python scripts/sample_stage_a_v1.py \\
        --ckpt outputs/stage_a_ttf_v1/fontdiffuser_last.pt \\
        --fonts-root D:/Char/ayueh/paper_reimpl/data_snapshot/fonts_free \\
        --output outputs/stage_a_ttf_v1/sample_grid.png \\
        --n 12 --ddim-steps 50 --cfg-scale 1.0 --device cuda:0
"""

from __future__ import annotations

import argparse
import random
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image

# Add src to path so we can import without an editable install.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fontdiffuser.model import FontDiffuserConfig, build_fontdiffuser
from fontdiffuser.sample import sample_ddim
from paper_reimpl_shared.data.ttf_pair_dataset import TTFCrossFontPairDataset
from paper_reimpl_shared.diffusion.gaussian import GaussianDiffusion


def _denorm_to_png(t: torch.Tensor) -> np.ndarray:
    """[1, H, W] tensor in [-1, 1] -> uint8 [H, W] in [0, 255]."""
    t = t.detach().cpu().clamp(-1, 1)
    arr = ((t + 1.0) * 127.5).round().to(torch.uint8).squeeze(0).numpy()
    return arr


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True, type=Path)
    p.add_argument("--fonts-root", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--n", type=int, default=12, help="number of (char, pair) samples")
    p.add_argument("--ddim-steps", type=int, default=50)
    p.add_argument("--cfg-scale", type=float, default=1.0)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--image-size", type=int, default=128)
    p.add_argument("--font-size-ratio", type=float, default=0.85)
    args = p.parse_args()

    device = torch.device(args.device)
    torch.manual_seed(args.seed)

    print(f"[sample] loading ckpt {args.ckpt}")
    blob = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg_dict = blob["cfg"]
    cfg = FontDiffuserConfig(**cfg_dict)
    model = build_fontdiffuser(cfg).to(device)
    model.load_state_dict(blob["model"])
    model.eval()

    diffusion = GaussianDiffusion(
        timesteps=1000,
        beta_start=1e-4,
        beta_end=2e-2,
        beta_schedule="linear",
        prediction_target="epsilon",
        device=device,
    )

    print(f"[sample] loading TTF dataset (fonts_root={args.fonts_root})")
    ds = TTFCrossFontPairDataset(
        fonts_root=args.fonts_root,
        image_size=args.image_size,
        content_channels=cfg.content_channels,
        font_size_ratio=args.font_size_ratio,
        length=10_000,
        ref_count=1,
        seed=args.seed,
        ensure_diff_source=True,
    )
    print(f"[sample] dataset has {len(ds.font_ids)} fonts, {len(ds.chars)} chars")

    rng = random.Random(args.seed)
    indices = rng.sample(range(len(ds)), args.n)

    cells = []  # list of (source_arr, ref_arr, gen_arr, meta_dict)
    for ii, idx in enumerate(indices):
        s = ds[idx]
        content = s["content"].unsqueeze(0).to(device)   # [1, C, H, W]
        ref = s["ref_images"][0].unsqueeze(0).to(device)  # [1, 1, H, W]
        target = sample_ddim(
            model=model,
            diffusion=diffusion,
            content=content,
            ref_image=ref,
            cfg_scale=args.cfg_scale,
            device=device,
        )  # [1, 1, H, W] in [-1, 1]

        # source content is the first content channel
        src_png = _denorm_to_png(content[0, :1])
        ref_png = _denorm_to_png(ref[0])
        gen_png = _denorm_to_png(target[0])
        meta = s["metadata"]
        cells.append((src_png, ref_png, gen_png, meta))
        print(
            f"[sample] {ii+1}/{args.n} char='{meta['char']}' "
            f"target={meta['target_font']} source={meta['source_font']} "
            f"ref={meta['ref_fonts'][0]}"
        )

    # Compose a (n × 3 × H × W) uint8 image, white separator between columns
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
    print(f"[sample] wrote {args.output}  size={out_h}x{out_w}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
