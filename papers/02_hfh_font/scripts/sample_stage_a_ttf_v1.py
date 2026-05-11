"""Eyeball sampler for 02_hfh_font Stage A TTF v1.

Loads the v1 ckpt and produces a 12x3 grid PNG of (source content | one
style ref | generated target). HFH samples in latent space then VAE-decodes,
which the script handles via sample.sample_latents + decode_samples.

Usage (from papers/02_hfh_font/):
    uv run python scripts/sample_stage_a_ttf_v1.py \\
        --ckpt outputs/stage_a_ttf_v1/hfh_font_last.pt \\
        --fonts-root D:/Char/ayueh/paper_reimpl/data_snapshot/fonts_free \\
        --output outputs/stage_a_ttf_v1/sample_grid.png \\
        --n 12 --ddim-steps 50 --cfg-scale 2.0 --device cuda:1
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

from hfh_font.dataset import build_collate
from hfh_font.model import ModelConfig, build_model
from hfh_font.sample import decode_samples, sample_latents
from paper_reimpl_shared.data.ttf_pair_dataset import TTFCrossFontPairDataset
from paper_reimpl_shared.diffusion.gaussian import GaussianDiffusion


def _denorm_to_png(t: torch.Tensor) -> np.ndarray:
    """[1, H, W] tensor in [-1, 1] -> uint8 [H, W] (white=255, ink=0)."""
    t = t.detach().cpu().clamp(-1, 1)
    arr = ((t + 1.0) * 127.5).round().to(torch.uint8).squeeze(0).numpy()
    return arr


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True, type=Path)
    p.add_argument("--fonts-root", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--n", type=int, default=12)
    p.add_argument("--ddim-steps", type=int, default=50)
    p.add_argument("--cfg-scale", type=float, default=2.0)
    p.add_argument("--device", default="cuda:1")
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--image-size", type=int, default=128)
    p.add_argument("--font-size-ratio", type=float, default=0.85)
    p.add_argument("--n-refs", type=int, default=4)
    args = p.parse_args()

    device = torch.device(args.device)
    torch.manual_seed(args.seed)

    print(f"[02-sample] loading ckpt {args.ckpt}")
    blob = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg_dict = blob["cfg"]
    cfg = ModelConfig.from_dict(cfg_dict)
    model = build_model(cfg).to(device)
    model.load_state_dict(blob["model"])
    model.eval()

    diffusion = GaussianDiffusion(
        timesteps=cfg.diffusion_timesteps,
        beta_start=1e-4,
        beta_end=2e-2,
        beta_schedule="linear",
        prediction_target=cfg.diffusion_target,
        device=device,
    )

    print(f"[02-sample] loading TTF dataset (fonts_root={args.fonts_root})")
    ds = TTFCrossFontPairDataset(
        fonts_root=args.fonts_root,
        image_size=args.image_size,
        content_channels=cfg.content_channels,
        font_size_ratio=args.font_size_ratio,
        length=10_000,
        ref_count=args.n_refs,
        seed=args.seed,
        ensure_diff_source=True,
    )
    print(f"[02-sample] {len(ds.font_ids)} fonts, {len(ds.chars)} chars")

    rng = random.Random(args.seed)
    indices = rng.sample(range(len(ds)), args.n)
    collate = build_collate(n_refs=args.n_refs)

    cells = []
    for ii, idx in enumerate(indices):
        item = ds[idx]
        # collate expects a list; we batch one example so the model sees [1, ...]
        batch = collate([item])
        # Move tensors to device
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(device)

        latents = sample_latents(
            model,
            diffusion,
            batch=batch,
            sampler="ddim",
            cfg_scale=args.cfg_scale,
            device=device,
        )
        decoded = decode_samples(model, latents)  # [1, 1, H, W] in [-1, 1]
        src_png = _denorm_to_png(item["content"][:1])
        ref_png = _denorm_to_png(item["ref_images"][0])
        gen_png = _denorm_to_png(decoded[0])

        meta = item["metadata"]
        cells.append((src_png, ref_png, gen_png))
        print(
            f"[02-sample] {ii+1}/{args.n} char='{meta['char']}' "
            f"target={meta['target_font']} source={meta['source_font']} "
            f"ref0={meta['ref_fonts'][0]}"
        )

    H = args.image_size
    sep = 4
    out_h = H * args.n + sep * (args.n - 1)
    out_w = H * 3 + sep * 2
    canvas = np.full((out_h, out_w), 255, dtype=np.uint8)
    for row, (src, ref, gen) in enumerate(cells):
        y = row * (H + sep)
        canvas[y : y + H, 0:H] = src
        canvas[y : y + H, H + sep : H + sep + H] = ref
        canvas[y : y + H, 2 * (H + sep) : 2 * (H + sep) + H] = gen

    args.output.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(canvas, mode="L").save(args.output)
    print(f"[02-sample] wrote {args.output} ({out_h}x{out_w})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
