"""Eyeball sampler for 06 Calliffusion Stage A v2.

Loads ckpt, samples 12 (char, font) prompts and runs ``sample_prompts``.
Calliffusion uses BERT text encoder + 1-channel U-Net.
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from paper_reimpl_shared.diffusion.gaussian import GaussianDiffusion

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import yaml

from calliffusion.model import CalliffusionUNetConfig, build_unet_from_yaml
from calliffusion.sample import sample_prompts
from calliffusion.text import build_text_encoder
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
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--image-size", type=int, default=64)
    p.add_argument("--cfg-scale", type=float, default=2.0)
    args = p.parse_args()

    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    print(f"[06-sample] loading ckpt {args.ckpt}")
    model_yaml = yaml.safe_load((ROOT / "src" / "calliffusion" / "configs" / "model.yaml").read_text())
    unet_raw = model_yaml.get("unet", {})
    cfg = CalliffusionUNetConfig(
        image_size=int(unet_raw.get("image_size", 64)),
        in_channels=int(unet_raw.get("in_channels", 1)),
        out_channels=int(unet_raw.get("out_channels", 1)),
        base_channels=int(unet_raw.get("base_channels", 64)),
        channel_mult=list(unet_raw.get("channel_mult", [1, 2, 4, 4])),
        num_res_blocks=int(unet_raw.get("num_res_blocks", 1)),
        time_emb_dim=int(unet_raw.get("time_emb_dim", 256)),
        context_dim=int(unet_raw.get("context_dim", 768)),
        num_heads=int(unet_raw.get("num_heads", 8)),
        dropout=float(unet_raw.get("dropout", 0.0)),
    )
    unet = build_unet_from_yaml({"unet": unet_raw})
    blob = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    state = blob["model"] if isinstance(blob, dict) and "model" in blob else blob
    miss, unexp = unet.load_state_dict(state, strict=False)
    print(f"[06-sample] U-Net state: missing={len(miss)} unexpected={len(unexp)}")
    unet.to(device).eval()

    text_cfg = model_yaml.get("text_encoder", {})
    text_encoder = build_text_encoder(text_cfg).to(device)
    text_encoder.eval()

    diffusion = GaussianDiffusion(
        timesteps=1000,
        beta_start=1e-4,
        beta_end=2e-2,
        beta_schedule="linear",
        prediction_target="epsilon",
        device=device,
    )

    print(f"[06-sample] loading TTF dataset from {args.fonts_root}")
    inner = TTFCrossFontPairDataset(
        fonts_root=args.fonts_root,
        image_size=args.image_size,
        content_channels=cfg.in_channels,
        font_size_ratio=0.85,
        length=10_000,
        ref_count=1,
        seed=args.seed,
        ensure_diff_source=True,
    )
    print(f"[06-sample] dataset has {len(inner.font_ids)} fonts, {len(inner.chars)} chars")

    indices = random.sample(range(len(inner)), args.n)
    cells = []
    for ii, idx in enumerate(indices):
        s = inner[idx]
        meta = s["metadata"]
        ch = meta["char"]
        target_font = meta["target_font"]
        prompt = f"{ch} {target_font}"
        gen = sample_prompts(
            unet, text_encoder, diffusion,
            prompts=[prompt],
            shape=(1, cfg.in_channels, args.image_size, args.image_size),
            sampler="ddim",
            cfg_scale=args.cfg_scale,
            device=device,
        )
        cells.append((_denorm(s["content"]), _denorm(s["ref_images"][0]), _denorm(gen[0]), meta))
        print(f"[06-sample] {ii+1}/{args.n} prompt='{prompt}'")

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
    print(f"[06-sample] wrote {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
