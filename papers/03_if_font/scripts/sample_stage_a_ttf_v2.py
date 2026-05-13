"""Eyeball sampler for 03 IF-Font Stage A v2 — produce a (source | ref | generated) grid.

Loads the Stage A v2 ckpt + pretrained VQGAN (already baked into ckpt's vq.* keys),
samples 12 (char, font-pair) examples from the TTF cross-font corpus, runs the
autoregressive ``model.sample()``, decodes via VQGAN, and writes a 12x3 grid PNG.

The model expects RGB images; the rendered grid is converted to grayscale for
display consistency with 01/02 sample grids.

Usage (from papers/03_if_font/):
    uv run python scripts/sample_stage_a_ttf_v2.py \\
        --ckpt outputs/stage_a_ttf_v2/if_font_last.pt \\
        --fonts-root D:/Char/ayueh/paper_reimpl/data_snapshot/fonts_free \\
        --output outputs/stage_a_ttf_v2/sample_grid.png \\
        --n 12 --device cpu
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

from if_font.dataset import _IFFontTTFAdapter, IFFontCollate
from if_font.ids import IDSTokenizer
from if_font.model import IFFontConfig, build_if_font
from paper_reimpl_shared.data.ttf_pair_dataset import TTFCrossFontPairDataset


def _denorm_rgb_to_gray_png(t: torch.Tensor) -> np.ndarray:
    """[C, H, W] tensor in [-1, 1] -> uint8 [H, W] grayscale (avg over channels)."""
    t = t.detach().cpu().clamp(-1, 1)
    if t.shape[0] == 3:
        t = t.mean(dim=0, keepdim=True)
    arr = ((t + 1.0) * 127.5).round().to(torch.uint8).squeeze(0).numpy()
    return arr


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True, type=Path)
    p.add_argument("--fonts-root", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--n", type=int, default=12)
    p.add_argument("--device", default="cpu")
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--image-size", type=int, default=128)
    p.add_argument("--font-size-ratio", type=float, default=0.85)
    p.add_argument("--max-refs", type=int, default=3)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top-k", type=int, default=100)
    args = p.parse_args()

    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    print(f"[03-sample] loading ckpt {args.ckpt}")
    blob = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = IFFontConfig(**blob["cfg"]) if not isinstance(blob["cfg"], IFFontConfig) else blob["cfg"]
    model = build_if_font(cfg)
    miss, unexp = model.load_state_dict(blob["model"], strict=False)
    print(f"[03-sample] state_dict loaded (missing={len(miss)} unexpected={len(unexp)})")
    model.to(device).eval()

    # Tokenizer: build from IDC only — matches train.py behavior when no
    # IDS resolver is available (which is what Stage A TTF runs used).
    tokenizer = IDSTokenizer.from_idc_only()
    if tokenizer.vocab_size < cfg.ids_vocab_size:
        tokenizer.freeze()

    print(f"[03-sample] loading TTF dataset (fonts_root={args.fonts_root})")
    inner = TTFCrossFontPairDataset(
        fonts_root=args.fonts_root,
        image_size=args.image_size,
        content_channels=cfg.in_channels,
        font_size_ratio=args.font_size_ratio,
        length=10_000,
        ref_count=args.max_refs,
        seed=args.seed,
        ensure_diff_source=True,
    )
    ds = _IFFontTTFAdapter(
        inner=inner,
        ids_resolver=None,
        ids_lookup=lambda _ch: "",
        max_refs=args.max_refs,
    )
    collate = IFFontCollate(
        tokenizer=tokenizer,
        max_refs=args.max_refs,
        ids_max_len=cfg.ids_max_len,
        in_channels=cfg.in_channels,
        fit_on_first_call=False,
    )
    print(f"[03-sample] dataset has {len(inner.font_ids)} fonts, {len(inner.chars)} chars")

    indices = random.sample(range(len(ds)), args.n)
    cells = []
    for ii, idx in enumerate(indices):
        item = ds[idx]
        batch = collate([item])
        with torch.no_grad():
            target = model.sample(
                ids_token_ids=batch["ids_token_ids"].to(device),
                ref_images=batch["ref_images"].to(device),
                coverage_sim=batch["coverage_sim"].to(device),
                temperature=args.temperature,
                top_k=args.top_k,
                sample=True,
            )  # [1, C, H, W] in [-1, 1]

        meta = item["metadata"]
        src_png = _denorm_rgb_to_gray_png(item["content"])
        ref_png = _denorm_rgb_to_gray_png(item["ref_images"][0])
        gen_png = _denorm_rgb_to_gray_png(target[0])
        cells.append((src_png, ref_png, gen_png, meta))
        print(
            f"[03-sample] {ii+1}/{args.n} char='{meta['char']}' "
            f"target={meta['target_font']} source={meta['source_font']} "
            f"ref={meta['ref_fonts'][0]}"
        )

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
    print(f"[03-sample] wrote {args.output}  size={out_h}x{out_w}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
