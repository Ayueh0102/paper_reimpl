"""08 sample with FIXED source font + varying writer_id.

Hypothesis: 08's content path (dense source bitmap fused per U-Net stage)
is so dominant that the output essentially copies the source glyph; the
writer_id integer only nudges style weakly. To verify, hold the source
font constant (noto_sans_sc, neutral hei) and ask the model to render the
same 12 chars under different writer_ids. If gen still looks mostly like
the source (sans), content-copy hypothesis is confirmed.

Output: 12 x 3 grid (source_fixed | gt_target_font | gen).
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

from dp_font.dataset import _DPFontTTFAdapter
from dp_font.model import DPFontConfig, build_dp_font
from dp_font.sample import sample_ddim
from paper_reimpl_shared.data.ttf_pair_dataset import (
    TTFCrossFontPairDataset,
    render_glyph,
    _array_to_tensor,
)


def _denorm(t):
    t = t.detach().cpu().clamp(-1, 1)
    if t.dim() == 3 and t.shape[0] >= 1:
        t = t[0:1]
    return ((t + 1.0) * 127.5).round().to(torch.uint8).squeeze(0).numpy()


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True, type=Path)
    p.add_argument("--fonts-root", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--n", type=int, default=12)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--image-size", type=int, default=80)
    p.add_argument("--cfg-scale", type=float, default=2.0)
    p.add_argument("--source-font", default="noto_sans_sc",
                   help="Fixed source font for content path (default: neutral sans)")
    args = p.parse_args()

    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    print(f"[08-fix] loading ckpt {args.ckpt}")
    blob = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = blob["cfg"] if isinstance(blob.get("cfg"), DPFontConfig) else DPFontConfig(**blob["cfg"])
    model = build_dp_font(cfg)
    state = blob["model"] if "model" in blob else blob
    miss, unexp = model.load_state_dict(state, strict=False)
    print(f"[08-fix] missing={len(miss)} unexpected={len(unexp)}")
    model.to(device).eval()

    diffusion = GaussianDiffusion(
        timesteps=1000, beta_start=1e-4, beta_end=2e-2,
        beta_schedule="cosine", prediction_target="epsilon", device=device,
    )

    inner = TTFCrossFontPairDataset(
        fonts_root=args.fonts_root,
        image_size=args.image_size,
        content_channels=cfg.content_channels,
        font_size_ratio=0.85,
        length=10_000,
        ref_count=1,
        seed=args.seed,
        ensure_diff_source=True,
    )
    ds = _DPFontTTFAdapter(
        inner=inner,
        stroke_vocab_size=cfg.stroke_vocab_size,
        stroke_seq_len=cfg.stroke_seq_len,
    )
    print(f"[08-fix] dataset {len(inner.font_ids)} fonts, {len(inner.chars)} chars, fixing source={args.source_font}")

    if args.source_font not in inner.font_ids:
        raise ValueError(f"source font {args.source_font} not in {inner.font_ids}")

    indices = random.sample(range(len(ds)), args.n)
    cells = []
    for ii, idx in enumerate(indices):
        s = ds[idx]
        meta = s["metadata"]
        ch = meta["char"]
        # Override content with FIXED source font's rendering of the same char.
        fixed_arr = inner._render(args.source_font, ch)
        fixed_content = _array_to_tensor(fixed_arr)
        if cfg.content_channels > 1:
            fixed_content = fixed_content.repeat(cfg.content_channels, 1, 1)
        content = fixed_content.unsqueeze(0).to(device)

        writer_id = torch.tensor([s.get("writer_id", 0)], dtype=torch.long, device=device)
        script_id = torch.tensor([s.get("script_id", 0)], dtype=torch.long, device=device)
        char_id = torch.tensor([s.get("char_id", 0)], dtype=torch.long, device=device)
        gen = sample_ddim(
            model=model, diffusion=diffusion,
            content=content,
            writer_id=writer_id, script_id=script_id, char_id=char_id,
            cfg_scale=args.cfg_scale,
            device=device,
        )
        cells.append((_denorm(fixed_content), _denorm(s["image"]), _denorm(gen[0]), meta))
        print(f"[08-fix] {ii+1}/{args.n} char='{ch}' target_font={meta['target_font']} writer_id={s.get('writer_id', 0)}")

    H = args.image_size
    sep = 4
    out_h = H * args.n + sep * (args.n - 1)
    out_w = H * 3 + sep * 2
    canvas = np.full((out_h, out_w), 255, dtype=np.uint8)
    for row, (src, gt, gen, _meta) in enumerate(cells):
        y = row * (H + sep)
        canvas[y : y + H, 0:H] = src
        canvas[y : y + H, H + sep : H + sep + H] = gt
        canvas[y : y + H, 2 * (H + sep) : 2 * (H + sep) + H] = gen

    args.output.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(canvas, mode="L").save(args.output)
    print(f"[08-fix] wrote {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
