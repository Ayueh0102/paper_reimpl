"""08 Stage B sample on real ernantang writers.

Picks 12 (writer, char) pairs from the enriched manifest, generates with
the model conditioned on each pair's writer_label_id + char_label_id,
and writes a 12x3 grid PNG (source-TTF | GT-ernantang | generated).

Source column uses lxgw_wenkai_regular (neutral kai) as character layout
input — model's content path expects 1-channel bitmap.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dp_font.model import DPFontConfig, build_dp_font
from dp_font.sample import sample_ddim
from paper_reimpl_shared.data.ttf_pair_dataset import render_glyph, _array_to_tensor
from paper_reimpl_shared.diffusion.gaussian import GaussianDiffusion


def _denorm(t):
    t = t.detach().cpu().clamp(-1, 1)
    if t.dim() == 3 and t.shape[0] >= 1:
        t = t[0:1]
    return ((t + 1.0) * 127.5).round().to(torch.uint8).squeeze(0).numpy()


def _load_pil_resize(path: str, size: int) -> np.ndarray:
    img = Image.open(path).convert("L").resize((size, size), Image.BILINEAR)
    return np.array(img, dtype=np.uint8)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True, type=Path)
    p.add_argument("--manifest", required=True, type=Path,
                   help="Enriched manifest JSONL (must have image_path + writer_label_id + char_label_id)")
    p.add_argument("--fonts-root", required=True, type=Path,
                   help="TTF fonts root (for source render)")
    p.add_argument("--source-font", default="lxgw_wenkai_regular",
                   help="Neutral kai font for source/content column")
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--n", type=int, default=12)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--image-size", type=int, default=80)
    p.add_argument("--cfg-scale", type=float, default=2.0)
    args = p.parse_args()

    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    print(f"[08-sb-sample] loading ckpt {args.ckpt}")
    blob = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg_dict = blob.get("cfg")
    cfg = cfg_dict if isinstance(cfg_dict, DPFontConfig) else DPFontConfig(**cfg_dict)
    model = build_dp_font(cfg)
    state = blob["model"] if isinstance(blob, dict) and "model" in blob else blob
    miss, unexp = model.load_state_dict(state, strict=False)
    print(f"[08-sb-sample] missing={len(miss)} unexpected={len(unexp)}")
    model.to(device).eval()

    diffusion = GaussianDiffusion(
        timesteps=1000, beta_start=1e-4, beta_end=2e-2,
        beta_schedule="cosine", prediction_target="epsilon", device=device,
    )

    print(f"[08-sb-sample] reading enriched manifest {args.manifest}")
    rows = []
    with args.manifest.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    print(f"[08-sb-sample] manifest has {len(rows)} rows")

    # Filter rows whose image actually exists on disk + diversify writers.
    valid = [r for r in rows if Path(r["image_path"]).exists()]
    print(f"[08-sb-sample] {len(valid)} rows have image on disk")
    if len(valid) < args.n:
        raise ValueError(f"Only {len(valid)} valid rows; need {args.n}")

    # Pick n diverse (writer, char) pairs.
    writers_seen = set()
    picks = []
    random.shuffle(valid)
    for r in valid:
        wid = str(r.get("writer_id", ""))
        if wid in writers_seen and len(writers_seen) < 12:
            continue
        writers_seen.add(wid)
        picks.append(r)
        if len(picks) >= args.n:
            break
    print(f"[08-sb-sample] picked {len(picks)} pairs from {len(writers_seen)} distinct writers")

    source_font = args.source_font
    cells = []
    for i, r in enumerate(picks):
        char = str(r["char_id"])
        # Source: TTF kai render
        src_arr = render_glyph(
            ttf_path=str(args.fonts_root / source_font / f"{source_font}.ttf"),
            char=char, image_size=args.image_size, font_size_ratio=0.85,
        )
        src_t = _array_to_tensor(src_arr)
        content = src_t.unsqueeze(0).to(device)

        # GT: ernantang real image
        gt_arr = _load_pil_resize(str(r["image_path"]), args.image_size)
        gt_t = (torch.from_numpy(gt_arr).float() / 127.5 - 1.0).unsqueeze(0)

        writer_id = torch.tensor([int(r["writer_label_id"])], dtype=torch.long, device=device)
        char_id = torch.tensor([int(r["char_label_id"])], dtype=torch.long, device=device)
        script_id = torch.tensor([int(r.get("script_label_id", 0))], dtype=torch.long, device=device)

        gen = sample_ddim(
            model=model, diffusion=diffusion,
            content=content,
            writer_id=writer_id, script_id=script_id, char_id=char_id,
            cfg_scale=args.cfg_scale, device=device,
        )

        cells.append((_denorm(src_t), gt_arr, _denorm(gen[0]), r))
        print(f"[08-sb-sample] {i+1}/{len(picks)} char='{char}' writer='{r['writer_id']}' (wid={int(r['writer_label_id'])})")

    # 12 x 3 grid
    H = args.image_size
    sep = 4
    out_h = H * len(cells) + sep * (len(cells) - 1)
    out_w = H * 3 + sep * 2
    canvas = np.full((out_h, out_w), 255, dtype=np.uint8)
    for row, (src, gt, gen, _r) in enumerate(cells):
        y = row * (H + sep)
        canvas[y : y + H, 0:H] = src
        canvas[y : y + H, H + sep : H + sep + H] = gt
        canvas[y : y + H, 2 * (H + sep) : 2 * (H + sep) + H] = gen

    args.output.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(canvas, mode="L").save(args.output)
    print(f"[08-sb-sample] wrote {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
