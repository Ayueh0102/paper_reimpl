"""01 FontDiffuser Stage B sample on real ernantang writers.

Pick ``n`` (writer, char) pairs from the enriched manifest. For each:
  * source  = TTF kai render of char (content path)
  * ref     = a DIFFERENT char by the same writer (style ref)
  * gen     = model.sample(content, ref)
  * gt      = actual ernantang PNG of (writer, char) from the manifest

Writes a 4-column grid: [source | ref | gen | gt].
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fontdiffuser.model import FontDiffuserConfig, build_fontdiffuser
from fontdiffuser.sample import sample_ddim
from paper_reimpl_shared.data.ttf_pair_dataset import render_glyph, _ttf_path_for
from paper_reimpl_shared.diffusion.gaussian import GaussianDiffusion


def _denorm(t):
    t = t.detach().cpu().clamp(-1, 1)
    if t.dim() == 3 and t.shape[0] >= 1:
        t = t[0:1]
    return ((t + 1.0) * 127.5).round().to(torch.uint8).squeeze(0).numpy()


def _load_pil_resize(path: str, size: int) -> np.ndarray:
    img = Image.open(path).convert("L").resize((size, size), Image.BILINEAR)
    return np.array(img, dtype=np.uint8)


def _img_to_tensor(arr: np.ndarray) -> torch.Tensor:
    """uint8 [H, W] → float [-1, 1] tensor [1, H, W]."""
    t = torch.from_numpy(arr).float() / 127.5 - 1.0
    return t.unsqueeze(0)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True, type=Path)
    p.add_argument("--manifest", required=True, type=Path)
    p.add_argument("--fonts-root", required=True, type=Path)
    p.add_argument("--source-font", default="lxgw_wenkai_regular")
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--n", type=int, default=12)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--cfg-scale", type=float, default=1.0)
    args = p.parse_args()

    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    print(f"[01-sb-sample] loading ckpt {args.ckpt}")
    blob = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = FontDiffuserConfig(**blob["cfg"]) if isinstance(blob["cfg"], dict) else blob["cfg"]
    model = build_fontdiffuser(cfg).to(device).eval()
    model.load_state_dict(blob["model"], strict=False)

    diffusion = GaussianDiffusion(
        timesteps=1000, beta_start=1e-4, beta_end=2e-2,
        beta_schedule="linear", prediction_target="epsilon", device=device,
    )

    print(f"[01-sb-sample] reading enriched manifest {args.manifest}")
    rows = []
    with args.manifest.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    valid = [r for r in rows if Path(r["image_path"]).exists()]
    print(f"[01-sb-sample] {len(valid)}/{len(rows)} rows have image on disk")

    # Group by writer so we can pick a different-char ref from same writer.
    by_writer: dict[str, list[dict]] = defaultdict(list)
    for r in valid:
        by_writer[str(r.get("writer_id", ""))].append(r)

    writers = [w for w, rs in by_writer.items() if len(rs) >= 2]
    random.shuffle(writers)
    picks = []
    for w in writers:
        if len(picks) >= args.n:
            break
        rs = by_writer[w]
        random.shuffle(rs)
        target = rs[0]
        ref = next((r for r in rs[1:] if str(r["char_id"]) != str(target["char_id"])), None)
        if ref is None:
            continue
        picks.append((target, ref))
    if len(picks) < args.n:
        raise ValueError(f"Only got {len(picks)} (target, ref) pairs; need {args.n}")
    print(f"[01-sb-sample] picked {len(picks)} (writer, target-char, ref-char) triples")

    source_ttf = _ttf_path_for(args.fonts_root, args.source_font)
    print(f"[01-sb-sample] source font {source_ttf}")
    image_size = int(cfg.image_size)

    cells = []
    for i, (tgt, ref) in enumerate(picks):
        char = str(tgt["char_id"])
        # Source = TTF kai of target char (broadcast to content_channels)
        src_arr = render_glyph(
            ttf_path=source_ttf, char=char,
            image_size=image_size, font_size_ratio=0.85,
        )
        src_t = _img_to_tensor(src_arr)  # [1, H, W]
        content = src_t.unsqueeze(0).to(device)  # [1, 1, H, W]
        if int(cfg.content_channels) > 1:
            content = content.expand(1, int(cfg.content_channels), image_size, image_size).contiguous()

        # Ref = another char by same writer (ernantang PNG)
        ref_arr = _load_pil_resize(str(ref["image_path"]), image_size)
        ref_t = _img_to_tensor(ref_arr).unsqueeze(0).to(device)  # [1, 1, H, W]

        # GT = actual ernantang PNG of (target_writer, target_char)
        gt_arr = _load_pil_resize(str(tgt["image_path"]), image_size)

        gen = sample_ddim(
            model=model, diffusion=diffusion,
            content=content, ref_image=ref_t,
            cfg_scale=args.cfg_scale, device=device,
        )

        cells.append((src_arr, ref_arr, _denorm(gen[0]), gt_arr))
        print(f"[01-sb-sample] {i+1}/{len(picks)} writer='{tgt['writer_id']}' "
              f"target='{char}' ref='{ref['char_id']}'")

    H = image_size
    sep = 4
    out_h = H * len(cells) + sep * (len(cells) - 1)
    out_w = H * 4 + sep * 3
    canvas = np.full((out_h, out_w), 255, dtype=np.uint8)
    for row, (src, ref, gen, gt) in enumerate(cells):
        y = row * (H + sep)
        for col, col_arr in enumerate((src, ref, gen, gt)):
            x = col * (H + sep)
            canvas[y : y + H, x : x + H] = col_arr

    args.output.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(canvas, mode="L").save(args.output)
    print(f"[01-sb-sample] wrote {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
