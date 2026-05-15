"""02 HFH-Font Stage B sample on real ernantang writers.

Picks ``n`` (writer, target_char, ref_char) triples from the enriched manifest.
For each:
  * source content = TTF kai render of target char
  * style ref      = a DIFFERENT char by the same writer (real ernantang PNG)
  * gen            = sample_latents → decode_samples
  * gt             = the actual ernantang PNG of (writer, target_char)

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

from hfh_font.model import ModelConfig, build_model
from hfh_font.sample import decode_samples, sample_latents
from paper_reimpl_shared.data.ttf_pair_dataset import render_glyph, _ttf_path_for
from paper_reimpl_shared.diffusion.gaussian import GaussianDiffusion


def _denorm(t: torch.Tensor) -> np.ndarray:
    t = t.detach().cpu().clamp(-1, 1)
    if t.dim() == 3 and t.shape[0] >= 1:
        t = t[0:1]
    return ((t + 1.0) * 127.5).round().to(torch.uint8).squeeze(0).numpy()


def _load_pil(path: str, size: int) -> np.ndarray:
    img = Image.open(path).convert("L").resize((size, size), Image.BILINEAR)
    return np.array(img, dtype=np.uint8)


def _img_to_tensor(arr: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(arr).float() / 127.5 - 1.0  # [-1, 1], shape [H, W]


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
    p.add_argument("--cfg-scale", type=float, default=2.0)
    p.add_argument("--n-refs", type=int, default=1)
    args = p.parse_args()

    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    print(f"[02-sb-sample] loading ckpt {args.ckpt}")
    blob = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = ModelConfig.from_dict(blob["cfg"]) if isinstance(blob["cfg"], dict) else blob["cfg"]
    model = build_model(cfg).to(device).eval()
    model.load_state_dict(blob["model"], strict=False)

    diffusion = GaussianDiffusion(
        timesteps=cfg.diffusion_timesteps, beta_start=1e-4, beta_end=2e-2,
        beta_schedule="linear", prediction_target=cfg.diffusion_target, device=device,
    )

    image_size = int(cfg.image_size)

    print(f"[02-sb-sample] reading enriched manifest {args.manifest}")
    rows = []
    with args.manifest.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    valid = [r for r in rows if Path(r["image_path"]).exists()]
    print(f"[02-sb-sample] {len(valid)}/{len(rows)} rows have image on disk")

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
        raise ValueError(f"Only {len(picks)} (target, ref) pairs; need {args.n}")

    source_ttf = _ttf_path_for(args.fonts_root, args.source_font)
    print(f"[02-sb-sample] source font {source_ttf}")

    cells = []
    for i, (tgt, ref) in enumerate(picks):
        char = str(tgt["char_id"])
        src_arr = render_glyph(
            ttf_path=source_ttf, char=char,
            image_size=image_size, font_size_ratio=0.85,
        )
        ref_arr = _load_pil(str(ref["image_path"]), image_size)
        gt_arr = _load_pil(str(tgt["image_path"]), image_size)

        # Hand-build a batch dict matching the train-time collate output.
        src_t = _img_to_tensor(src_arr).unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
        if int(cfg.content_channels) > 1:
            src_t = src_t.expand(1, int(cfg.content_channels), image_size, image_size).contiguous()
        gt_t = _img_to_tensor(gt_arr).unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
        ref_t = _img_to_tensor(ref_arr).unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
        # ref_images shape expected: [B, N, C, H, W]
        ref_images = ref_t.unsqueeze(1)  # [1, 1, 1, H, W]
        if args.n_refs > 1:
            ref_images = ref_images.expand(1, args.n_refs, 1, image_size, image_size).contiguous()
        ref_valid = torch.ones(1, args.n_refs, dtype=torch.bool)

        batch = {
            "image": gt_t,
            "content": src_t,
            "ref_images": ref_images,
            "ref_valid": ref_valid,
            "char_id": torch.tensor([int(tgt["char_label_id"])], dtype=torch.long),
            "writer_id": torch.tensor([int(tgt["writer_label_id"])], dtype=torch.long),
            "script_id": torch.tensor([int(tgt.get("script_label_id", 0))], dtype=torch.long),
            "style_family_id": torch.tensor([int(tgt.get("style_family_label_id", tgt.get("unit_label_id", 0)))], dtype=torch.long),
            "unit_id": torch.tensor([int(tgt.get("unit_label_id", 0))], dtype=torch.long),
        }

        latents = sample_latents(
            model, diffusion, batch=batch,
            sampler="ddim", cfg_scale=args.cfg_scale, device=device,
        )
        decoded = decode_samples(model, latents)  # [1, 1, H, W]

        cells.append((src_arr, ref_arr, _denorm(decoded[0]), gt_arr))
        print(f"[02-sb-sample] {i+1}/{len(picks)} writer='{tgt['writer_id']}' "
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
    print(f"[02-sb-sample] wrote {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
