"""05 QT-Font Stage B sample on real ernantang writers.

Picks 12 (writer, target_char, ref_char) triples from the enriched manifest
and produces a 12x4 grid PNG:

    col 0: source TTF kai render of the TARGET char (neutral kai)
    col 1: real ernantang PNG of a DIFFERENT char by the SAME writer (ref)
    col 2: QT-Font generation conditioned on source content + writer ref
    col 3: real ernantang PNG of the TARGET (writer, char) (GT)

QT-Font is a 3-state categorical D3PM over a sparse quadtree
({bg, contour, skeleton}). We re-use `qt_font.sample.sample_image`, which:

  1. Builds an octree topology from the content (source TTF) image.
  2. Initialises x_T ~ Uniform({0,1,2}) over leaf nodes.
  3. Runs the Gumbel-max reverse loop with stride `gap` over T=1000
     (gap=20 → 50 effective denoising steps; gap=50 → 20 steps).
  4. Decodes leaf labels back to a (B, 1, side, side) image in [-1, +1]
     where bg=-1, contour=0, skeleton=+1.

`side = 1 << cfg.depth` (64 px for the depth=6 Stage A ckpt).

This script falls back to a contour-octree placeholder if the full D3PM
reverse process fails (e.g. an axis-label decode that prints all bg). We
still emit a valid 4-col grid so downstream eyeball QA is unblocked.
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

from qt_font.model import D3PMUniform, QTFontConfig, build_qt_font  # noqa: E402
from qt_font.octree import build_octree_from_image, render_label_image  # noqa: E402
from qt_font.sample import sample_image  # noqa: E402
from paper_reimpl_shared.data.ttf_pair_dataset import (  # noqa: E402
    _array_to_tensor,
    _ttf_path_for,
    render_glyph,
)


# --------------------------------------------------------------------------- #
# Helpers                                                                      #
# --------------------------------------------------------------------------- #


def _denorm(t: torch.Tensor) -> np.ndarray:
    """Float tensor in [-1, 1] -> uint8 [0, 255] H×W array."""
    t = t.detach().cpu().clamp(-1, 1)
    if t.dim() == 3 and t.shape[0] >= 1:
        t = t[0:1]
    return ((t + 1.0) * 127.5).round().to(torch.uint8).squeeze(0).numpy()


def _load_pil_resize(path: str, size: int) -> np.ndarray:
    """PNG path -> uint8 grayscale (size, size)."""
    img = Image.open(path).convert("L").resize((size, size), Image.BILINEAR)
    return np.array(img, dtype=np.uint8)


def _png_to_tensor(path: str, size: int) -> torch.Tensor:
    """PNG path -> float tensor [1, size, size] in [-1, 1] (white=+1, ink=-1)."""
    arr = _load_pil_resize(path, size)
    f = arr.astype(np.float32) / 127.5 - 1.0
    return torch.from_numpy(f).unsqueeze(0)


def _render_octree_contour_placeholder(
    content: torch.Tensor, full_depth: int, depth: int
) -> torch.Tensor:
    """Fallback when the D3PM reverse fails: render the source content's
    octree contour back as a (1, 1, side, side) image."""
    octree = build_octree_from_image(content, full_depth=full_depth, depth=depth).to(
        content.device
    )
    label_img = render_label_image(octree, use_leaf_label=True)  # (B, side, side) long
    side = 1 << depth
    mapping = torch.tensor([-1.0, 0.0, 1.0], device=content.device)
    img = mapping[label_img].unsqueeze(1)  # (B, 1, side, side)
    return img


# --------------------------------------------------------------------------- #
# Main                                                                         #
# --------------------------------------------------------------------------- #


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True, type=Path)
    p.add_argument(
        "--manifest",
        required=True,
        type=Path,
        help="Enriched manifest JSONL (must have image_path + writer_id + char_id)",
    )
    p.add_argument(
        "--fonts-root",
        required=True,
        type=Path,
        help="TTF fonts root (for source kai render)",
    )
    p.add_argument(
        "--source-font",
        default="lxgw_wenkai_regular",
        help="Neutral kai font id (subdir name under --fonts-root) for source column",
    )
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--n", type=int, default=12)
    p.add_argument("--device", default="cuda:1")
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument(
        "--ddim-steps",
        type=int,
        default=50,
        help="Effective number of D3PM reverse steps. gap = max(1, T // ddim_steps).",
    )
    args = p.parse_args()

    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    # ---- Load checkpoint -------------------------------------------------- #
    print(f"[05-sb-sample] loading ckpt {args.ckpt}")
    blob = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg_dict = blob.get("cfg") if isinstance(blob, dict) else None
    if cfg_dict is None:
        raise ValueError(f"ckpt {args.ckpt} has no 'cfg' entry; cannot rebuild model")
    cfg = cfg_dict if isinstance(cfg_dict, QTFontConfig) else QTFontConfig(**cfg_dict)
    model = build_qt_font(cfg)
    state = blob["model"] if isinstance(blob, dict) and "model" in blob else blob
    miss, unexp = model.load_state_dict(state, strict=False)
    print(f"[05-sb-sample] missing={len(miss)} unexpected={len(unexp)}")
    model.to(device).eval()

    diffusion = D3PMUniform(
        n_states=cfg.n_states,
        timesteps=cfg.timesteps,
        schedule=cfg.schedule,
        beta_start=0.02,
        beta_end=1.0,
    ).to(device)

    image_size = 1 << cfg.depth  # 64 for depth=6
    gap = max(1, cfg.timesteps // max(1, args.ddim_steps))
    print(
        f"[05-sb-sample] cfg.depth={cfg.depth} -> image_size={image_size}; "
        f"T={cfg.timesteps} gap={gap} (~{cfg.timesteps // gap} reverse steps)"
    )

    # ---- Read manifest ---------------------------------------------------- #
    print(f"[05-sb-sample] reading enriched manifest {args.manifest}")
    rows = []
    with args.manifest.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    print(f"[05-sb-sample] manifest has {len(rows)} rows")

    # Only keep rows whose backing image actually exists.
    valid = [r for r in rows if Path(r["image_path"]).exists()]
    print(f"[05-sb-sample] {len(valid)} rows have image on disk")
    if len(valid) < args.n:
        raise ValueError(f"Only {len(valid)} valid rows; need at least {args.n}")

    # Group by writer so we can pull a same-writer ref for each target.
    by_writer: dict[str, list[dict]] = {}
    for r in valid:
        wid = str(r.get("writer_id", ""))
        by_writer.setdefault(wid, []).append(r)

    eligible_writers = [w for w, rs in by_writer.items() if len(rs) >= 2]
    print(
        f"[05-sb-sample] {len(eligible_writers)}/{len(by_writer)} writers have >=2 chars "
        f"(needed for distinct ref / target)"
    )

    # Pick n diverse (writer, target, ref) triples — prefer distinct writers,
    # but allow re-use if we run out.
    random.shuffle(eligible_writers)
    picks: list[tuple[dict, dict]] = []  # (target_row, ref_row)
    for wid in eligible_writers:
        rs = list(by_writer[wid])
        random.shuffle(rs)
        target = rs[0]
        # Try to pick a ref char that is different from the target char.
        ref = next(
            (r for r in rs[1:] if str(r.get("char_id")) != str(target.get("char_id"))),
            rs[1],
        )
        picks.append((target, ref))
        if len(picks) >= args.n:
            break

    # Top up by repeating writers if necessary.
    while len(picks) < args.n and eligible_writers:
        wid = random.choice(eligible_writers)
        rs = list(by_writer[wid])
        random.shuffle(rs)
        picks.append((rs[0], rs[1]))

    if len(picks) < args.n:
        raise ValueError(
            f"Could only assemble {len(picks)} triples; need {args.n}. "
            "Manifest does not have enough writers with >=2 chars."
        )
    print(
        f"[05-sb-sample] assembled {len(picks)} (writer, target, ref) triples "
        f"from {len({str(t.get('writer_id')) for t, _ in picks})} distinct writers"
    )

    # ---- Resolve source font --------------------------------------------- #
    source_ttf = _ttf_path_for(args.fonts_root, args.source_font)
    print(f"[05-sb-sample] source font resolved to {source_ttf}")

    # ---- Per-row generation ---------------------------------------------- #
    cells: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]] = []
    used_placeholder = False
    for i, (target, ref) in enumerate(picks):
        target_char = str(target["char_id"])
        ref_char = str(ref["char_id"])
        wid = str(target.get("writer_id", ""))

        # col 0: source TTF kai render of target char
        src_arr = render_glyph(
            ttf_path=source_ttf,
            char=target_char,
            image_size=image_size,
            font_size_ratio=0.85,
        )
        src_t = _array_to_tensor(src_arr)  # (1, H, W) in [-1, 1]
        content = src_t.unsqueeze(0).to(device)  # (1, 1, H, W)

        # col 1: real ernantang ref PNG (different char, same writer)
        ref_arr = _load_pil_resize(str(ref["image_path"]), image_size)
        ref_t = _png_to_tensor(str(ref["image_path"]), image_size).to(device)  # (1, H, W)
        # sample_image expects refs as (B, R, C, H, W).
        refs = ref_t.unsqueeze(0).unsqueeze(0)  # (1, 1, 1, H, W)

        # col 3: real ernantang GT (target writer + target char)
        gt_arr = _load_pil_resize(str(target["image_path"]), image_size)

        # col 2: QT-Font generation
        try:
            with torch.no_grad():
                gen = sample_image(
                    model,
                    diffusion,
                    batch_size=1,
                    content=content,
                    refs=refs,
                    gap=gap,
                )
            gen_arr = _denorm(gen[0])
        except Exception as e:  # pragma: no cover - defensive fallback
            used_placeholder = True
            print(
                f"[05-sb-sample] WARNING: full D3PM reverse failed at row {i} "
                f"({type(e).__name__}: {e}); falling back to octree-contour placeholder"
            )
            with torch.no_grad():
                gen = _render_octree_contour_placeholder(
                    content, full_depth=cfg.full_depth, depth=cfg.depth
                )
            gen_arr = _denorm(gen[0])

        cells.append((src_arr, ref_arr, gen_arr, gt_arr, {"target": target, "ref": ref}))
        print(
            f"[05-sb-sample] {i+1}/{len(picks)} "
            f"writer='{wid}' target='{target_char}' ref='{ref_char}'"
        )

    if used_placeholder:
        print(
            "[05-sb-sample] WARNING: at least one row used the octree-contour "
            "placeholder rather than full D3PM reverse — eyeball QA accordingly."
        )

    # ---- Compose 12x4 grid ------------------------------------------------ #
    H = image_size
    sep = 4
    n_rows = len(cells)
    n_cols = 4
    out_h = H * n_rows + sep * (n_rows - 1)
    out_w = H * n_cols + sep * (n_cols - 1)
    canvas = np.full((out_h, out_w), 255, dtype=np.uint8)
    for row, (src, ref, gen, gt, _meta) in enumerate(cells):
        y = row * (H + sep)
        for col, arr in enumerate((src, ref, gen, gt)):
            x = col * (H + sep)
            canvas[y : y + H, x : x + H] = arr

    args.output.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(canvas, mode="L").save(args.output)
    print(f"[05-sb-sample] wrote {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
