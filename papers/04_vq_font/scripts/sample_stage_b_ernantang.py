"""04 VQ-Font Stage B sample on real ernantang writers.

Pick ``n`` (writer, target-char, ref-char) triples from the enriched manifest.
For each:
  * source  = TTF kai render of target char  (initial_glyph)
  * ref     = a DIFFERENT char by the same writer (ernantang PNG, used as ref_glyph)
  * gen     = sample_vq_font(initial_glyph, ref_glyphs, structure_id)
  * gt      = actual ernantang PNG of (writer, target_char)

Writes a 4-column grid: [source | ref | gen | gt].

VQ-Font Stage 1+ checkpoint layout (per ``train.py:_save_transformer_ckpt``):
  * ``ckpt.pt``         -> ``{"transformer": ..., "vqgan": ...}`` (state dicts)
  * ``ckpt.pt.cfg.json`` -> ``{"vqgan": {...}, "transformer": {...}}``

We rebuild ``VQFontConfig`` from the sidecar JSON if present, otherwise fall
back to ``blob["cfg"]`` (older format) and finally the default model.yaml.
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
import yaml
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vq_font.model import (
    TransformerConfig,
    VQFontConfig,
    VQGANConfig,
    build_vq_font,
)
from vq_font.sample import sample_vq_font
from paper_reimpl_shared.data.ttf_pair_dataset import _ttf_path_for, render_glyph


def _denorm(t: torch.Tensor) -> np.ndarray:
    """[-1, 1] float tensor -> uint8 [H, W] numpy."""
    t = t.detach().cpu().clamp(-1, 1)
    if t.dim() == 3 and t.shape[0] >= 1:
        t = t[0:1]
    return ((t + 1.0) * 127.5).round().to(torch.uint8).squeeze(0).numpy()


def _load_pil_resize(path: str, size: int) -> np.ndarray:
    img = Image.open(path).convert("L").resize((size, size), Image.BILINEAR)
    return np.array(img, dtype=np.uint8)


def _arr_to_norm_tensor(arr: np.ndarray) -> torch.Tensor:
    """uint8 [H, W] -> float [-1, 1] tensor [H, W]."""
    return torch.from_numpy(arr).float() / 127.5 - 1.0


def _vqgan_cfg_from_dict(d: dict) -> VQGANConfig:
    return VQGANConfig(
        image_size=int(d.get("image_size", 128)),
        in_channels=int(d.get("in_channels", 1)),
        base_channels=int(d.get("base_channels", 32)),
        channel_mult=tuple(d.get("channel_mult", [1, 1, 2, 4])),
        z_channels=int(d.get("z_channels", 256)),
        embed_dim=int(d.get("embed_dim", 256)),
        num_embeddings=int(d.get("num_embeddings", 1024)),
        commitment_weight=float(d.get("commitment_weight", 0.25)),
        num_res_blocks=int(d.get("num_res_blocks", 3)),
        dropout=float(d.get("dropout", 0.0)),
    )


def _transformer_cfg_from_dict(d: dict) -> TransformerConfig:
    return TransformerConfig(
        image_size=int(d.get("image_size", 128)),
        latent_resolution=int(d.get("latent_resolution", 16)),
        embed_dim=int(d.get("embed_dim", 256)),
        num_blocks=int(d.get("num_blocks", 15)),
        num_heads=int(d.get("num_heads", 8)),
        mlp_ratio=float(d.get("mlp_ratio", 2.0)),
        dropout=float(d.get("dropout", 0.0)),
        num_refs=int(d.get("num_refs", 3)),
        codebook_size=int(d.get("codebook_size", 1024)),
        num_structures=int(d.get("num_structures", 13)),
    )


def _resolve_cfg(ckpt_path: Path, blob: dict) -> VQFontConfig:
    """Try sidecar JSON, then blob['cfg'], then default model.yaml."""
    sidecar = ckpt_path.with_suffix(ckpt_path.suffix + ".cfg.json")
    if sidecar.exists():
        print(f"[04-sb-sample] using sidecar cfg {sidecar}")
        raw = json.loads(sidecar.read_text())
        return VQFontConfig(
            vqgan=_vqgan_cfg_from_dict(raw.get("vqgan", {})),
            transformer=_transformer_cfg_from_dict(raw.get("transformer", {})),
        )
    cfg_blob = blob.get("cfg") if isinstance(blob, dict) else None
    if isinstance(cfg_blob, dict) and "vqgan" in cfg_blob and "transformer" in cfg_blob:
        print("[04-sb-sample] using cfg embedded in ckpt blob")
        return VQFontConfig(
            vqgan=_vqgan_cfg_from_dict(cfg_blob["vqgan"]),
            transformer=_transformer_cfg_from_dict(cfg_blob["transformer"]),
        )
    fallback = ROOT / "src" / "vq_font" / "configs" / "model.yaml"
    print(f"[04-sb-sample] no sidecar/cfg found; falling back to {fallback}")
    raw = yaml.safe_load(fallback.read_text())
    return VQFontConfig(
        vqgan=_vqgan_cfg_from_dict(raw.get("vqgan", {})),
        transformer=_transformer_cfg_from_dict(raw.get("transformer", {})),
    )


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True, type=Path)
    p.add_argument("--manifest", required=True, type=Path,
                   help="Enriched manifest JSONL (must have image_path + writer_id + char_id)")
    p.add_argument("--fonts-root", required=True, type=Path,
                   help="TTF fonts root (for source render)")
    p.add_argument("--source-font", default="lxgw_wenkai_regular",
                   help="Neutral kai font for source/content column")
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--n", type=int, default=12)
    p.add_argument("--device", default="cuda:1")
    p.add_argument("--seed", type=int, default=2026)
    args = p.parse_args()

    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    print(f"[04-sb-sample] loading ckpt {args.ckpt}")
    blob = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = _resolve_cfg(args.ckpt, blob)
    image_size = int(cfg.vqgan.image_size)
    num_refs = int(cfg.transformer.num_refs)
    print(f"[04-sb-sample] cfg.vqgan.image_size={image_size} cfg.transformer.num_refs={num_refs}")

    # Build with freeze_vqgan="none" so we can fully load weights into both
    # halves without requires_grad flips affecting state-dict loading.
    model = build_vq_font(cfg, freeze_vqgan="none")
    tr_state = blob.get("transformer")
    vq_state = blob.get("vqgan")
    if tr_state is None or vq_state is None:
        raise KeyError(
            f"ckpt missing 'transformer' or 'vqgan' state-dicts; keys={list(blob.keys())}"
        )
    miss_t, unexp_t = model.transformer.load_state_dict(tr_state, strict=False)
    miss_v, unexp_v = model.vqgan.load_state_dict(vq_state, strict=False)
    print(f"[04-sb-sample] transformer missing={len(miss_t)} unexpected={len(unexp_t)}")
    print(f"[04-sb-sample] vqgan       missing={len(miss_v)} unexpected={len(unexp_v)}")
    model.to(device).eval()

    print(f"[04-sb-sample] reading enriched manifest {args.manifest}")
    rows = []
    with args.manifest.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    valid = [r for r in rows if Path(r["image_path"]).exists()]
    print(f"[04-sb-sample] {len(valid)}/{len(rows)} rows have image on disk")

    # Group by writer so we can pick a different-char ref from same writer.
    by_writer: dict[str, list[dict]] = defaultdict(list)
    for r in valid:
        by_writer[str(r.get("writer_id", ""))].append(r)

    writers = [w for w, rs in by_writer.items() if len(rs) >= 2]
    random.shuffle(writers)
    picks: list[tuple[dict, dict]] = []
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
    print(f"[04-sb-sample] picked {len(picks)} (writer, target-char, ref-char) triples")

    source_ttf = _ttf_path_for(args.fonts_root, args.source_font)
    print(f"[04-sb-sample] source font resolved to {source_ttf}")

    cells = []
    with torch.no_grad():
        for i, (tgt, ref) in enumerate(picks):
            char = str(tgt["char_id"])
            # Source = TTF kai render of target char
            src_arr = render_glyph(
                ttf_path=source_ttf,
                char=char,
                image_size=image_size,
                font_size_ratio=0.85,
            )
            src_t = _arr_to_norm_tensor(src_arr)  # [H, W]

            # Ref = another char by same writer (ernantang PNG)
            ref_arr = _load_pil_resize(str(ref["image_path"]), image_size)
            ref_t = _arr_to_norm_tensor(ref_arr)  # [H, W]

            # GT = actual ernantang PNG of (target_writer, target_char)
            gt_arr = _load_pil_resize(str(tgt["image_path"]), image_size)

            # Pack model inputs.
            # initial_glyph: [1, 1, H, W]
            initial_glyph = src_t.unsqueeze(0).unsqueeze(0).to(device)
            # ref_glyphs: [1, R, 1, H, W] — repeat single ref to fill R slots
            single_ref = ref_t.unsqueeze(0).unsqueeze(0).to(device)  # [1, 1, H, W]
            ref_glyphs = single_ref.unsqueeze(1).expand(1, num_refs, 1, image_size, image_size).contiguous()
            # Train.py defaults to structure_id=1 (atomic/full-map) when missing
            # from batch — must match at inference or VQ-codes degenerate.
            structure_id = torch.ones(1, dtype=torch.long, device=device)

            gen = sample_vq_font(
                model=model,
                initial_glyph=initial_glyph,
                ref_glyphs=ref_glyphs,
                structure_id=structure_id,
                mode="argmax",
            )

            cells.append((src_arr, ref_arr, _denorm(gen[0]), gt_arr))
            print(f"[04-sb-sample] {i+1}/{len(picks)} char='{char}' writer='{tgt['writer_id']}'")

    # 4-column grid: source | ref | gen | gt
    H = image_size
    sep = 4
    out_h = H * len(cells) + sep * (len(cells) - 1)
    out_w = H * 4 + sep * 3
    canvas = np.full((out_h, out_w), 255, dtype=np.uint8)
    for row, (src, ref_, gen, gt) in enumerate(cells):
        y = row * (H + sep)
        for col, col_arr in enumerate((src, ref_, gen, gt)):
            x = col * (H + sep)
            canvas[y : y + H, x : x + H] = col_arr

    args.output.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(canvas, mode="L").save(args.output)
    print(f"[04-sb-sample] wrote {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
