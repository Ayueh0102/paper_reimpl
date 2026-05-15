#!/usr/bin/env python3
"""Compose all paper-reimpl 二南堂 sample grids into one side-by-side PNG.

Each per-paper grid has 12 rows. This script:
  - Parses each grid into rows (assumed 4-pixel separator between rows).
  - Resizes each row to a uniform `row_height` for fair comparison
    across papers that trained at different image sizes (64/80/128px).
  - Adds a paper header + per-column headers above each panel.
  - Composes the panels horizontally with a fixed gap.

Run after `fetch_weekend_results.sh` has scp'd the per-paper PNGs into
`--samples-dir` (default /tmp).
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from typing import List, Optional

from PIL import Image, ImageDraw, ImageFont


# --- Font ---------------------------------------------------------------

CJK_FONT_CANDIDATES = [
    "/System/Library/Fonts/PingFang.ttc",
    "/System/Library/Fonts/STHeiti Medium.ttc",
    "/Library/Fonts/Arial Unicode.ttf",
]


def load_font(size: int) -> ImageFont.ImageFont:
    for path in CJK_FONT_CANDIDATES:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except Exception:
                continue
    return ImageFont.load_default()


# --- Panel definition ---------------------------------------------------


@dataclass
class PaperPanel:
    key: str                # short id, e.g. "01"
    filename: str           # expected basename in samples_dir
    title: str              # header line, e.g. "01 FontDiffuser  SB long"
    cell_px: int            # native cell size (px) in the source grid
    col_headers: List[str]  # column labels above each column
    n_cols: int             # column count (should equal len(col_headers))
    n_rows: int = 12        # always 12 二南堂 rows
    row_sep_px: int = 4     # between-row separator (4 px)


PANELS: List[PaperPanel] = [
    PaperPanel(
        key="01",
        filename="01_sb_ernantang.png",
        title="01 FontDiffuser  SB long",
        cell_px=128,
        col_headers=["TTF", "writer-ref", "gen", "GT"],
        n_cols=4,
    ),
    PaperPanel(
        key="02",
        filename="02_sb_ernantang.png",
        title="02 HFH-Font  SB long",
        cell_px=128,
        col_headers=["TTF", "writer-ref", "gen", "GT"],
        n_cols=4,
    ),
    PaperPanel(
        key="04",
        filename="04_sb_ernantang.png",
        title="04 VQ-Font  SB long",
        cell_px=128,
        col_headers=["TTF", "writer-ref", "gen", "GT"],
        n_cols=4,
    ),
    PaperPanel(
        key="05",
        filename="05_sb_ernantang.png",
        title="05 QT-Font  SB 64px",
        cell_px=64,
        col_headers=["TTF", "writer-ref", "gen", "GT"],
        n_cols=4,
    ),
    PaperPanel(
        key="08",
        filename="08_sb_50k_pinn0_sample.png",
        title="08 DP-Font  SB 50k (PINN=0)",
        cell_px=80,
        col_headers=["TTF", "GT", "gen"],
        n_cols=3,
    ),
    PaperPanel(
        key="08x",
        filename="08_sb_extra_long_sample.png",
        title="08 DP-Font  SB extra-long 200k (PINN=0)",
        cell_px=80,
        col_headers=["TTF", "GT", "gen"],
        n_cols=3,
    ),
]


# --- Row parsing + resizing --------------------------------------------


def parse_grid_rows(
    img: Image.Image, panel: PaperPanel
) -> List[Image.Image]:
    """Slice the source grid into 12 row strips.

    We assume the grid is laid out as:
        cell (cell_px tall) + 4 px sep + cell + 4 px sep + ... x n_rows
    and that horizontally it contains `n_cols` cells (possibly with
    separators we leave in-place — we don't care about column geometry).

    The grid may have small top/bottom padding from the original maker
    (e.g. when train_*.py adds a title strip). We be tolerant: if the
    computed total height doesn't match the image, we fall back to
    splitting the image into 12 equal vertical strips.
    """
    w, h = img.size
    expected_h = panel.n_rows * panel.cell_px + (panel.n_rows - 1) * panel.row_sep_px

    rows: List[Image.Image] = []

    if abs(h - expected_h) <= panel.row_sep_px * 2:
        # Clean grid: exact cell_px + 4px sep layout.
        y = 0
        for i in range(panel.n_rows):
            rows.append(img.crop((0, y, w, y + panel.cell_px)))
            y += panel.cell_px + panel.row_sep_px
        return rows

    # Fallback: split into 12 equal vertical strips. Tolerates header
    # banners or padding the maker might have added.
    strip_h = h // panel.n_rows
    for i in range(panel.n_rows):
        y0 = i * strip_h
        y1 = (i + 1) * strip_h if i < panel.n_rows - 1 else h
        rows.append(img.crop((0, y0, w, y1)))
    return rows


def resize_row(row: Image.Image, target_h: int) -> Image.Image:
    w, h = row.size
    if h == target_h:
        return row
    scale = target_h / float(h)
    new_w = max(1, int(round(w * scale)))
    return row.resize((new_w, target_h), Image.LANCZOS)


# --- Panel rendering ----------------------------------------------------


HEADER_H = 28
COL_HEADER_H = 22
ROW_GAP = 4
PANEL_GAP = 40
PAD = 8


def render_panel(
    img: Image.Image, panel: PaperPanel, row_height: int
) -> Image.Image:
    rows = parse_grid_rows(img, panel)
    rows = [resize_row(r, row_height) for r in rows]

    panel_w = max(r.size[0] for r in rows)
    total_h = (
        HEADER_H
        + COL_HEADER_H
        + len(rows) * row_height
        + (len(rows) - 1) * ROW_GAP
        + PAD * 2
    )

    out = Image.new("RGB", (panel_w + PAD * 2, total_h), "white")
    draw = ImageDraw.Draw(out)

    title_font = load_font(16)
    col_font = load_font(13)

    # Title.
    draw.text((PAD, PAD), panel.title, fill="black", font=title_font)

    # Column headers — placed evenly across panel_w based on n_cols.
    col_w = panel_w / panel.n_cols
    y_colh = PAD + HEADER_H
    for i, name in enumerate(panel.col_headers):
        cx = PAD + int(i * col_w + col_w / 2)
        # rough centering: PIL text doesn't auto-center; use anchor if avail.
        try:
            draw.text((cx, y_colh), name, fill="#444", font=col_font, anchor="mt")
        except TypeError:
            # very old Pillow without anchor support
            draw.text((cx - 20, y_colh), name, fill="#444", font=col_font)

    # Paste rows.
    y = PAD + HEADER_H + COL_HEADER_H
    for r in rows:
        out.paste(r, (PAD, y))
        y += row_height + ROW_GAP

    # Light border around the panel for visual separation.
    draw.rectangle(
        [(0, 0), (out.size[0] - 1, out.size[1] - 1)], outline="#cccccc"
    )
    return out


# --- Main compose -------------------------------------------------------


def compose(samples_dir: str, output: str, row_height: int) -> None:
    present: List[tuple[PaperPanel, str]] = []
    missing: List[PaperPanel] = []
    for p in PANELS:
        path = os.path.join(samples_dir, p.filename)
        if os.path.exists(path):
            present.append((p, path))
        else:
            missing.append(p)

    print(f"[compose] samples_dir = {samples_dir}")
    print(f"[compose] present ({len(present)}):")
    for p, path in present:
        print(f"           {p.key}  {p.filename}")
    print(f"[compose] missing ({len(missing)}):")
    for p in missing:
        print(f"           {p.key}  {p.filename}")

    if not present:
        print("[compose] no panels present — nothing to compose. exiting.")
        sys.exit(1)

    panels_rendered: List[Image.Image] = []
    for p, path in present:
        try:
            img = Image.open(path).convert("RGB")
        except Exception as exc:
            print(f"[compose] WARN: failed to open {path}: {exc}")
            continue
        try:
            rendered = render_panel(img, p, row_height)
        except Exception as exc:
            print(f"[compose] WARN: failed to render {p.key}: {exc}")
            continue
        panels_rendered.append(rendered)

    if not panels_rendered:
        print("[compose] no panels rendered — exiting.")
        sys.exit(1)

    total_w = sum(im.size[0] for im in panels_rendered) + PANEL_GAP * (
        len(panels_rendered) - 1
    )
    max_h = max(im.size[1] for im in panels_rendered)

    canvas = Image.new("RGB", (total_w, max_h), "white")
    x = 0
    for im in panels_rendered:
        canvas.paste(im, (x, 0))
        x += im.size[0] + PANEL_GAP

    os.makedirs(os.path.dirname(os.path.abspath(output)) or ".", exist_ok=True)
    canvas.save(output)
    print(f"[compose] wrote {output}  ({canvas.size[0]}x{canvas.size[1]})")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--samples-dir", default="/tmp")
    ap.add_argument("--output", default="/tmp/paper_comparison.png")
    ap.add_argument(
        "--row-height",
        type=int,
        default=100,
        help="uniform height (px) to resize each row to before composing",
    )
    args = ap.parse_args()
    compose(args.samples_dir, args.output, args.row_height)


if __name__ == "__main__":
    main()
