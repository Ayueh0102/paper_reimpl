# Stage A Shakedown — 01 FontDiffuser

**Goal:** validate the full Phase-3 training pipeline on the lab server
(SSH → uv → CUDA → DataLoader → fwd/bwd → ckpt save) on synthetic data
*before* writing the real TTF cross-font pair dataset.

## Run

| field | value |
|---|---|
| Date | 2026-05-11 |
| Host | WIN-C20DRJGJ4S4 (lab server) |
| GPU | NVIDIA RTX 6000 Ada Generation 48 GB (cuda:0) |
| Driver | CUDA 12.8, 573.42 |
| torch | 2.11.0+cu128 |
| Config | `configs/train_stage_a_shakedown.yaml` + `configs/data_stage_a.yaml` + `configs/model.yaml` |
| Data source | synthetic (SyntheticCalligraphyDataset, `synthetic_length=64`) |
| Steps | 200 |
| Batch size | 16 |
| LR | 1e-4 (AdamW) |
| Grad clip | 1.0 |
| CFG drop | 0.1 |
| SCR weight | 0.0 (Stage A skips contrastive head) |
| Perceptual weight | 0.01 (VGG-16) |
| Offset L1 weight | 0.5 (RSI) |
| Diffusion | T=1000, β∈[1e-4, 2e-2] linear, eps prediction |
| Duration | 51 s |
| Throughput | ~63 samples/sec |
| ckpt | `outputs/stage_a_shakedown/fontdiffuser_last.pt` (117 MB) |
| ckpt sha256 | `4A05A906A225989A126B16E62CF6FAA1C6C3089952730D032D993EAFEB4F18D2` |
| Log | `reports/stage_a_shakedown_log.txt` |

## Loss trajectory

| step | loss_total | loss_simple | loss_perc | loss_offset |
|---:|---:|---:|---:|---:|
| 0   | 2.0614 | 1.1820 | 87.8624 | 0.0015 |
| 20  | 0.6041 | 0.1227 | 48.1294 | 0.0003 |
| 40  | 0.3218 | 0.1468 | 17.4892 | 0.0002 |
| 60  | 0.2218 | 0.1293 | 9.2427  | 0.0002 |
| 80  | 0.2259 | 0.1294 | 9.6418  | 0.0002 |
| 100 | 0.1774 | 0.0754 | 10.1937 | 0.0001 |
| 120 | 0.1701 | 0.0922 | 7.7782  | 0.0001 |
| 140 | 0.1184 | 0.0686 | 4.9800  | 0.0001 |
| 160 | 0.2252 | 0.1203 | 10.4833 | 0.0001 |
| 180 | 0.1022 | 0.0585 | 4.3725  | 0.0001 |

No NaN / Inf. Total loss drops 20× over 200 steps; expected on synthetic
data (model overfits to a fixed 64-sample random pool).

## What this validates

- [x] Lab server SSH path + Taildrop-equivalent scp workflow
- [x] uv venv with `[[tool.uv.index]] pytorch-cu128` resolves CUDA wheel
- [x] PyTorch CUDA build sees both RTX 6000 Ada
- [x] DataLoader on Windows with `num_workers=0` (the worker-spawn path is
      still TODO for real TTF / manifest training)
- [x] FontDiffuser fwd + bwd + AdamW step
- [x] VGG-16 perceptual loss loads from cached weights
- [x] RSI offset L1 contributes
- [x] Checkpoint save resolves to repo-relative `outputs/stage_a_shakedown/`
      (off-by-one in `Path(__file__).parents[3]` fixed → `[4]`)
- [x] `PYTHONUNBUFFERED=1` gives per-step log streaming

## Open items before real Stage A

1. **Real TTF cross-font dataset** — `dataset.py` only has synthetic and
   manifest paths. Stage A as written in the paper pretrains on cross-font
   pairs rendered from 13 free fonts (`data_snapshot/fonts_free/`). Need a
   new `TTFCrossFontPairDataset` that:
   - Iterates over a char vocabulary (e.g. GB2312 first 6763 codepoints).
   - For each (char, sample) pair, randomly picks a *source* font (content
     glyph rendered with that font) and a *target* font (the image we are
     supposed to recover), plus a *reference* font for the one-shot style.
   - Renders glyphs on the fly from `.ttf` via PIL / `freetype-py` or uses
     a pre-rendered cache under `data_snapshot/ttf_renders/<font>/<hex>.png`.
2. **Windows DataLoader workers** — when we wire the real dataset, retry
   `num_workers > 0` with proper `if __name__ == "__main__"` guarding on
   Windows spawn. Failure mode observed (attempt 2): workers hang on
   spawn, main process holds GPU memory at 0% util.
3. **Stage A schedule** — paper trains for hundreds of thousands of steps.
   Calibrate to ~0.5-1.0 of paper headline on RTX 6000 Ada; estimate ~25k
   steps / 0.5-1 GPU-day budget for first real run.

## Bugs fixed during shakedown

- `train.py` `ckpt_dir` base was `Path(__file__).parents[3]` → repo/papers,
  causing nested `repo/papers/papers/01_fontdiffuser/outputs/...`. Bumped
  to `parents[4]` to land at repo root. (commit `e7e2037`)
- `train_stage_a_shakedown.yaml` `max_epochs=1` capped the run at 4 batches
  on `synthetic_length=64 / bs=16`. Set `max_epochs=100` so the trainer
  cycles to hit `max_steps=200`. (commit `e7e2037`)
- `train_stage_a_shakedown.yaml` `num_workers=2` hung the trainer on
  Windows DataLoader worker spawn (12 GB held, 0 % util, 0-byte log for
  5 min). Set `num_workers=0` for the synthetic shakedown. (commit `840b038`)
- `run_01_stage_a_shakedown_gpu0.bat` added `PYTHONUNBUFFERED=1` so log
  flushes per-line instead of buffering until process exit. (commit `840b038`)
