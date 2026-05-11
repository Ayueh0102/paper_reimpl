# Stage {A|B|C} Results — {NN}_{SHORT}

**Date**: YYYY-MM-DD
**Lab server**: `$LAB_SSH_USER@$LAB_SSH_HOST` (Windows)
**GPU**: cuda:0 (RTX 6000 Ada 48GB) / cuda:1
**Backend**: lab_server

## Config

- Train YAML: `src/{SHORT}/configs/train_stage_{X}.yaml`
- Model YAML: `src/{SHORT}/configs/model.yaml`
- Data YAML: `src/{SHORT}/configs/data_stage_{X}.yaml`
- Init ckpt: `outputs/stage_{X-1}/global_step_N/...` or None

## Training metrics

| Step | loss | wall time |
|---|---|---|
| 1k | ... | ... |
| 5k | ... | ... |
| 10k | ... | ... |
| ... | ... | ... |
| final | ... | ... |

(Embed loss curve PNG if available.)

## Sampling

- Sampler: DDIM / DDPM / ...
- Steps: 50
- CFG scale: 3.0 / N/A
- Sample grid: ![](outputs/stage_{X}/sample_grid.png)

## Comparison to paper

| Metric | Paper reported | Ours |
|---|---|---|
| FID | ... | ... |
| char accuracy | ... | ... |

Notes on gap: ...

## Checkpoint

- Path on server: `D:\Char\ayueh\paper_reimpl\papers\{NN}_{SHORT}\outputs\stage_{X}\global_step_N\ckpt.pt`
- SHA256: ...
- Size: ... MB

## Issues encountered

- ...

## Verdict (DL reviewer fills)

See `dl_review_stage_{X}.md`.
