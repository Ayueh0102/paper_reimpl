# Code Review: hfh_font Phase 1 Gate 1

**Date**: 2026-05-11
**Reviewer**: automated (Claude Sonnet 4.6)
**Scope**: `src/hfh_font/` — `__init__.py`, `dataset.py`, `model.py`, `sample.py`, `train.py`
**Verdict**: APPROVE with nits (no CRITICAL or HIGH issues found)

---

## Security — Public-Repo Safety

**PASS.** Exhaustive grep for `ptri78322626`, `Masg844ssu`, `10.102.10.212`, `100.91.52.77`, `qaz741945`, `D:\Char\ayueh`, `D:/Char/ayueh`, `ayueh8812`, `ptri.org` across all files under `papers/02_hfh_font/` returned zero matches. No hardcoded credentials, IPs, or personal paths found.

No SQL queries, shell subprocess calls, eval/exec, pickle deserialization, or yaml.load (unsafe) patterns present.

---

## Backend Portability

**PASS.** All path construction uses `pathlib.Path`. The `ckpt_dir` resolution in `train.py:134-137` correctly uses `Path(__file__).resolve().parents[3]` to anchor relative paths to the repo root, which is identical on Mac and PC/Vast.ai. No Windows-style absolute literals (`C:\`, `D:\`, `E:\`) appear in source.

---

## Import Convention

**PASS.** All cross-package references correctly use `paper_reimpl_shared.*` (absolute package import). Zero instances of `from ..shared` or `import ..shared` found.

---

## `train.main()` Signature

**PASS.** Signature matches the specified contract:
```python
def main(args, *, data_cfg: dict[str, Any], model_cfg: dict[str, Any], train_cfg: dict[str, Any], paths: BackendPaths) -> int:
```
Returns `int` (0 = OK, 2 = non-finite loss). Keyword-only enforced via `*`.

---

## Reproducibility

**PARTIAL PASS — nit.**

`set_seed()` seeds `random`, `numpy`, `torch`, and `torch.cuda`. However:
- `torch.backends.cudnn.deterministic = True` and `torch.backends.cudnn.benchmark = False` are not set. For full cross-run reproducibility on CUDA these are required; omitting them is acceptable for Phase 1 smoke runs (num_workers=0 avoids the DataLoader seeding problem), but should be noted.
- No `worker_init_fn` / `generator` on `DataLoader`. Safe since `num_workers=0` is hardcoded throughout, but the comment on line 124 says "smoke / dry-run" — misleading because `num_workers` is `0` even for full training runs.

---

## Ruff (lint)

**PASS.** `ruff check src/hfh_font/` via the project venv returned: `All checks passed!`

---

## Black / mypy

Black and mypy are not installed in `.venv` (only `pytest` + `ruff` in `[dev]` extras per `pyproject.toml`). Black check and mypy could not be executed. Not blocking for Phase 1, but the `pyproject.toml` should add them to `[dev]` if the author intends to gate on them.

---

## Type Hints

**MEDIUM — two gaps:**

1. `train.py:54` — `args` parameter in `main()` is untyped. The caller (`paper_reimpl_shared.runner.entrypoint`) presumably passes an `argparse.Namespace`. Should be `argparse.Namespace` or a Protocol.

2. `model.py:592` / `model.py:642` — `diffusion` parameter in `compute_loss()` and `compute_sds_loss()` is untyped (bare name with a comment). Should import and annotate as `GaussianDiffusion`.

3. `dataset.py:69` — `build_collate()` return type is missing. Should be `Callable[[list[dict[str, Any]]], dict[str, Any]]`.

---

## Code Quality

**MEDIUM — one nit:**

`model.py:111` uses `assert` to validate constructor arguments:
```python
assert down_factor in (4, 8, 16), "down_factor must be one of 4/8/16"
```
`assert` is stripped when Python runs with `-O` (optimized). Prefer `ValueError` for runtime validation in `__init__`. Same applies to `model.py:282`.

---

## Logging

**MEDIUM — nit (not blocking):**

`train.py` uses `print()` throughout for all diagnostic output. For a library callable by `paper_reimpl_shared.runner`, this leaks to stdout unconditionally and cannot be filtered. Prefer `logging.getLogger(__name__)` with appropriate levels (`INFO` for progress, `ERROR` for the non-finite loss case). This is a consistent project-wide pattern choice, so flagging but not blocking.

---

## `num_workers` comment accuracy

**MEDIUM — nit:**

`train.py:124`: comment reads `# smoke / dry-run uses zero workers`, but `num_workers=0` is unconditional — it applies to full training runs too. Either promote `num_workers` to `train_cfg` (with `0` as default), or fix the comment to say `# always 0 in Phase 1`.

---

## Architecture / ML Correctness

No issues found with the training contract:
- Gradient flow verified by the smoke test (`test_smoke_build_and_train_step`).
- VAE encoding is under `torch.no_grad()` as expected for frozen-VAE diffusion training.
- CFG dropout independently masks each conditioning channel, correctly using `null_idx` slots.
- AdaLN-Zero and out-conv are zero-initialized (DiT-style identity init).
- SDS placeholder is clearly marked as such in docstring and `blind_impl.md`.
- The `_CrossAttention` guard `if tokens.numel() == 0: return x` correctly handles the no-refs case.

**One note**: `_CrossAttention` uses manual `q @ k^T / sqrt(d)` + `softmax` rather than `F.scaled_dot_product_attention` (PyTorch 2.0+). Not a bug, but `F.scaled_dot_product_attention` would give free Flash Attention on supported hardware. Low priority for Phase 1.

---

## Nit List (priority order)

| # | File | Line | Severity | Issue |
|---|------|------|----------|-------|
| 1 | `model.py` | 111, 282 | MEDIUM | `assert` in `__init__` → use `ValueError` |
| 2 | `model.py` | 592, 642 | MEDIUM | `diffusion` param untyped → annotate as `GaussianDiffusion` |
| 3 | `train.py` | 54 | MEDIUM | `args` param untyped → `argparse.Namespace` |
| 4 | `dataset.py` | 69 | MEDIUM | `build_collate()` return type missing |
| 5 | `train.py` | 124 | MEDIUM | `num_workers=0` comment misleads (not dry-run only) → expose via `train_cfg` or fix comment |
| 6 | `train.py` | 37,66,74,103,167,176,182 | MEDIUM | `print()` → `logging.getLogger(__name__)` |
| 7 | `model.py` | 63 | LOW | `diffusion_target: str` → `Literal["x0", "epsilon"]` |
| 8 | `train.py` | 32 | LOW | `set_seed()` does not set `torch.backends.cudnn.deterministic` — document intentional omission |
| 9 | `pyproject.toml` | 17 | LOW | `mypy` + `black` absent from `[dev]` extras; add if gate check is intended |
| 10 | `model.py` | 302 | LOW | Manual attention → consider `F.scaled_dot_product_attention` for Phase 2 |

---

## Final Verdict

**APPROVE — Gate 1 PASS.**

No CRITICAL (security, injection, secrets) or HIGH (bare except, missing context managers, concurrency hazards) issues. All ten issues are MEDIUM or LOW nits. The public-repo safety requirement is fully satisfied. The `train.main()` signature matches the contract. `paper_reimpl_shared.*` imports are used exclusively. Ruff is clean. Three pytest tests pass. The code is reviewable, documented, and correct for Phase 1 scope.

Recommend fixing nits 1–4 before Phase 2 (type safety) and nit 5 before multi-GPU training (num_workers will need to be > 0).
