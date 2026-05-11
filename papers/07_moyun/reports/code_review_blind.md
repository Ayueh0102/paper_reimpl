# Code Review — Moyun Blind Reimplementation
**Date**: 2026-05-11
**Reviewer**: Claude (automated, Phase 1 Gate 1)
**Scope**: `src/moyun/` — all 6 Python files
**Tools run**: ruff (All checks passed), mypy (not installed in venv), bandit (not installed in venv), manual read of all files and tests.

---

## Verdict

**APPROVED WITH WARNINGS** — No CRITICAL issues. Three HIGH issues (type annotation gaps) and several MEDIUM/LOW issues documented below. Safe to proceed to Phase 2 gate.

---

## CRITICAL — Security

No issues found. No SQL, no subprocess, no eval/exec, no hardcoded secrets, no YAML unsafe load, no path traversal from user input, no weak crypto.

---

## HIGH Issues

### [HIGH-1] Missing type annotations on `args` parameters
**Files**: `dataset.py:47`, `train.py:168`, `train.py:201`

The `args` parameter in `build_dataset`, `_build_dataloader`, and `main` is typed as bare `Any`-equivalent (no annotation at all). Public functions with unannotated parameters do not satisfy the project's PEP 8 / type-safety gate, and mypy will silently skip type checking inside those function bodies.

```python
# Current
def build_dataset(*, args, data_cfg: dict[str, Any], ...):

# Fix: define a protocol or use argparse.Namespace
import argparse
def build_dataset(*, args: argparse.Namespace, data_cfg: dict[str, Any], ...):
```

If a richer `args` shape is expected (e.g., from a custom runner), define a `typing.Protocol` with the required attributes (`synthetic`, `dry_run`, `device`).

### [HIGH-2] `MoyunConfig.extra` field typed as bare `dict` (missing type params)
**File**: `model.py:127`

```python
extra: dict = field(default_factory=dict)
```

`dict` without type parameters is `dict[Any, Any]`. Any content inserted into this field bypasses type checking downstream. Fix:

```python
extra: dict[str, Any] = field(default_factory=dict)
```

### [HIGH-3] `assert` used as a runtime invariant check in production path
**File**: `model.py:244`

```python
assert seq_len == hh * ww, f"L={seq_len} doesn't match grid {hh}x{ww}"
```

Python strips all `assert` statements when run with `-O` (optimise flag), which CI/inference runners sometimes enable. A hard invariant that, if violated, would cause a silent shape error downstream should use `raise ValueError(...)`.

```python
if seq_len != hh * ww:
    raise ValueError(f"L={seq_len} doesn't match grid {hh}x{ww}")
```

---

## MEDIUM Issues

### [MEDIUM-1] `print()` used throughout instead of `logging`
**File**: `train.py:236`, `train.py:261`, `train.py:272`, `train.py:274`

All training progress and checkpoint messages go via bare `print`. In a shared-runner context, the calling process may redirect or suppress stdout. Standard practice for library code is `logging.getLogger(__name__)`. The print calls are harmless for smoke runs but should be migrated before multi-GPU or daemon deployments.

### [MEDIUM-2] `Optional[int]` should be `int | None` (style drift)
**Files**: `model.py:113`, `model.py:118`, `mamba_block.py:66`, `train.py:41–43`

The codebase mixes `from __future__ import annotations` (which defers all annotations) with the older `Optional[T]` spelling from `typing`. With `from __future__ import annotations` already present in every file, `int | None` is valid and more readable. This is a style issue, not a bug; fix opportunistically.

### [MEDIUM-3] Sequential Python loop over timesteps is an O(L) bottleneck
**File**: `mamba_block.py:193–197`

```python
for t in range(seq_len):
    h = A_bar[:, t] * h + u[:, t]
    ...
```

This is documented as intentional for portability. It is correct but will be the dominant latency for any sequence longer than the smoke-test 16 tokens. The comment in the file already calls this out. **Not a bug** — but flag it here so the Gate 2 reviewer knows it is the planned performance regression and not an oversight.

### [MEDIUM-4] Seed control in `sample()` is partially dead code
**File**: `sample.py:64–84`

`init_noise` is created with a seeded generator but then immediately discarded (`_ = init_noise`) because the code delegates random generation to `diffusion.sample`. The comment acknowledges this but the dead variable still adds noise to the interface contract. Either pass `init_noise` to the shared sampler (if supported) or remove the generator and the `init_noise` variable entirely, and document that reproducibility depends on the shared sampler.

### [MEDIUM-5] `drop_last=False` in DataLoader risks non-uniform batch sizes
**File**: `train.py:185`

With `drop_last=False`, the final batch of each epoch may be smaller than `batch_size`. `compute_loss` uses `b = x0.shape[0]` correctly throughout, so this does not cause a crash. However, CFG dropout draws `b` Bernoulli samples; with a partial batch the dropout rate is slightly biased (negligible in practice but worth noting for reproducibility documentation).

---

## LOW / INFORMATIONAL

### [LOW-1] `_CollateNoRefs` wrapper is over-engineered for a one-liner
**File**: `train.py:158–165`

The class exists solely because DataLoader requires a picklable collate_fn (lambdas are not picklable with multi-process workers). This is the correct pattern when `num_workers > 0`. It is fine; just ensure the class is kept if `num_workers` is ever raised above 0 in YAML configs.

### [LOW-2] `bool(m.get("bidirectional", True))` coercion from YAML is fragile
**File**: `train.py:138`

YAML `false` deserialises to Python `False`, but `"false"` (string) deserialises to `True` (non-empty string). Since the YAML loader in the shared runner should produce native booleans, this is only a risk if someone edits the YAML to quote the value. Acceptable risk; document in the YAML template.

### [LOW-3] `_sinusoidal_time_embed` divides by `max(1, half)` defensively but `dim=0` is never expected
**File**: `model.py:137`

The guard is correct. Minor style: since `hidden_dim` is validated to be at least `1` in `MoyunConfig`, `dim=0` arriving here would indicate a config bug. A comment or an assertion would make the intent clearer.

### [LOW-4] Import of `Optional` from `typing` can be removed
**Files**: `mamba_block.py:65`, `model.py:49`

Both files use `from __future__ import annotations`, making runtime evaluation of annotations unnecessary. The `Optional` import is used only in annotations, so it can be removed (replace `Optional[int]` with `int | None`). Not an error but creates a minor dead import.

---

## Reproducibility Checklist

| Item | Status |
|---|---|
| Seed set for `random`, `numpy`, `torch`, `cuda` | PASS — `_seed_everything` covers all four |
| DataLoader `drop_last` effect on CFG dropout noted | WARN — see MEDIUM-5 |
| Sampling seed honoured end-to-end | WARN — see MEDIUM-4 (dead `init_noise`) |
| Checkpoint saves `cfg.__dict__` alongside weights | PASS |
| `dry_run` bypasses checkpoint write | PASS |
| Smoke test asserts finite loss and finite params | PASS |
| TripleLabel independence proven by gradient test | PASS |
| adaLN-Zero zero-init verified by gradient test | PASS |

---

## Portability Checklist

| Item | Status |
|---|---|
| No `mamba-ssm` / Triton import | PASS — pure-PyTorch scan |
| Runs on CPU (no `.cuda()` hard-coding) | PASS |
| Path resolution uses `pathlib.Path` | PASS |
| No platform-specific shell calls | PASS |
| `pyproject.toml` lists `torch>=2.1` minimum | PASS |

---

## PEP 8 / Style Summary

ruff reports **0 violations**. Manual review found the `Optional` / union-syntax drift (LOW-4 / MEDIUM-2) and the `dict` bare type (HIGH-2). No naming violations, no import order issues, no star imports.

---

## Summary of Required Actions Before Gate 2

1. **(HIGH-1)** Annotate `args` parameter in `build_dataset`, `_build_dataloader`, `main` — use `argparse.Namespace` or a Protocol.
2. **(HIGH-2)** Parameterize `MoyunConfig.extra` as `dict[str, Any]`.
3. **(HIGH-3)** Replace the `assert` in `PatchUnembed.forward` with `raise ValueError`.
4. **(MEDIUM-1)** Replace `print` in `train.py` with `logging.getLogger(__name__)`.
5. **(MEDIUM-4)** Resolve the dead `init_noise` variable in `sample.py` — either use it or remove it and document seeding behaviour.
