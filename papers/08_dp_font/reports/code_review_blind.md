# Code Review — DP-Font Blind Reimplementation
**Scope**: `src/dp_font/` (6 files, ~830 LOC)
**Gate**: Phase 1 Gate 1
**Date**: 2026-05-11
**Reviewer**: Claude (automated)
**Tools run**: ruff (all passed), mypy (not installed in venv), secret grep (clean)

---

## Verdict: WARNING — no CRITICAL issues; several HIGH / MEDIUM items

No security blockers. No data-leakage issues. Safe to continue to Gate 2 with the items below tracked as follow-up work before any real-data training run.

---

## CRITICAL — None found

---

## HIGH Issues

### [HIGH] `_move_batch` return-type annotation lies
**File**: `src/dp_font/train.py:226`
**Issue**: Annotated as `-> dict[str, torch.Tensor]` but the body silently passes non-Tensor values through unchanged (metadata dicts, lists). Any downstream code trusting the return type will encounter runtime surprises on heterogeneous batches.
**Fix**: Change return type to `dict[str, Any]` to match the body, or explicitly enumerate and convert/discard non-Tensor fields.

### [HIGH] `_FrozenCondAdapter` stores tensors as plain Python attributes, not `register_buffer`
**File**: `src/dp_font/sample.py:40-58`
**Issue**: `self.stroke_order`, `self.ink_intensity`, `self.font_size` are set as bare attributes. When `model.to(device)` or `model.half()` is called after the adapter is constructed (e.g. for AMP), these tensors are silently left on the original device/dtype. `nn.Module.to()` only moves registered parameters and buffers.
**Fix**: Replace the three plain assignments with `self.register_buffer("stroke_order", stroke_order)` (and similarly for the other two), handling `None` with `register_buffer(..., None)` if your PyTorch version supports it (≥1.10), or guard with an `if v is not None` branch.

### [HIGH] `assert` used as runtime input validation in production loss code
**File**: `src/dp_font/pinn_losses.py:127, 163, 196, 220`
**Issue**: `assert x0_pred.dim() == 4 ...` statements will silently become no-ops when Python runs with `-O` (optimised mode), which is common in Docker/production deployments. These are genuine preconditions, not debugging aids.
**Fix**: Replace with explicit `if` guards raising `ValueError`:
```python
if x0_pred.dim() != 4 or x0_pred.shape[1] != 1:
    raise ValueError(f"expect [B,1,H,W], got {tuple(x0_pred.shape)}")
```

### [HIGH] `main()` and `build_dataset()` use untyped `args` parameter
**File**: `src/dp_font/train.py:233`, `src/dp_font/dataset.py:208`
**Issue**: `args` is typed as the implicit `Any`. Every attribute access (`args.device`, `args.dry_run`, `args.synthetic`) is unchecked and will raise `AttributeError` at runtime if the caller supplies an incompatible namespace. This is the most common integration failure mode.
**Fix**: Define a minimal `Protocol` or `dataclasses.dataclass` for the expected args shape:
```python
from typing import Protocol

class _TrainArgs(Protocol):
    device: str
    dry_run: bool
    synthetic: bool
```
Or at minimum use `argparse.Namespace` as the annotation.

### [HIGH] `synthesise_stroke_order` can silently produce sequences longer than `seq_len`
**File**: `src/dp_font/dataset.py:68`
**Issue**: When `min_len > seq_len` (caller's responsibility), `max(1, seq_len - min_len + 1)` evaluates to `1`, so `length = min_len + 0 = min_len`. The loop then writes `min_len` entries into an `out` list of size `seq_len`, silently overwriting the padding. Currently `min_len` defaults to `1` and callers never override it, so this is latent — but it will break when a real stroke-order DB is plugged in if any character's stroke count exceeds `stroke_seq_len`.
**Fix**: Add a guard at the start of the function:
```python
if min_len > seq_len:
    raise ValueError(f"min_len={min_len} > seq_len={seq_len}")
```

---

## MEDIUM Issues

### [MEDIUM] `print()` used for all training telemetry instead of `logging`
**File**: `src/dp_font/train.py:270-312`
**Issue**: Four bare `print()` calls drive all training progress output. These cannot be redirected, levelled (INFO/DEBUG/WARNING), or suppressed by the shared runner's logging configuration. The project CLAUDE.md already specifies a shared entrypoint pattern; mixing `print` there breaks the contract.
**Fix**: Replace with `import logging; logger = logging.getLogger(__name__)` and use `logger.info(...)`.

### [MEDIUM] `_clamp` has an unused `fallback` parameter
**File**: `src/dp_font/model.py:200`
**Issue**: `def _clamp(self, ids, vocab, fallback)` — `fallback` is declared but never referenced inside the method body. The three call-sites pass `nulls["writer"]` twice (once as `vocab`, once as `fallback`). This suggests either the method signature was trimmed but not cleaned up, or there is a missing branch where `fallback` should replace `None` tensors (that logic currently lives in the `if w is None` block in `forward`).
**Fix**: Remove the `fallback` parameter and document the method's single responsibility clearly, or implement the fallback logic inside `_clamp` and remove the three `if w is None` blocks.

### [MEDIUM] `DataLoader` has no `generator` or `worker_init_fn` for reproducibility
**File**: `src/dp_font/train.py:216-223`
**Issue**: `_seed_everything` seeds the process RNG before the loader is built, but when `num_workers > 0` each worker spawns its own independent RNG state. Without a `generator=torch.Generator().manual_seed(seed)` on the `DataLoader` and a `worker_init_fn` that re-seeds inside the worker, data ordering differs between runs with multiple workers. This matters for the ablation validity required by the project rules.
**Fix**:
```python
g = torch.Generator()
g.manual_seed(seed)
DataLoader(..., generator=g, worker_init_fn=lambda wid: np.random.seed(seed + wid))
```

### [MEDIUM] `SiLU(inplace=True)` inside `time_mlp`, `scalar_proj`, and `fuse` Sequentials
**File**: `src/dp_font/model.py:176, 183, 424`
**Issue**: Inplace activation inside `nn.Sequential` blocks that are then used as inputs to residual additions (`h = h + self.scalar_proj(scalar)` at line 266) is technically safe here because the SiLU output is consumed before the addition — there is no aliasing. However, with AMP (`torch.cuda.amp.autocast`) or `torch.compile` enabled, inplace operations can conflict with the fused kernel's saved-tensor requirements and cause silent gradient corruption or recompilation fallbacks. The risk is currently low (CPU-only smoke tests), but will surface when GPU training + AMP is enabled.
**Fix**: Remove `inplace=True` from the three Sequential SiLU instances. The performance gain of inplace SiLU in a Linear→SiLU→Linear block is negligible.

### [MEDIUM] `mypy` not present in the venv / dev dependencies
**File**: `pyproject.toml`
**Issue**: `[dependency-groups] dev` only includes `ruff`. Mypy, bandit, and black are absent. The CI (if any) cannot run type-checking. Given that HIGH item #1 (wrong return type) and HIGH item #4 (untyped args) are both type errors, mypy would have caught them immediately.
**Fix**: Add `mypy>=1.9` and `bandit>=1.7` to `[dependency-groups] dev`.

### [MEDIUM] `collate_dp_font_batch` does not validate that all items have equal-length `stroke_order`
**File**: `src/dp_font/dataset.py:175-177`
**Issue**: `torch.tensor([list(item["stroke_order"]) for item in batch])` will raise a confusing `ValueError: expected sequence of length N at dim 1` if any two items in the batch have different-length stroke_order lists (which can happen if a caller mixes `DPFontDataset` items with manually constructed items that don't go through `synthesise_stroke_order`). The error message does not point to the offending item.
**Fix**: Assert lengths are uniform before the `torch.tensor` call, or use `torch.stack([torch.tensor(item["stroke_order"]) for item in batch])` after verifying lengths.

---

## LOW / Style Notes

- `__init__.py` exports only model symbols; `pinn_losses` and `compute_loss` referenced in the module docstring are not in `__all__`. Minor documentation–API mismatch.
- `DPFontUNet.__init__` is ~98 lines. Acceptable for a U-Net constructor but at the edge; extracting `_build_down_stages()` and `_build_up_stages()` would help reviewability.
- `stroke_pos_emb` index clamp at `model.py:242` correctly bounds at `stroke_seq_len` (size = `stroke_seq_len + 1`), so no off-by-one OOB. This was inspected and is correct.
- Attention math in `SelfAttn2D.forward` (lines 359–365): QK^T computation is correct; scale by `sqrt(c // num_heads)` is correct. No numerical issue found.
- The `stroke_continuity_penalty` neighbourhood-mean formula (`(9*blurred - x0) / 8`) is mathematically correct (8-neighbour mean).
- No hardcoded secrets, no `eval`/`exec`/`pickle`/`subprocess`, no `yaml.unsafe_load`, no path-traversal vectors.

---

## Reproducibility Assessment

| Criterion | Status |
|---|---|
| Deterministic data synthesis (SHA-256 hash) | PASS |
| `_seed_everything` covers process RNG | PASS |
| DataLoader multi-worker seed propagation | FAIL (MEDIUM above) |
| Checkpoint saves `cfg.__dict__` alongside state_dict | PASS |
| dry_run skips checkpoint write | PASS |
| Config parsed from YAML without `yaml.unsafe_load` | PASS (shared lib) |

---

## Summary Table

| Severity | Count | Gate impact |
|---|---|---|
| CRITICAL | 0 | — |
| HIGH | 5 | Block before GPU training begins |
| MEDIUM | 5 | Fix before Stage B/C handoff |
| LOW | 4 | Best-effort cleanup |

**Recommendation**: Address HIGH items 1–5 before the first real-manifest training run. MEDIUM items (especially DataLoader reproducibility) should be resolved before any ablation comparison is declared valid.
