# Code Review — QT-Font Phase 1 Gate 1

**Date**: 2026-05-11  
**Reviewer**: Claude (automated static + manual review)  
**Scope**: `src/qt_font/` — `__init__.py`, `dataset.py`, `model.py`, `sample.py`, `train.py`  
**Tools run**: ruff (project config `E,F,W,I,B,UP`), ruff (`--select ALL` extended pass), manual inspection  
**Baseline**: ruff default config → All checks passed. Extended analysis follows.

---

## Verdict

**WARNING — can merge with caution.**  
No CRITICAL security issues found. Two HIGH issues found (reproducibility + data duplication risk). Remainder are MEDIUM/style.

---

## CRITICAL — Security

None found. No SQL, shell injection, eval/exec, hardcoded secrets, unsafe YAML load, or weak crypto.

---

## HIGH — Correctness / Reproducibility

### [HIGH] `SyntheticConfig.seed` is stored but never consumed
**File**: `/Users/Ayueh/Char/paper_reimpl/papers/05_qt_font/src/qt_font/dataset.py:34,49-57`  
**Issue**: `SyntheticConfig` carries a `seed: int = 0` field. `SyntheticDataset.__iter__` calls `make_synthetic_batch(...)` in every iteration without passing a seed or seeding a local RNG. `make_synthetic_batch` has no `seed` parameter. The field is therefore a dead config value — it creates the appearance of reproducibility that does not exist. Two runs with the same YAML will produce different data.  
**Fix**: Either remove the `seed` field from `SyntheticConfig` (honest), or seed a `torch.Generator` per-sample derived from the iteration index, e.g.:
```python
g = torch.Generator()
for i in range(self.cfg.length):
    g.manual_seed(self.cfg.seed + i)
    # pass generator= to randint calls, or torch.manual_seed inside
```
Note that `make_synthetic_batch` itself would need to accept a `generator` kwarg — this requires a shared-library change.

---

### [HIGH] `IterableDataset` + `num_workers > 0` duplicates data silently
**File**: `/Users/Ayueh/Char/paper_reimpl/papers/05_qt_font/src/qt_font/dataset.py:37-64`  
**File**: `/Users/Ayueh/Char/paper_reimpl/papers/05_qt_font/src/qt_font/train.py:170`  
**Issue**: `SyntheticDataset` is an `IterableDataset` without a `worker_init_fn` or `torch.utils.data.get_worker_info()` guard. When `DataLoader` is constructed with `num_workers > 0`, each worker process runs the full iterator independently, multiplying the apparent epoch size by `num_workers`. The training loop does not guard against this. Currently `num_workers` defaults to 0 in the synthetic path, which avoids the bug, but the YAML config can override it without warning.  
**Fix**: Either assert `num_workers == 0` for `SyntheticDataset`, or implement `get_worker_info()` sharding in `__iter__`:
```python
info = torch.utils.data.get_worker_info()
if info is not None:
    per_worker = math.ceil(self.cfg.length / info.num_workers)
    start = info.id * per_worker
    end = min(start + per_worker, self.cfg.length)
else:
    start, end = 0, self.cfg.length
for i in range(start, end):
    ...
```

---

## HIGH — Code Quality

### [HIGH] `QTFontModel.__init__` exceeds 50-statement limit (PLR0915: 51 statements)
**File**: `/Users/Ayueh/Char/paper_reimpl/papers/05_qt_font/src/qt_font/model.py:428`  
**Issue**: The constructor is 84 lines with 51 statements, covering graph topology construction, buffer registration, and submodule building. This makes the constructor hard to test in parts and difficult to modify safely.  
**Fix**: Extract into three private helpers: `_build_graph_buffers()`, `_build_conditioning_modules()`, `_build_graph_modules()`.

---

### [HIGH] `encode_conditioning` and three `predict_logits_*` methods have > 5 arguments (PLR0913)
**File**: `/Users/Ayueh/Char/paper_reimpl/papers/05_qt_font/src/qt_font/model.py:523,611,655,680`  
**File**: `/Users/Ayueh/Char/paper_reimpl/papers/05_qt_font/src/qt_font/sample.py:22`  
**Issue**: Four methods accept 7–10 parameters; `sample_states` accepts 12. While the keyword-only enforcement (`*`) mitigates misuse at call sites, the signatures are unwieldy and will accumulate further arguments as Stage B/C conditioning is added.  
**Fix**: Introduce a `ConditioningBundle` dataclass:
```python
@dataclass
class ConditioningBundle:
    content: torch.Tensor
    char_id: torch.Tensor | None = None
    writer_id: torch.Tensor | None = None
    script_id: torch.Tensor | None = None
    ref_images: torch.Tensor | None = None
    ref_valid: torch.Tensor | None = None
```
All three `predict_logits_*` methods and `sample_states` can accept one `cond: ConditioningBundle` argument.

---

## MEDIUM — Best Practices / PEP 8

### [MEDIUM] `print()` throughout `train.py` instead of `logging`
**File**: `/Users/Ayueh/Char/paper_reimpl/papers/05_qt_font/src/qt_font/train.py:217,218,238,241,246`  
**Issue**: Five `print()` calls for training progress and diagnostics. When the shared entrypoint captures stdout or when multiple workers run in parallel, output ordering is unpredictable and cannot be redirected to a log file independently.  
**Fix**: Replace with `logging.getLogger(__name__)` and emit at `INFO` level.

---

### [MEDIUM] Unused unpacked variables (RUF059)
**File**: `/Users/Ayueh/Char/paper_reimpl/papers/05_qt_font/src/qt_font/model.py:122,318,581`  
**Issue**: Three locations unpack tensor shapes into variables that are immediately unused:
- Line 122: `B, C, H, W = image.shape` — `C` and the tuple are not referenced after the shape check.
- Line 318: `B, L, C = leaf_x.shape` — `L` is unused.
- Line 581: `B, L = leaf_states.shape` — both `B` and `L` are unused in `predict_state_logits`.  
**Fix**: Replace unused positions with `_`:
```python
B, _, H, W = image.shape   # line 122, C unused
B, _, C = leaf_x.shape     # line 318, L unused
```

---

### [MEDIUM] Magic literal `4` for quadrant count not named (PLR2004)
**File**: `/Users/Ayueh/Char/paper_reimpl/papers/05_qt_font/src/qt_font/model.py:119,350`  
**Issue**: The constant `4` (children per quadtree node) appears as a raw literal in comparisons and shapes. It is the foundational constant of the entire design but is never named.  
**Fix**: Add `_QUAD = 4` as a module-level constant.

---

### [MEDIUM] `__init__.py` module docstring lists only 4 exports; `__all__` exports 8
**File**: `/Users/Ayueh/Char/paper_reimpl/papers/05_qt_font/src/qt_font/__init__.py:1-10`  
**Issue**: The module docstring enumerates `QTFontConfig`, `QTFontModel`, `build_qt_font`, `D3PMUniform`, `build_quadtree_states` (5 entries) but omits `decode_states_to_image` and `quantize_to_states` which are exported in `__all__`. Docstring and `__all__` drift is a maintenance hazard.  
**Fix**: Add the two missing symbols to the docstring's `Public exports` list.

---

### [MEDIUM] `D3PMUniform` stores schedule tensors as plain attributes, not `nn.Module` buffers
**File**: `/Users/Ayueh/Char/paper_reimpl/papers/05_qt_font/src/qt_font/model.py:211-212`  
**Issue**: `D3PMUniform` is not a `nn.Module`; `self.betas` and `self.alphas_cumprod` are plain `torch.Tensor` attributes initialised on a given `device`. In `q_probs` the code uses `.to(x0.device)` to re-cast `alphas_cumprod` on every forward call. This is correct but incurs an unnecessary device-transfer check on each training step. If the caller constructs `D3PMUniform(device="cpu")` and then moves data to GPU, every call silently copies the schedule tensor across the bus.  
**Fix**: Either make `D3PMUniform` a `nn.Module` and register the tensors as buffers, or cache the device at construction and validate consistency in `q_probs` with an assertion.

---

### [MEDIUM] `sample_image` uses a magic-constant pseudo-logit trick
**File**: `/Users/Ayueh/Char/paper_reimpl/papers/05_qt_font/src/qt_font/sample.py:106`  
**Issue**:
```python
pseudo_logits = (one_hot * 10.0) - 5.0
```
This converts a one-hot into logits where the correct class scores 5 and all others score -5, then passes these through `decode_states_to_image`'s `softmax → expected value` path. The result is approximately correct but depends on the magnitude `10.0` being large enough relative to `K` to keep the expected value close to the argmax bin center. For small `K=4` this is fine; for larger `K` (e.g. 32+) the approximation degrades.  
**Fix**: Decode the states directly without the pseudo-logit indirection:
```python
# Pass argmax states as a (B, L) long tensor into quantize-then-upsample,
# or expose a decode_states_to_image overload that accepts class indices.
```
Alternatively, document the magic constant `10.0` as a named variable `_LOGIT_SCALE = 10.0` with a comment explaining the approximation error bound.

---

### [MEDIUM] N806: uppercase dimension variables flagged by PEP 8 naming convention
**File**: Multiple locations in `model.py`, `sample.py`, `train.py`  
**Issue**: Variables `B, L, K, N, C, H, W, R, P` are uppercase in function bodies, which ruff's N806 rule flags. In PyTorch ML code this is the near-universal convention (Tensor dimension naming) and most projects disable N806 explicitly.  
**Fix**: Add `"N806"` to the `[tool.ruff.lint] ignore` list in `pyproject.toml`. This is a one-line config change, not a code refactor. The current project config already ignores `E501` and `B008`; this is a similar pragmatic exception.

---

### [MEDIUM] `main()` parameters `args` and `paths` lack type annotations (ANN001)
**File**: `/Users/Ayueh/Char/paper_reimpl/papers/05_qt_font/src/qt_font/train.py:174,179`  
**Issue**: `main(args, *, ..., paths)` uses untyped `args` and `paths`. These are structural types from the shared entrypoint; even a minimal `Protocol` or `Any` with a comment would make the contract explicit.  
**Fix**: Define a minimal protocol or import the shared type if one exists:
```python
from typing import Any, Protocol
class _Args(Protocol):
    device: str
    synthetic: bool
    dry_run: bool
```

---

## Portability / Reproducibility Assessment

| Concern | Status |
|---|---|
| Cross-platform paths | OK — no hardcoded paths; all paths go through caller's `paths` object |
| Seed propagation | PARTIAL — `_seed_everything` covers global RNG but `SyntheticDataset` ignores its own `seed` field (HIGH above) |
| Deterministic ops | NOT SET — `torch.use_deterministic_algorithms(True)` not called; `AdaptiveAvgPool2d` is non-deterministic on CUDA |
| DataLoader determinism | RISK — no `worker_init_fn` for seed propagation per worker |
| Checkpoint save/restore | ABSENT — Phase 1 trains up to `max_steps` with no checkpoint write; acceptable for smoke scope, must be addressed in Phase 2 |
| CUDA availability guard | OK — `_seed_everything` checks `torch.cuda.is_available()` before seeding CUDA |
| `torch.backends.cudnn.deterministic` | NOT SET — should be set alongside `use_deterministic_algorithms` for full reproducibility |

---

## Summary

| Severity | Count | Items |
|---|---|---|
| CRITICAL | 0 | — |
| HIGH | 4 | Dead seed field, IterableDataset multi-worker duplication, oversized `__init__`, parameter-count explosion |
| MEDIUM | 7 | print→logging, unused unpacked vars, magic `4`, docstring drift, D3PMUniform device copy, pseudo-logit magic constant, missing type annotations |

**Approval**: WARNING — MEDIUM issues only after resolving the two HIGH correctness items (seed and multi-worker). The HIGH refactor items (constructor size, parameter count) can be deferred to Phase 2 but should be tracked.
