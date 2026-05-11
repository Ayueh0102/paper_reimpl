# Code Review — 01_fontdiffuser — Phase 1 (blind impl)

## Verdict: PASS-WITH-NITS

---

## Auto checks

- **ruff check**: FAIL — 10 issues found (7 auto-fixable, 3 manual)
- **mypy**: PASS — `Success: no issues found in 5 source files`
- **pytest**: PASS — 2/2 passed in 1.54 s (CPU-only, no real data)

---

## Manual checks

### 1. Public-repo safety (CRITICAL)
- [✓] `ptri78322626` — 0 matches
- [✓] `Masg844ssu` — 0 matches
- [✓] `10.102.10.212` / `100.91.52.77` — 0 matches
- [✓] `qaz741945` — 0 matches
- [✓] No `.env` / `secrets*` / `credentials*` files
- [✓] No hardcoded API tokens or passwords

### 2. Backend portability
- [✓] No `D:\Char\ayueh` in any `.py` file
- [✓] No `mother_repo_link/` direct import in `.py`
- [✓] Device from `args.device` (CLI), never hardcoded `cuda:0`
- [✓] `ckpt_dir` in YAMLs uses relative paths (`experiments/01_fontdiffuser/outputs/…`)

### 3. Reproducibility
- [✓] `_seed_everything()` sets `random`, `np.random`, `torch.manual_seed`, `cuda.manual_seed_all`
- [✓] Seed called at top of `main()` before any model/data construction
- [~] No `torch.use_deterministic_algorithms(True)` — acceptable; none of the YAMLs declare a
  `deterministic: true` key, so this gate passes. Add the key + guard if needed for ablations.

### 4. ruff / PEP 8 / type hints / Pythonic

**Unused imports (fixable with `ruff --fix`):**

- [✗] **NITS**: `dataset.py:15` — `pathlib.Path` imported but unused
- [✗] **NITS**: `dataset.py:20` — `paper_reimpl_shared.config.resolve_path` imported but unused
- [✗] **NITS**: `model.py:35` — `dataclasses.field` imported but unused
- [✗] **NITS**: `train.py:13` — `dataclasses.dataclass` imported but unused
- [✗] **NITS**: `train.py:15` — `typing.Iterable` imported but unused
- [✗] **NITS**: `train.py:24-25` — `CalligraphyJsonlDataset`, `SyntheticCalligraphyDataset` imported but unused
  (both are consumed by `build_dataset` via `dataset.py`; the re-import in `train.py` is redundant)

**Ambiguous variable names (E741):**

- [✗] **NITS**: `model.py:292` — `l = style_tokens.shape[1]` in `RSIBlock.forward`
- [✗] **NITS**: `model.py:565` — `l = tokens.shape[1]` in `FontDiffuser.encode_style`
  Fix: rename to `seq_len` or `n_tokens`.

**Unused assignment:**

- [✗] **NITS**: `model.py:558` — `device = self.style_null_token.device` assigned but never used
  Fix: remove the line; `tokens.device` is used implicitly via `.expand()`.

**Untyped `args` parameters (HIGH — type hints):**

- [✗] **NITS**: `dataset.py:48`, `train.py:190`, `train.py:254` — `args` has no type annotation.
  The shared entrypoint contract deliberately uses a duck-typed namespace, but `argparse.Namespace`
  or a `Protocol` stub would make the contract inspectable by mypy.
  Fix: `import argparse` and annotate as `args: argparse.Namespace` (or define a `TrainArgs` Protocol).

**Untyped `model_cfg` parameter in `dataset.build_dataset`:**

- [✗] **NITS**: `dataset.py:50` — `model_cfg` has no type annotation (it's `FontDiffuserConfig` at call site).
  Fix: `model_cfg: FontDiffuserConfig`.

**`print()` instead of `logging` (MEDIUM):**

- [✗] **NITS**: `train.py:298,323,336,338` — uses `print()` for training progress and checkpoint saving.
  Acceptable in a research reimpl if stdout capture is the intended log sink (e.g. `nohup` redirect),
  but a `logging.getLogger(__name__)` setup would make it filterable. No block.

**Function length (HIGH — code quality):**

- [✗] **NITS**: `model.py:370–461` — `FontDiffuserUNet.__init__` is 91 lines. Complex but structurally
  necessary (building parallel ModuleLists for down/up paths); no clean split point. Document the
  constraint or extract `_build_down_stages()` / `_build_up_stages()` helpers.
- [✗] **NITS**: `train.py:254–339` — `main()` is 85 lines. Acceptable as a top-level orchestrator;
  extract `_run_training_loop()` to bring it under 50 lines.

### 5. Testability

- [✓] `test_smoke.py` uses `paper_reimpl_shared.runner.smoke.make_synthetic_batch` — no disk I/O
- [✓] Tests run CPU-only; no CUDA assertion
- [✓] Both tests independently runnable (`pytest tests/test_smoke.py`)
- [~] **NITS**: `test_smoke_scr_loss_contributes` — `extractor` is frozen (`requires_grad=False`)
  but `extractor.eval()` is not called. `StyleExtractor` uses GroupNorm (not BatchNorm) so stats
  are batch-local and `train` mode is benign here, but adding `.eval()` is the correct convention
  for frozen inference modules.

### 6. AGENTS.md compliance

- [✓] `main(args, *, data_cfg, model_cfg, train_cfg, paths)` signature matches spec exactly
- [✓] All imports use `paper_reimpl_shared.*` (never `..shared` or relative sibling paths)
- [✓] No modification of `shared/` or sibling papers detected

---

## FAIL fixes (block Phase 2)

**None.** Zero CRITICAL or HIGH severity issues found.

---

## Nits (safe to fix before Phase 2 — do not block)

1. **Unused imports x7** — `dataset.py` (Path, resolve_path), `model.py` (field),
   `train.py` (dataclass, Iterable, CalligraphyJsonlDataset, SyntheticCalligraphyDataset).
   Run `ruff check --fix src/ tests/` to auto-remove.

2. **Ambiguous variable name `l` (E741)** — `model.py:292,565`.
   Rename to `seq_len` or `n_tokens`.

3. **Unused assignment `device`** — `model.py:558`. Remove the line.

4. **Untyped `args` and `model_cfg` parameters** — `dataset.py:48,50`, `train.py:190,254`.
   Annotate `args: argparse.Namespace` (or a Protocol) and `model_cfg: FontDiffuserConfig`.

5. **`print()` in `train.py`** — replace with `logging.getLogger(__name__)` for filterability
   in lab-server deployments where log level control matters.

6. **`FontDiffuserUNet.__init__` length** — consider extracting `_build_down_stages()` /
   `_build_up_stages()` helpers to bring the method under 50 lines.

7. **`main()` length** — extract `_run_training_loop()` to isolate the epoch/step loop.

8. **`extractor.eval()` missing in smoke test** — `test_smoke.py:135`. Add before calling
   `compute_loss` to match the convention enforced in `train.main()`.

---

## Additional observations (informational)

- **`RSIBlock` all-masked softmax risk** — theoretically the `masked_fill(..., -inf)` path in
  `RSIBlock.forward` could produce `NaN` if all style tokens were masked. In practice this cannot
  happen: `encode_style` always returns an all-ones mask (substituting `style_null_token` for
  dropped refs rather than masking them). The code path is safe but the mask parameter is
  functionally dead code today. Consider removing it or documenting the intent for when
  per-token masking is added.

- **Stage A `data_stage_a.yaml` has both `source: synthetic` and a `manifest:` key** — the manifest
  key is reachable only when `source` is changed to `manifest`, which is correct. However, having
  a `.yaml` that silently ignores one of its keys may surprise future maintainers. A comment
  making the mutual exclusivity explicit would help.

- **`_seed_everything` does not guard `torch.cuda.manual_seed_all` with `is_available()`** —
  it does: `if torch.cuda.is_available()`. Confirmed correct.
