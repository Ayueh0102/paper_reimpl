# Code Review — IF-Font Blind Reimplementation
**Scope**: `src/if_font/` (Phase 1 Gate 1)
**Date**: 2026-05-11
**Reviewer**: Claude Code (automated)
**Static analysis**: ruff — all checks passed; mypy — not installed in venv; black — not installed in venv
**Tests**: 8/8 passed

---

## Summary verdict: APPROVE WITH NOTES

No CRITICAL or security-blocking issues. Four HIGH findings (one is a correctness risk in production, three are type-annotation gaps). Multiple MEDIUM findings are noted.

---

## CRITICAL — Security

### SECRET GREP: CLEAN
No hardcoded passwords, API keys, tokens, or credentials found.

### [CRITICAL / Needs Awareness] Arbitrary code execution via `exec_module`
**File**: `/Users/Ayueh/Char/paper_reimpl/papers/03_if_font/src/if_font/dataset.py:65-70`
**Issue**: `load_ids_lookup` dynamically loads and executes an arbitrary `.py` file from a user-supplied path via `importlib.util.spec_from_file_location` + `exec_module`. If `ids_lookup_path` in the YAML is attacker-controlled, this is arbitrary code execution. In the current research context the YAML files are author-controlled and the path is from a trusted config, so this is not an exploitable vulnerability in practice — but it is a pattern that would block review in a production or multi-tenant deployment.
**Fix**: Add a comment in `load_ids_lookup`'s docstring explicitly stating the path must come from a trusted, author-controlled config. Consider checking that the resolved path is within an expected directory (e.g., `~/Char/datasets/`) before executing. For Gate 1 this is acceptable; flag before any public deployment.

---

## HIGH — Correctness / Reproducibility

### [HIGH] `cfg.__dict__` checkpoint serialization is fragile
**File**: `/Users/Ayueh/Char/paper_reimpl/papers/03_if_font/src/if_font/train.py:285`
```python
torch.save({"model": model.state_dict(), "cfg": cfg.__dict__}, path)
```
**Issue**: `IFFontConfig` is a `dataclass` with a nested `VQTokenizerConfig` field (`vq`). `cfg.__dict__` captures the top-level dict but the `vq` field remains a live `VQTokenizerConfig` object — not a plain dict. When the checkpoint is loaded back and the `VQTokenizerConfig` class definition has changed (e.g., a new field added with a default), `torch.load` will silently restore an older instance, and re-constructing an `IFFontConfig` from `cfg.__dict__` will fail or silently drop the new field. This is the most common source of unreproducible experiments.
**Fix**: Use `dataclasses.asdict(cfg)` which deep-converts the nested dataclass to a plain dict:
```python
import dataclasses
torch.save({"model": model.state_dict(), "cfg": dataclasses.asdict(cfg)}, path)
```
On load, reconstruct with `IFFontConfig(**ckpt["cfg"])` (with a small shim for the nested `vq` key).

### [HIGH] O(n²) prefix recomputation in `TransformerARDecoder.sample`
**File**: `/Users/Ayueh/Char/paper_reimpl/papers/03_if_font/src/if_font/model.py:600-617`
**Issue**: The AR sampler grows `prev` by one token each step and re-runs the full prefix through all 10 decoder blocks on every iteration. For a 256-token sequence (16×16 grid at 128×128) this is 256 forward passes over an increasingly long prefix, giving O(n²) compute. The author's comment "for a smoke test this is fine" is noted but the same `sample()` path is called from `IFFont.sample()` which is the inference entrypoint for production generation.
**Fix**: At Gate 1 this is acceptable for correctness. Before Stage B inference benchmarking, implement KV-cache (store past key/value projections per block, indexed by step). This is not a blocker for Gate 1 but must be addressed before large-scale generation runs.

---

## HIGH — Type Hints

### [HIGH] `args` parameter is untyped across `main`, `_build_dataloader`, `build_dataset`
**Files**:
- `/Users/Ayueh/Char/paper_reimpl/papers/03_if_font/src/if_font/train.py:157, 201`
- `/Users/Ayueh/Char/paper_reimpl/papers/03_if_font/src/if_font/dataset.py:237`

**Issue**: All three public entrypoints accept `args` with no type annotation. The actual attributes accessed (`args.synthetic`, `args.dry_run`, `args.device`) are duck-typed via `getattr`. This prevents mypy from catching callers that pass wrong object types and makes the API contract invisible to readers.
**Fix**: Define a `typing.Protocol` or `argparse.Namespace`-compatible type:
```python
from typing import Protocol

class _TrainArgs(Protocol):
    device: str
    dry_run: bool
    synthetic: bool
```
Or use `argparse.Namespace` if the shared runner always produces one.

### [HIGH] `model_cfg` in `build_dataset` is untyped
**File**: `/Users/Ayueh/Char/paper_reimpl/papers/03_if_font/src/if_font/dataset.py:239`
**Issue**: The `model_cfg` parameter has no annotation. Inside `build_dataset` the code accesses `model_cfg.image_size` and `model_cfg.in_channels` as attributes, but the `main()` path passes a typed `IFFontConfig` while the signature says nothing. Callers from outside the package can silently pass a raw dict (which would AttributeError at `.image_size`).
**Fix**: Annotate as `model_cfg: IFFontConfig`.

---

## HIGH — Pythonic Patterns

### [HIGH] Bare `assert` used for runtime contract in `build_context`
**File**: `/Users/Ayueh/Char/paper_reimpl/papers/03_if_font/src/if_font/model.py:691`
```python
assert ids_attention_mask is not None
```
**Issue**: `assert` is stripped in optimized mode (`python -O`). If anyone runs training or evaluation with `-O` (some deployment wrappers do this), the mask will not be validated and a confusing `TypeError` will surface much later when `ids_attention_mask` is indexed.
**Fix**: Replace with an explicit `if ... raise TypeError(...)`:
```python
if ids_attention_mask is None:
    raise TypeError("ids_attention_mask must be provided when ids_token_ids is not None")
```

---

## MEDIUM — Code Quality

### [MEDIUM] `print()` used throughout training loop instead of `logging`
**File**: `/Users/Ayueh/Char/paper_reimpl/papers/03_if_font/src/if_font/train.py:247, 272, 286, 288`
**Issue**: Four `print()` calls handle all training progress, checkpoint notification, and completion. This makes it impossible to redirect, filter by level, or suppress logs when the trainer is used as a library (e.g., from a sweep script). The CLAUDE.md principle "prefer clear scripts" and "each experiment should produce a short markdown report" implies structured logging.
**Fix**: Replace with `import logging; logger = logging.getLogger(__name__)` and `logger.info(...)`.

### [MEDIUM] `IFFontCollate._maybe_fit` mutates shared tokenizer state in a `DataLoader` worker
**File**: `/Users/Ayueh/Char/paper_reimpl/papers/03_if_font/src/if_font/dataset.py:170-175`
**Issue**: `IFFontCollate` is constructed once and passed as `collate_fn`. When `num_workers > 0`, PyTorch forks the collate callable into worker processes, so `_maybe_fit` executes in a subprocess. The `_fitted = True` flag and vocab updates never propagate back to the main process, meaning the tokenizer in the main process never gets fitted, and every worker independently re-fits on first batch. This is a silent inconsistency: the model's `ids_vocab_size` (fixed at construction) may not match the collate's tokenizer after worker-side fitting. In the current code `num_workers=0` during `dry_run` mitigates this for smoke tests, but a production run with `num_workers > 0` will hit this.
**Fix**: Fit the tokenizer fully before constructing the `DataLoader` (pre-scan the manifest for all IDS strings), and set `fit_on_first_call=False`. This is the correct pattern and avoids the multiprocessing mutation problem entirely.

### [MEDIUM] `ref_to_decoder_proj` lazy property pattern is unconventional and risky for `state_dict`
**File**: `/Users/Ayueh/Char/paper_reimpl/papers/03_if_font/src/if_font/model.py:665-676`
**Issue**: The `_ref_proj` module is built lazily via `add_module` on first property access. If `encode_refs_to_tokens` is never called before `state_dict()` (e.g., Stage A training where refs are never encoded), `_ref_proj` will be absent from the checkpoint. Loading that checkpoint into a model where `vq.embedding_dim != d_model` will then fail with a missing-key error.
**Fix**: Instantiate `_ref_proj` eagerly in `__init__`:
```python
self._ref_proj: nn.Module = (
    nn.Identity()
    if cfg.vq.embedding_dim == cfg.d_model
    else nn.Linear(cfg.vq.embedding_dim, cfg.d_model)
)
```

### [MEDIUM] `IDSTokenizer` is a `@dataclass` but is designed to be mutable after construction
**File**: `/Users/Ayueh/Char/paper_reimpl/papers/03_if_font/src/if_font/ids.py:81-92`
**Issue**: `@dataclass` with mutable `list` and `dict` fields and a `__post_init__` that populates them creates a subtle hazard: two `IDSTokenizer()` instances share no state, but `IDSTokenizer(vocab=some_list)` will share the `token_to_id` dict that gets rebuilt from `some_list`, which is correct. The actual issue is that `fit_from_strings` returns `self` (fluent API) while also mutating in place — users could store the return value thinking they have an independent copy. This is a minor API contract ambiguity, not a correctness bug in the current code.
**Fix**: Add a note in the docstring that `fit_from_strings` mutates in place and returns `self` for chaining only.

### [MEDIUM] `data_cfg` and `train_cfg` parameters are typed as bare `dict[str, Any]` without a schema
**Files**: `train.py:201`, `dataset.py:235`
**Issue**: Config dicts are accessed with many `.get()` calls with scattered defaults. A typo in a YAML key (e.g., `max_step` instead of `max_steps`) is silently swallowed as the default. This violates the "each claim needs an ablation" principle because incorrect config is indistinguishable from correct config until training diverges.
**Fix**: Consider a small `TypedDict` or Pydantic model for the train config. At minimum, add a validation pass at the top of `main()` that checks for unexpected keys and raises early.

### [MEDIUM] Magic number `4096` vocab headroom in `main`
**File**: `/Users/Ayueh/Char/paper_reimpl/papers/03_if_font/src/if_font/train.py:214`
```python
cfg.ids_vocab_size = max(cfg.ids_vocab_size, tokenizer.vocab_size + 4096)
```
**Issue**: The `4096` headroom is undocumented. If the actual corpus produces more than 4096 unique component characters not yet in the tokenizer, the model's embedding table will be too small, raising an out-of-bounds index error during the first collate.
**Fix**: Name the constant and document why 4096 was chosen:
```python
_IDS_VOCAB_HEADROOM = 4096  # upper bound on unique CJK leaf components in CHISE CNS table
```

---

## Approval

**Gate 1 decision: APPROVE WITH NOTES.**
- No CRITICAL security vulnerabilities (the dynamic import is acknowledged and YAML-controlled).
- The two correctness-risk HIGHs (`cfg.__dict__` checkpoint and lazy `_ref_proj`) must be resolved before Stage B checkpoint round-trip testing.
- The `IFFontCollate` multi-worker mutation is a latent bug that will surface once `num_workers > 0`; must be fixed before Stage B training launch.
- Type annotation gaps (`args`, `model_cfg`) should be closed in the next PR.

