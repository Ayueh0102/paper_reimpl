# Code Review — Calliffusion Phase 1 Gate 1

**Date**: 2026-05-11  
**Reviewer**: Claude Sonnet 4.6 (automated)  
**Scope**: `src/calliffusion/` — all 7 modules  
**Static analysis**: ruff ✅ (all checks passed); black/mypy/bandit not installed in venv  
**Verdict**: WARN — no CRITICAL issues; several HIGH and MEDIUM issues logged below

---

## CRITICAL — None found

No SQL injection, command injection, path traversal, eval/exec abuse, unsafe
deserialization, hardcoded secrets, or weak crypto patterns were detected.

---

## HIGH Issues

### H1 — Missing return-type annotation on two public functions
File: `src/calliffusion/train.py:81`, `train.py:108`

```python
def build_text_encoder_from_cfg(model_cfg: dict[str, Any]):   # no return type
def freeze_text_encoder(text_encoder, train_cfg: ....) -> None:  # text_encoder untyped
```

**Issue**: `build_text_encoder_from_cfg` has no return annotation; `freeze_text_encoder`
has no type for its `text_encoder` parameter. Both are called throughout the training
path. Without annotations mypy cannot catch a wrong encoder type silently passed in.

**Fix**:
```python
def build_text_encoder_from_cfg(model_cfg: dict[str, Any]) -> nn.Module: ...
def freeze_text_encoder(text_encoder: nn.Module, train_cfg: dict[str, Any]) -> None: ...
```

---

### H2 — `unfreeze()` predicate has no type annotation
File: `src/calliffusion/model.py:393`

```python
def unfreeze(self, predicate=lambda name: True) -> None:
```

**Issue**: `predicate` is a mutable default that is a callable — its type is
`Callable[[str], bool]`. Without the annotation, static tools cannot flag passing a
non-callable. This is a public method that will be called during Stage-C fine-tuning.

**Fix**:
```python
from collections.abc import Callable

def unfreeze(self, predicate: Callable[[str], bool] = lambda name: True) -> None:
```

---

### H3 — `CalliffusionUNetConfig.validate()` only checks `base_channels % num_heads`,
not all stage channels
File: `src/calliffusion/model.py:200–208`

**Issue**: `channel_mult` can produce stage channel widths that are NOT divisible by
`num_heads`. For example `base_channels=320, channel_mult=[1,2,4,4], num_heads=8` is
fine (320/640/1280 all divisible), but any non-power-of-2 multiplier can silently
produce a reshape crash deep inside `SpatialSelfAttention.forward` with a cryptic
`RuntimeError` rather than a clear config error at startup.

**Fix**: extend `validate()`:
```python
for mult in self.channel_mult:
    stage_ch = self.base_channels * mult
    if stage_ch % self.num_heads != 0:
        raise ValueError(
            f"stage channel {stage_ch} (base {self.base_channels} × mult {mult}) "
            f"not divisible by num_heads {self.num_heads}"
        )
```

---

### H4 — `CalliffusionPromptDataset.__getitem__` uses module-level `random.random()`
File: `src/calliffusion/dataset.py:89`

```python
if self.cfg.prompt_dropout_p > 0 and random.random() < self.cfg.prompt_dropout_p:
```

**Issue**: `SyntheticPromptDataset` correctly uses a seeded `random.Random` instance
(`self._rng`) so that results are reproducible across runs. But `CalliffusionPromptDataset`
uses the global `random` module, whose state is shared across threads and is not
controlled by `set_seed()` in `train.py`. Two training runs with the same seed will
produce different prompt-dropout patterns, breaking reproducibility for real-data runs.
DataLoader workers also fork the process, so dropout patterns will differ by worker rank.

**Fix**: accept an optional `seed` in `CalliffusionPromptDataset.__init__` and store a
`random.Random(seed)` instance; use it in `__getitem__`. Or use a deterministic hash of
`index` as the dropout gate so the dataset is stateless per-item.

---

### H5 — `StubTextEncoder._tokenize` mutates vocabulary inside a forward pass (not thread-safe)
File: `src/calliffusion/text.py:101–119`

**Issue**: Every call to `_tokenize` for an unseen token extends `self.token_to_id`,
increments `self._vocab_size`, and rebuilds `self.embedding` in-place. This happens
inside `encode()` which is called during `forward()`. If `num_workers > 0` in the
DataLoader (or if two threads call `encode()` concurrently), the embedding resize is a
data race. Even in single-worker mode this means the stub's embedding table is
non-deterministic across calls if the encounter order of tokens changes.

The production path (`BertTextEncoder`) does not have this issue. The risk is test-only,
but it can cause flaky CI if the order of test execution changes vocabulary growth.

**Fix**: pre-populate vocabulary during `add_special_tokens` and raise a clear error (or
silently map to `UNK_ID`) for tokens not in the pre-built vocab during `_tokenize`. This
matches the semantics of real BERT tokenization.

---

## MEDIUM Issues

### M1 — `print()` used throughout `train.py` instead of `logging`
File: `src/calliffusion/train.py:167,174,237,247,250,252`

**Issue**: A library/training module should use `logging.getLogger(__name__)` so callers
can control verbosity, redirect to files, or suppress output entirely. `print()` bypasses
the logging framework and cannot be filtered by level.

**Fix**: replace with:
```python
import logging
_log = logging.getLogger(__name__)
_log.info("[calliffusion] registered %d writer special tokens", added)
```

---

### M2 — `sampler` parameter is shadowed by local re-assignment
File: `src/calliffusion/sample.py:30,60`

```python
def sample_prompts(..., sampler: str = "ddpm", ...) -> torch.Tensor:
    ...
    sampler = sampler.lower()   # shadows the parameter
```

**Issue**: Shadowing a parameter with a same-name local is a minor clarity issue and
raises a ruff `PLW0127` / pylint `redefined-outer-name` warning. It also means the
original parameter value is lost (irrelevant here but inconsistent style).

**Fix**: use `sampler_name = sampler.lower()` and reference `sampler_name` in the
conditional below.

---

### M3 — Text-encoder dispatch uses fragile `hasattr` duck-typing
File: `src/calliffusion/train.py:219`, `src/calliffusion/sample.py:45`

```python
ctx_out = text_encoder.encode(prompts) if hasattr(text_encoder, "encode") else text_encoder(prompts)
```

**Issue**: Both `StubTextEncoder` and `BertTextEncoder` expose both `.encode()` and
`__call__`. The `hasattr` guard exists only to handle hypothetical third-party encoders
that lack `.encode()`. This pattern will silently fall back to `__call__` if `encode` is
accidentally deleted, and it is duplicated in two files.

**Fix**: define a `TextEncoder` `Protocol` or `ABC` with a mandatory `encode(prompts:
list[str]) -> TextEncoderOutput` signature, and type `text_encoder` as that protocol.
Then remove the `hasattr` guard.

---

### M4 — `train.py:text_encoder.train()` called unconditionally when BERT is frozen
File: `src/calliffusion/train.py:211`

**Issue**: When `freeze_text_encoder` freezes the BERT weights, calling
`text_encoder.train()` still sets all sub-modules (including `nn.Dropout` layers inside
BERT) to training mode, which will activate dropout during BERT's forward pass. If the
intent is to freeze BERT as a fixed feature extractor, it should remain in `eval()` mode
to disable dropout and ensure deterministic outputs.

**Fix**: only call `text_encoder.train()` when the text encoder has trainable parameters:
```python
if any(p.requires_grad for p in text_encoder.parameters()):
    text_encoder.train()
else:
    text_encoder.eval()
```

---

### M5 — `build_unet_from_yaml` silently accepts invalid keys without warning
File: `src/calliffusion/model.py:399–414`

**Issue**: Unknown YAML keys (typos such as `dropuot` instead of `dropout`) are silently
ignored because each field is extracted with `.get(..., default)`. There is no validation
that the supplied dict contains only recognised keys.

**Fix**: after building `cfg`, compare `set(section.keys())` against the known field
names and `warnings.warn()` or raise for unrecognised keys.

---

### M6 — `einops` listed as a dependency but never imported
File: `pyproject.toml:14`

**Issue**: `einops>=0.7` appears in `[project.dependencies]` but no file in
`src/calliffusion/` imports it. This inflates the install footprint and misleads
downstream readers about the architecture.

**Fix**: remove from `dependencies`; add to a comment or `[project.optional-dependencies]`
if it is planned for a later phase.

---

## LOW / Style Notes (no blocking impact)

- **L1** `model.py:417` — `cross_attention_modules` returns `Iterable[SpatialCrossAttention]`
  but is annotated as `Iterable` in the function signature return. The annotation on the
  `yield` expression is correct; the function signature should match.
  (Returns `Generator` at runtime; annotating as `Iterator[SpatialCrossAttention]` is cleaner.)

- **L2** `lora.py:65` — `apply_lora_to_module` accepts `target_substrings: Iterable[str]`
  but immediately iterates over it twice (once in the `any()` inside the comprehension).
  This is fine for `list`/`tuple` defaults, but would silently produce wrong results if
  the caller passes an exhausted generator. Accept `Sequence[str]` or materialise once
  with `target_substrings = tuple(target_substrings)` at the top of the function.

- **L3** `text.py:86–87` — variable `tok` is re-bound inside the loop:
  `tok = str(tok).strip()` shadows the loop variable of the same name. Works correctly,
  but the shadow is confusing. Rename to `raw` / `cleaned`.

---

## Reproducibility Assessment

| Concern | Status |
|---|---|
| Global random state in `CalliffusionPromptDataset` | **FAIL** (H4) |
| `set_seed` called before dataset build | OK |
| Dataclass config fully serialisable to YAML | OK |
| Checkpoint metadata not yet written by `main()` | Not implemented (Phase 1 scaffold) |
| Synthetic dataset is fully deterministic per seed | OK |

---

## Security Assessment

No user-controlled input flows into shell commands, SQL, `eval`, `pickle.loads`, or
`yaml.load`. `BertModel.from_pretrained` downloads from the HF Hub — acceptable for
research use; production deployments should pin model hashes.

---

## Summary

**Verdict: WARN — mergeable with caution.**  
No CRITICAL or security-blocking issues. The five HIGH items (H1–H5) should be resolved
before Stage B training begins, as H4 breaks reproducibility on real-data runs and H5
can cause flaky tests. MEDIUM items M1 and M4 should be addressed before any multi-GPU
or multi-worker run. The codebase is otherwise well-structured, clearly documented, and
the test suite exercises the core paths correctly.
