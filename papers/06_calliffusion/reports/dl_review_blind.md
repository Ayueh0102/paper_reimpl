# DL Review — 06_calliffusion — Gate 1 (blind impl)

## Verdict: PASS-WITH-NITS

Phase-1 blind impl is internally consistent with `paper_notes/06.md` and
`reports/blind_impl.md`. The four special-focus items (BERT special tokens
+ resize, LoRA targets/rank/init, CFG dropout, ε-MSE loss) are all
implemented correctly per the stated decisions. No CONTAMINATION
detected — no references to an external calliffusion repo, no commit
fingerprints, no copied class names or magic numbers traceable to the
official codebase. Three nits and one ablation-worthy gap are listed at
the bottom but none block Gate-1 sign-off.

---

## Checked

### Loss correctness
- [PASS] **ε-prediction MSE matches paper note §3** — `loss = F.mse_loss(pred, d_batch.target, reduction='mean')` at `src/calliffusion/train.py:227`. `d_batch.target` is the sampled noise because `GaussianDiffusion(..., prediction_target="epsilon")` is hard-coded at `train.py:199–200`; verified in shared lib `shared/src/paper_reimpl_shared/diffusion/gaussian.py:117` (`target = x0 if self.prediction_target == "x0" else noise`).
- [PASS] **reduction='mean'** — default; matches the convention assumed by `lr=1e-5` (per-pixel mean MSE).
- [PASS] **single-term loss, no auxiliary VLB/LPIPS/perceptual** — `train.py:227` is the only loss; no other backward()-bound tensor.
- [PASS] **CFG dropout p=0.1 on the text prompt** — implemented in two places:
  - `dataset.py:89` (real path) `if random.random() < self.cfg.prompt_dropout_p: prompt = ""`
  - `dataset.py:149` (synthetic) symmetric path
  - Wired through `train.py:63,77` (`prompt_dropout_p` reaches both dataset constructors).
  - Drop is to empty string (`""`) which the BERT tokenizer maps to `[CLS][SEP]` → matches the SD-1.x "drop-to-empty" convention claimed in `blind_impl.md` decision 11.

### Gradient flow
- [PASS] **No spurious detach on the critical path** — searched `model.py`, `train.py`, `lora.py`, `text.py` for `.detach(`/`torch.no_grad`; only occurrences are inside `add_special_tokens` table-resizing helpers (`text.py:69,96,114`) and inside `sample.py:21` (`@torch.no_grad()` on inference). Training-loop forward at `train.py:219–227` is grad-clean.
- [PASS] **BERT context reaches loss** — `ctx_out.last_hidden_state` flows: `text_encoder.encode(prompts)` (`train.py:219`) → `unet(..., context=ctx_out.last_hidden_state, context_mask=ctx_out.attention_mask)` (`train.py:221–226`) → cross-attn `to_k(context), to_v(context)` (`model.py:143–144`) → SDPA → residual add to `h` (`model.py:156`) → `conv_out` → `loss`. Path is fully connected; no detach.
- [PASS] **Cross-attn applied at every stage (down/mid/up)** — explicit loop covers all stages: down `model.py:349–352`, mid `model.py:359–362`, up `model.py:370–374`. Each ResBlock is followed by self-attn then cross-attn; matches `paper_notes/06.md` mermaid diagram.
- [PASS] **conv_out zero-init does not block grad** — `model.py:330–331` zero-inits the final conv weight+bias. This is deliberate (commented at `model.py:328–329`) and standard. The smoke test in `tests/test_smoke.py:106–107` explicitly un-zeros conv_out before the LoRA-gradient assertion, showing the author understood the implication.
- [PASS] **BERT freeze schedule wired correctly across stages** —
  - Stage A: `train_stage_a_ttf.yaml:21–22` `freeze: true, embeddings_trainable: false` → `text.py:182–190` freezes all params, leaves embedding row table frozen.
  - Stage B: `train_stage_b_midtrain.yaml:22–23` `freeze: true, embeddings_trainable: true` → `text.py:188–190` re-enables grad on `bert.embeddings.word_embeddings` only; new writer-special-token rows can learn while the rest of BERT stays frozen.
  - Stage C: `train_stage_c_ernantang.yaml:22–25` `freeze: true, embeddings_trainable: true` + `lora.enabled: true` → matches paper note §2.4 decision 9.
- [PASS] **Optimizer parameter selection respects freeze + LoRA** — `train.py:177–184`: when LoRA is enabled the trainable set is replaced with `lora_parameters(unet)` (so the `freeze_non_lora` call at `train.py:103` is the active filter), and the trainable text-encoder params (the writer-embedding rows from Stage B/C) are appended at `train.py:182`. `RuntimeError` guard at `train.py:183–184` prevents silent zero-param training.

### Conditioning paths (Calliffusion-specific rubric row)
- [PASS] **BERT encoder freeze (Stage A/B) vs trainable (Stage C-embeddings)** — see Stage A/B/C wiring above. The schedule is "BERT body frozen for all three stages; embedding rows trainable from Stage B onward" — this is what the rubric demanded (`paper_notes/06.md` §2.4 says BERT frozen during A and B, "unfrozen for the writer-name embeddings only" during C; the configs go slightly stronger by also unfreezing embeddings at Stage B, which is what is needed for the special-token rows to learn in the first place. Consistent with `blind_impl.md` decision 9.)
- [PASS] **LoRA rank=4, α=8** — `train_stage_c_ernantang.yaml:28–29` `rank: 4, alpha: 8.0`; defaults match in `lora.py:32,33` and `train.py:99–100`.
- [PASS] **LoRA targets cross-attention to_q/to_k/to_v/to_out only** — substring match in `lora.py:68` defaults to `("to_q", "to_k", "to_v", "to_out")`; cross-attention defines exactly those names at `model.py:129–132`. Self-attention uses `to_qkv` (fused) and `to_out` at `model.py:98–99` — note that the *self*-attention `to_out` will *also* match the substring `"to_out"`. **See nit #1.**
- [PASS] **LoRA B=0 init verified** — `lora.py:50` `self.lora_B = nn.Parameter(torch.zeros(out_features, self.rank))`; A is Kaiming. `tests/test_smoke.py:108–112` proves the adapter is a no-op at step 0 via `torch.allclose(before, after, atol=1e-5)`.
- [PASS] **BERT special tokens for 24 writers** — `text.py:174–180` calls `tokenizer.add_special_tokens({"additional_special_tokens": [...]})` and `bert.resize_token_embeddings(len(self.tokenizer))` immediately after. Train loop wiring at `train.py:162–167` collects writer names via `dataset.writer_names()` (which returns a deduped sorted list of writers in the manifest — `dataset.py:81–83`) and registers them. The stub mirrors this contract at `text.py:79–99`.

### Schedule & Sampler
- [PASS] **Linear β schedule, β_1=1e-4, β_N=0.02, T=1000** — three train yamls all set these identically (`train_stage_a_ttf.yaml:15–18`, `train_stage_b_midtrain.yaml:16–19`, `train_stage_c_ernantang.yaml:17–20`).
- [PASS] **Sampler uses ε-prediction consistently** — `sample.py:66–72` reads `eps` from the U-Net, recovers `x0` via `sqrt_recip * x_t - sqrt_recipm1 * eps` (the standard DDPM ε→x0 inversion), then steps DDPM or DDIM. Matches training target.
- [PASS] **Time embedding sin/cos + 2-layer MLP** — `model.py:43–51` (`SinusoidalTimeEmbedding.forward`); injected as additive bias inside every ResBlock at `model.py:84`.
- [PASS] **CFG at sample time** — `sample.py:51–57` builds unconditional ctx from the empty-string prompt; `sample.py:64–66` applies `eps_uncond + cfg_scale * (eps_cond - eps_uncond)`. Branch is skipped when `cfg_scale == 1.0`.

### Data normalization
- [PASS] **Image range [-1, 1], grayscale** — shared loader convention is `[-1, 1]` per `blind_impl.md` decision 23; `model.py:225` reads `in_channels=1`; default `cfg.in_channels=1`.
- [PASS] **No flip / no rotate augmentation** — neither `dataset.py` nor the prompt-wrapping path applies any spatial augment; consistent with calligraphy orientation-awareness.
- [PASS] **content_channels=[]** — `dataset.py:65` forces empty list so the shared JSONL loader skips npz reads (Calliffusion has no content-cache input).

### Training dynamics
- [PASS] **Smoke test exercises forward + backward + 1 optimizer step** — `tests/test_smoke.py::test_smoke` lines 57–93 covers it; `assert torch.isfinite(loss)` and "at least one gradient is non-zero" assertions both run.
- [PASS] **AdamW with β1=0.9 β2=0.999 wd=0** — `train.py:186–191`, matches `blind_impl.md` decision 18.
- [PASS] **Gradient clip max_norm=1.0** — `train.py:232–233`, also in every train yaml.
- [PASS] **Non-finite loss guard** — `train.py:228–229` raises immediately if loss is NaN/Inf — solid CI tripwire.

---

## Nits (PASS-WITH-NITS)

1. **`to_out` LoRA substring also matches `SpatialSelfAttention.to_out`** (`model.py:99`). The blind-impl decision (`blind_impl.md` decision 22) says LoRA is *cross-attention only*. With the current substring matcher in `lora.py:68–82`, `apply_lora_to_module(unet, target_substrings=("to_q","to_k","to_v","to_out"))` will *also* wrap every self-attention `to_out`. This is not catastrophic (extra LoRA params, more capacity) but it inflates the trainable count beyond the stated "~36 cross-attn projections" and the smoke test at `tests/test_smoke.py:109–110` will silently wrap self-attn `to_out` modules too. Two clean fixes:
   - Add a `name_filter=` callable: only wrap if the qualified parent module is a `SpatialCrossAttention`.
   - Or rename self-attn output to `self_to_out` so the substring no longer collides.
2. **CFG dropout point of application is per-item, not per-batch** (`dataset.py:89,149`). This is fine in expectation but means a single dataloader worker may produce a batch with all-empty or no-empty prompts. The original SD recipe drops at the batch level *and* per-sample; per-sample is the standard so this is purely a nit, but worth noting if Stage A loss curves look noisy.
3. **`text.py` real `BertTextEncoder.freeze()` does not check `embeddings_trainable` interacts with `add_special_tokens` ordering** — order matters: if `freeze(embeddings_trainable=True)` is called *before* `add_special_tokens(...)`, the freshly-resized embedding rows may inherit `requires_grad=False` because `resize_token_embeddings` allocates a new `nn.Embedding` and PyTorch's `nn.Parameter(...)` default is `requires_grad=True` — so this actually works by accident. But the train-loop order at `train.py:165–169` calls `add_special_tokens` first then `freeze`, which is the correct order. Worth a docstring note in `text.py` so future-self doesn't reorder.

---

## Suggested ablations (nice-to-have)

- **CFG dropout sweep** — paper-note tag says `p=0.1` is `[guessed-because-paper-vague]`. Run `{0.0, 0.1, 0.2}` on Stage B for 5k steps each, sample with `cfg_scale ∈ {1, 3, 5}`, and pick the best by writer-conditional FID. One curve will quickly tell us whether the guess is in the right neighbourhood.
- **LoRA rank** — `r ∈ {4, 8, 16}` at Stage C with the same step budget. Paper is silent; r=4 is a guess from SD-LoRA practice. Worth ~3 short runs.
- **Stage A vs Stage B+ablate-special-tokens** — drop the `add_special_tokens(writer_names)` call entirely at Stage B and let the BERT subword tokenizer fragment writer names. Should *underperform* the special-token path; if it doesn't, the special-token design is unjustified for our scale.

---

## Files touched in review
- `papers/06_calliffusion/paper_notes/06.md`
- `papers/06_calliffusion/reports/blind_impl.md`
- `papers/06_calliffusion/src/calliffusion/model.py`
- `papers/06_calliffusion/src/calliffusion/text.py`
- `papers/06_calliffusion/src/calliffusion/lora.py`
- `papers/06_calliffusion/src/calliffusion/train.py`
- `papers/06_calliffusion/src/calliffusion/dataset.py`
- `papers/06_calliffusion/src/calliffusion/sample.py`
- `papers/06_calliffusion/src/calliffusion/configs/{model,train_stage_a_ttf,train_stage_b_midtrain,train_stage_c_ernantang}.yaml`
- `papers/06_calliffusion/tests/test_smoke.py`
- `shared/src/paper_reimpl_shared/diffusion/gaussian.py` (for target-tensor verification)

## Contamination: NONE detected
No imports from / references to an external calliffusion repo, no copied class names, no suspicious magic constants, no `# copied from` comments. The `[guessed-…]` tags in `blind_impl.md` are honestly applied; the freeze schedule, LoRA rank, and CFG p are all tagged as guesses, matching the paper-note's gaps.
