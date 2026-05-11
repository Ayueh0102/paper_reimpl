# DL Review — 03_if_font — Gate 1 (blind implementation)

## Verdict: PASS-WITH-NITS

The implementation matches IF-Font's paper-cited architecture (VQGAN K=256
down-8 + 10-block AR Transformer with 2 self-attn + 1 cross-attn per block,
d=384, h=8), and all 8 smoke tests pass clean (no NaN under CFG drop,
gradient flows on every required branch, IDS conditioning flips logits).
The blind-impl decision log accurately marks paper-vague choices.

Three real concerns surfaced; none are blockers for Gate 1, but they should
be tracked into Stage B before any long run:

1. **Codebook EMA keeps updating in Stage B/C** — `train.py` never sets
   `model.vq.quantizer.eval()` or freezes VQ params, and EMA is gated only
   on `self.training`. Every Stage B/C step calls `quantizer.forward` once
   per target glyph + once per ref, so the codebook (and thus the AR
   target labels) drifts under the AR objective. Decay=0.99 means drift is
   slow but non-zero. This is a moving-target problem for CE on a fixed
   codebook.
2. **Ref-path gradient is fully detached from VQ** — the docstring at
   `src/if_font/model.py:658-660` claims gradient flows via straight-through,
   but `vq.encode()` returns only long indices and `quantizer.lookup()` is
   `F.embedding` on a *buffer* (the codebook is `register_buffer`, not a
   `nn.Parameter`). I verified empirically: backward from the ref-conditioned
   context produces zero gradient on `vq.encoder.*` and `None` on
   `refs.grad`. Functionally OK if the intent is "VQ is pre-trained, refs
   are a frozen tokenizer", but the comment is misleading and should be
   corrected.
3. **`encode_refs_to_tokens` triggers redundant EMA updates** — same root
   cause as (1). Refs are not training data for VQ; they shouldn't move the
   codebook. Fix in concert with (1).

Everything else on the rubric checked out: paper-cited CE eq. matches code
(`train.py:89-93`), reduction='mean' over [B,N] matches Ho-style scale,
EMA decay 0.99 + commitment 0.25 + straight-through implemented per
van den Oord 2017, BOS handling correct (token_embed has K+1 rows, head
has K outputs), causal mask is triu(diag=1), CFG drop has an explicit
all-key-masked guard at `model.py:396-400` that prevents the softmax NaN,
and the dropped IDS rows are still masked out at the decoder's cross-attn
level via `pad_mask`.

## Checked

### Loss correctness
- [✓] Loss formula matches paper eq.1 — `train.py:89-93` computes
      `F.cross_entropy(logits.reshape(-1, K), target_ids.reshape(-1),
      reduction='mean')` which is the `1/(B·N)` average in
      `paper_notes/03.md` §2 L_AR.
- [✓] Reduction = 'mean' — matches paper convention and is the right
      scale for `lr=2e-4` AdamW at paper's `batch=128`.
- [✓] Stage A/B/C weight schedule is implemented via
      `(ce_weight, vq_weight, recon_weight)` in `compute_loss`; YAMLs
      set Stage A = (0, 1, 1), Stage B/C = (1, 0, 0). Matches
      `paper_notes/03.md` §2.
- [✓] VQ commitment β=0.25 (`VQTokenizerConfig.commitment_weight`,
      `model.py:65`); straight-through is `quantized = z + (quantized -
      z).detach()` at `model.py:239`; codebook update is EMA in
      `model.py:220-231`. Matches van den Oord 2017 / Esser 2021.
- [✓] CFG dropout — `cfg_drop_prob=0.1` is set in stage_b / stage_c
      train YAMLs. `train.py:71-77` zeroes the IDS attention mask for
      `drop` rows, leaving refs intact. This is the correct CFG
      conditioning-drop convention.
- [✓] AR / diffusion target — not applicable; IF-Font is autoregressive
      (no β-schedule, no x0/ε prediction). `paper_notes/03.md` §1 calls
      this out and `blind_impl.md` decision 11 confirms.

### Gradient flow
- [✓] No accidental `.detach()` on the critical path. The two intentional
      detaches are correct: commitment loss `F.mse_loss(z,
      quantized.detach())` at `model.py:235`, and the straight-through
      trick at `model.py:239`.
- [✓] IDS conditioning path is alive — `test_if_font_ids_conditioning_path_active`
      proves changing IDS changes logits, and I separately verified
      `ids_encoder` receives gradient (norm ≈ 0.155) under CE loss alone.
- [✓] AR decoder receives strong gradient (≈ 8.5 norm in tiny config).
- [✓] EMA decay = 0.99 matches the "stable on small batches" claim in
      `blind_impl.md` decision 5.
- [✓] `clip_grad_norm_(model.parameters(), 1.0)` at `train.py:269` —
      standard AR transformer setting; `blind_impl.md` decision 17 flags
      it as paper-vague.
- [⚠] **Nit**: Ref path is gradient-disconnected from VQ encoder
      (`encode_refs_to_tokens` uses `vq.encode` → long indices →
      `quantizer.lookup` on a buffer). This is functionally fine when VQ
      is frozen in Stage B/C, but the comment at `model.py:658-660`
      saying "stays gradient-connected through straight-through estimator"
      is wrong — verified by backward + grad inspection. See backlog #2.

### Schedule & sampler
- [✓] No β-schedule needed (AR, not diffusion).
- [✓] AR sampler at `model.py:583-617` matches training-time convention:
      both use the same BOS token (`bos_index = K`), the same causal mask
      via `_causal_mask`, and the same `pos` embedding. Output is clamped
      to `[0, K-1]` before VQ decode to guard against any out-of-range
      sample (defensive but correct).
- [✓] Learned positional embedding is used (`_LearnedPositionalEmbedding`
      at `model.py:474`). Paper does not specify sin/cos vs learned —
      learned is the AR-transformer default. Flagged in `blind_impl.md`
      indirectly (architecture is paper-vague).
- [✓] Training and inference both use full `n_tokens = 256` decode steps;
      `paper_notes/03.md` §6 documents this.

### Data normalisation
- [✓] Image range — `paper_reimpl_shared` synthetic batch produces
      `[-1, 1]` tensors and the model treats them as such. Smoke tests
      backward-propagate fine.
- [✓] Augmentation — none in this scaffold yet (manifest dataset is
      Stage 1 plumbing). Calligraphy-safe defaults are inherited from
      `paper_reimpl_shared.data.legacy`.
- [✓] Content cache axis — IF-Font does **not** use a source-glyph
      bitmap channel (paper's headline claim is source-glyph-free; IDS
      replaces content). `paper_notes/03.md` §4 makes this explicit.

### Conditioning paths
- [✓] **IDS tokenization correctness** — `IDSTokenizer` (ids.py:81-226)
      character-tokenizes the IDS string: PAD/BOS/EOS/UNK at ids 0-3, the
      12 U+2FF0..U+2FFB IDCs at ids 4-15, leaf CJK chars appended via
      `add_token`/`fit_from_strings`. `STRUCTURE_NAMES` mirrors
      `~/Char/datasets/ids/scripts/lookup_ids.py` and supports all 12
      IDCs. The smoke test round-trips `⿰示畐` and `⿱艹⿴口十`. Encoding
      is per-Unicode-char with UNK fallback. This is correct for the
      IF-Font paper's "12 IDC + leaf components" definition.
- [✓] **VQ codebook EMA + straight-through** — both implemented in
      `VectorQuantizer.forward` (model.py:206-240). Codebook entries are
      EMA buffers updated via cluster_size + embed_avg accumulators with
      `cluster_size_norm` renormalisation per van den Oord eq. 8.
      Straight-through is `quantized = z + (quantized - z).detach()` at
      line 239 — the canonical form.
- [✓] **AR transformer cross-attn path** — `_DecoderBlock` (model.py:422-466)
      runs `n_self_attn_per_block=2` causal self-attn passes then one
      cross-attn over `context = [ids_ctx ; ref_ctx]` built in
      `build_context` (model.py:678-715). Cross-attn pad mask is the
      concatenation `[ids_mask, ref_mask]`, so CFG-dropped IDS rows
      receive zero attention weight on the IDS portion but still attend
      to refs. Verified by `test_if_font_ids_conditioning_path_active`.
- [✓] **Training CE reduction** — `reduction='mean'` over `[B·N]`
      positions matches Ho-style "mean over batch and spatial". Paper
      eq. L_AR has the same `1/(B·N)` normalisation.
- [✓] **CFG dropout does not NaN the encoder** — confirmed by
      `test_cfg_dropout_does_not_nan` and inspection of the safety guard
      at `model.py:396-400`: when an entire row's `key_padding_mask` is
      all-False (i.e. CFG drop zeroed the IDS mask), the attention layer
      unmasks position 0 for that row only, keeping the softmax finite.
      Downstream, the decoder's cross-attn `pad_mask` for that row is
      also all-False on the IDS slice, so the garbage IDS encoder output
      is masked out at consumption. Net effect: dropped-IDS rows are
      ref-only. Correct.

### Training dynamics
- [✓] Smoke test runs 1 optimizer step without NaN (8/8 pass).
- [✓] EMA decay 0.99 is in the standard band; tests `model.eval()`
      before any same-input comparison test to avoid drift confounds.
- [✓] Paper batch=128 vs our `train_stage_b: batch_size=32`,
      `train_stage_c: batch_size=16` — `blind_impl.md` decision 14
      documents this and the linear-lr-rule comment in
      `paper_notes/03.md` §6 covers it. lr in the YAMLs is 2e-4 / 5e-5
      which is in the right band for the smaller batch.
- [⚠] **Concern (codebook EMA in Stage B/C)** — see backlog #1.

## Suggested fixes (none required for Gate 1; track for Stage B)

These are not Gate-1 blockers but they should be addressed before the
Stage B mid-train launch.

1. **Disable codebook EMA outside Stage A.** Two options:
   - Add `if vq_weight == 0 and recon_weight == 0: model.vq.eval()` in
     `train.py:main()` before the training loop, OR
   - Add an explicit `freeze_vq: bool` flag to the train YAMLs that
     calls `for p in model.vq.parameters(): p.requires_grad_(False)`
     **and** sets `model.vq.quantizer.eval()` (or guards EMA on a
     separate `self.update_codebook` flag).
   `train_stage_b_midtrain.yaml:21-22` and
   `train_stage_c_ernantang.yaml:16-17` already set vq/recon weights
   to 0, but the codebook still moves because EMA is gated only on
   `self.training`. Fix targeted at `model.py:220` (add a runtime
   `if self.codebook_frozen: skip the no_grad block`) is the cleanest.

2. **Correct the misleading ref-path docstring.** `model.py:658-660`
   says "stays gradient-connected through the quantizer's straight-
   through estimator". This is false — `vq.encode` returns `indices`
   (long, no grad) and `quantizer.lookup` uses the codebook *buffer*
   via `F.embedding`. Either:
   - Rewrite the docstring to say "ref tokenization is gradient-
     disconnected by design; VQ is treated as a frozen tokenizer in
     Stage B/C", OR
   - If joint VQ + AR is desired, return `quantized` from
     `quantizer.forward` instead of running through `lookup(indices)`,
     and switch the codebook from `register_buffer` to `nn.Parameter`
     (this would also need a learning-rate change to avoid catastrophic
     codebook collapse during AR fine-tune).

3. **Smoke test for codebook stability under Stage B settings.** Add a
   regression test that runs `compute_loss` with `vq_weight=0,
   recon_weight=0` for 5 steps and asserts the codebook L2-distance
   from its initial state stays below a tolerance — would have caught
   the EMA-always-on issue.

## Nice-to-have (PASS-WITH-NITS)

- **Coverage of structure ablation** — `parse_structure_class` correctly
  handles the 12 IDC structure names + "atomic" + "unknown" but is not
  wired into any conditioning side-channel. A small "structure-id-only"
  baseline that strips IDS to its leading IDC would isolate the
  structural-component contribution from the leaf-token contribution.
- **KV cache for `TransformerARDecoder.sample`** — currently recomputes
  the full prefix every step (acknowledged in `blind_impl.md` known
  gaps). Fine for smoke; fix before any 256-token eval. Expected speed-up
  is roughly Lᵢ → 1 per step, so ~50× wall-clock on a 256-token decode.
- **Top-k / top-p sampling** — current sampler is temperature
  multinomial; add `top_k` / `top_p` for evaluation runs.
- **AdamW betas hard-coded** — `train.py:231` pins `betas=(0.9, 0.95)`
  inside `main`, not exposed in YAML. Add a `betas:` key to the train
  YAML for consistency with the rest of the repo.

## Suggested ablations (optional, for Stage B/C)

1. **Codebook size 256 vs 512 vs 1024** — paper-cited K=256 with our
   84-unit Ernantang subset means roughly 3 chars per codebook entry;
   K=512 might give better stroke-level resolution while still
   tractable. Stage A pretrain is the right place to test this.
2. **IDS encoder depth 2 vs 4 vs 6** — paper does not pin this
   (`blind_impl.md` decision 6 says we used 2). Going deeper costs almost
   nothing relative to the 10-block AR decoder and may help structural
   composition.
3. **CFG drop p ∈ {0.0, 0.1, 0.2}** — paper does not pin
   (`blind_impl.md` decision 8). p=0.0 should produce strictly tighter
   IDS adherence at inference; p=0.2 gives more sample diversity.
4. **Reference packing — concat vs interleaved vs separator-token** —
   `blind_impl.md` decision 7 flags this as a free choice. Concat is the
   simplest baseline.
5. **AR scan order — raster vs Z-order vs diagonal** — `blind_impl.md`
   decision 10. Raster is the default; diagonal can help in image
   modelling per Esser 2021.

## Contamination check

**NOT CONTAMINATED.** I did not consult any reference implementation
(no read of `github.com/Stareven233/IF-Font` or any other public
codebase). The review is based on `paper_notes/03.md`, `reports/blind_impl.md`,
the `src/if_font/` source tree, and the project rubric only. The
implementation contains explicit `[guessed-*]` annotations on all
paper-vague choices and the decisions match well-known defaults
(VQ-VAE / VQGAN / Ho-Salimans CFG / Esser 2021) without lifting any
code or hyperparameter that wasn't independently justifiable.
