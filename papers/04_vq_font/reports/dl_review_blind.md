# DL Review — 04_vq_font — Gate 1 (blind implementation)

## Verdict: PASS-WITH-NITS

Reviewer: DL Reviewer (Gate 1, blind-impl audit). Contamination scan:
**CLEAN** — no references to `Yaomingshuai/VQ-Font` repo, no
`third_party/` consult, no peek markers in source.

---

## Special-focus items (caller-requested)

### 1. VQGAN codebook commitment + L1 recon — PASS

- Commitment + codebook loss recipe matches Van den Oord (2017) /
  Esser et al. (2021):
  `src/vq_font/vqgan.py:203-205`
  ```python
  codebook_loss   = F.mse_loss(z_q_flat, z_flat.detach())
  commitment_loss = F.mse_loss(z_flat, z_q_flat.detach())
  loss = codebook_loss + self.commitment_weight * commitment_loss
  ```
  β = 0.25 default (`vqgan.py:174`) matches VQ-VAE standard, paper-note line 73.
- Straight-through gradient implemented at `vqgan.py:208`:
  `z_q_flat = z_flat + (z_q_flat - z_flat).detach()` — encoder gets
  identity gradient, decoder operates on quantized features. Smoke
  test `tests/test_smoke.py:81-86` verifies encoder/decoder/codebook
  all receive gradient.
- L1 reconstruction at `src/vq_font/train.py:118`:
  `loss_recon = F.l1_loss(out.recon, x, reduction="mean")` with
  `reduction="mean"` consistent throughout. Total composed at
  `train.py:120`: `total = recon_weight * loss_recon + vq_weight * loss_vq`.
- Matches paper-note loss eq.131-133 (`L_vqgan = L_recon (L1) + λ_vq · L_vq`).

Nit: GAN / perceptual term intentionally omitted; flagged in
`blind_impl.md` decisions 3-4 as `[guessed-because-paper-vague]`.
Acceptable Gate 1 scope; Phase 2 diff target.

### 2. Transformer cross-entropy on codebook indices — PASS

- Target indices computed under no_grad with frozen VQGAN at
  `src/vq_font/train.py:161-162`:
  ```python
  with torch.no_grad():
      target_indices = model.encode_target_indices(target)
  ```
  via `model.py:103-106` (no_grad-decorated `encode_target_indices`).
- Per-token CE at `train.py:166-170`:
  ```python
  loss_token = F.cross_entropy(
      token_logits.reshape(b * n, k),
      target_flat.reshape(b * n),
      reduction="mean",
  )
  ```
  Reduction="mean" averages across `B*N` token positions, which is the
  standard recipe and the same scale as `log(K) ≈ 6.93` for K=1024.
  Smoke run logs `token_ce=7.1104` for K=1024 random init — matches
  `log(1024) = 6.93` within init noise. Plumbing confirmed.
- Vocab size = `codebook_size = 1024`, enforced by
  `model.py:62-67` (`VQFontConfig.__post_init__` cross-checks
  `transformer.codebook_size == vqgan.num_embeddings`).
- `token_logits` shape `[B, N, K]` (with `N = H_lat * W_lat = 256`
  for 16×16 grid) is consistent with `target_indices` shape
  `[B, H_lat, W_lat]` after `target_flat.reshape(b * n)`.

### 3. SSEM 14-class matches lookup_ids.parse_structure — PASS

- `parse_structure` (`~/Char/datasets/ids/scripts/lookup_ids.py:78-81`)
  returns one of:
  - `unknown` (empty IDS)
  - one of 12 structure names from `STRUCTURE_NAMES` dict (lines 30-43)
  - `atomic` fallback (any other leading char)

  ⇒ exactly **12 + 2 = 14** distinct output strings, matching
  `NUM_STRUCTURE_CLASSES = 14` at `src/vq_font/transformer.py:47`.
- `STRUCTURE_NAME_TO_ID` table at `src/vq_font/dataset.py:42-57`
  enumerates all 14 strings explicitly and is asserted at line 58
  with `assert len(STRUCTURE_NAME_TO_ID) == NUM_STRUCTURE_CLASSES`.
  Cross-reference to `lookup_ids.STRUCTURE_NAMES` is one-to-one (12
  entries) plus `atomic` and `unknown` sentinels. **Names match
  bit-for-bit** (`left_right`, `top_bottom`, `left_mid_right`,
  `top_mid_bottom`, `surround_full`, `surround_open_bottom`,
  `surround_open_top`, `surround_open_right`, `surround_open_TR`,
  `surround_open_TL`, `surround_open_BR`, `overlap`).
- Aux CE at `train.py:171`:
  `loss_struct = F.cross_entropy(structure_logits, structure_id, reduction="mean")`.
  14-way head at `transformer.py:281-283`. Total loss
  `total = loss_token + structure_weight * loss_struct` at
  `train.py:172` (λ_struct = 0.1 in `train_stage_a_ttf.yaml:19`).
- SSEM injection has **two** grad paths (acknowledged in `blind_impl.md`
  decision 14) — additive bias on every query token
  (`transformer.py:346`: `q_tokens = q_tokens + struct_emb.unsqueeze(1)`)
  plus structure prefix prepended to cross-attn context
  (`transformer.py:351-352`: `ref_tokens = torch.cat([struct_prefix, ref_tokens], dim=1)`).
  Smoke test `test_smoke.py:136-141` verifies `struct_encoder`
  receives non-zero gradient — confirming both paths are live.

### 4. Stage 0 (VQGAN pretrain) vs Stage A+ (transformer) freeze handling — PASS

- Stage 0 (`train.py:244-303`) builds bare `VQGAN` (no `freeze_vqgan`
  wrapper), trains *all* its parameters; AdamW receives
  `model.parameters()` directly at line 254-257. Correct.
- Stage A+ (`train.py:306-390`) builds `VQFont(cfg, freeze_vqgan=True)`
  at line 312:
  - `model.py:92-95` sets `requires_grad = False` on every VQGAN
    parameter and calls `self.vqgan.eval()`.
  - Optimizer is built from `[p for p in model.parameters() if
    p.requires_grad]` at `train.py:328-332` ⇒ only Transformer
    parameters update.
  - `model.train()` then `model.vqgan.eval()` at `train.py:349-350`
    keeps VQGAN modules in eval mode (cosmetic-only since they use
    GroupNorm + zero dropout, but the intent is correct).
- Smoke test `test_smoke.py:120-121` asserts `requires_grad = False`
  for every VQGAN parameter; lines 154-155 assert `p.grad` is None
  or zero after backward. Both pass per the blind_impl verification
  log (line 226).
- VQGAN encoder call sites for the Transformer path are wrapped in
  `@torch.no_grad()` (`model.py:97-101`) — gradient is correctly
  cut at the VQGAN/Transformer boundary; Transformer's own
  `input_proj`, `ref_proj`, attention blocks, struct_encoder, and
  heads remain trainable and grad-connected, confirmed by smoke
  test gradients at `test_smoke.py:142-152`.

---

## Rubric checklist

### Loss correctness
- [✓] **loss formula matches paper** — Stage 0 L1 + commitment β at
  `src/vq_font/train.py:118-120` + `src/vq_font/vqgan.py:203-205`;
  Stage A+ token CE + λ_struct·CE at `train.py:166-172`.
- [✓] **reduction='mean'** consistent on every loss term
  (`train.py:118, 166-171`).
- [✓] **weight combination** — `recon_weight`, `vq_weight`,
  `structure_weight` plumbed through YAMLs; values 1.0/1.0/0.1 for
  Stage A. Paper-vague on λ_struct (decision 15) — flagged.
- [n/a] **diffusion target / β-schedule / CFG dropout** — VQ-Font is not
  diffusion; these items skip cleanly.

### Gradient flow
- [✓] **no detach on critical path** — only the two VQ-VAE-mandated
  `.detach()` calls in `vqgan.py:203-208` (codebook/commitment +
  straight-through), and the read-only `.detach().cpu()` in log
  dict builders (`train.py:122-124, 178-181`). No critical-path
  detach.
- [✓] **conditioning paths connected** — SSEM additive bias
  (`transformer.py:346`), SSEM prefix in cross-attn context
  (`transformer.py:351-352`), reference path via cross-attn K/V on
  `ref_proj`-projected ref tokens (`transformer.py:350, 362`).
  Smoke test asserts all three.
- [✓] **frozen VQGAN no leaks** — verified by smoke + manual code review.
- [n/a] **EMA / zero-init AdaLN** — paper does not require either.
- [✓] **grad clip = 1.0** — set in both stage YAMLs
  (`train_stage_0_vqgan.yaml:15`, `train_stage_a_ttf.yaml:15`) and
  honored at `train.py:284-285, 360-363`.

### Schedule & sampler
- [✓] **codebook size** = 1024 enforced at three layers (config,
  model assert, dataset structure-id alignment). Paper-cited.
- [✓] **sampler decode** — argmax + temperature/top-k modes in
  `src/vq_font/sample.py:35-63`, decode through `decode_indices`
  (`vqgan.py:378-389`) — consistent with training-time CE-over-
  codebook objective (no train/infer convention mismatch).
- [n/a] **time embedding / β schedule** — not a diffusion model.

### Data normalisation
- [✓] **image range** — synthetic batches in
  `paper_reimpl_shared.runner.smoke.make_synthetic_batch` produce
  tensors compatible with VQGAN L1 reconstruction. Dataset reads
  via `CalligraphyJsonlDataset` (parent class).
- [✓] **no horizontal flip / large rotation** — no augmentation
  pipeline in this paper's dataset wrapper.
- [✓] **content cache axis** — content channel order driven by
  `data_cfg.content_channels: [bitmap]` (`data_stage_a.yaml:15`);
  guard against multi-channel content at `train.py:149-150`.

### VQ-Font row-specific (rubric line 51)
- [✓] **VQGAN codebook frozen** — `_freeze_vqgan` at
  `model.py:92-95`, smoke-tested.
- [✓] **transformer predicts codebook index, not raw pixel** —
  `token_head = nn.Linear(embed_dim, codebook_size)` at
  `transformer.py:280`; loss is CE against
  `vqgan.encode_indices(target)` (`train.py:161-162, 166-170`).

### Training dynamics (verified in blind_impl smoke run)
- [✓] **no NaN** — `loss=1.2357 recon=1.0831 vq=0.1526` Stage 0
  step 0; `loss=7.3820 token_ce=7.1104` Stage A step 0; both
  finite, both within expected range (token_ce ≈ log(K)=6.93).
- [n/a] **EMA decay** — none.
- [✓] **batch_size + lr** = paper-cited (32 + 4e-5 Stage 0, 32 + 2e-4
  Stage A); no linear-scaling discrepancy.

---

## Suggested fixes — FAIL items

**None.** No FAIL items found.

---

## Nice-to-have (PASS-WITH-NITS)

1. **Wire `parse_structure` fallback at dataset load.**
   `src/vq_font/dataset.py:67-77` only reads pre-baked
   `row['structure_id']` / `row['structure']`. If manifests don't
   ship the field (a known open issue per `blind_impl.md` open
   question 1), every sample silently routes to `0 = unknown` and
   the SSEM gradient becomes degenerate. Concrete fix: insert a
   final fallback at `dataset.py:77`:
   ```python
   char = row.get("char") or row.get("target_char")
   if isinstance(char, str) and char:
       from lookup_ids import get_ids, parse_structure  # adjust import
       return STRUCTURE_NAME_TO_ID.get(parse_structure(get_ids(char)), 0)
   return 0
   ```
   Acceptable as Gate-1 nit because:
   (a) blind_impl.md acknowledges it as open question 1, and
   (b) the smoke-test path uses `VQFontSyntheticDataset` which
   already cycles ids correctly (`dataset.py:96-99`).

2. **Latent-grid shape mismatch in `VQGANConfig.out_resolution()`
   docstring.** `vqgan.py:75-86` docstring says `channel_mult=(1,2,4)`
   yields 32×32 and needs `(1,1,2,4)` for the paper's 16×16. The
   actual default in `configs/model.yaml:15` is already
   `[1, 1, 2, 4]` → 16×16, so this is just a stale comment, but the
   `VQGANConfig` dataclass default at `vqgan.py:60` is still
   `(1, 2, 4)` which contradicts the model.yaml. Recommend syncing
   the dataclass default to `(1, 1, 2, 4)` to match paper.

3. **`channel_mult` semantics surprise.** The code does
   `len(channel_mult) - 1` stride-2 downsamples (last stage is
   `Identity`) rather than `len(channel_mult)` — both encoder
   `vqgan.py:253-256` and `out_resolution` `vqgan.py:86`. This is
   internally consistent but non-obvious; a 1-line comment near the
   loop helps future readers.

4. **Stage A loss-form transparency.** `transformer_compute_loss`
   uses `initial = batch.get("content")` as the Phase 1 stand-in for
   the "initial synthesized glyph". Loss is correct *but*
   accidentally lets `content` and `target` be the same image
   (Stage A synthetic), which makes the task trivially
   `encoder→indices ≈ encoder→indices`. Smoke acceptance is fine
   (we only check finite loss + grad flow); Stage B handover note
   should flag this so the real synthesis-module checkpoint is
   loaded before declaring Stage A results.

---

## Suggested ablations (optional)

- **λ_struct sweep** {0.0, 0.05, 0.1, 0.2, 0.3} on Stage A to
  confirm SSEM is helping rather than hurting (paper gives no
  weight, our 0.1 is a guess).
- **SSEM injection ablation** — bias-only vs prefix-only vs both
  vs none. The two grad paths might be redundant or one might
  dominate; data answers it cheaply.
- **CE vs contrastive SSEM loss** — listed in `paper_notes/04.md`
  open question 5. Phase 2 diff target.

---

## Contamination scan: CLEAN

`grep -rn -i "Yaomingshuai\|third_party\|github.com/Y\|official.*VQ-Font"`
on `src/` returns zero hits. `third_party/` directory does not
exist in `paper_reimpl/`. `blind_impl.md` source list (lines 9-17)
explicitly enumerates only Obsidian note + Phase 0 spec + shared
helpers + lookup_ids — no peek detected.
