# GitHub Diff — 03_if_font

**STATUS**: cloned
**Date checked**: 2026-05-11
**Paper**: Chen et al., "IF-Font: Ideographic Description Sequence-Following Font Generation", NeurIPS 2024
**Official repo**: https://github.com/Stareven233/IF-Font
**Official commit SHA**: `0d8864d93ae27ad1705a5157938ddea1641baafa` ("Update the NeurIPS link and citation")
**Local clone path**: `/Users/Ayueh/Char/paper_reimpl/third_party/03_if_font/`

Our blind reimpl flagged 13 `[guessed-*]` entries in `papers/03_if_font/reports/blind_impl.md`. This document confirms or refutes each via official source.

---

## arch_deltas

The official architecture is **substantially different** from what `paper_notes/03.md` describes; the paper note appears to have over-simplified the system. The blind reimpl inherited those simplifications and is missing several major submodules.

### A1. Decoder block: 1 self-attn + 1 cross-attn, **not 2 + 1** [P0]

- Official `iffont/modules/nanogpt.py:152-163` (`Block2.forward`):
  - `x = x + self.attn(self.ln_1(x))` — one self-attn
  - `x = x + self.attn_cross(self.ln_3(x), style)` — one cross-attn
  - `x = x + self.mlp(self.ln_2(x))` — one FFN
- Blind: `src/if_font/model.py:437-473` (`_DecoderBlock`) iterates `self_attns` `n_self_attn_per_block` times, default **2**, then 1 cross-attn, then FFN.
- Blind note `paper_notes/03.md:52-53` and `:103` and `:133` repeatedly claim "2 self-attn + 1 cross-attn per block" — this is paper-note wrong, not just blind-impl wrong. The cited authority "training-config block" in our notes does not appear anywhere in the official code or README. Recommend: set `n_self_attn_per_block=1` (or remove that knob).

### A2. Pretrained external VQGAN, **not trained-from-scratch** [P0]

- Official uses `CompVis` `vq-f8-n256` checkpoint downloaded externally and **frozen** (`iffont/data/adapter.py:33-55`, `VQAdapter._load_vqgan` calls `model.freeze()` and overrides `model.train` to a no-op). README §"Data Preparation Step 3" instructs the user to download `f=8, VQ (Z=256, d=4)` from CompVis.
- Tokens are pre-computed into an HDF5 file before training (`iffont/data/datasets_h5.py:22-46` `dataset2h5`); the training loop only sees integer token grids, never raw images.
- VQGAN params: `n_embed=256`, `embed_dim=4`, `image_size` is whatever the pretrained model expects (RGB 128×128 → 16×16 token grid → 256 tokens).
- Blind: trains its own VQGAN from scratch (`model.py:VQEncoder/VectorQuantizer/VQDecoder`) with `embedding_dim=256` (`model.py:63`), `in_channels=1` (grayscale), and a hand-rolled EMA codebook. This is wrong on three dimensions:
  - **No reconstruction parity guaranteed** — the pretrained CompVis model has been trained on OpenImages with adversarial + perceptual losses, ours has only commitment + MSE.
  - **Channel mismatch**: blind is `embedding_dim=256` per codebook vector; official is `embed_dim=4`. The AR model's projection from codebook-embed → `d_model=384` is hugely different.
  - **Color mismatch**: blind is `in_channels=1`; official is RGB (`img_mode: RGB` in `iffont/config/base.yaml:96`).
- Blind has no Stage-A-of-VQGAN-training fallback that mimics CompVis training (it would need a discriminator + perceptual loss + ~OpenImages). Practical guidance: download CompVis `vq-f8-n256` and wrap as a frozen tokenizer; do not train VQ from scratch.

### A3. Missing module: **StyleEncoder + 3SA (Structure-Style Aggregation)** [P0]

- Official has a dedicated `StyleEncoder` (`iffont/modules/encoder.py:544-622`) that:
  1. Looks up VQ-quantized embeddings for each ref glyph (`encode_indices`).
  2. Passes them through a custom CNN stem (`_init_enc`).
  3. Weighted-averages refs by an IDS-coverage similarity score (`x_g = einsum(..., sim.softmax(dim=1))`, `encoder.py:610`).
  4. Runs a cross-attention from IDS-token queries to ref-feature keys/values (`_structure_style_aggregation`, `encoder.py:587-599`) — this is the "3SA" block of the paper.
  5. Concatenates the structure-aware style features `x_l` with the globally-averaged style `x_g` along the sequence dim (`encoder.py:614-616`).
- Blind: encodes refs with the VQ encoder, flattens, projects with a single `nn.Linear` (`model.py:665-684`, `encode_refs_to_tokens`). No similarity weighting, no IDS-conditioned cross-attention, no structure-style fusion. The decoder's cross-attention sees a flat `[B, N*256, d_model]` ref-token sequence with no inductive bias toward "which ref token matches which IDS slot".

### A4. Missing module: **MoCo / contrastive head** [P0]

- Official wraps the StyleEncoder in `MoCoWrapper` (`iffont/modules/encoder.py:647-716`):
  - Two style encoders, query `enc` and momentum-updated `enc_m` (cosine-schedule momentum, `momentum_update`, `encoder.py:689-695`).
  - A 2-layer MLP projector + predictor (`_build_projector_and_predictor_mlps`, `encoder.py:678-683`), with last BN, output dim 256.
  - Outputs both the style sequence `x_sss` and a contrastive feature pair `(cl_q, cl_m)` per sample.
  - In `Net2NetModel.training_step` (`models/net2net_model.py:181-198`), a supervised-contrastive loss `losses.sup_cl` is computed across a `CacheManagerCL` queue (size 10) of past (cl_q, font_id) pairs, optionally all-gathered across DDP ranks.
- Blind: completely absent. No contrastive head, no momentum encoder, no cache queue, no `sup_cl`. The blind has no `font_id` / writer-id signal anywhere in the model.
- Without the contrastive loss, the style branch will collapse to a content-leakage pathway in any non-trivial dataset. This is a P0 deviation if we ever train on real fonts.

### A5. IDS encoder: **just an `nn.Embedding`, no Transformer encoder** [P1]

- Official `IDSEncoder` (`iffont/modules/encoder.py:161-388`) is conceptually just `nn.Embedding(num_tokens, n_embd)` × 2 (one for AR cross-attention `ids_embed`, one for style-aggregation `ids_embed2`) plus the IDS-tree resolution machinery (`query_ids`, `triplet_equal_ids`, `level_analysis`). The previously-considered `nn.TransformerEncoder` is commented out (`encoder.py:198-201`).
- Blind: full 2-layer Transformer encoder with self-attention, FFN, positional embedding (`model.py:497-544`, `IDSTextEncoder`). This is a significant over-engineering on our side — not necessarily a bug, but inflates parameter count and may obscure the official inductive bias.
- The duplicated `embedding` and `embedding2` are because the AR decoder consumes raw token embeddings as the prefix-prepended context (no encoder), while the style-aggregation block needs a separately-learned IDS embedding for its 3SA cross-attention.

### A6. Conditioning is **prefix-prepended to AR sequence**, not separate cross-attention input [P1]

- Official `GPT.forward` (`iffont/modules/nanogpt.py:229-253`):
  - Concatenates `embeddings` (= IDS embed) in front of `tok_emb`: `tok_emb = torch.cat((embeddings, tok_emb), dim=1)`.
  - `block_size = 290 = 256 (target tokens) + 35 (IDS max_len) - 1 (the right-shift)`.
  - The decoder's `style` argument (= `x_sss` from StyleEncoder, = ref-derived) is consumed by `Block2.attn_cross` as a separate cross-attention K/V.
  - At the end, `logits[:, ids_embed.shape[1]-1:]` is sliced off so only the 256 image-token positions contribute to CE (`net2net_model.py:99`).
- Blind: never prepends IDS to the target sequence; instead, IDS-encoder output is **concatenated with ref tokens** and fed as cross-attention K/V (`model.py:723`). Causal mask only covers target positions, so IDS attends bidirectionally to itself anyway, but the *prefix-prepended* prediction loss difference is non-trivial — the official decoder gets IDS tokens at each causal step as autoregressive context, blind gets them as cross-attention only.

### A7. Per-head **QK-LayerNorm** + **DropKey attention mask** [P1]

- Official `CausalSelfAttention.__init__` (`nanogpt.py:55-56`) and `CrossAttention.__init__` (`nanogpt.py:91-99`) both register `ln_q` and `ln_k` — `nn.LayerNorm(head_dim)` applied to Q and K per head before SDPA (`nanogpt.py:66, 109`). This is the "QK-LayerNorm" stabilisation trick.
- Official also uses `DropKeyMask` (`nanogpt.py:54, 99`, `modules/blocks.py:8-40`): Bernoulli-sampled key-drop mask at attention training time.
- Both also use `F.scaled_dot_product_attention` (PyTorch ≥ 2.0 flash-attn path).
- Blind: hand-rolled `_MultiHeadAttention` (`model.py:357-411`) with `softmax + matmul`, no QK-LN, no DropKey. Numerically slower and less stable at d_model=384 + 10 blocks.

### A8. Weight tying between `wte` (token embed) and `lm_head` [P1]

- Official: `self.transformer.wte.weight = self.lm_head.weight` (`nanogpt.py:197`).
- Blind: separate `token_embed` (`model.py:561`) and `head` (`model.py:567`). Not bug per se but doubles the output-head parameter count and is a known minor regression in AR LM perplexity. The blind impl `blind_impl.md` "Known gaps" section already lists this — confirmed missing.

### A9. Per-block LayerNorm placement (Pre-LN) ✓ matches

- Both apply LayerNorm before the sublayer (Pre-LN). Official `Block2.forward` (`nanogpt.py:159-163`) matches blind `_DecoderBlock.forward` (`model.py:455-473`).

---

## loss_deltas

### L1. **Two losses in training, not three** [P0]

- Official `Net2NetModel.training_step` (`net2net_model.py:194-198`):
  - `l_sq = losses.sq(logits, x)` — cross-entropy on next-token VQ index (`losses.py:20-23`, `sq` = "soft-quantize" loss).
  - `l_cl = losses.sup_cl(cl_s, labels=font_id) / 2` — supervised contrastive on the MoCo-cached style features (`losses.py:97-164`).
  - **No VQ commitment loss**, **no reconstruction MSE** — VQGAN is frozen, so neither is needed.
- Blind: `compute_loss` (`train.py:46-104`) takes a (ce, vq, recon) weighted sum; YAML knobs `ce_weight`, `vq_weight`, `recon_weight`. Stage-A YAML re-trains VQGAN from scratch with `vq_weight=1`/`recon_weight=1`/`ce_weight=0`.
- Action: when we adopt a pretrained VQGAN, drop the VQ + recon terms entirely, and add `sup_cl` over a font-id batch.

### L2. **No CFG dropout in official** [P1]

- Official has no classifier-free guidance dropout. The model always conditions on both IDS and refs; sampling uses `top_k` only (`nanogpt.py:381-404`, `GPT.generate`).
- Blind: adds CFG with `cfg_drop_prob=0.1` (`blind_impl.md` item 8, `train.py:53,72-78`). This is a Ho-2022 reflex that the paper does not call for.
- Action: keep CFG behind a config flag and clearly mark as "diverging from paper" if used.

### L3. `losses.z_loss` available but unused [P3]

- Official defines a `z_loss` for stability (`losses.py:6-12`) but does not call it from `training_step`. Mention only.

---

## schedule_deltas

### S1. Optimizer and schedule **OneCycleLR + AdamW**, betas (0.9, 0.95) for transformer / (0.9, 0.999) for the rest [P1]

- Official `configure_optimizers` (`net2net_model.py:223-247`):
  - Two AdamW groups:
    - `netTransformer` parameters: AdamW (decay=0.01 from `weight_decay = 0.01` literal at `:224`), betas=(0.9, 0.95).
    - Everything else (ids_encoder, moco_wrapper): added as second group, betas=(0.9, 0.999) (`:233-236`).
  - Scheduler: `lr_scheduler.OneCycleLR(max_lr=self.learning_rate, total_steps=trainer.estimated_stepping_batches, pct_start=0.5/max_epochs, final_div_factor=10/25)` (`:238-244`). This is **OneCycle**, not warmup-cosine.
  - `model.learning_rate = accumulate_grad_batches * bs * base_lr` (`run.py:23-25`); `base_learning_rate: 4.5e-06` (`config/base.yaml:3`). With `bs=128` and `accumulate=1`: peak lr ≈ `5.76e-4`. With `train.yaml`'s `batch_size: 128` and `max_epochs: 15`.
  - `pct_start = 0.5 / 15 ≈ 0.033` (warmup is the first 3.3% of total steps).
- Blind: assumes "AdamW + warmup + cosine" (`blind_impl.md` item 14), no OneCycle code in train.py.

### S2. **Train batch_size = 128, max_epochs = 15** ✓ matches paper note

- Official `config/train.yaml:34` (`batch_size: 128`) and `config/train.yaml:6` (`max_epochs: 15`).
- Blind train YAML uses 16/32 due to small-GPU constraint — documented.

### S3. **`use_compile: False`**, **mixed-precision 16** [P3]

- `config/base.yaml:4` `use_compile: False`, `config/base.yaml:12` `precision: 16-mixed`. Blind has no compile/AMP wiring.

---

## conditioning_deltas

### C1. Refs are **pre-tokenized** (HDF5 of indices), not raw images [P1]

- Official `IFFontDataset.__getitem__` (`datasets_h5.py:117-135`) returns:
  - `x_idx`: target VQ tokens `[256]`
  - `c_idx`: ref VQ tokens `[n_ref, 256]`
  - `font_id`, `x_ch`, `c_ch`, `x_font`
- The model never touches raw pixels at training time. Cuts I/O and is the only way to make `bs=128` fit on a single GPU.
- Blind: dataset returns raw images; VQ encode runs in the model forward (`model.py:679-684`). For our Phase 3 this is fine for smoke tests but should switch to pre-tokenized for real training.

### C2. **Reference selection**: random `num_refs` from `train_ch` (the "seen" character subset), **not the query char** [P1]

- Official `datasets_h5.py:122-126`:
  ```py
  c_ch = tuple(random.sample(self.corpus_seen, k=self.num_refs))
  c_cid = tuple(map(lambda i: self.global_cid[i], c_ch))
  # if c_cid[0] == cid:
  #   c_cid[0], c_cid[-1] = c_cid[-1], c_cid[0]
  ```
  - Refs are sampled from **seen-char training subset** of the same font as the query.
  - Note: the swap that would prevent ref == query is commented out → refs **can** include the query char by chance (in the train split; if `train_ch ⊂ query_ch`, this never happens by construction).
- `num_refs`: train uses `num_refs + (num_refs&1)` (`datasets_h5.py:185`) = if odd, +1 (so always even); val uses `num_refs`. `train.yaml:40` sets `num_refs: 3` → train sees 4 refs, val sees 3.
- Blind: `n_refs: 1` default (`model.py:344`). Not a bug but a different training setting; paper effective `n_refs ≈ 4 train / 3 val`.

### C3. **Coverage-similarity routing** [P0]

- Official computes a scalar similarity per (query, ref) pair from their stroke-IDS sequences (`encoder.py:307-357`, `IDSEncoder.coverage`): the longest IDC-anchored common substring length, normalised by query IDS length. This drives the soft-weighting in StyleEncoder (`encoder.py:608-611`).
- Blind: no such signal exists. The decoder cannot prefer the most-similar ref over the least-similar.

### C4. **IDS mode = 'radical'** for train [P1]

- Official `config/base.yaml:69-70`:
  - `input_mode: ch` (so the encoder looks up IDS from char rather than ingesting raw IDS strings).
  - `ids_mode: radical` (so the IDS tree is resolved to radical-only leaves, not stroke leaves).
- Coverage similarity (`IDSEncoder.coverage`) internally always uses `stroke` mode (`encoder.py:340-344`).
- Blind: synthesises IDS strings without distinguishing radical vs stroke (`ids.py`, no `ids_mode`).

### C5. BOS/EOS/PAD conventions [P2]

- Official `IDSEncoder` special tokens: `'pad', 'sep'` only when `ids_mode == 'all'`, else just `'pad'` (`encoder.py:170`). No BOS, no EOS — sequence is right-padded with `pad` to `max_len`.
- For the AR decoder, there is no BOS prepended explicitly: `GPT.forward` shifts via `idx_cond if idx.size(1) <= block_size else idx[:, -block_size:]` and **at training time prepends `embeddings` (= IDS tokens) instead** (`nanogpt.py:241`). The "start token" of generation is just whatever the first IDS embedding is.
- At inference, `GPT.generate` starts from `idx = x[:, :0]` (an empty tensor) and conditions purely on `embeddings + x_style` (`net2net_model.py:140-149`).
- Blind: dedicated BOS index at codebook_size (`model.py:561-563`); IDS sequence has its own BOS/EOS/PAD/UNK in `ids.py:74-78`. Both schemes are internally consistent, just different.

---

## hparam_deltas

| Hyperparam | Paper note / blind | Official | Verdict |
|---|---|---|---|
| Decoder blocks (`n_layer`) | 10 | 10 (`train.yaml:27`) | ✓ matches |
| Decoder heads (`n_head`) | 8 | 8 (`train.yaml:28`) | ✓ matches |
| Decoder dim (`n_embd`) | 384 | 384 (`train.yaml:29`) | ✓ matches |
| Self-attn per block | 2 | **1** (`nanogpt.py:159-163`) | ✗ blind wrong (P0) |
| AR vocab_size | 256 | 256 (`train.yaml:25`) | ✓ matches |
| AR block_size | not pinned | **290** = 256 + 35 - 1 (`base.yaml:58`) | blind has 256 only |
| IDS max_len | 32 (blind), 35 (paper) | **35** (`base.yaml:67`) | blind shorter |
| Dropout | 0.0 (blind) | 0.1 (`base.yaml:62`) | blind too low |
| Bias in linears | True (blind) | **False** (`base.yaml:63`) | blind has bias |
| VQ embed_dim | 256 (blind) | **4** (pretrained `vq-f8-n256`) | blind 64× too wide |
| VQ codebook_size | 256 | 256 | ✓ matches |
| Image size | 128 | 128 (`base.yaml:93`) | ✓ matches |
| Image channels | 1 (grayscale) | **3** (RGB, `base.yaml:95`) | blind wrong |
| n_refs | 1 | 3 (val) / 4 (train) | blind much lower |
| Optimizer | AdamW | AdamW (2 groups) | ✓ |
| Adam betas (decoder) | (0.9, 0.95) | (0.9, 0.95) (`net2net_model.py:225`) | ✓ matches |
| Adam betas (other) | not split | **(0.9, 0.999)** (`net2net_model.py:236`) | blind same group |
| weight_decay | 0.05 (blind) | **0.01** (`net2net_model.py:224`) | blind too high |
| LR schedule | warmup+cosine | **OneCycleLR** | ✗ different |
| `base_learning_rate` | 1e-4 / 5e-4 | `4.5e-6` × bs × accum (`base.yaml:3`) | scaling differs |
| Grad clip | 1.0 | not set in trainer config | blind clips, official doesn't |
| Max epochs | 15 | 15 (`train.yaml:6`) | ✓ matches |
| Batch size | 128 | 128 (`train.yaml:34`) | ✓ matches |
| Mixed precision | not set | `16-mixed` (`base.yaml:12`) | blind FP32 only |

---

## data_pp_deltas

### D1. IDS dictionary: **BabelStone + custom supplement** [P0]

- Official sources (`iffont/data/cn.py:113-207`, `resolve_IDS_babelstone`):
  - Primary: `data/raw_files/babelstone.co.uk_CJK_IDS.TXT` (Andrew West's BabelStone CJK IDS, Unicode 15.0, 97058 entries, 2023-02-08; CC0-equivalent, see file header).
  - Supplement: `data/raw_files/ids_iffont.txt` (~3500 entries authored by the IF-Font team for chars BabelStone doesn't cover).
  - Reads with two regex patterns; resolves recursively via `resolve_stroke` (leaf = single Unicode glyph) or `resolve_radical` (leaf = first-level radical, no recursion into known compounds).
- Blind: `cns_unicode_ids.tsv` (CHISE-derived) via `lookup_ids.get_ids` in `~/Char/datasets/ids/scripts/lookup_ids.py` (`blind_impl.md` item 18). This is a different IDS curation. Coverage is similar (≥99.99% of GB2312/Big5) but **decomposition trees differ** — same character can have different IDS in the two sources, which means our content signal is **not** the same string the paper trained on. P0 if we want true reproduction; P2 if we just need a consistent IDS scheme.

### D2. IDC set: **12 chars + the literal '〇'** for atomic [P2]

- Official `cn.py:11` defines the same 12 IDC chars (`⿰⿱⿲⿳⿴⿵⿶⿷⿸⿹⿺⿻`) we have, but `cn.py:74` adds `〇` (ideographic number zero, U+3007) as an "atomic-char" marker in `resolve_IDS_Dictionary` (not used by `resolve_IDS_babelstone`, but defined).
- Blind `DEFAULT_IDC_CHARS` (`ids.py:31-44`) matches the 12, no atomic marker. Fine for us — we use `'atomic'` literal string for non-IDC chars (`ids.py:71`).

### D3. IDS-tree resolution: **triplet-equivalent expansion** [P1]

- Official `IDSEncoder.triplet_equal_ids` (`encoder.py:234-266`): when the IDS tree is e.g. `⿱(⿱(a,b),c)`, it expands to three equivalent forms `⿳(a,b,c)` / `⿱(a,⿱(b,c))` / `⿱(⿱(a,b),c)`, and `embed()` randomly picks one (`encoder.py:362-363`). This is a **data augmentation** on the IDS side.
- Blind: no IDS augmentation. The "Known gaps" section of `blind_impl.md` already lists this as a nice-to-have.

### D4. Pre-rendering pipeline [P2]

- Official `iffont/data/datasets.py:_synthesis_img` + `pil_to_tensor` → normalise to `[-1, 1]` → pass through frozen VQGAN → store integer indices in HDF5. One-time preprocessing; training only reads HDF5.
- Blind has a `dataset.py` (not inspected here in detail) and the smoke test uses synthetic batches; for Phase 3 we should adopt the same offline-tokenize pattern.

### D5. RGB vs grayscale [P0]

- Official: `img_mode: RGB` (`base.yaml:96`); pretrained VQGAN expects RGB.
- Blind: grayscale.
- If we switch to pretrained VQGAN, must render RGB (3 identical channels for monochrome glyphs, or use color-aware rendering).

---

## risk_of_bug

### P0 (will silently break results)

1. **Decoder block has 2 self-attns instead of 1** (`model.py:_DecoderBlock` line 437-473) — twice the self-attn FLOPs, drastically different inductive bias. Paper note misreading.
2. **VQGAN trained from scratch** instead of frozen pretrained CompVis `vq-f8-n256`. Means Stage A is a parallel-universe VQGAN with wrong channel width (256 vs 4), wrong color (gray vs RGB), wrong training objective (MSE-only vs adversarial+perceptual+commitment).
3. **No StyleEncoder / no 3SA / no coverage-weighted ref pooling** — refs go through a flat `nn.Linear`. The whole point of the paper is content-style disentanglement via 3SA; without it we are training a generic VQ-Transformer.
4. **No MoCo / no `sup_cl` contrastive loss** — the second of two training objectives is missing entirely. Style features will not learn font identity.
5. **Coverage similarity signal absent** — the ref-routing heuristic from `IDSEncoder.coverage` is not in our codebase.
6. **IDS dictionary mismatch** — CHISE-derived `cns_unicode_ids.tsv` vs BabelStone + `ids_iffont.txt`. Different IDS trees ⇒ different content tokens ⇒ unfaithful reproduction.
7. **Image channels mismatch** — grayscale vs RGB. Coupled with #2: not fixable without re-rendering or accepting tokenizer-from-scratch.

### P1 (will degrade quality or training stability)

8. **Conditioning routing differs**: official prepends IDS to AR input as a prefix; blind feeds IDS only via cross-attention. Without prefix conditioning, AR has weaker per-step semantic anchoring.
9. **QK-LayerNorm + DropKey absent** — official's two stabilisation tricks for d_model=384 × 10 blocks training. Blind will be less stable at long training runs.
10. **OneCycleLR vs warmup-cosine** — different LR shape; OneCycle has aggressive decay tail (`final_div_factor=10/25 = 0.4` means lr ends at 0.4 / 25 = 0.016 × max_lr, a sharp drop in the last 30%).
11. **Dropout 0.0 vs 0.1, bias=True vs False, wd=0.05 vs 0.01** — standard hyperparam drift, small effect per item, additive.
12. **IDS encoder is a full Transformer encoder** in blind, just an `nn.Embedding` in official — paper-vague but blind over-engineered.
13. **`n_refs=1` vs effectively 3-4** — fewer refs = less style information for the same parameter budget.
14. **CFG-dropout adds noise the paper does not call for** (blind item 8) — may hurt reproducibility comparisons.
15. **Triplet-equivalent IDS augmentation missing** — paper trains with this; blind does not.

### P2 (minor / hygiene)

16. **No weight tying** between `token_embed` and `head` — already listed in blind "Known gaps".
17. **No pre-tokenized HDF5 pipeline** — fine for smoke tests, must add for Phase 3 if we want bs=128.
18. **BOS/EOS/PAD policy differs** — blind has explicit BOS index, official uses IDS prefix instead. Functionally similar.
19. **`〇` atomic marker missing** in blind IDS vocab (relevant only if we ever import IDS_Dictionary.txt format).

### P3 (no observable impact)

20. **`losses.z_loss` defined but unused** in official — irrelevant to blind.
21. **`use_compile: False`** — blind has no compile path.

---

## improvements

These are blind decisions that are **better than the official** (rare) or are reasonable hardenings worth keeping:

- **Frozen-VQ flag** (`VectorQuantizer.update_codebook`, `model.py:196`) — useful even when we switch to the pretrained CompVis tokenizer (which is already frozen by `model.freeze()`). Our equivalent gives us a single switch.
- **`build_context` masking for all-masked rows** (`model.py:402-407`) — prevents NaN when CFG drops the full IDS mask. Official has no CFG so doesn't face this, but our safeguard is correct.
- **Synthetic IDS for smoke tests** (`blind_impl.md` item 19) — official has none; we explicitly support synthetic batches via `paper_reimpl_shared.runner.smoke.make_synthetic_batch`. Worth keeping as a smoke-test convention.
- **Type-annotated dataclasses + explicit `IFFontConfig`** — official has no equivalent (uses `LightningCLI` + yaml). Our scheme is easier to reason about for ablation.
- **Explicit `tests/` directory** — official has no unit tests.

---

## summary

Blind reimpl scores **3 / 13 [guessed-*] confirmed correct, 4 / 13 wrong-paper-note (not blind's fault), 6 / 13 deviation from official.** Tally below maps each guessed item to its github-diff verdict:

| # | Blind guess | Verdict |
|---|---|---|
| 1 | VQ codebook=256, downsample=8 | ✓ (matches via pretrained model) |
| 2 | 10 blocks / 8 heads / d=384, **2 self-attn + 1 cross-attn** | **✗ self-attn count wrong (P0)** — paper note misread |
| 3 | IDS replaces source glyph | ✓ |
| 4 | VQ widths `base_channels=64, channel_mult=(1,2,2,4)` | ✗ official uses pretrained CompVis (d=4, not 256) |
| 5 | EMA codebook β=0.99 | ✗ official uses frozen pretrained, no in-training EMA |
| 6 | IDS encoder = 2-layer Transformer | ✗ official uses only `nn.Embedding` |
| 7 | Concat IDS + refs as cross-attn K/V | ✗ official prefix-prepends IDS, refs via 3SA-StyleEncoder |
| 8 | CFG dropout 0.1 | ✗ official has no CFG |
| 9 | BOS = codebook_size index | ✗ official has no explicit BOS |
| 10 | AR scan = raster | ✓ (official also raster, default for VQGAN-AR) |
| 11 | CE on next-VQ-token | ✓ |
| 12 | Stage-A VQ commitment + recon MSE | ✗ no stage-A in official (VQ is pretrained external) |
| 13 | β = 0.25 commitment | ✓ (official uses same β in its EMAVQ helper, though unused for the frozen tokenizer) |

Top recommendations before Phase 3 training:

1. **Fix the decoder-block self-attn count to 1** — single-line config change, biggest model-shape divergence (P0, item A1).
2. **Decide stance on VQGAN**: either (a) download pretrained CompVis `vq-f8-n256` and wrap as a frozen tokenizer (matches paper), or (b) accept Stage-A-from-scratch and re-tag this as "IF-Font-inspired" not "IF-Font reproduction" (P0, A2).
3. **Add StyleEncoder + 3SA + MoCo + `sup_cl`** if we want the paper's content-style disentanglement (P0, A3+A4). This is ~600 LOC of new module code; the largest single deviation.
4. **Switch IDS source to BabelStone** + the IF-Font supplement file (already in the third_party clone under `data/raw_files/`) — copy or vendor those two files into our data pipeline (P0, D1).
5. **Smaller items** (P1): switch LR schedule to OneCycleLR, lower weight_decay to 0.01, raise dropout to 0.1, drop bias from Linears, raise IDS max_len to 35, add per-head QK-LN.
