# GitHub Diff Report — Paper 04 VQ-Font

**Phase**: 2 (post blind-impl audit)
**Official repo**: https://github.com/Yaomingshuai/VQ-Font (Yao et al., AAAI 2024)
**Cloned at**: `third_party/04_vq_font/` (depth=1)
**Our blind reimpl**: `papers/04_vq_font/src/vq_font/` + `reports/blind_impl.md`

This is a *read-only* audit. No code was copied or modified.

---

## TL;DR — Severity table

| Decision (blind ID) | Blind reimpl | Official repo | Severity | Action |
| --- | --- | --- | --- | --- |
| 1. Codebook size 1024 | 1024 | 1024 (`custom_vqgan.yaml:7`) | match | none |
| 5. Encoder/decoder widths | taming-style 4-stage `(1,1,2,4)` × `base=64` | **NOT taming Encoder/Decoder** — `content_enc_builder(C_in=1, C=32, C_out=256)` 3-stage downsample (`taming/models/vqgan.py:29-30`) + custom `dec_builder` | **HIGH** | redesign architecture before Stage 0 pretrain |
| 6. z_channels/embed_dim = 256 | 256/256 | 256/256 (`custom_vqgan.yaml:6-12`) | match | none |
| 3+4. No GAN / perceptual @ Stage 0 | omitted | **VQLPIPSWithDiscriminator** with `disc_start=10000, disc_weight=0.8, codebook_weight=1.0` (`custom_vqgan.yaml:20-27`); hinge GAN + LPIPS perceptual | **HIGH** | add disc + LPIPS for full reproduction |
| 7. Cross-attn 8 heads | 8 heads | 8 heads (`custom.yaml:12`, `generator.py:51`) | match | none |
| 8. Transformer depth | 6 blocks | **15 blocks** (`generator.py:51-52`) | **MEDIUM** | bump default depth |
| 8. Transformer MLP hidden | `mlp_ratio=4.0` → 1024 | `dim_mlp=512` → 2x ratio (`generator.py:51`, `former.py:17`) | **MEDIUM** | smaller MLP, ratio=2 |
| 11. AR vs bidirectional | bidirectional (parallel argmax) | **bidirectional**, parallel, self-attn only (`generator.py:999-1002`, `former.py:35-51`) | match | none |
| 12. 12 vs 13 structure classes | 14 (12 + atomic + unknown) | **13** classes (0..12) per `stru_all.json`; README says "12 (0..11)" but data + code (`generator.py:131-211`) cover 13 | **MEDIUM** | reconcile vocab; their 13 ≠ our 12-name list |
| 13/14. SSEM loss form / injection | CE on aux head + additive bias + prefix token in cross-attn | **No SSEM CE / no aux classifier**. SSEM = explicit per-class spatial-region averaging of cross-attn map; structure id selects which spatial blocks to average. (`generator.py:129-921`) | **CRITICAL** | rewrite SSEM as region-based attention-map recalibration |
| 15. λ_struct | 0.1 → 0.2 → 0.3 ramp | **N/A** — no structure CE term in repo | match w/ caveat | drop CE form or treat as our addition |
| 17. Stage 1 transformer recipe | 300k @ 2e-4, batch 32 | iter=1500001 (`custom.yaml:17`), g_lr=2e-4, batch=32 (`custom.yaml:15-18`) | minor | iter count diverges (1.5M vs 300k) |
| 16. Stage 0 VQGAN recipe | 200k @ 4e-5, batch 32 | `base_learning_rate=4.5e-6` per-sample → with `batch=8` and pytorch-lightning scaling → effective lr ≈ 4.5e-6 × accumulate × batch (`custom_vqgan.yaml:2, 33`); no iter count specified | minor | lr/batch policy uses LightningCLI defaults; iter count unknown |
| Pretrained VQGAN source | plan: pretrain on TTF corpus | They use `vqgan/1024_16*16_vaecoder.ckpt` (loaded at `generator.py:38`); `vqgan_data/{train,valid}.txt` lists training images. Built from rendered TTFs of foundertype fonts. | match in spirit | none |
| 9. 3 reference chars | R=3 | `kshot: 3` (`custom.yaml:13`) | match | none |
| 23. Initial synthesis stand-in | source content image | **NOT** the same — they use an FsFont-style memory + component-encoder pipeline (`generator.py:60-79`, `read_decode`) | **MEDIUM** | flagged in blind decision 23 |

---

## Detailed findings

### Special focus #1: VQGAN encoder/decoder channel widths

Their VQGAN is **not** the taming-transformers `Encoder/Decoder`. The `custom_vqgan.yaml:1-18` declares `target: taming.models.vqgan.VQModel` with `ch: 128, ch_mult: [1,2,2,4], attn_resolutions: [16]`, but the actual `taming/models/vqgan.py:29-30` overrides the constructor:

```
self.encoder = content_enc_builder(C_in = 1, C = 32, C_out=256)
self.decoder = dec_builder(C = 32, C_out = 1, norm = "in", out = 'tanh',C_content=256)
```

Inspecting `models/content_encoder.py:30-37`, the encoder is a **4-layer Conv stack** with widths `1 -> 32 -> 64 -> 128 -> 256 -> 256` and 3 stride-2 downsamples giving 16×16 features. The decoder (`models/decoder.py:59-68`) is `Res×3 (256ch) -> Conv upsample(128) -> Conv upsample(64) -> Conv upsample(32) -> Conv (1)` with InstanceNorm and Tanh output.

**Implication**: Our `VQGANEncoder/VQGANDecoder` (`src/vq_font/vqgan.py:232-317`) follows taming's res-block + bottleneck-attn pattern. Theirs is a simpler InstanceNorm Conv stack without bottleneck self-attention. Both produce 16×16×256, so the codebook interface is unchanged, but the inductive bias differs.

- `taming/models/vqgan.py:29` vs our `vqgan.py:232-275`
- `models/content_encoder.py:30-37` vs our `vqgan.py:240-275`
- `models/decoder.py:59-68` vs our `vqgan.py:281-317`

### Special focus #2: SSEM injection mechanism

This is the most significant blind-impl miss.

Our reimpl operationalizes SSEM as:

1. `StructureEncoder` (`transformer.py:177-195`): embed structure id, add as additive bias on every query token (`transformer.py:347-349`), and as a prefix token in cross-attn context (`transformer.py:351-356`).
2. `StructureHead` (`transformer.py:198-218`): auxiliary CE loss on pooled transformer output predicting structure id.

The official SSEM is **structurally different**. `generator.py:923-960` (`fusion_atten`) takes the 16×16×3HW cross-attention map between target tokens and reference tokens, and **explicitly partitions the spatial dimensions according to the structure id** (e.g. for `tar_stru==0` (top-bottom-ish), `i_[:7,:]` vs `i_[7:,:]` slices; for `tar_stru==2` (left-mid-right), 0:5 / 5:8 / 8: slices). It then:

- averages the attention map within each canonical structure region (`refer_similarity`, `cont_similarity` — `generator.py:129-286`),
- redistributes those region averages back to the per-token attention map via `fusion_am` (`generator.py:288-921`),
- re-softmaxes the recalibrated attention map (`generator.py:987`).

So SSEM in the official repo is a **hand-coded, per-class spatial recalibration of the cross-attention map**, not a learned embedding bias or an auxiliary classifier. There is no `nn.CrossEntropyLoss` on a structure head anywhere; no `structure_head` parameter exists.

Implication for us: our **decisions 13 (CE on aux head) and 14 (additive bias + prefix token)** are fully invented. They are reasonable learnable analogs but they are a different functional form. The paper-spirit equivalent would be to implement region-pooled attention map recalibration parameterized by structure id.

- `generator.py:923-960` (fusion_atten), `:129-212` (cont_similarity), `:214-286` (refer_similarity)
- vs `transformer.py:177-218`, `train.py:178` (our CE term)

### Special focus #3: 12 structure class definitions

- README claims **12 structures** indexed 0..11 (`README.md:37`).
- `meta/stru_all.json` contains labels `[0,1,2,3,4,5,6,7,8,9,10,11,12]` — **13 distinct values** across 3499 chars.
- `generator.py:131-211` explicitly branches on `tar_stru` values `0..12`, confirming **13** active classes in code.

So the official count is 13, not 12 as stated in their own README. Our blind tally is 14 (12 + atomic + unknown) — the 12 named in our `STRUCTURE_NAME_TO_ID` (`dataset.py:55-70`) come from `lookup_ids.parse_structure`. We cannot directly map our 12 names to their 13 numeric ids without a lookup table — they don't ship the name↔id map, only numeric labels.

**Action**: When reproducing on Ernantang data we should either (a) hand-build a 13-way mapping from their `stru_all.json` glyph examples to our IDS-derived names, or (b) treat our 14-class head as our own design and document the disagreement in the paper. The blind reimpl's 14-class vocab is now flagged as `[blind-impl-divergence]` rather than `[paper-cited]`.

- `meta/stru_all.json` (13 labels) vs `dataset.py:55-70` (14 entries)
- `README.md:37` (says 12) vs `generator.py:131-211` (uses 13)

### Special focus #4: Transformer bidirectional vs autoregressive

Bidirectional. `generator.py:51-52` builds `nn.Sequential(*[TransformerSALayer(...) for _ in range(15)])`. Each `TransformerSALayer` (`former.py:16-51`) is a **self-attention only** Pre-LN block — no causal mask, no cross-attn inside the block. The cross-attention between content and style happens **before** the transformer stack via the explicit `linears_key/value/query` projections + matmul in `read_decode` (`generator.py:962-994`). The transformer stack then refines the fused token features in parallel and `mlp_head` produces logits for `argmax` index selection (`generator.py:1001-1002`).

Our reimpl uses 6 transformer blocks each with self-attn + **cross-attn inside the block** (`transformer.py:147-169`). Functionally equivalent for the prediction target (parallel argmax over codebook) but architecturally re-arranged.

- `generator.py:51-52, 999-1002` vs `transformer.py:226-370`
- `former.py:16-51` (their block, self-attn only) vs `transformer.py:147-169` (our block, self+cross+MLP)

### Special focus #5: Pretrained VQGAN source

`generator.py:38` hard-codes `model.init_from_ckpt('vqgan/1024_16*16_vaecoder.ckpt')`. The Stage-0 trainer (`vqgan/custom_vqgan.yaml` consumed by `taming/main.py`) reads `vqgan_data/train.txt` and `vqgan_data/valid.txt` — both empty placeholders in the cloned repo. Per README §1 the user is expected to render font glyphs from foundertype.com `.ttf` files and list the per-image paths in those txts. So they pretrain VQGAN on a **TTF-rendered corpus** very similar to our planned Stage 0.

The shipped `1024_16*16_vaecoder.ckpt` file itself is **not in git** (we found no `*.ckpt` in the cloned tree). It is expected to be downloaded or trained by the user.

- `vqgan/custom_vqgan.yaml:1-43`, `train_vqgan.sh:1`, `README.md:55-65`
- vs our `train.py:289-360` (Stage 0 trainer)

### Special focus #6: λ_struct value

**No equivalent in official repo.** They do not have a structure-classification CE term. Their structure-aware enhancement is the attention-map recalibration described above, which is parameter-free (just region averaging).

The losses actually used in `CombinedTrainer.train` (`combined_trainer.py:115-119`) are:
- L1 reconstruction (`add_l1_loss_only_mainstructure`): weight `*2` (`base_trainer.py:98`)
- LPIPS perceptual: weight 1 (`base_trainer.py:104-106`)
- Cross-entropy on **codebook index** logits vs target's codebook indices: 1.0 for main + 0.5 for self-infer (`base_trainer.py:155-162`)
- GAN generator loss × `gan_w` × 0.002 (`base_trainer.py:200-207`, `custom.yaml` doesn't pin `gan_w` so it inherits `defaults.yaml:49` → `gan_w=1.0`)
- Discriminator hinge × `gan_w` × 0.002 (`base_trainer.py:222-237`)

So they have a token-CE term equivalent to ours (`train.py:173-177`) but **no structure CE**. Our `λ_struct` line item has no analog.

### Special focus #7: Stage 0 (VQGAN) iter count + lr

The official VQGAN config (`custom_vqgan.yaml:1-43`) is consumed by `taming/main.py` (pytorch-lightning + `LightningCLI`). It sets:
- `base_learning_rate: 4.5e-6` (`custom_vqgan.yaml:2`)
- `batch_size: 8` (`custom_vqgan.yaml:33`)
- `disc_start: 10000` (`custom_vqgan.yaml:25`) — discriminator kicks in at step 10k.
- No explicit max-step in this file. PyTorch-Lightning typically gets `--max_epochs` or `--max_steps` from CLI; the shell script `train_vqgan.sh:1` passes only `--base ... -t True`. No iter count is pinned.

The paper note (per our `blind_impl.md:111`) claims "200k iters, lr=4e-5, batch=32" for Stage 0. The repo's `4.5e-6` × Lightning's batch-size-aware scaling on batch=8 is in the same ballpark per-sample, but the explicit number does not match.

For Stage 1 (transformer), `cfgs/custom.yaml:17-22` pins:
- `iter: 1500001`
- `g_lr: 2e-4`, `d_lr: 8e-4`
- `step_size: 10000, gamma: 0.95` (StepLR decay)
- `batch_size: 32`

Our blind note's "300k iters" is **5× lower** than the official 1.5M. lr matches (2e-4). batch matches (32).

- `vqgan/custom_vqgan.yaml:1-43`, `cfgs/custom.yaml:14-22`
- vs `train.py:313, 433, 441` (our defaults)

---

## Other observations not in the focus list

1. **VectorQuantizer impl is near-identical**. Our `VectorQuantize` (`vqgan.py:167-225`) matches the legacy `VectorQuantizer` in `taming/modules/vqvae/quantize.py:9-90` — same Euclidean dist, same `loss = (z_q - z.detach())^2 + beta * (z_q.detach() - z)^2`, β=0.25. Official prefers `VectorQuantizer2` (a fixed-beta variant) in the imports of `taming/models/vqgan.py:6`. The mathematical objective is the same; minor numerical / stability differences only.

2. **Adam β values differ**. Official VQGAN uses `Adam(betas=(0.5, 0.9))` (`taming/models/vqgan.py:124-129`); official transformer uses `Adam` with `adam_betas: [0.0, 0.9]` (`cfgs/custom.yaml:24`) — very low β1. Ours uses `AdamW(betas=(0.9, 0.999))` (`train.py:307, 371`). For GAN-style training their (0, 0.9) is standard; AdamW(0.9, 0.999) is the "modern" default. Likely matters at Stage 0 if we add the discriminator.

3. **Skeleton refs**. Their `CombTrainDataset.sample_pair_style` (`dataset_transformer.py:46-54`) computes a skeleton of each reference using `skimage.morphology.skeletonize` and feeds it alongside the raw ref. Their Generator's `encode_write_comb` ingests `style_imgs_crose`, `style_imgs_fine` (1.2× / 0.8× scaled refs, `combined_trainer.py:89-94`) on top of the base refs. This is undocumented in the paper note. Our `ref_glyphs` interface is the base refs only.

4. **Per-region attention recalibration is the ONLY structure-aware module**. There is no SSEM head, no SSEM loss. The "structure-aware enhancement" of the paper title is `fusion_atten`. Re-reading the abstract with this in mind, it's consistent: "matching and fusion of styles at the structure level" — they recalibrate which spatial regions of the cross-attention map are emphasized based on the target character's structure class.

5. **Component encoder + Memory FsFont-style**. `generator.py:58-127` builds an FsFont-style component encoder + per-style memory bank. This is the "initial synthesis" pipeline; the transformer refines its output. Our blind reimpl stands in the source-content image at this slot (`train.py:151-156`) — viable for the smoke test but it is *not* the same upstream as the paper.

6. **Decoder partial-freeze**. `generator.py:40-49` freezes everything in `vqgan` *except* `decoder.layers.{0,1,2}.{conv1,conv2}` weights and `post_quant_conv`. Our reimpl freezes the entire VQGAN at Stage 1+ (`model.py:99-105`). They unfreeze 12 decoder conv tensors + post_quant_conv during transformer training — a partial fine-tune of the early decoder. This matches paper note "保留 codebook 與 decoder 後段權重不變" — but our reading "freeze entire VQGAN" is too strict. The early decoder layers are trainable.

7. **`disc_weight=0.8` and adaptive disc weight**. `taming/modules/losses/vqperceptual.py:64-75` computes an adaptive disc weight via `||∇nll|| / ||∇g||` clamped to [0, 1e4]. Stage 0 uses this. We have no Stage 0 discriminator at all.

---

## Recommendations (next iteration)

Priority order (highest first):

1. **Re-implement SSEM as region-pooled attention recalibration**, not as a CE head. New module `RegionAttentionRecalibrator(structure_id, attn_map)` with explicit per-class spatial-block masks. Keep our `StructureHead` only as an *auxiliary* (and rename `[blind-impl-divergence]`).
2. **Reconcile structure vocab to 13 classes** matching their `stru_all.json`, with the `lookup_ids.parse_structure` mapping built from inspected glyph examples. Document the 14→13 reduction.
3. **Add LPIPS perceptual + NLayer discriminator to Stage 0** (`disc_start=10000`, `disc_weight=0.8`, `codebook_weight=1.0`). Otherwise our Stage 0 codebook will under-perform theirs.
4. **Bump Transformer to 15 self-attn blocks, dim_mlp=512**, drop in-block cross-attn, move the content↔style cross-attention to a pre-stack stage like `read_decode` (`generator.py:962-994`).
5. **Restrict VQGAN freeze at Stage 1+ to codebook + late decoder only**, unfreezing early decoder layers per `generator.py:40-49`.
6. **Optional**: replace our taming-style residual encoder/decoder with their simpler InstanceNorm Conv stack (`content_enc_builder` + `dec_builder`). Lower priority — both produce the same 16×16×256 latent so codebook interface is unchanged.

---

## Files cited

Our reimpl:
- `/Users/Ayueh/Char/paper_reimpl/papers/04_vq_font/src/vq_font/vqgan.py` (lines 47-87, 167-225, 232-317, 341-396)
- `/Users/Ayueh/Char/paper_reimpl/papers/04_vq_font/src/vq_font/transformer.py` (lines 47-75, 147-169, 177-218, 251-370)
- `/Users/Ayueh/Char/paper_reimpl/papers/04_vq_font/src/vq_font/model.py` (lines 52-82, 94-105, 118-145)
- `/Users/Ayueh/Char/paper_reimpl/papers/04_vq_font/src/vq_font/dataset.py` (lines 55-76, 133-167)
- `/Users/Ayueh/Char/paper_reimpl/papers/04_vq_font/src/vq_font/train.py` (lines 71-104, 112-189, 305-360, 421-492)

Official repo:
- `/Users/Ayueh/Char/paper_reimpl/third_party/04_vq_font/vqgan/custom_vqgan.yaml` (lines 1-43)
- `/Users/Ayueh/Char/paper_reimpl/third_party/04_vq_font/cfgs/custom.yaml` (lines 12-32)
- `/Users/Ayueh/Char/paper_reimpl/third_party/04_vq_font/cfgs/defaults.yaml` (lines 1-57)
- `/Users/Ayueh/Char/paper_reimpl/third_party/04_vq_font/taming/models/vqgan.py` (lines 12-156)
- `/Users/Ayueh/Char/paper_reimpl/third_party/04_vq_font/taming/modules/vqvae/quantize.py` (lines 9-90)
- `/Users/Ayueh/Char/paper_reimpl/third_party/04_vq_font/taming/modules/losses/vqperceptual.py` (lines 34-137)
- `/Users/Ayueh/Char/paper_reimpl/third_party/04_vq_font/models/generator.py` (lines 33-79, 129-286, 288-921, 923-960, 962-1008)
- `/Users/Ayueh/Char/paper_reimpl/third_party/04_vq_font/models/former.py` (lines 16-77)
- `/Users/Ayueh/Char/paper_reimpl/third_party/04_vq_font/models/content_encoder.py` (lines 23-39)
- `/Users/Ayueh/Char/paper_reimpl/third_party/04_vq_font/models/decoder.py` (lines 51-70)
- `/Users/Ayueh/Char/paper_reimpl/third_party/04_vq_font/models/comp_encoder.py` (lines 38-60)
- `/Users/Ayueh/Char/paper_reimpl/third_party/04_vq_font/trainer/base_trainer.py` (lines 93-237)
- `/Users/Ayueh/Char/paper_reimpl/third_party/04_vq_font/trainer/combined_trainer.py` (lines 99-122)
- `/Users/Ayueh/Char/paper_reimpl/third_party/04_vq_font/datasets/dataset_transformer.py` (lines 17-100)
- `/Users/Ayueh/Char/paper_reimpl/third_party/04_vq_font/README.md` (lines 1-92)
- `/Users/Ayueh/Char/paper_reimpl/third_party/04_vq_font/meta/stru_all.json` (3499 entries, labels 0..12)
