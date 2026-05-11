# GitHub Diff — 01_fontdiffuser

**STATUS**: cloned
**Official repo**: https://github.com/yeungchenwa/FontDiffuser
**Official commit**: 7b28ce9c3b357f4fb23296622f458cf169803539
**Cloned to**: third_party/01_fontdiffuser/

The diff treats `papers/01_fontdiffuser/src/fontdiffuser/` (ours) as `OURS`
and `third_party/01_fontdiffuser/` (official) as `THEIRS`. Every claim is
anchored to a specific file:line on both sides.

## arch_deltas

### A1. The official "Style Encoder" is a CG-GAN trunk with spectral norm, three outputs, and is also used for `style_residual_features` consumed by the RSI/Offset module — our `StyleEncoder` is a 4-layer plain CNN with one output.

- OURS: `papers/01_fontdiffuser/src/fontdiffuser/model.py:175-206` — 4 strided convs (no SN, no residual feature pyramid), returns a single `[B, L, D]` token tensor.
- THEIRS: `third_party/01_fontdiffuser/src/modules/style_encoder.py:310-442` — `StyleEncoder` is built on `DBlock`+`SNConv2d` (spectral-normalised, CG-GAN style), returns `(style_emd, h_pooled, residual_features)`. Used at three places: (a) the `style_emd` is permuted into tokens for RSI cross-attn (model.py:34-37), (b) the **same** style image is *also* fed to the **content** encoder (model.py:43-44) to produce `style_content_res_features`, which is what the deformable-conv "Offset" module attends over in `StyleRSIUpBlock2D`. None of (b) exists in our reimpl.

### A2. The official model has a deformable-conv "Reference-Structure Interaction" (RSI) implemented as a **DCNv2 offset prediction on the skip connection**, plus an auxiliary **offset-magnitude loss** — our `RSIBlock` is a plain multi-head cross-attention add.

- OURS: `papers/01_fontdiffuser/src/fontdiffuser/model.py:258-303` — `RSIBlock` is `Q=unet feature, K/V=style tokens`, output is `x + proj(softmax(QK)V)`. No deformable conv, no offset.
- THEIRS: `third_party/01_fontdiffuser/src/modules/unet_blocks.py:423-587` (`StyleRSIUpBlock2D`) + `src/modules/attention.py:266-332` (`OffsetRefStrucInter`). For each up-stage:
  - `sc_inter_offset = OffsetRefStrucInter(res_hidden, style_content_feat)` predicts a `2*3*3 = 18` channel **DCN offset map** (`attention.py:299`).
  - `dcn_deform = torchvision.ops.DeformConv2d(skip_C, skip_C, k=3)` warps the skip features using that offset (`unet_blocks.py:469-477`, `unet_blocks.py:561`).
  - `offset_sum = mean(|offset|)` is accumulated across the whole up-path (`unet_blocks.py:557-558`) and returned as `offset_out_sum`, which becomes the `offset_loss = offset_out_sum / 2` in `train.py:196` and is weighted into the total via `offset_coefficient=0.5` (`train.py:214`, `configs/fontdiffuser.py:50`).
  - This is the actual paper's "RSI": a *learned spatial warp* of the content skip, regularised by the offset L1 magnitude. The `SpatialTransformer` cross-attention that fires alongside it (`unet_blocks.py:493-502`) is the **style** cross-attention, not RSI.

This is the single largest architectural delta.

### A3. The official "MCA" is a **concat-then-1×1-conv channel-attention block (with optional Squeeze-Excite)** — our `MCAFuse` is a **zero-init gated additive blend**.

- OURS: `papers/01_fontdiffuser/src/fontdiffuser/model.py:330-354` — `unet_feat + sigmoid(gate(proj(c))) * proj(c)` with gate weights init to 0. No SE, no concat.
- THEIRS: `third_party/01_fontdiffuser/src/modules/attention.py:359-414` (`ChannelAttnBlock`) — concat content+unet on channel dim, GN+SiLU+`Conv2d(C_in+C_content -> C_in)`, optional `SELayer(reduction=32)` with residual to `concat_feature`, then GN+SiLU+`Conv2d(C_in -> C_out)`. Default `channel_attn=True` per `configs/fontdiffuser.py:28`.

### A4. Official MCA fires only in the **middle two stages** of the U-Net; ours fires at **every stage**.

- OURS: `papers/01_fontdiffuser/src/fontdiffuser/model.py:391-407,425-445` — `down_mca[i]` and `up_mca[i]` are created for *every* stage (only skipped when `content_feats` is shorter than the stage list).
- THEIRS: `third_party/01_fontdiffuser/src/build.py:15-22` — `down_block_types=('DownBlock2D','MCADownBlock2D','MCADownBlock2D','DownBlock2D')`, `up_block_types=('UpBlock2D','StyleRSIUpBlock2D','StyleRSIUpBlock2D','UpBlock2D')`. The outer two stages are plain `DownBlock2D` / `UpBlock2D` with **no MCA and no RSI**. Mid-block always has MCA (`unet_blocks.py:111-217`).

### A5. Official RSI also fires only at the **inner up-stages 1 and 2**; ours fires at every stage in `attn_resolutions`.

- OURS: `papers/01_fontdiffuser/src/fontdiffuser/model.py:439-444` (up path) and `:401-406` (down path) — RSI is enabled whenever `resolutions[i] in attn_resolutions`. Default `attn_resolutions=(16,)` means **both** the down and up 16×16 stage carry RSI.
- THEIRS: down stages have **no RSI**; only `StyleRSIUpBlock2D` on the up path carries deformable RSI (`build.py:19-21`).

### A6. Style cross-attention on the down path: official applies it at every MCA-down stage. Ours skips it on the down path unless the resolution matches `attn_resolutions`.

- OURS: `papers/01_fontdiffuser/src/fontdiffuser/model.py:401-406` — `down_rsi` is gated by `attn_resolutions`.
- THEIRS: `third_party/01_fontdiffuser/src/modules/unet_blocks.py:278-289,326-329` — every `MCADownBlock2D` carries a `SpatialTransformer` style cross-attention regardless of resolution.

### A7. Self-attention in the U-Net: official does **not** have a separate self-attention block — only style-cross-attention via `SpatialTransformer` (the `attn1` inside `BasicTransformerBlock` *is* a self-attn, but it's bundled in the same block). Ours fires a discrete `SelfAttn2D` then a separate `RSIBlock`.

- OURS: `papers/01_fontdiffuser/src/fontdiffuser/model.py:481-483,506-508` — explicit `self.down_attn[i](h)` then `self.down_rsi[i](...)`.
- THEIRS: `third_party/01_fontdiffuser/src/modules/attention.py:94-113` — a `BasicTransformerBlock` has `attn1` (self), `attn2` (cross with style context), `ff`. Wrapped by `SpatialTransformer` (`attention.py:8-66`). Fires once per location.

### A8. Channel widths and stage count differ significantly.

- OURS: `papers/01_fontdiffuser/src/fontdiffuser/configs/model.yaml:13-14` — `base_channels=64`, `channel_mult=(1,2,4,4)` -> stage widths `(64, 128, 256, 256)`, 4 stages, resolution 128.
- THEIRS: `third_party/01_fontdiffuser/scripts/train_phase_1.sh:7-13` + `configs/fontdiffuser.py:19,22,29,31` — `resolution=96`, `unet_channels=(64,128,256,512)` (default), `content_start_channel=64`, `style_start_channel=64`. 4 stages, **last stage is 512 not 256**, plus the content encoder pyramid uses `[1,2,4,8]` channel mult (`content_encoder.py:343-345`).

### A9. Time embedding base width.

- OURS: `papers/01_fontdiffuser/src/fontdiffuser/model.py:80` — `time_input_dim = base_channels = 64`, then projected to `time_embed_dim=256` (model.yaml:17).
- THEIRS: `third_party/01_fontdiffuser/src/modules/unet.py:61,67-70` — `time_embed_dim = block_out_channels[0] * 4 = 256` (matches ours numerically) but the embedding is taken from `Timesteps(block_out_channels[0])` which is diffusers' implementation of HF DDPM. No major behavioural diff once dims match.

### A10. Input/output channels: official always `in=3, out=3` (RGB); ours `in=1, out=1` (grayscale).

- OURS: `papers/01_fontdiffuser/src/fontdiffuser/configs/model.yaml:9` — grayscale.
- THEIRS: `third_party/01_fontdiffuser/src/build.py:11-12`.

### A11. Content encoder architecture: official is a CG-GAN `DBlock`/SNConv pyramid that emits **5 features** for 128px (with explicit `save_featrues=[0,1,2,3,4]`). Ours is plain conv blocks mirroring `channel_mult`.

- OURS: `papers/01_fontdiffuser/src/fontdiffuser/model.py:143-167`.
- THEIRS: `third_party/01_fontdiffuser/src/modules/content_encoder.py:343-405`. The pyramid for 128 px is `[ch, 2ch, 4ch, 8ch, 16ch]` with `ch=64` -> widths `[64,128,256,512,1024]`. The final feature at 4×4 resolution is what the mid-block consumes (`build.py:30`: `cross_attention_dim = style_start_channel*16 = 1024`).

### A12. Style image is also fed through the *content* encoder to produce `style_content_res_features`, which are the **style-content features** that drive the DCN offset prediction.

- OURS: not present. We only feed `content` through the content encoder.
- THEIRS: `third_party/01_fontdiffuser/src/model.py:43-44` — `style_content_feature, style_content_res_features = self.content_encoder(style_images)`. These are then routed into `StyleRSIUpBlock2D` (`unet.py:282`, `unet_blocks.py:545`).

## loss_deltas (incl reduction, weights, SCR formulation)

### L1. The official Phase 1 loss has three terms; ours has only the diffusion term.

- OURS: `papers/01_fontdiffuser/src/fontdiffuser/train.py:117-138` — `loss = mse(eps_hat, eps) + scr_weight * loss_scr`. SCR is gated by `scr_weight` (0 in Stage A).
- THEIRS: `third_party/01_fontdiffuser/train.py:195-214`:
  ```
  loss = diff_loss + 0.01 * perceptual_loss + 0.5 * offset_loss
  ```
  - `diff_loss = F.mse_loss(noise_pred, noise, reduction='mean')` (train.py:195)
  - `perceptual_loss` = mean of MSE on VGG16 feature maps `[enc_1, enc_2, enc_3]` computed on **(predicted_x0, target_x0)** in the **non-normalised** image space (`src/criterion.py:27-44`, applied with re-normalise + ImageNet stats in `train.py:199-210`). Coefficient = 0.01 (`configs/fontdiffuser.py:49`).
  - `offset_loss` = the DCN offset L1 magnitude returned by every up-block, divided by 2 (`train.py:196`). Coefficient = 0.5 (`configs/fontdiffuser.py:50`).

### L2. SCR (Phase 2) is fundamentally different from our reimpl. **High-risk** delta.

- OURS: `papers/01_fontdiffuser/src/fontdiffuser/train.py:44-68` + `model.py:612-643` — a single tiny CNN `StyleExtractor` (3 strided convs + AvgPool + Linear, output 128-D L2-normalised). Loss is **supervised NT-Xent** with **within-batch writer_id positives** computed against z(x0_true) (gradient-blocked) vs z(x0_pred). Temperature `0.1`. We acknowledged in `blind_impl.md` that we substituted writer_id positives for "same-char-diff-style".
- THEIRS:
  - `third_party/01_fontdiffuser/src/modules/scr.py:9-96` — SCR wraps a **pretrained VGG-16 trunk** (`StyleFeatExtractor` uses `vgg = make_layers([...64,128,256,512,...])` from `scr_modules.py:108-125`), six per-layer linear projectors (each `[X -> 1024 -> 2048 -> 2048]` then L2-normalise, `scr_modules.py:48-105`), `InfoNCE(temperature=0.07, negative_mode='paired')` (`scr.py:30-33`, `configs/fontdiffuser.py:38`).
  - Positives are produced by **`kornia.augmentation.RandomResizedCrop(scale=(0.8,1.0), ratio=(0.75,1.33))`** on the *true* target image (`scr.py:36-39,51`). Same-style, different patch.
  - Negatives are **`num_neg=16` other-style same-content images** sampled per item by the dataset (`dataset/font_dataset.py:78-99`).
  - The SCR module itself was **pre-trained separately** for 210k steps and loaded as `scr_ckpt_path` (`scripts/train_phase_2.sh:9`).
  - NCE loss is summed across **6 layer outputs** (`nce_layers='0,1,2,3,4,5'`, default `'0,1,2,3'` in argparse) and averaged (`scr.py:84-96`).
  - SC coefficient = 0.01 (`configs/fontdiffuser.py:44`).

### L3. SCR runs on predicted-x0 in **train-time normalised** space (`pred_original_sample_norm`), not on re-normalised pixel space.

- OURS: feed `x0_pred` (the raw output of `diffusion.predict_x0`) into `StyleExtractor`.
- THEIRS: `third_party/01_fontdiffuser/train.py:217-228` — `scr(pred_original_sample_norm, target_images, neg_images, ...)`. `pred_original_sample_norm` is the `[-1,1]` predicted x0 (train.py:199-203). `target_images` is the `[-1,1]` normalised target. So inputs match (both `[-1,1]`), but the VGG-16 ingests them without ImageNet normalisation — this is a paper-internal convention. Our 1-channel grayscale extractor doesn't have this issue but the **information flow** is different (positives via crop-augmentation vs writer_id within batch).

### L4. Perceptual loss does **target-space ImageNet renormalisation**.

- OURS: not present.
- THEIRS: `third_party/01_fontdiffuser/train.py:204-210` and `utils.py` (`reNormalize_img`, `normalize_mean_std`). The predicted x0 is brought back to `[0,1]` then re-normalised by ImageNet stats before being fed to VGG16. Skipping this changes the loss magnitudes and the regularisation direction.

### L5. Reduction conventions.

- Both sides use `F.mse_loss(reduction='mean')` for the diffusion term. Equivalent.

## schedule_deltas (β, T, sampler)

### S1. β schedule.

- OURS: `papers/01_fontdiffuser/src/fontdiffuser/configs/train_stage_a_ttf.yaml:30` — `linear`, β=[1e-4, 2e-2].
- THEIRS: `third_party/01_fontdiffuser/src/build.py:66-72` + `configs/fontdiffuser.py:71` — `beta_schedule="scaled_linear"` (Stable-Diffusion convention: `β_t = (sqrt(β_start) + t/(T-1) * (sqrt(β_end) - sqrt(β_start)))**2`). Numerical: same endpoints, but the path between them is concave in β space. Trained noise scale at intermediate t differs.

### S2. T = 1000 on both sides (`build.py:67`, our yaml:27).

### S3. Sampler.

- OURS: shared `GaussianDiffusion.sample(... sampler='ddpm'|'ddim')`, CFG between conditional and null-style-token. `sample.py:74-87` sets `cfg_uncond_drops_content=False` (content kept in uncond branch).
- THEIRS: **`DPM-Solver++` (DPM-Solver v3)** via `FontDiffuserDPMPipeline` (`src/dpm_solver/pipeline_dpm_solver.py:8-117`). Default 20 inference steps (`configs/fontdiffuser.py:87`), order=2, multistep, `guidance_scale=7.5` (`configs/fontdiffuser.py:86`). CFG uncond uses **`torch.ones_like(content_images)` and `torch.ones_like(style_images)`** — i.e. "white image" inputs — for both content and style (`pipeline_dpm_solver.py:67-71`). This matches the train-time "white-out both content and style" CFG dropout (see CFG delta below).

### S4. Variance.

- THEIRS: `variance_type="fixed_small"`, `clip_sample=True` (`build.py:72-73`). Ours uses the shared GaussianDiffusion which (we believe) does fixed-small but does **not** clip x0 by default — needs check on the shared code if we ever switch sampler.

## conditioning_deltas (MCA / RSI placement)

(See A4, A5, A6, A7 above for placement.)

### C1. The official treatment of style is **two-headed**: a token sequence (from `style_encoder`) for style cross-attention, *and* a residual feature pyramid (from `content_encoder(style_image)`) for the deformable-conv offset. Ours uses only the token sequence.

### C2. CFG dropout strategy.

- OURS: `papers/01_fontdiffuser/src/fontdiffuser/train.py:102-107` — with probability `cfg_drop_prob` (default 0.1), set `ref_valid[:,0] = False`. This routes the style branch through the learned null-style token. **Content image is never dropped.**
- THEIRS: `third_party/01_fontdiffuser/train.py:182-186` — with probability `drop_prob=0.1`, **set both `content_images[i] = 1` and `style_images[i] = 1`** (white image, all pixels = `+1` after normalisation). So the uncond branch is *both* "no content" and "no style". This must match between train and sample, and the official `pipeline_dpm_solver.py:67-71` does the same `torch.ones_like` at sample time.

This is the explicit reason for our `cfg_uncond_drops_content=False` choice in `sample.py:85` — the two paths are **incompatible** by design.

## hparam_deltas (lr, optimizer, batch, grad clip, EMA)

### H1. Learning rate and schedule.

- OURS (Stage A): `learning_rate=1e-4`, no warmup, no schedule, 5000 steps (`train_stage_a_ttf.yaml:15,19`).
- THEIRS (Phase 1): `lr=1e-4`, **linear schedule with 10000-step warmup**, **440000 max train steps** (`scripts/train_phase_1.sh:21-23`). `lr_scheduler="linear"` via `diffusers.optimization.get_scheduler` (`train.py:133-137`).
- THEIRS (Phase 2): `lr=1e-5`, **constant schedule** with 1000-step warmup, **30000 max train steps** (`scripts/train_phase_2.sh:26-28`).

### H2. Batch size.

- OURS: 16 (`train_stage_a_ttf.yaml:13`).
- THEIRS: 16 (`scripts/train_phase_1.sh:14`). Same.

### H3. Resolution.

- OURS: 128.
- THEIRS: **96** for both content and style and target (`scripts/train_phase_1.sh:7-9`).

### H4. AdamW config.

- OURS: `betas=(0.9, 0.999), weight_decay=0.0, no eps override` (train.py:279-284, yaml `weight_decay: 0.0`).
- THEIRS: `betas=(0.9, 0.999), weight_decay=1e-2, eps=1e-8` (`configs/fontdiffuser.py:73-76`).

### H5. Grad clip.

- OURS: 1.0 (yaml:17).
- THEIRS: 1.0 (`configs/fontdiffuser.py:77`). Same.

### H6. EMA.

- OURS: not used.
- THEIRS: not used (`train.py` has no EMA logic). Same.

### H7. Mixed precision.

- OURS: not configured (fp32 default).
- THEIRS: `mixed_precision="no"` in the official launch scripts (`train_phase_1.sh:25`). Same default.

### H8. Phase split.

- OURS: stages A/B/C, all conducted with SCR weight ramp `0 -> 0.1 -> 0.2`. Single-network end-to-end.
- THEIRS: **two phases**. Phase 1 trains the diffusion model with diff+perceptual+offset for 440k steps. Phase 2 loads Phase-1 weights, **also loads a pretrained SCR ckpt**, freezes SCR, and trains the diffusion model with diff+perceptual+offset+sc for 30k more steps. The SCR ckpt itself was trained separately (sources for that pretrain are not in this repo — only `scr_ckpt_path` is referenced).

## data_pp_deltas (image norm, augmentation, ref/content pair sampling)

### D1. Image normalisation.

- BOTH: `transforms.Normalize([0.5], [0.5])` (mean=0.5, std=0.5) i.e. `[-1, 1]`. Same.
- THEIRS: `third_party/01_fontdiffuser/train.py:97-111` — separate transforms for content/style/target, all to `[-1, 1]`. Plus keeps a `nonorm_target_image` in `[0,1]` for VGG renorm.

### D2. Interpolation.

- BOTH: `BILINEAR`. Same.

### D3. Augmentation.

- OURS: none on content/style/target.
- THEIRS: none on content/style/target. SCR positives go through `kornia.augmentation.RandomResizedCrop(scale=(0.8,1.0), ratio=(0.75,1.33))` (`scr.py:36-39`).

### D4. Reference sampling.

- OURS: the synthetic / shared `CalligraphyJsonlDataset` selects up to `max_refs` reference glyphs by writer (or style family), excluding the query.
- THEIRS: `FontDataset.__getitem__` (`dataset/font_dataset.py:46-75`) — for each `target = style+content`, sample **one** style image by `random.choice(images_related_style except target)`. Then the **content image is the rendered TTF** for `content` (`dataset/font_dataset.py:52-53`).

### D5. Content image is always the rendered TTF reference, not an alternate style of the same character.

- THEIRS: hardcoded — `content_image_path = "{root}/{phase}/ContentImage/{content}.jpg"` (`dataset/font_dataset.py:52`).
- OURS: depends on the shared dataset wiring.

### D6. SCR negative sampling.

- OURS: within-batch writer-id positives, no explicit negatives (the contrastive denominator includes all other batch entries).
- THEIRS: `num_neg=16` explicit negatives sampled per item by **picking 16 distinct other styles and pulling the same-content glyph from each** (`dataset/font_dataset.py:78-99`). This is the literal paper "same-content-different-style" negative mining we substituted away.

## risk_of_bug (P0 — must fix before Phase 3)

1. **The DCN-offset RSI is the paper's actual contribution and we don't have it.** Our `RSIBlock` is a cross-attention head, not a deformable-conv warp. The offset-magnitude regulariser is also missing. Without it, the "Reference-Structure Interaction" claim is reduced to "style cross-attention" — which is exactly the GAN-baseline trick the paper says is insufficient. *Action: implement `OffsetRefStrucInter` + `DeformConv2d` skip warp + `offset_loss = mean(|offset|) * 0.5` in our `StyleRSIUpBlock2D` analogue.* Files to look at: `third_party/01_fontdiffuser/src/modules/attention.py:266-332` and `unet_blocks.py:423-587`.

2. **No VGG-16 perceptual loss on predicted x0.** The official Phase 1 loss is `MSE + 0.01 * Perceptual_VGG16 + 0.5 * Offset`, not just `MSE`. Without the perceptual term, the diffusion model has no "intelligible glyph" supervision in pixel space — it only has the noise MSE, which is known to under-supervise high-frequency stroke detail. *Action: port the `ContentPerceptualLoss` recipe (`third_party/01_fontdiffuser/src/criterion.py:27-44`) — VGG16 enc_1/enc_2/enc_3 MSE on `(predict_x0, target_x0)` with ImageNet renormalisation.*

3. **SCR mechanism is wrong end-to-end.** Five separate issues stack:
   - Extractor is a randomly-init 3-conv stub vs. a VGG-16 with 6 projector heads (`scr_modules.py:48-105`).
   - Loss is supervised-NT-Xent over writer_id vs. InfoNCE over **same-style RandomResizedCrop positives + explicit other-style-same-content negatives**.
   - No SCR pretrain stage (paper assumes ~210k pretrain steps).
   - Wrong temperature (`0.1` vs `0.07`).
   - We never sample `num_neg` negatives at the data layer (`dataset/font_dataset.py:78-99` is the recipe).
   *Action: this is paper-defining. Either implement faithfully or document the SCR ablation as a deliberate deviation in the reports.*

4. **CFG dropout mismatch between train and sample.** Official train-time CFG drop sets **both** `content` and `style` to white-ones (`train.py:182-186`). Official sample-time uncond is also white-ones for both (`pipeline_dpm_solver.py:67-71`). Our train-time only drops the style ref (style → null-token); content is preserved. We have already patched `sample.py:85` to disable `cfg_uncond_drops_content`, but if Stage B/C ever wants CFG with `cfg_scale>1.0` we'll either need to retrain with the official protocol or keep CFG off. *Action: either (a) switch to the official "white-out both" protocol or (b) be explicit that we sample with `cfg_scale=1.0`.* See `papers/01_fontdiffuser/reports/dl_review_blind.md` if it exists.

5. **MCA placement and operator both diverge.** Ours fires at every stage as a gated additive blend. Theirs fires only at the inner two stages as a `concat -> Conv1×1 -> SE -> Conv1×1` block (`attention.py:359-414` + `unet_blocks.py:220-339`). The outer two stages are plain `DownBlock2D` / `UpBlock2D`. This changes both representational bandwidth and channel budget. *Action: refactor `MCAFuse` and stage assignment to match the official block-type pattern.*

## risk_of_bug (P1 — should fix)

1. **β schedule is `linear` vs `scaled_linear`.** Endpoints match, intermediate noise levels differ. The shared diffusion utility supports both; consider switching for any "compare to FontDiffuser baseline" claim.

2. **No DPM-Solver++ sampler.** Official inference is 20-step DPM-Solver++ with `guidance_scale=7.5`. Ours uses DDPM (1000 steps) or DDIM. For evaluation throughput, DPM-Solver++ is ~50× faster and is what the official reports use, so any benchmark comparison without it is unfair to our reimpl.

3. **`weight_decay=0` vs `1e-2`.** Modest regularisation difference; AdamW with WD on Conv layers is a debated practice for DDPM, but the paper repo uses 1e-2.

4. **`scaled_lr=False` is fine but `lr_scheduler="linear"` with 10000-step warmup is absent.** Our flat-LR schedule will overshoot early in training; consider adding warmup.

5. **Channel widths smaller than the official.** `(64,128,256,256)` vs `(64,128,256,512)`. Halves the bottleneck representational width. Cheap to fix on a GPU run; relevant for any FID comparison.

6. **`channel_attn=True` (SELayer in MCA) is the official default.** Even if we keep the gated-add MCA, plugging in an SE block on the fused feature is a near-zero-LOC addition.

7. **Resolution choice (128 vs 96).** Doesn't matter for blind-impl correctness but matters for any "match the paper's number" claim.

8. **In/out channels (1 vs 3).** We're grayscale; official is RGB. For a Chinese-only experiment this is a reasonable simplification, but cross-paper comparisons need RGB.

9. **Content encoder also runs on the style image.** Missing in ours. Even without the DCN, the official model fuses both `content_encoder(content)` and `content_encoder(style)` features into the U-Net (the latter into the offset path).

10. **No `style_residual_features` pyramid usage.** Per A1 above.

11. **Self-attention is a separate block in ours.** Official `BasicTransformerBlock` bundles `attn1(self)+attn2(cross)+ff`. The ordering and norm placement differ.

12. **Style encoder uses spectral norm + CG-GAN `DBlock` in the official.** Ours is a plain CNN. Probably moot if the style embedding is detached for SCR, but matters if SCR is back-propped end-to-end.

13. **Down-sample uses `Conv2d stride=2` with `padding=1` in ours, vs. `Downsample2D(use_conv=True, padding=1)` (custom) in official.** Behaviour is nearly identical.

14. **`drop_prob=0.1` is identical**, but the dropout target is different (style only vs. both — already covered in P0 #4).

## Where we improved over official (rare but possible)

1. **Decoupled `cfg_scale=1.0` default** (`sample.py:51`, `sample.py:85`) — we explicitly disable CFG by default rather than baking in `guidance_scale=7.5`. The official `7.5` is fine for SD-style text-to-image but is unusually aggressive for one-shot image-to-image; making it a knob is cleaner for ablations.

2. **GroupNorm group-count probing** (`model.py:95-105`) — official crashes on awkward concat widths. We probe down 32 → 16 → ... → 1.

3. **Picklable collate** (`train.py:223-234`) — official `train.py` uses a class-level `CollateFN`, equivalent. Tie.

4. **`scr_weight=0` in Stage A explicitly skips the extractor forward pass** (`train.py:122`). Official Phase 1 has no SCR code path at all; trivially equivalent.

5. **Explicit `ref_valid` mask** instead of "white-out the tensor" sentinel. Cleaner separation of "missing" from "valid white pixel". Behaviourally different from official, but conceptually cleaner; needs to be reconciled if we want to claim "FontDiffuser-compatible".

6. **Reduced channel count.** Cheaper to train. Trade-off against P1 #5.

## Summary

- **P0 count: 5**
- **P1 count: 14**
- **Recommended fixes for reimpl-worker** (in implementation order):
  1. Add VGG-16 `ContentPerceptualLoss` (P0 #2). Smallest LOC, biggest expected sample-quality jump for Stage B/C.
  2. Implement the deformable-conv RSI (`OffsetRefStrucInter` + `DeformConv2d` skip warp + `offset_loss`) inside `StyleRSIUpBlock2D` (P0 #1, A2). Without this, "RSI" is misnamed.
  3. Rework MCA: concat+SE block, and gate it onto stages 1–2 only (P0 #5, A3, A4).
  4. Switch CFG dropout protocol to "white-out both content and style" if you want a working `cfg_scale>1.0` (P0 #4, C2). Otherwise document `cfg_scale=1.0` as a constraint.
  5. Build the full SCR pipeline: VGG-16 extractor, 6-layer projector heads, `kornia.RandomResizedCrop` positives, dataset-level `num_neg=16` other-style-same-content negatives, `InfoNCE(temperature=0.07)`, and a separate SCR pretraining run (P0 #3, L2). This is the paper's named contribution — substituting it changes the experimental claim.
  6. Re-baseline channel widths to `(64,128,256,512)` at resolution 96 if you want the FID number to be comparable (P1 #5, P1 #7).
  7. Swap sampler to DPM-Solver++ for any timed inference benchmark (P1 #2).
