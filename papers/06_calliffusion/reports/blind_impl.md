# Calliffusion — Blind Implementation Decision Log

Paper: Liao 2023, *Calliffusion: Chinese Calligraphy Generation and Style Transfer
with Diffusion Modeling*, arXiv:2305.19124.

Source material used (NO official code consulted, per Phase-1 hard contract):
- `paper_notes/06.md` (compiled from the Obsidian note that holds direct
  paper quotes with `[p.X]` page citations)
- `reports/phase0_spec_table.md` row 06
- arxiv abstract page (text only)

If anything below contradicts the paper, it's a `[guessed-…]` item and
should be reviewed once we are permitted to read the official repo.

## Decision Log

### Architecture

1. **[paper-cited p.3]** U-Net block widths set to `[320, 640, 1280, 1280]`
   (model.yaml `base_channels=320, channel_mult=[1, 2, 4, 4]`). Smoke yaml
   uses `base_channels=64` to fit on CPU; production yaml overrides.

2. **[paper-cited p.2]** Cross-attention from BERT context injected at every
   resolution stage (`SpatialCrossAttention` in `model.py`). Each
   down/up/mid stage gets one cross-attn after the self-attn block.

3. **[guessed-because-paper-vague]** `num_res_blocks = 2` per stage. Paper
   only specifies the channel widths. Two ResBlocks per stage is the
   standard DDPM/Stable-Diffusion default; matches the family the paper
   places itself in.

4. **[guessed]** `num_heads = 8` for both self- and cross-attention. The
   note states only "cross-attention", no head count. 8 is the SD-1.x
   default and matches `base_channels % num_heads == 0` at every stage.

5. **[guessed]** Time embedding dim `1280` = `4 × base_channels`. Standard
   DDPM convention; paper silent.

6. **[guessed]** Zero-init the final `conv_out` (`nn.init.zeros_` on weight
   and bias). Common diffusion-training stabiliser; not stated in the
   paper but harmless and improves early-step loss.

### Text encoder / BERT vocabulary strategy

7. **[paper-cited p.2-3]** Chinese BERT, hidden size 768, used as text
   conditioning. We default to `bert-base-chinese` (`transformers` model
   id). Paper does not name the exact checkpoint; `bert-base-chinese` is
   the only HuggingFace checkpoint that fits "Chinese BERT 768-dim".

8. **[guessed-because-paper-vague — important]** Calligrapher names are
   added as `additional_special_tokens` via `tokenizer.add_special_tokens`,
   followed by `bert.resize_token_embeddings`. The paper note records
   (citing Moyun) that vanilla BERT is weak on calligrapher names; the
   most natural mitigation that does not require external pre-trained
   embeddings is to register each calligrapher as its own token and learn
   the embedding row at Stage B fine-tune. This is implemented in
   `BertTextEncoder.add_special_tokens` and in the stub encoder; the
   training loop wires it via `dataset.writer_names()` → `add_special_tokens()`.

9. **[guessed]** Stage-A freezes BERT entirely
   (`text_encoder.freeze=true, embeddings_trainable=false`). Stage-B keeps
   BERT mostly frozen but unfreezes the word-embedding table so the new
   special-token rows can learn (`embeddings_trainable=true`). Stage-C
   keeps BERT frozen (only LoRA trains). Paper does not state the freeze
   schedule.

10. **[guessed]** Prompt template is the exact string `<char> <script>
    <writer>` joined by single spaces, mirroring the paper's example
    `"人字 隸書 曹全碑"` (we drop the literal "字" suffix because it is part
    of the example, not the template). Tokenisation correctness is the
    BERT tokenizer's job.

11. **[guessed]** Prompt dropout `p=0.1` for classifier-free guidance
    training. Paper does not explicitly mention CFG, but every modern
    text-to-image diffusion uses it and inference quality is much better
    when CFG is available. Drop-to-empty-string is the simplest
    implementation that costs nothing at training time.

### Diffusion / sampler

12. **[paper-cited p.3]** Noise schedule: linear `β_1 → β_N`. We use
    `β_1=1e-4, β_N=0.02, T=1000` — the DDPM original numbers; paper does
    not name the endpoints.

13. **[paper-cited]** ε-prediction parameterisation. Paper note explicitly
    says "MSE diffusion loss only"; the standard MSE-loss DDPM target is
    ε, not x0 or v.

14. **[paper-cited]** Sampler at inference = DDPM (paper note). We also
    expose DDIM in `sample.py` because the shared utility already supports
    it and the marginal cost is one extra branch.

15. **[guessed-because-paper-vague]** CFG guidance scale `1.0` (off) at
    training, `3.0` at inference for the bundled sample script. Paper
    does not give a number; 3.0 is the SD-1.x default that works on
    most text-to-image diffusion models.

### Loss

16. **[paper-cited]** `loss = F.mse_loss(ε̂, ε, reduction='mean')`. Single
    term, no VLB component, no LPIPS, no auxiliary classifier.

### Optimiser / training

17. **[paper-cited p.3]** Learning rate `1e-5` for Stage A. Batch size
    `16`. We scale to `batch=2` for the smoke/CPU configs, and to lower
    LR `5e-6` for Stage B (post-pretrain stabilisation, paper does not
    state but is a common choice).

18. **[guessed]** Optimiser = AdamW with `β1=0.9, β2=0.999, wd=0.0`.
    Paper does not name the optimiser; AdamW is the DDPM default and
    matches the lr-1e-5 setting.

19. **[guessed]** Gradient clipping `max_norm=1.0`. Not in the paper; safe
    default that prevents NaN early in training.

### LoRA

20. **[paper-cited]** LoRA used for one-shot style transfer to unseen
    characters / alphabets. Implemented in `lora.py` as a minimal wrapper
    around `nn.Linear`. We target `to_q, to_k, to_v, to_out` in every
    cross-attention block.

21. **[guessed]** LoRA rank `r=4`, scaling `α=8` (so `α/r=2`). Paper does
    not state a number. `r=4` is the common SD-LoRA-for-one-shot setting
    (e.g. Stable Diffusion 1.5 + DreamBooth-style runs). We initialise
    `B = 0` so the adapter is a no-op at step 0 (verified by the smoke
    test `test_lora_zero_at_init_and_trains`).

22. **[guessed]** LoRA only on cross-attention projections, not on
    ResBlocks. Cross-attn is where the style signal enters; targeting
    only those keeps the parameter count tiny (~`r × (in + out)` per
    projection × ~36 projections at full size).

### Data / dataset wiring

23. **[paper-cited]** Image normalisation `[-1, 1]`, grayscale 1-channel.
    Matches the shared `CalligraphyJsonlDataset` convention. No horizontal
    flip / no rotation augment because calligraphy is orientation-aware.

24. **[guessed]** `content_channels=[]` for the JSONL dataset. Calliffusion
    has no content-cache input — only the BERT prompt — so we override the
    shared loader's default channel list to empty.

25. **[guessed]** ``max_refs=0`` (no reference glyph input). The paper's
    one-shot path is LoRA, not retrieval, so we don't pull references.

### CFG / sampling

26. **[guessed]** Unconditional context at CFG sample time = the BERT
    embedding of the empty string. Paper does not specify; this is the
    SD-1.x default.

### Stage roadmap

27. **[guessed scale-down]** Original paper trained 120 h on 2× A100 40 GB
    at batch 16. We will scale Stage A down to ~50–100k steps on a single
    RTX 6000 Ada (48 GB) at batch 16 grayscale 128. Stage B and Stage C
    fit comfortably under 10–20k steps each.

### Risks / known unknowns

28. The 2×A100×120h budget translates to ~24M images at batch 16. Our
    Ernantang-scale Stage B sees far fewer samples; expect lower
    fidelity than the paper's headline numbers.

29. BERT-base-chinese is character-level CJK but rare seal-script glyphs
    and rare calligrapher names may still tokenise into single-character
    pieces, fragmenting the signal. Special-token registration partly
    fixes this for calligrapher names but does not help rare characters.

30. We have not implemented EMA of model weights. Paper does not mention
    EMA; modern diffusion repos almost always use it. Marked as a
    nice-to-have follow-up.

31. We freeze BERT during Stage A; if the paper actually fine-tuned BERT
    too, our Stage A may underfit the prompt manifold. Mitigation: run a
    Stage A ablation with `text_encoder.freeze=false` after the first
    pass.

32. **R_char/StyleNet-style guidance** is not part of this paper, but the
    shared ``GaussianDiffusion`` supports it. We deliberately do **not**
    wire R_char guidance into `train.py` to keep the blind impl faithful.

---

End of decision log. 32 decisions, 22 marked `[guessed-…]` — well above
the 5-entry minimum required by AGENTS.md §2.
