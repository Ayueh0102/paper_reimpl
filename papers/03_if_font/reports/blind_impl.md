# Blind reimpl decision log — 03 IF-Font (NeurIPS 2024)

This is a **blind reimplementation** done from the project's paper notes
without inspecting the official GitHub
(`github.com/Stareven233/IF-Font`). Every non-trivial design choice is
flagged as either `[paper-cited]` (verbatim from the note) or
`[guessed-*]` (filled in to make the implementation runnable).

## Source material used

- `/Users/Ayueh/Documents/Obsidian Vault/research/papers/033_IF-Font表意描述序列字體_NeurIPS2024.md`
- `/Users/Ayueh/Char/paper_reimpl/reports/phase0_spec_table.md` row 03
- IDS dictionary from `~/Char/datasets/ids/derived/cns_unicode_ids.tsv`
  with helper `~/Char/datasets/ids/scripts/lookup_ids.py`
  (paper says "IDS" but doesn't pin a dictionary; this is the dataset we
  already curated and it covers ≥99.99 % of GB2312/Big5 chars).

## Decision Log

### Architecture

1. **[paper-cited §"訓練配置"] VQGAN codebook size = 256, downsample = 8.**
   Implemented in `model.py:VQTokenizerConfig` (`codebook_size=256`,
   `channel_mult=(1, 2, 2, 4)` → 3 strided downsamples → factor 8).

2. **[paper-cited §"訓練配置"] AR Transformer = 10 blocks, 8 heads,
   dim 384, 2 self-attn + 1 cross-attn per block.** Implemented in
   `model.py:_DecoderBlock` and `IFFontConfig` defaults (`n_blocks=10`,
   `n_heads=8`, `d_model=384`, `n_self_attn_per_block=2`).

3. **[paper-cited] IDS replaces source glyph as the content signal.**
   Implemented in `ids.py:IDSTokenizer` and the `IDSTextEncoder` →
   cross-attention path in `model.py:IFFont.build_context`.

4. **[guessed-because-paper-vague] VQ encoder/decoder layer widths.**
   Paper only mentions "VQGAN" + downsample factor. We chose
   `base_channels=64`, `channel_mult=(1,2,2,4)`. Rationale: 4 stages with
   3 downsamples gives the required factor 8; 64→256 channel ladder is
   the smallest commonly-used VQGAN configuration that gives the codebook
   enough representational room.

5. **[guessed-because-paper-vague] VQ codebook update rule = EMA +
   commitment loss + straight-through.** Paper says "VQGAN" without
   specifying the original VQ-VAE update rule vs the EMA variant. We use
   EMA (decay=0.99) because it is more stable on small batches, which is
   important for Stage A on Ernantang's 4659 chars (paper trained on a
   larger custom dataset).

6. **[guessed-because-paper-vague] IDS text encoder = 2-layer Transformer
   encoder, 4 heads, 384 dim.** Paper says "text encoder" but pins no
   architecture. We chose a lightweight Transformer encoder; depth 2
   is the minimum that allows IDC ↔ leaf token interactions while keeping
   parameter count negligible compared to the 10-block AR decoder.

7. **[guessed-because-paper-vague] Reference tokens are flattened into a
   single sequence `[B, N*256, d_model]` and concatenated *after* the
   IDS context.** Paper does not specify ref-token packing or ordering.
   We use a simple concatenation and let the decoder's cross-attention
   learn to weight IDS vs ref tokens. An attention-mask separator
   approach is a possible future refinement.

8. **[guessed-from-Ho2022] CFG dropout probability = 0.1.** Paper does
   not state CFG hyperparameters; 0.1 is the well-known Ho & Salimans
   value used by every conditional diffusion / AR re-implementation we
   are aware of.

9. **[guessed] BOS token convention — codebook index `K = codebook_size`.**
   Paper does not name a start token. We add an extra row at the top of
   the AR token embedding table (`nn.Embedding(K+1, d_model)`) and use
   index `K` as BOS. This avoids reserving a codebook slot.

10. **[guessed] AR scan order = raster (row-major).** Paper does not
    specify; raster is the default for VQGAN-AR transformers (e.g.
    Esser 2021 / DALL·E 1). Diagonal / Z-order scan are obvious future
    ablations.

### Losses

11. **[paper-cited] Main loss = cross-entropy on next-VQ-token
    prediction.** Implemented in `train.py:compute_loss` (mean reduction
    over `[B, N]` positions). The reduction choice matches Ho-style "mean
    over batch *and* spatial" which is the right scale for AdamW lr at
    1e-4 with the paper's batch_size=128.

12. **[paper-cited] Stage A loss = VQ commitment + reconstruction MSE.**
    Implemented in `train.py:compute_loss` via the `(ce_weight,
    vq_weight, recon_weight)` triple. Stage A YAML sets
    `ce_weight=0, vq_weight=1, recon_weight=1`; Stage B/C sets
    `ce_weight=1` and the others to 0.

13. **[guessed-because-paper-vague] Commitment coefficient β = 0.25.**
    Paper does not pin β; we use the original VQ-VAE default.

### Training / data

14. **[paper-cited §"訓練配置"] AdamW + warmup + cosine schedule, batch
    128, 15 epochs.** Paper-cited numbers; our train YAMLs default to
    smaller batch (16/32) so a single mid-range GPU can fit the model
    plus the ref VQ encode pass.

15. **[guessed-because-paper-vague] AdamW betas = (0.9, 0.95).** Paper
    does not pin Adam betas; 0.9 / 0.95 is the standard
    Transformer-decoder choice (e.g. GPT-2 / VQGAN papers).

16. **[guessed-because-paper-vague] Weight decay = 0.05.** Same logic
    — Transformer-decoder default.

17. **[guessed] Gradient clip norm = 1.0.** Standard for AR
    Transformers; paper does not specify.

18. **[note-from-CLAUDE.md] IDS dictionary source = CHISE-derived
    `cns_unicode_ids.tsv`.** Paper cites "linguistically-defined IDS" but
    doesn't pin a dictionary release. We use the dataset already in
    `~/Char/datasets/ids/derived/`, accessed via `lookup_ids.get_ids`.
    The dataset README claims 99.99 % coverage of GB2312 + Big5 chars,
    which is more than enough for Ernantang's 4659-char target.

19. **[guessed-because-paper-vague] Synthetic IDS for smoke/dry-run.**
    The smoke test (and the synthetic dataset path) cannot rely on the
    external IDS TSV being present. We synthesise a deterministic IDS
    per row using `synthetic_ids_for_index(index)` — one of the 12 IDC
    chars plus two pseudo-leaf CJK chars in U+4E00..U+51E8. This keeps
    the IDS → text encoder gradient path active.

20. **[guessed] IDS max length = 32 tokens.** Long IDS like 「龘」
    have ~10–14 chars; 32 leaves headroom plus BOS/EOS. Paper does
    not pin this.

### Engineering / shared-runtime compatibility

21. **[shared-runtime-contract] Synthetic batch uses
    `paper_reimpl_shared.runner.smoke.make_synthetic_batch`.** The smoke
    test does *not* pass `char_id`/`writer_id` into the model — IF-Font
    only consumes (image, refs, IDS). This is intentional and matches
    paper §3 which says the system is "source-glyph-free".

22. **[shared-runtime-contract] `train.main(args, *, data_cfg, model_cfg,
    train_cfg, paths)` matches the entrypoint signature.** Validated
    against `papers/01_fontdiffuser/src/fontdiffuser/train.py`.

## Verification checklist (done criteria)

- [x] `uv sync` produces a working venv.
- [x] `uv run pytest tests/test_smoke.py -x` GREEN (6 tests).
- [x] `uv run python -m paper_reimpl_shared.runner.entrypoint --paper if_font --dry-run --synthetic --device cpu --train ... --model ... --data ... --data-backend mac_symlink` finishes 1 step.
- [x] `paper_notes/03.md` ≥ 800 words (≈ 970 words).
- [x] `reports/blind_impl.md` contains ≥ 5 `[guessed-*]` entries
      (currently 13 `[guessed-*]` items).

## Known gaps / nice-to-haves for Phase 2

- KV caching in `TransformerARDecoder.sample` — current implementation
  recomputes the full prefix every step. Fine for smoke; fix before
  long sequence eval.
- Real Stage A VQGAN adversarial loss (paper note says "VQGAN"; we use
  only the reconstruction + commitment terms, no discriminator).
- Tied weights between AR decoder token embedding and codebook
  embeddings — could improve token quality, but paper doesn't say to do
  it.
- Top-k / top-p AR sampling at inference (current is temperature
  multinomial).
- IDS data augmentation (e.g. randomly swap equivalent IDS forms) for
  robustness — useful nice-to-have, paper does not mention.
