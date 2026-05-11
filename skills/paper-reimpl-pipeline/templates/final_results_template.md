# Final Results — {NN}_{SHORT}

**Paper**: {Full Title, Venue, Year}
**Reimpl effort dates**: YYYY-MM-DD to YYYY-MM-DD
**Total GPU hours**: ...

## TL;DR

One paragraph: did the method work on 二南堂? Better/worse than our A3.15 v4_full baseline?

## Three-stage results

| Stage | Data | Steps | Final loss | Sample quality |
|---|---|---|---|---|
| A (TTF) | 13 fonts | ... | ... | ... |
| B (multi-writer) | 24 writers | ... | ... | ... |
| C (二南堂 finetune) | 4 writers (二南堂) | ... | ... | ... |

## Comparison vs A3.15 v4_full

Same 8 char × 4 writer grid:

| Method | Grid |
|---|---|
| Paper {NN} (this work) | ![](outputs/stage_c/comparison_ours.png) |
| A3.15 v4_full | ![](outputs/stage_c/comparison_a3_15.png) |

Qualitative observations:
- ...

## What we learned

1. **Design pattern that worked**: ...
2. **Design pattern that didn't transfer**: ...
3. **Surprising deltas vs official github**: see `github_diff.md` for full list

## Recommendations for `ernantang-jit` main line

- Adopt: ... (file path in `ernantang-jit` to patch)
- Avoid: ...
- Ablate further: ...

## Artifacts (on lab server)

- Stage A ckpt: `D:\Char\ayueh\paper_reimpl\papers\{NN}_{SHORT}\outputs\stage_a\...`
- Stage B ckpt: `D:\Char\ayueh\paper_reimpl\papers\{NN}_{SHORT}\outputs\stage_b\...`
- Stage C ckpt: `D:\Char\ayueh\paper_reimpl\papers\{NN}_{SHORT}\outputs\stage_c\...`
- Sample dumps: `outputs/stage_c/samples/`

## Reports

- `blind_impl.md` — Phase 1 decision log
- `dl_review_blind.md`, `code_review_blind.md` — Gate 1
- `github_diff.md` — Phase 2 diff
- `dl_review_post_diff.md`, `code_review_post_diff.md` — Gate 2
- `stage_a_results.md`, `dl_review_stage_a.md` — Gate 3
- `stage_b_results.md`, `dl_review_stage_b.md` — Gate 4
- `stage_c_results.md`, `dl_review_stage_c.md` — Gate 5
