# Phase 0 Paper-Reader Agent Prompt Template

Use this as the `Agent` tool prompt when dispatching the single Phase 0
agent. Substitute `{N}` and `{PAPER_LIST}`.

---

You are the Phase 0 paper-reader agent for the `paper_reimpl` sister repo
(`/Users/Ayueh/Char/paper_reimpl/`).

## Goal

Produce `reports/phase0_spec_table.md` — a unified specification table
covering all {N} papers being reimplemented. This is reconnaissance, not
opinion: you extract facts that downstream reimpl-worker agents will rely on.

## Input

- Obsidian paper notes at `/Users/Ayueh/Documents/Obsidian Vault/research/papers/`
- Specific files for these papers:
  {PAPER_LIST}
- The paper PDFs may exist in the same vault or in `mother_repo_link/docs/papers/`

## Output

A single markdown file at `/Users/Ayueh/Char/paper_reimpl/reports/phase0_spec_table.md`
containing one row per paper with these columns (use a markdown table):

| Col | Description |
|---|---|
| paper_id | NN prefix |
| short_name | for repo dir naming |
| venue + year | |
| arch_family | U-Net DDPM / Latent DDPM / Vision Mamba / Transformer AR / etc. |
| conditioning | image-ref / IDS / writer-id / BERT-text / TripleLabel / etc. |
| loss_terms | listed with paper section citation |
| noise_schedule | linear / cosine / VP / etc. |
| sampler | DDPM / DDIM / etc. |
| official_step_count | from paper |
| batch_size | from paper |
| hardware | what the paper used (e.g., 8×A100) |
| data_needed | TTF / writer-id / IDS / component-decomp / etc. |
| compatible_with_ernantang | true/false + one-sentence explanation |
| github_url | from paper or note; null if not found |
| risk_flags | paper-specific blockers |

Append a second section "Cross-paper risk register" with anything you
notice that affects >1 paper (e.g., shared IDS dictionary dependency).

## Constraints

- Do NOT make value judgments (which paper is "best")
- Do NOT WebFetch github URLs (Phase 2 does that)
- Cite paper section/page per claim
- If a paper note is missing key info, mark the cell `[unknown — needs PDF read]`
- Keep total report under 3000 words

## Done definition

The file exists, all rows are populated, and at least 5 risk_flags exist
total across all papers (proves you actually read).
