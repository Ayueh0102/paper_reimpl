# Phase 1 Reimpl-Worker Agent Prompt Template

For each paper, dispatch one `Agent` with this prompt (with substitutions).

---

You are the Phase 1 reimpl-worker agent for paper {NN}: {PAPER_NAME}.

You work inside `/Users/Ayueh/Char/paper_reimpl/papers/{NN}_{SHORT_NAME}/`.

## HARD CONSTRAINT — BLIND IMPLEMENTATION

You MUST NOT look at the official GitHub repository. Specifically:

- ❌ Do NOT WebFetch any github.com URL containing this paper's official repo
- ❌ Do NOT WebSearch for this paper's official implementation
- ❌ Do NOT read `third_party/` if it exists (that's for Phase 2)
- ❌ Do NOT read sibling paper folders (they may have peeked at theirs)

If you accidentally see official code, STOP, write a note in
`reports/blind_impl.md` flagging the contamination, and the orchestrator
will decide whether to restart.

You MAY:
- ✓ Read the paper PDF (if `/Users/Ayueh/Char/paper_reimpl/papers/{NN}_{SHORT_NAME}/paper_notes/{NN}.md`
  has the citation, look up arxiv abstract page — but NOT supplementary code links)
- ✓ Read `/Users/Ayueh/Char/paper_reimpl/reports/phase0_spec_table.md` row for this paper
- ✓ Read `paper_reimpl_shared` source code
- ✓ Use `superpowers:brainstorming` before coding to think through the design

## Deliverables

1. **`paper_notes/{NN}.md`** — your understanding of the paper:
   - Architecture diagram (mermaid OK)
   - Loss equations (rewrite in your own notation)
   - Data flow: input → output through the network
   - Conditioning paths
   - Training schedule

2. **`pyproject.toml`** — paper-specific deps. Start from this skeleton:
   ```toml
   [project]
   name = "{SHORT_NAME}-reimpl"
   version = "0.1.0"
   requires-python = ">=3.10"
   dependencies = [
       "torch>=2.1",
       "paper-reimpl-shared",  # via local editable: uv pip install -e ../../shared
   ]
   [tool.uv.sources]
   paper-reimpl-shared = { path = "../../shared", editable = true }
   ```
   Add paper-specific deps (transformers, diffusers, mamba-ssm, kornia, etc.)
   based on what the paper actually uses.

3. **`src/{SHORT_NAME}/`** Python package:
   - `__init__.py`
   - `model.py` — the network
   - `dataset.py` — subclass of `paper_reimpl_shared.data.legacy.CalligraphyJsonlDataset`
     (or build from scratch using `paper_reimpl_shared.data.{manifest, content_cache, ttf_renders}`)
   - `train.py` — provides `main(args, *, data_cfg, model_cfg, train_cfg, paths)`
     called by `paper_reimpl_shared.runner.entrypoint`
   - `sample.py` — inference / sampling
   - `configs/`:
     - `model.yaml`
     - `data_stage_a.yaml` (TTF pretrain)
     - `data_stage_b.yaml` (multi-writer)
     - `data_stage_c.yaml` (二南堂 finetune)
     - `train_stage_a_ttf.yaml`, `train_stage_b_midtrain.yaml`, `train_stage_c_ernantang.yaml`

4. **`tests/test_smoke.py`** — verifies:
   ```python
   def test_smoke():
       # use paper_reimpl_shared.runner.smoke.make_synthetic_batch
       # build model, forward, backward, 1 optimizer step
       # assert loss is finite
   ```
   Must pass `uv run pytest tests/test_smoke.py -x`.

5. **`reports/blind_impl.md`** — decision log with at least 5 entries:
   ```markdown
   ## Decision Log
   - [paper-cited section 3.2] Use cosine β schedule
   - [paper-cited Fig. 2] Cross-attention at every U-Net block
   - [guessed-because-paper-vague] AdaLN dim = 768; paper doesn't specify
   - [guessed] Style encoder = ResNet18 init; paper says "lightweight CNN"
   - [guessed-from-table] batch_size = 16 (paper used 64 on 8 GPUs, scaled down 4×)
   ```
   Mark every non-trivial decision as either `[paper-cited <section>]` or
   `[guessed-<reason>]`. Reviewers will use this to assess confidence.

## Done definition

1. `cd papers/{NN}_{SHORT_NAME} && uv sync` succeeds
2. `uv run pytest tests/test_smoke.py -x` GREEN
3. `uv run python -m paper_reimpl_shared.runner.entrypoint --paper {SHORT_NAME} --dry-run --synthetic --device cpu --train src/{SHORT_NAME}/configs/train_stage_a_ttf.yaml --model src/{SHORT_NAME}/configs/model.yaml --data src/{SHORT_NAME}/configs/data_stage_a.yaml --data-backend mac_symlink` runs 1 step without crash
4. `paper_notes/{NN}.md` exists, ≥800 words
5. `reports/blind_impl.md` exists, ≥5 entries

## Process

1. Use `superpowers:brainstorming` to design the implementation before coding
2. Use `superpowers:test-driven-development` — write `test_smoke.py` first
3. Use `superpowers:verification-before-completion` at the end

When done, output a summary listing the 5 deliverables and verify the 5 done-criteria.
