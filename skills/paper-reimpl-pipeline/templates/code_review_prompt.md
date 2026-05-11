# Code Reviewer Agent Adapter

Use `everything-claude-code:python-review` skill, but pass it this context:

---

Review `papers/{NN}_{SHORT}/src/{SHORT}/` for code quality. Focus on:

1. **Public-repo safety** (this repo will be pushed to public GitHub):
   - No hardcoded SSH passwords, IPs (specifically: no internal RFC1918 addresses), or usernames
   - All SSH credentials via env vars `LAB_SSH_*` only
   - No `.env`, no API tokens
   - Cross-reference local skill `~/.claude/skills/` for actual credential storage

2. **Reproducibility**:
   - `torch.manual_seed`, `np.random.seed`, `random.seed` set in `train.py`
   - deterministic flag if paper requires

3. **Backend portability**:
   - No hardcoded `D:\Char\ayueh` in `.py` files (docs/bat OK)
   - All data paths via `paper_reimpl_shared.data.manifest`
   - device from CLI flag, not hardcoded

4. **Standard PEP 8 / mypy** via `everything-claude-code:python-review`

5. **Testability**:
   - `tests/test_smoke.py` uses `paper_reimpl_shared.runner.smoke.make_synthetic_batch`
   - No real-data dependency in smoke test

Write verdict to `papers/{NN}_{SHORT}/reports/code_review_{PHASE}.md`:

```markdown
# Code Review — {NN}_{SHORT} — {PHASE}

## Verdict: PASS / PASS-WITH-NITS / FAIL

## Auto check
- ruff check: PASS
- mypy: PASS / FAIL (details)
- pytest: PASS / FAIL

## Manual check
- [✓] no hardcoded SSH creds (grep clean)
- [✓] no hardcoded data paths
- [✗] **FAIL**: src/dataset.py:33 hardcodes path
  Fix: use `manifest.content_cache_path(...)` instead

## FAIL fixes
...

## Nits
...
```
