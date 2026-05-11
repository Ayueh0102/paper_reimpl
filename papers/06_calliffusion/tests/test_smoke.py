"""Smoke test stub for calliffusion. Phase 1 worker replaces with real test."""
from __future__ import annotations

import pytest


@pytest.mark.skip(reason="Phase 1 reimpl-worker to implement")
def test_smoke():
    """Should:
    1. Build model with synthetic config
    2. Forward synthetic batch from paper_reimpl_shared.runner.smoke.make_synthetic_batch
    3. Backward + 1 optimizer step
    4. Assert loss is finite
    """
    pass
