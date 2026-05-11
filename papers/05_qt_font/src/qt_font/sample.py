"""Discrete-diffusion reverse process for QT-Font.

Differs from a typical Gaussian DDPM sampler: states are categorical, so the
reverse step is an argmax / sample on per-leaf categorical posteriors.

Public API
----------
* :func:`sample_states` — run the full reverse process in discrete state space.
* :func:`sample_image`  — convenience wrapper that decodes the final states to
  a pixel image via :func:`qt_font.model.decode_states_to_image`.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .model import ConditioningBundle, D3PMUniform, QTFontModel


def _bundle_from_kwargs(
    content: torch.Tensor,
    cond_kwargs: dict,
) -> ConditioningBundle:
    """Lift legacy keyword kwargs into a :class:`ConditioningBundle`.

    Lets callers keep passing ``char_id=...`` etc. positionally without forcing
    them to import the dataclass. Unknown kwargs are silently ignored so the
    surface stays compatible with the cross-paper sampler harness.
    """
    return ConditioningBundle(
        content=content,
        char_id=cond_kwargs.get("char_id"),
        script_id=cond_kwargs.get("script_id"),
        writer_id=cond_kwargs.get("writer_id"),
        style_family_id=cond_kwargs.get("style_family_id"),
        unit_id=cond_kwargs.get("unit_id"),
        ref_images=cond_kwargs.get("ref_images"),
        ref_valid=cond_kwargs.get("ref_valid"),
    )


@torch.no_grad()
def sample_states(
    model: QTFontModel,
    diffusion: D3PMUniform,
    *,
    batch_size: int,
    cond_bundle: ConditioningBundle,
    temperature: float = 1.0,
    greedy_final_step: bool = True,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Reverse discrete diffusion from uniform noise → predicted x_0 states.

    Strategy (D3PM-style x_0 parameterisation):
      1. Start x_T ~ Uniform(0, K).
      2. At step t, predict x_0 logits with the model.
      3. Sample x_{t-1} from the analytic posterior given (x_t, predicted x_0).
         For uniform D3PM this collapses to:
             p(x_{t-1} | x_t, x_0) ∝ q(x_{t-1} | x_0) · q(x_t | x_{t-1})
         which we approximate by re-q-sampling x_{t-1} from the predicted x_0
         posterior. This is the standard simplification and matches the
         x_0-conditional formulation used in D3PM section 3.2.
      4. At t=0, optionally take argmax (greedy_final_step).
    """
    cfg = model.cfg
    K = cfg.n_states
    L = (2**cfg.depth) ** 2
    # Initial state: uniform noise.
    x_t = torch.randint(0, K, (batch_size, L), device=device, dtype=torch.long)

    for t in reversed(range(diffusion.timesteps)):
        t_batch = torch.full((batch_size,), t, dtype=torch.long, device=device)
        logits = model.predict_logits_from_states(x_t, t_batch, cond_bundle)
        if temperature != 1.0:
            logits = logits / temperature
        probs = F.softmax(logits, dim=-1)
        if t == 0 and greedy_final_step:
            x_t = probs.argmax(dim=-1)
        else:
            # Sample predicted x_0 then re-add noise for step t-1.
            x0_hat = torch.multinomial(probs.reshape(-1, K), num_samples=1).reshape(batch_size, L)
            if t == 0:
                x_t = x0_hat
            else:
                t_prev = torch.full((batch_size,), t - 1, dtype=torch.long, device=device)
                x_t = diffusion.q_sample(x0_hat, t_prev)
    return x_t


@torch.no_grad()
def sample_image(
    model: QTFontModel,
    diffusion: D3PMUniform,
    *,
    batch_size: int,
    content: torch.Tensor,
    cond_bundle: ConditioningBundle | None = None,
    **cond_kwargs,
) -> torch.Tensor:
    """Decode the predicted final state into a pixel image (B, 1, H, W).

    Accepts either a pre-built ``cond_bundle`` or legacy keyword args
    (``char_id``, ``writer_id``, …) which are lifted into a bundle.
    """
    cfg = model.cfg
    if cond_bundle is None:
        cond_bundle = _bundle_from_kwargs(content, cond_kwargs)
    states = sample_states(
        model,
        diffusion,
        batch_size=batch_size,
        cond_bundle=cond_bundle,
        device=content.device,
    )
    # Decode argmax-state directly to bin centres in [-1, 1]. This avoids the
    # previous "pseudo-logits" indirection (one-hot × _LOGIT_SCALE −offset → softmax
    # → expected value) whose approximation degrades for large K.
    bin_centers = torch.linspace(-1.0, 1.0, cfg.n_states + 1, device=states.device)
    bin_centers = (bin_centers[:-1] + bin_centers[1:]) * 0.5  # (K,)
    expected = bin_centers[states]  # (B, L)
    grid = 2**cfg.depth
    expected_grid = expected.reshape(batch_size, 1, grid, grid)
    if cfg.image_size == grid:
        return expected_grid
    return F.interpolate(
        expected_grid, size=(cfg.image_size, cfg.image_size), mode="bilinear", align_corners=False
    )
