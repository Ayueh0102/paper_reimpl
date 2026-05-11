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

from .model import D3PMUniform, QTFontModel, decode_states_to_image


@torch.no_grad()
def sample_states(
    model: QTFontModel,
    diffusion: D3PMUniform,
    *,
    batch_size: int,
    content: torch.Tensor,
    char_id: torch.Tensor | None = None,
    writer_id: torch.Tensor | None = None,
    script_id: torch.Tensor | None = None,
    ref_images: torch.Tensor | None = None,
    ref_valid: torch.Tensor | None = None,
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
        logits = model.predict_logits_from_states(
            x_t,
            t_batch,
            content=content,
            char_id=char_id,
            writer_id=writer_id,
            script_id=script_id,
            ref_images=ref_images,
            ref_valid=ref_valid,
        )
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
    **cond_kwargs,
) -> torch.Tensor:
    """Decode the predicted final state into a pixel image (B, 1, H, W)."""
    cfg = model.cfg
    states = sample_states(
        model,
        diffusion,
        batch_size=batch_size,
        content=content,
        device=content.device,
        **cond_kwargs,
    )
    one_hot = F.one_hot(states, num_classes=cfg.n_states).float()
    # Convert one-hot back into "logits" (huge value at the picked class) so
    # decode_states_to_image takes argmax-equivalent expected value.
    pseudo_logits = (one_hot * 10.0) - 5.0
    return decode_states_to_image(
        pseudo_logits, depth=cfg.depth, n_states=cfg.n_states, image_size=cfg.image_size
    )
