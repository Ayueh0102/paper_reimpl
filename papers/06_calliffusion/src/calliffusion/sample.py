"""Calliffusion sampling helper.

Wraps the shared ``GaussianDiffusion`` sampler with the BERT-context plumbing
specific to this paper.

Public API:
    >>> sample_prompts(
    ...     unet, text_encoder, diffusion,
    ...     prompts=["人 隸書 曹全碑"],
    ...     shape=(1, 1, 64, 64),
    ...     device="cpu",
    ... )
"""

from __future__ import annotations

import torch
from paper_reimpl_shared.diffusion.gaussian import GaussianDiffusion


@torch.no_grad()
def sample_prompts(
    unet: torch.nn.Module,
    text_encoder,
    diffusion: GaussianDiffusion,
    *,
    prompts: list[str],
    shape: tuple[int, int, int, int],
    sampler: str = "ddpm",
    cfg_scale: float = 1.0,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Sample images from text prompts.

    Implementation note: the shared GaussianDiffusion.sample API was built
    around image/content/id-style conditioning, not BERT context. To keep
    things simple, we *re-implement* the inner loop here with the BERT path
    plugged in. The math is identical (ε-prediction DDPM/DDIM).
    """
    device = torch.device(device)
    unet.eval()
    text_encoder.eval()

    ctx_out = (
        text_encoder.encode(prompts) if hasattr(text_encoder, "encode") else text_encoder(prompts)
    )
    context = ctx_out.last_hidden_state.to(device)
    context_mask = ctx_out.attention_mask.to(device)

    # Optional unconditional context for CFG (empty prompts).
    if cfg_scale != 1.0:
        uncond = text_encoder.encode([""] * len(prompts))
        uncond_ctx = uncond.last_hidden_state.to(device)
        uncond_mask = uncond.attention_mask.to(device)
    else:
        uncond_ctx = None
        uncond_mask = None

    x_t = torch.randn(shape, device=device)
    sampler = sampler.lower()
    for step in reversed(range(diffusion.timesteps)):
        t = torch.full((shape[0],), step, dtype=torch.long, device=device)
        eps_cond = unet(x_t, t, context=context, context_mask=context_mask)
        if uncond_ctx is not None:
            eps_uncond = unet(x_t, t, context=uncond_ctx, context_mask=uncond_mask)
            eps = eps_uncond + cfg_scale * (eps_cond - eps_uncond)
        else:
            eps = eps_cond
        # x0 from epsilon
        sqrt_recip = diffusion.sqrt_recip_alphas_cumprod[step]
        sqrt_recipm1 = diffusion.sqrt_recipm1_alphas_cumprod[step]
        x0 = (sqrt_recip * x_t - sqrt_recipm1 * eps).clamp(-1.0, 1.0)
        if sampler == "ddim":
            alpha_bar_t = diffusion.alphas_cumprod[step]
            alpha_bar_prev = (
                diffusion.alphas_cumprod[step - 1] if step > 0 else torch.tensor(1.0, device=device)
            )
            pred_noise = (x_t - alpha_bar_t.sqrt() * x0) / (1.0 - alpha_bar_t).sqrt().clamp_min(1e-8)
            x_t = alpha_bar_prev.sqrt() * x0 + (1.0 - alpha_bar_prev).sqrt() * pred_noise
        else:
            mean = (
                diffusion.posterior_mean_coef1[step] * x0
                + diffusion.posterior_mean_coef2[step] * x_t
            )
            if step > 0:
                noise = torch.randn_like(x_t)
                x_t = mean + diffusion.posterior_variance[step].sqrt() * noise
            else:
                x_t = mean
    return x_t.clamp(-1.0, 1.0)
