"""DDPM β schedules (linear / cosine).

Extracted from mother repo's diffusion.py for clean import.
"""

from __future__ import annotations

import torch


def linear_beta_schedule(num_steps: int, beta_start: float = 1e-4, beta_end: float = 0.02) -> torch.Tensor:
    """Linear β schedule as used in DDPM (Ho et al. 2020)."""
    return torch.linspace(beta_start, beta_end, num_steps, dtype=torch.float64)


def cosine_beta_schedule(num_steps: int, s: float = 0.008) -> torch.Tensor:
    """Cosine schedule from Nichol & Dhariwal (2021) Improved DDPM."""
    steps = num_steps + 1
    x = torch.linspace(0, num_steps, steps, dtype=torch.float64)
    alphas_cumprod = torch.cos(((x / num_steps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)


def get_schedule(name: str, num_steps: int, **kwargs) -> torch.Tensor:
    if name == "linear":
        return linear_beta_schedule(num_steps, **kwargs)
    if name == "cosine":
        return cosine_beta_schedule(num_steps, **kwargs)
    raise ValueError(f"Unknown β schedule: {name}")
