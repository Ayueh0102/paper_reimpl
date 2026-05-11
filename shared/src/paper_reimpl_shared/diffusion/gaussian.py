"""Minimal DDPM utilities for x0/epsilon prediction experiments."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


def _extract(values: torch.Tensor, timesteps: torch.Tensor, shape: tuple[int, ...]) -> torch.Tensor:
    out = values.gather(0, timesteps.to(values.device))
    return out.reshape(timesteps.shape[0], *((1,) * (len(shape) - 1)))


def linear_beta_schedule(
    timesteps: int,
    *,
    beta_start: float,
    beta_end: float,
    device: torch.device | str,
) -> torch.Tensor:
    return torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float32, device=device)


def cosine_beta_schedule(
    timesteps: int,
    *,
    device: torch.device | str,
    s: float = 0.008,
    max_beta: float = 0.999,
) -> torch.Tensor:
    """Improved-DDPM cosine schedule."""

    steps = torch.arange(timesteps + 1, dtype=torch.float32, device=device)
    x = (steps / timesteps + s) / (1.0 + s)
    alpha_bar = torch.cos(x * torch.pi * 0.5).pow(2)
    alpha_bar = alpha_bar / alpha_bar[0]
    betas = 1.0 - (alpha_bar[1:] / alpha_bar[:-1])
    return betas.clamp(min=1e-8, max=max_beta).to(dtype=torch.float32)


def build_beta_schedule(
    *,
    schedule: str,
    timesteps: int,
    beta_start: float,
    beta_end: float,
    device: torch.device | str,
) -> torch.Tensor:
    schedule = str(schedule or "linear").lower()
    if schedule == "linear":
        return linear_beta_schedule(timesteps, beta_start=beta_start, beta_end=beta_end, device=device)
    if schedule == "cosine":
        return cosine_beta_schedule(timesteps, device=device)
    raise ValueError(f"Unsupported beta schedule: {schedule}")


@dataclass
class DiffusionBatch:
    x_t: torch.Tensor
    target: torch.Tensor
    noise: torch.Tensor
    timesteps: torch.Tensor


class GaussianDiffusion:
    """DDPM schedule wrapper.

    The mainline Plan A target is x0 prediction. Epsilon prediction remains an
    explicit ablation, so the target is configurable.
    """

    def __init__(
        self,
        *,
        timesteps: int = 1000,
        beta_start: float = 1e-4,
        beta_end: float = 2e-2,
        beta_schedule: str = "linear",
        prediction_target: str = "x0",
        device: torch.device | str = "cpu",
    ) -> None:
        if prediction_target not in {"x0", "epsilon"}:
            raise ValueError(f"Unsupported prediction target: {prediction_target}")
        self.timesteps = int(timesteps)
        self.prediction_target = prediction_target
        self.beta_schedule = str(beta_schedule or "linear").lower()
        betas = build_beta_schedule(
            schedule=self.beta_schedule,
            timesteps=self.timesteps,
            beta_start=beta_start,
            beta_end=beta_end,
            device=device,
        )
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = torch.cat([torch.ones(1, device=device), alphas_cumprod[:-1]], dim=0)

        self.betas = betas
        self.alphas = alphas
        self.alphas_cumprod = alphas_cumprod
        self.alphas_cumprod_prev = alphas_cumprod_prev
        self.sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / alphas_cumprod - 1.0)
        self.posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        self.posterior_mean_coef1 = betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        self.posterior_mean_coef2 = (1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alphas_cumprod)

    def sample_training_batch(self, x0: torch.Tensor) -> DiffusionBatch:
        batch = x0.shape[0]
        timesteps = torch.randint(0, self.timesteps, (batch,), device=x0.device, dtype=torch.long)
        noise = torch.randn_like(x0)
        x_t = self.q_sample(x0, timesteps, noise)
        target = x0 if self.prediction_target == "x0" else noise
        return DiffusionBatch(x_t=x_t, target=target, noise=noise, timesteps=timesteps)

    def q_sample(self, x0: torch.Tensor, timesteps: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        return (
            _extract(self.sqrt_alphas_cumprod, timesteps, x0.shape) * x0
            + _extract(self.sqrt_one_minus_alphas_cumprod, timesteps, x0.shape) * noise
        )

    def predict_x0(self, x_t: torch.Tensor, timesteps: torch.Tensor, model_pred: torch.Tensor) -> torch.Tensor:
        if self.prediction_target == "x0":
            return model_pred.clamp(-1.0, 1.0)
        return (
            _extract(self.sqrt_recip_alphas_cumprod, timesteps, x_t.shape) * x_t
            - _extract(self.sqrt_recipm1_alphas_cumprod, timesteps, x_t.shape) * model_pred
        ).clamp(-1.0, 1.0)

    def _model_pred(
        self,
        model,
        x_t: torch.Tensor,
        timesteps: torch.Tensor,
        *,
        content: torch.Tensor,
        char_id: torch.Tensor | None = None,
        script_id: torch.Tensor | None = None,
        writer_id: torch.Tensor | None = None,
        style_family_id: torch.Tensor | None = None,
        unit_id: torch.Tensor | None = None,
        ref_images: torch.Tensor | None = None,
        ref_valid: torch.Tensor | None = None,
        cfg_scale: float = 1.0,
        content_guidance_scale: float = 0.0,
        char_guidance_scale: float = 0.0,
    ) -> torch.Tensor:
        cond_pred = model(
            x_t,
            timesteps,
            content=content,
            char_id=char_id,
            script_id=script_id,
            writer_id=writer_id,
            style_family_id=style_family_id,
            unit_id=unit_id,
            ref_images=ref_images,
            ref_valid=ref_valid,
        )
        pred = cond_pred
        if float(cfg_scale) != 1.0:
            uncond_pred = model(
                x_t,
                timesteps,
                content=torch.zeros_like(content),
                char_id=None,
                script_id=None,
                writer_id=None,
                style_family_id=None,
                unit_id=None,
                ref_images=None,
                ref_valid=None,
            )
            pred = uncond_pred + float(cfg_scale) * (cond_pred - uncond_pred)
        if float(content_guidance_scale) != 0.0:
            no_content_pred = model(
                x_t,
                timesteps,
                content=torch.zeros_like(content),
                char_id=char_id,
                script_id=script_id,
                writer_id=writer_id,
                style_family_id=style_family_id,
                unit_id=unit_id,
                ref_images=ref_images,
                ref_valid=ref_valid,
            )
            pred = pred + float(content_guidance_scale) * (cond_pred - no_content_pred)
        if float(char_guidance_scale) != 0.0 and char_id is not None:
            no_char_pred = model(
                x_t,
                timesteps,
                content=content,
                char_id=None,
                script_id=script_id,
                writer_id=writer_id,
                style_family_id=style_family_id,
                unit_id=unit_id,
                ref_images=ref_images,
                ref_valid=ref_valid,
            )
            pred = pred + float(char_guidance_scale) * (cond_pred - no_char_pred)
        return pred

    def _r_char_guide_x0(
        self,
        x0: torch.Tensor,
        r_char_classifier,
        r_char_target: torch.Tensor,
        r_char_guidance_scale: float,
    ) -> torch.Tensor:
        """Push x0 prediction toward target_char via R_char classifier gradient.

        Classifier guidance (Dhariwal & Nichol 2021): nudge x0 in the direction
        that increases log p(target_char | x0). Sampling-time only, R_char
        weights are NOT updated. Avoids the "classifier baked into model
        weights as adversarial pattern" failure mode of using R_char as a
        training loss.
        """
        if r_char_classifier is None or float(r_char_guidance_scale) == 0.0 or r_char_target is None:
            return x0
        with torch.enable_grad():
            x0_grad = x0.detach().clone().requires_grad_(True)
            x0_clamped = x0_grad.clamp(-1.0, 1.0)
            logits = r_char_classifier(x0_clamped)
            log_probs = F.log_softmax(logits.float(), dim=-1)
            target_log_prob = log_probs.gather(-1, r_char_target.long().unsqueeze(-1)).sum()
            grad = torch.autograd.grad(target_log_prob, x0_grad)[0]
        return x0 + float(r_char_guidance_scale) * grad

    @torch.no_grad()
    def p_sample(
        self,
        model,
        x_t: torch.Tensor,
        timesteps: torch.Tensor,
        *,
        content: torch.Tensor,
        char_id: torch.Tensor | None = None,
        script_id: torch.Tensor | None = None,
        writer_id: torch.Tensor | None = None,
        style_family_id: torch.Tensor | None = None,
        unit_id: torch.Tensor | None = None,
        ref_images: torch.Tensor | None = None,
        ref_valid: torch.Tensor | None = None,
        cfg_scale: float = 1.0,
        content_guidance_scale: float = 0.0,
        char_guidance_scale: float = 0.0,
        r_char_classifier=None,
        r_char_target: torch.Tensor | None = None,
        r_char_guidance_scale: float = 0.0,
    ) -> torch.Tensor:
        model_pred = self._model_pred(
            model,
            x_t,
            timesteps,
            content=content,
            char_id=char_id,
            script_id=script_id,
            writer_id=writer_id,
            style_family_id=style_family_id,
            unit_id=unit_id,
            ref_images=ref_images,
            ref_valid=ref_valid,
            cfg_scale=cfg_scale,
            content_guidance_scale=content_guidance_scale,
            char_guidance_scale=char_guidance_scale,
        )
        x0 = self.predict_x0(x_t, timesteps, model_pred)
        x0 = self._r_char_guide_x0(x0, r_char_classifier, r_char_target, r_char_guidance_scale)
        mean = (
            _extract(self.posterior_mean_coef1, timesteps, x_t.shape) * x0
            + _extract(self.posterior_mean_coef2, timesteps, x_t.shape) * x_t
        )
        nonzero_mask = (timesteps != 0).float().reshape(x_t.shape[0], *((1,) * (x_t.ndim - 1)))
        noise = torch.randn_like(x_t)
        return mean + nonzero_mask * torch.sqrt(_extract(self.posterior_variance, timesteps, x_t.shape)) * noise

    @torch.no_grad()
    def p_sample_ddim(
        self,
        model,
        x_t: torch.Tensor,
        timesteps: torch.Tensor,
        *,
        content: torch.Tensor,
        char_id: torch.Tensor | None = None,
        script_id: torch.Tensor | None = None,
        writer_id: torch.Tensor | None = None,
        style_family_id: torch.Tensor | None = None,
        unit_id: torch.Tensor | None = None,
        ref_images: torch.Tensor | None = None,
        ref_valid: torch.Tensor | None = None,
        cfg_scale: float = 1.0,
        content_guidance_scale: float = 0.0,
        char_guidance_scale: float = 0.0,
        r_char_classifier=None,
        r_char_target: torch.Tensor | None = None,
        r_char_guidance_scale: float = 0.0,
    ) -> torch.Tensor:
        model_pred = self._model_pred(
            model,
            x_t,
            timesteps,
            content=content,
            char_id=char_id,
            script_id=script_id,
            writer_id=writer_id,
            style_family_id=style_family_id,
            unit_id=unit_id,
            ref_images=ref_images,
            ref_valid=ref_valid,
            cfg_scale=cfg_scale,
            content_guidance_scale=content_guidance_scale,
            char_guidance_scale=char_guidance_scale,
        )
        x0 = self.predict_x0(x_t, timesteps, model_pred)
        x0 = self._r_char_guide_x0(x0, r_char_classifier, r_char_target, r_char_guidance_scale)
        prev_timesteps = (timesteps - 1).clamp_min(0)
        alpha_bar_t = _extract(self.alphas_cumprod, timesteps, x_t.shape)
        alpha_bar_prev = _extract(self.alphas_cumprod, prev_timesteps, x_t.shape)
        pred_noise = (x_t - alpha_bar_t.sqrt() * x0) / (1.0 - alpha_bar_t).sqrt().clamp_min(1e-8)
        prev = alpha_bar_prev.sqrt() * x0 + (1.0 - alpha_bar_prev).sqrt() * pred_noise
        return torch.where((timesteps == 0).reshape(x_t.shape[0], *((1,) * (x_t.ndim - 1))).bool(), x0, prev)

    @torch.no_grad()
    def sample(
        self,
        model,
        *,
        shape: tuple[int, int, int, int],
        content: torch.Tensor,
        init_image: torch.Tensor | None = None,
        start_timestep: int | None = None,
        char_id: torch.Tensor | None = None,
        script_id: torch.Tensor | None = None,
        writer_id: torch.Tensor | None = None,
        style_family_id: torch.Tensor | None = None,
        unit_id: torch.Tensor | None = None,
        ref_images: torch.Tensor | None = None,
        ref_valid: torch.Tensor | None = None,
        sampler: str = "ddpm",
        cfg_scale: float = 1.0,
        content_guidance_scale: float = 0.0,
        char_guidance_scale: float = 0.0,
        r_char_classifier=None,
        r_char_target: torch.Tensor | None = None,
        r_char_guidance_scale: float = 0.0,
        device: torch.device | str,
    ) -> torch.Tensor:
        sampler = sampler.lower()
        if sampler not in {"ddpm", "ddim"}:
            raise ValueError(f"Unsupported sampler: {sampler}")
        start_step = self.timesteps - 1 if start_timestep is None else int(start_timestep)
        start_step = max(0, min(self.timesteps - 1, start_step))
        if init_image is None:
            x_t = torch.randn(shape, device=device)
        else:
            if tuple(init_image.shape) != shape:
                raise ValueError(f"init_image shape {tuple(init_image.shape)} does not match requested shape {shape}")
            t = torch.full((shape[0],), start_step, dtype=torch.long, device=device)
            x_t = self.q_sample(init_image.to(device), t, torch.randn(shape, device=device))
        for step in reversed(range(start_step + 1)):
            t = torch.full((shape[0],), step, dtype=torch.long, device=device)
            sample_step = self.p_sample_ddim if sampler == "ddim" else self.p_sample
            x_t = sample_step(
                model,
                x_t,
                t,
                content=content,
                char_id=char_id,
                script_id=script_id,
                writer_id=writer_id,
                style_family_id=style_family_id,
                unit_id=unit_id,
                ref_images=ref_images,
                ref_valid=ref_valid,
                cfg_scale=cfg_scale,
                content_guidance_scale=content_guidance_scale,
                char_guidance_scale=char_guidance_scale,
                r_char_classifier=r_char_classifier,
                r_char_target=r_char_target,
                r_char_guidance_scale=r_char_guidance_scale,
            )
        return x_t.clamp(-1.0, 1.0)
