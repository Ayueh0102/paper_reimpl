"""Stage 0 VQGAN loss — VQLPIPSWithDiscriminator (paper-faithful).

Reproduces ``taming/modules/losses/vqperceptual.py:VQLPIPSWithDiscriminator``
from the official repo. Components:

* Pixel L1 reconstruction.
* LPIPS perceptual loss (via the ``lpips`` PyPI package; falls back to
  ``torchmetrics.image.lpip.LearnedPerceptualImagePatchSimilarity`` if the
  ``lpips`` package is not installed).
* NLayer PatchGAN discriminator with hinge GAN loss.
* Adaptive discriminator weight (``calculate_adaptive_weight``) — ratio of
  ``||grad_nll||`` to ``||grad_g||`` on the last decoder layer.
* ``disc_start`` step gate — discriminator factor is 0 below this step,
  ``disc_factor=1.0`` after.

Defaults from ``vqgan/custom_vqgan.yaml``:
    disc_start = 10000
    disc_weight = 0.8
    codebook_weight = 1.0
    perceptual_weight = 1.0
    disc_in_channels = 1
    disc_num_layers = 3
    pixelloss_weight = 1.0

Grayscale handling: when the input is 1-channel, we tile it to 3 channels
before LPIPS (which is trained on RGB). The taming code path uses
``disc_in_channels=1`` so the discriminator stays 1-channel native.

This module deliberately keeps the loss separated from the VQGAN model so
Stage 0 trainer can construct it once and call it with ``optimizer_idx``
to alternate generator / discriminator updates.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "VQLPIPSLossConfig",
    "VQLPIPSWithDiscriminator",
    "NLayerDiscriminator",
    "hinge_d_loss",
    "vanilla_d_loss",
]


# --------------------------------------------------------------------------------------
# Discriminator (taming NLayerDiscriminator port, BatchNorm-free option)
# --------------------------------------------------------------------------------------


def _disc_weights_init(m: nn.Module) -> None:
    """Pix2Pix-style init: Conv ~ N(0, 0.02), BN scale=1 + bias=0."""
    name = m.__class__.__name__
    if "Conv" in name and hasattr(m, "weight") and m.weight is not None:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif "BatchNorm" in name and hasattr(m, "weight") and m.weight is not None:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        if getattr(m, "bias", None) is not None:
            nn.init.constant_(m.bias.data, 0.0)


class NLayerDiscriminator(nn.Module):
    """Pix2Pix / taming NLayerDiscriminator.

    Mirrors ``taming/modules/discriminator/model.py:NLayerDiscriminator``.
    """

    def __init__(self, input_nc: int = 1, ndf: int = 64, n_layers: int = 3) -> None:
        super().__init__()
        norm_layer = nn.BatchNorm2d
        use_bias = False  # BatchNorm absorbs bias

        kw = 4
        padw = 1
        layers: list[nn.Module] = [
            nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, inplace=True),
        ]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            layers += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw,
                          stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, inplace=True),
            ]
        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        layers += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw,
                      stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, inplace=True),
        ]
        layers += [
            nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw),
        ]
        self.main = nn.Sequential(*layers)
        self.apply(_disc_weights_init)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.main(x)


# --------------------------------------------------------------------------------------
# GAN losses (port of ``vqperceptual.py``)
# --------------------------------------------------------------------------------------


def hinge_d_loss(logits_real: torch.Tensor, logits_fake: torch.Tensor) -> torch.Tensor:
    """Hinge D loss: 0.5 * mean(relu(1 - real) + relu(1 + fake))."""
    loss_real = torch.mean(F.relu(1.0 - logits_real))
    loss_fake = torch.mean(F.relu(1.0 + logits_fake))
    return 0.5 * (loss_real + loss_fake)


def vanilla_d_loss(logits_real: torch.Tensor, logits_fake: torch.Tensor) -> torch.Tensor:
    """Vanilla GAN D loss using softplus (numerically stable BCE)."""
    return 0.5 * (
        torch.mean(F.softplus(-logits_real))
        + torch.mean(F.softplus(logits_fake))
    )


def _adopt_weight(weight: float, global_step: int, threshold: int, value: float = 0.0) -> float:
    """Return ``value`` while ``global_step < threshold`` else ``weight``."""
    return value if global_step < threshold else weight


# --------------------------------------------------------------------------------------
# LPIPS — best-effort import with fallback
# --------------------------------------------------------------------------------------


class _LpipsWrapper(nn.Module):
    """Thin wrapper that exposes ``forward(x, y) -> [B] perceptual loss``.

    Three backends, tried in order:

    1. ``lpips`` PyPI package (``lpips.LPIPS(net='vgg')``).
    2. ``torchmetrics.image.lpip.LearnedPerceptualImagePatchSimilarity``.
    3. ``_NaivePerceptual`` — a feature-MSE on a randomly-init small CNN
       so smoke tests don't crash when no LPIPS lib is installed. Emits a
       single ``warnings.warn`` so the user knows perceptual loss is
       placeholder.

    Inputs are tiled to 3 channels when 1-channel (LPIPS uses VGG which is
    RGB-native). Expected input range is ``[-1, 1]``.
    """

    def __init__(self) -> None:
        super().__init__()
        self._backend: nn.Module
        self._kind: str
        try:
            import lpips  # type: ignore[import-untyped]

            self._backend = lpips.LPIPS(net="vgg").eval()
            for p in self._backend.parameters():
                p.requires_grad = False
            self._kind = "lpips"
            return
        except Exception:  # noqa: BLE001 — lpips may not be installed.
            pass
        try:
            from torchmetrics.image.lpip import (  # type: ignore[import-not-found]
                LearnedPerceptualImagePatchSimilarity,
            )

            self._backend = LearnedPerceptualImagePatchSimilarity(net_type="vgg", normalize=False)
            self._backend.eval()
            for p in self._backend.parameters():
                p.requires_grad = False
            self._kind = "torchmetrics"
            return
        except Exception:  # noqa: BLE001
            pass

        warnings.warn(
            "vq_font.vqgan_loss: neither `lpips` nor `torchmetrics.image.lpip` "
            "is installed; using a random-init feature-MSE placeholder. "
            "Install `lpips` (or `torchmetrics[image]`) before real Stage 0 "
            "training.",
            RuntimeWarning,
            stacklevel=2,
        )
        self._backend = _NaivePerceptual()
        self._kind = "naive"

    @staticmethod
    def _to_3ch(x: torch.Tensor) -> torch.Tensor:
        if x.shape[1] == 1:
            return x.expand(-1, 3, -1, -1)
        if x.shape[1] == 3:
            return x
        # Other channel counts: collapse to grayscale then tile.
        return x.mean(dim=1, keepdim=True).expand(-1, 3, -1, -1)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x3 = self._to_3ch(x)
        y3 = self._to_3ch(y)
        if self._kind == "lpips":
            return self._backend(x3, y3)
        if self._kind == "torchmetrics":
            # torchmetrics returns a scalar; we want a per-batch loss for
            # ``mean`` downstream. Use the underlying network directly.
            return self._backend(x3, y3)
        return self._backend(x3, y3)


class _NaivePerceptual(nn.Module):
    """Tiny feature MSE so smoke tests run without `lpips` installed.

    NOT a substitute for real LPIPS. The trainer warns on construction.
    """

    def __init__(self) -> None:
        super().__init__()
        self.feat = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(inplace=False),
            nn.AvgPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(inplace=False),
            nn.AvgPool2d(2),
        )
        for p in self.feat.parameters():
            p.requires_grad = False

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        fx = self.feat(x)
        fy = self.feat(y)
        return F.mse_loss(fx, fy, reduction="mean").reshape(1)


# --------------------------------------------------------------------------------------
# Loss config + module
# --------------------------------------------------------------------------------------


@dataclass
class VQLPIPSLossConfig:
    """Hyperparameters for :class:`VQLPIPSWithDiscriminator`.

    Defaults from ``vqgan/custom_vqgan.yaml`` of the official repo.
    """

    disc_start: int = 10000
    """Iteration at which the discriminator first contributes (before this
    step, ``disc_factor = 0``)."""
    codebook_weight: float = 1.0
    pixelloss_weight: float = 1.0
    perceptual_weight: float = 1.0
    disc_num_layers: int = 3
    disc_in_channels: int = 1
    disc_factor: float = 1.0
    disc_weight: float = 0.8
    disc_ndf: int = 64
    disc_loss: str = "hinge"  # 'hinge' or 'vanilla'


class VQLPIPSWithDiscriminator(nn.Module):
    """Stage 0 VQGAN loss: L1 + LPIPS + hinge GAN + codebook commitment.

    Mirrors the official ``VQLPIPSWithDiscriminator`` (``vqperceptual.py:34``).
    Forward signature:

        forward(codebook_loss, inputs, reconstructions, optimizer_idx,
                global_step, last_layer=None, split='train')

    ``optimizer_idx`` selects which branch returns the loss:
      * 0 — generator update (L1 + LPIPS + GAN-g + codebook).
      * 1 — discriminator update (hinge real/fake).
    """

    def __init__(self, cfg: VQLPIPSLossConfig | None = None) -> None:
        super().__init__()
        cfg = cfg or VQLPIPSLossConfig()
        self.cfg = cfg
        self.codebook_weight = cfg.codebook_weight
        self.pixel_weight = cfg.pixelloss_weight
        self.perceptual_weight = cfg.perceptual_weight
        self.disc_factor = cfg.disc_factor
        self.disc_weight = cfg.disc_weight
        self.discriminator_iter_start = cfg.disc_start

        self.perceptual_loss = _LpipsWrapper()
        self.discriminator = NLayerDiscriminator(
            input_nc=cfg.disc_in_channels,
            ndf=cfg.disc_ndf,
            n_layers=cfg.disc_num_layers,
        )
        if cfg.disc_loss == "hinge":
            self.disc_loss = hinge_d_loss
        elif cfg.disc_loss == "vanilla":
            self.disc_loss = vanilla_d_loss
        else:
            raise ValueError(f"Unknown disc_loss {cfg.disc_loss!r}")

    # ----- adaptive weight (matches ``calculate_adaptive_weight``) -----

    def _adaptive_disc_weight(
        self,
        nll_loss: torch.Tensor,
        g_loss: torch.Tensor,
        last_layer: torch.Tensor,
    ) -> torch.Tensor:
        """``||grad_nll|| / (||grad_g|| + 1e-4)``, clamped to [0, 1e4]."""
        nll_grads = torch.autograd.grad(nll_loss, last_layer, retain_graph=True,
                                         create_graph=False, allow_unused=True)[0]
        g_grads = torch.autograd.grad(g_loss, last_layer, retain_graph=True,
                                       create_graph=False, allow_unused=True)[0]
        if nll_grads is None or g_grads is None:
            return torch.tensor(0.0, device=nll_loss.device)
        d_weight = torch.norm(nll_grads) / (torch.norm(g_grads) + 1e-4)
        d_weight = torch.clamp(d_weight, 0.0, 1e4).detach()
        return d_weight * self.discriminator_weight  # type: ignore[attr-defined]

    # The taming code uses ``self.discriminator_weight`` for the multiplicand.
    @property
    def discriminator_weight(self) -> float:
        return self.disc_weight

    def forward(
        self,
        codebook_loss: torch.Tensor,
        inputs: torch.Tensor,
        reconstructions: torch.Tensor,
        optimizer_idx: int,
        global_step: int,
        last_layer: torch.Tensor | None = None,
        split: str = "train",
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Forward (generator branch when ``optimizer_idx == 0``)."""
        # ----- Reconstruction (L1 + perceptual) -----
        rec_loss = torch.abs(inputs.contiguous() - reconstructions.contiguous())
        if self.perceptual_weight > 0:
            p_loss = self.perceptual_loss(inputs.contiguous(), reconstructions.contiguous())
            rec_loss = rec_loss + self.perceptual_weight * p_loss
        else:
            p_loss = inputs.new_zeros(1)
        nll_loss = torch.mean(rec_loss)

        if optimizer_idx == 0:
            # Generator update.
            logits_fake = self.discriminator(reconstructions.contiguous())
            g_loss = -torch.mean(logits_fake)
            if last_layer is not None and self.training:
                try:
                    d_weight = self._adaptive_disc_weight(nll_loss, g_loss, last_layer)
                except RuntimeError:
                    d_weight = torch.tensor(0.0, device=nll_loss.device)
            else:
                d_weight = torch.tensor(self.disc_weight, device=nll_loss.device)
            disc_factor = _adopt_weight(self.disc_factor, global_step, self.discriminator_iter_start)
            loss = (
                nll_loss
                + d_weight * disc_factor * g_loss
                + self.codebook_weight * codebook_loss.mean()
            )
            log = {
                f"{split}/total_loss": loss.detach(),
                f"{split}/quant_loss": codebook_loss.detach().mean(),
                f"{split}/nll_loss": nll_loss.detach(),
                f"{split}/rec_loss": rec_loss.detach().mean(),
                f"{split}/p_loss": p_loss.detach().mean()
                if isinstance(p_loss, torch.Tensor) else torch.tensor(0.0),
                f"{split}/d_weight": d_weight.detach() if isinstance(d_weight, torch.Tensor)
                else torch.tensor(float(d_weight)),
                f"{split}/disc_factor": torch.tensor(disc_factor),
                f"{split}/g_loss": g_loss.detach(),
            }
            return loss, log

        if optimizer_idx == 1:
            # Discriminator update.
            logits_real = self.discriminator(inputs.contiguous().detach())
            logits_fake = self.discriminator(reconstructions.contiguous().detach())
            disc_factor = _adopt_weight(self.disc_factor, global_step, self.discriminator_iter_start)
            d_loss = disc_factor * self.disc_loss(logits_real, logits_fake)
            log = {
                f"{split}/disc_loss": d_loss.detach(),
                f"{split}/logits_real": logits_real.detach().mean(),
                f"{split}/logits_fake": logits_fake.detach().mean(),
            }
            return d_loss, log

        raise ValueError(f"optimizer_idx must be 0 or 1, got {optimizer_idx}")
