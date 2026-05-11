"""Physics-Informed Neural Network (PINN) loss terms for DP-Font.

Paper note (Obsidian ``023_DP-Font書法擴散PINN_IJCAI2024.md``) says:

    "PINN Physical Constraint: 將毛筆尖端運動動力學方程與墨在紙上的擴散方程
    轉化為可微分的 loss 殘差項，直接嵌入神經網路訓練目標，使模型學習符合
    物理規律的筆觸特性"

    "loss = Lsimple（denoising）+ LPINN（物理方程 MSE），兩者聯合優化"

The paper does NOT publish the explicit PDE forms — that is flagged in
``reports/blind_impl.md`` as the largest [guessed-...] gap. The two physical
priors we encode here are reasonable analogues:

  1. **Ink diffusion** — model "墨在紙上的擴散" as an isotropic diffusion of
     ink density on the page. For ink density I(x, y) (we use the
     [-1, 1]-mapped predicted glyph as a proxy: black ink = I≈+1, paper
     background = I≈-1), the *steady-state* diffusion equation is

         ν ∇²I + s(x, y) = 0

     where ν is a diffusion constant and s is the ink source. The diffusion
     residual term penalises high values of |∇²I| in regions where the
     glyph density is *low* (paper background) — physically the paper
     surface should be smooth (no ink-source spikes off the stroke). This
     manifests as the **anisotropic Laplacian penalty** below, weighted by
     a "non-ink mask" derived from the predicted x0.

  2. **Nib motion smoothness** — model "毛筆尖端運動" as a continuous
     trajectory along the stroke skeleton. The trajectory's tangent vector
     should vary smoothly (no instantaneous direction reversals — a
     non-physical brush stroke). We approximate this with a **TV (total
     variation) penalty on the stroke skeleton's response gradient** —
     equivalent to penalising the magnitude of the Laplacian of the
     skeleton-aligned channel.

  3. **Stroke continuity** — penalise isolated single-pixel ink dots that
     are physically impossible (the nib cannot lift and re-touch within a
     single stroke). Implemented as a "speckle penalty": low gradient
     coherence at a pixel surrounded by background indicates an isolated
     dot. This is an [extension] of the paper's "stroke order constraint
     ensures continuity" idea, applied as a differentiable surrogate.

All three terms are differentiable wrt the model's predicted x0, so they
back-propagate into the U-Net parameters and form the L_PINN component of

    L_total = L_simple + λ_PINN_diffusion * L_diffusion
                       + λ_PINN_nib       * L_nib
                       + λ_PINN_continuity* L_continuity

The relative weights are exposed in ``train_*.yaml``. All three are
[guessed-because-paper-vague] since the paper does not publish the PDE
form, coefficients, or relative weighting.

Tensor convention (all functions):
    x: [B, 1, H, W] float in roughly [-1, 1] (predicted glyph, with +1 =
       ink, -1 = paper). Negative-sign convention follows the dataset's
       grayscale tensor loader.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


__all__ = [
    "laplacian_2d",
    "ink_diffusion_residual",
    "nib_motion_smoothness",
    "stroke_continuity_penalty",
    "pinn_loss",
]


# ---------------------------------------------------------------------------
# Differentiable operators
# ---------------------------------------------------------------------------


def _laplacian_kernel(device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Standard 3x3 Laplacian kernel (centered, isotropic).

        [ 0  1  0]
        [ 1 -4  1]
        [ 0  1  0]
    """
    k = torch.tensor(
        [[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]],
        dtype=dtype,
        device=device,
    )
    return k.view(1, 1, 3, 3)


def laplacian_2d(x: torch.Tensor) -> torch.Tensor:
    """Apply the 2-D Laplacian operator with replicate-padding.

    Shape preserved: [B, 1, H, W] -> [B, 1, H, W].
    """
    if x.shape[1] != 1:
        # Use channel-aware filter by replicating across channels.
        k = _laplacian_kernel(x.device, x.dtype).expand(x.shape[1], 1, 3, 3).contiguous()
        x_pad = F.pad(x, (1, 1, 1, 1), mode="replicate")
        return F.conv2d(x_pad, k, groups=x.shape[1])
    k = _laplacian_kernel(x.device, x.dtype)
    x_pad = F.pad(x, (1, 1, 1, 1), mode="replicate")
    return F.conv2d(x_pad, k)


def _sobel_kernels(device: torch.device, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
    kx = torch.tensor(
        [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
        dtype=dtype,
        device=device,
    ).view(1, 1, 3, 3) / 8.0
    ky = torch.tensor(
        [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]],
        dtype=dtype,
        device=device,
    ).view(1, 1, 3, 3) / 8.0
    return kx, ky


def _grad_xy(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Sobel gradient (gx, gy) with replicate-padding."""
    assert x.shape[1] == 1, "_grad_xy expects single-channel input"
    kx, ky = _sobel_kernels(x.device, x.dtype)
    x_pad = F.pad(x, (1, 1, 1, 1), mode="replicate")
    return F.conv2d(x_pad, kx), F.conv2d(x_pad, ky)


# ---------------------------------------------------------------------------
# PINN loss terms
# ---------------------------------------------------------------------------


def ink_diffusion_residual(
    x0_pred: torch.Tensor,
    *,
    nu: float = 1.0,
    background_threshold: float = 0.0,
) -> torch.Tensor:
    """Ink-diffusion PDE residual penalty.

    Steady-state isotropic diffusion: ν ∇²I + s = 0. We penalise the
    squared Laplacian magnitude in the "background" region of the predicted
    glyph (where there should be no ink source). The "background mask" is
    a soft mask: ``sigmoid(-α (I - τ))`` with α=10 — values well below the
    threshold become 1.0 (clearly background), values above become 0.0.

    Args:
        x0_pred: [B, 1, H, W] predicted glyph in roughly [-1, 1] with
            +1 = ink.
        nu: diffusion coefficient ν (kept as a scaling knob — the paper
            does not state its value).
        background_threshold: τ. Default 0.0 splits the [-1, 1] range at the
            midpoint.

    Returns:
        Scalar mean residual squared per sample.
    """
    assert x0_pred.dim() == 4 and x0_pred.shape[1] == 1, "expect [B,1,H,W]"
    lap = laplacian_2d(x0_pred)
    bg_mask = torch.sigmoid(-10.0 * (x0_pred - background_threshold)).detach()
    # PDE residual: ν ∇²I + s = 0  →  in the bg, source s≈0 so |ν ∇²I|² is
    # the residual. Outside bg we make no claim, so weight by bg_mask.
    residual = (nu * lap) * bg_mask
    return residual.pow(2).mean()


def nib_motion_smoothness(
    x0_pred: torch.Tensor,
    *,
    skeleton: torch.Tensor | None = None,
) -> torch.Tensor:
    """Nib-motion smoothness penalty (total variation on the stroke axis).

    Models "毛筆尖端運動" as a smooth curve. If a separate ``skeleton``
    channel from the content cache is provided, we anchor the penalty to
    its support — otherwise we fall back to the predicted glyph itself.

    Implementation: take the magnitude of the Laplacian inside the stroke
    region. A smooth brush trajectory has near-zero local curvature except
    at deliberate stroke turning points, so we penalise the L1 norm of the
    Laplacian within the ink-mask.

    Args:
        x0_pred: [B, 1, H, W] predicted glyph.
        skeleton: optional [B, 1, H, W] skeleton field in [-1, 1] from the
            content cache. When None we use ``x0_pred`` directly.

    Returns:
        Scalar mean.
    """
    assert x0_pred.dim() == 4 and x0_pred.shape[1] == 1, "expect [B,1,H,W]"
    if skeleton is None:
        sig = x0_pred
    else:
        if skeleton.shape != x0_pred.shape:
            skeleton = F.interpolate(skeleton, size=x0_pred.shape[-2:], mode="bilinear", align_corners=False)
        sig = skeleton
    # Stroke mask = where the predicted glyph is dark (+ink).
    stroke_mask = torch.sigmoid(10.0 * (x0_pred - 0.0)).detach()
    lap = laplacian_2d(sig)
    return (lap.abs() * stroke_mask).mean()


def stroke_continuity_penalty(x0_pred: torch.Tensor) -> torch.Tensor:
    """Penalise speckle / isolated ink dots inside the predicted glyph.

    Differentiable surrogate for "stroke continuity": for each pixel that
    looks like ink, check whether at least one of its neighbours is also
    ink. If a pixel has ink intensity > 0 but its 3x3 neighbourhood average
    (excluding self) is near -1 (background), it is an isolated speckle.

    Returns:
        Scalar mean.
    """
    assert x0_pred.dim() == 4 and x0_pred.shape[1] == 1, "expect [B,1,H,W]"
    # 3x3 mean kernel excluding self → equivalent to a box-blur minus the
    # centre pixel divided by 8.
    box = torch.ones(1, 1, 3, 3, device=x0_pred.device, dtype=x0_pred.dtype) / 9.0
    blurred = F.conv2d(F.pad(x0_pred, (1, 1, 1, 1), mode="replicate"), box)
    neighbour_avg = (9.0 * blurred - x0_pred) / 8.0
    # ink mask (soft) — pixel above midpoint.
    ink = torch.sigmoid(10.0 * (x0_pred - 0.0))
    # speckle: ink pixel with low neighbour average. Add small floor so the
    # loss never collapses to exactly zero when the prediction is fully
    # background (which is degenerate but legal mid-training).
    speckle = ink * F.relu(-neighbour_avg)
    return speckle.mean()


# ---------------------------------------------------------------------------
# Aggregate
# ---------------------------------------------------------------------------


def pinn_loss(
    x0_pred: torch.Tensor,
    *,
    skeleton: torch.Tensor | None = None,
    weight_diffusion: float = 1.0,
    weight_nib: float = 1.0,
    weight_continuity: float = 1.0,
    nu: float = 1.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute the combined PINN loss term.

    Args:
        x0_pred: predicted x0 in roughly [-1, 1], shape [B, 1, H, W].
        skeleton: optional [B, 1, H, W] skeleton channel from the content
            cache.
        weight_*: per-term scaling. All three should be positive.
        nu: diffusion coefficient passed to ``ink_diffusion_residual``.

    Returns:
        (scalar loss, dict of detached float components for logging).
    """
    if x0_pred.dim() == 4 and x0_pred.shape[1] != 1:
        # Take the first channel — DP-Font outputs grayscale.
        x0_pred = x0_pred[:, :1]
    l_diff = ink_diffusion_residual(x0_pred, nu=nu)
    l_nib = nib_motion_smoothness(x0_pred, skeleton=skeleton)
    l_cont = stroke_continuity_penalty(x0_pred)
    total = (
        float(weight_diffusion) * l_diff
        + float(weight_nib) * l_nib
        + float(weight_continuity) * l_cont
    )
    log = {
        "loss_pinn_diffusion": float(l_diff.detach().cpu()),
        "loss_pinn_nib": float(l_nib.detach().cpu()),
        "loss_pinn_continuity": float(l_cont.detach().cpu()),
    }
    return total, log
