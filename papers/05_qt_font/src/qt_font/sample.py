"""Discrete-diffusion reverse process for QT-Font — Phase 2 paper-aligned.

Implements ``q_posterior_logits`` + ``p_sample`` (Gumbel-max) from
``third_party/05_qt_font/main.py:215-273``. Sampling proceeds with a stride
``gap`` over the full ``T`` so the effective denoising steps = T / gap. The
official sampler uses ``gap=50`` with ``T=1000`` → 20 effective steps.

We split the schedule_deltas S4 cleanly:
* ``sample_labels`` runs the discrete reverse process on a **fixed-topology**
  GT-octree (the common training-time + smoke setting). This is what the
  multi-depth supervision was trained against.
* ``sample_image`` is a convenience wrapper that decodes the final 3-class
  labels back into a pixel image for the shared smoke harness.

The full "grow the output octree as you sample" path (`octree_grow` /
`octree_split` in ``third_party/05_qt_font/models/graph_diffusion.py:222-232``)
is **deferred** — it requires the full DualOctreeGNN encoder/decoder split
and is a Phase 3 follow-up. The current sampler still produces valid label
maps at the leaf depth, sufficient for evaluation.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .model import D3PMUniform, QTFontModel
from .octree import OctreeBatch

# --------------------------------------------------------------------------- #
# D3PM posterior + Gumbel-max sampler.                                         #
# --------------------------------------------------------------------------- #


def _q_posterior_logits(
    diffusion: D3PMUniform,
    x_start_logits: torch.Tensor,
    x_t: torch.Tensor,
    t: int,
    gap: int,
) -> torch.Tensor:
    """Compute ``log p(x_{t-gap} | x_t, x_start)``.

    Mirrors ``q_posterior_logits`` in
    ``third_party/05_qt_font/main.py:215-244``. ``x_start_logits`` carries the
    model's prediction of ``x_0`` (as logits over K classes); we multiply
    `q(x_t | x_{t-gap}, x_start) · q(x_{t-gap} | x_start)` in log-space.

    Parameters
    ----------
    x_start_logits : (N, K) — model logits for x_0.
    x_t            : (N,)   — current categorical state.
    t              : int    — current timestep.
    gap            : int    — stride (1 = no stride).
    """
    K = diffusion.n_states
    Q = diffusion.Q
    Q_T = diffusion.Q_T
    Q_cum = diffusion.Q_cum

    # fact1 = Q^(gap).T at row x_t, broadcast across K dims.
    if gap == 1:
        fact1 = Q_T[t, x_t]  # (N, K)
    else:
        # Re-multiply Q over the gap window: Q^(gap)_{t}.
        # We re-build a Q_tmp cumulative product over [t-gap+1, t].
        Q_tmp = torch.eye(K, device=Q.device, dtype=Q.dtype)
        for gap_t in range(t - gap + 1, t + 1):
            if gap_t < 0:
                continue
            Q_tmp = Q_tmp @ Q[gap_t]
        Q_tmp_T = Q_tmp.t()
        fact1 = Q_tmp_T[x_t]  # (N, K)

    # fact2 = (softmax(x_start_logits)) @ Q_cum[t - gap]
    if t - gap < 0:
        # No further accumulation needed — x_{t-gap} == x_start.
        tzero_logits = x_start_logits
        return tzero_logits
    soft_start = F.softmax(x_start_logits, dim=-1)
    fact2 = soft_start @ Q_cum[t - gap]
    out = torch.log(fact1 + 1e-8) + torch.log(fact2 + 1e-8)
    return out


def _p_sample(
    diffusion: D3PMUniform,
    x_start_logits: torch.Tensor,
    x_t: torch.Tensor,
    t: int,
    gap: int,
) -> torch.Tensor:
    """Sample ``x_{t-gap}`` via Gumbel-max on the posterior logits."""
    model_logits = _q_posterior_logits(diffusion, x_start_logits, x_t, t, gap)
    # On the final step (``t == 0``) the official code uses the raw x_start
    # logits without Gumbel noise. We mirror that.
    if t == 0:
        return model_logits.argmax(dim=-1)
    noise = torch.rand_like(model_logits).clamp(1e-8, 1.0)
    gumbel = -torch.log(-torch.log(noise))
    return torch.argmax(model_logits + gumbel, dim=-1)


# --------------------------------------------------------------------------- #
# Public sampling entrypoints.                                                 #
# --------------------------------------------------------------------------- #


@torch.no_grad()
def sample_labels(
    model: QTFontModel,
    diffusion: D3PMUniform,
    *,
    gt_octree: OctreeBatch,
    cond: torch.Tensor,
    gap: int = 50,
) -> torch.Tensor:
    """Run the reverse process on the GT-octree topology, returning leaf labels.

    Strategy:
      1. Initialise ``x_T`` ~ Uniform({0, …, K-1}) over leaf nodes.
      2. For t in T-gap, T-2*gap, …, 0:
         a. Run model with ``noisy_leaf_label = x_t`` → ``x_0`` logits.
         b. Sample ``x_{t-gap}`` via the D3PM posterior + Gumbel-max.
      3. Return final leaf labels.

    Returns
    -------
    leaf_labels : LongTensor (N_leaf,)
    """
    K = diffusion.n_states
    T = diffusion.timesteps
    leaf = gt_octree.levels[gt_octree.depth]
    device = cond.device

    x_t = torch.randint(0, K, (leaf.xy.shape[0],), device=device, dtype=torch.long)
    for t in range(T - gap, -1, -gap):
        # Build the dynamic noisy leaf label and rerun the model.
        logits_per_depth = model.predict_logits(
            gt_octree, cond, noisy_leaf_label=x_t
        )
        leaf_logits = logits_per_depth[gt_octree.depth]  # (N, K)
        x_t = _p_sample(diffusion, leaf_logits, x_t, t, gap)
    return x_t


@torch.no_grad()
def sample_image(
    model: QTFontModel,
    diffusion: D3PMUniform,
    *,
    batch_size: int,
    content: torch.Tensor,
    refs: torch.Tensor | None = None,
    cond_bundle=None,  # back-compat — accepted, ignored
    gap: int | None = None,
    **_legacy_kwargs,
) -> torch.Tensor:
    """Decode the predicted final state into a pixel image (B, 1, H, W).

    Build helper octrees from the content/ref images, run the reverse process
    on the **content octree's topology** (proxy for the GT topology — at
    inference time we don't have a GT octree). Returns a continuous-valued
    image in ``[-1, +1]`` with bg→-1, contour→0, skeleton→+1.

    The legacy ``cond_bundle``, ``char_id``, ... kwargs are accepted (and
    ignored) for the shared sampler harness.
    """
    del _legacy_kwargs, cond_bundle  # explicit deletion documents the back-compat

    cfg = model.cfg
    from .octree import build_octree_from_image

    # Content octree acts as the topology / decode target.
    target_octree = build_octree_from_image(
        content, full_depth=cfg.full_depth, depth=cfg.depth
    ).to(content.device)
    content_octree = target_octree

    style_octrees = None
    if refs is not None and cfg.use_style:
        R = refs.shape[1]
        style_octrees = [
            build_octree_from_image(
                refs[:, r, :1], full_depth=cfg.full_depth, depth=cfg.depth
            ).to(content.device)
            for r in range(R)
        ]

    timesteps = torch.zeros((batch_size,), device=content.device, dtype=torch.float32)
    cond = model.encode_conditioning(
        timesteps, content_octree=content_octree, style_octrees=style_octrees
    )
    gap_v = gap if gap is not None else max(1, diffusion.timesteps // 20)
    leaf_labels = sample_labels(
        model, diffusion, gt_octree=target_octree, cond=cond, gap=gap_v
    )

    # Render to image.
    leaf_lvl = target_octree.levels[target_octree.depth]
    side = 1 << target_octree.depth
    img = torch.zeros((batch_size, side, side), device=content.device)
    mapping = torch.tensor([-1.0, 0.0, 1.0], device=content.device)
    img[leaf_lvl.batch_id, leaf_lvl.xy[:, 0], leaf_lvl.xy[:, 1]] = mapping[leaf_labels]
    img = img.unsqueeze(1)
    if cfg.image_size != side:
        img = F.interpolate(
            img, size=(cfg.image_size, cfg.image_size), mode="bilinear", align_corners=False
        )
    return img


__all__ = ["sample_labels", "sample_image"]
