"""QT-Font losses — multi-depth split CE + leaf-depth axis CE.

Mirrors ``third_party/05_qt_font/losses/loss.py:14-31`` (``axis_loss``):

    for d in logits.keys():
        if d == max_depth:
            label_gt = F[:, 2] * 1 + F[:, 3] * 2   # 3-class axis label
        else:
            label_gt = octree.nempty_mask(d).long()
        loss_d = F.cross_entropy(logits[d], label_gt)

Phase 2 deltas covered: paper loss_deltas L1 (multi-depth supervision), L2
(per-depth weight vector, default all-ones).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .octree import OctreeBatch


def compute_multi_depth_ce(
    logits_per_depth: dict[int, torch.Tensor],
    gt_octree: OctreeBatch,
    *,
    weights_per_depth: dict[int, float] | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Multi-depth cross-entropy loss.

    Parameters
    ----------
    logits_per_depth : dict[int, Tensor]
        ``logits_per_depth[d]`` shape ``(N_d, 2)`` for inner depths,
        ``(N_d, K)`` (K=3) at the leaf depth.
    gt_octree : OctreeBatch
        Ground-truth octree built from the clean glyph. Provides per-depth
        ``split`` (inner) and ``leaf_label`` (leaf) targets.
    weights_per_depth : dict[int, float], optional
        Per-depth loss weight. Default = 1.0 everywhere, matching the official
        ``compute_octree_loss`` ``weights = [1.0] * 16`` (loss.py:39-40).

    Returns
    -------
    loss : scalar tensor
    log  : dict of per-depth losses + accuracies
    """
    log: dict[str, float] = {}
    parts: list[torch.Tensor] = []
    leaf_depth = gt_octree.depth

    for d in sorted(logits_per_depth.keys()):
        logit_d = logits_per_depth[d]
        lvl = gt_octree.levels[d]
        if d == leaf_depth:
            label_gt = lvl.leaf_label
            if label_gt is None:
                raise RuntimeError(
                    f"leaf level d={d} has leaf_label=None — build_octree_from_labels missed it"
                )
        else:
            label_gt = lvl.split

        # Defensive guard — if node-count mismatches between logits and labels
        # we'd silently misalign supervision. This trips loudly if the dataset
        # (used to build gt_octree) and the model's noisy_octree disagree on
        # topology at level d.
        if logit_d.shape[0] != label_gt.shape[0]:
            raise RuntimeError(
                f"logits depth={d} has N={logit_d.shape[0]} but label has N={label_gt.shape[0]}; "
                "the gt octree and the noisy octree have inconsistent topology"
            )

        loss_d = F.cross_entropy(logit_d, label_gt)
        w = 1.0 if weights_per_depth is None else float(weights_per_depth.get(d, 1.0))
        parts.append(loss_d * w)
        with torch.no_grad():
            acc = (logit_d.argmax(dim=-1) == label_gt).float().mean().item()
            log[f"loss_d{d}"] = float(loss_d.item())
            log[f"acc_d{d}"] = float(acc)

    loss = torch.stack(parts).sum()
    log["loss_total"] = float(loss.item())
    return loss, log


__all__ = ["compute_multi_depth_ce"]
