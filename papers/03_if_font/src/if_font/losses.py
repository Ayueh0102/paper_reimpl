"""IF-Font Phase-2 losses.

Two training losses, matching official `iffont/modules/losses.py`:

  * ``sq(logits, target)``  — CE on next-VQ-token prediction.
                              "soft-quantize" loss (paper's "L_AR" / "L_sq").
  * ``sup_cl(features, labels)`` — supervised contrastive loss on MoCo
                              style features keyed by font_id / writer_id.

Phase-2 drops the Phase-1 (vq_commit, recon_mse) terms because the VQGAN is
now a frozen pretrained tokenizer — there is nothing to commit to and the
reconstruction MSE is decoupled from the AR model entirely.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

__all__ = ["sq", "sup_cl"]


def sq(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Cross-entropy on flattened (logits, target_indices).

    Args:
        logits: [B, T, K] — VQ token logits over the codebook.
        target: [B, T] long — ground-truth VQ indices.
    """
    logits = logits.reshape(-1, logits.shape[-1])
    target = target.reshape(-1)
    return F.cross_entropy(logits, target)


def sup_cl(
    features: torch.Tensor,
    *,
    labels: torch.Tensor | None = None,
    mask: torch.Tensor | None = None,
    temperature: float = 0.07,
    contrast_mode: str = "all",
    base_temperature: float = 0.07,
) -> torch.Tensor:
    """Supervised Contrastive Learning loss (Khosla et al. 2020).

    Matches official `iffont/modules/losses.sup_cl` (Phase-2 alignment).

    Args:
        features: [B, V, D] — V views per sample (MoCo has V=2: query+momentum).
        labels: [B] long — class label (font_id / writer_id). If None and
            mask is None, degenerates to SimCLR (instance-discrimination).
        mask: [B, B] bool — explicit positive mask (XOR with labels).
        temperature, base_temperature: standard supcon scaling.
        contrast_mode: 'all' (anchors = all views) or 'one' (anchors = view 0).
    """
    features = F.normalize(features, p=2, dim=2)
    device = features.device
    batch_size, contrast_count = features.shape[0], features.shape[1]

    if features.dim() < 3:
        raise ValueError("`features` needs to be [B, V, ...]")
    if features.dim() > 3:
        features = features.view(batch_size, contrast_count, -1)

    if labels is not None and mask is not None:
        raise ValueError("Cannot provide both `labels` and `mask`")
    if labels is None and mask is None:
        mask_t = torch.eye(batch_size, dtype=torch.float32, device=device)
    elif labels is not None:
        labels = labels.contiguous().view(-1, 1)
        if labels.shape[0] != batch_size:
            raise ValueError("Num of labels does not match num of features")
        mask_t = torch.eq(labels, labels.T).float().to(device)
    else:
        mask_t = mask.float().to(device)  # type: ignore[union-attr]

    contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
    if contrast_mode == "one":
        anchor_feature = features[:, 0]
        anchor_count = 1
    elif contrast_mode == "all":
        anchor_feature = contrast_feature
        anchor_count = contrast_count
    else:
        raise ValueError(f"Unknown contrast_mode: {contrast_mode}")

    anchor_dot_contrast = torch.div(
        torch.matmul(anchor_feature, contrast_feature.T), temperature
    )
    logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
    logits = anchor_dot_contrast - logits_max.detach()

    mask_t = mask_t.repeat(anchor_count, contrast_count)
    logits_mask = torch.scatter(
        torch.ones_like(mask_t),
        1,
        torch.arange(batch_size * anchor_count, device=device).view(-1, 1),
        0,
    )
    mask_t = mask_t * logits_mask

    exp_logits = torch.exp(logits) * logits_mask
    log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-12)

    pos_per_anchor = mask_t.sum(1)
    # Guard against rows with zero positives (e.g. unique label in the batch):
    # those rows contribute 0 to the loss.
    safe = pos_per_anchor.clamp(min=1.0)
    mean_log_prob_pos = (mask_t * log_prob).sum(1) / safe
    loss = -(temperature / base_temperature) * mean_log_prob_pos
    loss = loss.view(anchor_count, batch_size)
    # Zero out rows where no positives exist.
    mask_valid = (pos_per_anchor > 0).view(anchor_count, batch_size).float()
    if mask_valid.sum() == 0:
        return torch.zeros((), device=device)
    return (loss * mask_valid).sum() / mask_valid.sum()
