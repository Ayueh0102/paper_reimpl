"""Minimal LoRA wrapper for ``nn.Linear``.

Used at Stage C / one-shot style transfer. Keeps the implementation
self-contained so the smoke test never imports ``peft``. The real training
pipeline can swap in ``peft.get_peft_model`` if desired — the rest of the
code only relies on ``apply_lora_to_module`` and ``lora_parameters``.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Iterator

import torch
import torch.nn as nn
import torch.nn.functional as F


class LoraLinear(nn.Module):
    """Low-rank residual adapter wrapping an existing ``nn.Linear``.

    ``y = base(x) + (scaling) * B(A(x))``

    Initialised so the adapter is a no-op at step 0 (``B = 0``). Base weights
    are frozen by default; only ``A`` and ``B`` train.
    """

    def __init__(
        self,
        base: nn.Linear,
        *,
        rank: int = 4,
        alpha: float = 8.0,
        dropout: float = 0.0,
        freeze_base: bool = True,
    ) -> None:
        super().__init__()
        if not isinstance(base, nn.Linear):
            raise TypeError(f"LoraLinear expects nn.Linear, got {type(base).__name__}")
        self.base = base
        if freeze_base:
            for p in self.base.parameters():
                p.requires_grad = False
        in_features = base.in_features
        out_features = base.out_features
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scaling = self.alpha / max(1, self.rank)
        self.lora_A = nn.Parameter(torch.empty(self.rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, self.rank))
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base(x)
        adapter_out = F.linear(self.dropout(x), self.lora_A)
        adapter_out = F.linear(adapter_out, self.lora_B)
        return base_out + self.scaling * adapter_out

    def lora_parameters(self) -> Iterator[nn.Parameter]:
        yield self.lora_A
        yield self.lora_B


def apply_lora_to_module(
    module: nn.Module,
    *,
    target_substrings: Iterable[str] = ("to_q", "to_k", "to_v", "to_out"),
    rank: int = 4,
    alpha: float = 8.0,
    dropout: float = 0.0,
) -> int:
    """Replace ``nn.Linear`` submodules whose qualified name matches any
    substring with a ``LoraLinear``. Returns the number of layers wrapped.
    """
    wrapped = 0
    targets: list[tuple[str, nn.Linear]] = []
    for name, sub in module.named_modules():
        if not isinstance(sub, nn.Linear):
            continue
        if not any(t in name for t in target_substrings):
            continue
        targets.append((name, sub))
    for name, sub in targets:
        parent_name, _, child_name = name.rpartition(".")
        parent = module.get_submodule(parent_name) if parent_name else module
        setattr(parent, child_name, LoraLinear(sub, rank=rank, alpha=alpha, dropout=dropout))
        wrapped += 1
    return wrapped


def lora_parameters(module: nn.Module) -> list[nn.Parameter]:
    """Collect every ``LoraLinear``'s A and B parameter for the optimizer."""
    params: list[nn.Parameter] = []
    for sub in module.modules():
        if isinstance(sub, LoraLinear):
            params.extend(list(sub.lora_parameters()))
    return params


def freeze_non_lora(module: nn.Module) -> None:
    """Freeze every parameter that is not a ``LoraLinear`` A/B."""
    lora_ids = {id(p) for p in lora_parameters(module)}
    for p in module.parameters():
        if id(p) not in lora_ids:
            p.requires_grad = False
