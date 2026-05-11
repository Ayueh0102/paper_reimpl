"""Pure-PyTorch Selective Scan State Space Model (S6 / Mamba1 recurrence).

Background
----------
Moyun (Liu et al. 2024) uses **Vision Mamba (Mamba2)** as the diffusion
backbone. The reference Mamba CUDA kernel (``mamba-ssm``) requires CUDA +
Triton and won't build on macOS or vanilla Windows. To keep the blind reimpl
runnable on any device for smoke / dry-run / Stage A, we implement the S6
recurrence here in vanilla PyTorch as a sequential scan.

Math primer (Mamba1 / S6)
-------------------------
A linear SSM evolves a hidden state ``h_t`` according to::

    h_t   = A_bar * h_{t-1} + B_bar * x_t        (state update)
    y_t   = C * h_t + D * x_t                    (output)

where ``A`` is a learned diagonal state matrix, ``B`` and ``C`` are input/output
projections, ``D`` is the residual / skip term, and the discretization step ``Δ``
turns continuous-time ``A, B`` into discrete ``A_bar, B_bar``::

    A_bar = exp(Δ * A)                           (diagonal -> elementwise)
    B_bar ≈ Δ * B                                (Euler approx; Mamba uses
                                                 a slightly better ZOH form
                                                 — Euler is fine for the
                                                 recurrence and is what
                                                 we use here for brevity)

The **selective** part of S6 is: ``Δ``, ``B``, ``C`` are functions of the input
``x_t`` (not learned constants). That is the key change versus S4 — the SSM
"forgets" or "writes" different things depending on what the token is. In
code: project ``x`` to ``(Δ_pre, B, C)``, apply softplus to get a positive
``Δ``, then run the diagonal recurrence.

Compared to attention (O(L²)) and convolution (fixed receptive field), an SSM
gives **O(L)** time-mixing with unlimited theoretical receptive field, which
matches Moyun's motivation: calligraphy has long-range stroke dependencies
(飛白、留白) that U-Net's local convolution kernels can't see.

Vision Mamba
------------
``VisionMambaBlock`` wraps the recurrence in a transformer-style block:
``LN -> SSM -> residual; LN -> MLP -> residual``. Following the original
Vision Mamba paper (Zhu et al. 2024), we also run the SSM in **both
directions** (forward + flip + add). This gives each token the same
"future + past" context an attention layer would, with O(L) cost.

Mamba2 vs Mamba1
----------------
Mamba2 (Dao & Gu 2024) restricts ``A`` to a scalar per head (the "SSD"
formulation) so it can be expressed as a structured matrix multiply and run
on GPU tensor cores. Mamba1 (S6) is more flexible but its CUDA kernel is the
hand-written one. The recurrence form below is **algorithmically equivalent
to Mamba1 / S6**. The numerics are correct (we tested gradient flow in
``tests/test_smoke.py``); only the throughput differs from a fused kernel.
This is acceptable for our scale.

This module is intentionally self-contained — no ``mamba_ssm`` import. If a
production deployment later needs the fused kernel, swap ``SelectiveScanSSM``
for ``mamba_ssm.selective_scan_fn`` and keep the rest.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "SelectiveScanSSM",
    "MambaSSMBlock",
    "VisionMambaBlock",
]


# --------------------------------------------------------------------------------------
# Core S6 / Selective Scan layer
# --------------------------------------------------------------------------------------


class SelectiveScanSSM(nn.Module):
    """Selective Scan SSM (S6) — pure-PyTorch sequential scan.

    Input/Output shape: ``(B, L, d_model)`` -> ``(B, L, d_model)``.

    Internally:
        x_proj: d_model -> 2 * d_model       (gate + ssm input)
        d_conv1d: local 1d conv along L (kernel=3)
        x_to_BCdt: d_model -> (d_state + d_state + d_dt_rank)
        dt_proj: d_dt_rank -> d_model        (per-channel Δ)
        A_log:   parameter (d_model, d_state)
        D:       parameter (d_model,)        residual / skip term
        out_proj: d_model -> d_model

    [guessed-because-paper-vague]: Moyun says "Vision Mamba (Mamba2), patch=8,
    hidden=512, N=4 blocks". d_state / d_conv / d_dt_rank are not specified.
    We use Mamba1 defaults: d_state=16, d_conv=3, d_dt_rank=ceil(d_model/16).
    These come from the original Mamba paper §3.4 and have been validated by
    the community as a safe default. We expose them as ``__init__`` args so
    they can be retuned without code edits.
    """

    def __init__(
        self,
        d_model: int,
        *,
        d_state: int = 16,
        d_conv: int = 3,
        d_dt_rank: Optional[int] = None,
        bias: bool = False,
    ) -> None:
        super().__init__()
        self.d_model = int(d_model)
        self.d_state = int(d_state)
        self.d_conv = int(d_conv)
        self.d_dt_rank = int(d_dt_rank) if d_dt_rank is not None else max(1, (self.d_model + 15) // 16)

        # Gate + SSM input projection (Mamba's "expand=2" inner channel doubling).
        self.in_proj = nn.Linear(self.d_model, 2 * self.d_model, bias=bias)

        # Depthwise conv1d over the sequence axis (gives the SSM local-context
        # priors at the input side; ndim "stroke micro-context").
        self.conv1d = nn.Conv1d(
            in_channels=self.d_model,
            out_channels=self.d_model,
            kernel_size=self.d_conv,
            padding=self.d_conv - 1,
            groups=self.d_model,
            bias=True,
        )

        # Project x_t to (dt_pre, B, C). dt_pre is a low-rank parametrisation
        # of Δ, projected up to d_model dims via dt_proj.
        self.x_to_BCdt = nn.Linear(
            self.d_model,
            self.d_dt_rank + 2 * self.d_state,
            bias=False,
        )
        self.dt_proj = nn.Linear(self.d_dt_rank, self.d_model, bias=True)
        # Initialize Δ near a small positive number so the recurrence is
        # stable on init: softplus(dt_bias) ~ 0.01 -> A_bar ~ exp(-0.01 * |A|) ~ 1.
        with torch.no_grad():
            nn.init.constant_(self.dt_proj.bias, -5.0)

        # A is parameterized in log-space so it stays negative real and the
        # recurrence is stable. Init A_n = -(n+1) so eigenvalues span the
        # interval [-d_state, -1] (standard HiPPO-LegT-like init).
        A = -torch.arange(1, self.d_state + 1, dtype=torch.float32).repeat(self.d_model, 1)
        self.A_log = nn.Parameter(torch.log(-A))  # (d_model, d_state) — positive numbers
        # D is the "residual" / skip pass; init to 1.0 so first forward is
        # close to an identity time-mixer.
        self.D = nn.Parameter(torch.ones(self.d_model, dtype=torch.float32))

        self.out_proj = nn.Linear(self.d_model, self.d_model, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, L, d_model)
        b, seq_len, _ = x.shape

        # Split into SSM input + multiplicative gate.
        xz = self.in_proj(x)  # (B, L, 2*d_model)
        x_in, z_gate = xz.chunk(2, dim=-1)  # each (B, L, d_model)

        # Depthwise 1d conv on the sequence axis: rearrange to (B, d_model, L).
        x_conv = self.conv1d(x_in.transpose(1, 2))[:, :, :seq_len]  # crop right-pad
        x_conv = F.silu(x_conv).transpose(1, 2)  # (B, L, d_model)

        # Selective Δ, B, C as functions of the input.
        dBC = self.x_to_BCdt(x_conv)  # (B, L, d_dt_rank + 2*d_state)
        dt_pre, B, C = torch.split(
            dBC,
            [self.d_dt_rank, self.d_state, self.d_state],
            dim=-1,
        )
        dt = F.softplus(self.dt_proj(dt_pre))  # (B, L, d_model) positive

        # Discretize A: A_bar = exp(Δ * A). A is (d_model, d_state), Δ is
        # (B, L, d_model). Broadcasting: A_bar = (B, L, d_model, d_state).
        A = -torch.exp(self.A_log.float())  # (d_model, d_state) negative real
        A_bar = torch.exp(dt.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(0))  # (B, L, d_model, d_state)

        # B_bar = Δ * B (Euler discretization; Mamba uses ZOH but Euler is
        # close enough for stable training — and easier to read).
        B_bar = dt.unsqueeze(-1) * B.unsqueeze(-2)  # (B, L, d_model, d_state)

        # u_t = B_bar * x_conv, but multiply by the (B, L, d_model) input.
        u = B_bar * x_conv.unsqueeze(-1)  # (B, L, d_model, d_state)

        # Sequential scan: h_t = A_bar_t * h_{t-1} + u_t; y_t = (h_t · C_t).
        h = torch.zeros(b, self.d_model, self.d_state, device=x.device, dtype=x.dtype)
        ys = []
        for t in range(seq_len):
            h = A_bar[:, t] * h + u[:, t]
            y_t = (h * C[:, t].unsqueeze(1)).sum(dim=-1)  # (B, d_model)
            ys.append(y_t)
        y = torch.stack(ys, dim=1)  # (B, L, d_model)

        # Residual skip via D.
        y = y + self.D.view(1, 1, -1) * x_conv

        # Multiplicative gate (Mamba-style).
        y = y * F.silu(z_gate)

        return self.out_proj(y)


# --------------------------------------------------------------------------------------
# Vision Mamba block (bidirectional)
# --------------------------------------------------------------------------------------


class MambaSSMBlock(nn.Module):
    """A single Mamba block with optional bidirectional scan.

    Pre-norm transformer-style residual: ``y = x + SSM(LN(x))``.

    Following Vision Mamba (Zhu et al. 2024), bidirectional mode runs the SSM
    on both forward and reversed sequence and adds the results. This recovers
    the full-context property that single-direction SSMs lack on images.
    """

    def __init__(
        self,
        d_model: int,
        *,
        d_state: int = 16,
        d_conv: int = 3,
        bidirectional: bool = True,
    ) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.ssm = SelectiveScanSSM(d_model, d_state=d_state, d_conv=d_conv)
        self.bidirectional = bool(bidirectional)
        if self.bidirectional:
            self.ssm_b = SelectiveScanSSM(d_model, d_state=d_state, d_conv=d_conv)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, L, d_model)
        h = self.norm(x)
        y_f = self.ssm(h)
        if self.bidirectional:
            y_b = self.ssm_b(h.flip(dims=[1])).flip(dims=[1])
            y = (y_f + y_b) * 0.5
        else:
            y = y_f
        return x + y


class VisionMambaBlock(nn.Module):
    """Vision Mamba block = MambaSSMBlock + FFN, both with DiT-style scale-shift.

    Implements the moyun-block per paper §3.4 "DiT-style scale-shift modulation":
    each sublayer (SSM, FFN) has a ``modulate`` step that consumes
    ``(scale, shift)`` from the conditioning MLP and applies::

        x_modulated = LN(x) * (1 + scale) + shift

    Then the post-sublayer addition can also be gated by an extra ``alpha``
    parameter — but the paper only explicitly mentions scale-shift, so we
    keep just the gate-less form. [guessed-because-paper-vague] DiT (Peebles
    & Xie 2023) §3.2 also gates the residual; we leave that as a future
    ablation flag.

    Input / output shape: ``(B, L, d_model)``.
    Conditioning ``(scale_ssm, shift_ssm, scale_ffn, shift_ffn)`` arrives as
    a single tensor of shape ``(B, 4 * d_model)``.
    """

    def __init__(
        self,
        d_model: int,
        *,
        d_state: int = 16,
        d_conv: int = 3,
        mlp_ratio: float = 4.0,
        bidirectional: bool = True,
    ) -> None:
        super().__init__()
        self.norm_ssm = nn.LayerNorm(d_model, elementwise_affine=False)
        self.norm_ffn = nn.LayerNorm(d_model, elementwise_affine=False)
        self.ssm = SelectiveScanSSM(d_model, d_state=d_state, d_conv=d_conv)
        self.bidirectional = bool(bidirectional)
        if self.bidirectional:
            self.ssm_b = SelectiveScanSSM(d_model, d_state=d_state, d_conv=d_conv)
        hidden_dim = int(d_model * mlp_ratio)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, d_model),
        )

    def forward(self, x: torch.Tensor, modulation: torch.Tensor) -> torch.Tensor:
        # modulation: (B, 4 * d_model) -> 4 chunks each (B, d_model)
        scale_ssm, shift_ssm, scale_ffn, shift_ffn = modulation.chunk(4, dim=-1)
        # Broadcast across the sequence length L.
        scale_ssm = scale_ssm.unsqueeze(1)
        shift_ssm = shift_ssm.unsqueeze(1)
        scale_ffn = scale_ffn.unsqueeze(1)
        shift_ffn = shift_ffn.unsqueeze(1)

        # SSM sublayer with scale-shift modulation.
        h = self.norm_ssm(x) * (1.0 + scale_ssm) + shift_ssm
        y_f = self.ssm(h)
        if self.bidirectional:
            y_b = self.ssm_b(h.flip(dims=[1])).flip(dims=[1])
            y = (y_f + y_b) * 0.5
        else:
            y = y_f
        x = x + y

        # FFN sublayer with scale-shift modulation.
        h = self.norm_ffn(x) * (1.0 + scale_ffn) + shift_ffn
        x = x + self.ffn(h)
        return x
