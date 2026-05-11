"""DP-Font (IJCAI 2024) blind reimplementation.

Diffusion + Physics-Informed Neural Network (PINN) for Chinese calligraphy
generation. Adds two physical-prior loss terms (ink-diffusion residual and
nib-motion smoothness) to the standard DDPM denoising objective, plus
multi-attribute guidance (writer / script / char / ink_intensity / font_size)
and a stroke-order sequence condition.

Public API:
    DPFont, DPFontConfig, build_dp_font  — the conditional U-Net.
    pinn_losses  — submodule with physics-informed loss terms.
    compute_loss  — combines L_simple + λ_PINN * L_PINN.
    main  — training entry point dispatched from
            ``paper_reimpl_shared.runner.entrypoint``.
"""
from .model import DPFont, DPFontConfig, build_dp_font

__all__ = ["DPFont", "DPFontConfig", "build_dp_font"]
