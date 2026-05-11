"""Stage-VAE pretrain for 02_hfh_font.

Trains ONLY ``model.vae`` (TinyVAE) on reconstruction + KL loss using the
shared TTF dataset for glyph images. Saves a VAE-only state_dict that
``train.py`` stage=a can load as a warm-start before freezing the VAE.

The paper says HFH-Font pretrains the VAE in Stage-A and freezes it for
Stages B/C, but the paper provides no architecture or checkpoint. Our
TinyVAE is a hand-rolled stand-in; this script gives it real glyph
weights instead of the random init the blind impl had.

Usage (from papers/02_hfh_font/):
    uv run python scripts/pretrain_vae.py \\
        --fonts-root D:/Char/ayueh/paper_reimpl/data_snapshot/fonts_free \\
        --output outputs/stage_vae/vae_last.pt \\
        --steps 5000 --batch-size 32 --device cuda:1
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hfh_font.model import ModelConfig, TinyVAE
from paper_reimpl_shared.data.ttf_pair_dataset import TTFCrossFontPairDataset


def _glyph_only_collate(batch: list[dict]) -> torch.Tensor:
    """Stack just the target image tensor; ignore everything else."""
    return torch.stack([b["image"] for b in batch], dim=0)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--fonts-root", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--steps", type=int, default=5000)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--kl-weight", type=float, default=1e-6)
    p.add_argument("--log-every", type=int, default=100)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--image-size", type=int, default=128)
    p.add_argument("--font-size-ratio", type=float, default=0.85)
    # Match HFHFontModel defaults so the trained ckpt loads into the full
    # model's `model.vae` submodule cleanly.
    p.add_argument("--latent-channels", type=int, default=4)
    p.add_argument("--vae-base-channels", type=int, default=32)
    p.add_argument("--down-factor", type=int, default=8)
    args = p.parse_args()

    device = torch.device(args.device)
    torch.manual_seed(args.seed)

    print(f"[vae-pretrain] building TinyVAE: latent={args.latent_channels} down={args.down_factor}")
    vae = TinyVAE(
        in_channels=1,
        base_channels=args.vae_base_channels,
        latent_channels=args.latent_channels,
        down_factor=args.down_factor,
    ).to(device)
    n_params = sum(p.numel() for p in vae.parameters())
    print(f"[vae-pretrain] VAE params = {n_params/1e6:.2f} M")

    print(f"[vae-pretrain] loading TTF dataset from {args.fonts_root}")
    ds = TTFCrossFontPairDataset(
        fonts_root=args.fonts_root,
        image_size=args.image_size,
        content_channels=1,
        font_size_ratio=args.font_size_ratio,
        length=args.batch_size * args.steps + 1024,
        ref_count=1,
        seed=args.seed,
        ensure_diff_source=False,
    )
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=_glyph_only_collate,
        drop_last=True,
    )

    optim = torch.optim.AdamW(vae.parameters(), lr=args.lr, weight_decay=0.0)
    vae.train()

    step = 0
    for batch_img in loader:
        x = batch_img.to(device)
        mean, log_var = vae.encode_distribution(x)
        z = vae.reparameterize(mean, log_var)
        x_hat = vae.decode(z)
        l_recon = F.l1_loss(x_hat, x)
        # β-VAE style: tiny KL so latent stays compact but recon dominates.
        l_kl = -0.5 * (1.0 + log_var - mean.pow(2) - log_var.exp()).mean()
        loss = l_recon + args.kl_weight * l_kl

        optim.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(vae.parameters(), 1.0)
        optim.step()

        if step % args.log_every == 0:
            print(
                f"[vae-pretrain] step={step} loss={loss.item():.4f} "
                f"l_recon={l_recon.item():.4f} l_kl={l_kl.item():.4f}"
            )
        step += 1
        if step >= args.steps:
            break

    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Save in a format compatible with HFHFontModel's `vae` submodule:
    # prefix all keys with "vae." so torch.load + load_state_dict(strict=False)
    # on the full model picks them up.
    state = {f"vae.{k}": v for k, v in vae.state_dict().items()}
    torch.save({"model": state, "stage": "vae", "step": step, "args": vars(args)}, args.output)
    print(f"[vae-pretrain] saved {args.output}  (state_dict has {len(state)} keys, prefixed 'vae.')")
    return 0


if __name__ == "__main__":
    sys.exit(main())
