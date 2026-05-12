"""Stage-1 VQGAN pretrain for 03_if_font.

Paper says IF-Font uses a frozen pretrained CompVis ``vq-f8-n256`` tokenizer.
We don't have those weights on the lab server, so this script trains a
shape-compatible stand-in (the existing ``_StubVQGANEncoder`` / ``_StubVQGANDecoder``
+ a 256-entry codebook) on the 13 OFL TTF fonts. The result is a frozen
tokenizer used by Phase-2 AR Transformer training.

Output state_dict matches ``VQTokenizerAdapter`` key layout
(``encoder.*``, ``decoder.*``, ``codebook``), so ``train.py``'s new
``vqgan_local_path`` config option can load it directly into the adapter.

Usage (from papers/03_if_font/):
    uv run python scripts/pretrain_vqgan.py \\
        --fonts-root D:/Char/ayueh/paper_reimpl/data_snapshot/fonts_free \\
        --output outputs/stage_vqgan/vqgan_last.pt \\
        --steps 30000 --batch-size 32 --device cuda:0
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from if_font.model import _StubVQGANDecoder, _StubVQGANEncoder, VQTokenizerConfig
from paper_reimpl_shared.data.ttf_pair_dataset import TTFCrossFontPairDataset


def _collate_image_rgb(batch: list[dict]) -> torch.Tensor:
    """Stack target images, expand 1-channel grayscale to 3-channel RGB."""
    imgs = torch.stack([b["image"] for b in batch], dim=0)
    if imgs.shape[1] == 1:
        imgs = imgs.expand(-1, 3, -1, -1).contiguous()
    return imgs


class VQGANPretrainer(nn.Module):
    """Pretrain wrapper: encoder + codebook (Parameter) + decoder + VQ losses."""

    def __init__(self, cfg: VQTokenizerConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.encoder = _StubVQGANEncoder(cfg)
        self.decoder = _StubVQGANDecoder(cfg)
        # Codebook trainable during pretrain; gets saved as a tensor in state_dict
        # under key "codebook" which matches the adapter's buffer name.
        embed = torch.randn(cfg.codebook_size, cfg.embedding_dim) * 0.02
        self.codebook = nn.Parameter(embed)

    def _quantize_ste(self, z: torch.Tensor):
        b, d, h, w = z.shape
        flat = z.permute(0, 2, 3, 1).reshape(-1, d)
        dist = (
            flat.pow(2).sum(dim=1, keepdim=True)
            - 2 * flat @ self.codebook.t()
            + self.codebook.pow(2).sum(dim=1).unsqueeze(0)
        )
        indices_flat = dist.argmin(dim=1)
        quant_flat = F.embedding(indices_flat, self.codebook)
        quant = quant_flat.view(b, h, w, d).permute(0, 3, 1, 2)
        # Straight-through estimator: encoder gradient flows through `z`.
        quant_st = z + (quant - z).detach()
        return quant_st, quant, indices_flat.view(b, h, w)

    def forward(self, x: torch.Tensor):
        z = self.encoder(x)
        quant_st, quant, indices = self._quantize_ste(z)
        x_hat = self.decoder(quant_st)
        return x_hat, z, quant, indices


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--fonts-root", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--steps", type=int, default=30000)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--commit-weight", type=float, default=0.25)
    p.add_argument("--log-every", type=int, default=200)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--image-size", type=int, default=128)
    p.add_argument("--font-size-ratio", type=float, default=0.85)
    p.add_argument("--codebook-size", type=int, default=256)
    p.add_argument("--embedding-dim", type=int, default=4)
    args = p.parse_args()

    device = torch.device(args.device)
    torch.manual_seed(args.seed)

    cfg = VQTokenizerConfig(
        image_size=args.image_size,
        in_channels=3,
        embedding_dim=args.embedding_dim,
        codebook_size=args.codebook_size,
        downsample_factor=8,
    )
    print(
        f"[vqgan-pretrain] cfg img={cfg.image_size} embed={cfg.embedding_dim} "
        f"codebook={cfg.codebook_size} tokens={cfg.n_tokens}"
    )

    module = VQGANPretrainer(cfg).to(device)
    n_params = sum(p.numel() for p in module.parameters())
    print(f"[vqgan-pretrain] params = {n_params/1e6:.2f} M")

    print(f"[vqgan-pretrain] loading TTF dataset from {args.fonts_root}")
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
        collate_fn=_collate_image_rgb,
        drop_last=True,
    )

    optim = torch.optim.AdamW(module.parameters(), lr=args.lr, weight_decay=0.0)
    module.train()

    step = 0
    for x in loader:
        x = x.to(device)
        x_hat, z, quant, indices = module(x)
        l_recon = F.l1_loss(x_hat, x)
        l_codebook = F.mse_loss(quant, z.detach())
        l_commit = F.mse_loss(z, quant.detach())
        loss = l_recon + l_codebook + args.commit_weight * l_commit

        optim.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(module.parameters(), 1.0)
        optim.step()

        if step % args.log_every == 0:
            with torch.no_grad():
                unique = int(indices.unique().numel())
            print(
                f"[vqgan-pretrain] step={step} loss={loss.item():.4f} "
                f"recon={l_recon.item():.4f} cb={l_codebook.item():.4f} "
                f"commit={l_commit.item():.4f} active_codes={unique}/{cfg.codebook_size}"
            )
        step += 1
        if step >= args.steps:
            break

    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Save state_dict matching VQTokenizerAdapter keys: encoder.*, decoder.*, codebook
    state = module.state_dict()
    torch.save(
        {"model": state, "stage": "vqgan", "step": step, "args": vars(args)},
        args.output,
    )
    print(f"[vqgan-pretrain] saved {args.output} (state_dict keys={len(state)})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
