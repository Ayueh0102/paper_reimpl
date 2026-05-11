"""R_char: lightweight character recognizer used as auxiliary supervision.

Input:  [B, 1, H, W] grayscale image in [-1, 1]
Output: [B, char_vocab_size] logits over char IDs

Used in two roles:
  1. Standalone evaluator (frozen, computes top-1 char accuracy on generated samples)
  2. Auxiliary training loss (frozen during diffusion training; CE between
     classifier(predicted_x0) and target char_label_id pushes the diffusion
     model to produce x0 whose char identity is recoverable)

Architecture: ResNet-style lite (4 stages, ~5M params). Trained from scratch
on calligraphy + TTF rendered images. Generalizes across writers / fonts
because the training set spans both.
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample: nn.Module | None = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x if self.downsample is None else self.downsample(x)
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        out = F.relu(out + identity, inplace=True)
        return out


class RCharResNet(nn.Module):
    """Lite ResNet for character recognition. ~5M params at default size.

    Stages:
        stem:    [B,1,256,256] -> [B,32,64,64]   (Conv 7x7 stride 4 + maxpool 2)
        stage1:  [B,32,64,64]  -> [B,64,32,32]   (2 BasicBlocks, downsample x2)
        stage2:  [B,64,32,32]  -> [B,128,16,16]
        stage3:  [B,128,16,16] -> [B,256,8,8]
        stage4:  [B,256,8,8]   -> [B,512,4,4]
        gap + fc -> [B, num_classes]
    """

    def __init__(self, *, num_classes: int, in_channels: int = 1, base_width: int = 32) -> None:
        super().__init__()
        b = base_width
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, b, kernel_size=7, stride=4, padding=3, bias=False),
            nn.BatchNorm2d(b),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.stage1 = self._make_stage(b, b * 2, n_blocks=2, stride=2)
        self.stage2 = self._make_stage(b * 2, b * 4, n_blocks=2, stride=2)
        self.stage3 = self._make_stage(b * 4, b * 8, n_blocks=2, stride=2)
        self.stage4 = self._make_stage(b * 8, b * 16, n_blocks=2, stride=2)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(b * 16, num_classes)
        self.dropout = nn.Dropout(0.1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    @staticmethod
    def _make_stage(in_c: int, out_c: int, *, n_blocks: int, stride: int) -> nn.Sequential:
        layers = [BasicBlock(in_c, out_c, stride=stride)]
        for _ in range(1, n_blocks):
            layers.append(BasicBlock(out_c, out_c, stride=1))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.gap(x).flatten(1)
        x = self.dropout(x)
        return self.fc(x)


def build_r_char(num_classes: int, in_channels: int = 1, base_width: int = 32) -> RCharResNet:
    return RCharResNet(num_classes=num_classes, in_channels=in_channels, base_width=base_width)


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
