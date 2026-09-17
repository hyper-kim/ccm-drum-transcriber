"""
src/train/model.py  (v3 — SE-ResNet, VRAM 최대 활용)

Input:  [B, 1, 128, 20]  (200ms context, 20 time frames)
Output: [B, num_classes]

설계:
  - SE-ResBlock (Squeeze-and-Excitation): 채널 어텐션으로 정확도 향상
  - 채널 수 최대화 (64→128→256→512→512): VRAM 충분히 활용
  - Mel축은 2씩 pooling, Time축 유지 → GAP로 처리
  - InstanceNorm2d: 배치 크기 독립
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SEBlock(nn.Module):
    """Squeeze-and-Excitation channel attention."""
    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = self.fc(x).view(x.size(0), x.size(1), 1, 1)
        return x * scale


class ResConvBlock(nn.Module):
    """
    Residual conv block with SE attention + InstanceNorm.
    pool=(2,1): mel축 절반으로, time축 유지.
    """
    def __init__(self, in_ch: int, out_ch: int,
                 pool: tuple = (2, 1), dropout: float = 0.2):
        super().__init__()
        self.conv1   = nn.Conv2d(in_ch,  out_ch, 3, padding=1, bias=False)
        self.norm1   = nn.InstanceNorm2d(out_ch, affine=True)
        self.conv2   = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.norm2   = nn.InstanceNorm2d(out_ch, affine=True)
        self.se      = SEBlock(out_ch)
        self.pool    = nn.MaxPool2d(pool)
        self.dropout = nn.Dropout2d(dropout)

        # residual projection (채널 수 다를 때)
        if in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, bias=False),
                nn.InstanceNorm2d(out_ch, affine=True),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.shortcut(x)
        x   = F.relu(self.norm1(self.conv1(x)))
        x   = self.norm2(self.conv2(x))
        x   = self.se(x)
        x   = F.relu(x + res)
        x   = self.pool(x)
        return self.dropout(x)


class DrumHitClassifier(nn.Module):
    """
    SE-ResNet drum hit classifier.

    Input:  [B, 1, 128, 20]   (200ms @ 100fps mel)
    After blocks mel 128→64→32→16→8→4 (5 pool steps)
    GAP → [B, 512]
    Output: [B, num_classes]
    """

    def __init__(self, num_classes: int = 11):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(64, affine=True),
            nn.ReLU(),
        )
        # mel: 128 → 64 → 32 → 16 → 8 → 4
        self.block1 = ResConvBlock(64,  128, pool=(2, 1), dropout=0.1)
        self.block2 = ResConvBlock(128, 256, pool=(2, 1), dropout=0.15)
        self.block3 = ResConvBlock(256, 384, pool=(2, 1), dropout=0.2)
        self.block4 = ResConvBlock(384, 512, pool=(2, 1), dropout=0.2)
        self.block5 = ResConvBlock(512, 512, pool=(2, 1), dropout=0.25)

        self.gap = nn.AdaptiveAvgPool2d(1)

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        from torch.utils.checkpoint import checkpoint
        x = self.stem(x)
        x = checkpoint(self.block1, x, use_reentrant=False)
        x = checkpoint(self.block2, x, use_reentrant=False)
        x = checkpoint(self.block3, x, use_reentrant=False)
        x = checkpoint(self.block4, x, use_reentrant=False)
        x = checkpoint(self.block5, x, use_reentrant=False)
        x = self.gap(x)
        return self.classifier(x)
