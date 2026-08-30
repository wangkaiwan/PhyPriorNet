"""Lightweight 2D U-Net for MRI->sCT (HU regression). 2.5D context via the
input channel dim (caller stacks (2k+1) adjacent slices)."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class _Block(nn.Module):
    def __init__(self, cin, cout, groups=8):
        super().__init__()
        self.n1 = nn.GroupNorm(min(groups, cin), cin)
        self.c1 = nn.Conv2d(cin, cout, 3, padding=1)
        self.n2 = nn.GroupNorm(min(groups, cout), cout)
        self.c2 = nn.Conv2d(cout, cout, 3, padding=1)
        self.skip = nn.Conv2d(cin, cout, 1) if cin != cout else nn.Identity()

    def forward(self, x):
        h = self.c1(F.silu(self.n1(x)))
        h = self.c2(F.silu(self.n2(h)))
        return h + self.skip(x)


class SCTUNet(nn.Module):
    def __init__(self, in_ch=5, base=32, levels=4):
        super().__init__()
        chs = [base * (2 ** i) for i in range(levels)]
        self.stem = nn.Conv2d(in_ch, base, 3, padding=1)
        self.down = nn.ModuleList()
        self.ds = nn.ModuleList()
        for i in range(levels - 1):
            self.down.append(_Block(chs[i], chs[i]))
            self.ds.append(nn.Conv2d(chs[i], chs[i + 1], 2, stride=2))
        self.mid = _Block(chs[-1], chs[-1])
        self.us = nn.ModuleList()
        self.up = nn.ModuleList()
        for i in reversed(range(levels - 1)):
            self.us.append(nn.ConvTranspose2d(chs[i + 1], chs[i], 2, stride=2))
            self.up.append(_Block(chs[i] * 2, chs[i]))
        self.head = nn.Conv2d(base, 1, 1)

    def forward(self, x):
        h = self.stem(x)
        skips = []
        for blk, d in zip(self.down, self.ds):
            h = blk(h); skips.append(h); h = d(h)
        h = self.mid(h)
        for u, blk, sk in zip(self.us, self.up, reversed(skips)):
            h = u(h); h = blk(torch.cat([h, sk], 1))
        return self.head(h)
