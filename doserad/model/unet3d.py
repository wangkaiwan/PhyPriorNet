"""Lightweight 3D residual U-Net with modality-FiLM conditioning for per-CP
dose regression. Output is non-negative (softplus)."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class FiLM(nn.Module):
    def __init__(self, n_modalities: int, ch: int):
        super().__init__()
        self.emb = nn.Embedding(n_modalities, ch * 2)

    def forward(self, x, modality):
        gb = self.emb(modality)
        g, b = gb.chunk(2, dim=1)
        g = g[:, :, None, None, None]; b = b[:, :, None, None, None]
        return x * (1 + g) + b


class ResBlock(nn.Module):
    def __init__(self, cin, cout, n_modalities, groups=8):
        super().__init__()
        self.n1 = nn.GroupNorm(min(groups, cin), cin)
        self.c1 = nn.Conv3d(cin, cout, 3, padding=1)
        self.n2 = nn.GroupNorm(min(groups, cout), cout)
        self.c2 = nn.Conv3d(cout, cout, 3, padding=1)
        self.film = FiLM(n_modalities, cout)
        self.skip = nn.Conv3d(cin, cout, 1) if cin != cout else nn.Identity()

    def forward(self, x, m):
        h = self.c1(F.silu(self.n1(x)))
        h = self.film(h, m)
        h = self.c2(F.silu(self.n2(h)))
        return h + self.skip(x)


class DilatedContext(nn.Module):
    """ASPP-style 3D context at the bottleneck: REDUCE channels, then parallel DILATED convs
    expand the receptive field cheaply (no quadratic cost) to capture LONG-RANGE lateral
    scatter — the physics the plain CNN's limited RF misses in lung/low-density. The channel
    REDUCTION keeps it light (~1M params) so it isolates the *long-range-context* effect from
    raw capacity, and stays runtime-cheap. Residual + FiLM."""
    def __init__(self, ch, n_modalities, dilations=(1, 2, 4), reduction=4, groups=8):
        super().__init__()
        r = max(ch // reduction, 16)
        self.norm = nn.GroupNorm(min(groups, ch), ch)
        self.reduce = nn.Conv3d(ch, r, 1)
        self.branches = nn.ModuleList(
            [nn.Conv3d(r, r, 3, padding=d, dilation=d) for d in dilations])
        self.fuse = nn.Conv3d(r * len(dilations), ch, 1)
        self.film = FiLM(n_modalities, ch)

    def forward(self, x, m):
        h = self.reduce(F.silu(self.norm(x)))
        h = torch.cat([b(h) for b in self.branches], dim=1)
        return x + self.film(self.fuse(h), m)


class SelfAttn3D(nn.Module):
    """3D self-attention over the (small) bottleneck volume = GLOBAL context. Pre-norm +
    residual. Quadratic in #voxels, so only use at the lowest resolution (e.g. 16^3)."""
    def __init__(self, ch, heads=4, groups=8):
        super().__init__()
        self.norm = nn.GroupNorm(min(groups, ch), ch)
        self.attn = nn.MultiheadAttention(ch, heads, batch_first=True)
        self.proj = nn.Conv3d(ch, ch, 1)

    def forward(self, x, m=None):
        b, c, d, h, w = x.shape
        t = F.silu(self.norm(x)).flatten(2).transpose(1, 2)   # (b, N, c)
        a, _ = self.attn(t, t, t)
        a = a.transpose(1, 2).reshape(b, c, d, h, w)
        return x + self.proj(a)


class DoseUNet3D(nn.Module):
    def __init__(self, in_ch=5, base=32, levels=4, n_modalities=2,
                 bottleneck="plain", attn_heads=4):
        """`bottleneck`: "plain" (default, = original net — keeps old checkpoints loadable),
        "dilated" (ASPP long-range context), "attn" (bottleneck self-attention), or
        "dilated+attn". The enhancement targets the lung/low-density long-range-scatter
        weakness. base (e.g. 64) is set via config. MUST re-benchmark the <=1s gate."""
        super().__init__()
        self.stem = nn.Conv3d(in_ch, base, 3, padding=1)
        chs = [base * (2 ** i) for i in range(levels)]
        self.down = nn.ModuleList()
        self.downsamp = nn.ModuleList()
        for i in range(levels - 1):
            self.down.append(ResBlock(chs[i], chs[i], n_modalities))
            self.downsamp.append(nn.Conv3d(chs[i], chs[i + 1], 2, stride=2))
        self.mid = ResBlock(chs[-1], chs[-1], n_modalities)
        self.ctx = DilatedContext(chs[-1], n_modalities) if "dilated" in bottleneck else None
        self.attn = SelfAttn3D(chs[-1], attn_heads) if "attn" in bottleneck else None
        self.upsamp = nn.ModuleList()
        self.up = nn.ModuleList()
        for i in reversed(range(levels - 1)):
            self.upsamp.append(nn.ConvTranspose3d(chs[i + 1], chs[i], 2, stride=2))
            self.up.append(ResBlock(chs[i] * 2, chs[i], n_modalities))
        self.head = nn.Conv3d(base, 1, 1)

    def forward(self, x, modality):
        m = modality.long()
        h = self.stem(x)
        skips = []
        for blk, ds in zip(self.down, self.downsamp):
            h = blk(h, m); skips.append(h); h = ds(h)
        h = self.mid(h, m)
        if self.ctx is not None:
            h = self.ctx(h, m)
        if self.attn is not None:
            h = self.attn(h, m)
        for us, blk, sk in zip(self.upsamp, self.up, reversed(skips)):
            h = us(h)
            h = blk(torch.cat([h, sk], dim=1), m)
        return F.softplus(self.head(h))
