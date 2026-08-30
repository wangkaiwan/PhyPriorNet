"""Masked GroupNorm v2 — padded-batch inference EXACTLY equal to per-sample inference.

Two ingredients (both required for exactness):
1. GroupNorm statistics from the VALID region only (per-sample slice; the padding zeros
   otherwise shift mean/var — the effect measured in the S12 accel study).
2. Re-zero the padding region after EVERY leaf module (hooks): GN's affine (and conv biases)
   turn padding zeros into nonzero constants; the next conv's valid-edge outputs would read
   them, whereas the unpadded reference sees implicit zero padding there. Zeroing after each
   layer restores the invariant "beyond the valid extent = 0 entering every op", which makes
   all spatially-local ops (convs, strided/transpose convs, dilated ASPP, activations with
   f(0)=0) reproduce the unpadded forward exactly inside the valid region.

Valid-extent at a feature map of size D vs input D0: k = round(log2(D0/D)), extent = ceil(d0/2^k)
(k=3,s=2,p=1 convs ceil-halve, and ceil-halving composes; transpose-conv up mirrors it).

Usage:
    from accel.masked_gn import install_masked_batching, valid_extents
    install_masked_batching(net)              # swap GNs + install zeroing hooks (idempotent)
    with valid_extents([(d,h,w), ...], full=(D0,H0,W0)):
        y = net(x_padded_batch, m)
    # outside the context: zero overhead, plain group_norm path.
"""
from __future__ import annotations

import contextlib
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

_CTX = {"extents": None, "full": None}


@contextlib.contextmanager
def valid_extents(extents, full):
    prev = (_CTX["extents"], _CTX["full"])
    _CTX["extents"], _CTX["full"] = list(extents), tuple(full)
    try:
        yield
    finally:
        _CTX["extents"], _CTX["full"] = prev


def _level_extents(D, H, W):
    """Per-sample valid (d,h,w) at a feature map of spatial size (D,H,W)."""
    D0, H0, W0 = _CTX["full"]
    k = max(0, round(math.log2(max(D0 / max(D, 1), 1e-9))))
    s = 1 << k
    return [(min(D, -(-d0 // s)), min(H, -(-h0 // s)), min(W, -(-w0 // s)))
            for d0, h0, w0 in _CTX["extents"]]


def _zero_padding_(t: torch.Tensor) -> torch.Tensor:
    """In-place: zero everything beyond each sample's valid extent."""
    if _CTX["extents"] is None or t.ndim != 5 or t.shape[0] != len(_CTX["extents"]):
        return t
    B, C, D, H, W = t.shape
    for i, (d, h, w) in enumerate(_level_extents(D, H, W)):
        if d < D:
            t[i, :, d:].zero_()
        if h < H:
            t[i, :, :d, h:].zero_()
        if w < W:
            t[i, :, :d, :h, w:].zero_()
    return t


class MaskedGroupNorm(nn.GroupNorm):
    """Per-sample valid-region statistics inside the context; plain group_norm outside."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if _CTX["extents"] is None or x.ndim != 5 or x.shape[0] != len(_CTX["extents"]):
            return F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
        B, C, D, H, W = x.shape
        out = torch.empty_like(x)
        for i, (d, h, w) in enumerate(_level_extents(D, H, W)):
            xi = x[i: i + 1, :, :d, :h, :w]
            yi = F.group_norm(xi.float(), self.num_groups, self.weight.float(),
                              self.bias.float(), self.eps).to(x.dtype)
            out[i].zero_()
            out[i, :, :d, :h, :w] = yi[0]
        return out


def install_masked_batching(net: nn.Module) -> tuple[int, int]:
    """Swap nn.GroupNorm -> MaskedGroupNorm (shared params) + add padding-zeroing hooks to
    every leaf module. Idempotent. Returns (n_gn_swapped, n_hooks)."""
    n_gn = 0
    for mod in list(net.modules()):
        for cname, child in list(mod.named_children()):
            if type(child) is nn.GroupNorm:
                mg = MaskedGroupNorm(child.num_groups, child.num_channels, eps=child.eps,
                                     affine=child.affine)
                if child.affine:
                    mg.weight = child.weight; mg.bias = child.bias
                setattr(mod, cname, mg)
                n_gn += 1
    n_hooks = 0
    for mod in net.modules():
        if len(list(mod.children())) == 0 and not getattr(mod, "_mgn_hooked", False):
            mod.register_forward_hook(
                lambda m, inp, out: _zero_padding_(out) if torch.is_tensor(out) else out)
            mod._mgn_hooked = True
            n_hooks += 1
    return n_gn, n_hooks
