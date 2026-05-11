"""
Encoder components for DeepUnmixing_v2.

Modules
-------
CNNEncoder          : 1x1 conv spectral reducer  (L -> C), same as original paper.
SwinEncoder         : Swin Transformer with shifted-window attention.
DualAttentionBlock  : Parallel spatial + spectral attention mechanism.

Tensor flow recap (original paper, Samson example):
    Input HSI           : (1, L=156, H=95, W=95)
    After CNNEncoder    : (1, C=24,  H=95, W=95)
    After Transformer   : (1, emb_dim=600)          # emb_dim = P * dim
    Reshape + upscale   : (1, P=3,   H=95, W=95)    # done in unmixer.py
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# =====================================================================
# 1.  CNN Feature Extractor  (identical to original paper, Table I)
# =====================================================================

class CNNEncoder(nn.Module):
    """
    Three-layer 1x1 Conv encoder: L -> 128 -> 64 -> C.

    Parameters
    ----------
    in_channels  : Number of spectral bands (L).
    out_channels : Reduced channel count (C).
                   Original paper: C = (emb_dim * P) // patch_size**2.
    dropout      : Dropout rate after first layer (default 0.25).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        mid_channels: Tuple[int, int] = (128, 64),
        dropout: float = 0.25,
    ):
        super().__init__()
        c1, c2 = mid_channels

        self.net = nn.Sequential(
            # Layer 1
            nn.Conv2d(in_channels, c1, kernel_size=1),
            nn.BatchNorm2d(c1, momentum=0.9),
            nn.Dropout(dropout),
            nn.LeakyReLU(inplace=True),
            # Layer 2
            nn.Conv2d(c1, c2, kernel_size=1),
            nn.BatchNorm2d(c2, momentum=0.9),
            nn.LeakyReLU(inplace=True),
            # Layer 3
            nn.Conv2d(c2, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels, momentum=0.5),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, L, H, W) -> (B, C, H, W)"""
        return self.net(x)


# =====================================================================
# 2.  Swin Transformer Encoder
# =====================================================================

# ---------- helpers ---------------------------------------------------

def _window_partition(x: torch.Tensor, ws: int) -> torch.Tensor:
    """(B, H, W, C) -> (B*nW, ws, ws, C)  where nW = (H/ws)*(W/ws)."""
    B, H, W, C = x.shape
    x = x.view(B, H // ws, ws, W // ws, ws, C)
    return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, ws, ws, C)


def _window_reverse(w: torch.Tensor, ws: int, H: int, W: int) -> torch.Tensor:
    """(B*nW, ws, ws, C) -> (B, H, W, C)."""
    B = int(w.shape[0] / (H * W / ws / ws))
    x = w.view(B, H // ws, W // ws, ws, ws, -1)
    return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)


# ---------- Window Attention -----------------------------------------

class WindowAttention(nn.Module):
    """W-MSA / SW-MSA with learnable relative position bias."""

    def __init__(self, dim: int, window_size: int, num_heads: int,
                 attn_drop: float = 0.0, proj_drop: float = 0.0):
        super().__init__()
        self.dim = dim
        self.ws = window_size
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5

        # Relative position bias table & index
        self.rpb_table = nn.Parameter(
            torch.zeros((2 * window_size - 1) ** 2, num_heads)
        )
        nn.init.trunc_normal_(self.rpb_table, std=0.02)

        coords = torch.stack(torch.meshgrid(
            torch.arange(window_size), torch.arange(window_size), indexing="ij"
        ))  # (2, ws, ws)
        flat = torch.flatten(coords, 1)  # (2, ws*ws)
        rel = flat[:, :, None] - flat[:, None, :]  # (2, ws*ws, ws*ws)
        rel = rel.permute(1, 2, 0).contiguous()
        rel[:, :, 0] += window_size - 1
        rel[:, :, 1] += window_size - 1
        rel[:, :, 0] *= 2 * window_size - 1
        self.register_buffer("rpb_index", rel.sum(-1))  # (ws*ws, ws*ws)

        self.qkv = nn.Linear(dim, 3 * dim)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """(nW*B, N, C) -> (nW*B, N, C)  where N = ws*ws."""
        B_, N, C = x.shape
        nh = self.num_heads
        qkv = self.qkv(x).reshape(B_, N, 3, nh, C // nh).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        attn = (q * self.scale) @ k.transpose(-2, -1)

        # Add relative position bias
        rpb = self.rpb_table[self.rpb_index.view(-1)].view(N, N, -1)
        attn = attn + rpb.permute(2, 0, 1).unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, nh, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, nh, N, N)

        attn = self.attn_drop(attn.softmax(dim=-1))
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        return self.proj_drop(self.proj(x))


# ---------- Swin Transformer Block -----------------------------------

class SwinBlock(nn.Module):
    """Single Swin Transformer block (W-MSA or SW-MSA + FFN)."""

    def __init__(self, dim: int, num_heads: int, window_size: int = 5,
                 shift_size: int = 0, mlp_ratio: float = 4.0,
                 drop: float = 0.0, attn_drop: float = 0.0):
        super().__init__()
        self.ws = window_size
        self.shift = shift_size

        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, window_size, num_heads,
                                    attn_drop=attn_drop, proj_drop=drop)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(int(dim * mlp_ratio), dim),
            nn.Dropout(drop),
        )
        # Cached mask
        self._mask: Optional[torch.Tensor] = None
        self._mask_hw: Tuple[int, int] = (0, 0)

    def _get_mask(self, Hp: int, Wp: int, device: torch.device):
        """Compute SW-MSA attention mask (cached)."""
        if self.shift == 0:
            return None
        if (Hp, Wp) == self._mask_hw and self._mask is not None:
            return self._mask

        img = torch.zeros((1, Hp, Wp, 1), device=device)
        cnt = 0
        for hs in (slice(0, -self.ws), slice(-self.ws, -self.shift),
                    slice(-self.shift, None)):
            for ws in (slice(0, -self.ws), slice(-self.ws, -self.shift),
                       slice(-self.shift, None)):
                img[:, hs, ws, :] = cnt
                cnt += 1
        mw = _window_partition(img, self.ws).view(-1, self.ws ** 2)
        mask = mw.unsqueeze(1) - mw.unsqueeze(2)
        mask = mask.masked_fill(mask != 0, -100.0).masked_fill(mask == 0, 0.0)
        self._mask = mask
        self._mask_hw = (Hp, Wp)
        return mask

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        """(B, H*W, C) -> (B, H*W, C)."""
        B, L, C = x.shape
        shortcut = x
        x = self.norm1(x).view(B, H, W, C)

        # Pad so H, W are divisible by window_size
        pr = (self.ws - W % self.ws) % self.ws
        pb = (self.ws - H % self.ws) % self.ws
        if pr > 0 or pb > 0:
            x = F.pad(x, (0, 0, 0, pr, 0, pb))
        Hp, Wp = x.shape[1], x.shape[2]

        # Cyclic shift
        if self.shift > 0:
            x = torch.roll(x, (-self.shift, -self.shift), (1, 2))

        # Window partition -> attention -> reverse
        xw = _window_partition(x, self.ws).view(-1, self.ws ** 2, C)
        xw = self.attn(xw, mask=self._get_mask(Hp, Wp, x.device))
        x = _window_reverse(xw.view(-1, self.ws, self.ws, C), self.ws, Hp, Wp)

        # Reverse shift
        if self.shift > 0:
            x = torch.roll(x, (self.shift, self.shift), (1, 2))

        # Remove padding
        if pr > 0 or pb > 0:
            x = x[:, :H, :W, :]

        x = shortcut + x.reshape(B, H * W, C)
        x = x + self.mlp(self.norm2(x))
        return x


class SwinEncoder(nn.Module):
    """
    Swin Transformer encoder for hyperspectral unmixing.

    Replaces the original ViT.  Uses window-based attention for
    O(n * ws^2) complexity instead of O(n^2).

    Parameters
    ----------
    in_channels : Channels from CNNEncoder (C).
    emb_dim     : Output embedding size  (= P * dim in original paper).
    spatial_size: Spatial side length H = W (e.g. 95 for Samson).
    depth       : Number of Swin blocks (pairs of W-MSA + SW-MSA).
    num_heads   : Attention heads per block.
    window_size : Window size for attention (default 5).
    mlp_ratio   : FFN expansion ratio.
    """

    def __init__(self, in_channels: int, emb_dim: int, spatial_size: int,
                 depth: int = 2, num_heads: int = 8, window_size: int = 5,
                 mlp_ratio: float = 4.0, drop: float = 0.0,
                 attn_drop: float = 0.0):
        super().__init__()
        self.H = self.W = spatial_size
        dim = in_channels  # token dimension = C

        # Build pairs of (W-MSA, SW-MSA)
        blocks = []
        for i in range(depth):
            blocks.append(SwinBlock(
                dim, num_heads, window_size,
                shift_size=0 if i % 2 == 0 else window_size // 2,
                mlp_ratio=mlp_ratio, drop=drop, attn_drop=attn_drop,
            ))
        self.blocks = nn.ModuleList(blocks)
        self.norm = nn.LayerNorm(dim)

        # Project pooled features to emb_dim (same shape as ViT cls token)
        self.head = nn.Linear(dim, emb_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (B, C, H, W) from CNNEncoder.

        Returns
        -------
        (B, emb_dim) — drop-in replacement for ViT cls token.
        """
        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1).reshape(B, H * W, C)  # (B, H*W, C)

        for blk in self.blocks:
            x = blk(x, H, W)

        x = self.norm(x)
        x = x.mean(dim=1)        # global average pool -> (B, C)
        x = self.head(x)          # project -> (B, emb_dim)
        return x


# =====================================================================
# 3.  Dual Attention Block  (spatial + spectral in parallel)
# =====================================================================

class _SpatialAttention(nn.Module):
    """
    Self-attention across spatial positions.

    Operates on (B, N, C) where N = H*W spatial tokens, C = channels.
    Uses optional windowed attention for efficiency when H*W is large.
    """

    def __init__(self, dim: int, num_heads: int = 4,
                 window_size: Optional[int] = None):
        super().__init__()
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        self.use_windows = window_size is not None
        self.ws = window_size or 0

        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)

    def _full_attn(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        nh = self.num_heads
        qkv = self.qkv(x).reshape(B, N, 3, nh, C // nh).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        attn = ((q * self.scale) @ k.transpose(-2, -1)).softmax(dim=-1)
        return self.proj((attn @ v).transpose(1, 2).reshape(B, N, C))

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        """(B, H*W, C) -> (B, H*W, C)."""
        if not self.use_windows:
            return self._full_attn(x)

        # Windowed spatial attention
        B, _, C = x.shape
        x = x.view(B, H, W, C)
        pr = (self.ws - W % self.ws) % self.ws
        pb = (self.ws - H % self.ws) % self.ws
        if pr > 0 or pb > 0:
            x = F.pad(x, (0, 0, 0, pr, 0, pb))
        Hp, Wp = x.shape[1], x.shape[2]
        xw = _window_partition(x, self.ws).view(-1, self.ws ** 2, C)
        xw = self._full_attn(xw)
        x = _window_reverse(xw.view(-1, self.ws, self.ws, C), self.ws, Hp, Wp)
        if pr > 0 or pb > 0:
            x = x[:, :H, :W, :]
        return x.reshape(B, H * W, C)


class _SpectralAttention(nn.Module):
    """
    Cross-Covariance Attention (XCA) for spectral channel interactions.

    Computes a (hd x hd) attention matrix that captures correlations
    between spectral channel dimensions, instead of (N x N) spatial.
    Based on XCiT (NeurIPS 2021).  O(N * C^2) — very efficient.
    """

    def __init__(self, dim: int, num_heads: int = 4):
        super().__init__()
        self.num_heads = num_heads
        assert dim % num_heads == 0, f"dim={dim} must be divisible by num_heads={num_heads}"
        self.head_dim = dim // num_heads

        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, N, C) -> (B, N, C).  Spectral cross-covariance attention."""
        B, N, C = x.shape
        nh, hd = self.num_heads, self.head_dim

        qkv = self.qkv(x).reshape(B, N, 3, nh, hd).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)  # each: (B, nh, N, hd)

        # L2-normalize q, k for stable cross-covariance
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        # Cross-covariance: (B, nh, hd, N) @ (B, nh, N, hd) -> (B, nh, hd, hd)
        attn = (q.transpose(-2, -1) @ k) * self.temperature
        attn = attn.softmax(dim=-1)

        # Apply: (B, nh, N, hd) @ (B, nh, hd, hd) -> (B, nh, N, hd)
        out = (v @ attn).transpose(1, 2).reshape(B, N, C)
        return self.proj(out)


class DualAttentionBlock(nn.Module):
    """
    Parallel spatial-spectral dual attention.

    Computes spatial attention (between pixels) and spectral attention
    (between channels) in parallel, then fuses with learned gate.

    Parameters
    ----------
    dim           : Token / channel dimension (C from CNNEncoder).
    spatial_size  : Side length H = W.
    num_heads     : Heads for spatial attention.
    spectral_heads: Heads for spectral attention.
    window_size   : If set, use windowed spatial attention.
    mlp_ratio     : FFN expansion ratio.
    """

    def __init__(self, dim: int, spatial_size: int,
                 num_heads: int = 4, spectral_heads: int = 4,
                 window_size: Optional[int] = None,
                 mlp_ratio: float = 4.0):
        super().__init__()
        self.H = self.W = spatial_size

        self.norm_sa = nn.LayerNorm(dim)
        self.spatial_attn = _SpatialAttention(dim, num_heads, window_size)

        self.norm_sp = nn.LayerNorm(dim)
        self.spectral_attn = _SpectralAttention(dim, spectral_heads)

        # Learnable fusion gate
        self.gate = nn.Parameter(torch.tensor(0.5))

        self.norm_ff = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(dim * mlp_ratio), dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (B, C, H, W) from CNNEncoder.

        Returns
        -------
        (B, C, H, W)  — same shape, features enriched with dual attention.
        """
        B, C, H, W = x.shape
        x_flat = x.permute(0, 2, 3, 1).reshape(B, H * W, C)  # (B, N, C)

        # Parallel spatial + spectral attention
        sa_out = self.spatial_attn(self.norm_sa(x_flat), H, W)
        sp_out = self.spectral_attn(self.norm_sp(x_flat))

        # Gated fusion + residual
        g = torch.sigmoid(self.gate)
        x_flat = x_flat + g * sa_out + (1 - g) * sp_out

        # FFN + residual
        x_flat = x_flat + self.mlp(self.norm_ff(x_flat))

        return x_flat.reshape(B, H, W, C).permute(0, 3, 1, 2)  # (B, C, H, W)


# =====================================================================
# Self-tests
# =====================================================================

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    for name, (L, C, H, P, dim) in [
        ("Samson", (156, 24, 95, 3, 200)),
        ("Apex",   (285, 32, 110, 4, 200)),
    ]:
        print(f"\n--- {name} ---")
        emb = P * dim
        x = torch.randn(1, L, H, H, device=device)

        cnn = CNNEncoder(L, C).to(device)
        feat = cnn(x)
        assert feat.shape == (1, C, H, H), f"CNNEncoder: {feat.shape}"
        print(f"  CNNEncoder   {x.shape} -> {feat.shape}")

        da = DualAttentionBlock(C, H, num_heads=4, spectral_heads=4,
                                window_size=5).to(device)
        da_out = da(feat)
        assert da_out.shape == (1, C, H, H), f"DualAttn: {da_out.shape}"
        print(f"  DualAttn     {feat.shape} -> {da_out.shape}")

        sw = SwinEncoder(C, emb, H, depth=2, num_heads=4,
                         window_size=5).to(device)
        emb_out = sw(feat)
        assert emb_out.shape == (1, emb), f"SwinEncoder: {emb_out.shape}"
        print(f"  SwinEncoder  {feat.shape} -> {emb_out.shape}")

    print("\nAll encoder shape checks PASSED!")
