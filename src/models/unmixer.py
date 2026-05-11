"""
DeepUnmixer -- configurable AutoEncoder for hyperspectral unmixing.

Supports two configurations:
    baseline  : CNNEncoder + ViT  + LinearDecoder    (original paper)
    improved  : CNNEncoder + DualAttention + Swin + NonLinearDecoder
"""

from __future__ import annotations
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import repeat
from einops.layers.torch import Rearrange

from .encoders import CNNEncoder, SwinEncoder, DualAttentionBlock
from .decoders import LinearDecoder, NonLinearDecoder
from ..core.initialization import initialize


# =====================================================================
# Baseline ViT (faithful re-implementation of original paper)
# =====================================================================

class _CrossAttention(nn.Module):
    """Multihead Self-Patch Attention — q from cls, k/v from all."""

    def __init__(self, dim: int, num_heads: int = 8):
        super().__init__()
        self.nh = num_heads
        self.scale = (dim // num_heads) ** -0.5
        self.wq = nn.Linear(dim, dim, bias=False)
        self.wk = nn.Linear(dim, dim, bias=False)
        self.wv = nn.Linear(dim, dim, bias=False)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, N, C = x.shape
        hd = C // self.nh
        q = self.wq(x[:, :1]).reshape(B, 1, self.nh, hd).permute(0, 2, 1, 3)
        k = self.wk(x).reshape(B, N, self.nh, hd).permute(0, 2, 1, 3)
        v = self.wv(x).reshape(B, N, self.nh, hd).permute(0, 2, 1, 3)
        attn = (q @ k.transpose(-2, -1) * self.scale).softmax(-1)
        return self.proj((attn @ v).transpose(1, 2).reshape(B, 1, C))


class _ViTEncoder(nn.Module):
    """
    Original ViT encoder with Multihead Self-Patch Attention.

    Same I/O interface as SwinEncoder:
        Input  : (B, C, H, W) from CNNEncoder
        Output : (B, emb_dim)
    """

    def __init__(self, in_channels: int, emb_dim: int, spatial_size: int,
                 patch_size: int = 5, depth: int = 2, num_heads: int = 8,
                 mlp_dim: int = 12):
        super().__init__()
        assert spatial_size % patch_size == 0
        num_patches = (spatial_size // patch_size) ** 2

        self.to_patches = Rearrange(
            'b c (h p1) (w p2) -> b (h w) (p1 p2 c)',
            p1=patch_size, p2=patch_size,
        )
        self.cls_token = nn.Parameter(torch.randn(1, 1, emb_dim))
        self.pos_emb = nn.Parameter(torch.randn(1, num_patches + 1, emb_dim))

        # Transformer blocks
        self.blocks = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(depth):
            self.blocks.append(nn.ModuleList([
                nn.LayerNorm(emb_dim),       # pre-norm for cross-attn
                _CrossAttention(emb_dim, num_heads),
                nn.LayerNorm(emb_dim),       # norm for patches
                nn.LayerNorm(emb_dim),       # pre-norm for FFN
                nn.Sequential(               # FFN
                    nn.Linear(emb_dim, mlp_dim), nn.GELU(),
                    nn.Linear(mlp_dim, emb_dim),
                ),
            ]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.to_patches(x)                        # (B, nP, emb_dim)
        B = x.shape[0]
        cls = repeat(self.cls_token, '1 1 d -> b 1 d', b=B)
        x = torch.cat([cls, x], dim=1)                # (B, nP+1, emb_dim)
        x = x + self.pos_emb[:, :x.shape[1]]

        for pre_attn, ca, norm_p, pre_ff, ff in self.blocks:
            cls_new = x[:, :1] + ca(pre_attn(x))      # update cls only
            x = torch.cat([cls_new, norm_p(x[:, 1:])], dim=1)
            x = x + ff(pre_ff(x))                     # FFN + residual

        return x[:, 0]                                 # (B, emb_dim)


# =====================================================================
# Weight utilities
# =====================================================================

class NonZeroClipper:
    """Clamp decoder weights to [eps, 1] (non-negative endmembers)."""
    def __call__(self, module):
        if hasattr(module, 'weight'):
            module.weight.data.clamp_(1e-6, 1.0)


# =====================================================================
# Main AutoEncoder
# =====================================================================

class DeepUnmixer(nn.Module):
    """
    Configurable AutoEncoder for hyperspectral unmixing.

    Parameters
    ----------
    num_endmembers       : P
    num_bands            : L
    spatial_size         : H = W
    encoder_type         : "vit" | "swin"
    decoder_type         : "linear" | "nonlinear"
    use_dual_attention   : whether to insert DualAttentionBlock after CNN
    patch_size           : patch size for ViT (must divide spatial_size)
    emb_dim_per_endmember: per-endmember embedding dim (called 'dim' in paper)
    transformer_depth    : number of transformer blocks
    num_heads            : attention heads
    window_size          : Swin window size
    mlp_dim              : ViT FFN hidden dim
    nonlinear_gamma      : initial gamma for PPNMM decoder
    """

    def __init__(
        self,
        num_endmembers: int,
        num_bands: int,
        spatial_size: int,
        encoder_type: str = "swin",
        decoder_type: str = "linear",
        use_dual_attention: bool = False,
        patch_size: int = 5,
        emb_dim_per_endmember: int = 200,
        transformer_depth: int = 2,
        num_heads: int = 8,
        window_size: int = 5,
        mlp_dim: int = 12,
        nonlinear_gamma: float = 1.0,
    ):
        super().__init__()
        self.P = num_endmembers
        self.L = num_bands
        self.H = self.W = spatial_size
        self.dim = emb_dim_per_endmember

        emb_dim = self.P * self.dim                    # total embedding
        C = emb_dim // (patch_size ** 2)               # CNN output channels

        # --- 1. CNN Encoder (spectral reduction L -> C) ---
        self.cnn_encoder = CNNEncoder(num_bands, C)

        # --- 2. Optional Dual Attention ---
        self.use_dual_attention = use_dual_attention
        if use_dual_attention:
            self.dual_attention = DualAttentionBlock(
                dim=C, spatial_size=spatial_size,
                num_heads=min(num_heads, C),
                spectral_heads=min(4, C),
                window_size=window_size,
            )

        # --- 3. Transformer Encoder ---
        self.encoder_type = encoder_type
        if encoder_type == "vit":
            self.transformer = _ViTEncoder(
                C, emb_dim, spatial_size, patch_size,
                depth=transformer_depth, num_heads=num_heads,
                mlp_dim=mlp_dim,
            )
        elif encoder_type == "swin":
            self.transformer = SwinEncoder(
                in_channels=C, emb_dim=emb_dim, spatial_size=spatial_size,
                depth=transformer_depth, num_heads=min(num_heads, C),
                window_size=window_size,
            )
        else:
            raise ValueError(f"Unknown encoder_type: {encoder_type}")

        # --- 4. Abundance map generation ---
        self.upscale = nn.Linear(self.dim, spatial_size ** 2)
        self.smooth = nn.Sequential(
            nn.Conv2d(self.P, self.P, kernel_size=3, padding=1),
            nn.Softmax(dim=1),
        )

        # --- 5. Decoder ---
        self.decoder_type = decoder_type
        if decoder_type == "linear":
            self.decoder = LinearDecoder(self.P, self.L)
        elif decoder_type == "nonlinear":
            self.decoder = NonLinearDecoder(self.P, self.L,
                                            gamma_init=nonlinear_gamma)
        else:
            raise ValueError(f"Unknown decoder_type: {decoder_type}")

    # ------------------------------------------------------------------
    # Weight initialization
    # ------------------------------------------------------------------

    @staticmethod
    def weights_init(m):
        """Kaiming init for Conv2d layers."""
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight.data)

    def init_decoder_weights(self, Y: torch.Tensor,
                             method: str = "sivm") -> None:
        """
        Initialize decoder endmember weights from data.

        Parameters
        ----------
        Y : (B, L, H, W) or (L, N) hyperspectral image.
        method : "vca", "sivm", or "kmeans_pp".
        """
        if Y.dim() == 4:
            Y = Y.squeeze(0).reshape(self.L, -1)    # (L, H*W)
        E = initialize(Y, self.P, method=method)
        self.decoder.set_endmembers(E)

    def get_clipper(self) -> NonZeroClipper:
        """Return a clipper for decoder weight clamping."""
        return NonZeroClipper()

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Full forward pass.

        Parameters
        ----------
        x : (B, L, H, W) — input hyperspectral cube.

        Returns
        -------
        abundances    : (B, P, H, W) — estimated abundance maps (sum-to-one).
        reconstructed : (B, L, H, W) — reconstructed HSI.
        """
        # 1. CNN feature extraction
        feat = self.cnn_encoder(x)              # (B, C, H, W)

        # 2. Optional dual attention
        if self.use_dual_attention:
            feat = self.dual_attention(feat)     # (B, C, H, W)

        # 3. Transformer encoding
        emb = self.transformer(feat)             # (B, emb_dim)

        # 4. Reshape -> upscale -> smooth -> softmax (abundance maps)
        emb = emb.view(-1, self.P, self.dim)     # (B, P, dim)
        abu = self.upscale(emb)                  # (B, P, H*W)
        abu = abu.view(-1, self.P, self.H, self.W)  # (B, P, H, W)
        abu = self.smooth(abu)                   # softmax along P dim

        # 5. Decode to reconstructed HSI
        recon = self.decoder(abu)                # (B, L, H, W)

        return abu, recon

    def __repr__(self) -> str:
        return (
            f"DeepUnmixer(P={self.P}, L={self.L}, spatial={self.H}x{self.W}, "
            f"encoder={self.encoder_type}, decoder={self.decoder_type}, "
            f"dual_attn={self.use_dual_attention})"
        )


# =====================================================================
# Self-tests
# =====================================================================

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    configs = [
        ("Samson-baseline", dict(
            num_endmembers=3, num_bands=156, spatial_size=95,
            encoder_type="vit", decoder_type="linear",
            use_dual_attention=False,
        )),
        ("Samson-improved", dict(
            num_endmembers=3, num_bands=156, spatial_size=95,
            encoder_type="swin", decoder_type="nonlinear",
            use_dual_attention=True, window_size=5,
        )),
        ("Apex-baseline", dict(
            num_endmembers=4, num_bands=285, spatial_size=110,
            encoder_type="vit", decoder_type="linear",
            use_dual_attention=False,
        )),
        ("Apex-improved", dict(
            num_endmembers=4, num_bands=285, spatial_size=110,
            encoder_type="swin", decoder_type="nonlinear",
            use_dual_attention=True, window_size=5,
        )),
    ]

    for name, cfg in configs:
        print(f"--- {name} ---")
        P, L, H = cfg["num_endmembers"], cfg["num_bands"], cfg["spatial_size"]

        model = DeepUnmixer(**cfg).to(device)
        model.apply(model.weights_init)
        print(f"  {model}")

        # Dummy input
        x = torch.randn(1, L, H, H, device=device)

        # Init decoder weights from data
        model.init_decoder_weights(x, method="sivm")

        # Forward pass
        abu, recon = model(x)

        # --- Asserts ---
        assert recon.shape == x.shape, \
            f"recon shape {recon.shape} != input {x.shape}"
        assert abu.shape == (1, P, H, H), \
            f"abu shape {abu.shape} != (1, {P}, {H}, {H})"

        # Sum-to-one check (softmax guarantees this)
        abu_sum = abu.sum(dim=1)  # (B, H, W)
        assert torch.allclose(abu_sum, torch.ones_like(abu_sum), atol=1e-5), \
            f"Sum-to-one violated! max diff = {(abu_sum - 1).abs().max():.6f}"

        print(f"  Input:      {x.shape}")
        print(f"  Abundances: {abu.shape}  (sum-to-one: OK)")
        print(f"  Recon:      {recon.shape}")
        print()

    print("All unmixer tests PASSED!")
