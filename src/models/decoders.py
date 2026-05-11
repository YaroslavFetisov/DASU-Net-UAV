"""
Decoder components for DeepUnmixing_v2.

Classes
-------
LinearDecoder    : Standard LMM decoder (Y = E * A).
NonLinearDecoder : Bilinear (PPNMM) decoder with second-order interactions.

Both decoders:
    Input  : (B, P, H, W)  — abundance maps (after softmax).
    Output : (B, L, H, W)  — reconstructed HSI.
    Weights correspond to endmember spectra and can be initialized
    from VCA/SiVM via ``set_endmembers()``.
"""

from __future__ import annotations

from itertools import combinations
from typing import Optional

import torch
import torch.nn as nn


class LinearDecoder(nn.Module):
    """
    Linear Mixing Model decoder:  Y_hat = E @ A   (per pixel).

    Implemented as a 1x1 Conv2d(P -> L, bias=False).
    The weight tensor has shape (L, P, 1, 1) and directly stores
    the endmember matrix E in (out_channels, in_channels, 1, 1) layout.

    Parameters
    ----------
    num_endmembers : P — number of endmembers (input channels).
    num_bands      : L — number of spectral bands (output channels).
    """

    def __init__(self, num_endmembers: int, num_bands: int):
        super().__init__()
        self.P = num_endmembers
        self.L = num_bands

        self.conv = nn.Conv2d(
            num_endmembers, num_bands,
            kernel_size=1, bias=False,
        )
        self.relu = nn.ReLU()

    def set_endmembers(self, E: torch.Tensor) -> None:
        """
        Initialize decoder weights from an endmember matrix.

        Parameters
        ----------
        E : (L, P) or (L, P, 1, 1) endmember spectra.
        """
        if E.dim() == 2:
            E = E.unsqueeze(-1).unsqueeze(-1)  # (L, P, 1, 1)
        with torch.no_grad():
            self.conv.weight.copy_(E)

    def get_endmembers(self) -> torch.Tensor:
        """Return current endmember matrix as (L, P)."""
        return self.conv.weight.data.squeeze(-1).squeeze(-1)

    def forward(self, abundances: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        abundances : (B, P, H, W)

        Returns
        -------
        reconstructed : (B, L, H, W)
        """
        return self.relu(self.conv(abundances))


class NonLinearDecoder(nn.Module):
    """
    Post-nonlinear Polynomial Mixing Model (PPNMM) decoder.

    Extends the LMM with bilinear interaction terms:

        Y_hat = E @ A  +  gamma * SUM_{i<j} (a_i * a_j) * (e_i o e_j)

    where:
        - ``a_i, a_j`` are abundance maps for endmembers i, j
        - ``e_i o e_j`` is the element-wise product of endmember spectra
        - ``gamma`` controls the strength of non-linear mixing
          (learnable, initialized from config)

    This models secondary reflections between materials (e.g., light
    bouncing between tree canopy and soil).

    Parameters
    ----------
    num_endmembers : P — number of endmembers.
    num_bands      : L — number of spectral bands.
    gamma_init     : Initial value for the non-linear mixing coefficient.
    """

    def __init__(self, num_endmembers: int, num_bands: int,
                 gamma_init: float = 1.0):
        super().__init__()
        self.P = num_endmembers
        self.L = num_bands
        self.num_pairs = num_endmembers * (num_endmembers - 1) // 2

        # Linear part: same 1x1 conv as LinearDecoder
        self.linear_conv = nn.Conv2d(
            num_endmembers, num_bands,
            kernel_size=1, bias=False,
        )

        # Bilinear interaction weight (learnable scalar)
        self.gamma = nn.Parameter(torch.tensor(gamma_init))

        # Bilinear endmember interactions: (L, num_pairs, 1, 1)
        # Registered as buffer (derived from weights, updated manually)
        self.register_buffer(
            "_pair_indices",
            torch.tensor(list(combinations(range(num_endmembers), 2)),
                         dtype=torch.long),
        )

        self.relu = nn.ReLU()

    def set_endmembers(self, E: torch.Tensor) -> None:
        """Initialize linear decoder weights from endmember matrix (L, P)."""
        if E.dim() == 2:
            E = E.unsqueeze(-1).unsqueeze(-1)
        with torch.no_grad():
            self.linear_conv.weight.copy_(E)

    def get_endmembers(self) -> torch.Tensor:
        """Return current endmember matrix as (L, P)."""
        return self.linear_conv.weight.data.squeeze(-1).squeeze(-1)

    def _compute_bilinear(self, abundances: torch.Tensor) -> torch.Tensor:
        """
        Compute the bilinear interaction term.

        Parameters
        ----------
        abundances : (B, P, H, W)

        Returns
        -------
        bilinear_term : (B, L, H, W)
        """
        E = self.get_endmembers()  # (L, P)
        idx = self._pair_indices   # (num_pairs, 2)

        # Bilinear abundance maps: a_i * a_j for each pair
        a_i = abundances[:, idx[:, 0]]  # (B, num_pairs, H, W)
        a_j = abundances[:, idx[:, 1]]  # (B, num_pairs, H, W)
        ab_bilinear = a_i * a_j         # (B, num_pairs, H, W)

        # Bilinear endmembers: e_i o e_j (element-wise product)
        e_i = E[:, idx[:, 0]]  # (L, num_pairs)
        e_j = E[:, idx[:, 1]]  # (L, num_pairs)
        E_bilinear = e_i * e_j  # (L, num_pairs)

        # Mix: (L, num_pairs) @ (num_pairs, B*H*W) -> (L, B*H*W) -> reshape
        B, _, H, W = abundances.shape
        ab_flat = ab_bilinear.reshape(B, self.num_pairs, -1)  # (B, np, H*W)

        # Batch matmul: (B, L, np) @ ... but E_bilinear is shared across batch
        # Efficient: einsum
        bilinear = torch.einsum("lp,bpn->bln", E_bilinear, ab_flat)
        return bilinear.reshape(B, self.L, H, W)

    def forward(self, abundances: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        abundances : (B, P, H, W)

        Returns
        -------
        reconstructed : (B, L, H, W)
        """
        # Linear part
        y_linear = self.linear_conv(abundances)

        # Bilinear part (PPNMM)
        y_bilinear = self._compute_bilinear(abundances)

        return self.relu(y_linear + self.gamma * y_bilinear)


# =====================================================================
# Self-tests
# =====================================================================

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    for name, (L, P, H) in [
        ("Samson", (156, 3, 95)),
        ("Apex",   (285, 4, 110)),
    ]:
        print(f"\n--- {name} ---")

        # Dummy abundance maps (after softmax, so sum-to-one)
        raw = torch.randn(1, P, H, H, device=device)
        abundances = torch.softmax(raw, dim=1)

        # Dummy endmembers
        E = torch.rand(L, P, device=device)

        # --- LinearDecoder ---
        ld = LinearDecoder(P, L).to(device)
        ld.set_endmembers(E)
        y_lin = ld(abundances)
        assert y_lin.shape == (1, L, H, H), f"LinearDecoder: {y_lin.shape}"
        E_back = ld.get_endmembers()
        assert E_back.shape == (L, P), f"get_endmembers: {E_back.shape}"
        assert torch.allclose(E_back, E), "Endmember round-trip failed"
        print(f"  LinearDecoder     {abundances.shape} -> {y_lin.shape}")

        # --- NonLinearDecoder ---
        nld = NonLinearDecoder(P, L, gamma_init=0.5).to(device)
        nld.set_endmembers(E)
        y_nlin = nld(abundances)
        assert y_nlin.shape == (1, L, H, H), f"NonLinearDecoder: {y_nlin.shape}"
        pairs = P * (P - 1) // 2
        print(f"  NonLinearDecoder  {abundances.shape} -> {y_nlin.shape} "
              f"({pairs} bilinear pairs, gamma={nld.gamma.item():.2f})")

        # Check non-linearity actually contributes
        diff = (y_nlin - y_lin).abs().mean().item()
        print(f"  Linear vs NonLinear diff: {diff:.6f}")
        assert diff > 0, "NonLinear decoder output identical to Linear!"

    print("\nAll decoder shape checks PASSED!")
