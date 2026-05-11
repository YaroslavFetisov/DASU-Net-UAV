"""
Endmember initialization algorithms for decoder weight initialization.

Functions
---------
vca         : Vertex Component Analysis (classical geometric method).
sivm        : Simplex Volume Maximization (greedy volume maximization).
kmeans_pp   : K-means++ based initialization.
initialize  : Dispatcher that picks the right algorithm by name.
"""

from __future__ import annotations

import torch
import numpy as np
from typing import Optional


def vca(Y: torch.Tensor, P: int, seed: int = 42) -> torch.Tensor:
    """
    Vertex Component Analysis (VCA).

    Projects data onto random directions and iteratively selects the
    most extreme pixels as endmembers.

    Parameters
    ----------
    Y : (L, N) hyperspectral data  (L bands, N pixels).
    P : Number of endmembers to extract.
    seed : Random seed.

    Returns
    -------
    E : (L, P) estimated endmember spectra.
    """
    L, N = Y.shape
    device = Y.device

    rng = torch.Generator(device=device)
    rng.manual_seed(seed)

    # 1) Dimensionality reduction via SVD (project to P-1 dims)
    Y_mean = Y.mean(dim=1, keepdim=True)
    Y_centered = Y - Y_mean
    U, S, _ = torch.linalg.svd(Y_centered, full_matrices=False)
    U_p = U[:, :P]  # (L, P)
    Y_proj = U_p.T @ Y_centered  # (P, N)

    # 2) Iteratively find endmembers
    indices = []
    for _ in range(P):
        # Random direction
        w = torch.randn(P, 1, device=device, generator=rng)
        # Project out the space of already-found endmembers
        if indices:
            E_found = Y_proj[:, indices]  # (P, k)
            # Orthogonal projector onto complement of E_found
            Q, _ = torch.linalg.qr(E_found)
            w = w - Q @ (Q.T @ w)

        w = w / (torch.norm(w) + 1e-10)
        # Project all pixels onto w and find the extreme
        projections = (w.T @ Y_proj).squeeze(0)  # (N,)
        idx = torch.argmax(torch.abs(projections)).item()
        indices.append(idx)

    return Y[:, indices]  # (L, P)


def sivm(Y: torch.Tensor, P: int) -> torch.Tensor:
    """
    Simplex Volume Maximization (SiVM).

    Greedy algorithm: iteratively selects the pixel that maximizes
    the volume of the simplex formed by already-selected endmembers.
    Uses successive orthogonal projection for numerical stability.

    Parameters
    ----------
    Y : (L, N) hyperspectral data.
    P : Number of endmembers.

    Returns
    -------
    E : (L, P) estimated endmember spectra.
    """
    L, N = Y.shape

    # First endmember: pixel with maximum L2 norm
    norms = torch.norm(Y, dim=0)  # (N,)
    idx = torch.argmax(norms).item()
    indices = [idx]
    selected = Y[:, idx:idx + 1]  # (L, 1)

    for _ in range(1, P):
        # Compute orthogonal projection residuals
        Q, _ = torch.linalg.qr(selected)  # (L, k)
        proj = Q @ (Q.T @ Y)              # (L, N)
        residuals = Y - proj              # (L, N)
        distances = torch.norm(residuals, dim=0)  # (N,)

        # Exclude already-selected indices
        for i in indices:
            distances[i] = 0.0

        idx = torch.argmax(distances).item()
        indices.append(idx)
        selected = Y[:, indices]  # (L, k+1)

    return Y[:, indices]  # (L, P)


def kmeans_pp(Y: torch.Tensor, P: int, seed: int = 42) -> torch.Tensor:
    """
    K-means++ initialization for endmember estimation.

    Selects P diverse pixels using the D^2 weighting scheme from
    K-means++ (Arthur & Vassilvitskii, 2007).

    Parameters
    ----------
    Y : (L, N) hyperspectral data.
    P : Number of endmembers (centers).
    seed : Random seed.

    Returns
    -------
    E : (L, P) estimated endmember spectra.
    """
    L, N = Y.shape
    device = Y.device

    rng = torch.Generator(device=device)
    rng.manual_seed(seed)

    # First center: random pixel
    idx = torch.randint(N, (1,), generator=rng, device=device).item()
    indices = [idx]

    for _ in range(1, P):
        # Compute min squared distance to nearest selected center
        centers = Y[:, indices]  # (L, k)
        # (L, N) vs (L, k) -> (k, N) pairwise sq distances
        dists = torch.cdist(centers.T, Y.T, p=2.0)  # (k, N)
        min_dists, _ = dists.min(dim=0)  # (N,)
        sq_dists = min_dists ** 2

        # Zero out already-selected
        for i in indices:
            sq_dists[i] = 0.0

        # Sample proportional to D^2
        probs = sq_dists / (sq_dists.sum() + 1e-10)
        idx = torch.multinomial(probs, 1, generator=rng).item()
        indices.append(idx)

    return Y[:, indices]  # (L, P)


def initialize(
    Y: torch.Tensor,
    P: int,
    method: str = "vca",
    seed: int = 42,
) -> torch.Tensor:
    """
    Dispatcher: initialize endmembers using the specified method.

    Parameters
    ----------
    Y      : (L, N) hyperspectral data.
    P      : Number of endmembers.
    method : "vca", "sivm", or "kmeans_pp".
    seed   : Random seed (where applicable).

    Returns
    -------
    E : (L, P) estimated endmember spectra.
    """
    method = method.lower()
    if method == "vca":
        return vca(Y, P, seed=seed)
    elif method == "sivm":
        return sivm(Y, P)
    elif method in ("kmeans_pp", "kmeans++", "kmeanspp"):
        return kmeans_pp(Y, P, seed=seed)
    else:
        raise ValueError(
            f"Unknown initialization method '{method}'. "
            f"Available: vca, sivm, kmeans_pp"
        )


# =====================================================================
# Self-tests
# =====================================================================

if __name__ == "__main__":
    print("Testing initialization algorithms...")

    L, N, P = 156, 9025, 3  # Samson-like
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Y = torch.rand(L, N, device=device) + 0.01  # positive reflectance

    for name in ["vca", "sivm", "kmeans_pp"]:
        E = initialize(Y, P, method=name)
        assert E.shape == (L, P), f"{name}: expected ({L},{P}), got {E.shape}"
        assert torch.all(torch.isfinite(E)), f"{name}: contains non-finite values"
        print(f"  {name:12s} -> E shape {E.shape}, "
              f"norms: {torch.norm(E, dim=0).tolist()}")

    print("\nAll initialization tests PASSED!")
