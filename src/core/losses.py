"""
Loss functions for hyperspectral unmixing training.

Classes
-------
SADLoss      : Spectral Angle Distance loss (scale-invariant).
MinVolLoss   : Minimum Volume regularization on endmember simplex.
SparsityLoss : L1 sparsity regularization on abundance maps.
TotalLoss    : Combined beta*MSE + gamma*SAD + delta*MinVol + lambda_reg*Sparsity.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SADLoss(nn.Module):
    """
    Spectral Angle Distance (SAD) loss.

    Measures the angle between predicted and target spectral vectors.
    Scale-invariant — focuses on spectral shape, not magnitude.

    Parameters
    ----------
    num_bands : L — number of spectral bands.
    eps       : small constant to prevent division by zero.
    """

    def __init__(self, num_bands: int, eps: float = 1e-8):
        super().__init__()
        self.L = num_bands
        self.eps = eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        pred   : (B, L, H, W) or (N, L) reconstructed spectra.
        target : same shape as pred — original spectra.

        Returns
        -------
        Scalar — mean SAD over all pixels.
        """
        if pred.dim() == 4:
            B, L, H, W = pred.shape
            pred = pred.permute(0, 2, 3, 1).reshape(-1, L)
            target = target.permute(0, 2, 3, 1).reshape(-1, L)

        dot = torch.sum(pred * target, dim=-1)
        pred_norm = torch.norm(pred, dim=-1)
        target_norm = torch.norm(target, dim=-1)

        cos_sim = dot / (pred_norm * target_norm + self.eps)
        cos_sim = torch.clamp(cos_sim, -1.0 + self.eps, 1.0 - self.eps)
        return torch.acos(cos_sim).mean()


class MinVolLoss(nn.Module):
    """
    Minimum Volume Regularization (MinVol).

    Penalizes the volume of the simplex formed by endmember spectra.
    A smaller simplex volume means endmembers are closer together and
    less "spread out" in spectral space — this prevents degenerate
    solutions where all endmembers collapse to the same spectrum, while
    simultaneously encouraging the simplest (most compact) explanation
    of the data.

    Implementation: log|det(E^T E + eps*I)| via Cholesky decomposition,
    which is numerically stable and fully differentiable.

    Parameters
    ----------
    eps : regularization added to the diagonal before log-det for
          numerical stability (prevents log(0) if E has near-zero
          singular values).

    Reference
    ---------
    Lin et al. "Minimum-Volume Constrained Nonnegative Matrix
    Factorization" (2007); Craig (1994) minimum-volume simplex.
    """

    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, E: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        E : (L, P) endmember spectra matrix from decoder.get_endmembers().

        Returns
        -------
        Scalar — log-volume of the endmember simplex.
        """
        # Gram matrix G = E^T E  (P x P)
        G = E.T @ E

        # Add eps * I for numerical stability before Cholesky
        P = G.shape[0]
        G = G + self.eps * torch.eye(P, device=G.device, dtype=G.dtype)

        # log|det(G)| = 2 * sum(log(diag(L)))  via Cholesky G = L L^T
        try:
            L_chol = torch.linalg.cholesky(G)
            log_vol = 2.0 * torch.log(torch.diagonal(L_chol)).sum()
        except RuntimeError:
            # Fallback to slogdet if Cholesky fails (numerically rare)
            sign, log_vol = torch.linalg.slogdet(G)
            # If det <= 0 (degenerate), return 0 without gradient
            if sign.item() <= 0:
                return torch.tensor(0.0, device=E.device, requires_grad=False)

        return log_vol


class SparsityLoss(nn.Module):
    """
    Abundance Sparsity Regularization (L1).

    Encourages each pixel to be explained by as few endmembers as
    possible (ideally 1-2 out of P).  Applied directly on the
    abundance map A (after Softmax, so values are in [0, 1]).

    Since sum-to-one is already enforced, this is equivalent to
    minimising the entropy of A, i.e., pushing distributions toward
    one-hot corners of the simplex.

    Parameters
    ----------
    reduction : "mean" or "sum" over pixels.
    """

    def __init__(self, reduction: str = "mean"):
        super().__init__()
        self.reduction = reduction

    def forward(self, A: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        A : (B, P, H, W) abundance maps (after softmax, values in [0,1]).

        Returns
        -------
        Scalar — mean L1 norm of each pixel's abundance vector.
        """
        # Per-pixel L1 norm along P dimension = sum of abs values
        # Since A >= 0 (softmax), |A| == A, so this is sum over P
        # Mean over all pixels → scalar
        l1_per_pixel = A.sum(dim=1)       # (B, H, W)  == 1.0 always (softmax)
        # Instead: use a diversity-penalising term:
        # kl(A || uniform) = sum A*log(A*P), where P cancels in log
        # But simplest stable form: -sum(A * log(A + eps))  (entropy)
        # Higher entropy → less sparse.  We minimise this = maximise sparsity.
        eps = 1e-8
        entropy = -(A * torch.log(A + eps)).sum(dim=1)  # (B, H, W)

        if self.reduction == "mean":
            return entropy.mean()
        return entropy.sum()


class TotalLoss(nn.Module):
    """
    Combined loss:  beta*MSE + gamma*SAD + delta*MinVol + lambda_reg*Sparsity.

    Parameters
    ----------
    num_bands   : L — for SADLoss.
    beta        : weight for MSE (reconstruction fidelity).
    gamma       : weight for SAD (spectral angle fidelity).
    delta       : weight for MinVol (endmember compactness).
    lambda_reg  : weight for Sparsity (encourage 1-hot abundances).
    eps_sad     : epsilon for SADLoss numerical safety.
    eps_vol     : epsilon for MinVolLoss Cholesky stability.
    """

    def __init__(
        self,
        num_bands: int,
        beta: float = 5e3,
        gamma: float = 3e-2,
        delta: float = 1e-2,
        lambda_reg: float = 1e-2,
        eps_sad: float = 1e-8,
        eps_vol: float = 1e-6,
    ):
        super().__init__()
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
        self.lambda_reg = lambda_reg

        self.mse = nn.MSELoss(reduction='mean')
        self.sad = SADLoss(num_bands, eps=eps_sad)
        self.minvol = MinVolLoss(eps=eps_vol)
        self.sparsity = SparsityLoss(reduction='mean')

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        abundances: torch.Tensor,
        endmembers: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        pred        : (B, L, H, W) reconstructed HSI.
        target      : (B, L, H, W) original HSI.
        abundances  : (B, P, H, W) abundance maps (after softmax).
        endmembers  : (L, P) endmember spectra from decoder.get_endmembers().

        Returns
        -------
        Scalar — total weighted loss.
        """
        loss_mse = self.beta * self.mse(pred, target)
        loss_sad = self.gamma * self.sad(pred, target)
        loss_vol = self.delta * self.minvol(endmembers)
        loss_spar = self.lambda_reg * self.sparsity(abundances)
        return loss_mse + loss_sad + loss_vol + loss_spar

    def breakdown(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        abundances: torch.Tensor,
        endmembers: torch.Tensor,
    ) -> dict:
        """Return individual loss terms (for logging)."""
        with torch.no_grad():
            return {
                "mse":  float(self.beta * self.mse(pred, target).item()),
                "sad":  float(self.gamma * self.sad(pred, target).item()),
                "vol":  float(self.delta * self.minvol(endmembers).item()),
                "spar": float(self.lambda_reg * self.sparsity(abundances).item()),
            }


# =====================================================================
# Self-tests
# =====================================================================

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    L, P, H, W = 156, 3, 95, 95
    target = torch.rand(1, L, H, W, device=device) + 0.01
    pred   = target + 0.05 * torch.randn_like(target)
    E      = torch.rand(L, P, device=device).clamp(1e-6, 1.0)
    # Softmax abundance maps (sum-to-one along P)
    A_raw  = torch.randn(1, P, H, W, device=device)
    A      = torch.softmax(A_raw, dim=1)

    # ---- SADLoss ------------------------------------------------
    sad_fn = SADLoss(L).to(device)
    sad_val = sad_fn(pred, target)
    assert torch.isfinite(sad_val) and sad_val > 0
    print(f"SADLoss:          {sad_val.item():.6f}  (finite, >0: OK)")

    sad_self = sad_fn(target, target)
    assert sad_self.item() < 1e-3
    print(f"SADLoss(x,x):     {sad_self.item():.8f}  (~0: OK)")

    zeros = torch.zeros_like(pred)
    assert torch.isfinite(sad_fn(zeros, target))
    print(f"SADLoss(0,x):     finite  (eps-guard: OK)")

    # ---- MinVolLoss ---------------------------------------------
    vol_fn = MinVolLoss().to(device)
    vol_val = vol_fn(E)
    assert torch.isfinite(vol_val), f"MinVolLoss NaN/Inf: {vol_val}"
    print(f"\nMinVolLoss:       {vol_val.item():.6f}  (finite: OK)")

    # Verify smaller simplex → smaller loss
    E_small = E * 0.1
    vol_small = vol_fn(E_small)
    assert vol_small < vol_val, "Smaller simplex should have smaller log-vol"
    print(f"MinVolLoss(0.1E): {vol_small.item():.6f}  (< full: OK)")

    # Gradient through MinVolLoss
    E_g = E.clone().requires_grad_(True)
    vol_fn(E_g).backward()
    assert E_g.grad is not None and torch.all(torch.isfinite(E_g.grad))
    print(f"MinVol gradient:  all finite  (OK)")

    # ---- SparsityLoss -------------------------------------------
    spar_fn = SparsityLoss().to(device)

    # Uniform A → high entropy → high sparsity loss
    A_uniform = torch.full((1, P, H, W), 1.0 / P, device=device)
    spar_uniform = spar_fn(A_uniform)

    # One-hot-ish A → low entropy → low sparsity loss
    A_onehot = torch.zeros(1, P, H, W, device=device)
    A_onehot[:, 0, :, :] = 1.0
    spar_onehot = spar_fn(A_onehot)

    assert torch.isfinite(spar_uniform) and torch.isfinite(spar_onehot)
    assert spar_onehot < spar_uniform, \
        f"One-hot should be sparser: {spar_onehot:.4f} vs {spar_uniform:.4f}"
    print(f"\nSparsityLoss(uniform):  {spar_uniform.item():.6f}")
    print(f"SparsityLoss(one-hot):  {spar_onehot.item():.6f}  (<uniform: OK)")

    # Gradient through SparsityLoss
    A_g = A.clone().requires_grad_(True)
    spar_fn(A_g).backward()
    assert A_g.grad is not None and torch.all(torch.isfinite(A_g.grad))
    print(f"Sparsity gradient:      all finite  (OK)")

    # ---- TotalLoss ----------------------------------------------
    total_fn = TotalLoss(
        num_bands=L, beta=5e3, gamma=3e-2, delta=1e-2, lambda_reg=1e-2
    ).to(device)

    total = total_fn(pred, target, A, E)
    assert torch.isfinite(total), f"TotalLoss NaN/Inf: {total}"
    print(f"\nTotalLoss:        {total.item():.4f}  (finite: OK)")

    bd = total_fn.breakdown(pred, target, A, E)
    print(f"  Breakdown: MSE={bd['mse']:.4f}  SAD={bd['sad']:.4f}  "
          f"Vol={bd['vol']:.4f}  Spar={bd['spar']:.4f}")

    # Full gradient flow through TotalLoss
    pred_g  = pred.clone().requires_grad_(True)
    E_g2    = E.clone().requires_grad_(True)
    A_g2    = A.clone().requires_grad_(True)
    total_fn(pred_g, target, A_g2, E_g2).backward()

    for name, t in [("pred", pred_g), ("E", E_g2), ("A", A_g2)]:
        assert t.grad is not None and torch.all(torch.isfinite(t.grad)), \
            f"Gradient of {name} has NaN/Inf!"
        print(f"  Grad {name}: shape={tuple(t.grad.shape)}, all finite: OK")

    print("\nAll loss tests PASSED!")
