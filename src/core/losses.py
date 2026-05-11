"""
Loss functions for hyperspectral unmixing training.

Classes
-------
SADLoss   : Spectral Angle Distance loss (scale-invariant).
TotalLoss : Combined beta * MSE + gamma * SAD loss.
"""

from __future__ import annotations

import torch
import torch.nn as nn


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
            # (B, L, H, W) -> (B*H*W, L)
            B, L, H, W = pred.shape
            pred = pred.permute(0, 2, 3, 1).reshape(-1, L)
            target = target.permute(0, 2, 3, 1).reshape(-1, L)

        # Dot product per pixel
        dot = torch.sum(pred * target, dim=-1)                # (N,)
        pred_norm = torch.norm(pred, dim=-1)                  # (N,)
        target_norm = torch.norm(target, dim=-1)              # (N,)

        # Cosine similarity clamped to [-1, 1] for numerical safety
        cos_sim = dot / (pred_norm * target_norm + self.eps)
        cos_sim = torch.clamp(cos_sim, -1.0 + self.eps, 1.0 - self.eps)

        angles = torch.acos(cos_sim)                          # (N,)
        return angles.mean()


class TotalLoss(nn.Module):
    """
    Combined reconstruction loss: beta * MSE + gamma * SAD.

    Parameters
    ----------
    num_bands : L — for SADLoss.
    beta      : weight for MSE (reconstruction error).
    gamma     : weight for SAD (spectral angle).
    eps       : SAD numerical safety.
    """

    def __init__(self, num_bands: int, beta: float = 5e3,
                 gamma: float = 3e-2, eps: float = 1e-8):
        super().__init__()
        self.beta = beta
        self.gamma = gamma
        self.mse = nn.MSELoss(reduction='mean')
        self.sad = SADLoss(num_bands, eps=eps)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        pred   : (B, L, H, W) reconstructed HSI.
        target : (B, L, H, W) original HSI.

        Returns
        -------
        Scalar — total weighted loss.
        """
        loss_mse = self.beta * self.mse(pred, target)
        loss_sad = self.gamma * self.sad(pred, target)
        return loss_mse + loss_sad


# =====================================================================
# Self-tests
# =====================================================================

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    L, H, W = 156, 95, 95
    target = torch.rand(1, L, H, W, device=device) + 0.01
    pred = target + 0.05 * torch.randn_like(target)  # slightly noisy

    # --- SADLoss ---
    sad_fn = SADLoss(L).to(device)
    sad_val = sad_fn(pred, target)
    assert torch.isfinite(sad_val), f"SADLoss is not finite: {sad_val}"
    assert sad_val > 0, "SADLoss should be > 0 for noisy pred"
    print(f"SADLoss:   {sad_val.item():.6f}  (finite: OK, >0: OK)")

    # --- SADLoss with identical inputs (should be ~0) ---
    sad_zero = sad_fn(target, target)
    assert sad_zero.item() < 1e-3, f"SADLoss(x, x) should be ~0, got {sad_zero}"
    print(f"SADLoss(x,x): {sad_zero.item():.8f}  (~0: OK)")

    # --- SADLoss with zero vector (eps protection) ---
    zeros = torch.zeros(1, L, H, W, device=device)
    sad_edge = sad_fn(zeros, target)
    assert torch.isfinite(sad_edge), f"SADLoss with zeros -> NaN/Inf!"
    print(f"SADLoss(0,x): {sad_edge.item():.6f}  (no NaN: OK)")

    # --- TotalLoss ---
    total_fn = TotalLoss(L, beta=5e3, gamma=3e-2).to(device)
    total = total_fn(pred, target)
    assert torch.isfinite(total), f"TotalLoss is not finite: {total}"
    print(f"\nTotalLoss: {total.item():.4f}  (finite: OK)")

    # --- Gradient flow check ---
    pred_g = pred.clone().requires_grad_(True)
    loss = total_fn(pred_g, target)
    loss.backward()
    assert pred_g.grad is not None, "No gradient!"
    assert torch.all(torch.isfinite(pred_g.grad)), "Gradient has NaN/Inf!"
    print(f"Gradient:  shape={pred_g.grad.shape}, all finite: OK")

    print("\nAll loss tests PASSED!")
