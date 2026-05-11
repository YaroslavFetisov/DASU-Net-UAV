"""
Evaluation metrics for hyperspectral unmixing.

Functions
---------
compute_sad         : SAD between estimated and GT endmembers.
compute_rmse        : RMSE between estimated and GT abundance maps.
match_endmembers    : Hungarian algorithm to align predicted endmembers to GT.
"""

from __future__ import annotations
from typing import Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment


def compute_sad(E_pred: np.ndarray, E_gt: np.ndarray,
                eps: float = 1e-10) -> Tuple[list, float]:
    """
    Spectral Angle Distance between predicted and GT endmembers.

    Parameters
    ----------
    E_pred : (L, P) predicted endmember spectra.
    E_gt   : (L, P) ground-truth endmember spectra.
    eps    : numerical safety for norm computation.

    Returns
    -------
    class_sad : list of P per-endmember SAD values (radians).
    mean_sad  : mean SAD across all endmembers.
    """
    P = E_pred.shape[1]
    class_sad = []
    for i in range(P):
        p_norm = np.linalg.norm(E_pred[:, i]) + eps
        g_norm = np.linalg.norm(E_gt[:, i]) + eps
        cos_sim = np.dot(E_pred[:, i], E_gt[:, i]) / (p_norm * g_norm)
        cos_sim = np.clip(cos_sim, -1.0, 1.0)
        class_sad.append(float(np.arccos(cos_sim)))
    return class_sad, float(np.mean(class_sad))


def compute_rmse(A_pred: np.ndarray, A_gt: np.ndarray
                 ) -> Tuple[list, float]:
    """
    Root Mean Squared Error between predicted and GT abundance maps.

    Parameters
    ----------
    A_pred : (H, W, P) predicted abundance maps.
    A_gt   : (H, W, P) ground-truth abundance maps.

    Returns
    -------
    class_rmse : list of P per-endmember RMSE values.
    mean_rmse  : overall RMSE across all endmembers and pixels.
    """
    H, W, P = A_gt.shape
    class_rmse = []
    for i in range(P):
        rmse_i = np.sqrt(np.mean((A_pred[:, :, i] - A_gt[:, :, i]) ** 2))
        class_rmse.append(float(rmse_i))
    mean_rmse = np.sqrt(np.mean((A_pred - A_gt) ** 2))
    return class_rmse, float(mean_rmse)


def _sad_matrix(E_pred: np.ndarray, E_gt: np.ndarray,
                eps: float = 1e-10) -> np.ndarray:
    """
    Compute P_pred x P_gt SAD cost matrix.

    Parameters
    ----------
    E_pred : (L, P_pred)
    E_gt   : (L, P_gt)

    Returns
    -------
    cost : (P_pred, P_gt)  SAD values.
    """
    P_pred = E_pred.shape[1]
    P_gt = E_gt.shape[1]
    cost = np.zeros((P_pred, P_gt))
    for i in range(P_pred):
        for j in range(P_gt):
            p_norm = np.linalg.norm(E_pred[:, i]) + eps
            g_norm = np.linalg.norm(E_gt[:, j]) + eps
            cos_sim = np.dot(E_pred[:, i], E_gt[:, j]) / (p_norm * g_norm)
            cos_sim = np.clip(cos_sim, -1.0, 1.0)
            cost[i, j] = np.arccos(cos_sim)
    return cost


def match_endmembers(
    E_pred: np.ndarray,
    E_gt: np.ndarray,
    A_pred: np.ndarray,
    A_gt: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Align predicted endmembers/abundances to GT using the Hungarian algorithm.

    The endmember ordering from the model is arbitrary.  This function
    finds the optimal permutation that minimizes the total SAD between
    predicted and GT endmembers, then reorders E_pred and A_pred.

    Parameters
    ----------
    E_pred : (L, P) predicted endmember spectra.
    E_gt   : (L, P) ground-truth endmember spectra.
    A_pred : (H, W, P) predicted abundance maps.
    A_gt   : (H, W, P) ground-truth abundance maps (unused in matching,
             included for API consistency).

    Returns
    -------
    E_matched : (L, P) reordered predicted endmembers.
    A_matched : (H, W, P) reordered predicted abundance maps.
    perm      : (P,) permutation indices applied.
    """
    cost = _sad_matrix(E_pred, E_gt)
    row_ind, col_ind = linear_sum_assignment(cost)
    # col_ind[i] = which GT endmember matches predicted endmember i
    # We need to reorder predicted to match GT order
    # perm[j] = which predicted endmember to put at position j
    perm = np.zeros(E_gt.shape[1], dtype=int)
    for r, c in zip(row_ind, col_ind):
        perm[c] = r

    E_matched = E_pred[:, perm]
    A_matched = A_pred[:, :, perm]
    return E_matched, A_matched, perm


# =====================================================================
# Self-tests
# =====================================================================

if __name__ == "__main__":
    np.random.seed(42)
    print("Testing metrics...\n")

    L, P, H, W = 156, 3, 95, 95

    # Create GT endmembers and abundances
    E_gt = np.random.rand(L, P).astype(np.float64)
    A_gt = np.random.dirichlet(np.ones(P), size=(H, W)).astype(np.float64)

    # --- Test 1: match_endmembers with known permutation ---
    known_perm = np.array([2, 0, 1])  # shuffle order
    E_shuffled = E_gt[:, known_perm]
    A_shuffled = A_gt[:, :, known_perm]

    E_matched, A_matched, found_perm = match_endmembers(
        E_shuffled, E_gt, A_shuffled, A_gt
    )

    # After matching, endmembers should be back to original order
    assert np.allclose(E_matched, E_gt, atol=1e-10), \
        "match_endmembers failed to recover original order!"
    assert np.allclose(A_matched, A_gt, atol=1e-10), \
        "Abundance reordering failed!"
    print(f"  Known perm:  {known_perm}")
    print(f"  Found perm:  {found_perm}")
    print(f"  Endmembers matched: OK")
    print(f"  Abundances matched: OK")

    # --- Test 2: compute_sad with matched endmembers ---
    sad_cls, sad_mean = compute_sad(E_matched, E_gt)
    assert all(s < 1e-4 for s in sad_cls), f"SAD should be ~0: {sad_cls}"
    print(f"\n  SAD (matched):  {sad_cls} -> mean={sad_mean:.10f}  (~0: OK)")

    # --- Test 3: compute_sad with noisy endmembers ---
    E_noisy = E_gt + 0.1 * np.random.randn(L, P)
    sad_cls_n, sad_mean_n = compute_sad(E_noisy, E_gt)
    assert all(s > 0 for s in sad_cls_n), "SAD should be > 0 for noisy"
    print(f"  SAD (noisy):    mean={sad_mean_n:.6f}  (>0: OK)")

    # --- Test 4: compute_rmse ---
    rmse_cls, rmse_mean = compute_rmse(A_gt, A_gt)
    assert rmse_mean < 1e-10, f"RMSE(x,x) should be 0, got {rmse_mean}"
    print(f"\n  RMSE (self):    {rmse_mean:.10f}  (~0: OK)")

    A_noisy = A_gt + 0.05 * np.random.randn(H, W, P)
    rmse_cls_n, rmse_mean_n = compute_rmse(A_noisy, A_gt)
    assert rmse_mean_n > 0, "RMSE should be > 0 for noisy"
    print(f"  RMSE (noisy):   {rmse_mean_n:.6f}  (>0: OK)")

    # --- Test 5: match with 4 endmembers (Apex-like) ---
    P4 = 4
    E_gt4 = np.random.rand(285, P4)
    perm4 = np.array([3, 1, 0, 2])
    E_shuf4 = E_gt4[:, perm4]
    A_gt4 = np.random.dirichlet(np.ones(P4), size=(110, 110))
    A_shuf4 = A_gt4[:, :, perm4]

    E_m4, A_m4, fp4 = match_endmembers(E_shuf4, E_gt4, A_shuf4, A_gt4)
    assert np.allclose(E_m4, E_gt4, atol=1e-10), "4-endmember matching failed!"
    print(f"\n  4-endmember match: perm={perm4} -> found={fp4}  OK")

    print("\nAll metrics tests PASSED!")
