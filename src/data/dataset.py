"""
HyperspectralDataset — PyTorch Dataset for hyperspectral .mat files.

Supported datasets: Samson, Apex.
Each .mat file contains:
    Y  — (L, N) hyperspectral image (L spectral bands, N = H*W pixels)
    A  — (P, N) abundance maps (P — number of endmembers)
    M  — (L, P) ground-truth endmembers
    M1 — (L, P) initial weights (VCA-initialization)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import scipy.io as sio
import torch
from torch.utils.data import Dataset


# ---------------------------------------------------------------------------
# Registry of dataset metadata
# ---------------------------------------------------------------------------
@dataclass
class DatasetMeta:
    """Metadata for a known hyperspectral unmixing dataset."""
    num_endmembers: int   # P — number of endmembers
    num_bands: int         # L — number of spectral bands
    spatial_size: int      # col — spatial side length (image is col×col)


DATASET_REGISTRY: dict[str, DatasetMeta] = {
    "samson": DatasetMeta(num_endmembers=3, num_bands=156, spatial_size=95),
    "apex":   DatasetMeta(num_endmembers=4, num_bands=285, spatial_size=110),
}


# ---------------------------------------------------------------------------
# PyTorch Dataset
# ---------------------------------------------------------------------------
class HyperspectralDataset(Dataset):
    """
    PyTorch Dataset for hyperspectral unmixing.

    Loads a single .mat file and provides pixel-level access:
        - __getitem__(i) → (spectrum_i, abundance_i)
        - spectrum_i  has shape (L,)   — reflectance vector
        - abundance_i has shape (P,)   — ground-truth fractional abundances

    Additionally exposes the full image/cube tensors for model training,
    since the original DeepTrans-HSU feeds the entire image as one batch.

    Parameters
    ----------
    dataset_name : str
        Name of the dataset ("samson" or "apex").
    data_dir : str | Path
        Directory containing ``{dataset_name}_dataset.mat``.
    device : torch.device | str, optional
        Device to place tensors on (default: ``"cpu"``).
    normalize : bool, optional
        If True, normalize spectra to [0, 1] per-band using min-max scaling.
    """

    def __init__(
        self,
        dataset_name: str,
        data_dir: str | Path = "./data/raw",
        device: torch.device | str = "cpu",
        normalize: bool = False,
    ) -> None:
        super().__init__()

        dataset_name = dataset_name.lower()
        if dataset_name not in DATASET_REGISTRY:
            raise ValueError(
                f"Unknown dataset '{dataset_name}'. "
                f"Available: {list(DATASET_REGISTRY.keys())}"
            )

        self.meta = DATASET_REGISTRY[dataset_name]
        self.dataset_name = dataset_name
        self.device = torch.device(device)

        # --- Load .mat file -----------------------------------------------
        mat_path = Path(data_dir) / f"{dataset_name}_dataset.mat"
        if not mat_path.exists():
            raise FileNotFoundError(
                f"Dataset file not found: {mat_path}\n"
                f"Download it and place into '{data_dir}/'."
            )

        data = sio.loadmat(str(mat_path))

        # Y: (L, N) → transpose to (N, L) for pixel-level indexing
        self.Y: torch.Tensor = (
            torch.from_numpy(data["Y"].astype(np.float32)).T.to(self.device)
        )

        # A: (P, N) → transpose to (N, P)
        self.A: torch.Tensor = (
            torch.from_numpy(data["A"].astype(np.float32)).T.to(self.device)
        )

        # M: (L, P) — ground-truth endmembers  (kept on CPU for evaluation)
        self.M: torch.Tensor = torch.from_numpy(
            data["M"].astype(np.float32)
        )

        # M1: (L, P) — VCA-initialized endmembers (decoder init weights)
        self.M1: torch.Tensor = torch.from_numpy(
            data["M1"].astype(np.float32)
        )

        # --- Optional normalization ---------------------------------------
        if normalize:
            y_min = self.Y.min(dim=0, keepdim=True).values
            y_max = self.Y.max(dim=0, keepdim=True).values
            self.Y = (self.Y - y_min) / (y_max - y_min + 1e-8)

        # --- Sanity checks ------------------------------------------------
        N = self.meta.spatial_size ** 2
        assert self.Y.shape == (N, self.meta.num_bands), (
            f"Expected Y shape ({N}, {self.meta.num_bands}), got {self.Y.shape}"
        )
        assert self.A.shape == (N, self.meta.num_endmembers), (
            f"Expected A shape ({N}, {self.meta.num_endmembers}), got {self.A.shape}"
        )

    # --- Dataset interface ------------------------------------------------

    def __len__(self) -> int:
        """Number of pixels."""
        return self.Y.shape[0]

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns
        -------
        spectrum : Tensor of shape (L,)
        abundance : Tensor of shape (P,)
        """
        return self.Y[index], self.A[index]

    # --- Convenience accessors (full image) -------------------------------

    @property
    def P(self) -> int:
        """Number of endmembers."""
        return self.meta.num_endmembers

    @property
    def L(self) -> int:
        """Number of spectral bands."""
        return self.meta.num_bands

    @property
    def col(self) -> int:
        """Spatial side length (image is col×col)."""
        return self.meta.spatial_size

    def get_image_cube(self) -> torch.Tensor:
        """
        Returns the full HSI as a 4-D tensor ready for Conv2D:
            shape (1, L, H, W)  where H = W = col.
        """
        return self.Y.T.reshape(1, self.L, self.col, self.col)

    def get_abundance_cube(self) -> torch.Tensor:
        """
        Returns the full abundance maps as a 3-D tensor:
            shape (H, W, P)  — consistent with evaluation code.
        """
        return self.A.reshape(self.col, self.col, self.P)

    def get_endmembers(self) -> torch.Tensor:
        """Ground-truth endmembers (L, P) on CPU."""
        return self.M

    def get_init_weight(self) -> torch.Tensor:
        """
        VCA-initialized endmembers reshaped for Conv2d decoder weight:
            shape (L, P, 1, 1).
        """
        return self.M1.unsqueeze(-1).unsqueeze(-1)

    def __repr__(self) -> str:
        return (
            f"HyperspectralDataset("
            f"name={self.dataset_name!r}, "
            f"pixels={len(self)}, "
            f"bands={self.L}, "
            f"endmembers={self.P}, "
            f"spatial={self.col}x{self.col})"
        )
