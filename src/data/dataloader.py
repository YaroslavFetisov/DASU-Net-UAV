"""
DataLoader factory — creates PyTorch DataLoader for the hyperspectral dataset.

Two operating modes:
    1. ``mode="full"`` (default)  — the entire dataset in one batch,
       like in the original DeepTrans-HSU (batch_size = col²).
    2. ``mode="patch"``  — supports mini-batch training with random
       pixel shuffling (for future experiments).
"""

from __future__ import annotations

from typing import Optional

import torch
from torch.utils.data import DataLoader

from .dataset import HyperspectralDataset


def create_dataloader(
    dataset: HyperspectralDataset,
    mode: str = "full",
    batch_size: Optional[int] = None,
    shuffle: bool = False,
    num_workers: int = 0,
    pin_memory: bool = False,
) -> DataLoader:
    """
    Create a DataLoader for a HyperspectralDataset.

    Parameters
    ----------
    dataset : HyperspectralDataset
        The loaded hyperspectral dataset.
    mode : {"full", "patch"}
        ``"full"``  — single batch containing all pixels (default, matches
                      original paper).
        ``"patch"`` — mini-batch mode; uses ``batch_size`` and ``shuffle``.
    batch_size : int, optional
        Number of pixels per batch.  Ignored when ``mode="full"`` (set
        automatically to ``col²``).
    shuffle : bool
        Whether to shuffle pixels every epoch.  Default ``False`` for full
        mode, can be ``True`` for patch mode.
    num_workers : int
        Number of data-loading subprocess workers (0 = main process).
    pin_memory : bool
        If True, DataLoader will copy tensors into pinned memory before
        returning them.  Useful when data lives on CPU but model on GPU.

    Returns
    -------
    DataLoader
        A PyTorch DataLoader that yields ``(spectra, abundances)`` batches.

    Examples
    --------
    >>> from src.data import HyperspectralDataset, create_dataloader
    >>> ds = HyperspectralDataset("samson", data_dir="./data/raw")
    >>> loader = create_dataloader(ds, mode="full")
    >>> for spectra, abundances in loader:
    ...     print(spectra.shape)  # (9025, 156)
    """

    if mode == "full":
        # Replicates original behaviour: one giant batch = all pixels
        batch_size = dataset.col ** 2
        shuffle = False
    elif mode == "patch":
        if batch_size is None:
            batch_size = 256  # reasonable default for mini-batch
    else:
        raise ValueError(f"Unknown mode '{mode}'. Choose 'full' or 'patch'.")

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )

    return loader
