"""
PyTorch Dataset classes for BioFlight imitation learning.
Provides tensor slicing, batching, and DataLoader constructors for processed flight arrays.
"""

from typing import Tuple
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class FlightNPZDataset(Dataset):
    """PyTorch Dataset loading preprocessed numpy arrays."""

    def __init__(self, X: np.ndarray, Y: np.ndarray):
        assert len(X) == len(Y), f"Mismatched shapes: {len(X)} vs {len(Y)}"
        self.X = torch.from_numpy(X).float()
        self.Y = torch.from_numpy(Y).float()

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.Y[idx]


def get_flight_dataloaders(
    npz_path: str = "data/processed/flight_data_processed.npz",
    batch_size: int = 256,
    num_workers: int = 0
) -> Tuple[DataLoader, DataLoader, DataLoader, dict]:
    """Loads npz file and returns (train_loader, val_loader, test_loader, metadata)."""
    data = np.load(npz_path)

    train_ds = FlightNPZDataset(data["X_train"], data["Y_train"])
    val_ds = FlightNPZDataset(data["X_val"], data["Y_val"])
    test_ds = FlightNPZDataset(data["X_test"], data["Y_test"])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    meta = {
        "input_dim": data["X_train"].shape[1],
        "target_dim": data["Y_train"].shape[1],
        "mean": data["mean"],
        "std": data["std"],
    }
    return train_loader, val_loader, test_loader, meta
