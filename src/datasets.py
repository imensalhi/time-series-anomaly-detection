from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset


class ContrastiveWindowsDataset(Dataset):
    def __init__(self, windows: np.ndarray, jitter_std: float = 0.05) -> None:
        self.windows = windows
        self.jitter_std = jitter_std

    def __len__(self) -> int:
        return len(self.windows)

    def _augment(self, window: np.ndarray) -> np.ndarray:
        w = window.copy()
        noise = np.random.normal(0, self.jitter_std, len(w)).astype(np.float32)
        shift = int(np.random.randint(-8, 9))
        w = np.roll(w + noise, shift)
        return w.astype(np.float32)

    def __getitem__(self, idx: int):
        base = self.windows[idx]
        v1 = torch.tensor(self._augment(base), dtype=torch.float32).unsqueeze(0)
        v2 = torch.tensor(self._augment(base), dtype=torch.float32).unsqueeze(0)
        return v1, v2
