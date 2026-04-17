from __future__ import annotations

from typing import Dict

import torch
from torch.utils.data import DataLoader


class FineTuningService:
    """Fine-tune a pre-trained model on new normal run data."""

    @staticmethod
    def fine_tune_autoencoder(
        model: torch.nn.Module,
        loader: DataLoader,
        device: torch.device,
        epochs: int = 5,
        learning_rate: float = 1e-4,
    ) -> Dict[str, list]:
        model.to(device)
        model.train()
        loss_fn = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        history = {"loss": []}
        for _ in range(epochs):
            total, count = 0.0, 0
            for (x,) in loader:
                x = x.to(device)
                recon = model(x)
                loss = loss_fn(recon, x)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total += float(loss.item()) * x.size(0)
                count += x.size(0)
            history["loss"].append(total / max(count, 1))

        return history
