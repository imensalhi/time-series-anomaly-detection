from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from .modeling_service import NTXentLoss


def train_autoencoder(
    model: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int,
    learning_rate: float,
) -> Dict[str, list]:
    model.to(device)
    criterion = torch.nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    history = {"train_loss": [], "val_loss": []}
    for _ in range(epochs):
        model.train()
        train_sum, train_count = 0.0, 0
        for (x,) in train_loader:
            x = x.to(device)
            pred = model(x)
            loss = criterion(pred, x)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_sum += float(loss.item()) * x.size(0)
            train_count += x.size(0)

        model.eval()
        val_sum, val_count = 0.0, 0
        with torch.no_grad():
            for (x,) in val_loader:
                x = x.to(device)
                pred = model(x)
                loss = criterion(pred, x)
                val_sum += float(loss.item()) * x.size(0)
                val_count += x.size(0)

        history["train_loss"].append(train_sum / max(train_count, 1))
        history["val_loss"].append(val_sum / max(val_count, 1))

    return history


def reconstruction_scores(model: torch.nn.Module, x: np.ndarray, device: torch.device) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        tensor = torch.tensor(x, dtype=torch.float32, device=device)
        pred = model(tensor)
        loss = ((tensor - pred) ** 2).mean(dim=(1, 2, 3))
    return loss.detach().cpu().numpy()


def train_contrastive(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    epochs: int,
    learning_rate: float,
    temperature: float,
) -> Dict[str, list]:
    model.to(device)
    loss_fn = NTXentLoss(temperature=temperature)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    history = {"loss": []}
    for _ in range(epochs):
        model.train()
        total_loss, n_batches = 0.0, 0
        for x1, x2 in loader:
            x1 = x1.to(device)
            x2 = x2.to(device)
            z1 = model(x1)
            z2 = model(x2)
            loss = loss_fn(z1, z2)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item())
            n_batches += 1

        history["loss"].append(total_loss / max(n_batches, 1))

    return history


def embedding_scores(
    model: torch.nn.Module,
    train_windows: np.ndarray,
    test_windows: np.ndarray,
    device: torch.device,
    batch_size: int = 512,
) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()

    def encode(arr: np.ndarray) -> np.ndarray:
        out = []
        with torch.no_grad():
            for i in range(0, len(arr), batch_size):
                batch = torch.tensor(arr[i : i + batch_size], dtype=torch.float32, device=device)
                emb = model.encode(batch).cpu().numpy()
                out.append(emb)
        return np.concatenate(out, axis=0)

    emb_train = encode(train_windows)
    emb_test = encode(test_windows)
    centroid = emb_train.mean(axis=0)

    train_scores = np.sqrt(((emb_train - centroid) ** 2).sum(axis=1))
    test_scores = np.sqrt(((emb_test - centroid) ** 2).sum(axis=1))
    return train_scores, test_scores
