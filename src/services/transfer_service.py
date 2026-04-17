from __future__ import annotations

import torch


class TransferLearningService:
    """Utilities for transfer learning between machine groups."""

    @staticmethod
    def load_and_freeze_encoder(model: torch.nn.Module, checkpoint_path: str) -> torch.nn.Module:
        state = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(state)

        if hasattr(model, "encoder"):
            for param in model.encoder.parameters():
                param.requires_grad = False
        elif hasattr(model, "backbone"):
            for param in model.backbone.parameters():
                param.requires_grad = False

        return model
