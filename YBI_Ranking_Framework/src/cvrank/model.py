from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch
from torch import nn


@dataclass
class MLPConfig:
    input_dim: int
    hidden_dims: tuple[int, ...] = (128, 64)
    dropout: float = 0.2
    dropout_mode: str = "all"
    batch_norm_mode: str = "all"


def normalize_batch_norm_mode(value: str | None) -> str:
    text = str(value or "all").strip().lower()
    aliases = {
        "all": "all",
        "each": "all",
        "every": "all",
        "first": "first",
        "first_only": "first",
        "1": "first",
        "none": "none",
        "off": "none",
        "disabled": "none",
        "0": "none",
    }
    normalized = aliases.get(text)
    if normalized is None:
        raise ValueError(f"Unsupported batch norm mode: {value!r}")
    return normalized


def should_apply_batch_norm(batch_norm_mode: str, layer_index: int) -> bool:
    normalized = normalize_batch_norm_mode(batch_norm_mode)
    if normalized == "all":
        return True
    if normalized == "first":
        return layer_index == 0
    return False


def normalize_dropout_mode(value: str | None) -> str:
    text = str(value or "all").strip().lower()
    aliases = {
        "all": "all",
        "each": "all",
        "every": "all",
        "first": "first",
        "first_only": "first",
        "1": "first",
        "none": "none",
        "off": "none",
        "disabled": "none",
        "0": "none",
    }
    normalized = aliases.get(text)
    if normalized is None:
        raise ValueError(f"Unsupported dropout mode: {value!r}")
    return normalized


def should_apply_dropout(dropout_mode: str, layer_index: int) -> bool:
    normalized = normalize_dropout_mode(dropout_mode)
    if normalized == "all":
        return True
    if normalized == "first":
        return layer_index == 0
    return False


class MLPBackbone(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: Iterable[int] = (128, 64),
        dropout: float = 0.2,
        dropout_mode: str = "all",
        batch_norm_mode: str = "all",
    ) -> None:
        super().__init__()
        hidden_dims = tuple(hidden_dims)
        dims = [input_dim, *hidden_dims]
        layers: list[nn.Module] = []
        normalized_dropout_mode = normalize_dropout_mode(dropout_mode)
        normalized_batch_norm_mode = normalize_batch_norm_mode(batch_norm_mode)
        for layer_index, (left, right) in enumerate(zip(dims[:-1], dims[1:], strict=False)):
            layers.append(nn.Linear(left, right))
            if should_apply_batch_norm(normalized_batch_norm_mode, layer_index):
                layers.append(nn.BatchNorm1d(right))
            layers.append(nn.ReLU())
            if float(dropout) > 0 and should_apply_dropout(normalized_dropout_mode, layer_index):
                layers.append(nn.Dropout(dropout))
        self.backbone = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


class CVRankNetModel(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: Iterable[int] = (128, 64, 32),
        dropout: float = 0.2,
        dropout_mode: str = "all",
        batch_norm_mode: str = "all",
    ) -> None:
        super().__init__()
        hidden_dims = tuple(hidden_dims)
        self.encoder = MLPBackbone(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            dropout=dropout,
            dropout_mode=dropout_mode,
            batch_norm_mode=batch_norm_mode,
        )
        last_dim = hidden_dims[-1] if hidden_dims else input_dim
        self.head = nn.Linear(last_dim, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.encoder(x)
        logit = self.head(features)
        prob = torch.sigmoid(logit)
        return logit, prob
