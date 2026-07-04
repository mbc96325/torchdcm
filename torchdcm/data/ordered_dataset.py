from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
import torch


@dataclass(frozen=True)
class OrderedChoiceDataset:
    """Tabular ordered outcome data."""

    y: torch.Tensor
    x: dict[str, torch.Tensor]
    weights: torch.Tensor
    categories: list[int]

    def __post_init__(self) -> None:
        if self.y.ndim != 1:
            raise ValueError("y must be one-dimensional.")
        if self.weights.ndim != 1 or len(self.weights) != self.n_obs:
            raise ValueError("weights must have one entry per observation.")
        for name, value in self.x.items():
            if value.ndim != 1 or len(value) != self.n_obs:
                raise ValueError(f"x[{name!r}] must have one value per observation.")

    @property
    def n_obs(self) -> int:
        return int(self.y.numel())

    @property
    def dtype(self) -> torch.dtype:
        return self.weights.dtype

    @property
    def device(self) -> torch.device:
        return self.weights.device

    def to(self, device: str | torch.device | None = None, dtype: torch.dtype | None = None) -> "OrderedChoiceDataset":
        dtype = dtype or self.dtype
        device = torch.device(device or self.device)
        if device == self.device and dtype == self.dtype:
            return self
        return OrderedChoiceDataset(
            y=self.y.to(device=device),
            x={name: value.to(device=device, dtype=dtype) for name, value in self.x.items()},
            weights=self.weights.to(device=device, dtype=dtype),
            categories=list(self.categories),
        )

    @classmethod
    def from_dataframe(
        cls,
        df: pd.DataFrame,
        *,
        outcome: str,
        variables: Iterable[str],
        categories: Iterable[int] | None = None,
        weight: str | None = None,
        dtype: torch.dtype = torch.float64,
        device: str | torch.device = "cpu",
    ) -> "OrderedChoiceDataset":
        variables = list(variables)
        missing = [column for column in [outcome, *variables] if column not in df.columns]
        if missing:
            raise ValueError(f"Missing columns: {missing}")
        if categories is None:
            categories = sorted(int(value) for value in pd.unique(df[outcome].dropna()))
        categories = list(categories)
        category_to_code = {category: index for index, category in enumerate(categories)}
        unknown = sorted(set(int(value) for value in pd.unique(df[outcome].dropna())) - set(category_to_code))
        if unknown:
            raise ValueError(f"Outcome contains values not present in categories: {unknown}")
        y = np.asarray([category_to_code[int(value)] for value in df[outcome]], dtype=np.int64)
        if weight is None:
            weights = np.ones(len(df), dtype=float)
        else:
            weights = df[weight].to_numpy(dtype=float, copy=True)
        return cls(
            y=torch.as_tensor(y, dtype=torch.long, device=device),
            x={
                name: torch.as_tensor(df[name].to_numpy(dtype=float, copy=True), dtype=dtype, device=device)
                for name in variables
            },
            weights=torch.as_tensor(weights, dtype=dtype, device=device),
            categories=categories,
        )
