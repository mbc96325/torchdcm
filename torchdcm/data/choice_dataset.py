from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

import numpy as np
import pandas as pd
import torch


@dataclass(frozen=True)
class ChoiceDataset:
    """Ragged long-format choice data.

    Each observation occupies the segment ``obs_ptr[n]:obs_ptr[n + 1]`` in the
    row-level tensors. ``chosen_row[n]`` is the global row index of the selected
    alternative for that observation.
    """

    obs_ptr: torch.Tensor
    alt_id: torch.Tensor
    chosen_row: torch.Tensor
    x_alt: dict[str, torch.Tensor]
    weights: torch.Tensor
    availability: torch.Tensor
    obs_ids: list
    alt_names: list[str]
    obs_to_ind: torch.Tensor | None = None
    individual_ids: list | None = None

    def __post_init__(self) -> None:
        if self.obs_ptr.ndim != 1:
            raise ValueError("obs_ptr must be one-dimensional.")
        if self.alt_id.ndim != 1:
            raise ValueError("alt_id must be one-dimensional.")
        if len(self.chosen_row) != self.n_obs:
            raise ValueError("chosen_row must have one entry per observation.")
        if len(self.weights) != self.n_obs:
            raise ValueError("weights must have one entry per observation.")
        if len(self.availability) != self.n_rows:
            raise ValueError("availability must have one entry per long row.")
        for name, value in self.x_alt.items():
            if len(value) != self.n_rows:
                raise ValueError(f"x_alt[{name!r}] must have one value per long row.")

    @property
    def n_obs(self) -> int:
        return int(self.obs_ptr.numel() - 1)

    @property
    def n_rows(self) -> int:
        return int(self.alt_id.numel())

    @property
    def n_alternatives(self) -> int:
        return len(self.alt_names)

    @property
    def dtype(self) -> torch.dtype:
        return self.weights.dtype

    @property
    def device(self) -> torch.device:
        return self.weights.device

    def to(self, device: str | torch.device | None = None, dtype: torch.dtype | None = None) -> "ChoiceDataset":
        dtype = dtype or self.dtype
        device = device or self.device
        return ChoiceDataset(
            obs_ptr=self.obs_ptr.to(device=device),
            alt_id=self.alt_id.to(device=device),
            chosen_row=self.chosen_row.to(device=device),
            x_alt={k: v.to(device=device, dtype=dtype) for k, v in self.x_alt.items()},
            weights=self.weights.to(device=device, dtype=dtype),
            availability=self.availability.to(device=device),
            obs_ids=list(self.obs_ids),
            alt_names=list(self.alt_names),
            obs_to_ind=None if self.obs_to_ind is None else self.obs_to_ind.to(device=device),
            individual_ids=None if self.individual_ids is None else list(self.individual_ids),
        )

    def cluster_codes(self, groups: str | Iterable | torch.Tensor | None = None) -> torch.Tensor | None:
        """Return integer cluster codes aligned to observations."""

        if groups is None:
            return None
        if isinstance(groups, torch.Tensor):
            if len(groups) != self.n_obs:
                raise ValueError("Cluster tensor must have one entry per observation.")
            return groups.to(device=self.device, dtype=torch.long)
        if isinstance(groups, str):
            if groups not in {"individual", "person_id", "panel"}:
                raise ValueError("String groups currently supports 'individual', 'person_id', or 'panel'.")
            if self.obs_to_ind is None:
                raise ValueError("Dataset has no individual_id mapping for cluster covariance.")
            return self.obs_to_ind.to(device=self.device, dtype=torch.long)
        values = list(groups)
        if len(values) != self.n_obs:
            raise ValueError("Cluster groups must have one entry per observation.")
        _, codes = np.unique(np.asarray(values), return_inverse=True)
        return torch.as_tensor(codes, dtype=torch.long, device=self.device)

    @classmethod
    def from_long(
        cls,
        df: pd.DataFrame,
        *,
        obs_id: str,
        alt_id: str,
        choice: str,
        variables: Iterable[str],
        availability: str | None = None,
        weight: str | None = None,
        individual_id: str | None = None,
        alt_order: Iterable[str] | None = None,
        dtype: torch.dtype = torch.float64,
        device: str | torch.device = "cpu",
    ) -> "ChoiceDataset":
        """Build a ragged dataset from one row per observation-alternative."""

        variables = list(variables)
        missing = [c for c in [obs_id, alt_id, choice, *variables] if c not in df.columns]
        if missing:
            raise ValueError(f"Missing columns in long data: {missing}")

        frame = df.copy()
        if alt_order is None:
            alt_names = list(pd.unique(frame[alt_id]))
        else:
            alt_names = list(alt_order)
        alt_to_code = {name: i for i, name in enumerate(alt_names)}
        unknown_alts = sorted(set(frame[alt_id]) - set(alt_to_code))
        if unknown_alts:
            raise ValueError(f"Unknown alternatives not present in alt_order: {unknown_alts}")

        frame["_torchdcm_alt_code"] = frame[alt_id].map(alt_to_code).astype(int)
        frame["_torchdcm_order"] = np.arange(len(frame))
        frame = frame.sort_values([obs_id, "_torchdcm_alt_code", "_torchdcm_order"], kind="stable")

        obs_ids = list(pd.unique(frame[obs_id]))
        starts = [0]
        chosen_rows: list[int] = []
        weights: list[float] = []
        obs_to_ind: list[int] = []
        individual_ids: list | None = [] if individual_id else None
        individual_map = {}

        row_cursor = 0
        for _, group in frame.groupby(obs_id, sort=False):
            n_rows = len(group)
            starts.append(starts[-1] + n_rows)
            chosen_mask = group[choice].to_numpy().astype(bool)
            if chosen_mask.sum() != 1:
                raise ValueError("Each observation must have exactly one chosen row.")
            chosen_rows.append(row_cursor + int(np.flatnonzero(chosen_mask)[0]))
            row_cursor += n_rows
            if weight is None:
                weights.append(1.0)
            else:
                unique_weights = pd.unique(group[weight])
                if len(unique_weights) != 1:
                    raise ValueError("Observation weights must be constant within an observation.")
                weights.append(float(unique_weights[0]))
            if individual_id:
                unique_ind = pd.unique(group[individual_id])
                if len(unique_ind) != 1:
                    raise ValueError("individual_id must be constant within an observation.")
                ind = unique_ind[0]
                if ind not in individual_map:
                    individual_map[ind] = len(individual_map)
                    assert individual_ids is not None
                    individual_ids.append(ind)
                obs_to_ind.append(individual_map[ind])

        avail_values = (
            np.ones(len(frame), dtype=bool)
            if availability is None
            else frame[availability].to_numpy().astype(bool)
        )
        chosen_avail = avail_values[np.asarray(chosen_rows, dtype=int)]
        if not chosen_avail.all():
            raise ValueError("Chosen alternatives must be available.")

        return cls(
            obs_ptr=torch.as_tensor(starts, dtype=torch.long, device=device),
            alt_id=torch.as_tensor(frame["_torchdcm_alt_code"].to_numpy(copy=True), dtype=torch.long, device=device),
            chosen_row=torch.as_tensor(chosen_rows, dtype=torch.long, device=device),
            x_alt={
                name: torch.as_tensor(frame[name].to_numpy(dtype=float, copy=True), dtype=dtype, device=device)
                for name in variables
            },
            weights=torch.as_tensor(weights, dtype=dtype, device=device),
            availability=torch.as_tensor(avail_values, dtype=torch.bool, device=device),
            obs_ids=obs_ids,
            alt_names=alt_names,
            obs_to_ind=None if individual_id is None else torch.as_tensor(obs_to_ind, dtype=torch.long, device=device),
            individual_ids=individual_ids,
        )

    @classmethod
    def from_wide(
        cls,
        df: pd.DataFrame,
        *,
        alternatives: Iterable[str],
        choice: str,
        variables: Mapping[str, Mapping[str, str] | str],
        availability: Mapping[str, str] | None = None,
        obs_id: str | None = None,
        weight: str | None = None,
        individual_id: str | None = None,
        dtype: torch.dtype = torch.float64,
        device: str | torch.device = "cpu",
    ) -> "ChoiceDataset":
        """Convert a wide mode-choice table to long format and return a dataset."""

        alternatives = list(alternatives)
        rows = []
        for row_no, (_, row) in enumerate(df.iterrows()):
            obs_value = row[obs_id] if obs_id else row_no
            chosen_alt = row[choice]
            for alt in alternatives:
                long_row = {
                    "_obs_id": obs_value,
                    "_alt_id": alt,
                    "_chosen": chosen_alt == alt,
                }
                if weight:
                    long_row["_weight"] = row[weight]
                if individual_id:
                    long_row["_individual_id"] = row[individual_id]
                if availability is not None:
                    long_row["_availability"] = bool(row[availability[alt]])
                for var_name, columns in variables.items():
                    if isinstance(columns, str):
                        column = columns.format(alt=alt, ALT=alt.upper(), lower=alt.lower())
                    else:
                        column = columns[alt]
                    long_row[var_name] = row[column]
                rows.append(long_row)
        long_df = pd.DataFrame(rows)
        return cls.from_long(
            long_df,
            obs_id="_obs_id",
            alt_id="_alt_id",
            choice="_chosen",
            variables=list(variables.keys()),
            availability="_availability" if availability is not None else None,
            weight="_weight" if weight else None,
            individual_id="_individual_id" if individual_id else None,
            alt_order=alternatives,
            dtype=dtype,
            device=device,
        )
