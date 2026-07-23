from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class PanelStructure:
    """Observation-to-individual mapping for repeated-choice likelihoods."""

    obs_to_unit: torch.Tensor
    unit_ids: list

    def __post_init__(self) -> None:
        if self.obs_to_unit.ndim != 1:
            raise ValueError("obs_to_unit must be one-dimensional.")
        if self.obs_to_unit.numel() and int(self.obs_to_unit.min().detach().cpu()) < 0:
            raise ValueError("obs_to_unit codes must be non-negative.")
        if self.obs_to_unit.numel() and int(self.obs_to_unit.max().detach().cpu()) >= len(self.unit_ids):
            raise ValueError("unit_ids must contain one entry for each unit code.")

    @property
    def n_obs(self) -> int:
        return int(self.obs_to_unit.numel())

    @property
    def n_units(self) -> int:
        return len(self.unit_ids)

    @property
    def device(self) -> torch.device:
        return self.obs_to_unit.device

    def to(self, device: str | torch.device | None = None) -> "PanelStructure":
        device = torch.device(device or self.device)
        if device == self.device:
            return self
        return PanelStructure(obs_to_unit=self.obs_to_unit.to(device=device), unit_ids=list(self.unit_ids))

    def sum_by_unit(self, values: torch.Tensor) -> torch.Tensor:
        """Sum observation-level values by individual.

        ``values`` must have observations on axis 0. Remaining dimensions are
        preserved, so this works for ``(n_obs,)`` and ``(n_obs, n_draws)``.
        """

        if values.shape[0] != self.n_obs:
            raise ValueError("values must have one row per observation.")
        output = torch.zeros(
            (self.n_units, *values.shape[1:]),
            dtype=values.dtype,
            device=values.device,
        )
        # ``index_add`` performs the group sum on the active device and avoids
        # a Python loop over individuals.
        return output.index_add(0, self.obs_to_unit.to(device=values.device), values)

    def logmeanexp_by_unit(self, obs_log_values: torch.Tensor, dim: int = 1) -> torch.Tensor:
        """Aggregate observation log-probabilities into panel log likelihoods.

        The common mixed-logit case passes an ``(n_obs, n_draws)`` tensor. The
        method first sums repeated observations for each individual and then
        averages over draws in log space.
        """

        unit_log_values = self.sum_by_unit(obs_log_values)
        # Sum across repeated choices before integrating over draws.  Reversing
        # these operations would incorrectly allow tastes to change by occasion.
        return torch.logsumexp(unit_log_values, dim=dim) - torch.log(
            torch.as_tensor(unit_log_values.shape[dim], dtype=unit_log_values.dtype, device=unit_log_values.device)
        )
