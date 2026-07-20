from __future__ import annotations

from dataclasses import dataclass
from math import erfc, sqrt
from pathlib import Path
from statistics import NormalDist
from typing import Iterable

import numpy as np
import pandas as pd
import torch

from torchdcm.results.report import EstimationReport


@dataclass
class ChoiceResults:
    model: object
    data: object
    params: torch.Tensor
    param_names: list[str]
    loglike: float
    null_loglike: float
    gradient: torch.Tensor
    hessian: torch.Tensor
    covariances: dict[str, torch.Tensor]
    cov_type: str
    n_obs: int
    n_params: int
    convergence_status: dict

    @property
    def values(self) -> np.ndarray:
        return self.params.detach().cpu().numpy()

    @property
    def bse(self) -> np.ndarray:
        diag = torch.diag(self.cov_params())
        return torch.sqrt(torch.clamp(diag, min=0)).detach().cpu().numpy()

    @property
    def tvalues(self) -> np.ndarray:
        """Backward-compatible alias for asymptotic z statistics."""

        return self.zvalues

    @property
    def zvalues(self) -> np.ndarray:
        return self.values / self.bse

    @property
    def pvalues(self) -> np.ndarray:
        return np.asarray([erfc(abs(float(z)) / sqrt(2.0)) for z in self.zvalues])

    @property
    def aic(self) -> float:
        return -2.0 * self.loglike + 2.0 * self.n_params

    @property
    def bic(self) -> float:
        return -2.0 * self.loglike + np.log(self.n_obs) * self.n_params

    @property
    def rho2(self) -> float:
        return 1.0 - self.loglike / self.null_loglike

    @property
    def rho2_bar(self) -> float:
        return 1.0 - (self.loglike - self.n_params) / self.null_loglike

    def cov_params(self, cov_type: str | None = None) -> torch.Tensor:
        cov_type = cov_type or self.cov_type
        if cov_type not in self.covariances:
            raise ValueError(f"Covariance type {cov_type!r} is not available.")
        return self.covariances[cov_type]

    def get_robustcov_results(self, cov_type: str, groups=None) -> "ChoiceResults":
        if cov_type == "cluster" and "cluster" not in self.covariances:
            cluster_codes = self.data.cluster_codes(groups)
            scores = self.model.scores(self.params, self.data)
            h_inv = self.covariances["classic"]
            meat = torch.zeros_like(h_inv)
            for code in range(int(cluster_codes.max().detach().cpu()) + 1):
                s = scores[cluster_codes == code].sum(dim=0)
                meat = meat + torch.outer(s, s)
            self.covariances["cluster"] = h_inv @ meat @ h_inv
        clone = ChoiceResults(**{**self.__dict__, "cov_type": cov_type})
        return clone

    def conf_int(self, alpha: float = 0.05) -> np.ndarray:
        if not 0.0 < alpha < 1.0:
            raise ValueError("alpha must lie strictly between zero and one.")
        z = NormalDist().inv_cdf(1.0 - alpha / 2.0)
        return np.column_stack([self.values - z * self.bse, self.values + z * self.bse])

    def report(
        self,
        *,
        cov_type: str | None = None,
        confidence_level: float = 0.95,
        title: str | None = None,
    ) -> EstimationReport:
        """Build a structured, single-model estimation report."""

        return EstimationReport.from_results(
            self,
            cov_type=cov_type,
            confidence_level=confidence_level,
            title=title,
        )

    def summary(
        self,
        *,
        cov_type: str | None = None,
        confidence_level: float = 0.95,
    ) -> str:
        """Render the organized estimation report as console text."""

        return self.report(cov_type=cov_type, confidence_level=confidence_level).to_text()

    def parameter_table(
        self,
        *,
        cov_type: str | None = None,
        confidence_level: float = 0.95,
    ) -> pd.DataFrame:
        """Return the report's parameter table as a pandas DataFrame."""

        return self.report(cov_type=cov_type, confidence_level=confidence_level).parameters.copy()

    def save_report(
        self,
        directory: str | Path,
        *,
        formats: Iterable[str] = ("html", "json", "csv", "latex", "text"),
        cov_type: str | None = None,
        confidence_level: float = 0.95,
        title: str | None = None,
    ) -> dict[str, list[Path]]:
        """Write a self-contained report artifact directory."""

        return self.report(
            cov_type=cov_type,
            confidence_level=confidence_level,
            title=title,
        ).save(directory, formats=formats)

    def predict_proba(self, data=None) -> np.ndarray:
        data = data or self.data
        return self.model.predict_proba(data, self.params).detach().cpu().numpy()

    def predict(self, data=None) -> list[str]:
        data = data or self.data
        return self.model.predict(data, self.params)

    def score(self, data=None) -> float:
        data = data or self.data
        return float(self.model.loglike(self.params, data).detach().cpu())

    def wtp(self, attribute: str, cost: str) -> dict[str, float]:
        """Delta-method WTP for ``- beta_attribute / beta_cost``."""

        idx_attr = self.param_names.index(attribute)
        idx_cost = self.param_names.index(cost)
        beta = self.params.detach()
        estimate = -beta[idx_attr] / beta[idx_cost]
        grad = torch.zeros_like(beta)
        grad[idx_attr] = -1.0 / beta[idx_cost]
        grad[idx_cost] = beta[idx_attr] / (beta[idx_cost] ** 2)
        var = grad @ self.cov_params() @ grad
        se = torch.sqrt(torch.clamp(var, min=0))
        z = estimate / se
        pvalue = erfc(abs(float(z.detach().cpu())) / sqrt(2.0))
        return {
            "estimate": float(estimate.detach().cpu()),
            "std_err": float(se.detach().cpu()),
            "z": float(z.detach().cpu()),
            "pvalue": float(pvalue),
        }

    def elasticity(self, variable: str, coefficient: str) -> np.ndarray:
        """Direct point elasticities for rows where ``variable`` is defined.

        For MNL, own elasticity is ``beta * x * (1 - P)`` for each long row.
        """

        probs = torch.as_tensor(self.predict_proba(), dtype=self.params.dtype)
        x = self.data.x_alt[variable].detach().cpu()
        beta = self.params[self.param_names.index(coefficient)].detach().cpu()
        return (beta * x * (1.0 - probs)).numpy()
