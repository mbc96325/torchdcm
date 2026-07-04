from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch

from torchdcm.data.choice_dataset import ChoiceDataset
from torchdcm.results.result import ChoiceResults
from torchdcm.spec.utility import UtilitySpec


@dataclass(frozen=True)
class CompiledUtility:
    design: torch.Tensor
    free_names: list[str]
    fixed_names: list[str]
    all_names: list[str]
    free_initial: torch.Tensor
    fixed_values: torch.Tensor
    fixed_design: torch.Tensor
    choice_set_width: int | None


class MultinomialLogit:
    """Multinomial/conditional logit model with ragged choice sets."""

    def __init__(
        self,
        spec: UtilitySpec,
        *,
        dtype: torch.dtype = torch.float64,
        device: str | torch.device = "cpu",
        max_iter: int = 200,
        tolerance_grad: float = 1e-7,
        line_search_fn: str | None = "strong_wolfe",
    ) -> None:
        self.spec = spec
        self.dtype = dtype
        self.device = torch.device(device)
        self.max_iter = max_iter
        self.tolerance_grad = tolerance_grad
        self.line_search_fn = line_search_fn
        self._compiled_cache: dict[int, CompiledUtility] = {}

    @classmethod
    def from_formula(cls, utilities: dict[str, str], **kwargs) -> "MultinomialLogit":
        return cls(UtilitySpec.from_formula(utilities), **kwargs)

    def compile(self, data: ChoiceDataset) -> CompiledUtility:
        data = data.to(device=self.device, dtype=self.dtype)
        cache_key = id(data)
        if cache_key in self._compiled_cache:
            return self._compiled_cache[cache_key]

        alt_to_code = {name: i for i, name in enumerate(data.alt_names)}
        missing_alts = sorted(set(self.spec.utilities) - set(alt_to_code))
        if missing_alts:
            raise ValueError(f"Specification contains alternatives not in data: {missing_alts}")

        params = self.spec.parameters
        free_params = [p for p in params if not p.fixed]
        fixed_params = [p for p in params if p.fixed]
        free_index = {p.name: i for i, p in enumerate(free_params)}
        fixed_index = {p.name: i for i, p in enumerate(fixed_params)}
        design = torch.zeros((data.n_rows, len(free_params)), dtype=self.dtype, device=self.device)
        fixed_design = torch.zeros((data.n_rows, len(fixed_params)), dtype=self.dtype, device=self.device)

        for alt_name, expr in self.spec.utilities.items():
            rows = data.alt_id == alt_to_code[alt_name]
            for term in expr.terms:
                values = (
                    torch.ones(data.n_rows, dtype=self.dtype, device=self.device)
                    if term.variable is None
                    else data.x_alt[term.variable].to(device=self.device, dtype=self.dtype)
                )
                contribution = term.multiplier * values
                if term.parameter.fixed:
                    fixed_design[rows, fixed_index[term.parameter.name]] += contribution[rows]
                else:
                    design[rows, free_index[term.parameter.name]] += contribution[rows]

        compiled = CompiledUtility(
            design=design,
            free_names=[p.name for p in free_params],
            fixed_names=[p.name for p in fixed_params],
            all_names=[p.name for p in params],
            free_initial=torch.as_tensor([p.init for p in free_params], dtype=self.dtype, device=self.device),
            fixed_values=torch.as_tensor([p.init for p in fixed_params], dtype=self.dtype, device=self.device),
            fixed_design=fixed_design,
            choice_set_width=_balanced_width(data),
        )
        self._compiled_cache[cache_key] = compiled
        return compiled

    def utilities(self, params: torch.Tensor, data: ChoiceDataset, compiled: CompiledUtility | None = None) -> torch.Tensor:
        data = data.to(device=self.device, dtype=self.dtype)
        compiled = compiled or self.compile(data)
        utility = compiled.design @ params
        if compiled.fixed_values.numel():
            utility = utility + compiled.fixed_design @ compiled.fixed_values
        return utility

    def loglike_per_obs(
        self,
        params: torch.Tensor,
        data: ChoiceDataset,
        compiled: CompiledUtility | None = None,
    ) -> torch.Tensor:
        data = data.to(device=self.device, dtype=self.dtype)
        compiled = compiled or self.compile(data)
        utility = self.utilities(params, data, compiled)
        width = compiled.choice_set_width
        if width is not None:
            utility_by_obs = utility.reshape(data.n_obs, width)
            availability = data.availability.reshape(data.n_obs, width)
            if not bool(availability.any(dim=1).all()):
                raise ValueError("Every observation must have at least one available alternative.")
            chosen_local = (data.chosen_row - data.obs_ptr[:-1]).reshape(-1, 1)
            chosen_utility = utility_by_obs.gather(1, chosen_local).squeeze(1)
            log_denom = torch.logsumexp(utility_by_obs.masked_fill(~availability, -torch.inf), dim=1)
            return data.weights * (chosen_utility - log_denom)

        parts = []
        for obs in range(data.n_obs):
            start = int(data.obs_ptr[obs])
            end = int(data.obs_ptr[obs + 1])
            segment = utility[start:end]
            mask = data.availability[start:end]
            if not bool(mask.any()):
                raise ValueError("Every observation must have at least one available alternative.")
            chosen = int(data.chosen_row[obs])
            log_denom = torch.logsumexp(segment[mask], dim=0)
            parts.append(data.weights[obs] * (utility[chosen] - log_denom))
        return torch.stack(parts)

    def loglike(self, params: torch.Tensor, data: ChoiceDataset, compiled: CompiledUtility | None = None) -> torch.Tensor:
        return self.loglike_per_obs(params, data, compiled).sum()

    def fit(
        self,
        data: ChoiceDataset,
        *,
        cov_type: Literal["classic", "robust", "cluster"] = "classic",
        groups=None,
        max_iter: int | None = None,
    ) -> ChoiceResults:
        data = data.to(device=self.device, dtype=self.dtype)
        compiled = self.compile(data)
        params = compiled.free_initial.clone().detach().requires_grad_(True)
        optimizer = torch.optim.LBFGS(
            [params],
            max_iter=max_iter or self.max_iter,
            tolerance_grad=self.tolerance_grad,
            line_search_fn=self.line_search_fn,
        )
        iterations = {"count": 0}

        def closure():
            optimizer.zero_grad(set_to_none=True)
            loss = -self.loglike(params, data, compiled)
            loss.backward()
            iterations["count"] += 1
            return loss

        optimizer.step(closure)
        final_params = params.detach().clone()
        final_params.requires_grad_(True)
        ll = self.loglike(final_params, data, compiled)
        grad = torch.autograd.grad(ll, final_params, create_graph=False)[0].detach()
        hessian_ll = torch.autograd.functional.hessian(lambda p: self.loglike(p, data, compiled), final_params)
        information = -hessian_ll.detach()
        cov_classic = _safe_pinv(information)
        covariances = {"classic": cov_classic}
        if cov_type in {"robust", "cluster"}:
            scores = self.scores(final_params.detach(), data, compiled)
            meat = scores.T @ scores
            covariances["robust"] = cov_classic @ meat @ cov_classic
            cluster_codes = data.cluster_codes(groups)
            if cluster_codes is not None:
                cluster_meat = _cluster_meat(scores, cluster_codes)
                covariances["cluster"] = cov_classic @ cluster_meat @ cov_classic
            elif cov_type == "cluster":
                raise ValueError("Cluster covariance requested, but no groups were supplied.")

        null_ll = self.null_loglike(data)
        return ChoiceResults(
            model=self,
            data=data,
            params=final_params.detach(),
            param_names=compiled.free_names,
            loglike=float(ll.detach().cpu()),
            null_loglike=float(null_ll.detach().cpu()),
            gradient=grad,
            hessian=information,
            covariances=covariances,
            cov_type=cov_type,
            n_obs=data.n_obs,
            n_params=len(compiled.free_names),
            convergence_status={
                "optimizer": "torch.optim.LBFGS",
                "closure_evaluations": iterations["count"],
                "gradient_norm": float(torch.linalg.vector_norm(grad).detach().cpu()),
            },
        )

    def scores(
        self,
        params: torch.Tensor,
        data: ChoiceDataset,
        compiled: CompiledUtility | None = None,
    ) -> torch.Tensor:
        data = data.to(device=self.device, dtype=self.dtype)
        compiled = compiled or self.compile(data)
        width = compiled.choice_set_width
        if width is not None:
            probabilities = self.predict_proba(data, params, compiled).reshape(data.n_obs, width)
            design = compiled.design.reshape(data.n_obs, width, len(compiled.free_names))
            chosen_local = (data.chosen_row - data.obs_ptr[:-1]).reshape(-1, 1, 1)
            chosen_design = design.gather(1, chosen_local.expand(-1, 1, design.shape[-1])).squeeze(1)
            expected_design = (probabilities.unsqueeze(-1) * design).sum(dim=1)
            return data.weights.unsqueeze(1) * (chosen_design - expected_design)

        rows = []
        for obs in range(data.n_obs):
            p = params.clone().detach().requires_grad_(True)
            value = self.loglike_per_obs(p, data, compiled)[obs]
            grad = torch.autograd.grad(value, p)[0]
            rows.append(grad.detach())
        return torch.stack(rows)

    def predict_proba(
        self,
        data: ChoiceDataset,
        params: torch.Tensor,
        compiled: CompiledUtility | None = None,
    ) -> torch.Tensor:
        data = data.to(device=self.device, dtype=self.dtype)
        compiled = compiled or self.compile(data)
        utility = self.utilities(params.to(device=self.device, dtype=self.dtype), data, compiled)
        width = compiled.choice_set_width
        if width is not None:
            availability = data.availability.reshape(data.n_obs, width)
            if not bool(availability.any(dim=1).all()):
                raise ValueError("Every observation must have at least one available alternative.")
            utility_by_obs = utility.reshape(data.n_obs, width)
            probs = torch.softmax(utility_by_obs.masked_fill(~availability, -torch.inf), dim=1)
            return probs.masked_fill(~availability, 0).reshape(data.n_rows)

        probs = torch.zeros(data.n_rows, dtype=self.dtype, device=self.device)
        for obs in range(data.n_obs):
            start = int(data.obs_ptr[obs])
            end = int(data.obs_ptr[obs + 1])
            mask = data.availability[start:end]
            seg = utility[start:end]
            values = torch.softmax(seg[mask], dim=0)
            local = torch.zeros(end - start, dtype=self.dtype, device=self.device)
            local[mask] = values
            probs[start:end] = local
        return probs

    def predict(self, data: ChoiceDataset, params: torch.Tensor) -> list[str]:
        data = data.to(device=self.device, dtype=self.dtype)
        probs = self.predict_proba(data, params)
        labels = []
        for obs in range(data.n_obs):
            start = int(data.obs_ptr[obs])
            end = int(data.obs_ptr[obs + 1])
            local_idx = int(torch.argmax(probs[start:end]).detach().cpu())
            labels.append(data.alt_names[int(data.alt_id[start + local_idx].detach().cpu())])
        return labels

    def null_loglike(self, data: ChoiceDataset) -> torch.Tensor:
        data = data.to(device=self.device, dtype=self.dtype)
        width = _balanced_width(data)
        if width is not None:
            n_available = data.availability.reshape(data.n_obs, width).sum(dim=1).to(dtype=data.dtype)
            if not bool((n_available > 0).all()):
                raise ValueError("Every observation must have at least one available alternative.")
            return (-data.weights * torch.log(n_available)).sum()

        values = []
        for obs in range(data.n_obs):
            start = int(data.obs_ptr[obs])
            end = int(data.obs_ptr[obs + 1])
            n_available = data.availability[start:end].sum().to(dtype=data.dtype)
            values.append(-data.weights[obs] * torch.log(n_available))
        return torch.stack(values).sum()


def _safe_pinv(matrix: torch.Tensor) -> torch.Tensor:
    return torch.linalg.pinv(matrix, hermitian=True)


def _balanced_width(data: ChoiceDataset) -> int | None:
    widths = data.obs_ptr[1:] - data.obs_ptr[:-1]
    if widths.numel() == 0:
        return None
    first = widths[0]
    if bool(torch.all(widths == first)):
        return int(first.detach().cpu())
    return None


def _cluster_meat(scores: torch.Tensor, cluster_codes: torch.Tensor) -> torch.Tensor:
    n_clusters = int(cluster_codes.max().detach().cpu()) + 1
    accum = []
    for code in range(n_clusters):
        accum.append(scores[cluster_codes == code].sum(dim=0))
    cluster_scores = torch.stack(accum)
    return cluster_scores.T @ cluster_scores
