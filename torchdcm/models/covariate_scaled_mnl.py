from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch

from torchdcm.data.choice_dataset import ChoiceDataset
from torchdcm.models.mnl import MultinomialLogit
from torchdcm.results.result import ChoiceResults
from torchdcm.spec.expressions import Expression, Term
from torchdcm.spec.parameters import Beta
from torchdcm.spec.utility import UtilitySpec


@dataclass(frozen=True)
class CovariateScale:
    """Alternative scale defined by ``scale = exp(log_scale)``.

    Use ``value`` for a fixed scale, or ``log_scale`` for a linear expression
    whose exponent gives the row-specific scale.
    """

    log_scale: Expression | Beta | None = None
    value: float | None = None

    def __post_init__(self) -> None:
        if self.log_scale is not None and self.value is not None:
            raise ValueError("Specify either log_scale or value, not both.")
        if self.value is not None and self.value <= 0:
            raise ValueError("Fixed scale values must be positive.")


@dataclass(frozen=True)
class CompiledCovariateScaledUtility:
    design: torch.Tensor
    fixed_design: torch.Tensor
    scale_design: torch.Tensor
    scale_fixed_design: torch.Tensor
    fixed_log_scale_by_row: torch.Tensor
    free_names: list[str]
    fixed_names: list[str]
    beta_names: list[str]
    scale_names: list[str]
    free_initial: torch.Tensor
    fixed_values: torch.Tensor
    scale_fixed_values: torch.Tensor
    choice_set_width: int | None


class CovariateScaledMultinomialLogit:
    """MNL with row-specific alternative scales ``scale_jn = exp(z_jn gamma)``."""

    def __init__(
        self,
        spec: UtilitySpec,
        scales: dict[str, CovariateScale | Expression | Beta | float],
        *,
        dtype: torch.dtype = torch.float64,
        device: str | torch.device = "cpu",
        max_iter: int = 200,
        tolerance_grad: float = 1e-7,
        line_search_fn: str | None = "strong_wolfe",
    ) -> None:
        self.spec = spec
        self.scales = {name: _as_scale_spec(scale) for name, scale in scales.items()}
        self.dtype = dtype
        self.device = torch.device(device)
        self.max_iter = max_iter
        self.tolerance_grad = tolerance_grad
        self.line_search_fn = line_search_fn
        self._compiled_cache: dict[int, CompiledCovariateScaledUtility] = {}

    @classmethod
    def from_formula(
        cls,
        utilities: dict[str, str],
        scales: dict[str, CovariateScale | Expression | Beta | float],
        **kwargs,
    ) -> "CovariateScaledMultinomialLogit":
        return cls(UtilitySpec.from_formula(utilities), scales, **kwargs)

    def compile(self, data: ChoiceDataset) -> CompiledCovariateScaledUtility:
        data = data.to(device=self.device, dtype=self.dtype)
        cache_key = id(data)
        if cache_key in self._compiled_cache:
            return self._compiled_cache[cache_key]

        missing = sorted(set(data.alt_names) - set(self.scales))
        if missing:
            raise ValueError(f"Scale specification is missing alternatives: {missing}")
        unknown = sorted(set(self.scales) - set(data.alt_names))
        if unknown:
            raise ValueError(f"Scale specification contains unknown alternatives: {unknown}")

        mnl = MultinomialLogit(self.spec, dtype=self.dtype, device=self.device)
        compiled_mnl = mnl.compile(data)
        scale_exprs = [scale.log_scale for scale in self.scales.values() if scale.log_scale is not None]
        scale_params = _collect_unique_parameters(expr.parameters for expr in scale_exprs)
        scale_free = [p for p in scale_params if not p.fixed]
        scale_fixed = [p for p in scale_params if p.fixed]

        beta_name_set = set(compiled_mnl.free_names) | set(compiled_mnl.fixed_names)
        conflicts = sorted(beta_name_set & {p.name for p in scale_params})
        if conflicts:
            raise ValueError(f"Scale parameter names conflict with utility parameters: {conflicts}")

        scale_free_index = {p.name: i for i, p in enumerate(scale_free)}
        scale_fixed_index = {p.name: i for i, p in enumerate(scale_fixed)}
        scale_design = torch.zeros((data.n_rows, len(scale_free)), dtype=self.dtype, device=self.device)
        scale_fixed_design = torch.zeros((data.n_rows, len(scale_fixed)), dtype=self.dtype, device=self.device)
        fixed_log_scale_by_row = torch.zeros(data.n_rows, dtype=self.dtype, device=self.device)
        alt_to_code = {name: i for i, name in enumerate(data.alt_names)}

        for alt_name, scale in self.scales.items():
            rows = data.alt_id == alt_to_code[alt_name]
            if scale.value is not None:
                fixed_log_scale_by_row[rows] = torch.log(
                    torch.as_tensor(float(scale.value), dtype=self.dtype, device=self.device)
                )
                continue
            if scale.log_scale is None:
                continue
            for term in scale.log_scale.terms:
                values = (
                    torch.ones(data.n_rows, dtype=self.dtype, device=self.device)
                    if term.variable is None
                    else data.x_alt[term.variable].to(device=self.device, dtype=self.dtype)
                )
                contribution = term.multiplier * values
                if term.parameter.fixed:
                    scale_fixed_design[rows, scale_fixed_index[term.parameter.name]] += contribution[rows]
                else:
                    scale_design[rows, scale_free_index[term.parameter.name]] += contribution[rows]

        compiled = CompiledCovariateScaledUtility(
            design=compiled_mnl.design,
            fixed_design=compiled_mnl.fixed_design,
            scale_design=scale_design,
            scale_fixed_design=scale_fixed_design,
            fixed_log_scale_by_row=fixed_log_scale_by_row,
            free_names=[*compiled_mnl.free_names, *[p.name for p in scale_free]],
            fixed_names=[*compiled_mnl.fixed_names, *[p.name for p in scale_fixed]],
            beta_names=compiled_mnl.free_names,
            scale_names=[p.name for p in scale_free],
            free_initial=torch.cat(
                [
                    compiled_mnl.free_initial,
                    torch.as_tensor([p.init for p in scale_free], dtype=self.dtype, device=self.device),
                ]
            ),
            fixed_values=compiled_mnl.fixed_values,
            scale_fixed_values=torch.as_tensor([p.init for p in scale_fixed], dtype=self.dtype, device=self.device),
            choice_set_width=compiled_mnl.choice_set_width,
        )
        self._compiled_cache[cache_key] = compiled
        return compiled

    def utilities(
        self,
        beta: torch.Tensor,
        data: ChoiceDataset,
        compiled: CompiledCovariateScaledUtility | None = None,
    ) -> torch.Tensor:
        data = data.to(device=self.device, dtype=self.dtype)
        compiled = compiled or self.compile(data)
        utility = compiled.design @ beta
        if compiled.fixed_values.numel():
            utility = utility + compiled.fixed_design @ compiled.fixed_values
        return utility

    def row_scales(
        self,
        params: torch.Tensor,
        data: ChoiceDataset,
        compiled: CompiledCovariateScaledUtility | None = None,
    ) -> torch.Tensor:
        compiled = compiled or self.compile(data)
        n_beta = len(compiled.beta_names)
        scale_params = params[n_beta:].to(device=self.device, dtype=self.dtype)
        log_scale = compiled.fixed_log_scale_by_row.clone()
        if scale_params.numel():
            log_scale = log_scale + compiled.scale_design @ scale_params
        if compiled.scale_fixed_values.numel():
            log_scale = log_scale + compiled.scale_fixed_design @ compiled.scale_fixed_values
        return torch.exp(log_scale)

    def scaled_utilities(
        self,
        params: torch.Tensor,
        data: ChoiceDataset,
        compiled: CompiledCovariateScaledUtility | None = None,
    ) -> torch.Tensor:
        compiled = compiled or self.compile(data)
        beta = params[: len(compiled.beta_names)].to(device=self.device, dtype=self.dtype)
        return self.utilities(beta, data, compiled) / self.row_scales(params, data, compiled)

    def loglike_per_obs(
        self,
        params: torch.Tensor,
        data: ChoiceDataset,
        compiled: CompiledCovariateScaledUtility | None = None,
    ) -> torch.Tensor:
        data = data.to(device=self.device, dtype=self.dtype)
        compiled = compiled or self.compile(data)
        scaled_utility = self.scaled_utilities(params.to(device=self.device, dtype=self.dtype), data, compiled)
        width = compiled.choice_set_width
        if width is not None:
            utility_by_obs = scaled_utility.reshape(data.n_obs, width)
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
            mask = data.availability[start:end]
            if not bool(mask.any()):
                raise ValueError("Every observation must have at least one available alternative.")
            chosen = int(data.chosen_row[obs])
            log_denom = torch.logsumexp(scaled_utility[start:end][mask], dim=0)
            parts.append(data.weights[obs] * (scaled_utility[chosen] - log_denom))
        return torch.stack(parts)

    def loglike(
        self,
        params: torch.Tensor,
        data: ChoiceDataset,
        compiled: CompiledCovariateScaledUtility | None = None,
    ) -> torch.Tensor:
        return self.loglike_per_obs(params, data, compiled).sum()

    def fit(
        self,
        data: ChoiceDataset,
        *,
        cov_type: Literal["classic"] = "classic",
        max_iter: int | None = None,
    ) -> ChoiceResults:
        if cov_type != "classic":
            raise NotImplementedError("CovariateScaledMultinomialLogit currently supports classic covariance only.")
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
        final_params = params.detach().clone().requires_grad_(True)
        ll = self.loglike(final_params, data, compiled)
        gradient = torch.autograd.grad(ll, final_params)[0].detach()
        hessian_ll = torch.autograd.functional.hessian(lambda p: self.loglike(p, data, compiled), final_params)
        information = -hessian_ll.detach()
        cov_classic = _safe_pinv(information)
        return ChoiceResults(
            model=self,
            data=data,
            params=final_params.detach(),
            param_names=compiled.free_names,
            loglike=float(ll.detach().cpu()),
            null_loglike=float(self.null_loglike(data).detach().cpu()),
            gradient=gradient,
            hessian=information,
            covariances={"classic": cov_classic},
            cov_type="classic",
            n_obs=data.n_obs,
            n_params=len(compiled.free_names),
            convergence_status={
                "optimizer": "torch.optim.LBFGS",
                "closure_evaluations": iterations["count"],
                "gradient_norm": float(torch.linalg.vector_norm(gradient).detach().cpu()),
            },
        )

    def predict_proba(
        self,
        data: ChoiceDataset,
        params: torch.Tensor,
        compiled: CompiledCovariateScaledUtility | None = None,
    ) -> torch.Tensor:
        data = data.to(device=self.device, dtype=self.dtype)
        compiled = compiled or self.compile(data)
        scaled_utility = self.scaled_utilities(params, data, compiled)
        width = compiled.choice_set_width
        if width is not None:
            availability = data.availability.reshape(data.n_obs, width)
            if not bool(availability.any(dim=1).all()):
                raise ValueError("Every observation must have at least one available alternative.")
            utility_by_obs = scaled_utility.reshape(data.n_obs, width)
            probs = torch.softmax(utility_by_obs.masked_fill(~availability, -torch.inf), dim=1)
            return probs.masked_fill(~availability, 0.0).reshape(data.n_rows)

        probs = torch.zeros(data.n_rows, dtype=self.dtype, device=self.device)
        for obs in range(data.n_obs):
            start = int(data.obs_ptr[obs])
            end = int(data.obs_ptr[obs + 1])
            mask = data.availability[start:end]
            values = torch.softmax(scaled_utility[start:end][mask], dim=0)
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


def _as_scale_spec(value: CovariateScale | Expression | Beta | float) -> CovariateScale:
    if isinstance(value, CovariateScale):
        return value
    if isinstance(value, Expression):
        return CovariateScale(log_scale=value)
    if isinstance(value, Beta):
        return CovariateScale(log_scale=Expression([Term(value, None, 1.0)]))
    if isinstance(value, (int, float)):
        return CovariateScale(value=float(value))
    raise TypeError(f"Cannot convert {type(value)!r} to a scale specification.")


def _collect_unique_parameters(groups) -> list[Beta]:
    params: dict[str, Beta] = {}
    for group in groups:
        for param in group:
            old = params.get(param.name)
            if old is not None and old != param:
                raise ValueError(f"Conflicting definitions for parameter {param.name!r}.")
            params[param.name] = param
    return list(params.values())


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
