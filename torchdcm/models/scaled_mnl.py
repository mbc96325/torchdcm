from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch

from torchdcm.data.choice_dataset import ChoiceDataset
from torchdcm.models.mnl import MultinomialLogit
from torchdcm.results.result import ChoiceResults
from torchdcm.spec.utility import UtilitySpec


@dataclass(frozen=True)
class AlternativeScale:
    """Alternative-specific positive utility scale."""

    init: float = 1.0
    fixed: bool = False
    name: str | None = None


@dataclass(frozen=True)
class CompiledScaledUtility:
    design: torch.Tensor
    free_names: list[str]
    fixed_names: list[str]
    beta_names: list[str]
    scale_names: list[str]
    free_initial: torch.Tensor
    fixed_values: torch.Tensor
    fixed_design: torch.Tensor
    scale_initial: torch.Tensor
    scale_fixed: torch.Tensor
    scale_is_fixed: torch.Tensor
    scale_by_row_index: torch.Tensor
    choice_set_width: int | None


class ScaledMultinomialLogit:
    """MNL with alternative-specific utility scale parameters."""

    def __init__(
        self,
        spec: UtilitySpec,
        scales: dict[str, AlternativeScale | float],
        *,
        dtype: torch.dtype = torch.float64,
        device: str | torch.device = "cpu",
        max_iter: int = 200,
        tolerance_grad: float = 1e-7,
        line_search_fn: str | None = "strong_wolfe",
        scale_min: float = 1e-6,
    ) -> None:
        self.spec = spec
        self.scales = {
            name: scale if isinstance(scale, AlternativeScale) else AlternativeScale(init=float(scale))
            for name, scale in scales.items()
        }
        self.dtype = dtype
        self.device = torch.device(device)
        self.max_iter = max_iter
        self.tolerance_grad = tolerance_grad
        self.line_search_fn = line_search_fn
        self.scale_min = scale_min
        self._compiled_cache: dict[int, CompiledScaledUtility] = {}

    @classmethod
    def from_formula(
        cls,
        utilities: dict[str, str],
        scales: dict[str, AlternativeScale | float],
        **kwargs,
    ) -> "ScaledMultinomialLogit":
        return cls(UtilitySpec.from_formula(utilities), scales, **kwargs)

    def compile(self, data: ChoiceDataset) -> CompiledScaledUtility:
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

        scale_initial = []
        scale_fixed = []
        scale_is_fixed = []
        scale_names = []
        for alt in data.alt_names:
            scale = self.scales[alt]
            init = float(scale.init)
            if init <= self.scale_min:
                raise ValueError(f"Scale initial value for {alt!r} must be greater than {self.scale_min}.")
            scale_initial.append(init)
            scale_fixed.append(init)
            scale_is_fixed.append(scale.fixed)
            scale_names.append(scale.name or f"SCALE_{alt.upper()}")
        if not any(scale_is_fixed):
            raise ValueError("At least one alternative scale must be fixed for identification.")

        free_scale_names = [name for name, fixed in zip(scale_names, scale_is_fixed) if not fixed]
        compiled = CompiledScaledUtility(
            design=compiled_mnl.design,
            free_names=[*compiled_mnl.free_names, *free_scale_names],
            fixed_names=compiled_mnl.fixed_names,
            beta_names=compiled_mnl.free_names,
            scale_names=scale_names,
            free_initial=compiled_mnl.free_initial,
            fixed_values=compiled_mnl.fixed_values,
            fixed_design=compiled_mnl.fixed_design,
            scale_initial=torch.as_tensor(scale_initial, dtype=self.dtype, device=self.device),
            scale_fixed=torch.as_tensor(scale_fixed, dtype=self.dtype, device=self.device),
            scale_is_fixed=torch.as_tensor(scale_is_fixed, dtype=torch.bool, device=self.device),
            scale_by_row_index=data.alt_id.to(device=self.device),
            choice_set_width=compiled_mnl.choice_set_width,
        )
        self._compiled_cache[cache_key] = compiled
        return compiled

    def utilities(
        self,
        beta: torch.Tensor,
        data: ChoiceDataset,
        compiled: CompiledScaledUtility | None = None,
    ) -> torch.Tensor:
        data = data.to(device=self.device, dtype=self.dtype)
        compiled = compiled or self.compile(data)
        utility = compiled.design @ beta
        if compiled.fixed_values.numel():
            utility = utility + compiled.fixed_design @ compiled.fixed_values
        return utility

    def scaled_utilities(
        self,
        params: torch.Tensor,
        data: ChoiceDataset,
        compiled: CompiledScaledUtility | None = None,
    ) -> torch.Tensor:
        compiled = compiled or self.compile(data)
        beta, scales = self._split_natural_params(params.to(device=self.device, dtype=self.dtype), compiled)
        utility = self.utilities(beta, data, compiled)
        return utility / scales[compiled.scale_by_row_index]

    def loglike_per_obs(
        self,
        params: torch.Tensor,
        data: ChoiceDataset,
        compiled: CompiledScaledUtility | None = None,
    ) -> torch.Tensor:
        data = data.to(device=self.device, dtype=self.dtype)
        compiled = compiled or self.compile(data)
        scaled_utility = self.scaled_utilities(params, data, compiled)
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
        compiled: CompiledScaledUtility | None = None,
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
            raise NotImplementedError("ScaledMultinomialLogit currently supports classic covariance only.")
        data = data.to(device=self.device, dtype=self.dtype)
        compiled = self.compile(data)
        internal_initial = torch.cat(
            [
                compiled.free_initial,
                self._scale_to_internal(compiled.scale_initial[~compiled.scale_is_fixed]),
            ]
        )
        internal_params = internal_initial.clone().detach().requires_grad_(True)
        optimizer = torch.optim.LBFGS(
            [internal_params],
            max_iter=max_iter or self.max_iter,
            tolerance_grad=self.tolerance_grad,
            line_search_fn=self.line_search_fn,
        )
        iterations = {"count": 0}

        def closure():
            optimizer.zero_grad(set_to_none=True)
            natural = self._internal_to_natural(internal_params, compiled)
            loss = -self.loglike(natural, data, compiled)
            loss.backward()
            iterations["count"] += 1
            return loss

        optimizer.step(closure)
        final_internal = internal_params.detach().clone().requires_grad_(True)
        final_natural = self._internal_to_natural(final_internal, compiled)
        ll = self.loglike(final_natural, data, compiled)
        natural_for_grad = final_natural.detach().clone().requires_grad_(True)
        gradient = torch.autograd.grad(self.loglike(natural_for_grad, data, compiled), natural_for_grad)[0].detach()
        hessian_internal = torch.autograd.functional.hessian(
            lambda p: self.loglike(self._internal_to_natural(p, compiled), data, compiled),
            final_internal,
        )
        cov_internal = _safe_pinv(-hessian_internal.detach())
        transform_jac = self._natural_jacobian(final_internal.detach(), compiled)
        cov_classic = transform_jac @ cov_internal @ transform_jac.T
        hessian_natural = torch.autograd.functional.hessian(lambda p: self.loglike(p, data, compiled), final_natural.detach())
        information = -hessian_natural.detach()
        return ChoiceResults(
            model=self,
            data=data,
            params=final_natural.detach(),
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
        compiled: CompiledScaledUtility | None = None,
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

    def _split_natural_params(
        self,
        params: torch.Tensor,
        compiled: CompiledScaledUtility,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        n_beta = len(compiled.beta_names)
        beta = params[:n_beta]
        scales = compiled.scale_fixed.clone()
        free_count = int((~compiled.scale_is_fixed).sum().detach().cpu())
        if free_count:
            scales[~compiled.scale_is_fixed] = params[n_beta : n_beta + free_count]
        return beta, scales

    def _internal_to_natural(self, internal: torch.Tensor, compiled: CompiledScaledUtility) -> torch.Tensor:
        n_beta = len(compiled.beta_names)
        beta = internal[:n_beta]
        scales = compiled.scale_fixed.clone()
        free_count = int((~compiled.scale_is_fixed).sum().detach().cpu())
        if free_count:
            scales[~compiled.scale_is_fixed] = self._internal_to_scale(internal[n_beta : n_beta + free_count])
        return torch.cat([beta, scales[~compiled.scale_is_fixed]])

    def _natural_jacobian(self, internal: torch.Tensor, compiled: CompiledScaledUtility) -> torch.Tensor:
        diag = torch.ones_like(internal)
        n_beta = len(compiled.beta_names)
        if internal.numel() > n_beta:
            diag[n_beta:] = torch.exp(internal[n_beta:])
        return torch.diag(diag)

    def _scale_to_internal(self, scales: torch.Tensor) -> torch.Tensor:
        if scales.numel() == 0:
            return scales
        return torch.log(torch.clamp(scales - self.scale_min, min=1e-12))

    def _internal_to_scale(self, internal: torch.Tensor) -> torch.Tensor:
        return self.scale_min + torch.exp(internal)


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
