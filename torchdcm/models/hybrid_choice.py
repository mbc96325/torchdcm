from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
from math import log, pi
from typing import Literal, Mapping

import torch

from torchdcm.data.choice_dataset import ChoiceDataset
from torchdcm.models.mnl import MultinomialLogit
from torchdcm.results.result import ChoiceResults
from torchdcm.spec.parameters import Beta
from torchdcm.spec.utility import UtilitySpec


Scalar = Beta | float | int | None


@dataclass(frozen=True)
class LatentVariable:
    """Normally distributed latent variable with an optional structural mean."""

    name: str
    intercept: Scalar = None
    coefficients: Mapping[str, Scalar] = field(default_factory=dict)
    sigma_init: float = 1.0
    sigma_name: str | None = None
    sigma_fixed: bool = True


@dataclass(frozen=True)
class ChoiceLatentEffect:
    """Alternative-specific utility loading for a latent variable."""

    alternative: str
    latent: str
    coefficient: Scalar


@dataclass(frozen=True)
class ContinuousIndicator:
    """Gaussian measurement equation for one observed indicator."""

    variable: str
    latent: str
    intercept: Scalar = None
    loading: Scalar = None
    sigma_init: float = 1.0
    sigma_name: str | None = None
    sigma_fixed: bool = False


@dataclass(frozen=True)
class _ScalarSpec:
    name: str | None
    init: float
    fixed: bool
    positive: bool = False


@dataclass(frozen=True)
class _CompiledLatentVariable:
    name: str
    intercept: _ScalarSpec
    coefficients: dict[str, _ScalarSpec]
    sigma: _ScalarSpec


@dataclass(frozen=True)
class _CompiledChoiceLatentEffect:
    alt_code: int
    latent_index: int
    coefficient: _ScalarSpec


@dataclass(frozen=True)
class _CompiledIndicator:
    variable: str
    latent_index: int
    intercept: _ScalarSpec
    loading: _ScalarSpec
    sigma: _ScalarSpec


@dataclass(frozen=True)
class CompiledHybridChoice:
    deterministic_design: torch.Tensor
    deterministic_fixed_design: torch.Tensor
    deterministic_names: list[str]
    deterministic_fixed_values: torch.Tensor
    free_names: list[str]
    free_initial: torch.Tensor
    positive_free: torch.Tensor
    latent_variables: list[_CompiledLatentVariable]
    choice_effects: list[_CompiledChoiceLatentEffect]
    indicators: list[_CompiledIndicator]
    draws: torch.Tensor
    row_to_obs: torch.Tensor
    choice_set_width: int | None


class HybridChoiceModel:
    """Hybrid choice model with MNL choice kernel and Gaussian measurement model.

    The implemented likelihood is the joint likelihood of the observed choice and
    continuous indicators, integrated over normally distributed latent variables
    using Monte Carlo draws.
    """

    def __init__(
        self,
        spec: UtilitySpec,
        *,
        latent_variables: list[LatentVariable],
        choice_effects: list[ChoiceLatentEffect] | None = None,
        indicators: list[ContinuousIndicator] | None = None,
        n_draws: int = 128,
        draws: torch.Tensor | None = None,
        seed: int = 12345,
        antithetic: bool = True,
        panel: bool = True,
        dtype: torch.dtype = torch.float64,
        device: str | torch.device = "cpu",
        max_iter: int = 200,
        tolerance_grad: float = 1e-7,
        line_search_fn: str | None = "strong_wolfe",
        sigma_min: float = 1e-9,
    ) -> None:
        if not latent_variables:
            raise ValueError("HybridChoiceModel requires at least one latent variable.")
        self.spec = spec
        self.latent_variables = list(latent_variables)
        self.choice_effects = list(choice_effects or [])
        self.indicators = list(indicators or [])
        self.n_draws = n_draws
        self.user_draws = draws
        self.seed = seed
        self.antithetic = antithetic
        self.panel = panel
        self.dtype = dtype
        self.device = torch.device(device)
        self.max_iter = max_iter
        self.tolerance_grad = tolerance_grad
        self.line_search_fn = line_search_fn
        self.sigma_min = sigma_min
        self._compiled_cache: dict[int, CompiledHybridChoice] = {}

    def compile(self, data: ChoiceDataset) -> CompiledHybridChoice:
        data = data.to(device=self.device, dtype=self.dtype)
        cache_key = id(data)
        if cache_key in self._compiled_cache:
            return self._compiled_cache[cache_key]

        deterministic = MultinomialLogit(self.spec, dtype=self.dtype, device=self.device).compile(data)
        alt_to_code = {name: i for i, name in enumerate(data.alt_names)}
        latent_index = {lv.name: i for i, lv in enumerate(self.latent_variables)}
        if len(latent_index) != len(self.latent_variables):
            raise ValueError("Latent variable names must be unique.")

        registry = _ParamRegistry()
        for name, init in zip(deterministic.free_names, deterministic.free_initial.detach().cpu().tolist()):
            registry.register(_ScalarSpec(name, float(init), fixed=False))
        for name, value in zip(deterministic.fixed_names, deterministic.fixed_values.detach().cpu().tolist()):
            registry.register(_ScalarSpec(name, float(value), fixed=True))

        compiled_lvs: list[_CompiledLatentVariable] = []
        for lv in self.latent_variables:
            missing = sorted(set(lv.coefficients) - set(data.x_obs))
            if missing:
                raise ValueError(f"Dataset is missing structural variables for {lv.name!r}: {missing}")
            intercept = _as_scalar(lv.intercept, f"LV_{lv.name}_INTERCEPT", fixed_zero=True)
            coefficients = {
                variable: _as_scalar(value, f"LV_{lv.name}_{variable}")
                for variable, value in lv.coefficients.items()
            }
            sigma = _as_sigma(lv.sigma_name or f"SIGMA_LV_{lv.name}", lv.sigma_init, lv.sigma_fixed)
            registry.register(intercept)
            for coefficient in coefficients.values():
                registry.register(coefficient)
            registry.register(sigma)
            compiled_lvs.append(
                _CompiledLatentVariable(
                    name=lv.name,
                    intercept=intercept,
                    coefficients=coefficients,
                    sigma=sigma,
                )
            )

        compiled_effects: list[_CompiledChoiceLatentEffect] = []
        for effect in self.choice_effects:
            if effect.alternative not in alt_to_code:
                raise ValueError(f"Unknown alternative in latent effect: {effect.alternative!r}")
            if effect.latent not in latent_index:
                raise ValueError(f"Unknown latent variable in latent effect: {effect.latent!r}")
            coefficient = _as_scalar(effect.coefficient, f"B_{effect.alternative}_{effect.latent}")
            registry.register(coefficient)
            compiled_effects.append(
                _CompiledChoiceLatentEffect(
                    alt_code=alt_to_code[effect.alternative],
                    latent_index=latent_index[effect.latent],
                    coefficient=coefficient,
                )
            )

        compiled_indicators: list[_CompiledIndicator] = []
        for indicator in self.indicators:
            if indicator.variable not in data.x_obs:
                raise ValueError(f"Dataset is missing indicator variable: {indicator.variable!r}")
            if indicator.latent not in latent_index:
                raise ValueError(f"Unknown latent variable in indicator: {indicator.latent!r}")
            intercept = _as_scalar(indicator.intercept, f"MEAS_{indicator.variable}_INTERCEPT", fixed_zero=True)
            loading = _as_scalar(indicator.loading, f"MEAS_{indicator.variable}_{indicator.latent}")
            sigma = _as_sigma(
                indicator.sigma_name or f"SIGMA_MEAS_{indicator.variable}",
                indicator.sigma_init,
                indicator.sigma_fixed,
            )
            registry.register(intercept)
            registry.register(loading)
            registry.register(sigma)
            compiled_indicators.append(
                _CompiledIndicator(
                    variable=indicator.variable,
                    latent_index=latent_index[indicator.latent],
                    intercept=intercept,
                    loading=loading,
                    sigma=sigma,
                )
            )

        compiled = CompiledHybridChoice(
            deterministic_design=deterministic.design,
            deterministic_fixed_design=deterministic.fixed_design,
            deterministic_names=deterministic.free_names,
            deterministic_fixed_values=deterministic.fixed_values,
            free_names=registry.free_names,
            free_initial=torch.as_tensor(registry.free_initial, dtype=self.dtype, device=self.device),
            positive_free=torch.as_tensor(registry.positive_free, dtype=torch.bool, device=self.device),
            latent_variables=compiled_lvs,
            choice_effects=compiled_effects,
            indicators=compiled_indicators,
            draws=self._make_draws(len(compiled_lvs)),
            row_to_obs=_row_to_obs(data),
            choice_set_width=deterministic.choice_set_width,
        )
        self._compiled_cache[cache_key] = compiled
        return compiled

    def latent_values_by_obs_draw(
        self,
        params: torch.Tensor,
        data: ChoiceDataset,
        compiled: CompiledHybridChoice | None = None,
    ) -> torch.Tensor:
        data = data.to(device=self.device, dtype=self.dtype)
        compiled = compiled or self.compile(data)
        means = []
        sigmas = []
        for lv in compiled.latent_variables:
            mean = torch.zeros(data.n_obs, dtype=self.dtype, device=self.device)
            mean = mean + self._scalar_value(lv.intercept, params, compiled)
            for variable, coefficient in lv.coefficients.items():
                mean = mean + self._scalar_value(coefficient, params, compiled) * data.x_obs[variable]
            means.append(mean)
            sigmas.append(torch.clamp(self._scalar_value(lv.sigma, params, compiled), min=self.sigma_min))

        mean_by_obs = torch.stack(means, dim=1)
        sigma = torch.stack(sigmas).reshape(1, 1, -1)
        if self.panel and data.obs_to_ind is not None:
            unit_mean = _first_obs_by_unit(mean_by_obs, data)
            unit_values = unit_mean.unsqueeze(1) + compiled.draws.unsqueeze(0) * sigma
            return unit_values[data.obs_to_ind]
        return mean_by_obs.unsqueeze(1) + compiled.draws.unsqueeze(0) * sigma

    def utilities_by_draw(
        self,
        params: torch.Tensor,
        data: ChoiceDataset,
        compiled: CompiledHybridChoice | None = None,
    ) -> torch.Tensor:
        data = data.to(device=self.device, dtype=self.dtype)
        compiled = compiled or self.compile(data)
        params = params.to(device=self.device, dtype=self.dtype)
        deterministic = params[: len(compiled.deterministic_names)]
        utility = compiled.deterministic_design @ deterministic
        if compiled.deterministic_fixed_values.numel():
            utility = utility + compiled.deterministic_fixed_design @ compiled.deterministic_fixed_values
        utility = utility.unsqueeze(1).expand(data.n_rows, compiled.draws.shape[0]).clone()
        if compiled.choice_effects:
            latent = self.latent_values_by_obs_draw(params, data, compiled)
            for effect in compiled.choice_effects:
                rows = data.alt_id == effect.alt_code
                if bool(rows.any()):
                    utility[rows] = utility[rows] + self._scalar_value(effect.coefficient, params, compiled) * latent[
                        compiled.row_to_obs[rows], :, effect.latent_index
                    ]
        return utility

    def measurement_log_density_by_obs_draw(
        self,
        params: torch.Tensor,
        data: ChoiceDataset,
        compiled: CompiledHybridChoice | None = None,
    ) -> torch.Tensor:
        data = data.to(device=self.device, dtype=self.dtype)
        compiled = compiled or self.compile(data)
        latent = self.latent_values_by_obs_draw(params, data, compiled)
        log_density = torch.zeros((data.n_obs, compiled.draws.shape[0]), dtype=self.dtype, device=self.device)
        log_two_pi = torch.as_tensor(log(2.0 * pi), dtype=self.dtype, device=self.device)
        for indicator in compiled.indicators:
            observed = data.x_obs[indicator.variable].reshape(-1, 1)
            intercept = self._scalar_value(indicator.intercept, params, compiled)
            loading = self._scalar_value(indicator.loading, params, compiled)
            sigma = torch.clamp(self._scalar_value(indicator.sigma, params, compiled), min=self.sigma_min)
            residual = (observed - intercept - loading * latent[:, :, indicator.latent_index]) / sigma
            contribution = -0.5 * log_two_pi - torch.log(sigma) - 0.5 * residual.square()
            log_density = log_density + torch.where(torch.isfinite(observed), contribution, torch.zeros_like(contribution))
        return log_density

    def loglike_per_unit(
        self,
        params: torch.Tensor,
        data: ChoiceDataset,
        compiled: CompiledHybridChoice | None = None,
    ) -> torch.Tensor:
        data = data.to(device=self.device, dtype=self.dtype)
        compiled = compiled or self.compile(data)
        obs_log = self._log_prob_per_obs_draw(params, data, compiled)
        if self.panel and data.obs_to_ind is not None:
            return data.panel_structure().logmeanexp_by_unit(obs_log)
        return torch.logsumexp(obs_log, dim=1) - torch.log(
            torch.as_tensor(compiled.draws.shape[0], dtype=self.dtype, device=self.device)
        )

    def loglike(
        self,
        params: torch.Tensor,
        data: ChoiceDataset,
        compiled: CompiledHybridChoice | None = None,
    ) -> torch.Tensor:
        return self.loglike_per_unit(params, data, compiled).sum()

    def fit(
        self,
        data: ChoiceDataset,
        *,
        cov_type: Literal["classic"] = "classic",
        max_iter: int | None = None,
    ) -> ChoiceResults:
        if cov_type != "classic":
            raise NotImplementedError("HybridChoiceModel currently supports classic covariance only.")
        data = data.to(device=self.device, dtype=self.dtype)
        compiled = self.compile(data)
        internal = self._initial_internal(compiled).clone().detach().requires_grad_(True)
        optimizer = torch.optim.LBFGS(
            [internal],
            max_iter=max_iter or self.max_iter,
            tolerance_grad=self.tolerance_grad,
            line_search_fn=self.line_search_fn,
        )
        iterations = {"count": 0}

        def closure():
            optimizer.zero_grad(set_to_none=True)
            natural = self._internal_to_natural(internal, compiled)
            loss = -self.loglike(natural, data, compiled)
            loss.backward()
            iterations["count"] += 1
            return loss

        optimizer.step(closure)
        final_internal = internal.detach().clone().requires_grad_(True)
        final_natural = self._internal_to_natural(final_internal, compiled)
        ll = self.loglike(final_natural, data, compiled)
        natural_for_grad = final_natural.detach().clone().requires_grad_(True)
        gradient = torch.autograd.grad(self.loglike(natural_for_grad, data, compiled), natural_for_grad)[0].detach()
        hessian_internal = torch.autograd.functional.hessian(
            lambda p: self.loglike(self._internal_to_natural(p, compiled), data, compiled),
            final_internal,
        )
        cov_internal = torch.linalg.pinv(-hessian_internal.detach(), hermitian=True)
        jac = self._natural_jacobian(final_internal.detach(), compiled)
        cov_classic = jac @ cov_internal @ jac.T
        hessian_natural = torch.autograd.functional.hessian(lambda p: self.loglike(p, data, compiled), final_natural.detach())
        return ChoiceResults(
            model=self,
            data=data,
            params=final_natural.detach(),
            param_names=compiled.free_names,
            loglike=float(ll.detach().cpu()),
            null_loglike=float(self.null_loglike(data).detach().cpu()),
            gradient=gradient,
            hessian=-hessian_natural.detach(),
            covariances={"classic": cov_classic},
            cov_type="classic",
            n_obs=data.n_obs,
            n_params=len(compiled.free_names),
            convergence_status={
                "optimizer": "torch.optim.LBFGS",
                "closure_evaluations": iterations["count"],
                "gradient_norm": float(torch.linalg.vector_norm(gradient).detach().cpu()),
                "n_draws": compiled.draws.shape[0],
                "panel": self.panel and data.obs_to_ind is not None,
            },
        )

    def predict_proba(
        self,
        data: ChoiceDataset,
        params: torch.Tensor,
        compiled: CompiledHybridChoice | None = None,
        *,
        condition_on_indicators: bool = False,
    ) -> torch.Tensor:
        data = data.to(device=self.device, dtype=self.dtype)
        compiled = compiled or self.compile(data)
        probabilities = self._prob_per_obs_alt_draw(params, data, compiled)
        if condition_on_indicators and compiled.indicators:
            weights = self._posterior_draw_weights(params, data, compiled)
        else:
            weights = torch.full(
                (data.n_obs, compiled.draws.shape[0]),
                1.0 / compiled.draws.shape[0],
                dtype=self.dtype,
                device=self.device,
            )
        width = compiled.choice_set_width
        if width is not None:
            return (probabilities * weights.unsqueeze(1)).sum(dim=2).reshape(data.n_rows)
        rows = []
        cursor = 0
        for obs in range(data.n_obs):
            start = int(data.obs_ptr[obs])
            end = int(data.obs_ptr[obs + 1])
            rows.append((probabilities[cursor : cursor + (end - start)] * weights[obs].unsqueeze(0)).sum(dim=1))
            cursor += end - start
        return torch.cat(rows)

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
        return MultinomialLogit(self.spec, dtype=self.dtype, device=self.device).null_loglike(data)

    def _log_prob_per_obs_draw(
        self,
        params: torch.Tensor,
        data: ChoiceDataset,
        compiled: CompiledHybridChoice,
    ) -> torch.Tensor:
        probabilities = self._prob_per_obs_alt_draw(params, data, compiled)
        width = compiled.choice_set_width
        if width is None:
            rows = []
            cursor = 0
            for obs in range(data.n_obs):
                start = int(data.obs_ptr[obs])
                end = int(data.obs_ptr[obs + 1])
                chosen_local = int(data.chosen_row[obs]) - start
                rows.append(torch.log(torch.clamp(probabilities[cursor + chosen_local], min=torch.finfo(self.dtype).tiny)))
                cursor += end - start
            choice_log = torch.stack(rows)
        else:
            chosen_local = (data.chosen_row - data.obs_ptr[:-1]).reshape(-1, 1, 1)
            chosen_prob = probabilities.gather(1, chosen_local.expand(-1, 1, probabilities.shape[2])).squeeze(1)
            choice_log = torch.log(torch.clamp(chosen_prob, min=torch.finfo(self.dtype).tiny))
        return data.weights.unsqueeze(1) * (choice_log + self.measurement_log_density_by_obs_draw(params, data, compiled))

    def _prob_per_obs_alt_draw(
        self,
        params: torch.Tensor,
        data: ChoiceDataset,
        compiled: CompiledHybridChoice,
    ) -> torch.Tensor:
        utility = self.utilities_by_draw(params, data, compiled)
        width = compiled.choice_set_width
        if width is not None:
            utility_by_obs = utility.reshape(data.n_obs, width, compiled.draws.shape[0])
            availability = data.availability.reshape(data.n_obs, width).unsqueeze(2)
            probs = torch.softmax(utility_by_obs.masked_fill(~availability, -torch.inf), dim=1)
            return probs.masked_fill(~availability, 0.0)
        rows = []
        for obs in range(data.n_obs):
            start = int(data.obs_ptr[obs])
            end = int(data.obs_ptr[obs + 1])
            mask = data.availability[start:end].unsqueeze(1)
            seg = utility[start:end]
            rows.append(torch.softmax(seg.masked_fill(~mask, -torch.inf), dim=0).masked_fill(~mask, 0.0))
        return torch.cat(rows, dim=0)

    def _posterior_draw_weights(
        self,
        params: torch.Tensor,
        data: ChoiceDataset,
        compiled: CompiledHybridChoice,
    ) -> torch.Tensor:
        measurement = data.weights.unsqueeze(1) * self.measurement_log_density_by_obs_draw(params, data, compiled)
        if self.panel and data.obs_to_ind is not None:
            unit_log = data.panel_structure().sum_by_unit(measurement)
            return torch.softmax(unit_log, dim=1)[data.obs_to_ind]
        return torch.softmax(measurement, dim=1)

    def _scalar_value(self, spec: _ScalarSpec, params: torch.Tensor, compiled: CompiledHybridChoice) -> torch.Tensor:
        if spec.name is None or spec.fixed:
            return torch.as_tensor(spec.init, dtype=self.dtype, device=self.device)
        return params.to(device=self.device, dtype=self.dtype)[compiled.free_names.index(spec.name)]

    def _initial_internal(self, compiled: CompiledHybridChoice) -> torch.Tensor:
        initial = compiled.free_initial.clone()
        if bool(compiled.positive_free.any()):
            initial[compiled.positive_free] = self._sigma_to_internal(initial[compiled.positive_free])
        return initial

    def _internal_to_natural(self, internal: torch.Tensor, compiled: CompiledHybridChoice) -> torch.Tensor:
        natural = internal.clone()
        if bool(compiled.positive_free.any()):
            natural[compiled.positive_free] = self._internal_to_sigma(internal[compiled.positive_free])
        return natural

    def _natural_jacobian(self, internal: torch.Tensor, compiled: CompiledHybridChoice) -> torch.Tensor:
        diag = torch.ones_like(internal)
        if bool(compiled.positive_free.any()):
            diag[compiled.positive_free] = torch.exp(internal[compiled.positive_free])
        return torch.diag(diag)

    def _sigma_to_internal(self, sigma: torch.Tensor) -> torch.Tensor:
        return torch.log(torch.clamp(sigma - self.sigma_min, min=1e-12))

    def _internal_to_sigma(self, internal: torch.Tensor) -> torch.Tensor:
        return self.sigma_min + torch.exp(internal)

    def _make_draws(self, n_latent: int) -> torch.Tensor:
        if self.user_draws is not None:
            draws = self.user_draws.to(device=self.device, dtype=self.dtype)
            if draws.ndim != 2 or draws.shape[1] != n_latent:
                raise ValueError("draws must have shape (n_draws, n_latent_variables).")
            return draws
        generator = torch.Generator(device="cpu")
        generator.manual_seed(self.seed)
        if self.antithetic:
            half = (self.n_draws + 1) // 2
            base = torch.randn((half, n_latent), generator=generator, dtype=self.dtype)
            draws = torch.cat([base, -base], dim=0)[: self.n_draws]
        else:
            draws = torch.randn((self.n_draws, n_latent), generator=generator, dtype=self.dtype)
        return draws.to(device=self.device)


class _ParamRegistry:
    def __init__(self) -> None:
        self._seen: OrderedDict[str, _ScalarSpec] = OrderedDict()

    def register(self, spec: _ScalarSpec) -> None:
        if spec.name is None:
            return
        old = self._seen.get(spec.name)
        if old is not None:
            if old.fixed != spec.fixed or old.positive != spec.positive or abs(old.init - spec.init) > 1e-12:
                raise ValueError(f"Conflicting definitions for parameter {spec.name!r}.")
            return
        self._seen[spec.name] = spec

    @property
    def free_names(self) -> list[str]:
        return [name for name, spec in self._seen.items() if not spec.fixed]

    @property
    def free_initial(self) -> list[float]:
        return [spec.init for spec in self._seen.values() if not spec.fixed]

    @property
    def positive_free(self) -> list[bool]:
        return [spec.positive for spec in self._seen.values() if not spec.fixed]


def _as_scalar(value: Scalar, default_name: str, *, fixed_zero: bool = False) -> _ScalarSpec:
    if value is None:
        if fixed_zero:
            return _ScalarSpec(None, 0.0, fixed=True)
        return _ScalarSpec(default_name, 0.0, fixed=False)
    if isinstance(value, Beta):
        return _ScalarSpec(value.name, float(value.init), value.fixed)
    if isinstance(value, (int, float)):
        return _ScalarSpec(None, float(value), fixed=True)
    raise TypeError(f"Expected a Beta, float, or None, got {type(value)!r}.")


def _as_sigma(name: str, init: float, fixed: bool) -> _ScalarSpec:
    if init <= 0:
        raise ValueError("Sigma initial values must be positive.")
    return _ScalarSpec(name, float(init), fixed=fixed, positive=True)


def _row_to_obs(data: ChoiceDataset) -> torch.Tensor:
    widths = data.obs_ptr[1:] - data.obs_ptr[:-1]
    return torch.repeat_interleave(torch.arange(data.n_obs, dtype=torch.long, device=data.device), widths)


def _first_obs_by_unit(values: torch.Tensor, data: ChoiceDataset) -> torch.Tensor:
    if data.obs_to_ind is None:
        return values
    output = torch.zeros((data.n_individuals, values.shape[1]), dtype=values.dtype, device=values.device)
    seen = torch.zeros(data.n_individuals, dtype=torch.bool, device=values.device)
    for obs in range(data.n_obs):
        unit = int(data.obs_to_ind[obs].detach().cpu())
        if not bool(seen[unit]):
            output[unit] = values[obs]
            seen[unit] = True
    return output
