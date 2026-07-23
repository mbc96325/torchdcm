from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch

from torchdcm.data.choice_dataset import ChoiceDataset
from torchdcm.models._optimization import (
    TrackedLBFGS,
    lbfgs_convergence_status,
)
from torchdcm.models.mnl import MultinomialLogit
from torchdcm.results.result import ChoiceResults
from torchdcm.spec.parameters import Beta
from torchdcm.spec.utility import UtilitySpec


@dataclass(frozen=True)
class WTPCoefficient:
    """WTP-space coefficient attached to an attribute variable."""

    name: str
    variable: str
    init: float = 0.0
    sigma_init: float = 0.1
    sigma_name: str | None = None
    distribution: Literal["normal", "lognormal", "negative_lognormal"] = "normal"
    fixed: bool = False
    sigma_fixed: bool = False


@dataclass(frozen=True)
class CompiledWTPMixedUtility:
    deterministic_design: torch.Tensor
    deterministic_fixed_design: torch.Tensor
    deterministic_names: list[str]
    deterministic_fixed_names: list[str]
    deterministic_initial: torch.Tensor
    deterministic_fixed_values: torch.Tensor
    free_names: list[str]
    cost_name: str
    cost_initial: torch.Tensor
    cost_is_fixed: bool
    wtp_names: list[str]
    wtp_initial: torch.Tensor
    wtp_is_fixed: torch.Tensor
    sigma_names: list[str]
    sigma_initial: torch.Tensor
    sigma_is_fixed: torch.Tensor
    distributions: list[str]
    chol_offdiag_names: list[str]
    chol_offdiag_initial: torch.Tensor
    cost_variable: str
    wtp_variables: list[str]
    draws: torch.Tensor
    choice_set_width: int | None


class WTPMixedLogit:
    """Mixed logit parameterized in willingness-to-pay space.

    Utilities are evaluated as:

    ``V = deterministic_terms + B_COST * cost + B_COST * sum(WTP_k * x_k)``.
    """

    def __init__(
        self,
        spec: UtilitySpec,
        *,
        cost: Beta,
        cost_variable: str,
        wtp_coefficients: list[WTPCoefficient],
        n_draws: int = 128,
        draws: torch.Tensor | None = None,
        seed: int = 12345,
        antithetic: bool = True,
        panel: bool = True,
        correlated: bool = False,
        dtype: torch.dtype = torch.float64,
        device: str | torch.device = "cpu",
        max_iter: int = 200,
        tolerance_grad: float = 1e-7,
        line_search_fn: str | None = "strong_wolfe",
        sigma_min: float = 0.0,
    ) -> None:
        self.spec = spec
        self.cost = cost
        self.cost_variable = cost_variable
        self.wtp_coefficients = list(wtp_coefficients)
        self.n_draws = n_draws
        self.user_draws = draws
        self.seed = seed
        self.antithetic = antithetic
        self.panel = panel
        self.correlated = correlated
        self.dtype = dtype
        self.device = torch.device(device)
        self.max_iter = max_iter
        self.tolerance_grad = tolerance_grad
        self.line_search_fn = line_search_fn
        self.sigma_min = sigma_min
        self._compiled_cache: dict[int, CompiledWTPMixedUtility] = {}

    def compile(self, data: ChoiceDataset) -> CompiledWTPMixedUtility:
        data = data.to(device=self.device, dtype=self.dtype)
        cache_key = id(data)
        if cache_key in self._compiled_cache:
            return self._compiled_cache[cache_key]
        missing = [name for name in [self.cost_variable, *(coef.variable for coef in self.wtp_coefficients)] if name not in data.x_alt]
        if missing:
            raise ValueError(f"Dataset is missing WTP-space variables: {missing}")
        supported = {"normal", "lognormal", "negative_lognormal"}
        invalid = [coef.distribution for coef in self.wtp_coefficients if coef.distribution not in supported]
        if invalid:
            raise ValueError(f"Unsupported WTP distributions: {invalid}")

        deterministic = MultinomialLogit(self.spec, dtype=self.dtype, device=self.device).compile(data)
        sigma_names = [coef.sigma_name or f"SIGMA_{coef.name}" for coef in self.wtp_coefficients]
        if len(set(sigma_names)) != len(sigma_names):
            raise ValueError("WTP sigma names must be unique.")
        wtp_names = [coef.name for coef in self.wtp_coefficients]
        if len(set(wtp_names)) != len(wtp_names):
            raise ValueError("WTP coefficient names must be unique.")

        free_names = list(deterministic.free_names)
        if not self.cost.fixed:
            free_names.append(self.cost.name)
        free_names.extend(coef.name for coef in self.wtp_coefficients if not coef.fixed)
        free_names.extend(name for name, coef in zip(sigma_names, self.wtp_coefficients) if not coef.sigma_fixed)
        chol_offdiag_names = []
        if self.correlated:
            for row, row_coef in enumerate(self.wtp_coefficients):
                for col in range(row):
                    chol_offdiag_names.append(f"CHOL_{row_coef.name}__{self.wtp_coefficients[col].name}")
        free_names.extend(chol_offdiag_names)

        compiled = CompiledWTPMixedUtility(
            deterministic_design=deterministic.design,
            deterministic_fixed_design=deterministic.fixed_design,
            deterministic_names=deterministic.free_names,
            deterministic_fixed_names=deterministic.fixed_names,
            deterministic_initial=deterministic.free_initial,
            deterministic_fixed_values=deterministic.fixed_values,
            free_names=free_names,
            cost_name=self.cost.name,
            cost_initial=torch.as_tensor(float(self.cost.init), dtype=self.dtype, device=self.device),
            cost_is_fixed=self.cost.fixed,
            wtp_names=wtp_names,
            wtp_initial=torch.as_tensor([coef.init for coef in self.wtp_coefficients], dtype=self.dtype, device=self.device),
            wtp_is_fixed=torch.as_tensor([coef.fixed for coef in self.wtp_coefficients], dtype=torch.bool, device=self.device),
            sigma_names=sigma_names,
            sigma_initial=torch.as_tensor([coef.sigma_init for coef in self.wtp_coefficients], dtype=self.dtype, device=self.device),
            sigma_is_fixed=torch.as_tensor([coef.sigma_fixed for coef in self.wtp_coefficients], dtype=torch.bool, device=self.device),
            distributions=[coef.distribution for coef in self.wtp_coefficients],
            chol_offdiag_names=chol_offdiag_names,
            chol_offdiag_initial=torch.zeros(len(chol_offdiag_names), dtype=self.dtype, device=self.device),
            cost_variable=self.cost_variable,
            wtp_variables=[coef.variable for coef in self.wtp_coefficients],
            draws=self._make_draws(len(self.wtp_coefficients)),
            choice_set_width=deterministic.choice_set_width,
        )
        self._compiled_cache[cache_key] = compiled
        return compiled

    def utilities_by_draw(
        self,
        params: torch.Tensor,
        data: ChoiceDataset,
        compiled: CompiledWTPMixedUtility | None = None,
    ) -> torch.Tensor:
        data = data.to(device=self.device, dtype=self.dtype)
        compiled = compiled or self.compile(data)
        params = params.to(device=self.device, dtype=self.dtype)
        deterministic, cost, wtp, sigmas, chol_offdiag = self._split_params(params, compiled)
        utility = compiled.deterministic_design @ deterministic
        if compiled.deterministic_fixed_values.numel():
            utility = utility + compiled.deterministic_fixed_design @ compiled.deterministic_fixed_values
        utility = utility.unsqueeze(1)
        cost_values = data.x_alt[compiled.cost_variable].to(device=self.device, dtype=self.dtype).unsqueeze(1)
        utility = utility + cost * cost_values
        if compiled.wtp_names:
            drawn_wtp = self._drawn_wtp(wtp, sigmas, chol_offdiag, compiled)
            for index, variable in enumerate(compiled.wtp_variables):
                values = data.x_alt[variable].to(device=self.device, dtype=self.dtype).unsqueeze(1)
                utility = utility + cost * drawn_wtp[:, index].unsqueeze(0) * values
        return utility

    def loglike_per_unit(
        self,
        params: torch.Tensor,
        data: ChoiceDataset,
        compiled: CompiledWTPMixedUtility | None = None,
    ) -> torch.Tensor:
        obs_log_prob = self._log_prob_per_obs_draw(params, data, compiled)
        compiled = compiled or self.compile(data.to(device=self.device, dtype=self.dtype))
        if self.panel and data.obs_to_ind is not None:
            return data.to(device=self.device, dtype=self.dtype).panel_structure().logmeanexp_by_unit(obs_log_prob)
        return torch.logsumexp(obs_log_prob, dim=1) - torch.log(
            torch.as_tensor(compiled.draws.shape[0], dtype=self.dtype, device=self.device)
        )

    def loglike(
        self,
        params: torch.Tensor,
        data: ChoiceDataset,
        compiled: CompiledWTPMixedUtility | None = None,
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
            raise NotImplementedError("WTPMixedLogit currently supports classic covariance only.")
        data = data.to(device=self.device, dtype=self.dtype)
        compiled = self.compile(data)
        internal_params = self._initial_internal(compiled).clone().detach().requires_grad_(True)
        optimizer = TrackedLBFGS(
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
        internal_gradient = torch.autograd.grad(ll, final_internal)[0].detach()
        convergence_status = lbfgs_convergence_status(
            optimizer,
            internal_gradient,
            final_loss=-ll,
            n_obs=data.n_obs,
            closure_evaluations=iterations["count"],
        )
        natural_for_grad = final_natural.detach().clone().requires_grad_(True)
        gradient = torch.autograd.grad(self.loglike(natural_for_grad, data, compiled), natural_for_grad)[0].detach()
        hessian_internal = torch.autograd.functional.hessian(
            lambda p: self.loglike(self._internal_to_natural(p, compiled), data, compiled),
            final_internal,
        )
        cov_internal = torch.linalg.pinv(-hessian_internal.detach(), hermitian=True)
        transform_jac = self._natural_jacobian(final_internal.detach(), compiled)
        cov_classic = transform_jac @ cov_internal @ transform_jac.T
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
                **convergence_status,
                "n_draws": compiled.draws.shape[0],
                "panel": self.panel and data.obs_to_ind is not None,
            },
        )

    def predict_proba(
        self,
        data: ChoiceDataset,
        params: torch.Tensor,
        compiled: CompiledWTPMixedUtility | None = None,
    ) -> torch.Tensor:
        data = data.to(device=self.device, dtype=self.dtype)
        compiled = compiled or self.compile(data)
        probabilities = self._prob_per_obs_alt_draw(params, data, compiled)
        if compiled.choice_set_width is None:
            return probabilities.mean(dim=1)
        return probabilities.mean(dim=2).reshape(data.n_rows)

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
            return (-data.weights * torch.log(n_available)).sum()
        values = []
        for obs in range(data.n_obs):
            start = int(data.obs_ptr[obs])
            end = int(data.obs_ptr[obs + 1])
            values.append(-data.weights[obs] * torch.log(data.availability[start:end].sum().to(dtype=data.dtype)))
        return torch.stack(values).sum()

    def _log_prob_per_obs_draw(
        self,
        params: torch.Tensor,
        data: ChoiceDataset,
        compiled: CompiledWTPMixedUtility | None = None,
    ) -> torch.Tensor:
        data = data.to(device=self.device, dtype=self.dtype)
        compiled = compiled or self.compile(data)
        probabilities = self._prob_per_obs_alt_draw(params, data, compiled)
        width = compiled.choice_set_width
        if width is None:
            rows = []
            for obs in range(data.n_obs):
                chosen = int(data.chosen_row[obs])
                rows.append(torch.log(torch.clamp(probabilities[chosen], min=torch.finfo(self.dtype).tiny)))
            return torch.stack(rows) * data.weights.unsqueeze(1)
        chosen_local = (data.chosen_row - data.obs_ptr[:-1]).reshape(-1, 1, 1)
        chosen_prob = probabilities.gather(1, chosen_local.expand(-1, 1, probabilities.shape[2])).squeeze(1)
        return torch.log(torch.clamp(chosen_prob, min=torch.finfo(self.dtype).tiny)) * data.weights.unsqueeze(1)

    def _prob_per_obs_alt_draw(
        self,
        params: torch.Tensor,
        data: ChoiceDataset,
        compiled: CompiledWTPMixedUtility,
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

    def _drawn_wtp(
        self,
        wtp: torch.Tensor,
        sigmas: torch.Tensor,
        chol_offdiag: torch.Tensor,
        compiled: CompiledWTPMixedUtility,
    ) -> torch.Tensor:
        if not compiled.wtp_names:
            return torch.zeros((compiled.draws.shape[0], 0), dtype=self.dtype, device=self.device)
        latent = wtp.unsqueeze(0) + compiled.draws @ self._cholesky_factor(sigmas, chol_offdiag, compiled).T
        values = []
        for index, distribution in enumerate(compiled.distributions):
            column = latent[:, index]
            if distribution == "normal":
                values.append(column)
            elif distribution == "lognormal":
                values.append(torch.exp(column))
            elif distribution == "negative_lognormal":
                values.append(-torch.exp(column))
            else:
                raise ValueError(f"Unsupported WTP distribution: {distribution}")
        return torch.stack(values, dim=1)

    def _split_params(
        self,
        params: torch.Tensor,
        compiled: CompiledWTPMixedUtility,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        params = params.to(device=self.device, dtype=self.dtype)
        cursor = 0
        deterministic = params[cursor : cursor + len(compiled.deterministic_names)]
        cursor += len(compiled.deterministic_names)
        if compiled.cost_is_fixed:
            cost = compiled.cost_initial
        else:
            cost = params[cursor]
            cursor += 1
        wtp = compiled.wtp_initial.clone()
        free_wtp = ~compiled.wtp_is_fixed
        free_wtp_count = int(free_wtp.sum().detach().cpu())
        if free_wtp_count:
            wtp[free_wtp] = params[cursor : cursor + free_wtp_count]
            cursor += free_wtp_count
        sigmas = compiled.sigma_initial.clone()
        free_sigma = ~compiled.sigma_is_fixed
        free_sigma_count = int(free_sigma.sum().detach().cpu())
        if free_sigma_count:
            sigmas[free_sigma] = params[cursor : cursor + free_sigma_count]
            cursor += free_sigma_count
        chol_offdiag = params[cursor : cursor + len(compiled.chol_offdiag_names)]
        return deterministic, cost, wtp, sigmas, chol_offdiag

    def _initial_internal(self, compiled: CompiledWTPMixedUtility) -> torch.Tensor:
        parts = [compiled.deterministic_initial]
        if not compiled.cost_is_fixed:
            parts.append(compiled.cost_initial.reshape(1))
        if bool((~compiled.wtp_is_fixed).any()):
            parts.append(compiled.wtp_initial[~compiled.wtp_is_fixed])
        if bool((~compiled.sigma_is_fixed).any()):
            parts.append(self._sigma_to_internal(compiled.sigma_initial[~compiled.sigma_is_fixed]))
        parts.append(compiled.chol_offdiag_initial)
        return torch.cat(parts) if parts else torch.zeros(0, dtype=self.dtype, device=self.device)

    def _internal_to_natural(self, internal: torch.Tensor, compiled: CompiledWTPMixedUtility) -> torch.Tensor:
        cursor = 0
        parts = []
        deterministic = internal[cursor : cursor + len(compiled.deterministic_names)]
        parts.append(deterministic)
        cursor += len(compiled.deterministic_names)
        if not compiled.cost_is_fixed:
            parts.append(internal[cursor : cursor + 1])
            cursor += 1
        free_wtp_count = int((~compiled.wtp_is_fixed).sum().detach().cpu())
        if free_wtp_count:
            parts.append(internal[cursor : cursor + free_wtp_count])
            cursor += free_wtp_count
        free_sigma_count = int((~compiled.sigma_is_fixed).sum().detach().cpu())
        if free_sigma_count:
            parts.append(self._internal_to_sigma(internal[cursor : cursor + free_sigma_count]))
            cursor += free_sigma_count
        if compiled.chol_offdiag_names:
            parts.append(internal[cursor : cursor + len(compiled.chol_offdiag_names)])
        return torch.cat(parts) if parts else torch.zeros(0, dtype=self.dtype, device=self.device)

    def _natural_jacobian(self, internal: torch.Tensor, compiled: CompiledWTPMixedUtility) -> torch.Tensor:
        diag = torch.ones_like(internal)
        cursor = len(compiled.deterministic_names) + (0 if compiled.cost_is_fixed else 1)
        cursor += int((~compiled.wtp_is_fixed).sum().detach().cpu())
        free_sigma_count = int((~compiled.sigma_is_fixed).sum().detach().cpu())
        if free_sigma_count:
            diag[cursor : cursor + free_sigma_count] = torch.exp(internal[cursor : cursor + free_sigma_count])
        return torch.diag(diag)

    def _cholesky_factor(
        self,
        sigmas: torch.Tensor,
        chol_offdiag: torch.Tensor,
        compiled: CompiledWTPMixedUtility,
    ) -> torch.Tensor:
        n_wtp = len(compiled.wtp_names)
        if n_wtp == 0:
            return torch.zeros((0, 0), dtype=self.dtype, device=self.device)
        if not self.correlated:
            return torch.diag(sigmas)
        cholesky = torch.zeros((n_wtp, n_wtp), dtype=self.dtype, device=self.device)
        index = torch.arange(n_wtp, device=self.device)
        cholesky[index, index] = sigmas
        cursor = 0
        for row in range(1, n_wtp):
            for col in range(row):
                cholesky[row, col] = chol_offdiag[cursor]
                cursor += 1
        return cholesky

    def _sigma_to_internal(self, sigmas: torch.Tensor) -> torch.Tensor:
        if sigmas.numel() == 0:
            return sigmas
        return torch.log(torch.clamp(sigmas - self.sigma_min, min=1e-12))

    def _internal_to_sigma(self, internal: torch.Tensor) -> torch.Tensor:
        return self.sigma_min + torch.exp(internal)

    def _make_draws(self, n_random: int) -> torch.Tensor:
        if n_random == 0:
            return torch.zeros((self.n_draws, 0), dtype=self.dtype, device=self.device)
        if self.user_draws is not None:
            draws = self.user_draws.to(device=self.device, dtype=self.dtype)
            if draws.ndim != 2 or draws.shape[1] != n_random:
                raise ValueError("draws must have shape (n_draws, n_wtp_coefficients).")
            return draws
        generator = torch.Generator(device="cpu")
        generator.manual_seed(self.seed)
        if self.antithetic:
            half = (self.n_draws + 1) // 2
            base = torch.randn((half, n_random), generator=generator, dtype=self.dtype)
            draws = torch.cat([base, -base], dim=0)[: self.n_draws]
        else:
            draws = torch.randn((self.n_draws, n_random), generator=generator, dtype=self.dtype)
        return draws.to(device=self.device)


def _balanced_width(data: ChoiceDataset) -> int | None:
    widths = data.obs_ptr[1:] - data.obs_ptr[:-1]
    if widths.numel() == 0:
        return None
    first = widths[0]
    if bool(torch.all(widths == first)):
        return int(first.detach().cpu())
    return None
