from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch

from torchdcm.data.choice_dataset import ChoiceDataset
from torchdcm.models.mnl import MultinomialLogit
from torchdcm.results.result import ChoiceResults
from torchdcm.spec.utility import UtilitySpec


@dataclass(frozen=True)
class RandomCoefficient:
    """Normal random coefficient specification.

    The mean is the corresponding ``Beta`` in the utility specification. The
    standard deviation is represented by ``sigma_name`` and constrained to be
    non-negative during estimation unless it is fixed.
    """

    name: str
    sigma_init: float = 0.1
    sigma_name: str | None = None
    fixed: bool = False


@dataclass(frozen=True)
class CompiledMixedUtility:
    design: torch.Tensor
    free_names: list[str]
    beta_names: list[str]
    sigma_names: list[str]
    free_initial: torch.Tensor
    fixed_values: torch.Tensor
    fixed_design: torch.Tensor
    random_beta_indices: torch.Tensor
    sigma_initial: torch.Tensor
    sigma_fixed: torch.Tensor
    sigma_is_fixed: torch.Tensor
    draws: torch.Tensor
    choice_set_width: int | None


class MixedLogit:
    """Mixed logit with normal random coefficients and simulated likelihood."""

    def __init__(
        self,
        spec: UtilitySpec,
        random_coefficients: list[RandomCoefficient] | dict[str, RandomCoefficient | float],
        *,
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
        sigma_min: float = 0.0,
    ) -> None:
        self.spec = spec
        if isinstance(random_coefficients, dict):
            self.random_coefficients = [
                value if isinstance(value, RandomCoefficient) else RandomCoefficient(name=name, sigma_init=float(value))
                for name, value in random_coefficients.items()
            ]
        else:
            self.random_coefficients = list(random_coefficients)
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
        self._compiled_cache: dict[int, CompiledMixedUtility] = {}

    @classmethod
    def from_formula(
        cls,
        utilities: dict[str, str],
        random_coefficients: list[RandomCoefficient] | dict[str, RandomCoefficient | float],
        **kwargs,
    ) -> "MixedLogit":
        return cls(UtilitySpec.from_formula(utilities), random_coefficients, **kwargs)

    def compile(self, data: ChoiceDataset) -> CompiledMixedUtility:
        data = data.to(device=self.device, dtype=self.dtype)
        cache_key = id(data)
        if cache_key in self._compiled_cache:
            return self._compiled_cache[cache_key]

        mnl = MultinomialLogit(self.spec, dtype=self.dtype, device=self.device)
        compiled_mnl = mnl.compile(data)
        beta_index = {name: i for i, name in enumerate(compiled_mnl.free_names)}
        missing = [rc.name for rc in self.random_coefficients if rc.name not in beta_index]
        if missing:
            raise ValueError(f"Random coefficients must refer to free utility parameters. Missing: {missing}")

        sigma_names = [rc.sigma_name or f"SIGMA_{rc.name}" for rc in self.random_coefficients]
        if len(set(sigma_names)) != len(sigma_names):
            raise ValueError("Random coefficient sigma names must be unique.")
        sigma_initial = torch.as_tensor([rc.sigma_init for rc in self.random_coefficients], dtype=self.dtype, device=self.device)
        if bool((sigma_initial < 0).any()):
            raise ValueError("Random coefficient sigma initial values must be non-negative.")
        sigma_fixed = sigma_initial.clone()
        sigma_is_fixed = torch.as_tensor([rc.fixed for rc in self.random_coefficients], dtype=torch.bool, device=self.device)
        random_beta_indices = torch.as_tensor(
            [beta_index[rc.name] for rc in self.random_coefficients],
            dtype=torch.long,
            device=self.device,
        )
        draws = self._make_draws(len(self.random_coefficients))
        free_sigma_names = [name for name, fixed in zip(sigma_names, sigma_is_fixed) if not bool(fixed)]

        compiled = CompiledMixedUtility(
            design=compiled_mnl.design,
            free_names=[*compiled_mnl.free_names, *free_sigma_names],
            beta_names=compiled_mnl.free_names,
            sigma_names=sigma_names,
            free_initial=compiled_mnl.free_initial,
            fixed_values=compiled_mnl.fixed_values,
            fixed_design=compiled_mnl.fixed_design,
            random_beta_indices=random_beta_indices,
            sigma_initial=sigma_initial,
            sigma_fixed=sigma_fixed,
            sigma_is_fixed=sigma_is_fixed,
            draws=draws,
            choice_set_width=compiled_mnl.choice_set_width,
        )
        self._compiled_cache[cache_key] = compiled
        return compiled

    def loglike_per_unit(
        self,
        params: torch.Tensor,
        data: ChoiceDataset,
        compiled: CompiledMixedUtility | None = None,
    ) -> torch.Tensor:
        data = data.to(device=self.device, dtype=self.dtype)
        compiled = compiled or self.compile(data)
        obs_log_prob = self._log_prob_per_obs_draw(params, data, compiled)
        if self.panel and data.obs_to_ind is not None:
            n_individuals = int(data.obs_to_ind.max().detach().cpu()) + 1
            panel_log_prob = torch.zeros((n_individuals, compiled.draws.shape[0]), dtype=self.dtype, device=self.device)
            panel_log_prob.index_add_(0, data.obs_to_ind, obs_log_prob)
            return torch.logsumexp(panel_log_prob, dim=1) - torch.log(
                torch.as_tensor(compiled.draws.shape[0], dtype=self.dtype, device=self.device)
            )
        return torch.logsumexp(obs_log_prob, dim=1) - torch.log(
            torch.as_tensor(compiled.draws.shape[0], dtype=self.dtype, device=self.device)
        )

    def loglike(self, params: torch.Tensor, data: ChoiceDataset, compiled: CompiledMixedUtility | None = None) -> torch.Tensor:
        return self.loglike_per_unit(params, data, compiled).sum()

    def fit(
        self,
        data: ChoiceDataset,
        *,
        cov_type: Literal["classic"] = "classic",
        max_iter: int | None = None,
    ) -> ChoiceResults:
        if cov_type != "classic":
            raise NotImplementedError("MixedLogit currently supports classic covariance only.")
        data = data.to(device=self.device, dtype=self.dtype)
        compiled = self.compile(data)
        internal_initial = torch.cat(
            [
                compiled.free_initial,
                self._sigma_to_internal(compiled.sigma_initial[~compiled.sigma_is_fixed]),
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
        final_internal = internal_params.detach().clone()
        final_internal.requires_grad_(True)
        final_natural = self._internal_to_natural(final_internal, compiled)
        ll = self.loglike(final_natural, data, compiled)
        natural_for_grad = final_natural.detach().clone().requires_grad_(True)
        gradient = torch.autograd.grad(self.loglike(natural_for_grad, data, compiled), natural_for_grad)[0].detach()
        hessian_internal = torch.autograd.functional.hessian(
            lambda p: self.loglike(self._internal_to_natural(p, compiled), data, compiled),
            final_internal,
        )
        information_internal = -hessian_internal.detach()
        cov_internal = _safe_pinv(information_internal)
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
                "n_draws": compiled.draws.shape[0],
                "panel": self.panel and data.obs_to_ind is not None,
            },
        )

    def predict_proba(
        self,
        data: ChoiceDataset,
        params: torch.Tensor,
        compiled: CompiledMixedUtility | None = None,
    ) -> torch.Tensor:
        data = data.to(device=self.device, dtype=self.dtype)
        compiled = compiled or self.compile(data)
        probabilities = self._prob_per_obs_alt_draw(params.to(device=self.device, dtype=self.dtype), data, compiled)
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

    def _drawn_betas(self, params: torch.Tensor, compiled: CompiledMixedUtility) -> torch.Tensor:
        means, sigmas = self._split_natural_params(params, compiled)
        betas = means.unsqueeze(0).expand(compiled.draws.shape[0], -1).clone()
        if compiled.random_beta_indices.numel():
            betas[:, compiled.random_beta_indices] = (
                means[compiled.random_beta_indices].unsqueeze(0) + compiled.draws * sigmas.unsqueeze(0)
            )
        return betas

    def _log_prob_per_obs_draw(
        self,
        params: torch.Tensor,
        data: ChoiceDataset,
        compiled: CompiledMixedUtility,
    ) -> torch.Tensor:
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
        compiled: CompiledMixedUtility,
    ) -> torch.Tensor:
        betas = self._drawn_betas(params, compiled)
        utility = compiled.design @ betas.T
        if compiled.fixed_values.numel():
            utility = utility + (compiled.fixed_design @ compiled.fixed_values).unsqueeze(1)
        width = compiled.choice_set_width
        if width is not None:
            utility_by_obs = utility.reshape(data.n_obs, width, compiled.draws.shape[0])
            availability = data.availability.reshape(data.n_obs, width).unsqueeze(2)
            if not bool(availability.any(dim=1).all()):
                raise ValueError("Every observation must have at least one available alternative.")
            probs = torch.softmax(utility_by_obs.masked_fill(~availability, -torch.inf), dim=1)
            return probs.masked_fill(~availability, 0.0)

        rows = []
        for obs in range(data.n_obs):
            start = int(data.obs_ptr[obs])
            end = int(data.obs_ptr[obs + 1])
            mask = data.availability[start:end].unsqueeze(1)
            seg = utility[start:end]
            probs = torch.softmax(seg.masked_fill(~mask, -torch.inf), dim=0).masked_fill(~mask, 0.0)
            rows.append(probs)
        return torch.cat(rows, dim=0)

    def _split_natural_params(
        self,
        params: torch.Tensor,
        compiled: CompiledMixedUtility,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        n_beta = len(compiled.beta_names)
        means = params[:n_beta]
        sigmas = compiled.sigma_fixed.clone()
        free_count = int((~compiled.sigma_is_fixed).sum().detach().cpu())
        if free_count:
            sigmas[~compiled.sigma_is_fixed] = params[n_beta : n_beta + free_count]
        return means, sigmas

    def _internal_to_natural(self, internal: torch.Tensor, compiled: CompiledMixedUtility) -> torch.Tensor:
        n_beta = len(compiled.beta_names)
        means = internal[:n_beta]
        sigmas = compiled.sigma_fixed.clone()
        free_count = int((~compiled.sigma_is_fixed).sum().detach().cpu())
        if free_count:
            sigmas[~compiled.sigma_is_fixed] = self._internal_to_sigma(internal[n_beta : n_beta + free_count])
        return torch.cat([means, sigmas[~compiled.sigma_is_fixed]])

    def _natural_jacobian(self, internal: torch.Tensor, compiled: CompiledMixedUtility) -> torch.Tensor:
        diag = torch.ones_like(internal)
        n_beta = len(compiled.beta_names)
        if internal.numel() > n_beta:
            diag[n_beta:] = torch.exp(internal[n_beta:])
        return torch.diag(diag)

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
                raise ValueError("draws must have shape (n_draws, n_random_coefficients).")
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


def _safe_pinv(matrix: torch.Tensor) -> torch.Tensor:
    return torch.linalg.pinv(matrix, hermitian=True)
