from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch

from torchdcm.data.ordered_dataset import OrderedChoiceDataset
from torchdcm.models._optimization import (
    TrackedLBFGS,
    lbfgs_convergence_status,
)
from torchdcm.results.result import ChoiceResults
from torchdcm.spec.expressions import Expression, Term
from torchdcm.spec.parameters import Beta


@dataclass(frozen=True)
class CompiledOrderedUtility:
    design: torch.Tensor
    fixed_design: torch.Tensor
    free_names: list[str]
    fixed_names: list[str]
    beta_names: list[str]
    threshold_names: list[str]
    free_initial: torch.Tensor
    fixed_values: torch.Tensor
    threshold_initial: torch.Tensor


class _BaseOrderedModel:
    distribution = ""

    def __init__(
        self,
        latent: Expression | Beta,
        thresholds: list[Beta | float],
        *,
        dtype: torch.dtype = torch.float64,
        device: str | torch.device = "cpu",
        max_iter: int = 200,
        tolerance_grad: float = 1e-7,
        line_search_fn: str | None = "strong_wolfe",
    ) -> None:
        if isinstance(latent, Beta):
            latent = Expression([Term(latent, None, 1.0)])
        if not isinstance(latent, Expression):
            raise TypeError("latent must be a Beta or Expression.")
        if len(thresholds) < 1:
            raise ValueError("At least one threshold is required.")
        self.latent = latent
        self.thresholds = list(thresholds)
        self.dtype = dtype
        self.device = torch.device(device)
        self.max_iter = max_iter
        self.tolerance_grad = tolerance_grad
        self.line_search_fn = line_search_fn
        self._compiled_cache: dict[int, CompiledOrderedUtility] = {}

    def compile(self, data: OrderedChoiceDataset) -> CompiledOrderedUtility:
        data = data.to(device=self.device, dtype=self.dtype)
        cache_key = id(data)
        if cache_key in self._compiled_cache:
            return self._compiled_cache[cache_key]

        params = self.latent.parameters
        free_params = [p for p in params if not p.fixed]
        fixed_params = [p for p in params if p.fixed]
        free_index = {p.name: i for i, p in enumerate(free_params)}
        fixed_index = {p.name: i for i, p in enumerate(fixed_params)}
        design = torch.zeros((data.n_obs, len(free_params)), dtype=self.dtype, device=self.device)
        fixed_design = torch.zeros((data.n_obs, len(fixed_params)), dtype=self.dtype, device=self.device)
        for term in self.latent.terms:
            values = (
                torch.ones(data.n_obs, dtype=self.dtype, device=self.device)
                if term.variable is None
                else data.x[term.variable].to(device=self.device, dtype=self.dtype)
            )
            contribution = term.multiplier * values
            if term.parameter.fixed:
                fixed_design[:, fixed_index[term.parameter.name]] += contribution
            else:
                design[:, free_index[term.parameter.name]] += contribution

        threshold_values = []
        threshold_names = []
        for index, threshold in enumerate(self.thresholds):
            if isinstance(threshold, Beta):
                threshold_values.append(float(threshold.init))
                threshold_names.append(threshold.name)
            else:
                threshold_values.append(float(threshold))
                threshold_names.append(f"TH_{index + 1}")
        if any(b <= a for a, b in zip(threshold_values, threshold_values[1:])):
            # Ordered probabilities are valid only for strictly increasing cut
            # points; the internal transform preserves this ordering thereafter.
            raise ValueError("Initial thresholds must be strictly increasing.")

        compiled = CompiledOrderedUtility(
            design=design,
            fixed_design=fixed_design,
            free_names=[p.name for p in free_params] + threshold_names,
            fixed_names=[p.name for p in fixed_params],
            beta_names=[p.name for p in free_params],
            threshold_names=threshold_names,
            free_initial=torch.cat(
                [
                    torch.as_tensor([p.init for p in free_params], dtype=self.dtype, device=self.device),
                    torch.as_tensor(threshold_values, dtype=self.dtype, device=self.device),
                ]
            ),
            fixed_values=torch.as_tensor([p.init for p in fixed_params], dtype=self.dtype, device=self.device),
            threshold_initial=torch.as_tensor(threshold_values, dtype=self.dtype, device=self.device),
        )
        self._compiled_cache[cache_key] = compiled
        return compiled

    def latent_value(
        self,
        beta: torch.Tensor,
        data: OrderedChoiceDataset,
        compiled: CompiledOrderedUtility | None = None,
    ) -> torch.Tensor:
        data = data.to(device=self.device, dtype=self.dtype)
        compiled = compiled or self.compile(data)
        value = compiled.design @ beta
        if compiled.fixed_values.numel():
            value = value + compiled.fixed_design @ compiled.fixed_values
        return value

    def probabilities(
        self,
        data: OrderedChoiceDataset,
        params: torch.Tensor,
        compiled: CompiledOrderedUtility | None = None,
    ) -> torch.Tensor:
        data = data.to(device=self.device, dtype=self.dtype)
        compiled = compiled or self.compile(data)
        params = params.to(device=self.device, dtype=self.dtype)
        beta, thresholds = self._split_natural_params(params, compiled)
        eta = self.latent_value(beta, data, compiled)
        # Add the fixed -infinity and +infinity boundaries through CDF values
        # 0 and 1, then difference adjacent cumulative probabilities.
        cdf_values = [torch.zeros_like(eta)]
        for threshold in thresholds:
            cdf_values.append(self._cdf(threshold - eta))
        cdf_values.append(torch.ones_like(eta))
        probs = []
        for lower, upper in zip(cdf_values, cdf_values[1:]):
            probs.append(torch.clamp(upper - lower, min=torch.finfo(self.dtype).tiny))
        return torch.stack(probs, dim=1)

    def loglike_per_obs(
        self,
        params: torch.Tensor,
        data: OrderedChoiceDataset,
        compiled: CompiledOrderedUtility | None = None,
    ) -> torch.Tensor:
        probs = self.probabilities(data, params, compiled)
        chosen = probs.gather(1, data.y.reshape(-1, 1)).squeeze(1)
        return data.weights * torch.log(torch.clamp(chosen, min=torch.finfo(self.dtype).tiny))

    def loglike(
        self,
        params: torch.Tensor,
        data: OrderedChoiceDataset,
        compiled: CompiledOrderedUtility | None = None,
    ) -> torch.Tensor:
        return self.loglike_per_obs(params, data, compiled).sum()

    def fit(
        self,
        data: OrderedChoiceDataset,
        *,
        cov_type: Literal["classic"] = "classic",
        max_iter: int | None = None,
    ) -> ChoiceResults:
        if cov_type != "classic":
            raise NotImplementedError("Ordered models currently support classic covariance only.")
        data = data.to(device=self.device, dtype=self.dtype)
        compiled = self.compile(data)
        internal_initial = torch.cat(
            [
                compiled.free_initial[: len(compiled.beta_names)],
                self._threshold_to_internal(compiled.threshold_initial),
            ]
        )
        internal_params = internal_initial.clone().detach().requires_grad_(True)
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
            convergence_status=convergence_status,
        )

    def predict_proba(self, data: OrderedChoiceDataset, params: torch.Tensor) -> torch.Tensor:
        return self.probabilities(data, params)

    def predict(self, data: OrderedChoiceDataset, params: torch.Tensor) -> list[int]:
        probs = self.predict_proba(data, params)
        indices = torch.argmax(probs, dim=1).detach().cpu().numpy()
        return [data.categories[int(index)] for index in indices]

    def null_loglike(self, data: OrderedChoiceDataset) -> torch.Tensor:
        counts = torch.bincount(data.y, weights=data.weights, minlength=len(data.categories))
        probs = torch.clamp(counts / counts.sum(), min=torch.finfo(data.dtype).tiny)
        return (data.weights * torch.log(probs[data.y])).sum()

    def _split_natural_params(
        self,
        params: torch.Tensor,
        compiled: CompiledOrderedUtility,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        n_beta = len(compiled.beta_names)
        return params[:n_beta], params[n_beta:]

    def _threshold_to_internal(self, thresholds: torch.Tensor) -> torch.Tensor:
        if thresholds.numel() == 1:
            return thresholds
        # The first threshold is unrestricted; later coordinates are log gaps.
        return torch.cat([thresholds[:1], torch.log(torch.clamp(thresholds[1:] - thresholds[:-1], min=1e-12))])

    def _internal_to_natural(self, internal: torch.Tensor, compiled: CompiledOrderedUtility) -> torch.Tensor:
        n_beta = len(compiled.beta_names)
        beta = internal[:n_beta]
        threshold_internal = internal[n_beta:]
        if threshold_internal.numel() == 1:
            thresholds = threshold_internal
        else:
            # Positive exponentiated gaps guarantee tau_1 < ... < tau_Q.
            thresholds = torch.cat(
                [
                    threshold_internal[:1],
                    threshold_internal[:1] + torch.cumsum(torch.exp(threshold_internal[1:]), dim=0),
                ]
            )
        return torch.cat([beta, thresholds])

    def _natural_jacobian(self, internal: torch.Tensor, compiled: CompiledOrderedUtility) -> torch.Tensor:
        n_beta = len(compiled.beta_names)
        jac = torch.eye(internal.numel(), dtype=self.dtype, device=self.device)
        n_thresholds = internal.numel() - n_beta
        if n_thresholds <= 1:
            return jac
        threshold_jac = torch.zeros((n_thresholds, n_thresholds), dtype=self.dtype, device=self.device)
        threshold_jac[:, 0] = 1.0
        increments = torch.exp(internal[n_beta + 1 :])
        for row in range(1, n_thresholds):
            threshold_jac[row, 1 : row + 1] = increments[:row]
        jac[n_beta:, n_beta:] = threshold_jac
        return jac

    def _cdf(self, value: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class OrderedLogit(_BaseOrderedModel):
    distribution = "logit"

    def _cdf(self, value: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(value)


class OrderedProbit(_BaseOrderedModel):
    distribution = "probit"

    def _cdf(self, value: torch.Tensor) -> torch.Tensor:
        return 0.5 * (1.0 + torch.erf(value / torch.sqrt(torch.as_tensor(2.0, dtype=self.dtype, device=self.device))))


def _safe_pinv(matrix: torch.Tensor) -> torch.Tensor:
    return torch.linalg.pinv(matrix, hermitian=True)
