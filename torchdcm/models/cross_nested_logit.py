from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch

from torchdcm.data.choice_dataset import ChoiceDataset
from torchdcm.models._optimization import (
    TrackedLBFGS,
    lbfgs_convergence_status,
)
from torchdcm.results.result import ChoiceResults
from torchdcm.spec.utility import UtilitySpec


@dataclass(frozen=True)
class CrossNest:
    """A cross-nested logit nest with fixed allocation weights."""

    allocations: dict[str, float]
    init: float = 0.8
    fixed: bool = False
    name: str | None = None


@dataclass(frozen=True)
class CompiledCrossNestedUtility:
    design: torch.Tensor
    free_names: list[str]
    fixed_names: list[str]
    beta_names: list[str]
    free_initial: torch.Tensor
    fixed_values: torch.Tensor
    fixed_design: torch.Tensor
    alpha_by_row: torch.Tensor
    nest_names: list[str]
    lambda_names: list[str]
    lambda_initial: torch.Tensor
    lambda_fixed: torch.Tensor
    lambda_is_fixed: torch.Tensor
    choice_set_width: int | None


class CrossNestedLogit:
    """Cross-nested logit with fixed allocation weights and estimated lambdas."""

    def __init__(
        self,
        spec: UtilitySpec,
        nests: dict[str, CrossNest | dict[str, float]],
        *,
        dtype: torch.dtype = torch.float64,
        device: str | torch.device = "cpu",
        max_iter: int = 200,
        tolerance_grad: float = 1e-7,
        line_search_fn: str | None = "strong_wolfe",
        lambda_min: float = 1e-4,
        allocation_tol: float = 1e-8,
    ) -> None:
        self.spec = spec
        self.nests = {
            name: nest if isinstance(nest, CrossNest) else CrossNest(dict(nest), name=name)
            for name, nest in nests.items()
        }
        if len(self.nests) < 2:
            raise ValueError("CrossNestedLogit requires at least two nests.")
        self.dtype = dtype
        self.device = torch.device(device)
        self.max_iter = max_iter
        self.tolerance_grad = tolerance_grad
        self.line_search_fn = line_search_fn
        self.lambda_min = lambda_min
        self.allocation_tol = allocation_tol
        self._compiled_cache: dict[int, CompiledCrossNestedUtility] = {}

    @classmethod
    def from_formula(
        cls,
        utilities: dict[str, str],
        nests: dict[str, CrossNest | dict[str, float]],
        **kwargs,
    ) -> "CrossNestedLogit":
        return cls(UtilitySpec.from_formula(utilities), nests, **kwargs)

    def compile(self, data: ChoiceDataset) -> CompiledCrossNestedUtility:
        data = data.to(device=self.device, dtype=self.dtype)
        cache_key = id(data)
        if cache_key in self._compiled_cache:
            return self._compiled_cache[cache_key]

        alt_to_code = {name: i for i, name in enumerate(data.alt_names)}
        missing_alts = sorted(set(self.spec.utilities) - set(alt_to_code))
        if missing_alts:
            raise ValueError(f"Specification contains alternatives not in data: {missing_alts}")

        alpha_by_alt = torch.zeros(
            (len(data.alt_names), len(self.nests)),
            dtype=self.dtype,
            device=self.device,
        )
        for nest_index, (nest_name, nest) in enumerate(self.nests.items()):
            for alt, alpha in nest.allocations.items():
                if alt not in alt_to_code:
                    raise ValueError(f"Nest {nest_name!r} contains unknown alternative {alt!r}.")
                if alpha < 0:
                    raise ValueError(f"Allocation for alternative {alt!r} in nest {nest_name!r} must be non-negative.")
                alpha_by_alt[alt_to_code[alt], nest_index] = float(alpha)
        allocation_sums = alpha_by_alt.sum(dim=1)
        if not bool(torch.all(torch.abs(allocation_sums - 1.0) <= self.allocation_tol)):
            bad = [
                data.alt_names[i]
                for i, value in enumerate(allocation_sums.detach().cpu().numpy())
                if abs(float(value) - 1.0) > self.allocation_tol
            ]
            raise ValueError(f"Allocation weights must sum to one for every alternative. Bad alternatives: {bad}")

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

        lambda_initial = []
        lambda_fixed = []
        lambda_is_fixed = []
        lambda_names = []
        for nest_name, nest in self.nests.items():
            init = float(nest.init)
            if not self.lambda_min < init <= 1.0:
                raise ValueError(f"Nest lambda initial value for {nest_name!r} must be in ({self.lambda_min}, 1].")
            lambda_initial.append(init)
            lambda_fixed.append(init)
            lambda_is_fixed.append(nest.fixed)
            lambda_names.append(nest.name or f"LAMBDA_{nest_name.upper()}")

        free_lambda_names = [name for name, fixed in zip(lambda_names, lambda_is_fixed) if not fixed]
        compiled = CompiledCrossNestedUtility(
            design=design,
            free_names=[p.name for p in free_params] + free_lambda_names,
            fixed_names=[p.name for p in fixed_params],
            beta_names=[p.name for p in free_params],
            free_initial=torch.as_tensor([p.init for p in free_params], dtype=self.dtype, device=self.device),
            fixed_values=torch.as_tensor([p.init for p in fixed_params], dtype=self.dtype, device=self.device),
            fixed_design=fixed_design,
            alpha_by_row=alpha_by_alt[data.alt_id],
            nest_names=list(self.nests),
            lambda_names=lambda_names,
            lambda_initial=torch.as_tensor(lambda_initial, dtype=self.dtype, device=self.device),
            lambda_fixed=torch.as_tensor(lambda_fixed, dtype=self.dtype, device=self.device),
            lambda_is_fixed=torch.as_tensor(lambda_is_fixed, dtype=torch.bool, device=self.device),
            choice_set_width=_balanced_width(data),
        )
        self._compiled_cache[cache_key] = compiled
        return compiled

    def utilities(
        self,
        beta: torch.Tensor,
        data: ChoiceDataset,
        compiled: CompiledCrossNestedUtility | None = None,
    ) -> torch.Tensor:
        data = data.to(device=self.device, dtype=self.dtype)
        compiled = compiled or self.compile(data)
        utility = compiled.design @ beta
        if compiled.fixed_values.numel():
            utility = utility + compiled.fixed_design @ compiled.fixed_values
        return utility

    def loglike_per_obs(
        self,
        params: torch.Tensor,
        data: ChoiceDataset,
        compiled: CompiledCrossNestedUtility | None = None,
    ) -> torch.Tensor:
        data = data.to(device=self.device, dtype=self.dtype)
        compiled = compiled or self.compile(data)
        beta, lambdas = self._split_natural_params(params.to(device=self.device, dtype=self.dtype), compiled)
        utility = self.utilities(beta, data, compiled)
        width = compiled.choice_set_width
        if width is not None:
            utility_by_obs = utility.reshape(data.n_obs, width)
            availability = data.availability.reshape(data.n_obs, width)
            alpha = compiled.alpha_by_row.reshape(data.n_obs, width, len(compiled.nest_names))
            if not bool(availability.any(dim=1).all()):
                raise ValueError("Every observation must have at least one available alternative.")

            iv_columns = []
            numerator_columns = []
            chosen_local = (data.chosen_row - data.obs_ptr[:-1]).reshape(-1, 1)
            chosen_utility = utility_by_obs.gather(1, chosen_local).squeeze(1)
            for nest_index in range(len(compiled.nest_names)):
                lam = lambdas[nest_index]
                nest_alpha = alpha[:, :, nest_index]
                mask = availability & (nest_alpha > 0)
                log_alpha = torch.log(torch.clamp(nest_alpha, min=torch.finfo(self.dtype).tiny))
                scaled = (log_alpha + utility_by_obs / lam).masked_fill(~mask, -torch.inf)
                iv = torch.logsumexp(scaled, dim=1)
                has_nest = mask.any(dim=1)
                iv_columns.append(iv.masked_fill(~has_nest, -torch.inf))
                chosen_alpha = nest_alpha.gather(1, chosen_local).squeeze(1)
                chosen_mask = chosen_alpha > 0
                numerator_columns.append(
                    (
                        torch.log(torch.clamp(chosen_alpha, min=torch.finfo(self.dtype).tiny))
                        + chosen_utility / lam
                        + (lam - 1.0) * iv
                    ).masked_fill(~(chosen_mask & has_nest), -torch.inf)
                )
            iv_by_nest = torch.stack(iv_columns, dim=1)
            denominator = torch.logsumexp(lambdas.reshape(1, -1) * iv_by_nest, dim=1)
            numerator = torch.logsumexp(torch.stack(numerator_columns, dim=1), dim=1)
            return data.weights * (numerator - denominator)

        parts = []
        for obs in range(data.n_obs):
            start = int(data.obs_ptr[obs])
            end = int(data.obs_ptr[obs + 1])
            mask = data.availability[start:end]
            if not bool(mask.any()):
                raise ValueError("Every observation must have at least one available alternative.")
            chosen = int(data.chosen_row[obs])
            chosen_local = chosen - start
            numerator_terms = []
            denominator_terms = []
            for nest_index in range(len(compiled.nest_names)):
                lam = lambdas[nest_index]
                nest_alpha = compiled.alpha_by_row[start:end, nest_index]
                nest_mask = mask & (nest_alpha > 0)
                if not bool(nest_mask.any()):
                    continue
                log_alpha = torch.log(torch.clamp(nest_alpha, min=torch.finfo(self.dtype).tiny))
                iv = torch.logsumexp((log_alpha + utility[start:end] / lam).masked_fill(~nest_mask, -torch.inf), dim=0)
                denominator_terms.append(lam * iv)
                chosen_alpha = nest_alpha[chosen_local]
                if bool(chosen_alpha > 0):
                    numerator_terms.append(torch.log(chosen_alpha) + utility[chosen] / lam + (lam - 1.0) * iv)
            if not numerator_terms:
                raise ValueError("Chosen alternative must have positive allocation in at least one nest.")
            parts.append(data.weights[obs] * (torch.logsumexp(torch.stack(numerator_terms), dim=0) - torch.logsumexp(torch.stack(denominator_terms), dim=0)))
        return torch.stack(parts)

    def loglike(
        self,
        params: torch.Tensor,
        data: ChoiceDataset,
        compiled: CompiledCrossNestedUtility | None = None,
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
            raise NotImplementedError("CrossNestedLogit currently supports classic covariance only.")
        data = data.to(device=self.device, dtype=self.dtype)
        compiled = self.compile(data)
        internal_initial = torch.cat(
            [
                compiled.free_initial,
                self._lambda_to_internal(compiled.lambda_initial[~compiled.lambda_is_fixed]),
            ]
        )
        starts = [internal_initial]
        n_beta = len(compiled.beta_names)
        n_free_lambda = int((~compiled.lambda_is_fixed).sum().detach().cpu())
        if n_free_lambda:
            lambda_starts = [
                self.lambda_min + 1e-6,
                max(self.lambda_min + 1e-6, 0.005),
                0.05,
                0.5,
                0.95,
            ]
            for value in lambda_starts:
                candidate = internal_initial.clone()
                candidate[n_beta:] = self._lambda_to_internal(
                    torch.full((n_free_lambda,), value, dtype=self.dtype, device=self.device)
                )
                starts.append(candidate)

        best_internal = None
        best_convergence_status = None
        best_ll = torch.as_tensor(-torch.inf, dtype=self.dtype, device=self.device)
        total_closures = 0
        for start in starts:
            internal_params = start.clone().detach().requires_grad_(True)
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
            total_closures += iterations["count"]
            candidate_internal = internal_params.detach().clone().requires_grad_(True)
            candidate_natural = self._internal_to_natural(
                candidate_internal,
                compiled,
            )
            candidate_ll = self.loglike(candidate_natural, data, compiled)
            candidate_gradient = torch.autograd.grad(
                candidate_ll,
                candidate_internal,
            )[0].detach()
            candidate_status = lbfgs_convergence_status(
                optimizer,
                candidate_gradient,
                final_loss=-candidate_ll,
                n_obs=data.n_obs,
                closure_evaluations=iterations["count"],
            )
            if bool(candidate_ll.detach() > best_ll):
                best_ll = candidate_ll.detach()
                best_internal = candidate_internal.detach()
                best_convergence_status = candidate_status

        assert best_internal is not None
        assert best_convergence_status is not None
        final_internal = best_internal.clone().detach().requires_grad_(True)
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
        null_ll = self.null_loglike(data)
        return ChoiceResults(
            model=self,
            data=data,
            params=final_natural.detach(),
            param_names=compiled.free_names,
            loglike=float(ll.detach().cpu()),
            null_loglike=float(null_ll.detach().cpu()),
            gradient=gradient,
            hessian=information,
            covariances={"classic": cov_classic},
            cov_type="classic",
            n_obs=data.n_obs,
            n_params=len(compiled.free_names),
            convergence_status={
                **best_convergence_status,
                "closure_evaluations": total_closures,
                "n_starts": len(starts),
            },
        )

    def predict_proba(
        self,
        data: ChoiceDataset,
        params: torch.Tensor,
        compiled: CompiledCrossNestedUtility | None = None,
    ) -> torch.Tensor:
        data = data.to(device=self.device, dtype=self.dtype)
        compiled = compiled or self.compile(data)
        beta, lambdas = self._split_natural_params(params.to(device=self.device, dtype=self.dtype), compiled)
        utility = self.utilities(beta, data, compiled)
        width = compiled.choice_set_width
        if width is not None:
            utility_by_obs = utility.reshape(data.n_obs, width)
            availability = data.availability.reshape(data.n_obs, width)
            alpha = compiled.alpha_by_row.reshape(data.n_obs, width, len(compiled.nest_names))
            if not bool(availability.any(dim=1).all()):
                raise ValueError("Every observation must have at least one available alternative.")
            iv_columns = []
            for nest_index in range(len(compiled.nest_names)):
                lam = lambdas[nest_index]
                nest_alpha = alpha[:, :, nest_index]
                mask = availability & (nest_alpha > 0)
                log_alpha = torch.log(torch.clamp(nest_alpha, min=torch.finfo(self.dtype).tiny))
                scaled = (log_alpha + utility_by_obs / lam).masked_fill(~mask, -torch.inf)
                iv_columns.append(torch.logsumexp(scaled, dim=1).masked_fill(~mask.any(dim=1), -torch.inf))
            iv_by_nest = torch.stack(iv_columns, dim=1)
            denominator = torch.logsumexp(lambdas.reshape(1, -1) * iv_by_nest, dim=1)
            contributions = []
            for nest_index in range(len(compiled.nest_names)):
                lam = lambdas[nest_index]
                nest_alpha = alpha[:, :, nest_index]
                mask = availability & (nest_alpha > 0)
                log_alpha = torch.log(torch.clamp(nest_alpha, min=torch.finfo(self.dtype).tiny))
                log_contrib = (
                    log_alpha
                    + utility_by_obs / lam
                    + (lam - 1.0) * iv_by_nest[:, nest_index].unsqueeze(1)
                    - denominator.unsqueeze(1)
                ).masked_fill(~mask, -torch.inf)
                contributions.append(torch.exp(log_contrib).masked_fill(~mask, 0.0))
            return torch.stack(contributions, dim=0).sum(dim=0).masked_fill(~availability, 0.0).reshape(data.n_rows)

        probs = torch.zeros(data.n_rows, dtype=self.dtype, device=self.device)
        for obs in range(data.n_obs):
            start = int(data.obs_ptr[obs])
            end = int(data.obs_ptr[obs + 1])
            mask = data.availability[start:end]
            if not bool(mask.any()):
                raise ValueError("Every observation must have at least one available alternative.")
            iv_terms = []
            ivs = []
            for nest_index in range(len(compiled.nest_names)):
                lam = lambdas[nest_index]
                nest_alpha = compiled.alpha_by_row[start:end, nest_index]
                nest_mask = mask & (nest_alpha > 0)
                if bool(nest_mask.any()):
                    log_alpha = torch.log(torch.clamp(nest_alpha, min=torch.finfo(self.dtype).tiny))
                    iv = torch.logsumexp((log_alpha + utility[start:end] / lam).masked_fill(~nest_mask, -torch.inf), dim=0)
                else:
                    iv = torch.as_tensor(-torch.inf, dtype=self.dtype, device=self.device)
                ivs.append(iv)
                iv_terms.append(lam * iv)
            denom = torch.logsumexp(torch.stack(iv_terms), dim=0)
            local = torch.zeros(end - start, dtype=self.dtype, device=self.device)
            for nest_index, iv in enumerate(ivs):
                lam = lambdas[nest_index]
                nest_alpha = compiled.alpha_by_row[start:end, nest_index]
                nest_mask = mask & (nest_alpha > 0)
                if not bool(nest_mask.any()):
                    continue
                log_alpha = torch.log(torch.clamp(nest_alpha, min=torch.finfo(self.dtype).tiny))
                local = local + torch.exp((log_alpha + utility[start:end] / lam + (lam - 1.0) * iv - denom).masked_fill(~nest_mask, -torch.inf)).masked_fill(~nest_mask, 0.0)
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
        compiled: CompiledCrossNestedUtility,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        n_beta = len(compiled.beta_names)
        beta = params[:n_beta]
        lambdas = compiled.lambda_fixed.clone()
        free_count = int((~compiled.lambda_is_fixed).sum().detach().cpu())
        if free_count:
            lambdas[~compiled.lambda_is_fixed] = params[n_beta : n_beta + free_count]
        return beta, lambdas

    def _internal_to_natural(self, internal: torch.Tensor, compiled: CompiledCrossNestedUtility) -> torch.Tensor:
        n_beta = len(compiled.beta_names)
        beta = internal[:n_beta]
        lambdas = compiled.lambda_fixed.clone()
        free_count = int((~compiled.lambda_is_fixed).sum().detach().cpu())
        if free_count:
            lambdas[~compiled.lambda_is_fixed] = self._internal_to_lambda(internal[n_beta : n_beta + free_count])
        return torch.cat([beta, lambdas[~compiled.lambda_is_fixed]])

    def _natural_jacobian(self, internal: torch.Tensor, compiled: CompiledCrossNestedUtility) -> torch.Tensor:
        diag = torch.ones_like(internal)
        n_beta = len(compiled.beta_names)
        if internal.numel() > n_beta:
            sigmoid = torch.sigmoid(internal[n_beta:])
            diag[n_beta:] = (1.0 - self.lambda_min) * sigmoid * (1.0 - sigmoid)
        return torch.diag(diag)

    def _lambda_to_internal(self, lambdas: torch.Tensor) -> torch.Tensor:
        if lambdas.numel() == 0:
            return lambdas
        scaled = (lambdas - self.lambda_min) / (1.0 - self.lambda_min)
        scaled = torch.clamp(scaled, min=1e-12, max=1.0 - 1e-12)
        return torch.logit(scaled)

    def _internal_to_lambda(self, internal: torch.Tensor) -> torch.Tensor:
        return self.lambda_min + (1.0 - self.lambda_min) * torch.sigmoid(internal)


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
