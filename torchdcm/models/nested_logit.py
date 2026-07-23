from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
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
class Nest:
    """A disjoint nested-logit nest.

    ``lambda`` is the dissimilarity parameter. Values close to one recover MNL
    within the nest structure, while smaller values imply stronger correlation.
    """

    alternatives: list[str]
    init: float = 0.8
    fixed: bool = False
    name: str | None = None


@dataclass(frozen=True)
class CompiledNestedUtility:
    design: torch.Tensor
    free_names: list[str]
    fixed_names: list[str]
    beta_names: list[str]
    free_initial: torch.Tensor
    fixed_values: torch.Tensor
    fixed_design: torch.Tensor
    nest_id: torch.Tensor
    nest_names: list[str]
    lambda_names: list[str]
    lambda_initial: torch.Tensor
    lambda_fixed: torch.Tensor
    lambda_is_fixed: torch.Tensor
    choice_set_width: int | None


class NestedLogit:
    """Nested logit model with disjoint nests and linear utilities."""

    def __init__(
        self,
        spec: UtilitySpec,
        nests: dict[str, Nest | list[str]],
        *,
        dtype: torch.dtype = torch.float64,
        device: str | torch.device = "cpu",
        max_iter: int = 200,
        tolerance_grad: float = 1e-7,
        line_search_fn: str | None = "strong_wolfe",
        lambda_min: float = 1e-4,
    ) -> None:
        self.spec = spec
        self.nests = {
            name: nest if isinstance(nest, Nest) else Nest(list(nest), name=name)
            for name, nest in nests.items()
        }
        self.dtype = dtype
        self.device = torch.device(device)
        self.max_iter = max_iter
        self.tolerance_grad = tolerance_grad
        self.line_search_fn = line_search_fn
        self.lambda_min = lambda_min
        self._compiled_cache: dict[int, CompiledNestedUtility] = {}

    @classmethod
    def from_formula(
        cls,
        utilities: dict[str, str],
        nests: dict[str, Nest | list[str]],
        **kwargs,
    ) -> "NestedLogit":
        return cls(UtilitySpec.from_formula(utilities), nests, **kwargs)

    def compile(self, data: ChoiceDataset) -> CompiledNestedUtility:
        data = data.to(device=self.device, dtype=self.dtype)
        cache_key = id(data)
        if cache_key in self._compiled_cache:
            return self._compiled_cache[cache_key]

        alt_to_code = {name: i for i, name in enumerate(data.alt_names)}
        missing_alts = sorted(set(self.spec.utilities) - set(alt_to_code))
        if missing_alts:
            raise ValueError(f"Specification contains alternatives not in data: {missing_alts}")

        nest_names = list(self.nests)
        nest_by_alt: dict[str, int] = {}
        # NL requires a partition: each alternative belongs to exactly one
        # nest.  Cross-membership is handled by CrossNestedLogit instead.
        for nest_index, (nest_name, nest) in enumerate(self.nests.items()):
            for alt in nest.alternatives:
                if alt not in alt_to_code:
                    raise ValueError(f"Nest {nest_name!r} contains unknown alternative {alt!r}.")
                if alt in nest_by_alt:
                    raise ValueError(f"Alternative {alt!r} appears in more than one nest.")
                nest_by_alt[alt] = nest_index
        uncovered = sorted(set(data.alt_names) - set(nest_by_alt))
        if uncovered:
            raise ValueError(f"Every data alternative must belong to exactly one nest. Missing: {uncovered}")

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

        nest_id_by_alt = torch.as_tensor(
            [nest_by_alt[alt] for alt in data.alt_names],
            dtype=torch.long,
            device=self.device,
        )
        # Expand alternative-level nest membership to long rows once so the
        # optimizer never performs string or dictionary lookups.
        nest_id = nest_id_by_alt[data.alt_id]
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
        compiled = CompiledNestedUtility(
            design=design,
            free_names=[p.name for p in free_params] + free_lambda_names,
            fixed_names=[p.name for p in fixed_params],
            beta_names=[p.name for p in free_params],
            free_initial=torch.as_tensor([p.init for p in free_params], dtype=self.dtype, device=self.device),
            fixed_values=torch.as_tensor([p.init for p in fixed_params], dtype=self.dtype, device=self.device),
            fixed_design=fixed_design,
            nest_id=nest_id,
            nest_names=nest_names,
            lambda_names=lambda_names,
            lambda_initial=torch.as_tensor(lambda_initial, dtype=self.dtype, device=self.device),
            lambda_fixed=torch.as_tensor(lambda_fixed, dtype=self.dtype, device=self.device),
            lambda_is_fixed=torch.as_tensor(lambda_is_fixed, dtype=torch.bool, device=self.device),
            choice_set_width=_balanced_width(data),
        )
        self._compiled_cache[cache_key] = compiled
        return compiled

    def utilities(self, beta: torch.Tensor, data: ChoiceDataset, compiled: CompiledNestedUtility | None = None) -> torch.Tensor:
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
        compiled: CompiledNestedUtility | None = None,
    ) -> torch.Tensor:
        data = data.to(device=self.device, dtype=self.dtype)
        compiled = compiled or self.compile(data)
        beta, lambdas = self._split_natural_params(params, compiled)
        utility = self.utilities(beta, data, compiled)
        width = compiled.choice_set_width
        if width is not None:
            # Balanced data use tensors with shape (observations, alternatives);
            # nests are reduced in parallel on the same device.
            utility_by_obs = utility.reshape(data.n_obs, width)
            availability = data.availability.reshape(data.n_obs, width)
            nest_by_obs = compiled.nest_id.reshape(data.n_obs, width)
            if not bool(availability.any(dim=1).all()):
                raise ValueError("Every observation must have at least one available alternative.")

            iv_columns = []
            nest_term_columns = []
            for nest_index in range(len(compiled.nest_names)):
                lam = lambdas[nest_index]
                mask = availability & (nest_by_obs == nest_index)
                scaled = (utility_by_obs / lam).masked_fill(~mask, -torch.inf)
                # The inclusive value is a stable log-sum-exp within one nest.
                iv = torch.logsumexp(scaled, dim=1)
                has_nest = mask.any(dim=1)
                iv_columns.append(iv)
                nest_term_columns.append((lam * iv).masked_fill(~has_nest, -torch.inf))
            iv_by_nest = torch.stack(iv_columns, dim=1)
            nest_terms = torch.stack(nest_term_columns, dim=1)
            denom = torch.logsumexp(nest_terms, dim=1)

            chosen_local = (data.chosen_row - data.obs_ptr[:-1]).reshape(-1, 1)
            chosen_utility = utility_by_obs.gather(1, chosen_local).squeeze(1)
            chosen_nest = nest_by_obs.gather(1, chosen_local)
            chosen_lambda = lambdas.gather(0, chosen_nest.squeeze(1))
            chosen_iv = iv_by_nest.gather(1, chosen_nest).squeeze(1)
            return data.weights * (chosen_utility / chosen_lambda + (chosen_lambda - 1.0) * chosen_iv - denom)

        parts = []
        for obs in range(data.n_obs):
            start = int(data.obs_ptr[obs])
            end = int(data.obs_ptr[obs + 1])
            chosen = int(data.chosen_row[obs])
            mask = data.availability[start:end]
            if not bool(mask.any()):
                raise ValueError("Every observation must have at least one available alternative.")

            nest_terms = []
            chosen_iv = None
            chosen_lambda = None
            chosen_nest = int(compiled.nest_id[chosen])
            for nest_index in range(len(compiled.nest_names)):
                local_nest = compiled.nest_id[start:end] == nest_index
                local_mask = mask & local_nest
                if not bool(local_mask.any()):
                    continue
                lam = lambdas[nest_index]
                iv = torch.logsumexp(utility[start:end][local_mask] / lam, dim=0)
                nest_terms.append(lam * iv)
                if nest_index == chosen_nest:
                    chosen_iv = iv
                    chosen_lambda = lam
            if chosen_iv is None or chosen_lambda is None:
                raise ValueError("Chosen alternative must be available and assigned to a nest.")
            denom = torch.logsumexp(torch.stack(nest_terms), dim=0)
            parts.append(
                data.weights[obs]
                * (utility[chosen] / chosen_lambda + (chosen_lambda - 1.0) * chosen_iv - denom)
            )
        return torch.stack(parts)

    def loglike(self, params: torch.Tensor, data: ChoiceDataset, compiled: CompiledNestedUtility | None = None) -> torch.Tensor:
        return self.loglike_per_obs(params, data, compiled).sum()

    def fit(
        self,
        data: ChoiceDataset,
        *,
        cov_type: Literal["classic", "robust", "cluster"] = "classic",
        groups=None,
        max_iter: int | None = None,
    ) -> ChoiceResults:
        fit_started = perf_counter()
        compile_started = perf_counter()
        data = data.to(device=self.device, dtype=self.dtype)
        compiled = self.compile(data)
        compile_seconds = perf_counter() - compile_started
        internal_initial = torch.cat(
            [
                compiled.free_initial,
                self._lambda_to_internal(compiled.lambda_initial[~compiled.lambda_is_fixed]),
            ]
        )
        # A shifted logistic transform keeps every free dissimilarity in
        # (lambda_min, 1) throughout optimization.
        initial_loglike = float(
            self.loglike(self._internal_to_natural(internal_initial, compiled), data, compiled).detach().cpu()
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

        optimization_started = perf_counter()
        optimizer.step(closure)
        optimization_seconds = perf_counter() - optimization_started
        inference_started = perf_counter()
        final_internal = internal_params.detach().clone()
        final_internal.requires_grad_(True)
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
        # Transform uncertainty from unrestricted coordinates to the reported
        # beta/lambda scale.
        transform_jac = self._natural_jacobian(final_internal.detach(), compiled)
        cov_classic = transform_jac @ cov_internal @ transform_jac.T
        hessian_natural = torch.autograd.functional.hessian(lambda p: self.loglike(p, data, compiled), final_natural.detach())
        information = -hessian_natural.detach()
        covariances = {"classic": cov_classic}
        n_clusters = None
        if cov_type in {"robust", "cluster"}:
            scores = self.scores(final_natural.detach(), data, compiled)
            meat = scores.T @ scores
            covariances["robust"] = cov_classic @ meat @ cov_classic
            cluster_codes = data.cluster_codes(groups)
            if cluster_codes is not None:
                n_clusters = int(torch.unique(cluster_codes).numel())
                cluster_meat = _cluster_meat(scores, cluster_codes)
                covariances["cluster"] = cov_classic @ cluster_meat @ cov_classic
            elif cov_type == "cluster":
                raise ValueError("Cluster covariance requested, but no groups were supplied.")

        null_ll = self.null_loglike(data)
        inference_seconds = perf_counter() - inference_started
        total_seconds = perf_counter() - fit_started
        return ChoiceResults(
            model=self,
            data=data,
            params=final_natural.detach(),
            param_names=compiled.free_names,
            loglike=float(ll.detach().cpu()),
            null_loglike=float(null_ll.detach().cpu()),
            gradient=gradient,
            hessian=information,
            covariances=covariances,
            cov_type=cov_type,
            n_obs=data.n_obs,
            n_params=len(compiled.free_names),
            convergence_status={
                **convergence_status,
                "initial_loglike": initial_loglike,
                "n_clusters": n_clusters,
                "compile_seconds": compile_seconds,
                "optimization_seconds": optimization_seconds,
                "inference_seconds": inference_seconds,
                "total_seconds": total_seconds,
            },
        )

    def scores(
        self,
        params: torch.Tensor,
        data: ChoiceDataset,
        compiled: CompiledNestedUtility | None = None,
    ) -> torch.Tensor:
        data = data.to(device=self.device, dtype=self.dtype)
        compiled = compiled or self.compile(data)
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
        compiled: CompiledNestedUtility | None = None,
    ) -> torch.Tensor:
        data = data.to(device=self.device, dtype=self.dtype)
        compiled = compiled or self.compile(data)
        beta, lambdas = self._split_natural_params(params.to(device=self.device, dtype=self.dtype), compiled)
        utility = self.utilities(beta, data, compiled)
        width = compiled.choice_set_width
        if width is not None:
            utility_by_obs = utility.reshape(data.n_obs, width)
            availability = data.availability.reshape(data.n_obs, width)
            nest_by_obs = compiled.nest_id.reshape(data.n_obs, width)
            if not bool(availability.any(dim=1).all()):
                raise ValueError("Every observation must have at least one available alternative.")

            conditional = []
            nest_terms = []
            for nest_index in range(len(compiled.nest_names)):
                lam = lambdas[nest_index]
                mask = availability & (nest_by_obs == nest_index)
                scaled = (utility_by_obs / lam).masked_fill(~mask, -torch.inf)
                iv = torch.logsumexp(scaled, dim=1)
                has_nest = mask.any(dim=1)
                conditional.append(torch.softmax(scaled, dim=1).masked_fill(~mask, 0.0))
                nest_terms.append((lam * iv).masked_fill(~has_nest, -torch.inf))
            nest_term_matrix = torch.stack(nest_terms, dim=1)
            nest_probs = torch.softmax(nest_term_matrix, dim=1)
            probs_by_obs = torch.zeros_like(utility_by_obs)
            for nest_index, cond_prob in enumerate(conditional):
                probs_by_obs = probs_by_obs + nest_probs[:, nest_index].unsqueeze(1) * cond_prob
            return probs_by_obs.masked_fill(~availability, 0.0).reshape(data.n_rows)

        probs = torch.zeros(data.n_rows, dtype=self.dtype, device=self.device)
        for obs in range(data.n_obs):
            start = int(data.obs_ptr[obs])
            end = int(data.obs_ptr[obs + 1])
            mask = data.availability[start:end]
            if not bool(mask.any()):
                raise ValueError("Every observation must have at least one available alternative.")
            nest_terms: dict[int, torch.Tensor] = {}
            conditional: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
            for nest_index in range(len(compiled.nest_names)):
                local_nest = compiled.nest_id[start:end] == nest_index
                local_mask = mask & local_nest
                if not bool(local_mask.any()):
                    continue
                lam = lambdas[nest_index]
                scaled = utility[start:end][local_mask] / lam
                iv = torch.logsumexp(scaled, dim=0)
                nest_terms[nest_index] = lam * iv
                conditional[nest_index] = (local_mask, torch.softmax(scaled, dim=0))
            denom = torch.logsumexp(torch.stack(list(nest_terms.values())), dim=0)
            local_probs = torch.zeros(end - start, dtype=self.dtype, device=self.device)
            for nest_index, term in nest_terms.items():
                nest_prob = torch.exp(term - denom)
                local_mask, cond_prob = conditional[nest_index]
                local_probs[local_mask] = nest_prob * cond_prob
            probs[start:end] = local_probs
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
        compiled: CompiledNestedUtility,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        n_beta = len(compiled.beta_names)
        beta = params[:n_beta]
        lambdas = compiled.lambda_fixed.clone()
        free_count = int((~compiled.lambda_is_fixed).sum().detach().cpu())
        if free_count:
            lambdas[~compiled.lambda_is_fixed] = params[n_beta : n_beta + free_count]
        return beta, lambdas

    def _internal_to_natural(self, internal: torch.Tensor, compiled: CompiledNestedUtility) -> torch.Tensor:
        n_beta = len(compiled.beta_names)
        beta = internal[:n_beta]
        lambdas = compiled.lambda_fixed.clone()
        free_count = int((~compiled.lambda_is_fixed).sum().detach().cpu())
        if free_count:
            lambdas[~compiled.lambda_is_fixed] = self._internal_to_lambda(internal[n_beta : n_beta + free_count])
        return torch.cat([beta, lambdas[~compiled.lambda_is_fixed]])

    def _natural_jacobian(self, internal: torch.Tensor, compiled: CompiledNestedUtility) -> torch.Tensor:
        diag = torch.ones_like(internal)
        n_beta = len(compiled.beta_names)
        if internal.numel() > n_beta:
            z = internal[n_beta:]
            sigmoid = torch.sigmoid(z)
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


def _cluster_meat(scores: torch.Tensor, cluster_codes: torch.Tensor) -> torch.Tensor:
    n_clusters = int(cluster_codes.max().detach().cpu()) + 1
    accum = []
    for code in range(n_clusters):
        accum.append(scores[cluster_codes == code].sum(dim=0))
    cluster_scores = torch.stack(accum)
    return cluster_scores.T @ cluster_scores
