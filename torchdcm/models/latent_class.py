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
from torchdcm.spec.expressions import Expression, Term
from torchdcm.spec.parameters import Beta
from torchdcm.spec.utility import UtilitySpec


@dataclass(frozen=True)
class CompiledLatentClassUtility:
    design: torch.Tensor
    fixed_design: torch.Tensor
    free_names: list[str]
    fixed_names: list[str]
    utility_free_names: list[str]
    membership_free_names: list[str]
    free_initial: torch.Tensor
    fixed_values: torch.Tensor
    membership_design: torch.Tensor
    membership_fixed_design: torch.Tensor
    membership_fixed_values: torch.Tensor
    choice_set_width: int | None


class LatentClassLogit:
    """Latent class logit with class-specific MNL utilities.

    The first class is the reference class with membership logit fixed to zero.
    Each additional class can use a linear membership expression such as
    ``Beta("CLASS_2") + Beta("CLASS_2_GA") * "ga"``.
    """

    def __init__(
        self,
        class_specs: list[UtilitySpec],
        *,
        class_membership: list[Beta | Expression] | None = None,
        dtype: torch.dtype = torch.float64,
        device: str | torch.device = "cpu",
        max_iter: int = 200,
        tolerance_grad: float = 1e-7,
        line_search_fn: str | None = "strong_wolfe",
    ) -> None:
        if len(class_specs) < 2:
            raise ValueError("LatentClassLogit requires at least two classes.")
        self.class_specs = list(class_specs)
        if class_membership is None:
            class_membership = [Beta(f"CLASS_{i + 1}", init=0.0) for i in range(1, len(class_specs))]
        if len(class_membership) != len(class_specs) - 1:
            raise ValueError("class_membership must contain one Beta for each non-reference class.")
        self.class_membership = [_as_expression(item) for item in class_membership]
        self.dtype = dtype
        self.device = torch.device(device)
        self.max_iter = max_iter
        self.tolerance_grad = tolerance_grad
        self.line_search_fn = line_search_fn
        self._compiled_cache: dict[int, CompiledLatentClassUtility] = {}

    def compile(self, data: ChoiceDataset) -> CompiledLatentClassUtility:
        data = data.to(device=self.device, dtype=self.dtype)
        cache_key = id(data)
        if cache_key in self._compiled_cache:
            return self._compiled_cache[cache_key]

        alt_to_code = {name: i for i, name in enumerate(data.alt_names)}
        for class_index, spec in enumerate(self.class_specs):
            missing_alts = sorted(set(spec.utilities) - set(alt_to_code))
            if missing_alts:
                raise ValueError(f"Class {class_index} contains alternatives not in data: {missing_alts}")

        utility_params = _collect_unique_parameters(spec.parameters for spec in self.class_specs)
        fixed_params = [p for p in utility_params if p.fixed]
        free_params = [p for p in utility_params if not p.fixed]
        membership_params = _collect_unique_parameters(expr.parameters for expr in self.class_membership)
        membership_fixed = [p for p in membership_params if p.fixed]
        membership_free = [p for p in membership_params if not p.fixed]

        free_names = [p.name for p in free_params]
        fixed_names = [p.name for p in fixed_params]
        for param in membership_free:
            if param.name in free_names:
                raise ValueError(f"Membership parameter name conflicts with utility parameter: {param.name!r}.")
            free_names.append(param.name)
        for param in membership_fixed:
            if param.name in fixed_names:
                raise ValueError(f"Membership parameter name conflicts with fixed utility parameter: {param.name!r}.")
            fixed_names.append(param.name)

        utility_free_index = {p.name: i for i, p in enumerate(free_params)}
        fixed_index = {p.name: i for i, p in enumerate(fixed_params)}
        design = torch.zeros(
            (len(self.class_specs), data.n_rows, len(free_params)),
            dtype=self.dtype,
            device=self.device,
        )
        fixed_design = torch.zeros(
            (len(self.class_specs), data.n_rows, len(fixed_params)),
            dtype=self.dtype,
            device=self.device,
        )

        # The leading class dimension lets all class-specific utilities share
        # one tensor evaluation even when coefficients differ by class.
        for class_index, spec in enumerate(self.class_specs):
            for alt_name, expr in spec.utilities.items():
                rows = data.alt_id == alt_to_code[alt_name]
                for term in expr.terms:
                    values = (
                        torch.ones(data.n_rows, dtype=self.dtype, device=self.device)
                        if term.variable is None
                        else data.x_alt[term.variable].to(device=self.device, dtype=self.dtype)
                    )
                    contribution = term.multiplier * values
                    if term.parameter.fixed:
                        fixed_design[class_index, rows, fixed_index[term.parameter.name]] += contribution[rows]
                    else:
                        design[class_index, rows, utility_free_index[term.parameter.name]] += contribution[rows]

        fixed_values = [p.init for p in fixed_params]
        membership_free_index = {p.name: i for i, p in enumerate(membership_free)}
        membership_fixed_index = {p.name: i for i, p in enumerate(membership_fixed)}
        membership_design = torch.zeros(
            (len(self.class_membership), data.n_obs, len(membership_free)),
            dtype=self.dtype,
            device=self.device,
        )
        membership_fixed_design = torch.zeros(
            (len(self.class_membership), data.n_obs, len(membership_fixed)),
            dtype=self.dtype,
            device=self.device,
        )
        # Membership covariates are read once from the first long row of each
        # observation and should therefore be constant across its alternatives.
        obs_rows = data.obs_ptr[:-1]
        for class_index, expr in enumerate(self.class_membership):
            for term in expr.terms:
                values = (
                    torch.ones(data.n_obs, dtype=self.dtype, device=self.device)
                    if term.variable is None
                    else data.x_alt[term.variable].to(device=self.device, dtype=self.dtype)[obs_rows]
                )
                contribution = term.multiplier * values
                if term.parameter.fixed:
                    membership_fixed_design[class_index, :, membership_fixed_index[term.parameter.name]] += contribution
                else:
                    membership_design[class_index, :, membership_free_index[term.parameter.name]] += contribution

        compiled = CompiledLatentClassUtility(
            design=design,
            fixed_design=fixed_design,
            free_names=free_names,
            fixed_names=fixed_names,
            utility_free_names=[p.name for p in free_params],
            membership_free_names=[p.name for p in membership_free],
            free_initial=torch.as_tensor(
                [p.init for p in free_params] + [p.init for p in membership_free],
                dtype=self.dtype,
                device=self.device,
            ),
            fixed_values=torch.as_tensor(fixed_values, dtype=self.dtype, device=self.device),
            membership_design=membership_design,
            membership_fixed_design=membership_fixed_design,
            membership_fixed_values=torch.as_tensor([p.init for p in membership_fixed], dtype=self.dtype, device=self.device),
            choice_set_width=_balanced_width(data),
        )
        self._compiled_cache[cache_key] = compiled
        return compiled

    def utilities(
        self,
        params: torch.Tensor,
        data: ChoiceDataset,
        compiled: CompiledLatentClassUtility | None = None,
    ) -> torch.Tensor:
        data = data.to(device=self.device, dtype=self.dtype)
        compiled = compiled or self.compile(data)
        n_utility = len(compiled.utility_free_names)
        utility_params = params[:n_utility].to(device=self.device, dtype=self.dtype)
        utility = compiled.design @ utility_params
        if compiled.fixed_values.numel():
            utility = utility + compiled.fixed_design @ compiled.fixed_values[: compiled.fixed_design.shape[-1]]
        return utility

    def class_logits(self, params: torch.Tensor, compiled: CompiledLatentClassUtility) -> torch.Tensor:
        n_utility = len(compiled.utility_free_names)
        logits = torch.zeros(
            (len(self.class_specs) - 1, compiled.membership_design.shape[1]),
            dtype=self.dtype,
            device=self.device,
        )
        # Class 1 is the reference with a zero membership index.  Free logits
        # are estimated only for the remaining classes.
        free_count = len(compiled.membership_free_names)
        if free_count:
            membership_params = params[n_utility : n_utility + free_count].to(device=self.device, dtype=self.dtype)
            logits = logits + compiled.membership_design @ membership_params
        if compiled.membership_fixed_values.numel():
            logits = logits + compiled.membership_fixed_design @ compiled.membership_fixed_values
        return torch.cat(
            [torch.zeros((1, logits.shape[1]), dtype=self.dtype, device=self.device), logits],
            dim=0,
        )

    def class_probabilities(self, params: torch.Tensor, data: ChoiceDataset | None = None) -> torch.Tensor:
        if data is None:
            if not self._compiled_cache:
                raise ValueError("Pass data before the model has been compiled.")
            compiled = next(iter(self._compiled_cache.values()))
        else:
            compiled = self.compile(data)
        probabilities = torch.softmax(self.class_logits(params.to(device=self.device, dtype=self.dtype), compiled), dim=0)
        if probabilities.shape[1] == 0:
            return probabilities.T
        if bool(torch.allclose(probabilities, probabilities[:, :1])):
            return probabilities[:, 0]
        return probabilities.T

    def _class_log_prob_per_obs(
        self,
        params: torch.Tensor,
        data: ChoiceDataset,
        compiled: CompiledLatentClassUtility,
    ) -> torch.Tensor:
        utility = self.utilities(params, data, compiled)
        width = compiled.choice_set_width
        if width is not None:
            utility_by_class_obs = utility.reshape(len(self.class_specs), data.n_obs, width)
            availability = data.availability.reshape(data.n_obs, width).unsqueeze(0)
            if not bool(availability.any(dim=2).all()):
                raise ValueError("Every observation must have at least one available alternative.")
            chosen_local = (data.chosen_row - data.obs_ptr[:-1]).reshape(1, -1, 1)
            chosen_utility = utility_by_class_obs.gather(
                2,
                chosen_local.expand(len(self.class_specs), -1, 1),
            ).squeeze(2)
            log_denom = torch.logsumexp(utility_by_class_obs.masked_fill(~availability, -torch.inf), dim=2)
            return chosen_utility - log_denom

        rows = []
        for obs in range(data.n_obs):
            start = int(data.obs_ptr[obs])
            end = int(data.obs_ptr[obs + 1])
            mask = data.availability[start:end]
            if not bool(mask.any()):
                raise ValueError("Every observation must have at least one available alternative.")
            chosen = int(data.chosen_row[obs])
            chosen_by_class = utility[:, chosen]
            log_denom = torch.logsumexp(utility[:, start:end].masked_fill(~mask.unsqueeze(0), -torch.inf), dim=1)
            rows.append(chosen_by_class - log_denom)
        return torch.stack(rows, dim=1)

    def loglike_per_obs(
        self,
        params: torch.Tensor,
        data: ChoiceDataset,
        compiled: CompiledLatentClassUtility | None = None,
    ) -> torch.Tensor:
        data = data.to(device=self.device, dtype=self.dtype)
        compiled = compiled or self.compile(data)
        params = params.to(device=self.device, dtype=self.dtype)
        class_log_probs = torch.log_softmax(self.class_logits(params, compiled), dim=0)
        obs_class_log_probs = self._class_log_prob_per_obs(params, data, compiled)
        # Marginalize the unobserved class in log space for numerical stability.
        return data.weights * torch.logsumexp(class_log_probs + obs_class_log_probs, dim=0)

    def loglike(
        self,
        params: torch.Tensor,
        data: ChoiceDataset,
        compiled: CompiledLatentClassUtility | None = None,
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
            raise NotImplementedError("LatentClassLogit currently supports classic covariance only.")
        data = data.to(device=self.device, dtype=self.dtype)
        compiled = self.compile(data)
        params = compiled.free_initial.clone().detach().requires_grad_(True)
        optimizer = TrackedLBFGS(
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
        convergence_status = lbfgs_convergence_status(
            optimizer,
            gradient,
            final_loss=-ll,
            n_obs=data.n_obs,
            closure_evaluations=iterations["count"],
        )
        hessian_ll = torch.autograd.functional.hessian(lambda p: self.loglike(p, data, compiled), final_params)
        information = -hessian_ll.detach()
        cov_classic = _safe_pinv(information)
        null_ll = self.null_loglike(data)
        return ChoiceResults(
            model=self,
            data=data,
            params=final_params.detach(),
            param_names=compiled.free_names,
            loglike=float(ll.detach().cpu()),
            null_loglike=float(null_ll.detach().cpu()),
            gradient=gradient,
            hessian=information,
            covariances={"classic": cov_classic},
            cov_type="classic",
            n_obs=data.n_obs,
            n_params=len(compiled.free_names),
            convergence_status=convergence_status,
        )

    def predict_proba(
        self,
        data: ChoiceDataset,
        params: torch.Tensor,
        compiled: CompiledLatentClassUtility | None = None,
    ) -> torch.Tensor:
        data = data.to(device=self.device, dtype=self.dtype)
        compiled = compiled or self.compile(data)
        params = params.to(device=self.device, dtype=self.dtype)
        utility = self.utilities(params, data, compiled)
        class_probs = torch.softmax(self.class_logits(params, compiled), dim=0)
        width = compiled.choice_set_width
        if width is not None:
            utility_by_class_obs = utility.reshape(len(self.class_specs), data.n_obs, width)
            availability = data.availability.reshape(data.n_obs, width).unsqueeze(0)
            if not bool(availability.any(dim=2).all()):
                raise ValueError("Every observation must have at least one available alternative.")
            probabilities = torch.softmax(utility_by_class_obs.masked_fill(~availability, -torch.inf), dim=2)
            probabilities = probabilities.masked_fill(~availability, 0.0)
            return (class_probs.unsqueeze(2) * probabilities).sum(dim=0).reshape(data.n_rows)

        probs = torch.zeros(data.n_rows, dtype=self.dtype, device=self.device)
        for obs in range(data.n_obs):
            start = int(data.obs_ptr[obs])
            end = int(data.obs_ptr[obs + 1])
            mask = data.availability[start:end]
            class_probabilities = torch.softmax(utility[:, start:end].masked_fill(~mask.unsqueeze(0), -torch.inf), dim=1)
            class_probabilities = class_probabilities.masked_fill(~mask.unsqueeze(0), 0.0)
            probs[start:end] = (class_probs[:, obs].reshape(-1, 1) * class_probabilities).sum(dim=0)
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

    def posterior_class_probabilities(
        self,
        data: ChoiceDataset,
        params: torch.Tensor,
        compiled: CompiledLatentClassUtility | None = None,
    ) -> torch.Tensor:
        data = data.to(device=self.device, dtype=self.dtype)
        compiled = compiled or self.compile(data)
        params = params.to(device=self.device, dtype=self.dtype)
        log_prior = torch.log_softmax(self.class_logits(params, compiled), dim=0)
        obs_class_log_probs = self._class_log_prob_per_obs(params, data, compiled)
        log_joint = log_prior + obs_class_log_probs
        return torch.softmax(log_joint, dim=0).T

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


def _collect_unique_parameters(groups) -> list[Beta]:
    params: dict[str, Beta] = {}
    for group in groups:
        for param in group:
            old = params.get(param.name)
            if old is not None and old != param:
                raise ValueError(f"Conflicting definitions for parameter {param.name!r}.")
            params[param.name] = param
    return list(params.values())


def _as_expression(value: Beta | Expression) -> Expression:
    if isinstance(value, Expression):
        return value
    if isinstance(value, Beta):
        return Expression([Term(value, None, 1.0)])
    raise TypeError(f"Cannot convert {type(value)!r} to a membership expression.")


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
