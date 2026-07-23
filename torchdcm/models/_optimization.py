from __future__ import annotations

from math import isfinite
from typing import Callable

import torch
from torch.optim.lbfgs import _strong_wolfe


NORMALIZED_GRADIENT_WARNING_TOLERANCE = 1e-5


class TrackedLBFGS(torch.optim.LBFGS):
    """PyTorch L-BFGS with an explicit record of its stopping condition."""

    termination_reason: str | None
    local_iterations: int
    local_function_evaluations: int

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.termination_reason = None
        self.local_iterations = 0
        self.local_function_evaluations = 0

    @torch.no_grad()
    def step(self, closure: Callable[[], torch.Tensor]):  # type: ignore[override]
        if len(self.param_groups) != 1:
            raise AssertionError(
                f"Expected exactly one param_group, got {len(self.param_groups)}"
            )

        closure = torch.enable_grad()(closure)
        group = self.param_groups[0]
        lr_value = group["lr"]
        lr = float(lr_value.detach().cpu()) if torch.is_tensor(lr_value) else float(lr_value)
        max_iter = int(group["max_iter"])
        max_eval = int(group["max_eval"])
        tolerance_grad = float(group["tolerance_grad"])
        tolerance_change = float(group["tolerance_change"])
        line_search_fn = group["line_search_fn"]
        history_size = int(group["history_size"])

        state = self.state[self._params[0]]
        state.setdefault("func_evals", 0)
        state.setdefault("n_iter", 0)

        original_loss = closure()
        loss = float(original_loss)
        current_evals = 1
        state["func_evals"] += 1

        flat_grad = self._gather_flat_grad()
        if bool(flat_grad.abs().max() <= tolerance_grad):
            self._record_stop("gradient_tolerance", 0, current_evals)
            return original_loss

        direction = state.get("d")
        step_size = state.get("t")
        old_dirs = state.get("old_dirs")
        old_steps = state.get("old_stps")
        reciprocal_curvatures = state.get("ro")
        hessian_scale = state.get("H_diag")
        previous_gradient = state.get("prev_flat_grad")
        previous_loss = state.get("prev_loss")

        local_iterations = 0
        stop_reason = "iteration_limit"
        while local_iterations < max_iter:
            local_iterations += 1
            state["n_iter"] += 1

            if state["n_iter"] == 1:
                direction = flat_grad.neg()
                old_dirs = []
                old_steps = []
                reciprocal_curvatures = []
                hessian_scale = 1
            else:
                gradient_difference = flat_grad.sub(previous_gradient)
                previous_step = direction.mul(step_size)
                curvature = gradient_difference.dot(previous_step)
                if curvature > 1e-10:
                    if len(old_dirs) == history_size:
                        old_dirs.pop(0)
                        old_steps.pop(0)
                        reciprocal_curvatures.pop(0)
                    old_dirs.append(gradient_difference)
                    old_steps.append(previous_step)
                    reciprocal_curvatures.append(1.0 / curvature)
                    hessian_scale = curvature / gradient_difference.dot(
                        gradient_difference
                    )

                history_length = len(old_dirs)
                if "al" not in state:
                    state["al"] = [None] * history_size
                coefficients = state["al"]
                approximate_direction = flat_grad.neg()
                for index in range(history_length - 1, -1, -1):
                    coefficients[index] = (
                        old_steps[index].dot(approximate_direction)
                        * reciprocal_curvatures[index]
                    )
                    approximate_direction.add_(
                        old_dirs[index], alpha=-coefficients[index]
                    )

                direction = approximate_direction.mul(hessian_scale)
                for index in range(history_length):
                    beta = (
                        old_dirs[index].dot(direction)
                        * reciprocal_curvatures[index]
                    )
                    direction.add_(
                        old_steps[index],
                        alpha=coefficients[index] - beta,
                    )

            if previous_gradient is None:
                previous_gradient = flat_grad.clone(
                    memory_format=torch.contiguous_format
                )
            else:
                previous_gradient.copy_(flat_grad)
            previous_loss = loss

            if state["n_iter"] == 1:
                step_size = min(1.0, 1.0 / float(flat_grad.abs().sum())) * lr
            else:
                step_size = lr

            directional_derivative = flat_grad.dot(direction)
            if directional_derivative > -tolerance_change:
                stop_reason = "directional_derivative_tolerance"
                break

            line_search_evals = 0
            if line_search_fn is not None:
                if line_search_fn != "strong_wolfe":
                    raise RuntimeError("only 'strong_wolfe' is supported")
                initial_parameters = self._clone_param()

                def objective(parameters, step, search_direction):
                    return self._directional_evaluate(
                        closure,
                        parameters,
                        step,
                        search_direction,
                    )

                loss, flat_grad, step_size, line_search_evals = _strong_wolfe(
                    objective,
                    initial_parameters,
                    step_size,
                    direction,
                    loss,
                    flat_grad,
                    directional_derivative,
                    max_ls=max_eval - current_evals,
                )
                self._add_grad(step_size, direction)
                gradient_condition = (
                    flat_grad.abs().max() <= tolerance_grad
                )
            else:
                self._add_grad(step_size, direction)
                gradient_condition = False
                if local_iterations != max_iter:
                    with torch.enable_grad():
                        loss = closure()
                    loss = float(loss)
                    flat_grad = self._gather_flat_grad()
                    gradient_condition = (
                        flat_grad.abs().max() <= tolerance_grad
                    )
                    line_search_evals = 1

            current_evals += line_search_evals
            state["func_evals"] += line_search_evals

            if local_iterations == max_iter:
                stop_reason = "iteration_limit"
                break
            if current_evals >= max_eval:
                stop_reason = "evaluation_limit"
                break
            if bool(gradient_condition):
                stop_reason = "gradient_tolerance"
                break
            if direction.mul(step_size).abs().max() <= tolerance_change:
                stop_reason = "step_tolerance"
                break
            if abs(loss - previous_loss) < tolerance_change:
                stop_reason = "function_tolerance"
                break

        state["d"] = direction
        state["t"] = step_size
        state["old_dirs"] = old_dirs
        state["old_stps"] = old_steps
        state["ro"] = reciprocal_curvatures
        state["H_diag"] = hessian_scale
        state["prev_flat_grad"] = previous_gradient
        state["prev_loss"] = previous_loss
        self._record_stop(stop_reason, local_iterations, current_evals)
        return original_loss

    def _record_stop(
        self,
        reason: str,
        iterations: int,
        function_evaluations: int,
    ) -> None:
        self.termination_reason = reason
        self.local_iterations = int(iterations)
        self.local_function_evaluations = int(function_evaluations)


def lbfgs_convergence_status(
    optimizer: TrackedLBFGS,
    internal_gradient: torch.Tensor,
    *,
    final_loss: torch.Tensor | float,
    n_obs: int,
    closure_evaluations: int,
    normalized_gradient_tolerance: float = (
        NORMALIZED_GRADIENT_WARNING_TOLERANCE
    ),
) -> dict[str, object]:
    """Translate an exact L-BFGS stop into report-ready diagnostics."""

    gradient = internal_gradient.detach()
    gradient_finite = bool(torch.isfinite(gradient).all())
    gradient_inf = (
        float(gradient.abs().max().cpu()) if gradient.numel() else 0.0
    )
    normalized_gradient = gradient_inf / max(int(n_obs), 1)
    loss_value = (
        float(final_loss.detach().cpu())
        if torch.is_tensor(final_loss)
        else float(final_loss)
    )
    finite = gradient_finite and isfinite(loss_value)
    reason = optimizer.termination_reason or "unknown"

    if not finite:
        message = "Stopped (non-finite objective or gradient)"
        success: bool | None = False
        warnings = ["The optimizer produced a non-finite objective or gradient."]
    elif reason == "gradient_tolerance":
        message = "Converged (gradient tolerance)"
        success = True
        warnings = []
    elif reason in {
        "directional_derivative_tolerance",
        "step_tolerance",
        "function_tolerance",
    }:
        message = "Converged (function/step tolerance)"
        success = normalized_gradient <= normalized_gradient_tolerance
        warnings = []
    elif reason == "iteration_limit":
        message = "Stopped (iteration limit)"
        success = False
        warnings = ["L-BFGS reached the maximum number of iterations."]
    elif reason == "evaluation_limit":
        message = "Stopped (evaluation limit)"
        success = False
        warnings = ["L-BFGS reached the maximum number of function evaluations."]
    else:
        message = "Completed (unclassified L-BFGS stop)"
        success = None
        warnings = ["L-BFGS did not expose a recognized stopping condition."]

    if (
        finite
        and normalized_gradient > normalized_gradient_tolerance
        and reason not in {"iteration_limit", "evaluation_limit"}
    ):
        success = False
        warnings.append(
            "The normalized internal gradient exceeds "
            f"{normalized_gradient_tolerance:.1e}."
        )

    group = optimizer.param_groups[0]
    return {
        "optimizer": "torch.optim.LBFGS",
        "success": success,
        "message": message,
        "termination_code": reason,
        "optimizer_iterations": optimizer.local_iterations,
        "closure_evaluations": int(closure_evaluations),
        "function_evaluations": optimizer.local_function_evaluations,
        "gradient_norm": gradient_inf,
        "gradient_norm_type": "Internal-parameter infinity norm",
        "normalized_gradient_norm": normalized_gradient,
        "normalized_gradient_tolerance": normalized_gradient_tolerance,
        "gradient_tolerance": float(group["tolerance_grad"]),
        "function_step_tolerance": float(group["tolerance_change"]),
        "warnings": warnings,
    }
