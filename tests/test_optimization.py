import torch

from torchdcm.models._optimization import (
    TrackedLBFGS,
    lbfgs_convergence_status,
)


def test_tracked_lbfgs_matches_pytorch_updates():
    target = torch.tensor([1.5, -0.75], dtype=torch.float64)

    def optimize(optimizer_class):
        parameter = torch.tensor(
            [-2.0, 2.0],
            dtype=torch.float64,
            requires_grad=True,
        )
        optimizer = optimizer_class(
            [parameter],
            max_iter=12,
            tolerance_grad=1e-10,
            line_search_fn="strong_wolfe",
        )

        def closure():
            optimizer.zero_grad(set_to_none=True)
            loss = (
                0.25 * (parameter - target).pow(4)
                + 0.5 * (parameter - target).square()
            ).sum()
            loss.backward()
            return loss

        optimizer.step(closure)
        return parameter.detach()

    expected = optimize(torch.optim.LBFGS)
    actual = optimize(TrackedLBFGS)
    assert torch.allclose(actual, expected, rtol=1e-12, atol=1e-12)


def run_quadratic(
    *,
    initial: float,
    scale: float = 1.0,
    max_iter: int = 20,
    tolerance_grad: float = 1e-7,
    tolerance_change: float = 1e-9,
    n_obs: int = 100,
):
    parameter = torch.tensor(
        [initial],
        dtype=torch.float64,
        requires_grad=True,
    )
    optimizer = TrackedLBFGS(
        [parameter],
        max_iter=max_iter,
        tolerance_grad=tolerance_grad,
        tolerance_change=tolerance_change,
    )
    evaluations = {"count": 0}

    def objective():
        return scale * (parameter - 3.0).square().sum()

    def closure():
        optimizer.zero_grad(set_to_none=True)
        loss = objective()
        loss.backward()
        evaluations["count"] += 1
        return loss

    optimizer.step(closure)
    final_loss = objective()
    gradient = torch.autograd.grad(final_loss, parameter)[0]
    status = lbfgs_convergence_status(
        optimizer,
        gradient,
        final_loss=final_loss,
        n_obs=n_obs,
        closure_evaluations=evaluations["count"],
    )
    return optimizer, status


def test_records_gradient_tolerance_stop():
    optimizer, status = run_quadratic(initial=3.0)

    assert optimizer.termination_reason == "gradient_tolerance"
    assert status["message"] == "Converged (gradient tolerance)"
    assert status["success"] is True
    assert status["gradient_norm"] == 0.0
    assert status["normalized_gradient_norm"] == 0.0
    assert status["warnings"] == []


def test_records_function_or_step_stop_without_false_warning():
    optimizer, status = run_quadratic(
        initial=0.0,
        scale=1e-6,
        tolerance_grad=1e-12,
        n_obs=100,
    )

    assert optimizer.termination_reason == "directional_derivative_tolerance"
    assert status["message"] == "Converged (function/step tolerance)"
    assert status["success"] is True
    assert status["normalized_gradient_norm"] < 1e-5
    assert status["warnings"] == []


def test_large_normalized_gradient_still_warns():
    _, status = run_quadratic(
        initial=0.0,
        scale=2e-6,
        tolerance_grad=1e-12,
        n_obs=1,
    )

    assert status["message"] == "Converged (function/step tolerance)"
    assert status["success"] is False
    assert status["normalized_gradient_norm"] > 1e-5
    assert "normalized internal gradient" in status["warnings"][0]


def test_iteration_limit_is_reported_as_warning():
    optimizer, status = run_quadratic(
        initial=0.0,
        max_iter=1,
        tolerance_grad=1e-20,
        tolerance_change=0.0,
        n_obs=100,
    )

    assert optimizer.termination_reason == "iteration_limit"
    assert status["message"] == "Stopped (iteration limit)"
    assert status["success"] is False
    assert status["warnings"] == [
        "L-BFGS reached the maximum number of iterations."
    ]


def test_evaluation_limit_is_reported_as_warning():
    parameter = torch.tensor(
        [0.0],
        dtype=torch.float64,
        requires_grad=True,
    )
    optimizer = TrackedLBFGS(
        [parameter],
        max_iter=10,
        max_eval=1,
        tolerance_grad=1e-20,
        tolerance_change=0.0,
    )
    evaluations = {"count": 0}

    def closure():
        optimizer.zero_grad(set_to_none=True)
        loss = (parameter - 3.0).square().sum()
        loss.backward()
        evaluations["count"] += 1
        return loss

    optimizer.step(closure)
    final_loss = (parameter - 3.0).square().sum()
    gradient = torch.autograd.grad(final_loss, parameter)[0]
    status = lbfgs_convergence_status(
        optimizer,
        gradient,
        final_loss=final_loss,
        n_obs=1,
        closure_evaluations=evaluations["count"],
    )

    assert optimizer.termination_reason == "evaluation_limit"
    assert status["message"] == "Stopped (evaluation limit)"
    assert status["success"] is False
    assert status["warnings"] == [
        "L-BFGS reached the maximum number of function evaluations."
    ]


def test_nonfinite_gradient_is_reported_as_warning():
    optimizer, status = run_quadratic(initial=3.0)
    nonfinite = lbfgs_convergence_status(
        optimizer,
        torch.tensor([float("nan")]),
        final_loss=float("nan"),
        n_obs=1,
        closure_evaluations=1,
    )

    assert nonfinite["message"] == "Stopped (non-finite objective or gradient)"
    assert nonfinite["success"] is False
    assert nonfinite["warnings"]
