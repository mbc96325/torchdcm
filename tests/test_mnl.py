import math

import numpy as np
import torch

from torchdcm import Beta, ChoiceDataset, MultinomialLogit, UtilitySpec
from torchdcm.datasets import make_london_like, make_swissmetro_like


def swissmetro_data(n_obs=120):
    df = make_swissmetro_like(n_obs=n_obs, seed=12)
    return ChoiceDataset.from_wide(
        df,
        alternatives=["TRAIN", "SM", "CAR"],
        choice="choice",
        variables={
            "time": {"TRAIN": "time_train", "SM": "time_sm", "CAR": "time_car"},
            "cost": {"TRAIN": "cost_train", "SM": "cost_sm", "CAR": "cost_car"},
        },
        availability={"TRAIN": "avail_train", "SM": "avail_sm", "CAR": "avail_car"},
        individual_id="person_id",
    )


def swissmetro_spec():
    spec = UtilitySpec()
    spec.utility("TRAIN", Beta("ASC_TRAIN") + Beta("B_TIME", init=-0.01) * "time" + Beta("B_COST", init=-0.1) * "cost")
    spec.utility("SM", Beta("B_TIME", init=-0.01) * "time" + Beta("B_COST", init=-0.1) * "cost")
    spec.utility("CAR", Beta("ASC_CAR") + Beta("B_TIME", init=-0.01) * "time" + Beta("B_COST", init=-0.1) * "cost")
    return spec


def test_choice_dataset_from_wide_invariants():
    data = swissmetro_data(30)
    assert data.n_obs == 30
    assert data.n_rows == 90
    assert set(data.x_alt) == {"time", "cost"}
    for obs in range(data.n_obs):
        start = int(data.obs_ptr[obs])
        end = int(data.obs_ptr[obs + 1])
        assert start <= int(data.chosen_row[obs]) < end
        assert bool(data.availability[int(data.chosen_row[obs])])


def test_probabilities_sum_to_one():
    data = swissmetro_data(50)
    model = MultinomialLogit(swissmetro_spec())
    compiled = model.compile(data)
    params = compiled.free_initial
    probs = model.predict_proba(data, params)
    for obs in range(data.n_obs):
        start = int(data.obs_ptr[obs])
        end = int(data.obs_ptr[obs + 1])
        assert torch.allclose(probs[start:end].sum(), torch.tensor(1.0, dtype=torch.float64))
        assert torch.all(probs[start:end][~data.availability[start:end]] == 0)


def test_gradient_matches_finite_difference():
    data = swissmetro_data(40)
    model = MultinomialLogit(swissmetro_spec())
    compiled = model.compile(data)
    params = compiled.free_initial.clone().detach().requires_grad_(True)
    ll = model.loglike(params, data, compiled)
    grad = torch.autograd.grad(ll, params)[0].detach().numpy()
    eps = 1e-5
    fd = []
    for k in range(len(params)):
        plus = params.detach().clone()
        minus = params.detach().clone()
        plus[k] += eps
        minus[k] -= eps
        fd.append(float((model.loglike(plus, data, compiled) - model.loglike(minus, data, compiled)) / (2 * eps)))
    assert np.allclose(grad, np.asarray(fd), rtol=1e-4, atol=1e-4)


def test_fit_returns_covariances_and_predictions():
    data = swissmetro_data(80)
    result = MultinomialLogit(swissmetro_spec(), max_iter=80).fit(data, cov_type="cluster", groups="person_id")
    assert math.isfinite(result.loglike)
    assert result.cov_params("classic").shape == (4, 4)
    assert result.cov_params("robust").shape == (4, 4)
    assert result.cov_params("cluster").shape == (4, 4)
    assert len(result.predict()) == data.n_obs
    assert result.predict_proba().shape == (data.n_rows,)
    assert "B_TIME" in result.summary()


def test_formula_api_and_london_case():
    df = make_london_like(n_obs=60, seed=4)
    data = ChoiceDataset.from_wide(
        df,
        alternatives=["tube", "bus", "car", "bike"],
        choice="choice",
        variables={
            "time": {a: f"time_{a}" for a in ["tube", "bus", "car", "bike"]},
            "cost": {a: f"cost_{a}" for a in ["tube", "bus", "car", "bike"]},
        },
        availability={a: f"avail_{a}" for a in ["tube", "bus", "car", "bike"]},
        individual_id="person_id",
    )
    model = MultinomialLogit.from_formula(
        {
            "tube": "ASC_TUBE + B_TIME * time + B_COST * cost",
            "bus": "ASC_BUS + B_TIME * time + B_COST * cost",
            "car": "ASC_CAR + B_TIME * time + B_COST * cost",
            "bike": "B_TIME * time + B_COST * cost",
        },
        max_iter=60,
    )
    result = model.fit(data, cov_type="robust")
    assert len(result.param_names) == 5
    assert np.isfinite(result.values).all()

