import math

import numpy as np
import torch

from torchdcm import Beta, ChoiceDataset, MultinomialLogit, Nest, NestedLogit, UtilitySpec
from torchdcm.datasets import make_swissmetro_like


def swissmetro_data(n_obs=100):
    df = make_swissmetro_like(n_obs=n_obs, seed=21)
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


def swissmetro_nests(lambda_public=0.8, fixed=False):
    return {
        "PUBLIC": Nest(["TRAIN", "SM"], init=lambda_public, fixed=fixed),
        "PRIVATE": Nest(["CAR"], init=1.0, fixed=True),
    }


def test_nested_logit_with_unit_lambdas_matches_mnl():
    data = swissmetro_data(80)
    spec = swissmetro_spec()
    mnl = MultinomialLogit(spec)
    nl = NestedLogit(spec, swissmetro_nests(lambda_public=1.0, fixed=True))
    compiled_mnl = mnl.compile(data)
    compiled_nl = nl.compile(data)
    params = compiled_mnl.free_initial

    assert compiled_nl.free_names == compiled_mnl.free_names
    assert torch.allclose(nl.loglike(params, data, compiled_nl), mnl.loglike(params, data, compiled_mnl))
    assert torch.allclose(
        nl.predict_proba(data, params, compiled_nl),
        mnl.predict_proba(data, params, compiled_mnl),
    )


def test_nested_logit_probabilities_sum_to_one():
    data = swissmetro_data(60)
    model = NestedLogit(swissmetro_spec(), swissmetro_nests(lambda_public=0.65))
    compiled = model.compile(data)
    params = torch.cat([compiled.free_initial, torch.tensor([0.65], dtype=torch.float64)])
    probs = model.predict_proba(data, params, compiled)

    for obs in range(data.n_obs):
        start = int(data.obs_ptr[obs])
        end = int(data.obs_ptr[obs + 1])
        assert torch.allclose(probs[start:end].sum(), torch.tensor(1.0, dtype=torch.float64))
        assert torch.all(probs[start:end][~data.availability[start:end]] == 0)


def test_nested_logit_fit_returns_lambda_and_covariance():
    data = swissmetro_data(90)
    result = NestedLogit(swissmetro_spec(), swissmetro_nests(lambda_public=0.8), max_iter=80).fit(data)

    assert math.isfinite(result.loglike)
    assert result.param_names == ["ASC_TRAIN", "B_TIME", "B_COST", "ASC_CAR", "LAMBDA_PUBLIC"]
    assert result.cov_params("classic").shape == (5, 5)
    assert 0.0001 < result.values[-1] <= 1.0
    assert len(result.predict()) == data.n_obs
    assert result.predict_proba().shape == (data.n_rows,)
    assert np.isfinite(result.bse).all()
