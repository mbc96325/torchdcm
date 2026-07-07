import inspect

import pytest
import torch

from torchdcm import (
    Beta,
    ChoiceDataset,
    CovariateScaledMultinomialLogit,
    CrossNestedLogit,
    ErrorComponentsLogit,
    HybridChoiceModel,
    LatentClassLogit,
    MixedLogit,
    MultinomialLogit,
    NestedLogit,
    OrderedLogit,
    OrderedProbit,
    RandomCoefficient,
    ScaledMultinomialLogit,
    UtilitySpec,
    WTPMixedLogit,
)
from torchdcm.datasets import make_swissmetro_like


def swissmetro_data(n_obs=60):
    df = make_swissmetro_like(n_obs=n_obs, seed=44)
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


@pytest.mark.parametrize(
    "model_class",
    [
        MultinomialLogit,
        NestedLogit,
        CrossNestedLogit,
        MixedLogit,
        WTPMixedLogit,
        LatentClassLogit,
        ScaledMultinomialLogit,
        CovariateScaledMultinomialLogit,
        OrderedLogit,
        OrderedProbit,
        HybridChoiceModel,
    ],
)
def test_models_accept_device_argument(model_class):
    assert "device" in inspect.signature(model_class).parameters


def test_error_components_forwards_device_argument():
    parameters = inspect.signature(ErrorComponentsLogit).parameters
    assert any(parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in parameters.values())


def test_mnl_uses_requested_device_for_estimation_and_prediction():
    data = swissmetro_data(40)
    model = MultinomialLogit(swissmetro_spec(), device="cpu", max_iter=20)
    result = model.fit(data)
    assert result.params.device.type == "cpu"
    assert result.hessian.device.type == "cpu"
    assert model.predict_proba(data, result.params).device.type == "cpu"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_mixed_logit_uses_cuda_for_simulated_likelihood():
    data = swissmetro_data(40)
    draws = torch.linspace(-1.0, 1.0, steps=16, dtype=torch.float64).reshape(16, 1)
    model = MixedLogit(
        swissmetro_spec(),
        [RandomCoefficient("B_TIME", sigma_init=0.2)],
        draws=draws,
        panel=False,
        device="cuda",
        max_iter=5,
    )
    compiled = model.compile(data)
    assert compiled.design.device.type == "cuda"
    params = torch.cat([compiled.free_initial, compiled.sigma_initial[~compiled.sigma_is_fixed]])
    ll = model.loglike(params, data, compiled)
    probs = model.predict_proba(data, params, compiled)
    assert ll.device.type == "cuda"
    assert probs.device.type == "cuda"
