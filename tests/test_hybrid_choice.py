import math

import pandas as pd
import torch

from torchdcm import (
    Beta,
    ChoiceDataset,
    ChoiceLatentEffect,
    ContinuousIndicator,
    HybridChoiceModel,
    LatentVariable,
    UtilitySpec,
)


def toy_data():
    df = pd.DataFrame(
        {
            "choice": ["B", "A", "B", "A"],
            "x_a": [0.0, 0.0, 0.0, 0.0],
            "x_b": [1.0, -0.5, 0.4, -1.2],
            "z": [0.0, 1.0, -1.0, 0.5],
            "attitude": [0.2, 0.4, -0.1, 0.0],
        }
    )
    return ChoiceDataset.from_wide(
        df,
        alternatives=["A", "B"],
        choice="choice",
        variables={"x": {"A": "x_a", "B": "x_b"}},
        obs_variables={"z": "z", "attitude": "attitude"},
    )


def base_spec():
    spec = UtilitySpec()
    spec.utility("A", Beta("ASC_A", init=0.0, fixed=True))
    spec.utility("B", Beta("ASC_B", init=0.2) + Beta("B_X", init=0.5, fixed=True) * "x")
    return spec


def test_choice_dataset_keeps_observation_level_variables():
    data = toy_data()

    assert set(data.x_obs) == {"z", "attitude"}
    assert data.x_obs["z"].shape == (data.n_obs,)
    assert data.to(dtype=torch.float32).x_obs["attitude"].dtype == torch.float32


def test_hybrid_choice_loglike_matches_manual_single_draw():
    data = toy_data()
    model = HybridChoiceModel(
        base_spec(),
        latent_variables=[
            LatentVariable(
                "ATT",
                intercept=Beta("G0", init=0.1),
                coefficients={"z": Beta("G_Z", init=0.2)},
                sigma_init=1.0,
                sigma_fixed=True,
            )
        ],
        choice_effects=[ChoiceLatentEffect("B", "ATT", Beta("B_ATT", init=0.3))],
        indicators=[
            ContinuousIndicator(
                "attitude",
                "ATT",
                intercept=0.0,
                loading=1.0,
                sigma_init=1.0,
                sigma_fixed=True,
            )
        ],
        draws=torch.zeros((1, 1), dtype=torch.float64),
    )
    compiled = model.compile(data)
    params = torch.tensor([0.2, 0.1, 0.2, 0.3], dtype=torch.float64)

    latent = 0.1 + 0.2 * data.x_obs["z"]
    utility_b = 0.2 + 0.5 * data.x_alt["x"][data.alt_id == 1] + 0.3 * latent
    p_b = torch.sigmoid(utility_b)
    chosen_is_b = torch.tensor([1.0, 0.0, 1.0, 0.0], dtype=torch.float64)
    choice_log = chosen_is_b * torch.log(p_b) + (1.0 - chosen_is_b) * torch.log1p(-p_b)
    residual = data.x_obs["attitude"] - latent
    meas_log = -0.5 * math.log(2.0 * math.pi) - 0.5 * residual.square()
    expected = (choice_log + meas_log).sum()

    assert compiled.free_names == ["ASC_B", "G0", "G_Z", "B_ATT"]
    assert torch.allclose(model.loglike(params, data, compiled), expected)


def test_hybrid_choice_predict_proba_and_fit():
    data = toy_data()
    model = HybridChoiceModel(
        base_spec(),
        latent_variables=[LatentVariable("ATT", intercept=0.0, sigma_init=1.0, sigma_fixed=True)],
        choice_effects=[ChoiceLatentEffect("B", "ATT", 0.0)],
        indicators=[
            ContinuousIndicator(
                "attitude",
                "ATT",
                intercept=0.0,
                loading=1.0,
                sigma_init=1.0,
                sigma_fixed=True,
            )
        ],
        draws=torch.tensor([[-1.0], [0.0], [1.0]], dtype=torch.float64),
        max_iter=5,
    )
    compiled = model.compile(data)
    params = torch.tensor([0.2], dtype=torch.float64)
    probs = model.predict_proba(data, params, compiled)
    posterior_probs = model.predict_proba(data, params, compiled, condition_on_indicators=True)
    result = model.fit(data, max_iter=3)

    assert probs.shape == (data.n_rows,)
    assert posterior_probs.shape == (data.n_rows,)
    assert result.param_names == ["ASC_B"]
    assert math.isfinite(result.loglike)
    for obs in range(data.n_obs):
        start = int(data.obs_ptr[obs])
        end = int(data.obs_ptr[obs + 1])
        assert torch.allclose(probs[start:end].sum(), torch.tensor(1.0, dtype=torch.float64))
        assert torch.allclose(posterior_probs[start:end].sum(), torch.tensor(1.0, dtype=torch.float64))
