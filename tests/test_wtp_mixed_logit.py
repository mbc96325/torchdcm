import torch

from torchdcm import Beta, ChoiceDataset, WTPCoefficient, WTPMixedLogit, UtilitySpec
from torchdcm.datasets import make_swissmetro_like


def swissmetro_panel_data(n_obs=12):
    df = make_swissmetro_like(n_obs=n_obs, seed=37)
    df["person_id"] = [index // 3 for index in range(len(df))]
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


def asc_spec():
    spec = UtilitySpec()
    spec.utility("TRAIN", Beta("ASC_TRAIN", init=0.2))
    spec.utility("SM", Beta("ASC_SM", init=0.0, fixed=True))
    spec.utility("CAR", Beta("ASC_CAR", init=0.4))
    return spec


def test_wtp_mixed_logit_parameter_order_and_probabilities():
    data = swissmetro_panel_data(9)
    draws = torch.tensor([[-1.0], [0.0], [1.0]], dtype=torch.float64)
    model = WTPMixedLogit(
        asc_spec(),
        cost=Beta("B_COST", init=-1.2),
        cost_variable="cost",
        wtp_coefficients=[WTPCoefficient("WTP_TIME", "time", init=0.5, sigma_init=0.2)],
        draws=draws,
    )
    compiled = model.compile(data)
    params = torch.tensor([0.2, 0.4, -1.2, 0.5, 0.2], dtype=torch.float64)
    probs = model.predict_proba(data, params, compiled)

    assert compiled.free_names == ["ASC_TRAIN", "ASC_CAR", "B_COST", "WTP_TIME", "SIGMA_WTP_TIME"]
    assert probs.shape == (data.n_rows,)
    for obs in range(data.n_obs):
        start = int(data.obs_ptr[obs])
        end = int(data.obs_ptr[obs + 1])
        assert torch.allclose(probs[start:end].sum(), torch.tensor(1.0, dtype=torch.float64))
    assert torch.isfinite(model.loglike(params, data, compiled))


def test_wtp_mixed_logit_utility_formula_matches_manual():
    data = swissmetro_panel_data(3)
    draws = torch.tensor([[0.0]], dtype=torch.float64)
    model = WTPMixedLogit(
        asc_spec(),
        cost=Beta("B_COST", init=-2.0),
        cost_variable="cost",
        wtp_coefficients=[WTPCoefficient("WTP_TIME", "time", init=0.5, sigma_init=0.1)],
        draws=draws,
    )
    compiled = model.compile(data)
    params = torch.tensor([0.2, 0.4, -2.0, 0.5, 0.1], dtype=torch.float64)
    utility = model.utilities_by_draw(params, data, compiled).squeeze(1)

    asc = torch.zeros(data.n_rows, dtype=torch.float64)
    asc[data.alt_id == 0] = 0.2
    asc[data.alt_id == 2] = 0.4
    expected = asc - 2.0 * data.x_alt["cost"] - 2.0 * 0.5 * data.x_alt["time"]

    assert torch.allclose(utility, expected)


def test_wtp_mixed_logit_correlated_wtp_draws():
    data = swissmetro_panel_data(6)
    draws = torch.tensor([[-1.0, -0.5], [0.0, 0.25], [1.0, 0.75]], dtype=torch.float64)
    model = WTPMixedLogit(
        asc_spec(),
        cost=Beta("B_COST", init=-1.0),
        cost_variable="cost",
        wtp_coefficients=[
            WTPCoefficient("WTP_TIME", "time", init=0.5, sigma_init=0.2),
            WTPCoefficient("WTP_COST_SHIFT", "cost", init=0.1, sigma_init=0.3),
        ],
        draws=draws,
        correlated=True,
    )
    compiled = model.compile(data)
    params = torch.tensor([0.2, 0.4, -1.0, 0.5, 0.1, 0.2, 0.3, 0.25], dtype=torch.float64)
    _, _, wtp, sigmas, chol = model._split_params(params, compiled)
    cholesky = model._cholesky_factor(sigmas, chol, compiled)
    drawn = model._drawn_wtp(wtp, sigmas, chol, compiled)

    assert compiled.chol_offdiag_names == ["CHOL_WTP_COST_SHIFT__WTP_TIME"]
    assert torch.allclose(cholesky, torch.tensor([[0.2, 0.0], [0.25, 0.3]], dtype=torch.float64))
    assert torch.allclose(drawn, wtp.unsqueeze(0) + draws @ cholesky.T)
    assert torch.isfinite(model.loglike(params, data, compiled))
