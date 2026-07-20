import torch

from torchdcm import Beta, ChoiceDataset, ErrorComponent, ErrorComponentsLogit, MixedLogit, RandomCoefficient, UtilitySpec
from torchdcm.datasets import make_swissmetro_like


def swissmetro_panel_data(n_obs=24):
    df = make_swissmetro_like(n_obs=n_obs, seed=31)
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


def swissmetro_spec():
    spec = UtilitySpec()
    spec.utility("TRAIN", Beta("ASC_TRAIN") + Beta("B_TIME", init=-0.01) * "time" + Beta("B_COST", init=-0.1) * "cost")
    spec.utility("SM", Beta("B_TIME", init=-0.01) * "time" + Beta("B_COST", init=-0.1) * "cost")
    spec.utility("CAR", Beta("ASC_CAR") + Beta("B_TIME", init=-0.01) * "time" + Beta("B_COST", init=-0.1) * "cost")
    return spec


def swissmetro_error_component_spec():
    spec = swissmetro_spec()
    spec.utility(
        "TRAIN",
        spec.utilities["TRAIN"] + Beta("EC_PUBLIC", fixed=True) * "ec_public_loading",
    )
    spec.utility(
        "SM",
        spec.utilities["SM"] + Beta("EC_PUBLIC", fixed=True) * "ec_public_loading",
    )
    return spec


def test_choice_dataset_exposes_panel_structure():
    data = swissmetro_panel_data(12)
    panel = data.panel_structure()

    assert data.has_panel
    assert data.n_individuals == 4
    assert panel.n_obs == data.n_obs
    assert panel.n_units == 4
    assert panel.unit_ids == [0, 1, 2, 3]


def test_panel_sum_by_unit_preserves_tail_dimensions():
    data = swissmetro_panel_data(9)
    panel = data.panel_structure()
    values = torch.arange(data.n_obs * 2, dtype=torch.float64).reshape(data.n_obs, 2)
    summed = panel.sum_by_unit(values)

    assert summed.shape == (3, 2)
    assert torch.allclose(summed[0], values[:3].sum(dim=0))
    assert torch.allclose(summed[1], values[3:6].sum(dim=0))
    assert torch.allclose(summed[2], values[6:9].sum(dim=0))


def test_mixed_logit_panel_likelihood_uses_panel_structure():
    data = swissmetro_panel_data(12)
    draws = torch.tensor([[-1.0], [0.0], [1.0]], dtype=torch.float64)
    model = MixedLogit(swissmetro_spec(), [RandomCoefficient("B_TIME", sigma_init=0.2)], draws=draws, panel=True)
    compiled = model.compile(data)
    params = torch.cat([compiled.free_initial, torch.tensor([0.2], dtype=torch.float64)])

    obs_log_prob = model._log_prob_per_obs_draw(params, data, compiled)
    manual = data.panel_structure().logmeanexp_by_unit(obs_log_prob)

    assert torch.allclose(model.loglike_per_unit(params, data, compiled), manual)
    assert torch.allclose(model.loglike(params, data, compiled), manual.sum())


def test_mixed_logit_lognormal_random_coefficients():
    data = swissmetro_panel_data(6)
    draws = torch.tensor([[-1.0, -1.0], [0.0, 0.0], [1.0, 1.0]], dtype=torch.float64)
    model = MixedLogit(
        swissmetro_spec(),
        [
            RandomCoefficient("B_TIME", sigma_init=0.2, distribution="negative_lognormal"),
            RandomCoefficient("B_COST", sigma_init=0.3, distribution="lognormal"),
        ],
        draws=draws,
    )
    compiled = model.compile(data)
    params = torch.cat([compiled.free_initial, torch.tensor([0.2, 0.3], dtype=torch.float64)])
    means, sigmas, _ = model._split_natural_params(params, compiled)
    drawn = model._drawn_betas(params, compiled)
    time_index = compiled.beta_names.index("B_TIME")
    cost_index = compiled.beta_names.index("B_COST")

    expected_time = -torch.exp(means[time_index] + draws[:, 0] * sigmas[0])
    expected_cost = torch.exp(means[cost_index] + draws[:, 1] * sigmas[1])

    assert torch.allclose(drawn[:, time_index], expected_time)
    assert torch.allclose(drawn[:, cost_index], expected_cost)
    assert torch.isfinite(model.loglike(params, data, compiled))


def test_mixed_logit_correlated_random_coefficients():
    data = swissmetro_panel_data(6)
    draws = torch.tensor([[-1.0, -0.5], [0.0, 0.25], [1.0, 0.75]], dtype=torch.float64)
    model = MixedLogit(
        swissmetro_spec(),
        [RandomCoefficient("B_TIME", sigma_init=0.2), RandomCoefficient("B_COST", sigma_init=0.3)],
        draws=draws,
        correlated=True,
    )
    compiled = model.compile(data)
    assert compiled.chol_offdiag_names == ["CHOL_B_COST__B_TIME"]

    params = torch.cat([compiled.free_initial, torch.tensor([0.2, 0.3, 0.4], dtype=torch.float64)])
    means, sigmas, chol_offdiag = model._split_natural_params(params, compiled)
    cholesky = model._cholesky_factor(sigmas, chol_offdiag, compiled)
    drawn = model._drawn_betas(params, compiled)
    time_index = compiled.beta_names.index("B_TIME")
    cost_index = compiled.beta_names.index("B_COST")
    latent_noise = draws @ cholesky.T

    assert torch.allclose(cholesky, torch.tensor([[0.2, 0.0], [0.4, 0.3]], dtype=torch.float64))
    assert torch.allclose(drawn[:, time_index], means[time_index] + latent_noise[:, 0])
    assert torch.allclose(drawn[:, cost_index], means[cost_index] + latent_noise[:, 1])
    assert torch.isfinite(model.loglike(params, data, compiled))


def test_mixed_logit_decomposed_utility_matches_full_product_when_all_betas_random():
    data = swissmetro_panel_data(6)
    spec = UtilitySpec()
    spec.utility("TRAIN", Beta("B_TIME", init=-0.01) * "time" + Beta("B_COST", init=-0.1) * "cost")
    spec.utility("SM", Beta("B_TIME", init=-0.01) * "time" + Beta("B_COST", init=-0.1) * "cost")
    spec.utility("CAR", Beta("B_TIME", init=-0.01) * "time" + Beta("B_COST", init=-0.1) * "cost")
    draws = torch.tensor([[-1.0, -0.5], [0.0, 0.25], [1.0, 0.75]], dtype=torch.float64)
    model = MixedLogit(
        spec,
        [RandomCoefficient("B_TIME", sigma_init=0.2), RandomCoefficient("B_COST", sigma_init=0.3)],
        draws=draws,
    )
    compiled = model.compile(data)
    params = torch.cat([compiled.free_initial, torch.tensor([0.2, 0.3], dtype=torch.float64)])

    decomposed = model._utility_per_row_draw(params, compiled)
    full_product = compiled.design @ model._drawn_betas(params, compiled).T

    assert compiled.random_beta_indices.numel() == compiled.design.shape[1]
    assert torch.allclose(decomposed, full_product)


def test_mixed_logit_fixed_beta_random_component():
    data = swissmetro_panel_data(6)
    loading = torch.isin(data.alt_id, torch.tensor([0, 1], dtype=torch.long)).to(dtype=torch.float64)
    data = ChoiceDataset(
        obs_ptr=data.obs_ptr,
        alt_id=data.alt_id,
        chosen_row=data.chosen_row,
        x_alt={**data.x_alt, "ec_public_loading": loading},
        weights=data.weights,
        availability=data.availability,
        obs_ids=data.obs_ids,
        alt_names=data.alt_names,
        obs_to_ind=data.obs_to_ind,
        individual_ids=data.individual_ids,
    )
    draws = torch.tensor([[-1.0], [0.0], [1.0]], dtype=torch.float64)
    model = MixedLogit(
        swissmetro_error_component_spec(),
        [RandomCoefficient("EC_PUBLIC", sigma_init=0.4)],
        draws=draws,
    )
    compiled = model.compile(data)
    params = torch.cat([compiled.free_initial, torch.tensor([0.4], dtype=torch.float64)])

    assert "EC_PUBLIC" not in compiled.beta_names
    assert compiled.sigma_names == ["SIGMA_EC_PUBLIC"]
    assert torch.isfinite(model.loglike(params, data, compiled))
    assert model.predict_proba(data, params, compiled).shape == (data.n_rows,)


def test_error_components_logit_matches_manual_mixed_spec():
    data = swissmetro_panel_data(6)
    draws = torch.tensor([[-1.0], [0.0], [1.0]], dtype=torch.float64)
    error_model = ErrorComponentsLogit(
        swissmetro_spec(),
        [ErrorComponent("PUBLIC", ["TRAIN", "SM"], sigma_init=0.4)],
        draws=draws,
    )
    manual_model = MixedLogit(
        swissmetro_error_component_spec(),
        [RandomCoefficient("EC_PUBLIC", sigma_init=0.4)],
        draws=draws,
    )
    manual_data = error_model._augment_data(data)
    error_compiled = error_model.compile(data)
    manual_compiled = manual_model.compile(manual_data)
    params = torch.cat([error_compiled.free_initial, torch.tensor([0.4], dtype=torch.float64)])

    assert error_compiled.free_names == manual_compiled.free_names
    assert torch.allclose(error_model.loglike(params, data, error_compiled), manual_model.loglike(params, manual_data, manual_compiled))
    assert torch.allclose(
        error_model.predict_proba(data, params, error_compiled),
        manual_model.predict_proba(manual_data, params, manual_compiled),
    )
