import json

import numpy as np
import torch

from torchdcm import (
    Beta,
    ChoiceDataset,
    MixedLogit,
    MultinomialLogit,
    Nest,
    NestedLogit,
    RandomCoefficient,
    UtilitySpec,
)
from torchdcm.datasets import make_swissmetro_like


def report_data(n_obs: int = 48) -> ChoiceDataset:
    frame = make_swissmetro_like(n_obs=n_obs, seed=81)
    return ChoiceDataset.from_wide(
        frame,
        alternatives=["TRAIN", "SM", "CAR"],
        choice="choice",
        variables={
            "time": {"TRAIN": "time_train", "SM": "time_sm", "CAR": "time_car"},
            "cost": {"TRAIN": "cost_train", "SM": "cost_sm", "CAR": "cost_car"},
        },
        availability={"TRAIN": "avail_train", "SM": "avail_sm", "CAR": "avail_car"},
        individual_id="person_id",
    )


def report_spec(*, fixed_car: bool = False) -> UtilitySpec:
    spec = UtilitySpec()
    spec.utility(
        "TRAIN",
        Beta("ASC_TRAIN") + Beta("B_TIME", init=-0.01) * "time" + Beta("B_COST", init=-0.1) * "cost",
    )
    spec.utility(
        "SM",
        Beta("B_TIME", init=-0.01) * "time" + Beta("B_COST", init=-0.1) * "cost",
    )
    spec.utility(
        "CAR",
        Beta("ASC_CAR", fixed=fixed_car) + Beta("B_TIME", init=-0.01) * "time" + Beta("B_COST", init=-0.1) * "cost",
    )
    return spec


def test_structured_report_and_all_exports(tmp_path):
    data = report_data()
    result = MultinomialLogit(report_spec(fixed_car=True), max_iter=35).fit(data, cov_type="robust")

    report = result.report(confidence_level=0.90)
    text = report.to_text()
    assert "Data summary" in text
    assert "Estimation and convergence" in text
    assert "Parameter estimates" in text
    assert "B_TIME" in text
    assert report.sections["Data summary"]["Alternative rows"] == data.n_rows
    assert report.sections["Data summary"]["Choice-set structure"] == "Balanced"
    assert set(report.alternatives["Alternative"]) == {"TRAIN", "SM", "CAR"}
    assert result.parameter_table().loc[result.parameter_table()["Parameter"] == "ASC_CAR", "Status"].item() == "Fixed"

    output = tmp_path / "mnl_report"
    written = result.save_report(output)
    assert set(written) == {"html", "json", "csv", "latex", "text"}
    assert (output / "report.html").is_file()
    assert (output / "result.json").is_file()
    assert (output / "parameters.csv").is_file()
    assert (output / "covariance.csv").is_file()
    assert (output / "correlation.csv").is_file()
    assert (output / "report.tex").is_file()
    assert (output / "summary.txt").is_file()

    payload = json.loads((output / "result.json").read_text(encoding="utf-8"))
    assert payload["schema_version"] == "1.0"
    assert payload["covariance_type"] == "robust"
    assert len(payload["parameters"]) == len(result.param_names) + 1
    assert "<h2>Model fit</h2>" in (output / "report.html").read_text(encoding="utf-8")


def test_nested_report_uses_one_as_lambda_h0_value():
    data = report_data(36)
    nests = {
        "PUBLIC": Nest(["TRAIN", "SM"], init=0.8),
        "PRIVATE": Nest(["CAR"], init=1.0, fixed=True),
    }
    result = NestedLogit(report_spec(), nests, max_iter=20).fit(data)
    report = result.report()

    assert report.sections["Model specification"]["Nests"]["PUBLIC"] == ["TRAIN", "SM"]
    lambda_row = report.parameters.loc[report.parameters["Parameter"] == "LAMBDA_PUBLIC"].iloc[0]
    assert lambda_row["Group"] == "Nest parameters"
    assert lambda_row["H₀ value"] == 1.0
    fixed_row = report.parameters.loc[report.parameters["Parameter"] == "LAMBDA_PRIVATE"].iloc[0]
    assert fixed_row["Status"] == "Fixed"


def test_mixed_logit_report_records_draw_configuration():
    data = report_data(24)
    draws = torch.tensor([[-1.0], [-0.25], [0.25], [1.0]], dtype=torch.float64)
    model = MixedLogit(
        report_spec(),
        [RandomCoefficient("B_TIME", sigma_init=0.2)],
        draws=draws,
        panel=True,
        max_iter=3,
    )
    result = model.fit(data)
    report = result.report()

    specification = report.sections["Model specification"]
    assert specification["Random coefficients"] == {"B_TIME": "normal"}
    assert specification["Panel integration"] is True
    assert report.sections["Estimation and convergence"]["Simulation draws"] == 4
    sigma = report.parameters.loc[report.parameters["Parameter"] == "SIGMA_B_TIME"].iloc[0]
    assert sigma["Group"] == "Random-coefficient scales"
    assert np.isfinite(sigma["Estimate"])
