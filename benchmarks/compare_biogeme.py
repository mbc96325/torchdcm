from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd

from torchdcm import Beta, ChoiceDataset, MultinomialLogit, UtilitySpec
from torchdcm.datasets import make_london_like, make_swissmetro_like


def build_case(case: str, n_obs: int, seed: int):
    if case == "swissmetro":
        df = make_swissmetro_like(n_obs=n_obs, seed=seed)
        alternatives = ["TRAIN", "SM", "CAR"]
        data = ChoiceDataset.from_wide(
            df,
            alternatives=alternatives,
            choice="choice",
            variables={
                "time": {"TRAIN": "time_train", "SM": "time_sm", "CAR": "time_car"},
                "cost": {"TRAIN": "cost_train", "SM": "cost_sm", "CAR": "cost_car"},
            },
            availability={"TRAIN": "avail_train", "SM": "avail_sm", "CAR": "avail_car"},
            individual_id="person_id",
        )
        spec = UtilitySpec()
        spec.utility("TRAIN", Beta("ASC_TRAIN") + Beta("B_TIME", init=-0.01) * "time" + Beta("B_COST", init=-0.1) * "cost")
        spec.utility("SM", Beta("B_TIME", init=-0.01) * "time" + Beta("B_COST", init=-0.1) * "cost")
        spec.utility("CAR", Beta("ASC_CAR") + Beta("B_TIME", init=-0.01) * "time" + Beta("B_COST", init=-0.1) * "cost")
        return df, data, spec, alternatives
    if case == "london":
        df = make_london_like(n_obs=n_obs, seed=seed)
        alternatives = ["tube", "bus", "car", "bike"]
        data = ChoiceDataset.from_wide(
            df,
            alternatives=alternatives,
            choice="choice",
            variables={
                "time": {a: f"time_{a}" for a in alternatives},
                "cost": {a: f"cost_{a}" for a in alternatives},
            },
            availability={a: f"avail_{a}" for a in alternatives},
            individual_id="person_id",
        )
        spec = UtilitySpec()
        spec.utility("tube", Beta("ASC_TUBE") + Beta("B_TIME", init=-0.05) * "time" + Beta("B_COST", init=-0.5) * "cost")
        spec.utility("bus", Beta("ASC_BUS") + Beta("B_TIME", init=-0.05) * "time" + Beta("B_COST", init=-0.5) * "cost")
        spec.utility("car", Beta("ASC_CAR") + Beta("B_TIME", init=-0.05) * "time" + Beta("B_COST", init=-0.5) * "cost")
        spec.utility("bike", Beta("B_TIME", init=-0.05) * "time" + Beta("B_COST", init=-0.5) * "cost")
        return df, data, spec, alternatives
    raise ValueError(f"Unknown case: {case}")


def run_torchdcm(data, spec):
    result = MultinomialLogit(spec).fit(data, cov_type="robust")
    return result


def run_biogeme(
    df: pd.DataFrame,
    alternatives: list[str],
    result_names: list[str],
    initial_values: dict[str, float] | None = None,
):
    mpl_cache = Path("/private/tmp/torchdcm_matplotlib")
    mpl_cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_cache))
    xdg_cache = Path("/private/tmp/torchdcm_cache")
    xdg_cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("XDG_CACHE_HOME", str(xdg_cache))
    try:
        import biogeme.biogeme as bio
        import biogeme.database as db
        from biogeme import models
        from biogeme.expressions import Beta, Variable
    except ImportError as exc:
        raise RuntimeError("Install the optional dependency with `pip install 'torchdcm[biogeme]'`.") from exc

    code_by_alt = {alt: i + 1 for i, alt in enumerate(alternatives)}
    wide_df = df.copy()
    wide_df["choice_code"] = wide_df["choice"].map(code_by_alt).astype(int)
    wide_df = wide_df.drop(columns=["choice"])
    bool_columns = wide_df.select_dtypes(include=["bool"]).columns
    wide_df[bool_columns] = wide_df[bool_columns].astype(int)
    database = db.Database("torchdcm_compare", wide_df)
    chosen_alt = Variable("choice_code")

    initial_values = initial_values or {}
    betas = {name: Beta(name, initial_values.get(name, 0.0), None, None, 0) for name in result_names}
    utility = {}
    av = {}
    for alt_name, code in code_by_alt.items():
        time = Variable(f"time_{alt_name.lower()}")
        cost = Variable(f"cost_{alt_name.lower()}")
        utility[code] = betas.get(f"ASC_{alt_name.upper()}", 0) + betas["B_TIME"] * time + betas["B_COST"] * cost
        av[code] = Variable(f"avail_{alt_name.lower()}")

    logprob = models.loglogit(utility, av, chosen_alt)
    biogeme = bio.BIOGEME(database, logprob)
    model_token = "_".join(alternatives).replace("-", "_")
    biogeme.model_name = f"torchdcm_compare_{model_token}_{len(df)}"
    biogeme.biogeme_parameters.set_value("save_iterations", False)
    estimates = biogeme.estimate()
    return {
        "params": estimates.get_beta_values(),
        "loglike": estimates.final_log_likelihood,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", choices=["swissmetro", "london"], default="swissmetro")
    parser.add_argument("--n-obs", type=int, default=300)
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument("--biogeme", action="store_true", help="Also run optional Biogeme estimate.")
    args = parser.parse_args()

    _, data, spec, _ = build_case(args.case, args.n_obs, args.seed)
    result = run_torchdcm(data, spec)
    print(result.summary())
    if args.biogeme:
        df, _, _, alternatives = build_case(args.case, args.n_obs, args.seed)
        print("\nBiogeme comparison")
        print(f"loglike_torchdcm: {result.loglike:.10f}")
        torch_params = dict(zip(result.param_names, result.values))
        biogeme_result = run_biogeme(df, alternatives, result.param_names, initial_values=torch_params)
        print(f"loglike_biogeme:  {biogeme_result['loglike']:.10f}")
        print(f"loglike_diff:     {result.loglike - biogeme_result['loglike']:.3e}")
        print(f"{'parameter':<18}{'torchdcm':>14}{'biogeme':>14}{'diff':>14}")
        for name in result.param_names:
            bio_value = biogeme_result["params"][name]
            torch_value = torch_params[name]
            print(f"{name:<18}{torch_value:>14.8g}{bio_value:>14.8g}{torch_value - bio_value:>14.3e}")


if __name__ == "__main__":
    main()
