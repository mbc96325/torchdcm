import numpy as np
import pytest

from benchmarks.compare_biogeme import build_case, run_biogeme, run_torchdcm


pytest.importorskip("biogeme")


def test_swissmetro_mnl_matches_biogeme():
    df, data, spec, alternatives = build_case("swissmetro", n_obs=60, seed=9)
    result = run_torchdcm(data, spec)
    torch_params = dict(zip(result.param_names, result.values))
    biogeme_result = run_biogeme(
        df,
        alternatives,
        result.param_names,
        initial_values=torch_params,
    )
    assert abs(result.loglike - biogeme_result["loglike"]) < 1e-6
    for name in result.param_names:
        assert np.isclose(torch_params[name], biogeme_result["params"][name], atol=1e-4)

