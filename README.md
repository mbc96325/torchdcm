# TorchDCM

<p align="center">
  <img src="https://raw.githubusercontent.com/mbc96325/torchdcm-paper/main/docs/assets/torchdcm-logo.png" alt="TorchDCM logo" width="86%">
</p>

<p align="center">
  <b>PyTorch-first discrete choice model estimation and econometric inference.</b>
</p>

<p align="center">
  <a href="#quick-start">Quick Start</a> |
  <a href="#model-zoo">Model Zoo</a> |
  <a href="#installation">Installation</a> |
  <a href="#benchmarks-and-validation">Benchmarks</a> |
  <a href="#development">Development</a>
</p>

<p align="center">
  <a href="https://pypi.org/project/torchdcm/"><img alt="PyPI" src="https://img.shields.io/pypi/v/torchdcm"></a>
  <a href="https://pypi.org/project/torchdcm/"><img alt="Python" src="https://img.shields.io/badge/python-3.10%2B-blue"></a>
  <a href="https://pytorch.org/"><img alt="PyTorch" src="https://img.shields.io/badge/backend-PyTorch-ee4c2c"></a>
  <a href="#model-zoo"><img alt="Models" src="https://img.shields.io/badge/models-MNL%20%7C%20NL%20%7C%20Mixed%20Logit%20%7C%20Hybrid-2b9348"></a>
  <a href="https://github.com/mbc96325/torchdcm-paper/blob/main/validation/README.md"><img alt="Benchmarks" src="https://img.shields.io/badge/benchmarks-Biogeme%20%7C%20Apollo%20%7C%20mlogit%20%7C%20xlogit-6c63ff"></a>
  <a href="#development"><img alt="Status" src="https://img.shields.io/badge/status-research%20prototype-f59f00"></a>
</p>

TorchDCM is the importable Python package for discrete choice model estimation.
This repository is intentionally package-first: it keeps the reusable
`torchdcm` implementation, unit tests, examples, and packaging metadata in one
small repo that users can install and import directly.

The software-paper repository is separate and contains public benchmark data,
validation wrappers, plots, comparison tables, generated results, and LaTeX:

<p align="center">
  <a href="https://github.com/mbc96325/torchdcm-paper/tree/main/validation"><b>torchdcm-paper: validation, benchmarks, datasets, and manuscript</b></a>
</p>

## Why TorchDCM

| Goal | What TorchDCM Provides |
| --- | --- |
| PyTorch-native estimation | Vectorized likelihoods written around tensors and automatic differentiation. |
| Econometric outputs | Classic, robust, and cluster covariance; WTP and elasticity helpers. |
| Model coverage | MNL, NL, CNL, mixed logit, WTP-space, latent class, scaled, ordered, and hybrid choice. |
| Reusable package | Clean import surface with examples and package-level tests. |
| Benchmark companion | Full comparisons live in `torchdcm-paper` and import this package. |

## Installation

```bash
python -m pip install torchdcm
```

For local development:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -e ".[dev]"
pytest
```

## Quick Start

```python
import torch

from torchdcm import Beta, ChoiceDataset, MultinomialLogit, UtilitySpec
from torchdcm.datasets import make_swissmetro_like

df = make_swissmetro_like(n_obs=300, seed=7)
data = ChoiceDataset.from_wide(
    df,
    alternatives=["TRAIN", "SM", "CAR"],
    choice="choice",
    variables={
        "time": {"TRAIN": "time_train", "SM": "time_sm", "CAR": "time_car"},
        "cost": {"TRAIN": "cost_train", "SM": "cost_sm", "CAR": "cost_car"},
    },
    availability={
        "TRAIN": "avail_train",
        "SM": "avail_sm",
        "CAR": "avail_car",
    },
    individual_id="person_id",
)

spec = UtilitySpec()
spec.utility(
    "TRAIN",
    Beta("ASC_TRAIN")
    + Beta("B_TIME", init=-0.01) * "time"
    + Beta("B_COST", init=-0.1) * "cost",
)
spec.utility(
    "SM",
    Beta("B_TIME", init=-0.01) * "time"
    + Beta("B_COST", init=-0.1) * "cost",
)
spec.utility(
    "CAR",
    Beta("ASC_CAR")
    + Beta("B_TIME", init=-0.01) * "time"
    + Beta("B_COST", init=-0.1) * "cost",
)

device = "cuda" if torch.cuda.is_available() else "cpu"
result = MultinomialLogit(spec, device=device).fit(data, cov_type="cluster", groups="person_id")
print(result.summary())
```

`summary()` renders an organized console report covering the model and data,
convergence diagnostics, fit statistics, inference, alternative shares, and
parameter estimates. The same structured report can be inspected as tables or
saved as a reproducible artifact directory:

```python
report = result.report(cov_type="cluster", confidence_level=0.95)
parameter_table = report.parameters

result.save_report(
    "outputs/swissmetro_mnl",
    formats=["html", "json", "csv", "latex", "text"],
)
```

The output directory contains a readable HTML report, a machine-readable JSON
record, parameter/covariance/correlation CSV files, a LaTeX fragment, and a
plain-text summary.

All estimators accept a standard PyTorch-style `device` argument. Passing
`device="cuda"` moves estimation, prediction, simulated likelihoods, and
covariance calculations for that model to CUDA when your PyTorch installation
has GPU support.

## Model Zoo

The current package includes:

- multinomial logit / conditional logit;
- nested logit and cross-nested logit;
- mixed logit, WTP-space mixed logit, and error-components logit;
- latent-class logit;
- scaled and covariate-scaled multinomial logit;
- ordered logit and ordered probit;
- hybrid choice with latent variables and Gaussian measurement indicators;
- classic, robust, and cluster covariance estimates;
- WTP and elasticity helpers.

## Benchmarks and Validation

The public
[`torchdcm-paper`](https://github.com/mbc96325/torchdcm-paper) repository
contains the validation system used to generate the software-paper results.
Each row below links the executable runner to its committed output rather than
to a separate illustrative example.

| Benchmark | Scope | Runner | Published output |
| --- | --- | --- | --- |
| Controlled synthetic data | MNL, NL, and MixL sweeps over sample size, alternatives, coefficients, correlation, and stress cases | [`compare_generated_choice_battery.py`](https://github.com/mbc96325/torchdcm-paper/blob/main/validation/benchmarks/compare_generated_choice_battery.py) | [MNL](https://github.com/mbc96325/torchdcm-paper/blob/main/validation/generated/generated_choice_battery_controlled_office.md), [NL](https://github.com/mbc96325/torchdcm-paper/blob/main/validation/generated/generated_choice_battery_table4_nl_office.md), [MixL](https://github.com/mbc96325/torchdcm-paper/blob/main/validation/generated/generated_choice_battery_table4_mixl_office.md) |
| Real-data MNL | TorchDCM, SciPy, Biogeme, Apollo, `mlogit`, `gmnl`, and `xlogit` | [`run_solver_attempt_matrix.py`](https://github.com/mbc96325/torchdcm-paper/blob/main/validation/benchmarks/run_solver_attempt_matrix.py) | [`solver_attempt_matrix_mnl_single_core_office.md`](https://github.com/mbc96325/torchdcm-paper/blob/main/validation/generated/solver_attempt_matrix_mnl_single_core_office.md) |
| Real-data NL | TorchDCM, Biogeme, and Apollo | [`compare_real_nested_logit_battery.py`](https://github.com/mbc96325/torchdcm-paper/blob/main/validation/benchmarks/compare_real_nested_logit_battery.py) | [`nested_real_battery_single_core_office.md`](https://github.com/mbc96325/torchdcm-paper/blob/main/validation/generated/nested_real_battery_single_core_office.md) |
| Real-data MixL | TorchDCM, Biogeme, and Apollo with aligned model specifications | [`compare_real_mixed_logit_battery.py`](https://github.com/mbc96325/torchdcm-paper/blob/main/validation/benchmarks/compare_real_mixed_logit_battery.py) | [`mixed_real_battery_apollo_office.md`](https://github.com/mbc96325/torchdcm-paper/blob/main/validation/generated/mixed_real_battery_apollo_office.md) |
| CPU--GPU scaling | MNL, NL, and MixL device comparisons | [`compare_torch_device_stress.py`](https://github.com/mbc96325/torchdcm-paper/blob/main/validation/benchmarks/compare_torch_device_stress.py) | [`torch_device_stress_battery.md`](https://github.com/mbc96325/torchdcm-paper/blob/main/validation/generated/torch_device_stress_battery.md) |
| Ordered responses | Ordered logit and ordered probit parity with Biogeme | [`compare_ordered_estimators.py`](https://github.com/mbc96325/torchdcm-paper/blob/main/validation/benchmarks/compare_ordered_estimators.py) | [Ordered logit](https://github.com/mbc96325/torchdcm-paper/blob/main/validation/generated/ordered_logit_single_core_office.json), [ordered probit](https://github.com/mbc96325/torchdcm-paper/blob/main/validation/generated/ordered_probit_single_core_office.json) |

See the [validation guide](https://github.com/mbc96325/torchdcm-paper/blob/main/validation/README.md),
[dataset catalog](https://github.com/mbc96325/torchdcm-paper/blob/main/datasets/dataset_index.csv),
and [manuscript](https://github.com/mbc96325/torchdcm-paper/blob/main/paper/main.pdf)
for the complete reproducibility workflow.

## Repository Layout

| Path | Purpose |
| --- | --- |
| `torchdcm/` | Importable package implementation. |
| `tests/` | Package-level unit tests. |
| `examples/` | Minimal runnable usage examples. |
| `docs/assets/` | GitHub README logo and cover assets. |
| `pyproject.toml` | Packaging metadata and dependencies. |

## Development

```bash
python -m pip install -e ".[dev]"
pytest
```

Benchmark and manuscript work should happen in the companion repository:

```bash
git clone https://github.com/mbc96325/torchdcm-paper.git
```
