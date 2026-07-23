# TorchDCM

<p align="center">
  <img src="https://raw.githubusercontent.com/mbc96325/torchdcm/main/docs/assets/torchdcm-logo.png" alt="TorchDCM logo" width="86%">
</p>

<p align="center">
  <b>PyTorch-first discrete choice model estimation and econometric inference.</b>
</p>

<p align="center">
  <a href="#installation">Installation</a> |
  <a href="#quick-start">Quick Start</a> |
  <a href="#executed-examples">Examples</a> |
  <a href="#model-zoo">Model Zoo</a> |
  <a href="#development">Development</a>
</p>

<p align="center">
  <a href="https://pypi.org/project/torchdcm/"><img alt="PyPI" src="https://img.shields.io/pypi/v/torchdcm"></a>
  <a href="https://pypi.org/project/torchdcm/"><img alt="Python" src="https://img.shields.io/badge/python-3.10%2B-blue"></a>
  <a href="https://pytorch.org/"><img alt="PyTorch" src="https://img.shields.io/badge/backend-PyTorch-ee4c2c"></a>
  <a href="#model-zoo"><img alt="Models" src="https://img.shields.io/badge/models-MNL%20%7C%20NL%20%7C%20Mixed%20Logit%20%7C%20Hybrid-2b9348"></a>
  <a href="https://github.com/mbc96325/torchdcm-paper"><img alt="Benchmarks" src="https://img.shields.io/badge/benchmarks-Biogeme%20%7C%20Apollo%20%7C%20mlogit%20%7C%20xlogit-6c63ff"></a>
  <a href="#development"><img alt="Status" src="https://img.shields.io/badge/status-research%20prototype-f59f00"></a>
</p>

TorchDCM is the importable Python package for discrete choice model estimation.
This repository is intentionally package-first: it keeps the reusable
`torchdcm` implementation, unit tests, examples, and packaging metadata in one
small repo that users can install and import directly.

The paper's reproducibility repository is separate and contains benchmark
runners, aligned datasets, validation utilities, and committed outputs:

<p align="center">
  <a href="https://github.com/mbc96325/torchdcm-paper"><b>torchdcm-paper: reproduce the validation and benchmark results</b></a>
</p>

## Why TorchDCM

| Goal | What TorchDCM Provides |
| --- | --- |
| PyTorch-native estimation | Vectorized likelihoods written around tensors and automatic differentiation. |
| Econometric outputs | Classic, robust, and cluster covariance; WTP and elasticity helpers. |
| Model coverage | MNL, NL, CNL, mixed logit, WTP-space, latent class, scaled, ordered, and hybrid choice. |
| Reusable package | Clean import surface, executed model notebooks, and package-level tests. |
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
plain-text summary. The convergence section records the exact L-BFGS stopping
condition, the internal-parameter gradient infinity norm, and that norm divided
by the number of observations. Function- or step-tolerance stops are reported
as converged when the normalized gradient does not exceed `1e-5`; iteration
limits, non-finite values, and larger normalized gradients remain warnings.

All estimators accept a standard PyTorch-style `device` argument. Passing
`device="cuda"` moves estimation, prediction, simulated likelihoods, and
covariance calculations for that model to CUDA when your PyTorch installation
has GPU support.

## Executed Examples

The [`examples/`](examples) directory contains self-contained Jupyter
notebooks for every public model family. Each notebook presents the model's
mathematical formulation, builds a nontrivial specification, runs full
estimation, and retains its rendered HTML report. They were executed on an AMD
Ryzen 9 9950X3D CPU (16 cores), 64 GB RAM, and an NVIDIA GeForce RTX 5090 GPU
(32 GB VRAM), running Ubuntu 24.04.4 with PyTorch 2.12.1 and CUDA 13.0. The
examples automatically select CUDA when it is available and can be changed to
CPU by setting `device = "cpu"`.

| Model | Executed notebook |
| --- | --- |
| Multinomial logit | [`01_multinomial_logit.ipynb`](examples/01_multinomial_logit.ipynb) |
| Nested logit | [`02_nested_logit.ipynb`](examples/02_nested_logit.ipynb) |
| Cross-nested logit | [`03_cross_nested_logit.ipynb`](examples/03_cross_nested_logit.ipynb) |
| Mixed logit | [`04_mixed_logit.ipynb`](examples/04_mixed_logit.ipynb) |
| WTP-space mixed logit | [`05_wtp_space_mixed_logit.ipynb`](examples/05_wtp_space_mixed_logit.ipynb) |
| Alternative-scaled MNL | [`06_scaled_multinomial_logit.ipynb`](examples/06_scaled_multinomial_logit.ipynb) |
| Covariate-scaled MNL | [`07_covariate_scaled_multinomial_logit.ipynb`](examples/07_covariate_scaled_multinomial_logit.ipynb) |
| Ordered logit | [`08_ordered_logit.ipynb`](examples/08_ordered_logit.ipynb) |
| Ordered probit | [`09_ordered_probit.ipynb`](examples/09_ordered_probit.ipynb) |
| Latent-class logit | [`10_latent_class_logit.ipynb`](examples/10_latent_class_logit.ipynb) |
| Error-components logit | [`11_error_components_logit.ipynb`](examples/11_error_components_logit.ipynb) |
| Hybrid choice | [`12_hybrid_choice.ipynb`](examples/12_hybrid_choice.ipynb) |
| Panel mixed logit | [`13_panel_mixed_logit.ipynb`](examples/13_panel_mixed_logit.ipynb) |
| Panel multinomial logit with cluster covariance | [`14_panel_multinomial_logit.ipynb`](examples/14_panel_multinomial_logit.ipynb) |

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

## Repository Layout

| Path | Purpose |
| --- | --- |
| `torchdcm/` | Importable package implementation. |
| `tests/` | Package-level unit tests. |
| `examples/` | Executed Jupyter notebooks covering every public model family. |
| `docs/assets/` | GitHub README logo and cover assets. |
| `pyproject.toml` | Packaging metadata and dependencies. |

## Development

```bash
python -m pip install -e ".[dev]"
pytest
```

Benchmark and validation work should happen in the companion repository:

```bash
git clone https://github.com/mbc96325/torchdcm-paper.git
```
