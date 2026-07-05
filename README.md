# TorchDCM

TorchDCM is a compact PyTorch package for discrete choice model estimation and
econometric inference.

The package repository is intentionally kept small: it contains the importable
`torchdcm` Python package, package tests, a minimal example, and packaging
metadata. Paper experiments, public benchmark datasets, estimator-comparison
wrappers, plots, and LaTeX source live in the companion repository:

https://github.com/mbc96325/torchdcm-paper

## Installation

```bash
python -m pip install git+https://github.com/mbc96325/torchdcm.git
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
    Beta("ASC_TRAIN") + Beta("B_TIME", init=-0.01) * "time" + Beta("B_COST", init=-0.1) * "cost",
)
spec.utility("SM", Beta("B_TIME", init=-0.01) * "time" + Beta("B_COST", init=-0.1) * "cost")
spec.utility(
    "CAR",
    Beta("ASC_CAR") + Beta("B_TIME", init=-0.01) * "time" + Beta("B_COST", init=-0.1) * "cost",
)

result = MultinomialLogit(spec).fit(data, cov_type="cluster", groups="person_id")
print(result.summary())
```

## Model Coverage

- Multinomial logit / conditional logit.
- Nested logit and cross-nested logit.
- Mixed logit, WTP-space mixed logit, and error-components logit.
- Latent-class logit.
- Scaled and covariate-scaled multinomial logit.
- Ordered logit and ordered probit.
- Hybrid choice with latent variables and Gaussian measurement indicators.
- Classic, robust, and cluster covariance estimates.
- WTP and elasticity helpers.

## Repository Layout

| Path | Purpose |
| --- | --- |
| `torchdcm/` | Importable package implementation. |
| `tests/` | Package-level unit tests. |
| `examples/` | Minimal runnable usage examples. |
| `pyproject.toml` | Packaging metadata and dependencies. |

