# TorchDCM

<p align="center">
  <img src="docs/assets/torchdcm-logo.svg" alt="TorchDCM logo" width="78%">
</p>

<p align="center">
  <b>PyTorch-first discrete choice estimation, inference, and estimator benchmarking.</b>
</p>

<p align="center">
  <a href="#quick-start">Quick Start</a> |
  <a href="datasets/README.md">Datasets</a> |
  <a href="docs/benchmarking.md">Benchmarks</a> |
  <a href="#model-zoo">Model Zoo</a> |
  <a href="#development">Development</a>
</p>

<p align="center">
  <img alt="Python" src="https://img.shields.io/badge/python-3.10%2B-blue">
  <img alt="PyTorch" src="https://img.shields.io/badge/backend-PyTorch-ee4c2c">
  <img alt="Benchmarks" src="https://img.shields.io/badge/benchmarks-Biogeme%20%7C%20Apollo%20%7C%20mlogit-2b9348">
  <img alt="Datasets" src="https://img.shields.io/badge/datasets-public%20choice%20data-6c63ff">
  <img alt="Status" src="https://img.shields.io/badge/status-research%20prototype-f59f00">
</p>

TorchDCM is a PyTorch-first toolkit for discrete choice model estimation,
econometric inference, and reproducible estimator benchmarking. It is designed
as a compact package plus a public benchmark system: model code stays in
`torchdcm/`, small public datasets live in `datasets/small/`, and heavyweight
validation against Biogeme, Apollo, and R estimators stays under `validation/`.

## Why TorchDCM

| Goal | What TorchDCM Provides |
| --- | --- |
| Fast estimators | Vectorized PyTorch likelihoods for ragged choice sets and panel data. |
| Econometric outputs | Classic, robust, and cluster covariance; WTP and elasticity helpers. |
| Model coverage | MNL, NL, CNL, Mixed Logit, WTP-space, latent class, scaled, ordered, and hybrid choice. |
| Reproducible parity | Aligned comparisons to Biogeme, Apollo, SciPy, and R `mlogit`. |
| Public benchmark data | 27 GitHub-small datasets plus processed-only large data releases. |

## Repository Map

| Path | Purpose |
| --- | --- |
| `torchdcm/` | Pure package implementation. |
| `datasets/` | Public data hub, GitHub-small data, and large processed-data link table. |
| `examples/` | Minimal runnable examples. |
| `docs/` | Data and benchmark documentation. |
| `validation/` | Raw downloads, estimator wrappers, and benchmark reports. Ignored by default. |
| `scripts/` | Dataset release and preprocessing utilities. |

## Benchmark Snapshot

| Case | Backends | Current Status |
| --- | --- | --- |
| Swissmetro MNL/NL/CNL | TorchDCM, Biogeme, Apollo, SciPy | aligned likelihood, parameters, covariance, probabilities, runtime split |
| Swissmetro Mixed/WTP/LC | TorchDCM, Biogeme, Apollo | fixed replay or Torch fit + replay with shared parameters/draws |
| Optima ordered models | TorchDCM, Biogeme | ordered logit/probit parity |
| R `mlogit` Fishing/ModeCanada | TorchDCM, R `mlogit` | full estimation with beta, SE, t-value, covariance diffs |
| LPMC London | TorchDCM data release | processed wide/long choice-set artifact ready for Google Drive |

## Quick Links

- [Dataset hub](datasets/README.md)
- [Small public datasets](datasets/small/small_datasets.csv)
- [Large processed-data links](datasets/large/google_drive_links.csv)
- [Data workflow notes](docs/data.md)
- [Benchmarking notes](docs/benchmarking.md)

## Model Zoo

The current implementation covers:

- ragged long-format choice sets;
- wide-to-long conversion for common mode choice data;
- multinomial/conditional logit estimation;
- fixed and free coefficients;
- availability and observation weights;
- classical, robust, and cluster covariance estimates;
- `fit`, `predict_proba`, `predict`, `score`, WTP, and simple elasticities;
- Swissmetro-style and London-style test fixtures.
- disjoint nested logit with estimated nest dissimilarity parameters.
- cross-nested logit with fixed allocation weights and estimated nest dissimilarity parameters.
- mixed logit with panel likelihood, normal/lognormal random coefficients, and Cholesky-correlated draws.
- WTP-space mixed logit with random WTP coefficients.
- error-components logit as zero-mean random coefficients with alternative loadings.
- scaled multinomial logit with alternative-specific utility scales.
- covariate-scaled multinomial logit with row-specific alternative scales.
- ordered logit and ordered probit for ordinal outcomes.
- latent class logit with class-specific MNL utilities and class membership constants.
- hybrid choice with normal latent variables and continuous Gaussian measurement indicators.

## Public Dataset Hub

Small datasets are materialized under `datasets/small/<dataset_id>/data.csv`
and can be committed to GitHub. Large datasets are published as processed
choice-set archives through Google Drive, with links tracked in
`datasets/large/google_drive_links.csv`.

Current release status:

| Class | Count | Location |
| --- | ---: | --- |
| GitHub-small datasets | 27 | `datasets/small/` |
| Large processed candidates | 8 | `datasets/large/google_drive_links.csv` |
| Local/remote raw validation downloads | 28 downloaded + 7 scoped | `validation/datasets/raw/` |

Large data release rule: only upload processed files with defined choice sets,
attributes, availability, IDs, and schema. Do not upload raw survey dumps.

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

B_TIME = Beta("B_TIME", init=-0.01)
B_COST = Beta("B_COST", init=-0.1)
ASC_TRAIN = Beta("ASC_TRAIN", init=0.0)
ASC_CAR = Beta("ASC_CAR", init=0.0)

spec = UtilitySpec()
spec.utility("TRAIN", ASC_TRAIN + B_TIME * "time" + B_COST * "cost")
spec.utility("SM", B_TIME * "time" + B_COST * "cost")
spec.utility("CAR", ASC_CAR + B_TIME * "time" + B_COST * "cost")

model = MultinomialLogit(spec)
res = model.fit(data, cov_type="robust")
print(res.summary())
```

## Nested Logit

```python
from torchdcm import Nest, NestedLogit

nests = {
    "PUBLIC": Nest(["TRAIN", "SM"], init=0.8),
    "PRIVATE": Nest(["CAR"], init=1.0, fixed=True),
}

nl = NestedLogit(spec, nests)
res = nl.fit(data)
print(res.summary())
```

## Mixed Logit

```python
from torchdcm import MixedLogit, RandomCoefficient

mixed = MixedLogit(
    spec,
    random_coefficients=[RandomCoefficient("B_TIME", sigma_init=0.1)],
    n_draws=128,
    seed=20260704,
    panel=True,
)
res = mixed.fit(data)
print(res.summary())
```

## Cross-Nested Logit

```python
from torchdcm import CrossNest, CrossNestedLogit

cross_nests = {
    "PUBLIC": CrossNest({"TRAIN": 0.7, "SM": 0.8, "CAR": 0.0}, init=0.8),
    "PRIVATE": CrossNest({"TRAIN": 0.3, "SM": 0.2, "CAR": 1.0}, init=0.9),
}

cnl = CrossNestedLogit(spec, cross_nests)
res = cnl.fit(data)
print(res.summary())
```

## Scaled Multinomial Logit

```python
from torchdcm import AlternativeScale, ScaledMultinomialLogit

scales = {
    "TRAIN": AlternativeScale(init=0.8),
    "SM": AlternativeScale(init=1.0, fixed=True),
    "CAR": AlternativeScale(init=1.2),
}

scaled = ScaledMultinomialLogit(spec, scales)
res = scaled.fit(data)
print(res.summary())
```

## Covariate-Scaled Multinomial Logit

```python
from torchdcm import CovariateScale, CovariateScaledMultinomialLogit

cov_scales = {
    "TRAIN": CovariateScale(Beta("LOG_SCALE_TRAIN_GA") * "ga"),
    "SM": CovariateScale(value=1.0),
    "CAR": CovariateScale(Beta("LOG_SCALE_CAR_GA") * "ga"),
}

cov_scaled = CovariateScaledMultinomialLogit(spec, cov_scales)
res = cov_scaled.fit(data)
print(res.summary())
```

## Ordered Logit / Probit

```python
from torchdcm import Beta, OrderedChoiceDataset, OrderedLogit

ordered_data = OrderedChoiceDataset.from_dataframe(
    df,
    outcome="Envir01",
    variables=["male", "highEducation", "haveGA", "ScaledIncome"],
    categories=[1, 2, 3, 4, 5, 6],
    weight="normalized_weight",
)
latent = (
    Beta("B_MALE") * "male"
    + Beta("B_HIGH_EDUCATION") * "highEducation"
    + Beta("B_GA") * "haveGA"
    + Beta("B_INCOME") * "ScaledIncome"
)
thresholds = [Beta(f"TH_{i}", init=value) for i, value in enumerate([-1, 0, 1, 2, 3], start=1)]

ordered = OrderedLogit(latent, thresholds)
res = ordered.fit(ordered_data)
print(res.summary())
```

## Latent Class Logit

```python
from torchdcm import Beta, LatentClassLogit, UtilitySpec

def class_spec(suffix):
    b_time = Beta(f"B_TIME_{suffix}", init=-0.01)
    b_cost = Beta(f"B_COST_{suffix}", init=-0.1)
    spec = UtilitySpec()
    spec.utility("TRAIN", Beta(f"ASC_TRAIN_{suffix}") + b_time * "time" + b_cost * "cost")
    spec.utility("SM", b_time * "time" + b_cost * "cost")
    spec.utility("CAR", Beta(f"ASC_CAR_{suffix}") + b_time * "time" + b_cost * "cost")
    return spec

membership = [Beta("CLASS_2") + Beta("CLASS_2_GA") * "ga"]
lc = LatentClassLogit([class_spec("C1"), class_spec("C2")], class_membership=membership)
res = lc.fit(data)
print(res.summary())
```

## Hybrid Choice

```python
from torchdcm import ChoiceLatentEffect, ContinuousIndicator, HybridChoiceModel, LatentVariable

data = ChoiceDataset.from_wide(
    df,
    alternatives=["TRAIN", "SM", "CAR"],
    choice="choice",
    variables={
        "time": {"TRAIN": "time_train", "SM": "time_sm", "CAR": "time_car"},
        "cost": {"TRAIN": "cost_train", "SM": "cost_sm", "CAR": "cost_car"},
    },
    obs_variables={"env": "Envir01", "income": "ScaledIncome"},
)

hybrid = HybridChoiceModel(
    spec,
    latent_variables=[
        LatentVariable(
            "ENV",
            intercept=Beta("G_ENV_0"),
            coefficients={"income": Beta("G_ENV_INCOME")},
            sigma_init=1.0,
            sigma_fixed=True,
        )
    ],
    choice_effects=[ChoiceLatentEffect("TRAIN", "ENV", Beta("B_TRAIN_ENV"))],
    indicators=[ContinuousIndicator("env", "ENV", loading=Beta("L_ENV"))],
    n_draws=256,
)
res = hybrid.fit(data)
print(res.summary())
```

## Development

Install the package in editable mode with test dependencies:

```bash
python -m pip install -e '.[dev]'
python -m pytest
```
