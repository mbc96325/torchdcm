# TorchDCM

TorchDCM is a small PyTorch-first prototype for estimating discrete choice
models. The current implementation covers the v0.1 MNL core and the first
v0.2 nested-logit building block from the project plan:

- ragged long-format choice sets;
- wide-to-long conversion for common mode choice data;
- multinomial/conditional logit estimation;
- fixed and free coefficients;
- availability and observation weights;
- classical, robust, and cluster covariance estimates;
- `fit`, `predict_proba`, `predict`, `score`, WTP, and simple elasticities;
- Swissmetro-style and London-style test fixtures.
- disjoint nested logit with estimated nest dissimilarity parameters.

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

## Development

Install the package in editable mode with test dependencies:

```bash
python -m pip install -e '.[dev]'
python -m pytest
```
