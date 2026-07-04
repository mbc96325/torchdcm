from __future__ import annotations

import numpy as np
import pandas as pd


def make_swissmetro_like(n_obs: int = 500, seed: int = 1) -> pd.DataFrame:
    """Synthetic Swissmetro-style mode choice data.

    Columns intentionally mirror the classic train/Swissmetro/car structure so
    examples and Biogeme comparisons can be written without external downloads.
    """

    rng = np.random.default_rng(seed)
    person_id = np.repeat(np.arange(max(1, n_obs // 3 + 1)), 3)[:n_obs]
    income = rng.lognormal(mean=3.4, sigma=0.35, size=n_obs)
    base_dist = rng.gamma(shape=2.0, scale=35.0, size=n_obs) + 5
    time_train = base_dist / rng.normal(0.95, 0.08, size=n_obs) + rng.normal(8, 4, size=n_obs)
    time_sm = base_dist / rng.normal(1.35, 0.10, size=n_obs) + rng.normal(4, 2, size=n_obs)
    time_car = base_dist / rng.normal(0.85, 0.10, size=n_obs) + rng.normal(3, 5, size=n_obs)
    cost_train = 0.22 * base_dist + rng.normal(2, 1.5, size=n_obs)
    cost_sm = 0.28 * base_dist + rng.normal(3, 1.2, size=n_obs)
    cost_car = 0.32 * base_dist + rng.normal(1, 2.0, size=n_obs)
    avail_train = rng.random(n_obs) > 0.03
    avail_sm = rng.random(n_obs) > 0.08
    avail_car = rng.random(n_obs) > 0.05
    for i in range(n_obs):
        if not (avail_train[i] or avail_sm[i] or avail_car[i]):
            avail_sm[i] = True
    utilities = np.column_stack(
        [
            0.30 - 0.030 * time_train - 0.090 * cost_train,
            0.00 - 0.030 * time_sm - 0.090 * cost_sm,
            0.55 - 0.030 * time_car - 0.090 * cost_car,
        ]
    )
    availability = np.column_stack([avail_train, avail_sm, avail_car])
    utilities = np.where(availability, utilities, -1e9)
    probs = _softmax(utilities)
    alts = np.asarray(["TRAIN", "SM", "CAR"])
    choices = [rng.choice(alts, p=probs[i]) for i in range(n_obs)]
    return pd.DataFrame(
        {
            "obs_id": np.arange(n_obs),
            "person_id": person_id,
            "income": income,
            "choice": choices,
            "time_train": np.maximum(time_train, 1.0),
            "time_sm": np.maximum(time_sm, 1.0),
            "time_car": np.maximum(time_car, 1.0),
            "cost_train": np.maximum(cost_train, 0.5),
            "cost_sm": np.maximum(cost_sm, 0.5),
            "cost_car": np.maximum(cost_car, 0.5),
            "avail_train": avail_train,
            "avail_sm": avail_sm,
            "avail_car": avail_car,
        }
    )


def _softmax(values: np.ndarray) -> np.ndarray:
    centered = values - values.max(axis=1, keepdims=True)
    exp = np.exp(centered)
    return exp / exp.sum(axis=1, keepdims=True)

