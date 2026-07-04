from __future__ import annotations

import numpy as np
import pandas as pd


def make_london_like(n_obs: int = 500, seed: int = 11) -> pd.DataFrame:
    """Synthetic London commute mode choice data."""

    rng = np.random.default_rng(seed)
    person_id = np.repeat(np.arange(max(1, n_obs // 4 + 1)), 4)[:n_obs]
    distance = rng.gamma(shape=2.3, scale=4.0, size=n_obs) + 0.5
    time_tube = 4 + 2.6 * distance + rng.normal(0, 3, n_obs)
    time_bus = 6 + 3.8 * distance + rng.normal(0, 4, n_obs)
    time_car = 3 + 3.0 * distance + rng.normal(0, 5, n_obs)
    time_bike = 2 + 5.0 * distance + rng.normal(0, 2, n_obs)
    cost_tube = 2.4 + 0.18 * distance + rng.normal(0, 0.2, n_obs)
    cost_bus = 1.75 + 0.05 * distance + rng.normal(0, 0.1, n_obs)
    cost_car = 1.0 + 0.55 * distance + rng.normal(0, 0.5, n_obs)
    cost_bike = np.zeros(n_obs)
    avail_tube = rng.random(n_obs) > 0.12
    avail_bus = rng.random(n_obs) > 0.04
    avail_car = rng.random(n_obs) > 0.20
    avail_bike = distance < rng.normal(9.0, 2.0, n_obs)
    for i in range(n_obs):
        if not (avail_tube[i] or avail_bus[i] or avail_car[i] or avail_bike[i]):
            avail_bus[i] = True
    utilities = np.column_stack(
        [
            0.25 - 0.075 * time_tube - 0.65 * cost_tube,
            0.05 - 0.075 * time_bus - 0.65 * cost_bus,
            0.35 - 0.075 * time_car - 0.65 * cost_car,
            -0.15 - 0.075 * time_bike - 0.65 * cost_bike,
        ]
    )
    availability = np.column_stack([avail_tube, avail_bus, avail_car, avail_bike])
    utilities = np.where(availability, utilities, -1e9)
    probs = _softmax(utilities)
    alts = np.asarray(["tube", "bus", "car", "bike"])
    choices = [rng.choice(alts, p=probs[i]) for i in range(n_obs)]
    return pd.DataFrame(
        {
            "obs_id": np.arange(n_obs),
            "person_id": person_id,
            "choice": choices,
            "distance": distance,
            "time_tube": np.maximum(time_tube, 0.5),
            "time_bus": np.maximum(time_bus, 0.5),
            "time_car": np.maximum(time_car, 0.5),
            "time_bike": np.maximum(time_bike, 0.5),
            "cost_tube": np.maximum(cost_tube, 0.0),
            "cost_bus": np.maximum(cost_bus, 0.0),
            "cost_car": np.maximum(cost_car, 0.0),
            "cost_bike": cost_bike,
            "avail_tube": avail_tube,
            "avail_bus": avail_bus,
            "avail_car": avail_car,
            "avail_bike": avail_bike,
        }
    )


def _softmax(values: np.ndarray) -> np.ndarray:
    centered = values - values.max(axis=1, keepdims=True)
    exp = np.exp(centered)
    return exp / exp.sum(axis=1, keepdims=True)

