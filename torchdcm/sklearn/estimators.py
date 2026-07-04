from __future__ import annotations

from torchdcm.data.choice_dataset import ChoiceDataset
from torchdcm.models.mnl import MultinomialLogit


class TorchDCMClassifier:
    """Small scikit-learn-style wrapper around ``MultinomialLogit``."""

    def __init__(self, spec, *, cov_type: str = "classic", **model_kwargs):
        self.spec = spec
        self.cov_type = cov_type
        self.model_kwargs = model_kwargs
        self.model_ = None
        self.result_ = None

    def get_params(self, deep: bool = True) -> dict:
        return {"spec": self.spec, "cov_type": self.cov_type, **self.model_kwargs}

    def set_params(self, **params):
        for key, value in params.items():
            if key == "spec":
                self.spec = value
            elif key == "cov_type":
                self.cov_type = value
            else:
                self.model_kwargs[key] = value
        return self

    def fit(self, data: ChoiceDataset, y=None):
        self.model_ = MultinomialLogit(self.spec, **self.model_kwargs)
        self.result_ = self.model_.fit(data, cov_type=self.cov_type)
        return self

    def predict_proba(self, data: ChoiceDataset):
        return self.result_.predict_proba(data)

    def predict(self, data: ChoiceDataset):
        return self.result_.predict(data)

    def score(self, data: ChoiceDataset, y=None):
        return self.result_.score(data)

