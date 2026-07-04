from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass

import torch

from torchdcm.data.choice_dataset import ChoiceDataset
from torchdcm.models.mixed_logit import MixedLogit, RandomCoefficient
from torchdcm.spec.expressions import Expression, Term
from torchdcm.spec.parameters import Beta
from torchdcm.spec.utility import UtilitySpec


@dataclass(frozen=True)
class ErrorComponent:
    """Zero-mean normal error component with alternative loadings."""

    name: str
    loadings: Mapping[str, float] | Iterable[str]
    sigma_init: float = 0.1
    sigma_name: str | None = None
    fixed: bool = False

    @property
    def parameter_name(self) -> str:
        return f"EC_{self.name}"

    @property
    def variable_name(self) -> str:
        return f"ec_{self.name.lower()}_loading"

    def loading_for(self, alternative: str) -> float:
        if isinstance(self.loadings, Mapping):
            return float(self.loadings.get(alternative, 0.0))
        return 1.0 if alternative in set(self.loadings) else 0.0


class ErrorComponentsLogit(MixedLogit):
    """Mixed logit convenience wrapper for error-components specifications."""

    def __init__(
        self,
        spec: UtilitySpec,
        components: list[ErrorComponent] | dict[str, Iterable[str] | Mapping[str, float]],
        *,
        random_coefficients: list[RandomCoefficient] | None = None,
        **kwargs,
    ) -> None:
        self.base_spec = spec
        self.components = _normalize_components(components)
        augmented_spec = self._augment_spec(spec)
        component_random_coefficients = [
            RandomCoefficient(
                component.parameter_name,
                sigma_init=component.sigma_init,
                sigma_name=component.sigma_name,
                fixed=component.fixed,
            )
            for component in self.components
        ]
        super().__init__(augmented_spec, [*(random_coefficients or []), *component_random_coefficients], **kwargs)
        self._augmented_data_cache: dict[tuple[int, torch.device, torch.dtype], ChoiceDataset] = {}

    def compile(self, data: ChoiceDataset):
        data = data.to(device=self.device, dtype=self.dtype)
        augmented = self._augment_data(data)
        return MixedLogit.compile(self, augmented)

    def _augment_spec(self, spec: UtilitySpec) -> UtilitySpec:
        augmented = UtilitySpec()
        for alternative, expression in spec.utilities.items():
            terms = list(expression.terms)
            for component in self.components:
                loading = component.loading_for(alternative)
                if loading:
                    terms.append(Term(Beta(component.parameter_name, init=0.0, fixed=True), component.variable_name, loading))
            augmented.utility(alternative, Expression(terms))
        return augmented

    def _augment_data(self, data: ChoiceDataset) -> ChoiceDataset:
        cache_key = (id(data), data.device, data.dtype)
        cached = self._augmented_data_cache.get(cache_key)
        if cached is not None:
            return cached
        alt_codes = data.alt_id.to(device=data.device)
        x_alt = dict(data.x_alt)
        for component in self.components:
            values = torch.zeros(data.n_rows, dtype=data.dtype, device=data.device)
            for alt_index, alternative in enumerate(data.alt_names):
                loading = component.loading_for(alternative)
                if loading:
                    values = values.masked_fill(alt_codes == alt_index, float(loading))
            x_alt[component.variable_name] = values
        augmented = ChoiceDataset(
            obs_ptr=data.obs_ptr,
            alt_id=data.alt_id,
            chosen_row=data.chosen_row,
            x_alt=x_alt,
            weights=data.weights,
            availability=data.availability,
            obs_ids=list(data.obs_ids),
            alt_names=list(data.alt_names),
            obs_to_ind=data.obs_to_ind,
            individual_ids=None if data.individual_ids is None else list(data.individual_ids),
            x_obs=dict(data.x_obs),
        )
        self._augmented_data_cache[cache_key] = augmented
        return augmented


def _normalize_components(
    components: list[ErrorComponent] | dict[str, Iterable[str] | Mapping[str, float]],
) -> list[ErrorComponent]:
    if isinstance(components, dict):
        return [ErrorComponent(name=name, loadings=loadings) for name, loadings in components.items()]
    return list(components)
