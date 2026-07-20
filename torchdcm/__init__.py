"""Public API for the TorchDCM prototype."""

from torchdcm.data.choice_dataset import ChoiceDataset
from torchdcm.data.ordered_dataset import OrderedChoiceDataset
from torchdcm.data.panel import PanelStructure
from torchdcm.models.covariate_scaled_mnl import CovariateScale, CovariateScaledMultinomialLogit
from torchdcm.models.cross_nested_logit import CrossNest, CrossNestedLogit
from torchdcm.models.error_components import ErrorComponent, ErrorComponentsLogit
from torchdcm.models.hybrid_choice import ChoiceLatentEffect, ContinuousIndicator, HybridChoiceModel, LatentVariable
from torchdcm.models.latent_class import LatentClassLogit
from torchdcm.models.mnl import MultinomialLogit
from torchdcm.models.mixed_logit import MixedLogit, RandomCoefficient
from torchdcm.models.nested_logit import Nest, NestedLogit
from torchdcm.models.ordered import OrderedLogit, OrderedProbit
from torchdcm.models.scaled_mnl import AlternativeScale, ScaledMultinomialLogit
from torchdcm.models.wtp_mixed_logit import WTPCoefficient, WTPMixedLogit
from torchdcm.results.report import EstimationReport
from torchdcm.spec.parameters import Beta
from torchdcm.spec.utility import UtilitySpec

__version__ = "0.1.0"

__all__ = [
    "Beta",
    "AlternativeScale",
    "ChoiceDataset",
    "ChoiceLatentEffect",
    "ContinuousIndicator",
    "CovariateScale",
    "CovariateScaledMultinomialLogit",
    "CrossNest",
    "CrossNestedLogit",
    "ErrorComponent",
    "ErrorComponentsLogit",
    "EstimationReport",
    "HybridChoiceModel",
    "LatentClassLogit",
    "LatentVariable",
    "MixedLogit",
    "MultinomialLogit",
    "Nest",
    "NestedLogit",
    "OrderedChoiceDataset",
    "OrderedLogit",
    "OrderedProbit",
    "PanelStructure",
    "RandomCoefficient",
    "ScaledMultinomialLogit",
    "UtilitySpec",
    "WTPCoefficient",
    "WTPMixedLogit",
]
