"""ML Models package for cross-sectional return prediction."""

from .decay_analyzer import DecayAnalyzer, DecayMetrics, RollingStats
from .elastic_net_model import (
    ElasticNetConfig,
    ElasticNetModel,
    ElasticNetResult,
    CoefficientInfo,
    ICHistoryPoint,
    StockPrediction,
    ELASTIC_NET_FEATURES,
    save_elastic_net_result,
    load_elastic_net_result,
)
from .gam_model import (
    GAMConfig,
    GAMModel,
    GAMResult,
    PartialDependence,
    save_gam_result,
    load_gam_result,
)
from .lightgbm_model import (
    LightGBMConfig,
    LightGBMModel,
    LightGBMResult,
    FeatureImportance,
    save_lightgbm_result,
    load_lightgbm_result,
)

__all__ = [
    "DecayAnalyzer",
    "DecayMetrics",
    "RollingStats",
    # Elastic Net
    "ElasticNetConfig",
    "ElasticNetModel",
    "ElasticNetResult",
    "CoefficientInfo",
    "ICHistoryPoint",
    "StockPrediction",
    "ELASTIC_NET_FEATURES",
    "save_elastic_net_result",
    "load_elastic_net_result",
    # GAM
    "GAMConfig",
    "GAMModel",
    "GAMResult",
    "PartialDependence",
    "save_gam_result",
    "load_gam_result",
    # LightGBM
    "LightGBMConfig",
    "LightGBMModel",
    "LightGBMResult",
    "FeatureImportance",
    "save_lightgbm_result",
    "load_lightgbm_result",
]
