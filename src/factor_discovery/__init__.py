"""Factor Discovery Engine - Discover optimal stock selection strategies from historical data."""

from .models import (
    ExclusionFilters,
    FactorDiscoveryRequest,
    FactorDiscoveryResult,
    FactorDiscoveryProgress,
    FactorDiscoverySummary,
    FactorResult,
    ThresholdResult,
    FilterSpec,
    CombinedStrategyResult,
    PipelineSettings,
    RecommendedStrategy,
)
from .runner import FactorDiscoveryRunner, run_factor_discovery
from .storage import FactorDiscoveryStorage, get_storage

__all__ = [
    # Models
    "ExclusionFilters",
    "FactorDiscoveryRequest",
    "FactorDiscoveryResult",
    "FactorDiscoveryProgress",
    "FactorDiscoverySummary",
    "FactorResult",
    "ThresholdResult",
    "FilterSpec",
    "CombinedStrategyResult",
    "PipelineSettings",
    "RecommendedStrategy",
    # Runner
    "FactorDiscoveryRunner",
    "run_factor_discovery",
    # Storage
    "FactorDiscoveryStorage",
    "get_storage",
]
