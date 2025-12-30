"""Pydantic models for Factor Discovery Engine."""

from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, Field


class ExclusionFilters(BaseModel):
    """Filters to exclude certain stocks from analysis."""

    # Altman zones to exclude
    exclude_altman_zones: list[str] = Field(
        default=["distress"],
        description="Altman zones to exclude: distress, grey, safe",
    )
    # Minimum Piotroski score (exclude below this)
    min_piotroski: int | None = Field(
        default=None,
        description="Exclude stocks with Piotroski score below this (1-9)",
    )
    # Quality tags to exclude (stocks WITH these tags are excluded)
    exclude_quality_tags: list[str] = Field(
        default=[],
        description="Exclude stocks with these quality tags",
    )
    # Quality tags to require (stocks WITHOUT these tags are excluded)
    require_quality_tags: list[str] = Field(
        default=[],
        description="Only include stocks with ALL these quality tags",
    )
    # Exclude penny stocks
    exclude_penny_stocks: bool = Field(
        default=False,
        description="Exclude stocks trading below $5",
    )
    # Exclude negative earnings
    exclude_negative_earnings: bool = Field(
        default=False,
        description="Exclude stocks with negative earnings yield",
    )


class FactorDiscoveryRequest(BaseModel):
    """Request to run factor discovery analysis."""

    quarters: list[str] = Field(
        ...,
        description="List of quarters to analyze (e.g., ['2024Q1', '2024Q2'])",
    )
    holding_periods: list[int] = Field(
        default=[1, 2, 3, 4],
        description="Holding periods in quarters (1-4)",
    )
    min_sample_size: int = Field(
        default=100,
        description="Minimum sample size for any threshold to be considered",
    )
    significance_level: float = Field(
        default=0.01,
        description="P-value threshold for statistical significance",
    )
    cost_haircut: float = Field(
        default=3.0,
        description="Only trust strategies with alpha > this percentage",
    )
    # Portfolio simulation settings
    portfolio_sizes: list[int] = Field(
        default=[20],
        description="Portfolio sizes to simulate (e.g., [10, 20, 50])",
    )
    ranking_method: str = Field(
        default="magic-formula",
        description="How to rank stocks: magic-formula, earnings-yield, roic, graham-score",
    )
    # Combination settings
    max_factors: int = Field(
        default=4,
        ge=1,
        le=55,
        description="Maximum number of factors to combine (1-55, default 4)",
    )
    # Exclusion filters
    exclusions: ExclusionFilters = Field(
        default_factory=ExclusionFilters,
        description="Filters to exclude certain stocks from analysis",
    )
    # Factor category selection (all enabled by default)
    factor_categories: list[str] = Field(
        default=[
            "scores",
            "raw_valuation",
            "raw_profitability",
            "raw_liquidity",
            "raw_leverage",
            "raw_efficiency",
            "raw_dividend",
            "stability",
            "growth",
            "boolean",
        ],
        description="Factor categories to analyze (e.g., scores, raw_valuation, raw_profitability)",
    )
    # Out-of-sample validation settings
    train_end_quarter: str | None = Field(
        default=None,
        description="Last quarter for training set (e.g., '2022Q4'). If None, uses all data.",
    )
    validation_end_quarter: str | None = Field(
        default=None,
        description="Last quarter for validation set (e.g., '2023Q4'). Test set is everything after.",
    )
    data_lag_quarters: int = Field(
        default=1,
        ge=0,
        le=4,
        description="Quarters to lag analysis data (1 = use Q1 data for Q2 decisions, prevents look-ahead bias)",
    )


class PortfolioStats(BaseModel):
    """Stats for a specific portfolio size simulation."""

    size: int = Field(..., description="Portfolio size (e.g., 20)")
    mean_alpha: float = Field(..., description="Average alpha for top-N stocks")
    sample_size: int = Field(..., description="Total observations (size × quarters)")
    win_rate: float = Field(..., description="Percentage with positive alpha")
    ci_lower: float = Field(..., description="95% CI lower bound")
    ci_upper: float = Field(..., description="95% CI upper bound")


class ThresholdResult(BaseModel):
    """Result for a single threshold test."""

    threshold: str = Field(
        ...,
        description="Threshold description (e.g., '>= 6' or 'safe')",
    )
    mean_alpha: float = Field(..., description="Average alpha for stocks passing threshold")
    sample_size: int = Field(..., description="Number of stocks passing threshold")
    lift: float = Field(
        ...,
        description="Lift ratio: P(alpha > 0 | threshold) / P(alpha > 0 | all)",
    )
    pvalue: float = Field(..., description="T-test p-value for difference")
    ci_lower: float = Field(..., description="95% confidence interval lower bound")
    ci_upper: float = Field(..., description="95% confidence interval upper bound")
    win_rate: float = Field(
        default=0.0,
        description="Percentage of stocks with positive alpha",
    )
    # FDR-corrected significance (Benjamini-Hochberg)
    fdr_significant: bool = Field(
        default=False,
        description="Whether significant after FDR correction (Benjamini-Hochberg)",
    )
    adjusted_pvalue: float | None = Field(
        default=None,
        description="FDR-adjusted p-value (Benjamini-Hochberg)",
    )


class FactorResult(BaseModel):
    """Analysis result for a single factor."""

    factor_name: str = Field(..., description="Name of the factor")
    factor_type: str = Field(
        ...,
        description="Type of factor: 'numerical', 'categorical', or 'boolean'",
    )
    holding_period: int = Field(..., description="Holding period in quarters")

    # Correlation stats (for numerical factors)
    correlation: Optional[float] = Field(
        default=None,
        description="Spearman correlation with alpha",
    )
    correlation_pvalue: Optional[float] = Field(
        default=None,
        description="P-value for correlation",
    )

    # Threshold results
    threshold_results: list[ThresholdResult] = Field(
        default_factory=list,
        description="Results for each threshold tested",
    )

    # Best threshold
    best_threshold: Optional[str] = Field(
        default=None,
        description="Best performing threshold",
    )
    best_threshold_alpha: Optional[float] = Field(
        default=None,
        description="Alpha at best threshold",
    )
    best_threshold_lift: Optional[float] = Field(
        default=None,
        description="Lift at best threshold",
    )
    best_threshold_pvalue: Optional[float] = Field(
        default=None,
        description="P-value at best threshold",
    )
    best_threshold_sample_size: Optional[int] = Field(
        default=None,
        description="Sample size at best threshold",
    )
    best_threshold_ci_lower: Optional[float] = Field(
        default=None,
        description="CI lower at best threshold",
    )
    best_threshold_ci_upper: Optional[float] = Field(
        default=None,
        description="CI upper at best threshold",
    )
    best_threshold_fdr_significant: Optional[bool] = Field(
        default=None,
        description="Whether best threshold is FDR-significant",
    )


class FilterSpec(BaseModel):
    """Specification for a single filter in a strategy."""

    factor: str = Field(..., description="Factor name")
    operator: str = Field(
        ...,
        description="Comparison operator: '>=', '<=', '==', 'in', 'not_in', 'has', 'not_has'",
    )
    value: Any = Field(..., description="Value to compare against")


class CombinedStrategyResult(BaseModel):
    """Result for a combination of factors."""

    filters: list[FilterSpec] = Field(..., description="List of filters in strategy")
    mean_alpha: float = Field(..., description="Average alpha (all stocks)")
    sample_size: int = Field(..., description="Number of matching stocks (all)")
    lift: float = Field(..., description="Lift ratio")
    win_rate: float = Field(..., description="Percentage with positive alpha")
    ci_lower: float = Field(..., description="95% CI lower bound")
    ci_upper: float = Field(..., description="95% CI upper bound")
    # Portfolio-specific stats (top-N stocks per quarter)
    portfolio_stats: dict[int, PortfolioStats] = Field(
        default_factory=dict,
        description="Stats for each portfolio size: {20: PortfolioStats, 50: ...}",
    )
    # Out-of-sample metrics (only populated if train/validation splits used)
    train_alpha: float | None = Field(
        default=None,
        description="Average alpha on training set",
    )
    train_sample_size: int | None = Field(
        default=None,
        description="Sample size in training set",
    )
    validation_alpha: float | None = Field(
        default=None,
        description="Average alpha on validation set",
    )
    validation_sample_size: int | None = Field(
        default=None,
        description="Sample size in validation set",
    )
    test_alpha: float | None = Field(
        default=None,
        description="Average alpha on test set",
    )
    test_sample_size: int | None = Field(
        default=None,
        description="Sample size in test set",
    )
    overfit_ratio: float | None = Field(
        default=None,
        description="validation_alpha / train_alpha - values < 0.5 indicate overfitting",
    )


class PipelineSettings(BaseModel):
    """Settings that map directly to the Pipeline UI controls."""

    # Survival Gates
    piotroski_enabled: bool = Field(default=False)
    piotroski_min: int = Field(default=5)
    altman_enabled: bool = Field(default=False)
    altman_zone: str = Field(default="safe")

    # Quality
    quality_enabled: bool = Field(default=False)
    min_quality: str = Field(default="weak")  # weak, average, compounder
    excluded_tags: list[str] = Field(default_factory=list)
    required_tags: list[str] = Field(default_factory=list)

    # Valuation Lenses
    graham_enabled: bool = Field(default=False)
    graham_mode: str = Field(default="strict")
    graham_min: int = Field(default=5)
    magic_formula_enabled: bool = Field(default=False)
    mf_top_pct: int = Field(default=20)
    peg_enabled: bool = Field(default=False)
    max_peg: float = Field(default=1.5)
    net_net_enabled: bool = Field(default=False)
    fama_french_enabled: bool = Field(default=False)
    ff_top_pct: int = Field(default=30)
    min_lenses: int = Field(default=1)
    strict_mode: bool = Field(default=False)

    # Raw factor filters (factors not mapped to specific UI controls)
    raw_filters: list[dict] = Field(default_factory=list)


class RecommendedStrategy(BaseModel):
    """Recommended strategy for a given holding period."""

    holding_period: int = Field(..., description="Holding period in quarters")
    pipeline_settings: PipelineSettings = Field(
        ...,
        description="Settings for the Pipeline page",
    )
    expected_alpha: float = Field(..., description="Expected alpha percentage (all stocks)")
    expected_alpha_ci_lower: float = Field(..., description="95% CI lower bound")
    expected_alpha_ci_upper: float = Field(..., description="95% CI upper bound")
    expected_win_rate: float = Field(..., description="Expected win rate percentage")
    sample_size: int = Field(..., description="Number of historical matches (all stocks)")
    confidence_score: float = Field(
        ...,
        description="Overall confidence score (0-1)",
    )
    key_factors: list[dict] = Field(
        default_factory=list,
        description="Top contributing factors with their stats",
    )
    # Portfolio-specific stats
    portfolio_stats: dict[int, PortfolioStats] = Field(
        default_factory=dict,
        description="Stats for each portfolio size: {20: PortfolioStats, 50: ...}",
    )


class FactorDiscoveryProgress(BaseModel):
    """Progress update for a running analysis."""

    run_id: str = Field(..., description="Run ID")
    status: str = Field(
        ...,
        description="Status: 'running', 'completed', 'failed', 'cancelled'",
    )
    phase: str = Field(
        ...,
        description="Current phase: 'building_dataset', 'analyzing_factors', etc.",
    )
    progress: float = Field(..., description="Progress 0.0 to 1.0")
    current_factor: Optional[str] = Field(
        default=None,
        description="Currently processing factor",
    )
    current_holding_period: Optional[int] = Field(
        default=None,
        description="Currently processing holding period",
    )
    estimated_remaining_seconds: Optional[int] = Field(
        default=None,
        description="Estimated time remaining",
    )


class FactorDiscoveryResult(BaseModel):
    """Complete result of a factor discovery run."""

    run_id: str = Field(..., description="Unique run ID")
    status: str = Field(..., description="Status")
    created_at: datetime = Field(..., description="When the run started")
    completed_at: Optional[datetime] = Field(
        default=None,
        description="When the run completed",
    )
    duration_seconds: Optional[float] = Field(
        default=None,
        description="Total duration",
    )
    config: FactorDiscoveryRequest = Field(..., description="Request configuration")
    total_observations: int = Field(
        ...,
        description="Total (stock, quarter, hold) tuples analyzed",
    )

    # Results by holding period
    factor_results: dict[int, list[FactorResult]] = Field(
        default_factory=dict,
        description="Factor results by holding period",
    )
    combined_results: dict[int, list[CombinedStrategyResult]] = Field(
        default_factory=dict,
        description="Combined strategy results by holding period",
    )
    recommended_strategies: dict[int, RecommendedStrategy] = Field(
        default_factory=dict,
        description="Recommended strategies by holding period",
    )

    # Summary
    best_holding_period: Optional[int] = Field(
        default=None,
        description="Holding period with highest alpha",
    )
    best_alpha: Optional[float] = Field(
        default=None,
        description="Best alpha achieved",
    )


class FactorDiscoverySummary(BaseModel):
    """Summary of a factor discovery run (for history listing)."""

    run_id: str
    created_at: datetime
    status: str
    quarters_analyzed: int
    best_holding_period: Optional[int]
    best_alpha: Optional[float]
    duration_seconds: Optional[float]
