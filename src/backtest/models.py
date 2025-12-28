"""Pydantic models for grid search backtesting."""

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field


class SurvivalConfig(BaseModel):
    """Survival gate configuration."""

    altman_enabled: bool = True
    altman_zone: Literal["safe", "grey"] = "safe"
    piotroski_enabled: bool = True
    piotroski_min: int = Field(default=5, ge=0, le=9)


class QualityConfig(BaseModel):
    """Quality filter configuration."""

    enabled: bool = False
    min_quality: Literal["weak", "average", "compounder"] = "weak"
    required_tags: list[str] = Field(default_factory=list)
    excluded_tags: list[str] = Field(default_factory=list)


class ValuationConfig(BaseModel):
    """Valuation lenses configuration."""

    # Graham lens
    graham_enabled: bool = True
    graham_mode: Literal["strict", "modern", "garp", "relaxed"] = "modern"
    graham_min: int = Field(default=5, ge=0, le=8)

    # Magic Formula lens
    magic_formula_enabled: bool = True
    mf_top_pct: int = Field(default=20, ge=1, le=100)

    # PEG lens
    peg_enabled: bool = True
    max_peg: float = Field(default=1.5, ge=0, le=10)

    # Net-Net lens
    net_net_enabled: bool = True

    # Fama-French B/M lens
    fama_french_enabled: bool = False
    ff_top_pct: int = Field(default=30, ge=1, le=100)

    # At-least-N logic
    min_lenses: int = Field(default=1, ge=0, le=5)
    strict_mode: bool = False  # If True, ALL enabled lenses must pass


class StrategyConfig(BaseModel):
    """Complete strategy configuration."""

    id: str = ""
    name: str = ""
    survival: SurvivalConfig = Field(default_factory=SurvivalConfig)
    quality: QualityConfig = Field(default_factory=QualityConfig)
    valuation: ValuationConfig = Field(default_factory=ValuationConfig)


class GridDimension(BaseModel):
    """A dimension to vary in the grid search."""

    name: str  # e.g., "altman_zone", "piotroski_min"
    values: list[Any]  # e.g., ["safe", "grey"] or [5, 6, 7, 8, 9]


class GridSearchRequest(BaseModel):
    """Request to run a grid search."""

    base_strategy: StrategyConfig = Field(default_factory=StrategyConfig)
    dimensions: list[GridDimension] = Field(default_factory=list)
    quarters: list[str]  # e.g., ["2022Q4", "2023Q1", "2023Q2"]
    holding_periods: list[int] = Field(default=[1, 2, 3, 4])  # Quarters to hold


class SimulationResult(BaseModel):
    """Result of a single simulation."""

    strategy_id: str
    strategy_name: str
    strategy_config: dict[str, Any] = Field(default_factory=dict)
    buy_quarter: str
    sell_quarter: str
    holding_period: int
    stock_count: int
    symbols: list[str] = Field(default_factory=list)
    portfolio_return: float
    benchmark_return: float
    alpha: float
    win_rate: float  # % of stocks with positive return
    winners: int = 0
    losers: int = 0


class GridSearchProgress(BaseModel):
    """Progress update for grid search."""

    search_id: str
    status: Literal["running", "completed", "failed"] = "running"
    total_simulations: int
    completed: int
    current_strategy: str | None = None
    current_quarter: str | None = None
    estimated_remaining_seconds: int | None = None
    error: str | None = None


class StrategyHoldingPeriod(BaseModel):
    """Holding period performance for a specific strategy."""

    holding_period: int
    avg_alpha: float
    avg_return: float
    simulation_count: int


class StrategyAggregate(BaseModel):
    """Aggregated results for a single strategy across all quarters/holds."""

    strategy_id: str
    strategy_name: str
    strategy_config: dict[str, Any] = Field(default_factory=dict)
    simulation_count: int  # How many simulations this strategy had
    avg_alpha: float
    avg_return: float
    avg_win_rate: float
    avg_stock_count: float
    min_alpha: float
    max_alpha: float
    # Holding period breakdown for THIS strategy
    by_holding_period: list[StrategyHoldingPeriod] = Field(default_factory=list)
    best_holding_period: int | None = None  # Which hold time has best alpha


class HoldingPeriodAggregate(BaseModel):
    """Aggregated results for a single holding period across all strategies/quarters."""

    holding_period: int  # 1, 2, 3, or 4 quarters
    simulation_count: int
    avg_alpha: float
    avg_return: float
    avg_win_rate: float


class GridSearchResult(BaseModel):
    """Complete grid search results."""

    id: str
    started_at: datetime
    completed_at: datetime | None = None
    total_simulations: int
    completed_simulations: int = 0
    duration_seconds: float | None = None
    results: list[SimulationResult] = Field(default_factory=list)
    best_by_alpha: list[SimulationResult] = Field(default_factory=list)
    best_by_win_rate: list[SimulationResult] = Field(default_factory=list)
    # Aggregated views
    by_strategy: list[StrategyAggregate] = Field(default_factory=list)
    by_holding_period: list[HoldingPeriodAggregate] = Field(default_factory=list)
    request_config: dict[str, Any] = Field(default_factory=dict)


# Constants for available dimensions
AVAILABLE_DIMENSIONS = {
    # Survival
    "altman_zone": ["safe", "grey"],
    "piotroski_min": [3, 4, 5, 6, 7, 8, 9],
    # Quality
    "quality_enabled": [True, False],
    "min_quality": ["weak", "average", "compounder"],
    # Quality tags (each can be required or not)
    "tag_durable_compounder": [True, False],
    "tag_cash_machine": [True, False],
    "tag_deep_value": [True, False],
    "tag_heavy_reinvestor": [True, False],
    "tag_premium_priced": [True, False],
    "tag_volatile_returns": [True, False],
    "tag_weak_moat_signal": [True, False],
    "tag_earnings_quality_concern": [True, False],
    # Valuation - Graham
    "graham_enabled": [True, False],
    "graham_mode": ["strict", "modern", "garp", "relaxed"],
    "graham_min": [3, 4, 5, 6, 7, 8],
    # Valuation - Magic Formula
    "magic_formula_enabled": [True, False],
    "mf_top_pct": [10, 20, 30, 50],
    # Valuation - PEG
    "peg_enabled": [True, False],
    "max_peg": [0.5, 1.0, 1.5, 2.0, 2.5, 3.0],
    # Valuation - Net-Net
    "net_net_enabled": [True, False],
    # Valuation - Fama-French
    "fama_french_enabled": [True, False],
    "ff_top_pct": [20, 30, 40, 50],
    # Valuation logic
    "min_lenses": [0, 1, 2, 3, 4, 5],
    "strict_mode": [True, False],
}

# Tag name mappings
QUALITY_TAGS = {
    "tag_durable_compounder": "Durable Compounder",
    "tag_cash_machine": "Cash Machine",
    "tag_deep_value": "Deep Value",
    "tag_heavy_reinvestor": "Heavy Reinvestor",
    "tag_premium_priced": "Premium Priced",
    "tag_volatile_returns": "Volatile Returns",
    "tag_weak_moat_signal": "Weak Moat Signal",
    "tag_earnings_quality_concern": "Earnings Quality Concern",
}
