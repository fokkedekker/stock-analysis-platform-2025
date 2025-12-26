"""Pydantic models for analysis results."""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


class BaseAnalysisResult(BaseModel):
    """Base model for analysis results."""

    symbol: str
    analysis_quarter: str
    computed_at: datetime | None = None
    data_quality: float | None = Field(
        None, ge=0, le=1, description="Data completeness score (0-1)"
    )
    missing_fields: list[str] | None = None


class GrahamResult(BaseAnalysisResult):
    """Benjamin Graham 7 criteria analysis result."""

    mode: Literal["strict", "modern", "garp", "relaxed"] = "strict"

    # Individual criteria pass/fail
    adequate_size: bool | None = None
    current_ratio_pass: bool | None = None
    debt_coverage_pass: bool | None = None
    earnings_stability: bool | None = None
    dividend_record: bool | None = None
    earnings_growth_pass: bool | None = None
    pe_ratio_pass: bool | None = None
    pb_ratio_pass: bool | None = None

    # Computed values
    revenue: float | None = None
    current_ratio: float | None = None
    net_current_assets: float | None = None
    long_term_debt: float | None = None
    eps_5yr_growth: float | None = None
    pe_ratio: float | None = None
    pb_ratio: float | None = None
    pe_x_pb: float | None = None

    # Score
    criteria_passed: int | None = Field(None, ge=0, le=8)


class MagicFormulaResult(BaseAnalysisResult):
    """Joel Greenblatt Magic Formula analysis result."""

    # Computed values
    ebit: float | None = None
    enterprise_value: float | None = None
    earnings_yield: float | None = None
    net_working_capital: float | None = None
    net_fixed_assets: float | None = None
    return_on_capital: float | None = None

    # Rankings (within universe)
    earnings_yield_rank: int | None = None
    return_on_capital_rank: int | None = None
    combined_rank: int | None = None


class PiotroskiResult(BaseAnalysisResult):
    """Piotroski F-Score analysis result."""

    # Profitability signals (4)
    roa_positive: bool | None = None
    operating_cf_positive: bool | None = None
    roa_increasing: bool | None = None
    accruals_signal: bool | None = None

    # Leverage signals (3)
    leverage_decreasing: bool | None = None
    current_ratio_increasing: bool | None = None
    no_dilution: bool | None = None

    # Efficiency signals (2)
    gross_margin_increasing: bool | None = None
    asset_turnover_increasing: bool | None = None

    # Computed values
    roa: float | None = None
    roa_prior: float | None = None
    operating_cash_flow: float | None = None
    net_income: float | None = None
    long_term_debt: float | None = None
    long_term_debt_prior: float | None = None
    current_ratio: float | None = None
    current_ratio_prior: float | None = None
    shares_outstanding: int | None = None
    shares_outstanding_prior: int | None = None
    gross_margin: float | None = None
    gross_margin_prior: float | None = None
    asset_turnover: float | None = None
    asset_turnover_prior: float | None = None

    # Score
    f_score: int | None = Field(None, ge=0, le=9)


class AltmanResult(BaseAnalysisResult):
    """Altman Z-Score analysis result."""

    # Components
    working_capital: float | None = None
    total_assets: float | None = None
    retained_earnings: float | None = None
    ebit: float | None = None
    market_cap: float | None = None
    total_liabilities: float | None = None
    revenue: float | None = None

    # Ratios (X1-X5)
    x1_wc_ta: float | None = None
    x2_re_ta: float | None = None
    x3_ebit_ta: float | None = None
    x4_mc_tl: float | None = None
    x5_rev_ta: float | None = None

    # Score and zone
    z_score: float | None = None
    zone: Literal["safe", "grey", "distress"] | None = None


class ROICQualityResult(BaseAnalysisResult):
    """ROIC/Quality screen analysis result."""

    # Computed values
    ebit: float | None = None
    effective_tax_rate: float | None = None
    nopat: float | None = None
    total_debt: float | None = None
    total_equity: float | None = None
    cash: float | None = None
    invested_capital: float | None = None
    roic: float | None = None
    operating_cash_flow: float | None = None
    capital_expenditure: float | None = None
    free_cash_flow: float | None = None
    debt_to_equity: float | None = None

    # Pass/fail criteria
    roic_pass: bool | None = None
    fcf_positive_5yr: bool | None = None
    debt_to_equity_pass: bool | None = None

    # Score
    criteria_passed: int | None = Field(None, ge=0, le=3)


class GARPPEGResult(BaseAnalysisResult):
    """GARP/PEG model analysis result."""

    # Computed values
    price: float | None = None
    eps: float | None = None
    pe_ratio: float | None = None
    eps_growth_1yr: float | None = None
    eps_growth_3yr: float | None = None
    eps_growth_5yr: float | None = None
    eps_cagr: float | None = None
    peg_ratio: float | None = None

    # Pass/fail
    growth_pass: bool | None = None
    peg_pass: bool | None = None


class FamaFrenchResult(BaseAnalysisResult):
    """Fama-French factor inputs analysis result."""

    # Computed values
    book_value: float | None = None
    market_cap: float | None = None
    book_to_market: float | None = None
    gross_profit: float | None = None
    total_assets: float | None = None
    profitability: float | None = None
    assets_prior: float | None = None
    asset_growth: float | None = None

    # Percentile rankings
    book_to_market_percentile: float | None = None
    profitability_percentile: float | None = None
    asset_growth_percentile: float | None = None


class NetNetResult(BaseAnalysisResult):
    """Net-Net deep value analysis result."""

    # Computed values
    current_assets: float | None = None
    total_liabilities: float | None = None
    ncav: float | None = None
    market_cap: float | None = None
    ncav_per_share: float | None = None
    price: float | None = None
    discount_to_ncav: float | None = None

    # Pass/fail
    trading_below_ncav: bool | None = None
    deep_value: bool | None = None  # Trading below 67% of NCAV


class StockRanking(BaseModel):
    """Combined stock ranking across all valuation systems."""

    symbol: str
    analysis_quarter: str
    computed_at: datetime | None = None

    # Individual scores
    graham_score: int | None = None
    graham_mode: str | None = None
    magic_formula_rank: int | None = None
    piotroski_score: int | None = None
    altman_zone: str | None = None
    altman_z_score: float | None = None
    roic_pass: bool | None = None
    roic: float | None = None
    peg_ratio: float | None = None
    peg_pass: bool | None = None
    net_net_discount: float | None = None
    net_net_pass: bool | None = None

    # Composite score
    composite_score: float | None = None


class AnalysisSummary(BaseModel):
    """Summary of all analyses for a single stock."""

    symbol: str
    analysis_quarter: str

    graham: GrahamResult | None = None
    magic_formula: MagicFormulaResult | None = None
    piotroski: PiotroskiResult | None = None
    altman: AltmanResult | None = None
    roic_quality: ROICQualityResult | None = None
    garp_peg: GARPPEGResult | None = None
    fama_french: FamaFrenchResult | None = None
    net_net: NetNetResult | None = None
    ranking: StockRanking | None = None
