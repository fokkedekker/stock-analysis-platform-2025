"""Pydantic models for API responses and analysis results."""

from src.models.financial_data import (
    Ticker,
    CompanyProfile,
    IncomeStatement,
    BalanceSheet,
    CashFlowStatement,
    KeyMetrics,
    Dividend,
)
from src.models.analysis_results import (
    GrahamResult,
    MagicFormulaResult,
    PiotroskiResult,
    AltmanResult,
    ROICQualityResult,
    GARPPEGResult,
    FamaFrenchResult,
    NetNetResult,
    StockRanking,
    AnalysisSummary,
)

__all__ = [
    # Financial data models
    "Ticker",
    "CompanyProfile",
    "IncomeStatement",
    "BalanceSheet",
    "CashFlowStatement",
    "KeyMetrics",
    "Dividend",
    # Analysis result models
    "GrahamResult",
    "MagicFormulaResult",
    "PiotroskiResult",
    "AltmanResult",
    "ROICQualityResult",
    "GARPPEGResult",
    "FamaFrenchResult",
    "NetNetResult",
    "StockRanking",
    "AnalysisSummary",
]
