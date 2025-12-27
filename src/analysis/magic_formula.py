"""Joel Greenblatt Magic Formula analysis."""

import json
import logging
from datetime import datetime, timezone
from typing import Any

from src.analysis.base import BaseAnalyzer, to_float, quarter_to_end_date

logger = logging.getLogger(__name__)


class MagicFormulaAnalyzer(BaseAnalyzer):
    """Analyzer for Joel Greenblatt's Magic Formula.

    Magic Formula ranks stocks by:
    1. Earnings Yield = EBIT / Enterprise Value
    2. Return on Capital = EBIT / (Net Working Capital + Net Fixed Assets)

    Stocks are ranked on each metric, then combined for a final rank.
    """

    def analyze(self, symbol: str, quarter: str) -> dict[str, Any]:
        """Run Magic Formula analysis for a stock.

        Args:
            symbol: Stock ticker symbol.
            quarter: Analysis quarter.

        Returns:
            Dictionary with Magic Formula analysis results.
        """
        results = {
            "symbol": symbol,
            "analysis_quarter": quarter,
            "computed_at": datetime.now(timezone.utc),
            "ebit": None,
            "enterprise_value": None,
            "earnings_yield": None,
            "net_working_capital": None,
            "net_fixed_assets": None,
            "return_on_capital": None,
            "earnings_yield_rank": None,
            "return_on_capital_rank": None,
            "combined_rank": None,
            "data_quality": None,
            "missing_fields": [],
        }

        # Get financial data (filtered by quarter end date for point-in-time analysis)
        as_of = quarter_to_end_date(quarter)
        income_stmts = self.get_income_statements(symbol, "annual", limit=1, as_of_date=as_of)
        balance_sheets = self.get_balance_sheets(symbol, "annual", limit=1, as_of_date=as_of)
        profile = self.get_company_profile(symbol, quarter=quarter)

        if not income_stmts or not balance_sheets or not profile:
            results["data_quality"] = 0.0
            results["missing_fields"] = ["income_statements", "balance_sheets", "profile"]
            return results

        latest_income = income_stmts[0]
        latest_balance = balance_sheets[0]

        # Calculate EBIT (convert from Decimal if needed)
        ebit = to_float(latest_income.get("ebit") or latest_income.get("operating_income"))
        results["ebit"] = ebit

        # Calculate Enterprise Value
        # EV = Market Cap + Total Debt - Cash
        market_cap = to_float(profile.get("market_cap"))
        total_debt = to_float(latest_balance.get("total_debt")) or 0.0
        cash = to_float(latest_balance.get("cash_and_equivalents")) or 0.0

        if market_cap is not None:
            enterprise_value = market_cap + total_debt - cash
            results["enterprise_value"] = enterprise_value

            # Calculate Earnings Yield
            if enterprise_value > 0 and ebit is not None:
                results["earnings_yield"] = ebit / enterprise_value

        # Calculate Net Working Capital
        current_assets = to_float(latest_balance.get("current_assets")) or 0.0
        current_liabilities = to_float(latest_balance.get("current_liabilities")) or 0.0
        net_working_capital = current_assets - current_liabilities
        results["net_working_capital"] = net_working_capital

        # Calculate Net Fixed Assets
        ppe = to_float(latest_balance.get("property_plant_equipment")) or 0.0
        results["net_fixed_assets"] = ppe

        # Calculate Return on Capital
        invested_capital = net_working_capital + ppe
        if invested_capital > 0 and ebit is not None:
            results["return_on_capital"] = ebit / invested_capital

        # Calculate data quality
        required_fields = ["ebit", "enterprise_value", "earnings_yield", "return_on_capital"]
        results["data_quality"], results["missing_fields"] = self.calculate_data_quality(
            required_fields, results
        )

        return results

    def save_results(self, results: dict[str, Any]) -> None:
        """Save Magic Formula analysis results to database."""
        conn = self._get_conn()
        try:
            conn.execute(
                """
                INSERT INTO magic_formula_results (
                    symbol, analysis_quarter, computed_at,
                    ebit, enterprise_value, earnings_yield,
                    net_working_capital, net_fixed_assets, return_on_capital,
                    earnings_yield_rank, return_on_capital_rank, combined_rank,
                    data_quality, missing_fields
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT (symbol, analysis_quarter) DO UPDATE SET
                    computed_at = EXCLUDED.computed_at,
                    ebit = EXCLUDED.ebit,
                    enterprise_value = EXCLUDED.enterprise_value,
                    earnings_yield = EXCLUDED.earnings_yield,
                    net_working_capital = EXCLUDED.net_working_capital,
                    net_fixed_assets = EXCLUDED.net_fixed_assets,
                    return_on_capital = EXCLUDED.return_on_capital,
                    data_quality = EXCLUDED.data_quality,
                    missing_fields = EXCLUDED.missing_fields
                """,
                (
                    results["symbol"],
                    results["analysis_quarter"],
                    results["computed_at"],
                    results["ebit"],
                    results["enterprise_value"],
                    results["earnings_yield"],
                    results["net_working_capital"],
                    results["net_fixed_assets"],
                    results["return_on_capital"],
                    results["earnings_yield_rank"],
                    results["return_on_capital_rank"],
                    results["combined_rank"],
                    results["data_quality"],
                    json.dumps(results["missing_fields"]),
                ),
            )
        finally:
            if self._should_close_conn(conn):
                conn.close()

    def compute_rankings(self, quarter: str) -> None:
        """Compute rankings across all stocks for the quarter.

        This should be called after all individual analyses are complete.

        Args:
            quarter: Analysis quarter.
        """
        with self.db.get_connection() as conn:
            # Rank by earnings yield (higher is better)
            conn.execute(
                """
                UPDATE magic_formula_results
                SET earnings_yield_rank = ranked.rank
                FROM (
                    SELECT symbol,
                           ROW_NUMBER() OVER (ORDER BY earnings_yield DESC NULLS LAST) as rank
                    FROM magic_formula_results
                    WHERE analysis_quarter = ?
                    AND earnings_yield IS NOT NULL
                ) ranked
                WHERE magic_formula_results.symbol = ranked.symbol
                AND magic_formula_results.analysis_quarter = ?
                """,
                (quarter, quarter),
            )

            # Rank by return on capital (higher is better)
            conn.execute(
                """
                UPDATE magic_formula_results
                SET return_on_capital_rank = ranked.rank
                FROM (
                    SELECT symbol,
                           ROW_NUMBER() OVER (ORDER BY return_on_capital DESC NULLS LAST) as rank
                    FROM magic_formula_results
                    WHERE analysis_quarter = ?
                    AND return_on_capital IS NOT NULL
                ) ranked
                WHERE magic_formula_results.symbol = ranked.symbol
                AND magic_formula_results.analysis_quarter = ?
                """,
                (quarter, quarter),
            )

            # Compute combined rank (sum of both ranks, lower is better)
            conn.execute(
                """
                UPDATE magic_formula_results
                SET combined_rank = ranked.rank
                FROM (
                    SELECT symbol,
                           ROW_NUMBER() OVER (
                               ORDER BY (COALESCE(earnings_yield_rank, 9999) +
                                        COALESCE(return_on_capital_rank, 9999))
                           ) as rank
                    FROM magic_formula_results
                    WHERE analysis_quarter = ?
                ) ranked
                WHERE magic_formula_results.symbol = ranked.symbol
                AND magic_formula_results.analysis_quarter = ?
                """,
                (quarter, quarter),
            )

            logger.info(f"Computed Magic Formula rankings for {quarter}")
