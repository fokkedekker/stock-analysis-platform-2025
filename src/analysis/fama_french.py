"""Fama-French factor inputs analysis."""

import json
import logging
from datetime import datetime, timezone
from typing import Any

from src.analysis.base import BaseAnalyzer, to_float

logger = logging.getLogger(__name__)


class FamaFrenchAnalyzer(BaseAnalyzer):
    """Analyzer for Fama-French factor inputs.

    Factors:
    - Book-to-Market = Book Value / Market Cap (Value factor)
    - Profitability = Gross Profit / Total Assets (RMW factor)
    - Asset Growth = (Assets_t - Assets_t-1) / Assets_t-1 (CMA factor)

    Typical factor exposure thresholds:
    - Book-to-Market: Top 30% = Value
    - Profitability: Top 30% = Robust
    - Asset Growth: Bottom 30% = Conservative
    """

    def analyze(self, symbol: str, quarter: str) -> dict[str, Any]:
        """Run Fama-French factor analysis for a stock.

        Args:
            symbol: Stock ticker symbol.
            quarter: Analysis quarter.

        Returns:
            Dictionary with Fama-French factor analysis results.
        """
        results = {
            "symbol": symbol,
            "analysis_quarter": quarter,
            "computed_at": datetime.now(timezone.utc),
            "book_value": None,
            "market_cap": None,
            "book_to_market": None,
            "gross_profit": None,
            "total_assets": None,
            "profitability": None,
            "assets_prior": None,
            "asset_growth": None,
            "book_to_market_percentile": None,
            "profitability_percentile": None,
            "asset_growth_percentile": None,
            "data_quality": None,
            "missing_fields": [],
        }

        # Get financial data
        income_stmts = self.get_income_statements(symbol, "annual", limit=1)
        balance_sheets = self.get_balance_sheets(symbol, "annual", limit=2)
        profile = self.get_company_profile(symbol)

        if not income_stmts or not balance_sheets or not profile:
            results["data_quality"] = 0.0
            results["missing_fields"] = ["income_statements", "balance_sheets", "profile"]
            return results

        latest_income = income_stmts[0]
        latest_balance = balance_sheets[0]
        prior_balance = balance_sheets[1] if len(balance_sheets) > 1 else {}

        # Book-to-Market
        book_value = latest_balance.get("total_equity") or latest_balance.get(
            "total_stockholders_equity"
        )
        market_cap = profile.get("market_cap")

        results["book_value"] = book_value
        results["market_cap"] = market_cap

        if book_value is not None and market_cap is not None and market_cap > 0:
            results["book_to_market"] = book_value / market_cap

        # Profitability (RMW)
        gross_profit = latest_income.get("gross_profit")
        total_assets = latest_balance.get("total_assets")

        results["gross_profit"] = gross_profit
        results["total_assets"] = total_assets

        if gross_profit is not None and total_assets is not None and total_assets > 0:
            results["profitability"] = gross_profit / total_assets

        # Asset Growth (CMA)
        assets_prior = prior_balance.get("total_assets")
        results["assets_prior"] = assets_prior

        if total_assets is not None and assets_prior is not None and assets_prior > 0:
            results["asset_growth"] = (total_assets - assets_prior) / assets_prior

        # Calculate data quality
        required_fields = ["book_to_market", "profitability", "asset_growth"]
        results["data_quality"], results["missing_fields"] = self.calculate_data_quality(
            required_fields, results
        )

        return results

    def save_results(self, results: dict[str, Any]) -> None:
        """Save Fama-French factor analysis results to database."""
        conn = self._get_conn()
        try:
            conn.execute(
                """
                INSERT INTO fama_french_results (
                    symbol, analysis_quarter, computed_at,
                    book_value, market_cap, book_to_market,
                    gross_profit, total_assets, profitability,
                    assets_prior, asset_growth,
                    book_to_market_percentile, profitability_percentile, asset_growth_percentile,
                    data_quality, missing_fields
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT (symbol, analysis_quarter) DO UPDATE SET
                    computed_at = EXCLUDED.computed_at,
                    book_to_market = EXCLUDED.book_to_market,
                    profitability = EXCLUDED.profitability,
                    asset_growth = EXCLUDED.asset_growth,
                    data_quality = EXCLUDED.data_quality,
                    missing_fields = EXCLUDED.missing_fields
                """,
                (
                    results["symbol"],
                    results["analysis_quarter"],
                    results["computed_at"],
                    results["book_value"],
                    results["market_cap"],
                    results["book_to_market"],
                    results["gross_profit"],
                    results["total_assets"],
                    results["profitability"],
                    results["assets_prior"],
                    results["asset_growth"],
                    results["book_to_market_percentile"],
                    results["profitability_percentile"],
                    results["asset_growth_percentile"],
                    results["data_quality"],
                    json.dumps(results["missing_fields"]),
                ),
            )
        finally:
            if self._should_close_conn(conn):
                conn.close()

    def compute_percentiles(self, quarter: str) -> None:
        """Compute percentile rankings across all stocks for the quarter.

        Args:
            quarter: Analysis quarter.
        """
        with self.db.get_connection() as conn:
            # Book-to-Market percentile
            conn.execute(
                """
                UPDATE fama_french_results
                SET book_to_market_percentile = ranked.pctl
                FROM (
                    SELECT symbol,
                           PERCENT_RANK() OVER (ORDER BY book_to_market) as pctl
                    FROM fama_french_results
                    WHERE analysis_quarter = ?
                    AND book_to_market IS NOT NULL
                ) ranked
                WHERE fama_french_results.symbol = ranked.symbol
                AND fama_french_results.analysis_quarter = ?
                """,
                (quarter, quarter),
            )

            # Profitability percentile
            conn.execute(
                """
                UPDATE fama_french_results
                SET profitability_percentile = ranked.pctl
                FROM (
                    SELECT symbol,
                           PERCENT_RANK() OVER (ORDER BY profitability) as pctl
                    FROM fama_french_results
                    WHERE analysis_quarter = ?
                    AND profitability IS NOT NULL
                ) ranked
                WHERE fama_french_results.symbol = ranked.symbol
                AND fama_french_results.analysis_quarter = ?
                """,
                (quarter, quarter),
            )

            # Asset Growth percentile (inverted - lower is better)
            conn.execute(
                """
                UPDATE fama_french_results
                SET asset_growth_percentile = ranked.pctl
                FROM (
                    SELECT symbol,
                           PERCENT_RANK() OVER (ORDER BY asset_growth DESC) as pctl
                    FROM fama_french_results
                    WHERE analysis_quarter = ?
                    AND asset_growth IS NOT NULL
                ) ranked
                WHERE fama_french_results.symbol = ranked.symbol
                AND fama_french_results.analysis_quarter = ?
                """,
                (quarter, quarter),
            )

            logger.info(f"Computed Fama-French percentiles for {quarter}")
