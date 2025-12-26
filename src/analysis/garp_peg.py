"""GARP (Growth at a Reasonable Price) / PEG analysis."""

import json
import logging
from datetime import datetime, timezone
from typing import Any

from src.analysis.base import BaseAnalyzer, to_float

logger = logging.getLogger(__name__)


class GARPPEGAnalyzer(BaseAnalyzer):
    """Analyzer for GARP/PEG model.

    PEG Ratio = P/E / EPS Growth Rate

    Criteria:
    - EPS Growth >= 10% CAGR (3-5 year)
    - PEG <= 1.5
    """

    # Thresholds
    MIN_EPS_GROWTH = 0.10  # 10%
    MAX_PEG = 1.5

    def analyze(self, symbol: str, quarter: str) -> dict[str, Any]:
        """Run GARP/PEG analysis for a stock.

        Args:
            symbol: Stock ticker symbol.
            quarter: Analysis quarter.

        Returns:
            Dictionary with GARP/PEG analysis results.
        """
        results = {
            "symbol": symbol,
            "analysis_quarter": quarter,
            "computed_at": datetime.now(timezone.utc),
            "price": None,
            "eps": None,
            "pe_ratio": None,
            "eps_growth_1yr": None,
            "eps_growth_3yr": None,
            "eps_growth_5yr": None,
            "eps_cagr": None,
            "peg_ratio": None,
            "growth_pass": None,
            "peg_pass": None,
            "data_quality": None,
            "missing_fields": [],
        }

        # Get financial data
        income_stmts = self.get_income_statements(symbol, "annual", limit=6)
        profile = self.get_company_profile(symbol)

        if not income_stmts or not profile:
            results["data_quality"] = 0.0
            results["missing_fields"] = ["income_statements", "profile"]
            return results

        # Get current price (convert from Decimal if needed)
        price = to_float(profile.get("price"))
        results["price"] = price

        # Get EPS values (convert from Decimal if needed)
        # Data is already ordered by fiscal_date DESC, so index 0 is most recent
        eps_values = [to_float(stmt.get("eps")) for stmt in income_stmts]
        eps_values = [e for e in eps_values if e is not None]

        if len(eps_values) >= 2:
            current_eps = eps_values[0]
            results["eps"] = current_eps

            # Calculate P/E ratio from price and EPS (API doesn't provide it)
            if price is not None and current_eps is not None and current_eps > 0:
                results["pe_ratio"] = price / current_eps

            # Calculate 1-year growth
            prior_eps = eps_values[1]
            if prior_eps and prior_eps > 0:
                results["eps_growth_1yr"] = (current_eps - prior_eps) / abs(prior_eps)

            # Calculate 3-year CAGR
            if len(eps_values) >= 4:
                eps_3yr_ago = eps_values[3]
                if eps_3yr_ago and eps_3yr_ago > 0 and current_eps > 0:
                    results["eps_growth_3yr"] = (current_eps / eps_3yr_ago) ** (1.0 / 3.0) - 1.0

            # Calculate 5-year CAGR
            if len(eps_values) >= 6:
                eps_5yr_ago = eps_values[5]
                if eps_5yr_ago and eps_5yr_ago > 0 and current_eps > 0:
                    results["eps_growth_5yr"] = (current_eps / eps_5yr_ago) ** (1.0 / 5.0) - 1.0

            # Use best available CAGR
            if results["eps_growth_5yr"] is not None:
                results["eps_cagr"] = results["eps_growth_5yr"]
            elif results["eps_growth_3yr"] is not None:
                results["eps_cagr"] = results["eps_growth_3yr"]
            elif results["eps_growth_1yr"] is not None:
                results["eps_cagr"] = results["eps_growth_1yr"]

            # Check growth threshold
            if results["eps_cagr"] is not None:
                results["growth_pass"] = results["eps_cagr"] >= self.MIN_EPS_GROWTH

            # Calculate PEG ratio
            pe = results["pe_ratio"]
            growth = results["eps_cagr"]
            if pe is not None and pe > 0 and growth is not None and growth > 0:
                # Convert growth to percentage for PEG (e.g., 0.15 -> 15)
                growth_pct = growth * 100
                results["peg_ratio"] = pe / growth_pct
                results["peg_pass"] = results["peg_ratio"] <= self.MAX_PEG

        # Calculate data quality
        required_fields = ["pe_ratio", "eps_cagr", "peg_ratio"]
        results["data_quality"], results["missing_fields"] = self.calculate_data_quality(
            required_fields, results
        )

        return results

    def save_results(self, results: dict[str, Any]) -> None:
        """Save GARP/PEG analysis results to database."""
        conn = self._get_conn()
        try:
            conn.execute(
                """
                INSERT INTO garp_peg_results (
                    symbol, analysis_quarter, computed_at,
                    price, eps, pe_ratio,
                    eps_growth_1yr, eps_growth_3yr, eps_growth_5yr, eps_cagr,
                    peg_ratio, growth_pass, peg_pass,
                    data_quality, missing_fields
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT (symbol, analysis_quarter) DO UPDATE SET
                    computed_at = EXCLUDED.computed_at,
                    pe_ratio = EXCLUDED.pe_ratio,
                    eps_cagr = EXCLUDED.eps_cagr,
                    peg_ratio = EXCLUDED.peg_ratio,
                    growth_pass = EXCLUDED.growth_pass,
                    peg_pass = EXCLUDED.peg_pass,
                    data_quality = EXCLUDED.data_quality,
                    missing_fields = EXCLUDED.missing_fields
                """,
                (
                    results["symbol"],
                    results["analysis_quarter"],
                    results["computed_at"],
                    results["price"],
                    results["eps"],
                    results["pe_ratio"],
                    results["eps_growth_1yr"],
                    results["eps_growth_3yr"],
                    results["eps_growth_5yr"],
                    results["eps_cagr"],
                    results["peg_ratio"],
                    results["growth_pass"],
                    results["peg_pass"],
                    results["data_quality"],
                    json.dumps(results["missing_fields"]),
                ),
            )
        finally:
            if self._should_close_conn(conn):
                conn.close()
