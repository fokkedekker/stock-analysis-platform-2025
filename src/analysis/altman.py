"""Altman Z-Score analysis for bankruptcy prediction."""

import json
import logging
from datetime import datetime, timezone
from typing import Any

from src.analysis.base import BaseAnalyzer, to_float, quarter_to_end_date

logger = logging.getLogger(__name__)


class AltmanAnalyzer(BaseAnalyzer):
    """Analyzer for Altman Z-Score.

    Z-Score = 1.2×X1 + 1.4×X2 + 3.3×X3 + 0.6×X4 + 1.0×X5

    Where:
    - X1 = Working Capital / Total Assets
    - X2 = Retained Earnings / Total Assets
    - X3 = EBIT / Total Assets
    - X4 = Market Cap / Total Liabilities
    - X5 = Revenue / Total Assets

    Zones:
    - Z > 3.0: Safe zone
    - 1.8 < Z <= 3.0: Grey zone
    - Z <= 1.8: Distress zone
    """

    # Z-Score coefficients
    COEF_X1 = 1.2  # Working Capital / Total Assets
    COEF_X2 = 1.4  # Retained Earnings / Total Assets
    COEF_X3 = 3.3  # EBIT / Total Assets
    COEF_X4 = 0.6  # Market Cap / Total Liabilities
    COEF_X5 = 1.0  # Revenue / Total Assets

    # Zone thresholds
    SAFE_THRESHOLD = 3.0
    DISTRESS_THRESHOLD = 1.8

    def analyze(self, symbol: str, quarter: str) -> dict[str, Any]:
        """Run Altman Z-Score analysis for a stock.

        Args:
            symbol: Stock ticker symbol.
            quarter: Analysis quarter.

        Returns:
            Dictionary with Altman Z-Score analysis results.
        """
        results = {
            "symbol": symbol,
            "analysis_quarter": quarter,
            "computed_at": datetime.now(timezone.utc),
            "working_capital": None,
            "total_assets": None,
            "retained_earnings": None,
            "ebit": None,
            "market_cap": None,
            "total_liabilities": None,
            "revenue": None,
            "x1_wc_ta": None,
            "x2_re_ta": None,
            "x3_ebit_ta": None,
            "x4_mc_tl": None,
            "x5_rev_ta": None,
            "z_score": None,
            "zone": None,
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

        # Extract values (convert from Decimal if needed)
        current_assets = to_float(latest_balance.get("current_assets"))
        current_liabilities = to_float(latest_balance.get("current_liabilities"))
        total_assets = to_float(latest_balance.get("total_assets"))
        retained_earnings = to_float(latest_balance.get("retained_earnings"))
        total_liabilities = to_float(latest_balance.get("total_liabilities"))
        ebit = to_float(latest_income.get("ebit") or latest_income.get("operating_income"))
        revenue = to_float(latest_income.get("revenue"))
        market_cap = to_float(profile.get("market_cap"))

        # Store raw values
        results["total_assets"] = total_assets
        results["retained_earnings"] = retained_earnings
        results["ebit"] = ebit
        results["market_cap"] = market_cap
        results["total_liabilities"] = total_liabilities
        results["revenue"] = revenue

        # Calculate working capital
        if current_assets is not None and current_liabilities is not None:
            results["working_capital"] = current_assets - current_liabilities

        # Calculate ratios if we have total assets
        if total_assets and total_assets > 0:
            # X1: Working Capital / Total Assets
            if results["working_capital"] is not None:
                results["x1_wc_ta"] = results["working_capital"] / total_assets

            # X2: Retained Earnings / Total Assets
            if retained_earnings is not None:
                results["x2_re_ta"] = retained_earnings / total_assets

            # X3: EBIT / Total Assets
            if ebit is not None:
                results["x3_ebit_ta"] = ebit / total_assets

            # X5: Revenue / Total Assets
            if revenue is not None:
                results["x5_rev_ta"] = revenue / total_assets

        # X4: Market Cap / Total Liabilities
        if market_cap is not None and total_liabilities and total_liabilities > 0:
            results["x4_mc_tl"] = market_cap / total_liabilities

        # Calculate Z-Score if we have all components
        x1 = results["x1_wc_ta"]
        x2 = results["x2_re_ta"]
        x3 = results["x3_ebit_ta"]
        x4 = results["x4_mc_tl"]
        x5 = results["x5_rev_ta"]

        if all(x is not None for x in [x1, x2, x3, x4, x5]):
            z_score = (
                self.COEF_X1 * x1
                + self.COEF_X2 * x2
                + self.COEF_X3 * x3
                + self.COEF_X4 * x4
                + self.COEF_X5 * x5
            )
            results["z_score"] = z_score

            # Determine zone
            if z_score > self.SAFE_THRESHOLD:
                results["zone"] = "safe"
            elif z_score > self.DISTRESS_THRESHOLD:
                results["zone"] = "grey"
            else:
                results["zone"] = "distress"

        # Calculate data quality
        required_fields = ["x1_wc_ta", "x2_re_ta", "x3_ebit_ta", "x4_mc_tl", "x5_rev_ta"]
        results["data_quality"], results["missing_fields"] = self.calculate_data_quality(
            required_fields, results
        )

        return results

    def save_results(self, results: dict[str, Any]) -> None:
        """Save Altman Z-Score analysis results to database."""
        conn = self._get_conn()
        try:
            conn.execute(
                """
                INSERT INTO altman_results (
                    symbol, analysis_quarter, computed_at,
                    working_capital, total_assets, retained_earnings, ebit,
                    market_cap, total_liabilities, revenue,
                    x1_wc_ta, x2_re_ta, x3_ebit_ta, x4_mc_tl, x5_rev_ta,
                    z_score, zone, data_quality, missing_fields
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT (symbol, analysis_quarter) DO UPDATE SET
                    computed_at = EXCLUDED.computed_at,
                    working_capital = EXCLUDED.working_capital,
                    total_assets = EXCLUDED.total_assets,
                    z_score = EXCLUDED.z_score,
                    zone = EXCLUDED.zone,
                    data_quality = EXCLUDED.data_quality,
                    missing_fields = EXCLUDED.missing_fields
                """,
                (
                    results["symbol"],
                    results["analysis_quarter"],
                    results["computed_at"],
                    results["working_capital"],
                    results["total_assets"],
                    results["retained_earnings"],
                    results["ebit"],
                    results["market_cap"],
                    results["total_liabilities"],
                    results["revenue"],
                    results["x1_wc_ta"],
                    results["x2_re_ta"],
                    results["x3_ebit_ta"],
                    results["x4_mc_tl"],
                    results["x5_rev_ta"],
                    results["z_score"],
                    results["zone"],
                    results["data_quality"],
                    json.dumps(results["missing_fields"]),
                ),
            )
        finally:
            if self._should_close_conn(conn):
                conn.close()
