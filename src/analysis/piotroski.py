"""Piotroski F-Score analysis (9 binary signals)."""

import json
import logging
from datetime import datetime, timezone
from typing import Any

from src.analysis.base import BaseAnalyzer, to_float, quarter_to_end_date

logger = logging.getLogger(__name__)


class PiotroskiAnalyzer(BaseAnalyzer):
    """Analyzer for Piotroski F-Score.

    F-Score consists of 9 binary signals:
    - Profitability (4 points): ROA > 0, CFO > 0, ΔROA > 0, CFO > Net Income
    - Leverage/Liquidity (3 points): ΔLT Debt < 0, ΔCurrent Ratio > 0, No dilution
    - Operating Efficiency (2 points): ΔGross Margin > 0, ΔAsset Turnover > 0
    """

    def analyze(self, symbol: str, quarter: str) -> dict[str, Any]:
        """Run Piotroski F-Score analysis for a stock.

        Args:
            symbol: Stock ticker symbol.
            quarter: Analysis quarter.

        Returns:
            Dictionary with Piotroski analysis results.
        """
        results = {
            "symbol": symbol,
            "analysis_quarter": quarter,
            "computed_at": datetime.now(timezone.utc),
            # Profitability signals
            "roa_positive": None,
            "operating_cf_positive": None,
            "roa_increasing": None,
            "accruals_signal": None,
            # Leverage signals
            "leverage_decreasing": None,
            "current_ratio_increasing": None,
            "no_dilution": None,
            # Efficiency signals
            "gross_margin_increasing": None,
            "asset_turnover_increasing": None,
            # Values
            "roa": None,
            "roa_prior": None,
            "operating_cash_flow": None,
            "net_income": None,
            "long_term_debt": None,
            "long_term_debt_prior": None,
            "current_ratio": None,
            "current_ratio_prior": None,
            "shares_outstanding": None,
            "shares_outstanding_prior": None,
            "gross_margin": None,
            "gross_margin_prior": None,
            "asset_turnover": None,
            "asset_turnover_prior": None,
            # Score
            "f_score": 0,
            "data_quality": None,
            "missing_fields": [],
        }

        # Get financial data (need 2 years for YoY comparison)
        # Filtered by quarter end date for point-in-time analysis
        as_of = quarter_to_end_date(quarter)
        income_stmts = self.get_income_statements(symbol, "annual", limit=2, as_of_date=as_of)
        balance_sheets = self.get_balance_sheets(symbol, "annual", limit=2, as_of_date=as_of)
        cash_flows = self.get_cash_flow_statements(symbol, "annual", limit=1, as_of_date=as_of)

        if not income_stmts or not balance_sheets:
            results["data_quality"] = 0.0
            results["missing_fields"] = ["income_statements", "balance_sheets"]
            return results

        latest_income = income_stmts[0]
        prior_income = income_stmts[1] if len(income_stmts) > 1 else {}
        latest_balance = balance_sheets[0]
        prior_balance = balance_sheets[1] if len(balance_sheets) > 1 else {}
        latest_cf = cash_flows[0] if cash_flows else {}

        # === PROFITABILITY (4 signals) ===

        # 1. ROA > 0 (safe_divide handles Decimal conversion)
        net_income = to_float(latest_income.get("net_income"))
        total_assets = to_float(latest_balance.get("total_assets"))
        roa = self.safe_divide(net_income, total_assets)
        results["roa"] = roa
        results["net_income"] = net_income
        if roa is not None:
            results["roa_positive"] = roa > 0
            if results["roa_positive"]:
                results["f_score"] += 1

        # 2. Operating Cash Flow > 0
        operating_cf = to_float(latest_cf.get("operating_cash_flow"))
        results["operating_cash_flow"] = operating_cf
        if operating_cf is not None:
            results["operating_cf_positive"] = operating_cf > 0
            if results["operating_cf_positive"]:
                results["f_score"] += 1

        # 3. ΔROA > 0 (ROA increasing)
        prior_net_income = to_float(prior_income.get("net_income"))
        prior_total_assets = to_float(prior_balance.get("total_assets"))
        roa_prior = self.safe_divide(prior_net_income, prior_total_assets)
        results["roa_prior"] = roa_prior
        if roa is not None and roa_prior is not None:
            results["roa_increasing"] = roa > roa_prior
            if results["roa_increasing"]:
                results["f_score"] += 1

        # 4. CFO > Net Income (accruals signal)
        if operating_cf is not None and net_income is not None:
            results["accruals_signal"] = operating_cf > net_income
            if results["accruals_signal"]:
                results["f_score"] += 1

        # === LEVERAGE / LIQUIDITY (3 signals) ===

        # 5. Long-term debt decreased
        lt_debt = to_float(latest_balance.get("long_term_debt")) or 0.0
        lt_debt_prior = to_float(prior_balance.get("long_term_debt")) or 0.0
        results["long_term_debt"] = lt_debt
        results["long_term_debt_prior"] = lt_debt_prior
        results["leverage_decreasing"] = lt_debt < lt_debt_prior
        if results["leverage_decreasing"]:
            results["f_score"] += 1

        # 6. Current ratio increased
        current_assets = latest_balance.get("current_assets")
        current_liabilities = latest_balance.get("current_liabilities")
        current_ratio = self.safe_divide(current_assets, current_liabilities)
        results["current_ratio"] = current_ratio

        prior_current_assets = prior_balance.get("current_assets")
        prior_current_liabilities = prior_balance.get("current_liabilities")
        current_ratio_prior = self.safe_divide(prior_current_assets, prior_current_liabilities)
        results["current_ratio_prior"] = current_ratio_prior

        if current_ratio is not None and current_ratio_prior is not None:
            results["current_ratio_increasing"] = current_ratio > current_ratio_prior
            if results["current_ratio_increasing"]:
                results["f_score"] += 1

        # 7. No dilution (shares outstanding didn't increase)
        shares = latest_income.get("weighted_avg_shares") or latest_balance.get(
            "common_shares_outstanding"
        )
        shares_prior = prior_income.get("weighted_avg_shares") or prior_balance.get(
            "common_shares_outstanding"
        )
        results["shares_outstanding"] = shares
        results["shares_outstanding_prior"] = shares_prior
        if shares is not None and shares_prior is not None:
            results["no_dilution"] = shares <= shares_prior
            if results["no_dilution"]:
                results["f_score"] += 1

        # === OPERATING EFFICIENCY (2 signals) ===

        # 8. Gross margin increased
        gross_profit = latest_income.get("gross_profit")
        revenue = latest_income.get("revenue")
        gross_margin = self.safe_divide(gross_profit, revenue)
        results["gross_margin"] = gross_margin

        prior_gross_profit = prior_income.get("gross_profit")
        prior_revenue = prior_income.get("revenue")
        gross_margin_prior = self.safe_divide(prior_gross_profit, prior_revenue)
        results["gross_margin_prior"] = gross_margin_prior

        if gross_margin is not None and gross_margin_prior is not None:
            results["gross_margin_increasing"] = gross_margin > gross_margin_prior
            if results["gross_margin_increasing"]:
                results["f_score"] += 1

        # 9. Asset turnover increased
        asset_turnover = self.safe_divide(revenue, total_assets)
        results["asset_turnover"] = asset_turnover

        asset_turnover_prior = self.safe_divide(prior_revenue, prior_total_assets)
        results["asset_turnover_prior"] = asset_turnover_prior

        if asset_turnover is not None and asset_turnover_prior is not None:
            results["asset_turnover_increasing"] = asset_turnover > asset_turnover_prior
            if results["asset_turnover_increasing"]:
                results["f_score"] += 1

        # Calculate data quality
        required_fields = ["roa", "operating_cash_flow", "current_ratio", "gross_margin"]
        results["data_quality"], results["missing_fields"] = self.calculate_data_quality(
            required_fields, results
        )

        return results

    def save_results(self, results: dict[str, Any]) -> None:
        """Save Piotroski analysis results to database."""
        conn = self._get_conn()
        try:
            conn.execute(
                """
                INSERT INTO piotroski_results (
                    symbol, analysis_quarter, computed_at,
                    roa_positive, operating_cf_positive, roa_increasing, accruals_signal,
                    leverage_decreasing, current_ratio_increasing, no_dilution,
                    gross_margin_increasing, asset_turnover_increasing,
                    roa, roa_prior, operating_cash_flow, net_income,
                    long_term_debt, long_term_debt_prior,
                    current_ratio, current_ratio_prior,
                    shares_outstanding, shares_outstanding_prior,
                    gross_margin, gross_margin_prior,
                    asset_turnover, asset_turnover_prior,
                    f_score, data_quality, missing_fields
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT (symbol, analysis_quarter) DO UPDATE SET
                    computed_at = EXCLUDED.computed_at,
                    roa_positive = EXCLUDED.roa_positive,
                    operating_cf_positive = EXCLUDED.operating_cf_positive,
                    roa_increasing = EXCLUDED.roa_increasing,
                    accruals_signal = EXCLUDED.accruals_signal,
                    leverage_decreasing = EXCLUDED.leverage_decreasing,
                    current_ratio_increasing = EXCLUDED.current_ratio_increasing,
                    no_dilution = EXCLUDED.no_dilution,
                    gross_margin_increasing = EXCLUDED.gross_margin_increasing,
                    asset_turnover_increasing = EXCLUDED.asset_turnover_increasing,
                    f_score = EXCLUDED.f_score,
                    data_quality = EXCLUDED.data_quality,
                    missing_fields = EXCLUDED.missing_fields
                """,
                (
                    results["symbol"],
                    results["analysis_quarter"],
                    results["computed_at"],
                    results["roa_positive"],
                    results["operating_cf_positive"],
                    results["roa_increasing"],
                    results["accruals_signal"],
                    results["leverage_decreasing"],
                    results["current_ratio_increasing"],
                    results["no_dilution"],
                    results["gross_margin_increasing"],
                    results["asset_turnover_increasing"],
                    results["roa"],
                    results["roa_prior"],
                    results["operating_cash_flow"],
                    results["net_income"],
                    results["long_term_debt"],
                    results["long_term_debt_prior"],
                    results["current_ratio"],
                    results["current_ratio_prior"],
                    results["shares_outstanding"],
                    results["shares_outstanding_prior"],
                    results["gross_margin"],
                    results["gross_margin_prior"],
                    results["asset_turnover"],
                    results["asset_turnover_prior"],
                    results["f_score"],
                    results["data_quality"],
                    json.dumps(results["missing_fields"]),
                ),
            )
        finally:
            if self._should_close_conn(conn):
                conn.close()
