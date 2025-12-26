"""ROIC/Quality screen analysis."""

import json
import logging
from datetime import datetime, timezone
from typing import Any

from src.analysis.base import BaseAnalyzer, to_float

logger = logging.getLogger(__name__)


class ROICQualityAnalyzer(BaseAnalyzer):
    """Analyzer for ROIC/Quality screen.

    Quality criteria:
    - ROIC >= 12%
    - Free Cash Flow positive for >= 5 years
    - Debt-to-Equity < 1.0

    Where:
    - NOPAT = EBIT × (1 - Effective Tax Rate)
    - Invested Capital = Total Debt + Equity - Cash
    - ROIC = NOPAT / Invested Capital
    - Free Cash Flow = Operating Cash Flow - CapEx
    """

    # Thresholds
    MIN_ROIC = 0.12  # 12%
    MAX_DEBT_TO_EQUITY = 1.0
    MIN_FCF_POSITIVE_YEARS = 5

    def analyze(self, symbol: str, quarter: str) -> dict[str, Any]:
        """Run ROIC/Quality analysis for a stock.

        Args:
            symbol: Stock ticker symbol.
            quarter: Analysis quarter.

        Returns:
            Dictionary with ROIC/Quality analysis results.
        """
        results = {
            "symbol": symbol,
            "analysis_quarter": quarter,
            "computed_at": datetime.now(timezone.utc),
            "ebit": None,
            "effective_tax_rate": None,
            "nopat": None,
            "total_debt": None,
            "total_equity": None,
            "cash": None,
            "invested_capital": None,
            "roic": None,
            "operating_cash_flow": None,
            "capital_expenditure": None,
            "free_cash_flow": None,
            "debt_to_equity": None,
            "roic_pass": None,
            "fcf_positive_5yr": None,
            "debt_to_equity_pass": None,
            "criteria_passed": 0,
            "data_quality": None,
            "missing_fields": [],
        }

        # Get financial data
        income_stmts = self.get_income_statements(symbol, "annual", limit=1)
        balance_sheets = self.get_balance_sheets(symbol, "annual", limit=1)
        cash_flows = self.get_cash_flow_statements(symbol, "annual", limit=5)

        if not income_stmts or not balance_sheets:
            results["data_quality"] = 0.0
            results["missing_fields"] = ["income_statements", "balance_sheets"]
            return results

        latest_income = income_stmts[0]
        latest_balance = balance_sheets[0]

        # Calculate EBIT (convert from Decimal if needed)
        ebit = to_float(latest_income.get("ebit") or latest_income.get("operating_income"))
        results["ebit"] = ebit

        # Calculate effective tax rate
        income_before_tax = to_float(latest_income.get("income_before_tax"))
        income_tax = to_float(latest_income.get("income_tax_expense"))
        if income_before_tax and income_before_tax > 0 and income_tax is not None:
            effective_tax_rate = income_tax / income_before_tax
            results["effective_tax_rate"] = max(0.0, min(1.0, effective_tax_rate))  # Clamp 0-100%
        else:
            results["effective_tax_rate"] = 0.25  # Default assumption

        # Calculate NOPAT
        if ebit is not None and results["effective_tax_rate"] is not None:
            results["nopat"] = ebit * (1.0 - results["effective_tax_rate"])

        # Get balance sheet items (convert from Decimal if needed)
        total_debt = to_float(latest_balance.get("total_debt")) or 0.0
        total_equity = to_float(
            latest_balance.get("total_equity") or latest_balance.get("total_stockholders_equity")
        )
        cash = to_float(latest_balance.get("cash_and_equivalents")) or 0.0

        results["total_debt"] = total_debt
        results["total_equity"] = total_equity
        results["cash"] = cash

        # Calculate Invested Capital
        if total_equity is not None:
            invested_capital = total_debt + total_equity - cash
            results["invested_capital"] = invested_capital

            # Calculate ROIC
            if results["nopat"] is not None and invested_capital > 0:
                results["roic"] = results["nopat"] / invested_capital
                results["roic_pass"] = results["roic"] >= self.MIN_ROIC
                if results["roic_pass"]:
                    results["criteria_passed"] += 1

            # Calculate Debt-to-Equity
            if total_equity > 0:
                results["debt_to_equity"] = total_debt / total_equity
                results["debt_to_equity_pass"] = results["debt_to_equity"] < self.MAX_DEBT_TO_EQUITY
                if results["debt_to_equity_pass"]:
                    results["criteria_passed"] += 1

        # Check Free Cash Flow for 5 years
        if cash_flows:
            latest_cf = cash_flows[0]
            results["operating_cash_flow"] = to_float(latest_cf.get("operating_cash_flow"))
            results["capital_expenditure"] = to_float(latest_cf.get("capital_expenditure"))
            results["free_cash_flow"] = to_float(latest_cf.get("free_cash_flow"))

            # Check if FCF positive for required years
            fcf_values = [to_float(cf.get("free_cash_flow")) for cf in cash_flows]
            fcf_valid = [f for f in fcf_values if f is not None]

            if len(fcf_valid) >= self.MIN_FCF_POSITIVE_YEARS:
                results["fcf_positive_5yr"] = all(f > 0 for f in fcf_valid)
            elif len(fcf_valid) > 0:
                # Partial data - check what we have
                results["fcf_positive_5yr"] = all(f > 0 for f in fcf_valid)

            if results["fcf_positive_5yr"]:
                results["criteria_passed"] += 1

        # Calculate data quality
        required_fields = ["roic", "free_cash_flow", "debt_to_equity"]
        results["data_quality"], results["missing_fields"] = self.calculate_data_quality(
            required_fields, results
        )

        return results

    def save_results(self, results: dict[str, Any]) -> None:
        """Save ROIC/Quality analysis results to database."""
        conn = self._get_conn()
        try:
            conn.execute(
                """
                INSERT INTO roic_quality_results (
                    symbol, analysis_quarter, computed_at,
                    ebit, effective_tax_rate, nopat,
                    total_debt, total_equity, cash, invested_capital,
                    roic, operating_cash_flow, capital_expenditure, free_cash_flow,
                    debt_to_equity, roic_pass, fcf_positive_5yr, debt_to_equity_pass,
                    criteria_passed, data_quality, missing_fields
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT (symbol, analysis_quarter) DO UPDATE SET
                    computed_at = EXCLUDED.computed_at,
                    roic = EXCLUDED.roic,
                    free_cash_flow = EXCLUDED.free_cash_flow,
                    debt_to_equity = EXCLUDED.debt_to_equity,
                    roic_pass = EXCLUDED.roic_pass,
                    fcf_positive_5yr = EXCLUDED.fcf_positive_5yr,
                    debt_to_equity_pass = EXCLUDED.debt_to_equity_pass,
                    criteria_passed = EXCLUDED.criteria_passed,
                    data_quality = EXCLUDED.data_quality,
                    missing_fields = EXCLUDED.missing_fields
                """,
                (
                    results["symbol"],
                    results["analysis_quarter"],
                    results["computed_at"],
                    results["ebit"],
                    results["effective_tax_rate"],
                    results["nopat"],
                    results["total_debt"],
                    results["total_equity"],
                    results["cash"],
                    results["invested_capital"],
                    results["roic"],
                    results["operating_cash_flow"],
                    results["capital_expenditure"],
                    results["free_cash_flow"],
                    results["debt_to_equity"],
                    results["roic_pass"],
                    results["fcf_positive_5yr"],
                    results["debt_to_equity_pass"],
                    results["criteria_passed"],
                    results["data_quality"],
                    json.dumps(results["missing_fields"]),
                ),
            )
        finally:
            if self._should_close_conn(conn):
                conn.close()
