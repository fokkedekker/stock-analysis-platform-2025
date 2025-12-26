"""Benjamin Graham 7 criteria analysis."""

import json
import logging
from datetime import datetime, timezone
from typing import Any, Literal

from src.analysis.base import BaseAnalyzer, to_float

logger = logging.getLogger(__name__)

# Graham criteria thresholds by mode
THRESHOLDS = {
    "strict": {
        "min_revenue": 500_000_000,  # $500M
        "min_current_ratio": 2.0,
        "min_eps_years": 5,
        "min_dividend_years": 5,
        "min_eps_growth": 0.33,  # 33%
        "max_pe": 15,
        "max_pb": 1.5,
        "max_pe_x_pb": 22.5,
    },
    "modern": {
        "min_revenue": 500_000_000,
        "min_current_ratio": 1.5,
        "min_eps_years": 5,
        "min_dividend_years": 5,
        "min_eps_growth": 0.20,
        "max_pe": 25,
        "max_pb": 3.0,
        "max_pe_x_pb": 50,
    },
    "garp": {
        "min_revenue": 100_000_000,
        "min_current_ratio": 1.0,
        "min_eps_years": 3,
        "min_dividend_years": 0,  # No dividend requirement
        "min_eps_growth": 0.15,
        "max_pe": 30,
        "max_pb": 10.0,  # No P/B requirement
        "max_pe_x_pb": 100,
    },
    "relaxed": {
        "min_revenue": 200_000_000,
        "min_current_ratio": 1.2,
        "min_eps_years": 3,
        "min_dividend_years": 3,
        "min_eps_growth": 0.15,
        "max_pe": 20,
        "max_pb": 2.5,
        "max_pe_x_pb": 35,
    },
}


class GrahamAnalyzer(BaseAnalyzer):
    """Analyzer for Benjamin Graham's 7 criteria."""

    def __init__(self, mode: Literal["strict", "modern", "garp", "relaxed"] = "strict"):
        """Initialize Graham analyzer.

        Args:
            mode: Analysis mode determining threshold strictness.
        """
        super().__init__()
        self.mode = mode
        self.thresholds = THRESHOLDS[mode]

    def analyze(self, symbol: str, quarter: str) -> dict[str, Any]:
        """Run Graham analysis for a stock.

        Args:
            symbol: Stock ticker symbol.
            quarter: Analysis quarter.

        Returns:
            Dictionary with Graham analysis results.
        """
        results = {
            "symbol": symbol,
            "analysis_quarter": quarter,
            "computed_at": datetime.now(timezone.utc),
            "mode": self.mode,
            "adequate_size": None,
            "current_ratio_pass": None,
            "debt_coverage_pass": None,
            "earnings_stability": None,
            "dividend_record": None,
            "earnings_growth_pass": None,
            "pe_ratio_pass": None,
            "pb_ratio_pass": None,
            "revenue": None,
            "current_ratio": None,
            "net_current_assets": None,
            "long_term_debt": None,
            "eps_5yr_growth": None,
            "pe_ratio": None,
            "pb_ratio": None,
            "pe_x_pb": None,
            "criteria_passed": 0,
            "data_quality": None,
            "missing_fields": [],
        }

        # Get financial data
        income_stmts = self.get_income_statements(symbol, "annual", limit=6)
        balance_sheets = self.get_balance_sheets(symbol, "annual", limit=2)
        profile = self.get_company_profile(symbol)
        dividends = self.get_dividends(symbol, years=6)

        if not income_stmts or not balance_sheets:
            results["data_quality"] = 0.0
            results["missing_fields"] = ["income_statements", "balance_sheets"]
            return results

        latest_income = income_stmts[0] if income_stmts else {}
        latest_balance = balance_sheets[0] if balance_sheets else {}

        # Criterion 1: Adequate Size (Revenue >= $500M)
        revenue = latest_income.get("revenue")
        results["revenue"] = revenue
        if revenue is not None:
            results["adequate_size"] = revenue >= self.thresholds["min_revenue"]
            if results["adequate_size"]:
                results["criteria_passed"] += 1

        # Criterion 2a: Current Ratio >= 2.0
        current_assets = latest_balance.get("current_assets")
        current_liabilities = latest_balance.get("current_liabilities")
        current_ratio = self.safe_divide(current_assets, current_liabilities)
        results["current_ratio"] = current_ratio
        if current_ratio is not None:
            results["current_ratio_pass"] = current_ratio >= self.thresholds["min_current_ratio"]
            if results["current_ratio_pass"]:
                results["criteria_passed"] += 1

        # Criterion 2b: Long-term Debt <= Net Current Assets
        long_term_debt = latest_balance.get("long_term_debt") or 0
        net_current_assets = (current_assets or 0) - (current_liabilities or 0)
        results["long_term_debt"] = long_term_debt
        results["net_current_assets"] = net_current_assets
        if current_assets is not None and current_liabilities is not None:
            results["debt_coverage_pass"] = long_term_debt <= net_current_assets
            if results["debt_coverage_pass"]:
                results["criteria_passed"] += 1

        # Criterion 3: Earnings Stability (Positive EPS for 5 years)
        eps_values = [stmt.get("eps") for stmt in income_stmts[:5]]
        eps_valid = [e for e in eps_values if e is not None]
        if len(eps_valid) >= self.thresholds["min_eps_years"]:
            results["earnings_stability"] = all(e > 0 for e in eps_valid)
            if results["earnings_stability"]:
                results["criteria_passed"] += 1

        # Criterion 4: Dividend Record (Dividends for 5 consecutive years)
        # ex_date can be datetime.date or string - convert to string first
        dividend_years = len(set(str(d.get("ex_date"))[:4] for d in dividends if d.get("amount") and d.get("ex_date")))
        results["dividend_record"] = dividend_years >= self.thresholds["min_dividend_years"]
        if results["dividend_record"]:
            results["criteria_passed"] += 1

        # Criterion 5: Earnings Growth (>= 33% over 5 years)
        if len(eps_valid) >= 2:
            eps_current = eps_valid[0]
            eps_5yr_ago = eps_valid[-1] if len(eps_valid) >= 5 else eps_valid[-1]
            if eps_5yr_ago and eps_5yr_ago > 0:
                eps_growth = (eps_current - eps_5yr_ago) / abs(eps_5yr_ago)
                results["eps_5yr_growth"] = eps_growth
                results["earnings_growth_pass"] = eps_growth >= self.thresholds["min_eps_growth"]
                if results["earnings_growth_pass"]:
                    results["criteria_passed"] += 1

        # Criterion 6: P/E Ratio <= 15
        # Calculate P/E from price and EPS (API doesn't provide it)
        price = to_float(profile.get("price")) if profile else None
        latest_eps = to_float(latest_income.get("eps"))
        pe_ratio = None
        if price is not None and latest_eps is not None and latest_eps > 0:
            pe_ratio = price / latest_eps
        results["pe_ratio"] = pe_ratio
        if pe_ratio is not None and pe_ratio > 0:
            results["pe_ratio_pass"] = pe_ratio <= self.thresholds["max_pe"]
            if results["pe_ratio_pass"]:
                results["criteria_passed"] += 1

        # Criterion 7: P/B Ratio <= 1.5 OR (P/E × P/B) <= 22.5
        # Calculate P/B from price and book value per share
        total_equity = to_float(latest_balance.get("total_equity") or latest_balance.get("total_stockholders_equity"))
        shares = to_float(latest_balance.get("common_shares_outstanding"))
        pb_ratio = None
        if price is not None and total_equity is not None and shares is not None and shares > 0:
            book_value_per_share = total_equity / shares
            if book_value_per_share > 0:
                pb_ratio = price / book_value_per_share
        results["pb_ratio"] = pb_ratio
        if pe_ratio is not None and pb_ratio is not None and pe_ratio > 0 and pb_ratio > 0:
            pe_x_pb = pe_ratio * pb_ratio
            results["pe_x_pb"] = pe_x_pb
            results["pb_ratio_pass"] = (
                pb_ratio <= self.thresholds["max_pb"]
                or pe_x_pb <= self.thresholds["max_pe_x_pb"]
            )
            if results["pb_ratio_pass"]:
                results["criteria_passed"] += 1

        # Calculate data quality
        required_fields = [
            "revenue",
            "current_ratio",
            "net_current_assets",
            "pe_ratio",
            "pb_ratio",
        ]
        results["data_quality"], results["missing_fields"] = self.calculate_data_quality(
            required_fields, results
        )

        return results

    def save_results(self, results: dict[str, Any]) -> None:
        """Save Graham analysis results to database."""
        conn = self._get_conn()
        try:
            conn.execute(
                """
                INSERT INTO graham_results (
                    symbol, analysis_quarter, computed_at, mode,
                    adequate_size, current_ratio_pass, debt_coverage_pass,
                    earnings_stability, dividend_record, earnings_growth_pass,
                    pe_ratio_pass, pb_ratio_pass, revenue, current_ratio,
                    net_current_assets, long_term_debt, eps_5yr_growth,
                    pe_ratio, pb_ratio, pe_x_pb, criteria_passed,
                    data_quality, missing_fields
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT (symbol, analysis_quarter, mode) DO UPDATE SET
                    computed_at = EXCLUDED.computed_at,
                    adequate_size = EXCLUDED.adequate_size,
                    current_ratio_pass = EXCLUDED.current_ratio_pass,
                    debt_coverage_pass = EXCLUDED.debt_coverage_pass,
                    earnings_stability = EXCLUDED.earnings_stability,
                    dividend_record = EXCLUDED.dividend_record,
                    earnings_growth_pass = EXCLUDED.earnings_growth_pass,
                    pe_ratio_pass = EXCLUDED.pe_ratio_pass,
                    pb_ratio_pass = EXCLUDED.pb_ratio_pass,
                    revenue = EXCLUDED.revenue,
                    current_ratio = EXCLUDED.current_ratio,
                    net_current_assets = EXCLUDED.net_current_assets,
                    long_term_debt = EXCLUDED.long_term_debt,
                    eps_5yr_growth = EXCLUDED.eps_5yr_growth,
                    pe_ratio = EXCLUDED.pe_ratio,
                    pb_ratio = EXCLUDED.pb_ratio,
                    pe_x_pb = EXCLUDED.pe_x_pb,
                    criteria_passed = EXCLUDED.criteria_passed,
                    data_quality = EXCLUDED.data_quality,
                    missing_fields = EXCLUDED.missing_fields
                """,
                (
                    results["symbol"],
                    results["analysis_quarter"],
                    results["computed_at"],
                    results["mode"],
                    results["adequate_size"],
                    results["current_ratio_pass"],
                    results["debt_coverage_pass"],
                    results["earnings_stability"],
                    results["dividend_record"],
                    results["earnings_growth_pass"],
                    results["pe_ratio_pass"],
                    results["pb_ratio_pass"],
                    results["revenue"],
                    results["current_ratio"],
                    results["net_current_assets"],
                    results["long_term_debt"],
                    results["eps_5yr_growth"],
                    results["pe_ratio"],
                    results["pb_ratio"],
                    results["pe_x_pb"],
                    results["criteria_passed"],
                    results["data_quality"],
                    json.dumps(results["missing_fields"]),
                ),
            )
        finally:
            if self._should_close_conn(conn):
                conn.close()
