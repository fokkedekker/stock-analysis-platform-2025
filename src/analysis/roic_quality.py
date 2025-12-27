"""ROIC/Quality screen analysis."""

import json
import logging
import statistics
from datetime import datetime, timezone
from typing import Any, TYPE_CHECKING

from src.analysis.base import BaseAnalyzer, to_float, quarter_to_end_date

if TYPE_CHECKING:
    from src.analysis.bulk_loader import BulkDataLoader

logger = logging.getLogger(__name__)


class ROICQualityAnalyzer(BaseAnalyzer):
    """Analyzer for ROIC/Quality screen.

    Quality criteria:
    - ROIC >= 12%
    - Free Cash Flow positive for >= 5 years
    - Debt-to-Equity < 1.0

    Enhanced quality metrics (v2):
    - ROIC Stability: Std dev of ROIC over 5 years
    - Gross Margin Stability: Std dev of gross margin over 5 years
    - FCF / Net Income: Earnings quality check
    - Reinvestment Rate: (CapEx + R&D) / OCF
    - FCF Yield: FCF / Enterprise Value
    - EV / EBIT: Operating valuation

    Where:
    - NOPAT = EBIT × (1 - Effective Tax Rate)
    - Invested Capital = Total Debt + Equity - Cash
    - ROIC = NOPAT / Invested Capital
    - Free Cash Flow = Operating Cash Flow - CapEx
    """

    # Original thresholds
    MIN_ROIC = 0.12  # 12%
    MAX_DEBT_TO_EQUITY = 1.0
    MIN_FCF_POSITIVE_YEARS = 5

    # Stability thresholds (std dev)
    ROIC_STABLE_THRESHOLD = 0.03      # < 3% std dev = stable
    ROIC_VOLATILE_THRESHOLD = 0.08    # > 8% std dev = volatile
    GM_STABLE_THRESHOLD = 0.02        # < 2% std dev = stable
    GM_VOLATILE_THRESHOLD = 0.05      # > 5% std dev = volatile

    # Quality thresholds
    FCF_NI_STRONG_THRESHOLD = 1.0     # FCF/NI >= 1.0 = strong
    FCF_NI_POOR_THRESHOLD = 0.5       # FCF/NI < 0.5 = poor
    REINVEST_LOW_THRESHOLD = 0.10     # < 10% = low reinvestment
    REINVEST_HIGH_THRESHOLD = 0.30    # > 30% = aggressive

    # Valuation thresholds
    FCF_YIELD_ATTRACTIVE = 0.10       # >= 10% = attractive
    FCF_YIELD_FAIR = 0.05             # 5-10% = fair
    EV_EBIT_CHEAP = 10                # <= 10 = cheap
    EV_EBIT_EXPENSIVE = 15            # > 15 = expensive

    def __init__(self, bulk_loader: "BulkDataLoader | None" = None):
        """Initialize ROIC/Quality analyzer.

        Args:
            bulk_loader: Optional BulkDataLoader for high-performance batch analysis.
        """
        super().__init__(bulk_loader=bulk_loader)

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
            # NEW: Stability metrics
            "roic_std_dev": None,
            "roic_stability_tag": None,
            "gross_margin_std_dev": None,
            "gross_margin_stability_tag": None,
            # NEW: Quality metrics
            "fcf_to_net_income": None,
            "earnings_quality_tag": None,
            "reinvestment_rate": None,
            "reinvestment_tag": None,
            # NEW: Valuation metrics
            "fcf_yield": None,
            "ev_to_ebit": None,
            "valuation_tag": None,
            # NEW: Aggregate tags
            "quality_tags": [],
        }

        # Get financial data (filtered by quarter end date for point-in-time analysis)
        # Fetch 5 years for stability calculations
        as_of = quarter_to_end_date(quarter)
        income_stmts = self.get_income_statements(symbol, "annual", limit=5, as_of_date=as_of)
        balance_sheets = self.get_balance_sheets(symbol, "annual", limit=5, as_of_date=as_of)
        cash_flows = self.get_cash_flow_statements(symbol, "annual", limit=5, as_of_date=as_of)
        key_metrics = self.get_key_metrics(symbol, "annual", limit=5, as_of_date=as_of)

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

        # ============================================================
        # NEW: Calculate enhanced quality and valuation metrics
        # ============================================================

        quality_tags = []

        # 1. ROIC Stability (std dev of ROIC over 5 years)
        # Try key_metrics.roic first, then raw_json.returnOnInvestedCapital
        if key_metrics:
            roic_values = []
            for km in key_metrics:
                roic = to_float(km.get("roic"))
                if roic is None:
                    # Try extracting from raw_json (may be string or dict)
                    raw = km.get("raw_json")
                    if raw:
                        if isinstance(raw, str):
                            try:
                                raw = json.loads(raw)
                            except json.JSONDecodeError:
                                raw = None
                        if raw and isinstance(raw, dict):
                            roic = to_float(raw.get("returnOnInvestedCapital"))
                if roic is not None:
                    roic_values.append(roic)
            if len(roic_values) >= 3:
                results["roic_std_dev"] = statistics.stdev(roic_values)
                if results["roic_std_dev"] < self.ROIC_STABLE_THRESHOLD:
                    results["roic_stability_tag"] = "stable"
                elif results["roic_std_dev"] > self.ROIC_VOLATILE_THRESHOLD:
                    results["roic_stability_tag"] = "volatile"
                    quality_tags.append("Volatile Returns")
                else:
                    results["roic_stability_tag"] = "moderate"

        # 2. Gross Margin Stability (std dev over 5 years from income statements)
        # Calculate gross margin from gross_profit / revenue since gross_profit_ratio may be NULL
        if income_stmts:
            gm_values = []
            for inc in income_stmts:
                gp = to_float(inc.get("gross_profit"))
                rev = to_float(inc.get("revenue"))
                if gp is not None and rev is not None and rev > 0:
                    gm_values.append(gp / rev)
            if len(gm_values) >= 3:
                results["gross_margin_std_dev"] = statistics.stdev(gm_values)
                if results["gross_margin_std_dev"] < self.GM_STABLE_THRESHOLD:
                    results["gross_margin_stability_tag"] = "stable"
                elif results["gross_margin_std_dev"] > self.GM_VOLATILE_THRESHOLD:
                    results["gross_margin_stability_tag"] = "volatile"
                    quality_tags.append("Weak Moat Signal")
                else:
                    results["gross_margin_stability_tag"] = "moderate"

        # Check for "Durable Compounder" tag
        if (results["roic_stability_tag"] == "stable" and
            results["gross_margin_stability_tag"] == "stable"):
            quality_tags.append("Durable Compounder")

        # 3. FCF / Net Income (earnings quality)
        if income_stmts and cash_flows:
            net_income = to_float(income_stmts[0].get("net_income"))
            fcf = to_float(cash_flows[0].get("free_cash_flow"))
            if net_income is not None and net_income > 0 and fcf is not None:
                results["fcf_to_net_income"] = fcf / net_income
                if results["fcf_to_net_income"] >= self.FCF_NI_STRONG_THRESHOLD:
                    results["earnings_quality_tag"] = "strong"
                elif results["fcf_to_net_income"] < self.FCF_NI_POOR_THRESHOLD:
                    results["earnings_quality_tag"] = "poor"
                    quality_tags.append("Earnings Quality Concern")
                else:
                    results["earnings_quality_tag"] = "acceptable"

        # 4. Reinvestment Rate = (CapEx + R&D) / OCF
        if income_stmts and cash_flows:
            capex = abs(to_float(cash_flows[0].get("capital_expenditure")) or 0)
            rd = to_float(income_stmts[0].get("research_and_development")) or 0
            ocf = to_float(cash_flows[0].get("operating_cash_flow"))
            if ocf is not None and ocf > 0:
                results["reinvestment_rate"] = (capex + rd) / ocf
                if results["reinvestment_rate"] < self.REINVEST_LOW_THRESHOLD:
                    results["reinvestment_tag"] = "low"
                elif results["reinvestment_rate"] > self.REINVEST_HIGH_THRESHOLD:
                    results["reinvestment_tag"] = "aggressive"
                    quality_tags.append("Heavy Reinvestor")
                else:
                    results["reinvestment_tag"] = "moderate"

        # 5. FCF Yield = FCF / Enterprise Value
        if key_metrics and cash_flows:
            fcf = to_float(cash_flows[0].get("free_cash_flow"))
            ev = to_float(key_metrics[0].get("enterprise_value"))
            if fcf is not None and ev is not None and ev > 0:
                results["fcf_yield"] = fcf / ev

        # 6. EV / EBIT
        if key_metrics and income_stmts:
            ev = to_float(key_metrics[0].get("enterprise_value"))
            ebit = to_float(income_stmts[0].get("ebit"))
            if ev is not None and ebit is not None and ebit > 0:
                results["ev_to_ebit"] = ev / ebit

        # Determine valuation tag based on FCF Yield and EV/EBIT
        fcf_yield = results["fcf_yield"]
        ev_ebit = results["ev_to_ebit"]
        if fcf_yield is not None and ev_ebit is not None:
            if fcf_yield >= self.FCF_YIELD_ATTRACTIVE or ev_ebit <= self.EV_EBIT_CHEAP:
                results["valuation_tag"] = "deep_value"
                quality_tags.append("Deep Value")
            elif fcf_yield < self.FCF_YIELD_FAIR or ev_ebit > self.EV_EBIT_EXPENSIVE:
                results["valuation_tag"] = "premium"
                quality_tags.append("Premium Priced")
            else:
                results["valuation_tag"] = "fair"

        # Check for "Cash Machine" tag
        if (results["earnings_quality_tag"] == "strong" and
            results["fcf_yield"] is not None and
            results["fcf_yield"] >= 0.08):  # 8% FCF yield
            quality_tags.append("Cash Machine")

        results["quality_tags"] = quality_tags

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
                    criteria_passed, data_quality, missing_fields,
                    roic_std_dev, roic_stability_tag,
                    gross_margin_std_dev, gross_margin_stability_tag,
                    fcf_to_net_income, earnings_quality_tag,
                    reinvestment_rate, reinvestment_tag,
                    fcf_yield, ev_to_ebit, valuation_tag,
                    quality_tags
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT (symbol, analysis_quarter) DO UPDATE SET
                    computed_at = EXCLUDED.computed_at,
                    ebit = EXCLUDED.ebit,
                    effective_tax_rate = EXCLUDED.effective_tax_rate,
                    nopat = EXCLUDED.nopat,
                    total_debt = EXCLUDED.total_debt,
                    total_equity = EXCLUDED.total_equity,
                    cash = EXCLUDED.cash,
                    invested_capital = EXCLUDED.invested_capital,
                    roic = EXCLUDED.roic,
                    operating_cash_flow = EXCLUDED.operating_cash_flow,
                    capital_expenditure = EXCLUDED.capital_expenditure,
                    free_cash_flow = EXCLUDED.free_cash_flow,
                    debt_to_equity = EXCLUDED.debt_to_equity,
                    roic_pass = EXCLUDED.roic_pass,
                    fcf_positive_5yr = EXCLUDED.fcf_positive_5yr,
                    debt_to_equity_pass = EXCLUDED.debt_to_equity_pass,
                    criteria_passed = EXCLUDED.criteria_passed,
                    data_quality = EXCLUDED.data_quality,
                    missing_fields = EXCLUDED.missing_fields,
                    roic_std_dev = EXCLUDED.roic_std_dev,
                    roic_stability_tag = EXCLUDED.roic_stability_tag,
                    gross_margin_std_dev = EXCLUDED.gross_margin_std_dev,
                    gross_margin_stability_tag = EXCLUDED.gross_margin_stability_tag,
                    fcf_to_net_income = EXCLUDED.fcf_to_net_income,
                    earnings_quality_tag = EXCLUDED.earnings_quality_tag,
                    reinvestment_rate = EXCLUDED.reinvestment_rate,
                    reinvestment_tag = EXCLUDED.reinvestment_tag,
                    fcf_yield = EXCLUDED.fcf_yield,
                    ev_to_ebit = EXCLUDED.ev_to_ebit,
                    valuation_tag = EXCLUDED.valuation_tag,
                    quality_tags = EXCLUDED.quality_tags
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
                    results["roic_std_dev"],
                    results["roic_stability_tag"],
                    results["gross_margin_std_dev"],
                    results["gross_margin_stability_tag"],
                    results["fcf_to_net_income"],
                    results["earnings_quality_tag"],
                    results["reinvestment_rate"],
                    results["reinvestment_tag"],
                    results["fcf_yield"],
                    results["ev_to_ebit"],
                    results["valuation_tag"],
                    json.dumps(results["quality_tags"]),
                ),
            )
        finally:
            if self._should_close_conn(conn):
                conn.close()
