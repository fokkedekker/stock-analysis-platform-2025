"""Net-Net deep value analysis."""

import json
import logging
from datetime import datetime, timezone
from typing import Any

from src.analysis.base import BaseAnalyzer, to_float

logger = logging.getLogger(__name__)


class NetNetAnalyzer(BaseAnalyzer):
    """Analyzer for Net-Net deep value strategy.

    Net Current Asset Value (NCAV) = Current Assets - Total Liabilities

    Criteria:
    - Market Cap < NCAV (basic net-net)
    - Market Cap < 67% of NCAV (deep value / preferred)
    """

    # Threshold for deep value
    DEEP_VALUE_DISCOUNT = 0.67  # Trading below 67% of NCAV

    def analyze(self, symbol: str, quarter: str) -> dict[str, Any]:
        """Run Net-Net analysis for a stock.

        Args:
            symbol: Stock ticker symbol.
            quarter: Analysis quarter.

        Returns:
            Dictionary with Net-Net analysis results.
        """
        results = {
            "symbol": symbol,
            "analysis_quarter": quarter,
            "computed_at": datetime.now(timezone.utc),
            "current_assets": None,
            "total_liabilities": None,
            "ncav": None,
            "market_cap": None,
            "ncav_per_share": None,
            "price": None,
            "discount_to_ncav": None,
            "trading_below_ncav": None,
            "deep_value": None,
            "data_quality": None,
            "missing_fields": [],
        }

        # Get financial data
        balance_sheets = self.get_balance_sheets(symbol, "annual", limit=1)
        profile = self.get_company_profile(symbol)

        if not balance_sheets or not profile:
            results["data_quality"] = 0.0
            results["missing_fields"] = ["balance_sheets", "profile"]
            return results

        latest_balance = balance_sheets[0]

        # Get values (convert from Decimal if needed)
        current_assets = to_float(latest_balance.get("current_assets"))
        total_liabilities = to_float(latest_balance.get("total_liabilities"))
        market_cap = to_float(profile.get("market_cap"))
        price = to_float(profile.get("price"))
        shares_outstanding = to_float(profile.get("shares_outstanding"))

        results["current_assets"] = current_assets
        results["total_liabilities"] = total_liabilities
        results["market_cap"] = market_cap
        results["price"] = price

        # Calculate NCAV
        if current_assets is not None and total_liabilities is not None:
            ncav = current_assets - total_liabilities
            results["ncav"] = ncav

            # Calculate NCAV per share
            if shares_outstanding and shares_outstanding > 0:
                results["ncav_per_share"] = ncav / shares_outstanding

            # Calculate discount to NCAV
            if market_cap is not None and ncav != 0:
                if ncav > 0:
                    results["discount_to_ncav"] = market_cap / ncav
                    results["trading_below_ncav"] = market_cap < ncav
                    results["deep_value"] = market_cap < (ncav * self.DEEP_VALUE_DISCOUNT)
                else:
                    # Negative NCAV - not a net-net
                    results["trading_below_ncav"] = False
                    results["deep_value"] = False

        # Calculate data quality
        required_fields = ["ncav", "market_cap", "discount_to_ncav"]
        results["data_quality"], results["missing_fields"] = self.calculate_data_quality(
            required_fields, results
        )

        return results

    def save_results(self, results: dict[str, Any]) -> None:
        """Save Net-Net analysis results to database."""
        conn = self._get_conn()
        try:
            conn.execute(
                """
                INSERT INTO net_net_results (
                    symbol, analysis_quarter, computed_at,
                    current_assets, total_liabilities, ncav,
                    market_cap, ncav_per_share, price, discount_to_ncav,
                    trading_below_ncav, deep_value,
                    data_quality, missing_fields
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT (symbol, analysis_quarter) DO UPDATE SET
                    computed_at = EXCLUDED.computed_at,
                    ncav = EXCLUDED.ncav,
                    market_cap = EXCLUDED.market_cap,
                    discount_to_ncav = EXCLUDED.discount_to_ncav,
                    trading_below_ncav = EXCLUDED.trading_below_ncav,
                    deep_value = EXCLUDED.deep_value,
                    data_quality = EXCLUDED.data_quality,
                    missing_fields = EXCLUDED.missing_fields
                """,
                (
                    results["symbol"],
                    results["analysis_quarter"],
                    results["computed_at"],
                    results["current_assets"],
                    results["total_liabilities"],
                    results["ncav"],
                    results["market_cap"],
                    results["ncav_per_share"],
                    results["price"],
                    results["discount_to_ncav"],
                    results["trading_below_ncav"],
                    results["deep_value"],
                    results["data_quality"],
                    json.dumps(results["missing_fields"]),
                ),
            )
        finally:
            if self._should_close_conn(conn):
                conn.close()
