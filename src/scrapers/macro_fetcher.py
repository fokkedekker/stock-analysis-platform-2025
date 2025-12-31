"""Macro economic data fetcher for regime analysis.

Only fetches treasury rates (CPI/GDP/unemployment have no historical data in FMP).
"""

import json
import logging
from datetime import datetime
from typing import Any

from src.database.connection import get_db_manager
from src.scrapers.fmp_client import FMPClient

logger = logging.getLogger(__name__)

# Regime classification threshold
RATE_CHANGE_THRESHOLD = 0.25  # 25bps = rising/falling


def _date_to_quarter(date_str: str) -> str:
    """Convert date string to quarter format (e.g., '2024Q1')."""
    dt = datetime.strptime(date_str[:10], "%Y-%m-%d")
    quarter = (dt.month - 1) // 3 + 1
    return f"{dt.year}Q{quarter}"


class MacroFetcher:
    """Fetches treasury rates and computes rate regime classifications."""

    def __init__(self, client: FMPClient):
        """Initialize with FMP client."""
        self.client = client
        self.db = get_db_manager()

    async def fetch_all_macro_data(
        self,
        from_date: str | None = None,
        to_date: str | None = None,
    ) -> dict[str, int]:
        """Fetch treasury rates from FMP.

        Args:
            from_date: Start date (YYYY-MM-DD). Defaults to 2010-01-01.
            to_date: End date (YYYY-MM-DD). Defaults to today.

        Returns:
            Dict with count of quarters saved.
        """
        from_date = from_date or "2010-01-01"
        to_date = to_date or datetime.now().strftime("%Y-%m-%d")

        logger.info(f"Fetching treasury rates from {from_date} to {to_date}")

        treasury_data = await self.client.get_treasury_rates(from_date, to_date)
        quarters_saved = 0
        if treasury_data:
            quarters_saved = self._save_treasury_rates(treasury_data)
            logger.info(f"Saved {quarters_saved} quarters of treasury rates")

        return {"treasury_rates": quarters_saved}

    def _save_treasury_rates(self, data: list[dict[str, Any]]) -> int:
        """Save treasury rates aggregated by quarter."""
        # Group by quarter
        quarterly_data: dict[str, list[dict]] = {}
        for record in data:
            date_str = record.get("date")
            if not date_str:
                continue
            quarter = _date_to_quarter(date_str)
            if quarter not in quarterly_data:
                quarterly_data[quarter] = []
            quarterly_data[quarter].append(record)

        quarters_saved = 0
        with self.db.get_connection() as conn:
            for quarter, records in quarterly_data.items():
                # Use end-of-quarter values (last record in quarter)
                records.sort(key=lambda x: x.get("date", ""), reverse=True)
                latest = records[0]

                # Map FMP field names to our schema
                treasury_1m = latest.get("month1")
                treasury_3m = latest.get("month3")
                treasury_6m = latest.get("month6")
                treasury_1y = latest.get("year1")
                treasury_2y = latest.get("year2")
                treasury_5y = latest.get("year5")
                treasury_10y = latest.get("year10")
                treasury_30y = latest.get("year30")

                # Calculate yield curve spread (10Y - 2Y)
                yield_spread = None
                if treasury_10y is not None and treasury_2y is not None:
                    yield_spread = treasury_10y - treasury_2y

                conn.execute(
                    """
                    INSERT OR REPLACE INTO macro_indicators
                    (quarter, indicator_date, treasury_1m, treasury_3m, treasury_6m,
                     treasury_1y, treasury_2y, treasury_5y, treasury_10y, treasury_30y,
                     yield_curve_spread, fetched_at, raw_json)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, now(), ?)
                    """,
                    [
                        quarter,
                        latest.get("date"),
                        treasury_1m,
                        treasury_3m,
                        treasury_6m,
                        treasury_1y,
                        treasury_2y,
                        treasury_5y,
                        treasury_10y,
                        treasury_30y,
                        yield_spread,
                        json.dumps(latest),
                    ],
                )
                quarters_saved += 1

        return quarters_saved

    def compute_regime_flags(self) -> int:
        """Compute rate regime classifications from treasury data.

        Returns:
            Number of quarters with regime flags computed.
        """
        logger.info("Computing regime flags...")

        with self.db.get_connection() as conn:
            result = conn.execute(
                """
                SELECT quarter, treasury_10y
                FROM macro_indicators
                ORDER BY quarter
                """
            ).fetchall()

            if not result:
                logger.warning("No macro indicators found")
                return 0

            quarters_data = [{"quarter": row[0], "treasury_10y": row[1]} for row in result]

            quarters_computed = 0
            for i, data in enumerate(quarters_data):
                quarter = data["quarter"]

                # Rate regime: compare to previous quarter
                rate_regime = "stable"
                rate_change = None
                if i > 0 and data["treasury_10y"] is not None:
                    prev_rate = quarters_data[i - 1]["treasury_10y"]
                    if prev_rate is not None:
                        rate_change = data["treasury_10y"] - prev_rate
                        if rate_change >= RATE_CHANGE_THRESHOLD:
                            rate_regime = "rising"
                        elif rate_change <= -RATE_CHANGE_THRESHOLD:
                            rate_regime = "falling"

                conn.execute(
                    """
                    INSERT OR REPLACE INTO regime_flags
                    (quarter, computed_at, rate_regime, rate_change_qoq)
                    VALUES (?, now(), ?, ?)
                    """,
                    [quarter, rate_regime, rate_change],
                )
                quarters_computed += 1

            logger.info(f"Computed regime flags for {quarters_computed} quarters")
            return quarters_computed
