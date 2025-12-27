"""Bulk data loader for high-performance batch analysis.

Loads all financial data in a few queries instead of per-symbol queries,
providing ~1000x speedup for batch analysis runs.
"""

import logging
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from typing import Any

from src.database.connection import get_db_manager

logger = logging.getLogger(__name__)


class BulkDataLoader:
    """Bulk loader that pre-loads all financial data for fast access.

    Usage:
        loader = BulkDataLoader(period="annual", as_of_date="2024-09-30")
        income_stmts = loader.get_income_statements("AAPL", limit=5)
    """

    def __init__(
        self,
        period: str = "annual",
        as_of_date: str | None = None,
        include_dividends: bool = True,
        include_profiles: bool = True,
    ):
        """Initialize bulk loader.

        Args:
            period: 'annual' or 'quarter' for financial statements.
            as_of_date: Only include data with fiscal_date <= this date.
            include_dividends: Whether to load dividend data.
            include_profiles: Whether to load company profiles.
        """
        self.period = period
        self.as_of_date = as_of_date
        self.db = get_db_manager()

        # Data storage: symbol -> list of records (ordered by fiscal_date DESC)
        self.income_by_symbol: dict[str, list[dict[str, Any]]] = defaultdict(list)
        self.balance_by_symbol: dict[str, list[dict[str, Any]]] = defaultdict(list)
        self.cashflow_by_symbol: dict[str, list[dict[str, Any]]] = defaultdict(list)
        self.metrics_by_symbol: dict[str, list[dict[str, Any]]] = defaultdict(list)
        self.dividends_by_symbol: dict[str, list[dict[str, Any]]] = defaultdict(list)
        self.profiles_by_symbol_quarter: dict[tuple[str, str], dict[str, Any]] = {}

        self._load_all(include_dividends, include_profiles)

    def _load_all(self, include_dividends: bool, include_profiles: bool) -> None:
        """Load all financial data in bulk queries."""
        with self.db.get_connection() as conn:
            # Build WHERE clause for as_of_date filtering
            date_filter = ""
            date_params: tuple = ()
            if self.as_of_date:
                date_filter = "AND fiscal_date <= ?"
                date_params = (self.as_of_date,)

            # Load income statements
            logger.info("Loading income statements...")
            result = conn.execute(
                f"""
                SELECT * FROM income_statements
                WHERE period = ?
                {date_filter}
                ORDER BY symbol, fiscal_date DESC
                """,
                (self.period,) + date_params,
            ).fetchall()
            columns = [desc[0] for desc in conn.description]
            for row in result:
                record = dict(zip(columns, row))
                self.income_by_symbol[record["symbol"]].append(record)
            logger.info(f"Loaded {len(result):,} income statements for {len(self.income_by_symbol):,} symbols")

            # Load balance sheets
            logger.info("Loading balance sheets...")
            result = conn.execute(
                f"""
                SELECT * FROM balance_sheets
                WHERE period = ?
                {date_filter}
                ORDER BY symbol, fiscal_date DESC
                """,
                (self.period,) + date_params,
            ).fetchall()
            columns = [desc[0] for desc in conn.description]
            for row in result:
                record = dict(zip(columns, row))
                self.balance_by_symbol[record["symbol"]].append(record)
            logger.info(f"Loaded {len(result):,} balance sheets")

            # Load cash flow statements
            logger.info("Loading cash flow statements...")
            result = conn.execute(
                f"""
                SELECT * FROM cash_flow_statements
                WHERE period = ?
                {date_filter}
                ORDER BY symbol, fiscal_date DESC
                """,
                (self.period,) + date_params,
            ).fetchall()
            columns = [desc[0] for desc in conn.description]
            for row in result:
                record = dict(zip(columns, row))
                self.cashflow_by_symbol[record["symbol"]].append(record)
            logger.info(f"Loaded {len(result):,} cash flow statements")

            # Load key metrics
            logger.info("Loading key metrics...")
            result = conn.execute(
                f"""
                SELECT * FROM key_metrics
                WHERE period = ?
                {date_filter}
                ORDER BY symbol, fiscal_date DESC
                """,
                (self.period,) + date_params,
            ).fetchall()
            columns = [desc[0] for desc in conn.description]
            for row in result:
                record = dict(zip(columns, row))
                self.metrics_by_symbol[record["symbol"]].append(record)
            logger.info(f"Loaded {len(result):,} key metrics")

            # Load dividends (if requested)
            if include_dividends:
                logger.info("Loading dividends...")
                if self.as_of_date:
                    # Calculate 5-year lookback from as_of_date
                    as_of_dt = datetime.strptime(self.as_of_date, "%Y-%m-%d")
                    cutoff = (as_of_dt - timedelta(days=5 * 365)).strftime("%Y-%m-%d")
                    result = conn.execute(
                        """
                        SELECT * FROM dividends
                        WHERE ex_date >= ? AND ex_date <= ?
                        ORDER BY symbol, ex_date DESC
                        """,
                        (cutoff, self.as_of_date),
                    ).fetchall()
                else:
                    cutoff = (datetime.now(timezone.utc) - timedelta(days=5 * 365)).strftime("%Y-%m-%d")
                    result = conn.execute(
                        """
                        SELECT * FROM dividends
                        WHERE ex_date >= ?
                        ORDER BY symbol, ex_date DESC
                        """,
                        (cutoff,),
                    ).fetchall()
                columns = [desc[0] for desc in conn.description]
                for row in result:
                    record = dict(zip(columns, row))
                    self.dividends_by_symbol[record["symbol"]].append(record)
                logger.info(f"Loaded {len(result):,} dividends")

            # Load company profiles (if requested)
            if include_profiles:
                logger.info("Loading company profiles...")
                result = conn.execute(
                    "SELECT * FROM company_profiles ORDER BY symbol, fiscal_quarter DESC"
                ).fetchall()
                columns = [desc[0] for desc in conn.description]
                for row in result:
                    record = dict(zip(columns, row))
                    key = (record["symbol"], record["fiscal_quarter"])
                    self.profiles_by_symbol_quarter[key] = record
                logger.info(f"Loaded {len(result):,} company profiles")

    def get_income_statements(self, symbol: str, limit: int = 10) -> list[dict[str, Any]]:
        """Get income statements for a symbol.

        Args:
            symbol: Stock ticker symbol.
            limit: Maximum number of records.

        Returns:
            List of income statement records ordered by date descending.
        """
        return self.income_by_symbol.get(symbol, [])[:limit]

    def get_balance_sheets(self, symbol: str, limit: int = 10) -> list[dict[str, Any]]:
        """Get balance sheets for a symbol.

        Args:
            symbol: Stock ticker symbol.
            limit: Maximum number of records.

        Returns:
            List of balance sheet records ordered by date descending.
        """
        return self.balance_by_symbol.get(symbol, [])[:limit]

    def get_cash_flow_statements(self, symbol: str, limit: int = 10) -> list[dict[str, Any]]:
        """Get cash flow statements for a symbol.

        Args:
            symbol: Stock ticker symbol.
            limit: Maximum number of records.

        Returns:
            List of cash flow statement records ordered by date descending.
        """
        return self.cashflow_by_symbol.get(symbol, [])[:limit]

    def get_key_metrics(self, symbol: str, limit: int = 10) -> list[dict[str, Any]]:
        """Get key metrics for a symbol.

        Args:
            symbol: Stock ticker symbol.
            limit: Maximum number of records.

        Returns:
            List of key metrics records ordered by date descending.
        """
        return self.metrics_by_symbol.get(symbol, [])[:limit]

    def get_dividends(self, symbol: str) -> list[dict[str, Any]]:
        """Get dividends for a symbol.

        Args:
            symbol: Stock ticker symbol.

        Returns:
            List of dividend records ordered by ex_date descending.
        """
        return self.dividends_by_symbol.get(symbol, [])

    def get_company_profile(self, symbol: str, quarter: str | None = None) -> dict[str, Any] | None:
        """Get company profile for a symbol.

        Args:
            symbol: Stock ticker symbol.
            quarter: Specific quarter (e.g., '2024Q1'). If None, returns most recent.

        Returns:
            Company profile dict or None.
        """
        if quarter:
            return self.profiles_by_symbol_quarter.get((symbol, quarter))

        # Return most recent profile for this symbol
        for key, profile in self.profiles_by_symbol_quarter.items():
            if key[0] == symbol:
                return profile
        return None

    def get_symbols(self) -> list[str]:
        """Get list of all symbols with data.

        Returns:
            List of symbol strings.
        """
        symbols = set()
        symbols.update(self.income_by_symbol.keys())
        symbols.update(self.balance_by_symbol.keys())
        return sorted(symbols)
