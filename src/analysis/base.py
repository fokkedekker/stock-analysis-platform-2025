"""Base analyzer class for all valuation systems."""

import json
import logging
from abc import ABC, abstractmethod
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Any, Generator

from src.database.connection import get_db_manager


def to_float(val):
    """Convert value to float, handling Decimal and None.

    DuckDB returns Decimal for numeric columns, which can't be multiplied
    with Python floats directly. This helper safely converts to float.
    """
    if val is None:
        return None
    if isinstance(val, Decimal):
        return float(val)
    try:
        return float(val)
    except (TypeError, ValueError):
        return None

logger = logging.getLogger(__name__)


class BaseAnalyzer(ABC):
    """Abstract base class for all fundamental analysis systems."""

    def __init__(self):
        """Initialize base analyzer."""
        self.db = get_db_manager()
        self._conn = None  # Reusable connection for batch operations

    @contextmanager
    def connection_scope(self) -> Generator[None, None, None]:
        """Context manager for reusing a single connection across multiple operations.

        Usage:
            with analyzer.connection_scope():
                for symbol in symbols:
                    analyzer.analyze_and_save(symbol, quarter)
        """
        self._conn = self.db.connect()
        try:
            yield
        finally:
            self._conn.close()
            self._conn = None

    def _get_conn(self):
        """Get the current connection - reusable or new."""
        if self._conn is not None:
            return self._conn
        return self.db.connect()

    def _should_close_conn(self, conn) -> bool:
        """Check if we should close this connection (only if it's not our reusable one)."""
        return self._conn is None

    @abstractmethod
    def analyze(self, symbol: str, quarter: str) -> dict[str, Any]:
        """Run analysis for a single stock.

        Args:
            symbol: Stock ticker symbol.
            quarter: Analysis quarter (e.g., '2024Q4').

        Returns:
            Dictionary with analysis results.
        """
        pass

    @abstractmethod
    def save_results(self, results: dict[str, Any]) -> None:
        """Save analysis results to database.

        Args:
            results: Analysis results dictionary.
        """
        pass

    def get_income_statements(
        self,
        symbol: str,
        period: str = "annual",
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """Get income statements for a symbol.

        Args:
            symbol: Stock ticker symbol.
            period: 'annual' or 'quarter'.
            limit: Maximum number of records.

        Returns:
            List of income statement records ordered by date descending.
        """
        conn = self._get_conn()
        try:
            result = conn.execute(
                """
                SELECT * FROM income_statements
                WHERE symbol = ? AND period = ?
                ORDER BY fiscal_date DESC
                LIMIT ?
                """,
                (symbol, period, limit),
            ).fetchall()

            columns = [desc[0] for desc in conn.description]
            return [dict(zip(columns, row)) for row in result]
        finally:
            if self._should_close_conn(conn):
                conn.close()

    def get_balance_sheets(
        self,
        symbol: str,
        period: str = "annual",
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """Get balance sheets for a symbol."""
        conn = self._get_conn()
        try:
            result = conn.execute(
                """
                SELECT * FROM balance_sheets
                WHERE symbol = ? AND period = ?
                ORDER BY fiscal_date DESC
                LIMIT ?
                """,
                (symbol, period, limit),
            ).fetchall()

            columns = [desc[0] for desc in conn.description]
            return [dict(zip(columns, row)) for row in result]
        finally:
            if self._should_close_conn(conn):
                conn.close()

    def get_cash_flow_statements(
        self,
        symbol: str,
        period: str = "annual",
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """Get cash flow statements for a symbol."""
        conn = self._get_conn()
        try:
            result = conn.execute(
                """
                SELECT * FROM cash_flow_statements
                WHERE symbol = ? AND period = ?
                ORDER BY fiscal_date DESC
                LIMIT ?
                """,
                (symbol, period, limit),
            ).fetchall()

            columns = [desc[0] for desc in conn.description]
            return [dict(zip(columns, row)) for row in result]
        finally:
            if self._should_close_conn(conn):
                conn.close()

    def get_key_metrics(
        self,
        symbol: str,
        period: str = "annual",
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """Get key metrics for a symbol."""
        conn = self._get_conn()
        try:
            result = conn.execute(
                """
                SELECT * FROM key_metrics
                WHERE symbol = ? AND period = ?
                ORDER BY fiscal_date DESC
                LIMIT ?
                """,
                (symbol, period, limit),
            ).fetchall()

            columns = [desc[0] for desc in conn.description]
            return [dict(zip(columns, row)) for row in result]
        finally:
            if self._should_close_conn(conn):
                conn.close()

    def get_company_profile(self, symbol: str) -> dict[str, Any] | None:
        """Get most recent company profile."""
        conn = self._get_conn()
        try:
            result = conn.execute(
                """
                SELECT * FROM company_profiles
                WHERE symbol = ?
                ORDER BY fiscal_quarter DESC
                LIMIT 1
                """,
                (symbol,),
            ).fetchone()

            if result:
                columns = [desc[0] for desc in conn.description]
                return dict(zip(columns, result))
            return None
        finally:
            if self._should_close_conn(conn):
                conn.close()

    def get_dividends(
        self,
        symbol: str,
        years: int = 5,
    ) -> list[dict[str, Any]]:
        """Get dividend history for a symbol."""
        conn = self._get_conn()
        try:
            # Calculate cutoff date in Python since DuckDB doesn't support parameterized intervals
            cutoff_date = (datetime.now(timezone.utc) - timedelta(days=years * 365)).strftime('%Y-%m-%d')
            result = conn.execute(
                """
                SELECT * FROM dividends
                WHERE symbol = ?
                AND ex_date >= ?
                ORDER BY ex_date DESC
                """,
                (symbol, cutoff_date),
            ).fetchall()

            columns = [desc[0] for desc in conn.description]
            return [dict(zip(columns, row)) for row in result]
        finally:
            if self._should_close_conn(conn):
                conn.close()

    def calculate_data_quality(
        self,
        required_fields: list[str],
        data: dict[str, Any],
    ) -> tuple[float, list[str]]:
        """Calculate data quality score based on missing fields.

        Args:
            required_fields: List of field names required for analysis.
            data: Dictionary of available data.

        Returns:
            Tuple of (quality_score, missing_fields).
        """
        missing = []
        for field in required_fields:
            value = data.get(field)
            if value is None:
                missing.append(field)

        if not required_fields:
            return 1.0, []

        quality = 1.0 - (len(missing) / len(required_fields))
        return quality, missing

    @staticmethod
    def safe_divide(numerator: float | None, denominator: float | None) -> float | None:
        """Safely divide two numbers, returning None if invalid."""
        num = to_float(numerator)
        denom = to_float(denominator)
        if num is None or denom is None or denom == 0:
            return None
        return num / denom

    @staticmethod
    def get_current_quarter() -> str:
        """Get current fiscal quarter string."""
        now = datetime.now(timezone.utc)
        quarter = (now.month - 1) // 3 + 1
        return f"{now.year}Q{quarter}"

    def analyze_and_save(self, symbol: str, quarter: str | None = None) -> dict[str, Any]:
        """Run analysis and save results.

        Args:
            symbol: Stock ticker symbol.
            quarter: Analysis quarter. Uses current if not provided.

        Returns:
            Analysis results.
        """
        quarter = quarter or self.get_current_quarter()
        results = self.analyze(symbol, quarter)
        self.save_results(results)
        return results
