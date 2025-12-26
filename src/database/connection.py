"""DuckDB connection management with WAL mode for concurrent access."""

import logging
from contextlib import contextmanager
from pathlib import Path
from typing import Generator

import duckdb

from src.config import get_settings

logger = logging.getLogger(__name__)


class DatabaseManager:
    """Manages DuckDB connections with proper configuration."""

    def __init__(self, db_path: Path | str | None = None):
        """Initialize database manager.

        Args:
            db_path: Path to database file. Uses settings default if not provided.
        """
        settings = get_settings()
        self.db_path = Path(db_path) if db_path else settings.database_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._connection: duckdb.DuckDBPyConnection | None = None

    def connect(self) -> duckdb.DuckDBPyConnection:
        """Create a new connection to the database."""
        conn = duckdb.connect(str(self.db_path))
        conn.execute("PRAGMA enable_progress_bar=false")
        return conn

    @contextmanager
    def get_connection(self) -> Generator[duckdb.DuckDBPyConnection, None, None]:
        """Context manager for database connections.

        DuckDB operates in autocommit mode by default, so each execute()
        is automatically committed. We wrap in BEGIN/COMMIT for explicit
        transaction control when multiple operations need to be atomic.

        Yields:
            DuckDB connection with transaction management.
        """
        conn = self.connect()
        try:
            conn.execute("BEGIN TRANSACTION")
            yield conn
            conn.execute("COMMIT")
        except Exception:
            try:
                conn.execute("ROLLBACK")
            except Exception:
                # Ignore rollback errors (e.g., no active transaction)
                pass
            raise
        finally:
            conn.close()

    @contextmanager
    def transaction(self, conn: duckdb.DuckDBPyConnection) -> Generator[duckdb.DuckDBPyConnection, None, None]:
        """Context manager for transactions on an existing connection.

        Use this when you want to reuse a connection across multiple operations
        but still have transaction safety for each operation.

        Args:
            conn: Existing database connection.

        Yields:
            The same connection wrapped in a transaction.
        """
        try:
            conn.execute("BEGIN TRANSACTION")
            yield conn
            conn.execute("COMMIT")
        except Exception:
            try:
                conn.execute("ROLLBACK")
            except Exception:
                pass
            raise

    def execute(self, query: str, params: tuple | list | None = None) -> duckdb.DuckDBPyRelation:
        """Execute a query and return results.

        Args:
            query: SQL query to execute.
            params: Optional parameters for the query.

        Returns:
            Query results as a DuckDB relation.
        """
        with self.get_connection() as conn:
            if params:
                return conn.execute(query, params)
            return conn.execute(query)

    def execute_many(self, query: str, params_list: list[tuple]) -> None:
        """Execute a query with multiple parameter sets.

        Args:
            query: SQL query to execute.
            params_list: List of parameter tuples.
        """
        with self.get_connection() as conn:
            conn.executemany(query, params_list)


# Global database manager instance
_db_manager: DatabaseManager | None = None


def get_db_manager() -> DatabaseManager:
    """Get or create the global database manager."""
    global _db_manager
    if _db_manager is None:
        _db_manager = DatabaseManager()
    return _db_manager


def get_connection() -> Generator[duckdb.DuckDBPyConnection, None, None]:
    """Get a database connection from the global manager."""
    return get_db_manager().get_connection()
