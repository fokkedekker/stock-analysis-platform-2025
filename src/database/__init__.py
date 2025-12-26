"""Database module for DuckDB connection and schema management."""

from src.database.connection import get_connection, DatabaseManager
from src.database.schema import create_all_tables

__all__ = ["get_connection", "DatabaseManager", "create_all_tables"]
