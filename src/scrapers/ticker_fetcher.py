"""Ticker fetcher for NYSE/NASDAQ stocks using Massive API."""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import httpx

from src.database.connection import get_db_manager

logger = logging.getLogger(__name__)

# Massive API configuration
MASSIVE_API_KEY = "HT99wi_26T5ZlMpFMIWWpIFpSXGekMi1"
MASSIVE_BASE_URL = "https://api.massive.com/v3/reference/tickers"

# Exchange codes in Massive API
# XNYS = NYSE, XNAS = NASDAQ
INCLUDED_EXCHANGES = {"XNYS", "XNAS"}

# Asset type: CS = Common Stock
INCLUDED_TYPE = "CS"


class TickerFetcher:
    """Fetches and filters stock tickers from Massive API."""

    def __init__(self, fmp_client=None):
        """Initialize ticker fetcher.

        Args:
            fmp_client: Optional FMP client (not used for ticker fetching).
        """
        self.db = get_db_manager()
        self.api_key = MASSIVE_API_KEY

    async def fetch_all_tickers(self) -> list[dict[str, Any]]:
        """Fetch all US common stocks from Massive API.

        Returns:
            List of all stock tickers.
        """
        logger.info("Fetching stock list from Massive API...")

        all_tickers = []

        async with httpx.AsyncClient(timeout=30.0) as client:
            # Initial request
            url = MASSIVE_BASE_URL
            params = {
                "market": "stocks",
                "active": "true",
                "type": INCLUDED_TYPE,
                "limit": 1000,
                "apiKey": self.api_key,
            }

            page = 1
            while url:
                logger.info(f"Fetching page {page}...")

                if page == 1:
                    response = await client.get(url, params=params)
                else:
                    # next_url already includes all params
                    response = await client.get(url)

                response.raise_for_status()
                data = response.json()

                results = data.get("results", [])
                all_tickers.extend(results)

                # Get next page URL
                url = data.get("next_url")
                if url:
                    # Add API key to next_url
                    url = f"{url}&apiKey={self.api_key}"

                page += 1

        logger.info(f"Fetched {len(all_tickers)} total tickers from Massive API")
        return all_tickers

    def filter_tickers(
        self,
        tickers: list[dict[str, Any]],
        exchanges: set[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Filter tickers by exchange.

        Args:
            tickers: List of ticker dictionaries.
            exchanges: Set of exchange codes to include (e.g., XNYS, XNAS).

        Returns:
            Filtered list of tickers.
        """
        exchanges = exchanges or INCLUDED_EXCHANGES

        filtered = []
        for ticker in tickers:
            exchange = ticker.get("primary_exchange", "")

            if exchange in exchanges:
                filtered.append(ticker)

        logger.info(
            f"Filtered {len(tickers)} tickers down to {len(filtered)} "
            f"(exchanges: {exchanges})"
        )
        return filtered

    def save_to_database(self, tickers: list[dict[str, Any]]) -> int:
        """Save tickers to the database.

        Args:
            tickers: List of ticker dictionaries from Massive API.

        Returns:
            Number of tickers saved.
        """
        now = datetime.utcnow()

        # Map Massive exchange codes to readable names
        exchange_map = {
            "XNYS": "NYSE",
            "XNAS": "NASDAQ",
            "XASE": "AMEX",
        }

        with self.db.get_connection() as conn:
            saved_count = 0
            for ticker in tickers:
                try:
                    exchange_code = ticker.get("primary_exchange", "")
                    exchange_name = exchange_map.get(exchange_code, exchange_code)

                    conn.execute(
                        """
                        INSERT INTO tickers (
                            symbol, name, exchange, sector, industry,
                            asset_type, is_active, created_at, updated_at, raw_json
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        ON CONFLICT (symbol) DO UPDATE SET
                            name = EXCLUDED.name,
                            exchange = EXCLUDED.exchange,
                            is_active = EXCLUDED.is_active,
                            updated_at = EXCLUDED.updated_at,
                            raw_json = EXCLUDED.raw_json
                        """,
                        (
                            ticker.get("ticker"),
                            ticker.get("name"),
                            exchange_name,
                            None,  # Sector not in Massive response
                            None,  # Industry not in Massive response
                            ticker.get("type"),
                            ticker.get("active", True),
                            now,
                            now,
                            json.dumps(ticker),
                        ),
                    )
                    saved_count += 1
                except Exception as e:
                    logger.error(f"Error saving ticker {ticker.get('ticker')}: {e}")

            logger.info(f"Saved {saved_count} tickers to database")
            return saved_count

    def get_active_symbols(self) -> list[str]:
        """Get list of active ticker symbols from database.

        Returns:
            List of symbol strings.
        """
        with self.db.get_connection() as conn:
            result = conn.execute(
                "SELECT symbol FROM tickers WHERE is_active = TRUE ORDER BY symbol"
            ).fetchall()
            return [row[0] for row in result]

    def get_ticker_count(self) -> int:
        """Get count of active tickers.

        Returns:
            Number of active tickers.
        """
        with self.db.get_connection() as conn:
            result = conn.execute(
                "SELECT COUNT(*) FROM tickers WHERE is_active = TRUE"
            ).fetchone()
            return result[0] if result else 0

    async def fetch_and_save(
        self,
        exchanges: set[str] | None = None,
    ) -> int:
        """Fetch tickers from Massive API and save to database.

        Args:
            exchanges: Set of exchange codes to include.

        Returns:
            Number of tickers saved.
        """
        all_tickers = await self.fetch_all_tickers()
        filtered = self.filter_tickers(all_tickers, exchanges)
        return self.save_to_database(filtered)

    def export_to_file(self, filepath: str | Path) -> None:
        """Export active tickers to a JSON file.

        Args:
            filepath: Path to output file.
        """
        symbols = self.get_active_symbols()
        Path(filepath).write_text(json.dumps(symbols, indent=2))
        logger.info(f"Exported {len(symbols)} symbols to {filepath}")
