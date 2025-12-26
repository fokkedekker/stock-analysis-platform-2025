#!/usr/bin/env python3
"""Quarterly update script - fetches new data and refreshes profiles."""

import asyncio
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from rich.console import Console
from rich.logging import RichHandler

from src.config import get_settings
from src.database.connection import get_db_manager
from src.scrapers.fmp_client import FMPClient
from src.scrapers.ticker_fetcher import TickerFetcher
from src.scrapers.data_fetcher import DataFetcher, CheckpointManager

logging.basicConfig(
    level=logging.WARNING,
    format="%(message)s",
    handlers=[RichHandler(rich_tracebacks=True)],
)
logger = logging.getLogger(__name__)
console = Console()


async def main():
    """Run quarterly update."""
    settings = get_settings()
    db = get_db_manager()

    console.print("[bold blue]Stock Analysis - Quarterly Update[/bold blue]")
    console.print()

    # Check current data status
    with db.get_connection() as conn:
        ticker_count = conn.execute(
            "SELECT COUNT(*) FROM tickers WHERE is_active = TRUE"
        ).fetchone()[0]
        latest_quarter = conn.execute(
            "SELECT MAX(fiscal_quarter) FROM company_profiles"
        ).fetchone()[0]

    console.print(f"Current tickers: {ticker_count}")
    console.print(f"Latest quarter: {latest_quarter}")
    console.print()

    async with FMPClient() as client:
        # Step 1: Update ticker list
        console.print("[yellow]Step 1: Updating ticker list...[/yellow]")
        ticker_fetcher = TickerFetcher(client)
        new_count = await ticker_fetcher.fetch_and_save()
        console.print(f"[green]Updated {new_count} tickers[/green]")

        # Get symbols
        symbols = ticker_fetcher.get_active_symbols()
        console.print(f"\n[yellow]Step 2: Refreshing profiles for {len(symbols)} stocks...[/yellow]")

        # For quarterly update, we mainly need to refresh profiles and get latest quarter
        # Use a fresh checkpoint for this run
        checkpoint_path = Path(settings.CHECKPOINT_PATH).parent / "quarterly_checkpoint.json"
        checkpoint = CheckpointManager(checkpoint_path)
        checkpoint.clear()

        data_fetcher = DataFetcher(client, checkpoint_path)

        # Estimate time
        api_calls = len(symbols) * 2  # Profile + latest income statement
        est_minutes = api_calls / settings.RATE_LIMIT_PER_MINUTE
        console.print(f"Estimated API calls: {api_calls:,}")
        console.print(f"Estimated time: {est_minutes:.1f} minutes")

        # Fetch data
        stats = await data_fetcher.fetch_all(symbols, include_quarterly=True)

        console.print()
        console.print("[bold green]Quarterly update complete![/bold green]")
        console.print(f"Updated: {stats['completed']}")
        console.print(f"Failed: {stats['failed']}")


if __name__ == "__main__":
    asyncio.run(main())
