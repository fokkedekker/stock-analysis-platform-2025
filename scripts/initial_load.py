#!/usr/bin/env python3
"""Initial data load script - fetches all financial data for all tickers."""

import asyncio
import logging
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from rich.console import Console
from rich.logging import RichHandler

from src.config import get_settings
from src.database.schema import create_all_tables
from src.scrapers.fmp_client import FMPClient
from src.scrapers.ticker_fetcher import TickerFetcher
from src.scrapers.data_fetcher import DataFetcher

# Configure logging - only show warnings and errors
logging.basicConfig(
    level=logging.WARNING,
    format="%(message)s",
    handlers=[RichHandler(rich_tracebacks=True)],
)
logger = logging.getLogger(__name__)
console = Console()


async def main():
    """Run initial data load."""
    settings = get_settings()

    console.print("[bold blue]Stock Analysis - Initial Data Load[/bold blue]")
    console.print(f"Database: {settings.DATABASE_PATH}")
    console.print(f"Rate limit: {settings.RATE_LIMIT_PER_MINUTE} req/min")
    console.print()

    # Initialize database
    console.print("[yellow]Initializing database...[/yellow]")
    create_all_tables()

    # Step 1: Fetch tickers from Massive API
    console.print("[yellow]Step 1: Fetching stock tickers from Massive API...[/yellow]")
    ticker_fetcher = TickerFetcher()
    ticker_count = await ticker_fetcher.fetch_and_save()
    console.print(f"[green]Saved {ticker_count} tickers[/green]")

    async with FMPClient() as client:

        # Get list of symbols to fetch
        symbols = ticker_fetcher.get_active_symbols()
        console.print(f"[yellow]Step 2: Fetching financial data for {len(symbols)} stocks...[/yellow]")

        # Calculate estimated time
        api_calls = len(symbols) * 11  # 11 endpoints per ticker (annual + quarterly)
        est_minutes = api_calls / settings.RATE_LIMIT_PER_MINUTE
        console.print(f"Estimated API calls: {api_calls:,}")
        console.print(f"Estimated time: {est_minutes:.1f} minutes ({est_minutes/60:.1f} hours)")
        console.print()

        # Confirm before proceeding
        if not console.input("[bold]Continue with data fetch? (y/n): [/bold]").lower().startswith("y"):
            console.print("[red]Aborted.[/red]")
            return

        # Step 2: Fetch financial data
        data_fetcher = DataFetcher(client)
        stats = await data_fetcher.fetch_all(symbols, include_quarterly=True)

        console.print()
        console.print("[bold green]Data fetch complete![/bold green]")
        console.print(f"Completed: {stats['completed']}")
        console.print(f"Failed: {stats['failed']}")
        console.print(f"Skipped: {stats['skipped']}")

        # Report failed symbols
        if stats["failed"] > 0:
            failed = data_fetcher.checkpoint.get_failed_symbols()
            console.print(f"\n[yellow]Failed symbols ({len(failed)}):[/yellow]")
            for sym in failed[:20]:
                console.print(f"  - {sym}")
            if len(failed) > 20:
                console.print(f"  ... and {len(failed) - 20} more")


if __name__ == "__main__":
    asyncio.run(main())
