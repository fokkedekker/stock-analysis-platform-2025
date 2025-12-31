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
from src.database.schema import create_all_tables
from src.scrapers.fmp_client import FMPClient
from src.scrapers.ticker_fetcher import TickerFetcher
from src.scrapers.data_fetcher import DataFetcher, CheckpointManager
from src.scrapers.macro_fetcher import MacroFetcher

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

    # Ensure all tables exist (including macro tables)
    create_all_tables()

    console.print("[bold blue]Stock Analysis - Quarterly Update[/bold blue]")
    console.print()

    # Check current data status
    with db.get_connection() as conn:
        ticker_count = conn.execute(
            "SELECT COUNT(*) FROM tickers WHERE is_active = TRUE"
        ).fetchone()[0]
        latest_quarter_row = conn.execute(
            """
            SELECT DISTINCT
                CAST(EXTRACT(YEAR FROM fiscal_date) AS INTEGER) || 'Q' || CASE
                    WHEN EXTRACT(MONTH FROM fiscal_date) <= 3 THEN '1'
                    WHEN EXTRACT(MONTH FROM fiscal_date) <= 6 THEN '2'
                    WHEN EXTRACT(MONTH FROM fiscal_date) <= 9 THEN '3'
                    ELSE '4'
                END as quarter
            FROM income_statements
            WHERE period = 'quarter' AND fiscal_date IS NOT NULL
            ORDER BY quarter DESC
            LIMIT 1
            """
        ).fetchone()
        latest_quarter = latest_quarter_row[0] if latest_quarter_row else "N/A"

    console.print(f"Current tickers: {ticker_count}")
    console.print(f"Latest quarter: {latest_quarter}")
    console.print()

    async with FMPClient() as client:
        # Step 0: Update treasury rates for rate regime
        console.print("[yellow]Step 0: Updating treasury rates...[/yellow]")
        macro_fetcher = MacroFetcher(client)
        macro_stats = await macro_fetcher.fetch_all_macro_data()
        quarters_with_regimes = macro_fetcher.compute_regime_flags()
        console.print(f"[green]Treasury rates: {macro_stats['treasury_rates']} quarters, rate regimes: {quarters_with_regimes}[/green]")
        console.print()

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

        console.print()
        console.print("[bold yellow]Next step:[/bold yellow] Run analysis to update scores:")
        console.print("  python scripts/run_analysis.py --parallel-workers 8")


if __name__ == "__main__":
    asyncio.run(main())
