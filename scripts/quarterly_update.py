#!/usr/bin/env python3
"""Quarterly update script - fetches new data and refreshes profiles."""

import asyncio
import logging
import sys
from datetime import datetime, timezone
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


def sync_spy_prices(console: Console) -> int:
    """Sync SPY quarterly prices using yfinance.

    Returns number of quarters added.
    """
    try:
        import yfinance as yf
    except ImportError:
        console.print("[yellow]yfinance not installed, skipping SPY sync[/yellow]")
        return 0

    db = get_db_manager()

    # Determine date range needed
    now = datetime.now(timezone.utc)
    start_year = 2010
    end_year = now.year

    # Generate all quarters we might need
    all_quarters = []
    for year in range(start_year, end_year + 1):
        for q in range(1, 5):
            quarter_str = f"{year}Q{q}"
            all_quarters.append(quarter_str)

    # Check which quarters we already have
    with db.get_connection() as conn:
        existing = conn.execute("SELECT quarter FROM spy_prices").fetchall()
        existing_quarters = {row[0] for row in existing}

    missing_quarters = [q for q in all_quarters if q not in existing_quarters]

    if not missing_quarters:
        return 0

    # Fetch SPY data from yfinance
    spy = yf.Ticker('SPY')
    hist = spy.history(start=f'{start_year}-01-01', end=f'{end_year + 1}-01-01')

    if hist.empty:
        console.print("[yellow]Could not fetch SPY data from yfinance[/yellow]")
        return 0

    # Map quarters to end dates
    quarter_end_map = {
        1: ('03', '31'), 2: ('06', '30'), 3: ('09', '30'), 4: ('12', '31')
    }

    # Get prices for missing quarters
    new_prices = {}
    for quarter in missing_quarters:
        year = int(quarter[:4])
        q = int(quarter[5])
        month, day = quarter_end_map[q]
        date_str = f"{year}-{month}-{day}"

        # Find closest date on or before quarter end
        mask = hist.index <= date_str
        if mask.any():
            closest = hist.index[mask][-1]
            price = float(hist.loc[closest, 'Close'])
            new_prices[quarter] = round(price, 2)

    # Insert new prices
    if new_prices:
        with db.get_connection() as conn:
            for quarter, price in new_prices.items():
                conn.execute(
                    "INSERT OR REPLACE INTO spy_prices (quarter, price, fetched_at) VALUES (?, ?, CURRENT_TIMESTAMP)",
                    (quarter, price)
                )

    return len(new_prices)

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
        # Step 0a: Update treasury rates for rate regime
        console.print("[yellow]Step 0a: Updating treasury rates...[/yellow]")
        macro_fetcher = MacroFetcher(client)
        macro_stats = await macro_fetcher.fetch_all_macro_data()
        quarters_with_regimes = macro_fetcher.compute_regime_flags()
        console.print(f"[green]Treasury rates: {macro_stats['treasury_rates']} quarters, rate regimes: {quarters_with_regimes}[/green]")

        # Step 0b: Sync SPY benchmark prices
        console.print("[yellow]Step 0b: Syncing SPY benchmark prices...[/yellow]")
        spy_added = sync_spy_prices(console)
        if spy_added > 0:
            console.print(f"[green]Added {spy_added} SPY quarterly prices[/green]")
        else:
            console.print("[green]SPY prices up to date[/green]")
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
