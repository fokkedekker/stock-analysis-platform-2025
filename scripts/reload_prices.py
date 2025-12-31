#!/usr/bin/env python3
"""Reload all price data with split adjustments.

This script:
1. Wipes all existing price data from company_profiles
2. Clears the prices checkpoint
3. Re-runs historical price loading with split adjustments

Use this when price data is corrupted or needs to be recalculated with splits.
"""

import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from rich.console import Console
from rich.table import Table

from src.database.connection import get_db_manager
from src.database.schema import create_all_tables

console = Console()


def analyze_current_prices():
    """Analyze current price data to show what will be wiped."""
    db = get_db_manager()

    with db.get_connection() as conn:
        total = conn.execute("SELECT COUNT(*) FROM company_profiles").fetchone()[0]

        quarters = conn.execute("""
            SELECT fiscal_quarter, COUNT(*) as cnt
            FROM company_profiles
            GROUP BY fiscal_quarter
            ORDER BY fiscal_quarter
        """).fetchall()

        # Check for extreme prices (likely unadjusted)
        extreme = conn.execute("""
            SELECT COUNT(*) FROM company_profiles WHERE price > 1000
        """).fetchone()[0]

    return {
        "total": total,
        "quarters": quarters,
        "extreme_prices": extreme,
    }


def wipe_price_data():
    """Delete all rows from company_profiles."""
    db = get_db_manager()

    with db.get_connection() as conn:
        conn.execute("DELETE FROM company_profiles")
        remaining = conn.execute("SELECT COUNT(*) FROM company_profiles").fetchone()[0]

    return remaining


def clear_prices_checkpoint():
    """Clear the prices checkpoint so historical_load.py will re-fetch."""
    checkpoint_path = Path("data/historical_checkpoint.json")

    if not checkpoint_path.exists():
        console.print("[dim]No checkpoint file found.[/dim]")
        return 0

    with open(checkpoint_path, "r") as f:
        data = json.load(f)

    # Checkpoint format is {symbol}:{endpoint} as keys
    # e.g., "AAPL:prices", "AAPL:financials"
    price_count = 0

    # Clear completed prices
    keys_to_remove = [k for k in data.get("completed", {}).keys() if k.endswith(":prices")]
    for key in keys_to_remove:
        del data["completed"][key]
        price_count += 1

    # Also clear failed prices
    failed_keys = [k for k in data.get("failed", {}).keys() if k.endswith(":prices")]
    for key in failed_keys:
        del data["failed"][key]

    with open(checkpoint_path, "w") as f:
        json.dump(data, f, indent=2)

    return price_count


async def reload_prices(from_quarter: str, to_quarter: str, skip_confirm: bool = False):
    """Run the full reload process."""
    # Import here to avoid circular imports
    from scripts.historical_load import (
        generate_quarter_range,
        HistoricalDataFetcher,
        display_existing_data_status,
    )
    from src.config import get_settings
    from src.scrapers.fmp_client import FMPClient

    settings = get_settings()

    console.print("[bold red]Stock Analysis - Price Data Reload[/bold red]")
    console.print()
    console.print("[yellow]This will:[/yellow]")
    console.print("  1. Delete ALL existing price data from company_profiles")
    console.print("  2. Clear the prices checkpoint")
    console.print("  3. Re-fetch all historical prices with split adjustments")
    console.print()

    # Show current state
    analysis = analyze_current_prices()
    console.print(f"[cyan]Current data:[/cyan] {analysis['total']:,} price records")
    console.print(f"[cyan]Extreme prices (>$1000):[/cyan] {analysis['extreme_prices']:,}")
    console.print()

    # Parse quarters
    quarters = generate_quarter_range(from_quarter, to_quarter)
    quarters.reverse()  # Most recent first
    console.print(f"[cyan]Quarters to reload:[/cyan] {from_quarter} to {to_quarter} ({len(quarters)} quarters)")
    console.print()

    if not skip_confirm:
        confirm = console.input("[bold red]Type 'RELOAD' to confirm: [/bold red]")
        if confirm != "RELOAD":
            console.print("[red]Aborted.[/red]")
            return

    # Step 1: Wipe price data
    console.print()
    console.print("[bold]Step 1: Wiping existing price data...[/bold]")
    remaining = wipe_price_data()
    console.print(f"[green]Done. Remaining records: {remaining}[/green]")

    # Step 2: Clear checkpoint
    console.print()
    console.print("[bold]Step 2: Clearing prices checkpoint...[/bold]")
    cleared = clear_prices_checkpoint()
    console.print(f"[green]Cleared checkpoint for {cleared:,} symbols[/green]")

    # Step 3: Re-fetch prices
    console.print()
    console.print("[bold]Step 3: Re-fetching prices with split adjustments...[/bold]")
    console.print(f"[dim]This fetches splits + prices for each stock and applies adjustments[/dim]")
    console.print()

    async with FMPClient() as client:
        fetcher = HistoricalDataFetcher(
            client, quarters,
            force_update=True,  # Overwrite any existing
        )
        symbols = fetcher.get_active_symbols()

        console.print(f"Symbols to process: {len(symbols):,}")
        # 2 API calls per symbol (splits + prices)
        est_calls = len(symbols) * 2
        console.print(f"Est. API calls: {est_calls:,}")
        console.print(f"Est. time: {est_calls / settings.RATE_LIMIT_PER_MINUTE:.1f} min")
        console.print()

        # Only fetch prices (financials should already be in DB)
        await fetcher.fetch_all(symbols, num_workers=settings.NUM_WORKERS)

    console.print()
    console.print("[bold green]Price reload complete![/bold green]")
    console.print()
    console.print("[bold]Next steps:[/bold]")
    console.print("  1. Run analysis to recalculate all metrics:")
    console.print(f"     python scripts/run_analysis.py --from {from_quarter} --to {to_quarter} -y")


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Wipe and reload all price data with split adjustments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/reload_prices.py --from 2010Q1 --to 2025Q4
  python scripts/reload_prices.py --from 2015Q1 --to 2025Q4 -y
        """
    )

    parser.add_argument("--from", dest="from_quarter", type=str, required=True,
                        help="Start quarter (e.g., 2010Q1)")
    parser.add_argument("--to", dest="to_quarter", type=str, required=True,
                        help="End quarter (e.g., 2025Q4)")
    parser.add_argument("-y", "--yes", action="store_true",
                        help="Skip confirmation prompt")

    args = parser.parse_args()

    # Ensure tables exist
    create_all_tables()

    asyncio.run(reload_prices(
        from_quarter=args.from_quarter,
        to_quarter=args.to_quarter,
        skip_confirm=args.yes,
    ))


if __name__ == "__main__":
    main()
