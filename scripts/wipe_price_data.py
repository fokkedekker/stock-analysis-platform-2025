#!/usr/bin/env python3
"""Wipe all price data from the database.

This script removes all price-related data from company_profiles
and clears the prices checkpoint so historical_load.py will re-fetch.

Run this when price data is corrupted and needs a clean re-sync.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from rich.console import Console
from rich.table import Table

from src.database.connection import get_db_manager

console = Console()


def analyze_price_data():
    """Analyze current price data quality."""
    db = get_db_manager()

    with db.get_connection() as conn:
        # Total profiles
        total = conn.execute("SELECT COUNT(*) FROM company_profiles").fetchone()[0]

        # Profiles with extreme prices
        extreme_high = conn.execute(
            "SELECT COUNT(*) FROM company_profiles WHERE price > 10000"
        ).fetchone()[0]

        # Profiles with very high prices (potentially bad)
        high = conn.execute(
            "SELECT COUNT(*) FROM company_profiles WHERE price > 5000 AND price <= 10000"
        ).fetchone()[0]

        # NULL prices
        null_prices = conn.execute(
            "SELECT COUNT(*) FROM company_profiles WHERE price IS NULL"
        ).fetchone()[0]

        # Unique quarters
        quarters = conn.execute(
            "SELECT DISTINCT fiscal_quarter FROM company_profiles ORDER BY fiscal_quarter"
        ).fetchall()

        # Sample of extreme prices
        extreme_samples = conn.execute("""
            SELECT symbol, fiscal_quarter, price
            FROM company_profiles
            WHERE price > 5000
            ORDER BY price DESC
            LIMIT 20
        """).fetchall()

    return {
        "total": total,
        "extreme_high": extreme_high,
        "high": high,
        "null_prices": null_prices,
        "quarters": [q[0] for q in quarters],
        "extreme_samples": extreme_samples,
    }


def wipe_price_data(dry_run: bool = True):
    """Wipe all price data from company_profiles."""
    db = get_db_manager()

    with db.get_connection() as conn:
        if dry_run:
            count = conn.execute("SELECT COUNT(*) FROM company_profiles").fetchone()[0]
            console.print(f"[yellow]DRY RUN: Would delete {count:,} rows from company_profiles[/yellow]")
            return count

        # Delete all rows from company_profiles
        conn.execute("DELETE FROM company_profiles")
        count = conn.execute("SELECT COUNT(*) FROM company_profiles").fetchone()[0]
        console.print(f"[green]Deleted all rows from company_profiles. Remaining: {count}[/green]")

        return count


def clear_price_checkpoint():
    """Clear the prices checkpoint so historical_load.py will re-fetch."""
    checkpoint_path = Path("data/historical_checkpoint.json")

    if not checkpoint_path.exists():
        console.print("[dim]No checkpoint file found.[/dim]")
        return

    with open(checkpoint_path, "r") as f:
        data = json.load(f)

    # Count how many price checkpoints we have
    price_count = 0
    for symbol, endpoints in data.get("completed", {}).items():
        if "prices" in endpoints:
            price_count += 1

    console.print(f"[cyan]Found {price_count:,} symbols with prices checkpoint[/cyan]")

    # Remove all "prices" entries from checkpoint
    for symbol in list(data.get("completed", {}).keys()):
        if "prices" in data["completed"][symbol]:
            del data["completed"][symbol]["prices"]
            # If no endpoints left for this symbol, remove the symbol
            if not data["completed"][symbol]:
                del data["completed"][symbol]

    # Also clear failed prices
    for symbol in list(data.get("failed", {}).keys()):
        if "prices" in data["failed"][symbol]:
            del data["failed"][symbol]["prices"]
            if not data["failed"][symbol]:
                del data["failed"][symbol]

    # Save updated checkpoint
    with open(checkpoint_path, "w") as f:
        json.dump(data, f, indent=2)

    console.print(f"[green]Cleared prices checkpoint for {price_count:,} symbols[/green]")


def main(dry_run: bool = True, skip_confirm: bool = False):
    console.print("[bold red]Stock Analysis - Price Data Wipe Tool[/bold red]")
    console.print()

    # Analyze current state
    console.print("[bold]Current Price Data Analysis:[/bold]")
    analysis = analyze_price_data()

    table = Table(show_header=False)
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Total profiles", f"{analysis['total']:,}")
    table.add_row("Extreme prices (>$10,000)", f"{analysis['extreme_high']:,}")
    table.add_row("High prices ($5,000-$10,000)", f"{analysis['high']:,}")
    table.add_row("NULL prices", f"{analysis['null_prices']:,}")
    table.add_row("Quarters covered", f"{len(analysis['quarters'])}")

    console.print(table)

    if analysis['extreme_samples']:
        console.print("\n[bold]Sample of extreme prices:[/bold]")
        sample_table = Table()
        sample_table.add_column("Symbol")
        sample_table.add_column("Quarter")
        sample_table.add_column("Price", justify="right")

        for symbol, quarter, price in analysis['extreme_samples'][:10]:
            sample_table.add_row(symbol, quarter, f"${price:,.2f}")

        console.print(sample_table)

    console.print()

    if dry_run:
        console.print("[yellow]DRY RUN MODE - No changes will be made[/yellow]")
        console.print()
        console.print("[bold]This will:[/bold]")
        console.print(f"  1. Delete all {analysis['total']:,} rows from company_profiles")
        console.print("  2. Clear the 'prices' checkpoint for all symbols")
        console.print()
        console.print("[bold]After running this, you need to:[/bold]")
        console.print("  python scripts/historical_load.py --from 2010Q1 --to 2025Q4 --fresh -y")
        return

    if not skip_confirm:
        console.print("[bold red]WARNING: This will delete ALL price data![/bold red]")
        if not console.input("[bold]Type 'DELETE' to confirm: [/bold]") == "DELETE":
            console.print("[red]Aborted.[/red]")
            return

    console.print()
    console.print("[bold]Wiping price data...[/bold]")
    wipe_price_data(dry_run=False)

    console.print()
    console.print("[bold]Clearing price checkpoint...[/bold]")
    clear_price_checkpoint()

    console.print()
    console.print("[bold green]Done![/bold green]")
    console.print()
    console.print("[bold]Next steps:[/bold]")
    console.print("  1. Re-sync prices from FMP API:")
    console.print("     python scripts/historical_load.py --from 2010Q1 --to 2025Q4 --fresh -y")
    console.print()
    console.print("  2. Re-run analysis after prices are synced:")
    console.print("     python scripts/run_analysis.py --from 2010Q1 --to 2025Q4 -y")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Wipe all price data for clean re-sync")
    parser.add_argument("--execute", action="store_true", help="Actually delete data (default: dry run)")
    parser.add_argument("-y", "--yes", action="store_true", help="Skip confirmation")

    args = parser.parse_args()

    main(dry_run=not args.execute, skip_confirm=args.yes)
