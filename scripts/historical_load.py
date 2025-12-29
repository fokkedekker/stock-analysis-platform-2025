#!/usr/bin/env python3
"""Historical data load - fetches EVERYTHING needed for analysis for a date range.

This script fetches:
1. Financial statements (income, balance, cashflow, metrics) with sufficient history
2. Historical prices for quarter-end dates
3. Calculates PE/PB ratios from the data
"""

import asyncio
import json
import logging
import sys
from dataclasses import dataclass
from datetime import datetime, date, timedelta, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent))

from rich.console import Console
from rich.logging import RichHandler
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, TimeElapsedColumn

from src.config import get_settings
from src.database.connection import get_db_manager
from src.scrapers.fmp_client import FMPClient
from src.scrapers.data_fetcher import CheckpointManager, DataFetcher

logging.basicConfig(
    level=logging.WARNING,
    format="%(message)s",
    handlers=[RichHandler(rich_tracebacks=True)],
)
logger = logging.getLogger(__name__)
console = Console()


def get_quarter_end_dates(num_quarters: int = 8) -> list[tuple[str, date]]:
    """Get quarter-end dates for the last N quarters."""
    now = datetime.now(timezone.utc)
    year = now.year
    quarter = (now.month - 1) // 3 + 1

    quarters = []
    for _ in range(num_quarters):
        end_date = date(year, [3, 6, 9, 12][quarter - 1], [31, 30, 30, 31][quarter - 1])
        quarters.append((f"{year}Q{quarter}", end_date))
        quarter -= 1
        if quarter == 0:
            quarter = 4
            year -= 1
    return quarters


def parse_quarter(quarter_str: str) -> tuple[int, int]:
    """Parse quarter string like '2023Q4' into (year, quarter)."""
    import re
    match = re.match(r'^(\d{4})Q([1-4])$', quarter_str.upper())
    if not match:
        raise ValueError(f"Invalid quarter format: '{quarter_str}'. Expected format: 2024Q1, 2024Q2, etc.")
    year, q = int(match.group(1)), int(match.group(2))

    if year < 1990 or year > 2030:
        raise ValueError(f"Year {year} seems unreasonable (expected 1990-2030)")

    return year, q


def quarter_to_end_date(year: int, quarter: int) -> date:
    """Convert year/quarter to quarter-end date."""
    end_months = [3, 6, 9, 12]
    end_days = [31, 30, 30, 31]
    return date(year, end_months[quarter - 1], end_days[quarter - 1])


def generate_quarter_range(from_q: str, to_q: str) -> list[tuple[str, date]]:
    """Generate list of quarters from start to end (inclusive)."""
    from_year, from_qtr = parse_quarter(from_q)
    to_year, to_qtr = parse_quarter(to_q)

    quarters = []
    year, qtr = from_year, from_qtr

    while (year, qtr) <= (to_year, to_qtr):
        end_date = quarter_to_end_date(year, qtr)
        quarters.append((f"{year}Q{qtr}", end_date))

        qtr += 1
        if qtr > 4:
            qtr = 1
            year += 1

        if len(quarters) > 80:
            raise ValueError("Quarter range too large (max 80 quarters / 20 years)")

    return quarters


def generate_year_quarters(year: int) -> list[tuple[str, date]]:
    """Generate all 4 quarters for a given year."""
    return [(f"{year}Q{q}", quarter_to_end_date(year, q)) for q in range(1, 5)]


def calculate_required_limit(quarters: list[tuple[str, date]]) -> int:
    """Calculate how many quarters of financial data we need.

    We need data going back to the earliest quarter, plus 4 extra quarters
    for TTM calculations, plus some buffer.
    """
    now = datetime.now(timezone.utc).date()
    earliest_quarter = min(q[1] for q in quarters)

    # Calculate quarters from now to earliest
    quarters_back = ((now.year - earliest_quarter.year) * 4 +
                     ((now.month - 1) // 3 + 1) -
                     ((earliest_quarter.month - 1) // 3 + 1))

    # Add 8 quarters buffer for TTM and safety
    return max(quarters_back + 8, 30)


@dataclass
class FetchTask:
    symbol: str


class HistoricalDataFetcher:
    """Fetches ALL historical data needed for analysis."""

    def __init__(self, fmp_client: FMPClient, quarters: list[tuple[str, date]],
                 checkpoint_path: str = "data/historical_checkpoint.json",
                 force_update: bool = False,
                 financial_limit: int = 60):
        self.client = fmp_client
        self.db = get_db_manager()
        self.quarters = quarters
        self.checkpoint = CheckpointManager(checkpoint_path)
        self.force_update = force_update
        self.financial_limit = financial_limit
        self.data_fetcher = DataFetcher(fmp_client, checkpoint_path)

        # Calculate date range for prices
        self.from_date = (min(q[1] for q in quarters) - timedelta(days=10)).isoformat()
        self.to_date = max(q[1] for q in quarters).isoformat()

    def get_active_symbols(self) -> list[str]:
        with self.db.get_connection() as conn:
            return [r[0] for r in conn.execute(
                "SELECT symbol FROM tickers WHERE is_active = true ORDER BY symbol"
            ).fetchall()]

    def _find_price_for_date(self, prices: list[dict], target: date) -> dict | None:
        """Find price record for target date or nearest prior trading day."""
        target_str = target.isoformat()
        for p in prices:
            if p.get("date", "") <= target_str:
                return p
        return None

    def _get_financial_data(self, symbol: str, quarter_end: date, conn) -> dict:
        """Get EPS and book value for a quarter."""
        result = {"eps": None, "book_value_per_share": None, "shares_outstanding": None}

        # TTM EPS
        row = conn.execute("""
            SELECT SUM(eps) FROM (
                SELECT eps FROM income_statements
                WHERE symbol = ? AND period = 'quarter' AND fiscal_date <= ? AND eps IS NOT NULL
                ORDER BY fiscal_date DESC LIMIT 4
            )
        """, (symbol, quarter_end.isoformat())).fetchone()
        if row and row[0]:
            result["eps"] = float(row[0])

        # Book value per share
        row = conn.execute("""
            SELECT book_value_per_share FROM key_metrics
            WHERE symbol = ? AND period = 'quarter' AND fiscal_date <= ? AND book_value_per_share IS NOT NULL
            ORDER BY fiscal_date DESC LIMIT 1
        """, (symbol, quarter_end.isoformat())).fetchone()
        if row and row[0]:
            result["book_value_per_share"] = float(row[0])

        # Shares outstanding
        row = conn.execute("""
            SELECT common_shares_outstanding FROM balance_sheets
            WHERE symbol = ? AND period = 'quarter' AND fiscal_date <= ? AND common_shares_outstanding IS NOT NULL
            ORDER BY fiscal_date DESC LIMIT 1
        """, (symbol, quarter_end.isoformat())).fetchone()
        if row and row[0]:
            result["shares_outstanding"] = int(row[0])

        return result

    def _save_profile(self, symbol: str, quarter: str, price: float, market_cap: float | None,
                      pe_ratio: float | None, pb_ratio: float | None, shares: int | None,
                      conn, force_update: bool = False) -> None:
        """Save to company_profiles."""
        if force_update:
            conn.execute("""
                INSERT INTO company_profiles (symbol, fiscal_quarter, price, market_cap, pe_ratio, pb_ratio, shares_outstanding, fetched_at, raw_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT (symbol, fiscal_quarter) DO UPDATE SET
                    price = EXCLUDED.price, market_cap = EXCLUDED.market_cap, pe_ratio = EXCLUDED.pe_ratio,
                    pb_ratio = EXCLUDED.pb_ratio, shares_outstanding = EXCLUDED.shares_outstanding, fetched_at = EXCLUDED.fetched_at
            """, (symbol, quarter, price, market_cap, pe_ratio, pb_ratio, shares,
                  datetime.now(timezone.utc), json.dumps({"source": "historical_load"})))
        else:
            conn.execute("""
                INSERT INTO company_profiles (symbol, fiscal_quarter, price, market_cap, pe_ratio, pb_ratio, shares_outstanding, fetched_at, raw_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT (symbol, fiscal_quarter) DO NOTHING
            """, (symbol, quarter, price, market_cap, pe_ratio, pb_ratio, shares,
                  datetime.now(timezone.utc), json.dumps({"source": "historical_load"})))

    async def _fetch_financials_worker(self, queue: asyncio.Queue, stats: dict, stats_lock: asyncio.Lock,
                                        progress: Progress | None, task_id: int | None, worker_id: int) -> None:
        """Worker for fetching financial statements."""
        db_conn = self.db.connect()
        try:
            while True:
                try:
                    task: FetchTask = queue.get_nowait()
                except asyncio.QueueEmpty:
                    break

                symbol = task.symbol
                try:
                    if self.checkpoint.is_endpoint_completed(symbol, "financials"):
                        async with stats_lock:
                            stats["skipped"] += 1
                        continue

                    # Fetch all financial statements with extended limit
                    income = await self.client.get_income_statement(symbol, "quarter", self.financial_limit)
                    balance = await self.client.get_balance_sheet(symbol, "quarter", self.financial_limit)
                    cashflow = await self.client.get_cash_flow(symbol, "quarter", self.financial_limit)
                    metrics = await self.client.get_key_metrics(symbol, "quarter", self.financial_limit)

                    # Save to database using DataFetcher's save methods
                    self.data_fetcher.save_income_statements(symbol, income, "quarter", db_conn)
                    self.data_fetcher.save_balance_sheets(symbol, balance, "quarter", db_conn)
                    self.data_fetcher.save_cash_flow_statements(symbol, cashflow, "quarter", db_conn)
                    self.data_fetcher.save_key_metrics(symbol, metrics, "quarter", db_conn)

                    await self.checkpoint.mark_endpoint_completed(symbol, "financials")
                    async with stats_lock:
                        stats["completed"] += 1

                except Exception as e:
                    logger.error(f"Worker {worker_id}: {symbol} financials failed: {e}")
                    await self.checkpoint.mark_endpoint_failed(symbol, "financials", str(e))
                    async with stats_lock:
                        stats["failed"] += 1

                finally:
                    queue.task_done()
                    if progress and task_id is not None:
                        progress.update(task_id, advance=1)
        finally:
            db_conn.close()

    async def _fetch_prices_worker(self, queue: asyncio.Queue, stats: dict, stats_lock: asyncio.Lock,
                                    progress: Progress | None, task_id: int | None, worker_id: int) -> None:
        """Worker for fetching historical prices."""
        db_conn = self.db.connect()
        try:
            while True:
                try:
                    task: FetchTask = queue.get_nowait()
                except asyncio.QueueEmpty:
                    break

                symbol = task.symbol
                try:
                    if self.checkpoint.is_endpoint_completed(symbol, "prices"):
                        async with stats_lock:
                            stats["skipped"] += 1
                        continue

                    prices = await self.client.get_historical_prices(symbol, self.from_date, self.to_date)

                    if not prices:
                        await self.checkpoint.mark_endpoint_completed(symbol, "prices")
                        async with stats_lock:
                            stats["skipped"] += 1
                        continue

                    for quarter_str, quarter_end in self.quarters:
                        price_rec = self._find_price_for_date(prices, quarter_end)
                        if not price_rec:
                            continue

                        price = price_rec.get("adjClose") or price_rec.get("close")
                        if not price:
                            continue
                        price = float(price)

                        fin = self._get_financial_data(symbol, quarter_end, db_conn)
                        pe = price / fin["eps"] if fin["eps"] and fin["eps"] > 0 else None
                        pb = price / fin["book_value_per_share"] if fin["book_value_per_share"] and fin["book_value_per_share"] > 0 else None

                        self._save_profile(symbol, quarter_str, price, None, pe, pb, None, db_conn, self.force_update)

                    await self.checkpoint.mark_endpoint_completed(symbol, "prices")
                    async with stats_lock:
                        stats["completed"] += 1

                except Exception as e:
                    logger.error(f"Worker {worker_id}: {symbol} prices failed: {e}")
                    await self.checkpoint.mark_endpoint_failed(symbol, "prices", str(e))
                    async with stats_lock:
                        stats["failed"] += 1

                finally:
                    queue.task_done()
                    if progress and task_id is not None:
                        progress.update(task_id, advance=1)
        finally:
            db_conn.close()

    async def fetch_all(self, symbols: list[str], num_workers: int = 10) -> dict:
        """Fetch all historical data for all symbols."""
        settings = get_settings()
        self.checkpoint.start_session()

        # Phase 1: Fetch financial statements
        console.print("\n[bold cyan]Phase 1: Fetching financial statements[/bold cyan]")
        console.print(f"[dim]Fetching {self.financial_limit} quarters of history per symbol[/dim]")

        queue: asyncio.Queue[FetchTask] = asyncio.Queue()
        total = 0
        skipped = 0

        for sym in symbols:
            if self.checkpoint.is_endpoint_completed(sym, "financials"):
                skipped += 1
            else:
                await queue.put(FetchTask(sym))
                total += 1

        stats_financials = {"completed": 0, "failed": 0, "skipped": skipped}
        stats_lock = asyncio.Lock()

        if total == 0:
            console.print("[green]All financial statements already fetched![/green]")
        else:
            console.print(f"[cyan]Fetching {total:,} symbols ({skipped:,} skipped)[/cyan]")
            # 4 API calls per symbol (income, balance, cashflow, metrics)
            console.print(f"[dim]Est. time: {(total * 4) / settings.RATE_LIMIT_PER_MINUTE:.1f} min[/dim]")

            with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"),
                          BarColumn(), TaskProgressColumn(), TimeElapsedColumn()) as progress:
                task_id = progress.add_task("Fetching financials...", total=total)
                workers = [asyncio.create_task(self._fetch_financials_worker(queue, stats_financials, stats_lock, progress, task_id, i))
                           for i in range(num_workers)]
                await asyncio.gather(*workers)

            console.print(f"  Completed: {stats_financials['completed']:,}, Failed: {stats_financials['failed']:,}, Skipped: {stats_financials['skipped']:,}")

        # Phase 2: Fetch historical prices and calculate PE/PB
        console.print("\n[bold cyan]Phase 2: Fetching historical prices[/bold cyan]")
        console.print(f"[dim]Date range: {self.from_date} to {self.to_date}[/dim]")

        queue = asyncio.Queue()
        total = 0
        skipped = 0

        for sym in symbols:
            if self.checkpoint.is_endpoint_completed(sym, "prices"):
                skipped += 1
            else:
                await queue.put(FetchTask(sym))
                total += 1

        stats_prices = {"completed": 0, "failed": 0, "skipped": skipped}
        stats_lock = asyncio.Lock()

        if total == 0:
            console.print("[green]All prices already fetched![/green]")
        else:
            console.print(f"[cyan]Fetching {total:,} symbols ({skipped:,} skipped)[/cyan]")
            console.print(f"[dim]Est. time: {total / settings.RATE_LIMIT_PER_MINUTE:.1f} min[/dim]")

            with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"),
                          BarColumn(), TaskProgressColumn(), TimeElapsedColumn()) as progress:
                task_id = progress.add_task("Fetching prices...", total=total)
                workers = [asyncio.create_task(self._fetch_prices_worker(queue, stats_prices, stats_lock, progress, task_id, i))
                           for i in range(num_workers)]
                await asyncio.gather(*workers)

            console.print(f"  Completed: {stats_prices['completed']:,}, Failed: {stats_prices['failed']:,}, Skipped: {stats_prices['skipped']:,}")

        self.checkpoint.force_save()

        console.print()
        console.print("[bold green]Done![/bold green]")

        return {"financials": stats_financials, "prices": stats_prices}


def get_existing_data_status(quarters: list[tuple[str, date]]) -> dict[str, int]:
    """Check what data already exists in the database."""
    db = get_db_manager()
    quarter_strs = [q[0] for q in quarters]

    with db.get_connection() as conn:
        placeholders = ", ".join(["?" for _ in quarter_strs])
        result = conn.execute(f"""
            SELECT fiscal_quarter, COUNT(*) as cnt
            FROM company_profiles
            WHERE fiscal_quarter IN ({placeholders})
            GROUP BY fiscal_quarter
            ORDER BY fiscal_quarter DESC
        """, quarter_strs).fetchall()

        return {row[0]: row[1] for row in result}


def display_existing_data_status(quarters: list[tuple[str, date]], force_update: bool) -> None:
    """Display what data exists and what will happen."""
    status = get_existing_data_status(quarters)

    console.print("\n[bold]Company Profiles Status:[/bold]")
    console.print("-" * 50)

    for quarter_str, _ in quarters:
        count = status.get(quarter_str, 0)
        if count > 0:
            if force_update:
                console.print(f"  {quarter_str}: {count:,} existing [yellow](will be overwritten)[/yellow]")
            else:
                console.print(f"  {quarter_str}: {count:,} existing [green](will be preserved)[/green]")
        else:
            console.print(f"  {quarter_str}: [dim]no data[/dim] [cyan](will be fetched)[/cyan]")

    console.print("-" * 50)


async def main(quarters: list[tuple[str, date]], force_update: bool = False,
               skip_confirm: bool = False, dry_run: bool = False, fresh: bool = False):
    settings = get_settings()
    checkpoint_path = Path("data/historical_checkpoint.json")

    # Calculate required financial data limit
    financial_limit = calculate_required_limit(quarters)

    console.print("[bold blue]Stock Analysis - Historical Data Load[/bold blue]")
    console.print(f"Rate limit: {settings.RATE_LIMIT_PER_MINUTE} req/min")
    console.print()

    console.print("[yellow]Quarters to process:[/yellow]", ", ".join(q[0] for q in quarters))
    console.print(f"[dim]Financial statement limit: {financial_limit} quarters[/dim]")

    if force_update:
        console.print("[red]Mode: FORCE UPDATE (will overwrite existing data)[/red]")
    else:
        console.print("[green]Mode: SKIP EXISTING (will preserve existing data)[/green]")

    if fresh:
        console.print("[yellow]Fresh start: checkpoint will be cleared[/yellow]")

    display_existing_data_status(quarters, force_update)

    if dry_run:
        console.print("\n[yellow]Dry run - no changes will be made.[/yellow]")
        return

    if fresh and checkpoint_path.exists():
        checkpoint_path.unlink()
        console.print("[dim]Checkpoint cleared.[/dim]")

    async with FMPClient() as client:
        fetcher = HistoricalDataFetcher(
            client, quarters,
            force_update=force_update,
            financial_limit=financial_limit
        )
        symbols = fetcher.get_active_symbols()

        console.print(f"\nSymbols to process: {len(symbols):,}")
        # 4 API calls for financials + 1 for prices = 5 per symbol
        est_calls = len(symbols) * 5
        console.print(f"Est. API calls: {est_calls:,}")
        console.print(f"Est. time: {est_calls / settings.RATE_LIMIT_PER_MINUTE:.1f} min")
        console.print()

        if not skip_confirm:
            if not console.input("[bold]Continue? (y/n): [/bold]").lower().startswith("y"):
                console.print("[red]Aborted.[/red]")
                return

        await fetcher.fetch_all(symbols, num_workers=settings.NUM_WORKERS)

        console.print()
        console.print("[bold]Next: run analysis for each quarter[/bold]")
        console.print(f"  python scripts/run_analysis.py --from {quarters[-1][0]} --to {quarters[0][0]}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Load ALL historical data needed for analysis (financials + prices).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/historical_load.py --from 2010Q1 --to 2023Q4 --fresh
  python scripts/historical_load.py --year 2023 --fresh
  python scripts/historical_load.py --quarters 2023Q1 2023Q2 --fresh
  python scripts/historical_load.py --dry-run                # Preview
  python scripts/historical_load.py --force                  # Overwrite existing
        """
    )

    quarter_group = parser.add_mutually_exclusive_group()
    quarter_group.add_argument(
        "--quarters", type=str, nargs="+", metavar="QUARTER",
        help="Explicit list of quarters (e.g., 2023Q1 2023Q2)"
    )
    quarter_group.add_argument(
        "--year", type=int, metavar="YEAR",
        help="Sync all 4 quarters for a specific year"
    )

    parser.add_argument("--from", dest="from_quarter", type=str, metavar="QUARTER",
                        help="Start quarter for range (requires --to)")
    parser.add_argument("--to", dest="to_quarter", type=str, metavar="QUARTER",
                        help="End quarter for range (requires --from)")
    parser.add_argument("-n", "--num-quarters", type=int, default=8, metavar="N",
                        help="Number of quarters back from current (default: 8)")
    parser.add_argument("--force", action="store_true",
                        help="Overwrite existing data (default: skip existing)")
    parser.add_argument("--fresh", action="store_true",
                        help="Clear checkpoint and start fresh")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be done without making changes")
    parser.add_argument("-y", "--yes", action="store_true",
                        help="Skip confirmation prompt")

    args = parser.parse_args()

    if (args.from_quarter is None) != (args.to_quarter is None):
        parser.error("--from and --to must be used together")

    try:
        if args.quarters:
            quarters = []
            for q in args.quarters:
                year, qtr = parse_quarter(q)
                quarters.append((q.upper(), quarter_to_end_date(year, qtr)))
            quarters.sort(key=lambda x: x[1], reverse=True)

        elif args.from_quarter and args.to_quarter:
            quarters = generate_quarter_range(args.from_quarter, args.to_quarter)
            quarters.reverse()

        elif args.year:
            quarters = generate_year_quarters(args.year)
            quarters.reverse()

        else:
            quarters = get_quarter_end_dates(args.num_quarters)

    except ValueError as e:
        console.print(f"[red]Error: {e}[/red]")
        sys.exit(1)

    asyncio.run(main(
        quarters=quarters,
        force_update=args.force,
        skip_confirm=args.yes,
        dry_run=args.dry_run,
        fresh=args.fresh
    ))
