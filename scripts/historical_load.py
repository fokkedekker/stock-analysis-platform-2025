#!/usr/bin/env python3
"""Historical price data load - extends DataFetcher pattern for historical prices."""

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
from src.scrapers.data_fetcher import CheckpointManager

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


@dataclass
class FetchTask:
    symbol: str


class HistoricalPriceFetcher:
    """Fetches historical prices using the same pattern as DataFetcher."""

    def __init__(self, fmp_client: FMPClient, quarters: list[tuple[str, date]], checkpoint_path: str = "data/historical_checkpoint.json"):
        self.client = fmp_client
        self.db = get_db_manager()
        self.quarters = quarters
        self.checkpoint = CheckpointManager(checkpoint_path)

        # Calculate date range
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
        for p in prices:  # prices are sorted descending
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
                      pe_ratio: float | None, pb_ratio: float | None, shares: int | None, conn) -> None:
        """Save to company_profiles."""
        conn.execute("""
            INSERT INTO company_profiles (symbol, fiscal_quarter, price, market_cap, pe_ratio, pb_ratio, shares_outstanding, fetched_at, raw_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT (symbol, fiscal_quarter) DO UPDATE SET
                price = EXCLUDED.price, market_cap = EXCLUDED.market_cap, pe_ratio = EXCLUDED.pe_ratio,
                pb_ratio = EXCLUDED.pb_ratio, shares_outstanding = EXCLUDED.shares_outstanding, fetched_at = EXCLUDED.fetched_at
        """, (symbol, quarter, price, market_cap, pe_ratio, pb_ratio, shares,
              datetime.now(timezone.utc), json.dumps({"source": "historical_load"})))

    async def _worker(self, queue: asyncio.Queue, stats: dict, stats_lock: asyncio.Lock,
                      progress: Progress | None, task_id: int | None, worker_id: int) -> None:
        """Worker - matches DataFetcher pattern."""
        db_conn = self.db.connect()
        try:
            while True:
                try:
                    task: FetchTask = queue.get_nowait()
                except asyncio.QueueEmpty:
                    break

                symbol = task.symbol
                try:
                    # Check if already done
                    if self.checkpoint.is_endpoint_completed(symbol, "historical_prices"):
                        async with stats_lock:
                            stats["skipped"] += 1
                        continue

                    # Fetch prices
                    prices = await self.client.get_historical_prices(symbol, self.from_date, self.to_date)

                    if not prices:
                        await self.checkpoint.mark_endpoint_completed(symbol, "historical_prices")
                        async with stats_lock:
                            stats["skipped"] += 1
                        continue

                    # Process each quarter
                    for quarter_str, quarter_end in self.quarters:
                        price_rec = self._find_price_for_date(prices, quarter_end)
                        if not price_rec:
                            continue

                        price = price_rec.get("adjClose") or price_rec.get("close")
                        if not price:
                            continue
                        price = float(price)

                        fin = self._get_financial_data(symbol, quarter_end, db_conn)

                        # Don't calculate market_cap - it overwrites correct API data
                        # Don't use shares_outstanding - balance sheet has wrong data
                        pe = price / fin["eps"] if fin["eps"] and fin["eps"] > 0 else None
                        pb = price / fin["book_value_per_share"] if fin["book_value_per_share"] and fin["book_value_per_share"] > 0 else None

                        self._save_profile(symbol, quarter_str, price, None, pe, pb, None, db_conn)

                    await self.checkpoint.mark_endpoint_completed(symbol, "historical_prices")
                    async with stats_lock:
                        stats["completed"] += 1

                except Exception as e:
                    logger.error(f"Worker {worker_id}: {symbol} failed: {e}")
                    await self.checkpoint.mark_endpoint_failed(symbol, "historical_prices", str(e))
                    async with stats_lock:
                        stats["failed"] += 1

                finally:
                    queue.task_done()
                    if progress and task_id is not None:
                        progress.update(task_id, advance=1)
        finally:
            db_conn.close()

    async def fetch_all(self, symbols: list[str], num_workers: int = 10) -> dict:
        """Fetch historical prices for all symbols."""
        settings = get_settings()

        self.checkpoint.start_session()

        queue: asyncio.Queue[FetchTask] = asyncio.Queue()
        total = 0
        skipped = 0

        for sym in symbols:
            if self.checkpoint.is_endpoint_completed(sym, "historical_prices"):
                skipped += 1
            else:
                await queue.put(FetchTask(sym))
                total += 1

        stats = {"completed": 0, "failed": 0, "skipped": skipped}
        stats_lock = asyncio.Lock()

        if total == 0:
            console.print("[green]All symbols already completed![/green]")
            return stats

        console.print(f"[cyan]Fetching {total:,} symbols with {num_workers} workers[/cyan]")
        console.print(f"[dim]Skipped {skipped:,} already-completed[/dim]")
        console.print(f"[dim]Date range: {self.from_date} to {self.to_date}[/dim]")

        with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"),
                      BarColumn(), TaskProgressColumn(), TimeElapsedColumn()) as progress:
            task_id = progress.add_task(f"Fetching ({num_workers} workers)...", total=total)
            workers = [asyncio.create_task(self._worker(queue, stats, stats_lock, progress, task_id, i))
                       for i in range(num_workers)]
            await asyncio.gather(*workers)

        self.checkpoint.force_save()

        console.print()
        console.print("[bold green]Done![/bold green]")
        console.print(f"  Completed: {stats['completed']:,}")
        console.print(f"  Skipped: {stats['skipped']:,}")
        console.print(f"  Failed: {stats['failed']:,}")

        return stats


async def main(skip_confirm: bool = False):
    settings = get_settings()

    console.print("[bold blue]Stock Analysis - Historical Price Load[/bold blue]")
    console.print(f"Rate limit: {settings.RATE_LIMIT_PER_MINUTE} req/min")
    console.print()

    quarters = get_quarter_end_dates(8)
    console.print("[yellow]Quarters:[/yellow]", ", ".join(q[0] for q in quarters))
    console.print()

    async with FMPClient() as client:
        fetcher = HistoricalPriceFetcher(client, quarters)
        symbols = fetcher.get_active_symbols()

        console.print(f"Symbols: {len(symbols):,}")
        console.print(f"Est. time: {len(symbols) / settings.RATE_LIMIT_PER_MINUTE:.1f} min")
        console.print()

        if not skip_confirm:
            if not console.input("[bold]Continue? (y/n): [/bold]").lower().startswith("y"):
                console.print("[red]Aborted.[/red]")
                return

        await fetcher.fetch_all(symbols, num_workers=settings.NUM_WORKERS)

        console.print()
        console.print("[bold]Next: run analysis for each quarter[/bold]")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-y", "--yes", action="store_true", help="Skip confirmation")
    args = parser.parse_args()
    asyncio.run(main(skip_confirm=args.yes))
