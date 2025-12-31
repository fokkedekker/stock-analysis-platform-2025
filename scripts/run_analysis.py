#!/usr/bin/env python3
"""Run all analysis systems on stored data.

Usage:
    python scripts/run_analysis.py                              # Run all analyzers for current quarter
    python scripts/run_analysis.py --quarter 2024Q3             # Run for specific quarter
    python scripts/run_analysis.py --all-quarters               # Run for all quarters with data
    python scripts/run_analysis.py --all-quarters --parallel-workers 4  # Run quarters in parallel
    python scripts/run_analysis.py --start roic                 # Start from ROIC analyzer
    python scripts/run_analysis.py --only graham                # Run only Graham analyzers
    python scripts/run_analysis.py --list                       # List available analyzers
"""

import argparse
import concurrent.futures
import logging
import signal
import sys
from datetime import datetime, timezone
from multiprocessing import cpu_count
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.logging import RichHandler

# Global flag for graceful shutdown
_shutdown_requested = False

def signal_handler(signum, frame):
    """Handle interrupt signals gracefully."""
    global _shutdown_requested
    _shutdown_requested = True
    print("\n\nShutdown requested, finishing current operation...")

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

from src.database.connection import get_db_manager
from src.analysis.base import quarter_to_end_date
from src.analysis.bulk_loader import BulkDataLoader
from src.analysis.graham import GrahamAnalyzer
from src.analysis.magic_formula import MagicFormulaAnalyzer
from src.analysis.piotroski import PiotroskiAnalyzer
from src.analysis.altman import AltmanAnalyzer
from src.analysis.roic_quality import ROICQualityAnalyzer
from src.analysis.garp_peg import GARPPEGAnalyzer
from src.analysis.fama_french import FamaFrenchAnalyzer
from src.analysis.net_net import NetNetAnalyzer

logging.basicConfig(
    level=logging.DEBUG,
    format="%(message)s",
    handlers=[RichHandler(rich_tracebacks=True)],
)
logger = logging.getLogger(__name__)
console = Console()


def get_current_quarter() -> str:
    """Get current fiscal quarter string."""
    now = datetime.now(timezone.utc)
    quarter = (now.month - 1) // 3 + 1
    return f"{now.year}Q{quarter}"


def get_symbols() -> list[str]:
    """Get list of active symbols from database."""
    db = get_db_manager()
    with db.get_connection() as conn:
        result = conn.execute(
            "SELECT symbol FROM tickers WHERE is_active = TRUE ORDER BY symbol"
        ).fetchall()
        return [row[0] for row in result]


def generate_quarter_range(from_q: str, to_q: str) -> list[str]:
    """Generate list of quarters from start to end (inclusive), most recent first."""
    import re

    def parse_quarter(q: str) -> tuple[int, int]:
        match = re.match(r'^(\d{4})Q([1-4])$', q.upper())
        if not match:
            raise ValueError(f"Invalid quarter format: {q}")
        return int(match.group(1)), int(match.group(2))

    from_year, from_qtr = parse_quarter(from_q)
    to_year, to_qtr = parse_quarter(to_q)

    quarters = []
    year, qtr = from_year, from_qtr

    while (year, qtr) <= (to_year, to_qtr):
        quarters.append(f"{year}Q{qtr}")
        qtr += 1
        if qtr > 4:
            qtr = 1
            year += 1

    return list(reversed(quarters))  # Most recent first


def get_available_quarters() -> list[str]:
    """Get list of quarters that have financial data (from statements, not profiles).

    This derives quarters from actual fiscal dates in income statements,
    which have the correct fiscal period dates from FMP API.
    """
    db = get_db_manager()
    with db.get_connection() as conn:
        result = conn.execute(
            """
            SELECT DISTINCT
                CAST(EXTRACT(YEAR FROM fiscal_date) AS INTEGER) || 'Q' || CASE
                    WHEN EXTRACT(MONTH FROM fiscal_date) <= 3 THEN '1'
                    WHEN EXTRACT(MONTH FROM fiscal_date) <= 6 THEN '2'
                    WHEN EXTRACT(MONTH FROM fiscal_date) <= 9 THEN '3'
                    ELSE '4'
                END as quarter
            FROM income_statements
            WHERE period = 'quarter'
            AND fiscal_date IS NOT NULL
            ORDER BY quarter DESC
            """
        ).fetchall()
        return [row[0] for row in result]


def run_analyzer(analyzer, symbols: list[str], quarter: str, name: str):
    """Run a single analyzer on all symbols."""
    global _shutdown_requested
    success = 0
    failed = 0

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
    ) as progress:
        task = progress.add_task(f"Running {name}...", total=len(symbols))

        # Use connection_scope to reuse a single DB connection for all symbols
        with analyzer.connection_scope():
            for symbol in symbols:
                if _shutdown_requested:
                    console.print("\n[yellow]Interrupted![/yellow]")
                    break

                try:
                    analyzer.analyze_and_save(symbol, quarter)
                    success += 1
                except Exception as e:
                    logger.debug(f"Failed {name} for {symbol}: {e}")
                    failed += 1

                progress.update(task, advance=1)

    console.print(f"  [green]{success} succeeded[/green], [red]{failed} failed[/red]")
    return success, failed, _shutdown_requested


ANALYZERS = [
    ("graham-strict", GrahamAnalyzer, {"mode": "strict"}, "Graham (strict)"),
    ("graham-modern", GrahamAnalyzer, {"mode": "modern"}, "Graham (modern)"),
    ("magic-formula", MagicFormulaAnalyzer, {}, "Magic Formula"),
    ("piotroski", PiotroskiAnalyzer, {}, "Piotroski F-Score"),
    ("altman", AltmanAnalyzer, {}, "Altman Z-Score"),
    ("roic", ROICQualityAnalyzer, {}, "ROIC/Quality"),
    ("garp", GARPPEGAnalyzer, {}, "GARP/PEG"),
    ("fama-french", FamaFrenchAnalyzer, {}, "Fama-French"),
    ("net-net", NetNetAnalyzer, {}, "Net-Net"),
]


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run stock analysis systems.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/run_analysis.py                    # Run all analyzers for current quarter
  python scripts/run_analysis.py --quarter 2024Q3   # Run for specific quarter
  python scripts/run_analysis.py --all-quarters     # Run for all quarters with data
  python scripts/run_analysis.py --start roic       # Start from ROIC analyzer
  python scripts/run_analysis.py --only graham      # Run only Graham analyzers
  python scripts/run_analysis.py --only roic garp   # Run only ROIC and GARP
  python scripts/run_analysis.py --list             # List available analyzers
  python scripts/run_analysis.py --skip-rankings    # Skip post-processing rankings
        """,
    )
    parser.add_argument(
        "--quarter",
        type=str,
        metavar="QUARTER",
        help="Quarter to analyze (e.g., 2024Q3). Default: current quarter",
    )
    parser.add_argument(
        "--quarters",
        type=str,
        nargs="+",
        metavar="QUARTER",
        help="List of quarters to analyze (e.g., 2010Q1 2010Q2 2010Q3)",
    )
    parser.add_argument(
        "--from",
        dest="from_quarter",
        type=str,
        metavar="QUARTER",
        help="Start quarter for range (e.g., 2010Q1). Use with --to",
    )
    parser.add_argument(
        "--to",
        dest="to_quarter",
        type=str,
        metavar="QUARTER",
        help="End quarter for range (e.g., 2025Q4). Use with --from",
    )
    parser.add_argument(
        "--all-quarters",
        action="store_true",
        help="Run analysis for all quarters that have data in company_profiles",
    )
    parser.add_argument(
        "--start",
        type=str,
        metavar="ANALYZER",
        help="Start from this analyzer (skips preceding ones)",
    )
    parser.add_argument(
        "--only",
        type=str,
        nargs="+",
        metavar="ANALYZER",
        help="Run only these analyzers",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available analyzers and exit",
    )
    parser.add_argument(
        "--skip-rankings",
        action="store_true",
        help="Skip computing rankings/percentiles at the end",
    )
    parser.add_argument(
        "--parallel-workers",
        type=int,
        metavar="N",
        help=f"Analyze N symbols in parallel using threads (default: sequential, recommended: 4-8, max: {cpu_count()})",
    )
    return parser.parse_args()


def list_analyzers():
    """Print list of available analyzers."""
    console.print("[bold]Available analyzers:[/bold]\n")
    for key, _, _, name in ANALYZERS:
        console.print(f"  {key:15} - {name}")
    console.print("\nUse partial names to match multiple: 'graham' matches both graham-strict and graham-modern")


def run_analysis_for_quarter(quarter: str, args, symbols: list[str]) -> tuple[int, int, bool]:
    """Run all selected analyzers for a single quarter.

    Args:
        quarter: Quarter string (e.g., "2024Q3").
        args: Parsed command line arguments.
        symbols: List of symbols to analyze.

    Returns:
        Tuple of (success_count, failed_count, was_interrupted).
    """
    global _shutdown_requested

    console.print(f"\n[bold cyan]═══ Analyzing quarter: {quarter} ═══[/bold cyan]")

    # Load all data upfront for high-performance analysis
    as_of_date = quarter_to_end_date(quarter)
    console.print(f"[dim]Loading data (as_of_date={as_of_date})...[/dim]")
    import time
    start_time = time.time()
    bulk_loader = BulkDataLoader(period="annual", as_of_date=as_of_date)
    load_time = time.time() - start_time
    console.print(f"[dim]Data loaded in {load_time:.1f}s[/dim]")

    # Filter analyzers based on args
    analyzers_to_run = []
    started = args.start is None  # If no --start, start immediately

    for key, cls, kwargs, name in ANALYZERS:
        # Check if we should start from this analyzer
        if args.start and args.start.lower() in key.lower():
            started = True

        if not started:
            continue

        # Check if this analyzer matches --only filter
        if args.only:
            matches = any(only.lower() in key.lower() for only in args.only)
            if not matches:
                continue

        # Pass bulk_loader to analyzer for high-performance mode
        analyzers_to_run.append((cls(bulk_loader=bulk_loader, **kwargs), name))

    if not analyzers_to_run:
        console.print("[red]No analyzers matched the specified criteria.[/red]")
        return 0, 0, False

    total_success = 0
    total_failed = 0

    for analyzer, name in analyzers_to_run:
        if _shutdown_requested:
            return total_success, total_failed, True

        console.print(f"\n[yellow]{name}[/yellow]")
        success, failed, interrupted = run_analyzer(analyzer, symbols, quarter, name)
        total_success += success
        total_failed += failed
        if interrupted:
            return total_success, total_failed, True

    if not args.skip_rankings:
        # Compute rankings for Magic Formula
        console.print("\n[yellow]Computing Magic Formula rankings...[/yellow]")
        mf_analyzer = MagicFormulaAnalyzer()
        mf_analyzer.compute_rankings(quarter)
        console.print("  [green]Done[/green]")

        # Compute Fama-French percentiles
        console.print("\n[yellow]Computing Fama-French percentiles...[/yellow]")
        ff_analyzer = FamaFrenchAnalyzer()
        ff_analyzer.compute_percentiles(quarter)
        console.print("  [green]Done[/green]")

    return total_success, total_failed, False


def run_analyzer_parallel(analyzer_cls, analyzer_kwargs: dict, symbols: list[str], quarter: str, name: str, num_workers: int):
    """Run analyzer on symbols in parallel using thread pool.

    DuckDB supports concurrent access from multiple threads in the same process,
    but NOT from multiple processes. So we use ThreadPoolExecutor here.
    """
    global _shutdown_requested
    from concurrent.futures import ThreadPoolExecutor, as_completed
    import threading

    success = 0
    failed = 0
    lock = threading.Lock()

    def analyze_one(symbol: str) -> bool:
        if _shutdown_requested:
            return False
        # Each thread creates its own analyzer instance with its own connection
        analyzer = analyzer_cls(**analyzer_kwargs)
        try:
            analyzer.analyze_and_save(symbol, quarter)
            return True
        except Exception:
            return False

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
    ) as progress:
        task = progress.add_task(f"Running {name}...", total=len(symbols))

        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = {executor.submit(analyze_one, s): s for s in symbols}

            for future in as_completed(futures):
                if _shutdown_requested:
                    executor.shutdown(wait=False, cancel_futures=True)
                    break

                if future.result():
                    success += 1
                else:
                    failed += 1
                progress.update(task, advance=1)

    console.print(f"  [green]{success} succeeded[/green], [red]{failed} failed[/red]")
    return success, failed


def main():
    """Run all analyses."""
    args = parse_args()

    if args.list:
        list_analyzers()
        return

    console.print("[bold blue]Stock Analysis - Run All Analyses[/bold blue]")
    console.print()

    symbols = get_symbols()

    if not symbols:
        console.print("[red]No symbols found. Run initial_load.py first.[/red]")
        return

    console.print(f"Symbols to analyze: {len(symbols)}")

    # Determine which quarters to analyze
    if args.quarters:
        quarters = [q.upper() for q in args.quarters]
        console.print(f"Quarters to analyze: {', '.join(quarters[:5])}{'...' if len(quarters) > 5 else ''} ({len(quarters)} quarters)")
    elif args.from_quarter and args.to_quarter:
        quarters = generate_quarter_range(args.from_quarter, args.to_quarter)
        console.print(f"Quarters to analyze: {args.from_quarter} to {args.to_quarter} ({len(quarters)} quarters)")
    elif args.all_quarters:
        quarters = get_available_quarters()
        if not quarters:
            console.print("[red]No quarters found in income_statements. Run initial_load.py first.[/red]")
            return
        console.print(f"Quarters to analyze: {', '.join(quarters)}")
    elif args.quarter:
        quarters = [args.quarter]
        console.print(f"Analysis quarter: {args.quarter}")
    else:
        quarters = [get_current_quarter()]
        console.print(f"Analysis quarter: {quarters[0]} (current)")

    console.print()

    # Filter analyzers based on args (for display purposes)
    analyzers_to_run = []
    started = args.start is None

    for key, cls, kwargs, name in ANALYZERS:
        if args.start and args.start.lower() in key.lower():
            started = True
        if not started:
            console.print(f"[dim]Skipping {name}[/dim]")
            continue
        if args.only:
            matches = any(only.lower() in key.lower() for only in args.only)
            if not matches:
                continue
        analyzers_to_run.append(name)

    if not analyzers_to_run:
        console.print("[red]No analyzers matched the specified criteria.[/red]")
        console.print("Use --list to see available analyzers.")
        return

    grand_total_success = 0
    grand_total_failed = 0

    # Parallel execution mode (parallelize symbols with threads, not quarters with processes)
    # DuckDB only supports multi-thread, not multi-process concurrent access
    if args.parallel_workers:
        num_workers = min(args.parallel_workers, cpu_count())
        console.print(f"[cyan]Running with {num_workers} parallel workers per analyzer[/cyan]")
        console.print("[dim]Press Ctrl+C to cancel[/dim]")

        for quarter in quarters:
            if _shutdown_requested:
                break

            console.print(f"\n[bold cyan]═══ Analyzing quarter: {quarter} ═══[/bold cyan]")

            # Load all data upfront for high-performance analysis
            as_of_date = quarter_to_end_date(quarter)
            console.print(f"[dim]Loading data (as_of_date={as_of_date})...[/dim]")
            import time
            start_time = time.time()
            bulk_loader = BulkDataLoader(period="annual", as_of_date=as_of_date)
            load_time = time.time() - start_time
            console.print(f"[dim]Data loaded in {load_time:.1f}s[/dim]")

            # Filter analyzers
            analyzers_to_run = []
            started = args.start is None

            for key, cls, kwargs, name in ANALYZERS:
                if args.start and args.start.lower() in key.lower():
                    started = True
                if not started:
                    continue
                if args.only:
                    matches = any(only.lower() in key.lower() for only in args.only)
                    if not matches:
                        continue
                analyzers_to_run.append((cls, kwargs, name))

            for cls, kwargs, name in analyzers_to_run:
                if _shutdown_requested:
                    break

                console.print(f"\n[yellow]{name}[/yellow]")
                success, failed = run_analyzer_parallel(cls, {**kwargs, "bulk_loader": bulk_loader}, symbols, quarter, name, num_workers)
                grand_total_success += success
                grand_total_failed += failed

            # Compute rankings
            if not args.skip_rankings and not _shutdown_requested:
                console.print("\n[yellow]Computing Magic Formula rankings...[/yellow]")
                mf_analyzer = MagicFormulaAnalyzer()
                mf_analyzer.compute_rankings(quarter)
                console.print("  [green]Done[/green]")

                console.print("\n[yellow]Computing Fama-French percentiles...[/yellow]")
                ff_analyzer = FamaFrenchAnalyzer()
                ff_analyzer.compute_percentiles(quarter)
                console.print("  [green]Done[/green]")

        if _shutdown_requested:
            console.print("\n[red]Analysis interrupted by user.[/red]")
            console.print(f"Partial results: {grand_total_success} succeeded, {grand_total_failed} failed")
            return
    else:
        # Sequential execution (original behavior)
        for quarter in quarters:
            success, failed, interrupted = run_analysis_for_quarter(quarter, args, symbols)
            grand_total_success += success
            grand_total_failed += failed

            if interrupted:
                console.print("\n[red]Analysis interrupted by user.[/red]")
                console.print(f"Partial results: {grand_total_success} succeeded, {grand_total_failed} failed")
                return

    console.print()
    console.print("[bold green]Analysis complete![/bold green]")
    if len(quarters) > 1:
        console.print(f"Quarters analyzed: {len(quarters)}")
    console.print(f"Total: {grand_total_success} succeeded, {grand_total_failed} failed")


if __name__ == "__main__":
    main()
