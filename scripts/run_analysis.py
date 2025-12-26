#!/usr/bin/env python3
"""Run all analysis systems on stored data.

Usage:
    python scripts/run_analysis.py                    # Run all analyzers for current quarter
    python scripts/run_analysis.py --quarter 2024Q3   # Run for specific quarter
    python scripts/run_analysis.py --all-quarters     # Run for all quarters with data
    python scripts/run_analysis.py --start roic       # Start from ROIC analyzer
    python scripts/run_analysis.py --only graham      # Run only Graham analyzers
    python scripts/run_analysis.py --list             # List available analyzers
"""

import argparse
import logging
import signal
import sys
from datetime import datetime, timezone
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


def get_available_quarters() -> list[str]:
    """Get list of quarters that have data in company_profiles."""
    db = get_db_manager()
    with db.get_connection() as conn:
        result = conn.execute(
            """
            SELECT DISTINCT fiscal_quarter
            FROM company_profiles
            WHERE fiscal_quarter IS NOT NULL
            ORDER BY fiscal_quarter
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

        analyzers_to_run.append((cls(**kwargs), name))

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
    if args.all_quarters:
        quarters = get_available_quarters()
        if not quarters:
            console.print("[red]No quarters found in company_profiles. Run historical_load.py first.[/red]")
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
