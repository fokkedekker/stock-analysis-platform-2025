#!/usr/bin/env python3
"""Load historical SPY prices for benchmark calculations.

Hardcoded quarterly SPY prices from Yahoo Finance historical data.
"""

import logging
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from rich.console import Console
from rich.table import Table

from src.database.connection import get_db_manager
from src.database.schema import create_all_tables

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
console = Console()

# SPY quarterly closing prices from Yahoo Finance
# Using month-end close prices for each quarter
SPY_QUARTERLY_PRICES = {
    # 2020 (approximate from historical data)
    "2020Q1": 257.62,   # Mar 31, 2020
    "2020Q2": 306.18,   # Jun 30, 2020
    "2020Q3": 334.06,   # Sep 30, 2020
    "2020Q4": 373.88,   # Dec 31, 2020
    # 2021 (from Yahoo Finance monthly data)
    "2021Q1": 396.33,   # Mar 2021
    "2021Q2": 428.06,   # Jun 2021
    "2021Q3": 429.14,   # Sep 2021
    "2021Q4": 474.96,   # Dec 2021
    # 2022
    "2022Q1": 451.64,   # Mar 2022
    "2022Q2": 377.25,   # Jun 2022
    "2022Q3": 357.18,   # Sep 2022
    "2022Q4": 382.43,   # Dec 2022
    # 2023
    "2023Q1": 409.39,   # Mar 2023
    "2023Q2": 443.28,   # Jun 2023
    "2023Q3": 427.48,   # Sep 2023
    "2023Q4": 475.31,   # Dec 2023
    # 2024
    "2024Q1": 523.07,   # Mar 2024
    "2024Q2": 544.22,   # Jun 2024
    "2024Q3": 573.76,   # Sep 2024
    "2024Q4": 586.08,   # Dec 2024
    # 2025
    "2025Q1": 559.39,   # Mar 2025
    "2025Q2": 617.85,   # Jun 2025
    "2025Q3": 666.18,   # Sep 2025
    "2025Q4": 690.31,   # Dec 2025
}


def main():
    """Load SPY prices into database."""
    console.print("[bold blue]SPY Benchmark Price Loader[/bold blue]")
    console.print()

    # Ensure database tables exist
    create_all_tables()

    db = get_db_manager()
    with db.get_connection() as conn:
        # Ensure table exists
        conn.execute("""
            CREATE TABLE IF NOT EXISTS spy_prices (
                quarter VARCHAR PRIMARY KEY,
                price DECIMAL NOT NULL,
                price_date DATE,
                fetched_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Insert prices
        for quarter, price in SPY_QUARTERLY_PRICES.items():
            conn.execute(
                """
                INSERT OR REPLACE INTO spy_prices (quarter, price, fetched_at)
                VALUES (?, ?, CURRENT_TIMESTAMP)
                """,
                (quarter, price),
            )

    console.print(f"[green]Saved {len(SPY_QUARTERLY_PRICES)} SPY quarter prices to database[/green]")

    # Show summary table
    table = Table(title="SPY Quarterly Prices")
    table.add_column("Quarter", style="cyan")
    table.add_column("Price", justify="right", style="green")
    table.add_column("QoQ Return", justify="right")

    sorted_quarters = sorted(SPY_QUARTERLY_PRICES.keys())
    prev_price = None
    for quarter in sorted_quarters:
        price = SPY_QUARTERLY_PRICES[quarter]
        if prev_price:
            ret = ((price - prev_price) / prev_price) * 100
            ret_str = f"[{'green' if ret >= 0 else 'red'}]{ret:+.1f}%[/]"
        else:
            ret_str = "—"
        table.add_row(quarter, f"${price:.2f}", ret_str)
        prev_price = price

    console.print(table)


if __name__ == "__main__":
    main()
