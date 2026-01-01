"""
Sector Returns Calculator.

Pre-calculates average returns per sector for each quarter/holding period combination.
Used for sector-adjusted alpha calculation in Elastic Net training.
"""

import logging
from datetime import datetime
from statistics import median

from src.database.connection import get_db_manager

logger = logging.getLogger(__name__)


def _add_quarters(quarter: str, n: int) -> str:
    """Add n quarters to a quarter string (e.g., '2024Q1' + 2 = '2024Q3')."""
    year = int(quarter[:4])
    q = int(quarter[5])
    total_quarters = (year * 4 + q - 1) + n
    new_year = total_quarters // 4
    new_q = (total_quarters % 4) + 1
    return f"{new_year}Q{new_q}"


def calculate_sector_returns(
    holding_periods: list[int] | None = None,
    force_recalculate: bool = False,
) -> dict[str, int]:
    """
    Calculate and store sector average returns for all quarters.

    Args:
        holding_periods: List of holding periods to calculate (default: [1, 2, 4])
        force_recalculate: If True, recalculate all. If False, only calculate missing.

    Returns:
        Dict with counts: {"calculated": N, "skipped": M}
    """
    if holding_periods is None:
        holding_periods = [1, 2, 4]

    db = get_db_manager()
    stats = {"calculated": 0, "skipped": 0}

    with db.get_connection() as conn:
        # Get all quarters with price data
        quarters_result = conn.execute("""
            SELECT DISTINCT fiscal_quarter
            FROM company_profiles
            WHERE price IS NOT NULL AND price > 0
            ORDER BY fiscal_quarter
        """).fetchall()
        all_quarters = [r[0] for r in quarters_result]

        if not all_quarters:
            logger.warning("No price data found")
            return stats

        logger.info(f"Found {len(all_quarters)} quarters with price data")

        # Get sectors
        sectors_result = conn.execute("""
            SELECT DISTINCT sector
            FROM tickers
            WHERE sector IS NOT NULL AND sector != ''
        """).fetchall()
        all_sectors = [r[0] for r in sectors_result]
        logger.info(f"Found {len(all_sectors)} sectors")

        # Check existing calculations if not forcing
        existing = set()
        if not force_recalculate:
            existing_result = conn.execute("""
                SELECT sector, quarter, holding_period
                FROM sector_returns
            """).fetchall()
            existing = {(r[0], r[1], r[2]) for r in existing_result}
            logger.info(f"Found {len(existing)} existing calculations")

        # Load all prices into memory for efficiency
        prices_result = conn.execute("""
            SELECT cp.symbol, cp.fiscal_quarter, cp.price, t.sector
            FROM company_profiles cp
            JOIN tickers t ON cp.symbol = t.symbol
            WHERE cp.price IS NOT NULL AND cp.price > 0
            AND t.sector IS NOT NULL AND t.sector != ''
        """).fetchall()

        # Build price lookup: {quarter: {symbol: (price, sector)}}
        price_data: dict[str, dict[str, tuple[float, str]]] = {}
        for symbol, quarter, price, sector in prices_result:
            if quarter not in price_data:
                price_data[quarter] = {}
            price_data[quarter][symbol] = (float(price), sector)

        logger.info(f"Loaded prices for {len(price_data)} quarters")

        # Calculate sector returns for each quarter/holding period
        now = datetime.utcnow()

        for hp in holding_periods:
            logger.info(f"Calculating sector returns for holding period {hp}Q...")

            for buy_quarter in all_quarters:
                sell_quarter = _add_quarters(buy_quarter, hp)

                # Need prices at both buy and sell quarters
                if buy_quarter not in price_data or sell_quarter not in price_data:
                    continue

                buy_prices = price_data[buy_quarter]
                sell_prices = price_data[sell_quarter]

                # Group returns by sector
                sector_returns: dict[str, list[float]] = {s: [] for s in all_sectors}

                for symbol, (buy_price, sector) in buy_prices.items():
                    if symbol not in sell_prices:
                        continue

                    sell_price, _ = sell_prices[symbol]
                    if buy_price <= 0:
                        continue

                    stock_return = ((sell_price - buy_price) / buy_price) * 100

                    # Filter extreme returns (data errors)
                    if stock_return > 300 or stock_return < -90:
                        continue

                    sector_returns[sector].append(stock_return)

                # Store sector averages
                for sector, returns in sector_returns.items():
                    if not returns:
                        continue

                    # Skip if already calculated and not forcing
                    if (sector, buy_quarter, hp) in existing:
                        stats["skipped"] += 1
                        continue

                    avg_return = sum(returns) / len(returns)
                    median_return = median(returns) if returns else None
                    stock_count = len(returns)

                    conn.execute("""
                        INSERT INTO sector_returns (
                            sector, quarter, holding_period,
                            avg_return, median_return, stock_count, calculated_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?)
                        ON CONFLICT (sector, quarter, holding_period) DO UPDATE SET
                            avg_return = EXCLUDED.avg_return,
                            median_return = EXCLUDED.median_return,
                            stock_count = EXCLUDED.stock_count,
                            calculated_at = EXCLUDED.calculated_at
                    """, (sector, buy_quarter, hp, avg_return, median_return, stock_count, now))

                    stats["calculated"] += 1

        logger.info(f"Sector returns calculation complete: {stats}")
        return stats


def get_sector_returns_coverage() -> dict:
    """Get statistics about sector returns coverage."""
    db = get_db_manager()

    with db.get_connection() as conn:
        total = conn.execute("SELECT COUNT(*) FROM sector_returns").fetchone()[0]

        by_hp = conn.execute("""
            SELECT holding_period, COUNT(*) as cnt, COUNT(DISTINCT quarter) as quarters
            FROM sector_returns
            GROUP BY holding_period
            ORDER BY holding_period
        """).fetchall()

        by_sector = conn.execute("""
            SELECT sector, COUNT(*) as cnt
            FROM sector_returns
            GROUP BY sector
            ORDER BY cnt DESC
        """).fetchall()

        latest = conn.execute("""
            SELECT MAX(quarter) as latest_quarter, MAX(calculated_at) as last_calc
            FROM sector_returns
        """).fetchone()

    return {
        "total_records": total,
        "by_holding_period": [(hp, cnt, quarters) for hp, cnt, quarters in by_hp],
        "by_sector": [(s, cnt) for s, cnt in by_sector],
        "latest_quarter": latest[0],
        "last_calculated": latest[1],
    }


def ensure_sector_returns_current(holding_periods: list[int] | None = None) -> bool:
    """
    Check if sector returns are up to date and calculate if needed.

    Returns True if data was recalculated, False if already current.
    """
    if holding_periods is None:
        holding_periods = [1, 2, 4]

    db = get_db_manager()

    with db.get_connection() as conn:
        # Get latest quarter with price data
        latest_price_quarter = conn.execute("""
            SELECT MAX(fiscal_quarter)
            FROM company_profiles
            WHERE price IS NOT NULL AND price > 0
        """).fetchone()[0]

        if not latest_price_quarter:
            return False

        # Check if we have sector returns for this quarter (for all holding periods)
        for hp in holding_periods:
            existing = conn.execute("""
                SELECT COUNT(*)
                FROM sector_returns
                WHERE quarter = ? AND holding_period = ?
            """, (latest_price_quarter, hp)).fetchone()[0]

            if existing == 0:
                logger.info(f"Missing sector returns for {latest_price_quarter} HP={hp}, calculating...")
                calculate_sector_returns(holding_periods=holding_periods)
                return True

    return False
