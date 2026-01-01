"""Backtest simulation API routes."""

import logging
from decimal import Decimal

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from src.database.connection import get_db_manager

logger = logging.getLogger(__name__)
router = APIRouter()


class BacktestRequest(BaseModel):
    """Request model for backtest simulation."""

    symbols: list[str]
    buy_quarter: str  # e.g., "2024Q1"
    holding_period: int | None = None  # Number of quarters to hold (e.g., 4), None = hold until latest
    benchmark_return: float | None = None  # Manual override


def _add_quarters(quarter: str, num_quarters: int) -> str:
    """Add num_quarters to a quarter string like '2024Q1'."""
    year = int(quarter[:4])
    q = int(quarter[5])

    total_quarters = (year * 4) + q + num_quarters
    new_year = (total_quarters - 1) // 4
    new_q = ((total_quarters - 1) % 4) + 1

    return f"{new_year}Q{new_q}"


def _prev_quarter(quarter: str) -> str:
    """Get the previous quarter. E.g., '2025Q1' -> '2024Q4'."""
    return _add_quarters(quarter, -1)


class QuarterlyReturn(BaseModel):
    """Return for a specific quarter."""

    quarter: str
    return_pct: float


class StockReturn(BaseModel):
    """Return data for a single stock."""

    symbol: str
    name: str
    buy_price: float
    current_price: float
    total_return: float  # percentage
    quarterly_returns: list[QuarterlyReturn]  # Progressive returns from buy date


class BacktestResult(BaseModel):
    """Complete backtest simulation result."""

    buy_quarter: str
    sell_quarter: str  # When to sell (either holding_period end or latest)
    quarters_held: int
    holding_period: int | None  # Requested holding period (None = hold until latest)
    stocks: list[StockReturn]
    winners: list[StockReturn]
    losers: list[StockReturn]
    portfolio_return: float  # Equal-weighted average
    benchmark_return: float
    alpha: float
    quarterly_portfolio_returns: list[QuarterlyReturn]
    quarterly_benchmark_returns: list[QuarterlyReturn]


def _get_quarters_from(conn, buy_quarter: str) -> list[str]:
    """Get all quarters from buy_quarter onwards (inclusive), sorted chronologically.

    When buying 'in Q1 2025', we buy at the START of Q1 (= end of Q4 2024),
    so Q1 2025 is our first quarter of returns.
    """
    result = conn.execute(
        """
        SELECT DISTINCT fiscal_quarter
        FROM company_profiles
        WHERE fiscal_quarter >= ?
        ORDER BY fiscal_quarter ASC
        """,
        (buy_quarter,),
    ).fetchall()
    return [row[0] for row in result]


def _to_float(value) -> float | None:
    """Convert Decimal or other numeric types to float."""
    if value is None:
        return None
    if isinstance(value, Decimal):
        return float(value)
    return float(value)


def _calculate_stock_returns(
    conn, symbol: str, buy_quarter: str, quarters: list[str]
) -> StockReturn | None:
    """Calculate returns for a single stock across all quarters.

    Buy price is from the PREVIOUS quarter (start of buy_quarter = end of prev quarter).
    quarters should include buy_quarter onwards.
    """
    # Get price from previous quarter (buy price) and all quarters for returns
    prev_quarter = _prev_quarter(buy_quarter)
    all_quarters = [prev_quarter] + quarters
    placeholders = ",".join(["?" for _ in all_quarters])

    result = conn.execute(
        f"""
        SELECT cp.fiscal_quarter, cp.price, t.name
        FROM company_profiles cp
        JOIN tickers t ON t.symbol = cp.symbol
        WHERE cp.symbol = ?
          AND cp.fiscal_quarter IN ({placeholders})
        ORDER BY cp.fiscal_quarter ASC
        """,
        (symbol, *all_quarters),
    ).fetchall()

    if not result:
        return None

    # Build price lookup
    prices = {}
    name = None
    for row in result:
        quarter, price, stock_name = row
        prices[quarter] = _to_float(price)
        name = stock_name

    # Buy price is from previous quarter (start of buy_quarter)
    buy_price = prices.get(prev_quarter)
    if buy_price is None or buy_price == 0:
        return None

    # Calculate progressive returns for each quarter (including buy_quarter)
    quarterly_returns = []
    for q in quarters:
        q_price = prices.get(q)
        if q_price is not None:
            return_pct = ((q_price - buy_price) / buy_price) * 100
            quarterly_returns.append(QuarterlyReturn(quarter=q, return_pct=round(return_pct, 2)))

    if not quarterly_returns:
        return None

    # Current price is the last available price
    current_price = prices.get(quarters[-1]) if quarters else buy_price
    total_return = quarterly_returns[-1].return_pct if quarterly_returns else 0

    return StockReturn(
        symbol=symbol,
        name=name or symbol,
        buy_price=round(buy_price, 2),
        current_price=round(current_price, 2) if current_price else 0,
        total_return=total_return,
        quarterly_returns=quarterly_returns,
    )


def _calculate_portfolio_returns(stock_returns: list[StockReturn]) -> list[QuarterlyReturn]:
    """Calculate equal-weighted portfolio returns across quarters."""
    if not stock_returns:
        return []

    # Get all unique quarters
    all_quarters = set()
    for sr in stock_returns:
        for qr in sr.quarterly_returns:
            all_quarters.add(qr.quarter)

    # Sort quarters chronologically
    sorted_quarters = sorted(all_quarters)

    # Calculate equal-weighted average for each quarter
    portfolio_returns = []
    for q in sorted_quarters:
        returns_in_quarter = []
        for sr in stock_returns:
            for qr in sr.quarterly_returns:
                if qr.quarter == q:
                    returns_in_quarter.append(qr.return_pct)
                    break

        if returns_in_quarter:
            avg_return = sum(returns_in_quarter) / len(returns_in_quarter)
            portfolio_returns.append(QuarterlyReturn(quarter=q, return_pct=round(avg_return, 2)))

    return portfolio_returns


def _create_manual_benchmark(
    total_return: float, quarters: list[str]
) -> list[QuarterlyReturn]:
    """Create benchmark returns assuming linear progression to manual total."""
    if not quarters:
        return []

    # Simple linear interpolation
    num_quarters = len(quarters)
    benchmark_returns = []
    for i, q in enumerate(quarters):
        # Progressive return
        progress = (i + 1) / num_quarters
        return_pct = total_return * progress
        benchmark_returns.append(QuarterlyReturn(quarter=q, return_pct=round(return_pct, 2)))

    return benchmark_returns


def _get_sp500_returns_from_db(
    conn, buy_quarter: str, quarters: list[str]
) -> list[QuarterlyReturn]:
    """Get S&P 500 (SPY) returns from the spy_prices table.

    Uses actual quarterly SPY prices synced from yfinance.
    Buy price is from the PREVIOUS quarter (start of buy_quarter = end of prev quarter).
    quarters should include buy_quarter onwards.
    """
    prev_quarter = _prev_quarter(buy_quarter)
    all_quarters = [prev_quarter] + quarters
    placeholders = ",".join(["?" for _ in all_quarters])

    result = conn.execute(
        f"""
        SELECT quarter, price
        FROM spy_prices
        WHERE quarter IN ({placeholders})
        ORDER BY quarter ASC
        """,
        all_quarters,
    ).fetchall()

    if not result:
        logger.warning(f"No SPY prices found in database for quarters: {all_quarters[:3]}...")
        return []

    # Build quarter -> price mapping
    prices = {row[0]: float(row[1]) for row in result}

    # Buy price is from previous quarter (start of buy_quarter)
    buy_price = prices.get(prev_quarter)
    if not buy_price:
        logger.warning(f"No SPY price for previous quarter {prev_quarter}")
        return []

    logger.info(f"SPY buy price for {buy_quarter} (from {prev_quarter}): ${buy_price:.2f}")

    # Calculate returns for each quarter (including buy_quarter)
    benchmark_returns = []
    for q in quarters:
        q_price = prices.get(q)
        if q_price:
            return_pct = ((q_price - buy_price) / buy_price) * 100
            benchmark_returns.append(QuarterlyReturn(quarter=q, return_pct=round(return_pct, 2)))
            logger.info(f"SPY return for {q}: {return_pct:.2f}% (${q_price:.2f})")
        else:
            logger.warning(f"Missing SPY price for quarter {q}")

    return benchmark_returns


@router.post("/simulate", response_model=BacktestResult)
async def simulate_buy(request: BacktestRequest) -> BacktestResult:
    """
    Simulate buying stocks at a historical quarter and calculate returns.

    If holding_period is provided, sells after that many quarters.
    Otherwise, holds until the latest available quarter.

    Returns individual stock performance, portfolio aggregate, and S&P 500 comparison.
    """
    if not request.symbols:
        raise HTTPException(status_code=400, detail="No symbols provided")

    db = get_db_manager()

    with db.get_connection() as conn:
        # 1. Get all quarters from buy_quarter onwards (inclusive)
        # When buying "in Q1 2025", we buy at START of Q1 (= end of Q4 2024)
        # so Q1 2025 is our first quarter of returns
        all_quarters = _get_quarters_from(conn, request.buy_quarter)

        if not all_quarters:
            raise HTTPException(
                status_code=400, detail=f"No quarters found from {request.buy_quarter}"
            )

        # 2. Determine sell quarter based on holding period
        # 4Q hold from Q1 means: Q1, Q2, Q3, Q4 (sell at end of Q4)
        # So sell_quarter = buy_quarter + (holding_period - 1)
        if request.holding_period is not None and request.holding_period > 0:
            sell_quarter = _add_quarters(request.buy_quarter, request.holding_period - 1)
            # Filter quarters to only include up to sell_quarter
            quarters = [q for q in all_quarters if q <= sell_quarter]
            if not quarters:
                # If sell_quarter is in the future, use all available quarters
                quarters = all_quarters
                sell_quarter = quarters[-1]
        else:
            quarters = all_quarters
            sell_quarter = quarters[-1]

        # 3. Calculate returns for each stock
        stock_returns = []
        for symbol in request.symbols:
            returns = _calculate_stock_returns(conn, symbol, request.buy_quarter, quarters)
            if returns:
                stock_returns.append(returns)

        if not stock_returns:
            raise HTTPException(
                status_code=404,
                detail="Could not calculate returns for any of the provided symbols",
            )

        # 4. Calculate portfolio returns (equal-weighted)
        portfolio_returns = _calculate_portfolio_returns(stock_returns)

        # 5. Get benchmark returns (S&P 500) from database
        if request.benchmark_return is not None:
            benchmark_returns = _create_manual_benchmark(request.benchmark_return, quarters)
        else:
            benchmark_returns = _get_sp500_returns_from_db(conn, request.buy_quarter, quarters)
            if not benchmark_returns:
                raise HTTPException(
                    status_code=500,
                    detail=f"No SPY benchmark data available for {request.buy_quarter}. "
                    "Run quarterly_update.py to sync SPY prices.",
                )

        # 6. Split winners/losers
        winners = [s for s in stock_returns if s.total_return > 0]
        losers = [s for s in stock_returns if s.total_return <= 0]

        # Sort by return
        winners = sorted(winners, key=lambda x: x.total_return, reverse=True)
        losers = sorted(losers, key=lambda x: x.total_return)

        # Calculate final returns
        final_portfolio_return = portfolio_returns[-1].return_pct if portfolio_returns else 0
        final_benchmark_return = benchmark_returns[-1].return_pct if benchmark_returns else 0

        return BacktestResult(
            buy_quarter=request.buy_quarter,
            sell_quarter=sell_quarter,
            quarters_held=len(quarters),
            holding_period=request.holding_period,
            stocks=stock_returns,
            winners=winners,
            losers=losers,
            portfolio_return=final_portfolio_return,
            benchmark_return=final_benchmark_return,
            alpha=round(final_portfolio_return - final_benchmark_return, 2),
            quarterly_portfolio_returns=portfolio_returns,
            quarterly_benchmark_returns=benchmark_returns,
        )
