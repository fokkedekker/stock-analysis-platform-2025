"""Backtest simulation API routes."""

import logging
from decimal import Decimal

import httpx
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from src.config import get_settings
from src.database.connection import get_db_manager

logger = logging.getLogger(__name__)
router = APIRouter()


class BacktestRequest(BaseModel):
    """Request model for backtest simulation."""

    symbols: list[str]
    buy_quarter: str  # e.g., "2024Q1"
    benchmark_return: float | None = None  # Manual override


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
    latest_quarter: str
    quarters_held: int
    stocks: list[StockReturn]
    winners: list[StockReturn]
    losers: list[StockReturn]
    portfolio_return: float  # Equal-weighted average
    benchmark_return: float
    alpha: float
    quarterly_portfolio_returns: list[QuarterlyReturn]
    quarterly_benchmark_returns: list[QuarterlyReturn]


def _get_quarters_after(conn, buy_quarter: str) -> list[str]:
    """Get all quarters after the buy quarter, sorted chronologically."""
    result = conn.execute(
        """
        SELECT DISTINCT fiscal_quarter
        FROM company_profiles
        WHERE fiscal_quarter > ?
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
    conn, symbol: str, buy_quarter: str, subsequent_quarters: list[str]
) -> StockReturn | None:
    """Calculate returns for a single stock across all quarters."""
    # Get all prices from buy quarter onwards
    all_quarters = [buy_quarter] + subsequent_quarters
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

    # Need buy price
    buy_price = prices.get(buy_quarter)
    if buy_price is None or buy_price == 0:
        return None

    # Calculate progressive returns for each subsequent quarter
    quarterly_returns = []
    for q in subsequent_quarters:
        q_price = prices.get(q)
        if q_price is not None:
            return_pct = ((q_price - buy_price) / buy_price) * 100
            quarterly_returns.append(QuarterlyReturn(quarter=q, return_pct=round(return_pct, 2)))

    if not quarterly_returns:
        return None

    # Current price is the last available price
    current_price = prices.get(subsequent_quarters[-1]) if subsequent_quarters else buy_price
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


async def _fetch_sp500_returns(
    buy_quarter: str, subsequent_quarters: list[str]
) -> list[QuarterlyReturn]:
    """Fetch S&P 500 (SPY ETF) returns from FMP API."""
    from datetime import datetime, timedelta

    settings = get_settings()
    if not settings.FMP_API_KEY:
        logger.warning("No FMP_API_KEY, using default benchmark")
        return _create_manual_benchmark(10.0, subsequent_quarters)

    # Map quarters to approximate dates (end of quarter)
    def quarter_to_date(q: str) -> str:
        year = q[:4]
        qnum = int(q[-1])
        month = qnum * 3
        day = "31" if month in [3, 12] else "30"
        return f"{year}-{month:02d}-{day}"

    try:
        all_quarters = [buy_quarter] + subsequent_quarters
        dates = [quarter_to_date(q) for q in all_quarters]

        # Calculate date range for the API - go back a bit before buy date
        from_date = (datetime.strptime(dates[0], "%Y-%m-%d") - timedelta(days=30)).strftime(
            "%Y-%m-%d"
        )
        to_date = (datetime.strptime(dates[-1], "%Y-%m-%d") + timedelta(days=7)).strftime(
            "%Y-%m-%d"
        )

        logger.info(f"Fetching SPY data from {from_date} to {to_date}")

        # Fetch historical price for SPY with date range
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{settings.FMP_BASE_URL}/api/v3/historical-price-full/SPY",
                params={
                    "apikey": settings.FMP_API_KEY,
                    "from": from_date,
                    "to": to_date,
                },
                timeout=30.0,
            )
            response.raise_for_status()
            data = response.json()

        if "historical" not in data or not data["historical"]:
            logger.warning(f"No historical data for SPY in response: {data.keys()}")
            return _create_manual_benchmark(10.0, subsequent_quarters)

        logger.info(f"Got {len(data['historical'])} SPY price records")

        # Build date -> price mapping
        historical = {h["date"]: h["close"] for h in data["historical"]}

        # Find closest price for each date
        def find_closest_price(target_date: str) -> float | None:
            # Try exact date first
            if target_date in historical:
                return historical[target_date]
            # Try nearby dates (within 14 days - weekends, holidays)
            target = datetime.strptime(target_date, "%Y-%m-%d")
            for delta in range(-14, 15):
                check_date = (target + timedelta(days=delta)).strftime("%Y-%m-%d")
                if check_date in historical:
                    return historical[check_date]
            logger.warning(f"Could not find SPY price near {target_date}")
            return None

        # Get buy price
        buy_price = find_closest_price(dates[0])
        if not buy_price:
            logger.warning(f"Could not find SPY price for buy quarter {buy_quarter}")
            return _create_manual_benchmark(10.0, subsequent_quarters)

        logger.info(f"SPY buy price for {buy_quarter}: ${buy_price:.2f}")

        # Calculate returns for each subsequent quarter
        benchmark_returns = []
        for i, q in enumerate(subsequent_quarters):
            q_price = find_closest_price(dates[i + 1])
            if q_price:
                return_pct = ((q_price - buy_price) / buy_price) * 100
                benchmark_returns.append(QuarterlyReturn(quarter=q, return_pct=round(return_pct, 2)))
                logger.info(f"SPY return for {q}: {return_pct:.2f}% (${q_price:.2f})")

        if not benchmark_returns:
            logger.warning("No benchmark returns calculated, using default")
            return _create_manual_benchmark(10.0, subsequent_quarters)

        return benchmark_returns

    except Exception as e:
        logger.error(f"Error fetching SPY data: {e}")
        return _create_manual_benchmark(10.0, subsequent_quarters)


@router.post("/simulate", response_model=BacktestResult)
async def simulate_buy(request: BacktestRequest) -> BacktestResult:
    """
    Simulate buying stocks at a historical quarter and calculate returns.

    Returns individual stock performance, portfolio aggregate, and S&P 500 comparison.
    """
    if not request.symbols:
        raise HTTPException(status_code=400, detail="No symbols provided")

    db = get_db_manager()

    with db.get_connection() as conn:
        # 1. Get all quarters after buy_quarter
        quarters = _get_quarters_after(conn, request.buy_quarter)

        if not quarters:
            raise HTTPException(
                status_code=400, detail=f"No quarters found after {request.buy_quarter}"
            )

        # 2. Calculate returns for each stock
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

        # 3. Calculate portfolio returns (equal-weighted)
        portfolio_returns = _calculate_portfolio_returns(stock_returns)

        # 4. Get benchmark returns (S&P 500)
        if request.benchmark_return is not None:
            benchmark_returns = _create_manual_benchmark(request.benchmark_return, quarters)
        else:
            benchmark_returns = await _fetch_sp500_returns(request.buy_quarter, quarters)

        # 5. Split winners/losers
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
            latest_quarter=quarters[-1] if quarters else request.buy_quarter,
            quarters_held=len(quarters),
            stocks=stock_returns,
            winners=winners,
            losers=losers,
            portfolio_return=final_portfolio_return,
            benchmark_return=final_benchmark_return,
            alpha=round(final_portfolio_return - final_benchmark_return, 2),
            quarterly_portfolio_returns=portfolio_returns,
            quarterly_benchmark_returns=benchmark_returns,
        )
