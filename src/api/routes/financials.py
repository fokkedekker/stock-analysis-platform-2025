"""Financial data API routes."""

from fastapi import APIRouter, HTTPException, Query

from src.database.connection import get_db_manager
from src.scrapers.fmp_client import FMPClient

router = APIRouter()


def _get_financial_data(table: str, symbol: str, period: str, limit: int) -> list[dict]:
    """Helper to fetch financial data from a table."""
    db = get_db_manager()

    with db.get_connection() as conn:
        result = conn.execute(
            f"""
            SELECT * FROM {table}
            WHERE UPPER(symbol) = UPPER(?)
            AND period = ?
            ORDER BY fiscal_date DESC
            LIMIT ?
            """,
            (symbol, period, limit),
        ).fetchall()

        if not result:
            return []

        columns = [desc[0] for desc in conn.description]
        return [dict(zip(columns, row)) for row in result]


@router.get("/{symbol}/profile")
async def get_profile(symbol: str):
    """Get company profile for a symbol."""
    db = get_db_manager()

    with db.get_connection() as conn:
        result = conn.execute(
            """
            SELECT * FROM company_profiles
            WHERE UPPER(symbol) = UPPER(?)
            ORDER BY fiscal_quarter DESC
            LIMIT 1
            """,
            (symbol,),
        ).fetchone()

        if not result:
            raise HTTPException(status_code=404, detail=f"Profile not found for {symbol}")

        columns = [desc[0] for desc in conn.description]
        return dict(zip(columns, result))


@router.get("/{symbol}/income-statement")
async def get_income_statements(
    symbol: str,
    period: str = Query("annual", regex="^(annual|quarter)$"),
    limit: int = Query(10, ge=1, le=50),
):
    """Get income statements for a symbol."""
    data = _get_financial_data("income_statements", symbol, period, limit)
    if not data:
        raise HTTPException(status_code=404, detail=f"No income statements found for {symbol}")
    return {"symbol": symbol, "period": period, "statements": data}


@router.get("/{symbol}/balance-sheet")
async def get_balance_sheets(
    symbol: str,
    period: str = Query("annual", regex="^(annual|quarter)$"),
    limit: int = Query(10, ge=1, le=50),
):
    """Get balance sheets for a symbol."""
    data = _get_financial_data("balance_sheets", symbol, period, limit)
    if not data:
        raise HTTPException(status_code=404, detail=f"No balance sheets found for {symbol}")
    return {"symbol": symbol, "period": period, "statements": data}


@router.get("/{symbol}/cash-flow")
async def get_cash_flow_statements(
    symbol: str,
    period: str = Query("annual", regex="^(annual|quarter)$"),
    limit: int = Query(10, ge=1, le=50),
):
    """Get cash flow statements for a symbol."""
    data = _get_financial_data("cash_flow_statements", symbol, period, limit)
    if not data:
        raise HTTPException(status_code=404, detail=f"No cash flow statements found for {symbol}")
    return {"symbol": symbol, "period": period, "statements": data}


@router.get("/{symbol}/metrics")
async def get_key_metrics(
    symbol: str,
    period: str = Query("annual", regex="^(annual|quarter)$"),
    limit: int = Query(10, ge=1, le=50),
):
    """Get key metrics for a symbol."""
    data = _get_financial_data("key_metrics", symbol, period, limit)
    if not data:
        raise HTTPException(status_code=404, detail=f"No key metrics found for {symbol}")
    return {"symbol": symbol, "period": period, "metrics": data}


@router.get("/{symbol}/dividends")
async def get_dividends(
    symbol: str,
    limit: int = Query(50, ge=1, le=200),
):
    """Get dividend history for a symbol."""
    db = get_db_manager()

    with db.get_connection() as conn:
        result = conn.execute(
            """
            SELECT * FROM dividends
            WHERE UPPER(symbol) = UPPER(?)
            ORDER BY ex_date DESC
            LIMIT ?
            """,
            (symbol, limit),
        ).fetchall()

        if not result:
            return {"symbol": symbol, "dividends": []}

        columns = [desc[0] for desc in conn.description]
        dividends = [dict(zip(columns, row)) for row in result]

    return {"symbol": symbol, "dividends": dividends}


@router.get("/{symbol}/historical-prices")
async def get_historical_prices(
    symbol: str,
    from_date: str | None = Query(None, alias="from", description="Start date YYYY-MM-DD"),
    to_date: str | None = Query(None, alias="to", description="End date YYYY-MM-DD"),
):
    """Get historical daily prices for a symbol from FMP API.

    Returns daily OHLCV data sorted by date ascending (oldest first).
    """
    async with FMPClient() as client:
        prices = await client.get_historical_prices(symbol, from_date, to_date)

    if not prices:
        raise HTTPException(status_code=404, detail=f"No historical prices found for {symbol}")

    # Sort by date ascending (oldest first) for charting
    prices_sorted = sorted(prices, key=lambda x: x.get("date", ""))

    return {"symbol": symbol.upper(), "prices": prices_sorted}
