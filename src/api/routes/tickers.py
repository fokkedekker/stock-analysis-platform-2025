"""Ticker-related API routes."""

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from src.database.connection import get_db_manager

router = APIRouter()


class TickerResponse(BaseModel):
    """Response model for ticker data."""

    symbol: str
    name: str | None
    exchange: str | None
    sector: str | None
    industry: str | None
    is_active: bool


class TickerListResponse(BaseModel):
    """Response model for ticker list."""

    total: int
    page: int
    page_size: int
    tickers: list[TickerResponse]


@router.get("", response_model=TickerListResponse)
async def list_tickers(
    page: int = Query(1, ge=1),
    page_size: int = Query(100, ge=1, le=1000),
    exchange: str | None = None,
    sector: str | None = None,
):
    """List all tickers with pagination and filtering."""
    db = get_db_manager()
    offset = (page - 1) * page_size

    with db.get_connection() as conn:
        # Build query
        query = "SELECT * FROM tickers WHERE is_active = TRUE"
        params = []

        if exchange:
            query += " AND UPPER(exchange) = UPPER(?)"
            params.append(exchange)

        if sector:
            query += " AND UPPER(sector) = UPPER(?)"
            params.append(sector)

        # Get total count
        count_query = query.replace("SELECT *", "SELECT COUNT(*)")
        total = conn.execute(count_query, params).fetchone()[0]

        # Get page
        query += " ORDER BY symbol LIMIT ? OFFSET ?"
        params.extend([page_size, offset])

        result = conn.execute(query, params).fetchall()
        columns = [desc[0] for desc in conn.description]

        tickers = [TickerResponse(**dict(zip(columns, row))) for row in result]

    return TickerListResponse(
        total=total,
        page=page,
        page_size=page_size,
        tickers=tickers,
    )


@router.get("/search")
async def search_tickers(
    q: str = Query(..., min_length=1, description="Search query"),
    limit: int = Query(20, ge=1, le=100),
):
    """Search tickers by symbol or name."""
    db = get_db_manager()

    with db.get_connection() as conn:
        result = conn.execute(
            """
            SELECT symbol, name, exchange, sector
            FROM tickers
            WHERE is_active = TRUE
            AND (UPPER(symbol) LIKE UPPER(?) OR UPPER(name) LIKE UPPER(?))
            ORDER BY
                CASE WHEN UPPER(symbol) = UPPER(?) THEN 0
                     WHEN UPPER(symbol) LIKE UPPER(?) THEN 1
                     ELSE 2 END,
                symbol
            LIMIT ?
            """,
            (f"%{q}%", f"%{q}%", q, f"{q}%", limit),
        ).fetchall()

        return [
            {"symbol": row[0], "name": row[1], "exchange": row[2], "sector": row[3]}
            for row in result
        ]


@router.get("/{symbol}", response_model=TickerResponse)
async def get_ticker(symbol: str):
    """Get ticker details by symbol."""
    db = get_db_manager()

    with db.get_connection() as conn:
        result = conn.execute(
            "SELECT * FROM tickers WHERE UPPER(symbol) = UPPER(?)", (symbol,)
        ).fetchone()

        if not result:
            raise HTTPException(status_code=404, detail=f"Ticker {symbol} not found")

        columns = [desc[0] for desc in conn.description]
        return TickerResponse(**dict(zip(columns, result)))


@router.get("/exchanges/list")
async def list_exchanges():
    """Get list of all exchanges in the database."""
    db = get_db_manager()

    with db.get_connection() as conn:
        result = conn.execute(
            """
            SELECT exchange, COUNT(*) as count
            FROM tickers
            WHERE is_active = TRUE AND exchange IS NOT NULL
            GROUP BY exchange
            ORDER BY count DESC
            """
        ).fetchall()

        return [{"exchange": row[0], "count": row[1]} for row in result]


@router.get("/sectors/list")
async def list_sectors():
    """Get list of all sectors in the database."""
    db = get_db_manager()

    with db.get_connection() as conn:
        result = conn.execute(
            """
            SELECT sector, COUNT(*) as count
            FROM tickers
            WHERE is_active = TRUE AND sector IS NOT NULL
            GROUP BY sector
            ORDER BY count DESC
            """
        ).fetchall()

        return [{"sector": row[0], "count": row[1]} for row in result]
