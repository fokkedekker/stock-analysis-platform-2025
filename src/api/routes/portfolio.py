"""Portfolio tracking API routes."""

import json
import logging
from datetime import datetime, date
from decimal import Decimal
from uuid import uuid4

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from src.database.connection import get_db_manager

logger = logging.getLogger(__name__)
router = APIRouter()


# ============================================================================
# Helper Functions
# ============================================================================


def add_quarters(quarter: str, n: int) -> str:
    """Add n quarters to a quarter string. e.g., '2023Q1' + 2 = '2023Q3'."""
    year = int(quarter[:4])
    q = int(quarter[5])
    total_q = (year * 4 + q - 1) + n
    new_year = total_q // 4
    new_q = (total_q % 4) + 1
    return f"{new_year}Q{new_q}"


def quarter_to_date(quarter: str) -> date:
    """Convert quarter string to end-of-quarter date."""
    year = int(quarter[:4])
    q = int(quarter[5])
    if q == 1:
        return date(year, 3, 31)
    elif q == 2:
        return date(year, 6, 30)
    elif q == 3:
        return date(year, 9, 30)
    else:
        return date(year, 12, 31)


def days_until_quarter_end(target_quarter: str) -> int:
    """Calculate days until target quarter end."""
    target_date = quarter_to_date(target_quarter)
    today = date.today()
    return (target_date - today).days


def _to_float(value) -> float | None:
    """Convert Decimal or other numeric types to float."""
    if value is None:
        return None
    if isinstance(value, Decimal):
        return float(value)
    return float(value)


def _generate_id() -> str:
    """Generate a short unique ID."""
    return str(uuid4())[:12]


# ============================================================================
# Request Models
# ============================================================================


class CreateBatchRequest(BaseModel):
    """Request to create a new portfolio batch."""

    name: str | None = None
    buy_quarter: str = Field(..., description="e.g., '2024Q1'")
    strategy_id: str | None = Field(None, description="Saved strategy ID")
    holding_period: int = Field(4, ge=1, le=20, description="Quarters to hold")
    total_invested: float = Field(..., gt=0, description="Total $ to invest")
    symbols: list[str] = Field(..., min_length=1, description="Stocks to buy")
    allocations: dict[str, float] | None = Field(
        None, description="Optional per-symbol allocation. Defaults to equal."
    )
    notes: str | None = None


class UpdatePositionRequest(BaseModel):
    """Update position allocation."""

    invested_amount: float = Field(..., gt=0)


class SellPositionRequest(BaseModel):
    """Mark position as sold."""

    sell_quarter: str
    sell_price: float | None = None  # Auto-fetched if None
    notes: str | None = None


class SellBatchRequest(BaseModel):
    """Mark entire batch as sold."""

    sell_quarter: str
    notes: str | None = None


# ============================================================================
# Response Models
# ============================================================================


class TrancheResponse(BaseModel):
    """A single tranche within a position."""

    id: str
    buy_quarter: str
    buy_price: float | None
    invested_amount: float
    tranche_target_sell_quarter: str
    source_batch_id: str
    created_at: str
    # Calculated at response time
    current_price: float | None = None
    unrealized_return: float | None = None
    unrealized_alpha: float | None = None
    # If sold
    sell_price: float | None = None
    sell_quarter: str | None = None
    realized_return: float | None = None
    realized_alpha: float | None = None


class PositionResponse(BaseModel):
    """A position with all its tranches."""

    id: str
    batch_id: str
    symbol: str
    name: str | None = None
    status: str  # 'open' or 'sold'
    invested_amount: float
    target_sell_quarter: str
    created_at: str
    # Aggregated from tranches
    tranches: list[TrancheResponse] = Field(default_factory=list)
    total_invested: float = 0
    # Current values (calculated)
    current_price: float | None = None
    current_value: float | None = None
    unrealized_return_pct: float | None = None
    unrealized_alpha_pct: float | None = None
    # Days until target sell
    days_until_sell: int | None = None
    # Realized (if sold)
    realized_return: float | None = None
    realized_alpha: float | None = None


class BatchResponse(BaseModel):
    """A batch with summary."""

    id: str
    name: str | None
    buy_quarter: str
    strategy_id: str | None
    strategy_name: str | None = None
    holding_period: int
    total_invested: float
    status: str
    created_at: str
    notes: str | None
    # Positions
    positions: list[PositionResponse] = Field(default_factory=list)
    position_count: int = 0
    # Aggregated values
    current_value: float | None = None
    unrealized_return_pct: float | None = None
    unrealized_alpha_pct: float | None = None
    target_sell_quarter: str = ""
    days_until_sell: int | None = None
    # Realized (if sold/partial)
    realized_return: float | None = None
    realized_alpha: float | None = None


class DashboardResponse(BaseModel):
    """Full portfolio dashboard."""

    # Summary
    total_invested: float = 0
    total_current_value: float | None = None
    total_unrealized_return: float | None = None
    total_unrealized_alpha: float | None = None
    # Batches grouped by status
    active_batches: list[BatchResponse] = Field(default_factory=list)
    sold_batches: list[BatchResponse] = Field(default_factory=list)
    # Alerts
    sell_alerts: list[PositionResponse] = Field(default_factory=list)
    # Performance
    total_realized_return: float | None = None
    total_realized_alpha: float | None = None


class AlertsResponse(BaseModel):
    """Positions approaching sell date."""

    alerts: list[PositionResponse] = Field(default_factory=list)
    count: int = 0


class PerformanceByStrategy(BaseModel):
    """Historical performance for a strategy."""

    strategy_id: str
    strategy_name: str
    total_trades: int = 0
    total_invested: float = 0
    total_returned: float = 0
    avg_return: float | None = None
    avg_alpha: float | None = None
    win_rate: float | None = None


class PerformanceResponse(BaseModel):
    """Historical performance summary."""

    by_strategy: list[PerformanceByStrategy] = Field(default_factory=list)
    overall_return: float | None = None
    overall_alpha: float | None = None
    overall_win_rate: float | None = None


class CheckMergeResponse(BaseModel):
    """Response for checking merge candidates."""

    existing_positions: dict[str, dict] = Field(default_factory=dict)
    merge_candidates: list[str] = Field(default_factory=list)


# ============================================================================
# Price & Return Calculation Helpers
# ============================================================================


def get_stock_price(conn, symbol: str, quarter: str) -> float | None:
    """Get stock price for a specific quarter."""
    result = conn.execute(
        """
        SELECT price FROM company_profiles
        WHERE UPPER(symbol) = UPPER(?) AND fiscal_quarter = ?
        AND price IS NOT NULL AND price > 0
        """,
        (symbol, quarter),
    ).fetchone()
    return _to_float(result[0]) if result else None


def get_current_stock_price(conn, symbol: str) -> tuple[float | None, str | None]:
    """Get the latest available price and quarter for a stock."""
    result = conn.execute(
        """
        SELECT price, fiscal_quarter FROM company_profiles
        WHERE UPPER(symbol) = UPPER(?) AND price IS NOT NULL AND price > 0
        ORDER BY fiscal_quarter DESC
        LIMIT 1
        """,
        (symbol,),
    ).fetchone()
    if result:
        return _to_float(result[0]), result[1]
    return None, None


def get_spy_price(conn, quarter: str) -> float | None:
    """Get SPY price for a specific quarter."""
    result = conn.execute(
        "SELECT price FROM spy_prices WHERE quarter = ?", (quarter,)
    ).fetchone()
    return _to_float(result[0]) if result else None


def get_current_spy_price(conn) -> tuple[float | None, str | None]:
    """Get the latest SPY price and quarter."""
    result = conn.execute(
        "SELECT price, quarter FROM spy_prices ORDER BY quarter DESC LIMIT 1"
    ).fetchone()
    if result:
        return _to_float(result[0]), result[1]
    return None, None


def calculate_return(buy_price: float, current_price: float) -> float:
    """Calculate percentage return."""
    if buy_price <= 0:
        return 0.0
    return ((current_price - buy_price) / buy_price) * 100


def calculate_position_returns(
    conn, position_id: str, current_quarter: str | None = None
) -> dict:
    """Calculate returns for a position with multiple tranches."""
    # Get tranches
    tranches = conn.execute(
        """
        SELECT id, buy_quarter, buy_price, invested_amount, source_batch_id,
               tranche_target_sell_quarter
        FROM position_tranches
        WHERE position_id = ?
        """,
        (position_id,),
    ).fetchall()

    if not tranches:
        return {
            "unrealized_return_pct": None,
            "unrealized_alpha_pct": None,
            "current_value": None,
        }

    # Get position symbol
    pos = conn.execute(
        "SELECT symbol FROM portfolio_positions WHERE id = ?", (position_id,)
    ).fetchone()
    if not pos:
        return {
            "unrealized_return_pct": None,
            "unrealized_alpha_pct": None,
            "current_value": None,
        }

    symbol = pos[0]
    current_price, current_q = get_current_stock_price(conn, symbol)
    if not current_price:
        return {
            "unrealized_return_pct": None,
            "unrealized_alpha_pct": None,
            "current_value": None,
        }

    current_spy, _ = get_current_spy_price(conn)

    total_invested = sum(_to_float(t[3]) or 0 for t in tranches)
    weighted_return = 0.0
    weighted_alpha = 0.0

    for tranche in tranches:
        buy_price = _to_float(tranche[2])
        invested = _to_float(tranche[3]) or 0
        buy_quarter = tranche[1]

        if not buy_price or buy_price <= 0 or total_invested <= 0:
            continue

        weight = invested / total_invested
        stock_return = (current_price - buy_price) / buy_price

        # Calculate SPY return for same period
        spy_buy = get_spy_price(conn, buy_quarter)
        if spy_buy and current_spy:
            spy_return = (current_spy - spy_buy) / spy_buy
            tranche_alpha = stock_return - spy_return
        else:
            tranche_alpha = 0.0

        weighted_return += stock_return * weight
        weighted_alpha += tranche_alpha * weight

    return {
        "unrealized_return_pct": weighted_return * 100,
        "unrealized_alpha_pct": weighted_alpha * 100,
        "current_value": total_invested * (1 + weighted_return),
        "current_price": current_price,
    }


# ============================================================================
# API Endpoints - Batches
# ============================================================================


@router.post("/batches", response_model=BatchResponse)
async def create_batch(request: CreateBatchRequest) -> BatchResponse:
    """
    Create a new portfolio batch with positions.

    Handles merge logic: if any symbol has an existing open position,
    a new tranche is added to that position instead of creating a new one.
    """
    batch_id = _generate_id()
    now = datetime.now().isoformat()

    db = get_db_manager()
    with db.get_connection() as conn:
        # Calculate target sell quarter
        target_sell = add_quarters(request.buy_quarter, request.holding_period)

        # Get strategy name if provided
        strategy_name = None
        if request.strategy_id:
            result = conn.execute(
                "SELECT name FROM saved_strategies WHERE id = ?",
                (request.strategy_id,),
            ).fetchone()
            if result:
                strategy_name = result[0]

        # Create batch
        conn.execute(
            """
            INSERT INTO portfolio_batches (
                id, name, buy_quarter, strategy_id, holding_period,
                total_invested, created_at, status, notes
            ) VALUES (?, ?, ?, ?, ?, ?, ?, 'active', ?)
            """,
            (
                batch_id,
                request.name or f"{request.buy_quarter} Batch",
                request.buy_quarter,
                request.strategy_id,
                request.holding_period,
                request.total_invested,
                now,
                request.notes,
            ),
        )

        # Calculate allocations
        symbols = [s.upper() for s in request.symbols]
        if request.allocations:
            allocations = {k.upper(): v for k, v in request.allocations.items()}
        else:
            # Equal weight
            per_stock = request.total_invested / len(symbols)
            allocations = {s: per_stock for s in symbols}

        # Check for existing open positions
        placeholders = ",".join(["?" for _ in symbols])
        existing = conn.execute(
            f"""
            SELECT id, symbol FROM portfolio_positions
            WHERE UPPER(symbol) IN ({placeholders}) AND status = 'open'
            """,
            symbols,
        ).fetchall()
        existing_map = {row[1].upper(): row[0] for row in existing}

        positions_created = []

        for symbol in symbols:
            invested = allocations.get(symbol, request.total_invested / len(symbols))
            buy_price = get_stock_price(conn, symbol, request.buy_quarter)

            if symbol in existing_map:
                # MERGE: Add tranche to existing position
                position_id = existing_map[symbol]
                tranche_id = _generate_id()

                conn.execute(
                    """
                    INSERT INTO position_tranches (
                        id, position_id, buy_quarter, buy_price, invested_amount,
                        source_batch_id, tranche_target_sell_quarter, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        tranche_id,
                        position_id,
                        request.buy_quarter,
                        buy_price,
                        invested,
                        batch_id,
                        target_sell,
                        now,
                    ),
                )

                # Update position's target sell to max
                conn.execute(
                    """
                    UPDATE portfolio_positions
                    SET target_sell_quarter = ?,
                        invested_amount = invested_amount + ?
                    WHERE id = ? AND target_sell_quarter < ?
                    """,
                    (target_sell, invested, position_id, target_sell),
                )

                # Also update if not less (just add the amount)
                conn.execute(
                    """
                    UPDATE portfolio_positions
                    SET invested_amount = invested_amount + ?
                    WHERE id = ? AND target_sell_quarter >= ?
                    """,
                    (invested, position_id, target_sell),
                )

                # Record merge transaction
                conn.execute(
                    """
                    INSERT INTO portfolio_transactions (
                        id, transaction_type, batch_id, position_id, symbol,
                        quarter, price, amount, created_at
                    ) VALUES (?, 'merge', ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        _generate_id(),
                        batch_id,
                        position_id,
                        symbol,
                        request.buy_quarter,
                        buy_price,
                        invested,
                        now,
                    ),
                )

            else:
                # NEW: Create position with single tranche
                position_id = _generate_id()
                tranche_id = _generate_id()

                conn.execute(
                    """
                    INSERT INTO portfolio_positions (
                        id, batch_id, symbol, status, invested_amount,
                        target_sell_quarter, created_at
                    ) VALUES (?, ?, ?, 'open', ?, ?, ?)
                    """,
                    (position_id, batch_id, symbol, invested, target_sell, now),
                )

                conn.execute(
                    """
                    INSERT INTO position_tranches (
                        id, position_id, buy_quarter, buy_price, invested_amount,
                        source_batch_id, tranche_target_sell_quarter, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        tranche_id,
                        position_id,
                        request.buy_quarter,
                        buy_price,
                        invested,
                        batch_id,
                        target_sell,
                        now,
                    ),
                )

                # Record buy transaction
                conn.execute(
                    """
                    INSERT INTO portfolio_transactions (
                        id, transaction_type, batch_id, position_id, symbol,
                        quarter, price, amount, created_at
                    ) VALUES (?, 'buy', ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        _generate_id(),
                        batch_id,
                        position_id,
                        symbol,
                        request.buy_quarter,
                        buy_price,
                        invested,
                        now,
                    ),
                )

                positions_created.append(position_id)

    # Return the created batch (outside transaction so it's committed)
    return await get_batch(batch_id)


@router.get("/batches", response_model=list[BatchResponse])
async def list_batches(status: str | None = None) -> list[BatchResponse]:
    """
    List all portfolio batches.

    Args:
        status: Filter by status ('active', 'sold', 'partial')
    """
    db = get_db_manager()
    with db.get_connection() as conn:
        if status:
            result = conn.execute(
                """
                SELECT id FROM portfolio_batches
                WHERE status = ?
                ORDER BY created_at DESC
                """,
                (status,),
            ).fetchall()
        else:
            result = conn.execute(
                "SELECT id FROM portfolio_batches ORDER BY created_at DESC"
            ).fetchall()

        batches = []
        for row in result:
            batch = await get_batch(row[0])
            batches.append(batch)

        return batches


@router.get("/batches/{batch_id}", response_model=BatchResponse)
async def get_batch(batch_id: str) -> BatchResponse:
    """Get a batch with all its positions and calculated values."""
    db = get_db_manager()
    with db.get_connection() as conn:
        batch = conn.execute(
            """
            SELECT id, name, buy_quarter, strategy_id, holding_period,
                   total_invested, created_at, status, notes
            FROM portfolio_batches
            WHERE id = ?
            """,
            (batch_id,),
        ).fetchone()

        if not batch:
            raise HTTPException(status_code=404, detail=f"Batch {batch_id} not found")

        # Get strategy name
        strategy_name = None
        if batch[3]:
            result = conn.execute(
                "SELECT name FROM saved_strategies WHERE id = ?", (batch[3],)
            ).fetchone()
            if result:
                strategy_name = result[0]

        # Get positions
        positions = conn.execute(
            """
            SELECT id, batch_id, symbol, status, invested_amount,
                   target_sell_quarter, created_at, actual_sell_price,
                   actual_sell_quarter, realized_return, realized_alpha
            FROM portfolio_positions
            WHERE batch_id = ?
            ORDER BY symbol
            """,
            (batch_id,),
        ).fetchall()

        position_responses = []
        total_current_value = 0.0
        total_invested = 0.0

        for pos in positions:
            pos_id = pos[0]
            symbol = pos[2]
            invested = _to_float(pos[4]) or 0
            total_invested += invested

            # Get ticker name
            ticker = conn.execute(
                "SELECT name FROM tickers WHERE UPPER(symbol) = UPPER(?)", (symbol,)
            ).fetchone()
            name = ticker[0] if ticker else None

            # Get tranches
            tranches = conn.execute(
                """
                SELECT id, buy_quarter, buy_price, invested_amount, source_batch_id,
                       tranche_target_sell_quarter, created_at, sell_price,
                       sell_quarter, tranche_return, tranche_alpha
                FROM position_tranches
                WHERE position_id = ?
                """,
                (pos_id,),
            ).fetchall()

            tranche_responses = [
                TrancheResponse(
                    id=t[0],
                    buy_quarter=t[1],
                    buy_price=_to_float(t[2]),
                    invested_amount=_to_float(t[3]) or 0,
                    source_batch_id=t[4],
                    tranche_target_sell_quarter=t[5],
                    created_at=t[6] if isinstance(t[6], str) else t[6].isoformat() if t[6] else "",
                    sell_price=_to_float(t[7]),
                    sell_quarter=t[8],
                    realized_return=_to_float(t[9]),
                    realized_alpha=_to_float(t[10]),
                )
                for t in tranches
            ]

            # Calculate returns for open positions
            returns_data = {}
            if pos[3] == "open":
                returns_data = calculate_position_returns(conn, pos_id)
                if returns_data.get("current_value"):
                    total_current_value += returns_data["current_value"]

            days_left = days_until_quarter_end(pos[5]) if pos[3] == "open" else None

            position_responses.append(
                PositionResponse(
                    id=pos_id,
                    batch_id=pos[1],
                    symbol=symbol,
                    name=name,
                    status=pos[3],
                    invested_amount=invested,
                    target_sell_quarter=pos[5],
                    created_at=pos[6] if isinstance(pos[6], str) else pos[6].isoformat() if pos[6] else "",
                    tranches=tranche_responses,
                    total_invested=sum(t.invested_amount for t in tranche_responses),
                    current_price=returns_data.get("current_price"),
                    current_value=returns_data.get("current_value"),
                    unrealized_return_pct=returns_data.get("unrealized_return_pct"),
                    unrealized_alpha_pct=returns_data.get("unrealized_alpha_pct"),
                    days_until_sell=days_left,
                    realized_return=_to_float(pos[9]),
                    realized_alpha=_to_float(pos[10]),
                )
            )

        target_sell = add_quarters(batch[2], batch[4])
        days_left = days_until_quarter_end(target_sell) if batch[7] == "active" else None

        # Calculate batch-level returns
        batch_return = None
        batch_alpha = None
        if total_invested > 0 and total_current_value > 0:
            batch_return = ((total_current_value - total_invested) / total_invested) * 100
            # Calculate batch alpha vs SPY
            spy_buy = get_spy_price(conn, batch[2])
            spy_current, _ = get_current_spy_price(conn)
            if spy_buy and spy_current:
                spy_return = ((spy_current - spy_buy) / spy_buy) * 100
                batch_alpha = batch_return - spy_return

        return BatchResponse(
            id=batch[0],
            name=batch[1],
            buy_quarter=batch[2],
            strategy_id=batch[3],
            strategy_name=strategy_name,
            holding_period=batch[4],
            total_invested=_to_float(batch[5]) or 0,
            status=batch[7],
            created_at=batch[6] if isinstance(batch[6], str) else batch[6].isoformat() if batch[6] else "",
            notes=batch[8],
            positions=position_responses,
            position_count=len(position_responses),
            current_value=total_current_value if total_current_value > 0 else None,
            unrealized_return_pct=batch_return,
            unrealized_alpha_pct=batch_alpha,
            target_sell_quarter=target_sell,
            days_until_sell=days_left,
        )


@router.delete("/batches/{batch_id}")
async def delete_batch(batch_id: str) -> dict:
    """Delete a batch and all its positions."""
    db = get_db_manager()
    with db.get_connection() as conn:
        # Get position IDs first
        positions = conn.execute(
            "SELECT id FROM portfolio_positions WHERE batch_id = ?", (batch_id,)
        ).fetchall()

        # Delete tranches
        for pos in positions:
            conn.execute(
                "DELETE FROM position_tranches WHERE position_id = ?", (pos[0],)
            )

        # Delete positions
        conn.execute("DELETE FROM portfolio_positions WHERE batch_id = ?", (batch_id,))

        # Delete transactions
        conn.execute(
            "DELETE FROM portfolio_transactions WHERE batch_id = ?", (batch_id,)
        )

        # Delete batch
        result = conn.execute(
            "DELETE FROM portfolio_batches WHERE id = ?", (batch_id,)
        )

        if result.rowcount == 0:
            raise HTTPException(status_code=404, detail=f"Batch {batch_id} not found")

        return {"status": "deleted", "batch_id": batch_id}


@router.post("/batches/{batch_id}/sell")
async def sell_batch(batch_id: str, request: SellBatchRequest) -> BatchResponse:
    """Mark entire batch as sold and calculate returns."""
    db = get_db_manager()
    with db.get_connection() as conn:
        # Get all open positions in batch
        positions = conn.execute(
            """
            SELECT id, symbol FROM portfolio_positions
            WHERE batch_id = ? AND status = 'open'
            """,
            (batch_id,),
        ).fetchall()

        if not positions:
            raise HTTPException(
                status_code=400, detail="No open positions in this batch"
            )

        now = datetime.now().isoformat()
        spy_sell = get_spy_price(conn, request.sell_quarter)

        for pos in positions:
            pos_id = pos[0]
            symbol = pos[1]

            # Get sell price
            sell_price = get_stock_price(conn, symbol, request.sell_quarter)
            if not sell_price:
                logger.warning(f"No price for {symbol} in {request.sell_quarter}")
                continue

            # Calculate returns per tranche
            tranches = conn.execute(
                """
                SELECT id, buy_quarter, buy_price, invested_amount
                FROM position_tranches
                WHERE position_id = ?
                """,
                (pos_id,),
            ).fetchall()

            total_return = 0.0
            total_alpha = 0.0
            total_invested = sum(_to_float(t[3]) or 0 for t in tranches)

            for tranche in tranches:
                buy_price = _to_float(tranche[2])
                invested = _to_float(tranche[3]) or 0
                buy_quarter = tranche[1]

                if not buy_price or buy_price <= 0:
                    continue

                tranche_return = calculate_return(buy_price, sell_price)

                # Calculate alpha
                spy_buy = get_spy_price(conn, buy_quarter)
                if spy_buy and spy_sell:
                    spy_return = calculate_return(spy_buy, spy_sell)
                    tranche_alpha = tranche_return - spy_return
                else:
                    tranche_alpha = 0.0

                # Update tranche
                conn.execute(
                    """
                    UPDATE position_tranches
                    SET sell_price = ?, sell_quarter = ?,
                        tranche_return = ?, tranche_alpha = ?
                    WHERE id = ?
                    """,
                    (sell_price, request.sell_quarter, tranche_return, tranche_alpha, tranche[0]),
                )

                # Weight for position-level calculation
                if total_invested > 0:
                    weight = invested / total_invested
                    total_return += tranche_return * weight
                    total_alpha += tranche_alpha * weight

            # Update position
            conn.execute(
                """
                UPDATE portfolio_positions
                SET status = 'sold', actual_sell_price = ?,
                    actual_sell_quarter = ?, realized_return = ?,
                    realized_alpha = ?
                WHERE id = ?
                """,
                (sell_price, request.sell_quarter, total_return, total_alpha, pos_id),
            )

            # Record sell transaction
            conn.execute(
                """
                INSERT INTO portfolio_transactions (
                    id, transaction_type, batch_id, position_id, symbol,
                    quarter, price, return_pct, alpha_pct, created_at
                ) VALUES (?, 'sell', ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    _generate_id(),
                    batch_id,
                    pos_id,
                    symbol,
                    request.sell_quarter,
                    sell_price,
                    total_return,
                    total_alpha,
                    now,
                ),
            )

        # Update batch status
        conn.execute(
            "UPDATE portfolio_batches SET status = 'sold' WHERE id = ?", (batch_id,)
        )

        return await get_batch(batch_id)


# ============================================================================
# API Endpoints - Positions
# ============================================================================


@router.put("/positions/{position_id}", response_model=PositionResponse)
async def update_position(
    position_id: str, request: UpdatePositionRequest
) -> PositionResponse:
    """Update position allocation amount."""
    db = get_db_manager()
    with db.get_connection() as conn:
        result = conn.execute(
            """
            UPDATE portfolio_positions
            SET invested_amount = ?
            WHERE id = ?
            """,
            (request.invested_amount, position_id),
        )

        if result.rowcount == 0:
            raise HTTPException(
                status_code=404, detail=f"Position {position_id} not found"
            )

        # Get the updated position via batch
        pos = conn.execute(
            "SELECT batch_id FROM portfolio_positions WHERE id = ?", (position_id,)
        ).fetchone()

        batch = await get_batch(pos[0])
        for p in batch.positions:
            if p.id == position_id:
                return p

        raise HTTPException(status_code=404, detail="Position not found after update")


@router.delete("/positions/{position_id}")
async def delete_position(position_id: str) -> dict:
    """Remove a position from its batch."""
    db = get_db_manager()
    with db.get_connection() as conn:
        # Delete tranches first
        conn.execute(
            "DELETE FROM position_tranches WHERE position_id = ?", (position_id,)
        )

        # Delete position
        result = conn.execute(
            "DELETE FROM portfolio_positions WHERE id = ?", (position_id,)
        )

        if result.rowcount == 0:
            raise HTTPException(
                status_code=404, detail=f"Position {position_id} not found"
            )

        return {"status": "deleted", "position_id": position_id}


@router.post("/positions/{position_id}/sell")
async def sell_position(
    position_id: str, request: SellPositionRequest
) -> PositionResponse:
    """Mark a single position as sold."""
    db = get_db_manager()
    with db.get_connection() as conn:
        pos = conn.execute(
            "SELECT symbol, batch_id FROM portfolio_positions WHERE id = ?",
            (position_id,),
        ).fetchone()

        if not pos:
            raise HTTPException(
                status_code=404, detail=f"Position {position_id} not found"
            )

        symbol = pos[0]
        batch_id = pos[1]

        # Get sell price
        sell_price = request.sell_price
        if not sell_price:
            sell_price = get_stock_price(conn, symbol, request.sell_quarter)

        if not sell_price:
            raise HTTPException(
                status_code=400,
                detail=f"No price available for {symbol} in {request.sell_quarter}",
            )

        spy_sell = get_spy_price(conn, request.sell_quarter)

        # Calculate returns per tranche
        tranches = conn.execute(
            """
            SELECT id, buy_quarter, buy_price, invested_amount
            FROM position_tranches
            WHERE position_id = ?
            """,
            (position_id,),
        ).fetchall()

        total_return = 0.0
        total_alpha = 0.0
        total_invested = sum(_to_float(t[3]) or 0 for t in tranches)

        for tranche in tranches:
            buy_price = _to_float(tranche[2])
            invested = _to_float(tranche[3]) or 0
            buy_quarter = tranche[1]

            if not buy_price or buy_price <= 0:
                continue

            tranche_return = calculate_return(buy_price, sell_price)

            spy_buy = get_spy_price(conn, buy_quarter)
            if spy_buy and spy_sell:
                spy_return = calculate_return(spy_buy, spy_sell)
                tranche_alpha = tranche_return - spy_return
            else:
                tranche_alpha = 0.0

            conn.execute(
                """
                UPDATE position_tranches
                SET sell_price = ?, sell_quarter = ?,
                    tranche_return = ?, tranche_alpha = ?
                WHERE id = ?
                """,
                (sell_price, request.sell_quarter, tranche_return, tranche_alpha, tranche[0]),
            )

            if total_invested > 0:
                weight = invested / total_invested
                total_return += tranche_return * weight
                total_alpha += tranche_alpha * weight

        # Update position
        conn.execute(
            """
            UPDATE portfolio_positions
            SET status = 'sold', actual_sell_price = ?,
                actual_sell_quarter = ?, realized_return = ?,
                realized_alpha = ?
            WHERE id = ?
            """,
            (sell_price, request.sell_quarter, total_return, total_alpha, position_id),
        )

        # Check if all positions in batch are sold
        open_count = conn.execute(
            """
            SELECT COUNT(*) FROM portfolio_positions
            WHERE batch_id = ? AND status = 'open'
            """,
            (batch_id,),
        ).fetchone()[0]

        if open_count == 0:
            conn.execute(
                "UPDATE portfolio_batches SET status = 'sold' WHERE id = ?",
                (batch_id,),
            )
        else:
            conn.execute(
                "UPDATE portfolio_batches SET status = 'partial' WHERE id = ?",
                (batch_id,),
            )

        # Record transaction
        conn.execute(
            """
            INSERT INTO portfolio_transactions (
                id, transaction_type, batch_id, position_id, symbol,
                quarter, price, return_pct, alpha_pct, created_at
            ) VALUES (?, 'sell', ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                _generate_id(),
                batch_id,
                position_id,
                symbol,
                request.sell_quarter,
                sell_price,
                total_return,
                total_alpha,
                datetime.now().isoformat(),
            ),
        )

        batch = await get_batch(batch_id)
        for p in batch.positions:
            if p.id == position_id:
                return p

        raise HTTPException(status_code=404, detail="Position not found after update")


# ============================================================================
# API Endpoints - Dashboard & Alerts
# ============================================================================


@router.get("/dashboard", response_model=DashboardResponse)
async def get_dashboard() -> DashboardResponse:
    """Get full portfolio dashboard with summary and alerts."""
    active_batches = await list_batches(status="active")
    sold_batches = await list_batches(status="sold")
    partial_batches = await list_batches(status="partial")

    # Combine active and partial
    all_active = active_batches + partial_batches

    # Calculate totals
    total_invested = sum(b.total_invested for b in all_active)
    total_current = sum(b.current_value or 0 for b in all_active)
    total_realized_return = None
    total_realized_alpha = None

    # Calculate realized from sold positions
    realized_returns = []
    realized_alphas = []
    for batch in sold_batches + partial_batches:
        for pos in batch.positions:
            if pos.status == "sold" and pos.realized_return is not None:
                realized_returns.append(pos.realized_return)
            if pos.status == "sold" and pos.realized_alpha is not None:
                realized_alphas.append(pos.realized_alpha)

    if realized_returns:
        total_realized_return = sum(realized_returns) / len(realized_returns)
    if realized_alphas:
        total_realized_alpha = sum(realized_alphas) / len(realized_alphas)

    # Get alerts (positions within 30 days of sell date)
    alerts_response = await get_alerts(days=30)

    # Calculate unrealized totals
    total_unrealized_return = None
    total_unrealized_alpha = None
    if total_invested > 0 and total_current > 0:
        total_unrealized_return = ((total_current - total_invested) / total_invested) * 100

    return DashboardResponse(
        total_invested=total_invested,
        total_current_value=total_current if total_current > 0 else None,
        total_unrealized_return=total_unrealized_return,
        total_unrealized_alpha=total_unrealized_alpha,
        active_batches=all_active,
        sold_batches=sold_batches,
        sell_alerts=alerts_response.alerts,
        total_realized_return=total_realized_return,
        total_realized_alpha=total_realized_alpha,
    )


@router.get("/alerts", response_model=AlertsResponse)
async def get_alerts(days: int = 7) -> AlertsResponse:
    """Get positions approaching sell date."""
    db = get_db_manager()
    alerts = []

    with db.get_connection() as conn:
        positions = conn.execute(
            """
            SELECT id, batch_id, symbol, target_sell_quarter
            FROM portfolio_positions
            WHERE status = 'open'
            """
        ).fetchall()

        for pos in positions:
            days_left = days_until_quarter_end(pos[3])
            if days_left <= days:
                batch = await get_batch(pos[1])
                for p in batch.positions:
                    if p.id == pos[0]:
                        alerts.append(p)
                        break

    # Sort by days until sell
    alerts.sort(key=lambda x: x.days_until_sell or 999)

    return AlertsResponse(alerts=alerts, count=len(alerts))


@router.get("/performance", response_model=PerformanceResponse)
async def get_performance() -> PerformanceResponse:
    """Get historical performance by strategy."""
    db = get_db_manager()

    with db.get_connection() as conn:
        # Get all sold positions grouped by strategy
        results = conn.execute(
            """
            SELECT
                b.strategy_id,
                COALESCE(s.name, 'No Strategy') as strategy_name,
                COUNT(p.id) as total_trades,
                SUM(p.invested_amount) as total_invested,
                AVG(p.realized_return) as avg_return,
                AVG(p.realized_alpha) as avg_alpha,
                SUM(CASE WHEN p.realized_alpha > 0 THEN 1 ELSE 0 END) as wins,
                COUNT(p.id) as total
            FROM portfolio_positions p
            JOIN portfolio_batches b ON b.id = p.batch_id
            LEFT JOIN saved_strategies s ON s.id = b.strategy_id
            WHERE p.status = 'sold' AND p.realized_return IS NOT NULL
            GROUP BY b.strategy_id, s.name
            """
        ).fetchall()

        by_strategy = []
        all_returns = []
        all_alphas = []
        all_wins = 0
        all_total = 0

        for row in results:
            wins = row[6] or 0
            total = row[7] or 0
            win_rate = (wins / total * 100) if total > 0 else None

            by_strategy.append(
                PerformanceByStrategy(
                    strategy_id=row[0] or "none",
                    strategy_name=row[1],
                    total_trades=row[2] or 0,
                    total_invested=_to_float(row[3]) or 0,
                    avg_return=_to_float(row[4]),
                    avg_alpha=_to_float(row[5]),
                    win_rate=win_rate,
                )
            )

            if row[4] is not None:
                all_returns.append(_to_float(row[4]))
            if row[5] is not None:
                all_alphas.append(_to_float(row[5]))
            all_wins += wins
            all_total += total

        overall_return = sum(all_returns) / len(all_returns) if all_returns else None
        overall_alpha = sum(all_alphas) / len(all_alphas) if all_alphas else None
        overall_win_rate = (all_wins / all_total * 100) if all_total > 0 else None

        return PerformanceResponse(
            by_strategy=by_strategy,
            overall_return=overall_return,
            overall_alpha=overall_alpha,
            overall_win_rate=overall_win_rate,
        )


@router.post("/check-merge", response_model=CheckMergeResponse)
async def check_merge(symbols: list[str]) -> CheckMergeResponse:
    """Check if any symbols have existing open positions (merge candidates)."""
    db = get_db_manager()

    with db.get_connection() as conn:
        symbols_upper = [s.upper() for s in symbols]
        placeholders = ",".join(["?" for _ in symbols_upper])

        existing = conn.execute(
            f"""
            SELECT p.id, p.symbol, p.batch_id, p.invested_amount,
                   p.target_sell_quarter, b.name as batch_name
            FROM portfolio_positions p
            JOIN portfolio_batches b ON b.id = p.batch_id
            WHERE UPPER(p.symbol) IN ({placeholders}) AND p.status = 'open'
            """,
            symbols_upper,
        ).fetchall()

        existing_map = {}
        merge_candidates = []

        for row in existing:
            symbol = row[1].upper()
            existing_map[symbol] = {
                "position_id": row[0],
                "batch_id": row[2],
                "batch_name": row[5],
                "invested_amount": _to_float(row[3]),
                "target_sell_quarter": row[4],
            }
            merge_candidates.append(symbol)

        return CheckMergeResponse(
            existing_positions=existing_map, merge_candidates=merge_candidates
        )
