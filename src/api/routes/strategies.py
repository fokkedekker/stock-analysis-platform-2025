"""Saved strategies API routes."""

import json
import logging
from datetime import datetime
from typing import Any
from uuid import uuid4

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from src.database.connection import get_db_manager

logger = logging.getLogger(__name__)
router = APIRouter()


# ============================================================================
# Models
# ============================================================================


class RawFactorFilter(BaseModel):
    """A single raw factor filter."""

    factor: str = Field(..., description="Factor name (e.g., 'pe_ratio', 'roe')")
    operator: str = Field(..., description="Comparison operator: '>=', '<=', '=='")
    value: float | str = Field(..., description="Threshold value")


class PipelineSettings(BaseModel):
    """Pipeline settings including raw factor filters."""

    # Survival Gates
    piotroski_enabled: bool = False
    piotroski_min: int = 5
    altman_enabled: bool = False
    altman_zone: str = "safe"

    # Quality
    quality_enabled: bool = False
    min_quality: str = "weak"
    excluded_tags: list[str] = Field(default_factory=list)
    required_tags: list[str] = Field(default_factory=list)

    # Valuation Lenses
    graham_enabled: bool = False
    graham_mode: str = "strict"
    graham_min: int = 5
    magic_formula_enabled: bool = False
    mf_top_pct: int = 20
    peg_enabled: bool = False
    max_peg: float = 1.5
    net_net_enabled: bool = False
    fama_french_enabled: bool = False
    ff_top_pct: int = 30
    min_lenses: int = 0  # Support raw-filter-only strategies
    strict_mode: bool = False

    # NEW: Raw factor filters
    raw_filters: list[RawFactorFilter] = Field(default_factory=list)


class SaveStrategyRequest(BaseModel):
    """Request to save a new strategy."""

    name: str = Field(..., description="Strategy name")
    holding_period: int | None = Field(None, description="Holding period in quarters")
    settings: PipelineSettings = Field(..., description="Pipeline settings")
    expected_alpha: float | None = Field(None, description="Expected alpha from Factor Discovery")
    expected_alpha_ci_lower: float | None = Field(None, description="95% CI lower bound")
    expected_alpha_ci_upper: float | None = Field(None, description="95% CI upper bound")
    win_rate: float | None = Field(None, description="Expected win rate")
    sample_size: int | None = Field(None, description="Historical sample size")
    source: str = Field("manual", description="Source: 'factor_discovery' or 'manual'")


class SavedStrategy(BaseModel):
    """A saved strategy."""

    id: str
    name: str
    holding_period: int | None
    settings: PipelineSettings
    expected_alpha: float | None
    expected_alpha_ci_lower: float | None
    expected_alpha_ci_upper: float | None
    win_rate: float | None
    sample_size: int | None
    source: str
    created_at: str
    updated_at: str


class UpdateStrategyRequest(BaseModel):
    """Request to update an existing strategy."""

    name: str | None = None
    holding_period: int | None = None
    settings: PipelineSettings | None = None
    expected_alpha: float | None = None
    expected_alpha_ci_lower: float | None = None
    expected_alpha_ci_upper: float | None = None
    win_rate: float | None = None
    sample_size: int | None = None


# ============================================================================
# API Endpoints
# ============================================================================


@router.post("", response_model=SavedStrategy)
async def create_strategy(request: SaveStrategyRequest) -> SavedStrategy:
    """
    Create a new saved strategy.

    Returns the created strategy with its generated ID.
    """
    strategy_id = str(uuid4())[:12]
    now = datetime.now().isoformat()

    db = get_db_manager()
    with db.get_connection() as conn:
        settings_json = json.dumps(request.settings.model_dump())

        conn.execute(
            """
            INSERT INTO saved_strategies (
                id, name, holding_period, settings_json,
                expected_alpha, expected_alpha_ci_lower, expected_alpha_ci_upper,
                win_rate, sample_size, source, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                strategy_id,
                request.name,
                request.holding_period,
                settings_json,
                request.expected_alpha,
                request.expected_alpha_ci_lower,
                request.expected_alpha_ci_upper,
                request.win_rate,
                request.sample_size,
                request.source,
                now,
                now,
            ),
        )

    logger.info(f"Created strategy {strategy_id}: {request.name}")

    return SavedStrategy(
        id=strategy_id,
        name=request.name,
        holding_period=request.holding_period,
        settings=request.settings,
        expected_alpha=request.expected_alpha,
        expected_alpha_ci_lower=request.expected_alpha_ci_lower,
        expected_alpha_ci_upper=request.expected_alpha_ci_upper,
        win_rate=request.win_rate,
        sample_size=request.sample_size,
        source=request.source,
        created_at=now,
        updated_at=now,
    )


@router.get("", response_model=list[SavedStrategy])
async def list_strategies() -> list[SavedStrategy]:
    """
    List all saved strategies.

    Returns strategies sorted by creation date (newest first).
    """
    db = get_db_manager()
    with db.get_connection() as conn:
        rows = conn.execute(
            """
            SELECT id, name, holding_period, settings_json,
                   expected_alpha, expected_alpha_ci_lower, expected_alpha_ci_upper,
                   win_rate, sample_size, source, created_at, updated_at
            FROM saved_strategies
            ORDER BY created_at DESC
            """
        ).fetchall()

    strategies = []
    for row in rows:
        settings_dict = json.loads(row[3]) if row[3] else {}
        strategies.append(
            SavedStrategy(
                id=row[0],
                name=row[1],
                holding_period=row[2],
                settings=PipelineSettings(**settings_dict),
                expected_alpha=row[4],
                expected_alpha_ci_lower=row[5],
                expected_alpha_ci_upper=row[6],
                win_rate=row[7],
                sample_size=row[8],
                source=row[9] or "manual",
                created_at=str(row[10]) if row[10] else "",
                updated_at=str(row[11]) if row[11] else "",
            )
        )

    return strategies


@router.get("/{strategy_id}", response_model=SavedStrategy)
async def get_strategy(strategy_id: str) -> SavedStrategy:
    """
    Get a single saved strategy by ID.
    """
    db = get_db_manager()
    with db.get_connection() as conn:
        row = conn.execute(
            """
            SELECT id, name, holding_period, settings_json,
                   expected_alpha, expected_alpha_ci_lower, expected_alpha_ci_upper,
                   win_rate, sample_size, source, created_at, updated_at
            FROM saved_strategies
            WHERE id = ?
            """,
            (strategy_id,),
        ).fetchone()

    if not row:
        raise HTTPException(status_code=404, detail=f"Strategy {strategy_id} not found")

    settings_dict = json.loads(row[3]) if row[3] else {}

    return SavedStrategy(
        id=row[0],
        name=row[1],
        holding_period=row[2],
        settings=PipelineSettings(**settings_dict),
        expected_alpha=row[4],
        expected_alpha_ci_lower=row[5],
        expected_alpha_ci_upper=row[6],
        win_rate=row[7],
        sample_size=row[8],
        source=row[9] or "manual",
        created_at=str(row[10]) if row[10] else "",
        updated_at=str(row[11]) if row[11] else "",
    )


@router.put("/{strategy_id}", response_model=SavedStrategy)
async def update_strategy(strategy_id: str, request: UpdateStrategyRequest) -> SavedStrategy:
    """
    Update an existing strategy.

    Only provided fields will be updated.
    """
    # First check if strategy exists
    db = get_db_manager()
    with db.get_connection() as conn:
        existing = conn.execute(
            "SELECT id FROM saved_strategies WHERE id = ?",
            (strategy_id,),
        ).fetchone()

        if not existing:
            raise HTTPException(status_code=404, detail=f"Strategy {strategy_id} not found")

        # Build update query dynamically
        updates = []
        params = []

        if request.name is not None:
            updates.append("name = ?")
            params.append(request.name)

        if request.holding_period is not None:
            updates.append("holding_period = ?")
            params.append(request.holding_period)

        if request.settings is not None:
            updates.append("settings_json = ?")
            params.append(json.dumps(request.settings.model_dump()))

        if request.expected_alpha is not None:
            updates.append("expected_alpha = ?")
            params.append(request.expected_alpha)

        if request.expected_alpha_ci_lower is not None:
            updates.append("expected_alpha_ci_lower = ?")
            params.append(request.expected_alpha_ci_lower)

        if request.expected_alpha_ci_upper is not None:
            updates.append("expected_alpha_ci_upper = ?")
            params.append(request.expected_alpha_ci_upper)

        if request.win_rate is not None:
            updates.append("win_rate = ?")
            params.append(request.win_rate)

        if request.sample_size is not None:
            updates.append("sample_size = ?")
            params.append(request.sample_size)

        # Always update updated_at
        updates.append("updated_at = ?")
        params.append(datetime.now().isoformat())

        params.append(strategy_id)

        if updates:
            query = f"UPDATE saved_strategies SET {', '.join(updates)} WHERE id = ?"
            conn.execute(query, params)

    logger.info(f"Updated strategy {strategy_id}")

    # Return the updated strategy
    return await get_strategy(strategy_id)


@router.delete("/{strategy_id}")
async def delete_strategy(strategy_id: str) -> dict:
    """
    Delete a saved strategy.
    """
    db = get_db_manager()
    with db.get_connection() as conn:
        # Check if exists
        existing = conn.execute(
            "SELECT id FROM saved_strategies WHERE id = ?",
            (strategy_id,),
        ).fetchone()

        if not existing:
            raise HTTPException(status_code=404, detail=f"Strategy {strategy_id} not found")

        conn.execute("DELETE FROM saved_strategies WHERE id = ?", (strategy_id,))

    logger.info(f"Deleted strategy {strategy_id}")

    return {"status": "deleted", "id": strategy_id}


# ============================================================================
# Strategy Signals Endpoint
# ============================================================================


class StrategySignal(BaseModel):
    """A single buy/sell signal."""

    buy_quarter: str
    buy_date: str
    sell_quarter: str
    sell_date: str
    buy_price: float | None
    sell_price: float | None
    stock_return: float | None
    spy_return: float | None
    alpha: float | None
    matched: bool


class StrategySignalsResponse(BaseModel):
    """Response with all signals for a stock."""

    symbol: str
    strategy_id: str
    strategy_name: str
    holding_period: int
    signals: list[StrategySignal]
    total_return: float | None
    total_alpha: float | None
    avg_alpha_per_trade: float | None
    num_trades: int
    win_rate: float | None


def _add_quarters(quarter: str, n: int) -> str:
    """Add n quarters to a quarter string. e.g., '2023Q1' + 2 = '2023Q3'."""
    year = int(quarter[:4])
    q = int(quarter[5])
    total_q = (year * 4 + q - 1) + n
    new_year = total_q // 4
    new_q = (total_q % 4) + 1
    return f"{new_year}Q{new_q}"


def _quarter_to_date(quarter: str) -> str:
    """Convert quarter string to end-of-quarter date."""
    year = int(quarter[:4])
    q = int(quarter[5])
    if q == 1:
        return f"{year}-03-31"
    elif q == 2:
        return f"{year}-06-30"
    elif q == 3:
        return f"{year}-09-30"
    else:
        return f"{year}-12-31"


def _check_stock_matches_strategy(stock_data: dict, settings: PipelineSettings) -> bool:
    """Check if a stock's analysis data matches the strategy filters."""

    # Stage 1: Survival Gates
    if settings.altman_enabled:
        zone = stock_data.get("altman_zone")
        if settings.altman_zone == "safe" and zone != "safe":
            return False
        if settings.altman_zone == "grey" and zone not in ("safe", "grey"):
            return False
        # "distress" allows all zones

    if settings.piotroski_enabled:
        score = stock_data.get("piotroski_score")
        if score is None or score < settings.piotroski_min:
            return False

    # Stage 2: Quality
    if settings.quality_enabled:
        roic = stock_data.get("roic")
        fcf_5yr = stock_data.get("fcf_positive_5yr")

        if settings.min_quality == "compounder":
            if not (roic and roic >= 0.15 and fcf_5yr):
                return False
        elif settings.min_quality == "average":
            if not (roic and roic >= 0.08):
                return False

        # Check quality tags
        quality_tags = stock_data.get("quality_tags")
        if quality_tags and isinstance(quality_tags, str):
            try:
                tags_list = json.loads(quality_tags)
            except json.JSONDecodeError:
                tags_list = []
        elif isinstance(quality_tags, list):
            tags_list = quality_tags
        else:
            tags_list = []

        for tag in settings.excluded_tags:
            if tag in tags_list:
                return False

        for tag in settings.required_tags:
            if tag not in tags_list:
                return False

    # Stage 3: Valuation Lenses
    lenses_passed = 0
    lenses_active = 0

    if settings.graham_enabled:
        lenses_active += 1
        graham_score = stock_data.get("graham_score")
        if graham_score and graham_score >= settings.graham_min:
            lenses_passed += 1

    if settings.net_net_enabled:
        lenses_active += 1
        if stock_data.get("trading_below_ncav"):
            lenses_passed += 1

    if settings.peg_enabled:
        lenses_active += 1
        peg = stock_data.get("peg_ratio")
        if peg and 0 < peg <= settings.max_peg:
            lenses_passed += 1

    if settings.magic_formula_enabled:
        lenses_active += 1
        mf_rank = stock_data.get("magic_formula_rank")
        # Calculate threshold based on percentage
        threshold = int(5000 * (settings.mf_top_pct / 100))
        if mf_rank and mf_rank <= threshold:
            lenses_passed += 1

    if settings.fama_french_enabled:
        lenses_active += 1
        ff_bm = stock_data.get("book_to_market_percentile")
        threshold = 1.0 - (settings.ff_top_pct / 100)
        if ff_bm and ff_bm >= threshold:
            lenses_passed += 1

    if lenses_active > 0:
        if settings.strict_mode:
            if lenses_passed != lenses_active:
                return False
        else:
            if lenses_passed < settings.min_lenses:
                return False

    # Stage 4: Raw Filters
    for rf in settings.raw_filters:
        factor = rf.factor
        operator = rf.operator
        value = rf.value

        stock_value = stock_data.get(factor)
        if stock_value is None:
            return False

        try:
            stock_value = float(stock_value)
            filter_value = float(value)
        except (TypeError, ValueError):
            if operator == "==":
                if str(stock_value) != str(value):
                    return False
            continue

        if operator == ">=" and stock_value < filter_value:
            return False
        elif operator == "<=" and stock_value > filter_value:
            return False
        elif operator == ">" and stock_value <= filter_value:
            return False
        elif operator == "<" and stock_value >= filter_value:
            return False
        elif operator == "==" and stock_value != filter_value:
            return False
        elif operator == "!=" and stock_value == filter_value:
            return False

    return True


def _create_signal(
    buy_quarter: str,
    sell_quarter: str,
    stock_prices: dict[str, float],
    spy_prices: dict[str, float],
    current_stock_price: float | None = None,
    current_spy_price: float | None = None,
) -> StrategySignal:
    """Create a StrategySignal with calculated returns.

    For open positions (sell_quarter in the future), uses current prices
    to calculate unrealized returns.
    """
    buy_date = _quarter_to_date(buy_quarter)
    sell_date = _quarter_to_date(sell_quarter)

    buy_price = stock_prices.get(buy_quarter)
    sell_price = stock_prices.get(sell_quarter)

    # For open positions, use current price for unrealized returns
    is_open_position = sell_price is None
    effective_sell_price = sell_price if sell_price else current_stock_price

    # Calculate returns
    stock_return = None
    spy_return = None
    alpha = None

    if buy_price and effective_sell_price and buy_price > 0:
        stock_return = round(((effective_sell_price - buy_price) / buy_price) * 100, 2)

    spy_buy = spy_prices.get(buy_quarter)
    spy_sell = spy_prices.get(sell_quarter)
    # For open positions, use current SPY price
    effective_spy_sell = spy_sell if spy_sell else current_spy_price

    if spy_buy and effective_spy_sell and spy_buy > 0:
        spy_return = round(((effective_spy_sell - spy_buy) / spy_buy) * 100, 2)

    if stock_return is not None and spy_return is not None:
        alpha = round(stock_return - spy_return, 2)

    return StrategySignal(
        buy_quarter=buy_quarter,
        buy_date=buy_date,
        sell_quarter=sell_quarter,
        sell_date=sell_date,
        buy_price=buy_price,
        sell_price=sell_price if not is_open_position else None,  # Keep null for open positions
        stock_return=stock_return,
        spy_return=spy_return,
        alpha=alpha,
        matched=True,
    )


@router.get("/{strategy_id}/signals/{symbol}", response_model=StrategySignalsResponse)
async def get_strategy_signals(strategy_id: str, symbol: str) -> StrategySignalsResponse:
    """
    Get buy/sell signals for a stock based on a saved strategy.

    Uses a "rolling hold" model:
    - Buy when stock first matches the strategy
    - If stock keeps matching in subsequent quarters, extend the sell date
    - Sell when stock stops matching AND holding period expires
    - Then look for next buy opportunity
    """
    # Load strategy
    strategy = await get_strategy(strategy_id)
    holding_period = strategy.holding_period or 1
    settings = strategy.settings

    db = get_db_manager()

    with db.get_connection() as conn:
        # Get all available analysis quarters
        quarters_result = conn.execute(
            """
            SELECT DISTINCT analysis_quarter
            FROM graham_results
            ORDER BY analysis_quarter
            """
        ).fetchall()
        all_quarters = [row[0] for row in quarters_result]

        # Load SPY prices
        spy_result = conn.execute("SELECT quarter, price FROM spy_prices").fetchall()
        spy_prices = {row[0]: float(row[1]) for row in spy_result}

        # Load stock prices from company_profiles
        prices_result = conn.execute(
            """
            SELECT fiscal_quarter, price
            FROM company_profiles
            WHERE UPPER(symbol) = UPPER(?)
            AND price IS NOT NULL AND price > 0
            """,
            (symbol,),
        ).fetchall()
        stock_prices = {row[0]: float(row[1]) for row in prices_result}

        # Get current/latest stock price (most recent quarter)
        current_stock_price = None
        if stock_prices:
            latest_quarter = max(stock_prices.keys())
            current_stock_price = stock_prices[latest_quarter]

        # Get current/latest SPY price
        current_spy_price = None
        if spy_prices:
            latest_spy_quarter = max(spy_prices.keys())
            current_spy_price = spy_prices[latest_spy_quarter]

        # Build signals using rolling hold model
        signals: list[StrategySignal] = []

        # Track current position
        in_position = False
        buy_quarter: str | None = None
        last_match_quarter: str | None = None
        dropped_out = False  # Track if stock dropped out since buy

        for quarter in all_quarters:
            # Get all analysis data for this stock in this quarter
            stock_data = _query_stock_analysis(conn, symbol, quarter)

            if not stock_data:
                # No data for this quarter - if in position, check if we should sell
                if in_position and last_match_quarter:
                    dropped_out = True
                    planned_sell = _add_quarters(last_match_quarter, holding_period)
                    if quarter >= planned_sell:
                        # Holding period expired, close position
                        signals.append(_create_signal(
                            buy_quarter, planned_sell, stock_prices, spy_prices,
                            current_stock_price, current_spy_price
                        ))
                        in_position = False
                        buy_quarter = None
                        last_match_quarter = None
                        dropped_out = False
                continue

            # Check if stock matches strategy
            matched = _check_stock_matches_strategy(stock_data, settings)

            if matched:
                if not in_position:
                    # Open new position
                    in_position = True
                    buy_quarter = quarter
                    last_match_quarter = quarter
                    dropped_out = False
                else:
                    # Already in position - check if we should close old and start new
                    if dropped_out and last_match_quarter:
                        planned_sell = _add_quarters(last_match_quarter, holding_period)
                        if quarter >= planned_sell:
                            # Stock dropped out, we're past planned sell, now matches again
                            # Close old position and start new one
                            signals.append(_create_signal(
                                buy_quarter, planned_sell, stock_prices, spy_prices,
                                current_stock_price, current_spy_price
                            ))
                            # Start new position
                            buy_quarter = quarter
                            last_match_quarter = quarter
                            dropped_out = False
                        else:
                            # Stock came back before planned sell - extend position
                            last_match_quarter = quarter
                            dropped_out = False
                    else:
                        # Continuous match - just extend position
                        last_match_quarter = quarter
            else:
                # Stock doesn't match this quarter
                if in_position and last_match_quarter:
                    dropped_out = True
                    planned_sell = _add_quarters(last_match_quarter, holding_period)
                    if quarter >= planned_sell:
                        # Holding period expired, close position
                        signals.append(_create_signal(
                            buy_quarter, planned_sell, stock_prices, spy_prices,
                            current_stock_price, current_spy_price
                        ))
                        in_position = False
                        buy_quarter = None
                        last_match_quarter = None
                        dropped_out = False

        # Handle position still open at end of data
        if in_position and buy_quarter and last_match_quarter:
            planned_sell = _add_quarters(last_match_quarter, holding_period)
            signals.append(_create_signal(
                buy_quarter, planned_sell, stock_prices, spy_prices,
                current_stock_price, current_spy_price
            ))

    # Calculate aggregate stats
    valid_trades = [s for s in signals if s.stock_return is not None]
    num_trades = len(valid_trades)

    total_return = None
    total_alpha = None
    avg_alpha = None
    win_rate = None

    if num_trades > 0:
        # Compound return
        compound_factor = 1.0
        for s in valid_trades:
            compound_factor *= (1 + s.stock_return / 100)
        total_return = round((compound_factor - 1) * 100, 2)

        # Sum of alpha
        alphas = [s.alpha for s in valid_trades if s.alpha is not None]
        if alphas:
            total_alpha = round(sum(alphas), 2)
            avg_alpha = round(sum(alphas) / len(alphas), 2)
            win_rate = round(len([a for a in alphas if a > 0]) / len(alphas) * 100, 1)

    return StrategySignalsResponse(
        symbol=symbol.upper(),
        strategy_id=strategy_id,
        strategy_name=strategy.name,
        holding_period=holding_period,
        signals=signals,
        total_return=total_return,
        total_alpha=total_alpha,
        avg_alpha_per_trade=avg_alpha,
        num_trades=num_trades,
        win_rate=win_rate,
    )


def _query_stock_analysis(conn: Any, symbol: str, quarter: str) -> dict | None:
    """Query all analysis data for a stock in a given quarter."""
    result = conn.execute(
        """
        SELECT
            -- Altman
            a.z_score as altman_z_score,
            a.zone as altman_zone,
            -- Piotroski
            p.f_score as piotroski_score,
            -- ROIC Quality
            r.roic,
            r.fcf_positive_5yr,
            r.quality_tags,
            r.reinvestment_rate,
            r.roic_std_dev,
            r.gross_margin_std_dev,
            r.fcf_to_net_income,
            r.fcf_yield,
            r.ev_to_ebit,
            r.debt_to_equity as roic_debt_to_equity,
            r.free_cash_flow,
            -- Graham
            g.criteria_passed as graham_score,
            -- Net-Net
            nn.trading_below_ncav,
            nn.discount_to_ncav as net_net_discount,
            -- PEG
            gp.peg_ratio,
            gp.eps_growth_1yr,
            gp.eps_growth_3yr,
            gp.eps_growth_5yr,
            -- Magic Formula
            mf.combined_rank as magic_formula_rank,
            mf.earnings_yield,
            mf.return_on_capital as mf_roic,
            -- Fama-French
            ff.book_to_market_percentile,
            ff.profitability_percentile,
            -- Key metrics (get latest up to quarter end)
            km.pe_ratio,
            km.pb_ratio,
            km.price_to_sales,
            km.roe,
            km.roa,
            km.gross_profit_margin,
            km.operating_profit_margin,
            km.net_profit_margin,
            km.current_ratio,
            km.quick_ratio,
            km.debt_ratio,
            km.debt_to_equity,
            km.debt_to_assets,
            km.interest_coverage,
            km.dividend_yield,
            km.price_to_free_cash_flow,
            km.ev_to_sales,
            km.ev_to_ebitda,
            km.net_debt_to_ebitda,
            km.payout_ratio
        FROM (SELECT 1) dummy
        LEFT JOIN altman_results a ON UPPER(a.symbol) = UPPER(?) AND a.analysis_quarter = ?
        LEFT JOIN piotroski_results p ON UPPER(p.symbol) = UPPER(?) AND p.analysis_quarter = ?
        LEFT JOIN roic_quality_results r ON UPPER(r.symbol) = UPPER(?) AND r.analysis_quarter = ?
        LEFT JOIN graham_results g ON UPPER(g.symbol) = UPPER(?) AND g.analysis_quarter = ? AND g.mode = 'strict'
        LEFT JOIN net_net_results nn ON UPPER(nn.symbol) = UPPER(?) AND nn.analysis_quarter = ?
        LEFT JOIN garp_peg_results gp ON UPPER(gp.symbol) = UPPER(?) AND gp.analysis_quarter = ?
        LEFT JOIN magic_formula_results mf ON UPPER(mf.symbol) = UPPER(?) AND mf.analysis_quarter = ?
        LEFT JOIN fama_french_results ff ON UPPER(ff.symbol) = UPPER(?) AND ff.analysis_quarter = ?
        LEFT JOIN (
            SELECT *
            FROM key_metrics
            WHERE UPPER(symbol) = UPPER(?)
            AND fiscal_date <= ?
            ORDER BY fiscal_date DESC
            LIMIT 1
        ) km ON 1=1
        """,
        (
            symbol, quarter,  # altman
            symbol, quarter,  # piotroski
            symbol, quarter,  # roic
            symbol, quarter,  # graham
            symbol, quarter,  # net_net
            symbol, quarter,  # garp_peg
            symbol, quarter,  # magic_formula
            symbol, quarter,  # fama_french
            symbol, _quarter_to_date(quarter),  # key_metrics
        ),
    ).fetchone()

    if not result:
        return None

    columns = [desc[0] for desc in conn.description]
    return dict(zip(columns, result))
