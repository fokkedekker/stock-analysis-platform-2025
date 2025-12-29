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
    min_lenses: int = 1
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
