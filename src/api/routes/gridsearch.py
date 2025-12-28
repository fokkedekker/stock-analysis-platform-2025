"""Grid search backtesting API routes."""

import asyncio
import json
import logging
from datetime import datetime
from typing import Any
from uuid import uuid4

from fastapi import APIRouter, BackgroundTasks, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from src.backtest import (
    GridSearchProgress,
    GridSearchRequest,
    GridSearchResult,
    GridSearchRunner,
    SimulationDataPreloader,
    StrategyBuilder,
)
from src.backtest.models import AVAILABLE_DIMENSIONS
from src.database.connection import get_db_manager

logger = logging.getLogger(__name__)
router = APIRouter()

# In-memory storage for running/completed searches
_running_searches: dict[str, dict] = {}
_completed_searches: dict[str, GridSearchResult] = {}
_cancel_flags: dict[str, bool] = {}  # Track cancellation requests


def _save_search_to_db(search_id: str, result: GridSearchResult, name: str | None = None) -> None:
    """Save a completed grid search to the database."""
    db = get_db_manager()
    with db.get_connection() as conn:
        # Serialize results to JSON
        results_json = json.dumps([r.model_dump() for r in result.results])
        best_by_alpha_json = json.dumps([r.model_dump() for r in result.best_by_alpha])
        best_by_win_rate_json = json.dumps([r.model_dump() for r in result.best_by_win_rate])
        request_config_json = json.dumps(result.request_config)

        # Get best metrics
        best_alpha = result.best_by_alpha[0].alpha if result.best_by_alpha else None
        best_win_rate = result.best_by_win_rate[0].win_rate if result.best_by_win_rate else None

        # Generate a name if not provided
        if not name:
            dims = result.request_config.get("dimensions", [])
            dim_names = [d.get("name", "unknown") for d in dims]
            name = " + ".join(dim_names[:3]) if dim_names else "Unnamed Search"

        conn.execute(
            """
            INSERT INTO grid_searches (
                id, name, started_at, completed_at, status,
                total_simulations, completed_simulations, duration_seconds,
                request_config, best_alpha, best_win_rate,
                results_json, best_by_alpha_json, best_by_win_rate_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                search_id,
                name,
                result.started_at,
                result.completed_at,
                "completed",
                result.total_simulations,
                result.completed_simulations,
                result.duration_seconds,
                request_config_json,
                best_alpha,
                best_win_rate,
                results_json,
                best_by_alpha_json,
                best_by_win_rate_json,
            ),
        )
        logger.info(f"Saved grid search {search_id} to database")


def _load_search_from_db(search_id: str) -> GridSearchResult | None:
    """Load a grid search result from the database."""
    db = get_db_manager()
    with db.get_connection() as conn:
        row = conn.execute(
            """
            SELECT id, started_at, completed_at, total_simulations, completed_simulations,
                   duration_seconds, request_config, results_json, best_by_alpha_json, best_by_win_rate_json
            FROM grid_searches
            WHERE id = ?
            """,
            (search_id,),
        ).fetchone()

        if not row:
            return None

        from src.backtest.models import SimulationResult

        # Parse JSON fields
        results = [SimulationResult(**r) for r in json.loads(row[7])]
        best_by_alpha = [SimulationResult(**r) for r in json.loads(row[8])]
        best_by_win_rate = [SimulationResult(**r) for r in json.loads(row[9])]
        request_config = json.loads(row[6])

        return GridSearchResult(
            id=row[0],
            started_at=row[1],
            completed_at=row[2],
            total_simulations=row[3],
            completed_simulations=row[4],
            duration_seconds=row[5],
            results=results,
            best_by_alpha=best_by_alpha,
            best_by_win_rate=best_by_win_rate,
            request_config=request_config,
        )


class PreviewRequest(BaseModel):
    """Request for previewing a grid search."""

    dimensions: list[dict[str, Any]]  # [{name: str, values: list}]
    quarters: list[str]
    holding_periods: list[int] = [1, 2, 3, 4]


class PreviewResponse(BaseModel):
    """Response for grid search preview."""

    strategy_count: int
    total_simulations: int
    estimated_seconds: float
    quarters_with_data: list[str]
    holding_periods_valid: list[int]


class StartResponse(BaseModel):
    """Response when starting a grid search."""

    search_id: str
    status: str
    total_simulations: int


async def _get_available_quarters() -> list[str]:
    """Get all quarters with analysis data."""
    db = get_db_manager()
    with db.get_connection() as conn:
        result = conn.execute(
            """
            SELECT DISTINCT analysis_quarter
            FROM (
                SELECT analysis_quarter FROM graham_results
                UNION
                SELECT analysis_quarter FROM magic_formula_results
                UNION
                SELECT analysis_quarter FROM piotroski_results
                UNION
                SELECT analysis_quarter FROM altman_results
            )
            ORDER BY analysis_quarter DESC
            """
        ).fetchall()
        return [row[0] for row in result]


async def _get_quarters_with_prices() -> set[str]:
    """Get all quarters that have price data."""
    db = get_db_manager()
    with db.get_connection() as conn:
        result = conn.execute(
            """
            SELECT DISTINCT fiscal_quarter
            FROM company_profiles
            WHERE price IS NOT NULL
            ORDER BY fiscal_quarter
            """
        ).fetchall()
        return {row[0] for row in result}


def _add_quarters(quarter: str, periods: int) -> str:
    """Add N quarters to a quarter string."""
    year = int(quarter[:4])
    q = int(quarter[-1])
    total_quarters = year * 4 + q + periods
    new_year = (total_quarters - 1) // 4
    new_q = ((total_quarters - 1) % 4) + 1
    return f"{new_year}Q{new_q}"


def _run_grid_search_task(
    search_id: str,
    request: GridSearchRequest,
) -> None:
    """Background task to run grid search."""
    try:
        # Initialize cancel flag
        _cancel_flags[search_id] = False

        # Get all quarters needed (buy quarters + potential sell quarters)
        all_quarters = set(request.quarters)
        for q in request.quarters:
            for hold in request.holding_periods:
                all_quarters.add(_add_quarters(q, hold))

        # Preload data
        preloader = SimulationDataPreloader(list(all_quarters))
        preloader.preload()

        # Debug: Log what data was loaded
        logger.info(f"Requested quarters: {sorted(request.quarters)}")
        logger.info(f"All quarters needed (buy+sell): {sorted(all_quarters)}")
        logger.info(f"Available price quarters in DB: {sorted(preloader.available_price_quarters)}")
        logger.info(f"Analysis data loaded for quarters: {sorted(preloader.analysis_data.keys())}")

        # Check which sell quarters are valid
        valid_combos = 0
        for q in request.quarters:
            for hold in request.holding_periods:
                sell_q = preloader.get_sell_quarter(q, hold)
                if sell_q:
                    valid_combos += 1
                else:
                    calc_sell = _add_quarters(q, hold)
                    logger.warning(f"No price data for sell quarter: buy={q}, hold={hold}Q, sell={calc_sell}")
        logger.info(f"Valid buy/sell/hold combinations: {valid_combos}")

        # Check cancellation after preload
        if _cancel_flags.get(search_id, False):
            _running_searches[search_id]["status"] = "cancelled"
            return

        # Progress callback with cancellation check
        def progress_callback(progress: GridSearchProgress) -> None:
            _running_searches[search_id]["progress"] = progress
            logger.debug(f"Progress update: {progress.completed}/{progress.total_simulations}")

        # Cancel check function
        def should_cancel() -> bool:
            return _cancel_flags.get(search_id, False)

        # Update status to show simulations are starting
        logger.info(f"Starting simulations for search {search_id}")

        # Run grid search with 8 workers (M2 Mac 8-core)
        runner = GridSearchRunner(
            preloader=preloader,
            num_workers=8,
            progress_callback=progress_callback,
            cancel_check=should_cancel,
        )
        result = runner.run(request)

        # Check if cancelled
        if _cancel_flags.get(search_id, False):
            _running_searches[search_id]["status"] = "cancelled"
            return

        # Store result in memory
        _completed_searches[search_id] = result
        _running_searches[search_id]["status"] = "completed"
        _running_searches[search_id]["result"] = result

        # Save to database for persistence
        try:
            _save_search_to_db(search_id, result)
        except Exception as e:
            logger.error(f"Failed to save grid search to database: {e}")

    except Exception as e:
        logger.error(f"Grid search failed: {e}")
        _running_searches[search_id]["status"] = "failed"
        _running_searches[search_id]["error"] = str(e)
    finally:
        # Clean up cancel flag
        _cancel_flags.pop(search_id, None)


@router.get("/dimensions")
async def get_available_dimensions() -> dict:
    """Get all available grid search dimensions and their values."""
    return {
        "dimensions": AVAILABLE_DIMENSIONS,
        "groups": {
            "survival": ["altman_zone", "piotroski_min"],
            "quality": ["quality_enabled", "min_quality"],
            "quality_tags": [k for k in AVAILABLE_DIMENSIONS if k.startswith("tag_")],
            "valuation_graham": ["graham_enabled", "graham_mode", "graham_min"],
            "valuation_magic_formula": ["magic_formula_enabled", "mf_top_pct"],
            "valuation_peg": ["peg_enabled", "max_peg"],
            "valuation_net_net": ["net_net_enabled"],
            "valuation_fama_french": ["fama_french_enabled", "ff_top_pct"],
            "valuation_logic": ["min_lenses", "strict_mode"],
        },
    }


@router.get("/quarters")
async def get_quarters() -> dict:
    """Get all available quarters for grid search.

    Returns quarters sorted by completeness (most valid holding periods first).
    Recommends the 4 quarters with ALL holding periods valid (oldest complete quarters).
    """
    analysis_quarters = await _get_available_quarters()
    price_quarters = await _get_quarters_with_prices()

    # Score each quarter by how many holding periods have valid sell data
    # Quarters with all 4 holding periods valid are best for backtesting
    quarter_scores = []
    for quarter in analysis_quarters:
        valid_holds = 0
        for hold in [1, 2, 3, 4]:
            sell_quarter = _add_quarters(quarter, hold)
            if sell_quarter in price_quarters:
                valid_holds += 1
        if valid_holds > 0:
            quarter_scores.append((quarter, valid_holds))

    # Sort by:
    # 1. Number of valid holding periods (descending - more complete is better)
    # 2. Quarter date (ascending - older quarters first for complete data)
    quarter_scores.sort(key=lambda x: (-x[1], x[0]))

    valid_buy_quarters = [q[0] for q in quarter_scores]

    # Get quarters with ALL 4 holding periods valid (best for backtesting)
    complete_quarters = [q[0] for q in quarter_scores if q[1] == 4]

    # Recommend oldest complete quarters (they have full sell data)
    recommended = complete_quarters[:4] if complete_quarters else valid_buy_quarters[:4]

    logger.info(f"Available quarters: {valid_buy_quarters[:8]}")
    logger.info(f"Complete quarters (all holds valid): {complete_quarters}")
    logger.info(f"Recommended: {recommended}")

    return {
        "analysis_quarters": valid_buy_quarters,
        "price_quarters": sorted(price_quarters),
        "recommended": recommended,
    }


@router.post("/preview")
async def preview_grid_search(request: PreviewRequest) -> PreviewResponse:
    """
    Preview a grid search without running it.

    Returns combination count and estimated runtime.
    """
    from src.backtest.models import GridDimension

    # Convert dimensions
    dimensions = [GridDimension(name=d["name"], values=d["values"]) for d in request.dimensions]

    # Count strategy combinations
    strategy_count = StrategyBuilder.count_combinations(dimensions)

    # Get quarters with price data
    price_quarters = await _get_quarters_with_prices()

    # Calculate valid simulations
    valid_quarters = [q for q in request.quarters if q in price_quarters]
    valid_holding_periods = []
    valid_simulation_count = 0

    for hold in request.holding_periods:
        has_valid = False
        for quarter in valid_quarters:
            sell_quarter = _add_quarters(quarter, hold)
            if sell_quarter in price_quarters:
                valid_simulation_count += strategy_count
                has_valid = True
        if has_valid:
            valid_holding_periods.append(hold)

    # Estimate runtime: ~8ms per simulation with 6 workers
    estimated_seconds = (valid_simulation_count * 0.008)

    return PreviewResponse(
        strategy_count=strategy_count,
        total_simulations=valid_simulation_count,
        estimated_seconds=round(estimated_seconds, 1),
        quarters_with_data=valid_quarters,
        holding_periods_valid=valid_holding_periods,
    )


@router.post("/start")
async def start_grid_search(
    request: GridSearchRequest,
    background_tasks: BackgroundTasks,
) -> StartResponse:
    """
    Start a grid search simulation.

    Returns immediately with a search_id for progress tracking.
    """
    from src.backtest.models import GridDimension

    search_id = str(uuid4())

    # Calculate total simulations
    dimensions = [GridDimension(name=d.name, values=d.values) for d in request.dimensions]
    strategy_count = StrategyBuilder.count_combinations(dimensions)
    price_quarters = await _get_quarters_with_prices()

    total_simulations = 0
    for quarter in request.quarters:
        for hold in request.holding_periods:
            sell_quarter = _add_quarters(quarter, hold)
            if sell_quarter in price_quarters:
                total_simulations += strategy_count

    # Initialize tracking
    _running_searches[search_id] = {
        "status": "running",
        "started_at": datetime.now(),
        "total_simulations": total_simulations,
        "progress": GridSearchProgress(
            search_id=search_id,
            status="running",
            total_simulations=total_simulations,
            completed=0,
        ),
    }

    # Start background task
    background_tasks.add_task(_run_grid_search_task, search_id, request)

    return StartResponse(
        search_id=search_id,
        status="started",
        total_simulations=total_simulations,
    )


@router.post("/cancel/{search_id}")
async def cancel_search(search_id: str) -> dict:
    """Cancel a running grid search."""
    if search_id not in _running_searches:
        raise HTTPException(status_code=404, detail="Search not found")

    status = _running_searches[search_id].get("status")
    if status != "running":
        raise HTTPException(status_code=400, detail=f"Search is not running (status: {status})")

    # Set cancel flag
    _cancel_flags[search_id] = True
    _running_searches[search_id]["status"] = "cancelling"

    return {"cancelled": True, "search_id": search_id}


@router.get("/progress/{search_id}")
async def get_progress_stream(search_id: str):
    """
    Stream progress updates via Server-Sent Events.
    """

    async def generate():
        while True:
            if search_id not in _running_searches:
                yield f"data: {json.dumps({'error': 'Search not found'})}\n\n"
                break

            search_data = _running_searches[search_id]
            status = search_data.get("status", "running")

            if status == "completed":
                yield f"data: {json.dumps({'completed': True, 'search_id': search_id})}\n\n"
                break
            elif status == "cancelled" or status == "cancelling":
                yield f"data: {json.dumps({'status': 'cancelled', 'search_id': search_id})}\n\n"
                break
            elif status == "failed":
                error = search_data.get("error", "Unknown error")
                yield f"data: {json.dumps({'error': error, 'failed': True})}\n\n"
                break
            else:
                progress = search_data.get("progress")
                if progress:
                    yield f"data: {json.dumps(progress.model_dump())}\n\n"
                else:
                    yield f"data: {json.dumps({'status': 'initializing'})}\n\n"

            await asyncio.sleep(0.5)

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )


@router.get("/status/{search_id}")
async def get_status(search_id: str) -> dict:
    """Get current status of a grid search."""
    if search_id in _completed_searches:
        result = _completed_searches[search_id]
        return {
            "status": "completed",
            "search_id": search_id,
            "total_simulations": result.total_simulations,
            "completed_simulations": result.completed_simulations,
            "duration_seconds": result.duration_seconds,
        }

    if search_id in _running_searches:
        data = _running_searches[search_id]
        progress = data.get("progress")
        return {
            "status": data.get("status", "running"),
            "search_id": search_id,
            "total_simulations": data.get("total_simulations", 0),
            "completed": progress.completed if progress else 0,
            "error": data.get("error"),
        }

    raise HTTPException(status_code=404, detail="Search not found")


def _get_result(search_id: str) -> GridSearchResult | None:
    """Get result from memory or database."""
    # Check memory first
    if search_id in _completed_searches:
        return _completed_searches[search_id]
    # Check database
    return _load_search_from_db(search_id)


@router.get("/history")
async def get_history(limit: int = 20, offset: int = 0) -> dict:
    """Get list of past grid searches."""
    db = get_db_manager()
    with db.get_connection() as conn:
        # Get total count
        total = conn.execute("SELECT COUNT(*) FROM grid_searches").fetchone()[0]

        # Get recent searches (summary only, no full results)
        rows = conn.execute(
            """
            SELECT id, name, started_at, completed_at, status,
                   total_simulations, completed_simulations, duration_seconds,
                   best_alpha, best_win_rate, request_config
            FROM grid_searches
            ORDER BY started_at DESC
            LIMIT ? OFFSET ?
            """,
            (limit, offset),
        ).fetchall()

        searches = []
        for row in rows:
            request_config = json.loads(row[10]) if row[10] else {}
            dims = request_config.get("dimensions", [])
            quarters = request_config.get("quarters", [])

            searches.append({
                "id": row[0],
                "name": row[1],
                "started_at": row[2].isoformat() if row[2] else None,
                "completed_at": row[3].isoformat() if row[3] else None,
                "status": row[4],
                "total_simulations": row[5],
                "completed_simulations": row[6],
                "duration_seconds": row[7],
                "best_alpha": row[8],
                "best_win_rate": row[9],
                "dimension_count": len(dims),
                "quarter_count": len(quarters),
            })

        return {
            "total": total,
            "offset": offset,
            "limit": limit,
            "searches": searches,
        }


@router.get("/results/{search_id}")
async def get_results(search_id: str) -> dict:
    """Get completed grid search results."""
    # Check if still running
    if search_id in _running_searches:
        status = _running_searches[search_id].get("status")
        if status == "running":
            raise HTTPException(status_code=202, detail="Search still running")
        elif status == "failed":
            error = _running_searches[search_id].get("error")
            raise HTTPException(status_code=500, detail=f"Search failed: {error}")

    # Get result from memory or database
    result = _get_result(search_id)
    if not result:
        raise HTTPException(status_code=404, detail="Results not found")

    # Compute aggregations if not present (for old results from DB)
    by_strategy = result.by_strategy if result.by_strategy else []
    by_holding_period = result.by_holding_period if result.by_holding_period else []

    # If aggregations are empty but we have results, compute them
    if not by_strategy and result.results:
        from src.backtest.grid_runner import _compute_aggregations
        by_strategy_list, by_hold_list = _compute_aggregations(result.results)
        by_strategy = by_strategy_list
        by_holding_period = by_hold_list

    return {
        "id": result.id,
        "started_at": result.started_at.isoformat() if hasattr(result.started_at, 'isoformat') else str(result.started_at),
        "completed_at": result.completed_at.isoformat() if result.completed_at and hasattr(result.completed_at, 'isoformat') else str(result.completed_at) if result.completed_at else None,
        "total_simulations": result.total_simulations,
        "completed_simulations": result.completed_simulations,
        "duration_seconds": result.duration_seconds,
        "best_by_alpha": [r.model_dump() for r in result.best_by_alpha],
        "best_by_win_rate": [r.model_dump() for r in result.best_by_win_rate],
        "by_strategy": [s.model_dump() for s in by_strategy],
        "by_holding_period": [h.model_dump() for h in by_holding_period],
        "request_config": result.request_config,
        "all_results_count": len(result.results),
    }


@router.get("/results/{search_id}/all")
async def get_all_results(
    search_id: str,
    sort_by: str = "alpha",
    limit: int = 100,
    offset: int = 0,
) -> dict:
    """Get all results with pagination and sorting."""
    result = _get_result(search_id)
    if not result:
        raise HTTPException(status_code=404, detail="Results not found")

    all_results = result.results

    # Sort
    if sort_by == "alpha":
        all_results = sorted(all_results, key=lambda x: x.alpha, reverse=True)
    elif sort_by == "win_rate":
        all_results = sorted(all_results, key=lambda x: x.win_rate, reverse=True)
    elif sort_by == "stock_count":
        all_results = sorted(all_results, key=lambda x: x.stock_count, reverse=True)
    elif sort_by == "portfolio_return":
        all_results = sorted(all_results, key=lambda x: x.portfolio_return, reverse=True)

    # Paginate
    paginated = all_results[offset : offset + limit]

    return {
        "total": len(all_results),
        "offset": offset,
        "limit": limit,
        "results": [r.model_dump() for r in paginated],
    }


@router.get("/results/{search_id}/export")
async def export_results(search_id: str, format: str = "json") -> dict:
    """Export results as JSON (CSV could be added later)."""
    result = _get_result(search_id)
    if not result:
        raise HTTPException(status_code=404, detail="Results not found")

    if format == "json":
        return {
            "search_id": result.id,
            "started_at": result.started_at.isoformat(),
            "completed_at": result.completed_at.isoformat() if result.completed_at else None,
            "duration_seconds": result.duration_seconds,
            "total_simulations": result.total_simulations,
            "request_config": result.request_config,
            "results": [r.model_dump() for r in result.results],
        }
    else:
        raise HTTPException(status_code=400, detail=f"Unsupported format: {format}")


@router.delete("/results/{search_id}")
async def delete_results(search_id: str) -> dict:
    """Delete a grid search result from memory."""
    deleted = False

    if search_id in _completed_searches:
        del _completed_searches[search_id]
        deleted = True

    if search_id in _running_searches:
        del _running_searches[search_id]
        deleted = True

    if not deleted:
        raise HTTPException(status_code=404, detail="Search not found")

    return {"deleted": True, "search_id": search_id}
