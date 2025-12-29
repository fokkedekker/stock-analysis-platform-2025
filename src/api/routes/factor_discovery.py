"""Factor Discovery API routes."""

import asyncio
import logging
from typing import Optional

from fastapi import APIRouter, BackgroundTasks, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from src.database.connection import get_db_manager
from src.factor_discovery import (
    FactorDiscoveryProgress,
    FactorDiscoveryRequest,
    FactorDiscoveryResult,
    FactorDiscoverySummary,
    FactorDiscoveryRunner,
    RecommendedStrategy,
    get_storage,
)

logger = logging.getLogger(__name__)
router = APIRouter()

# Track running analyses for progress reporting and cancellation
_running_analyses: dict[str, dict] = {}


class RunResponse(BaseModel):
    """Response for starting a new run."""

    run_id: str
    status: str
    estimated_duration_seconds: int


class QuartersResponse(BaseModel):
    """Response for available quarters."""

    quarters: list[str]
    total: int
    latest: str | None


class RecommendedResponse(BaseModel):
    """Response for recommended strategies."""

    best_holding_period: int | None
    strategies: dict[int, RecommendedStrategy]


@router.post("/run", response_model=RunResponse)
async def start_analysis(
    request: FactorDiscoveryRequest,
    background_tasks: BackgroundTasks,
) -> RunResponse:
    """
    Start a new factor discovery analysis.

    This runs in the background. Use /progress/{run_id} to track progress
    and /results/{run_id} to get results when complete.
    """
    import uuid

    run_id = str(uuid.uuid4())[:12]

    # Initialize tracking
    _running_analyses[run_id] = {
        "status": "starting",
        "progress": 0.0,
        "phase": "initializing",
        "cancelled": False,
    }

    # Define progress callback
    def progress_callback(update: FactorDiscoveryProgress):
        if run_id in _running_analyses:
            _running_analyses[run_id].update({
                "status": update.status,
                "progress": update.progress,
                "phase": update.phase,
                "current_factor": update.current_factor,
                "current_holding_period": update.current_holding_period,
            })

    # Define cancel check
    def cancel_check() -> bool:
        return _running_analyses.get(run_id, {}).get("cancelled", False)

    # Run analysis in background
    async def run_analysis():
        try:
            runner = FactorDiscoveryRunner(
                num_workers=8,
                progress_callback=progress_callback,
                cancel_check=cancel_check,
            )

            # Run in thread pool to avoid blocking
            import concurrent.futures
            from functools import partial

            loop = asyncio.get_event_loop()
            with concurrent.futures.ThreadPoolExecutor() as pool:
                result = await loop.run_in_executor(pool, partial(runner.run, request, run_id))

            # Save result
            storage = get_storage()
            storage.save_result(result)

            _running_analyses[run_id]["status"] = result.status
            _running_analyses[run_id]["progress"] = 1.0

            logger.info(f"Factor discovery {run_id} completed with status {result.status}")

        except Exception as e:
            logger.exception(f"Factor discovery {run_id} failed: {e}")
            _running_analyses[run_id]["status"] = "failed"
            _running_analyses[run_id]["error"] = str(e)

    background_tasks.add_task(run_analysis)

    # Estimate duration based on quarters and holding periods
    num_quarters = len(request.quarters)
    num_hps = len(request.holding_periods)
    estimated_seconds = max(30, num_quarters * num_hps * 3)

    return RunResponse(
        run_id=run_id,
        status="running",
        estimated_duration_seconds=estimated_seconds,
    )


@router.get("/progress/{run_id}")
async def get_progress(run_id: str):
    """
    Get progress updates for a running analysis via SSE.

    Returns Server-Sent Events stream with progress updates.
    """
    import json

    async def event_generator():
        last_progress = -1.0

        while True:
            if run_id not in _running_analyses:
                # Check if run exists in database (completed before we started watching)
                storage = get_storage()
                result = storage.load_result(run_id)
                if result:
                    yield f"data: {json.dumps({'status': result.status, 'progress': 1.0, 'phase': 'complete'})}\n\n"
                else:
                    yield f"data: {json.dumps({'status': 'not_found', 'error': 'Run not found'})}\n\n"
                break

            info = _running_analyses[run_id]
            current_progress = info.get("progress", 0.0)

            # Only send update if progress changed
            if current_progress != last_progress:
                last_progress = current_progress
                yield f"data: {json.dumps(info)}\n\n"

            # Check if complete
            if info.get("status") in ("completed", "failed", "cancelled"):
                break

            await asyncio.sleep(0.5)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )


@router.get("/results/{run_id}", response_model=FactorDiscoveryResult)
async def get_results(run_id: str) -> FactorDiscoveryResult:
    """
    Get complete results for a factor discovery run.

    Returns full analysis including factor results, combined strategies,
    and recommendations for each holding period.
    """
    storage = get_storage()
    result = storage.load_result(run_id)

    if not result:
        raise HTTPException(status_code=404, detail=f"Run {run_id} not found")

    return result


@router.get("/results/{run_id}/recommended", response_model=RecommendedResponse)
async def get_recommended(
    run_id: str,
    holding_period: Optional[int] = None,
) -> RecommendedResponse:
    """
    Get recommended strategies for Pipeline integration.

    Args:
        run_id: The analysis run ID
        holding_period: Optional specific holding period (returns all if not specified)
    """
    storage = get_storage()
    result = storage.load_result(run_id)

    if not result:
        raise HTTPException(status_code=404, detail=f"Run {run_id} not found")

    strategies = result.recommended_strategies

    if holding_period is not None:
        if holding_period not in strategies:
            raise HTTPException(
                status_code=404,
                detail=f"No recommendation for holding period {holding_period}Q",
            )
        strategies = {holding_period: strategies[holding_period]}

    return RecommendedResponse(
        best_holding_period=result.best_holding_period,
        strategies=strategies,
    )


@router.get("/history", response_model=list[FactorDiscoverySummary])
async def list_runs(limit: int = 50) -> list[FactorDiscoverySummary]:
    """
    List past factor discovery runs.

    Returns summary information for each run, sorted by date descending.
    """
    storage = get_storage()
    return storage.list_runs(limit=limit)


@router.post("/cancel/{run_id}")
async def cancel_run(run_id: str) -> dict:
    """
    Cancel a running analysis.

    Sets a cancellation flag that the runner checks periodically.
    """
    if run_id not in _running_analyses:
        raise HTTPException(status_code=404, detail=f"Run {run_id} not found or already completed")

    _running_analyses[run_id]["cancelled"] = True
    _running_analyses[run_id]["status"] = "cancelling"

    return {"status": "cancelling", "run_id": run_id}


@router.delete("/{run_id}")
async def delete_run(run_id: str) -> dict:
    """
    Delete a factor discovery run and all its results.

    Cannot delete a running analysis - cancel it first.
    """
    # Check if running
    if run_id in _running_analyses:
        status = _running_analyses[run_id].get("status")
        if status not in ("completed", "failed", "cancelled"):
            raise HTTPException(
                status_code=400,
                detail="Cannot delete a running analysis. Cancel it first.",
            )
        # Clean up tracking
        del _running_analyses[run_id]

    storage = get_storage()
    deleted = storage.delete_run(run_id)

    if not deleted:
        raise HTTPException(status_code=404, detail=f"Run {run_id} not found")

    return {"status": "deleted", "run_id": run_id}


@router.get("/quarters", response_model=QuartersResponse)
async def get_available_quarters() -> QuartersResponse:
    """
    Get all available quarters for factor discovery analysis.

    Returns quarters that have price data and can be used for analysis.
    """
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

        quarters = [row[0] for row in result]

    return QuartersResponse(
        quarters=quarters,
        total=len(quarters),
        latest=quarters[-1] if quarters else None,
    )


@router.get("/factors")
async def list_factors() -> dict:
    """
    List all factors that will be analyzed.

    Returns the configuration for numerical, categorical, and boolean factors,
    along with available categories for filtering.
    """
    from src.factor_discovery.factor_analyzer import FactorAnalyzer

    return {
        # Factor categories for UI selection
        "categories": FactorAnalyzer.FACTOR_CATEGORIES,
        # All factors with their configurations
        "numerical": [
            {
                "name": name,
                "label": config.get("label", name),
                "thresholds": config.get("thresholds", []),
                "direction": config.get("direction", ">="),
                "category": config.get("category", "scores"),
            }
            for name, config in FactorAnalyzer.NUMERICAL_FACTORS.items()
        ],
        "categorical": [
            {
                "name": name,
                "label": config.get("label", name),
                "categories": config.get("categories", []),
                "category": config.get("category", "scores"),
            }
            for name, config in FactorAnalyzer.CATEGORICAL_FACTORS.items()
        ],
        "boolean": [
            {
                "name": name,
                "label": config.get("label", name),
                "positive": config.get("positive", True),
                "category": config.get("category", "boolean"),
            }
            for name, config in FactorAnalyzer.BOOLEAN_FACTORS.items()
        ],
    }
