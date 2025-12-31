"""Elastic Net API routes."""

import asyncio
import concurrent.futures
import json
import logging
from functools import partial
from typing import Any

from fastapi import APIRouter, BackgroundTasks, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from src.database.connection import get_db_manager
from src.ml_models import (
    ElasticNetConfig,
    ElasticNetModel,
    ELASTIC_NET_FEATURES,
    save_elastic_net_result,
    load_elastic_net_result,
)

logger = logging.getLogger(__name__)
router = APIRouter()

# Track running analyses for progress reporting and cancellation
_running_analyses: dict[str, dict] = {}


class ElasticNetRequest(BaseModel):
    """Request to start Elastic Net training."""

    quarters: list[str]
    holding_period: int = 4
    train_end_quarter: str | None = None
    features: list[str] | None = None
    l1_ratios: list[float] | None = None
    cv_folds: int = 5
    winsorize_percentile: float = 0.01


class RunResponse(BaseModel):
    """Response for starting a new run."""

    run_id: str
    status: str
    estimated_duration_seconds: int


class CoefficientResponse(BaseModel):
    """Coefficient result."""

    feature_name: str
    coefficient: float
    coefficient_std: float
    stability_score: float
    importance_rank: int


class ICHistoryResponse(BaseModel):
    """IC history point."""

    quarter: str
    ic: float
    ic_pvalue: float
    n_samples: int


class PredictionResponse(BaseModel):
    """Stock prediction."""

    symbol: str
    predicted_alpha: float
    predicted_rank: int


class ElasticNetResultResponse(BaseModel):
    """Full result response."""

    run_id: str
    status: str
    error_message: str | None
    duration_seconds: float
    # Performance
    train_ic: float | None
    test_ic: float | None
    n_train_samples: int
    n_test_samples: int
    # Model params
    best_alpha: float | None
    best_l1_ratio: float | None
    n_features_selected: int
    # Config
    holding_period: int
    train_end_quarter: str | None
    # Data
    coefficients: list[CoefficientResponse]
    ic_history: list[ICHistoryResponse]
    predictions: list[PredictionResponse]


class RunSummary(BaseModel):
    """Summary of a run for listing."""

    run_id: str
    status: str
    created_at: str
    holding_period: int
    train_ic: float | None
    test_ic: float | None
    n_features_selected: int


@router.post("/run", response_model=RunResponse)
async def start_training(
    request: ElasticNetRequest,
    background_tasks: BackgroundTasks,
) -> RunResponse:
    """
    Start Elastic Net model training.

    This runs in the background. Use /progress/{run_id} to track progress
    and /results/{run_id} to get results when complete.
    """
    import uuid

    run_id = str(uuid.uuid4())[:8]

    # Initialize tracking
    _running_analyses[run_id] = {
        "status": "starting",
        "progress": 0,
        "stage": "initializing",
        "message": "Starting training...",
        "cancelled": False,
    }

    # Build config - pass run_id so model uses the same ID as API
    config = ElasticNetConfig(
        quarters=request.quarters,
        holding_period=request.holding_period,
        train_end_quarter=request.train_end_quarter,
        features=request.features or ELASTIC_NET_FEATURES.copy(),
        l1_ratios=request.l1_ratios or [0.1, 0.5, 0.7, 0.9, 0.95, 1.0],
        cv_folds=request.cv_folds,
        winsorize_percentile=request.winsorize_percentile,
        run_id=run_id,  # Use same run_id as API
    )

    # Define progress callback
    def progress_callback(update: dict):
        if run_id in _running_analyses:
            _running_analyses[run_id].update({
                "stage": update.get("stage", ""),
                "progress": update.get("percent", 0),
                "message": update.get("message", ""),
            })

    # Define cancel check
    def cancel_check() -> bool:
        return _running_analyses.get(run_id, {}).get("cancelled", False)

    # Run training in background
    async def run_training():
        try:
            model = ElasticNetModel(
                progress_callback=progress_callback,
                cancel_check=cancel_check,
            )

            # Run in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            with concurrent.futures.ThreadPoolExecutor() as pool:
                result = await loop.run_in_executor(pool, partial(model.train, config))

            # Save result to database
            save_elastic_net_result(result)

            _running_analyses[run_id]["status"] = result.status
            _running_analyses[run_id]["progress"] = 100
            _running_analyses[run_id]["result_run_id"] = result.run_id

            logger.info(f"Elastic Net {run_id} completed with status {result.status}")

        except Exception as e:
            logger.exception(f"Elastic Net {run_id} failed: {e}")
            _running_analyses[run_id]["status"] = "failed"
            _running_analyses[run_id]["error"] = str(e)

    background_tasks.add_task(run_training)

    # Estimate duration
    num_quarters = len(request.quarters)
    estimated_seconds = max(30, num_quarters * 5)

    return RunResponse(
        run_id=run_id,
        status="running",
        estimated_duration_seconds=estimated_seconds,
    )


@router.get("/progress/{run_id}")
async def get_progress(run_id: str):
    """
    Get progress updates for a running training via SSE.

    Returns Server-Sent Events stream with progress updates.
    """

    async def event_generator():
        last_progress = -1
        last_stage = ""

        while True:
            if run_id not in _running_analyses:
                # Check if run exists in database (completed before we started watching)
                try:
                    result = load_elastic_net_result(run_id)
                    if result:
                        yield f"data: {json.dumps({'status': result['run']['status'], 'progress': 100, 'stage': 'complete'})}\n\n"
                    else:
                        yield f"data: {json.dumps({'status': 'not_found', 'error': 'Run not found'})}\n\n"
                except Exception:
                    yield f"data: {json.dumps({'status': 'not_found', 'error': 'Run not found'})}\n\n"
                break

            info = _running_analyses[run_id]
            current_progress = info.get("progress", 0)
            current_stage = info.get("stage", "")

            # Send update if progress OR stage changed
            if current_progress != last_progress or current_stage != last_stage:
                last_progress = current_progress
                last_stage = current_stage
                yield f"data: {json.dumps(info)}\n\n"

            # Check if complete
            if info.get("status") in ("completed", "failed", "cancelled"):
                break

            await asyncio.sleep(0.3)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )


@router.get("/results/{run_id}", response_model=ElasticNetResultResponse)
async def get_results(run_id: str) -> ElasticNetResultResponse:
    """
    Get complete results for an Elastic Net run.
    """
    try:
        data = load_elastic_net_result(run_id)
    except ValueError:
        raise HTTPException(status_code=404, detail=f"Run {run_id} not found")

    run = data["run"]
    config_json = json.loads(run.get("config_json", "{}"))

    return ElasticNetResultResponse(
        run_id=run["id"],
        status=run["status"],
        error_message=run.get("error_message"),
        duration_seconds=run.get("duration_seconds", 0),
        train_ic=run.get("train_ic"),
        test_ic=run.get("test_ic"),
        n_train_samples=run.get("n_train_samples", 0),
        n_test_samples=run.get("n_test_samples", 0),
        best_alpha=run.get("best_alpha"),
        best_l1_ratio=run.get("best_l1_ratio"),
        n_features_selected=run.get("n_features_selected", 0),
        holding_period=run.get("holding_period", 4),
        train_end_quarter=run.get("train_end_quarter"),
        coefficients=[
            CoefficientResponse(
                feature_name=c["feature_name"],
                coefficient=c["coefficient"],
                coefficient_std=c.get("coefficient_std", 0),
                stability_score=c.get("stability_score", 0),
                importance_rank=c.get("importance_rank", 0),
            )
            for c in data["coefficients"]
        ],
        ic_history=[
            ICHistoryResponse(
                quarter=ic["quarter"],
                ic=ic["ic"],
                ic_pvalue=ic.get("ic_pvalue", 0),
                n_samples=ic.get("n_samples", 0),
            )
            for ic in data["ic_history"]
        ],
        predictions=[
            PredictionResponse(
                symbol=p["symbol"],
                predicted_alpha=p["predicted_alpha"],
                predicted_rank=p["predicted_rank"],
            )
            for p in data["predictions"]
        ],
    )


@router.get("/coefficients/{run_id}")
async def get_coefficients(run_id: str, limit: int = 50) -> list[CoefficientResponse]:
    """
    Get coefficient table for a run.
    """
    try:
        data = load_elastic_net_result(run_id)
    except ValueError:
        raise HTTPException(status_code=404, detail=f"Run {run_id} not found")

    return [
        CoefficientResponse(
            feature_name=c["feature_name"],
            coefficient=c["coefficient"],
            coefficient_std=c.get("coefficient_std", 0),
            stability_score=c.get("stability_score", 0),
            importance_rank=c.get("importance_rank", 0),
        )
        for c in data["coefficients"][:limit]
    ]


@router.get("/ic-history/{run_id}")
async def get_ic_history(run_id: str) -> list[ICHistoryResponse]:
    """
    Get IC over time for a run.
    """
    try:
        data = load_elastic_net_result(run_id)
    except ValueError:
        raise HTTPException(status_code=404, detail=f"Run {run_id} not found")

    return [
        ICHistoryResponse(
            quarter=ic["quarter"],
            ic=ic["ic"],
            ic_pvalue=ic.get("ic_pvalue", 0),
            n_samples=ic.get("n_samples", 0),
        )
        for ic in data["ic_history"]
    ]


@router.get("/predictions/{run_id}")
async def get_predictions(run_id: str, limit: int = 100) -> list[PredictionResponse]:
    """
    Get stock predictions for a run.
    """
    try:
        data = load_elastic_net_result(run_id)
    except ValueError:
        raise HTTPException(status_code=404, detail=f"Run {run_id} not found")

    return [
        PredictionResponse(
            symbol=p["symbol"],
            predicted_alpha=p["predicted_alpha"],
            predicted_rank=p["predicted_rank"],
        )
        for p in data["predictions"][:limit]
    ]


@router.get("/history")
async def list_runs(limit: int = 50) -> list[RunSummary]:
    """
    List past Elastic Net runs.
    """
    db = get_db_manager()

    with db.get_connection() as conn:
        result = conn.execute(
            """
            SELECT id, status, created_at, holding_period,
                   train_ic, test_ic, n_features_selected
            FROM ml_model_runs
            WHERE model_type = 'elastic_net'
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()

    return [
        RunSummary(
            run_id=row[0],
            status=row[1],
            created_at=str(row[2]) if row[2] else "",
            holding_period=row[3] or 4,
            train_ic=row[4],
            test_ic=row[5],
            n_features_selected=row[6] or 0,
        )
        for row in result
    ]


@router.post("/cancel/{run_id}")
async def cancel_run(run_id: str) -> dict:
    """
    Cancel a running training.
    """
    if run_id not in _running_analyses:
        raise HTTPException(
            status_code=404, detail=f"Run {run_id} not found or already completed"
        )

    _running_analyses[run_id]["cancelled"] = True
    _running_analyses[run_id]["status"] = "cancelling"

    return {"status": "cancelling", "run_id": run_id}


@router.get("/features")
async def list_features() -> dict:
    """
    List available features for Elastic Net.
    """
    return {
        "features": ELASTIC_NET_FEATURES,
        "total": len(ELASTIC_NET_FEATURES),
    }
