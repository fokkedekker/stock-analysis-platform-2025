"""LightGBM API routes."""

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
    LightGBMConfig,
    LightGBMModel,
    ELASTIC_NET_FEATURES,
    save_lightgbm_result,
    load_lightgbm_result,
)

# Import shared signal types from elastic_net
from src.api.routes.elastic_net import (
    MLModelSignal,
    MLModelSignalsResponse,
    _add_quarters,
    _quarter_to_date,
    _create_ml_signal,
)

logger = logging.getLogger(__name__)
router = APIRouter()

# Track running analyses for progress reporting and cancellation
_running_analyses: dict[str, dict] = {}


class LightGBMRequest(BaseModel):
    """Request to start LightGBM training."""

    quarters: list[str]
    holding_period: int = 4
    train_end_quarter: str | None = None
    features: list[str] | None = None
    n_optuna_trials: int = 50
    winsorize_percentile: float = 0.01
    target_type: str = "raw"


class RunResponse(BaseModel):
    """Response for starting a new run."""

    run_id: str
    status: str
    estimated_duration_seconds: int


class FeatureImportanceResponse(BaseModel):
    """Feature importance result."""

    feature_name: str
    importance_gain: float
    importance_split: float
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


class LightGBMResultResponse(BaseModel):
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
    best_params: dict[str, Any]
    n_features_selected: int
    # Config
    holding_period: int
    train_end_quarter: str | None
    # Data
    feature_importances: list[FeatureImportanceResponse]
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
    request: LightGBMRequest,
    background_tasks: BackgroundTasks,
) -> RunResponse:
    """
    Start LightGBM model training.

    This runs in the background. Use /progress/{run_id} to track progress
    and /results/{run_id} to get results when complete.
    """
    import uuid
    from src.factor_discovery.sector_calculator import ensure_sector_returns_current

    run_id = str(uuid.uuid4())[:8]

    # Ensure sector returns are calculated if needed for target_type
    if request.target_type in ("sector_adjusted", "full_adjusted"):
        ensure_sector_returns_current(holding_periods=[request.holding_period])

    # Initialize tracking
    _running_analyses[run_id] = {
        "status": "starting",
        "progress": 0,
        "stage": "initializing",
        "message": "Starting training...",
        "cancelled": False,
    }

    # Build config
    config = LightGBMConfig(
        quarters=request.quarters,
        holding_period=request.holding_period,
        train_end_quarter=request.train_end_quarter,
        features=request.features or ELASTIC_NET_FEATURES.copy(),
        n_optuna_trials=request.n_optuna_trials,
        winsorize_percentile=request.winsorize_percentile,
        run_id=run_id,
        target_type=request.target_type,
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
            model = LightGBMModel(
                progress_callback=progress_callback,
                cancel_check=cancel_check,
            )

            # Run in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            with concurrent.futures.ThreadPoolExecutor() as pool:
                result = await loop.run_in_executor(pool, partial(model.train, config))

                # Save result to database
                await loop.run_in_executor(
                    pool,
                    partial(save_lightgbm_result, result, progress_callback=progress_callback)
                )

            _running_analyses[run_id]["status"] = result.status
            _running_analyses[run_id]["progress"] = 100
            _running_analyses[run_id]["result_run_id"] = result.run_id

            logger.info(f"LightGBM {run_id} completed with status {result.status}")

        except Exception as e:
            logger.exception(f"LightGBM {run_id} failed: {e}")
            _running_analyses[run_id]["status"] = "failed"
            _running_analyses[run_id]["error"] = str(e)

    background_tasks.add_task(run_training)

    # Estimate duration (LightGBM with Optuna is slower)
    num_quarters = len(request.quarters)
    estimated_seconds = max(60, num_quarters * 8 + request.n_optuna_trials * 2)

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
                # Check if run exists in database
                try:
                    result = load_lightgbm_result(run_id)
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

            if current_progress != last_progress or current_stage != last_stage:
                last_progress = current_progress
                last_stage = current_stage
                yield f"data: {json.dumps(info)}\n\n"

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


@router.get("/results/{run_id}", response_model=LightGBMResultResponse)
async def get_results(run_id: str) -> LightGBMResultResponse:
    """
    Get complete results for a LightGBM run.
    """
    try:
        data = load_lightgbm_result(run_id)
    except ValueError:
        raise HTTPException(status_code=404, detail=f"Run {run_id} not found")

    run = data["run"]
    config_json = json.loads(run.get("config_json", "{}"))

    return LightGBMResultResponse(
        run_id=run["id"],
        status=run["status"],
        error_message=run.get("error_message"),
        duration_seconds=run.get("duration_seconds", 0),
        train_ic=run.get("train_ic"),
        test_ic=run.get("test_ic"),
        n_train_samples=run.get("n_train_samples", 0),
        n_test_samples=run.get("n_test_samples", 0),
        best_params=config_json.get("best_params", {}),
        n_features_selected=run.get("n_features_selected", 0),
        holding_period=run.get("holding_period", 4),
        train_end_quarter=run.get("train_end_quarter"),
        feature_importances=[
            FeatureImportanceResponse(
                feature_name=fi["feature_name"],
                importance_gain=fi.get("importance_gain", 0),
                importance_split=fi.get("importance_split", 0),
                importance_rank=fi.get("importance_rank", 0),
            )
            for fi in data["feature_importances"]
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


@router.get("/feature-importance/{run_id}")
async def get_feature_importance(run_id: str, limit: int = 50) -> list[FeatureImportanceResponse]:
    """
    Get feature importance table for a run.
    """
    try:
        data = load_lightgbm_result(run_id)
    except ValueError:
        raise HTTPException(status_code=404, detail=f"Run {run_id} not found")

    return [
        FeatureImportanceResponse(
            feature_name=fi["feature_name"],
            importance_gain=fi.get("importance_gain", 0),
            importance_split=fi.get("importance_split", 0),
            importance_rank=fi.get("importance_rank", 0),
        )
        for fi in data["feature_importances"][:limit]
    ]


@router.get("/ic-history/{run_id}")
async def get_ic_history(run_id: str) -> list[ICHistoryResponse]:
    """
    Get IC over time for a run.
    """
    try:
        data = load_lightgbm_result(run_id)
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
        data = load_lightgbm_result(run_id)
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
    List past LightGBM runs.
    """
    db = get_db_manager()

    with db.get_connection() as conn:
        result = conn.execute(
            """
            SELECT id, status, created_at, holding_period,
                   train_ic, test_ic, n_features_selected
            FROM ml_model_runs
            WHERE model_type = 'lightgbm'
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
    List available features for LightGBM.
    """
    return {
        "features": ELASTIC_NET_FEATURES,
        "total": len(ELASTIC_NET_FEATURES),
    }


# ============================================================================
# Stock Page Signals
# ============================================================================


@router.get("/signals/{run_id}/{symbol}", response_model=MLModelSignalsResponse)
async def get_model_signals(
    run_id: str,
    symbol: str,
    top_percentile: int = 20,
) -> MLModelSignalsResponse:
    """
    Get buy/sell signals for a stock based on LightGBM model predictions.

    Uses LIVE calculation of model scores (same as Pipeline apply-model endpoint).
    Implements a "rolling hold" model:
    - Buy when stock first enters top N% by model score
    - If stock keeps qualifying in subsequent quarters, extend the sell date
    - Sell when stock stops qualifying AND holding period expires
    """
    from src.api.routes.screener import apply_model_to_current

    # Load model data
    try:
        model_data = load_lightgbm_result(run_id)
    except ValueError:
        raise HTTPException(status_code=404, detail=f"Run {run_id} not found")

    run = model_data["run"]
    holding_period = run.get("holding_period", 4)
    created_at = run.get("created_at", "")

    # Format model name
    date_str = str(created_at)[:10] if created_at else ""
    model_name = f"LightGBM - {date_str}"

    db = get_db_manager()

    with db.get_connection() as conn:
        # Get quarters where this symbol has analysis data (2020+)
        symbol_quarters_result = conn.execute(
            """
            SELECT DISTINCT analysis_quarter
            FROM roic_quality_results
            WHERE UPPER(symbol) = UPPER(?)
            AND analysis_quarter >= '2020Q1'
            ORDER BY analysis_quarter
            """,
            (symbol,),
        ).fetchall()
        symbol_quarters = [row[0] for row in symbol_quarters_result]

        if not symbol_quarters:
            return MLModelSignalsResponse(
                symbol=symbol.upper(),
                run_id=run_id,
                model_name=model_name,
                holding_period=holding_period,
                signals=[],
                total_return=None,
                total_alpha=None,
                avg_alpha_per_trade=None,
                num_trades=0,
                win_rate=None,
            )

        # Use the SAME apply-model logic as Pipeline page
        symbol_data: dict[str, dict] = {}
        quarter_totals: dict[str, int] = {}

        for quarter in symbol_quarters:
            result = await apply_model_to_current(
                run_id=run_id,
                top_percentile=100,
                min_score=None,
                quarter=quarter,
                limit=1000,
            )

            stocks = result.get("stocks", [])
            total_scored = result.get("total_scored", len(stocks))
            quarter_totals[quarter] = total_scored

            symbol_upper = symbol.upper()
            for stock in stocks:
                if stock.get("symbol", "").upper() == symbol_upper:
                    symbol_data[quarter] = {
                        "score": stock.get("ml_score", 0),
                        "rank": stock.get("ml_rank", 0),
                    }
                    break

        all_quarters = symbol_quarters

        # Load stock prices
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

        current_stock_price = None
        if stock_prices:
            latest_quarter = max(stock_prices.keys())
            current_stock_price = stock_prices[latest_quarter]

        # Load SPY prices
        spy_result = conn.execute(
            """
            SELECT quarter, price
            FROM spy_prices
            WHERE price IS NOT NULL AND price > 0
            """
        ).fetchall()
        spy_prices = {row[0]: float(row[1]) for row in spy_result}

        current_spy_price = None
        if spy_prices:
            latest_spy_quarter = max(spy_prices.keys())
            current_spy_price = spy_prices[latest_spy_quarter]

        # Build signals using rolling hold model
        signals: list[MLModelSignal] = []

        in_position = False
        buy_quarter: str | None = None
        last_match_quarter: str | None = None
        dropped_out = False
        last_pred_alpha: float = 0.0
        last_pred_rank: int = 0

        for quarter in all_quarters:
            total_stocks = quarter_totals.get(quarter, 0)
            pred = symbol_data.get(quarter)

            if not pred or total_stocks == 0:
                if in_position and last_match_quarter:
                    dropped_out = True
                    planned_sell = _add_quarters(last_match_quarter, holding_period)
                    if quarter >= planned_sell:
                        signals.append(_create_ml_signal(
                            buy_quarter, planned_sell, stock_prices, spy_prices,
                            current_stock_price, current_spy_price,
                            last_pred_alpha, last_pred_rank,
                        ))
                        in_position = False
                        buy_quarter = None
                        last_match_quarter = None
                        dropped_out = False
                continue

            rank = pred["rank"]
            percentile = (rank / total_stocks) * 100
            matched = percentile <= top_percentile

            if matched:
                if not in_position:
                    in_position = True
                    buy_quarter = quarter
                    last_match_quarter = quarter
                    dropped_out = False
                    last_pred_alpha = pred["score"]
                    last_pred_rank = rank
                else:
                    if dropped_out and last_match_quarter:
                        planned_sell = _add_quarters(last_match_quarter, holding_period)
                        if quarter >= planned_sell:
                            signals.append(_create_ml_signal(
                                buy_quarter, planned_sell, stock_prices, spy_prices,
                                current_stock_price, current_spy_price,
                                last_pred_alpha, last_pred_rank,
                            ))
                            buy_quarter = quarter
                            last_match_quarter = quarter
                            dropped_out = False
                            last_pred_alpha = pred["score"]
                            last_pred_rank = rank
                        else:
                            last_match_quarter = quarter
                            dropped_out = False
                            last_pred_alpha = pred["score"]
                            last_pred_rank = rank
                    else:
                        last_match_quarter = quarter
                        last_pred_alpha = pred["score"]
                        last_pred_rank = rank
            else:
                if in_position and last_match_quarter:
                    dropped_out = True
                    planned_sell = _add_quarters(last_match_quarter, holding_period)
                    if quarter >= planned_sell:
                        signals.append(_create_ml_signal(
                            buy_quarter, planned_sell, stock_prices, spy_prices,
                            current_stock_price, current_spy_price,
                            last_pred_alpha, last_pred_rank,
                        ))
                        in_position = False
                        buy_quarter = None
                        last_match_quarter = None
                        dropped_out = False

        # Handle position still open
        if in_position and buy_quarter and last_match_quarter:
            planned_sell = _add_quarters(last_match_quarter, holding_period)
            signals.append(_create_ml_signal(
                buy_quarter, planned_sell, stock_prices, spy_prices,
                current_stock_price, current_spy_price,
                last_pred_alpha, last_pred_rank,
            ))

    # Calculate aggregate stats
    valid_trades = [s for s in signals if s.stock_return is not None]
    num_trades = len(valid_trades)

    total_return = None
    total_alpha = None
    avg_alpha = None
    win_rate = None

    if num_trades > 0:
        stock_compound = 1.0
        for s in valid_trades:
            stock_compound *= (1 + s.stock_return / 100)
        total_return = round((stock_compound - 1) * 100, 2)

        spy_compound = 1.0
        for s in valid_trades:
            if s.spy_return is not None:
                spy_compound *= (1 + s.spy_return / 100)
        total_spy_return = (spy_compound - 1) * 100

        total_alpha = round(total_return - total_spy_return, 2)

        alphas = [s.alpha for s in valid_trades if s.alpha is not None]
        if alphas:
            avg_alpha = round(sum(alphas) / len(alphas), 2)
            win_rate = round(len([a for a in alphas if a > 0]) / len(alphas) * 100, 1)

    return MLModelSignalsResponse(
        symbol=symbol.upper(),
        run_id=run_id,
        model_name=model_name,
        holding_period=holding_period,
        signals=signals,
        total_return=total_return,
        total_alpha=total_alpha,
        avg_alpha_per_trade=avg_alpha,
        num_trades=num_trades,
        win_rate=win_rate,
    )
