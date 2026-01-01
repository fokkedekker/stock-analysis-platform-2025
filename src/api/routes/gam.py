"""GAM (Generalized Additive Model) API routes."""

import asyncio
import concurrent.futures
import json
import logging
from functools import partial

from fastapi import APIRouter, BackgroundTasks, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from src.database.connection import get_db_manager
from src.ml_models import (
    GAMConfig,
    GAMModel,
    ELASTIC_NET_FEATURES,
    save_gam_result,
    load_gam_result,
)
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


class GAMRequest(BaseModel):
    """Request to start GAM training."""

    quarters: list[str]
    holding_period: int = 4
    train_end_quarter: str | None = None
    features: list[str] | None = None
    n_splines: int = 15
    lam: float = 0.6
    cv_folds: int = 5
    winsorize_percentile: float = 0.01
    target_type: str = "raw"


class RunResponse(BaseModel):
    """Response for starting a new run."""

    run_id: str
    status: str
    estimated_duration_seconds: int


class PartialDependenceResponse(BaseModel):
    """Partial dependence curve for one feature."""

    feature_name: str
    x_values: list[float]
    y_values: list[float]
    optimal_min: float | None
    optimal_max: float | None
    peak_x: float | None
    peak_y: float
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


class GAMResultResponse(BaseModel):
    """Full result response."""

    run_id: str
    status: str
    error_message: str | None
    duration_seconds: float
    # Performance
    train_ic: float | None
    test_ic: float | None
    train_r2: float | None
    n_train_samples: int
    n_test_samples: int
    # Model params
    n_features: int
    best_lam: float | None
    # Config
    holding_period: int
    train_end_quarter: str | None
    # Data
    partial_dependences: list[PartialDependenceResponse]
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
    n_features: int


@router.post("/run", response_model=RunResponse)
async def start_training(
    request: GAMRequest,
    background_tasks: BackgroundTasks,
) -> RunResponse:
    """
    Start GAM model training.

    This runs in the background. Use /progress/{run_id} to track progress
    and /results/{run_id} to get results when complete.
    """
    import uuid
    from src.factor_discovery.sector_calculator import ensure_sector_returns_current

    run_id = str(uuid.uuid4())[:8]

    # Ensure sector returns are calculated if needed
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
    config = GAMConfig(
        quarters=request.quarters,
        holding_period=request.holding_period,
        train_end_quarter=request.train_end_quarter,
        features=request.features or ELASTIC_NET_FEATURES.copy(),
        n_splines=request.n_splines,
        lam=request.lam,
        cv_folds=request.cv_folds,
        winsorize_percentile=request.winsorize_percentile,
        run_id=run_id,
        target_type=request.target_type,
    )

    def progress_callback(update: dict):
        if run_id in _running_analyses:
            _running_analyses[run_id].update({
                "stage": update.get("stage", ""),
                "progress": update.get("percent", 0),
                "message": update.get("message", ""),
            })

    def cancel_check() -> bool:
        return _running_analyses.get(run_id, {}).get("cancelled", False)

    async def run_training():
        try:
            model = GAMModel(
                progress_callback=progress_callback,
                cancel_check=cancel_check,
            )

            loop = asyncio.get_event_loop()
            with concurrent.futures.ThreadPoolExecutor() as pool:
                result = await loop.run_in_executor(pool, partial(model.train, config))

                await loop.run_in_executor(
                    pool,
                    partial(save_gam_result, result, progress_callback=progress_callback)
                )

            _running_analyses[run_id]["status"] = result.status
            _running_analyses[run_id]["progress"] = 100
            _running_analyses[run_id]["result_run_id"] = result.run_id

            logger.info(f"GAM {run_id} completed with status {result.status}")

        except Exception as e:
            logger.exception(f"GAM {run_id} failed: {e}")
            _running_analyses[run_id]["status"] = "failed"
            _running_analyses[run_id]["error"] = str(e)

    background_tasks.add_task(run_training)

    num_quarters = len(request.quarters)
    estimated_seconds = max(45, num_quarters * 8)  # GAM is slower than Elastic Net

    return RunResponse(
        run_id=run_id,
        status="running",
        estimated_duration_seconds=estimated_seconds,
    )


@router.get("/progress/{run_id}")
async def get_progress(run_id: str):
    """
    Get progress updates for a running training via SSE.
    """

    async def event_generator():
        last_progress = -1
        last_stage = ""

        while True:
            if run_id not in _running_analyses:
                try:
                    result = load_gam_result(run_id)
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


@router.get("/results/{run_id}", response_model=GAMResultResponse)
async def get_results(run_id: str) -> GAMResultResponse:
    """
    Get complete results for a GAM run.
    """
    try:
        data = load_gam_result(run_id)
    except ValueError:
        raise HTTPException(status_code=404, detail=f"Run {run_id} not found")

    run = data["run"]
    config_json = json.loads(run.get("config_json", "{}"))

    return GAMResultResponse(
        run_id=run["id"],
        status=run["status"],
        error_message=run.get("error_message"),
        duration_seconds=run.get("duration_seconds", 0),
        train_ic=run.get("train_ic"),
        test_ic=run.get("test_ic"),
        train_r2=run.get("train_r2"),
        n_train_samples=run.get("n_train_samples", 0),
        n_test_samples=run.get("n_test_samples", 0),
        n_features=run.get("n_features_selected", 0),
        best_lam=run.get("best_alpha"),  # lam is stored in best_alpha column
        holding_period=run.get("holding_period", 4),
        train_end_quarter=run.get("train_end_quarter"),
        partial_dependences=[
            PartialDependenceResponse(
                feature_name=pd["feature_name"],
                x_values=pd["x_values"],
                y_values=pd["y_values"],
                optimal_min=pd.get("optimal_min"),
                optimal_max=pd.get("optimal_max"),
                peak_x=pd.get("peak_x"),
                peak_y=pd["peak_y"],
                importance_rank=pd.get("importance_rank", 0),
            )
            for pd in data["partial_dependences"]
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


@router.get("/partial-dependence/{run_id}")
async def get_partial_dependence(run_id: str, limit: int = 50) -> list[PartialDependenceResponse]:
    """
    Get partial dependence curves for a run.
    """
    try:
        data = load_gam_result(run_id)
    except ValueError:
        raise HTTPException(status_code=404, detail=f"Run {run_id} not found")

    return [
        PartialDependenceResponse(
            feature_name=pd["feature_name"],
            x_values=pd["x_values"],
            y_values=pd["y_values"],
            optimal_min=pd.get("optimal_min"),
            optimal_max=pd.get("optimal_max"),
            peak_x=pd.get("peak_x"),
            peak_y=pd["peak_y"],
            importance_rank=pd.get("importance_rank", 0),
        )
        for pd in data["partial_dependences"][:limit]
    ]


@router.get("/ic-history/{run_id}")
async def get_ic_history(run_id: str) -> list[ICHistoryResponse]:
    """
    Get IC over time for a run.
    """
    try:
        data = load_gam_result(run_id)
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
        data = load_gam_result(run_id)
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
    List past GAM runs.
    """
    db = get_db_manager()

    with db.get_connection() as conn:
        result = conn.execute(
            """
            SELECT id, status, created_at, holding_period,
                   train_ic, test_ic, n_features_selected
            FROM ml_model_runs
            WHERE model_type = 'gam'
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
            n_features=row[6] or 0,
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
    List available features for GAM.
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
    Get buy/sell signals for a stock based on GAM model predictions.

    Uses LIVE calculation of model scores (same as Pipeline apply-model endpoint).
    Implements rolling hold model for position management.
    """
    from src.api.routes.screener import apply_model_to_current

    try:
        model_data = load_gam_result(run_id)
    except ValueError:
        raise HTTPException(status_code=404, detail=f"Run {run_id} not found")

    run = model_data["run"]
    holding_period = run.get("holding_period", 4)
    created_at = run.get("created_at", "")

    date_str = str(created_at)[:10] if created_at else ""
    model_name = f"GAM - {date_str}"

    db = get_db_manager()

    with db.get_connection() as conn:
        # Get quarters where this specific symbol has analysis data
        # Only go back to 2020 since that's what the chart shows
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

        # Use the EXACT SAME apply-model logic as the Pipeline page
        # This guarantees rankings match between Pipeline and signals
        symbol_data: dict[str, dict] = {}
        quarter_totals: dict[str, int] = {}

        for quarter in symbol_quarters:
            # Call the same apply-model endpoint logic
            result = await apply_model_to_current(
                run_id=run_id,
                top_percentile=100,  # Get all stocks to calculate rank
                min_score=None,
                quarter=quarter,
                limit=1000,
            )

            stocks = result.get("stocks", [])
            total_scored = result.get("total_scored", len(stocks))
            quarter_totals[quarter] = total_scored

            # Find this symbol's rank
            symbol_upper = symbol.upper()
            for stock in stocks:
                if stock.get("symbol", "").upper() == symbol_upper:
                    symbol_data[quarter] = {
                        "score": stock.get("ml_score", 0),
                        "rank": stock.get("ml_rank", 0),
                    }
                    break

        all_quarters = symbol_quarters

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

        signals: list[MLModelSignal] = []

        in_position = False
        buy_quarter: str | None = None
        last_match_quarter: str | None = None
        dropped_out = False  # Track if stock dropped out of top N% since buy
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
                    # Start new position
                    in_position = True
                    buy_quarter = quarter
                    last_match_quarter = quarter
                    dropped_out = False
                    last_pred_alpha = pred["score"]
                    last_pred_rank = rank
                else:
                    # Already in position - check if we should close old and start new
                    if dropped_out and last_match_quarter:
                        planned_sell = _add_quarters(last_match_quarter, holding_period)
                        if quarter >= planned_sell:
                            # Stock dropped out, we're past planned sell, now matches again
                            # Close old position and start new one
                            signals.append(_create_ml_signal(
                                buy_quarter, planned_sell, stock_prices, spy_prices,
                                current_stock_price, current_spy_price,
                                last_pred_alpha, last_pred_rank,
                            ))
                            # Start new position
                            buy_quarter = quarter
                            last_match_quarter = quarter
                            dropped_out = False
                            last_pred_alpha = pred["score"]
                            last_pred_rank = rank
                        else:
                            # Stock came back before planned sell - extend position
                            last_match_quarter = quarter
                            dropped_out = False
                            last_pred_alpha = pred["score"]
                            last_pred_rank = rank
                    else:
                        # Continuous match - just extend position
                        last_match_quarter = quarter
                        last_pred_alpha = pred["score"]
                        last_pred_rank = rank
            else:
                # Not matched
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

        if in_position and buy_quarter and last_match_quarter:
            planned_sell = _add_quarters(last_match_quarter, holding_period)
            signals.append(_create_ml_signal(
                buy_quarter, planned_sell, stock_prices, spy_prices,
                current_stock_price, current_spy_price,
                last_pred_alpha, last_pred_rank,
            ))

    valid_trades = [s for s in signals if s.stock_return is not None]
    num_trades = len(valid_trades)

    total_return = None
    total_alpha = None
    avg_alpha = None
    win_rate = None

    if num_trades > 0:
        # Calculate compound stock return
        stock_compound = 1.0
        for s in valid_trades:
            stock_compound *= (1 + s.stock_return / 100)
        total_return = round((stock_compound - 1) * 100, 2)

        # Calculate compound SPY return for consistent alpha calculation
        spy_compound = 1.0
        for s in valid_trades:
            if s.spy_return is not None:
                spy_compound *= (1 + s.spy_return / 100)
        total_spy_return = (spy_compound - 1) * 100

        # Total alpha = compound stock return - compound SPY return
        total_alpha = round(total_return - total_spy_return, 2)

        # Avg alpha per trade and win rate use individual trade alphas
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
