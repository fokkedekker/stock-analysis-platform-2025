"""Grid search simulation runner with multiprocessing support."""

import logging
import multiprocessing
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import Callable
from uuid import uuid4


from .data_preloader import SimulationDataPreloader
from .models import (
    GridSearchProgress,
    GridSearchRequest,
    GridSearchResult,
    HoldingPeriodAggregate,
    SimulationResult,
    StrategyAggregate,
    StrategyConfig,
    StrategyHoldingPeriod,
)
from .strategy_builder import StrategyBuilder

logger = logging.getLogger(__name__)


def _compute_aggregations(
    results: list[SimulationResult],
) -> tuple[list[StrategyAggregate], list[HoldingPeriodAggregate]]:
    """Compute aggregated results by strategy and by holding period."""
    from collections import defaultdict

    # Only include results with stocks
    valid_results = [r for r in results if r.stock_count > 0]

    # Aggregate by strategy
    by_strategy_map: dict[str, list[SimulationResult]] = defaultdict(list)
    for r in valid_results:
        by_strategy_map[r.strategy_id].append(r)

    strategy_aggregates = []
    for strategy_id, sims in by_strategy_map.items():
        if not sims:
            continue

        # Compute holding period breakdown for this strategy
        by_hold_for_strategy: dict[int, list[SimulationResult]] = defaultdict(list)
        for s in sims:
            by_hold_for_strategy[s.holding_period].append(s)

        holding_periods = []
        for hold_period, hold_sims in sorted(by_hold_for_strategy.items()):
            if hold_sims:
                holding_periods.append(
                    StrategyHoldingPeriod(
                        holding_period=hold_period,
                        avg_alpha=round(sum(s.alpha for s in hold_sims) / len(hold_sims), 2),
                        avg_return=round(sum(s.portfolio_return for s in hold_sims) / len(hold_sims), 2),
                        simulation_count=len(hold_sims),
                    )
                )

        # Find best holding period for this strategy
        best_hold = None
        if holding_periods:
            best_hp = max(holding_periods, key=lambda x: x.avg_alpha)
            best_hold = best_hp.holding_period

        strategy_aggregates.append(
            StrategyAggregate(
                strategy_id=strategy_id,
                strategy_name=sims[0].strategy_name,
                strategy_config=sims[0].strategy_config,
                simulation_count=len(sims),
                avg_alpha=round(sum(s.alpha for s in sims) / len(sims), 2),
                avg_return=round(sum(s.portfolio_return for s in sims) / len(sims), 2),
                avg_win_rate=round(sum(s.win_rate for s in sims) / len(sims), 1),
                avg_stock_count=round(sum(s.stock_count for s in sims) / len(sims), 0),
                min_alpha=round(min(s.alpha for s in sims), 2),
                max_alpha=round(max(s.alpha for s in sims), 2),
                by_holding_period=holding_periods,
                best_holding_period=best_hold,
            )
        )

    # Sort by average alpha descending
    strategy_aggregates.sort(key=lambda x: x.avg_alpha, reverse=True)

    # Aggregate by holding period (overall across all strategies)
    by_hold_map: dict[int, list[SimulationResult]] = defaultdict(list)
    for r in valid_results:
        by_hold_map[r.holding_period].append(r)

    hold_aggregates = []
    for hold_period, sims in sorted(by_hold_map.items()):
        if not sims:
            continue
        hold_aggregates.append(
            HoldingPeriodAggregate(
                holding_period=hold_period,
                simulation_count=len(sims),
                avg_alpha=round(sum(s.alpha for s in sims) / len(sims), 2),
                avg_return=round(sum(s.portfolio_return for s in sims) / len(sims), 2),
                avg_win_rate=round(sum(s.win_rate for s in sims) / len(sims), 1),
            )
        )

    return strategy_aggregates, hold_aggregates


def _run_single_simulation(
    strategy_dict: dict,
    buy_quarter: str,
    sell_quarter: str,
    holding_period: int,
    preloader_data: dict,
) -> SimulationResult:
    """
    Run a single simulation task.

    This function is designed to be called in a worker process.
    All data is passed as dicts to avoid pickling issues.
    """
    # Reconstruct strategy from dict
    from .models import QualityConfig, StrategyConfig, SurvivalConfig, ValuationConfig

    strategy = StrategyConfig(
        id=strategy_dict["id"],
        name=strategy_dict["name"],
        survival=SurvivalConfig(**strategy_dict["survival"]),
        quality=QualityConfig(**strategy_dict["quality"]),
        valuation=ValuationConfig(**strategy_dict["valuation"]),
    )

    # Reconstruct minimal preloader for this simulation
    analysis_data = preloader_data["analysis_data"]
    price_data = preloader_data["price_data"]
    spy_prices = preloader_data["spy_prices"]

    # Get stocks passing strategy for buy quarter
    if buy_quarter not in analysis_data:
        return SimulationResult(
            strategy_id=strategy.id,
            strategy_name=strategy.name,
            strategy_config=strategy_dict,
            buy_quarter=buy_quarter,
            sell_quarter=sell_quarter,
            holding_period=holding_period,
            stock_count=0,
            symbols=[],
            portfolio_return=0.0,
            benchmark_return=0.0,
            alpha=0.0,
            win_rate=0.0,
            winners=0,
            losers=0,
        )

    # Filter stocks that pass strategy
    quarter_data = analysis_data[buy_quarter]
    mf_total = len(quarter_data)
    mf_rank_threshold = int(mf_total * strategy.valuation.mf_top_pct / 100)

    passing_symbols = []
    for symbol, data in quarter_data.items():
        if _stock_passes_strategy_dict(data, strategy, mf_rank_threshold):
            passing_symbols.append(symbol)

    if not passing_symbols:
        return SimulationResult(
            strategy_id=strategy.id,
            strategy_name=strategy.name,
            strategy_config=strategy_dict,
            buy_quarter=buy_quarter,
            sell_quarter=sell_quarter,
            holding_period=holding_period,
            stock_count=0,
            symbols=[],
            portfolio_return=0.0,
            benchmark_return=0.0,
            alpha=0.0,
            win_rate=0.0,
            winners=0,
            losers=0,
        )

    # Calculate returns
    returns = []
    buy_prices = price_data.get(buy_quarter, {})
    sell_prices = price_data.get(sell_quarter, {})

    for symbol in passing_symbols:
        buy_price = buy_prices.get(symbol)
        sell_price = sell_prices.get(symbol)

        if buy_price and sell_price and buy_price > 0:
            ret = ((sell_price - buy_price) / buy_price) * 100
            returns.append((symbol, ret))

    if not returns:
        return SimulationResult(
            strategy_id=strategy.id,
            strategy_name=strategy.name,
            strategy_config=strategy_dict,
            buy_quarter=buy_quarter,
            sell_quarter=sell_quarter,
            holding_period=holding_period,
            stock_count=0,
            symbols=[],
            portfolio_return=0.0,
            benchmark_return=0.0,
            alpha=0.0,
            win_rate=0.0,
            winners=0,
            losers=0,
        )

    # Calculate portfolio metrics
    portfolio_return = sum(r[1] for r in returns) / len(returns)
    winners = sum(1 for r in returns if r[1] > 0)
    losers = len(returns) - winners

    # Calculate benchmark return
    spy_buy = spy_prices.get(buy_quarter)
    spy_sell = spy_prices.get(sell_quarter)
    if spy_buy and spy_sell and spy_buy > 0:
        benchmark_return = ((spy_sell - spy_buy) / spy_buy) * 100
    else:
        # Default ~2.5% per quarter
        benchmark_return = 2.5 * holding_period

    return SimulationResult(
        strategy_id=strategy.id,
        strategy_name=strategy.name,
        strategy_config=strategy_dict,
        buy_quarter=buy_quarter,
        sell_quarter=sell_quarter,
        holding_period=holding_period,
        stock_count=len(returns),
        symbols=[r[0] for r in returns],
        portfolio_return=round(portfolio_return, 2),
        benchmark_return=round(benchmark_return, 2),
        alpha=round(portfolio_return - benchmark_return, 2),
        win_rate=round(winners / len(returns) * 100, 1) if returns else 0.0,
        winners=winners,
        losers=losers,
    )


def _stock_passes_strategy_dict(
    data: dict,
    strategy: StrategyConfig,
    mf_rank_threshold: int,
) -> bool:
    """Check if a single stock passes the strategy criteria (worker function)."""
    # Stage 1: Survival Gates
    if strategy.survival.altman_enabled:
        zone = data.get("altman_zone")
        # distress = allow all, grey = allow safe+grey, safe = allow safe only
        if strategy.survival.altman_zone == "safe":
            if zone != "safe":
                return False
        elif strategy.survival.altman_zone == "grey":
            if zone not in ("safe", "grey"):
                return False
        # distress: allow all zones (no filtering)

    if strategy.survival.piotroski_enabled:
        score = data.get("piotroski_score")
        if score is None or score < strategy.survival.piotroski_min:
            return False

    # Stage 2: Quality Filter
    if strategy.quality.enabled:
        roic = data.get("roic")
        fcf_5yr = data.get("fcf_positive_5yr")

        if roic is not None and roic >= 0.15 and fcf_5yr:
            quality_label = "compounder"
        elif roic is not None and roic >= 0.08:
            quality_label = "average"
        else:
            quality_label = "weak"

        if strategy.quality.min_quality == "compounder":
            if quality_label != "compounder":
                return False
        elif strategy.quality.min_quality == "average":
            if quality_label not in ("compounder", "average"):
                return False

    # Quality tags
    if strategy.quality.required_tags:
        stock_tags = data.get("quality_tags_set", set())
        if not (stock_tags & set(strategy.quality.required_tags)):
            return False

    if strategy.quality.excluded_tags:
        stock_tags = data.get("quality_tags_set", set())
        if stock_tags & set(strategy.quality.excluded_tags):
            return False

    # Stage 3: Valuation Lenses
    lenses_passed = 0
    lenses_active = 0

    if strategy.valuation.graham_enabled:
        lenses_active += 1
        graham_score = data.get("graham_score")
        if graham_score is not None and graham_score >= strategy.valuation.graham_min:
            lenses_passed += 1

    if strategy.valuation.net_net_enabled:
        lenses_active += 1
        if data.get("trading_below_ncav"):
            lenses_passed += 1

    if strategy.valuation.peg_enabled:
        lenses_active += 1
        peg = data.get("peg_ratio")
        if peg is not None and 0 < peg <= strategy.valuation.max_peg:
            lenses_passed += 1

    if strategy.valuation.magic_formula_enabled:
        lenses_active += 1
        mf_rank = data.get("magic_formula_rank")
        if mf_rank is not None and mf_rank <= mf_rank_threshold:
            lenses_passed += 1

    if strategy.valuation.fama_french_enabled:
        lenses_active += 1
        bm_pct = data.get("book_to_market_percentile")
        threshold = 1.0 - strategy.valuation.ff_top_pct / 100.0
        if bm_pct is not None and bm_pct >= threshold:
            lenses_passed += 1

    if strategy.valuation.strict_mode:
        if lenses_passed != lenses_active:
            return False
    else:
        if lenses_passed < strategy.valuation.min_lenses:
            return False

    return True


class GridSearchRunner:
    """
    Runs grid search simulations using multiprocessing.

    Designed for M2 Mac 8-core:
    - Uses all 8 workers for maximum parallelism
    - Chunks strategies for efficient load distribution
    - Reports progress via callback (every 5 sims, or every 1 for first 20)
    """

    def __init__(
        self,
        preloader: SimulationDataPreloader,
        num_workers: int = 6,
        progress_callback: Callable[[GridSearchProgress], None] | None = None,
        cancel_check: Callable[[], bool] | None = None,
    ):
        """
        Initialize the grid search runner.

        Args:
            preloader: Preloaded data for simulations
            num_workers: Number of parallel workers
            progress_callback: Function to call with progress updates
            cancel_check: Function that returns True if search should be cancelled
        """
        self.preloader = preloader
        self.num_workers = num_workers
        self.progress_callback = progress_callback
        self.cancel_check = cancel_check
        self.search_id = str(uuid4())

    def run(
        self,
        request: GridSearchRequest,
    ) -> GridSearchResult:
        """
        Run all simulations in parallel.

        Returns complete grid search result.
        """
        start_time = datetime.now()

        # Generate all strategies
        strategies = list(
            StrategyBuilder.generate_strategies(request.base_strategy, request.dimensions)
        )

        # Build simulation tasks
        tasks = []
        for strategy in strategies:
            for buy_quarter in request.quarters:
                for hold in request.holding_periods:
                    sell_quarter = self.preloader.get_sell_quarter(buy_quarter, hold)
                    if sell_quarter:
                        tasks.append(
                            (
                                StrategyBuilder.strategy_to_dict(strategy),
                                buy_quarter,
                                sell_quarter,
                                hold,
                            )
                        )

        total = len(tasks)
        if total == 0:
            return GridSearchResult(
                id=self.search_id,
                started_at=start_time,
                completed_at=datetime.now(),
                total_simulations=0,
                completed_simulations=0,
                duration_seconds=0,
                results=[],
                best_by_alpha=[],
                best_by_win_rate=[],
                request_config=request.model_dump(),
            )

        logger.info(f"Starting grid search with {total} simulations using {self.num_workers} workers")

        # Prepare preloader data for workers (picklable dict)
        preloader_data = {
            "analysis_data": self.preloader.analysis_data,
            "price_data": self.preloader.price_data,
            "spy_prices": self.preloader.spy_prices,
        }

        results: list[SimulationResult] = []
        completed = 0
        sim_start_time = time.time()

        # Send initial progress update before starting
        if self.progress_callback:
            self.progress_callback(
                GridSearchProgress(
                    search_id=self.search_id,
                    status="running",
                    total_simulations=total,
                    completed=0,
                    current_strategy="Starting workers...",
                    current_quarter=None,
                    estimated_remaining_seconds=None,
                )
            )

        # Run simulations in parallel
        logger.info(f"Submitting {len(tasks)} tasks to ProcessPoolExecutor...")
        executor = None
        cancelled = False
        try:
            executor = ProcessPoolExecutor(max_workers=self.num_workers)
            futures = {
                executor.submit(
                    _run_single_simulation,
                    task[0],  # strategy_dict
                    task[1],  # buy_quarter
                    task[2],  # sell_quarter
                    task[3],  # holding_period
                    preloader_data,
                ): task
                for task in tasks
            }
            logger.info(f"All {len(futures)} futures submitted, waiting for completion...")

            for future in as_completed(futures):
                # Check for cancellation
                if self.cancel_check and self.cancel_check():
                    logger.info("Grid search cancelled by user - stopping gracefully")
                    cancelled = True
                    break

                try:
                    result = future.result(timeout=60)  # 60s timeout per simulation
                    results.append(result)
                except Exception as e:
                    # Suppress errors from cancelled/terminated processes
                    error_str = str(e)
                    if "terminated" not in error_str.lower() and "cancel" not in error_str.lower():
                        logger.error(f"Simulation failed: {e}")
                    continue

                completed += 1

                # Report progress every 5 simulations (or every one for first 20)
                should_report = (completed <= 20) or (completed % 5 == 0)
                if self.progress_callback and should_report:
                    elapsed = time.time() - sim_start_time
                    rate = completed / elapsed if elapsed > 0 else 0
                    remaining = (total - completed) / rate if rate > 0 else None

                    self.progress_callback(
                        GridSearchProgress(
                            search_id=self.search_id,
                            status="running",
                            total_simulations=total,
                            completed=completed,
                            current_strategy=result.strategy_name if result else None,
                            current_quarter=result.buy_quarter if result else None,
                            estimated_remaining_seconds=int(remaining) if remaining else None,
                        )
                    )

        except Exception as e:
            logger.error(f"ProcessPoolExecutor failed: {e}")
            raise
        finally:
            # Always clean up the executor properly
            if executor is not None:
                logger.info("Shutting down ProcessPoolExecutor...")
                try:
                    # Cancel pending futures and shutdown
                    executor.shutdown(wait=False, cancel_futures=True)
                except Exception as e:
                    logger.warning(f"Error during executor shutdown: {e}")

        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()

        # Handle cancellation
        if cancelled:
            logger.info(f"Grid search cancelled after {completed} of {total} simulations")
            if self.progress_callback:
                self.progress_callback(
                    GridSearchProgress(
                        search_id=self.search_id,
                        status="cancelled",
                        total_simulations=total,
                        completed=completed,
                        current_strategy=None,
                        current_quarter=None,
                        estimated_remaining_seconds=0,
                    )
                )
            # Return partial results with aggregations
            results_with_stocks = [r for r in results if r.stock_count > 0]
            by_strategy, by_hold = _compute_aggregations(results)
            return GridSearchResult(
                id=self.search_id,
                started_at=start_time,
                completed_at=end_time,
                total_simulations=total,
                completed_simulations=completed,
                duration_seconds=round(duration, 2),
                results=results,
                best_by_alpha=sorted(results_with_stocks, key=lambda x: x.alpha, reverse=True)[:20],
                best_by_win_rate=sorted(results_with_stocks, key=lambda x: x.win_rate, reverse=True)[:20],
                by_strategy=by_strategy,
                by_holding_period=by_hold,
                request_config=request.model_dump(),
            )

        # Sort results
        results_with_stocks = [r for r in results if r.stock_count > 0]
        best_by_alpha = sorted(results_with_stocks, key=lambda x: x.alpha, reverse=True)[:20]
        best_by_win_rate = sorted(results_with_stocks, key=lambda x: x.win_rate, reverse=True)[:20]

        # Compute aggregations
        by_strategy, by_hold = _compute_aggregations(results)

        # Final progress update
        if self.progress_callback:
            self.progress_callback(
                GridSearchProgress(
                    search_id=self.search_id,
                    status="completed",
                    total_simulations=total,
                    completed=total,
                    current_strategy=None,
                    current_quarter=None,
                    estimated_remaining_seconds=0,
                )
            )

        logger.info(
            f"Grid search completed: {total} simulations in {duration:.2f}s "
            f"({total / duration:.1f} sim/s)"
        )

        return GridSearchResult(
            id=self.search_id,
            started_at=start_time,
            completed_at=end_time,
            total_simulations=total,
            completed_simulations=completed,
            duration_seconds=round(duration, 2),
            results=results,
            best_by_alpha=best_by_alpha,
            best_by_win_rate=best_by_win_rate,
            by_strategy=by_strategy,
            by_holding_period=by_hold,
            request_config=request.model_dump(),
        )

    def run_sync(
        self,
        request: GridSearchRequest,
    ) -> GridSearchResult:
        """
        Run simulations synchronously (single-threaded).

        Useful for debugging or when multiprocessing is not available.
        """
        start_time = datetime.now()

        strategies = list(
            StrategyBuilder.generate_strategies(request.base_strategy, request.dimensions)
        )

        preloader_data = {
            "analysis_data": self.preloader.analysis_data,
            "price_data": self.preloader.price_data,
            "spy_prices": self.preloader.spy_prices,
        }

        tasks = []
        for strategy in strategies:
            for buy_quarter in request.quarters:
                for hold in request.holding_periods:
                    sell_quarter = self.preloader.get_sell_quarter(buy_quarter, hold)
                    if sell_quarter:
                        tasks.append(
                            (
                                StrategyBuilder.strategy_to_dict(strategy),
                                buy_quarter,
                                sell_quarter,
                                hold,
                            )
                        )

        total = len(tasks)
        results = []

        for i, task in enumerate(tasks):
            result = _run_single_simulation(
                task[0], task[1], task[2], task[3], preloader_data
            )
            results.append(result)

            if self.progress_callback and (i + 1) % 10 == 0:
                self.progress_callback(
                    GridSearchProgress(
                        search_id=self.search_id,
                        status="running",
                        total_simulations=total,
                        completed=i + 1,
                        current_strategy=result.strategy_name,
                        current_quarter=result.buy_quarter,
                        estimated_remaining_seconds=None,
                    )
                )

        results_with_stocks = [r for r in results if r.stock_count > 0]
        best_by_alpha = sorted(results_with_stocks, key=lambda x: x.alpha, reverse=True)[:20]
        best_by_win_rate = sorted(results_with_stocks, key=lambda x: x.win_rate, reverse=True)[:20]

        # Compute aggregations
        by_strategy, by_hold = _compute_aggregations(results)

        end_time = datetime.now()

        return GridSearchResult(
            id=self.search_id,
            started_at=start_time,
            completed_at=end_time,
            total_simulations=total,
            completed_simulations=len(results),
            duration_seconds=round((end_time - start_time).total_seconds(), 2),
            results=results,
            best_by_alpha=best_by_alpha,
            best_by_win_rate=best_by_win_rate,
            by_strategy=by_strategy,
            by_holding_period=by_hold,
            request_config=request.model_dump(),
        )
