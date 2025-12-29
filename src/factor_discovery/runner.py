"""Factor Discovery Runner - Orchestrates the full analysis.

Coordinates dataset building, factor analysis, combination finding,
and recommendation generation with parallel execution.
"""

import logging
import time
import uuid
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from typing import Any, Callable, Optional

from .combination_finder import CombinationFinder
from .dataset_builder import DatasetBuilder
from .factor_analyzer import FactorAnalyzer
from .models import (
    CombinedStrategyResult,
    FactorDiscoveryProgress,
    FactorDiscoveryRequest,
    FactorDiscoveryResult,
    FactorResult,
    RecommendedStrategy,
)
from .recommendation_generator import generate_all_recommendations

logger = logging.getLogger(__name__)


def _analyze_factor_worker(
    factor_type: str,
    factor_name: str,
    holding_period: int,
    data: list[dict],
    config: dict,
    min_sample_size: int,
) -> Optional[FactorResult]:
    """
    Worker function for parallel factor analysis.

    This function runs in a separate process, so it must be
    a top-level function (not a method) to be picklable.

    Args:
        factor_type: 'numerical', 'categorical', or 'boolean'
        factor_name: Name of the factor
        holding_period: Holding period in quarters
        data: Dataset slice for this holding period
        config: Factor configuration
        min_sample_size: Minimum sample size

    Returns:
        FactorResult or None if analysis failed
    """
    try:
        if factor_type == "numerical":
            return FactorAnalyzer.analyze_numerical(
                data=data,
                factor_name=factor_name,
                thresholds=config["thresholds"],
                direction=config["direction"],
                holding_period=holding_period,
                min_sample_size=min_sample_size,
            )
        elif factor_type == "categorical":
            return FactorAnalyzer.analyze_categorical(
                data=data,
                factor_name=factor_name,
                categories=config["categories"],
                holding_period=holding_period,
                min_sample_size=min_sample_size,
            )
        elif factor_type == "boolean":
            return FactorAnalyzer.analyze_boolean(
                data=data,
                factor_name=factor_name,
                holding_period=holding_period,
                min_sample_size=min_sample_size,
            )
        else:
            logger.warning(f"Unknown factor type: {factor_type}")
            return None
    except Exception as e:
        logger.error(f"Error analyzing {factor_name} for {holding_period}Q: {e}")
        return None


class FactorDiscoveryRunner:
    """
    Orchestrates the full factor discovery analysis.

    Flow:
    1. Build dataset from database
    2. Analyze individual factors (parallel)
    3. Find best combinations (per holding period)
    4. Generate recommendations
    5. Store results
    """

    def __init__(
        self,
        num_workers: int = 8,
        progress_callback: Optional[Callable[[FactorDiscoveryProgress], None]] = None,
        cancel_check: Optional[Callable[[], bool]] = None,
    ):
        """
        Initialize the runner.

        Args:
            num_workers: Number of parallel workers for factor analysis
            progress_callback: Optional callback for progress updates
            cancel_check: Optional function to check if run should be cancelled
        """
        self.num_workers = num_workers
        self.progress_callback = progress_callback
        self.cancel_check = cancel_check

    def run(self, request: FactorDiscoveryRequest, run_id: str | None = None) -> FactorDiscoveryResult:
        """
        Run the full factor discovery analysis.

        Args:
            request: Configuration for the analysis
            run_id: Optional run ID (generated if not provided)

        Returns:
            FactorDiscoveryResult with all findings
        """
        if run_id is None:
            run_id = str(uuid.uuid4())[:12]
        start_time = time.time()
        created_at = datetime.now()

        logger.info(f"Starting factor discovery run {run_id}")
        logger.info(f"Quarters: {request.quarters}")
        logger.info(f"Holding periods: {request.holding_periods}")

        try:
            # Phase 1: Build dataset
            self._report_progress(run_id, "building_dataset", 0.0)
            dataset = self._build_dataset(request)
            self._report_progress(run_id, "building_dataset", 1.0)

            if self._is_cancelled():
                return self._cancelled_result(run_id, created_at, request)

            # Phase 2: Analyze individual factors (parallel)
            self._report_progress(run_id, "analyzing_factors", 0.0)
            factor_results = self._analyze_factors_parallel(
                dataset=dataset,
                holding_periods=request.holding_periods,
                min_sample_size=request.min_sample_size,
                factor_categories=request.factor_categories,
            )
            self._report_progress(run_id, "analyzing_factors", 1.0)

            if self._is_cancelled():
                return self._cancelled_result(run_id, created_at, request)

            # Phase 3: Find combinations (per holding period)
            self._report_progress(run_id, "finding_combinations", 0.0)
            combined_results = self._find_combinations(
                dataset=dataset,
                factor_results=factor_results,
                holding_periods=request.holding_periods,
                min_sample_size=request.min_sample_size,
                significance_level=request.significance_level,
                portfolio_sizes=request.portfolio_sizes,
                ranking_method=request.ranking_method,
                max_factors=request.max_factors,
            )
            self._report_progress(run_id, "finding_combinations", 1.0)

            if self._is_cancelled():
                return self._cancelled_result(run_id, created_at, request)

            # Phase 4: Generate recommendations
            self._report_progress(run_id, "generating_recommendations", 0.0)
            recommended_strategies = generate_all_recommendations(
                factor_results_by_hp=factor_results,
                combined_results_by_hp=combined_results,
                cost_haircut=request.cost_haircut,
            )
            self._report_progress(run_id, "generating_recommendations", 1.0)

            # Find best overall
            best_hp, best_alpha = self._find_best_overall(recommended_strategies)

            duration = time.time() - start_time

            result = FactorDiscoveryResult(
                run_id=run_id,
                status="completed",
                created_at=created_at,
                completed_at=datetime.now(),
                duration_seconds=round(duration, 2),
                config=request,
                total_observations=dataset["metadata"]["total_rows"],
                factor_results=factor_results,
                combined_results=combined_results,
                recommended_strategies=recommended_strategies,
                best_holding_period=best_hp,
                best_alpha=best_alpha,
            )

            logger.info(
                f"Factor discovery completed in {duration:.1f}s. "
                f"Best alpha: {best_alpha}% at {best_hp}Q hold"
            )

            return result

        except Exception as e:
            logger.exception(f"Factor discovery failed: {e}")
            return FactorDiscoveryResult(
                run_id=run_id,
                status="failed",
                created_at=created_at,
                completed_at=datetime.now(),
                duration_seconds=time.time() - start_time,
                config=request,
                total_observations=0,
                factor_results={},
                combined_results={},
                recommended_strategies={},
            )

    def _build_dataset(self, request: FactorDiscoveryRequest) -> dict:
        """Build the analysis dataset."""
        # Convert exclusions to dict for builder
        exclusions = request.exclusions.model_dump() if request.exclusions else {}

        builder = DatasetBuilder(
            quarters=request.quarters,
            holding_periods=request.holding_periods,
            exclusions=exclusions,
        )
        return builder.build()

    def _analyze_factors_parallel(
        self,
        dataset: dict,
        holding_periods: list[int],
        min_sample_size: int,
        factor_categories: list[str] | None = None,
    ) -> dict[int, list[FactorResult]]:
        """
        Analyze all factors across all holding periods in parallel.

        Uses ProcessPoolExecutor to distribute work across CPU cores.

        Args:
            dataset: The built dataset
            holding_periods: List of holding periods
            min_sample_size: Minimum sample size for valid results
            factor_categories: Optional list of factor categories to include

        Returns:
            Dict of holding_period -> list of FactorResult
        """
        # Get factors filtered by category (or all if not specified)
        if factor_categories:
            factors = FactorAnalyzer.get_factors_by_category(factor_categories)
            numerical_factors = factors["numerical"]
            categorical_factors = factors["categorical"]
            boolean_factors = factors["boolean"]
        else:
            numerical_factors = FactorAnalyzer.NUMERICAL_FACTORS
            categorical_factors = FactorAnalyzer.CATEGORICAL_FACTORS
            boolean_factors = FactorAnalyzer.BOOLEAN_FACTORS

        logger.info(
            f"Analyzing {len(numerical_factors)} numerical, "
            f"{len(categorical_factors)} categorical, "
            f"{len(boolean_factors)} boolean factors"
        )

        # Prepare data slices for each holding period
        data_by_hp = {}
        for hp in holding_periods:
            data_by_hp[hp] = [
                d for d in dataset["data"] if d["holding_period"] == hp
            ]
            logger.info(f"Holding period {hp}Q: {len(data_by_hp[hp])} observations")

        # Build task list
        tasks = []

        for hp in holding_periods:
            hp_data = data_by_hp[hp]

            if not hp_data:
                continue

            # Numerical factors
            for factor_name, config in numerical_factors.items():
                tasks.append(
                    ("numerical", factor_name, hp, hp_data, config, min_sample_size)
                )

            # Categorical factors
            for factor_name, config in categorical_factors.items():
                tasks.append(
                    ("categorical", factor_name, hp, hp_data, config, min_sample_size)
                )

            # Boolean factors
            for factor_name, config in boolean_factors.items():
                tasks.append(
                    ("boolean", factor_name, hp, hp_data, config, min_sample_size)
                )

        logger.info(f"Running {len(tasks)} factor analyses with {self.num_workers} workers")

        # Initialize results
        results: dict[int, list[FactorResult]] = {hp: [] for hp in holding_periods}

        # Run in parallel
        with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
            futures = {
                executor.submit(_analyze_factor_worker, *task): task
                for task in tasks
            }

            completed = 0
            total = len(futures)

            for future in as_completed(futures):
                task = futures[future]
                factor_type, factor_name, hp = task[0], task[1], task[2]

                try:
                    result = future.result()
                    if result:
                        results[hp].append(result)
                except Exception as e:
                    logger.error(f"Task failed for {factor_name} {hp}Q: {e}")

                completed += 1
                progress = 0.0 + (completed / total) * 1.0
                self._report_progress(
                    "running",
                    "analyzing_factors",
                    progress,
                    current_factor=factor_name,
                    current_holding_period=hp,
                )

                if self._is_cancelled():
                    executor.shutdown(wait=False, cancel_futures=True)
                    break

        # Log summary
        for hp in holding_periods:
            logger.info(f"Holding period {hp}Q: {len(results[hp])} factors analyzed")

        return results

    def _find_combinations(
        self,
        dataset: dict,
        factor_results: dict[int, list[FactorResult]],
        holding_periods: list[int],
        min_sample_size: int,
        significance_level: float,
        portfolio_sizes: list[int] | None = None,
        ranking_method: str = "magic-formula",
        max_factors: int = 4,
    ) -> dict[int, list[CombinedStrategyResult]]:
        """
        Find best factor combinations for each holding period.

        Args:
            dataset: The full dataset
            factor_results: Individual factor results by holding period
            holding_periods: List of holding periods
            min_sample_size: Minimum sample size
            significance_level: P-value threshold
            portfolio_sizes: Portfolio sizes to simulate
            ranking_method: How to rank stocks for top-N selection

        Returns:
            Dict of holding_period -> list of CombinedStrategyResult
        """
        if portfolio_sizes is None:
            portfolio_sizes = [20]

        combined_results = {}
        total_hps = len(holding_periods)

        for hp_idx, hp in enumerate(holding_periods):
            hp_data = [d for d in dataset["data"] if d["holding_period"] == hp]
            hp_factors = factor_results.get(hp, [])

            if not hp_data or not hp_factors:
                combined_results[hp] = []
                continue

            # Create progress callback for this holding period
            def make_progress_cb(hp_index: int, total: int):
                def cb(combo_progress: float):
                    # Each HP contributes 1/total of the overall progress
                    overall = (hp_index + combo_progress) / total
                    self._report_progress(
                        "running",
                        "finding_combinations",
                        overall,
                    )
                return cb

            finder = CombinationFinder(
                factor_results=hp_factors,
                min_sample_size=min_sample_size,
                significance_level=significance_level,
                progress_callback=make_progress_cb(hp_idx, total_hps),
            )

            combinations = finder.find_best_combinations(
                data=hp_data,
                max_factors=max_factors,
                top_factors_count=max(6, max_factors + 2),  # Always have at least 2 more factors than max
                top_n_results=20,
                portfolio_sizes=portfolio_sizes,
                ranking_method=ranking_method,
                num_workers=self.num_workers,  # Use runner's worker count
            )

            combined_results[hp] = combinations
            logger.info(f"Holding period {hp}Q: Found {len(combinations)} valid combinations")

        return combined_results

    def _find_best_overall(
        self, strategies: dict[int, RecommendedStrategy]
    ) -> tuple[Optional[int], Optional[float]]:
        """Find the best holding period and alpha across all strategies."""
        if not strategies:
            return None, None

        best_hp = None
        best_alpha = None

        for hp, strategy in strategies.items():
            if best_alpha is None or strategy.expected_alpha > best_alpha:
                best_hp = hp
                best_alpha = strategy.expected_alpha

        return best_hp, best_alpha

    def _report_progress(
        self,
        run_id: str,
        phase: str,
        progress: float,
        current_factor: Optional[str] = None,
        current_holding_period: Optional[int] = None,
    ) -> None:
        """Report progress to callback if set."""
        if self.progress_callback:
            update = FactorDiscoveryProgress(
                run_id=run_id,
                status="running",
                phase=phase,
                progress=progress,
                current_factor=current_factor,
                current_holding_period=current_holding_period,
            )
            try:
                self.progress_callback(update)
            except Exception as e:
                logger.warning(f"Progress callback failed: {e}")

    def _is_cancelled(self) -> bool:
        """Check if the run should be cancelled."""
        if self.cancel_check:
            try:
                return self.cancel_check()
            except Exception:
                return False
        return False

    def _cancelled_result(
        self,
        run_id: str,
        created_at: datetime,
        request: FactorDiscoveryRequest,
    ) -> FactorDiscoveryResult:
        """Return a cancelled result."""
        return FactorDiscoveryResult(
            run_id=run_id,
            status="cancelled",
            created_at=created_at,
            completed_at=datetime.now(),
            config=request,
            total_observations=0,
            factor_results={},
            combined_results={},
            recommended_strategies={},
        )


def run_factor_discovery(
    quarters: list[str],
    holding_periods: list[int] = [1, 2, 3, 4],
    min_sample_size: int = 100,
    significance_level: float = 0.01,
    cost_haircut: float = 3.0,
    num_workers: int = 8,
) -> FactorDiscoveryResult:
    """
    Convenience function to run factor discovery.

    Args:
        quarters: List of quarters to analyze
        holding_periods: Holding periods in quarters
        min_sample_size: Minimum sample size for thresholds
        significance_level: P-value threshold
        cost_haircut: Minimum alpha to trust
        num_workers: Number of parallel workers

    Returns:
        FactorDiscoveryResult
    """
    request = FactorDiscoveryRequest(
        quarters=quarters,
        holding_periods=holding_periods,
        min_sample_size=min_sample_size,
        significance_level=significance_level,
        cost_haircut=cost_haircut,
    )

    runner = FactorDiscoveryRunner(num_workers=num_workers)
    return runner.run(request)
