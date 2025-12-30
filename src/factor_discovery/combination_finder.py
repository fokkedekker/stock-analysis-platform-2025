"""Combination finder for factor discovery.

Finds optimal combinations of factors that predict alpha.
Uses parallel processing for combination testing.
"""

import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import combinations
from typing import Any

import numpy as np

from collections import defaultdict

from .factor_analyzer import bootstrap_ci
from .models import CombinedStrategyResult, FactorResult, FilterSpec, PortfolioStats

logger = logging.getLogger(__name__)


# =============================================================================
# Worker functions for parallel processing (must be at module level)
# =============================================================================


def _test_combination_worker(
    combo_filters: list[dict],
    data: list[dict],
    min_sample_size: int,
    overall_win_rate: float,
    portfolio_sizes: list[int],
    ranking_method: str,
) -> dict | None:
    """
    Worker function to test a single combination of filters.

    Must be a top-level function for multiprocessing to pickle it.

    Args:
        combo_filters: List of filter dicts (factor, operator, value)
        data: Full dataset of observations
        min_sample_size: Minimum sample size required
        overall_win_rate: Overall win rate for lift calculation
        portfolio_sizes: Portfolio sizes to simulate
        ranking_method: How to rank stocks

    Returns:
        Dict with result data, or None if combination invalid
    """
    # Apply filters
    filtered = data
    for f in combo_filters:
        filtered = [d for d in filtered if _passes_filter_worker(d, f)]

    if len(filtered) < min_sample_size:
        return None

    # Calculate stats
    alphas = [d["alpha"] for d in filtered]
    mean_alpha = float(np.mean(alphas))
    win_rate = sum(1 for a in alphas if a > 0) / len(alphas) * 100

    # Calculate lift
    lift = (win_rate / 100) / overall_win_rate if overall_win_rate > 0 else 1.0

    # Bootstrap CI
    ci_lower, ci_upper = bootstrap_ci(alphas)

    # Calculate portfolio stats
    portfolio_stats = _calculate_portfolio_stats_worker(
        filtered, portfolio_sizes, ranking_method
    )

    return {
        "filters": combo_filters,
        "mean_alpha": round(mean_alpha, 4),
        "sample_size": len(filtered),
        "lift": round(lift, 4),
        "win_rate": round(win_rate, 2),
        "ci_lower": round(ci_lower, 4),
        "ci_upper": round(ci_upper, 4),
        "portfolio_stats": portfolio_stats,
    }


def _passes_filter_worker(observation: dict, filter_spec: dict) -> bool:
    """Check if observation passes a filter (worker version)."""
    value = observation.get(filter_spec["factor"])

    if value is None:
        return False

    op = filter_spec["operator"]
    threshold = filter_spec["value"]

    if op == ">=":
        return value >= threshold
    elif op == "<=":
        return value <= threshold
    elif op == ">":
        return value > threshold
    elif op == "<":
        return value < threshold
    elif op == "==":
        return value == threshold
    elif op == "!=":
        return value != threshold
    elif op == "in":
        return value in threshold
    elif op == "not_in":
        return value not in threshold
    else:
        return True


def _calculate_portfolio_stats_worker(
    filtered_data: list[dict],
    portfolio_sizes: list[int],
    ranking_method: str,
) -> dict[int, dict]:
    """Calculate portfolio stats (worker version)."""
    if not filtered_data:
        return {}

    # Group by buy_quarter
    by_quarter: dict[str, list[dict]] = defaultdict(list)
    for obs in filtered_data:
        quarter = obs.get("buy_quarter")
        if quarter:
            by_quarter[quarter].append(obs)

    results: dict[int, dict] = {}

    for size in portfolio_sizes:
        all_alphas: list[float] = []

        for quarter, stocks in by_quarter.items():
            # Sort by ranking method
            sorted_stocks = _rank_stocks_worker(stocks, ranking_method)

            # Take top N
            top_n = sorted_stocks[:size]

            for s in top_n:
                alpha = s.get("alpha")
                if alpha is not None:
                    all_alphas.append(alpha)

        if not all_alphas:
            continue

        mean_alpha = float(np.mean(all_alphas))
        win_rate = (sum(1 for a in all_alphas if a > 0) / len(all_alphas)) * 100
        ci_lower, ci_upper = bootstrap_ci(all_alphas)

        results[size] = {
            "size": size,
            "mean_alpha": round(mean_alpha, 4),
            "sample_size": len(all_alphas),
            "win_rate": round(win_rate, 2),
            "ci_lower": round(ci_lower, 4),
            "ci_upper": round(ci_upper, 4),
        }

    return results


def _rank_stocks_worker(stocks: list[dict], ranking_method: str) -> list[dict]:
    """Rank stocks (worker version)."""
    if ranking_method == "magic-formula":
        return sorted(stocks, key=lambda x: x.get("magic_formula_rank") or 9999)
    elif ranking_method == "earnings-yield":
        return sorted(stocks, key=lambda x: -(x.get("earnings_yield") or -9999))
    elif ranking_method == "roic":
        return sorted(stocks, key=lambda x: -(x.get("roic") or -9999))
    elif ranking_method == "graham-score":
        return sorted(stocks, key=lambda x: -(x.get("graham_score") or -9999))
    else:
        return sorted(stocks, key=lambda x: x.get("magic_formula_rank") or 9999)


class CombinationFinder:
    """
    Find best combinations of factors.

    Takes top individual factors and tests all combinations
    of 1, 2, 3, 4 factors to find the best-performing strategies.
    """

    def __init__(
        self,
        factor_results: list[FactorResult],
        min_sample_size: int = 100,
        significance_level: float = 0.05,
        progress_callback: Any = None,
    ):
        """
        Initialize the combination finder.

        Args:
            factor_results: Results from individual factor analysis
            min_sample_size: Minimum sample size for valid combinations
            significance_level: P-value threshold for selecting top factors
            progress_callback: Optional callback for progress updates (0.0-1.0)
        """
        self.factor_results = factor_results
        self.min_sample_size = min_sample_size
        self.significance_level = significance_level
        self.progress_callback = progress_callback

    def find_best_combinations(
        self,
        data: list[dict],
        max_factors: int = 4,
        top_factors_count: int = 6,
        top_n_results: int = 20,
        portfolio_sizes: list[int] | None = None,
        ranking_method: str = "magic-formula",
        num_workers: int = 8,
    ) -> list[CombinedStrategyResult]:
        """
        Test combinations of top factors using parallel processing.

        1. Select top N factors by lift (that are significant)
        2. Test all 1, 2, 3, ... max_factors combinations in parallel
        3. Return top 20 by alpha

        Args:
            data: Dataset of observations with all metrics
            max_factors: Maximum number of factors to combine
            top_factors_count: Number of top factors to consider
            top_n_results: Number of top combinations to return
            portfolio_sizes: List of portfolio sizes to simulate (e.g., [10, 20, 50])
            ranking_method: How to rank stocks (magic-formula, earnings-yield, roic, graham-score)
            num_workers: Number of parallel workers

        Returns:
            List of CombinedStrategyResult sorted by mean alpha
        """
        if portfolio_sizes is None:
            portfolio_sizes = [20]
        if not data:
            logger.warning("No data provided for combination analysis")
            return []

        # Get top factors that are significant
        top_factors = self._select_top_factors(top_factors_count)

        if not top_factors:
            logger.warning("No significant factors found for combination analysis")
            return []

        logger.info(
            f"Testing combinations of {len(top_factors)} factors: "
            f"{[f.factor_name for f in top_factors]}"
        )

        # Calculate overall win rate for lift calculation
        all_alphas = [d["alpha"] for d in data]
        overall_win_rate = sum(1 for a in all_alphas if a > 0) / len(all_alphas)

        # Build all combinations to test
        all_combos = []
        for n in range(1, min(max_factors + 1, len(top_factors) + 1)):
            for combo in combinations(top_factors, n):
                # Build filter specs as dicts (picklable)
                filters = []
                for factor in combo:
                    if factor.best_threshold is None:
                        continue
                    operator, value = self._parse_threshold(
                        factor.best_threshold, factor.factor_type
                    )
                    filters.append({
                        "factor": factor.factor_name,
                        "operator": operator,
                        "value": value,
                    })
                if filters:
                    all_combos.append(filters)

        total_combos = len(all_combos)
        logger.info(f"Testing {total_combos} combinations with {num_workers} workers")

        results = []

        # Use parallel processing for combination testing
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = {
                executor.submit(
                    _test_combination_worker,
                    combo_filters,
                    data,
                    self.min_sample_size,
                    overall_win_rate,
                    portfolio_sizes,
                    ranking_method,
                ): combo_filters
                for combo_filters in all_combos
            }

            completed = 0
            for future in as_completed(futures):
                completed += 1
                if completed % 100 == 0:
                    logger.info(f"Tested {completed}/{total_combos} combinations")

                # Report progress
                if self.progress_callback and total_combos > 0:
                    try:
                        self.progress_callback(completed / total_combos)
                    except Exception:
                        pass

                try:
                    result_dict = future.result()
                    if result_dict is not None:
                        # Convert dict back to FilterSpec and PortfolioStats objects
                        filters = [
                            FilterSpec(
                                factor=f["factor"],
                                operator=f["operator"],
                                value=f["value"],
                            )
                            for f in result_dict["filters"]
                        ]
                        portfolio_stats = {
                            size: PortfolioStats(**stats)
                            for size, stats in result_dict["portfolio_stats"].items()
                        }

                        results.append(
                            CombinedStrategyResult(
                                filters=filters,
                                mean_alpha=result_dict["mean_alpha"],
                                sample_size=result_dict["sample_size"],
                                lift=result_dict["lift"],
                                win_rate=result_dict["win_rate"],
                                ci_lower=result_dict["ci_lower"],
                                ci_upper=result_dict["ci_upper"],
                                portfolio_stats=portfolio_stats,
                            )
                        )
                except Exception as e:
                    logger.error(f"Combination test failed: {e}")

        # Sort by mean alpha (descending) and return top N
        results = sorted(results, key=lambda r: r.mean_alpha, reverse=True)

        logger.info(
            f"Found {len(results)} valid combinations, returning top {top_n_results}"
        )

        return results[:top_n_results]

    def _select_top_factors(self, top_n: int) -> list[FactorResult]:
        """
        Select top N factors for combination testing.

        Criteria:
        - Must have a best threshold defined
        - Must be FDR-significant (Benjamini-Hochberg corrected)
        - Sorted by lift (descending)

        Args:
            top_n: Number of top factors to select

        Returns:
            List of top FactorResult objects
        """
        # Filter for FDR-significant factors with valid thresholds
        # Prefer FDR-significant, fall back to raw p-value if FDR not available
        significant_factors = [
            f
            for f in self.factor_results
            if f.best_threshold is not None
            and f.best_threshold_lift is not None
            and (
                # Check FDR significance if available
                f.best_threshold_fdr_significant is True
                or (
                    # Fall back to raw p-value if FDR not computed
                    f.best_threshold_fdr_significant is None
                    and f.best_threshold_pvalue is not None
                    and f.best_threshold_pvalue < self.significance_level
                )
            )
        ]

        # Sort by lift (descending)
        significant_factors = sorted(
            significant_factors,
            key=lambda f: f.best_threshold_lift or 0,
            reverse=True,
        )

        return significant_factors[:top_n]

    def _calculate_portfolio_stats(
        self,
        filtered_data: list[dict],
        portfolio_sizes: list[int],
        ranking_method: str,
    ) -> dict[int, PortfolioStats]:
        """
        Calculate stats for top-N portfolios per quarter.

        Groups stocks by quarter, ranks them, takes top-N, and calculates stats.

        Args:
            filtered_data: Observations that passed all filters
            portfolio_sizes: List of portfolio sizes (e.g., [10, 20, 50])
            ranking_method: How to rank stocks

        Returns:
            Dict mapping portfolio size to PortfolioStats
        """
        if not filtered_data:
            return {}

        # Group by buy_quarter
        by_quarter: dict[str, list[dict]] = defaultdict(list)
        for obs in filtered_data:
            quarter = obs.get("buy_quarter")
            if quarter:
                by_quarter[quarter].append(obs)

        results: dict[int, PortfolioStats] = {}

        for size in portfolio_sizes:
            all_alphas: list[float] = []

            for quarter, stocks in by_quarter.items():
                # Sort by ranking method
                sorted_stocks = self._rank_stocks(stocks, ranking_method)

                # Take top N (or all if fewer available)
                top_n = sorted_stocks[:size]

                # Collect alphas
                for s in top_n:
                    alpha = s.get("alpha")
                    if alpha is not None:
                        all_alphas.append(alpha)

            # Skip if no data
            if not all_alphas:
                continue

            # Calculate stats
            mean_alpha = float(np.mean(all_alphas))
            win_rate = (sum(1 for a in all_alphas if a > 0) / len(all_alphas)) * 100
            ci_lower, ci_upper = bootstrap_ci(all_alphas)

            results[size] = PortfolioStats(
                size=size,
                mean_alpha=round(mean_alpha, 4),
                sample_size=len(all_alphas),
                win_rate=round(win_rate, 2),
                ci_lower=round(ci_lower, 4),
                ci_upper=round(ci_upper, 4),
            )

        return results

    def _rank_stocks(
        self, stocks: list[dict], ranking_method: str
    ) -> list[dict]:
        """
        Rank stocks by the specified method.

        Args:
            stocks: List of stock observations
            ranking_method: Ranking method (magic-formula, earnings-yield, roic, graham-score)

        Returns:
            Sorted list (best stocks first)
        """
        if ranking_method == "magic-formula":
            # Lower rank is better
            return sorted(
                stocks,
                key=lambda x: x.get("magic_formula_rank") or 9999
            )
        elif ranking_method == "earnings-yield":
            # Higher is better
            return sorted(
                stocks,
                key=lambda x: -(x.get("earnings_yield") or -9999)
            )
        elif ranking_method == "roic":
            # Higher is better
            return sorted(
                stocks,
                key=lambda x: -(x.get("roic") or -9999)
            )
        elif ranking_method == "graham-score":
            # Higher is better
            return sorted(
                stocks,
                key=lambda x: -(x.get("graham_score") or -9999)
            )
        else:
            # Default: magic formula
            return sorted(
                stocks,
                key=lambda x: x.get("magic_formula_rank") or 9999
            )

    def _build_filters(self, factors: tuple[FactorResult, ...]) -> list[FilterSpec]:
        """
        Build filter specifications from factor results.

        Args:
            factors: Tuple of FactorResult objects

        Returns:
            List of FilterSpec objects
        """
        filters = []

        for factor in factors:
            if factor.best_threshold is None:
                continue

            operator, value = self._parse_threshold(
                factor.best_threshold, factor.factor_type
            )

            filters.append(
                FilterSpec(
                    factor=factor.factor_name,
                    operator=operator,
                    value=value,
                )
            )

        return filters

    def _parse_threshold(
        self, threshold: str, factor_type: str
    ) -> tuple[str, Any]:
        """
        Parse a threshold string into operator and value.

        Examples:
            ">= 6" -> (">=", 6)
            "<= 1.5" -> ("<=", 1.5)
            "safe" -> ("==", "safe")
            "True" -> ("==", True)

        Args:
            threshold: Threshold string
            factor_type: Type of factor (numerical, categorical, boolean)

        Returns:
            Tuple of (operator, value)
        """
        threshold = threshold.strip()

        # Handle boolean thresholds
        if factor_type == "boolean":
            if threshold.lower() in ("true", "yes", "1"):
                return ("==", True)
            else:
                return ("==", False)

        # Handle categorical thresholds
        if factor_type == "categorical":
            return ("==", threshold)

        # Handle numerical thresholds with operators
        if threshold.startswith(">="):
            value_str = threshold[2:].strip()
            return (">=", self._parse_numeric(value_str))
        elif threshold.startswith("<="):
            value_str = threshold[2:].strip()
            return ("<=", self._parse_numeric(value_str))
        elif threshold.startswith(">"):
            value_str = threshold[1:].strip()
            return (">", self._parse_numeric(value_str))
        elif threshold.startswith("<"):
            value_str = threshold[1:].strip()
            return ("<", self._parse_numeric(value_str))
        elif threshold.startswith("=="):
            value_str = threshold[2:].strip()
            return ("==", self._parse_numeric(value_str))
        else:
            # Assume it's just a value for equality
            return ("==", self._parse_numeric(threshold))

    def _parse_numeric(self, value_str: str) -> float | int:
        """Parse a numeric string to int or float."""
        try:
            # Try int first
            if "." not in value_str:
                return int(value_str)
            return float(value_str)
        except ValueError:
            return float(value_str)

    def _apply_filters(
        self, data: list[dict], filters: list[FilterSpec]
    ) -> list[dict]:
        """
        Apply filters to the dataset.

        Args:
            data: List of observation dicts
            filters: List of filter specifications

        Returns:
            Filtered list of observations
        """
        filtered = data

        for f in filters:
            filtered = [d for d in filtered if self._passes_filter(d, f)]

        return filtered

    def _passes_filter(self, observation: dict, filter_spec: FilterSpec) -> bool:
        """
        Check if an observation passes a filter.

        Args:
            observation: Single observation dict
            filter_spec: Filter to apply

        Returns:
            True if observation passes the filter
        """
        value = observation.get(filter_spec.factor)

        # Skip if value is None
        if value is None:
            return False

        op = filter_spec.operator
        threshold = filter_spec.value

        if op == ">=":
            return value >= threshold
        elif op == "<=":
            return value <= threshold
        elif op == ">":
            return value > threshold
        elif op == "<":
            return value < threshold
        elif op == "==":
            return value == threshold
        elif op == "!=":
            return value != threshold
        elif op == "in":
            return value in threshold
        elif op == "not_in":
            return value not in threshold
        elif op == "has":
            # For checking if a value exists in a collection
            if isinstance(value, (list, set, tuple)):
                return threshold in value
            return False
        elif op == "not_has":
            if isinstance(value, (list, set, tuple)):
                return threshold not in value
            return True
        else:
            logger.warning(f"Unknown operator: {op}")
            return True

    def get_strategy_description(self, strategy: CombinedStrategyResult) -> str:
        """
        Generate a human-readable description of a strategy.

        Args:
            strategy: CombinedStrategyResult to describe

        Returns:
            String description
        """
        parts = []

        for f in strategy.filters:
            if f.operator == "==":
                if isinstance(f.value, bool):
                    if f.value:
                        parts.append(f"{f.factor}")
                    else:
                        parts.append(f"NOT {f.factor}")
                else:
                    parts.append(f"{f.factor} = {f.value}")
            elif f.operator in (">=", "<=", ">", "<"):
                parts.append(f"{f.factor} {f.operator} {f.value}")
            else:
                parts.append(f"{f.factor} {f.operator} {f.value}")

        return " AND ".join(parts)
