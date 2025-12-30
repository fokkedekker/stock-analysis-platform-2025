"""Factor Decay Analyzer - Rolling window stability analysis for factors."""

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy import stats


@dataclass
class RollingStats:
    """Statistics computed over rolling windows."""

    alphas: list[float]  # Alpha for each window
    ics: list[float]  # Information coefficient for each window
    pvalues: list[float]  # p-value for IC significance
    window_quarters: list[str]  # End quarter for each window
    sample_sizes: list[int]  # Sample size for each window


@dataclass
class DecayMetrics:
    """Factor stability metrics from rolling window analysis."""

    decay_score: float  # % of windows with positive alpha (0-1)
    ic_stability: float  # % of windows with significant IC (0-1)
    alpha_trend: float  # Direction of alpha over time (-1 to +1)
    n_windows: int  # Number of rolling windows analyzed
    recent_alpha: float | None  # Most recent window alpha
    mean_ic: float | None  # Average information coefficient

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "decay_score": self.decay_score,
            "ic_stability": self.ic_stability,
            "alpha_trend": self.alpha_trend,
            "n_windows": self.n_windows,
            "recent_alpha": self.recent_alpha,
            "mean_ic": self.mean_ic,
        }


class DecayAnalyzer:
    """
    Analyzes factor stability over rolling time windows.

    A factor with consistent positive alpha across many windows is more
    trustworthy than one with high alpha but erratic behavior.
    """

    def __init__(
        self,
        window_quarters: int = 20,  # 5 years default
        min_window_samples: int = 30,  # Minimum samples per window
        ic_significance_level: float = 0.05,  # p-value threshold for IC
    ):
        self.window_quarters = window_quarters
        self.min_window_samples = min_window_samples
        self.ic_significance_level = ic_significance_level

    def analyze_factor(
        self,
        observations: list[dict[str, Any]],
        factor_name: str,
        threshold: float | str | bool,
        operator: str,  # ">=", "<=", "==", "has", "not_has"
        holding_period: int,
    ) -> DecayMetrics | None:
        """
        Analyze factor stability for a specific threshold.

        Args:
            observations: List of observation dicts from dataset_builder
            factor_name: Name of the factor to analyze
            threshold: Threshold value to test
            operator: Comparison operator
            holding_period: Holding period to analyze (for filtering)

        Returns:
            DecayMetrics if enough data, None otherwise
        """
        # Filter to correct holding period
        filtered_obs = [
            o for o in observations if o.get("holding_period") == holding_period
        ]

        if len(filtered_obs) < self.window_quarters * 10:
            return None  # Not enough data

        # Get unique quarters sorted chronologically
        quarters = sorted(set(o.get("buy_quarter", "") for o in filtered_obs))

        if len(quarters) < self.window_quarters:
            return None  # Not enough quarters for rolling analysis

        # Compute rolling statistics
        rolling_stats = self._compute_rolling_stats(
            filtered_obs, quarters, factor_name, threshold, operator
        )

        if rolling_stats.n_windows < 3:
            return None  # Need at least 3 windows for trend

        # Compute decay metrics from rolling stats
        return self._compute_decay_metrics(rolling_stats)

    def _compute_rolling_stats(
        self,
        observations: list[dict[str, Any]],
        quarters: list[str],
        factor_name: str,
        threshold: float | str | bool,
        operator: str,
    ) -> RollingStats:
        """Compute statistics for each rolling window."""
        n_windows = len(quarters) - self.window_quarters + 1

        alphas = []
        ics = []
        pvalues = []
        window_quarters = []
        sample_sizes = []

        for i in range(n_windows):
            window_q = quarters[i : i + self.window_quarters]
            window_end = window_q[-1]

            # Filter observations to this window
            window_obs = [o for o in observations if o.get("buy_quarter") in window_q]

            if len(window_obs) < self.min_window_samples:
                continue

            # Get factor values and alphas
            factor_values = []
            obs_alphas = []

            for o in window_obs:
                fv = o.get(factor_name)
                alpha = o.get("alpha")

                if fv is not None and alpha is not None:
                    factor_values.append(fv)
                    obs_alphas.append(alpha)

            if len(factor_values) < self.min_window_samples:
                continue

            factor_values = np.array(factor_values)
            obs_alphas = np.array(obs_alphas)

            # Apply threshold filter
            passes = self._apply_threshold(factor_values, threshold, operator)

            if passes.sum() < self.min_window_samples:
                continue

            # Calculate alpha for stocks passing threshold
            passing_alphas = obs_alphas[passes]
            window_alpha = float(np.mean(passing_alphas))
            alphas.append(window_alpha)

            # Calculate Information Coefficient (rank correlation)
            try:
                ic, pvalue = stats.spearmanr(factor_values, obs_alphas)
                if np.isnan(ic):
                    ic, pvalue = 0.0, 1.0
            except Exception:
                ic, pvalue = 0.0, 1.0

            ics.append(float(ic))
            pvalues.append(float(pvalue))
            window_quarters.append(window_end)
            sample_sizes.append(int(passes.sum()))

        return RollingStats(
            alphas=alphas,
            ics=ics,
            pvalues=pvalues,
            window_quarters=window_quarters,
            sample_sizes=sample_sizes,
        )

    @property
    def n_windows(self) -> int:
        """Property to get number of windows from RollingStats."""
        return 0  # Will be set from RollingStats

    def _apply_threshold(
        self,
        values: np.ndarray,
        threshold: float | str | bool,
        operator: str,
    ) -> np.ndarray:
        """Apply threshold filter to values, returning boolean mask."""
        if operator == ">=":
            return values >= float(threshold)
        elif operator == "<=":
            return values <= float(threshold)
        elif operator == ">":
            return values > float(threshold)
        elif operator == "<":
            return values < float(threshold)
        elif operator == "==":
            if isinstance(threshold, bool):
                return values == threshold
            elif isinstance(threshold, str):
                return np.array([str(v) == threshold for v in values])
            else:
                return values == threshold
        elif operator == "has" or operator == "not_has":
            # Boolean: True if should have tag, False if should not
            target = operator == "has"
            return np.array([bool(v) == target for v in values])
        else:
            # Default: treat as >=
            return values >= float(threshold)

    def _compute_decay_metrics(self, rolling_stats: RollingStats) -> DecayMetrics:
        """Compute decay metrics from rolling statistics."""
        alphas = rolling_stats.alphas
        ics = rolling_stats.ics
        pvalues = rolling_stats.pvalues
        n_windows = len(alphas)

        if n_windows == 0:
            return DecayMetrics(
                decay_score=0.0,
                ic_stability=0.0,
                alpha_trend=0.0,
                n_windows=0,
                recent_alpha=None,
                mean_ic=None,
            )

        # Decay score: % of windows with positive alpha
        positive_windows = sum(1 for a in alphas if a > 0)
        decay_score = positive_windows / n_windows

        # IC stability: % of windows with significant IC
        significant_ic = sum(
            1 for p in pvalues if p < self.ic_significance_level
        )
        ic_stability = significant_ic / n_windows

        # Alpha trend: direction over time (-1 to +1)
        alpha_trend = self._compute_trend(alphas)

        # Recent alpha (last window)
        recent_alpha = alphas[-1] if alphas else None

        # Mean IC
        mean_ic = float(np.mean(ics)) if ics else None

        return DecayMetrics(
            decay_score=decay_score,
            ic_stability=ic_stability,
            alpha_trend=alpha_trend,
            n_windows=n_windows,
            recent_alpha=recent_alpha,
            mean_ic=mean_ic,
        )

    def _compute_trend(self, values: list[float]) -> float:
        """
        Compute trend direction of values over time.

        Returns:
            Value from -1 (declining) to +1 (increasing)
            0 = flat/no trend
        """
        if len(values) < 3:
            return 0.0

        # Simple approach: compare first half vs second half
        mid = len(values) // 2
        first_half = np.mean(values[:mid])
        second_half = np.mean(values[mid:])

        # Normalize by average magnitude
        avg_magnitude = (abs(first_half) + abs(second_half)) / 2
        if avg_magnitude < 0.01:  # Essentially zero
            return 0.0

        # Trend = (second - first) / avg_magnitude, clamped to [-1, 1]
        raw_trend = (second_half - first_half) / avg_magnitude
        return float(np.clip(raw_trend, -1.0, 1.0))


# Convenience function for use in factor_analyzer
def compute_factor_decay(
    observations: list[dict[str, Any]],
    factor_name: str,
    threshold: float | str | bool,
    operator: str,
    holding_period: int,
    window_quarters: int = 20,
) -> DecayMetrics | None:
    """
    Convenience function to compute decay metrics for a factor.

    Args:
        observations: List of observation dicts
        factor_name: Factor to analyze
        threshold: Threshold value
        operator: Comparison operator
        holding_period: Holding period in quarters
        window_quarters: Size of rolling window (default 20 = 5 years)

    Returns:
        DecayMetrics if enough data, None otherwise
    """
    analyzer = DecayAnalyzer(window_quarters=window_quarters)
    return analyzer.analyze_factor(
        observations=observations,
        factor_name=factor_name,
        threshold=threshold,
        operator=operator,
        holding_period=holding_period,
    )
