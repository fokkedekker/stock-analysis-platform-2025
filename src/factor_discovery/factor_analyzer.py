"""Factor analyzer for statistical analysis of individual factors.

Implements statistical methods:
- Spearman correlation
- Mean alpha comparison
- Lift ratio
- Welch's t-test
- Bootstrap confidence intervals
- Chi-squared test (for categorical)
"""

import logging
from typing import Any

import numpy as np
from scipy import stats

from .models import FactorResult, ThresholdResult

logger = logging.getLogger(__name__)


def bootstrap_ci(
    values: list[float],
    n_bootstrap: int = 1000,
    ci: float = 0.95,
) -> tuple[float, float]:
    """
    Calculate bootstrap confidence interval for mean.

    Args:
        values: List of values to bootstrap
        n_bootstrap: Number of bootstrap samples
        ci: Confidence interval level (0.95 = 95%)

    Returns:
        Tuple of (lower_bound, upper_bound)
    """
    if len(values) < 2:
        mean = np.mean(values) if values else 0.0
        return mean, mean

    np.random.seed(42)  # For reproducibility
    means = []
    arr = np.array(values)

    for _ in range(n_bootstrap):
        sample = np.random.choice(arr, size=len(arr), replace=True)
        means.append(np.mean(sample))

    lower = np.percentile(means, (1 - ci) / 2 * 100)
    upper = np.percentile(means, (1 + ci) / 2 * 100)

    return float(lower), float(upper)


class FactorAnalyzer:
    """Analyzes individual factors for predictive power."""

    # Factor configurations: name -> {thresholds, direction}
    NUMERICAL_FACTORS = {
        "piotroski_score": {
            "thresholds": [3, 4, 5, 6, 7, 8],
            "direction": ">=",
            "label": "Piotroski F-Score",
        },
        "graham_score": {
            "thresholds": [3, 4, 5, 6, 7],
            "direction": ">=",
            "label": "Graham Score",
        },
        "altman_z_score": {
            "thresholds": [1.8, 2.0, 2.5, 3.0],
            "direction": ">=",
            "label": "Altman Z-Score",
        },
        "roic": {
            "thresholds": [0.08, 0.10, 0.12, 0.15, 0.20],
            "direction": ">=",
            "label": "ROIC",
        },
        "peg_ratio": {
            "thresholds": [0.5, 1.0, 1.5, 2.0, 2.5, 3.0],
            "direction": "<=",
            "label": "PEG Ratio",
        },
        "magic_formula_rank": {
            "thresholds": [100, 200, 500, 1000],
            "direction": "<=",
            "label": "Magic Formula Rank",
        },
        "book_to_market_percentile": {
            "thresholds": [0.50, 0.60, 0.70, 0.80],
            "direction": ">=",
            "label": "Book/Market Percentile",
        },
    }

    CATEGORICAL_FACTORS = {
        "altman_zone": {
            "categories": ["safe", "grey", "distress"],
            "label": "Altman Zone",
        },
    }

    # Boolean factors: has_X means "stock has this tag"
    BOOLEAN_FACTORS = {
        # Positive tags (require = good)
        "has_durable_compounder": {"label": "Durable Compounder", "positive": True},
        "has_cash_machine": {"label": "Cash Machine", "positive": True},
        "has_deep_value": {"label": "Deep Value", "positive": True},
        "has_heavy_reinvestor": {"label": "Heavy Reinvestor", "positive": True},
        # Negative tags (exclude = good)
        "has_premium_priced": {"label": "Premium Priced", "positive": False},
        "has_volatile_returns": {"label": "Volatile Returns", "positive": False},
        "has_weak_moat_signal": {"label": "Weak Moat Signal", "positive": False},
        "has_earnings_quality_concern": {"label": "Earnings Quality Concern", "positive": False},
        # Other booleans
        "trading_below_ncav": {"label": "Trading Below NCAV", "positive": True},
        "fcf_positive_5yr": {"label": "FCF Positive 5yr", "positive": True},
    }

    @staticmethod
    def analyze_numerical(
        data: list[dict],
        factor_name: str,
        thresholds: list,
        direction: str,
        min_sample_size: int = 100,
        holding_period: int = 1,
    ) -> FactorResult:
        """
        Analyze a numerical factor.

        Args:
            data: List of observation dicts with 'alpha' and factor_name keys
            factor_name: Name of the factor to analyze
            thresholds: List of threshold values to test
            direction: '>=' or '<='
            min_sample_size: Minimum sample size for threshold to be valid
            holding_period: Holding period for this analysis

        Returns:
            FactorResult with all statistics
        """
        # Filter to rows with valid factor values
        valid_data = [d for d in data if d.get(factor_name) is not None]

        if len(valid_data) < min_sample_size:
            logger.warning(
                f"Factor {factor_name}: only {len(valid_data)} valid rows, "
                f"need {min_sample_size}"
            )
            return FactorResult(
                factor_name=factor_name,
                factor_type="numerical",
                holding_period=holding_period,
                threshold_results=[],
            )

        values = [d[factor_name] for d in valid_data]
        alphas = [d["alpha"] for d in valid_data]

        # Spearman correlation
        try:
            corr, corr_pvalue = stats.spearmanr(values, alphas)
            corr = float(corr) if not np.isnan(corr) else None
            corr_pvalue = float(corr_pvalue) if not np.isnan(corr_pvalue) else None
        except Exception:
            corr, corr_pvalue = None, None

        # Overall stats for lift calculation
        overall_win_rate = sum(1 for a in alphas if a > 0) / len(alphas)

        # Threshold analysis
        threshold_results = []
        for thresh in thresholds:
            # Create mask based on direction
            if direction == ">=":
                mask = [v >= thresh for v in values]
                thresh_str = f">= {thresh}"
            else:  # <=
                mask = [v <= thresh for v in values]
                thresh_str = f"<= {thresh}"

            above_alphas = [a for a, m in zip(alphas, mask) if m]
            below_alphas = [a for a, m in zip(alphas, mask) if not m]

            if len(above_alphas) < min_sample_size:
                continue

            mean_alpha = float(np.mean(above_alphas))
            sample_size = len(above_alphas)

            # Win rate and lift
            threshold_win_rate = sum(1 for a in above_alphas if a > 0) / len(above_alphas)
            lift = threshold_win_rate / overall_win_rate if overall_win_rate > 0 else 1.0

            # Welch's t-test
            if len(below_alphas) >= 10:
                try:
                    _, pvalue = stats.ttest_ind(
                        above_alphas, below_alphas, equal_var=False
                    )
                    pvalue = float(pvalue) if not np.isnan(pvalue) else 1.0
                except Exception:
                    pvalue = 1.0
            else:
                pvalue = 1.0

            # Bootstrap CI
            ci_lower, ci_upper = bootstrap_ci(above_alphas)

            threshold_results.append(
                ThresholdResult(
                    threshold=thresh_str,
                    mean_alpha=round(mean_alpha, 4),
                    sample_size=sample_size,
                    lift=round(lift, 4),
                    pvalue=round(pvalue, 6),
                    ci_lower=round(ci_lower, 4),
                    ci_upper=round(ci_upper, 4),
                    win_rate=round(threshold_win_rate * 100, 2),
                )
            )

        # Find best threshold (by alpha, among significant ones)
        best = None
        valid_results = [r for r in threshold_results if r.pvalue < 0.05]
        if valid_results:
            best = max(valid_results, key=lambda r: r.mean_alpha)
        elif threshold_results:
            best = max(threshold_results, key=lambda r: r.mean_alpha)

        return FactorResult(
            factor_name=factor_name,
            factor_type="numerical",
            holding_period=holding_period,
            correlation=round(corr, 4) if corr is not None else None,
            correlation_pvalue=round(corr_pvalue, 6) if corr_pvalue is not None else None,
            threshold_results=threshold_results,
            best_threshold=best.threshold if best else None,
            best_threshold_alpha=best.mean_alpha if best else None,
            best_threshold_lift=best.lift if best else None,
            best_threshold_pvalue=best.pvalue if best else None,
            best_threshold_sample_size=best.sample_size if best else None,
            best_threshold_ci_lower=best.ci_lower if best else None,
            best_threshold_ci_upper=best.ci_upper if best else None,
        )

    @staticmethod
    def analyze_categorical(
        data: list[dict],
        factor_name: str,
        categories: list[str],
        min_sample_size: int = 100,
        holding_period: int = 1,
    ) -> FactorResult:
        """
        Analyze a categorical factor.

        Args:
            data: List of observation dicts
            factor_name: Name of the factor
            categories: List of possible categories
            min_sample_size: Minimum sample size per category
            holding_period: Holding period for this analysis

        Returns:
            FactorResult with category analysis
        """
        # Filter to rows with valid factor values
        valid_data = [d for d in data if d.get(factor_name) in categories]

        if len(valid_data) < min_sample_size:
            return FactorResult(
                factor_name=factor_name,
                factor_type="categorical",
                holding_period=holding_period,
                threshold_results=[],
            )

        alphas = [d["alpha"] for d in valid_data]
        overall_win_rate = sum(1 for a in alphas if a > 0) / len(alphas)

        # Analyze each category
        threshold_results = []
        category_alphas = {}

        for cat in categories:
            cat_data = [d for d in valid_data if d.get(factor_name) == cat]
            if len(cat_data) < min_sample_size:
                continue

            cat_alphas = [d["alpha"] for d in cat_data]
            other_alphas = [d["alpha"] for d in valid_data if d.get(factor_name) != cat]

            category_alphas[cat] = cat_alphas

            mean_alpha = float(np.mean(cat_alphas))
            sample_size = len(cat_alphas)

            # Win rate and lift
            cat_win_rate = sum(1 for a in cat_alphas if a > 0) / len(cat_alphas)
            lift = cat_win_rate / overall_win_rate if overall_win_rate > 0 else 1.0

            # T-test vs other categories
            if len(other_alphas) >= 10:
                try:
                    _, pvalue = stats.ttest_ind(cat_alphas, other_alphas, equal_var=False)
                    pvalue = float(pvalue) if not np.isnan(pvalue) else 1.0
                except Exception:
                    pvalue = 1.0
            else:
                pvalue = 1.0

            # Bootstrap CI
            ci_lower, ci_upper = bootstrap_ci(cat_alphas)

            threshold_results.append(
                ThresholdResult(
                    threshold=cat,
                    mean_alpha=round(mean_alpha, 4),
                    sample_size=sample_size,
                    lift=round(lift, 4),
                    pvalue=round(pvalue, 6),
                    ci_lower=round(ci_lower, 4),
                    ci_upper=round(ci_upper, 4),
                    win_rate=round(cat_win_rate * 100, 2),
                )
            )

        # Chi-squared test for overall independence
        # (Not used for p-value, just for correlation equivalent)
        # We'll use the best category as the "threshold"

        # Find best category
        best = None
        valid_results = [r for r in threshold_results if r.pvalue < 0.05]
        if valid_results:
            best = max(valid_results, key=lambda r: r.mean_alpha)
        elif threshold_results:
            best = max(threshold_results, key=lambda r: r.mean_alpha)

        return FactorResult(
            factor_name=factor_name,
            factor_type="categorical",
            holding_period=holding_period,
            correlation=None,  # Not applicable for categorical
            correlation_pvalue=None,
            threshold_results=threshold_results,
            best_threshold=best.threshold if best else None,
            best_threshold_alpha=best.mean_alpha if best else None,
            best_threshold_lift=best.lift if best else None,
            best_threshold_pvalue=best.pvalue if best else None,
            best_threshold_sample_size=best.sample_size if best else None,
            best_threshold_ci_lower=best.ci_lower if best else None,
            best_threshold_ci_upper=best.ci_upper if best else None,
        )

    @staticmethod
    def analyze_boolean(
        data: list[dict],
        factor_name: str,
        positive: bool = True,
        min_sample_size: int = 100,
        holding_period: int = 1,
    ) -> FactorResult:
        """
        Analyze a boolean factor (has tag vs doesn't have).

        Args:
            data: List of observation dicts
            factor_name: Name of the boolean factor
            positive: If True, "has" is expected to be better; if False, "doesn't have" is better
            min_sample_size: Minimum sample size
            holding_period: Holding period for this analysis

        Returns:
            FactorResult with True/False analysis
        """
        if len(data) < min_sample_size:
            return FactorResult(
                factor_name=factor_name,
                factor_type="boolean",
                holding_period=holding_period,
                threshold_results=[],
            )

        alphas = [d["alpha"] for d in data]
        overall_win_rate = sum(1 for a in alphas if a > 0) / len(alphas)

        # Split by True/False
        true_data = [d for d in data if d.get(factor_name) is True]
        false_data = [d for d in data if d.get(factor_name) is False]

        threshold_results = []

        # Analyze True
        if len(true_data) >= min_sample_size:
            true_alphas = [d["alpha"] for d in true_data]
            false_alphas = [d["alpha"] for d in false_data] if false_data else []

            mean_alpha = float(np.mean(true_alphas))
            sample_size = len(true_alphas)
            win_rate = sum(1 for a in true_alphas if a > 0) / len(true_alphas)
            lift = win_rate / overall_win_rate if overall_win_rate > 0 else 1.0

            # T-test
            if len(false_alphas) >= 10:
                try:
                    _, pvalue = stats.ttest_ind(true_alphas, false_alphas, equal_var=False)
                    pvalue = float(pvalue) if not np.isnan(pvalue) else 1.0
                except Exception:
                    pvalue = 1.0
            else:
                pvalue = 1.0

            ci_lower, ci_upper = bootstrap_ci(true_alphas)

            # For negative tags, "True" means exclude, so label differently
            label = "has" if positive else "exclude"
            threshold_results.append(
                ThresholdResult(
                    threshold=label,
                    mean_alpha=round(mean_alpha, 4),
                    sample_size=sample_size,
                    lift=round(lift, 4),
                    pvalue=round(pvalue, 6),
                    ci_lower=round(ci_lower, 4),
                    ci_upper=round(ci_upper, 4),
                    win_rate=round(win_rate * 100, 2),
                )
            )

        # Analyze False
        if len(false_data) >= min_sample_size:
            false_alphas = [d["alpha"] for d in false_data]
            true_alphas = [d["alpha"] for d in true_data] if true_data else []

            mean_alpha = float(np.mean(false_alphas))
            sample_size = len(false_alphas)
            win_rate = sum(1 for a in false_alphas if a > 0) / len(false_alphas)
            lift = win_rate / overall_win_rate if overall_win_rate > 0 else 1.0

            # T-test
            if len(true_alphas) >= 10:
                try:
                    _, pvalue = stats.ttest_ind(false_alphas, true_alphas, equal_var=False)
                    pvalue = float(pvalue) if not np.isnan(pvalue) else 1.0
                except Exception:
                    pvalue = 1.0
            else:
                pvalue = 1.0

            ci_lower, ci_upper = bootstrap_ci(false_alphas)

            # For negative tags, "False" means allow/don't filter
            label = "doesn't have" if positive else "allow"
            threshold_results.append(
                ThresholdResult(
                    threshold=label,
                    mean_alpha=round(mean_alpha, 4),
                    sample_size=sample_size,
                    lift=round(lift, 4),
                    pvalue=round(pvalue, 6),
                    ci_lower=round(ci_lower, 4),
                    ci_upper=round(ci_upper, 4),
                    win_rate=round(win_rate * 100, 2),
                )
            )

        # Find best (which side has higher alpha)
        best = None
        if threshold_results:
            valid_results = [r for r in threshold_results if r.pvalue < 0.05]
            if valid_results:
                best = max(valid_results, key=lambda r: r.mean_alpha)
            else:
                best = max(threshold_results, key=lambda r: r.mean_alpha)

        return FactorResult(
            factor_name=factor_name,
            factor_type="boolean",
            holding_period=holding_period,
            correlation=None,
            correlation_pvalue=None,
            threshold_results=threshold_results,
            best_threshold=best.threshold if best else None,
            best_threshold_alpha=best.mean_alpha if best else None,
            best_threshold_lift=best.lift if best else None,
            best_threshold_pvalue=best.pvalue if best else None,
            best_threshold_sample_size=best.sample_size if best else None,
            best_threshold_ci_lower=best.ci_lower if best else None,
            best_threshold_ci_upper=best.ci_upper if best else None,
        )

    @classmethod
    def analyze_all_factors(
        cls,
        data: list[dict],
        holding_period: int,
        min_sample_size: int = 100,
    ) -> list[FactorResult]:
        """
        Analyze all configured factors.

        Args:
            data: List of observation dicts
            holding_period: Holding period for this analysis
            min_sample_size: Minimum sample size

        Returns:
            List of FactorResult for all factors
        """
        results = []

        # Numerical factors
        for name, config in cls.NUMERICAL_FACTORS.items():
            result = cls.analyze_numerical(
                data=data,
                factor_name=name,
                thresholds=config["thresholds"],
                direction=config["direction"],
                min_sample_size=min_sample_size,
                holding_period=holding_period,
            )
            results.append(result)

        # Categorical factors
        for name, config in cls.CATEGORICAL_FACTORS.items():
            result = cls.analyze_categorical(
                data=data,
                factor_name=name,
                categories=config["categories"],
                min_sample_size=min_sample_size,
                holding_period=holding_period,
            )
            results.append(result)

        # Boolean factors
        for name, config in cls.BOOLEAN_FACTORS.items():
            result = cls.analyze_boolean(
                data=data,
                factor_name=name,
                positive=config["positive"],
                min_sample_size=min_sample_size,
                holding_period=holding_period,
            )
            results.append(result)

        return results
