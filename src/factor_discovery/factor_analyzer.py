"""Factor analyzer for statistical analysis of individual factors.

Implements statistical methods:
- Spearman correlation
- Mean alpha comparison
- Lift ratio
- Welch's t-test
- Bootstrap confidence intervals
- Chi-squared test (for categorical)
- Benjamini-Hochberg FDR correction for multiple hypothesis testing
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


def apply_fdr_correction(
    factor_results: list[FactorResult],
    alpha: float = 0.05,
) -> list[FactorResult]:
    """
    Apply Benjamini-Hochberg FDR correction across ALL threshold tests.

    This corrects for multiple hypothesis testing. When testing 50 factors
    with 5 thresholds each = 250 tests, at p=0.05 you'd expect ~12 false
    positives by chance. FDR correction reduces this false positive rate.

    Args:
        factor_results: List of FactorResult from all factors analyzed
        alpha: Significance level (default 0.05)

    Returns:
        Updated factor_results with fdr_significant and adjusted_pvalue set
    """
    # Collect all p-values with their locations
    pvalue_locations: list[tuple[int, int, float]] = []  # (factor_idx, threshold_idx, pvalue)

    for f_idx, factor in enumerate(factor_results):
        for t_idx, threshold in enumerate(factor.threshold_results):
            if threshold.pvalue is not None:
                pvalue_locations.append((f_idx, t_idx, threshold.pvalue))

    if not pvalue_locations:
        return factor_results

    # Sort by p-value (ascending) for BH procedure
    pvalue_locations.sort(key=lambda x: x[2])

    n_tests = len(pvalue_locations)
    logger.info(f"Applying FDR correction to {n_tests} hypothesis tests")

    # Benjamini-Hochberg procedure:
    # For sorted p-values p(1) <= p(2) <= ... <= p(n)
    # Find largest k where p(k) <= (k/n) * alpha
    # Reject all hypotheses with p-value <= p(k)

    # Calculate BH critical values and adjusted p-values
    rejection_threshold = 0.0
    for rank, (f_idx, t_idx, pvalue) in enumerate(pvalue_locations, start=1):
        # BH critical value for this rank
        bh_critical = (rank / n_tests) * alpha

        # BH-adjusted p-value: p_adj = p * n / rank (capped at 1.0)
        adjusted_pvalue = min(pvalue * n_tests / rank, 1.0)

        # Update the threshold result
        threshold_result = factor_results[f_idx].threshold_results[t_idx]
        threshold_result.adjusted_pvalue = round(adjusted_pvalue, 6)

        # Track the largest p-value that passes BH criterion
        if pvalue <= bh_critical:
            rejection_threshold = pvalue

    # Now mark all with p-value <= rejection_threshold as significant
    n_significant = 0
    for f_idx, t_idx, pvalue in pvalue_locations:
        threshold_result = factor_results[f_idx].threshold_results[t_idx]
        threshold_result.fdr_significant = pvalue <= rejection_threshold
        if threshold_result.fdr_significant:
            n_significant += 1

    logger.info(
        f"FDR correction: {n_significant}/{n_tests} tests significant "
        f"at alpha={alpha} (rejection threshold p={rejection_threshold:.6f})"
    )

    return factor_results


def reselect_best_thresholds(
    factor_results: list[FactorResult],
    use_fdr: bool = True,
) -> list[FactorResult]:
    """
    Re-select the best threshold for each factor after FDR correction.

    Args:
        factor_results: List of FactorResult with FDR correction applied
        use_fdr: If True, use fdr_significant; if False, use raw p-value < 0.05

    Returns:
        Updated factor_results with best_threshold fields updated
    """
    for factor in factor_results:
        if not factor.threshold_results:
            continue

        # Find best threshold among significant ones
        if use_fdr:
            valid_results = [r for r in factor.threshold_results if r.fdr_significant]
        else:
            valid_results = [r for r in factor.threshold_results if r.pvalue < 0.05]

        if valid_results:
            best = max(valid_results, key=lambda r: r.mean_alpha)
        elif factor.threshold_results:
            # Fall back to best by alpha if none significant
            best = max(factor.threshold_results, key=lambda r: r.mean_alpha)
        else:
            best = None

        # Update factor's best threshold fields
        if best:
            factor.best_threshold = best.threshold
            factor.best_threshold_alpha = best.mean_alpha
            factor.best_threshold_lift = best.lift
            factor.best_threshold_pvalue = best.pvalue
            factor.best_threshold_sample_size = best.sample_size
            factor.best_threshold_ci_lower = best.ci_lower
            factor.best_threshold_ci_upper = best.ci_upper
            factor.best_threshold_fdr_significant = best.fdr_significant

    return factor_results


class FactorAnalyzer:
    """Analyzes individual factors for predictive power."""

    # =========================================================================
    # Pre-computed Scores (existing)
    # =========================================================================
    SCORE_FACTORS = {
        "piotroski_score": {
            "thresholds": [3, 4, 5, 6, 7, 8],
            "direction": ">=",
            "label": "Piotroski F-Score",
            "category": "scores",
        },
        "graham_score": {
            "thresholds": [3, 4, 5, 6, 7],
            "direction": ">=",
            "label": "Graham Score",
            "category": "scores",
        },
        "altman_z_score": {
            "thresholds": [1.8, 2.0, 2.5, 3.0],
            "direction": ">=",
            "label": "Altman Z-Score",
            "category": "scores",
        },
        "roic": {
            "thresholds": [0.08, 0.10, 0.12, 0.15, 0.20],
            "direction": ">=",
            "label": "ROIC (Analysis)",
            "category": "scores",
        },
        "peg_ratio": {
            "thresholds": [0.5, 1.0, 1.5, 2.0, 2.5, 3.0],
            "direction": "<=",
            "label": "PEG Ratio",
            "category": "scores",
        },
        "magic_formula_rank": {
            "thresholds": [100, 200, 500, 1000],
            "direction": "<=",
            "label": "Magic Formula Rank",
            "category": "scores",
        },
        "book_to_market_percentile": {
            "thresholds": [0.50, 0.60, 0.70, 0.80],
            "direction": ">=",
            "label": "Book/Market Percentile",
            "category": "scores",
        },
    }

    # =========================================================================
    # Raw Valuation Metrics (from key_metrics)
    # =========================================================================
    RAW_VALUATION_FACTORS = {
        "pe_ratio": {
            "thresholds": [5, 10, 15, 20, 30],
            "direction": "<=",
            "label": "P/E Ratio",
            "category": "raw_valuation",
        },
        "pb_ratio": {
            "thresholds": [0.5, 1.0, 1.5, 2.0, 3.0],
            "direction": "<=",
            "label": "P/B Ratio",
            "category": "raw_valuation",
        },
        "price_to_sales": {
            "thresholds": [0.5, 1.0, 2.0, 3.0, 5.0],
            "direction": "<=",
            "label": "P/S Ratio",
            "category": "raw_valuation",
        },
        "price_to_free_cash_flow": {
            "thresholds": [5, 10, 15, 20, 30],
            "direction": "<=",
            "label": "P/FCF Ratio",
            "category": "raw_valuation",
        },
        "price_to_operating_cash_flow": {
            "thresholds": [5, 10, 15, 20],
            "direction": "<=",
            "label": "P/OCF Ratio",
            "category": "raw_valuation",
        },
        "ev_to_ebitda": {
            "thresholds": [5, 8, 10, 12, 15],
            "direction": "<=",
            "label": "EV/EBITDA",
            "category": "raw_valuation",
        },
        "ev_to_sales": {
            "thresholds": [1, 2, 3, 5, 8],
            "direction": "<=",
            "label": "EV/Sales",
            "category": "raw_valuation",
        },
        "ev_to_free_cash_flow": {
            "thresholds": [10, 15, 20, 25, 30],
            "direction": "<=",
            "label": "EV/FCF",
            "category": "raw_valuation",
        },
        "ev_to_operating_cash_flow": {
            "thresholds": [8, 12, 16, 20],
            "direction": "<=",
            "label": "EV/OCF",
            "category": "raw_valuation",
        },
    }

    # =========================================================================
    # Raw Profitability Metrics (from key_metrics)
    # =========================================================================
    RAW_PROFITABILITY_FACTORS = {
        "roe": {
            "thresholds": [0.05, 0.10, 0.15, 0.20, 0.25],
            "direction": ">=",
            "label": "ROE",
            "category": "raw_profitability",
        },
        "roa": {
            "thresholds": [0.03, 0.05, 0.08, 0.10, 0.15],
            "direction": ">=",
            "label": "ROA",
            "category": "raw_profitability",
        },
        "return_on_tangible_assets": {
            "thresholds": [0.05, 0.10, 0.15, 0.20],
            "direction": ">=",
            "label": "ROTA",
            "category": "raw_profitability",
        },
        "gross_profit_margin": {
            "thresholds": [0.20, 0.30, 0.40, 0.50, 0.60],
            "direction": ">=",
            "label": "Gross Margin",
            "category": "raw_profitability",
        },
        "operating_profit_margin": {
            "thresholds": [0.05, 0.10, 0.15, 0.20, 0.25],
            "direction": ">=",
            "label": "Operating Margin",
            "category": "raw_profitability",
        },
        "net_profit_margin": {
            "thresholds": [0.03, 0.05, 0.08, 0.10, 0.15],
            "direction": ">=",
            "label": "Net Margin",
            "category": "raw_profitability",
        },
    }

    # =========================================================================
    # Raw Liquidity Metrics (from key_metrics)
    # =========================================================================
    RAW_LIQUIDITY_FACTORS = {
        "current_ratio": {
            "thresholds": [1.0, 1.5, 2.0, 2.5, 3.0],
            "direction": ">=",
            "label": "Current Ratio",
            "category": "raw_liquidity",
        },
        "quick_ratio": {
            "thresholds": [0.5, 1.0, 1.5, 2.0],
            "direction": ">=",
            "label": "Quick Ratio",
            "category": "raw_liquidity",
        },
        "cash_ratio": {
            "thresholds": [0.1, 0.2, 0.3, 0.5],
            "direction": ">=",
            "label": "Cash Ratio",
            "category": "raw_liquidity",
        },
    }

    # =========================================================================
    # Raw Leverage Metrics (from key_metrics)
    # =========================================================================
    RAW_LEVERAGE_FACTORS = {
        "debt_ratio": {
            "thresholds": [0.3, 0.4, 0.5, 0.6, 0.7],
            "direction": "<=",
            "label": "Debt Ratio",
            "category": "raw_leverage",
        },
        "debt_to_equity": {
            "thresholds": [0.25, 0.50, 1.0, 1.5, 2.0],
            "direction": "<=",
            "label": "Debt/Equity",
            "category": "raw_leverage",
        },
        "debt_to_assets": {
            "thresholds": [0.2, 0.3, 0.4, 0.5, 0.6],
            "direction": "<=",
            "label": "Debt/Assets",
            "category": "raw_leverage",
        },
        "net_debt_to_ebitda": {
            "thresholds": [1, 2, 3, 4, 5],
            "direction": "<=",
            "label": "Net Debt/EBITDA",
            "category": "raw_leverage",
        },
        "interest_coverage": {
            "thresholds": [2, 4, 6, 8, 10],
            "direction": ">=",
            "label": "Interest Coverage",
            "category": "raw_leverage",
        },
    }

    # =========================================================================
    # Raw Efficiency Metrics (from key_metrics)
    # =========================================================================
    RAW_EFFICIENCY_FACTORS = {
        "asset_turnover": {
            "thresholds": [0.5, 1.0, 1.5, 2.0],
            "direction": ">=",
            "label": "Asset Turnover",
            "category": "raw_efficiency",
        },
        "inventory_turnover": {
            "thresholds": [3, 5, 7, 10, 15],
            "direction": ">=",
            "label": "Inventory Turnover",
            "category": "raw_efficiency",
        },
        "receivables_turnover": {
            "thresholds": [5, 8, 10, 12, 15],
            "direction": ">=",
            "label": "Receivables Turnover",
            "category": "raw_efficiency",
        },
        "payables_turnover": {
            "thresholds": [4, 6, 8, 10, 12],
            "direction": ">=",
            "label": "Payables Turnover",
            "category": "raw_efficiency",
        },
    }

    # =========================================================================
    # Raw Dividend Metrics (from key_metrics)
    # =========================================================================
    RAW_DIVIDEND_FACTORS = {
        "dividend_yield": {
            "thresholds": [0.01, 0.02, 0.03, 0.04, 0.05],
            "direction": ">=",
            "label": "Dividend Yield",
            "category": "raw_dividend",
        },
        "payout_ratio": {
            "thresholds": [0.30, 0.50, 0.70, 0.90],
            "direction": "<=",
            "label": "Payout Ratio",
            "category": "raw_dividend",
        },
    }

    # =========================================================================
    # Stability Metrics (from roic_quality_results)
    # =========================================================================
    STABILITY_FACTORS = {
        "roic_std_dev": {
            "thresholds": [0.02, 0.05, 0.08, 0.10],
            "direction": "<=",
            "label": "ROIC Std Dev",
            "category": "stability",
        },
        "gross_margin_std_dev": {
            "thresholds": [0.02, 0.05, 0.08, 0.10],
            "direction": "<=",
            "label": "Gross Margin Std Dev",
            "category": "stability",
        },
        "fcf_to_net_income": {
            "thresholds": [0.5, 0.8, 1.0, 1.2],
            "direction": ">=",
            "label": "FCF/Net Income",
            "category": "stability",
        },
        "reinvestment_rate": {
            "thresholds": [0.2, 0.4, 0.6, 0.8],
            "direction": ">=",
            "label": "Reinvestment Rate",
            "category": "stability",
        },
        "fcf_yield": {
            "thresholds": [0.03, 0.05, 0.08, 0.10],
            "direction": ">=",
            "label": "FCF Yield",
            "category": "stability",
        },
    }

    # =========================================================================
    # Growth Metrics (from garp_peg_results)
    # =========================================================================
    GROWTH_FACTORS = {
        "eps_growth_1yr": {
            "thresholds": [0.05, 0.10, 0.15, 0.20, 0.30],
            "direction": ">=",
            "label": "EPS Growth 1yr",
            "category": "growth",
        },
        "eps_growth_3yr": {
            "thresholds": [0.05, 0.10, 0.15, 0.20],
            "direction": ">=",
            "label": "EPS Growth 3yr",
            "category": "growth",
        },
        "eps_growth_5yr": {
            "thresholds": [0.05, 0.10, 0.15],
            "direction": ">=",
            "label": "EPS Growth 5yr",
            "category": "growth",
        },
        "eps_cagr": {
            "thresholds": [0.05, 0.10, 0.15, 0.20],
            "direction": ">=",
            "label": "EPS CAGR",
            "category": "growth",
        },
    }

    # =========================================================================
    # Combined: All Numerical Factors
    # =========================================================================
    NUMERICAL_FACTORS = {
        **SCORE_FACTORS,
        **RAW_VALUATION_FACTORS,
        **RAW_PROFITABILITY_FACTORS,
        **RAW_LIQUIDITY_FACTORS,
        **RAW_LEVERAGE_FACTORS,
        **RAW_EFFICIENCY_FACTORS,
        **RAW_DIVIDEND_FACTORS,
        **STABILITY_FACTORS,
        **GROWTH_FACTORS,
    }

    # =========================================================================
    # Factor Categories (for UI)
    # =========================================================================
    FACTOR_CATEGORIES = [
        {"id": "scores", "label": "Pre-computed Scores", "count": len(SCORE_FACTORS)},
        {"id": "raw_valuation", "label": "Valuation Ratios", "count": len(RAW_VALUATION_FACTORS)},
        {"id": "raw_profitability", "label": "Profitability", "count": len(RAW_PROFITABILITY_FACTORS)},
        {"id": "raw_liquidity", "label": "Liquidity", "count": len(RAW_LIQUIDITY_FACTORS)},
        {"id": "raw_leverage", "label": "Leverage", "count": len(RAW_LEVERAGE_FACTORS)},
        {"id": "raw_efficiency", "label": "Efficiency", "count": len(RAW_EFFICIENCY_FACTORS)},
        {"id": "raw_dividend", "label": "Dividends", "count": len(RAW_DIVIDEND_FACTORS)},
        {"id": "stability", "label": "Stability", "count": len(STABILITY_FACTORS)},
        {"id": "growth", "label": "Growth", "count": len(GROWTH_FACTORS)},
        {"id": "boolean", "label": "Quality Tags", "count": 10},  # Boolean factors
    ]

    CATEGORICAL_FACTORS = {
        "altman_zone": {
            "categories": ["safe", "grey", "distress"],
            "label": "Altman Zone",
            "category": "scores",
        },
    }

    # Boolean factors: has_X means "stock has this tag"
    BOOLEAN_FACTORS = {
        # Positive tags (require = good)
        "has_durable_compounder": {"label": "Durable Compounder", "positive": True, "category": "boolean"},
        "has_cash_machine": {"label": "Cash Machine", "positive": True, "category": "boolean"},
        "has_deep_value": {"label": "Deep Value", "positive": True, "category": "boolean"},
        "has_heavy_reinvestor": {"label": "Heavy Reinvestor", "positive": True, "category": "boolean"},
        # Negative tags (exclude = good)
        "has_premium_priced": {"label": "Premium Priced", "positive": False, "category": "boolean"},
        "has_volatile_returns": {"label": "Volatile Returns", "positive": False, "category": "boolean"},
        "has_weak_moat_signal": {"label": "Weak Moat Signal", "positive": False, "category": "boolean"},
        "has_earnings_quality_concern": {"label": "Earnings Quality Concern", "positive": False, "category": "boolean"},
        # Other booleans
        "trading_below_ncav": {"label": "Trading Below NCAV", "positive": True, "category": "boolean"},
        "fcf_positive_5yr": {"label": "FCF Positive 5yr", "positive": True, "category": "boolean"},
    }

    @classmethod
    def get_factors_by_category(cls, categories: list[str]) -> dict:
        """
        Get factors filtered by category.

        Args:
            categories: List of category IDs to include

        Returns:
            Dict with 'numerical', 'categorical', 'boolean' keys
        """
        numerical = {
            name: config
            for name, config in cls.NUMERICAL_FACTORS.items()
            if config.get("category", "scores") in categories
        }

        categorical = {
            name: config
            for name, config in cls.CATEGORICAL_FACTORS.items()
            if config.get("category", "scores") in categories
        }

        boolean = {
            name: config
            for name, config in cls.BOOLEAN_FACTORS.items()
            if config.get("category", "boolean") in categories
        }

        return {
            "numerical": numerical,
            "categorical": categorical,
            "boolean": boolean,
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
        apply_fdr: bool = True,
        fdr_alpha: float = 0.05,
    ) -> list[FactorResult]:
        """
        Analyze all configured factors.

        Args:
            data: List of observation dicts
            holding_period: Holding period for this analysis
            min_sample_size: Minimum sample size
            apply_fdr: Whether to apply FDR correction for multiple testing
            fdr_alpha: Significance level for FDR correction

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

        # Apply FDR correction across ALL threshold tests
        if apply_fdr and results:
            results = apply_fdr_correction(results, alpha=fdr_alpha)
            results = reselect_best_thresholds(results, use_fdr=True)

        return results
