"""Recommendation generator for factor discovery.

Converts best factor combinations into Pipeline-ready settings.
"""

import logging
from typing import Optional

from .models import (
    CombinedStrategyResult,
    FactorResult,
    FilterSpec,
    PipelineSettings,
    RecommendedStrategy,
)

logger = logging.getLogger(__name__)


class RecommendationGenerator:
    """
    Generate recommended Pipeline settings from factor analysis.

    Maps discovered factors to the Pipeline UI controls.
    """

    # Mapping of factor names to Pipeline settings
    FACTOR_TO_PIPELINE = {
        # Survival gates
        "piotroski_score": {"group": "survival", "setting": "piotroski"},
        "altman_zone": {"group": "survival", "setting": "altman"},
        "altman_z_score": {"group": "survival", "setting": "altman_zscore"},
        # Quality tags (boolean factors)
        "has_premium_priced": {"group": "quality", "tag": "Premium Priced", "positive": False},
        "has_volatile_returns": {"group": "quality", "tag": "Volatile Returns", "positive": False},
        "has_weak_moat_signal": {"group": "quality", "tag": "Weak Moat Signal", "positive": False},
        "has_earnings_quality_concern": {
            "group": "quality",
            "tag": "Earnings Quality Concern",
            "positive": False,
        },
        "has_durable_compounder": {"group": "quality", "tag": "Durable Compounder", "positive": True},
        "has_cash_machine": {"group": "quality", "tag": "Cash Machine", "positive": True},
        "has_deep_value": {"group": "quality", "tag": "Deep Value", "positive": True},
        "has_heavy_reinvestor": {"group": "quality", "tag": "Heavy Reinvestor", "positive": True},
        # ROIC quality
        "roic": {"group": "quality", "setting": "roic"},
        "fcf_positive_5yr": {"group": "quality", "setting": "fcf"},
        # Valuation lenses
        "graham_score": {"group": "valuation", "setting": "graham"},
        "magic_formula_rank": {"group": "valuation", "setting": "magic_formula"},
        "peg_ratio": {"group": "valuation", "setting": "peg"},
        "trading_below_ncav": {"group": "valuation", "setting": "net_net"},
        "book_to_market_percentile": {"group": "valuation", "setting": "fama_french"},
        "profitability_percentile": {"group": "valuation", "setting": "fama_french"},
        "net_net_discount": {"group": "valuation", "setting": "net_net"},
        "earnings_yield": {"group": "valuation", "setting": "magic_formula"},
    }

    def __init__(
        self,
        factor_results: list[FactorResult],
        combined_results: list[CombinedStrategyResult],
        holding_period: int,
        cost_haircut: float = 3.0,
    ):
        """
        Initialize the recommendation generator.

        Args:
            factor_results: Individual factor analysis results
            combined_results: Combined strategy results
            holding_period: Holding period in quarters
            cost_haircut: Minimum alpha threshold for recommendations
        """
        self.factor_results = factor_results
        self.combined_results = combined_results
        self.holding_period = holding_period
        self.cost_haircut = cost_haircut

        # Build lookup for factor results
        self.factor_lookup = {f.factor_name: f for f in factor_results}

    def generate(self) -> Optional[RecommendedStrategy]:
        """
        Generate the recommended strategy.

        Returns:
            RecommendedStrategy or None if no valid strategy found
        """
        if not self.combined_results:
            logger.warning("No combined results to generate recommendation from")
            return None

        # Find the best strategy that meets the cost haircut
        best_strategy = None
        for strategy in self.combined_results:
            if strategy.ci_lower > self.cost_haircut:
                best_strategy = strategy
                break
            elif strategy.mean_alpha > self.cost_haircut and best_strategy is None:
                best_strategy = strategy

        if best_strategy is None:
            # Fall back to the top strategy if none meet haircut
            best_strategy = self.combined_results[0]
            logger.info(
                f"No strategy meets cost haircut {self.cost_haircut}%, "
                f"using best available with alpha {best_strategy.mean_alpha}%"
            )

        # Convert to Pipeline settings
        pipeline_settings = self._build_pipeline_settings(best_strategy.filters)

        # Calculate confidence score
        confidence_score = self._calculate_confidence(best_strategy)

        # Extract key contributing factors
        key_factors = self._extract_key_factors(best_strategy.filters)

        return RecommendedStrategy(
            holding_period=self.holding_period,
            pipeline_settings=pipeline_settings,
            expected_alpha=best_strategy.mean_alpha,
            expected_alpha_ci_lower=best_strategy.ci_lower,
            expected_alpha_ci_upper=best_strategy.ci_upper,
            expected_win_rate=best_strategy.win_rate,
            sample_size=best_strategy.sample_size,
            confidence_score=round(confidence_score, 2),
            key_factors=key_factors,
            portfolio_stats=best_strategy.portfolio_stats,
        )

    def _build_pipeline_settings(self, filters: list[FilterSpec]) -> PipelineSettings:
        """
        Convert filter specifications to Pipeline settings.

        Args:
            filters: List of filters from the best strategy

        Returns:
            PipelineSettings ready for the Pipeline UI
        """
        settings = PipelineSettings()
        excluded_tags = []
        required_tags = []

        for f in filters:
            mapping = self.FACTOR_TO_PIPELINE.get(f.factor)
            if not mapping:
                logger.warning(f"No Pipeline mapping for factor: {f.factor}")
                continue

            group = mapping.get("group")

            # Handle quality tags
            if "tag" in mapping:
                tag = mapping["tag"]
                is_positive = mapping.get("positive", True)

                if f.operator == "==" and f.value is True:
                    if is_positive:
                        required_tags.append(tag)
                    else:
                        # If "has_premium_priced == True", exclude it
                        excluded_tags.append(tag)
                elif f.operator == "==" and f.value is False:
                    if not is_positive:
                        # If "has_premium_priced == False", that's good (exclude it)
                        excluded_tags.append(tag)
                    # Positive tags with False value means don't require them
                continue

            # Handle survival gates
            if group == "survival":
                setting = mapping.get("setting")

                if setting == "piotroski":
                    settings.piotroski_enabled = True
                    if f.operator == ">=" and isinstance(f.value, (int, float)):
                        settings.piotroski_min = int(f.value)

                elif setting == "altman":
                    settings.altman_enabled = True
                    if f.operator == "==" and isinstance(f.value, str):
                        settings.altman_zone = f.value

                elif setting == "altman_zscore":
                    # Z-score can be mapped to zone
                    settings.altman_enabled = True
                    if isinstance(f.value, (int, float)):
                        if f.value >= 2.99:
                            settings.altman_zone = "safe"
                        elif f.value >= 1.81:
                            settings.altman_zone = "grey"
                        else:
                            settings.altman_zone = "distress"

            # Handle quality settings
            elif group == "quality":
                setting = mapping.get("setting")

                if setting == "roic":
                    settings.quality_enabled = True
                    if f.operator == ">=" and isinstance(f.value, (int, float)):
                        if f.value >= 0.15:
                            settings.min_quality = "compounder"
                        elif f.value >= 0.10:
                            settings.min_quality = "average"
                        else:
                            settings.min_quality = "weak"

                elif setting == "fcf":
                    settings.quality_enabled = True

            # Handle valuation lenses
            elif group == "valuation":
                setting = mapping.get("setting")

                if setting == "graham":
                    settings.graham_enabled = True
                    if f.operator == ">=" and isinstance(f.value, (int, float)):
                        settings.graham_min = int(f.value)

                elif setting == "magic_formula":
                    settings.magic_formula_enabled = True
                    if f.operator == "<=" and isinstance(f.value, (int, float)):
                        # Convert rank to top percentage (rough estimate)
                        # Assuming ~5000 stocks total
                        pct = min(100, int(f.value / 50))
                        settings.mf_top_pct = max(10, pct)

                elif setting == "peg":
                    settings.peg_enabled = True
                    if f.operator == "<=" and isinstance(f.value, (int, float)):
                        settings.max_peg = float(f.value)

                elif setting == "net_net":
                    settings.net_net_enabled = True

                elif setting == "fama_french":
                    settings.fama_french_enabled = True
                    if f.operator == ">=" and isinstance(f.value, (int, float)):
                        # Convert percentile to top percentage
                        settings.ff_top_pct = 100 - int(f.value)

        # Apply collected tags
        if excluded_tags:
            settings.quality_enabled = True
            settings.excluded_tags = excluded_tags
        if required_tags:
            settings.quality_enabled = True
            settings.required_tags = required_tags

        # Set min_lenses if valuation filters were applied
        valuation_enabled = sum([
            settings.graham_enabled,
            settings.magic_formula_enabled,
            settings.peg_enabled,
            settings.net_net_enabled,
            settings.fama_french_enabled,
        ])
        if valuation_enabled > 0:
            settings.min_lenses = 1

        return settings

    def _calculate_confidence(self, strategy: CombinedStrategyResult) -> float:
        """
        Calculate a confidence score for the strategy.

        Based on:
        - Sample size (more samples = higher confidence)
        - Confidence interval width (narrower = higher confidence)
        - Distance from zero (if CI lower bound > 0, more confident)

        Args:
            strategy: The strategy to score

        Returns:
            Confidence score 0.0 to 1.0
        """
        # Sample size component (max out at 500)
        sample_score = min(1.0, strategy.sample_size / 500)

        # CI width component (narrower is better)
        ci_width = strategy.ci_upper - strategy.ci_lower
        # Normalize: 2% width = 1.0, 20% width = 0.2
        ci_score = max(0.1, min(1.0, 4 / (ci_width + 2)))

        # Significance component (CI lower > 0 is important)
        if strategy.ci_lower > 0:
            sig_score = min(1.0, strategy.ci_lower / 5)  # Max out at 5% lower bound
        else:
            sig_score = 0.2  # Penalty for CI crossing zero

        # Win rate component
        win_score = strategy.win_rate / 100  # Already 0-100, normalize to 0-1

        # Weighted average
        confidence = (
            0.25 * sample_score
            + 0.25 * ci_score
            + 0.30 * sig_score
            + 0.20 * win_score
        )

        return min(1.0, max(0.0, confidence))

    def _extract_key_factors(self, filters: list[FilterSpec]) -> list[dict]:
        """
        Extract key contributing factors with their statistics.

        Args:
            filters: Filters from the strategy

        Returns:
            List of factor info dicts
        """
        key_factors = []

        for f in filters:
            factor_result = self.factor_lookup.get(f.factor)

            factor_info = {
                "name": f.factor,
                "threshold": f"{f.operator} {f.value}",
            }

            if factor_result:
                factor_info["lift"] = factor_result.best_threshold_lift
                factor_info["alpha"] = factor_result.best_threshold_alpha
                factor_info["pvalue"] = factor_result.best_threshold_pvalue
                factor_info["sample_size"] = factor_result.best_threshold_sample_size

            key_factors.append(factor_info)

        # Sort by lift (descending)
        key_factors = sorted(
            key_factors, key=lambda x: x.get("lift", 0) or 0, reverse=True
        )

        return key_factors


def generate_all_recommendations(
    factor_results_by_hp: dict[int, list[FactorResult]],
    combined_results_by_hp: dict[int, list[CombinedStrategyResult]],
    cost_haircut: float = 3.0,
) -> dict[int, RecommendedStrategy]:
    """
    Generate recommendations for all holding periods.

    Args:
        factor_results_by_hp: Factor results keyed by holding period
        combined_results_by_hp: Combined results keyed by holding period
        cost_haircut: Minimum alpha threshold

    Returns:
        Dict of holding period -> RecommendedStrategy
    """
    recommendations = {}

    for hp in factor_results_by_hp:
        factor_results = factor_results_by_hp.get(hp, [])
        combined_results = combined_results_by_hp.get(hp, [])

        generator = RecommendationGenerator(
            factor_results=factor_results,
            combined_results=combined_results,
            holding_period=hp,
            cost_haircut=cost_haircut,
        )

        recommendation = generator.generate()
        if recommendation:
            recommendations[hp] = recommendation

    return recommendations
