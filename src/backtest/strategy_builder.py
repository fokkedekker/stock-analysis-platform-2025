"""Strategy combination generator for grid search."""

import hashlib
from itertools import product
from typing import Any, Iterator

from .models import (
    GridDimension,
    QUALITY_TAGS,
    QualityConfig,
    StrategyConfig,
    SurvivalConfig,
    ValuationConfig,
)


def _apply_dimension(strategy: StrategyConfig, name: str, value: Any) -> None:
    """Apply a dimension value to a strategy configuration.

    Auto-enables filters when their parameters are varied.
    """
    # Survival dimensions
    if name == "altman_zone":
        strategy.survival.altman_zone = value
        strategy.survival.altman_enabled = True  # Auto-enable when zone is set
    elif name == "altman_enabled":
        strategy.survival.altman_enabled = value
    elif name == "piotroski_min":
        strategy.survival.piotroski_min = value
        strategy.survival.piotroski_enabled = True  # Auto-enable when min is set
    elif name == "piotroski_enabled":
        strategy.survival.piotroski_enabled = value

    # Quality dimensions
    elif name == "quality_enabled":
        strategy.quality.enabled = value
    elif name == "min_quality":
        strategy.quality.min_quality = value
        strategy.quality.enabled = True  # Auto-enable when min_quality is set

    # Quality tag dimensions - negative tags go to excluded_tags, positive to required_tags
    elif name.startswith("tag_"):
        tag_name = QUALITY_TAGS.get(name)
        if tag_name and value:
            # Negative tags should be EXCLUDED when enabled
            negative_tags = {"Premium Priced", "Volatile Returns", "Weak Moat Signal", "Earnings Quality Concern"}
            if tag_name in negative_tags:
                if tag_name not in strategy.quality.excluded_tags:
                    strategy.quality.excluded_tags.append(tag_name)
            else:
                # Positive tags should be REQUIRED when enabled
                if tag_name not in strategy.quality.required_tags:
                    strategy.quality.required_tags.append(tag_name)

    # Valuation - Graham
    elif name == "graham_enabled":
        strategy.valuation.graham_enabled = value
    elif name == "graham_mode":
        strategy.valuation.graham_mode = value
        strategy.valuation.graham_enabled = True  # Auto-enable when mode is set
        if strategy.valuation.min_lenses < 1:
            strategy.valuation.min_lenses = 1  # Ensure filter applies
    elif name == "graham_min":
        strategy.valuation.graham_min = value
        strategy.valuation.graham_enabled = True  # Auto-enable when min is set
        if strategy.valuation.min_lenses < 1:
            strategy.valuation.min_lenses = 1  # Ensure filter applies

    # Valuation - Magic Formula
    elif name == "magic_formula_enabled":
        strategy.valuation.magic_formula_enabled = value
    elif name == "mf_top_pct":
        strategy.valuation.mf_top_pct = value
        strategy.valuation.magic_formula_enabled = True
        if strategy.valuation.min_lenses < 1:
            strategy.valuation.min_lenses = 1

    # Valuation - PEG
    elif name == "peg_enabled":
        strategy.valuation.peg_enabled = value
    elif name == "max_peg":
        strategy.valuation.max_peg = value
        strategy.valuation.peg_enabled = True
        if strategy.valuation.min_lenses < 1:
            strategy.valuation.min_lenses = 1

    # Valuation - Net-Net
    elif name == "net_net_enabled":
        strategy.valuation.net_net_enabled = value
        if value and strategy.valuation.min_lenses < 1:
            strategy.valuation.min_lenses = 1

    # Valuation - Fama-French
    elif name == "fama_french_enabled":
        strategy.valuation.fama_french_enabled = value
    elif name == "ff_top_pct":
        strategy.valuation.ff_top_pct = value
        strategy.valuation.fama_french_enabled = True
        if strategy.valuation.min_lenses < 1:
            strategy.valuation.min_lenses = 1

    # Valuation logic
    elif name == "min_lenses":
        strategy.valuation.min_lenses = value
    elif name == "strict_mode":
        strategy.valuation.strict_mode = value


def _generate_id(dim_names: list[str], combo: tuple) -> str:
    """Generate a unique ID for a strategy combination."""
    parts = [f"{name}={value}" for name, value in zip(dim_names, combo)]
    id_string = "|".join(parts)
    return hashlib.md5(id_string.encode()).hexdigest()[:12]


def _generate_name(dim_names: list[str], combo: tuple) -> str:
    """Generate a human-readable name for a strategy combination."""
    # Negative tags (exclusion filters)
    negative_tag_names = {
        "tag_premium_priced", "tag_volatile_returns",
        "tag_weak_moat_signal", "tag_earnings_quality_concern"
    }
    # Positive tags (requirement filters)
    positive_tag_names = {
        "tag_durable_compounder", "tag_cash_machine",
        "tag_deep_value", "tag_heavy_reinvestor"
    }

    parts = []
    for name, value in zip(dim_names, combo):
        # Shorten common dimension names
        short_name = name.replace("_enabled", "").replace("_min", "").replace("min_", "")
        short_name = short_name.replace("tag_", "")  # Remove tag_ prefix

        if isinstance(value, bool):
            if name in negative_tag_names:
                # For negative tags: true=exclude, false=allow
                parts.append(f"{short_name}={'exclude' if value else 'allow'}")
            elif name in positive_tag_names:
                # For positive tags: true=require, false=skip
                parts.append(f"{short_name}={'require' if value else 'skip'}")
            else:
                # Generic boolean
                parts.append(f"{short_name}={'on' if value else 'off'}")
        elif isinstance(value, float):
            parts.append(f"{short_name}={value:.1f}")
        else:
            parts.append(f"{short_name}={value}")
    return " | ".join(parts) if parts else "baseline"


def _deep_copy_strategy(strategy: StrategyConfig) -> StrategyConfig:
    """Create a deep copy of a strategy configuration."""
    return StrategyConfig(
        id=strategy.id,
        name=strategy.name,
        survival=SurvivalConfig(
            altman_enabled=strategy.survival.altman_enabled,
            altman_zone=strategy.survival.altman_zone,
            piotroski_enabled=strategy.survival.piotroski_enabled,
            piotroski_min=strategy.survival.piotroski_min,
        ),
        quality=QualityConfig(
            enabled=strategy.quality.enabled,
            min_quality=strategy.quality.min_quality,
            required_tags=list(strategy.quality.required_tags),
            excluded_tags=list(strategy.quality.excluded_tags),
        ),
        valuation=ValuationConfig(
            graham_enabled=strategy.valuation.graham_enabled,
            graham_mode=strategy.valuation.graham_mode,
            graham_min=strategy.valuation.graham_min,
            magic_formula_enabled=strategy.valuation.magic_formula_enabled,
            mf_top_pct=strategy.valuation.mf_top_pct,
            peg_enabled=strategy.valuation.peg_enabled,
            max_peg=strategy.valuation.max_peg,
            net_net_enabled=strategy.valuation.net_net_enabled,
            fama_french_enabled=strategy.valuation.fama_french_enabled,
            ff_top_pct=strategy.valuation.ff_top_pct,
            min_lenses=strategy.valuation.min_lenses,
            strict_mode=strategy.valuation.strict_mode,
        ),
    )


class StrategyBuilder:
    """Generates strategy combinations for grid search."""

    @staticmethod
    def generate_strategies(
        base_strategy: StrategyConfig,
        dimensions: list[GridDimension],
    ) -> Iterator[StrategyConfig]:
        """
        Generate all strategy combinations.

        For each dimension, varies that parameter while keeping others fixed.
        Uses Cartesian product of all dimension values.
        """
        if not dimensions:
            # No dimensions to vary, return base strategy
            strategy = _deep_copy_strategy(base_strategy)
            strategy.id = "baseline"
            strategy.name = "Baseline"
            yield strategy
            return

        # Extract dimension names and values
        dim_names = [d.name for d in dimensions]
        dim_values = [d.values for d in dimensions]

        # Generate Cartesian product
        for combo in product(*dim_values):
            strategy = _deep_copy_strategy(base_strategy)

            # Apply each dimension value
            for name, value in zip(dim_names, combo):
                _apply_dimension(strategy, name, value)

            # Generate unique ID and name
            strategy.id = _generate_id(dim_names, combo)
            strategy.name = _generate_name(dim_names, combo)

            yield strategy

    @staticmethod
    def count_combinations(dimensions: list[GridDimension]) -> int:
        """Count total combinations without generating them."""
        if not dimensions:
            return 1

        count = 1
        for d in dimensions:
            count *= len(d.values)
        return count

    @staticmethod
    def strategy_to_dict(strategy: StrategyConfig) -> dict:
        """Convert strategy config to a serializable dict."""
        return {
            "id": strategy.id,
            "name": strategy.name,
            "survival": {
                "altman_enabled": strategy.survival.altman_enabled,
                "altman_zone": strategy.survival.altman_zone,
                "piotroski_enabled": strategy.survival.piotroski_enabled,
                "piotroski_min": strategy.survival.piotroski_min,
            },
            "quality": {
                "enabled": strategy.quality.enabled,
                "min_quality": strategy.quality.min_quality,
                "required_tags": strategy.quality.required_tags,
                "excluded_tags": strategy.quality.excluded_tags,
            },
            "valuation": {
                "graham_enabled": strategy.valuation.graham_enabled,
                "graham_mode": strategy.valuation.graham_mode,
                "graham_min": strategy.valuation.graham_min,
                "magic_formula_enabled": strategy.valuation.magic_formula_enabled,
                "mf_top_pct": strategy.valuation.mf_top_pct,
                "peg_enabled": strategy.valuation.peg_enabled,
                "max_peg": strategy.valuation.max_peg,
                "net_net_enabled": strategy.valuation.net_net_enabled,
                "fama_french_enabled": strategy.valuation.fama_french_enabled,
                "ff_top_pct": strategy.valuation.ff_top_pct,
                "min_lenses": strategy.valuation.min_lenses,
                "strict_mode": strategy.valuation.strict_mode,
            },
        }
