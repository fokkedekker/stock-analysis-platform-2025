"""Storage module for factor discovery results.

Handles saving and loading analysis results from the database.
"""

import json
import logging
from datetime import datetime
from typing import Optional

from src.database.connection import get_db_manager

from .models import (
    CombinedStrategyResult,
    FactorDiscoveryRequest,
    FactorDiscoveryResult,
    FactorDiscoverySummary,
    FactorResult,
    FilterSpec,
    PipelineSettings,
    RecommendedStrategy,
    ThresholdResult,
)

logger = logging.getLogger(__name__)


class FactorDiscoveryStorage:
    """Handles persistence of factor discovery results."""

    def save_result(self, result: FactorDiscoveryResult) -> None:
        """
        Save a complete factor discovery result to the database.

        Args:
            result: The result to save
        """
        db = get_db_manager()

        with db.get_connection() as conn:
            # 1. Save run metadata
            self._save_run(conn, result)

            # 2. Save factor results
            for hp, factors in result.factor_results.items():
                for factor in factors:
                    self._save_factor_result(conn, result.run_id, hp, factor)

            # 3. Save combined strategy results
            for hp, strategies in result.combined_results.items():
                for rank, strategy in enumerate(strategies, 1):
                    self._save_combined_result(conn, result.run_id, hp, rank, strategy)

            # 4. Save recommended strategies
            for hp, recommendation in result.recommended_strategies.items():
                self._save_recommendation(conn, result.run_id, hp, recommendation)

            logger.info(f"Saved factor discovery result {result.run_id}")

    def _save_run(self, conn, result: FactorDiscoveryResult) -> None:
        """Save the run metadata."""
        conn.execute(
            """
            INSERT INTO factor_analysis_runs (
                id, created_at, completed_at, status,
                quarters_analyzed, holding_periods, total_observations,
                config, error_message, duration_seconds
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT (id) DO UPDATE SET
                completed_at = EXCLUDED.completed_at,
                status = EXCLUDED.status,
                total_observations = EXCLUDED.total_observations,
                error_message = EXCLUDED.error_message,
                duration_seconds = EXCLUDED.duration_seconds
            """,
            (
                result.run_id,
                result.created_at,
                result.completed_at,
                result.status,
                json.dumps(result.config.quarters),
                json.dumps(result.config.holding_periods),
                result.total_observations,
                result.config.model_dump_json(),
                None,
                result.duration_seconds,
            ),
        )

    def _save_factor_result(
        self, conn, run_id: str, holding_period: int, factor: FactorResult
    ) -> None:
        """Save an individual factor result."""
        # Serialize threshold results to JSON
        threshold_json = json.dumps(
            [t.model_dump() for t in factor.threshold_results]
        )

        conn.execute(
            """
            INSERT INTO factor_results (
                run_id, holding_period, factor_name, factor_type,
                correlation, correlation_pvalue, threshold_results,
                best_threshold, best_threshold_alpha, best_threshold_lift,
                best_threshold_pvalue, best_threshold_sample_size,
                best_threshold_ci_lower, best_threshold_ci_upper
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT (run_id, holding_period, factor_name) DO UPDATE SET
                correlation = EXCLUDED.correlation,
                correlation_pvalue = EXCLUDED.correlation_pvalue,
                threshold_results = EXCLUDED.threshold_results,
                best_threshold = EXCLUDED.best_threshold,
                best_threshold_alpha = EXCLUDED.best_threshold_alpha,
                best_threshold_lift = EXCLUDED.best_threshold_lift,
                best_threshold_pvalue = EXCLUDED.best_threshold_pvalue,
                best_threshold_sample_size = EXCLUDED.best_threshold_sample_size,
                best_threshold_ci_lower = EXCLUDED.best_threshold_ci_lower,
                best_threshold_ci_upper = EXCLUDED.best_threshold_ci_upper
            """,
            (
                run_id,
                holding_period,
                factor.factor_name,
                factor.factor_type,
                factor.correlation,
                factor.correlation_pvalue,
                threshold_json,
                factor.best_threshold,
                factor.best_threshold_alpha,
                factor.best_threshold_lift,
                factor.best_threshold_pvalue,
                factor.best_threshold_sample_size,
                factor.best_threshold_ci_lower,
                factor.best_threshold_ci_upper,
            ),
        )

    def _save_combined_result(
        self,
        conn,
        run_id: str,
        holding_period: int,
        rank: int,
        strategy: CombinedStrategyResult,
    ) -> None:
        """Save a combined strategy result."""
        filters_json = json.dumps([f.model_dump() for f in strategy.filters])

        conn.execute(
            """
            INSERT INTO combined_strategy_results (
                run_id, holding_period, strategy_rank,
                filters, mean_alpha, sample_size, lift,
                win_rate, ci_lower, ci_upper
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT (run_id, holding_period, strategy_rank) DO UPDATE SET
                filters = EXCLUDED.filters,
                mean_alpha = EXCLUDED.mean_alpha,
                sample_size = EXCLUDED.sample_size,
                lift = EXCLUDED.lift,
                win_rate = EXCLUDED.win_rate,
                ci_lower = EXCLUDED.ci_lower,
                ci_upper = EXCLUDED.ci_upper
            """,
            (
                run_id,
                holding_period,
                rank,
                filters_json,
                strategy.mean_alpha,
                strategy.sample_size,
                strategy.lift,
                strategy.win_rate,
                strategy.ci_lower,
                strategy.ci_upper,
            ),
        )

    def _save_recommendation(
        self,
        conn,
        run_id: str,
        holding_period: int,
        recommendation: RecommendedStrategy,
    ) -> None:
        """Save a recommended strategy."""
        pipeline_json = recommendation.pipeline_settings.model_dump_json()
        key_factors_json = json.dumps(recommendation.key_factors)

        conn.execute(
            """
            INSERT INTO recommended_strategies (
                run_id, holding_period, pipeline_settings,
                expected_alpha, expected_alpha_ci_lower, expected_alpha_ci_upper,
                expected_win_rate, sample_size, confidence_score, key_factors
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT (run_id, holding_period) DO UPDATE SET
                pipeline_settings = EXCLUDED.pipeline_settings,
                expected_alpha = EXCLUDED.expected_alpha,
                expected_alpha_ci_lower = EXCLUDED.expected_alpha_ci_lower,
                expected_alpha_ci_upper = EXCLUDED.expected_alpha_ci_upper,
                expected_win_rate = EXCLUDED.expected_win_rate,
                sample_size = EXCLUDED.sample_size,
                confidence_score = EXCLUDED.confidence_score,
                key_factors = EXCLUDED.key_factors
            """,
            (
                run_id,
                holding_period,
                pipeline_json,
                recommendation.expected_alpha,
                recommendation.expected_alpha_ci_lower,
                recommendation.expected_alpha_ci_upper,
                recommendation.expected_win_rate,
                recommendation.sample_size,
                recommendation.confidence_score,
                key_factors_json,
            ),
        )

    def load_result(self, run_id: str) -> Optional[FactorDiscoveryResult]:
        """
        Load a complete factor discovery result from the database.

        Args:
            run_id: The run ID to load

        Returns:
            FactorDiscoveryResult or None if not found
        """
        db = get_db_manager()

        with db.get_connection() as conn:
            # 1. Load run metadata
            run_row = conn.execute(
                """
                SELECT id, created_at, completed_at, status,
                       quarters_analyzed, holding_periods, total_observations,
                       config, error_message, duration_seconds
                FROM factor_analysis_runs
                WHERE id = ?
                """,
                (run_id,),
            ).fetchone()

            if not run_row:
                return None

            # Parse run data
            run_data = dict(
                zip(
                    [
                        "id",
                        "created_at",
                        "completed_at",
                        "status",
                        "quarters_analyzed",
                        "holding_periods",
                        "total_observations",
                        "config",
                        "error_message",
                        "duration_seconds",
                    ],
                    run_row,
                )
            )

            config = FactorDiscoveryRequest.model_validate_json(run_data["config"])
            holding_periods = json.loads(run_data["holding_periods"])

            # 2. Load factor results
            factor_results = self._load_factor_results(conn, run_id, holding_periods)

            # 3. Load combined results
            combined_results = self._load_combined_results(conn, run_id, holding_periods)

            # 4. Load recommendations
            recommended_strategies = self._load_recommendations(
                conn, run_id, holding_periods
            )

            # Find best
            best_hp = None
            best_alpha = None
            for hp, rec in recommended_strategies.items():
                if best_alpha is None or rec.expected_alpha > best_alpha:
                    best_hp = hp
                    best_alpha = rec.expected_alpha

            return FactorDiscoveryResult(
                run_id=run_data["id"],
                status=run_data["status"],
                created_at=run_data["created_at"],
                completed_at=run_data["completed_at"],
                duration_seconds=run_data["duration_seconds"],
                config=config,
                total_observations=run_data["total_observations"] or 0,
                factor_results=factor_results,
                combined_results=combined_results,
                recommended_strategies=recommended_strategies,
                best_holding_period=best_hp,
                best_alpha=best_alpha,
            )

    def _load_factor_results(
        self, conn, run_id: str, holding_periods: list[int]
    ) -> dict[int, list[FactorResult]]:
        """Load factor results from database."""
        results: dict[int, list[FactorResult]] = {hp: [] for hp in holding_periods}

        rows = conn.execute(
            """
            SELECT holding_period, factor_name, factor_type,
                   correlation, correlation_pvalue, threshold_results,
                   best_threshold, best_threshold_alpha, best_threshold_lift,
                   best_threshold_pvalue, best_threshold_sample_size,
                   best_threshold_ci_lower, best_threshold_ci_upper
            FROM factor_results
            WHERE run_id = ?
            ORDER BY holding_period, factor_name
            """,
            (run_id,),
        ).fetchall()

        for row in rows:
            data = dict(
                zip(
                    [
                        "holding_period",
                        "factor_name",
                        "factor_type",
                        "correlation",
                        "correlation_pvalue",
                        "threshold_results",
                        "best_threshold",
                        "best_threshold_alpha",
                        "best_threshold_lift",
                        "best_threshold_pvalue",
                        "best_threshold_sample_size",
                        "best_threshold_ci_lower",
                        "best_threshold_ci_upper",
                    ],
                    row,
                )
            )

            # Parse threshold results
            threshold_data = json.loads(data["threshold_results"] or "[]")
            threshold_results = [ThresholdResult(**t) for t in threshold_data]

            factor = FactorResult(
                factor_name=data["factor_name"],
                factor_type=data["factor_type"],
                holding_period=data["holding_period"],
                correlation=data["correlation"],
                correlation_pvalue=data["correlation_pvalue"],
                threshold_results=threshold_results,
                best_threshold=data["best_threshold"],
                best_threshold_alpha=data["best_threshold_alpha"],
                best_threshold_lift=data["best_threshold_lift"],
                best_threshold_pvalue=data["best_threshold_pvalue"],
                best_threshold_sample_size=data["best_threshold_sample_size"],
                best_threshold_ci_lower=data["best_threshold_ci_lower"],
                best_threshold_ci_upper=data["best_threshold_ci_upper"],
            )

            hp = data["holding_period"]
            if hp in results:
                results[hp].append(factor)

        return results

    def _load_combined_results(
        self, conn, run_id: str, holding_periods: list[int]
    ) -> dict[int, list[CombinedStrategyResult]]:
        """Load combined strategy results from database."""
        results: dict[int, list[CombinedStrategyResult]] = {
            hp: [] for hp in holding_periods
        }

        rows = conn.execute(
            """
            SELECT holding_period, strategy_rank, filters,
                   mean_alpha, sample_size, lift, win_rate,
                   ci_lower, ci_upper
            FROM combined_strategy_results
            WHERE run_id = ?
            ORDER BY holding_period, strategy_rank
            """,
            (run_id,),
        ).fetchall()

        for row in rows:
            data = dict(
                zip(
                    [
                        "holding_period",
                        "strategy_rank",
                        "filters",
                        "mean_alpha",
                        "sample_size",
                        "lift",
                        "win_rate",
                        "ci_lower",
                        "ci_upper",
                    ],
                    row,
                )
            )

            # Parse filters
            filters_data = json.loads(data["filters"] or "[]")
            filters = [FilterSpec(**f) for f in filters_data]

            strategy = CombinedStrategyResult(
                filters=filters,
                mean_alpha=data["mean_alpha"],
                sample_size=data["sample_size"],
                lift=data["lift"],
                win_rate=data["win_rate"],
                ci_lower=data["ci_lower"],
                ci_upper=data["ci_upper"],
            )

            hp = data["holding_period"]
            if hp in results:
                results[hp].append(strategy)

        return results

    def _load_recommendations(
        self, conn, run_id: str, holding_periods: list[int]
    ) -> dict[int, RecommendedStrategy]:
        """Load recommended strategies from database."""
        results: dict[int, RecommendedStrategy] = {}

        rows = conn.execute(
            """
            SELECT holding_period, pipeline_settings,
                   expected_alpha, expected_alpha_ci_lower, expected_alpha_ci_upper,
                   expected_win_rate, sample_size, confidence_score, key_factors
            FROM recommended_strategies
            WHERE run_id = ?
            """,
            (run_id,),
        ).fetchall()

        for row in rows:
            data = dict(
                zip(
                    [
                        "holding_period",
                        "pipeline_settings",
                        "expected_alpha",
                        "expected_alpha_ci_lower",
                        "expected_alpha_ci_upper",
                        "expected_win_rate",
                        "sample_size",
                        "confidence_score",
                        "key_factors",
                    ],
                    row,
                )
            )

            pipeline_settings = PipelineSettings.model_validate_json(
                data["pipeline_settings"]
            )
            key_factors = json.loads(data["key_factors"] or "[]")

            recommendation = RecommendedStrategy(
                holding_period=data["holding_period"],
                pipeline_settings=pipeline_settings,
                expected_alpha=data["expected_alpha"],
                expected_alpha_ci_lower=data["expected_alpha_ci_lower"],
                expected_alpha_ci_upper=data["expected_alpha_ci_upper"],
                expected_win_rate=data["expected_win_rate"],
                sample_size=data["sample_size"],
                confidence_score=data["confidence_score"],
                key_factors=key_factors,
            )

            results[data["holding_period"]] = recommendation

        return results

    def list_runs(self, limit: int = 50) -> list[FactorDiscoverySummary]:
        """
        List past factor discovery runs.

        Args:
            limit: Maximum number of runs to return

        Returns:
            List of FactorDiscoverySummary objects
        """
        db = get_db_manager()

        with db.get_connection() as conn:
            rows = conn.execute(
                """
                SELECT
                    r.id,
                    r.created_at,
                    r.status,
                    r.total_observations,
                    r.duration_seconds,
                    r.quarters_analyzed,
                    (
                        SELECT MAX(rs.expected_alpha)
                        FROM recommended_strategies rs
                        WHERE rs.run_id = r.id
                    ) as best_alpha,
                    (
                        SELECT rs.holding_period
                        FROM recommended_strategies rs
                        WHERE rs.run_id = r.id
                        ORDER BY rs.expected_alpha DESC
                        LIMIT 1
                    ) as best_holding_period
                FROM factor_analysis_runs r
                ORDER BY r.created_at DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()

            summaries = []
            for row in rows:
                data = dict(
                    zip(
                        [
                            "run_id",
                            "created_at",
                            "status",
                            "total_observations",
                            "duration_seconds",
                            "quarters_analyzed",
                            "best_alpha",
                            "best_holding_period",
                        ],
                        row,
                    )
                )

                quarters = json.loads(data["quarters_analyzed"] or "[]")

                summaries.append(
                    FactorDiscoverySummary(
                        run_id=data["run_id"],
                        created_at=data["created_at"],
                        status=data["status"],
                        quarters_analyzed=len(quarters),
                        best_holding_period=data["best_holding_period"],
                        best_alpha=data["best_alpha"],
                        duration_seconds=data["duration_seconds"],
                    )
                )

            return summaries

    def delete_run(self, run_id: str) -> bool:
        """
        Delete a factor discovery run and all its results.

        Args:
            run_id: The run ID to delete

        Returns:
            True if deleted, False if not found
        """
        db = get_db_manager()

        with db.get_connection() as conn:
            # Check if exists
            exists = conn.execute(
                "SELECT 1 FROM factor_analysis_runs WHERE id = ?",
                (run_id,),
            ).fetchone()

            if not exists:
                return False

            # Delete in order (foreign key friendly)
            conn.execute(
                "DELETE FROM recommended_strategies WHERE run_id = ?",
                (run_id,),
            )
            conn.execute(
                "DELETE FROM combined_strategy_results WHERE run_id = ?",
                (run_id,),
            )
            conn.execute(
                "DELETE FROM factor_results WHERE run_id = ?",
                (run_id,),
            )
            conn.execute(
                "DELETE FROM factor_analysis_runs WHERE id = ?",
                (run_id,),
            )

            logger.info(f"Deleted factor discovery run {run_id}")
            return True

    def get_recommended_strategy(
        self, run_id: str, holding_period: Optional[int] = None
    ) -> Optional[RecommendedStrategy]:
        """
        Get just the recommended strategy for a run.

        Args:
            run_id: The run ID
            holding_period: Specific holding period, or None for best

        Returns:
            RecommendedStrategy or None
        """
        db = get_db_manager()

        with db.get_connection() as conn:
            if holding_period:
                row = conn.execute(
                    """
                    SELECT holding_period, pipeline_settings,
                           expected_alpha, expected_alpha_ci_lower, expected_alpha_ci_upper,
                           expected_win_rate, sample_size, confidence_score, key_factors
                    FROM recommended_strategies
                    WHERE run_id = ? AND holding_period = ?
                    """,
                    (run_id, holding_period),
                ).fetchone()
            else:
                # Get best by expected_alpha
                row = conn.execute(
                    """
                    SELECT holding_period, pipeline_settings,
                           expected_alpha, expected_alpha_ci_lower, expected_alpha_ci_upper,
                           expected_win_rate, sample_size, confidence_score, key_factors
                    FROM recommended_strategies
                    WHERE run_id = ?
                    ORDER BY expected_alpha DESC
                    LIMIT 1
                    """,
                    (run_id,),
                ).fetchone()

            if not row:
                return None

            data = dict(
                zip(
                    [
                        "holding_period",
                        "pipeline_settings",
                        "expected_alpha",
                        "expected_alpha_ci_lower",
                        "expected_alpha_ci_upper",
                        "expected_win_rate",
                        "sample_size",
                        "confidence_score",
                        "key_factors",
                    ],
                    row,
                )
            )

            return RecommendedStrategy(
                holding_period=data["holding_period"],
                pipeline_settings=PipelineSettings.model_validate_json(
                    data["pipeline_settings"]
                ),
                expected_alpha=data["expected_alpha"],
                expected_alpha_ci_lower=data["expected_alpha_ci_lower"],
                expected_alpha_ci_upper=data["expected_alpha_ci_upper"],
                expected_win_rate=data["expected_win_rate"],
                sample_size=data["sample_size"],
                confidence_score=data["confidence_score"],
                key_factors=json.loads(data["key_factors"] or "[]"),
            )


# Singleton instance
_storage: Optional[FactorDiscoveryStorage] = None


def get_storage() -> FactorDiscoveryStorage:
    """Get the singleton storage instance."""
    global _storage
    if _storage is None:
        _storage = FactorDiscoveryStorage()
    return _storage
