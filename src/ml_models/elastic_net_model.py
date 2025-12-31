"""
Elastic Net model for cross-sectional alpha prediction.

Key features:
1. Cross-sectional Z-score normalization per quarter
2. Winsorization at 1st/99th percentile
3. Time-series CV (train on past quarters, validate on future)
4. Coefficient stability tracking across CV folds
"""

import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
from scipy import stats
from sklearn.linear_model import ElasticNetCV

from src.database.connection import get_db_manager
from src.factor_discovery.dataset_builder import DatasetBuilder

logger = logging.getLogger(__name__)


# Features to use in the model - matches factor_analyzer.py NUMERICAL_FACTORS + BOOLEAN_FACTORS
# This ensures consistency between Factor Discovery and Elastic Net

ELASTIC_NET_FEATURES = [
    # =========================================================================
    # Pre-computed Scores (7 factors)
    # =========================================================================
    "piotroski_score",
    "graham_score",
    "altman_z_score",
    "roic",
    "peg_ratio",
    "magic_formula_rank",
    "book_to_market_percentile",

    # =========================================================================
    # Raw Valuation Metrics (9 factors)
    # =========================================================================
    "pe_ratio",
    "pb_ratio",
    "price_to_sales",
    "price_to_free_cash_flow",
    "price_to_operating_cash_flow",
    "ev_to_ebitda",
    "ev_to_sales",
    "ev_to_free_cash_flow",
    "ev_to_operating_cash_flow",

    # =========================================================================
    # Raw Profitability Metrics (6 factors)
    # =========================================================================
    "roe",
    "roa",
    "return_on_tangible_assets",
    "gross_profit_margin",
    "operating_profit_margin",
    "net_profit_margin",

    # =========================================================================
    # Raw Liquidity Metrics (3 factors)
    # =========================================================================
    "current_ratio",
    "quick_ratio",
    "cash_ratio",

    # =========================================================================
    # Raw Leverage Metrics (5 factors)
    # =========================================================================
    "debt_ratio",
    "debt_to_equity",
    "debt_to_assets",
    "net_debt_to_ebitda",
    "interest_coverage",

    # =========================================================================
    # Raw Efficiency Metrics (4 factors)
    # =========================================================================
    "asset_turnover",
    "inventory_turnover",
    "receivables_turnover",
    "payables_turnover",

    # =========================================================================
    # Raw Dividend Metrics (2 factors)
    # =========================================================================
    "dividend_yield",
    "payout_ratio",

    # =========================================================================
    # Stability Metrics (5 factors)
    # =========================================================================
    "roic_std_dev",
    "gross_margin_std_dev",
    "fcf_to_net_income",
    "reinvestment_rate",
    "fcf_yield",

    # =========================================================================
    # Growth Metrics (4 factors)
    # =========================================================================
    "eps_growth_1yr",
    "eps_growth_3yr",
    "eps_growth_5yr",
    "eps_cagr",

    # =========================================================================
    # Regime Factors (1 numerical)
    # =========================================================================
    "rate_momentum",

    # =========================================================================
    # Fama-French Percentiles (2 factors, beyond book_to_market_percentile)
    # =========================================================================
    "profitability_percentile",
    "asset_growth_percentile",

    # =========================================================================
    # Additional derived metrics
    # =========================================================================
    "earnings_yield",  # From Magic Formula

    # =========================================================================
    # Boolean factors (as 0/1 numerical) - 10 factors
    # =========================================================================
    "has_durable_compounder",
    "has_cash_machine",
    "has_deep_value",
    "has_heavy_reinvestor",
    "has_premium_priced",
    "has_volatile_returns",
    "has_weak_moat_signal",
    "has_earnings_quality_concern",
    "trading_below_ncav",
    "fcf_positive_5yr",
]
# Total: 7 + 9 + 6 + 3 + 5 + 4 + 2 + 5 + 4 + 1 + 2 + 1 + 10 = 59 features


@dataclass
class ElasticNetConfig:
    """Configuration for Elastic Net model training."""

    holding_period: int = 4
    quarters: list[str] = field(default_factory=list)
    train_end_quarter: str | None = None
    min_samples_per_quarter: int = 50
    winsorize_percentile: float = 0.01  # 1st/99th
    l1_ratios: list[float] = field(default_factory=lambda: [0.1, 0.5, 0.7, 0.9, 0.95, 1.0])
    cv_folds: int = 5
    features: list[str] = field(default_factory=lambda: ELASTIC_NET_FEATURES.copy())
    run_id: str | None = None  # Optional: pass in to use specific run_id


@dataclass
class CoefficientInfo:
    """Information about a single coefficient."""

    feature_name: str
    coefficient: float
    coefficient_std: float
    stability_score: float  # % of CV folds with same sign
    importance_rank: int


@dataclass
class ICHistoryPoint:
    """IC (Information Coefficient) for one quarter."""

    quarter: str
    ic: float
    ic_pvalue: float
    n_samples: int


@dataclass
class StockPrediction:
    """Prediction for a single stock."""

    symbol: str
    predicted_alpha: float
    predicted_rank: int


@dataclass
class ElasticNetResult:
    """Result of Elastic Net training."""

    run_id: str
    config: ElasticNetConfig
    status: str
    error_message: str | None
    duration_seconds: float
    # Performance metrics
    train_ic: float | None
    test_ic: float | None
    n_train_samples: int
    n_test_samples: int
    # Model parameters
    best_alpha: float | None
    best_l1_ratio: float | None
    n_features_selected: int
    # Coefficients
    coefficients: list[CoefficientInfo]
    # IC over time
    ic_history: list[ICHistoryPoint]
    # Current predictions (latest quarter)
    predictions: list[StockPrediction]
    # Raw data for storage
    train_quarters: list[str]
    test_quarters: list[str]


class ElasticNetModel:
    """
    Elastic Net regression for cross-sectional alpha prediction.

    Methodology:
    1. Load observations from DatasetBuilder
    2. Z-score features cross-sectionally (within each quarter)
    3. Winsorize outliers at 1/99 percentile
    4. Split by time: train on earlier quarters, test on later
    5. Fit ElasticNetCV with time-series CV
    6. Track coefficient stability across folds
    7. Generate predictions for latest quarter
    """

    def __init__(
        self,
        progress_callback: Callable[[dict], None] | None = None,
        cancel_check: Callable[[], bool] | None = None,
    ):
        """
        Initialize model.

        Args:
            progress_callback: Optional callback for progress updates
            cancel_check: Optional function to check if training should be cancelled
        """
        self.progress_callback = progress_callback
        self.cancel_check = cancel_check
        self._model = None
        self._feature_names: list[str] = []
        self._feature_means: dict[str, dict] = {}  # {quarter: {feature: mean}}
        self._feature_stds: dict[str, dict] = {}  # {quarter: {feature: std}}

    def _report_progress(self, stage: str, percent: int, message: str):
        """Report progress if callback is set."""
        if self.progress_callback:
            self.progress_callback(
                {"stage": stage, "percent": percent, "message": message}
            )

    def _check_cancelled(self):
        """Check if training should be cancelled."""
        if self.cancel_check and self.cancel_check():
            raise InterruptedError("Training cancelled by user")

    def train(self, config: ElasticNetConfig) -> ElasticNetResult:
        """
        Train the Elastic Net model.

        Args:
            config: Training configuration

        Returns:
            ElasticNetResult with coefficients, performance metrics, and predictions
        """
        # Use provided run_id or generate a new one
        run_id = config.run_id if config.run_id else str(uuid.uuid4())[:8]
        start_time = time.time()

        try:
            self._report_progress("init", 0, "Loading dataset...")

            # 1. Build dataset using DatasetBuilder
            builder = DatasetBuilder(
                quarters=config.quarters,
                holding_periods=[config.holding_period],
                data_lag_quarters=1,
            )
            dataset = builder.build()
            observations = dataset["data"]

            if not observations:
                raise ValueError("No observations found for given quarters")

            self._report_progress("init", 10, f"Loaded {len(observations)} observations")

            # Filter to selected holding period
            observations = [
                obs for obs in observations if obs["holding_period"] == config.holding_period
            ]

            logger.info(f"After holding period filter: {len(observations)} observations")

            # 2. Prepare feature matrix
            self._report_progress("prepare", 15, "Preparing features...")
            self._check_cancelled()

            X, y, quarters, symbols = self._prepare_data(observations, config.features)
            self._feature_names = config.features

            logger.info(f"Feature matrix shape: {X.shape}")

            # 3. Split by time
            self._report_progress("split", 20, "Splitting train/test...")

            if config.train_end_quarter:
                train_mask = quarters <= config.train_end_quarter
            else:
                # Default: use first 80% of quarters for training
                unique_quarters = sorted(set(quarters))
                split_idx = int(len(unique_quarters) * 0.8)
                train_quarters_set = set(unique_quarters[:split_idx])
                train_mask = np.array([q in train_quarters_set for q in quarters])

            test_mask = ~train_mask

            X_train, y_train = X[train_mask], y[train_mask]
            X_test, y_test = X[test_mask], y[test_mask]
            quarters_train = quarters[train_mask]
            quarters_test = quarters[test_mask]

            train_quarters_list = sorted(set(quarters_train))
            test_quarters_list = sorted(set(quarters_test))

            logger.info(
                f"Train: {len(X_train)} samples, {len(train_quarters_list)} quarters"
            )
            logger.info(
                f"Test: {len(X_test)} samples, {len(test_quarters_list)} quarters"
            )

            if len(X_train) < 100:
                raise ValueError(f"Not enough training samples: {len(X_train)}")

            # 4. Preprocess (Z-score per quarter, winsorize)
            self._report_progress("preprocess", 30, "Preprocessing features...")
            self._check_cancelled()

            X_train_proc = self._preprocess_fit_transform(
                X_train, quarters_train, config.winsorize_percentile
            )
            X_test_proc = self._preprocess_transform(
                X_test, quarters_test, config.winsorize_percentile
            )

            # 5. Fit ElasticNetCV
            self._report_progress("train", 40, "Training model...")
            self._check_cancelled()

            self._model = ElasticNetCV(
                l1_ratio=config.l1_ratios,
                cv=config.cv_folds,
                max_iter=10000,
                n_jobs=-1,
            )
            self._model.fit(X_train_proc, y_train)

            logger.info(f"Best alpha: {self._model.alpha_:.6f}")
            logger.info(f"Best l1_ratio: {self._model.l1_ratio_:.2f}")

            # 6. Calculate coefficient stability
            self._report_progress("stability", 60, "Analyzing coefficient stability...")
            self._check_cancelled()

            coef_stability = self._calculate_coefficient_stability(
                X_train_proc, y_train, quarters_train, config
            )

            # Build coefficient info list
            coefficients = []
            coefs = self._model.coef_
            abs_coefs = np.abs(coefs)
            ranks = (-abs_coefs).argsort().argsort() + 1  # 1 = most important

            for i, feature in enumerate(self._feature_names):
                coefficients.append(
                    CoefficientInfo(
                        feature_name=feature,
                        coefficient=float(coefs[i]),
                        coefficient_std=coef_stability.get(feature, {}).get("std", 0.0),
                        stability_score=coef_stability.get(feature, {}).get(
                            "stability", 0.0
                        ),
                        importance_rank=int(ranks[i]),
                    )
                )

            # Sort by absolute coefficient
            coefficients.sort(key=lambda x: abs(x.coefficient), reverse=True)

            n_features_selected = int((abs_coefs > 1e-6).sum())

            # 7. Calculate IC over time
            self._report_progress("ic", 70, "Calculating IC history...")
            self._check_cancelled()

            ic_history = []

            # Calculate IC for test quarters
            for quarter in sorted(set(quarters_test)):
                q_mask = quarters_test == quarter
                if q_mask.sum() < config.min_samples_per_quarter:
                    continue

                y_pred = self._model.predict(X_test_proc[q_mask])
                y_actual = y_test[q_mask]

                ic, pvalue = stats.spearmanr(y_pred, y_actual)
                if np.isnan(ic):
                    continue

                ic_history.append(
                    ICHistoryPoint(
                        quarter=quarter,
                        ic=float(ic),
                        ic_pvalue=float(pvalue),
                        n_samples=int(q_mask.sum()),
                    )
                )

            # 8. Calculate overall IC
            if len(X_train_proc) > 0:
                y_train_pred = self._model.predict(X_train_proc)
                train_ic, _ = stats.spearmanr(y_train_pred, y_train)
                train_ic = float(train_ic) if not np.isnan(train_ic) else None
            else:
                train_ic = None

            if len(X_test_proc) > 0:
                y_test_pred = self._model.predict(X_test_proc)
                test_ic, _ = stats.spearmanr(y_test_pred, y_test)
                test_ic = float(test_ic) if not np.isnan(test_ic) else None
            else:
                test_ic = None

            # 9. Generate predictions for latest quarter
            self._report_progress("predict", 85, "Generating predictions...")
            self._check_cancelled()

            predictions = self._generate_predictions(
                observations, config, X, quarters, symbols
            )

            self._report_progress("done", 100, "Training complete")

            duration = time.time() - start_time

            return ElasticNetResult(
                run_id=run_id,
                config=config,
                status="completed",
                error_message=None,
                duration_seconds=duration,
                train_ic=train_ic,
                test_ic=test_ic,
                n_train_samples=len(X_train),
                n_test_samples=len(X_test),
                best_alpha=float(self._model.alpha_),
                best_l1_ratio=float(self._model.l1_ratio_),
                n_features_selected=n_features_selected,
                coefficients=coefficients,
                ic_history=ic_history,
                predictions=predictions,
                train_quarters=train_quarters_list,
                test_quarters=test_quarters_list,
            )

        except InterruptedError:
            return ElasticNetResult(
                run_id=run_id,
                config=config,
                status="cancelled",
                error_message="Training cancelled by user",
                duration_seconds=time.time() - start_time,
                train_ic=None,
                test_ic=None,
                n_train_samples=0,
                n_test_samples=0,
                best_alpha=None,
                best_l1_ratio=None,
                n_features_selected=0,
                coefficients=[],
                ic_history=[],
                predictions=[],
                train_quarters=[],
                test_quarters=[],
            )

        except Exception as e:
            logger.exception(f"Error training Elastic Net: {e}")
            return ElasticNetResult(
                run_id=run_id,
                config=config,
                status="failed",
                error_message=str(e),
                duration_seconds=time.time() - start_time,
                train_ic=None,
                test_ic=None,
                n_train_samples=0,
                n_test_samples=0,
                best_alpha=None,
                best_l1_ratio=None,
                n_features_selected=0,
                coefficients=[],
                ic_history=[],
                predictions=[],
                train_quarters=[],
                test_quarters=[],
            )

    def _prepare_data(
        self, observations: list[dict], features: list[str]
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepare feature matrix and target vector.

        Args:
            observations: List of observation dicts from DatasetBuilder
            features: List of feature names to use

        Returns:
            Tuple of (X, y, quarters, symbols) as numpy arrays
        """
        n_samples = len(observations)
        n_features = len(features)

        X = np.full((n_samples, n_features), np.nan)
        y = np.zeros(n_samples)
        quarters = []
        symbols = []

        for i, obs in enumerate(observations):
            y[i] = obs.get("alpha", 0.0)
            quarters.append(obs.get("buy_quarter", ""))
            symbols.append(obs.get("symbol", ""))

            for j, feature in enumerate(features):
                val = obs.get(feature)
                if val is not None:
                    X[i, j] = float(val)

        # Handle missing values: fill with column median
        for j in range(n_features):
            col = X[:, j]
            valid_mask = ~np.isnan(col)
            if valid_mask.sum() > 0:
                median = np.median(col[valid_mask])
                X[~valid_mask, j] = median
            else:
                X[:, j] = 0  # All missing - fill with 0

        return X, y, np.array(quarters), np.array(symbols)

    def _preprocess_fit_transform(
        self, X: np.ndarray, quarters: np.ndarray, winsorize_pct: float
    ) -> np.ndarray:
        """
        Fit preprocessing parameters and transform training data.

        Per-quarter cross-sectional Z-score normalization with winsorization.
        """
        X_proc = X.copy()
        self._feature_means = {}
        self._feature_stds = {}

        for quarter in np.unique(quarters):
            mask = quarters == quarter
            X_q = X_proc[mask]

            # Winsorize
            for col in range(X_q.shape[1]):
                p_low, p_high = np.percentile(
                    X_q[:, col], [winsorize_pct * 100, (1 - winsorize_pct) * 100]
                )
                X_q[:, col] = np.clip(X_q[:, col], p_low, p_high)

            # Z-score
            means = X_q.mean(axis=0)
            stds = X_q.std(axis=0) + 1e-8

            self._feature_means[quarter] = means
            self._feature_stds[quarter] = stds

            X_q = (X_q - means) / stds
            X_proc[mask] = X_q

        return X_proc

    def _preprocess_transform(
        self, X: np.ndarray, quarters: np.ndarray, winsorize_pct: float
    ) -> np.ndarray:
        """
        Transform test data using fitted parameters (or compute new for unseen quarters).
        """
        X_proc = X.copy()

        for quarter in np.unique(quarters):
            mask = quarters == quarter
            X_q = X_proc[mask]

            # Winsorize
            for col in range(X_q.shape[1]):
                p_low, p_high = np.percentile(
                    X_q[:, col], [winsorize_pct * 100, (1 - winsorize_pct) * 100]
                )
                X_q[:, col] = np.clip(X_q[:, col], p_low, p_high)

            # Use stored parameters if available, otherwise compute new
            if quarter in self._feature_means:
                means = self._feature_means[quarter]
                stds = self._feature_stds[quarter]
            else:
                means = X_q.mean(axis=0)
                stds = X_q.std(axis=0) + 1e-8

            X_q = (X_q - means) / stds
            X_proc[mask] = X_q

        return X_proc

    def _calculate_coefficient_stability(
        self,
        X: np.ndarray,
        y: np.ndarray,
        quarters: np.ndarray,
        config: ElasticNetConfig,
    ) -> dict[str, dict]:
        """
        Calculate coefficient stability across time-series CV folds.

        Returns:
            Dict of {feature_name: {"std": float, "stability": float}}
            where stability = % of folds with same sign as final coefficient
        """
        unique_quarters = sorted(set(quarters))
        n_quarters = len(unique_quarters)

        # Need at least some quarters for meaningful stability
        if n_quarters < 5:
            return {f: {"std": 0.0, "stability": 1.0} for f in self._feature_names}

        coef_history = []

        # Rolling window CV: train on 60% of quarters, validate on next 20%
        min_train = max(5, n_quarters // 3)

        for i in range(min_train, n_quarters - 2):
            train_quarters_set = set(unique_quarters[:i])
            train_mask = np.array([q in train_quarters_set for q in quarters])

            if train_mask.sum() < 100:
                continue

            try:
                model = ElasticNetCV(
                    l1_ratio=config.l1_ratios,
                    cv=min(config.cv_folds, 3),
                    max_iter=5000,
                    n_jobs=-1,  # Use all CPU cores
                )
                model.fit(X[train_mask], y[train_mask])
                coef_history.append(model.coef_.copy())
            except Exception:
                continue

        if len(coef_history) < 2:
            return {f: {"std": 0.0, "stability": 1.0} for f in self._feature_names}

        coef_array = np.array(coef_history)  # (n_folds, n_features)
        final_coefs = self._model.coef_

        stability = {}
        for i, feature in enumerate(self._feature_names):
            col_coefs = coef_array[:, i]
            final_sign = np.sign(final_coefs[i]) if final_coefs[i] != 0 else 0

            # % of folds with same sign
            if final_sign != 0:
                same_sign = (np.sign(col_coefs) == final_sign).mean()
            else:
                same_sign = 1.0  # If final is 0, count as stable

            stability[feature] = {
                "std": float(col_coefs.std()),
                "stability": float(same_sign),
            }

        return stability

    def _generate_predictions(
        self,
        observations: list[dict],
        config: ElasticNetConfig,
        X: np.ndarray,
        quarters: np.ndarray,
        symbols: np.ndarray,
    ) -> list[StockPrediction]:
        """Generate predictions for the latest quarter."""
        if self._model is None:
            return []

        # Find latest quarter
        latest_quarter = max(set(quarters))
        latest_mask = quarters == latest_quarter

        if latest_mask.sum() == 0:
            return []

        # Preprocess latest quarter data
        X_latest = X[latest_mask]
        symbols_latest = symbols[latest_mask]

        X_latest_proc = self._preprocess_transform(
            X_latest, np.array([latest_quarter] * len(X_latest)), config.winsorize_percentile
        )

        # Generate predictions
        y_pred = self._model.predict(X_latest_proc)

        # Rank predictions (1 = highest predicted alpha)
        ranks = (-y_pred).argsort().argsort() + 1

        predictions = []
        for i in range(len(symbols_latest)):
            predictions.append(
                StockPrediction(
                    symbol=symbols_latest[i],
                    predicted_alpha=float(y_pred[i]),
                    predicted_rank=int(ranks[i]),
                )
            )

        # Sort by rank
        predictions.sort(key=lambda x: x.predicted_rank)

        return predictions

    def predict(self, X: np.ndarray, quarters: np.ndarray) -> np.ndarray:
        """Predict alpha for new data using fitted model."""
        if self._model is None:
            raise ValueError("Model not trained")

        X_proc = self._preprocess_transform(X, quarters, 0.01)
        return self._model.predict(X_proc)


def save_elastic_net_result(result: ElasticNetResult) -> str:
    """
    Save Elastic Net result to database.

    Returns:
        run_id
    """
    db = get_db_manager()

    with db.get_connection() as conn:
        # Save to ml_model_runs
        config_json = json.dumps(
            {
                "holding_period": result.config.holding_period,
                "quarters": result.config.quarters,
                "train_end_quarter": result.config.train_end_quarter,
                "features": result.config.features,
                "l1_ratios": result.config.l1_ratios,
                "cv_folds": result.config.cv_folds,
            }
        )

        conn.execute(
            """
            INSERT INTO ml_model_runs (
                id, model_type, status, holding_period,
                train_end_quarter, test_start_quarter,
                config_json, train_ic, test_ic,
                n_train_samples, n_test_samples,
                best_alpha, best_l1_ratio, n_features_selected,
                duration_seconds, error_message, completed_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, now())
            """,
            (
                result.run_id,
                "elastic_net",
                result.status,
                result.config.holding_period,
                result.train_quarters[-1] if result.train_quarters else "",
                result.test_quarters[0] if result.test_quarters else "",
                config_json,
                result.train_ic,
                result.test_ic,
                result.n_train_samples,
                result.n_test_samples,
                result.best_alpha,
                result.best_l1_ratio,
                result.n_features_selected,
                result.duration_seconds,
                result.error_message,
            ),
        )

        # Save coefficients
        for coef in result.coefficients:
            conn.execute(
                """
                INSERT INTO ml_model_coefficients (
                    run_id, feature_name, coefficient,
                    coefficient_std, stability_score, feature_importance_rank
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    result.run_id,
                    coef.feature_name,
                    coef.coefficient,
                    coef.coefficient_std,
                    coef.stability_score,
                    coef.importance_rank,
                ),
            )

        # Save IC history
        for ic_point in result.ic_history:
            conn.execute(
                """
                INSERT INTO ml_model_ic_history (
                    run_id, quarter, ic, ic_pvalue, n_samples
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (
                    result.run_id,
                    ic_point.quarter,
                    ic_point.ic,
                    ic_point.ic_pvalue,
                    ic_point.n_samples,
                ),
            )

        # Save predictions (top 200)
        for pred in result.predictions[:200]:
            latest_quarter = result.config.quarters[-1] if result.config.quarters else ""
            conn.execute(
                """
                INSERT INTO ml_model_predictions (
                    run_id, symbol, quarter, predicted_alpha, predicted_rank
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (
                    result.run_id,
                    pred.symbol,
                    latest_quarter,
                    pred.predicted_alpha,
                    pred.predicted_rank,
                ),
            )

    return result.run_id


def load_elastic_net_result(run_id: str) -> dict[str, Any]:
    """Load Elastic Net result from database."""
    db = get_db_manager()

    with db.get_connection() as conn:
        # Load run
        run = conn.execute(
            """
            SELECT * FROM ml_model_runs WHERE id = ?
            """,
            (run_id,),
        ).fetchone()

        if not run:
            raise ValueError(f"Run not found: {run_id}")

        columns = [desc[0] for desc in conn.description]
        run_dict = dict(zip(columns, run))

        # Load coefficients
        coefs = conn.execute(
            """
            SELECT feature_name, coefficient, coefficient_std,
                   stability_score, feature_importance_rank
            FROM ml_model_coefficients
            WHERE run_id = ?
            ORDER BY feature_importance_rank
            """,
            (run_id,),
        ).fetchall()

        coefficients = [
            {
                "feature_name": row[0],
                "coefficient": row[1],
                "coefficient_std": row[2],
                "stability_score": row[3],
                "importance_rank": row[4],
            }
            for row in coefs
        ]

        # Load IC history
        ic_history = conn.execute(
            """
            SELECT quarter, ic, ic_pvalue, n_samples
            FROM ml_model_ic_history
            WHERE run_id = ?
            ORDER BY quarter
            """,
            (run_id,),
        ).fetchall()

        ic_history_list = [
            {
                "quarter": row[0],
                "ic": row[1],
                "ic_pvalue": row[2],
                "n_samples": row[3],
            }
            for row in ic_history
        ]

        # Load predictions
        predictions = conn.execute(
            """
            SELECT symbol, predicted_alpha, predicted_rank
            FROM ml_model_predictions
            WHERE run_id = ?
            ORDER BY predicted_rank
            """,
            (run_id,),
        ).fetchall()

        predictions_list = [
            {
                "symbol": row[0],
                "predicted_alpha": row[1],
                "predicted_rank": row[2],
            }
            for row in predictions
        ]

        return {
            "run": run_dict,
            "coefficients": coefficients,
            "ic_history": ic_history_list,
            "predictions": predictions_list,
        }
