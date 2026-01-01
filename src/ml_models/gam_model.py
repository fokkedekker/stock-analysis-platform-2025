"""
GAM (Generalized Additive Model) for cross-sectional alpha prediction.

Key features:
1. Non-linear relationships via spline basis functions
2. Partial dependence plots show effect shape for each feature
3. Identifies "sweet spots" (optimal ranges) for each feature
4. Same preprocessing as Elastic Net (Z-score per quarter, winsorization)
"""

import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
from pygam import LinearGAM, s
from scipy import stats

from src.database.connection import get_db_manager
from src.factor_discovery.dataset_builder import DatasetBuilder
from src.ml_models.elastic_net_model import (
    ELASTIC_NET_FEATURES,
    ICHistoryPoint,
    StockPrediction,
)

logger = logging.getLogger(__name__)


@dataclass
class GAMConfig:
    """Configuration for GAM model training."""

    holding_period: int = 4
    quarters: list[str] = field(default_factory=list)
    train_end_quarter: str | None = None
    min_samples_per_quarter: int = 50
    winsorize_percentile: float = 0.01  # 1st/99th
    n_splines: int = 15  # Number of spline basis functions per feature
    lam: float = 0.6  # Regularization (higher = smoother)
    cv_folds: int = 5
    features: list[str] = field(default_factory=lambda: ELASTIC_NET_FEATURES.copy())
    run_id: str | None = None
    target_type: str = "raw"  # "raw", "beta_adjusted", "sector_adjusted", "full_adjusted"


@dataclass
class PartialDependence:
    """Partial dependence for one feature."""

    feature_name: str
    x_values: list[float]  # 100 points across feature range
    y_values: list[float]  # Effect on alpha at each x
    optimal_min: float | None  # Start of sweet spot
    optimal_max: float | None  # End of sweet spot
    peak_x: float | None  # X value with max effect
    peak_y: float  # Max effect value
    importance_rank: int


@dataclass
class GAMResult:
    """Result of GAM training."""

    run_id: str
    config: GAMConfig
    status: str
    error_message: str | None
    duration_seconds: float
    # Performance metrics
    train_ic: float | None
    test_ic: float | None
    train_r2: float | None  # Pseudo R² from GAM
    n_train_samples: int
    n_test_samples: int
    # Model info
    n_features: int
    best_lam: float | None
    # Partial dependences
    partial_dependences: list[PartialDependence]
    # IC over time
    ic_history: list[ICHistoryPoint]
    # Current predictions (latest quarter)
    predictions: list[StockPrediction]
    # Raw data for storage
    train_quarters: list[str]
    test_quarters: list[str]


class GAMModel:
    """
    Generalized Additive Model for cross-sectional alpha prediction.

    Methodology:
    1. Load observations from DatasetBuilder
    2. Z-score features cross-sectionally (within each quarter)
    3. Winsorize outliers at 1/99 percentile
    4. Split by time: train on earlier quarters, test on later
    5. Fit GAM with spline terms for each feature
    6. Extract partial dependence curves
    7. Find optimal ranges (where effect is >= 80% of max)
    8. Generate predictions for latest quarter
    """

    def __init__(
        self,
        progress_callback: Callable[[dict], None] | None = None,
        cancel_check: Callable[[], bool] | None = None,
    ):
        self.progress_callback = progress_callback
        self.cancel_check = cancel_check
        self._model: LinearGAM | None = None
        self._feature_names: list[str] = []
        self._feature_means: dict[str, np.ndarray] = {}
        self._feature_stds: dict[str, np.ndarray] = {}

    def _report_progress(self, stage: str, percent: int, message: str):
        if self.progress_callback:
            self.progress_callback(
                {"stage": stage, "percent": percent, "message": message}
            )

    def _check_cancelled(self):
        if self.cancel_check and self.cancel_check():
            raise InterruptedError("Training cancelled by user")

    def train(self, config: GAMConfig) -> GAMResult:
        """
        Train the GAM model.

        Args:
            config: Training configuration

        Returns:
            GAMResult with partial dependences, performance metrics, and predictions
        """
        run_id = config.run_id if config.run_id else str(uuid.uuid4())[:8]
        start_time = time.time()

        try:
            self._report_progress("init", 0, "Loading dataset...")

            # 1. Build dataset using DatasetBuilder
            builder = DatasetBuilder(
                quarters=config.quarters,
                holding_periods=[config.holding_period],
                data_lag_quarters=1,
                target_type=config.target_type,
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

            logger.info(f"Train: {len(X_train)} samples, {len(train_quarters_list)} quarters")
            logger.info(f"Test: {len(X_test)} samples, {len(test_quarters_list)} quarters")

            if len(X_train) < 100:
                raise ValueError(f"Not enough training samples: {len(X_train)}")

            # 4. Preprocess (Z-score per quarter, winsorize)
            self._report_progress("preprocess", 25, "Preprocessing features...")
            self._check_cancelled()

            X_train_proc = self._preprocess_fit_transform(
                X_train, quarters_train, config.winsorize_percentile
            )
            X_test_proc = self._preprocess_transform(
                X_test, quarters_test, config.winsorize_percentile
            )

            # 5. Build and fit GAM
            self._report_progress("fitting", 30, "Building GAM model...")
            self._check_cancelled()

            # Build GAM with spline terms for each feature
            n_features = len(config.features)

            # Start with first term
            gam = s(0, n_splines=config.n_splines, lam=config.lam)

            # Add remaining terms
            for i in range(1, n_features):
                gam = gam + s(i, n_splines=config.n_splines, lam=config.lam)

            self._model = LinearGAM(gam)

            self._report_progress("fitting", 40, "Fitting GAM model...")
            self._model.fit(X_train_proc, y_train)

            logger.info("GAM model fitted successfully")

            # Get R² (compute manually for reliability)
            train_r2 = None
            try:
                from sklearn.metrics import r2_score
                y_pred_train = self._model.predict(X_train_proc)
                train_r2 = float(r2_score(y_train, y_pred_train))
                logger.info(f"Train R²: {train_r2:.4f}")
            except Exception as e:
                logger.warning(f"Could not compute R²: {e}")

            # 6. Compute partial dependence
            self._report_progress("partial_dep", 50, "Computing partial dependence...")
            self._check_cancelled()

            partial_dependences = self._compute_partial_dependence(X_train_proc, config)

            # 7. Calculate IC over time for test quarters
            self._report_progress("ic", 70, "Calculating IC history...")
            self._check_cancelled()

            ic_history = []

            # Calculate IC for test quarters only
            for quarter in test_quarters_list:
                q_mask = quarters == quarter
                if q_mask.sum() < 10:
                    continue

                X_quarter = X[q_mask]
                X_quarter_proc = self._preprocess_transform(
                    X_quarter, np.array([quarter] * len(X_quarter)), config.winsorize_percentile
                )
                y_quarter = y[q_mask]

                # Generate predictions and calculate IC
                y_pred = self._model.predict(X_quarter_proc)
                ic, pvalue = stats.spearmanr(y_pred, y_quarter)
                if not np.isnan(ic):
                    ic_history.append(
                        ICHistoryPoint(
                            quarter=quarter,
                            ic=float(ic),
                            ic_pvalue=float(pvalue),
                            n_samples=int(q_mask.sum()),
                        )
                    )

            # 8. Calculate overall IC
            self._report_progress("ic", 80, "Calculating overall IC...")

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
            self._report_progress("predict", 90, "Generating predictions...")
            self._check_cancelled()

            predictions = self._generate_predictions(
                observations, config, X, quarters, symbols
            )

            self._report_progress("done", 100, "Training complete")

            duration = time.time() - start_time

            return GAMResult(
                run_id=run_id,
                config=config,
                status="completed",
                error_message=None,
                duration_seconds=duration,
                train_ic=train_ic,
                test_ic=test_ic,
                train_r2=train_r2,
                n_train_samples=len(X_train),
                n_test_samples=len(X_test),
                n_features=n_features,
                best_lam=config.lam,
                partial_dependences=partial_dependences,
                ic_history=ic_history,
                predictions=predictions,
                train_quarters=train_quarters_list,
                test_quarters=test_quarters_list,
            )

        except InterruptedError:
            return GAMResult(
                run_id=run_id,
                config=config,
                status="cancelled",
                error_message="Training cancelled by user",
                duration_seconds=time.time() - start_time,
                train_ic=None,
                test_ic=None,
                train_r2=None,
                n_train_samples=0,
                n_test_samples=0,
                n_features=0,
                best_lam=None,
                partial_dependences=[],
                ic_history=[],
                predictions=[],
                train_quarters=[],
                test_quarters=[],
            )

        except Exception as e:
            logger.exception(f"Error training GAM: {e}")
            return GAMResult(
                run_id=run_id,
                config=config,
                status="failed",
                error_message=str(e),
                duration_seconds=time.time() - start_time,
                train_ic=None,
                test_ic=None,
                train_r2=None,
                n_train_samples=0,
                n_test_samples=0,
                n_features=0,
                best_lam=None,
                partial_dependences=[],
                ic_history=[],
                predictions=[],
                train_quarters=[],
                test_quarters=[],
            )

    def _prepare_data(
        self, observations: list[dict], features: list[str]
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Prepare feature matrix and target vector."""
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
                X[:, j] = 0

        return X, y, np.array(quarters), np.array(symbols)

    def _preprocess_fit_transform(
        self, X: np.ndarray, quarters: np.ndarray, winsorize_pct: float
    ) -> np.ndarray:
        """Fit preprocessing parameters and transform training data."""
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
        """Transform test data using fitted parameters."""
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

            # Use stored parameters if available
            if quarter in self._feature_means:
                means = self._feature_means[quarter]
                stds = self._feature_stds[quarter]
            else:
                means = X_q.mean(axis=0)
                stds = X_q.std(axis=0) + 1e-8

            X_q = (X_q - means) / stds
            X_proc[mask] = X_q

        return X_proc

    def _compute_partial_dependence(
        self, X: np.ndarray, config: GAMConfig
    ) -> list[PartialDependence]:
        """
        Compute partial dependence for each feature.

        Returns list of PartialDependence objects with effect curves and optimal ranges.
        """
        if self._model is None:
            return []

        partial_deps = []
        importance_scores = []

        for i, feature_name in enumerate(self._feature_names):
            try:
                # Generate grid of x values
                x_min, x_max = X[:, i].min(), X[:, i].max()
                x_grid = np.linspace(x_min, x_max, 100)

                # Get partial dependence using pyGAM's built-in method
                # Create a matrix for prediction with the grid values for this feature
                # and mean values for other features
                X_pd = np.tile(X.mean(axis=0), (100, 1))
                X_pd[:, i] = x_grid

                # Get the partial effect for this term
                y_pd = self._model.partial_dependence(term=i, X=X_pd)

                # Find optimal range (where effect >= 80% of max)
                optimal_min, optimal_max, peak_x = self._find_optimal_range(
                    x_grid, y_pd, threshold_pct=0.8
                )

                peak_y = float(y_pd.max())
                importance = abs(peak_y)  # Use max effect as importance
                importance_scores.append((i, importance))

                partial_deps.append(
                    PartialDependence(
                        feature_name=feature_name,
                        x_values=x_grid.tolist(),
                        y_values=y_pd.tolist(),
                        optimal_min=optimal_min,
                        optimal_max=optimal_max,
                        peak_x=peak_x,
                        peak_y=peak_y,
                        importance_rank=0,  # Will be set after sorting
                    )
                )

            except Exception as e:
                logger.warning(f"Could not compute partial dependence for {feature_name}: {e}")
                continue

        # Rank by importance (max effect magnitude)
        importance_scores.sort(key=lambda x: x[1], reverse=True)
        rank_map = {idx: rank + 1 for rank, (idx, _) in enumerate(importance_scores)}

        for pd in partial_deps:
            feature_idx = self._feature_names.index(pd.feature_name)
            pd.importance_rank = rank_map.get(feature_idx, len(self._feature_names))

        # Sort by importance rank
        partial_deps.sort(key=lambda x: x.importance_rank)

        return partial_deps

    def _find_optimal_range(
        self, x_grid: np.ndarray, y_pd: np.ndarray, threshold_pct: float = 0.8
    ) -> tuple[float | None, float | None, float | None]:
        """
        Find the range of x values where effect is >= threshold_pct of max.

        Returns:
            (optimal_min, optimal_max, peak_x)
        """
        if len(y_pd) == 0:
            return None, None, None

        max_effect = y_pd.max()
        min_effect = y_pd.min()
        effect_range = max_effect - min_effect

        if effect_range < 1e-6:
            # No meaningful effect
            return None, None, None

        # Find where effect is above threshold
        threshold = min_effect + (effect_range * threshold_pct)
        above_threshold = x_grid[y_pd >= threshold]

        peak_idx = np.argmax(y_pd)
        peak_x = float(x_grid[peak_idx])

        if len(above_threshold) > 0:
            return float(above_threshold.min()), float(above_threshold.max()), peak_x
        else:
            return None, None, peak_x

    def _generate_predictions(
        self,
        observations: list[dict],
        config: GAMConfig,
        X: np.ndarray,
        quarters: np.ndarray,
        symbols: np.ndarray,
    ) -> list[StockPrediction]:
        """Generate predictions for the latest quarter."""
        if self._model is None:
            return []

        latest_quarter = max(set(quarters))
        latest_mask = quarters == latest_quarter

        if latest_mask.sum() == 0:
            return []

        X_latest = X[latest_mask]
        symbols_latest = symbols[latest_mask]

        X_latest_proc = self._preprocess_transform(
            X_latest, np.array([latest_quarter] * len(X_latest)), config.winsorize_percentile
        )

        y_pred = self._model.predict(X_latest_proc)
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

        predictions.sort(key=lambda x: x.predicted_rank)
        return predictions

    def predict(self, X: np.ndarray, quarters: np.ndarray) -> np.ndarray:
        """Predict alpha for new data using fitted model."""
        if self._model is None:
            raise ValueError("Model not trained")

        X_proc = self._preprocess_transform(X, quarters, 0.01)
        return self._model.predict(X_proc)


def save_gam_result(
    result: GAMResult,
    progress_callback: Callable[[dict], None] | None = None,
) -> str:
    """
    Save GAM result to database.

    Args:
        result: The training result to save
        progress_callback: Optional callback for progress updates

    Returns:
        run_id
    """
    db = get_db_manager()

    def report(message: str):
        if progress_callback:
            progress_callback({"stage": "saving", "percent": 100, "message": message})

    report("Saving model configuration...")

    with db.get_connection() as conn:
        # Save to ml_model_runs (reuse table with model_type='gam')
        config_json = json.dumps(
            {
                "holding_period": result.config.holding_period,
                "quarters": result.config.quarters,
                "train_end_quarter": result.config.train_end_quarter,
                "features": result.config.features,
                "n_splines": result.config.n_splines,
                "lam": result.config.lam,
                "cv_folds": result.config.cv_folds,
                "target_type": result.config.target_type,
            }
        )

        conn.execute(
            """
            INSERT INTO ml_model_runs (
                id, model_type, status, holding_period,
                train_end_quarter, test_start_quarter,
                config_json, train_ic, test_ic, train_r2,
                n_train_samples, n_test_samples,
                best_alpha, best_l1_ratio, n_features_selected,
                duration_seconds, error_message, completed_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, now())
            """,
            (
                result.run_id,
                "gam",
                result.status,
                result.config.holding_period,
                result.train_quarters[-1] if result.train_quarters else "",
                result.test_quarters[0] if result.test_quarters else "",
                config_json,
                result.train_ic,
                result.test_ic,
                result.train_r2,
                result.n_train_samples,
                result.n_test_samples,
                result.best_lam,  # Store lam in best_alpha column
                None,  # No l1_ratio for GAM
                result.n_features,
                result.duration_seconds,
                result.error_message,
            ),
        )

        # Save partial dependences
        report("Saving partial dependences...")
        for pd in result.partial_dependences:
            conn.execute(
                """
                INSERT INTO ml_model_partial_dependence (
                    run_id, feature_name, x_values, y_values,
                    optimal_min, optimal_max, peak_x, peak_y,
                    importance_rank
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    result.run_id,
                    pd.feature_name,
                    json.dumps(pd.x_values),
                    json.dumps(pd.y_values),
                    pd.optimal_min,
                    pd.optimal_max,
                    pd.peak_x,
                    pd.peak_y,
                    pd.importance_rank,
                ),
            )

        # Save IC history
        report("Saving IC history...")
        ic_data = [
            (
                result.run_id,
                ic_point.quarter,
                ic_point.ic,
                ic_point.ic_pvalue,
                ic_point.n_samples,
            )
            for ic_point in result.ic_history
        ]
        conn.executemany(
            """
            INSERT INTO ml_model_ic_history (
                run_id, quarter, ic, ic_pvalue, n_samples
            ) VALUES (?, ?, ?, ?, ?)
            """,
            ic_data,
        )

    return result.run_id


def load_gam_result(run_id: str) -> dict[str, Any]:
    """Load GAM result from database."""
    db = get_db_manager()

    with db.get_connection() as conn:
        # Load run
        run = conn.execute(
            "SELECT * FROM ml_model_runs WHERE id = ?",
            (run_id,),
        ).fetchone()

        if not run:
            raise ValueError(f"Run not found: {run_id}")

        columns = [desc[0] for desc in conn.description]
        run_dict = dict(zip(columns, run))

        # Load partial dependences
        pds = conn.execute(
            """
            SELECT feature_name, x_values, y_values,
                   optimal_min, optimal_max, peak_x, peak_y, importance_rank
            FROM ml_model_partial_dependence
            WHERE run_id = ?
            ORDER BY importance_rank
            """,
            (run_id,),
        ).fetchall()

        partial_dependences = [
            {
                "feature_name": row[0],
                "x_values": json.loads(row[1]),
                "y_values": json.loads(row[2]),
                "optimal_min": row[3],
                "optimal_max": row[4],
                "peak_x": row[5],
                "peak_y": row[6],
                "importance_rank": row[7],
            }
            for row in pds
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

        return {
            "run": run_dict,
            "partial_dependences": partial_dependences,
            "ic_history": ic_history_list,
            "predictions": [],  # Predictions are now calculated live
        }
