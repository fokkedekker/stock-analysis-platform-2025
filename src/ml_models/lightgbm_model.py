"""
LightGBM model for cross-sectional alpha prediction.

Key features:
1. Optuna hyperparameter tuning
2. Early stopping with validation set
3. Cross-sectional Z-score normalization per quarter
4. Winsorization at 1st/99th percentile
5. Time-series CV (train on past quarters, validate on future)
6. Feature importance tracking (gain and split count)
"""

import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable

import lightgbm as lgb
import numpy as np
import optuna
from scipy import stats

from src.database.connection import get_db_manager
from src.factor_discovery.dataset_builder import DatasetBuilder
from src.ml_models.elastic_net_model import ELASTIC_NET_FEATURES, ICHistoryPoint, StockPrediction

logger = logging.getLogger(__name__)

# Suppress Optuna logging
optuna.logging.set_verbosity(optuna.logging.WARNING)


@dataclass
class LightGBMConfig:
    """Configuration for LightGBM model training."""

    holding_period: int = 4
    quarters: list[str] = field(default_factory=list)
    train_end_quarter: str | None = None
    min_samples_per_quarter: int = 50
    winsorize_percentile: float = 0.01  # 1st/99th
    features: list[str] = field(default_factory=lambda: ELASTIC_NET_FEATURES.copy())
    run_id: str | None = None
    target_type: str = "raw"  # "raw", "beta_adjusted", "sector_adjusted", "full_adjusted"

    # Optuna tuning
    n_optuna_trials: int = 50

    # LightGBM defaults (will be overridden by Optuna)
    num_leaves: int = 31
    learning_rate: float = 0.05
    n_estimators: int = 500
    max_depth: int = -1
    min_child_samples: int = 20


@dataclass
class FeatureImportance:
    """Information about a single feature's importance."""

    feature_name: str
    importance_gain: float  # Sum of gains for this feature across all splits
    importance_split: float  # Number of times this feature is used for splitting
    importance_rank: int


@dataclass
class LightGBMResult:
    """Result of LightGBM training."""

    run_id: str
    config: LightGBMConfig
    status: str
    error_message: str | None
    duration_seconds: float
    # Performance metrics
    train_ic: float | None
    test_ic: float | None
    n_train_samples: int
    n_test_samples: int
    # Best hyperparameters from Optuna
    best_params: dict[str, Any]
    n_features_selected: int
    # Feature importance
    feature_importances: list[FeatureImportance]
    # IC over time
    ic_history: list[ICHistoryPoint]
    # Current predictions (latest quarter)
    predictions: list[StockPrediction]
    # Raw data for storage
    train_quarters: list[str]
    test_quarters: list[str]


class LightGBMModel:
    """
    LightGBM gradient boosting for cross-sectional alpha prediction.

    Methodology:
    1. Load observations from DatasetBuilder
    2. Z-score features cross-sectionally (within each quarter)
    3. Winsorize outliers at 1/99 percentile
    4. Split by time: train (60%), validation (20%), test (20%)
    5. Use Optuna to tune hyperparameters on train+validation
    6. Train final model with early stopping
    7. Extract feature importance (gain and split count)
    8. Generate predictions for latest quarter
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
        self._model: lgb.LGBMRegressor | None = None
        self._feature_names: list[str] = []
        self._feature_means: dict[str, np.ndarray] = {}
        self._feature_stds: dict[str, np.ndarray] = {}
        self._best_params: dict[str, Any] = {}

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

    def train(self, config: LightGBMConfig) -> LightGBMResult:
        """
        Train the LightGBM model.

        Args:
            config: Training configuration

        Returns:
            LightGBMResult with feature importances, performance metrics, and predictions
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

            self._report_progress("init", 5, f"Loaded {len(observations)} observations")

            # Filter to selected holding period
            observations = [
                obs for obs in observations if obs["holding_period"] == config.holding_period
            ]

            logger.info(f"After holding period filter: {len(observations)} observations")

            # 2. Prepare feature matrix
            self._report_progress("prepare", 10, "Preparing features...")
            self._check_cancelled()

            X, y, quarters, symbols = self._prepare_data(observations, config.features)
            self._feature_names = config.features

            logger.info(f"Feature matrix shape: {X.shape}")

            # 3. Split by time: 60% train, 20% validation, 20% test
            self._report_progress("split", 15, "Splitting train/validation/test...")

            if config.train_end_quarter:
                # Use explicit split point
                train_mask = quarters <= config.train_end_quarter
                remaining_quarters = sorted(set(quarters[~train_mask]))
                val_split_idx = len(remaining_quarters) // 2
                val_quarters_set = set(remaining_quarters[:val_split_idx])
                test_quarters_set = set(remaining_quarters[val_split_idx:])
                val_mask = np.array([q in val_quarters_set for q in quarters])
                test_mask = np.array([q in test_quarters_set for q in quarters])
            else:
                # Default: 60/20/20 split
                unique_quarters = sorted(set(quarters))
                n = len(unique_quarters)
                train_idx = int(n * 0.6)
                val_idx = int(n * 0.8)

                train_quarters_set = set(unique_quarters[:train_idx])
                val_quarters_set = set(unique_quarters[train_idx:val_idx])
                test_quarters_set = set(unique_quarters[val_idx:])

                train_mask = np.array([q in train_quarters_set for q in quarters])
                val_mask = np.array([q in val_quarters_set for q in quarters])
                test_mask = np.array([q in test_quarters_set for q in quarters])

            X_train, y_train = X[train_mask], y[train_mask]
            X_val, y_val = X[val_mask], y[val_mask]
            X_test, y_test = X[test_mask], y[test_mask]
            quarters_train = quarters[train_mask]
            quarters_val = quarters[val_mask]
            quarters_test = quarters[test_mask]

            train_quarters_list = sorted(set(quarters_train))
            val_quarters_list = sorted(set(quarters_val))
            test_quarters_list = sorted(set(quarters_test))

            logger.info(f"Train: {len(X_train)} samples, {len(train_quarters_list)} quarters")
            logger.info(f"Val: {len(X_val)} samples, {len(val_quarters_list)} quarters")
            logger.info(f"Test: {len(X_test)} samples, {len(test_quarters_list)} quarters")

            if len(X_train) < 100:
                raise ValueError(f"Not enough training samples: {len(X_train)}")
            if len(X_val) < 50:
                raise ValueError(f"Not enough validation samples: {len(X_val)}")

            # 4. Preprocess (Z-score per quarter, winsorize)
            self._report_progress("preprocess", 20, "Preprocessing features...")
            self._check_cancelled()

            X_train_proc = self._preprocess_fit_transform(
                X_train, quarters_train, config.winsorize_percentile
            )
            X_val_proc = self._preprocess_transform(
                X_val, quarters_val, config.winsorize_percentile
            )
            X_test_proc = self._preprocess_transform(
                X_test, quarters_test, config.winsorize_percentile
            )

            # 5. Optuna hyperparameter tuning
            self._report_progress("optuna", 25, f"Tuning hyperparameters (0/{config.n_optuna_trials} trials)...")
            self._check_cancelled()

            best_params = self._run_optuna(
                X_train_proc, y_train, X_val_proc, y_val, config
            )
            self._best_params = best_params

            # 6. Train final model with best params + early stopping
            self._report_progress("train", 70, "Training final model...")
            self._check_cancelled()

            self._model = lgb.LGBMRegressor(
                **best_params,
                n_estimators=1000,  # High value, early stopping will control
                verbosity=-1,
                force_col_wise=True,
            )

            self._model.fit(
                X_train_proc,
                y_train,
                eval_set=[(X_val_proc, y_val)],
                callbacks=[lgb.early_stopping(50, verbose=False)],
            )

            logger.info(f"Best iteration: {self._model.best_iteration_}")
            logger.info(f"Best params: {best_params}")

            # 7. Extract feature importance
            self._report_progress("importance", 75, "Extracting feature importance...")

            feature_importances = self._extract_feature_importance()
            n_features_selected = len([f for f in feature_importances if f.importance_gain > 0])

            # 8. Calculate IC over time for test quarters
            self._report_progress("ic", 80, "Calculating IC history...")
            self._check_cancelled()

            ic_history = []

            for quarter in test_quarters_list:
                q_mask = quarters == quarter
                if q_mask.sum() < 10:
                    continue

                X_quarter = X[q_mask]
                X_quarter_proc = self._preprocess_transform(
                    X_quarter, np.array([quarter] * len(X_quarter)), config.winsorize_percentile
                )
                y_quarter = y[q_mask]

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

            # 9. Calculate overall IC
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

            # 10. Generate predictions for latest quarter
            self._report_progress("predict", 90, "Generating predictions...")
            self._check_cancelled()

            predictions = self._generate_predictions(
                observations, config, X, quarters, symbols
            )

            self._report_progress("done", 100, "Training complete")

            duration = time.time() - start_time

            return LightGBMResult(
                run_id=run_id,
                config=config,
                status="completed",
                error_message=None,
                duration_seconds=duration,
                train_ic=train_ic,
                test_ic=test_ic,
                n_train_samples=len(X_train),
                n_test_samples=len(X_test),
                best_params=best_params,
                n_features_selected=n_features_selected,
                feature_importances=feature_importances,
                ic_history=ic_history,
                predictions=predictions,
                train_quarters=train_quarters_list + val_quarters_list,
                test_quarters=test_quarters_list,
            )

        except InterruptedError:
            return LightGBMResult(
                run_id=run_id,
                config=config,
                status="cancelled",
                error_message="Training cancelled by user",
                duration_seconds=time.time() - start_time,
                train_ic=None,
                test_ic=None,
                n_train_samples=0,
                n_test_samples=0,
                best_params={},
                n_features_selected=0,
                feature_importances=[],
                ic_history=[],
                predictions=[],
                train_quarters=[],
                test_quarters=[],
            )

        except Exception as e:
            logger.exception(f"Error training LightGBM: {e}")
            return LightGBMResult(
                run_id=run_id,
                config=config,
                status="failed",
                error_message=str(e),
                duration_seconds=time.time() - start_time,
                train_ic=None,
                test_ic=None,
                n_train_samples=0,
                n_test_samples=0,
                best_params={},
                n_features_selected=0,
                feature_importances=[],
                ic_history=[],
                predictions=[],
                train_quarters=[],
                test_quarters=[],
            )

    def _run_optuna(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        config: LightGBMConfig,
    ) -> dict[str, Any]:
        """
        Run Optuna hyperparameter tuning.

        Returns:
            Best hyperparameters dict
        """
        trial_count = [0]  # Mutable counter for progress

        def objective(trial: optuna.Trial) -> float:
            self._check_cancelled()

            params = {
                "num_leaves": trial.suggest_int("num_leaves", 15, 63),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
                "max_depth": trial.suggest_int("max_depth", 3, 12),
                "min_child_samples": trial.suggest_int("min_child_samples", 10, 50),
                "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
                "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
                "bagging_freq": 5,
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            }

            model = lgb.LGBMRegressor(
                **params,
                n_estimators=500,
                verbosity=-1,
                force_col_wise=True,
            )

            model.fit(
                X_train,
                y_train,
                eval_set=[(X_val, y_val)],
                callbacks=[lgb.early_stopping(30, verbose=False)],
            )

            y_pred = model.predict(X_val)
            ic, _ = stats.spearmanr(y_pred, y_val)

            trial_count[0] += 1
            percent = 25 + int((trial_count[0] / config.n_optuna_trials) * 45)
            self._report_progress(
                "optuna",
                percent,
                f"Tuning hyperparameters ({trial_count[0]}/{config.n_optuna_trials} trials, IC={ic:.4f})"
            )

            return ic if not np.isnan(ic) else -1.0

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=config.n_optuna_trials, show_progress_bar=False)

        logger.info(f"Best Optuna IC: {study.best_value:.4f}")
        logger.info(f"Best params: {study.best_params}")

        return study.best_params

    def _prepare_data(
        self, observations: list[dict], features: list[str]
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepare feature matrix and target vector.
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
                X[:, j] = 0

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
        Transform data using fitted parameters (or compute new for unseen quarters).
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

    def _extract_feature_importance(self) -> list[FeatureImportance]:
        """Extract feature importance from trained model."""
        if self._model is None:
            return []

        importance_gain = self._model.feature_importances_
        importance_split = self._model.booster_.feature_importance(importance_type="split")

        # Rank by gain (most important = 1)
        ranks = (-importance_gain).argsort().argsort() + 1

        feature_importances = []
        for i, feature in enumerate(self._feature_names):
            feature_importances.append(
                FeatureImportance(
                    feature_name=feature,
                    importance_gain=float(importance_gain[i]),
                    importance_split=float(importance_split[i]),
                    importance_rank=int(ranks[i]),
                )
            )

        # Sort by rank
        feature_importances.sort(key=lambda x: x.importance_rank)

        return feature_importances

    def _generate_predictions(
        self,
        observations: list[dict],
        config: LightGBMConfig,
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


def save_lightgbm_result(
    result: LightGBMResult,
    progress_callback: Callable[[dict], None] | None = None,
) -> str:
    """
    Save LightGBM result to database.

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
        # Save to ml_model_runs
        config_json = json.dumps(
            {
                "holding_period": result.config.holding_period,
                "quarters": result.config.quarters,
                "train_end_quarter": result.config.train_end_quarter,
                "features": result.config.features,
                "n_optuna_trials": result.config.n_optuna_trials,
                "target_type": result.config.target_type,
                "best_params": result.best_params,
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
                "lightgbm",
                result.status,
                result.config.holding_period,
                result.train_quarters[-1] if result.train_quarters else "",
                result.test_quarters[0] if result.test_quarters else "",
                config_json,
                result.train_ic,
                result.test_ic,
                result.n_train_samples,
                result.n_test_samples,
                None,  # best_alpha not applicable for LightGBM
                None,  # best_l1_ratio not applicable for LightGBM
                result.n_features_selected,
                result.duration_seconds,
                result.error_message,
            ),
        )

        # Save feature importance
        report("Saving feature importance...")
        importance_data = [
            (
                result.run_id,
                fi.feature_name,
                fi.importance_gain,
                fi.importance_split,
                fi.importance_rank,
            )
            for fi in result.feature_importances
        ]
        conn.executemany(
            """
            INSERT INTO ml_model_feature_importance (
                run_id, feature_name, importance_gain,
                importance_split, importance_rank
            ) VALUES (?, ?, ?, ?, ?)
            """,
            importance_data,
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


def load_lightgbm_result(run_id: str) -> dict[str, Any]:
    """Load LightGBM result from database."""
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

        # Load feature importance
        importances = conn.execute(
            """
            SELECT feature_name, importance_gain, importance_split, importance_rank
            FROM ml_model_feature_importance
            WHERE run_id = ?
            ORDER BY importance_rank
            """,
            (run_id,),
        ).fetchall()

        feature_importances = [
            {
                "feature_name": row[0],
                "importance_gain": row[1],
                "importance_split": row[2],
                "importance_rank": row[3],
            }
            for row in importances
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
            "feature_importances": feature_importances,
            "ic_history": ic_history_list,
            "predictions": [],
        }
