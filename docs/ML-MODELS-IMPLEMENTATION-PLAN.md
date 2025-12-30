# ML Models Implementation Plan
## A Quant's Guide to Adding Cross-Sectional Return Prediction

**Author:** Claude (Senior Quant Persona)
**Date:** December 2024
**Status:** Planning

---

## Executive Summary

This document outlines a rigorous, phased approach to adding machine learning models for cross-sectional stock return prediction. The plan prioritizes robustness over complexity, with explicit safeguards against overfitting.

**Key Principles:**
1. Simpler models first (Elastic Net before XGBoost)
2. Factor stability before factor magnitude
3. Regime features, not regime-specific models (data too small)
4. Out-of-sample validation at every step

---

## Table of Contents

1. [Current State Analysis](#1-current-state-analysis)
2. [Priority 1: Factor Decay Modeling](#2-priority-1-factor-decay-modeling)
3. [Priority 2: Regime Data Infrastructure](#3-priority-2-regime-data-infrastructure)
4. [Priority 3: Elastic Net Regression](#4-priority-3-elastic-net-regression)
5. [Priority 4: GAM (Generalized Additive Model)](#5-priority-4-gam)
6. [Priority 5: Gradient Boosted Trees](#6-priority-5-gradient-boosted-trees)
7. [Frontend Architecture](#7-frontend-architecture)
8. [Database Schema Changes](#8-database-schema-changes)
9. [API Endpoints](#9-api-endpoints)
10. [Risk Mitigation](#10-risk-mitigation)
11. [Success Metrics](#11-success-metrics)
12. [Implementation Timeline](#12-implementation-timeline)

---

## 1. Current State Analysis

### What We Have (Factor Screening Model)

The current Factor Discovery system is a **threshold-based filter optimizer**:

```
Input: 55 factors × multiple thresholds × 4 holding periods
Process: Test each (factor, threshold) → measure alpha
Output: Optimal filter rules (e.g., "PE < 15 AND ROIC > 12%")
```

**Strengths:**
- Interpretable rules
- No black-box models
- Direct mapping to Pipeline UI

**Weaknesses:**
- Binary pass/fail (loses information)
- No conditional effects (PE works differently at different ROIC levels)
- Assumes linear thresholds (misses sweet spots)
- No factor decay tracking

### Data Availability

| Metric | Current Value | Notes |
|--------|---------------|-------|
| Quarters of data | ~50 | 2012-2024 |
| Stocks per quarter | ~500-800 | After filtering |
| Total observations | ~25,000-40,000 | Varies by holding period |
| Features | 55 | Numerical + categorical |

**Data Constraint:** This is **small** for ML. Most academic studies use 100+ years of data. We must use strong regularization and simple models.

### Dependencies Available (pyproject.toml)

Currently installed:
- `numpy`
- `scipy` (stats only)

Need to add:
- `scikit-learn` (Elastic Net, preprocessing)
- `pygam` (GAM)
- `lightgbm` or `xgboost` (trees, last priority)

---

## 2. Priority 1: Factor Decay Modeling

### Why This First?

Factor decay analysis improves the existing system without adding model complexity. It answers: **"Is this factor reliable over time, or did it just work in one period?"**

A factor with 3% alpha and 90% stability beats a factor with 6% alpha and 40% stability.

### Implementation

#### 2.1 Rolling Window Statistics

For each factor and threshold, compute over rolling 5-year windows:
- Rolling alpha
- Rolling t-statistic
- Rolling Information Coefficient (IC = Spearman correlation with future returns)

```python
# New file: src/factor_discovery/decay_analyzer.py

class DecayAnalyzer:
    """Analyzes factor stability over time."""

    def __init__(self, window_quarters: int = 20):  # 5 years
        self.window_quarters = window_quarters

    def compute_rolling_stats(
        self,
        factor_values: np.ndarray,
        alphas: np.ndarray,
        quarters: np.ndarray,
        threshold: float,
        operator: str
    ) -> RollingStats:
        """Compute rolling alpha, IC, and significance."""

        unique_quarters = sorted(set(quarters))
        n_windows = len(unique_quarters) - self.window_quarters + 1

        rolling_alphas = []
        rolling_ics = []
        rolling_pvalues = []

        for i in range(n_windows):
            window_quarters = unique_quarters[i:i + self.window_quarters]
            mask = np.isin(quarters, window_quarters)

            # Apply threshold filter
            if operator == ">=":
                passes = factor_values[mask] >= threshold
            else:
                passes = factor_values[mask] <= threshold

            window_alphas = alphas[mask]

            # Alpha for stocks passing filter
            if passes.sum() >= 30:  # Minimum sample
                alpha = window_alphas[passes].mean()
                rolling_alphas.append(alpha)

                # IC (correlation between factor and alpha)
                ic, pval = spearmanr(factor_values[mask], window_alphas)
                rolling_ics.append(ic)
                rolling_pvalues.append(pval)

        return RollingStats(
            alphas=rolling_alphas,
            ics=rolling_ics,
            pvalues=rolling_pvalues
        )
```

#### 2.2 Decay Score Calculation

```python
def compute_decay_score(rolling_stats: RollingStats) -> DecayMetrics:
    """
    Decay Score = % of rolling windows with positive alpha

    Interpretation:
    - > 80%: Very stable factor
    - 60-80%: Moderately stable
    - 40-60%: Unstable, use with caution
    - < 40%: Dead factor, don't use
    """
    n_windows = len(rolling_stats.alphas)

    # Positive alpha windows
    positive_windows = sum(1 for a in rolling_stats.alphas if a > 0)
    decay_score = positive_windows / n_windows if n_windows > 0 else 0

    # Significant IC windows (p < 0.05)
    sig_windows = sum(1 for p in rolling_stats.pvalues if p < 0.05)
    ic_stability = sig_windows / n_windows if n_windows > 0 else 0

    # Trend detection (is alpha declining?)
    if len(rolling_stats.alphas) >= 4:
        recent_alpha = np.mean(rolling_stats.alphas[-4:])
        early_alpha = np.mean(rolling_stats.alphas[:4])
        alpha_trend = (recent_alpha - early_alpha) / abs(early_alpha) if early_alpha != 0 else 0
    else:
        alpha_trend = 0

    return DecayMetrics(
        decay_score=decay_score,
        ic_stability=ic_stability,
        alpha_trend=alpha_trend,
        n_windows=n_windows,
        recent_alpha=rolling_stats.alphas[-1] if rolling_stats.alphas else None,
        mean_ic=np.mean(rolling_stats.ics) if rolling_stats.ics else None
    )
```

#### 2.3 Integration with Factor Discovery

Modify `FactorResult` model to include decay metrics:

```python
# Add to src/factor_discovery/models.py

class DecayMetrics(BaseModel):
    """Factor stability metrics over time."""
    decay_score: float = Field(..., description="% of windows with positive alpha (0-1)")
    ic_stability: float = Field(..., description="% of windows with significant IC (0-1)")
    alpha_trend: float = Field(..., description="Change in alpha over time (-1 to +1)")
    n_windows: int = Field(..., description="Number of rolling windows analyzed")
    recent_alpha: float | None = Field(None, description="Alpha in most recent window")
    mean_ic: float | None = Field(None, description="Average information coefficient")


class FactorResult(BaseModel):
    # ... existing fields ...

    # NEW: Decay metrics
    decay_metrics: DecayMetrics | None = Field(
        default=None,
        description="Factor stability over rolling windows"
    )
```

#### 2.4 UI Display

Add decay score column to factor results table:
- Color code: Green (>80%), Yellow (60-80%), Orange (40-60%), Red (<40%)
- Show trend arrow (up/down/flat)
- Filter option: "Show only stable factors (decay > 70%)"

#### 2.5 Use in Combination Selection

Penalize unstable factors in combination scoring:

```python
def adjusted_alpha(raw_alpha: float, decay_score: float) -> float:
    """
    Reduce alpha for unstable factors.
    A factor with 50% decay score gets its alpha halved.
    """
    stability_multiplier = decay_score  # 0 to 1
    return raw_alpha * stability_multiplier
```

### Files to Modify

| File | Changes |
|------|---------|
| `src/factor_discovery/models.py` | Add `DecayMetrics` model |
| `src/factor_discovery/factor_analyzer.py` | Compute decay for each factor |
| `src/factor_discovery/combination_finder.py` | Use adjusted alpha |
| `frontend/src/app/factor-discovery/page.tsx` | Display decay scores |

---

## 3. Priority 2: Regime Data Infrastructure

### Available Data Sources (FMP API)

FMP provides economic data through these endpoints:

| Endpoint | Data | Use For |
|----------|------|---------|
| `/stable/treasury-rates` | 1M, 3M, 6M, 1Y, 2Y, 5Y, 10Y, 30Y yields | Interest rate regime |
| `/stable/economic-indicators?name=CPI` | Consumer Price Index | Inflation regime |
| `/stable/economic-indicators?name=GDP` | GDP growth | Economic growth regime |
| `/stable/economic-indicators?name=unemploymentRate` | Unemployment | Labor market regime |
| `/stable/batch-index-quotes` | S&P 500, indices | Market returns |
| `/stable/market-risk-premium` | Market risk premium | Risk appetite |

### Implementation

#### 3.1 New Database Table

```sql
-- Add to src/database/schema.py

CREATE TABLE IF NOT EXISTS macro_indicators (
    date DATE PRIMARY KEY,
    quarter VARCHAR(10),  -- e.g., "2024Q1"

    -- Interest Rates
    treasury_3m DECIMAL(8,4),
    treasury_2y DECIMAL(8,4),
    treasury_10y DECIMAL(8,4),
    yield_curve_spread DECIMAL(8,4),  -- 10Y - 2Y

    -- Rate changes (QoQ)
    rate_change_qoq DECIMAL(8,4),  -- 10Y change
    rate_rising BOOLEAN,

    -- Inflation
    cpi_yoy DECIMAL(8,4),  -- CPI year-over-year %
    inflation_high BOOLEAN,  -- CPI > 3%

    -- Market
    sp500_return_1q DECIMAL(8,4),  -- S&P 500 quarterly return
    sp500_volatility_20d DECIMAL(8,4),  -- Rolling 20-day vol (annualized)
    volatility_high BOOLEAN,  -- Vol > 20%

    -- Derived Regime Flags
    regime_rates VARCHAR(20),  -- "rising", "falling", "stable"
    regime_inflation VARCHAR(20),  -- "high", "moderate", "low"
    regime_volatility VARCHAR(20),  -- "high", "normal", "low"

    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_macro_quarter ON macro_indicators(quarter);
```

#### 3.2 FMP Client Extension

```python
# Add to src/scrapers/fmp_client.py

class FMPClient:
    # ... existing code ...

    async def get_treasury_rates(
        self,
        from_date: str | None = None,
        to_date: str | None = None
    ) -> list[dict]:
        """Fetch historical Treasury rates."""
        params = {}
        if from_date:
            params["from"] = from_date
        if to_date:
            params["to"] = to_date

        return await self._get("/stable/treasury-rates", params=params)

    async def get_economic_indicator(
        self,
        name: str,  # "CPI", "GDP", "unemploymentRate"
        from_date: str | None = None,
        to_date: str | None = None
    ) -> list[dict]:
        """Fetch historical economic indicator."""
        params = {"name": name}
        if from_date:
            params["from"] = from_date
        if to_date:
            params["to"] = to_date

        return await self._get("/stable/economic-indicators", params=params)

    async def get_index_historical(
        self,
        symbol: str = "^GSPC",  # S&P 500
        from_date: str | None = None,
        to_date: str | None = None
    ) -> list[dict]:
        """Fetch historical index prices."""
        endpoint = f"/stable/historical-price-eod/full?symbol={symbol}"
        params = {}
        if from_date:
            params["from"] = from_date
        if to_date:
            params["to"] = to_date

        return await self._get(endpoint, params=params)
```

#### 3.3 Macro Data Fetcher

```python
# New file: src/scrapers/macro_fetcher.py

class MacroFetcher:
    """Fetches and processes macro economic indicators."""

    def __init__(self, fmp_client: FMPClient, db: DuckDBConnection):
        self.fmp = fmp_client
        self.db = db

    async def fetch_and_store(self, from_date: str = "2010-01-01"):
        """Fetch all macro data and compute regime flags."""

        # 1. Fetch raw data
        treasury = await self.fmp.get_treasury_rates(from_date=from_date)
        cpi = await self.fmp.get_economic_indicator("CPI", from_date=from_date)
        sp500 = await self.fmp.get_index_historical("^GSPC", from_date=from_date)

        # 2. Process into quarterly observations
        quarterly_data = self._aggregate_to_quarters(treasury, cpi, sp500)

        # 3. Compute regime flags
        for q in quarterly_data:
            q["regime_rates"] = self._classify_rate_regime(q)
            q["regime_inflation"] = self._classify_inflation_regime(q)
            q["regime_volatility"] = self._classify_vol_regime(q)

        # 4. Store in database
        self._upsert_macro_data(quarterly_data)

    def _classify_rate_regime(self, q: dict) -> str:
        """Classify interest rate regime."""
        change = q.get("rate_change_qoq", 0)
        if change > 0.25:  # +25bps
            return "rising"
        elif change < -0.25:
            return "falling"
        return "stable"

    def _classify_inflation_regime(self, q: dict) -> str:
        """Classify inflation regime."""
        cpi = q.get("cpi_yoy", 2.0)
        if cpi > 4.0:
            return "high"
        elif cpi < 2.0:
            return "low"
        return "moderate"

    def _classify_vol_regime(self, q: dict) -> str:
        """Classify volatility regime."""
        vol = q.get("sp500_volatility_20d", 15)
        if vol > 25:
            return "high"
        elif vol < 12:
            return "low"
        return "normal"
```

#### 3.4 Script to Load Historical Data

```python
# New file: scripts/load_macro_data.py

async def main():
    """Load historical macro data from FMP."""
    fmp = FMPClient()
    db = get_db_connection()

    fetcher = MacroFetcher(fmp, db)
    await fetcher.fetch_and_store(from_date="2010-01-01")

    print("Macro data loaded successfully")

    # Print summary
    result = db.execute("""
        SELECT
            regime_rates,
            regime_inflation,
            regime_volatility,
            COUNT(*) as quarters
        FROM macro_indicators
        GROUP BY 1, 2, 3
        ORDER BY quarters DESC
    """).fetchall()

    print("\nRegime distribution:")
    for row in result:
        print(f"  {row}")
```

#### 3.5 Integration with Dataset Builder

Add regime features to training data:

```python
# Modify src/factor_discovery/dataset_builder.py

def build_dataset(self, quarters: list[str], ...) -> pd.DataFrame:
    # ... existing code ...

    # Join macro indicators
    macro_df = self._load_macro_data(quarters)
    df = df.merge(macro_df, on="quarter", how="left")

    # Add regime features
    df["regime_rates_rising"] = (df["regime_rates"] == "rising").astype(int)
    df["regime_rates_falling"] = (df["regime_rates"] == "falling").astype(int)
    df["regime_inflation_high"] = (df["regime_inflation"] == "high").astype(int)
    df["regime_vol_high"] = (df["regime_volatility"] == "high").astype(int)

    return df
```

### Important: Regime as Features, Not Model Splits

**DO NOT** train separate models per regime. With 50 quarters:
- High inflation quarters: ~8-10
- Rising rate quarters: ~12-15
- High vol quarters: ~10

This is **not enough data** for separate models.

**Instead:** Add regime as features. Let the ML model learn interactions like "Value works better when rates are rising."

---

## 4. Priority 3: Elastic Net Regression

### Why Elastic Net?

- **Linear:** Interpretable coefficients
- **Regularized:** L1 (sparsity) + L2 (stability) penalties
- **Robust:** Handles multicollinearity in financial features
- **Baseline:** Sanity check before complex models

### Implementation

#### 4.1 Model Architecture

```python
# New file: src/ml_models/elastic_net_model.py

from sklearn.linear_model import ElasticNetCV
from sklearn.preprocessing import StandardScaler
import numpy as np

class ElasticNetAlphaModel:
    """
    Cross-sectional alpha prediction using Elastic Net.

    Key design decisions:
    - Z-score features per quarter (cross-sectional, not time-series)
    - Winsorize outliers at 1st/99th percentile
    - Time-series CV (no future leakage)
    - Output: predicted alpha rank, not raw prediction
    """

    def __init__(
        self,
        l1_ratio: float = 0.5,  # Balance between L1 and L2
        cv_folds: int = 5,
        min_train_quarters: int = 20
    ):
        self.l1_ratio = l1_ratio
        self.cv_folds = cv_folds
        self.min_train_quarters = min_train_quarters
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = None

    def preprocess(self, X: np.ndarray, quarters: np.ndarray) -> np.ndarray:
        """
        Cross-sectional preprocessing per quarter.

        1. Winsorize at 1/99 percentile
        2. Z-score within each quarter
        """
        X_processed = X.copy()

        for q in np.unique(quarters):
            mask = quarters == q
            X_q = X_processed[mask]

            # Winsorize
            for col in range(X_q.shape[1]):
                p1, p99 = np.percentile(X_q[:, col], [1, 99])
                X_q[:, col] = np.clip(X_q[:, col], p1, p99)

            # Z-score
            X_q = (X_q - X_q.mean(axis=0)) / (X_q.std(axis=0) + 1e-8)
            X_processed[mask] = X_q

        return X_processed

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        quarters: np.ndarray,
        feature_names: list[str]
    ) -> dict:
        """
        Train with time-series cross-validation.

        Returns:
            Training diagnostics including coefficient stability
        """
        self.feature_names = feature_names

        # Preprocess
        X_proc = self.preprocess(X, quarters)

        # Time-series CV: train on past, validate on future
        unique_quarters = sorted(np.unique(quarters))

        cv_results = []
        coef_history = []

        for i in range(self.min_train_quarters, len(unique_quarters) - 1):
            train_quarters = unique_quarters[:i]
            val_quarter = unique_quarters[i]

            train_mask = np.isin(quarters, train_quarters)
            val_mask = quarters == val_quarter

            # Fit on training data
            model = ElasticNetCV(
                l1_ratio=self.l1_ratio,
                cv=self.cv_folds,
                max_iter=10000
            )
            model.fit(X_proc[train_mask], y[train_mask])

            # Predict on validation
            y_pred = model.predict(X_proc[val_mask])

            # Compute IC (rank correlation)
            ic, _ = spearmanr(y_pred, y[val_mask])
            cv_results.append({
                "quarter": val_quarter,
                "ic": ic,
                "n_samples": val_mask.sum()
            })

            coef_history.append(model.coef_.copy())

        # Final model on all data
        self.model = ElasticNetCV(
            l1_ratio=self.l1_ratio,
            cv=self.cv_folds,
            max_iter=10000
        )
        self.model.fit(X_proc, y)

        # Coefficient stability analysis
        coef_stability = self._analyze_coefficient_stability(coef_history)

        return {
            "cv_results": cv_results,
            "mean_ic": np.mean([r["ic"] for r in cv_results]),
            "ic_std": np.std([r["ic"] for r in cv_results]),
            "coefficients": dict(zip(feature_names, self.model.coef_)),
            "coefficient_stability": coef_stability,
            "alpha": self.model.alpha_,
            "n_features_selected": (np.abs(self.model.coef_) > 1e-6).sum()
        }

    def _analyze_coefficient_stability(
        self,
        coef_history: list[np.ndarray]
    ) -> dict[str, float]:
        """
        Analyze how stable each coefficient is over time.

        Returns:
            coefficient_stability[feature_name] = % of windows with same sign
        """
        coef_array = np.array(coef_history)  # (n_windows, n_features)

        stability = {}
        for i, name in enumerate(self.feature_names):
            coefs = coef_array[:, i]
            # What % of time does coefficient have same sign as final?
            final_sign = np.sign(coefs[-1])
            same_sign = (np.sign(coefs) == final_sign).mean()
            stability[name] = same_sign

        return stability

    def predict(self, X: np.ndarray, quarters: np.ndarray) -> np.ndarray:
        """Predict alpha for new data."""
        X_proc = self.preprocess(X, quarters)
        return self.model.predict(X_proc)

    def rank_stocks(
        self,
        X: np.ndarray,
        quarters: np.ndarray,
        symbols: np.ndarray
    ) -> list[dict]:
        """
        Rank stocks by predicted alpha within each quarter.

        Returns list of {quarter, symbol, predicted_alpha, rank}
        """
        predictions = self.predict(X, quarters)

        results = []
        for q in np.unique(quarters):
            mask = quarters == q
            q_preds = predictions[mask]
            q_symbols = symbols[mask]

            # Rank within quarter (1 = highest predicted alpha)
            ranks = (-q_preds).argsort().argsort() + 1

            for sym, pred, rank in zip(q_symbols, q_preds, ranks):
                results.append({
                    "quarter": q,
                    "symbol": sym,
                    "predicted_alpha": float(pred),
                    "rank": int(rank),
                    "percentile": int(100 * (1 - rank / len(q_preds)))
                })

        return results
```

#### 4.2 Output Format

The model outputs:
1. **Predicted alpha** for each stock in each quarter
2. **Rank** within quarter (for "top N" selection)
3. **Coefficients** with stability metrics
4. **IC** (information coefficient) over time

#### 4.3 Converting to Pipeline Settings

```python
# Add to src/ml_models/pipeline_converter.py

def elastic_net_to_pipeline(
    model: ElasticNetAlphaModel,
    threshold_percentile: int = 20  # Top 20% by predicted alpha
) -> PipelineSettings:
    """
    Convert Elastic Net predictions to Pipeline settings.

    Strategy: "Take top N% of stocks by model score"

    This is fundamentally different from factor thresholds.
    We store the model coefficients and apply at inference time.
    """

    # Get significant coefficients
    coefs = model.model.coef_
    feature_names = model.feature_names

    significant_features = [
        {"factor": name, "coefficient": float(coef), "stability": model.coefficient_stability[name]}
        for name, coef in zip(feature_names, coefs)
        if abs(coef) > 0.01
    ]
    significant_features.sort(key=lambda x: abs(x["coefficient"]), reverse=True)

    return PipelineSettings(
        # Store model info for inference
        ml_model_type="elastic_net",
        ml_model_threshold_percentile=threshold_percentile,
        ml_model_features=significant_features,
        # Raw filters empty - we use model ranking instead
        raw_filters=[]
    )
```

#### 4.4 Database Storage

```sql
-- Add to schema.py

CREATE TABLE IF NOT EXISTS ml_model_runs (
    run_id VARCHAR PRIMARY KEY,
    model_type VARCHAR NOT NULL,  -- "elastic_net", "gam", "xgboost"
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    -- Config
    holding_period INT,
    train_quarters TEXT,  -- JSON array
    feature_categories TEXT,  -- JSON array
    hyperparameters TEXT,  -- JSON object

    -- Results
    mean_ic DECIMAL(8,4),
    ic_std DECIMAL(8,4),
    n_features_selected INT,

    -- Serialized model (for inference)
    model_blob BLOB,

    -- Status
    status VARCHAR DEFAULT 'completed'
);

CREATE TABLE IF NOT EXISTS ml_model_coefficients (
    run_id VARCHAR REFERENCES ml_model_runs(run_id),
    feature_name VARCHAR NOT NULL,
    coefficient DECIMAL(12,6),
    stability DECIMAL(6,4),  -- % of windows with same sign
    PRIMARY KEY (run_id, feature_name)
);

CREATE TABLE IF NOT EXISTS ml_model_predictions (
    run_id VARCHAR REFERENCES ml_model_runs(run_id),
    quarter VARCHAR NOT NULL,
    symbol VARCHAR NOT NULL,
    predicted_alpha DECIMAL(10,4),
    rank INT,
    percentile INT,
    PRIMARY KEY (run_id, quarter, symbol)
);
```

---

## 5. Priority 4: GAM (Generalized Additive Model)

### Why GAM?

- **Non-linear:** Captures sweet spots and saturation
- **Interpretable:** Can plot partial dependence for each feature
- **Additive:** No black-box interactions (unlike trees)
- **Visual:** Shows "where does PE ratio stop helping?"

### Implementation

#### 5.1 Model Architecture

```python
# New file: src/ml_models/gam_model.py

from pygam import LinearGAM, s, f
import numpy as np

class GAMAlphaModel:
    """
    Generalized Additive Model for alpha prediction.

    y = s(feature1) + s(feature2) + ... + f(categorical1) + ...

    Where s() is a smooth spline term and f() is a factor term.
    """

    def __init__(
        self,
        n_splines: int = 10,
        lam: float = 0.6,  # Regularization (higher = smoother)
    ):
        self.n_splines = n_splines
        self.lam = lam
        self.model = None
        self.feature_names = None
        self.feature_types = None  # "numerical" or "categorical"

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        quarters: np.ndarray,
        feature_names: list[str],
        feature_types: list[str]
    ) -> dict:
        """Train GAM with grid search for smoothing parameter."""

        self.feature_names = feature_names
        self.feature_types = feature_types

        # Preprocess (same as Elastic Net)
        X_proc = self._preprocess(X, quarters)

        # Build GAM formula
        terms = []
        for i, (name, ftype) in enumerate(zip(feature_names, feature_types)):
            if ftype == "numerical":
                terms.append(s(i, n_splines=self.n_splines, lam=self.lam))
            else:
                terms.append(f(i))

        # Fit GAM
        self.model = LinearGAM(terms[0])
        for term in terms[1:]:
            self.model += term

        self.model.fit(X_proc, y)

        # Extract partial dependence for each feature
        partial_deps = self._compute_partial_dependence(X_proc)

        return {
            "r_squared": self.model.statistics_["pseudo_r2"]["explained_deviance"],
            "partial_dependence": partial_deps,
            "feature_importance": self._compute_feature_importance(X_proc, y)
        }

    def _compute_partial_dependence(self, X: np.ndarray) -> dict:
        """
        Compute partial dependence plots for each feature.

        This shows the shape of each feature's effect on alpha.
        """
        partial_deps = {}

        for i, name in enumerate(self.feature_names):
            if self.feature_types[i] == "numerical":
                # Create grid of values
                x_grid = np.linspace(X[:, i].min(), X[:, i].max(), 100)

                # Get partial dependence
                pdep = self.model.partial_dependence(term=i, X=x_grid)

                partial_deps[name] = {
                    "x": x_grid.tolist(),
                    "y": pdep.tolist(),
                    "optimal_range": self._find_optimal_range(x_grid, pdep)
                }

        return partial_deps

    def _find_optimal_range(
        self,
        x_grid: np.ndarray,
        pdep: np.ndarray,
        threshold_pct: float = 0.8
    ) -> dict:
        """
        Find the range of x values where the partial dependence is highest.

        This identifies "sweet spots" - e.g., "PE between 8 and 15 is optimal"
        """
        max_effect = pdep.max()
        threshold = max_effect * threshold_pct

        above_threshold = x_grid[pdep >= threshold]

        if len(above_threshold) > 0:
            return {
                "min": float(above_threshold.min()),
                "max": float(above_threshold.max()),
                "peak": float(x_grid[pdep.argmax()])
            }
        return None
```

#### 5.2 Partial Dependence Visualization

The key output of GAM is partial dependence plots. For the frontend:

```typescript
// Component to display partial dependence plot
interface PartialDependenceProps {
  featureName: string;
  x: number[];
  y: number[];
  optimalRange: { min: number; max: number; peak: number } | null;
}

function PartialDependencePlot({ featureName, x, y, optimalRange }: PartialDependenceProps) {
  // Render line chart with x (feature values) vs y (partial effect on alpha)
  // Highlight optimal range in green
  // Mark peak with vertical line
}
```

#### 5.3 GAM Insights → Pipeline Rules

Unlike Elastic Net, GAM can identify **non-monotonic** relationships:

```python
def gam_to_pipeline_rules(
    partial_deps: dict[str, dict]
) -> list[FilterSpec]:
    """
    Convert GAM partial dependence to filter rules.

    If PE shows peak effect at 10-15, output rule: "10 <= PE <= 15"
    """
    rules = []

    for feature, pdep in partial_deps.items():
        optimal = pdep.get("optimal_range")
        if optimal:
            # Two-sided filter for sweet spot
            rules.append(FilterSpec(
                factor=feature,
                operator=">=",
                value=optimal["min"]
            ))
            rules.append(FilterSpec(
                factor=feature,
                operator="<=",
                value=optimal["max"]
            ))

    return rules
```

---

## 6. Priority 5: Gradient Boosted Trees

### Why Last?

- **Overfitting risk:** Trees memorize noise with small data
- **Requires more data:** Need 100k+ observations ideally
- **Black box:** Feature importance doesn't show direction
- **Hyperparameter sensitive:** Many knobs to tune

### When to Use

Only consider XGBoost/LightGBM if:
1. You have 100+ quarters of data
2. Elastic Net IC < 0.03 (linear model failing)
3. Strong reason to believe conditional effects exist

### Implementation (Minimal)

```python
# New file: src/ml_models/tree_model.py

from lightgbm import LGBMRegressor
import numpy as np

class TreeAlphaModel:
    """
    Gradient Boosted Trees for alpha prediction.

    WARNING: High overfitting risk with financial data.
    Use only as last resort with strict regularization.
    """

    def __init__(self):
        self.model = LGBMRegressor(
            # AGGRESSIVE regularization
            n_estimators=100,  # Few trees
            max_depth=3,  # Shallow trees
            min_child_samples=50,  # Large leaves
            subsample=0.7,  # Row sampling
            colsample_bytree=0.7,  # Column sampling
            reg_alpha=0.1,  # L1
            reg_lambda=0.1,  # L2
            learning_rate=0.05,  # Slow learning
        )

    def train(self, X, y, quarters, feature_names):
        """Train with early stopping on time-series validation."""

        # Split: train on first 80%, validate on last 20%
        unique_quarters = sorted(np.unique(quarters))
        split_idx = int(len(unique_quarters) * 0.8)

        train_quarters = unique_quarters[:split_idx]
        val_quarters = unique_quarters[split_idx:]

        train_mask = np.isin(quarters, train_quarters)
        val_mask = np.isin(quarters, val_quarters)

        self.model.fit(
            X[train_mask], y[train_mask],
            eval_set=[(X[val_mask], y[val_mask])],
            callbacks=[early_stopping(10)]  # Stop if no improvement
        )

        # Feature importance
        importance = dict(zip(feature_names, self.model.feature_importances_))

        return {
            "feature_importance": importance,
            "n_trees_used": self.model.best_iteration_,
            "overfit_warning": self._check_overfitting(X, y, train_mask, val_mask)
        }

    def _check_overfitting(self, X, y, train_mask, val_mask) -> str | None:
        """Check for overfitting symptoms."""
        train_ic = spearmanr(self.model.predict(X[train_mask]), y[train_mask])[0]
        val_ic = spearmanr(self.model.predict(X[val_mask]), y[val_mask])[0]

        ratio = val_ic / train_ic if train_ic > 0 else 0

        if ratio < 0.3:
            return "SEVERE: Model memorizing training data"
        elif ratio < 0.5:
            return "WARNING: Significant overfitting detected"
        elif ratio < 0.7:
            return "CAUTION: Moderate overfitting"
        return None
```

---

## 7. Frontend Architecture

### Recommended: Single "Models" Page with Tabs

```
/models
├── Tab: Factor Screening (current Factor Discovery)
├── Tab: Elastic Net
├── Tab: GAM
└── Tab: XGBoost (future)
```

### Shared Components

```
src/components/models/
├── SharedConfig.tsx          # Quarter selection, holding period, exclusions
├── FeatureSelector.tsx       # Factor categories
├── ProgressTracker.tsx       # SSE progress display
├── SaveStrategyButton.tsx    # Save to pipeline
└── ModelResults/
    ├── FactorScreeningResults.tsx
    ├── ElasticNetResults.tsx
    │   ├── CoefficientTable.tsx
    │   ├── CoefficientStabilityChart.tsx
    │   └── ICOverTimeChart.tsx
    ├── GAMResults.tsx
    │   ├── PartialDependencePlots.tsx
    │   └── OptimalRangesTable.tsx
    └── TreeResults.tsx
        ├── FeatureImportanceChart.tsx
        └── OverfitWarning.tsx
```

### Page Structure

```tsx
// src/app/models/page.tsx

export default function ModelsPage() {
  const [modelType, setModelType] = useState<"factor_screening" | "elastic_net" | "gam" | "xgboost">("factor_screening");

  // Shared state
  const [quarters, setQuarters] = useState<string[]>([]);
  const [holdingPeriod, setHoldingPeriod] = useState(4);
  const [featureCategories, setFeatureCategories] = useState<string[]>([...]);

  // Model-specific state
  const [runId, setRunId] = useState<string | null>(null);
  const [results, setResults] = useState<ModelResults | null>(null);

  return (
    <div>
      {/* Model Type Selector */}
      <ModelTypeSelector value={modelType} onChange={setModelType} />

      {/* Shared Configuration */}
      <SharedConfig
        quarters={quarters}
        setQuarters={setQuarters}
        holdingPeriod={holdingPeriod}
        setHoldingPeriod={setHoldingPeriod}
        featureCategories={featureCategories}
        setFeatureCategories={setFeatureCategories}
      />

      {/* Model-Specific Configuration */}
      {modelType === "elastic_net" && <ElasticNetConfig ... />}
      {modelType === "gam" && <GAMConfig ... />}
      {modelType === "xgboost" && <XGBoostConfig ... />}

      {/* Run Button & Progress */}
      <RunModelButton modelType={modelType} ... />
      {runId && <ProgressTracker runId={runId} />}

      {/* Results */}
      {results && (
        <ModelResults modelType={modelType} results={results} />
      )}

      {/* Save to Pipeline */}
      {results && <SaveStrategyButton results={results} />}
    </div>
  );
}
```

---

## 8. Database Schema Changes

### Summary of New Tables

```sql
-- 1. Macro indicators for regime features
CREATE TABLE macro_indicators (...);  -- See Section 3

-- 2. ML model runs
CREATE TABLE ml_model_runs (...);  -- See Section 4

-- 3. Model coefficients (Elastic Net, GAM)
CREATE TABLE ml_model_coefficients (...);

-- 4. Model predictions
CREATE TABLE ml_model_predictions (...);

-- 5. Factor decay metrics (extends existing)
ALTER TABLE factor_results ADD COLUMN decay_score DECIMAL(6,4);
ALTER TABLE factor_results ADD COLUMN ic_stability DECIMAL(6,4);
ALTER TABLE factor_results ADD COLUMN alpha_trend DECIMAL(6,4);
```

### Migration Script

```python
# scripts/migrate_add_ml_tables.py

def migrate():
    """Add ML model tables to database."""
    db = get_db_connection()

    db.execute(MACRO_INDICATORS_SQL)
    db.execute(ML_MODEL_RUNS_SQL)
    db.execute(ML_MODEL_COEFFICIENTS_SQL)
    db.execute(ML_MODEL_PREDICTIONS_SQL)

    # Add decay columns to existing table
    db.execute("""
        ALTER TABLE factor_results
        ADD COLUMN IF NOT EXISTS decay_score DECIMAL(6,4)
    """)
    # etc.
```

---

## 9. API Endpoints

### New Routes

```python
# src/api/routes/ml_models.py

router = APIRouter(prefix="/ml-models", tags=["ML Models"])

@router.post("/elastic-net/run")
async def run_elastic_net(request: ElasticNetRequest) -> {"run_id": str}:
    """Start Elastic Net training."""

@router.get("/elastic-net/progress/{run_id}")
async def get_progress(run_id: str) -> EventSourceResponse:
    """SSE stream for training progress."""

@router.get("/elastic-net/results/{run_id}")
async def get_results(run_id: str) -> ElasticNetResult:
    """Get training results with coefficients."""

@router.post("/gam/run")
async def run_gam(request: GAMRequest) -> {"run_id": str}:
    """Start GAM training."""

@router.get("/gam/results/{run_id}")
async def get_gam_results(run_id: str) -> GAMResult:
    """Get GAM results with partial dependence."""

@router.get("/predictions/{run_id}")
async def get_predictions(run_id: str, quarter: str) -> list[StockPrediction]:
    """Get model predictions for a specific quarter."""

@router.get("/macro/indicators")
async def get_macro_indicators(
    from_quarter: str = None,
    to_quarter: str = None
) -> list[MacroIndicator]:
    """Get macro indicators for regime analysis."""
```

---

## 10. Risk Mitigation

### Overfitting Safeguards

| Risk | Mitigation |
|------|------------|
| Too many features | L1 regularization (Elastic Net), limit features |
| Data snooping | Strict time-series CV, no future leakage |
| Multiple testing | Track all experiments, use FDR correction |
| Curve fitting | Penalize unstable factors (decay score) |
| Trees memorizing | Aggressive regularization, shallow trees |

### Implementation Safeguards

```python
class ModelValidator:
    """Validates model results before saving."""

    def validate(self, results: dict, model_type: str) -> list[str]:
        warnings = []

        # IC sanity check
        if results.get("mean_ic", 0) > 0.10:
            warnings.append("IC > 10% is suspiciously high. Check for leakage.")

        # Overfitting check
        if results.get("overfit_ratio", 1.0) < 0.5:
            warnings.append("Validation alpha < 50% of training alpha. Overfitting likely.")

        # Sample size check
        if results.get("n_samples", 0) < 5000:
            warnings.append("Sample size < 5000. Results may not generalize.")

        # Coefficient stability
        unstable = [
            f for f, s in results.get("coefficient_stability", {}).items()
            if s < 0.5
        ]
        if unstable:
            warnings.append(f"Unstable coefficients: {unstable}")

        return warnings
```

### Red Flag Dashboard

Display prominent warnings in UI:
- "Overfitting Detected" badge
- IC declining trend chart
- Comparison: in-sample vs out-of-sample metrics

---

## 11. Success Metrics

### Model Quality Metrics

| Metric | Target | Red Flag |
|--------|--------|----------|
| Mean IC | > 0.02 | < 0.01 |
| IC Stability (% positive) | > 70% | < 50% |
| Overfit Ratio (val/train) | > 0.7 | < 0.5 |
| Coefficient Stability | > 60% | < 40% |
| Decay Score (factors used) | > 70% | < 50% |

### Business Metrics

| Metric | Measurement |
|--------|-------------|
| Alpha improvement | Backtest: Model ranking vs current ranking |
| Win rate | % of model top picks with positive alpha |
| Stability | Consistent performance across quarters |

---

## 12. Implementation Timeline

### Phase 1: Factor Decay (Foundation)

**Files:**
- `src/factor_discovery/decay_analyzer.py` (new)
- `src/factor_discovery/models.py` (modify)
- `src/factor_discovery/factor_analyzer.py` (modify)
- `frontend/src/app/factor-discovery/page.tsx` (modify)

**Output:** Decay scores displayed in Factor Discovery results

### Phase 2: Regime Infrastructure

**Files:**
- `src/database/schema.py` (modify)
- `src/scrapers/fmp_client.py` (modify)
- `src/scrapers/macro_fetcher.py` (new)
- `scripts/load_macro_data.py` (new)
- `src/factor_discovery/dataset_builder.py` (modify)

**Output:** Regime features available in dataset

### Phase 3: Elastic Net

**Files:**
- `src/ml_models/` (new directory)
- `src/ml_models/elastic_net_model.py` (new)
- `src/ml_models/pipeline_converter.py` (new)
- `src/api/routes/ml_models.py` (new)
- `frontend/src/app/models/page.tsx` (new or modify factor-discovery)
- `frontend/src/components/models/` (new)

**Output:** Working Elastic Net with coefficient display

### Phase 4: GAM

**Files:**
- `src/ml_models/gam_model.py` (new)
- `frontend/src/components/models/GAMResults.tsx` (new)
- `frontend/src/components/models/PartialDependencePlots.tsx` (new)

**Output:** GAM with partial dependence visualization

### Phase 5: XGBoost (Optional)

**Files:**
- `src/ml_models/tree_model.py` (new)
- `frontend/src/components/models/TreeResults.tsx` (new)

**Output:** XGBoost with overfitting warnings

---

## Appendix A: Python Dependencies

Add to `pyproject.toml`:

```toml
[project]
dependencies = [
    # Existing
    "numpy>=1.24",
    "scipy>=1.10",
    "pandas>=2.0",
    "duckdb>=0.9",
    "fastapi>=0.100",
    "pydantic>=2.0",

    # New for ML
    "scikit-learn>=1.3",
    "pygam>=0.9",

    # Optional (Phase 5)
    # "lightgbm>=4.0",
    # "xgboost>=2.0",
]
```

---

## Appendix B: FMP API Endpoints Reference

```python
# Macro data endpoints (to add to fmp_client.py)

MACRO_ENDPOINTS = {
    # Interest rates
    "treasury_rates": "/stable/treasury-rates",

    # Economic indicators (use ?name= parameter)
    "economic_indicators": "/stable/economic-indicators",
    # Available names: GDP, CPI, unemploymentRate, federalFundsRate,
    #                  retailSales, industrialProduction, etc.

    # Market data
    "market_risk_premium": "/stable/market-risk-premium",
    "batch_index_quotes": "/stable/batch-index-quotes",

    # Index constituents
    "sp500_constituent": "/stable/sp500-constituent",
}
```

---

## Appendix C: Quant Checklist

Before deploying any model:

- [ ] Mean IC > 0.02 over 20+ quarters
- [ ] IC positive in > 70% of quarters
- [ ] Overfit ratio (validation/train) > 0.7
- [ ] No data leakage (features known at prediction time)
- [ ] Transaction costs considered
- [ ] Sample size > 5000 observations
- [ ] Coefficient stability > 60% (for Elastic Net)
- [ ] Decay score > 70% for all factors used
- [ ] Tested across different market regimes
- [ ] Compared to simple baseline (equal weight, single factor)
