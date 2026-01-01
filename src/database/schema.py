"""DuckDB schema definitions for stock analysis data."""

import logging

import duckdb

from src.database.connection import get_db_manager

logger = logging.getLogger(__name__)

# Raw data tables SQL
TICKERS_TABLE = """
CREATE TABLE IF NOT EXISTS tickers (
    symbol VARCHAR PRIMARY KEY,
    name VARCHAR,
    exchange VARCHAR,
    sector VARCHAR,
    industry VARCHAR,
    asset_type VARCHAR,
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    raw_json JSON
)
"""

COMPANY_PROFILES_TABLE = """
CREATE TABLE IF NOT EXISTS company_profiles (
    symbol VARCHAR NOT NULL,
    fiscal_quarter VARCHAR NOT NULL,
    price DECIMAL,
    market_cap DECIMAL,
    pe_ratio DECIMAL,
    pb_ratio DECIMAL,
    beta DECIMAL,
    shares_outstanding BIGINT,
    avg_volume BIGINT,
    dividend_yield DECIMAL,
    fetched_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    raw_json JSON,
    PRIMARY KEY(symbol, fiscal_quarter)
)
"""

INCOME_STATEMENTS_TABLE = """
CREATE TABLE IF NOT EXISTS income_statements (
    symbol VARCHAR NOT NULL,
    fiscal_date DATE NOT NULL,
    period VARCHAR NOT NULL,
    fiscal_year INTEGER,
    reported_currency VARCHAR,
    revenue DECIMAL,
    cost_of_revenue DECIMAL,
    gross_profit DECIMAL,
    gross_profit_ratio DECIMAL,
    research_and_development DECIMAL,
    selling_general_admin DECIMAL,
    operating_expenses DECIMAL,
    operating_income DECIMAL,
    operating_income_ratio DECIMAL,
    interest_expense DECIMAL,
    ebit DECIMAL,
    ebitda DECIMAL,
    income_before_tax DECIMAL,
    income_tax_expense DECIMAL,
    net_income DECIMAL,
    net_income_ratio DECIMAL,
    eps DECIMAL,
    eps_diluted DECIMAL,
    weighted_avg_shares BIGINT,
    weighted_avg_shares_diluted BIGINT,
    fetched_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    raw_json JSON,
    PRIMARY KEY(symbol, fiscal_date, period)
)
"""

BALANCE_SHEETS_TABLE = """
CREATE TABLE IF NOT EXISTS balance_sheets (
    symbol VARCHAR NOT NULL,
    fiscal_date DATE NOT NULL,
    period VARCHAR NOT NULL,
    fiscal_year INTEGER,
    reported_currency VARCHAR,
    -- Assets
    total_assets DECIMAL,
    current_assets DECIMAL,
    cash_and_equivalents DECIMAL,
    short_term_investments DECIMAL,
    net_receivables DECIMAL,
    inventory DECIMAL,
    other_current_assets DECIMAL,
    non_current_assets DECIMAL,
    property_plant_equipment DECIMAL,
    goodwill DECIMAL,
    intangible_assets DECIMAL,
    long_term_investments DECIMAL,
    -- Liabilities
    total_liabilities DECIMAL,
    current_liabilities DECIMAL,
    accounts_payable DECIMAL,
    short_term_debt DECIMAL,
    deferred_revenue DECIMAL,
    other_current_liabilities DECIMAL,
    non_current_liabilities DECIMAL,
    long_term_debt DECIMAL,
    deferred_tax_liabilities DECIMAL,
    other_non_current_liabilities DECIMAL,
    total_debt DECIMAL,
    -- Equity
    total_equity DECIMAL,
    common_stock DECIMAL,
    retained_earnings DECIMAL,
    treasury_stock DECIMAL,
    other_comprehensive_income DECIMAL,
    total_stockholders_equity DECIMAL,
    minority_interest DECIMAL,
    -- Shares
    common_shares_outstanding BIGINT,
    fetched_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    raw_json JSON,
    PRIMARY KEY(symbol, fiscal_date, period)
)
"""

CASH_FLOW_STATEMENTS_TABLE = """
CREATE TABLE IF NOT EXISTS cash_flow_statements (
    symbol VARCHAR NOT NULL,
    fiscal_date DATE NOT NULL,
    period VARCHAR NOT NULL,
    fiscal_year INTEGER,
    reported_currency VARCHAR,
    -- Operating
    net_income DECIMAL,
    depreciation_amortization DECIMAL,
    deferred_income_tax DECIMAL,
    stock_based_compensation DECIMAL,
    change_in_working_capital DECIMAL,
    accounts_receivables DECIMAL,
    inventory DECIMAL,
    accounts_payables DECIMAL,
    other_working_capital DECIMAL,
    other_non_cash_items DECIMAL,
    operating_cash_flow DECIMAL,
    -- Investing
    investments_in_ppe DECIMAL,
    acquisitions DECIMAL,
    purchases_of_investments DECIMAL,
    sales_of_investments DECIMAL,
    other_investing DECIMAL,
    investing_cash_flow DECIMAL,
    -- Financing
    debt_repayment DECIMAL,
    common_stock_issued DECIMAL,
    common_stock_repurchased DECIMAL,
    dividends_paid DECIMAL,
    other_financing DECIMAL,
    financing_cash_flow DECIMAL,
    -- Summary
    net_change_in_cash DECIMAL,
    cash_at_beginning DECIMAL,
    cash_at_end DECIMAL,
    capital_expenditure DECIMAL,
    free_cash_flow DECIMAL,
    fetched_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    raw_json JSON,
    PRIMARY KEY(symbol, fiscal_date, period)
)
"""

KEY_METRICS_TABLE = """
CREATE TABLE IF NOT EXISTS key_metrics (
    symbol VARCHAR NOT NULL,
    fiscal_date DATE NOT NULL,
    period VARCHAR NOT NULL,
    -- Valuation
    revenue_per_share DECIMAL,
    net_income_per_share DECIMAL,
    operating_cash_flow_per_share DECIMAL,
    free_cash_flow_per_share DECIMAL,
    cash_per_share DECIMAL,
    book_value_per_share DECIMAL,
    tangible_book_value_per_share DECIMAL,
    shareholders_equity_per_share DECIMAL,
    interest_debt_per_share DECIMAL,
    -- Ratios
    pe_ratio DECIMAL,
    price_to_sales DECIMAL,
    pb_ratio DECIMAL,
    price_to_free_cash_flow DECIMAL,
    price_to_operating_cash_flow DECIMAL,
    ev_to_sales DECIMAL,
    ev_to_ebitda DECIMAL,
    ev_to_operating_cash_flow DECIMAL,
    ev_to_free_cash_flow DECIMAL,
    enterprise_value DECIMAL,
    -- Profitability
    roe DECIMAL,
    roa DECIMAL,
    roic DECIMAL,
    return_on_tangible_assets DECIMAL,
    gross_profit_margin DECIMAL,
    operating_profit_margin DECIMAL,
    net_profit_margin DECIMAL,
    -- Liquidity
    current_ratio DECIMAL,
    quick_ratio DECIMAL,
    cash_ratio DECIMAL,
    -- Leverage
    debt_ratio DECIMAL,
    debt_to_equity DECIMAL,
    debt_to_assets DECIMAL,
    net_debt_to_ebitda DECIMAL,
    interest_coverage DECIMAL,
    -- Efficiency
    asset_turnover DECIMAL,
    inventory_turnover DECIMAL,
    receivables_turnover DECIMAL,
    payables_turnover DECIMAL,
    -- Per share
    dividend_yield DECIMAL,
    payout_ratio DECIMAL,
    fetched_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    raw_json JSON,
    PRIMARY KEY(symbol, fiscal_date, period)
)
"""

DIVIDENDS_TABLE = """
CREATE TABLE IF NOT EXISTS dividends (
    symbol VARCHAR NOT NULL,
    ex_date DATE NOT NULL,
    declaration_date DATE,
    record_date DATE,
    payment_date DATE,
    amount DECIMAL,
    adjusted_amount DECIMAL,
    fetched_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    raw_json JSON,
    PRIMARY KEY(symbol, ex_date)
)
"""

# Analysis results tables
GRAHAM_RESULTS_TABLE = """
CREATE TABLE IF NOT EXISTS graham_results (
    symbol VARCHAR NOT NULL,
    analysis_quarter VARCHAR NOT NULL,
    computed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    mode VARCHAR NOT NULL DEFAULT 'strict',
    -- Individual criteria pass/fail
    adequate_size BOOLEAN,
    current_ratio_pass BOOLEAN,
    debt_coverage_pass BOOLEAN,
    earnings_stability BOOLEAN,
    dividend_record BOOLEAN,
    earnings_growth_pass BOOLEAN,
    pe_ratio_pass BOOLEAN,
    pb_ratio_pass BOOLEAN,
    -- Computed values
    revenue DECIMAL,
    current_ratio DECIMAL,
    net_current_assets DECIMAL,
    long_term_debt DECIMAL,
    eps_5yr_growth DECIMAL,
    pe_ratio DECIMAL,
    pb_ratio DECIMAL,
    pe_x_pb DECIMAL,
    -- Score
    criteria_passed INTEGER,
    data_quality DECIMAL,
    missing_fields JSON,
    PRIMARY KEY(symbol, analysis_quarter, mode)
)
"""

MAGIC_FORMULA_RESULTS_TABLE = """
CREATE TABLE IF NOT EXISTS magic_formula_results (
    symbol VARCHAR NOT NULL,
    analysis_quarter VARCHAR NOT NULL,
    computed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    -- Computed values
    ebit DECIMAL,
    enterprise_value DECIMAL,
    earnings_yield DECIMAL,
    net_working_capital DECIMAL,
    net_fixed_assets DECIMAL,
    return_on_capital DECIMAL,
    -- Rankings
    earnings_yield_rank INTEGER,
    return_on_capital_rank INTEGER,
    combined_rank INTEGER,
    data_quality DECIMAL,
    missing_fields JSON,
    PRIMARY KEY(symbol, analysis_quarter)
)
"""

PIOTROSKI_RESULTS_TABLE = """
CREATE TABLE IF NOT EXISTS piotroski_results (
    symbol VARCHAR NOT NULL,
    analysis_quarter VARCHAR NOT NULL,
    computed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    -- Profitability signals (4)
    roa_positive BOOLEAN,
    operating_cf_positive BOOLEAN,
    roa_increasing BOOLEAN,
    accruals_signal BOOLEAN,
    -- Leverage signals (3)
    leverage_decreasing BOOLEAN,
    current_ratio_increasing BOOLEAN,
    no_dilution BOOLEAN,
    -- Efficiency signals (2)
    gross_margin_increasing BOOLEAN,
    asset_turnover_increasing BOOLEAN,
    -- Computed values
    roa DECIMAL,
    roa_prior DECIMAL,
    operating_cash_flow DECIMAL,
    net_income DECIMAL,
    long_term_debt DECIMAL,
    long_term_debt_prior DECIMAL,
    current_ratio DECIMAL,
    current_ratio_prior DECIMAL,
    shares_outstanding BIGINT,
    shares_outstanding_prior BIGINT,
    gross_margin DECIMAL,
    gross_margin_prior DECIMAL,
    asset_turnover DECIMAL,
    asset_turnover_prior DECIMAL,
    -- Score
    f_score INTEGER,
    data_quality DECIMAL,
    missing_fields JSON,
    PRIMARY KEY(symbol, analysis_quarter)
)
"""

ALTMAN_RESULTS_TABLE = """
CREATE TABLE IF NOT EXISTS altman_results (
    symbol VARCHAR NOT NULL,
    analysis_quarter VARCHAR NOT NULL,
    computed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    -- Components
    working_capital DECIMAL,
    total_assets DECIMAL,
    retained_earnings DECIMAL,
    ebit DECIMAL,
    market_cap DECIMAL,
    total_liabilities DECIMAL,
    revenue DECIMAL,
    -- Ratios
    x1_wc_ta DECIMAL,
    x2_re_ta DECIMAL,
    x3_ebit_ta DECIMAL,
    x4_mc_tl DECIMAL,
    x5_rev_ta DECIMAL,
    -- Score
    z_score DECIMAL,
    zone VARCHAR,
    data_quality DECIMAL,
    missing_fields JSON,
    PRIMARY KEY(symbol, analysis_quarter)
)
"""

ROIC_QUALITY_RESULTS_TABLE = """
CREATE TABLE IF NOT EXISTS roic_quality_results (
    symbol VARCHAR NOT NULL,
    analysis_quarter VARCHAR NOT NULL,
    computed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    -- Computed values
    ebit DECIMAL,
    effective_tax_rate DECIMAL,
    nopat DECIMAL,
    total_debt DECIMAL,
    total_equity DECIMAL,
    cash DECIMAL,
    invested_capital DECIMAL,
    roic DECIMAL,
    operating_cash_flow DECIMAL,
    capital_expenditure DECIMAL,
    free_cash_flow DECIMAL,
    debt_to_equity DECIMAL,
    -- Pass/fail
    roic_pass BOOLEAN,
    fcf_positive_5yr BOOLEAN,
    debt_to_equity_pass BOOLEAN,
    -- Score
    criteria_passed INTEGER,
    data_quality DECIMAL,
    missing_fields JSON,
    -- NEW: Stability metrics (5yr std dev)
    roic_std_dev DECIMAL,
    roic_stability_tag VARCHAR,
    gross_margin_std_dev DECIMAL,
    gross_margin_stability_tag VARCHAR,
    -- NEW: Quality metrics
    fcf_to_net_income DECIMAL,
    earnings_quality_tag VARCHAR,
    reinvestment_rate DECIMAL,
    reinvestment_tag VARCHAR,
    -- NEW: Valuation metrics
    fcf_yield DECIMAL,
    ev_to_ebit DECIMAL,
    valuation_tag VARCHAR,
    -- NEW: Aggregate tags
    quality_tags JSON,
    PRIMARY KEY(symbol, analysis_quarter)
)
"""

GARP_PEG_RESULTS_TABLE = """
CREATE TABLE IF NOT EXISTS garp_peg_results (
    symbol VARCHAR NOT NULL,
    analysis_quarter VARCHAR NOT NULL,
    computed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    -- Computed values
    price DECIMAL,
    eps DECIMAL,
    pe_ratio DECIMAL,
    eps_growth_1yr DECIMAL,
    eps_growth_3yr DECIMAL,
    eps_growth_5yr DECIMAL,
    eps_cagr DECIMAL,
    peg_ratio DECIMAL,
    -- Pass/fail
    growth_pass BOOLEAN,
    peg_pass BOOLEAN,
    data_quality DECIMAL,
    missing_fields JSON,
    PRIMARY KEY(symbol, analysis_quarter)
)
"""

FAMA_FRENCH_RESULTS_TABLE = """
CREATE TABLE IF NOT EXISTS fama_french_results (
    symbol VARCHAR NOT NULL,
    analysis_quarter VARCHAR NOT NULL,
    computed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    -- Computed values
    book_value DECIMAL,
    market_cap DECIMAL,
    book_to_market DECIMAL,
    gross_profit DECIMAL,
    total_assets DECIMAL,
    profitability DECIMAL,
    assets_prior DECIMAL,
    asset_growth DECIMAL,
    -- Percentile rankings
    book_to_market_percentile DECIMAL,
    profitability_percentile DECIMAL,
    asset_growth_percentile DECIMAL,
    data_quality DECIMAL,
    missing_fields JSON,
    PRIMARY KEY(symbol, analysis_quarter)
)
"""

NET_NET_RESULTS_TABLE = """
CREATE TABLE IF NOT EXISTS net_net_results (
    symbol VARCHAR NOT NULL,
    analysis_quarter VARCHAR NOT NULL,
    computed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    -- Computed values
    current_assets DECIMAL,
    total_liabilities DECIMAL,
    ncav DECIMAL,
    market_cap DECIMAL,
    ncav_per_share DECIMAL,
    price DECIMAL,
    discount_to_ncav DECIMAL,
    -- Pass/fail
    trading_below_ncav BOOLEAN,
    deep_value BOOLEAN,
    data_quality DECIMAL,
    missing_fields JSON,
    PRIMARY KEY(symbol, analysis_quarter)
)
"""

STOCK_RANKINGS_TABLE = """
CREATE TABLE IF NOT EXISTS stock_rankings (
    symbol VARCHAR NOT NULL,
    analysis_quarter VARCHAR NOT NULL,
    computed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    -- Individual scores
    graham_score INTEGER,
    graham_mode VARCHAR,
    magic_formula_rank INTEGER,
    piotroski_score INTEGER,
    altman_zone VARCHAR,
    altman_z_score DECIMAL,
    roic_pass BOOLEAN,
    roic DECIMAL,
    peg_ratio DECIMAL,
    peg_pass BOOLEAN,
    net_net_discount DECIMAL,
    net_net_pass BOOLEAN,
    -- Composite
    composite_score DECIMAL,
    PRIMARY KEY(symbol, analysis_quarter)
)
"""

# Fetch tracking table - uses auto-increment rowid which DuckDB handles implicitly
FETCH_LOG_TABLE = """
CREATE TABLE IF NOT EXISTS fetch_log (
    symbol VARCHAR NOT NULL,
    endpoint VARCHAR NOT NULL,
    status VARCHAR NOT NULL,
    error_message VARCHAR,
    fetched_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    response_time_ms INTEGER
)
-- Note: fetch_log has no primary key as it's an append-only log table
"""

# SPY benchmark prices table
SPY_PRICES_TABLE = """
CREATE TABLE IF NOT EXISTS spy_prices (
    quarter VARCHAR PRIMARY KEY,
    price DECIMAL NOT NULL,
    price_date DATE,
    fetched_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)
"""

# Sector average returns table (for sector-adjusted alpha calculation)
SECTOR_RETURNS_TABLE = """
CREATE TABLE IF NOT EXISTS sector_returns (
    sector VARCHAR NOT NULL,
    quarter VARCHAR NOT NULL,
    holding_period INT NOT NULL,
    avg_return DECIMAL,
    median_return DECIMAL,
    stock_count INT,
    calculated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY(sector, quarter, holding_period)
)
"""

# ============================================================================
# Macroeconomic Regime Tables
# ============================================================================

MACRO_INDICATORS_TABLE = """
CREATE TABLE IF NOT EXISTS macro_indicators (
    quarter VARCHAR PRIMARY KEY,
    indicator_date DATE,
    treasury_1m DECIMAL,
    treasury_3m DECIMAL,
    treasury_6m DECIMAL,
    treasury_1y DECIMAL,
    treasury_2y DECIMAL,
    treasury_5y DECIMAL,
    treasury_10y DECIMAL,
    treasury_30y DECIMAL,
    yield_curve_spread DECIMAL,
    fetched_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    raw_json JSON
)
"""

REGIME_FLAGS_TABLE = """
CREATE TABLE IF NOT EXISTS regime_flags (
    quarter VARCHAR PRIMARY KEY,
    computed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    rate_regime VARCHAR,
    rate_change_qoq DECIMAL
)
"""

# Grid search results table
GRID_SEARCHES_TABLE = """
CREATE TABLE IF NOT EXISTS grid_searches (
    id VARCHAR PRIMARY KEY,
    name VARCHAR,
    started_at TIMESTAMP NOT NULL,
    completed_at TIMESTAMP,
    status VARCHAR NOT NULL DEFAULT 'running',
    total_simulations INTEGER,
    completed_simulations INTEGER,
    duration_seconds DECIMAL,
    -- Request configuration (JSON)
    request_config JSON,
    -- Results summary
    best_alpha DECIMAL,
    best_win_rate DECIMAL,
    -- Full results (JSON array of simulation results)
    results_json JSON,
    -- Top results for quick display
    best_by_alpha_json JSON,
    best_by_win_rate_json JSON
)
"""

# ============================================================================
# Factor Discovery Tables
# ============================================================================

# Run metadata table
FACTOR_ANALYSIS_RUNS_TABLE = """
CREATE TABLE IF NOT EXISTS factor_analysis_runs (
    id VARCHAR PRIMARY KEY,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMP,
    status VARCHAR NOT NULL DEFAULT 'running',
    quarters_analyzed JSON,
    holding_periods JSON,
    total_observations INTEGER,
    config JSON,
    error_message VARCHAR,
    duration_seconds DECIMAL
)
"""

# Individual factor results
FACTOR_RESULTS_TABLE = """
CREATE TABLE IF NOT EXISTS factor_results (
    run_id VARCHAR NOT NULL,
    holding_period INTEGER NOT NULL,
    factor_name VARCHAR NOT NULL,
    factor_type VARCHAR NOT NULL,
    -- Overall stats
    correlation DOUBLE,
    correlation_pvalue DOUBLE,
    -- Threshold analysis (JSON array)
    threshold_results JSON,
    -- Best threshold
    best_threshold VARCHAR,
    best_threshold_alpha DOUBLE,
    best_threshold_lift DOUBLE,
    best_threshold_pvalue DOUBLE,
    best_threshold_sample_size INTEGER,
    best_threshold_ci_lower DOUBLE,
    best_threshold_ci_upper DOUBLE,
    -- Decay metrics (rolling window stability)
    decay_score DOUBLE,
    decay_ic_stability DOUBLE,
    decay_alpha_trend DOUBLE,
    decay_n_windows INTEGER,
    decay_recent_alpha DOUBLE,
    decay_mean_ic DOUBLE,
    PRIMARY KEY (run_id, holding_period, factor_name)
)
"""

# Combined strategy results
COMBINED_STRATEGY_RESULTS_TABLE = """
CREATE TABLE IF NOT EXISTS combined_strategy_results (
    run_id VARCHAR NOT NULL,
    holding_period INTEGER NOT NULL,
    strategy_rank INTEGER NOT NULL,
    -- Strategy definition
    filters JSON,
    -- Performance
    mean_alpha DOUBLE,
    sample_size INTEGER,
    lift DOUBLE,
    win_rate DOUBLE,
    ci_lower DOUBLE,
    ci_upper DOUBLE,
    PRIMARY KEY (run_id, holding_period, strategy_rank)
)
"""

# Recommended strategies (one per run per holding period)
RECOMMENDED_STRATEGIES_TABLE = """
CREATE TABLE IF NOT EXISTS recommended_strategies (
    run_id VARCHAR NOT NULL,
    holding_period INTEGER NOT NULL,
    -- Pipeline-ready settings
    pipeline_settings JSON,
    -- Expected performance
    expected_alpha DOUBLE,
    expected_alpha_ci_lower DOUBLE,
    expected_alpha_ci_upper DOUBLE,
    expected_win_rate DOUBLE,
    sample_size INTEGER,
    confidence_score DOUBLE,
    -- Explanation
    key_factors JSON,
    PRIMARY KEY (run_id, holding_period)
)
"""

# ============================================================================
# ML Model Tables (Elastic Net, GAM, etc.)
# ============================================================================

ML_MODEL_RUNS_TABLE = """
CREATE TABLE IF NOT EXISTS ml_model_runs (
    id VARCHAR PRIMARY KEY,
    model_type VARCHAR NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMP,
    status VARCHAR NOT NULL DEFAULT 'running',
    holding_period INTEGER NOT NULL,
    train_end_quarter VARCHAR NOT NULL,
    test_start_quarter VARCHAR NOT NULL,
    config_json JSON NOT NULL,
    train_ic DOUBLE,
    test_ic DOUBLE,
    train_r2 DOUBLE,
    n_train_samples INTEGER,
    n_test_samples INTEGER,
    best_alpha DOUBLE,
    best_l1_ratio DOUBLE,
    n_features_selected INTEGER,
    duration_seconds DOUBLE,
    error_message VARCHAR
)
"""

ML_MODEL_COEFFICIENTS_TABLE = """
CREATE TABLE IF NOT EXISTS ml_model_coefficients (
    run_id VARCHAR NOT NULL,
    feature_name VARCHAR NOT NULL,
    coefficient DOUBLE NOT NULL,
    coefficient_std DOUBLE,
    stability_score DOUBLE,
    feature_importance_rank INTEGER,
    PRIMARY KEY (run_id, feature_name)
)
"""

ML_MODEL_IC_HISTORY_TABLE = """
CREATE TABLE IF NOT EXISTS ml_model_ic_history (
    run_id VARCHAR NOT NULL,
    quarter VARCHAR NOT NULL,
    ic DOUBLE NOT NULL,
    ic_pvalue DOUBLE,
    n_samples INTEGER,
    PRIMARY KEY (run_id, quarter)
)
"""

# GAM-specific: Partial dependence curves
ML_MODEL_PARTIAL_DEPENDENCE_TABLE = """
CREATE TABLE IF NOT EXISTS ml_model_partial_dependence (
    run_id VARCHAR NOT NULL,
    feature_name VARCHAR NOT NULL,
    x_values JSON NOT NULL,
    y_values JSON NOT NULL,
    optimal_min DOUBLE,
    optimal_max DOUBLE,
    peak_x DOUBLE,
    peak_y DOUBLE NOT NULL,
    importance_rank INTEGER NOT NULL,
    PRIMARY KEY (run_id, feature_name)
)
"""

# LightGBM-specific: Feature importance (gain and split count)
ML_MODEL_FEATURE_IMPORTANCE_TABLE = """
CREATE TABLE IF NOT EXISTS ml_model_feature_importance (
    run_id VARCHAR NOT NULL,
    feature_name VARCHAR NOT NULL,
    importance_gain DOUBLE NOT NULL,
    importance_split DOUBLE NOT NULL,
    importance_rank INTEGER NOT NULL,
    PRIMARY KEY (run_id, feature_name)
)
"""

# ============================================================================
# User Saved Strategies Table
# ============================================================================

SAVED_STRATEGIES_TABLE = """
CREATE TABLE IF NOT EXISTS saved_strategies (
    id VARCHAR PRIMARY KEY,
    name VARCHAR NOT NULL,
    holding_period INTEGER,
    -- Full settings as JSON (includes PipelineSettings + raw_filters)
    settings_json JSON NOT NULL,
    -- Performance expectations from Factor Discovery
    expected_alpha DOUBLE,
    expected_alpha_ci_lower DOUBLE,
    expected_alpha_ci_upper DOUBLE,
    win_rate DOUBLE,
    sample_size INTEGER,
    -- Metadata
    source VARCHAR DEFAULT 'manual',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)
"""

# ============================================================================
# Portfolio Tracking Tables
# ============================================================================

PORTFOLIO_BATCHES_TABLE = """
CREATE TABLE IF NOT EXISTS portfolio_batches (
    id VARCHAR PRIMARY KEY,
    name VARCHAR,
    buy_quarter VARCHAR NOT NULL,
    strategy_id VARCHAR,
    holding_period INTEGER NOT NULL DEFAULT 4,
    total_invested DECIMAL NOT NULL DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    status VARCHAR NOT NULL DEFAULT 'active',
    notes VARCHAR
)
"""

PORTFOLIO_POSITIONS_TABLE = """
CREATE TABLE IF NOT EXISTS portfolio_positions (
    id VARCHAR PRIMARY KEY,
    batch_id VARCHAR NOT NULL,
    symbol VARCHAR NOT NULL,
    status VARCHAR NOT NULL DEFAULT 'open',
    invested_amount DECIMAL NOT NULL,
    target_sell_quarter VARCHAR NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    actual_sell_price DECIMAL,
    actual_sell_quarter VARCHAR,
    realized_return DECIMAL,
    realized_alpha DECIMAL
)
"""

POSITION_TRANCHES_TABLE = """
CREATE TABLE IF NOT EXISTS position_tranches (
    id VARCHAR PRIMARY KEY,
    position_id VARCHAR NOT NULL,
    buy_quarter VARCHAR NOT NULL,
    buy_price DECIMAL NOT NULL,
    invested_amount DECIMAL NOT NULL,
    source_batch_id VARCHAR NOT NULL,
    tranche_target_sell_quarter VARCHAR NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    sell_price DECIMAL,
    sell_quarter VARCHAR,
    tranche_return DECIMAL,
    tranche_alpha DECIMAL
)
"""

PORTFOLIO_TRANSACTIONS_TABLE = """
CREATE TABLE IF NOT EXISTS portfolio_transactions (
    id VARCHAR PRIMARY KEY,
    transaction_type VARCHAR NOT NULL,
    batch_id VARCHAR,
    position_id VARCHAR,
    symbol VARCHAR NOT NULL,
    quarter VARCHAR NOT NULL,
    price DECIMAL,
    amount DECIMAL,
    return_pct DECIMAL,
    alpha_pct DECIMAL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)
"""

ALL_TABLES = [
    ("tickers", TICKERS_TABLE),
    ("company_profiles", COMPANY_PROFILES_TABLE),
    ("income_statements", INCOME_STATEMENTS_TABLE),
    ("balance_sheets", BALANCE_SHEETS_TABLE),
    ("cash_flow_statements", CASH_FLOW_STATEMENTS_TABLE),
    ("key_metrics", KEY_METRICS_TABLE),
    ("dividends", DIVIDENDS_TABLE),
    ("graham_results", GRAHAM_RESULTS_TABLE),
    ("magic_formula_results", MAGIC_FORMULA_RESULTS_TABLE),
    ("piotroski_results", PIOTROSKI_RESULTS_TABLE),
    ("altman_results", ALTMAN_RESULTS_TABLE),
    ("roic_quality_results", ROIC_QUALITY_RESULTS_TABLE),
    ("garp_peg_results", GARP_PEG_RESULTS_TABLE),
    ("fama_french_results", FAMA_FRENCH_RESULTS_TABLE),
    ("net_net_results", NET_NET_RESULTS_TABLE),
    ("stock_rankings", STOCK_RANKINGS_TABLE),
    ("fetch_log", FETCH_LOG_TABLE),
    ("spy_prices", SPY_PRICES_TABLE),
    ("sector_returns", SECTOR_RETURNS_TABLE),
    # Macro regime tables
    ("macro_indicators", MACRO_INDICATORS_TABLE),
    ("regime_flags", REGIME_FLAGS_TABLE),
    ("grid_searches", GRID_SEARCHES_TABLE),
    # Factor Discovery tables
    ("factor_analysis_runs", FACTOR_ANALYSIS_RUNS_TABLE),
    ("factor_results", FACTOR_RESULTS_TABLE),
    ("combined_strategy_results", COMBINED_STRATEGY_RESULTS_TABLE),
    ("recommended_strategies", RECOMMENDED_STRATEGIES_TABLE),
    # ML Model tables
    ("ml_model_runs", ML_MODEL_RUNS_TABLE),
    ("ml_model_coefficients", ML_MODEL_COEFFICIENTS_TABLE),
    ("ml_model_ic_history", ML_MODEL_IC_HISTORY_TABLE),
    ("ml_model_partial_dependence", ML_MODEL_PARTIAL_DEPENDENCE_TABLE),
    ("ml_model_feature_importance", ML_MODEL_FEATURE_IMPORTANCE_TABLE),
    # User saved strategies
    ("saved_strategies", SAVED_STRATEGIES_TABLE),
    # Portfolio tracking
    ("portfolio_batches", PORTFOLIO_BATCHES_TABLE),
    ("portfolio_positions", PORTFOLIO_POSITIONS_TABLE),
    ("position_tranches", POSITION_TRANCHES_TABLE),
    ("portfolio_transactions", PORTFOLIO_TRANSACTIONS_TABLE),
]

# Performance indexes for analysis queries
ALL_INDEXES = [
    ("idx_income_period_symbol", "CREATE INDEX IF NOT EXISTS idx_income_period_symbol ON income_statements(period, symbol, fiscal_date DESC)"),
    ("idx_balance_period_symbol", "CREATE INDEX IF NOT EXISTS idx_balance_period_symbol ON balance_sheets(period, symbol, fiscal_date DESC)"),
    ("idx_cashflow_period_symbol", "CREATE INDEX IF NOT EXISTS idx_cashflow_period_symbol ON cash_flow_statements(period, symbol, fiscal_date DESC)"),
    ("idx_metrics_period_symbol", "CREATE INDEX IF NOT EXISTS idx_metrics_period_symbol ON key_metrics(period, symbol, fiscal_date DESC)"),
    ("idx_dividends_symbol", "CREATE INDEX IF NOT EXISTS idx_dividends_symbol ON dividends(symbol, ex_date DESC)"),
    ("idx_profiles_quarter", "CREATE INDEX IF NOT EXISTS idx_profiles_quarter ON company_profiles(fiscal_quarter, symbol)"),
    # Portfolio tracking indexes
    ("idx_positions_batch", "CREATE INDEX IF NOT EXISTS idx_positions_batch ON portfolio_positions(batch_id)"),
    ("idx_positions_symbol", "CREATE INDEX IF NOT EXISTS idx_positions_symbol ON portfolio_positions(symbol)"),
    ("idx_positions_status", "CREATE INDEX IF NOT EXISTS idx_positions_status ON portfolio_positions(status)"),
    ("idx_tranches_position", "CREATE INDEX IF NOT EXISTS idx_tranches_position ON position_tranches(position_id)"),
    ("idx_transactions_symbol", "CREATE INDEX IF NOT EXISTS idx_transactions_symbol ON portfolio_transactions(symbol)"),
]


def create_all_tables(conn: duckdb.DuckDBPyConnection | None = None) -> None:
    """Create all database tables if they don't exist.

    Args:
        conn: Optional existing connection. Creates new one if not provided.
    """
    if conn is None:
        db = get_db_manager()
        with db.get_connection() as conn:
            _create_tables(conn)
    else:
        _create_tables(conn)


def _create_tables(conn: duckdb.DuckDBPyConnection) -> None:
    """Internal function to create tables with given connection."""
    for table_name, create_sql in ALL_TABLES:
        try:
            conn.execute(create_sql)
            logger.info(f"Created/verified table: {table_name}")
        except Exception as e:
            logger.error(f"Error creating table {table_name}: {e}")
            raise

    # Create indexes for performance
    for index_name, create_sql in ALL_INDEXES:
        try:
            conn.execute(create_sql)
            logger.info(f"Created/verified index: {index_name}")
        except Exception as e:
            logger.error(f"Error creating index {index_name}: {e}")
            raise


def create_indexes(conn: duckdb.DuckDBPyConnection | None = None) -> None:
    """Create performance indexes on existing database.

    This can be run on an existing database to add indexes.

    Args:
        conn: Optional existing connection. Creates new one if not provided.
    """
    if conn is None:
        db = get_db_manager()
        with db.get_connection() as conn:
            _create_indexes(conn)
    else:
        _create_indexes(conn)


def _create_indexes(conn: duckdb.DuckDBPyConnection) -> None:
    """Internal function to create indexes."""
    for index_name, create_sql in ALL_INDEXES:
        try:
            conn.execute(create_sql)
            logger.info(f"Created index: {index_name}")
        except Exception as e:
            logger.error(f"Error creating index {index_name}: {e}")
            raise


def drop_all_tables(conn: duckdb.DuckDBPyConnection | None = None) -> None:
    """Drop all tables. Use with caution!

    Args:
        conn: Optional existing connection.
    """
    if conn is None:
        db = get_db_manager()
        with db.get_connection() as conn:
            _drop_tables(conn)
    else:
        _drop_tables(conn)


def _drop_tables(conn: duckdb.DuckDBPyConnection) -> None:
    """Internal function to drop tables."""
    for table_name, _ in reversed(ALL_TABLES):
        try:
            conn.execute(f"DROP TABLE IF EXISTS {table_name}")
            logger.info(f"Dropped table: {table_name}")
        except Exception as e:
            logger.error(f"Error dropping table {table_name}: {e}")
