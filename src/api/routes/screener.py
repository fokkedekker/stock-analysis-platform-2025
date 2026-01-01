"""Stock screener API routes."""

import json
from typing import Any

from fastapi import APIRouter, Query

from src.database.connection import get_db_manager

router = APIRouter()


# ============================================================================
# Raw Factor Filter Helpers
# ============================================================================

# Map factor names to table aliases and column names
FACTOR_COLUMN_MAP = {
    # From key_metrics (km)
    "pe_ratio": "km.pe_ratio",
    "pb_ratio": "km.pb_ratio",
    "price_to_sales": "km.price_to_sales",
    "price_to_free_cash_flow": "km.price_to_free_cash_flow",
    "price_to_operating_cash_flow": "km.price_to_operating_cash_flow",
    "ev_to_sales": "km.ev_to_sales",
    "ev_to_ebitda": "km.ev_to_ebitda",
    "ev_to_operating_cash_flow": "km.ev_to_operating_cash_flow",
    "ev_to_free_cash_flow": "km.ev_to_free_cash_flow",
    "roe": "km.roe",
    "roa": "km.roa",
    "return_on_tangible_assets": "km.return_on_tangible_assets",
    "gross_profit_margin": "km.gross_profit_margin",
    "operating_profit_margin": "km.operating_profit_margin",
    "net_profit_margin": "km.net_profit_margin",
    "current_ratio": "km.current_ratio",
    "quick_ratio": "km.quick_ratio",
    "cash_ratio": "km.cash_ratio",
    "debt_ratio": "km.debt_ratio",
    "debt_to_equity": "km.debt_to_equity",
    "debt_to_assets": "km.debt_to_assets",
    "net_debt_to_ebitda": "km.net_debt_to_ebitda",
    "interest_coverage": "km.interest_coverage",
    "asset_turnover": "km.asset_turnover",
    "inventory_turnover": "km.inventory_turnover",
    "receivables_turnover": "km.receivables_turnover",
    "payables_turnover": "km.payables_turnover",
    "dividend_yield": "km.dividend_yield",
    "payout_ratio": "km.payout_ratio",
    # From roic_quality_results (r) - already joined
    "roic": "r.roic",
    "roic_std_dev": "r.roic_std_dev",
    "gross_margin_std_dev": "r.gross_margin_std_dev",
    "fcf_to_net_income": "r.fcf_to_net_income",
    "reinvestment_rate": "r.reinvestment_rate",
    "fcf_yield": "r.fcf_yield",
    "ev_to_ebit": "r.ev_to_ebit",
    # From garp_peg_results (gp) - already joined
    "eps_growth_1yr": "gp.eps_growth_1yr",
    "eps_growth_3yr": "gp.eps_growth_3yr",
    "eps_growth_5yr": "gp.eps_growth_5yr",
    "eps_cagr": "gp.eps_cagr",
    "peg_ratio": "gp.peg_ratio",
    # From other tables - already joined
    "graham_score": "g.criteria_passed",
    "piotroski_score": "p.f_score",
    "magic_formula_rank": "mf.combined_rank",
    "earnings_yield": "mf.earnings_yield",
    "return_on_capital": "mf.return_on_capital",
    "z_score": "a.z_score",
}


def apply_raw_filter(stock: dict[str, Any], factor: str, operator: str, value: float | str) -> bool:
    """Check if a stock passes a single raw filter.

    Returns True if the stock passes the filter, False otherwise.
    """
    # Get the stock's value for this factor
    stock_value = stock.get(factor)

    # If value is None/null, stock fails the filter
    if stock_value is None:
        return False

    # Handle numeric comparisons
    try:
        stock_value = float(stock_value)
        filter_value = float(value)
    except (TypeError, ValueError):
        # Non-numeric comparison (e.g., string equality)
        if operator == "==":
            return str(stock_value) == str(value)
        return False

    # Apply the operator
    if operator == ">=":
        return stock_value >= filter_value
    elif operator == "<=":
        return stock_value <= filter_value
    elif operator == ">":
        return stock_value > filter_value
    elif operator == "<":
        return stock_value < filter_value
    elif operator == "==":
        return stock_value == filter_value
    elif operator == "!=":
        return stock_value != filter_value

    return False


def apply_raw_filters(stock: dict[str, Any], filters: list[dict]) -> bool:
    """Apply all raw filters to a stock. Returns True if stock passes ALL filters."""
    for f in filters:
        if not apply_raw_filter(stock, f["factor"], f["operator"], f["value"]):
            return False
    return True


def quarter_to_date(quarter_str: str | None) -> str | None:
    """Convert quarter string (e.g., '2024Q3') to end-of-quarter date string."""
    if not quarter_str:
        return None
    try:
        year = int(quarter_str[:4])
        q = int(quarter_str[5])
        if q == 1:
            return f"{year}-03-31"
        elif q == 2:
            return f"{year}-06-30"
        elif q == 3:
            return f"{year}-09-30"
        else:  # q == 4
            return f"{year}-12-31"
    except (ValueError, IndexError):
        return None


@router.get("/quarters")
async def get_available_quarters():
    """Return all quarters that have analysis data, sorted descending."""
    db = get_db_manager()

    with db.get_connection() as conn:
        result = conn.execute(
            """
            SELECT DISTINCT analysis_quarter
            FROM (
                SELECT analysis_quarter FROM graham_results
                UNION
                SELECT analysis_quarter FROM magic_formula_results
                UNION
                SELECT analysis_quarter FROM piotroski_results
                UNION
                SELECT analysis_quarter FROM altman_results
                UNION
                SELECT analysis_quarter FROM roic_quality_results
                UNION
                SELECT analysis_quarter FROM garp_peg_results
                UNION
                SELECT analysis_quarter FROM fama_french_results
                UNION
                SELECT analysis_quarter FROM net_net_results
            )
            ORDER BY analysis_quarter DESC
            """
        ).fetchall()

        quarters = [row[0] for row in result]
        return {
            "quarters": quarters,
            "latest": quarters[0] if quarters else None,
        }


@router.get("/graham")
async def screen_graham(
    mode: str = Query("strict", regex="^(strict|modern|garp|relaxed)$"),
    min_score: int = Query(5, ge=0, le=7),
    limit: int = Query(100, ge=1, le=1000),
    quarter: str | None = Query(None, description="Analysis quarter (e.g. 2024Q3). Defaults to latest."),
):
    """Screen stocks by Graham criteria."""
    db = get_db_manager()

    with db.get_connection() as conn:
        result = conn.execute(
            """
            SELECT g.*, t.name, t.sector, t.exchange
            FROM graham_results g
            JOIN tickers t ON g.symbol = t.symbol
            WHERE g.mode = ?
            AND g.criteria_passed >= ?
            AND g.analysis_quarter = COALESCE(?, (
                SELECT MAX(analysis_quarter) FROM graham_results WHERE mode = ?
            ))
            ORDER BY g.criteria_passed DESC, g.pe_ratio ASC NULLS LAST
            LIMIT ?
            """,
            (mode, min_score, quarter, mode, limit),
        ).fetchall()

        columns = [desc[0] for desc in conn.description]
        return {
            "screen": "graham",
            "mode": mode,
            "min_score": min_score,
            "count": len(result),
            "stocks": [dict(zip(columns, row)) for row in result],
        }


@router.get("/magic-formula")
async def screen_magic_formula(
    top: int = Query(50, ge=1, le=500),
    quarter: str | None = Query(None, description="Analysis quarter (e.g. 2024Q3). Defaults to latest."),
):
    """Screen stocks by Magic Formula ranking."""
    db = get_db_manager()

    with db.get_connection() as conn:
        result = conn.execute(
            """
            SELECT mf.*, t.name, t.sector, t.exchange
            FROM magic_formula_results mf
            JOIN tickers t ON mf.symbol = t.symbol
            WHERE mf.combined_rank IS NOT NULL
            AND mf.analysis_quarter = COALESCE(?, (
                SELECT MAX(analysis_quarter) FROM magic_formula_results
            ))
            ORDER BY mf.combined_rank ASC
            LIMIT ?
            """,
            (quarter, top),
        ).fetchall()

        columns = [desc[0] for desc in conn.description]
        return {
            "screen": "magic_formula",
            "top": top,
            "count": len(result),
            "stocks": [dict(zip(columns, row)) for row in result],
        }


@router.get("/piotroski")
async def screen_piotroski(
    min_score: int = Query(7, ge=0, le=9),
    limit: int = Query(100, ge=1, le=1000),
    quarter: str | None = Query(None, description="Analysis quarter (e.g. 2024Q3). Defaults to latest."),
):
    """Screen stocks by Piotroski F-Score."""
    db = get_db_manager()

    with db.get_connection() as conn:
        result = conn.execute(
            """
            SELECT p.*, t.name, t.sector, t.exchange
            FROM piotroski_results p
            JOIN tickers t ON p.symbol = t.symbol
            WHERE p.f_score >= ?
            AND p.analysis_quarter = COALESCE(?, (
                SELECT MAX(analysis_quarter) FROM piotroski_results
            ))
            ORDER BY p.f_score DESC
            LIMIT ?
            """,
            (min_score, quarter, limit),
        ).fetchall()

        columns = [desc[0] for desc in conn.description]
        return {
            "screen": "piotroski",
            "min_score": min_score,
            "count": len(result),
            "stocks": [dict(zip(columns, row)) for row in result],
        }


@router.get("/altman")
async def screen_altman(
    zone: str = Query("safe", regex="^(safe|grey|distress)$"),
    limit: int = Query(100, ge=1, le=1000),
    quarter: str | None = Query(None, description="Analysis quarter (e.g. 2024Q3). Defaults to latest."),
):
    """Screen stocks by Altman Z-Score zone."""
    db = get_db_manager()

    with db.get_connection() as conn:
        result = conn.execute(
            """
            SELECT a.*, t.name, t.sector, t.exchange
            FROM altman_results a
            JOIN tickers t ON a.symbol = t.symbol
            WHERE a.zone = ?
            AND a.analysis_quarter = COALESCE(?, (
                SELECT MAX(analysis_quarter) FROM altman_results
            ))
            ORDER BY a.z_score DESC
            LIMIT ?
            """,
            (zone, quarter, limit),
        ).fetchall()

        columns = [desc[0] for desc in conn.description]
        return {
            "screen": "altman",
            "zone": zone,
            "count": len(result),
            "stocks": [dict(zip(columns, row)) for row in result],
        }


@router.get("/roic")
async def screen_roic(
    min_roic: float = Query(0.12, ge=0, le=1),
    require_fcf: bool = Query(True),
    limit: int = Query(100, ge=1, le=1000),
    quarter: str | None = Query(None, description="Analysis quarter (e.g. 2024Q3). Defaults to latest."),
):
    """Screen stocks by ROIC/Quality criteria."""
    db = get_db_manager()

    with db.get_connection() as conn:
        query = """
            SELECT r.*, t.name, t.sector, t.exchange
            FROM roic_quality_results r
            JOIN tickers t ON r.symbol = t.symbol
            WHERE r.roic >= ?
            AND r.analysis_quarter = COALESCE(?, (
                SELECT MAX(analysis_quarter) FROM roic_quality_results
            ))
        """
        params = [min_roic, quarter]

        if require_fcf:
            query += " AND r.fcf_positive_5yr = TRUE"

        query += " ORDER BY r.roic DESC LIMIT ?"
        params.append(limit)

        result = conn.execute(query, params).fetchall()

        columns = [desc[0] for desc in conn.description]
        return {
            "screen": "roic",
            "min_roic": min_roic,
            "require_fcf": require_fcf,
            "count": len(result),
            "stocks": [dict(zip(columns, row)) for row in result],
        }


@router.get("/peg")
async def screen_peg(
    max_peg: float = Query(1.5, ge=0, le=10),
    min_growth: float = Query(0.10, ge=0, le=1),
    limit: int = Query(100, ge=1, le=1000),
    quarter: str | None = Query(None, description="Analysis quarter (e.g. 2024Q3). Defaults to latest."),
):
    """Screen stocks by PEG ratio."""
    db = get_db_manager()

    with db.get_connection() as conn:
        result = conn.execute(
            """
            SELECT g.*, t.name, t.sector, t.exchange
            FROM garp_peg_results g
            JOIN tickers t ON g.symbol = t.symbol
            WHERE g.peg_ratio <= ?
            AND g.peg_ratio > 0
            AND g.eps_cagr >= ?
            AND g.analysis_quarter = COALESCE(?, (
                SELECT MAX(analysis_quarter) FROM garp_peg_results
            ))
            ORDER BY g.peg_ratio ASC
            LIMIT ?
            """,
            (max_peg, min_growth, quarter, limit),
        ).fetchall()

        columns = [desc[0] for desc in conn.description]
        return {
            "screen": "peg",
            "max_peg": max_peg,
            "min_growth": min_growth,
            "count": len(result),
            "stocks": [dict(zip(columns, row)) for row in result],
        }


@router.get("/net-net")
async def screen_net_net(
    max_discount: float = Query(0.67, ge=0, le=1),
    limit: int = Query(100, ge=1, le=1000),
    quarter: str | None = Query(None, description="Analysis quarter (e.g. 2024Q3). Defaults to latest."),
):
    """Screen for Net-Net stocks trading below NCAV."""
    db = get_db_manager()

    with db.get_connection() as conn:
        result = conn.execute(
            """
            SELECT n.*, t.name, t.sector, t.exchange
            FROM net_net_results n
            JOIN tickers t ON n.symbol = t.symbol
            WHERE n.trading_below_ncav = TRUE
            AND n.discount_to_ncav <= ?
            AND n.analysis_quarter = COALESCE(?, (
                SELECT MAX(analysis_quarter) FROM net_net_results
            ))
            ORDER BY n.discount_to_ncav ASC
            LIMIT ?
            """,
            (max_discount, quarter, limit),
        ).fetchall()

        columns = [desc[0] for desc in conn.description]
        return {
            "screen": "net_net",
            "max_discount": max_discount,
            "count": len(result),
            "stocks": [dict(zip(columns, row)) for row in result],
        }


@router.get("/fama-french")
async def screen_fama_french(
    min_profitability: float = Query(0.0, ge=0, le=1),
    min_book_to_market: float = Query(0.0, ge=0, le=1),
    max_asset_growth: float = Query(1.0, ge=0, le=1),
    limit: int = Query(200, ge=1, le=1000),
    quarter: str | None = Query(None, description="Analysis quarter (e.g. 2024Q3). Defaults to latest."),
):
    """Screen stocks by Fama-French factor exposures.

    Factors:
    - Book-to-Market (HML): Higher is value, lower is growth
    - Profitability (RMW): Higher is more profitable
    - Asset Growth (CMA): Lower is more conservative investment
    """
    db = get_db_manager()

    with db.get_connection() as conn:
        result = conn.execute(
            """
            SELECT ff.*, t.name, t.sector, t.exchange
            FROM fama_french_results ff
            JOIN tickers t ON ff.symbol = t.symbol
            WHERE ff.analysis_quarter = COALESCE(?, (
                SELECT MAX(analysis_quarter) FROM fama_french_results
            ))
            AND (ff.profitability_percentile >= ? OR ff.profitability_percentile IS NULL)
            AND (ff.book_to_market_percentile >= ? OR ff.book_to_market_percentile IS NULL)
            AND (ff.asset_growth_percentile <= ? OR ff.asset_growth_percentile IS NULL)
            ORDER BY ff.profitability_percentile DESC NULLS LAST
            LIMIT ?
            """,
            (quarter, min_profitability, min_book_to_market, max_asset_growth, limit),
        ).fetchall()

        columns = [desc[0] for desc in conn.description]
        return {
            "screen": "fama_french",
            "min_profitability": min_profitability,
            "min_book_to_market": min_book_to_market,
            "max_asset_growth": max_asset_growth,
            "count": len(result),
            "stocks": [dict(zip(columns, row)) for row in result],
        }


@router.get("/rankings")
async def screen_rankings(
    min_systems: int = Query(1, ge=1, le=8),
    limit: int = Query(100, ge=1, le=1000),
    quarter: str | None = Query(None, description="Analysis quarter (e.g. 2024Q3). Defaults to latest."),
):
    """Get stocks ranked by number of valuation systems they pass.

    Pass criteria:
    - Graham: criteria_passed = 8 (all criteria, strict mode)
    - Magic Formula: combined_rank <= 100
    - Piotroski: f_score >= 7
    - Altman: zone = 'safe'
    - ROIC: roic_pass = TRUE
    - GARP/PEG: peg_pass = TRUE
    - Fama-French: profitability_percentile >= 0.7
    - Net-Net: trading_below_ncav = TRUE
    """
    db = get_db_manager()

    with db.get_connection() as conn:
        # Get the latest analysis quarter for each system
        result = conn.execute(
            """
            WITH latest_quarters AS (
                SELECT 'graham' as sys, COALESCE(?, MAX(analysis_quarter)) as q FROM graham_results WHERE mode = 'strict'
                UNION ALL
                SELECT 'magic_formula', COALESCE(?, MAX(analysis_quarter)) FROM magic_formula_results
                UNION ALL
                SELECT 'piotroski', COALESCE(?, MAX(analysis_quarter)) FROM piotroski_results
                UNION ALL
                SELECT 'altman', COALESCE(?, MAX(analysis_quarter)) FROM altman_results
                UNION ALL
                SELECT 'roic', COALESCE(?, MAX(analysis_quarter)) FROM roic_quality_results
                UNION ALL
                SELECT 'peg', COALESCE(?, MAX(analysis_quarter)) FROM garp_peg_results
                UNION ALL
                SELECT 'fama_french', COALESCE(?, MAX(analysis_quarter)) FROM fama_french_results
                UNION ALL
                SELECT 'net_net', COALESCE(?, MAX(analysis_quarter)) FROM net_net_results
            ),
            stock_passes AS (
                SELECT
                    t.symbol,
                    t.name,
                    t.sector,
                    t.exchange,
                    -- Graham pass (all 8 criteria)
                    CASE WHEN g.criteria_passed = 8 THEN 1 ELSE 0 END as graham_pass,
                    g.criteria_passed as graham_score,
                    -- Magic Formula pass (top 100)
                    CASE WHEN mf.combined_rank <= 100 THEN 1 ELSE 0 END as magic_formula_pass,
                    mf.combined_rank as magic_formula_rank,
                    -- Piotroski pass (F-Score >= 7)
                    CASE WHEN p.f_score >= 7 THEN 1 ELSE 0 END as piotroski_pass,
                    p.f_score as piotroski_score,
                    -- Altman pass (safe zone)
                    CASE WHEN a.zone = 'safe' THEN 1 ELSE 0 END as altman_pass,
                    a.z_score as altman_z_score,
                    a.zone as altman_zone,
                    -- ROIC pass
                    CASE WHEN r.roic_pass = TRUE THEN 1 ELSE 0 END as roic_pass,
                    r.roic as roic_value,
                    -- PEG pass
                    CASE WHEN gp.peg_pass = TRUE THEN 1 ELSE 0 END as peg_pass,
                    gp.peg_ratio as peg_ratio,
                    -- Fama-French pass (high profitability)
                    CASE WHEN ff.profitability_percentile >= 0.7 THEN 1 ELSE 0 END as fama_french_pass,
                    ff.profitability_percentile as fama_french_profitability,
                    -- Net-Net pass
                    CASE WHEN nn.trading_below_ncav = TRUE THEN 1 ELSE 0 END as net_net_pass,
                    nn.discount_to_ncav as net_net_discount
                FROM tickers t
                LEFT JOIN graham_results g ON t.symbol = g.symbol
                    AND g.mode = 'strict'
                    AND g.analysis_quarter = (SELECT q FROM latest_quarters WHERE sys = 'graham')
                LEFT JOIN magic_formula_results mf ON t.symbol = mf.symbol
                    AND mf.analysis_quarter = (SELECT q FROM latest_quarters WHERE sys = 'magic_formula')
                LEFT JOIN piotroski_results p ON t.symbol = p.symbol
                    AND p.analysis_quarter = (SELECT q FROM latest_quarters WHERE sys = 'piotroski')
                LEFT JOIN altman_results a ON t.symbol = a.symbol
                    AND a.analysis_quarter = (SELECT q FROM latest_quarters WHERE sys = 'altman')
                LEFT JOIN roic_quality_results r ON t.symbol = r.symbol
                    AND r.analysis_quarter = (SELECT q FROM latest_quarters WHERE sys = 'roic')
                LEFT JOIN garp_peg_results gp ON t.symbol = gp.symbol
                    AND gp.analysis_quarter = (SELECT q FROM latest_quarters WHERE sys = 'peg')
                LEFT JOIN fama_french_results ff ON t.symbol = ff.symbol
                    AND ff.analysis_quarter = (SELECT q FROM latest_quarters WHERE sys = 'fama_french')
                LEFT JOIN net_net_results nn ON t.symbol = nn.symbol
                    AND nn.analysis_quarter = (SELECT q FROM latest_quarters WHERE sys = 'net_net')
                WHERE t.is_active = TRUE
            )
            SELECT
                symbol,
                name,
                sector,
                exchange,
                graham_pass,
                graham_score,
                magic_formula_pass,
                magic_formula_rank,
                piotroski_pass,
                piotroski_score,
                altman_pass,
                altman_z_score,
                altman_zone,
                roic_pass,
                roic_value,
                peg_pass,
                peg_ratio,
                fama_french_pass,
                fama_french_profitability,
                net_net_pass,
                net_net_discount,
                (COALESCE(graham_pass, 0) + COALESCE(magic_formula_pass, 0) +
                 COALESCE(piotroski_pass, 0) + COALESCE(altman_pass, 0) +
                 COALESCE(roic_pass, 0) + COALESCE(peg_pass, 0) +
                 COALESCE(fama_french_pass, 0) + COALESCE(net_net_pass, 0)) as systems_passed
            FROM stock_passes
            WHERE (COALESCE(graham_pass, 0) + COALESCE(magic_formula_pass, 0) +
                   COALESCE(piotroski_pass, 0) + COALESCE(altman_pass, 0) +
                   COALESCE(roic_pass, 0) + COALESCE(peg_pass, 0) +
                   COALESCE(fama_french_pass, 0) + COALESCE(net_net_pass, 0)) >= ?
            ORDER BY systems_passed DESC, symbol ASC
            LIMIT ?
            """,
            (
                quarter, quarter, quarter, quarter,  # For latest_quarters CTE
                quarter, quarter, quarter, quarter,  # For latest_quarters CTE
                min_systems, limit,
            ),
        ).fetchall()

        columns = [desc[0] for desc in conn.description]
        return {
            "screen": "rankings",
            "min_systems": min_systems,
            "quarter": quarter,
            "count": len(result),
            "stocks": [dict(zip(columns, row)) for row in result],
        }


@router.get("/pipeline")
async def screen_pipeline(
    # Stage 1: Survival (hard gates, ON by default)
    require_altman: bool = Query(True, description="Require Altman Z-Score gate"),
    altman_zone: str = Query("safe", regex="^(safe|grey|distress)$", description="Minimum Altman zone"),
    require_piotroski: bool = Query(True, description="Require Piotroski F-Score gate"),
    piotroski_min: int = Query(5, ge=0, le=9, description="Minimum Piotroski F-Score"),
    # Stage 2: Quality (classification, filtering optional)
    quality_filter: bool = Query(False, description="Filter by quality label"),
    min_quality: str = Query("weak", regex="^(compounder|average|weak)$", description="Minimum quality"),
    # Stage 3: Valuation (at-least-N logic)
    min_valuation_lenses: int = Query(1, ge=0, le=5, description="Minimum lenses to pass"),
    strict_mode: bool = Query(False, description="Require ALL selected lenses to pass"),
    # Lens toggles
    lens_graham: bool = Query(True, description="Include Graham lens"),
    lens_net_net: bool = Query(True, description="Include Net-Net lens"),
    lens_peg: bool = Query(True, description="Include PEG lens"),
    lens_magic_formula: bool = Query(True, description="Include Magic Formula lens"),
    lens_fama_french_bm: bool = Query(False, description="Include Fama-French B/M lens"),
    # Lens thresholds
    graham_mode: str = Query("modern", regex="^(strict|modern|garp|relaxed)$"),
    graham_min: int = Query(5, ge=0, le=8),
    max_peg: float = Query(1.5, ge=0, le=10),
    mf_top_pct: int = Query(20, ge=1, le=100),
    ff_bm_top_pct: int = Query(30, ge=1, le=100),
    # Tag filter (optional)
    quality_tags_filter: str | None = Query(
        None,
        description="Comma-separated list of quality tags to REQUIRE (stock must have ALL of these)",
    ),
    excluded_quality_tags: str | None = Query(
        None,
        description="Comma-separated list of quality tags to EXCLUDE (stock must NOT have any of these)",
    ),
    # Raw factor filters (JSON array from Factor Discovery)
    raw_filters: str | None = Query(
        None,
        description="JSON array of raw factor filters, e.g. [{\"factor\": \"roic\", \"operator\": \">=\", \"value\": 0.12}]",
    ),
    # Ranking
    rank_by: str = Query(
        "magic-formula",
        regex="^(magic-formula|earnings-yield|roic|peg|graham-score|net-net-discount)$",
    ),
    limit: int = Query(200, ge=1, le=1000),
    quarter: str | None = Query(None, description="Analysis quarter (e.g. 2024Q3). Defaults to latest."),
):
    """Pipeline screener with 4-stage decision model.

    Stage 1 - Survival (hard gates): Altman Z-Score, Piotroski F-Score
    Stage 2 - Quality (classification): ROIC-based labels
    Stage 3 - Valuation (at-least-N): Graham, Net-Net, PEG, Magic Formula, Fama-French B/M
    Stage 4 - Factor Exposure: Context only (always returned, never filters)
    """
    db = get_db_manager()

    with db.get_connection() as conn:
        # Get total stock count for percentile calculations
        total_stocks_row = conn.execute(
            "SELECT COUNT(*) FROM magic_formula_results WHERE analysis_quarter = COALESCE(?, (SELECT MAX(analysis_quarter) FROM magic_formula_results))",
            (quarter,),
        ).fetchone()
        total_stocks = total_stocks_row[0] if total_stocks_row else 1

        # Magic formula top N% threshold
        mf_rank_threshold = int(total_stocks * mf_top_pct / 100)

        result = conn.execute(
            """
            WITH latest_quarters AS (
                SELECT 'graham' as sys, COALESCE(?, MAX(analysis_quarter)) as q FROM graham_results WHERE mode = ?
                UNION ALL SELECT 'magic_formula', COALESCE(?, MAX(analysis_quarter)) FROM magic_formula_results
                UNION ALL SELECT 'piotroski', COALESCE(?, MAX(analysis_quarter)) FROM piotroski_results
                UNION ALL SELECT 'altman', COALESCE(?, MAX(analysis_quarter)) FROM altman_results
                UNION ALL SELECT 'roic', COALESCE(?, MAX(analysis_quarter)) FROM roic_quality_results
                UNION ALL SELECT 'peg', COALESCE(?, MAX(analysis_quarter)) FROM garp_peg_results
                UNION ALL SELECT 'fama_french', COALESCE(?, MAX(analysis_quarter)) FROM fama_french_results
                UNION ALL SELECT 'net_net', COALESCE(?, MAX(analysis_quarter)) FROM net_net_results
            ),
            pipeline_data AS (
                SELECT
                    t.symbol,
                    t.name,
                    t.sector,
                    t.exchange,

                    -- Stage 1: Survival data
                    a.z_score as altman_z_score,
                    a.zone as altman_zone,
                    p.f_score as piotroski_score,

                    -- Stage 2: Quality data (ROIC-based classification)
                    r.roic,
                    r.free_cash_flow,
                    r.fcf_positive_5yr,
                    CASE
                        WHEN r.roic >= 0.15 AND r.fcf_positive_5yr = TRUE THEN 'compounder'
                        WHEN r.roic >= 0.08 THEN 'average'
                        ELSE 'weak'
                    END as quality_label,
                    -- Quality metrics (stability, valuation, tags)
                    r.roic_stability_tag,
                    r.gross_margin_stability_tag,
                    r.fcf_yield,
                    r.ev_to_ebit,
                    r.valuation_tag,
                    r.quality_tags,
                    -- Additional stability/growth metrics for raw filters
                    r.roic_std_dev,
                    r.gross_margin_std_dev,
                    r.fcf_to_net_income,
                    r.reinvestment_rate,

                    -- Stage 3: Valuation lens data
                    g.criteria_passed as graham_score,
                    g.pe_ratio as graham_pe,
                    g.pb_ratio as graham_pb,
                    nn.trading_below_ncav,
                    nn.discount_to_ncav as net_net_discount,
                    gp.peg_ratio,
                    gp.eps_cagr,
                    -- Additional growth metrics for raw filters
                    gp.eps_growth_1yr,
                    gp.eps_growth_3yr,
                    gp.eps_growth_5yr,
                    mf.combined_rank as magic_formula_rank,
                    mf.earnings_yield,
                    mf.return_on_capital as mf_roic,
                    ff.book_to_market_percentile,

                    -- Stage 4: Factor exposure (context only)
                    ff.profitability_percentile,
                    ff.asset_growth_percentile,
                    ff.book_to_market,
                    ff.profitability,
                    ff.asset_growth,

                    -- Raw metrics from key_metrics (for raw factor filters)
                    km.pe_ratio,
                    km.pb_ratio,
                    km.price_to_sales,
                    km.price_to_free_cash_flow,
                    km.price_to_operating_cash_flow,
                    km.ev_to_sales,
                    km.ev_to_ebitda,
                    km.ev_to_free_cash_flow,
                    km.ev_to_operating_cash_flow,
                    km.roe,
                    km.roa,
                    km.return_on_tangible_assets,
                    km.gross_profit_margin,
                    km.operating_profit_margin,
                    km.net_profit_margin,
                    km.current_ratio,
                    km.quick_ratio,
                    km.cash_ratio,
                    km.debt_ratio,
                    km.debt_to_equity,
                    km.debt_to_assets,
                    km.net_debt_to_ebitda,
                    km.interest_coverage,
                    km.asset_turnover,
                    km.inventory_turnover,
                    km.receivables_turnover,
                    km.payables_turnover,
                    km.dividend_yield,
                    km.payout_ratio

                FROM tickers t
                LEFT JOIN altman_results a ON t.symbol = a.symbol
                    AND a.analysis_quarter = (SELECT q FROM latest_quarters WHERE sys = 'altman')
                LEFT JOIN piotroski_results p ON t.symbol = p.symbol
                    AND p.analysis_quarter = (SELECT q FROM latest_quarters WHERE sys = 'piotroski')
                LEFT JOIN roic_quality_results r ON t.symbol = r.symbol
                    AND r.analysis_quarter = (SELECT q FROM latest_quarters WHERE sys = 'roic')
                LEFT JOIN graham_results g ON t.symbol = g.symbol
                    AND g.mode = ?
                    AND g.analysis_quarter = (SELECT q FROM latest_quarters WHERE sys = 'graham')
                LEFT JOIN net_net_results nn ON t.symbol = nn.symbol
                    AND nn.analysis_quarter = (SELECT q FROM latest_quarters WHERE sys = 'net_net')
                LEFT JOIN garp_peg_results gp ON t.symbol = gp.symbol
                    AND gp.analysis_quarter = (SELECT q FROM latest_quarters WHERE sys = 'peg')
                LEFT JOIN magic_formula_results mf ON t.symbol = mf.symbol
                    AND mf.analysis_quarter = (SELECT q FROM latest_quarters WHERE sys = 'magic_formula')
                LEFT JOIN fama_french_results ff ON t.symbol = ff.symbol
                    AND ff.analysis_quarter = (SELECT q FROM latest_quarters WHERE sys = 'fama_french')
                -- Key metrics: get the latest data up to the quarter end date
                LEFT JOIN (
                    SELECT DISTINCT ON (symbol)
                        symbol,
                        pe_ratio, pb_ratio, price_to_sales, price_to_free_cash_flow,
                        price_to_operating_cash_flow, ev_to_ebitda, ev_to_sales,
                        ev_to_free_cash_flow, ev_to_operating_cash_flow,
                        roe, roa, return_on_tangible_assets,
                        gross_profit_margin, operating_profit_margin, net_profit_margin,
                        current_ratio, quick_ratio, cash_ratio,
                        debt_ratio, debt_to_equity, debt_to_assets, net_debt_to_ebitda,
                        interest_coverage,
                        asset_turnover, inventory_turnover, receivables_turnover, payables_turnover,
                        dividend_yield, payout_ratio
                    FROM key_metrics
                    WHERE fiscal_date <= COALESCE(?, CURRENT_DATE)
                    ORDER BY symbol, fiscal_date DESC
                ) km ON t.symbol = km.symbol
                WHERE t.is_active = TRUE
            ),
            scored AS (
                SELECT *,
                    -- Survival gate results
                    -- distress = allow all, grey = allow safe+grey, safe = allow safe only
                    CASE
                        WHEN ? = 'distress' THEN TRUE
                        WHEN altman_zone = 'safe' THEN TRUE
                        WHEN altman_zone = 'grey' AND ? IN ('grey', 'distress') THEN TRUE
                        ELSE FALSE
                    END as altman_passed,
                    CASE WHEN piotroski_score >= ? THEN TRUE ELSE FALSE END as piotroski_passed,

                    -- Valuation lens pass/fail
                    CASE WHEN graham_score >= ? THEN TRUE ELSE FALSE END as graham_passed,
                    CASE WHEN trading_below_ncav = TRUE THEN TRUE ELSE FALSE END as net_net_passed,
                    CASE WHEN peg_ratio > 0 AND peg_ratio <= ? THEN TRUE ELSE FALSE END as peg_passed,
                    CASE WHEN magic_formula_rank <= ? THEN TRUE ELSE FALSE END as magic_formula_passed,
                    CASE WHEN book_to_market_percentile >= (1.0 - ? / 100.0) THEN TRUE ELSE FALSE END as ff_bm_passed

                FROM pipeline_data
            ),
            with_lens_count AS (
                SELECT *,
                    -- Count active lenses that passed
                    (CASE WHEN ? AND graham_passed THEN 1 ELSE 0 END +
                     CASE WHEN ? AND net_net_passed THEN 1 ELSE 0 END +
                     CASE WHEN ? AND peg_passed THEN 1 ELSE 0 END +
                     CASE WHEN ? AND magic_formula_passed THEN 1 ELSE 0 END +
                     CASE WHEN ? AND ff_bm_passed THEN 1 ELSE 0 END) as lenses_passed,
                    -- Count active lenses total
                    (CASE WHEN ? THEN 1 ELSE 0 END +
                     CASE WHEN ? THEN 1 ELSE 0 END +
                     CASE WHEN ? THEN 1 ELSE 0 END +
                     CASE WHEN ? THEN 1 ELSE 0 END +
                     CASE WHEN ? THEN 1 ELSE 0 END) as lenses_active
                FROM scored
            )
            SELECT * FROM with_lens_count
            WHERE
                -- Stage 1: Survival gates (if enabled)
                (? = FALSE OR altman_passed = TRUE)
                AND (? = FALSE OR piotroski_passed = TRUE)
                -- Stage 2: Quality filter (if enabled)
                AND (? = FALSE OR (
                    CASE ?
                        WHEN 'compounder' THEN quality_label = 'compounder'
                        WHEN 'average' THEN quality_label IN ('compounder', 'average')
                        ELSE TRUE
                    END
                ))
                -- Stage 3: Valuation (at-least-N or strict mode)
                -- If no lenses are active, skip this filter entirely
                AND (
                    lenses_active = 0
                    OR (
                        CASE WHEN ? THEN lenses_passed = lenses_active
                        ELSE lenses_passed >= ?
                        END
                    )
                )
            ORDER BY
                CASE ?
                    WHEN 'magic-formula' THEN magic_formula_rank
                    WHEN 'earnings-yield' THEN -earnings_yield
                    WHEN 'roic' THEN -roic
                    WHEN 'peg' THEN peg_ratio
                    WHEN 'graham-score' THEN -graham_score
                    WHEN 'net-net-discount' THEN net_net_discount
                END NULLS LAST
            LIMIT ?
            """,
            (
                # For latest_quarters CTE (8 quarter params + 1 graham_mode)
                quarter, graham_mode,  # graham
                quarter,  # magic_formula
                quarter,  # piotroski
                quarter,  # altman
                quarter,  # roic
                quarter,  # peg
                quarter,  # fama_french
                quarter,  # net_net
                graham_mode,  # For graham join
                quarter_to_date(quarter),  # For key_metrics join (fiscal_date <= quarter_end)
                altman_zone,  # Altman zone threshold (for distress check)
                altman_zone,  # Altman zone threshold (for grey check)
                piotroski_min,  # Piotroski threshold
                graham_min,  # Graham min score
                max_peg,  # PEG max
                mf_rank_threshold,  # Magic Formula rank threshold
                ff_bm_top_pct,  # Fama-French B/M percentile
                # Lens toggles for counting passed
                lens_graham,
                lens_net_net,
                lens_peg,
                lens_magic_formula,
                lens_fama_french_bm,
                # Lens toggles for counting active
                lens_graham,
                lens_net_net,
                lens_peg,
                lens_magic_formula,
                lens_fama_french_bm,
                # Stage 1 filters
                require_altman,
                require_piotroski,
                # Stage 2 filter
                quality_filter,
                min_quality,
                # Stage 3 filter
                strict_mode,
                min_valuation_lenses,
                # Ranking
                rank_by,
                limit,
            ),
        ).fetchall()

        columns = [desc[0] for desc in conn.description]

        # Parse required tags filter if provided
        required_tags: set[str] = set()
        if quality_tags_filter:
            required_tags = set(t.strip() for t in quality_tags_filter.split(",") if t.strip())

        # Parse excluded tags filter if provided
        excluded_tag_set: set[str] = set()
        if excluded_quality_tags:
            excluded_tag_set = set(t.strip() for t in excluded_quality_tags.split(",") if t.strip())

        # Parse raw factor filters if provided
        parsed_raw_filters: list[dict] = []
        if raw_filters:
            try:
                parsed_raw_filters = json.loads(raw_filters)
            except json.JSONDecodeError:
                pass  # Invalid JSON, ignore raw filters

        # All possible quality tags
        ALL_QUALITY_TAGS = {
            "Durable Compounder", "Cash Machine", "Deep Value", "Heavy Reinvestor",
            "Volatile Returns", "Earnings Quality Concern", "Premium Priced", "Weak Moat Signal"
        }

        stocks = []
        for row in result:
            stock = dict(zip(columns, row))

            # Parse quality_tags from JSON string
            stock_tags = set()
            if stock.get("quality_tags"):
                try:
                    stock_tags = set(json.loads(stock["quality_tags"]))
                except (json.JSONDecodeError, TypeError):
                    stock_tags = set()

            # Apply tag filters if specified
            # Required tags: stock must have ALL of these (if any specified)
            if required_tags:
                if not required_tags.issubset(stock_tags):
                    continue
            # Excluded tags: stock must NOT have any of these (if any specified)
            if excluded_tag_set:
                if stock_tags & excluded_tag_set:
                    continue

            # Apply raw factor filters if specified
            if parsed_raw_filters and not apply_raw_filters(stock, parsed_raw_filters):
                continue

            # Build valuation lenses passed list
            lenses = []
            if lens_graham and stock.get("graham_passed"):
                lenses.append("graham")
            if lens_net_net and stock.get("net_net_passed"):
                lenses.append("net-net")
            if lens_peg and stock.get("peg_passed"):
                lenses.append("peg")
            if lens_magic_formula and stock.get("magic_formula_passed"):
                lenses.append("magic-formula")
            if lens_fama_french_bm and stock.get("ff_bm_passed"):
                lenses.append("fama-french-bm")
            stock["valuation_lenses_passed"] = lenses
            stocks.append(stock)

        return {
            "screen": "pipeline",
            "config": {
                "survival": {
                    "require_altman": require_altman,
                    "altman_zone": altman_zone,
                    "require_piotroski": require_piotroski,
                    "piotroski_min": piotroski_min,
                },
                "quality": {
                    "filter_enabled": quality_filter,
                    "min_quality": min_quality,
                },
                "valuation": {
                    "min_lenses": min_valuation_lenses,
                    "strict_mode": strict_mode,
                    "lenses": {
                        "graham": {"enabled": lens_graham, "mode": graham_mode, "min_score": graham_min},
                        "net_net": {"enabled": lens_net_net},
                        "peg": {"enabled": lens_peg, "max_peg": max_peg},
                        "magic_formula": {"enabled": lens_magic_formula, "top_pct": mf_top_pct},
                        "fama_french_bm": {"enabled": lens_fama_french_bm, "top_pct": ff_bm_top_pct},
                    },
                },
                "raw_filters": parsed_raw_filters,
                "rank_by": rank_by,
            },
            "count": len(stocks),
            "stocks": stocks,
        }


@router.get("/combined")
async def screen_combined(
    graham_min: int = Query(None, ge=0, le=7),
    piotroski_min: int = Query(None, ge=0, le=9),
    altman_zone: str = Query(None, regex="^(safe|grey|distress)$"),
    min_roic: float = Query(None, ge=0, le=1),
    max_peg: float = Query(None, ge=0, le=10),
    limit: int = Query(100, ge=1, le=1000),
    quarter: str | None = Query(None, description="Analysis quarter (e.g. 2024Q3). Defaults to latest."),
):
    """Combined screener with multiple criteria."""
    db = get_db_manager()

    with db.get_connection() as conn:
        # Start with tickers
        query = """
            SELECT t.symbol, t.name, t.sector, t.exchange
        """
        joins = []
        conditions = ["t.is_active = TRUE"]
        params = []

        if graham_min is not None:
            query += ", g.criteria_passed as graham_score"
            joins.append(
                "LEFT JOIN graham_results g ON t.symbol = g.symbol AND g.mode = 'strict' "
                "AND g.analysis_quarter = COALESCE(?, (SELECT MAX(analysis_quarter) FROM graham_results WHERE mode = 'strict'))"
            )
            params.append(quarter)
            conditions.append("g.criteria_passed >= ?")
            params.append(graham_min)

        if piotroski_min is not None:
            query += ", p.f_score as piotroski_score"
            joins.append(
                "LEFT JOIN piotroski_results p ON t.symbol = p.symbol "
                "AND p.analysis_quarter = COALESCE(?, (SELECT MAX(analysis_quarter) FROM piotroski_results))"
            )
            params.append(quarter)
            conditions.append("p.f_score >= ?")
            params.append(piotroski_min)

        if altman_zone is not None:
            query += ", a.z_score, a.zone as altman_zone"
            joins.append(
                "LEFT JOIN altman_results a ON t.symbol = a.symbol "
                "AND a.analysis_quarter = COALESCE(?, (SELECT MAX(analysis_quarter) FROM altman_results))"
            )
            params.append(quarter)
            conditions.append("a.zone = ?")
            params.append(altman_zone)

        if min_roic is not None:
            query += ", r.roic"
            joins.append(
                "LEFT JOIN roic_quality_results r ON t.symbol = r.symbol "
                "AND r.analysis_quarter = COALESCE(?, (SELECT MAX(analysis_quarter) FROM roic_quality_results))"
            )
            params.append(quarter)
            conditions.append("r.roic >= ?")
            params.append(min_roic)

        if max_peg is not None:
            query += ", gp.peg_ratio"
            joins.append(
                "LEFT JOIN garp_peg_results gp ON t.symbol = gp.symbol "
                "AND gp.analysis_quarter = COALESCE(?, (SELECT MAX(analysis_quarter) FROM garp_peg_results))"
            )
            params.append(quarter)
            conditions.append("gp.peg_ratio <= ? AND gp.peg_ratio > 0")
            params.append(max_peg)

        query += f" FROM tickers t {' '.join(joins)}"
        query += f" WHERE {' AND '.join(conditions)}"
        query += f" ORDER BY t.symbol LIMIT ?"
        params.append(limit)

        result = conn.execute(query, params).fetchall()
        columns = [desc[0] for desc in conn.description]

        return {
            "screen": "combined",
            "filters": {
                "graham_min": graham_min,
                "piotroski_min": piotroski_min,
                "altman_zone": altman_zone,
                "min_roic": min_roic,
                "max_peg": max_peg,
            },
            "count": len(result),
            "stocks": [dict(zip(columns, row)) for row in result],
        }


# ============================================================================
# Live ML Model Scoring Functions (used by signal generation)
# ============================================================================


def _get_stock_data_for_quarter(conn, quarter: str) -> list[dict]:
    """Load all stock data with features for a specific quarter.

    Returns list of dicts with symbol and all feature values.
    """
    result = conn.execute(
        """
        SELECT
            t.symbol,
            t.name,

            -- Pre-computed Scores (7)
            p.f_score as piotroski_score,
            g.criteria_passed as graham_score,
            a.z_score as altman_z_score,
            r.roic,
            gp.peg_ratio,
            mf.combined_rank as magic_formula_rank,
            ff.book_to_market_percentile,

            -- Raw Valuation Metrics (9)
            km.pe_ratio,
            km.pb_ratio,
            km.price_to_sales,
            km.price_to_free_cash_flow,
            km.price_to_operating_cash_flow,
            km.ev_to_ebitda,
            km.ev_to_sales,
            km.ev_to_free_cash_flow,
            km.ev_to_operating_cash_flow,

            -- Raw Profitability Metrics (6)
            km.roe,
            km.roa,
            km.return_on_tangible_assets,
            km.gross_profit_margin,
            km.operating_profit_margin,
            km.net_profit_margin,

            -- Raw Liquidity Metrics (3)
            km.current_ratio,
            km.quick_ratio,
            km.cash_ratio,

            -- Raw Leverage Metrics (5)
            km.debt_ratio,
            km.debt_to_equity,
            km.debt_to_assets,
            km.net_debt_to_ebitda,
            km.interest_coverage,

            -- Raw Efficiency Metrics (4)
            km.asset_turnover,
            km.inventory_turnover,
            km.receivables_turnover,
            km.payables_turnover,

            -- Raw Dividend Metrics (2)
            km.dividend_yield,
            km.payout_ratio,

            -- Stability Metrics (5)
            r.roic_std_dev,
            r.gross_margin_std_dev,
            r.fcf_to_net_income,
            r.reinvestment_rate,
            r.fcf_yield,

            -- Growth Metrics (4)
            gp.eps_growth_1yr,
            gp.eps_growth_3yr,
            gp.eps_growth_5yr,
            gp.eps_cagr,

            -- Regime Factors (1)
            mi.rate_momentum,

            -- Fama-French Percentiles (2)
            ff.profitability_percentile,
            ff.asset_growth_percentile,

            -- Additional derived metrics (1)
            mf.earnings_yield,

            -- Boolean factors - need to derive from quality_tags
            r.quality_tags,
            nn.trading_below_ncav,
            r.fcf_positive_5yr

        FROM tickers t
        LEFT JOIN (
            SELECT DISTINCT ON (symbol) *
            FROM key_metrics
            WHERE fiscal_date <= (
                SELECT MAX(fiscal_date) FROM key_metrics
            )
            ORDER BY symbol, fiscal_date DESC
        ) km ON t.symbol = km.symbol
        LEFT JOIN roic_quality_results r ON t.symbol = r.symbol AND r.analysis_quarter = ?
        LEFT JOIN piotroski_results p ON t.symbol = p.symbol AND p.analysis_quarter = ?
        LEFT JOIN graham_results g ON t.symbol = g.symbol AND g.analysis_quarter = ? AND g.mode = 'modern'
        LEFT JOIN magic_formula_results mf ON t.symbol = mf.symbol AND mf.analysis_quarter = ?
        LEFT JOIN garp_peg_results gp ON t.symbol = gp.symbol AND gp.analysis_quarter = ?
        LEFT JOIN altman_results a ON t.symbol = a.symbol AND a.analysis_quarter = ?
        LEFT JOIN fama_french_results ff ON t.symbol = ff.symbol AND ff.analysis_quarter = ?
        LEFT JOIN net_net_results nn ON t.symbol = nn.symbol AND nn.analysis_quarter = ?
        LEFT JOIN (
            SELECT DISTINCT ON (indicator_date)
                indicator_date,
                (treasury_10y - LAG(treasury_10y, 4) OVER (ORDER BY indicator_date)) as rate_momentum
            FROM macro_indicators
            ORDER BY indicator_date DESC
            LIMIT 1
        ) mi ON TRUE
        WHERE t.is_active = TRUE
        """,
        (quarter, quarter, quarter, quarter, quarter, quarter, quarter, quarter),
    ).fetchall()

    columns = [desc[0] for desc in conn.description]
    stocks = [dict(zip(columns, row)) for row in result]

    # Parse boolean tags from quality_tags JSON
    QUALITY_TAG_MAP = {
        "Durable Compounder": "has_durable_compounder",
        "Cash Machine": "has_cash_machine",
        "Deep Value": "has_deep_value",
        "Heavy Reinvestor": "has_heavy_reinvestor",
        "Premium Priced": "has_premium_priced",
        "Volatile Returns": "has_volatile_returns",
        "Weak Moat Signal": "has_weak_moat_signal",
        "Earnings Quality Concern": "has_earnings_quality_concern",
    }

    for stock in stocks:
        quality_tags_json = stock.get("quality_tags")
        stock_tags = set()
        if quality_tags_json:
            try:
                stock_tags = set(json.loads(quality_tags_json))
            except (json.JSONDecodeError, TypeError):
                pass

        for tag_name, feature_name in QUALITY_TAG_MAP.items():
            stock[feature_name] = 1.0 if tag_name in stock_tags else 0.0

        stock["trading_below_ncav"] = 1.0 if stock.get("trading_below_ncav") else 0.0
        stock["fcf_positive_5yr"] = 1.0 if stock.get("fcf_positive_5yr") else 0.0

    return stocks


def calculate_live_rankings_elastic_net(
    conn, run_id: str, quarter: str
) -> dict[str, dict]:
    """Calculate live Elastic Net model scores and ranks for all stocks in a quarter.

    Returns: {symbol: {"score": float, "rank": int}}
    """
    from src.ml_models import load_elastic_net_result

    # Load model coefficients
    model_data = load_elastic_net_result(run_id)
    coefficients = {c["feature_name"]: c["coefficient"] for c in model_data["coefficients"]}
    features_needed = [f for f, c in coefficients.items() if abs(c) > 1e-8]

    if not features_needed:
        return {}

    # Get stock data for this quarter
    stocks = _get_stock_data_for_quarter(conn, quarter)
    if not stocks:
        return {}

    # Collect feature values for all stocks
    stock_feature_vals: list[dict[str, float | None]] = []
    for stock in stocks:
        stock_vals = {}
        for feature in features_needed:
            val = stock.get(feature)
            if val is not None:
                try:
                    stock_vals[feature] = float(val)
                except (TypeError, ValueError):
                    stock_vals[feature] = None
            else:
                stock_vals[feature] = None
        stock_feature_vals.append(stock_vals)

    # Calculate z-scores for each feature across all stocks
    feature_values: dict[str, list[float]] = {f: [] for f in features_needed}
    for stock_vals in stock_feature_vals:
        for feature in features_needed:
            val = stock_vals.get(feature)
            if val is not None:
                feature_values[feature].append(val)

    # Calculate mean and std for each feature
    feature_stats: dict[str, tuple[float, float]] = {}
    for feature in features_needed:
        vals = feature_values[feature]
        if len(vals) > 0:
            mean = sum(vals) / len(vals)
            variance = sum((v - mean) ** 2 for v in vals) / len(vals) if len(vals) > 1 else 1.0
            std = variance ** 0.5 if variance > 0 else 1.0
            feature_stats[feature] = (mean, std)
        else:
            feature_stats[feature] = (0.0, 1.0)

    # Score each stock
    scored_stocks = []
    for i, stock in enumerate(stocks):
        stock_vals = stock_feature_vals[i]
        score = 0.0
        features_used = 0

        for feature in features_needed:
            val = stock_vals.get(feature)
            if val is not None:
                mean, std = feature_stats[feature]
                z_score = (val - mean) / std if std > 0 else 0.0
                coef = coefficients.get(feature, 0.0)
                score += z_score * coef
                features_used += 1

        # Only include stocks with enough features
        if features_used >= len(features_needed) * 0.5:
            scored_stocks.append({
                "symbol": stock["symbol"],
                "score": score,
            })

    # Sort by score descending and assign ranks
    scored_stocks.sort(key=lambda x: x["score"], reverse=True)

    result = {}
    for i, s in enumerate(scored_stocks):
        result[s["symbol"]] = {"score": s["score"], "rank": i + 1}

    return result


def calculate_live_rankings_gam(
    conn, run_id: str, quarter: str
) -> dict[str, dict]:
    """Calculate live GAM model scores and ranks for all stocks in a quarter.

    Returns: {symbol: {"score": float, "rank": int}}
    """
    import numpy as np
    from src.ml_models import load_gam_result

    # Load partial dependences
    model_data = load_gam_result(run_id)
    partial_deps = model_data["partial_dependences"]
    features_needed = [pd["feature_name"] for pd in partial_deps]

    if not features_needed:
        return {}

    pd_lookup = {pd["feature_name"]: pd for pd in partial_deps}

    # Get stock data for this quarter
    stocks = _get_stock_data_for_quarter(conn, quarter)
    if not stocks:
        return {}

    # Score each stock using partial dependence curves
    scored_stocks = []
    for stock in stocks:
        score = 0.0
        features_used = 0

        for feature in features_needed:
            val = stock.get(feature)
            if val is not None:
                try:
                    val = float(val)
                    pd_data = pd_lookup.get(feature)
                    if pd_data:
                        x_values = pd_data["x_values"]
                        y_values = pd_data["y_values"]
                        effect = float(np.interp(val, x_values, y_values))
                        score += effect
                        features_used += 1
                except (TypeError, ValueError):
                    pass

        # Only include stocks with enough features
        if features_used >= len(features_needed) * 0.5:
            scored_stocks.append({
                "symbol": stock["symbol"],
                "score": score,
            })

    # Sort by score descending and assign ranks
    scored_stocks.sort(key=lambda x: x["score"], reverse=True)

    result = {}
    for i, s in enumerate(scored_stocks):
        result[s["symbol"]] = {"score": s["score"], "rank": i + 1}

    return result


def get_quarters_with_analysis_data(conn) -> list[str]:
    """Get all quarters that have analysis data, sorted ascending."""
    result = conn.execute(
        "SELECT DISTINCT analysis_quarter FROM roic_quality_results ORDER BY analysis_quarter"
    ).fetchall()
    return [row[0] for row in result]


@router.get("/apply-model/{run_id}")
async def apply_model_to_current(
    run_id: str,
    top_percentile: int = Query(20, ge=1, le=100),
    min_score: float = Query(None, description="Minimum model score (z-score) to include"),
    quarter: str | None = Query(None, description="Analysis quarter. Defaults to latest."),
    limit: int = Query(200, ge=1, le=1000),
):
    """Apply a saved ML model (Elastic Net or GAM) to current stocks and return ranked predictions.

    This endpoint allows you to use a trained ML model as a filter in the Pipeline.
    - For Elastic Net: Stocks are scored using the model's coefficients applied to z-scored features.
    - For GAM: Stocks are scored by summing interpolated partial dependence effects for each feature.

    Args:
        run_id: The ML model run ID to apply
        top_percentile: Only return stocks in top N% by model score
        min_score: Alternatively, filter by minimum model score (z-score)
        quarter: Analysis quarter to use for stock data
        limit: Maximum stocks to return
    """
    from src.ml_models import load_elastic_net_result, load_gam_result, load_lightgbm_result
    import numpy as np

    db = get_db_manager()

    # First, determine the model type from the database
    with db.get_connection() as conn:
        model_row = conn.execute(
            "SELECT model_type FROM ml_model_runs WHERE id = ?",
            (run_id,)
        ).fetchone()

    if not model_row:
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail=f"Model run {run_id} not found")

    model_type = model_row[0]

    # Load model data based on type
    if model_type == "gam":
        try:
            model_data = load_gam_result(run_id)
        except ValueError:
            from fastapi import HTTPException
            raise HTTPException(status_code=404, detail=f"GAM model run {run_id} not found")

        partial_deps = model_data["partial_dependences"]
        features_needed = [pd["feature_name"] for pd in partial_deps]

        if not features_needed:
            return {
                "screen": "ml_model",
                "run_id": run_id,
                "model_type": "gam",
                "count": 0,
                "stocks": [],
                "error": "Model has no partial dependence data",
            }

        # Build lookup for partial dependence curves
        pd_lookup = {pd["feature_name"]: pd for pd in partial_deps}
    elif model_type == "lightgbm":
        # LightGBM: Use feature importance as weights for z-scored features
        try:
            model_data = load_lightgbm_result(run_id)
        except ValueError:
            from fastapi import HTTPException
            raise HTTPException(status_code=404, detail=f"LightGBM model run {run_id} not found")

        feature_importances = model_data["feature_importances"]
        # Use importance_gain as the weight (similar to coefficients)
        coefficients = {fi["feature_name"]: fi["importance_gain"] for fi in feature_importances}
        features_needed = [f for f, c in coefficients.items() if c > 0]

        if not features_needed:
            return {
                "screen": "ml_model",
                "run_id": run_id,
                "model_type": "lightgbm",
                "count": 0,
                "stocks": [],
                "error": "Model has no non-zero feature importances",
            }
    else:
        # Elastic Net
        try:
            model_data = load_elastic_net_result(run_id)
        except ValueError:
            from fastapi import HTTPException
            raise HTTPException(status_code=404, detail=f"Elastic Net model run {run_id} not found")

        coefficients = {c["feature_name"]: c["coefficient"] for c in model_data["coefficients"]}
        features_needed = [f for f, c in coefficients.items() if abs(c) > 1e-8]

        if not features_needed:
            return {
                "screen": "ml_model",
                "run_id": run_id,
                "model_type": "elastic_net",
                "count": 0,
                "stocks": [],
                "error": "Model has no non-zero coefficients",
            }

    with db.get_connection() as conn:
        # Get the latest quarter if not specified
        if not quarter:
            quarter_row = conn.execute(
                "SELECT MAX(analysis_quarter) FROM roic_quality_results"
            ).fetchone()
            quarter = quarter_row[0] if quarter_row else None

        if not quarter:
            return {
                "screen": "ml_model",
                "run_id": run_id,
                "count": 0,
                "stocks": [],
                "error": "No analysis data available",
            }

        # Get current stock data with all features matching ELASTIC_NET_FEATURES
        result = conn.execute(
            """
            SELECT
                t.symbol,
                t.name,
                t.sector,
                t.exchange,

                -- Pre-computed Scores (7)
                p.f_score as piotroski_score,
                g.criteria_passed as graham_score,
                a.z_score as altman_z_score,
                r.roic,
                gp.peg_ratio,
                mf.combined_rank as magic_formula_rank,
                ff.book_to_market_percentile,

                -- Raw Valuation Metrics (9)
                km.pe_ratio,
                km.pb_ratio,
                km.price_to_sales,
                km.price_to_free_cash_flow,
                km.price_to_operating_cash_flow,
                km.ev_to_ebitda,
                km.ev_to_sales,
                km.ev_to_free_cash_flow,
                km.ev_to_operating_cash_flow,

                -- Raw Profitability Metrics (6)
                km.roe,
                km.roa,
                km.return_on_tangible_assets,
                km.gross_profit_margin,
                km.operating_profit_margin,
                km.net_profit_margin,

                -- Raw Liquidity Metrics (3)
                km.current_ratio,
                km.quick_ratio,
                km.cash_ratio,

                -- Raw Leverage Metrics (5)
                km.debt_ratio,
                km.debt_to_equity,
                km.debt_to_assets,
                km.net_debt_to_ebitda,
                km.interest_coverage,

                -- Raw Efficiency Metrics (4)
                km.asset_turnover,
                km.inventory_turnover,
                km.receivables_turnover,
                km.payables_turnover,

                -- Raw Dividend Metrics (2)
                km.dividend_yield,
                km.payout_ratio,

                -- Stability Metrics (5)
                r.roic_std_dev,
                r.gross_margin_std_dev,
                r.fcf_to_net_income,
                r.reinvestment_rate,
                r.fcf_yield,

                -- Growth Metrics (4)
                gp.eps_growth_1yr,
                gp.eps_growth_3yr,
                gp.eps_growth_5yr,
                gp.eps_cagr,

                -- Regime Factors (1)
                mi.rate_momentum,

                -- Fama-French Percentiles (2)
                ff.profitability_percentile,
                ff.asset_growth_percentile,

                -- Additional derived metrics (1)
                mf.earnings_yield,

                -- Boolean factors (10) - need to derive from quality_tags
                r.quality_tags,
                nn.trading_below_ncav,
                r.fcf_positive_5yr

            FROM tickers t
            LEFT JOIN (
                SELECT DISTINCT ON (symbol) *
                FROM key_metrics
                WHERE fiscal_date <= (
                    SELECT MAX(fiscal_date) FROM key_metrics
                )
                ORDER BY symbol, fiscal_date DESC
            ) km ON t.symbol = km.symbol
            LEFT JOIN roic_quality_results r ON t.symbol = r.symbol AND r.analysis_quarter = ?
            LEFT JOIN piotroski_results p ON t.symbol = p.symbol AND p.analysis_quarter = ?
            LEFT JOIN graham_results g ON t.symbol = g.symbol AND g.analysis_quarter = ? AND g.mode = 'modern'
            LEFT JOIN magic_formula_results mf ON t.symbol = mf.symbol AND mf.analysis_quarter = ?
            LEFT JOIN garp_peg_results gp ON t.symbol = gp.symbol AND gp.analysis_quarter = ?
            LEFT JOIN altman_results a ON t.symbol = a.symbol AND a.analysis_quarter = ?
            LEFT JOIN fama_french_results ff ON t.symbol = ff.symbol AND ff.analysis_quarter = ?
            LEFT JOIN net_net_results nn ON t.symbol = nn.symbol AND nn.analysis_quarter = ?
            LEFT JOIN (
                SELECT DISTINCT ON (indicator_date)
                    indicator_date,
                    (treasury_10y - LAG(treasury_10y, 4) OVER (ORDER BY indicator_date)) as rate_momentum
                FROM macro_indicators
                ORDER BY indicator_date DESC
                LIMIT 1
            ) mi ON TRUE
            WHERE t.is_active = TRUE
            """,
            (quarter, quarter, quarter, quarter, quarter, quarter, quarter, quarter),
        ).fetchall()

        columns = [desc[0] for desc in conn.description]
        stocks = [dict(zip(columns, row)) for row in result]

    # Parse boolean tags from quality_tags JSON
    QUALITY_TAG_MAP = {
        "Durable Compounder": "has_durable_compounder",
        "Cash Machine": "has_cash_machine",
        "Deep Value": "has_deep_value",
        "Heavy Reinvestor": "has_heavy_reinvestor",
        "Premium Priced": "has_premium_priced",
        "Volatile Returns": "has_volatile_returns",
        "Weak Moat Signal": "has_weak_moat_signal",
        "Earnings Quality Concern": "has_earnings_quality_concern",
    }

    for stock in stocks:
        # Parse quality_tags JSON and set boolean flags
        quality_tags_json = stock.get("quality_tags")
        stock_tags = set()
        if quality_tags_json:
            try:
                stock_tags = set(json.loads(quality_tags_json))
            except (json.JSONDecodeError, TypeError):
                pass

        for tag_name, feature_name in QUALITY_TAG_MAP.items():
            stock[feature_name] = 1.0 if tag_name in stock_tags else 0.0

        # Convert trading_below_ncav and fcf_positive_5yr to 0/1
        stock["trading_below_ncav"] = 1.0 if stock.get("trading_below_ncav") else 0.0
        stock["fcf_positive_5yr"] = 1.0 if stock.get("fcf_positive_5yr") else 0.0

    # Collect feature values for all stocks
    stock_feature_vals: list[dict[str, float | None]] = []

    for stock in stocks:
        stock_vals = {}
        for feature in features_needed:
            val = stock.get(feature)
            if val is not None:
                try:
                    stock_vals[feature] = float(val)
                except (TypeError, ValueError):
                    stock_vals[feature] = None
            else:
                stock_vals[feature] = None
        stock_feature_vals.append(stock_vals)

    # Scoring differs by model type
    if model_type == "gam":
        # GAM: Score by interpolating partial dependence curves
        scored_stocks = []
        for i, stock in enumerate(stocks):
            stock_vals = stock_feature_vals[i]
            score = 0.0
            features_used = 0

            for feature in features_needed:
                val = stock_vals.get(feature)
                if val is not None:
                    pd_data = pd_lookup.get(feature)
                    if pd_data:
                        x_values = pd_data["x_values"]
                        y_values = pd_data["y_values"]
                        # Interpolate the effect for this feature value
                        effect = float(np.interp(val, x_values, y_values))
                        score += effect
                        features_used += 1

            # Only include stocks with enough features
            if features_used >= len(features_needed) * 0.5:
                scored_stocks.append({
                    **stock,
                    "ml_score": score,
                    "features_used": features_used,
                })
    elif model_type in ("elastic_net", "lightgbm"):
        # Elastic Net & LightGBM: Score using z-scored features * coefficients/importance
        # Calculate z-scores for each feature across all stocks
        feature_values: dict[str, list[float]] = {f: [] for f in features_needed}

        for stock_vals in stock_feature_vals:
            for feature in features_needed:
                val = stock_vals.get(feature)
                if val is not None:
                    feature_values[feature].append(val)

        # Calculate mean and std for each feature
        feature_stats: dict[str, tuple[float, float]] = {}
        for feature in features_needed:
            vals = feature_values[feature]
            if len(vals) > 0:
                mean = sum(vals) / len(vals)
                variance = sum((v - mean) ** 2 for v in vals) / len(vals) if len(vals) > 1 else 1.0
                std = variance ** 0.5 if variance > 0 else 1.0
                feature_stats[feature] = (mean, std)
            else:
                feature_stats[feature] = (0.0, 1.0)

        # Score each stock
        scored_stocks = []
        for i, stock in enumerate(stocks):
            stock_vals = stock_feature_vals[i]
            score = 0.0
            features_used = 0

            for feature in features_needed:
                val = stock_vals.get(feature)
                if val is not None:
                    mean, std = feature_stats[feature]
                    z_score = (val - mean) / std if std > 0 else 0.0
                    coef = coefficients.get(feature, 0.0)
                    score += z_score * coef
                    features_used += 1

            # Only include stocks with enough features
            if features_used >= len(features_needed) * 0.5:
                scored_stocks.append({
                    **stock,
                    "ml_score": score,
                    "features_used": features_used,
                })

    # Sort by score descending
    scored_stocks.sort(key=lambda x: x["ml_score"], reverse=True)

    # Apply filters
    if min_score is not None:
        scored_stocks = [s for s in scored_stocks if s["ml_score"] >= min_score]
    elif top_percentile < 100:
        cutoff_idx = max(1, int(len(scored_stocks) * top_percentile / 100))
        scored_stocks = scored_stocks[:cutoff_idx]

    # Add rank
    for i, stock in enumerate(scored_stocks):
        stock["ml_rank"] = i + 1

    # Apply limit
    scored_stocks = scored_stocks[:limit]

    return {
        "screen": "ml_model",
        "run_id": run_id,
        "model_type": model_type,
        "quarter": quarter,
        "config": {
            "top_percentile": top_percentile,
            "min_score": min_score,
            "features_used": features_needed,
            "n_features": len(features_needed),
        },
        "count": len(scored_stocks),
        "total_scored": len(scored_stocks),
        "stocks": scored_stocks,
    }


@router.get("/ml-models")
async def list_ml_models(limit: int = Query(20, ge=1, le=100)):
    """List available ML models that can be applied to stocks."""
    db = get_db_manager()

    with db.get_connection() as conn:
        result = conn.execute(
            """
            SELECT id, model_type, created_at, status, holding_period,
                   train_ic, test_ic, n_features_selected
            FROM ml_model_runs
            WHERE status = 'completed'
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()

    return {
        "models": [
            {
                "run_id": row[0],
                "model_type": row[1],
                "created_at": str(row[2]) if row[2] else None,
                "status": row[3],
                "holding_period": row[4],
                "train_ic": row[5],
                "test_ic": row[6],
                "n_features_selected": row[7] or 0,
            }
            for row in result
        ],
        "total": len(result),
    }
