"""Stock screener API routes."""

from fastapi import APIRouter, Query

from src.database.connection import get_db_manager

router = APIRouter()


@router.get("/graham")
async def screen_graham(
    mode: str = Query("strict", regex="^(strict|modern|garp|relaxed)$"),
    min_score: int = Query(5, ge=0, le=7),
    limit: int = Query(100, ge=1, le=1000),
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
            AND g.analysis_quarter = (
                SELECT MAX(analysis_quarter) FROM graham_results WHERE mode = ?
            )
            ORDER BY g.criteria_passed DESC, g.pe_ratio ASC NULLS LAST
            LIMIT ?
            """,
            (mode, min_score, mode, limit),
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
            AND mf.analysis_quarter = (
                SELECT MAX(analysis_quarter) FROM magic_formula_results
            )
            ORDER BY mf.combined_rank ASC
            LIMIT ?
            """,
            (top,),
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
            AND p.analysis_quarter = (
                SELECT MAX(analysis_quarter) FROM piotroski_results
            )
            ORDER BY p.f_score DESC
            LIMIT ?
            """,
            (min_score, limit),
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
            AND a.analysis_quarter = (
                SELECT MAX(analysis_quarter) FROM altman_results
            )
            ORDER BY a.z_score DESC
            LIMIT ?
            """,
            (zone, limit),
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
):
    """Screen stocks by ROIC/Quality criteria."""
    db = get_db_manager()

    with db.get_connection() as conn:
        query = """
            SELECT r.*, t.name, t.sector, t.exchange
            FROM roic_quality_results r
            JOIN tickers t ON r.symbol = t.symbol
            WHERE r.roic >= ?
            AND r.analysis_quarter = (
                SELECT MAX(analysis_quarter) FROM roic_quality_results
            )
        """
        params = [min_roic]

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
            AND g.analysis_quarter = (
                SELECT MAX(analysis_quarter) FROM garp_peg_results
            )
            ORDER BY g.peg_ratio ASC
            LIMIT ?
            """,
            (max_peg, min_growth, limit),
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
            AND n.analysis_quarter = (
                SELECT MAX(analysis_quarter) FROM net_net_results
            )
            ORDER BY n.discount_to_ncav ASC
            LIMIT ?
            """,
            (max_discount, limit),
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
            WHERE ff.analysis_quarter = (
                SELECT MAX(analysis_quarter) FROM fama_french_results
            )
            AND (ff.profitability_percentile >= ? OR ff.profitability_percentile IS NULL)
            AND (ff.book_to_market_percentile >= ? OR ff.book_to_market_percentile IS NULL)
            AND (ff.asset_growth_percentile <= ? OR ff.asset_growth_percentile IS NULL)
            ORDER BY ff.profitability_percentile DESC NULLS LAST
            LIMIT ?
            """,
            (min_profitability, min_book_to_market, max_asset_growth, limit),
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
                SELECT 'graham' as sys, MAX(analysis_quarter) as q FROM graham_results WHERE mode = 'strict'
                UNION ALL
                SELECT 'magic_formula', MAX(analysis_quarter) FROM magic_formula_results
                UNION ALL
                SELECT 'piotroski', MAX(analysis_quarter) FROM piotroski_results
                UNION ALL
                SELECT 'altman', MAX(analysis_quarter) FROM altman_results
                UNION ALL
                SELECT 'roic', MAX(analysis_quarter) FROM roic_quality_results
                UNION ALL
                SELECT 'peg', MAX(analysis_quarter) FROM garp_peg_results
                UNION ALL
                SELECT 'fama_french', MAX(analysis_quarter) FROM fama_french_results
                UNION ALL
                SELECT 'net_net', MAX(analysis_quarter) FROM net_net_results
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
            (min_systems, limit),
        ).fetchall()

        columns = [desc[0] for desc in conn.description]
        return {
            "screen": "rankings",
            "min_systems": min_systems,
            "count": len(result),
            "stocks": [dict(zip(columns, row)) for row in result],
        }


@router.get("/combined")
async def screen_combined(
    graham_min: int = Query(None, ge=0, le=7),
    piotroski_min: int = Query(None, ge=0, le=9),
    altman_zone: str = Query(None, regex="^(safe|grey|distress)$"),
    min_roic: float = Query(None, ge=0, le=1),
    max_peg: float = Query(None, ge=0, le=10),
    limit: int = Query(100, ge=1, le=1000),
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
                "LEFT JOIN graham_results g ON t.symbol = g.symbol AND g.mode = 'strict'"
            )
            conditions.append("g.criteria_passed >= ?")
            params.append(graham_min)

        if piotroski_min is not None:
            query += ", p.f_score as piotroski_score"
            joins.append("LEFT JOIN piotroski_results p ON t.symbol = p.symbol")
            conditions.append("p.f_score >= ?")
            params.append(piotroski_min)

        if altman_zone is not None:
            query += ", a.z_score, a.zone as altman_zone"
            joins.append("LEFT JOIN altman_results a ON t.symbol = a.symbol")
            conditions.append("a.zone = ?")
            params.append(altman_zone)

        if min_roic is not None:
            query += ", r.roic"
            joins.append("LEFT JOIN roic_quality_results r ON t.symbol = r.symbol")
            conditions.append("r.roic >= ?")
            params.append(min_roic)

        if max_peg is not None:
            query += ", gp.peg_ratio"
            joins.append("LEFT JOIN garp_peg_results gp ON t.symbol = gp.symbol")
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
