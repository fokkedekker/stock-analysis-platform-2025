"""Analysis results API routes."""

from fastapi import APIRouter, HTTPException, Query

from src.database.connection import get_db_manager

router = APIRouter()


def _get_latest_analysis(table: str, symbol: str) -> dict | None:
    """Get latest analysis result for a symbol."""
    db = get_db_manager()

    with db.get_connection() as conn:
        result = conn.execute(
            f"""
            SELECT * FROM {table}
            WHERE UPPER(symbol) = UPPER(?)
            ORDER BY analysis_quarter DESC
            LIMIT 1
            """,
            (symbol,),
        ).fetchone()

        if not result:
            return None

        columns = [desc[0] for desc in conn.description]
        return dict(zip(columns, result))


@router.get("/{symbol}")
async def get_all_analyses(symbol: str):
    """Get all analysis results for a symbol."""
    graham = _get_latest_analysis("graham_results", symbol)
    magic_formula = _get_latest_analysis("magic_formula_results", symbol)
    piotroski = _get_latest_analysis("piotroski_results", symbol)
    altman = _get_latest_analysis("altman_results", symbol)
    roic = _get_latest_analysis("roic_quality_results", symbol)
    garp = _get_latest_analysis("garp_peg_results", symbol)
    fama_french = _get_latest_analysis("fama_french_results", symbol)
    net_net = _get_latest_analysis("net_net_results", symbol)

    if not any([graham, magic_formula, piotroski, altman, roic, garp, fama_french, net_net]):
        raise HTTPException(status_code=404, detail=f"No analyses found for {symbol}")

    return {
        "symbol": symbol,
        "graham": graham,
        "magic_formula": magic_formula,
        "piotroski": piotroski,
        "altman": altman,
        "roic_quality": roic,
        "garp_peg": garp,
        "fama_french": fama_french,
        "net_net": net_net,
    }


@router.get("/{symbol}/graham")
async def get_graham_analysis(
    symbol: str,
    mode: str = Query("strict", regex="^(strict|modern|garp|relaxed)$"),
):
    """Get Graham analysis for a symbol."""
    db = get_db_manager()

    with db.get_connection() as conn:
        result = conn.execute(
            """
            SELECT * FROM graham_results
            WHERE UPPER(symbol) = UPPER(?)
            AND mode = ?
            ORDER BY analysis_quarter DESC
            LIMIT 1
            """,
            (symbol, mode),
        ).fetchone()

        if not result:
            raise HTTPException(status_code=404, detail=f"Graham analysis not found for {symbol}")

        columns = [desc[0] for desc in conn.description]
        return dict(zip(columns, result))


@router.get("/{symbol}/magic-formula")
async def get_magic_formula_analysis(symbol: str):
    """Get Magic Formula analysis for a symbol."""
    result = _get_latest_analysis("magic_formula_results", symbol)
    if not result:
        raise HTTPException(
            status_code=404, detail=f"Magic Formula analysis not found for {symbol}"
        )
    return result


@router.get("/{symbol}/piotroski")
async def get_piotroski_analysis(symbol: str):
    """Get Piotroski F-Score analysis for a symbol."""
    result = _get_latest_analysis("piotroski_results", symbol)
    if not result:
        raise HTTPException(status_code=404, detail=f"Piotroski analysis not found for {symbol}")
    return result


@router.get("/{symbol}/altman")
async def get_altman_analysis(symbol: str):
    """Get Altman Z-Score analysis for a symbol."""
    result = _get_latest_analysis("altman_results", symbol)
    if not result:
        raise HTTPException(status_code=404, detail=f"Altman analysis not found for {symbol}")
    return result


@router.get("/{symbol}/roic")
async def get_roic_analysis(symbol: str):
    """Get ROIC/Quality analysis for a symbol."""
    result = _get_latest_analysis("roic_quality_results", symbol)
    if not result:
        raise HTTPException(status_code=404, detail=f"ROIC analysis not found for {symbol}")
    return result


@router.get("/{symbol}/peg")
async def get_peg_analysis(symbol: str):
    """Get GARP/PEG analysis for a symbol."""
    result = _get_latest_analysis("garp_peg_results", symbol)
    if not result:
        raise HTTPException(status_code=404, detail=f"PEG analysis not found for {symbol}")
    return result


@router.get("/{symbol}/fama-french")
async def get_fama_french_analysis(symbol: str):
    """Get Fama-French factor analysis for a symbol."""
    result = _get_latest_analysis("fama_french_results", symbol)
    if not result:
        raise HTTPException(
            status_code=404, detail=f"Fama-French analysis not found for {symbol}"
        )
    return result


@router.get("/{symbol}/net-net")
async def get_net_net_analysis(symbol: str):
    """Get Net-Net analysis for a symbol."""
    result = _get_latest_analysis("net_net_results", symbol)
    if not result:
        raise HTTPException(status_code=404, detail=f"Net-Net analysis not found for {symbol}")
    return result
