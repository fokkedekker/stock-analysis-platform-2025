"""AI-powered stock explanation endpoint with tool calling support."""

import json
import logging
from typing import AsyncGenerator, Generator

import httpx
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from src.config import get_settings
from src.database.connection import get_db_manager

logger = logging.getLogger(__name__)
router = APIRouter()

# Tool definitions for Cerebras
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_stock_news",
            "strict": True,
            "description": "Fetch recent news articles for a specific stock ticker. Use this when the user asks about recent news, events, what's happening with the stock, or wants to understand what might be driving price movements.",
            "parameters": {
                "type": "object",
                "properties": {
                    "symbol": {
                        "type": "string",
                        "description": "Stock ticker symbol (e.g., AAPL, NVDA)",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Number of articles to fetch (1-20)",
                    },
                },
                "required": ["symbol"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_press_releases",
            "strict": True,
            "description": "Fetch official company press releases. Use this when the user asks about company announcements, earnings releases, acquisitions, executive changes, or official statements.",
            "parameters": {
                "type": "object",
                "properties": {
                    "symbol": {
                        "type": "string",
                        "description": "Stock ticker symbol (e.g., AAPL, NVDA)",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Number of press releases to fetch (1-10)",
                    },
                },
                "required": ["symbol"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_market_news",
            "strict": True,
            "description": "Fetch general market and economic news. Use this when the user asks about broader market conditions, economic factors, macro trends, or industry-wide news.",
            "parameters": {
                "type": "object",
                "properties": {
                    "limit": {
                        "type": "integer",
                        "description": "Number of articles to fetch (1-20)",
                    },
                },
                "required": [],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_sector_peers",
            "strict": True,
            "description": "Get peer companies in the same sector for comparison. Use this when the user asks about competitors, industry comparison, how the stock compares to others, or relative performance.",
            "parameters": {
                "type": "object",
                "properties": {
                    "sector": {
                        "type": "string",
                        "description": "Sector name (e.g., Technology, Healthcare, Financial Services)",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Number of peer companies to fetch (1-20)",
                    },
                },
                "required": ["sector"],
                "additionalProperties": False,
            },
        },
    },
]


# Tool executor functions
async def execute_get_stock_news(symbol: str, limit: int = 5) -> str:
    """Fetch stock news from FMP API (stable endpoint)."""
    settings = get_settings()
    if not settings.FMP_API_KEY:
        return json.dumps({"error": "FMP API key not configured"})

    try:
        async with httpx.AsyncClient() as client:
            # Use the new stable API endpoint
            response = await client.get(
                f"{settings.FMP_BASE_URL}/stable/news/stock",
                params={
                    "symbols": symbol.upper(),
                    "limit": min(limit, 20),
                    "apikey": settings.FMP_API_KEY,
                },
                timeout=30.0,
            )
            response.raise_for_status()
            news = response.json()

            if not news:
                return json.dumps({"message": f"No recent news found for {symbol}"})

            # Format for LLM consumption
            formatted = []
            for n in news[:limit]:
                text = n.get("text", "")
                summary = text[:400] + "..." if len(text) > 400 else text
                formatted.append(
                    {
                        "title": n.get("title", ""),
                        "date": n.get("publishedDate", ""),
                        "source": n.get("site", ""),
                        "summary": summary,
                    }
                )

            logger.info(f"Fetched {len(formatted)} news articles for {symbol}")
            return json.dumps(formatted)

    except Exception as e:
        logger.error(f"Error fetching stock news: {e}")
        return json.dumps({"error": f"Failed to fetch news: {str(e)}"})


async def execute_get_press_releases(symbol: str, limit: int = 5) -> str:
    """Fetch press releases from FMP API (stable endpoint)."""
    settings = get_settings()
    if not settings.FMP_API_KEY:
        return json.dumps({"error": "FMP API key not configured"})

    try:
        async with httpx.AsyncClient() as client:
            # Use the new stable API endpoint
            response = await client.get(
                f"{settings.FMP_BASE_URL}/stable/news/press-releases",
                params={
                    "symbols": symbol.upper(),
                    "limit": min(limit, 10),
                    "apikey": settings.FMP_API_KEY,
                },
                timeout=30.0,
            )
            response.raise_for_status()
            releases = response.json()

            if not releases:
                return json.dumps({"message": f"No press releases found for {symbol}"})

            # Format for LLM consumption
            formatted = []
            for r in releases[:limit]:
                text = r.get("text", "")
                summary = text[:500] + "..." if len(text) > 500 else text
                formatted.append(
                    {
                        "title": r.get("title", ""),
                        "date": r.get("publishedDate", ""),
                        "summary": summary,
                    }
                )

            logger.info(f"Fetched {len(formatted)} press releases for {symbol}")
            return json.dumps(formatted)

    except Exception as e:
        logger.error(f"Error fetching press releases: {e}")
        return json.dumps({"error": f"Failed to fetch press releases: {str(e)}"})


async def execute_get_market_news(limit: int = 10) -> str:
    """Fetch general market news from FMP API (stable endpoint)."""
    settings = get_settings()
    if not settings.FMP_API_KEY:
        return json.dumps({"error": "FMP API key not configured"})

    try:
        async with httpx.AsyncClient() as client:
            # Use the new stable API endpoint for general news
            response = await client.get(
                f"{settings.FMP_BASE_URL}/stable/news/general-latest",
                params={"limit": min(limit, 20), "apikey": settings.FMP_API_KEY},
                timeout=30.0,
            )
            response.raise_for_status()
            news = response.json()

            if not news:
                return json.dumps({"message": "No market news available"})

            # Format for LLM consumption
            formatted = []
            for n in news[:limit]:
                text = n.get("text", "")
                summary = text[:400] + "..." if len(text) > 400 else text
                formatted.append(
                    {
                        "title": n.get("title", ""),
                        "date": n.get("publishedDate", ""),
                        "source": n.get("site", ""),
                        "summary": summary,
                    }
                )

            logger.info(f"Fetched {len(formatted)} market news articles")
            return json.dumps(formatted)

    except Exception as e:
        logger.error(f"Error fetching market news: {e}")
        return json.dumps({"error": f"Failed to fetch market news: {str(e)}"})


async def execute_get_sector_peers(sector: str, limit: int = 10) -> str:
    """Fetch peer companies in the same sector from FMP API (stable endpoint)."""
    settings = get_settings()
    if not settings.FMP_API_KEY:
        return json.dumps({"error": "FMP API key not configured"})

    try:
        async with httpx.AsyncClient() as client:
            # Use the new stable company-screener endpoint
            response = await client.get(
                f"{settings.FMP_BASE_URL}/stable/company-screener",
                params={
                    "sector": sector,
                    "marketCapMoreThan": 1000000000,  # > $1B market cap
                    "limit": min(limit, 20),
                    "apikey": settings.FMP_API_KEY,
                },
                timeout=30.0,
            )
            response.raise_for_status()
            peers = response.json()

            if not peers:
                return json.dumps({"message": f"No peer companies found in {sector} sector"})

            # Format for LLM consumption
            formatted = []
            for p in peers[:limit]:
                formatted.append(
                    {
                        "symbol": p.get("symbol", ""),
                        "name": p.get("companyName", ""),
                        "market_cap": p.get("marketCap", 0),
                        "sector": p.get("sector", ""),
                        "industry": p.get("industry", ""),
                        "price": p.get("price", 0),
                        "beta": p.get("beta", 0),
                    }
                )

            logger.info(f"Fetched {len(formatted)} peer companies in {sector}")
            return json.dumps(formatted)

    except Exception as e:
        logger.error(f"Error fetching sector peers: {e}")
        return json.dumps({"error": f"Failed to fetch sector peers: {str(e)}"})


# Map tool names to executor functions
TOOL_EXECUTORS = {
    "get_stock_news": execute_get_stock_news,
    "get_press_releases": execute_get_press_releases,
    "get_market_news": execute_get_market_news,
    "get_sector_peers": execute_get_sector_peers,
}


class ChatMessage(BaseModel):
    """A single chat message."""

    role: str  # 'user' or 'assistant'
    content: str


class ExplainRequest(BaseModel):
    """Request body for explain endpoint."""

    messages: list[ChatMessage]


def _get_analysis_data(symbol: str) -> dict:
    """Fetch all analysis data for a symbol from the database."""
    db = get_db_manager()

    data = {
        "symbol": symbol,
        "profile": None,
        "graham": None,
        "magic_formula": None,
        "piotroski": None,
        "altman": None,
        "roic": None,
        "peg": None,
        "fama_french": None,
        "net_net": None,
    }

    with db.get_connection() as conn:
        # Fetch company profile
        result = conn.execute(
            """
            SELECT * FROM company_profiles
            WHERE UPPER(symbol) = UPPER(?)
            ORDER BY fiscal_quarter DESC
            LIMIT 1
            """,
            (symbol,),
        ).fetchone()
        if result:
            columns = [desc[0] for desc in conn.description]
            data["profile"] = dict(zip(columns, result))

        # Fetch Graham analysis
        result = conn.execute(
            """
            SELECT * FROM graham_results
            WHERE UPPER(symbol) = UPPER(?)
            ORDER BY analysis_quarter DESC
            LIMIT 1
            """,
            (symbol,),
        ).fetchone()
        if result:
            columns = [desc[0] for desc in conn.description]
            data["graham"] = dict(zip(columns, result))

        # Fetch Magic Formula analysis
        result = conn.execute(
            """
            SELECT * FROM magic_formula_results
            WHERE UPPER(symbol) = UPPER(?)
            ORDER BY analysis_quarter DESC
            LIMIT 1
            """,
            (symbol,),
        ).fetchone()
        if result:
            columns = [desc[0] for desc in conn.description]
            data["magic_formula"] = dict(zip(columns, result))

        # Fetch Piotroski analysis
        result = conn.execute(
            """
            SELECT * FROM piotroski_results
            WHERE UPPER(symbol) = UPPER(?)
            ORDER BY analysis_quarter DESC
            LIMIT 1
            """,
            (symbol,),
        ).fetchone()
        if result:
            columns = [desc[0] for desc in conn.description]
            data["piotroski"] = dict(zip(columns, result))

        # Fetch Altman analysis
        result = conn.execute(
            """
            SELECT * FROM altman_results
            WHERE UPPER(symbol) = UPPER(?)
            ORDER BY analysis_quarter DESC
            LIMIT 1
            """,
            (symbol,),
        ).fetchone()
        if result:
            columns = [desc[0] for desc in conn.description]
            data["altman"] = dict(zip(columns, result))

        # Fetch ROIC analysis
        result = conn.execute(
            """
            SELECT * FROM roic_quality_results
            WHERE UPPER(symbol) = UPPER(?)
            ORDER BY analysis_quarter DESC
            LIMIT 1
            """,
            (symbol,),
        ).fetchone()
        if result:
            columns = [desc[0] for desc in conn.description]
            data["roic"] = dict(zip(columns, result))

        # Fetch PEG analysis
        result = conn.execute(
            """
            SELECT * FROM garp_peg_results
            WHERE UPPER(symbol) = UPPER(?)
            ORDER BY analysis_quarter DESC
            LIMIT 1
            """,
            (symbol,),
        ).fetchone()
        if result:
            columns = [desc[0] for desc in conn.description]
            data["peg"] = dict(zip(columns, result))

        # Fetch Fama-French analysis
        result = conn.execute(
            """
            SELECT * FROM fama_french_results
            WHERE UPPER(symbol) = UPPER(?)
            ORDER BY analysis_quarter DESC
            LIMIT 1
            """,
            (symbol,),
        ).fetchone()
        if result:
            columns = [desc[0] for desc in conn.description]
            data["fama_french"] = dict(zip(columns, result))

        # Fetch Net-Net analysis
        result = conn.execute(
            """
            SELECT * FROM net_net_results
            WHERE UPPER(symbol) = UPPER(?)
            ORDER BY analysis_quarter DESC
            LIMIT 1
            """,
            (symbol,),
        ).fetchone()
        if result:
            columns = [desc[0] for desc in conn.description]
            data["net_net"] = dict(zip(columns, result))

    return data


def _format_value(value, format_type: str = "number") -> str:
    """Format a value for display in the prompt."""
    if value is None:
        return "N/A"
    # Convert Decimal to float for arithmetic
    if hasattr(value, "__float__"):
        num_value = float(value)
    else:
        num_value = value
    if format_type == "currency":
        if abs(num_value) >= 1e12:
            return f"${num_value / 1e12:.2f}T"
        if abs(num_value) >= 1e9:
            return f"${num_value / 1e9:.2f}B"
        if abs(num_value) >= 1e6:
            return f"${num_value / 1e6:.2f}M"
        return f"${num_value:,.2f}"
    if format_type == "percent":
        return f"{num_value * 100:.2f}%"
    if format_type == "bool":
        return "Yes" if value else "No"
    if isinstance(num_value, float):
        return f"{num_value:.2f}"
    return str(value)


def _build_system_prompt(data: dict) -> str:
    """Build comprehensive system prompt with all analysis data."""
    symbol = data["symbol"]
    profile = data.get("profile") or {}
    graham = data.get("graham") or {}
    magic_formula = data.get("magic_formula") or {}
    piotroski = data.get("piotroski") or {}
    altman = data.get("altman") or {}
    roic = data.get("roic") or {}
    peg = data.get("peg") or {}
    fama_french = data.get("fama_french") or {}
    net_net = data.get("net_net") or {}

    # Determine survival status
    altman_pass = altman.get("zone") == "safe"
    piotroski_score = piotroski.get("f_score")
    piotroski_pass = piotroski_score is not None and piotroski_score >= 5
    survival_pass = altman_pass and piotroski_pass

    # Determine quality label
    roic_value = roic.get("roic")
    fcf_positive_5yr = roic.get("fcf_positive_5yr")
    if roic_value and roic_value >= 0.15 and fcf_positive_5yr:
        quality_label = "Compounder"
    elif roic_value and roic_value >= 0.08:
        quality_label = "Average"
    else:
        quality_label = "Weak"

    # Count valuation lenses passed
    graham_pass = (graham.get("criteria_passed") or 0) >= 5
    peg_pass = peg.get("peg_pass", False)
    combined_rank = magic_formula.get("combined_rank")
    mf_pass = combined_rank is not None and combined_rank <= 100
    net_net_pass = net_net.get("trading_below_ncav", False)
    bm_percentile = fama_french.get("book_to_market_percentile")
    ff_bm_pass = bm_percentile is not None and bm_percentile >= 0.7

    lenses = []
    if graham_pass:
        lenses.append("Graham")
    if net_net_pass:
        lenses.append("Net-Net")
    if peg_pass:
        lenses.append("PEG")
    if mf_pass:
        lenses.append("Magic Formula")
    if ff_bm_pass:
        lenses.append("Fama-French B/M")

    # Determine size label
    market_cap = profile.get("market_cap")
    if market_cap:
        if market_cap >= 10e9:
            size_label = "Large-Cap"
        elif market_cap >= 2e9:
            size_label = "Mid-Cap"
        else:
            size_label = "Small-Cap"
    else:
        size_label = "Unknown"

    sector = profile.get("sector", "Unknown")

    prompt = f"""You are a stock analysis assistant explaining what the analysis data means for THIS specific stock. Your audience is everyday investors.

## AVAILABLE TOOLS
You have access to tools that can fetch external data when needed:
- **get_stock_news**: Fetch recent news articles for {symbol}
- **get_press_releases**: Get official company press releases for {symbol}
- **get_market_news**: Get general market/economic news
- **get_sector_peers**: Find peer companies in the {sector} sector

USE THESE TOOLS when the user asks about:
- Recent news or events affecting the stock
- What might be driving price movements
- Company announcements or press releases
- Market conditions or economic factors
- How {symbol} compares to competitors

DO NOT call tools for the initial analysis - only when the user specifically asks follow-up questions about news, events, or comparisons.

IMPORTANT RULES:
1. ONLY use the data provided below. Never make up numbers or metrics.
2. If data is missing or marked as "N/A", explicitly state "data not available"
3. Be balanced - discuss both strengths and weaknesses
4. **INTERPRET THE SPECIFIC VALUES - DON'T EXPLAIN THE METRICS:**
   - BAD: "The Altman Z-Score measures bankruptcy risk. A score above 3.0 is safe. This company has 3.27."
   - GOOD: "With a Z-Score of 3.27, this company is financially stable - bankruptcy isn't a concern."
   - BAD: "The Piotroski F-Score measures financial health improvements year-over-year across 9 criteria."
   - GOOD: "A perfect 9/9 F-Score means this company is improving on every financial metric - profitability up, debt down, cash flow strong."
   - BAD: "P/E ratio compares price to earnings."
   - GOOD: "At a P/E of 14.37, you're paying $14.37 for every $1 of earnings - that's reasonable for a company of this quality."
   - BAD: "PEG ratio compares price to growth rate."
   - GOOD: "A PEG of 0.54 is very attractive - the stock price doesn't fully reflect its growth potential."
5. Focus on what the numbers tell us about THIS stock, not what metrics mean in general
6. Be concise - get to the point quickly
7. Format using Markdown:
   - Use **bold** for key verdicts
   - Use headings (##, ###) to structure sections
   - Use bullet points for lists
   - Use > blockquotes for key takeaways

## COMPANY OVERVIEW
Symbol: {symbol}
Company: {profile.get('company_name', 'N/A')}
Sector: {profile.get('sector', 'N/A')}
Industry: {profile.get('industry', 'N/A')}
Current Price: {_format_value(profile.get('price'), 'currency')}
Market Cap: {_format_value(profile.get('market_cap'), 'currency')}
P/E Ratio: {_format_value(profile.get('pe_ratio'))}
P/B Ratio: {_format_value(profile.get('pb_ratio'))}

## STAGE 1: SURVIVAL ANALYSIS (Hard Exclusion)
These gates determine if the stock is financially stable enough to even consider.

### Altman Z-Score (Bankruptcy Risk)
- Z-Score: {_format_value(altman.get('z_score'))}
- Zone: {altman.get('zone', 'N/A')} (safe > 3.0, grey 1.8-3.0, distress < 1.8)
- Components:
  - Working Capital / Total Assets (X1): {_format_value(altman.get('x1_wc_ta'))}
  - Retained Earnings / Total Assets (X2): {_format_value(altman.get('x2_re_ta'))}
  - EBIT / Total Assets (X3): {_format_value(altman.get('x3_ebit_ta'))}
  - Market Cap / Total Liabilities (X4): {_format_value(altman.get('x4_mc_tl'))}
  - Revenue / Total Assets (X5): {_format_value(altman.get('x5_rev_ta'))}
- SURVIVAL GATE PASSED: {_format_value(altman_pass, 'bool')}

### Piotroski F-Score (Financial Health)
- F-Score: {piotroski.get('f_score', 'N/A')}/9
- Profitability Signals (4 points):
  - ROA Positive: {_format_value(piotroski.get('roa_positive'), 'bool')}
  - Operating Cash Flow Positive: {_format_value(piotroski.get('operating_cf_positive'), 'bool')}
  - ROA Increasing YoY: {_format_value(piotroski.get('roa_increasing'), 'bool')}
  - Cash Flow > Net Income (Accruals): {_format_value(piotroski.get('accruals_signal'), 'bool')}
- Leverage/Liquidity Signals (3 points):
  - Leverage Decreasing: {_format_value(piotroski.get('leverage_decreasing'), 'bool')}
  - Current Ratio Increasing: {_format_value(piotroski.get('current_ratio_increasing'), 'bool')}
  - No Share Dilution: {_format_value(piotroski.get('no_dilution'), 'bool')}
- Efficiency Signals (2 points):
  - Gross Margin Increasing: {_format_value(piotroski.get('gross_margin_increasing'), 'bool')}
  - Asset Turnover Increasing: {_format_value(piotroski.get('asset_turnover_increasing'), 'bool')}
- SURVIVAL GATE PASSED: {_format_value(piotroski_pass, 'bool')} (requires >= 5)

### OVERALL SURVIVAL: {_format_value(survival_pass, 'bool')}

## STAGE 2: QUALITY CLASSIFICATION
This stage classifies the business quality - it does NOT exclude stocks.

### ROIC Quality Metrics
- ROIC: {_format_value(roic.get('roic'), 'percent')} (Compounder >= 15%, Average >= 8%, Weak < 8%)
- Free Cash Flow: {_format_value(roic.get('free_cash_flow'), 'currency')}
- FCF Positive 5+ Years: {_format_value(roic.get('fcf_positive_5yr'), 'bool')}
- Debt-to-Equity: {_format_value(roic.get('debt_to_equity'))}
- NOPAT: {_format_value(roic.get('nopat'), 'currency')}
- Invested Capital: {_format_value(roic.get('invested_capital'), 'currency')}

### QUALITY LABEL: {quality_label}

## STAGE 3: VALUATION LENSES (Buy Eligibility)
Each lens is an independent buy rationale. A stock becomes buy-eligible if it passes at least N lenses.

### Graham Analysis (mode: {graham.get('mode', 'N/A')})
- Criteria Passed: {graham.get('criteria_passed', 'N/A')}/8
- Individual Criteria:
  - Adequate Size (Revenue {_format_value(graham.get('revenue'), 'currency')}): {_format_value(graham.get('adequate_size'), 'bool')}
  - Current Ratio ({_format_value(graham.get('current_ratio'))}): {_format_value(graham.get('current_ratio_pass'), 'bool')}
  - Debt Coverage: {_format_value(graham.get('debt_coverage_pass'), 'bool')}
  - Earnings Stability: {_format_value(graham.get('earnings_stability'), 'bool')}
  - Dividend Record: {_format_value(graham.get('dividend_record'), 'bool')}
  - Earnings Growth ({_format_value(graham.get('eps_5yr_growth'), 'percent')}): {_format_value(graham.get('earnings_growth_pass'), 'bool')}
  - P/E Ratio ({_format_value(graham.get('pe_ratio'))}): {_format_value(graham.get('pe_ratio_pass'), 'bool')}
  - P/B Ratio ({_format_value(graham.get('pb_ratio'))}): {_format_value(graham.get('pb_ratio_pass'), 'bool')}
- P/E x P/B: {_format_value(graham.get('pe_x_pb'))} (should be < 22.5)
- GRAHAM LENS PASSED: {_format_value(graham_pass, 'bool')} (requires >= 5 criteria)

### Net-Net Analysis (Deep Value)
- Current Assets: {_format_value(net_net.get('current_assets'), 'currency')}
- Total Liabilities: {_format_value(net_net.get('total_liabilities'), 'currency')}
- NCAV: {_format_value(net_net.get('ncav'), 'currency')}
- NCAV Per Share: {_format_value(net_net.get('ncav_per_share'), 'currency')}
- Stock Price: {_format_value(net_net.get('price'), 'currency')}
- Market Cap: {_format_value(net_net.get('market_cap'), 'currency')}
- Discount to NCAV: {_format_value(net_net.get('discount_to_ncav'), 'percent')}
- Trading Below NCAV: {_format_value(net_net.get('trading_below_ncav'), 'bool')}
- Deep Value (<67% NCAV): {_format_value(net_net.get('deep_value'), 'bool')}
- NET-NET LENS PASSED: {_format_value(net_net_pass, 'bool')}

### PEG/GARP Analysis (Growth at Reasonable Price)
- Price: {_format_value(peg.get('price'), 'currency')}
- EPS: {_format_value(peg.get('eps'), 'currency')}
- P/E Ratio: {_format_value(peg.get('pe_ratio'))}
- EPS Growth 1yr: {_format_value(peg.get('eps_growth_1yr'), 'percent')}
- EPS Growth 3yr: {_format_value(peg.get('eps_growth_3yr'), 'percent')}
- EPS Growth 5yr: {_format_value(peg.get('eps_growth_5yr'), 'percent')}
- EPS CAGR (used): {_format_value(peg.get('eps_cagr'), 'percent')}
- PEG Ratio: {_format_value(peg.get('peg_ratio'))} (< 1.0 undervalued, 1.0-1.5 fair, > 1.5 overvalued)
- Growth Pass (>= 10%): {_format_value(peg.get('growth_pass'), 'bool')}
- PEG Pass (<= 1.5): {_format_value(peg.get('peg_pass'), 'bool')}
- PEG LENS PASSED: {_format_value(peg_pass, 'bool')}

### Magic Formula (Quality + Value)
- EBIT: {_format_value(magic_formula.get('ebit'), 'currency')}
- Enterprise Value: {_format_value(magic_formula.get('enterprise_value'), 'currency')}
- Earnings Yield: {_format_value(magic_formula.get('earnings_yield'), 'percent')}
- Net Working Capital: {_format_value(magic_formula.get('net_working_capital'), 'currency')}
- Net Fixed Assets: {_format_value(magic_formula.get('net_fixed_assets'), 'currency')}
- Return on Capital: {_format_value(magic_formula.get('return_on_capital'), 'percent')}
- Earnings Yield Rank: #{magic_formula.get('earnings_yield_rank', 'N/A')}
- Return on Capital Rank: #{magic_formula.get('return_on_capital_rank', 'N/A')}
- Combined Rank: #{magic_formula.get('combined_rank', 'N/A')}
- MAGIC FORMULA LENS PASSED: {_format_value(mf_pass, 'bool')} (top 100 = top ~5-10%)

### Fama-French Value (Book-to-Market)
- Book Value: {_format_value(fama_french.get('book_value'), 'currency')}
- Market Cap: {_format_value(fama_french.get('market_cap'), 'currency')}
- Book-to-Market Ratio: {_format_value(fama_french.get('book_to_market'))}
- B/M Percentile: {_format_value(fama_french.get('book_to_market_percentile'), 'percent')}
- FAMA-FRENCH B/M LENS PASSED: {_format_value(ff_bm_pass, 'bool')} (requires >= 70th percentile)

### VALUATION LENSES SUMMARY
- Lenses Passed: {len(lenses)}/5
- Passing Lenses: {', '.join(lenses) if lenses else 'None'}

## STAGE 4: FACTOR EXPOSURE (Context Only)
This information is for portfolio context and NEVER used as a filter.

### Fama-French Factor Analysis
- Size: {size_label} (Market Cap: {_format_value(profile.get('market_cap'), 'currency')})
- Profitability: {_format_value(fama_french.get('profitability'), 'percent')}
- Profitability Percentile: {_format_value(fama_french.get('profitability_percentile'), 'percent')}
- Asset Growth: {_format_value(fama_french.get('asset_growth'), 'percent')}
- Asset Growth Percentile: {_format_value(fama_french.get('asset_growth_percentile'), 'percent')}

## PIPELINE SUMMARY
- Survival Gates: {_format_value(survival_pass, 'bool')}
- Quality Classification: {quality_label}
- Valuation Lenses Passed: {len(lenses)}/5 ({', '.join(lenses) if lenses else 'None'})

## YOUR TASK
Analyze this stock concisely. Focus on what the specific values mean for THIS company.

### 1. Survival Check
Is this company financially stable? Interpret the Z-Score and F-Score values directly.

### 2. Business Quality
What do the ROIC and cash flow numbers tell us about this specific business?

### 3. Valuation
For each lens that passed, explain why THIS stock looks attractively priced (use the actual numbers).
For lenses that failed, briefly note what that means.

### 4. Key Risks
What warning signs exist in THIS company's data?

### 5. Final Verdict
2-3 sentences: Buy candidate or not? Be direct.

Keep it concise. Interpret the values, don't define the metrics.
"""
    return prompt


def _stream_cerebras_response(messages: list[dict]) -> Generator[str, None, None]:
    """Stream chunks from Cerebras API."""
    try:
        from cerebras.cloud.sdk import Cerebras
    except ImportError:
        yield f"data: {json.dumps({'content': 'Error: cerebras-cloud-sdk not installed. Run: pip install cerebras-cloud-sdk'})}\n\n"
        yield "data: [DONE]\n\n"
        return

    settings = get_settings()
    if not settings.CEREBRAS_API_KEY:
        yield f"data: {json.dumps({'content': 'Error: CEREBRAS_API_KEY not set in environment variables.'})}\n\n"
        yield "data: [DONE]\n\n"
        return

    try:
        client = Cerebras(api_key=settings.CEREBRAS_API_KEY)

        stream = client.chat.completions.create(
            messages=messages,
            model="zai-glm-4.6",
            stream=True,
            max_completion_tokens=4096,
            temperature=0.6,
            top_p=0.95,
        )

        for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                content = chunk.choices[0].delta.content
                yield f"data: {json.dumps({'content': content})}\n\n"

    except Exception as e:
        yield f"data: {json.dumps({'content': f'Error calling Cerebras API: {str(e)}'})}\n\n"

    yield "data: [DONE]\n\n"


async def _stream_cerebras_response_with_tools(
    messages: list[dict], symbol: str
) -> AsyncGenerator[str, None]:
    """Stream response with tool calling support.

    This function implements a tool-calling loop:
    1. Send messages to Cerebras with tool definitions
    2. If model returns a tool call, execute it and continue
    3. When model returns content (no tool call), stream the final response
    """
    settings = get_settings()

    if not settings.CEREBRAS_API_KEY:
        yield f"data: {json.dumps({'content': 'Error: CEREBRAS_API_KEY not set in environment variables.'})}\n\n"
        yield "data: [DONE]\n\n"
        return

    # Keep track of the messages for multi-turn tool calling
    working_messages = messages.copy()
    max_tool_calls = 5  # Safety limit to prevent infinite loops

    async with httpx.AsyncClient() as client:
        for tool_iteration in range(max_tool_calls):
            try:
                # Make non-streaming call to check for tool calls
                response = await client.post(
                    "https://api.cerebras.ai/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {settings.CEREBRAS_API_KEY}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": "zai-glm-4.6",
                        "messages": working_messages,
                        "tools": TOOLS,
                        "parallel_tool_calls": False,
                        "max_completion_tokens": 4096,
                        "temperature": 0.6,
                    },
                    timeout=60.0,
                )
                response.raise_for_status()
                result = response.json()

                if not result.get("choices"):
                    logger.error(f"No choices in Cerebras response: {result}")
                    yield f"data: {json.dumps({'content': 'Error: Invalid response from AI model.'})}\n\n"
                    yield "data: [DONE]\n\n"
                    return

                choice = result["choices"][0]["message"]

                # Check if model wants to call a tool
                if choice.get("tool_calls"):
                    tool_call = choice["tool_calls"][0]
                    func_name = tool_call["function"]["name"]
                    func_args_str = tool_call["function"]["arguments"]

                    try:
                        func_args = json.loads(func_args_str)
                    except json.JSONDecodeError:
                        func_args = {}

                    logger.info(f"Tool call: {func_name} with args: {func_args}")

                    # Yield status update to frontend
                    yield f"data: {json.dumps({'tool_call': func_name, 'args': func_args})}\n\n"

                    # Execute the tool
                    if func_name in TOOL_EXECUTORS:
                        tool_result = await TOOL_EXECUTORS[func_name](**func_args)
                    else:
                        tool_result = json.dumps({"error": f"Unknown tool: {func_name}"})

                    logger.info(f"Tool result length: {len(tool_result)} chars")

                    # Add assistant's tool call message
                    working_messages.append(
                        {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": tool_call["id"],
                                    "type": "function",
                                    "function": {
                                        "name": func_name,
                                        "arguments": func_args_str,
                                    },
                                }
                            ],
                        }
                    )

                    # Add tool response
                    working_messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call["id"],
                            "content": tool_result,
                        }
                    )

                    # Continue loop - model may call more tools or generate response
                    continue

                # No tool call - model wants to generate final response
                # Stream the final response
                logger.info("No tool call, streaming final response")

                # Use streaming for final response
                stream_response = await client.post(
                    "https://api.cerebras.ai/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {settings.CEREBRAS_API_KEY}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": "zai-glm-4.6",
                        "messages": working_messages,
                        "stream": True,
                        "max_completion_tokens": 4096,
                        "temperature": 0.6,
                    },
                    timeout=120.0,
                )
                stream_response.raise_for_status()

                async for line in stream_response.aiter_lines():
                    if line.startswith("data: "):
                        data_str = line[6:]
                        if data_str == "[DONE]":
                            break
                        try:
                            chunk = json.loads(data_str)
                            if (
                                chunk.get("choices")
                                and chunk["choices"][0].get("delta", {}).get("content")
                            ):
                                content = chunk["choices"][0]["delta"]["content"]
                                yield f"data: {json.dumps({'content': content})}\n\n"
                        except json.JSONDecodeError:
                            continue

                yield "data: [DONE]\n\n"
                return  # Exit after streaming response

            except httpx.HTTPStatusError as e:
                logger.error(f"HTTP error from Cerebras: {e.response.status_code} - {e.response.text}")
                yield f"data: {json.dumps({'content': f'Error: AI service returned {e.response.status_code}'})}\n\n"
                yield "data: [DONE]\n\n"
                return

            except Exception as e:
                logger.error(f"Error in tool calling loop: {e}")
                yield f"data: {json.dumps({'content': f'Error: {str(e)}'})}\n\n"
                yield "data: [DONE]\n\n"
                return

        # If we hit the tool call limit
        logger.warning(f"Hit max tool calls limit ({max_tool_calls})")
        yield f"data: {json.dumps({'content': 'I made several data requests but need to provide you an answer now. Based on the information gathered...'})}\n\n"
        yield "data: [DONE]\n\n"


@router.post("/{symbol}")
async def explain_stock(symbol: str, request: ExplainRequest):
    """Generate AI explanation for a stock with chat history support."""
    # Fetch all analysis data
    data = _get_analysis_data(symbol)

    # Check if we have any data
    has_data = any(
        data.get(key)
        for key in ["graham", "magic_formula", "piotroski", "altman", "roic", "peg", "fama_french", "net_net"]
    )
    if not has_data:
        raise HTTPException(status_code=404, detail=f"No analysis data found for {symbol}")

    # Build system prompt with all data embedded
    system_prompt = _build_system_prompt(data)

    # Build messages array with system prompt + chat history
    messages = [{"role": "system", "content": system_prompt}]
    for msg in request.messages:
        messages.append({"role": msg.role, "content": msg.content})

    # Stream response from Cerebras with tool calling support
    return StreamingResponse(
        _stream_cerebras_response_with_tools(messages, symbol),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
