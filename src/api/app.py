"""FastAPI application setup."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.routes import tickers, financials, analysis, screener

app = FastAPI(
    title="Stock Analysis API",
    description="Fundamental valuation systems for NYSE/NASDAQ stocks",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS middleware for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(tickers.router, prefix="/api/v1/tickers", tags=["Tickers"])
app.include_router(financials.router, prefix="/api/v1/financials", tags=["Financials"])
app.include_router(analysis.router, prefix="/api/v1/analysis", tags=["Analysis"])
app.include_router(screener.router, prefix="/api/v1/screener", tags=["Screener"])


@app.get("/")
async def root():
    """Root endpoint with API info."""
    return {
        "name": "Stock Analysis API",
        "version": "0.1.0",
        "docs": "/docs",
    }


@app.get("/api/v1/status")
async def status():
    """Get API status and data freshness."""
    from src.database.connection import get_db_manager

    db = get_db_manager()

    with db.get_connection() as conn:
        # Get counts
        ticker_count = conn.execute(
            "SELECT COUNT(*) FROM tickers WHERE is_active = TRUE"
        ).fetchone()[0]

        profile_count = conn.execute("SELECT COUNT(*) FROM company_profiles").fetchone()[0]

        # Get latest fetch dates
        latest_profile = conn.execute(
            "SELECT MAX(fetched_at) FROM company_profiles"
        ).fetchone()[0]

        latest_income = conn.execute(
            "SELECT MAX(fetched_at) FROM income_statements"
        ).fetchone()[0]

        # Get analysis counts
        graham_count = conn.execute("SELECT COUNT(*) FROM graham_results").fetchone()[0]
        piotroski_count = conn.execute("SELECT COUNT(*) FROM piotroski_results").fetchone()[0]

    return {
        "status": "healthy",
        "data": {
            "tickers": ticker_count,
            "profiles": profile_count,
            "graham_analyses": graham_count,
            "piotroski_analyses": piotroski_count,
        },
        "last_updated": {
            "profiles": str(latest_profile) if latest_profile else None,
            "income_statements": str(latest_income) if latest_income else None,
        },
    }
