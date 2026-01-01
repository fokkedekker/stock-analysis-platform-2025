"""Data fetcher with checkpoint/resume capability for bulk operations."""

import asyncio
import json
import logging
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import duckdb
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, TimeElapsedColumn
from rich.console import Console

from src.config import get_settings
from src.database.connection import get_db_manager
from src.scrapers.fmp_client import FMPClient

logger = logging.getLogger(__name__)
console = Console()

# All endpoints to fetch per symbol
ALL_ENDPOINTS = [
    "profile",
    "income_annual",
    "income_quarterly",
    "balance_annual",
    "balance_quarterly",
    "cashflow_annual",
    "cashflow_quarterly",
    "metrics_annual",
    "metrics_quarterly",
    "ratios_annual",
    "ratios_quarterly",
    "dividends",
]


@dataclass
class FetchTask:
    """A single fetch task (symbol + endpoint)."""
    symbol: str
    endpoint: str


class CheckpointManager:
    """Manages checkpoint state for resumable data fetching at endpoint level."""

    def __init__(self, checkpoint_path: str | Path | None = None):
        """Initialize checkpoint manager.

        Args:
            checkpoint_path: Path to checkpoint file. Uses settings default if not provided.
        """
        settings = get_settings()
        self.checkpoint_path = Path(checkpoint_path or settings.CHECKPOINT_PATH)
        self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        self._state: dict[str, Any] = self._load()
        self._lock = asyncio.Lock()
        self._save_counter = 0

    def _load(self) -> dict[str, Any]:
        """Load checkpoint state from file."""
        if self.checkpoint_path.exists():
            try:
                return json.loads(self.checkpoint_path.read_text())
            except Exception as e:
                logger.warning(f"Failed to load checkpoint: {e}")
        return {"completed": {}, "failed": {}, "started_at": None}

    def _save(self) -> None:
        """Save checkpoint state to file."""
        self.checkpoint_path.write_text(json.dumps(self._state, indent=2, default=str))

    async def _save_async(self) -> None:
        """Save checkpoint periodically (every 10 updates to reduce I/O)."""
        self._save_counter += 1
        if self._save_counter >= 10:
            self._save()
            self._save_counter = 0

    def is_endpoint_completed(self, symbol: str, endpoint: str) -> bool:
        """Check if a specific endpoint for a symbol has been completed."""
        key = f"{symbol}:{endpoint}"
        return key in self._state.get("completed", {})

    def is_symbol_completed(self, symbol: str) -> bool:
        """Check if all endpoints for a symbol are completed."""
        completed = self._state.get("completed", {})
        for endpoint in ALL_ENDPOINTS:
            if f"{symbol}:{endpoint}" not in completed:
                return False
        return True

    async def mark_endpoint_completed(self, symbol: str, endpoint: str) -> None:
        """Mark a specific endpoint as completed."""
        async with self._lock:
            if "completed" not in self._state:
                self._state["completed"] = {}
            key = f"{symbol}:{endpoint}"
            self._state["completed"][key] = datetime.utcnow().isoformat()
            await self._save_async()

    async def mark_endpoint_failed(self, symbol: str, endpoint: str, error: str) -> None:
        """Mark a specific endpoint as failed."""
        async with self._lock:
            if "failed" not in self._state:
                self._state["failed"] = {}
            key = f"{symbol}:{endpoint}"
            self._state["failed"][key] = {
                "error": error,
                "timestamp": datetime.utcnow().isoformat(),
            }
            await self._save_async()

    def get_completed_count(self) -> int:
        """Get number of completed endpoint fetches."""
        return len(self._state.get("completed", {}))

    def get_failed_count(self) -> int:
        """Get number of failed endpoint fetches."""
        return len(self._state.get("failed", {}))

    def clear(self) -> None:
        """Clear checkpoint state."""
        self._state = {"completed": {}, "failed": {}, "started_at": None}
        self._save()

    def start_session(self) -> None:
        """Mark the start of a new fetching session."""
        self._state["started_at"] = datetime.utcnow().isoformat()
        self._save()

    def force_save(self) -> None:
        """Force save the checkpoint (call at end of session)."""
        self._save()


class DataFetcher:
    """Fetches financial data for all tickers with progress tracking and resume capability."""

    def __init__(
        self,
        fmp_client: FMPClient,
        checkpoint_path: str | Path | None = None,
    ):
        """Initialize data fetcher.

        Args:
            fmp_client: Configured FMP API client.
            checkpoint_path: Path to checkpoint file.
        """
        self.client = fmp_client
        self.db = get_db_manager()
        self.checkpoint = CheckpointManager(checkpoint_path)

    async def fetch_symbol(self, symbol: str, include_quarterly: bool = True) -> dict[str, Any]:
        """Fetch all data for a single symbol.

        Args:
            symbol: Stock ticker symbol.
            include_quarterly: Whether to include quarterly data.

        Returns:
            Dictionary with all fetched data.
        """
        return await self.client.fetch_all_data(symbol, include_quarterly)

    def save_profile(
        self,
        symbol: str,
        profile_data: list[dict[str, Any]] | None,
        conn: duckdb.DuckDBPyConnection | None = None,
    ) -> None:
        """Save company profile to database.

        Args:
            symbol: Stock ticker symbol.
            profile_data: Profile data from FMP.
            conn: Optional existing database connection to reuse.
        """
        if not profile_data or len(profile_data) == 0:
            return

        profile = profile_data[0]
        quarter = self._get_current_quarter()
        now = datetime.utcnow()

        params = (
            symbol,
            quarter,
            self._safe_decimal(profile.get("price")),
            self._safe_decimal(profile.get("marketCap")),  # API returns marketCap not mktCap
            self._safe_decimal(profile.get("pe")),
            self._safe_decimal(profile.get("priceToBook")),
            self._safe_decimal(profile.get("beta")),
            self._safe_int(profile.get("sharesOutstanding")),
            self._safe_int(profile.get("volAvg")),
            self._safe_decimal(profile.get("lastDiv")),
            now,
            json.dumps(profile),
        )

        query = """
            INSERT INTO company_profiles (
                symbol, fiscal_quarter, price, market_cap, pe_ratio, pb_ratio,
                beta, shares_outstanding, avg_volume, dividend_yield,
                fetched_at, raw_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT (symbol, fiscal_quarter) DO UPDATE SET
                price = EXCLUDED.price,
                market_cap = EXCLUDED.market_cap,
                pe_ratio = EXCLUDED.pe_ratio,
                pb_ratio = EXCLUDED.pb_ratio,
                beta = EXCLUDED.beta,
                shares_outstanding = EXCLUDED.shares_outstanding,
                avg_volume = EXCLUDED.avg_volume,
                dividend_yield = EXCLUDED.dividend_yield,
                fetched_at = EXCLUDED.fetched_at,
                raw_json = EXCLUDED.raw_json
        """

        # Also update tickers table with sector/industry
        sector = profile.get("sector")
        industry = profile.get("industry")
        tickers_query = """
            UPDATE tickers
            SET sector = ?, industry = ?, updated_at = ?
            WHERE symbol = ?
        """
        tickers_params = (sector, industry, now, symbol)

        if conn is not None:
            with self.db.transaction(conn) as txn:
                txn.execute(query, params)
                if sector or industry:
                    txn.execute(tickers_query, tickers_params)
        else:
            with self.db.get_connection() as c:
                c.execute(query, params)
                if sector or industry:
                    c.execute(tickers_query, tickers_params)

    def save_income_statements(
        self,
        symbol: str,
        statements: list[dict[str, Any]] | None,
        period: str,
        conn: duckdb.DuckDBPyConnection | None = None,
    ) -> None:
        """Save income statements to database.

        Args:
            symbol: Stock ticker symbol.
            statements: List of income statement records.
            period: 'annual' or 'quarter'.
            conn: Optional existing database connection to reuse.
        """
        if not statements:
            return

        now = datetime.utcnow()

        ctx = self.db.transaction(conn) if conn is not None else self.db.get_connection()
        with ctx as db_conn:
            for stmt in statements:
                try:
                    db_conn.execute(
                        """
                        INSERT INTO income_statements (
                            symbol, fiscal_date, period, fiscal_year, reported_currency,
                            revenue, cost_of_revenue, gross_profit, gross_profit_ratio,
                            research_and_development, selling_general_admin, operating_expenses,
                            operating_income, operating_income_ratio, interest_expense,
                            ebit, ebitda, income_before_tax, income_tax_expense,
                            net_income, net_income_ratio, eps, eps_diluted,
                            weighted_avg_shares, weighted_avg_shares_diluted,
                            fetched_at, raw_json
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        ON CONFLICT (symbol, fiscal_date, period) DO UPDATE SET
                            fiscal_year = EXCLUDED.fiscal_year,
                            reported_currency = EXCLUDED.reported_currency,
                            revenue = EXCLUDED.revenue,
                            cost_of_revenue = EXCLUDED.cost_of_revenue,
                            gross_profit = EXCLUDED.gross_profit,
                            gross_profit_ratio = EXCLUDED.gross_profit_ratio,
                            research_and_development = EXCLUDED.research_and_development,
                            selling_general_admin = EXCLUDED.selling_general_admin,
                            operating_expenses = EXCLUDED.operating_expenses,
                            operating_income = EXCLUDED.operating_income,
                            operating_income_ratio = EXCLUDED.operating_income_ratio,
                            interest_expense = EXCLUDED.interest_expense,
                            ebit = EXCLUDED.ebit,
                            ebitda = EXCLUDED.ebitda,
                            income_before_tax = EXCLUDED.income_before_tax,
                            income_tax_expense = EXCLUDED.income_tax_expense,
                            net_income = EXCLUDED.net_income,
                            net_income_ratio = EXCLUDED.net_income_ratio,
                            eps = EXCLUDED.eps,
                            eps_diluted = EXCLUDED.eps_diluted,
                            weighted_avg_shares = EXCLUDED.weighted_avg_shares,
                            weighted_avg_shares_diluted = EXCLUDED.weighted_avg_shares_diluted,
                            fetched_at = EXCLUDED.fetched_at,
                            raw_json = EXCLUDED.raw_json
                        """,
                        (
                            symbol,
                            stmt.get("date"),
                            period,
                            self._safe_int(stmt.get("calendarYear")),
                            stmt.get("reportedCurrency"),
                            self._safe_decimal(stmt.get("revenue")),
                            self._safe_decimal(stmt.get("costOfRevenue")),
                            self._safe_decimal(stmt.get("grossProfit")),
                            self._safe_decimal(stmt.get("grossProfitRatio")),
                            self._safe_decimal(stmt.get("researchAndDevelopmentExpenses")),
                            self._safe_decimal(stmt.get("sellingGeneralAndAdministrativeExpenses")),
                            self._safe_decimal(stmt.get("operatingExpenses")),
                            self._safe_decimal(stmt.get("operatingIncome")),
                            self._safe_decimal(stmt.get("operatingIncomeRatio")),
                            self._safe_decimal(stmt.get("interestExpense")),
                            self._safe_decimal(stmt.get("ebit")),  # EBIT from FMP
                            self._safe_decimal(stmt.get("ebitda")),
                            self._safe_decimal(stmt.get("incomeBeforeTax")),
                            self._safe_decimal(stmt.get("incomeTaxExpense")),
                            self._safe_decimal(stmt.get("netIncome")),
                            self._safe_decimal(stmt.get("netIncomeRatio")),
                            self._safe_decimal(stmt.get("eps")),
                            self._safe_decimal(stmt.get("epsdiluted")),
                            self._safe_int(stmt.get("weightedAverageShsOut")),
                            self._safe_int(stmt.get("weightedAverageShsOutDil")),
                            now,
                            json.dumps(stmt),
                        ),
                    )
                except Exception as e:
                    logger.error(f"Error saving income statement for {symbol}: {e}")

    def save_balance_sheets(
        self,
        symbol: str,
        statements: list[dict[str, Any]] | None,
        period: str,
        conn: duckdb.DuckDBPyConnection | None = None,
    ) -> None:
        """Save balance sheets to database."""
        if not statements:
            return

        now = datetime.utcnow()

        ctx = self.db.transaction(conn) if conn is not None else self.db.get_connection()
        with ctx as db_conn:
            for stmt in statements:
                try:
                    db_conn.execute(
                        """
                        INSERT INTO balance_sheets (
                            symbol, fiscal_date, period, fiscal_year, reported_currency,
                            total_assets, current_assets, cash_and_equivalents, short_term_investments,
                            net_receivables, inventory, other_current_assets, non_current_assets,
                            property_plant_equipment, goodwill, intangible_assets, long_term_investments,
                            total_liabilities, current_liabilities, accounts_payable, short_term_debt,
                            deferred_revenue, other_current_liabilities, non_current_liabilities,
                            long_term_debt, deferred_tax_liabilities, other_non_current_liabilities,
                            total_debt, total_equity, common_stock, retained_earnings, treasury_stock,
                            other_comprehensive_income, total_stockholders_equity, minority_interest,
                            common_shares_outstanding, fetched_at, raw_json
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        ON CONFLICT (symbol, fiscal_date, period) DO UPDATE SET
                            fiscal_year = EXCLUDED.fiscal_year,
                            total_assets = EXCLUDED.total_assets,
                            current_assets = EXCLUDED.current_assets,
                            cash_and_equivalents = EXCLUDED.cash_and_equivalents,
                            total_liabilities = EXCLUDED.total_liabilities,
                            current_liabilities = EXCLUDED.current_liabilities,
                            long_term_debt = EXCLUDED.long_term_debt,
                            total_debt = EXCLUDED.total_debt,
                            total_equity = EXCLUDED.total_equity,
                            retained_earnings = EXCLUDED.retained_earnings,
                            common_shares_outstanding = EXCLUDED.common_shares_outstanding,
                            fetched_at = EXCLUDED.fetched_at,
                            raw_json = EXCLUDED.raw_json
                        """,
                        (
                            symbol,
                            stmt.get("date"),
                            period,
                            self._safe_int(stmt.get("calendarYear")),
                            stmt.get("reportedCurrency"),
                            self._safe_decimal(stmt.get("totalAssets")),
                            self._safe_decimal(stmt.get("totalCurrentAssets")),
                            self._safe_decimal(stmt.get("cashAndCashEquivalents")),
                            self._safe_decimal(stmt.get("shortTermInvestments")),
                            self._safe_decimal(stmt.get("netReceivables")),
                            self._safe_decimal(stmt.get("inventory")),
                            self._safe_decimal(stmt.get("otherCurrentAssets")),
                            self._safe_decimal(stmt.get("totalNonCurrentAssets")),
                            self._safe_decimal(stmt.get("propertyPlantEquipmentNet")),
                            self._safe_decimal(stmt.get("goodwill")),
                            self._safe_decimal(stmt.get("intangibleAssets")),
                            self._safe_decimal(stmt.get("longTermInvestments")),
                            self._safe_decimal(stmt.get("totalLiabilities")),
                            self._safe_decimal(stmt.get("totalCurrentLiabilities")),
                            self._safe_decimal(stmt.get("accountPayables")),
                            self._safe_decimal(stmt.get("shortTermDebt")),
                            self._safe_decimal(stmt.get("deferredRevenue")),
                            self._safe_decimal(stmt.get("otherCurrentLiabilities")),
                            self._safe_decimal(stmt.get("totalNonCurrentLiabilities")),
                            self._safe_decimal(stmt.get("longTermDebt")),
                            self._safe_decimal(stmt.get("deferredTaxLiabilitiesNonCurrent")),
                            self._safe_decimal(stmt.get("otherNonCurrentLiabilities")),
                            self._safe_decimal(stmt.get("totalDebt")),
                            self._safe_decimal(stmt.get("totalEquity")),
                            self._safe_decimal(stmt.get("commonStock")),
                            self._safe_decimal(stmt.get("retainedEarnings")),
                            self._safe_decimal(stmt.get("treasuryStock")),
                            self._safe_decimal(stmt.get("accumulatedOtherComprehensiveIncomeLoss")),
                            self._safe_decimal(stmt.get("totalStockholdersEquity")),
                            self._safe_decimal(stmt.get("minorityInterest")),
                            None,  # common_shares_outstanding - FMP balance sheet doesn't provide shares count
                            now,
                            json.dumps(stmt),
                        ),
                    )
                except Exception as e:
                    logger.error(f"Error saving balance sheet for {symbol}: {e}")

    def save_cash_flow_statements(
        self,
        symbol: str,
        statements: list[dict[str, Any]] | None,
        period: str,
        conn: duckdb.DuckDBPyConnection | None = None,
    ) -> None:
        """Save cash flow statements to database."""
        if not statements:
            return

        now = datetime.utcnow()

        ctx = self.db.transaction(conn) if conn is not None else self.db.get_connection()
        with ctx as db_conn:
            for stmt in statements:
                try:
                    db_conn.execute(
                        """
                        INSERT INTO cash_flow_statements (
                            symbol, fiscal_date, period, fiscal_year, reported_currency,
                            net_income, depreciation_amortization, deferred_income_tax,
                            stock_based_compensation, change_in_working_capital,
                            accounts_receivables, inventory, accounts_payables,
                            other_working_capital, other_non_cash_items, operating_cash_flow,
                            investments_in_ppe, acquisitions, purchases_of_investments,
                            sales_of_investments, other_investing, investing_cash_flow,
                            debt_repayment, common_stock_issued, common_stock_repurchased,
                            dividends_paid, other_financing, financing_cash_flow,
                            net_change_in_cash, cash_at_beginning, cash_at_end,
                            capital_expenditure, free_cash_flow, fetched_at, raw_json
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        ON CONFLICT (symbol, fiscal_date, period) DO UPDATE SET
                            operating_cash_flow = EXCLUDED.operating_cash_flow,
                            capital_expenditure = EXCLUDED.capital_expenditure,
                            free_cash_flow = EXCLUDED.free_cash_flow,
                            dividends_paid = EXCLUDED.dividends_paid,
                            fetched_at = EXCLUDED.fetched_at,
                            raw_json = EXCLUDED.raw_json
                        """,
                        (
                            symbol,
                            stmt.get("date"),
                            period,
                            self._safe_int(stmt.get("calendarYear")),
                            stmt.get("reportedCurrency"),
                            self._safe_decimal(stmt.get("netIncome")),
                            self._safe_decimal(stmt.get("depreciationAndAmortization")),
                            self._safe_decimal(stmt.get("deferredIncomeTax")),
                            self._safe_decimal(stmt.get("stockBasedCompensation")),
                            self._safe_decimal(stmt.get("changeInWorkingCapital")),
                            self._safe_decimal(stmt.get("accountsReceivables")),
                            self._safe_decimal(stmt.get("inventory")),
                            self._safe_decimal(stmt.get("accountsPayables")),
                            self._safe_decimal(stmt.get("otherWorkingCapital")),
                            self._safe_decimal(stmt.get("otherNonCashItems")),
                            self._safe_decimal(stmt.get("operatingCashFlow")),
                            self._safe_decimal(stmt.get("investmentsInPropertyPlantAndEquipment")),
                            self._safe_decimal(stmt.get("acquisitionsNet")),
                            self._safe_decimal(stmt.get("purchasesOfInvestments")),
                            self._safe_decimal(stmt.get("salesMaturitiesOfInvestments")),
                            self._safe_decimal(stmt.get("otherInvestingActivites")),
                            self._safe_decimal(stmt.get("netCashUsedForInvestingActivites")),
                            self._safe_decimal(stmt.get("debtRepayment")),
                            self._safe_decimal(stmt.get("commonStockIssued")),
                            self._safe_decimal(stmt.get("commonStockRepurchased")),
                            self._safe_decimal(stmt.get("dividendsPaid")),
                            self._safe_decimal(stmt.get("otherFinancingActivites")),
                            self._safe_decimal(stmt.get("netCashUsedProvidedByFinancingActivities")),
                            self._safe_decimal(stmt.get("netChangeInCash")),
                            self._safe_decimal(stmt.get("cashAtBeginningOfPeriod")),
                            self._safe_decimal(stmt.get("cashAtEndOfPeriod")),
                            self._safe_decimal(stmt.get("capitalExpenditure")),
                            self._safe_decimal(stmt.get("freeCashFlow")),
                            now,
                            json.dumps(stmt),
                        ),
                    )
                except Exception as e:
                    logger.error(f"Error saving cash flow for {symbol}: {e}")

    def save_key_metrics(
        self,
        symbol: str,
        metrics: list[dict[str, Any]] | None,
        period: str,
        conn: duckdb.DuckDBPyConnection | None = None,
    ) -> None:
        """Save key metrics to database."""
        if not metrics:
            return

        now = datetime.utcnow()

        ctx = self.db.transaction(conn) if conn is not None else self.db.get_connection()
        with ctx as db_conn:
            for m in metrics:
                try:
                    db_conn.execute(
                        """
                        INSERT INTO key_metrics (
                            symbol, fiscal_date, period,
                            revenue_per_share, net_income_per_share, operating_cash_flow_per_share,
                            free_cash_flow_per_share, cash_per_share, book_value_per_share,
                            tangible_book_value_per_share, shareholders_equity_per_share,
                            interest_debt_per_share, pe_ratio, price_to_sales, pb_ratio,
                            price_to_free_cash_flow, price_to_operating_cash_flow,
                            ev_to_sales, ev_to_ebitda, ev_to_operating_cash_flow,
                            ev_to_free_cash_flow, enterprise_value, roe, roa, roic,
                            return_on_tangible_assets, gross_profit_margin, operating_profit_margin,
                            net_profit_margin, current_ratio, quick_ratio, cash_ratio,
                            debt_ratio, debt_to_equity, debt_to_assets, net_debt_to_ebitda,
                            interest_coverage, asset_turnover, inventory_turnover,
                            receivables_turnover, payables_turnover, dividend_yield, payout_ratio,
                            fetched_at, raw_json
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        ON CONFLICT (symbol, fiscal_date, period) DO UPDATE SET
                            roic = EXCLUDED.roic,
                            roa = EXCLUDED.roa,
                            roe = EXCLUDED.roe,
                            current_ratio = EXCLUDED.current_ratio,
                            debt_to_equity = EXCLUDED.debt_to_equity,
                            fetched_at = EXCLUDED.fetched_at,
                            raw_json = EXCLUDED.raw_json
                        """,
                        (
                            symbol,
                            m.get("date"),
                            period,
                            self._safe_decimal(m.get("revenuePerShare")),
                            self._safe_decimal(m.get("netIncomePerShare")),
                            self._safe_decimal(m.get("operatingCashFlowPerShare")),
                            self._safe_decimal(m.get("freeCashFlowPerShare")),
                            self._safe_decimal(m.get("cashPerShare")),
                            self._safe_decimal(m.get("bookValuePerShare")),
                            self._safe_decimal(m.get("tangibleBookValuePerShare")),
                            self._safe_decimal(m.get("shareholdersEquityPerShare")),
                            self._safe_decimal(m.get("interestDebtPerShare")),
                            self._safe_decimal(m.get("peRatio")),
                            self._safe_decimal(m.get("priceToSalesRatio")),
                            self._safe_decimal(m.get("pbRatio")),
                            self._safe_decimal(m.get("priceToFreeCashFlowRatio")),
                            self._safe_decimal(m.get("priceToOperatingCashFlowRatio")),
                            self._safe_decimal(m.get("evToSales")),
                            self._safe_decimal(m.get("evToEBITDA")),
                            self._safe_decimal(m.get("evToOperatingCashFlow")),
                            self._safe_decimal(m.get("evToFreeCashFlow")),
                            self._safe_decimal(m.get("enterpriseValue")),
                            self._safe_decimal(m.get("returnOnEquity")),
                            self._safe_decimal(m.get("returnOnAssets")),
                            self._safe_decimal(m.get("returnOnInvestedCapital")),
                            self._safe_decimal(m.get("returnOnTangibleAssets")),
                            self._safe_decimal(m.get("grossProfitMargin")),
                            self._safe_decimal(m.get("operatingProfitMargin")),
                            self._safe_decimal(m.get("netProfitMargin")),
                            self._safe_decimal(m.get("currentRatio")),
                            self._safe_decimal(m.get("quickRatio")),
                            self._safe_decimal(m.get("cashRatio")),
                            self._safe_decimal(m.get("debtRatio")),
                            self._safe_decimal(m.get("debtEquityRatio")),
                            self._safe_decimal(m.get("debtToAssets")),
                            self._safe_decimal(m.get("netDebtToEBITDA")),
                            self._safe_decimal(m.get("interestCoverage")),
                            self._safe_decimal(m.get("assetTurnover")),
                            self._safe_decimal(m.get("inventoryTurnover")),
                            self._safe_decimal(m.get("receivablesTurnover")),
                            self._safe_decimal(m.get("payablesTurnover")),
                            self._safe_decimal(m.get("dividendYield")),
                            self._safe_decimal(m.get("payoutRatio")),
                            now,
                            json.dumps(m),
                        ),
                    )
                except Exception as e:
                    logger.error(f"Error saving key metrics for {symbol}: {e}")

    def save_ratios(
        self,
        symbol: str,
        ratios: list[dict[str, Any]] | None,
        period: str,
        conn: duckdb.DuckDBPyConnection | None = None,
    ) -> None:
        """Save ratios data to key_metrics table (updates existing rows).

        The /ratios endpoint provides PE, PB, margins, turnover ratios, etc.
        that are not available in the /key-metrics endpoint.
        This updates existing key_metrics rows with the additional ratio data.
        """
        if not ratios:
            return

        ctx = self.db.transaction(conn) if conn is not None else self.db.get_connection()
        with ctx as db_conn:
            for r in ratios:
                try:
                    db_conn.execute(
                        """
                        UPDATE key_metrics SET
                            pe_ratio = COALESCE(?, pe_ratio),
                            pb_ratio = COALESCE(?, pb_ratio),
                            price_to_sales = COALESCE(?, price_to_sales),
                            price_to_free_cash_flow = COALESCE(?, price_to_free_cash_flow),
                            price_to_operating_cash_flow = COALESCE(?, price_to_operating_cash_flow),
                            gross_profit_margin = COALESCE(?, gross_profit_margin),
                            operating_profit_margin = COALESCE(?, operating_profit_margin),
                            net_profit_margin = COALESCE(?, net_profit_margin),
                            quick_ratio = COALESCE(?, quick_ratio),
                            cash_ratio = COALESCE(?, cash_ratio),
                            debt_ratio = COALESCE(?, debt_ratio),
                            debt_to_equity = COALESCE(?, debt_to_equity),
                            debt_to_assets = COALESCE(?, debt_to_assets),
                            interest_coverage = COALESCE(?, interest_coverage),
                            asset_turnover = COALESCE(?, asset_turnover),
                            inventory_turnover = COALESCE(?, inventory_turnover),
                            receivables_turnover = COALESCE(?, receivables_turnover),
                            payables_turnover = COALESCE(?, payables_turnover),
                            dividend_yield = COALESCE(?, dividend_yield),
                            payout_ratio = COALESCE(?, payout_ratio),
                            revenue_per_share = COALESCE(?, revenue_per_share),
                            net_income_per_share = COALESCE(?, net_income_per_share),
                            operating_cash_flow_per_share = COALESCE(?, operating_cash_flow_per_share),
                            free_cash_flow_per_share = COALESCE(?, free_cash_flow_per_share),
                            cash_per_share = COALESCE(?, cash_per_share),
                            book_value_per_share = COALESCE(?, book_value_per_share),
                            tangible_book_value_per_share = COALESCE(?, tangible_book_value_per_share),
                            shareholders_equity_per_share = COALESCE(?, shareholders_equity_per_share),
                            interest_debt_per_share = COALESCE(?, interest_debt_per_share)
                        WHERE symbol = ? AND fiscal_date = ? AND period = ?
                        """,
                        (
                            self._safe_decimal(r.get("priceToEarningsRatio")),
                            self._safe_decimal(r.get("priceToBookRatio")),
                            self._safe_decimal(r.get("priceToSalesRatio")),
                            self._safe_decimal(r.get("priceToFreeCashFlowRatio")),
                            self._safe_decimal(r.get("priceToOperatingCashFlowRatio")),
                            self._safe_decimal(r.get("grossProfitMargin")),
                            self._safe_decimal(r.get("operatingProfitMargin")),
                            self._safe_decimal(r.get("netProfitMargin")),
                            self._safe_decimal(r.get("quickRatio")),
                            self._safe_decimal(r.get("cashRatio")),
                            self._safe_decimal(r.get("debtToCapitalRatio")),
                            self._safe_decimal(r.get("debtToEquityRatio")),
                            self._safe_decimal(r.get("debtToAssetsRatio")),
                            self._safe_decimal(r.get("interestCoverageRatio")),
                            self._safe_decimal(r.get("assetTurnover")),
                            self._safe_decimal(r.get("inventoryTurnover")),
                            self._safe_decimal(r.get("receivablesTurnover")),
                            self._safe_decimal(r.get("payablesTurnover")),
                            self._safe_decimal(r.get("dividendYield")),
                            self._safe_decimal(r.get("dividendPayoutRatio")),
                            self._safe_decimal(r.get("revenuePerShare")),
                            self._safe_decimal(r.get("netIncomePerShare")),
                            self._safe_decimal(r.get("operatingCashFlowPerShare")),
                            self._safe_decimal(r.get("freeCashFlowPerShare")),
                            self._safe_decimal(r.get("cashPerShare")),
                            self._safe_decimal(r.get("bookValuePerShare")),
                            self._safe_decimal(r.get("tangibleBookValuePerShare")),
                            self._safe_decimal(r.get("shareholdersEquityPerShare")),
                            self._safe_decimal(r.get("interestDebtPerShare")),
                            symbol,
                            r.get("date"),
                            period,
                        ),
                    )
                except Exception as e:
                    logger.error(f"Error saving ratios for {symbol}: {e}")

    def save_dividends(
        self,
        symbol: str,
        dividend_data: list[dict[str, Any]] | dict[str, Any] | None,
        conn: duckdb.DuckDBPyConnection | None = None,
    ) -> None:
        """Save dividend history to database."""
        if not dividend_data:
            return

        # Handle both list format (new API) and dict format (old API with "historical" key)
        if isinstance(dividend_data, dict):
            dividends = dividend_data.get("historical", [])
        else:
            dividends = dividend_data

        if not dividends:
            return

        now = datetime.utcnow()
        saved_count = 0
        skipped_count = 0

        ctx = self.db.transaction(conn) if conn is not None else self.db.get_connection()
        with ctx as db_conn:
            for div in dividends:
                # ex_date (date field) is required for primary key - skip if missing/empty
                ex_date = self._safe_date(div.get("date"))
                if not ex_date:
                    skipped_count += 1
                    continue

                try:
                    db_conn.execute(
                        """
                        INSERT INTO dividends (
                            symbol, ex_date, declaration_date, record_date,
                            payment_date, amount, adjusted_amount, fetched_at, raw_json
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                        ON CONFLICT (symbol, ex_date) DO UPDATE SET
                            amount = EXCLUDED.amount,
                            adjusted_amount = EXCLUDED.adjusted_amount,
                            fetched_at = EXCLUDED.fetched_at,
                            raw_json = EXCLUDED.raw_json
                        """,
                        (
                            symbol,
                            ex_date,
                            self._safe_date(div.get("declarationDate")),
                            self._safe_date(div.get("recordDate")),
                            self._safe_date(div.get("paymentDate")),
                            self._safe_decimal(div.get("dividend")),
                            self._safe_decimal(div.get("adjDividend")),
                            now,
                            json.dumps(div),
                        ),
                    )
                    saved_count += 1
                except Exception as e:
                    logger.error(f"Error saving dividend for {symbol}: {e}")

        if skipped_count > 0:
            logger.debug(f"Skipped {skipped_count} dividends for {symbol} (missing date)")

    def save_endpoint_data(
        self,
        symbol: str,
        endpoint: str,
        data: list[dict[str, Any]] | None,
        conn: duckdb.DuckDBPyConnection | None = None,
    ) -> None:
        """Save data for a single endpoint.

        Args:
            symbol: Stock ticker symbol.
            endpoint: Endpoint name (profile, income_annual, etc.).
            data: Data returned from the API.
            conn: Optional existing database connection to reuse.
        """
        if endpoint == "profile":
            self.save_profile(symbol, data, conn)
        elif endpoint == "income_annual":
            self.save_income_statements(symbol, data, "annual", conn)
        elif endpoint == "income_quarterly":
            self.save_income_statements(symbol, data, "quarter", conn)
        elif endpoint == "balance_annual":
            self.save_balance_sheets(symbol, data, "annual", conn)
        elif endpoint == "balance_quarterly":
            self.save_balance_sheets(symbol, data, "quarter", conn)
        elif endpoint == "cashflow_annual":
            self.save_cash_flow_statements(symbol, data, "annual", conn)
        elif endpoint == "cashflow_quarterly":
            self.save_cash_flow_statements(symbol, data, "quarter", conn)
        elif endpoint == "metrics_annual":
            self.save_key_metrics(symbol, data, "annual", conn)
        elif endpoint == "metrics_quarterly":
            self.save_key_metrics(symbol, data, "quarter", conn)
        elif endpoint == "ratios_annual":
            self.save_ratios(symbol, data, "annual", conn)
        elif endpoint == "ratios_quarterly":
            self.save_ratios(symbol, data, "quarter", conn)
        elif endpoint == "dividends":
            self.save_dividends(symbol, data, conn)

    def save_all_data(self, data: dict[str, Any]) -> None:
        """Save all fetched data to database.

        Args:
            data: Dictionary with all data types from fetch_all_data.
        """
        symbol = data["symbol"]

        self.save_profile(symbol, data.get("profile"))
        self.save_income_statements(symbol, data.get("income_statement_annual"), "annual")
        self.save_income_statements(symbol, data.get("income_statement_quarterly"), "quarter")
        self.save_balance_sheets(symbol, data.get("balance_sheet_annual"), "annual")
        self.save_balance_sheets(symbol, data.get("balance_sheet_quarterly"), "quarter")
        self.save_cash_flow_statements(symbol, data.get("cash_flow_annual"), "annual")
        self.save_cash_flow_statements(symbol, data.get("cash_flow_quarterly"), "quarter")
        self.save_key_metrics(symbol, data.get("key_metrics_annual"), "annual")
        self.save_key_metrics(symbol, data.get("key_metrics_quarterly"), "quarter")
        self.save_dividends(symbol, data.get("dividends"))

    async def _worker(
        self,
        queue: asyncio.Queue,
        stats: dict[str, int],
        stats_lock: asyncio.Lock,
        progress: Progress | None,
        task_id: int | None,
        worker_id: int,
    ) -> None:
        """Worker coroutine that processes tasks from the queue.

        Args:
            queue: Task queue to pull from.
            stats: Shared stats dictionary.
            stats_lock: Lock for updating stats.
            progress: Optional progress bar.
            task_id: Optional progress task ID.
            worker_id: Worker identifier for logging.
        """
        # Create a single DB connection for this worker to reuse
        db_conn = self.db.connect()
        try:
            while True:
                try:
                    fetch_task: FetchTask = queue.get_nowait()
                except asyncio.QueueEmpty:
                    break

                try:
                    # Fetch the data
                    data = await self.client.fetch_endpoint(fetch_task.symbol, fetch_task.endpoint)

                    # Save to database (reusing connection)
                    self.save_endpoint_data(fetch_task.symbol, fetch_task.endpoint, data, db_conn)

                    # Mark as completed (checkpoint logic unchanged)
                    await self.checkpoint.mark_endpoint_completed(fetch_task.symbol, fetch_task.endpoint)

                    async with stats_lock:
                        stats["completed"] += 1

                except Exception as e:
                    logger.error(f"Worker {worker_id}: {fetch_task.symbol}/{fetch_task.endpoint} failed: {e}")
                    await self.checkpoint.mark_endpoint_failed(fetch_task.symbol, fetch_task.endpoint, str(e))
                    async with stats_lock:
                        stats["failed"] += 1

                finally:
                    queue.task_done()
                    if progress and task_id is not None:
                        progress.update(task_id, advance=1)
        finally:
            # Close the connection when worker is done
            db_conn.close()

    async def fetch_all(
        self,
        symbols: list[str],
        include_quarterly: bool = True,
        skip_completed: bool = True,
        show_progress: bool = True,
    ) -> dict[str, int]:
        """Fetch data for all symbols using worker pool for high throughput.

        Args:
            symbols: List of ticker symbols to fetch.
            include_quarterly: Whether to include quarterly data.
            skip_completed: Whether to skip already-completed endpoints.
            show_progress: Whether to show progress bar.

        Returns:
            Dictionary with counts of completed and failed endpoint fetches.
        """
        settings = get_settings()
        num_workers = settings.NUM_WORKERS

        self.checkpoint.start_session()

        # Build task queue, skipping completed endpoints
        queue: asyncio.Queue[FetchTask] = asyncio.Queue()
        total_tasks = 0
        skipped_tasks = 0

        endpoints_to_use = ALL_ENDPOINTS if include_quarterly else [
            e for e in ALL_ENDPOINTS if "quarterly" not in e
        ]

        for symbol in symbols:
            for endpoint in endpoints_to_use:
                if skip_completed and self.checkpoint.is_endpoint_completed(symbol, endpoint):
                    skipped_tasks += 1
                    continue
                await queue.put(FetchTask(symbol, endpoint))
                total_tasks += 1

        stats = {"completed": 0, "failed": 0, "skipped": skipped_tasks}
        stats_lock = asyncio.Lock()

        if total_tasks == 0:
            console.print("[green]All endpoints already completed![/green]")
            return stats

        console.print(f"[cyan]Fetching {total_tasks:,} endpoints with {num_workers} workers[/cyan]")
        console.print(f"[dim]Skipped {skipped_tasks:,} already-completed endpoints[/dim]")
        console.print(f"[dim]Target rate: {settings.RATE_LIMIT_PER_MINUTE} req/min[/dim]")

        start_time = time.monotonic()

        if show_progress:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                TimeElapsedColumn(),
            ) as progress:
                task_id = progress.add_task(
                    f"Fetching ({num_workers} workers)...",
                    total=total_tasks
                )

                # Create and run workers
                workers = [
                    asyncio.create_task(
                        self._worker(queue, stats, stats_lock, progress, task_id, i)
                    )
                    for i in range(num_workers)
                ]
                await asyncio.gather(*workers)
        else:
            workers = [
                asyncio.create_task(
                    self._worker(queue, stats, stats_lock, None, None, i)
                )
                for i in range(num_workers)
            ]
            await asyncio.gather(*workers)

        # Force save checkpoint at end
        self.checkpoint.force_save()

        elapsed = time.monotonic() - start_time
        actual_rate = (stats["completed"] + stats["failed"]) / (elapsed / 60) if elapsed > 0 else 0

        console.print()
        console.print(f"[bold green]Fetch complete![/bold green]")
        console.print(f"  Completed: {stats['completed']:,}")
        console.print(f"  Failed: {stats['failed']:,}")
        console.print(f"  Skipped: {stats['skipped']:,}")
        console.print(f"  Time: {elapsed:.1f}s")
        console.print(f"  Actual rate: {actual_rate:.0f} req/min")

        return stats

    @staticmethod
    def _safe_decimal(value: Any, clamp: bool = True) -> float | None:
        """Safely convert value to decimal/float.

        Args:
            value: Value to convert
            clamp: If True, clamp to DECIMAL(18,3) range to avoid overflow
        """
        if value is None:
            return None
        try:
            result = float(value)
            # DECIMAL(18,3) max is ~999,999,999,999,999.999
            if clamp and abs(result) > 999_999_999_999_999:
                return None  # Skip absurd values from penny stocks
            return result
        except (ValueError, TypeError):
            return None

    @staticmethod
    def _safe_int(value: Any) -> int | None:
        """Safely convert value to integer."""
        if value is None:
            return None
        try:
            return int(float(value))
        except (ValueError, TypeError):
            return None

    @staticmethod
    def _safe_date(value: Any) -> str | None:
        """Safely convert value to date string, returning None for empty/invalid values."""
        if value is None:
            return None
        if isinstance(value, str):
            # Return None for empty strings
            value = value.strip()
            if not value:
                return None
            # Basic validation: should be YYYY-MM-DD format
            if len(value) >= 10 and value[4] == "-" and value[7] == "-":
                return value
            return None
        return None

    @staticmethod
    def _get_current_quarter() -> str:
        """Get current fiscal quarter string (e.g., '2024Q4')."""
        now = datetime.utcnow()
        quarter = (now.month - 1) // 3 + 1
        return f"{now.year}Q{quarter}"
