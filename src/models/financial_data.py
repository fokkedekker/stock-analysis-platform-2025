"""Pydantic models for financial data from FMP API."""

from datetime import date, datetime
from typing import Any

from pydantic import BaseModel, Field


class Ticker(BaseModel):
    """Stock ticker metadata."""

    symbol: str
    name: str | None = None
    exchange: str | None = None
    sector: str | None = None
    industry: str | None = None
    asset_type: str | None = None
    is_active: bool = True
    created_at: datetime | None = None
    updated_at: datetime | None = None


class CompanyProfile(BaseModel):
    """Company profile with current market data."""

    symbol: str
    fiscal_quarter: str
    price: float | None = None
    market_cap: float | None = None
    pe_ratio: float | None = None
    pb_ratio: float | None = None
    beta: float | None = None
    shares_outstanding: int | None = None
    avg_volume: int | None = None
    dividend_yield: float | None = None
    fetched_at: datetime | None = None


class IncomeStatement(BaseModel):
    """Income statement record."""

    symbol: str
    fiscal_date: date
    period: str  # 'annual' or 'quarter'
    fiscal_year: int | None = None
    reported_currency: str | None = None

    # Revenue and profit
    revenue: float | None = None
    cost_of_revenue: float | None = None
    gross_profit: float | None = None
    gross_profit_ratio: float | None = None

    # Operating expenses
    research_and_development: float | None = None
    selling_general_admin: float | None = None
    operating_expenses: float | None = None
    operating_income: float | None = None
    operating_income_ratio: float | None = None

    # Interest and earnings
    interest_expense: float | None = None
    ebit: float | None = None
    ebitda: float | None = None
    income_before_tax: float | None = None
    income_tax_expense: float | None = None

    # Net income
    net_income: float | None = None
    net_income_ratio: float | None = None
    eps: float | None = None
    eps_diluted: float | None = None
    weighted_avg_shares: int | None = None
    weighted_avg_shares_diluted: int | None = None

    fetched_at: datetime | None = None


class BalanceSheet(BaseModel):
    """Balance sheet record."""

    symbol: str
    fiscal_date: date
    period: str
    fiscal_year: int | None = None
    reported_currency: str | None = None

    # Assets
    total_assets: float | None = None
    current_assets: float | None = None
    cash_and_equivalents: float | None = None
    short_term_investments: float | None = None
    net_receivables: float | None = None
    inventory: float | None = None
    other_current_assets: float | None = None
    non_current_assets: float | None = None
    property_plant_equipment: float | None = None
    goodwill: float | None = None
    intangible_assets: float | None = None
    long_term_investments: float | None = None

    # Liabilities
    total_liabilities: float | None = None
    current_liabilities: float | None = None
    accounts_payable: float | None = None
    short_term_debt: float | None = None
    deferred_revenue: float | None = None
    other_current_liabilities: float | None = None
    non_current_liabilities: float | None = None
    long_term_debt: float | None = None
    deferred_tax_liabilities: float | None = None
    other_non_current_liabilities: float | None = None
    total_debt: float | None = None

    # Equity
    total_equity: float | None = None
    common_stock: float | None = None
    retained_earnings: float | None = None
    treasury_stock: float | None = None
    other_comprehensive_income: float | None = None
    total_stockholders_equity: float | None = None
    minority_interest: float | None = None
    common_shares_outstanding: int | None = None

    fetched_at: datetime | None = None


class CashFlowStatement(BaseModel):
    """Cash flow statement record."""

    symbol: str
    fiscal_date: date
    period: str
    fiscal_year: int | None = None
    reported_currency: str | None = None

    # Operating activities
    net_income: float | None = None
    depreciation_amortization: float | None = None
    deferred_income_tax: float | None = None
    stock_based_compensation: float | None = None
    change_in_working_capital: float | None = None
    accounts_receivables: float | None = None
    inventory: float | None = None
    accounts_payables: float | None = None
    other_working_capital: float | None = None
    other_non_cash_items: float | None = None
    operating_cash_flow: float | None = None

    # Investing activities
    investments_in_ppe: float | None = None
    acquisitions: float | None = None
    purchases_of_investments: float | None = None
    sales_of_investments: float | None = None
    other_investing: float | None = None
    investing_cash_flow: float | None = None

    # Financing activities
    debt_repayment: float | None = None
    common_stock_issued: float | None = None
    common_stock_repurchased: float | None = None
    dividends_paid: float | None = None
    other_financing: float | None = None
    financing_cash_flow: float | None = None

    # Summary
    net_change_in_cash: float | None = None
    cash_at_beginning: float | None = None
    cash_at_end: float | None = None
    capital_expenditure: float | None = None
    free_cash_flow: float | None = None

    fetched_at: datetime | None = None


class KeyMetrics(BaseModel):
    """Key financial metrics record."""

    symbol: str
    fiscal_date: date
    period: str

    # Per share metrics
    revenue_per_share: float | None = None
    net_income_per_share: float | None = None
    operating_cash_flow_per_share: float | None = None
    free_cash_flow_per_share: float | None = None
    cash_per_share: float | None = None
    book_value_per_share: float | None = None
    tangible_book_value_per_share: float | None = None
    shareholders_equity_per_share: float | None = None
    interest_debt_per_share: float | None = None

    # Valuation ratios
    pe_ratio: float | None = None
    price_to_sales: float | None = None
    pb_ratio: float | None = None
    price_to_free_cash_flow: float | None = None
    price_to_operating_cash_flow: float | None = None
    ev_to_sales: float | None = None
    ev_to_ebitda: float | None = None
    ev_to_operating_cash_flow: float | None = None
    ev_to_free_cash_flow: float | None = None
    enterprise_value: float | None = None

    # Profitability
    roe: float | None = None
    roa: float | None = None
    roic: float | None = None
    return_on_tangible_assets: float | None = None
    gross_profit_margin: float | None = None
    operating_profit_margin: float | None = None
    net_profit_margin: float | None = None

    # Liquidity
    current_ratio: float | None = None
    quick_ratio: float | None = None
    cash_ratio: float | None = None

    # Leverage
    debt_ratio: float | None = None
    debt_to_equity: float | None = None
    debt_to_assets: float | None = None
    net_debt_to_ebitda: float | None = None
    interest_coverage: float | None = None

    # Efficiency
    asset_turnover: float | None = None
    inventory_turnover: float | None = None
    receivables_turnover: float | None = None
    payables_turnover: float | None = None

    # Dividends
    dividend_yield: float | None = None
    payout_ratio: float | None = None

    fetched_at: datetime | None = None


class Dividend(BaseModel):
    """Dividend record."""

    symbol: str
    ex_date: date
    declaration_date: date | None = None
    record_date: date | None = None
    payment_date: date | None = None
    amount: float | None = None
    adjusted_amount: float | None = None
    fetched_at: datetime | None = None
