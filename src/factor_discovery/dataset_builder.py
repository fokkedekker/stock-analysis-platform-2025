"""Dataset builder for factor discovery analysis.

Builds a flat dataset of (stock, quarter, holding_period, alpha, metrics)
for statistical analysis.
"""

import json
import logging
from decimal import Decimal
from typing import Any

from src.database.connection import get_db_manager

logger = logging.getLogger(__name__)


def _to_float(value: Any) -> float | None:
    """Convert various numeric types to float."""
    if value is None:
        return None
    if isinstance(value, Decimal):
        return float(value)
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _add_quarters(quarter: str, periods: int) -> str:
    """Add N quarters to a quarter string."""
    year = int(quarter[:4])
    q = int(quarter[-1])

    total_quarters = year * 4 + q + periods
    new_year = (total_quarters - 1) // 4
    new_q = ((total_quarters - 1) % 4) + 1

    return f"{new_year}Q{new_q}"


def _subtract_quarters(quarter: str, periods: int) -> str:
    """Subtract N quarters from a quarter string."""
    return _add_quarters(quarter, -periods)


def _compare_quarters(q1: str, q2: str) -> int:
    """Compare two quarters. Returns -1 if q1 < q2, 0 if equal, 1 if q1 > q2."""
    if q1 == q2:
        return 0
    return -1 if q1 < q2 else 1


def split_data_by_quarters(
    data: list[dict],
    train_end_quarter: str | None = None,
    validation_end_quarter: str | None = None,
) -> tuple[list[dict], list[dict], list[dict]]:
    """
    Split dataset into train, validation, and test sets by quarter.

    Args:
        data: List of observation dicts (must have 'buy_quarter' key)
        train_end_quarter: Last quarter for training (e.g., '2022Q4')
        validation_end_quarter: Last quarter for validation (e.g., '2023Q4')

    Returns:
        Tuple of (train_data, validation_data, test_data)

    If no splits specified, returns (all_data, [], [])
    """
    if not train_end_quarter:
        # No split requested - use all data for training
        return data, [], []

    train_data = []
    validation_data = []
    test_data = []

    for obs in data:
        buy_q = obs.get("buy_quarter", "")

        if buy_q <= train_end_quarter:
            train_data.append(obs)
        elif validation_end_quarter and buy_q <= validation_end_quarter:
            validation_data.append(obs)
        else:
            # Everything after validation_end (or train_end if no validation)
            if validation_end_quarter:
                test_data.append(obs)
            else:
                validation_data.append(obs)

    logger.info(
        f"Data split: train={len(train_data)}, "
        f"validation={len(validation_data)}, test={len(test_data)}"
    )

    return train_data, validation_data, test_data


def get_valid_buy_quarters(
    holding_period: int,
    available_quarters: list[str],
    exclude_latest: bool = True,
) -> list[str]:
    """
    Return quarters where we can calculate complete returns.

    Args:
        holding_period: Number of quarters to hold
        available_quarters: All quarters with price data (sorted)
        exclude_latest: If True, exclude the latest quarter (for trading)

    Returns:
        List of valid buy quarters
    """
    if not available_quarters:
        return []

    # Sort quarters
    sorted_quarters = sorted(available_quarters)

    # Exclude latest quarter if requested (we'll trade on it)
    if exclude_latest and len(sorted_quarters) > 1:
        analysis_quarters = sorted_quarters[:-1]
    else:
        analysis_quarters = sorted_quarters

    available_set = set(available_quarters)
    valid = []

    for buy_q in analysis_quarters:
        sell_q = _add_quarters(buy_q, holding_period)
        if sell_q in available_set:
            valid.append(buy_q)

    return valid


class DatasetBuilder:
    """
    Builds the analysis dataset from existing tables.

    Creates a flat list of dicts, each containing:
    - symbol, buy_quarter, holding_period
    - alpha (stock return - SPY return)
    - All metrics from analysis tables
    - Boolean flags for quality tags
    """

    # Quality tags to track as boolean fields
    QUALITY_TAGS = [
        "Durable Compounder",
        "Cash Machine",
        "Deep Value",
        "Heavy Reinvestor",
        "Premium Priced",
        "Volatile Returns",
        "Weak Moat Signal",
        "Earnings Quality Concern",
    ]

    def __init__(
        self,
        quarters: list[str],
        holding_periods: list[int],
        exclusions: dict | None = None,
        data_lag_quarters: int = 1,
    ):
        """
        Initialize builder.

        Args:
            quarters: List of quarters to analyze (e.g., ['2024Q1', '2024Q2'])
            holding_periods: List of holding periods (e.g., [1, 2, 3, 4])
            exclusions: Optional exclusion filters dict with keys:
                - exclude_altman_zones: list of zones to exclude
                - min_piotroski: minimum score (exclude below)
                - exclude_quality_tags: tags to exclude
                - require_quality_tags: tags to require
                - exclude_penny_stocks: bool
                - exclude_negative_earnings: bool
            data_lag_quarters: Quarters to lag analysis data (1 = use Q1 data for Q2 decisions).
                              This prevents look-ahead bias since earnings aren't released
                              until after the quarter ends. Default is 1 (conservative).
        """
        self.quarters = quarters
        self.holding_periods = holding_periods
        self.exclusions = exclusions or {}
        self.data_lag_quarters = data_lag_quarters

    def build(self) -> dict:
        """
        Build the complete dataset.

        Returns:
            Dict with structure:
            {
                "data": [list of observation dicts],
                "metadata": {
                    "total_rows": int,
                    "quarters": list,
                    "holding_periods": list,
                    "latest_quarter": str,
                    "valid_buy_quarters": {holding_period: [quarters]}
                }
            }
        """
        db = get_db_manager()
        observations = []

        with db.get_connection() as conn:
            logger.info(f"Building dataset for {len(self.quarters)} quarters...")

            # 1. Load all available price quarters
            available_quarters = self._get_available_quarters(conn)
            logger.info(f"Found {len(available_quarters)} quarters with price data")

            # 2. Determine valid buy quarters for each holding period
            valid_buy_quarters = {}
            for hp in self.holding_periods:
                valid_buy_quarters[hp] = get_valid_buy_quarters(
                    hp, available_quarters, exclude_latest=True
                )
                logger.info(
                    f"Holding period {hp}Q: {len(valid_buy_quarters[hp])} valid buy quarters"
                )

            # 3. Load SPY prices
            spy_prices = self._load_spy_prices(conn)
            logger.info(f"Loaded {len(spy_prices)} SPY prices")

            # 4. Load stock prices for all quarters
            price_data = self._load_prices(conn, available_quarters)

            # 5. For each quarter, load analysis data and build observations
            # Cache analysis data by quarter to avoid re-querying
            analysis_cache: dict[str, dict] = {}

            for quarter in self.quarters:
                if quarter not in available_quarters:
                    logger.warning(f"Skipping quarter {quarter} - no price data")
                    continue

                # Apply data lag: use analysis from N quarters prior
                # This prevents look-ahead bias (can't use Q2 earnings to make Q2 buy decision)
                analysis_quarter = _subtract_quarters(quarter, self.data_lag_quarters)

                # Load analysis results (with caching)
                if analysis_quarter not in analysis_cache:
                    analysis_cache[analysis_quarter] = self._query_unified_analysis(
                        conn, analysis_quarter
                    )
                analysis_data = analysis_cache[analysis_quarter]

                if self.data_lag_quarters > 0:
                    logger.debug(
                        f"Buy quarter {quarter}: using analysis from {analysis_quarter} "
                        f"(lag={self.data_lag_quarters}), {len(analysis_data)} stocks"
                    )
                else:
                    logger.debug(f"Loaded {len(analysis_data)} stocks for {quarter}")

                # Build observations for each holding period
                for hp in self.holding_periods:
                    if quarter not in valid_buy_quarters.get(hp, []):
                        continue

                    sell_quarter = _add_quarters(quarter, hp)
                    spy_return = self._calculate_spy_return(spy_prices, quarter, sell_quarter)

                    for symbol, data in analysis_data.items():
                        # Calculate stock return
                        buy_price = price_data.get(quarter, {}).get(symbol)
                        sell_price = price_data.get(sell_quarter, {}).get(symbol)

                        if not buy_price or not sell_price or buy_price <= 0:
                            continue

                        stock_return = ((sell_price - buy_price) / buy_price) * 100
                        alpha = stock_return - spy_return

                        # Check exclusion filters
                        if self._should_exclude(data, buy_price):
                            continue

                        # Build observation
                        obs = self._build_observation(
                            symbol=symbol,
                            buy_quarter=quarter,
                            holding_period=hp,
                            stock_return=stock_return,
                            spy_return=spy_return,
                            alpha=alpha,
                            data=data,
                        )
                        observations.append(obs)

            logger.info(f"Built dataset with {len(observations)} observations")

        # Calculate latest quarter for metadata
        latest_quarter = sorted(available_quarters)[-1] if available_quarters else None

        return {
            "data": observations,
            "metadata": {
                "total_rows": len(observations),
                "quarters": self.quarters,
                "holding_periods": self.holding_periods,
                "latest_quarter": latest_quarter,
                "valid_buy_quarters": valid_buy_quarters,
                "data_lag_quarters": self.data_lag_quarters,
            },
        }

    def _get_available_quarters(self, conn) -> list[str]:
        """Get all quarters with price data."""
        result = conn.execute(
            """
            SELECT DISTINCT fiscal_quarter
            FROM company_profiles
            WHERE price IS NOT NULL
            ORDER BY fiscal_quarter
            """
        ).fetchall()
        return [row[0] for row in result]

    def _load_spy_prices(self, conn) -> dict[str, float]:
        """Load SPY prices from database."""
        spy_prices = {}
        try:
            result = conn.execute(
                """
                SELECT quarter, price
                FROM spy_prices
                ORDER BY quarter
                """
            ).fetchall()

            for quarter, price in result:
                spy_prices[quarter] = _to_float(price)

        except Exception as e:
            logger.warning(f"Could not load SPY prices: {e}")

        return spy_prices

    def _load_prices(self, conn, quarters: list[str]) -> dict[str, dict[str, float]]:
        """Load all stock prices for given quarters."""
        price_data = {}
        for quarter in quarters:
            result = conn.execute(
                """
                SELECT symbol, price
                FROM company_profiles
                WHERE fiscal_quarter = ?
                  AND price IS NOT NULL
                  AND price > 0
                """,
                (quarter,),
            ).fetchall()
            price_data[quarter] = {row[0]: _to_float(row[1]) for row in result if row[1]}
        return price_data

    def _calculate_spy_return(
        self,
        spy_prices: dict[str, float],
        buy_quarter: str,
        sell_quarter: str,
    ) -> float:
        """Calculate SPY return between quarters."""
        buy_price = spy_prices.get(buy_quarter)
        sell_price = spy_prices.get(sell_quarter)

        if buy_price and sell_price and buy_price > 0:
            return ((sell_price - buy_price) / buy_price) * 100

        # Default benchmark return if SPY data not available
        return 2.5  # ~2.5% per quarter average

    def _query_unified_analysis(self, conn, quarter: str) -> dict[str, dict]:
        """
        Query all analysis results for a quarter into a unified structure.

        Returns dict: symbol -> {all analysis fields}
        """
        # Convert quarter to date range for key_metrics matching
        # e.g., "2024Q1" -> year=2024, q=1
        year = int(quarter[:4])
        q = int(quarter[-1])

        # Calculate the end date of the quarter for point-in-time matching
        if q == 1:
            quarter_end = f"{year}-03-31"
        elif q == 2:
            quarter_end = f"{year}-06-30"
        elif q == 3:
            quarter_end = f"{year}-09-30"
        else:  # q == 4
            quarter_end = f"{year}-12-31"

        result = conn.execute(
            """
            SELECT
                t.symbol,
                t.name,
                t.sector,

                -- Altman
                a.z_score as altman_z_score,
                a.zone as altman_zone,

                -- Piotroski
                p.f_score as piotroski_score,

                -- ROIC/Quality (existing + new stability metrics)
                r.roic,
                r.free_cash_flow,
                r.fcf_positive_5yr,
                r.quality_tags,
                r.roic_std_dev,
                r.gross_margin_std_dev,
                r.fcf_to_net_income,
                r.reinvestment_rate,
                r.fcf_yield,

                -- Graham
                g.criteria_passed as graham_score,
                g.pe_ratio as graham_pe,
                g.pb_ratio as graham_pb,

                -- Net-Net
                nn.trading_below_ncav,
                nn.discount_to_ncav as net_net_discount,

                -- PEG (existing + new growth metrics)
                gp.peg_ratio,
                gp.eps_cagr,
                gp.peg_pass,
                gp.eps_growth_1yr,
                gp.eps_growth_3yr,
                gp.eps_growth_5yr,

                -- Magic Formula
                mf.combined_rank as magic_formula_rank,
                mf.earnings_yield,
                mf.return_on_capital as mf_roic,

                -- Fama-French
                ff.book_to_market_percentile,
                ff.profitability_percentile,

                -- Key Metrics (raw financial ratios)
                km.pe_ratio,
                km.pb_ratio,
                km.price_to_sales,
                km.price_to_free_cash_flow,
                km.price_to_operating_cash_flow,
                km.ev_to_ebitda,
                km.ev_to_sales,
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
                AND a.analysis_quarter = ?
            LEFT JOIN piotroski_results p ON t.symbol = p.symbol
                AND p.analysis_quarter = ?
            LEFT JOIN roic_quality_results r ON t.symbol = r.symbol
                AND r.analysis_quarter = ?
            LEFT JOIN graham_results g ON t.symbol = g.symbol
                AND g.analysis_quarter = ?
                AND g.mode = 'modern'
            LEFT JOIN net_net_results nn ON t.symbol = nn.symbol
                AND nn.analysis_quarter = ?
            LEFT JOIN garp_peg_results gp ON t.symbol = gp.symbol
                AND gp.analysis_quarter = ?
            LEFT JOIN magic_formula_results mf ON t.symbol = mf.symbol
                AND mf.analysis_quarter = ?
            LEFT JOIN fama_french_results ff ON t.symbol = ff.symbol
                AND ff.analysis_quarter = ?
            -- Key metrics: get the latest annual data up to the quarter end date
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
                WHERE fiscal_date <= ?
                ORDER BY symbol, fiscal_date DESC
            ) km ON t.symbol = km.symbol
            WHERE t.is_active = TRUE
            """,
            (quarter,) * 8 + (quarter_end,),
        ).fetchall()

        columns = [desc[0] for desc in conn.description]
        data = {}

        for row in result:
            record = dict(zip(columns, row))
            symbol = record["symbol"]

            # Convert Decimals to floats
            for key in [
                # Existing analysis metrics
                "altman_z_score",
                "roic",
                "free_cash_flow",
                "graham_pe",
                "graham_pb",
                "net_net_discount",
                "peg_ratio",
                "eps_cagr",
                "earnings_yield",
                "mf_roic",
                "book_to_market_percentile",
                "profitability_percentile",
                # New ROIC stability metrics
                "roic_std_dev",
                "gross_margin_std_dev",
                "fcf_to_net_income",
                "reinvestment_rate",
                "fcf_yield",
                # New growth metrics
                "eps_growth_1yr",
                "eps_growth_3yr",
                "eps_growth_5yr",
                # Key metrics - valuation
                "pe_ratio",
                "pb_ratio",
                "price_to_sales",
                "price_to_free_cash_flow",
                "price_to_operating_cash_flow",
                "ev_to_ebitda",
                "ev_to_sales",
                "ev_to_free_cash_flow",
                "ev_to_operating_cash_flow",
                # Key metrics - profitability
                "roe",
                "roa",
                "return_on_tangible_assets",
                "gross_profit_margin",
                "operating_profit_margin",
                "net_profit_margin",
                # Key metrics - liquidity
                "current_ratio",
                "quick_ratio",
                "cash_ratio",
                # Key metrics - leverage
                "debt_ratio",
                "debt_to_equity",
                "debt_to_assets",
                "net_debt_to_ebitda",
                "interest_coverage",
                # Key metrics - efficiency
                "asset_turnover",
                "inventory_turnover",
                "receivables_turnover",
                "payables_turnover",
                # Key metrics - dividends
                "dividend_yield",
                "payout_ratio",
            ]:
                if key in record:
                    record[key] = _to_float(record[key])

            # Parse quality tags JSON into set
            if record.get("quality_tags"):
                try:
                    record["quality_tags_set"] = set(json.loads(record["quality_tags"]))
                except (json.JSONDecodeError, TypeError):
                    record["quality_tags_set"] = set()
            else:
                record["quality_tags_set"] = set()

            data[symbol] = record

        return data

    def _build_observation(
        self,
        symbol: str,
        buy_quarter: str,
        holding_period: int,
        stock_return: float,
        spy_return: float,
        alpha: float,
        data: dict,
    ) -> dict:
        """
        Build a single observation dict with all metrics.

        Converts complex types to primitives for pickling.
        """
        # Start with core fields
        obs = {
            "symbol": symbol,
            "buy_quarter": buy_quarter,
            "holding_period": holding_period,
            "stock_return": round(stock_return, 4),
            "spy_return": round(spy_return, 4),
            "alpha": round(alpha, 4),
        }

        # =====================================================================
        # Pre-computed Scores (existing)
        # =====================================================================
        obs["piotroski_score"] = data.get("piotroski_score")
        obs["graham_score"] = data.get("graham_score")
        obs["altman_z_score"] = data.get("altman_z_score")
        obs["roic"] = data.get("roic")
        obs["peg_ratio"] = data.get("peg_ratio")
        obs["magic_formula_rank"] = data.get("magic_formula_rank")
        obs["book_to_market_percentile"] = data.get("book_to_market_percentile")
        obs["net_net_discount"] = data.get("net_net_discount")
        obs["earnings_yield"] = data.get("earnings_yield")
        obs["profitability_percentile"] = data.get("profitability_percentile")

        # =====================================================================
        # Stability Metrics (from roic_quality_results)
        # =====================================================================
        obs["roic_std_dev"] = data.get("roic_std_dev")
        obs["gross_margin_std_dev"] = data.get("gross_margin_std_dev")
        obs["fcf_to_net_income"] = data.get("fcf_to_net_income")
        obs["reinvestment_rate"] = data.get("reinvestment_rate")
        obs["fcf_yield"] = data.get("fcf_yield")

        # =====================================================================
        # Growth Metrics (from garp_peg_results)
        # =====================================================================
        obs["eps_growth_1yr"] = data.get("eps_growth_1yr")
        obs["eps_growth_3yr"] = data.get("eps_growth_3yr")
        obs["eps_growth_5yr"] = data.get("eps_growth_5yr")
        obs["eps_cagr"] = data.get("eps_cagr")

        # =====================================================================
        # Raw Valuation Metrics (from key_metrics)
        # =====================================================================
        obs["pe_ratio"] = data.get("pe_ratio")
        obs["pb_ratio"] = data.get("pb_ratio")
        obs["price_to_sales"] = data.get("price_to_sales")
        obs["price_to_free_cash_flow"] = data.get("price_to_free_cash_flow")
        obs["price_to_operating_cash_flow"] = data.get("price_to_operating_cash_flow")
        obs["ev_to_ebitda"] = data.get("ev_to_ebitda")
        obs["ev_to_sales"] = data.get("ev_to_sales")
        obs["ev_to_free_cash_flow"] = data.get("ev_to_free_cash_flow")
        obs["ev_to_operating_cash_flow"] = data.get("ev_to_operating_cash_flow")

        # =====================================================================
        # Raw Profitability Metrics (from key_metrics)
        # =====================================================================
        obs["roe"] = data.get("roe")
        obs["roa"] = data.get("roa")
        obs["return_on_tangible_assets"] = data.get("return_on_tangible_assets")
        obs["gross_profit_margin"] = data.get("gross_profit_margin")
        obs["operating_profit_margin"] = data.get("operating_profit_margin")
        obs["net_profit_margin"] = data.get("net_profit_margin")

        # =====================================================================
        # Raw Liquidity Metrics (from key_metrics)
        # =====================================================================
        obs["current_ratio"] = data.get("current_ratio")
        obs["quick_ratio"] = data.get("quick_ratio")
        obs["cash_ratio"] = data.get("cash_ratio")

        # =====================================================================
        # Raw Leverage Metrics (from key_metrics)
        # =====================================================================
        obs["debt_ratio"] = data.get("debt_ratio")
        obs["debt_to_equity"] = data.get("debt_to_equity")
        obs["debt_to_assets"] = data.get("debt_to_assets")
        obs["net_debt_to_ebitda"] = data.get("net_debt_to_ebitda")
        obs["interest_coverage"] = data.get("interest_coverage")

        # =====================================================================
        # Raw Efficiency Metrics (from key_metrics)
        # =====================================================================
        obs["asset_turnover"] = data.get("asset_turnover")
        obs["inventory_turnover"] = data.get("inventory_turnover")
        obs["receivables_turnover"] = data.get("receivables_turnover")
        obs["payables_turnover"] = data.get("payables_turnover")

        # =====================================================================
        # Raw Dividend Metrics (from key_metrics)
        # =====================================================================
        obs["dividend_yield"] = data.get("dividend_yield")
        obs["payout_ratio"] = data.get("payout_ratio")

        # =====================================================================
        # Categorical metrics
        # =====================================================================
        obs["altman_zone"] = data.get("altman_zone")

        # =====================================================================
        # Boolean metrics
        # =====================================================================
        obs["trading_below_ncav"] = bool(data.get("trading_below_ncav"))
        obs["fcf_positive_5yr"] = bool(data.get("fcf_positive_5yr"))
        obs["peg_pass"] = bool(data.get("peg_pass"))

        # Add quality tag boolean fields
        quality_tags = data.get("quality_tags_set", set())
        obs["has_durable_compounder"] = "Durable Compounder" in quality_tags
        obs["has_cash_machine"] = "Cash Machine" in quality_tags
        obs["has_deep_value"] = "Deep Value" in quality_tags
        obs["has_heavy_reinvestor"] = "Heavy Reinvestor" in quality_tags
        obs["has_premium_priced"] = "Premium Priced" in quality_tags
        obs["has_volatile_returns"] = "Volatile Returns" in quality_tags
        obs["has_weak_moat_signal"] = "Weak Moat Signal" in quality_tags
        obs["has_earnings_quality_concern"] = "Earnings Quality Concern" in quality_tags

        # =====================================================================
        # Additional derived fields
        # =====================================================================
        obs["sector"] = data.get("sector")
        obs["name"] = data.get("name")

        return obs

    def _should_exclude(self, data: dict, price: float | None) -> bool:
        """
        Check if a stock should be excluded based on exclusion filters.

        Args:
            data: Analysis data for the stock
            price: Stock price (for penny stock filter)

        Returns:
            True if stock should be excluded
        """
        if not self.exclusions:
            return False

        # Check Altman zone exclusions
        exclude_zones = self.exclusions.get("exclude_altman_zones", [])
        if exclude_zones:
            altman_zone = data.get("altman_zone")
            if altman_zone and altman_zone in exclude_zones:
                return True

        # Check minimum Piotroski score
        min_piotroski = self.exclusions.get("min_piotroski")
        if min_piotroski is not None:
            piotroski = data.get("piotroski_score")
            if piotroski is not None and piotroski < min_piotroski:
                return True

        # Check quality tag exclusions
        quality_tags = data.get("quality_tags_set", set())

        exclude_tags = self.exclusions.get("exclude_quality_tags", [])
        for tag in exclude_tags:
            if tag in quality_tags:
                return True

        # Check required quality tags
        require_tags = self.exclusions.get("require_quality_tags", [])
        for tag in require_tags:
            if tag not in quality_tags:
                return True

        # Check penny stock exclusion
        if self.exclusions.get("exclude_penny_stocks"):
            if price is not None and price < 5.0:
                return True

        # Check negative earnings exclusion
        if self.exclusions.get("exclude_negative_earnings"):
            earnings_yield = data.get("earnings_yield")
            if earnings_yield is not None and earnings_yield < 0:
                return True

        return False
