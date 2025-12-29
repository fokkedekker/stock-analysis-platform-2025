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
        """
        self.quarters = quarters
        self.holding_periods = holding_periods
        self.exclusions = exclusions or {}

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
            for quarter in self.quarters:
                if quarter not in available_quarters:
                    logger.warning(f"Skipping quarter {quarter} - no price data")
                    continue

                # Load analysis results for this quarter
                analysis_data = self._query_unified_analysis(conn, quarter)
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

                -- ROIC/Quality
                r.roic,
                r.free_cash_flow,
                r.fcf_positive_5yr,
                r.quality_tags,

                -- Graham
                g.criteria_passed as graham_score,
                g.pe_ratio as graham_pe,
                g.pb_ratio as graham_pb,

                -- Net-Net
                nn.trading_below_ncav,
                nn.discount_to_ncav as net_net_discount,

                -- PEG
                gp.peg_ratio,
                gp.eps_cagr,
                gp.peg_pass,

                -- Magic Formula
                mf.combined_rank as magic_formula_rank,
                mf.earnings_yield,
                mf.return_on_capital as mf_roic,

                -- Fama-French
                ff.book_to_market_percentile,
                ff.profitability_percentile

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
            WHERE t.is_active = TRUE
            """,
            (quarter,) * 8,
        ).fetchall()

        columns = [desc[0] for desc in conn.description]
        data = {}

        for row in result:
            record = dict(zip(columns, row))
            symbol = record["symbol"]

            # Convert Decimals to floats
            for key in [
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

        # Add numerical metrics
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

        # Add categorical metrics
        obs["altman_zone"] = data.get("altman_zone")

        # Add boolean metrics
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

        # Additional derived fields
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
