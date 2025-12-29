"""Data preloader for grid search simulations.

Loads all analysis results and prices into memory for fast filtering and return calculations.
"""

import json
import logging
from decimal import Decimal
from typing import Any

from src.database.connection import get_db_manager

from .models import StrategyConfig

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


class SimulationDataPreloader:
    """
    Preloads all data needed for grid search simulations.

    Loads once, used by all worker processes via pickling.
    Data structure optimized for fast filtering operations.
    """

    def __init__(self, quarters: list[str]):
        """
        Initialize preloader with list of quarters to load.

        Args:
            quarters: List of quarter strings like ["2023Q1", "2023Q2", ...]
        """
        self.quarters = quarters
        # quarter -> symbol -> analysis data dict
        self.analysis_data: dict[str, dict[str, dict]] = {}
        # quarter -> symbol -> price
        self.price_data: dict[str, dict[str, float]] = {}
        # quarter -> SPY price
        self.spy_prices: dict[str, float] = {}
        # All quarters with price data (for sell quarter validation)
        self.available_price_quarters: set[str] = set()

    def preload(self) -> None:
        """Load all data needed for simulations."""
        db = get_db_manager()

        with db.get_connection() as conn:
            logger.info(f"Preloading data for {len(self.quarters)} quarters...")

            # 1. Load all analysis results for all quarters
            self._load_analysis_results(conn)

            # 2. Load all stock prices for all quarters
            self._load_prices(conn)

            # 3. Load SPY prices for benchmark
            self._load_spy_prices(conn)

            logger.info(
                f"Preloaded {sum(len(d) for d in self.analysis_data.values())} stock records "
                f"across {len(self.quarters)} quarters"
            )

    def _load_analysis_results(self, conn) -> None:
        """Load all analysis results into memory, organized by quarter."""
        for quarter in self.quarters:
            self.analysis_data[quarter] = self._query_unified_analysis(conn, quarter)
            logger.debug(f"Loaded {len(self.analysis_data[quarter])} stocks for {quarter}")

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

            # Parse quality tags JSON
            if record.get("quality_tags"):
                try:
                    record["quality_tags_set"] = set(json.loads(record["quality_tags"]))
                except (json.JSONDecodeError, TypeError):
                    record["quality_tags_set"] = set()
            else:
                record["quality_tags_set"] = set()

            data[symbol] = record

        return data

    def _load_prices(self, conn) -> None:
        """Load all stock prices from company_profiles."""
        # First, find all quarters with price data
        all_quarters_result = conn.execute(
            """
            SELECT DISTINCT fiscal_quarter
            FROM company_profiles
            WHERE price IS NOT NULL
            ORDER BY fiscal_quarter
            """
        ).fetchall()
        self.available_price_quarters = {row[0] for row in all_quarters_result}

        # Load prices for all available quarters (not just analysis quarters)
        for quarter in self.available_price_quarters:
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
            self.price_data[quarter] = {
                row[0]: _to_float(row[1]) for row in result if row[1]
            }
            logger.debug(f"Loaded {len(self.price_data[quarter])} prices for {quarter}")

    def _load_spy_prices(self, conn) -> None:
        """Load real SPY benchmark prices from database.

        Falls back to synthetic prices if no data in database.
        Run scripts/load_spy_prices.py to populate the spy_prices table.
        """
        if not self.available_price_quarters:
            return

        # Try to load from database
        try:
            result = conn.execute(
                """
                SELECT quarter, price
                FROM spy_prices
                ORDER BY quarter
                """
            ).fetchall()

            if result:
                for quarter, price in result:
                    self.spy_prices[quarter] = _to_float(price)
                logger.info(f"Loaded {len(self.spy_prices)} real SPY prices from database")
                return

        except Exception as e:
            logger.warning(f"Could not load SPY prices from database: {e}")

        # Fallback to synthetic prices if no data
        logger.warning("No SPY prices in database, using synthetic 6% annual growth")
        self._generate_synthetic_spy_prices()

    def _generate_synthetic_spy_prices(self) -> None:
        """Generate synthetic SPY prices as fallback (6% annual growth)."""
        base_price = 100.0
        quarterly_return = 0.015  # 1.5% per quarter
        base_year = 2020
        base_quarter = 1

        def quarters_since_base(q: str) -> int:
            year = int(q[:4])
            qnum = int(q[-1])
            return (year - base_year) * 4 + (qnum - base_quarter)

        for quarter in sorted(self.available_price_quarters):
            periods = quarters_since_base(quarter)
            price = base_price * ((1 + quarterly_return) ** periods)
            self.spy_prices[quarter] = round(price, 2)

    def get_stocks_passing_strategy(
        self,
        strategy: StrategyConfig,
        quarter: str,
        mf_total_stocks: int | None = None,
    ) -> list[str]:
        """
        Filter stocks that pass a given strategy for a quarter.

        Uses in-memory data, no database queries.

        Args:
            strategy: Strategy configuration to apply
            quarter: Quarter to filter for
            mf_total_stocks: Total stocks for Magic Formula percentile calculation

        Returns:
            List of symbols that pass all strategy criteria
        """
        if quarter not in self.analysis_data:
            return []

        passing = []
        quarter_data = self.analysis_data[quarter]

        # Calculate Magic Formula rank threshold if needed
        if mf_total_stocks is None:
            mf_total_stocks = len(quarter_data)
        mf_rank_threshold = int(mf_total_stocks * strategy.valuation.mf_top_pct / 100)

        for symbol, data in quarter_data.items():
            if self._stock_passes_strategy(data, strategy, mf_rank_threshold):
                passing.append(symbol)

        return passing

    def _stock_passes_strategy(
        self,
        data: dict,
        strategy: StrategyConfig,
        mf_rank_threshold: int,
    ) -> bool:
        """Check if a single stock passes the strategy criteria."""
        # Stage 1: Survival Gates
        if strategy.survival.altman_enabled:
            zone = data.get("altman_zone")
            # distress = allow all, grey = allow safe+grey, safe = allow safe only
            if strategy.survival.altman_zone == "safe":
                if zone != "safe":
                    return False
            elif strategy.survival.altman_zone == "grey":
                if zone not in ("safe", "grey"):
                    return False
            # distress: allow all zones (no filtering)

        if strategy.survival.piotroski_enabled:
            score = data.get("piotroski_score")
            if score is None or score < strategy.survival.piotroski_min:
                return False

        # Stage 2: Quality Filter
        if strategy.quality.enabled:
            roic = data.get("roic")
            fcf_5yr = data.get("fcf_positive_5yr")

            # Determine quality label
            if roic is not None and roic >= 0.15 and fcf_5yr:
                quality_label = "compounder"
            elif roic is not None and roic >= 0.08:
                quality_label = "average"
            else:
                quality_label = "weak"

            # Check minimum quality
            if strategy.quality.min_quality == "compounder":
                if quality_label != "compounder":
                    return False
            elif strategy.quality.min_quality == "average":
                if quality_label not in ("compounder", "average"):
                    return False

        # Quality tags filter
        if strategy.quality.required_tags:
            stock_tags = data.get("quality_tags_set", set())
            # Stock must have at least one required tag
            if not (stock_tags & set(strategy.quality.required_tags)):
                return False

        if strategy.quality.excluded_tags:
            stock_tags = data.get("quality_tags_set", set())
            # Stock must not have any excluded tags
            if stock_tags & set(strategy.quality.excluded_tags):
                return False

        # Stage 3: Valuation Lenses (at-least-N logic)
        lenses_passed = 0
        lenses_active = 0

        # Graham lens
        if strategy.valuation.graham_enabled:
            lenses_active += 1
            graham_score = data.get("graham_score")
            if graham_score is not None and graham_score >= strategy.valuation.graham_min:
                lenses_passed += 1

        # Net-Net lens
        if strategy.valuation.net_net_enabled:
            lenses_active += 1
            if data.get("trading_below_ncav"):
                lenses_passed += 1

        # PEG lens
        if strategy.valuation.peg_enabled:
            lenses_active += 1
            peg = data.get("peg_ratio")
            if peg is not None and 0 < peg <= strategy.valuation.max_peg:
                lenses_passed += 1

        # Magic Formula lens
        if strategy.valuation.magic_formula_enabled:
            lenses_active += 1
            mf_rank = data.get("magic_formula_rank")
            if mf_rank is not None and mf_rank <= mf_rank_threshold:
                lenses_passed += 1

        # Fama-French B/M lens
        if strategy.valuation.fama_french_enabled:
            lenses_active += 1
            bm_pct = data.get("book_to_market_percentile")
            threshold = 1.0 - strategy.valuation.ff_top_pct / 100.0
            if bm_pct is not None and bm_pct >= threshold:
                lenses_passed += 1

        # Check valuation logic
        if strategy.valuation.strict_mode:
            # All active lenses must pass
            if lenses_passed != lenses_active:
                return False
        else:
            # At least N lenses must pass
            if lenses_passed < strategy.valuation.min_lenses:
                return False

        return True

    def calculate_return(
        self,
        symbol: str,
        buy_quarter: str,
        sell_quarter: str,
    ) -> float | None:
        """Calculate return for a stock between quarters."""
        buy_price = self.price_data.get(buy_quarter, {}).get(symbol)
        sell_price = self.price_data.get(sell_quarter, {}).get(symbol)

        if buy_price and sell_price and buy_price > 0:
            return ((sell_price - buy_price) / buy_price) * 100
        return None

    def calculate_spy_return(
        self,
        buy_quarter: str,
        sell_quarter: str,
    ) -> float:
        """Calculate SPY benchmark return between quarters."""
        buy_price = self.spy_prices.get(buy_quarter)
        sell_price = self.spy_prices.get(sell_quarter)

        if buy_price and sell_price and buy_price > 0:
            return ((sell_price - buy_price) / buy_price) * 100

        # Default benchmark return if SPY data not available
        # Estimate ~2.5% per quarter average
        return 2.5

    def get_sell_quarter(self, buy_quarter: str, holding_periods: int) -> str | None:
        """
        Get the sell quarter given buy quarter and holding period.

        Returns None if sell quarter doesn't have price data.
        """
        sell_quarter = self._add_quarters(buy_quarter, holding_periods)
        if sell_quarter in self.available_price_quarters:
            return sell_quarter
        return None

    def _add_quarters(self, quarter: str, periods: int) -> str:
        """Add N quarters to a quarter string."""
        year = int(quarter[:4])
        q = int(quarter[-1])

        total_quarters = year * 4 + q + periods
        new_year = (total_quarters - 1) // 4
        new_q = ((total_quarters - 1) % 4) + 1

        return f"{new_year}Q{new_q}"
