"""Financial Modeling Prep API client with rate limiting and fault tolerance."""

import asyncio
import logging
import time
from typing import Any

import httpx
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log,
)

from src.config import get_settings

logger = logging.getLogger(__name__)


class LeakyBucketRateLimiter:
    """Leaky bucket rate limiter for steady request rate without bursts."""

    def __init__(self, calls_per_minute: int = 650):
        """Initialize rate limiter.

        Args:
            calls_per_minute: Target API calls per minute.
        """
        self.min_interval = 60.0 / calls_per_minute  # Time between calls
        self.next_allowed = 0.0
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        """Acquire permission to make a request, waiting if necessary."""
        # Quickly grab a time slot under the lock
        async with self._lock:
            now = time.monotonic()
            if now >= self.next_allowed:
                # Can go immediately
                self.next_allowed = now + self.min_interval
                wait_needed = 0
            else:
                # Schedule for later
                wait_needed = self.next_allowed - now
                self.next_allowed += self.min_interval

        # Sleep OUTSIDE the lock so other workers can schedule
        if wait_needed > 0:
            await asyncio.sleep(wait_needed)


class FMPClient:
    """Async client for Financial Modeling Prep API."""

    ENDPOINTS = {
        "stock_list": "/stable/stock-list",
        "profile": "/stable/profile",
        "income_statement": "/stable/income-statement",
        "balance_sheet": "/stable/balance-sheet-statement",
        "cash_flow": "/stable/cash-flow-statement",
        "key_metrics": "/stable/key-metrics",
        "ratios": "/stable/ratios",
        "dividends": "/stable/dividends",
        "historical_prices": "/stable/historical-price-eod/full",
    }

    def __init__(
        self,
        api_key: str | None = None,
        rate_limit: int | None = None,
    ):
        """Initialize FMP client.

        Args:
            api_key: FMP API key. Uses settings if not provided.
            rate_limit: Requests per minute. Uses settings if not provided.
        """
        settings = get_settings()
        self.api_key = api_key or settings.FMP_API_KEY
        self.base_url = settings.FMP_BASE_URL
        self.rate_limiter = LeakyBucketRateLimiter(rate_limit or settings.RATE_LIMIT_PER_MINUTE)
        self._client: httpx.AsyncClient | None = None
        self._max_retries = settings.MAX_RETRIES
        self._retry_min_wait = settings.RETRY_MIN_WAIT
        self._retry_max_wait = settings.RETRY_MAX_WAIT

    async def __aenter__(self) -> "FMPClient":
        """Enter async context."""
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(30.0, connect=10.0),
            limits=httpx.Limits(max_keepalive_connections=20, max_connections=100),
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit async context."""
        if self._client:
            await self._client.aclose()
            self._client = None

    def _get_retry_decorator(self):
        """Get retry decorator with current settings."""
        return retry(
            stop=stop_after_attempt(self._max_retries),
            wait=wait_exponential(
                multiplier=1,
                min=self._retry_min_wait,
                max=self._retry_max_wait,
            ),
            retry=retry_if_exception_type(
                (httpx.TimeoutException, httpx.HTTPStatusError, httpx.ConnectError)
            ),
            before_sleep=before_sleep_log(logger, logging.WARNING),
        )

    async def _request(
        self,
        endpoint: str,
        params: dict[str, Any] | None = None,
    ) -> dict[str, Any] | list[dict[str, Any]]:
        """Make an API request with rate limiting and retries.

        Args:
            endpoint: API endpoint path.
            params: Query parameters.

        Returns:
            JSON response data. Returns empty list for 404 (no data available).

        Raises:
            httpx.HTTPStatusError: For non-retryable HTTP errors (except 404).
            ValueError: For invalid API responses.
        """
        if self._client is None:
            raise RuntimeError("Client not initialized. Use 'async with' context.")

        await self.rate_limiter.acquire()

        url = f"{self.base_url}{endpoint}"
        request_params = {"apikey": self.api_key}
        if params:
            request_params.update(params)

        @self._get_retry_decorator()
        async def _do_request():
            start_time = time.monotonic()
            response = await self._client.get(url, params=request_params)

            elapsed_ms = int((time.monotonic() - start_time) * 1000)
            logger.debug(f"Request to {endpoint} completed in {elapsed_ms}ms")

            # Check for rate limiting - should retry
            if response.status_code == 429:
                logger.warning("Rate limited by FMP, backing off...")
                raise httpx.HTTPStatusError(
                    "Rate limited",
                    request=response.request,
                    response=response,
                )

            # 404 means no data available - return empty list, don't retry
            if response.status_code == 404:
                logger.debug(f"No data found for {endpoint} (404)")
                return []

            # 402 or premium subscription error - skip this symbol, don't retry
            if response.status_code == 402:
                logger.debug(f"Premium symbol not in plan for {endpoint} (402)")
                return []

            # Check for premium subscription message in response
            if response.status_code == 200:
                text = response.text
                if "not available under your current subscription" in text:
                    logger.debug(f"Premium symbol not in plan for {endpoint}")
                    return []

            response.raise_for_status()
            return response.json()

        return await _do_request()

    async def get_stock_list(self) -> list[dict[str, Any]]:
        """Get list of all available stocks.

        Returns:
            List of stock dictionaries with symbol, name, exchange, etc.
        """
        return await self._request(self.ENDPOINTS["stock_list"])

    async def get_profile(self, symbol: str) -> list[dict[str, Any]]:
        """Get company profile.

        Args:
            symbol: Stock ticker symbol.

        Returns:
            Company profile data.
        """
        return await self._request(
            self.ENDPOINTS["profile"],
            params={"symbol": symbol},
        )

    async def get_income_statement(
        self,
        symbol: str,
        period: str = "annual",
        limit: int = 30,
    ) -> list[dict[str, Any]]:
        """Get income statements.

        Args:
            symbol: Stock ticker symbol.
            period: 'annual' or 'quarter'.
            limit: Number of periods to retrieve.

        Returns:
            List of income statement records.
        """
        return await self._request(
            self.ENDPOINTS["income_statement"],
            params={"symbol": symbol, "period": period, "limit": limit},
        )

    async def get_balance_sheet(
        self,
        symbol: str,
        period: str = "annual",
        limit: int = 30,
    ) -> list[dict[str, Any]]:
        """Get balance sheets.

        Args:
            symbol: Stock ticker symbol.
            period: 'annual' or 'quarter'.
            limit: Number of periods to retrieve.

        Returns:
            List of balance sheet records.
        """
        return await self._request(
            self.ENDPOINTS["balance_sheet"],
            params={"symbol": symbol, "period": period, "limit": limit},
        )

    async def get_cash_flow(
        self,
        symbol: str,
        period: str = "annual",
        limit: int = 30,
    ) -> list[dict[str, Any]]:
        """Get cash flow statements.

        Args:
            symbol: Stock ticker symbol.
            period: 'annual' or 'quarter'.
            limit: Number of periods to retrieve.

        Returns:
            List of cash flow statement records.
        """
        return await self._request(
            self.ENDPOINTS["cash_flow"],
            params={"symbol": symbol, "period": period, "limit": limit},
        )

    async def get_key_metrics(
        self,
        symbol: str,
        period: str = "annual",
        limit: int = 30,
    ) -> list[dict[str, Any]]:
        """Get key financial metrics.

        Args:
            symbol: Stock ticker symbol.
            period: 'annual' or 'quarter'.
            limit: Number of periods to retrieve.

        Returns:
            List of key metrics records.
        """
        return await self._request(
            self.ENDPOINTS["key_metrics"],
            params={"symbol": symbol, "period": period, "limit": limit},
        )

    async def get_ratios(
        self,
        symbol: str,
        period: str = "annual",
        limit: int = 30,
    ) -> list[dict[str, Any]]:
        """Get financial ratios (PE, PB, margins, etc.).

        Args:
            symbol: Stock ticker symbol.
            period: 'annual' or 'quarter'.
            limit: Number of periods to retrieve.

        Returns:
            List of ratio records with PE, PB, margins, turnover, etc.
        """
        return await self._request(
            self.ENDPOINTS["ratios"],
            params={"symbol": symbol, "period": period, "limit": limit},
        )

    async def get_dividends(self, symbol: str) -> list[dict[str, Any]]:
        """Get dividend history.

        Args:
            symbol: Stock ticker symbol.

        Returns:
            Dividend history data (list of dividend records).
        """
        return await self._request(
            self.ENDPOINTS["dividends"],
            params={"symbol": symbol},
        )

    async def get_historical_prices(
        self,
        symbol: str,
        from_date: str | None = None,
        to_date: str | None = None,
    ) -> list[dict[str, Any]]:
        """Get historical end-of-day prices.

        Args:
            symbol: Stock ticker symbol.
            from_date: Start date in YYYY-MM-DD format (optional).
            to_date: End date in YYYY-MM-DD format (optional).

        Returns:
            List of daily price records with date, open, high, low, close,
            adjClose, volume, etc. Sorted by date descending (most recent first).
        """
        params = {"symbol": symbol}
        if from_date:
            params["from"] = from_date
        if to_date:
            params["to"] = to_date

        result = await self._request(self.ENDPOINTS["historical_prices"], params=params)

        # FMP returns {"symbol": "...", "historical": [...]} for this endpoint
        if isinstance(result, dict) and "historical" in result:
            return result["historical"]
        return result if isinstance(result, list) else []

    async def fetch_endpoint(self, symbol: str, endpoint: str) -> list[dict[str, Any]]:
        """Fetch a single endpoint for a symbol.

        Args:
            symbol: Stock ticker symbol.
            endpoint: One of 'profile', 'income_annual', 'income_quarterly',
                     'balance_annual', 'balance_quarterly', 'cashflow_annual',
                     'cashflow_quarterly', 'metrics_annual', 'metrics_quarterly',
                     'ratios_annual', 'ratios_quarterly', 'dividends'.

        Returns:
            API response data.
        """
        endpoint_map = {
            "profile": lambda: self.get_profile(symbol),
            "income_annual": lambda: self.get_income_statement(symbol, period="annual"),
            "income_quarterly": lambda: self.get_income_statement(symbol, period="quarter"),
            "balance_annual": lambda: self.get_balance_sheet(symbol, period="annual"),
            "balance_quarterly": lambda: self.get_balance_sheet(symbol, period="quarter"),
            "cashflow_annual": lambda: self.get_cash_flow(symbol, period="annual"),
            "cashflow_quarterly": lambda: self.get_cash_flow(symbol, period="quarter"),
            "metrics_annual": lambda: self.get_key_metrics(symbol, period="annual"),
            "metrics_quarterly": lambda: self.get_key_metrics(symbol, period="quarter"),
            "ratios_annual": lambda: self.get_ratios(symbol, period="annual"),
            "ratios_quarterly": lambda: self.get_ratios(symbol, period="quarter"),
            "dividends": lambda: self.get_dividends(symbol),
        }

        if endpoint not in endpoint_map:
            raise ValueError(f"Unknown endpoint: {endpoint}")

        return await endpoint_map[endpoint]()

    async def fetch_all_data(
        self,
        symbol: str,
        include_quarterly: bool = True,
    ) -> dict[str, Any]:
        """Fetch all financial data for a symbol.

        Args:
            symbol: Stock ticker symbol.
            include_quarterly: Whether to include quarterly data.

        Returns:
            Dictionary with all data types as keys.
        """
        result = {
            "symbol": symbol,
            "profile": None,
            "income_statement_annual": None,
            "income_statement_quarterly": None,
            "balance_sheet_annual": None,
            "balance_sheet_quarterly": None,
            "cash_flow_annual": None,
            "cash_flow_quarterly": None,
            "key_metrics_annual": None,
            "key_metrics_quarterly": None,
            "dividends": None,
            "errors": [],
        }

        # Fetch profile
        try:
            result["profile"] = await self.get_profile(symbol)
        except Exception as e:
            logger.warning(f"Failed to fetch profile for {symbol}: {e}")
            result["errors"].append({"endpoint": "profile", "error": str(e)})

        # Fetch income statements
        try:
            result["income_statement_annual"] = await self.get_income_statement(
                symbol, period="annual"
            )
        except Exception as e:
            logger.warning(f"Failed to fetch annual income statement for {symbol}: {e}")
            result["errors"].append({"endpoint": "income_statement_annual", "error": str(e)})

        if include_quarterly:
            try:
                result["income_statement_quarterly"] = await self.get_income_statement(
                    symbol, period="quarter"
                )
            except Exception as e:
                logger.warning(f"Failed to fetch quarterly income statement for {symbol}: {e}")
                result["errors"].append(
                    {"endpoint": "income_statement_quarterly", "error": str(e)}
                )

        # Fetch balance sheets
        try:
            result["balance_sheet_annual"] = await self.get_balance_sheet(symbol, period="annual")
        except Exception as e:
            logger.warning(f"Failed to fetch annual balance sheet for {symbol}: {e}")
            result["errors"].append({"endpoint": "balance_sheet_annual", "error": str(e)})

        if include_quarterly:
            try:
                result["balance_sheet_quarterly"] = await self.get_balance_sheet(
                    symbol, period="quarter"
                )
            except Exception as e:
                logger.warning(f"Failed to fetch quarterly balance sheet for {symbol}: {e}")
                result["errors"].append({"endpoint": "balance_sheet_quarterly", "error": str(e)})

        # Fetch cash flow statements
        try:
            result["cash_flow_annual"] = await self.get_cash_flow(symbol, period="annual")
        except Exception as e:
            logger.warning(f"Failed to fetch annual cash flow for {symbol}: {e}")
            result["errors"].append({"endpoint": "cash_flow_annual", "error": str(e)})

        if include_quarterly:
            try:
                result["cash_flow_quarterly"] = await self.get_cash_flow(symbol, period="quarter")
            except Exception as e:
                logger.warning(f"Failed to fetch quarterly cash flow for {symbol}: {e}")
                result["errors"].append({"endpoint": "cash_flow_quarterly", "error": str(e)})

        # Fetch key metrics
        try:
            result["key_metrics_annual"] = await self.get_key_metrics(symbol, period="annual")
        except Exception as e:
            logger.warning(f"Failed to fetch annual key metrics for {symbol}: {e}")
            result["errors"].append({"endpoint": "key_metrics_annual", "error": str(e)})

        if include_quarterly:
            try:
                result["key_metrics_quarterly"] = await self.get_key_metrics(
                    symbol, period="quarter"
                )
            except Exception as e:
                logger.warning(f"Failed to fetch quarterly key metrics for {symbol}: {e}")
                result["errors"].append({"endpoint": "key_metrics_quarterly", "error": str(e)})

        # Fetch dividends
        try:
            result["dividends"] = await self.get_dividends(symbol)
        except Exception as e:
            logger.warning(f"Failed to fetch dividends for {symbol}: {e}")
            result["errors"].append({"endpoint": "dividends", "error": str(e)})

        return result
