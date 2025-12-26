# Stock Analysis API

Fundamental stock analysis API implementing 7 valuation systems for NYSE/NASDAQ stocks.

## Valuation Systems

1. **Benjamin Graham 7 Criteria** - Classic value investing criteria
2. **Magic Formula** (Greenblatt) - Earnings yield + return on capital ranking
3. **Piotroski F-Score** - 9-signal financial health score
4. **Altman Z-Score** - Bankruptcy prediction model
5. **ROIC/Quality Screen** - Return on invested capital analysis
6. **GARP/PEG** - Growth at a reasonable price
7. **Net-Net** - Deep value / NCAV analysis

## Setup

### 1. Install dependencies

```bash
pip install -e .
```

### 2. Configure API key

Copy `.env.example` to `.env` and add your FMP API key:

```bash
cp .env.example .env
# Edit .env and add your FMP_API_KEY
```

### 3. Initialize database and load data

```bash
# Initial data load (~40 minutes for 5K stocks at 750 req/min)
python scripts/initial_load.py

# Run all analyses
python scripts/run_analysis.py
```

### 4. Start the API server

```bash
python -m src.main
# or
uvicorn src.api.app:app --reload
```

API will be available at http://localhost:8000

## API Endpoints

### Tickers
- `GET /api/v1/tickers` - List all tickers
- `GET /api/v1/tickers/search?q=AAPL` - Search tickers
- `GET /api/v1/tickers/{symbol}` - Get ticker details

### Financial Data
- `GET /api/v1/financials/{symbol}/profile` - Company profile
- `GET /api/v1/financials/{symbol}/income-statement` - Income statements
- `GET /api/v1/financials/{symbol}/balance-sheet` - Balance sheets
- `GET /api/v1/financials/{symbol}/cash-flow` - Cash flow statements
- `GET /api/v1/financials/{symbol}/metrics` - Key metrics

### Analysis Results
- `GET /api/v1/analysis/{symbol}` - All analyses for a stock
- `GET /api/v1/analysis/{symbol}/graham` - Graham analysis
- `GET /api/v1/analysis/{symbol}/magic-formula` - Magic Formula
- `GET /api/v1/analysis/{symbol}/piotroski` - Piotroski F-Score
- `GET /api/v1/analysis/{symbol}/altman` - Altman Z-Score
- `GET /api/v1/analysis/{symbol}/roic` - ROIC/Quality
- `GET /api/v1/analysis/{symbol}/peg` - GARP/PEG
- `GET /api/v1/analysis/{symbol}/net-net` - Net-Net

### Screeners
- `GET /api/v1/screener/graham?min_score=5` - Graham screen
- `GET /api/v1/screener/magic-formula?top=50` - Top Magic Formula stocks
- `GET /api/v1/screener/piotroski?min_score=7` - High F-Score stocks
- `GET /api/v1/screener/altman?zone=safe` - Safe Z-Score stocks
- `GET /api/v1/screener/roic?min_roic=0.15` - High ROIC stocks
- `GET /api/v1/screener/peg?max_peg=1.0` - Low PEG stocks
- `GET /api/v1/screener/net-net` - Net-Net opportunities
- `GET /api/v1/screener/combined` - Multi-criteria screen

### Status
- `GET /api/v1/status` - API status and data freshness

## Quarterly Update

Run quarterly to refresh data:

```bash
python scripts/quarterly_update.py
python scripts/run_analysis.py
```

## Data Storage

- All data stored in DuckDB (`data/stock_analysis.duckdb`)
- Raw JSON responses preserved for future analysis
- Timestamped for incremental updates

## Rate Limits

- FMP Premium: 750 requests/minute
- Initial load: ~40 minutes for 5K stocks
- Quarterly update: ~10 minutes
