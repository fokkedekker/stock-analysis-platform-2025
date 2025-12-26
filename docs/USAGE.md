# Stock Analysis - Usage Guide

## Prerequisites

- Python 3.10+
- FMP API key (Premium tier recommended for 750 req/min)
- Massive API key (for ticker list)

## Installation

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -e .
```

## Configuration

Create a `.env` file in the project root:

```bash
FMP_API_KEY=your_fmp_api_key_here
DATABASE_PATH=data/stock_analysis.duckdb
RATE_LIMIT_PER_MINUTE=750
```

---

## Initial Data Load

Fetches all financial data for ~5,000 NYSE/NASDAQ stocks. Run this once when setting up.

```bash
source venv/bin/activate
python scripts/initial_load.py
```

**What it does:**
1. Fetches stock tickers from Massive API (NYSE/NASDAQ common stocks)
2. For each ticker, fetches from FMP:
   - Company profile
   - Income statements (annual + quarterly, 30 years)
   - Balance sheets (annual + quarterly)
   - Cash flow statements (annual + quarterly)
   - Key metrics (annual + quarterly)
   - Dividend history

**Estimated time:** ~40-60 minutes at 750 req/min

**Resume capability:** If the script fails or is interrupted, simply run it again. It will skip already-completed symbols and resume from where it left off.

To force a fresh start, delete the checkpoint file:
```bash
rm data/fetch_checkpoint.json
```

---

## Run Analysis

After loading data, run all valuation analyses:

```bash
source venv/bin/activate
python scripts/run_analysis.py
```

**Analyses performed:**
- Benjamin Graham 7 Criteria (strict + modern modes)
- Magic Formula (Greenblatt)
- Piotroski F-Score
- Altman Z-Score
- ROIC/Quality Screen
- GARP/PEG Ratio
- Fama-French Factors
- Net-Net (NCAV)

**Estimated time:** ~5-10 minutes (local computation, no API calls)

---

## Quarterly Update

Run quarterly to refresh data with new filings:

```bash
source venv/bin/activate
python scripts/quarterly_update.py
```

**What it does:**
1. Refreshes company profiles for all stocks
2. Fetches any new quarterly/annual statements
3. Updates key metrics

**Estimated time:** ~10-15 minutes

After updating data, run the analysis again:
```bash
python scripts/run_analysis.py
```

---

## Start the API Server

```bash
source venv/bin/activate
python -m src.main
# or
uvicorn src.api.app:app --reload
```

API available at: http://localhost:8000

API docs at: http://localhost:8000/docs

---

## Troubleshooting

### Script interrupted or failed
Simply run the script again - it will resume from the checkpoint.

### Rate limit errors
The script handles rate limiting automatically. If you see persistent 429 errors, your API tier may have a lower limit. Update `RATE_LIMIT_PER_MINUTE` in `.env`.

### Missing data for some stocks
Some stocks may not have complete data (e.g., recent IPOs, foreign listings). The system handles this gracefully and stores what's available.

### Reset everything
```bash
rm data/stock_analysis.duckdb
rm data/fetch_checkpoint.json
python scripts/initial_load.py
```
