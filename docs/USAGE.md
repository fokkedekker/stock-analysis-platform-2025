# Stock Analysis - Usage Guide

## Prerequisites

- Python 3.10+
- FMP API key (Premium tier recommended for 750 req/min)

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

## Scripts Overview

| Script | Purpose | When to Use |
|--------|---------|-------------|
| `initial_load.py` | Fetch all financial data (30 years of statements) | Fresh setup only |
| `historical_load.py` | Fetch historical prices, create quarterly profiles | Fresh setup only |
| `run_analysis.py` | Run all analyzers (Graham, Piotroski, etc.) | After any data load |
| `quarterly_update.py` | Refresh profiles and latest statements | Every quarter |

---

## Workflow: Fresh Setup (New Machine)

Run these steps in order when setting up from scratch:

### Step 1: Fetch All Financial Data

```bash
source venv/bin/activate
python scripts/initial_load.py
```

**What it does:**
- Fetches stock tickers from Massive API (NYSE/NASDAQ common stocks)
- For each ticker (~5,000), fetches from FMP:
  - Company profile
  - Income statements (annual + quarterly, 30 years)
  - Balance sheets (annual + quarterly)
  - Cash flow statements (annual + quarterly)
  - Key metrics (annual + quarterly)
  - Dividend history

**Estimated time:** ~40-60 minutes at 750 req/min

**Resume capability:** If interrupted, run again - it resumes from checkpoint.

### Step 2: Fetch Historical Prices

```bash
python scripts/historical_load.py -y
```

**What it does:**
- Fetches historical prices for the last 8 quarters
- Creates company profiles with correct quarterly prices (for backtesting)
- Calculates P/E and P/B ratios at each quarter end

**Estimated time:** ~10-15 minutes

### Step 3: Run Analysis for All Quarters

```bash
python scripts/run_analysis.py --all-quarters --parallel-workers 8
```

**What it does:**
- Runs all 9 analyzers for each quarter with data
- Uses parallel threads for speed (recommended: 4-8 workers)

**Estimated time:** ~30-60 minutes with 8 parallel workers

---

## Workflow: Quarterly Maintenance

Run every quarter after new earnings are released (typically mid-quarter):

### Step 1: Fetch New Data

```bash
source venv/bin/activate
python scripts/quarterly_update.py
```

**What it does:**
- Updates ticker list
- Refreshes company profiles with current prices
- Fetches any new quarterly/annual statements

**Estimated time:** ~10-15 minutes

### Step 2: Run Analysis

```bash
python scripts/run_analysis.py --parallel-workers 8
```

This runs all analyzers for the current quarter only.

---

## Script Reference

### run_analysis.py

```bash
python scripts/run_analysis.py [options]
```

| Flag | Description |
|------|-------------|
| *(no flags)* | Run all analyzers for current quarter |
| `--quarter 2024Q3` | Run for specific quarter |
| `--all-quarters` | Run for all quarters with data (up to 8) |
| `--parallel-workers N` | Run N symbols in parallel (recommended: 4-8) |
| `--only graham` | Run only analyzers matching "graham" |
| `--only roic garp` | Run only ROIC and GARP analyzers |
| `--start roic` | Start from ROIC analyzer (skip prior ones) |
| `--skip-rankings` | Skip post-processing rankings/percentiles |
| `--list` | List all available analyzers |

**Examples:**

```bash
# Run all analyzers for current quarter with parallelization
python scripts/run_analysis.py --parallel-workers 8

# Run only Graham analyzers for Q3 2024
python scripts/run_analysis.py --quarter 2024Q3 --only graham

# Run all quarters (for backtesting setup)
python scripts/run_analysis.py --all-quarters --parallel-workers 8

# List available analyzers
python scripts/run_analysis.py --list
```

### Available Analyzers

| Key | Name | Description |
|-----|------|-------------|
| `graham-strict` | Graham (strict) | Benjamin Graham's 7 criteria, original thresholds |
| `graham-modern` | Graham (modern) | Graham criteria with relaxed modern thresholds |
| `magic-formula` | Magic Formula | Greenblatt's earnings yield + ROIC ranking |
| `piotroski` | Piotroski F-Score | 9-point financial strength score |
| `altman` | Altman Z-Score | Bankruptcy prediction model |
| `roic` | ROIC/Quality | Return on invested capital screen |
| `garp` | GARP/PEG | Growth at reasonable price (PEG ratio) |
| `fama-french` | Fama-French | Size, value, profitability factors |
| `net-net` | Net-Net | NCAV (net current asset value) screen |

---

## Starting the Application

### Backend API

```bash
source venv/bin/activate
python -m src.main
# or
uvicorn src.api.app:app --reload
```

API available at: http://localhost:8000
API docs at: http://localhost:8000/docs

### Frontend

```bash
cd frontend
npm install
npm run dev
```

Frontend available at: http://localhost:3000

---

## Troubleshooting

### Script interrupted or failed
Simply run the script again - it will resume from the checkpoint.

### Rate limit errors
The script handles rate limiting automatically. If you see persistent 429 errors, your API tier may have a lower limit. Update `RATE_LIMIT_PER_MINUTE` in `.env`.

### Missing data for some stocks
Some stocks may not have complete data (e.g., recent IPOs, foreign listings). The system handles this gracefully and stores what's available.

### DuckDB locking errors
DuckDB only supports one writer process at a time. If you see "Could not set lock on file" errors:
- Wait for any running scripts to finish
- Use `--parallel-workers` for parallelization (uses threads, not processes)

### Reset everything
```bash
rm data/stock_analysis.duckdb
rm data/fetch_checkpoint.json
rm data/historical_checkpoint.json
rm data/quarterly_checkpoint.json
python scripts/initial_load.py
```

### Python version issues
Make sure you're using Python 3.10+ and have activated the virtual environment:
```bash
source venv/bin/activate
python --version  # Should be 3.10+
```
