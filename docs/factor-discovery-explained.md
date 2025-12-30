# Factor Discovery System: A Complete Guide

## What This Document Is

This document explains exactly how our Factor Discovery system works—the statistical methods it uses, what they mean, and how to interpret the results. Every technical concept is paired with a plain-English explanation.

---

## Table of Contents

1. [The Big Picture](#the-big-picture)
2. [Phase 1: Building the Dataset](#phase-1-building-the-dataset)
3. [Phase 2: Analyzing Individual Factors](#phase-2-analyzing-individual-factors)
4. [Phase 3: Combining Factors](#phase-3-combining-factors)
5. [Phase 4: Generating Recommendations](#phase-4-generating-recommendations)
6. [Statistical Methods Explained](#statistical-methods-explained)
7. [How to Interpret Results](#how-to-interpret-results)
8. [Limitations and Caveats](#limitations-and-caveats)

---

## The Big Picture

### What Factor Discovery Does

Factor Discovery answers the question: **"Which characteristics of stocks predict whether they'll beat the market?"**

For example:
- Do stocks with high Piotroski scores (strong fundamentals) tend to outperform?
- Do cheap stocks (low P/E ratio) beat expensive ones?
- Does combining multiple factors work better than using just one?

### The Basic Logic

1. **Look at the past**: We examine thousands of historical stock-quarter combinations
2. **Calculate alpha**: For each stock, we measure how much it beat (or lost to) the S&P 500
3. **Find patterns**: We test whether certain characteristics (factors) predicted higher alpha
4. **Validate statistically**: We use rigorous statistics to ensure patterns are real, not random noise
5. **Combine winners**: We test combinations of the best individual factors
6. **Recommend a strategy**: We output the best-performing combination as a trading strategy

---

## Phase 1: Building the Dataset

### What Happens

The system builds a massive spreadsheet where each row is one "observation"—a specific stock in a specific quarter with a specific holding period.

**Example row:**
| Symbol | Buy Quarter | Holding Period | Stock Return | SPY Return | Alpha | Piotroski | P/E | ROIC | ... |
|--------|-------------|----------------|--------------|------------|-------|-----------|-----|------|-----|
| AAPL   | 2023Q1      | 4 quarters     | 25.3%        | 12.1%      | 13.2% | 7         | 22  | 0.31 | ... |

### Alpha Calculation

**Technical**: `Alpha = Stock Return - SPY Return`

**Plain English**: Alpha is how much the stock beat (or lost to) the market. If Apple returned 25% and the S&P 500 returned 12%, Apple's alpha is +13%. This is what we're trying to predict.

### Data Lag (Look-Ahead Bias Prevention)

**Technical**: By default, `data_lag_quarters = 1`, meaning buy decisions in Q2 use analysis data from Q1.

**Plain English**: When you're deciding what to buy in April (Q2), you only know the financial data through December (Q1), because Q1 earnings haven't been released yet. We simulate this by using "stale" data, which is more realistic. Without this, we'd be cheating by using data that wasn't actually available at decision time.

### Dataset Size

A typical analysis might have:
- 20 quarters of data
- 4 holding periods (1Q, 2Q, 3Q, 4Q)
- ~3,000-5,000 stocks per quarter

Total: **~300,000-400,000 observations**

---

## Phase 2: Analyzing Individual Factors

This is where we test each factor separately to see if it predicts alpha.

### What We Test

We test **55+ factors** across categories:

| Category | Examples | What They Measure |
|----------|----------|-------------------|
| Scores | Piotroski, Graham, Altman | Pre-calculated quality/value scores |
| Valuation | P/E, P/B, EV/EBITDA | How cheap/expensive is the stock |
| Profitability | ROE, ROIC, margins | How profitable is the company |
| Leverage | Debt/Equity, Interest Coverage | Financial risk |
| Stability | ROIC Std Dev, FCF consistency | How consistent are results |
| Growth | EPS growth rates | Is the company growing |
| Quality Tags | Durable Compounder, Deep Value | Qualitative labels |

### Threshold Testing

For each numerical factor, we test multiple thresholds.

**Example: Piotroski Score (0-9 scale)**

We ask:
- Do stocks with Piotroski ≥ 3 beat those below 3?
- Do stocks with Piotroski ≥ 5 beat those below 5?
- Do stocks with Piotroski ≥ 7 beat those below 7?

For each threshold, we calculate statistics on the stocks that pass.

### The Statistics We Calculate

For each threshold, we compute:

#### 1. Mean Alpha

**Technical**: Average alpha of stocks passing the threshold

**Plain English**: On average, how much did stocks meeting this criterion beat the market?

**Example**: Stocks with Piotroski ≥ 6 had mean alpha of +4.2%, meaning they beat the S&P 500 by 4.2% on average.

#### 2. Sample Size

**Technical**: Number of observations (stock-quarter-holding_period combinations) meeting the threshold

**Plain English**: How many data points do we have? More is better—results from 1,000 observations are more trustworthy than results from 50.

**Minimum**: We require at least 100 observations (configurable) for any threshold to be considered.

#### 3. Win Rate

**Technical**: Percentage of observations with alpha > 0

**Plain English**: What percentage of these stocks actually beat the market? A 55% win rate means slightly more stocks won than lost.

#### 4. Lift

**Technical**: `Lift = Win Rate (threshold group) / Win Rate (all stocks)`

**Plain English**: How much better is this group at winning than average? A lift of 1.15 means this group wins 15% more often than a random stock.

**Example**: If all stocks have a 50% win rate, and stocks with Piotroski ≥ 6 have a 57.5% win rate, the lift is 57.5/50 = 1.15x.

#### 5. P-Value (Welch's t-test)

**Technical**: Probability that the observed difference in mean alpha between the threshold group and the rest happened by random chance, assuming no true difference exists.

**Plain English**: How confident are we that this pattern is real and not just luck?

| P-Value | Interpretation |
|---------|----------------|
| p < 0.01 | Very strong evidence (99%+ confident) |
| p < 0.05 | Strong evidence (95%+ confident) |
| p < 0.10 | Moderate evidence (90%+ confident) |
| p > 0.10 | Weak evidence (could easily be chance) |

**Example**: p = 0.003 means there's only a 0.3% chance that the difference we see is due to random variation.

#### 6. 95% Confidence Interval (Bootstrap)

**Technical**: Range within which we're 95% confident the true mean alpha falls, calculated via 1,000 bootstrap resamples.

**Plain English**: We're 95% sure the real average alpha is somewhere in this range.

**Example**: Mean alpha = 4.2% with 95% CI [2.1%, 6.3%] means we're confident the true alpha is between 2.1% and 6.3%. The fact that the entire range is positive is a good sign.

### Spearman Correlation

**Technical**: Rank correlation between factor value and alpha across all observations.

**Plain English**: Do higher values of this factor generally correspond to higher alpha? A correlation of +0.15 means "somewhat yes." A correlation of -0.10 means "higher values actually predict lower alpha."

**Why Spearman (not Pearson)?**: Spearman uses ranks, so it's not thrown off by outliers. If one stock has P/E of 1000, Pearson would be distorted, but Spearman just sees it as "highest P/E."

---

## Phase 3: Combining Factors

Single factors rarely tell the whole story. A cheap stock might be cheap because it's dying. By combining factors, we can filter for stocks that are cheap AND healthy.

### How Combinations Work

After individual factor analysis, we:

1. **Select top factors**: Take the 6-8 factors with highest lift that are statistically significant
2. **Test all combinations**: Try every combo of 1, 2, 3, and 4 factors
3. **Apply filters**: For each combo, filter the dataset to stocks passing ALL criteria
4. **Calculate stats**: Compute mean alpha, win rate, CI, etc. for each combo
5. **Rank by alpha**: Return the top 20 combinations

**Example Combination**:
- Piotroski ≥ 6 **AND**
- P/E ≤ 15 **AND**
- ROIC ≥ 12%

This filters for stocks with strong fundamentals (Piotroski), cheap valuation (P/E), and high profitability (ROIC).

### Why We Limit Factor Count

Testing more factors:
- 6 factors = 63 combinations (fast)
- 10 factors = 1,023 combinations (reasonable)
- 20 factors = 1,048,575 combinations (slow)
- 55 factors = astronomical (impossible)

We cap at the top factors to keep runtime reasonable while still finding good strategies.

### Portfolio Simulation

For each combination, we also simulate "what if you only bought the top N stocks per quarter?"

**Why?** You might not want to buy 500 stocks. We simulate portfolios of the top 10, 20, or 50 stocks, ranked by a method like Magic Formula or Earnings Yield.

**Result**: You might find that buying the top 20 Magic Formula stocks (that pass your filters) has a 6.5% alpha, while buying all stocks passing the filter has only 4.2% alpha. Concentration can help.

---

## Phase 4: Generating Recommendations

### Selecting the Best Strategy

We pick the strategy where:
1. The 95% CI lower bound exceeds the "cost haircut" (default 3%), OR
2. The mean alpha exceeds the cost haircut (if no strategy meets condition 1)

**Plain English**: We want strategies where we're confident the real alpha is at least 3%, not strategies that might be 0% with some unlucky variance.

### Confidence Score

The system calculates an overall confidence score (0-100%) based on:

| Component | Weight | What It Measures |
|-----------|--------|------------------|
| Sample size | 25% | More data = more confidence |
| CI width | 25% | Narrow CI = more precise estimate |
| Significance | 30% | CI lower bound > 0 = we're confident alpha is positive |
| Win rate | 20% | Higher win rate = more consistent |

### Pipeline Settings

The recommendation includes specific settings you can use in the Pipeline UI:
- Which survival gates to enable (Piotroski, Altman)
- Which valuation lenses to use (Graham, Magic Formula)
- Which quality tags to require or exclude
- Threshold values for each

---

## Statistical Methods Explained

### Welch's t-test

**What it is**: A statistical test comparing the means of two groups when we can't assume they have equal variance.

**Why we use it**: We're comparing "stocks passing the threshold" vs "stocks failing the threshold." These groups may have different spreads of returns, so Welch's test is more appropriate than a standard t-test.

**Formula** (simplified):
```
t = (mean₁ - mean₂) / sqrt(variance₁/n₁ + variance₂/n₂)
```

**Plain English**: We calculate how different the two averages are, accounting for how spread out each group is and how many observations we have.

### Bootstrap Confidence Intervals

**What it is**: A method to estimate uncertainty by repeatedly resampling our data.

**How it works**:
1. Take our 1,000 observations
2. Randomly pick 1,000 with replacement (some picked twice, some not at all)
3. Calculate the mean of this sample
4. Repeat 1,000 times
5. The middle 95% of those means is our CI

**Why we use it**: Bootstrap makes no assumptions about the distribution of returns. Stock returns can be weird (heavy tails, skewed), so bootstrap is more robust than assuming normal distribution.

**Plain English**: We simulate "what if we had different samples from the same universe" a thousand times, and see what range of results we get.

### Benjamini-Hochberg FDR Correction

**The Problem (Multiple Testing)**:

If we test 50 factors at p < 0.05, we'd expect 2-3 to appear significant **by pure chance**, even if none actually predict alpha.

**The Solution**:

FDR (False Discovery Rate) correction adjusts our p-values to account for multiple testing.

**How it works**:
1. Sort all p-values from smallest to largest
2. For each, calculate an adjusted threshold: `(rank / total_tests) * 0.05`
3. Find the largest p-value that beats its threshold
4. All tests with p-values at or below that one are "FDR significant"

**Plain English**: Instead of asking "is this single result significant?", we ask "of all the things I'm calling significant, what fraction might be false alarms?" FDR keeps the false alarm rate under control.

**Example**:
- We test 200 thresholds
- At p < 0.05, we'd expect 10 false positives
- After FDR correction, maybe only 15 pass instead of 30, but those 15 are more trustworthy

---

## How to Interpret Results

### What "Good" Looks Like

| Metric | Poor | OK | Good | Excellent |
|--------|------|-------|------|-----------|
| Mean Alpha | < 2% | 2-4% | 4-8% | > 8% |
| Win Rate | < 52% | 52-55% | 55-60% | > 60% |
| Lift | < 1.05 | 1.05-1.10 | 1.10-1.20 | > 1.20 |
| P-Value | > 0.10 | 0.05-0.10 | 0.01-0.05 | < 0.01 |
| CI Lower | < 0% | 0-2% | 2-4% | > 4% |
| Sample Size | < 200 | 200-500 | 500-1000 | > 1000 |
| FDR Significant | No | - | Yes | Yes |

### Red Flags

1. **Large sample size but small effect**: 10,000 observations with 0.5% alpha might be statistically significant but economically meaningless.

2. **CI crosses zero**: If the 95% CI is [-1%, +5%], there's a reasonable chance the true alpha is negative.

3. **Very few stocks pass**: A strategy with 50 historical observations is unreliable, even if stats look good.

4. **Non-FDR-significant**: If a factor passes raw p < 0.05 but fails FDR correction, be skeptical.

5. **Out-of-sample degradation**: If train alpha is 8% but validation alpha is 2%, the strategy may be overfit.

### The Overfit Ratio

When using train/validation/test splits:

**Overfit Ratio = Validation Alpha / Training Alpha**

| Ratio | Interpretation |
|-------|----------------|
| > 0.8 | Great! Strategy holds up out-of-sample |
| 0.5-0.8 | Acceptable, some overfitting but core signal persists |
| 0.3-0.5 | Concerning, significant overfitting |
| < 0.3 | Strategy is probably overfit, don't trust training results |

---

## Limitations and Caveats

### What We Can't Account For

1. **Survivorship Bias**: Our database only has currently active stocks. Delisted companies (bankruptcies, acquisitions) are missing, which may inflate returns.

2. **Implementation Slippage**: We assume you can buy at the exact quarter-end price. In reality, there's slippage.

3. **Capacity**: Strategies that work great on paper may fail when you try to deploy $10M because you move the market.

4. **Regime Change**: A factor that worked 2015-2023 might not work 2024+ if market dynamics change.

### The 30-50% Rule

**Expect live performance to be 30-50% of backtested performance.**

If the system shows 10% alpha, plan for 3-5% in reality after accounting for:
- Overfitting to historical data
- Implementation costs
- Survivorship bias
- Data look-ahead (even with our lag, some subtle look-ahead may exist)

### Statistical vs Economic Significance

A result can be:
- **Statistically significant but economically worthless**: p = 0.001, alpha = 0.3%
- **Economically interesting but statistically noisy**: alpha = 5%, p = 0.15

Ideally, you want both: high alpha AND low p-value AND narrow CI with lower bound > 0.

---

## Summary

Factor Discovery is a systematic way to answer: "What predicted stock outperformance in the past?"

It uses:
- **Large datasets**: Hundreds of thousands of observations
- **Multiple factors**: 55+ characteristics tested
- **Rigorous statistics**: t-tests, bootstrap CIs, FDR correction
- **Combination search**: Finding multi-factor strategies
- **Realistic simulation**: Data lag, portfolio constraints

The output is a recommended strategy with expected alpha, confidence intervals, and the specific filters to apply.

**Remember**: Past performance doesn't guarantee future results, but a strategy with strong historical evidence, robust statistics, and good out-of-sample performance is more trustworthy than a hunch.
