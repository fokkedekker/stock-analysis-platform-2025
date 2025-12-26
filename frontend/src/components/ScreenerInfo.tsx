"use client"

import { useState } from "react"
import { RiInformationLine } from "@remixicon/react"
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
  DialogClose,
} from "@/components/Dialog"
import { Button } from "@/components/Button"

interface Criterion {
  name: string
  description: string
  threshold?: string
}

interface ScreenerInfoProps {
  title: string
  description: string
  source?: string
  criteria: Criterion[]
  scoring?: string
  interpretation?: { label: string; description: string }[]
}

export function ScreenerInfo({
  title,
  description,
  source,
  criteria,
  scoring,
  interpretation,
}: ScreenerInfoProps) {
  const [open, setOpen] = useState(false)

  return (
    <Dialog open={open} onOpenChange={setOpen}>
      <DialogTrigger asChild>
        <button
          className="inline-flex items-center gap-1.5 text-sm text-gray-500 hover:text-gray-700 dark:hover:text-gray-300 transition-colors"
          title="Learn more about this screener"
        >
          <RiInformationLine className="w-4 h-4" />
          <span>How it works</span>
        </button>
      </DialogTrigger>
      <DialogContent className="max-w-2xl max-h-[85vh] overflow-y-auto">
        <DialogHeader>
          <DialogTitle>{title}</DialogTitle>
        </DialogHeader>

        <div className="mt-4 space-y-6 text-sm">
          <div>
            <p className="text-gray-600 dark:text-gray-400">{description}</p>
            {source && (
              <p className="mt-2 text-xs text-gray-500">Source: {source}</p>
            )}
          </div>

          <div>
            <h3 className="font-semibold text-gray-900 dark:text-gray-100 mb-3">
              Criteria
            </h3>
            <div className="space-y-3">
              {criteria.map((criterion, idx) => (
                <div
                  key={idx}
                  className="p-3 rounded-lg bg-gray-50 dark:bg-gray-900 border border-gray-200 dark:border-gray-800"
                >
                  <div className="font-medium text-gray-900 dark:text-gray-100">
                    {criterion.name}
                  </div>
                  <div className="text-gray-600 dark:text-gray-400 mt-1">
                    {criterion.description}
                  </div>
                  {criterion.threshold && (
                    <div className="text-xs text-indigo-600 dark:text-indigo-400 mt-1">
                      Threshold: {criterion.threshold}
                    </div>
                  )}
                </div>
              ))}
            </div>
          </div>

          {scoring && (
            <div>
              <h3 className="font-semibold text-gray-900 dark:text-gray-100 mb-2">
                Scoring
              </h3>
              <p className="text-gray-600 dark:text-gray-400">{scoring}</p>
            </div>
          )}

          {interpretation && interpretation.length > 0 && (
            <div>
              <h3 className="font-semibold text-gray-900 dark:text-gray-100 mb-3">
                Interpretation
              </h3>
              <div className="space-y-2">
                {interpretation.map((item, idx) => (
                  <div key={idx} className="flex gap-3">
                    <span className="font-medium text-gray-900 dark:text-gray-100 min-w-[80px]">
                      {item.label}:
                    </span>
                    <span className="text-gray-600 dark:text-gray-400">
                      {item.description}
                    </span>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>

        <div className="mt-6 flex justify-end">
          <DialogClose asChild>
            <Button variant="secondary">Close</Button>
          </DialogClose>
        </div>
      </DialogContent>
    </Dialog>
  )
}

// Pre-configured screener info data
export const SCREENER_INFO = {
  graham: {
    title: "Benjamin Graham's 7 Criteria",
    description:
      "Benjamin Graham, the father of value investing, outlined these criteria in 'The Intelligent Investor' to identify financially sound companies trading at reasonable valuations. These defensive criteria are designed to minimize downside risk.",
    source: "The Intelligent Investor (1949)",
    criteria: [
      {
        name: "Adequate Size",
        description: "Company should have sufficient revenue to ensure stability",
        threshold: "Revenue > $500M (modern) or $100M (relaxed)",
      },
      {
        name: "Current Ratio",
        description: "Current assets should cover current liabilities comfortably",
        threshold: "Current Ratio ≥ 2.0",
      },
      {
        name: "Debt Coverage",
        description: "Long-term debt should be covered by working capital",
        threshold: "LT Debt < Net Current Assets",
      },
      {
        name: "Earnings Stability",
        description: "Positive earnings in each of the past 10 years",
        threshold: "No losses in 10 years",
      },
      {
        name: "Dividend Record",
        description: "Uninterrupted dividend payments",
        threshold: "20 years (strict) or 10 years (modern)",
      },
      {
        name: "Earnings Growth",
        description: "Minimum growth in per-share earnings",
        threshold: "33% growth over 10 years (≈3% annually)",
      },
      {
        name: "P/E Ratio",
        description: "Price relative to average 3-year earnings",
        threshold: "P/E ≤ 15",
      },
      {
        name: "P/B Ratio",
        description: "Price relative to book value (or P/E × P/B ≤ 22.5)",
        threshold: "P/B ≤ 1.5 or P/E × P/B ≤ 22.5",
      },
    ],
    scoring: "Each criterion passed adds 1 point. Maximum score is 7.",
    interpretation: [
      { label: "7/7", description: "Perfect Graham stock - rare" },
      { label: "5-6", description: "Strong value candidate" },
      { label: "3-4", description: "Moderate - review individual criteria" },
      { label: "0-2", description: "Does not meet Graham's standards" },
    ],
  },

  magicFormula: {
    title: "Joel Greenblatt's Magic Formula",
    description:
      "The Magic Formula ranks stocks by combining earnings yield (cheap) and return on capital (quality). Buy the top-ranked stocks that are both cheap AND high-quality businesses.",
    source: "The Little Book That Beats the Market (2005)",
    criteria: [
      {
        name: "Earnings Yield",
        description: "EBIT / Enterprise Value - measures how cheap a stock is",
        threshold: "Ranked vs all stocks (lower rank = higher yield)",
      },
      {
        name: "Return on Capital",
        description: "EBIT / (Net Fixed Assets + Working Capital) - measures business quality",
        threshold: "Ranked vs all stocks (lower rank = higher ROC)",
      },
    ],
    scoring:
      "Combined Rank = Earnings Yield Rank + Return on Capital Rank. Lower combined rank is better.",
    interpretation: [
      { label: "Top 30", description: "Best candidates - consider buying" },
      { label: "Top 50", description: "Strong candidates" },
      { label: "Top 100", description: "Worth investigating" },
    ],
  },

  piotroski: {
    title: "Piotroski F-Score",
    description:
      "Joseph Piotroski developed this 9-point scoring system to identify financially strong value stocks. Each binary signal tests profitability, leverage/liquidity, or operating efficiency.",
    source: "Journal of Accounting Research (2000)",
    criteria: [
      {
        name: "ROA Positive",
        description: "Net income / Total assets > 0",
        threshold: "Positive ROA = 1 point",
      },
      {
        name: "Operating Cash Flow Positive",
        description: "Cash from operations > 0",
        threshold: "Positive OCF = 1 point",
      },
      {
        name: "ROA Increasing",
        description: "Current ROA > Prior year ROA",
        threshold: "YoY improvement = 1 point",
      },
      {
        name: "Accruals Signal",
        description: "Cash from operations > Net income (quality of earnings)",
        threshold: "OCF > Net Income = 1 point",
      },
      {
        name: "Leverage Decreasing",
        description: "Long-term debt ratio decreased",
        threshold: "Lower debt ratio = 1 point",
      },
      {
        name: "Current Ratio Increasing",
        description: "Improved liquidity position",
        threshold: "Higher current ratio = 1 point",
      },
      {
        name: "No Dilution",
        description: "No new shares issued",
        threshold: "Shares unchanged or decreased = 1 point",
      },
      {
        name: "Gross Margin Increasing",
        description: "Improved pricing power or efficiency",
        threshold: "Higher gross margin = 1 point",
      },
      {
        name: "Asset Turnover Increasing",
        description: "Better asset utilization",
        threshold: "Higher turnover = 1 point",
      },
    ],
    scoring: "Sum of all 9 binary signals (0 or 1 each). Maximum score is 9.",
    interpretation: [
      { label: "8-9", description: "Strong - high probability of outperformance" },
      { label: "7", description: "Good financial health" },
      { label: "5-6", description: "Average" },
      { label: "0-4", description: "Weak - potential value trap or distress" },
    ],
  },

  altman: {
    title: "Altman Z-Score",
    description:
      "Edward Altman's bankruptcy prediction model uses 5 financial ratios to assess a company's financial health and probability of bankruptcy within 2 years.",
    source: "Journal of Finance (1968)",
    criteria: [
      {
        name: "X1: Working Capital / Total Assets",
        description: "Measures liquidity relative to company size",
        threshold: "Coefficient: 1.2",
      },
      {
        name: "X2: Retained Earnings / Total Assets",
        description: "Measures cumulative profitability and age",
        threshold: "Coefficient: 1.4",
      },
      {
        name: "X3: EBIT / Total Assets",
        description: "Measures operating efficiency",
        threshold: "Coefficient: 3.3",
      },
      {
        name: "X4: Market Cap / Total Liabilities",
        description: "Measures solvency (market-based)",
        threshold: "Coefficient: 0.6",
      },
      {
        name: "X5: Revenue / Total Assets",
        description: "Measures asset turnover",
        threshold: "Coefficient: 1.0",
      },
    ],
    scoring:
      "Z = 1.2×X1 + 1.4×X2 + 3.3×X3 + 0.6×X4 + 1.0×X5",
    interpretation: [
      { label: "Z > 2.99", description: "Safe Zone - low bankruptcy risk" },
      { label: "1.81 - 2.99", description: "Grey Zone - moderate risk, needs monitoring" },
      { label: "Z < 1.81", description: "Distress Zone - high bankruptcy risk" },
    ],
  },

  roic: {
    title: "ROIC Quality Screen",
    description:
      "Return on Invested Capital measures how efficiently a company generates returns from its capital investments. High ROIC indicates a competitive moat.",
    source: "Various value investing frameworks",
    criteria: [
      {
        name: "ROIC",
        description: "NOPAT / Invested Capital - core profitability measure",
        threshold: "ROIC > 15% (good) or > 20% (excellent)",
      },
      {
        name: "Positive Free Cash Flow",
        description: "Company generates real cash, not just accounting profits",
        threshold: "FCF positive for 5 consecutive years",
      },
      {
        name: "Reasonable Debt",
        description: "Not overleveraged",
        threshold: "Debt/Equity < 1.0",
      },
    ],
    scoring: "Stocks are ranked by ROIC. Higher is better.",
    interpretation: [
      { label: "> 25%", description: "Exceptional - likely has strong moat" },
      { label: "15-25%", description: "Good - above cost of capital" },
      { label: "10-15%", description: "Average" },
      { label: "< 10%", description: "Poor - may be destroying value" },
    ],
  },

  peg: {
    title: "PEG Ratio (GARP)",
    description:
      "Growth at a Reasonable Price (GARP) uses the PEG ratio to find growing companies that aren't overvalued. PEG balances P/E against expected growth.",
    source: "Peter Lynch, One Up on Wall Street",
    criteria: [
      {
        name: "P/E Ratio",
        description: "Price / Earnings - valuation measure",
        threshold: "Used in PEG calculation",
      },
      {
        name: "EPS Growth Rate",
        description: "Historical or expected earnings growth",
        threshold: "Positive growth required",
      },
      {
        name: "PEG Ratio",
        description: "P/E divided by growth rate",
        threshold: "PEG ≤ 1.0 is attractive",
      },
    ],
    scoring: "PEG = P/E ÷ EPS Growth Rate. Lower is better.",
    interpretation: [
      { label: "< 0.5", description: "Potentially very undervalued" },
      { label: "0.5 - 1.0", description: "Fairly valued for growth" },
      { label: "1.0 - 2.0", description: "May be fully valued" },
      { label: "> 2.0", description: "Expensive relative to growth" },
    ],
  },

  famaFrench: {
    title: "Fama-French Factor Screen",
    description:
      "Based on Nobel Prize-winning research, this screen identifies stocks with factor exposures historically associated with higher returns: small size, high value, high profitability, and low investment.",
    source: "Fama & French (1993, 2015)",
    criteria: [
      {
        name: "Book-to-Market (Value)",
        description: "High B/M stocks historically outperform",
        threshold: "Higher percentile = more value exposure",
      },
      {
        name: "Profitability",
        description: "Operating profitability / Book equity",
        threshold: "Higher percentile = stronger profitability",
      },
      {
        name: "Asset Growth (Investment)",
        description: "Low asset growth stocks outperform",
        threshold: "Lower percentile = conservative investment",
      },
    ],
    scoring:
      "Stocks ranked by percentile in each factor. Ideal: high value, high profitability, low investment.",
  },

  netNet: {
    title: "Net-Net (NCAV) Strategy",
    description:
      "Benjamin Graham's most conservative strategy: buy stocks trading below their Net Current Asset Value (current assets minus ALL liabilities). These are essentially priced below liquidation value.",
    source: "Security Analysis (1934)",
    criteria: [
      {
        name: "NCAV Calculation",
        description: "Net Current Asset Value = Current Assets - Total Liabilities",
        threshold: "Must be positive",
      },
      {
        name: "Discount to NCAV",
        description: "Market cap compared to NCAV",
        threshold: "Trading below 100% NCAV",
      },
      {
        name: "Deep Value",
        description: "Graham's preferred margin of safety",
        threshold: "Trading below 66% of NCAV",
      },
    ],
    scoring: "Discount = Market Cap / NCAV. Lower percentage means deeper value.",
    interpretation: [
      { label: "< 66%", description: "Deep value - Graham's preferred zone" },
      { label: "66-100%", description: "Trading below liquidation value" },
      { label: "> 100%", description: "Not a net-net" },
    ],
  },
}
