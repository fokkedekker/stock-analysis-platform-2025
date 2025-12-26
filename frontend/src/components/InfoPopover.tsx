"use client"

import { RiQuestionLine } from "@remixicon/react"
import { Popover, PopoverContent, PopoverTrigger } from "@/components/Popover"

interface InfoPopoverProps {
  title: string
  children: React.ReactNode
}

export function InfoPopover({ title, children }: InfoPopoverProps) {
  return (
    <Popover>
      <PopoverTrigger asChild>
        <button
          onClick={(e) => e.stopPropagation()}
          className="inline-flex items-center justify-center w-4 h-4 rounded-full text-gray-400 hover:text-indigo-600 dark:hover:text-indigo-400 transition-colors focus:outline-none"
          aria-label={`Info about ${title}`}
        >
          <RiQuestionLine className="w-4 h-4" />
        </button>
      </PopoverTrigger>
      <PopoverContent
        className="w-80 p-4 shadow-lg rounded-xl border border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-900"
        side="top"
        align="center"
        sideOffset={8}
      >
        <div className="space-y-3">
          <h4 className="font-semibold text-base text-gray-900 dark:text-gray-100 border-b border-gray-100 dark:border-gray-800 pb-2">{title}</h4>
          <div className="text-sm text-gray-600 dark:text-gray-300 space-y-2 leading-relaxed">
            {children}
          </div>
        </div>
      </PopoverContent>
    </Popover>
  )
}

// Pre-built info content for each metric
export function AltmanZScoreInfo() {
  return (
    <InfoPopover title="Altman Z-Score">
      <p>
        <strong>What it is:</strong> A formula that predicts the likelihood of a company going bankrupt within 2 years. Created by NYU professor Edward Altman in 1968.
      </p>
      <p>
        <strong>Why it matters:</strong> Helps filter out financially distressed companies before investing. Even a cheap stock is a bad investment if the company goes bankrupt.
      </p>
      <div className="mt-2 space-y-1">
        <p className="font-medium text-gray-700 dark:text-gray-300">Score Zones:</p>
        <ul className="list-disc list-inside space-y-0.5 text-xs">
          <li><span className="text-green-600 dark:text-green-400 font-medium">Safe Zone (&gt;2.99)</span> - Low bankruptcy risk</li>
          <li><span className="text-yellow-600 dark:text-yellow-400 font-medium">Grey Zone (1.81-2.99)</span> - Some risk, needs monitoring</li>
          <li><span className="text-red-600 dark:text-red-400 font-medium">Distress Zone (&lt;1.81)</span> - High bankruptcy risk</li>
        </ul>
      </div>
      <div className="mt-2 space-y-1">
        <p className="font-medium text-gray-700 dark:text-gray-300">Options:</p>
        <ul className="list-disc list-inside space-y-0.5 text-xs">
          <li><strong>Safe zone only</strong> - Most conservative, only Z &gt; 2.99</li>
          <li><strong>Include grey zone</strong> - Allows Z &gt; 1.81 for more opportunities</li>
        </ul>
      </div>
    </InfoPopover>
  )
}

export function PiotroskiFScoreInfo() {
  return (
    <InfoPopover title="Piotroski F-Score">
      <p>
        <strong>What it is:</strong> A 9-point scoring system that measures a company's financial strength. Created by Stanford professor Joseph Piotroski in 2000.
      </p>
      <p>
        <strong>Why it matters:</strong> Identifies companies with strong and improving fundamentals. High F-Score companies historically outperform low F-Score companies.
      </p>
      <div className="mt-2 space-y-1">
        <p className="font-medium text-gray-700 dark:text-gray-300">Score Ranges:</p>
        <ul className="list-disc list-inside space-y-0.5 text-xs">
          <li><span className="text-green-600 dark:text-green-400 font-medium">7-9</span> - Strong financials (buy candidates)</li>
          <li><span className="text-yellow-600 dark:text-yellow-400 font-medium">5-6</span> - Average financials</li>
          <li><span className="text-red-600 dark:text-red-400 font-medium">0-4</span> - Weak financials (avoid)</li>
        </ul>
      </div>
      <div className="mt-2 space-y-1">
        <p className="font-medium text-gray-700 dark:text-gray-300">9 Signals Measured:</p>
        <ul className="list-disc list-inside space-y-0.5 text-xs">
          <li>Profitability (4): ROA, Operating CF, ROA trend, Accruals</li>
          <li>Leverage (3): Debt ratio, Current ratio, Share dilution</li>
          <li>Efficiency (2): Gross margin, Asset turnover</li>
        </ul>
      </div>
    </InfoPopover>
  )
}

export function GrahamInfo() {
  return (
    <InfoPopover title="Graham Number">
      <p>
        <strong>What it is:</strong> A valuation method from Benjamin Graham (Warren Buffett's mentor) that scores stocks on 8 criteria focusing on value and safety.
      </p>
      <p>
        <strong>Why it matters:</strong> Identifies undervalued stocks with a margin of safety. Graham's principles are the foundation of value investing.
      </p>
      <div className="mt-2 space-y-1">
        <p className="font-medium text-gray-700 dark:text-gray-300">8 Criteria:</p>
        <ul className="list-disc list-inside space-y-0.5 text-xs">
          <li>Adequate company size</li>
          <li>Strong financial condition (current ratio)</li>
          <li>Earnings stability (10 years positive)</li>
          <li>Dividend record</li>
          <li>Earnings growth</li>
          <li>Moderate P/E ratio</li>
          <li>Moderate price-to-book</li>
          <li>Graham Number discount</li>
        </ul>
      </div>
      <div className="mt-2 space-y-1">
        <p className="font-medium text-gray-700 dark:text-gray-300">Modes:</p>
        <ul className="list-disc list-inside space-y-0.5 text-xs">
          <li><strong>Strict</strong> - Original Graham criteria (hard to pass)</li>
          <li><strong>Modern</strong> - Updated thresholds for today's market</li>
          <li><strong>GARP</strong> - Growth at reasonable price focus</li>
          <li><strong>Relaxed</strong> - Most lenient interpretation</li>
        </ul>
      </div>
    </InfoPopover>
  )
}

export function NetNetInfo() {
  return (
    <InfoPopover title="Net-Net (NCAV)">
      <p>
        <strong>What it is:</strong> A deep value strategy where you buy stocks trading below their Net Current Asset Value (current assets minus ALL liabilities).
      </p>
      <p>
        <strong>Why it matters:</strong> These are extreme bargains - you're paying less than the company's liquidation value. Very rare but historically very profitable.
      </p>
      <div className="mt-2 space-y-1">
        <p className="font-medium text-gray-700 dark:text-gray-300">The Math:</p>
        <p className="text-xs">NCAV = Current Assets - Total Liabilities</p>
        <p className="text-xs">If Stock Price &lt; NCAV per share = Net-Net</p>
      </div>
      <div className="mt-2 space-y-1">
        <p className="font-medium text-gray-700 dark:text-gray-300">Key Points:</p>
        <ul className="list-disc list-inside space-y-0.5 text-xs">
          <li>Very conservative - ignores fixed assets entirely</li>
          <li>Rare to find in modern markets (more common in small caps)</li>
          <li>Graham's favorite strategy for individual investors</li>
        </ul>
      </div>
    </InfoPopover>
  )
}

export function PEGInfo() {
  return (
    <InfoPopover title="PEG Ratio">
      <p>
        <strong>What it is:</strong> Price/Earnings divided by Earnings Growth rate. Popularized by Peter Lynch to find growth stocks at reasonable prices.
      </p>
      <p>
        <strong>Why it matters:</strong> A low P/E alone might miss great growth companies. PEG accounts for growth, helping identify undervalued growth stocks.
      </p>
      <div className="mt-2 space-y-1">
        <p className="font-medium text-gray-700 dark:text-gray-300">The Math:</p>
        <p className="text-xs">PEG = P/E Ratio / Expected Earnings Growth %</p>
        <p className="text-xs">Example: P/E of 20 with 20% growth = PEG of 1.0</p>
      </div>
      <div className="mt-2 space-y-1">
        <p className="font-medium text-gray-700 dark:text-gray-300">Thresholds:</p>
        <ul className="list-disc list-inside space-y-0.5 text-xs">
          <li><span className="text-green-600 dark:text-green-400 font-medium">&lt;1.0</span> - Undervalued (great buy signal)</li>
          <li><span className="text-yellow-600 dark:text-yellow-400 font-medium">1.0-1.5</span> - Fairly valued</li>
          <li><span className="text-red-600 dark:text-red-400 font-medium">&gt;2.0</span> - Likely overvalued for growth</li>
        </ul>
      </div>
      <div className="mt-2">
        <p className="text-xs"><strong>Max PEG setting:</strong> Filters for stocks with PEG below this value.</p>
      </div>
    </InfoPopover>
  )
}

export function MagicFormulaInfo() {
  return (
    <InfoPopover title="Magic Formula">
      <p>
        <strong>What it is:</strong> Joel Greenblatt's ranking system that combines quality (ROIC) with value (Earnings Yield) to find cheap, high-quality stocks.
      </p>
      <p>
        <strong>Why it matters:</strong> Systematically identifies companies that are both good businesses AND cheap. Backtests showed 30%+ annual returns.
      </p>
      <div className="mt-2 space-y-1">
        <p className="font-medium text-gray-700 dark:text-gray-300">How it Works:</p>
        <ul className="list-disc list-inside space-y-0.5 text-xs">
          <li>Rank all stocks by ROIC (high = good)</li>
          <li>Rank all stocks by Earnings Yield (high = good)</li>
          <li>Add the two ranks together</li>
          <li>Lowest combined rank = best opportunities</li>
        </ul>
      </div>
      <div className="mt-2 space-y-1">
        <p className="font-medium text-gray-700 dark:text-gray-300">Top % Setting:</p>
        <ul className="list-disc list-inside space-y-0.5 text-xs">
          <li><strong>Top 10%</strong> - Only the highest-ranked stocks</li>
          <li><strong>Top 20%</strong> - Good balance of quality and selection</li>
          <li><strong>Top 30-50%</strong> - More stocks, less selective</li>
        </ul>
      </div>
    </InfoPopover>
  )
}

export function FamaFrenchBMInfo() {
  return (
    <InfoPopover title="Fama-French Book-to-Market">
      <p>
        <strong>What it is:</strong> Based on Nobel Prize-winning research showing that high book-to-market (value) stocks outperform low book-to-market (growth) stocks over time.
      </p>
      <p>
        <strong>Why it matters:</strong> Captures the "value premium" - one of the most documented anomalies in finance. Value stocks historically outperform by 3-5% annually.
      </p>
      <div className="mt-2 space-y-1">
        <p className="font-medium text-gray-700 dark:text-gray-300">The Math:</p>
        <p className="text-xs">Book-to-Market = Book Value / Market Cap</p>
        <p className="text-xs">High B/M = Value stock, Low B/M = Growth stock</p>
      </div>
      <div className="mt-2 space-y-1">
        <p className="font-medium text-gray-700 dark:text-gray-300">Top % Setting:</p>
        <ul className="list-disc list-inside space-y-0.5 text-xs">
          <li><strong>Top 20%</strong> - Deep value stocks only</li>
          <li><strong>Top 30%</strong> - Standard value threshold</li>
          <li><strong>Top 40-50%</strong> - Moderate value tilt</li>
        </ul>
      </div>
      <div className="mt-2">
        <p className="text-xs"><strong>Note:</strong> Fama-French also identified size and momentum factors, but we focus on value here.</p>
      </div>
    </InfoPopover>
  )
}
