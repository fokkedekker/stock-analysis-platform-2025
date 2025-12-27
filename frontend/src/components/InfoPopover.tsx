"use client"

import { RiQuestionLine, RiCloseLine } from "@remixicon/react"
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
  DialogClose,
} from "@/components/Dialog"

interface InfoModalProps {
  title: string
  children: React.ReactNode
}

// Base modal component for all info dialogs
export function InfoPopover({ title, children }: InfoModalProps) {
  return (
    <Dialog>
      <DialogTrigger asChild>
        <button
          onClick={(e) => e.stopPropagation()}
          className="inline-flex items-center justify-center w-4 h-4 rounded-full text-gray-400 hover:text-indigo-600 dark:hover:text-indigo-400 transition-colors focus:outline-none"
          aria-label={`Info about ${title}`}
        >
          <RiQuestionLine className="w-4 h-4" />
        </button>
      </DialogTrigger>
      <DialogContent className="max-w-xl max-h-[85vh] overflow-y-auto">
        <DialogHeader className="flex flex-row items-center justify-between">
          <DialogTitle>{title}</DialogTitle>
          <DialogClose asChild>
            <button className="text-gray-400 hover:text-gray-600 dark:hover:text-gray-300">
              <RiCloseLine className="w-5 h-5" />
            </button>
          </DialogClose>
        </DialogHeader>
        <div className="mt-4 text-sm text-gray-600 dark:text-gray-300 space-y-4 leading-relaxed">
          {children}
        </div>
      </DialogContent>
    </Dialog>
  )
}

// Pre-built info content for each metric
export function AltmanZScoreInfo() {
  return (
    <InfoPopover title="Altman Z-Score">
      <p>
        <strong>What this means for the stock:</strong> The Z-Score tells you if this company is at risk of going bankrupt. A low score means the company might not survive - and even the cheapest stock is worthless if the company goes bust.
      </p>

      <div className="bg-gray-50 dark:bg-gray-800 rounded-lg p-4 space-y-3">
        <h4 className="font-semibold text-gray-900 dark:text-gray-100">Score Zones</h4>
        <div className="space-y-2">
          <div className="flex items-start gap-3">
            <span className="px-2 py-1 rounded text-xs font-medium bg-green-100 text-green-700 dark:bg-green-900 dark:text-green-300">Safe Zone (&gt;2.99)</span>
            <p className="text-sm">Financially healthy. Low bankruptcy risk. Safe to analyze further based on valuation.</p>
          </div>
          <div className="flex items-start gap-3">
            <span className="px-2 py-1 rounded text-xs font-medium bg-yellow-100 text-yellow-700 dark:bg-yellow-900 dark:text-yellow-300">Grey Zone (1.81-2.99)</span>
            <p className="text-sm">Caution zone. Company has some financial stress. Could go either way - needs monitoring.</p>
          </div>
          <div className="flex items-start gap-3">
            <span className="px-2 py-1 rounded text-xs font-medium bg-red-100 text-red-700 dark:bg-red-900 dark:text-red-300">Distress Zone (&lt;1.81)</span>
            <p className="text-sm">Danger! High probability of bankruptcy within 2 years. Avoid unless you're a distressed debt specialist.</p>
          </div>
        </div>
      </div>

      <div>
        <h4 className="font-semibold text-gray-900 dark:text-gray-100 mb-2">Filter Options</h4>
        <ul className="space-y-1 text-sm">
          <li><strong>Safe zone only</strong> - Most conservative. Only shows companies with Z &gt; 2.99. Recommended for most investors.</li>
          <li><strong>Include grey zone</strong> - Shows Z &gt; 1.81. More opportunities but higher risk. Use if you're comfortable with turnarounds.</li>
        </ul>
      </div>
    </InfoPopover>
  )
}

export function PiotroskiFScoreInfo() {
  return (
    <InfoPopover title="Piotroski F-Score">
      <p>
        <strong>What this means for the stock:</strong> The F-Score measures financial health and momentum. A high score means the company's fundamentals are strong AND improving. Companies with high F-Scores historically outperform.
      </p>

      <div className="bg-gray-50 dark:bg-gray-800 rounded-lg p-4 space-y-3">
        <h4 className="font-semibold text-gray-900 dark:text-gray-100">Score Ranges</h4>
        <div className="space-y-2">
          <div className="flex items-start gap-3">
            <span className="px-2 py-1 rounded text-xs font-medium bg-green-100 text-green-700 dark:bg-green-900 dark:text-green-300">7-9 Points</span>
            <p className="text-sm">Strong financials with positive trends. These companies are getting healthier. Great buy candidates.</p>
          </div>
          <div className="flex items-start gap-3">
            <span className="px-2 py-1 rounded text-xs font-medium bg-yellow-100 text-yellow-700 dark:bg-yellow-900 dark:text-yellow-300">5-6 Points</span>
            <p className="text-sm">Mixed signals. Some things improving, others not. Requires deeper analysis.</p>
          </div>
          <div className="flex items-start gap-3">
            <span className="px-2 py-1 rounded text-xs font-medium bg-red-100 text-red-700 dark:bg-red-900 dark:text-red-300">0-4 Points</span>
            <p className="text-sm">Weak and deteriorating fundamentals. Financial health is declining. Avoid or short.</p>
          </div>
        </div>
      </div>

      <div>
        <h4 className="font-semibold text-gray-900 dark:text-gray-100 mb-2">The 9 Signals (1 point each)</h4>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-3 text-sm">
          <div>
            <p className="font-medium text-gray-700 dark:text-gray-300">Profitability (4)</p>
            <ul className="text-xs text-gray-500 space-y-0.5">
              <li>• Positive ROA</li>
              <li>• Positive Operating CF</li>
              <li>• ROA improving</li>
              <li>• CF &gt; Net Income</li>
            </ul>
          </div>
          <div>
            <p className="font-medium text-gray-700 dark:text-gray-300">Leverage (3)</p>
            <ul className="text-xs text-gray-500 space-y-0.5">
              <li>• Debt ratio down</li>
              <li>• Current ratio up</li>
              <li>• No share dilution</li>
            </ul>
          </div>
          <div>
            <p className="font-medium text-gray-700 dark:text-gray-300">Efficiency (2)</p>
            <ul className="text-xs text-gray-500 space-y-0.5">
              <li>• Gross margin up</li>
              <li>• Asset turnover up</li>
            </ul>
          </div>
        </div>
      </div>
    </InfoPopover>
  )
}

export function GrahamInfo() {
  return (
    <InfoPopover title="Graham Number">
      <p>
        <strong>What this means for the stock:</strong> Benjamin Graham (Warren Buffett's mentor) developed these criteria to find stocks with a "margin of safety." A high Graham score means the stock is cheap, the company is financially solid, and has a track record of consistent earnings.
      </p>

      <div className="bg-gray-50 dark:bg-gray-800 rounded-lg p-4">
        <h4 className="font-semibold text-gray-900 dark:text-gray-100 mb-3">The 8 Criteria</h4>
        <div className="space-y-2 text-sm">
          <div className="flex items-start gap-2">
            <span className="text-green-600">✓</span>
            <div><strong>Adequate Size</strong> - Large enough to be stable (avoids risky micro-caps)</div>
          </div>
          <div className="flex items-start gap-2">
            <span className="text-green-600">✓</span>
            <div><strong>Strong Financial Condition</strong> - Current assets cover current liabilities 2:1</div>
          </div>
          <div className="flex items-start gap-2">
            <span className="text-green-600">✓</span>
            <div><strong>Earnings Stability</strong> - Positive earnings for 10 consecutive years</div>
          </div>
          <div className="flex items-start gap-2">
            <span className="text-green-600">✓</span>
            <div><strong>Dividend Record</strong> - Uninterrupted dividends for 20 years</div>
          </div>
          <div className="flex items-start gap-2">
            <span className="text-green-600">✓</span>
            <div><strong>Earnings Growth</strong> - At least 33% growth over 10 years</div>
          </div>
          <div className="flex items-start gap-2">
            <span className="text-green-600">✓</span>
            <div><strong>Moderate P/E</strong> - P/E ratio below 15 (you're not overpaying)</div>
          </div>
          <div className="flex items-start gap-2">
            <span className="text-green-600">✓</span>
            <div><strong>Moderate P/B</strong> - Price-to-Book below 1.5 (asset backing)</div>
          </div>
          <div className="flex items-start gap-2">
            <span className="text-green-600">✓</span>
            <div><strong>Graham Number</strong> - Stock trades below intrinsic value formula</div>
          </div>
        </div>
      </div>

      <div>
        <h4 className="font-semibold text-gray-900 dark:text-gray-100 mb-2">Mode Options</h4>
        <ul className="space-y-1 text-sm">
          <li><strong>Strict</strong> - Original 1949 criteria. Very few stocks pass. For purists.</li>
          <li><strong>Modern</strong> - Updated thresholds for today's market. Balanced approach.</li>
          <li><strong>GARP</strong> - Growth at Reasonable Price. Allows higher P/E for growing companies.</li>
          <li><strong>Relaxed</strong> - Most lenient. Good for finding candidates to research further.</li>
        </ul>
      </div>
    </InfoPopover>
  )
}

export function NetNetInfo() {
  return (
    <InfoPopover title="Net-Net (NCAV)">
      <p>
        <strong>What this means for the stock:</strong> A "net-net" is trading below its liquidation value. Even if the company shut down today and sold only its current assets (cash, receivables, inventory), shareholders would get more than the current stock price. This is the ultimate margin of safety.
      </p>

      <div className="bg-gray-50 dark:bg-gray-800 rounded-lg p-4">
        <h4 className="font-semibold text-gray-900 dark:text-gray-100 mb-2">The Math</h4>
        <div className="font-mono text-sm bg-white dark:bg-gray-900 p-3 rounded">
          <p>NCAV = Current Assets - Total Liabilities</p>
          <p className="mt-1 text-green-600 dark:text-green-400">If Market Cap &lt; NCAV → Net-Net!</p>
        </div>
      </div>

      <div>
        <h4 className="font-semibold text-gray-900 dark:text-gray-100 mb-2">What This Means</h4>
        <ul className="space-y-2 text-sm">
          <li><strong>Extreme Bargain</strong> - You're buying $1 of assets for less than $1. The business operations come "free."</li>
          <li><strong>Very Rare</strong> - In efficient markets, net-nets are uncommon. Usually found in small caps or distressed situations.</li>
          <li><strong>High Win Rate</strong> - Graham found these had excellent returns historically, even with some failures.</li>
          <li><strong>Requires Patience</strong> - The market may take time to recognize the value. Can be "value traps" if the company keeps burning cash.</li>
        </ul>
      </div>

      <div className="bg-yellow-50 dark:bg-yellow-900/30 rounded-lg p-3 text-sm">
        <strong>Caution:</strong> Net-nets are often cheap for a reason. Always check why the stock is so hated and whether the company is burning through its assets.
      </div>
    </InfoPopover>
  )
}

export function PEGInfo() {
  return (
    <InfoPopover title="PEG Ratio">
      <p>
        <strong>What this means for the stock:</strong> The PEG ratio tells you if you're paying a fair price for growth. A stock with a P/E of 30 might seem expensive, but if it's growing earnings at 30% per year, the PEG is 1.0 - which is fair value for growth investors.
      </p>

      <div className="bg-gray-50 dark:bg-gray-800 rounded-lg p-4">
        <h4 className="font-semibold text-gray-900 dark:text-gray-100 mb-2">The Math</h4>
        <div className="font-mono text-sm bg-white dark:bg-gray-900 p-3 rounded">
          <p>PEG = P/E Ratio ÷ Earnings Growth Rate</p>
          <p className="mt-2 text-gray-500">Example: P/E of 25, Growth of 25% → PEG = 1.0</p>
        </div>
      </div>

      <div className="bg-gray-50 dark:bg-gray-800 rounded-lg p-4 space-y-3">
        <h4 className="font-semibold text-gray-900 dark:text-gray-100">What PEG Tells You</h4>
        <div className="space-y-2">
          <div className="flex items-start gap-3">
            <span className="px-2 py-1 rounded text-xs font-medium bg-green-100 text-green-700 dark:bg-green-900 dark:text-green-300">PEG &lt; 1.0</span>
            <p className="text-sm">Undervalued for growth. You're getting growth at a discount. Peter Lynch's favorite territory.</p>
          </div>
          <div className="flex items-start gap-3">
            <span className="px-2 py-1 rounded text-xs font-medium bg-yellow-100 text-yellow-700 dark:bg-yellow-900 dark:text-yellow-300">PEG 1.0-1.5</span>
            <p className="text-sm">Fairly valued. Paying a reasonable price for expected growth. Not a bargain, but not overpriced.</p>
          </div>
          <div className="flex items-start gap-3">
            <span className="px-2 py-1 rounded text-xs font-medium bg-red-100 text-red-700 dark:bg-red-900 dark:text-red-300">PEG &gt; 2.0</span>
            <p className="text-sm">Expensive for growth. Either the stock is overvalued or growth expectations are too low.</p>
          </div>
        </div>
      </div>

      <div className="text-sm">
        <strong>Max PEG Setting:</strong> Filters for stocks with PEG below your threshold. Lower = more selective, higher = more stocks.
      </div>
    </InfoPopover>
  )
}

export function MagicFormulaInfo() {
  return (
    <InfoPopover title="Magic Formula">
      <p>
        <strong>What this means for the stock:</strong> Joel Greenblatt's Magic Formula finds stocks that are both high-quality businesses AND cheap. A top-ranked stock earns high returns on capital (good business) and has a high earnings yield (cheap price). It's quality + value combined.
      </p>

      <div className="bg-gray-50 dark:bg-gray-800 rounded-lg p-4">
        <h4 className="font-semibold text-gray-900 dark:text-gray-100 mb-3">How It Works</h4>
        <div className="space-y-2 text-sm">
          <div className="flex items-start gap-2">
            <span className="font-bold text-blue-600">1.</span>
            <div><strong>Rank by ROIC</strong> - Return on Invested Capital. Higher = better business.</div>
          </div>
          <div className="flex items-start gap-2">
            <span className="font-bold text-blue-600">2.</span>
            <div><strong>Rank by Earnings Yield</strong> - EBIT/Enterprise Value. Higher = cheaper price.</div>
          </div>
          <div className="flex items-start gap-2">
            <span className="font-bold text-blue-600">3.</span>
            <div><strong>Add the ranks</strong> - Lowest combined rank = best opportunities.</div>
          </div>
        </div>
      </div>

      <div>
        <h4 className="font-semibold text-gray-900 dark:text-gray-100 mb-2">Why It Works</h4>
        <p className="text-sm">High ROIC companies have competitive advantages. High earnings yield means you're buying cheap. The combination systematically finds quality bargains. Greenblatt's backtests showed 30%+ annual returns over 17 years.</p>
      </div>

      <div>
        <h4 className="font-semibold text-gray-900 dark:text-gray-100 mb-2">Top % Setting</h4>
        <ul className="space-y-1 text-sm">
          <li><strong>Top 10%</strong> - Only the best of the best. Very selective, fewer stocks.</li>
          <li><strong>Top 20%</strong> - Good balance. Enough stocks to diversify.</li>
          <li><strong>Top 30-50%</strong> - More stocks to choose from, less selective.</li>
        </ul>
      </div>
    </InfoPopover>
  )
}

export function FamaFrenchBMInfo() {
  return (
    <InfoPopover title="Fama-French Book-to-Market">
      <p>
        <strong>What this means for the stock:</strong> Nobel Prize-winning research shows that "value" stocks (high book-to-market) outperform "growth" stocks (low book-to-market) over time. A high B/M percentile means this stock is in value territory - historically a good sign for long-term returns.
      </p>

      <div className="bg-gray-50 dark:bg-gray-800 rounded-lg p-4">
        <h4 className="font-semibold text-gray-900 dark:text-gray-100 mb-2">The Math</h4>
        <div className="font-mono text-sm bg-white dark:bg-gray-900 p-3 rounded">
          <p>Book-to-Market = Book Value ÷ Market Cap</p>
          <p className="mt-1 text-gray-500">High B/M = Value stock (underpriced)</p>
          <p className="text-gray-500">Low B/M = Growth stock (premium priced)</p>
        </div>
      </div>

      <div>
        <h4 className="font-semibold text-gray-900 dark:text-gray-100 mb-2">The Value Premium</h4>
        <p className="text-sm">Fama and French found that value stocks (top 30% by B/M) outperform growth stocks by 3-5% annually over long periods. This "value premium" is one of the most documented patterns in finance. The theory: value stocks are riskier or overlooked, so they offer higher returns.</p>
      </div>

      <div>
        <h4 className="font-semibold text-gray-900 dark:text-gray-100 mb-2">Top % Setting</h4>
        <ul className="space-y-1 text-sm">
          <li><strong>Top 20%</strong> - Deep value only. The most underpriced by this metric.</li>
          <li><strong>Top 30%</strong> - Standard academic definition of "value stocks."</li>
          <li><strong>Top 40-50%</strong> - Moderate value tilt. More diversified selection.</li>
        </ul>
      </div>
    </InfoPopover>
  )
}

export function QualityClassificationInfo() {
  return (
    <InfoPopover title="Quality Classification">
      <p>
        <strong>What this means for the stock:</strong> This classification tells you whether the company creates or destroys shareholder value based on its Return on Invested Capital (ROIC). High ROIC companies turn every dollar invested into more than a dollar of profit - that's wealth compounding.
      </p>

      <div className="space-y-4">
        <div className="bg-emerald-50 dark:bg-emerald-900/30 rounded-lg p-4">
          <h4 className="font-semibold text-emerald-700 dark:text-emerald-300 mb-2">Compounder (ROIC ≥ 15% + 5yr FCF)</h4>
          <p className="text-sm">This business consistently turns $1 of investment into $1.15+ of profit, year after year. These are rare, high-quality businesses with likely competitive advantages - think Apple, Costco, or Visa. They compound wealth over time. Great for long-term holding. You can pay a fair price and still do well.</p>
        </div>

        <div className="bg-yellow-50 dark:bg-yellow-900/30 rounded-lg p-4">
          <h4 className="font-semibold text-yellow-700 dark:text-yellow-300 mb-2">Average (ROIC 8-15%)</h4>
          <p className="text-sm">Decent business that covers its cost of capital but isn't exceptional. These companies earn their keep but don't have obvious moats. Could be a turnaround candidate, a cyclical at the wrong point, or just a boring but stable company. Requires deeper analysis - needs to be cheaper than a Compounder to be attractive.</p>
        </div>

        <div className="bg-gray-100 dark:bg-gray-800 rounded-lg p-4">
          <h4 className="font-semibold text-gray-700 dark:text-gray-300 mb-2">Weak (ROIC &lt; 8%)</h4>
          <p className="text-sm">This company destroys shareholder value over time - it earns less than investors could get elsewhere. Shareholders would be better off if management returned capital. Only consider if you believe in a major turnaround or if it's trading at deep net-net value (you're buying the assets, not the business).</p>
        </div>
      </div>
    </InfoPopover>
  )
}

export function QualityTagsInfo() {
  return (
    <InfoPopover title="Quality Tags">
      <p>
        <strong>What these tags mean:</strong> Quick signals about business quality, valuation, and potential risks. Multiple tags paint a fuller picture of the stock.
      </p>

      <div className="space-y-4">
        <div className="border-l-4 border-emerald-500 pl-4">
          <h4 className="font-semibold text-emerald-600 dark:text-emerald-400">Durable Compounder</h4>
          <p className="text-sm mt-1">Rock-solid business with predictable profits. Both ROIC and gross margins have been stable for 5+ years - this company has pricing power and consistent demand. Safe to hold through market downturns. Think Coca-Cola or Johnson & Johnson. These rarely go on sale, so even a fair price can work.</p>
        </div>

        <div className="border-l-4 border-blue-500 pl-4">
          <h4 className="font-semibold text-blue-600 dark:text-blue-400">Cash Machine</h4>
          <p className="text-sm mt-1">Generates more actual cash than accounting profits suggest. Free Cash Flow exceeds Net Income, and you're getting a high FCF yield on your investment. The earnings are real, not accounting tricks. Very attractive - this is the kind of stock that can fund dividends, buybacks, or acquisitions.</p>
        </div>

        <div className="border-l-4 border-purple-500 pl-4">
          <h4 className="font-semibold text-purple-600 dark:text-purple-400">Deep Value</h4>
          <p className="text-sm mt-1">Stock is priced cheaply relative to cash generation (high FCF yield or low EV/EBIT). The market may be overly pessimistic or simply overlooking this company. Potential for significant upside if sentiment improves or fundamentals are recognized. Classic value investing territory.</p>
        </div>

        <div className="border-l-4 border-amber-500 pl-4">
          <h4 className="font-semibold text-amber-600 dark:text-amber-400">Heavy Reinvestor</h4>
          <p className="text-sm mt-1">Company plows most of its cash back into the business (R&D, CapEx, expansion). This is great for growth companies building the future - but concerning if the growth isn't materializing. Check revenue trends. Amazon was a Heavy Reinvestor for years before it paid off.</p>
        </div>

        <div className="border-l-4 border-gray-400 pl-4">
          <h4 className="font-semibold text-gray-600 dark:text-gray-400">Premium Priced</h4>
          <p className="text-sm mt-1">You're paying up for this stock - low FCF yield and/or high EV/EBIT. Future returns may be muted unless growth exceeds already-high expectations. Only worth it for truly exceptional businesses. Make sure the quality justifies the premium, or you may be buying at the top.</p>
        </div>

        <div className="border-l-4 border-red-500 pl-4">
          <h4 className="font-semibold text-red-600 dark:text-red-400">Volatile Returns</h4>
          <p className="text-sm mt-1">Profits swing wildly from year to year (high ROIC standard deviation). Could be a cyclical business (steel, oil, airlines) where this is normal, or could indicate poor management and unpredictable results. Harder to value accurately and riskier to hold. Requires understanding the cycle.</p>
        </div>

        <div className="border-l-4 border-yellow-500 pl-4">
          <h4 className="font-semibold text-yellow-600 dark:text-yellow-400">Weak Moat Signal</h4>
          <p className="text-sm mt-1">Gross margins fluctuate significantly, suggesting the company lacks pricing power. Competition may be eroding margins, or the business is in a commodity-like industry. Be cautious about paying premium prices - this company may struggle to maintain profitability.</p>
        </div>

        <div className="border-l-4 border-orange-500 pl-4">
          <h4 className="font-semibold text-orange-600 dark:text-orange-400">Earnings Quality Concern</h4>
          <p className="text-sm mt-1">Reported profits don't convert to actual cash (FCF much lower than Net Income). Could be aggressive revenue recognition, large receivables that may not collect, one-time gains, or heavy working capital needs. Dig deeper before trusting the earnings numbers. The cash flow statement tells the real story.</p>
        </div>
      </div>
    </InfoPopover>
  )
}
