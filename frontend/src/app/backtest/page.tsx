"use client"

import { useSearchParams } from "next/navigation"
import Link from "next/link"
import { useEffect, useState, Suspense } from "react"
import {
  RiArrowLeftLine,
  RiArrowUpLine,
  RiArrowDownLine,
  RiLineChartLine,
  RiExchangeDollarLine,
  RiPercentLine,
} from "@remixicon/react"
import { simulateBuy, BacktestResult, StockReturn } from "@/lib/api"
import { Button } from "@/components/Button"
import { Input } from "@/components/Input"

function formatQuarter(q: string): string {
  // Convert "2024Q1" to "Q1 2024"
  if (q.length === 6 && q.includes("Q")) {
    const year = q.slice(0, 4)
    const qNum = q.slice(4)
    return `${qNum} ${year}`
  }
  return q
}

function formatReturnPct(value: number): string {
  const sign = value >= 0 ? "+" : ""
  return `${sign}${value.toFixed(2)}%`
}

function formatPrice(value: number): string {
  return `$${value.toFixed(2)}`
}

function ReturnBadge({ value }: { value: number }) {
  const isPositive = value >= 0
  return (
    <span
      className={`inline-flex items-center px-2 py-0.5 rounded text-sm font-semibold ${
        isPositive
          ? "bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200"
          : "bg-red-100 text-red-800 dark:bg-red-900 dark:text-red-200"
      }`}
    >
      {isPositive ? <RiArrowUpLine className="size-4 mr-1" /> : <RiArrowDownLine className="size-4 mr-1" />}
      {formatReturnPct(value)}
    </span>
  )
}

function SummaryCard({
  title,
  value,
  subtitle,
  icon,
  variant = "default",
}: {
  title: string
  value: string
  subtitle?: string
  icon: React.ReactNode
  variant?: "default" | "positive" | "negative"
}) {
  const bgColor =
    variant === "positive"
      ? "bg-green-50 dark:bg-green-950 border-green-200 dark:border-green-800"
      : variant === "negative"
      ? "bg-red-50 dark:bg-red-950 border-red-200 dark:border-red-800"
      : "bg-white dark:bg-gray-950 border-gray-200 dark:border-gray-800"

  const textColor =
    variant === "positive"
      ? "text-green-700 dark:text-green-300"
      : variant === "negative"
      ? "text-red-700 dark:text-red-300"
      : "text-gray-900 dark:text-gray-100"

  return (
    <div className={`rounded-lg border p-4 ${bgColor}`}>
      <div className="flex items-center gap-2 text-gray-500 dark:text-gray-400 mb-1">
        {icon}
        <span className="text-sm">{title}</span>
      </div>
      <div className={`text-2xl font-bold ${textColor}`}>{value}</div>
      {subtitle && <div className="text-xs text-gray-500 mt-1">{subtitle}</div>}
    </div>
  )
}

function StockTable({ stocks, title }: { stocks: StockReturn[]; title: string }) {
  if (stocks.length === 0) return null

  return (
    <div className="rounded-lg border border-gray-200 dark:border-gray-800 overflow-hidden">
      <div className="bg-gray-50 dark:bg-gray-900 px-4 py-3 border-b border-gray-200 dark:border-gray-800">
        <h3 className="font-semibold text-gray-900 dark:text-gray-100">{title}</h3>
      </div>
      <table className="min-w-full divide-y divide-gray-200 dark:divide-gray-800">
        <thead className="bg-gray-50 dark:bg-gray-900">
          <tr>
            <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">Symbol</th>
            <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">Name</th>
            <th className="px-4 py-3 text-right text-xs font-medium text-gray-500 uppercase">Buy Price</th>
            <th className="px-4 py-3 text-right text-xs font-medium text-gray-500 uppercase">Current</th>
            <th className="px-4 py-3 text-right text-xs font-medium text-gray-500 uppercase">Return</th>
          </tr>
        </thead>
        <tbody className="bg-white dark:bg-gray-950 divide-y divide-gray-200 dark:divide-gray-800">
          {stocks.map((stock) => (
            <tr key={stock.symbol} className="hover:bg-gray-50 dark:hover:bg-gray-900">
              <td className="px-4 py-3">
                <Link
                  href={`/stock/${stock.symbol}`}
                  className="text-sm font-medium text-indigo-600 hover:text-indigo-500 dark:text-indigo-400"
                >
                  {stock.symbol}
                </Link>
              </td>
              <td className="px-4 py-3 text-sm text-gray-900 dark:text-gray-100 max-w-[200px] truncate">
                {stock.name}
              </td>
              <td className="px-4 py-3 text-sm text-gray-600 dark:text-gray-400 text-right">
                {formatPrice(stock.buy_price)}
              </td>
              <td className="px-4 py-3 text-sm text-gray-900 dark:text-gray-100 text-right">
                {formatPrice(stock.current_price)}
              </td>
              <td className="px-4 py-3 text-right">
                <ReturnBadge value={stock.total_return} />
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

function QuarterProgressionChart({
  portfolioReturns,
  benchmarkReturns,
}: {
  portfolioReturns: { quarter: string; return_pct: number }[]
  benchmarkReturns: { quarter: string; return_pct: number }[]
}) {
  if (portfolioReturns.length === 0) return null

  const maxReturn = Math.max(
    ...portfolioReturns.map((r) => Math.abs(r.return_pct)),
    ...benchmarkReturns.map((r) => Math.abs(r.return_pct)),
    10
  )

  return (
    <div className="rounded-lg border border-gray-200 dark:border-gray-800 p-4">
      <h3 className="font-semibold text-gray-900 dark:text-gray-100 mb-4">Quarter-by-Quarter Returns</h3>
      <div className="space-y-3">
        {portfolioReturns.map((pr, idx) => {
          const br = benchmarkReturns[idx]
          const portfolioWidth = (Math.abs(pr.return_pct) / maxReturn) * 100
          const benchmarkWidth = br ? (Math.abs(br.return_pct) / maxReturn) * 100 : 0

          return (
            <div key={pr.quarter} className="space-y-1">
              <div className="flex items-center justify-between text-sm">
                <span className="text-gray-600 dark:text-gray-400">{formatQuarter(pr.quarter)}</span>
                <div className="flex items-center gap-4">
                  <span className="text-gray-900 dark:text-gray-100 font-medium">
                    Portfolio: {formatReturnPct(pr.return_pct)}
                  </span>
                  {br && (
                    <span className="text-gray-500">S&P 500: {formatReturnPct(br.return_pct)}</span>
                  )}
                </div>
              </div>
              <div className="flex gap-2">
                <div className="flex-1 h-3 bg-gray-100 dark:bg-gray-800 rounded overflow-hidden">
                  <div
                    className={`h-full rounded ${
                      pr.return_pct >= 0 ? "bg-green-500" : "bg-red-500"
                    }`}
                    style={{ width: `${portfolioWidth}%` }}
                  />
                </div>
                <div className="flex-1 h-3 bg-gray-100 dark:bg-gray-800 rounded overflow-hidden">
                  <div
                    className={`h-full rounded ${
                      br && br.return_pct >= 0 ? "bg-blue-400" : "bg-orange-400"
                    }`}
                    style={{ width: `${benchmarkWidth}%` }}
                  />
                </div>
              </div>
            </div>
          )
        })}
      </div>
      <div className="flex items-center gap-4 mt-4 text-xs text-gray-500">
        <div className="flex items-center gap-1">
          <div className="w-3 h-3 rounded bg-green-500" />
          <span>Portfolio (positive)</span>
        </div>
        <div className="flex items-center gap-1">
          <div className="w-3 h-3 rounded bg-red-500" />
          <span>Portfolio (negative)</span>
        </div>
        <div className="flex items-center gap-1">
          <div className="w-3 h-3 rounded bg-blue-400" />
          <span>S&P 500</span>
        </div>
      </div>
    </div>
  )
}

function BacktestContent() {
  const searchParams = useSearchParams()
  const [result, setResult] = useState<BacktestResult | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [manualBenchmark, setManualBenchmark] = useState("")

  const symbols = searchParams.get("symbols")?.split(",") || []
  const quarter = searchParams.get("quarter") || ""

  const runBacktest = async (benchmark?: number) => {
    if (!symbols.length || !quarter) {
      setError("Missing symbols or quarter")
      setLoading(false)
      return
    }

    try {
      setLoading(true)
      setError(null)
      const data = await simulateBuy({
        symbols,
        buy_quarter: quarter,
        benchmark_return: benchmark,
      })
      setResult(data)
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to run backtest")
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    runBacktest()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [searchParams])

  const handleManualBenchmark = () => {
    const value = parseFloat(manualBenchmark)
    if (!isNaN(value)) {
      runBacktest(value)
    }
  }

  if (loading) {
    return (
      <div className="flex items-center justify-center py-20">
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-indigo-600" />
      </div>
    )
  }

  if (error) {
    return (
      <div className="p-6">
        <div className="rounded-lg border border-red-200 bg-red-50 p-4 dark:border-red-800 dark:bg-red-950">
          <p className="text-sm text-red-600 dark:text-red-400">{error}</p>
        </div>
        <Link href="/" className="mt-4 inline-flex items-center text-indigo-600 hover:text-indigo-500">
          <RiArrowLeftLine className="size-4 mr-1" />
          Back to Pipeline
        </Link>
      </div>
    )
  }

  if (!result) {
    return null
  }

  const alphaVariant = result.alpha >= 0 ? "positive" : "negative"

  return (
    <div className="p-4 sm:p-6 lg:p-8 max-w-6xl mx-auto">
      {/* Header */}
      <div className="mb-6">
        <Link
          href="/"
          className="inline-flex items-center text-sm text-gray-500 hover:text-gray-700 dark:hover:text-gray-300 mb-2"
        >
          <RiArrowLeftLine className="size-4 mr-1" />
          Back to Pipeline
        </Link>
        <h1 className="text-2xl font-bold text-gray-900 dark:text-gray-50">Backtest Results</h1>
        <p className="mt-1 text-sm text-gray-500">
          {formatQuarter(result.buy_quarter)} to {formatQuarter(result.latest_quarter)} ({result.quarters_held}{" "}
          quarter{result.quarters_held !== 1 ? "s" : ""})
        </p>
      </div>

      {/* Summary Cards */}
      <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 mb-6">
        <SummaryCard
          title="Portfolio Return"
          value={formatReturnPct(result.portfolio_return)}
          subtitle={`${result.stocks.length} stocks, equal-weighted`}
          icon={<RiLineChartLine className="size-5" />}
          variant={result.portfolio_return >= 0 ? "positive" : "negative"}
        />
        <SummaryCard
          title="S&P 500 Return"
          value={formatReturnPct(result.benchmark_return)}
          subtitle="Benchmark comparison"
          icon={<RiExchangeDollarLine className="size-5" />}
        />
        <SummaryCard
          title="Alpha"
          value={formatReturnPct(result.alpha)}
          subtitle="Excess return vs. market"
          icon={<RiPercentLine className="size-5" />}
          variant={alphaVariant}
        />
      </div>

      {/* Manual Benchmark Override */}
      <div className="mb-6 p-4 rounded-lg bg-gray-50 dark:bg-gray-900 border border-gray-200 dark:border-gray-800">
        <div className="flex items-center gap-4">
          <span className="text-sm text-gray-600 dark:text-gray-400">Override S&P 500 return:</span>
          <Input
            type="number"
            step="0.1"
            placeholder="e.g., 12.5"
            value={manualBenchmark}
            onChange={(e) => setManualBenchmark(e.target.value)}
            className="w-32"
          />
          <span className="text-sm text-gray-500">%</span>
          <Button variant="secondary" onClick={handleManualBenchmark} disabled={!manualBenchmark}>
            Recalculate
          </Button>
        </div>
      </div>

      {/* Quarter Progression */}
      <div className="mb-6">
        <QuarterProgressionChart
          portfolioReturns={result.quarterly_portfolio_returns}
          benchmarkReturns={result.quarterly_benchmark_returns}
        />
      </div>

      {/* Winners & Losers */}
      <div className="space-y-6">
        <StockTable
          stocks={result.winners}
          title={`Winners (${result.winners.length})`}
        />
        <StockTable
          stocks={result.losers}
          title={`Losers (${result.losers.length})`}
        />
      </div>
    </div>
  )
}

export default function BacktestPage() {
  return (
    <Suspense
      fallback={
        <div className="flex items-center justify-center py-20">
          <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-indigo-600" />
        </div>
      }
    >
      <BacktestContent />
    </Suspense>
  )
}
