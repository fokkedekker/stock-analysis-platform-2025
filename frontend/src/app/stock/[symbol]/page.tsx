"use client"

import { useEffect, useState, useMemo } from "react"
import { useParams } from "next/navigation"
import Link from "next/link"
import {
  getStockAnalysis,
  getStockProfile,
  getIncomeStatements,
  getBalanceSheets,
  getCashFlows,
  getDividends,
  getHistoricalPrices,
  getStrategySignals,
  getAvailableMLModels,
  getMLModelSignals,
  formatNumber,
  formatCurrency,
  StockAnalysis,
  StockProfile,
  HistoricalPrice,
  StrategySignalsResponse,
  MLModelSummary,
  MLModelSignalsResponse,
} from "@/lib/api"
import { loadStrategies, SavedStrategy } from "@/lib/saved-strategies"
import StockPriceChart from "@/components/StockPriceChart"
import {
  getAltmanNarrative,
  getPiotroskiNarrative,
  getSurvivalSummary,
  getROICNarrative,
  getFCFNarrative,
  getQualityTagNarrative,
  getQualitySummary,
  getGrahamNarrative,
  getNetNetNarrative,
  getPEGNarrative,
  getMagicFormulaNarrative,
  getFamaFrenchNarrative,
  getValuationSummary,
  getSizeNarrative,
  getProfitabilityNarrative,
  getFactorSummary,
} from "@/lib/stockNarratives"
import {
  RiArrowLeftLine,
  RiShieldCheckLine,
  RiSparklingLine,
} from "@remixicon/react"
import { StockExplainChat } from "@/components/StockExplainChat"
import { Button } from "@/components/Button"

function StageCard({
  title,
  icon,
  children,
  status,
}: {
  title: string
  icon?: React.ReactNode
  children: React.ReactNode
  status?: "pass" | "fail" | "neutral"
}) {
  const borderColor =
    status === "pass"
      ? "border-green-200 dark:border-green-900"
      : status === "fail"
      ? "border-red-200 dark:border-red-900"
      : "border-gray-200 dark:border-gray-800"
  const bgColor =
    status === "pass"
      ? "bg-green-50 dark:bg-green-950/30"
      : status === "fail"
      ? "bg-red-50 dark:bg-red-950/30"
      : "bg-white dark:bg-gray-950"

  return (
    <div className={`p-4 rounded-lg border-2 ${borderColor} ${bgColor}`}>
      <div className="flex items-center gap-2 mb-3">
        {icon}
        <h3 className="text-sm font-semibold text-gray-900 dark:text-gray-100 uppercase tracking-wide">
          {title}
        </h3>
      </div>
      {children}
    </div>
  )
}

export default function StockDetailPage() {
  const params = useParams()
  const symbol = params.symbol as string

  const [analysis, setAnalysis] = useState<StockAnalysis | null>(null)
  const [profile, setProfile] = useState<StockProfile | null>(null)
  const [incomeStatements, setIncomeStatements] = useState<any[]>([])
  const [balanceSheets, setBalanceSheets] = useState<any[]>([])
  const [cashFlows, setCashFlows] = useState<any[]>([])
  const [dividends, setDividends] = useState<any[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [activeTab, setActiveTab] = useState<"income" | "balance" | "cashflow" | "dividends">("income")
  const [showExplain, setShowExplain] = useState(false)

  // Strategy chart state
  const [strategies, setStrategies] = useState<SavedStrategy[]>([])
  const [selectedStrategyId, setSelectedStrategyId] = useState<string>("")
  const [historicalPrices, setHistoricalPrices] = useState<HistoricalPrice[]>([])
  const [strategySignals, setStrategySignals] = useState<StrategySignalsResponse | null>(null)
  const [chartLoading, setChartLoading] = useState(false)

  // ML Model chart state
  const [mlModels, setMlModels] = useState<MLModelSummary[]>([])
  const [selectedMLModelId, setSelectedMLModelId] = useState<string>("")
  const [mlModelSignals, setMlModelSignals] = useState<MLModelSignalsResponse | null>(null)
  const [mlModelLoading, setMlModelLoading] = useState(false)

  useEffect(() => {
    async function fetchData() {
      try {
        setLoading(true)
        const [analysisData, profileData, incomeData, balanceData, cashData, divData] = await Promise.all([
          getStockAnalysis(symbol).catch(() => null),
          getStockProfile(symbol).catch(() => null),
          getIncomeStatements(symbol, "annual", 5).catch(() => []),
          getBalanceSheets(symbol, "annual", 5).catch(() => []),
          getCashFlows(symbol, "annual", 5).catch(() => []),
          getDividends(symbol, 20).catch(() => []),
        ])
        setAnalysis(analysisData)
        setProfile(profileData)
        setIncomeStatements(incomeData?.statements || incomeData || [])
        setBalanceSheets(balanceData?.statements || balanceData || [])
        setCashFlows(cashData?.statements || cashData || [])
        setDividends(divData?.dividends || divData || [])
        setError(null)
      } catch (err) {
        setError("Failed to fetch stock data. Is the API server running?")
        console.error(err)
      } finally {
        setLoading(false)
      }
    }
    fetchData()
  }, [symbol])

  // Load saved strategies, ML models, and historical prices
  useEffect(() => {
    async function loadChartData() {
      try {
        // Load strategies and ML models in parallel
        const [savedStrategies, mlModelsResult] = await Promise.all([
          loadStrategies(),
          getAvailableMLModels().catch(() => ({ models: [], total: 0 })),
        ])
        setStrategies(savedStrategies)
        setMlModels(mlModelsResult.models)

        // Load historical prices (5 years)
        const fiveYearsAgo = new Date()
        fiveYearsAgo.setFullYear(fiveYearsAgo.getFullYear() - 5)
        const fromDate = fiveYearsAgo.toISOString().split("T")[0]

        const priceData = await getHistoricalPrices(symbol, fromDate).catch(() => null)
        if (priceData) {
          setHistoricalPrices(priceData.prices)
        }
      } catch (err) {
        console.error("Failed to load chart data:", err)
      }
    }
    loadChartData()
  }, [symbol])

  // Load strategy signals when a strategy is selected
  useEffect(() => {
    async function loadSignals() {
      if (!selectedStrategyId) {
        setStrategySignals(null)
        return
      }

      setChartLoading(true)
      try {
        const signals = await getStrategySignals(selectedStrategyId, symbol)
        setStrategySignals(signals)
      } catch (err) {
        console.error("Failed to load strategy signals:", err)
        setStrategySignals(null)
      } finally {
        setChartLoading(false)
      }
    }
    loadSignals()
  }, [selectedStrategyId, symbol])

  // Load ML model signals when a model is selected
  useEffect(() => {
    async function loadMLSignals() {
      if (!selectedMLModelId) {
        setMlModelSignals(null)
        return
      }

      // Find the selected model to get its type
      const selectedModel = mlModels.find(m => m.run_id === selectedMLModelId)
      const modelType = selectedModel?.model_type || 'elastic_net'

      setMlModelLoading(true)
      try {
        const signals = await getMLModelSignals(selectedMLModelId, symbol, 20, modelType)
        setMlModelSignals(signals)
      } catch (err) {
        console.error("Failed to load ML model signals:", err)
        setMlModelSignals(null)
      } finally {
        setMlModelLoading(false)
      }
    }
    loadMLSignals()
  }, [selectedMLModelId, symbol, mlModels])

  // Calculate visible signal stats (matching what's shown on chart)
  const visibleMLStats = useMemo(() => {
    if (!mlModelSignals || !historicalPrices.length) {
      return null
    }

    // Get price date range
    const dates = historicalPrices.map(p => new Date(p.date).getTime())
    const minDate = Math.min(...dates)
    const maxDate = Math.max(...dates)

    // Helper to check if date is in range
    const isInRange = (dateStr: string) => {
      const d = new Date(dateStr).getTime()
      return d >= minDate && d <= maxDate
    }

    // Count signals where buy_date is in range (these have visible buy markers)
    const visibleBuySignals = mlModelSignals.signals.filter(
      s => s.matched && isInRange(s.buy_date)
    )
    const numVisibleBuys = visibleBuySignals.length

    // For stats, we need complete trades (both buy and sell prices available)
    const completeTrades = visibleBuySignals.filter(
      s => s.stock_return !== null && isInRange(s.sell_date)
    )

    if (completeTrades.length === 0) {
      return {
        numTrades: numVisibleBuys,
        totalNumTrades: mlModelSignals.num_trades,
        totalReturn: null,
        totalAlpha: null,
        avgAlpha: null,
        winRate: null,
      }
    }

    const validTrades = completeTrades

    // Calculate compound stock return
    let stockCompound = 1.0
    let spyCompound = 1.0
    for (const s of validTrades) {
      stockCompound *= (1 + (s.stock_return || 0) / 100)
      if (s.spy_return !== null) {
        spyCompound *= (1 + s.spy_return / 100)
      }
    }

    const totalReturn = (stockCompound - 1) * 100
    const totalSpyReturn = (spyCompound - 1) * 100
    const totalAlpha = totalReturn - totalSpyReturn

    // Avg alpha and win rate
    const alphas = validTrades
      .map(s => s.alpha)
      .filter((a): a is number => a !== null)
    const avgAlpha = alphas.length > 0 ? alphas.reduce((a, b) => a + b, 0) / alphas.length : null
    const winRate = alphas.length > 0
      ? (alphas.filter(a => a > 0).length / alphas.length) * 100
      : null

    return {
      numTrades: numVisibleBuys,
      totalNumTrades: mlModelSignals.num_trades,
      totalReturn: Math.round(totalReturn * 10) / 10,
      totalAlpha: Math.round(totalAlpha * 10) / 10,
      avgAlpha: avgAlpha !== null ? Math.round(avgAlpha * 100) / 100 : null,
      winRate: winRate !== null ? Math.round(winRate) : null,
    }
  }, [mlModelSignals, historicalPrices])

  if (loading) {
    return (
      <div className="flex items-center justify-center py-24">
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-indigo-600"></div>
      </div>
    )
  }

  if (error) {
    return (
      <div className="p-4 sm:p-6 lg:p-8">
        <div className="rounded-lg border border-red-200 bg-red-50 p-4 dark:border-red-900 dark:bg-red-950">
          <p className="text-sm text-red-600 dark:text-red-400">{error}</p>
        </div>
      </div>
    )
  }

  const graham = analysis?.graham
  const magicFormula = analysis?.magic_formula
  const piotroski = analysis?.piotroski
  const altman = analysis?.altman
  const roic = analysis?.roic_quality
  const peg = analysis?.garp_peg
  const famaFrench = analysis?.fama_french
  const netNet = analysis?.net_net

  // Determine survival status
  const altmanPass = altman?.zone === "safe"
  const piotroskiPass = piotroski?.f_score >= 5
  const survivalPass = altmanPass && piotroskiPass

  // Determine quality label
  const roicValue = roic?.roic
  const qualityLabel =
    roicValue && roicValue >= 0.15 && roic?.fcf_positive_5yr
      ? "Compounder"
      : roicValue && roicValue >= 0.08
      ? "Average"
      : "Weak"

  // Count valuation lenses passed
  const grahamPass = graham?.criteria_passed >= 5
  const pegPass = peg?.peg_pass
  const mfPass = magicFormula?.combined_rank <= Math.max(100, (magicFormula?.combined_rank || 0) * 0.2)
  const netNetPass = netNet?.trading_below_ncav
  const ffBmPass = famaFrench?.book_to_market_percentile >= 0.7

  const lensesPassed = [grahamPass, pegPass, mfPass, netNetPass, ffBmPass].filter(Boolean)

  return (
    <div className="p-4 sm:p-6 lg:p-8">
      {/* Header */}
      <div className="mb-6">
        <Link
          href="/"
          className="inline-flex items-center gap-1 text-sm text-gray-500 hover:text-gray-700 mb-4"
        >
          <RiArrowLeftLine className="w-4 h-4" />
          Back to Pipeline
        </Link>
        <div className="flex items-start justify-between">
          <div>
            <h1 className="text-3xl font-bold text-gray-900 dark:text-gray-50">{symbol}</h1>
            {profile && (
              <p className="mt-1 text-gray-500">
                Price: {formatCurrency(profile.price)} | Market Cap: {formatCurrency(profile.market_cap)}
              </p>
            )}
          </div>
          <div className="flex items-start gap-4">
            <Button onClick={() => setShowExplain(true)}>
              <RiSparklingLine className="size-4 shrink-0 mr-1.5" />
              Explain This Stock
            </Button>
            {profile && (
              <div className="text-right text-sm text-gray-500">
                <div>P/E: {formatNumber(profile.pe_ratio, 1)}</div>
                <div>P/B: {formatNumber(profile.pb_ratio, 2)}</div>
                <div>Beta: {formatNumber(profile.beta, 2)}</div>
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Stock Price Chart with Strategy Signals */}
      {historicalPrices.length > 0 && (
        <div className="mb-8">
          <div className="flex items-center justify-between mb-4 flex-wrap gap-4">
            <h2 className="text-lg font-semibold text-gray-900 dark:text-gray-50">
              Price History
            </h2>
            <div className="flex items-center gap-6 flex-wrap">
              {/* Strategy Selector */}
              <div className="flex items-center gap-2">
                <label className="text-sm text-gray-600 dark:text-gray-400">
                  Strategy:
                </label>
                <select
                  value={selectedStrategyId}
                  onChange={(e) => setSelectedStrategyId(e.target.value)}
                  className="px-3 py-1.5 text-sm border border-gray-300 dark:border-gray-600 rounded-md bg-white dark:bg-gray-800 text-gray-900 dark:text-gray-100"
                >
                  <option value="">None</option>
                  {strategies.map((s) => (
                    <option key={s.id} value={s.id}>
                      {s.name} ({s.holding_period || 1}Q hold)
                    </option>
                  ))}
                </select>
              </div>
              {/* ML Model Selector */}
              <div className="flex items-center gap-2">
                <label className="text-sm text-gray-600 dark:text-gray-400">
                  ML Model:
                </label>
                <select
                  value={selectedMLModelId}
                  onChange={(e) => setSelectedMLModelId(e.target.value)}
                  className="px-3 py-1.5 text-sm border border-gray-300 dark:border-gray-600 rounded-md bg-white dark:bg-gray-800 text-gray-900 dark:text-gray-100"
                >
                  <option value="">None</option>
                  {mlModels.map((m) => (
                    <option key={m.run_id} value={m.run_id}>
                      {m.model_type === 'lightgbm' ? 'LightGBM' : m.model_type === 'gam' ? 'GAM' : 'Elastic Net'} {m.created_at ? new Date(m.created_at).toLocaleDateString() : ""} (IC: {m.test_ic?.toFixed(3) || "N/A"})
                    </option>
                  ))}
                </select>
              </div>
            </div>
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
            {/* Chart */}
            <div className="lg:col-span-3 bg-white dark:bg-gray-900 rounded-lg border border-gray-200 dark:border-gray-800 p-4">
              <StockPriceChart
                prices={historicalPrices}
                signals={[
                  ...(strategySignals?.signals || []),
                  ...(mlModelSignals?.signals || []),
                ]}
                loading={chartLoading || mlModelLoading}
              />
            </div>

            {/* Performance Summary */}
            <div className="space-y-4">
              {/* Strategy Performance Summary */}
              {strategySignals && (
                <div className="bg-white dark:bg-gray-900 rounded-lg border border-gray-200 dark:border-gray-800 p-4">
                  <h3 className="text-sm font-semibold text-gray-900 dark:text-gray-100 mb-4">
                    Strategy Performance
                  </h3>
                  <div className="space-y-3">
                    <div>
                      <p className="text-xs text-gray-500 dark:text-gray-400">Strategy</p>
                      <p className="text-sm font-medium text-gray-900 dark:text-gray-100">
                        {strategySignals.strategy_name}
                      </p>
                    </div>
                    <div>
                      <p className="text-xs text-gray-500 dark:text-gray-400">Holding Period</p>
                      <p className="text-sm font-medium text-gray-900 dark:text-gray-100">
                        {strategySignals.holding_period} quarter{strategySignals.holding_period !== 1 ? "s" : ""}
                      </p>
                    </div>
                    <div className="pt-2 border-t border-gray-200 dark:border-gray-700">
                      <p className="text-xs text-gray-500 dark:text-gray-400">Number of Trades</p>
                      <p className="text-lg font-bold text-gray-900 dark:text-gray-100">
                        {strategySignals.num_trades}
                      </p>
                    </div>
                    {strategySignals.total_return !== null && (
                      <div>
                        <p className="text-xs text-gray-500 dark:text-gray-400">Total Return</p>
                        <p className={`text-lg font-bold ${
                          strategySignals.total_return >= 0
                            ? "text-green-600 dark:text-green-400"
                            : "text-red-600 dark:text-red-400"
                        }`}>
                          {strategySignals.total_return >= 0 ? "+" : ""}{strategySignals.total_return.toFixed(1)}%
                        </p>
                      </div>
                    )}
                    {strategySignals.total_alpha !== null && (
                      <div>
                        <p className="text-xs text-gray-500 dark:text-gray-400">Total Alpha vs S&P 500</p>
                        <p className={`text-lg font-bold ${
                          strategySignals.total_alpha >= 0
                            ? "text-green-600 dark:text-green-400"
                            : "text-red-600 dark:text-red-400"
                        }`}>
                          {strategySignals.total_alpha >= 0 ? "+" : ""}{strategySignals.total_alpha.toFixed(1)}%
                        </p>
                      </div>
                    )}
                    {strategySignals.avg_alpha_per_trade !== null && (
                      <div>
                        <p className="text-xs text-gray-500 dark:text-gray-400">Avg Alpha per Trade</p>
                        <p className={`text-sm font-medium ${
                          strategySignals.avg_alpha_per_trade >= 0
                            ? "text-green-600 dark:text-green-400"
                            : "text-red-600 dark:text-red-400"
                        }`}>
                          {strategySignals.avg_alpha_per_trade >= 0 ? "+" : ""}{strategySignals.avg_alpha_per_trade.toFixed(2)}%
                        </p>
                      </div>
                    )}
                    {strategySignals.win_rate !== null && (
                      <div>
                        <p className="text-xs text-gray-500 dark:text-gray-400">Win Rate</p>
                        <p className="text-sm font-medium text-gray-900 dark:text-gray-100">
                          {strategySignals.win_rate.toFixed(0)}%
                        </p>
                      </div>
                    )}
                  </div>
                </div>
              )}

              {/* ML Model Performance Summary - uses visible stats matching chart */}
              {mlModelSignals && visibleMLStats && (
                <div className="bg-white dark:bg-gray-900 rounded-lg border border-indigo-200 dark:border-indigo-800 p-4">
                  <h3 className="text-sm font-semibold text-indigo-600 dark:text-indigo-400 mb-4">
                    ML Model Performance
                  </h3>
                  <div className="space-y-3">
                    <div>
                      <p className="text-xs text-gray-500 dark:text-gray-400">Model</p>
                      <p className="text-sm font-medium text-gray-900 dark:text-gray-100">
                        {mlModelSignals.model_name}
                      </p>
                    </div>
                    <div>
                      <p className="text-xs text-gray-500 dark:text-gray-400">Holding Period</p>
                      <p className="text-sm font-medium text-gray-900 dark:text-gray-100">
                        {mlModelSignals.holding_period} quarter{mlModelSignals.holding_period !== 1 ? "s" : ""}
                      </p>
                    </div>
                    <div className="pt-2 border-t border-gray-200 dark:border-gray-700">
                      <p className="text-xs text-gray-500 dark:text-gray-400">Trades in View</p>
                      <p className="text-lg font-bold text-gray-900 dark:text-gray-100">
                        {visibleMLStats.numTrades}
                        {visibleMLStats.totalNumTrades > visibleMLStats.numTrades && (
                          <span className="text-xs font-normal text-gray-500 ml-1">
                            of {visibleMLStats.totalNumTrades} total
                          </span>
                        )}
                      </p>
                    </div>
                    {visibleMLStats.totalReturn !== null && (
                      <div>
                        <p className="text-xs text-gray-500 dark:text-gray-400">Total Return</p>
                        <p className={`text-lg font-bold ${
                          visibleMLStats.totalReturn >= 0
                            ? "text-green-600 dark:text-green-400"
                            : "text-red-600 dark:text-red-400"
                        }`}>
                          {visibleMLStats.totalReturn >= 0 ? "+" : ""}{visibleMLStats.totalReturn.toFixed(1)}%
                        </p>
                      </div>
                    )}
                    {visibleMLStats.totalAlpha !== null && (
                      <div>
                        <p className="text-xs text-gray-500 dark:text-gray-400">Total Alpha vs S&P 500</p>
                        <p className={`text-lg font-bold ${
                          visibleMLStats.totalAlpha >= 0
                            ? "text-green-600 dark:text-green-400"
                            : "text-red-600 dark:text-red-400"
                        }`}>
                          {visibleMLStats.totalAlpha >= 0 ? "+" : ""}{visibleMLStats.totalAlpha.toFixed(1)}%
                        </p>
                      </div>
                    )}
                    {visibleMLStats.avgAlpha !== null && (
                      <div>
                        <p className="text-xs text-gray-500 dark:text-gray-400">Avg Alpha per Trade</p>
                        <p className={`text-sm font-medium ${
                          visibleMLStats.avgAlpha >= 0
                            ? "text-green-600 dark:text-green-400"
                            : "text-red-600 dark:text-red-400"
                        }`}>
                          {visibleMLStats.avgAlpha >= 0 ? "+" : ""}{visibleMLStats.avgAlpha.toFixed(2)}%
                        </p>
                      </div>
                    )}
                    {visibleMLStats.winRate !== null && (
                      <div>
                        <p className="text-xs text-gray-500 dark:text-gray-400">Win Rate</p>
                        <p className="text-sm font-medium text-gray-900 dark:text-gray-100">
                          {visibleMLStats.winRate}%
                        </p>
                      </div>
                    )}
                  </div>
                </div>
              )}

              {/* Empty state when nothing selected */}
              {!strategySignals && !mlModelSignals && (
                <div className="bg-white dark:bg-gray-900 rounded-lg border border-gray-200 dark:border-gray-800 p-4">
                  <h3 className="text-sm font-semibold text-gray-900 dark:text-gray-100 mb-4">
                    Performance
                  </h3>
                  <p className="text-sm text-gray-500 dark:text-gray-400">
                    Select a strategy or ML model to see how it would have performed on this stock
                  </p>
                </div>
              )}
            </div>
          </div>
        </div>
      )}

      {/* 4-Stage Pipeline Analysis */}
      <div className="space-y-4 mb-8">
        {/* Stage 1: Survival */}
        <StageCard
          title="Survival"
          icon={<RiShieldCheckLine className="w-5 h-5 text-green-600" />}
          status={survivalPass ? "pass" : survivalPass === false ? "fail" : "neutral"}
        >
          <div className="space-y-4 text-sm text-gray-700 dark:text-gray-300 leading-relaxed">
            <p>{getAltmanNarrative(altman?.z_score, altman?.zone, symbol)}</p>
            <p>{getPiotroskiNarrative(piotroski?.f_score, symbol)}</p>
            <p className="font-medium text-gray-900 dark:text-gray-100 pt-2 border-t border-gray-200 dark:border-gray-700">
              {getSurvivalSummary(altmanPass, piotroskiPass, symbol)}
            </p>
          </div>
        </StageCard>

        {/* Stage 2: Quality */}
        <StageCard
          title="Quality"
          icon={
            <span
              className={`w-5 h-5 flex items-center justify-center rounded-full text-xs font-bold ${
                qualityLabel === "Compounder"
                  ? "bg-emerald-100 text-emerald-700 dark:bg-emerald-900 dark:text-emerald-300"
                  : qualityLabel === "Average"
                  ? "bg-yellow-100 text-yellow-700 dark:bg-yellow-900 dark:text-yellow-300"
                  : "bg-gray-100 text-gray-600 dark:bg-gray-800 dark:text-gray-400"
              }`}
            >
              Q
            </span>
          }
          status={qualityLabel === "Compounder" ? "pass" : qualityLabel === "Average" ? "neutral" : "fail"}
        >
          <div className="space-y-4 text-sm text-gray-700 dark:text-gray-300 leading-relaxed">
            <p>{getROICNarrative(roic?.roic, symbol)}</p>
            <p>{getFCFNarrative(roic?.free_cash_flow, roic?.fcf_positive_5yr, symbol)}</p>

            {/* Quality Tags with Narratives */}
            {(() => {
              let parsedTags: string[] = []
              if (roic?.quality_tags) {
                if (typeof roic.quality_tags === "string") {
                  try {
                    parsedTags = JSON.parse(roic.quality_tags)
                  } catch {
                    parsedTags = []
                  }
                } else if (Array.isArray(roic.quality_tags)) {
                  parsedTags = roic.quality_tags
                }
              }

              if (parsedTags.length === 0) return null

              const tagColors: Record<string, string> = {
                "Durable Compounder": "bg-emerald-100 text-emerald-700 dark:bg-emerald-900 dark:text-emerald-300",
                "Cash Machine": "bg-blue-100 text-blue-700 dark:bg-blue-900 dark:text-blue-300",
                "Deep Value": "bg-purple-100 text-purple-700 dark:bg-purple-900 dark:text-purple-300",
                "Heavy Reinvestor": "bg-amber-100 text-amber-700 dark:bg-amber-900 dark:text-amber-300",
                "Volatile Returns": "bg-red-100 text-red-700 dark:bg-red-900 dark:text-red-300",
                "Earnings Quality Concern": "bg-orange-100 text-orange-700 dark:bg-orange-900 dark:text-orange-300",
                "Premium Priced": "bg-gray-100 text-gray-600 dark:bg-gray-800 dark:text-gray-400",
                "Weak Moat Signal": "bg-yellow-100 text-yellow-700 dark:bg-yellow-900 dark:text-yellow-300",
              }

              return (
                <div className="space-y-3 pt-2 border-t border-gray-200 dark:border-gray-700">
                  {parsedTags.map((tag: string) => (
                    <div key={tag}>
                      <span
                        className={`inline-flex items-center px-2 py-1 rounded-full text-xs font-medium mb-1 ${tagColors[tag] || "bg-gray-100 text-gray-600 dark:bg-gray-800 dark:text-gray-400"}`}
                      >
                        {tag}
                      </span>
                      <p className="text-gray-600 dark:text-gray-400">{getQualityTagNarrative(tag)}</p>
                    </div>
                  ))}
                </div>
              )
            })()}

            <p className="font-medium text-gray-900 dark:text-gray-100 pt-2 border-t border-gray-200 dark:border-gray-700">
              {getQualitySummary(qualityLabel, symbol)}
            </p>
          </div>
        </StageCard>

        {/* Stage 3: Valuation Lenses */}
        <StageCard
          title="Valuation Lenses"
          icon={
            <span className="w-5 h-5 flex items-center justify-center rounded-full bg-blue-100 text-blue-700 dark:bg-blue-900 dark:text-blue-300 text-xs font-bold">
              V
            </span>
          }
          status={lensesPassed.length >= 2 ? "pass" : lensesPassed.length >= 1 ? "neutral" : "fail"}
        >
          <div className="space-y-4 text-sm text-gray-700 dark:text-gray-300 leading-relaxed">
            <p>{getGrahamNarrative(graham?.criteria_passed, symbol)}</p>
            <p>{getNetNetNarrative(netNet?.discount_to_ncav, netNet?.trading_below_ncav, symbol)}</p>
            <p>{getPEGNarrative(peg?.peg_ratio, peg?.peg_pass, symbol)}</p>
            <p>{getMagicFormulaNarrative(magicFormula?.combined_rank, symbol)}</p>
            <p>{getFamaFrenchNarrative(famaFrench?.book_to_market_percentile, symbol)}</p>

            <p className="font-medium text-gray-900 dark:text-gray-100 pt-2 border-t border-gray-200 dark:border-gray-700">
              {getValuationSummary(
                lensesPassed.length,
                [
                  grahamPass && "Graham",
                  netNetPass && "Net-Net",
                  pegPass && "PEG",
                  mfPass && "Magic Formula",
                  ffBmPass && "Fama-French",
                ].filter(Boolean) as string[]
              )}
            </p>
          </div>
        </StageCard>

        {/* Stage 4: Factor Exposure */}
        <StageCard
          title="Factor Exposure"
          icon={
            <span className="w-5 h-5 flex items-center justify-center rounded-full bg-gray-100 text-gray-600 dark:bg-gray-800 dark:text-gray-400 text-xs font-bold">
              F
            </span>
          }
          status="neutral"
        >
          <div className="space-y-4 text-sm text-gray-700 dark:text-gray-300 leading-relaxed">
            <p className="text-xs text-gray-500 italic">Factor exposure helps with portfolio construction - understanding how different stocks complement each other based on academic risk factors.</p>
            <p>{getSizeNarrative(profile?.market_cap, symbol)}</p>
            <p>{getProfitabilityNarrative(famaFrench?.profitability_percentile, symbol)}</p>
            <p className="font-medium text-gray-900 dark:text-gray-100 pt-2 border-t border-gray-200 dark:border-gray-700">
              {getFactorSummary(
                profile?.market_cap,
                famaFrench?.book_to_market_percentile,
                famaFrench?.profitability_percentile,
                symbol
              )}
            </p>
          </div>
        </StageCard>
      </div>

      {/* Financial Data Tabs */}
      <div>
        <h2 className="text-lg font-semibold text-gray-900 dark:text-gray-50 mb-4">
          Financial Statements
        </h2>
        <div className="border-b border-gray-200 dark:border-gray-800 mb-4">
          <nav className="-mb-px flex space-x-8">
            {[
              { id: "income", label: "Income Statement" },
              { id: "balance", label: "Balance Sheet" },
              { id: "cashflow", label: "Cash Flow" },
              { id: "dividends", label: "Dividends" },
            ].map((tab) => (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id as any)}
                className={`py-2 px-1 border-b-2 text-sm font-medium ${
                  activeTab === tab.id
                    ? "border-indigo-500 text-indigo-600"
                    : "border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300"
                }`}
              >
                {tab.label}
              </button>
            ))}
          </nav>
        </div>

        {activeTab === "income" && incomeStatements.length > 0 && (
          <div className="overflow-x-auto">
            <table className="min-w-full divide-y divide-gray-200 dark:divide-gray-800">
              <thead className="bg-gray-50 dark:bg-gray-900">
                <tr>
                  <th className="px-4 py-2 text-left text-xs font-medium text-gray-500">Date</th>
                  <th className="px-4 py-2 text-right text-xs font-medium text-gray-500">Revenue</th>
                  <th className="px-4 py-2 text-right text-xs font-medium text-gray-500">Gross Profit</th>
                  <th className="px-4 py-2 text-right text-xs font-medium text-gray-500">Operating Income</th>
                  <th className="px-4 py-2 text-right text-xs font-medium text-gray-500">Net Income</th>
                  <th className="px-4 py-2 text-right text-xs font-medium text-gray-500">EPS</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-gray-200 dark:divide-gray-800">
                {incomeStatements.slice(0, 5).map((stmt: any, i: number) => (
                  <tr key={i}>
                    <td className="px-4 py-2 text-sm text-gray-900 dark:text-gray-100">
                      {stmt.fiscal_date || stmt.date}
                    </td>
                    <td className="px-4 py-2 text-sm text-right text-gray-500">{formatCurrency(stmt.revenue)}</td>
                    <td className="px-4 py-2 text-sm text-right text-gray-500">{formatCurrency(stmt.gross_profit)}</td>
                    <td className="px-4 py-2 text-sm text-right text-gray-500">
                      {formatCurrency(stmt.operating_income)}
                    </td>
                    <td className="px-4 py-2 text-sm text-right text-gray-500">{formatCurrency(stmt.net_income)}</td>
                    <td className="px-4 py-2 text-sm text-right text-gray-500">{formatNumber(stmt.eps, 2)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}

        {activeTab === "balance" && balanceSheets.length > 0 && (
          <div className="overflow-x-auto">
            <table className="min-w-full divide-y divide-gray-200 dark:divide-gray-800">
              <thead className="bg-gray-50 dark:bg-gray-900">
                <tr>
                  <th className="px-4 py-2 text-left text-xs font-medium text-gray-500">Date</th>
                  <th className="px-4 py-2 text-right text-xs font-medium text-gray-500">Total Assets</th>
                  <th className="px-4 py-2 text-right text-xs font-medium text-gray-500">Total Liabilities</th>
                  <th className="px-4 py-2 text-right text-xs font-medium text-gray-500">Total Equity</th>
                  <th className="px-4 py-2 text-right text-xs font-medium text-gray-500">Cash</th>
                  <th className="px-4 py-2 text-right text-xs font-medium text-gray-500">Total Debt</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-gray-200 dark:divide-gray-800">
                {balanceSheets.slice(0, 5).map((stmt: any, i: number) => (
                  <tr key={i}>
                    <td className="px-4 py-2 text-sm text-gray-900 dark:text-gray-100">
                      {stmt.fiscal_date || stmt.date}
                    </td>
                    <td className="px-4 py-2 text-sm text-right text-gray-500">{formatCurrency(stmt.total_assets)}</td>
                    <td className="px-4 py-2 text-sm text-right text-gray-500">
                      {formatCurrency(stmt.total_liabilities)}
                    </td>
                    <td className="px-4 py-2 text-sm text-right text-gray-500">{formatCurrency(stmt.total_equity)}</td>
                    <td className="px-4 py-2 text-sm text-right text-gray-500">
                      {formatCurrency(stmt.cash_and_equivalents)}
                    </td>
                    <td className="px-4 py-2 text-sm text-right text-gray-500">{formatCurrency(stmt.total_debt)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}

        {activeTab === "cashflow" && cashFlows.length > 0 && (
          <div className="overflow-x-auto">
            <table className="min-w-full divide-y divide-gray-200 dark:divide-gray-800">
              <thead className="bg-gray-50 dark:bg-gray-900">
                <tr>
                  <th className="px-4 py-2 text-left text-xs font-medium text-gray-500">Date</th>
                  <th className="px-4 py-2 text-right text-xs font-medium text-gray-500">Operating CF</th>
                  <th className="px-4 py-2 text-right text-xs font-medium text-gray-500">CapEx</th>
                  <th className="px-4 py-2 text-right text-xs font-medium text-gray-500">Free Cash Flow</th>
                  <th className="px-4 py-2 text-right text-xs font-medium text-gray-500">Dividends Paid</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-gray-200 dark:divide-gray-800">
                {cashFlows.slice(0, 5).map((stmt: any, i: number) => (
                  <tr key={i}>
                    <td className="px-4 py-2 text-sm text-gray-900 dark:text-gray-100">
                      {stmt.fiscal_date || stmt.date}
                    </td>
                    <td className="px-4 py-2 text-sm text-right text-gray-500">
                      {formatCurrency(stmt.operating_cash_flow)}
                    </td>
                    <td className="px-4 py-2 text-sm text-right text-gray-500">
                      {formatCurrency(stmt.capital_expenditure)}
                    </td>
                    <td className="px-4 py-2 text-sm text-right text-gray-500">
                      {formatCurrency(stmt.free_cash_flow)}
                    </td>
                    <td className="px-4 py-2 text-sm text-right text-gray-500">
                      {formatCurrency(stmt.dividends_paid)}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}

        {activeTab === "dividends" && dividends.length > 0 && (
          <div className="overflow-x-auto">
            <table className="min-w-full divide-y divide-gray-200 dark:divide-gray-800">
              <thead className="bg-gray-50 dark:bg-gray-900">
                <tr>
                  <th className="px-4 py-2 text-left text-xs font-medium text-gray-500">Ex-Date</th>
                  <th className="px-4 py-2 text-left text-xs font-medium text-gray-500">Payment Date</th>
                  <th className="px-4 py-2 text-right text-xs font-medium text-gray-500">Amount</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-gray-200 dark:divide-gray-800">
                {dividends.slice(0, 20).map((div: any, i: number) => (
                  <tr key={i}>
                    <td className="px-4 py-2 text-sm text-gray-900 dark:text-gray-100">{div.ex_date}</td>
                    <td className="px-4 py-2 text-sm text-gray-500">{div.payment_date}</td>
                    <td className="px-4 py-2 text-sm text-right text-gray-500">${formatNumber(div.amount, 4)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}

        {((activeTab === "income" && incomeStatements.length === 0) ||
          (activeTab === "balance" && balanceSheets.length === 0) ||
          (activeTab === "cashflow" && cashFlows.length === 0) ||
          (activeTab === "dividends" && dividends.length === 0)) && (
          <div className="text-center py-8 text-gray-500">No data available for this section.</div>
        )}
      </div>

      {/* AI Explain Chat Panel */}
      <StockExplainChat
        symbol={symbol}
        open={showExplain}
        onOpenChange={setShowExplain}
      />
    </div>
  )
}
