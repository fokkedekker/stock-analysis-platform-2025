"use client"

import { useEffect, useState } from "react"
import { useParams } from "next/navigation"
import Link from "next/link"
import {
  getStockAnalysis,
  getStockProfile,
  getIncomeStatements,
  getBalanceSheets,
  getCashFlows,
  getDividends,
  formatNumber,
  formatCurrency,
  formatPercent,
  StockAnalysis,
  StockProfile,
} from "@/lib/api"
import {
  RiArrowLeftLine,
  RiCheckLine,
  RiCloseLine,
  RiShieldCheckLine,
  RiSparklingLine,
} from "@remixicon/react"
import { StockExplainChat } from "@/components/StockExplainChat"
import { Button } from "@/components/Button"

function PassBadge({ pass, label }: { pass: boolean | null | undefined; label?: string }) {
  if (pass === true) {
    return (
      <div className="flex items-center gap-1 text-green-600">
        <RiCheckLine className="w-4 h-4" />
        {label && <span className="text-sm font-medium">{label}</span>}
      </div>
    )
  }
  if (pass === false) {
    return (
      <div className="flex items-center gap-1 text-red-500">
        <RiCloseLine className="w-4 h-4" />
        {label && <span className="text-sm font-medium">{label}</span>}
      </div>
    )
  }
  return <span className="text-gray-400 text-sm">N/A</span>
}

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

      {/* 4-Stage Pipeline Analysis */}
      <div className="space-y-4 mb-8">
        {/* Stage 1: Survival */}
        <StageCard
          title="Survival"
          icon={<RiShieldCheckLine className="w-5 h-5 text-green-600" />}
          status={survivalPass ? "pass" : survivalPass === false ? "fail" : "neutral"}
        >
          <div className="grid grid-cols-2 gap-4">
            <div>
              <div className="flex items-center justify-between mb-1">
                <span className="text-sm font-medium text-gray-700 dark:text-gray-300">Altman Z-Score</span>
                <PassBadge pass={altmanPass} />
              </div>
              <div className="text-2xl font-bold text-gray-900 dark:text-gray-100">
                {altman ? formatNumber(altman.z_score, 2) : "N/A"}
              </div>
              <div className="text-xs text-gray-500">
                Zone: {altman?.zone || "N/A"}
              </div>
            </div>
            <div>
              <div className="flex items-center justify-between mb-1">
                <span className="text-sm font-medium text-gray-700 dark:text-gray-300">Piotroski F-Score</span>
                <PassBadge pass={piotroskiPass} />
              </div>
              <div className="text-2xl font-bold text-gray-900 dark:text-gray-100">
                {piotroski ? `${piotroski.f_score}/9` : "N/A"}
              </div>
            </div>
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
          <div className="flex items-center gap-4 mb-3">
            <span
              className={`inline-flex items-center px-3 py-1 rounded-full text-sm font-medium ${
                qualityLabel === "Compounder"
                  ? "bg-emerald-100 text-emerald-700 dark:bg-emerald-900 dark:text-emerald-300"
                  : qualityLabel === "Average"
                  ? "bg-yellow-100 text-yellow-700 dark:bg-yellow-900 dark:text-yellow-300"
                  : "bg-gray-100 text-gray-600 dark:bg-gray-800 dark:text-gray-400"
              }`}
            >
              {qualityLabel}
            </span>
          </div>
          <div className="grid grid-cols-3 gap-4">
            <div>
              <div className="text-xs text-gray-500 mb-1">ROIC</div>
              <div className="text-lg font-bold text-gray-900 dark:text-gray-100">
                {roic ? formatPercent(roic.roic) : "N/A"}
              </div>
            </div>
            <div>
              <div className="text-xs text-gray-500 mb-1">Free Cash Flow</div>
              <div className="text-lg font-bold text-gray-900 dark:text-gray-100">
                {roic?.free_cash_flow != null ? formatCurrency(roic.free_cash_flow) : "N/A"}
              </div>
            </div>
            <div>
              <div className="text-xs text-gray-500 mb-1">FCF 5yr+</div>
              <div className="text-lg font-bold text-gray-900 dark:text-gray-100">
                {roic?.fcf_positive_5yr ? "Yes" : "No"}
              </div>
            </div>
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
          <div className="flex flex-wrap gap-2 mb-4">
            {grahamPass && (
              <span className="inline-flex items-center px-2 py-1 rounded text-xs font-medium bg-blue-100 text-blue-700 dark:bg-blue-900 dark:text-blue-300">
                Graham
              </span>
            )}
            {netNetPass && (
              <span className="inline-flex items-center px-2 py-1 rounded text-xs font-medium bg-purple-100 text-purple-700 dark:bg-purple-900 dark:text-purple-300">
                Net-Net
              </span>
            )}
            {pegPass && (
              <span className="inline-flex items-center px-2 py-1 rounded text-xs font-medium bg-teal-100 text-teal-700 dark:bg-teal-900 dark:text-teal-300">
                PEG
              </span>
            )}
            {mfPass && (
              <span className="inline-flex items-center px-2 py-1 rounded text-xs font-medium bg-orange-100 text-orange-700 dark:bg-orange-900 dark:text-orange-300">
                Magic Formula
              </span>
            )}
            {ffBmPass && (
              <span className="inline-flex items-center px-2 py-1 rounded text-xs font-medium bg-pink-100 text-pink-700 dark:bg-pink-900 dark:text-pink-300">
                FF B/M
              </span>
            )}
            {lensesPassed.length === 0 && (
              <span className="text-gray-400 text-sm">No valuation lenses passed</span>
            )}
          </div>
          <div className="grid grid-cols-2 md:grid-cols-5 gap-4 text-sm">
            <div>
              <div className="flex items-center justify-between">
                <span className="text-gray-500">Graham</span>
                <PassBadge pass={grahamPass} />
              </div>
              <div className="text-gray-900 dark:text-gray-100">
                {graham ? `${graham.criteria_passed}/8` : "N/A"}
              </div>
            </div>
            <div>
              <div className="flex items-center justify-between">
                <span className="text-gray-500">Net-Net</span>
                <PassBadge pass={netNetPass} />
              </div>
              <div className="text-gray-900 dark:text-gray-100">
                {netNet ? formatPercent(netNet.discount_to_ncav) : "N/A"}
              </div>
            </div>
            <div>
              <div className="flex items-center justify-between">
                <span className="text-gray-500">PEG</span>
                <PassBadge pass={pegPass} />
              </div>
              <div className="text-gray-900 dark:text-gray-100">
                {peg ? formatNumber(peg.peg_ratio, 2) : "N/A"}
              </div>
            </div>
            <div>
              <div className="flex items-center justify-between">
                <span className="text-gray-500">MF Rank</span>
                <PassBadge pass={mfPass} />
              </div>
              <div className="text-gray-900 dark:text-gray-100">
                {magicFormula ? `#${magicFormula.combined_rank}` : "N/A"}
              </div>
            </div>
            <div>
              <div className="flex items-center justify-between">
                <span className="text-gray-500">FF B/M</span>
                <PassBadge pass={ffBmPass} />
              </div>
              <div className="text-gray-900 dark:text-gray-100">
                {famaFrench?.book_to_market_percentile
                  ? `${(famaFrench.book_to_market_percentile * 100).toFixed(0)}%ile`
                  : "N/A"}
              </div>
            </div>
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
          <p className="text-xs text-gray-500 mb-3">Context for portfolio construction (never used as a filter)</p>
          <div className="grid grid-cols-3 gap-4">
            <div>
              <div className="text-xs text-gray-500 mb-1">Size</div>
              <div className="text-sm font-medium text-gray-900 dark:text-gray-100">
                {profile?.market_cap
                  ? profile.market_cap >= 10e9
                    ? "Large-Cap"
                    : profile.market_cap >= 2e9
                    ? "Mid-Cap"
                    : "Small-Cap"
                  : "N/A"}
              </div>
            </div>
            <div>
              <div className="text-xs text-gray-500 mb-1">Value (B/M)</div>
              <div className="text-sm font-medium text-gray-900 dark:text-gray-100">
                {famaFrench?.book_to_market_percentile
                  ? famaFrench.book_to_market_percentile >= 0.7
                    ? "Value"
                    : famaFrench.book_to_market_percentile <= 0.3
                    ? "Growth"
                    : "Neutral"
                  : "N/A"}
              </div>
            </div>
            <div>
              <div className="text-xs text-gray-500 mb-1">Profitability</div>
              <div className="text-sm font-medium text-gray-900 dark:text-gray-100">
                {famaFrench?.profitability_percentile
                  ? famaFrench.profitability_percentile >= 0.7
                    ? "High"
                    : famaFrench.profitability_percentile <= 0.3
                    ? "Low"
                    : "Medium"
                  : "N/A"}
              </div>
            </div>
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
