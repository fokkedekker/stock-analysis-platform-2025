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
import { RiArrowLeftLine, RiCheckLine, RiCloseLine } from "@remixicon/react"

function PassBadge({ pass, label }: { pass: boolean | null | undefined; label?: string }) {
  if (pass === true) {
    return (
      <div className="flex items-center gap-1 text-green-600">
        <RiCheckLine className="w-4 h-4" />
        {label && <span className="text-xs">{label}</span>}
      </div>
    )
  }
  if (pass === false) {
    return (
      <div className="flex items-center gap-1 text-red-500">
        <RiCloseLine className="w-4 h-4" />
        {label && <span className="text-xs">{label}</span>}
      </div>
    )
  }
  return <span className="text-gray-400">N/A</span>
}

function AnalysisCard({
  title,
  pass,
  score,
  details,
}: {
  title: string
  pass: boolean | null
  score?: string
  details?: string[]
}) {
  return (
    <div className={`p-4 rounded-lg border ${pass ? 'border-green-200 bg-green-50 dark:border-green-900 dark:bg-green-950' : pass === false ? 'border-red-200 bg-red-50 dark:border-red-900 dark:bg-red-950' : 'border-gray-200 dark:border-gray-800'}`}>
      <div className="flex items-center justify-between mb-2">
        <h3 className="text-sm font-medium text-gray-900 dark:text-gray-100">{title}</h3>
        <PassBadge pass={pass} />
      </div>
      {score && (
        <div className="text-lg font-bold text-gray-900 dark:text-gray-100 mb-1">{score}</div>
      )}
      {details && details.length > 0 && (
        <ul className="text-xs text-gray-500 space-y-0.5">
          {details.map((d, i) => (
            <li key={i}>{d}</li>
          ))}
        </ul>
      )}
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
  const [activeTab, setActiveTab] = useState<'income' | 'balance' | 'cashflow' | 'dividends'>('income')

  useEffect(() => {
    async function fetchData() {
      try {
        setLoading(true)
        const [analysisData, profileData, incomeData, balanceData, cashData, divData] = await Promise.all([
          getStockAnalysis(symbol).catch(() => null),
          getStockProfile(symbol).catch(() => null),
          getIncomeStatements(symbol, 'annual', 5).catch(() => []),
          getBalanceSheets(symbol, 'annual', 5).catch(() => []),
          getCashFlows(symbol, 'annual', 5).catch(() => []),
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

  return (
    <div className="p-4 sm:p-6 lg:p-8">
      {/* Header */}
      <div className="mb-8">
        <Link
          href="/"
          className="inline-flex items-center gap-1 text-sm text-gray-500 hover:text-gray-700 mb-4"
        >
          <RiArrowLeftLine className="w-4 h-4" />
          Back to Rankings
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
          {profile && (
            <div className="text-right text-sm text-gray-500">
              <div>P/E: {formatNumber(profile.pe_ratio, 1)}</div>
              <div>P/B: {formatNumber(profile.pb_ratio, 2)}</div>
              <div>Beta: {formatNumber(profile.beta, 2)}</div>
            </div>
          )}
        </div>
      </div>

      {/* Analysis Cards */}
      <div className="mb-8">
        <h2 className="text-lg font-semibold text-gray-900 dark:text-gray-50 mb-4">
          Analysis Summary
        </h2>
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
          <AnalysisCard
            title="Graham"
            pass={graham?.criteria_passed === 8}
            score={graham ? `${graham.criteria_passed}/8` : undefined}
            details={graham ? [`P/E: ${formatNumber(graham.pe_ratio, 1)}`, `P/B: ${formatNumber(graham.pb_ratio, 2)}`] : undefined}
          />
          <AnalysisCard
            title="Magic Formula"
            pass={magicFormula?.combined_rank <= 100}
            score={magicFormula ? `Rank #${magicFormula.combined_rank}` : undefined}
            details={magicFormula ? [`EY: ${formatPercent(magicFormula.earnings_yield)}`, `ROC: ${formatPercent(magicFormula.return_on_capital)}`] : undefined}
          />
          <AnalysisCard
            title="Piotroski"
            pass={piotroski?.f_score >= 7}
            score={piotroski ? `F-Score: ${piotroski.f_score}/9` : undefined}
          />
          <AnalysisCard
            title="Altman"
            pass={altman?.zone === 'safe'}
            score={altman ? `Z: ${formatNumber(altman.z_score, 2)}` : undefined}
            details={altman ? [`Zone: ${altman.zone}`] : undefined}
          />
          <AnalysisCard
            title="ROIC"
            pass={roic?.roic_pass}
            score={roic ? formatPercent(roic.roic) : undefined}
            details={roic ? [`FCF 5yr: ${roic.fcf_positive_5yr ? 'Yes' : 'No'}`, `D/E: ${formatNumber(roic.debt_to_equity, 2)}`] : undefined}
          />
          <AnalysisCard
            title="PEG"
            pass={peg?.peg_pass}
            score={peg ? `PEG: ${formatNumber(peg.peg_ratio, 2)}` : undefined}
            details={peg ? [`Growth: ${formatPercent(peg.eps_cagr)}`] : undefined}
          />
          <AnalysisCard
            title="Fama-French"
            pass={famaFrench?.profitability_percentile >= 0.7}
            score={famaFrench ? `Prof: ${(famaFrench.profitability_percentile * 100).toFixed(0)}%ile` : undefined}
          />
          <AnalysisCard
            title="Net-Net"
            pass={netNet?.trading_below_ncav}
            score={netNet ? `${formatPercent(netNet.discount_to_ncav)} of NCAV` : undefined}
            details={netNet?.deep_value ? ['Deep Value!'] : undefined}
          />
        </div>
      </div>

      {/* Financial Data Tabs */}
      <div>
        <h2 className="text-lg font-semibold text-gray-900 dark:text-gray-50 mb-4">
          Financial Statements
        </h2>
        <div className="border-b border-gray-200 dark:border-gray-800 mb-4">
          <nav className="-mb-px flex space-x-8">
            {[
              { id: 'income', label: 'Income Statement' },
              { id: 'balance', label: 'Balance Sheet' },
              { id: 'cashflow', label: 'Cash Flow' },
              { id: 'dividends', label: 'Dividends' },
            ].map((tab) => (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id as any)}
                className={`py-2 px-1 border-b-2 text-sm font-medium ${
                  activeTab === tab.id
                    ? 'border-indigo-500 text-indigo-600'
                    : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
                }`}
              >
                {tab.label}
              </button>
            ))}
          </nav>
        </div>

        {activeTab === 'income' && incomeStatements.length > 0 && (
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
                    <td className="px-4 py-2 text-sm text-gray-900 dark:text-gray-100">{stmt.fiscal_date || stmt.date}</td>
                    <td className="px-4 py-2 text-sm text-right text-gray-500">{formatCurrency(stmt.revenue)}</td>
                    <td className="px-4 py-2 text-sm text-right text-gray-500">{formatCurrency(stmt.gross_profit)}</td>
                    <td className="px-4 py-2 text-sm text-right text-gray-500">{formatCurrency(stmt.operating_income)}</td>
                    <td className="px-4 py-2 text-sm text-right text-gray-500">{formatCurrency(stmt.net_income)}</td>
                    <td className="px-4 py-2 text-sm text-right text-gray-500">{formatNumber(stmt.eps, 2)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}

        {activeTab === 'balance' && balanceSheets.length > 0 && (
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
                    <td className="px-4 py-2 text-sm text-gray-900 dark:text-gray-100">{stmt.fiscal_date || stmt.date}</td>
                    <td className="px-4 py-2 text-sm text-right text-gray-500">{formatCurrency(stmt.total_assets)}</td>
                    <td className="px-4 py-2 text-sm text-right text-gray-500">{formatCurrency(stmt.total_liabilities)}</td>
                    <td className="px-4 py-2 text-sm text-right text-gray-500">{formatCurrency(stmt.total_equity)}</td>
                    <td className="px-4 py-2 text-sm text-right text-gray-500">{formatCurrency(stmt.cash_and_equivalents)}</td>
                    <td className="px-4 py-2 text-sm text-right text-gray-500">{formatCurrency(stmt.total_debt)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}

        {activeTab === 'cashflow' && cashFlows.length > 0 && (
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
                    <td className="px-4 py-2 text-sm text-gray-900 dark:text-gray-100">{stmt.fiscal_date || stmt.date}</td>
                    <td className="px-4 py-2 text-sm text-right text-gray-500">{formatCurrency(stmt.operating_cash_flow)}</td>
                    <td className="px-4 py-2 text-sm text-right text-gray-500">{formatCurrency(stmt.capital_expenditure)}</td>
                    <td className="px-4 py-2 text-sm text-right text-gray-500">{formatCurrency(stmt.free_cash_flow)}</td>
                    <td className="px-4 py-2 text-sm text-right text-gray-500">{formatCurrency(stmt.dividends_paid)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}

        {activeTab === 'dividends' && dividends.length > 0 && (
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

        {((activeTab === 'income' && incomeStatements.length === 0) ||
          (activeTab === 'balance' && balanceSheets.length === 0) ||
          (activeTab === 'cashflow' && cashFlows.length === 0) ||
          (activeTab === 'dividends' && dividends.length === 0)) && (
          <div className="text-center py-8 text-gray-500">
            No data available for this section.
          </div>
        )}
      </div>
    </div>
  )
}
