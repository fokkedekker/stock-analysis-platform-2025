"use client"

import { useEffect, useState, useMemo } from "react"
import Link from "next/link"
import { Badge } from "@/components/Badge"
import { Pagination } from "@/components/Pagination"
import { getPiotroskiStocks, PiotroskiStock } from "@/lib/api"
import { RiCheckLine, RiCloseLine } from "@remixicon/react"
import { ScreenerInfo, SCREENER_INFO } from "@/components/ScreenerInfo"

const ITEMS_PER_PAGE = 50

const SIGNALS = [
  { key: "roa_positive", label: "ROA Positive", abbr: "ROA+", category: "Profitability" },
  { key: "operating_cf_positive", label: "Operating CF Positive", abbr: "OCF+", category: "Profitability" },
  { key: "roa_increasing", label: "ROA Increasing", abbr: "ROA↑", category: "Profitability" },
  { key: "accruals_signal", label: "Accruals (OCF > NI)", abbr: "Acc", category: "Profitability" },
  { key: "leverage_decreasing", label: "Leverage Decreasing", abbr: "Lev↓", category: "Leverage" },
  { key: "current_ratio_increasing", label: "Current Ratio Increasing", abbr: "CR↑", category: "Leverage" },
  { key: "no_dilution", label: "No Share Dilution", abbr: "NoDil", category: "Leverage" },
  { key: "gross_margin_increasing", label: "Gross Margin Increasing", abbr: "GM↑", category: "Efficiency" },
  { key: "asset_turnover_increasing", label: "Asset Turnover Increasing", abbr: "AT↑", category: "Efficiency" },
] as const

type SignalKey = typeof SIGNALS[number]["key"]

function SignalBadge({ pass }: { pass: boolean | null }) {
  if (pass === true) {
    return <RiCheckLine className="w-4 h-4 text-green-600" />
  }
  if (pass === false) {
    return <RiCloseLine className="w-4 h-4 text-red-500" />
  }
  return <span className="text-gray-400">—</span>
}

export default function PiotroskiPage() {
  const [allStocks, setAllStocks] = useState<PiotroskiStock[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [selectedSignals, setSelectedSignals] = useState<Set<SignalKey>>(new Set())
  const [currentPage, setCurrentPage] = useState(1)

  useEffect(() => {
    async function fetchData() {
      try {
        setLoading(true)
        // Fetch all stocks (min_score=0 to get everything)
        const data = await getPiotroskiStocks(0, 500)
        setAllStocks(data.stocks)
        setError(null)
      } catch (err) {
        setError("Failed to fetch data. Is the API server running?")
        console.error(err)
      } finally {
        setLoading(false)
      }
    }
    fetchData()
  }, [])

  // Filter stocks based on selected signals
  const filteredStocks = useMemo(() => {
    if (selectedSignals.size === 0) return allStocks
    const signalKeys = Array.from(selectedSignals)
    return allStocks.filter((stock) => {
      for (const signalKey of signalKeys) {
        if (stock[signalKey] !== true) return false
      }
      return true
    })
  }, [allStocks, selectedSignals])

  // Paginate filtered stocks
  const totalPages = Math.ceil(filteredStocks.length / ITEMS_PER_PAGE)
  const paginatedStocks = useMemo(() => {
    const start = (currentPage - 1) * ITEMS_PER_PAGE
    return filteredStocks.slice(start, start + ITEMS_PER_PAGE)
  }, [filteredStocks, currentPage])

  // Reset to page 1 when filters change
  useEffect(() => {
    setCurrentPage(1)
  }, [selectedSignals])

  const toggleSignal = (key: SignalKey) => {
    setSelectedSignals((prev) => {
      const next = new Set(prev)
      if (next.has(key)) {
        next.delete(key)
      } else {
        next.add(key)
      }
      return next
    })
  }

  const clearFilters = () => setSelectedSignals(new Set())

  const selectCategory = (category: string) => {
    const categorySignals = SIGNALS.filter(s => s.category === category).map(s => s.key)
    setSelectedSignals(new Set(categorySignals))
  }

  const allSignalKeys = SIGNALS.map(s => s.key)
  const allSelected = selectedSignals.size === SIGNALS.length

  const toggleAll = () => {
    if (allSelected) {
      setSelectedSignals(new Set())
    } else {
      setSelectedSignals(new Set(allSignalKeys))
    }
  }

  // Group signals by category
  const signalsByCategory = {
    Profitability: SIGNALS.filter(s => s.category === "Profitability"),
    Leverage: SIGNALS.filter(s => s.category === "Leverage"),
    Efficiency: SIGNALS.filter(s => s.category === "Efficiency"),
  }

  return (
    <div className="p-4 sm:p-6 lg:p-8">
      <div className="mb-8">
        <div className="flex items-start justify-between gap-4">
          <div>
            <h1 className="text-2xl font-bold text-gray-900 dark:text-gray-50">
              Piotroski F-Score
            </h1>
            <p className="mt-1 text-sm text-gray-500">
              9 binary signals measuring profitability, leverage, and efficiency
            </p>
          </div>
          <ScreenerInfo {...SCREENER_INFO.piotroski} />
        </div>
      </div>

      {/* Signal Filter by Category */}
      <div className="mb-6 space-y-4">
        {Object.entries(signalsByCategory).map(([category, signals]) => (
          <div key={category}>
            <div className="flex items-center gap-2 mb-2">
              <span className="text-sm font-medium text-gray-700 dark:text-gray-300">
                {category}
              </span>
              <button
                onClick={() => selectCategory(category)}
                className="text-xs text-indigo-600 hover:text-indigo-500"
              >
                Select all
              </button>
            </div>
            <div className="flex flex-wrap items-center gap-2">
              {signals.map((signal) => (
                <label
                  key={signal.key}
                  className={`flex items-center gap-2 px-3 py-1.5 rounded-lg border cursor-pointer transition-colors text-sm ${
                    selectedSignals.has(signal.key)
                      ? "border-indigo-500 bg-indigo-50 dark:bg-indigo-950"
                      : "border-gray-200 dark:border-gray-800 hover:border-gray-300"
                  }`}
                >
                  <input
                    type="checkbox"
                    checked={selectedSignals.has(signal.key)}
                    onChange={() => toggleSignal(signal.key)}
                    className="w-3.5 h-3.5 rounded border-gray-300 text-indigo-600 focus:ring-indigo-500"
                  />
                  <span className="text-gray-700 dark:text-gray-300">
                    {signal.label}
                  </span>
                </label>
              ))}
            </div>
          </div>
        ))}

        <div className="flex items-center gap-4 pt-2">
          <button
            onClick={toggleAll}
            className="text-sm text-indigo-600 hover:text-indigo-500 font-medium"
          >
            {allSelected ? "Unselect all" : "Select all"}
          </button>
          <p className="text-sm text-gray-500">
            {selectedSignals.size === 0
              ? `Showing all ${filteredStocks.length} stocks`
              : `Showing ${filteredStocks.length} stocks passing ${selectedSignals.size} selected signal${selectedSignals.size > 1 ? "s" : ""}`}
          </p>
        </div>
      </div>

      {error && (
        <div className="rounded-lg border border-red-200 bg-red-50 p-4 dark:border-red-900 dark:bg-red-950 mb-4">
          <p className="text-sm text-red-600 dark:text-red-400">{error}</p>
        </div>
      )}

      {loading && (
        <div className="flex items-center justify-center py-12">
          <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-indigo-600"></div>
        </div>
      )}

      {!loading && !error && (
        <div className="overflow-x-auto rounded-lg border border-gray-200 dark:border-gray-800">
          <table className="min-w-full divide-y divide-gray-200 dark:divide-gray-800">
            <thead className="bg-gray-50 dark:bg-gray-900">
              <tr>
                <th className="px-4 py-3 text-right text-xs font-medium text-gray-500 uppercase w-12">#</th>
                <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">Symbol</th>
                <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">Name</th>
                <th className="px-4 py-3 text-center text-xs font-medium text-gray-500 uppercase">F-Score</th>
                <th className="px-4 py-3 text-center text-xs font-medium text-gray-500 uppercase" title="ROA Positive">ROA+</th>
                <th className="px-4 py-3 text-center text-xs font-medium text-gray-500 uppercase" title="Operating CF Positive">OCF+</th>
                <th className="px-4 py-3 text-center text-xs font-medium text-gray-500 uppercase" title="ROA Increasing">ROA↑</th>
                <th className="px-4 py-3 text-center text-xs font-medium text-gray-500 uppercase" title="Accruals Signal">Acc</th>
                <th className="px-4 py-3 text-center text-xs font-medium text-gray-500 uppercase" title="Leverage Decreasing">Lev↓</th>
                <th className="px-4 py-3 text-center text-xs font-medium text-gray-500 uppercase" title="Current Ratio Increasing">CR↑</th>
                <th className="px-4 py-3 text-center text-xs font-medium text-gray-500 uppercase" title="No Dilution">NoDil</th>
                <th className="px-4 py-3 text-center text-xs font-medium text-gray-500 uppercase" title="Gross Margin Increasing">GM↑</th>
                <th className="px-4 py-3 text-center text-xs font-medium text-gray-500 uppercase" title="Asset Turnover Increasing">AT↑</th>
              </tr>
            </thead>
            <tbody className="bg-white divide-y divide-gray-200 dark:bg-gray-950 dark:divide-gray-800">
              {paginatedStocks.map((stock, index) => (
                <tr key={stock.symbol} className="hover:bg-gray-50 dark:hover:bg-gray-900">
                  <td className="px-4 py-3 whitespace-nowrap text-right text-sm text-gray-500">{(currentPage - 1) * ITEMS_PER_PAGE + index + 1}</td>
                  <td className="px-4 py-3 whitespace-nowrap">
                    <Link href={`/stock/${stock.symbol}`} className="text-sm font-medium text-indigo-600 hover:text-indigo-500">
                      {stock.symbol}
                    </Link>
                  </td>
                  <td className="px-4 py-3 whitespace-nowrap text-sm text-gray-900 dark:text-gray-100 max-w-[200px] truncate">
                    {stock.name}
                  </td>
                  <td className="px-4 py-3 text-center">
                    <Badge variant={stock.f_score >= 7 ? "success" : stock.f_score >= 5 ? "warning" : "error"}>
                      {stock.f_score}/9
                    </Badge>
                  </td>
                  <td className="px-4 py-3 text-center"><SignalBadge pass={stock.roa_positive} /></td>
                  <td className="px-4 py-3 text-center"><SignalBadge pass={stock.operating_cf_positive} /></td>
                  <td className="px-4 py-3 text-center"><SignalBadge pass={stock.roa_increasing} /></td>
                  <td className="px-4 py-3 text-center"><SignalBadge pass={stock.accruals_signal} /></td>
                  <td className="px-4 py-3 text-center"><SignalBadge pass={stock.leverage_decreasing} /></td>
                  <td className="px-4 py-3 text-center"><SignalBadge pass={stock.current_ratio_increasing} /></td>
                  <td className="px-4 py-3 text-center"><SignalBadge pass={stock.no_dilution} /></td>
                  <td className="px-4 py-3 text-center"><SignalBadge pass={stock.gross_margin_increasing} /></td>
                  <td className="px-4 py-3 text-center"><SignalBadge pass={stock.asset_turnover_increasing} /></td>
                </tr>
              ))}
            </tbody>
          </table>
          <Pagination
            currentPage={currentPage}
            totalPages={totalPages}
            onPageChange={setCurrentPage}
            totalItems={filteredStocks.length}
            itemsPerPage={ITEMS_PER_PAGE}
          />
        </div>
      )}

      {!loading && !error && filteredStocks.length === 0 && (
        <div className="text-center py-12 text-gray-500">
          No stocks found matching the selected signals.
        </div>
      )}
    </div>
  )
}
