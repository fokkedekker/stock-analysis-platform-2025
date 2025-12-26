"use client"

import { useEffect, useState, useMemo } from "react"
import Link from "next/link"
import { Badge } from "@/components/Badge"
import { Pagination } from "@/components/Pagination"
import { useQuarter } from "@/contexts/QuarterContext"
import { getGrahamStocks, GrahamStock, formatNumber } from "@/lib/api"
import { RiCheckLine, RiCloseLine } from "@remixicon/react"
import { ScreenerInfo, SCREENER_INFO } from "@/components/ScreenerInfo"

const ITEMS_PER_PAGE = 50

const CRITERIA = [
  { key: "adequate_size", label: "Adequate Size", abbr: "Size" },
  { key: "current_ratio_pass", label: "Current Ratio ≥ 2", abbr: "CR" },
  { key: "debt_coverage_pass", label: "Debt Coverage", abbr: "Debt" },
  { key: "earnings_stability", label: "Earnings Stability", abbr: "Earn" },
  { key: "dividend_record", label: "Dividend Record", abbr: "Div" },
  { key: "earnings_growth_pass", label: "Earnings Growth", abbr: "Grow" },
  { key: "pe_ratio_pass", label: "P/E Ratio", abbr: "P/E" },
  { key: "pb_ratio_pass", label: "P/B Ratio", abbr: "P/B" },
] as const

type CriteriaKey = typeof CRITERIA[number]["key"]

// Mode presets define which criteria are required for each mode
const MODE_PRESETS: Record<string, CriteriaKey[]> = {
  strict: [
    "adequate_size",
    "current_ratio_pass",
    "debt_coverage_pass",
    "earnings_stability",
    "dividend_record",
    "earnings_growth_pass",
    "pe_ratio_pass",
    "pb_ratio_pass",
  ],
  simplified: ["pe_ratio_pass", "pb_ratio_pass"],
  modern: ["pe_ratio_pass", "pb_ratio_pass"],
  garp: ["adequate_size", "earnings_stability", "earnings_growth_pass", "pe_ratio_pass"],
  relaxed: [], // No auto-selection, user chooses
}

const MODE_DESCRIPTIONS: Record<string, string> = {
  strict: "All 8 Graham criteria",
  simplified: "Only P/E and P/B ratios (P/E × P/B ≤ 22.5)",
  modern: "P/E ≤ 25, P/B ≤ 3.0 (updated thresholds)",
  garp: "Growth focus: no dividend or P/B requirement",
  relaxed: "Custom selection - choose your own criteria",
}

function CriteriaBadge({ pass }: { pass: boolean | null }) {
  if (pass === true) {
    return <RiCheckLine className="w-4 h-4 text-green-600" />
  }
  if (pass === false) {
    return <RiCloseLine className="w-4 h-4 text-red-500" />
  }
  return <span className="text-gray-400">—</span>
}

export default function GrahamPage() {
  const { quarter } = useQuarter()
  const [allStocks, setAllStocks] = useState<GrahamStock[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [mode, setMode] = useState<"strict" | "simplified" | "modern" | "garp" | "relaxed">("strict")
  const [selectedCriteria, setSelectedCriteria] = useState<Set<CriteriaKey>>(
    new Set(MODE_PRESETS.strict)
  )
  const [currentPage, setCurrentPage] = useState(1)

  // Auto-select criteria when mode changes
  useEffect(() => {
    const preset = MODE_PRESETS[mode] || []
    setSelectedCriteria(new Set(preset))
  }, [mode])

  useEffect(() => {
    async function fetchData() {
      try {
        setLoading(true)
        // Fetch all stocks (min_score=0 to get everything)
        const data = await getGrahamStocks(mode, 0, 500, quarter)
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
  }, [mode, quarter])

  // Filter stocks based on selected criteria
  const filteredStocks = useMemo(() => {
    if (selectedCriteria.size === 0) return allStocks
    const criteriaKeys = Array.from(selectedCriteria)
    return allStocks.filter((stock) => {
      for (const criteriaKey of criteriaKeys) {
        if (stock[criteriaKey] !== true) return false
      }
      return true
    })
  }, [allStocks, selectedCriteria])

  // Paginate filtered stocks
  const totalPages = Math.ceil(filteredStocks.length / ITEMS_PER_PAGE)
  const paginatedStocks = useMemo(() => {
    const start = (currentPage - 1) * ITEMS_PER_PAGE
    return filteredStocks.slice(start, start + ITEMS_PER_PAGE)
  }, [filteredStocks, currentPage])

  // Reset to page 1 when filters change
  useEffect(() => {
    setCurrentPage(1)
  }, [selectedCriteria, mode])

  const toggleCriteria = (key: CriteriaKey) => {
    setSelectedCriteria((prev) => {
      const next = new Set(prev)
      if (next.has(key)) {
        next.delete(key)
      } else {
        next.add(key)
      }
      return next
    })
  }

  const allCriteriaKeys = CRITERIA.map(c => c.key)
  const allSelected = selectedCriteria.size === CRITERIA.length

  const toggleAll = () => {
    if (allSelected) {
      setSelectedCriteria(new Set())
    } else {
      setSelectedCriteria(new Set(allCriteriaKeys))
    }
  }

  return (
    <div className="p-4 sm:p-6 lg:p-8">
      <div className="mb-8">
        <div className="flex items-start justify-between gap-4">
          <div>
            <h1 className="text-2xl font-bold text-gray-900 dark:text-gray-50">
              Benjamin Graham Criteria
            </h1>
            <p className="mt-1 text-sm text-gray-500">
              Classic value investing criteria from &quot;The Intelligent Investor&quot;
            </p>
          </div>
          <ScreenerInfo {...SCREENER_INFO.graham} />
        </div>
      </div>

      {/* Mode Filter */}
      <div className="mb-4">
        <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
          Mode
        </label>
        <select
          value={mode}
          onChange={(e) => setMode(e.target.value as any)}
          className="rounded-md border border-gray-300 dark:border-gray-700 bg-white dark:bg-gray-900 px-3 py-2 pr-8 text-sm"
        >
          <option value="strict">Strict (All 8 Criteria)</option>
          <option value="simplified">Simplified (P/E & P/B only)</option>
          <option value="modern">Modern (Relaxed P/E & P/B)</option>
          <option value="garp">GARP (Growth Focus)</option>
          <option value="relaxed">Custom Selection</option>
        </select>
        <p className="mt-1 text-xs text-gray-500">{MODE_DESCRIPTIONS[mode]}</p>
      </div>

      {/* Criteria Filter Checkboxes */}
      <div className="mb-6">
        <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
          Required Criteria
        </label>
        <div className="flex flex-wrap items-center gap-3">
          {CRITERIA.map((criteria) => (
            <label
              key={criteria.key}
              className={`flex items-center gap-2 px-3 py-2 rounded-lg border cursor-pointer transition-colors ${
                selectedCriteria.has(criteria.key)
                  ? "border-indigo-500 bg-indigo-50 dark:bg-indigo-950"
                  : "border-gray-200 dark:border-gray-800 hover:border-gray-300"
              }`}
            >
              <input
                type="checkbox"
                checked={selectedCriteria.has(criteria.key)}
                onChange={() => toggleCriteria(criteria.key)}
                className="w-4 h-4 rounded border-gray-300 text-indigo-600 focus:ring-indigo-500"
              />
              <span className="text-sm font-medium text-gray-700 dark:text-gray-300">
                {criteria.label}
              </span>
            </label>
          ))}
          <button
            onClick={toggleAll}
            className="px-3 py-2 text-sm text-indigo-600 hover:text-indigo-500 font-medium"
          >
            {allSelected ? "Unselect all" : "Select all"}
          </button>
        </div>
        <p className="mt-2 text-sm text-gray-500">
          {selectedCriteria.size === 0
            ? `Showing all ${filteredStocks.length} stocks`
            : `Showing ${filteredStocks.length} stocks passing ${selectedCriteria.size} selected criteria`}
        </p>
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
                <th className="px-4 py-3 text-center text-xs font-medium text-gray-500 uppercase">Score</th>
                <th className="px-4 py-3 text-center text-xs font-medium text-gray-500 uppercase" title="Adequate Size">Size</th>
                <th className="px-4 py-3 text-center text-xs font-medium text-gray-500 uppercase" title="Current Ratio">CR</th>
                <th className="px-4 py-3 text-center text-xs font-medium text-gray-500 uppercase" title="Debt Coverage">Debt</th>
                <th className="px-4 py-3 text-center text-xs font-medium text-gray-500 uppercase" title="Earnings Stability">Earn</th>
                <th className="px-4 py-3 text-center text-xs font-medium text-gray-500 uppercase" title="Dividend Record">Div</th>
                <th className="px-4 py-3 text-center text-xs font-medium text-gray-500 uppercase" title="Earnings Growth">Grow</th>
                <th className="px-4 py-3 text-center text-xs font-medium text-gray-500 uppercase" title="P/E Ratio">P/E</th>
                <th className="px-4 py-3 text-center text-xs font-medium text-gray-500 uppercase" title="P/B Ratio">P/B</th>
                <th className="px-4 py-3 text-right text-xs font-medium text-gray-500 uppercase">P/E Val</th>
                <th className="px-4 py-3 text-right text-xs font-medium text-gray-500 uppercase">P/B Val</th>
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
                    <Badge variant={stock.criteria_passed >= 5 ? "success" : stock.criteria_passed >= 3 ? "warning" : "default"}>
                      {stock.criteria_passed}/8
                    </Badge>
                  </td>
                  <td className="px-4 py-3 text-center"><CriteriaBadge pass={stock.adequate_size} /></td>
                  <td className="px-4 py-3 text-center"><CriteriaBadge pass={stock.current_ratio_pass} /></td>
                  <td className="px-4 py-3 text-center"><CriteriaBadge pass={stock.debt_coverage_pass} /></td>
                  <td className="px-4 py-3 text-center"><CriteriaBadge pass={stock.earnings_stability} /></td>
                  <td className="px-4 py-3 text-center"><CriteriaBadge pass={stock.dividend_record} /></td>
                  <td className="px-4 py-3 text-center"><CriteriaBadge pass={stock.earnings_growth_pass} /></td>
                  <td className="px-4 py-3 text-center"><CriteriaBadge pass={stock.pe_ratio_pass} /></td>
                  <td className="px-4 py-3 text-center"><CriteriaBadge pass={stock.pb_ratio_pass} /></td>
                  <td className="px-4 py-3 text-right text-sm text-gray-500">{formatNumber(stock.pe_ratio, 1)}</td>
                  <td className="px-4 py-3 text-right text-sm text-gray-500">{formatNumber(stock.pb_ratio, 2)}</td>
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
          No stocks found matching the selected criteria.
        </div>
      )}
    </div>
  )
}
