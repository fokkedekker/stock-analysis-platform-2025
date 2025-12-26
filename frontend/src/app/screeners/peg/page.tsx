"use client"

import { useEffect, useState, useMemo } from "react"
import Link from "next/link"
import { Pagination } from "@/components/Pagination"
import { useQuarter } from "@/contexts/QuarterContext"
import { getPegStocks, PegStock, formatNumber, formatPercent } from "@/lib/api"
import { RiCheckLine, RiCloseLine } from "@remixicon/react"
import { ScreenerInfo, SCREENER_INFO } from "@/components/ScreenerInfo"

const ITEMS_PER_PAGE = 50

function PassBadge({ pass }: { pass: boolean | null }) {
  if (pass === true) {
    return <RiCheckLine className="w-4 h-4 text-green-600" />
  }
  if (pass === false) {
    return <RiCloseLine className="w-4 h-4 text-red-500" />
  }
  return <span className="text-gray-400">—</span>
}

export default function PegPage() {
  const { quarter } = useQuarter()
  const [stocks, setStocks] = useState<PegStock[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [maxPeg, setMaxPeg] = useState(10)
  const [minGrowth, setMinGrowth] = useState(0)
  const [currentPage, setCurrentPage] = useState(1)

  // Paginate stocks
  const totalPages = Math.ceil(stocks.length / ITEMS_PER_PAGE)
  const paginatedStocks = useMemo(() => {
    const start = (currentPage - 1) * ITEMS_PER_PAGE
    return stocks.slice(start, start + ITEMS_PER_PAGE)
  }, [stocks, currentPage])

  // Reset to page 1 when filters change
  useEffect(() => {
    setCurrentPage(1)
  }, [maxPeg, minGrowth])

  useEffect(() => {
    async function fetchData() {
      try {
        setLoading(true)
        const data = await getPegStocks(maxPeg, minGrowth, 200, quarter)
        setStocks(data.stocks)
        setError(null)
      } catch (err) {
        setError("Failed to fetch data. Is the API server running?")
        console.error(err)
      } finally {
        setLoading(false)
      }
    }
    fetchData()
  }, [maxPeg, minGrowth, quarter])

  return (
    <div className="p-4 sm:p-6 lg:p-8">
      <div className="mb-8">
        <div className="flex items-start justify-between gap-4">
          <div>
            <h1 className="text-2xl font-bold text-gray-900 dark:text-gray-50">
              GARP / PEG Ratio
            </h1>
            <p className="mt-1 text-sm text-gray-500">
              Growth at a Reasonable Price - stocks with low PEG ratios and strong earnings growth
            </p>
          </div>
          <ScreenerInfo {...SCREENER_INFO.peg} />
        </div>
      </div>

      <div className="flex flex-wrap gap-4 mb-6">
        <div>
          <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
            Max PEG
          </label>
          <select
            value={maxPeg}
            onChange={(e) => setMaxPeg(parseFloat(e.target.value))}
            className="rounded-md border border-gray-300 dark:border-gray-700 bg-white dark:bg-gray-900 px-3 py-2 pr-8 text-sm"
          >
            <option value={0.5}>0.5</option>
            <option value={1}>1.0</option>
            <option value={1.5}>1.5</option>
            <option value={2}>2.0</option>
            <option value={10}>Any</option>
          </select>
        </div>
        <div>
          <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
            Min Growth
          </label>
          <select
            value={minGrowth}
            onChange={(e) => setMinGrowth(parseFloat(e.target.value))}
            className="rounded-md border border-gray-300 dark:border-gray-700 bg-white dark:bg-gray-900 px-3 py-2 pr-8 text-sm"
          >
            <option value={0}>Any</option>
            <option value={0.1}>10%+</option>
            <option value={0.15}>15%+</option>
            <option value={0.2}>20%+</option>
            <option value={0.25}>25%+</option>
          </select>
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
                <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">Sector</th>
                <th className="px-4 py-3 text-right text-xs font-medium text-gray-500 uppercase">PEG Ratio</th>
                <th className="px-4 py-3 text-center text-xs font-medium text-gray-500 uppercase">PEG Pass</th>
                <th className="px-4 py-3 text-right text-xs font-medium text-gray-500 uppercase">P/E</th>
                <th className="px-4 py-3 text-right text-xs font-medium text-gray-500 uppercase">EPS CAGR</th>
                <th className="px-4 py-3 text-center text-xs font-medium text-gray-500 uppercase">Growth Pass</th>
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
                  <td className="px-4 py-3 whitespace-nowrap text-sm text-gray-500">
                    {stock.sector || "—"}
                  </td>
                  <td className="px-4 py-3 text-right">
                    <span className={`text-sm font-medium ${stock.peg_ratio <= 1 ? 'text-green-600' : stock.peg_ratio <= 1.5 ? 'text-yellow-600' : 'text-gray-900 dark:text-gray-100'}`}>
                      {formatNumber(stock.peg_ratio, 2)}
                    </span>
                  </td>
                  <td className="px-4 py-3 text-center"><PassBadge pass={stock.peg_pass} /></td>
                  <td className="px-4 py-3 text-right text-sm text-gray-500">
                    {formatNumber(stock.pe_ratio, 1)}
                  </td>
                  <td className="px-4 py-3 text-right text-sm text-gray-500">
                    {formatPercent(stock.eps_cagr)}
                  </td>
                  <td className="px-4 py-3 text-center"><PassBadge pass={stock.growth_pass} /></td>
                </tr>
              ))}
            </tbody>
          </table>
          <Pagination
            currentPage={currentPage}
            totalPages={totalPages}
            onPageChange={setCurrentPage}
            totalItems={stocks.length}
            itemsPerPage={ITEMS_PER_PAGE}
          />
        </div>
      )}
    </div>
  )
}
