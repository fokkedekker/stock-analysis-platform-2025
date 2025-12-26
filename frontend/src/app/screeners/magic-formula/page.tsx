"use client"

import { useEffect, useState, useMemo } from "react"
import Link from "next/link"
import { Badge } from "@/components/Badge"
import { Pagination } from "@/components/Pagination"
import { useQuarter } from "@/contexts/QuarterContext"
import { getMagicFormulaStocks, MagicFormulaStock, formatPercent } from "@/lib/api"
import { ScreenerInfo, SCREENER_INFO } from "@/components/ScreenerInfo"

const ITEMS_PER_PAGE = 50

export default function MagicFormulaPage() {
  const { quarter } = useQuarter()
  const [stocks, setStocks] = useState<MagicFormulaStock[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [top, setTop] = useState(100)
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
  }, [top])

  useEffect(() => {
    async function fetchData() {
      try {
        setLoading(true)
        const data = await getMagicFormulaStocks(top, quarter)
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
  }, [top, quarter])

  return (
    <div className="p-4 sm:p-6 lg:p-8">
      <div className="mb-8">
        <div className="flex items-start justify-between gap-4">
          <div>
            <h1 className="text-2xl font-bold text-gray-900 dark:text-gray-50">
              Magic Formula (Joel Greenblatt)
            </h1>
            <p className="mt-1 text-sm text-gray-500">
              Ranks stocks by combined Earnings Yield and Return on Capital
            </p>
          </div>
          <ScreenerInfo {...SCREENER_INFO.magicFormula} />
        </div>
      </div>

      <div className="flex gap-4 mb-6">
        <div>
          <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
            Show Top
          </label>
          <select
            value={top}
            onChange={(e) => setTop(parseInt(e.target.value))}
            className="rounded-md border border-gray-300 dark:border-gray-700 bg-white dark:bg-gray-900 px-3 py-2 pr-8 text-sm"
          >
            <option value={50}>50</option>
            <option value={100}>100</option>
            <option value={200}>200</option>
            <option value={500}>500</option>
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
                <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">Rank</th>
                <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">Symbol</th>
                <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">Name</th>
                <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">Sector</th>
                <th className="px-4 py-3 text-right text-xs font-medium text-gray-500 uppercase">Earnings Yield</th>
                <th className="px-4 py-3 text-right text-xs font-medium text-gray-500 uppercase">EY Rank</th>
                <th className="px-4 py-3 text-right text-xs font-medium text-gray-500 uppercase">Return on Capital</th>
                <th className="px-4 py-3 text-right text-xs font-medium text-gray-500 uppercase">ROC Rank</th>
              </tr>
            </thead>
            <tbody className="bg-white divide-y divide-gray-200 dark:bg-gray-950 dark:divide-gray-800">
              {paginatedStocks.map((stock, index) => (
                <tr key={stock.symbol} className="hover:bg-gray-50 dark:hover:bg-gray-900">
                  <td className="px-4 py-3 whitespace-nowrap text-right text-sm text-gray-500">{(currentPage - 1) * ITEMS_PER_PAGE + index + 1}</td>
                  <td className="px-4 py-3 whitespace-nowrap">
                    <Badge variant={stock.combined_rank <= 30 ? "success" : stock.combined_rank <= 100 ? "warning" : "default"}>
                      #{stock.combined_rank}
                    </Badge>
                  </td>
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
                  <td className="px-4 py-3 text-right text-sm text-gray-900 dark:text-gray-100">
                    {formatPercent(stock.earnings_yield)}
                  </td>
                  <td className="px-4 py-3 text-right text-sm text-gray-500">
                    {stock.earnings_yield_rank || "—"}
                  </td>
                  <td className="px-4 py-3 text-right text-sm text-gray-900 dark:text-gray-100">
                    {formatPercent(stock.return_on_capital)}
                  </td>
                  <td className="px-4 py-3 text-right text-sm text-gray-500">
                    {stock.return_on_capital_rank || "—"}
                  </td>
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
