"use client"

import { useEffect, useState, useMemo } from "react"
import Link from "next/link"
import { Badge } from "@/components/Badge"
import { Pagination } from "@/components/Pagination"
import { ScreenerInfo, SCREENER_INFO } from "@/components/ScreenerInfo"
import { getFamaFrenchStocks, FamaFrenchStock, formatPercent } from "@/lib/api"

const ITEMS_PER_PAGE = 50

export default function FamaFrenchPage() {
  const [stocks, setStocks] = useState<FamaFrenchStock[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [minProfitability, setMinProfitability] = useState(0)
  const [minBookToMarket, setMinBookToMarket] = useState(0)
  const [maxAssetGrowth, setMaxAssetGrowth] = useState(1)
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
  }, [minProfitability, minBookToMarket, maxAssetGrowth])

  useEffect(() => {
    async function fetchData() {
      try {
        setLoading(true)
        const data = await getFamaFrenchStocks(minProfitability, minBookToMarket, maxAssetGrowth, 500)
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
  }, [minProfitability, minBookToMarket, maxAssetGrowth])

  return (
    <div className="p-4 sm:p-6 lg:p-8">
      <div className="mb-8">
        <div className="flex items-start justify-between gap-4">
          <div>
            <h1 className="text-2xl font-bold text-gray-900 dark:text-gray-50">
              Fama-French Factors
            </h1>
            <p className="mt-1 text-sm text-gray-500">
              Factor exposure analysis: Book-to-Market (HML), Profitability (RMW), Investment (CMA)
            </p>
          </div>
          <ScreenerInfo {...SCREENER_INFO.famaFrench} />
        </div>
      </div>

      <div className="mb-6 p-4 rounded-lg bg-blue-50 dark:bg-blue-950 border border-blue-200 dark:border-blue-800">
        <p className="text-sm text-blue-800 dark:text-blue-200">
          High profitability (RMW) and high book-to-market (HML) factors have historically predicted higher returns.
          Low asset growth (CMA) indicates conservative investment, also associated with higher returns.
        </p>
      </div>

      {/* Filters */}
      <div className="flex flex-wrap gap-4 mb-6">
        <div>
          <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
            Min Profitability %ile
          </label>
          <select
            value={minProfitability}
            onChange={(e) => setMinProfitability(parseFloat(e.target.value))}
            className="rounded-md border border-gray-300 dark:border-gray-700 bg-white dark:bg-gray-900 px-3 py-2 pr-8 text-sm"
          >
            <option value={0}>Any</option>
            <option value={0.5}>Top 50%</option>
            <option value={0.7}>Top 30%</option>
            <option value={0.8}>Top 20%</option>
            <option value={0.9}>Top 10%</option>
          </select>
        </div>
        <div>
          <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
            Min Book/Market %ile
          </label>
          <select
            value={minBookToMarket}
            onChange={(e) => setMinBookToMarket(parseFloat(e.target.value))}
            className="rounded-md border border-gray-300 dark:border-gray-700 bg-white dark:bg-gray-900 px-3 py-2 pr-8 text-sm"
          >
            <option value={0}>Any</option>
            <option value={0.5}>Top 50% (Value)</option>
            <option value={0.7}>Top 30% (Deep Value)</option>
            <option value={0.8}>Top 20%</option>
          </select>
        </div>
        <div>
          <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
            Max Asset Growth %ile
          </label>
          <select
            value={maxAssetGrowth}
            onChange={(e) => setMaxAssetGrowth(parseFloat(e.target.value))}
            className="rounded-md border border-gray-300 dark:border-gray-700 bg-white dark:bg-gray-900 px-3 py-2 pr-8 text-sm"
          >
            <option value={1}>Any</option>
            <option value={0.5}>Bottom 50% (Conservative)</option>
            <option value={0.3}>Bottom 30%</option>
            <option value={0.2}>Bottom 20%</option>
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
                <th className="px-4 py-3 text-right text-xs font-medium text-gray-500 uppercase" title="Profitability (RMW)">Profit %ile</th>
                <th className="px-4 py-3 text-right text-xs font-medium text-gray-500 uppercase" title="Book-to-Market (HML)">B/M %ile</th>
                <th className="px-4 py-3 text-right text-xs font-medium text-gray-500 uppercase" title="Asset Growth (CMA)">Growth %ile</th>
                <th className="px-4 py-3 text-right text-xs font-medium text-gray-500 uppercase">Profitability</th>
                <th className="px-4 py-3 text-right text-xs font-medium text-gray-500 uppercase">B/M Ratio</th>
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
                    {stock.profitability_percentile !== null ? (
                      <Badge variant={stock.profitability_percentile >= 0.7 ? "success" : stock.profitability_percentile >= 0.3 ? "warning" : "default"}>
                        {(stock.profitability_percentile * 100).toFixed(0)}%
                      </Badge>
                    ) : (
                      <span className="text-gray-400">—</span>
                    )}
                  </td>
                  <td className="px-4 py-3 text-right">
                    {stock.book_to_market_percentile !== null ? (
                      <Badge variant={stock.book_to_market_percentile >= 0.7 ? "success" : stock.book_to_market_percentile >= 0.3 ? "warning" : "default"}>
                        {(stock.book_to_market_percentile * 100).toFixed(0)}%
                      </Badge>
                    ) : (
                      <span className="text-gray-400">—</span>
                    )}
                  </td>
                  <td className="px-4 py-3 text-right">
                    {stock.asset_growth_percentile !== null ? (
                      <Badge variant={stock.asset_growth_percentile <= 0.3 ? "success" : stock.asset_growth_percentile <= 0.5 ? "warning" : "default"}>
                        {(stock.asset_growth_percentile * 100).toFixed(0)}%
                      </Badge>
                    ) : (
                      <span className="text-gray-400">—</span>
                    )}
                  </td>
                  <td className="px-4 py-3 text-right text-sm text-gray-500">
                    {formatPercent(stock.profitability)}
                  </td>
                  <td className="px-4 py-3 text-right text-sm text-gray-500">
                    {stock.book_to_market !== null ? stock.book_to_market.toFixed(2) : "—"}
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

      {!loading && !error && stocks.length === 0 && (
        <div className="text-center py-12 text-gray-500">
          No stocks found matching the selected criteria.
        </div>
      )}
    </div>
  )
}
