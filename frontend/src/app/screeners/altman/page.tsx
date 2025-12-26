"use client"

import { useEffect, useState, useMemo } from "react"
import Link from "next/link"
import { Badge } from "@/components/Badge"
import { Pagination } from "@/components/Pagination"
import { getAltmanStocks, AltmanStock } from "@/lib/api"
import { ScreenerInfo, SCREENER_INFO } from "@/components/ScreenerInfo"

const ITEMS_PER_PAGE = 50

const formatNumber = (val: number | null, decimals = 2) => {
  if (val === null || val === undefined) return 'N/A';
  return val.toFixed(decimals);
}

function ZoneBadge({ zone }: { zone: string | null }) {
  if (zone === "safe") {
    return <Badge variant="success">Safe</Badge>
  }
  if (zone === "grey") {
    return <Badge variant="warning">Grey</Badge>
  }
  if (zone === "distress") {
    return <Badge variant="error">Distress</Badge>
  }
  return <Badge variant="default">N/A</Badge>
}

export default function AltmanPage() {
  const [stocks, setStocks] = useState<AltmanStock[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [zone, setZone] = useState<"safe" | "grey" | "distress">("safe")
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
  }, [zone])

  useEffect(() => {
    async function fetchData() {
      try {
        setLoading(true)
        const data = await getAltmanStocks(zone, 200)
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
  }, [zone])

  return (
    <div className="p-4 sm:p-6 lg:p-8">
      <div className="mb-8">
        <div className="flex items-start justify-between gap-4">
          <div>
            <h1 className="text-2xl font-bold text-gray-900 dark:text-gray-50">
              Altman Z-Score
            </h1>
            <p className="mt-1 text-sm text-gray-500">
              Bankruptcy risk indicator: Safe (&gt;2.99), Grey (1.81-2.99), Distress (&lt;1.81)
            </p>
          </div>
          <ScreenerInfo {...SCREENER_INFO.altman} />
        </div>
      </div>

      <div className="flex gap-4 mb-6">
        <div>
          <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
            Zone
          </label>
          <select
            value={zone}
            onChange={(e) => setZone(e.target.value as any)}
            className="rounded-md border border-gray-300 dark:border-gray-700 bg-white dark:bg-gray-900 px-3 py-2 pr-8 text-sm"
          >
            <option value="safe">Safe Zone (Z &gt; 2.99)</option>
            <option value="grey">Grey Zone (1.81 - 2.99)</option>
            <option value="distress">Distress Zone (Z &lt; 1.81)</option>
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
                <th className="px-4 py-3 text-center text-xs font-medium text-gray-500 uppercase">Zone</th>
                <th className="px-4 py-3 text-right text-xs font-medium text-gray-500 uppercase">Z-Score</th>
                <th className="px-4 py-3 text-right text-xs font-medium text-gray-500 uppercase" title="Working Capital / Total Assets">X1 (WC/TA)</th>
                <th className="px-4 py-3 text-right text-xs font-medium text-gray-500 uppercase" title="Retained Earnings / Total Assets">X2 (RE/TA)</th>
                <th className="px-4 py-3 text-right text-xs font-medium text-gray-500 uppercase" title="EBIT / Total Assets">X3 (EBIT/TA)</th>
                <th className="px-4 py-3 text-right text-xs font-medium text-gray-500 uppercase" title="Market Cap / Total Liabilities">X4 (MC/TL)</th>
                <th className="px-4 py-3 text-right text-xs font-medium text-gray-500 uppercase" title="Revenue / Total Assets">X5 (Rev/TA)</th>
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
                  <td className="px-4 py-3 text-center">
                    <ZoneBadge zone={stock.zone} />
                  </td>
                  <td className="px-4 py-3 text-right text-sm font-medium text-gray-900 dark:text-gray-100">
                    {formatNumber(stock.z_score, 2)}
                  </td>
                  <td className="px-4 py-3 text-right text-sm text-gray-500">{formatNumber(stock.x1_wc_ta, 3)}</td>
                  <td className="px-4 py-3 text-right text-sm text-gray-500">{formatNumber(stock.x2_re_ta, 3)}</td>
                  <td className="px-4 py-3 text-right text-sm text-gray-500">{formatNumber(stock.x3_ebit_ta, 3)}</td>
                  <td className="px-4 py-3 text-right text-sm text-gray-500">{formatNumber(stock.x4_mc_tl, 3)}</td>
                  <td className="px-4 py-3 text-right text-sm text-gray-500">{formatNumber(stock.x5_rev_ta, 3)}</td>
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
