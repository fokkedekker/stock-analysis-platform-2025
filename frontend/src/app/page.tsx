"use client"

import { useEffect, useState, useMemo } from "react"
import Link from "next/link"
import { Badge } from "@/components/Badge"
import { Pagination } from "@/components/Pagination"
import { getRankings, RankingStock } from "@/lib/api"
import { RiCheckLine, RiCloseLine } from "@remixicon/react"

const ITEMS_PER_PAGE = 50

const SYSTEMS = [
  { key: "graham_pass", label: "Graham", abbr: "GR" },
  { key: "magic_formula_pass", label: "Magic Formula", abbr: "MF" },
  { key: "piotroski_pass", label: "Piotroski", abbr: "PI" },
  { key: "altman_pass", label: "Altman", abbr: "AL" },
  { key: "roic_pass", label: "ROIC", abbr: "RC" },
  { key: "peg_pass", label: "PEG", abbr: "PG" },
  { key: "fama_french_pass", label: "Fama-French", abbr: "FF" },
  { key: "net_net_pass", label: "Net-Net", abbr: "NN" },
] as const

type SystemKey = typeof SYSTEMS[number]["key"]

function PassBadge({ pass }: { pass: number }) {
  if (pass === 1) {
    return (
      <span className="inline-flex items-center justify-center w-5 h-5 rounded-full bg-green-100 dark:bg-green-900">
        <RiCheckLine className="w-3 h-3 text-green-600 dark:text-green-400" />
      </span>
    )
  }
  return (
    <span className="inline-flex items-center justify-center w-5 h-5 rounded-full bg-gray-100 dark:bg-gray-800">
      <RiCloseLine className="w-3 h-3 text-gray-400" />
    </span>
  )
}

export default function RankingsPage() {
  const [allStocks, setAllStocks] = useState<RankingStock[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [selectedSystems, setSelectedSystems] = useState<Set<SystemKey>>(new Set())
  const [currentPage, setCurrentPage] = useState(1)

  useEffect(() => {
    async function fetchData() {
      try {
        setLoading(true)
        // Fetch all stocks (min_systems=1 to get everything with at least 1 pass)
        const data = await getRankings(1, 500)
        setAllStocks(data.stocks)
        setError(null)
      } catch (err) {
        setError("Failed to fetch rankings. Is the API server running?")
        console.error(err)
      } finally {
        setLoading(false)
      }
    }
    fetchData()
  }, [])

  // Filter stocks based on selected systems
  const filteredStocks = useMemo(() => {
    if (selectedSystems.size === 0) return allStocks
    const systemKeys = Array.from(selectedSystems)
    return allStocks.filter((stock) => {
      for (const systemKey of systemKeys) {
        if (stock[systemKey] !== 1) return false
      }
      return true
    })
  }, [allStocks, selectedSystems])

  // Paginate filtered stocks
  const totalPages = Math.ceil(filteredStocks.length / ITEMS_PER_PAGE)
  const paginatedStocks = useMemo(() => {
    const start = (currentPage - 1) * ITEMS_PER_PAGE
    return filteredStocks.slice(start, start + ITEMS_PER_PAGE)
  }, [filteredStocks, currentPage])

  // Reset to page 1 when filters change
  useEffect(() => {
    setCurrentPage(1)
  }, [selectedSystems])

  const toggleSystem = (key: SystemKey) => {
    setSelectedSystems((prev) => {
      const next = new Set(prev)
      if (next.has(key)) {
        next.delete(key)
      } else {
        next.add(key)
      }
      return next
    })
  }

  const allSystemKeys = SYSTEMS.map(s => s.key)
  const allSelected = selectedSystems.size === SYSTEMS.length

  const toggleAll = () => {
    if (allSelected) {
      setSelectedSystems(new Set())
    } else {
      setSelectedSystems(new Set(allSystemKeys))
    }
  }

  return (
    <div className="p-4 sm:p-6 lg:p-8">
      <div className="mb-8">
        <h1 className="text-2xl font-bold text-gray-900 dark:text-gray-50">
          Multi-System Rankings
        </h1>
        <p className="mt-1 text-sm text-gray-500">
          Select systems to filter stocks that pass all selected criteria
        </p>
      </div>

      {/* System Filter Checkboxes */}
      <div className="mb-6">
        <div className="flex flex-wrap items-center gap-3">
          {SYSTEMS.map((system) => (
            <label
              key={system.key}
              className={`flex items-center gap-2 px-3 py-2 rounded-lg border cursor-pointer transition-colors ${
                selectedSystems.has(system.key)
                  ? "border-indigo-500 bg-indigo-50 dark:bg-indigo-950"
                  : "border-gray-200 dark:border-gray-800 hover:border-gray-300"
              }`}
            >
              <input
                type="checkbox"
                checked={selectedSystems.has(system.key)}
                onChange={() => toggleSystem(system.key)}
                className="w-4 h-4 rounded border-gray-300 text-indigo-600 focus:ring-indigo-500"
              />
              <span className="text-sm font-medium text-gray-700 dark:text-gray-300">
                {system.label}
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
          {selectedSystems.size === 0
            ? `Showing all ${filteredStocks.length} stocks`
            : `Showing ${filteredStocks.length} stocks passing ${selectedSystems.size} selected system${selectedSystems.size > 1 ? "s" : ""}`}
        </p>
      </div>

      {/* Error State */}
      {error && (
        <div className="rounded-lg border border-red-200 bg-red-50 p-4 dark:border-red-900 dark:bg-red-950">
          <p className="text-sm text-red-600 dark:text-red-400">{error}</p>
        </div>
      )}

      {/* Loading State */}
      {loading && (
        <div className="flex items-center justify-center py-12">
          <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-indigo-600"></div>
        </div>
      )}

      {/* Table */}
      {!loading && !error && (
        <div className="overflow-x-auto rounded-lg border border-gray-200 dark:border-gray-800">
          <table className="min-w-full divide-y divide-gray-200 dark:divide-gray-800">
            <thead className="bg-gray-50 dark:bg-gray-900">
              <tr>
                <th className="px-4 py-3 text-right text-xs font-medium text-gray-500 uppercase tracking-wider w-12">
                  #
                </th>
                <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Symbol
                </th>
                <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Name
                </th>
                <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Sector
                </th>
                <th className="px-4 py-3 text-center text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Systems
                </th>
                <th className="px-4 py-3 text-center text-xs font-medium text-gray-500 uppercase tracking-wider" title="Graham">
                  GR
                </th>
                <th className="px-4 py-3 text-center text-xs font-medium text-gray-500 uppercase tracking-wider" title="Magic Formula">
                  MF
                </th>
                <th className="px-4 py-3 text-center text-xs font-medium text-gray-500 uppercase tracking-wider" title="Piotroski">
                  PI
                </th>
                <th className="px-4 py-3 text-center text-xs font-medium text-gray-500 uppercase tracking-wider" title="Altman">
                  AL
                </th>
                <th className="px-4 py-3 text-center text-xs font-medium text-gray-500 uppercase tracking-wider" title="ROIC">
                  RC
                </th>
                <th className="px-4 py-3 text-center text-xs font-medium text-gray-500 uppercase tracking-wider" title="PEG">
                  PG
                </th>
                <th className="px-4 py-3 text-center text-xs font-medium text-gray-500 uppercase tracking-wider" title="Fama-French">
                  FF
                </th>
                <th className="px-4 py-3 text-center text-xs font-medium text-gray-500 uppercase tracking-wider" title="Net-Net">
                  NN
                </th>
              </tr>
            </thead>
            <tbody className="bg-white divide-y divide-gray-200 dark:bg-gray-950 dark:divide-gray-800">
              {paginatedStocks.map((stock, index) => (
                <tr
                  key={stock.symbol}
                  className="hover:bg-gray-50 dark:hover:bg-gray-900 cursor-pointer"
                >
                  <td className="px-4 py-3 whitespace-nowrap text-right text-sm text-gray-500">
                    {(currentPage - 1) * ITEMS_PER_PAGE + index + 1}
                  </td>
                  <td className="px-4 py-3 whitespace-nowrap">
                    <Link
                      href={`/stock/${stock.symbol}`}
                      className="text-sm font-medium text-indigo-600 hover:text-indigo-500 dark:text-indigo-400"
                    >
                      {stock.symbol}
                    </Link>
                  </td>
                  <td className="px-4 py-3 whitespace-nowrap text-sm text-gray-900 dark:text-gray-100 max-w-[200px] truncate">
                    {stock.name}
                  </td>
                  <td className="px-4 py-3 whitespace-nowrap text-sm text-gray-500">
                    {stock.sector || "—"}
                  </td>
                  <td className="px-4 py-3 whitespace-nowrap text-center">
                    <Badge
                      variant={
                        stock.systems_passed >= 5
                          ? "success"
                          : stock.systems_passed >= 3
                          ? "warning"
                          : "default"
                      }
                    >
                      {stock.systems_passed}/8
                    </Badge>
                  </td>
                  <td className="px-4 py-3 text-center">
                    <PassBadge pass={stock.graham_pass} />
                  </td>
                  <td className="px-4 py-3 text-center">
                    <PassBadge pass={stock.magic_formula_pass} />
                  </td>
                  <td className="px-4 py-3 text-center">
                    <PassBadge pass={stock.piotroski_pass} />
                  </td>
                  <td className="px-4 py-3 text-center">
                    <PassBadge pass={stock.altman_pass} />
                  </td>
                  <td className="px-4 py-3 text-center">
                    <PassBadge pass={stock.roic_pass} />
                  </td>
                  <td className="px-4 py-3 text-center">
                    <PassBadge pass={stock.peg_pass} />
                  </td>
                  <td className="px-4 py-3 text-center">
                    <PassBadge pass={stock.fama_french_pass} />
                  </td>
                  <td className="px-4 py-3 text-center">
                    <PassBadge pass={stock.net_net_pass} />
                  </td>
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
          No stocks found matching the selected systems.
        </div>
      )}
    </div>
  )
}
