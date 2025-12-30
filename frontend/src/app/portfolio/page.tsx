"use client"

import { useEffect, useState, useCallback } from "react"
import Link from "next/link"
import {
  RiArrowUpLine,
  RiArrowDownLine,
  RiBriefcaseLine,
  RiDeleteBinLine,
  RiExpandUpDownLine,
  RiMoneyDollarCircleLine,
  RiPercentLine,
  RiAlertLine,
  RiCheckLine,
} from "@remixicon/react"
import {
  getDashboard,
  deleteBatch,
  deletePosition,
  sellBatch,
  sellPosition,
  Dashboard,
  Batch,
  Position,
} from "@/lib/api"
import { Button } from "@/components/Button"

function formatCurrency(value: number | null | undefined): string {
  if (value === null || value === undefined) return "N/A"
  return `$${value.toLocaleString(undefined, { minimumFractionDigits: 0, maximumFractionDigits: 0 })}`
}

function formatReturnPct(value: number | null | undefined): string {
  if (value === null || value === undefined) return "N/A"
  const sign = value >= 0 ? "+" : ""
  return `${sign}${value.toFixed(2)}%`
}

function formatQuarter(q: string): string {
  if (q.length === 6 && q.includes("Q")) {
    const year = q.slice(0, 4)
    const qNum = q.slice(4)
    return `${qNum} ${year}`
  }
  return q
}

function ReturnBadge({ value, size = "sm" }: { value: number | null | undefined; size?: "sm" | "lg" }) {
  if (value === null || value === undefined) return <span className="text-gray-500">N/A</span>
  const isPositive = value >= 0
  const sizeClasses = size === "lg" ? "px-3 py-1 text-base" : "px-2 py-0.5 text-sm"
  return (
    <span
      className={`inline-flex items-center rounded font-semibold ${sizeClasses} ${
        isPositive
          ? "bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200"
          : "bg-red-100 text-red-800 dark:bg-red-900 dark:text-red-200"
      }`}
    >
      {isPositive ? <RiArrowUpLine className="size-4 mr-1" /> : <RiArrowDownLine className="size-4 mr-1" />}
      {formatReturnPct(value)}
    </span>
  )
}

function SummaryCard({
  title,
  value,
  subtitle,
  icon,
  variant = "default",
}: {
  title: string
  value: string
  subtitle?: string
  icon: React.ReactNode
  variant?: "default" | "positive" | "negative"
}) {
  const bgColor =
    variant === "positive"
      ? "bg-green-50 dark:bg-green-950 border-green-200 dark:border-green-800"
      : variant === "negative"
      ? "bg-red-50 dark:bg-red-950 border-red-200 dark:border-red-800"
      : "bg-white dark:bg-gray-950 border-gray-200 dark:border-gray-800"

  const textColor =
    variant === "positive"
      ? "text-green-700 dark:text-green-300"
      : variant === "negative"
      ? "text-red-700 dark:text-red-300"
      : "text-gray-900 dark:text-gray-100"

  return (
    <div className={`rounded-lg border p-4 ${bgColor}`}>
      <div className="flex items-center gap-2 text-gray-500 dark:text-gray-400 mb-1">
        {icon}
        <span className="text-sm">{title}</span>
      </div>
      <div className={`text-2xl font-bold ${textColor}`}>{value}</div>
      {subtitle && <div className="text-xs text-gray-500 mt-1">{subtitle}</div>}
    </div>
  )
}

function AlertBanner({ alerts }: { alerts: Position[] }) {
  if (alerts.length === 0) return null

  return (
    <div className="rounded-lg border border-yellow-300 bg-yellow-50 dark:border-yellow-800 dark:bg-yellow-950 p-4">
      <div className="flex items-center gap-2 mb-3">
        <RiAlertLine className="size-5 text-yellow-600 dark:text-yellow-400" />
        <span className="font-semibold text-yellow-800 dark:text-yellow-200">
          Sell Alerts ({alerts.length})
        </span>
      </div>
      <div className="space-y-2">
        {alerts.map((pos) => (
          <div
            key={pos.id}
            className="flex items-center justify-between bg-white dark:bg-gray-900 rounded p-2"
          >
            <div className="flex items-center gap-3">
              <Link
                href={`/stock/${pos.symbol}`}
                className="font-medium text-indigo-600 hover:text-indigo-500 dark:text-indigo-400"
              >
                {pos.symbol}
              </Link>
              <span className="text-sm text-gray-600 dark:text-gray-400">
                Sell in {pos.days_until_sell} days ({formatQuarter(pos.target_sell_quarter)})
              </span>
            </div>
            <div className="flex items-center gap-2">
              <ReturnBadge value={pos.unrealized_return_pct} />
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}

function PositionRow({
  position,
  onDelete,
  onSell,
}: {
  position: Position
  onDelete: () => void
  onSell: () => void
}) {
  const isSold = position.status === "sold"

  return (
    <tr className="hover:bg-gray-50 dark:hover:bg-gray-900">
      <td className="px-4 py-3">
        <Link
          href={`/stock/${position.symbol}`}
          className="text-sm font-medium text-indigo-600 hover:text-indigo-500 dark:text-indigo-400"
        >
          {position.symbol}
        </Link>
        {position.name && (
          <div className="text-xs text-gray-500 truncate max-w-[150px]">{position.name}</div>
        )}
      </td>
      <td className="px-4 py-3 text-sm text-gray-600 dark:text-gray-400 text-right">
        {formatCurrency(position.invested_amount)}
      </td>
      <td className="px-4 py-3 text-sm text-right">
        {isSold ? (
          <span className="text-gray-500">Sold</span>
        ) : position.current_value ? (
          formatCurrency(position.current_value)
        ) : (
          "N/A"
        )}
      </td>
      <td className="px-4 py-3 text-right">
        <ReturnBadge value={isSold ? position.realized_return : position.unrealized_return_pct} />
      </td>
      <td className="px-4 py-3 text-right">
        <ReturnBadge value={isSold ? position.realized_alpha : position.unrealized_alpha_pct} />
      </td>
      <td className="px-4 py-3 text-sm text-gray-600 dark:text-gray-400 text-center">
        {isSold ? (
          <span className="text-green-600 dark:text-green-400">
            <RiCheckLine className="size-4 inline" />
          </span>
        ) : (
          <span>{position.days_until_sell}d</span>
        )}
      </td>
      <td className="px-4 py-3 text-right">
        <div className="flex items-center justify-end gap-1">
          {!isSold && (
            <button
              onClick={onSell}
              className="p-1 text-green-600 hover:bg-green-100 dark:text-green-400 dark:hover:bg-green-900 rounded"
              title="Mark as Sold"
            >
              <RiCheckLine className="size-4" />
            </button>
          )}
          <button
            onClick={onDelete}
            className="p-1 text-red-600 hover:bg-red-100 dark:text-red-400 dark:hover:bg-red-900 rounded"
            title="Delete"
          >
            <RiDeleteBinLine className="size-4" />
          </button>
        </div>
      </td>
    </tr>
  )
}

function BatchCard({
  batch,
  onRefresh,
}: {
  batch: Batch
  onRefresh: () => void
}) {
  const [expanded, setExpanded] = useState(false)
  const [sellDialogOpen, setSellDialogOpen] = useState(false)
  const [sellQuarter, setSellQuarter] = useState(batch.target_sell_quarter)
  const [selling, setSelling] = useState(false)
  const [deleting, setDeleting] = useState(false)

  const handleDeleteBatch = async () => {
    if (!confirm(`Delete batch "${batch.name}" and all its positions?`)) return
    setDeleting(true)
    try {
      await deleteBatch(batch.id)
      onRefresh()
    } catch (e) {
      alert(e instanceof Error ? e.message : "Failed to delete batch")
    } finally {
      setDeleting(false)
    }
  }

  const handleSellBatch = async () => {
    setSelling(true)
    try {
      await sellBatch(batch.id, { sell_quarter: sellQuarter })
      setSellDialogOpen(false)
      onRefresh()
    } catch (e) {
      alert(e instanceof Error ? e.message : "Failed to sell batch")
    } finally {
      setSelling(false)
    }
  }

  const handleDeletePosition = async (positionId: string) => {
    if (!confirm("Delete this position?")) return
    try {
      await deletePosition(positionId)
      onRefresh()
    } catch (e) {
      alert(e instanceof Error ? e.message : "Failed to delete position")
    }
  }

  const handleSellPosition = async (positionId: string) => {
    const quarter = prompt("Enter sell quarter (e.g., 2024Q4):", batch.target_sell_quarter)
    if (!quarter) return
    try {
      await sellPosition(positionId, { sell_quarter: quarter })
      onRefresh()
    } catch (e) {
      alert(e instanceof Error ? e.message : "Failed to sell position")
    }
  }

  const isSold = batch.status === "sold"
  const isPartial = batch.status === "partial"

  return (
    <div className="rounded-lg border border-gray-200 dark:border-gray-800 overflow-hidden">
      {/* Header */}
      <div
        className={`p-4 cursor-pointer ${
          isSold
            ? "bg-gray-50 dark:bg-gray-900"
            : "bg-white dark:bg-gray-950 hover:bg-gray-50 dark:hover:bg-gray-900"
        }`}
        onClick={() => setExpanded(!expanded)}
      >
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <RiExpandUpDownLine
              className={`size-5 text-gray-400 transition-transform ${expanded ? "rotate-180" : ""}`}
            />
            <div>
              <div className="font-semibold text-gray-900 dark:text-gray-100">
                {batch.name || `${formatQuarter(batch.buy_quarter)} Batch`}
              </div>
              <div className="text-sm text-gray-500">
                {batch.position_count} positions
                {batch.strategy_name && ` | ${batch.strategy_name}`}
                {isPartial && " | Partially Sold"}
              </div>
            </div>
          </div>
          <div className="flex items-center gap-4">
            <div className="text-right">
              <div className="text-sm text-gray-500">Invested</div>
              <div className="font-semibold">{formatCurrency(batch.total_invested)}</div>
            </div>
            {!isSold && batch.current_value && (
              <div className="text-right">
                <div className="text-sm text-gray-500">Current</div>
                <div className="font-semibold">{formatCurrency(batch.current_value)}</div>
              </div>
            )}
            <div className="text-right">
              <div className="text-sm text-gray-500">{isSold ? "Return" : "Unrealized"}</div>
              <ReturnBadge value={isSold ? batch.realized_return : batch.unrealized_return_pct} />
            </div>
            <div className="text-right">
              <div className="text-sm text-gray-500">Alpha</div>
              <ReturnBadge value={isSold ? batch.realized_alpha : batch.unrealized_alpha_pct} />
            </div>
            {!isSold && (
              <div className="text-right">
                <div className="text-sm text-gray-500">Sell</div>
                <div className="text-sm font-medium">{formatQuarter(batch.target_sell_quarter)}</div>
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Expanded Content */}
      {expanded && (
        <div className="border-t border-gray-200 dark:border-gray-800">
          {/* Actions */}
          <div className="px-4 py-2 bg-gray-50 dark:bg-gray-900 flex items-center gap-2">
            {!isSold && (
              <Button
                variant="secondary"
                onClick={(e) => {
                  e.stopPropagation()
                  setSellDialogOpen(true)
                }}
              >
                <RiCheckLine className="size-4 mr-1" />
                Sell All
              </Button>
            )}
            <Button
              variant="destructive"
              onClick={(e) => {
                e.stopPropagation()
                handleDeleteBatch()
              }}
              disabled={deleting}
            >
              <RiDeleteBinLine className="size-4 mr-1" />
              {deleting ? "Deleting..." : "Delete Batch"}
            </Button>
          </div>

          {/* Positions Table */}
          <table className="min-w-full divide-y divide-gray-200 dark:divide-gray-800">
            <thead className="bg-gray-50 dark:bg-gray-900">
              <tr>
                <th className="px-4 py-2 text-left text-xs font-medium text-gray-500 uppercase">
                  Symbol
                </th>
                <th className="px-4 py-2 text-right text-xs font-medium text-gray-500 uppercase">
                  Invested
                </th>
                <th className="px-4 py-2 text-right text-xs font-medium text-gray-500 uppercase">
                  Current
                </th>
                <th className="px-4 py-2 text-right text-xs font-medium text-gray-500 uppercase">
                  Return
                </th>
                <th className="px-4 py-2 text-right text-xs font-medium text-gray-500 uppercase">
                  Alpha
                </th>
                <th className="px-4 py-2 text-center text-xs font-medium text-gray-500 uppercase">
                  Sell
                </th>
                <th className="px-4 py-2 text-right text-xs font-medium text-gray-500 uppercase">
                  Actions
                </th>
              </tr>
            </thead>
            <tbody className="bg-white dark:bg-gray-950 divide-y divide-gray-200 dark:divide-gray-800">
              {batch.positions.map((pos) => (
                <PositionRow
                  key={pos.id}
                  position={pos}
                  onDelete={() => handleDeletePosition(pos.id)}
                  onSell={() => handleSellPosition(pos.id)}
                />
              ))}
            </tbody>
          </table>
        </div>
      )}

      {/* Sell Dialog */}
      {sellDialogOpen && (
        <div
          className="fixed inset-0 z-50 flex items-center justify-center bg-black/50"
          onClick={() => setSellDialogOpen(false)}
        >
          <div
            className="bg-white dark:bg-gray-900 rounded-lg p-6 max-w-md w-full mx-4"
            onClick={(e) => e.stopPropagation()}
          >
            <h3 className="text-lg font-semibold mb-4">Sell All Positions</h3>
            <div className="mb-4">
              <label className="block text-sm font-medium mb-1">Sell Quarter</label>
              <input
                type="text"
                value={sellQuarter}
                onChange={(e) => setSellQuarter(e.target.value)}
                className="w-full px-3 py-2 border rounded-md dark:bg-gray-800 dark:border-gray-700"
                placeholder="e.g., 2024Q4"
              />
            </div>
            <div className="flex justify-end gap-2">
              <Button variant="secondary" onClick={() => setSellDialogOpen(false)}>
                Cancel
              </Button>
              <Button onClick={handleSellBatch} disabled={selling}>
                {selling ? "Selling..." : "Confirm Sell"}
              </Button>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}

export default function PortfolioPage() {
  const [dashboard, setDashboard] = useState<Dashboard | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  const loadDashboard = useCallback(async () => {
    try {
      const data = await getDashboard()
      setDashboard(data)
      setError(null)
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed to load dashboard")
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    loadDashboard()
  }, [loadDashboard])

  if (loading) {
    return (
      <div className="p-8">
        <div className="animate-pulse space-y-4">
          <div className="h-8 bg-gray-200 dark:bg-gray-800 rounded w-48" />
          <div className="grid grid-cols-4 gap-4">
            {[1, 2, 3, 4].map((i) => (
              <div key={i} className="h-24 bg-gray-200 dark:bg-gray-800 rounded" />
            ))}
          </div>
        </div>
      </div>
    )
  }

  if (error) {
    return (
      <div className="p-8">
        <div className="rounded-lg border border-red-300 bg-red-50 dark:border-red-800 dark:bg-red-950 p-4">
          <p className="text-red-800 dark:text-red-200">{error}</p>
          <Button variant="secondary" onClick={loadDashboard} className="mt-2">
            Retry
          </Button>
        </div>
      </div>
    )
  }

  if (!dashboard) return null

  const hasPositions = dashboard.active_batches.length > 0 || dashboard.sold_batches.length > 0

  return (
    <div className="p-4 sm:p-6 lg:p-8 space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-3">
          <RiBriefcaseLine className="size-8 text-indigo-600" />
          <div>
            <h1 className="text-2xl font-bold text-gray-900 dark:text-gray-100">Portfolio</h1>
            <p className="text-sm text-gray-500">Track your positions and performance</p>
          </div>
        </div>
      </div>

      {/* Summary Cards */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
        <SummaryCard
          title="Total Invested"
          value={formatCurrency(dashboard.total_invested)}
          icon={<RiMoneyDollarCircleLine className="size-4" />}
        />
        <SummaryCard
          title="Current Value"
          value={formatCurrency(dashboard.total_current_value)}
          icon={<RiMoneyDollarCircleLine className="size-4" />}
          variant={
            dashboard.total_unrealized_return
              ? dashboard.total_unrealized_return >= 0
                ? "positive"
                : "negative"
              : "default"
          }
        />
        <SummaryCard
          title="Unrealized Return"
          value={formatReturnPct(dashboard.total_unrealized_return)}
          icon={<RiPercentLine className="size-4" />}
          variant={
            dashboard.total_unrealized_return
              ? dashboard.total_unrealized_return >= 0
                ? "positive"
                : "negative"
              : "default"
          }
        />
        <SummaryCard
          title="Realized Return"
          value={formatReturnPct(dashboard.total_realized_return)}
          subtitle={
            dashboard.total_realized_alpha
              ? `Alpha: ${formatReturnPct(dashboard.total_realized_alpha)}`
              : undefined
          }
          icon={<RiCheckLine className="size-4" />}
          variant={
            dashboard.total_realized_return
              ? dashboard.total_realized_return >= 0
                ? "positive"
                : "negative"
              : "default"
          }
        />
      </div>

      {/* Alerts */}
      <AlertBanner alerts={dashboard.sell_alerts} />

      {/* Empty State */}
      {!hasPositions && (
        <div className="text-center py-12 rounded-lg border border-dashed border-gray-300 dark:border-gray-700">
          <RiBriefcaseLine className="size-12 mx-auto text-gray-400 mb-4" />
          <h3 className="text-lg font-medium text-gray-900 dark:text-gray-100 mb-2">
            No positions yet
          </h3>
          <p className="text-gray-500 mb-4">
            Go to a screener page and click &quot;Add to Portfolio&quot; to start tracking positions.
          </p>
          <Link href="/">
            <Button>Go to Pipeline</Button>
          </Link>
        </div>
      )}

      {/* Active Batches */}
      {dashboard.active_batches.length > 0 && (
        <div className="space-y-4">
          <h2 className="text-lg font-semibold text-gray-900 dark:text-gray-100">
            Active Positions ({dashboard.active_batches.length} batches)
          </h2>
          {dashboard.active_batches.map((batch) => (
            <BatchCard key={batch.id} batch={batch} onRefresh={loadDashboard} />
          ))}
        </div>
      )}

      {/* Sold Batches */}
      {dashboard.sold_batches.length > 0 && (
        <div className="space-y-4">
          <h2 className="text-lg font-semibold text-gray-900 dark:text-gray-100">
            History ({dashboard.sold_batches.length} batches sold)
          </h2>
          {dashboard.sold_batches.map((batch) => (
            <BatchCard key={batch.id} batch={batch} onRefresh={loadDashboard} />
          ))}
        </div>
      )}
    </div>
  )
}
