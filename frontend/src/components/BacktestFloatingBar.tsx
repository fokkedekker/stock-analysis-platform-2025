"use client"

import { useState } from "react"
import { RiPlayLine, RiCloseLine, RiBriefcaseLine } from "@remixicon/react"
import { Button } from "@/components/Button"
import { createBatch, checkMerge, CreateBatchRequest } from "@/lib/api"

interface BacktestFloatingBarProps {
  selectedCount: number
  selectedSymbols: string[]
  quarter: string | null
  onClear: () => void
  onSimulate: () => void
  onPortfolioAdded?: () => void
  disabled?: boolean
}

export function BacktestFloatingBar({
  selectedCount,
  selectedSymbols,
  quarter,
  onClear,
  onSimulate,
  onPortfolioAdded,
  disabled,
}: BacktestFloatingBarProps) {
  const [showPortfolioModal, setShowPortfolioModal] = useState(false)
  const [batchName, setBatchName] = useState("")
  const [totalInvested, setTotalInvested] = useState("10000")
  const [holdingPeriod, setHoldingPeriod] = useState("4")
  const [creating, setCreating] = useState(false)
  const [mergeInfo, setMergeInfo] = useState<string[]>([])

  if (selectedCount === 0) return null

  const handleOpenPortfolioModal = async () => {
    setBatchName(`${quarter} Batch`)
    setShowPortfolioModal(true)

    // Check for merge candidates
    try {
      const result = await checkMerge(selectedSymbols)
      setMergeInfo(result.merge_candidates)
    } catch (e) {
      console.error("Failed to check merge:", e)
    }
  }

  const handleCreateBatch = async () => {
    if (!quarter) return

    setCreating(true)
    try {
      const request: CreateBatchRequest = {
        name: batchName || undefined,
        buy_quarter: quarter,
        holding_period: parseInt(holdingPeriod) || 4,
        total_invested: parseFloat(totalInvested) || 10000,
        symbols: selectedSymbols,
      }

      await createBatch(request)
      setShowPortfolioModal(false)
      onClear()
      onPortfolioAdded?.()
      alert("Stocks added to portfolio!")
    } catch (e) {
      alert(e instanceof Error ? e.message : "Failed to add to portfolio")
    } finally {
      setCreating(false)
    }
  }

  return (
    <>
      <div className="fixed bottom-6 left-1/2 -translate-x-1/2 z-50">
        <div className="flex items-center gap-4 px-6 py-3 rounded-full bg-gray-900 dark:bg-gray-800 text-white shadow-xl border border-gray-700">
          <span className="text-sm font-medium">
            {selectedCount} stock{selectedCount !== 1 ? "s" : ""} selected
          </span>
          <button
            onClick={onClear}
            className="flex items-center gap-1 text-sm text-gray-400 hover:text-white transition-colors"
          >
            <RiCloseLine className="size-4" />
            Clear
          </button>
          <Button
            onClick={onSimulate}
            disabled={disabled}
            className="bg-indigo-600 hover:bg-indigo-500 border-indigo-600"
          >
            <RiPlayLine className="size-4 mr-1.5" />
            Simulate Buy
          </Button>
          <Button
            onClick={handleOpenPortfolioModal}
            disabled={disabled || !quarter}
            className="bg-green-600 hover:bg-green-500 border-green-600"
          >
            <RiBriefcaseLine className="size-4 mr-1.5" />
            Add to Portfolio
          </Button>
        </div>
      </div>

      {/* Portfolio Modal */}
      {showPortfolioModal && (
        <div
          className="fixed inset-0 z-[60] flex items-center justify-center bg-black/50"
          onClick={() => setShowPortfolioModal(false)}
        >
          <div
            className="bg-white dark:bg-gray-900 rounded-lg p-6 max-w-md w-full mx-4 shadow-xl"
            onClick={(e) => e.stopPropagation()}
          >
            <h3 className="text-lg font-semibold mb-4 text-gray-900 dark:text-gray-100">
              Add to Portfolio
            </h3>

            {mergeInfo.length > 0 && (
              <div className="mb-4 p-3 rounded bg-yellow-50 dark:bg-yellow-950 border border-yellow-200 dark:border-yellow-800">
                <p className="text-sm text-yellow-800 dark:text-yellow-200">
                  <strong>Note:</strong> {mergeInfo.join(", ")} already have open positions.
                  New tranches will be added to extend the holding period.
                </p>
              </div>
            )}

            <div className="space-y-4">
              <div>
                <label className="block text-sm font-medium mb-1 text-gray-700 dark:text-gray-300">
                  Batch Name
                </label>
                <input
                  type="text"
                  value={batchName}
                  onChange={(e) => setBatchName(e.target.value)}
                  className="w-full px-3 py-2 border rounded-md dark:bg-gray-800 dark:border-gray-700 text-gray-900 dark:text-gray-100"
                  placeholder="e.g., Q4 2024 Graham Picks"
                />
              </div>

              <div>
                <label className="block text-sm font-medium mb-1 text-gray-700 dark:text-gray-300">
                  Total Investment ($)
                </label>
                <input
                  type="number"
                  value={totalInvested}
                  onChange={(e) => setTotalInvested(e.target.value)}
                  className="w-full px-3 py-2 border rounded-md dark:bg-gray-800 dark:border-gray-700 text-gray-900 dark:text-gray-100"
                  min="100"
                  step="100"
                />
                <p className="text-xs text-gray-500 mt-1">
                  ${(parseFloat(totalInvested) / selectedCount).toFixed(0)} per stock (equal weight)
                </p>
              </div>

              <div>
                <label className="block text-sm font-medium mb-1 text-gray-700 dark:text-gray-300">
                  Holding Period (quarters)
                </label>
                <select
                  value={holdingPeriod}
                  onChange={(e) => setHoldingPeriod(e.target.value)}
                  className="w-full px-3 py-2 border rounded-md dark:bg-gray-800 dark:border-gray-700 text-gray-900 dark:text-gray-100"
                >
                  <option value="1">1 Quarter</option>
                  <option value="2">2 Quarters</option>
                  <option value="3">3 Quarters</option>
                  <option value="4">4 Quarters (1 Year)</option>
                  <option value="8">8 Quarters (2 Years)</option>
                </select>
              </div>

              <div className="bg-gray-50 dark:bg-gray-800 rounded p-3">
                <p className="text-sm text-gray-600 dark:text-gray-400">
                  <strong>Buy Quarter:</strong> {quarter}
                </p>
                <p className="text-sm text-gray-600 dark:text-gray-400">
                  <strong>Stocks:</strong> {selectedSymbols.slice(0, 5).join(", ")}
                  {selectedCount > 5 && ` +${selectedCount - 5} more`}
                </p>
              </div>
            </div>

            <div className="flex justify-end gap-2 mt-6">
              <Button variant="secondary" onClick={() => setShowPortfolioModal(false)}>
                Cancel
              </Button>
              <Button onClick={handleCreateBatch} disabled={creating}>
                {creating ? "Adding..." : "Add to Portfolio"}
              </Button>
            </div>
          </div>
        </div>
      )}
    </>
  )
}
