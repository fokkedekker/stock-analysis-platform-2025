"use client"

import { RiPlayLine, RiCloseLine } from "@remixicon/react"
import { Button } from "@/components/Button"

interface BacktestFloatingBarProps {
  selectedCount: number
  onClear: () => void
  onSimulate: () => void
  disabled?: boolean
}

export function BacktestFloatingBar({
  selectedCount,
  onClear,
  onSimulate,
  disabled,
}: BacktestFloatingBarProps) {
  if (selectedCount === 0) return null

  return (
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
      </div>
    </div>
  )
}
