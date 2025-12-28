"use client"

import { QualityLabel, QualityTag, QUALITY_TAGS } from "@/lib/api"
import {
  QualityClassificationInfo,
  QualityTagsInfo,
} from "@/components/InfoPopover"

export interface QualityClassificationProps {
  qualityFilter: boolean
  setQualityFilter: (v: boolean) => void
  minQuality: QualityLabel
  setMinQuality: (v: QualityLabel) => void
  selectedTags: Set<QualityTag>
  setSelectedTags: (v: Set<QualityTag>) => void
}

export function QualityClassification({
  qualityFilter,
  setQualityFilter,
  minQuality,
  setMinQuality,
  selectedTags,
  setSelectedTags,
}: QualityClassificationProps) {
  const tagColors: Record<string, string> = {
    "Durable Compounder": "bg-emerald-100 text-emerald-700 dark:bg-emerald-900 dark:text-emerald-300 border-emerald-300 dark:border-emerald-700",
    "Cash Machine": "bg-blue-100 text-blue-700 dark:bg-blue-900 dark:text-blue-300 border-blue-300 dark:border-blue-700",
    "Deep Value": "bg-purple-100 text-purple-700 dark:bg-purple-900 dark:text-purple-300 border-purple-300 dark:border-purple-700",
    "Heavy Reinvestor": "bg-amber-100 text-amber-700 dark:bg-amber-900 dark:text-amber-300 border-amber-300 dark:border-amber-700",
    "Volatile Returns": "bg-red-100 text-red-700 dark:bg-red-900 dark:text-red-300 border-red-300 dark:border-red-700",
    "Earnings Quality Concern": "bg-orange-100 text-orange-700 dark:bg-orange-900 dark:text-orange-300 border-orange-300 dark:border-orange-700",
    "Premium Priced": "bg-gray-100 text-gray-600 dark:bg-gray-800 dark:text-gray-400 border-gray-300 dark:border-gray-600",
    "Weak Moat Signal": "bg-yellow-100 text-yellow-700 dark:bg-yellow-900 dark:text-yellow-300 border-yellow-300 dark:border-yellow-700",
  }

  return (
    <div className="mb-4 p-4 rounded-lg border border-gray-200 dark:border-gray-800 bg-white dark:bg-gray-950">
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center gap-2">
          <span className="w-5 h-5 flex items-center justify-center rounded-full bg-yellow-100 dark:bg-yellow-900 text-yellow-700 dark:text-yellow-300 text-xs font-bold">
            2
          </span>
          <h2 className="text-sm font-semibold text-gray-900 dark:text-gray-100">
            Stage 2: Quality Classification
          </h2>
          <QualityClassificationInfo />
          <span className="text-xs text-gray-500">Labels based on ROIC</span>
        </div>
        <label className="flex items-center gap-2 cursor-pointer">
          <input
            type="checkbox"
            checked={qualityFilter}
            onChange={(e) => setQualityFilter(e.target.checked)}
            className="w-4 h-4 rounded border-gray-300 text-indigo-600 focus:ring-indigo-500"
          />
          <span className="text-sm text-gray-600 dark:text-gray-400">Filter by quality</span>
        </label>
      </div>
      <div className="flex items-center gap-4">
        <div className="flex items-center gap-2">
          <span className="inline-flex items-center px-2 py-0.5 rounded text-xs font-medium bg-emerald-100 text-emerald-700 dark:bg-emerald-900 dark:text-emerald-300">
            Compounder
          </span>
          <span className="text-xs text-gray-500">ROIC 15%+ with FCF</span>
        </div>
        <div className="flex items-center gap-2">
          <span className="inline-flex items-center px-2 py-0.5 rounded text-xs font-medium bg-yellow-100 text-yellow-700 dark:bg-yellow-900 dark:text-yellow-300">
            Average
          </span>
          <span className="text-xs text-gray-500">ROIC 8-15%</span>
        </div>
        <div className="flex items-center gap-2">
          <span className="inline-flex items-center px-2 py-0.5 rounded text-xs font-medium bg-gray-100 text-gray-600 dark:bg-gray-800 dark:text-gray-400">
            Weak
          </span>
          <span className="text-xs text-gray-500">ROIC &lt;8%</span>
        </div>
        {qualityFilter && (
          <select
            value={minQuality}
            onChange={(e) => setMinQuality(e.target.value as QualityLabel)}
            className="ml-4 text-sm rounded-lg border-gray-300 dark:border-gray-700 dark:bg-gray-800"
          >
            <option value="compounder">Compounders only</option>
            <option value="average">Average or better</option>
            <option value="weak">All</option>
          </select>
        )}
      </div>

      {/* Tag Filter */}
      <div className="mt-4 pt-4 border-t border-gray-200 dark:border-gray-700">
        <div className="flex items-center justify-between mb-2">
          <span className="text-sm font-medium text-gray-700 dark:text-gray-300 inline-flex items-center gap-1">
            Filter by Tags
            <QualityTagsInfo />
          </span>
          {selectedTags.size > 0 && (
            <button
              onClick={() => setSelectedTags(new Set())}
              className="text-xs text-indigo-600 hover:text-indigo-500"
            >
              Clear all
            </button>
          )}
        </div>
        <div className="flex flex-wrap gap-2">
          {QUALITY_TAGS.map((tag) => {
            const isSelected = selectedTags.has(tag)
            return (
              <button
                key={tag}
                onClick={() => {
                  const next = new Set(selectedTags)
                  if (next.has(tag)) next.delete(tag)
                  else next.add(tag)
                  setSelectedTags(next)
                }}
                className={`px-2 py-1 rounded-full text-xs font-medium border-2 transition-all ${
                  isSelected
                    ? tagColors[tag]
                    : "bg-gray-50 text-gray-500 border-gray-200 dark:bg-gray-900 dark:text-gray-500 dark:border-gray-700 hover:border-gray-300"
                }`}
              >
                {tag}
              </button>
            )
          })}
        </div>
        {selectedTags.size > 0 && (
          <p className="mt-2 text-xs text-gray-500">
            Showing stocks with selected tags only, excluding stocks with unselected tags ({selectedTags.size} selected)
          </p>
        )}
      </div>
    </div>
  )
}
