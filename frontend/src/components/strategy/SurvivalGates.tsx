"use client"

import {
  AltmanZScoreInfo,
  PiotroskiFScoreInfo,
} from "@/components/InfoPopover"

export interface SurvivalGatesProps {
  requireAltman: boolean
  setRequireAltman: (v: boolean) => void
  altmanZone: "safe" | "grey" | "distress"
  setAltmanZone: (v: "safe" | "grey" | "distress") => void
  requirePiotroski: boolean
  setRequirePiotroski: (v: boolean) => void
  piotroskiMin: number
  setPiotroskiMin: (v: number) => void
}

export function SurvivalGates({
  requireAltman,
  setRequireAltman,
  altmanZone,
  setAltmanZone,
  requirePiotroski,
  setRequirePiotroski,
  piotroskiMin,
  setPiotroskiMin,
}: SurvivalGatesProps) {
  return (
    <div className="mb-4 p-4 rounded-lg border border-gray-200 dark:border-gray-800 bg-white dark:bg-gray-950">
      <div className="flex items-center gap-2 mb-3">
        <span className="w-5 h-5 flex items-center justify-center rounded-full bg-green-100 dark:bg-green-900 text-green-700 dark:text-green-300 text-xs font-bold">
          1
        </span>
        <h2 className="text-sm font-semibold text-gray-900 dark:text-gray-100">
          Stage 1: Survival Gates
        </h2>
        <span className="text-xs text-gray-500">Hard filters - excludes financially distressed companies</span>
      </div>
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {/* Altman */}
        <div className="flex items-center justify-between p-3 rounded-lg bg-gray-50 dark:bg-gray-900">
          <div>
            <label className="flex items-center gap-2 cursor-pointer">
              <input
                type="checkbox"
                checked={requireAltman}
                onChange={(e) => setRequireAltman(e.target.checked)}
                className="w-4 h-4 rounded border-gray-300 text-indigo-600 focus:ring-indigo-500"
              />
              <span className="text-sm font-medium text-gray-700 dark:text-gray-300">
                Altman Z-Score
              </span>
              <AltmanZScoreInfo />
            </label>
            <p className="text-xs text-gray-500 mt-1 ml-6">Exclude bankruptcy risk</p>
          </div>
          <select
            value={altmanZone}
            onChange={(e) => setAltmanZone(e.target.value as "safe" | "grey" | "distress")}
            disabled={!requireAltman}
            className="text-sm rounded-lg border-gray-300 dark:border-gray-700 dark:bg-gray-800 disabled:opacity-50"
          >
            <option value="safe">Safe zone only</option>
            <option value="grey">Include grey zone</option>
            <option value="distress">Include distress</option>
          </select>
        </div>
        {/* Piotroski */}
        <div className="flex items-center justify-between p-3 rounded-lg bg-gray-50 dark:bg-gray-900">
          <div>
            <label className="flex items-center gap-2 cursor-pointer">
              <input
                type="checkbox"
                checked={requirePiotroski}
                onChange={(e) => setRequirePiotroski(e.target.checked)}
                className="w-4 h-4 rounded border-gray-300 text-indigo-600 focus:ring-indigo-500"
              />
              <span className="text-sm font-medium text-gray-700 dark:text-gray-300">
                Piotroski F-Score
              </span>
              <PiotroskiFScoreInfo />
            </label>
            <p className="text-xs text-gray-500 mt-1 ml-6">Exclude weak financials</p>
          </div>
          <select
            value={piotroskiMin}
            onChange={(e) => setPiotroskiMin(Number(e.target.value))}
            disabled={!requirePiotroski}
            className="text-sm rounded-lg border-gray-300 dark:border-gray-700 dark:bg-gray-800 disabled:opacity-50"
          >
            {[1, 2, 3, 4, 5, 6, 7, 8, 9].map((n) => (
              <option key={n} value={n}>
                Min {n}/9
              </option>
            ))}
          </select>
        </div>
      </div>
    </div>
  )
}
