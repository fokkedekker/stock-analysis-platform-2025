"use client"

import { useMemo } from "react"
import { RiSettings4Line } from "@remixicon/react"
import {
  GrahamInfo,
  NetNetInfo,
  PEGInfo,
  MagicFormulaInfo,
  FamaFrenchBMInfo,
} from "@/components/InfoPopover"

export type GrahamMode = "strict" | "modern" | "garp" | "relaxed"

export interface ValuationLensesProps {
  minLenses: number
  setMinLenses: (v: number) => void
  strictMode: boolean
  setStrictMode: (v: boolean) => void
  lensGraham: boolean
  setLensGraham: (v: boolean) => void
  lensNetNet: boolean
  setLensNetNet: (v: boolean) => void
  lensPeg: boolean
  setLensPeg: (v: boolean) => void
  lensMagicFormula: boolean
  setLensMagicFormula: (v: boolean) => void
  lensFamaFrenchBm: boolean
  setLensFamaFrenchBm: (v: boolean) => void
  grahamMode: GrahamMode
  setGrahamMode: (v: GrahamMode) => void
  grahamMin: number
  setGrahamMin: (v: number) => void
  maxPeg: number
  setMaxPeg: (v: number) => void
  mfTopPct: number
  setMfTopPct: (v: number) => void
  ffBmTopPct: number
  setFfBmTopPct: (v: number) => void
  showAdvanced: boolean
  setShowAdvanced: (v: boolean) => void
}

function LensSlider({
  value,
  max,
  onChange,
}: {
  value: number
  max: number
  onChange: (v: number) => void
}) {
  return (
    <div className="flex items-center gap-3">
      <input
        type="range"
        min={0}
        max={max}
        value={value}
        onChange={(e) => onChange(Number(e.target.value))}
        className="w-32 h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer dark:bg-gray-700 accent-indigo-600"
      />
      <span className="text-sm font-medium text-gray-700 dark:text-gray-300 w-20">
        At least {value}
      </span>
    </div>
  )
}

export function ValuationLenses({
  minLenses,
  setMinLenses,
  strictMode,
  setStrictMode,
  lensGraham,
  setLensGraham,
  lensNetNet,
  setLensNetNet,
  lensPeg,
  setLensPeg,
  lensMagicFormula,
  setLensMagicFormula,
  lensFamaFrenchBm,
  setLensFamaFrenchBm,
  grahamMode,
  setGrahamMode,
  grahamMin,
  setGrahamMin,
  maxPeg,
  setMaxPeg,
  mfTopPct,
  setMfTopPct,
  ffBmTopPct,
  setFfBmTopPct,
  showAdvanced,
  setShowAdvanced,
}: ValuationLensesProps) {
  const activeLensCount = useMemo(() => {
    return [lensGraham, lensNetNet, lensPeg, lensMagicFormula, lensFamaFrenchBm].filter(Boolean).length
  }, [lensGraham, lensNetNet, lensPeg, lensMagicFormula, lensFamaFrenchBm])

  return (
    <div className="mb-4 p-4 rounded-lg border border-gray-200 dark:border-gray-800 bg-white dark:bg-gray-950">
      <div className="flex items-center gap-2 mb-3">
        <span className="w-5 h-5 flex items-center justify-center rounded-full bg-blue-100 dark:bg-blue-900 text-blue-700 dark:text-blue-300 text-xs font-bold">
          3
        </span>
        <h2 className="text-sm font-semibold text-gray-900 dark:text-gray-100">
          Stage 3: Valuation Lenses
        </h2>
        <span className="text-xs text-gray-500">Independent buy signals - "at least N must pass"</span>
      </div>

      {/* Slider - only show when at least one lens is active */}
      <div className="mb-4 flex items-center gap-4">
        {activeLensCount === 0 ? (
          <span className="text-sm text-gray-500 italic">No valuation filters active</span>
        ) : (
          <LensSlider value={minLenses} max={activeLensCount} onChange={setMinLenses} />
        )}
        <label className="flex items-center gap-2 cursor-pointer">
          <input
            type="checkbox"
            checked={strictMode}
            onChange={(e) => setStrictMode(e.target.checked)}
            className="w-4 h-4 rounded border-gray-300 text-indigo-600 focus:ring-indigo-500"
          />
          <span className="text-sm text-gray-600 dark:text-gray-400">
            Strict mode (must pass ALL selected)
          </span>
        </label>
      </div>

      {/* Lens toggles */}
      <div className="grid grid-cols-2 md:grid-cols-5 gap-3">
        {/* Graham */}
        <div
          className={`p-3 rounded-lg border-2 transition-colors ${
            lensGraham
              ? "border-blue-500 bg-blue-50 dark:bg-blue-950"
              : "border-gray-200 dark:border-gray-700"
          }`}
        >
          <label className="flex items-center gap-2 cursor-pointer">
            <input
              type="checkbox"
              checked={lensGraham}
              onChange={(e) => setLensGraham(e.target.checked)}
              className="w-4 h-4 rounded border-gray-300 text-blue-600 focus:ring-blue-500"
            />
            <span className="text-sm font-medium text-gray-700 dark:text-gray-300">Graham</span>
            <GrahamInfo />
          </label>
          {lensGraham && showAdvanced && (
            <div className="mt-2 space-y-1">
              <select
                value={grahamMode}
                onChange={(e) => setGrahamMode(e.target.value as GrahamMode)}
                className="w-full text-xs rounded border-gray-300 dark:border-gray-700 dark:bg-gray-800"
              >
                <option value="strict">Strict</option>
                <option value="modern">Modern</option>
                <option value="garp">GARP</option>
                <option value="relaxed">Relaxed</option>
              </select>
              <select
                value={grahamMin}
                onChange={(e) => setGrahamMin(Number(e.target.value))}
                className="w-full text-xs rounded border-gray-300 dark:border-gray-700 dark:bg-gray-800"
              >
                {[3, 4, 5, 6, 7, 8].map((n) => (
                  <option key={n} value={n}>
                    Min {n}/8
                  </option>
                ))}
              </select>
            </div>
          )}
        </div>

        {/* Net-Net */}
        <div
          className={`p-3 rounded-lg border-2 transition-colors ${
            lensNetNet
              ? "border-purple-500 bg-purple-50 dark:bg-purple-950"
              : "border-gray-200 dark:border-gray-700"
          }`}
        >
          <label className="flex items-center gap-2 cursor-pointer">
            <input
              type="checkbox"
              checked={lensNetNet}
              onChange={(e) => setLensNetNet(e.target.checked)}
              className="w-4 h-4 rounded border-gray-300 text-purple-600 focus:ring-purple-500"
            />
            <span className="text-sm font-medium text-gray-700 dark:text-gray-300">Net-Net</span>
            <NetNetInfo />
          </label>
          <p className="text-xs text-gray-500 mt-1">Below NCAV</p>
        </div>

        {/* PEG */}
        <div
          className={`p-3 rounded-lg border-2 transition-colors ${
            lensPeg
              ? "border-teal-500 bg-teal-50 dark:bg-teal-950"
              : "border-gray-200 dark:border-gray-700"
          }`}
        >
          <label className="flex items-center gap-2 cursor-pointer">
            <input
              type="checkbox"
              checked={lensPeg}
              onChange={(e) => setLensPeg(e.target.checked)}
              className="w-4 h-4 rounded border-gray-300 text-teal-600 focus:ring-teal-500"
            />
            <span className="text-sm font-medium text-gray-700 dark:text-gray-300">PEG</span>
            <PEGInfo />
          </label>
          {lensPeg && showAdvanced && (
            <div className="mt-2">
              <label className="text-xs text-gray-500">Max PEG</label>
              <input
                type="number"
                step="0.1"
                value={maxPeg}
                onChange={(e) => setMaxPeg(Number(e.target.value))}
                className="w-full text-xs rounded border-gray-300 dark:border-gray-700 dark:bg-gray-800"
              />
            </div>
          )}
        </div>

        {/* Magic Formula */}
        <div
          className={`p-3 rounded-lg border-2 transition-colors ${
            lensMagicFormula
              ? "border-orange-500 bg-orange-50 dark:bg-orange-950"
              : "border-gray-200 dark:border-gray-700"
          }`}
        >
          <label className="flex items-center gap-2 cursor-pointer">
            <input
              type="checkbox"
              checked={lensMagicFormula}
              onChange={(e) => setLensMagicFormula(e.target.checked)}
              className="w-4 h-4 rounded border-gray-300 text-orange-600 focus:ring-orange-500"
            />
            <span className="text-sm font-medium text-gray-700 dark:text-gray-300">Magic Formula</span>
            <MagicFormulaInfo />
          </label>
          {lensMagicFormula && showAdvanced && (
            <div className="mt-2">
              <label className="text-xs text-gray-500">Top %</label>
              <select
                value={mfTopPct}
                onChange={(e) => setMfTopPct(Number(e.target.value))}
                className="w-full text-xs rounded border-gray-300 dark:border-gray-700 dark:bg-gray-800"
              >
                {[10, 20, 30, 40, 50, 60, 70, 80, 90, 100].map((n) => (
                  <option key={n} value={n}>
                    Top {n}%
                  </option>
                ))}
              </select>
            </div>
          )}
        </div>

        {/* Fama-French B/M */}
        <div
          className={`p-3 rounded-lg border-2 transition-colors ${
            lensFamaFrenchBm
              ? "border-pink-500 bg-pink-50 dark:bg-pink-950"
              : "border-gray-200 dark:border-gray-700"
          }`}
        >
          <label className="flex items-center gap-2 cursor-pointer">
            <input
              type="checkbox"
              checked={lensFamaFrenchBm}
              onChange={(e) => setLensFamaFrenchBm(e.target.checked)}
              className="w-4 h-4 rounded border-gray-300 text-pink-600 focus:ring-pink-500"
            />
            <span className="text-sm font-medium text-gray-700 dark:text-gray-300">FF B/M</span>
            <FamaFrenchBMInfo />
          </label>
          {lensFamaFrenchBm && showAdvanced && (
            <div className="mt-2">
              <label className="text-xs text-gray-500">Top %</label>
              <select
                value={ffBmTopPct}
                onChange={(e) => setFfBmTopPct(Number(e.target.value))}
                className="w-full text-xs rounded border-gray-300 dark:border-gray-700 dark:bg-gray-800"
              >
                {[10, 20, 30, 40, 50, 60, 70, 80, 90, 100].map((n) => (
                  <option key={n} value={n}>
                    Top {n}%
                  </option>
                ))}
              </select>
            </div>
          )}
        </div>
      </div>

      {/* Advanced toggle */}
      <button
        onClick={() => setShowAdvanced(!showAdvanced)}
        className="mt-3 flex items-center gap-1 text-sm text-indigo-600 hover:text-indigo-500"
      >
        <RiSettings4Line className="w-4 h-4" />
        {showAdvanced ? "Hide" : "Show"} lens thresholds
      </button>
    </div>
  )
}
