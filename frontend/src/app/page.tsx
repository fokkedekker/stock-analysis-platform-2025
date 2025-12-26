"use client"

import { useEffect, useState, useMemo, useCallback } from "react"
import Link from "next/link"
import { useRouter } from "next/navigation"
import { Pagination } from "@/components/Pagination"
import { BacktestFloatingBar } from "@/components/BacktestFloatingBar"
import { useQuarter } from "@/contexts/QuarterContext"
import {
  getPipelineStocks,
  PipelineStock,
  PipelineParams,
  RankMethod,
  ValuationLens,
  QualityLabel,
  formatNumber,
  formatPercent,
} from "@/lib/api"
import {
  RiShieldCheckLine,
  RiSettings4Line,
} from "@remixicon/react"
import {
  AltmanZScoreInfo,
  PiotroskiFScoreInfo,
  GrahamInfo,
  NetNetInfo,
  PEGInfo,
  MagicFormulaInfo,
  FamaFrenchBMInfo,
} from "@/components/InfoPopover"

const ITEMS_PER_PAGE = 50

// Survival badge component
function SurvivalBadges({
  altmanPassed,
  altmanZScore,
  piotroskiPassed,
  piotroskiScore,
}: {
  altmanPassed: boolean
  altmanZScore: number | null
  piotroskiPassed: boolean
  piotroskiScore: number | null
}) {
  return (
    <div className="flex items-center gap-1">
      <span
        className={`inline-flex items-center justify-center w-6 h-6 rounded text-xs font-medium ${
          altmanPassed
            ? "bg-green-100 text-green-700 dark:bg-green-900 dark:text-green-300"
            : "bg-red-100 text-red-700 dark:bg-red-900 dark:text-red-300"
        }`}
        title={`Altman Z-Score: ${altmanZScore?.toFixed(2) ?? "N/A"}`}
      >
        A
      </span>
      <span
        className={`inline-flex items-center justify-center w-6 h-6 rounded text-xs font-medium ${
          piotroskiPassed
            ? "bg-green-100 text-green-700 dark:bg-green-900 dark:text-green-300"
            : "bg-red-100 text-red-700 dark:bg-red-900 dark:text-red-300"
        }`}
        title={`Piotroski F-Score: ${piotroskiScore ?? "N/A"}/9`}
      >
        P
      </span>
    </div>
  )
}

// Quality label component
function QualityBadge({ label, roic }: { label: QualityLabel; roic: number | null }) {
  const config = {
    compounder: {
      bg: "bg-emerald-100 dark:bg-emerald-900",
      text: "text-emerald-700 dark:text-emerald-300",
      short: "Comp",
    },
    average: {
      bg: "bg-yellow-100 dark:bg-yellow-900",
      text: "text-yellow-700 dark:text-yellow-300",
      short: "Avg",
    },
    weak: {
      bg: "bg-gray-100 dark:bg-gray-800",
      text: "text-gray-600 dark:text-gray-400",
      short: "Weak",
    },
  }
  const c = config[label] || config.weak

  return (
    <span
      className={`inline-flex items-center px-2 py-0.5 rounded text-xs font-medium ${c.bg} ${c.text}`}
      title={`ROIC: ${roic ? formatPercent(roic) : "N/A"}`}
    >
      {c.short}
    </span>
  )
}

// Valuation chips component
function ValuationChips({ lenses }: { lenses: ValuationLens[] }) {
  const lensConfig: Record<ValuationLens, { label: string; color: string }> = {
    graham: { label: "Graham", color: "bg-blue-100 text-blue-700 dark:bg-blue-900 dark:text-blue-300" },
    "net-net": { label: "Net-Net", color: "bg-purple-100 text-purple-700 dark:bg-purple-900 dark:text-purple-300" },
    peg: { label: "PEG", color: "bg-teal-100 text-teal-700 dark:bg-teal-900 dark:text-teal-300" },
    "magic-formula": { label: "MF", color: "bg-orange-100 text-orange-700 dark:bg-orange-900 dark:text-orange-300" },
    "fama-french-bm": { label: "FF", color: "bg-pink-100 text-pink-700 dark:bg-pink-900 dark:text-pink-300" },
  }

  if (lenses.length === 0) {
    return <span className="text-gray-400 text-xs">None</span>
  }

  return (
    <div className="flex flex-wrap gap-1">
      {lenses.map((lens) => (
        <span
          key={lens}
          className={`inline-flex items-center px-1.5 py-0.5 rounded text-xs font-medium ${lensConfig[lens].color}`}
        >
          {lensConfig[lens].label}
        </span>
      ))}
    </div>
  )
}

// Slider component for min lenses
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

const RANK_OPTIONS: { value: RankMethod; label: string }[] = [
  { value: "magic-formula", label: "Magic Formula" },
  { value: "earnings-yield", label: "Earnings Yield" },
  { value: "roic", label: "ROIC" },
  { value: "peg", label: "PEG Ratio" },
  { value: "graham-score", label: "Graham Score" },
  { value: "net-net-discount", label: "Net-Net Discount" },
]

export default function PipelinePage() {
  const router = useRouter()
  const { quarter } = useQuarter()
  const [stocks, setStocks] = useState<PipelineStock[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [currentPage, setCurrentPage] = useState(1)
  const [selectedStocks, setSelectedStocks] = useState<Set<string>>(new Set())

  // Stage 1: Survival
  const [requireAltman, setRequireAltman] = useState(true)
  const [altmanZone, setAltmanZone] = useState<"safe" | "grey">("safe")
  const [requirePiotroski, setRequirePiotroski] = useState(true)
  const [piotroskiMin, setPiotroskiMin] = useState(6)

  // Stage 2: Quality
  const [qualityFilter, setQualityFilter] = useState(true)
  const [minQuality, setMinQuality] = useState<QualityLabel>("compounder")

  // Stage 3: Valuation
  const [minLenses, setMinLenses] = useState(3)
  const [strictMode, setStrictMode] = useState(false)
  const [lensGraham, setLensGraham] = useState(true)
  const [lensNetNet, setLensNetNet] = useState(true)
  const [lensPeg, setLensPeg] = useState(true)
  const [lensMagicFormula, setLensMagicFormula] = useState(true)
  const [lensFamaFrenchBm, setLensFamaFrenchBm] = useState(true)
  const [grahamMode, setGrahamMode] = useState<"strict" | "modern" | "garp" | "relaxed">("strict")
  const [grahamMin, setGrahamMin] = useState(8)
  const [maxPeg, setMaxPeg] = useState(1.5)
  const [mfTopPct, setMfTopPct] = useState(20)
  const [ffBmTopPct, setFfBmTopPct] = useState(30)

  // Ranking
  const [rankBy, setRankBy] = useState<RankMethod>("magic-formula")

  // Advanced toggle
  const [showAdvanced, setShowAdvanced] = useState(true)

  const activeLensCount = useMemo(() => {
    return [lensGraham, lensNetNet, lensPeg, lensMagicFormula, lensFamaFrenchBm].filter(Boolean).length
  }, [lensGraham, lensNetNet, lensPeg, lensMagicFormula, lensFamaFrenchBm])

  const fetchData = useCallback(async () => {
    try {
      setLoading(true)
      const params: PipelineParams = {
        require_altman: requireAltman,
        altman_zone: altmanZone,
        require_piotroski: requirePiotroski,
        piotroski_min: piotroskiMin,
        quality_filter: qualityFilter,
        min_quality: minQuality,
        min_valuation_lenses: minLenses,
        strict_mode: strictMode,
        lens_graham: lensGraham,
        lens_net_net: lensNetNet,
        lens_peg: lensPeg,
        lens_magic_formula: lensMagicFormula,
        lens_fama_french_bm: lensFamaFrenchBm,
        graham_mode: grahamMode,
        graham_min: grahamMin,
        max_peg: maxPeg,
        mf_top_pct: mfTopPct,
        ff_bm_top_pct: ffBmTopPct,
        rank_by: rankBy,
        limit: 500,
      }
      const data = await getPipelineStocks(params, quarter)
      setStocks(data.stocks)
      setError(null)
    } catch (err) {
      setError("Failed to fetch stocks. Is the API server running?")
      console.error(err)
    } finally {
      setLoading(false)
    }
  }, [
    quarter,
    requireAltman,
    altmanZone,
    requirePiotroski,
    piotroskiMin,
    qualityFilter,
    minQuality,
    minLenses,
    strictMode,
    lensGraham,
    lensNetNet,
    lensPeg,
    lensMagicFormula,
    lensFamaFrenchBm,
    grahamMode,
    grahamMin,
    maxPeg,
    mfTopPct,
    ffBmTopPct,
    rankBy,
  ])

  useEffect(() => {
    fetchData()
  }, [fetchData])

  // Reset to page 1 when filters change
  useEffect(() => {
    setCurrentPage(1)
  }, [stocks])

  // Clear selection when stocks or quarter changes
  useEffect(() => {
    setSelectedStocks(new Set())
  }, [stocks, quarter])

  // Selection handlers
  const toggleStock = (symbol: string) => {
    setSelectedStocks((prev) => {
      const next = new Set(prev)
      if (next.has(symbol)) next.delete(symbol)
      else next.add(symbol)
      return next
    })
  }

  const selectAll = () => setSelectedStocks(new Set(stocks.map((s) => s.symbol)))
  const clearSelection = () => setSelectedStocks(new Set())

  const handleSimulateBuy = () => {
    if (!quarter || selectedStocks.size === 0) return
    const symbols = Array.from(selectedStocks).join(",")
    router.push(`/backtest?symbols=${symbols}&quarter=${quarter}`)
  }

  // Pagination
  const totalPages = Math.ceil(stocks.length / ITEMS_PER_PAGE)
  const paginatedStocks = useMemo(() => {
    const start = (currentPage - 1) * ITEMS_PER_PAGE
    return stocks.slice(start, start + ITEMS_PER_PAGE)
  }, [stocks, currentPage])

  return (
    <div className="p-4 sm:p-6 lg:p-8">
      <div className="mb-6">
        <h1 className="text-2xl font-bold text-gray-900 dark:text-gray-50">
          Stock Pipeline
        </h1>
        <p className="mt-1 text-sm text-gray-500">
          Filter stocks through survival gates, classify quality, and find buy candidates via valuation lenses
        </p>
      </div>

      {/* Stage 1: Survival Gates */}
      <div className="mb-4 p-4 rounded-lg border border-gray-200 dark:border-gray-800 bg-white dark:bg-gray-950">
        <div className="flex items-center gap-2 mb-3">
          <RiShieldCheckLine className="w-5 h-5 text-green-600" />
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
              onChange={(e) => setAltmanZone(e.target.value as "safe" | "grey")}
              disabled={!requireAltman}
              className="text-sm rounded-lg border-gray-300 dark:border-gray-700 dark:bg-gray-800 disabled:opacity-50"
            >
              <option value="safe">Safe zone only</option>
              <option value="grey">Include grey zone</option>
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
              {[3, 4, 5, 6, 7, 8].map((n) => (
                <option key={n} value={n}>
                  Min {n}/9
                </option>
              ))}
            </select>
          </div>
        </div>
      </div>

      {/* Stage 2: Quality Classification */}
      <div className="mb-4 p-4 rounded-lg border border-gray-200 dark:border-gray-800 bg-white dark:bg-gray-950">
        <div className="flex items-center justify-between mb-3">
          <div className="flex items-center gap-2">
            <span className="w-5 h-5 flex items-center justify-center rounded-full bg-yellow-100 dark:bg-yellow-900 text-yellow-700 dark:text-yellow-300 text-xs font-bold">
              2
            </span>
            <h2 className="text-sm font-semibold text-gray-900 dark:text-gray-100">
              Stage 2: Quality Classification
            </h2>
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
      </div>

      {/* Stage 3: Valuation Lenses */}
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

        {/* Slider */}
        <div className="mb-4 flex items-center gap-4">
          <LensSlider value={minLenses} max={activeLensCount} onChange={setMinLenses} />
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
                  onChange={(e) => setGrahamMode(e.target.value as typeof grahamMode)}
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
                  {[10, 20, 30, 50].map((n) => (
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
                  {[20, 30, 40, 50].map((n) => (
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

      {/* Ranking */}
      <div className="mb-6 flex items-center gap-4">
        <span className="text-sm font-medium text-gray-700 dark:text-gray-300">Rank by:</span>
        <select
          value={rankBy}
          onChange={(e) => setRankBy(e.target.value as RankMethod)}
          className="text-sm rounded-lg border-gray-300 dark:border-gray-700 dark:bg-gray-800"
        >
          {RANK_OPTIONS.map((opt) => (
            <option key={opt.value} value={opt.value}>
              {opt.label}
            </option>
          ))}
        </select>
        <span className="text-sm text-gray-500">
          Showing {stocks.length} stocks
        </span>
      </div>

      {/* Error State */}
      {error && (
        <div className="rounded-lg border border-red-200 bg-red-50 p-4 dark:border-red-900 dark:bg-red-950 mb-4">
          <p className="text-sm text-red-600 dark:text-red-400">{error}</p>
        </div>
      )}

      {/* Loading State */}
      {loading && (
        <div className="flex items-center justify-center py-12">
          <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-indigo-600"></div>
        </div>
      )}

      {/* Results Table */}
      {!loading && !error && (
        <div className="overflow-x-auto rounded-lg border border-gray-200 dark:border-gray-800">
          <table className="min-w-full divide-y divide-gray-200 dark:divide-gray-800">
            <thead className="bg-gray-50 dark:bg-gray-900">
              <tr>
                {quarter && (
                  <th className="px-3 py-3 w-10">
                    <input
                      type="checkbox"
                      checked={selectedStocks.size === stocks.length && stocks.length > 0}
                      onChange={(e) => (e.target.checked ? selectAll() : clearSelection())}
                      className="w-4 h-4 rounded border-gray-300 text-indigo-600 focus:ring-indigo-500"
                      title="Select all"
                    />
                  </th>
                )}
                <th className="px-3 py-3 text-right text-xs font-medium text-gray-500 uppercase tracking-wider w-10">
                  #
                </th>
                <th className="px-3 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Symbol
                </th>
                <th className="px-3 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Name
                </th>
                <th className="px-3 py-3 text-center text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Survival
                </th>
                <th className="px-3 py-3 text-center text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Quality
                </th>
                <th className="px-3 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Valuation Lenses
                </th>
                <th className="px-3 py-3 text-right text-xs font-medium text-gray-500 uppercase tracking-wider">
                  {RANK_OPTIONS.find((r) => r.value === rankBy)?.label || "Rank"}
                </th>
              </tr>
            </thead>
            <tbody className="bg-white divide-y divide-gray-200 dark:bg-gray-950 dark:divide-gray-800">
              {paginatedStocks.map((stock, index) => (
                <tr
                  key={stock.symbol}
                  className="hover:bg-gray-50 dark:hover:bg-gray-900"
                >
                  {quarter && (
                    <td className="px-3 py-3">
                      <input
                        type="checkbox"
                        checked={selectedStocks.has(stock.symbol)}
                        onChange={() => toggleStock(stock.symbol)}
                        className="w-4 h-4 rounded border-gray-300 text-indigo-600 focus:ring-indigo-500"
                      />
                    </td>
                  )}
                  <td className="px-3 py-3 whitespace-nowrap text-right text-sm text-gray-500">
                    {(currentPage - 1) * ITEMS_PER_PAGE + index + 1}
                  </td>
                  <td className="px-3 py-3 whitespace-nowrap">
                    <Link
                      href={`/stock/${stock.symbol}`}
                      className="text-sm font-medium text-indigo-600 hover:text-indigo-500 dark:text-indigo-400"
                    >
                      {stock.symbol}
                    </Link>
                  </td>
                  <td className="px-3 py-3 whitespace-nowrap text-sm text-gray-900 dark:text-gray-100 max-w-[180px] truncate">
                    {stock.name}
                  </td>
                  <td className="px-3 py-3 text-center">
                    <SurvivalBadges
                      altmanPassed={stock.altman_passed}
                      altmanZScore={stock.altman_z_score}
                      piotroskiPassed={stock.piotroski_passed}
                      piotroskiScore={stock.piotroski_score}
                    />
                  </td>
                  <td className="px-3 py-3 text-center">
                    <QualityBadge label={stock.quality_label} roic={stock.roic} />
                  </td>
                  <td className="px-3 py-3">
                    <ValuationChips lenses={stock.valuation_lenses_passed} />
                  </td>
                  <td className="px-3 py-3 text-right text-sm text-gray-900 dark:text-gray-100">
                    {rankBy === "magic-formula" && stock.magic_formula_rank != null
                      ? `#${stock.magic_formula_rank}`
                      : rankBy === "earnings-yield" && stock.earnings_yield != null
                      ? formatPercent(stock.earnings_yield)
                      : rankBy === "roic" && stock.roic != null
                      ? formatPercent(stock.roic)
                      : rankBy === "peg" && stock.peg_ratio != null
                      ? formatNumber(stock.peg_ratio, 2)
                      : rankBy === "graham-score" && stock.graham_score != null
                      ? `${stock.graham_score}/8`
                      : rankBy === "net-net-discount" && stock.net_net_discount != null
                      ? formatPercent(stock.net_net_discount)
                      : "—"}
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
          No stocks found matching the pipeline criteria. Try relaxing some filters.
        </div>
      )}

      {/* Floating action bar for backtest - only show when historical quarter selected */}
      {quarter && (
        <BacktestFloatingBar
          selectedCount={selectedStocks.size}
          onClear={clearSelection}
          onSimulate={handleSimulateBuy}
        />
      )}
    </div>
  )
}
