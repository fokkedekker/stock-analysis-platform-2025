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
  QualityTag,
  QUALITY_TAGS,
  formatNumber,
  formatPercent,
} from "@/lib/api"
import {
  loadStrategies,
  SavedStrategy,
  deleteStrategy,
  formatStrategyDate,
} from "@/lib/saved-strategies"
import { SurvivalGates } from "@/components/strategy/SurvivalGates"
import { QualityClassification } from "@/components/strategy/QualityClassification"
import { ValuationLenses, GrahamMode } from "@/components/strategy/ValuationLenses"

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

// Quality tags chips component (shows quality_tags from ROIC analysis)
function QualityTagsChips({ tags }: { tags: string[] | string | null }) {
  // Handle null, empty, or string (JSON) input
  let parsedTags: string[] = []
  if (tags) {
    if (typeof tags === "string") {
      try {
        parsedTags = JSON.parse(tags)
      } catch {
        parsedTags = []
      }
    } else if (Array.isArray(tags)) {
      parsedTags = tags
    }
  }

  if (parsedTags.length === 0) {
    return <span className="text-gray-400 text-xs">—</span>
  }

  const tagColors: Record<string, string> = {
    "Durable Compounder": "bg-emerald-100 text-emerald-700 dark:bg-emerald-900 dark:text-emerald-300",
    "Cash Machine": "bg-blue-100 text-blue-700 dark:bg-blue-900 dark:text-blue-300",
    "Deep Value": "bg-purple-100 text-purple-700 dark:bg-purple-900 dark:text-purple-300",
    "Heavy Reinvestor": "bg-amber-100 text-amber-700 dark:bg-amber-900 dark:text-amber-300",
    "Volatile Returns": "bg-red-100 text-red-700 dark:bg-red-900 dark:text-red-300",
    "Earnings Quality Concern": "bg-orange-100 text-orange-700 dark:bg-orange-900 dark:text-orange-300",
    "Premium Priced": "bg-gray-100 text-gray-600 dark:bg-gray-800 dark:text-gray-400",
    "Weak Moat Signal": "bg-yellow-100 text-yellow-700 dark:bg-yellow-900 dark:text-yellow-300",
  }

  return (
    <div className="flex flex-wrap gap-1">
      {parsedTags.map((tag) => (
        <span
          key={tag}
          className={`inline-flex items-center px-1.5 py-0.5 rounded text-xs font-medium ${tagColors[tag] || "bg-gray-100 text-gray-600 dark:bg-gray-800 dark:text-gray-400"}`}
        >
          {tag}
        </span>
      ))}
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
  const [altmanZone, setAltmanZone] = useState<"safe" | "grey" | "distress">("safe")
  const [requirePiotroski, setRequirePiotroski] = useState(true)
  const [piotroskiMin, setPiotroskiMin] = useState(6)

  // Stage 2: Quality
  const [qualityFilter, setQualityFilter] = useState(true)
  const [minQuality, setMinQuality] = useState<QualityLabel>("compounder")
  const [selectedTags, setSelectedTags] = useState<Set<QualityTag>>(new Set())

  // Stage 3: Valuation
  const [minLenses, setMinLenses] = useState(3)
  const [strictMode, setStrictMode] = useState(false)
  const [lensGraham, setLensGraham] = useState(true)
  const [lensNetNet, setLensNetNet] = useState(true)
  const [lensPeg, setLensPeg] = useState(true)
  const [lensMagicFormula, setLensMagicFormula] = useState(true)
  const [lensFamaFrenchBm, setLensFamaFrenchBm] = useState(true)
  const [grahamMode, setGrahamMode] = useState<GrahamMode>("strict")
  const [grahamMin, setGrahamMin] = useState(8)
  const [maxPeg, setMaxPeg] = useState(1.5)
  const [mfTopPct, setMfTopPct] = useState(20)
  const [ffBmTopPct, setFfBmTopPct] = useState(30)

  // Ranking
  const [rankBy, setRankBy] = useState<RankMethod>("magic-formula")

  // Advanced toggle
  const [showAdvanced, setShowAdvanced] = useState(true)

  // Saved strategies
  const [savedStrategies, setSavedStrategies] = useState<SavedStrategy[]>([])
  const [selectedStrategyId, setSelectedStrategyId] = useState<string | null>(null)

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
        quality_tags_filter: selectedTags.size > 0 ? Array.from(selectedTags) : undefined,
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
    selectedTags,
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

  // Load saved strategies on mount (client-side only)
  useEffect(() => {
    setSavedStrategies(loadStrategies())
  }, [])

  // Apply a saved strategy to all filters
  const applyStrategy = (strategy: SavedStrategy) => {
    const s = strategy.settings
    if (!s) return

    // Survival gates
    setRequirePiotroski(s.piotroski_enabled ?? false)
    setPiotroskiMin(s.piotroski_min ?? 6)
    setRequireAltman(s.altman_enabled ?? false)
    setAltmanZone((s.altman_zone as "safe" | "grey" | "distress") ?? "safe")

    // Quality
    setQualityFilter(s.quality_enabled ?? false)
    setMinQuality((s.min_quality as QualityLabel) ?? "compounder")

    // Build tag set from required tags only
    const tagSet = new Set<QualityTag>()
    if (s.required_tags && Array.isArray(s.required_tags)) {
      s.required_tags.forEach(tag => {
        if (QUALITY_TAGS.includes(tag as QualityTag)) {
          tagSet.add(tag as QualityTag)
        }
      })
    }
    setSelectedTags(tagSet)

    // Valuation lenses
    setLensGraham(s.graham_enabled ?? false)
    setGrahamMode((s.graham_mode as GrahamMode) ?? "strict")
    setGrahamMin(s.graham_min ?? 8)
    setLensMagicFormula(s.magic_formula_enabled ?? false)
    setMfTopPct(s.mf_top_pct ?? 20)
    setLensPeg(s.peg_enabled ?? false)
    setMaxPeg(s.max_peg ?? 1.5)
    setLensNetNet(s.net_net_enabled ?? false)
    setLensFamaFrenchBm(s.fama_french_enabled ?? false)
    setFfBmTopPct(s.ff_top_pct ?? 30)
    setMinLenses(s.min_lenses ?? 3)
    setStrictMode(s.strict_mode ?? false)

    setSelectedStrategyId(strategy.id)
  }

  // Clear strategy and reset to defaults
  const clearStrategy = () => {
    setSelectedStrategyId(null)
    // Reset to defaults
    setRequireAltman(true)
    setAltmanZone("safe")
    setRequirePiotroski(true)
    setPiotroskiMin(6)
    setQualityFilter(true)
    setMinQuality("compounder")
    setSelectedTags(new Set())
    setMinLenses(3)
    setStrictMode(false)
    setLensGraham(true)
    setLensNetNet(true)
    setLensPeg(true)
    setLensMagicFormula(true)
    setLensFamaFrenchBm(true)
    setGrahamMode("strict")
    setGrahamMin(8)
    setMaxPeg(1.5)
    setMfTopPct(20)
    setFfBmTopPct(30)
  }

  // Delete a saved strategy
  const handleDeleteStrategy = (id: string) => {
    deleteStrategy(id)
    setSavedStrategies(loadStrategies())
    if (selectedStrategyId === id) {
      setSelectedStrategyId(null)
    }
  }

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

      {/* Saved Strategy Selector */}
      {savedStrategies.length > 0 && (
        <div className="mb-6 p-4 bg-indigo-50 dark:bg-indigo-950/50 rounded-lg border border-indigo-200 dark:border-indigo-800">
          <div className="flex items-center justify-between flex-wrap gap-3">
            <div className="flex items-center gap-3">
              <span className="text-sm font-medium text-indigo-700 dark:text-indigo-300">
                Load Strategy:
              </span>
              <select
                value={selectedStrategyId || ""}
                onChange={(e) => {
                  const strategy = savedStrategies.find(s => s.id === e.target.value)
                  if (strategy) applyStrategy(strategy)
                }}
                className="text-sm rounded-lg border-indigo-300 dark:border-indigo-700 dark:bg-gray-800 focus:ring-indigo-500"
              >
                <option value="">Select saved strategy...</option>
                {savedStrategies.map((strategy) => (
                  <option key={strategy.id} value={strategy.id}>
                    {strategy.name} (Alpha: {strategy.expected_alpha.toFixed(1)}%)
                  </option>
                ))}
              </select>
            </div>
            <div className="flex items-center gap-2">
              {selectedStrategyId && (
                <>
                  <span className="text-xs text-indigo-600 dark:text-indigo-400">
                    {(() => {
                      const s = savedStrategies.find(s => s.id === selectedStrategyId)
                      return s ? `${s.holding_period}Q hold • ${formatStrategyDate(s.created_at)}` : ""
                    })()}
                  </span>
                  <button
                    onClick={clearStrategy}
                    className="text-sm px-3 py-1 rounded bg-gray-200 dark:bg-gray-700 text-gray-700 dark:text-gray-300 hover:bg-gray-300 dark:hover:bg-gray-600"
                  >
                    Clear
                  </button>
                  <button
                    onClick={() => {
                      if (selectedStrategyId && confirm("Delete this strategy?")) {
                        handleDeleteStrategy(selectedStrategyId)
                      }
                    }}
                    className="text-sm px-3 py-1 rounded bg-red-100 dark:bg-red-900 text-red-700 dark:text-red-300 hover:bg-red-200 dark:hover:bg-red-800"
                  >
                    Delete
                  </button>
                </>
              )}
            </div>
          </div>
          {selectedStrategyId && (
            <div className="mt-2 text-xs text-indigo-600 dark:text-indigo-400">
              {(() => {
                const s = savedStrategies.find(s => s.id === selectedStrategyId)
                if (!s) return null
                return (
                  <span>
                    Expected Alpha: {s.expected_alpha.toFixed(1)}%
                    (95% CI: {s.expected_alpha_ci_lower.toFixed(1)}% to {s.expected_alpha_ci_upper.toFixed(1)}%)
                    • Win Rate: {s.win_rate.toFixed(0)}%
                    • Sample: {s.sample_size.toLocaleString()} stocks
                  </span>
                )
              })()}
            </div>
          )}
        </div>
      )}

      {/* Stage 1: Survival Gates */}
      <SurvivalGates
        requireAltman={requireAltman}
        setRequireAltman={setRequireAltman}
        altmanZone={altmanZone}
        setAltmanZone={setAltmanZone}
        requirePiotroski={requirePiotroski}
        setRequirePiotroski={setRequirePiotroski}
        piotroskiMin={piotroskiMin}
        setPiotroskiMin={setPiotroskiMin}
      />

      {/* Stage 2: Quality Classification */}
      <QualityClassification
        qualityFilter={qualityFilter}
        setQualityFilter={setQualityFilter}
        minQuality={minQuality}
        setMinQuality={setMinQuality}
        selectedTags={selectedTags}
        setSelectedTags={setSelectedTags}
      />

      {/* Stage 3: Valuation Lenses */}
      <ValuationLenses
        minLenses={minLenses}
        setMinLenses={setMinLenses}
        strictMode={strictMode}
        setStrictMode={setStrictMode}
        lensGraham={lensGraham}
        setLensGraham={setLensGraham}
        lensNetNet={lensNetNet}
        setLensNetNet={setLensNetNet}
        lensPeg={lensPeg}
        setLensPeg={setLensPeg}
        lensMagicFormula={lensMagicFormula}
        setLensMagicFormula={setLensMagicFormula}
        lensFamaFrenchBm={lensFamaFrenchBm}
        setLensFamaFrenchBm={setLensFamaFrenchBm}
        grahamMode={grahamMode}
        setGrahamMode={setGrahamMode}
        grahamMin={grahamMin}
        setGrahamMin={setGrahamMin}
        maxPeg={maxPeg}
        setMaxPeg={setMaxPeg}
        mfTopPct={mfTopPct}
        setMfTopPct={setMfTopPct}
        ffBmTopPct={ffBmTopPct}
        setFfBmTopPct={setFfBmTopPct}
        showAdvanced={showAdvanced}
        setShowAdvanced={setShowAdvanced}
      />

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
                  Tags
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
                    <QualityTagsChips tags={stock.quality_tags} />
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
