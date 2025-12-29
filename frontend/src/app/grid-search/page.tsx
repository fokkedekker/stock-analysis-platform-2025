"use client";

import { useState, useEffect, useCallback } from "react";
import {
  RiPlayLine,
  RiEyeLine,
  RiSearchLine,
  RiCheckLine,
  RiCloseLine,
  RiArrowLeftLine,
  RiDownloadLine,
  RiTimeLine,
  RiStockLine,
  RiPercentLine,
  RiTrophyLine,
  RiStopLine,
  RiHistoryLine,
  RiArrowRightLine,
} from "@remixicon/react";
import {
  getAvailableQuarters,
  previewGridSearch,
  startGridSearch,
  getGridSearchResults,
  createProgressEventSource,
  cancelGridSearch,
  getSearchHistory,
  type GridDimension,
  type PreviewResponse,
  type GridSearchProgress,
  type GridSearchResults,
  type SearchHistoryItem,
  type StrategyConfig,
} from "@/lib/gridsearch-api";
import { QualityLabel, QualityTag } from "@/lib/api";
import { SurvivalGates } from "@/components/strategy/SurvivalGates";
import { QualityClassification } from "@/components/strategy/QualityClassification";
import { ValuationLenses, GrahamMode } from "@/components/strategy/ValuationLenses";
import { TestVariablesSection } from "@/components/strategy/TestVariablesSection";

type ViewState = "configure" | "running" | "results";

function formatDuration(seconds: number): string {
  if (seconds < 60) return `${Math.round(seconds)}s`;
  const mins = Math.floor(seconds / 60);
  const secs = Math.round(seconds % 60);
  return `${mins}m ${secs}s`;
}

function formatPercent(value: number): string {
  const sign = value >= 0 ? "+" : "";
  return `${sign}${value.toFixed(2)}%`;
}

function formatQuarter(q: string): string {
  const match = q.match(/^(\d{4})Q(\d)$/);
  if (match) {
    return `Q${match[2]} ${match[1]}`;
  }
  return q;
}

// Builder Component
function GridSearchBuilder({
  onStart,
}: {
  onStart: (searchId: string, totalSimulations: number) => void;
}) {
  // Base strategy state - matches Pipeline page
  const [requireAltman, setRequireAltman] = useState(true);
  const [altmanZone, setAltmanZone] = useState<"safe" | "grey" | "distress">("safe");
  const [requirePiotroski, setRequirePiotroski] = useState(true);
  const [piotroskiMin, setPiotroskiMin] = useState(6);

  const [qualityFilter, setQualityFilter] = useState(true);
  const [minQuality, setMinQuality] = useState<QualityLabel>("compounder");
  const [selectedTags, setSelectedTags] = useState<Set<QualityTag>>(new Set());

  const [minLenses, setMinLenses] = useState(3);
  const [strictMode, setStrictMode] = useState(false);
  const [lensGraham, setLensGraham] = useState(true);
  const [lensNetNet, setLensNetNet] = useState(true);
  const [lensPeg, setLensPeg] = useState(true);
  const [lensMagicFormula, setLensMagicFormula] = useState(true);
  const [lensFamaFrenchBm, setLensFamaFrenchBm] = useState(true);
  const [grahamMode, setGrahamMode] = useState<GrahamMode>("strict");
  const [grahamMin, setGrahamMin] = useState(6);
  const [maxPeg, setMaxPeg] = useState(1.5);
  const [mfTopPct, setMfTopPct] = useState(20);
  const [ffBmTopPct, setFfBmTopPct] = useState(30);
  const [showAdvanced, setShowAdvanced] = useState(true);

  // Test variables state
  const [testVariables, setTestVariables] = useState<Map<string, (string | number | boolean)[]>>(new Map());

  // Simulation config state
  const [availableQuarters, setAvailableQuarters] = useState<string[]>([]);
  const [selectedQuarters, setSelectedQuarters] = useState<string[]>([]);
  const [holdingPeriods, setHoldingPeriods] = useState<number[]>([1, 2, 3, 4]);
  const [preview, setPreview] = useState<PreviewResponse | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Fetch available quarters on mount
  useEffect(() => {
    getAvailableQuarters()
      .then((data) => {
        setAvailableQuarters(data.recommended || data.analysis_quarters);
        setSelectedQuarters(data.recommended?.slice(0, 4) || []);
      })
      .catch((err) => setError(err.message));
  }, []);

  // Toggle quarter selection
  const toggleQuarter = (quarter: string) => {
    setSelectedQuarters((prev) =>
      prev.includes(quarter) ? prev.filter((q) => q !== quarter) : [...prev, quarter]
    );
    setPreview(null);
  };

  // Toggle holding period
  const toggleHoldingPeriod = (period: number) => {
    setHoldingPeriods((prev) =>
      prev.includes(period) ? prev.filter((p) => p !== period) : [...prev, period].sort()
    );
    setPreview(null);
  };

  // Build base strategy config
  const buildBaseStrategy = useCallback((): StrategyConfig => {
    return {
      survival: {
        altman_enabled: requireAltman,
        altman_zone: altmanZone,
        piotroski_enabled: requirePiotroski,
        piotroski_min: piotroskiMin,
      },
      quality: {
        enabled: qualityFilter,
        min_quality: minQuality,
        required_tags: Array.from(selectedTags),
        excluded_tags: [],
      },
      valuation: {
        graham_enabled: lensGraham,
        graham_mode: grahamMode,
        graham_min: grahamMin,
        magic_formula_enabled: lensMagicFormula,
        mf_top_pct: mfTopPct,
        peg_enabled: lensPeg,
        max_peg: maxPeg,
        net_net_enabled: lensNetNet,
        fama_french_enabled: lensFamaFrenchBm,
        ff_top_pct: ffBmTopPct,
        min_lenses: minLenses,
        strict_mode: strictMode,
      },
    };
  }, [
    requireAltman, altmanZone, requirePiotroski, piotroskiMin,
    qualityFilter, minQuality, selectedTags,
    lensGraham, grahamMode, grahamMin, lensMagicFormula, mfTopPct,
    lensPeg, maxPeg, lensNetNet, lensFamaFrenchBm, ffBmTopPct,
    minLenses, strictMode,
  ]);

  // Build dimensions from test variables
  const buildDimensions = useCallback((): GridDimension[] => {
    const dimensions: GridDimension[] = [];
    testVariables.forEach((values, name) => {
      if (values.length > 0) {
        dimensions.push({ name, values });
      }
    });
    return dimensions;
  }, [testVariables]);

  // Preview handler
  const handlePreview = async () => {
    if (selectedQuarters.length === 0) {
      setError("Select at least one quarter");
      return;
    }
    if (holdingPeriods.length === 0) {
      setError("Select at least one holding period");
      return;
    }

    setIsLoading(true);
    setError(null);

    try {
      const dimensions = buildDimensions();
      const result = await previewGridSearch({
        dimensions,
        quarters: selectedQuarters,
        holding_periods: holdingPeriods,
      });
      setPreview(result);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Preview failed");
    } finally {
      setIsLoading(false);
    }
  };

  // Start handler
  const handleStart = async () => {
    if (!preview) {
      await handlePreview();
      return;
    }

    setIsLoading(true);
    setError(null);

    try {
      const dimensions = buildDimensions();
      const baseStrategy = buildBaseStrategy();

      // Debug: log what we're sending
      console.log("Starting simulation with dimensions:", dimensions);
      console.log("Test variables map:", Array.from(testVariables.entries()));

      const result = await startGridSearch({
        base_strategy: baseStrategy,
        dimensions,
        quarters: selectedQuarters,
        holding_periods: holdingPeriods,
      });

      onStart(result.search_id, result.total_simulations);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to start");
      setIsLoading(false);
    }
  };

  // Clear preview when test variables change
  useEffect(() => {
    setPreview(null);
  }, [testVariables]);

  return (
    <div className="space-y-6">
      {error && (
        <div className="bg-red-50 dark:bg-red-950 border border-red-200 dark:border-red-800 rounded-lg p-4 text-red-700 dark:text-red-300">
          {error}
        </div>
      )}

      {/* Base Strategy Section */}
      <div className="border-2 border-indigo-200 dark:border-indigo-800 rounded-lg p-4">
        <h2 className="text-lg font-semibold mb-4 text-indigo-700 dark:text-indigo-300">
          Base Strategy (Constant Values)
        </h2>
        <p className="text-sm text-gray-500 dark:text-gray-400 mb-4">
          These settings stay fixed for all simulations. Only variables added to "Test" below will vary.
        </p>

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

        <QualityClassification
          qualityFilter={qualityFilter}
          setQualityFilter={setQualityFilter}
          minQuality={minQuality}
          setMinQuality={setMinQuality}
          selectedTags={selectedTags}
          setSelectedTags={setSelectedTags}
        />

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
      </div>

      {/* Test Variables Section */}
      <TestVariablesSection
        selectedVariables={testVariables}
        setSelectedVariables={setTestVariables}
      />

      {/* Quarter Selection */}
      <div className="bg-white dark:bg-gray-900 rounded-lg border border-gray-200 dark:border-gray-800 p-6">
        <h2 className="text-lg font-semibold mb-4">Buy Quarters</h2>
        <p className="text-sm text-gray-500 dark:text-gray-400 mb-4">
          Select which quarters to test buying stocks in.
        </p>
        <div className="flex flex-wrap gap-2">
          {availableQuarters.map((q) => (
            <button
              key={q}
              onClick={() => toggleQuarter(q)}
              className={`px-3 py-1.5 text-sm rounded-lg border transition-colors ${
                selectedQuarters.includes(q)
                  ? "bg-indigo-100 dark:bg-indigo-900 border-indigo-300 dark:border-indigo-700 text-indigo-700 dark:text-indigo-300"
                  : "bg-gray-50 dark:bg-gray-800 border-gray-200 dark:border-gray-700 text-gray-700 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-700"
              }`}
            >
              {selectedQuarters.includes(q) && <RiCheckLine className="inline w-4 h-4 mr-1" />}
              {formatQuarter(q)}
            </button>
          ))}
        </div>
      </div>

      {/* Holding Periods */}
      <div className="bg-white dark:bg-gray-900 rounded-lg border border-gray-200 dark:border-gray-800 p-6">
        <h2 className="text-lg font-semibold mb-4">Holding Periods</h2>
        <p className="text-sm text-gray-500 dark:text-gray-400 mb-4">
          How long to hold stocks before measuring returns.
        </p>
        <div className="flex gap-4">
          {[1, 2, 3, 4].map((period) => (
            <button
              key={period}
              onClick={() => toggleHoldingPeriod(period)}
              className={`px-4 py-2 text-sm rounded-lg border transition-colors ${
                holdingPeriods.includes(period)
                  ? "bg-indigo-100 dark:bg-indigo-900 border-indigo-300 dark:border-indigo-700 text-indigo-700 dark:text-indigo-300"
                  : "bg-gray-50 dark:bg-gray-800 border-gray-200 dark:border-gray-700 text-gray-700 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-700"
              }`}
            >
              {holdingPeriods.includes(period) && <RiCheckLine className="inline w-4 h-4 mr-1" />}
              {period}Q
            </button>
          ))}
        </div>
      </div>

      {/* Preview Panel */}
      {preview && (
        <div className="bg-indigo-50 dark:bg-indigo-950 rounded-lg border border-indigo-200 dark:border-indigo-800 p-6">
          <div className="grid grid-cols-3 gap-6 text-center">
            <div>
              <div className="text-3xl font-bold text-indigo-600 dark:text-indigo-400">
                {preview.strategy_count.toLocaleString()}
              </div>
              <div className="text-sm text-gray-600 dark:text-gray-400">Strategies</div>
            </div>
            <div>
              <div className="text-3xl font-bold text-indigo-600 dark:text-indigo-400">
                {preview.total_simulations.toLocaleString()}
              </div>
              <div className="text-sm text-gray-600 dark:text-gray-400">Simulations</div>
            </div>
            <div>
              <div className="text-3xl font-bold text-indigo-600 dark:text-indigo-400">
                ~{formatDuration(preview.estimated_seconds)}
              </div>
              <div className="text-sm text-gray-600 dark:text-gray-400">Est. Runtime</div>
            </div>
          </div>
        </div>
      )}

      {/* Action Buttons */}
      <div className="flex gap-4">
        <button
          onClick={handlePreview}
          disabled={isLoading}
          className="flex items-center gap-2 px-4 py-2 bg-gray-100 dark:bg-gray-800 text-gray-700 dark:text-gray-300 rounded-lg hover:bg-gray-200 dark:hover:bg-gray-700 disabled:opacity-50"
        >
          <RiEyeLine className="w-5 h-5" />
          Preview
        </button>
        <button
          onClick={handleStart}
          disabled={isLoading || selectedQuarters.length === 0}
          className="flex items-center gap-2 px-6 py-2 bg-indigo-600 text-white rounded-lg hover:bg-indigo-700 disabled:opacity-50"
        >
          <RiPlayLine className="w-5 h-5" />
          {isLoading ? "Starting..." : "Start Simulation"}
        </button>
      </div>
    </div>
  );
}

// Progress Component
function GridSearchProgressView({
  searchId,
  totalSimulations,
  onComplete,
  onCancel,
}: {
  searchId: string;
  totalSimulations: number;
  onComplete: () => void;
  onCancel: () => void;
}) {
  const [progress, setProgress] = useState<GridSearchProgress | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [isCancelling, setIsCancelling] = useState(false);

  const handleCancel = async () => {
    setIsCancelling(true);
    try {
      await cancelGridSearch(searchId);
      onCancel();
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed to cancel");
      setIsCancelling(false);
    }
  };

  useEffect(() => {
    const eventSource = createProgressEventSource(searchId);

    eventSource.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);

        if (data.completed === true) {
          eventSource.close();
          onComplete();
        } else if (data.status === "cancelled" || data.status === "cancelling") {
          eventSource.close();
          onCancel();
        } else if (data.status === "completed") {
          eventSource.close();
          onComplete();
        } else if (data.error || data.failed) {
          setError(data.error || "Search failed");
          eventSource.close();
        } else {
          setProgress(data);
        }
      } catch (e) {
        console.error("Failed to parse progress:", e);
      }
    };

    eventSource.onerror = () => {
      setError("Connection lost. Please refresh.");
      eventSource.close();
    };

    return () => eventSource.close();
  }, [searchId, onComplete, onCancel]);

  const completed = progress?.completed || 0;
  const percentComplete = totalSimulations > 0 ? (completed / totalSimulations) * 100 : 0;

  return (
    <div className="bg-white dark:bg-gray-900 rounded-lg border border-gray-200 dark:border-gray-800 p-8">
      {error ? (
        <div className="text-center text-red-600 dark:text-red-400">
          <RiCloseLine className="w-12 h-12 mx-auto mb-4" />
          <p>{error}</p>
        </div>
      ) : (
        <div className="space-y-6">
          <div className="text-center">
            <RiSearchLine className="w-12 h-12 mx-auto mb-4 text-indigo-600 dark:text-indigo-400 animate-pulse" />
            <h2 className="text-xl font-semibold">Running Simulation</h2>
          </div>

          <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-4 overflow-hidden">
            <div
              className="h-full bg-indigo-600 transition-all duration-300 ease-out"
              style={{ width: `${percentComplete}%` }}
            />
          </div>

          <div className="flex justify-between text-sm text-gray-600 dark:text-gray-400">
            <span>
              {completed.toLocaleString()} / {totalSimulations.toLocaleString()} simulations
            </span>
            <span>
              {progress?.estimated_remaining_seconds
                ? `~${formatDuration(progress.estimated_remaining_seconds)} remaining`
                : "Calculating..."}
            </span>
          </div>

          {progress?.current_strategy && (
            <div className="text-sm text-center text-gray-500 dark:text-gray-400">
              Testing: <span className="font-mono">{progress.current_strategy}</span>
              {progress.current_quarter && ` (${formatQuarter(progress.current_quarter)})`}
            </div>
          )}

          <div className="flex justify-center pt-4">
            <button
              onClick={handleCancel}
              disabled={isCancelling}
              className="flex items-center gap-2 px-4 py-2 bg-red-100 dark:bg-red-900 text-red-700 dark:text-red-300 rounded-lg hover:bg-red-200 dark:hover:bg-red-800 disabled:opacity-50 transition-colors"
            >
              <RiStopLine className="w-5 h-5" />
              {isCancelling ? "Cancelling..." : "Cancel Simulation"}
            </button>
          </div>
        </div>
      )}
    </div>
  );
}

// Results Component
function GridSearchResultsView({
  searchId,
  onNewSearch,
}: {
  searchId: string;
  onNewSearch: () => void;
}) {
  const [results, setResults] = useState<GridSearchResults | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [holdingPeriodFilter, setHoldingPeriodFilter] = useState<number | "all">("all");

  useEffect(() => {
    getGridSearchResults(searchId)
      .then(setResults)
      .catch((err) => setError(err.message))
      .finally(() => setIsLoading(false));
  }, [searchId]);

  if (isLoading) {
    return (
      <div className="text-center py-12">
        <RiSearchLine className="w-12 h-12 mx-auto mb-4 animate-spin text-indigo-600" />
        <p>Loading results...</p>
      </div>
    );
  }

  if (error || !results) {
    return (
      <div className="text-center py-12 text-red-600">
        <RiCloseLine className="w-12 h-12 mx-auto mb-4" />
        <p>{error || "Failed to load results"}</p>
      </div>
    );
  }

  // Get unique holding periods from results
  const availableHoldingPeriods = Array.from(
    new Set((results.best_by_alpha || []).map(r => r.holding_period))
  ).sort();

  // Filter results by holding period
  const filteredByAlpha = holdingPeriodFilter === "all"
    ? results.best_by_alpha || []
    : (results.best_by_alpha || []).filter(r => r.holding_period === holdingPeriodFilter);

  const bestByAlpha = filteredByAlpha.slice(0, 20);
  const avgAlpha =
    bestByAlpha.length > 0
      ? bestByAlpha.reduce((sum, r) => sum + r.alpha, 0) / bestByAlpha.length
      : 0;
  const totalSimulations = results.total_simulations ?? 0;
  const durationSeconds = results.duration_seconds ?? 0;

  // Build strategy + holding period combinations from by_strategy data (has proper averages)
  const strategyHoldingCombos = (() => {
    const byStrategy = results.by_strategy || [];
    const combos: {
      strategy_name: string;
      holding_period: number;
      avg_alpha: number;
      avg_return: number;
      avg_win_rate: number;
      avg_stock_count: number;
      quarters_tested: number;
    }[] = [];

    // Flatten: each strategy × each holding period = one row
    byStrategy.forEach(strategy => {
      const holdingPeriods = strategy.by_holding_period || [];
      holdingPeriods.forEach(hp => {
        // Filter by holding period if selected
        if (holdingPeriodFilter !== "all" && hp.holding_period !== holdingPeriodFilter) {
          return;
        }
        combos.push({
          strategy_name: strategy.strategy_name,
          holding_period: hp.holding_period,
          avg_alpha: hp.avg_alpha,
          avg_return: hp.avg_return,
          avg_win_rate: strategy.avg_win_rate, // Use strategy-level since not in hp
          avg_stock_count: strategy.avg_stock_count, // Use strategy-level since not in hp
          quarters_tested: hp.simulation_count,
        });
      });
    });

    return combos.sort((a, b) => b.avg_alpha - a.avg_alpha);
  })();

  return (
    <div className="space-y-6">
      {/* Holding Period Filter */}
      {availableHoldingPeriods.length > 1 && (
        <div className="bg-white dark:bg-gray-900 rounded-lg border border-gray-200 dark:border-gray-800 p-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <span className="text-sm font-medium text-gray-700 dark:text-gray-300">
                Filter by Holding Period:
              </span>
              <div className="flex gap-2">
                <button
                  onClick={() => setHoldingPeriodFilter("all")}
                  className={`px-3 py-1.5 text-sm rounded-lg border transition-colors ${
                    holdingPeriodFilter === "all"
                      ? "bg-indigo-100 dark:bg-indigo-900 border-indigo-300 dark:border-indigo-700 text-indigo-700 dark:text-indigo-300"
                      : "bg-gray-50 dark:bg-gray-800 border-gray-200 dark:border-gray-700 text-gray-700 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-700"
                  }`}
                >
                  All
                </button>
                {availableHoldingPeriods.map((period) => (
                  <button
                    key={period}
                    onClick={() => setHoldingPeriodFilter(period)}
                    className={`px-3 py-1.5 text-sm rounded-lg border transition-colors ${
                      holdingPeriodFilter === period
                        ? "bg-indigo-100 dark:bg-indigo-900 border-indigo-300 dark:border-indigo-700 text-indigo-700 dark:text-indigo-300"
                        : "bg-gray-50 dark:bg-gray-800 border-gray-200 dark:border-gray-700 text-gray-700 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-700"
                    }`}
                  >
                    {period}Q
                  </button>
                ))}
              </div>
            </div>
            {holdingPeriodFilter !== "all" && (
              <span className="text-sm text-indigo-600 dark:text-indigo-400">
                Showing {holdingPeriodFilter}-quarter hold results only
              </span>
            )}
          </div>
        </div>
      )}

      {/* Winner Card - Best Strategy + Holding Period Combo (averaged) */}
      {strategyHoldingCombos.length > 0 && (() => {
        const best = strategyHoldingCombos[0];
        return (
          <div className="bg-gradient-to-r from-green-50 to-emerald-50 dark:from-green-950 dark:to-emerald-950 rounded-lg border-2 border-green-300 dark:border-green-700 p-6">
            <div className="flex items-center gap-2 mb-3">
              <RiTrophyLine className="w-6 h-6 text-green-600" />
              <span className="text-lg font-semibold text-green-800 dark:text-green-200">
                Recommended: Best Strategy + Holding Period
              </span>
              <span className="text-sm text-gray-500">(averaged across {best.quarters_tested} buy quarters)</span>
            </div>
            <div className="grid grid-cols-4 gap-6">
              <div>
                <div className="text-sm text-gray-600 dark:text-gray-400 mb-1">Avg Alpha</div>
                <div className={`text-3xl font-bold ${best.avg_alpha >= 0 ? "text-green-600" : "text-red-600"}`}>
                  {formatPercent(best.avg_alpha)}
                </div>
              </div>
              <div>
                <div className="text-sm text-gray-600 dark:text-gray-400 mb-1">Hold For</div>
                <div className="text-2xl font-bold text-gray-900 dark:text-gray-100">
                  {best.holding_period} Quarter{best.holding_period > 1 ? "s" : ""}
                </div>
              </div>
              <div>
                <div className="text-sm text-gray-600 dark:text-gray-400 mb-1">Avg Return</div>
                <div className={`text-2xl font-bold ${best.avg_return >= 0 ? "text-green-600" : "text-red-600"}`}>
                  {formatPercent(best.avg_return)}
                </div>
              </div>
              <div>
                <div className="text-sm text-gray-600 dark:text-gray-400 mb-1">Avg Stocks / Win Rate</div>
                <div className="text-2xl font-bold text-gray-900 dark:text-gray-100">
                  {Math.round(best.avg_stock_count)} / {best.avg_win_rate.toFixed(0)}%
                </div>
              </div>
            </div>
            <div className="mt-4 pt-4 border-t border-green-200 dark:border-green-800">
              <div className="text-sm text-gray-600 dark:text-gray-400 mb-1">Strategy Settings</div>
              <div className="font-mono text-sm text-gray-800 dark:text-gray-200 bg-white/50 dark:bg-black/20 rounded px-3 py-2">
                {best.strategy_name}
              </div>
            </div>
          </div>
        );
      })()}

      {/* Summary Cards */}
      <div className="grid grid-cols-3 gap-4">
        <div className="bg-white dark:bg-gray-900 rounded-lg border border-gray-200 dark:border-gray-800 p-4">
          <div className="flex items-center gap-2 text-gray-500 dark:text-gray-400 mb-1">
            <RiPercentLine className="w-5 h-5" />
            <span className="text-sm">Avg Alpha (Top 20)</span>
          </div>
          <div className={`text-2xl font-bold ${avgAlpha >= 0 ? "text-green-600" : "text-red-600"}`}>
            {formatPercent(avgAlpha)}
          </div>
        </div>

        <div className="bg-white dark:bg-gray-900 rounded-lg border border-gray-200 dark:border-gray-800 p-4">
          <div className="flex items-center gap-2 text-gray-500 dark:text-gray-400 mb-1">
            <RiStockLine className="w-5 h-5" />
            <span className="text-sm">Total Simulations</span>
          </div>
          <div className="text-2xl font-bold text-gray-900 dark:text-gray-100">
            {totalSimulations.toLocaleString()}
          </div>
        </div>

        <div className="bg-white dark:bg-gray-900 rounded-lg border border-gray-200 dark:border-gray-800 p-4">
          <div className="flex items-center gap-2 text-gray-500 dark:text-gray-400 mb-1">
            <RiTimeLine className="w-5 h-5" />
            <span className="text-sm">Duration</span>
          </div>
          <div className="text-2xl font-bold text-gray-900 dark:text-gray-100">
            {formatDuration(durationSeconds)}
          </div>
        </div>
      </div>

      {/* Strategy + Holding Period Performance (the actionable table) */}
      {strategyHoldingCombos.length > 0 && (
        <div className="bg-white dark:bg-gray-900 rounded-lg border border-gray-200 dark:border-gray-800 overflow-hidden">
          <div className="px-6 py-4 border-b border-gray-200 dark:border-gray-800">
            <h2 className="text-lg font-semibold">
              Strategy + Holding Period Rankings
            </h2>
            <p className="text-sm text-gray-500 mt-1">
              Each row = one strategy + one holding period, averaged across all buy quarters tested
            </p>
            <div className="mt-2 text-xs text-gray-400 flex flex-wrap gap-x-4 gap-y-1">
              <span><strong>Hold:</strong> quarters to hold</span>
              <span><strong>Quarters:</strong> # of buy quarters tested</span>
              <span><strong>Avg Alpha:</strong> return minus S&P 500 (positive = beat market)</span>
              <span><strong>Win Rate:</strong> % of stocks with positive returns</span>
            </div>
          </div>
          <div className="overflow-x-auto">
            <table className="w-full">
              <thead className="bg-gray-50 dark:bg-gray-800">
                <tr>
                  <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">Strategy</th>
                  <th className="px-4 py-3 text-center text-xs font-medium text-gray-500 uppercase">Hold</th>
                  <th className="px-4 py-3 text-center text-xs font-medium text-gray-500 uppercase">Quarters</th>
                  <th className="px-4 py-3 text-right text-xs font-medium text-gray-500 uppercase">Avg Stocks</th>
                  <th className="px-4 py-3 text-right text-xs font-medium text-gray-500 uppercase">Avg Return</th>
                  <th className="px-4 py-3 text-right text-xs font-medium text-gray-500 uppercase">Avg Alpha</th>
                  <th className="px-4 py-3 text-right text-xs font-medium text-gray-500 uppercase">Win Rate</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-gray-200 dark:divide-gray-800">
                {strategyHoldingCombos.slice(0, 20).map((s, i) => (
                  <tr key={i} className={`hover:bg-gray-50 dark:hover:bg-gray-800 ${i === 0 ? "bg-green-50 dark:bg-green-900/20" : ""}`}>
                    <td className="px-4 py-3 text-sm font-mono text-gray-900 dark:text-gray-100">
                      {s.strategy_name}
                    </td>
                    <td className="px-4 py-3 text-sm text-center">
                      <span className="inline-flex items-center px-2 py-1 rounded bg-indigo-100 dark:bg-indigo-900 text-indigo-700 dark:text-indigo-300 font-semibold">
                        {s.holding_period}Q
                      </span>
                    </td>
                    <td className="px-4 py-3 text-sm text-center text-gray-600 dark:text-gray-400">
                      {s.quarters_tested}
                    </td>
                    <td className="px-4 py-3 text-sm text-right text-gray-600 dark:text-gray-400">
                      {Math.round(s.avg_stock_count)}
                    </td>
                    <td className={`px-4 py-3 text-sm text-right font-medium ${s.avg_return >= 0 ? "text-green-600" : "text-red-600"}`}>
                      {formatPercent(s.avg_return)}
                    </td>
                    <td className={`px-4 py-3 text-sm text-right font-bold ${s.avg_alpha >= 0 ? "text-green-600" : "text-red-600"}`}>
                      {formatPercent(s.avg_alpha)}
                    </td>
                    <td className="px-4 py-3 text-sm text-right text-gray-600 dark:text-gray-400">
                      {s.avg_win_rate.toFixed(1)}%
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* Top Individual Simulations */}
      <div className="bg-white dark:bg-gray-900 rounded-lg border border-gray-200 dark:border-gray-800 overflow-hidden">
        <div className="px-6 py-4 border-b border-gray-200 dark:border-gray-800">
          <h2 className="text-lg font-semibold">
            Top 20 Individual Simulations by Alpha
            {holdingPeriodFilter !== "all" && ` (${holdingPeriodFilter}Q Hold)`}
          </h2>
          <p className="text-sm text-gray-500 mt-1">
            {holdingPeriodFilter === "all"
              ? "Best single quarter/strategy combinations"
              : `Best ${holdingPeriodFilter}-quarter hold combinations`}
          </p>
        </div>
        <div className="overflow-x-auto">
          <table className="w-full">
            <thead className="bg-gray-50 dark:bg-gray-800">
              <tr>
                <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">
                  Strategy
                </th>
                <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">
                  Quarter
                </th>
                {holdingPeriodFilter === "all" && (
                  <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">
                    Hold
                  </th>
                )}
                <th className="px-4 py-3 text-right text-xs font-medium text-gray-500 uppercase">
                  Stocks
                </th>
                <th className="px-4 py-3 text-right text-xs font-medium text-gray-500 uppercase">
                  Return
                </th>
                <th className="px-4 py-3 text-right text-xs font-medium text-gray-500 uppercase">
                  Benchmark
                </th>
                <th className="px-4 py-3 text-right text-xs font-medium text-gray-500 uppercase">
                  Alpha
                </th>
                <th className="px-4 py-3 text-right text-xs font-medium text-gray-500 uppercase">
                  Win Rate
                </th>
              </tr>
            </thead>
            <tbody className="divide-y divide-gray-200 dark:divide-gray-800">
              {bestByAlpha.length === 0 ? (
                <tr>
                  <td colSpan={holdingPeriodFilter === "all" ? 8 : 7} className="px-4 py-8 text-center text-gray-500">
                    No strategies found with matching stocks. Try different settings.
                  </td>
                </tr>
              ) : bestByAlpha.map((r, i) => (
                <tr key={i} className="hover:bg-gray-50 dark:hover:bg-gray-800">
                  <td className="px-4 py-3 text-sm font-mono text-gray-900 dark:text-gray-100 max-w-xs truncate">
                    {r.strategy_name}
                  </td>
                  <td className="px-4 py-3 text-sm text-gray-600 dark:text-gray-400">
                    {formatQuarter(r.buy_quarter)}
                  </td>
                  {holdingPeriodFilter === "all" && (
                    <td className="px-4 py-3 text-sm text-gray-600 dark:text-gray-400">
                      {r.holding_period}Q
                    </td>
                  )}
                  <td className="px-4 py-3 text-sm text-right text-gray-600 dark:text-gray-400">
                    {r.stock_count}
                  </td>
                  <td
                    className={`px-4 py-3 text-sm text-right font-medium ${
                      r.portfolio_return >= 0 ? "text-green-600" : "text-red-600"
                    }`}
                  >
                    {formatPercent(r.portfolio_return)}
                  </td>
                  <td className="px-4 py-3 text-sm text-right text-gray-600 dark:text-gray-400">
                    {formatPercent(r.benchmark_return)}
                  </td>
                  <td
                    className={`px-4 py-3 text-sm text-right font-bold ${
                      r.alpha >= 0 ? "text-green-600" : "text-red-600"
                    }`}
                  >
                    {formatPercent(r.alpha)}
                  </td>
                  <td className="px-4 py-3 text-sm text-right text-gray-600 dark:text-gray-400">
                    {r.win_rate.toFixed(1)}%
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Actions */}
      <div className="flex gap-4">
        <button
          onClick={onNewSearch}
          className="flex items-center gap-2 px-4 py-2 bg-gray-100 dark:bg-gray-800 text-gray-700 dark:text-gray-300 rounded-lg hover:bg-gray-200 dark:hover:bg-gray-700"
        >
          <RiArrowLeftLine className="w-5 h-5" />
          New Simulation
        </button>
        <button
          onClick={() => {
            const json = JSON.stringify(results, null, 2);
            const blob = new Blob([json], { type: "application/json" });
            const url = URL.createObjectURL(blob);
            const a = document.createElement("a");
            a.href = url;
            a.download = `simulation-${searchId}.json`;
            a.click();
          }}
          className="flex items-center gap-2 px-4 py-2 bg-gray-100 dark:bg-gray-800 text-gray-700 dark:text-gray-300 rounded-lg hover:bg-gray-200 dark:hover:bg-gray-700"
        >
          <RiDownloadLine className="w-5 h-5" />
          Export JSON
        </button>
      </div>
    </div>
  );
}

// History Component
function SearchHistory({
  onSelectSearch,
}: {
  onSelectSearch: (searchId: string) => void;
}) {
  const [history, setHistory] = useState<SearchHistoryItem[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [isExpanded, setIsExpanded] = useState(false);

  useEffect(() => {
    getSearchHistory(10)
      .then((data) => setHistory(data.searches))
      .catch(console.error)
      .finally(() => setIsLoading(false));
  }, []);

  if (isLoading || history.length === 0) {
    return null;
  }

  return (
    <div className="mt-6 bg-white dark:bg-gray-900 rounded-lg border border-gray-200 dark:border-gray-800">
      <button
        onClick={() => setIsExpanded(!isExpanded)}
        className="w-full px-4 py-3 flex items-center justify-between text-left hover:bg-gray-50 dark:hover:bg-gray-800 rounded-lg"
      >
        <div className="flex items-center gap-2">
          <RiHistoryLine className="w-5 h-5 text-gray-500" />
          <span className="font-medium text-gray-900 dark:text-gray-100">
            Previous Simulations ({history.length})
          </span>
        </div>
        <RiArrowRightLine
          className={`w-5 h-5 text-gray-400 transition-transform ${
            isExpanded ? "rotate-90" : ""
          }`}
        />
      </button>

      {isExpanded && (
        <div className="border-t border-gray-200 dark:border-gray-800">
          <div className="divide-y divide-gray-100 dark:divide-gray-800">
            {history.map((item) => (
              <button
                key={item.id}
                onClick={() => onSelectSearch(item.id)}
                className="w-full px-4 py-3 flex items-center justify-between hover:bg-gray-50 dark:hover:bg-gray-800 text-left"
              >
                <div className="flex-1 min-w-0">
                  <div className="font-medium text-gray-900 dark:text-gray-100 truncate">
                    {item.name}
                  </div>
                  <div className="text-sm text-gray-500 dark:text-gray-400">
                    {new Date(item.started_at).toLocaleDateString()}{" "}
                    {new Date(item.started_at).toLocaleTimeString()} •{" "}
                    {item.total_simulations?.toLocaleString() ?? '?'} simulations
                  </div>
                </div>
                <div className="ml-4 text-right">
                  {item.best_alpha != null && typeof item.best_alpha === 'number' && (
                    <div
                      className={`font-semibold ${
                        item.best_alpha >= 0 ? "text-green-600" : "text-red-600"
                      }`}
                    >
                      {item.best_alpha >= 0 ? "+" : ""}
                      {item.best_alpha.toFixed(1)}% α
                    </div>
                  )}
                  <div className="text-xs text-gray-500">
                    {typeof item.duration_seconds === 'number' ? `${item.duration_seconds.toFixed(1)}s` : '—'}
                  </div>
                </div>
              </button>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

// Main Page
export default function GridSearchPage() {
  const [viewState, setViewState] = useState<ViewState>("configure");
  const [searchId, setSearchId] = useState<string | null>(null);
  const [totalSimulations, setTotalSimulations] = useState<number>(0);

  const handleStartSearch = (id: string, total: number) => {
    setSearchId(id);
    setTotalSimulations(total);
    setViewState("running");
  };

  const handleSearchComplete = () => {
    setViewState("results");
  };

  const handleSearchCancel = () => {
    setSearchId(null);
    setTotalSimulations(0);
    setViewState("configure");
  };

  const handleNewSearch = () => {
    setSearchId(null);
    setTotalSimulations(0);
    setViewState("configure");
  };

  const handleSelectPastSearch = (id: string) => {
    setSearchId(id);
    setViewState("results");
  };

  return (
    <div className="p-6 max-w-6xl mx-auto">
      <div className="mb-6">
        <h1 className="text-2xl font-bold text-gray-900 dark:text-gray-100">
          Strategy Simulations
        </h1>
        <p className="text-gray-600 dark:text-gray-400 mt-1">
          Set your base strategy, then test variations to find what works best
        </p>
      </div>

      {viewState === "configure" && (
        <>
          <GridSearchBuilder onStart={handleStartSearch} />
          <SearchHistory onSelectSearch={handleSelectPastSearch} />
        </>
      )}

      {viewState === "running" && searchId && (
        <GridSearchProgressView
          searchId={searchId}
          totalSimulations={totalSimulations}
          onComplete={handleSearchComplete}
          onCancel={handleSearchCancel}
        />
      )}

      {viewState === "results" && searchId && (
        <GridSearchResultsView searchId={searchId} onNewSearch={handleNewSearch} />
      )}
    </div>
  );
}
