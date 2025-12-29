"use client";

import { useState, useEffect } from "react";
import { RiPlayLine, RiStopLine, RiSaveLine, RiHistoryLine, RiArrowRightLine } from "@remixicon/react";
import { Button } from "@/components/Button";
import { ProgressBar } from "@/components/ProgressBar";
import {
  startFactorDiscovery,
  getProgressStream,
  getResults,
  getHistory,
  getAvailableQuarters,
  cancelRun,
  formatFactorName,
  formatPValue,
  getAlphaColorClass,
  getPhaseDisplayName,
  type FactorDiscoveryRequest,
  type FactorDiscoveryResult,
  type FactorDiscoverySummary,
  type ProgressUpdate,
  type CombinedStrategyResult,
  type FilterSpec,
  type PipelineSettings,
} from "@/lib/factor-discovery-api";
import {
  saveStrategy,
  generateStrategyName,
  formatStrategyDate,
} from "@/lib/saved-strategies";

type ViewState = "configure" | "running" | "results";

/**
 * Convert filter specifications to Pipeline settings.
 * This mirrors the backend's _build_pipeline_settings logic.
 */
function filtersToSettings(filters: FilterSpec[]): PipelineSettings {
  const settings: PipelineSettings = {
    piotroski_enabled: false,
    piotroski_min: 5,
    altman_enabled: false,
    altman_zone: "safe",
    quality_enabled: false,
    min_quality: "weak",
    excluded_tags: [],
    required_tags: [],
    graham_enabled: false,
    graham_mode: "strict",
    graham_min: 5,
    magic_formula_enabled: false,
    mf_top_pct: 20,
    peg_enabled: false,
    max_peg: 1.5,
    net_net_enabled: false,
    fama_french_enabled: false,
    ff_top_pct: 30,
    min_lenses: 1,
    strict_mode: false,
  };

  for (const f of filters) {
    const factor = f.factor.toLowerCase();

    // Piotroski
    if (factor === "piotroski_score") {
      settings.piotroski_enabled = true;
      if (typeof f.value === "number") {
        settings.piotroski_min = f.value;
      }
    }

    // Altman
    if (factor === "altman_zone") {
      settings.altman_enabled = true;
      settings.altman_zone = String(f.value);
    }
    if (factor === "altman_z_score" && typeof f.value === "number") {
      settings.altman_enabled = true;
      if (f.value >= 2.99) settings.altman_zone = "safe";
      else if (f.value >= 1.81) settings.altman_zone = "grey";
      else settings.altman_zone = "distress";
    }

    // Quality tags
    if (factor.startsWith("has_")) {
      const tagMap: Record<string, string> = {
        has_premium_priced: "Premium Priced",
        has_volatile_returns: "Volatile Returns",
        has_weak_moat_signal: "Weak Moat Signal",
        has_durable_compounder: "Durable Compounder",
        has_cash_machine: "Cash Machine",
        has_deep_value: "Deep Value",
        has_heavy_reinvestor: "Heavy Reinvestor",
        has_earnings_quality_concern: "Earnings Quality Concern",
      };
      const tag = tagMap[factor];
      if (tag) {
        settings.quality_enabled = true;
        if (f.value === true) {
          settings.required_tags.push(tag);
        } else if (f.value === false) {
          settings.excluded_tags.push(tag);
        }
      }
    }

    // Quality label
    if (factor === "quality_label") {
      settings.quality_enabled = true;
      settings.min_quality = String(f.value);
    }

    // Graham
    if (factor === "graham_score" && typeof f.value === "number") {
      settings.graham_enabled = true;
      settings.graham_min = f.value;
    }

    // Book to market (Fama-French)
    if (factor === "book_to_market_percentile" && typeof f.value === "number") {
      settings.fama_french_enabled = true;
      settings.ff_top_pct = Math.round((1 - f.value) * 100);
    }

    // PEG
    if (factor === "peg_ratio" && typeof f.value === "number") {
      settings.peg_enabled = true;
      settings.max_peg = f.value;
    }

    // Net-net
    if (factor === "net_net_discount") {
      settings.net_net_enabled = true;
    }

    // Magic formula
    if (factor === "magic_formula_rank" && typeof f.value === "number") {
      settings.magic_formula_enabled = true;
    }
  }

  return settings;
}

export default function FactorDiscoveryPage() {
  // View state
  const [viewState, setViewState] = useState<ViewState>("configure");

  // Configure state
  const [availableQuarters, setAvailableQuarters] = useState<string[]>([]);
  const [selectedQuarters, setSelectedQuarters] = useState<Set<string>>(new Set());
  const [holdingPeriods, setHoldingPeriods] = useState<Set<number>>(new Set([1, 2, 3, 4]));
  const [minSampleSize, setMinSampleSize] = useState(100);
  const [significanceLevel, setSignificanceLevel] = useState(0.01);
  const [costHaircut, setCostHaircut] = useState(3.0);
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [portfolioSizes, setPortfolioSizes] = useState<number[]>([20]);
  const [rankingMethod, setRankingMethod] = useState<string>("magic-formula");
  const [maxFactors, setMaxFactors] = useState<number>(4);

  // Exclusion filters
  const [excludeAltmanZones, setExcludeAltmanZones] = useState<string[]>(["distress"]);
  const [minPiotroski, setMinPiotroski] = useState<number | null>(null);
  const [excludeQualityTags, setExcludeQualityTags] = useState<string[]>([]);
  const [excludePennyStocks, setExcludePennyStocks] = useState(false);
  const [excludeNegativeEarnings, setExcludeNegativeEarnings] = useState(false);

  // Running state
  const [runId, setRunId] = useState<string | null>(null);
  const [progress, setProgress] = useState(0);
  const [phase, setPhase] = useState("");
  const [currentFactor, setCurrentFactor] = useState<string | null>(null);
  const [isCancelling, setIsCancelling] = useState(false);

  // Results state
  const [result, setResult] = useState<FactorDiscoveryResult | null>(null);
  const [selectedHP, setSelectedHP] = useState<number>(4);
  const [history, setHistory] = useState<FactorDiscoverySummary[]>([]);
  const [showHistory, setShowHistory] = useState(false);

  // Error state
  const [error, setError] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(false);

  // Load available quarters on mount
  useEffect(() => {
    async function loadQuarters() {
      try {
        const data = await getAvailableQuarters();
        setAvailableQuarters(data.quarters);
        // Select all quarters by default
        setSelectedQuarters(new Set(data.quarters));
      } catch (err) {
        console.error("Failed to load quarters:", err);
        setError("Failed to load available quarters");
      }
    }
    loadQuarters();
  }, []);

  // Load history when showing history section
  useEffect(() => {
    if (showHistory) {
      getHistory().then(setHistory).catch(console.error);
    }
  }, [showHistory]);

  // SSE progress connection
  useEffect(() => {
    if (viewState !== "running" || !runId) return;

    const eventSource = getProgressStream(runId);

    eventSource.onmessage = (event) => {
      try {
        const data: ProgressUpdate = JSON.parse(event.data);
        setProgress(data.progress);
        setPhase(data.phase);
        setCurrentFactor(data.current_factor || null);

        if (data.status === "completed") {
          eventSource.close();
          fetchResults(runId);
        } else if (data.status === "failed") {
          eventSource.close();
          setError(data.error || "Analysis failed");
          setViewState("configure");
        } else if (data.status === "cancelled") {
          eventSource.close();
          setViewState("configure");
        }
      } catch (err) {
        console.error("Failed to parse progress:", err);
      }
    };

    eventSource.onerror = () => {
      eventSource.close();
      // Try to fetch results in case it completed
      fetchResults(runId);
    };

    return () => eventSource.close();
  }, [viewState, runId]);

  const fetchResults = async (id: string) => {
    try {
      const data = await getResults(id);
      setResult(data);
      if (data.best_holding_period) {
        setSelectedHP(data.best_holding_period);
      }
      setViewState("results");
    } catch (err) {
      console.error("Failed to fetch results:", err);
      setError("Failed to load results");
      setViewState("configure");
    }
  };

  const handleStartAnalysis = async () => {
    if (selectedQuarters.size === 0) {
      setError("Please select at least one quarter");
      return;
    }
    if (holdingPeriods.size === 0) {
      setError("Please select at least one holding period");
      return;
    }

    setError(null);
    setIsLoading(true);

    try {
      const request: FactorDiscoveryRequest = {
        quarters: Array.from(selectedQuarters).sort(),
        holding_periods: Array.from(holdingPeriods).sort(),
        min_sample_size: minSampleSize,
        significance_level: significanceLevel,
        cost_haircut: costHaircut,
        portfolio_sizes: portfolioSizes,
        ranking_method: rankingMethod,
        max_factors: maxFactors,
        exclusions: {
          exclude_altman_zones: excludeAltmanZones,
          min_piotroski: minPiotroski,
          exclude_quality_tags: excludeQualityTags,
          require_quality_tags: [],
          exclude_penny_stocks: excludePennyStocks,
          exclude_negative_earnings: excludeNegativeEarnings,
        },
      };

      const response = await startFactorDiscovery(request);
      setRunId(response.run_id);
      setProgress(0);
      setPhase("initializing");
      setViewState("running");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to start analysis");
    } finally {
      setIsLoading(false);
    }
  };

  const handleCancel = async () => {
    if (!runId) return;
    setIsCancelling(true);
    try {
      await cancelRun(runId);
    } catch (err) {
      console.error("Failed to cancel:", err);
    } finally {
      setIsCancelling(false);
    }
  };

  const handleSaveStrategy = (hp: number) => {
    if (!result) return;

    const strategy = result.recommended_strategies[hp];
    if (!strategy) return;

    const defaultName = generateStrategyName(hp);
    const name = prompt("Enter a name for this strategy:", defaultName);

    if (!name) return; // User cancelled

    saveStrategy({
      name: name.trim() || defaultName,
      holding_period: hp,
      settings: strategy.pipeline_settings,
      expected_alpha: strategy.expected_alpha,
      expected_alpha_ci_lower: strategy.expected_alpha_ci_lower,
      expected_alpha_ci_upper: strategy.expected_alpha_ci_upper,
      win_rate: strategy.expected_win_rate,
      sample_size: strategy.sample_size,
      source: "factor_discovery",
    });

    alert(`Strategy "${name}" saved! You can load it in the Pipeline page.`);
  };

  const handleSaveCombo = (hp: number, combo: CombinedStrategyResult, index: number) => {
    const defaultName = `Strategy #${index + 1} (${hp}Q) - ${new Date().toLocaleDateString()}`;
    const name = prompt("Enter a name for this strategy:", defaultName);

    if (!name) return; // User cancelled

    const settings = filtersToSettings(combo.filters);

    saveStrategy({
      name: name.trim() || defaultName,
      holding_period: hp,
      settings,
      expected_alpha: combo.mean_alpha,
      expected_alpha_ci_lower: combo.ci_lower,
      expected_alpha_ci_upper: combo.ci_upper,
      win_rate: combo.win_rate,
      sample_size: combo.sample_size,
      source: "factor_discovery",
    });

    alert(`Strategy "${name}" saved! You can load it in the Pipeline page.`);
  };

  const handleLoadHistoryResult = async (summary: FactorDiscoverySummary) => {
    try {
      const data = await getResults(summary.run_id);
      setResult(data);
      if (data.best_holding_period) {
        setSelectedHP(data.best_holding_period);
      }
      setViewState("results");
      setShowHistory(false);
    } catch (err) {
      setError("Failed to load historical result");
    }
  };

  const toggleQuarter = (quarter: string) => {
    setSelectedQuarters((prev) => {
      const next = new Set(prev);
      if (next.has(quarter)) {
        next.delete(quarter);
      } else {
        next.add(quarter);
      }
      return next;
    });
  };

  const toggleHoldingPeriod = (hp: number) => {
    setHoldingPeriods((prev) => {
      const next = new Set(prev);
      if (next.has(hp)) {
        next.delete(hp);
      } else {
        next.add(hp);
      }
      return next;
    });
  };

  const selectAllQuarters = () => setSelectedQuarters(new Set(availableQuarters));
  const deselectAllQuarters = () => setSelectedQuarters(new Set());

  // Render based on view state
  if (viewState === "running") {
    return (
      <div className="p-6 max-w-4xl mx-auto">
        <h1 className="text-2xl font-bold text-gray-900 dark:text-gray-100 mb-2">
          Factor Discovery
        </h1>
        <p className="text-gray-600 dark:text-gray-400 mb-8">
          Analyzing factors to find optimal strategy...
        </p>

        <div className="bg-white dark:bg-gray-900 rounded-xl border border-gray-200 dark:border-gray-800 p-8">
          <div className="text-center mb-6">
            <div className="text-4xl font-bold text-indigo-600 dark:text-indigo-400 mb-2">
              {Math.round(progress * 100)}%
            </div>
            <div className="text-lg text-gray-600 dark:text-gray-400">
              {getPhaseDisplayName(phase)}
            </div>
            {currentFactor && (
              <div className="text-sm text-gray-500 dark:text-gray-500 mt-1">
                Analyzing: {formatFactorName(currentFactor)}
              </div>
            )}
          </div>

          <ProgressBar value={progress * 100} max={100} className="mb-8" />

          <div className="flex justify-center">
            <Button
              variant="secondary"
              onClick={handleCancel}
              disabled={isCancelling}
            >
              <RiStopLine className="w-4 h-4 mr-2" />
              {isCancelling ? "Cancelling..." : "Cancel"}
            </Button>
          </div>
        </div>
      </div>
    );
  }

  if (viewState === "results" && result) {
    const strategy = result.recommended_strategies[selectedHP];
    const factors = result.factor_results[selectedHP] || [];

    return (
      <div className="p-6 max-w-6xl mx-auto">
        <div className="flex items-center justify-between mb-6">
          <div>
            <h1 className="text-2xl font-bold text-gray-900 dark:text-gray-100">
              Factor Discovery Results
            </h1>
            <p className="text-gray-600 dark:text-gray-400">
              Analyzed {result.total_observations.toLocaleString()} observations in{" "}
              {result.duration_seconds?.toFixed(1)}s
            </p>
          </div>
          <Button variant="secondary" onClick={() => setViewState("configure")}>
            New Analysis
          </Button>
        </div>

        {/* Holding Period Tabs */}
        <div className="flex gap-2 mb-6">
          {[1, 2, 3, 4].map((hp) => (
            <button
              key={hp}
              onClick={() => setSelectedHP(hp)}
              className={`px-4 py-2 rounded-lg font-medium transition-colors ${
                selectedHP === hp
                  ? "bg-indigo-600 text-white"
                  : "bg-gray-100 dark:bg-gray-800 text-gray-700 dark:text-gray-300 hover:bg-gray-200 dark:hover:bg-gray-700"
              } ${result.best_holding_period === hp ? "ring-2 ring-indigo-400" : ""}`}
            >
              {hp}Q Hold
              {result.best_holding_period === hp && (
                <span className="ml-1 text-xs">(Best)</span>
              )}
            </button>
          ))}
        </div>

        {/* Recommended Strategy Card */}
        {strategy && (
          <div className="bg-white dark:bg-gray-900 rounded-xl border border-gray-200 dark:border-gray-800 p-6 mb-6">
            <div className="flex items-start justify-between mb-4">
              <h2 className="text-lg font-semibold text-gray-900 dark:text-gray-100">
                Recommended Strategy for {selectedHP}Q Hold
              </h2>
              <Button variant="primary" onClick={() => handleSaveStrategy(selectedHP)}>
                <RiSaveLine className="w-4 h-4 mr-2" />
                Save Strategy
              </Button>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-6">
              <div className="bg-gray-50 dark:bg-gray-800 rounded-lg p-4">
                <div className="text-sm text-gray-500 dark:text-gray-400">Expected Alpha</div>
                <div className={`text-2xl font-bold ${getAlphaColorClass(strategy.expected_alpha)}`}>
                  {strategy.expected_alpha > 0 ? "+" : ""}
                  {strategy.expected_alpha.toFixed(1)}%
                </div>
                <div className="text-xs text-gray-500 dark:text-gray-500">
                  95% CI: {strategy.expected_alpha_ci_lower.toFixed(1)}% to{" "}
                  {strategy.expected_alpha_ci_upper.toFixed(1)}%
                </div>
              </div>

              <div className="bg-gray-50 dark:bg-gray-800 rounded-lg p-4">
                <div className="text-sm text-gray-500 dark:text-gray-400">Win Rate</div>
                <div className="text-2xl font-bold text-gray-900 dark:text-gray-100">
                  {strategy.expected_win_rate.toFixed(0)}%
                </div>
                <div className="text-xs text-gray-500 dark:text-gray-500">
                  Stocks with positive alpha
                </div>
              </div>

              <div className="bg-gray-50 dark:bg-gray-800 rounded-lg p-4">
                <div className="text-sm text-gray-500 dark:text-gray-400">Sample Size</div>
                <div className="text-2xl font-bold text-gray-900 dark:text-gray-100">
                  {strategy.sample_size.toLocaleString()}
                </div>
                <div className="text-xs text-gray-500 dark:text-gray-500">
                  Historical matches
                </div>
              </div>

              <div className="bg-gray-50 dark:bg-gray-800 rounded-lg p-4">
                <div className="text-sm text-gray-500 dark:text-gray-400">Confidence</div>
                <div className="text-2xl font-bold text-gray-900 dark:text-gray-100">
                  {(strategy.confidence_score * 100).toFixed(0)}%
                </div>
                <div className="text-xs text-gray-500 dark:text-gray-500">
                  Overall confidence score
                </div>
              </div>
            </div>

            {strategy.key_factors.length > 0 && (
              <div>
                <h3 className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                  Key Factors
                </h3>
                <div className="space-y-1">
                  {strategy.key_factors.map((factor, i) => (
                    <div
                      key={i}
                      className="flex items-center text-sm text-gray-600 dark:text-gray-400"
                    >
                      <span className="w-2 h-2 bg-indigo-500 rounded-full mr-2" />
                      <span className="font-medium">{formatFactorName(factor.name)}</span>
                      <span className="mx-2">{factor.threshold}</span>
                      {factor.lift && (
                        <span className="text-gray-500">(lift {factor.lift.toFixed(2)}x)</span>
                      )}
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Portfolio Size Comparison */}
            {strategy.portfolio_stats && Object.keys(strategy.portfolio_stats).length > 0 && (
              <div className="mt-6 pt-6 border-t border-gray-200 dark:border-gray-700">
                <h3 className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-3">
                  Portfolio Size Comparison
                </h3>
                <p className="text-xs text-gray-500 dark:text-gray-400 mb-3">
                  What if you only bought the top N stocks each quarter?
                </p>
                <div className="overflow-x-auto">
                  <table className="w-full text-sm">
                    <thead>
                      <tr className="border-b border-gray-200 dark:border-gray-700">
                        <th className="text-left py-2 px-3 font-medium text-gray-500 dark:text-gray-400">Portfolio</th>
                        <th className="text-right py-2 px-3 font-medium text-gray-500 dark:text-gray-400">Alpha</th>
                        <th className="text-right py-2 px-3 font-medium text-gray-500 dark:text-gray-400">Win Rate</th>
                        <th className="text-right py-2 px-3 font-medium text-gray-500 dark:text-gray-400">Samples</th>
                        <th className="text-right py-2 px-3 font-medium text-gray-500 dark:text-gray-400">95% CI</th>
                      </tr>
                    </thead>
                    <tbody>
                      {Object.entries(strategy.portfolio_stats)
                        .sort(([a], [b]) => Number(a) - Number(b))
                        .map(([size, stats]) => (
                          <tr key={size} className="border-b border-gray-100 dark:border-gray-800">
                            <td className="py-2 px-3 font-medium text-gray-900 dark:text-gray-100">
                              Top {size}
                            </td>
                            <td className={`py-2 px-3 text-right font-medium ${getAlphaColorClass(stats.mean_alpha)}`}>
                              {stats.mean_alpha > 0 ? "+" : ""}{stats.mean_alpha.toFixed(1)}%
                            </td>
                            <td className="py-2 px-3 text-right text-gray-600 dark:text-gray-400">
                              {stats.win_rate.toFixed(0)}%
                            </td>
                            <td className="py-2 px-3 text-right text-gray-600 dark:text-gray-400">
                              {stats.sample_size.toLocaleString()}
                            </td>
                            <td className="py-2 px-3 text-right text-gray-500 dark:text-gray-500">
                              [{stats.ci_lower.toFixed(1)}%, {stats.ci_upper.toFixed(1)}%]
                            </td>
                          </tr>
                        ))}
                      {/* Add "All Stocks" row for comparison */}
                      <tr className="border-b border-gray-100 dark:border-gray-800 bg-gray-50 dark:bg-gray-800/50">
                        <td className="py-2 px-3 font-medium text-gray-600 dark:text-gray-400 italic">
                          All stocks
                        </td>
                        <td className={`py-2 px-3 text-right font-medium ${getAlphaColorClass(strategy.expected_alpha)}`}>
                          {strategy.expected_alpha > 0 ? "+" : ""}{strategy.expected_alpha.toFixed(1)}%
                        </td>
                        <td className="py-2 px-3 text-right text-gray-600 dark:text-gray-400">
                          {strategy.expected_win_rate.toFixed(0)}%
                        </td>
                        <td className="py-2 px-3 text-right text-gray-600 dark:text-gray-400">
                          {strategy.sample_size.toLocaleString()}
                        </td>
                        <td className="py-2 px-3 text-right text-gray-500 dark:text-gray-500">
                          [{strategy.expected_alpha_ci_lower.toFixed(1)}%, {strategy.expected_alpha_ci_upper.toFixed(1)}%]
                        </td>
                      </tr>
                    </tbody>
                  </table>
                </div>
              </div>
            )}
          </div>
        )}

        {/* Factor Analysis Table */}
        <div className="bg-white dark:bg-gray-900 rounded-xl border border-gray-200 dark:border-gray-800 p-6 mb-6">
          <h2 className="text-lg font-semibold text-gray-900 dark:text-gray-100 mb-4">
            Factor Analysis ({selectedHP}Q Hold)
          </h2>

          <div className="overflow-x-auto">
            <table className="w-full">
              <thead>
                <tr className="border-b border-gray-200 dark:border-gray-700">
                  <th className="text-left py-3 px-2 text-sm font-medium text-gray-500 dark:text-gray-400">
                    Factor
                  </th>
                  <th className="text-left py-3 px-2 text-sm font-medium text-gray-500 dark:text-gray-400">
                    Best Threshold
                  </th>
                  <th className="text-right py-3 px-2 text-sm font-medium text-gray-500 dark:text-gray-400">
                    Alpha
                  </th>
                  <th className="text-right py-3 px-2 text-sm font-medium text-gray-500 dark:text-gray-400">
                    Lift
                  </th>
                  <th className="text-right py-3 px-2 text-sm font-medium text-gray-500 dark:text-gray-400">
                    p-value
                  </th>
                  <th className="text-right py-3 px-2 text-sm font-medium text-gray-500 dark:text-gray-400">
                    Sample
                  </th>
                </tr>
              </thead>
              <tbody>
                {factors
                  .filter((f) => f.best_threshold !== null)
                  .sort((a, b) => (b.best_threshold_lift || 0) - (a.best_threshold_lift || 0))
                  .map((factor) => (
                    <tr
                      key={factor.factor_name}
                      className="border-b border-gray-100 dark:border-gray-800 hover:bg-gray-50 dark:hover:bg-gray-800"
                    >
                      <td className="py-3 px-2 font-medium text-gray-900 dark:text-gray-100">
                        {formatFactorName(factor.factor_name)}
                      </td>
                      <td className="py-3 px-2 text-gray-600 dark:text-gray-400">
                        {factor.best_threshold}
                      </td>
                      <td
                        className={`py-3 px-2 text-right font-medium ${getAlphaColorClass(
                          factor.best_threshold_alpha || 0
                        )}`}
                      >
                        {factor.best_threshold_alpha !== null
                          ? `${factor.best_threshold_alpha > 0 ? "+" : ""}${factor.best_threshold_alpha.toFixed(
                              1
                            )}%`
                          : "—"}
                      </td>
                      <td className="py-3 px-2 text-right text-gray-900 dark:text-gray-100">
                        {factor.best_threshold_lift !== null
                          ? `${factor.best_threshold_lift.toFixed(2)}x`
                          : "—"}
                      </td>
                      <td className="py-3 px-2 text-right text-gray-600 dark:text-gray-400">
                        {formatPValue(factor.best_threshold_pvalue)}
                      </td>
                      <td className="py-3 px-2 text-right text-gray-600 dark:text-gray-400">
                        {factor.best_threshold_sample_size?.toLocaleString() || "—"}
                      </td>
                    </tr>
                  ))}
              </tbody>
            </table>
          </div>
        </div>

        {/* Alternative Strategies */}
        {result.combined_results[selectedHP] && result.combined_results[selectedHP].length > 0 && (
          <div className="bg-white dark:bg-gray-900 rounded-xl border border-gray-200 dark:border-gray-800 p-6 mb-6">
            <h2 className="text-lg font-semibold text-gray-900 dark:text-gray-100 mb-4">
              Alternative Strategies ({selectedHP}Q Hold)
            </h2>
            <p className="text-sm text-gray-500 dark:text-gray-400 mb-4">
              Top {result.combined_results[selectedHP].length} factor combinations ranked by alpha.
              Save any strategy you prefer.
            </p>

            <div className="space-y-3">
              {result.combined_results[selectedHP]
                .sort((a, b) => b.mean_alpha - a.mean_alpha)
                .map((combo, index) => (
                  <div
                    key={index}
                    className={`border rounded-lg p-4 ${
                      index === 0
                        ? "border-indigo-300 dark:border-indigo-700 bg-indigo-50 dark:bg-indigo-950/30"
                        : "border-gray-200 dark:border-gray-700"
                    }`}
                  >
                    <div className="flex items-start justify-between">
                      <div className="flex-1">
                        <div className="flex items-center gap-3 mb-2">
                          <span className={`text-lg font-bold ${getAlphaColorClass(combo.mean_alpha)}`}>
                            {combo.mean_alpha > 0 ? "+" : ""}{combo.mean_alpha.toFixed(1)}%
                          </span>
                          <span className="text-sm text-gray-500 dark:text-gray-400">
                            (95% CI: {combo.ci_lower.toFixed(1)}% to {combo.ci_upper.toFixed(1)}%)
                          </span>
                          <span className="text-sm text-gray-500 dark:text-gray-400">
                            • Win: {combo.win_rate.toFixed(0)}%
                          </span>
                          <span className="text-sm text-gray-500 dark:text-gray-400">
                            • n={combo.sample_size.toLocaleString()}
                          </span>
                          {index === 0 && (
                            <span className="px-2 py-0.5 bg-indigo-100 dark:bg-indigo-900 text-indigo-700 dark:text-indigo-300 text-xs rounded-full">
                              Best
                            </span>
                          )}
                        </div>
                        <div className="flex flex-wrap gap-2">
                          {combo.filters.map((filter, i) => (
                            <span
                              key={i}
                              className="px-2 py-1 bg-gray-100 dark:bg-gray-800 rounded text-sm text-gray-700 dark:text-gray-300"
                            >
                              {formatFactorName(filter.factor)} {filter.operator} {String(filter.value)}
                            </span>
                          ))}
                        </div>
                      </div>
                      <Button
                        variant="secondary"
                        onClick={() => handleSaveCombo(selectedHP, combo, index)}
                        className="ml-4 flex-shrink-0"
                      >
                        <RiSaveLine className="w-4 h-4 mr-1" />
                        Save
                      </Button>
                    </div>
                  </div>
                ))}
            </div>
          </div>
        )}

        {/* Disclaimer */}
        <div className="bg-amber-50 dark:bg-amber-950 border border-amber-200 dark:border-amber-800 rounded-lg p-4 text-amber-800 dark:text-amber-200 text-sm">
          Based on {result.config.quarters.length} quarters of historical data. Past performance
          does not guarantee future results. Only strategies with alpha &gt; {costHaircut}% are
          shown to account for trading costs.
        </div>
      </div>
    );
  }

  // Configure View (default)
  return (
    <div className="p-6 max-w-4xl mx-auto">
      <h1 className="text-2xl font-bold text-gray-900 dark:text-gray-100 mb-2">
        Factor Discovery
      </h1>
      <p className="text-gray-600 dark:text-gray-400 mb-8">
        Analyze historical data to discover which factors predict alpha. The system will recommend
        optimal Pipeline settings based on statistical analysis.
      </p>

      {error && (
        <div className="bg-red-50 dark:bg-red-950 border border-red-200 dark:border-red-800 rounded-lg p-4 text-red-700 dark:text-red-300 mb-6">
          {error}
        </div>
      )}

      {/* Quarter Selection */}
      <div className="bg-white dark:bg-gray-900 rounded-xl border border-gray-200 dark:border-gray-800 p-6 mb-6">
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-lg font-semibold text-gray-900 dark:text-gray-100">
            Quarters to Analyze
          </h2>
          <div className="flex gap-2">
            <button
              onClick={selectAllQuarters}
              className="text-sm text-indigo-600 dark:text-indigo-400 hover:underline"
            >
              Select All
            </button>
            <span className="text-gray-300 dark:text-gray-600">|</span>
            <button
              onClick={deselectAllQuarters}
              className="text-sm text-indigo-600 dark:text-indigo-400 hover:underline"
            >
              Clear
            </button>
          </div>
        </div>

        <div className="flex flex-wrap gap-2">
          {availableQuarters.map((quarter) => (
            <button
              key={quarter}
              onClick={() => toggleQuarter(quarter)}
              className={`px-3 py-1.5 rounded-lg text-sm font-medium transition-colors ${
                selectedQuarters.has(quarter)
                  ? "bg-indigo-600 text-white"
                  : "bg-gray-100 dark:bg-gray-800 text-gray-700 dark:text-gray-300 hover:bg-gray-200 dark:hover:bg-gray-700"
              }`}
            >
              {quarter}
            </button>
          ))}
        </div>

        <p className="text-sm text-gray-500 dark:text-gray-400 mt-3">
          {selectedQuarters.size} of {availableQuarters.length} quarters selected
        </p>
      </div>

      {/* Holding Periods */}
      <div className="bg-white dark:bg-gray-900 rounded-xl border border-gray-200 dark:border-gray-800 p-6 mb-6">
        <h2 className="text-lg font-semibold text-gray-900 dark:text-gray-100 mb-4">
          Holding Periods
        </h2>

        <div className="flex gap-3">
          {[1, 2, 3, 4].map((hp) => (
            <button
              key={hp}
              onClick={() => toggleHoldingPeriod(hp)}
              className={`px-4 py-2 rounded-lg font-medium transition-colors ${
                holdingPeriods.has(hp)
                  ? "bg-indigo-600 text-white"
                  : "bg-gray-100 dark:bg-gray-800 text-gray-700 dark:text-gray-300 hover:bg-gray-200 dark:hover:bg-gray-700"
              }`}
            >
              {hp} Quarter{hp > 1 ? "s" : ""}
            </button>
          ))}
        </div>

        <p className="text-sm text-gray-500 dark:text-gray-400 mt-3">
          Analyze performance for holding stocks 1-4 quarters
        </p>
      </div>

      {/* Advanced Settings */}
      <div className="bg-white dark:bg-gray-900 rounded-xl border border-gray-200 dark:border-gray-800 p-6 mb-6">
        <button
          onClick={() => setShowAdvanced(!showAdvanced)}
          className="flex items-center justify-between w-full text-left"
        >
          <h2 className="text-lg font-semibold text-gray-900 dark:text-gray-100">
            Advanced Settings
          </h2>
          <span className="text-gray-400">{showAdvanced ? "−" : "+"}</span>
        </button>

        {showAdvanced && (
          <div className="mt-4 space-y-6">
            {/* Portfolio Simulation */}
            <div className="p-4 bg-indigo-50 dark:bg-indigo-950/30 rounded-lg border border-indigo-200 dark:border-indigo-800">
              <h3 className="text-sm font-semibold text-indigo-900 dark:text-indigo-100 mb-3">
                Portfolio Simulation
              </h3>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div>
                  <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                    Portfolio Sizes to Test
                  </label>
                  <div className="flex flex-wrap gap-2">
                    {[10, 20, 30, 50, 100].map((size) => (
                      <button
                        key={size}
                        onClick={() => {
                          if (portfolioSizes.includes(size)) {
                            setPortfolioSizes(portfolioSizes.filter((s) => s !== size));
                          } else {
                            setPortfolioSizes([...portfolioSizes, size].sort((a, b) => a - b));
                          }
                        }}
                        className={`px-3 py-1.5 rounded-lg text-sm font-medium transition-colors ${
                          portfolioSizes.includes(size)
                            ? "bg-indigo-600 text-white"
                            : "bg-white dark:bg-gray-800 text-gray-700 dark:text-gray-300 border border-gray-200 dark:border-gray-700"
                        }`}
                      >
                        Top {size}
                      </button>
                    ))}
                  </div>
                  <p className="text-xs text-gray-500 mt-2">
                    Simulate buying only the top N stocks per quarter
                  </p>
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                    Rank Stocks By
                  </label>
                  <select
                    value={rankingMethod}
                    onChange={(e) => setRankingMethod(e.target.value)}
                    className="w-full px-3 py-2 rounded-lg border border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-800 text-gray-900 dark:text-gray-100"
                  >
                    <option value="magic-formula">Magic Formula Rank</option>
                    <option value="earnings-yield">Earnings Yield (higher is better)</option>
                    <option value="roic">ROIC (higher is better)</option>
                    <option value="graham-score">Graham Score (higher is better)</option>
                  </select>
                  <p className="text-xs text-gray-500 mt-2">
                    How to rank stocks when selecting top N
                  </p>
                </div>
              </div>
            </div>

            {/* Exclusion Filters */}
            <div className="p-4 bg-red-50 dark:bg-red-950/30 rounded-lg border border-red-200 dark:border-red-800">
              <h3 className="text-sm font-semibold text-red-900 dark:text-red-100 mb-3">
                Stock Exclusions
              </h3>
              <p className="text-xs text-red-700 dark:text-red-300 mb-4">
                Exclude certain stocks from the analysis (these will never be considered)
              </p>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                {/* Altman Zones */}
                <div>
                  <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                    Exclude Altman Zones
                  </label>
                  <div className="flex flex-wrap gap-2">
                    {["distress", "grey", "safe"].map((zone) => (
                      <button
                        key={zone}
                        onClick={() => {
                          if (excludeAltmanZones.includes(zone)) {
                            setExcludeAltmanZones(excludeAltmanZones.filter((z) => z !== zone));
                          } else {
                            setExcludeAltmanZones([...excludeAltmanZones, zone]);
                          }
                        }}
                        className={`px-3 py-1.5 rounded-lg text-sm font-medium transition-colors capitalize ${
                          excludeAltmanZones.includes(zone)
                            ? "bg-red-600 text-white"
                            : "bg-white dark:bg-gray-800 text-gray-700 dark:text-gray-300 border border-gray-200 dark:border-gray-700"
                        }`}
                      >
                        {zone}
                      </button>
                    ))}
                  </div>
                  <p className="text-xs text-gray-500 mt-2">
                    Distress = high bankruptcy risk
                  </p>
                </div>

                {/* Min Piotroski */}
                <div>
                  <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                    Minimum Piotroski Score
                  </label>
                  <select
                    value={minPiotroski ?? ""}
                    onChange={(e) => setMinPiotroski(e.target.value ? parseInt(e.target.value) : null)}
                    className="w-full px-3 py-2 rounded-lg border border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-800 text-gray-900 dark:text-gray-100"
                  >
                    <option value="">No minimum</option>
                    {[1, 2, 3, 4, 5, 6, 7, 8].map((n) => (
                      <option key={n} value={n}>
                        {n}+ (exclude below {n})
                      </option>
                    ))}
                  </select>
                  <p className="text-xs text-gray-500 mt-2">
                    Exclude weak fundamentals (0-9 scale)
                  </p>
                </div>

                {/* Quality Tags to Exclude */}
                <div className="md:col-span-2">
                  <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                    Exclude Quality Tags
                  </label>
                  <div className="flex flex-wrap gap-2">
                    {[
                      "Premium Priced",
                      "Volatile Returns",
                      "Weak Moat Signal",
                      "Earnings Quality Concern",
                    ].map((tag) => (
                      <button
                        key={tag}
                        onClick={() => {
                          if (excludeQualityTags.includes(tag)) {
                            setExcludeQualityTags(excludeQualityTags.filter((t) => t !== tag));
                          } else {
                            setExcludeQualityTags([...excludeQualityTags, tag]);
                          }
                        }}
                        className={`px-3 py-1.5 rounded-lg text-sm font-medium transition-colors ${
                          excludeQualityTags.includes(tag)
                            ? "bg-red-600 text-white"
                            : "bg-white dark:bg-gray-800 text-gray-700 dark:text-gray-300 border border-gray-200 dark:border-gray-700"
                        }`}
                      >
                        {tag}
                      </button>
                    ))}
                  </div>
                  <p className="text-xs text-gray-500 mt-2">
                    Stocks with these warning tags will be excluded
                  </p>
                </div>

                {/* Boolean Exclusions */}
                <div className="md:col-span-2 flex flex-wrap gap-4">
                  <label className="flex items-center gap-2 cursor-pointer">
                    <input
                      type="checkbox"
                      checked={excludePennyStocks}
                      onChange={(e) => setExcludePennyStocks(e.target.checked)}
                      className="w-4 h-4 rounded border-gray-300 text-red-600 focus:ring-red-500"
                    />
                    <span className="text-sm text-gray-700 dark:text-gray-300">
                      Exclude penny stocks (&lt;$5)
                    </span>
                  </label>

                  <label className="flex items-center gap-2 cursor-pointer">
                    <input
                      type="checkbox"
                      checked={excludeNegativeEarnings}
                      onChange={(e) => setExcludeNegativeEarnings(e.target.checked)}
                      className="w-4 h-4 rounded border-gray-300 text-red-600 focus:ring-red-500"
                    />
                    <span className="text-sm text-gray-700 dark:text-gray-300">
                      Exclude negative earnings
                    </span>
                  </label>
                </div>
              </div>
            </div>

            {/* Other Settings */}
            <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
              <div>
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                  Max Factors
                </label>
                <select
                  value={maxFactors}
                  onChange={(e) => setMaxFactors(parseInt(e.target.value))}
                  className="w-full px-3 py-2 rounded-lg border border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-800 text-gray-900 dark:text-gray-100"
                >
                  {[2, 3, 4, 5, 6, 8, 10, 12, 15, 18].map((n) => (
                    <option key={n} value={n}>
                      {n} factors {n === 4 ? "(recommended)" : n >= 8 ? "(slow)" : ""}
                    </option>
                  ))}
                </select>
                <p className="text-xs text-gray-500 mt-1">
                  Max factors to combine (more = slower)
                </p>
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                  Min Sample Size
                </label>
                <input
                  type="number"
                  value={minSampleSize}
                  onChange={(e) => setMinSampleSize(parseInt(e.target.value) || 100)}
                  className="w-full px-3 py-2 rounded-lg border border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-800 text-gray-900 dark:text-gray-100"
                />
                <p className="text-xs text-gray-500 mt-1">
                  Minimum observations for valid threshold
                </p>
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                  Significance Level
                </label>
                <input
                  type="number"
                  step="0.01"
                  value={significanceLevel}
                  onChange={(e) => setSignificanceLevel(parseFloat(e.target.value) || 0.01)}
                  className="w-full px-3 py-2 rounded-lg border border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-800 text-gray-900 dark:text-gray-100"
                />
                <p className="text-xs text-gray-500 mt-1">
                  p-value threshold (0.01 = 99% confidence)
                </p>
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                  Cost Haircut (%)
                </label>
                <input
                  type="number"
                  step="0.5"
                  value={costHaircut}
                  onChange={(e) => setCostHaircut(parseFloat(e.target.value) || 3.0)}
                  className="w-full px-3 py-2 rounded-lg border border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-800 text-gray-900 dark:text-gray-100"
                />
                <p className="text-xs text-gray-500 mt-1">
                  Only trust strategies with alpha above this
                </p>
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Preview & Run Button */}
      <div className="bg-white dark:bg-gray-900 rounded-xl border border-gray-200 dark:border-gray-800 p-6 mb-6">
        <div className="flex items-center justify-between">
          <div>
            <div className="text-sm text-gray-500 dark:text-gray-400">
              Estimated observations:
            </div>
            <div className="text-2xl font-bold text-gray-900 dark:text-gray-100">
              ~{(selectedQuarters.size * 5000 * holdingPeriods.size).toLocaleString()}
            </div>
            <div className="text-sm text-gray-500 dark:text-gray-400">
              across {selectedQuarters.size} quarters, {holdingPeriods.size} holding periods
            </div>
          </div>

          <Button
            variant="primary"
            onClick={handleStartAnalysis}
            disabled={isLoading || selectedQuarters.size === 0 || holdingPeriods.size === 0}
          >
            <RiPlayLine className="w-4 h-4 mr-2" />
            {isLoading ? "Starting..." : "Run Factor Discovery"}
          </Button>
        </div>
      </div>

      {/* History Section */}
      <div className="bg-white dark:bg-gray-900 rounded-xl border border-gray-200 dark:border-gray-800 p-6">
        <button
          onClick={() => setShowHistory(!showHistory)}
          className="flex items-center justify-between w-full text-left"
        >
          <h2 className="text-lg font-semibold text-gray-900 dark:text-gray-100 flex items-center">
            <RiHistoryLine className="w-5 h-5 mr-2" />
            Previous Runs
          </h2>
          <span className="text-gray-400">{showHistory ? "−" : "+"}</span>
        </button>

        {showHistory && (
          <div className="mt-4">
            {history.length === 0 ? (
              <p className="text-gray-500 dark:text-gray-400">No previous runs found.</p>
            ) : (
              <div className="space-y-2">
                {history.map((run) => (
                  <button
                    key={run.run_id}
                    onClick={() => handleLoadHistoryResult(run)}
                    className="w-full flex items-center justify-between p-3 rounded-lg bg-gray-50 dark:bg-gray-800 hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors"
                  >
                    <div className="text-left">
                      <div className="font-medium text-gray-900 dark:text-gray-100">
                        {formatStrategyDate(run.created_at)}
                      </div>
                      <div className="text-sm text-gray-500 dark:text-gray-400">
                        {run.quarters_analyzed} quarters &bull; Best: {run.best_holding_period}Q hold
                        &bull;{" "}
                        {run.duration_seconds ? `${run.duration_seconds.toFixed(1)}s` : "—"}
                      </div>
                    </div>
                    <div className="flex items-center gap-3">
                      {run.best_alpha && (
                        <span
                          className={`font-medium ${getAlphaColorClass(run.best_alpha)}`}
                        >
                          {run.best_alpha > 0 ? "+" : ""}
                          {run.best_alpha.toFixed(1)}%
                        </span>
                      )}
                      <RiArrowRightLine className="w-4 h-4 text-gray-400" />
                    </div>
                  </button>
                ))}
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
