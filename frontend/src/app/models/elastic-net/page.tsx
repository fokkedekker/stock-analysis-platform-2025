"use client";

import { useState, useEffect } from "react";
import {
  RiPlayLine,
  RiStopLine,
  RiSaveLine,
  RiHistoryLine,
  RiArrowRightLine,
  RiLineChartLine,
} from "@remixicon/react";
import { Button } from "@/components/Button";
import { ProgressBar } from "@/components/ProgressBar";
import { InfoTooltip } from "@/components/InfoTooltip";
import {
  startElasticNet,
  getProgressStream,
  getResults,
  getHistory,
  getFeatures,
  cancelRun,
  formatFeatureName,
  formatIC,
  formatCoefficient,
  getCoefficientColorClass,
  getICColorClass,
  getStabilityColorClass,
  getStageDisplayName,
  type ElasticNetRequest,
  type ElasticNetResult,
  type RunSummary,
  type ProgressUpdate,
  type CoefficientResult,
  type ICHistoryPoint,
} from "@/lib/elastic-net-api";
import { getAvailableQuarters } from "@/lib/factor-discovery-api";

type ViewState = "configure" | "running" | "results";

export default function ElasticNetPage() {
  // View state
  const [viewState, setViewState] = useState<ViewState>("configure");

  // Configure state
  const [availableQuarters, setAvailableQuarters] = useState<string[]>([]);
  const [selectedQuarters, setSelectedQuarters] = useState<Set<string>>(new Set());
  const [holdingPeriod, setHoldingPeriod] = useState(4);
  const [trainEndQuarter, setTrainEndQuarter] = useState<string | null>(null);
  const [cvFolds, setCvFolds] = useState(5);
  const [winsorizePercentile, setWinsorizePercentile] = useState(0.01);
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [availableFeatures, setAvailableFeatures] = useState<string[]>([]);
  const [selectedFeatures, setSelectedFeatures] = useState<Set<string>>(new Set());

  // Running state
  const [runId, setRunId] = useState<string | null>(null);
  const [progress, setProgress] = useState(0);
  const [stage, setStage] = useState("");
  const [message, setMessage] = useState("");
  const [isCancelling, setIsCancelling] = useState(false);

  // Results state
  const [result, setResult] = useState<ElasticNetResult | null>(null);
  const [history, setHistory] = useState<RunSummary[]>([]);
  const [showHistory, setShowHistory] = useState(false);
  const [coeffSort, setCoeffSort] = useState<"importance" | "coefficient" | "stability">("importance");

  // Error state
  const [error, setError] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(false);

  // Load available quarters and features on mount
  useEffect(() => {
    async function loadData() {
      try {
        const [quartersData, featuresData] = await Promise.all([
          getAvailableQuarters(),
          getFeatures(),
        ]);
        setAvailableQuarters(quartersData.quarters);
        setSelectedQuarters(new Set(quartersData.quarters));
        setAvailableFeatures(featuresData.features);
        setSelectedFeatures(new Set(featuresData.features));

        // Set default train end quarter (75% through the data)
        if (quartersData.quarters.length > 4) {
          const splitIdx = Math.floor(quartersData.quarters.length * 0.75);
          setTrainEndQuarter(quartersData.quarters[splitIdx]);
        }
      } catch (err) {
        console.error("Failed to load data:", err);
        setError("Failed to load available data");
      }
    }
    loadData();
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
        setStage(data.stage);
        setMessage(data.message || "");

        if (data.status === "completed") {
          eventSource.close();
          // Use the result_run_id if available (the actual DB run ID)
          fetchResults(data.result_run_id || runId);
        } else if (data.status === "failed") {
          eventSource.close();
          setError(data.error || "Training failed");
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
      setViewState("results");
    } catch (err) {
      console.error("Failed to fetch results:", err);
      setError("Failed to load results");
      setViewState("configure");
    }
  };

  const handleStartTraining = async () => {
    if (selectedQuarters.size === 0) {
      setError("Please select at least one quarter");
      return;
    }

    setError(null);
    setIsLoading(true);

    try {
      const request: ElasticNetRequest = {
        quarters: Array.from(selectedQuarters).sort(),
        holding_period: holdingPeriod,
        train_end_quarter: trainEndQuarter,
        features: selectedFeatures.size === availableFeatures.length
          ? null // Use all features (default)
          : Array.from(selectedFeatures),
        cv_folds: cvFolds,
        winsorize_percentile: winsorizePercentile,
      };

      const response = await startElasticNet(request);
      setRunId(response.run_id);
      setProgress(0);
      setStage("initializing");
      setMessage("Starting training...");
      setViewState("running");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to start training");
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

  const handleLoadHistoryResult = async (summary: RunSummary) => {
    try {
      const data = await getResults(summary.run_id);
      setResult(data);
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

  const selectAllQuarters = () => setSelectedQuarters(new Set(availableQuarters));
  const deselectAllQuarters = () => setSelectedQuarters(new Set());

  const getSortedCoefficients = (coefficients: CoefficientResult[]) => {
    const sorted = [...coefficients];
    switch (coeffSort) {
      case "importance":
        return sorted.sort((a, b) => a.importance_rank - b.importance_rank);
      case "coefficient":
        return sorted.sort((a, b) => Math.abs(b.coefficient) - Math.abs(a.coefficient));
      case "stability":
        return sorted.sort((a, b) => b.stability_score - a.stability_score);
      default:
        return sorted;
    }
  };

  // Render based on view state
  if (viewState === "running") {
    return (
      <div className="p-6 max-w-4xl mx-auto">
        <div className="flex items-center gap-3 mb-6">
          <div className="p-3 bg-indigo-100 dark:bg-indigo-900 rounded-lg">
            <RiLineChartLine className="w-6 h-6 text-indigo-600 dark:text-indigo-400" />
          </div>
          <div>
            <h1 className="text-2xl font-bold text-gray-900 dark:text-gray-100">
              Elastic Net Training
            </h1>
            <p className="text-gray-600 dark:text-gray-400">
              Training model on historical data...
            </p>
          </div>
        </div>

        <div className="bg-white dark:bg-gray-900 rounded-xl border border-gray-200 dark:border-gray-800 p-8">
          <div className="text-center mb-6">
            <div className="text-4xl font-bold text-indigo-600 dark:text-indigo-400 mb-2">
              {Math.round(progress)}%
            </div>
            <div className="text-lg text-gray-600 dark:text-gray-400">
              {getStageDisplayName(stage)}
            </div>
            {message && (
              <div className="text-sm text-gray-500 dark:text-gray-500 mt-1">
                {message}
              </div>
            )}
          </div>

          <ProgressBar value={progress} max={100} className="mb-8" />

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
    const sortedCoefficients = getSortedCoefficients(result.coefficients);
    const nonZeroCoefficients = sortedCoefficients.filter(c => Math.abs(c.coefficient) > 0.0001);

    return (
      <div className="p-6 max-w-6xl mx-auto">
        <div className="flex items-center justify-between mb-6">
          <div className="flex items-center gap-3">
            <div className="p-3 bg-indigo-100 dark:bg-indigo-900 rounded-lg">
              <RiLineChartLine className="w-6 h-6 text-indigo-600 dark:text-indigo-400" />
            </div>
            <div>
              <h1 className="text-2xl font-bold text-gray-900 dark:text-gray-100">
                Elastic Net Results
              </h1>
              <p className="text-gray-600 dark:text-gray-400">
                Training completed in {result.duration_seconds.toFixed(1)}s
              </p>
            </div>
          </div>
          <Button variant="secondary" onClick={() => setViewState("configure")}>
            New Training
          </Button>
        </div>

        {/* Summary Metrics */}
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-6">
          <div className="bg-white dark:bg-gray-900 rounded-xl border border-gray-200 dark:border-gray-800 p-4">
            <div className="text-sm text-gray-500 dark:text-gray-400">
              Train IC
              <InfoTooltip content="Information Coefficient on training data. Measures correlation between predicted and actual returns. Higher is better. 0.05+ is good, 0.10+ is excellent." />
            </div>
            <div className={`text-2xl font-bold ${getICColorClass(result.train_ic)}`}>
              {formatIC(result.train_ic)}
            </div>
            <div className="text-xs text-gray-500 dark:text-gray-500">
              {result.n_train_samples.toLocaleString()} samples
            </div>
          </div>

          <div className="bg-white dark:bg-gray-900 rounded-xl border border-gray-200 dark:border-gray-800 p-4">
            <div className="text-sm text-gray-500 dark:text-gray-400">
              Test IC
              <InfoTooltip content="Information Coefficient on out-of-sample test data. This is the true measure of model performance. Compare to Train IC to detect overfitting." />
            </div>
            <div className={`text-2xl font-bold ${getICColorClass(result.test_ic)}`}>
              {formatIC(result.test_ic)}
            </div>
            <div className="text-xs text-gray-500 dark:text-gray-500">
              {result.n_test_samples.toLocaleString()} samples
            </div>
          </div>

          <div className="bg-white dark:bg-gray-900 rounded-xl border border-gray-200 dark:border-gray-800 p-4">
            <div className="text-sm text-gray-500 dark:text-gray-400">
              Features Selected
              <InfoTooltip content="Number of features with non-zero coefficients. Elastic Net regularization shrinks unimportant coefficients to exactly zero." />
            </div>
            <div className="text-2xl font-bold text-gray-900 dark:text-gray-100">
              {result.n_features_selected}
            </div>
            <div className="text-xs text-gray-500 dark:text-gray-500">
              of {result.coefficients.length} total
            </div>
          </div>

          <div className="bg-white dark:bg-gray-900 rounded-xl border border-gray-200 dark:border-gray-800 p-4">
            <div className="text-sm text-gray-500 dark:text-gray-400">
              Model Parameters
              <InfoTooltip content="Best alpha (regularization strength) and L1 ratio found via cross-validation. L1=1 is pure Lasso, L1=0 is pure Ridge." />
            </div>
            <div className="text-lg font-bold text-gray-900 dark:text-gray-100">
              L1: {result.best_l1_ratio?.toFixed(2) ?? "—"}
            </div>
            <div className="text-xs text-gray-500 dark:text-gray-500">
              Alpha: {result.best_alpha?.toExponential(2) ?? "—"}
            </div>
          </div>
        </div>

        {/* Coefficients Table */}
        <div className="bg-white dark:bg-gray-900 rounded-xl border border-gray-200 dark:border-gray-800 p-6 mb-6">
          <div className="flex items-center justify-between mb-4">
            <h2 className="text-lg font-semibold text-gray-900 dark:text-gray-100">
              Feature Coefficients ({nonZeroCoefficients.length} non-zero)
            </h2>
            <div className="flex gap-2">
              <select
                value={coeffSort}
                onChange={(e) => setCoeffSort(e.target.value as typeof coeffSort)}
                className="px-3 py-1.5 rounded-lg border border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-800 text-sm"
              >
                <option value="importance">Sort by Importance</option>
                <option value="coefficient">Sort by Coefficient</option>
                <option value="stability">Sort by Stability</option>
              </select>
            </div>
          </div>

          <div className="overflow-x-auto">
            <table className="w-full">
              <thead>
                <tr className="border-b border-gray-200 dark:border-gray-700">
                  <th className="text-left py-3 px-2 text-sm font-medium text-gray-500 dark:text-gray-400">
                    Rank
                  </th>
                  <th className="text-left py-3 px-2 text-sm font-medium text-gray-500 dark:text-gray-400">
                    Feature
                  </th>
                  <th className="text-right py-3 px-2 text-sm font-medium text-gray-500 dark:text-gray-400">
                    Coefficient
                    <InfoTooltip content="The weight assigned to this feature. Positive = higher values predict higher returns. Negative = higher values predict lower returns." />
                  </th>
                  <th className="text-right py-3 px-2 text-sm font-medium text-gray-500 dark:text-gray-400">
                    Std Dev
                    <InfoTooltip content="Standard deviation of coefficient across CV folds. Lower = more consistent estimate." />
                  </th>
                  <th className="text-center py-3 px-2 text-sm font-medium text-gray-500 dark:text-gray-400">
                    Stability
                    <InfoTooltip content="Percentage of CV folds where this coefficient had the same sign. 100% = always consistent direction. Below 60% is concerning." />
                  </th>
                </tr>
              </thead>
              <tbody>
                {nonZeroCoefficients.slice(0, 30).map((coef) => (
                  <tr
                    key={coef.feature_name}
                    className="border-b border-gray-100 dark:border-gray-800 hover:bg-gray-50 dark:hover:bg-gray-800"
                  >
                    <td className="py-3 px-2 text-gray-500 dark:text-gray-400">
                      #{coef.importance_rank}
                    </td>
                    <td className="py-3 px-2 font-medium text-gray-900 dark:text-gray-100">
                      {formatFeatureName(coef.feature_name)}
                    </td>
                    <td className={`py-3 px-2 text-right font-mono ${getCoefficientColorClass(coef.coefficient)}`}>
                      {formatCoefficient(coef.coefficient)}
                    </td>
                    <td className="py-3 px-2 text-right text-gray-500 dark:text-gray-400 font-mono">
                      {coef.coefficient_std > 0 ? `+/-${coef.coefficient_std.toFixed(4)}` : "—"}
                    </td>
                    <td className="py-3 px-2 text-center">
                      <span className={`inline-flex items-center px-2 py-0.5 rounded text-xs font-medium ${getStabilityColorClass(coef.stability_score)}`}>
                        {(coef.stability_score * 100).toFixed(0)}%
                      </span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          {nonZeroCoefficients.length > 30 && (
            <p className="text-sm text-gray-500 dark:text-gray-400 mt-3">
              Showing top 30 of {nonZeroCoefficients.length} non-zero coefficients
            </p>
          )}
        </div>

        {/* IC History Chart */}
        {result.ic_history.length > 0 && (
          <div className="bg-white dark:bg-gray-900 rounded-xl border border-gray-200 dark:border-gray-800 p-6 mb-6">
            <h2 className="text-lg font-semibold text-gray-900 dark:text-gray-100 mb-4">
              IC Over Time
              <InfoTooltip content="Information Coefficient by quarter. Shows how well the model predicted returns in each period. Consistent positive IC indicates robust predictive power." />
            </h2>

            <div className="overflow-x-auto">
              <div className="flex items-end gap-1 h-32 min-w-max">
                {result.ic_history.map((point) => {
                  const maxIC = Math.max(...result.ic_history.map(p => Math.abs(p.ic)));
                  const heightPercent = maxIC > 0 ? (Math.abs(point.ic) / maxIC) * 100 : 0;
                  const isPositive = point.ic >= 0;

                  return (
                    <div
                      key={point.quarter}
                      className="flex flex-col items-center"
                      title={`${point.quarter}: IC=${point.ic.toFixed(4)}, n=${point.n_samples}`}
                    >
                      <div
                        className={`w-6 rounded-t ${isPositive ? "bg-green-500" : "bg-red-500"}`}
                        style={{ height: `${heightPercent}%`, minHeight: "4px" }}
                      />
                      <div className="text-xs text-gray-400 mt-1 -rotate-45 origin-top-left whitespace-nowrap">
                        {point.quarter}
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>

            <div className="flex items-center gap-6 mt-4 text-sm">
              <div className="flex items-center gap-2">
                <div className="w-3 h-3 bg-green-500 rounded" />
                <span className="text-gray-600 dark:text-gray-400">Positive IC</span>
              </div>
              <div className="flex items-center gap-2">
                <div className="w-3 h-3 bg-red-500 rounded" />
                <span className="text-gray-600 dark:text-gray-400">Negative IC</span>
              </div>
              <div className="text-gray-500 dark:text-gray-400 ml-auto">
                Mean IC: {formatIC(result.ic_history.reduce((sum, p) => sum + p.ic, 0) / result.ic_history.length)}
              </div>
            </div>
          </div>
        )}

        {/* Top Predictions */}
        {result.predictions.length > 0 && (
          <div className="bg-white dark:bg-gray-900 rounded-xl border border-gray-200 dark:border-gray-800 p-6 mb-6">
            <h2 className="text-lg font-semibold text-gray-900 dark:text-gray-100 mb-4">
              Top Stock Predictions
              <InfoTooltip content="Stocks with highest predicted alpha for the latest quarter. These predictions are based on the trained model applied to current fundamental data." />
            </h2>

            <div className="overflow-x-auto">
              <table className="w-full">
                <thead>
                  <tr className="border-b border-gray-200 dark:border-gray-700">
                    <th className="text-left py-3 px-2 text-sm font-medium text-gray-500 dark:text-gray-400">
                      Rank
                    </th>
                    <th className="text-left py-3 px-2 text-sm font-medium text-gray-500 dark:text-gray-400">
                      Symbol
                    </th>
                    <th className="text-right py-3 px-2 text-sm font-medium text-gray-500 dark:text-gray-400">
                      Predicted Alpha (Z-score)
                    </th>
                  </tr>
                </thead>
                <tbody>
                  {result.predictions.slice(0, 20).map((pred) => (
                    <tr
                      key={pred.symbol}
                      className="border-b border-gray-100 dark:border-gray-800 hover:bg-gray-50 dark:hover:bg-gray-800"
                    >
                      <td className="py-2 px-2 text-gray-500 dark:text-gray-400">
                        #{pred.predicted_rank}
                      </td>
                      <td className="py-2 px-2 font-medium text-gray-900 dark:text-gray-100">
                        <a
                          href={`/stock/${pred.symbol}`}
                          className="text-indigo-600 dark:text-indigo-400 hover:underline"
                        >
                          {pred.symbol}
                        </a>
                      </td>
                      <td className={`py-2 px-2 text-right font-mono ${getCoefficientColorClass(pred.predicted_alpha)}`}>
                        {pred.predicted_alpha > 0 ? "+" : ""}
                        {pred.predicted_alpha.toFixed(3)}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>

            {result.predictions.length > 20 && (
              <p className="text-sm text-gray-500 dark:text-gray-400 mt-3">
                Showing top 20 of {result.predictions.length} predictions
              </p>
            )}
          </div>
        )}

        {/* Pipeline Integration Notice */}
        <div className="bg-indigo-50 dark:bg-indigo-950 border border-indigo-200 dark:border-indigo-800 rounded-lg p-4 text-indigo-800 dark:text-indigo-200 text-sm">
          <strong>Pipeline Integration:</strong> This model&apos;s predictions can be used to filter stocks in the Pipeline.
          Use the model run ID <code className="bg-indigo-100 dark:bg-indigo-900 px-1 rounded">{result.run_id}</code> to
          apply this model&apos;s predictions to the screener.
        </div>
      </div>
    );
  }

  // Configure View (default)
  return (
    <div className="p-6 max-w-4xl mx-auto">
      <div className="flex items-center gap-3 mb-6">
        <div className="p-3 bg-indigo-100 dark:bg-indigo-900 rounded-lg">
          <RiLineChartLine className="w-6 h-6 text-indigo-600 dark:text-indigo-400" />
        </div>
        <div>
          <h1 className="text-2xl font-bold text-gray-900 dark:text-gray-100">
            Elastic Net Regression
          </h1>
          <p className="text-gray-600 dark:text-gray-400">
            Train a regularized linear model to predict cross-sectional alpha
          </p>
        </div>
      </div>

      {error && (
        <div className="bg-red-50 dark:bg-red-950 border border-red-200 dark:border-red-800 rounded-lg p-4 text-red-700 dark:text-red-300 mb-6">
          {error}
        </div>
      )}

      {/* Quarter Selection */}
      <div className="bg-white dark:bg-gray-900 rounded-xl border border-gray-200 dark:border-gray-800 p-6 mb-6">
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-lg font-semibold text-gray-900 dark:text-gray-100">
            Training Data
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

      {/* Holding Period */}
      <div className="bg-white dark:bg-gray-900 rounded-xl border border-gray-200 dark:border-gray-800 p-6 mb-6">
        <h2 className="text-lg font-semibold text-gray-900 dark:text-gray-100 mb-4">
          Holding Period
        </h2>

        <div className="flex gap-3">
          {[1, 2, 3, 4].map((hp) => (
            <button
              key={hp}
              onClick={() => setHoldingPeriod(hp)}
              className={`px-4 py-2 rounded-lg font-medium transition-colors ${
                holdingPeriod === hp
                  ? "bg-indigo-600 text-white"
                  : "bg-gray-100 dark:bg-gray-800 text-gray-700 dark:text-gray-300 hover:bg-gray-200 dark:hover:bg-gray-700"
              }`}
            >
              {hp} Quarter{hp > 1 ? "s" : ""}
            </button>
          ))}
        </div>

        <p className="text-sm text-gray-500 dark:text-gray-400 mt-3">
          Predict returns over this holding period
        </p>
      </div>

      {/* Train/Test Split */}
      <div className="bg-white dark:bg-gray-900 rounded-xl border border-gray-200 dark:border-gray-800 p-6 mb-6">
        <h2 className="text-lg font-semibold text-gray-900 dark:text-gray-100 mb-4">
          Train/Test Split
          <InfoTooltip content="The model trains on data before this quarter and tests on data after. This ensures no data leakage - we only test on future data the model hasn't seen." />
        </h2>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div>
            <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
              Train End Quarter
            </label>
            <select
              value={trainEndQuarter || ""}
              onChange={(e) => setTrainEndQuarter(e.target.value || null)}
              className="w-full px-3 py-2 rounded-lg border border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-800 text-gray-900 dark:text-gray-100"
            >
              <option value="">Auto (75% split)</option>
              {availableQuarters.slice(4, -2).map((q) => (
                <option key={q} value={q}>{q}</option>
              ))}
            </select>
            <p className="text-xs text-gray-500 mt-2">
              Data up to this quarter is used for training
            </p>
          </div>

          <div className="bg-gray-50 dark:bg-gray-800 rounded-lg p-4">
            <div className="text-sm text-gray-600 dark:text-gray-400">
              {trainEndQuarter ? (
                <>
                  <div><strong>Train:</strong> Quarters before {trainEndQuarter}</div>
                  <div><strong>Test:</strong> Quarters after {trainEndQuarter}</div>
                </>
              ) : (
                <>
                  <div><strong>Train:</strong> ~75% of selected quarters</div>
                  <div><strong>Test:</strong> ~25% of selected quarters</div>
                </>
              )}
            </div>
          </div>
        </div>
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
          <span className="text-gray-400">{showAdvanced ? "-" : "+"}</span>
        </button>

        {showAdvanced && (
          <div className="mt-4 space-y-6">
            {/* Cross-Validation */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div>
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                  CV Folds
                  <InfoTooltip content="Number of cross-validation folds. More folds = more robust coefficient estimates but slower training." />
                </label>
                <select
                  value={cvFolds}
                  onChange={(e) => setCvFolds(parseInt(e.target.value))}
                  className="w-full px-3 py-2 rounded-lg border border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-800 text-gray-900 dark:text-gray-100"
                >
                  {[3, 5, 10].map((n) => (
                    <option key={n} value={n}>{n} folds</option>
                  ))}
                </select>
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                  Winsorize Percentile
                  <InfoTooltip content="Clip extreme values at this percentile. Reduces impact of outliers. 0.01 means clip at 1st/99th percentile." />
                </label>
                <select
                  value={winsorizePercentile}
                  onChange={(e) => setWinsorizePercentile(parseFloat(e.target.value))}
                  className="w-full px-3 py-2 rounded-lg border border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-800 text-gray-900 dark:text-gray-100"
                >
                  <option value={0.005}>0.5% / 99.5%</option>
                  <option value={0.01}>1% / 99%</option>
                  <option value={0.02}>2% / 98%</option>
                  <option value={0.05}>5% / 95%</option>
                </select>
              </div>
            </div>

            {/* Feature Selection */}
            <div className="p-4 bg-blue-50 dark:bg-blue-950/30 rounded-lg border border-blue-200 dark:border-blue-800">
              <div className="flex items-center justify-between mb-3">
                <h3 className="text-sm font-semibold text-blue-900 dark:text-blue-100">
                  Features ({selectedFeatures.size} selected)
                </h3>
                <div className="flex gap-2">
                  <button
                    onClick={() => setSelectedFeatures(new Set(availableFeatures))}
                    className="text-xs text-blue-600 dark:text-blue-400 hover:underline"
                  >
                    Select All
                  </button>
                  <span className="text-gray-300">|</span>
                  <button
                    onClick={() => setSelectedFeatures(new Set())}
                    className="text-xs text-blue-600 dark:text-blue-400 hover:underline"
                  >
                    Clear
                  </button>
                </div>
              </div>

              <div className="flex flex-wrap gap-1 max-h-40 overflow-y-auto">
                {availableFeatures.map((feature) => (
                  <button
                    key={feature}
                    onClick={() => {
                      const next = new Set(selectedFeatures);
                      if (next.has(feature)) {
                        next.delete(feature);
                      } else {
                        next.add(feature);
                      }
                      setSelectedFeatures(next);
                    }}
                    className={`px-2 py-1 rounded text-xs font-medium transition-colors ${
                      selectedFeatures.has(feature)
                        ? "bg-blue-600 text-white"
                        : "bg-white dark:bg-gray-800 text-gray-600 dark:text-gray-400 border border-gray-200 dark:border-gray-700"
                    }`}
                  >
                    {formatFeatureName(feature)}
                  </button>
                ))}
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Run Button */}
      <div className="bg-white dark:bg-gray-900 rounded-xl border border-gray-200 dark:border-gray-800 p-6 mb-6">
        <div className="flex items-center justify-between">
          <div>
            <div className="text-sm text-gray-500 dark:text-gray-400">
              Ready to train
            </div>
            <div className="text-lg font-medium text-gray-900 dark:text-gray-100">
              {selectedQuarters.size} quarters, {selectedFeatures.size} features, {holdingPeriod}Q hold
            </div>
          </div>

          <Button
            variant="primary"
            onClick={handleStartTraining}
            disabled={isLoading || selectedQuarters.size === 0 || selectedFeatures.size === 0}
          >
            <RiPlayLine className="w-4 h-4 mr-2" />
            {isLoading ? "Starting..." : "Train Model"}
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
          <span className="text-gray-400">{showHistory ? "-" : "+"}</span>
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
                        {new Date(run.created_at).toLocaleString()}
                      </div>
                      <div className="text-sm text-gray-500 dark:text-gray-400">
                        {run.holding_period}Q hold &bull; {run.n_features_selected} features &bull; {run.status}
                      </div>
                    </div>
                    <div className="flex items-center gap-3">
                      {run.test_ic !== null && (
                        <span className={`font-medium ${getICColorClass(run.test_ic)}`}>
                          IC: {formatIC(run.test_ic)}
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
