/**
 * Elastic Net API Client
 *
 * Handles all API calls to the Elastic Net ML model backend.
 */

const API_BASE = "http://localhost:8000/api/v1";

// ============================================================================
// Types
// ============================================================================

export interface ElasticNetRequest {
  quarters: string[];
  holding_period: number;
  train_end_quarter?: string | null;
  features?: string[] | null;
  l1_ratios?: number[] | null;
  cv_folds: number;
  winsorize_percentile: number;
}

export interface RunResponse {
  run_id: string;
  status: string;
  estimated_duration_seconds: number;
}

export interface ProgressUpdate {
  status: string;
  progress: number;
  stage: string;
  message: string;
  error?: string;
  cancelled?: boolean;
  result_run_id?: string;
}

export interface CoefficientResult {
  feature_name: string;
  coefficient: number;
  coefficient_std: number;
  stability_score: number;
  importance_rank: number;
}

export interface ICHistoryPoint {
  quarter: string;
  ic: number;
  ic_pvalue: number;
  n_samples: number;
}

export interface StockPrediction {
  symbol: string;
  predicted_alpha: number;
  predicted_rank: number;
}

export interface ElasticNetResult {
  run_id: string;
  status: string;
  error_message: string | null;
  duration_seconds: number;
  // Performance metrics
  train_ic: number | null;
  test_ic: number | null;
  n_train_samples: number;
  n_test_samples: number;
  // Model parameters
  best_alpha: number | null;
  best_l1_ratio: number | null;
  n_features_selected: number;
  // Config
  holding_period: number;
  train_end_quarter: string | null;
  // Data
  coefficients: CoefficientResult[];
  ic_history: ICHistoryPoint[];
  predictions: StockPrediction[];
}

export interface RunSummary {
  run_id: string;
  status: string;
  created_at: string;
  holding_period: number;
  train_ic: number | null;
  test_ic: number | null;
  n_features_selected: number;
}

export interface FeaturesResponse {
  features: string[];
  total: number;
}

// ============================================================================
// API Functions
// ============================================================================

/**
 * Start a new Elastic Net training run.
 */
export async function startElasticNet(
  request: ElasticNetRequest
): Promise<RunResponse> {
  const response = await fetch(`${API_BASE}/elastic-net/run`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(request),
  });

  if (!response.ok) {
    const error = await response.text();
    throw new Error(`Failed to start Elastic Net training: ${error}`);
  }

  return response.json();
}

/**
 * Get progress updates via Server-Sent Events.
 * Returns an EventSource that emits ProgressUpdate objects.
 */
export function getProgressStream(runId: string): EventSource {
  return new EventSource(`${API_BASE}/elastic-net/progress/${runId}`);
}

/**
 * Get complete results for an Elastic Net run.
 */
export async function getResults(runId: string): Promise<ElasticNetResult> {
  const response = await fetch(`${API_BASE}/elastic-net/results/${runId}`);

  if (!response.ok) {
    throw new Error(`Failed to get results: ${response.statusText}`);
  }

  return response.json();
}

/**
 * Get coefficients for a run.
 */
export async function getCoefficients(
  runId: string,
  limit = 50
): Promise<CoefficientResult[]> {
  const response = await fetch(
    `${API_BASE}/elastic-net/coefficients/${runId}?limit=${limit}`
  );

  if (!response.ok) {
    throw new Error(`Failed to get coefficients: ${response.statusText}`);
  }

  return response.json();
}

/**
 * Get IC history for a run.
 */
export async function getICHistory(runId: string): Promise<ICHistoryPoint[]> {
  const response = await fetch(`${API_BASE}/elastic-net/ic-history/${runId}`);

  if (!response.ok) {
    throw new Error(`Failed to get IC history: ${response.statusText}`);
  }

  return response.json();
}

/**
 * Get stock predictions for a run.
 */
export async function getPredictions(
  runId: string,
  limit = 100
): Promise<StockPrediction[]> {
  const response = await fetch(
    `${API_BASE}/elastic-net/predictions/${runId}?limit=${limit}`
  );

  if (!response.ok) {
    throw new Error(`Failed to get predictions: ${response.statusText}`);
  }

  return response.json();
}

/**
 * Get list of past Elastic Net runs.
 */
export async function getHistory(limit = 50): Promise<RunSummary[]> {
  const response = await fetch(`${API_BASE}/elastic-net/history?limit=${limit}`);

  if (!response.ok) {
    throw new Error(`Failed to get history: ${response.statusText}`);
  }

  return response.json();
}

/**
 * Cancel a running training.
 */
export async function cancelRun(runId: string): Promise<{ status: string; run_id: string }> {
  const response = await fetch(`${API_BASE}/elastic-net/cancel/${runId}`, {
    method: "POST",
  });

  if (!response.ok) {
    throw new Error(`Failed to cancel run: ${response.statusText}`);
  }

  return response.json();
}

/**
 * Get available features for Elastic Net.
 */
export async function getFeatures(): Promise<FeaturesResponse> {
  const response = await fetch(`${API_BASE}/elastic-net/features`);

  if (!response.ok) {
    throw new Error(`Failed to get features: ${response.statusText}`);
  }

  return response.json();
}

// ============================================================================
// Utility Functions
// ============================================================================

/**
 * Format a feature name for display.
 */
export function formatFeatureName(name: string): string {
  // Convert snake_case to Title Case
  return name
    .replace(/_/g, " ")
    .replace(/\b\w/g, (l) => l.toUpperCase());
}

/**
 * Format IC value for display.
 */
export function formatIC(ic: number | null): string {
  if (ic === null) return "—";
  return ic.toFixed(4);
}

/**
 * Format coefficient for display.
 */
export function formatCoefficient(coef: number): string {
  if (Math.abs(coef) < 0.0001) return "~0";
  return coef.toFixed(4);
}

/**
 * Get color class based on coefficient value.
 */
export function getCoefficientColorClass(coef: number): string {
  if (coef > 0) return "text-green-600 dark:text-green-400";
  if (coef < 0) return "text-red-600 dark:text-red-400";
  return "text-gray-600 dark:text-gray-400";
}

/**
 * Get color class based on IC value.
 */
export function getICColorClass(ic: number | null): string {
  if (ic === null) return "text-gray-600 dark:text-gray-400";
  if (ic > 0.05) return "text-green-600 dark:text-green-400";
  if (ic > 0) return "text-green-500 dark:text-green-300";
  if (ic > -0.05) return "text-red-500 dark:text-red-300";
  return "text-red-600 dark:text-red-400";
}

/**
 * Get stability badge color.
 */
export function getStabilityColorClass(stability: number): string {
  if (stability >= 0.8) return "bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200";
  if (stability >= 0.6) return "bg-yellow-100 text-yellow-800 dark:bg-yellow-900 dark:text-yellow-200";
  return "bg-red-100 text-red-800 dark:bg-red-900 dark:text-red-200";
}

/**
 * Get stage display name.
 */
export function getStageDisplayName(stage: string): string {
  const stageNames: Record<string, string> = {
    initializing: "Initializing",
    loading_data: "Loading Data",
    preprocessing: "Preprocessing Features",
    splitting: "Splitting Train/Test",
    fitting: "Fitting Model",
    evaluating: "Evaluating Performance",
    calculating_stability: "Calculating Stability",
    generating_predictions: "Generating Predictions",
    saving: "Saving Results",
    complete: "Complete",
  };
  return stageNames[stage] || stage;
}
