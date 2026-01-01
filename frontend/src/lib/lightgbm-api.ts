/**
 * LightGBM API Client
 *
 * Handles all API calls to the LightGBM ML model backend.
 */

const API_BASE = "http://localhost:8000/api/v1";

// ============================================================================
// Types
// ============================================================================

export interface LightGBMRequest {
  quarters: string[];
  holding_period: number;
  train_end_quarter?: string | null;
  features?: string[] | null;
  n_optuna_trials: number;
  winsorize_percentile: number;
  target_type?: string; // "raw", "beta_adjusted", "sector_adjusted", "full_adjusted"
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

export interface FeatureImportanceResult {
  feature_name: string;
  importance_gain: number;
  importance_split: number;
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

export interface LightGBMResult {
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
  best_params: Record<string, number>;
  n_features_selected: number;
  // Config
  holding_period: number;
  train_end_quarter: string | null;
  // Data
  feature_importances: FeatureImportanceResult[];
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
 * Start a new LightGBM training run.
 */
export async function startLightGBM(
  request: LightGBMRequest
): Promise<RunResponse> {
  const response = await fetch(`${API_BASE}/lightgbm/run`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(request),
  });

  if (!response.ok) {
    const error = await response.text();
    throw new Error(`Failed to start LightGBM training: ${error}`);
  }

  return response.json();
}

/**
 * Get progress updates via Server-Sent Events.
 * Returns an EventSource that emits ProgressUpdate objects.
 */
export function getProgressStream(runId: string): EventSource {
  return new EventSource(`${API_BASE}/lightgbm/progress/${runId}`);
}

/**
 * Get complete results for a LightGBM run.
 */
export async function getResults(runId: string): Promise<LightGBMResult> {
  const response = await fetch(`${API_BASE}/lightgbm/results/${runId}`);

  if (!response.ok) {
    throw new Error(`Failed to get results: ${response.statusText}`);
  }

  return response.json();
}

/**
 * Get feature importance for a run.
 */
export async function getFeatureImportance(
  runId: string,
  limit = 50
): Promise<FeatureImportanceResult[]> {
  const response = await fetch(
    `${API_BASE}/lightgbm/feature-importance/${runId}?limit=${limit}`
  );

  if (!response.ok) {
    throw new Error(`Failed to get feature importance: ${response.statusText}`);
  }

  return response.json();
}

/**
 * Get IC history for a run.
 */
export async function getICHistory(runId: string): Promise<ICHistoryPoint[]> {
  const response = await fetch(`${API_BASE}/lightgbm/ic-history/${runId}`);

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
    `${API_BASE}/lightgbm/predictions/${runId}?limit=${limit}`
  );

  if (!response.ok) {
    throw new Error(`Failed to get predictions: ${response.statusText}`);
  }

  return response.json();
}

/**
 * Get list of past LightGBM runs.
 */
export async function getHistory(limit = 50): Promise<RunSummary[]> {
  const response = await fetch(`${API_BASE}/lightgbm/history?limit=${limit}`);

  if (!response.ok) {
    throw new Error(`Failed to get history: ${response.statusText}`);
  }

  return response.json();
}

/**
 * Cancel a running training.
 */
export async function cancelRun(runId: string): Promise<{ status: string; run_id: string }> {
  const response = await fetch(`${API_BASE}/lightgbm/cancel/${runId}`, {
    method: "POST",
  });

  if (!response.ok) {
    throw new Error(`Failed to cancel run: ${response.statusText}`);
  }

  return response.json();
}

/**
 * Get available features for LightGBM.
 */
export async function getFeatures(): Promise<FeaturesResponse> {
  const response = await fetch(`${API_BASE}/lightgbm/features`);

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
 * Format feature importance for display.
 */
export function formatImportance(importance: number): string {
  if (importance < 0.001) return "~0";
  return importance.toFixed(4);
}

/**
 * Get color class based on importance value.
 */
export function getImportanceColorClass(importance: number): string {
  if (importance > 0) return "text-blue-600 dark:text-blue-400";
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
 * Get importance rank badge color.
 */
export function getImportanceRankColorClass(rank: number): string {
  if (rank <= 5) return "bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200";
  if (rank <= 15) return "bg-blue-100 text-blue-800 dark:bg-blue-900 dark:text-blue-200";
  if (rank <= 30) return "bg-yellow-100 text-yellow-800 dark:bg-yellow-900 dark:text-yellow-200";
  return "bg-gray-100 text-gray-800 dark:bg-gray-900 dark:text-gray-200";
}

/**
 * Get stage display name.
 */
export function getStageDisplayName(stage: string): string {
  const stageNames: Record<string, string> = {
    initializing: "Initializing",
    init: "Initializing",
    loading_data: "Loading Data",
    prepare: "Preparing Features",
    preprocessing: "Preprocessing Features",
    preprocess: "Preprocessing Features",
    split: "Splitting Train/Val/Test",
    optuna: "Tuning Hyperparameters",
    train: "Training Final Model",
    importance: "Extracting Feature Importance",
    ic: "Calculating IC History",
    predict: "Generating Predictions",
    saving: "Saving Results",
    done: "Complete",
    complete: "Complete",
  };
  return stageNames[stage] || stage;
}

/**
 * Format best params for display.
 */
export function formatBestParams(params: Record<string, number>): string {
  if (!params || Object.keys(params).length === 0) return "—";
  const items = Object.entries(params).map(([k, v]) => {
    const formattedValue = typeof v === "number" && !Number.isInteger(v)
      ? v.toFixed(4)
      : v;
    return `${formatFeatureName(k)}: ${formattedValue}`;
  });
  return items.join(", ");
}
