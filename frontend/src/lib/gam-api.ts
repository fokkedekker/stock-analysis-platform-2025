/**
 * GAM API Client
 *
 * Handles all API calls to the GAM ML model backend.
 */

const API_BASE = "http://localhost:8000/api/v1";

// ============================================================================
// Types
// ============================================================================

export interface GAMRequest {
  quarters: string[];
  holding_period: number;
  train_end_quarter?: string | null;
  features?: string[] | null;
  n_splines: number;
  lam: number;
  cv_folds: number;
  winsorize_percentile: number;
  target_type?: string;
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

export interface PartialDependence {
  feature_name: string;
  x_values: number[];
  y_values: number[];
  optimal_min: number | null;
  optimal_max: number | null;
  peak_x: number | null;
  peak_y: number;
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

export interface GAMResult {
  run_id: string;
  status: string;
  error_message: string | null;
  duration_seconds: number;
  // Performance metrics
  train_ic: number | null;
  test_ic: number | null;
  train_r2: number | null;
  n_train_samples: number;
  n_test_samples: number;
  // Model parameters
  n_features: number;
  best_lam: number | null;
  // Config
  holding_period: number;
  train_end_quarter: string | null;
  // Data
  partial_dependences: PartialDependence[];
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
  n_features: number;
}

export interface FeaturesResponse {
  features: string[];
  total: number;
}

// ============================================================================
// API Functions
// ============================================================================

/**
 * Start a new GAM training run.
 */
export async function startGAM(request: GAMRequest): Promise<RunResponse> {
  const response = await fetch(`${API_BASE}/gam/run`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(request),
  });

  if (!response.ok) {
    const error = await response.text();
    throw new Error(`Failed to start GAM training: ${error}`);
  }

  return response.json();
}

/**
 * Get progress updates via Server-Sent Events.
 */
export function getProgressStream(runId: string): EventSource {
  return new EventSource(`${API_BASE}/gam/progress/${runId}`);
}

/**
 * Get complete results for a GAM run.
 */
export async function getResults(runId: string): Promise<GAMResult> {
  const response = await fetch(`${API_BASE}/gam/results/${runId}`);

  if (!response.ok) {
    throw new Error(`Failed to get results: ${response.statusText}`);
  }

  return response.json();
}

/**
 * Get partial dependences for a run.
 */
export async function getPartialDependences(
  runId: string,
  limit = 50
): Promise<PartialDependence[]> {
  const response = await fetch(
    `${API_BASE}/gam/partial-dependence/${runId}?limit=${limit}`
  );

  if (!response.ok) {
    throw new Error(`Failed to get partial dependences: ${response.statusText}`);
  }

  return response.json();
}

/**
 * Get IC history for a run.
 */
export async function getICHistory(runId: string): Promise<ICHistoryPoint[]> {
  const response = await fetch(`${API_BASE}/gam/ic-history/${runId}`);

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
    `${API_BASE}/gam/predictions/${runId}?limit=${limit}`
  );

  if (!response.ok) {
    throw new Error(`Failed to get predictions: ${response.statusText}`);
  }

  return response.json();
}

/**
 * Get list of past GAM runs.
 */
export async function getHistory(limit = 50): Promise<RunSummary[]> {
  const response = await fetch(`${API_BASE}/gam/history?limit=${limit}`);

  if (!response.ok) {
    throw new Error(`Failed to get history: ${response.statusText}`);
  }

  return response.json();
}

/**
 * Cancel a running training.
 */
export async function cancelRun(runId: string): Promise<{ status: string; run_id: string }> {
  const response = await fetch(`${API_BASE}/gam/cancel/${runId}`, {
    method: "POST",
  });

  if (!response.ok) {
    throw new Error(`Failed to cancel run: ${response.statusText}`);
  }

  return response.json();
}

/**
 * Get available features for GAM.
 */
export async function getFeatures(): Promise<FeaturesResponse> {
  const response = await fetch(`${API_BASE}/gam/features`);

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
    split: "Splitting Train/Test",
    splitting: "Splitting Train/Test",
    fitting: "Fitting GAM Model",
    grid_search: "Optimizing Lambda",
    partial_dep: "Computing Partial Dependence",
    ic: "Calculating IC History",
    predict: "Generating Predictions",
    saving: "Saving Results",
    done: "Complete",
    complete: "Complete",
  };
  return stageNames[stage] || stage;
}

/**
 * Format optimal range for display.
 */
export function formatOptimalRange(
  min: number | null,
  max: number | null
): string {
  if (min === null || max === null) return "—";
  return `${min.toFixed(2)} to ${max.toFixed(2)}`;
}

/**
 * Get color class for peak effect.
 */
export function getPeakEffectColorClass(peak: number): string {
  if (peak > 0.01) return "text-green-600 dark:text-green-400";
  if (peak > 0) return "text-green-500 dark:text-green-300";
  if (peak > -0.01) return "text-red-500 dark:text-red-300";
  return "text-red-600 dark:text-red-400";
}
