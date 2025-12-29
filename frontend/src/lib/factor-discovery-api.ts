/**
 * Factor Discovery API Client
 *
 * Handles all API calls to the factor discovery backend.
 */

const API_BASE = "http://localhost:8000/api/v1";

// ============================================================================
// Types
// ============================================================================

export interface ExclusionFilters {
  exclude_altman_zones: string[];
  min_piotroski: number | null;
  exclude_quality_tags: string[];
  require_quality_tags: string[];
  exclude_penny_stocks: boolean;
  exclude_negative_earnings: boolean;
}

export interface FactorDiscoveryRequest {
  quarters: string[];
  holding_periods: number[];
  min_sample_size: number;
  significance_level: number;
  cost_haircut: number;
  portfolio_sizes: number[];
  ranking_method: string;
  max_factors: number;
  exclusions: ExclusionFilters;
}

export interface PortfolioStats {
  size: number;
  mean_alpha: number;
  sample_size: number;
  win_rate: number;
  ci_lower: number;
  ci_upper: number;
}

export interface ThresholdResult {
  threshold: string;
  mean_alpha: number;
  sample_size: number;
  lift: number;
  pvalue: number;
  ci_lower: number;
  ci_upper: number;
  win_rate: number;
}

export interface FactorResult {
  factor_name: string;
  factor_type: "numerical" | "categorical" | "boolean";
  holding_period: number;
  correlation: number | null;
  correlation_pvalue: number | null;
  threshold_results: ThresholdResult[];
  best_threshold: string | null;
  best_threshold_alpha: number | null;
  best_threshold_lift: number | null;
  best_threshold_pvalue: number | null;
  best_threshold_sample_size: number | null;
  best_threshold_ci_lower: number | null;
  best_threshold_ci_upper: number | null;
}

export interface FilterSpec {
  factor: string;
  operator: string;
  value: string | number | boolean;
}

export interface CombinedStrategyResult {
  filters: FilterSpec[];
  mean_alpha: number;
  sample_size: number;
  lift: number;
  win_rate: number;
  ci_lower: number;
  ci_upper: number;
  portfolio_stats: Record<number, PortfolioStats>;
}

export interface PipelineSettings {
  piotroski_enabled: boolean;
  piotroski_min: number;
  altman_enabled: boolean;
  altman_zone: string;
  quality_enabled: boolean;
  min_quality: string;
  excluded_tags: string[];
  required_tags: string[];
  graham_enabled: boolean;
  graham_mode: string;
  graham_min: number;
  magic_formula_enabled: boolean;
  mf_top_pct: number;
  peg_enabled: boolean;
  max_peg: number;
  net_net_enabled: boolean;
  fama_french_enabled: boolean;
  ff_top_pct: number;
  min_lenses: number;
  strict_mode: boolean;
}

export interface RecommendedStrategy {
  holding_period: number;
  pipeline_settings: PipelineSettings;
  expected_alpha: number;
  expected_alpha_ci_lower: number;
  expected_alpha_ci_upper: number;
  expected_win_rate: number;
  sample_size: number;
  confidence_score: number;
  key_factors: Array<{
    name: string;
    threshold: string;
    lift?: number;
    alpha?: number;
    pvalue?: number;
    sample_size?: number;
  }>;
  portfolio_stats: Record<number, PortfolioStats>;
}

export interface FactorDiscoveryResult {
  run_id: string;
  status: "running" | "completed" | "failed" | "cancelled";
  created_at: string;
  completed_at: string | null;
  duration_seconds: number | null;
  config: FactorDiscoveryRequest;
  total_observations: number;
  factor_results: Record<number, FactorResult[]>;
  combined_results: Record<number, CombinedStrategyResult[]>;
  recommended_strategies: Record<number, RecommendedStrategy>;
  best_holding_period: number | null;
  best_alpha: number | null;
}

export interface FactorDiscoverySummary {
  run_id: string;
  created_at: string;
  status: string;
  quarters_analyzed: number;
  best_holding_period: number | null;
  best_alpha: number | null;
  duration_seconds: number | null;
}

export interface RunResponse {
  run_id: string;
  status: string;
  estimated_duration_seconds: number;
}

export interface ProgressUpdate {
  status: string;
  phase: string;
  progress: number;
  current_factor?: string;
  current_holding_period?: number;
  error?: string;
}

export interface QuartersResponse {
  quarters: string[];
  total: number;
  latest: string | null;
}

export interface RecommendedResponse {
  best_holding_period: number | null;
  strategies: Record<number, RecommendedStrategy>;
}

export interface FactorConfig {
  name: string;
  label: string;
  thresholds?: number[];
  direction?: string;
  categories?: string[];
  positive?: boolean;
}

export interface FactorsResponse {
  numerical: FactorConfig[];
  categorical: FactorConfig[];
  boolean: FactorConfig[];
}

// ============================================================================
// API Functions
// ============================================================================

/**
 * Start a new factor discovery analysis.
 */
export async function startFactorDiscovery(
  request: FactorDiscoveryRequest
): Promise<RunResponse> {
  const response = await fetch(`${API_BASE}/factor-discovery/run`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(request),
  });

  if (!response.ok) {
    const error = await response.text();
    throw new Error(`Failed to start factor discovery: ${error}`);
  }

  return response.json();
}

/**
 * Get progress updates via Server-Sent Events.
 * Returns an EventSource that emits ProgressUpdate objects.
 */
export function getProgressStream(runId: string): EventSource {
  return new EventSource(`${API_BASE}/factor-discovery/progress/${runId}`);
}

/**
 * Get complete results for a factor discovery run.
 */
export async function getResults(runId: string): Promise<FactorDiscoveryResult> {
  const response = await fetch(`${API_BASE}/factor-discovery/results/${runId}`);

  if (!response.ok) {
    throw new Error(`Failed to get results: ${response.statusText}`);
  }

  return response.json();
}

/**
 * Get just the recommended strategies for a run.
 */
export async function getRecommended(
  runId: string,
  holdingPeriod?: number
): Promise<RecommendedResponse> {
  let url = `${API_BASE}/factor-discovery/results/${runId}/recommended`;
  if (holdingPeriod !== undefined) {
    url += `?holding_period=${holdingPeriod}`;
  }

  const response = await fetch(url);

  if (!response.ok) {
    throw new Error(`Failed to get recommendations: ${response.statusText}`);
  }

  return response.json();
}

/**
 * Get list of past factor discovery runs.
 */
export async function getHistory(limit = 50): Promise<FactorDiscoverySummary[]> {
  const response = await fetch(`${API_BASE}/factor-discovery/history?limit=${limit}`);

  if (!response.ok) {
    throw new Error(`Failed to get history: ${response.statusText}`);
  }

  return response.json();
}

/**
 * Cancel a running analysis.
 */
export async function cancelRun(runId: string): Promise<void> {
  const response = await fetch(`${API_BASE}/factor-discovery/cancel/${runId}`, {
    method: "POST",
  });

  if (!response.ok) {
    throw new Error(`Failed to cancel run: ${response.statusText}`);
  }
}

/**
 * Delete a factor discovery run and its results.
 */
export async function deleteRun(runId: string): Promise<void> {
  const response = await fetch(`${API_BASE}/factor-discovery/${runId}`, {
    method: "DELETE",
  });

  if (!response.ok) {
    throw new Error(`Failed to delete run: ${response.statusText}`);
  }
}

/**
 * Get available quarters for analysis.
 */
export async function getAvailableQuarters(): Promise<QuartersResponse> {
  const response = await fetch(`${API_BASE}/factor-discovery/quarters`);

  if (!response.ok) {
    throw new Error(`Failed to get quarters: ${response.statusText}`);
  }

  return response.json();
}

/**
 * Get list of all factors that will be analyzed.
 */
export async function getFactors(): Promise<FactorsResponse> {
  const response = await fetch(`${API_BASE}/factor-discovery/factors`);

  if (!response.ok) {
    throw new Error(`Failed to get factors: ${response.statusText}`);
  }

  return response.json();
}

// ============================================================================
// Utility Functions
// ============================================================================

/**
 * Format a factor name for display.
 */
export function formatFactorName(name: string): string {
  // Convert snake_case to Title Case
  return name
    .replace(/^has_/, "")
    .replace(/_/g, " ")
    .replace(/\b\w/g, (l) => l.toUpperCase());
}

/**
 * Format a p-value for display.
 */
export function formatPValue(pvalue: number | null): string {
  if (pvalue === null) return "—";
  if (pvalue < 0.001) return "< 0.001";
  return pvalue.toFixed(3);
}

/**
 * Get color class based on alpha value.
 */
export function getAlphaColorClass(alpha: number): string {
  if (alpha > 0) return "text-green-600 dark:text-green-400";
  if (alpha < 0) return "text-red-600 dark:text-red-400";
  return "text-gray-600 dark:text-gray-400";
}

/**
 * Get phase display name.
 */
export function getPhaseDisplayName(phase: string): string {
  const phaseNames: Record<string, string> = {
    building_dataset: "Building Dataset",
    analyzing_factors: "Analyzing Factors",
    finding_combinations: "Finding Combinations",
    generating_recommendations: "Generating Recommendations",
    complete: "Complete",
  };
  return phaseNames[phase] || phase;
}
