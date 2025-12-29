/**
 * Grid Search API client functions
 */

const API_BASE = 'http://localhost:8000/api/v1';

// Types

export interface GridDimension {
  name: string;
  values: (string | number | boolean)[];
}

export interface SurvivalConfig {
  altman_enabled: boolean;
  altman_zone: 'safe' | 'grey' | 'distress';
  piotroski_enabled: boolean;
  piotroski_min: number;
}

export interface QualityConfig {
  enabled: boolean;
  min_quality: 'weak' | 'average' | 'compounder';
  required_tags: string[];
  excluded_tags: string[];
}

export interface ValuationConfig {
  graham_enabled: boolean;
  graham_mode: 'strict' | 'modern' | 'garp' | 'relaxed';
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

export interface StrategyConfig {
  id?: string;
  name?: string;
  survival: SurvivalConfig;
  quality: QualityConfig;
  valuation: ValuationConfig;
}

export interface GridSearchRequest {
  base_strategy: StrategyConfig;
  dimensions: GridDimension[];
  quarters: string[];
  holding_periods: number[];
}

export interface PreviewRequest {
  dimensions: GridDimension[];
  quarters: string[];
  holding_periods: number[];
}

export interface PreviewResponse {
  strategy_count: number;
  total_simulations: number;
  estimated_seconds: number;
  quarters_with_data: string[];
  holding_periods_valid: number[];
}

export interface StartResponse {
  search_id: string;
  status: string;
  total_simulations: number;
}

export interface GridSearchProgress {
  search_id: string;
  status: 'running' | 'completed' | 'failed';
  total_simulations: number;
  completed: number;
  current_strategy?: string;
  current_quarter?: string;
  estimated_remaining_seconds?: number;
  error?: string;
}

export interface SimulationResult {
  strategy_id: string;
  strategy_name: string;
  strategy_config: Record<string, unknown>;
  buy_quarter: string;
  sell_quarter: string;
  holding_period: number;
  stock_count: number;
  symbols: string[];
  portfolio_return: number;
  benchmark_return: number;
  alpha: number;
  win_rate: number;
  winners: number;
  losers: number;
}

export interface StrategyHoldingPeriod {
  holding_period: number;
  avg_alpha: number;
  avg_return: number;
  simulation_count: number;
}

export interface StrategyAggregate {
  strategy_id: string;
  strategy_name: string;
  strategy_config: Record<string, unknown>;
  simulation_count: number;
  avg_alpha: number;
  avg_return: number;
  avg_win_rate: number;
  avg_stock_count: number;
  min_alpha: number;
  max_alpha: number;
  by_holding_period: StrategyHoldingPeriod[];
  best_holding_period: number | null;
}

export interface HoldingPeriodAggregate {
  holding_period: number;
  simulation_count: number;
  avg_alpha: number;
  avg_return: number;
  avg_win_rate: number;
}

export interface GridSearchResults {
  id: string;
  started_at: string;
  completed_at: string | null;
  total_simulations: number;
  completed_simulations: number;
  duration_seconds: number;
  best_by_alpha: SimulationResult[];
  best_by_win_rate: SimulationResult[];
  by_strategy: StrategyAggregate[];
  by_holding_period: HoldingPeriodAggregate[];
  request_config: Record<string, unknown>;
  all_results_count: number;
}

export interface DimensionsResponse {
  dimensions: Record<string, (string | number | boolean)[]>;
  groups: Record<string, string[]>;
}

export interface QuartersResponse {
  analysis_quarters: string[];
  price_quarters: string[];
  recommended: string[];
}

// API Functions

export async function getAvailableDimensions(): Promise<DimensionsResponse> {
  const response = await fetch(`${API_BASE}/gridsearch/dimensions`);
  if (!response.ok) throw new Error('Failed to fetch dimensions');
  return response.json();
}

export async function getAvailableQuarters(): Promise<QuartersResponse> {
  const response = await fetch(`${API_BASE}/gridsearch/quarters`);
  if (!response.ok) throw new Error('Failed to fetch quarters');
  return response.json();
}

export async function previewGridSearch(request: PreviewRequest): Promise<PreviewResponse> {
  const response = await fetch(`${API_BASE}/gridsearch/preview`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(request),
  });
  if (!response.ok) throw new Error('Failed to preview grid search');
  return response.json();
}

export async function startGridSearch(request: GridSearchRequest): Promise<StartResponse> {
  const response = await fetch(`${API_BASE}/gridsearch/start`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(request),
  });
  if (!response.ok) throw new Error('Failed to start grid search');
  return response.json();
}

export async function getGridSearchStatus(searchId: string): Promise<{ status: string; completed?: number; total_simulations?: number }> {
  const response = await fetch(`${API_BASE}/gridsearch/status/${searchId}`);
  if (!response.ok) throw new Error('Failed to get status');
  return response.json();
}

export async function cancelGridSearch(searchId: string): Promise<{ cancelled: boolean; search_id: string }> {
  const response = await fetch(`${API_BASE}/gridsearch/cancel/${searchId}`, {
    method: 'POST',
  });
  if (!response.ok) throw new Error('Failed to cancel search');
  return response.json();
}

export async function getGridSearchResults(searchId: string): Promise<GridSearchResults> {
  const response = await fetch(`${API_BASE}/gridsearch/results/${searchId}`);
  if (!response.ok) {
    if (response.status === 202) {
      throw new Error('Search still running');
    }
    throw new Error('Failed to get results');
  }
  return response.json();
}

export async function getAllResults(
  searchId: string,
  sortBy: 'alpha' | 'win_rate' | 'stock_count' | 'portfolio_return' = 'alpha',
  limit: number = 100,
  offset: number = 0
): Promise<{ total: number; offset: number; limit: number; results: SimulationResult[] }> {
  const response = await fetch(
    `${API_BASE}/gridsearch/results/${searchId}/all?sort_by=${sortBy}&limit=${limit}&offset=${offset}`
  );
  if (!response.ok) throw new Error('Failed to get all results');
  return response.json();
}

export function createProgressEventSource(searchId: string): EventSource {
  return new EventSource(`${API_BASE}/gridsearch/progress/${searchId}`);
}

export interface SearchHistoryItem {
  id: string;
  name: string;
  started_at: string;
  completed_at: string | null;
  status: string;
  total_simulations: number;
  completed_simulations: number;
  duration_seconds: number;
  best_alpha: number | null;
  best_win_rate: number | null;
  dimension_count: number;
  quarter_count: number;
}

export interface SearchHistoryResponse {
  total: number;
  offset: number;
  limit: number;
  searches: SearchHistoryItem[];
}

export async function getSearchHistory(limit: number = 20, offset: number = 0): Promise<SearchHistoryResponse> {
  const response = await fetch(`${API_BASE}/gridsearch/history?limit=${limit}&offset=${offset}`);
  if (!response.ok) throw new Error('Failed to get search history');
  return response.json();
}

// Default strategy config - matches Pipeline page defaults
export function getDefaultStrategyConfig(): StrategyConfig {
  return {
    survival: {
      altman_enabled: true,
      altman_zone: 'safe',
      piotroski_enabled: true,
      piotroski_min: 6,
    },
    quality: {
      enabled: true,
      min_quality: 'compounder',
      required_tags: [],
      excluded_tags: [],
    },
    valuation: {
      graham_enabled: true,
      graham_mode: 'strict',
      graham_min: 6,
      magic_formula_enabled: true,
      mf_top_pct: 20,
      peg_enabled: true,
      max_peg: 1.5,
      net_net_enabled: true,
      fama_french_enabled: true,
      ff_top_pct: 30,
      min_lenses: 3,
      strict_mode: false,
    },
  };
}

// Dimension display helpers
export const DIMENSION_LABELS: Record<string, string> = {
  altman_zone: 'Altman Zone',
  piotroski_min: 'Piotroski Min',
  quality_enabled: 'Quality Filter',
  min_quality: 'Min Quality',
  graham_enabled: 'Graham',
  graham_mode: 'Graham Mode',
  graham_min: 'Graham Min',
  magic_formula_enabled: 'Magic Formula',
  mf_top_pct: 'MF Top %',
  peg_enabled: 'PEG',
  max_peg: 'Max PEG',
  net_net_enabled: 'Net-Net',
  fama_french_enabled: 'Fama-French',
  ff_top_pct: 'FF Top %',
  min_lenses: 'Min Lenses',
  strict_mode: 'Strict Mode',
  tag_durable_compounder: 'Durable Compounder',
  tag_cash_machine: 'Cash Machine',
  tag_deep_value: 'Deep Value',
  tag_heavy_reinvestor: 'Heavy Reinvestor',
  tag_premium_priced: 'Premium Priced',
  tag_volatile_returns: 'Volatile Returns',
  tag_weak_moat_signal: 'Weak Moat Signal',
  tag_earnings_quality_concern: 'Earnings Quality Concern',
};

export const DIMENSION_GROUPS = {
  survival: {
    label: 'Survival Gates',
    dimensions: ['altman_zone', 'piotroski_min'],
  },
  quality: {
    label: 'Quality Filters',
    dimensions: ['quality_enabled', 'min_quality'],
  },
  quality_tags: {
    label: 'Quality Tags',
    dimensions: [
      'tag_durable_compounder',
      'tag_cash_machine',
      'tag_deep_value',
      'tag_heavy_reinvestor',
      'tag_premium_priced',
      'tag_volatile_returns',
      'tag_weak_moat_signal',
      'tag_earnings_quality_concern',
    ],
  },
  valuation_graham: {
    label: 'Graham Lens',
    dimensions: ['graham_enabled', 'graham_mode', 'graham_min'],
  },
  valuation_magic_formula: {
    label: 'Magic Formula Lens',
    dimensions: ['magic_formula_enabled', 'mf_top_pct'],
  },
  valuation_peg: {
    label: 'PEG Lens',
    dimensions: ['peg_enabled', 'max_peg'],
  },
  valuation_net_net: {
    label: 'Net-Net Lens',
    dimensions: ['net_net_enabled'],
  },
  valuation_fama_french: {
    label: 'Fama-French Lens',
    dimensions: ['fama_french_enabled', 'ff_top_pct'],
  },
  valuation_logic: {
    label: 'Valuation Logic',
    dimensions: ['min_lenses', 'strict_mode'],
  },
};
