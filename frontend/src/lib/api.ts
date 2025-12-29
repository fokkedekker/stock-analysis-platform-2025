const API_BASE = 'http://localhost:8000/api/v1';

// Types
export interface Stock {
  symbol: string;
  name: string;
  sector: string;
  exchange: string;
}

export interface RankingStock extends Stock {
  graham_pass: number;
  graham_score: number | null;
  magic_formula_pass: number;
  magic_formula_rank: number | null;
  piotroski_pass: number;
  piotroski_score: number | null;
  altman_pass: number;
  altman_z_score: number | null;
  altman_zone: string | null;
  roic_pass: number;
  roic_value: number | null;
  peg_pass: number;
  peg_ratio: number | null;
  fama_french_pass: number;
  fama_french_profitability: number | null;
  net_net_pass: number;
  net_net_discount: number | null;
  systems_passed: number;
}

export interface GrahamStock extends Stock {
  criteria_passed: number;
  pe_ratio: number | null;
  pb_ratio: number | null;
  revenue: number | null;
  adequate_size: boolean;
  current_ratio_pass: boolean;
  debt_coverage_pass: boolean;
  earnings_stability: boolean;
  dividend_record: boolean;
  earnings_growth_pass: boolean;
  pe_ratio_pass: boolean;
  pb_ratio_pass: boolean;
}

export interface MagicFormulaStock extends Stock {
  combined_rank: number;
  earnings_yield: number | null;
  return_on_capital: number | null;
  earnings_yield_rank: number | null;
  return_on_capital_rank: number | null;
}

export interface PiotroskiStock extends Stock {
  f_score: number;
  roa_positive: boolean;
  operating_cf_positive: boolean;
  roa_increasing: boolean;
  accruals_signal: boolean;
  leverage_decreasing: boolean;
  current_ratio_increasing: boolean;
  no_dilution: boolean;
  gross_margin_increasing: boolean;
  asset_turnover_increasing: boolean;
}

export interface AltmanStock extends Stock {
  z_score: number;
  zone: 'safe' | 'grey' | 'distress';
  x1_wc_ta: number | null;
  x2_re_ta: number | null;
  x3_ebit_ta: number | null;
  x4_mc_tl: number | null;
  x5_rev_ta: number | null;
}

export interface RoicStock extends Stock {
  roic: number;
  roic_pass: boolean;
  fcf_positive_5yr: boolean;
  debt_to_equity_pass: boolean;
  free_cash_flow: number | null;
  debt_to_equity: number | null;
}

export interface PegStock extends Stock {
  peg_ratio: number;
  pe_ratio: number | null;
  eps_cagr: number | null;
  growth_pass: boolean;
  peg_pass: boolean;
}

export interface FamaFrenchStock extends Stock {
  book_to_market: number | null;
  profitability: number | null;
  asset_growth: number | null;
  book_to_market_percentile: number | null;
  profitability_percentile: number | null;
  asset_growth_percentile: number | null;
}

export interface NetNetStock extends Stock {
  ncav: number | null;
  market_cap: number | null;
  discount_to_ncav: number;
  trading_below_ncav: boolean;
  deep_value: boolean;
}

// Pipeline types
export type QualityLabel = 'compounder' | 'average' | 'weak';
export type RankMethod = 'magic-formula' | 'earnings-yield' | 'roic' | 'peg' | 'graham-score' | 'net-net-discount';
export type ValuationLens = 'graham' | 'net-net' | 'peg' | 'magic-formula' | 'fama-french-bm';
export type QualityTag =
  | 'Durable Compounder'
  | 'Cash Machine'
  | 'Deep Value'
  | 'Heavy Reinvestor'
  | 'Volatile Returns'
  | 'Earnings Quality Concern'
  | 'Premium Priced'
  | 'Weak Moat Signal';

export const QUALITY_TAGS: QualityTag[] = [
  'Durable Compounder',
  'Cash Machine',
  'Deep Value',
  'Heavy Reinvestor',
  'Volatile Returns',
  'Earnings Quality Concern',
  'Premium Priced',
  'Weak Moat Signal',
];

export interface PipelineStock extends Stock {
  // Stage 1: Survival
  altman_z_score: number | null;
  altman_zone: string | null;
  altman_passed: boolean;
  piotroski_score: number | null;
  piotroski_passed: boolean;

  // Stage 2: Quality
  roic: number | null;
  free_cash_flow: number | null;
  fcf_positive_5yr: boolean | null;
  quality_label: QualityLabel;
  // NEW: Quality metrics (stability, valuation, tags)
  roic_stability_tag: string | null;
  gross_margin_stability_tag: string | null;
  fcf_yield: number | null;
  ev_to_ebit: number | null;
  valuation_tag: string | null;
  quality_tags: string[] | string | null; // Can be array or JSON string from API

  // Stage 3: Valuation
  graham_score: number | null;
  graham_pe: number | null;
  graham_pb: number | null;
  graham_passed: boolean;
  trading_below_ncav: boolean | null;
  net_net_discount: number | null;
  net_net_passed: boolean;
  peg_ratio: number | null;
  eps_cagr: number | null;
  peg_passed: boolean;
  magic_formula_rank: number | null;
  earnings_yield: number | null;
  mf_roic: number | null;
  magic_formula_passed: boolean;
  book_to_market_percentile: number | null;
  ff_bm_passed: boolean;
  lenses_passed: number;
  lenses_active: number;
  valuation_lenses_passed: ValuationLens[];

  // Stage 4: Factor Exposure
  profitability_percentile: number | null;
  asset_growth_percentile: number | null;
  book_to_market: number | null;
  profitability: number | null;
  asset_growth: number | null;
}

export interface PipelineConfig {
  survival: {
    require_altman: boolean;
    altman_zone: string;
    require_piotroski: boolean;
    piotroski_min: number;
  };
  quality: {
    filter_enabled: boolean;
    min_quality: string;
  };
  valuation: {
    min_lenses: number;
    strict_mode: boolean;
    lenses: {
      graham: { enabled: boolean; mode: string; min_score: number };
      net_net: { enabled: boolean };
      peg: { enabled: boolean; max_peg: number };
      magic_formula: { enabled: boolean; top_pct: number };
      fama_french_bm: { enabled: boolean; top_pct: number };
    };
  };
  rank_by: RankMethod;
}

export interface PipelineResponse {
  screen: 'pipeline';
  config: PipelineConfig;
  count: number;
  stocks: PipelineStock[];
}

export interface PipelineParams {
  // Stage 1
  require_altman?: boolean;
  altman_zone?: 'safe' | 'grey' | 'distress';
  require_piotroski?: boolean;
  piotroski_min?: number;
  // Stage 2
  quality_filter?: boolean;
  min_quality?: QualityLabel;
  quality_tags_filter?: QualityTag[];
  // Stage 3
  min_valuation_lenses?: number;
  strict_mode?: boolean;
  lens_graham?: boolean;
  lens_net_net?: boolean;
  lens_peg?: boolean;
  lens_magic_formula?: boolean;
  lens_fama_french_bm?: boolean;
  graham_mode?: 'strict' | 'modern' | 'garp' | 'relaxed';
  graham_min?: number;
  max_peg?: number;
  mf_top_pct?: number;
  ff_bm_top_pct?: number;
  // Ranking
  rank_by?: RankMethod;
  limit?: number;
}

export interface ScreenerResponse<T> {
  screen: string;
  count: number;
  stocks: T[];
}

export interface StockProfile {
  symbol: string;
  price: number | null;
  market_cap: number | null;
  pe_ratio: number | null;
  pb_ratio: number | null;
  beta: number | null;
  dividend_yield: number | null;
}

export interface StockAnalysis {
  symbol: string;
  graham: any;
  magic_formula: any;
  piotroski: any;
  altman: any;
  roic_quality: any;
  garp_peg: any;
  fama_french: any;
  net_net: any;
}

// Quarters
export interface QuartersResponse {
  quarters: string[];
  latest: string | null;
}

export async function getQuarters(): Promise<QuartersResponse> {
  const res = await fetch(`${API_BASE}/screener/quarters`);
  if (!res.ok) throw new Error('Failed to fetch quarters');
  return res.json();
}

// Stock Search
export interface SearchResult {
  symbol: string;
  name: string;
  exchange: string | null;
  sector: string | null;
}

export async function searchStocks(query: string, limit = 10): Promise<SearchResult[]> {
  if (!query || query.length < 1) return [];
  const res = await fetch(`${API_BASE}/tickers/search?q=${encodeURIComponent(query)}&limit=${limit}`);
  if (!res.ok) return [];
  return res.json();
}

// API Functions

// Rankings
export async function getRankings(minSystems = 1, limit = 100, quarter?: string | null): Promise<ScreenerResponse<RankingStock>> {
  let url = `${API_BASE}/screener/rankings?min_systems=${minSystems}&limit=${limit}`;
  if (quarter) url += `&quarter=${quarter}`;
  const res = await fetch(url);
  if (!res.ok) throw new Error('Failed to fetch rankings');
  return res.json();
}

// Pipeline
export async function getPipelineStocks(params: PipelineParams = {}, quarter?: string | null): Promise<PipelineResponse> {
  const searchParams = new URLSearchParams();

  // Stage 1
  if (params.require_altman !== undefined) searchParams.set('require_altman', String(params.require_altman));
  if (params.altman_zone) searchParams.set('altman_zone', params.altman_zone);
  if (params.require_piotroski !== undefined) searchParams.set('require_piotroski', String(params.require_piotroski));
  if (params.piotroski_min !== undefined) searchParams.set('piotroski_min', String(params.piotroski_min));

  // Stage 2
  if (params.quality_filter !== undefined) searchParams.set('quality_filter', String(params.quality_filter));
  if (params.min_quality) searchParams.set('min_quality', params.min_quality);
  if (params.quality_tags_filter && params.quality_tags_filter.length > 0) {
    searchParams.set('quality_tags_filter', params.quality_tags_filter.join(','));
  }

  // Stage 3
  if (params.min_valuation_lenses !== undefined) searchParams.set('min_valuation_lenses', String(params.min_valuation_lenses));
  if (params.strict_mode !== undefined) searchParams.set('strict_mode', String(params.strict_mode));
  if (params.lens_graham !== undefined) searchParams.set('lens_graham', String(params.lens_graham));
  if (params.lens_net_net !== undefined) searchParams.set('lens_net_net', String(params.lens_net_net));
  if (params.lens_peg !== undefined) searchParams.set('lens_peg', String(params.lens_peg));
  if (params.lens_magic_formula !== undefined) searchParams.set('lens_magic_formula', String(params.lens_magic_formula));
  if (params.lens_fama_french_bm !== undefined) searchParams.set('lens_fama_french_bm', String(params.lens_fama_french_bm));
  if (params.graham_mode) searchParams.set('graham_mode', params.graham_mode);
  if (params.graham_min !== undefined) searchParams.set('graham_min', String(params.graham_min));
  if (params.max_peg !== undefined) searchParams.set('max_peg', String(params.max_peg));
  if (params.mf_top_pct !== undefined) searchParams.set('mf_top_pct', String(params.mf_top_pct));
  if (params.ff_bm_top_pct !== undefined) searchParams.set('ff_bm_top_pct', String(params.ff_bm_top_pct));

  // Ranking
  if (params.rank_by) searchParams.set('rank_by', params.rank_by);
  if (params.limit !== undefined) searchParams.set('limit', String(params.limit));

  // Quarter
  if (quarter) searchParams.set('quarter', quarter);

  const url = `${API_BASE}/screener/pipeline?${searchParams.toString()}`;
  const res = await fetch(url);
  if (!res.ok) throw new Error('Failed to fetch pipeline stocks');
  return res.json();
}

// Graham
export async function getGrahamStocks(mode = 'strict', minScore = 0, limit = 100, quarter?: string | null): Promise<ScreenerResponse<GrahamStock>> {
  let url = `${API_BASE}/screener/graham?mode=${mode}&min_score=${minScore}&limit=${limit}`;
  if (quarter) url += `&quarter=${quarter}`;
  const res = await fetch(url);
  if (!res.ok) throw new Error('Failed to fetch Graham stocks');
  return res.json();
}

// Magic Formula
export async function getMagicFormulaStocks(top = 100, quarter?: string | null): Promise<ScreenerResponse<MagicFormulaStock>> {
  let url = `${API_BASE}/screener/magic-formula?top=${top}`;
  if (quarter) url += `&quarter=${quarter}`;
  const res = await fetch(url);
  if (!res.ok) throw new Error('Failed to fetch Magic Formula stocks');
  return res.json();
}

// Piotroski
export async function getPiotroskiStocks(minScore = 0, limit = 100, quarter?: string | null): Promise<ScreenerResponse<PiotroskiStock>> {
  let url = `${API_BASE}/screener/piotroski?min_score=${minScore}&limit=${limit}`;
  if (quarter) url += `&quarter=${quarter}`;
  const res = await fetch(url);
  if (!res.ok) throw new Error('Failed to fetch Piotroski stocks');
  return res.json();
}

// Altman
export async function getAltmanStocks(zone = 'safe', limit = 100, quarter?: string | null): Promise<ScreenerResponse<AltmanStock>> {
  let url = `${API_BASE}/screener/altman?zone=${zone}&limit=${limit}`;
  if (quarter) url += `&quarter=${quarter}`;
  const res = await fetch(url);
  if (!res.ok) throw new Error('Failed to fetch Altman stocks');
  return res.json();
}

// ROIC
export async function getRoicStocks(minRoic = 0, requireFcf = false, limit = 100, quarter?: string | null): Promise<ScreenerResponse<RoicStock>> {
  let url = `${API_BASE}/screener/roic?min_roic=${minRoic}&require_fcf=${requireFcf}&limit=${limit}`;
  if (quarter) url += `&quarter=${quarter}`;
  const res = await fetch(url);
  if (!res.ok) throw new Error('Failed to fetch ROIC stocks');
  return res.json();
}

// PEG
export async function getPegStocks(maxPeg = 10, minGrowth = 0, limit = 100, quarter?: string | null): Promise<ScreenerResponse<PegStock>> {
  let url = `${API_BASE}/screener/peg?max_peg=${maxPeg}&min_growth=${minGrowth}&limit=${limit}`;
  if (quarter) url += `&quarter=${quarter}`;
  const res = await fetch(url);
  if (!res.ok) throw new Error('Failed to fetch PEG stocks');
  return res.json();
}

// Fama-French
export async function getFamaFrenchStocks(
  minProfitability = 0,
  minBookToMarket = 0,
  maxAssetGrowth = 1,
  limit = 200,
  quarter?: string | null
): Promise<ScreenerResponse<FamaFrenchStock>> {
  let url = `${API_BASE}/screener/fama-french?min_profitability=${minProfitability}&min_book_to_market=${minBookToMarket}&max_asset_growth=${maxAssetGrowth}&limit=${limit}`;
  if (quarter) url += `&quarter=${quarter}`;
  const res = await fetch(url);
  if (!res.ok) throw new Error('Failed to fetch Fama-French stocks');
  return res.json();
}

// Net-Net
export async function getNetNetStocks(maxDiscount = 1, limit = 100, quarter?: string | null): Promise<ScreenerResponse<NetNetStock>> {
  let url = `${API_BASE}/screener/net-net?max_discount=${maxDiscount}&limit=${limit}`;
  if (quarter) url += `&quarter=${quarter}`;
  const res = await fetch(url);
  if (!res.ok) throw new Error('Failed to fetch Net-Net stocks');
  return res.json();
}

// Stock Detail
export async function getStockAnalysis(symbol: string): Promise<StockAnalysis> {
  const res = await fetch(`${API_BASE}/analysis/${symbol}`);
  if (!res.ok) throw new Error('Failed to fetch stock analysis');
  return res.json();
}

export async function getStockProfile(symbol: string): Promise<StockProfile> {
  const res = await fetch(`${API_BASE}/financials/${symbol}/profile`);
  if (!res.ok) throw new Error('Failed to fetch stock profile');
  return res.json();
}

export async function getIncomeStatements(symbol: string, period = 'annual', limit = 10) {
  const res = await fetch(`${API_BASE}/financials/${symbol}/income-statement?period=${period}&limit=${limit}`);
  if (!res.ok) throw new Error('Failed to fetch income statements');
  return res.json();
}

export async function getBalanceSheets(symbol: string, period = 'annual', limit = 10) {
  const res = await fetch(`${API_BASE}/financials/${symbol}/balance-sheet?period=${period}&limit=${limit}`);
  if (!res.ok) throw new Error('Failed to fetch balance sheets');
  return res.json();
}

export async function getCashFlows(symbol: string, period = 'annual', limit = 10) {
  const res = await fetch(`${API_BASE}/financials/${symbol}/cash-flow?period=${period}&limit=${limit}`);
  if (!res.ok) throw new Error('Failed to fetch cash flows');
  return res.json();
}

export async function getDividends(symbol: string, limit = 50) {
  const res = await fetch(`${API_BASE}/financials/${symbol}/dividends?limit=${limit}`);
  if (!res.ok) throw new Error('Failed to fetch dividends');
  return res.json();
}

// Utility
export function formatNumber(value: number | null | undefined, decimals = 2): string {
  if (value === null || value === undefined) return 'N/A';
  return value.toLocaleString(undefined, { minimumFractionDigits: decimals, maximumFractionDigits: decimals });
}

export function formatCurrency(value: number | null | undefined): string {
  if (value === null || value === undefined) return 'N/A';
  if (value >= 1e12) return `$${(value / 1e12).toFixed(2)}T`;
  if (value >= 1e9) return `$${(value / 1e9).toFixed(2)}B`;
  if (value >= 1e6) return `$${(value / 1e6).toFixed(2)}M`;
  return `$${value.toLocaleString()}`;
}

export function formatPercent(value: number | null | undefined): string {
  if (value === null || value === undefined) return 'N/A';
  return `${(value * 100).toFixed(2)}%`;
}

// AI Explain
export interface ChatMessage {
  role: 'user' | 'assistant';
  content: string;
}

export async function explainStock(
  symbol: string,
  messages: ChatMessage[]
): Promise<Response> {
  // Returns raw Response for streaming
  return fetch(`${API_BASE}/explain/${symbol}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ messages }),
  });
}

// Backtest types
export interface BacktestRequest {
  symbols: string[];
  buy_quarter: string;
  benchmark_return?: number;
}

export interface QuarterlyReturn {
  quarter: string;
  return_pct: number;
}

export interface StockReturn {
  symbol: string;
  name: string;
  buy_price: number;
  current_price: number;
  total_return: number;
  quarterly_returns: QuarterlyReturn[];
}

export interface BacktestResult {
  buy_quarter: string;
  latest_quarter: string;
  quarters_held: number;
  stocks: StockReturn[];
  winners: StockReturn[];
  losers: StockReturn[];
  portfolio_return: number;
  benchmark_return: number;
  alpha: number;
  quarterly_portfolio_returns: QuarterlyReturn[];
  quarterly_benchmark_returns: QuarterlyReturn[];
}

export async function simulateBuy(request: BacktestRequest): Promise<BacktestResult> {
  const response = await fetch(`${API_BASE}/backtest/simulate`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(request),
  });
  if (!response.ok) {
    const error = await response.json().catch(() => ({}));
    throw new Error(error.detail || 'Backtest simulation failed');
  }
  return response.json();
}
