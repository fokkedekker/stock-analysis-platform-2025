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

// API Functions

// Rankings
export async function getRankings(minSystems = 1, limit = 100): Promise<ScreenerResponse<RankingStock>> {
  const res = await fetch(`${API_BASE}/screener/rankings?min_systems=${minSystems}&limit=${limit}`);
  if (!res.ok) throw new Error('Failed to fetch rankings');
  return res.json();
}

// Graham
export async function getGrahamStocks(mode = 'strict', minScore = 0, limit = 100): Promise<ScreenerResponse<GrahamStock>> {
  const res = await fetch(`${API_BASE}/screener/graham?mode=${mode}&min_score=${minScore}&limit=${limit}`);
  if (!res.ok) throw new Error('Failed to fetch Graham stocks');
  return res.json();
}

// Magic Formula
export async function getMagicFormulaStocks(top = 100): Promise<ScreenerResponse<MagicFormulaStock>> {
  const res = await fetch(`${API_BASE}/screener/magic-formula?top=${top}`);
  if (!res.ok) throw new Error('Failed to fetch Magic Formula stocks');
  return res.json();
}

// Piotroski
export async function getPiotroskiStocks(minScore = 0, limit = 100): Promise<ScreenerResponse<PiotroskiStock>> {
  const res = await fetch(`${API_BASE}/screener/piotroski?min_score=${minScore}&limit=${limit}`);
  if (!res.ok) throw new Error('Failed to fetch Piotroski stocks');
  return res.json();
}

// Altman
export async function getAltmanStocks(zone = 'safe', limit = 100): Promise<ScreenerResponse<AltmanStock>> {
  const res = await fetch(`${API_BASE}/screener/altman?zone=${zone}&limit=${limit}`);
  if (!res.ok) throw new Error('Failed to fetch Altman stocks');
  return res.json();
}

// ROIC
export async function getRoicStocks(minRoic = 0, requireFcf = false, limit = 100): Promise<ScreenerResponse<RoicStock>> {
  const res = await fetch(`${API_BASE}/screener/roic?min_roic=${minRoic}&require_fcf=${requireFcf}&limit=${limit}`);
  if (!res.ok) throw new Error('Failed to fetch ROIC stocks');
  return res.json();
}

// PEG
export async function getPegStocks(maxPeg = 10, minGrowth = 0, limit = 100): Promise<ScreenerResponse<PegStock>> {
  const res = await fetch(`${API_BASE}/screener/peg?max_peg=${maxPeg}&min_growth=${minGrowth}&limit=${limit}`);
  if (!res.ok) throw new Error('Failed to fetch PEG stocks');
  return res.json();
}

// Fama-French
export async function getFamaFrenchStocks(
  minProfitability = 0,
  minBookToMarket = 0,
  maxAssetGrowth = 1,
  limit = 200
): Promise<ScreenerResponse<FamaFrenchStock>> {
  const res = await fetch(
    `${API_BASE}/screener/fama-french?min_profitability=${minProfitability}&min_book_to_market=${minBookToMarket}&max_asset_growth=${maxAssetGrowth}&limit=${limit}`
  );
  if (!res.ok) throw new Error('Failed to fetch Fama-French stocks');
  return res.json();
}

// Net-Net
export async function getNetNetStocks(maxDiscount = 1, limit = 100): Promise<ScreenerResponse<NetNetStock>> {
  const res = await fetch(`${API_BASE}/screener/net-net?max_discount=${maxDiscount}&limit=${limit}`);
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
