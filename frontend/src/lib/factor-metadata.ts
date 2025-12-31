/**
 * Factor Metadata
 *
 * Contains all available factors for the Advanced Filters panel.
 * This mirrors the factor definitions in the backend FactorAnalyzer.
 */

export interface FactorDefinition {
  name: string;
  label: string;
  category: FactorCategory;
  direction: ">=" | "<="; // Default comparison direction (e.g., >= for ROE, <= for P/E)
  thresholds: number[]; // Suggested threshold values
  description?: string;
}

export type FactorCategory =
  | "valuation"
  | "profitability"
  | "liquidity"
  | "leverage"
  | "efficiency"
  | "stability"
  | "growth"
  | "scores"
  | "regime";

export interface CategoryInfo {
  id: FactorCategory;
  label: string;
  description: string;
}

// ============================================================================
// Factor Categories
// ============================================================================

export const FACTOR_CATEGORIES: CategoryInfo[] = [
  {
    id: "valuation",
    label: "Valuation Ratios",
    description: "Price-based metrics that measure stock value relative to fundamentals",
  },
  {
    id: "profitability",
    label: "Profitability",
    description: "Measures of how efficiently a company generates profits",
  },
  {
    id: "liquidity",
    label: "Liquidity",
    description: "Measures of a company's ability to meet short-term obligations",
  },
  {
    id: "leverage",
    label: "Leverage",
    description: "Measures of a company's debt and financial risk",
  },
  {
    id: "efficiency",
    label: "Efficiency",
    description: "Measures of how efficiently a company uses its assets",
  },
  {
    id: "stability",
    label: "Stability",
    description: "Measures of consistency and quality of earnings",
  },
  {
    id: "growth",
    label: "Growth",
    description: "Measures of historical and expected growth rates",
  },
  {
    id: "scores",
    label: "Pre-computed Scores",
    description: "Composite scores from valuation systems",
  },
  {
    id: "regime",
    label: "Market Regimes",
    description: "Macroeconomic regime factors based on treasury rates",
  },
];

// ============================================================================
// Factor Definitions by Category
// ============================================================================

export const VALUATION_FACTORS: FactorDefinition[] = [
  {
    name: "pe_ratio",
    label: "P/E Ratio",
    category: "valuation",
    direction: "<=",
    thresholds: [5, 10, 15, 20, 30],
    description: "Price-to-Earnings ratio",
  },
  {
    name: "pb_ratio",
    label: "P/B Ratio",
    category: "valuation",
    direction: "<=",
    thresholds: [0.5, 1.0, 1.5, 2.0, 3.0],
    description: "Price-to-Book ratio",
  },
  {
    name: "price_to_sales",
    label: "P/S Ratio",
    category: "valuation",
    direction: "<=",
    thresholds: [0.5, 1.0, 2.0, 3.0, 5.0],
    description: "Price-to-Sales ratio",
  },
  {
    name: "price_to_free_cash_flow",
    label: "P/FCF Ratio",
    category: "valuation",
    direction: "<=",
    thresholds: [5, 10, 15, 20, 30],
    description: "Price-to-Free Cash Flow ratio",
  },
  {
    name: "ev_to_sales",
    label: "EV/Sales",
    category: "valuation",
    direction: "<=",
    thresholds: [0.5, 1.0, 2.0, 3.0, 5.0],
    description: "Enterprise Value to Sales",
  },
  {
    name: "ev_to_ebitda",
    label: "EV/EBITDA",
    category: "valuation",
    direction: "<=",
    thresholds: [5, 8, 10, 12, 15],
    description: "Enterprise Value to EBITDA",
  },
  {
    name: "ev_to_ebit",
    label: "EV/EBIT",
    category: "valuation",
    direction: "<=",
    thresholds: [8, 10, 12, 15, 20],
    description: "Enterprise Value to EBIT",
  },
  {
    name: "fcf_yield",
    label: "FCF Yield",
    category: "valuation",
    direction: ">=",
    thresholds: [0.03, 0.05, 0.07, 0.10, 0.15],
    description: "Free Cash Flow Yield (FCF/EV)",
  },
  {
    name: "earnings_yield",
    label: "Earnings Yield",
    category: "valuation",
    direction: ">=",
    thresholds: [0.05, 0.08, 0.10, 0.12, 0.15],
    description: "Earnings Yield (EBIT/EV)",
  },
];

export const PROFITABILITY_FACTORS: FactorDefinition[] = [
  {
    name: "roe",
    label: "ROE",
    category: "profitability",
    direction: ">=",
    thresholds: [0.05, 0.10, 0.15, 0.20, 0.25],
    description: "Return on Equity",
  },
  {
    name: "roa",
    label: "ROA",
    category: "profitability",
    direction: ">=",
    thresholds: [0.03, 0.05, 0.08, 0.10, 0.15],
    description: "Return on Assets",
  },
  {
    name: "roic",
    label: "ROIC",
    category: "profitability",
    direction: ">=",
    thresholds: [0.08, 0.10, 0.12, 0.15, 0.20],
    description: "Return on Invested Capital",
  },
  {
    name: "return_on_tangible_assets",
    label: "ROTA",
    category: "profitability",
    direction: ">=",
    thresholds: [0.05, 0.10, 0.15, 0.20],
    description: "Return on Tangible Assets",
  },
  {
    name: "gross_profit_margin",
    label: "Gross Margin",
    category: "profitability",
    direction: ">=",
    thresholds: [0.20, 0.30, 0.40, 0.50, 0.60],
    description: "Gross Profit Margin",
  },
  {
    name: "operating_profit_margin",
    label: "Operating Margin",
    category: "profitability",
    direction: ">=",
    thresholds: [0.05, 0.10, 0.15, 0.20, 0.25],
    description: "Operating Profit Margin",
  },
  {
    name: "net_profit_margin",
    label: "Net Margin",
    category: "profitability",
    direction: ">=",
    thresholds: [0.03, 0.05, 0.08, 0.10, 0.15],
    description: "Net Profit Margin",
  },
];

export const LIQUIDITY_FACTORS: FactorDefinition[] = [
  {
    name: "current_ratio",
    label: "Current Ratio",
    category: "liquidity",
    direction: ">=",
    thresholds: [1.0, 1.5, 2.0, 2.5, 3.0],
    description: "Current Assets / Current Liabilities",
  },
  {
    name: "quick_ratio",
    label: "Quick Ratio",
    category: "liquidity",
    direction: ">=",
    thresholds: [0.5, 1.0, 1.5, 2.0],
    description: "(Current Assets - Inventory) / Current Liabilities",
  },
  {
    name: "cash_ratio",
    label: "Cash Ratio",
    category: "liquidity",
    direction: ">=",
    thresholds: [0.1, 0.2, 0.3, 0.5],
    description: "Cash / Current Liabilities",
  },
];

export const LEVERAGE_FACTORS: FactorDefinition[] = [
  {
    name: "debt_ratio",
    label: "Debt Ratio",
    category: "leverage",
    direction: "<=",
    thresholds: [0.3, 0.4, 0.5, 0.6, 0.7],
    description: "Total Debt / Total Assets",
  },
  {
    name: "debt_to_equity",
    label: "Debt/Equity",
    category: "leverage",
    direction: "<=",
    thresholds: [0.25, 0.50, 1.0, 1.5, 2.0],
    description: "Total Debt / Total Equity",
  },
  {
    name: "debt_to_assets",
    label: "Debt/Assets",
    category: "leverage",
    direction: "<=",
    thresholds: [0.2, 0.3, 0.4, 0.5, 0.6],
    description: "Total Debt / Total Assets",
  },
  {
    name: "net_debt_to_ebitda",
    label: "Net Debt/EBITDA",
    category: "leverage",
    direction: "<=",
    thresholds: [1, 2, 3, 4, 5],
    description: "Net Debt / EBITDA",
  },
  {
    name: "interest_coverage",
    label: "Interest Coverage",
    category: "leverage",
    direction: ">=",
    thresholds: [2, 4, 6, 8, 10],
    description: "EBIT / Interest Expense",
  },
];

export const EFFICIENCY_FACTORS: FactorDefinition[] = [
  {
    name: "asset_turnover",
    label: "Asset Turnover",
    category: "efficiency",
    direction: ">=",
    thresholds: [0.5, 1.0, 1.5, 2.0],
    description: "Revenue / Total Assets",
  },
  {
    name: "inventory_turnover",
    label: "Inventory Turnover",
    category: "efficiency",
    direction: ">=",
    thresholds: [3, 5, 7, 10, 15],
    description: "COGS / Average Inventory",
  },
  {
    name: "receivables_turnover",
    label: "Receivables Turnover",
    category: "efficiency",
    direction: ">=",
    thresholds: [5, 8, 10, 12, 15],
    description: "Revenue / Average Receivables",
  },
  {
    name: "payables_turnover",
    label: "Payables Turnover",
    category: "efficiency",
    direction: ">=",
    thresholds: [4, 6, 8, 10, 12],
    description: "COGS / Average Payables",
  },
];

export const STABILITY_FACTORS: FactorDefinition[] = [
  {
    name: "roic_std_dev",
    label: "ROIC Std Dev",
    category: "stability",
    direction: "<=",
    thresholds: [0.02, 0.05, 0.08, 0.10],
    description: "Standard deviation of ROIC over 5 years (lower = more stable)",
  },
  {
    name: "gross_margin_std_dev",
    label: "Gross Margin Std Dev",
    category: "stability",
    direction: "<=",
    thresholds: [0.02, 0.05, 0.08, 0.10],
    description: "Standard deviation of Gross Margin over 5 years",
  },
  {
    name: "fcf_to_net_income",
    label: "FCF/Net Income",
    category: "stability",
    direction: ">=",
    thresholds: [0.5, 0.8, 1.0, 1.2],
    description: "Free Cash Flow / Net Income (earnings quality)",
  },
  {
    name: "reinvestment_rate",
    label: "Reinvestment Rate",
    category: "stability",
    direction: ">=",
    thresholds: [0.2, 0.4, 0.6, 0.8],
    description: "CapEx / Operating Cash Flow",
  },
];

export const GROWTH_FACTORS: FactorDefinition[] = [
  {
    name: "eps_growth_1yr",
    label: "EPS Growth 1yr",
    category: "growth",
    direction: ">=",
    thresholds: [0.05, 0.10, 0.15, 0.20, 0.30],
    description: "1-year EPS growth rate",
  },
  {
    name: "eps_growth_3yr",
    label: "EPS Growth 3yr",
    category: "growth",
    direction: ">=",
    thresholds: [0.05, 0.10, 0.15, 0.20],
    description: "3-year EPS growth rate",
  },
  {
    name: "eps_growth_5yr",
    label: "EPS Growth 5yr",
    category: "growth",
    direction: ">=",
    thresholds: [0.05, 0.10, 0.15],
    description: "5-year EPS growth rate",
  },
  {
    name: "eps_cagr",
    label: "EPS CAGR",
    category: "growth",
    direction: ">=",
    thresholds: [0.05, 0.10, 0.15, 0.20],
    description: "Compound Annual Growth Rate of EPS",
  },
];

export const SCORE_FACTORS: FactorDefinition[] = [
  {
    name: "graham_score",
    label: "Graham Score",
    category: "scores",
    direction: ">=",
    thresholds: [4, 5, 6, 7, 8],
    description: "Graham criteria passed (0-8)",
  },
  {
    name: "piotroski_score",
    label: "Piotroski F-Score",
    category: "scores",
    direction: ">=",
    thresholds: [5, 6, 7, 8, 9],
    description: "Piotroski F-Score (0-9)",
  },
  {
    name: "magic_formula_rank",
    label: "Magic Formula Rank",
    category: "scores",
    direction: "<=",
    thresholds: [50, 100, 200, 500],
    description: "Combined rank from Magic Formula",
  },
  {
    name: "peg_ratio",
    label: "PEG Ratio",
    category: "scores",
    direction: "<=",
    thresholds: [0.5, 1.0, 1.5, 2.0],
    description: "Price/Earnings to Growth ratio",
  },
];

export const REGIME_FACTORS: FactorDefinition[] = [
  {
    name: "rate_momentum",
    label: "Rate Momentum",
    category: "regime",
    direction: ">=",
    thresholds: [-0.5, -0.25, 0, 0.25, 0.5],
    description: "10Y Treasury quarter-over-quarter change (percentage points)",
  },
];

// ============================================================================
// All Factors Combined
// ============================================================================

export const ALL_FACTORS: FactorDefinition[] = [
  ...VALUATION_FACTORS,
  ...PROFITABILITY_FACTORS,
  ...LIQUIDITY_FACTORS,
  ...LEVERAGE_FACTORS,
  ...EFFICIENCY_FACTORS,
  ...STABILITY_FACTORS,
  ...GROWTH_FACTORS,
  ...SCORE_FACTORS,
  ...REGIME_FACTORS,
];

// ============================================================================
// Helper Functions
// ============================================================================

/**
 * Get factors grouped by category.
 */
export function getFactorsByCategory(): Record<FactorCategory, FactorDefinition[]> {
  return {
    valuation: VALUATION_FACTORS,
    profitability: PROFITABILITY_FACTORS,
    liquidity: LIQUIDITY_FACTORS,
    leverage: LEVERAGE_FACTORS,
    efficiency: EFFICIENCY_FACTORS,
    stability: STABILITY_FACTORS,
    growth: GROWTH_FACTORS,
    scores: SCORE_FACTORS,
    regime: REGIME_FACTORS,
  };
}

/**
 * Get a factor definition by name.
 */
export function getFactorByName(name: string): FactorDefinition | undefined {
  return ALL_FACTORS.find((f) => f.name === name);
}

/**
 * Format a factor value for display.
 */
export function formatFactorValue(value: number, factor: FactorDefinition): string {
  // Percentage values (ratios under 1 that represent percentages)
  const percentageFactors = [
    "roe",
    "roa",
    "roic",
    "return_on_tangible_assets",
    "gross_profit_margin",
    "operating_profit_margin",
    "net_profit_margin",
    "fcf_yield",
    "earnings_yield",
    "eps_growth_1yr",
    "eps_growth_3yr",
    "eps_growth_5yr",
    "eps_cagr",
    "roic_std_dev",
    "gross_margin_std_dev",
  ];

  if (percentageFactors.includes(factor.name)) {
    return `${(value * 100).toFixed(1)}%`;
  }

  // Regular ratios
  if (value >= 100) {
    return value.toFixed(0);
  } else if (value >= 10) {
    return value.toFixed(1);
  }
  return value.toFixed(2);
}

/**
 * Get available operators for a factor.
 */
export function getOperators(): { value: string; label: string }[] {
  return [
    { value: ">=", label: ">=" },
    { value: "<=", label: "<=" },
    { value: ">", label: ">" },
    { value: "<", label: "<" },
    { value: "==", label: "=" },
  ];
}
