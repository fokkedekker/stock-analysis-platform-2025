/**
 * Saved Strategies Helper
 *
 * Manages saving and loading pipeline strategies to/from the backend API.
 */

import type { PipelineSettings, RawFactorFilter } from "./factor-discovery-api";

const API_BASE = "http://localhost:8000/api/v1";

export interface SavedStrategy {
  id: string;
  name: string;
  holding_period: number | null;
  settings: PipelineSettings;
  expected_alpha: number | null;
  expected_alpha_ci_lower: number | null;
  expected_alpha_ci_upper: number | null;
  win_rate: number | null;
  sample_size: number | null;
  created_at: string;
  updated_at: string;
  source: "factor_discovery" | "manual";
}

export interface SaveStrategyInput {
  name: string;
  holding_period?: number;
  settings: PipelineSettings;
  expected_alpha?: number;
  expected_alpha_ci_lower?: number;
  expected_alpha_ci_upper?: number;
  win_rate?: number;
  sample_size?: number;
  source?: "factor_discovery" | "manual";
}

/**
 * Load all saved strategies from the API.
 */
export async function loadStrategies(): Promise<SavedStrategy[]> {
  try {
    const response = await fetch(`${API_BASE}/strategies`);
    if (!response.ok) {
      throw new Error(`Failed to load strategies: ${response.statusText}`);
    }
    return response.json();
  } catch (error) {
    console.error("Failed to load saved strategies:", error);
    return [];
  }
}

/**
 * Save a new strategy to the API.
 */
export async function saveStrategy(
  strategy: SaveStrategyInput
): Promise<SavedStrategy> {
  const response = await fetch(`${API_BASE}/strategies`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      name: strategy.name,
      holding_period: strategy.holding_period,
      settings: strategy.settings,
      expected_alpha: strategy.expected_alpha,
      expected_alpha_ci_lower: strategy.expected_alpha_ci_lower,
      expected_alpha_ci_upper: strategy.expected_alpha_ci_upper,
      win_rate: strategy.win_rate,
      sample_size: strategy.sample_size,
      source: strategy.source || "manual",
    }),
  });

  if (!response.ok) {
    const error = await response.text();
    throw new Error(`Failed to save strategy: ${error}`);
  }

  return response.json();
}

/**
 * Delete a strategy by ID.
 */
export async function deleteStrategy(id: string): Promise<boolean> {
  try {
    const response = await fetch(`${API_BASE}/strategies/${id}`, {
      method: "DELETE",
    });

    if (!response.ok) {
      throw new Error(`Failed to delete strategy: ${response.statusText}`);
    }

    return true;
  } catch (error) {
    console.error("Failed to delete strategy:", error);
    return false;
  }
}

/**
 * Update an existing strategy.
 */
export async function updateStrategy(
  id: string,
  updates: Partial<SaveStrategyInput>
): Promise<SavedStrategy | null> {
  try {
    const response = await fetch(`${API_BASE}/strategies/${id}`, {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(updates),
    });

    if (!response.ok) {
      throw new Error(`Failed to update strategy: ${response.statusText}`);
    }

    return response.json();
  } catch (error) {
    console.error("Failed to update strategy:", error);
    return null;
  }
}

/**
 * Get a strategy by ID.
 */
export async function getStrategy(id: string): Promise<SavedStrategy | null> {
  try {
    const response = await fetch(`${API_BASE}/strategies/${id}`);

    if (!response.ok) {
      if (response.status === 404) return null;
      throw new Error(`Failed to get strategy: ${response.statusText}`);
    }

    return response.json();
  } catch (error) {
    console.error("Failed to get strategy:", error);
    return null;
  }
}

/**
 * Format a date for display.
 */
export function formatStrategyDate(dateString: string): string {
  const date = new Date(dateString);
  return date.toLocaleDateString("en-US", {
    month: "short",
    day: "numeric",
    year: "numeric",
  });
}

/**
 * Generate a default name for a strategy.
 */
export function generateStrategyName(holdingPeriod: number): string {
  const date = new Date().toLocaleDateString("en-US", {
    month: "numeric",
    day: "numeric",
    year: "numeric",
  });
  return `Factor Discovery ${holdingPeriod}Q - ${date}`;
}

/**
 * Create empty/default PipelineSettings.
 */
export function createDefaultSettings(): PipelineSettings {
  return {
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
    min_lenses: 0,  // Default to 0 so raw-filter-only strategies work
    strict_mode: false,
    raw_filters: [],
  };
}

// Re-export types
export type { RawFactorFilter };
