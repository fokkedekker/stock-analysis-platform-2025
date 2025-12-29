/**
 * Saved Strategies Helper
 *
 * Manages saving and loading pipeline strategies to/from localStorage.
 */

import type { PipelineSettings } from "./factor-discovery-api";

const STORAGE_KEY = "saved_strategies";

export interface SavedStrategy {
  id: string;
  name: string;
  holding_period: number;
  settings: PipelineSettings;
  expected_alpha: number;
  expected_alpha_ci_lower: number;
  expected_alpha_ci_upper: number;
  win_rate: number;
  sample_size: number;
  created_at: string;
  source: "factor_discovery" | "manual";
}

/**
 * Generate a unique ID for a strategy.
 */
function generateId(): string {
  return `strategy_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
}

/**
 * Load all saved strategies from localStorage.
 */
export function loadStrategies(): SavedStrategy[] {
  if (typeof window === "undefined") return [];

  try {
    const saved = localStorage.getItem(STORAGE_KEY);
    if (!saved) return [];
    return JSON.parse(saved) as SavedStrategy[];
  } catch (error) {
    console.error("Failed to load saved strategies:", error);
    return [];
  }
}

/**
 * Save a new strategy to localStorage.
 */
export function saveStrategy(
  strategy: Omit<SavedStrategy, "id" | "created_at">
): SavedStrategy {
  const strategies = loadStrategies();

  const newStrategy: SavedStrategy = {
    ...strategy,
    id: generateId(),
    created_at: new Date().toISOString(),
  };

  strategies.push(newStrategy);
  localStorage.setItem(STORAGE_KEY, JSON.stringify(strategies));

  return newStrategy;
}

/**
 * Delete a strategy by ID.
 */
export function deleteStrategy(id: string): boolean {
  const strategies = loadStrategies();
  const index = strategies.findIndex((s) => s.id === id);

  if (index === -1) return false;

  strategies.splice(index, 1);
  localStorage.setItem(STORAGE_KEY, JSON.stringify(strategies));

  return true;
}

/**
 * Update an existing strategy.
 */
export function updateStrategy(
  id: string,
  updates: Partial<Omit<SavedStrategy, "id" | "created_at">>
): SavedStrategy | null {
  const strategies = loadStrategies();
  const index = strategies.findIndex((s) => s.id === id);

  if (index === -1) return null;

  strategies[index] = { ...strategies[index], ...updates };
  localStorage.setItem(STORAGE_KEY, JSON.stringify(strategies));

  return strategies[index];
}

/**
 * Get a strategy by ID.
 */
export function getStrategy(id: string): SavedStrategy | null {
  const strategies = loadStrategies();
  return strategies.find((s) => s.id === id) || null;
}

/**
 * Clear all saved strategies.
 */
export function clearAllStrategies(): void {
  localStorage.removeItem(STORAGE_KEY);
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
