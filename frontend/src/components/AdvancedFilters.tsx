"use client";

import { useState } from "react";
import type { RawFactorFilter } from "@/lib/api";
import {
  ALL_FACTORS,
  FACTOR_CATEGORIES,
  getFactorByName,
  getOperators,
  formatFactorValue,
  type FactorCategory,
} from "@/lib/factor-metadata";

interface AdvancedFiltersProps {
  filters: RawFactorFilter[];
  onChange: (filters: RawFactorFilter[]) => void;
}

export default function AdvancedFilters({
  filters,
  onChange,
}: AdvancedFiltersProps) {
  const [isExpanded, setIsExpanded] = useState(filters.length > 0);

  // State for adding a new filter
  const [selectedCategory, setSelectedCategory] = useState<FactorCategory | "">(
    ""
  );
  const [selectedFactor, setSelectedFactor] = useState<string>("");
  const [selectedOperator, setSelectedOperator] = useState<string>(">=");
  const [filterValue, setFilterValue] = useState<string>("");

  // Get factors for selected category
  const categoryFactors = selectedCategory
    ? ALL_FACTORS.filter((f) => f.category === selectedCategory)
    : [];

  // Get factor definition for selected factor
  const factorDef = selectedFactor ? getFactorByName(selectedFactor) : null;

  // Handle category change
  const handleCategoryChange = (category: FactorCategory | "") => {
    setSelectedCategory(category);
    setSelectedFactor("");
    setFilterValue("");
  };

  // Handle factor change
  const handleFactorChange = (factorName: string) => {
    setSelectedFactor(factorName);
    const factor = getFactorByName(factorName);
    if (factor) {
      // Set default operator based on factor direction
      setSelectedOperator(factor.direction);
      // Set default value to first threshold
      if (factor.thresholds.length > 0) {
        setFilterValue(String(factor.thresholds[0]));
      }
    }
  };

  // Add a new filter
  const handleAddFilter = () => {
    if (!selectedFactor || !filterValue) return;

    const newFilter: RawFactorFilter = {
      factor: selectedFactor,
      operator: selectedOperator,
      value: parseFloat(filterValue),
    };

    // Check if filter for this factor already exists
    const existingIndex = filters.findIndex(
      (f) => f.factor === selectedFactor
    );

    if (existingIndex >= 0) {
      // Replace existing filter
      const newFilters = [...filters];
      newFilters[existingIndex] = newFilter;
      onChange(newFilters);
    } else {
      // Add new filter
      onChange([...filters, newFilter]);
    }

    // Reset form
    setSelectedCategory("");
    setSelectedFactor("");
    setFilterValue("");
  };

  // Remove a filter
  const handleRemoveFilter = (index: number) => {
    const newFilters = filters.filter((_, i) => i !== index);
    onChange(newFilters);
  };

  // Clear all filters
  const handleClearAll = () => {
    onChange([]);
  };

  // Format filter for display
  const formatFilter = (filter: RawFactorFilter): string => {
    const factor = getFactorByName(filter.factor);
    const label = factor?.label || filter.factor;
    const value =
      typeof filter.value === "number" && factor
        ? formatFactorValue(filter.value, factor)
        : String(filter.value);
    return `${label} ${filter.operator} ${value}`;
  };

  return (
    <div className="border border-slate-200 dark:border-slate-700 rounded-lg overflow-hidden">
      {/* Header */}
      <button
        onClick={() => setIsExpanded(!isExpanded)}
        className="w-full flex items-center justify-between px-4 py-3 bg-slate-50 dark:bg-slate-800 hover:bg-slate-100 dark:hover:bg-slate-700 transition-colors"
      >
        <div className="flex items-center gap-2">
          <span className="font-medium text-slate-900 dark:text-slate-100">
            Advanced Filters
          </span>
          {filters.length > 0 && (
            <span className="px-2 py-0.5 text-xs font-medium bg-blue-100 dark:bg-blue-900 text-blue-700 dark:text-blue-300 rounded-full">
              {filters.length}
            </span>
          )}
        </div>
        <svg
          className={`w-5 h-5 text-slate-500 transition-transform ${
            isExpanded ? "rotate-180" : ""
          }`}
          fill="none"
          viewBox="0 0 24 24"
          stroke="currentColor"
        >
          <path
            strokeLinecap="round"
            strokeLinejoin="round"
            strokeWidth={2}
            d="M19 9l-7 7-7-7"
          />
        </svg>
      </button>

      {/* Content */}
      {isExpanded && (
        <div className="p-4 space-y-4">
          {/* Active Filters */}
          {filters.length > 0 && (
            <div className="space-y-2">
              <div className="flex items-center justify-between">
                <span className="text-sm font-medium text-slate-700 dark:text-slate-300">
                  Active Filters
                </span>
                <button
                  onClick={handleClearAll}
                  className="text-xs text-red-600 dark:text-red-400 hover:underline"
                >
                  Clear All
                </button>
              </div>
              <div className="flex flex-wrap gap-2">
                {filters.map((filter, index) => (
                  <div
                    key={index}
                    className="flex items-center gap-1.5 px-3 py-1.5 bg-blue-50 dark:bg-blue-900/30 text-blue-800 dark:text-blue-200 rounded-full text-sm"
                  >
                    <span>{formatFilter(filter)}</span>
                    <button
                      onClick={() => handleRemoveFilter(index)}
                      className="ml-1 text-blue-600 dark:text-blue-400 hover:text-blue-800 dark:hover:text-blue-200"
                    >
                      <svg
                        className="w-4 h-4"
                        fill="none"
                        viewBox="0 0 24 24"
                        stroke="currentColor"
                      >
                        <path
                          strokeLinecap="round"
                          strokeLinejoin="round"
                          strokeWidth={2}
                          d="M6 18L18 6M6 6l12 12"
                        />
                      </svg>
                    </button>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Add Filter Form */}
          <div className="space-y-3">
            <span className="text-sm font-medium text-slate-700 dark:text-slate-300">
              Add Filter
            </span>
            <div className="flex flex-wrap gap-2">
              {/* Category Select */}
              <select
                value={selectedCategory}
                onChange={(e) =>
                  handleCategoryChange(e.target.value as FactorCategory | "")
                }
                className="px-3 py-2 text-sm border border-slate-300 dark:border-slate-600 rounded-md bg-white dark:bg-slate-800 text-slate-900 dark:text-slate-100"
              >
                <option value="">Select Category...</option>
                {FACTOR_CATEGORIES.map((cat) => (
                  <option key={cat.id} value={cat.id}>
                    {cat.label}
                  </option>
                ))}
              </select>

              {/* Factor Select */}
              <select
                value={selectedFactor}
                onChange={(e) => handleFactorChange(e.target.value)}
                disabled={!selectedCategory}
                className="px-3 py-2 text-sm border border-slate-300 dark:border-slate-600 rounded-md bg-white dark:bg-slate-800 text-slate-900 dark:text-slate-100 disabled:opacity-50"
              >
                <option value="">Select Factor...</option>
                {categoryFactors.map((f) => (
                  <option key={f.name} value={f.name}>
                    {f.label}
                  </option>
                ))}
              </select>

              {/* Operator Select */}
              <select
                value={selectedOperator}
                onChange={(e) => setSelectedOperator(e.target.value)}
                disabled={!selectedFactor}
                className="px-3 py-2 text-sm border border-slate-300 dark:border-slate-600 rounded-md bg-white dark:bg-slate-800 text-slate-900 dark:text-slate-100 disabled:opacity-50 w-20"
              >
                {getOperators().map((op) => (
                  <option key={op.value} value={op.value}>
                    {op.label}
                  </option>
                ))}
              </select>

              {/* Value Input / Select */}
              {factorDef && factorDef.thresholds.length > 0 ? (
                <select
                  value={filterValue}
                  onChange={(e) => setFilterValue(e.target.value)}
                  disabled={!selectedFactor}
                  className="px-3 py-2 text-sm border border-slate-300 dark:border-slate-600 rounded-md bg-white dark:bg-slate-800 text-slate-900 dark:text-slate-100 disabled:opacity-50 w-32"
                >
                  <option value="">Value...</option>
                  {factorDef.thresholds.map((t) => (
                    <option key={t} value={t}>
                      {formatFactorValue(t, factorDef)}
                    </option>
                  ))}
                </select>
              ) : (
                <input
                  type="number"
                  value={filterValue}
                  onChange={(e) => setFilterValue(e.target.value)}
                  disabled={!selectedFactor}
                  placeholder="Value..."
                  step="any"
                  className="px-3 py-2 text-sm border border-slate-300 dark:border-slate-600 rounded-md bg-white dark:bg-slate-800 text-slate-900 dark:text-slate-100 disabled:opacity-50 w-32"
                />
              )}

              {/* Add Button */}
              <button
                onClick={handleAddFilter}
                disabled={!selectedFactor || !filterValue}
                className="px-4 py-2 text-sm font-medium text-white bg-blue-600 hover:bg-blue-700 disabled:bg-slate-400 rounded-md transition-colors"
              >
                Add
              </button>
            </div>

            {/* Factor Description */}
            {factorDef && (
              <p className="text-xs text-slate-500 dark:text-slate-400">
                {factorDef.description}
              </p>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
