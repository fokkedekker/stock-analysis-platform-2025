"use client"

import { useState } from "react"
import { RiAddLine, RiCloseLine, RiCheckLine } from "@remixicon/react"

// Variable definitions with their possible test values
export interface TestVariable {
  name: string
  label: string
  values: { value: string | number | boolean; label: string }[]
  group: "survival" | "quality" | "quality_tags_positive" | "quality_tags_negative" | "valuation"
}

export const TEST_VARIABLES: TestVariable[] = [
  // Survival
  {
    name: "altman_zone",
    label: "Altman Zone",
    values: [
      { value: "safe", label: "Safe" },
      { value: "grey", label: "Grey" },
    ],
    group: "survival",
  },
  {
    name: "piotroski_min",
    label: "Piotroski Min",
    values: [
      { value: 3, label: "3" },
      { value: 4, label: "4" },
      { value: 5, label: "5" },
      { value: 6, label: "6" },
      { value: 7, label: "7" },
      { value: 8, label: "8" },
      { value: 9, label: "9" },
    ],
    group: "survival",
  },
  // Quality
  {
    name: "quality_enabled",
    label: "Quality Filter",
    values: [
      { value: true, label: "On" },
      { value: false, label: "Off" },
    ],
    group: "quality",
  },
  {
    name: "min_quality",
    label: "Min Quality",
    values: [
      { value: "weak", label: "Weak" },
      { value: "average", label: "Average" },
      { value: "compounder", label: "Compounder" },
    ],
    group: "quality",
  },
  // Quality Tags - Positive (require when on)
  {
    name: "tag_durable_compounder",
    label: "Require Durable Compounder",
    values: [
      { value: true, label: "Require" },
      { value: false, label: "Don't require" },
    ],
    group: "quality_tags_positive",
  },
  {
    name: "tag_cash_machine",
    label: "Require Cash Machine",
    values: [
      { value: true, label: "Require" },
      { value: false, label: "Don't require" },
    ],
    group: "quality_tags_positive",
  },
  {
    name: "tag_deep_value",
    label: "Require Deep Value",
    values: [
      { value: true, label: "Require" },
      { value: false, label: "Don't require" },
    ],
    group: "quality_tags_positive",
  },
  {
    name: "tag_heavy_reinvestor",
    label: "Require Heavy Reinvestor",
    values: [
      { value: true, label: "Require" },
      { value: false, label: "Don't require" },
    ],
    group: "quality_tags_positive",
  },
  // Quality Tags - Negative (exclude when on)
  {
    name: "tag_premium_priced",
    label: "Exclude Premium Priced",
    values: [
      { value: true, label: "Exclude" },
      { value: false, label: "Don't exclude" },
    ],
    group: "quality_tags_negative",
  },
  {
    name: "tag_volatile_returns",
    label: "Exclude Volatile Returns",
    values: [
      { value: true, label: "Exclude" },
      { value: false, label: "Don't exclude" },
    ],
    group: "quality_tags_negative",
  },
  {
    name: "tag_weak_moat_signal",
    label: "Exclude Weak Moat",
    values: [
      { value: true, label: "Exclude" },
      { value: false, label: "Don't exclude" },
    ],
    group: "quality_tags_negative",
  },
  {
    name: "tag_earnings_quality_concern",
    label: "Exclude Earnings Concern",
    values: [
      { value: true, label: "Exclude" },
      { value: false, label: "Don't exclude" },
    ],
    group: "quality_tags_negative",
  },
  // Valuation
  {
    name: "graham_mode",
    label: "Graham Mode",
    values: [
      { value: "strict", label: "Strict" },
      { value: "modern", label: "Modern" },
      { value: "garp", label: "GARP" },
      { value: "relaxed", label: "Relaxed" },
    ],
    group: "valuation",
  },
  {
    name: "graham_min",
    label: "Graham Min",
    values: [
      { value: 3, label: "3/8" },
      { value: 4, label: "4/8" },
      { value: 5, label: "5/8" },
      { value: 6, label: "6/8" },
      { value: 7, label: "7/8" },
      { value: 8, label: "8/8" },
    ],
    group: "valuation",
  },
  {
    name: "mf_top_pct",
    label: "Magic Formula Top %",
    values: [
      { value: 10, label: "10%" },
      { value: 20, label: "20%" },
      { value: 30, label: "30%" },
      { value: 50, label: "50%" },
    ],
    group: "valuation",
  },
  {
    name: "max_peg",
    label: "Max PEG",
    values: [
      { value: 0.5, label: "0.5" },
      { value: 1.0, label: "1.0" },
      { value: 1.5, label: "1.5" },
      { value: 2.0, label: "2.0" },
      { value: 2.5, label: "2.5" },
      { value: 3.0, label: "3.0" },
    ],
    group: "valuation",
  },
  {
    name: "net_net_enabled",
    label: "Net-Net",
    values: [
      { value: true, label: "On" },
      { value: false, label: "Off" },
    ],
    group: "valuation",
  },
  {
    name: "fama_french_enabled",
    label: "Fama-French",
    values: [
      { value: true, label: "On" },
      { value: false, label: "Off" },
    ],
    group: "valuation",
  },
  {
    name: "ff_top_pct",
    label: "FF Top %",
    values: [
      { value: 20, label: "20%" },
      { value: 30, label: "30%" },
      { value: 40, label: "40%" },
      { value: 50, label: "50%" },
    ],
    group: "valuation",
  },
  {
    name: "min_lenses",
    label: "Min Lenses",
    values: [
      { value: 0, label: "0" },
      { value: 1, label: "1" },
      { value: 2, label: "2" },
      { value: 3, label: "3" },
      { value: 4, label: "4" },
      { value: 5, label: "5" },
    ],
    group: "valuation",
  },
  {
    name: "strict_mode",
    label: "Strict Mode",
    values: [
      { value: true, label: "On" },
      { value: false, label: "Off" },
    ],
    group: "valuation",
  },
]

const GROUP_LABELS: Record<string, string> = {
  survival: "Survival Gates",
  quality: "Quality Filters",
  quality_tags_positive: "Quality Tags (Require)",
  quality_tags_negative: "Quality Tags (Exclude)",
  valuation: "Valuation Lenses",
}

export interface TestVariablesSectionProps {
  // Map of variable name -> selected test values
  selectedVariables: Map<string, (string | number | boolean)[]>
  setSelectedVariables: (v: Map<string, (string | number | boolean)[]>) => void
}

export function TestVariablesSection({
  selectedVariables,
  setSelectedVariables,
}: TestVariablesSectionProps) {
  const [expandedVariable, setExpandedVariable] = useState<string | null>(null)

  const isVariableActive = (name: string) => selectedVariables.has(name)

  const getSelectedValues = (name: string) => selectedVariables.get(name) || []

  const toggleVariable = (name: string) => {
    const newMap = new Map(selectedVariables)
    if (newMap.has(name)) {
      newMap.delete(name)
      setSelectedVariables(newMap)
      setExpandedVariable(null)
    } else {
      // Add with all values selected by default
      const variable = TEST_VARIABLES.find((v) => v.name === name)
      if (variable) {
        newMap.set(
          name,
          variable.values.map((v) => v.value)
        )
        setSelectedVariables(newMap)
        setExpandedVariable(name)
      }
    }
  }

  const toggleValue = (varName: string, value: string | number | boolean) => {
    const newMap = new Map(selectedVariables)
    const currentValues = newMap.get(varName) || []
    const hasValue = currentValues.includes(value)

    if (hasValue) {
      // Remove value (but keep at least one)
      if (currentValues.length > 1) {
        newMap.set(
          varName,
          currentValues.filter((v) => v !== value)
        )
      }
    } else {
      // Add value
      newMap.set(varName, [...currentValues, value])
    }
    setSelectedVariables(newMap)
  }

  const getVariablesByGroup = (group: string) =>
    TEST_VARIABLES.filter((v) => v.group === group)

  const groups = ["survival", "quality", "quality_tags_positive", "quality_tags_negative", "valuation"]

  return (
    <div className="p-4 rounded-lg border border-gray-200 dark:border-gray-800 bg-white dark:bg-gray-950">
      <div className="flex items-center gap-2 mb-3">
        <span className="w-5 h-5 flex items-center justify-center rounded-full bg-purple-100 dark:bg-purple-900 text-purple-700 dark:text-purple-300 text-xs font-bold">
          +
        </span>
        <h2 className="text-sm font-semibold text-gray-900 dark:text-gray-100">
          Variables to Test
        </h2>
        <span className="text-xs text-gray-500">
          Click to add, then select which values to test
        </span>
      </div>

      <div className="space-y-4">
        {groups.map((group) => (
          <div key={group}>
            <h3 className="text-xs font-medium text-gray-500 uppercase mb-2">
              {GROUP_LABELS[group]}
            </h3>
            <div className="flex flex-wrap gap-2">
              {getVariablesByGroup(group).map((variable) => {
                const isActive = isVariableActive(variable.name)
                const selectedValues = getSelectedValues(variable.name)
                const isExpanded = expandedVariable === variable.name

                return (
                  <div key={variable.name} className="relative">
                    <button
                      onClick={() => {
                        if (isActive) {
                          setExpandedVariable(isExpanded ? null : variable.name)
                        } else {
                          toggleVariable(variable.name)
                        }
                      }}
                      className={`px-3 py-1.5 text-sm rounded-lg border transition-colors flex items-center gap-1 ${
                        isActive
                          ? "bg-purple-100 dark:bg-purple-900 border-purple-300 dark:border-purple-700 text-purple-700 dark:text-purple-300"
                          : "bg-gray-50 dark:bg-gray-800 border-gray-200 dark:border-gray-700 text-gray-600 dark:text-gray-400 hover:bg-gray-100 dark:hover:bg-gray-700"
                      }`}
                    >
                      {isActive ? (
                        <RiCheckLine className="w-4 h-4" />
                      ) : (
                        <RiAddLine className="w-4 h-4" />
                      )}
                      {variable.label}
                      {isActive && (
                        <span className="ml-1 text-xs opacity-70">
                          ({selectedValues.length})
                        </span>
                      )}
                    </button>

                    {/* Value selector popover */}
                    {isActive && isExpanded && (
                      <div className="absolute top-full left-0 mt-1 z-10 bg-white dark:bg-gray-900 rounded-lg border border-gray-200 dark:border-gray-700 shadow-lg p-2 min-w-[150px]">
                        <div className="flex items-center justify-between mb-2">
                          <span className="text-xs font-medium text-gray-700 dark:text-gray-300">
                            Test values:
                          </span>
                          <button
                            onClick={() => toggleVariable(variable.name)}
                            className="text-red-500 hover:text-red-600"
                            title="Remove variable"
                          >
                            <RiCloseLine className="w-4 h-4" />
                          </button>
                        </div>
                        <div className="flex flex-wrap gap-1">
                          {variable.values.map((v) => {
                            const isSelected = selectedValues.includes(v.value)
                            return (
                              <button
                                key={String(v.value)}
                                onClick={() => toggleValue(variable.name, v.value)}
                                className={`px-2 py-1 text-xs rounded border transition-colors ${
                                  isSelected
                                    ? "bg-purple-100 dark:bg-purple-900 border-purple-300 dark:border-purple-700 text-purple-700 dark:text-purple-300"
                                    : "bg-gray-50 dark:bg-gray-800 border-gray-200 dark:border-gray-600 text-gray-500 hover:bg-gray-100"
                                }`}
                              >
                                {v.label}
                              </button>
                            )
                          })}
                        </div>
                        <button
                          onClick={() => setExpandedVariable(null)}
                          className="mt-2 text-xs text-gray-500 hover:text-gray-700 w-full text-center"
                        >
                          Done
                        </button>
                      </div>
                    )}
                  </div>
                )
              })}
            </div>
          </div>
        ))}
      </div>

      {/* Active variables summary */}
      {selectedVariables.size > 0 && (
        <div className="mt-4 pt-4 border-t border-gray-200 dark:border-gray-700">
          <div className="flex items-center justify-between mb-2">
            <span className="text-sm font-medium text-gray-700 dark:text-gray-300">
              Active test variables ({selectedVariables.size}):
            </span>
            <button
              onClick={() => setSelectedVariables(new Map())}
              className="text-xs text-red-500 hover:text-red-600"
            >
              Clear all
            </button>
          </div>
          <div className="flex flex-wrap gap-2">
            {Array.from(selectedVariables.entries()).map(([name, values]) => {
              const variable = TEST_VARIABLES.find((v) => v.name === name)
              if (!variable) return null
              return (
                <div
                  key={name}
                  className="bg-purple-50 dark:bg-purple-950 border border-purple-200 dark:border-purple-800 rounded-lg px-2 py-1 text-sm"
                >
                  <span className="font-medium text-purple-700 dark:text-purple-300">
                    {variable.label}:
                  </span>
                  <span className="ml-1 text-purple-600 dark:text-purple-400">
                    {values
                      .map((v) => {
                        const found = variable.values.find((vv) => vv.value === v)
                        return found?.label || String(v)
                      })
                      .join(", ")}
                  </span>
                  <button
                    onClick={() => toggleVariable(name)}
                    className="ml-2 text-purple-400 hover:text-purple-600"
                  >
                    <RiCloseLine className="w-4 h-4 inline" />
                  </button>
                </div>
              )
            })}
          </div>
        </div>
      )}
    </div>
  )
}
