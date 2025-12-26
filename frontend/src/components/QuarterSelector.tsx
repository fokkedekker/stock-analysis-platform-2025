"use client";

import { useQuarter } from "@/contexts/QuarterContext";

function formatQuarter(q: string): string {
  // Convert "2024Q3" to "Q3 2024"
  const match = q.match(/^(\d{4})Q(\d)$/);
  if (match) {
    return `Q${match[2]} ${match[1]}`;
  }
  return q;
}

export function QuarterSelector() {
  const { quarter, availableQuarters, setQuarter, isLoading } = useQuarter();

  if (isLoading) {
    return (
      <div className="px-3 py-2">
        <div className="h-9 bg-gray-100 dark:bg-gray-800 rounded animate-pulse" />
      </div>
    );
  }

  if (availableQuarters.length === 0) {
    return null;
  }

  return (
    <div className="px-3 py-2">
      <label className="block text-xs font-medium text-gray-500 dark:text-gray-400 mb-1">
        Analysis Period
      </label>
      <select
        value={quarter || ""}
        onChange={(e) => setQuarter(e.target.value || null)}
        className="w-full px-3 py-2 text-sm bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 dark:focus:ring-blue-400"
      >
        <option value="">Latest</option>
        {availableQuarters.map((q) => (
          <option key={q} value={q}>
            {formatQuarter(q)}
          </option>
        ))}
      </select>
    </div>
  );
}
