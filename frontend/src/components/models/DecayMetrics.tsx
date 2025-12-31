"use client";

import { useState, useRef, useEffect } from "react";
import { createPortal } from "react-dom";
import { DecayMetrics } from "@/lib/factor-discovery-api";

interface DecayMetricsProps {
  metrics: DecayMetrics | null;
  showDetails?: boolean;
}

/**
 * Get the color class for a decay score.
 * Higher decay scores (more stable factors) get better colors.
 */
function getDecayColorClass(score: number): string {
  if (score >= 0.8) return "bg-emerald-100 text-emerald-700 dark:bg-emerald-900/50 dark:text-emerald-300";
  if (score >= 0.6) return "bg-yellow-100 text-yellow-700 dark:bg-yellow-900/50 dark:text-yellow-300";
  if (score >= 0.4) return "bg-orange-100 text-orange-700 dark:bg-orange-900/50 dark:text-orange-300";
  return "bg-red-100 text-red-700 dark:bg-red-900/50 dark:text-red-300";
}

/**
 * Get trend indicator arrow based on alpha_trend value.
 * Positive trend (> 0.1) = improving
 * Negative trend (< -0.1) = declining
 * Otherwise = stable
 */
function getTrendIndicator(trend: number): { icon: string; label: string } {
  if (trend > 0.1) return { icon: "\u2191", label: "Improving" };
  if (trend < -0.1) return { icon: "\u2193", label: "Declining" };
  return { icon: "\u2192", label: "Stable" };
}

/**
 * A badge showing the decay score with color coding.
 * Shows % of rolling 3-year windows with positive alpha.
 */
export function DecayScoreBadge({ metrics }: DecayMetricsProps) {
  const [isVisible, setIsVisible] = useState(false);
  const [position, setPosition] = useState({ top: 0, left: 0 });
  const badgeRef = useRef<HTMLSpanElement>(null);
  const [mounted, setMounted] = useState(false);

  useEffect(() => {
    setMounted(true);
  }, []);

  useEffect(() => {
    if (isVisible && badgeRef.current) {
      const rect = badgeRef.current.getBoundingClientRect();
      setPosition({
        top: rect.top - 8,
        left: rect.left + rect.width / 2,
      });
    }
  }, [isVisible]);

  if (!metrics) {
    return <span className="text-gray-400 dark:text-gray-600">&mdash;</span>;
  }

  const score = metrics.decay_score;
  const colorClass = getDecayColorClass(score);
  const trend = getTrendIndicator(metrics.alpha_trend);

  const tooltip = isVisible && mounted && (
    <div
      className="fixed z-[9999] w-56 p-3 text-xs bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded-lg shadow-lg"
      style={{
        top: position.top,
        left: position.left,
        transform: "translate(-50%, -100%)",
      }}
    >
      <div className="space-y-1">
        <p className="font-medium text-gray-900 dark:text-gray-100">Factor Stability</p>
        <p className="text-gray-500 dark:text-gray-400">
          {(score * 100).toFixed(0)}% of {metrics.n_windows} rolling 3-year windows had positive alpha
        </p>
        <div className="space-y-0.5 pt-1 border-t border-gray-200 dark:border-gray-700">
          <div className="flex justify-between">
            <span className="text-gray-500 dark:text-gray-400">Trend:</span>
            <span className="font-medium text-gray-900 dark:text-gray-100">{trend.label} {trend.icon}</span>
          </div>
          <div className="flex justify-between">
            <span className="text-gray-500 dark:text-gray-400">IC Stability:</span>
            <span className="font-medium text-gray-900 dark:text-gray-100">{(metrics.ic_stability * 100).toFixed(0)}%</span>
          </div>
          {metrics.recent_alpha !== null && (
            <div className="flex justify-between">
              <span className="text-gray-500 dark:text-gray-400">Recent Alpha:</span>
              <span className={`font-medium ${metrics.recent_alpha > 0 ? "text-green-600 dark:text-green-400" : "text-red-600 dark:text-red-400"}`}>
                {metrics.recent_alpha > 0 ? "+" : ""}{metrics.recent_alpha.toFixed(1)}%
              </span>
            </div>
          )}
        </div>
      </div>
      <div className="absolute top-full left-1/2 -translate-x-1/2 -mt-1">
        <div className="border-4 border-transparent border-t-white dark:border-t-gray-800" />
      </div>
    </div>
  );

  return (
    <span className="relative inline-flex items-center">
      <span
        ref={badgeRef}
        className={`inline-flex items-center gap-1 px-2 py-0.5 rounded text-xs font-medium cursor-help ${colorClass}`}
        onMouseEnter={() => setIsVisible(true)}
        onMouseLeave={() => setIsVisible(false)}
        onFocus={() => setIsVisible(true)}
        onBlur={() => setIsVisible(false)}
      >
        {(score * 100).toFixed(0)}% {trend.icon}
      </span>
      {mounted && typeof document !== "undefined" && createPortal(tooltip, document.body)}
    </span>
  );
}

/**
 * Detailed decay metrics display for expanded views.
 */
export function DecayMetricsDetail({ metrics }: DecayMetricsProps) {
  if (!metrics) {
    return (
      <div className="text-sm text-gray-500 dark:text-gray-400">
        Stability metrics not available (insufficient data)
      </div>
    );
  }

  const trend = getTrendIndicator(metrics.alpha_trend);
  const stabilityColor = getDecayColorClass(metrics.decay_score);

  return (
    <div className="space-y-3">
      <div className="flex items-center gap-2">
        <span className={`px-2 py-1 rounded text-sm font-medium ${stabilityColor}`}>
          {(metrics.decay_score * 100).toFixed(0)}% Stable
        </span>
        <span className="text-sm text-gray-600 dark:text-gray-400">
          ({metrics.n_windows} rolling windows)
        </span>
      </div>

      <div className="grid grid-cols-2 gap-4 text-sm">
        <div>
          <span className="text-gray-500 dark:text-gray-400">Alpha Trend:</span>
          <span className="ml-2 font-medium">
            {trend.label} {trend.icon}
          </span>
        </div>

        <div>
          <span className="text-gray-500 dark:text-gray-400">IC Stability:</span>
          <span className="ml-2 font-medium">
            {(metrics.ic_stability * 100).toFixed(0)}%
          </span>
        </div>

        {metrics.recent_alpha !== null && (
          <div>
            <span className="text-gray-500 dark:text-gray-400">Recent Alpha:</span>
            <span className={`ml-2 font-medium ${metrics.recent_alpha > 0 ? "text-green-600 dark:text-green-400" : "text-red-600 dark:text-red-400"}`}>
              {metrics.recent_alpha > 0 ? "+" : ""}
              {metrics.recent_alpha.toFixed(2)}%
            </span>
          </div>
        )}

        {metrics.mean_ic !== null && (
          <div>
            <span className="text-gray-500 dark:text-gray-400">Mean IC:</span>
            <span className="ml-2 font-medium">
              {metrics.mean_ic.toFixed(3)}
            </span>
          </div>
        )}
      </div>
    </div>
  );
}

/**
 * Simple inline stability indicator without tooltip.
 */
export function DecayIndicator({ metrics }: DecayMetricsProps) {
  if (!metrics) {
    return null;
  }

  const score = metrics.decay_score;
  const trend = getTrendIndicator(metrics.alpha_trend);

  // Color based on score
  let colorClass: string;
  if (score >= 0.8) colorClass = "text-emerald-600 dark:text-emerald-400";
  else if (score >= 0.6) colorClass = "text-yellow-600 dark:text-yellow-400";
  else if (score >= 0.4) colorClass = "text-orange-600 dark:text-orange-400";
  else colorClass = "text-red-600 dark:text-red-400";

  return (
    <span className={`text-xs ${colorClass}`}>
      {(score * 100).toFixed(0)}% {trend.icon}
    </span>
  );
}
