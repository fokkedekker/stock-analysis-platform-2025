"use client";

import React, { useMemo, useState, useCallback } from "react";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  ReferenceDot,
  ReferenceArea,
} from "recharts";
import type { HistoricalPrice, StrategySignal } from "@/lib/api";

interface StockPriceChartProps {
  prices: HistoricalPrice[];
  signals: StrategySignal[];
  loading?: boolean;
}

// Convert quarter date to display format
function formatQuarterLabel(quarter: string): string {
  // e.g., "2023Q1" -> "Q1'23"
  const year = quarter.slice(2, 4);
  const q = quarter.slice(4);
  return `${q}'${year}`;
}

// Format date for tooltip
function formatDate(dateStr: string): string {
  const date = new Date(dateStr);
  return date.toLocaleDateString("en-US", {
    month: "short",
    day: "numeric",
    year: "numeric",
  });
}

// Format currency
function formatPrice(value: number): string {
  return `$${value.toFixed(2)}`;
}

export default function StockPriceChart({
  prices,
  signals,
  loading = false,
}: StockPriceChartProps) {
  // Transform price data for the chart
  const chartData = useMemo(() => {
    return prices.map((p) => ({
      date: p.date,
      price: p.adjClose || p.close,
    }));
  }, [prices]);

  // Get date range from price data for filtering signals
  const dateRange = useMemo(() => {
    if (prices.length === 0) return { min: null, max: null };
    const dates = prices.map((p) => new Date(p.date).getTime());
    return { min: Math.min(...dates), max: Math.max(...dates) };
  }, [prices]);

  // Helper to check if a date is within the chart range
  const isDateInRange = (dateStr: string) => {
    if (!dateRange.min || !dateRange.max) return false;
    const d = new Date(dateStr).getTime();
    return d >= dateRange.min && d <= dateRange.max;
  };

  // Helper to find closest price data point to a target date
  const findClosestPriceData = (
    pricesData: HistoricalPrice[],
    targetDate: string
  ): { date: string; price: number } | null => {
    const targetTime = new Date(targetDate).getTime();
    let closest: HistoricalPrice | null = null;
    let closestDiff = Infinity;

    for (const p of pricesData) {
      const pTime = new Date(p.date).getTime();
      const diff = Math.abs(pTime - targetTime);
      if (diff < closestDiff) {
        closestDiff = diff;
        closest = p;
      }
    }

    if (!closest) return null;
    return {
      date: closest.date,
      price: closest.adjClose || closest.close,
    };
  };

  // Signals where BOTH buy and sell are in range (for complete trades with shading)
  const completeVisibleSignals = useMemo(() => {
    return signals.filter(
      (s) => s.matched && isDateInRange(s.buy_date) && isDateInRange(s.sell_date)
    );
  }, [signals, dateRange]);

  // Create buy markers for ALL matched signals where buy_date is in range
  // This shows all buy signals visible on the chart
  const buyMarkers = useMemo(() => {
    return signals
      .filter((s) => s.matched && isDateInRange(s.buy_date))
      .map((s) => {
        const priceData = findClosestPriceData(prices, s.buy_date);
        if (!priceData) return null;
        return {
          date: priceData.date,
          price: priceData.price,
          quarter: s.buy_quarter,
          type: "buy" as const,
        };
      })
      .filter((m): m is NonNullable<typeof m> => m !== null);
  }, [signals, prices, dateRange]);

  // Create sell markers for ALL matched signals where sell_date is in range
  const sellMarkers = useMemo(() => {
    return signals
      .filter((s) => s.matched && s.sell_price !== null && isDateInRange(s.sell_date))
      .map((s) => {
        const priceData = findClosestPriceData(prices, s.sell_date);
        if (!priceData) return null;
        return {
          date: priceData.date,
          price: priceData.price,
          quarter: s.sell_quarter,
          type: "sell" as const,
        };
      })
      .filter((m): m is NonNullable<typeof m> => m !== null);
  }, [signals, prices, dateRange]);

  // Drag selection state for return calculation
  const [dragStart, setDragStart] = useState<{ date: string; price: number } | null>(null);
  const [dragEnd, setDragEnd] = useState<{ date: string; price: number } | null>(null);
  const [isDragging, setIsDragging] = useState(false);

  // Calculate return between drag points
  const dragReturn = useMemo(() => {
    if (!dragStart || !dragEnd) return null;
    const returnPct = ((dragEnd.price - dragStart.price) / dragStart.price) * 100;
    return returnPct;
  }, [dragStart, dragEnd]);

  // Handle mouse events for drag selection
  const handleMouseDown = useCallback((e: any) => {
    if (e && e.activePayload && e.activePayload.length > 0) {
      const payload = e.activePayload[0].payload;
      setDragStart({ date: payload.date, price: payload.price });
      setDragEnd({ date: payload.date, price: payload.price });
      setIsDragging(true);
    }
  }, []);

  const handleMouseMove = useCallback((e: any) => {
    if (isDragging && e && e.activePayload && e.activePayload.length > 0) {
      const payload = e.activePayload[0].payload;
      setDragEnd({ date: payload.date, price: payload.price });
    }
  }, [isDragging]);

  const handleMouseUp = useCallback(() => {
    setIsDragging(false);
  }, []);

  // Custom tooltip
  const CustomTooltip = ({ active, payload }: any) => {
    if (active && payload && payload.length) {
      const data = payload[0].payload;
      // Check if this date is a buy or sell signal
      const buySignal = buyMarkers.find((m) => m.date === data.date);
      const sellSignal = sellMarkers.find((m) => m.date === data.date);

      return (
        <div className="bg-white dark:bg-gray-900 border border-gray-200 dark:border-gray-700 rounded-lg shadow-lg p-3">
          <p className="text-sm font-medium text-gray-900 dark:text-gray-100">
            {formatDate(data.date)}
          </p>
          <p className="text-sm text-gray-600 dark:text-gray-400">
            Price: {formatPrice(data.price)}
          </p>
          {buySignal && (
            <p className="text-sm font-medium text-green-600 dark:text-green-400">
              BUY SIGNAL ({formatQuarterLabel(buySignal.quarter)})
            </p>
          )}
          {sellSignal && (
            <p className="text-sm font-medium text-red-600 dark:text-red-400">
              SELL SIGNAL ({formatQuarterLabel(sellSignal.quarter)})
            </p>
          )}
        </div>
      );
    }
    return null;
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center h-80 bg-gray-50 dark:bg-gray-900 rounded-lg">
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600"></div>
      </div>
    );
  }

  if (prices.length === 0) {
    return (
      <div className="flex items-center justify-center h-80 bg-gray-50 dark:bg-gray-900 rounded-lg">
        <p className="text-gray-500 dark:text-gray-400">
          No price data available
        </p>
      </div>
    );
  }

  return (
    <div className="w-full relative">
      {/* Drag selection return display */}
      {dragStart && dragEnd && dragReturn !== null && (
        <div className="absolute top-2 left-1/2 -translate-x-1/2 z-10 bg-white dark:bg-gray-800 border border-purple-300 dark:border-purple-600 rounded-lg shadow-lg px-4 py-2 flex items-center gap-3">
          <div className="text-sm text-gray-600 dark:text-gray-400">
            {formatDate(dragStart.date)} → {formatDate(dragEnd.date)}
          </div>
          <div className={`text-lg font-semibold ${dragReturn >= 0 ? 'text-green-600 dark:text-green-400' : 'text-red-600 dark:text-red-400'}`}>
            {dragReturn >= 0 ? '+' : ''}{dragReturn.toFixed(2)}%
          </div>
          <div className="text-xs text-gray-500 dark:text-gray-400">
            {formatPrice(dragStart.price)} → {formatPrice(dragEnd.price)}
          </div>
          <button
            onClick={() => { setDragStart(null); setDragEnd(null); }}
            className="ml-2 text-gray-400 hover:text-gray-600 dark:hover:text-gray-200"
            title="Clear selection"
          >
            ✕
          </button>
        </div>
      )}
      <ResponsiveContainer width="100%" height={320}>
        <LineChart
          data={chartData}
          margin={{ top: 20, right: 30, left: 20, bottom: 20 }}
          onMouseDown={handleMouseDown}
          onMouseMove={handleMouseMove}
          onMouseUp={handleMouseUp}
          onMouseLeave={handleMouseUp}
        >
          <CartesianGrid
            strokeDasharray="3 3"
            stroke="#e5e7eb"
            className="dark:stroke-gray-700"
          />
          <XAxis
            dataKey="date"
            tickFormatter={(date) => {
              const d = new Date(date);
              return d.toLocaleDateString("en-US", {
                month: "short",
                year: "2-digit",
              });
            }}
            tick={{ fill: "#6b7280", fontSize: 12 }}
            tickLine={{ stroke: "#9ca3af" }}
            axisLine={{ stroke: "#9ca3af" }}
            interval="preserveStartEnd"
            minTickGap={50}
          />
          <YAxis
            tickFormatter={(value) => `$${value}`}
            tick={{ fill: "#6b7280", fontSize: 12 }}
            tickLine={{ stroke: "#9ca3af" }}
            axisLine={{ stroke: "#9ca3af" }}
            domain={["auto", "auto"]}
            width={60}
          />
          <Tooltip content={<CustomTooltip />} />

          {/* Shaded areas for holding periods (only for complete trades) */}
          {completeVisibleSignals.map((s, i) => {
            const buyData = findClosestPriceData(prices, s.buy_date);
            const sellData = findClosestPriceData(prices, s.sell_date);
            if (!buyData || !sellData) return null;
            return (
              <ReferenceArea
                key={`area-${i}`}
                x1={buyData.date}
                x2={sellData.date}
                fill={s.alpha && s.alpha > 0 ? "#22c55e" : "#ef4444"}
                fillOpacity={0.1}
              />
            );
          })}

          {/* Drag selection area for return calculation */}
          {dragStart && dragEnd && (
            <ReferenceArea
              x1={dragStart.date}
              x2={dragEnd.date}
              fill="#8b5cf6"
              fillOpacity={0.2}
              stroke="#8b5cf6"
              strokeWidth={1}
              strokeDasharray="4 4"
            />
          )}

          {/* Price line */}
          <Line
            type="monotone"
            dataKey="price"
            stroke="#3b82f6"
            strokeWidth={2}
            dot={false}
            activeDot={{ r: 4, fill: "#3b82f6" }}
          />

          {/* Buy markers */}
          {buyMarkers.map((marker, i) => (
            <ReferenceDot
              key={`buy-${i}`}
              x={marker.date}
              y={marker.price!}
              r={8}
              fill="#22c55e"
              stroke="#fff"
              strokeWidth={2}
            />
          ))}

          {/* Sell markers */}
          {sellMarkers.map((marker, i) => (
            <ReferenceDot
              key={`sell-${i}`}
              x={marker.date}
              y={marker.price!}
              r={8}
              fill="#ef4444"
              stroke="#fff"
              strokeWidth={2}
            />
          ))}
        </LineChart>
      </ResponsiveContainer>

      {/* Legend */}
      {(buyMarkers.length > 0 || sellMarkers.length > 0) && (
        <div className="flex items-center justify-center gap-6 mt-2">
          <div className="flex items-center gap-2">
            <div className="w-4 h-4 rounded-full bg-green-500"></div>
            <span className="text-sm text-gray-600 dark:text-gray-400">
              Buy Signal
            </span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-4 h-4 rounded-full bg-red-500"></div>
            <span className="text-sm text-gray-600 dark:text-gray-400">
              Sell Signal
            </span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-4 h-2 bg-green-500/20 border border-green-500"></div>
            <span className="text-sm text-gray-600 dark:text-gray-400">
              Positive Alpha
            </span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-4 h-2 bg-red-500/20 border border-red-500"></div>
            <span className="text-sm text-gray-600 dark:text-gray-400">
              Negative Alpha
            </span>
          </div>
        </div>
      )}
    </div>
  );
}
