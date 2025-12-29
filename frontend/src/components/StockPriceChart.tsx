"use client";

import React, { useMemo } from "react";
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

  // Create markers for buy/sell signals
  // Use the actual date from price data (not quarter-end date) for proper chart positioning
  const buyMarkers = useMemo(() => {
    return signals
      .filter((s) => s.matched)
      .map((s) => {
        const priceData = findClosestPriceData(prices, s.buy_date);
        if (!priceData) return null;
        return {
          date: priceData.date, // Use actual trading date from price data
          price: priceData.price,
          quarter: s.buy_quarter,
          type: "buy" as const,
        };
      })
      .filter((m): m is NonNullable<typeof m> => m !== null);
  }, [signals, prices]);

  const sellMarkers = useMemo(() => {
    return signals
      .filter((s) => s.matched && s.sell_price !== null)
      .map((s) => {
        const priceData = findClosestPriceData(prices, s.sell_date);
        if (!priceData) return null;
        return {
          date: priceData.date, // Use actual trading date from price data
          price: priceData.price,
          quarter: s.sell_quarter,
          type: "sell" as const,
        };
      })
      .filter((m): m is NonNullable<typeof m> => m !== null);
  }, [signals, prices]);

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
    <div className="w-full">
      <ResponsiveContainer width="100%" height={320}>
        <LineChart
          data={chartData}
          margin={{ top: 20, right: 30, left: 20, bottom: 20 }}
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

          {/* Shaded areas for holding periods */}
          {signals
            .filter((s) => s.matched)
            .map((s, i) => {
              const buyData = findClosestPriceData(prices, s.buy_date);
              const sellData = findClosestPriceData(prices, s.sell_date);
              if (!buyData) return null;
              // For sell, use the last date in prices if sell_date is in the future
              const sellDate = sellData?.date || prices[prices.length - 1]?.date;
              if (!sellDate) return null;
              return (
                <ReferenceArea
                  key={`area-${i}`}
                  x1={buyData.date}
                  x2={sellDate}
                  fill={s.alpha && s.alpha > 0 ? "#22c55e" : "#ef4444"}
                  fillOpacity={0.1}
                />
              );
            })}

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
      {signals.length > 0 && (
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
