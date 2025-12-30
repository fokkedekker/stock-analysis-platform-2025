"use client";

import { RiBarChartBoxLine } from "@remixicon/react";

export default function GAMPage() {
  return (
    <div className="p-6 max-w-4xl mx-auto">
      <div className="flex items-center gap-3 mb-6">
        <div className="p-3 bg-indigo-100 dark:bg-indigo-900 rounded-lg">
          <RiBarChartBoxLine className="w-6 h-6 text-indigo-600 dark:text-indigo-400" />
        </div>
        <div>
          <h1 className="text-2xl font-bold text-gray-900 dark:text-gray-100">
            Generalized Additive Model (GAM)
          </h1>
          <p className="text-gray-600 dark:text-gray-400">
            Non-linear but interpretable alpha prediction
          </p>
        </div>
      </div>

      <div className="bg-white dark:bg-gray-900 rounded-xl border border-gray-200 dark:border-gray-800 p-8 text-center">
        <div className="text-6xl mb-4">Coming Soon</div>
        <p className="text-gray-600 dark:text-gray-400 max-w-md mx-auto">
          GAM models capture non-linear relationships between factors and alpha while remaining
          interpretable. Unlike black-box models, GAM shows the exact shape of each factor's
          effect, revealing "sweet spots" and saturation points.
        </p>

        <div className="mt-8 p-4 bg-gray-50 dark:bg-gray-800 rounded-lg text-left max-w-md mx-auto">
          <h3 className="font-semibold text-gray-900 dark:text-gray-100 mb-2">What it will do:</h3>
          <ul className="text-sm text-gray-600 dark:text-gray-400 space-y-1">
            <li>- Show partial dependence plots for each factor</li>
            <li>- Identify optimal ranges (e.g., "P/E 8-15 is best")</li>
            <li>- Capture curvature without overfitting</li>
            <li>- Convert insights to Pipeline filter rules</li>
          </ul>
        </div>
      </div>
    </div>
  );
}
