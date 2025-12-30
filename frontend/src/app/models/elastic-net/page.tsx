"use client";

import { RiLineChartLine } from "@remixicon/react";

export default function ElasticNetPage() {
  return (
    <div className="p-6 max-w-4xl mx-auto">
      <div className="flex items-center gap-3 mb-6">
        <div className="p-3 bg-indigo-100 dark:bg-indigo-900 rounded-lg">
          <RiLineChartLine className="w-6 h-6 text-indigo-600 dark:text-indigo-400" />
        </div>
        <div>
          <h1 className="text-2xl font-bold text-gray-900 dark:text-gray-100">
            Elastic Net Regression
          </h1>
          <p className="text-gray-600 dark:text-gray-400">
            Cross-sectional alpha prediction with regularized linear regression
          </p>
        </div>
      </div>

      <div className="bg-white dark:bg-gray-900 rounded-xl border border-gray-200 dark:border-gray-800 p-8 text-center">
        <div className="text-6xl mb-4">Coming Soon</div>
        <p className="text-gray-600 dark:text-gray-400 max-w-md mx-auto">
          Elastic Net regression combines L1 (Lasso) and L2 (Ridge) regularization to predict
          stock alpha directly. This model will provide coefficient stability diagnostics and
          serve as a baseline for more complex ML models.
        </p>

        <div className="mt-8 p-4 bg-gray-50 dark:bg-gray-800 rounded-lg text-left max-w-md mx-auto">
          <h3 className="font-semibold text-gray-900 dark:text-gray-100 mb-2">What it will do:</h3>
          <ul className="text-sm text-gray-600 dark:text-gray-400 space-y-1">
            <li>- Predict expected alpha for each stock</li>
            <li>- Show coefficient magnitudes and stability</li>
            <li>- Provide Information Coefficient (IC) over time</li>
            <li>- Output stock rankings for Pipeline integration</li>
          </ul>
        </div>
      </div>
    </div>
  );
}
