"use client"

import { useState } from "react"
import { RiInformationLine } from "@remixicon/react"

interface InfoTooltipProps {
  content: string
  className?: string
}

export function InfoTooltip({ content, className = "" }: InfoTooltipProps) {
  const [isVisible, setIsVisible] = useState(false)

  return (
    <span className={`relative inline-flex items-center ${className}`}>
      <button
        type="button"
        className="text-gray-400 hover:text-gray-600 dark:hover:text-gray-300 transition-colors"
        onMouseEnter={() => setIsVisible(true)}
        onMouseLeave={() => setIsVisible(false)}
        onFocus={() => setIsVisible(true)}
        onBlur={() => setIsVisible(false)}
        aria-label="More information"
      >
        <RiInformationLine className="w-4 h-4" />
      </button>
      {isVisible && (
        <div className="absolute z-50 bottom-full left-1/2 -translate-x-1/2 mb-2 w-64 p-3 text-xs text-gray-700 dark:text-gray-300 bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded-lg shadow-lg">
          {content}
          <div className="absolute top-full left-1/2 -translate-x-1/2 -mt-1">
            <div className="border-4 border-transparent border-t-white dark:border-t-gray-800" />
          </div>
        </div>
      )}
    </span>
  )
}
