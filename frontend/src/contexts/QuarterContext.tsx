"use client";

import { createContext, useContext, useState, useEffect, ReactNode } from "react";

const API_BASE = "http://localhost:8000/api/v1";

interface QuarterContextType {
  quarter: string | null; // null = use latest
  availableQuarters: string[];
  setQuarter: (q: string | null) => void;
  isLoading: boolean;
  displayQuarter: string; // For display purposes
}

const QuarterContext = createContext<QuarterContextType | undefined>(undefined);

export function QuarterProvider({ children }: { children: ReactNode }) {
  const [quarter, setQuarterState] = useState<string | null>(null);
  const [availableQuarters, setAvailableQuarters] = useState<string[]>([]);
  const [latestQuarter, setLatestQuarter] = useState<string>("");
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    // Load saved quarter from localStorage
    const saved = localStorage.getItem("selectedQuarter");
    if (saved && saved !== "null") {
      setQuarterState(saved);
    }

    // Fetch available quarters
    async function fetchQuarters() {
      try {
        const res = await fetch(`${API_BASE}/screener/quarters`);
        if (res.ok) {
          const data = await res.json();
          setAvailableQuarters(data.quarters || []);
          setLatestQuarter(data.latest || "");
        }
      } catch (error) {
        console.error("Failed to fetch quarters:", error);
      } finally {
        setIsLoading(false);
      }
    }

    fetchQuarters();
  }, []);

  const setQuarter = (q: string | null) => {
    setQuarterState(q);
    if (q === null) {
      localStorage.removeItem("selectedQuarter");
    } else {
      localStorage.setItem("selectedQuarter", q);
    }
  };

  // Display quarter: show actual quarter being used
  const displayQuarter = quarter || latestQuarter || "Latest";

  return (
    <QuarterContext.Provider
      value={{
        quarter,
        availableQuarters,
        setQuarter,
        isLoading,
        displayQuarter,
      }}
    >
      {children}
    </QuarterContext.Provider>
  );
}

export function useQuarter() {
  const context = useContext(QuarterContext);
  if (context === undefined) {
    throw new Error("useQuarter must be used within a QuarterProvider");
  }
  return context;
}
