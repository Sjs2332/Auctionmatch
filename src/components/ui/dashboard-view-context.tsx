"use client";

import * as React from "react";

export type DashboardView = "analyzer";

export interface Vehicle {
  auction_date: string;
  auction_time: string;
  lot_link: string;
  lot_number: string;
  image_link: string;
  year_make_model: string;
  damage: string;
  bid_cap: string;
  reasoning: string;
  location: string;
  miles: string;
}

export interface AnalysisResult {
  dealer_patterns: Record<string, unknown>;
  inventory: {
    total_count: number;
    filtered_count: number;
    [key: string]: unknown;
  };
  bidding_analysis: Record<string, unknown>;
  watchlist: {
    requested_min_score: number;
    adaptive_min_score: number;
    result_count: number;
    primary: Vehicle[];
    secondary: Vehicle[];
    top_results: Vehicle[];
    score_stats?: {
      min: number;
      max: number;
      average: number;
    };
    make_distribution?: Record<string, number>;
  };
}

type DashboardViewContextValue = {
  view: DashboardView;
  setView: (view: DashboardView) => void;
  isAnalyzing: boolean;
  setIsAnalyzing: (analyzing: boolean) => void;
  result: AnalysisResult | null;
  setResult: (result: AnalysisResult | null) => void;
  error: string | null;
  setError: (error: string | null) => void;
  logs: string[];
  addLog: (log: string) => void;
  clearLogs: () => void;
};

const DashboardViewContext = React.createContext<DashboardViewContextValue | null>(null);

export function DashboardViewProvider({
  children,
  initialView = "analyzer",
}: React.PropsWithChildren<{ initialView?: DashboardView }>) {
  const [view, setView] = React.useState<DashboardView>(initialView);
  const [isAnalyzing, setIsAnalyzing] = React.useState(false);
  const [result, setResult] = React.useState<AnalysisResult | null>(null);
  const [error, setError] = React.useState<string | null>(null);
  const [logs, setLogs] = React.useState<string[]>([]);

  const addLog = React.useCallback((log: string) => {
    setLogs((prev) => [...prev, log]);
  }, []);

  const clearLogs = React.useCallback(() => {
    setLogs([]);
  }, []);

  const contextValue = React.useMemo(
    () => ({
      view,
      setView,
      isAnalyzing,
      setIsAnalyzing,
      result,
      setResult,
      error,
      setError,
      logs,
      addLog,
      clearLogs,
    }),
    [view, isAnalyzing, result, error, logs, addLog, clearLogs],
  );

  return <DashboardViewContext.Provider value={contextValue}>{children}</DashboardViewContext.Provider>;
}

export function useDashboardView() {
  const context = React.useContext(DashboardViewContext);
  if (!context) {
    throw new Error("useDashboardView must be used within a DashboardViewProvider.");
  }
  return context;
}
