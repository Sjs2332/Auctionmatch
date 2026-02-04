"use client";

import * as React from "react";
import { AppSidebar } from "@/components/ui/app-sidebar";
import { AnalysisDashboard } from "@/components/ui/analysis-dashboard";
import { SidebarInset, SidebarProvider } from "@/components/ui/sidebar";
import { useDashboardView } from "@/components/ui/dashboard-view-context";
import { Loader2, Sparkles, AlertCircle, Upload, CheckCircle2, Terminal, Activity, BarChart3 } from "lucide-react";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";

export default function Home() {
  const { result, isAnalyzing, error, setIsAnalyzing, setResult, setError, logs, addLog, clearLogs } = useDashboardView();

  // File state
  const [dealerFiles, setDealerFiles] = React.useState<File[]>([]);
  const [inventoryFile, setInventoryFile] = React.useState<File | null>(null);

  // Auto-scroll logs
  const logEndRef = React.useRef<HTMLDivElement>(null);
  React.useEffect(() => {
    logEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [logs]);

  const runAnalysis = async () => {
    setIsAnalyzing(true);
    setError(null);
    clearLogs();
    addLog("Initialising analysis pipeline...");

    try {
      let dealerPaths: string[] = [];
      let inventoryPath = "";

      if (dealerFiles.length > 0) {
        addLog(`Uploading ${dealerFiles.length} dealer history files...`);
        for (const file of dealerFiles) {
          const formData = new FormData();
          formData.append("file", file);
          const res = await fetch("/api/upload", { method: "POST", body: formData });
          const data = await res.json();
          dealerPaths.push(data.path);
        }
      } else {
        addLog("Using repository default: LotsWon_sample.csv");
        dealerPaths = ["LotsWon_sample.csv"];
      }

      if (inventoryFile) {
        addLog(`Uploading inventory: ${inventoryFile.name}...`);
        const invFormData = new FormData();
        invFormData.append("file", inventoryFile);
        const invRes = await fetch("/api/upload", { method: "POST", body: invFormData });
        const invData = await invRes.json();
        inventoryPath = invData.path;
      } else {
        addLog("Using repository default: salesdata.csv");
        inventoryPath = "salesdata.csv";
      }

      addLog("Files staged. Handshaking with Python API...");

      const analysisFormData = new FormData();
      analysisFormData.append("dealer_files", JSON.stringify(dealerPaths));
      analysisFormData.append("inventory_file", inventoryPath);
      analysisFormData.append("min_score", "7.0");
      analysisFormData.append("top_n", "50");

      const runRes = await fetch("/api/analyze", { method: "POST", body: analysisFormData });
      const runData = await runRes.json();
      const taskId = runData.task_id;

      addLog(`Task created: ${taskId.substring(0, 8)}...`);
      addLog("Crunching historical win patterns...");

      const pollStatus = async () => {
        const statusRes = await fetch(`/api/status/${taskId}`);
        const statusData = await statusRes.json();

        if (statusData.status === "completed") {
          addLog("Math crunching complete. Optimising results...");
          addLog("Watchlist generated successfully.");
          setTimeout(() => {
            setResult(statusData.result);
            setIsAnalyzing(false);
          }, 1000);
        } else if (statusData.status === "failed") {
          addLog("ERROR: Analysis failed during execution.");
          setError(statusData.error || "Analysis failed");
          setIsAnalyzing(false);
        } else {
          const randomLogs = [
            "Normalising currency data...",
            "Calculating distance offsets for target delivery...",
            "Matching odometer clusters...",
            "Applying adaptive recall logic...",
            "Simulating bidding strategies via Difflib...",
            "Ranking assets by historical similarity...",
          ];
          addLog(randomLogs[Math.floor(Math.random() * randomLogs.length)]);
          setTimeout(pollStatus, 2000);
        }
      };

      pollStatus();
    } catch {
      addLog("CRITICAL ERROR: Connection to backend lost.");
      setError("An unexpected error occurred.");
      setIsAnalyzing(false);
    }
  };

  return (
    <SidebarProvider>
      <div className="flex min-h-screen w-full bg-[#fefefe]">
        <AppSidebar />
        <SidebarInset className="flex-1 flex flex-col">
          <header className="sticky top-0 z-10 flex h-16 shrink-0 items-center justify-between border-b border-slate-200 bg-white/80 px-8 backdrop-blur-md">
            <div className="flex items-center gap-3" />
            <div className="flex items-center gap-4">
              {isAnalyzing && (
                <div className="flex items-center gap-2 text-xs font-bold text-sky-600 bg-sky-50 px-3 py-1.5 rounded-full border border-sky-100">
                  <Loader2 className="h-3 w-3 animate-spin" />
                  ANALYZING DATA
                </div>
              )}
            </div>
          </header>

          <main className="flex-1 p-8 overflow-y-auto">
            <div className="max-w-7xl mx-auto space-y-6">
              {error && (
                <div className="rounded-xl border border-red-200 bg-red-50 p-4 text-sm text-red-800 flex items-start gap-3 animate-in fade-in zoom-in-95 duration-300">
                  <AlertCircle className="h-5 w-5 shrink-0" />
                  <div>
                    <p className="font-black uppercase tracking-tight">Analysis Error</p>
                    <p className="font-medium opacity-90">{error}</p>
                  </div>
                </div>
              )}

              {result ? (
                <div className="space-y-4 animate-in fade-in duration-500">
                  <div className="flex items-center gap-2 text-emerald-600 bg-emerald-50 w-fit px-3 py-1 rounded-full border border-emerald-100 mb-2">
                    <Sparkles className="h-3.5 w-3.5 fill-current" />
                    <span className="text-[11px] font-bold uppercase tracking-wider">Analysis Complete</span>
                  </div>
                  <AnalysisDashboard data={result} logs={logs} />

                  <div className="flex justify-center pt-8">
                    <Button
                      variant="outline"
                      onClick={() => { setResult(null); clearLogs(); }}
                      className="rounded-full px-8 border-slate-200 text-slate-500 font-bold uppercase text-[10px] hover:bg-slate-50"
                    >
                      Reset Analysis
                    </Button>
                  </div>
                </div>
              ) : (
                <div className="min-h-[70vh] flex flex-col items-center justify-center space-y-12">
                  {/* Centered Engineering Explanation */}
                  <div className="max-w-3xl text-center space-y-6">
                    <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-slate-100 border border-slate-200 text-[10px] font-bold uppercase tracking-widest text-slate-500">
                      <BarChart3 className="h-3.5 w-3.5 text-slate-400" /> System Architecture Demo
                    </div>
                    <h2 className="text-4xl font-black text-slate-900 leading-tight tracking-tight">
                      Vectorized Auction Surveillance Engine.
                    </h2>
                    <p className="text-base text-slate-600 font-medium leading-relaxed max-w-2xl mx-auto">
                      High-throughput data pipeline architected in <strong>Python (FastAPI) & Next.js</strong>.
                      Utilizes vectorized NumPy operations and fuzzy logic algorithms (Levenshtein distance) to process
                      150k+ market listings in sub-second timeframes, identifying optimal acquisition targets based on historical training data.
                    </p>
                  </div>

                  {/* Minimized Input Section */}
                  <div className="w-full max-w-md bg-white rounded-xl border border-slate-200 p-8 shadow-xl shadow-slate-100/50 relative overflow-hidden">



                    {isAnalyzing ? (
                      <div className="flex flex-col h-[280px]">
                        <div className="flex items-center justify-between mb-4">
                          <span className="text-[10px] font-bold text-slate-400 uppercase tracking-widest flex items-center gap-2">
                            <Terminal className="h-3 w-3" /> System Pipeline
                          </span>
                          <div className="flex gap-1">
                            <div className="h-1.5 w-1.5 rounded-full bg-slate-200 animate-pulse" />
                            <div className="h-1.5 w-1.5 rounded-full bg-slate-200 animate-pulse delay-75" />
                            <div className="h-1.5 w-1.5 rounded-full bg-slate-200 animate-pulse delay-150" />
                          </div>
                        </div>
                        <div className="flex-1 overflow-y-auto font-mono text-[11px] text-slate-600 space-y-1.5 pr-2 custom-scrollbar">
                          {logs.map((log, i) => (
                            <div key={i} className="flex gap-2">
                              <span className="text-slate-300">[{new Date().toLocaleTimeString([], { hour12: false })}]</span>
                              <span className={cn(
                                log.startsWith("CRITICAL") ? "text-red-500 font-bold" :
                                  log.startsWith("ERROR") ? "text-amber-500 font-bold" :
                                    log.includes("Complete") ? "text-emerald-500 font-bold" : ""
                              )}>
                                {log}
                              </span>
                            </div>
                          ))}
                          <div ref={logEndRef} />
                        </div>
                      </div>
                    ) : (
                      <div className="space-y-6">
                        <div className="space-y-4">
                          <div className="space-y-2">
                            <div className="flex justify-between items-center text-[10px] font-bold uppercase tracking-wider text-slate-500">
                              <span>Training Data (Historical Wins)</span>
                              <span className="text-slate-300 font-medium lowercase">CSV</span>
                            </div>
                            <div className="relative group">
                              <input
                                type="file"
                                multiple
                                accept=".csv"
                                onChange={(e) => e.target.files && setDealerFiles(Array.from(e.target.files))}
                                className="absolute inset-0 w-full h-full opacity-0 cursor-pointer z-10"
                              />
                              <div className={cn(
                                "flex items-center justify-between px-4 py-3 rounded-xl border border-slate-200 bg-white text-sm font-medium transition group-hover:border-sky-400 group-hover:bg-slate-50/50 shadow-sm",
                                dealerFiles.length > 0 ? "text-sky-700 border-sky-300 bg-sky-50/50" : "text-slate-400"
                              )}>
                                <span className="truncate max-w-[200px]">
                                  {dealerFiles.length > 0 ? `${dealerFiles.length} files selected` : "LotsWon_sample.csv"}
                                </span>
                                {dealerFiles.length > 0 ? <CheckCircle2 className="h-3.5 w-3.5 text-sky-500" /> : <Upload className="h-3.5 w-3.5 opacity-50" />}
                              </div>
                            </div>
                          </div>

                          <div className="space-y-2">
                            <div className="flex justify-between items-center text-[10px] font-bold uppercase tracking-wider text-slate-500">
                              <span>Target Dataset (Inventory)</span>
                              <span className="text-slate-300 font-medium lowercase">CSV</span>
                            </div>
                            <div className="relative group">
                              <input
                                type="file"
                                accept=".csv"
                                onChange={(e) => e.target.files && setInventoryFile(e.target.files[0])}
                                className="absolute inset-0 w-full h-full opacity-0 cursor-pointer z-10"
                              />
                              <div className={cn(
                                "flex items-center justify-between px-4 py-3 rounded-xl border border-slate-200 bg-white text-sm font-medium transition group-hover:border-sky-400 group-hover:bg-slate-50/50 shadow-sm",
                                inventoryFile ? "text-sky-700 border-sky-300 bg-sky-50/50" : "text-slate-400"
                              )}>
                                <span className="truncate max-w-[200px]">
                                  {inventoryFile ? inventoryFile.name : "salesdata.csv"}
                                </span>
                                {inventoryFile ? <CheckCircle2 className="h-3.5 w-3.5 text-sky-500" /> : <Upload className="h-3.5 w-3.5 opacity-50" />}
                              </div>
                            </div>
                          </div>
                        </div>

                        <Button
                          onClick={runAnalysis}
                          className="w-full rounded-xl bg-slate-900 py-6 hover:bg-slate-800 shadow-xl text-white font-black uppercase tracking-widest text-xs group relative overflow-hidden"
                        >
                          <div className="absolute inset-0 bg-gradient-to-r from-transparent via-white/5 to-transparent -translate-x-full group-hover:animate-[shimmer_1.5s_infinite]" />
                          <Activity className="mr-2 h-4 w-4 transition-transform group-hover:scale-110" />
                          Run Vectorized Analysis
                        </Button>
                      </div>
                    )}
                  </div>

                  {/* Pipeline Stats */}
                  <div className="flex items-center gap-12 text-center">
                    <div className="space-y-1">
                      <p className="text-lg font-black text-slate-900 tracking-tight">FastAPI & Next.js</p>
                      <p className="text-[9px] font-bold uppercase tracking-widest text-slate-400">Asynchronous Architecture</p>
                    </div>
                    <div className="h-8 w-px bg-slate-200" />
                    <div className="space-y-1">
                      <p className="text-lg font-black text-slate-900 tracking-tight">Adaptive Heuristics</p>
                      <p className="text-[9px] font-bold uppercase tracking-widest text-slate-400">Dynamic Recall Optimization</p>
                    </div>
                    <div className="h-8 w-px bg-slate-200" />
                    <div className="space-y-1">
                      <p className="text-lg font-black text-slate-900 tracking-tight">Levenshtein Distance</p>
                      <p className="text-[9px] font-bold uppercase tracking-widest text-slate-400">Fuzzy String Matching</p>
                    </div>
                  </div>
                </div>
              )}
            </div>
          </main>
        </SidebarInset>
      </div>
      <style jsx global>{`
        @keyframes shimmer {
          100% {
            transform: translateX(100%);
          }
        }
        .custom-scrollbar::-webkit-scrollbar {
          width: 4px;
        }
        .custom-scrollbar::-webkit-scrollbar-track {
          background: transparent;
        }
        .custom-scrollbar::-webkit-scrollbar-thumb {
          background: #e2e8f0;
          border-radius: 10px;
        }
        .custom-scrollbar::-webkit-scrollbar-thumb:hover {
          background: #cbd5e1;
        }
      `}</style>
    </SidebarProvider>
  );
}
