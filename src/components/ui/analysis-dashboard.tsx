"use client";

import * as React from "react";
import Image from "next/image";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { Badge } from "@/components/ui/badge";
import { TrendingUp, Target, ShieldCheck, MapPin, Gauge, Activity, Terminal, Binary } from "lucide-react";

const FALLBACK_IMAGE = "/window.svg";

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

export type AnalysisData = {
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
};

type AnalysisDashboardProps = {
  data: AnalysisData;
  logs?: string[];
};



function VehicleThumbnail({ src, alt }: { src: string; alt: string }) {
  const [failed, setFailed] = React.useState(false);
  const resolvedSrc = failed || !src ? FALLBACK_IMAGE : src;

  return (
    <Image
      src={resolvedSrc}
      alt={alt}
      width={100}
      height={100}
      unoptimized
      onError={() => setFailed(true)}
      className="h-full w-full object-cover transition-transform group-hover:scale-105"
    />
  );
}

function WatchlistTable({ rows }: { rows: Vehicle[] }) {
  if (!rows.length) {
    return (
      <div className="rounded-xl border border-slate-200 bg-slate-50 px-4 py-8 text-center text-sm text-slate-600">
        No vehicles match this filter yet.
      </div>
    );
  }
  return (
    <div className="overflow-hidden rounded-xl border border-slate-200 bg-white shadow-sm">
      <Table>
        <TableHeader>
          <TableRow className="bg-slate-50/50 hover:bg-slate-50/50">
            <TableHead className="w-[300px] text-xs font-bold uppercase tracking-wider text-slate-500">Vehicle</TableHead>
            <TableHead className="text-xs font-bold uppercase tracking-wider text-slate-500">Auction</TableHead>
            <TableHead className="text-xs font-bold uppercase tracking-wider text-slate-500">Stats</TableHead>
            <TableHead className="text-xs font-bold uppercase tracking-wider text-slate-500 text-right">Bid Limit</TableHead>
            <TableHead className="w-[100px]"></TableHead>
          </TableRow>
        </TableHeader>
        <TableBody>
          {rows.map((vehicle, idx) => (
            <TableRow key={`${vehicle.lot_number}-${idx}`} className="group hover:bg-slate-50/50 border-slate-100">
              <TableCell>
                <div className="flex items-center gap-4">
                  <div className="relative h-14 w-20 shrink-0 overflow-hidden rounded-lg bg-slate-100 border border-slate-200">
                    <VehicleThumbnail
                      src={vehicle.image_link}
                      alt={vehicle.year_make_model}
                    />
                  </div>
                  <div className="min-w-0 space-y-1">
                    <p className="truncate text-sm font-bold text-slate-900 leading-tight">
                      {vehicle.year_make_model}
                    </p>
                    <div className="text-[11px] text-slate-500 truncate flex items-center gap-1">
                      <Badge variant="outline" className="text-[9px] py-0 px-1 border-slate-200 bg-slate-100 text-slate-600 font-bold uppercase">
                        Lot {vehicle.lot_number}
                      </Badge>
                      <span className="text-slate-300">|</span>
                      {vehicle.damage}
                    </div>
                  </div>
                </div>
              </TableCell>
              <TableCell>
                <div className="space-y-1">
                  <p className="text-sm font-semibold text-slate-700">{vehicle.auction_date}</p>
                  <p className="text-[11px] text-slate-500 font-medium">{vehicle.auction_time}</p>
                </div>
              </TableCell>
              <TableCell>
                <div className="space-y-1.5">
                  <div className="flex items-center gap-1.5 text-slate-600">
                    <MapPin className="h-3 w-3 text-sky-500" />
                    <span className="text-[11px] font-medium truncate max-w-[150px]">{vehicle.location}</span>
                  </div>
                  <div className="flex items-center gap-1.5 text-slate-600">
                    <Gauge className="h-3 w-3 text-amber-500" />
                    <span className="text-[11px] font-medium">{vehicle.miles}</span>
                  </div>
                </div>
              </TableCell>
              <TableCell className="text-right">
                <div className="space-y-1">
                  <p className="text-lg font-black text-slate-900 tabular-nums">
                    {vehicle.bid_cap}
                  </p>
                  <p className="text-[10px] text-slate-400 font-medium">Data-backed cap</p>
                </div>
              </TableCell>
              <TableCell>
                <a
                  href={vehicle.lot_link}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="inline-flex h-8 items-center justify-center rounded-lg bg-slate-900 px-3 text-xs font-bold text-white transition hover:bg-slate-800 shadow-sm"
                >
                  View Link
                </a>
              </TableCell>
            </TableRow>
          ))}
        </TableBody>
      </Table>
    </div>
  );
}

export function AnalysisDashboard({ data, logs }: AnalysisDashboardProps) {
  const watchlistRows = React.useMemo(() => data.watchlist.top_results, [data.watchlist.top_results]);
  const stats = data.watchlist.score_stats;
  const makes = data.watchlist.make_distribution || {};

  return (
    <div className="space-y-8 animate-in fade-in slide-in-from-bottom-4 duration-1000">
      {/* Header Summary Stats */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <Card className="border-slate-200 shadow-sm bg-white overflow-hidden group">
          <div className="absolute top-0 right-0 p-2 opacity-10 group-hover:opacity-20 transition">
            <TrendingUp className="h-10 w-10 text-sky-600" />
          </div>
          <CardHeader className="pb-2">
            <CardDescription className="text-xs font-bold uppercase tracking-wider text-slate-500">Matches Found</CardDescription>
            <CardTitle className="text-3xl font-black text-slate-900">{data.watchlist.result_count}</CardTitle>
          </CardHeader>
        </Card>
        <Card className="border-slate-200 shadow-sm bg-white overflow-hidden group">
          <div className="absolute top-0 right-0 p-2 opacity-10 group-hover:opacity-20 transition">
            <Target className="h-10 w-10 text-emerald-600" />
          </div>
          <CardHeader className="pb-2">
            <CardDescription className="text-xs font-bold uppercase tracking-wider text-slate-500">Adaptive Threshold</CardDescription>
            <CardTitle className="text-3xl font-black text-slate-900">{data.watchlist.adaptive_min_score.toFixed(1)}</CardTitle>
          </CardHeader>
        </Card>
        <Card className="border-slate-200 shadow-sm bg-white overflow-hidden group">
          <div className="absolute top-0 right-0 p-2 opacity-10 group-hover:opacity-20 transition">
            <ShieldCheck className="h-10 w-10 text-amber-600" />
          </div>
          <CardHeader className="pb-2">
            <CardDescription className="text-xs font-bold uppercase tracking-wider text-slate-500">Average Score</CardDescription>
            <CardTitle className="text-3xl font-black text-slate-900">{stats?.average?.toFixed(1) || '0.0'}</CardTitle>
          </CardHeader>
        </Card>
        <Card className="border-slate-200 shadow-sm bg-sky-600 text-white overflow-hidden">
          <CardHeader className="pb-2">
            <CardDescription className="text-xs font-bold uppercase tracking-wider text-sky-100">Top Market Hub</CardDescription>
            <CardTitle className="text-2xl font-black">{Object.keys(makes)[0] || 'N/A'}</CardTitle>
          </CardHeader>
        </Card>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
        <div className="lg:col-span-2 space-y-6">
          <div className="flex items-center justify-between">
            <div>
              <h2 className="text-2xl font-black text-slate-900 tracking-tight">Active Watchlist</h2>
              <p className="text-slate-500 text-sm font-medium">Hand-picked by AI based on your buying power and history.</p>
            </div>
            <Badge className="bg-sky-600 font-bold px-3 py-1">A-Tier Matches</Badge>
          </div>
          <WatchlistTable rows={watchlistRows} />
        </div>

        <div className="space-y-6">
          <h2 className="text-2xl font-black text-slate-900 tracking-tight">Pattern Insights</h2>

          <Card className="border-slate-200 shadow-sm bg-white">
            <CardHeader className="pb-3 border-b border-slate-100">
              <CardTitle className="text-sm font-bold uppercase text-slate-600">Make Distribution</CardTitle>
            </CardHeader>
            <CardContent className="pt-4 space-y-4">
              {Object.entries(makes).slice(0, 5).map(([make, ratio]) => (
                <div key={make} className="space-y-1.5">
                  <div className="flex justify-between text-xs font-bold">
                    <span className="text-slate-900 uppercase tracking-wide">{make}</span>
                    <span className="text-slate-500">{(ratio * 100).toFixed(0)}%</span>
                  </div>
                  <div className="h-2 w-full bg-slate-100 rounded-full overflow-hidden">
                    <div
                      className="h-full bg-sky-600 transition-all duration-1000"
                      style={{ width: `${ratio * 100}%` }}
                    />
                  </div>
                </div>
              ))}
            </CardContent>
          </Card>

          <Card className="border-slate-200 shadow-sm bg-white text-slate-900 overflow-hidden relative">
            <div className="absolute top-0 right-0 p-4 opacity-5">
              <Activity className="h-12 w-12 text-slate-900" />
            </div>
            <CardHeader className="pb-2">
              <CardTitle className="text-lg font-bold">Analysis Strategy</CardTitle>
              <CardDescription className="text-slate-500 text-xs font-medium">Derived from {data.inventory.total_count?.toLocaleString() || '0'} scanned assets.</CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="grid grid-cols-2 gap-4">
                <div className="space-y-1">
                  <p className="text-[10px] font-bold uppercase tracking-widest text-slate-400">Denied</p>
                  <p className="text-xl font-black text-rose-600/80">
                    {((data.inventory.total_count || 0) - (data.watchlist.result_count || 0)).toLocaleString()}
                  </p>
                </div>
                <div className="space-y-1">
                  <p className="text-[10px] font-bold uppercase tracking-widest text-slate-400">Yield</p>
                  <p className="text-xl font-black text-emerald-600/80">
                    {data.inventory.total_count ? ((data.watchlist.result_count / data.inventory.total_count) * 100).toFixed(2) : '0'}%
                  </p>
                </div>
              </div>
              <p className="text-sm text-slate-600 leading-relaxed font-medium pt-2 border-t border-slate-100 italic">
                Targeted acquisition matches identified via historical pattern replication.
              </p>
            </CardContent>
          </Card>

          {/* Terminal Log Box */}
          <Card className="border-slate-200 shadow-sm bg-slate-50 text-slate-600 overflow-hidden font-mono text-[10px] leading-relaxed -mt-4">
            <div className="flex items-center justify-between px-4 py-2 bg-slate-100 border-b border-slate-200">
              <div className="flex items-center gap-2">
                <div className="flex gap-1.5 opacity-40">
                  <div className="h-1.5 w-1.5 rounded-full bg-slate-400" />
                  <div className="h-1.5 w-1.5 rounded-full bg-slate-400" />
                  <div className="h-1.5 w-1.5 rounded-full bg-slate-400" />
                </div>
                <span className="text-[9px] font-bold text-slate-500 uppercase tracking-widest ml-2">Vectorized.Analytical.Proof-of-Work</span>
              </div>
              <div className="flex items-center gap-2 opacity-30 text-slate-900">
                <Terminal className="h-3 w-3" />
                <Binary className="h-3 w-3" />
              </div>
            </div>
            <div className="p-4 max-h-[250px] overflow-y-auto space-y-1 custom-scrollbar bg-white/50">
              {logs && logs.length > 0 ? (
                logs.map((log, i) => (
                  <div key={i} className="flex gap-4">
                    <span className="text-slate-300 select-none font-bold w-6 text-right">{i + 1}</span>
                    <span className="break-all font-medium italic opacity-80">{log}</span>
                  </div>
                ))
              ) : (
                <div className="flex flex-col items-center justify-center py-8 opacity-40 italic">
                  <p>[0.0.0.1] Awaiting signal...</p>
                  <p className="text-[8px] uppercase mt-1 tracking-tighter">No historical logs found for current session</p>
                </div>
              )}
            </div>
          </Card>
        </div>
      </div>
    </div>
  );
}

// Keep the Demo version for initial view
export const MOCK_ANALYSIS: AnalysisData = {
  dealer_patterns: {},
  inventory: { total_count: 154200, filtered_count: 350 },
  bidding_analysis: {},
  watchlist: {
    requested_min_score: 7.0,
    adaptive_min_score: 7.5,
    result_count: 42,
    score_stats: { min: 7.1, max: 9.4, average: 8.2 },
    make_distribution: { Tesla: 0.45, Ford: 0.30, Toyota: 0.25 },
    primary: [],
    secondary: [],
    top_results: [
      {
        auction_date: "2024-10-16",
        auction_time: "10:00 EDT",
        lot_link: "#",
        lot_number: "45291032",
        image_link: "https://images.unsplash.com/photo-1552519507-da3b142c6e3d?auto=format&fit=crop&w=300&q=80",
        year_make_model: "2022 TESLA MODEL Y",
        damage: "REAR END",
        bid_cap: "$24,500",
        reasoning: "Score 8.5 | High-signal make match.",
        location: "MD - BALTIMORE EAST",
        miles: "18,450 mi",
      },
      {
        auction_date: "2024-10-18",
        auction_time: "12:30 EDT",
        lot_link: "#",
        lot_number: "12348901",
        image_link: "https://images.unsplash.com/photo-1617813489386-1e4f79ad4f98?auto=format&fit=crop&w=300&q=80",
        year_make_model: "2021 FORD F-150 LARIAT",
        damage: "FRONT END",
        bid_cap: "$26,800",
        reasoning: "Score 8.2 | matches your truck rebuild profile.",
        location: "TX - HOUSTON",
        miles: "42,000 mi",
      }
    ],
  },
};

export function DemoAnalysisDashboard() {
  return <AnalysisDashboard data={MOCK_ANALYSIS} />;
}
