"use client";

import React, { useState, useEffect, useCallback } from "react";
import { AnalyticsSummary, getAnalyticsSummary } from "@/lib/api";

export default function AnalyticsWorkspaceView() {
  const [summary, setSummary] = useState<AnalyticsSummary | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const fetchAnalytics = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const data = await getAnalyticsSummary();
      setSummary(data);
    } catch (err: any) {
      setError(err.message || "Failed to load security analytics.");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchAnalytics();
  }, [fetchAnalytics]);

  return (
    <div className="p-6 md:p-8 space-y-8 max-w-[1600px] mx-auto min-h-full">
      {/* Header */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 border-b border-zinc-800/80 pb-6">
        <div>
          <div className="flex items-center gap-3">
            <h1 className="text-2xl font-bold tracking-tight text-white flex items-center gap-2.5">
              <span className="w-3 h-3 rounded-full bg-emerald-400 animate-pulse" />
              Security Intelligence Analytics &amp; Diagnostics
            </h1>
            <span className="px-2.5 py-0.5 rounded-full text-xs font-mono font-semibold bg-emerald-500/10 text-emerald-400 border border-emerald-500/30">
              Live Telemetry
            </span>
          </div>
          <p className="text-sm text-zinc-400 mt-1.5">
            System-wide detection validation ratios, tracking continuity metrics, AI detector health, and invariant parity checks.
          </p>
        </div>

        <button
          onClick={fetchAnalytics}
          className="flex items-center gap-2 px-3.5 py-2 bg-zinc-800 hover:bg-zinc-700 text-zinc-200 rounded-lg text-xs font-semibold font-mono border border-zinc-700 transition-all"
        >
          <svg className="w-3.5 h-3.5 text-emerald-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
          </svg>
          Refresh Telemetry
        </button>
      </div>

      {/* Loading & Error States */}
      {loading && (
        <div className="flex flex-col items-center justify-center py-20 text-zinc-500 space-y-3">
          <div className="w-7 h-7 border-2 border-emerald-500 border-t-transparent rounded-full animate-spin" />
          <span className="text-sm font-mono">Aggregating system-wide telemetry...</span>
        </div>
      )}

      {error && !loading && (
        <div className="p-4 bg-red-500/10 border border-red-500/30 rounded-xl text-red-400 flex items-center justify-between">
          <span className="text-sm font-medium">{error}</span>
          <button onClick={fetchAnalytics} className="px-3 py-1 bg-red-500/20 text-red-300 rounded text-xs">
            Retry
          </button>
        </div>
      )}

      {/* Analytics Content */}
      {!loading && !error && summary && (
        <div className="space-y-8">
          {/* Top Metric Cards */}
          <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-6 gap-4">
            <div className="bg-zinc-900/80 border border-zinc-800 p-4 rounded-xl space-y-1">
              <span className="text-xs font-medium text-zinc-400">Ingested Videos</span>
              <div className="text-2xl font-bold font-mono text-white">
                {summary.total_videos}
              </div>
              <span className="text-[11px] text-zinc-500 font-mono">
                {summary.processed_videos} processed
              </span>
            </div>

            <div className="bg-zinc-900/80 border border-zinc-800 p-4 rounded-xl space-y-1">
              <span className="text-xs font-medium text-zinc-400">Camera Sources</span>
              <div className="text-2xl font-bold font-mono text-sky-400">
                {summary.total_cameras ?? 0}
              </div>
              <span className="text-[11px] text-zinc-500 font-mono">Registered cameras</span>
            </div>

            <div className="bg-zinc-900/80 border border-zinc-800 p-4 rounded-xl space-y-1">
              <span className="text-xs font-medium text-zinc-400">Validated Detections</span>
              <div className="text-2xl font-bold font-mono text-emerald-400">
                {summary.validated_detections_count.toLocaleString()}
              </div>
              <span className="text-[11px] text-zinc-500 font-mono">
                of {summary.raw_observations_count.toLocaleString()} raw
              </span>
            </div>

            <div className="bg-zinc-900/80 border border-zinc-800 p-4 rounded-xl space-y-1">
              <span className="text-xs font-medium text-zinc-400">Rejected Detections</span>
              <div className="text-2xl font-bold font-mono text-red-400">
                {summary.rejected_detections_count.toLocaleString()}
              </div>
              <span className="text-[11px] text-zinc-500 font-mono">False positives filtered</span>
            </div>

            <div className="bg-zinc-900/80 border border-amber-800/50 p-4 rounded-xl space-y-1">
              <span className="text-xs font-medium text-amber-400">Uncertain Detections</span>
              <div className="text-2xl font-bold font-mono text-amber-400">
                {(summary.uncertain_detections_count ?? 0).toLocaleString()}
              </div>
              <span className="text-[11px] text-amber-600/80 font-mono">Require analyst review</span>
            </div>

            <div className="bg-zinc-900/80 border border-zinc-800 p-4 rounded-xl space-y-1">
              <span className="text-xs font-medium text-zinc-400">Anonymous Tracks</span>
              <div className="text-2xl font-bold font-mono text-blue-400">
                {summary.total_tracks.toLocaleString()}
              </div>
              <span className="text-[11px] text-zinc-500 font-mono">Continuity preserved</span>
            </div>
          </div>

          {/* Core Pipeline Parity Invariant Box */}
          <div className="bg-zinc-900/60 border border-zinc-800 p-5 rounded-xl space-y-3">
            <div className="flex items-center justify-between">
              <h3 className="text-sm font-bold text-zinc-200 uppercase tracking-wider flex items-center gap-2">
                <span className={`w-2.5 h-2.5 rounded-full ${summary.parity_consistent ? "bg-emerald-400" : "bg-red-400"}`} />
                Pipeline Verification &amp; Parity Invariant
              </h3>
              <span className={`px-2 py-0.5 rounded text-xs font-mono font-bold ${
                summary.parity_consistent ? "bg-emerald-500/10 text-emerald-400 border border-emerald-500/30" : "bg-red-500/10 text-red-400 border border-red-500/30"
              }`}>
                {summary.parity_consistent ? "INVARIANT VERIFIED" : "PARITY MISMATCH"}
              </span>
            </div>

            <p className="text-xs text-zinc-400 leading-relaxed">
              Authoritative pipeline invariant: <span className="font-mono text-zinc-200">RAW = VALID + REJECTED + UNCERTAIN</span>. All visual detections must resolve to one of three states — no detections are silently discarded.
              {summary.parity_formula && <span className="block mt-1 font-mono text-emerald-400 text-[11px]">{summary.parity_formula}</span>}
            </p>

            <div className="grid grid-cols-1 sm:grid-cols-4 gap-4 pt-2 font-mono text-xs">
              <div className="bg-zinc-950 p-3 rounded-lg border border-zinc-800/80">
                <span className="text-zinc-500 block text-[10px] uppercase">Raw Observations</span>
                <span className="text-base font-bold text-blue-400">{summary.raw_observations_count.toLocaleString()}</span>
              </div>
              <div className="bg-zinc-950 p-3 rounded-lg border border-emerald-800/50">
                <span className="text-zinc-500 block text-[10px] uppercase">Validated (VALID)</span>
                <span className="text-base font-bold text-emerald-400">{summary.validated_detections_count.toLocaleString()}</span>
              </div>
              <div className="bg-zinc-950 p-3 rounded-lg border border-zinc-800/80">
                <span className="text-zinc-500 block text-[10px] uppercase">Rejected (FALSE POS.)</span>
                <span className="text-base font-bold text-zinc-400">{summary.rejected_detections_count.toLocaleString()}</span>
              </div>
              <div className="bg-zinc-950 p-3 rounded-lg border border-amber-800/50">
                <span className="text-amber-600 block text-[10px] uppercase">Uncertain (REVIEW)</span>
                <span className="text-base font-bold text-amber-400">{(summary.uncertain_detections_count ?? 0).toLocaleString()}</span>
              </div>
            </div>
          </div>

          {/* Cases Distribution & Specialized Visual Detectors */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {/* Case Management Distribution */}
            <div className="bg-zinc-900/60 border border-zinc-800 p-5 rounded-xl space-y-4">
              <div className="flex items-center justify-between">
                <h3 className="text-sm font-bold text-zinc-200 uppercase tracking-wider flex items-center gap-2">
                  <span className="w-2 h-2 rounded-full bg-cyan-400" />
                  Case Management Lifecycle ({summary.total_cases})
                </h3>
              </div>

              <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 font-mono text-xs">
                <div className="bg-zinc-950 p-3 rounded-lg border border-zinc-800">
                  <span className="text-zinc-500 block text-[10px] uppercase">Open</span>
                  <span className="text-base font-bold text-white">{summary.cases_by_status.OPEN || 0}</span>
                </div>
                <div className="bg-zinc-950 p-3 rounded-lg border border-zinc-800">
                  <span className="text-zinc-500 block text-[10px] uppercase">Investigating</span>
                  <span className="text-base font-bold text-cyan-400">{summary.cases_by_status.INVESTIGATING || 0}</span>
                </div>
                <div className="bg-zinc-950 p-3 rounded-lg border border-zinc-800">
                  <span className="text-zinc-500 block text-[10px] uppercase">Review</span>
                  <span className="text-base font-bold text-amber-400">{summary.cases_by_status.REVIEW || 0}</span>
                </div>
                <div className="bg-zinc-950 p-3 rounded-lg border border-zinc-800">
                  <span className="text-zinc-500 block text-[10px] uppercase">Closed</span>
                  <span className="text-base font-bold text-zinc-400">{summary.cases_by_status.CLOSED || 0}</span>
                </div>
              </div>

              <div className="pt-2 border-t border-zinc-800/60">
                <span className="text-xs text-zinc-400 font-medium block mb-2">Priority Distribution</span>
                <div className="flex items-center gap-3 font-mono text-xs">
                  <span className="text-red-400">Critical: {summary.cases_by_priority.CRITICAL || 0}</span>
                  <span className="text-amber-400">High: {summary.cases_by_priority.HIGH || 0}</span>
                  <span className="text-blue-400">Medium: {summary.cases_by_priority.MEDIUM || 0}</span>
                  <span className="text-zinc-400">Low: {summary.cases_by_priority.LOW || 0}</span>
                </div>
              </div>
            </div>

            {/* Specialized Detectors Telemetry */}
            <div className="bg-zinc-900/60 border border-zinc-800 p-5 rounded-xl space-y-4">
              <div className="flex items-center justify-between">
                <h3 className="text-sm font-bold text-zinc-200 uppercase tracking-wider flex items-center gap-2">
                  <span className="w-2 h-2 rounded-full bg-orange-400" />
                  Specialized Visual Detectors
                </h3>
              </div>

              <div className="grid grid-cols-3 gap-3 font-mono text-xs">
                <div className="bg-zinc-950 p-3 rounded-lg border border-zinc-800">
                  <span className="text-zinc-500 block text-[10px] uppercase">Smoke RAW</span>
                  <span className="text-base font-bold text-zinc-300">{summary.specialized_counts.smoke || 0}</span>
                </div>
                <div className="bg-zinc-950 p-3 rounded-lg border border-zinc-800">
                  <span className="text-zinc-500 block text-[10px] uppercase">Fire RAW</span>
                  <span className="text-base font-bold text-zinc-300">{summary.specialized_counts.fire || 0}</span>
                </div>
                <div className="bg-zinc-950 p-3 rounded-lg border border-zinc-800">
                  <span className="text-zinc-500 block text-[10px] uppercase">Weapon RAW</span>
                  <span className="text-base font-bold text-zinc-300">{summary.specialized_counts.weapon || 0}</span>
                </div>
              </div>

              <p className="text-xs text-zinc-500 pt-2 border-t border-zinc-800/60">
                <strong className="text-amber-400/80">RAW counts</strong> — unvalidated observations from the specialized detector pipeline. Zero validated alerts on standard CCTV footage is expected behavior (Burglary, Highway, 4K).
              </p>
            </div>
          </div>

          {/* AI Intelligence Detector Health Grid */}
          <div className="space-y-4">
            <h2 className="text-lg font-bold text-zinc-200 flex items-center gap-2">
              <span className="w-2 h-2 rounded-full bg-emerald-400" />
              AI Intelligence Modules Health
            </h2>

            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
              {summary.detector_health.map((det) => (
                <div
                  key={det.id}
                  className="bg-zinc-900/80 border border-zinc-800 rounded-xl p-4 flex items-center justify-between"
                >
                  <div className="space-y-0.5">
                    <h4 className="text-sm font-semibold text-zinc-200">{det.name}</h4>
                    <p className="text-[11px] font-mono text-zinc-500 uppercase">{det.type}</p>
                  </div>
                  <span className="px-2 py-0.5 rounded text-[11px] font-mono font-bold bg-emerald-500/10 text-emerald-400 border border-emerald-500/30 uppercase">
                    {det.status}
                  </span>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
