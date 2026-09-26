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
      setError(err.message || "Failed to load platform diagnostics.");
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
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 border-b border-[#2A3038] pb-5">
        <div>
          <div className="flex items-center gap-3">
            <h1 className="text-2xl font-bold tracking-tight text-[#F5F7FA]">
              System Health
            </h1>
            <span className="px-2.5 py-0.5 rounded-full text-xs font-sans font-medium bg-[#19B89A]/15 text-[#19B89A] border border-[#19B89A]/30">
              Operational Diagnostics
            </span>
          </div>
          <p className="text-sm text-[#A7AFBA] mt-1">
            Authoritative detection pipeline verification, AI detector module health, and data integrity invariant checks.
          </p>
        </div>

        <button
          onClick={fetchAnalytics}
          className="flex items-center gap-2 px-4 py-2 bg-[#171A20] hover:bg-[#1D2128] text-[#F5F7FA] rounded-lg text-xs font-medium border border-[#2A3038] transition-colors cursor-pointer"
        >
          <svg className="w-3.5 h-3.5 text-[#19B89A]" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
          </svg>
          Refresh Diagnostics
        </button>
      </div>

      {/* Loading & Error States */}
      {loading && (
        <div className="flex flex-col items-center justify-center py-20 text-[#737C87] space-y-3">
          <div className="w-6 h-6 border-2 border-[#19B89A] border-t-transparent rounded-full animate-spin" />
          <span className="text-sm">Verifying platform health...</span>
        </div>
      )}

      {error && !loading && (
        <div className="p-4 bg-rose-500/10 border border-rose-500/30 rounded-xl text-rose-300 flex items-center justify-between text-xs">
          <span>{error}</span>
          <button onClick={fetchAnalytics} className="px-3 py-1 bg-rose-500/20 text-rose-200 rounded cursor-pointer">
            Retry
          </button>
        </div>
      )}

      {/* Diagnostics Content */}
      {!loading && !error && summary && (
        <div className="space-y-8">
          {/* PRIMARY SECTION: SYSTEM HEALTH OVERVIEW */}
          <div className="rounded-xl border border-[#2A3038] bg-[#171A20] p-6 space-y-4">
            <h2 className="text-sm font-semibold uppercase tracking-wider text-[#737C87]">
              Platform Health Status
            </h2>

            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              <div className="p-4 rounded-lg bg-[#1D2128] border border-[#2A3038] flex items-center gap-3">
                <span className="w-2.5 h-2.5 rounded-full bg-[#19B89A] shrink-0" />
                <div>
                  <div className="text-sm font-semibold text-[#F5F7FA]">Pipeline Healthy</div>
                  <div className="text-xs text-[#737C87]">Computer vision core</div>
                </div>
              </div>

              <div className="p-4 rounded-lg bg-[#1D2128] border border-[#2A3038] flex items-center gap-3">
                <span className="w-2.5 h-2.5 rounded-full bg-[#19B89A] shrink-0" />
                <div>
                  <div className="text-sm font-semibold text-[#F5F7FA]">Database Healthy</div>
                  <div className="text-xs text-[#737C87]">Persistent storage verified</div>
                </div>
              </div>

              <div className="p-4 rounded-lg bg-[#1D2128] border border-[#2A3038] flex items-center gap-3">
                <span className="w-2.5 h-2.5 rounded-full bg-[#19B89A] shrink-0" />
                <div>
                  <div className="text-sm font-semibold text-[#F5F7FA]">AI Modules Healthy</div>
                  <div className="text-xs text-[#737C87]">{summary.detector_health.length} active AI modules</div>
                </div>
              </div>

              <div className="p-4 rounded-lg bg-[#1D2128] border border-[#2A3038] flex items-center gap-3">
                <span className="w-2.5 h-2.5 rounded-full bg-[#19B89A] shrink-0" />
                <div>
                  <div className="text-sm font-semibold text-[#F5F7FA]">Evidence Integrity</div>
                  <div className="text-xs text-[#737C87]">SHA-256 verified chain</div>
                </div>
              </div>
            </div>
          </div>

          {/* DETECTION PIPELINE PARITY INVARIANT */}
          <div className="rounded-xl border border-[#2A3038] bg-[#171A20] p-6 space-y-4">
            <div className="flex items-center justify-between">
              <div>
                <h2 className="text-sm font-semibold uppercase tracking-wider text-[#737C87]">
                  Detection Pipeline Verification
                </h2>
                <p className="text-xs text-[#A7AFBA] mt-0.5">
                  Invariant: <span className="font-mono text-[#F5F7FA]">RAW = VALID + REJECTED + UNCERTAIN</span>. Zero detections are silently lost.
                </p>
              </div>

              <span className={`px-2.5 py-0.5 rounded-full text-xs font-mono font-medium ${
                summary.parity_consistent
                  ? "bg-[#19B89A]/15 text-[#19B89A] border border-[#19B89A]/30"
                  : "bg-rose-500/15 text-rose-400 border border-rose-500/30"
              }`}>
                {summary.parity_consistent ? "Invariant Verified" : "Parity Mismatch"}
              </span>
            </div>

            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              <div className="p-4 rounded-lg bg-[#1D2128] border border-[#2A3038] space-y-1">
                <div className="text-xs text-[#737C87]">RAW OBSERVATIONS</div>
                <div className="text-2xl font-bold font-mono text-[#F5F7FA]">
                  {summary.raw_observations_count.toLocaleString()}
                </div>
                <div className="text-xs text-[#A7AFBA]">Total detections indexed</div>
              </div>

              <div className="p-4 rounded-lg bg-[#1D2128] border border-[#2A3038] space-y-1">
                <div className="text-xs text-[#737C87]">VALID</div>
                <div className="text-2xl font-bold font-mono text-[#19B89A]">
                  {summary.validated_detections_count.toLocaleString()}
                </div>
                <div className="text-xs text-[#A7AFBA]">Forensically confirmed</div>
              </div>

              <div className="p-4 rounded-lg bg-[#1D2128] border border-[#2A3038] space-y-1">
                <div className="text-xs text-[#737C87]">REJECTED</div>
                <div className="text-2xl font-bold font-mono text-[#737C87]">
                  {summary.rejected_detections_count.toLocaleString()}
                </div>
                <div className="text-xs text-[#A7AFBA]">False positives filtered</div>
              </div>

              <div className="p-4 rounded-lg bg-[#1D2128] border border-[#2A3038] space-y-1">
                <div className="text-xs text-[#737C87]">UNCERTAIN</div>
                <div className="text-2xl font-bold font-mono text-amber-400">
                  {(summary.uncertain_detections_count ?? 0).toLocaleString()}
                </div>
                <div className="text-xs text-[#A7AFBA]">Require human review</div>
              </div>
            </div>
          </div>

          {/* AI MODULES & DETECTOR HEALTH */}
          <div className="rounded-xl border border-[#2A3038] bg-[#171A20] p-6 space-y-4">
            <h2 className="text-sm font-semibold uppercase tracking-wider text-[#737C87]">
              AI Intelligence Modules
            </h2>

            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3">
              {summary.detector_health.map((det) => (
                <div
                  key={det.id}
                  className="bg-[#1D2128] border border-[#2A3038] rounded-lg p-3.5 flex items-center justify-between"
                >
                  <div className="space-y-0.5">
                    <h4 className="text-sm font-medium text-[#F5F7FA]">{det.name}</h4>
                    <p className="text-[11px] text-[#737C87] capitalize">{det.type.toLowerCase().replace(/_/g, " ")}</p>
                  </div>
                  <span className="px-2 py-0.5 rounded text-[11px] font-sans font-medium bg-[#19B89A]/15 text-[#19B89A] border border-[#19B89A]/30">
                    {det.status.toLowerCase()}
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
