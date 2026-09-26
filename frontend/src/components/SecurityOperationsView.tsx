"use client";

import React, { useState, useEffect, useCallback } from "react";
import {
  CyberEventItem,
  SecurityAssetItem,
  CyberPhysicalCorrelationItem,
  CyberInvestigationResult,
  listCyberEvents,
  listSecurityAssets,
  correlateCyberPhysical,
  seedCyberDemoTelemetry,
  investigateCyber,
  createCyberEvent,
} from "@/lib/api";

interface SecurityOperationsViewProps {
  onInvestigateVideo?: (videoId: string) => void;
  onNavigateToCase?: (caseId: string) => void;
}

export default function SecurityOperationsView({
  onInvestigateVideo,
  onNavigateToCase,
}: SecurityOperationsViewProps) {
  // Top-level state
  const [assets, setAssets] = useState<SecurityAssetItem[]>([]);
  const [events, setEvents] = useState<CyberEventItem[]>([]);
  const [correlations, setCorrelations] = useState<CyberPhysicalCorrelationItem[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Filter state
  const [selectedAsset, setSelectedAsset] = useState<string>("all");
  const [selectedSeverity, setSelectedSeverity] = useState<string>("all");
  const [selectedEventType, setSelectedEventType] = useState<string>("all");
  const [selectedStatus, setSelectedStatus] = useState<string>("all");
  const [searchQuery, setSearchQuery] = useState<string>("");

  // Sub-tabs: "events" | "agent" | "correlations" | "assets"
  const [activeTab, setActiveTab] = useState<"overview" | "events" | "agent" | "correlations">("overview");

  // Agent Investigation State
  const [investigationQuery, setInvestigationQuery] = useState<string>(
    "Investigate security anomalies associated with CAM-NORTH-01 between 22:00 and 22:30."
  );
  const [investigationAsset, setInvestigationAsset] = useState<string>("CAM-NORTH-01");
  const [isInvestigating, setIsInvestigating] = useState(false);
  const [investigationResult, setInvestigationResult] = useState<CyberInvestigationResult | null>(null);
  const [investigationError, setInvestigationError] = useState<string | null>(null);

  // Event Detail Modal
  const [selectedEvent, setSelectedEvent] = useState<CyberEventItem | null>(null);

  // Evidence Preview Modal
  const [previewEvidence, setPreviewEvidence] = useState<{
    evidence_id: string;
    object_class?: string | null;
    timestamp: number;
    snapshot_url?: string | null;
    clip_url?: string | null;
    playback_url?: string | null;
  } | null>(null);

  // New Event Modal
  const [isNewEventOpen, setIsNewEventOpen] = useState(false);
  const [newEventAsset, setNewEventAsset] = useState("CAM-NORTH-01");
  const [newEventType, setNewEventType] = useState("CONFIGURATION_CHANGE");
  const [newEventSeverity, setNewEventSeverity] = useState<"LOW" | "MEDIUM" | "HIGH" | "CRITICAL">("HIGH");
  const [newEventDesc, setNewEventDesc] = useState("");
  const [newEventIP, setNewEventIP] = useState("192.168.10.88");
  const [isCreatingEvent, setIsCreatingEvent] = useState(false);
  const [createEventError, setCreateEventError] = useState<string | null>(null);

  // Action feedback
  const [seedingFeedback, setSeedingFeedback] = useState<string | null>(null);

  // Fetch all data
  const fetchData = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const [assetRes, eventRes, corrRes] = await Promise.all([
        listSecurityAssets(),
        listCyberEvents({
          asset_label: selectedAsset !== "all" ? selectedAsset : undefined,
          severity: selectedSeverity !== "all" ? selectedSeverity : undefined,
          event_type: selectedEventType !== "all" ? selectedEventType : undefined,
          status: selectedStatus !== "all" ? selectedStatus : undefined,
          search: searchQuery || undefined,
          limit: 100,
        }),
        correlateCyberPhysical({
          asset_label_or_id: selectedAsset !== "all" ? selectedAsset : undefined,
          time_window_seconds: 900.0,
        }),
      ]);

      setAssets(assetRes.assets || []);
      setEvents(eventRes.events || []);
      setCorrelations(corrRes.correlations || []);
    } catch (err: any) {
      setError(err.message || "Failed to load security operations data.");
    } finally {
      setLoading(false);
    }
  }, [selectedAsset, selectedSeverity, selectedEventType, selectedStatus, searchQuery]);

  useEffect(() => {
    fetchData();
  }, [fetchData]);

  // Seed demo data handler
  const handleSeedDemo = async () => {
    setSeedingFeedback("Seeding deterministic replay telemetry...");
    try {
      const res = await seedCyberDemoTelemetry();
      setSeedingFeedback(`Seeded ${res.total_demo_events} deterministic events (provenance: REPLAYED_TELEMETRY)`);
      await fetchData();
      setTimeout(() => setSeedingFeedback(null), 3000);
    } catch (err: any) {
      setSeedingFeedback(`Error: ${err.message}`);
    }
  };

  // Agent investigation handler
  const handleRunInvestigation = async (queryText?: string, targetAsset?: string) => {
    const q = queryText || investigationQuery;
    const a = targetAsset || investigationAsset;
    if (!q.trim()) return;

    setIsInvestigating(true);
    setInvestigationError(null);
    try {
      const res = await investigateCyber({
        query: q.trim(),
        asset_label: a || undefined,
        time_window_seconds: 900.0,
      });
      setInvestigationResult(res);
      setActiveTab("agent");
    } catch (err: any) {
      setInvestigationError(err.message || "Investigation failed.");
    } finally {
      setIsInvestigating(false);
    }
  };

  // Create event submit handler
  const handleCreateSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!newEventDesc.trim()) {
      setCreateEventError("Description is required.");
      return;
    }
    setIsCreatingEvent(true);
    setCreateEventError(null);
    try {
      await createCyberEvent({
        asset_label: newEventAsset,
        event_type: newEventType,
        severity: newEventSeverity,
        description: newEventDesc.trim(),
        source_ip: newEventIP.trim() || undefined,
        provenance: "MANUAL_ANALYST_EVENT",
        status: "NEW",
      });
      setIsNewEventOpen(false);
      setNewEventDesc("");
      await fetchData();
    } catch (err: any) {
      setCreateEventError(err.message || "Failed to create cyber event.");
    } finally {
      setIsCreatingEvent(false);
    }
  };

  const getSeverityBadge = (sev: string) => {
    switch (sev) {
      case "CRITICAL":
        return "bg-red-500/10 text-red-400 border-red-500/30";
      case "HIGH":
        return "bg-amber-500/10 text-amber-400 border-amber-500/30";
      case "MEDIUM":
        return "bg-yellow-500/10 text-yellow-300 border-yellow-500/30";
      default:
        return "bg-[#1D2128] text-[#A7AFBA] border-[#2A3038]";
    }
  };

  return (
    <div className="p-4 md:p-6 lg:p-8 space-y-6 max-w-[1750px] 2xl:max-w-[2150px] mx-auto min-h-full">
      {/* Top Header */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 border-b border-[#2A3038] pb-5">
        <div>
          <div className="flex items-center gap-3 flex-wrap">
            <h1 className="text-xl md:text-2xl font-bold tracking-tight text-[#F5F7FA]">
              Security Operations
            </h1>
            <span className="px-3 py-1 rounded-md text-xs font-mono font-semibold bg-amber-500/10 text-amber-400 border border-amber-500/30">
              PROTOTYPE / REPLAYED TELEMETRY
            </span>
          </div>
          <p className="text-xs md:text-sm text-[#A7AFBA] mt-1.5">
            Cyber telemetry and physical CCTV correlation prototype using replayed telemetry.
          </p>
        </div>

        <div className="flex items-center gap-3 flex-wrap">
          <button
            onClick={handleSeedDemo}
            className="flex items-center gap-2 px-3.5 py-2 bg-[#1D2128] hover:bg-[#2A3038] text-[#A7AFBA] hover:text-[#F5F7FA] border border-[#2A3038] rounded-lg text-xs font-medium transition-all cursor-pointer"
          >
            <svg className="w-3.5 h-3.5 text-amber-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
            </svg>
            Seed Demo Telemetry
          </button>

          <button
            onClick={() => setIsNewEventOpen(true)}
            className="flex items-center gap-2 px-3.5 py-2 bg-[#1D2128] hover:bg-[#2A3038] text-[#A7AFBA] hover:text-[#F5F7FA] border border-[#2A3038] rounded-lg text-xs font-medium transition-all cursor-pointer"
          >
            <span>+ Log Cyber Event</span>
          </button>

          <button
            onClick={() => handleRunInvestigation()}
            className="flex items-center gap-2 px-4 py-2 bg-[#19B89A] hover:bg-[#159e84] text-[#0F1115] rounded-lg text-xs font-semibold transition-all shadow-sm cursor-pointer"
          >
            <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
            </svg>
            Investigate with Sentinel
          </button>
        </div>
      </div>

      {/* Feedback banner */}
      {seedingFeedback && (
        <div className="p-3 bg-amber-500/10 border border-amber-500/30 rounded-xl text-amber-300 text-xs font-mono flex items-center justify-between">
          <span>{seedingFeedback}</span>
          <button onClick={() => setSeedingFeedback(null)} className="text-amber-400 hover:underline">✕</button>
        </div>
      )}

      {/* Workspace Navigation Tabs */}
      <div className="flex border-b border-[#2A3038] gap-6 text-sm">
        <button
          onClick={() => setActiveTab("overview")}
          className={`pb-3 text-xs md:text-sm transition-colors cursor-pointer border-b-2 flex items-center gap-2 ${
            activeTab === "overview"
              ? "border-[#19B89A] text-[#19B89A] font-semibold"
              : "border-transparent text-[#A7AFBA] hover:text-[#F5F7FA]"
          }`}
        >
          <span>Security Assets ({assets.length})</span>
        </button>

        <button
          onClick={() => setActiveTab("agent")}
          className={`pb-3 text-xs md:text-sm transition-colors cursor-pointer border-b-2 flex items-center gap-2 ${
            activeTab === "agent"
              ? "border-[#19B89A] text-[#19B89A] font-semibold"
              : "border-transparent text-[#A7AFBA] hover:text-[#F5F7FA]"
          }`}
        >
          <span>Agent Investigation</span>
        </button>

        <button
          onClick={() => setActiveTab("correlations")}
          className={`pb-3 text-xs md:text-sm transition-colors cursor-pointer border-b-2 flex items-center gap-2 ${
            activeTab === "correlations"
              ? "border-[#19B89A] text-[#19B89A] font-semibold"
              : "border-transparent text-[#A7AFBA] hover:text-[#F5F7FA]"
          }`}
        >
          <span>Correlations ({correlations.length})</span>
        </button>

        <button
          onClick={() => setActiveTab("events")}
          className={`pb-3 text-xs md:text-sm transition-colors cursor-pointer border-b-2 flex items-center gap-2 ${
            activeTab === "events"
              ? "border-[#19B89A] text-[#19B89A] font-semibold"
              : "border-transparent text-[#A7AFBA] hover:text-[#F5F7FA]"
          }`}
        >
          <span>Telemetry Log ({events.length})</span>
        </button>
      </div>

      {/* Tab 1: Security Assets Overview */}
      {activeTab === "overview" && (
        <div className="space-y-6">
          <div className="flex items-center justify-between">
            <div>
              <h2 className="text-lg font-bold text-[#F5F7FA]">Security assets</h2>
              <p className="text-xs text-[#A7AFBA] mt-0.5">Camera endpoints and sensory assets integrated with telemetry</p>
            </div>
            <span className="text-xs text-[#737C87]">
              Select an asset to focus investigation
            </span>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-5">
            {assets.map((asset) => (
              <div
                key={asset.asset_id}
                onClick={() => {
                  setSelectedAsset(asset.asset_label);
                  setInvestigationAsset(asset.asset_label);
                }}
                className={`p-5 rounded-xl border transition-all cursor-pointer flex flex-col justify-between space-y-4 ${
                  selectedAsset === asset.asset_label
                    ? "bg-[#171A20] border-[#19B89A] shadow-lg shadow-[#19B89A]/5"
                    : "bg-[#171A20] border-[#2A3038] hover:border-[#3B434D]"
                }`}
              >
                <div className="space-y-3">
                  <div className="flex items-start justify-between">
                    <div>
                      <h3 className="text-base font-bold text-[#F5F7FA] flex items-center gap-2">
                        {asset.asset_label}
                        {selectedAsset === asset.asset_label && (
                          <span className="w-2 h-2 rounded-full bg-[#19B89A]" />
                        )}
                      </h3>
                      <p className="text-xs text-[#A7AFBA] mt-0.5">
                        {asset.position_hint}
                      </p>
                    </div>
                    {asset.high_severity_cyber_events > 0 ? (
                      <span className="px-2 py-0.5 rounded text-xs font-semibold bg-amber-500/10 text-amber-400 border border-amber-500/30">
                        {asset.high_severity_cyber_events} alerts
                      </span>
                    ) : (
                      <span className="px-2 py-0.5 rounded text-xs font-medium bg-emerald-500/10 text-emerald-400 border border-emerald-500/30">
                        Healthy
                      </span>
                    )}
                  </div>

                  <div className="space-y-1.5 text-xs text-[#A7AFBA] pt-1">
                    <div className="flex items-center justify-between">
                      <span>Cyber events:</span>
                      <span className="font-semibold text-[#F5F7FA]">{asset.total_cyber_events}</span>
                    </div>
                    <div className="flex items-center justify-between">
                      <span>Physical incidents:</span>
                      <span className="font-semibold text-[#F5F7FA]">{asset.physical_incident_count}</span>
                    </div>
                    {asset.video_filename && (
                      <div className="flex items-center justify-between text-[11px] text-[#737C87] pt-1 border-t border-[#2A3038]">
                        <span>Footage:</span>
                        <span className="truncate max-w-[150px]">{asset.video_filename}</span>
                      </div>
                    )}
                  </div>
                </div>

                <div className="flex items-center gap-2 pt-2 border-t border-[#2A3038]">
                  <button
                    onClick={(e) => {
                      e.stopPropagation();
                      setInvestigationAsset(asset.asset_label);
                      setInvestigationQuery(`Investigate security anomalies associated with ${asset.asset_label} between 22:00 and 22:30.`);
                      handleRunInvestigation(`Investigate security anomalies associated with ${asset.asset_label} between 22:00 and 22:30.`, asset.asset_label);
                    }}
                    className="flex-1 px-3 py-2 bg-[#19B89A] hover:bg-[#159e84] text-[#0F1115] rounded-lg text-xs font-semibold transition-all text-center cursor-pointer"
                  >
                    Investigate
                  </button>

                  {asset.video_id && onInvestigateVideo && (
                    <button
                      onClick={(e) => {
                        e.stopPropagation();
                        onInvestigateVideo(asset.video_id!);
                      }}
                      className="px-3 py-2 bg-[#1D2128] hover:bg-[#2A3038] text-[#A7AFBA] hover:text-[#F5F7FA] border border-[#2A3038] rounded-lg text-xs font-medium transition-all cursor-pointer"
                    >
                      CCTV Footage
                    </button>
                  )}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Tab 2: Sentinel Agent Investigation */}
      {activeTab === "agent" && (
        <div className="space-y-6">
          <div className="p-5 rounded-xl bg-[#171A20] border border-[#2A3038] space-y-4">
            <div className="flex flex-col md:flex-row md:items-center justify-between gap-3">
              <div>
                <h2 className="text-base font-bold text-[#F5F7FA]">
                  Sentinel investigation agent
                </h2>
                <p className="text-xs text-[#A7AFBA] mt-0.5">
                  Controlled tool execution across cyber telemetry and physical CCTV intelligence.
                </p>
              </div>

              <div className="flex items-center gap-2">
                <span className="text-xs text-[#737C87]">Target Asset:</span>
                <select
                  value={investigationAsset}
                  onChange={(e) => setInvestigationAsset(e.target.value)}
                  className="bg-[#0F1115] border border-[#2A3038] rounded-lg px-3 py-1.5 text-xs text-[#F5F7FA] focus:outline-none focus:border-[#19B89A]"
                >
                  <option value="CAM-NORTH-01">CAM-NORTH-01 (North Gate)</option>
                  <option value="CAM-LOBBY-02">CAM-LOBBY-02 (Lobby)</option>
                  <option value="CAM-PERIM-03">CAM-PERIM-03 (Perimeter)</option>
                </select>
              </div>
            </div>

            {/* Quick Prompts */}
            <div className="flex items-center gap-2 flex-wrap text-xs">
              <span className="text-[#737C87]">Suggested:</span>
              {[
                "Investigate security anomalies associated with CAM-NORTH-01 between 22:00 and 22:30.",
                "Correlate configuration changes with physical incidents on CAM-NORTH-01.",
                "Analyze connection drops and stream integrity on CAM-LOBBY-02.",
              ].map((q, idx) => (
                <button
                  key={idx}
                  onClick={() => {
                    setInvestigationQuery(q);
                    handleRunInvestigation(q, investigationAsset);
                  }}
                  className="px-2.5 py-1 bg-[#1D2128] hover:bg-[#2A3038] text-[#A7AFBA] hover:text-[#F5F7FA] border border-[#2A3038] rounded-lg transition-colors text-xs cursor-pointer"
                >
                  &ldquo;{q.slice(0, 48)}...&rdquo;
                </button>
              ))}
            </div>

            {/* Inquiry Input Form */}
            <form
              onSubmit={(e) => {
                e.preventDefault();
                handleRunInvestigation();
              }}
              className="flex items-center gap-3 pt-2"
            >
              <input
                type="text"
                value={investigationQuery}
                onChange={(e) => setInvestigationQuery(e.target.value)}
                placeholder="Enter investigation prompt for Sentinel Agent..."
                className="flex-1 bg-[#0F1115] border border-[#2A3038] rounded-lg px-4 py-2.5 text-sm text-[#F5F7FA] focus:outline-none focus:border-[#19B89A]"
              />
              <button
                type="submit"
                disabled={isInvestigating}
                className="px-5 py-2.5 bg-[#19B89A] hover:bg-[#159e84] text-[#0F1115] font-semibold text-sm rounded-lg transition-all shadow-sm disabled:opacity-50 cursor-pointer flex items-center gap-2"
              >
                {isInvestigating ? (
                  <>
                    <div className="w-4 h-4 border-2 border-[#0F1115] border-t-transparent rounded-full animate-spin" />
                    <span>Investigating...</span>
                  </>
                ) : (
                  <span>Investigate with Sentinel</span>
                )}
              </button>
            </form>
          </div>

          {investigationError && (
            <div className="p-4 bg-red-500/10 border border-red-500/30 rounded-xl text-red-400 text-sm">
              {investigationError}
            </div>
          )}

          {/* Investigation Output */}
          {investigationResult && (
            <div className="space-y-6">
              {/* Agent Safe Execution Trace (Audit Checklist) */}
              <div className="p-5 rounded-xl bg-[#171A20] border border-[#2A3038] space-y-3">
                <div className="flex items-center justify-between border-b border-[#2A3038] pb-3">
                  <h3 className="text-sm font-bold text-[#F5F7FA] flex items-center gap-2">
                    <span className="w-2 h-2 rounded-full bg-emerald-400" />
                    Safe execution trace
                  </h3>
                  <span className="text-xs text-[#737C87]">
                    {investigationResult.actions_taken?.length || 0} controlled steps executed
                  </span>
                </div>

                <div className="grid grid-cols-1 md:grid-cols-2 gap-2 text-xs">
                  {investigationResult.actions_taken?.map((act, idx) => (
                    <div
                      key={idx}
                      className="p-2.5 rounded-lg bg-[#0F1115] border border-[#2A3038] flex items-start gap-2.5"
                    >
                      <span className="text-emerald-400 font-bold flex-none">✓ Step {act.step}</span>
                      <div className="space-y-0.5 min-w-0">
                        <div className="text-[#F5F7FA] font-medium">{act.action}</div>
                        <div className="text-[#737C87] text-[11px] truncate">{act.detail}</div>
                      </div>
                    </div>
                  ))}
                </div>
              </div>

              {/* Synthesized Findings */}
              <div className="p-6 rounded-xl bg-[#171A20] border border-[#2A3038] space-y-4">
                <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 border-b border-[#2A3038] pb-4">
                  <div>
                    <span className="text-xs text-[#19B89A] font-semibold">
                      Grounded assessment
                    </span>
                    <h3 className="text-lg font-bold text-[#F5F7FA]">
                      Findings for {investigationResult.security_asset}
                    </h3>
                  </div>

                  <div className="flex items-center gap-2.5 flex-wrap">
                    <span className="px-3 py-1 rounded text-xs font-semibold bg-[#1D2128] text-[#19B89A] border border-[#2A3038]">
                      {investigationResult.gemini_used ? "AI-assisted reasoning" : "Deterministic synthesis"}
                    </span>
                    <span className="px-3 py-1 rounded text-xs font-semibold bg-amber-500/10 text-amber-400 border border-amber-500/30">
                      {investigationResult.status}
                    </span>
                    <span className="px-3 py-1 rounded text-xs font-mono text-[#A7AFBA] bg-[#1D2128] border border-[#2A3038]">
                      Ceiling: {investigationResult.assessment_score}
                    </span>
                  </div>
                </div>

                {/* Narrative text block */}
                <div className="p-5 rounded-xl bg-[#0F1115] border border-[#2A3038] text-sm md:text-base font-sans text-[#F5F7FA] whitespace-pre-wrap leading-relaxed">
                  {investigationResult.findings}
                </div>

                {/* Supporting Physical Evidence Row */}
                {investigationResult.supporting_evidence && investigationResult.supporting_evidence.length > 0 && (
                  <div className="space-y-2 pt-2">
                    <h4 className="text-xs font-semibold text-[#A7AFBA]">
                      Preserved physical CCTV evidence ({investigationResult.supporting_evidence.length})
                    </h4>
                    <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 gap-3">
                      {investigationResult.supporting_evidence.map((ev, i) => (
                        <div
                          key={i}
                          onClick={() => setPreviewEvidence(ev)}
                          className="p-3 rounded-lg bg-[#0F1115] border border-[#2A3038] hover:border-[#19B89A]/50 transition-all cursor-pointer flex items-center justify-between"
                        >
                          <div>
                            <span className="text-emerald-400 font-mono text-xs">
                              {ev.evidence_id}
                            </span>
                            <p className="text-xs text-[#A7AFBA] mt-0.5">
                              {ev.object_class} @ {ev.timestamp?.toFixed(1)}s
                            </p>
                          </div>
                          <span className="text-xs text-[#19B89A] hover:underline">
                            View &rarr;
                          </span>
                        </div>
                      ))}
                    </div>
                  </div>
                )}

                {/* Governance Banner */}
                <div className="rounded-lg border border-amber-500/30 bg-amber-500/10 p-3.5 text-xs text-amber-300 space-y-1">
                  <div className="font-semibold flex items-center gap-2">
                    <span>⚠ Human review mandatory (Review ceiling &le; 0.65)</span>
                  </div>
                  <p className="text-xs leading-relaxed text-amber-300/80">
                    Sentinel correlation reflects observable asset identity and temporal proximity only. Attacker identity, intent, and legal culpability are fundamentally not inferred.
                  </p>
                </div>
              </div>
            </div>
          )}
        </div>
      )}
      {/* Tab 3: Correlations View */}
      {activeTab === "correlations" && (
        <div className="space-y-4">
          <div className="flex items-center justify-between">
            <div>
              <h2 className="text-lg font-bold text-[#F5F7FA]">Cyber-physical correlations</h2>
              <p className="text-xs text-[#A7AFBA] mt-0.5">Digital anomalies aligned with physical incident windows</p>
            </div>
            <span className="text-xs text-[#737C87]">
              Correlated hypotheses: {correlations.length}
            </span>
          </div>

          <div className="space-y-3">
            {correlations.map((corr) => (
              <div
                key={corr.correlation_id}
                className="p-5 rounded-xl bg-[#171A20] border border-[#2A3038] hover:border-[#3B434D] transition-all space-y-3"
              >
                <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 border-b border-[#2A3038] pb-3">
                  <div className="flex items-center gap-2.5">
                    <span className="px-2.5 py-0.5 rounded text-xs font-semibold bg-[#1D2128] text-[#19B89A] border border-[#2A3038]">
                      {corr.correlation_hypothesis}
                    </span>
                    <span className="text-sm font-semibold text-[#F5F7FA]">
                      Asset: {corr.security_asset}
                    </span>
                  </div>

                  <div className="flex items-center gap-2 text-xs">
                    <span className="px-2 py-0.5 rounded bg-amber-500/10 text-amber-400 border border-amber-500/30 font-medium">
                      {corr.status}
                    </span>
                    <span className="text-[#737C87] font-mono">
                      Score: {corr.confidence_score}
                    </span>
                  </div>
                </div>

                {/* Side-by-side Cyber & Physical mapping */}
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-xs">
                  <div className="p-3.5 rounded-lg bg-[#0F1115] border border-[#2A3038] space-y-1">
                    <div className="text-[#19B89A] font-semibold text-xs">
                      Digital cyber event
                    </div>
                    <div className="text-[#F5F7FA] font-medium">
                      {corr.cyber_event?.event_type} ({corr.cyber_event?.severity})
                    </div>
                    <p className="text-[#A7AFBA] text-xs">
                      {corr.cyber_event?.description}
                    </p>
                    <div className="text-[11px] text-[#737C87] pt-1">
                      Source IP: {corr.cyber_event?.source_ip || "N/A"} &bull; Rel TS: {corr.cyber_event?.timestamp_seconds?.toFixed(1) || "--"}s
                    </div>
                  </div>

                  <div className="p-3.5 rounded-lg bg-[#0F1115] border border-[#2A3038] space-y-1">
                    <div className="text-emerald-400 font-semibold text-xs">
                      Physical CCTV incident
                    </div>
                    <div className="text-[#F5F7FA] font-medium">
                      {corr.physical_incident?.category?.replace(/_/g, " ")}
                    </div>
                    <p className="text-[#A7AFBA] text-xs line-clamp-2">
                      {corr.physical_incident?.storyline}
                    </p>
                    <div className="text-[11px] text-[#737C87] pt-1">
                      Incident TS: {corr.physical_incident?.start_time?.toFixed(1)}s &bull; Delta: {corr.temporal_delta_seconds ? `${corr.temporal_delta_seconds.toFixed(1)}s` : "Concurrent"}
                    </div>
                  </div>
                </div>

                <div className="text-xs text-[#A7AFBA] bg-[#0F1115] p-3 rounded-lg border border-[#2A3038]">
                  <span className="text-[#737C87] font-medium">Correlation narrative: </span>
                  {corr.correlation_narrative}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Tab 4: All Cyber Events Table */}
      {activeTab === "events" && (
        <div className="space-y-4">
          {/* Filter Bar */}
          <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-5 gap-3 p-4 bg-[#171A20] border border-[#2A3038] rounded-xl text-xs">
            <div>
              <label className="text-[#A7AFBA] block mb-1">Security Asset</label>
              <select
                value={selectedAsset}
                onChange={(e) => setSelectedAsset(e.target.value)}
                className="w-full bg-[#0F1115] border border-[#2A3038] rounded-lg p-2 text-[#F5F7FA]"
              >
                <option value="all">All Assets</option>
                <option value="CAM-NORTH-01">CAM-NORTH-01</option>
                <option value="CAM-LOBBY-02">CAM-LOBBY-02</option>
                <option value="CAM-PERIM-03">CAM-PERIM-03</option>
              </select>
            </div>

            <div>
              <label className="text-[#A7AFBA] block mb-1">Severity</label>
              <select
                value={selectedSeverity}
                onChange={(e) => setSelectedSeverity(e.target.value)}
                className="w-full bg-[#0F1115] border border-[#2A3038] rounded-lg p-2 text-[#F5F7FA]"
              >
                <option value="all">All Severities</option>
                <option value="CRITICAL">CRITICAL</option>
                <option value="HIGH">HIGH</option>
                <option value="MEDIUM">MEDIUM</option>
                <option value="LOW">LOW</option>
              </select>
            </div>

            <div>
              <label className="text-[#A7AFBA] block mb-1">Event Type</label>
              <select
                value={selectedEventType}
                onChange={(e) => setSelectedEventType(e.target.value)}
                className="w-full bg-[#0F1115] border border-[#2A3038] rounded-lg p-2 text-[#F5F7FA]"
              >
                <option value="all">All Event Types</option>
                <option value="CONFIGURATION_CHANGE">CONFIGURATION_CHANGE</option>
                <option value="AUTHENTICATION_ANOMALY">AUTHENTICATION_ANOMALY</option>
                <option value="SECURITY_POLICY_VIOLATION">SECURITY_POLICY_VIOLATION</option>
                <option value="CONNECTION_ANOMALY">CONNECTION_ANOMALY</option>
                <option value="STREAM_INTEGRITY_ANOMALY">STREAM_INTEGRITY_ANOMALY</option>
              </select>
            </div>

            <div>
              <label className="text-[#A7AFBA] block mb-1">Status</label>
              <select
                value={selectedStatus}
                onChange={(e) => setSelectedStatus(e.target.value)}
                className="w-full bg-[#0F1115] border border-[#2A3038] rounded-lg p-2 text-[#F5F7FA]"
              >
                <option value="all">All Statuses</option>
                <option value="NEW">NEW</option>
                <option value="INVESTIGATING">INVESTIGATING</option>
                <option value="CONFIRMED">CONFIRMED</option>
                <option value="DISMISSED">DISMISSED</option>
              </select>
            </div>

            <div>
              <label className="text-[#A7AFBA] block mb-1">Search Keywords</label>
              <input
                type="text"
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                placeholder="Search IP, description..."
                className="w-full bg-[#0F1115] border border-[#2A3038] rounded-lg p-2 text-[#F5F7FA]"
              />
            </div>
          </div>

          {/* Table */}
          <div className="overflow-x-auto bg-[#171A20] border border-[#2A3038] rounded-xl">
            <table className="w-full text-left text-xs text-[#A7AFBA]">
              <thead className="bg-[#1D2128] text-[#F5F7FA] border-b border-[#2A3038] text-[11px] font-semibold">
                <tr>
                  <th className="py-3 px-4">Event Type</th>
                  <th className="py-3 px-4">Severity</th>
                  <th className="py-3 px-4">Security Asset</th>
                  <th className="py-3 px-4">Provenance</th>
                  <th className="py-3 px-4">Source IP</th>
                  <th className="py-3 px-4">Status</th>
                  <th className="py-3 px-4">Description</th>
                  <th className="py-3 px-4 text-right">Actions</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-[#2A3038]">
                {events.map((ev) => (
                  <tr key={ev.id} className="hover:bg-[#1D2128]/50 transition-colors">
                    <td className="py-3 px-4 font-semibold text-[#F5F7FA]">
                      {ev.event_type}
                    </td>
                    <td className="py-3 px-4">
                      <span className={`px-2 py-0.5 rounded text-[10px] font-semibold border uppercase ${getSeverityBadge(ev.severity)}`}>
                        {ev.severity}
                      </span>
                    </td>
                    <td className="py-3 px-4 text-[#19B89A] font-medium">
                      {ev.asset_label}
                    </td>
                    <td className="py-3 px-4">
                      <span className="px-2 py-0.5 rounded text-[10px] bg-amber-500/10 text-amber-400 border border-amber-500/30 uppercase font-semibold">
                        {ev.provenance}
                      </span>
                    </td>
                    <td className="py-3 px-4 font-mono text-xs text-[#A7AFBA]">
                      {ev.source_ip || "--"}
                    </td>
                    <td className="py-3 px-4">
                      <span className="text-[#A7AFBA]">{ev.status}</span>
                    </td>
                    <td className="py-3 px-4 max-w-xs truncate text-[#A7AFBA]" title={ev.description}>
                      {ev.description}
                    </td>
                    <td className="py-3 px-4 text-right space-x-2">
                      <button
                        onClick={() => setSelectedEvent(ev)}
                        className="px-2.5 py-1 bg-[#1D2128] hover:bg-[#2A3038] text-[#F5F7FA] border border-[#2A3038] rounded-md transition-all text-xs cursor-pointer"
                      >
                        Details
                      </button>
                      <button
                        onClick={() => {
                          const q = `Investigate cybersecurity event '${ev.event_type}' on ${ev.asset_label}: ${ev.description}`;
                          setInvestigationAsset(ev.asset_label);
                          setInvestigationQuery(q);
                          handleRunInvestigation(q, ev.asset_label);
                        }}
                        className="px-2.5 py-1 bg-[#19B89A]/15 hover:bg-[#19B89A] text-[#19B89A] hover:text-[#0F1115] border border-[#19B89A]/30 rounded-md transition-all text-xs font-semibold cursor-pointer"
                      >
                        Investigate
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* Event Details Modal */}
      {selectedEvent && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/80 backdrop-blur-sm p-4">
          <div className="bg-[#171A20] border border-[#2A3038] rounded-xl max-w-lg w-full p-6 space-y-4 shadow-2xl">
            <div className="flex items-center justify-between border-b border-[#2A3038] pb-3">
              <h3 className="text-base font-bold text-[#F5F7FA]">
                Cybersecurity event details
              </h3>
              <button
                onClick={() => setSelectedEvent(null)}
                className="text-[#737C87] hover:text-[#F5F7FA] cursor-pointer"
              >
                ✕
              </button>
            </div>

            <div className="space-y-3 text-xs">
              <div className="flex justify-between py-1.5 border-b border-[#2A3038]">
                <span className="text-[#737C87]">Event ID:</span>
                <span className="text-[#F5F7FA] font-mono">{selectedEvent.id}</span>
              </div>
              <div className="flex justify-between py-1.5 border-b border-[#2A3038]">
                <span className="text-[#737C87]">Type:</span>
                <span className="text-[#19B89A] font-medium">{selectedEvent.event_type}</span>
              </div>
              <div className="flex justify-between py-1.5 border-b border-[#2A3038]">
                <span className="text-[#737C87]">Severity:</span>
                <span className={`px-2 py-0.5 rounded text-[10px] font-semibold border uppercase ${getSeverityBadge(selectedEvent.severity)}`}>
                  {selectedEvent.severity}
                </span>
              </div>
              <div className="flex justify-between py-1.5 border-b border-[#2A3038]">
                <span className="text-[#737C87]">Security Asset:</span>
                <span className="text-[#F5F7FA] font-medium">{selectedEvent.asset_label}</span>
              </div>
              <div className="flex justify-between py-1.5 border-b border-[#2A3038]">
                <span className="text-[#737C87]">Provenance:</span>
                <span className="text-amber-400 font-mono text-[11px]">{selectedEvent.provenance}</span>
              </div>
              <div className="flex justify-between py-1.5 border-b border-[#2A3038]">
                <span className="text-[#737C87]">Source IP &amp; Port:</span>
                <span className="text-[#A7AFBA] font-mono">{selectedEvent.source_ip || "--"} : {selectedEvent.destination_port || "--"}</span>
              </div>
              <div className="py-2 border-b border-[#2A3038] space-y-1">
                <span className="text-[#737C87] block">Description:</span>
                <p className="text-[#F5F7FA] leading-relaxed">{selectedEvent.description}</p>
              </div>

              {selectedEvent.structured_metadata && Object.keys(selectedEvent.structured_metadata).length > 0 && (
                <div className="space-y-1">
                  <span className="text-[#737C87] block">Structured Metadata:</span>
                  <pre className="p-3 bg-[#0F1115] rounded-lg border border-[#2A3038] text-[11px] overflow-x-auto text-[#A7AFBA] font-mono">
                    {JSON.stringify(selectedEvent.structured_metadata, null, 2)}
                  </pre>
                </div>
              )}
            </div>

            <div className="flex justify-between items-center pt-2">
              <button
                onClick={() => {
                  const asset = selectedEvent.asset_label;
                  const query = `Investigate cybersecurity event '${selectedEvent.event_type}' on ${asset}: ${selectedEvent.description}`;
                  setSelectedEvent(null);
                  setInvestigationAsset(asset);
                  setInvestigationQuery(query);
                  handleRunInvestigation(query, asset);
                }}
                className="px-3.5 py-2 bg-[#19B89A] hover:bg-[#159e84] text-[#0F1115] rounded-lg text-xs font-semibold cursor-pointer flex items-center gap-1.5"
              >
                Investigate with Sentinel
              </button>
              <button
                onClick={() => setSelectedEvent(null)}
                className="px-4 py-2 bg-[#1D2128] hover:bg-[#2A3038] text-[#F5F7FA] border border-[#2A3038] rounded-lg text-xs font-medium cursor-pointer"
              >
                Close
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Evidence Preview Modal */}
      {previewEvidence && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/85 backdrop-blur-sm p-4">
          <div className="bg-[#171A20] border border-[#2A3038] rounded-xl max-w-2xl w-full p-6 space-y-4 shadow-2xl">
            <div className="flex items-center justify-between border-b border-[#2A3038] pb-3">
              <div>
                <h3 className="text-base font-bold text-[#F5F7FA]">
                  Preserved Physical CCTV Evidence
                </h3>
                <span className="text-xs font-mono text-[#19B89A]">
                  ID: {previewEvidence.evidence_id} &bull; Timestamp: {previewEvidence.timestamp?.toFixed(1)}s
                </span>
              </div>
              <button
                onClick={() => setPreviewEvidence(null)}
                className="text-[#737C87] hover:text-[#F5F7FA] cursor-pointer"
              >
                ✕
              </button>
            </div>

            <div className="space-y-3">
              {(previewEvidence.playback_url || previewEvidence.clip_url) ? (
                <div className="rounded-lg overflow-hidden bg-black aspect-video flex items-center justify-center border border-[#2A3038]">
                  <video
                    key={previewEvidence.evidence_id}
                    controls
                    autoPlay
                    playsInline
                    className="w-full h-full object-contain"
                  >
                    <source src={(previewEvidence.playback_url || previewEvidence.clip_url) ?? undefined} type="video/mp4" />
                    {previewEvidence.clip_url && previewEvidence.playback_url && (
                      <source src={previewEvidence.clip_url} type="video/mp4" />
                    )}
                    Your browser does not support HTML5 video playback.
                  </video>
                </div>
              ) : previewEvidence.snapshot_url ? (
                <div className="rounded-lg overflow-hidden bg-black flex items-center justify-center max-h-[400px] border border-[#2A3038]">
                  <img
                    src={previewEvidence.snapshot_url}
                    alt="Evidence Snapshot"
                    className="max-h-[400px] w-auto object-contain"
                  />
                </div>
              ) : (
                <div className="p-8 text-center text-[#737C87] text-xs">
                  Evidence media preview unavailable
                </div>
              )}
            </div>

            <div className="flex justify-end pt-2">
              <button
                onClick={() => setPreviewEvidence(null)}
                className="px-4 py-2 bg-[#1D2128] hover:bg-[#2A3038] text-[#F5F7FA] border border-[#2A3038] rounded-lg text-xs font-medium cursor-pointer"
              >
                Close Preview
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Log Cyber Event Modal */}
      {isNewEventOpen && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/80 backdrop-blur-sm p-4">
          <div className="bg-[#171A20] border border-[#2A3038] rounded-xl max-w-md w-full p-6 space-y-4 shadow-2xl">
            <div className="flex items-center justify-between border-b border-[#2A3038] pb-3">
              <h3 className="text-base font-bold text-[#F5F7FA]">
                Log cybersecurity telemetry event
              </h3>
              <button
                onClick={() => setIsNewEventOpen(false)}
                className="text-[#737C87] hover:text-[#F5F7FA] cursor-pointer"
              >
                ✕
              </button>
            </div>

            {createEventError && (
              <div className="p-3 bg-red-500/10 border border-red-500/30 rounded-lg text-red-400 text-xs">
                {createEventError}
              </div>
            )}

            <form onSubmit={handleCreateSubmit} className="space-y-3.5 text-xs">
              <div>
                <label className="text-[#A7AFBA] block mb-1">Target Security Asset *</label>
                <select
                  value={newEventAsset}
                  onChange={(e) => setNewEventAsset(e.target.value)}
                  className="w-full bg-[#0F1115] border border-[#2A3038] rounded-lg p-2 text-[#F5F7FA]"
                >
                  <option value="CAM-NORTH-01">CAM-NORTH-01 (North Gate)</option>
                  <option value="CAM-LOBBY-02">CAM-LOBBY-02 (Lobby)</option>
                  <option value="CAM-PERIM-03">CAM-PERIM-03 (Perimeter)</option>
                </select>
              </div>

              <div>
                <label className="text-[#A7AFBA] block mb-1">Event Type *</label>
                <select
                  value={newEventType}
                  onChange={(e) => setNewEventType(e.target.value)}
                  className="w-full bg-[#0F1115] border border-[#2A3038] rounded-lg p-2 text-[#F5F7FA]"
                >
                  <option value="CONFIGURATION_CHANGE">CONFIGURATION_CHANGE</option>
                  <option value="AUTHENTICATION_ANOMALY">AUTHENTICATION_ANOMALY</option>
                  <option value="SECURITY_POLICY_VIOLATION">SECURITY_POLICY_VIOLATION</option>
                  <option value="CONNECTION_ANOMALY">CONNECTION_ANOMALY</option>
                  <option value="STREAM_INTEGRITY_ANOMALY">STREAM_INTEGRITY_ANOMALY</option>
                  <option value="DIGITAL_INTEGRITY_ANOMALY">DIGITAL_INTEGRITY_ANOMALY</option>
                </select>
              </div>

              <div>
                <label className="text-[#A7AFBA] block mb-1">Severity *</label>
                <select
                  value={newEventSeverity}
                  onChange={(e) => setNewEventSeverity(e.target.value as any)}
                  className="w-full bg-[#0F1115] border border-[#2A3038] rounded-lg p-2 text-[#F5F7FA]"
                >
                  <option value="LOW">LOW</option>
                  <option value="MEDIUM">MEDIUM</option>
                  <option value="HIGH">HIGH</option>
                  <option value="CRITICAL">CRITICAL</option>
                </select>
              </div>

              <div>
                <label className="text-[#A7AFBA] block mb-1">Source IP Address</label>
                <input
                  type="text"
                  value={newEventIP}
                  onChange={(e) => setNewEventIP(e.target.value)}
                  className="w-full bg-[#0F1115] border border-[#2A3038] rounded-lg p-2 text-[#F5F7FA] font-mono text-xs"
                />
              </div>

              <div>
                <label className="text-[#A7AFBA] block mb-1">Telemetry Description *</label>
                <textarea
                  rows={3}
                  required
                  value={newEventDesc}
                  onChange={(e) => setNewEventDesc(e.target.value)}
                  placeholder="Describe digital observation or policy trigger..."
                  className="w-full bg-[#0F1115] border border-[#2A3038] rounded-lg p-2 text-[#F5F7FA] text-xs"
                />
              </div>

              <div className="flex justify-end gap-3 pt-2">
                <button
                  type="button"
                  onClick={() => setIsNewEventOpen(false)}
                  className="px-4 py-2 bg-[#1D2128] hover:bg-[#2A3038] text-[#A7AFBA] hover:text-[#F5F7FA] border border-[#2A3038] rounded-lg font-medium cursor-pointer"
                >
                  Cancel
                </button>
                <button
                  type="submit"
                  disabled={isCreatingEvent}
                  className="px-4 py-2 bg-[#19B89A] hover:bg-[#159e84] text-[#0F1115] font-semibold rounded-lg cursor-pointer"
                >
                  {isCreatingEvent ? "Saving..." : "Log Event"}
                </button>
              </div>
            </form>
          </div>
        </div>
      )}
    </div>
  );
}
