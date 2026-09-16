"use client";

import React, { useState, useEffect, useCallback } from "react";
import {
  UnifiedTimelineEntry,
  InvestigationResult,
  TrackInvestigationResult,
  EvidenceBundle,
  investigationSearch,
  investigationStructuredQuery,
  getInvestigationTimeline,
  investigationTrack,
  createBundle,
  listBundles,
  Phase17InvestigationQueryRequest,
} from "@/lib/api";

interface ForensicSearchPanelProps {
  videoId: string;
  videoDuration?: number;
  onSeek?: (timestamp: number, eventId?: string) => void;
}

export function ForensicSearchPanel({
  videoId,
  videoDuration,
  onSeek,
}: ForensicSearchPanelProps) {
  // Mode selection: "search" | "structured" | "timeline" | "bundles" | "track"
  const [activeSubTab, setActiveSubTab] = useState<
    "search" | "structured" | "timeline" | "bundles" | "track"
  >("search");

  // Search mode state
  const [searchQuery, setSearchQuery] = useState<string>("");
  const [isSearching, setIsSearching] = useState<boolean>(false);
  const [searchResult, setSearchResult] = useState<InvestigationResult | null>(null);
  const [searchError, setSearchError] = useState<string | null>(null);

  // Structured query state
  const [structuredCategory, setStructuredCategory] = useState<string>("all");
  const [structuredDecision, setStructuredDecision] = useState<string>("all");
  const [minScore, setMinScore] = useState<number>(0);
  const [evidenceOnly, setEvidenceOnly] = useState<boolean>(false);
  const [reviewOnly, setReviewOnly] = useState<boolean>(false);
  const [timeStart, setTimeStart] = useState<string>("");
  const [timeEnd, setTimeEnd] = useState<string>("");

  // Timeline state
  const [timelineEntries, setTimelineEntries] = useState<UnifiedTimelineEntry[]>([]);
  const [timelineLoading, setTimelineLoading] = useState<boolean>(false);
  const [timelineLayerFilter, setTimelineLayerFilter] = useState<string>("all");
  const [includeRejectedTimeline, setIncludeRejectedTimeline] = useState<boolean>(false);
  const [timelineCounts, setTimelineCounts] = useState<{
    detection_events: number;
    security_events: number;
    correlated_incidents: number;
    evidence: number;
  }>({ detection_events: 0, security_events: 0, correlated_incidents: 0, evidence: 0 });

  // Track investigation state
  const [selectedTrackId, setSelectedTrackId] = useState<string>("");
  const [trackResult, setTrackResult] = useState<TrackInvestigationResult | null>(null);
  const [isTrackLoading, setIsTrackLoading] = useState<boolean>(false);
  const [trackError, setTrackError] = useState<string | null>(null);

  // Evidence bundles state
  const [bundles, setBundles] = useState<EvidenceBundle[]>([]);
  const [isBundlesLoading, setIsBundlesLoading] = useState<boolean>(false);
  const [bundleName, setBundleName] = useState<string>("");
  const [bundleNotes, setBundleNotes] = useState<string>("");
  const [isCreatingBundle, setIsCreatingBundle] = useState<boolean>(false);
  const [bundleMessage, setBundleMessage] = useState<string | null>(null);

  // Quick prompt suggestions
  const QUICK_QUERIES = [
    "What vehicles were detected?",
    "Find incidents around entrance gate",
    "Show events with confirmed evidence",
    "Identify review-required security events",
    "Summarize high-severity activity",
  ];

  // Execute Natural Language Search
  const handleSearch = async (queryText?: string) => {
    const q = queryText !== undefined ? queryText : searchQuery;
    if (!q.trim() || !videoId) return;

    setIsSearching(true);
    setSearchError(null);
    try {
      const res = await investigationSearch(videoId, q, videoDuration);
      setSearchResult(res);
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : "Search failed";
      setSearchError(msg);
    } finally {
      setIsSearching(false);
    }
  };

  // Execute Structured Query
  const handleStructuredQuery = async () => {
    if (!videoId) return;
    setIsSearching(true);
    setSearchError(null);
    try {
      const req: Phase17InvestigationQueryRequest = {
        result_limit: 50,
      };
      if (structuredCategory !== "all") {
        req.incident_categories = [structuredCategory];
      }
      if (structuredDecision !== "all") {
        req.validation_decisions = [structuredDecision];
      }
      if (minScore > 0) {
        req.min_assessment_score = minScore;
      }
      if (evidenceOnly) {
        req.evidence_required = true;
      }
      if (reviewOnly) {
        req.review_required_only = true;
      }
      if (timeStart) {
        req.time_start = parseFloat(timeStart);
      }
      if (timeEnd) {
        req.time_end = parseFloat(timeEnd);
      }

      const res = await investigationStructuredQuery(videoId, req);
      setSearchResult(res);
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : "Query failed";
      setSearchError(msg);
    } finally {
      setIsSearching(false);
    }
  };

  // Fetch Unified Forensic Timeline
  const fetchTimeline = useCallback(async () => {
    if (!videoId) return;
    setTimelineLoading(true);
    try {
      const res = await getInvestigationTimeline(
        videoId,
        undefined,
        undefined,
        includeRejectedTimeline
      );
      setTimelineEntries(res.timeline || []);
      if (res.layer_counts) {
        setTimelineCounts(res.layer_counts);
      }
    } catch {
      // ignore
    } finally {
      setTimelineLoading(false);
    }
  }, [videoId, includeRejectedTimeline]);

  // Fetch Track Details
  const handleInvestigateTrack = async (tId?: string) => {
    const id = tId || selectedTrackId;
    if (!id.trim() || !videoId) return;
    setSelectedTrackId(id);
    setActiveSubTab("track");
    setIsTrackLoading(true);
    setTrackError(null);
    try {
      const res = await investigationTrack(videoId, id.trim());
      setTrackResult(res);
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : "Track investigation failed";
      setTrackError(msg);
    } finally {
      setIsTrackLoading(false);
    }
  };

  // Fetch Bundles
  const fetchBundles = useCallback(async () => {
    if (!videoId) return;
    setIsBundlesLoading(true);
    try {
      const res = await listBundles(videoId);
      setBundles(res.bundles || []);
    } catch {
      // ignore
    } finally {
      setIsBundlesLoading(false);
    }
  }, [videoId]);

  // Create Bundle from search result
  const handleCreateBundleFromSearch = async () => {
    if (!videoId || !bundleName.trim()) return;
    setIsCreatingBundle(true);
    setBundleMessage(null);
    try {
      const incidentIds = (searchResult?.matched_incidents || []).map(
        (i) => i.incident_id
      );
      const res = await createBundle(videoId, {
        bundle_name: bundleName.trim(),
        notes: bundleNotes.trim() || undefined,
        selected_incident_ids: incidentIds,
        storyline_text: searchResult?.grounded_answer || undefined,
      });
      setBundleMessage(`Bundle created: ${res.bundle.bundle_id}`);
      setBundleName("");
      setBundleNotes("");
      fetchBundles();
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : "Bundle creation failed";
      setBundleMessage(`Error: ${msg}`);
    } finally {
      setIsCreatingBundle(false);
    }
  };

  // Load timeline and bundles on mount
  useEffect(() => {
    if (activeSubTab === "timeline") {
      fetchTimeline();
    } else if (activeSubTab === "bundles") {
      fetchBundles();
    }
  }, [activeSubTab, fetchTimeline, fetchBundles]);

  // Helpers for badge styling
  const getDecisionBadge = (decision?: string) => {
    switch (decision?.toUpperCase()) {
      case "CONFIRMED":
        return "bg-emerald-500/20 text-emerald-300 border-emerald-500/40";
      case "PROBABLE":
        return "bg-blue-500/20 text-blue-300 border-blue-500/40";
      case "REVIEW_REQUIRED":
        return "bg-amber-500/20 text-amber-300 border-amber-500/40";
      case "REJECTED":
        return "bg-rose-500/20 text-rose-300 border-rose-500/40";
      default:
        return "bg-zinc-800 text-zinc-300 border-zinc-700";
    }
  };

  const getReliabilityBadge = (rel?: string) => {
    switch (rel?.toUpperCase()) {
      case "HIGH":
        return "bg-emerald-950/60 text-emerald-400 border-emerald-700/50";
      case "MEDIUM":
        return "bg-amber-950/60 text-amber-400 border-amber-700/50";
      case "LOW":
        return "bg-rose-950/60 text-rose-400 border-rose-700/50";
      default:
        return "bg-zinc-900 text-zinc-400 border-zinc-800";
    }
  };

  const getLayerColor = (layer: string) => {
    switch (layer) {
      case "correlated_incident":
        return "border-cyan-500/50 bg-cyan-950/20 text-cyan-300";
      case "security_event":
        return "border-indigo-500/50 bg-indigo-950/20 text-indigo-300";
      case "evidence":
        return "border-amber-500/50 bg-amber-950/20 text-amber-300";
      case "detection_event":
      default:
        return "border-zinc-700 bg-zinc-900/40 text-zinc-300";
    }
  };

  const filteredTimeline = timelineEntries.filter((entry) => {
    if (timelineLayerFilter === "all") return true;
    return entry.layer === timelineLayerFilter;
  });

  return (
    <div className="space-y-6">
      {/* Header Banner */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 bg-zinc-950 border border-zinc-800 p-5 rounded-xl">
        <div>
          <h3 className="text-base font-bold text-zinc-100 flex items-center gap-2">
            <span className="w-2.5 h-2.5 rounded-full bg-cyan-400 animate-pulse" />
            FORENSIC SEARCH &amp; INVESTIGATION WORKBENCH
            <span className="text-[10px] font-mono uppercase bg-cyan-500/20 text-cyan-300 border border-cyan-500/40 px-2 py-0.5 rounded">
              AI Query Engine
            </span>
          </h3>
          <p className="text-xs text-zinc-400 mt-1">
            Natural-language queries, structured multi-signal filtering, unified forensic timeline, track lifecycle reconstruction, and evidence bundling.
          </p>
        </div>

        {/* Sub-Navigation Tabs */}
        <div className="flex items-center gap-1 bg-zinc-900 p-1 rounded-lg border border-zinc-800 text-xs font-mono">
          <button
            onClick={() => setActiveSubTab("search")}
            className={`px-3 py-1.5 rounded font-semibold transition-colors flex items-center gap-1.5 ${
              activeSubTab === "search"
                ? "bg-cyan-600 text-white"
                : "text-zinc-400 hover:text-zinc-200"
            }`}
          >
            <span>🔍 Query Search</span>
          </button>

          <button
            onClick={() => setActiveSubTab("structured")}
            className={`px-3 py-1.5 rounded font-semibold transition-colors flex items-center gap-1.5 ${
              activeSubTab === "structured"
                ? "bg-indigo-600 text-white"
                : "text-zinc-400 hover:text-zinc-200"
            }`}
          >
            <span>⚙️ Filters</span>
          </button>

          <button
            onClick={() => setActiveSubTab("timeline")}
            className={`px-3 py-1.5 rounded font-semibold transition-colors flex items-center gap-1.5 ${
              activeSubTab === "timeline"
                ? "bg-emerald-600 text-white"
                : "text-zinc-400 hover:text-zinc-200"
            }`}
          >
            <span>⏱️ Unified Timeline</span>
          </button>

          <button
            onClick={() => setActiveSubTab("track")}
            className={`px-3 py-1.5 rounded font-semibold transition-colors flex items-center gap-1.5 ${
              activeSubTab === "track"
                ? "bg-purple-600 text-white"
                : "text-zinc-400 hover:text-zinc-200"
            }`}
          >
            <span>🎯 Track Deep Dive</span>
          </button>

          <button
            onClick={() => setActiveSubTab("bundles")}
            className={`px-3 py-1.5 rounded font-semibold transition-colors flex items-center gap-1.5 ${
              activeSubTab === "bundles"
                ? "bg-amber-600 text-white"
                : "text-zinc-400 hover:text-zinc-200"
            }`}
          >
            <span>📦 Bundles</span>
          </button>
        </div>
      </div>

      {/* ------------------------------------------------------------- */}
      {/* SUBTAB: NATURAL LANGUAGE SEARCH */}
      {/* ------------------------------------------------------------- */}
      {activeSubTab === "search" && (
        <div className="space-y-4">
          {/* Query Bar */}
          <div className="bg-zinc-900 border border-zinc-800 p-4 rounded-xl space-y-3">
            <div className="flex gap-2">
              <input
                type="text"
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                onKeyDown={(e) => e.key === "Enter" && handleSearch()}
                placeholder="Ask Sentinel (e.g. 'Show people entering the gate between 0s and 30s' or 'Find confirmed incidents with evidence')..."
                className="flex-1 bg-zinc-950 border border-zinc-700 text-zinc-100 text-sm rounded-lg px-4 py-2.5 focus:outline-none focus:border-cyan-500 font-mono"
              />
              <button
                onClick={() => handleSearch()}
                disabled={isSearching || !searchQuery.trim()}
                className="bg-cyan-600 hover:bg-cyan-500 disabled:opacity-50 text-white text-xs font-semibold px-5 py-2.5 rounded-lg transition-colors font-mono flex items-center gap-2"
              >
                {isSearching ? (
                  <>
                    <span className="h-2 w-2 rounded-full bg-white animate-ping" />
                    Searching...
                  </>
                ) : (
                  <>🔍 Run Investigation</>
                )}
              </button>
            </div>

            {/* Quick Prompts */}
            <div className="flex flex-wrap items-center gap-2 pt-1">
              <span className="text-[11px] font-mono text-zinc-500">Quick queries:</span>
              {QUICK_QUERIES.map((q, idx) => (
                <button
                  key={idx}
                  onClick={() => {
                    setSearchQuery(q);
                    handleSearch(q);
                  }}
                  className="text-[11px] font-mono bg-zinc-800/80 hover:bg-zinc-700 text-cyan-300 px-2.5 py-1 rounded border border-zinc-700 transition-colors"
                >
                  {q}
                </button>
              ))}
            </div>
          </div>

          {searchError && (
            <div className="p-3 bg-red-950/40 border border-red-500/40 text-red-300 rounded-lg text-xs font-mono">
              Investigation Error: {searchError}
            </div>
          )}

          {/* Search Result Overview */}
          {searchResult && (
            <div className="space-y-4">
              {/* Grounded Synthesis Card */}
              <div className="bg-zinc-950 border border-cyan-500/40 rounded-xl p-5 space-y-3">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <span className="text-xs font-mono font-bold uppercase text-cyan-400 bg-cyan-950/80 border border-cyan-700/50 px-2 py-0.5 rounded">
                      Grounded Answer
                    </span>
                    <span className="text-xs text-zinc-400 font-mono">
                      Matches: {searchResult.total_results} total items
                    </span>
                  </div>

                  {searchResult.diagnostics && (
                    <div className="flex items-center gap-3 text-[11px] font-mono text-zinc-500">
                      <span>⚡ {searchResult.diagnostics.total_time_ms} ms</span>
                      <span>Incidents: {searchResult.diagnostics.incidents_found}</span>
                      <span>Events: {searchResult.diagnostics.events_found}</span>
                      <span>Tracks: {searchResult.diagnostics.tracks_found}</span>
                      <span>Evidence: {searchResult.diagnostics.evidence_found}</span>
                    </div>
                  )}
                </div>

                <p className="text-sm text-zinc-200 leading-relaxed font-sans">
                  {searchResult.grounded_answer || "No synthesis text available."}
                </p>

                {searchResult.interpretation && (
                  <div className="text-xs text-zinc-400 font-mono bg-zinc-900/60 p-2.5 rounded border border-zinc-800">
                    <span className="text-zinc-500">Query Interpretation:</span> {searchResult.interpretation}
                  </div>
                )}
              </div>

              {/* Matched Incidents List */}
              {searchResult.matched_incidents && searchResult.matched_incidents.length > 0 && (
                <div className="space-y-3">
                  <h4 className="text-xs font-mono uppercase text-zinc-400 tracking-wider">
                    Matched Correlated Incidents ({searchResult.matched_incidents.length})
                  </h4>
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                    {searchResult.matched_incidents.map((inc) => (
                      <div
                        key={inc.incident_id}
                        className="bg-zinc-900/80 border border-zinc-800 hover:border-zinc-700 p-4 rounded-xl space-y-3 transition-colors"
                      >
                        <div className="flex items-start justify-between gap-2">
                          <div>
                            <span className="text-xs font-bold text-zinc-100 uppercase tracking-wide">
                              {inc.incident_category}
                              {inc.incident_subcategory ? ` : ${inc.incident_subcategory}` : ""}
                            </span>
                            <div className="text-[11px] font-mono text-zinc-400 mt-0.5">
                              {inc.start_time.toFixed(1)}s &rarr; {inc.end_time.toFixed(1)}s ({inc.duration.toFixed(1)}s)
                            </div>
                          </div>

                          <div className="flex flex-wrap items-center gap-1.5 justify-end">
                            <span
                              className={`text-[10px] font-mono font-bold px-2 py-0.5 rounded border ${getDecisionBadge(
                                inc.validation_decision
                              )}`}
                            >
                              {inc.validation_decision}
                            </span>
                            <span
                              className={`text-[10px] font-mono px-1.5 py-0.5 rounded border ${getReliabilityBadge(
                                inc.reliability_rating
                              )}`}
                            >
                              {inc.reliability_rating}
                            </span>
                          </div>
                        </div>

                        <p className="text-xs text-zinc-300 leading-relaxed">
                          {inc.storyline || "No narrative details recorded."}
                        </p>

                        <div className="flex flex-wrap items-center justify-between gap-2 pt-2 border-t border-zinc-800 text-[11px] font-mono">
                          <div className="flex items-center gap-2">
                            <span className="text-zinc-500">Final Assessment:</span>
                            <span className="text-cyan-400 font-bold">
                              {(inc.assessment_score * 100).toFixed(0)}%
                            </span>
                            <span className="text-zinc-500">Pattern Evidence Strength:</span>
                            <span className="text-emerald-400">
                              {(inc.evidence_strength * 100).toFixed(0)}%
                            </span>
                          </div>

                          <div className="flex items-center gap-2">
                            {onSeek && (
                              <button
                                onClick={() => onSeek(inc.start_time, inc.incident_id)}
                                className="bg-zinc-800 hover:bg-zinc-700 text-zinc-200 px-2 py-0.5 rounded text-[10px] transition-colors"
                              >
                                ▶ Seek to {inc.start_time.toFixed(1)}s
                              </button>
                            )}

                            {inc.primary_track_ids && inc.primary_track_ids[0] && (
                              <button
                                onClick={() => handleInvestigateTrack(inc.primary_track_ids[0])}
                                className="bg-purple-900/60 hover:bg-purple-800 text-purple-200 px-2 py-0.5 rounded text-[10px] transition-colors"
                              >
                                Track {inc.primary_track_ids[0]}
                              </button>
                            )}
                          </div>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          )}
        </div>
      )}

      {/* ------------------------------------------------------------- */}
      {/* SUBTAB: STRUCTURED FILTERS */}
      {/* ------------------------------------------------------------- */}
      {activeSubTab === "structured" && (
        <div className="space-y-4">
          <div className="bg-zinc-900 border border-zinc-800 p-5 rounded-xl space-y-4">
            <h4 className="text-xs font-mono uppercase text-zinc-400 tracking-wider">
              Multi-Signal Precision Query Parameters
            </h4>

            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4 text-xs font-mono">
              {/* Category */}
              <div>
                <label className="text-zinc-400 block mb-1">Incident Category</label>
                <select
                  value={structuredCategory}
                  onChange={(e) => setStructuredCategory(e.target.value)}
                  className="w-full bg-zinc-950 border border-zinc-700 text-zinc-200 rounded-lg p-2 focus:outline-none focus:border-indigo-500"
                >
                  <option value="all">All Categories</option>
                  <option value="VEHICLE">Vehicle Activity</option>
                  <option value="PERSON">Person Activity</option>
                  <option value="PROPERTY">Property / Asset</option>
                  <option value="CROWD">Crowd Dynamics</option>
                  <option value="ZONE">Zone Incursion</option>
                  <option value="SPECIALIZED">Specialized Visual</option>
                </select>
              </div>

              {/* Validation Decision */}
              <div>
                <label className="text-zinc-400 block mb-1">Validation Decision</label>
                <select
                  value={structuredDecision}
                  onChange={(e) => setStructuredDecision(e.target.value)}
                  className="w-full bg-zinc-950 border border-zinc-700 text-zinc-200 rounded-lg p-2 focus:outline-none focus:border-indigo-500"
                >
                  <option value="all">All Decisions</option>
                  <option value="CONFIRMED">CONFIRMED (Strong)</option>
                  <option value="PROBABLE">PROBABLE (Likely)</option>
                  <option value="REVIEW_REQUIRED">REVIEW_REQUIRED (&le; 0.65)</option>
                  <option value="REJECTED">REJECTED (Filtered out)</option>
                </select>
              </div>

              {/* Time Window */}
              <div>
                <label className="text-zinc-400 block mb-1">Time Range (Start / End s)</label>
                <div className="flex items-center gap-2">
                  <input
                    type="number"
                    placeholder="0.0"
                    value={timeStart}
                    onChange={(e) => setTimeStart(e.target.value)}
                    className="w-1/2 bg-zinc-950 border border-zinc-700 text-zinc-200 rounded-lg p-2 focus:outline-none focus:border-indigo-500"
                  />
                  <span className="text-zinc-600">&rarr;</span>
                  <input
                    type="number"
                    placeholder="60.0"
                    value={timeEnd}
                    onChange={(e) => setTimeEnd(e.target.value)}
                    className="w-1/2 bg-zinc-950 border border-zinc-700 text-zinc-200 rounded-lg p-2 focus:outline-none focus:border-indigo-500"
                  />
                </div>
              </div>

              {/* Min Assessment Score */}
              <div>
                <label className="text-zinc-400 block mb-1">
                  Min Assessment Score: {(minScore * 100).toFixed(0)}%
                </label>
                <input
                  type="range"
                  min="0"
                  max="1"
                  step="0.05"
                  value={minScore}
                  onChange={(e) => setMinScore(parseFloat(e.target.value))}
                  className="w-full accent-indigo-500 mt-2"
                />
              </div>
            </div>

            {/* Checkboxes */}
            <div className="flex flex-wrap items-center gap-6 pt-2 border-t border-zinc-800 text-xs font-mono">
              <label className="flex items-center gap-2 cursor-pointer text-zinc-300">
                <input
                  type="checkbox"
                  checked={evidenceOnly}
                  onChange={(e) => setEvidenceOnly(e.target.checked)}
                  className="accent-indigo-500 rounded"
                />
                Require Evidence (Clip/Snapshot)
              </label>

              <label className="flex items-center gap-2 cursor-pointer text-zinc-300">
                <input
                  type="checkbox"
                  checked={reviewOnly}
                  onChange={(e) => setReviewOnly(e.target.checked)}
                  className="accent-amber-500 rounded"
                />
                Review Required Only
              </label>
            </div>

            {/* Submit */}
            <div className="flex justify-end pt-2">
              <button
                onClick={handleStructuredQuery}
                disabled={isSearching}
                className="bg-indigo-600 hover:bg-indigo-500 disabled:opacity-50 text-white text-xs font-semibold px-6 py-2.5 rounded-lg transition-colors font-mono flex items-center gap-2"
              >
                {isSearching ? "Executing..." : "Execute Structured Query"}
              </button>
            </div>
          </div>

          {/* Results render identically to search results */}
          {searchResult && searchResult.matched_incidents.length > 0 && (
            <div className="space-y-3">
              <h4 className="text-xs font-mono uppercase text-zinc-400 tracking-wider">
                Filtered Incidents ({searchResult.matched_incidents.length})
              </h4>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                {searchResult.matched_incidents.map((inc) => (
                  <div
                    key={inc.incident_id}
                    className="bg-zinc-900/80 border border-zinc-800 p-4 rounded-xl space-y-2"
                  >
                    <div className="flex items-center justify-between">
                      <span className="text-xs font-bold text-zinc-200">
                        {inc.incident_category}
                      </span>
                      <span
                        className={`text-[10px] font-mono px-2 py-0.5 rounded border ${getDecisionBadge(
                          inc.validation_decision
                        )}`}
                      >
                        {inc.validation_decision}
                      </span>
                    </div>
                    <p className="text-xs text-zinc-400">{inc.storyline}</p>
                    <div className="flex items-center justify-between text-[11px] font-mono text-zinc-500 pt-2 border-t border-zinc-800">
                      <span>Final Assessment: {(inc.assessment_score * 100).toFixed(0)}%</span>
                      {onSeek && (
                        <button
                          onClick={() => onSeek(inc.start_time, inc.incident_id)}
                          className="text-indigo-400 hover:underline"
                        >
                          Seek {inc.start_time.toFixed(1)}s
                        </button>
                      )}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      )}

      {/* ------------------------------------------------------------- */}
      {/* SUBTAB: UNIFIED TIMELINE */}
      {/* ------------------------------------------------------------- */}
      {activeSubTab === "timeline" && (
        <div className="space-y-4">
          {/* Controls Bar */}
          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 bg-zinc-900 border border-zinc-800 p-4 rounded-xl">
            <div className="flex items-center gap-2">
              <span className="text-xs font-mono text-zinc-400">Layer:</span>
              <select
                value={timelineLayerFilter}
                onChange={(e) => setTimelineLayerFilter(e.target.value)}
                className="bg-zinc-950 border border-zinc-700 text-zinc-200 rounded px-2.5 py-1 text-xs font-mono"
              >
                <option value="all">All Layers ({timelineEntries.length})</option>
                <option value="correlated_incident">
                  Correlated Incidents ({timelineCounts.correlated_incidents})
                </option>
                <option value="security_event">
                  Security Events ({timelineCounts.security_events})
                </option>
                <option value="evidence">
                  Evidence Vault Items ({timelineCounts.evidence})
                </option>
                <option value="detection_event">
                  Detection Events ({timelineCounts.detection_events})
                </option>
              </select>

              <label className="flex items-center gap-1.5 text-xs font-mono text-zinc-400 ml-3 cursor-pointer">
                <input
                  type="checkbox"
                  checked={includeRejectedTimeline}
                  onChange={(e) => setIncludeRejectedTimeline(e.target.checked)}
                  className="accent-emerald-500"
                />
                Include Rejected
              </label>
            </div>

            <button
              onClick={() => fetchTimeline()}
              disabled={timelineLoading}
              className="bg-zinc-800 hover:bg-zinc-700 text-zinc-200 text-xs font-mono px-3 py-1.5 rounded transition-colors"
            >
              {timelineLoading ? "Refreshing..." : "↻ Refresh Timeline"}
            </button>
          </div>

          {/* Timeline Feed */}
          {timelineLoading ? (
            <div className="p-8 text-center text-zinc-500 text-xs font-mono">
              Loading unified multi-layer timeline...
            </div>
          ) : filteredTimeline.length === 0 ? (
            <div className="p-8 text-center text-zinc-500 text-xs font-mono bg-zinc-950/40 border border-zinc-800 rounded-xl">
              No timeline events found matching the selected layer.
            </div>
          ) : (
            <div className="space-y-2">
              {filteredTimeline.map((item, idx) => (
                <div
                  key={`${item.id}-${idx}`}
                  className={`border rounded-lg p-3 flex flex-col sm:flex-row sm:items-center justify-between gap-3 text-xs transition-colors ${getLayerColor(
                    item.layer
                  )}`}
                >
                  <div className="flex items-start sm:items-center gap-3">
                    <span className="font-mono text-zinc-400 bg-zinc-950/60 px-2 py-1 rounded text-[11px]">
                      {item.timestamp.toFixed(1)}s
                      {item.end_timestamp > item.timestamp
                        ? ` - ${item.end_timestamp.toFixed(1)}s`
                        : ""}
                    </span>

                    <span className="font-mono text-[10px] uppercase tracking-wider px-2 py-0.5 rounded bg-zinc-950/60 text-zinc-300">
                      {item.layer.replace("_", " ")}
                    </span>

                    <span className="font-semibold text-zinc-100">
                      {item.label}
                    </span>

                    {item.validation_decision && (
                      <span
                        className={`text-[9px] font-mono px-1.5 py-0.5 rounded border ${getDecisionBadge(
                          item.validation_decision
                        )}`}
                      >
                        {item.validation_decision}
                      </span>
                    )}

                    {item.assessment_score !== undefined && (
                      <span className="text-[10px] font-mono text-cyan-400">
                        {(item.assessment_score * 100).toFixed(0)}%
                      </span>
                    )}
                  </div>

                  <div className="flex items-center gap-2">
                    {item.track_id && (
                      <button
                        onClick={() => handleInvestigateTrack(item.track_id)}
                        className="text-[10px] font-mono bg-purple-900/60 hover:bg-purple-800 text-purple-200 px-2 py-0.5 rounded transition-colors"
                      >
                        Track: {item.track_id}
                      </button>
                    )}

                    {onSeek && (
                      <button
                        onClick={() => onSeek(item.timestamp, item.id)}
                        className="text-[10px] font-mono bg-zinc-800 hover:bg-zinc-700 text-zinc-200 px-2.5 py-1 rounded transition-colors"
                      >
                        ▶ Seek
                      </button>
                    )}
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      )}

      {/* ------------------------------------------------------------- */}
      {/* SUBTAB: TRACK DEEP DIVE */}
      {/* ------------------------------------------------------------- */}
      {activeSubTab === "track" && (
        <div className="space-y-4">
          <div className="bg-zinc-900 border border-zinc-800 p-4 rounded-xl flex gap-2">
            <input
              type="text"
              value={selectedTrackId}
              onChange={(e) => setSelectedTrackId(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && handleInvestigateTrack()}
              placeholder="Enter Track ID to reconstruct lifecycle (e.g. 'track_0', 'track_1')..."
              className="flex-1 bg-zinc-950 border border-zinc-700 text-zinc-100 text-sm rounded-lg px-4 py-2 focus:outline-none focus:border-purple-500 font-mono"
            />
            <button
              onClick={() => handleInvestigateTrack()}
              disabled={isTrackLoading || !selectedTrackId.trim()}
              className="bg-purple-600 hover:bg-purple-500 disabled:opacity-50 text-white text-xs font-semibold px-5 py-2 rounded-lg transition-colors font-mono"
            >
              {isTrackLoading ? "Reconstructing..." : "Deep Dive"}
            </button>
          </div>

          {trackError && (
            <div className="p-3 bg-red-950/40 border border-red-500/40 text-red-300 rounded-lg text-xs font-mono">
              Track Error: {trackError}
            </div>
          )}

          {trackResult && (
            <div className="space-y-4">
              {/* Track summary card */}
              <div className="bg-zinc-950 border border-purple-500/40 rounded-xl p-5 space-y-3">
                <div className="flex items-center justify-between">
                  <h4 className="text-sm font-bold text-zinc-100 flex items-center gap-2">
                    <span className="w-2 h-2 rounded-full bg-purple-400 animate-pulse" />
                    TRACK LIFECYCLE: {trackResult.track_id}
                  </h4>
                  {trackResult.track && (
                    <span className="text-xs font-mono bg-purple-900/60 text-purple-300 px-2 py-0.5 rounded">
                      Class: {trackResult.track.object_class}
                    </span>
                  )}
                </div>

                {trackResult.track && (
                  <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 text-xs font-mono bg-zinc-900/70 p-3 rounded-lg border border-zinc-800">
                    <div>
                      <span className="text-zinc-500 block">First Seen:</span>
                      <span className="text-zinc-200">{trackResult.track.first_seen.toFixed(1)}s</span>
                    </div>
                    <div>
                      <span className="text-zinc-500 block">Last Seen:</span>
                      <span className="text-zinc-200">{trackResult.track.last_seen.toFixed(1)}s</span>
                    </div>
                    <div>
                      <span className="text-zinc-500 block">Duration:</span>
                      <span className="text-zinc-200">{trackResult.track.duration_seconds.toFixed(1)}s</span>
                    </div>
                    <div>
                      <span className="text-zinc-500 block">Detections:</span>
                      <span className="text-zinc-200">{trackResult.track.detection_count}</span>
                    </div>
                  </div>
                )}

                {/* Lifecycle Phases */}
                {trackResult.lifecycle?.phases && trackResult.lifecycle.phases.length > 0 && (
                  <div className="space-y-2 pt-2">
                    <span className="text-xs font-mono uppercase text-zinc-400">Reconstructed Phases:</span>
                    <div className="space-y-1.5">
                      {trackResult.lifecycle.phases.map((ph, idx) => (
                        <div
                          key={idx}
                          className="bg-zinc-900 border border-zinc-800 p-2.5 rounded flex items-center justify-between text-xs font-mono"
                        >
                          <div className="flex items-center gap-2">
                            <span className="text-purple-400 font-bold">{ph.phase}</span>
                            <span className="text-zinc-400">&mdash; {ph.note}</span>
                          </div>
                          {ph.timestamp !== undefined && (
                            <span className="text-zinc-500">{ph.timestamp.toFixed(1)}s</span>
                          )}
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            </div>
          )}
        </div>
      )}

      {/* ------------------------------------------------------------- */}
      {/* SUBTAB: EVIDENCE BUNDLES */}
      {/* ------------------------------------------------------------- */}
      {activeSubTab === "bundles" && (
        <div className="space-y-4">
          {/* Create Bundle Card */}
          <div className="bg-zinc-900 border border-zinc-800 p-5 rounded-xl space-y-3">
            <h4 className="text-xs font-mono uppercase text-zinc-400 tracking-wider">
              Create Evidence Bundle
            </h4>
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
              <input
                type="text"
                value={bundleName}
                onChange={(e) => setBundleName(e.target.value)}
                placeholder="Bundle Name (e.g. 'Gate Incursion Case #1')..."
                className="bg-zinc-950 border border-zinc-700 text-zinc-100 text-xs rounded-lg px-3 py-2 focus:outline-none focus:border-amber-500 font-mono"
              />
              <input
                type="text"
                value={bundleNotes}
                onChange={(e) => setBundleNotes(e.target.value)}
                placeholder="Investigator Notes (optional)..."
                className="bg-zinc-950 border border-zinc-700 text-zinc-100 text-xs rounded-lg px-3 py-2 focus:outline-none focus:border-amber-500 font-mono"
              />
            </div>

            <div className="flex items-center justify-between pt-2">
              <span className="text-[11px] font-mono text-zinc-500">
                {searchResult
                  ? `Package ${searchResult.matched_incidents.length} matched incidents from active query`
                  : "Run a query first to automatically include matched incidents"}
              </span>

              <button
                onClick={handleCreateBundleFromSearch}
                disabled={isCreatingBundle || !bundleName.trim()}
                className="bg-amber-600 hover:bg-amber-500 disabled:opacity-50 text-white text-xs font-semibold px-4 py-2 rounded-lg transition-colors font-mono"
              >
                {isCreatingBundle ? "Saving..." : "Save Evidence Bundle"}
              </button>
            </div>

            {bundleMessage && (
              <div className="p-2.5 bg-zinc-950 border border-zinc-700 text-amber-300 text-xs font-mono rounded">
                {bundleMessage}
              </div>
            )}
          </div>

          {/* Existing Bundles List */}
          <div className="space-y-3">
            <h4 className="text-xs font-mono uppercase text-zinc-400 tracking-wider">
              Saved Bundles ({bundles.length})
            </h4>

            {isBundlesLoading ? (
              <div className="p-6 text-center text-zinc-500 text-xs font-mono">
                Loading saved bundles...
              </div>
            ) : bundles.length === 0 ? (
              <div className="p-6 text-center text-zinc-500 text-xs font-mono bg-zinc-950/40 border border-zinc-800 rounded-xl">
                No saved evidence bundles found for this video.
              </div>
            ) : (
              <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                {bundles.map((b) => (
                  <div
                    key={b.bundle_id}
                    className="bg-zinc-900 border border-zinc-800 p-4 rounded-xl space-y-2 text-xs font-mono"
                  >
                    <div className="flex items-center justify-between">
                      <span className="font-bold text-amber-400">{b.bundle_name}</span>
                      <span className="text-[10px] text-zinc-500">{b.bundle_id}</span>
                    </div>
                    {b.notes && <p className="text-zinc-400 text-xs font-sans">{b.notes}</p>}
                    <div className="flex items-center gap-3 text-[11px] text-zinc-500 pt-1 border-t border-zinc-800">
                      <span>Incidents: {b.selected_incident_ids?.length || 0}</span>
                      <span>Events: {b.selected_event_ids?.length || 0}</span>
                      <span>Tracks: {b.selected_track_ids?.length || 0}</span>
                      <span>Evidence: {b.selected_evidence_ids?.length || 0}</span>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
