"use client";

import React, { useState, useEffect, useCallback } from "react";
import {
  GlobalIncidentItem,
  listAllIncidents,
  listVideos,
  VideoListItem,
  Case,
  listCases,
  linkIncidentToCase,
} from "@/lib/api";

interface IncidentsWorkbenchViewProps {
  onInvestigateVideo?: (videoId: string) => void;
  onOpenReplay?: (caseId: string, incidentId: string) => void;
}

export default function IncidentsWorkbenchView({
  onInvestigateVideo,
  onOpenReplay,
}: IncidentsWorkbenchViewProps) {
  const [incidents, setIncidents] = useState<GlobalIncidentItem[]>([]);
  const [videos, setVideos] = useState<VideoListItem[]>([]);
  const [cases, setCases] = useState<Case[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Filters
  const [selectedVideo, setSelectedVideo] = useState("all");
  const [selectedCategory, setSelectedCategory] = useState("all");
  const [selectedDecision, setSelectedDecision] = useState("all");
  const [searchQuery, setSearchQuery] = useState("");
  const [expandedIncidentId, setExpandedIncidentId] = useState<string | null>(null);
  const [totalIncidents, setTotalIncidents] = useState<number>(0);

  // Add to Case Modal
  const [isAddCaseOpen, setIsAddCaseOpen] = useState(false);
  const [targetIncident, setTargetIncident] = useState<GlobalIncidentItem | null>(null);
  const [selectedCaseId, setSelectedCaseId] = useState("");
  const [isAddingToCase, setIsAddingToCase] = useState(false);
  const [addCaseSuccess, setAddCaseSuccess] = useState<string | null>(null);
  const [addCaseError, setAddCaseError] = useState<string | null>(null);

  const fetchData = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const [incRes, vidRes, caseRes] = await Promise.all([
        listAllIncidents({
          video_id: selectedVideo !== "all" ? selectedVideo : undefined,
          category: selectedCategory !== "all" ? selectedCategory : undefined,
          decision: selectedDecision !== "all" ? selectedDecision : undefined,
          search: searchQuery || undefined,
          limit: 150,
        }),
        listVideos({ limit: 100 }),
        listCases({ limit: 100 }),
      ]);
      setIncidents(incRes.incidents || []);
      setTotalIncidents(incRes.total ?? incRes.incidents?.length ?? 0);
      setVideos(vidRes.videos || []);
      setCases(caseRes.cases || []);
      if (caseRes.cases?.length > 0 && !selectedCaseId) {
        setSelectedCaseId(caseRes.cases[0].id);
      }
    } catch (err: any) {
      setError(err.message || "Failed to load incidents.");
    } finally {
      setLoading(false);
    }
  }, [selectedVideo, selectedCategory, selectedDecision, searchQuery, selectedCaseId]);

  useEffect(() => {
    fetchData();
  }, [fetchData]);

  const handleAddToCase = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!targetIncident || !selectedCaseId) return;
    setIsAddingToCase(true);
    setAddCaseError(null);
    setAddCaseSuccess(null);
    try {
      await linkIncidentToCase(selectedCaseId, targetIncident.id);
      setAddCaseSuccess(`Incident successfully linked to case.`);
      setTimeout(() => {
        setIsAddCaseOpen(false);
        setAddCaseSuccess(null);
      }, 1200);
    } catch (err: any) {
      setAddCaseError(err.message || "Failed to link incident to case.");
    } finally {
      setIsAddingToCase(false);
    }
  };

  return (
    <div className="p-6 md:p-8 space-y-8 max-w-[1600px] mx-auto min-h-full">
      {/* Header */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 border-b border-zinc-800/80 pb-6">
        <div>
          <div className="flex items-center gap-3">
            <h1 className="text-2xl font-bold tracking-tight text-white flex items-center gap-2.5">
              <span className="w-3 h-3 rounded-full bg-cyan-400 animate-pulse" />
              Incident Intelligence Workbench
            </h1>
            <span className="px-2.5 py-0.5 rounded-full text-xs font-mono font-semibold bg-cyan-500/10 text-cyan-400 border border-cyan-500/30">
              {totalIncidents} Correlated Incidents
            </span>
          </div>
          <p className="text-sm text-zinc-400 mt-1.5">
            Correlated multi-signal security incidents, observational hypotheses, and human-in-the-loop validation decisions.
          </p>
        </div>

        {/* Global HITL Standard Banner */}
        <div className="flex items-center gap-2 px-3 py-1.5 rounded-lg bg-zinc-900 border border-zinc-800 text-xs text-zinc-400 font-mono">
          <span className="w-2 h-2 rounded-full bg-amber-400" />
          <span>REVIEW_REQUIRED &le; 0.65 Ceiling Preserved</span>
        </div>
      </div>

      {/* Filter Controls */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3 bg-zinc-900/60 p-4 rounded-xl border border-zinc-800">
        <div>
          <label className="block text-xs font-medium text-zinc-400 mb-1">Source Video</label>
          <select
            value={selectedVideo}
            onChange={(e) => setSelectedVideo(e.target.value)}
            className="w-full bg-zinc-950 border border-zinc-800 rounded-lg p-2 text-xs text-zinc-200 focus:outline-none focus:border-cyan-500"
          >
            <option value="all">All Ingested Videos ({videos.length})</option>
            {videos.map((v) => (
              <option key={v.id} value={v.id}>{v.filename}</option>
            ))}
          </select>
        </div>

        <div>
          <label className="block text-xs font-medium text-zinc-400 mb-1">Incident Category</label>
          <select
            value={selectedCategory}
            onChange={(e) => setSelectedCategory(e.target.value)}
            className="w-full bg-zinc-950 border border-zinc-800 rounded-lg p-2 text-xs text-zinc-200 focus:outline-none focus:border-cyan-500"
          >
            <option value="all">All Categories</option>
            <option value="THEFT_OR_UNAUTHORIZED_REMOVAL">Theft / Removal</option>
            <option value="PROLONGED_LOITERING">Prolonged Loitering</option>
            <option value="PERIMETER_BREACH">Perimeter Breach</option>
            <option value="VEHICLE_COLLISION_OR_IMPACT">Vehicle Collision</option>
            <option value="RESTRICTED_ZONE_INTRUSION">Restricted Intrusion</option>
            <option value="SUSPICIOUS_TRANSFER">Suspicious Transfer</option>
          </select>
        </div>

        <div>
          <label className="block text-xs font-medium text-zinc-400 mb-1">Decision Status</label>
          <select
            value={selectedDecision}
            onChange={(e) => setSelectedDecision(e.target.value)}
            className="w-full bg-zinc-950 border border-zinc-800 rounded-lg p-2 text-xs text-zinc-200 focus:outline-none focus:border-cyan-500"
          >
            <option value="all">All Decisions</option>
            <option value="REVIEW_REQUIRED">REVIEW_REQUIRED (&le; 0.65)</option>
            <option value="ACCEPTED">ACCEPTED (&gt; 0.65)</option>
            <option value="REJECTED">REJECTED</option>
          </select>
        </div>

        <div>
          <label className="block text-xs font-medium text-zinc-400 mb-1">Search Narrative</label>
          <input
            type="text"
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            placeholder="Search narrative keywords..."
            className="w-full bg-zinc-950 border border-zinc-800 rounded-lg p-2 text-xs text-zinc-200 focus:outline-none focus:border-cyan-500"
          />
        </div>
      </div>

      {/* Loading & Error States */}
      {loading && (
        <div className="flex flex-col items-center justify-center py-20 text-zinc-500 space-y-3">
          <div className="w-7 h-7 border-2 border-cyan-500 border-t-transparent rounded-full animate-spin" />
          <span className="text-sm font-mono">Loading correlated security incidents...</span>
        </div>
      )}

      {error && !loading && (
        <div className="p-4 bg-red-500/10 border border-red-500/30 rounded-xl text-red-400 flex items-center justify-between">
          <span className="text-sm font-medium">{error}</span>
          <button onClick={fetchData} className="px-3 py-1 bg-red-500/20 text-red-300 rounded text-xs">
            Retry
          </button>
        </div>
      )}

      {/* Incidents List */}
      {!loading && !error && incidents.length > 0 && (
        <div className="space-y-4">
          {incidents.map((inc) => {
            const isExpanded = expandedIncidentId === inc.id;
            const isReviewRequired = inc.assessment_score <= 0.65 || inc.validation_decision === "REVIEW_REQUIRED";

            return (
              <div
                key={inc.id}
                className="bg-zinc-900/80 border border-zinc-800 hover:border-zinc-700 rounded-xl p-5 space-y-4 transition-all shadow-md"
              >
                {/* Header Row */}
                <div className="flex flex-col md:flex-row md:items-center justify-between gap-3">
                  <div className="flex items-center gap-3">
                    <span className={`w-3 h-3 rounded-full ${isReviewRequired ? "bg-amber-400" : "bg-emerald-400"}`} />
                    <div>
                      <div className="flex items-center gap-2.5 flex-wrap">
                        <h3 className="text-base font-bold text-white">
                          {inc.title}
                        </h3>
                        <span className="px-2 py-0.5 rounded text-[11px] font-mono font-semibold bg-cyan-500/10 text-cyan-400 border border-cyan-500/30">
                          {inc.incident_category}
                        </span>
                        {isReviewRequired ? (
                          <span className="px-2 py-0.5 rounded text-[11px] font-mono font-bold bg-amber-500/10 text-amber-400 border border-amber-500/30">
                            REVIEW_REQUIRED (&le; 0.65)
                          </span>
                        ) : (
                          <span className="px-2 py-0.5 rounded text-[11px] font-mono font-bold bg-emerald-500/10 text-emerald-400 border border-emerald-500/30">
                            {inc.validation_decision}
                          </span>
                        )}
                      </div>
                      <p className="text-xs text-zinc-400 font-mono mt-0.5">
                        Source: <span className="text-zinc-300 font-semibold">{inc.source_video_name}</span> &bull; 
                        Time: <span className="text-zinc-300">{inc.start_time.toFixed(1)}s &ndash; {inc.end_time.toFixed(1)}s</span> ({inc.duration.toFixed(1)}s duration)
                      </p>
                    </div>
                  </div>

                  {/* Assessment Scores */}
                  <div className="flex items-center gap-4 bg-zinc-950/80 px-3.5 py-2 rounded-lg border border-zinc-800 text-xs font-mono">
                    <div>
                      <span className="text-zinc-500 block text-[10px] uppercase">Final Assessment</span>
                      <span className={`font-bold text-sm ${inc.assessment_score > 0.65 ? "text-emerald-400" : "text-amber-400"}`}>
                        {(inc.assessment_score * 100).toFixed(0)}%
                      </span>
                    </div>
                    <div className="border-l border-zinc-800 pl-3">
                      <span className="text-zinc-500 block text-[10px] uppercase">Pattern Evidence</span>
                      <span className="text-blue-400 font-bold text-sm">
                        {(inc.evidence_strength * 100).toFixed(0)}%
                      </span>
                    </div>
                    <div className="border-l border-zinc-800 pl-3">
                      <span className="text-zinc-500 block text-[10px] uppercase">Reliability</span>
                      <span className="text-zinc-300 font-semibold uppercase">{inc.reliability_rating}</span>
                    </div>
                  </div>
                </div>

                {/* Narrative Summary */}
                <div className="text-sm text-zinc-300 leading-relaxed bg-zinc-950/40 p-3.5 rounded-lg border border-zinc-800/60">
                  {inc.storyline}
                </div>

                {/* Key Signals & Participating Tracks */}
                <div className="flex flex-wrap items-center gap-2 text-xs font-mono">
                  <span className="text-zinc-500 font-medium">Anonymous Tracks:</span>
                  {inc.participating_tracks.length > 0 ? (
                    inc.participating_tracks.map((tid) => (
                      <span key={tid} className="px-2 py-0.5 rounded bg-zinc-800 text-zinc-300 border border-zinc-700">
                        {tid}
                      </span>
                    ))
                  ) : (
                    <span className="text-zinc-600">None tracked</span>
                  )}

                  {inc.involved_object_classes.length > 0 && (
                    <>
                      <span className="text-zinc-500 font-medium ml-3">Classes:</span>
                      {inc.involved_object_classes.map((cls) => (
                        <span key={cls} className="px-2 py-0.5 rounded bg-blue-500/10 text-blue-300 border border-blue-500/20 capitalize">
                          {cls}
                        </span>
                      ))}
                    </>
                  )}
                </div>

                {/* Expandable Forensic Provenance */}
                {isExpanded && (
                  <div className="mt-3 pt-3 border-t border-zinc-800 space-y-3 text-xs bg-zinc-950/60 p-3.5 rounded-lg">
                    <h4 className="font-bold text-zinc-200 uppercase tracking-wider text-[11px]">
                      Incident Signals &amp; Contradictory Evidence
                    </h4>
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-3 text-zinc-400">
                      <div>
                        <span className="text-zinc-300 font-semibold block mb-1">Evidence References:</span>
                        {inc.evidence_ids.length > 0 ? (
                          <div className="flex flex-wrap gap-1">
                            {inc.evidence_ids.map((eid) => (
                              <span key={eid} className="px-1.5 py-0.5 rounded bg-zinc-800 font-mono text-[11px] text-zinc-300">
                                {eid}
                              </span>
                            ))}
                          </div>
                        ) : (
                          <span className="text-zinc-600">No physical snapshots linked</span>
                        )}
                      </div>

                      <div>
                        <span className="text-zinc-300 font-semibold block mb-1">Negative / Limiting Evidence:</span>
                        {inc.negative_evidence && inc.negative_evidence.length > 0 ? (
                          <div className="space-y-1">
                            {inc.negative_evidence.map((neg: any, i: number) => (
                              <p key={i} className="text-amber-400/90 font-mono text-[11px]">
                                &bull; {typeof neg === "string" ? neg : JSON.stringify(neg)}
                              </p>
                            ))}
                          </div>
                        ) : (
                          <span className="text-zinc-500">No contradictory signals detected</span>
                        )}
                      </div>
                    </div>
                  </div>
                )}

                {/* Actions Row */}
                <div className="flex items-center justify-between pt-2 border-t border-zinc-800/60 flex-wrap gap-3">
                  <button
                    onClick={() => setExpandedIncidentId(isExpanded ? null : inc.id)}
                    className="text-xs text-zinc-400 hover:text-zinc-200 font-mono flex items-center gap-1.5"
                  >
                    <span>{isExpanded ? "▲ Hide Technical Provenance" : "▼ Show Technical Provenance"}</span>
                  </button>

                  <div className="flex items-center gap-2">
                    {onInvestigateVideo && (
                      <button
                        onClick={() => onInvestigateVideo(inc.video_id)}
                        className="px-3 py-1.5 bg-zinc-800 hover:bg-zinc-700 text-zinc-200 rounded-lg text-xs font-semibold transition-all"
                      >
                        Open Video Workbench
                      </button>
                    )}

                    <button
                      onClick={() => {
                        setTargetIncident(inc);
                        setIsAddCaseOpen(true);
                      }}
                      className="px-3 py-1.5 bg-emerald-600/20 hover:bg-emerald-600/30 text-emerald-400 border border-emerald-500/40 rounded-lg text-xs font-semibold transition-all"
                    >
                      + Add to Case
                    </button>
                  </div>
                </div>
              </div>
            );
          })}
        </div>
      )}

      {/* Empty State */}
      {!loading && !error && incidents.length === 0 && (
        <div className="rounded-xl border border-zinc-800 bg-zinc-900/40 p-12 text-center space-y-4">
          <div className="w-12 h-12 rounded-full bg-zinc-800 flex items-center justify-center mx-auto text-zinc-400">
            <svg className="w-6 h-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
            </svg>
          </div>
          <div className="space-y-1">
            <h3 className="text-base font-bold text-zinc-200">No security incidents detected</h3>
            <p className="text-sm text-zinc-500 max-w-md mx-auto">
              No correlated security incidents match your selected filters. Run video intelligence analysis to detect security events.
            </p>
          </div>
        </div>
      )}

      {/* Add to Case Modal */}
      {isAddCaseOpen && targetIncident && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/75 backdrop-blur-sm p-4">
          <div className="bg-zinc-900 border border-zinc-800 rounded-xl max-w-md w-full p-6 space-y-4 shadow-2xl">
            <div className="flex items-center justify-between border-b border-zinc-800 pb-3">
              <h3 className="text-base font-bold text-white">Attach Incident to Security Case</h3>
              <button
                onClick={() => setIsAddCaseOpen(false)}
                className="text-zinc-500 hover:text-zinc-300"
              >
                ✕
              </button>
            </div>

            {addCaseSuccess && (
              <div className="p-3 bg-emerald-500/10 border border-emerald-500/30 rounded text-emerald-400 text-xs">
                {addCaseSuccess}
              </div>
            )}

            {addCaseError && (
              <div className="p-3 bg-red-500/10 border border-red-500/30 rounded text-red-400 text-xs">
                {addCaseError}
              </div>
            )}

            <div className="bg-zinc-950 p-3 rounded-lg border border-zinc-800 text-xs space-y-1.5 font-mono">
              <div><span className="text-zinc-500">Incident:</span> <span className="text-zinc-200">{targetIncident.title}</span></div>
              <div><span className="text-zinc-500">Category:</span> <span className="text-cyan-400">{targetIncident.incident_category}</span></div>
              <div><span className="text-zinc-500">Video:</span> <span className="text-zinc-300">{targetIncident.source_video_name}</span></div>
            </div>

            <form onSubmit={handleAddToCase} className="space-y-4 text-sm">
              <div>
                <label className="block text-zinc-300 font-medium mb-1">Select Case *</label>
                {cases.length > 0 ? (
                  <select
                    required
                    value={selectedCaseId}
                    onChange={(e) => setSelectedCaseId(e.target.value)}
                    className="w-full bg-zinc-950 border border-zinc-800 rounded-lg p-2.5 text-zinc-200 focus:outline-none focus:border-cyan-500 font-mono text-xs"
                  >
                    {cases.map((c) => (
                      <option key={c.id} value={c.id}>
                        {c.case_number || c.id} &ndash; {c.title} ({c.priority})
                      </option>
                    ))}
                  </select>
                ) : (
                  <p className="text-xs text-amber-400">No security cases found. Create a case first.</p>
                )}
              </div>

              <div className="flex justify-end gap-3 pt-2">
                <button
                  type="button"
                  onClick={() => setIsAddCaseOpen(false)}
                  className="px-4 py-2 bg-zinc-800 hover:bg-zinc-700 text-zinc-300 rounded-lg font-medium"
                >
                  Cancel
                </button>
                <button
                  type="submit"
                  disabled={!selectedCaseId || isAddingToCase || cases.length === 0}
                  className="px-4 py-2 bg-emerald-600 hover:bg-emerald-500 text-zinc-950 font-bold rounded-lg transition-all disabled:opacity-50"
                >
                  {isAddingToCase ? "Attaching..." : "Attach Incident"}
                </button>
              </div>
            </form>
          </div>
        </div>
      )}
    </div>
  );
}
