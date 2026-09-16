"use client";

import React, { useState, useEffect } from "react";
import {
  CorrelatedIncident,
  getVideoCorrelatedIncidents,
} from "@/lib/api";

interface CorrelatedIncidentsViewProps {
  videoId: string;
  onSeek?: (timestamp: number, eventId?: string) => void;
}

export function CorrelatedIncidentsView({ videoId, onSeek }: CorrelatedIncidentsViewProps) {
  const [incidents, setIncidents] = useState<CorrelatedIncident[]>([]);
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [selectedCategory, setSelectedCategory] = useState<string>("all");
  const [selectedSeverity, setSelectedSeverity] = useState<string>("all");
  const [expandedId, setExpandedId] = useState<string | null>(null);

  const fetchIncidents = async () => {
    setIsLoading(true);
    try {
      const cat = selectedCategory !== "all" ? selectedCategory : undefined;
      const sev = selectedSeverity !== "all" ? selectedSeverity : undefined;
      const res = await getVideoCorrelatedIncidents(videoId, cat, sev);
      setIncidents(res.correlated_incidents || []);
    } catch {
      // ignore
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    if (videoId) {
      fetchIncidents();
    }
  }, [videoId, selectedCategory, selectedSeverity]);

  const toggleExpand = (id: string) => {
    setExpandedId((prev) => (prev === id ? null : id));
  };

  return (
    <div className="space-y-4">
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 bg-zinc-950/80 border border-zinc-800 p-4 rounded-xl">
        <div>
          <h4 className="text-sm font-bold text-zinc-100 flex items-center gap-2">
            <span className="w-2.5 h-2.5 rounded-full bg-cyan-400 animate-pulse" />
            CORRELATED INCIDENTS &amp; MULTI-SIGNAL STORYLINES
            <span className="text-[10px] font-mono uppercase bg-cyan-500/20 text-cyan-300 border border-cyan-500/40 px-2 py-0.5 rounded">
              Multi-Signal Fusion
            </span>
          </h4>
          <p className="text-xs text-zinc-400 mt-1">
            Synthesized incident storylines arbitrating competing hypotheses and uniting multi-signal observations.
          </p>
        </div>

        <div className="flex items-center gap-2 font-mono text-xs">
          <select
            value={selectedCategory}
            onChange={(e) => setSelectedCategory(e.target.value)}
            className="bg-zinc-900 border border-zinc-700 text-zinc-200 rounded px-2.5 py-1 text-xs"
          >
            <option value="all">All Categories</option>
            <option value="VEHICLE">Vehicle</option>
            <option value="PROPERTY">Property</option>
            <option value="PERSON">Person</option>
            <option value="CROWD">Crowd</option>
            <option value="ZONE">Zone</option>
            <option value="SPECIALIZED">Specialized Visual</option>
          </select>

          <select
            value={selectedSeverity}
            onChange={(e) => setSelectedSeverity(e.target.value)}
            className="bg-zinc-900 border border-zinc-700 text-zinc-200 rounded px-2.5 py-1 text-xs"
          >
            <option value="all">All Reliability</option>
            <option value="HIGH">High Reliability</option>
            <option value="MEDIUM">Medium Reliability</option>
            <option value="LOW">Low Reliability</option>
          </select>

          <button
            onClick={fetchIncidents}
            disabled={isLoading}
            className="bg-zinc-800 hover:bg-zinc-700 text-zinc-200 px-3 py-1 rounded border border-zinc-700 transition-colors"
          >
            {isLoading ? "Refreshing..." : "Refresh"}
          </button>
        </div>
      </div>

      {incidents.length === 0 ? (
        <div className="rounded-xl border border-zinc-800 bg-zinc-950/40 p-8 text-center text-xs text-zinc-400 font-mono">
          {isLoading ? "Loading correlated incidents..." : "No correlated incidents recorded. Run security analysis to fuse multi-signal candidates."}
        </div>
      ) : (
        <div className="space-y-3">
          {incidents.map((ci) => {
            const isExpanded = expandedId === ci.incident_id;
            const isAccepted = ci.validation_decision === "ACCEPTED";
            const scorePct = Math.round(ci.assessment_score * 100);

            return (
              <div
                key={ci.incident_id}
                className={`rounded-xl border transition-all ${
                  isAccepted
                    ? "bg-zinc-900/90 border-emerald-500/40 hover:border-emerald-500/70"
                    : "bg-zinc-900/70 border-amber-500/30 hover:border-amber-500/60"
                }`}
              >
                <div
                  onClick={() => toggleExpand(ci.incident_id)}
                  className="p-4 cursor-pointer flex flex-col md:flex-row md:items-center justify-between gap-3"
                >
                  <div className="space-y-1.5 flex-1">
                    <div className="flex flex-wrap items-center gap-2">
                      <span className="font-mono text-xs font-bold text-zinc-300 bg-zinc-800 px-2 py-0.5 rounded border border-zinc-700">
                        {ci.incident_category}
                      </span>
                      {ci.incident_subcategory && (
                        <span className="font-mono text-[11px] text-zinc-400">
                          / {ci.incident_subcategory}
                        </span>
                      )}
                      <span
                        className={`font-mono text-[10px] font-bold px-2 py-0.5 rounded border ${
                          isAccepted
                            ? "bg-emerald-500/20 text-emerald-400 border-emerald-500/40"
                            : "bg-amber-500/20 text-amber-300 border-amber-500/40"
                        }`}
                      >
                        {ci.validation_decision.replace(/_/g, " ")}
                      </span>
                      <span className="font-mono text-[11px] text-zinc-400">
                        Reliability: <b className="text-zinc-200">{ci.reliability_rating}</b> &bull; Final Assessment: <b className="text-zinc-200">{scorePct}%</b>
                        {ci.evidence_strength != null && (
                          <span> &bull; Evidence Strength: <b className="text-zinc-300">{Math.round(ci.evidence_strength * 100)}%</b></span>
                        )}
                      </span>
                    </div>

                    <div className="text-xs text-zinc-200 font-sans leading-relaxed pt-1">
                      {ci.storyline}
                    </div>

                    <div className="flex flex-wrap items-center gap-2 pt-1 font-mono text-[11px] text-zinc-400">
                      <span>Time: <b>{ci.start_time.toFixed(1)}s - {ci.end_time.toFixed(1)}s</b> ({ci.duration.toFixed(1)}s)</span>
                      <span>&bull;</span>
                      <span>Primary Tracks: <b className="text-zinc-300">{ci.primary_track_ids.join(", ") || "None"}</b></span>
                      {ci.involved_object_classes.length > 0 && (
                        <>
                          <span>&bull;</span>
                          <span>Entities: {ci.involved_object_classes.join(", ")}</span>
                        </>
                      )}
                      <span>&bull;</span>
                      <span>Fused Candidates: {ci.source_candidate_ids.length}</span>
                    </div>
                  </div>

                  <div className="flex items-center gap-2 self-end md:self-center font-mono text-xs">
                    {onSeek && (
                      <button
                        type="button"
                        onClick={(e) => {
                          e.stopPropagation();
                          onSeek(ci.start_time, ci.incident_id);
                        }}
                        className="bg-zinc-800 hover:bg-zinc-700 text-cyan-400 border border-zinc-700 px-3 py-1.5 rounded transition-colors"
                      >
                        Seek to {ci.start_time.toFixed(1)}s
                      </button>
                    )}
                    <button
                      type="button"
                      className="text-zinc-400 hover:text-zinc-200 px-2 py-1"
                    >
                      {isExpanded ? "Collapse ▲" : "Provenance ▼"}
                    </button>
                  </div>
                </div>

                {isExpanded && (
                  <div className="border-t border-zinc-800 bg-black/40 p-4 rounded-b-xl space-y-3 font-mono text-xs">
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                      <div className="space-y-1">
                        <span className="text-zinc-500 uppercase tracking-wider text-[10px] block">Supporting Tracks &amp; Signals</span>
                        <div className="text-zinc-300">
                          <div>Supporting Tracks: {ci.supporting_track_ids.join(", ") || "None"}</div>
                          <div>Supporting Signals: {ci.supporting_signal_ids.join(", ") || "None"}</div>
                          <div>Detectors: {ci.source_detector_ids.join(", ") || "None"}</div>
                        </div>
                      </div>

                      <div className="space-y-1">
                        <span className="text-zinc-500 uppercase tracking-wider text-[10px] block">Evidence &amp; Provenance</span>
                        <div className="text-zinc-300">
                          <div>Evidence IDs: {ci.evidence_ids.join(", ") || "None"}</div>
                          <div>Negative Evidence: {ci.negative_evidence.join(", ") || "None observed"}</div>
                          <div>Zones: {ci.zone_ids.join(", ") || "Scene wide"}</div>
                        </div>
                      </div>
                    </div>

                    {ci.provenance && Object.keys(ci.provenance).length > 0 && (
                      <div className="mt-2 pt-2 border-t border-zinc-800/80">
                        <span className="text-zinc-500 uppercase tracking-wider text-[10px] block mb-1">Audit Provenance Metadata</span>
                        <pre className="text-[11px] text-zinc-400 bg-zinc-950 p-2.5 rounded border border-zinc-800 overflow-x-auto">
                          {JSON.stringify(ci.provenance, null, 2)}
                        </pre>
                      </div>
                    )}
                  </div>
                )}
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}
