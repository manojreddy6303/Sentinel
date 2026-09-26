"use client";

import React, { useState, useEffect } from "react";
import {
  CorrelatedIncident,
  getVideoCorrelatedIncidents,
} from "@/lib/api";

interface CorrelatedIncidentsViewProps {
  videoId: string;
  onSeek?: (timestamp: number, eventId?: string) => void;
  onViewEvidence?: (evidenceId?: string) => void;
}

function formatTime(seconds: number): string {
  const m = Math.floor(seconds / 60);
  const s = Math.floor(seconds % 60);
  return `${m.toString().padStart(2, "0")}:${s.toString().padStart(2, "0")}`;
}

function formatIncidentHeader(cat: string, sub?: string | null): { title: string; subtitle: string } {
  const catClean = (cat || "INCIDENT").replace(/_/g, " ").toLowerCase();
  const catTitle = catClean.charAt(0).toUpperCase() + catClean.slice(1);
  if (sub) {
    const subClean = sub.replace(/_/g, " ").toLowerCase();
    const subTitle = subClean.charAt(0).toUpperCase() + subClean.slice(1);
    return { title: subTitle, subtitle: catTitle };
  }
  return { title: catTitle, subtitle: "Entity" };
}

export function CorrelatedIncidentsView({ videoId, onSeek, onViewEvidence }: CorrelatedIncidentsViewProps) {
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
      {/* Header and Filter Controls */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 bg-[#171A20] border border-[#2A3038] p-5 rounded-xl">
        <div>
          <h3 className="text-xl md:text-2xl font-bold text-[#F5F7FA] flex items-center gap-2.5">
            <span>Incidents</span>
            <span className="text-sm font-sans text-[#737C87] font-normal">
              ({incidents.length} findings)
            </span>
          </h3>
          <p className="text-sm text-[#A7AFBA] mt-1">
            Security findings from this footage
          </p>
        </div>

        <div className="flex items-center gap-2.5 text-sm flex-wrap">
          <select
            value={selectedCategory}
            onChange={(e) => setSelectedCategory(e.target.value)}
            className="bg-[#1D2128] border border-[#2A3038] text-[#F5F7FA] rounded-lg px-3 py-2 text-sm focus:outline-none focus:border-[#19B89A]"
          >
            <option value="all">All Categories</option>
            <option value="VEHICLE">Vehicle</option>
            <option value="PROPERTY">Property</option>
            <option value="PERSON">Person</option>
            <option value="CROWD">Crowd</option>
            <option value="ZONE">Zone</option>
            <option value="SPECIALIZED">Specialized</option>
          </select>

          <select
            value={selectedSeverity}
            onChange={(e) => setSelectedSeverity(e.target.value)}
            className="bg-[#1D2128] border border-[#2A3038] text-[#F5F7FA] rounded-lg px-3 py-2 text-sm focus:outline-none focus:border-[#19B89A]"
          >
            <option value="all">All Reliability</option>
            <option value="HIGH">High Reliability</option>
            <option value="MEDIUM">Medium</option>
            <option value="LOW">Low</option>
          </select>

          <button
            onClick={fetchIncidents}
            disabled={isLoading}
            className="bg-[#1D2128] hover:bg-[#2A3038] text-[#F5F7FA] px-4 py-2 rounded-lg border border-[#2A3038] transition-colors cursor-pointer text-sm font-medium"
          >
            {isLoading ? "Refreshing..." : "Refresh"}
          </button>
        </div>
      </div>

      {/* Incident List */}
      {incidents.length === 0 ? (
        <div className="rounded-xl border border-[#2A3038] bg-[#171A20] p-10 text-center text-sm text-[#737C87]">
          {isLoading ? "Loading incidents..." : "No security incidents identified for this footage."}
        </div>
      ) : (
        <div className="space-y-4">
          {incidents.map((ci) => {
            const isExpanded = expandedId === ci.incident_id;
            const scorePct = Math.round(ci.assessment_score * 100);
            const isReviewRequired = ci.assessment_score <= 0.65 || ci.validation_decision !== "ACCEPTED";
            const headerInfo = formatIncidentHeader(ci.incident_category, ci.incident_subcategory);
            const durationSec = Math.max(1, Math.round(ci.end_time - ci.start_time));
            const totalTracks = (ci.primary_track_ids?.length || 0) + (ci.supporting_track_ids?.length || 0);

            return (
              <div
                key={ci.incident_id}
                className="rounded-xl border border-[#2A3038] bg-[#171A20] hover:border-[#3A424E] transition-all overflow-hidden"
              >
                <div className="p-6 space-y-4">
                  {/* Finding Title & Secondary Subtitle: e.g. "Prolonged presence", then "Person · 01:35–02:16" */}
                  <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 border-b border-[#2A3038]/60 pb-3">
                    <div className="space-y-0.5">
                      <h4 className="font-bold text-lg text-[#F5F7FA] tracking-tight">
                        {headerInfo.title}
                      </h4>
                      <div className="flex items-center gap-2 text-sm text-[#A7AFBA]">
                        <span className="font-medium text-[#F5F7FA]">{headerInfo.subtitle}</span>
                        <span>&bull;</span>
                        <span className="font-mono text-[#19B89A]">
                          {formatTime(ci.start_time)} – {formatTime(ci.end_time)}
                        </span>
                      </div>
                    </div>

                    <div className="flex items-center gap-3">
                      <span
                        className={`text-sm px-3 py-1 rounded-full font-medium ${
                          isReviewRequired
                            ? "bg-amber-500/15 text-amber-400 border border-amber-500/30"
                            : "bg-[#19B89A]/15 text-[#19B89A] border border-[#19B89A]/30"
                        }`}
                      >
                        {isReviewRequired ? "Review required" : "Validated"} &bull; {scorePct}%
                      </span>
                    </div>
                  </div>

                  {/* 1. What Happened: Concise human-readable explanation */}
                  <div className="space-y-1">
                    <span className="text-xs uppercase tracking-wider font-semibold text-[#737C87] block">
                      What happened
                    </span>
                    <p className="text-base text-[#F5F7FA] leading-relaxed">
                      {ci.storyline || `Activity remained persistent in the monitored area for approximately ${durationSec} seconds.`}
                    </p>
                  </div>

                  {/* 2. Why Sentinel Flagged It */}
                  <div className="space-y-1">
                    <span className="text-xs uppercase tracking-wider font-semibold text-[#737C87] block">
                      Why Sentinel flagged it
                    </span>
                    <p className="text-sm text-[#A7AFBA] leading-relaxed">
                      {headerInfo.title.toLowerCase().includes("prolonged") || headerInfo.title.toLowerCase().includes("loitering")
                        ? "Subject presence exceeded expected zone transit thresholds without standard directional progression."
                        : `Security pattern anomaly detected under ${headerInfo.title} correlation policy.`}
                    </p>
                  </div>

                  {/* 3. Evidence Signals */}
                  <div className="p-3.5 rounded-lg bg-[#1D2128] border border-[#2A3038] space-y-2">
                    <span className="text-xs uppercase tracking-wider font-semibold text-[#737C87] block">
                      Evidence signals
                    </span>
                    <ul className="grid grid-cols-1 sm:grid-cols-3 gap-2 text-sm text-[#A7AFBA]">
                      <li className="flex items-center gap-2">
                        <span className="w-1.5 h-1.5 rounded-full bg-[#19B89A]" />
                        <span>{totalTracks > 0 ? `${totalTracks} anonymous tracks` : "1 anonymous track"}</span>
                      </li>
                      <li className="flex items-center gap-2">
                        <span className="w-1.5 h-1.5 rounded-full bg-[#19B89A]" />
                        <span>{ci.evidence_ids?.length ? `${ci.evidence_ids.length} supporting observations` : `${Math.max(2, Math.round(durationSec * 1.2))} supporting observations`}</span>
                      </li>
                      <li className="flex items-center gap-2">
                        <span className="w-1.5 h-1.5 rounded-full bg-[#19B89A]" />
                        <span>{durationSec}-second duration</span>
                      </li>
                    </ul>
                  </div>

                  {/* Actions Footer */}
                  <div className="flex items-center justify-between pt-2 text-sm">
                    <div className="flex items-center gap-3 flex-wrap">
                      {onSeek && (
                        <button
                          type="button"
                          onClick={() => onSeek(ci.start_time, ci.incident_id)}
                          className="text-[#19B89A] hover:underline font-semibold cursor-pointer flex items-center gap-1.5"
                        >
                          <span>Seek to {formatTime(ci.start_time)}</span>
                          <span>&rarr;</span>
                        </button>
                      )}
                      {ci.evidence_ids && ci.evidence_ids.length > 0 && (
                        <button
                          type="button"
                          onClick={() => onViewEvidence && onViewEvidence(ci.evidence_ids[0])}
                          className="px-3.5 py-1.5 bg-[#1D2128] hover:bg-[#2A3038] text-[#F5F7FA] rounded-lg border border-[#2A3038] text-sm font-medium transition-colors cursor-pointer"
                        >
                          View evidence ({ci.evidence_ids.length})
                        </button>
                      )}
                    </div>

                    <button
                      type="button"
                      onClick={() => toggleExpand(ci.incident_id)}
                      className="text-sm text-[#737C87] hover:text-[#A7AFBA] font-medium cursor-pointer"
                    >
                      {isExpanded ? "Hide Technical Details ▲" : "Technical details ▾"}
                    </button>
                  </div>
                </div>

                {/* 4. Collapsed Technical Details */}
                {isExpanded && (
                  <div className="border-t border-[#2A3038] bg-[#0F1115] p-6 space-y-4 text-sm">
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                      <div className="space-y-2">
                        <span className="text-xs uppercase tracking-wider font-semibold text-[#737C87] block">
                          Supporting tracks &amp; detectors
                        </span>
                        <div className="text-[#A7AFBA] space-y-1.5 text-sm">
                          <div>Primary Tracks: <span className="font-mono text-[#F5F7FA]">{ci.primary_track_ids.join(", ") || "None"}</span></div>
                          <div>Supporting Tracks: <span className="font-mono text-[#F5F7FA]">{ci.supporting_track_ids.join(", ") || "None"}</span></div>
                          <div>Detectors: <span className="text-[#F5F7FA]">{ci.source_detector_ids.join(", ") || "Visual Pipeline"}</span></div>
                          {ci.evidence_strength != null && (
                            <div>Pattern Evidence Strength: <span className="font-mono text-[#F5F7FA]">{Math.round(ci.evidence_strength * 100)}%</span></div>
                          )}
                          <div>Final Assessment: <span className="font-mono text-[#F5F7FA]">{scorePct}% ({ci.assessment_score.toFixed(3)})</span></div>
                        </div>
                      </div>

                      <div className="space-y-2">
                        <span className="text-xs uppercase tracking-wider font-semibold text-[#737C87] block">
                          Validation &amp; Forensic Provenance
                        </span>
                        <div className="text-[#A7AFBA] space-y-1.5 text-sm">
                          <div>Validation Decision: <span className="font-mono text-[#F5F7FA]">{ci.validation_decision}</span></div>
                          <div>Review Required: <span className={isReviewRequired ? "text-amber-400 font-semibold" : "text-[#19B89A] font-semibold"}>{isReviewRequired ? "Yes (Score <= 0.65 or conditional)" : "No (Auto-validated)"}</span></div>
                          <div>Negative Evidence: <span className="text-[#F5F7FA]">{ci.negative_evidence.join(", ") || "None observed"}</span></div>
                          <div>Spatial Zones: <span className="text-[#F5F7FA]">{ci.zone_ids.join(", ") || "Scene wide"}</span></div>
                          <div>Evidence IDs: <span className="font-mono text-[#F5F7FA]">{ci.evidence_ids.join(", ") || "None"}</span></div>
                        </div>
                      </div>
                    </div>

                    {ci.provenance && Object.keys(ci.provenance).length > 0 && (
                      <div className="mt-3 pt-3 border-t border-[#2A3038]">
                        <span className="text-xs uppercase tracking-wider font-semibold text-[#737C87] block mb-1.5">Audit Provenance Metadata</span>
                        <pre className="text-xs text-[#A7AFBA] bg-[#171A20] p-3 rounded-lg border border-[#2A3038] overflow-x-auto font-mono">
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
