"use client";

import React, { useState, useEffect, useCallback } from "react";
import {
  GlobalEvidenceItem,
  listAllEvidence,
  listVideos,
  VideoListItem,
  Case,
  listCases,
  createCaseBookmark,
} from "@/lib/api";

interface EvidenceVaultViewProps {
  onInvestigateVideo?: (videoId: string) => void;
}

export default function EvidenceVaultView({ onInvestigateVideo }: EvidenceVaultViewProps) {
  const [evidenceList, setEvidenceList] = useState<GlobalEvidenceItem[]>([]);
  const [videos, setVideos] = useState<VideoListItem[]>([]);
  const [cases, setCases] = useState<Case[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Filters
  const [selectedVideo, setSelectedVideo] = useState("all");
  const [selectedClass, setSelectedClass] = useState("all");
  const [selectedStatus, setSelectedStatus] = useState("VALID");
  const [minConfidence, setMinConfidence] = useState(0);
  const [searchQuery, setSearchQuery] = useState("");

  // Preview Modal
  const [previewItem, setPreviewItem] = useState<GlobalEvidenceItem | null>(null);
  const [previewMode, setPreviewMode] = useState<"snapshot" | "annotated" | "clip">("snapshot");

  // Add to Case Modal
  const [isAddCaseOpen, setIsAddCaseOpen] = useState(false);
  const [targetEvidence, setTargetEvidence] = useState<GlobalEvidenceItem | null>(null);
  const [selectedCaseId, setSelectedCaseId] = useState("");
  const [isAdding, setIsAdding] = useState(false);
  const [addFeedback, setAddFeedback] = useState<string | null>(null);
  const [failedImages, setFailedImages] = useState<Record<string, boolean>>({});

  const fetchData = useCallback(async () => {
    setLoading(true);
    setError(null);
    setFailedImages({});
    try {
      const [evRes, vidRes, caseRes] = await Promise.all([
        listAllEvidence({
          video_id: selectedVideo !== "all" ? selectedVideo : undefined,
          object_class: selectedClass !== "all" ? selectedClass : undefined,
          validation_status: selectedStatus !== "all" ? selectedStatus : undefined,
          min_confidence: minConfidence > 0 ? minConfidence : undefined,
          search: searchQuery || undefined,
          limit: 100,
        }),
        listVideos({ limit: 100 }),
        listCases({ limit: 100 }),
      ]);
      setEvidenceList(evRes.evidence || []);
      setVideos(vidRes.videos || []);
      setCases(caseRes.cases || []);
      if (caseRes.cases?.length > 0 && !selectedCaseId) {
        setSelectedCaseId(caseRes.cases[0].id);
      }
    } catch (err: any) {
      setError(err.message || "Failed to load evidence vault.");
    } finally {
      setLoading(false);
    }
  }, [selectedVideo, selectedClass, selectedStatus, minConfidence, searchQuery, selectedCaseId]);

  useEffect(() => {
    fetchData();
  }, [fetchData]);

  const handleLinkToCase = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!targetEvidence || !selectedCaseId) return;
    setIsAdding(true);
    setAddFeedback(null);
    try {
      await createCaseBookmark(selectedCaseId, {
        video_id: targetEvidence.video_id,
        timestamp_seconds: targetEvidence.timestamp_seconds,
        title: `Evidence: ${targetEvidence.object_class} @ ${targetEvidence.timestamp_seconds.toFixed(1)}s`,
        description: targetEvidence.notes || `Preserved evidence item ${targetEvidence.id}`,
        linked_evidence_id: targetEvidence.id,
      });
      setAddFeedback("Evidence bookmarked to case.");
      setTimeout(() => {
        setIsAddCaseOpen(false);
        setAddFeedback(null);
      }, 1200);
    } catch (err: any) {
      setAddFeedback(`Failed to link: ${err.message}`);
    } finally {
      setIsAdding(false);
    }
  };

  return (
    <div className="p-6 md:p-8 space-y-8 max-w-[1600px] mx-auto min-h-full">
      {/* Header */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 border-b border-zinc-800/80 pb-6">
        <div>
          <div className="flex items-center gap-3">
            <h1 className="text-2xl font-bold tracking-tight text-white flex items-center gap-2.5">
              <span className="w-3 h-3 rounded-full bg-emerald-400 animate-pulse" />
              Evidence Vault
            </h1>
            <span className="px-2.5 py-0.5 rounded-full text-xs font-mono font-semibold bg-emerald-500/10 text-emerald-400 border border-emerald-500/30">
              {evidenceList.length} Preserved Artifacts
            </span>
          </div>
          <p className="text-sm text-zinc-400 mt-1.5">
            Immutable forensic snapshots, extracted video clips, integrity chains, and verified physical evidence.
          </p>
        </div>

        <div className="flex items-center gap-2 px-3 py-1.5 rounded-lg bg-zinc-900 border border-zinc-800 text-xs text-zinc-400 font-mono">
          <span className="w-2 h-2 rounded-full bg-emerald-400" />
          <span>Chain of Custody &bull; Non-Destructive Ingest</span>
        </div>
      </div>

      {/* Filter Controls */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-5 gap-3 bg-zinc-900/60 p-4 rounded-xl border border-zinc-800">
        <div>
          <label className="block text-xs font-medium text-zinc-400 mb-1">Source Video</label>
          <select
            value={selectedVideo}
            onChange={(e) => setSelectedVideo(e.target.value)}
            className="w-full bg-zinc-950 border border-zinc-800 rounded-lg p-2 text-xs text-zinc-200 focus:outline-none focus:border-emerald-500"
          >
            <option value="all">All Videos ({videos.length})</option>
            {videos.map((v) => (
              <option key={v.id} value={v.id}>{v.filename}</option>
            ))}
          </select>
        </div>

        <div>
          <label className="block text-xs font-medium text-zinc-400 mb-1">Object Class</label>
          <select
            value={selectedClass}
            onChange={(e) => setSelectedClass(e.target.value)}
            className="w-full bg-zinc-950 border border-zinc-800 rounded-lg p-2 text-xs text-zinc-200 focus:outline-none focus:border-emerald-500"
          >
            <option value="all">All Classes</option>
            <option value="person">Person</option>
            <option value="car">Car</option>
            <option value="truck">Truck</option>
            <option value="motorcycle">Motorcycle</option>
            <option value="backpack">Backpack</option>
            <option value="handbag">Handbag</option>
            <option value="suitcase">Suitcase</option>
          </select>
        </div>

        <div>
          <label className="block text-xs font-medium text-zinc-400 mb-1">Validation Status</label>
          <select
            value={selectedStatus}
            onChange={(e) => setSelectedStatus(e.target.value)}
            className="w-full bg-zinc-950 border border-zinc-800 rounded-lg p-2 text-xs text-zinc-200 focus:outline-none focus:border-emerald-500 font-mono"
          >
            <option value="VALID">VALID (Forensic)</option>
            <option value="all">All Records</option>
            <option value="REJECTED">REJECTED (Filtered)</option>
            <option value="UNCERTAIN">UNCERTAIN (Review)</option>
          </select>
        </div>

        <div>
          <label className="block text-xs font-medium text-zinc-400 mb-1">
            Min Confidence: {(minConfidence * 100).toFixed(0)}%
          </label>
          <input
            type="range"
            min="0"
            max="0.95"
            step="0.05"
            value={minConfidence}
            onChange={(e) => setMinConfidence(parseFloat(e.target.value))}
            className="w-full accent-emerald-500 h-2 bg-zinc-950 rounded cursor-pointer mt-1.5"
          />
        </div>

        <div>
          <label className="block text-xs font-medium text-zinc-400 mb-1">Search Keywords</label>
          <input
            type="text"
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            placeholder="Search notes or filenames..."
            className="w-full bg-zinc-950 border border-zinc-800 rounded-lg p-2 text-xs text-zinc-200 focus:outline-none focus:border-emerald-500"
          />
        </div>
      </div>

      {/* Loading & Error States */}
      {loading && (
        <div className="flex flex-col items-center justify-center py-20 text-zinc-500 space-y-3">
          <div className="w-7 h-7 border-2 border-emerald-500 border-t-transparent rounded-full animate-spin" />
          <span className="text-sm font-mono">Loading preserved evidence vault...</span>
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

      {/* Evidence Items Grid */}
      {!loading && !error && evidenceList.length > 0 && (
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-5">
          {evidenceList.map((item) => (
            <div
              key={item.id}
              className="bg-zinc-900/80 border border-zinc-800 hover:border-zinc-700 rounded-xl overflow-hidden transition-all shadow-md flex flex-col justify-between"
            >
              {/* Media Thumbnail Container */}
              <div className="relative aspect-video bg-zinc-950 flex items-center justify-center overflow-hidden group">
                {!failedImages[item.id] && item.has_annotated && item.annotated_url ? (
                  <img
                    src={item.annotated_url}
                    alt={item.object_class || "Evidence"}
                    className="w-full h-full object-cover group-hover:scale-105 transition-transform duration-200"
                    loading="lazy"
                    onError={() => setFailedImages((prev) => ({ ...prev, [item.id]: true }))}
                  />
                ) : !failedImages[item.id] && item.has_snapshot && item.snapshot_url ? (
                  <img
                    src={item.snapshot_url}
                    alt={item.object_class || "Evidence"}
                    className="w-full h-full object-cover group-hover:scale-105 transition-transform duration-200"
                    loading="lazy"
                    onError={() => setFailedImages((prev) => ({ ...prev, [item.id]: true }))}
                  />
                ) : (
                  <div className="text-zinc-600 flex flex-col items-center justify-center p-3 text-xs space-y-1 w-full h-full">
                    <svg className="w-7 h-7 text-zinc-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
                    </svg>
                    <span className="font-mono text-[10px] text-zinc-500">
                      {item.has_clip ? "Clip Media Only" : "Artifact Unavailable"}
                    </span>
                  </div>
                )}

                {/* Overlays on image */}
                <div className="absolute top-2 left-2 flex gap-1.5">
                  <span className="px-2 py-0.5 rounded text-[10px] font-mono font-bold bg-black/75 text-emerald-400 border border-emerald-500/30 capitalize">
                    {item.object_class}
                  </span>
                  <span className="px-2 py-0.5 rounded text-[10px] font-mono font-bold bg-black/75 text-blue-400 border border-blue-500/30">
                    {(item.confidence * 100).toFixed(0)}%
                  </span>
                </div>

                <div className="absolute bottom-2 right-2 px-2 py-0.5 rounded text-[10px] font-mono font-semibold bg-black/80 text-zinc-300">
                  {item.timestamp_seconds.toFixed(2)}s
                </div>
              </div>

              {/* Card Body */}
              <div className="p-4 space-y-3 flex-1 flex flex-col justify-between">
                <div className="space-y-1.5">
                  <p className="text-xs text-zinc-400 font-mono truncate" title={item.source_video_name}>
                    {item.source_video_name}
                  </p>
                  {item.notes && (
                    <p className="text-xs text-zinc-300 line-clamp-2">
                      {item.notes}
                    </p>
                  )}
                  <div className="flex items-center gap-2 text-[11px] font-mono text-zinc-500">
                    <span>Type: {item.evidence_type}</span>
                    <span>&bull;</span>
                    <span className={
                      item.validation_status === "VALID"
                        ? "text-emerald-400 font-semibold"
                        : item.validation_status === "REJECTED"
                        ? "text-rose-400 font-semibold"
                        : "text-amber-400 font-semibold"
                    }>
                      {item.validation_status}
                    </span>
                  </div>
                </div>

                {/* Actions */}
                <div className="pt-3 border-t border-zinc-800 flex items-center justify-between gap-2">
                  <div className="flex items-center gap-1.5">
                    {item.has_annotated && item.annotated_url && (
                      <button
                        onClick={() => {
                          setPreviewItem(item);
                          setPreviewMode("annotated");
                        }}
                        className="px-2.5 py-1 bg-zinc-800 hover:bg-zinc-700 text-zinc-200 rounded text-xs font-medium"
                      >
                        View
                      </button>
                    )}
                    {item.has_clip && item.clip_url && (
                      <button
                        onClick={() => {
                          setPreviewItem(item);
                          setPreviewMode("clip");
                        }}
                        className="px-2.5 py-1 bg-blue-600/20 hover:bg-blue-600/30 text-blue-300 rounded text-xs font-semibold"
                      >
                        Clip
                      </button>
                    )}
                  </div>

                  <button
                    onClick={() => {
                      setTargetEvidence(item);
                      setIsAddCaseOpen(true);
                    }}
                    className="px-2.5 py-1 bg-emerald-600/20 hover:bg-emerald-600/30 text-emerald-400 rounded text-xs font-semibold"
                  >
                    + Case
                  </button>
                </div>
              </div>
            </div>
          ))}
        </div>
      )}

      {/* Empty State */}
      {!loading && !error && evidenceList.length === 0 && (
        <div className="rounded-xl border border-zinc-800 bg-zinc-900/40 p-12 text-center space-y-4">
          <div className="w-12 h-12 rounded-full bg-zinc-800 flex items-center justify-center mx-auto text-zinc-400">
            <svg className="w-6 h-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 8h14M5 8a2 2 0 110-4h14a2 2 0 110 4M5 8v10a2 2 0 002 2h10a2 2 0 002-2V8m-9 4h4" />
            </svg>
          </div>
          <div className="space-y-1">
            <h3 className="text-base font-bold text-zinc-200">No preserved evidence available</h3>
            <p className="text-sm text-zinc-500 max-w-md mx-auto">
              No evidence items match your filters. Run video intelligence or extract snapshots from detected incidents.
            </p>
          </div>
        </div>
      )}

      {/* Full Preview Modal */}
      {previewItem && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/85 backdrop-blur-md p-4">
          <div className="bg-zinc-900 border border-zinc-800 rounded-xl max-w-3xl w-full p-6 space-y-4 shadow-2xl">
            <div className="flex items-center justify-between border-b border-zinc-800 pb-3">
              <div>
                <h3 className="text-base font-bold text-white">
                  Evidence Artifact: {previewItem.object_class} ({(previewItem.confidence * 100).toFixed(0)}%)
                </h3>
                <p className="text-xs text-zinc-400 font-mono">
                  {previewItem.source_video_name} &bull; Timestamp: {previewItem.timestamp_seconds.toFixed(2)}s
                </p>
              </div>
              <button
                onClick={() => setPreviewItem(null)}
                className="text-zinc-500 hover:text-zinc-300 text-lg"
              >
                ✕
              </button>
            </div>

            {/* Media Area */}
            <div className="bg-zinc-950 rounded-lg overflow-hidden flex items-center justify-center aspect-video">
              {previewMode === "clip" && previewItem.clip_url ? (
                <video
                  src={previewItem.clip_url}
                  controls
                  autoPlay
                  className="w-full h-full object-contain"
                />
              ) : previewMode === "annotated" && previewItem.annotated_url ? (
                <img
                  src={previewItem.annotated_url}
                  alt="Annotated Evidence"
                  className="w-full h-full object-contain"
                  onError={() => {
                    if (previewItem.snapshot_url) {
                      setPreviewMode("snapshot");
                    }
                  }}
                />
              ) : previewItem.snapshot_url ? (
                <img
                  src={previewItem.snapshot_url}
                  alt="Snapshot"
                  className="w-full h-full object-contain"
                />
              ) : (
                <div className="text-zinc-500 flex flex-col items-center justify-center p-6 text-sm">
                  <span>Preview Artifact Unavailable</span>
                </div>
              )}
            </div>

            {/* Mode toggles */}
            <div className="flex items-center justify-between pt-2">
              <div className="flex items-center gap-2">
                {previewItem.has_annotated && (
                  <button
                    onClick={() => setPreviewMode("annotated")}
                    className={`px-3 py-1 rounded text-xs font-semibold ${
                      previewMode === "annotated" ? "bg-emerald-600 text-zinc-950" : "bg-zinc-800 text-zinc-300"
                    }`}
                  >
                    Annotated
                  </button>
                )}
                {previewItem.has_snapshot && (
                  <button
                    onClick={() => setPreviewMode("snapshot")}
                    className={`px-3 py-1 rounded text-xs font-semibold ${
                      previewMode === "snapshot" ? "bg-emerald-600 text-zinc-950" : "bg-zinc-800 text-zinc-300"
                    }`}
                  >
                    Raw Snapshot
                  </button>
                )}
                {previewItem.has_clip && (
                  <button
                    onClick={() => setPreviewMode("clip")}
                    className={`px-3 py-1 rounded text-xs font-semibold ${
                      previewMode === "clip" ? "bg-emerald-600 text-zinc-950" : "bg-zinc-800 text-zinc-300"
                    }`}
                  >
                    Video Clip
                  </button>
                )}
              </div>

              <button
                onClick={() => setPreviewItem(null)}
                className="px-4 py-1.5 bg-zinc-800 hover:bg-zinc-700 text-zinc-200 rounded text-xs font-medium"
              >
                Close
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Add to Case Modal */}
      {isAddCaseOpen && targetEvidence && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/75 backdrop-blur-sm p-4">
          <div className="bg-zinc-900 border border-zinc-800 rounded-xl max-w-md w-full p-6 space-y-4 shadow-2xl">
            <div className="flex items-center justify-between border-b border-zinc-800 pb-3">
              <h3 className="text-base font-bold text-white">Bookmark Evidence to Case</h3>
              <button
                onClick={() => setIsAddCaseOpen(false)}
                className="text-zinc-500 hover:text-zinc-300"
              >
                ✕
              </button>
            </div>

            {addFeedback && (
              <div className="p-3 bg-emerald-500/10 border border-emerald-500/30 rounded text-emerald-400 text-xs">
                {addFeedback}
              </div>
            )}

            <form onSubmit={handleLinkToCase} className="space-y-4 text-sm">
              <div>
                <label className="block text-zinc-300 font-medium mb-1">Select Case *</label>
                {cases.length > 0 ? (
                  <select
                    required
                    value={selectedCaseId}
                    onChange={(e) => setSelectedCaseId(e.target.value)}
                    className="w-full bg-zinc-950 border border-zinc-800 rounded-lg p-2.5 text-zinc-200 focus:outline-none focus:border-emerald-500 font-mono text-xs"
                  >
                    {cases.map((c) => (
                      <option key={c.id} value={c.id}>
                        {c.case_number || c.id} &ndash; {c.title}
                      </option>
                    ))}
                  </select>
                ) : (
                  <p className="text-xs text-amber-400">No cases found. Create a case first.</p>
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
                  disabled={!selectedCaseId || isAdding || cases.length === 0}
                  className="px-4 py-2 bg-emerald-600 hover:bg-emerald-500 text-zinc-950 font-bold rounded-lg transition-all disabled:opacity-50"
                >
                  {isAdding ? "Bookmarking..." : "Bookmark Evidence"}
                </button>
              </div>
            </form>
          </div>
        </div>
      )}
    </div>
  );
}
