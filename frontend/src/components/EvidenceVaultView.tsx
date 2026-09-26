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
  videoId?: string;
  onInvestigateVideo?: (videoId: string) => void;
  onSeek?: (timestamp: number) => void;
}

function formatTime(seconds: number): string {
  const m = Math.floor(seconds / 60);
  const s = (seconds % 60).toFixed(1);
  return `${m.toString().padStart(2, "0")}:${s.padStart(4, "0")}`;
}

export default function EvidenceVaultView({ videoId, onInvestigateVideo, onSeek }: EvidenceVaultViewProps) {
  const [evidenceList, setEvidenceList] = useState<GlobalEvidenceItem[]>([]);
  const [videos, setVideos] = useState<VideoListItem[]>([]);
  const [cases, setCases] = useState<Case[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Filters
  const [selectedVideo, setSelectedVideo] = useState(videoId || "all");
  const [selectedClass, setSelectedClass] = useState("all");
  const [selectedStatus, setSelectedStatus] = useState("VALID");
  const [minConfidence, setMinConfidence] = useState(0);
  const [searchQuery, setSearchQuery] = useState("");

  useEffect(() => {
    if (videoId) {
      setSelectedVideo(videoId);
    }
  }, [videoId]);

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
  const [totalEvidence, setTotalEvidence] = useState<number>(0);

  const activeVideoId = videoId || (selectedVideo !== "all" ? selectedVideo : undefined);

  const fetchData = useCallback(async () => {
    setLoading(true);
    setError(null);
    setFailedImages({});
    try {
      const [evRes, vidRes, caseRes] = await Promise.all([
        listAllEvidence({
          video_id: activeVideoId,
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
      setTotalEvidence(evRes.total ?? evRes.evidence?.length ?? 0);
      setVideos(vidRes.videos || []);
      setCases(caseRes.cases || []);
      if (caseRes.cases?.length > 0 && !selectedCaseId) {
        setSelectedCaseId(caseRes.cases[0].id);
      }
    } catch (err: any) {
      setError(err.message || "Failed to load evidence.");
    } finally {
      setLoading(false);
    }
  }, [activeVideoId, selectedClass, selectedStatus, minConfidence, searchQuery, selectedCaseId]);

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
    <div className="p-4 md:p-6 space-y-6 max-w-[1600px] mx-auto min-h-full">
      {/* Header */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 border-b border-[#2A3038] pb-5">
        <div>
          <div className="flex items-center gap-3">
            <h2 className="text-2xl md:text-3xl font-bold tracking-tight text-[#F5F7FA]">
              {videoId ? "Evidence" : "Evidence Vault"}
            </h2>
            <span className="px-3 py-1 rounded-full text-xs md:text-sm font-sans font-medium bg-[#19B89A]/15 text-[#19B89A] border border-[#19B89A]/30">
              {videoId ? `${totalEvidence} Preserved Investigation Artifacts` : `${totalEvidence} Platform-wide Preserved Artifacts`}
            </span>
          </div>
          <p className="text-sm md:text-base text-[#A7AFBA] mt-1.5">
            {videoId
              ? "Preserved evidence from this investigation"
              : "Preserved forensic snapshots, annotated frames, video clips, and chain-of-custody artifacts."}
          </p>
        </div>

        <div className="flex items-center gap-2 px-3.5 py-2 rounded-lg bg-[#171A20] border border-[#2A3038] text-xs md:text-sm text-[#737C87]">
          <span className="w-2 h-2 rounded-full bg-[#19B89A]" />
          <span>Chain of Custody &bull; Preserved Frames</span>
        </div>
      </div>

      {/* Filter Controls */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4 bg-[#171A20] p-5 rounded-xl border border-[#2A3038]">
        {!videoId && (
          <div>
            <label className="block text-xs font-semibold uppercase tracking-wider text-[#737C87] mb-1.5">Source Video</label>
            <select
              value={selectedVideo}
              onChange={(e) => setSelectedVideo(e.target.value)}
              className="w-full bg-[#1D2128] border border-[#2A3038] rounded-lg p-2.5 text-sm text-[#F5F7FA] focus:outline-none focus:border-[#19B89A]"
            >
              <option value="all">Platform-wide / All Videos ({videos.length})</option>
              {videos.map((v) => (
                <option key={v.id} value={v.id}>
                  {v.id === "0d4d92f9-19f8-42e3-925f-1931cb557705" ? `★ ${v.filename} (Canonical Benchmark)` : v.filename}
                </option>
              ))}
            </select>
          </div>
        )}

        <div>
          <label className="block text-xs font-semibold uppercase tracking-wider text-[#737C87] mb-1.5">Object Class</label>
          <select
            value={selectedClass}
            onChange={(e) => setSelectedClass(e.target.value)}
            className="w-full bg-[#1D2128] border border-[#2A3038] rounded-lg p-2.5 text-sm text-[#F5F7FA] focus:outline-none focus:border-[#19B89A]"
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
          <label className="block text-xs font-semibold uppercase tracking-wider text-[#737C87] mb-1.5">Validation Status</label>
          <select
            value={selectedStatus}
            onChange={(e) => setSelectedStatus(e.target.value)}
            className="w-full bg-[#1D2128] border border-[#2A3038] rounded-lg p-2.5 text-sm text-[#F5F7FA] focus:outline-none focus:border-[#19B89A]"
          >
            <option value="VALID">Validated (Forensic)</option>
            <option value="all">All Records</option>
            <option value="REJECTED">Rejected (Filtered)</option>
            <option value="UNCERTAIN">Uncertain (Review)</option>
          </select>
        </div>

        <div>
          <label className="block text-xs font-semibold uppercase tracking-wider text-[#737C87] mb-1.5">Search Notes</label>
          <input
            type="text"
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            placeholder="Search notes or labels..."
            className="w-full bg-[#1D2128] border border-[#2A3038] rounded-lg p-2.5 text-sm text-[#F5F7FA] placeholder-[#737C87] focus:outline-none focus:border-[#19B89A]"
          />
        </div>
      </div>

      {/* Loading & Error States */}
      {loading && (
        <div className="flex flex-col items-center justify-center py-20 text-[#737C87] space-y-3">
          <div className="w-6 h-6 border-2 border-[#19B89A] border-t-transparent rounded-full animate-spin" />
          <span className="text-sm">Loading preserved evidence vault...</span>
        </div>
      )}

      {error && !loading && (
        <div className="p-4 bg-rose-500/10 border border-rose-500/30 rounded-xl text-rose-300 flex items-center justify-between text-xs">
          <span>{error}</span>
          <button onClick={fetchData} className="px-3 py-1 bg-rose-500/20 text-rose-200 rounded cursor-pointer">
            Retry
          </button>
        </div>
      )}

      {/* Evidence Items Grid: 2-column responsive layout making Evidence a visual centerpiece */}
      {!loading && !error && evidenceList.length > 0 && (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {evidenceList.map((item) => {
            const confPct = Math.round(item.confidence * 100);
            const isReviewRequired = item.validation_status !== "VALID" || item.confidence <= 0.65;
            const findingTitle = item.notes && item.notes.length > 3
              ? item.notes
              : item.object_class
              ? `${item.object_class.charAt(0).toUpperCase() + item.object_class.slice(1)} finding`
              : "Preserved forensic finding";

            return (
              <div
                key={item.id}
                className="bg-[#171A20] border border-[#2A3038] hover:border-[#3A424E] rounded-xl overflow-hidden transition-all shadow-md flex flex-col justify-between group"
              >
                {/* Large Media Thumbnail */}
                <div
                  onClick={() => {
                    setPreviewItem(item);
                    setPreviewMode(item.has_annotated ? "annotated" : "snapshot");
                  }}
                  className="relative aspect-[16/10] bg-[#0F1115] flex items-center justify-center overflow-hidden cursor-pointer"
                >
                  {!failedImages[item.id] && item.has_annotated && item.annotated_url ? (
                    <img
                      src={item.annotated_url}
                      alt={item.object_class || "Evidence"}
                      className="w-full h-full object-cover group-hover:scale-[1.02] transition-transform duration-300"
                      loading="lazy"
                      onError={() => setFailedImages((prev) => ({ ...prev, [item.id]: true }))}
                    />
                  ) : !failedImages[item.id] && item.has_snapshot && item.snapshot_url ? (
                    <img
                      src={item.snapshot_url}
                      alt={item.object_class || "Evidence"}
                      className="w-full h-full object-cover group-hover:scale-[1.02] transition-transform duration-300"
                      loading="lazy"
                      onError={() => setFailedImages((prev) => ({ ...prev, [item.id]: true }))}
                    />
                  ) : (
                    <div className="text-[#737C87] flex flex-col items-center justify-center p-8 space-y-2">
                      <svg className="w-10 h-10 text-[#737C87]" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
                      </svg>
                      <span className="text-xs text-[#737C87]">
                        {item.has_clip ? "Forensic Video Clip Available" : "Preserved Forensic Frame"}
                      </span>
                    </div>
                  )}

                  {/* Overlays */}
                  <div className="absolute top-3 left-3 flex gap-2">
                    <span className="px-3 py-1 rounded-md text-xs font-semibold uppercase tracking-wider bg-black/80 text-[#F5F7FA] border border-[#2A3038] backdrop-blur-sm">
                      {item.object_class || "Artifact"}
                    </span>
                    {item.has_clip && (
                      <span className="px-2.5 py-1 rounded-md text-xs font-semibold bg-[#19B89A]/90 text-[#0F1115] shadow-sm">
                        VIDEO CLIP
                      </span>
                    )}
                  </div>

                  <div className="absolute bottom-3 right-3 flex items-center gap-2">
                    {onSeek && (
                      <button
                        onClick={(e) => {
                          e.stopPropagation();
                          onSeek(item.timestamp_seconds);
                        }}
                        className="px-2.5 py-1 rounded-md text-xs font-medium bg-[#1D2128]/90 hover:bg-[#2A3038] text-[#F5F7FA] border border-[#2A3038] transition-colors cursor-pointer"
                        title="Seek player to this frame"
                      >
                        Seek &rarr;
                      </button>
                    )}
                    <span className="px-3 py-1 rounded-md text-xs font-mono font-bold bg-black/90 text-[#19B89A] border border-[#19B89A]/30">
                      {formatTime(item.timestamp_seconds)}
                    </span>
                  </div>
                </div>

                {/* Forensic Card Body */}
                <div className="p-6 space-y-4 flex-1 flex flex-col justify-between">
                  <div className="space-y-3">
                    <div className="flex flex-col sm:flex-row sm:items-start justify-between gap-2">
                      <h4 className="text-lg font-bold text-[#F5F7FA] tracking-tight leading-snug line-clamp-2" title={findingTitle}>
                        {findingTitle}
                      </h4>
                      <div className="flex flex-wrap items-center gap-1.5 shrink-0">
                        <span className="text-[11px] font-semibold px-2.5 py-0.5 rounded-full bg-emerald-500/15 text-emerald-400 border border-emerald-500/30 uppercase tracking-wider">
                          {item.validation_status === "VALID" ? "Evidence: Validated" : "Evidence: Preserved"}
                        </span>
                        <span
                          className={`text-[11px] font-semibold px-2.5 py-0.5 rounded-full ${
                            isReviewRequired
                              ? "bg-amber-500/15 text-amber-400 border border-amber-500/30"
                              : "bg-[#19B89A]/15 text-[#19B89A] border border-[#19B89A]/30"
                          }`}
                        >
                          {isReviewRequired ? `Final Assessment: ${confPct}% — REVIEW_REQUIRED` : `Final Assessment: ${confPct}%`}
                        </span>
                      </div>
                    </div>

                    <div className="space-y-1 text-sm text-[#A7AFBA]">
                      <div className="flex items-center gap-2">
                        <span className="text-[#737C87]">Source video:</span>
                        <span className="font-mono text-[#F5F7FA] truncate max-w-[260px]" title={item.source_video_name}>
                          {item.source_video_name || "Active footage"}
                        </span>
                      </div>
                      <div className="flex items-center gap-2">
                        <span className="text-[#737C87]">Timestamp:</span>
                        <span className="font-mono text-[#19B89A]">{formatTime(item.timestamp_seconds)} ({item.timestamp_seconds.toFixed(2)}s)</span>
                      </div>
                    </div>
                  </div>

                  {/* Actions */}
                  <div className="pt-4 border-t border-[#2A3038] flex items-center justify-between gap-3">
                    <button
                      onClick={() => {
                        setPreviewItem(item);
                        setPreviewMode(item.has_annotated ? "annotated" : "snapshot");
                      }}
                      className="px-5 py-2.5 bg-[#1D2128] hover:bg-[#2A3038] text-[#F5F7FA] rounded-lg border border-[#2A3038] font-semibold text-sm cursor-pointer transition-colors"
                    >
                      Open evidence &rarr;
                    </button>

                    <button
                      onClick={() => {
                        setTargetEvidence(item);
                        setIsAddCaseOpen(true);
                      }}
                      className="px-4 py-2.5 bg-[#19B89A] hover:bg-[#16A489] text-[#0F1115] rounded-lg font-bold text-sm cursor-pointer transition-colors shadow-sm"
                    >
                      + Add to case
                    </button>
                  </div>
                </div>
              </div>
            );
          })}
        </div>
      )}

      {/* Empty State */}
      {!loading && !error && evidenceList.length === 0 && (
        <div className="rounded-xl border border-[#2A3038] bg-[#171A20] p-12 text-center space-y-3">
          <h3 className="text-base font-semibold text-[#F5F7FA]">No evidence artifacts found</h3>
          <p className="text-xs text-[#737C87] max-w-md mx-auto">
            Evidence artifacts appear when Sentinel identifies and preserves critical scene moments from surveillance video.
          </p>
        </div>
      )}

      {/* Full Preview Modal */}
      {previewItem && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/80 backdrop-blur-sm p-4">
          <div className="bg-[#171A20] border border-[#2A3038] rounded-xl max-w-3xl w-full p-6 space-y-4 shadow-2xl">
            <div className="flex items-center justify-between border-b border-[#2A3038] pb-3">
              <div>
                <h3 className="text-base font-bold text-[#F5F7FA]">
                  Evidence Artifact: {previewItem.object_class} ({(previewItem.confidence * 100).toFixed(0)}%)
                </h3>
                <p className="text-xs text-[#737C87]">
                  {previewItem.source_video_name} &bull; Timestamp: {formatTime(previewItem.timestamp_seconds)}
                </p>
              </div>
              <button
                onClick={() => setPreviewItem(null)}
                className="text-[#737C87] hover:text-[#F5F7FA] text-lg cursor-pointer"
              >
                ✕
              </button>
            </div>

            {/* Media Area */}
            <div className="bg-[#0F1115] rounded-lg overflow-hidden flex items-center justify-center aspect-video border border-[#2A3038]">
              {previewMode === "clip" && (previewItem.playback_url || previewItem.clip_url) ? (
                <video
                  key={previewItem.evidence_id || previewItem.id}
                  controls
                  autoPlay
                  playsInline
                  className="w-full h-full object-contain"
                >
                  <source src={(previewItem.playback_url || previewItem.clip_url) ?? undefined} type="video/mp4" />
                  {previewItem.clip_url && previewItem.playback_url && (
                    <source src={previewItem.clip_url} type="video/mp4" />
                  )}
                  Your browser does not support HTML5 video playback.
                </video>
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
                <div className="text-[#737C87] flex flex-col items-center justify-center p-6 text-xs">
                  <span>Artifact Unavailable</span>
                </div>
              )}
            </div>

            {/* Mode toggles */}
            <div className="flex items-center justify-between pt-2">
              <div className="flex items-center gap-2">
                {previewItem.has_annotated && (
                  <button
                    onClick={() => setPreviewMode("annotated")}
                    className={`px-3 py-1.5 rounded-lg text-xs font-medium cursor-pointer transition-colors ${
                      previewMode === "annotated" ? "bg-[#19B89A] text-[#0F1115] font-semibold" : "bg-[#1D2128] text-[#A7AFBA]"
                    }`}
                  >
                    Annotated
                  </button>
                )}
                {previewItem.has_snapshot && (
                  <button
                    onClick={() => setPreviewMode("snapshot")}
                    className={`px-3 py-1.5 rounded-lg text-xs font-medium cursor-pointer transition-colors ${
                      previewMode === "snapshot" ? "bg-[#19B89A] text-[#0F1115] font-semibold" : "bg-[#1D2128] text-[#A7AFBA]"
                    }`}
                  >
                    Snapshot
                  </button>
                )}
                {previewItem.has_clip && (
                  <button
                    onClick={() => setPreviewMode("clip")}
                    className={`px-3 py-1.5 rounded-lg text-xs font-medium cursor-pointer transition-colors ${
                      previewMode === "clip" ? "bg-[#19B89A] text-[#0F1115] font-semibold" : "bg-[#1D2128] text-[#A7AFBA]"
                    }`}
                  >
                    Video Clip
                  </button>
                )}
              </div>

              <button
                onClick={() => setPreviewItem(null)}
                className="px-4 py-1.5 bg-[#1D2128] hover:bg-[#2A3038] text-[#F5F7FA] rounded-lg text-xs font-medium cursor-pointer"
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
          <div className="bg-[#171A20] border border-[#2A3038] rounded-xl max-w-md w-full p-6 space-y-4 shadow-xl">
            <div className="flex items-center justify-between border-b border-[#2A3038] pb-3">
              <h3 className="text-base font-bold text-[#F5F7FA]">Add Evidence to Case</h3>
              <button
                onClick={() => setIsAddCaseOpen(false)}
                className="text-[#737C87] hover:text-[#F5F7FA] text-lg cursor-pointer"
              >
                ✕
              </button>
            </div>

            <form onSubmit={handleLinkToCase} className="space-y-4">
              <div>
                <label className="block text-xs font-medium text-[#737C87] mb-1">Select Case Dossier</label>
                <select
                  value={selectedCaseId}
                  onChange={(e) => setSelectedCaseId(e.target.value)}
                  className="w-full bg-[#1D2128] border border-[#2A3038] rounded-lg p-2.5 text-xs text-[#F5F7FA] focus:outline-none focus:border-[#19B89A]"
                >
                  {cases.map((c) => (
                    <option key={c.id} value={c.id}>
                      {c.case_number || c.id} — {c.title}
                    </option>
                  ))}
                </select>
              </div>

              {addFeedback && (
                <div className="text-xs p-2 rounded bg-[#1D2128] text-[#19B89A]">
                  {addFeedback}
                </div>
              )}

              <div className="flex justify-end gap-2 pt-2">
                <button
                  type="button"
                  onClick={() => setIsAddCaseOpen(false)}
                  className="px-4 py-2 bg-[#1D2128] hover:bg-[#2A3038] text-[#F5F7FA] text-xs rounded-lg cursor-pointer"
                >
                  Cancel
                </button>
                <button
                  type="submit"
                  disabled={isAdding}
                  className="px-4 py-2 bg-[#19B89A] hover:bg-[#16A489] text-[#0F1115] font-semibold text-xs rounded-lg cursor-pointer transition-colors"
                >
                  {isAdding ? "Saving..." : "Save to Case"}
                </button>
              </div>
            </form>
          </div>
        </div>
      )}
    </div>
  );
}
