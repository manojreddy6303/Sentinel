"use client";

import React, { useState, useEffect, useCallback, useRef } from "react";
import {
  Case,
  CaseTimelineItem,
  CaseBookmark,
  CaseNote,
  CaseAnnotation,
  CaseActivity,
  IncidentReplayContext,
  IncidentExplanation,
  CaseTopologyData,
  getCase,
  updateCase,
  getCaseTimeline,
  listCaseBookmarks,
  createCaseBookmark,
  deleteCaseBookmark,
  listCaseNotes,
  createCaseNote,
  deleteCaseNote,
  listCaseAnnotations,
  createCaseAnnotation,
  deleteCaseAnnotation,
  getIncidentReplayContext,
  getIncidentFocusData,
  getIncidentExplanation,
  getCaseTopology,
  getCaseStoryline,
  getCaseActivities,
  exportCaseData,
  linkVideoToCase,
  unlinkVideoFromCase,
  linkCameraToCase,
  unlinkCameraFromCase,
  linkIncidentToCase,
  unlinkIncidentFromCase,
  linkEvidenceToCase,
  unlinkEvidenceFromCase,
  listVideos,
  listAllCameras,
  listAllIncidents,
  listAllEvidence,
  VideoListItem,
  GlobalCameraItem,
  GlobalIncidentItem,
  GlobalEvidenceItem,
  API_BASE_URL,
} from "@/lib/api";

interface CaseWorkspaceViewProps {
  caseId: string;
  onBack: () => void;
  onSelectCase?: (caseId: string) => void;
  onUploadVideoForCase?: () => void;
  onInvestigateVideo?: (videoId: string) => void;
}

type WorkspaceTab =
  | "overview"
  | "timeline"
  | "focus"
  | "multicamera"
  | "bookmarks"
  | "notes"
  | "storyline";

export default function CaseWorkspaceView({
  caseId,
  onBack,
  onSelectCase,
  onUploadVideoForCase,
  onInvestigateVideo,
}: CaseWorkspaceViewProps) {
  const [caseData, setCaseData] = useState<Case | null>(null);
  const [activeTab, setActiveTab] = useState<WorkspaceTab>("overview");
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Timeline
  const [timelineItems, setTimelineItems] = useState<CaseTimelineItem[]>([]);
  const [timelineLoading, setTimelineLoading] = useState(false);
  const [activeLayerFilter, setActiveLayerFilter] = useState<string>("");

  // Focus & Replay
  const [selectedIncidentId, setSelectedIncidentId] = useState<string | null>(null);
  const [replayContext, setReplayContext] = useState<IncidentReplayContext | null>(null);
  const [explanation, setExplanation] = useState<IncidentExplanation | null>(null);
  const [focusLoading, setFocusLoading] = useState(false);
  const [preRoll, setPreRoll] = useState<number>(5.0);
  const [postRoll, setPostRoll] = useState<number>(5.0);

  // Topology
  const [topology, setTopology] = useState<CaseTopologyData | null>(null);
  const [syncTime, setSyncTime] = useState<number>(0);
  const [isSyncPlaying, setIsSyncPlaying] = useState<boolean>(false);

  // Bookmarks, Notes, Annotations
  const [bookmarks, setBookmarks] = useState<CaseBookmark[]>([]);
  const [notes, setNotes] = useState<CaseNote[]>([]);
  const [annotations, setAnnotations] = useState<CaseAnnotation[]>([]);
  const [activities, setActivities] = useState<CaseActivity[]>([]);
  const [storyline, setStoryline] = useState<any[]>([]);

  // Modals & Inputs
  const [isAddBookmarkOpen, setIsAddBookmarkOpen] = useState(false);
  const [bmTitle, setBmTitle] = useState("");
  const [bmTime, setBmTime] = useState(0);
  const [bmVideoId, setBmVideoId] = useState("");

  const [newNoteText, setNewNoteText] = useState("");
  const [newNoteType, setNewNoteType] = useState<any>("CASE");

  const [isLinkEntityOpen, setIsLinkEntityOpen] = useState(false);
  const [entityInputType, setEntityInputType] = useState<"video" | "camera" | "incident" | "evidence">("video");
  const [availableVideos, setAvailableVideos] = useState<VideoListItem[]>([]);
  const [availableCameras, setAvailableCameras] = useState<GlobalCameraItem[]>([]);
  const [availableIncidents, setAvailableIncidents] = useState<GlobalIncidentItem[]>([]);
  const [availableEvidence, setAvailableEvidence] = useState<GlobalEvidenceItem[]>([]);
  const [entitySearchQuery, setEntitySearchQuery] = useState("");
  const [loadingEntities, setLoadingEntities] = useState(false);
  const [linkingActionId, setLinkingActionId] = useState<string | null>(null);
  const [linkError, setLinkError] = useState<string | null>(null);

  // Video playback ref
  const videoRef = useRef<HTMLVideoElement | null>(null);

  const loadCase = useCallback(async () => {
    setLoading(true);
    try {
      const data = await getCase(caseId);
      setCaseData(data);
      if (data.linked_videos && data.linked_videos.length > 0) {
        setBmVideoId(data.linked_videos[0].video_id);
      }
    } catch (err: any) {
      setError(err.message || "Failed to load case.");
    } finally {
      setLoading(false);
    }
  }, [caseId]);

  const loadTimeline = useCallback(async () => {
    setTimelineLoading(true);
    try {
      const res = await getCaseTimeline(caseId, {
        layers: activeLayerFilter || undefined,
        limit: 200,
      });
      setTimelineItems(res.timeline || []);
    } catch (err) {
      console.error(err);
    } finally {
      setTimelineLoading(false);
    }
  }, [caseId, activeLayerFilter]);

  const loadBookmarks = useCallback(async () => {
    try {
      const res = await listCaseBookmarks(caseId);
      setBookmarks(res.bookmarks || []);
    } catch (err) {
      console.error(err);
    }
  }, [caseId]);

  const loadNotes = useCallback(async () => {
    try {
      const res = await listCaseNotes(caseId);
      setNotes(res.notes || []);
    } catch (err) {
      console.error(err);
    }
  }, [caseId]);

  const loadAnnotations = useCallback(async () => {
    try {
      const res = await listCaseAnnotations(caseId);
      setAnnotations(res.annotations || []);
    } catch (err) {
      console.error(err);
    }
  }, [caseId]);

  const loadTopology = useCallback(async () => {
    try {
      const res = await getCaseTopology(caseId);
      setTopology(res);
    } catch (err) {
      console.error(err);
    }
  }, [caseId]);

  const loadStoryline = useCallback(async () => {
    try {
      const res = await getCaseStoryline(caseId);
      setStoryline(res.storyline || []);
    } catch (err) {
      console.error(err);
    }
  }, [caseId]);

  const loadActivities = useCallback(async () => {
    try {
      const res = await getCaseActivities(caseId, 50);
      setActivities(res.activities || []);
    } catch (err) {
      console.error(err);
    }
  }, [caseId]);

  useEffect(() => {
    loadCase();
    loadBookmarks();
    loadNotes();
    loadAnnotations();
    loadActivities();
  }, [loadCase, loadBookmarks, loadNotes, loadAnnotations, loadActivities]);

  useEffect(() => {
    if (activeTab === "timeline") loadTimeline();
    if (activeTab === "multicamera") loadTopology();
    if (activeTab === "storyline") loadStoryline();
  }, [activeTab, loadTimeline, loadTopology, loadStoryline]);

  const handleStatusChange = async (newStatus: any) => {
    try {
      const updated = await updateCase(caseId, { status: newStatus });
      setCaseData(updated);
      await loadActivities();
    } catch (err: any) {
      alert(`Status update failed: ${err.message}`);
    }
  };

  const handleExport = async () => {
    try {
      const data = await exportCaseData(caseId);
      const blob = new Blob([JSON.stringify(data, null, 2)], { type: "application/json" });
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `${caseData?.case_number || "case"}_forensic_export.json`;
      a.click();
      URL.revokeObjectURL(url);
      await loadActivities();
    } catch (err: any) {
      alert(`Export failed: ${err.message}`);
    }
  };

  const handleSelectIncident = async (incId: string) => {
    setSelectedIncidentId(incId);
    setFocusLoading(true);
    try {
      const [rep, exp] = await Promise.all([
        getIncidentReplayContext(caseId, incId, preRoll, postRoll),
        getIncidentExplanation(caseId, incId),
      ]);
      setReplayContext(rep);
      setExplanation(exp);
      if (videoRef.current && rep.replay_context) {
        videoRef.current.currentTime = rep.replay_context.replay_start;
      }
    } catch (err: any) {
      console.error("Focus load error:", err);
    } finally {
      setFocusLoading(false);
    }
  };

  const handleCreateBookmark = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!bmTitle.trim() || !bmVideoId) return;
    try {
      await createCaseBookmark(caseId, {
        video_id: bmVideoId,
        timestamp_seconds: bmTime,
        title: bmTitle.trim(),
        author: caseData?.assigned_investigator || "Investigator",
      });
      setIsAddBookmarkOpen(false);
      setBmTitle("");
      await loadBookmarks();
      await loadActivities();
    } catch (err: any) {
      alert(`Create bookmark failed: ${err.message}`);
    }
  };

  const handleCreateNote = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!newNoteText.trim()) return;
    try {
      await createCaseNote(caseId, {
        content: newNoteText.trim(),
        author: caseData?.assigned_investigator || "Investigator",
        associated_type: newNoteType,
      });
      setNewNoteText("");
      await loadNotes();
      await loadActivities();
    } catch (err: any) {
      alert(`Note creation failed: ${err.message}`);
    }
  };

  const loadAvailableEntities = useCallback(async () => {
    setLoadingEntities(true);
    setLinkError(null);
    try {
      const [vRes, cRes, iRes, eRes] = await Promise.all([
        listVideos({ limit: 100 }),
        listAllCameras({ limit: 100 }),
        listAllIncidents({ limit: 100 }),
        listAllEvidence({ limit: 100 }),
      ]);
      setAvailableVideos(vRes.videos || []);
      setAvailableCameras(cRes.cameras || []);
      setAvailableIncidents(iRes.incidents || []);
      setAvailableEvidence(eRes.evidence || []);
    } catch (err: any) {
      setLinkError(err.message || "Failed to load entities.");
    } finally {
      setLoadingEntities(false);
    }
  }, []);

  useEffect(() => {
    if (isLinkEntityOpen) {
      loadAvailableEntities();
    }
  }, [isLinkEntityOpen, loadAvailableEntities]);

  const handleLinkSpecificEntity = async (type: "video" | "camera" | "incident" | "evidence", id: string) => {
    setLinkingActionId(id);
    setLinkError(null);
    try {
      if (type === "video") {
        await linkVideoToCase(caseId, id);
      } else if (type === "camera") {
        await linkCameraToCase(caseId, id);
      } else if (type === "incident") {
        await linkIncidentToCase(caseId, id);
      } else if (type === "evidence") {
        await linkEvidenceToCase(caseId, id);
      }
      await loadCase();
      await loadActivities();
      setIsLinkEntityOpen(false);
    } catch (err: any) {
      setLinkError(`Linking failed: ${err.message}`);
    } finally {
      setLinkingActionId(null);
    }
  };

  const handleUnlinkSpecificEntity = async (type: "video" | "camera" | "incident" | "evidence", id: string) => {
    try {
      if (type === "video") {
        await unlinkVideoFromCase(caseId, id);
      } else if (type === "camera") {
        await unlinkCameraFromCase(caseId, id);
      } else if (type === "incident") {
        await unlinkIncidentFromCase(caseId, id);
      } else if (type === "evidence") {
        await unlinkEvidenceFromCase(caseId, id);
      }
      await loadCase();
      await loadActivities();
    } catch (err: any) {
      alert(`Unlink failed: ${err.message}`);
    }
  };

  if (loading && !caseData) {
    return (
      <div className="flex flex-col items-center justify-center h-full text-zinc-500 space-y-2">
        <div className="w-6 h-6 border-2 border-emerald-500 border-t-transparent rounded-full animate-spin" />
        <span className="text-xs font-mono">Loading case workspace...</span>
      </div>
    );
  }

  if (error || !caseData) {
    return (
      <div className="p-6 text-center space-y-4">
        <div className="text-rose-400 font-semibold">{error || "Case not found."}</div>
        <button
          onClick={onBack}
          className="px-3.5 py-1.5 bg-zinc-800 hover:bg-zinc-700 text-zinc-200 text-xs rounded-lg cursor-pointer"
        >
          &larr; Back to Cases
        </button>
      </div>
    );
  }

  return (
    <div className="w-full h-full flex flex-col space-y-3 overflow-hidden">
      {/* Case Header Banner */}
      <div className="bg-zinc-900/90 border border-zinc-800 rounded-xl p-3.5 flex flex-col md:flex-row items-start md:items-center justify-between gap-3 shadow-md shrink-0">
        <div className="flex items-center gap-3">
          <button
            onClick={onBack}
            className="p-1.5 bg-zinc-800 hover:bg-zinc-700 text-zinc-300 rounded-lg transition-colors cursor-pointer text-xs"
            title="Back to Case List"
          >
            &larr; Cases
          </button>

          <div>
            <div className="flex items-center gap-2">
              <span className="font-mono text-xs font-bold text-emerald-400">
                {caseData.case_number}
              </span>
              <span className="text-xs text-zinc-600">•</span>
              <h2 className="text-base font-bold text-white tracking-wide">{caseData.title}</h2>
            </div>
            <div className="flex items-center gap-2.5 text-[11px] text-zinc-400 mt-0.5 font-mono">
              <span>Investigator: {caseData.assigned_investigator || "Unassigned"}</span>
              <span>•</span>
              <span>Priority: {caseData.priority}</span>
              {caseData.tags && caseData.tags.length > 0 && (
                <>
                  <span>•</span>
                  <span>Tags: {caseData.tags.join(", ")}</span>
                </>
              )}
            </div>
          </div>
        </div>

        {/* Status selector & Export button */}
        <div className="flex items-center gap-2.5 self-end md:self-auto">
          <div className="flex items-center gap-1.5 bg-zinc-950 border border-zinc-800 rounded-lg px-2.5 py-1 text-xs">
            <span className="text-zinc-500 font-mono text-[10px] uppercase">Status:</span>
            <select
              value={caseData.status}
              onChange={(e) => handleStatusChange(e.target.value)}
              className="bg-transparent text-emerald-400 font-mono font-semibold text-xs focus:outline-none cursor-pointer"
            >
              <option value="OPEN" className="bg-zinc-900 text-emerald-400">OPEN</option>
              <option value="INVESTIGATING" className="bg-zinc-900 text-indigo-400">INVESTIGATING</option>
              <option value="REVIEW" className="bg-zinc-900 text-purple-400">REVIEW</option>
              <option value="CLOSED" className="bg-zinc-900 text-zinc-400">CLOSED</option>
            </select>
          </div>

          {onUploadVideoForCase && (
            <button
              onClick={onUploadVideoForCase}
              className="flex items-center gap-1.5 px-3 py-1.5 bg-emerald-600 hover:bg-emerald-500 text-zinc-950 font-bold text-xs rounded-lg transition-colors cursor-pointer shadow-sm"
              title="Upload surveillance video and automatically associate with this case"
            >
              <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-8l-4-4m0 0L8 8m4-4v12" />
              </svg>
              <span>+ Upload Video</span>
            </button>
          )}

          <button
            onClick={() => {
              setEntityInputType("video");
              setIsLinkEntityOpen(true);
            }}
            className="px-2.5 py-1.5 bg-zinc-800 hover:bg-zinc-700 text-zinc-200 text-xs rounded-lg font-mono transition-colors cursor-pointer"
          >
            Link Existing Footage
          </button>

          <button
            onClick={handleExport}
            className="flex items-center gap-1.5 px-3 py-1.5 bg-zinc-800 hover:bg-zinc-700 text-zinc-200 font-semibold text-xs rounded-lg transition-colors cursor-pointer border border-zinc-700"
          >
            <span>⬇</span> Export Case
          </button>
        </div>
      </div>

      {/* Navigation Sub-Tabs */}
      <div className="flex items-center gap-1 border-b border-zinc-800/80 shrink-0 overflow-x-auto text-xs font-mono">
        {[
          { id: "overview", label: "Overview & Entities", icon: "📊" },
          { id: "timeline", label: "Unified Timeline", icon: "⏱️" },
          { id: "focus", label: "Focus & Replay", icon: "🎯" },
          { id: "multicamera", label: "Multi-Camera & Topology", icon: "🎥" },
          { id: "bookmarks", label: `Bookmarks (${bookmarks.length})`, icon: "📌" },
          { id: "notes", label: `Analyst Notes (${notes.length})`, icon: "📝" },
          { id: "storyline", label: "Storyline Notebook", icon: "📖" },
        ].map((t) => (
          <button
            key={t.id}
            onClick={() => setActiveTab(t.id as WorkspaceTab)}
            className={`flex items-center gap-1.5 px-3.5 py-2 border-b-2 font-semibold transition-all cursor-pointer whitespace-nowrap ${
              activeTab === t.id
                ? "border-emerald-500 text-emerald-400 bg-emerald-950/20"
                : "border-transparent text-zinc-400 hover:text-zinc-200 hover:bg-zinc-900/50"
            }`}
          >
            <span>{t.icon}</span>
            <span>{t.label}</span>
          </button>
        ))}
      </div>

      {/* Workspace Body */}
      <div className="flex-1 overflow-y-auto pr-1">
        {/* TAB 1: OVERVIEW */}
        {activeTab === "overview" && (
          <div className="space-y-4">
            {caseData.description && (
              <div className="bg-zinc-900/60 border border-zinc-800 rounded-xl p-3.5">
                <div className="text-[11px] font-mono text-zinc-400 uppercase tracking-wider mb-1">
                  Investigation Synopsis
                </div>
                <p className="text-xs text-zinc-300 leading-relaxed">{caseData.description}</p>
              </div>
            )}

            {/* Empty State Guidance: Case Created, Ready for Footage */}
            {(!caseData.linked_videos || caseData.linked_videos.length === 0) && (
              <div className="bg-gradient-to-r from-emerald-950/40 via-zinc-900/70 to-zinc-900/40 border border-emerald-800/40 rounded-xl p-5 shadow-lg space-y-4">
                <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
                  <div className="space-y-1">
                    <div className="flex items-center gap-2 text-xs font-mono text-emerald-400 font-bold uppercase tracking-wider">
                      <span className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse" />
                      Case Initialized &bull; Ready for Surveillance Footage
                    </div>
                    <h3 className="text-base font-bold text-white">
                      Upload or Link Footage to Begin Security Analysis
                    </h3>
                    <p className="text-xs text-zinc-400 max-w-2xl leading-relaxed">
                      Every security investigation is grounded in surveillance video. You can upload new CCTV footage directly from your laptop or link surveillance footage that has already been ingested into Sentinel.
                    </p>
                  </div>

                  <div className="flex flex-wrap items-center gap-2.5 shrink-0">
                    {onUploadVideoForCase && (
                      <button
                        onClick={onUploadVideoForCase}
                        className="px-4 py-2.5 bg-emerald-600 hover:bg-emerald-500 text-zinc-950 font-bold text-xs rounded-lg transition-all shadow-md flex items-center gap-2 cursor-pointer"
                      >
                        <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-8l-4-4m0 0L8 8m4-4v12" />
                        </svg>
                        <span>+ Upload Video</span>
                      </button>
                    )}
                    <button
                      onClick={() => {
                        setEntityInputType("video");
                        setIsLinkEntityOpen(true);
                      }}
                      className="px-3.5 py-2.5 bg-zinc-800 hover:bg-zinc-700 text-zinc-200 text-xs font-semibold rounded-lg border border-zinc-700 transition-colors cursor-pointer flex items-center gap-1.5"
                    >
                      <span>🔗</span>
                      <span>Link Existing Ingested Video</span>
                    </button>
                  </div>
                </div>

                {/* Secondary quick-links */}
                <div className="pt-3 border-t border-zinc-800/80 flex flex-wrap items-center gap-3 text-xs text-zinc-400">
                  <span className="font-mono text-[11px] text-zinc-500">Other options:</span>
                  <button
                    onClick={() => {
                      setEntityInputType("incident");
                      setIsLinkEntityOpen(true);
                    }}
                    className="text-amber-400 hover:underline cursor-pointer flex items-center gap-1"
                  >
                    <span>⚠️</span> Link Existing Incident
                  </button>
                  <span>&bull;</span>
                  <button
                    onClick={() => {
                      setEntityInputType("camera");
                      setIsLinkEntityOpen(true);
                    }}
                    className="text-indigo-400 hover:underline cursor-pointer flex items-center gap-1"
                  >
                    <span>📹</span> Associate CCTV Camera
                  </button>
                  <span>&bull;</span>
                  <button
                    onClick={() => {
                      setEntityInputType("evidence");
                      setIsLinkEntityOpen(true);
                    }}
                    className="text-cyan-400 hover:underline cursor-pointer flex items-center gap-1"
                  >
                    <span>🛡️</span> Link Preserved Evidence
                  </button>
                </div>
              </div>
            )}

            {/* Explicitly Linked Entities: Videos, Cameras, Incidents, Evidence */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-3.5">
              {/* 1. Linked Videos */}
              <div className="bg-zinc-900/60 border border-zinc-800 rounded-xl p-3.5 space-y-2.5">
                <div className="flex items-center justify-between">
                  <h3 className="text-xs font-mono uppercase tracking-wider text-emerald-400 font-bold flex items-center gap-1.5">
                    <span>🎥</span> Linked Videos ({caseData.linked_videos?.length || 0})
                  </h3>
                  <div className="flex items-center gap-2">
                    {onUploadVideoForCase && (
                      <button
                        onClick={onUploadVideoForCase}
                        className="text-[11px] text-emerald-400 hover:underline cursor-pointer font-semibold"
                      >
                        + Upload Video
                      </button>
                    )}
                    <span className="text-zinc-600 text-xs">&bull;</span>
                    <button
                      onClick={() => {
                        setEntityInputType("video");
                        setIsLinkEntityOpen(true);
                      }}
                      className="text-[11px] text-zinc-400 hover:text-zinc-200 cursor-pointer"
                    >
                      Link Existing
                    </button>
                  </div>
                </div>

                {caseData.linked_videos && caseData.linked_videos.length > 0 ? (
                  <div className="space-y-2">
                    {caseData.linked_videos.map((v) => (
                      <div
                        key={v.video_id}
                        className="flex flex-col sm:flex-row sm:items-center justify-between p-2.5 bg-zinc-950/80 border border-zinc-800/80 rounded-lg text-xs gap-2"
                      >
                        <div className="min-w-0">
                          <div className="font-semibold text-zinc-200 truncate">{v.filename}</div>
                          <div className="text-[10px] text-zinc-500 font-mono flex items-center gap-2 mt-0.5">
                            <span>ID: {v.video_id.slice(0, 12)}...</span>
                            <span>&bull;</span>
                            <span>{v.duration_seconds}s @ {v.fps} FPS</span>
                          </div>
                        </div>
                        <div className="flex items-center gap-2 flex-none">
                          <span className="text-[10px] font-mono bg-emerald-950 text-emerald-400 px-1.5 py-0.5 rounded border border-emerald-800">
                            {v.status}
                          </span>
                          {onInvestigateVideo && (
                            <button
                              onClick={() => onInvestigateVideo(v.video_id)}
                              className="px-2 py-0.5 bg-zinc-800 hover:bg-emerald-600 hover:text-zinc-950 text-zinc-200 text-[11px] font-mono rounded border border-zinc-700 transition-all cursor-pointer"
                            >
                              Investigate &rarr;
                            </button>
                          )}
                          <button
                            onClick={() => handleUnlinkSpecificEntity("video", v.video_id)}
                            className="text-zinc-500 hover:text-red-400 text-xs px-1 cursor-pointer"
                            title="Unlink video from case"
                          >
                            &times;
                          </button>
                        </div>
                      </div>
                    ))}
                  </div>
                ) : (
                  <div className="text-xs text-zinc-500 italic py-3">
                    No surveillance footage linked yet. Use &ldquo;+ Upload Video&rdquo; to add footage from your laptop.
                  </div>
                )}
              </div>

              {/* 2. Linked Cameras */}
              <div className="bg-zinc-900/60 border border-zinc-800 rounded-xl p-3.5 space-y-2.5">
                <div className="flex items-center justify-between">
                  <h3 className="text-xs font-mono uppercase tracking-wider text-indigo-400 font-bold flex items-center gap-1.5">
                    <span>📹</span> Linked Cameras ({caseData.linked_cameras?.length || 0})
                  </h3>
                  <button
                    onClick={() => {
                      setEntityInputType("camera");
                      setIsLinkEntityOpen(true);
                    }}
                    className="text-[11px] text-indigo-400 hover:underline cursor-pointer"
                  >
                    + Link Camera
                  </button>
                </div>

                {caseData.linked_cameras && caseData.linked_cameras.length > 0 ? (
                  <div className="space-y-1.5">
                    {caseData.linked_cameras.map((c) => (
                      <div
                        key={c.camera_id}
                        className="flex items-center justify-between p-2 bg-zinc-950/80 border border-zinc-800/80 rounded-lg text-xs"
                      >
                        <div>
                          <div className="font-semibold text-zinc-200">{c.camera_label}</div>
                          <div className="text-[10px] text-zinc-500 font-mono">
                            Offset: {c.clock_offset_seconds}s &bull; {c.position_hint || "General area"}
                          </div>
                        </div>
                        <button
                          onClick={() => handleUnlinkSpecificEntity("camera", c.camera_id)}
                          className="text-zinc-500 hover:text-red-400 text-xs px-1 cursor-pointer"
                          title="Unlink camera"
                        >
                          &times;
                        </button>
                      </div>
                    ))}
                  </div>
                ) : (
                  <div className="text-xs text-zinc-500 italic py-3">No cameras linked yet.</div>
                )}
              </div>

              {/* 3. Linked Incidents */}
              <div className="bg-zinc-900/60 border border-zinc-800 rounded-xl p-3.5 space-y-2.5">
                <div className="flex items-center justify-between">
                  <h3 className="text-xs font-mono uppercase tracking-wider text-amber-400 font-bold flex items-center gap-1.5">
                    <span>⚠️</span> Linked Incidents ({caseData.linked_incidents?.length || 0})
                  </h3>
                  <button
                    onClick={() => {
                      setEntityInputType("incident");
                      setIsLinkEntityOpen(true);
                    }}
                    className="text-[11px] text-amber-400 hover:underline cursor-pointer"
                  >
                    + Link Incident
                  </button>
                </div>

                {caseData.linked_incidents && caseData.linked_incidents.length > 0 ? (
                  <div className="space-y-1.5">
                    {caseData.linked_incidents.map((inc) => (
                      <div
                        key={inc.incident_id}
                        className="flex items-center justify-between p-2 bg-zinc-950/80 border border-zinc-800/80 rounded-lg text-xs"
                      >
                        <div>
                          <div className="font-semibold text-zinc-200 flex items-center gap-2">
                            <span>{inc.incident_category || "Incident"}</span>
                            <span className="text-[10px] font-mono px-1 py-0.2 bg-amber-500/10 text-amber-400 rounded">
                              {inc.validation_decision || "REVIEW"}
                            </span>
                          </div>
                          <div className="text-[10px] text-zinc-500 font-mono">
                            ID: {inc.incident_id.slice(0, 12)}... &bull; {inc.start_time !== undefined && inc.end_time !== undefined ? `${inc.start_time.toFixed(1)}s - ${inc.end_time.toFixed(1)}s` : ""}
                          </div>
                        </div>
                        <button
                          onClick={() => handleUnlinkSpecificEntity("incident", inc.incident_id)}
                          className="text-zinc-500 hover:text-red-400 text-xs px-1 cursor-pointer"
                          title="Unlink incident"
                        >
                          &times;
                        </button>
                      </div>
                    ))}
                  </div>
                ) : (
                  <div className="text-xs text-zinc-500 italic py-3">No incidents linked yet.</div>
                )}
              </div>

              {/* 4. Linked Evidence */}
              <div className="bg-zinc-900/60 border border-zinc-800 rounded-xl p-3.5 space-y-2.5">
                <div className="flex items-center justify-between">
                  <h3 className="text-xs font-mono uppercase tracking-wider text-cyan-400 font-bold flex items-center gap-1.5">
                    <span>🛡️</span> Linked Evidence ({caseData.linked_evidence?.length || 0})
                  </h3>
                  <button
                    onClick={() => {
                      setEntityInputType("evidence");
                      setIsLinkEntityOpen(true);
                    }}
                    className="text-[11px] text-cyan-400 hover:underline cursor-pointer"
                  >
                    + Add Evidence
                  </button>
                </div>

                {caseData.linked_evidence && caseData.linked_evidence.length > 0 ? (
                  <div className="space-y-1.5">
                    {caseData.linked_evidence.map((ev) => (
                      <div
                        key={ev.evidence_id}
                        className="flex items-center justify-between p-2 bg-zinc-950/80 border border-zinc-800/80 rounded-lg text-xs"
                      >
                        <div>
                          <div className="font-semibold text-zinc-200 flex items-center gap-2">
                            <span>{ev.object_class || "Observation"} Evidence</span>
                            <span className="text-[10px] font-mono px-1 py-0.2 bg-cyan-500/10 text-cyan-400 rounded">
                              {ev.validation_status || "VALID"}
                            </span>
                          </div>
                          <div className="text-[10px] text-zinc-500 font-mono">
                            ID: {ev.evidence_id.slice(0, 12)}... &bull; @{ev.timestamp_seconds?.toFixed(1)}s &bull; {ev.evidence_type || "snapshot"}
                          </div>
                        </div>
                        <button
                          onClick={() => handleUnlinkSpecificEntity("evidence", ev.evidence_id)}
                          className="text-zinc-500 hover:text-red-400 text-xs px-1 cursor-pointer"
                          title="Unlink evidence"
                        >
                          &times;
                        </button>
                      </div>
                    ))}
                  </div>
                ) : (
                  <div className="text-xs text-zinc-500 italic py-3">No evidence linked yet.</div>
                )}
              </div>
            </div>

            {/* Audit Trail */}
            <div className="bg-zinc-900/60 border border-zinc-800 rounded-xl p-3.5 space-y-2.5">
              <h3 className="text-xs font-mono uppercase tracking-wider text-zinc-400 font-bold flex items-center gap-1.5">
                <span>📋</span> Activity Audit Log
              </h3>
              <div className="space-y-1.5 max-h-56 overflow-y-auto pr-1">
                {activities.map((a) => (
                  <div
                    key={a.id}
                    className="flex items-start justify-between p-2 bg-zinc-950/50 border border-zinc-800/50 rounded-lg text-xs"
                  >
                    <div>
                      <span className="font-semibold text-zinc-300 font-mono text-[11px]">
                        [{a.action_type}]
                      </span>{" "}
                      <span className="text-zinc-400">{a.description}</span>
                    </div>
                    <div className="text-[10px] font-mono text-zinc-500 shrink-0 ml-2">
                      {new Date(a.created_at).toLocaleTimeString()}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        )}

        {/* TAB 2: UNIFIED TIMELINE */}
        {activeTab === "timeline" && (
          <div className="space-y-3">
            <div className="flex flex-wrap items-center justify-between gap-2 bg-zinc-900/60 p-2.5 rounded-lg border border-zinc-800 text-xs">
              <div className="flex items-center gap-1.5">
                <span className="font-mono text-zinc-400 uppercase text-[10px]">Filter Layer:</span>
                {["", "detection_event", "security_event", "correlated_incident", "evidence", "bookmark", "analyst_note"].map(
                  (layer) => (
                    <button
                      key={layer}
                      onClick={() => setActiveLayerFilter(layer)}
                      className={`px-2 py-0.5 rounded text-[10px] font-mono transition-colors cursor-pointer ${
                        activeLayerFilter === layer
                          ? "bg-emerald-600 text-zinc-950 font-bold"
                          : "bg-zinc-800 text-zinc-400 hover:text-zinc-200"
                      }`}
                    >
                      {layer ? layer.replace("_", " ").toUpperCase() : "ALL"}
                    </button>
                  )
                )}
              </div>

              <div className="text-zinc-500 font-mono text-xs">
                Total: {timelineItems.length} entries
              </div>
            </div>

            {timelineLoading ? (
              <div className="py-12 text-center text-zinc-500 text-xs font-mono">
                Loading unified case timeline...
              </div>
            ) : timelineItems.length === 0 ? (
              <div className="py-12 text-center text-zinc-500 text-xs font-mono">
                No timeline events found matching criteria.
              </div>
            ) : (
              <div className="space-y-1.5">
                {timelineItems.map((item) => (
                  <div
                    key={`${item.layer}-${item.id}`}
                    onClick={() => {
                      if (item.layer === "correlated_incident" || item.layer === "security_event") {
                        handleSelectIncident(item.id);
                        setActiveTab("focus");
                      }
                    }}
                    className={`flex items-start justify-between p-2.5 bg-zinc-900/80 border border-zinc-800 hover:border-zinc-700 rounded-lg text-xs transition-all ${
                      item.layer === "correlated_incident" || item.layer === "security_event"
                        ? "cursor-pointer hover:bg-zinc-850"
                        : ""
                    }`}
                  >
                    <div className="flex items-start gap-3">
                      <div className="font-mono text-emerald-400 font-bold w-12 shrink-0">
                        {item.timestamp.toFixed(1)}s
                      </div>
                      <div>
                        <div className="flex items-center gap-2">
                          <span
                            className={`text-[9px] font-mono font-bold px-1.5 py-0.2 rounded border ${
                              item.layer === "correlated_incident"
                                ? "bg-amber-950 text-amber-300 border-amber-800"
                                : item.layer === "security_event"
                                ? "bg-rose-950 text-rose-300 border-rose-800"
                                : item.layer === "evidence"
                                ? "bg-cyan-950 text-cyan-300 border-cyan-800"
                                : item.layer === "bookmark"
                                ? "bg-purple-950 text-purple-300 border-purple-800"
                                : item.layer === "analyst_note"
                                ? "bg-blue-950 text-blue-300 border-blue-800"
                                : "bg-zinc-800 text-zinc-300 border-zinc-700"
                            }`}
                          >
                            {item.layer.replace("_", " ").toUpperCase()}
                          </span>
                          <span className="font-semibold text-zinc-200">{item.label}</span>
                          {item.validation_decision && (
                            <span className="text-[9px] font-mono bg-zinc-800 text-zinc-400 px-1 rounded">
                              {item.validation_decision}
                            </span>
                          )}
                        </div>
                        {item.detail && (
                          <p className="text-zinc-400 text-[11px] mt-0.5 leading-relaxed">
                            {item.detail}
                          </p>
                        )}
                      </div>
                    </div>

                    <div className="text-[10px] font-mono text-zinc-500 shrink-0 ml-4 text-right">
                      <div>{item.camera_label || item.source_name || "Video"}</div>
                      {item.layer === "correlated_incident" && (
                        <div className="text-emerald-400 text-[10px] underline">Inspect Focus &rarr;</div>
                      )}
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        )}

        {/* TAB 3: FOCUS & REPLAY */}
        {activeTab === "focus" && (
          <div className="space-y-4">
            {focusLoading ? (
              <div className="py-12 text-center text-zinc-500 text-xs font-mono">
                Assembling forensic focus context...
              </div>
            ) : !selectedIncidentId || !replayContext ? (
              <div className="p-8 text-center bg-zinc-900/40 border border-dashed border-zinc-800 rounded-xl space-y-2">
                <div className="text-2xl">🎯</div>
                <div className="text-sm font-semibold text-zinc-300">No Incident Selected for Focus</div>
                <p className="text-xs text-zinc-500">
                  Select a correlated incident or security event from the Unified Timeline or Incident list to enter Focus Mode.
                </p>
              </div>
            ) : (
              <div className="space-y-4">
                {/* Replay Player Window */}
                <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
                  <div className="lg:col-span-2 bg-zinc-900/90 border border-zinc-800 rounded-xl p-3.5 space-y-3">
                    <div className="flex items-center justify-between border-b border-zinc-800 pb-2">
                      <div className="flex items-center gap-2">
                        <span className="text-emerald-400 font-bold text-xs">REPLAY INCIDENT</span>
                        <span className="text-zinc-500 text-xs font-mono">
                          [{replayContext.replay_context.replay_start}s &rarr; {replayContext.replay_context.replay_end}s]
                        </span>
                      </div>
                      <span className="text-[10px] font-mono bg-zinc-800 text-zinc-300 px-2 py-0.5 rounded">
                        Window: BEFORE &rarr; CONTEXT &rarr; INCIDENT &rarr; AFTER
                      </span>
                    </div>

                    <div className="aspect-video bg-black rounded-lg overflow-hidden relative border border-zinc-800">
                      <video
                        ref={videoRef}
                        controls
                        src={`${API_BASE_URL}/api/videos/${replayContext.video_id}/stream`}
                        className="w-full h-full object-contain"
                      />
                    </div>

                    {/* Pre/Post Roll Controls */}
                    <div className="flex items-center justify-between text-xs font-mono text-zinc-400 bg-zinc-950 p-2 rounded-lg border border-zinc-800">
                      <div className="flex items-center gap-2">
                        <span>Pre-Roll Buffer:</span>
                        <select
                          value={preRoll}
                          onChange={(e) => {
                            const val = parseFloat(e.target.value);
                            setPreRoll(val);
                            if (selectedIncidentId) handleSelectIncident(selectedIncidentId);
                          }}
                          className="bg-zinc-900 border border-zinc-700 rounded px-1.5 py-0.5 text-zinc-200"
                        >
                          <option value="2.0">2.0s</option>
                          <option value="5.0">5.0s</option>
                          <option value="10.0">10.0s</option>
                        </select>
                      </div>

                      <div className="flex items-center gap-2">
                        <span>Post-Roll Buffer:</span>
                        <select
                          value={postRoll}
                          onChange={(e) => {
                            const val = parseFloat(e.target.value);
                            setPostRoll(val);
                            if (selectedIncidentId) handleSelectIncident(selectedIncidentId);
                          }}
                          className="bg-zinc-900 border border-zinc-700 rounded px-1.5 py-0.5 text-zinc-200"
                        >
                          <option value="2.0">2.0s</option>
                          <option value="5.0">5.0s</option>
                          <option value="10.0">10.0s</option>
                        </select>
                      </div>
                    </div>
                  </div>

                  {/* "WHY DID SENTINEL FLAG THIS?" CARD */}
                  <div className="bg-zinc-900/90 border border-zinc-800 rounded-xl p-4 space-y-3 flex flex-col justify-between">
                    <div className="space-y-3">
                      <div className="border-b border-zinc-800 pb-2">
                        <h3 className="text-xs font-mono font-bold text-amber-400 uppercase tracking-wider flex items-center gap-1.5">
                          <span>💡</span> Why Did Sentinel Flag This?
                        </h3>
                        <p className="text-[11px] text-zinc-400 mt-0.5">
                          Authentic signal explainability grounded in deterministic database evidence.
                        </p>
                      </div>

                      {explanation && (
                        <div className="space-y-2.5 text-xs">
                          <div>
                            <div className="text-[10px] font-mono text-emerald-400 uppercase font-bold tracking-wider mb-1">
                              Supporting Signals
                            </div>
                            <ul className="space-y-1">
                              {explanation.supporting_signals.map((sig, idx) => (
                                <li key={idx} className="flex items-start gap-1.5 text-zinc-300 text-[11px]">
                                  <span className="text-emerald-400">✓</span>
                                  <span>{sig}</span>
                                </li>
                              ))}
                            </ul>
                          </div>

                          <div>
                            <div className="text-[10px] font-mono text-amber-400 uppercase font-bold tracking-wider mb-1">
                              Limiting / Contradicting Signals
                            </div>
                            <ul className="space-y-1">
                              {explanation.limiting_signals.map((sig, idx) => (
                                <li key={idx} className="flex items-start gap-1.5 text-zinc-400 text-[11px]">
                                  <span className="text-amber-400">•</span>
                                  <span>{sig}</span>
                                </li>
                              ))}
                            </ul>
                          </div>

                          <div className="pt-2 border-t border-zinc-800 space-y-1.5">
                            <div className="flex items-center justify-between font-mono text-xs">
                              <span className="text-zinc-400">Pattern Evidence Strength:</span>
                              <span className="text-emerald-400 font-bold">
                                {explanation.pattern_evidence_strength}
                              </span>
                            </div>
                            <div className="flex items-center justify-between font-mono text-xs">
                              <span className="text-zinc-400">Final Assessment:</span>
                              <span className="text-amber-300 font-bold">
                                {explanation.final_assessment_score.toFixed(2)} — {explanation.validation_decision}
                              </span>
                            </div>
                          </div>
                        </div>
                      )}
                    </div>

                    {/* Human Verification Notice Banner */}
                    <div className="p-2.5 bg-amber-950/40 border border-amber-800/50 rounded-lg text-[10px] text-amber-300 font-mono leading-relaxed">
                      ⚠️ <strong>HUMAN VERIFICATION REQUIRED:</strong> Correct abstention is preferred over a false alert.
                      Pattern evidence describes observational video metrics only.
                    </div>
                  </div>
                </div>
              </div>
            )}
          </div>
        )}

        {/* TAB 4: MULTI-CAMERA & TOPOLOGY */}
        {activeTab === "multicamera" && (
          <div className="space-y-4">
            <div className="bg-zinc-900/90 border border-zinc-800 rounded-xl p-4 space-y-3">
              <div className="flex items-center justify-between border-b border-zinc-800 pb-2">
                <div>
                  <h3 className="text-xs font-mono font-bold text-white uppercase tracking-wider flex items-center gap-1.5">
                    <span>🗺️</span> Camera Topology &amp; Physical Transitions
                  </h3>
                  <p className="text-[11px] text-zinc-400 mt-0.5">
                    Spatial field-of-view layout, physical adjacency transitions, and confirmed cross-camera associations.
                  </p>
                </div>
                <span className="text-xs font-mono text-emerald-400 bg-emerald-950 px-2 py-0.5 rounded border border-emerald-800">
                  {topology?.total_cameras || 0} Cameras Registered
                </span>
              </div>

              {/* Visual Graph View */}
              {topology && topology.nodes.length > 0 ? (
                <div className="bg-zinc-950 border border-zinc-800 rounded-lg p-6 flex flex-wrap items-center justify-center gap-8 min-h-[220px]">
                  {topology.nodes.map((n, idx) => (
                    <div
                      key={n.id}
                      className="flex flex-col items-center p-3 bg-zinc-900 border-2 border-emerald-500/60 rounded-xl shadow-lg min-w-[140px] text-center"
                    >
                      <span className="text-2xl mb-1">📹</span>
                      <div className="font-bold text-xs text-zinc-100">{n.label}</div>
                      <div className="text-[10px] font-mono text-zinc-400">{n.position_hint || "Corridor"}</div>
                      <div className="text-[9px] font-mono text-emerald-400 mt-1">Offset: {n.clock_offset_seconds}s</div>
                    </div>
                  ))}
                </div>
              ) : (
                <div className="py-8 text-center text-zinc-500 text-xs font-mono">
                  No cameras registered to topology. Link cameras from the Overview tab.
                </div>
              )}
            </div>

            {/* Synchronized Playback Grid */}
            <div className="bg-zinc-900/90 border border-zinc-800 rounded-xl p-4 space-y-3">
              <div className="flex items-center justify-between border-b border-zinc-800 pb-2">
                <h3 className="text-xs font-mono font-bold text-white uppercase tracking-wider flex items-center gap-1.5">
                  <span>⏱️</span> Synchronized Multi-Camera Playback
                </h3>
                <div className="flex items-center gap-2">
                  <span className="text-xs font-mono text-zinc-400">Master Time: {syncTime.toFixed(1)}s</span>
                  <button
                    onClick={() => setIsSyncPlaying(!isSyncPlaying)}
                    className="px-2.5 py-1 bg-emerald-600 hover:bg-emerald-500 text-zinc-950 font-bold text-xs rounded transition-colors cursor-pointer"
                  >
                    {isSyncPlaying ? "Pause All" : "Play Synchronized"}
                  </button>
                </div>
              </div>

              {caseData.linked_cameras && caseData.linked_cameras.length > 0 ? (
                <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                  {caseData.linked_cameras.map((c) => (
                    <div key={c.camera_id} className="bg-zinc-950 border border-zinc-800 rounded-lg p-2.5 space-y-1.5">
                      <div className="flex items-center justify-between text-xs font-mono">
                        <span className="font-bold text-zinc-200">{c.camera_label}</span>
                        <span className="text-zinc-500">Aligned Time: {(syncTime + c.clock_offset_seconds).toFixed(1)}s</span>
                      </div>
                      <div className="aspect-video bg-black rounded overflow-hidden">
                        <video
                          controls
                          src={`${API_BASE_URL}/api/videos/${c.video_id}/stream`}
                          className="w-full h-full object-contain"
                        />
                      </div>
                    </div>
                  ))}
                </div>
              ) : (
                <div className="py-6 text-center text-zinc-500 text-xs font-mono">
                  Link multiple cameras to activate synchronized multi-view playback.
                </div>
              )}
            </div>
          </div>
        )}

        {/* TAB 5: BOOKMARKS */}
        {activeTab === "bookmarks" && (
          <div className="space-y-3">
            <div className="flex items-center justify-between bg-zinc-900/60 p-2.5 rounded-lg border border-zinc-800">
              <span className="text-xs font-mono text-zinc-400">Pinned Key Investigation Timestamps</span>
              <button
                onClick={() => setIsAddBookmarkOpen(true)}
                className="px-3 py-1 bg-emerald-600 hover:bg-emerald-500 text-zinc-950 font-bold text-xs rounded-lg transition-colors cursor-pointer"
              >
                + Add Bookmark
              </button>
            </div>

            {bookmarks.length === 0 ? (
              <div className="py-12 text-center text-zinc-500 text-xs font-mono">
                No bookmarks created yet.
              </div>
            ) : (
              <div className="space-y-2">
                {bookmarks.map((bm) => (
                  <div
                    key={bm.id}
                    className="flex items-center justify-between p-3 bg-zinc-900/80 border border-zinc-800 rounded-xl text-xs"
                  >
                    <div className="flex items-center gap-3">
                      <span className="text-lg">📌</span>
                      <div>
                        <div className="font-bold text-zinc-200">{bm.title}</div>
                        <div className="text-[11px] text-zinc-400 font-mono">
                          Timestamp: {bm.timestamp_seconds.toFixed(1)}s • Author: {bm.author || "Investigator"}
                        </div>
                        {bm.description && (
                          <div className="text-[11px] text-zinc-400 mt-0.5">{bm.description}</div>
                        )}
                      </div>
                    </div>

                    <button
                      onClick={async () => {
                        await deleteCaseBookmark(caseId, bm.id);
                        await loadBookmarks();
                        await loadActivities();
                      }}
                      className="text-zinc-500 hover:text-rose-400 p-1 cursor-pointer"
                      title="Delete Bookmark"
                    >
                      🗑️
                    </button>
                  </div>
                ))}
              </div>
            )}
          </div>
        )}

        {/* TAB 6: NOTES */}
        {activeTab === "notes" && (
          <div className="space-y-4">
            {/* Create Note Form */}
            <form onSubmit={handleCreateNote} className="bg-zinc-900/80 border border-zinc-800 rounded-xl p-3.5 space-y-3">
              <div className="flex items-center justify-between">
                <span className="text-xs font-mono font-bold text-blue-400 uppercase tracking-wider">
                  Add Analyst Note
                </span>
                <span className="text-[10px] font-mono bg-blue-950 text-blue-300 px-2 py-0.5 rounded border border-blue-800">
                  CLASSIFICATION: ANALYST_NOTE
                </span>
              </div>

              <textarea
                rows={2}
                required
                value={newNoteText}
                onChange={(e) => setNewNoteText(e.target.value)}
                placeholder="Enter investigator observations, hypothesis, or interview corroboration..."
                className="w-full bg-zinc-950 border border-zinc-800 rounded-lg p-2 text-zinc-200 text-xs focus:outline-none focus:border-blue-500"
              />

              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2 text-xs">
                  <span className="text-zinc-400 font-mono text-[10px]">Association:</span>
                  <select
                    value={newNoteType}
                    onChange={(e) => setNewNoteType(e.target.value)}
                    className="bg-zinc-950 border border-zinc-800 rounded px-2 py-0.5 text-zinc-300 text-xs"
                  >
                    <option value="CASE">Case</option>
                    <option value="INCIDENT">Incident</option>
                    <option value="EVIDENCE">Evidence</option>
                    <option value="TRACK">Track</option>
                    <option value="CAMERA">Camera</option>
                  </select>
                </div>

                <button
                  type="submit"
                  className="px-3.5 py-1 bg-blue-600 hover:bg-blue-500 text-white font-semibold text-xs rounded-lg transition-colors cursor-pointer"
                >
                  Save Note
                </button>
              </div>
            </form>

            {/* Notes List */}
            <div className="space-y-2">
              {notes.map((n) => (
                <div
                  key={n.id}
                  className="p-3 bg-zinc-900/80 border border-zinc-800 rounded-xl text-xs space-y-1"
                >
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-2">
                      <span className="text-[9px] font-mono font-bold bg-blue-950 text-blue-300 px-1.5 py-0.5 rounded border border-blue-800">
                        ANALYST NOTE
                      </span>
                      <span className="font-semibold text-zinc-200">{n.author}</span>
                      <span className="text-[10px] text-zinc-500 font-mono">
                        ({n.associated_type})
                      </span>
                    </div>
                    <button
                      onClick={async () => {
                        await deleteCaseNote(caseId, n.id);
                        await loadNotes();
                        await loadActivities();
                      }}
                      className="text-zinc-500 hover:text-rose-400 cursor-pointer"
                      title="Delete Note"
                    >
                      🗑️
                    </button>
                  </div>
                  <p className="text-zinc-300 text-xs leading-relaxed mt-1">{n.content}</p>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* TAB 7: STORYLINE */}
        {activeTab === "storyline" && (
          <div className="space-y-3">
            <div className="bg-zinc-900/60 p-3 rounded-lg border border-zinc-800 flex items-center justify-between">
              <div>
                <h3 className="text-xs font-mono font-bold text-white uppercase tracking-wider">
                  Case Storyline &amp; Investigative Narrative
                </h3>
                <p className="text-[11px] text-zinc-400 mt-0.5">
                  Sequential multi-camera narrative synthesized from correlated observations, events, and notes.
                </p>
              </div>
              <button
                onClick={handleExport}
                className="px-3 py-1 bg-emerald-600 hover:bg-emerald-500 text-zinc-950 font-bold text-xs rounded transition-colors cursor-pointer"
              >
                Export Storyline
              </button>
            </div>

            {storyline.length === 0 ? (
              <div className="py-12 text-center text-zinc-500 text-xs font-mono">
                No storyline steps generated yet.
              </div>
            ) : (
              <div className="relative border-l-2 border-emerald-500/30 ml-4 pl-4 space-y-4 my-4">
                {storyline.map((step, idx) => (
                  <div key={step.step_id || idx} className="relative group">
                    <div className="absolute -left-[23px] top-1.5 w-3 h-3 rounded-full bg-emerald-500 ring-4 ring-zinc-950" />
                    <div className="bg-zinc-900/80 border border-zinc-800 p-3 rounded-lg text-xs space-y-1">
                      <div className="flex items-center justify-between">
                        <span className="font-mono text-emerald-400 font-bold">
                          {step.timestamp_formatted} ({step.timestamp.toFixed(1)}s)
                        </span>
                        <span className="text-[10px] font-mono bg-zinc-800 text-zinc-400 px-1.5 py-0.5 rounded">
                          {step.camera_label}
                        </span>
                      </div>
                      <div className="font-semibold text-zinc-200">{step.title}</div>
                      <p className="text-zinc-400 leading-relaxed text-[11px]">{step.description}</p>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        )}
      </div>

      {/* Add Bookmark Modal */}
      {isAddBookmarkOpen && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/75 backdrop-blur-sm p-4">
          <div className="bg-zinc-900 border border-zinc-800 rounded-xl max-w-md w-full p-5 space-y-4 shadow-2xl">
            <h3 className="text-sm font-bold text-white">Create Bookmark Pin</h3>
            <form onSubmit={handleCreateBookmark} className="space-y-3 text-xs">
              <div>
                <label className="block text-zinc-300 font-medium mb-1">Title</label>
                <input
                  type="text"
                  required
                  value={bmTitle}
                  onChange={(e) => setBmTitle(e.target.value)}
                  placeholder="e.g. Suspect approached rear gate"
                  className="w-full bg-zinc-950 border border-zinc-800 rounded p-2 text-zinc-200"
                />
              </div>

              <div>
                <label className="block text-zinc-300 font-medium mb-1">Timestamp (seconds)</label>
                <input
                  type="number"
                  step="0.1"
                  required
                  value={bmTime}
                  onChange={(e) => setBmTime(parseFloat(e.target.value) || 0)}
                  className="w-full bg-zinc-950 border border-zinc-800 rounded p-2 text-zinc-200"
                />
              </div>

              <div>
                <label className="block text-zinc-300 font-medium mb-1">Target Video</label>
                <select
                  value={bmVideoId}
                  onChange={(e) => setBmVideoId(e.target.value)}
                  className="w-full bg-zinc-950 border border-zinc-800 rounded p-2 text-zinc-200"
                >
                  {caseData.linked_videos?.map((v) => (
                    <option key={v.video_id} value={v.video_id}>
                      {v.filename}
                    </option>
                  ))}
                </select>
              </div>

              <div className="flex justify-end gap-2 pt-2">
                <button
                  type="button"
                  onClick={() => setIsAddBookmarkOpen(false)}
                  className="px-3 py-1 bg-zinc-800 text-zinc-300 rounded"
                >
                  Cancel
                </button>
                <button
                  type="submit"
                  className="px-3 py-1 bg-emerald-600 text-zinc-950 font-bold rounded"
                >
                  Pin Bookmark
                </button>
              </div>
            </form>
          </div>
        </div>
      )}

      {/* Link Entity Modal (Searchable Selector - No Manual UUID) */}
      {isLinkEntityOpen && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/80 backdrop-blur-sm p-4">
          <div className="bg-zinc-900 border border-zinc-800 rounded-xl max-w-2xl w-full p-6 space-y-4 shadow-2xl flex flex-col max-h-[85vh]">
            <div className="flex items-center justify-between border-b border-zinc-800 pb-3">
              <div>
                <h3 className="text-base font-bold text-white flex items-center gap-2">
                  <span>🔗</span> Link Existing Surveillance Footage &amp; Entities
                </h3>
                <p className="text-xs text-zinc-400 mt-0.5">
                  Select surveillance footage or intelligence findings that have already been ingested into Sentinel.
                </p>
              </div>
              <button
                onClick={() => setIsLinkEntityOpen(false)}
                className="text-zinc-500 hover:text-zinc-300 text-base cursor-pointer"
              >
                ✕
              </button>
            </div>

            {/* Laptop upload callout inside modal */}
            {onUploadVideoForCase && (
              <div className="bg-emerald-950/40 border border-emerald-800/50 rounded-lg p-3 flex flex-col sm:flex-row sm:items-center justify-between gap-3">
                <div className="text-xs text-zinc-300">
                  <strong className="text-emerald-400">Need to upload new footage from your laptop?</strong>
                  <p className="text-zinc-400 text-[11px] mt-0.5">
                    Select a file from your computer to ingest and automatically associate with this case.
                  </p>
                </div>
                <button
                  type="button"
                  onClick={() => {
                    setIsLinkEntityOpen(false);
                    onUploadVideoForCase();
                  }}
                  className="px-3 py-1.5 bg-emerald-600 hover:bg-emerald-500 text-zinc-950 font-bold text-xs rounded-lg transition-colors cursor-pointer shrink-0"
                >
                  + Upload Video
                </button>
              </div>
            )}

            {linkError && (
              <div className="p-3 bg-red-500/10 border border-red-500/30 text-red-400 text-xs rounded">
                {linkError}
              </div>
            )}

            {/* Type Switcher Tabs */}
            <div className="flex items-center gap-2 border-b border-zinc-800 pb-2 text-xs font-mono">
              <button
                onClick={() => setEntityInputType("video")}
                className={`px-3 py-1.5 rounded-lg font-bold transition-all cursor-pointer ${
                  entityInputType === "video"
                    ? "bg-emerald-600 text-zinc-950"
                    : "bg-zinc-800 text-zinc-300 hover:bg-zinc-700"
                }`}
              >
                🎥 Existing Videos ({availableVideos.length})
              </button>
              <button
                onClick={() => setEntityInputType("camera")}
                className={`px-3 py-1.5 rounded-lg font-bold transition-all cursor-pointer ${
                  entityInputType === "camera"
                    ? "bg-emerald-600 text-zinc-950"
                    : "bg-zinc-800 text-zinc-300 hover:bg-zinc-700"
                }`}
              >
                📹 Cameras ({availableCameras.length})
              </button>
              <button
                onClick={() => setEntityInputType("incident")}
                className={`px-3 py-1.5 rounded-lg font-bold transition-all cursor-pointer ${
                  entityInputType === "incident"
                    ? "bg-emerald-600 text-zinc-950"
                    : "bg-zinc-800 text-zinc-300 hover:bg-zinc-700"
                }`}
              >
                ⚠️ Incidents ({availableIncidents.length})
              </button>
              <button
                onClick={() => setEntityInputType("evidence")}
                className={`px-3 py-1.5 rounded-lg font-bold transition-all cursor-pointer ${
                  entityInputType === "evidence"
                    ? "bg-emerald-600 text-zinc-950"
                    : "bg-zinc-800 text-zinc-300 hover:bg-zinc-700"
                }`}
              >
                🛡️ Evidence ({availableEvidence.length})
              </button>
            </div>

            {/* Search Input */}
            <div>
              <input
                type="text"
                value={entitySearchQuery}
                onChange={(e) => setEntitySearchQuery(e.target.value)}
                placeholder={`Search available ${entityInputType}s...`}
                className="w-full bg-zinc-950 border border-zinc-800 rounded-lg p-2.5 text-xs text-zinc-200 placeholder-zinc-500 focus:outline-none focus:border-emerald-500"
              />
            </div>

            {/* Entity List */}
            <div className="flex-1 overflow-y-auto space-y-2 pr-1 min-h-[260px]">
              {loadingEntities ? (
                <div className="py-12 text-center text-zinc-500 font-mono text-xs">
                  Loading available entities...
                </div>
              ) : entityInputType === "video" ? (
                availableVideos
                  .filter((v) => !entitySearchQuery || v.filename.toLowerCase().includes(entitySearchQuery.toLowerCase()))
                  .map((v) => {
                    const isAlreadyLinked = caseData.linked_videos?.some((lv) => lv.video_id === v.id);
                    const camAssoc = availableCameras.find((c) => c.id === v.camera_id);
                    return (
                      <div
                        key={v.id}
                        className="flex items-center justify-between p-3 bg-zinc-950 rounded-lg border border-zinc-800 hover:border-zinc-700 text-xs"
                      >
                        <div className="space-y-1 max-w-[440px]">
                          <div className="text-zinc-200 font-bold font-mono truncate" title={v.filename}>
                            {v.filename}
                          </div>
                          <div className="text-zinc-400 text-[11px] font-mono flex flex-wrap gap-2">
                            <span>Duration: {v.duration_seconds ? `${v.duration_seconds.toFixed(1)}s` : "--"}</span>
                            <span>&bull; FPS: {v.fps || "--"}</span>
                            <span>&bull; Detections: {v.detections_count}</span>
                            {camAssoc && (
                              <span className="text-indigo-400">&bull; Camera: {camAssoc.camera_label}</span>
                            )}
                          </div>
                          <div className="text-zinc-500 text-[10px] font-mono">
                            Status: <span className="text-emerald-400 font-semibold">{v.status}</span>
                            {v.uploaded_at && ` &bull; Uploaded: ${new Date(v.uploaded_at).toLocaleString()}`}
                          </div>
                        </div>

                        {isAlreadyLinked ? (
                          <span className="text-[11px] text-zinc-500 font-mono">Already Linked</span>
                        ) : (
                          <button
                            disabled={linkingActionId === v.id}
                            onClick={() => handleLinkSpecificEntity("video", v.id)}
                            className="px-3 py-1.5 bg-emerald-600 hover:bg-emerald-500 text-zinc-950 font-bold rounded text-xs transition-all disabled:opacity-50 cursor-pointer"
                          >
                            {linkingActionId === v.id ? "Linking..." : "+ Link Video"}
                          </button>
                        )}
                      </div>
                    );
                  })
              ) : entityInputType === "camera" ? (
                availableCameras
                  .filter((c) => !entitySearchQuery || c.camera_label.toLowerCase().includes(entitySearchQuery.toLowerCase()))
                  .map((c) => {
                    const isAlreadyLinked = caseData.linked_cameras?.some((lc) => lc.camera_id === c.id);
                    return (
                      <div
                        key={c.id}
                        className="flex items-center justify-between p-3 bg-zinc-950 rounded-lg border border-zinc-800 hover:border-zinc-700 text-xs"
                      >
                        <div className="space-y-1 max-w-[440px]">
                          <div className="text-zinc-200 font-bold font-mono truncate flex items-center gap-2">
                            <span>{c.camera_label}</span>
                            <span className="text-[10px] font-mono px-1.5 py-0.2 bg-emerald-500/10 text-emerald-400 rounded border border-emerald-500/30 uppercase font-semibold">
                              {c.status || "ACTIVE"}
                            </span>
                          </div>
                          <div className="text-zinc-400 text-[11px] font-mono">
                            Location: {c.location || c.position_hint || "Facility Area"} &bull; Coverage: {c.coverage_description || c.field_of_view_hint || "Perimeter"}
                          </div>
                          <div className="text-zinc-500 text-[10px] font-mono">
                            Associated Footage: {c.associated_videos_count !== undefined ? c.associated_videos_count : (c.video_id ? 1 : 0)} video(s)
                            {c.video_filename && ` &bull; Source: ${c.video_filename}`}
                          </div>
                        </div>

                        {isAlreadyLinked ? (
                          <span className="text-[11px] text-zinc-500 font-mono">Already Linked</span>
                        ) : (
                          <button
                            disabled={linkingActionId === c.id}
                            onClick={() => handleLinkSpecificEntity("camera", c.id)}
                            className="px-3 py-1.5 bg-emerald-600 hover:bg-emerald-500 text-zinc-950 font-bold rounded text-xs transition-all disabled:opacity-50 cursor-pointer"
                          >
                            {linkingActionId === c.id ? "Linking..." : "+ Link Camera"}
                          </button>
                        )}
                      </div>
                    );
                  })
              ) : entityInputType === "incident" ? (
                availableIncidents
                  .filter((inc) => !entitySearchQuery || inc.title.toLowerCase().includes(entitySearchQuery.toLowerCase()) || inc.incident_category.toLowerCase().includes(entitySearchQuery.toLowerCase()))
                  .map((inc) => {
                    const isAlreadyLinked = caseData.linked_incidents?.some((li) => li.incident_id === inc.id);
                    return (
                      <div
                        key={inc.id}
                        className="flex items-center justify-between p-3 bg-zinc-950 rounded-lg border border-zinc-800 hover:border-zinc-700 text-xs"
                      >
                        <div className="space-y-1 max-w-[440px]">
                          <div className="text-zinc-200 font-bold flex items-center gap-2">
                            <span>{inc.title}</span>
                            <span className="text-[10px] font-mono px-1.5 py-0.2 bg-cyan-500/10 text-cyan-400 rounded">
                              {inc.incident_category}
                            </span>
                            <span className="text-[10px] font-mono px-1 py-0.2 bg-zinc-800 text-zinc-400 rounded">
                              {inc.validation_decision || inc.decision || "REVIEW"}
                            </span>
                          </div>
                          <div className="text-zinc-400 text-[11px] font-mono">
                            Video: {inc.source_video_name || "Surveillance"} &bull; Time: {inc.start_time.toFixed(1)}s &ndash; {inc.end_time.toFixed(1)}s
                          </div>
                          <div className="text-zinc-500 text-[10px] font-mono">
                            Assessment: {(inc.assessment_score * 100).toFixed(0)}% &bull; Evidence Count: {inc.evidence_ids?.length || 0}
                          </div>
                        </div>

                        {isAlreadyLinked ? (
                          <span className="text-[11px] text-zinc-500 font-mono">Already Linked</span>
                        ) : (
                          <button
                            disabled={linkingActionId === inc.id}
                            onClick={() => handleLinkSpecificEntity("incident", inc.id)}
                            className="px-3 py-1.5 bg-emerald-600 hover:bg-emerald-500 text-zinc-950 font-bold rounded text-xs transition-all disabled:opacity-50 cursor-pointer"
                          >
                            {linkingActionId === inc.id ? "Linking..." : "+ Link Incident"}
                          </button>
                        )}
                      </div>
                    );
                  })
              ) : (
                availableEvidence
                  .filter((ev) => !entitySearchQuery || ev.object_class.toLowerCase().includes(entitySearchQuery.toLowerCase()) || ev.evidence_id.toLowerCase().includes(entitySearchQuery.toLowerCase()))
                  .map((ev) => {
                    const isAlreadyLinked = caseData.linked_evidence?.some((le) => le.evidence_id === ev.evidence_id);
                    return (
                      <div
                        key={ev.evidence_id}
                        className="flex items-center justify-between p-3 bg-zinc-950 rounded-lg border border-zinc-800 hover:border-zinc-700 text-xs"
                      >
                        <div className="space-y-1 max-w-[440px]">
                          <div className="text-zinc-200 font-bold flex items-center gap-2">
                            <span>{ev.object_class.toUpperCase()} Observation</span>
                            <span className="text-[10px] font-mono px-1.5 py-0.2 bg-cyan-500/10 text-cyan-400 rounded">
                              {ev.validation_status}
                            </span>
                            <span className="text-[10px] font-mono px-1 py-0.2 bg-zinc-800 text-zinc-400 rounded">
                              {ev.evidence_type}
                            </span>
                          </div>
                          <div className="text-zinc-400 text-[11px] font-mono">
                            Source: {ev.source_video_name} &bull; Timestamp: {ev.timestamp_seconds.toFixed(1)}s
                          </div>
                          <div className="text-zinc-500 text-[10px] font-mono">
                            Confidence: {(ev.confidence * 100).toFixed(0)}% &bull; ID: {ev.evidence_id}
                          </div>
                        </div>

                        {isAlreadyLinked ? (
                          <span className="text-[11px] text-zinc-500 font-mono">Already Linked</span>
                        ) : (
                          <button
                            disabled={linkingActionId === ev.evidence_id}
                            onClick={() => handleLinkSpecificEntity("evidence", ev.evidence_id)}
                            className="px-3 py-1.5 bg-emerald-600 hover:bg-emerald-500 text-zinc-950 font-bold rounded text-xs transition-all disabled:opacity-50 cursor-pointer"
                          >
                            {linkingActionId === ev.evidence_id ? "Linking..." : "+ Link Evidence"}
                          </button>
                        )}
                      </div>
                    );
                  })
              )}
            </div>

            <div className="flex justify-end pt-2 border-t border-zinc-800">
              <button
                type="button"
                onClick={() => setIsLinkEntityOpen(false)}
                className="px-4 py-2 bg-zinc-800 hover:bg-zinc-700 text-zinc-300 rounded-lg text-xs font-medium cursor-pointer"
              >
                Close
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
