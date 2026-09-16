"use client";

import React, { useState, useEffect, useRef, useMemo, ChangeEvent, DragEvent } from "react";
import {
  uploadVideo,
  getVideoStreamUrl,
  getVideoPlaybackUrl,
  getVideoPlaybackStatus,
  getVideoMetadata,
  processVideo,
  getVideoEvents,
  getVideoTimeline,
  investigateVideo,
  aiInvestigateVideo,
  createEvidence,
  getVideoEvidence,
  getEvidenceSnapshotUrl,
  getEvidenceAnnotatedUrl,
  getEvidenceClipUrl,
  getEvidencePlaybackUrl,
  UploadVideoResponse,
  ProcessVideoResponse,
  DetectionEvent,
  GroupedEvent,
  InvestigationResponse,
  AIInvestigateResponse,
  EvidenceItem,
  getVideoTracks,
  getVehicleAttributes,
  getFaceDetections,
  getSecurityEvents,
  getVideoZones,
  createVideoZone,
  deleteVideoZone,
  runSecurityAnalysis,
  TrackedObjectItem,
  VehicleAttributeItem,
  FaceDetectionItem,
  SecurityZoneItem,
  SecurityEventItem,
  generateVideoReport,
  getVideoReports,
  getReportDownloadUrl,
  getReportViewUrl,
  ReportItem,
  SpecializedObservationItem,
  getSpecializedObservations,
  DetectorHealthResponse,
  DetectorHealthItem,
  getVideoDetectorHealth,
  createCase,
  linkVideoToCase,
} from "@/lib/api";
import { CorrelatedIncidentsView } from "@/components/CorrelatedIncidentsView";
import { ForensicSearchPanel } from "@/components/ForensicSearchPanel";
import MultiCameraSessionPanel from "@/components/MultiCameraSessionPanel";

const SUPPORTED_EXTENSIONS = [".mp4", ".mov", ".avi", ".mkv", ".webm"];
const MAX_SIZE_MB = 500;

type ProcessingState = "idle" | "processing" | "completed" | "failed";
type ViewMode = "timeline" | "raw" | "intelligence" | "correlation" | "forensic" | "vault" | "reports" | "specialized" | "diagnostics" | "multicamera";

// Capitalise the first letter of each word in the class name or return fallback
function formatClassName(
  name?: string | null,
  fallback = "General Security Activity"
): string {
  if (!name || typeof name !== "string" || !name.trim()) {
    return fallback;
  }
  return name
    .replace(/_/g, " ")
    .split(" ")
    .filter(Boolean)
    .map((w) => w.charAt(0).toUpperCase() + w.slice(1).toLowerCase())
    .join(" ");
}

function formatTimestamp(seconds: number): string {
  const m = Math.floor(seconds / 60);
  const s = (seconds % 60).toFixed(2);
  return m > 0 ? `${m}m ${s}s` : `${s}s`;
}

const CLASS_COLORS: Record<string, string> = {
  person: "text-blue-400 bg-blue-500/10 border-blue-500/30",
  car: "text-amber-400 bg-amber-500/10 border-amber-500/30",
  truck: "text-orange-400 bg-orange-500/10 border-orange-500/30",
  bus: "text-orange-400 bg-orange-500/10 border-orange-500/30",
  motorcycle: "text-purple-400 bg-purple-500/10 border-purple-500/30",
  bicycle: "text-purple-400 bg-purple-500/10 border-purple-500/30",
  backpack: "text-emerald-400 bg-emerald-500/10 border-emerald-500/30",
  handbag: "text-emerald-400 bg-emerald-500/10 border-emerald-500/30",
  suitcase: "text-emerald-400 bg-emerald-500/10 border-emerald-500/30",
};

const getClassColor = (cls?: string | null) =>
  (cls && typeof cls === "string" ? CLASS_COLORS[cls.toLowerCase()] : undefined) ||
  "text-zinc-300 bg-zinc-800/60 border-zinc-700";

const PRIORITY_BADGES: Record<string, string> = {
  HIGH: "bg-red-500/20 text-red-400 border-red-500/40",
  NORMAL: "bg-blue-500/20 text-blue-400 border-blue-500/40",
  LOW: "bg-zinc-800 text-zinc-400 border-zinc-700",
};

export interface VideoUploadProps {
  initialViewMode?: ViewMode;
  initialVideoId?: string;
  activeCaseId?: string | null;
  onNavigateToCase?: (caseId: string) => void;
  onNavigateToIncidents?: () => void;
  onNavigateToEvidence?: () => void;
}

export default function VideoUpload({
  initialViewMode,
  initialVideoId,
  activeCaseId,
  onNavigateToCase,
  onNavigateToIncidents,
  onNavigateToEvidence,
}: VideoUploadProps = {}) {
  // Upload state
  const [dragActive, setDragActive] = useState<boolean>(false);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [isUploading, setIsUploading] = useState<boolean>(false);
  const [uploadProgress, setUploadProgress] = useState<number>(0);
  const [uploadResult, setUploadResult] = useState<UploadVideoResponse | null>(null);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [copiedId, setCopiedId] = useState<boolean>(false);

  // Video -> Case Workflow state
  const [isCreateCaseModalOpen, setIsCreateCaseModalOpen] = useState<boolean>(false);
  const [caseTitleInput, setCaseTitleInput] = useState<string>("");
  const [caseDescInput, setCaseDescInput] = useState<string>("");
  const [casePriorityInput, setCasePriorityInput] = useState<"LOW" | "MEDIUM" | "HIGH" | "CRITICAL">("HIGH");
  const [isCaseSubmitting, setIsCaseSubmitting] = useState<boolean>(false);
  const [caseWorkflowFeedback, setCaseWorkflowFeedback] = useState<string | null>(null);

  // Playback state (browser compatibility & conversion)
  const [playbackState, setPlaybackState] = useState<"ready" | "preparing" | "unavailable">("ready");
  const [playbackMessage, setPlaybackMessage] = useState<string | null>(null);
  const [existingVideoIdInput, setExistingVideoIdInput] = useState<string>("");
  const [isLoadingExisting, setIsLoadingExisting] = useState<boolean>(false);

  // Processing & Timeline state
  const [processingState, setProcessingState] = useState<ProcessingState>("idle");
  const [processingResult, setProcessingResult] = useState<ProcessVideoResponse | null>(null);
  const [processingError, setProcessingError] = useState<string | null>(null);
  const [events, setEvents] = useState<DetectionEvent[]>([]);
  const [timelineEvents, setTimelineEvents] = useState<GroupedEvent[]>([]);
  const [activeView, setActiveView] = useState<ViewMode>(initialViewMode || "timeline");

  useEffect(() => {
    if (initialViewMode) {
      setActiveView(initialViewMode);
    }
  }, [initialViewMode]);

  const [classFilter, setClassFilter] = useState<string>("all");
  const [minConfidence, setMinConfidence] = useState<number>(0);
  const [activeEventId, setActiveEventId] = useState<string | null>(null);

  // Phase 5A: Natural-Language Investigation State
  const [investigationQuery, setInvestigationQuery] = useState<string>("");
  const [isInvestigating, setIsInvestigating] = useState<boolean>(false);
  const [investigationResponse, setInvestigationResponse] = useState<InvestigationResponse | null>(null);
  const [investigationError, setInvestigationError] = useState<string | null>(null);

  // Phase 7: LLM-Assisted Evidence-Grounded Investigation State
  const [investigationMode, setInvestigationMode] = useState<"ai" | "deterministic">("ai");
  const [aiResponse, setAiResponse] = useState<AIInvestigateResponse | null>(null);
  const [aiHistory, setAiHistory] = useState<Array<{ role: string; content: string }>>([]);

  // Phase 6: Evidence Extraction & Vault State
  const [evidenceList, setEvidenceList] = useState<EvidenceItem[]>([]);
  const [capturingEventId, setCapturingEventId] = useState<string | null>(null);
  const [evidenceFeedback, setEvidenceFeedback] = useState<{ type: "success" | "error"; message: string } | null>(null);
  const [selectedEvidence, setSelectedEvidence] = useState<EvidenceItem | null>(null);
  const [evidenceModalMode, setEvidenceModalMode] = useState<"snapshot" | "annotated" | "clip">("snapshot");

  const fetchEvidence = async (videoId: string) => {
    try {
      const res = await getVideoEvidence(videoId);
      setEvidenceList(res.evidence);
    } catch {
      // ignore
    }
  };

  // Phase 8: Advanced Security Intelligence State
  const [tracks, setTracks] = useState<TrackedObjectItem[]>([]);
  const [vehicleAttributes, setVehicleAttributes] = useState<VehicleAttributeItem[]>([]);
  const [faceDetections, setFaceDetections] = useState<FaceDetectionItem[]>([]);
  const [securityEvents, setSecurityEvents] = useState<SecurityEventItem[]>([]);
  const [securityZones, setSecurityZones] = useState<SecurityZoneItem[]>([]);
  const [isAnalyzingIntelligence, setIsAnalyzingIntelligence] = useState<boolean>(false);
  const [intelligenceFeedback, setIntelligenceFeedback] = useState<string | null>(null);

  // New Zone Form State
  const [newZoneName, setNewZoneName] = useState<string>("");
  const [newZonePreset, setNewZonePreset] = useState<string>("gate");
  const [isCreatingZone, setIsCreatingZone] = useState<boolean>(false);

  const fetchSecurityIntelligence = async (videoId: string) => {
    try {
      const [tracksRes, attrsRes, facesRes, eventsRes, zonesRes] = await Promise.all([
        getVideoTracks(videoId).catch(() => ({ tracks: [] })),
        getVehicleAttributes(videoId).catch(() => ({ attributes: [] })),
        getFaceDetections(videoId).catch(() => ({ faces: [] })),
        getSecurityEvents(videoId).catch(() => ({ events: [] })),
        getVideoZones(videoId).catch(() => ({ zones: [] })),
      ]);
      setTracks(tracksRes.tracks || []);
      setVehicleAttributes(attrsRes.attributes || []);
      setFaceDetections(facesRes.faces || []);
      setSecurityEvents(eventsRes.events || []);
      setSecurityZones(zonesRes.zones || []);
    } catch {
      // ignore
    }
  };

  // Phase 9: Incident Dossier State & Handlers
  const [reportsList, setReportsList] = useState<ReportItem[]>([]);
  const [isGeneratingReport, setIsGeneratingReport] = useState<boolean>(false);
  const [reportFeedback, setReportFeedback] = useState<{ type: "success" | "error"; message: string } | null>(null);
  const [reportTitleInput, setReportTitleInput] = useState<string>("SECURITY INCIDENT DOSSIER");

  const fetchReports = async (videoId: string) => {
    try {
      const res = await getVideoReports(videoId);
      setReportsList(res.reports || []);
    } catch {
      // ignore
    }
  };

  // Phase 15: Specialized Visual Detection State
  const [specializedList, setSpecializedList] = useState<SpecializedObservationItem[]>([]);
  const [specializedFilter, setSpecializedFilter] = useState<"all" | "fire" | "smoke" | "weapon" | "pose">("all");
  const [specializedViewMode, setSpecializedViewMode] = useState<"aggregated" | "raw">("aggregated");
  const [expandedIncidentIds, setExpandedIncidentIds] = useState<Set<string>>(new Set());
  const [isFetchingSpecialized, setIsFetchingSpecialized] = useState<boolean>(false);

  const toggleIncidentExpanded = (id: string) => {
    setExpandedIncidentIds((prev) => {
      const next = new Set(prev);
      if (next.has(id)) {
        next.delete(id);
      } else {
        next.add(id);
      }
      return next;
    });
  };

  const specializedAggregatedIncidents = useMemo(() => {
    // Extract specialized security events if available
    const specEvents = securityEvents.filter((e) =>
      ["POTENTIAL_FIRE", "POTENTIAL_SMOKE", "POTENTIAL_FIRE_SMOKE", "POTENTIAL_WEAPON_VISUAL"].includes(e.event_type)
    );
    if (specEvents.length > 0) {
      return specEvents.map((ev) => {
        const meta = ev.incident_metadata || {};
        const epId = meta.episode_id;
        const targetObsIds = Array.isArray(meta.supporting_observation_ids)
          ? new Set(meta.supporting_observation_ids)
          : null;

        let relatedObs: SpecializedObservationItem[] = [];

        if (targetObsIds && targetObsIds.size > 0) {
          relatedObs = specializedList.filter((obs) => targetObsIds.has(obs.id));
        } else if (epId) {
          relatedObs = specializedList.filter((obs) => obs.episode_id === epId);
        } else {
          // Fallback: match class and time window
          relatedObs = specializedList.filter((obs) => {
            const cls = (obs.class_name || "").toLowerCase();
            const evCls = (ev.object_class || "").toLowerCase();
            const matchesClass = ev.event_type.toLowerCase().includes(cls) || evCls.includes(cls);
            const matchesTime = obs.timestamp_seconds >= (ev.timestamp - 1.0) && obs.timestamp_seconds <= (ev.timestamp + (ev.duration_seconds || 1.0) + 1.0);
            return matchesClass && matchesTime;
          });
        }

        const authoritativeCount = meta.observation_count ?? (relatedObs.length > 0 ? relatedObs.length : 1);
        const segmentCount = meta.segment_count ?? 1;

        return {
          id: ev.id,
          event_type: ev.event_type,
          start_time: ev.timestamp,
          end_time: ev.timestamp + (ev.duration_seconds || 0.0),
          duration_seconds: ev.duration_seconds || 0.0,
          confidence: ev.confidence,
          severity: ev.severity,
          validation_status: ev.human_verification_required ? "REVIEW_REQUIRED" : "ACCEPTED",
          description: ev.description,
          observations_count: authoritativeCount,
          segment_count: segmentCount,
          supporting_observations: relatedObs,
        };
      });
    }

    // Authoritative source of truth: frontend never invents incidents when backend engine abstained
    return [];
  }, [securityEvents, specializedList]);


  const fetchSpecialized = async (videoId: string) => {
    setIsFetchingSpecialized(true);
    try {
      const res = await getSpecializedObservations(videoId);
      setSpecializedList(res.observations || []);
    } catch {
      // ignore
    } finally {
      setIsFetchingSpecialized(false);
    }
  };

  const [detectorHealth, setDetectorHealth] = useState<DetectorHealthResponse | null>(null);
  const [isFetchingHealth, setIsFetchingHealth] = useState<boolean>(false);

  const fetchHealth = async (videoId: string) => {
    setIsFetchingHealth(true);
    try {
      const res = await getVideoDetectorHealth(videoId);
      setDetectorHealth(res);
    } catch {
      // ignore
    } finally {
      setIsFetchingHealth(false);
    }
  };

  const handleGenerateReport = async () => {
    if (!uploadResult?.video_id) return;
    setIsGeneratingReport(true);
    setReportFeedback(null);
    try {
      const res = await generateVideoReport(uploadResult.video_id, {
        title: reportTitleInput.trim() || "SECURITY INCIDENT DOSSIER",
      });
      setReportsList((prev) => [res.report, ...prev]);
      setReportFeedback({
        type: "success",
        message: `Incident dossier generated successfully (Report ID: ${res.report.report_id})`,
      });
    } catch (err: any) {
      setReportFeedback({
        type: "error",
        message: err.message || "Failed to generate incident dossier.",
      });
    } finally {
      setIsGeneratingReport(false);
    }
  };

  // Monitor and handle browser playback compatibility
  useEffect(() => {
    if (!uploadResult?.video_id) {
      setPlaybackState("ready");
      setPlaybackMessage(null);
      return;
    }

    let isMounted = true;
    let pollTimer: ReturnType<typeof setTimeout> | null = null;

    const checkPlayback = async () => {
      try {
        const res = await getVideoPlaybackStatus(uploadResult.video_id);
        if (!isMounted) return;

        if (res.status === "converting") {
          setPlaybackState("preparing");
          setPlaybackMessage("Preparing browser-compatible playback…");
          pollTimer = setTimeout(checkPlayback, 1200);
        } else if (res.status === "needs_conversion") {
          setPlaybackState("preparing");
          setPlaybackMessage("Preparing browser-compatible playback…");
          // Trigger conversion via GET /playback
          fetch(getVideoPlaybackUrl(uploadResult.video_id)).catch(() => {});
          pollTimer = setTimeout(checkPlayback, 1200);
        } else if (res.status === "ready") {
          setPlaybackState("ready");
          setPlaybackMessage(null);
          if (videoRef.current) {
            const currentSrc = videoRef.current.currentSrc;
            const targetSrc = getVideoPlaybackUrl(uploadResult.video_id);
            if (!currentSrc || !currentSrc.includes("/playback")) {
              videoRef.current.src = targetSrc;
              videoRef.current.load();
            }
          }
        } else {
          setPlaybackState("unavailable");
          setPlaybackMessage("Playback unavailable");
        }
      } catch {
        if (isMounted) {
          setPlaybackState("ready");
        }
      }
    };

    checkPlayback();

    return () => {
      isMounted = false;
      if (pollTimer) clearTimeout(pollTimer);
    };
  }, [uploadResult?.video_id]);

  const handleVideoError = () => {
    if (uploadResult?.video_id) {
      setPlaybackState("preparing");
      setPlaybackMessage("Preparing browser-compatible playback…");
      // Trigger transcode and re-check
      fetch(getVideoPlaybackUrl(uploadResult.video_id)).catch(() => {});
      setTimeout(async () => {
        try {
          const res = await getVideoPlaybackStatus(uploadResult.video_id);
          if (res.status === "ready") {
            setPlaybackState("ready");
            if (videoRef.current) {
              videoRef.current.src = getVideoPlaybackUrl(uploadResult.video_id);
              videoRef.current.load();
            }
          } else if (res.status === "unavailable") {
            setPlaybackState("unavailable");
            setPlaybackMessage("Playback unavailable");
          }
        } catch {
          // ignore
        }
      }, 1500);
    }
  };

  const handleLoadExistingVideo = async (videoIdToLoad?: string) => {
    const id = (videoIdToLoad || existingVideoIdInput).trim();
    if (!id || isLoadingExisting) return;
    setIsLoadingExisting(true);
    setErrorMessage(null);
    try {
      const meta = await getVideoMetadata(id);
      setUploadResult({
        video_id: meta.video_id,
        filename: meta.filename,
        status: meta.status,
        message: "Video loaded from storage",
        file_size: meta.file_size_bytes,
        stream_url: meta.stream_url,
        playback_url: meta.playback_url,
      });

      // Synchronize processingResult if video has already been processed
      if (meta.status === "processed" && meta.duration_seconds) {
        setProcessingResult({
          video_id: meta.video_id,
          status: "completed",
          duration_seconds: meta.duration_seconds,
          fps: meta.fps || 30.0,
          frames_processed: meta.frames_processed || 0,
          detections_count: meta.detections_count || 0,
          raw_detections_count: meta.raw_detections_count || meta.detections_count || 0,
          grouped_events_count: meta.grouped_events_count,
        });
      }

      // Fetch timeline and events
      try {
        const timelineRes = await getVideoTimeline(id);
        setTimelineEvents(timelineRes.events || []);
        const eventsResponse = await getVideoEvents(id, { include_unvalidated: true });
        setEvents(eventsResponse.events || []);
        if (timelineRes.events && timelineRes.events.length > 0) {
          setProcessingState("completed");
        }
      } catch {
        // Not yet processed
      }

      fetchEvidence(id);
      fetchSecurityIntelligence(id);
      fetchReports(id);
      fetchSpecialized(id);
    } catch (err: unknown) {
      setErrorMessage(err instanceof Error ? err.message : `Failed to load video '${id}'.`);
    } finally {
      setIsLoadingExisting(false);
    }
  };

  // Load initial video if passed via props
  useEffect(() => {
    if (initialVideoId) {
      handleLoadExistingVideo(initialVideoId);
    }
  }, [initialVideoId]);

  const handleCreateCaseFromThisVideo = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!uploadResult?.video_id) return;
    setIsCaseSubmitting(true);
    setCaseWorkflowFeedback(null);
    try {
      const created = await createCase({
        title: caseTitleInput.trim() || `Security Investigation: ${uploadResult.filename}`,
        description: caseDescInput.trim() || `Security investigation case created directly from video ${uploadResult.filename}.`,
        priority: casePriorityInput,
        assigned_investigator: "Security Analyst (Local Session)",
        tags: ["CCTV", "Video Intelligence"],
      });
      await linkVideoToCase(created.id, uploadResult.video_id);
      setCaseWorkflowFeedback(`Case ${created.case_number || created.id} created! Navigating to case workspace...`);
      setTimeout(() => {
        setIsCreateCaseModalOpen(false);
        setCaseWorkflowFeedback(null);
        if (onNavigateToCase) {
          onNavigateToCase(created.id);
        }
      }, 800);
    } catch (err: any) {
      setCaseWorkflowFeedback(`Error: ${err.message}`);
    } finally {
      setIsCaseSubmitting(false);
    }
  };

  const handleAddToActiveCase = async () => {
    if (!activeCaseId || !uploadResult?.video_id) return;
    try {
      await linkVideoToCase(activeCaseId, uploadResult.video_id);
      alert(`Video '${uploadResult.filename}' linked to active case.`);
      if (onNavigateToCase) {
        onNavigateToCase(activeCaseId);
      }
    } catch (err: any) {
      alert(`Failed to add to active case: ${err.message}`);
    }
  };

  const handleRunSecurityAnalysis = async () => {
    if (!uploadResult?.video_id) return;
    setIsAnalyzingIntelligence(true);
    setIntelligenceFeedback(null);
    try {
      await runSecurityAnalysis(uploadResult.video_id);
      await fetchSecurityIntelligence(uploadResult.video_id);
      setIntelligenceFeedback("Security Intelligence analysis completed and synchronized.");
    } catch (err: unknown) {
      setIntelligenceFeedback(err instanceof Error ? err.message : "Intelligence analysis failed.");
    } finally {
      setIsAnalyzingIntelligence(false);
    }
  };

  const handleCreateZone = async () => {
    if (!uploadResult?.video_id || !newZoneName.trim()) return;
    setIsCreatingZone(true);
    try {
      let polygon = [[100, 100], [300, 100], [300, 300], [100, 300]];
      if (newZonePreset === "gate") {
        polygon = [[50, 50], [250, 50], [250, 350], [50, 350]];
      } else if (newZonePreset === "perimeter") {
        polygon = [[0, 0], [400, 0], [400, 150], [0, 150]];
      } else if (newZonePreset === "loading") {
        polygon = [[150, 150], [380, 150], [380, 380], [150, 380]];
      }
      await createVideoZone(uploadResult.video_id, {
        name: newZoneName.trim(),
        polygon,
        target_classes: ["person", "car"],
        alert_on_entry: true,
        loitering_threshold_seconds: 25.0,
      });
      setNewZoneName("");
      await fetchSecurityIntelligence(uploadResult.video_id);
    } catch (err: unknown) {
      alert(err instanceof Error ? err.message : "Failed to create zone.");
    } finally {
      setIsCreatingZone(false);
    }
  };

  const handleDeleteZone = async (zoneId: string) => {
    if (!uploadResult?.video_id) return;
    try {
      await deleteVideoZone(uploadResult.video_id, zoneId);
      await fetchSecurityIntelligence(uploadResult.video_id);
    } catch (err: unknown) {
      alert(err instanceof Error ? err.message : "Failed to delete zone.");
    }
  };

  const handleCaptureEvidence = async (
    timestamp: number,
    eventId?: string | null,
    evidenceType: "snapshot_and_clip" | "snapshot_only" | "clip_only" = "snapshot_and_clip"
  ) => {
    if (!uploadResult?.video_id) return;
    const captureKey = eventId || `${timestamp}`;
    setCapturingEventId(captureKey);
    setEvidenceFeedback(null);
    try {
      const res = await createEvidence(uploadResult.video_id, {
        timestamp,
        event_id: eventId,
        evidence_type: evidenceType,
        pre_seconds: 3.0,
        post_seconds: 3.0,
      });
      setEvidenceFeedback({
        type: "success",
        message: res.is_duplicate
          ? `Evidence #EV-${res.evidence_id.slice(-6)} already verified in vault.`
          : `Evidence #EV-${res.evidence_id.slice(-6)} captured successfully (${res.evidence_type.replace(/_/g, " ")}).`,
      });
      // Refresh evidence list
      const updated = await getVideoEvidence(uploadResult.video_id);
      setEvidenceList(updated.evidence);
    } catch (err: unknown) {
      setEvidenceFeedback({
        type: "error",
        message: err instanceof Error ? err.message : "Evidence capture failed.",
      });
    } finally {
      setCapturingEventId(null);
    }
  };

  const AI_EXAMPLE_QUESTIONS = [
    "Summarize this video",
    "What happened around 8 seconds?",
    "What were the main detected objects?",
    "Were there any noteworthy activity periods?",
    "Which events should I review?",
    "Show me the most relevant evidence",
  ];

  const DETERMINISTIC_EXAMPLE_QUESTIONS = [
    "Show all people detected",
    "What vehicles were detected?",
    "Show cars between 8 and 12 seconds",
    "Show detections above 70% confidence",
    "How many people were detected?",
    "Show events between 10 and 15 seconds",
  ];

  const handleInvestigate = async (queryToRun?: string) => {
    const q = queryToRun !== undefined ? queryToRun : investigationQuery;
    if (!q.trim() || !uploadResult?.video_id || isInvestigating) return;
    setInvestigationQuery(q);
    setIsInvestigating(true);
    setInvestigationError(null);

    if (investigationMode === "ai") {
      try {
        const res = await aiInvestigateVideo(uploadResult.video_id, {
          query: q.trim(),
          history: aiHistory,
        });
        setAiResponse(res);
        setAiHistory((prev) => [
          ...prev.slice(-4),
          { role: "user", content: q.trim() },
          { role: "assistant", content: res.answer },
        ]);
      } catch (err: unknown) {
        setInvestigationError(
          err instanceof Error ? err.message : "AI Investigation service unavailable. Please try again."
        );
      } finally {
        setIsInvestigating(false);
      }
    } else {
      try {
        const res = await investigateVideo(uploadResult.video_id, q.trim());
        setInvestigationResponse(res);
      } catch (err: unknown) {
        setInvestigationError(
          err instanceof Error ? err.message : "Investigation service unavailable. Please try again."
        );
      } finally {
        setIsInvestigating(false);
      }
    }
  };

  // Expanded events state for Phase 4.1
  const [expandedEventIds, setExpandedEventIds] = useState<Record<string, boolean>>({});

  const toggleEventExpanded = (eventId: string, e: React.MouseEvent) => {
    e.stopPropagation();
    setExpandedEventIds((prev) => ({ ...prev, [eventId]: !prev[eventId] }));
  };

  // Helper to filter raw detections that belong to a grouped event time window
  const getEventDetections = (groupedEvent: GroupedEvent): DetectionEvent[] => {
    return events.filter((det) => {
      // Grouped timeline events reflect validated observations only
      if (det.validation_status === "REJECTED") {
        return false;
      }
      // Must fall within event time window (allowing small float margin)
      if (det.timestamp < groupedEvent.start_time - 0.05 || det.timestamp > groupedEvent.end_time + 0.05) {
        return false;
      }
      // Must satisfy active class filter
      if (classFilter !== "all" && det.object_class !== classFilter) {
        return false;
      }
      // Must satisfy active min confidence filter
      if (det.confidence < minConfidence) {
        return false;
      }
      return true;
    });
  };

  // ---- Refs ----
  const fileInputRef = useRef<HTMLInputElement>(null);
  const videoRef = useRef<HTMLVideoElement>(null);

  // Derived class list from timeline & raw events
  const availableClasses = Array.from(
    new Set([
      ...events.map((e) => e.object_class).filter((c): c is string => Boolean(c)),
      ...timelineEvents.flatMap((g) => g.objects.map((o) => o.class)).filter((c): c is string => Boolean(c)),
    ])
  ).sort();

  const filteredTimelineEvents = timelineEvents.filter((g) => {
    if (classFilter !== "all" && !g.objects.some((o) => o.class === classFilter)) return false;
    if (g.max_confidence < minConfidence) return false;
    return true;
  });

  const filteredRawEvents = events.filter((e) => {
    if (classFilter !== "all" && e.object_class !== classFilter) return false;
    if (e.confidence < minConfidence) return false;
    return true;
  });

  // ---- File validation ----
  const validateFile = (file: File): string | null => {
    if (!file) return "No file selected.";
    if (file.size === 0) return "Selected file is empty (0 bytes). Please choose a valid video.";
    const ext = "." + file.name.split(".").pop()?.toLowerCase();
    if (!SUPPORTED_EXTENSIONS.includes(ext))
      return `Unsupported format '${ext}'. Supported: ${SUPPORTED_EXTENSIONS.join(", ")}`;
    if (file.size > MAX_SIZE_MB * 1024 * 1024)
      return `File exceeds maximum limit of ${MAX_SIZE_MB}MB.`;
    return null;
  };

  const handleFile = (file: File) => {
    setErrorMessage(null);
    setUploadResult(null);
    setUploadProgress(0);
    setProcessingState("idle");
    setProcessingResult(null);
    setProcessingError(null);
    setEvents([]);
    setTimelineEvents([]);

    const validationError = validateFile(file);
    if (validationError) {
      setErrorMessage(validationError);
      setSelectedFile(null);
      if (previewUrl) {
        URL.revokeObjectURL(previewUrl);
        setPreviewUrl(null);
      }
      return;
    }

    setSelectedFile(file);
    if (previewUrl) URL.revokeObjectURL(previewUrl);
    setPreviewUrl(URL.createObjectURL(file));
  };

  // ---- Drag & Drop Handler ----
  const handleDrag = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive(true);
    } else if (e.type === "dragleave") {
      setDragActive(false);
    }
  };

  const handleDrop = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    if (e.dataTransfer.files?.[0]) handleFile(e.dataTransfer.files[0]);
  };

  const handleFileInputChange = (e: ChangeEvent<HTMLInputElement>) => {
    if (e.target.files?.[0]) handleFile(e.target.files[0]);
  };

  const handleBrowseClick = () => {
    if (fileInputRef.current && !isUploading) fileInputRef.current.click();
  };

  // ---- Upload ----
  const handleUpload = async () => {
    if (!selectedFile || isUploading) return;
    setIsUploading(true);
    setUploadProgress(0);
    setErrorMessage(null);
    try {
      const response = await uploadVideo(selectedFile, setUploadProgress);
      setUploadResult(response);
      fetchEvidence(response.video_id);
      if (activeCaseId) {
        try {
          await linkVideoToCase(activeCaseId, response.video_id);
          setCaseWorkflowFeedback("Video automatically associated with active case.");
        } catch (linkErr: any) {
          console.warn("Auto-link video to active case error:", linkErr);
        }
      }
    } catch (err: unknown) {
      setErrorMessage(err instanceof Error ? err.message : "Failed to upload video.");
    } finally {
      setIsUploading(false);
    }
  };

  // ---- Phase 4: Analyze Video + Fetch Timeline ----
  const handleAnalyze = async () => {
    if (!uploadResult?.video_id || processingState === "processing") return;
    setProcessingState("processing");
    setProcessingError(null);
    setEvents([]);
    setTimelineEvents([]);
    setProcessingResult(null);

    try {
      const result = await processVideo(uploadResult.video_id);
      setProcessingResult(result);

      // Fetch grouped timeline events and raw events
      const timelineRes = await getVideoTimeline(uploadResult.video_id);
      setTimelineEvents(timelineRes.events);

      const eventsResponse = await getVideoEvents(uploadResult.video_id, { include_unvalidated: true });
      setEvents(eventsResponse.events);

      // Fetch existing evidence
      fetchEvidence(uploadResult.video_id);

      // Fetch Phase 8 security intelligence
      fetchSecurityIntelligence(uploadResult.video_id);

      // Fetch Phase 9 reports
      fetchReports(uploadResult.video_id);

      // Fetch Phase 15 specialized visual observations
      fetchSpecialized(uploadResult.video_id);

      setProcessingState("completed");
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : "Processing failed.";
      setProcessingError(msg);
      setProcessingState("failed");
    }
  };

  // ---- Seek video to timestamp ----
  const handleSeekToTimestamp = (timestamp: number, eventId?: string | null) => {
    if (eventId) setActiveEventId(eventId);
    if (videoRef.current) {
      videoRef.current.currentTime = timestamp;
      videoRef.current.play().catch(() => {/* user interaction fallback */});
      videoRef.current.scrollIntoView({ behavior: "smooth", block: "nearest" });
    }
  };

  // ---- Reset ----
  const handleReset = () => {
    if (previewUrl) URL.revokeObjectURL(previewUrl);
    setSelectedFile(null);
    setPreviewUrl(null);
    setUploadResult(null);
    setErrorMessage(null);
    setUploadProgress(0);
    setCopiedId(false);
    setProcessingState("idle");
    setProcessingResult(null);
    setProcessingError(null);
    setEvents([]);
    setTimelineEvents([]);
    setClassFilter("all");
    setMinConfidence(0);
    setActiveEventId(null);
    setInvestigationQuery("");
    setIsInvestigating(false);
    setInvestigationResponse(null);
    setInvestigationError(null);
    setEvidenceList([]);
    setCapturingEventId(null);
    setEvidenceFeedback(null);
    setSelectedEvidence(null);
    if (fileInputRef.current) fileInputRef.current.value = "";
  };

  const handleCopyId = () => {
    if (uploadResult?.video_id) {
      navigator.clipboard.writeText(uploadResult.video_id);
      setCopiedId(true);
      setTimeout(() => setCopiedId(false), 2500);
    }
  };

  const formatBytes = (bytes: number): string => {
    if (bytes === 0) return "0 Bytes";
    const k = 1024;
    const sizes = ["Bytes", "KB", "MB", "GB"];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + " " + sizes[i];
  };

  return (
    <div className="w-full bg-zinc-900/60 border border-zinc-800 rounded-xl p-6 md:p-8 space-y-6">

      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 border-b border-zinc-800 pb-4">
        <div>
          <h2 className="text-lg font-semibold text-zinc-100 flex items-center gap-2">
            <span className="flex h-2.5 w-2.5 rounded-full bg-emerald-500 animate-pulse" />
            Surveillance Video Ingestion & Intelligence Workbench
          </h2>
          <p className="text-xs text-zinc-400 mt-0.5">
            Surveillance Video Pipeline: Ingest &rarr; Decoding &rarr; YOLO Object Detection &rarr; Spatial Intelligence &rarr; Incident Correlation &rarr; Forensic Search
          </p>
        </div>

        {uploadResult && (
          <button
            onClick={handleReset}
            className="self-start sm:self-auto text-xs font-mono text-zinc-400 hover:text-zinc-200 border border-zinc-700 hover:border-zinc-500 px-3 py-1.5 rounded-lg transition-colors"
          >
            + Upload Another Video
          </button>
        )}
      </div>

      {/* Error Alert */}
      {errorMessage && (
        <div className="rounded-lg bg-red-500/10 border border-red-500/30 p-4 flex items-start justify-between gap-3 text-sm text-red-300">
          <div className="flex items-start gap-2">
            <svg className="w-5 h-5 text-red-400 shrink-0 mt-0.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
            </svg>
            <div>
              <span className="font-semibold text-red-400">Error: </span>
              {errorMessage}
            </div>
          </div>
          <button onClick={() => setErrorMessage(null)} className="text-red-400 hover:text-red-200 text-xs font-mono">
            Dismiss
          </button>
        </div>
      )}

      {/* Upload Zone */}
      {!uploadResult && (
        <div className="space-y-4">
          <div
            onDragEnter={handleDrag} onDragLeave={handleDrag} onDragOver={handleDrag} onDrop={handleDrop}
            onClick={handleBrowseClick}
            className={`relative border-2 border-dashed rounded-xl p-8 md:p-12 text-center cursor-pointer transition-all duration-200 ${
              dragActive
                ? "border-emerald-500 bg-emerald-500/5 ring-4 ring-emerald-500/10"
                : "border-zinc-700 hover:border-zinc-500 bg-zinc-950/40 hover:bg-zinc-900/40"
            }`}
          >
            <input
              ref={fileInputRef}
              type="file"
              accept=".mp4,.mov,.avi,.mkv,.webm,video/mp4,video/quicktime,video/x-msvideo"
              onChange={handleFileInputChange}
              className="hidden"
              disabled={isUploading}
            />
            <div className="flex flex-col items-center justify-center space-y-3">
              <div className="p-3 bg-zinc-800/80 border border-zinc-700 rounded-full text-zinc-300">
                <svg className="w-8 h-8 text-emerald-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.75}
                    d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
                </svg>
              </div>
              <div className="space-y-1">
                <p className="text-sm font-medium text-zinc-200">
                  <span className="text-emerald-400 underline underline-offset-2">Click to browse</span>{" "}
                  or drag and drop surveillance video here
                </p>
                <p className="text-xs text-zinc-500 font-mono">
                  Supported: MP4, MOV, AVI, MKV, WebM (Max {MAX_SIZE_MB}MB)
                </p>
              </div>
            </div>
          </div>

          {/* Selected File Card */}
          {selectedFile && (
            <div className="bg-zinc-950 border border-zinc-800 rounded-lg p-4 flex flex-col sm:flex-row sm:items-center justify-between gap-4">
              <div className="flex items-center gap-3 overflow-hidden">
                <div className="p-2 bg-zinc-900 border border-zinc-700 rounded text-emerald-400 shrink-0">
                  <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
                      d="M15 10l4.553-2.276A1 1 0 0121 8.618v6.764a1 1 0 01-1.447.894L15 14M5 18h8a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z" />
                  </svg>
                </div>
                <div className="min-w-0">
                  <p className="text-sm font-medium text-zinc-200 truncate">{selectedFile.name}</p>
                  <p className="text-xs text-zinc-500 font-mono">{formatBytes(selectedFile.size)} • {selectedFile.type || "video"}</p>
                </div>
              </div>
              <div className="flex items-center gap-2 self-end sm:self-auto">
                <button onClick={handleReset} disabled={isUploading}
                  className="text-xs font-mono text-zinc-400 hover:text-zinc-200 px-3 py-2 rounded-lg border border-zinc-800 hover:border-zinc-700 disabled:opacity-50 transition-colors">
                  Cancel
                </button>
                <button onClick={handleUpload} disabled={isUploading}
                  className="flex items-center gap-2 bg-emerald-600 hover:bg-emerald-500 disabled:bg-emerald-800/50 text-white text-xs font-medium px-4 py-2 rounded-lg shadow transition-colors">
                  {isUploading ? (
                    <>
                      <svg className="animate-spin h-4 w-4" fill="none" viewBox="0 0 24 24">
                        <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                        <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
                      </svg>
                      <span>Uploading ({uploadProgress}%)</span>
                    </>
                  ) : (
                    <>
                      <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-8l-4-4m0 0L8 8m4-4v12" />
                      </svg>
                      <span>Upload Video</span>
                    </>
                  )}
                </button>
              </div>
            </div>
          )}

          {/* Advanced / Developer Tools: Load Video by Reference ID */}
          <details className="group bg-zinc-950/70 border border-zinc-800 rounded-lg p-3 text-xs transition-all">
            <summary className="cursor-pointer text-zinc-400 font-mono flex items-center justify-between select-none">
              <span className="flex items-center gap-2">
                <span className="h-1.5 w-1.5 rounded-full bg-zinc-500 group-open:bg-blue-400" />
                <span className="font-medium text-zinc-300">Advanced / Developer Tools: Direct Video Reference ID</span>
              </span>
              <span className="text-zinc-500 group-open:rotate-180 transition-transform text-[11px]">&darr;</span>
            </summary>
            <div className="mt-3 pt-3 border-t border-zinc-800/80 flex flex-col sm:flex-row items-center justify-between gap-3">
              <div className="text-zinc-400 font-mono text-[11px]">
                Inspect existing surveillance recording by UUID:
              </div>
              <div className="flex items-center gap-2 w-full sm:w-auto">
                <input
                  type="text"
                  value={existingVideoIdInput}
                  onChange={(e) => setExistingVideoIdInput(e.target.value)}
                  onKeyDown={(e) => {
                    if (e.key === "Enter") {
                      e.preventDefault();
                      handleLoadExistingVideo();
                    }
                  }}
                  placeholder="Surveillance Video ID (UUID)"
                  className="bg-zinc-900 border border-zinc-700 rounded px-3 py-1.5 text-zinc-200 font-mono text-xs w-full sm:w-72 focus:outline-none focus:border-emerald-500"
                />
                <button
                  type="button"
                  onClick={() => handleLoadExistingVideo()}
                  disabled={isLoadingExisting || !existingVideoIdInput.trim()}
                  className="px-3.5 py-1.5 bg-zinc-800 hover:bg-zinc-700 text-zinc-200 rounded font-mono text-xs border border-zinc-700 disabled:opacity-40 transition-colors shrink-0 font-medium cursor-pointer"
                >
                  {isLoadingExisting ? "Loading..." : "Load Video"}
                </button>
              </div>
            </div>
          </details>

          {/* Progress Bar */}
          {isUploading && (
            <div className="space-y-1.5 pt-1">
              <div className="flex justify-between text-xs font-mono text-zinc-400">
                <span>Streaming to storage/uploads/...</span>
                <span className="text-emerald-400">{uploadProgress}%</span>
              </div>
              <div className="w-full bg-zinc-800 rounded-full h-2 overflow-hidden">
                <div className="bg-emerald-500 h-2 rounded-full transition-all duration-150" style={{ width: `${uploadProgress}%` }} />
              </div>
            </div>
          )}
        </div>
      )}

      {/* Success State + Video Player + Phase 4 Controls */}
      {uploadResult && (
        <div className="space-y-6 animate-in fade-in duration-300">
          {/* Success Banner */}
          <div className="rounded-lg bg-emerald-500/10 border border-emerald-500/30 p-4 flex flex-col sm:flex-row sm:items-center justify-between gap-3">
            <div className="flex items-center gap-3">
              <div className="p-1.5 bg-emerald-500/20 text-emerald-400 rounded-full">
                <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                </svg>
              </div>
              <div>
                <p className="text-sm font-semibold text-emerald-400">
                  {uploadResult.message || "Video successfully uploaded"}
                </p>
                <p className="text-xs text-zinc-400 font-mono">
                  Stored under: storage/uploads/{uploadResult.filename}
                </p>
              </div>
            </div>
            <div className="flex items-center gap-2 bg-zinc-950/80 border border-zinc-800 px-3 py-1.5 rounded-md text-xs font-mono">
              <span className="text-zinc-500">video_id:</span>
              <span className="text-zinc-200 truncate max-w-[140px] sm:max-w-[200px]">{uploadResult.video_id}</span>
              <button onClick={handleCopyId} className="text-emerald-400 hover:text-emerald-300 text-[11px] underline ml-1" title="Copy Video ID">
                {copiedId ? "Copied!" : "Copy"}
              </button>
            </div>
          </div>

          {/* Contextual Video -> Case Actions Bar */}
          <div className="flex flex-wrap items-center justify-between gap-3 bg-zinc-900/90 border border-zinc-800 p-3.5 rounded-xl shadow-md font-mono text-xs">
            <div className="flex items-center gap-2">
              <span className="w-2.5 h-2.5 rounded-full bg-emerald-400" />
              <span className="text-zinc-300 font-sans font-semibold text-sm">Security Video Workflow:</span>
              <span className="text-zinc-400 truncate max-w-xs">{uploadResult.filename}</span>
            </div>

            <div className="flex items-center gap-2 flex-wrap">
              <button
                type="button"
                onClick={() => {
                  setCaseTitleInput(`Security Investigation: ${uploadResult.filename}`);
                  setCaseDescInput(`Case created from ingested surveillance video ${uploadResult.filename}.`);
                  setIsCreateCaseModalOpen(true);
                }}
                className="flex items-center gap-1.5 px-3.5 py-1.5 bg-emerald-600 hover:bg-emerald-500 text-zinc-950 font-bold rounded-lg transition-all shadow-sm font-sans cursor-pointer text-xs"
              >
                <span>📁</span>
                <span>Create Case from This Video</span>
              </button>

              {activeCaseId && (
                <button
                  type="button"
                  onClick={handleAddToActiveCase}
                  className="flex items-center gap-1.5 px-3.5 py-1.5 bg-blue-600/20 hover:bg-blue-600/30 text-blue-300 border border-blue-500/40 rounded-lg transition-all font-sans font-semibold cursor-pointer text-xs"
                >
                  <span>+</span>
                  <span>Add to Active Case</span>
                </button>
              )}

              {onNavigateToIncidents && (
                <button
                  type="button"
                  onClick={onNavigateToIncidents}
                  className="px-3 py-1.5 bg-zinc-800 hover:bg-zinc-700 text-zinc-300 rounded-lg transition-all font-sans cursor-pointer text-xs"
                >
                  View Incidents
                </button>
              )}

              {onNavigateToEvidence && (
                <button
                  type="button"
                  onClick={onNavigateToEvidence}
                  className="px-3 py-1.5 bg-zinc-800 hover:bg-zinc-700 text-zinc-300 rounded-lg transition-all font-sans cursor-pointer text-xs"
                >
                  View Evidence
                </button>
              )}
            </div>
          </div>

          {/* Video Player */}
          <div className="space-y-2">
            <div className="flex items-center justify-between text-xs text-zinc-400 font-mono">
              <span>Sentinel Surveillance Player</span>
              {processingState === "completed" ? (
                <span className="text-emerald-400 font-semibold">
                  Detection &amp; Event Intelligence Complete — {timelineEvents.length} Grouped Timeline Events ({events.length} Raw)
                </span>
              ) : (
                <span className="text-emerald-400 font-semibold">Ready for Analysis</span>
              )}
            </div>
            <div className="relative rounded-xl overflow-hidden bg-black border border-zinc-800 aspect-video flex items-center justify-center">
              {playbackState === "preparing" && (
                <div className="absolute inset-0 z-10 bg-zinc-950/85 backdrop-blur-sm flex flex-col items-center justify-center p-6 text-center space-y-3 animate-in fade-in duration-200">
                  <div className="p-3 bg-emerald-500/10 border border-emerald-500/30 rounded-full text-emerald-400 animate-pulse">
                    <svg className="animate-spin h-6 w-6 text-emerald-400" fill="none" viewBox="0 0 24 24">
                      <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                      <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
                    </svg>
                  </div>
                  <div className="space-y-1">
                    <p className="text-sm font-semibold text-zinc-100">
                      Preparing browser-compatible playback…
                    </p>
                    <p className="text-xs text-zinc-400 font-mono max-w-sm">
                      Transcoding video stream to standard H.264 (AVC) for smooth HTML5 playback and timeline seeking.
                    </p>
                  </div>
                </div>
              )}

              {playbackState === "unavailable" && (
                <div className="absolute inset-0 z-10 bg-zinc-950/90 flex flex-col items-center justify-center p-6 text-center space-y-2">
                  <div className="p-2.5 bg-red-500/10 border border-red-500/30 rounded-full text-red-400">
                    <svg className="w-6 h-6 text-red-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
                    </svg>
                  </div>
                  <p className="text-sm font-semibold text-zinc-200">Playback unavailable</p>
                  <p className="text-xs text-zinc-500">Video format could not be decoded for browser streaming.</p>
                </div>
              )}

              <video
                ref={videoRef}
                controls
                className="w-full h-full object-contain"
                src={uploadResult ? getVideoPlaybackUrl(uploadResult.video_id) : (previewUrl || "")}
                onError={handleVideoError}
              >
                Your browser does not support the HTML5 video tag.
              </video>
            </div>
          </div>

          {/* Video Attributes */}
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 text-xs font-mono bg-zinc-950 p-4 rounded-lg border border-zinc-800">
            <div>
              <span className="text-zinc-500 block">Original Name</span>
              <span className="text-zinc-200 truncate block font-sans font-medium" title={uploadResult.filename}>{uploadResult.filename}</span>
            </div>
            <div>
              <span className="text-zinc-500 block">Status</span>
              <span className={`font-semibold uppercase ${processingState === "completed" ? "text-blue-400" : "text-emerald-400"}`}>
                {processingState === "completed" ? "PROCESSED" : uploadResult.status}
              </span>
            </div>
            <div>
              <span className="text-zinc-500 block">File Size</span>
              <span className="text-zinc-200">{uploadResult.file_size ? formatBytes(uploadResult.file_size) : "Persisted"}</span>
            </div>
            <div>
              <span className="text-zinc-500 block">Database</span>
              <span className="text-emerald-400 font-semibold">PostgreSQL / SQLite</span>
            </div>
            {processingResult && (
              <>
                <div>
                  <span className="text-zinc-500 block">Duration</span>
                  <span className="text-zinc-200">{processingResult.duration_seconds.toFixed(1)}s</span>
                </div>
                <div>
                  <span className="text-zinc-500 block">FPS</span>
                  <span className="text-zinc-200">{processingResult.fps.toFixed(1)}</span>
                </div>
                <div>
                  <span className="text-zinc-500 block">Raw Observations</span>
                  <span className="text-blue-400 font-semibold">{processingResult.raw_detections_count ?? processingResult.detections_count}</span>
                </div>
                <div>
                  <span className="text-zinc-500 block">Validated Detections</span>
                  <span className="text-emerald-400 font-semibold">{processingResult.detections_count}</span>
                </div>
                <div>
                  <span className="text-zinc-500 block">Timeline Events</span>
                  <span className="text-emerald-400 font-bold">{processingResult.grouped_events_count ?? timelineEvents.length}</span>
                </div>
              </>
            )}
          </div>

          {/* ---- PHASE 4: INTELLIGENCE ANALYSIS & TIMELINE WORKBENCH ---- */}
          <div className="border-t border-zinc-800 pt-4 space-y-4">
            <div className="flex items-center justify-between">
              <div>
                <h3 className="text-sm font-semibold text-zinc-200 flex items-center gap-2">
                  <svg className="w-4 h-4 text-emerald-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
                      d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
                  </svg>
                  Investigator Event Timeline
                </h3>
                <p className="text-xs text-zinc-500 mt-0.5">
                  Raw YOLO detections aggregated into structured, timestamped investigation events
                </p>
              </div>

              {/* ANALYZE / RE-ANALYZE Button */}
              {processingState !== "completed" ? (
                <button
                  id="analyze-video-btn"
                  onClick={handleAnalyze}
                  disabled={processingState === "processing"}
                  className={`flex items-center gap-2 px-5 py-2.5 rounded-lg text-sm font-semibold transition-all shadow-lg ${
                    processingState === "processing"
                      ? "bg-blue-700/50 text-blue-300 cursor-not-allowed"
                      : processingState === "failed"
                      ? "bg-red-600 hover:bg-red-500 text-white"
                      : "bg-emerald-600 hover:bg-emerald-500 text-white hover:shadow-emerald-500/20"
                  }`}
                >
                  {processingState === "processing" ? (
                    <>
                      <svg className="animate-spin h-4 w-4" fill="none" viewBox="0 0 24 24">
                        <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                        <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
                      </svg>
                      Processing &amp; Grouping...
                    </>
                  ) : (
                    <>
                      <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
                          d="M14.752 11.168l-3.197-2.132A1 1 0 0010 9.87v4.263a1 1 0 001.555.832l3.197-2.132a1 1 0 000-1.664z" />
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
                          d="M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                      </svg>
                      ANALYZE &amp; GENERATE TIMELINE
                    </>
                  )}
                </button>
              ) : (
                <button
                  onClick={handleAnalyze}
                  className="flex items-center gap-2 px-4 py-2 rounded-lg text-xs font-mono text-zinc-400 hover:text-zinc-200 border border-zinc-700 hover:border-zinc-500 transition-colors"
                >
                  Re-analyze
                </button>
              )}
            </div>

            {/* Processing Feedback */}
            {processingState === "processing" && (
              <div className="rounded-lg bg-emerald-500/10 border border-emerald-500/30 p-4">
                <div className="flex items-center gap-3">
                  <svg className="animate-spin h-5 w-5 text-emerald-400 shrink-0" fill="none" viewBox="0 0 24 24">
                    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                    <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
                  </svg>
                  <div>
                    <p className="text-sm font-semibold text-emerald-400">Extracting Detections &amp; Grouping Events...</p>
                    <p className="text-xs text-zinc-400 mt-0.5">
                      Running OpenCV frame sampling, YOLOv8 detection, temporal event window grouping, and relational DB persistence.
                    </p>
                  </div>
                </div>
              </div>
            )}

            {processingState === "failed" && processingError && (
              <div className="rounded-lg bg-red-500/10 border border-red-500/30 p-4 flex items-start justify-between gap-3">
                <div className="flex items-start gap-2">
                  <svg className="w-5 h-5 text-red-400 shrink-0 mt-0.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                  </svg>
                  <div>
                    <p className="text-sm font-semibold text-red-400">Processing Failed</p>
                    <p className="text-xs text-zinc-400 mt-0.5">{processingError}</p>
                  </div>
                </div>
                <button onClick={() => setProcessingError(null)} className="text-red-400 hover:text-red-200 text-xs font-mono">Dismiss</button>
              </div>
            )}

            {/* Results & Timeline Display */}
            {processingState === "completed" && (
              <div className="space-y-6">

                {/* ========================================================================= */}
                {/* PHASE 5A: ASK SENTINEL — NATURAL-LANGUAGE INVESTIGATION PANEL             */}
                {/* ========================================================================= */}
                <div id="ask-sentinel-panel" className="bg-zinc-950/90 border border-emerald-500/30 rounded-xl p-5 md:p-6 space-y-4 shadow-xl">
                  <div className="flex flex-col md:flex-row md:items-center justify-between gap-3 border-b border-zinc-800 pb-3">
                    <div className="flex items-center gap-2.5">
                      <div className="p-1.5 bg-emerald-500/10 border border-emerald-500/30 rounded-md text-emerald-400">
                        <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 10h.01M12 10h.01M16 10h.01M9 16H5a2 2 0 01-2-2V6a2 2 0 012-2h14a2 2 0 012 2v8a2 2 0 01-2 2h-5l-5 5v-5z" />
                        </svg>
                      </div>
                      <div>
                        <h3 className="text-sm font-bold text-zinc-100 flex items-center gap-2">
                          ASK SENTINEL
                          <span className="text-[10px] font-mono font-semibold uppercase bg-emerald-500/20 text-emerald-400 border border-emerald-500/40 px-2 py-0.5 rounded">
                            {investigationMode === "ai" ? "Evidence-Grounded AI" : "Deterministic Engine"}
                          </span>
                        </h3>
                        <p className="text-xs text-zinc-400">
                          {investigationMode === "ai"
                            ? "Ask natural questions grounded strictly in database events & evidence"
                            : "Query video detection and event records with deterministic filters"}
                        </p>
                      </div>
                    </div>

                    <div className="flex items-center gap-2">
                      {/* Mode Switcher */}
                      <div className="flex items-center gap-1 bg-zinc-900 border border-zinc-800 p-1 rounded-lg">
                        <button
                          type="button"
                          onClick={() => {
                            setInvestigationMode("ai");
                            setInvestigationError(null);
                          }}
                          className={`px-2.5 py-1 rounded text-xs font-mono font-semibold transition-all flex items-center gap-1.5 ${
                            investigationMode === "ai"
                              ? "bg-emerald-600 text-white shadow-sm"
                              : "text-zinc-400 hover:text-zinc-200"
                          }`}
                        >
                          <span className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse"></span>
                          AI-ASSISTED
                        </button>
                        <button
                          type="button"
                          onClick={() => {
                            setInvestigationMode("deterministic");
                            setInvestigationError(null);
                          }}
                          className={`px-2.5 py-1 rounded text-xs font-mono font-semibold transition-all ${
                            investigationMode === "deterministic"
                              ? "bg-zinc-700 text-white shadow-sm"
                              : "text-zinc-400 hover:text-zinc-200"
                          }`}
                        >
                          DETERMINISTIC
                        </button>
                      </div>

                      {(aiResponse || investigationResponse) && (
                        <button
                          onClick={() => {
                            setAiResponse(null);
                            setInvestigationResponse(null);
                            setInvestigationQuery("");
                            setInvestigationError(null);
                          }}
                          className="text-xs font-mono text-zinc-500 hover:text-zinc-300 transition-colors ml-1"
                        >
                          Clear
                        </button>
                      )}
                    </div>
                  </div>

                  {/* Search Input Bar */}
                  <form
                    onSubmit={(e) => {
                      e.preventDefault();
                      handleInvestigate();
                    }}
                    className="flex flex-col sm:flex-row gap-2"
                  >
                    <div className="relative flex-1">
                      <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none text-zinc-500">
                        <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
                        </svg>
                      </div>
                      <input
                        id="investigation-query-input"
                        type="text"
                        value={investigationQuery}
                        onChange={(e) => setInvestigationQuery(e.target.value)}
                        placeholder={
                          investigationMode === "ai"
                            ? "Ask anything about this video (e.g. 'Summarize this video', 'What happened around 8s?')..."
                            : "Query detections (e.g. 'Show cars between 8 and 12 seconds')..."
                        }
                        className="w-full pl-9 pr-4 py-2.5 bg-zinc-900 border border-zinc-700 hover:border-zinc-600 focus:border-emerald-500 focus:ring-1 focus:ring-emerald-500 rounded-lg text-sm text-zinc-100 placeholder-zinc-500 focus:outline-none transition-all font-sans"
                        disabled={isInvestigating}
                      />
                    </div>
                    <button
                      id="investigate-btn"
                      type="submit"
                      disabled={isInvestigating || !investigationQuery.trim()}
                      className="flex items-center justify-center gap-2 px-5 py-2.5 bg-emerald-600 hover:bg-emerald-500 disabled:bg-zinc-800 disabled:text-zinc-600 text-white rounded-lg text-sm font-semibold transition-all shadow-md shrink-0"
                    >
                      {isInvestigating ? (
                        <>
                          <svg className="animate-spin h-4 w-4" fill="none" viewBox="0 0 24 24">
                            <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                            <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
                          </svg>
                          <span>{investigationMode === "ai" ? "Analyzing..." : "Investigating..."}</span>
                        </>
                      ) : (
                        <>
                          <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
                          </svg>
                          <span>{investigationMode === "ai" ? "Ask AI" : "Investigate"}</span>
                        </>
                      )}
                    </button>
                  </form>

                  {/* Investigating Loading Indicator */}
                  {isInvestigating && (
                    <div className="rounded-lg bg-emerald-500/10 border border-emerald-500/30 p-3.5 flex items-center gap-3 animate-pulse">
                      <svg className="animate-spin h-4 w-4 text-emerald-400 shrink-0" fill="none" viewBox="0 0 24 24">
                        <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                        <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
                      </svg>
                      <span className="text-xs text-emerald-300 font-mono">
                        {investigationMode === "ai"
                          ? "Sentinel AI is analyzing detected events, temporal clusters, and correlating evidence..."
                          : "Querying indexed detection database..."}
                      </span>
                    </div>
                  )}

                  {/* Example Questions */}
                  <div className="space-y-1.5">
                    <span className="text-[11px] font-mono text-zinc-500 uppercase tracking-wider">
                      {investigationMode === "ai" ? "Suggested AI Questions:" : "Example Queries:"}
                    </span>
                    <div className="flex flex-wrap gap-2">
                      {(investigationMode === "ai" ? AI_EXAMPLE_QUESTIONS : DETERMINISTIC_EXAMPLE_QUESTIONS).map((q) => (
                        <button
                          key={q}
                          type="button"
                          onClick={() => handleInvestigate(q)}
                          disabled={isInvestigating}
                          className="text-xs font-mono bg-zinc-900 hover:bg-zinc-800 text-zinc-300 hover:text-emerald-400 border border-zinc-800 hover:border-emerald-500/50 px-2.5 py-1 rounded-md transition-all text-left"
                        >
                          • {q}
                        </button>
                      ))}
                    </div>
                  </div>

                  {/* Backend Error State */}
                  {investigationError && (
                    <div className="rounded-lg bg-red-500/10 border border-red-500/30 p-3 text-xs text-red-300 flex items-center justify-between">
                      <span>{investigationError}</span>
                      <button onClick={() => setInvestigationError(null)} className="text-red-400 hover:text-red-200 ml-2">Dismiss</button>
                    </div>
                  )}

                  {/* Phase 7: AI Results Section */}
                  {aiResponse && (
                    <div className="mt-4 pt-4 border-t border-zinc-800/80 space-y-4 animate-in fade-in duration-200">
                      {/* Response Header / Status */}
                      <div className="flex flex-wrap items-center justify-between gap-2">
                        <div className="flex items-center gap-2">
                          <span
                            className={`text-xs font-mono font-bold px-2.5 py-0.5 rounded border uppercase ${
                              aiResponse.mode === "ai_assisted"
                                ? "bg-emerald-500/20 text-emerald-400 border-emerald-500/40"
                                : aiResponse.mode === "guardrail_enforced"
                                ? "bg-purple-500/20 text-purple-400 border-purple-500/40"
                                : "bg-amber-500/20 text-amber-400 border-amber-500/40"
                            }`}
                          >
                            {aiResponse.mode === "ai_assisted"
                              ? "AI-Assisted Investigation"
                              : aiResponse.mode === "guardrail_enforced"
                              ? "Safety Guardrail Enforced"
                              : "AI Reasoning Unavailable — Deterministic Fallback"}
                          </span>
                          {aiResponse.count !== undefined && (
                            <span className="text-[11px] font-mono px-2 py-0.5 rounded bg-zinc-800 text-zinc-300">
                              Matches: {aiResponse.count}
                            </span>
                          )}
                        </div>
                        <span className="text-[11px] font-mono text-zinc-500 truncate max-w-xs">
                          Query: &ldquo;{aiResponse.query}&rdquo;
                        </span>
                      </div>

                      {/* Grounded AI Answer Box */}
                      <div className="bg-zinc-900 border border-zinc-800 rounded-xl p-5 space-y-3 shadow-inner">
                        <div className="flex items-center gap-2 text-xs font-mono text-emerald-400 uppercase tracking-wider font-semibold">
                          <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                          </svg>
                          <span>Sentinel Ground Truth Analysis</span>
                        </div>
                        <div className="text-sm text-zinc-200 leading-relaxed whitespace-pre-line font-sans">
                          {aiResponse.answer}
                        </div>
                      </div>

                      {/* Video Summary Card if present */}
                      {aiResponse.summary && (
                        <div className="bg-zinc-900/80 border border-zinc-800 rounded-xl p-4 space-y-3">
                          <div className="text-xs font-mono font-bold text-zinc-300 uppercase tracking-wider">
                            VIDEO METRICS BREAKDOWN
                          </div>
                          <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
                            <div className="bg-zinc-950 p-3 rounded-lg border border-zinc-800/80">
                              <span className="text-[10px] font-mono text-zinc-500 uppercase block">Duration</span>
                              <span className="text-sm font-mono font-bold text-emerald-400">
                                {aiResponse.summary.duration_seconds ? `${aiResponse.summary.duration_seconds.toFixed(2)}s` : "N/A"}
                              </span>
                            </div>
                            <div className="bg-zinc-950 p-3 rounded-lg border border-zinc-800/80">
                              <span className="text-[10px] font-mono text-zinc-500 uppercase block">Total Detections</span>
                              <span className="text-sm font-mono font-bold text-zinc-100">
                                {aiResponse.summary.total_detections ?? 0}
                              </span>
                            </div>
                            <div className="bg-zinc-950 p-3 rounded-lg border border-zinc-800/80">
                              <span className="text-[10px] font-mono text-zinc-500 uppercase block">Timeline Events</span>
                              <span className="text-sm font-mono font-bold text-blue-400">
                                {aiResponse.summary.total_events ?? 0}
                              </span>
                            </div>
                            <div className="bg-zinc-950 p-3 rounded-lg border border-zinc-800/80">
                              <span className="text-[10px] font-mono text-zinc-500 uppercase block">Preserved Evidence</span>
                              <span className="text-sm font-mono font-bold text-amber-400">
                                {aiResponse.summary.total_evidence ?? 0}
                              </span>
                            </div>
                          </div>
                          {aiResponse.summary.detected_classes && (
                            <div className="pt-2">
                              <span className="text-[10px] font-mono text-zinc-500 uppercase block mb-1.5">Detected Object Classes:</span>
                              <div className="flex flex-wrap gap-1.5">
                                {Object.entries(aiResponse.summary.detected_classes).map(([cls, cnt]) => (
                                  <span key={cls} className="text-xs font-mono px-2 py-0.5 rounded bg-zinc-800 border border-zinc-700 text-zinc-300">
                                    {formatClassName(cls)}: <strong className="text-emerald-400">{cnt}</strong>
                                  </span>
                                ))}
                              </div>
                            </div>
                          )}
                        </div>
                      )}

                      {/* Correlated Evidence Section */}
                      {aiResponse.sources.evidence && aiResponse.sources.evidence.length > 0 ? (
                        <div className="bg-zinc-900/60 border border-amber-500/30 rounded-xl p-4 space-y-3">
                          <div className="flex items-center justify-between">
                            <span className="text-xs font-mono font-bold text-amber-400 uppercase tracking-wider flex items-center gap-1.5">
                              <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 15v2m-6 4h12a2 2 0 002-2v-6a2 2 0 00-2-2H6a2 2 0 00-2 2v6a2 2 0 002 2zm10-10V7a4 4 0 00-8 0v4h8z" />
                              </svg>
                              Correlated Evidence Records ({aiResponse.sources.evidence.length})
                            </span>
                          </div>
                          <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
                            {aiResponse.sources.evidence.map((ev) => (
                              <div key={ev.evidence_id} className="bg-zinc-950 border border-zinc-800 rounded-lg p-3 space-y-2 flex flex-col justify-between">
                                <div className="flex items-center justify-between">
                                  <span className="text-xs font-mono font-bold text-zinc-200">
                                    #EV-{ev.evidence_id.slice(-6)}
                                  </span>
                                  <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-amber-500/20 text-amber-300 border border-amber-500/40">
                                    {ev.evidence_type.replace(/_/g, " ")}
                                  </span>
                                </div>
                                <div className="text-xs font-mono text-zinc-400">
                                  Target Timestamp: <strong className="text-emerald-400">{formatTimestamp(ev.timestamp)}</strong>
                                  {ev.duration_seconds && ` • Window: ${ev.duration_seconds}s`}
                                </div>
                                <div className="flex flex-wrap items-center gap-2 pt-1 font-mono">
                                  {ev.has_snapshot && (
                                    <a
                                      href={getEvidenceSnapshotUrl(ev.evidence_id)}
                                      target="_blank"
                                      rel="noopener noreferrer"
                                      className="text-[11px] bg-zinc-800 hover:bg-zinc-700 text-zinc-300 px-2 py-1 rounded transition-colors"
                                    >
                                      [Snapshot]
                                    </a>
                                  )}
                                  {ev.has_annotated && (
                                    <a
                                      href={getEvidenceAnnotatedUrl(ev.evidence_id)}
                                      target="_blank"
                                      rel="noopener noreferrer"
                                      className="text-[11px] bg-zinc-800 hover:bg-zinc-700 text-amber-300 px-2 py-1 rounded transition-colors"
                                    >
                                      [Annotated]
                                    </a>
                                  )}
                                  {ev.has_clip && (
                                    <a
                                      href={getEvidencePlaybackUrl(ev.evidence_id)}
                                      target="_blank"
                                      rel="noreferrer"
                                      className="text-[11px] bg-zinc-800 hover:bg-zinc-700 text-emerald-400 px-2 py-1 rounded transition-colors"
                                    >
                                      [Play Clip]
                                    </a>
                                  )}
                                  <button
                                    type="button"
                                    onClick={() => handleSeekToTimestamp(ev.timestamp, ev.event_id)}
                                    className="text-[11px] bg-emerald-600 hover:bg-emerald-500 text-white px-2.5 py-1 rounded ml-auto transition-colors"
                                  >
                                    Jump &rarr;
                                  </button>
                                </div>
                              </div>
                            ))}
                          </div>
                        </div>
                      ) : (
                        aiResponse.sources.detections && aiResponse.sources.detections.length > 0 && (
                          <div className="bg-zinc-900/40 border border-zinc-800 rounded-lg p-3 flex flex-col sm:flex-row items-center justify-between gap-3 text-xs">
                            <span className="text-zinc-400 font-mono">
                              No preserved evidence currently exists for this event window.
                            </span>
                            <button
                              type="button"
                              onClick={() => {
                                const first = aiResponse.sources.detections[0];
                                if (first && first.timestamp !== undefined) {
                                  handleCaptureEvidence(first.timestamp, first.event_id, "snapshot_and_clip");
                                }
                              }}
                              disabled={capturingEventId !== null}
                              className="px-3 py-1.5 bg-amber-600 hover:bg-amber-500 disabled:bg-zinc-800 text-white rounded font-mono text-xs font-semibold flex items-center gap-1.5 transition-colors shrink-0"
                            >
                              <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3 9a2 2 0 012-2h.93a2 2 0 001.664-.89l.812-1.22A2 2 0 0110.07 4h3.86a2 2 0 011.664.89l.812 1.22A2 2 0 0018.07 7H19a2 2 0 012 2v9a2 2 0 01-2 2H5a2 2 0 01-2-2V9z" />
                              </svg>
                              <span>Capture Evidence</span>
                            </button>
                          </div>
                        )
                      )}

                      {/* Relevant Detections Citations */}
                      {aiResponse.sources.detections && aiResponse.sources.detections.length > 0 && (
                        <div className="space-y-2">
                          <span className="text-xs font-mono text-zinc-400 uppercase tracking-wider block">
                            Referenced Detections ({aiResponse.sources.detections.length}):
                          </span>
                          <div className="space-y-1.5 max-h-56 overflow-y-auto pr-1">
                            {aiResponse.sources.detections.map((det) => (
                              <div
                                key={det.event_id}
                                onClick={() => handleSeekToTimestamp(det.timestamp, det.event_id)}
                                className="rounded-lg border border-zinc-800 bg-zinc-900/70 hover:bg-zinc-800/80 p-2.5 flex items-center justify-between cursor-pointer transition-colors"
                              >
                                <div className="flex items-center gap-2.5">
                                  <span className="text-xs font-mono font-bold text-emerald-400 px-2 py-0.5 bg-black/40 rounded border border-emerald-500/30">
                                    {formatTimestamp(det.timestamp)}
                                  </span>
                                  <span className="text-xs font-bold text-zinc-200">
                                    {det.object_class
                                      ? formatClassName(det.object_class)
                                      : det.event_type
                                      ? det.event_type.replace(/_/g, " ").toUpperCase()
                                      : formatClassName(null)}
                                  </span>
                                  {det.confidence != null && (
                                    <span className="text-[11px] font-mono text-zinc-400">
                                      ({Math.round(det.confidence * 100)}% conf)
                                    </span>
                                  )}
                                </div>
                                <button
                                  type="button"
                                  onClick={(e) => {
                                    e.stopPropagation();
                                    handleSeekToTimestamp(det.timestamp, det.event_id);
                                  }}
                                  className="text-xs font-mono text-zinc-400 hover:text-emerald-400"
                                >
                                  [Jump]
                                </button>
                              </div>
                            ))}
                          </div>
                        </div>
                      )}

                      {/* Limitations Notice */}
                      {aiResponse.limitations && (
                        <div className="text-[11px] font-mono text-zinc-500 border-t border-zinc-800/60 pt-2.5">
                          {aiResponse.limitations.join(" ")}
                        </div>
                      )}
                    </div>
                  )}

                  {/* Results Section */}
                  {investigationResponse && (
                    <div className="mt-4 pt-4 border-t border-zinc-800/80 space-y-3 animate-in fade-in duration-200">
                      {/* Response Header / Status */}
                      <div className="flex flex-wrap items-center justify-between gap-2">
                        <div className="flex items-center gap-2">
                          <span className="text-xs font-mono font-bold text-zinc-200 uppercase">
                            Results: {investigationResponse.count}{" "}
                            {investigationResponse.result_type === "events"
                              ? "matching events"
                              : investigationResponse.result_type === "count"
                              ? "detections counted"
                              : "matching detections"}
                          </span>
                          {investigationResponse.interpreted_filters?.object_class && (
                            <span className="text-[11px] font-mono px-2 py-0.5 rounded bg-emerald-500/10 border border-emerald-500/30 text-emerald-400">
                              Class: {formatClassName(investigationResponse.interpreted_filters.object_class)}
                            </span>
                          )}
                          {(investigationResponse.interpreted_filters?.start_time != null ||
                            investigationResponse.interpreted_filters?.end_time != null) && (
                            <span className="text-[11px] font-mono px-2 py-0.5 rounded bg-blue-500/10 border border-blue-500/30 text-blue-400">
                              Time: {investigationResponse.interpreted_filters?.start_time ?? 0}s –{" "}
                              {investigationResponse.interpreted_filters?.end_time ?? "end"}s
                            </span>
                          )}
                          {investigationResponse.interpreted_filters?.min_confidence != null && (
                            <span className="text-[11px] font-mono px-2 py-0.5 rounded bg-purple-500/10 border border-purple-500/30 text-purple-400">
                              Conf &ge; {Math.round(investigationResponse.interpreted_filters.min_confidence * 100)}%
                            </span>
                          )}
                        </div>
                        <span className="text-[11px] font-mono text-zinc-500 truncate max-w-xs">
                          Query: &ldquo;{investigationResponse.query}&rdquo;
                        </span>
                      </div>

                      {/* Unsupported Message Banner */}
                      {!investigationResponse.is_supported && (
                        <div className="rounded-lg bg-amber-500/10 border border-amber-500/30 p-4 text-xs text-amber-300 space-y-1">
                          <div className="flex items-center gap-2 font-semibold">
                            <svg className="w-4 h-4 text-amber-400 shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
                            </svg>
                            <span>Unsupported Investigation Scope</span>
                          </div>
                          <p className="text-zinc-400 leading-relaxed pl-6">{investigationResponse.message}</p>
                        </div>
                      )}

                      {/* Count Query Display */}
                      {investigationResponse.is_supported && investigationResponse.result_type === "count" && (
                        <div className="bg-zinc-900 border border-zinc-800 rounded-lg p-5 flex flex-col sm:flex-row items-center justify-between gap-4">
                          <div>
                            <div className="text-3xl font-extrabold text-emerald-400 font-mono">
                              {investigationResponse.count}
                            </div>
                            <p className="text-xs text-zinc-300 mt-1">{investigationResponse.message}</p>
                          </div>
                        </div>
                      )}

                      {/* Zero Results */}
                      {investigationResponse.is_supported &&
                        investigationResponse.result_type !== "count" &&
                        investigationResponse.results.length === 0 && (
                          <div className="rounded-lg border border-zinc-800 bg-zinc-900/40 p-6 text-center">
                            <p className="text-zinc-400 text-xs font-mono">{investigationResponse.message}</p>
                            <p className="text-zinc-500 text-[11px] mt-1">Try widening the time window or adjusting confidence threshold.</p>
                          </div>
                        )}

                      {/* Detection Cards List */}
                      {investigationResponse.is_supported &&
                        investigationResponse.result_type === "detections" &&
                        investigationResponse.results.length > 0 && (
                          <div className="space-y-2 max-h-80 overflow-y-auto pr-1">
                            {investigationResponse.results.map((item, idx) => (
                              <div
                                key={item.event_id || idx}
                                onClick={() => item.timestamp !== undefined && handleSeekToTimestamp(item.timestamp, item.event_id)}
                                className={`rounded-lg border p-3 flex flex-col sm:flex-row sm:items-center justify-between gap-3 cursor-pointer hover:brightness-110 transition-all ${
                                  item.object_class ? getClassColor(item.object_class) : "border-zinc-800 bg-zinc-900"
                                }`}
                              >
                                <div className="flex items-center gap-3">
                                  {item.timestamp !== undefined && (
                                    <button
                                      onClick={(e) => {
                                        e.stopPropagation();
                                        handleSeekToTimestamp(item.timestamp!, item.event_id);
                                      }}
                                      className="font-mono text-xs bg-black/40 hover:bg-black/60 border border-current/40 text-emerald-400 font-bold px-2.5 py-1.5 rounded transition-all shrink-0"
                                    >
                                      {formatTimestamp(item.timestamp)}
                                    </button>
                                  )}
                                  <div>
                                    <div className="text-sm font-bold tracking-wide">
                                      {item.object_class
                                        ? formatClassName(item.object_class).toUpperCase()
                                        : item.event_type
                                        ? item.event_type.replace(/_/g, " ").toUpperCase()
                                        : "GENERAL SECURITY ACTIVITY"}
                                    </div>
                                    <div className="text-[11px] opacity-75 font-mono">
                                      Confidence: {item.confidence !== undefined ? `${Math.round(item.confidence * 100)}%` : "N/A"}
                                      {item.frame_number !== undefined && ` • Frame ${item.frame_number}`}
                                      {item.bounding_box && (
                                        <span>
                                          {" "}• BBox [{Math.round(item.bounding_box.x1)},{Math.round(item.bounding_box.y1)} &rarr;{" "}
                                          {Math.round(item.bounding_box.x2)},{Math.round(item.bounding_box.y2)}]
                                        </span>
                                      )}
                                    </div>
                                  </div>
                                </div>

                                <div className="flex items-center gap-2 self-end sm:self-center font-mono">
                                  <button
                                    type="button"
                                    onClick={(e) => {
                                      e.stopPropagation();
                                      if (item.timestamp !== undefined) handleSeekToTimestamp(item.timestamp, item.event_id);
                                    }}
                                    className="text-xs bg-zinc-950/80 hover:bg-emerald-600 hover:text-white text-zinc-300 border border-zinc-700 px-3 py-1.5 rounded transition-colors"
                                  >
                                    [Jump to timestamp]
                                  </button>
                                  <button
                                    type="button"
                                    onClick={(e) => {
                                      e.stopPropagation();
                                      if (item.timestamp !== undefined) handleCaptureEvidence(item.timestamp, item.event_id, "snapshot_and_clip");
                                    }}
                                    disabled={capturingEventId === (item.event_id || `${item.timestamp}`)}
                                    className="text-xs bg-amber-500/10 hover:bg-amber-500/20 text-amber-300 hover:text-amber-200 border border-amber-500/30 px-3 py-1.5 rounded flex items-center gap-1.5 transition-colors disabled:opacity-50"
                                  >
                                    <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3 9a2 2 0 012-2h.93a2 2 0 001.664-.89l.812-1.22A2 2 0 0110.07 4h3.86a2 2 0 011.664.89l.812 1.22A2 2 0 0018.07 7H19a2 2 0 012 2v9a2 2 0 01-2 2H5a2 2 0 01-2-2V9z" />
                                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 13a3 3 0 11-6 0 3 3 0 016 0z" />
                                    </svg>
                                    <span>{capturingEventId === (item.event_id || `${item.timestamp}`) ? "Capturing..." : "Capture Evidence"}</span>
                                  </button>
                                </div>
                              </div>
                            ))}
                          </div>
                        )}

                      {/* Events Results List */}
                      {investigationResponse.is_supported &&
                        investigationResponse.result_type === "events" &&
                        investigationResponse.results.length > 0 && (
                          <div className="space-y-2 max-h-80 overflow-y-auto pr-1">
                            {investigationResponse.results.map((item, idx) => (
                              <div
                                key={item.event_id || idx}
                                onClick={() => item.start_time !== undefined && handleSeekToTimestamp(item.start_time, item.event_id)}
                                className="rounded-lg border border-zinc-800 bg-zinc-900/90 p-3 flex flex-col sm:flex-row sm:items-center justify-between gap-3 cursor-pointer hover:border-emerald-500/50 hover:bg-zinc-900 transition-all"
                              >
                                <div className="flex items-center gap-3">
                                  {item.start_time !== undefined && (
                                    <button
                                      onClick={(e) => {
                                        e.stopPropagation();
                                        handleSeekToTimestamp(item.start_time!, item.event_id);
                                      }}
                                      className="font-mono text-xs bg-emerald-500/20 hover:bg-emerald-500/30 text-emerald-400 border border-emerald-500/40 font-bold px-2.5 py-1.5 rounded transition-all shrink-0"
                                    >
                                      {formatTimestamp(item.start_time)}
                                      {item.end_time !== undefined && item.end_time > item.start_time && (
                                        <span className="block text-[10px] text-emerald-500 font-normal">
                                          - {formatTimestamp(item.end_time)}
                                        </span>
                                      )}
                                    </button>
                                  )}
                                  <div>
                                    <div className="text-sm font-bold text-zinc-100 flex items-center gap-2">
                                      {item.event_type?.replace(/_/g, " ") || "EVENT"}
                                      {item.priority && (
                                        <span className={`text-[10px] font-mono font-semibold px-2 py-0.5 rounded border ${PRIORITY_BADGES[item.priority] || PRIORITY_BADGES["NORMAL"]}`}>
                                          {item.priority}
                                        </span>
                                      )}
                                    </div>
                                    <div className="text-xs text-zinc-400 mt-1 font-mono">
                                      {item.total_detections} detections • Max Conf: {item.max_confidence ? `${Math.round(item.max_confidence * 100)}%` : "N/A"}
                                    </div>
                                  </div>
                                </div>

                                <div className="flex items-center gap-2 self-end sm:self-center font-mono">
                                  <button
                                    type="button"
                                    onClick={(e) => {
                                      e.stopPropagation();
                                      if (item.start_time !== undefined) handleSeekToTimestamp(item.start_time, item.event_id);
                                    }}
                                    className="text-xs bg-zinc-950/80 hover:bg-emerald-600 hover:text-white text-zinc-300 border border-zinc-700 px-3 py-1.5 rounded transition-colors"
                                  >
                                    [Jump to timestamp]
                                  </button>
                                  <button
                                    type="button"
                                    onClick={(e) => {
                                      e.stopPropagation();
                                      if (item.start_time !== undefined) handleCaptureEvidence(item.start_time, item.event_id, "snapshot_and_clip");
                                    }}
                                    disabled={capturingEventId === item.event_id}
                                    className="text-xs bg-amber-500/10 hover:bg-amber-500/20 text-amber-300 hover:text-amber-200 border border-amber-500/30 px-3 py-1.5 rounded flex items-center gap-1.5 transition-colors disabled:opacity-50"
                                  >
                                    <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3 9a2 2 0 012-2h.93a2 2 0 001.664-.89l.812-1.22A2 2 0 0110.07 4h3.86a2 2 0 011.664.89l.812 1.22A2 2 0 0018.07 7H19a2 2 0 012 2v9a2 2 0 01-2 2H5a2 2 0 01-2-2V9z" />
                                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 13a3 3 0 11-6 0 3 3 0 016 0z" />
                                    </svg>
                                    <span>{capturingEventId === item.event_id ? "Capturing..." : "Capture Evidence"}</span>
                                  </button>
                                </div>
                              </div>
                            ))}
                          </div>
                        )}
                    </div>
                  )}
                </div>

                {/* Evidence Action Feedback Banner */}
                {evidenceFeedback && (
                  <div className={`rounded-lg p-3.5 text-xs flex items-center justify-between border animate-in fade-in duration-200 ${
                    evidenceFeedback.type === "success"
                      ? "bg-amber-500/10 border-amber-500/30 text-amber-300"
                      : "bg-red-500/10 border-red-500/30 text-red-300"
                  }`}>
                    <div className="flex items-center gap-2">
                      <svg className="w-4 h-4 shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
                      </svg>
                      <span>{evidenceFeedback.message}</span>
                    </div>
                    <button onClick={() => setEvidenceFeedback(null)} className="underline hover:opacity-80 ml-3">Dismiss</button>
                  </div>
                )}

                {/* View Switcher & Filters */}
                <div className="flex flex-wrap gap-4 items-center justify-between bg-zinc-950 p-3 rounded-lg border border-zinc-800">
                  <div className="flex items-center gap-1 bg-zinc-900 p-1 rounded-md border border-zinc-800 text-xs font-mono">
                    <button
                      onClick={() => setActiveView("timeline")}
                      className={`px-3 py-1.5 rounded font-semibold transition-colors ${
                        activeView === "timeline" ? "bg-emerald-600 text-white" : "text-zinc-400 hover:text-zinc-200"
                      }`}
                    >
                      Investigator Timeline ({filteredTimelineEvents.length})
                    </button>
                    <button
                      onClick={() => setActiveView("raw")}
                      className={`px-3 py-1.5 rounded font-semibold transition-colors ${
                        activeView === "raw" ? "bg-blue-600 text-white" : "text-zinc-400 hover:text-zinc-200"
                      }`}
                    >
                      Raw Detections ({filteredRawEvents.length})
                    </button>
                    <button
                      id="security-intelligence-tab"
                      onClick={() => {
                        setActiveView("intelligence");
                        if (uploadResult?.video_id) fetchSecurityIntelligence(uploadResult.video_id);
                      }}
                      className={`px-3 py-1.5 rounded font-semibold transition-colors flex items-center gap-1.5 ${
                        activeView === "intelligence" ? "bg-indigo-600 text-white" : "text-zinc-400 hover:text-zinc-200"
                      }`}
                    >
                      <span className="h-2 w-2 rounded-full bg-indigo-400 animate-pulse" />
                      Security Intelligence ({securityEvents.length} Events • {tracks.length} Tracks)
                    </button>
                    <button
                      id="correlated-incidents-tab"
                      onClick={() => setActiveView("correlation")}
                      className={`px-3 py-1.5 rounded font-semibold transition-colors flex items-center gap-1.5 ${
                        activeView === "correlation" ? "bg-cyan-600 text-white" : "text-zinc-400 hover:text-zinc-200"
                      }`}
                    >
                      <span className="h-2 w-2 rounded-full bg-cyan-400 animate-pulse" />
                      Correlated Storylines
                    </button>
                    <button
                      id="forensic-search-tab"
                      onClick={() => setActiveView("forensic")}
                      className={`px-3 py-1.5 rounded font-semibold transition-colors flex items-center gap-1.5 ${
                        activeView === "forensic" ? "bg-cyan-600 text-white" : "text-zinc-400 hover:text-zinc-200"
                      }`}
                    >
                      <span className="h-2 w-2 rounded-full bg-cyan-400 animate-pulse" />
                      🔍 Forensic Search
                    </button>
                    <button
                      id="evidence-vault-tab"
                      onClick={() => setActiveView("vault")}
                      className={`px-3 py-1.5 rounded font-semibold transition-colors flex items-center gap-1.5 ${
                        activeView === "vault" ? "bg-amber-600 text-white" : "text-zinc-400 hover:text-zinc-200"
                      }`}
                    >
                      <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 15v2m-6 4h12a2 2 0 002-2v-6a2 2 0 00-2-2H6a2 2 0 00-2 2v6a2 2 0 002 2zm10-10V7a4 4 0 00-8 0v4h8z" />
                      </svg>
                      Evidence Vault ({evidenceList.length})
                    </button>
                    <button
                      id="incident-dossier-tab"
                      onClick={() => {
                        setActiveView("reports");
                        if (uploadResult?.video_id) fetchReports(uploadResult.video_id);
                      }}
                      className={`px-3 py-1.5 rounded font-semibold transition-colors flex items-center gap-1.5 ${
                        activeView === "reports" ? "bg-emerald-700 text-white" : "text-zinc-400 hover:text-zinc-200"
                      }`}
                    >
                      <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                      </svg>
                      Incident Dossier ({reportsList.length})
                    </button>
                    <button
                      id="specialized-visual-tab"
                      onClick={() => {
                        setActiveView("specialized");
                        if (uploadResult?.video_id) fetchSpecialized(uploadResult.video_id);
                      }}
                      className={`px-3 py-1.5 rounded font-semibold transition-colors flex items-center gap-1.5 ${
                        activeView === "specialized" ? "bg-rose-600 text-white" : "text-zinc-400 hover:text-zinc-200"
                      }`}
                    >
                      <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M17.657 18.657A8 8 0 016.343 7.343S7 9 9 10c0-2 .5-5 2.986-7C14 5 16.09 5.777 17.656 7.343A7.975 7.975 0 0120 13a7.975 7.975 0 01-2.343 5.657z" />
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.879 16.121A3 3 0 1012.015 11L11 14H9c0 .768.293 1.536.879 2.121z" />
                      </svg>
                      Specialized Visual ({specializedList.length})
                    </button>
                    <button
                      id="detector-diagnostics-tab"
                      onClick={() => {
                        setActiveView("diagnostics");
                        if (uploadResult?.video_id) fetchHealth(uploadResult.video_id);
                      }}
                      className={`px-3 py-1.5 rounded font-semibold transition-colors flex items-center gap-1.5 ${
                        activeView === "diagnostics" ? "bg-teal-700 text-white" : "text-zinc-400 hover:text-zinc-200"
                      }`}
                    >
                      <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m5.618-4.016A11.955 11.955 0 0112 2.944a11.955 11.955 0 01-8.618 3.04A12.02 12.02 0 003 9c0 5.591 3.824 10.29 9 11.622 5.176-1.332 9-6.03 9-11.622 0-1.042-.133-2.052-.382-3.016z" />
                      </svg>
                      Diagnostics & Health
                    </button>
                    <button
                      id="multi-camera-tab"
                      onClick={() => setActiveView("multicamera")}
                      className={`px-3 py-1.5 rounded font-semibold transition-colors flex items-center gap-1.5 ${
                        activeView === "multicamera" ? "bg-violet-700 text-white" : "text-zinc-400 hover:text-zinc-200"
                      }`}
                    >
                      <span className="h-2 w-2 rounded-full bg-violet-400 animate-pulse" />
                      🎥 Multi-Camera Sessions
                    </button>
                  </div>

                  {/* Filter Controls (Shown for Timeline & Raw) */}
                  {activeView !== "vault" && activeView !== "intelligence" && activeView !== "reports" && activeView !== "specialized" && activeView !== "diagnostics" && activeView !== "correlation" && activeView !== "forensic" && activeView !== "multicamera" && (
                    <div className="flex flex-wrap gap-3 items-center">
                      <span className="text-xs font-mono text-zinc-500">Filter Class:</span>
                      <select
                        value={classFilter}
                        onChange={(e) => setClassFilter(e.target.value)}
                        className="bg-zinc-900 border border-zinc-700 text-zinc-200 text-xs rounded-lg px-3 py-1.5 focus:outline-none focus:border-emerald-500"
                      >
                        <option value="all">All Classes</option>
                        {availableClasses.map((cls) => (
                          <option key={cls} value={cls}>{formatClassName(cls)}</option>
                        ))}
                      </select>

                      <label className="flex items-center gap-2 text-xs text-zinc-400 font-mono">
                        Min Confidence:
                        <input
                          type="range"
                          min={0}
                          max={1}
                          step={0.05}
                          value={minConfidence}
                          onChange={(e) => setMinConfidence(parseFloat(e.target.value))}
                          className="w-24 accent-emerald-500"
                        />
                        <span className="text-emerald-400 w-8">{Math.round(minConfidence * 100)}%</span>
                      </label>
                    </div>
                  )}
                </div>

                {/* TIMELINE VIEW */}
                {activeView === "timeline" && (
                  <div>
                    {filteredTimelineEvents.length === 0 ? (
                      <div className="rounded-lg border border-zinc-800 bg-zinc-950/50 p-8 text-center">
                        <p className="text-zinc-400 text-sm">No timeline events match the current filter criteria.</p>
                      </div>
                    ) : (
                      <div className="space-y-3 max-h-[560px] overflow-y-auto pr-1">
                        {filteredTimelineEvents.map((event) => {
                          const isSelected = activeEventId === event.event_id;
                          const priorityStyle = PRIORITY_BADGES[event.priority] || PRIORITY_BADGES["NORMAL"];

                          return (
                            <div
                              key={event.event_id}
                              onClick={() => handleSeekToTimestamp(event.start_time, event.event_id)}
                              className={`rounded-lg border p-4 cursor-pointer transition-all ${
                                isSelected
                                  ? "border-emerald-500 bg-emerald-500/10 ring-2 ring-emerald-500/30"
                                  : "border-zinc-800 bg-zinc-950/80 hover:border-zinc-700 hover:bg-zinc-900/50"
                              }`}
                            >
                              <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3">
                                {/* Time & Event Title */}
                                <div className="flex items-start gap-3">
                                  <button
                                    onClick={(e) => {
                                      e.stopPropagation();
                                      handleSeekToTimestamp(event.start_time, event.event_id);
                                    }}
                                    className="shrink-0 font-mono text-xs bg-emerald-500/20 hover:bg-emerald-500/30 text-emerald-400 border border-emerald-500/40 rounded px-2.5 py-1.5 text-center font-bold"
                                  >
                                    {formatTimestamp(event.start_time)}
                                    {event.duration_seconds > 0.1 && (
                                      <span className="block text-[10px] text-emerald-500 font-normal">
                                        +{event.duration_seconds}s
                                      </span>
                                    )}
                                  </button>

                                  <div>
                                    <div className="flex items-center gap-2">
                                      <span className="text-sm font-bold text-zinc-100">
                                        {event.event_type ? event.event_type.replace(/_/g, " ") : "Event"}
                                      </span>
                                      <span className={`text-[10px] font-mono font-semibold px-2 py-0.5 rounded border ${priorityStyle}`}>
                                        PRIORITY: {event.priority}
                                      </span>
                                    </div>

                                    {/* Objects Summary with Strict Counting Semantics */}
                                    <div className="flex flex-wrap gap-2 mt-1.5">
                                      {event.objects.map((obj: any) => {
                                        const trackCount = obj.track_count !== undefined && obj.track_count !== null
                                          ? obj.track_count
                                          : (Array.isArray(obj.track_ids) ? obj.track_ids.length : undefined);
                                        const detCount = obj.detection_count ?? obj.count ?? 1;
                                        const label = trackCount && trackCount > 0
                                          ? `${formatClassName(obj.class)} · ${trackCount} ${trackCount === 1 ? "track" : "tracks"} · ${detCount} detections`
                                          : `${formatClassName(obj.class)} · ${detCount} detections`;
                                        const tooltip = trackCount && trackCount > 0
                                          ? `${detCount} detection observations across ${trackCount} anonymous tracks`
                                          : `${detCount} detection observations across analyzed frames`;

                                        return (
                                          <span
                                            key={obj.class}
                                            title={tooltip}
                                            className={`text-xs font-mono px-2 py-0.5 rounded border ${getClassColor(obj.class)}`}
                                          >
                                            {label}
                                          </span>
                                        );
                                      })}
                                    </div>
                                  </div>
                                </div>

                                  {/* Metrics, Evidence & Expand toggle */}
                                  <div className="flex items-center gap-3 text-right self-end sm:self-center font-mono">
                                    <div>
                                      <div className="text-xs text-zinc-400">Detections</div>
                                      <div className="text-sm font-bold text-zinc-200">{event.total_detections}</div>
                                    </div>
                                    <div className="border-l border-zinc-800 pl-3">
                                      <div className="text-xs text-zinc-400">Max Conf</div>
                                      <div className="text-sm font-bold text-emerald-400">
                                        {Math.round(event.max_confidence * 100)}%
                                      </div>
                                    </div>
                                    <button
                                      onClick={(e) => {
                                        e.stopPropagation();
                                        handleCaptureEvidence(event.start_time, event.event_id, "snapshot_and_clip");
                                      }}
                                      disabled={capturingEventId === event.event_id}
                                      className="border-l border-zinc-800 pl-3 text-xs font-sans text-amber-400 hover:text-amber-300 flex items-center gap-1 font-medium disabled:opacity-50"
                                      title="Preserve event snapshot & clip in Evidence Vault"
                                    >
                                      <span>{capturingEventId === event.event_id ? "Capturing..." : "Capture Event"}</span>
                                    </button>
                                    <button
                                      onClick={(e) => toggleEventExpanded(event.event_id, e)}
                                      className="border-l border-zinc-800 pl-3 text-xs font-sans text-emerald-400 hover:text-emerald-300 flex items-center gap-1 font-medium"
                                    >
                                      <span>{expandedEventIds[event.event_id] ? "Hide ▲" : "Detections ▼"}</span>
                                    </button>
                                  </div>
                                </div>

                                {/* Expandable Individual Detections List */}
                                {expandedEventIds[event.event_id] && (
                                  <div className="mt-4 pt-3 border-t border-zinc-800 space-y-2 animate-in fade-in duration-200">
                                    <div className="flex items-center justify-between text-xs font-mono text-zinc-400">
                                      <span>Individual Detections ({getEventDetections(event).length})</span>
                                      <span className="text-[10px] text-zinc-500">Click timestamp to seek video</span>
                                    </div>

                                    {getEventDetections(event).length === 0 ? (
                                      <p className="text-xs text-zinc-500 italic py-1 font-mono">No detections match current filter.</p>
                                    ) : (
                                      <div className="space-y-1.5 max-h-60 overflow-y-auto pr-1">
                                        {getEventDetections(event).map((det) => (
                                          <div
                                            key={det.event_id}
                                            onClick={(e) => {
                                              e.stopPropagation();
                                              handleSeekToTimestamp(det.timestamp, event.event_id);
                                            }}
                                            className={`rounded border p-2 flex items-center justify-between gap-3 text-xs font-mono transition-all hover:brightness-125 ${getClassColor(det.object_class)}`}
                                          >
                                            <div className="flex items-center gap-2">
                                              <span className="bg-black/40 px-2 py-0.5 rounded text-zinc-200 font-bold">
                                                {formatTimestamp(det.timestamp)}
                                              </span>
                                              <span className="font-semibold font-sans">
                                                {formatClassName(det.object_class)}
                                              </span>
                                              {det.frame_number !== undefined && (
                                                <span className="text-zinc-500 text-[11px]">
                                                  (Frame {det.frame_number})
                                                </span>
                                              )}
                                            </div>

                                            <div className="flex items-center gap-3">
                                              <span className="text-[11px] text-zinc-400">
                                                BBox [{Math.round(det.bounding_box.x1)},{Math.round(det.bounding_box.y1)} → {Math.round(det.bounding_box.x2)},{Math.round(det.bounding_box.y2)}]
                                              </span>
                                              <span className="font-bold text-zinc-100">
                                                {Math.round(det.confidence * 100)}%
                                              </span>
                                              <button
                                                onClick={(e) => {
                                                  e.stopPropagation();
                                                  handleCaptureEvidence(det.timestamp, det.event_id, "snapshot_and_clip");
                                                }}
                                                disabled={capturingEventId === det.event_id}
                                                className="px-2 py-0.5 rounded bg-black/40 hover:bg-amber-500/20 text-amber-300 hover:text-amber-200 border border-amber-500/30 text-[10px] flex items-center gap-1 disabled:opacity-50 transition-colors"
                                              >
                                                {capturingEventId === det.event_id ? "..." : "Capture"}
                                              </button>
                                            </div>
                                          </div>
                                        ))}
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
                )}

                {/* RAW DETECTIONS VIEW */}
                {activeView === "raw" && (
                  <div>
                    {filteredRawEvents.length === 0 ? (
                      <div className="rounded-lg border border-zinc-800 bg-zinc-950/50 p-8 text-center">
                        <p className="text-zinc-400 text-sm">No raw detections match the current filter criteria.</p>
                      </div>
                    ) : (
                      <div className="space-y-2 max-h-[520px] overflow-y-auto pr-1">
                        {filteredRawEvents.map((event) => (
                          <div
                            key={event.event_id}
                            onClick={() => handleSeekToTimestamp(event.timestamp)}
                            className={`rounded-lg border p-3 flex items-center justify-between gap-3 cursor-pointer hover:brightness-110 transition-all ${getClassColor(event.object_class)}`}
                          >
                            <div className="flex items-center gap-3 min-w-0">
                              <button
                                className="shrink-0 font-mono text-xs bg-black/30 hover:bg-black/50 border border-current/30 rounded px-2 py-1 transition-colors min-w-[60px] text-center"
                              >
                                {formatTimestamp(event.timestamp)}
                              </button>
                              <div className="min-w-0">
                                <div className="flex items-center gap-2">
                                  <p className="text-sm font-semibold truncate">
                                    {event.object_class
                                      ? `${formatClassName(event.object_class)} detected`
                                      : event.event_type
                                      ? event.event_type.replace(/_/g, " ").toUpperCase()
                                      : "General Security Activity detected"}
                                  </p>
                                  {event.validation_status === "REJECTED" ? (
                                    <span className="text-[10px] font-mono font-bold bg-rose-500/20 text-rose-400 border border-rose-500/40 px-1.5 py-0.5 rounded">
                                      REJECTED
                                    </span>
                                  ) : (
                                    <span className="text-[10px] font-mono font-bold bg-emerald-500/20 text-emerald-400 border border-emerald-500/40 px-1.5 py-0.5 rounded">
                                      VALIDATED
                                    </span>
                                  )}
                                </div>
                                <p className="text-xs opacity-70 font-mono truncate">
                                  Frame {event.frame_number} • BBox [{Math.round(event.bounding_box.x1)},{Math.round(event.bounding_box.y1)} → {Math.round(event.bounding_box.x2)},{Math.round(event.bounding_box.y2)}]
                                  {event.validation_reason && ` • ${event.validation_reason}`}
                                </p>
                              </div>
                            </div>
                            <div className="shrink-0 flex items-center gap-3">
                              <div className="text-right">
                                <div className="text-sm font-bold">{Math.round(event.confidence * 100)}%</div>
                                <div className="text-[10px] opacity-60 font-mono uppercase">confidence</div>
                              </div>
                              <button
                                onClick={(e) => {
                                  e.stopPropagation();
                                  handleCaptureEvidence(event.timestamp, event.event_id, "snapshot_and_clip");
                                }}
                                disabled={capturingEventId === event.event_id}
                                className="px-2.5 py-1.5 rounded bg-black/40 hover:bg-amber-500/20 text-amber-300 hover:text-amber-200 border border-amber-500/30 text-xs font-mono disabled:opacity-50 transition-colors"
                              >
                                {capturingEventId === event.event_id ? "Capturing..." : "Capture"}
                              </button>
                            </div>
                          </div>
                        ))}
                      </div>
                    )}
                  </div>
                )}

                {/* EVIDENCE VAULT VIEW */}
                {activeView === "vault" && (
                  <div className="space-y-4 animate-in fade-in duration-200">
                    <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 bg-zinc-950 p-4 rounded-lg border border-zinc-800">
                      <div className="flex items-center gap-2.5">
                        <div className="p-2 bg-amber-500/10 border border-amber-500/30 rounded-md text-amber-400">
                          <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 15v2m-6 4h12a2 2 0 002-2v-6a2 2 0 00-2-2H6a2 2 0 00-2 2v6a2 2 0 002 2zm10-10V7a4 4 0 00-8 0v4h8z" />
                          </svg>
                        </div>
                        <div>
                          <h3 className="text-sm font-bold text-zinc-100 flex items-center gap-2">
                            EVIDENCE VAULT
                            <span className="text-[10px] font-mono font-semibold uppercase bg-amber-500/20 text-amber-400 border border-amber-500/40 px-2 py-0.5 rounded">
                              Forensic Preservation
                            </span>
                          </h3>
                          <p className="text-xs text-zinc-400">
                            {evidenceList.length} preserved evidence records (snapshots and clips extracted from CCTV source)
                          </p>
                        </div>
                      </div>
                      {uploadResult?.video_id && (
                        <button
                          onClick={() => fetchEvidence(uploadResult.video_id)}
                          className="text-xs font-mono text-zinc-400 hover:text-zinc-200 border border-zinc-800 px-3 py-1.5 rounded transition-colors"
                        >
                          Refresh Vault
                        </button>
                      )}
                    </div>

                    {evidenceList.length === 0 ? (
                      <div className="rounded-lg border border-zinc-800 bg-zinc-950/50 p-12 text-center space-y-2">
                        <p className="text-zinc-300 font-medium text-sm">No evidence captured yet for this video.</p>
                        <p className="text-zinc-500 text-xs max-w-md mx-auto">
                          Click the <span className="text-amber-400 font-mono">[Capture Evidence]</span> button on any timeline event, individual detection, or Ask Sentinel search result to generate verified snapshot and video clip evidence.
                        </p>
                      </div>
                    ) : (
                      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                        {evidenceList.map((item) => (
                          <div
                            key={item.evidence_id}
                            className="rounded-xl border border-zinc-800 bg-zinc-950/80 hover:border-amber-500/50 transition-all p-4 space-y-3 flex flex-col justify-between group"
                          >
                            <div className="space-y-2.5">
                              {/* Card Header */}
                              <div className="flex items-center justify-between">
                                <span className="font-mono text-xs font-bold text-amber-400 bg-amber-500/10 border border-amber-500/30 px-2 py-0.5 rounded">
                                  #EV-{item.evidence_id.slice(-6).toUpperCase()}
                                </span>
                                <span className="text-[10px] font-mono text-zinc-400 uppercase bg-zinc-900 border border-zinc-800 px-2 py-0.5 rounded">
                                  {item.evidence_type.replace(/_/g, " ")}
                                </span>
                              </div>

                              {/* Media Thumbnail */}
                              <div
                                onClick={() => {
                                  setSelectedEvidence(item);
                                  setEvidenceModalMode("snapshot");
                                }}
                                className="relative aspect-video rounded-lg overflow-hidden bg-zinc-900 border border-zinc-800 cursor-pointer group-hover:border-zinc-700"
                              >
                                {item.has_snapshot ? (
                                  <img
                                    src={getEvidenceSnapshotUrl(item.evidence_id)}
                                    alt={`Evidence ${item.evidence_id}`}
                                    className="w-full h-full object-cover group-hover:scale-105 transition-transform duration-300"
                                  />
                                ) : (
                                  <div className="w-full h-full flex items-center justify-center text-zinc-600 text-xs font-mono">
                                    Clip Only
                                  </div>
                                )}
                                <div className="absolute bottom-2 left-2 bg-black/75 backdrop-blur-sm px-2 py-0.5 rounded text-[11px] font-mono text-emerald-400 font-bold">
                                  {formatTimestamp(item.timestamp)}
                                </div>
                                {item.has_clip && (
                                  <div className="absolute bottom-2 right-2 bg-black/75 backdrop-blur-sm px-1.5 py-0.5 rounded text-[10px] font-mono text-amber-300 flex items-center gap-1">
                                    <svg className="w-3 h-3" fill="currentColor" viewBox="0 0 24 24"><path d="M8 5v14l11-7z"/></svg>
                                    {item.duration_seconds ? `${item.duration_seconds}s` : "Clip"}
                                  </div>
                                )}
                              </div>

                              {/* Metadata */}
                              <div className="space-y-1 text-xs font-mono">
                                <div className="flex items-center justify-between">
                                  <span className="text-zinc-500">Target Object:</span>
                                  <span className="text-zinc-200 font-semibold font-sans">
                                    {item.object_class ? formatClassName(item.object_class) : "Grouped Event"}
                                  </span>
                                </div>
                                {item.confidence != null && (
                                  <div className="flex items-center justify-between">
                                    <span className="text-zinc-500">Confidence:</span>
                                    <span className="text-emerald-400 font-bold">{Math.round(item.confidence * 100)}%</span>
                                  </div>
                                )}
                                <div className="flex items-center justify-between">
                                  <span className="text-zinc-500">Source Video:</span>
                                  <span className="text-zinc-400 truncate max-w-[150px]" title={item.source_video_name}>
                                    {item.source_video_name}
                                  </span>
                                </div>
                                {item.bounding_box && (
                                  <div className="flex items-center justify-between text-[11px]">
                                    <span className="text-zinc-500">Bounding Box:</span>
                                    <span className="text-zinc-400">
                                      [{Math.round(item.bounding_box.x1)},{Math.round(item.bounding_box.y1)} → {Math.round(item.bounding_box.x2)},{Math.round(item.bounding_box.y2)}]
                                    </span>
                                  </div>
                                )}
                              </div>
                            </div>

                            {/* Actions */}
                            <div className="pt-2 border-t border-zinc-800/80 flex items-center justify-between gap-2 font-mono text-xs">
                              <div className="flex items-center gap-1.5">
                                {item.has_snapshot && (
                                  <button
                                    onClick={() => {
                                      setSelectedEvidence(item);
                                      setEvidenceModalMode("snapshot");
                                    }}
                                    className="px-2 py-1 rounded bg-zinc-900 hover:bg-zinc-800 text-zinc-300 hover:text-white border border-zinc-800 text-[11px] transition-colors"
                                  >
                                    View Snapshot
                                  </button>
                                )}
                                {item.has_clip && (
                                  <button
                                    onClick={() => {
                                      setSelectedEvidence(item);
                                      setEvidenceModalMode("clip");
                                    }}
                                    className="px-2 py-1 rounded bg-amber-500/10 hover:bg-amber-500/20 text-amber-400 hover:text-amber-300 border border-amber-500/30 text-[11px] transition-colors"
                                  >
                                    Play Clip
                                  </button>
                                )}
                              </div>
                              <button
                                onClick={() => handleSeekToTimestamp(item.timestamp, item.event_id || undefined)}
                                className="text-emerald-400 hover:text-emerald-300 text-[11px] underline"
                                title="Seek surveillance player to this timestamp"
                              >
                                Jump
                              </button>
                            </div>
                          </div>
                        ))}
                      </div>
                    )}
                  </div>
                )}

                {/* ========================================================================= */}
                {/* PHASE 8: ADVANCED SECURITY INTELLIGENCE & COMPUTER VISION PANEL           */}
                {/* ========================================================================= */}
                {activeView === "intelligence" && (
                  <div className="space-y-6 animate-in fade-in duration-200">
                    {/* Intelligence Header & Action Bar */}
                    <div className="bg-zinc-950/90 border border-indigo-500/30 rounded-xl p-5 md:p-6 space-y-4 shadow-xl">
                      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 border-b border-zinc-800 pb-4">
                        <div className="flex items-center gap-3">
                          <div className="p-2 bg-indigo-500/10 border border-indigo-500/30 rounded-lg text-indigo-400">
                            <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m5.618-4.016A11.955 11.955 0 0112 2.944a11.955 11.955 0 01-8.618 3.04A12.02 12.02 0 003 9c0 5.591 3.824 10.29 9 11.622 5.176-1.332 9-6.03 9-11.622 0-1.042-.133-2.052-.382-3.016z" />
                            </svg>
                          </div>
                          <div>
                            <h3 className="text-base font-bold text-zinc-100 flex items-center gap-2">
                              SECURITY INTELLIGENCE &amp; COMPUTER VISION
                              <span className="text-[10px] font-mono font-semibold uppercase bg-indigo-500/20 text-indigo-300 border border-indigo-500/40 px-2 py-0.5 rounded">
                                Observable AI Analytics
                              </span>
                            </h3>
                            <p className="text-xs text-zinc-400">
                              Multi-frame tracking, vehicle color analysis, anonymous face regions, zone intrusions, loitering, and activity anomalies.
                            </p>
                          </div>
                        </div>

                        <div className="flex items-center gap-2 self-end md:self-auto">
                          <button
                            id="run-security-analysis-btn"
                            type="button"
                            onClick={handleRunSecurityAnalysis}
                            disabled={isAnalyzingIntelligence}
                            className="flex items-center gap-2 bg-indigo-600 hover:bg-indigo-500 disabled:bg-indigo-800/50 text-white text-xs font-semibold px-4 py-2 rounded-lg shadow transition-colors"
                          >
                            {isAnalyzingIntelligence ? (
                              <>
                                <svg className="animate-spin h-3.5 w-3.5" fill="none" viewBox="0 0 24 24">
                                  <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                                  <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
                                </svg>
                                <span>Analyzing Intelligence...</span>
                              </>
                            ) : (
                              <>
                                <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
                                </svg>
                                <span>Run Security Analysis</span>
                              </>
                            )}
                          </button>
                        </div>
                      </div>

                      {/* Feedback Banner */}
                      {intelligenceFeedback && (
                        <div className="rounded-lg p-3 text-xs bg-indigo-500/10 border border-indigo-500/30 text-indigo-300 flex items-center justify-between">
                          <span>{intelligenceFeedback}</span>
                          <button onClick={() => setIntelligenceFeedback(null)} className="underline hover:opacity-80 ml-3">Dismiss</button>
                        </div>
                      )}

                      {/* Intelligence Metrics Grid */}
                      <div className="grid grid-cols-2 sm:grid-cols-5 gap-3 pt-1">
                        <div className="bg-zinc-900/80 border border-zinc-800 rounded-lg p-3 text-center">
                          <div className="text-xl font-bold font-mono text-indigo-400">{tracks.length}</div>
                          <div className="text-[11px] text-zinc-400 font-mono mt-0.5">Object Tracks</div>
                        </div>
                        <div className="bg-zinc-900/80 border border-zinc-800 rounded-lg p-3 text-center">
                          <div className="text-xl font-bold font-mono text-amber-400">{vehicleAttributes.length}</div>
                          <div className="text-[11px] text-zinc-400 font-mono mt-0.5">Vehicle Colors</div>
                        </div>
                        <div className="bg-zinc-900/80 border border-zinc-800 rounded-lg p-3 text-center">
                          <div className="text-xl font-bold font-mono text-cyan-400">{faceDetections.length}</div>
                          <div className="text-[11px] text-zinc-400 font-mono mt-0.5">Face Regions</div>
                        </div>
                        <div className="bg-zinc-900/80 border border-zinc-800 rounded-lg p-3 text-center">
                          <div className="text-xl font-bold font-mono text-red-400">{securityEvents.length}</div>
                          <div className="text-[11px] text-zinc-400 font-mono mt-0.5">Security Events</div>
                        </div>
                        <div className="bg-zinc-900/80 border border-zinc-800 rounded-lg p-3 text-center col-span-2 sm:col-span-1">
                          <div className="text-xl font-bold font-mono text-emerald-400">{securityZones.length}</div>
                          <div className="text-[11px] text-zinc-400 font-mono mt-0.5">Defined Zones</div>
                        </div>
                      </div>

                      {/* Phase 10: Universal Incident Intelligence Status */}
                      <div className="bg-zinc-900/50 border border-zinc-800/80 rounded-lg p-2.5 flex flex-wrap items-center justify-between gap-2 text-xs">
                        <div className="flex items-center gap-2">
                          <span className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse" />
                          <span className="font-mono font-medium text-zinc-200">Incident Intelligence Engine</span>
                          <span className="text-[10px] font-mono text-zinc-400 border border-zinc-700 bg-zinc-800/60 px-1.5 py-0.5 rounded">Universal v1.0</span>
                        </div>
                        <div className="flex items-center gap-2 font-mono text-[11px] text-zinc-400">
                          <span className="text-zinc-500">Detectors:</span>
                          <span className="text-emerald-400 font-semibold">Active</span>
                          <span className="text-zinc-600">|</span>
                          <span className="text-zinc-500">Signals:</span>
                          <span className="text-indigo-400 font-semibold">Multi-Sensor</span>
                          <span className="text-zinc-600">|</span>
                          <span className="text-zinc-500">Verification:</span>
                          <span className="text-amber-400 font-semibold">Human Required</span>
                        </div>
                      </div>
                    </div>

                    {/* Section 1: Security Intelligence Events */}
                    <div className="space-y-3">
                      <div className="flex items-center justify-between">
                        <h4 className="text-xs font-mono font-bold uppercase tracking-wider text-zinc-300 flex items-center gap-2">
                          <span className="w-2 h-2 rounded-full bg-red-400" />
                          Security Intelligence Events ({securityEvents.length})
                        </h4>
                        <span className="text-[11px] text-zinc-500 font-mono">Observable rules &amp; correlations</span>
                      </div>

                      {securityEvents.length === 0 ? (
                        <div className="rounded-lg border border-zinc-800 bg-zinc-950/50 p-6 text-center text-xs text-zinc-400 font-mono">
                          No security events currently recorded. Define a zone or click &ldquo;Run Security Analysis&rdquo; to evaluate intrusions, prolonged presence, and activity peaks.
                        </div>
                      ) : (
                        <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                          {securityEvents.map((ev) => {
                            const isTheft = ev.event_type === "POTENTIAL_THEFT";
                            const isIntrusion = ev.event_type === "POTENTIAL_INTRUSION";
                            const isLoitering = ev.event_type === "PROLONGED_PRESENCE";
                            const isAbandoned = ev.event_type === "POTENTIAL_ABANDONED_OBJECT";
                            const isActivity = ev.event_type === "HIGH_ACTIVITY_PERIOD";
                            const isCollision = ev.event_type === "POTENTIAL_VEHICLE_COLLISION";
                            const isNearCollision = ev.event_type === "POTENTIAL_NEAR_COLLISION";
                            const isSuddenStop = ev.event_type === "POTENTIAL_SUDDEN_VEHICLE_STOP";
                            const isWrongWay = ev.event_type === "POTENTIAL_WRONG_WAY_VEHICLE";
                            const isUnusualTrajectory = ev.event_type === "POTENTIAL_UNUSUAL_VEHICLE_TRAJECTORY";
                            const isStationaryVehicle = ev.event_type === "POTENTIAL_STATIONARY_VEHICLE";
                            const isFall = ev.event_type === "POTENTIAL_PERSON_FALL";
                            const isPersonDown = ev.event_type === "POTENTIAL_PERSON_DOWN";
                            const isPanicRunning = ev.event_type === "POTENTIAL_PANIC_RUNNING";
                            const isRapidMovement = ev.event_type === "UNUSUAL_RAPID_PERSON_MOVEMENT";
                            const isAltercation = ev.event_type === "POTENTIAL_PHYSICAL_ALTERCATION";
                            const isForcedMovement = ev.event_type === "POTENTIAL_FORCED_MOVEMENT";
                            const isFollowing = ev.event_type === "PERSON_FOLLOWING";
                            const isCoordinated = ev.event_type === "COORDINATED_PERSON_MOVEMENT";
                            const isLeftBehind = ev.event_type === "POTENTIAL_OBJECT_LEFT_BEHIND";
                            const isPickup = ev.event_type === "POTENTIAL_OBJECT_PICKUP";
                            const isDisplacement = ev.event_type === "POTENTIAL_OBJECT_DISPLACEMENT";
                            const isTampering = ev.event_type === "POTENTIAL_PROPERTY_TAMPERING";
                            const isRestrictedObj = ev.event_type === "POTENTIAL_RESTRICTED_OBJECT_MOVEMENT";
                            const isRemoval = ev.event_type === "POTENTIAL_OBJECT_REMOVAL";
                            const isDensity = ev.event_type === "HIGH_PEDESTRIAN_DENSITY";
                            const isDensityIncrease = ev.event_type === "CROWD_DENSITY_INCREASE";
                            const isSurge = ev.event_type === "POTENTIAL_CROWD_SURGE";
                            const isDispersal = ev.event_type === "POTENTIAL_CROWD_DISPERSAL";
                            const isUnusualCrowdMovement = ev.event_type === "POTENTIAL_UNUSUAL_CROWD_MOVEMENT";
                            const isRestrictedZoneCrowding = ev.event_type === "POTENTIAL_RESTRICTED_ZONE_CROWDING";
                            const isUnusualZoneActivity = ev.event_type === "POTENTIAL_UNUSUAL_ZONE_ACTIVITY";
                            const isZoneOccupancy = ev.event_type === "ZONE_OCCUPANCY_OBSERVATION";

                            const badgeLabel = isTheft
                              ? "Potential Theft Pattern"
                              : isIntrusion
                              ? "Potential Intrusion"
                              : isLoitering
                              ? "Prolonged Presence"
                              : isAbandoned
                              ? "Abandoned Object"
                              : isActivity
                              ? "High Activity Period"
                              : isCollision
                              ? "Potential Vehicle Collision"
                              : isNearCollision
                              ? "Potential Near Collision"
                              : isSuddenStop
                              ? "Potential Sudden Stop"
                              : isWrongWay
                              ? "Potential Wrong-Way Vehicle"
                              : isUnusualTrajectory
                              ? "Unusual Vehicle Trajectory"
                              : isStationaryVehicle
                              ? "Potential Stationary Vehicle"
                              : isFall
                              ? "Potential Person Fall"
                              : isPersonDown
                              ? "Potential Person Down"
                              : isPanicRunning
                              ? "Potential Panic / Running"
                              : isRapidMovement
                              ? "Unusual Rapid Movement"
                              : isAltercation
                              ? "Potential Physical Altercation"
                              : isForcedMovement
                              ? "Potential Forced Movement"
                              : isFollowing
                              ? "Person Following"
                              : isCoordinated
                              ? "Potential Coordinated Movement"
                              : isLeftBehind
                              ? "Potential Object Left Behind"
                              : isPickup
                              ? "Potential Object Pickup"
                              : isDisplacement
                              ? "Potential Object Displacement"
                              : isTampering
                              ? "Potential Property Tampering"
                              : isRestrictedObj
                              ? "Restricted Object Movement"
                              : isRemoval
                              ? "Potential Object Removal"
                              : isDensity
                              ? "High Pedestrian Density"
                              : isDensityIncrease
                              ? "Crowd Density Increase"
                              : isSurge
                              ? "Potential Crowd Surge"
                              : isDispersal
                              ? "Potential Crowd Dispersal"
                              : isUnusualCrowdMovement
                              ? "Potential Unusual Crowd Movement"
                              : isRestrictedZoneCrowding
                              ? "Restricted-Zone Crowding"
                              : isUnusualZoneActivity
                              ? "Potential Unusual Zone Activity"
                              : isZoneOccupancy
                              ? "Zone Occupancy Observation"
                              : ev.event_type ? ev.event_type.replace(/_/g, " ") : "Security Event";

                            const badgeStyle = isTheft
                              ? "bg-rose-500/20 text-rose-300 border-rose-500/40 font-semibold"
                              : isIntrusion
                              ? "bg-red-500/20 text-red-400 border-red-500/40"
                              : isLoitering
                              ? "bg-amber-500/20 text-amber-400 border-amber-500/40"
                              : isAbandoned
                              ? "bg-purple-500/20 text-purple-400 border-purple-500/40"
                              : isActivity
                              ? "bg-blue-500/20 text-blue-400 border-blue-500/40"
                              : isCollision
                              ? "bg-orange-500/20 text-orange-400 border-orange-500/40"
                              : isNearCollision
                              ? "bg-yellow-500/20 text-yellow-400 border-yellow-500/40"
                              : isSuddenStop
                              ? "bg-amber-500/20 text-amber-300 border-amber-500/40"
                              : isWrongWay
                              ? "bg-red-600/20 text-red-300 border-red-600/40 font-semibold"
                              : isUnusualTrajectory
                              ? "bg-indigo-500/20 text-indigo-300 border-indigo-500/40"
                              : isStationaryVehicle
                              ? "bg-zinc-500/20 text-zinc-300 border-zinc-500/40"
                              : isFall
                              ? "bg-red-500/25 text-red-300 border-red-500/50 font-semibold"
                              : isPersonDown
                              ? "bg-amber-500/25 text-amber-300 border-amber-500/50 font-semibold"
                              : isPanicRunning
                              ? "bg-orange-500/20 text-orange-300 border-orange-500/40"
                              : isRapidMovement
                              ? "bg-purple-500/20 text-purple-300 border-purple-500/40"
                              : isAltercation
                              ? "bg-rose-600/25 text-rose-200 border-rose-600/50 font-semibold"
                              : isForcedMovement
                              ? "bg-amber-600/25 text-amber-200 border-amber-600/50 font-semibold"
                              : isFollowing
                              ? "bg-cyan-500/20 text-cyan-300 border-cyan-500/40"
                              : isCoordinated
                              ? "bg-teal-500/20 text-teal-300 border-teal-500/40"
                              : isLeftBehind
                              ? "bg-indigo-500/20 text-indigo-300 border-indigo-500/40"
                              : isPickup
                              ? "bg-sky-500/20 text-sky-300 border-sky-500/40"
                              : isDisplacement
                              ? "bg-blue-500/20 text-blue-300 border-blue-500/40"
                              : isTampering
                              ? "bg-amber-600/25 text-amber-300 border-amber-600/50 font-semibold"
                              : isRestrictedObj
                              ? "bg-red-500/20 text-red-300 border-red-500/40"
                              : isRemoval
                              ? "bg-violet-500/20 text-violet-300 border-violet-500/40"
                              : isDensity
                              ? "bg-blue-500/20 text-blue-300 border-blue-500/40"
                              : isDensityIncrease
                              ? "bg-indigo-500/20 text-indigo-300 border-indigo-500/40"
                              : isSurge
                              ? "bg-amber-500/25 text-amber-200 border-amber-500/50 font-semibold"
                              : isDispersal
                              ? "bg-teal-500/20 text-teal-300 border-teal-500/40"
                              : isUnusualCrowdMovement
                              ? "bg-purple-500/20 text-purple-300 border-purple-500/40"
                              : isRestrictedZoneCrowding
                              ? "bg-red-500/25 text-red-200 border-red-500/50 font-semibold"
                              : isUnusualZoneActivity
                              ? "bg-orange-500/20 text-orange-300 border-orange-500/40"
                              : isZoneOccupancy
                              ? "bg-cyan-500/20 text-cyan-300 border-cyan-500/40"
                              : "bg-emerald-500/20 text-emerald-400 border-emerald-500/40";

                            const matchedEvidence = evidenceList.find(
                              (e) => (ev.evidence_id && e.evidence_id === ev.evidence_id) || e.event_id === ev.id || Math.abs(e.timestamp - ev.timestamp) <= 1.0
                            );

                            return (
                              <div
                                key={ev.id}
                                className={`rounded-xl border bg-zinc-950/80 p-4 space-y-3 flex flex-col justify-between transition-all ${
                                  isTheft ? "border-rose-900/60 hover:border-rose-700 bg-rose-950/10" : "border-zinc-800 hover:border-zinc-700"
                                }`}
                              >
                                <div className="space-y-2">
                                  <div className="flex items-center justify-between gap-2">
                                    <span className={`text-[10px] font-mono uppercase px-2 py-0.5 rounded border ${badgeStyle}`}>
                                      {badgeLabel}
                                    </span>
                                    <div className="flex items-center gap-2">
                                      {matchedEvidence && (
                                        <span className="text-[10px] font-mono text-emerald-400 bg-emerald-500/10 border border-emerald-500/30 px-1.5 py-0.5 rounded">
                                          Evidence Preserved
                                        </span>
                                      )}
                                      <span className={`text-[10px] font-mono uppercase px-1.5 py-0.5 rounded ${
                                        ev.severity === "HIGH" ? "text-red-400 bg-red-500/10" : "text-zinc-400 bg-zinc-900"
                                      }`}>
                                        {ev.severity}
                                      </span>
                                    </div>
                                  </div>

                                  <p className="text-xs text-zinc-200 font-medium leading-relaxed">
                                    {ev.description}
                                  </p>

                                  <div className="flex flex-wrap gap-2 text-[11px] font-mono text-zinc-400">
                                    {ev.track_id && (
                                      <span className="bg-zinc-900 px-2 py-0.5 rounded border border-zinc-800 text-indigo-300">
                                        Track: {ev.track_id}
                                      </span>
                                    )}
                                    {ev.object_class && (
                                      <span className="bg-zinc-900 px-2 py-0.5 rounded border border-zinc-800 text-cyan-300">
                                        Target: {ev.object_class}
                                      </span>
                                    )}
                                    {ev.duration_seconds != null && ev.duration_seconds > 0 && (
                                      <span className="bg-zinc-900 px-2 py-0.5 rounded border border-zinc-800 text-amber-300">
                                        Duration: {ev.duration_seconds.toFixed(1)}s
                                      </span>
                                    )}
                                    {ev.zone_name && (
                                      <span className="bg-zinc-900 px-2 py-0.5 rounded border border-zinc-800 text-amber-300">
                                        Zone: {ev.zone_name}
                                      </span>
                                    )}
                                    <span className="bg-zinc-900 px-2 py-0.5 rounded border border-zinc-800 text-zinc-300">
                                      Pattern Evidence Strength: {Math.round(ev.confidence * 100)}%
                                    </span>
                                    {(() => {
                                      const valDec = ev.validation_decision || ev.incident_metadata?.validation_decision;
                                      const isAccepted = valDec === "ACCEPTED";
                                      const isReview = valDec === "REVIEW_REQUIRED" || (!isAccepted && Boolean(ev.human_verification_required));
                                      if (isAccepted) {
                                        return (
                                          <span className="bg-emerald-500/15 px-2 py-0.5 rounded border border-emerald-500/40 text-emerald-300 font-mono text-[10px]">
                                            ACCEPTED
                                          </span>
                                        );
                                      } else if (isReview) {
                                        return (
                                          <span className="bg-amber-500/15 px-2 py-0.5 rounded border border-amber-500/40 text-amber-300 font-mono text-[10px]">
                                            Review Required
                                          </span>
                                        );
                                      }
                                      return null;
                                    })()}
                                  </div>

                                  {Array.isArray(ev.observable_signals) && ev.observable_signals.length > 0 && (
                                    <div className="bg-zinc-900/60 p-2 rounded-lg border border-zinc-800/80 space-y-1">
                                      <span className="text-[10px] text-zinc-400 font-mono block uppercase">Observable Physical Signals:</span>
                                      {ev.observable_signals.map((sig, idx) => (
                                        <div key={idx} className="text-[11px] font-mono text-zinc-300 flex items-center gap-1.5">
                                          <span className="text-zinc-500">•</span>
                                          <span>{sig}</span>
                                        </div>
                                      ))}
                                    </div>
                                  )}
                                </div>

                                <div className="flex items-center justify-between pt-2 border-t border-zinc-800/80">
                                  <button
                                    type="button"
                                    onClick={() => handleSeekToTimestamp(ev.timestamp, ev.id)}
                                    className="text-xs font-mono text-emerald-400 hover:text-emerald-300 flex items-center gap-1"
                                  >
                                    <span>▶ Jump {formatTimestamp(ev.timestamp)}</span>
                                  </button>

                                  {matchedEvidence ? (
                                    <button
                                      type="button"
                                      onClick={() => {
                                        setSelectedEvidence(matchedEvidence);
                                        setEvidenceModalMode("clip");
                                      }}
                                      className="text-xs font-mono bg-emerald-500/15 hover:bg-emerald-500/25 text-emerald-300 border border-emerald-500/40 px-3 py-1 rounded transition-colors flex items-center gap-1.5 font-semibold"
                                    >
                                      <span>🎬 Review Evidence</span>
                                    </button>
                                  ) : (
                                    <button
                                      type="button"
                                      onClick={() => handleCaptureEvidence(ev.timestamp, ev.id, "snapshot_and_clip")}
                                      disabled={capturingEventId === ev.id}
                                      className="text-xs font-mono bg-amber-500/10 hover:bg-amber-500/20 text-amber-300 border border-amber-500/30 px-2.5 py-1 rounded transition-colors disabled:opacity-50"
                                    >
                                      {capturingEventId === ev.id ? "Capturing..." : "Capture Evidence"}
                                    </button>
                                  )}
                                </div>
                              </div>
                            );
                          })}
                        </div>
                      )}
                    </div>

                    {/* Section 2: Multi-Frame Object Tracking */}
                    <div className="space-y-3">
                      <div className="flex items-center justify-between">
                        <h4 className="text-xs font-mono font-bold uppercase tracking-wider text-zinc-300 flex items-center gap-2">
                          <span className="w-2 h-2 rounded-full bg-indigo-400" />
                          Multi-Frame Tracked Objects ({tracks.length})
                        </h4>
                        <span className="text-[11px] text-zinc-500 font-mono">Correlated across frames (No biometric identity)</span>
                      </div>

                      {tracks.length === 0 ? (
                        <div className="rounded-lg border border-zinc-800 bg-zinc-950/50 p-6 text-center text-xs text-zinc-400 font-mono">
                          No multi-frame tracks generated yet. Click &ldquo;Run Security Analysis&rdquo; above.
                        </div>
                      ) : (
                        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3">
                          {tracks.map((trk) => (
                            <div
                              key={trk.track_id}
                              className="rounded-lg border border-zinc-800 bg-zinc-950/80 p-3 space-y-2 hover:border-zinc-700 transition-colors"
                            >
                              <div className="flex items-center justify-between">
                                <span className="font-mono text-xs font-bold text-indigo-400 bg-indigo-500/10 border border-indigo-500/30 px-2 py-0.5 rounded">
                                  {trk.track_id}
                                </span>
                                <span className={`text-[10px] font-mono px-2 py-0.5 rounded ${getClassColor(trk.object_class)}`}>
                                  {formatClassName(trk.object_class)}
                                </span>
                              </div>

                              <div className="text-xs font-mono text-zinc-400 space-y-0.5">
                                <div>Duration: <span className="text-zinc-200">{trk.duration_seconds.toFixed(1)}s</span> ({trk.first_seen.toFixed(1)}s &rarr; {trk.last_seen.toFixed(1)}s)</div>
                                <div>Detections: <span className="text-zinc-200">{trk.detection_count}</span> • Max Conf: <span className="text-emerald-400">{Math.round(trk.max_confidence * 100)}%</span></div>
                                {trk.color && (
                                  <div className="flex items-center gap-1.5 pt-0.5">
                                    <span>Analyzed Color:</span>
                                    <span className="text-amber-300 font-semibold uppercase">{trk.color}</span>
                                    {trk.color_confidence && <span className="text-zinc-500">({Math.round(trk.color_confidence * 100)}%)</span>}
                                  </div>
                                )}
                              </div>

                              <button
                                type="button"
                                onClick={() => handleSeekToTimestamp(trk.first_seen)}
                                className="w-full text-xs font-mono bg-zinc-900 hover:bg-zinc-800 border border-zinc-800 text-zinc-300 hover:text-white py-1.5 rounded transition-colors text-center"
                              >
                                Jump to first seen ({formatTimestamp(trk.first_seen)})
                              </button>
                            </div>
                          ))}
                        </div>
                      )}
                    </div>

                    {/* Section 3: Vehicle Color Analysis */}
                    <div className="space-y-3">
                      <div className="flex items-center justify-between">
                        <h4 className="text-xs font-mono font-bold uppercase tracking-wider text-zinc-300 flex items-center gap-2">
                          <span className="w-2 h-2 rounded-full bg-amber-400" />
                          Vehicle Visual Color Classifications ({vehicleAttributes.length})
                        </h4>
                        <span className="text-[11px] text-zinc-500 font-mono">HSV chromatic space extraction</span>
                      </div>

                      {vehicleAttributes.length === 0 ? (
                        <div className="rounded-lg border border-zinc-800 bg-zinc-950/50 p-6 text-center text-xs text-zinc-400 font-mono">
                          No vehicle color records extracted. Run Security Analysis on video with detected vehicles.
                        </div>
                      ) : (
                        <div className="grid grid-cols-2 sm:grid-cols-4 lg:grid-cols-6 gap-2.5">
                          {vehicleAttributes.map((attr, idx) => (
                            <div
                              key={attr.id || idx}
                              onClick={() => handleSeekToTimestamp(attr.timestamp)}
                              className="rounded-lg border border-zinc-800 bg-zinc-950/80 p-2.5 space-y-1.5 cursor-pointer hover:border-amber-500/50 transition-colors"
                            >
                              <div className="flex items-center justify-between">
                                <span className="font-mono text-xs font-bold text-amber-300 uppercase flex items-center gap-1">
                                  <span
                                    className="w-2 h-2 rounded-full inline-block"
                                    style={{ backgroundColor: attr.color === "white" ? "#fff" : attr.color === "black" ? "#222" : attr.color }}
                                  />
                                  {attr.color}
                                </span>
                                <span className="text-[10px] font-mono text-zinc-400">{Math.round(attr.confidence * 100)}%</span>
                              </div>
                              <div className="text-[11px] font-mono text-zinc-400">
                                {formatTimestamp(attr.timestamp)} • {formatClassName(attr.object_class, "Vehicle")}
                              </div>
                            </div>
                          ))}
                        </div>
                      )}
                    </div>

                    {/* Section 4: Face Visual Region Detections (Strict Safety) */}
                    <div className="space-y-3">
                      <div className="rounded-lg bg-cyan-500/10 border border-cyan-500/30 p-3.5 space-y-1">
                        <div className="flex items-center gap-2 text-xs font-bold text-cyan-300 font-mono">
                          <svg className="w-4 h-4 text-cyan-400 shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                          </svg>
                          <span>STRICT OBSERVATIONAL CONSTRAINT — NO BIOMETRIC IDENTIFICATION</span>
                        </div>
                        <p className="text-[11px] text-zinc-400 leading-relaxed pl-6">
                          In compliance with Sentinel security standards, facial recognition, person identification, biometric indexing, and name inference are strictly disabled. These records represent anonymous visual detection bounding boxes only.
                        </p>
                      </div>

                      <div className="flex items-center justify-between">
                        <h4 className="text-xs font-mono font-bold uppercase tracking-wider text-zinc-300 flex items-center gap-2">
                          <span className="w-2 h-2 rounded-full bg-cyan-400" />
                          Anonymous Face Visual Regions ({faceDetections.length})
                        </h4>
                      </div>

                      {faceDetections.length === 0 ? (
                        <div className="rounded-lg border border-zinc-800 bg-zinc-950/50 p-6 text-center text-xs text-zinc-400 font-mono">
                          No face visual regions detected for this video.
                        </div>
                      ) : (
                        <div className="grid grid-cols-2 sm:grid-cols-4 lg:grid-cols-6 gap-2.5">
                          {faceDetections.map((f, idx) => (
                            <div
                              key={f.id || idx}
                              onClick={() => handleSeekToTimestamp(f.timestamp)}
                              className="rounded-lg border border-zinc-800 bg-zinc-950/80 p-2.5 space-y-1 cursor-pointer hover:border-cyan-500/50 transition-colors font-mono text-xs"
                            >
                              <div className="text-cyan-400 font-bold">{formatTimestamp(f.timestamp)}</div>
                              <div className="text-zinc-400 text-[11px]">Conf: {Math.round(f.confidence * 100)}%</div>
                              {f.track_id && <div className="text-[10px] text-indigo-400 truncate">{f.track_id}</div>}
                            </div>
                          ))}
                        </div>
                      )}
                    </div>

                    {/* Section 5: Restricted Security Zones */}
                    <div className="space-y-4 pt-2 border-t border-zinc-800">
                      <div className="flex items-center justify-between">
                        <h4 className="text-xs font-mono font-bold uppercase tracking-wider text-zinc-300 flex items-center gap-2">
                          <span className="w-2 h-2 rounded-full bg-emerald-400" />
                          Restricted Security Zones ({securityZones.length})
                        </h4>
                        <span className="text-[11px] text-zinc-500 font-mono">Polygon intrusion &amp; loitering bounds</span>
                      </div>

                      {/* Defined Zones List */}
                      {securityZones.length > 0 && (
                        <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
                          {securityZones.map((z) => (
                            <div key={z.zone_id} className="rounded-lg border border-zinc-800 bg-zinc-950/80 p-3 space-y-2 flex items-center justify-between">
                              <div>
                                <div className="text-xs font-bold text-zinc-200">{z.name}</div>
                                <div className="text-[10px] font-mono text-zinc-400">
                                  {z.target_classes?.join(", ") || "person, car"} • {z.polygon?.length || 0} vertices
                                </div>
                              </div>
                              <button
                                type="button"
                                onClick={() => handleDeleteZone(z.zone_id)}
                                className="text-xs font-mono text-red-400 hover:text-red-300 p-1 rounded hover:bg-zinc-900"
                                title="Delete Zone"
                              >
                                ✕
                              </button>
                            </div>
                          ))}
                        </div>
                      )}

                      {/* Add Zone Widget */}
                      <div className="rounded-xl border border-zinc-800 bg-zinc-950/60 p-4 space-y-3">
                        <div className="text-xs font-bold text-zinc-200 font-mono">Define Restricted Security Zone</div>
                        <div className="flex flex-col sm:flex-row gap-2">
                          <input
                            type="text"
                            placeholder="Zone Name (e.g. Restricted Gate, Perimeter A)"
                            value={newZoneName}
                            onChange={(e) => setNewZoneName(e.target.value)}
                            className="bg-zinc-900 border border-zinc-700 text-zinc-200 text-xs rounded-lg px-3 py-2 flex-1 focus:outline-none focus:border-indigo-500 font-mono"
                          />
                          <select
                            value={newZonePreset}
                            onChange={(e) => setNewZonePreset(e.target.value)}
                            className="bg-zinc-900 border border-zinc-700 text-zinc-200 text-xs rounded-lg px-3 py-2 focus:outline-none focus:border-indigo-500 font-mono"
                          >
                            <option value="gate">Preset: Entrance Gate (50,50 &rarr; 250,350)</option>
                            <option value="perimeter">Preset: Perimeter Boundary (0,0 &rarr; 400,150)</option>
                            <option value="loading">Preset: Loading Bay (150,150 &rarr; 380,380)</option>
                          </select>
                          <button
                            type="button"
                            onClick={handleCreateZone}
                            disabled={isCreatingZone || !newZoneName.trim()}
                            className="bg-emerald-600 hover:bg-emerald-500 disabled:opacity-50 text-white text-xs font-semibold px-4 py-2 rounded-lg transition-colors font-mono"
                          >
                            {isCreatingZone ? "Adding..." : "+ Add Zone"}
                          </button>
                        </div>
                      </div>
                    </div>
                  </div>
                )}

                {/* PHASE 16: CORRELATED INCIDENTS VIEW */}
                {activeView === "correlation" && uploadResult?.video_id && (
                  <div className="animate-in fade-in duration-200">
                    <CorrelatedIncidentsView
                      videoId={uploadResult.video_id}
                      onSeek={handleSeekToTimestamp}
                    />
                  </div>
                )}

                {/* PHASE 17: FORENSIC INVESTIGATION WORKBENCH */}
                {activeView === "forensic" && uploadResult?.video_id && (
                  <div className="animate-in fade-in duration-200">
                    <ForensicSearchPanel
                      videoId={uploadResult.video_id}
                      videoDuration={processingResult?.duration_seconds}
                      onSeek={handleSeekToTimestamp}
                    />
                  </div>
                )}

                {/* INCIDENT DOSSIER VIEW */}
                {activeView === "reports" && (
                  <div className="space-y-5 animate-in fade-in duration-200">
                    {/* Top Action Banner */}
                    <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 bg-zinc-950 p-5 rounded-xl border border-zinc-800">
                      <div className="flex items-start sm:items-center gap-3">
                        <div className="p-2.5 bg-emerald-500/10 border border-emerald-500/30 rounded-lg text-emerald-400">
                          <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                          </svg>
                        </div>
                        <div>
                          <h3 className="text-base font-bold text-zinc-100 flex items-center gap-2">
                            SECURITY INCIDENT DOSSIER GENERATOR
                            <span className="text-[10px] font-mono font-semibold uppercase bg-emerald-500/20 text-emerald-300 border border-emerald-500/40 px-2 py-0.5 rounded">
                              Forensic Standard
                            </span>
                          </h3>
                          <p className="text-xs text-zinc-400 mt-0.5">
                            Compile professional, audit-ready PDF investigation dossiers with executive summaries, detection analytics, security events, and embedded evidence.
                          </p>
                        </div>
                      </div>

                      <div className="flex items-center gap-2">
                        <button
                          id="generate-dossier-btn"
                          onClick={handleGenerateReport}
                          disabled={isGeneratingReport}
                          className="bg-emerald-600 hover:bg-emerald-500 disabled:opacity-50 text-white text-xs font-semibold px-4 py-2.5 rounded-lg transition-colors flex items-center gap-2 font-mono shadow-lg shadow-emerald-950/40 cursor-pointer"
                        >
                          {isGeneratingReport ? (
                            <>
                              <span className="h-2.5 w-2.5 rounded-full bg-white animate-ping" />
                              <span>Compiling Dossier PDF...</span>
                            </>
                          ) : (
                            <>
                              <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4v16m8-8H4" />
                              </svg>
                              <span>Generate Incident Dossier</span>
                            </>
                          )}
                        </button>
                      </div>
                    </div>

                    {/* Generation Feedback Banner */}
                    {reportFeedback && (
                      <div
                        id="report-feedback-banner"
                        className={`p-3.5 rounded-lg border text-xs font-mono flex items-center justify-between ${
                          reportFeedback.type === "success"
                            ? "bg-emerald-950/40 border-emerald-500/40 text-emerald-300"
                            : "bg-red-950/40 border-red-500/40 text-red-300"
                        }`}
                      >
                        <span>{reportFeedback.message}</span>
                        <button
                          onClick={() => setReportFeedback(null)}
                          className="text-zinc-400 hover:text-white text-sm ml-2"
                        >
                          ✕
                        </button>
                      </div>
                    )}

                    {/* Latest Report Highlight Card */}
                    {reportsList.length > 0 && (
                      <div className="bg-zinc-900/70 border border-emerald-500/30 rounded-xl p-5 space-y-4">
                        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 border-b border-zinc-800 pb-3">
                          <div className="space-y-1">
                            <div className="flex items-center gap-2">
                              <span className="h-2 w-2 rounded-full bg-emerald-400 animate-pulse" />
                              <span className="text-xs font-mono font-bold text-emerald-400 uppercase tracking-wide">
                                Latest Incident Dossier
                              </span>
                              <span className="text-xs font-mono text-zinc-300 bg-zinc-800 px-2 py-0.5 rounded border border-zinc-700">
                                {reportsList[0].report_id}
                              </span>
                            </div>
                            <h4 className="text-sm font-semibold text-zinc-100">{reportsList[0].title}</h4>
                          </div>

                          <div className="flex items-center gap-2 font-mono text-xs">
                            <a
                              id="view-report-link"
                              href={getReportViewUrl(reportsList[0].report_id)}
                              target="_blank"
                              rel="noopener noreferrer"
                              className="bg-zinc-800 hover:bg-zinc-700 text-zinc-200 border border-zinc-700 px-3.5 py-2 rounded-lg transition-colors flex items-center gap-1.5"
                            >
                              <svg className="w-3.5 h-3.5 text-zinc-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M2.458 12C3.732 7.943 7.523 5 12 5c4.478 0 8.268 2.943 9.542 7-1.274 4.057-5.064 7-9.542 7-4.477 0-8.268-2.943-9.542-7z" />
                              </svg>
                              <span>View Report</span>
                            </a>
                            <a
                              id="download-report-link"
                              href={getReportDownloadUrl(reportsList[0].report_id)}
                              download
                              className="bg-emerald-600 hover:bg-emerald-500 text-white font-semibold px-3.5 py-2 rounded-lg transition-colors flex items-center gap-1.5 shadow-md shadow-emerald-950/30"
                            >
                              <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4" />
                              </svg>
                              <span>Download PDF</span>
                            </a>
                          </div>
                        </div>

                        {/* Report Key Stats Grid */}
                        <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 font-mono text-xs">
                          <div className="bg-zinc-950/60 p-3 rounded-lg border border-zinc-800/80">
                            <span className="text-zinc-500 block text-[11px]">Pages & Size</span>
                            <span className="text-zinc-100 font-bold">
                              {reportsList[0].page_count} Pages &bull; {(reportsList[0].file_size_bytes / 1024).toFixed(1)} KB
                            </span>
                          </div>
                          <div className="bg-zinc-950/60 p-3 rounded-lg border border-zinc-800/80">
                            <span className="text-zinc-500 block text-[11px]">Detections Indexed</span>
                            <span className="text-blue-400 font-bold">
                              {reportsList[0].metadata?.total_detections ?? 0} detections
                            </span>
                          </div>
                          <div className="bg-zinc-950/60 p-3 rounded-lg border border-zinc-800/80">
                            <span className="text-zinc-500 block text-[11px]">Security Events</span>
                            <span className="text-amber-400 font-bold">
                              {reportsList[0].metadata?.total_security_events ?? 0} events
                            </span>
                          </div>
                          <div className="bg-zinc-950/60 p-3 rounded-lg border border-zinc-800/80">
                            <span className="text-zinc-500 block text-[11px]">Takeaway / Theft Alert</span>
                            <span className={reportsList[0].metadata?.has_theft_event ? "text-red-400 font-bold" : "text-emerald-400 font-bold"}>
                              {reportsList[0].metadata?.has_theft_event ? "POTENTIAL THEFT FLAGGED" : "None Flagged"}
                            </span>
                          </div>
                        </div>
                      </div>
                    )}

                    {/* All Generated Reports History */}
                    <div className="bg-zinc-950 rounded-xl border border-zinc-800 overflow-hidden">
                      <div className="p-4 border-b border-zinc-800 flex items-center justify-between">
                        <h4 className="text-xs font-mono font-bold text-zinc-300 uppercase tracking-wide">
                          Generated Dossier Archives ({reportsList.length})
                        </h4>
                        {uploadResult?.video_id && (
                          <button
                            onClick={() => fetchReports(uploadResult.video_id)}
                            className="text-xs font-mono text-zinc-400 hover:text-zinc-200"
                          >
                            Refresh Archives
                          </button>
                        )}
                      </div>

                      {reportsList.length === 0 ? (
                        <div className="p-10 text-center space-y-3">
                          <div className="w-12 h-12 rounded-full bg-zinc-900 border border-zinc-800 flex items-center justify-center mx-auto text-zinc-500">
                            <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                            </svg>
                          </div>
                          <div className="space-y-1">
                            <p className="text-sm font-semibold text-zinc-300">No Incident Dossiers Generated</p>
                            <p className="text-xs text-zinc-500 max-w-md mx-auto">
                              Generate an audit-ready PDF report compiling the chronological timeline, security events, detection statistics, and embedded evidence snapshots.
                            </p>
                          </div>
                        </div>
                      ) : (
                        <div className="divide-y divide-zinc-800/80 font-mono text-xs">
                          {reportsList.map((rep) => (
                            <div key={rep.id} className="p-4 flex flex-col sm:flex-row sm:items-center justify-between gap-3 hover:bg-zinc-900/30 transition-colors">
                              <div className="space-y-1">
                                <div className="flex items-center gap-2">
                                  <span className="font-bold text-zinc-100">{rep.report_id}</span>
                                  <span className="text-[10px] text-zinc-400 bg-zinc-900 px-2 py-0.5 rounded border border-zinc-800">
                                    {rep.page_count} Pages
                                  </span>
                                  <span className="text-[10px] text-zinc-500">
                                    {(rep.file_size_bytes / 1024).toFixed(1)} KB
                                  </span>
                                </div>
                                <p className="text-zinc-400 font-sans text-xs">{rep.title}</p>
                                <span className="text-[11px] text-zinc-500">
                                  {rep.generated_at ? new Date(rep.generated_at).toLocaleString() : "Generated"}
                                </span>
                              </div>

                              <div className="flex items-center gap-2">
                                <a
                                  href={getReportViewUrl(rep.report_id)}
                                  target="_blank"
                                  rel="noopener noreferrer"
                                  className="px-3 py-1.5 bg-zinc-900 hover:bg-zinc-800 text-zinc-300 border border-zinc-700 rounded-md transition-colors flex items-center gap-1"
                                >
                                  View
                                </a>
                                <a
                                  href={getReportDownloadUrl(rep.report_id)}
                                  download
                                  className="px-3 py-1.5 bg-emerald-600/20 hover:bg-emerald-600/30 text-emerald-300 border border-emerald-500/30 rounded-md transition-colors flex items-center gap-1"
                                >
                                  Download PDF
                                </a>
                              </div>
                            </div>
                          ))}
                        </div>
                      )}
                    </div>
                  </div>
                )}

                {/* ------------------------------------------------------------- */}
                {/* Phase 15: Specialized Visual Intelligence View                */}
                {/* ------------------------------------------------------------- */}
                {activeView === "specialized" && (
                  <div className="space-y-6">
                    {/* Header Banner */}
                    <div className="bg-gradient-to-r from-rose-950/40 via-zinc-900 to-zinc-900 border border-rose-500/30 rounded-xl p-5 flex flex-col sm:flex-row sm:items-center justify-between gap-4">
                      <div className="space-y-1">
                        <div className="flex items-center gap-2">
                          <span className="h-2.5 w-2.5 rounded-full bg-rose-500 animate-pulse" />
                          <h3 className="text-sm font-mono font-bold text-rose-400 uppercase tracking-wide">
                            Specialized Visual Intelligence
                          </h3>
                        </div>
                        <p className="text-xs text-zinc-400 font-sans max-w-2xl leading-relaxed">
                          Forensic visual detection layer for fire, smoke, weapons/suspicious objects, and posture dynamics. All detections evaluate temporal persistence and negative evidence arbitration; mandatory human review required.
                        </p>
                      </div>
                      {uploadResult?.video_id && (
                        <button
                          onClick={() => fetchSpecialized(uploadResult.video_id)}
                          disabled={isFetchingSpecialized}
                          className="bg-zinc-800 hover:bg-zinc-700 text-zinc-200 text-xs font-mono px-3.5 py-2 rounded-lg border border-zinc-700 flex items-center gap-1.5 shrink-0 transition-colors"
                        >
                          <svg className={`w-3.5 h-3.5 ${isFetchingSpecialized ? "animate-spin" : ""}`} fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
                          </svg>
                          <span>{isFetchingSpecialized ? "Refreshing..." : "Refresh"}</span>
                        </button>
                      )}
                    </div>

                    {/* View Mode Toggle: Aggregated Incidents vs Raw Observations */}
                    <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 bg-zinc-950 p-3 rounded-xl border border-zinc-800">
                      <div className="flex items-center gap-2">
                        <span className="text-xs text-zinc-400 font-mono">View Mode:</span>
                        <div className="flex items-center bg-zinc-900 rounded-lg p-0.5 border border-zinc-800">
                          <button
                            onClick={() => setSpecializedViewMode("aggregated")}
                            className={`px-3 py-1 text-xs font-mono rounded-md transition-all ${
                              specializedViewMode === "aggregated"
                                ? "bg-rose-600 text-white font-bold shadow"
                                : "text-zinc-400 hover:text-zinc-200"
                            }`}
                          >
                            Aggregated Incidents ({specializedAggregatedIncidents.length})
                          </button>
                          <button
                            onClick={() => setSpecializedViewMode("raw")}
                            className={`px-3 py-1 text-xs font-mono rounded-md transition-all ${
                              specializedViewMode === "raw"
                                ? "bg-rose-600 text-white font-bold shadow"
                                : "text-zinc-400 hover:text-zinc-200"
                            }`}
                          >
                            Raw Observations ({specializedList.length})
                          </button>
                        </div>
                      </div>
                      <span className="text-[11px] text-zinc-500 font-mono">
                        {specializedViewMode === "aggregated"
                          ? "Continuous visual episodes grouped temporally"
                          : "Frame-by-frame forensic detections"}
                      </span>
                    </div>

                    {/* Sub-Filters */}
                    <div className="flex flex-wrap items-center gap-2 font-mono text-xs">
                      <button
                        onClick={() => setSpecializedFilter("all")}
                        className={`px-3 py-1.5 rounded-lg border transition-all ${
                          specializedFilter === "all"
                            ? "bg-rose-600 text-white border-rose-500 font-bold"
                            : "bg-zinc-900 text-zinc-400 border-zinc-800 hover:text-zinc-200"
                        }`}
                      >
                        All ({specializedList.length})
                      </button>
                      <button
                        onClick={() => setSpecializedFilter("fire")}
                        className={`px-3 py-1.5 rounded-lg border transition-all ${
                          specializedFilter === "fire"
                            ? "bg-amber-600 text-white border-amber-500 font-bold"
                            : "bg-zinc-900 text-zinc-400 border-zinc-800 hover:text-zinc-200"
                        }`}
                      >
                        Fire ({specializedList.filter(o => o.class_name.toLowerCase().includes("fire")).length})
                      </button>
                      <button
                        onClick={() => setSpecializedFilter("smoke")}
                        className={`px-3 py-1.5 rounded-lg border transition-all ${
                          specializedFilter === "smoke"
                            ? "bg-slate-600 text-white border-slate-500 font-bold"
                            : "bg-zinc-900 text-zinc-400 border-zinc-800 hover:text-zinc-200"
                        }`}
                      >
                        Smoke ({specializedList.filter(o => o.class_name.toLowerCase().includes("smoke")).length})
                      </button>
                      <button
                        onClick={() => setSpecializedFilter("weapon")}
                        className={`px-3 py-1.5 rounded-lg border transition-all ${
                          specializedFilter === "weapon"
                            ? "bg-red-600 text-white border-red-500 font-bold"
                            : "bg-zinc-900 text-zinc-400 border-zinc-800 hover:text-zinc-200"
                        }`}
                      >
                        Weapon / Object ({specializedList.filter(o => o.class_name.toLowerCase().includes("weapon") || o.class_name.toLowerCase().includes("knife") || o.class_name.toLowerCase().includes("gun")).length})
                      </button>
                      <button
                        onClick={() => setSpecializedFilter("pose")}
                        className={`px-3 py-1.5 rounded-lg border transition-all ${
                          specializedFilter === "pose"
                            ? "bg-indigo-600 text-white border-indigo-500 font-bold"
                            : "bg-zinc-900 text-zinc-400 border-zinc-800 hover:text-zinc-200"
                        }`}
                      >
                        Pose / Action ({specializedList.filter(o => o.class_name.toLowerCase().includes("pose") || o.class_name.toLowerCase().includes("posture") || o.detector_name.toLowerCase().includes("pose")).length})
                      </button>
                    </div>

                    {/* Observations List */}
                    {specializedList.length === 0 ? (
                      <div className="rounded-xl border border-zinc-800 bg-zinc-950 p-12 text-center space-y-3">
                        <div className="w-12 h-12 rounded-full bg-zinc-900 border border-zinc-800 flex items-center justify-center mx-auto text-zinc-500">
                          <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M17.657 18.657A8 8 0 016.343 7.343S7 9 9 10c0-2 .5-5 2.986-7C14 5 16.09 5.777 17.656 7.343A7.975 7.975 0 0120 13a7.975 7.975 0 01-2.343 5.657z" />
                          </svg>
                        </div>
                        <div className="space-y-1">
                          <p className="text-sm font-semibold text-zinc-300">No Specialized Visual Observations</p>
                          <p className="text-xs text-zinc-500 max-w-md mx-auto">
                            Specialized models evaluate frame sequences for persistent chromatic, plume, or silhouette signatures. Transient reflections, global fog, and single-frame flickers are rejected by negative evidence filters.
                          </p>
                        </div>
                      </div>
                    ) : specializedViewMode === "aggregated" ? (
                      /* AGGREGATED INCIDENTS VIEW */
                      <div className="space-y-3 font-mono">
                        {specializedAggregatedIncidents
                          .filter((item) => {
                            if (specializedFilter === "all") return true;
                            const et = (item.event_type || "").toLowerCase();
                            if (specializedFilter === "fire") return et.includes("fire");
                            if (specializedFilter === "smoke") return et.includes("smoke");
                            if (specializedFilter === "weapon") return et.includes("weapon") || et.includes("knife") || et.includes("gun");
                            if (specializedFilter === "pose") return et.includes("pose") || et.includes("posture");
                            return true;
                          })
                          .map((item) => {
                            const isAccepted = item.validation_status === "VALID" || item.validation_status === "ACCEPTED";
                            const isRejected = item.validation_status === "REJECTED";
                            const isExpanded = expandedIncidentIds.has(item.id);

                            return (
                              <div
                                key={item.id}
                                className={`rounded-xl border p-4 transition-all bg-zinc-900/70 ${
                                  isAccepted
                                    ? "border-emerald-500/40 hover:border-emerald-500/60"
                                    : isRejected
                                    ? "border-zinc-800/80 opacity-60 hover:opacity-100"
                                    : "border-amber-500/40 hover:border-amber-500/60"
                                }`}
                              >
                                <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3">
                                  <div className="flex items-center gap-3">
                                    <div className="flex items-center gap-1.5">
                                      <button
                                        onClick={() => handleSeekToTimestamp(item.start_time)}
                                        className="text-xs font-bold bg-black/50 hover:bg-black/80 border border-zinc-700 hover:border-rose-500 text-rose-400 px-2.5 py-1.5 rounded-lg transition-all"
                                        title="Jump to incident start"
                                      >
                                        {formatTimestamp(item.start_time)}
                                      </button>
                                      <span className="text-zinc-500 text-xs">→</span>
                                      <button
                                        onClick={() => handleSeekToTimestamp(item.end_time)}
                                        className="text-xs font-bold bg-black/50 hover:bg-black/80 border border-zinc-700 hover:border-rose-500 text-rose-400 px-2.5 py-1.5 rounded-lg transition-all"
                                        title="Jump to incident end"
                                      >
                                        {formatTimestamp(item.end_time)}
                                      </button>
                                      {item.duration_seconds > 0 && (
                                        <span className="text-[10px] text-zinc-400 bg-zinc-800/80 px-2 py-0.5 rounded border border-zinc-700">
                                          {item.duration_seconds.toFixed(1)}s
                                        </span>
                                      )}
                                    </div>
                                    <div>
                                      <div className="flex items-center gap-2">
                                        <span className="text-sm font-bold text-zinc-100 uppercase">
                                          {item.event_type.replace(/_/g, " ")}
                                        </span>
                                        {isAccepted && (
                                          <span className="text-[10px] font-bold px-2 py-0.5 rounded border bg-emerald-500/20 text-emerald-300 border-emerald-500/40">
                                            ACCEPTED
                                          </span>
                                        )}
                                        {!isAccepted && !isRejected && (
                                          <span className="text-[10px] font-bold px-2 py-0.5 rounded border bg-amber-500/20 text-amber-300 border-amber-500/40">
                                            REVIEW REQUIRED
                                          </span>
                                        )}
                                        {isRejected && (
                                          <span className="text-[10px] font-bold px-2 py-0.5 rounded border bg-rose-500/20 text-rose-400 border-rose-500/40">
                                            REJECTED
                                          </span>
                                        )}
                                      </div>
                                      <div className="text-xs text-zinc-400 mt-0.5 flex flex-wrap items-center gap-2">
                                        <span>Observations: <strong className="text-zinc-200">{item.observations_count}</strong></span>
                                        {item.segment_count > 1 && (
                                          <>
                                            <span>&bull;</span>
                                            <span>Segments: <strong className="text-zinc-200">{item.segment_count}</strong></span>
                                          </>
                                        )}
                                        <span>&bull;</span>
                                        <span>Max Confidence: <strong className="text-rose-400">{(item.confidence * 100).toFixed(0)}%</strong></span>
                                        <span>&bull;</span>
                                        <span>Status: <strong className="text-zinc-300">{item.validation_status}</strong></span>
                                      </div>
                                    </div>
                                  </div>

                                  {/* Actions */}
                                  <div className="flex items-center gap-2">
                                    <button
                                      onClick={() => toggleIncidentExpanded(item.id)}
                                      className="text-xs px-3 py-1.5 rounded-lg border border-zinc-700 bg-zinc-800/80 hover:bg-zinc-700 text-zinc-200 transition-all flex items-center gap-1.5"
                                    >
                                      <span>{isExpanded ? "Hide Observations" : `View Supporting Observations (${item.supporting_observations.length > 0 ? item.supporting_observations.length : item.observations_count})`}</span>
                                      <svg
                                        className={`w-3.5 h-3.5 transition-transform ${isExpanded ? "rotate-180" : ""}`}
                                        fill="none"
                                        stroke="currentColor"
                                        viewBox="0 0 24 24"
                                      >
                                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
                                      </svg>
                                    </button>
                                  </div>
                                </div>

                                {/* Expanded Supporting Observations Drawer */}
                                {isExpanded && (
                                  <div className="mt-4 pt-4 border-t border-zinc-800/80 space-y-2">
                                    <p className="text-xs text-zinc-400 font-semibold mb-2">
                                      Underlying Forensic Observations ({item.supporting_observations.length > 0 ? item.supporting_observations.length : item.observations_count}):
                                    </p>
                                    <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 gap-2.5 max-h-80 overflow-y-auto pr-1">
                                      {item.supporting_observations.map((obs) => (
                                        <div
                                          key={obs.id}
                                          className="p-2.5 rounded-lg border border-zinc-800 bg-zinc-950 flex flex-col justify-between gap-2 text-xs"
                                        >
                                          <div className="flex items-center justify-between">
                                            <span className="font-bold text-zinc-200">
                                              Frame #{obs.supporting_evidence?.frame_index ?? Math.round(obs.timestamp_seconds * 30)}
                                            </span>
                                            <button
                                              onClick={() => handleSeekToTimestamp(obs.timestamp_seconds)}
                                              className="text-[11px] font-mono text-rose-400 hover:underline"
                                            >
                                              {formatTimestamp(obs.timestamp_seconds)}
                                            </button>
                                          </div>
                                          <div className="text-[11px] text-zinc-400 flex items-center justify-between">
                                            <span>Strength: {((obs.evidence_strength || 0) * 100).toFixed(0)}%</span>
                                            <span className="text-zinc-500 font-mono text-[10px]">{obs.detector_name}</span>
                                          </div>
                                          <button
                                            onClick={() => handleSeekToTimestamp(obs.timestamp_seconds)}
                                            className="mt-1 text-[10px] bg-rose-600/20 text-rose-300 hover:bg-rose-600/40 border border-rose-500/30 rounded py-1 text-center transition-colors"
                                          >
                                            Jump to Frame
                                          </button>
                                        </div>
                                      ))}
                                    </div>
                                  </div>
                                )}
                              </div>
                            );
                          })}
                      </div>
                    ) : (
                      /* RAW OBSERVATIONS VIEW */
                      <div className="space-y-3 font-mono">
                        {specializedList
                          .filter((item) => {
                            if (specializedFilter === "all") return true;
                            const cn = (item.class_name || "").toLowerCase();
                            const dn = (item.detector_name || "").toLowerCase();
                            if (specializedFilter === "fire") return cn.includes("fire");
                            if (specializedFilter === "smoke") return cn.includes("smoke");
                            if (specializedFilter === "weapon") return cn.includes("weapon") || cn.includes("knife") || cn.includes("gun");
                            if (specializedFilter === "pose") return cn.includes("pose") || cn.includes("posture") || dn.includes("pose");
                            return true;
                          })
                          .map((item) => {
                            const isAccepted = item.validation_status === "VALID" || item.validation_status === "ACCEPTED";
                            const isRejected = item.validation_status === "REJECTED";
                            const isReview = !isAccepted && !isRejected;

                            return (
                              <div
                                key={item.id}
                                className={`rounded-xl border p-4 transition-all bg-zinc-900/60 ${
                                  isAccepted
                                    ? "border-emerald-500/40 hover:border-emerald-500/60"
                                    : isRejected
                                    ? "border-zinc-800/80 opacity-60 hover:opacity-100"
                                    : "border-amber-500/40 hover:border-amber-500/60"
                                }`}
                              >
                                <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3">
                                  <div className="flex items-center gap-3">
                                    <button
                                      onClick={() => handleSeekToTimestamp(item.timestamp_seconds)}
                                      className="text-xs font-bold bg-black/40 hover:bg-black/70 border border-zinc-700 hover:border-emerald-500 text-emerald-400 px-2.5 py-1.5 rounded-lg transition-all"
                                      title="Jump to video timestamp"
                                    >
                                      {formatTimestamp(item.timestamp_seconds)}
                                    </button>
                                    <div>
                                      <div className="flex items-center gap-2">
                                        <span className="text-sm font-bold text-zinc-100 uppercase">
                                          {item.class_name.replace(/_/g, " ")}
                                        </span>
                                        {/* Validation Status Badge */}
                                        {isAccepted && (
                                          <span className="text-[10px] font-bold px-2 py-0.5 rounded border bg-emerald-500/20 text-emerald-300 border-emerald-500/40">
                                            ACCEPTED
                                          </span>
                                        )}
                                        {isReview && (
                                          <span className="text-[10px] font-bold px-2 py-0.5 rounded border bg-amber-500/20 text-amber-300 border-amber-500/40">
                                            REVIEW REQUIRED
                                          </span>
                                        )}
                                        {isRejected && (
                                          <span className="text-[10px] font-bold px-2 py-0.5 rounded border bg-rose-500/20 text-rose-400 border-rose-500/40">
                                            REJECTED
                                          </span>
                                        )}
                                      </div>
                                      <div className="text-xs text-zinc-400 mt-0.5 flex flex-wrap items-center gap-2">
                                        <span>Detector: <strong className="text-zinc-300">{item.detector_name}</strong></span>
                                        {item.detector_version && <span>({item.detector_version})</span>}
                                        <span>&bull;</span>
                                        <span>Evidence Strength: <strong className="text-rose-400">{(item.evidence_strength * 100).toFixed(0)}%</strong></span>
                                      </div>
                                    </div>
                                  </div>

                                  <div className="flex items-center gap-2 self-end sm:self-center font-mono">
                                    <button
                                      type="button"
                                      onClick={() => handleSeekToTimestamp(item.timestamp_seconds)}
                                      className="text-xs bg-zinc-950/80 hover:bg-emerald-600 hover:text-white text-zinc-300 border border-zinc-700 px-3 py-1.5 rounded transition-colors"
                                    >
                                      [Jump]
                                    </button>
                                    <button
                                      type="button"
                                      onClick={() => handleCaptureEvidence(item.timestamp_seconds, item.event_id, "snapshot_and_clip")}
                                      disabled={capturingEventId !== null}
                                      className="text-xs bg-amber-500/10 hover:bg-amber-500/20 text-amber-300 hover:text-amber-200 border border-amber-500/30 px-3 py-1.5 rounded flex items-center gap-1.5 transition-colors disabled:opacity-50"
                                    >
                                      <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3 9a2 2 0 012-2h.93a2 2 0 001.664-.89l.812-1.22A2 2 0 0110.07 4h3.86a2 2 0 011.664.89l.812 1.22A2 2 0 0018.07 7H19a2 2 0 012 2v9a2 2 0 01-2 2H5a2 2 0 01-2-2V9z" />
                                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 13a3 3 0 11-6 0 3 3 0 016 0z" />
                                      </svg>
                                      <span>Capture Evidence</span>
                                    </button>
                                  </div>
                                </div>

                                {/* Supporting Evidence Details */}
                                {item.supporting_evidence && Object.keys(item.supporting_evidence).length > 0 && (
                                  <div className="mt-3 pt-3 border-t border-zinc-800/80 text-[11px] text-zinc-400 bg-zinc-950/40 p-2.5 rounded-lg space-y-1">
                                    <span className="text-zinc-500 font-bold block uppercase tracking-wider text-[10px]">
                                      Observational Provenance & Supporting Signals
                                    </span>
                                    <div className="flex flex-wrap gap-x-4 gap-y-1">
                                      {Object.entries(item.supporting_evidence).map(([k, v]) => (
                                        <span key={k}>
                                          <span className="text-zinc-500">{k}:</span>{" "}
                                          <span className="text-zinc-300">{typeof v === "number" ? v.toFixed(3) : String(v)}</span>
                                        </span>
                                      ))}
                                      {item.bbox && (
                                        <span>
                                          <span className="text-zinc-500">Region:</span>{" "}
                                          <span className="text-zinc-300">[{item.bbox.x1.toFixed(0)}, {item.bbox.y1.toFixed(0)} &rarr; {item.bbox.x2.toFixed(0)}, {item.bbox.y2.toFixed(0)}]</span>
                                        </span>
                                      )}
                                    </div>
                                  </div>
                                )}
                              </div>
                            );
                          })}
                      </div>
                    )}
                  </div>
                )}

                {/* ------------------------------------------------------------- */}
                {/* Global Reliability: Diagnostics & Detector Health View       */}
                {/* ------------------------------------------------------------- */}
                {activeView === "diagnostics" && (
                  <div className="space-y-6">
                    {/* Header Banner */}
                    <div className="bg-gradient-to-r from-teal-950/40 via-zinc-900 to-zinc-900 border border-teal-500/30 rounded-xl p-5 flex flex-col sm:flex-row sm:items-center justify-between gap-4">
                      <div className="space-y-1">
                        <div className="flex items-center gap-2">
                          <span className="h-2.5 w-2.5 rounded-full bg-teal-500 animate-pulse" />
                          <h3 className="text-sm font-mono font-bold text-teal-400 uppercase tracking-wide">
                            Sentinel Detector Health & Diagnostic Telemetry
                          </h3>
                        </div>
                        <p className="text-xs text-zinc-400 font-sans max-w-2xl leading-relaxed">
                          Authoritative operational health, model provenance, and validation status across all object detectors, specialized visual models, and incident inference engines.
                        </p>
                      </div>
                      {uploadResult?.video_id && (
                        <button
                          onClick={() => fetchHealth(uploadResult.video_id)}
                          disabled={isFetchingHealth}
                          className="bg-zinc-800 hover:bg-zinc-700 text-zinc-200 text-xs font-mono px-3.5 py-2 rounded-lg border border-zinc-700 flex items-center gap-1.5 shrink-0 transition-colors"
                        >
                          <svg className={`w-3.5 h-3.5 ${isFetchingHealth ? "animate-spin" : ""}`} fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
                          </svg>
                          <span>{isFetchingHealth ? "Polling Health..." : "Refresh Health"}</span>
                        </button>
                      )}
                    </div>

                    {/* Summary Metric Cards */}
                    <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
                      <div className="bg-zinc-900/70 border border-zinc-800 p-3.5 rounded-xl">
                        <span className="text-[11px] font-mono text-zinc-400 uppercase block">Total Detectors</span>
                        <span className="text-xl font-mono font-bold text-zinc-100">
                          {detectorHealth?.health?.total_detectors ?? 0}
                        </span>
                      </div>
                      <div className="bg-zinc-900/70 border border-zinc-800 p-3.5 rounded-xl">
                        <span className="text-[11px] font-mono text-zinc-400 uppercase block">Operational</span>
                        <span className="text-xl font-mono font-bold text-emerald-400">
                          {detectorHealth?.health?.healthy_detectors ?? 0} Active
                        </span>
                      </div>
                      <div className="bg-zinc-900/70 border border-zinc-800 p-3.5 rounded-xl">
                        <span className="text-[11px] font-mono text-zinc-400 uppercase block">Total Observations</span>
                        <span className="text-xl font-mono font-bold text-blue-400">
                          {detectorHealth?.health?.total_observations ?? 0}
                        </span>
                        <span className="text-[10px] text-zinc-500 block">
                          {detectorHealth?.health?.total_validated ?? 0} Validated • {detectorHealth?.health?.total_rejected ?? 0} Rejected
                        </span>
                      </div>
                      <div className="bg-zinc-900/70 border border-zinc-800 p-3.5 rounded-xl">
                        <span className="text-[11px] font-mono text-zinc-400 uppercase block">System Failures</span>
                        <span className={`text-xl font-mono font-bold ${
                          (detectorHealth?.health?.total_failures ?? 0) > 0 ? "text-rose-400" : "text-emerald-400"
                        }`}>
                          {detectorHealth?.health?.total_failures ?? 0}
                        </span>
                      </div>
                    </div>

                    {/* Detectors Table */}
                    <div className="bg-zinc-900/50 border border-zinc-800 rounded-xl overflow-hidden">
                      <div className="p-3 bg-zinc-900 border-b border-zinc-800 flex items-center justify-between">
                        <h4 className="text-xs font-mono font-bold text-zinc-300 uppercase tracking-wider">
                          Registered Detector Registry & Provenance
                        </h4>
                        <span className="text-[10px] font-mono text-zinc-500">
                          Canonical Contract v2.0
                        </span>
                      </div>
                      <div className="overflow-x-auto">
                        <table className="w-full text-left font-mono text-xs">
                          <thead className="bg-zinc-950/60 text-zinc-400 border-b border-zinc-800/80">
                            <tr>
                              <th className="p-3">Detector Name</th>
                              <th className="p-3">Version</th>
                              <th className="p-3">Status</th>
                              <th className="p-3">Inference Engine</th>
                              <th className="p-3">Observations</th>
                              <th className="p-3">Failures</th>
                            </tr>
                          </thead>
                          <tbody className="divide-y divide-zinc-800/50">
                            {!detectorHealth?.health?.detectors || detectorHealth.health.detectors.length === 0 ? (
                              <tr>
                                <td colSpan={6} className="p-6 text-center text-zinc-500">
                                  No detector telemetry available for this session. Run video analysis to initialize detector health.
                                </td>
                              </tr>
                            ) : (
                              detectorHealth.health.detectors.map((det) => (
                                <tr key={det.name} className="hover:bg-zinc-900/40 transition-colors">
                                  <td className="p-3 font-semibold text-zinc-200">
                                    {det.name}
                                  </td>
                                  <td className="p-3 text-zinc-400">
                                    {det.version}
                                  </td>
                                  <td className="p-3">
                                    <span className={`px-2 py-0.5 rounded text-[10px] font-bold border ${
                                      det.status === "AVAILABLE"
                                        ? "bg-emerald-500/10 text-emerald-300 border-emerald-500/30"
                                        : det.status === "FAILED"
                                        ? "bg-rose-500/10 text-rose-400 border-rose-500/30"
                                        : "bg-zinc-800 text-zinc-400 border-zinc-700"
                                    }`}>
                                      {det.status}
                                    </span>
                                  </td>
                                  <td className="p-3">
                                    <span className="px-2 py-0.5 rounded text-[10px] font-medium bg-zinc-800/80 text-zinc-300 border border-zinc-700">
                                      {det.model_type}
                                    </span>
                                  </td>
                                  <td className="p-3">
                                    <span className="text-zinc-200">{det.observations}</span>{" "}
                                    <span className="text-zinc-500 text-[10px]">
                                      ({det.validated_observations} val / {det.rejected_observations} rej)
                                    </span>
                                  </td>
                                  <td className="p-3">
                                    <span className={det.failures > 0 ? "text-rose-400 font-bold" : "text-zinc-500"}>
                                      {det.failures}
                                    </span>
                                    {det.error_message && (
                                      <span className="block text-[10px] text-rose-400 truncate max-w-xs" title={det.error_message}>
                                        {det.error_message}
                                      </span>
                                    )}
                                  </td>
                                </tr>
                              ))
                            )}
                          </tbody>
                        </table>
                      </div>
                    </div>

                    {/* Architectural Contract Note */}
                    <div className="bg-zinc-950/70 border border-zinc-800/80 p-4 rounded-xl text-xs space-y-2">
                      <div className="flex items-center gap-2 text-teal-400 font-mono font-semibold">
                        <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                        </svg>
                        <span>Sentinel Universal Reliability Contract Invariants</span>
                      </div>
                      <p className="text-zinc-400 leading-relaxed text-[11px]">
                        Every observation adheres to strict invariants: finite non-negative timestamps, normalized clamped bounding box geometry, model class registry isolation, and class-aware duplicate suppression. Tracks undergo multi-state lifecycle management (Tentative &rarr; Confirmed &rarr; Coasting &rarr; Lost/Ended). Incidents require multi-signal grounding and negative evidence arbitration.
                      </p>
                    </div>
                  </div>
                )}
              </div>
            )}
          </div>
        </div>
      )}

                {/* ── Multi-Camera Sessions ── */}
                {activeView === "multicamera" && (
                  <div className="p-2">
                    <MultiCameraSessionPanel />
                  </div>
                )}

      {/* EVIDENCE PREVIEW MODAL */}
      {selectedEvidence && (
        <div className="fixed inset-0 z-50 bg-black/80 backdrop-blur-sm flex items-center justify-center p-4">
          <div className="bg-zinc-950 border border-zinc-800 rounded-2xl max-w-4xl w-full max-h-[90vh] overflow-y-auto p-6 space-y-4 shadow-2xl animate-in zoom-in-95 duration-150">
            {/* Modal Header */}
            <div className="flex items-center justify-between border-b border-zinc-800 pb-3">
              <div className="flex items-center gap-3">
                <span className="font-mono text-xs font-bold text-amber-400 bg-amber-500/10 border border-amber-500/30 px-2.5 py-1 rounded">
                  #EV-{selectedEvidence.evidence_id.slice(-6).toUpperCase()}
                </span>
                <h3 className="text-base font-bold text-zinc-100">
                  Forensic Evidence Preview
                </h3>
                <span className="text-xs font-mono text-zinc-400">
                  {formatTimestamp(selectedEvidence.timestamp)}
                </span>
              </div>
              <button
                onClick={() => setSelectedEvidence(null)}
                className="text-zinc-400 hover:text-white p-1 rounded-lg hover:bg-zinc-900 transition-colors text-lg leading-none"
              >
                ✕
              </button>
            </div>

            {/* Mode Switcher */}
            <div className="flex items-center gap-2 font-mono text-xs border-b border-zinc-800 pb-3">
              {selectedEvidence.has_snapshot && (
                <button
                  onClick={() => setEvidenceModalMode("snapshot")}
                  className={`px-3 py-1.5 rounded transition-colors ${
                    evidenceModalMode === "snapshot"
                      ? "bg-emerald-600 text-white font-semibold"
                      : "bg-zinc-900 text-zinc-400 hover:text-zinc-200"
                  }`}
                >
                  📷 Original Snapshot
                </button>
              )}
              {selectedEvidence.has_annotated && (
                <button
                  onClick={() => setEvidenceModalMode("annotated")}
                  className={`px-3 py-1.5 rounded transition-colors ${
                    evidenceModalMode === "annotated"
                      ? "bg-emerald-600 text-white font-semibold"
                      : "bg-zinc-900 text-zinc-400 hover:text-zinc-200"
                  }`}
                >
                  🎯 Annotated BBox Overlay
                </button>
              )}
              {selectedEvidence.has_clip && (
                <button
                  onClick={() => setEvidenceModalMode("clip")}
                  className={`px-3 py-1.5 rounded transition-colors ${
                    evidenceModalMode === "clip"
                      ? "bg-amber-600 text-white font-semibold"
                      : "bg-zinc-900 text-zinc-400 hover:text-zinc-200"
                  }`}
                >
                  🎬 Evidence Clip ({selectedEvidence.duration_seconds || 6}s)
                </button>
              )}
            </div>

            {/* Media Display Area */}
            <div className="relative rounded-xl overflow-hidden bg-black border border-zinc-800 aspect-video flex items-center justify-center">
              {evidenceModalMode === "snapshot" && selectedEvidence.has_snapshot && (
                <img
                  src={getEvidenceSnapshotUrl(selectedEvidence.evidence_id)}
                  alt="Evidence Snapshot"
                  className="w-full h-full object-contain"
                />
              )}
              {evidenceModalMode === "annotated" && selectedEvidence.has_annotated && (
                <img
                  src={getEvidenceAnnotatedUrl(selectedEvidence.evidence_id)}
                  alt="Annotated Evidence Snapshot"
                  className="w-full h-full object-contain"
                />
              )}
              {evidenceModalMode === "clip" && selectedEvidence.has_clip && (
                <video
                  controls
                  autoPlay
                  src={getEvidencePlaybackUrl(selectedEvidence.evidence_id)}
                  className="w-full h-full object-contain"
                >
                  Your browser does not support HTML5 video tag.
                </video>
              )}
            </div>

            {/* Forensic Provenance Details */}
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 bg-zinc-900/60 p-4 rounded-xl border border-zinc-800 text-xs font-mono">
              <div>
                <span className="text-zinc-500 block">Object Class</span>
                <span className="text-zinc-100 font-bold font-sans">
                  {selectedEvidence.object_class ? formatClassName(selectedEvidence.object_class) : "Event"}
                </span>
              </div>
              <div>
                <span className="text-zinc-500 block">Confidence</span>
                <span className="text-emerald-400 font-bold">
                  {selectedEvidence.confidence != null ? `${Math.round(selectedEvidence.confidence * 100)}%` : "N/A"}
                </span>
              </div>
              <div>
                <span className="text-zinc-500 block">Source File</span>
                <span className="text-zinc-200 truncate block font-sans" title={selectedEvidence.source_video_name}>
                  {selectedEvidence.source_video_name}
                </span>
              </div>
              <div>
                <span className="text-zinc-500 block">Preserved At</span>
                <span className="text-zinc-400 block">
                  {selectedEvidence.created_at ? new Date(selectedEvidence.created_at).toLocaleString() : "Persisted"}
                </span>
              </div>
            </div>

            {/* Modal Actions */}
            <div className="flex items-center justify-between pt-2">
              <button
                onClick={() => {
                  handleSeekToTimestamp(selectedEvidence.timestamp, selectedEvidence.event_id || undefined);
                  setSelectedEvidence(null);
                }}
                className="text-xs font-mono bg-emerald-600 hover:bg-emerald-500 text-white px-4 py-2 rounded-lg transition-colors flex items-center gap-2"
              >
                <span>Seek Surveillance Player to {formatTimestamp(selectedEvidence.timestamp)}</span>
              </button>
              <button
                onClick={() => setSelectedEvidence(null)}
                className="text-xs font-mono text-zinc-400 hover:text-zinc-200 border border-zinc-800 hover:border-zinc-700 px-4 py-2 rounded-lg transition-colors"
              >
                Close
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Create Case from Video Modal */}
      {isCreateCaseModalOpen && uploadResult && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/80 backdrop-blur-sm p-4">
          <div className="bg-zinc-900 border border-zinc-800 rounded-xl max-w-md w-full p-6 space-y-4 shadow-2xl">
            <div className="flex items-center justify-between border-b border-zinc-800 pb-3">
              <h3 className="text-base font-bold text-white flex items-center gap-2">
                <span>📁</span> Create Case from Video
              </h3>
              <button
                onClick={() => setIsCreateCaseModalOpen(false)}
                className="text-zinc-500 hover:text-zinc-300"
              >
                ✕
              </button>
            </div>

            {caseWorkflowFeedback && (
              <div className={`p-3 rounded text-xs ${
                caseWorkflowFeedback.startsWith("Error")
                  ? "bg-red-500/10 border border-red-500/30 text-red-400"
                  : "bg-emerald-500/10 border border-emerald-500/30 text-emerald-400"
              }`}>
                {caseWorkflowFeedback}
              </div>
            )}

            <form onSubmit={handleCreateCaseFromThisVideo} className="space-y-4 text-sm">
              <div>
                <label className="block text-zinc-300 font-medium mb-1">Case Title *</label>
                <input
                  type="text"
                  required
                  value={caseTitleInput}
                  onChange={(e) => setCaseTitleInput(e.target.value)}
                  className="w-full bg-zinc-950 border border-zinc-800 rounded-lg p-2.5 text-zinc-200 text-xs focus:outline-none focus:border-emerald-500"
                />
              </div>

              <div>
                <label className="block text-zinc-300 font-medium mb-1">Priority</label>
                <select
                  value={casePriorityInput}
                  onChange={(e: any) => setCasePriorityInput(e.target.value)}
                  className="w-full bg-zinc-950 border border-zinc-800 rounded-lg p-2.5 text-zinc-200 text-xs focus:outline-none focus:border-emerald-500 font-mono"
                >
                  <option value="LOW">LOW</option>
                  <option value="MEDIUM">MEDIUM</option>
                  <option value="HIGH">HIGH</option>
                  <option value="CRITICAL">CRITICAL</option>
                </select>
              </div>

              <div>
                <label className="block text-zinc-300 font-medium mb-1">Case Synopsis</label>
                <textarea
                  rows={3}
                  value={caseDescInput}
                  onChange={(e) => setCaseDescInput(e.target.value)}
                  className="w-full bg-zinc-950 border border-zinc-800 rounded-lg p-2.5 text-zinc-200 text-xs focus:outline-none focus:border-emerald-500 resize-none"
                />
              </div>

              <div className="bg-zinc-950 p-2.5 rounded-lg border border-zinc-800/80 text-[11px] font-mono text-zinc-400 space-y-1">
                <div>Source Video: <span className="text-zinc-200">{uploadResult.filename}</span></div>
                <div>Status: <span className="text-emerald-400 font-bold">Auto-links to new case workspace</span></div>
              </div>

              <div className="flex justify-end gap-3 pt-2">
                <button
                  type="button"
                  onClick={() => setIsCreateCaseModalOpen(false)}
                  className="px-4 py-2 bg-zinc-800 hover:bg-zinc-700 text-zinc-300 rounded-lg text-xs font-medium"
                >
                  Cancel
                </button>
                <button
                  type="submit"
                  disabled={isCaseSubmitting}
                  className="px-4 py-2 bg-emerald-600 hover:bg-emerald-500 text-zinc-950 font-bold rounded-lg text-xs transition-all disabled:opacity-50"
                >
                  {isCaseSubmitting ? "Creating & Linking..." : "Create & Open Case"}
                </button>
              </div>
            </form>
          </div>
        </div>
      )}
    </div>
  );
}


