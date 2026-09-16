/**
 * Sentinel Frontend API Client
 * Interfaces with the Sentinel FastAPI backend service.
 */

export interface UploadVideoResponse {
  video_id: string;
  filename: string;
  status: string;
  message: string;
  file_size?: number;
  stream_url?: string;
  playback_url?: string;
}

export interface VideoMetadata {
  video_id: string;
  filename: string;
  saved_filename?: string;
  file_size_bytes?: number;
  content_type?: string;
  status: string;
  uploaded_at?: string;
  processed_at?: string;
  duration_seconds?: number;
  fps?: number;
  frames_processed?: number;
  detections_count?: number;
  raw_detections_count?: number;
  grouped_events_count?: number;
  stream_url?: string;
  playback_url?: string;
}

export interface VideoPlaybackStatusResponse {
  video_id: string;
  status: "ready" | "converting" | "needs_conversion" | "unavailable";
  is_compatible: boolean;
  is_transcoded: boolean;
  message: string;
}

export interface HealthResponse {
  status: string;
  service: string;
  version: string;
  environment: string;
  message: string;
}

// ---------------------------------------------------------------------------
// Phase 3: Video Intelligence Types
// ---------------------------------------------------------------------------

export interface BoundingBox {
  x1: number;
  y1: number;
  x2: number;
  y2: number;
}

export interface DetectionEvent {
  event_id: string;
  video_id: string;
  event_type: string;
  object_class?: string | null;
  class_id?: number | null;
  timestamp: number;
  confidence: number;
  bounding_box: BoundingBox;
  frame_number?: number;
  video_filename?: string;
  fps?: number;
  video_duration?: number;
  validation_status?: string;
  validation_reason?: string | null;
}

export interface ProcessVideoResponse {
  video_id: string;
  status: string;
  duration_seconds: number;
  fps: number;
  frames_processed: number;
  detections_count: number;
  raw_detections_count?: number;
  grouped_events_count?: number;
}

export interface VideoEventsResponse {
  video_id: string;
  total_events: number;
  raw_events_count?: number;
  validated_events_count?: number;
  rejected_events_count?: number;
  filters: {
    object_class: string | null;
    min_confidence: number | null;
    include_unvalidated?: boolean;
  };
  events: DetectionEvent[];
}

// ---------------------------------------------------------------------------
// Phase 4: Event Intelligence & Timeline Types
// ---------------------------------------------------------------------------

export interface ObjectSummary {
  class: string;
  count: number;
}

export interface GroupedEvent {
  event_id: string;
  video_id: string;
  event_type: string;
  start_time: number;
  end_time: number;
  duration_seconds: number;
  objects: ObjectSummary[];
  total_detections: number;
  max_confidence: number;
  priority: "LOW" | "NORMAL" | "HIGH";
}

export interface VideoTimelineResponse {
  video_id: string;
  total_events: number;
  filters: {
    object_class: string | null;
    min_confidence: number | null;
    start_time: number | null;
    end_time: number | null;
  };
  events: GroupedEvent[];
}

// ---------------------------------------------------------------------------
// API client configuration
// ---------------------------------------------------------------------------

export const API_BASE_URL = (
  process.env.NEXT_PUBLIC_API_URL || "http://127.0.0.1:8000"
).replace(/\/+$/, "");

/**
 * Safely format error response details into a human-readable string.
 * Prevents [object Object] serialization when FastAPI returns validation error arrays or dictionaries.
 */
export function extractErrorMessage(err: any, fallback: string): string {
  if (!err) return fallback;
  if (typeof err === "string") return err;
  if (err.detail !== undefined && err.detail !== null) {
    if (typeof err.detail === "string") return err.detail;
    if (Array.isArray(err.detail)) {
      return err.detail
        .map((d: any) => {
          if (typeof d === "string") return d;
          if (d?.msg) {
            const field = Array.isArray(d.loc)
              ? d.loc.filter((x: any) => x !== "body" && x !== "query").join(".")
              : "";
            return field ? `${field}: ${d.msg}` : d.msg;
          }
          return JSON.stringify(d);
        })
        .join("; ");
    }
    if (typeof err.detail === "object") {
      if (err.detail.message) return String(err.detail.message);
      return JSON.stringify(err.detail);
    }
  }
  if (err.message && typeof err.message === "string") return err.message;
  return fallback;
}

/**
 * Get the full stream URL for an uploaded video by ID.
 */
export function getVideoStreamUrl(videoId: string): string {
  return `${API_BASE_URL}/api/videos/${encodeURIComponent(videoId)}/stream`;
}

/**
 * Get the full browser-compatible playback URL for an uploaded video by ID.
 * Supports RFC 7233 HTTP Range requests for instant scrubbable HTML5 seeking.
 */
export function getVideoPlaybackUrl(videoId: string): string {
  return `${API_BASE_URL}/api/videos/${encodeURIComponent(videoId)}/playback`;
}

/**
 * Fetch playback status for an uploaded video.
 */
export async function getVideoPlaybackStatus(
  videoId: string
): Promise<VideoPlaybackStatusResponse> {
  const response = await fetch(
    `${API_BASE_URL}/api/videos/${encodeURIComponent(videoId)}/playback-status`
  );
  if (!response.ok) {
    throw new Error(`Failed to fetch playback status: ${response.status}`);
  }
  return response.json();
}

/**
 * Fetch metadata for an existing video by ID.
 */
export async function getVideoMetadata(videoId: string): Promise<VideoMetadata> {
  const response = await fetch(
    `${API_BASE_URL}/api/videos/${encodeURIComponent(videoId)}`
  );
  if (!response.ok) {
    throw new Error(`Failed to load video metadata: ${response.status}`);
  }
  return response.json();
}

/**
 * Fetch health status of the backend API.
 */
export async function checkApiHealth(): Promise<HealthResponse> {
  const response = await fetch(`${API_BASE_URL}/api/health`);
  if (!response.ok) {
    throw new Error(`Health check failed with status: ${response.status}`);
  }
  return response.json();
}

/**
 * Uploads a video file using XMLHttpRequest to provide upload progress events.
 *
 * @param file The video file to upload.
 * @param onProgress Callback receiving the progress percentage (0 - 100).
 * @returns Promise resolving to the UploadVideoResponse.
 */
export function uploadVideo(
  file: File,
  onProgress?: (percent: number) => void
): Promise<UploadVideoResponse> {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    const formData = new FormData();
    formData.append("file", file, file.name);

    xhr.open("POST", `${API_BASE_URL}/api/videos/upload`);

    if (xhr.upload && onProgress) {
      xhr.upload.onprogress = (event) => {
        if (event.lengthComputable) {
          const percent = Math.round((event.loaded / event.total) * 100);
          onProgress(percent);
        }
      };
    }

    xhr.onload = () => {
      if (xhr.status >= 200 && xhr.status < 300) {
        try {
          const data: UploadVideoResponse = JSON.parse(xhr.responseText);
          resolve(data);
        } catch {
          reject(new Error("Failed to parse server upload response."));
        }
      } else {
        let errorMessage = `Upload failed with status code ${xhr.status}`;
        try {
          const errorJson = JSON.parse(xhr.responseText);
          errorMessage = extractErrorMessage(errorJson, errorMessage);
        } catch {
          if (xhr.statusText) {
            errorMessage = `${errorMessage}: ${xhr.statusText}`;
          }
        }
        reject(new Error(errorMessage));
      }
    };

    xhr.onerror = () => {
      reject(
        new Error(
          `Unable to connect to Sentinel backend at ${API_BASE_URL}. Ensure the backend server is running.`
        )
      );
    };

    xhr.ontimeout = () => {
      reject(new Error("Video upload timed out. Please try again."));
    };

    xhr.send(formData);
  });
}

/**
 * Retrieve metadata for a specific video ID.
 */
export async function getVideoInfo(videoId: string): Promise<VideoMetadata> {
  const response = await fetch(`${API_BASE_URL}/api/videos/${encodeURIComponent(videoId)}`);
  if (!response.ok) {
    let errorMsg = `Failed to get video info (Status ${response.status})`;
    try {
      const err = await response.json();
      errorMsg = extractErrorMessage(err, errorMsg);
    } catch {
      // ignore
    }
    throw new Error(errorMsg);
  }
  return response.json();
}

// ---------------------------------------------------------------------------
// Phase 3 & 4: Video Processing & Timeline API
// ---------------------------------------------------------------------------

/**
 * Trigger OpenCV + YOLO processing pipeline for an uploaded video.
 * Synchronous on the backend for MVP; runs the full pipeline and returns a summary.
 */
export async function processVideo(videoId: string): Promise<ProcessVideoResponse> {
  const response = await fetch(
    `${API_BASE_URL}/api/videos/${encodeURIComponent(videoId)}/process`,
    { method: "POST" }
  );
  if (!response.ok) {
    let errorMsg = `Processing failed (Status ${response.status})`;
    try {
      const err = await response.json();
      errorMsg = extractErrorMessage(err, errorMsg);
    } catch {
      // ignore
    }
    throw new Error(errorMsg);
  }
  return response.json();
}

/**
 * Retrieve raw detection events for a processed video with optional filters.
 */
export async function getVideoEvents(
  videoId: string,
  params?: { object_class?: string; min_confidence?: number; include_unvalidated?: boolean }
): Promise<VideoEventsResponse> {
  const url = new URL(`${API_BASE_URL}/api/videos/${encodeURIComponent(videoId)}/events`);
  if (params?.object_class) url.searchParams.set("object_class", params.object_class);
  if (params?.min_confidence !== undefined) {
    url.searchParams.set("min_confidence", String(params.min_confidence));
  }
  if (params?.include_unvalidated) {
    url.searchParams.set("include_unvalidated", "true");
  }

  const response = await fetch(url.toString());
  if (!response.ok) {
    let errorMsg = `Failed to retrieve events (Status ${response.status})`;
    try {
      const err = await response.json();
      errorMsg = extractErrorMessage(err, errorMsg);
    } catch {
      // ignore
    }
    throw new Error(errorMsg);
  }
  return response.json();
}

/**
 * Retrieve grouped investigation events for the video timeline with optional filters.
 */
export async function getVideoTimeline(
  videoId: string,
  params?: {
    object_class?: string;
    min_confidence?: number;
    start_time?: number;
    end_time?: number;
  }
): Promise<VideoTimelineResponse> {
  const url = new URL(`${API_BASE_URL}/api/videos/${encodeURIComponent(videoId)}/timeline`);
  if (params?.object_class) url.searchParams.set("object_class", params.object_class);
  if (params?.min_confidence !== undefined) {
    url.searchParams.set("min_confidence", String(params.min_confidence));
  }
  if (params?.start_time !== undefined) {
    url.searchParams.set("start_time", String(params.start_time));
  }
  if (params?.end_time !== undefined) {
    url.searchParams.set("end_time", String(params.end_time));
  }

  const response = await fetch(url.toString());
  if (!response.ok) {
    let errorMsg = `Failed to retrieve timeline (Status ${response.status})`;
    try {
      const err = await response.json();
      errorMsg = extractErrorMessage(err, errorMsg);
    } catch {
      // ignore
    }
    throw new Error(errorMsg);
  }
  return response.json();
}

// ---------------------------------------------------------------------------
// Phase 5A: Natural-Language Investigation Types & API
// ---------------------------------------------------------------------------

export interface InvestigationResultItem {
  event_id: string;
  video_id: string;
  timestamp?: number;
  start_time?: number;
  end_time?: number;
  duration_seconds?: number;
  object_class?: string | null;
  confidence?: number;
  max_confidence?: number;
  frame_number?: number;
  bounding_box?: BoundingBox;
  event_type?: string | null;
  objects?: ObjectSummary[];
  total_detections?: number;
  priority?: "LOW" | "NORMAL" | "HIGH";
}

export interface InvestigationResponse {
  query: string;
  video_id: string;
  is_supported: boolean;
  result_type: "detections" | "events" | "count" | "unsupported" | "error";
  interpreted_filters: {
    object_class?: string | null;
    start_time?: number | null;
    end_time?: number | null;
    min_confidence?: number | null;
    max_confidence?: number | null;
  };
  count: number;
  message: string;
  results: InvestigationResultItem[];
}

/**
 * Execute natural language investigation query against video detections.
 */
export async function investigateVideo(
  videoId: string,
  query: string
): Promise<InvestigationResponse> {
  const response = await fetch(
    `${API_BASE_URL}/api/videos/${encodeURIComponent(videoId)}/investigate`,
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query }),
    }
  );

  if (!response.ok) {
    let errorMsg = `Investigation failed (Status ${response.status})`;
    try {
      const err = await response.json();
      errorMsg = extractErrorMessage(err, errorMsg);
    } catch {
      // ignore
    }
    throw new Error(errorMsg);
  }
  return response.json();
}

// ---------------------------------------------------------------------------
// Phase 7: LLM-Assisted Video Investigation Types and API
// ---------------------------------------------------------------------------

export interface AIInvestigatePayload {
  query: string;
  history?: Array<{ role: string; content: string }>;
}

export interface AISourceDetection {
  event_id: string;
  timestamp: number;
  object_class?: string | null;
  event_type?: string | null;
  confidence: number;
  bounding_box?: BoundingBox;
}

export interface AISourceEvent {
  event_id: string;
  event_type: string;
  start_time: number;
  end_time: number;
  total_detections: number;
}

export interface AISourceEvidence {
  evidence_id: string;
  event_id?: string | null;
  timestamp: number;
  evidence_type: string;
  object_class?: string | null;
  confidence?: number | null;
  has_snapshot: boolean;
  has_annotated: boolean;
  has_clip: boolean;
  duration_seconds?: number | null;
  start_time?: number | null;
  end_time?: number | null;
}

export interface VideoSummaryData {
  duration_seconds?: number;
  total_detections?: number;
  detected_classes?: Record<string, number>;
  total_events?: number;
  peak_window?: string;
  total_evidence?: number;
}

export interface AIInvestigateResponse {
  video_id: string;
  query: string;
  mode: "ai_assisted" | "deterministic_fallback" | "guardrail_enforced";
  is_supported: boolean;
  answer: string;
  count?: number;
  structured_query?: Record<string, unknown>;
  summary?: VideoSummaryData;
  sources: {
    detections: AISourceDetection[];
    events: AISourceEvent[];
    evidence: AISourceEvidence[];
  };
  limitations: string[];
}

/**
 * Execute LLM-assisted, evidence-grounded inquiry against Sentinel database.
 */
export async function aiInvestigateVideo(
  videoId: string,
  payload: AIInvestigatePayload
): Promise<AIInvestigateResponse> {
  const response = await fetch(
    `${API_BASE_URL}/api/videos/${encodeURIComponent(videoId)}/ai-investigate`,
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    }
  );

  if (!response.ok) {
    let errorMsg = `AI Investigation failed (Status ${response.status})`;
    try {
      const err = await response.json();
      errorMsg = extractErrorMessage(err, errorMsg);
    } catch {
      // ignore
    }
    throw new Error(errorMsg);
  }
  return response.json();
}

// ---------------------------------------------------------------------------
// Phase 6: Evidence Extraction & Vault Types and API
// ---------------------------------------------------------------------------

export interface EvidenceItem {
  evidence_id: string;
  video_id: string;
  event_id?: string | null;
  evidence_type: "snapshot_and_clip" | "snapshot_only" | "clip_only";
  timestamp: number;
  source_video_name: string;
  object_class?: string | null;
  confidence?: number | null;
  bounding_box?: BoundingBox | null;
  start_time?: number | null;
  end_time?: number | null;
  duration_seconds?: number | null;
  has_snapshot: boolean;
  has_annotated: boolean;
  has_clip: boolean;
  snapshot_url?: string | null;
  annotated_snapshot_url?: string | null;
  clip_url?: string | null;
  playback_url?: string | null;
  created_at?: string | null;
  is_duplicate?: boolean;
}

export interface CreateEvidencePayload {
  timestamp: number;
  event_id?: string | null;
  evidence_type?: "snapshot_and_clip" | "snapshot_only" | "clip_only";
  pre_seconds?: number;
  post_seconds?: number;
  notes?: string;
}

export interface ListEvidenceResponse {
  video_id: string;
  total_evidence: number;
  evidence: EvidenceItem[];
}

export function getEvidenceSnapshotUrl(evidenceId: string): string {
  return `${API_BASE_URL}/api/evidence/${encodeURIComponent(evidenceId)}/snapshot`;
}

export function getEvidenceAnnotatedUrl(evidenceId: string): string {
  return `${API_BASE_URL}/api/evidence/${encodeURIComponent(evidenceId)}/annotated`;
}

export function getEvidenceClipUrl(evidenceId: string): string {
  return `${API_BASE_URL}/api/evidence/${encodeURIComponent(evidenceId)}/clip`;
}

export function getEvidencePlaybackUrl(evidenceId: string): string {
  return `${API_BASE_URL}/api/evidence/${encodeURIComponent(evidenceId)}/playback`;
}

export async function getEvidencePlaybackStatus(evidenceId: string): Promise<{
  evidence_id: string;
  status: "ready" | "preparing" | "needs_conversion" | "failed";
  is_compatible: boolean;
  is_transcoded: boolean;
  message: string;
}> {
  const response = await fetch(
    `${API_BASE_URL}/api/evidence/${encodeURIComponent(evidenceId)}/playback-status`
  );
  if (!response.ok) {
    throw new Error(`Failed to fetch evidence playback status: ${response.status}`);
  }
  return response.json();
}

/**
 * Capture evidence (snapshot and/or clip) for a video detection or event.
 */
export async function createEvidence(
  videoId: string,
  payload: CreateEvidencePayload
): Promise<EvidenceItem> {
  const response = await fetch(
    `${API_BASE_URL}/api/videos/${encodeURIComponent(videoId)}/evidence`,
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    }
  );

  if (!response.ok) {
    let errorMsg = `Evidence capture failed (Status ${response.status})`;
    try {
      const err = await response.json();
      errorMsg = extractErrorMessage(err, errorMsg);
    } catch {
      // ignore
    }
    throw new Error(errorMsg);
  }
  return response.json();
}

/**
 * Retrieve all evidence records for a given video.
 */
export async function getVideoEvidence(videoId: string): Promise<ListEvidenceResponse> {
  const response = await fetch(
    `${API_BASE_URL}/api/videos/${encodeURIComponent(videoId)}/evidence`
  );

  if (!response.ok) {
    let errorMsg = `Failed to retrieve evidence list (Status ${response.status})`;
    try {
      const err = await response.json();
      errorMsg = extractErrorMessage(err, errorMsg);
    } catch {
      // ignore
    }
    throw new Error(errorMsg);
  }
  return response.json();
}

/**
 * Retrieve metadata for a single evidence item.
 */
export async function getEvidenceItem(evidenceId: string): Promise<EvidenceItem> {
  const response = await fetch(
    `${API_BASE_URL}/api/evidence/${encodeURIComponent(evidenceId)}`
  );

  if (!response.ok) {
    let errorMsg = `Failed to retrieve evidence item (Status ${response.status})`;
    try {
      const err = await response.json();
      errorMsg = extractErrorMessage(err, errorMsg);
    } catch {
      // ignore
    }
    throw new Error(errorMsg);
  }
  return response.json();
}

// ---------------------------------------------------------------------------
// Phase 8: Advanced Security Intelligence Types & Functions
// ---------------------------------------------------------------------------

export interface TrackedObjectItem {
  track_id: string;
  object_class?: string | null;
  first_seen: number;
  last_seen: number;
  duration_seconds: number;
  detection_count: number;
  max_confidence: number;
  color?: string | null;
  color_confidence?: number | null;
  active: boolean;
}

export interface VehicleAttributeItem {
  id: string;
  track_id?: string | null;
  object_class?: string | null;
  color: string;
  confidence: number;
  timestamp: number;
  bounding_box: BoundingBox;
}

export interface FaceDetectionItem {
  id: string;
  track_id?: string | null;
  timestamp: number;
  confidence: number;
  bounding_box: BoundingBox;
}

export interface SecurityZoneItem {
  zone_id: string;
  name: string;
  polygon: number[][];
  target_classes?: string[];
  enabled?: boolean;
}

export interface SecurityEventItem {
  id: string;
  video_id: string;
  event_type: string;
  timestamp: number;
  duration_seconds?: number | null;
  track_id?: string | null;
  object_class?: string | null;
  severity: "LOW" | "NORMAL" | "HIGH";
  confidence: number;
  zone_name?: string | null;
  description: string;
  observable_signals?: string[] | Record<string, any>;
  bounding_box?: BoundingBox | null;
  evidence_id?: string | null;
  detector_name?: string | null;
  detector_version?: string | null;
  category?: string | null;
  human_verification_required?: boolean | null;
  validation_decision?: string | null;
  incident_metadata?: Record<string, any> | null;
}

export async function getVideoTracks(
  videoId: string,
  objectClass?: string
): Promise<{ video_id: string; total_tracks: number; tracks: TrackedObjectItem[] }> {
  const params = new URLSearchParams();
  if (objectClass) params.set("object_class", objectClass);
  const q = params.toString() ? `?${params.toString()}` : "";
  const res = await fetch(`${API_BASE_URL}/api/videos/${encodeURIComponent(videoId)}/tracks${q}`);
  if (!res.ok) throw new Error(`Failed to fetch tracks (${res.status})`);
  return res.json();
}

export async function getVehicleAttributes(
  videoId: string,
  color?: string
): Promise<{ video_id: string; total_attributes: number; attributes: VehicleAttributeItem[] }> {
  const params = new URLSearchParams();
  if (color) params.set("color", color);
  const q = params.toString() ? `?${params.toString()}` : "";
  const res = await fetch(`${API_BASE_URL}/api/videos/${encodeURIComponent(videoId)}/attributes${q}`);
  if (!res.ok) throw new Error(`Failed to fetch vehicle attributes (${res.status})`);
  return res.json();
}

export async function getFaceDetections(
  videoId: string
): Promise<{ video_id: string; safety_notice: string; total_faces: number; faces: FaceDetectionItem[] }> {
  const res = await fetch(`${API_BASE_URL}/api/videos/${encodeURIComponent(videoId)}/faces`);
  if (!res.ok) throw new Error(`Failed to fetch face detections (${res.status})`);
  return res.json();
}

export async function getSecurityEvents(
  videoId: string,
  eventType?: string,
  severity?: string
): Promise<{ video_id: string; total_events: number; events: SecurityEventItem[] }> {
  const params = new URLSearchParams();
  if (eventType) params.set("event_type", eventType);
  if (severity) params.set("severity", severity);
  const q = params.toString() ? `?${params.toString()}` : "";
  const res = await fetch(`${API_BASE_URL}/api/videos/${encodeURIComponent(videoId)}/security-events${q}`);
  if (!res.ok) throw new Error(`Failed to fetch security events (${res.status})`);
  return res.json();
}

export async function getVideoZones(
  videoId: string
): Promise<{ video_id: string; total_zones: number; zones: SecurityZoneItem[] }> {
  const res = await fetch(`${API_BASE_URL}/api/videos/${encodeURIComponent(videoId)}/zones`);
  if (!res.ok) throw new Error(`Failed to fetch zones (${res.status})`);
  return res.json();
}

export async function createVideoZone(
  videoId: string,
  payload: {
    name: string;
    polygon: number[][];
    target_classes?: string[];
    alert_on_entry?: boolean;
    loitering_threshold_seconds?: number;
  }
): Promise<{ video_id: string; status: string; zone: SecurityZoneItem }> {
  const res = await fetch(`${API_BASE_URL}/api/videos/${encodeURIComponent(videoId)}/zones`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!res.ok) throw new Error(`Failed to create zone (${res.status})`);
  return res.json();
}

export async function deleteVideoZone(
  videoId: string,
  zoneId: string
): Promise<{ video_id: string; zone_id: string; status: string }> {
  const res = await fetch(
    `${API_BASE_URL}/api/videos/${encodeURIComponent(videoId)}/zones/${encodeURIComponent(zoneId)}`,
    { method: "DELETE" }
  );
  if (!res.ok) throw new Error(`Failed to delete zone (${res.status})`);
  return res.json();
}

export async function runSecurityAnalysis(
  videoId: string,
  payload?: { zones?: any[] }
): Promise<any> {
  const res = await fetch(
    `${API_BASE_URL}/api/videos/${encodeURIComponent(videoId)}/run-security-analysis`,
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload || {}),
    }
  );
  if (!res.ok) throw new Error(`Failed to run security analysis (${res.status})`);
  return res.json();
}

// ---------------------------------------------------------------------------
// Phase 9: Incident Dossier & Evidence-Based Reporting Types & API
// ---------------------------------------------------------------------------

export interface ReportItem {
  id: string;
  report_id: string;
  video_id: string;
  title: string;
  report_type: string;
  file_size_bytes: number;
  page_count: number;
  status: string;
  metadata: {
    total_detections?: number;
    total_tracks?: number;
    total_security_events?: number;
    has_theft_event?: boolean;
    theft_event_count?: number;
    total_evidence?: number;
    page_count?: number;
    generated_at?: string;
    [key: string]: any;
  };
  generated_at: string;
  download_url: string;
  view_url: string;
}

export interface GenerateReportPayload {
  title?: string;
  classification?: string;
  queries?: string[];
}

export function getReportDownloadUrl(reportId: string): string {
  return `${API_BASE_URL}/api/reports/${encodeURIComponent(reportId)}/download`;
}

export function getReportViewUrl(reportId: string): string {
  return `${API_BASE_URL}/api/reports/${encodeURIComponent(reportId)}/view`;
}

export async function generateVideoReport(
  videoId: string,
  payload?: GenerateReportPayload
): Promise<{ status: string; message: string; report: ReportItem }> {
  const res = await fetch(
    `${API_BASE_URL}/api/videos/${encodeURIComponent(videoId)}/reports/generate`,
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload || {}),
    }
  );
  if (!res.ok) {
    let errorMsg = `Report generation failed (Status ${res.status})`;
    try {
      const err = await res.json();
      errorMsg = extractErrorMessage(err, errorMsg);
    } catch {
      // ignore
    }
    throw new Error(errorMsg);
  }
  return res.json();
}

export async function getVideoReports(
  videoId: string
): Promise<{ status: string; video_id: string; count: number; reports: ReportItem[] }> {
  const res = await fetch(
    `${API_BASE_URL}/api/videos/${encodeURIComponent(videoId)}/reports`
  );
  if (!res.ok) {
    let errorMsg = `Failed to list reports (Status ${res.status})`;
    try {
      const err = await res.json();
      errorMsg = extractErrorMessage(err, errorMsg);
    } catch {
      // ignore
    }
    throw new Error(errorMsg);
  }
  return res.json();
}

export async function getReportDetails(
  reportId: string
): Promise<{ status: string; report: ReportItem }> {
  const res = await fetch(
    `${API_BASE_URL}/api/reports/${encodeURIComponent(reportId)}`
  );
  if (!res.ok) {
    let errorMsg = `Failed to retrieve report details (Status ${res.status})`;
    try {
      const err = await res.json();
      errorMsg = extractErrorMessage(err, errorMsg);
    } catch {
      // ignore
    }
    throw new Error(errorMsg);
  }
  return res.json();
}

// ---------------------------------------------------------------------------
// Phase 15: Specialized Visual Detection Types & Client
// ---------------------------------------------------------------------------

export interface SpecializedObservationItem {
  id: string;
  video_id: string;
  event_id?: string | null;
  episode_id?: string | null;
  detector_name: string;
  detector_version?: string | null;
  class_name: string;
  timestamp_seconds: number;
  confidence: number;
  validation_status: "RAW" | "VALID" | "UNCERTAIN" | "REJECTED" | string;
  bbox?: { x1: number; y1: number; x2: number; y2: number } | null;
  evidence_strength: number;
  supporting_evidence?: Record<string, any> | null;
  created_at?: string | null;
}

export async function getSpecializedObservations(
  videoId: string
): Promise<{ status: string; video_id: string; count: number; observations: SpecializedObservationItem[] }> {
  const res = await fetch(
    `${API_BASE_URL}/api/videos/${encodeURIComponent(videoId)}/specialized`
  );
  if (!res.ok) {
    let errorMsg = `Failed to retrieve specialized observations (Status ${res.status})`;
    try {
      const err = await res.json();
      errorMsg = extractErrorMessage(err, errorMsg);
    } catch {
      // ignore
    }
    throw new Error(errorMsg);
  }
  return res.json();
}

export interface DetectorHealthItem {
  name: string;
  version: string;
  status: "AVAILABLE" | "NOT_CONFIGURED" | "UNAVAILABLE" | "FAILED" | string;
  model_type: "trained_model" | "heuristic_cv" | "rule_based" | "foundation_only" | string;
  configured: boolean;
  observations: number;
  validated_observations: number;
  rejected_observations: number;
  failures: number;
  error_message?: string | null;
  metadata?: Record<string, any>;
}

export interface DetectorHealthResponse {
  status: string;
  video_id: string;
  health: {
    total_detectors: number;
    healthy_detectors: number;
    failed_detectors: number;
    unconfigured_detectors: number;
    total_observations: number;
    total_validated: number;
    total_rejected: number;
    total_failures: number;
    detectors: DetectorHealthItem[];
  };
}

export async function getVideoDetectorHealth(
  videoId: string
): Promise<DetectorHealthResponse> {
  const res = await fetch(
    `${API_BASE_URL}/api/videos/${encodeURIComponent(videoId)}/detector-health`
  );
  if (!res.ok) {
    let errorMsg = `Failed to retrieve detector health (Status ${res.status})`;
    try {
      const err = await res.json();
      errorMsg = extractErrorMessage(err, errorMsg);
    } catch {
      // ignore
    }
    throw new Error(errorMsg);
  }
  return res.json();
}

// ---------------------------------------------------------------------------
// Phase 16: Advanced Incident Correlation & Storylines
// ---------------------------------------------------------------------------

export interface CorrelatedIncident {
  id: string;
  incident_id: string;
  video_id: string;
  incident_category: string;
  incident_subcategory?: string | null;
  start_time: number;
  end_time: number;
  duration: number;
  primary_track_ids: string[];
  supporting_track_ids: string[];
  involved_object_classes: string[];
  source_candidate_ids: string[];
  source_detector_ids: string[];
  supporting_signal_ids: string[];
  evidence_ids: string[];
  zone_ids: string[];
  assessment_score: number;
  evidence_strength: number;
  reliability_rating: "HIGH" | "MEDIUM" | "LOW" | string;
  validation_decision: "ACCEPTED" | "REVIEW_REQUIRED" | "ABSTAINED" | string;
  negative_evidence: string[];
  contextual_factors: Record<string, any>;
  storyline: string;
  provenance: Record<string, any>;
  created_at?: string;
  updated_at?: string;
}

export interface CorrelatedIncidentsResponse {
  video_id: string;
  total_count: number;
  correlated_incidents: CorrelatedIncident[];
}

export async function getVideoCorrelatedIncidents(
  videoId: string,
  category?: string,
  severity?: string,
  validationDecision?: string
): Promise<CorrelatedIncidentsResponse> {
  const params = new URLSearchParams();
  if (category) params.append("category", category);
  if (severity) params.append("severity", severity);
  if (validationDecision) params.append("validation_decision", validationDecision);

  const qs = params.toString() ? `?${params.toString()}` : "";
  const res = await fetch(
    `${API_BASE_URL}/api/videos/${encodeURIComponent(videoId)}/correlated-incidents${qs}`
  );
  if (!res.ok) {
    let errorMsg = `Failed to retrieve correlated incidents (Status ${res.status})`;
    try {
      const err = await res.json();
      errorMsg = extractErrorMessage(err, errorMsg);
    } catch {
      // ignore
    }
    throw new Error(errorMsg);
  }
  return res.json();
}


// ===========================================================================
// Phase 17: Advanced Forensic Investigation API Types & Client Functions
// ===========================================================================

export interface UnifiedTimelineEntry {
  layer: "detection_event" | "security_event" | "correlated_incident" | "evidence";
  timestamp: number;
  end_timestamp: number;
  id: string;
  label: string;
  validation_decision?: string;
  assessment_score?: number;
  reliability_rating?: string;
  primary_track_ids?: string[];
  severity?: string;
  track_id?: string;
  description?: string;
  has_snapshot?: boolean;
  has_clip?: boolean;
  event_id?: string;
  total_detections?: number;
  priority?: string;
  details?: Record<string, unknown>;
}

export interface UnifiedTimelineResponse {
  video_id: string;
  is_supported: boolean;
  result_type: string;
  start_time_filter?: number;
  end_time_filter?: number;
  total_entries: number;
  timeline: UnifiedTimelineEntry[];
  layer_counts: {
    detection_events: number;
    security_events: number;
    correlated_incidents: number;
    evidence: number;
  };
}

export interface InvestigationMatchedIncident {
  incident_id: string;
  video_id: string;
  incident_category: string;
  incident_subcategory?: string;
  start_time: number;
  end_time: number;
  duration: number;
  primary_track_ids: string[];
  supporting_track_ids: string[];
  involved_object_classes: string[];
  assessment_score: number;
  evidence_strength: number;
  reliability_rating: string;
  validation_decision: string;
  storyline: string;
  evidence_ids: string[];
  negative_evidence: unknown[];
}

export interface InvestigationResult {
  query: Record<string, unknown>;
  interpretation: string;
  matched_incidents: InvestigationMatchedIncident[];
  matched_events: unknown[];
  matched_tracks: unknown[];
  matched_evidence: unknown[];
  timeline: UnifiedTimelineEntry[];
  relationships: unknown[];
  negative_evidence: unknown[];
  total_results: number;
  truncated: boolean;
  source_video_id: string;
  provenance: Record<string, unknown>;
  grounded_answer: string;
  diagnostics: {
    db_query_time_ms: number;
    total_time_ms: number;
    incidents_found: number;
    events_found: number;
    tracks_found: number;
    evidence_found: number;
    fallback_used: boolean;
  };
  query_text?: string;
}

export interface TrackInvestigationResult {
  video_id: string;
  track_id: string;
  is_supported: boolean;
  message: string;
  result_type: string;
  track?: {
    track_id: string;
    object_class: string;
    first_seen: number;
    last_seen: number;
    duration_seconds: number;
    detection_count: number;
    max_confidence: number;
    color?: string;
    active: boolean;
  };
  lifecycle?: {
    track_id: string;
    object_class: string;
    phases: Array<{ phase: string; timestamp?: number; note: string }>;
    observational_note: string;
  };
  security_events: unknown[];
  related_incidents: InvestigationMatchedIncident[];
  evidence: unknown[];
}

export interface Phase17InvestigationQueryRequest {
  time_start?: number;
  time_end?: number;
  incident_categories?: string[];
  validation_decisions?: string[];
  min_assessment_score?: number;
  max_assessment_score?: number;
  reliability_levels?: string[];
  track_ids?: string[];
  object_classes?: string[];
  zone_ids?: string[];
  evidence_required?: boolean;
  correlated_only?: boolean;
  review_required_only?: boolean;
  rejected_only?: boolean;
  search_text?: string;
  sort_order?: string;
  result_limit?: number;
  result_offset?: number;
}

export interface EvidenceBundle {
  bundle_id: string;
  video_id: string;
  bundle_name: string;
  selected_incident_ids: string[];
  selected_event_ids: string[];
  selected_track_ids: string[];
  selected_evidence_ids: string[];
  storyline_text?: string;
  notes?: string;
  provenance: Record<string, unknown>;
  created_at?: string;
  updated_at?: string;
}

// ---------------------------------------------------------------------------
// Phase 17 API Functions
// ---------------------------------------------------------------------------

async function handleApiResponse<T>(res: Response, context: string): Promise<T> {
  if (!res.ok) {
    let errorMsg = `${context} failed (Status ${res.status})`;
    try {
      const err = await res.json();
      errorMsg = extractErrorMessage(err, errorMsg);
    } catch {
      // ignore
    }
    throw new Error(errorMsg);
  }
  return res.json();
}

/** Phase 17: Get unified forensic timeline for a video. */
export async function getInvestigationTimeline(
  videoId: string,
  startTime?: number,
  endTime?: number,
  includeRejected = false
): Promise<UnifiedTimelineResponse> {
  const params = new URLSearchParams();
  if (startTime !== undefined) params.append("start_time", String(startTime));
  if (endTime !== undefined) params.append("end_time", String(endTime));
  if (includeRejected) params.append("include_rejected", "true");
  const qs = params.toString() ? `?${params.toString()}` : "";
  const res = await fetch(
    `${API_BASE_URL}/api/videos/${encodeURIComponent(videoId)}/investigation/timeline${qs}`
  );
  return handleApiResponse<UnifiedTimelineResponse>(res, "Timeline retrieval");
}

/** Phase 17: Natural-language investigation search. */
export async function investigationSearch(
  videoId: string,
  query: string,
  videoDuration?: number
): Promise<InvestigationResult> {
  const params = new URLSearchParams({ q: query });
  if (videoDuration !== undefined) params.append("video_duration", String(videoDuration));
  const res = await fetch(
    `${API_BASE_URL}/api/videos/${encodeURIComponent(videoId)}/investigation/search?${params.toString()}`
  );
  return handleApiResponse<InvestigationResult>(res, "Investigation search");
}

/** Phase 17: Structured investigation query. */
export async function investigationStructuredQuery(
  videoId: string,
  body: Phase17InvestigationQueryRequest
): Promise<InvestigationResult> {
  const res = await fetch(
    `${API_BASE_URL}/api/videos/${encodeURIComponent(videoId)}/investigation/query`,
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    }
  );
  return handleApiResponse<InvestigationResult>(res, "Structured query");
}

/** Phase 17: Full track investigation. */
export async function investigationTrack(
  videoId: string,
  trackId: string
): Promise<TrackInvestigationResult> {
  const res = await fetch(
    `${API_BASE_URL}/api/videos/${encodeURIComponent(videoId)}/investigation/tracks/${encodeURIComponent(trackId)}`
  );
  return handleApiResponse<TrackInvestigationResult>(res, "Track investigation");
}

/** Phase 17: Investigation evidence retrieval. */
export async function investigationEvidence(
  videoId: string,
  options?: {
    startTime?: number;
    endTime?: number;
    eventId?: string;
    incidentId?: string;
  }
): Promise<{ video_id: string; total_count: number; evidence: unknown[] }> {
  const params = new URLSearchParams();
  if (options?.startTime !== undefined) params.append("start_time", String(options.startTime));
  if (options?.endTime !== undefined) params.append("end_time", String(options.endTime));
  if (options?.eventId) params.append("event_id", options.eventId);
  if (options?.incidentId) params.append("incident_id", options.incidentId);
  const qs = params.toString() ? `?${params.toString()}` : "";
  const res = await fetch(
    `${API_BASE_URL}/api/videos/${encodeURIComponent(videoId)}/investigation/evidence${qs}`
  );
  return handleApiResponse(res, "Evidence retrieval");
}

/** Phase 17: Create evidence bundle. */
export async function createBundle(
  videoId: string,
  body: {
    bundle_name?: string;
    selected_incident_ids?: string[];
    selected_event_ids?: string[];
    selected_track_ids?: string[];
    selected_evidence_ids?: string[];
    storyline_text?: string;
    notes?: string;
  }
): Promise<{ status: string; bundle: EvidenceBundle }> {
  const res = await fetch(
    `${API_BASE_URL}/api/videos/${encodeURIComponent(videoId)}/investigation/bundle`,
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    }
  );
  return handleApiResponse(res, "Bundle creation");
}

/** Phase 17: Retrieve evidence bundle by ID. */
export async function getBundle(
  videoId: string,
  bundleId: string
): Promise<{ video_id: string; bundle: EvidenceBundle }> {
  const res = await fetch(
    `${API_BASE_URL}/api/videos/${encodeURIComponent(videoId)}/investigation/bundle/${encodeURIComponent(bundleId)}`
  );
  return handleApiResponse(res, "Bundle retrieval");
}

/** Phase 17: List all evidence bundles for a video. */
export async function listBundles(
  videoId: string
): Promise<{ video_id: string; total_count: number; bundles: EvidenceBundle[] }> {
  const res = await fetch(
    `${API_BASE_URL}/api/videos/${encodeURIComponent(videoId)}/investigation/bundles`
  );
  return handleApiResponse(res, "Bundle list");
}

// ============================================================
// Phase 18: Multi-Camera Session & Cross-Camera Intelligence
// ============================================================

// --- Type definitions ---

export interface SurveillanceSession {
  id: string;
  name: string;
  site_name: string | null;
  description: string | null;
  status: "active" | "archived";
  created_at: string;
  updated_at: string;
  cameras?: CameraSource[];
  camera_count?: number;
}

export interface CameraSource {
  id: string;
  session_id: string;
  video_id: string;
  camera_label: string;
  position_hint: string | null;
  field_of_view_hint: string | null;
  adjacency_hints: string[];
  created_at: string;
}

export interface CrossCameraAssociation {
  id: string;
  session_id: string;
  source_camera_id: string;
  source_video_id: string;
  source_track_id: string;
  source_last_seen: number | null;
  target_camera_id: string;
  target_video_id: string;
  target_track_id: string;
  target_first_seen: number | null;
  confidence: number;
  association_type: "SAME_OBJECT" | "PROBABLE_SAME" | "POSSIBLE_SAME";
  attribute_match_score: number | null;
  trajectory_compatibility_score: number | null;
  temporal_gap_seconds: number | null;
  temporal_plausibility_score: number | null;
  evidence_basis: Array<{
    signal_type: string;
    description: string;
    score: number;
    metadata: Record<string, unknown>;
  }>;
  analyst_review_required: boolean;
  analyst_verdict: "PENDING" | "CONFIRMED" | "REJECTED";
  analyst_notes: string | null;
  analyst_verdict_at: string | null;
  created_at: string;
  privacy_note: string;
}

export interface CrossCameraAnalysisResult {
  associations_created: number;
  associations: CrossCameraAssociation[];
  summary: {
    session_id: string;
    camera_count: number;
    total_tracks_analyzed: number;
    total_associations: number;
    associations_by_type: Record<string, number>;
    pending_review: number;
    confirmed: number;
    rejected: number;
    average_confidence: number;
    privacy_note: string;
  };
}

export interface CrossCameraTimelineItem {
  type: "security_event" | "detection_event";
  camera_label: string;
  video_id: string;
  timestamp: number;
  event_type: string;
  id: string;
  [key: string]: unknown;
}

// --- Session CRUD ---

/** Phase 18: Create a new surveillance session. */
export async function createSession(body: {
  name: string;
  site_name?: string;
  description?: string;
}): Promise<SurveillanceSession> {
  const res = await fetch(`${API_BASE_URL}/api/sessions`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  return handleApiResponse(res, "Create session");
}

/** Phase 18: List all surveillance sessions. */
export async function listSessions(
  limit = 50,
  offset = 0
): Promise<{ total: number; sessions: SurveillanceSession[] }> {
  const params = new URLSearchParams({ limit: String(limit), offset: String(offset) });
  const res = await fetch(`${API_BASE_URL}/api/sessions?${params}`);
  return handleApiResponse(res, "List sessions");
}

/** Phase 18: Get full session detail. */
export async function getSession(sessionId: string): Promise<SurveillanceSession> {
  const res = await fetch(`${API_BASE_URL}/api/sessions/${encodeURIComponent(sessionId)}`);
  return handleApiResponse(res, "Get session");
}

/** Phase 18: Delete a session (cascade). */
export async function deleteSession(sessionId: string): Promise<void> {
  const res = await fetch(`${API_BASE_URL}/api/sessions/${encodeURIComponent(sessionId)}`, {
    method: "DELETE",
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error((err as { detail?: string }).detail || `Delete session failed: ${res.status}`);
  }
}

// --- Camera management ---

/** Phase 18: Add a camera (video) to a session. */
export async function addCameraToSession(
  sessionId: string,
  body: {
    video_id: string;
    camera_label: string;
    position_hint?: string;
    field_of_view_hint?: string;
    adjacency_hints?: string[];
  }
): Promise<CameraSource> {
  const res = await fetch(`${API_BASE_URL}/api/sessions/${encodeURIComponent(sessionId)}/cameras`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  return handleApiResponse(res, "Add camera to session");
}

/** Phase 18: List cameras in a session. */
export async function getCamerasForSession(
  sessionId: string
): Promise<{ session_id: string; camera_count: number; cameras: CameraSource[] }> {
  const res = await fetch(
    `${API_BASE_URL}/api/sessions/${encodeURIComponent(sessionId)}/cameras`
  );
  return handleApiResponse(res, "List cameras");
}

/** Phase 18: Remove a camera from a session. */
export async function removeCameraFromSession(
  sessionId: string,
  cameraId: string
): Promise<void> {
  const res = await fetch(
    `${API_BASE_URL}/api/sessions/${encodeURIComponent(sessionId)}/cameras/${encodeURIComponent(cameraId)}`,
    { method: "DELETE" }
  );
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(
      (err as { detail?: string }).detail || `Remove camera failed: ${res.status}`
    );
  }
}

// --- Cross-camera analysis ---

/** Phase 18: Trigger cross-camera association analysis for a session. */
export async function runCrossCameraAnalysis(
  sessionId: string
): Promise<CrossCameraAnalysisResult> {
  const res = await fetch(
    `${API_BASE_URL}/api/sessions/${encodeURIComponent(sessionId)}/analyze`,
    { method: "POST" }
  );
  return handleApiResponse(res, "Cross-camera analysis");
}

// --- Association queries ---

/** Phase 18: List cross-camera associations with optional filters. */
export async function getAssociations(
  sessionId: string,
  params?: {
    min_confidence?: number;
    association_type?: string;
    analyst_verdict?: string;
    limit?: number;
    offset?: number;
  }
): Promise<{ session_id: string; total: number; associations: CrossCameraAssociation[] }> {
  const q = new URLSearchParams();
  if (params?.min_confidence !== undefined) q.append("min_confidence", String(params.min_confidence));
  if (params?.association_type) q.append("association_type", params.association_type);
  if (params?.analyst_verdict) q.append("analyst_verdict", params.analyst_verdict);
  if (params?.limit !== undefined) q.append("limit", String(params.limit));
  if (params?.offset !== undefined) q.append("offset", String(params.offset));
  const res = await fetch(
    `${API_BASE_URL}/api/sessions/${encodeURIComponent(sessionId)}/associations?${q}`
  );
  return handleApiResponse(res, "List associations");
}

/** Phase 18: Get full detail of a specific cross-camera association. */
export async function getAssociation(
  sessionId: string,
  associationId: string
): Promise<CrossCameraAssociation> {
  const res = await fetch(
    `${API_BASE_URL}/api/sessions/${encodeURIComponent(sessionId)}/associations/${encodeURIComponent(associationId)}`
  );
  return handleApiResponse(res, "Get association");
}

/** Phase 18: Update analyst verdict on an association. */
export async function updateAnalystVerdict(
  sessionId: string,
  associationId: string,
  body: { verdict: "PENDING" | "CONFIRMED" | "REJECTED"; notes?: string }
): Promise<CrossCameraAssociation> {
  const res = await fetch(
    `${API_BASE_URL}/api/sessions/${encodeURIComponent(sessionId)}/associations/${encodeURIComponent(associationId)}/verdict`,
    {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    }
  );
  return handleApiResponse(res, "Update analyst verdict");
}

/** Phase 18: Delete a cross-camera association. */
export async function deleteAssociation(
  sessionId: string,
  associationId: string
): Promise<void> {
  const res = await fetch(
    `${API_BASE_URL}/api/sessions/${encodeURIComponent(sessionId)}/associations/${encodeURIComponent(associationId)}`,
    { method: "DELETE" }
  );
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(
      (err as { detail?: string }).detail || `Delete association failed: ${res.status}`
    );
  }
}

// --- Timeline ---

/** Phase 18: Get cross-camera merged chronological timeline. */
export async function getCrossCameraTimeline(
  sessionId: string
): Promise<{ session_id: string; total_events: number; timeline: CrossCameraTimelineItem[] }> {
  const res = await fetch(
    `${API_BASE_URL}/api/sessions/${encodeURIComponent(sessionId)}/timeline`
  );
  return handleApiResponse(res, "Cross-camera timeline");
}

// --- Track associations ---

/** Phase 18: Get cross-camera associations for a specific track. */
export async function getTrackAssociations(
  sessionId: string,
  videoId: string,
  trackId: string
): Promise<{
  session_id: string;
  video_id: string;
  track_id: string;
  associations: CrossCameraAssociation[];
}> {
  const res = await fetch(
    `${API_BASE_URL}/api/sessions/${encodeURIComponent(sessionId)}/tracks/${encodeURIComponent(videoId)}/${encodeURIComponent(trackId)}/associations`
  );
  return handleApiResponse(res, "Track associations");
}


// ===========================================================================
// Phase 19: Forensic Case Management & Investigation Workspace API
// ===========================================================================

export interface Case {
  id: string;
  case_id?: string;
  case_number: string;
  title: string;
  description?: string | null;
  status: "OPEN" | "INVESTIGATING" | "REVIEW" | "CLOSED";
  priority: "LOW" | "MEDIUM" | "HIGH" | "CRITICAL";
  assigned_investigator?: string | null;
  tags: string[];
  summary?: string | null;
  created_at?: string | null;
  updated_at?: string | null;
  counts?: {
    videos: number;
    cameras: number;
    incidents: number;
    evidence?: number;
    bookmarks: number;
    notes: number;
    annotations: number;
  };
  linked_videos?: Array<{
    link_id: string;
    video_id: string;
    filename: string;
    duration_seconds: number;
    fps: number;
    status: string;
    uploaded_at?: string | null;
    link_notes?: string | null;
  }>;
  linked_cameras?: Array<{
    link_id: string;
    camera_id: string;
    session_id: string;
    camera_label: string;
    position_hint?: string | null;
    field_of_view_hint?: string | null;
    video_id: string;
    clock_offset_seconds: number;
    notes?: string | null;
  }>;
  linked_incidents?: Array<{
    link_id: string;
    incident_id: string;
    incident_type: string;
    notes?: string | null;
    added_at?: string;
    incident_category?: string;
    start_time?: number;
    end_time?: number;
    validation_decision?: string;
    storyline?: string;
  }>;
  linked_evidence?: Array<{
    link_id: string;
    evidence_id: string;
    notes?: string | null;
    added_at?: string;
    video_id?: string;
    source_video_name?: string;
    timestamp_seconds?: number;
    evidence_type?: string;
    object_class?: string;
    validation_status?: string;
    snapshot_path?: string;
  }>;
}

export interface CaseBookmark {
  id: string;
  case_id: string;
  video_id: string;
  camera_id?: string | null;
  timestamp_seconds: number;
  title: string;
  description?: string | null;
  linked_incident_id?: string | null;
  linked_track_id?: string | null;
  linked_evidence_id?: string | null;
  author?: string | null;
  created_at?: string | null;
  updated_at?: string | null;
}

export interface CaseNote {
  id: string;
  case_id: string;
  content: string;
  author: string;
  associated_type: "CASE" | "INCIDENT" | "EVIDENCE" | "TRACK" | "TIMESTAMP" | "CAMERA";
  associated_id?: string | null;
  timestamp_seconds?: number | null;
  note_classification: string;
  created_at?: string | null;
  updated_at?: string | null;
}

export interface CaseAnnotation {
  id: string;
  case_id: string;
  video_id: string;
  camera_id?: string | null;
  timestamp_seconds: number;
  end_timestamp_seconds?: number | null;
  annotation_type: "POINT" | "REGION" | "TIMESTAMP_MARKER" | "TEXT_NOTE" | "INCIDENT_MARKER";
  data: Record<string, any>;
  author: string;
  created_at?: string | null;
  updated_at?: string | null;
}

export interface CaseActivity {
  id: string;
  action_type: string;
  description: string;
  details?: Record<string, any> | null;
  actor: string;
  created_at: string;
}

export interface CaseTimelineItem {
  layer: "detection_event" | "security_event" | "correlated_incident" | "evidence" | "bookmark" | "analyst_note" | "annotation";
  id: string;
  video_id?: string;
  source_name?: string;
  camera_label?: string | null;
  timestamp: number;
  end_timestamp?: number;
  label: string;
  detail?: string;
  severity?: string;
  track_id?: string;
  assessment_score?: number;
  evidence_strength?: number;
  reliability_rating?: string;
  validation_decision?: string;
  has_snapshot?: boolean;
  has_clip?: boolean;
  author?: string;
  annotation_type?: string;
  data?: Record<string, any>;
  provenance?: Record<string, any>;
}

export interface IncidentReplayContext {
  incident_id: string;
  video_id: string;
  video_filename: string;
  category: string;
  subcategory?: string | null;
  assessment_score: number;
  validation_decision: string;
  storyline?: string;
  primary_tracks: string[];
  incident_window: [number, number];
  replay_context: {
    pre_roll_seconds: number;
    post_roll_seconds: number;
    replay_start: number;
    replay_end: number;
    duration: number;
  };
  stream_url: string;
  evidence_clips: Array<{
    evidence_id: string;
    timestamp: number;
    clip_url: string;
    snapshot_url?: string | null;
  }>;
  camera_contexts: Array<{
    camera_id: string;
    camera_label: string;
    video_id: string;
    clock_offset_seconds: number;
    aligned_replay_window: [number, number];
    stream_url: string;
  }>;
}

export interface IncidentExplanation {
  incident_id: string;
  supporting_signals: string[];
  limiting_signals: string[];
  pattern_evidence_strength: string;
  pattern_evidence_strength_value: number;
  final_assessment_score: number;
  validation_decision: string;
  reliability_rating: string;
  human_verification_required: boolean;
  human_verification_notice: string;
  provenance: Record<string, any>;
}

export interface CaseTopologyData {
  case_id: string;
  total_cameras: number;
  nodes: Array<{
    id: string;
    label: string;
    position_hint?: string;
    field_of_view_hint?: string;
    video_id: string;
    clock_offset_seconds: number;
  }>;
  edges: Array<{
    id?: string;
    source: string;
    target: string;
    type: string;
    confidence?: number;
    association_type?: string;
    verdict?: string;
    temporal_gap?: number;
    label: string;
    status: string;
  }>;
}

// --- Case CRUD Client Functions ---

export async function createCase(body: {
  title: string;
  description?: string;
  priority?: string;
  assigned_investigator?: string;
  tags?: string[];
  summary?: string;
  initial_video_ids?: string[];
  initial_camera_ids?: string[];
}): Promise<Case> {
  const res = await fetch(`${API_BASE_URL}/api/cases`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  return handleApiResponse(res, "Create case");
}

export async function listCases(params?: {
  status?: string;
  priority?: string;
  tag?: string;
  search?: string;
  limit?: number;
  offset?: number;
}): Promise<{ total: number; cases: Case[] }> {
  const q = new URLSearchParams();
  if (params?.status) q.append("status", params.status);
  if (params?.priority) q.append("priority", params.priority);
  if (params?.tag) q.append("tag", params.tag);
  if (params?.search) q.append("search", params.search);
  if (params?.limit) q.append("limit", String(params.limit));
  if (params?.offset) q.append("offset", String(params.offset));

  const qs = q.toString() ? `?${q.toString()}` : "";
  const res = await fetch(`${API_BASE_URL}/api/cases${qs}`);
  return handleApiResponse(res, "List cases");
}

export async function getCases(params?: {
  status?: string;
  priority?: string;
  tag?: string;
  search?: string;
  limit?: number;
  offset?: number;
}): Promise<Case[]> {
  // Default to limit=100 for responsive loading
  const res = await listCases({ limit: 100, ...params });
  return res.cases;
}

export async function getCase(caseId: string): Promise<Case> {
  const res = await fetch(`${API_BASE_URL}/api/cases/${encodeURIComponent(caseId)}`);
  return handleApiResponse(res, "Get case");
}

export async function updateCase(
  caseId: string,
  body: {
    title?: string;
    description?: string;
    status?: string;
    priority?: string;
    assigned_investigator?: string;
    tags?: string[];
    summary?: string;
  }
): Promise<Case> {
  const res = await fetch(`${API_BASE_URL}/api/cases/${encodeURIComponent(caseId)}`, {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  return handleApiResponse(res, "Update case");
}

export async function deleteCase(caseId: string): Promise<void> {
  const res = await fetch(`${API_BASE_URL}/api/cases/${encodeURIComponent(caseId)}`, {
    method: "DELETE",
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error((err as { detail?: string }).detail || `Delete case failed: ${res.status}`);
  }
}

// --- Entity Linking ---

export async function linkVideoToCase(caseId: string, videoId: string, notes?: string): Promise<any> {
  const res = await fetch(`${API_BASE_URL}/api/cases/${encodeURIComponent(caseId)}/videos`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ video_id: videoId, notes }),
  });
  return handleApiResponse(res, "Link video");
}

export async function unlinkVideoFromCase(caseId: string, videoId: string): Promise<void> {
  const res = await fetch(
    `${API_BASE_URL}/api/cases/${encodeURIComponent(caseId)}/videos/${encodeURIComponent(videoId)}`,
    { method: "DELETE" }
  );
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error((err as { detail?: string }).detail || `Unlink video failed: ${res.status}`);
  }
}

export async function linkCameraToCase(
  caseId: string,
  cameraId: string,
  clockOffsetSeconds: number = 0.0,
  notes?: string
): Promise<any> {
  const res = await fetch(`${API_BASE_URL}/api/cases/${encodeURIComponent(caseId)}/cameras`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ camera_id: cameraId, clock_offset_seconds: clockOffsetSeconds, notes }),
  });
  return handleApiResponse(res, "Link camera");
}

export async function unlinkCameraFromCase(caseId: string, cameraId: string): Promise<void> {
  const res = await fetch(
    `${API_BASE_URL}/api/cases/${encodeURIComponent(caseId)}/cameras/${encodeURIComponent(cameraId)}`,
    { method: "DELETE" }
  );
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error((err as { detail?: string }).detail || `Unlink camera failed: ${res.status}`);
  }
}

export async function linkIncidentToCase(
  caseId: string,
  incidentId: string,
  incidentType: string = "CORRELATED",
  notes?: string
): Promise<any> {
  const res = await fetch(`${API_BASE_URL}/api/cases/${encodeURIComponent(caseId)}/incidents`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ incident_id: incidentId, incident_type: incidentType, notes }),
  });
  return handleApiResponse(res, "Link incident");
}

export async function unlinkIncidentFromCase(caseId: string, incidentId: string): Promise<void> {
  const res = await fetch(
    `${API_BASE_URL}/api/cases/${encodeURIComponent(caseId)}/incidents/${encodeURIComponent(incidentId)}`,
    { method: "DELETE" }
  );
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error((err as { detail?: string }).detail || `Unlink incident failed: ${res.status}`);
  }
}

export async function linkEvidenceToCase(
  caseId: string,
  evidenceId: string,
  notes?: string
): Promise<{ link_id: string; case_id: string; evidence_id: string; notes?: string; added_at: string }> {
  const res = await fetch(`${API_BASE_URL}/api/cases/${encodeURIComponent(caseId)}/evidence`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ evidence_id: evidenceId, notes }),
  });
  return handleApiResponse(res, "Link evidence");
}

export async function listCaseEvidence(caseId: string): Promise<{ case_id: string; total: number; evidence: any[] }> {
  const res = await fetch(`${API_BASE_URL}/api/cases/${encodeURIComponent(caseId)}/evidence`);
  return handleApiResponse(res, "List case evidence");
}

export async function unlinkEvidenceFromCase(caseId: string, evidenceId: string): Promise<void> {
  const res = await fetch(
    `${API_BASE_URL}/api/cases/${encodeURIComponent(caseId)}/evidence/${encodeURIComponent(evidenceId)}`,
    { method: "DELETE" }
  );
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error((err as { detail?: string }).detail || `Unlink evidence failed: ${res.status}`);
  }
}

// --- Bookmarks ---

export async function createCaseBookmark(
  caseId: string,
  body: {
    video_id: string;
    timestamp_seconds: number;
    title: string;
    description?: string;
    camera_id?: string;
    linked_incident_id?: string;
    linked_track_id?: string;
    linked_evidence_id?: string;
    author?: string;
  }
): Promise<CaseBookmark> {
  const res = await fetch(`${API_BASE_URL}/api/cases/${encodeURIComponent(caseId)}/bookmarks`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  return handleApiResponse(res, "Create bookmark");
}

export async function listCaseBookmarks(caseId: string, videoId?: string): Promise<{ total: number; bookmarks: CaseBookmark[] }> {
  const q = videoId ? `?video_id=${encodeURIComponent(videoId)}` : "";
  const res = await fetch(`${API_BASE_URL}/api/cases/${encodeURIComponent(caseId)}/bookmarks${q}`);
  return handleApiResponse(res, "List bookmarks");
}

export async function deleteCaseBookmark(caseId: string, bookmarkId: string): Promise<void> {
  const res = await fetch(
    `${API_BASE_URL}/api/cases/${encodeURIComponent(caseId)}/bookmarks/${encodeURIComponent(bookmarkId)}`,
    { method: "DELETE" }
  );
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error((err as { detail?: string }).detail || `Delete bookmark failed: ${res.status}`);
  }
}

// --- Notes ---

export async function createCaseNote(
  caseId: string,
  body: {
    content: string;
    author?: string;
    associated_type?: string;
    associated_id?: string;
    timestamp_seconds?: number;
  }
): Promise<CaseNote> {
  const res = await fetch(`${API_BASE_URL}/api/cases/${encodeURIComponent(caseId)}/notes`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  return handleApiResponse(res, "Create note");
}

export async function listCaseNotes(
  caseId: string,
  associatedType?: string,
  associatedId?: string
): Promise<{ total: number; notes: CaseNote[] }> {
  const q = new URLSearchParams();
  if (associatedType) q.append("associated_type", associatedType);
  if (associatedId) q.append("associated_id", associatedId);
  const qs = q.toString() ? `?${q.toString()}` : "";
  const res = await fetch(`${API_BASE_URL}/api/cases/${encodeURIComponent(caseId)}/notes${qs}`);
  return handleApiResponse(res, "List notes");
}

export async function deleteCaseNote(caseId: string, noteId: string): Promise<void> {
  const res = await fetch(
    `${API_BASE_URL}/api/cases/${encodeURIComponent(caseId)}/notes/${encodeURIComponent(noteId)}`,
    { method: "DELETE" }
  );
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error((err as { detail?: string }).detail || `Delete note failed: ${res.status}`);
  }
}

// --- Annotations ---

export async function createCaseAnnotation(
  caseId: string,
  body: {
    video_id: string;
    timestamp_seconds: number;
    end_timestamp_seconds?: number;
    annotation_type?: string;
    data: Record<string, any>;
    author?: string;
    camera_id?: string;
  }
): Promise<CaseAnnotation> {
  const res = await fetch(`${API_BASE_URL}/api/cases/${encodeURIComponent(caseId)}/annotations`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  return handleApiResponse(res, "Create annotation");
}

export async function listCaseAnnotations(
  caseId: string,
  params?: { video_id?: string; start_time?: number; end_time?: number }
): Promise<{ total: number; annotations: CaseAnnotation[] }> {
  const q = new URLSearchParams();
  if (params?.video_id) q.append("video_id", params.video_id);
  if (params?.start_time !== undefined) q.append("start_time", String(params.start_time));
  if (params?.end_time !== undefined) q.append("end_time", String(params.end_time));
  const qs = q.toString() ? `?${q.toString()}` : "";
  const res = await fetch(`${API_BASE_URL}/api/cases/${encodeURIComponent(caseId)}/annotations${qs}`);
  return handleApiResponse(res, "List annotations");
}

export async function deleteCaseAnnotation(caseId: string, annotationId: string): Promise<void> {
  const res = await fetch(
    `${API_BASE_URL}/api/cases/${encodeURIComponent(caseId)}/annotations/${encodeURIComponent(annotationId)}`,
    { method: "DELETE" }
  );
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error((err as { detail?: string }).detail || `Delete annotation failed: ${res.status}`);
  }
}

// --- Case Timeline ---

export async function getCaseTimeline(
  caseId: string,
  params?: { start_time?: number; end_time?: number; layers?: string; limit?: number; offset?: number }
): Promise<{ case_id: string; total_entries: number; offset: number; limit: number; layer_counts: Record<string, number>; timeline: CaseTimelineItem[] }> {
  const q = new URLSearchParams();
  if (params?.start_time !== undefined) q.append("start_time", String(params.start_time));
  if (params?.end_time !== undefined) q.append("end_time", String(params.end_time));
  if (params?.layers) q.append("layers", params.layers);
  if (params?.limit) q.append("limit", String(params.limit));
  if (params?.offset) q.append("offset", String(params.offset));
  const qs = q.toString() ? `?${q.toString()}` : "";
  const res = await fetch(`${API_BASE_URL}/api/cases/${encodeURIComponent(caseId)}/timeline${qs}`);
  return handleApiResponse(res, "Get case timeline");
}

// --- Replay, Focus, Explainability, Topology, Storyline, Export ---

export async function getIncidentReplayContext(
  caseId: string,
  incidentId: string,
  preRoll: number = 5.0,
  postRoll: number = 5.0
): Promise<IncidentReplayContext> {
  const res = await fetch(
    `${API_BASE_URL}/api/cases/${encodeURIComponent(caseId)}/incidents/${encodeURIComponent(incidentId)}/replay?pre_roll_seconds=${preRoll}&post_roll_seconds=${postRoll}`
  );
  return handleApiResponse(res, "Get incident replay context");
}

export async function getIncidentFocusData(caseId: string, incidentId: string): Promise<any> {
  const res = await fetch(
    `${API_BASE_URL}/api/cases/${encodeURIComponent(caseId)}/incidents/${encodeURIComponent(incidentId)}/focus`
  );
  return handleApiResponse(res, "Get incident focus data");
}

export async function getIncidentExplanation(caseId: string, incidentId: string): Promise<IncidentExplanation> {
  const res = await fetch(
    `${API_BASE_URL}/api/cases/${encodeURIComponent(caseId)}/incidents/${encodeURIComponent(incidentId)}/explain`
  );
  return handleApiResponse(res, "Get incident explanation");
}

export async function getCaseTopology(caseId: string): Promise<CaseTopologyData> {
  const res = await fetch(`${API_BASE_URL}/api/cases/${encodeURIComponent(caseId)}/topology`);
  return handleApiResponse(res, "Get case topology");
}

export async function getCaseStoryline(caseId: string): Promise<{ case_id: string; case_number: string; case_title: string; total_steps: number; storyline: any[] }> {
  const res = await fetch(`${API_BASE_URL}/api/cases/${encodeURIComponent(caseId)}/storyline`);
  return handleApiResponse(res, "Get case storyline");
}

export async function getCaseActivities(caseId: string, limit: number = 50): Promise<{ case_id: string; total: number; activities: CaseActivity[] }> {
  const res = await fetch(`${API_BASE_URL}/api/cases/${encodeURIComponent(caseId)}/activities?limit=${limit}`);
  return handleApiResponse(res, "Get case activities");
}

export async function exportCaseData(caseId: string): Promise<Record<string, any>> {
  const res = await fetch(`${API_BASE_URL}/api/cases/${encodeURIComponent(caseId)}/export`);
  return handleApiResponse(res, "Export case data");
}

// ---------------------------------------------------------------------------
// Phase 19.1: Global Entity & Workspace Integration APIs
// ---------------------------------------------------------------------------

export interface VideoListItem {
  id: string;
  video_id: string;
  filename: string;
  original_filename: string;
  file_size_bytes: number;
  duration_seconds: number | null;
  fps: number | null;
  frame_count: number | null;
  status: string;
  camera_id?: string | null;
  uploaded_at: string | null;
  processed_at: string | null;
  detections_count: number;
  raw_detections_count: number;
  incidents_count: number;
  evidence_count: number;
  playback_url: string;
  stream_url: string;
}

export async function listVideos(params?: {
  search?: string;
  status?: string;
  limit?: number;
  offset?: number;
}): Promise<{ total: number; videos: VideoListItem[]; limit: number; offset: number }> {
  const q = new URLSearchParams();
  if (params?.search) q.append("search", params.search);
  if (params?.status) q.append("status", params.status);
  if (params?.limit) q.append("limit", String(params.limit));
  if (params?.offset) q.append("offset", String(params.offset));
  const qs = q.toString() ? `?${q.toString()}` : "";
  const res = await fetch(`${API_BASE_URL}/api/videos${qs}`);
  return handleApiResponse(res, "List security videos");
}

export interface GlobalEvidenceItem {
  id: string;
  evidence_id: string;
  video_id: string;
  source_video_name: string;
  evidence_type: string;
  validation_status: string;
  timestamp_seconds: number;
  object_class: string;
  confidence: number;
  bounding_box: any;
  start_time: number;
  end_time: number;
  duration_seconds: number;
  notes: string | null;
  created_at: string | null;
  has_snapshot: boolean;
  has_annotated: boolean;
  has_clip: boolean;
  snapshot_url: string | null;
  annotated_url: string | null;
  clip_url: string | null;
  playback_url: string | null;
}

export async function listAllEvidence(params?: {
  video_id?: string;
  object_class?: string;
  validation_status?: string;
  min_confidence?: number;
  search?: string;
  limit?: number;
  offset?: number;
}): Promise<{ total: number; evidence: GlobalEvidenceItem[]; limit: number; offset: number }> {
  const q = new URLSearchParams();
  if (params?.video_id) q.append("video_id", params.video_id);
  if (params?.object_class) q.append("object_class", params.object_class);
  if (params?.validation_status) q.append("validation_status", params.validation_status);
  if (params?.min_confidence !== undefined) q.append("min_confidence", String(params.min_confidence));
  if (params?.search) q.append("search", params.search);
  if (params?.limit) q.append("limit", String(params.limit));
  if (params?.offset) q.append("offset", String(params.offset));
  const qs = q.toString() ? `?${q.toString()}` : "";
  const res = await fetch(`${API_BASE_URL}/api/evidence${qs}`);
  return handleApiResponse(res, "List evidence items");
}

export interface GlobalIncidentItem {
  id: string;
  incident_id: string;
  video_id: string;
  source_video_name: string;
  incident_category: string;
  category: string;
  title: string;
  start_time: number;
  end_time: number;
  duration: number;
  assessment_score: number;
  evidence_strength: number;
  reliability_rating: string;
  validation_decision: string;
  decision: string;
  storyline: string;
  narrative: string;
  participating_tracks: string[];
  primary_track_ids: string[];
  supporting_track_ids: string[];
  involved_object_classes: string[];
  evidence_ids: string[];
  negative_evidence: any[];
  contextual_factors: any[];
  human_verification_required: number;
  review_required: boolean;
  created_at: string | null;
}

export async function listAllIncidents(params?: {
  video_id?: string;
  category?: string;
  decision?: string;
  min_score?: number;
  search?: string;
  limit?: number;
  offset?: number;
}): Promise<{ total: number; incidents: GlobalIncidentItem[]; limit: number; offset: number }> {
  const q = new URLSearchParams();
  if (params?.video_id) q.append("video_id", params.video_id);
  if (params?.category) q.append("category", params.category);
  if (params?.decision) q.append("decision", params.decision);
  if (params?.min_score !== undefined) q.append("min_score", String(params.min_score));
  if (params?.search) q.append("search", params.search);
  if (params?.limit) q.append("limit", String(params.limit));
  if (params?.offset) q.append("offset", String(params.offset));
  const qs = q.toString() ? `?${q.toString()}` : "";
  const res = await fetch(`${API_BASE_URL}/api/incidents${qs}`);
  return handleApiResponse(res, "List security incidents");
}

export interface GlobalCameraItem {
  id: string;
  camera_id: string;
  camera_label: string;
  status?: string;
  video_id?: string | null;
  session_id?: string | null;
  session_name?: string;
  site_name?: string;
  position_hint?: string | null;
  field_of_view_hint?: string | null;
  location?: string | null;
  coverage_description?: string | null;
  adjacency_hints?: string[];
  associated_videos_count?: number;
  video_filename?: string | null;
  video_duration?: number | null;
  video_fps?: number | null;
  video_status?: string | null;
  playback_url?: string | null;
  created_at?: string | null;
}

export async function listAllCameras(params?: {
  search?: string;
  limit?: number;
  offset?: number;
}): Promise<{ total: number; cameras: GlobalCameraItem[]; limit: number; offset: number }> {
  const q = new URLSearchParams();
  if (params?.search) q.append("search", params.search);
  if (params?.limit) q.append("limit", String(params.limit));
  if (params?.offset) q.append("offset", String(params.offset));
  const qs = q.toString() ? `?${q.toString()}` : "";
  const res = await fetch(`${API_BASE_URL}/api/cameras${qs}`);
  return handleApiResponse(res, "List camera sources");
}

export async function registerCamera(data: {
  camera_label: string;
  video_id?: string;
  session_id?: string;
  position_hint?: string;
  field_of_view_hint?: string;
  location?: string;
  coverage_description?: string;
  adjacency_hints?: string[];
}): Promise<GlobalCameraItem> {
  const res = await fetch(`${API_BASE_URL}/api/cameras`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(data),
  });
  return handleApiResponse(res, "Register camera source");
}

export interface AnalyticsSummary {
  total_videos: number;
  processed_videos: number;
  total_cameras: number;
  raw_observations_count: number;
  validated_detections_count: number;
  rejected_detections_count: number;
  uncertain_detections_count: number;
  parity_consistent: boolean;
  parity_formula?: string;
  total_tracks: number;
  total_security_events: number;
  total_correlated_incidents: number;
  total_evidence: number;
  validated_evidence_count: number;
  total_cases: number;
  open_cases?: number;
  review_required_cases?: number;
  closed_cases?: number;
  cases_by_status: Record<string, number>;
  cases_by_priority: Record<string, number>;
  review_required_count: number;
  review_ceiling: number;
  specialized_counts: Record<string, number>;
  detector_health: Array<{ id: string; name: string; status: string; type: string }>;
}

export async function getAnalyticsSummary(): Promise<AnalyticsSummary> {
  const res = await fetch(`${API_BASE_URL}/api/analytics`);
  return handleApiResponse(res, "Get analytics summary");
}

export async function listAllReports(): Promise<{ status: string; count: number; reports: ReportItem[] }> {
  const res = await fetch(`${API_BASE_URL}/api/reports`);
  return handleApiResponse(res, "List all incident reports");
}



