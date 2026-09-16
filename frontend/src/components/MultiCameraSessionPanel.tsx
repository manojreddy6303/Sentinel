/**
 * MultiCameraSessionPanel.tsx — Phase 18
 *
 * Premium multi-camera surveillance session management and cross-camera intelligence panel.
 * Sections:
 *  1. Session Manager — create / list / select sessions
 *  2. Camera Management — add cameras, define adjacency
 *  3. Analysis — trigger cross-camera association analysis
 *  4. Association Review — review, confirm, reject hypotheses
 *  5. Cross-Camera Timeline — merged events across all cameras
 */
import React, { useState, useEffect, useCallback } from "react";
import {
  SurveillanceSession,
  CameraSource,
  CrossCameraAssociation,
  CrossCameraTimelineItem,
  CrossCameraAnalysisResult,
  createSession,
  listSessions,
  getSession,
  deleteSession,
  addCameraToSession,
  getCamerasForSession,
  removeCameraFromSession,
  runCrossCameraAnalysis,
  getAssociations,
  updateAnalystVerdict,
  deleteAssociation,
  getCrossCameraTimeline,
} from "../lib/api";

// ─── Confidence helpers ────────────────────────────────────────────────────

function confidenceColor(c: number): string {
  if (c >= 0.80) return "#10b981";
  if (c >= 0.60) return "#f59e0b";
  if (c >= 0.40) return "#6366f1";
  return "#ef4444";
}

function confidenceBadge(type: string): React.ReactNode {
  const colors: Record<string, string> = {
    SAME_OBJECT: "#10b981",
    PROBABLE_SAME: "#f59e0b",
    POSSIBLE_SAME: "#6366f1",
  };
  return (
    <span style={{
      background: colors[type] || "#6b7280",
      color: "#fff",
      borderRadius: 4,
      padding: "2px 8px",
      fontSize: 11,
      fontWeight: 700,
      letterSpacing: "0.04em",
    }}>
      {type.replace(/_/g, " ")}
    </span>
  );
}

function verdictBadge(v: string): React.ReactNode {
  const colors: Record<string, string> = {
    PENDING: "#6b7280",
    CONFIRMED: "#10b981",
    REJECTED: "#ef4444",
  };
  return (
    <span style={{
      background: colors[v] || "#6b7280",
      color: "#fff",
      borderRadius: 4,
      padding: "2px 8px",
      fontSize: 11,
      fontWeight: 700,
    }}>
      {v}
    </span>
  );
}

// ─── Sub-panel: Session List ───────────────────────────────────────────────

interface SessionListProps {
  sessions: SurveillanceSession[];
  selectedId: string | null;
  onSelect: (id: string) => void;
  onDelete: (id: string) => void;
  loading: boolean;
}

function SessionList({ sessions, selectedId, onSelect, onDelete, loading }: SessionListProps) {
  if (loading) return <div style={{ color: "#6b7280", padding: 16 }}>Loading sessions…</div>;
  if (!sessions.length) return <div style={{ color: "#6b7280", padding: 16 }}>No sessions yet. Create one above.</div>;
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
      {sessions.map(s => (
        <div
          key={s.id}
          onClick={() => onSelect(s.id)}
          style={{
            background: selectedId === s.id ? "rgba(99,102,241,0.18)" : "rgba(255,255,255,0.04)",
            border: `1px solid ${selectedId === s.id ? "rgba(99,102,241,0.6)" : "rgba(255,255,255,0.08)"}`,
            borderRadius: 10,
            padding: "12px 16px",
            cursor: "pointer",
            transition: "all 0.18s",
            display: "flex",
            alignItems: "center",
            gap: 12,
          }}
        >
          <div style={{ fontSize: 20 }}>🎥</div>
          <div style={{ flex: 1 }}>
            <div style={{ color: "#e2e8f0", fontWeight: 600, fontSize: 14 }}>{s.name}</div>
            <div style={{ color: "#94a3b8", fontSize: 12 }}>{s.site_name || "—"} · {s.status}</div>
          </div>
          <button
            id={`delete-session-${s.id}`}
            onClick={e => { e.stopPropagation(); onDelete(s.id); }}
            style={{
              background: "rgba(239,68,68,0.15)",
              border: "1px solid rgba(239,68,68,0.4)",
              color: "#ef4444",
              borderRadius: 6,
              padding: "4px 10px",
              cursor: "pointer",
              fontSize: 12,
            }}
          >
            Delete
          </button>
        </div>
      ))}
    </div>
  );
}

// ─── Sub-panel: Camera Manager ─────────────────────────────────────────────

interface CameraManagerProps {
  sessionId: string;
  cameras: CameraSource[];
  onRefresh: () => void;
}

function CameraManager({ sessionId, cameras, onRefresh }: CameraManagerProps) {
  const [videoId, setVideoId] = useState("");
  const [label, setLabel] = useState("");
  const [posHint, setPosHint] = useState("");
  const [fovHint, setFovHint] = useState("");
  const [adjHints, setAdjHints] = useState("");
  const [adding, setAdding] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleAdd = async () => {
    if (!videoId.trim() || !label.trim()) {
      setError("Video ID and camera label are required.");
      return;
    }
    setAdding(true);
    setError(null);
    try {
      await addCameraToSession(sessionId, {
        video_id: videoId.trim(),
        camera_label: label.trim(),
        position_hint: posHint.trim() || undefined,
        field_of_view_hint: fovHint.trim() || undefined,
        adjacency_hints: adjHints ? adjHints.split(",").map(s => s.trim()).filter(Boolean) : undefined,
      });
      setVideoId(""); setLabel(""); setPosHint(""); setFovHint(""); setAdjHints("");
      onRefresh();
    } catch (e: unknown) {
      setError((e as Error).message);
    } finally {
      setAdding(false);
    }
  };

  const handleRemove = async (cameraId: string) => {
    try {
      await removeCameraFromSession(sessionId, cameraId);
      onRefresh();
    } catch (e: unknown) {
      setError((e as Error).message);
    }
  };

  return (
    <div>
      <h4 style={{ color: "#a78bfa", marginBottom: 12, fontSize: 13, fontWeight: 700, letterSpacing: "0.06em" }}>
        ADD CAMERA
      </h4>
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 8, marginBottom: 8 }}>
        <input
          id="cam-video-id"
          placeholder="Video ID"
          value={videoId}
          onChange={e => setVideoId(e.target.value)}
          style={inputStyle}
        />
        <input
          id="cam-label"
          placeholder="Camera Label (e.g. Entrance A)"
          value={label}
          onChange={e => setLabel(e.target.value)}
          style={inputStyle}
        />
        <input
          id="cam-pos-hint"
          placeholder="Position hint (optional)"
          value={posHint}
          onChange={e => setPosHint(e.target.value)}
          style={inputStyle}
        />
        <input
          id="cam-fov-hint"
          placeholder="Field of view hint (optional)"
          value={fovHint}
          onChange={e => setFovHint(e.target.value)}
          style={inputStyle}
        />
      </div>
      <input
        id="cam-adj-hints"
        placeholder="Adjacent camera labels, comma-separated (e.g. Entrance B, Rear Exit)"
        value={adjHints}
        onChange={e => setAdjHints(e.target.value)}
        style={{ ...inputStyle, width: "100%", boxSizing: "border-box" }}
      />
      {error && <div style={{ color: "#f87171", fontSize: 12, marginTop: 6 }}>{error}</div>}
      <button
        id="add-camera-btn"
        onClick={handleAdd}
        disabled={adding}
        style={{ ...primaryBtnStyle, marginTop: 10 }}
      >
        {adding ? "Adding…" : "➕ Add Camera"}
      </button>

      {cameras.length > 0 && (
        <div style={{ marginTop: 16 }}>
          <h4 style={{ color: "#a78bfa", fontSize: 13, fontWeight: 700, marginBottom: 8, letterSpacing: "0.06em" }}>
            REGISTERED CAMERAS ({cameras.length})
          </h4>
          {cameras.map(c => (
            <div
              key={c.id}
              style={{
                background: "rgba(255,255,255,0.04)",
                border: "1px solid rgba(255,255,255,0.08)",
                borderRadius: 8,
                padding: "10px 14px",
                marginBottom: 6,
                display: "flex",
                alignItems: "flex-start",
                gap: 10,
              }}
            >
              <div style={{ fontSize: 18 }}>📷</div>
              <div style={{ flex: 1 }}>
                <div style={{ color: "#e2e8f0", fontWeight: 600, fontSize: 13 }}>{c.camera_label}</div>
                <div style={{ color: "#94a3b8", fontSize: 11 }}>Video: {c.video_id}</div>
                {c.position_hint && <div style={{ color: "#64748b", fontSize: 11 }}>📍 {c.position_hint}</div>}
                {c.adjacency_hints?.length > 0 && (
                  <div style={{ color: "#64748b", fontSize: 11 }}>↔ {c.adjacency_hints.join(", ")}</div>
                )}
              </div>
              <button
                id={`remove-camera-${c.id}`}
                onClick={() => handleRemove(c.id)}
                style={{ background: "rgba(239,68,68,0.15)", border: "1px solid rgba(239,68,68,0.4)", color: "#ef4444", borderRadius: 6, padding: "3px 8px", cursor: "pointer", fontSize: 11 }}
              >
                Remove
              </button>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

// ─── Sub-panel: Association Review ────────────────────────────────────────

interface AssociationTableProps {
  sessionId: string;
  associations: CrossCameraAssociation[];
  onRefresh: () => void;
}

function AssociationTable({ sessionId, associations, onRefresh }: AssociationTableProps) {
  const [expanded, setExpanded] = useState<string | null>(null);

  const handleVerdict = async (assocId: string, verdict: "CONFIRMED" | "REJECTED") => {
    try {
      await updateAnalystVerdict(sessionId, assocId, { verdict });
      onRefresh();
    } catch (e: unknown) {
      console.error(e);
    }
  };

  const handleDelete = async (assocId: string) => {
    try {
      await deleteAssociation(sessionId, assocId);
      onRefresh();
    } catch (e: unknown) {
      console.error(e);
    }
  };

  if (!associations.length) {
    return <div style={{ color: "#6b7280", padding: 16 }}>No associations found. Run analysis first.</div>;
  }

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
      {associations.map(a => (
        <div
          key={a.id}
          style={{
            background: "rgba(255,255,255,0.04)",
            border: `1px solid ${a.analyst_verdict === "CONFIRMED" ? "rgba(16,185,129,0.4)" : a.analyst_verdict === "REJECTED" ? "rgba(239,68,68,0.25)" : "rgba(255,255,255,0.09)"}`,
            borderRadius: 10,
            padding: "12px 16px",
          }}
        >
          <div style={{ display: "flex", alignItems: "center", gap: 10, flexWrap: "wrap" }}>
            <div style={{ flex: 1 }}>
              <div style={{ color: "#e2e8f0", fontSize: 13, fontWeight: 600 }}>
                {a.source_track_id} → {a.target_track_id}
              </div>
              <div style={{ color: "#94a3b8", fontSize: 11 }}>
                Cam {a.source_camera_id.slice(-6)} → Cam {a.target_camera_id.slice(-6)} ·
                Gap: {a.temporal_gap_seconds !== null ? `${a.temporal_gap_seconds.toFixed(1)}s` : "—"}
              </div>
            </div>
            {confidenceBadge(a.association_type)}
            <div style={{ color: confidenceColor(a.confidence), fontWeight: 700, fontSize: 14 }}>
              {(a.confidence * 100).toFixed(0)}%
            </div>
            {verdictBadge(a.analyst_verdict)}
          </div>

          <div style={{ display: "flex", gap: 6, marginTop: 10, flexWrap: "wrap" }}>
            {a.analyst_verdict !== "CONFIRMED" && (
              <button
                id={`confirm-assoc-${a.id}`}
                onClick={() => handleVerdict(a.id, "CONFIRMED")}
                style={{ background: "rgba(16,185,129,0.15)", border: "1px solid rgba(16,185,129,0.4)", color: "#10b981", borderRadius: 6, padding: "4px 12px", cursor: "pointer", fontSize: 12 }}
              >
                ✓ Confirm
              </button>
            )}
            {a.analyst_verdict !== "REJECTED" && (
              <button
                id={`reject-assoc-${a.id}`}
                onClick={() => handleVerdict(a.id, "REJECTED")}
                style={{ background: "rgba(239,68,68,0.12)", border: "1px solid rgba(239,68,68,0.35)", color: "#ef4444", borderRadius: 6, padding: "4px 12px", cursor: "pointer", fontSize: 12 }}
              >
                ✗ Reject
              </button>
            )}
            <button
              id={`expand-assoc-${a.id}`}
              onClick={() => setExpanded(expanded === a.id ? null : a.id)}
              style={{ background: "rgba(255,255,255,0.06)", border: "1px solid rgba(255,255,255,0.12)", color: "#94a3b8", borderRadius: 6, padding: "4px 12px", cursor: "pointer", fontSize: 12 }}
            >
              {expanded === a.id ? "▲ Hide Evidence" : "▼ Evidence Basis"}
            </button>
            <button
              id={`delete-assoc-${a.id}`}
              onClick={() => handleDelete(a.id)}
              style={{ background: "transparent", border: "1px solid rgba(239,68,68,0.3)", color: "#ef4444", borderRadius: 6, padding: "4px 12px", cursor: "pointer", fontSize: 12 }}
            >
              🗑
            </button>
          </div>

          {expanded === a.id && a.evidence_basis.length > 0 && (
            <div style={{ marginTop: 12, borderTop: "1px solid rgba(255,255,255,0.07)", paddingTop: 10 }}>
              <div style={{ color: "#64748b", fontSize: 11, marginBottom: 6, fontWeight: 700 }}>EVIDENCE BASIS</div>
              {a.evidence_basis.map((ev, idx) => (
                <div key={idx} style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 4 }}>
                  <div style={{
                    width: 50, height: 4, borderRadius: 2,
                    background: `linear-gradient(to right, ${confidenceColor(ev.score)} ${ev.score * 100}%, rgba(255,255,255,0.08) 0)`,
                  }} />
                  <span style={{ color: "#a78bfa", fontSize: 11, fontWeight: 600 }}>{ev.signal_type}</span>
                  <span style={{ color: "#94a3b8", fontSize: 11 }}>{ev.description}</span>
                </div>
              ))}
              <div style={{ color: "#475569", fontSize: 10, marginTop: 8, fontStyle: "italic" }}>
                {a.privacy_note}
              </div>
            </div>
          )}
        </div>
      ))}
    </div>
  );
}

// ─── Sub-panel: Timeline ──────────────────────────────────────────────────

function TimelinePanel({ items }: { items: CrossCameraTimelineItem[] }) {
  const cameras = Array.from(new Set(items.map(i => i.camera_label)));

  return (
    <div>
      {cameras.length > 0 && (
        <div style={{ display: "flex", gap: 8, marginBottom: 12, flexWrap: "wrap" }}>
          {cameras.map(c => (
            <span key={c} style={{
              background: "rgba(99,102,241,0.15)",
              border: "1px solid rgba(99,102,241,0.35)",
              color: "#a78bfa",
              borderRadius: 6,
              padding: "3px 10px",
              fontSize: 12,
              fontWeight: 600,
            }}>📷 {c}</span>
          ))}
        </div>
      )}
      {items.length === 0 ? (
        <div style={{ color: "#6b7280", padding: 12 }}>No events in timeline.</div>
      ) : (
        <div style={{ display: "flex", flexDirection: "column", gap: 4, maxHeight: 400, overflowY: "auto" }}>
          {items.map((item, idx) => (
            <div
              key={`${item.id}-${idx}`}
              style={{
                background: item.type === "security_event" ? "rgba(245,158,11,0.08)" : "rgba(99,102,241,0.07)",
                border: `1px solid ${item.type === "security_event" ? "rgba(245,158,11,0.2)" : "rgba(99,102,241,0.18)"}`,
                borderRadius: 7,
                padding: "8px 12px",
                display: "flex",
                alignItems: "center",
                gap: 10,
              }}
            >
              <span style={{ color: "#94a3b8", fontSize: 11, minWidth: 46, fontFamily: "monospace" }}>
                {item.timestamp.toFixed(1)}s
              </span>
              <span style={{
                background: item.type === "security_event" ? "rgba(245,158,11,0.2)" : "rgba(99,102,241,0.2)",
                color: item.type === "security_event" ? "#f59e0b" : "#a78bfa",
                borderRadius: 4,
                padding: "1px 6px",
                fontSize: 10,
                fontWeight: 700,
              }}>
                {item.camera_label}
              </span>
              <span style={{ color: "#e2e8f0", fontSize: 12 }}>{item.event_type}</span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

// ─── Styles ───────────────────────────────────────────────────────────────

const inputStyle: React.CSSProperties = {
  background: "rgba(255,255,255,0.05)",
  border: "1px solid rgba(255,255,255,0.12)",
  borderRadius: 8,
  padding: "9px 12px",
  color: "#e2e8f0",
  fontSize: 13,
  outline: "none",
  width: "100%",
  boxSizing: "border-box",
};

const primaryBtnStyle: React.CSSProperties = {
  background: "linear-gradient(135deg, #6366f1, #8b5cf6)",
  border: "none",
  borderRadius: 8,
  color: "#fff",
  padding: "9px 20px",
  cursor: "pointer",
  fontSize: 13,
  fontWeight: 700,
  letterSpacing: "0.03em",
};

const sectionStyle: React.CSSProperties = {
  background: "rgba(255,255,255,0.03)",
  border: "1px solid rgba(255,255,255,0.08)",
  borderRadius: 14,
  padding: "20px 24px",
  marginBottom: 16,
};

const tabStyle = (active: boolean): React.CSSProperties => ({
  padding: "8px 18px",
  borderRadius: 8,
  border: active ? "1px solid rgba(99,102,241,0.6)" : "1px solid transparent",
  background: active ? "rgba(99,102,241,0.18)" : "transparent",
  color: active ? "#a78bfa" : "#6b7280",
  cursor: "pointer",
  fontWeight: 700,
  fontSize: 13,
  transition: "all 0.15s",
});

// ─── Main Component ───────────────────────────────────────────────────────

type Tab = "sessions" | "cameras" | "analysis" | "associations" | "timeline";

const MultiCameraSessionPanel: React.FC = () => {
  const [tab, setTab] = useState<Tab>("sessions");
  const [sessions, setSessions] = useState<SurveillanceSession[]>([]);
  const [selectedSessionId, setSelectedSessionId] = useState<string | null>(null);
  const [selectedSession, setSelectedSession] = useState<SurveillanceSession | null>(null);
  const [cameras, setCameras] = useState<CameraSource[]>([]);
  const [associations, setAssociations] = useState<CrossCameraAssociation[]>([]);
  const [timeline, setTimeline] = useState<CrossCameraTimelineItem[]>([]);
  const [analysisResult, setAnalysisResult] = useState<CrossCameraAnalysisResult | null>(null);

  const [sessionsLoading, setSessionsLoading] = useState(false);
  const [analysisRunning, setAnalysisRunning] = useState(false);
  const [analysisError, setAnalysisError] = useState<string | null>(null);

  // Create session form
  const [newName, setNewName] = useState("");
  const [newSite, setNewSite] = useState("");
  const [newDesc, setNewDesc] = useState("");
  const [creating, setCreating] = useState(false);
  const [createError, setCreateError] = useState<string | null>(null);

  // Verdict filter
  const [verdictFilter, setVerdictFilter] = useState<string>("");
  const [typeFilter, setTypeFilter] = useState<string>("");

  const loadSessions = useCallback(async () => {
    setSessionsLoading(true);
    try {
      const r = await listSessions();
      setSessions(r.sessions);
    } catch (e) {
      console.error(e);
    } finally {
      setSessionsLoading(false);
    }
  }, []);

  const loadCameras = useCallback(async (sid: string) => {
    try {
      const r = await getCamerasForSession(sid);
      setCameras(r.cameras);
    } catch (e) {
      console.error(e);
    }
  }, []);

  const loadAssociations = useCallback(async (sid: string) => {
    try {
      const r = await getAssociations(sid, {
        analyst_verdict: verdictFilter || undefined,
        association_type: typeFilter || undefined,
        limit: 100,
      });
      setAssociations(r.associations);
    } catch (e) {
      console.error(e);
    }
  }, [verdictFilter, typeFilter]);

  const loadTimeline = useCallback(async (sid: string) => {
    try {
      const r = await getCrossCameraTimeline(sid);
      setTimeline(r.timeline);
    } catch (e) {
      console.error(e);
    }
  }, []);

  useEffect(() => { loadSessions(); }, [loadSessions]);

  useEffect(() => {
    if (!selectedSessionId) return;
    loadCameras(selectedSessionId);
    loadAssociations(selectedSessionId);
    if (tab === "timeline") loadTimeline(selectedSessionId);
  }, [selectedSessionId, tab, loadCameras, loadAssociations, loadTimeline]);

  const handleSelectSession = async (id: string) => {
    setSelectedSessionId(id);
    try {
      const s = await getSession(id);
      setSelectedSession(s);
    } catch (e) {
      console.error(e);
    }
  };

  const handleDeleteSession = async (id: string) => {
    try {
      await deleteSession(id);
      if (selectedSessionId === id) {
        setSelectedSessionId(null);
        setSelectedSession(null);
      }
      loadSessions();
    } catch (e) {
      console.error(e);
    }
  };

  const handleCreateSession = async () => {
    if (!newName.trim()) { setCreateError("Name is required."); return; }
    setCreating(true);
    setCreateError(null);
    try {
      const s = await createSession({ name: newName.trim(), site_name: newSite.trim() || undefined, description: newDesc.trim() || undefined });
      setNewName(""); setNewSite(""); setNewDesc("");
      await loadSessions();
      await handleSelectSession(s.id);
    } catch (e: unknown) {
      setCreateError((e as Error).message);
    } finally {
      setCreating(false);
    }
  };

  const handleRunAnalysis = async () => {
    if (!selectedSessionId) return;
    setAnalysisRunning(true);
    setAnalysisError(null);
    try {
      const r = await runCrossCameraAnalysis(selectedSessionId);
      setAnalysisResult(r);
      setAssociations(r.associations);
    } catch (e: unknown) {
      setAnalysisError((e as Error).message);
    } finally {
      setAnalysisRunning(false);
    }
  };

  return (
    <div style={{
      fontFamily: "'Inter', 'Segoe UI', sans-serif",
      color: "#e2e8f0",
      padding: "24px 0",
      minHeight: 400,
    }}>
      {/* Header */}
      <div style={{ marginBottom: 20 }}>
        <h2 style={{
          fontSize: 20,
          fontWeight: 800,
          background: "linear-gradient(135deg, #a78bfa, #6366f1)",
          WebkitBackgroundClip: "text",
          WebkitTextFillColor: "transparent",
          marginBottom: 4,
        }}>
          🎥 Multi-Camera Session Intelligence
        </h2>
        <div style={{ color: "#94a3b8", fontSize: 12, marginTop: 4 }}>
          Combine multiple CCTV sources for cross-camera investigative analysis &bull; Single-video investigations do not require a multi-camera session.
        </div>
      </div>

      {/* Tab bar */}
      <div style={{ display: "flex", gap: 6, marginBottom: 20, flexWrap: "wrap" }}>
        {(["sessions", "cameras", "analysis", "associations", "timeline"] as Tab[]).map(t => (
          <button key={t} id={`tab-${t}`} style={tabStyle(tab === t)} onClick={() => setTab(t)}>
            {{ sessions: "📋 Sessions", cameras: "📷 Cameras", analysis: "⚡ Analysis", associations: "🔗 Associations", timeline: "📅 Timeline" }[t]}
          </button>
        ))}
      </div>

      {/* ── Sessions tab ── */}
      {tab === "sessions" && (
        <div>
          <div style={sectionStyle}>
            <h3 style={{ color: "#a78bfa", fontSize: 13, fontWeight: 700, marginBottom: 14, letterSpacing: "0.06em" }}>
              CREATE SESSION
            </h3>
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 8, marginBottom: 8 }}>
              <input id="session-name" placeholder="Session name *" value={newName} onChange={e => setNewName(e.target.value)} style={inputStyle} />
              <input id="session-site" placeholder="Site name (optional)" value={newSite} onChange={e => setNewSite(e.target.value)} style={inputStyle} />
            </div>
            <input id="session-desc" placeholder="Description (optional)" value={newDesc} onChange={e => setNewDesc(e.target.value)} style={{ ...inputStyle, width: "100%", boxSizing: "border-box", marginBottom: 8 }} />
            {createError && <div style={{ color: "#f87171", fontSize: 12, marginBottom: 6 }}>{createError}</div>}
            <button id="create-session-btn" onClick={handleCreateSession} disabled={creating} style={primaryBtnStyle}>
              {creating ? "Creating…" : "➕ Create Session"}
            </button>
          </div>
          <div style={sectionStyle}>
            <h3 style={{ color: "#a78bfa", fontSize: 13, fontWeight: 700, marginBottom: 12, letterSpacing: "0.06em" }}>
              SESSIONS ({sessions.length})
            </h3>
            <SessionList
              sessions={sessions}
              selectedId={selectedSessionId}
              onSelect={handleSelectSession}
              onDelete={handleDeleteSession}
              loading={sessionsLoading}
            />
          </div>
        </div>
      )}

      {/* ── Cameras tab ── */}
      {tab === "cameras" && (
        <div style={sectionStyle}>
          {!selectedSessionId ? (
            <div style={{ color: "#6b7280" }}>Select a session first from the Sessions tab.</div>
          ) : (
            <>
              <div style={{ color: "#94a3b8", fontSize: 12, marginBottom: 14 }}>
                Session: <strong style={{ color: "#e2e8f0" }}>{selectedSession?.name}</strong>
              </div>
              <CameraManager
                sessionId={selectedSessionId}
                cameras={cameras}
                onRefresh={() => selectedSessionId && loadCameras(selectedSessionId)}
              />
            </>
          )}
        </div>
      )}

      {/* ── Analysis tab ── */}
      {tab === "analysis" && (
        <div style={sectionStyle}>
          {!selectedSessionId ? (
            <div style={{ color: "#6b7280" }}>Select a session first from the Sessions tab.</div>
          ) : (
            <>
              <div style={{ color: "#94a3b8", fontSize: 12, marginBottom: 14 }}>
                Session: <strong style={{ color: "#e2e8f0" }}>{selectedSession?.name}</strong> · {cameras.length} camera(s)
              </div>
              <div style={{
                background: "rgba(99,102,241,0.08)",
                border: "1px solid rgba(99,102,241,0.2)",
                borderRadius: 10,
                padding: 16,
                marginBottom: 16,
                fontSize: 12,
                color: "#94a3b8",
              }}>
                <strong style={{ color: "#a78bfa" }}>ℹ Privacy Notice</strong><br />
                Cross-camera analysis uses observable attribute matching (object class, color, size, trajectory direction, temporal gap) only.
                No identity claims, facial recognition, or biometric data are produced.
                All associations are hypotheses requiring analyst review.
              </div>
              <button
                id="run-analysis-btn"
                onClick={handleRunAnalysis}
                disabled={analysisRunning || cameras.length < 2}
                style={{
                  ...primaryBtnStyle,
                  opacity: cameras.length < 2 ? 0.5 : 1,
                  cursor: cameras.length < 2 ? "not-allowed" : "pointer",
                }}
              >
                {analysisRunning ? "⏳ Running analysis…" : "⚡ Run Cross-Camera Analysis"}
              </button>
              {cameras.length < 2 && (
                <div style={{ color: "#f59e0b", fontSize: 12, marginTop: 8 }}>
                  At least 2 cameras required to run analysis.
                </div>
              )}
              {analysisError && (
                <div style={{ color: "#f87171", fontSize: 12, marginTop: 8 }}>{analysisError}</div>
              )}
              {analysisResult && (
                <div style={{ marginTop: 20 }}>
                  <div style={{ display: "grid", gridTemplateColumns: "repeat(4,1fr)", gap: 10 }}>
                    {[
                      { label: "Associations", value: analysisResult.associations_created },
                      { label: "Cameras", value: analysisResult.summary.camera_count },
                      { label: "Tracks Analysed", value: analysisResult.summary.total_tracks_analyzed },
                      { label: "Pending Review", value: analysisResult.summary.pending_review },
                    ].map(m => (
                      <div key={m.label} style={{
                        background: "rgba(255,255,255,0.04)",
                        border: "1px solid rgba(255,255,255,0.08)",
                        borderRadius: 10,
                        padding: "12px 16px",
                        textAlign: "center",
                      }}>
                        <div style={{ color: "#a78bfa", fontSize: 22, fontWeight: 800 }}>{m.value}</div>
                        <div style={{ color: "#64748b", fontSize: 11 }}>{m.label}</div>
                      </div>
                    ))}
                  </div>
                  {Object.keys(analysisResult.summary.associations_by_type).length > 0 && (
                    <div style={{ display: "flex", gap: 8, marginTop: 12, flexWrap: "wrap" }}>
                      {Object.entries(analysisResult.summary.associations_by_type).map(([k, v]) => (
                        <span key={k} style={{ background: "rgba(255,255,255,0.06)", borderRadius: 6, padding: "4px 10px", fontSize: 12, color: "#e2e8f0" }}>
                          {k.replace(/_/g, " ")}: <strong>{v}</strong>
                        </span>
                      ))}
                    </div>
                  )}
                </div>
              )}
            </>
          )}
        </div>
      )}

      {/* ── Associations tab ── */}
      {tab === "associations" && (
        <div style={sectionStyle}>
          {!selectedSessionId ? (
            <div style={{ color: "#6b7280" }}>Select a session first from the Sessions tab.</div>
          ) : (
            <>
              <div style={{ display: "flex", gap: 8, marginBottom: 14, flexWrap: "wrap" }}>
                <select id="verdict-filter" value={verdictFilter} onChange={e => setVerdictFilter(e.target.value)} style={{ ...inputStyle, width: "auto" }}>
                  <option value="">All Verdicts</option>
                  <option value="PENDING">Pending</option>
                  <option value="CONFIRMED">Confirmed</option>
                  <option value="REJECTED">Rejected</option>
                </select>
                <select id="type-filter" value={typeFilter} onChange={e => setTypeFilter(e.target.value)} style={{ ...inputStyle, width: "auto" }}>
                  <option value="">All Types</option>
                  <option value="SAME_OBJECT">Same Object</option>
                  <option value="PROBABLE_SAME">Probable Same</option>
                  <option value="POSSIBLE_SAME">Possible Same</option>
                </select>
                <button id="refresh-assoc-btn" onClick={() => selectedSessionId && loadAssociations(selectedSessionId)} style={{ ...primaryBtnStyle, padding: "8px 14px" }}>
                  ↻ Refresh
                </button>
              </div>
              <AssociationTable
                sessionId={selectedSessionId}
                associations={associations}
                onRefresh={() => selectedSessionId && loadAssociations(selectedSessionId)}
              />
            </>
          )}
        </div>
      )}

      {/* ── Timeline tab ── */}
      {tab === "timeline" && (
        <div style={sectionStyle}>
          {!selectedSessionId ? (
            <div style={{ color: "#6b7280" }}>Select a session first from the Sessions tab.</div>
          ) : (
            <>
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 14 }}>
                <div style={{ color: "#94a3b8", fontSize: 12 }}>
                  Cross-camera timeline · {timeline.length} event(s)
                </div>
                <button id="refresh-timeline-btn" onClick={() => selectedSessionId && loadTimeline(selectedSessionId)} style={{ ...primaryBtnStyle, padding: "7px 14px", fontSize: 12 }}>
                  ↻ Refresh
                </button>
              </div>
              <TimelinePanel items={timeline} />
            </>
          )}
        </div>
      )}
    </div>
  );
};

export default MultiCameraSessionPanel;
