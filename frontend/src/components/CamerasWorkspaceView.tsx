"use client";

import React, { useState, useEffect, useCallback } from "react";
import {
  GlobalCameraItem,
  VideoListItem,
  listAllCameras,
  listVideos,
  registerCamera,
} from "@/lib/api";

interface CamerasWorkspaceViewProps {
  onInvestigateVideo?: (videoId: string) => void;
  onIngestVideo?: () => void;
  onCreateCaseFromVideo?: (videoId: string, filename: string) => void;
  onOpenMultiCamera?: () => void;
  onDataChanged?: () => void;
}

export default function CamerasWorkspaceView({
  onInvestigateVideo,
  onIngestVideo,
  onCreateCaseFromVideo,
  onOpenMultiCamera,
  onDataChanged,
}: CamerasWorkspaceViewProps) {
  const [cameras, setCameras] = useState<GlobalCameraItem[]>([]);
  const [videos, setVideos] = useState<VideoListItem[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [searchQuery, setSearchQuery] = useState("");

  // Register Camera Modal
  const [isRegisterOpen, setIsRegisterOpen] = useState(false);
  const [selectedVideoId, setSelectedVideoId] = useState("");
  const [cameraLabel, setCameraLabel] = useState("");
  const [positionHint, setPositionHint] = useState("");
  const [fovHint, setFovHint] = useState("");
  const [isRegistering, setIsRegistering] = useState(false);
  const [registerError, setRegisterError] = useState<string | null>(null);
  const [totalCameras, setTotalCameras] = useState<number>(0);



  const fetchData = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const [camsRes, vidsRes] = await Promise.all([
        listAllCameras({ search: searchQuery || undefined }),
        listVideos({ limit: 100 }),
      ]);
      setCameras(camsRes.cameras || []);
      setTotalCameras(camsRes.total ?? camsRes.cameras?.length ?? 0);
      setVideos(vidsRes.videos || []);
      if (!selectedVideoId && vidsRes.videos?.length > 0) {
        const canonical = vidsRes.videos.find((v: any) => v.id === "0d4d92f9-19f8-42e3-925f-1931cb557705");
        setSelectedVideoId(canonical ? canonical.id : vidsRes.videos[0].id);
      }
    } catch (err: any) {
      setError(err.message || "Failed to load camera sources.");
    } finally {
      setLoading(false);
    }
  }, [searchQuery, selectedVideoId]);

  useEffect(() => {
    fetchData();
  }, [fetchData]);

  const handleRegisterSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!cameraLabel.trim()) {
      setRegisterError("Camera label is required.");
      return;
    }
    setIsRegistering(true);
    setRegisterError(null);
    try {
      await registerCamera({
        camera_label: cameraLabel.trim(),
        video_id: selectedVideoId || undefined,
        position_hint: positionHint.trim() || undefined,
        field_of_view_hint: fovHint.trim() || undefined,
      });
      setIsRegisterOpen(false);
      setCameraLabel("");
      setPositionHint("");
      setFovHint("");
      await fetchData();
      if (onDataChanged) onDataChanged();
    } catch (err: any) {
      setRegisterError(err.message || "Failed to register camera source.");
    } finally {
      setIsRegistering(false);
    }
  };



  return (
    <div className="p-6 md:p-8 space-y-8 max-w-[1600px] mx-auto min-h-full">
      {/* Header & Actions */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 border-b border-zinc-800/80 pb-6">
        <div>
          <div className="flex items-center gap-3">
            <h1 className="text-2xl font-bold tracking-tight text-white flex items-center gap-2.5">
              <span className="w-3 h-3 rounded-full bg-emerald-500 animate-pulse" />
              Camera &amp; Video Sources
            </h1>
            <span className="px-2.5 py-0.5 rounded-full text-xs font-mono font-semibold bg-emerald-500/10 text-emerald-400 border border-emerald-500/30">
              {totalCameras} Cameras Active
            </span>
          </div>
          <p className="text-sm text-zinc-400 mt-1.5">
            CCTV camera streams, physical sensor locations, spatial topology, and ingested video feeds.
          </p>
        </div>

        <div className="flex items-center flex-wrap gap-3">
          <button
            onClick={() => {
              if (onIngestVideo) {
                onIngestVideo();
              }
            }}
            className="flex items-center gap-2 px-4 py-2 bg-zinc-800 hover:bg-zinc-700 text-zinc-100 rounded-lg text-sm font-semibold transition-all border border-zinc-700 hover:border-zinc-600 shadow-sm cursor-pointer"
          >
            <svg className="w-4 h-4 text-emerald-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-8l-4-4m0 0L8 8m4-4v12" />
            </svg>
            Ingest Video
          </button>

          <button
            onClick={() => setIsRegisterOpen(true)}
            className="flex items-center gap-2 px-4 py-2 bg-emerald-600 hover:bg-emerald-500 text-zinc-950 rounded-lg text-sm font-bold transition-all shadow-md hover:shadow-emerald-500/20"
          >
            <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4v16m8-8H4" />
            </svg>
            Register Camera
          </button>

          {onOpenMultiCamera && (
            <button
              onClick={onOpenMultiCamera}
              className="flex items-center gap-2 px-4 py-2 bg-blue-600/20 hover:bg-blue-600/30 text-blue-300 border border-blue-500/40 rounded-lg text-sm font-semibold transition-all"
            >
              <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 10l4.553-2.276A1 1 0 0121 8.618v6.764a1 1 0 01-1.447.894L15 14M5 18h8a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z" />
              </svg>
              Multi-Camera Grid
            </button>
          )}
        </div>
      </div>

      {/* Search & Filter Bar */}
      <div className="flex flex-col sm:flex-row items-center gap-3">
        <div className="relative flex-1 w-full">
          <svg className="absolute left-3.5 top-2.5 w-4 h-4 text-zinc-500" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
          </svg>
          <input
            type="text"
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            placeholder="Search by camera label, location hint, or FOV..."
            className="w-full pl-10 pr-4 py-2 bg-zinc-900 border border-zinc-800 rounded-lg text-sm text-zinc-200 placeholder-zinc-500 focus:outline-none focus:border-emerald-500"
          />
        </div>
      </div>

      {/* Loading & Error States */}
      {loading && (
        <div className="flex flex-col items-center justify-center py-20 text-zinc-500 space-y-3">
          <div className="w-7 h-7 border-2 border-emerald-500 border-t-transparent rounded-full animate-spin" />
          <span className="text-sm font-mono">Loading camera topology and video feeds...</span>
        </div>
      )}

      {error && !loading && (
        <div className="p-4 bg-red-500/10 border border-red-500/30 rounded-xl text-red-400 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <svg className="w-5 h-5 flex-none" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
            </svg>
            <span className="text-sm font-medium">{error}</span>
          </div>
          <button
            onClick={fetchData}
            className="px-3 py-1 bg-red-500/20 hover:bg-red-500/30 text-red-300 rounded text-xs font-semibold"
          >
            Retry
          </button>
        </div>
      )}

      {/* Section 1: Registered Camera Sources Grid */}
      {!loading && !error && cameras.length > 0 && (
        <div className="space-y-4">
          <h2 className="text-lg font-bold text-zinc-200 flex items-center gap-2">
            <span className="w-2 h-2 rounded-full bg-emerald-400" />
            1. Registered CCTV Camera Sources ({cameras.length})
          </h2>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-5">
            {cameras.map((cam) => (
              <div
                key={cam.id}
                className="bg-zinc-900/80 border border-zinc-800 hover:border-zinc-700 rounded-xl p-5 space-y-4 transition-all shadow-md hover:shadow-lg flex flex-col justify-between"
              >
                <div className="space-y-3">
                  <div className="flex items-start justify-between gap-2">
                    <div>
                      <h3 className="text-base font-bold text-white flex items-center gap-2">
                        {cam.camera_label}
                      </h3>
                      <p className="text-xs text-zinc-400 font-mono mt-0.5">
                        {cam.site_name || "Facility"} &bull; {cam.session_name || "Surveillance"}
                      </p>
                    </div>
                    <span className="px-2 py-0.5 rounded text-[11px] font-mono font-semibold bg-emerald-500/10 text-emerald-400 border border-emerald-500/30 uppercase">
                      {cam.status || "ACTIVE"}
                    </span>
                  </div>

                  <div className="bg-zinc-950/60 rounded-lg p-3 border border-zinc-800/60 space-y-2 text-xs">
                    <div className="flex justify-between">
                      <span className="text-zinc-500 font-medium">Location:</span>
                      <span className="text-zinc-300 font-medium">{cam.location || cam.position_hint || "General Facility Area"}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-zinc-500 font-medium">Coverage:</span>
                      <span className="text-zinc-300 truncate max-w-[200px]" title={cam.coverage_description || cam.field_of_view_hint || "Default Perimeter"}>
                        {cam.coverage_description || cam.field_of_view_hint || "Default Perimeter"}
                      </span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-zinc-500 font-medium">Associated Footage:</span>
                      <span className="text-emerald-400 font-mono font-bold">
                        {cam.associated_videos_count !== undefined ? cam.associated_videos_count : (cam.video_id ? 1 : 0)} video(s)
                      </span>
                    </div>
                    {cam.video_filename && (
                      <div className="flex justify-between">
                        <span className="text-zinc-500 font-medium">Latest Recording:</span>
                        <span className="text-zinc-300 font-mono truncate max-w-[180px]" title={cam.video_filename}>
                          {cam.video_filename}
                        </span>
                      </div>
                    )}
                  </div>
                </div>

                <div className="flex items-center gap-2 pt-2 border-t border-zinc-800/60">
                  {cam.video_id && onInvestigateVideo && (
                    <button
                      onClick={() => onInvestigateVideo(cam.video_id!)}
                      className="flex-1 px-3 py-1.5 bg-emerald-600/20 hover:bg-emerald-600/30 text-emerald-300 border border-emerald-500/40 rounded-lg text-xs font-semibold transition-all text-center"
                    >
                      Investigate Feed
                    </button>
                  )}
                  {cam.video_id && onCreateCaseFromVideo && (
                    <button
                      onClick={() => onCreateCaseFromVideo(cam.video_id!, cam.camera_label)}
                      className="px-3 py-1.5 bg-zinc-800 hover:bg-zinc-700 text-zinc-200 border border-zinc-700 rounded-lg text-xs font-medium transition-all"
                    >
                      Create Case
                    </button>
                  )}
                  {!cam.video_id && (
                    <span className="text-xs text-zinc-500 font-mono italic py-1">Hardware Registered &bull; Awaiting Ingest Link</span>
                  )}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Section 2: Ingested Video Sources Grid */}
      {!loading && !error && videos.length > 0 && (
        <div className="space-y-4 pt-4 border-t border-zinc-800/60">
          <div className="flex items-center justify-between">
            <h2 className="text-lg font-bold text-zinc-200 flex items-center gap-2">
              <span className="w-2 h-2 rounded-full bg-cyan-400" />
              2. Ingested Surveillance Video Files ({videos.length})
            </h2>
            <span className="text-xs text-zinc-500">Persisted in Sentinel storage</span>
          </div>

          <div className="overflow-x-auto bg-zinc-900/60 border border-zinc-800 rounded-xl">
            <table className="w-full text-left text-sm text-zinc-300">
              <thead className="bg-zinc-950/80 text-xs font-semibold text-zinc-400 uppercase tracking-wider border-b border-zinc-800">
                <tr>
                  <th className="py-3 px-4">Filename</th>
                  <th className="py-3 px-4">Camera Source</th>
                  <th className="py-3 px-4">Status</th>
                  <th className="py-3 px-4">Duration</th>
                  <th className="py-3 px-4">FPS</th>
                  <th className="py-3 px-4">Detections</th>
                  <th className="py-3 px-4">Incidents</th>
                  <th className="py-3 px-4 text-right">Actions</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-zinc-800/60 font-mono text-xs">
                {videos.map((vid) => (
                  <tr key={vid.id} className="hover:bg-zinc-800/30 transition-colors">
                    <td className="py-3 px-4 font-medium text-white truncate max-w-xs" title={vid.filename}>
                      {vid.filename}
                    </td>
                    <td className="py-3 px-4">
                      {vid.camera_id ? (
                        <span className="px-2 py-0.5 rounded text-[11px] font-mono bg-indigo-950/80 text-indigo-300 border border-indigo-800/50 font-bold">
                          {cameras.find((c) => c.id === vid.camera_id)?.camera_label || "Assigned"}
                        </span>
                      ) : (
                        <span className="text-zinc-600 text-[11px] italic font-mono">Unassigned</span>
                      )}
                    </td>
                    <td className="py-3 px-4">
                      <span className={`px-2 py-0.5 rounded text-[11px] font-semibold uppercase ${
                        vid.status === "processed"
                          ? "bg-emerald-500/10 text-emerald-400 border border-emerald-500/30"
                          : "bg-blue-500/10 text-blue-400 border border-blue-500/30"
                      }`}>
                        {vid.status}
                      </span>
                    </td>
                    <td className="py-3 px-4 text-zinc-400">
                      {vid.duration_seconds ? `${vid.duration_seconds.toFixed(1)}s` : "--"}
                    </td>
                    <td className="py-3 px-4 text-zinc-400">
                      {vid.fps ? vid.fps.toFixed(1) : "--"}
                    </td>
                    <td className="py-3 px-4 text-zinc-300">
                      <span className="text-emerald-400 font-bold">{vid.detections_count}</span>
                      <span className="text-zinc-500 ml-1">({vid.raw_detections_count} raw)</span>
                    </td>
                    <td className="py-3 px-4">
                      <span className="text-cyan-400 font-bold">{vid.incidents_count}</span>
                    </td>
                    <td className="py-3 px-4 text-right space-x-2">
                      {onInvestigateVideo && (
                        <button
                          onClick={() => onInvestigateVideo(vid.id)}
                          className="px-2.5 py-1 bg-zinc-800 hover:bg-zinc-700 text-zinc-200 rounded transition-all text-[11px] font-sans font-medium"
                        >
                          Investigate
                        </button>
                      )}
                      {onCreateCaseFromVideo && (
                        <button
                          onClick={() => onCreateCaseFromVideo(vid.id, vid.filename)}
                          className="px-2.5 py-1 bg-emerald-600/20 hover:bg-emerald-600/30 text-emerald-400 rounded transition-all text-[11px] font-sans font-semibold"
                        >
                          + Case
                        </button>
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* Section 3: Multi-Camera Analysis & Spatial Sessions */}
      {!loading && !error && (
        <div className="space-y-4 pt-6 border-t border-zinc-800/60">
          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3">
            <div>
              <h2 className="text-lg font-bold text-zinc-200 flex items-center gap-2">
                <span className="w-2 h-2 rounded-full bg-indigo-400" />
                3. Multi-Camera Analysis &amp; Spatial Sessions
              </h2>
              <p className="text-xs text-zinc-400 mt-1">
                Cross-camera tracking, topological adjacency mapping, and temporal handover verification.
              </p>
            </div>
            {onOpenMultiCamera && (
              <button
                onClick={onOpenMultiCamera}
                className="flex items-center gap-2 px-4 py-2 bg-indigo-600/20 hover:bg-indigo-600/30 text-indigo-300 border border-indigo-500/40 rounded-lg text-xs font-mono font-semibold transition-all cursor-pointer shrink-0"
              >
                <span>Launch Multi-Camera Grid &rarr;</span>
              </button>
            )}
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div className="p-4 rounded-xl bg-zinc-900/60 border border-zinc-800 space-y-2">
              <div className="text-xs font-mono text-zinc-400">SPATIAL TOPOLOGY</div>
              <div className="text-sm font-semibold text-zinc-200">Adjacency &amp; Overlap Mapping</div>
              <p className="text-xs text-zinc-500 leading-relaxed">
                Cameras calibrated with spatial hints enable automated blind-spot estimation and transition zones.
              </p>
            </div>
            <div className="p-4 rounded-xl bg-zinc-900/60 border border-zinc-800 space-y-2">
              <div className="text-xs font-mono text-indigo-400">ANONYMOUS RE-ID</div>
              <div className="text-sm font-semibold text-zinc-200">Appearance &amp; Motion Continuity</div>
              <p className="text-xs text-zinc-500 leading-relaxed">
                Associates trajectories across separate camera feeds without biometrics or facial recognition.
              </p>
            </div>
            <div className="p-4 rounded-xl bg-zinc-900/60 border border-zinc-800 space-y-2">
              <div className="text-xs font-mono text-emerald-400">SYNCHRONIZED PLAYBACK</div>
              <div className="text-sm font-semibold text-zinc-200">Unified Multi-Feed Replay</div>
              <p className="text-xs text-zinc-500 leading-relaxed">
                Interactive synchronized surveillance playback with multi-angle track correlation and timeline scrubbing.
              </p>
            </div>
          </div>
        </div>
      )}

      {/* Empty State */}
      {!loading && !error && cameras.length === 0 && videos.length === 0 && (
        <div className="rounded-xl border border-zinc-800 bg-zinc-900/40 p-12 text-center space-y-4">
          <div className="w-12 h-12 rounded-full bg-zinc-800 flex items-center justify-center mx-auto text-zinc-400">
            <svg className="w-6 h-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 10l4.553-2.276A1 1 0 0121 8.618v6.764a1 1 0 01-1.447.894L15 14M5 18h8a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z" />
            </svg>
          </div>
          <div className="space-y-1">
            <h3 className="text-base font-bold text-zinc-200">No camera sources configured</h3>
            <p className="text-sm text-zinc-500 max-w-md mx-auto">
              Ingest CCTV surveillance video or register a camera source to begin security intelligence monitoring.
            </p>
          </div>
          <button
            onClick={() => onIngestVideo?.()}
            className="px-4 py-2 bg-emerald-600 hover:bg-emerald-500 text-zinc-950 font-bold rounded-lg text-sm transition-all cursor-pointer"
          >
            Ingest Video
          </button>
        </div>
      )}

      {/* Register Camera Modal */}
      {isRegisterOpen && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/75 backdrop-blur-sm p-4">
          <div className="bg-zinc-900 border border-zinc-800 rounded-xl max-w-md w-full p-6 space-y-4 shadow-2xl">
            <div className="flex items-center justify-between border-b border-zinc-800 pb-3">
              <h3 className="text-base font-bold text-white">Register Camera Source</h3>
              <button
                onClick={() => setIsRegisterOpen(false)}
                className="text-zinc-500 hover:text-zinc-300"
              >
                ✕
              </button>
            </div>

            {registerError && (
              <div className="p-3 bg-red-500/10 border border-red-500/30 rounded text-red-400 text-xs">
                {registerError}
              </div>
            )}

            <form onSubmit={handleRegisterSubmit} className="space-y-4 text-sm">
              <div>
                <label className="block text-zinc-300 font-medium mb-1">Camera Label *</label>
                <input
                  type="text"
                  required
                  value={cameraLabel}
                  onChange={(e) => setCameraLabel(e.target.value)}
                  placeholder="e.g. North Perimeter Gate"
                  className="w-full bg-zinc-950 border border-zinc-800 rounded-lg p-2.5 text-zinc-200 focus:outline-none focus:border-emerald-500"
                />
              </div>

              <div>
                <label className="block text-zinc-300 font-medium mb-1">Associate Footage (Optional)</label>
                <select
                  value={selectedVideoId}
                  onChange={(e) => setSelectedVideoId(e.target.value)}
                  className="w-full bg-zinc-950 border border-zinc-800 rounded-lg p-2.5 text-zinc-200 focus:outline-none focus:border-emerald-500 font-mono text-xs"
                >
                  <option value="">-- No video initially (Physical hardware only) --</option>
                  {videos.map((v) => (
                    <option key={v.id} value={v.id}>
                      {v.filename} ({v.duration_seconds ? `${v.duration_seconds.toFixed(0)}s` : "Unknown duration"})
                    </option>
                  ))}
                </select>
              </div>

              <div>
                <label className="block text-zinc-300 font-medium mb-1">Location Hint</label>
                <input
                  type="text"
                  value={positionHint}
                  onChange={(e) => setPositionHint(e.target.value)}
                  placeholder="e.g. Northeast corner of Building B"
                  className="w-full bg-zinc-950 border border-zinc-800 rounded-lg p-2.5 text-zinc-200 focus:outline-none focus:border-emerald-500 text-xs"
                />
              </div>

              <div>
                <label className="block text-zinc-300 font-medium mb-1">Field of View Hint</label>
                <input
                  type="text"
                  value={fovHint}
                  onChange={(e) => setFovHint(e.target.value)}
                  placeholder="e.g. Covers pedestrian walkway and fence"
                  className="w-full bg-zinc-950 border border-zinc-800 rounded-lg p-2.5 text-zinc-200 focus:outline-none focus:border-emerald-500 text-xs"
                />
              </div>

              <div className="flex justify-end gap-3 pt-2">
                <button
                  type="button"
                  onClick={() => setIsRegisterOpen(false)}
                  className="px-4 py-2 bg-zinc-800 hover:bg-zinc-700 text-zinc-300 rounded-lg font-medium"
                >
                  Cancel
                </button>
                <button
                  type="submit"
                  disabled={isRegistering}
                  className="px-4 py-2 bg-emerald-600 hover:bg-emerald-500 text-zinc-950 font-bold rounded-lg transition-all"
                >
                  {isRegistering ? "Registering..." : "Save Camera"}
                </button>
              </div>
            </form>
          </div>
        </div>
      )}


    </div>
  );
}
