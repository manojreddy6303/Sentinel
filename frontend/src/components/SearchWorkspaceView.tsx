"use client";

import React, { useState, useEffect, useCallback } from "react";
import { ForensicSearchPanel } from "./ForensicSearchPanel";
import { listVideos, VideoListItem, Case } from "@/lib/api";

interface SearchWorkspaceViewProps {
  initialVideoId?: string;
  activeCase?: Case | null;
  onSeek?: (timestamp: number, eventId?: string) => void;
  onIngestVideo?: () => void;
}

export default function SearchWorkspaceView({
  initialVideoId,
  activeCase,
  onSeek,
  onIngestVideo,
}: SearchWorkspaceViewProps) {
  const [videos, setVideos] = useState<VideoListItem[]>([]);
  const [selectedVideoId, setSelectedVideoId] = useState<string>(initialVideoId || "");
  const [selectedVideoDuration, setSelectedVideoDuration] = useState<number | undefined>(undefined);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const fetchVideos = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await listVideos({ limit: 100 });
      const vids = res.videos || [];
      setVideos(vids);
      
      // If an initialVideoId was provided, resolve its duration
      if (initialVideoId) {
        const found = vids.find((v) => v.id === initialVideoId);
        if (found) {
          setSelectedVideoId(found.id);
          setSelectedVideoDuration(found.duration_seconds || undefined);
        }
      }
      // Note: We deliberately do NOT silently select vids[0] if no initialVideoId was provided.
      // The user chooses or searches within their intended video scope explicitly.
    } catch (err: any) {
      setError(err.message || "Failed to load surveillance footage list.");
    } finally {
      setLoading(false);
    }
  }, [initialVideoId]);

  useEffect(() => {
    fetchVideos();
  }, [fetchVideos]);

  // Update selection when initialVideoId changes from external navigation
  useEffect(() => {
    if (initialVideoId) {
      setSelectedVideoId(initialVideoId);
      const found = videos.find((v) => v.id === initialVideoId);
      if (found) {
        setSelectedVideoDuration(found.duration_seconds || undefined);
      }
    }
  }, [initialVideoId, videos]);

  const handleVideoChange = (vidId: string) => {
    setSelectedVideoId(vidId);
    const found = videos.find((v) => v.id === vidId);
    setSelectedVideoDuration(found?.duration_seconds || undefined);
  };

  const selectedVideo = videos.find((v) => v.id === selectedVideoId);

  return (
    <div className="p-4 md:p-6 lg:p-8 space-y-6 max-w-[1750px] 2xl:max-w-[2150px] mx-auto min-h-full">
      {/* Header & Scope Selector */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 border-b border-zinc-800/80 pb-5">
        <div>
          <div className="flex items-center gap-3">
            <h1 className="text-xl md:text-2xl font-bold tracking-tight text-white flex items-center gap-2.5">
              <span className="w-3 h-3 rounded-full bg-blue-400 animate-pulse" />
              Forensic &amp; Natural-Language Search
            </h1>
            <span className="px-2.5 py-0.5 rounded-full text-xs font-mono font-semibold bg-blue-500/10 text-blue-400 border border-blue-500/30">
              Deterministic + Grounded AI
            </span>
          </div>
          <p className="text-xs md:text-sm text-zinc-400 mt-1">
            Execute temporal queries, track investigations, and evidence discovery across ingested CCTV footage.
          </p>
        </div>

        {/* Video & Case Scope Controls */}
        <div className="flex flex-wrap items-center gap-3">
          {activeCase && (
            <div className="flex items-center gap-1.5 px-3 py-1.5 bg-zinc-900 border border-zinc-800 rounded-xl text-xs font-mono">
              <span className="text-zinc-500">Case Scope:</span>
              <span className="text-cyan-400 font-semibold">{activeCase.case_number || activeCase.id}</span>
            </div>
          )}

          <div className="flex items-center gap-2 bg-zinc-900 border border-zinc-800 p-1.5 rounded-xl">
            <label className="text-xs font-mono text-zinc-400 font-medium pl-2">Target Footage:</label>
            <select
              value={selectedVideoId}
              onChange={(e) => handleVideoChange(e.target.value)}
              className="bg-zinc-950 border border-zinc-700 text-zinc-200 text-xs rounded-lg px-3 py-1.5 focus:outline-none focus:border-blue-500 font-mono cursor-pointer max-w-[280px] truncate"
            >
              <option value="">-- Select Surveillance Footage --</option>
              {videos.map((v) => {
                const isCaseVideo = activeCase?.linked_videos?.some((lv) => lv.video_id === v.id);
                return (
                  <option key={v.id} value={v.id}>
                    {isCaseVideo ? "★ [Active Case] " : ""}
                    {v.filename} ({v.duration_seconds ? `${v.duration_seconds.toFixed(0)}s` : "--"})
                  </option>
                );
              })}
            </select>
          </div>

          {onIngestVideo && (
            <button
              onClick={onIngestVideo}
              className="px-3.5 py-2 bg-zinc-800 hover:bg-zinc-700 text-zinc-200 text-xs font-semibold rounded-xl border border-zinc-700 transition-colors cursor-pointer flex items-center gap-1.5"
            >
              <svg className="w-3.5 h-3.5 text-emerald-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-8l-4-4m0 0L8 8m4-4v12" />
              </svg>
              <span>+ Ingest Footage</span>
            </button>
          )}
        </div>
      </div>

      {/* Error display if API fails */}
      {error && (
        <div className="p-4 bg-red-500/10 border border-red-500/30 rounded-xl flex items-center justify-between text-xs text-red-400">
          <span>{error}</span>
          <button
            onClick={fetchVideos}
            className="px-3 py-1 bg-red-500/20 hover:bg-red-500/30 rounded font-semibold cursor-pointer"
          >
            Retry
          </button>
        </div>
      )}

      {/* Main Content Area */}
      {selectedVideoId ? (
        <div className="space-y-4">
          {/* Active Target Video Indicator */}
          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 px-4 py-2.5 bg-blue-950/20 border border-blue-900/40 rounded-xl text-xs">
            <div className="flex items-center gap-2 text-zinc-300 font-mono">
              <span className="w-2 h-2 rounded-full bg-blue-400" />
              <span>Active Search Scope:</span>
              <strong className="text-white">{selectedVideo?.filename || selectedVideoId}</strong>
              {selectedVideo?.duration_seconds && (
                <span className="text-zinc-500">&bull; Duration: {selectedVideo.duration_seconds.toFixed(1)}s</span>
              )}
              {selectedVideo?.fps && (
                <span className="text-zinc-500">&bull; {selectedVideo.fps} FPS</span>
              )}
            </div>
            <button
              onClick={() => setSelectedVideoId("")}
              className="text-blue-400 hover:underline font-mono text-[11px] cursor-pointer self-start sm:self-auto"
            >
              Change Target Footage &rarr;
            </button>
          </div>

          <ForensicSearchPanel
            videoId={selectedVideoId}
            videoDuration={selectedVideoDuration}
            onSeek={onSeek}
          />
        </div>
      ) : loading ? (
        <div className="flex flex-col items-center justify-center py-24 text-zinc-500 space-y-3">
          <div className="w-8 h-8 border-2 border-blue-500 border-t-transparent rounded-full animate-spin" />
          <span className="text-sm font-mono">Loading surveillance footage library...</span>
        </div>
      ) : (
        /* Empty / Selection Guide: Prompt user to pick a footage target or upload from laptop */
        <div className="space-y-6">
          <div className="rounded-2xl border border-zinc-800 bg-zinc-900/50 p-8 md:p-12 text-center space-y-4 max-w-3xl mx-auto shadow-xl">
            <div className="w-14 h-14 rounded-2xl bg-blue-500/10 border border-blue-500/20 flex items-center justify-center mx-auto text-blue-400">
              <svg className="w-7 h-7" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
              </svg>
            </div>
            <div className="space-y-1.5">
              <h3 className="text-lg font-bold text-white">Select Surveillance Footage for Forensic Search</h3>
              <p className="text-xs md:text-sm text-zinc-400 max-w-lg mx-auto leading-relaxed">
                Forensic and natural-language search operates directly against an ingested recording&apos;s timeline, detection tracks, and preserved evidence. Select a video below to begin querying.
              </p>
            </div>

            {onIngestVideo && (
              <div className="pt-2">
                <button
                  onClick={onIngestVideo}
                  className="px-5 py-2.5 bg-emerald-600 hover:bg-emerald-500 text-zinc-950 font-bold text-xs rounded-xl transition-all shadow-md inline-flex items-center gap-2 cursor-pointer"
                >
                  <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-8l-4-4m0 0L8 8m4-4v12" />
                  </svg>
                  <span>+ Ingest Surveillance Video</span>
                </button>
              </div>
            )}
          </div>

          {/* Quick Footage Selector Cards */}
          {videos.length > 0 && (
            <div className="space-y-3">
              <div className="text-xs font-mono text-zinc-400 uppercase tracking-wider flex items-center justify-between">
                <span>Available Surveillance Footage ({videos.length})</span>
                {activeCase && (
                  <span className="text-cyan-400">★ Highlighting active case footage</span>
                )}
              </div>
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3.5">
                {videos.map((v) => {
                  const isCaseVideo = activeCase?.linked_videos?.some((lv) => lv.video_id === v.id);
                  return (
                    <div
                      key={v.id}
                      onClick={() => handleVideoChange(v.id)}
                      className={`p-4 rounded-xl border transition-all cursor-pointer group flex flex-col justify-between space-y-3 ${
                        isCaseVideo
                          ? "bg-cyan-950/20 border-cyan-800/60 hover:border-cyan-500 hover:bg-cyan-950/40"
                          : "bg-zinc-900/60 border-zinc-800 hover:border-blue-500/60 hover:bg-zinc-900"
                      }`}
                    >
                      <div className="space-y-1.5">
                        <div className="flex items-center justify-between gap-2">
                          <span className="font-mono text-xs font-bold text-zinc-200 group-hover:text-white truncate" title={v.filename}>
                            {v.filename}
                          </span>
                          {isCaseVideo && (
                            <span className="text-[10px] font-mono px-1.5 py-0.2 bg-cyan-500/20 text-cyan-300 rounded border border-cyan-500/40 flex-none">
                              Active Case
                            </span>
                          )}
                        </div>
                        <div className="text-[11px] text-zinc-400 font-mono flex items-center gap-2">
                          <span>{v.duration_seconds ? `${v.duration_seconds.toFixed(1)}s` : "Unknown duration"}</span>
                          <span>&bull;</span>
                          <span>{v.fps || "--"} FPS</span>
                          <span>&bull;</span>
                          <span className="text-emerald-400">{v.status}</span>
                        </div>
                      </div>

                      <div className="flex items-center justify-between pt-2 border-t border-zinc-800/80 text-xs">
                        <span className="text-[10px] font-mono text-zinc-500">
                          ID: {v.id.slice(0, 10)}...
                        </span>
                        <span className="text-blue-400 font-medium group-hover:underline flex items-center gap-1 font-mono text-[11px]">
                          <span>Search Footage</span>
                          <span>&rarr;</span>
                        </span>
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
