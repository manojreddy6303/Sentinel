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
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 border-b border-[#2A3038] pb-5">
        <div>
          <div className="flex items-center gap-3">
            <h1 className="text-xl md:text-2xl font-bold tracking-tight text-[#F5F7FA]">
              Forensic Search
            </h1>
            <span className="px-2.5 py-0.5 rounded-full text-xs font-semibold bg-[#1D2128] text-[#19B89A] border border-[#2A3038]">
              Natural language &amp; timeline
            </span>
          </div>
          <p className="text-xs md:text-sm text-[#A7AFBA] mt-1">
            Search security events, person presence, and evidence across ingested CCTV footage.
          </p>
        </div>

        {/* Video & Case Scope Controls */}
        <div className="flex flex-wrap items-center gap-3">
          {activeCase && (
            <div className="flex items-center gap-1.5 px-3 py-1.5 bg-[#171A20] border border-[#2A3038] rounded-xl text-xs">
              <span className="text-[#737C87]">Case Scope:</span>
              <span className="text-[#19B89A] font-semibold">{activeCase.case_number || activeCase.id}</span>
            </div>
          )}

          <div className="flex items-center gap-2 bg-[#171A20] border border-[#2A3038] p-1.5 rounded-xl">
            <label className="text-xs text-[#A7AFBA] font-medium pl-2">Target Footage:</label>
            <select
              value={selectedVideoId}
              onChange={(e) => handleVideoChange(e.target.value)}
              className="bg-[#0F1115] border border-[#2A3038] text-[#F5F7FA] text-xs rounded-lg px-3 py-1.5 focus:outline-none focus:border-[#19B89A] cursor-pointer max-w-[280px] truncate"
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
              className="px-3.5 py-2 bg-[#1D2128] hover:bg-[#2A3038] text-[#F5F7FA] text-xs font-semibold rounded-xl border border-[#2A3038] transition-colors cursor-pointer flex items-center gap-1.5"
            >
              <svg className="w-3.5 h-3.5 text-[#19B89A]" fill="none" stroke="currentColor" viewBox="0 0 24 24">
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
          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 px-4 py-2.5 bg-[#171A20] border border-[#2A3038] rounded-xl text-xs">
            <div className="flex items-center gap-2 text-[#A7AFBA]">
              <span className="w-2 h-2 rounded-full bg-[#19B89A]" />
              <span>Active scope:</span>
              <strong className="text-[#F5F7FA]">{selectedVideo?.filename || selectedVideoId}</strong>
              {selectedVideo?.duration_seconds && (
                <span className="text-[#737C87]">&bull; Duration: {selectedVideo.duration_seconds.toFixed(1)}s</span>
              )}
              {selectedVideo?.fps && (
                <span className="text-[#737C87]">&bull; {selectedVideo.fps} FPS</span>
              )}
            </div>
            <button
              onClick={() => setSelectedVideoId("")}
              className="text-[#19B89A] hover:underline text-xs cursor-pointer self-start sm:self-auto"
            >
              Change target footage &rarr;
            </button>
          </div>

          <ForensicSearchPanel
            videoId={selectedVideoId}
            videoDuration={selectedVideoDuration}
            onSeek={onSeek}
          />
        </div>
      ) : loading ? (
        <div className="flex flex-col items-center justify-center py-24 text-[#737C87] space-y-3">
          <div className="w-8 h-8 border-2 border-[#19B89A] border-t-transparent rounded-full animate-spin" />
          <span className="text-sm">Loading surveillance footage library...</span>
        </div>
      ) : (
        /* Empty / Selection Guide */
        <div className="space-y-6">
          <div className="rounded-2xl border border-[#2A3038] bg-[#171A20] p-8 md:p-12 text-center space-y-4 max-w-3xl mx-auto shadow-sm">
            <div className="w-14 h-14 rounded-2xl bg-[#19B89A]/10 border border-[#19B89A]/20 flex items-center justify-center mx-auto text-[#19B89A]">
              <svg className="w-7 h-7" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
              </svg>
            </div>
            <div className="space-y-1.5">
              <h3 className="text-lg font-bold text-[#F5F7FA]">Select surveillance footage for forensic search</h3>
              <p className="text-xs md:text-sm text-[#A7AFBA] max-w-lg mx-auto leading-relaxed">
                Forensic and natural-language search operates directly against an ingested recording&apos;s timeline, detection tracks, and preserved evidence. Select a video below to begin querying.
              </p>
            </div>

            {onIngestVideo && (
              <div className="pt-2">
                <button
                  onClick={onIngestVideo}
                  className="px-5 py-2.5 bg-[#19B89A] hover:bg-[#159e84] text-[#0F1115] font-semibold text-xs rounded-xl transition-all shadow-sm inline-flex items-center gap-2 cursor-pointer"
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
              <div className="text-xs text-[#A7AFBA] flex items-center justify-between">
                <span>Available surveillance footage ({videos.length})</span>
                {activeCase && (
                  <span className="text-[#19B89A]">★ Highlighting active case footage</span>
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
                          ? "bg-[#171A20] border-[#19B89A] hover:shadow-md"
                          : "bg-[#171A20] border-[#2A3038] hover:border-[#3B434D]"
                      }`}
                    >
                      <div className="space-y-1.5">
                        <div className="flex items-center justify-between gap-2">
                          <span className="text-xs font-semibold text-[#F5F7FA] group-hover:text-white truncate" title={v.filename}>
                            {v.filename}
                          </span>
                          {isCaseVideo && (
                            <span className="text-[10px] font-mono px-1.5 py-0.5 bg-[#19B89A]/15 text-[#19B89A] rounded border border-[#19B89A]/30 flex-none">
                              Active Case
                            </span>
                          )}
                        </div>
                        <div className="text-[11px] text-[#A7AFBA] flex items-center gap-2">
                          <span>{v.duration_seconds ? `${v.duration_seconds.toFixed(1)}s` : "Unknown duration"}</span>
                          <span>&bull;</span>
                          <span>{v.fps || "--"} FPS</span>
                          <span>&bull;</span>
                          <span className="text-emerald-400">{v.status}</span>
                        </div>
                      </div>

                      <div className="flex items-center justify-between pt-2 border-t border-[#2A3038] text-xs">
                        <span className="text-[10px] font-mono text-[#737C87]">
                          ID: {v.id.slice(0, 10)}...
                        </span>
                        <span className="text-[#19B89A] font-medium group-hover:underline flex items-center gap-1 text-[11px]">
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
