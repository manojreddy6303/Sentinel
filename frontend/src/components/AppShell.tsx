"use client";

import React, { useState, useEffect, useCallback } from "react";
import {
  checkApiHealth,
  HealthResponse,
  getCases,
  getCase,
  Case,
  getAnalyticsSummary,
  AnalyticsSummary,
  listVideos,
  VideoListItem,
  getVideoMetadata,
  VideoMetadata,
} from "@/lib/api";
import CaseManagementView from "./CaseManagementView";
import CaseWorkspaceView from "./CaseWorkspaceView";
import VideoUpload from "./VideoUpload";
import MultiCameraSessionPanel from "./MultiCameraSessionPanel";
import CamerasWorkspaceView from "./CamerasWorkspaceView";
import IncidentsWorkbenchView from "./IncidentsWorkbenchView";
import EvidenceVaultView from "./EvidenceVaultView";
import ReportsWorkspaceView from "./ReportsWorkspaceView";
import AnalyticsWorkspaceView from "./AnalyticsWorkspaceView";
import SearchWorkspaceView from "./SearchWorkspaceView";
import SecurityOperationsView from "./SecurityOperationsView";

export type NavItem =
  | "investigate"
  | "cases"
  | "security-ops"
  | "system"
  | "dashboard"
  | "cameras"
  | "live"
  | "incidents"
  | "search"
  | "evidence"
  | "reports"
  | "analytics";

export default function AppShell() {
  const [activeNav, setActiveNav] = useState<NavItem>("investigate");
  const [viewingCaseId, setViewingCaseId] = useState<string | null>(null);
  const [activeCase, setActiveCase] = useState<Case | null>(null);
  const [currentVideoId, setCurrentVideoId] = useState<string | null>(null);
  const [currentVideoName, setCurrentVideoName] = useState<string | null>(null);
  const [currentVideoMeta, setCurrentVideoMeta] = useState<VideoMetadata | null>(null);
  const [sidebarCollapsed, setSidebarCollapsed] = useState<boolean>(false);
  const [healthStatus, setHealthStatus] = useState<HealthResponse | null>(null);
  const [healthError, setHealthError] = useState<boolean>(false);
  const [cases, setCases] = useState<Case[]>([]);
  const [loadingCases, setLoadingCases] = useState<boolean>(false);
  const [casesError, setCasesError] = useState<string | null>(null);
  const [analytics, setAnalytics] = useState<AnalyticsSummary | null>(null);
  const [loadingAnalytics, setLoadingAnalytics] = useState<boolean>(false);
  const [recentVideos, setRecentVideos] = useState<VideoListItem[]>([]);
  const [loadingVideos, setLoadingVideos] = useState<boolean>(false);
  const [inspectingVideoId, setInspectingVideoId] = useState<string | null>(null);
  const [isFootageModalOpen, setIsFootageModalOpen] = useState<boolean>(false);

  // Poll backend health status
  useEffect(() => {
    checkApiHealth()
      .then((res) => {
        setHealthStatus(res);
        setHealthError(false);
      })
      .catch(() => setHealthError(true));

    const healthInterval = setInterval(() => {
      checkApiHealth()
        .then((res) => {
          setHealthStatus(res);
          setHealthError(false);
        })
        .catch(() => setHealthError(true));
    }, 20000);

    return () => {
      clearInterval(healthInterval);
    };
  }, []);

  // Fetch dashboard summary metrics and available video footage
  const loadDashboardData = useCallback(async () => {
    setLoadingCases(true);
    setLoadingAnalytics(true);
    setLoadingVideos(true);
    setCasesError(null);
    try {
      const [caseData, analyticsData, videosData] = await Promise.allSettled([
        getCases(),
        getAnalyticsSummary(),
        listVideos({ limit: 50 }),
      ]);
      if (caseData.status === "fulfilled") {
        setCases(caseData.value || []);
      } else {
        setCasesError(caseData.reason?.message || "Failed to load security cases.");
      }
      if (analyticsData.status === "fulfilled") {
        setAnalytics(analyticsData.value);
      }
      if (videosData.status === "fulfilled") {
        setRecentVideos(videosData.value?.videos || []);
      }
    } catch (err: any) {
      setCasesError(err.message || "Failed to load dashboard data.");
    } finally {
      setLoadingCases(false);
      setLoadingAnalytics(false);
      setLoadingVideos(false);
    }
  }, []);

  // Refresh platform data on mount and whenever navigating
  useEffect(() => {
    loadDashboardData();
  }, [activeNav, loadDashboardData]);

  // Open a specific case: sets global activeCase and opens case workspace view
  const handleOpenCase = useCallback(async (caseId: string) => {
    setViewingCaseId(caseId);
    setActiveNav("cases");
    setInspectingVideoId(null);
    try {
      const c = await getCase(caseId);
      setActiveCase(c);
    } catch (err) {
      console.error("Failed to load case details:", err);
    }
  }, []);

  // Close the active case session
  const handleCloseActiveCase = () => {
    setActiveCase(null);
    setViewingCaseId(null);
  };

  // Fetch active video metadata when currentVideoId changes
  useEffect(() => {
    if (currentVideoId && currentVideoId !== "new") {
      getVideoMetadata(currentVideoId)
        .then((meta) => {
          setCurrentVideoMeta(meta);
          if (meta.filename) setCurrentVideoName(meta.filename);
        })
        .catch(() => {});
    } else {
      setCurrentVideoMeta(null);
    }
  }, [currentVideoId]);

  // Case / Video consistency audit:
  // If activeCase does not contain the active video, clear the stale case context immediately.
  useEffect(() => {
    if (currentVideoId && currentVideoId !== "new" && activeCase) {
      const isVideoInCase = activeCase.linked_videos?.some((lv) => lv.video_id === currentVideoId);
      if (!isVideoInCase) {
        setActiveCase(null);
      }
    }
  }, [currentVideoId, activeCase]);

  // Open direct video investigation workstation
  const handleOpenVideoInvestigation = (videoId: string, filename?: string) => {
    setInspectingVideoId(videoId);
    setCurrentVideoId(videoId);
    if (filename) setCurrentVideoName(filename);
    setIsFootageModalOpen(false);
    if (activeCase && !activeCase.linked_videos?.some((lv) => lv.video_id === videoId)) {
      setActiveCase(null);
    }
    getVideoMetadata(videoId)
      .then((m) => {
        setCurrentVideoMeta(m);
        if (m.filename) setCurrentVideoName(m.filename);
      })
      .catch(() => {});
  };

  // Open direct surveillance video ingestion (upload from laptop)
  const handleOpenIngestion = () => {
    setInspectingVideoId("new");
    setActiveNav("investigate");
    setIsFootageModalOpen(false);
  };

  // Handle switching navigation: clears sub-inspection state
  const handleNavChange = (nav: NavItem) => {
    setActiveNav(nav);
    setInspectingVideoId(null);
  };

  // Find the active demo burglary video (latest processed)
  const demoBurglaryVideo =
    recentVideos.find((v) => v.filename.toLowerCase().includes("burglary") && v.status === "processed") ||
    recentVideos.find((v) => v.filename.toLowerCase().includes("burglary")) ||
    recentVideos[0];

  return (
    <div className="w-screen h-screen overflow-hidden flex flex-col bg-[#0F1115] text-[#F5F7FA] select-none font-sans">
      {/* ── TOP HEADER ── */}
      <header className="h-14 flex-none border-b border-[#2A3038] bg-[#171A20] px-5 flex items-center justify-between gap-4 z-40">
        {/* Brand & Subtitle */}
        <div className="flex items-center gap-3">
          <div
            onClick={() => {
              setActiveNav("investigate");
              setInspectingVideoId(null);
            }}
            className="flex items-center gap-2 cursor-pointer group"
          >
            <span className="w-2.5 h-2.5 rounded-full bg-[#19B89A] transition-transform group-hover:scale-125" />
            <span className="font-bold tracking-wider text-base text-[#F5F7FA]">SENTINEL</span>
          </div>
          <span className="hidden sm:inline-block text-xs text-[#737C87] border-l border-[#2A3038] pl-3">
            AI Security Investigation
          </span>
        </div>

        {/* Center: Active Case Context (only shown if case contains active video or on case views) */}
        <div className="flex-1 max-w-lg mx-4 hidden md:flex items-center justify-center">
          {activeCase && (!currentVideoId || activeCase.linked_videos?.some((lv) => lv.video_id === currentVideoId)) ? (
            <div className="flex items-center gap-2 bg-[#1D2128] border border-[#2A3038] rounded-md px-3 py-1 text-xs">
              <span className="text-[#737C87]">Case:</span>
              <span className="font-mono font-semibold text-[#19B89A]">
                {activeCase.case_number || activeCase.case_id || activeCase.id}
              </span>
              <span className="text-[#A7AFBA] truncate max-w-[180px]">{activeCase.title}</span>
              <button
                onClick={() => {
                  setViewingCaseId(activeCase.id);
                  setActiveNav("cases");
                  setInspectingVideoId(null);
                }}
                className="text-xs text-[#19B89A] hover:underline ml-1 cursor-pointer font-medium"
              >
                Workspace &rarr;
              </button>
              <button
                onClick={handleCloseActiveCase}
                className="text-[#737C87] hover:text-[#F5F7FA] ml-1 text-sm cursor-pointer p-0.5"
                title="Close active case"
              >
                &times;
              </button>
            </div>
          ) : null}
        </div>

        {/* Right Status */}
        <div className="flex items-center gap-3">
          <div className="flex items-center gap-2 text-xs font-sans">
            {healthStatus ? (
              <span className="flex items-center gap-1.5 text-[#19B89A]">
                <span className="w-2 h-2 rounded-full bg-[#19B89A]" />
                <span className="text-[#A7AFBA]">System online</span>
              </span>
            ) : healthError ? (
              <span className="flex items-center gap-1.5 text-rose-400">
                <span className="w-2 h-2 rounded-full bg-rose-500" />
                <span>System offline</span>
              </span>
            ) : (
              <span className="text-[#737C87] text-xs">Connecting...</span>
            )}
          </div>
        </div>
      </header>

      {/* ── MAIN WORKSPACE CONTAINER ── */}
      <div className="flex-1 flex overflow-hidden">
        {/* ── SIDEBAR NAVIGATION ── */}
        <aside
          className={`flex-none border-r border-[#2A3038] bg-[#171A20] transition-all duration-200 flex flex-col justify-between ${
            sidebarCollapsed ? "w-16" : "w-56"
          }`}
        >
          {/* Quiet Primary Navigation Items */}
          <div className="p-2 space-y-1">
            {/* 1. Investigate */}
            <button
              onClick={() => handleNavChange("investigate")}
              title={sidebarCollapsed ? "Investigate" : undefined}
              className={`w-full flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm transition-colors text-left cursor-pointer ${
                activeNav === "investigate"
                  ? "bg-[#19B89A]/15 text-[#19B89A] font-semibold border-l-2 border-[#19B89A]"
                  : "text-[#A7AFBA] hover:text-[#F5F7FA] hover:bg-[#1D2128] border-l-2 border-transparent"
              }`}
            >
              <svg className="w-4 h-4 shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
              </svg>
              {!sidebarCollapsed && <span className="truncate">Investigate</span>}
            </button>

            {/* 2. Cases */}
            <button
              onClick={() => handleNavChange("cases")}
              title={sidebarCollapsed ? "Cases" : undefined}
              className={`w-full flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm transition-colors text-left cursor-pointer ${
                activeNav === "cases"
                  ? "bg-[#19B89A]/15 text-[#19B89A] font-semibold border-l-2 border-[#19B89A]"
                  : "text-[#A7AFBA] hover:text-[#F5F7FA] hover:bg-[#1D2128] border-l-2 border-transparent"
              }`}
            >
              <svg className="w-4 h-4 shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 8h14M5 8a2 2 0 110-4h14a2 2 0 110 4M5 8v10a2 2 0 002 2h10a2 2 0 002-2V8m-9 4h4" />
              </svg>
              {!sidebarCollapsed && <span className="truncate">Cases</span>}
            </button>

            {/* 3. Security Operations */}
            <button
              onClick={() => handleNavChange("security-ops")}
              title={sidebarCollapsed ? "Security Operations" : undefined}
              className={`w-full flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm transition-colors text-left cursor-pointer ${
                activeNav === "security-ops"
                  ? "bg-[#19B89A]/15 text-[#19B89A] font-semibold border-l-2 border-[#19B89A]"
                  : "text-[#A7AFBA] hover:text-[#F5F7FA] hover:bg-[#1D2128] border-l-2 border-transparent"
              }`}
            >
              <svg className="w-4 h-4 shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m5.618-4.016A11.955 11.955 0 0112 2.944a11.955 11.955 0 01-8.618 3.04A12.02 12.02 0 003 9c0 5.591 3.824 10.29 9 11.622 5.176-1.332 9-6.03 9-11.622 0-1.042-.133-2.052-.382-3.016z" />
              </svg>
              {!sidebarCollapsed && (
                <div className="flex items-center justify-between w-full">
                  <span className="truncate">Security Operations</span>
                  <span className="text-[10px] text-[#737C87] font-mono">Demo</span>
                </div>
              )}
            </button>

            {/* 4. System Health */}
            <button
              onClick={() => handleNavChange("system")}
              title={sidebarCollapsed ? "System Health" : undefined}
              className={`w-full flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm transition-colors text-left cursor-pointer ${
                activeNav === "system" || activeNav === "analytics"
                  ? "bg-[#19B89A]/15 text-[#19B89A] font-semibold border-l-2 border-[#19B89A]"
                  : "text-[#A7AFBA] hover:text-[#F5F7FA] hover:bg-[#1D2128] border-l-2 border-transparent"
              }`}
            >
              <svg className="w-4 h-4 shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
              </svg>
              {!sidebarCollapsed && <span className="truncate">System Health</span>}
            </button>
          </div>

          {/* Bottom Sidebar Collapse */}
          <div className="p-3 border-t border-[#2A3038]">
            <button
              onClick={() => setSidebarCollapsed(!sidebarCollapsed)}
              className="w-full py-1.5 px-2 rounded text-xs text-[#737C87] hover:text-[#F5F7FA] hover:bg-[#1D2128] flex items-center justify-center gap-2 cursor-pointer transition-colors"
              title={sidebarCollapsed ? "Expand sidebar" : "Collapse sidebar"}
            >
              <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                {sidebarCollapsed ? (
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 5l7 7-7 7M5 5l7 7-7 7" />
                ) : (
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M11 19l-7-7 7-7m8 14l-7-7 7-7" />
                )}
              </svg>
              {!sidebarCollapsed && <span>Collapse</span>}
            </button>
          </div>
        </aside>

        {/* ── WORKSPACE CONTENT REGION ── */}
        <main className="flex-1 overflow-y-auto overflow-x-hidden p-6 md:p-8 bg-[#0F1115]">
          {/* Sub-Inspection Mode: Active Investigation Workspace */}
          {inspectingVideoId ? (
            <div className="w-full space-y-5 max-w-[1600px] mx-auto">
              {/* TOP CONTEXT HEADER */}
              <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 p-5 bg-[#171A20] border border-[#2A3038] rounded-xl shadow-sm">
                <div className="flex items-center gap-4 flex-wrap">
                  <button
                    onClick={() => setInspectingVideoId(null)}
                    className="px-4 py-2 bg-[#1D2128] hover:bg-[#2A3038] text-[#F5F7FA] text-sm font-semibold rounded-lg border border-[#2A3038] flex items-center gap-2 transition-colors cursor-pointer"
                  >
                    <span>&larr;</span>
                    <span>Investigations</span>
                  </button>

                  <div className="h-8 w-px bg-[#2A3038] hidden sm:block" />

                  <div>
                    <div className="text-xs uppercase tracking-wider font-semibold text-[#737C87]">
                      SENTINEL INVESTIGATION
                    </div>
                    {/* When a video is selected, uploaded, or analyzed, ALWAYS show context-appropriate title + real metadata */}
                    {(currentVideoId && currentVideoId !== "new") || (inspectingVideoId && inspectingVideoId !== "new") ? (
                      <div className="space-y-1 mt-0.5">
                        <div className="flex items-center gap-3 flex-wrap">
                          <h2 className="text-xl md:text-2xl font-bold text-[#F5F7FA] tracking-tight">
                            {(() => {
                              const name = (currentVideoName || currentVideoMeta?.filename || currentVideoId || inspectingVideoId || "").toLowerCase();
                              if (name.includes("burglary")) return "Burglary investigation";
                              if (name.includes("theft")) return "Theft investigation";
                              if (name.includes("robbery")) return "Robbery investigation";
                              if (name.includes("assault")) return "Assault investigation";
                              if (name.includes("arrest")) return "Arrest investigation";
                              if (name.includes("perimeter") || name.includes("intrusi")) return "Perimeter security investigation";
                              if (name.includes("cam") || name.includes("camera")) {
                                const camMatch = name.match(/cam(?:era)?[_-]?(\d+)/i);
                                return camMatch ? `Camera ${camMatch[1]} investigation` : "Surveillance camera investigation";
                              }
                              const clean = (currentVideoName || currentVideoMeta?.filename || currentVideoId || inspectingVideoId || "")
                                .replace(/^[a-f0-9-]{36}_/i, "")
                                .replace(/\.[a-zA-Z0-9]+$/, "")
                                .replace(/[_-]+/g, " ")
                                .trim();
                              if (clean) {
                                return clean.charAt(0).toUpperCase() + clean.slice(1).toLowerCase() + " investigation";
                              }
                              return "Security investigation";
                            })()}
                          </h2>
                          <span className="px-2.5 py-0.5 rounded text-xs font-bold uppercase tracking-wider bg-[#19B89A]/15 text-[#19B89A] border border-[#19B89A]/30">
                            {currentVideoMeta?.status === "processed" || currentVideoMeta?.status === "completed" ? "ANALYZED" : (currentVideoMeta?.status?.toUpperCase() || "ANALYZED")}
                          </span>
                        </div>
                        <div className="flex items-center gap-2 flex-wrap text-sm text-[#A7AFBA]">
                          <span className="font-mono text-[#F5F7FA]">
                            {(currentVideoName || currentVideoMeta?.filename || currentVideoId || inspectingVideoId || "").replace(/^[a-f0-9-]{36}_/i, "")}
                          </span>
                          {currentVideoMeta?.duration_seconds && currentVideoMeta.duration_seconds > 0 ? (
                            <>
                              <span className="text-[#737C87]">&bull;</span>
                              <span>
                                {Math.floor(currentVideoMeta.duration_seconds / 60)}m {Math.round(currentVideoMeta.duration_seconds % 60)}s
                              </span>
                            </>
                          ) : null}
                          {currentVideoMeta?.fps ? (
                            <>
                              <span className="text-[#737C87]">&bull;</span>
                              <span>{Math.round(currentVideoMeta.fps)} FPS</span>
                            </>
                          ) : null}
                        </div>
                      </div>
                    ) : (
                      <div className="flex items-center gap-2.5 mt-0.5">
                        <span className="w-2.5 h-2.5 rounded-full bg-[#19B89A] animate-pulse" />
                        <h2 className="text-xl font-bold text-[#F5F7FA]">
                          New security footage investigation
                        </h2>
                      </div>
                    )}
                  </div>
                </div>

                <div className="flex items-center gap-3">
                  {activeCase && (!currentVideoId || activeCase.linked_videos?.some((lv) => lv.video_id === currentVideoId)) ? (
                    <div className="flex items-center gap-2.5 text-sm bg-[#0F1115] px-4 py-2 rounded-lg border border-[#2A3038]">
                      <span className="text-[#737C87]">Case:</span>
                      <span className="text-[#19B89A] font-semibold">{activeCase.case_number || activeCase.id}</span>
                      <button
                        onClick={() => {
                          setViewingCaseId(activeCase.id);
                          setActiveNav("cases");
                          setInspectingVideoId(null);
                        }}
                        className="text-sm text-[#19B89A] hover:underline cursor-pointer ml-1 font-medium"
                      >
                        Workspace &rarr;
                      </button>
                    </div>
                  ) : (
                    <button
                      onClick={() => {
                        setViewingCaseId(null);
                        setActiveNav("cases");
                      }}
                      className="px-4 py-2 bg-[#19B89A] hover:bg-[#16A489] text-[#0F1115] text-sm font-bold rounded-lg cursor-pointer transition-colors shadow-sm"
                    >
                      Create Case
                    </button>
                  )}
                </div>
              </div>

              <VideoUpload
                key={inspectingVideoId || "new"}
                initialVideoId={inspectingVideoId === "new" ? undefined : inspectingVideoId}
                activeCaseId={activeCase?.id || null}
                onVideoLoaded={(vid, fname) => {
                  setCurrentVideoId(vid);
                  setInspectingVideoId(vid);
                  if (fname) setCurrentVideoName(fname);
                  getVideoMetadata(vid)
                    .then((m) => {
                      setCurrentVideoMeta(m);
                      if (m.filename) setCurrentVideoName(m.filename);
                    })
                    .catch(() => {});
                }}
                onNavigateToCase={(caseId) => {
                  handleOpenCase(caseId);
                }}
                onNavigateToIncidents={() => {
                  setInspectingVideoId(null);
                  setActiveNav("incidents");
                }}
                onNavigateToEvidence={() => {
                  setInspectingVideoId(null);
                  setActiveNav("evidence");
                }}
              />
            </div>
          ) : (
            <>
              {/* 1. VIEW: INVESTIGATE HUB / CLEAN LANDING DASHBOARD */}
              {(activeNav === "investigate" || activeNav === "dashboard") && (
                <div className="w-full max-w-[1280px] mx-auto space-y-10 animate-in fade-in duration-200 py-4">
                  {/* HERO SECTION */}
                  <div className="relative overflow-hidden rounded-2xl bg-[#171A20] border border-[#2A3038] p-8 md:p-12 shadow-sm">
                    <div className="space-y-4 max-w-3xl">
                      <div className="inline-block text-xs font-semibold uppercase tracking-wider text-[#19B89A]">
                        AI Security Investigation
                      </div>

                      <h1 className="text-3xl md:text-4xl lg:text-5xl font-extrabold text-[#F5F7FA] tracking-tight leading-tight">
                        Turn security footage into evidence-backed investigations.
                      </h1>

                      <p className="text-base md:text-lg text-[#A7AFBA] leading-relaxed font-normal">
                        Analyze surveillance footage, reconstruct events, correlate evidence, and investigate incidents with AI-assisted reasoning.
                      </p>

                      <div className="flex flex-wrap items-center gap-3 pt-4">
                        <button
                          onClick={handleOpenIngestion}
                          className="px-6 py-3 bg-[#19B89A] hover:bg-[#16A489] text-[#0F1115] font-semibold text-sm rounded-lg transition-colors flex items-center gap-2 cursor-pointer shadow-sm"
                        >
                          <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2.5} d="M12 4v16m8-8H4" />
                          </svg>
                          <span>Start New Investigation</span>
                        </button>

                        <button
                          onClick={() => {
                            setViewingCaseId(null);
                            setActiveNav("cases");
                          }}
                          className="px-5 py-3 bg-[#1D2128] hover:bg-[#2A3038] text-[#F5F7FA] font-medium text-sm rounded-lg border border-[#2A3038] transition-colors cursor-pointer"
                        >
                          Open Existing Case
                        </button>

                        {demoBurglaryVideo && (
                          <button
                            onClick={() => {
                              handleOpenVideoInvestigation(demoBurglaryVideo.id, demoBurglaryVideo.filename);
                            }}
                            className="px-4 py-3 text-xs text-[#A7AFBA] hover:text-[#F5F7FA] hover:bg-[#1D2128] rounded-lg transition-colors cursor-pointer flex items-center gap-1.5"
                          >
                            <span>Quick Demo: {demoBurglaryVideo.filename} &rarr;</span>
                          </button>
                        )}
                      </div>
                    </div>
                  </div>

                  {/* FOUR CLEAN METRICS (PLATFORM-WIDE) */}
                  <div className="space-y-2">
                    <div className="flex items-center justify-between text-xs font-semibold uppercase tracking-wider text-[#737C87] px-1">
                      <span>Platform-Wide Overview (All Stored Footage)</span>
                      <span className="text-[#A7AFBA] font-normal normal-case">Global Platform Totals</span>
                    </div>
                    <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                      <div className="p-5 rounded-xl bg-[#171A20] border border-[#2A3038] space-y-1">
                        <div className="text-xs text-[#737C87]">Platform Footage</div>
                        <div className="text-3xl font-bold text-[#F5F7FA]">
                          {analytics?.total_videos ?? recentVideos.length}
                        </div>
                        <div className="text-xs text-[#A7AFBA]">Processed video records</div>
                      </div>

                      <div className="p-5 rounded-xl bg-[#171A20] border border-[#2A3038] space-y-1">
                        <div className="text-xs text-[#737C87]">Platform Incidents</div>
                        <div className="text-3xl font-bold text-[#F5F7FA]">
                          {analytics?.total_correlated_incidents ?? 0}
                        </div>
                        <div className="text-xs text-[#A7AFBA]">Total platform findings</div>
                      </div>

                      <div className="p-5 rounded-xl bg-[#171A20] border border-[#2A3038] space-y-1">
                        <div className="text-xs text-[#737C87]">Platform Evidence</div>
                        <div className="text-3xl font-bold text-[#F5F7FA]">
                          {analytics?.validated_evidence_count ?? analytics?.total_evidence ?? 0}
                        </div>
                        <div className="text-xs text-[#A7AFBA]">Total preserved artifacts</div>
                      </div>

                      <div className="p-5 rounded-xl bg-[#171A20] border border-[#2A3038] space-y-1">
                        <div className="text-xs text-[#737C87]">Platform Cases</div>
                        <div className="text-3xl font-bold text-[#F5F7FA]">
                          {analytics?.total_cases ?? cases.length}
                        </div>
                        <div className="text-xs text-[#A7AFBA]">Active security dossiers</div>
                      </div>
                    </div>
                  </div>

                  {/* VISUAL WORKFLOW EXPLAINER */}
                  <div className="rounded-xl border border-[#2A3038] bg-[#171A20] p-6 space-y-4">
                    <div className="text-xs font-semibold uppercase tracking-wider text-[#737C87]">
                      Investigation Workflow
                    </div>

                    <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-5 gap-3">
                      <div className="p-3.5 rounded-lg bg-[#1D2128] border border-[#2A3038] space-y-1 min-w-0">
                        <div className="text-xs text-[#19B89A] font-semibold">Step 1</div>
                        <div className="text-sm font-semibold text-[#F5F7FA] truncate">Upload footage</div>
                        <div className="text-xs text-[#737C87] leading-snug">CCTV or surveillance video</div>
                      </div>

                      <div className="p-3.5 rounded-lg bg-[#1D2128] border border-[#2A3038] space-y-1 min-w-0">
                        <div className="text-xs text-[#19B89A] font-semibold">Step 2</div>
                        <div className="text-sm font-semibold text-[#F5F7FA] truncate">Detect &amp; understand</div>
                        <div className="text-xs text-[#737C87] leading-snug">Objects, movement &amp; tracks</div>
                      </div>

                      <div className="p-3.5 rounded-lg bg-[#1D2128] border border-[#2A3038] space-y-1 min-w-0">
                        <div className="text-xs text-[#19B89A] font-semibold">Step 3</div>
                        <div className="text-sm font-semibold text-[#F5F7FA] truncate">Correlate evidence</div>
                        <div className="text-xs text-[#737C87] leading-snug">Snapshots, clips &amp; timestamps</div>
                      </div>

                      <div className="p-3.5 rounded-lg bg-[#1D2128] border border-[#2A3038] space-y-1 min-w-0">
                        <div className="text-xs text-[#19B89A] font-semibold">Step 4</div>
                        <div className="text-sm font-semibold text-[#F5F7FA] truncate">Investigate</div>
                        <div className="text-xs text-[#737C87] leading-snug">Natural language reasoning</div>
                      </div>

                      <div className="p-3.5 rounded-lg bg-[#1D2128] border border-[#2A3038] space-y-1 min-w-0">
                        <div className="text-xs text-[#19B89A] font-semibold">Step 5</div>
                        <div className="text-sm font-semibold text-[#F5F7FA] truncate" title="Explain findings">Explain findings</div>
                        <div className="text-xs text-[#737C87] leading-snug">Evidence-backed case dossier</div>
                      </div>
                    </div>
                  </div>

                  {/* SECONDARY ACCESS: PREVIOUSLY INGESTED FOOTAGE */}
                  {recentVideos.length > 0 && (
                    <div className="flex items-center justify-between text-xs text-[#737C87] pt-2">
                      <span>{recentVideos.length} surveillance footage files available in storage.</span>
                      <button
                        onClick={() => setIsFootageModalOpen(true)}
                        className="text-[#19B89A] hover:underline cursor-pointer font-medium"
                      >
                        Browse footage library &rarr;
                      </button>
                    </div>
                  )}

                  {/* SAFETY & INTEGRITY SAFEGUARDS */}
                  <div className="rounded-xl border border-[#2A3038] bg-[#171A20]/60 p-5 space-y-2">
                    <div className="font-semibold text-[#F5F7FA] flex items-center gap-2 text-sm">
                      <span className="w-2 h-2 rounded-full bg-[#19B89A]" />
                      Safety &amp; Integrity Safeguards
                    </div>
                    <p className="leading-relaxed text-[#A7AFBA] text-xs">
                      Sentinel operates as an investigative decision-support assistant. Final assessment scores &le; 0.65 require human confirmation. All object tracking is strictly anonymous; facial recognition, biometric matching, and identity attribution are prohibited.
                    </p>
                  </div>
                </div>
              )}

              {/* 2. VIEW: CASES */}
              {activeNav === "cases" && (
                <div className="w-full">
                  {viewingCaseId ? (
                    <CaseWorkspaceView
                      caseId={viewingCaseId}
                      onBack={() => {
                        setViewingCaseId(null);
                        loadDashboardData();
                      }}
                      onSelectCase={(newId: string) => handleOpenCase(newId)}
                      onUploadVideoForCase={() => {
                        handleOpenIngestion();
                      }}
                      onInvestigateVideo={(vId: string) => handleOpenVideoInvestigation(vId)}
                    />
                  ) : (
                    <CaseManagementView
                      onOpenCase={(id: string) => handleOpenCase(id)}
                      onSelectCase={(id: string) => handleOpenCase(id)}
                      onDataChanged={loadDashboardData}
                    />
                  )}
                </div>
              )}

              {/* 3. VIEW: SECURITY OPERATIONS (Cybersecurity & Physical Correlation) */}
              {activeNav === "security-ops" && (
                <div className="w-full">
                  <SecurityOperationsView
                    onInvestigateVideo={(videoId: string) => handleOpenVideoInvestigation(videoId)}
                    onNavigateToCase={(caseId: string) => handleOpenCase(caseId)}
                  />
                </div>
              )}

              {/* 4. VIEW: SYSTEM HEALTH (Platform Diagnostics & Analytics) */}
              {(activeNav === "system" || activeNav === "analytics") && (
                <div className="w-full">
                  <AnalyticsWorkspaceView />
                </div>
              )}

              {/* FALLBACK PRESERVED VIEWS FOR DEEP LINKS AND CALLBACK HANDLERS */}
              {activeNav === "cameras" && (
                <div className="w-full">
                  <CamerasWorkspaceView
                    onInvestigateVideo={(videoId: string) => handleOpenVideoInvestigation(videoId)}
                    onIngestVideo={handleOpenIngestion}
                    onCreateCaseFromVideo={() => {
                      setViewingCaseId(null);
                      setActiveNav("cases");
                    }}
                    onOpenMultiCamera={() => setActiveNav("live")}
                    onDataChanged={loadDashboardData}
                  />
                </div>
              )}

              {activeNav === "live" && (
                <div className="w-full">
                  <MultiCameraSessionPanel />
                </div>
              )}

              {activeNav === "incidents" && (
                <div className="w-full">
                  <IncidentsWorkbenchView
                    onInvestigateVideo={(videoId: string) => handleOpenVideoInvestigation(videoId)}
                    onOpenReplay={(caseId: string) => {
                      handleOpenCase(caseId);
                    }}
                  />
                </div>
              )}

              {activeNav === "search" && (
                <div className="w-full">
                  <SearchWorkspaceView
                    initialVideoId={currentVideoId || inspectingVideoId || undefined}
                    activeCase={activeCase}
                    onSeek={() => {}}
                    onIngestVideo={handleOpenIngestion}
                  />
                </div>
              )}

              {activeNav === "evidence" && (
                <div className="w-full">
                  <EvidenceVaultView
                    onInvestigateVideo={(videoId: string) => setInspectingVideoId(videoId)}
                  />
                </div>
              )}

              {activeNav === "reports" && (
                <div className="w-full">
                  <ReportsWorkspaceView />
                </div>
              )}
            </>
          )}
        </main>
      </div>

      {/* FOOTAGE SELECTION MODAL */}
      {isFootageModalOpen && (
        <div className="fixed inset-0 z-50 bg-black/75 backdrop-blur-sm flex items-center justify-center p-4">
          <div className="bg-[#171A20] border border-[#2A3038] rounded-xl max-w-2xl w-full p-6 space-y-4 shadow-xl">
            <div className="flex items-center justify-between border-b border-[#2A3038] pb-3">
              <h3 className="text-base font-bold text-[#F5F7FA]">Available Surveillance Footage</h3>
              <button
                onClick={() => setIsFootageModalOpen(false)}
                className="text-[#737C87] hover:text-[#F5F7FA] text-lg cursor-pointer"
              >
                &times;
              </button>
            </div>

            <div className="space-y-2 max-h-96 overflow-y-auto pr-1">
              {[...recentVideos]
                .sort((a, b) => {
                  const timeA = new Date(a.uploaded_at || 0).getTime();
                  const timeB = new Date(b.uploaded_at || 0).getTime();
                  return timeB - timeA;
                })
                .map((video) => {
                  const isBurglaryActive = video.id === demoBurglaryVideo?.id;
                  return (
                    <div
                      key={video.id}
                      onClick={() => handleOpenVideoInvestigation(video.id, video.filename)}
                      className={`p-3.5 rounded-lg border cursor-pointer flex items-center justify-between transition-colors ${
                        isBurglaryActive
                          ? "bg-[#14231E] border-[#19B89A]/50 hover:border-[#19B89A]"
                          : "bg-[#1D2128] border-[#2A3038] hover:border-[#19B89A]"
                      }`}
                    >
                      <div className="min-w-0">
                        <div className="flex items-center gap-2">
                          <span className="text-sm font-semibold text-[#F5F7FA] truncate">{video.filename}</span>
                          {isBurglaryActive && (
                            <span className="px-2 py-0.5 rounded text-[10px] font-mono font-semibold bg-[#19B89A]/20 text-[#19B89A] border border-[#19B89A]/40 shrink-0">
                              LATEST PROCESSED
                            </span>
                          )}
                        </div>
                        <div className="text-xs text-[#737C87] font-mono mt-0.5">
                          {video.incidents_count ? `${video.incidents_count} incidents` : "0 incidents"} &bull;{" "}
                          {video.evidence_count ? `${video.evidence_count} evidence` : "0 evidence"} &bull;{" "}
                          {video.duration_seconds ? `${Math.round(video.duration_seconds)}s` : "Stored"}
                        </div>
                      </div>
                      <span className="text-xs font-semibold text-[#19B89A] shrink-0 ml-4">
                        Open &rarr;
                      </span>
                    </div>
                  );
                })}
            </div>

            <div className="pt-3 border-t border-[#2A3038] flex justify-end">
              <button
                onClick={() => setIsFootageModalOpen(false)}
                className="px-4 py-2 bg-[#1D2128] hover:bg-[#2A3038] text-[#F5F7FA] rounded-lg text-xs font-medium cursor-pointer"
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
