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

export type NavItem =
  | "dashboard"
  | "cases"
  | "cameras"
  | "live"
  | "incidents"
  | "search"
  | "evidence"
  | "reports"
  | "analytics";

interface NavGroup {
  name: string;
  items: Array<{
    id: NavItem;
    label: string;
    description: string;
    icon: React.ReactNode;
    badge?: string | number;
  }>;
}

export default function AppShell() {
  const [activeNav, setActiveNav] = useState<NavItem>("dashboard");
  const [viewingCaseId, setViewingCaseId] = useState<string | null>(null);
  const [activeCase, setActiveCase] = useState<Case | null>(null);
  const [currentVideoId, setCurrentVideoId] = useState<string | null>(null);
  const [currentVideoName, setCurrentVideoName] = useState<string | null>(null);
  const [sidebarCollapsed, setSidebarCollapsed] = useState<boolean>(false);
  const [healthStatus, setHealthStatus] = useState<HealthResponse | null>(null);
  const [healthError, setHealthError] = useState<boolean>(false);
  const [currentTime, setCurrentTime] = useState<string>("");
  const [cases, setCases] = useState<Case[]>([]);
  const [loadingCases, setLoadingCases] = useState<boolean>(false);
  const [casesError, setCasesError] = useState<string | null>(null);
  const [analytics, setAnalytics] = useState<AnalyticsSummary | null>(null);
  const [loadingAnalytics, setLoadingAnalytics] = useState<boolean>(false);
  const [analyticsError, setAnalyticsError] = useState<string | null>(null);
  const [inspectingVideoId, setInspectingVideoId] = useState<string | null>(null);

  // Poll health and live UTC clock
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
    }, 15000);

    const clockInterval = setInterval(() => {
      const now = new Date();
      setCurrentTime(now.toTimeString().split(" ")[0]);
    }, 1000);
    setCurrentTime(new Date().toTimeString().split(" ")[0]);

    return () => {
      clearInterval(healthInterval);
      clearInterval(clockInterval);
    };
  }, []);

  // Fetch dashboard summary metrics with reliable error & loading tracking
  const loadDashboardData = useCallback(async () => {
    setLoadingCases(true);
    setLoadingAnalytics(true);
    setCasesError(null);
    setAnalyticsError(null);
    try {
      const [caseData, analyticsData] = await Promise.allSettled([
        getCases(),
        getAnalyticsSummary(),
      ]);
      if (caseData.status === "fulfilled") {
        setCases(caseData.value || []);
      } else {
        setCasesError(caseData.reason?.message || "Failed to load security cases.");
      }
      if (analyticsData.status === "fulfilled") {
        setAnalytics(analyticsData.value);
      } else {
        setAnalyticsError(analyticsData.reason?.message || "Failed to load platform analytics.");
      }
    } catch (err: any) {
      setCasesError(err.message || "Failed to load dashboard data.");
    } finally {
      setLoadingCases(false);
      setLoadingAnalytics(false);
    }
  }, []);

  // Refresh platform dashboard data on mount and whenever navigating
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

  // Explicitly close the active case session
  const handleCloseActiveCase = () => {
    setActiveCase(null);
    setViewingCaseId(null);
  };

  // Open direct video investigation workstation
  const handleOpenVideoInvestigation = (videoId: string, filename?: string) => {
    setInspectingVideoId(videoId);
    setCurrentVideoId(videoId);
    if (filename) setCurrentVideoName(filename);
  };

  // Open direct surveillance video ingestion (upload from laptop)
  const handleOpenIngestion = () => {
    setInspectingVideoId("new");
  };

  // Handle switching navigation: clears sub-inspection state
  const handleNavChange = (nav: NavItem) => {
    setActiveNav(nav);
    setInspectingVideoId(null);
  };

  // Nav items structured into 3 clear operational sections
  const navGroups: NavGroup[] = [
    {
      name: "OPERATIONS",
      items: [
        {
          id: "dashboard",
          label: "Dashboard",
          description: "Security Command Center",
          icon: (
            <svg className="w-4.5 h-4.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2H6a2 2 0 01-2-2V6zM14 6a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2h-2a2 2 0 01-2-2V6zM4 16a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2H6a2 2 0 01-2-2v-2zM14 16a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2h-2a2 2 0 01-2-2v-2z" />
            </svg>
          ),
        },
        {
          id: "live",
          label: "Live / Multi-Cam",
          description: "Multi-Camera Session Intelligence",
          icon: (
            <svg className="w-4.5 h-4.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 17V7m0 10a2 2 0 01-2 2H5a2 2 0 01-2-2V7a2 2 0 012-2h2a2 2 0 012 2m0 10a2 2 0 002 2h2a2 2 0 002-2M9 7a2 2 0 012-2h2a2 2 0 012 2m0 10V7m0 10a2 2 0 002 2h2a2 2 0 002-2V7a2 2 0 00-2-2h-2a2 2 0 00-2 2" />
            </svg>
          ),
        },
        {
          id: "cameras",
          label: "Cameras",
          description: "CCTV & Video Management",
          badge: analytics?.total_cameras !== undefined ? analytics.total_cameras : undefined,
          icon: (
            <svg className="w-4.5 h-4.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 10l4.553-2.276A1 1 0 0121 8.618v6.764a1 1 0 01-1.447.894L15 14M5 18h8a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z" />
            </svg>
          ),
        },
      ],
    },
    {
      name: "INTELLIGENCE",
      items: [
        {
          id: "incidents",
          label: "Incidents",
          description: "Correlated Incidents Workbench",
          badge: analytics?.total_correlated_incidents !== undefined && analytics.total_correlated_incidents > 0 ? analytics.total_correlated_incidents : undefined,
          icon: (
            <svg className="w-4.5 h-4.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
            </svg>
          ),
        },
        {
          id: "search",
          label: "Search",
          description: "Cross-Modal Query Engine",
          icon: (
            <svg className="w-4.5 h-4.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
            </svg>
          ),
        },
        {
          id: "evidence",
          label: "Evidence",
          description: "Preserved Evidence Vault",
          badge: analytics?.validated_evidence_count !== undefined && analytics.validated_evidence_count > 0 
            ? analytics.validated_evidence_count 
            : (analytics?.total_evidence !== undefined && analytics.total_evidence > 0 ? analytics.total_evidence : undefined),
          icon: (
            <svg className="w-4.5 h-4.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m5.618-4.016A11.955 11.955 0 0112 2.944a11.955 11.955 0 01-8.618 3.04A12.02 12.02 0 003 9c0 5.591 3.824 10.29 9 11.622 5.176-1.332 9-6.03 9-11.622 0-1.042-.133-2.052-.382-3.016z" />
            </svg>
          ),
        },
        {
          id: "analytics",
          label: "Analytics",
          description: "Intelligence & Health Diagnostics",
          icon: (
            <svg className="w-4.5 h-4.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M16 8v8m-4-5v5m-4-2v2m-2 4h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
            </svg>
          ),
        },
      ],
    },
    {
      name: "INVESTIGATION",
      items: [
        {
          id: "cases",
          label: "Cases",
          description: "Security Case Management",
          badge: analytics?.total_cases !== undefined && analytics.total_cases > 0 ? analytics.total_cases : (cases.length > 0 ? cases.length : undefined),
          icon: (
            <svg className="w-4.5 h-4.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 8h14M5 8a2 2 0 110-4h14a2 2 0 110 4M5 8v10a2 2 0 002 2h10a2 2 0 002-2V8m-9 4h4" />
            </svg>
          ),
        },
        {
          id: "reports",
          label: "Reports",
          description: "Incident Dossiers & Summaries",
          icon: (
            <svg className="w-4.5 h-4.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 17v-2m3 2v-4m3 4v-6m2 10H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
            </svg>
          ),
        },
      ],
    },
  ];

  return (
    <div className="w-screen h-screen overflow-hidden flex flex-col bg-zinc-950 text-zinc-100 select-none">
      {/* ── TOP HEADER ── */}
      <header className="h-14 flex-none border-b border-zinc-800/80 bg-zinc-900/95 backdrop-blur px-4 flex items-center justify-between gap-4 z-40">
        {/* Brand & Subtitle */}
        <div className="flex items-center gap-3">
          <div className="flex items-center gap-2">
            <span className="relative flex h-3 w-3">
              <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75"></span>
              <span className="relative inline-flex rounded-full h-3 w-3 bg-emerald-500"></span>
            </span>
            <span className="font-mono font-black tracking-widest text-lg text-white">SENTINEL</span>
          </div>
          <span className="hidden sm:inline-block text-[12px] font-mono text-zinc-400 border-l border-zinc-700 pl-3">
            Security Video Intelligence
          </span>
        </div>

        {/* Center: Current Selected Case Quick Banner */}
        <div className="flex-1 max-w-xl mx-4 hidden lg:flex items-center justify-center">
          {activeCase ? (
            <div className="flex items-center gap-2.5 bg-zinc-950 border border-emerald-500/40 rounded-md px-3.5 py-1.5 text-xs shadow-sm">
              <span className="font-mono text-zinc-400 text-[11px] font-semibold">CURRENT CASE:</span>
              <span className="font-mono font-bold text-emerald-400">
                {activeCase.case_number || activeCase.case_id || activeCase.id}
              </span>
              <span className="text-zinc-200 truncate max-w-[200px] font-medium">{activeCase.title}</span>
              <span className="text-[10px] font-mono px-1.5 py-0.5 rounded bg-zinc-800 text-zinc-300 uppercase font-semibold">
                {activeCase.status}
              </span>
              <button
                onClick={() => {
                  setViewingCaseId(activeCase.id);
                  setActiveNav("cases");
                  setInspectingVideoId(null);
                }}
                className="text-xs font-mono text-cyan-400 hover:text-cyan-300 ml-1 underline cursor-pointer font-medium"
              >
                Workspace &rarr;
              </button>
              <button
                onClick={handleCloseActiveCase}
                className="text-zinc-400 hover:text-rose-400 ml-1.5 text-sm cursor-pointer p-0.5 rounded hover:bg-zinc-800"
                title="Close active case session"
              >
                &times;
              </button>
            </div>
          ) : (
            <div className="flex items-center gap-2 text-xs text-zinc-400 font-mono">
              <span className="w-2 h-2 rounded-full bg-zinc-600" />
              <span>No case currently active &mdash;</span>
              <button
                onClick={() => {
                  setViewingCaseId(null);
                  setActiveNav("cases");
                  setInspectingVideoId(null);
                }}
                className="text-emerald-400 hover:underline cursor-pointer font-medium"
              >
                Select or create a security case
              </button>
            </div>
          )}
        </div>

        {/* Right Status & Neutral Analyst Avatar */}
        <div className="flex items-center gap-3">
          {/* Safeguard badge */}
          <div className="hidden xl:flex items-center gap-2 text-[11px] font-mono bg-zinc-950 border border-zinc-800 px-3 py-1 rounded text-zinc-300">
            <span className="h-2 w-2 rounded-full bg-amber-400" />
            <span>HITL (≤ 0.65)</span>
            <span className="text-zinc-700">|</span>
            <span className="h-2 w-2 rounded-full bg-purple-400" />
            <span>NO BIOMETRICS</span>
          </div>

          {/* Backend Health Status */}
          <div className="flex items-center gap-2 font-mono text-xs bg-zinc-950 border border-zinc-800 px-3 py-1 rounded">
            {healthStatus ? (
              <span className="flex items-center gap-1.5 text-emerald-400 font-semibold">
                <span className="h-2 w-2 rounded-full bg-emerald-500" />
                <span>ONLINE</span>
                <span className="text-zinc-500 text-[11px]">v{healthStatus.version}</span>
              </span>
            ) : healthError ? (
              <span className="flex items-center gap-1.5 text-red-400 font-semibold">
                <span className="h-2 w-2 rounded-full bg-red-500" />
                <span>OFFLINE</span>
              </span>
            ) : (
              <span className="text-zinc-500 text-xs">Connecting...</span>
            )}
          </div>

          {/* Neutral Analyst context */}
          <div className="flex items-center gap-2.5 border-l border-zinc-800 pl-3">
            <div className="h-8 w-8 rounded-full bg-zinc-800 border border-zinc-700 flex items-center justify-center text-xs font-mono font-bold text-zinc-300">
              SEC
            </div>
            <div className="hidden sm:block text-left text-xs leading-tight">
              <div className="font-semibold text-zinc-200">Security Analyst</div>
              <div className="text-[11px] font-mono text-zinc-500">Local Session</div>
            </div>
          </div>
        </div>
      </header>

      {/* ── MAIN WORKSPACE CONTAINER ── */}
      <div className="flex-1 flex overflow-hidden">
        {/* ── SIDEBAR NAVIGATION ── */}
        <aside
          className={`flex-none border-r border-zinc-800/80 bg-zinc-900/70 transition-all duration-200 flex flex-col justify-between ${
            sidebarCollapsed ? "w-16" : "w-64"
          }`}
        >
          {/* Grouped Nav List */}
          <div className="p-3 space-y-4 overflow-y-auto">
            {navGroups.map((group) => (
              <div key={group.name} className="space-y-1">
                {!sidebarCollapsed && (
                  <div className="text-[11px] font-mono uppercase tracking-wider text-zinc-500 px-3 py-1 font-semibold">
                    {group.name}
                  </div>
                )}
                {sidebarCollapsed && (
                  <div className="w-full text-center text-zinc-600 text-xs font-mono py-0.5">•</div>
                )}
                {group.items.map((item) => {
                  const isActive = activeNav === item.id;
                  return (
                    <button
                      key={item.id}
                      onClick={() => handleNavChange(item.id)}
                      title={sidebarCollapsed ? `${item.label} — ${item.description}` : undefined}
                      className={`w-full flex items-center gap-3 px-3 py-2.5 rounded-md text-[13px] font-medium transition-all text-left cursor-pointer ${
                        isActive
                          ? "bg-emerald-500/15 text-emerald-300 border border-emerald-500/40 shadow-sm font-semibold"
                          : "text-zinc-400 hover:text-zinc-100 hover:bg-zinc-800/70 border border-transparent"
                      }`}
                    >
                      <span className={`flex-none ${isActive ? "text-emerald-400" : "text-zinc-400"}`}>
                        {item.icon}
                      </span>
                      {!sidebarCollapsed && (
                        <div className="flex-1 flex items-center justify-between min-w-0">
                          <span className="truncate">{item.label}</span>
                          {item.badge !== undefined && (
                            <span className="ml-2 text-[11px] font-mono px-1.5 py-0.5 rounded bg-zinc-800 text-zinc-300 border border-zinc-700 font-normal">
                              {item.badge}
                            </span>
                          )}
                        </div>
                      )}
                    </button>
                  );
                })}
              </div>
            ))}
          </div>

          {/* Bottom Sidebar Controls & Safeguards */}
          <div className="p-3 border-t border-zinc-800/80 space-y-2">
            {!sidebarCollapsed && (
              <div className="p-3 rounded bg-zinc-950/70 border border-zinc-800/80 text-[11px] text-zinc-400 font-mono space-y-1">
                <div className="text-zinc-200 font-semibold flex items-center gap-1.5">
                  <span className="w-2 h-2 rounded-full bg-emerald-400" />
                  Observational Integrity
                </div>
                <p className="leading-normal text-zinc-400">
                  Strict anonymous tracking. Zero biometric inference. Human review mandatory for scores &le; 0.65.
                </p>
              </div>
            )}

            <button
              onClick={() => setSidebarCollapsed(!sidebarCollapsed)}
              className="w-full py-2 px-3 rounded text-xs font-mono text-zinc-400 hover:text-zinc-200 hover:bg-zinc-800/60 flex items-center justify-center gap-2 cursor-pointer border border-zinc-800/60 transition-colors"
              title={sidebarCollapsed ? "Expand sidebar" : "Collapse sidebar"}
            >
              <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                {sidebarCollapsed ? (
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 5l7 7-7 7M5 5l7 7-7 7" />
                ) : (
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M11 19l-7-7 7-7m8 14l-7-7 7-7" />
                )}
              </svg>
              {!sidebarCollapsed && <span>Collapse Sidebar</span>}
            </button>
          </div>
        </aside>

        {/* ── WORKSPACE CONTENT REGION ── */}
        <main className="flex-1 overflow-y-auto overflow-x-hidden p-4 lg:p-6 bg-zinc-950">
          {/* Sub-Inspection Mode: Deep Video Analysis on Request */}
          {inspectingVideoId ? (
            <div className="w-full space-y-4 max-w-[1750px] 2xl:max-w-[2150px] mx-auto">
              <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 p-3.5 bg-zinc-900/90 border border-zinc-800 rounded-xl shadow-md">
                <div className="flex items-center gap-3">
                  <button
                    onClick={() => setInspectingVideoId(null)}
                    className="px-3.5 py-1.5 bg-zinc-800 hover:bg-zinc-700 text-zinc-200 text-xs font-mono rounded-lg border border-zinc-700 flex items-center gap-2 transition-colors cursor-pointer"
                  >
                    <span>&larr;</span>
                    <span>Return to {activeNav.toUpperCase()}</span>
                  </button>
                  <span className="text-xs text-zinc-300 font-mono">
                    {inspectingVideoId === "new" ? (
                      <span className="text-emerald-400 font-semibold flex items-center gap-1.5">
                        <span className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse" />
                        Ingest &amp; Analyze Surveillance Video
                      </span>
                    ) : (
                      <>
                        Video Intelligence Pipeline:{" "}
                        <strong className="text-emerald-400 font-semibold">{currentVideoName || inspectingVideoId}</strong>
                      </>
                    )}
                  </span>
                </div>
                {activeCase && (
                  <div className="flex items-center gap-2.5 text-xs font-mono bg-zinc-950 px-3 py-1.5 rounded-lg border border-zinc-800">
                    <span className="text-zinc-500">Active Case:</span>
                    <span className="text-cyan-400 font-semibold">{activeCase.case_number || activeCase.id}</span>
                    <button
                      onClick={() => {
                        setViewingCaseId(activeCase.id);
                        setActiveNav("cases");
                        setInspectingVideoId(null);
                      }}
                      className="text-xs text-cyan-400 hover:underline cursor-pointer ml-1"
                    >
                      View Case &rarr;
                    </button>
                  </div>
                )}
              </div>
              <VideoUpload
                initialVideoId={inspectingVideoId === "new" ? undefined : inspectingVideoId}
                activeCaseId={activeCase?.id || null}
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
              {/* 1. VIEW: DASHBOARD */}
              {activeNav === "dashboard" && (
                <div className="w-full max-w-[1750px] 2xl:max-w-[2150px] mx-auto space-y-6">
                  {/* Header */}
                  <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 border-b border-zinc-800 pb-5">
                    <div>
                      <h1 className="text-2xl font-bold text-white tracking-wide">
                        Sentinel Security Command Center
                      </h1>
                      <p className="text-xs md:text-sm text-zinc-400 mt-1">
                        Operational security intelligence, active case investigations, CCTV feeds, and verified threat signals.
                      </p>
                    </div>
                    <div className="flex items-center gap-2.5">
                      <button
                        onClick={() => {
                          setViewingCaseId(null);
                          setActiveNav("cases");
                        }}
                        className="px-4 py-2 bg-emerald-600 hover:bg-emerald-500 text-zinc-950 font-bold text-xs rounded-lg transition-colors flex items-center gap-1.5 cursor-pointer shadow-sm"
                      >
                        <span>+</span>
                        <span>New Security Case</span>
                      </button>
                      <button
                        onClick={handleOpenIngestion}
                        className="px-4 py-2 bg-zinc-800 hover:bg-zinc-700 text-zinc-200 text-xs font-semibold rounded-lg border border-zinc-700 transition-colors cursor-pointer flex items-center gap-2"
                      >
                        <svg className="w-3.5 h-3.5 text-emerald-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-8l-4-4m0 0L8 8m4-4v12" />
                        </svg>
                        <span>Ingest Surveillance Video</span>
                      </button>
                    </div>
                  </div>

                  {/* Primary KPI Cards */}
                  <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
                    <div className="p-4 rounded-lg bg-zinc-900/70 border border-zinc-800 space-y-1">
                      <div className="text-xs font-mono text-zinc-400">OPEN CASES</div>
                      <div className="text-2xl font-bold text-white">
                        {analytics?.open_cases ?? (analytics?.cases_by_status ? (analytics.cases_by_status.OPEN || 0) + (analytics.cases_by_status.INVESTIGATING || 0) : cases.filter((c) => c.status !== "CLOSED").length)}
                      </div>
                      <div className="text-xs text-zinc-500">
                        {analytics?.total_cases ?? cases.length} total dossiers
                      </div>
                    </div>
                    <div className="p-4 rounded-lg bg-zinc-900/70 border border-zinc-800 space-y-1">
                      <div className="text-xs font-mono text-zinc-400">CORRELATED INCIDENTS</div>
                      <div className="text-2xl font-bold text-amber-400">
                        {analytics?.total_correlated_incidents ?? 0}
                      </div>
                      <div className="text-xs text-zinc-500">
                        {analytics?.review_required_count ?? 0} require human review
                      </div>
                    </div>
                    <div className="p-4 rounded-lg bg-zinc-900/70 border border-zinc-800 space-y-1">
                      <div className="text-xs font-mono text-zinc-400">PRESERVED EVIDENCE</div>
                      <div className="text-2xl font-bold text-cyan-400">
                        {analytics?.validated_evidence_count ?? analytics?.total_evidence ?? 0}
                      </div>
                      <div className="text-xs text-zinc-500">
                        {analytics?.validated_evidence_count !== undefined && analytics?.total_evidence !== undefined && analytics.total_evidence > analytics.validated_evidence_count
                          ? `${analytics.validated_evidence_count} validated (${analytics.total_evidence} total)`
                          : "Tamper-evident snapshots & clips"}
                      </div>
                    </div>
                    <div className="p-4 rounded-lg bg-zinc-900/70 border border-zinc-800 space-y-1">
                      <div className="text-xs font-mono text-zinc-400">CORE INTELLIGENCE</div>
                      <div className="text-2xl font-bold text-emerald-400">
                        {healthStatus ? "ONLINE" : "CONNECTING"}
                      </div>
                      <div className="text-xs text-zinc-500">
                        {analytics?.parity_consistent
                          ? `RAW Parity Verified — ${analytics.parity_formula ?? ""}`
                          : "Parity check pending"}
                      </div>
                    </div>
                  </div>

                  {/* Security Intelligence Lifecycle (NO Phase branding) */}
                  <div className="rounded-lg border border-zinc-800 bg-zinc-900/40 p-4 space-y-3">
                    <div className="flex items-center justify-between">
                      <span className="text-xs font-mono uppercase text-zinc-400 tracking-wider font-semibold">
                        Sentinel Security Intelligence Lifecycle
                      </span>
                      <span className="text-xs font-mono text-emerald-400 bg-emerald-500/10 px-2.5 py-0.5 rounded border border-emerald-500/30">
                        Preserved Ground-Truth Chain
                      </span>
                    </div>
                    <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-6 gap-2 text-center text-xs">
                      <div className="p-3 rounded bg-zinc-900/80 border border-zinc-800">
                        <div className="font-mono text-[11px] text-zinc-500">01</div>
                        <div className="font-semibold text-zinc-200 mt-0.5">INGEST</div>
                        <div className="text-[11px] text-zinc-500 mt-1">CCTV & Multi-Cam</div>
                      </div>
                      <div className="p-3 rounded bg-zinc-900/80 border border-zinc-800">
                        <div className="font-mono text-[11px] text-zinc-500">02</div>
                        <div className="font-semibold text-zinc-200 mt-0.5">DETECT & TRACK</div>
                        <div className="text-[11px] text-zinc-500 mt-1">YOLO + Anonymous</div>
                      </div>
                      <div className="p-3 rounded bg-zinc-900/80 border border-zinc-800">
                        <div className="font-mono text-[11px] text-zinc-500">03</div>
                        <div className="font-semibold text-zinc-200 mt-0.5">INTELLIGENCE</div>
                        <div className="text-[11px] text-zinc-500 mt-1">Spatial & Multi-Signal</div>
                      </div>
                      <div className="p-3 rounded bg-zinc-900/80 border border-zinc-800">
                        <div className="font-mono text-[11px] text-zinc-500">04</div>
                        <div className="font-semibold text-zinc-200 mt-0.5">CORRELATION</div>
                        <div className="text-[11px] text-zinc-500 mt-1">Cross-Camera & Story</div>
                      </div>
                      <div className="p-3 rounded bg-emerald-950/30 border border-emerald-500/40">
                        <div className="font-mono text-[11px] text-emerald-400 font-bold">05</div>
                        <div className="font-semibold text-emerald-300 mt-0.5">INVESTIGATION</div>
                        <div className="text-[11px] text-emerald-400/80 mt-1">Case & Focus Replay</div>
                      </div>
                      <div className="p-3 rounded bg-zinc-900/80 border border-zinc-800">
                        <div className="font-mono text-[11px] text-zinc-500">06</div>
                        <div className="font-semibold text-zinc-200 mt-0.5">EVIDENCE & AUDIT</div>
                        <div className="text-[11px] text-zinc-500 mt-1">Dossiers & Verification</div>
                      </div>
                    </div>
                  </div>

                  {/* Operational Quick Access */}
                  <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                    <div
                      onClick={() => setActiveNav("incidents")}
                      className="p-4 rounded-lg bg-zinc-900/60 border border-zinc-800 hover:border-zinc-700 cursor-pointer transition-colors space-y-2"
                    >
                      <div className="flex items-center justify-between">
                        <span className="font-semibold text-sm text-zinc-200">Incident Workbench</span>
                        <span className="text-xs font-mono text-emerald-400">&rarr;</span>
                      </div>
                      <p className="text-xs text-zinc-400 leading-relaxed">
                        Review correlated incident storylines, multi-signal evidence strength, and review-required triggers.
                      </p>
                    </div>
                    <div
                      onClick={() => setActiveNav("evidence")}
                      className="p-4 rounded-lg bg-zinc-900/60 border border-zinc-800 hover:border-zinc-700 cursor-pointer transition-colors space-y-2"
                    >
                      <div className="flex items-center justify-between">
                        <span className="font-semibold text-sm text-zinc-200">Evidence Vault</span>
                        <span className="text-xs font-mono text-emerald-400">&rarr;</span>
                      </div>
                      <p className="text-xs text-zinc-400 leading-relaxed">
                        Browse tamper-evident snapshots, annotated crops, and video clips preserved across all ingested feeds.
                      </p>
                    </div>
                    <div
                      onClick={() => setActiveNav("cameras")}
                      className="p-4 rounded-lg bg-zinc-900/60 border border-zinc-800 hover:border-zinc-700 cursor-pointer transition-colors space-y-2"
                    >
                      <div className="flex items-center justify-between">
                        <span className="font-semibold text-sm text-zinc-200">Cameras & Video Feeds</span>
                        <span className="text-xs font-mono text-emerald-400">&rarr;</span>
                      </div>
                      <p className="text-xs text-zinc-400 leading-relaxed">
                        Manage CCTV camera topologies, inspect ingested surveillance videos, and configure multi-camera sessions.
                      </p>
                    </div>
                  </div>

                  {/* Recent Cases Section */}
                  <div className="rounded-lg border border-zinc-800 bg-zinc-900/50 p-5 space-y-4">
                    <div className="flex items-center justify-between">
                      <div className="flex items-center gap-2.5">
                        <h2 className="text-base font-semibold text-white">Recent Security Cases</h2>
                        <span className="px-2 py-0.5 rounded text-[11px] font-mono bg-zinc-800 text-zinc-400 border border-zinc-700">
                          {analytics?.total_cases ?? cases.length} Total
                        </span>
                      </div>
                      <button
                        onClick={() => {
                          setViewingCaseId(null);
                          setActiveNav("cases");
                        }}
                        className="text-xs text-emerald-400 hover:underline cursor-pointer font-medium"
                      >
                        View all cases &rarr;
                      </button>
                    </div>

                    {loadingCases ? (
                      <div className="text-center py-8 text-xs text-zinc-500 font-mono space-y-2">
                        <div className="w-5 h-5 border-2 border-emerald-500 border-t-transparent rounded-full animate-spin mx-auto" />
                        <span>Loading security cases...</span>
                      </div>
                    ) : casesError ? (
                      <div className="text-center py-8 space-y-3 bg-red-950/20 border border-red-900/30 rounded-lg p-4">
                        <div className="text-xs text-red-400 font-mono">Unable to load security cases: {casesError}</div>
                        <button
                          onClick={loadDashboardData}
                          className="px-3 py-1.5 bg-zinc-800 hover:bg-zinc-700 text-zinc-200 text-xs font-mono rounded border border-zinc-700 cursor-pointer"
                        >
                          Retry Loading Cases
                        </button>
                      </div>
                    ) : cases.length === 0 ? (
                      <div className="text-center py-10 text-xs text-zinc-500 font-mono space-y-2 border border-dashed border-zinc-800 rounded-lg">
                        <p>No security cases recorded yet.</p>
                        <button
                          onClick={() => {
                            setViewingCaseId(null);
                            setActiveNav("cases");
                          }}
                          className="px-3 py-1.5 bg-emerald-600/20 text-emerald-400 hover:bg-emerald-600/30 border border-emerald-500/30 rounded text-xs font-semibold cursor-pointer"
                        >
                          + Create First Security Case
                        </button>
                      </div>
                    ) : (
                      <div className="space-y-2.5">
                        {cases.slice(0, 5).map((c) => (
                          <div
                            key={c.id}
                            onClick={() => handleOpenCase(c.id)}
                            className="flex flex-col sm:flex-row sm:items-center justify-between p-3.5 rounded-lg bg-zinc-950/60 border border-zinc-800/80 hover:border-emerald-500/50 hover:bg-zinc-900/50 transition-all gap-3 cursor-pointer group"
                          >
                            <div className="space-y-1 min-w-0">
                              <div className="flex items-center gap-2.5">
                                <span className="font-mono text-xs text-emerald-400 font-bold group-hover:underline">
                                  {c.case_number || c.case_id || c.id}
                                </span>
                                <span className="text-xs font-semibold text-zinc-200 truncate">
                                  {c.title}
                                </span>
                              </div>
                              <p className="text-xs text-zinc-400 line-clamp-1">
                                {c.description || "No description provided."}
                              </p>
                            </div>

                            <div className="flex items-center gap-3 flex-none">
                              <span
                                className={`text-[11px] font-mono px-2 py-0.5 rounded border uppercase font-semibold ${
                                  c.priority === "CRITICAL"
                                    ? "bg-red-500/20 text-red-400 border-red-500/40"
                                    : c.priority === "HIGH"
                                    ? "bg-amber-500/20 text-amber-400 border-amber-500/40"
                                    : "bg-zinc-800 text-zinc-400 border-zinc-700"
                                }`}
                              >
                                {c.priority}
                              </span>
                              <span className="text-[11px] font-mono px-2 py-0.5 rounded bg-zinc-800 text-zinc-300 border border-zinc-700 uppercase">
                                {c.status}
                              </span>
                              <button
                                onClick={(e) => {
                                  e.stopPropagation();
                                  handleOpenCase(c.id);
                                }}
                                className="px-3 py-1 bg-zinc-800 hover:bg-emerald-600 hover:text-zinc-950 text-zinc-200 text-xs font-mono rounded border border-zinc-700 transition-all cursor-pointer font-medium"
                              >
                                Open Workspace
                              </button>
                            </div>
                          </div>
                        ))}
                      </div>
                    )}
                  </div>

                  {/* Ethics & Reliability Banner */}
                  <div className="rounded-lg border border-zinc-800/80 bg-zinc-900/30 p-4 text-xs text-zinc-400 space-y-2">
                    <div className="font-semibold text-zinc-300 flex items-center gap-2">
                      <span className="w-2 h-2 rounded-full bg-emerald-400" />
                      Sentinel Reliability &amp; Privacy Safeguards
                    </div>
                    <p className="leading-relaxed">
                      Sentinel strictly operates as an investigative decision-support assistant. Final assessment scores &le; 0.65 are classified as <span className="text-amber-400 font-mono font-semibold">REVIEW_REQUIRED</span> and mandate human analyst confirmation. All track IDs are strictly anonymous; facial recognition, biometric matching, and identity attribution are fundamentally prohibited.
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

              {/* 3. VIEW: CAMERAS (Dedicated Cameras & Video Workspace) */}
              {activeNav === "cameras" && (
                <div className="w-full">
                  <CamerasWorkspaceView
                    onInvestigateVideo={(videoId: string) => handleOpenVideoInvestigation(videoId)}
                    onIngestVideo={handleOpenIngestion}
                    onCreateCaseFromVideo={(videoId: string) => {
                      setViewingCaseId(null);
                      setActiveNav("cases");
                    }}
                    onOpenMultiCamera={() => setActiveNav("live")}
                    onDataChanged={loadDashboardData}
                  />
                </div>
              )}

              {/* 4. VIEW: LIVE / MULTI-CAMERA */}
              {activeNav === "live" && (
                <div className="w-full">
                  <MultiCameraSessionPanel />
                </div>
              )}

              {/* 5. VIEW: INCIDENTS (Dedicated Correlated Incidents Workbench) */}
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

              {/* 6. VIEW: SEARCH (Dedicated Investigation / Search Workspace) */}
              {activeNav === "search" && (
                <div className="w-full">
                  <SearchWorkspaceView
                    initialVideoId={currentVideoId || inspectingVideoId || undefined}
                    activeCase={activeCase}
                    onSeek={(ts: number) => {
                      // seek handled inside search view
                    }}
                    onIngestVideo={handleOpenIngestion}
                  />
                </div>
              )}

              {/* 7. VIEW: EVIDENCE (Dedicated Evidence Vault) */}
              {activeNav === "evidence" && (
                <div className="w-full">
                  <EvidenceVaultView
                    onInvestigateVideo={(videoId: string) => setInspectingVideoId(videoId)}
                  />
                </div>
              )}

              {/* 8. VIEW: REPORTS (Dedicated Reports & Dossier Workspace) */}
              {activeNav === "reports" && (
                <div className="w-full">
                  <ReportsWorkspaceView />
                </div>
              )}

              {/* 9. VIEW: ANALYTICS (Dedicated Intelligence & Health Workspace) */}
              {activeNav === "analytics" && (
                <div className="w-full">
                  <AnalyticsWorkspaceView />
                </div>
              )}
            </>
          )}
        </main>
      </div>

      {/* ── BOTTOM STATUS BAR ── */}
      <footer className="h-8 flex-none bg-zinc-900/95 border-t border-zinc-800 px-4 flex items-center justify-between text-xs font-mono text-zinc-400 select-none z-30">
        <div className="flex items-center gap-3">
          <span className="text-zinc-200 font-semibold">SENTINEL OPERATIONS PLATFORM</span>
          <span className="text-zinc-700">|</span>
          <span>
            WORKSPACE: <strong className="text-emerald-400 uppercase">{activeNav}</strong>
          </span>
          <span className="text-zinc-700">|</span>
          <span>
            CASE:{" "}
            {activeCase ? (
              <span className="text-cyan-400 font-semibold">
                {activeCase.case_number || activeCase.case_id || activeCase.id}
              </span>
            ) : (
              <span className="text-zinc-500">UNASSIGNED</span>
            )}
          </span>
        </div>

        <div className="hidden md:flex items-center gap-4 text-zinc-500 text-[11px]">
          <span>RELIABILITY CEILING: REVIEW_REQUIRED &le; 0.65</span>
          <span>•</span>
          <span>BIOMETRICS: DISABLED</span>
          <span>•</span>
          <span>TRACKING: ANONYMOUS</span>
        </div>

        <div className="flex items-center gap-3">
          <span className="text-zinc-500">UTC: {currentTime || "--:--:--"}</span>
          <span className="text-zinc-700">|</span>
          <span className="flex items-center gap-1.5 text-emerald-400 font-semibold">
            <span className="h-2 w-2 rounded-full bg-emerald-400" />
            CORE ACTIVE
          </span>
        </div>
      </footer>
    </div>
  );
}
