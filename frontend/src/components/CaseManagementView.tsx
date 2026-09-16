"use client";

import React, { useState, useEffect, useCallback } from "react";
import { Case, listCases, createCase, deleteCase, updateCase } from "@/lib/api";

interface CaseManagementViewProps {
  onOpenCase?: (caseId: string) => void;
  onSelectCase?: (caseId: string) => void;
  onDataChanged?: () => void;
}

export default function CaseManagementView({ onOpenCase, onSelectCase, onDataChanged }: CaseManagementViewProps) {
  const handleOpenCase = (id: string) => {
    if (onOpenCase) onOpenCase(id);
    if (onSelectCase) onSelectCase(id);
  };
  const [cases, setCases] = useState<Case[]>([]);
  const [totalCount, setTotalCount] = useState(0);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Filters
  const [searchQuery, setSearchQuery] = useState("");
  const [statusFilter, setStatusFilter] = useState<string>("");
  const [priorityFilter, setPriorityFilter] = useState<string>("");

  // Create Modal
  const [isCreateOpen, setIsCreateOpen] = useState(false);
  const [newTitle, setNewTitle] = useState("");
  const [newDesc, setNewDesc] = useState("");
  const [newPriority, setNewPriority] = useState<"LOW" | "MEDIUM" | "HIGH" | "CRITICAL">("MEDIUM");
  const [newInvestigator, setNewInvestigator] = useState("Security Analyst (Local Session)");
  const [newTags, setNewTags] = useState("Security, Perimeter");
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [createError, setCreateError] = useState<string | null>(null);

  const fetchCases = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await listCases({
        status: statusFilter || undefined,
        priority: priorityFilter || undefined,
        search: searchQuery || undefined,
        limit: 100,
      });
      setCases(res.cases || []);
      setTotalCount(res.total !== undefined ? res.total : (res.cases || []).length);
    } catch (err: any) {
      setError(err.message || "Failed to load security cases.");
    } finally {
      setLoading(false);
    }
  }, [statusFilter, priorityFilter, searchQuery]);

  useEffect(() => {
    fetchCases();
  }, [fetchCases]);

  const handleCreateSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!newTitle.trim()) {
      setCreateError("Title is required.");
      return;
    }
    setIsSubmitting(true);
    setCreateError(null);
    try {
      const tagsArray = newTags
        .split(",")
        .map((t) => t.trim())
        .filter(Boolean);
      const created = await createCase({
        title: newTitle.trim(),
        description: newDesc.trim() || undefined,
        priority: newPriority,
        assigned_investigator: newInvestigator.trim() || undefined,
        tags: tagsArray,
      });
      setIsCreateOpen(false);
      setNewTitle("");
      setNewDesc("");
      await fetchCases();
      if (onDataChanged) onDataChanged();
      handleOpenCase(created.id);
    } catch (err: any) {
      setCreateError(err.message || "Failed to create case.");
    } finally {
      setIsSubmitting(false);
    }
  };

  const handleDelete = async (e: React.MouseEvent, caseId: string, caseNumber: string) => {
    e.stopPropagation();
    if (!confirm(`Are you sure you want to permanently delete case ${caseNumber}?`)) return;
    try {
      await deleteCase(caseId);
      await fetchCases();
      if (onDataChanged) onDataChanged();
    } catch (err: any) {
      alert(`Delete failed: ${err.message}`);
    }
  };

  // Metrics summary
  const displayedCount = cases.length;
  const openCount = cases.filter((c) => c.status === "OPEN" || c.status === "INVESTIGATING").length;
  const reviewCount = cases.filter((c) => c.status === "REVIEW").length;
  const closedCount = cases.filter((c) => c.status === "CLOSED").length;

  const getPriorityBadge = (p: string) => {
    switch (p) {
      case "CRITICAL":
        return "bg-rose-500/20 text-rose-300 border-rose-600/50";
      case "HIGH":
        return "bg-amber-500/20 text-amber-300 border-amber-600/50";
      case "MEDIUM":
        return "bg-blue-500/20 text-blue-300 border-blue-600/50";
      case "LOW":
        return "bg-zinc-800 text-zinc-400 border-zinc-700";
      default:
        return "bg-zinc-800 text-zinc-400 border-zinc-700";
    }
  };

  const getStatusBadge = (s: string) => {
    switch (s) {
      case "OPEN":
        return "bg-emerald-500/20 text-emerald-300 border-emerald-600/50";
      case "INVESTIGATING":
        return "bg-cyan-500/20 text-cyan-300 border-cyan-600/50";
      case "REVIEW":
        return "bg-purple-500/20 text-purple-300 border-purple-600/50";
      case "CLOSED":
        return "bg-zinc-800 text-zinc-400 border-zinc-700";
      default:
        return "bg-zinc-800 text-zinc-300 border-zinc-700";
    }
  };

  const priorityColor = getPriorityBadge;
  const statusColor = getStatusBadge;

  return (
    <div className="w-full h-full flex flex-col space-y-6 overflow-hidden p-6 md:p-8 max-w-[1600px] mx-auto">
      {/* Top Banner & Stats */}
      <div className="flex flex-col md:flex-row items-start md:items-center justify-between gap-4 border-b border-zinc-800/80 pb-5">
        <div>
          <div className="flex items-center gap-2.5">
            <h1 className="text-2xl font-bold tracking-tight text-white flex items-center gap-2.5">
              <span className="w-3 h-3 rounded-full bg-emerald-500 animate-pulse" />
              Security Case Management
            </h1>
            <span className="text-xs font-mono uppercase bg-emerald-950/80 text-emerald-300 px-2.5 py-0.5 rounded border border-emerald-800/50">
              Workspace
            </span>
          </div>
          <p className="text-sm text-zinc-400 mt-1">
            Multi-video investigations, cross-camera timelines, evidence bookmarks &amp; storyline tracking.
          </p>
        </div>

        <div className="flex items-center gap-2.5">
          <button
            onClick={() => setIsCreateOpen(true)}
            className="flex items-center gap-2 px-4 py-2 bg-emerald-600 hover:bg-emerald-500 text-zinc-950 font-bold text-sm rounded-lg shadow-md hover:shadow-emerald-500/20 transition-all cursor-pointer"
          >
            <span className="text-base leading-none font-bold">+</span>
            New Security Case
          </button>
        </div>
      </div>

      {/* Metrics Row */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
        <div className="bg-zinc-900/90 border border-zinc-800 p-3 rounded-lg flex items-center justify-between">
          <div>
            <div className="text-[11px] font-mono text-zinc-400 uppercase tracking-wider">Total Cases</div>
            <div className="text-xl font-bold text-white font-mono mt-0.5">{totalCount}</div>
          </div>
          <span className="text-xl opacity-70">📂</span>
        </div>

        <div className="bg-zinc-900/90 border border-zinc-800 p-3 rounded-lg flex items-center justify-between">
          <div>
            <div className="text-[11px] font-mono text-emerald-400 uppercase tracking-wider">Active / Open</div>
            <div className="text-xl font-bold text-emerald-300 font-mono mt-0.5">{openCount}</div>
          </div>
          <span className="text-xl text-emerald-400 opacity-70">🔍</span>
        </div>

        <div className="bg-zinc-900/90 border border-zinc-800 p-3 rounded-lg flex items-center justify-between">
          <div>
            <div className="text-[11px] font-mono text-purple-400 uppercase tracking-wider">Review Required</div>
            <div className="text-xl font-bold text-purple-300 font-mono mt-0.5">{reviewCount}</div>
          </div>
          <span className="text-xl text-purple-400 opacity-70">⚖️</span>
        </div>

        <div className="bg-zinc-900/90 border border-zinc-800 p-3 rounded-lg flex items-center justify-between">
          <div>
            <div className="text-[11px] font-mono text-zinc-400 uppercase tracking-wider">Closed Cases</div>
            <div className="text-xl font-bold text-zinc-300 font-mono mt-0.5">{closedCount}</div>
          </div>
          <span className="text-xl text-zinc-500 opacity-70">🔒</span>
        </div>
      </div>

      {/* Filter Bar */}
      <div className="flex flex-wrap items-center justify-between gap-3 bg-zinc-900/60 p-2.5 rounded-lg border border-zinc-800/80 text-xs">
        <div className="flex items-center gap-2 flex-1 min-w-[240px]">
          <span className="text-zinc-500 text-sm">🔎</span>
          <input
            type="text"
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            placeholder="Search by title, case number, or keywords..."
            className="w-full bg-zinc-950 border border-zinc-800 rounded px-2.5 py-1 text-zinc-200 placeholder-zinc-500 focus:outline-none focus:border-emerald-500 text-xs"
          />
        </div>

        <div className="flex items-center gap-2">
          <select
            value={statusFilter}
            onChange={(e) => setStatusFilter(e.target.value)}
            className="bg-zinc-950 border border-zinc-800 rounded px-2.5 py-1 text-zinc-300 focus:outline-none focus:border-emerald-500 text-xs"
          >
            <option value="">Status: All</option>
            <option value="OPEN">Open</option>
            <option value="INVESTIGATING">Investigating</option>
            <option value="REVIEW">Review</option>
            <option value="CLOSED">Closed</option>
          </select>

          <select
            value={priorityFilter}
            onChange={(e) => setPriorityFilter(e.target.value)}
            className="bg-zinc-950 border border-zinc-800 rounded px-2.5 py-1 text-zinc-300 focus:outline-none focus:border-emerald-500 text-xs"
          >
            <option value="">Priority: All</option>
            <option value="CRITICAL">Critical</option>
            <option value="HIGH">High</option>
            <option value="MEDIUM">Medium</option>
            <option value="LOW">Low</option>
          </select>

          {(searchQuery || statusFilter || priorityFilter) && (
            <button
              onClick={() => {
                setSearchQuery("");
                setStatusFilter("");
                setPriorityFilter("");
              }}
              className="text-zinc-400 hover:text-zinc-200 px-2 py-1 text-xs cursor-pointer"
            >
              Reset
            </button>
          )}
        </div>
      </div>

      {/* Case List Grid */}
      <div className="flex-1 overflow-y-auto space-y-2.5 pr-1">
        {loading ? (
          <div className="flex flex-col items-center justify-center py-20 text-zinc-500 space-y-2">
            <div className="w-6 h-6 border-2 border-emerald-500 border-t-transparent rounded-full animate-spin" />
            <span className="text-xs font-mono">Loading cases...</span>
          </div>
        ) : error ? (
          <div className="p-4 bg-rose-950/30 border border-rose-800/50 rounded-lg text-rose-300 text-xs">
            Error loading cases: {error}
          </div>
        ) : cases.length === 0 ? (
          <div className="flex flex-col items-center justify-center py-16 text-center bg-zinc-900/30 border border-dashed border-zinc-800 rounded-xl p-8 space-y-3">
            <div className="text-3xl opacity-60">📁</div>
            <div className="text-sm font-semibold text-zinc-300">No Forensic Cases Found</div>
            <p className="text-xs text-zinc-500 max-w-sm">
              {searchQuery || statusFilter || priorityFilter
                ? "No cases match your active filters. Try resetting search criteria."
                : "Create your first forensic case to organize multi-camera surveillance footage, build unified timelines, and preserve evidence."}
            </p>
            <button
              onClick={() => setIsCreateOpen(true)}
              className="mt-2 px-3.5 py-1.5 bg-emerald-600 hover:bg-emerald-500 text-zinc-950 font-semibold text-xs rounded-lg transition-colors cursor-pointer"
            >
              Create New Case
            </button>
          </div>
        ) : (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3">
            {cases.map((c) => (
              <div
                key={c.id}
                onClick={() => handleOpenCase(c.id)}
                className="group bg-zinc-900/80 hover:bg-zinc-900 border border-zinc-800 hover:border-zinc-700 rounded-xl p-4 transition-all duration-150 cursor-pointer flex flex-col justify-between space-y-3 relative shadow-sm"
              >
                <div>
                  <div className="flex items-start justify-between gap-2">
                    <span className="font-mono text-[11px] text-zinc-400 font-semibold tracking-wider">
                      {c.case_number}
                    </span>
                    <div className="flex items-center gap-1.5">
                      <span className={`text-[10px] font-mono font-bold px-2 py-0.5 rounded border ${priorityColor(c.priority)}`}>
                        {c.priority}
                      </span>
                      <span className={`text-[10px] font-mono font-bold px-2 py-0.5 rounded border ${statusColor(c.status)}`}>
                        {c.status}
                      </span>
                    </div>
                  </div>

                  <h3 className="text-sm font-bold text-zinc-100 group-hover:text-emerald-300 transition-colors mt-1.5 leading-snug">
                    {c.title}
                  </h3>

                  {c.description && (
                    <p className="text-xs text-zinc-400 line-clamp-2 mt-1 leading-relaxed">
                      {c.description}
                    </p>
                  )}

                  {/* Tags */}
                  {c.tags && c.tags.length > 0 && (
                    <div className="flex flex-wrap gap-1 mt-2.5">
                      {c.tags.slice(0, 4).map((tag, idx) => (
                        <span
                          key={idx}
                          className="text-[10px] font-mono bg-zinc-800/80 text-zinc-400 px-1.5 py-0.5 rounded border border-zinc-700/50"
                        >
                          #{tag}
                        </span>
                      ))}
                      {c.tags.length > 4 && (
                        <span className="text-[10px] font-mono text-zinc-500 self-center">
                          +{c.tags.length - 4}
                        </span>
                      )}
                    </div>
                  )}
                </div>

                {/* Footer Entity Counts & Actions */}
                <div className="pt-2.5 border-t border-zinc-800/80 flex items-center justify-between text-[11px] text-zinc-500 font-mono">
                  <div className="flex items-center gap-2.5">
                    <span title="Linked Videos" className="flex items-center gap-1 text-zinc-400">
                      🎥 {c.counts?.videos ?? 0}
                    </span>
                    <span title="Linked Incidents" className="flex items-center gap-1 text-zinc-400">
                      ⚠️ {c.counts?.incidents ?? 0}
                    </span>
                    <span title="Bookmarks" className="flex items-center gap-1 text-zinc-400">
                      📌 {c.counts?.bookmarks ?? 0}
                    </span>
                    <span title="Notes" className="flex items-center gap-1 text-zinc-400">
                      📝 {c.counts?.notes ?? 0}
                    </span>
                  </div>

                  <div className="flex items-center gap-1">
                    <button
                      onClick={(e) => handleDelete(e, c.id, c.case_number)}
                      title="Delete Case"
                      className="text-zinc-500 hover:text-rose-400 p-1 rounded transition-colors cursor-pointer"
                    >
                      🗑️
                    </button>
                  </div>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Create Case Modal */}
      {isCreateOpen && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/75 backdrop-blur-sm p-4">
          <div className="bg-zinc-900 border border-zinc-800 rounded-xl max-w-lg w-full p-5 space-y-4 shadow-2xl">
            <div className="flex items-center justify-between border-b border-zinc-800 pb-3">
              <h2 className="text-base font-bold text-white flex items-center gap-2">
                <span className="text-emerald-400">+</span>
                Create New Forensic Case
              </h2>
              <button
                onClick={() => setIsCreateOpen(false)}
                className="text-zinc-500 hover:text-zinc-300 text-lg leading-none cursor-pointer"
              >
                &times;
              </button>
            </div>

            {createError && (
              <div className="p-2.5 bg-rose-950/50 border border-rose-800 text-rose-300 text-xs rounded">
                {createError}
              </div>
            )}

            <form onSubmit={handleCreateSubmit} className="space-y-3.5 text-xs">
              <div>
                <label className="block text-zinc-300 font-medium mb-1">
                  Case Title <span className="text-rose-400">*</span>
                </label>
                <input
                  type="text"
                  required
                  value={newTitle}
                  onChange={(e) => setNewTitle(e.target.value)}
                  placeholder="e.g. North Gate Unauthorized Breach Investigation"
                  className="w-full bg-zinc-950 border border-zinc-800 rounded-lg p-2 text-zinc-200 placeholder-zinc-500 focus:outline-none focus:border-emerald-500 text-xs"
                />
              </div>

              <div>
                <label className="block text-zinc-300 font-medium mb-1">Description / Brief</label>
                <textarea
                  rows={3}
                  value={newDesc}
                  onChange={(e) => setNewDesc(e.target.value)}
                  placeholder="Brief synopsis of the incident, source CCTV, and initial observations..."
                  className="w-full bg-zinc-950 border border-zinc-800 rounded-lg p-2 text-zinc-200 placeholder-zinc-500 focus:outline-none focus:border-emerald-500 text-xs"
                />
              </div>

              <div className="grid grid-cols-2 gap-3">
                <div>
                  <label className="block text-zinc-300 font-medium mb-1">Priority</label>
                  <select
                    value={newPriority}
                    onChange={(e: any) => setNewPriority(e.target.value)}
                    className="w-full bg-zinc-950 border border-zinc-800 rounded-lg p-2 text-zinc-200 focus:outline-none focus:border-emerald-500 text-xs"
                  >
                    <option value="LOW">Low</option>
                    <option value="MEDIUM">Medium</option>
                    <option value="HIGH">High</option>
                    <option value="CRITICAL">Critical</option>
                  </select>
                </div>

                <div>
                  <label className="block text-zinc-300 font-medium mb-1">Security Analyst</label>
                  <input
                    type="text"
                    value={newInvestigator}
                    onChange={(e) => setNewInvestigator(e.target.value)}
                    placeholder="Analyst Name"
                    className="w-full bg-zinc-950 border border-zinc-800 rounded-lg p-2 text-zinc-200 focus:outline-none focus:border-emerald-500 text-xs"
                  />
                </div>
              </div>

              <div>
                <label className="block text-zinc-300 font-medium mb-1">Tags (comma-separated)</label>
                <input
                  type="text"
                  value={newTags}
                  onChange={(e) => setNewTags(e.target.value)}
                  placeholder="Theft, Perimeter, Loading Bay, Night Shift"
                  className="w-full bg-zinc-950 border border-zinc-800 rounded-lg p-2 text-zinc-200 focus:outline-none focus:border-emerald-500 text-xs"
                />
              </div>

              <div className="pt-3 border-t border-zinc-800 flex items-center justify-end gap-2">
                <button
                  type="button"
                  onClick={() => setIsCreateOpen(false)}
                  className="px-3.5 py-1.5 bg-zinc-800 hover:bg-zinc-700 text-zinc-300 rounded-lg transition-colors cursor-pointer"
                >
                  Cancel
                </button>
                <button
                  type="submit"
                  disabled={isSubmitting}
                  className="px-4 py-1.5 bg-emerald-600 hover:bg-emerald-500 disabled:opacity-50 text-zinc-950 font-semibold rounded-lg transition-colors cursor-pointer"
                >
                  {isSubmitting ? "Creating..." : "Create Case"}
                </button>
              </div>
            </form>
          </div>
        </div>
      )}
    </div>
  );
}
