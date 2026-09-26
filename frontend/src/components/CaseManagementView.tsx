"use client";

import React, { useState, useEffect, useCallback } from "react";
import { Case, listCases, createCase, deleteCase } from "@/lib/api";

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

  const priorityColor = (p: string) => {
    switch (p?.toUpperCase()) {
      case "CRITICAL":
        return "text-rose-400 bg-rose-500/10 border-rose-500/30";
      case "HIGH":
        return "text-amber-400 bg-amber-500/10 border-amber-500/30";
      case "MEDIUM":
        return "text-[#19B89A] bg-[#19B89A]/10 border-[#19B89A]/30";
      default:
        return "text-[#A7AFBA] bg-[#1D2128] border-[#2A3038]";
    }
  };

  const statusColor = (s: string) => {
    switch (s?.toUpperCase()) {
      case "CLOSED":
        return "text-[#737C87] bg-[#1D2128] border-[#2A3038]";
      case "REVIEW":
        return "text-amber-400 bg-amber-500/10 border-amber-500/30";
      case "INVESTIGATING":
        return "text-[#19B89A] bg-[#19B89A]/10 border-[#19B89A]/30";
      default:
        return "text-[#F5F7FA] bg-[#1D2128] border-[#2A3038]";
    }
  };

  const openCount = cases.filter((c) => c.status !== "CLOSED").length;
  const reviewCount = cases.filter((c) => c.status === "REVIEW" || c.priority === "HIGH" || c.priority === "CRITICAL").length;
  const closedCount = cases.filter((c) => c.status === "CLOSED").length;

  return (
    <div className="p-6 md:p-8 space-y-6 max-w-[1600px] mx-auto min-h-full">
      {/* Header */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 border-b border-[#2A3038] pb-5">
        <div>
          <div className="flex items-center gap-3">
            <h1 className="text-2xl font-bold tracking-tight text-[#F5F7FA]">
              Security Cases
            </h1>
            <span className="text-xs font-sans font-medium bg-[#19B89A]/15 text-[#19B89A] px-2.5 py-0.5 rounded-full border border-[#19B89A]/30">
              {totalCount} Dossiers
            </span>
          </div>
          <p className="text-sm text-[#A7AFBA] mt-1">
            Active investigation dossiers, cross-camera event timelines, and preserved evidence bookmarks.
          </p>
        </div>

        <button
          onClick={() => setIsCreateOpen(true)}
          className="flex items-center gap-2 px-5 py-2.5 bg-[#19B89A] hover:bg-[#16A489] text-[#0F1115] font-semibold text-xs rounded-lg transition-colors cursor-pointer shadow-sm"
        >
          <span>+ Start New Case</span>
        </button>
      </div>

      {/* Metrics Row */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
        <div className="bg-[#171A20] border border-[#2A3038] p-4 rounded-xl space-y-1">
          <div className="text-xs text-[#737C87]">Total Cases</div>
          <div className="text-2xl font-bold text-[#F5F7FA] font-mono">{totalCount}</div>
        </div>

        <div className="bg-[#171A20] border border-[#2A3038] p-4 rounded-xl space-y-1">
          <div className="text-xs text-[#737C87]">Active / Open</div>
          <div className="text-2xl font-bold text-[#19B89A] font-mono">{openCount}</div>
        </div>

        <div className="bg-[#171A20] border border-[#2A3038] p-4 rounded-xl space-y-1">
          <div className="text-xs text-[#737C87]">Review Required</div>
          <div className="text-2xl font-bold text-amber-400 font-mono">{reviewCount}</div>
        </div>

        <div className="bg-[#171A20] border border-[#2A3038] p-4 rounded-xl space-y-1">
          <div className="text-xs text-[#737C87]">Closed Dossiers</div>
          <div className="text-2xl font-bold text-[#737C87] font-mono">{closedCount}</div>
        </div>
      </div>

      {/* Filter Bar */}
      <div className="flex flex-wrap items-center justify-between gap-3 bg-[#171A20] p-3 rounded-xl border border-[#2A3038] text-xs">
        <div className="flex items-center gap-2 flex-1 min-w-[240px]">
          <input
            type="text"
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            placeholder="Search by case title, case number, or keywords..."
            className="w-full bg-[#1D2128] border border-[#2A3038] rounded-lg px-3 py-1.5 text-[#F5F7FA] placeholder-[#737C87] focus:outline-none focus:border-[#19B89A] text-xs"
          />
        </div>

        <div className="flex items-center gap-2">
          <select
            value={statusFilter}
            onChange={(e) => setStatusFilter(e.target.value)}
            className="bg-[#1D2128] border border-[#2A3038] rounded-lg px-2.5 py-1.5 text-[#F5F7FA] focus:outline-none focus:border-[#19B89A] text-xs"
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
            className="bg-[#1D2128] border border-[#2A3038] rounded-lg px-2.5 py-1.5 text-[#F5F7FA] focus:outline-none focus:border-[#19B89A] text-xs"
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
              className="text-[#737C87] hover:text-[#F5F7FA] px-2 py-1 text-xs cursor-pointer"
            >
              Reset
            </button>
          )}
        </div>
      </div>

      {/* Case Dossiers Grid */}
      <div>
        {loading ? (
          <div className="flex flex-col items-center justify-center py-20 text-[#737C87] space-y-2">
            <div className="w-6 h-6 border-2 border-[#19B89A] border-t-transparent rounded-full animate-spin" />
            <span className="text-xs">Loading cases...</span>
          </div>
        ) : error ? (
          <div className="p-4 bg-rose-500/10 border border-rose-500/30 rounded-xl text-rose-300 text-xs">
            {error}
          </div>
        ) : cases.length === 0 ? (
          <div className="flex flex-col items-center justify-center py-16 text-center bg-[#171A20] border border-dashed border-[#2A3038] rounded-xl p-8 space-y-3">
            <div className="text-sm font-semibold text-[#F5F7FA]">No security cases found</div>
            <p className="text-xs text-[#737C87] max-w-sm">
              Create your first security case to organize surveillance footage, correlate timelines, and preserve findings.
            </p>
            <button
              onClick={() => setIsCreateOpen(true)}
              className="mt-2 px-4 py-2 bg-[#19B89A] hover:bg-[#16A489] text-[#0F1115] font-semibold text-xs rounded-lg transition-colors cursor-pointer"
            >
              Create New Case
            </button>
          </div>
        ) : (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {cases.map((c) => (
              <div
                key={c.id}
                onClick={() => handleOpenCase(c.id)}
                className="group bg-[#171A20] hover:bg-[#1D2128] border border-[#2A3038] hover:border-[#3A424E] rounded-xl p-5 transition-all cursor-pointer flex flex-col justify-between space-y-4 shadow-sm"
              >
                <div className="space-y-2.5">
                  <div className="flex items-center justify-between gap-2">
                    <span className="font-mono text-xs text-[#19B89A] font-semibold">
                      {c.case_number}
                    </span>
                    <div className="flex items-center gap-1.5">
                      <span className={`text-[10px] font-medium px-2 py-0.5 rounded border ${priorityColor(c.priority)}`}>
                        {c.priority}
                      </span>
                      <span className={`text-[10px] font-medium px-2 py-0.5 rounded border ${statusColor(c.status)}`}>
                        {c.status}
                      </span>
                    </div>
                  </div>

                  <h3 className="text-base font-bold text-[#F5F7FA] group-hover:text-[#19B89A] transition-colors leading-snug">
                    {c.title}
                  </h3>

                  {c.description && (
                    <p className="text-xs text-[#A7AFBA] line-clamp-2 leading-relaxed">
                      {c.description}
                    </p>
                  )}
                </div>

                {/* Secondary Information & Open Action */}
                <div className="pt-3 border-t border-[#2A3038] flex items-center justify-between text-xs">
                  <div className="text-[#737C87] text-[11px] font-mono">
                    <span>{c.counts?.videos ?? 0} videos</span>
                    <span className="mx-1">&bull;</span>
                    <span>{c.counts?.incidents ?? 0} incidents</span>
                    <span className="mx-1">&bull;</span>
                    <span>{c.counts?.evidence ?? 0} evidence</span>
                  </div>

                  <div className="flex items-center gap-2">
                    <span className="text-[#19B89A] font-medium group-hover:translate-x-0.5 transition-transform flex items-center gap-1">
                      Open Case &rarr;
                    </span>
                    <button
                      onClick={(e) => handleDelete(e, c.id, c.case_number)}
                      className="text-[#737C87] hover:text-rose-400 p-1 rounded transition-colors"
                      title="Delete case"
                    >
                      &times;
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
          <div className="bg-[#171A20] border border-[#2A3038] rounded-xl max-w-lg w-full p-6 space-y-4 shadow-2xl">
            <div className="flex items-center justify-between border-b border-[#2A3038] pb-3">
              <h3 className="text-base font-bold text-[#F5F7FA]">Start New Security Case</h3>
              <button
                onClick={() => setIsCreateOpen(false)}
                className="text-[#737C87] hover:text-[#F5F7FA] text-lg cursor-pointer"
              >
                ✕
              </button>
            </div>

            {createError && (
              <div className="p-3 bg-rose-500/10 border border-rose-500/30 rounded-lg text-rose-300 text-xs">
                {createError}
              </div>
            )}

            <form onSubmit={handleCreateSubmit} className="space-y-4 text-xs">
              <div>
                <label className="block text-[#737C87] mb-1 font-medium">Case Title *</label>
                <input
                  type="text"
                  required
                  value={newTitle}
                  onChange={(e) => setNewTitle(e.target.value)}
                  placeholder="e.g. North Gate Perimeter Intrusion Investigation"
                  className="w-full bg-[#1D2128] border border-[#2A3038] rounded-lg p-2.5 text-[#F5F7FA] placeholder-[#737C87] focus:outline-none focus:border-[#19B89A]"
                />
              </div>

              <div>
                <label className="block text-[#737C87] mb-1 font-medium">Description</label>
                <textarea
                  rows={3}
                  value={newDesc}
                  onChange={(e) => setNewDesc(e.target.value)}
                  placeholder="Investigation summary, operational context, and initial findings..."
                  className="w-full bg-[#1D2128] border border-[#2A3038] rounded-lg p-2.5 text-[#F5F7FA] placeholder-[#737C87] focus:outline-none focus:border-[#19B89A] resize-none"
                />
              </div>

              <div className="grid grid-cols-2 gap-3">
                <div>
                  <label className="block text-[#737C87] mb-1 font-medium">Priority</label>
                  <select
                    value={newPriority}
                    onChange={(e) => setNewPriority(e.target.value as any)}
                    className="w-full bg-[#1D2128] border border-[#2A3038] rounded-lg p-2.5 text-[#F5F7FA] focus:outline-none focus:border-[#19B89A]"
                  >
                    <option value="LOW">Low</option>
                    <option value="MEDIUM">Medium</option>
                    <option value="HIGH">High</option>
                    <option value="CRITICAL">Critical</option>
                  </select>
                </div>

                <div>
                  <label className="block text-[#737C87] mb-1 font-medium">Assigned Investigator</label>
                  <input
                    type="text"
                    value={newInvestigator}
                    onChange={(e) => setNewInvestigator(e.target.value)}
                    className="w-full bg-[#1D2128] border border-[#2A3038] rounded-lg p-2.5 text-[#F5F7FA] focus:outline-none focus:border-[#19B89A]"
                  />
                </div>
              </div>

              <div className="flex justify-end gap-2 pt-3 border-t border-[#2A3038]">
                <button
                  type="button"
                  onClick={() => setIsCreateOpen(false)}
                  className="px-4 py-2 bg-[#1D2128] hover:bg-[#2A3038] text-[#F5F7FA] rounded-lg font-medium cursor-pointer"
                >
                  Cancel
                </button>
                <button
                  type="submit"
                  disabled={isSubmitting}
                  className="px-5 py-2 bg-[#19B89A] hover:bg-[#16A489] text-[#0F1115] font-semibold rounded-lg cursor-pointer transition-colors"
                >
                  {isSubmitting ? "Creating..." : "Create Case Dossier"}
                </button>
              </div>
            </form>
          </div>
        </div>
      )}
    </div>
  );
}
