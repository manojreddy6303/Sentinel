"use client";

import React, { useState, useEffect, useCallback } from "react";
import {
  ReportItem,
  listAllReports,
  listVideos,
  VideoListItem,
  generateVideoReport,
  getReportDownloadUrl,
  getReportViewUrl,
} from "@/lib/api";

export default function ReportsWorkspaceView() {
  const [reports, setReports] = useState<ReportItem[]>([]);
  const [videos, setVideos] = useState<VideoListItem[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Generate Report Modal
  const [isGenerateOpen, setIsGenerateOpen] = useState(false);
  const [selectedVideoId, setSelectedVideoId] = useState("");
  const [reportTitle, setReportTitle] = useState("Security Incident Dossier");
  const [reportNotes, setReportNotes] = useState("");
  const [isGenerating, setIsGenerating] = useState(false);
  const [generateError, setGenerateError] = useState<string | null>(null);

  const fetchData = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const [repRes, vidRes] = await Promise.all([
        listAllReports(),
        listVideos({ limit: 100 }),
      ]);
      setReports(repRes.reports || []);
      const processed = (vidRes.videos || []).filter((v) => v.status === "processed");
      setVideos(processed);
      if (processed.length > 0 && !selectedVideoId) {
        setSelectedVideoId(processed[0].id);
      }
    } catch (err: any) {
      setError(err.message || "Failed to load incident dossiers.");
    } finally {
      setLoading(false);
    }
  }, [selectedVideoId]);

  useEffect(() => {
    fetchData();
  }, [fetchData]);

  const handleGenerateSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!selectedVideoId) {
      setGenerateError("Please select a video.");
      return;
    }
    setIsGenerating(true);
    setGenerateError(null);
    try {
      await generateVideoReport(selectedVideoId, {
        title: reportTitle.trim() || "Incident Dossier",
        classification: "CONFIDENTIAL_SECURITY",
      });
      setIsGenerateOpen(false);
      setReportNotes("");
      await fetchData();
    } catch (err: any) {
      setGenerateError(err.message || "Failed to generate report.");
    } finally {
      setIsGenerating(false);
    }
  };

  return (
    <div className="p-6 md:p-8 space-y-8 max-w-[1600px] mx-auto min-h-full">
      {/* Header */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 border-b border-zinc-800/80 pb-6">
        <div>
          <div className="flex items-center gap-3">
            <h1 className="text-2xl font-bold tracking-tight text-white flex items-center gap-2.5">
              <span className="w-3 h-3 rounded-full bg-amber-400 animate-pulse" />
              Reports &amp; Dossiers
            </h1>
            <span className="px-2.5 py-0.5 rounded-full text-xs font-mono font-semibold bg-amber-500/10 text-amber-400 border border-amber-500/30">
              {reports.length} Reports
            </span>
          </div>
          <p className="text-sm text-zinc-400 mt-1.5">
            Cryptographically sealed PDF incident dossiers, investigation summaries, and executive telemetry exports.
          </p>
        </div>

        <button
          onClick={() => setIsGenerateOpen(true)}
          className="flex items-center gap-2 px-4 py-2 bg-emerald-600 hover:bg-emerald-500 text-zinc-950 font-bold rounded-lg text-sm transition-all shadow-md hover:shadow-emerald-500/20"
        >
          <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4v16m8-8H4" />
          </svg>
          Generate Incident Dossier
        </button>
      </div>

      {/* Loading & Error States */}
      {loading && (
        <div className="flex flex-col items-center justify-center py-20 text-zinc-500 space-y-3">
          <div className="w-7 h-7 border-2 border-amber-500 border-t-transparent rounded-full animate-spin" />
          <span className="text-sm font-mono">Loading incident dossiers...</span>
        </div>
      )}

      {error && !loading && (
        <div className="p-4 bg-red-500/10 border border-red-500/30 rounded-xl text-red-400 flex items-center justify-between">
          <span className="text-sm font-medium">{error}</span>
          <button onClick={fetchData} className="px-3 py-1 bg-red-500/20 text-red-300 rounded text-xs">
            Retry
          </button>
        </div>
      )}

      {/* Reports Table */}
      {!loading && !error && reports.length > 0 && (
        <div className="overflow-x-auto bg-zinc-900/60 border border-zinc-800 rounded-xl">
          <table className="w-full text-left text-sm text-zinc-300">
            <thead className="bg-zinc-950/80 text-xs font-semibold text-zinc-400 uppercase tracking-wider border-b border-zinc-800 font-mono">
              <tr>
                <th className="py-3.5 px-5">Report ID</th>
                <th className="py-3.5 px-5">Title</th>
                <th className="py-3.5 px-5">Type</th>
                <th className="py-3.5 px-5">Pages</th>
                <th className="py-3.5 px-5">Size</th>
                <th className="py-3.5 px-5">Generated</th>
                <th className="py-3.5 px-5 text-right">Actions</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-zinc-800/60 font-mono text-xs">
              {reports.map((rep) => (
                <tr key={rep.report_id} className="hover:bg-zinc-800/30 transition-colors">
                  <td className="py-3.5 px-5 text-zinc-300 font-bold">
                    {rep.report_id}
                  </td>
                  <td className="py-3.5 px-5 text-white font-sans font-medium">
                    {rep.title}
                  </td>
                  <td className="py-3.5 px-5">
                    <span className="px-2 py-0.5 rounded bg-zinc-800 text-zinc-300 border border-zinc-700 uppercase text-[11px]">
                      {rep.report_type}
                    </span>
                  </td>
                  <td className="py-3.5 px-5 text-zinc-400">
                    {rep.page_count} {rep.page_count === 1 ? "page" : "pages"}
                  </td>
                  <td className="py-3.5 px-5 text-zinc-400">
                    {(rep.file_size_bytes / 1024).toFixed(1)} KB
                  </td>
                  <td className="py-3.5 px-5 text-zinc-500">
                    {new Date(rep.generated_at).toLocaleString()}
                  </td>
                  <td className="py-3.5 px-5 text-right space-x-2 font-sans font-medium">
                    <a
                      href={getReportViewUrl(rep.report_id)}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="px-2.5 py-1 bg-zinc-800 hover:bg-zinc-700 text-zinc-200 rounded transition-all text-xs inline-block"
                    >
                      View PDF
                    </a>
                    <a
                      href={getReportDownloadUrl(rep.report_id)}
                      download
                      className="px-2.5 py-1 bg-amber-500/20 hover:bg-amber-500/30 text-amber-300 rounded transition-all text-xs inline-block"
                    >
                      Download
                    </a>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {/* Empty State */}
      {!loading && !error && reports.length === 0 && (
        <div className="rounded-xl border border-zinc-800 bg-zinc-900/40 p-12 text-center space-y-4">
          <div className="w-12 h-12 rounded-full bg-zinc-800 flex items-center justify-center mx-auto text-zinc-400">
            <svg className="w-6 h-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
            </svg>
          </div>
          <div className="space-y-1">
            <h3 className="text-base font-bold text-zinc-200">No incident dossiers generated yet</h3>
            <p className="text-sm text-zinc-500 max-w-md mx-auto">
              Generate a formal PDF incident dossier for any processed video to preserve an executive record.
            </p>
          </div>
          <button
            onClick={() => setIsGenerateOpen(true)}
            className="px-4 py-2 bg-emerald-600 hover:bg-emerald-500 text-zinc-950 font-bold rounded-lg text-sm transition-all"
          >
            Generate First Dossier
          </button>
        </div>
      )}

      {/* Generate Report Modal */}
      {isGenerateOpen && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/75 backdrop-blur-sm p-4">
          <div className="bg-zinc-900 border border-zinc-800 rounded-xl max-w-md w-full p-6 space-y-4 shadow-2xl">
            <div className="flex items-center justify-between border-b border-zinc-800 pb-3">
              <h3 className="text-base font-bold text-white">Generate Incident Dossier PDF</h3>
              <button
                onClick={() => setIsGenerateOpen(false)}
                className="text-zinc-500 hover:text-zinc-300"
              >
                ✕
              </button>
            </div>

            {generateError && (
              <div className="p-3 bg-red-500/10 border border-red-500/30 rounded text-red-400 text-xs">
                {generateError}
              </div>
            )}

            <form onSubmit={handleGenerateSubmit} className="space-y-4 text-sm">
              <div>
                <label className="block text-zinc-300 font-medium mb-1">Select Processed Video *</label>
                {videos.length > 0 ? (
                  <select
                    required
                    value={selectedVideoId}
                    onChange={(e) => setSelectedVideoId(e.target.value)}
                    className="w-full bg-zinc-950 border border-zinc-800 rounded-lg p-2.5 text-zinc-200 focus:outline-none focus:border-amber-500 font-mono text-xs"
                  >
                    {videos.map((v) => (
                      <option key={v.id} value={v.id}>
                        {v.filename} ({v.duration_seconds ? `${v.duration_seconds.toFixed(0)}s` : ""})
                      </option>
                    ))}
                  </select>
                ) : (
                  <p className="text-xs text-amber-400">No processed videos found. Please analyze a video first.</p>
                )}
              </div>

              <div>
                <label className="block text-zinc-300 font-medium mb-1">Dossier Title</label>
                <input
                  type="text"
                  required
                  value={reportTitle}
                  onChange={(e) => setReportTitle(e.target.value)}
                  placeholder="e.g. Perimeter Intrusion Incident Dossier"
                  className="w-full bg-zinc-950 border border-zinc-800 rounded-lg p-2.5 text-zinc-200 focus:outline-none focus:border-amber-500 text-xs"
                />
              </div>

              <div>
                <label className="block text-zinc-300 font-medium mb-1">Analyst Notes (Optional)</label>
                <textarea
                  rows={3}
                  value={reportNotes}
                  onChange={(e) => setReportNotes(e.target.value)}
                  placeholder="Additional context or investigative remarks..."
                  className="w-full bg-zinc-950 border border-zinc-800 rounded-lg p-2.5 text-zinc-200 focus:outline-none focus:border-amber-500 text-xs resize-none"
                />
              </div>

              <div className="flex justify-end gap-3 pt-2">
                <button
                  type="button"
                  onClick={() => setIsGenerateOpen(false)}
                  className="px-4 py-2 bg-zinc-800 hover:bg-zinc-700 text-zinc-300 rounded-lg font-medium"
                >
                  Cancel
                </button>
                <button
                  type="submit"
                  disabled={!selectedVideoId || isGenerating || videos.length === 0}
                  className="px-4 py-2 bg-emerald-600 hover:bg-emerald-500 text-zinc-950 font-bold rounded-lg transition-all disabled:opacity-50"
                >
                  {isGenerating ? "Compiling PDF..." : "Generate PDF"}
                </button>
              </div>
            </form>
          </div>
        </div>
      )}
    </div>
  );
}
