"use client";

import { useState } from "react";
import { analyzeMessage, type AnalyzeResult } from "@/lib/api";

type AnalysisPanelProps = {
  onAnalysis: (result: AnalyzeResult) => void;
};

export default function AnalysisPanel({ onAnalysis }: AnalysisPanelProps) {
  const [message, setMessage] = useState("");
  const [result, setResult] = useState<Awaited<ReturnType<typeof analyzeMessage>> | null>(null);
  const [analyzing, setAnalyzing] = useState(false);
  const [error, setError] = useState("");

  async function handleAnalyze() {
    if (!message.trim()) return;

    setAnalyzing(true);
    setError("");

    try {
      const response = await analyzeMessage(message.trim());
      setResult(response);
      onAnalysis(response);
    } catch (err) {
      setResult(null);
      setError(err instanceof Error ? err.message : "Analysis failed");
    } finally {
      setAnalyzing(false);
    }
  }

  return (
    <section className="rounded-3xl border border-white/10 bg-white/[0.035] p-6 shadow-2xl shadow-black/20">
      <div className="mb-5">
        <p className="text-xs font-medium uppercase tracking-[0.22em] text-zinc-500">
          Message Analysis
        </p>
        <h2 className="mt-1 text-xl font-semibold text-white">
          Inspect a suspicious message
        </h2>
        <p className="mt-2 max-w-2xl text-sm leading-6 text-zinc-400">
          Paste an email, message, or other suspicious text to assess its
          phishing and scam risk.
        </p>
      </div>

      <textarea
        value={message}
        onChange={(event) => setMessage(event.target.value)}
        placeholder="Paste suspicious email or message here..."
        className="min-h-56 w-full resize-none rounded-2xl border border-white/10 bg-black/20 p-5 text-sm leading-7 text-white outline-none placeholder:text-zinc-600 focus:border-white/20 focus:ring-1 focus:ring-white/10"
        aria-label="Suspicious message"
      />

      <div className="mt-4 flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
        <p className="text-xs text-zinc-600">
          {message.length.toLocaleString()} characters
        </p>

        <button
          type="button"
          onClick={handleAnalyze}
          className="rounded-full border border-white/10 bg-white px-6 py-3 text-sm font-semibold text-black transition hover:bg-zinc-200 disabled:cursor-not-allowed disabled:opacity-40"
          disabled={!message.trim() || analyzing}
        >
          {analyzing ? "Analyzing..." : "Analyze Threat"}
        </button>
      </div>

      {error && (
        <p className="mt-4 text-sm text-red-300">{error}</p>
      )}

      {result && (
        <div className="mt-6 rounded-2xl border border-white/10 bg-black/20 p-5">
          <div className="flex items-end justify-between gap-4">
            <div>
              <p className="text-xs uppercase tracking-[0.2em] text-zinc-500">
                Model assessment
              </p>
              <p className="mt-2 text-3xl font-semibold text-white">
                {(result.risk_score * 100).toFixed(1)}%
              </p>
            </div>

            <p className="text-sm font-medium uppercase tracking-[0.18em] text-zinc-300">
              {result.risk_level} risk
            </p>
          </div>

          <p className="mt-3 text-xs text-zinc-600">
            Model: {result.model} · Device: {result.device}
          </p>
        </div>
      )}

    </section>
  );
}
