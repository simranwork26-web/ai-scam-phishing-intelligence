"use client";

import { useState } from "react";
import AnalysisPanel from "@/components/AnalysisPanel";
import ThreatProfile from "@/components/ThreatProfile";
import SignalCard from "@/components/SignalCard";
import SystemStatus from "@/components/SystemStatus";
import type { AnalyzeResult } from "@/lib/api";

const SIGNAL_DETAILS: Record<string, { label: string; description: string }> = {
  financial: {
    label: "Financial request",
    description:
      "The message contains language associated with payment or financial action.",
  },
  authority: {
    label: "Authority cue",
    description:
      "The message contains language associated with an organizational or security authority.",
  },
  urgency: {
    label: "Urgency",
    description:
      "The message contains language that pressures the recipient toward immediate action.",
  },
};

export default function Home() {
  const [analysis, setAnalysis] = useState<AnalyzeResult | null>(null);

  const detectedSignals = analysis?.signals ?? [];

  return (
    <main className="min-h-screen bg-[#09090b] text-white">
      <div className="mx-auto min-h-screen max-w-7xl px-6 py-6 lg:px-10">
        <header className="flex items-center justify-between border-b border-white/10 pb-5">
          <div>
            <div className="flex items-center gap-3">
              <span className="h-2.5 w-2.5 rounded-full bg-white shadow-[0_0_14px_rgba(255,255,255,0.8)]" />
              <span className="text-sm font-semibold tracking-[0.22em]">
                SENTINEL
              </span>
            </div>
            <p className="mt-2 text-xs text-zinc-500">
              AI SCAM & PHISHING INTELLIGENCE
            </p>
          </div>

          <SystemStatus />
        </header>

        <section className="py-16 lg:py-20">
          <div className="max-w-4xl">
            <p className="text-xs font-medium uppercase tracking-[0.3em] text-zinc-500">
              Defensive Intelligence
            </p>

            <h1 className="mt-5 text-5xl font-semibold tracking-tight text-white sm:text-6xl lg:text-7xl">
              Know what you&apos;re
              <span className="block text-zinc-500">
                about to trust.
              </span>
            </h1>

            <p className="mt-7 max-w-2xl text-base leading-8 text-zinc-400 sm:text-lg">
              Analyze suspicious messages, inspect URL risk, and surface the
              social-engineering signals behind a phishing assessment.
            </p>
          </div>
        </section>

        <section className="grid gap-6 lg:grid-cols-[1.25fr_0.75fr]">
          <AnalysisPanel onAnalysis={setAnalysis} />

          <div className="space-y-6">
            <ThreatProfile
              score={analysis?.risk_score ?? 0}
              signals={detectedSignals}
            />

            <div className="rounded-3xl border border-white/10 bg-white/[0.035] p-6">
              <p className="text-xs font-medium uppercase tracking-[0.22em] text-zinc-500">
                Detected Evidence
              </p>

              <div className="mt-5 space-y-3">
                {detectedSignals.length > 0 ? (
                  detectedSignals.map((signal) => {
                    const detail = SIGNAL_DETAILS[signal];

                    if (!detail) return null;

                    return (
                      <SignalCard
                        key={signal}
                        label={detail.label}
                        description={detail.description}
                      />
                    );
                  })
                ) : (
                  <div className="rounded-2xl border border-white/5 bg-black/20 p-4">
                    <p className="text-sm text-zinc-500">
                      {analysis
                        ? "No auxiliary manipulation indicators detected."
                        : "Run an analysis to view supporting evidence."}
                    </p>
                  </div>
                )}
              </div>
            </div>
          </div>
        </section>

        <footer className="mt-16 border-t border-white/10 py-6">
          <div className="flex flex-col gap-2 text-xs text-zinc-600 sm:flex-row sm:items-center sm:justify-between">
            <span>Research prototype · Defensive use</span>
            <span>Text · URL · Social Engineering Signals</span>
          </div>
        </footer>
      </div>
    </main>
  );
}
