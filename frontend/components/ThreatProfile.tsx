type ThreatProfileProps = {
  score: number;
  signals: string[];
};

const signals = [
  { key: "urgency", label: "Urgency", angle: -90 },
  { key: "financial", label: "Financial", angle: -18 },
  { key: "authority", label: "Authority", angle: 54 },
] as const;

export default function ThreatProfile({
  score,
  signals: detectedSignals,
}: ThreatProfileProps) {
  const active = {
    urgency: detectedSignals.includes("urgency"),
    financial: detectedSignals.includes("financial"),
    authority: detectedSignals.includes("authority"),
  };

  return (
    <section className="relative overflow-hidden rounded-3xl border border-white/10 bg-white/[0.035] p-6 shadow-2xl shadow-black/20">
      <div className="absolute inset-0 bg-[radial-gradient(circle_at_center,rgba(255,255,255,0.06),transparent_55%)]" />

      <div className="relative">
        <div className="mb-6 flex items-center justify-between">
          <div>
            <p className="text-xs font-medium uppercase tracking-[0.22em] text-zinc-500">
              Threat Profile
            </p>
            <h2 className="mt-1 text-xl font-semibold text-white">
              Signal distribution
            </h2>
          </div>

          <div className="text-right">
            <p className="text-xs uppercase tracking-[0.18em] text-zinc-500">
              Risk
            </p>
            <p className="text-2xl font-semibold text-white">
              {(score * 100).toFixed(1)}%
            </p>
          </div>
        </div>

        <div className="relative mx-auto aspect-square w-full max-w-[420px]">
          <div className="absolute left-1/2 top-1/2 h-[68%] w-[68%] -translate-x-1/2 -translate-y-1/2 rounded-full border border-white/10" />
          <div className="absolute left-1/2 top-1/2 h-[46%] w-[46%] -translate-x-1/2 -translate-y-1/2 rounded-full border border-white/10" />

          <div className="absolute left-1/2 top-1/2 h-2 w-2 -translate-x-1/2 -translate-y-1/2 rounded-full bg-white shadow-[0_0_24px_rgba(255,255,255,0.75)]" />

          {signals.map((signal) => {
            const radians = (signal.angle * Math.PI) / 180;
            const radius = 40;
            const x = 50 + Math.cos(radians) * radius;
            const y = 50 + Math.sin(radians) * radius;
            const enabled = active[signal.key];

            return (
              <div
                key={signal.key}
                className="absolute"
                style={{
                  left: `${x}%`,
                  top: `${y}%`,
                  transform: "translate(-50%, -50%)",
                }}
              >
                <div
                  className={`flex h-16 min-w-24 flex-col items-center justify-center rounded-2xl border px-3 transition-all ${
                    enabled
                      ? "border-white/20 bg-white/10 shadow-lg shadow-white/5"
                      : "border-white/5 bg-black/20 opacity-45"
                  }`}
                >
                  <span
                    className={`mb-1 h-2 w-2 rounded-full ${
                      enabled
                        ? "bg-white shadow-[0_0_10px_rgba(255,255,255,0.8)]"
                        : "bg-zinc-700"
                    }`}
                  />
                  <span className="text-[11px] font-medium text-zinc-300">
                    {signal.label}
                  </span>
                </div>
              </div>
            );
          })}

          <div className="absolute left-1/2 top-1/2 -translate-x-1/2 -translate-y-1/2 text-center">
            <p className="text-[10px] uppercase tracking-[0.25em] text-zinc-500">
              Assessment
            </p>
            <p className="mt-1 text-lg font-semibold text-white">
              {score >= 0.9
                ? "High Risk"
                : score >= 0.5
                  ? "Review"
                  : "Low Risk"}
            </p>
          </div>
        </div>
      </div>
    </section>
  );
}
