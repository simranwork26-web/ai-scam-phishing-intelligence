type SignalCardProps = {
  label: string;
  description: string;
  active?: boolean;
};

export default function SignalCard({
  label,
  description,
  active = true,
}: SignalCardProps) {
  return (
    <div
      className={`rounded-2xl border p-4 transition ${
        active
          ? "border-white/10 bg-white/[0.04]"
          : "border-white/5 bg-white/[0.02] opacity-45"
      }`}
    >
      <div className="flex items-start gap-3">
        <span
          className={`mt-1 h-2.5 w-2.5 shrink-0 rounded-full ${
            active
              ? "bg-white shadow-[0_0_12px_rgba(255,255,255,0.65)]"
              : "bg-zinc-700"
          }`}
        />

        <div>
          <p className="text-sm font-medium text-white">{label}</p>
          <p className="mt-1 text-sm leading-6 text-zinc-400">
            {description}
          </p>
        </div>
      </div>
    </div>
  );
}
