"use client";

import { useEffect, useState } from "react";
import { checkBackendHealth } from "@/lib/api";

export default function SystemStatus() {
  const [status, setStatus] = useState<"checking" | "online" | "offline">(
    "checking"
  );

  useEffect(() => {
    checkBackendHealth()
      .then(() => setStatus("online"))
      .catch(() => setStatus("offline"));
  }, []);

  const label =
    status === "online"
      ? "Analysis engine online"
      : status === "offline"
        ? "Analysis engine offline"
        : "Checking analysis engine";

  return (
    <div className="hidden items-center gap-2 text-right sm:flex">
      <span
        className={`h-2 w-2 rounded-full ${
          status === "online"
            ? "bg-white shadow-[0_0_10px_rgba(255,255,255,0.8)]"
            : status === "offline"
              ? "bg-zinc-600"
              : "animate-pulse bg-zinc-500"
        }`}
      />
      <div>
        <p className="text-[10px] uppercase tracking-[0.2em] text-zinc-600">
          System
        </p>
        <p className="mt-1 text-xs text-zinc-400">{label}</p>
      </div>
    </div>
  );
}
