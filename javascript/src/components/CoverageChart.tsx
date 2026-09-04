import { useEffect, useState } from "react";
import { fetchCoverage, type CoverageResponse } from "../api";

export default function CoverageChart() {
  const [data, setData] = useState<CoverageResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetchCoverage().then(setData).catch((err) => setError(String(err)));
  }, []);

  if (error) return <p className="error">{error}</p>;
  if (!data) return <p className="muted">Loading coverage...</p>;

  const rows = Object.entries(data.coverage).sort((a, b) => a[1] - b[1]);

  return (
    <div style={{ maxHeight: 420, overflowY: "auto" }}>
      {rows.map(([name, pct]) => (
        <div className="cov-row" key={name}>
          <span>{name}</span>
          <div className="cov-track">
            <div
              className="cov-fill"
              style={{ width: `${Math.round(pct * 100)}%`, background: pct > 0.7 ? "var(--good)" : "var(--bad)" }}
            />
          </div>
          <span className="muted">{Math.round(pct * 100)}%</span>
        </div>
      ))}
    </div>
  );
}
