import { fetchCoverage, type CoverageResponse } from "../api";
import { useFetchData } from "../useFetchData";

const MAX_LIST_HEIGHT_PX = 420;
const COVERAGE_GOOD_THRESHOLD = 0.7;

export default function CoverageChart() {
  const { data, error } = useFetchData<CoverageResponse>(fetchCoverage);

  if (error) return <p className="error">{error}</p>;
  if (!data) return <p className="muted">Loading coverage...</p>;

  const rows = Object.entries(data.coverage).sort((a, b) => a[1] - b[1]);

  return (
    <div style={{ maxHeight: MAX_LIST_HEIGHT_PX, overflowY: "auto" }}>
      {rows.map(([name, pct]) => (
        <div className="cov-row" key={name}>
          <span>{name}</span>
          <div className="cov-track">
            <div
              className="cov-fill"
              style={{
                width: `${Math.round(pct * 100)}%`,
                background: pct > COVERAGE_GOOD_THRESHOLD ? "var(--good)" : "var(--bad)",
              }}
            />
          </div>
          <span className="muted">{Math.round(pct * 100)}%</span>
        </div>
      ))}
    </div>
  );
}
