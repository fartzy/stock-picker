import { useState } from "react";
import CoverageChart from "./components/CoverageChart";
import CorrelationHeatmap from "./components/CorrelationHeatmap";
import Registry from "./components/Registry";
import TradeHistory from "./components/TradeHistory";

type Tab = "trading" | "features";

const TABS: { id: Tab; label: string }[] = [
  { id: "trading", label: "Trading" },
  { id: "features", label: "Feature Store" },
];

export default function App() {
  const [tab, setTab] = useState<Tab>("trading");

  return (
    <div className="page">
      <header>
        <h1>stock-picker</h1>
        <p className="muted">Feature catalog, coverage, correlation, and registry -- served live from the FastAPI backend.</p>
      </header>

      <div className="tab-bar">
        {TABS.map((t) => (
          <button
            key={t.id}
            className={`tab-button ${tab === t.id ? "active" : ""}`}
            onClick={() => setTab(t.id)}
          >
            {t.label}
          </button>
        ))}
      </div>

      {tab === "trading" && (
        <section>
          <h2>Trade History</h2>
          <div className="panel">
            <TradeHistory />
          </div>
        </section>
      )}

      {tab === "features" && (
        <>
          <section>
            <h2>Registry</h2>
            <div className="panel">
              <Registry />
            </div>
          </section>

          <section>
            <h2>Feature Coverage</h2>
            <p className="summary-line">
              % of rows with a real (non-null) value for that feature, sorted lowest first -- naturally lower
              for longer-lookback features (a 120-day window can't populate until day 120), not a data-quality
              problem.
            </p>
            <div className="panel">
              <CoverageChart />
            </div>
          </section>

          <section>
            <h2>Correlation &amp; redundancy</h2>
            <div className="panel">
              <CorrelationHeatmap />
            </div>
          </section>
        </>
      )}
    </div>
  );
}
