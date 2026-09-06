import { useState } from "react";
import CorrelationHeatmap from "./components/CorrelationHeatmap";
import PriceHistory from "./components/PriceHistory";
import Registry from "./components/Registry";
import TradeHistory from "./components/TradeHistory";
import TrainingPanel from "./components/TrainingPanel";

type Tab = "trading" | "features" | "prices";

const TABS: { id: Tab; label: string }[] = [
  { id: "trading", label: "Trading" },
  { id: "features", label: "Feature Store" },
  { id: "prices", label: "Prices" },
];

export default function App() {
  const [tab, setTab] = useState<Tab>("trading");

  return (
    <div className="page">
      <header>
        <h1>
          stock<span style={{ color: "var(--accent)" }}>picker</span>
        </h1>
        <p className="muted">
          Trading with live P&amp;L, a feature store (registry, coverage, correlation, pruning), and
          price history, served live from the FastAPI backend.
        </p>
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
            <h2>Training</h2>
            <div className="panel">
              <TrainingPanel />
            </div>
          </section>

          <section>
            <h2>Registry</h2>
            <p className="muted">
              Coverage, importance, and pruning are per-feature attributes below, not separate views.
              Sort by either to scan for problems. Pruned/excluded features are actually excluded from
              training, not just hidden here.
            </p>
            <div className="panel">
              <Registry />
            </div>
          </section>

          <section>
            <h2>Correlation &amp; Redundancy</h2>
            <div className="panel">
              <CorrelationHeatmap />
            </div>
          </section>
        </>
      )}

      {tab === "prices" && (
        <section>
          <h2>Price History</h2>
          <div className="panel">
            <PriceHistory />
          </div>
        </section>
      )}
    </div>
  );
}
