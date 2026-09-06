import { useState } from "react";
import PriceHistory from "./components/PriceHistory";
import Registry from "./components/Registry";
import TradeHistory from "./components/TradeHistory";
import TrainingPanel from "./components/TrainingPanel";

type Tab = "trading" | "features" | "data";

const TABS: { id: Tab; label: string }[] = [
  { id: "trading", label: "Trading" },
  { id: "features", label: "Feature Store" },
  { id: "data", label: "Data" },
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
              Coverage, importance, correlation, and pruning are per-feature attributes below, not
              separate views. Sort by coverage or importance to scan for problems. Pruned/excluded
              features are actually excluded from training, not just hidden here.
            </p>
            <div className="panel">
              <Registry />
            </div>
          </section>
        </>
      )}

      {tab === "data" && (
        <section>
          <h2>Ticker Data</h2>
          <div className="panel">
            <PriceHistory />
          </div>
        </section>
      )}
    </div>
  );
}
