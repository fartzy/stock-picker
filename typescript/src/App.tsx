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
  // One-shot handoff from the Data tab's feature columns to Registry: set,
  // switched to, then cleared once Registry has scrolled to/highlighted it.
  const [pendingFeature, setPendingFeature] = useState<string | null>(null);

  return (
    <div className="page">
      <header>
        <h1>
          stock<span style={{ color: "var(--accent)" }}>picker</span>
        </h1>
        <p className="muted">Live P&amp;L trading, a feature store, and price history.</p>
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
              Per-feature coverage, importance, correlation, and pruning. Excluded features are
              actually excluded from training, not just hidden here.
            </p>
            <div className="panel">
              <Registry pendingFeature={pendingFeature} onFeatureFocused={() => setPendingFeature(null)} />
            </div>
          </section>
        </>
      )}

      {tab === "data" && (
        <section>
          <h2>Ticker Data</h2>
          <div className="panel">
            <PriceHistory
              onNavigateToFeature={(feature) => {
                setPendingFeature(feature);
                setTab("features");
              }}
            />
          </div>
        </section>
      )}
    </div>
  );
}
