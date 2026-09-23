import { useState } from "react";
import BuySignal from "./components/BuySignal";
import MorningCheck from "./components/MorningCheck";
import ModelPicker from "./components/ModelPicker";
import PriceHistory from "./components/PriceHistory";
import Registry from "./components/Registry";
import RunHistory from "./components/RunHistory";
import TradeHistory from "./components/TradeHistory";
import TrainingPanel from "./components/TrainingPanel";
import WhatIf from "./components/WhatIf";

type Tab = "trading" | "whatif" | "testrun" | "features" | "models" | "data";

const TABS: { id: Tab; label: string }[] = [
  { id: "trading", label: "Trading" },
  { id: "whatif", label: "What if" },
  { id: "testrun", label: "Test run" },
  { id: "features", label: "Feature Store" },
  { id: "models", label: "Models" },
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
        <p className="muted">
          Morning picks and trade history.
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
        <>
          <section>
            <h2>What should I buy this morning?</h2>
            <div className="panel-hero">
              <BuySignal />
            </div>
          </section>

          <section>
            <h2>Trade History</h2>
            <div className="panel">
              <TradeHistory />
            </div>
          </section>
        </>
      )}

      {tab === "whatif" && (
        <section>
          <h2>What if</h2>
          <p className="muted">Morning lists, open to close. News + gap down: skip. News + gap up: still buy.</p>
          <div className="panel">
            <WhatIf />
          </div>
        </section>
      )}

      {tab === "testrun" && (
        <section>
          <h2>Test run</h2>
          <p className="muted">Fake opens. Live scan stays.</p>
          <div className="panel">
            <MorningCheck />
          </div>
        </section>
      )}

      {tab === "features" && (
        <section>
          <h2>Registry</h2>
          <p className="muted">Pruned features are excluded from training, not just hidden here.</p>
          <div className="panel">
            <Registry pendingFeature={pendingFeature} onFeatureFocused={() => setPendingFeature(null)} />
          </div>
        </section>
      )}

      {tab === "models" && (
        <>
          <section>
            <h2>Training</h2>
            <div className="panel">
              <ModelPicker />
              <TrainingPanel />
            </div>
          </section>

          <section>
            <h2>Training History</h2>
            <div className="panel">
              <RunHistory />
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
