import { useState } from "react";
import BuySignal from "./components/BuySignal";
import MorningCheck from "./components/MorningCheck";
import ModelPicker from "./components/ModelPicker";
import Overnight from "./components/Overnight";
import OvernightQuickLook from "./components/OvernightQuickLook";
import OvernightModelPanel from "./components/OvernightModelPanel";
import OvernightFeaturePanel from "./components/OvernightFeaturePanel";
import PriceHistory from "./components/PriceHistory";
import Registry from "./components/Registry";
import RunHistory from "./components/RunHistory";
import TradeHistory from "./components/TradeHistory";
import TrainingPanel from "./components/TrainingPanel";
import WhatIf from "./components/WhatIf";

type Tab = "trading" | "overnight" | "whatif" | "testrun" | "features" | "models" | "data";

const TABS: { id: Tab; label: string }[] = [
  { id: "trading", label: "Trading" },
  { id: "overnight", label: "Overnight" },
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
  const [overnightTicker, setOvernightTicker] = useState("");
  const [overnightRequest, setOvernightRequest] = useState(0);
  const [quickTicker, setQuickTicker] = useState<string | null>(null);

  function openOvernight(ticker = "") {
    if (ticker) {
      setQuickTicker(ticker);
      return;
    }
    setOvernightTicker(ticker);
    setOvernightRequest((value) => value + 1);
    setTab("overnight");
  }

  function openFullOvernight() {
    setOvernightTicker(quickTicker ?? "");
    setOvernightRequest((value) => value + 1);
    setQuickTicker(null);
    setTab("overnight");
  }

  return (
    <div className="page">
      <header>
        <h1>
          stock<span style={{ color: "var(--accent)" }}>picker</span>
        </h1>
        <p className="muted">
          Morning picks, overnight scenarios, and trade history.
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
              <BuySignal onOpenOvernight={openOvernight} />
            </div>
          </section>

          <section>
            <h2>Trade History</h2>
            <div className="panel">
              <TradeHistory onOpenOvernight={openOvernight} />
            </div>
          </section>
        </>
      )}

      {tab === "overnight" && (
        <section>
          <h2>Tomorrow’s open</h2>
          <div className="panel"><Overnight key={overnightRequest} initialTicker={overnightTicker} /></div>
        </section>
      )}

      {tab === "whatif" && (
        <section>
          <h2>What if</h2>
          <p className="muted">This book remains an open → close comparison. Historical overnight replay will appear separately once earlier-date model artifacts are available. <button className="overnight-text-link" type="button" onClick={() => openOvernight()}>Open today’s overnight scenario</button></p>
          <div className="panel">
            <WhatIf />
          </div>
        </section>
      )}

      {tab === "testrun" && (
        <section>
          <h2>Test run</h2>
          <p className="muted">Fake opens. Live scan stays.</p>
          <p className="muted">To test an assumed close without changing the morning check, use <button className="overnight-text-link" type="button" onClick={() => openOvernight()}>the overnight scenario</button>.</p>
          <div className="panel">
            <MorningCheck />
          </div>
        </section>
      )}

      {tab === "features" && (
        <section>
          <h2>Registry</h2>
          <p className="muted">Pruned features are excluded from training, not just hidden here.</p>
          <OvernightFeaturePanel />
          <div className="panel">
            <Registry pendingFeature={pendingFeature} onFeatureFocused={() => setPendingFeature(null)} />
          </div>
        </section>
      )}

      {tab === "models" && (
        <>
          <section><h2>Overnight model</h2><div className="panel"><OvernightModelPanel onOpenOvernight={() => openOvernight()} /></div></section>
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
              onOpenOvernight={openOvernight}
              onNavigateToFeature={(feature) => {
                setPendingFeature(feature);
                setTab("features");
              }}
            />
          </div>
        </section>
      )}
      {quickTicker && <OvernightQuickLook key={quickTicker} ticker={quickTicker}
        onClose={() => setQuickTicker(null)} onOpenFull={openFullOvernight} />}
    </div>
  );
}
