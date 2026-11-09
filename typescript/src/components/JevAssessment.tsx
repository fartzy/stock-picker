import { useEffect, useRef, useState } from "react";
import { assessJev, type JevForecast } from "../jevApi";
import { JEV_QUESTION_LABELS, jevPriceBasis, jevProbability, jevTime } from "../jevDisplay";

const DEFAULT_DOWNSIDE_PERCENT = "3";
const MAX_VISIBLE_HEADLINES = 8;

export default function JevAssessment({ ticker, kind, onClose }: {
  ticker: string;
  kind: "rank" | "fit";
  onClose: () => void;
}) {
  const dialog = useRef<HTMLDialogElement>(null);
  const request = useRef(0);
  const [mode, setMode] = useState<"entry" | "overnight">("entry");
  const [entry, setEntry] = useState("");
  const [cost, setCost] = useState("0");
  const [downside, setDownside] = useState(DEFAULT_DOWNSIDE_PERCENT);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [forecast, setForecast] = useState<JevForecast | null>(null);

  useEffect(() => {
    const element = dialog.current;
    element?.showModal();
    return () => { request.current += 1; element?.close(); };
  }, []);

  function invalidate() {
    request.current += 1;
    setBusy(false);
    setError(null);
    setForecast(null);
  }

  async function assess() {
    const thisRequest = ++request.current;
    setBusy(true);
    setError(null);
    setForecast(null);
    try {
      const result = await assessJev({ ticker, kind, mode,
        ...(mode === "entry" && entry.trim() ? { entry_price: Number(entry) } : {}),
        round_trip_cost_per_share: Number(cost), downside_fraction: Number(downside) / 100,
      });
      if (thisRequest === request.current) setForecast(result);
    } catch (cause) {
      if (thisRequest === request.current) setError(cause instanceof Error ? cause.message : "Assessment unavailable");
    } finally {
      if (thisRequest === request.current) setBusy(false);
    }
  }

  return <dialog ref={dialog} className="overnight-quick-dialog jev-dialog" aria-labelledby="jev-title"
    onCancel={(event) => { event.preventDefault(); onClose(); }}
    onClick={(event) => { if (event.target === event.currentTarget) onClose(); }}>
    <div className="overnight-quick-inner">
      <div className="overnight-quick-head">
        <div><span>Jev assessment · {kind === "rank" ? "Rank" : "Fit"}</span><h2 id="jev-title">{ticker} from here</h2></div>
        <button type="button" className="overnight-quick-close" aria-label="Close Jev assessment" onClick={onClose}>×</button>
      </div>
      <p className="muted">Experimental probabilities. These have not yet earned a role in automatic selection or position sizing.</p>
      <form onSubmit={(event) => { event.preventDefault(); void assess(); }}>
        <div className="jev-form-grid">
          <label>Decision
            <select className="form-select" value={mode} onChange={(event) => { setMode(event.target.value as "entry" | "overnight"); invalidate(); }}>
              <option value="entry">Enter now, exit at close</option>
              <option value="overnight">Near-close overnight assessment</option>
            </select>
          </label>
          {mode === "entry" && <label>Assumed entry price ($)
            <input className="form-input" type="number" min="0.0001" step="any" inputMode="decimal" value={entry}
              placeholder="Use a fresh observed trade" onChange={(event) => { setEntry(event.target.value); invalidate(); }} />
          </label>}
          <label>Round-trip costs per share ($)
            <input className="form-input" type="number" min="0" step="any" inputMode="decimal" value={cost} required
              onChange={(event) => { setCost(event.target.value); invalidate(); }} />
          </label>
          {mode === "entry" && <label>Downside level below entry (%)
            <input className="form-input" type="number" min="0.01" max="99" step="any" inputMode="decimal" value={downside} required
              onChange={(event) => { setDownside(event.target.value); invalidate(); }} />
          </label>}
        </div>
        <p className="muted jev-note">{mode === "entry" && "Blank price fetches a dated last trade when you assess. It is a reference price, not a guaranteed fill. "}Costs include your assumed fees, spread and slippage; zero means no cost allowance.</p>
        {mode === "overnight" && <p className="muted jev-note">A separate actual-close-to-next-open question, available in the final 30 minutes of the cash session. Its entry price is resolved from the official close later.</p>}
        <button className="btn-primary" type="submit" disabled={busy}>{busy ? "Assessing and saving…" : mode === "entry" ? "Assess this entry" : "Assess overnight"}</button>
      </form>
      {error && <p className="error" role="alert">{error}</p>}
      {forecast && <section className="jev-result" aria-live="polite">
        {forecast.issue && <p className="error">Assessment saved, but probabilities are unavailable: {forecast.issue}</p>}
        <div className="jev-probabilities">
          {Object.entries(forecast.answers).filter(([key]) => key in JEV_QUESTION_LABELS).map(([key, probability]) =>
            <div className="jev-probability" key={key}><span>{JEV_QUESTION_LABELS[key]}</span><strong>{jevProbability(probability)}</strong></div>)}
        </div>
        <p>{jevPriceBasis(forecast.target.entry_basis)} <strong>{forecast.target.entry_price === null ? "Pending close" : `$${forecast.target.entry_price.toFixed(4)}`}</strong> · {jevTime(forecast.target.entry_observed_at)}<br />
          Exit: {jevTime(forecast.target.exit_at)} · Costs: ${forecast.target.round_trip_cost_per_share.toFixed(4)}/share</p>
        {forecast.target.downside_price != null && <p>Downside event: a trade at or below <strong>${forecast.target.downside_price.toFixed(4)}</strong> after this assessment becomes available and before close.</p>}
        <p className="muted">News coverage: <strong>{forecast.coverage.news_status}</strong> · checked {jevTime(forecast.coverage.news_checked_at)}.<br />
          Features through {forecast.coverage.feature_snapshot_date ?? "unknown"} · {forecast.coverage.news_sources.join(", ") || "No news sources recorded"}</p>
        {[...forecast.coverage.news_issues, ...forecast.coverage.issues].length > 0 && <p className="jev-coverage">Evidence limits: {[...forecast.coverage.news_issues, ...forecast.coverage.issues].join("; ")}</p>}
        <details className="jev-evidence"><summary>Evidence used</summary>
          {(forecast.state?.news?.articles?.length ?? 0) > 0
            ? <ul>{forecast.state?.news?.articles?.slice(0, MAX_VISIBLE_HEADLINES).map((article, index) =>
              <li key={`${article.published_at}-${index}`}><strong>{article.headline ?? "Headline unavailable"}</strong><br />
                <span className="muted">{article.source ?? "Source unavailable"} · {jevTime(article.published_at)}</span>
                {article.summary && <p>{article.summary}</p>}
              </li>)}</ul>
            : <p className="muted">No dated headlines were included. Check the coverage status above.</p>}
        </details>
        <p className="muted jev-note">Saved {jevTime(forecast.completed_at)} · {forecast.model ?? "Model unavailable"}. Reassess to create a new record. The original stays in history.</p>
      </section>}
    </div>
  </dialog>;
}
