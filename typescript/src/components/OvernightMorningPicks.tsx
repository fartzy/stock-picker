import { useState } from "react";
import {
  DEFAULT_BUY_THRESHOLD,
  fetchBuySignal,
  type BuySignalResponse,
} from "../api";
import { useFetchData } from "../useFetchData";

const VISIBLE_PICK_COUNT = 12;
const REFRESH_MS = 60_000;

export default function OvernightMorningPicks({
  selectedTicker,
  onSelect,
  openPositions,
}: {
  selectedTicker: string;
  onSelect: (ticker: string) => void;
  openPositions: ReadonlyArray<readonly [string, number]>;
}) {
  const [kind, setKind] = useState<"rank" | "fit">("rank");
  const [showAll, setShowAll] = useState(false);
  const [query, setQuery] = useState("");
  const { data: rank, error: rankError } = useFetchData<BuySignalResponse>(
    () => fetchBuySignal(DEFAULT_BUY_THRESHOLD, false, "rank"),
    { intervalMs: REFRESH_MS },
  );
  const { data: fit, error: fitError } = useFetchData<BuySignalResponse>(
    () => fetchBuySignal(DEFAULT_BUY_THRESHOLD, false, "fit"),
    { intervalMs: REFRESH_MS },
  );
  const active = kind === "rank" ? rank : fit;
  const signals = active?.signals ?? [];
  const visible = showAll ? signals : signals.slice(0, VISIBLE_PICK_COUNT);
  const error = kind === "rank" ? rankError : fitError;
  const candidates = new Map<string, string>();
  openPositions.forEach(([ticker, shares]) => candidates.set(ticker, `Open · ${shares.toLocaleString()} shares`));
  rank?.signals.forEach((signal, index) => {
    if (!candidates.has(signal.ticker)) candidates.set(signal.ticker, `Rank #${index + 1}`);
  });
  fit?.signals.forEach((signal, index) => {
    if (!candidates.has(signal.ticker)) candidates.set(signal.ticker, `Fit #${index + 1}`);
  });
  const search = query.trim().toUpperCase();
  const matches = search
    ? [...candidates].filter(([ticker]) => ticker.includes(search)).slice(0, 10)
    : [];

  function choose(ticker: string) {
    onSelect(ticker);
    setQuery("");
  }

  return (
    <section className="overnight-picks" aria-label="Choose a ticker">
      <div className="overnight-picks-head">
        <div>
          <h4>Choose a ticker</h4>
          <span>Morning picks · {active?.as_of ?? "Loading…"}</span>
        </div>
        <div className="overnight-picks-switch" role="group" aria-label="Pick model">
          {(["rank", "fit"] as const).map((option) => (
            <button
              key={option}
              type="button"
              className={kind === option ? "active" : ""}
              aria-pressed={kind === option}
              onClick={() => { setKind(option); setShowAll(false); }}
            >
              {option === "rank" ? "Rank" : "Fit"}
              <span>{(option === "rank" ? rank : fit)?.signals.length ?? "—"}</span>
            </button>
          ))}
        </div>
      </div>
      {openPositions.length > 0 && !search && (
        <div className="overnight-open-picks">
          <span>Open positions</span>
          {openPositions.map(([ticker, shares]) => (
            <button key={ticker} type="button" className="overnight-suggestion"
              aria-pressed={selectedTicker === ticker} onClick={() => choose(ticker)}>
              <strong>{ticker}</strong><small>{shares.toLocaleString()} shares</small>
            </button>
          ))}
        </div>
      )}
      <input
        className="form-input overnight-ticker-search"
        type="search"
        value={query}
        onChange={(event) => setQuery(event.target.value)}
        onKeyDown={(event) => {
          if (event.key === "Enter" && matches.length > 0) {
            event.preventDefault();
            choose(matches[0][0]);
          }
        }}
        aria-label="Search open positions and morning picks"
        placeholder="Search positions, Rank or Fit…"
      />
      {search ? (
        <div className="overnight-search-results" aria-label="Matching tickers">
          {matches.length > 0 ? matches.map(([ticker, source]) => (
            <button key={ticker} type="button" className="overnight-suggestion"
              aria-pressed={selectedTicker === ticker} onClick={() => choose(ticker)}>
              <strong>{ticker}</strong><small>{source}</small>
            </button>
          )) : <span className="muted">No matching saved picks or positions. You can still enter any ticker below.</span>}
        </div>
      ) : (
        <>
          {error && !active ? <p className="error">Could not load {kind} picks.</p> : null}
          {active && signals.length === 0 ? <p className="muted">No {kind} picks for this scan.</p> : null}
          {signals.length > 0 && (
            <>
              <div className={`overnight-pick-grid ${showAll ? "show-all" : ""}`}>
                {visible.map((signal, index) => (
                  <button
                    key={signal.ticker}
                    type="button"
                    className={selectedTicker === signal.ticker ? "selected" : ""}
                    aria-label={`Use ${signal.ticker} in the overnight scenario`}
                    aria-pressed={selectedTicker === signal.ticker}
                    onClick={() => choose(signal.ticker)}
                  >
                    <span className="overnight-pick-rank">{index + 1}</span>
                    <strong>{signal.ticker}</strong>
                    <small>{kind === "rank" ? signal.predicted_return.toFixed(3) : `${(signal.predicted_return * 100).toFixed(2)}%`}</small>
                    <span className="overnight-pick-arrow" aria-hidden="true">↗</span>
                  </button>
                ))}
              </div>
              {signals.length > VISIBLE_PICK_COUNT && (
                <button className="overnight-picks-more" type="button" onClick={() => setShowAll((value) => !value)}>
                  {showAll ? "Show fewer" : `Show all ${signals.length}`}
                </button>
              )}
            </>
          )}
        </>
      )}
    </section>
  );
}
