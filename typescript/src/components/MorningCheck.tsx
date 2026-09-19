import { useState } from "react";
import {
  fetchMorningCheck,
  runMorningCheck,
  type MorningCheckResponse,
  type TimedPass,
} from "../api";
import { formatUsd } from "../format";
import { useFetchData } from "../useFetchData";

const POLL_MS = 2000;

function seconds(value: number | null | undefined): string {
  if (value === null || value === undefined) return "—";
  return `${value.toFixed(1)}s`;
}

function PassBlock({ pass, isRank }: { pass: TimedPass; isRank: boolean }) {
  return (
    <details className="view-card">
      <summary>
        <strong>{pass.which}</strong>
        {` · ${seconds(pass.seconds)} · scored ${pass.scored_count} · ${pass.n_picks} picks · ${pass.skipped_count} skipped`}
      </summary>
      {pass.picks.length > 0 && (
        <table className="trade-table" style={{ marginTop: "var(--space-2)" }}>
          <thead>
            <tr>
              <th>Ticker</th>
              <th className="trade-num">{isRank ? "Score" : "Predicted"}</th>
              <th className="trade-num">Fake open</th>
            </tr>
          </thead>
          <tbody>
            {pass.picks.map((pick) => (
              <tr key={`${pass.which}-${pick.ticker}`}>
                <td className="trade-ticker">{pick.ticker}</td>
                <td className="trade-num">
                  {isRank
                    ? pick.predicted_return.toFixed(4)
                    : `${(pick.predicted_return * 100).toFixed(2)}%`}
                </td>
                <td className="trade-num">{formatUsd(pick.open_price)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </details>
  );
}

export default function MorningCheck() {
  const [which, setWhich] = useState<"rank" | "fit" | "both">("both");
  const [refresh, setRefresh] = useState(0);
  const [busy, setBusy] = useState(false);
  const { data, error } = useFetchData<MorningCheckResponse>(fetchMorningCheck, {
    deps: [refresh],
    intervalMs: POLL_MS,
  });

  async function start() {
    setBusy(true);
    try {
      await runMorningCheck(which);
      setRefresh((n) => n + 1);
    } finally {
      setBusy(false);
    }
  }

  const running = data?.status === "running";

  return (
    <div>
      <p className="muted">
        Fake opens = last Close for every name. No Yahoo. Rank and Fit run at the
        same time when you press Both.
      </p>
      <div className="meta-row" style={{ marginTop: "var(--space-2)" }}>
        {(["rank", "fit", "both"] as const).map((option) => (
          <button
            key={option}
            type="button"
            className={which === option ? "primary-button" : ""}
            onClick={() => setWhich(option)}
            disabled={running || busy}
          >
            {option === "both" ? "Both (parallel)" : option === "rank" ? "Rank" : "Fit"}
          </button>
        ))}
        <button type="button" className="primary-button" onClick={start} disabled={running || busy}>
          {running ? "Running..." : "Run morning check"}
        </button>
      </div>
      {error && <p className="error">{error}</p>}
      {data && data.status !== "idle" && (
        <div style={{ marginTop: "var(--space-3)" }}>
          <p className="period-row">
            <strong>{data.status}</strong>
            {data.quote_seconds !== null && ` · fake quotes ${seconds(data.quote_seconds)} · ${data.n_quotes} names`}
            {data.error ? ` · ${data.error}` : ""}
          </p>
          {data.passes.map((pass) => (
            <PassBlock key={pass.which} pass={pass} isRank={pass.which === "rank"} />
          ))}
          {data.quotes.length > 0 && (
            <details className="view-card">
              <summary>
                Fake opens · {data.quotes.length} names
              </summary>
              <div style={{ overflowX: "auto", maxHeight: 360, overflowY: "auto" }}>
                <table className="trade-table">
                  <thead>
                    <tr>
                      <th>Ticker</th>
                      <th className="trade-num">Fake open</th>
                      <th className="trade-num">Last close</th>
                    </tr>
                  </thead>
                  <tbody>
                    {data.quotes.map((quote) => (
                      <tr key={quote.ticker}>
                        <td className="trade-ticker">{quote.ticker}</td>
                        <td className="trade-num">{formatUsd(quote.fake_open)}</td>
                        <td className="trade-num">{formatUsd(quote.last_close)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </details>
          )}
        </div>
      )}
    </div>
  );
}
