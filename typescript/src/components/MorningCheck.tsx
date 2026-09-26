import { useState } from "react";
import {
  fetchMorningCheck,
  fetchMorningCheckRuns,
  loadMorningCheck,
  runMorningCheck,
  type MorningCheckResponse,
  type TimedPass,
} from "../api";
import { useFetchData } from "../useFetchData";
import { DataTable, NewsCell, ScoreCell, TickerCell, UsdCell } from "./DataTable";

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
        <div style={{ marginTop: "var(--space-2)" }}>
          <DataTable
            rows={pass.picks}
            rowKey={(pick) => `${pass.which}-${pick.ticker}`}
            columns={[
              { key: "ticker", header: "Ticker", cell: (pick) => <TickerCell ticker={pick.ticker} /> },
              {
                key: "score",
                header: isRank ? "Score" : "Predicted",
                numeric: true,
                cell: (pick) => <ScoreCell value={pick.predicted_return} isRank={isRank} />,
              },
              { key: "open", header: "Fake open", numeric: true, cell: (pick) => <UsdCell value={pick.open_price} /> },
              { key: "news", header: "News", cell: (pick) => <NewsCell flag={pick.news_flag} /> },
            ]}
          />
        </div>
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
  const savedRuns = useFetchData(fetchMorningCheckRuns, {
    deps: [refresh, data?.completed_at],
  });
  const runs = savedRuns.data;

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
      <div className="meta-row">
        <div className="list-toggle" role="group" aria-label="Which models">
          {(["rank", "fit", "both"] as const).map((option) => (
            <button
              key={option}
              type="button"
              className={which === option ? "active" : ""}
              onClick={() => setWhich(option)}
              disabled={running || busy}
            >
              {option === "both" ? "Both" : option === "rank" ? "Rank" : "Fit"}
            </button>
          ))}
        </div>
        <button type="button" className="btn-primary" onClick={start} disabled={running || busy}>
          {running ? "Running..." : "Run morning check"}
        </button>
        {runs && runs.length > 0 && (
          <select
            className="form-select"
            value=""
            disabled={running || busy}
            onChange={async (event) => {
              const id = event.target.value;
              if (!id) return;
              setBusy(true);
              try {
                await loadMorningCheck(id);
                setRefresh((n) => n + 1);
              } finally {
                setBusy(false);
              }
            }}
          >
            <option value="">Earlier runs</option>
            {runs.map((run) => (
              <option key={run.id} value={run.id}>
                {(run.started_at
                  ? new Date(run.started_at).toLocaleString(undefined, {
                      month: "short",
                      day: "numeric",
                      hour: "numeric",
                      minute: "2-digit",
                    })
                  : run.id) + (run.which ? ` · ${run.which}` : "")}
              </option>
            ))}
          </select>
        )}
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
                <DataTable
                  rows={data.quotes}
                  rowKey={(quote) => quote.ticker}
                  columns={[
                    { key: "ticker", header: "Ticker", cell: (quote) => <TickerCell ticker={quote.ticker} /> },
                    { key: "open", header: "Fake open", numeric: true, cell: (quote) => <UsdCell value={quote.fake_open} /> },
                    { key: "close", header: "Last close", numeric: true, cell: (quote) => <UsdCell value={quote.last_close} /> },
                  ]}
                />
              </div>
            </details>
          )}
        </div>
      )}
    </div>
  );
}
