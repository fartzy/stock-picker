import { useEffect, useRef, useState } from "react";
import {
  fetchIntradayEntryModel, fetchIntradayEntryTraining, predictIntradayEntry, trainIntradayEntry,
  type IntradayEntryMetrics, type IntradayEntryPrediction, type IntradayEntryResponse, type IntradayEntryValidation,
} from "../api";
import { entryClock, entryEvidenceMessage, entryPrice, entryProbability, entryReturn, entryScoreAvailable, MAX_ENTRY_TICKERS, parseEntryTickers } from "../intradayEntry";
import { useFetchData } from "../useFetchData";

const metric = (value: number | null | undefined) => value == null || !Number.isFinite(value) ? "—" : value.toFixed(3);
const message = (error: unknown) => error instanceof Error ? error.message : "The request could not complete.";

function MetricsTable({ data }: { data: IntradayEntryMetrics }) {
  return (
    <div className="entry-table-scroll"><table className="trade-table">
      <thead><tr><th>Measure</th><th className="trade-num">Entry model</th><th className="trade-num">Time-of-day baseline</th><th className="trade-num">Logistic baseline</th></tr></thead>
      <tbody>
        <tr><td>Profit probability error <small>(Brier; lower is better)</small></td><td className="trade-num">{metric(data.brier)}</td><td className="trade-num">{metric(data.baseline_brier)}</td><td className="trade-num">{metric(data.logistic_brier)}</td></tr>
        <tr><td>Return error <small>(average absolute error)</small></td><td className="trade-num">{entryReturn(data.return_mae).replace(/^\+/, "")}</td><td className="trade-num">{entryReturn(data.baseline_return_mae).replace(/^\+/, "")}</td><td className="trade-num">—</td></tr>
        <tr><td>Downside probability error <small>(Brier)</small></td><td className="trade-num">{metric(data.downside_brier)}</td><td className="trade-num">—</td><td className="trade-num">—</td></tr>
      </tbody>
    </table></div>
  );
}

function ValidationDetails({ data }: { data: IntradayEntryValidation }) {
  const primary = data.rank1;
  const reliability = primary.reliability ?? [];
  return (
    <details className="entry-details">
      <summary>Rank #1 test results · {primary.sessions} held-out days · {primary.rows.toLocaleString()} snapshots</summary>
      <div className="entry-detail-body">
        <p className="entry-meta">Primary evaluation: Rank #1 · {data.test_start} to {data.test_end}. Repeated snapshots are weighted within days.</p>
        <MetricsTable data={primary} />
        {reliability.length > 0 && (
          <details className="entry-nested-details">
            <summary>Rank #1: do predicted odds match outcomes?</summary>
            <div className="entry-table-scroll"><table className="trade-table">
              <thead><tr><th>Probability range</th><th className="trade-num">Snapshots</th><th className="trade-num">Days</th><th className="trade-num">Predicted</th><th className="trade-num">Actually profitable</th></tr></thead>
              <tbody>{reliability.map((bin) => (
                <tr key={`${bin.lower}-${bin.upper}`}><td>{entryProbability(bin.lower)}–{entryProbability(bin.upper)}</td><td className="trade-num">{bin.count.toLocaleString()}</td><td className="trade-num">{bin.sessions}</td><td className="trade-num">{entryProbability(bin.predicted)}</td><td className="trade-num">{entryProbability(bin.observed)}</td></tr>
              ))}</tbody>
            </table></div>
          </details>
        )}
        <details className="entry-nested-details">
          <summary>All sampled ranks · {data.test_sessions} days · {data.test_rows.toLocaleString()} snapshots</summary>
          <MetricsTable data={data} />
        </details>
        <details className="entry-nested-details">
          <summary>Lower ranks · {data.lower_ranks.sessions} days · {data.lower_ranks.rows.toLocaleString()} snapshots</summary>
          <p className="entry-meta">{data.lower_ranks.stock_days} stock-days. Sparse, selectively cached coverage; not a complete top-20 sample.</p>
          <MetricsTable data={data.lower_ranks} />
        </details>
        <p className="entry-meta">Training {data.train_start}–{data.train_end} · Calibration {data.calibration_start}–{data.calibration_end}</p>
      </div>
    </details>
  );
}

function PredictionRow({ row }: { row: IntradayEntryPrediction }) {
  const available = entryScoreAvailable(row);
  return (
    <tr>
      <td><strong>{row.ticker}</strong>{row.rank != null && <small className="entry-cell-note">Morning #{row.rank}</small>}</td>
      <td className="trade-num">{entryPrice(row.entry_price)}<small className="entry-cell-note">{entryClock(row.observed_at)}</small></td>
      <td className="trade-num">{entryReturn(row.return_from_open)}</td>
      {available ? <>
        <td className="trade-num entry-odds">{entryProbability(row.probability_profit)}</td>
        <td className={`trade-num ${(row.expected_return ?? 0) > 0 ? "positive" : (row.expected_return ?? 0) < 0 ? "negative" : ""}`}>{entryReturn(row.expected_return)}</td>
        <td className="trade-num">{entryProbability(row.probability_loss_2pct)}</td>
        <td className="entry-result-note">
          {row.minutes_remaining == null ? "—" : `${Math.floor(row.minutes_remaining)} min left`}
          {row.warnings.length > 0 && <details className="entry-nested-details">
            <summary>{row.warnings.length} {row.warnings.length === 1 ? "note" : "notes"}</summary>
            {row.warnings.map((warning) => <small className="entry-cell-note" key={warning}>{warning}</small>)}
          </details>}
        </td>
      </> : <td colSpan={4} className="entry-result-note">{row.reason || "No current estimate available."}</td>}
    </tr>
  );
}

export default function IntradayEntryPanel() {
  const [reload, setReload] = useState(0);
  const { data: training, error: trainingError } = useFetchData(fetchIntradayEntryTraining, { deps: [reload], intervalMs: 4000 });
  const { data: model, error: modelError } = useFetchData(fetchIntradayEntryModel, {
    deps: [reload, training?.completed_at], intervalMs: 15000,
  });
  const [input, setInput] = useState("");
  const initialized = useRef(false);
  const request = useRef(0);
  const alive = useRef(true);
  const [starting, setStarting] = useState(false);
  const [refreshing, setRefreshing] = useState(false);
  const [actionError, setActionError] = useState<string | null>(null);
  const [result, setResult] = useState<IntradayEntryResponse | null>(null);
  const parsed = parseEntryTickers(input);
  const trainingBusy = starting || training?.status === "running";

  // React's documented cleanup pattern prevents a late response from an old
  // selection/unmounted panel replacing a current result.
  // https://react.dev/reference/react/useEffect#fetching-data-with-effects
  useEffect(() => { alive.current = true; return () => { alive.current = false; request.current += 1; }; }, []);
  useEffect(() => {
    if (!initialized.current && model?.default_tickers.length) {
      setInput(model.default_tickers.slice(0, 10).join(", "));
      initialized.current = true;
    }
  }, [model?.default_tickers]);
  useEffect(() => {
    request.current += 1;
    setRefreshing(false);
    setResult(null);
  }, [model?.trained_at]);

  async function refresh() {
    if (parsed.error || refreshing) return;
    const token = ++request.current;
    setRefreshing(true); setActionError(null); setResult(null);
    try {
      const next = await predictIntradayEntry(parsed.tickers);
      if (alive.current && token === request.current) setResult(next);
    } catch (error) {
      if (alive.current && token === request.current) setActionError(message(error));
    } finally {
      if (alive.current && token === request.current) setRefreshing(false);
    }
  }

  async function train() {
    if (trainingBusy) return;
    setStarting(true); setActionError(null);
    try {
      await trainIntradayEntry();
    } catch (error) {
      if (alive.current) setActionError(message(error));
    } finally {
      if (alive.current) { setStarting(false); setReload((value) => value + 1); }
    }
  }

  if (!model) return <div className="entry-panel"><p className={modelError ? "error" : "entry-meta"}>{modelError || "Loading entry model…"}</p>{modelError && <button type="button" onClick={() => setReload((value) => value + 1)}>Retry</button>}</div>;
  return (
    <div className="entry-panel">
      <div className="entry-model-head">
        <div><strong>Buy now → today’s close</strong><p className="entry-meta">{model.ready ? `Saved model · ${model.feature_count} features · Trained through ${model.trained_through ?? "—"}` : "Train the helper before checking current entries."}</p></div>
        <div className="entry-training-actions"><span className="entry-badge">Experimental</span><button type="button" disabled={trainingBusy || refreshing} onClick={() => void train()}>{trainingBusy ? "Training…" : model.ready ? "Retrain helper" : "Train helper"}</button></div>
      </div>
      {model.ready && <p className="entry-evidence-note">Experimental: {entryEvidenceMessage(model.validation?.rank1)}</p>}
      {trainingBusy && <p className="entry-status" role="status">{training?.phase || "Training from saved historical prices…"}</p>}
      {(training?.error || trainingError) && <p className="error" role="alert">{training?.error || trainingError}</p>}
      {modelError && <p className="error" role="alert">Model status could not refresh: {modelError}</p>}
      <form className="entry-check" onSubmit={(event) => { event.preventDefault(); void refresh(); }}>
        <div className="entry-selection-heading"><label htmlFor="entry-tickers">Stocks to check</label><div className="entry-rank-shortcuts"><span>Morning Rank</span>{[10, 20].map((count) => <button type="button" key={count} disabled={refreshing || !model.default_tickers.length} onClick={() => { initialized.current = true; setInput(model.default_tickers.slice(0, count).join(", ")); }}>Top {count}</button>)}</div></div>
        <div className="entry-input-row"><textarea id="entry-tickers" rows={2} value={input} spellCheck={false} autoCapitalize="characters" disabled={refreshing} aria-describedby="entry-selection-hint" aria-invalid={Boolean(input && parsed.error)} onChange={(event) => { initialized.current = true; setInput(event.target.value); }} placeholder="Enter stock symbols, separated by commas" /><button className="btn-primary" type="submit" disabled={!model.ready || Boolean(parsed.error) || refreshing}>{refreshing ? "Checking prices…" : "Refresh odds"}</button></div>
        <div id="entry-selection-hint" className="entry-selection-hint"><span>{parsed.error && input ? parsed.error : `${parsed.tickers.length} / ${MAX_ENTRY_TICKERS} stocks selected`}</span><span>New prices · saved model · zero assumed expenses</span></div>
      </form>
      {actionError && <p className="error" role="alert">{actionError}</p>}
      {refreshing && <p className="entry-status" role="status">Updating {parsed.tickers.length} stocks…</p>}
      {result && <div className="entry-results" aria-live="polite">
        <div className="entry-results-head"><strong>Current entry estimates</strong><span>{result.session} · Checked {entryClock(result.generated_at)} Chicago · {(result.elapsed_ms / 1000).toFixed(1)}s</span></div>
        <p className="entry-meta">Each estimate uses the observed entry price below and exits at today’s close.</p>
        <div className="entry-table-scroll"><table className="trade-table entry-results-table">
          <thead><tr><th>Stock</th><th className="trade-num">Entry / time¹</th><th className="trade-num">From open</th><th className="trade-num">Chance of profit</th><th className="trade-num">Expected return</th><th className="trade-num" title="Probability the closing price is more than 2% below the entry price.">Close loss &gt;2%</th><th>Status</th></tr></thead>
          <tbody>{result.predictions.map((row) => <PredictionRow key={row.ticker} row={row} />)}</tbody>
        </table></div>
        <p className="entry-meta">¹ Chicago time. Downside means a closing loss over 2% from entry, not a temporary dip.</p>
      </div>}
      {!result && !refreshing && model.ready && <p className="entry-empty">Choose your stocks and refresh to assess an entry at their latest prices.</p>}
      <p className="entry-evidence-note">Rank #1 is the primary evaluation. Other ranks have sparse, selectively cached historical coverage.</p>
      {model.validation && <ValidationDetails data={model.validation} />}
      <details className="entry-details"><summary>Model inputs · {model.feature_count} features</summary><dl className="entry-feature-list entry-detail-body">{model.features.map((feature) => <div key={feature.name}><dt>{feature.name.replaceAll("_", " ")}</dt><dd>{feature.description}</dd></div>)}</dl></details>
      {!!model.notes.length && <details className="entry-details"><summary>Data and model notes</summary><ul className="entry-model-notes">{model.notes.map((note) => <li key={note}>{note}</li>)}</ul></details>}
    </div>
  );
}
