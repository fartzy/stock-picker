import { useState } from "react";
import { fetchOvernightActuals, type OvernightActualRow } from "../api";
import { formatUsd } from "../format";
import {
  simulateOvernightHold, summarizeOvernightHold,
  type SimulatedDay, type SimulatedPick, type TradeSizes,
} from "../whatIf";

const HOLD_PRESETS = [10, 30, 50, 100];

function outcomeLabel(status: string, reason: string | null | undefined): string {
  if (status === "awaiting_next_open") return "Awaiting next open";
  if (status === "awaiting_close") return "Awaiting close";
  if (status === "price_mismatch") return "Close sources disagree";
  if (status === "corporate_action") return `Corporate action${reason ? `: ${reason}` : ""}`;
  return reason ? `Unavailable: ${reason}` : "Unavailable";
}

function StrategyHoldRows({
  label, rows, dollarsPerTrade, baselinePnl, fraction, actuals,
}: {
  label: string;
  rows: SimulatedPick[];
  dollarsPerTrade: number;
  baselinePnl: number | null;
  fraction: number;
  actuals: Map<string, OvernightActualRow>;
}) {
  if (!rows.length) return null;
  const outcomes = rows.map((row) => simulateOvernightHold(row, actuals.get(row.ticker), dollarsPerTrade, fraction));
  const summary = summarizeOvernightHold(outcomes, baselinePnl);
  return (
    <div className="overnight-hold-list">
      <h4>{label}</h4>
      <p className="muted">
        {summary.scenarioPnl === null
          ? `${summary.observed}/${summary.included} included picks have verified next opens; no complete basket total yet.`
          : `Additional ${formatUsd(summary.incrementalPnl)} versus selling all at the close · hypothetical P&L ${formatUsd(summary.scenarioPnl)} (baseline ${formatUsd(baselinePnl!)}).`}
      </p>
      <div className="overnight-hold-table-wrap"><table className="overnight-hold-table">
        <thead><tr><th>Ticker</th><th>Close</th><th>Next open</th><th>Held portion</th><th>Extra vs close</th><th>End value</th></tr></thead>
        <tbody>{rows.map((row, index) => {
          const actual = actuals.get(row.ticker);
          const outcome = outcomes[index];
          return <tr key={`${row.rank}-${row.ticker}`}>
            <td><strong>{row.ticker}</strong>{!outcome.includedInTotals && <small> · excluded from totals</small>}</td>
            <td>{row.close_price === null ? "—" : formatUsd(row.close_price)}</td>
            <td>{actual?.status === "observed" && actual.next_open !== null ? formatUsd(actual.next_open) : "—"}</td>
            <td>{row.open_price && outcome.status === "observed"
              ? `${((dollarsPerTrade / row.open_price) * fraction).toFixed(2)} shares`
              : "—"}</td>
            <td>{outcome.incrementalPnl === null
              ? <span className="muted">{outcomeLabel(actual?.status === "corporate_action" ? "corporate_action" : outcome.status, actual?.reason)}</span>
              : <span className={outcome.incrementalPnl >= 0 ? "quote-diff-up" : "quote-diff-down"}>{formatUsd(outcome.incrementalPnl)}</span>}</td>
            <td>{outcome.endingValue === null ? "—" : formatUsd(outcome.endingValue)}</td>
          </tr>;
        })}</tbody>
      </table></div>
    </div>
  );
}

export default function OvernightHoldComparison({ day, kind, tradeSizes }: {
  day: SimulatedDay;
  kind: "rank" | "fit" | "both";
  tradeSizes: TradeSizes;
}) {
  const [presetPercent, setPresetPercent] = useState<number | null>(null);
  const [customPercent, setCustomPercent] = useState("");
  const [actuals, setActuals] = useState<Map<string, OvernightActualRow> | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const rows = [
    ...(kind !== "rank" ? day.fit : []),
    ...(kind !== "fit" ? day.rank : []),
  ];
  const customValue = Number(customPercent);
  const holdPercent = customPercent !== ""
    ? (Number.isFinite(customValue) && customValue > 0 && customValue <= 100 ? customValue : null)
    : presetPercent;

  async function load() {
    if (holdPercent === null) return;
    const tickers = [...new Set(rows.map((row) => row.ticker))];
    if (tickers.length > 30) {
      setError("Select at most 30 distinct names in the Rank/Fit top controls for this comparison.");
      return;
    }
    setBusy(true);
    setError(null);
    try {
      const response = await fetchOvernightActuals(day.as_of, tickers);
      setActuals(new Map(response.rows.map((row) => [row.ticker, row])));
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "Could not load observed next opens");
    } finally {
      setBusy(false);
    }
  }

  return (
    <details className="overnight-hold-panel">
      <summary><span>Hold to next open</span><small>Historical comparison</small></summary>
      <div className="overnight-hold-controls">
        <span className="overnight-hold-label">Hold</span>
        <div className="overnight-hold-presets">{HOLD_PRESETS.map((percent) => (
          <button key={percent} type="button" className={percent === presetPercent && customPercent === "" ? "active" : ""}
            aria-pressed={percent === presetPercent && customPercent === ""}
            onClick={() => { setPresetPercent(percent); setCustomPercent(""); }}>{percent}%</button>
        ))}</div>
        <label className="overnight-hold-custom">Custom <input className="form-input" type="number" inputMode="numeric" min="1" max="100" step="1"
          value={customPercent} onChange={(event) => { setCustomPercent(event.target.value); setPresetPercent(null); }} placeholder="%" aria-label="Custom hold percentage" /></label>
        <button className="btn-primary" type="button" disabled={busy || rows.length === 0 || holdPercent === null} onClick={load}>
          {busy ? "Checking…" : actuals ? "Refresh opens" : "Compare"}
        </button>
      </div>
      <div className="overnight-hold-footnote">
        <span>Actual next opens · Rank/Fit totals unchanged</span>
        <details><summary>Method</summary><p>Historical comparison using verified raw prices. Fractional shares; fees and slippage excluded. Not a forecast.</p></details>
      </div>
      {error && <p className="error" role="alert">{error}</p>}
      {actuals && holdPercent !== null && <>
        {kind !== "rank" && <StrategyHoldRows label="Fit" rows={day.fit} dollarsPerTrade={tradeSizes.fit}
          baselinePnl={day.fit_money.pnl} fraction={holdPercent / 100} actuals={actuals} />}
        {kind !== "fit" && <StrategyHoldRows label="Rank" rows={day.rank} dollarsPerTrade={tradeSizes.rank}
          baselinePnl={day.rank_money.pnl} fraction={holdPercent / 100} actuals={actuals} />}
        <p className="muted">Incomplete baskets exclude missing opens, corporate actions, and close-price mismatches.</p>
      </>}
    </details>
  );
}
