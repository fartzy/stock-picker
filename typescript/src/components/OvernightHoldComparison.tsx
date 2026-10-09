import { useState } from "react";
import type { OvernightActualRow } from "../api";
import { formatUsd } from "../format";
import {
  selectOvernightExitRows, simulateOvernightHold, summarizeOvernightExit, summarizeOvernightHistory,
  type MoneySummary, type OvernightExitMode, type OvernightExitRule, type OvernightExitSummary,
  type PickKind, type SimulatedDay, type SimulatedPick, type StrategyKind, type TradeSizes,
} from "../whatIf";

export type LoadOvernightActuals = (asOf: string, tickers: string[], force?: boolean) => Promise<void>;
export type ActualsByDay = ReadonlyMap<string, ReadonlyMap<string, OvernightActualRow>>;

const EMPTY_ACTUALS = new Map<string, OvernightActualRow>();

interface StrategyRows {
  kind: StrategyKind;
  rows: SimulatedPick[];
  money: MoneySummary;
  dollarsPerTrade: number;
}

function strategiesFor(day: SimulatedDay, kind: PickKind, tradeSizes: TradeSizes): StrategyRows[] {
  return [
    ...(kind !== "fit" ? [{ kind: "rank" as const, rows: day.rank, money: day.rank_money, dollarsPerTrade: tradeSizes.rank }] : []),
    ...(kind !== "rank" ? [{ kind: "fit" as const, rows: day.fit, money: day.fit_money, dollarsPerTrade: tradeSizes.fit }] : []),
  ];
}

function MoneyResult({ pnl, endingValue, extraPnl, pending, endingLabel = "End" }: {
  pnl: number | null;
  endingValue: number | null;
  extraPnl?: number | null;
  pending?: string;
  endingLabel?: string;
}) {
  if (pnl === null || endingValue === null) return <span className="overnight-exit-pending">{pending ?? "Pending"}</span>;
  return <span className="overnight-exit-money">
    <strong className={pnl >= 0 ? "quote-diff-up" : "quote-diff-down"}>{formatUsd(pnl)}</strong>
    <small>{endingLabel} {formatUsd(endingValue)}</small>
    {extraPnl !== undefined && extraPnl !== null && <small className={extraPnl >= 0 ? "quote-diff-up" : "quote-diff-down"}>
      {extraPnl >= 0 ? "+" : ""}{formatUsd(extraPnl)} vs close
    </small>}
  </span>;
}

function ScenarioResult({ result, customEmpty = false, endingLabel, pendingLabel }: {
  result: OvernightExitSummary;
  customEmpty?: boolean;
  endingLabel?: string;
  pendingLabel?: string;
}) {
  if (customEmpty) return <span className="overnight-exit-pending">Choose tickers</span>;
  const pending = pendingLabel ?? (result.selected > 0 ? `Pending · ${result.observed}/${result.selected} opens` : "Awaiting close");
  return <MoneyResult pnl={result.pnl} endingValue={result.endingValue} extraPnl={result.extraPnl}
    pending={pending} endingLabel={endingLabel} />;
}

function actualStatus(actual: OvernightActualRow | undefined): string {
  if (!actual) return "Not checked";
  if (actual.status === "observed") return "Verified";
  if (actual.status === "awaiting_next_open") return "Awaiting open";
  if (actual.status === "corporate_action") return "Corporate action";
  return actual.reason ? `Unavailable · ${actual.reason}` : "Unavailable";
}

export default function OvernightHoldComparison({ day, kind, tradeSizes, actuals, loadActuals, portion, selectedRule, custom }: {
  day: SimulatedDay;
  kind: PickKind;
  tradeSizes: TradeSizes;
  actuals?: ReadonlyMap<string, OvernightActualRow>;
  loadActuals: LoadOvernightActuals;
  portion: number;
  selectedRule: OvernightExitMode;
  custom: Record<StrategyKind, ReadonlySet<string>>;
}) {
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const strategies = strategiesFor(day, kind, tradeSizes);
  const rows = strategies.flatMap((strategy) => strategy.rows);
  const tickers = [...new Set(rows.map((row) => row.ticker))];
  const observed = actuals ?? EMPTY_ACTUALS;
  const fraction = portion / 100;
  const resultFor = (strategy: StrategyRows, rule: OvernightExitRule) => summarizeOvernightExit(
    strategy.rows, observed, strategy.dollarsPerTrade, strategy.money, fraction, rule, custom[strategy.kind],
  );

  async function load(force: boolean) {
    if (tickers.length === 0) return;
    setBusy(true);
    setError(null);
    try {
      await loadActuals(day.as_of, tickers, force);
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "Could not check next opens");
    } finally {
      setBusy(false);
    }
  }

  return <details className="overnight-hold-panel" open={selectedRule !== "close"} onToggle={(event) => {
    if (event.currentTarget.open && !busy && tickers.some((ticker) => !actuals?.has(ticker)
      || actuals.get(ticker)?.status === "awaiting_next_open")) void load(false);
  }}>
    <summary><span>Next-open replay</span><small>Actual prices</small></summary>
    <div className="overnight-exit-toolbar">
      <span className="muted">{portion}% held · other shares sell at close</span>
      <button type="button" className="icon-btn" disabled={busy || !tickers.length} onClick={() => void load(true)}>
        {busy ? "Checking…" : actuals ? "Refresh opens" : "Check opens"}
      </button>
    </div>
    {error && <p className="error" role="alert">{error}</p>}
    <div className="overnight-exit-table-wrap"><table className="overnight-exit-table" aria-label="Daily next-open exit comparison">
      <thead><tr><th>Exit rule</th>{strategies.map((strategy) => <th key={strategy.kind}>{strategy.kind === "rank" ? "Rank" : "Fit"} P&amp;L / end</th>)}</tr></thead>
      <tbody>
        <tr className={selectedRule === "close" ? "is-selected" : undefined}><th scope="row">Sell at close <small>Current result</small></th>{strategies.map((strategy) => <td key={strategy.kind}>
          <MoneyResult pnl={strategy.money.pending ? null : strategy.money.pnl}
            endingValue={strategy.money.pending ? null : strategy.money.endingValue} pending="Awaiting close" />
        </td>)}</tr>
        {(["all", "down", "custom"] as const).map((rule) => <tr key={rule} className={selectedRule === rule ? "is-selected" : undefined}><th scope="row">
          {rule === "all" ? "Hold all included" : rule === "down" ? "Hold down today" : "My selection"}
          <small>{rule === "down" ? "Today’s losers" : rule === "custom" ? "Pick in tables below" : "Same shares"}</small>
        </th>{strategies.map((strategy) => {
          const result = resultFor(strategy, rule);
          return <td key={strategy.kind}><ScenarioResult result={result} customEmpty={rule === "custom" && result.selected === 0} /></td>;
        })}</tr>)}
      </tbody>
    </table></div>
    {selectedRule === "custom" && <p className="overnight-exit-instruction">Choose <strong>Next open</strong> beside each stock in the Rank and Fit tables.</p>}
    <details className="overnight-exit-price-details"><summary>Per-ticker prices</summary>
      <div className="overnight-exit-table-wrap"><table className="overnight-exit-table">
        <thead><tr><th>Ticker</th><th>Today</th><th>Close</th><th>Next open</th><th>Status</th></tr></thead>
        <tbody>{strategies.flatMap((strategy) => strategy.rows.map((row) => {
          const actual = observed.get(row.ticker);
          const included = row.hypothetical.status === "scored" || row.hypothetical.status === "held";
          const outcome = simulateOvernightHold(row, actual, strategy.dollarsPerTrade, fraction);
          const status = outcome.status === "price_mismatch" ? "Close sources disagree"
            : outcome.status === "awaiting_close" ? "Awaiting close" : actualStatus(actual);
          return <tr key={`${strategy.kind}-${row.rank}-${row.ticker}`}>
            <th scope="row">{row.ticker} <small>{strategy.kind === "rank" ? "Rank" : "Fit"}</small></th>
            <td>{row.session_return === null ? "—" : `${(row.session_return * 100).toFixed(2)}%`}</td>
            <td>{row.close_price === null ? "—" : formatUsd(row.close_price)}</td>
            <td>{actual?.status === "observed" && actual.next_open !== null ? formatUsd(actual.next_open) : "—"}</td>
            <td>{included ? status : "Excluded from totals"}</td>
          </tr>;
        }))}</tbody>
      </table></div>
    </details>
    <details className="overnight-exit-method"><summary>Method</summary>
      <p>Hindsight using verified raw close and next open. Fractional shares; fees and slippage excluded. The open is a price proxy, not a guaranteed fill. Missing or action-affected opens keep a basket pending. The original close totals never change.</p>
    </details>
  </details>;
}

export function OvernightHistoryComparison({ days, kind, tradeSizes, actualsByDay, loadActuals, portion, selectedRule }: {
  days: SimulatedDay[];
  kind: PickKind;
  tradeSizes: TradeSizes;
  actualsByDay: ActualsByDay;
  loadActuals: LoadOvernightActuals;
  portion: number;
  selectedRule: OvernightExitRule;
}) {
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const enabled: StrategyKind[] = kind === "both" ? ["rank", "fit"] : [kind];
  const summaries = enabled.map((strategy) => ({
    strategy,
    result: summarizeOvernightHistory(days, actualsByDay, strategy, tradeSizes[strategy], portion / 100),
  }));

  async function loadHistory() {
    const byDate = new Map<string, Set<string>>();
    for (const day of days) {
      if (day.scan_id || !enabled.some((strategy) => day[`${strategy}_money`].invested > 0)) continue;
      const names = byDate.get(day.as_of) ?? new Set<string>();
      enabled.forEach((strategy) => selectOvernightExitRows(day[strategy], "all")
        .forEach((row) => names.add(row.ticker)));
      byDate.set(day.as_of, names);
    }
    const jobs = [...byDate].map(([asOf, names]) => ({ asOf, tickers: [...names] }));
    if (!jobs.length) return;
    setBusy(true);
    setError(null);
    let cursor = 0;
    let failures = 0;
    async function worker() {
      while (cursor < jobs.length) {
        const job = jobs[cursor++];
        try { await loadActuals(job.asOf, job.tickers, true); }
        catch { failures += 1; }
      }
    }
    await Promise.all(Array.from({ length: Math.min(4, jobs.length) }, worker));
    if (failures) setError(`${failures} ${failures === 1 ? "day" : "days"} could not be checked. Retry to refresh.`);
    setBusy(false);
  }

  return <section className="overnight-history-panel" aria-label="Historical next-open replay">
    <div className="overnight-history-head"><div><h3>Next-open replay across history</h3>
      <p>Hypothetical exits at actual opens</p></div><span className="overnight-history-badge">HINDSIGHT</span></div>
    <div className="overnight-exit-toolbar">
      <span className="muted">{portion}% held · remaining shares sell at close</span>
      <button type="button" className="icon-btn" disabled={busy} onClick={() => void loadHistory()}>
        {busy ? "Checking history…" : actualsByDay.size ? "Refresh actual opens" : "Compare actual opens"}
      </button>
      <span className="muted">Same complete days for both rules</span>
    </div>
    {error && <p className="error" role="alert">{error}</p>}
    <div className="overnight-history-grid" aria-live="polite">{summaries.map(({ strategy, result }) => {
      const pendingLabel = result.eligibleDays === 0 ? "No eligible days"
        : actualsByDay.size === 0 ? "Check actual opens" : "No complete days yet";
      return <div className="overnight-history-strategy" key={strategy}>
      <div className="overnight-exit-picker-head"><strong>{strategy === "rank" ? "Rank" : "Fit"}</strong>
        <span>{result.matchedDays}/{result.eligibleDays} complete days</span></div>
      <div className="overnight-history-line"><span>Sell at close</span>
        <MoneyResult pnl={result.baselinePnl} endingValue={result.baselineEndingValue}
          endingLabel="Sum of exits" pending={pendingLabel} /></div>
      <div className={`overnight-history-line${selectedRule === "all" ? " is-selected" : ""}`}><span>Hold all</span>
        <ScenarioResult result={result.all} endingLabel="Sum of exits" pendingLabel={pendingLabel} /></div>
      <div className={`overnight-history-line${selectedRule === "down" ? " is-selected" : ""}`}><span>Hold down today</span>
        <ScenarioResult result={result.down} endingLabel="Sum of exits" pendingLabel={pendingLabel} /></div>
    </div>})}</div>
    <p className="overnight-history-note">Matched, complete live days only. Exit values sum the same per-trade notional across days; they are not compounded. Custom picks stay exploratory.</p>
  </section>;
}
