import { useCallback, useEffect, useRef, useState } from "react";
import {
  fetchBenchmarkReturns,
  fetchBlacklistedTickers,
  fetchFirstTouch,
  fetchHoldPath,
  fetchOvernightActuals,
  fetchPaperBook,
  fetchTrainingRuns,
  rebuildPaperBook,
  replayPaperBook,
  type PaperListStats,
  type OvernightActualRow,
  type TrainingRunRecord,
} from "../api";
import { useFetchData } from "../useFetchData";
import { ColumnTitle, DataTable, NewsCell, ScoreCell, TickerCell, UsdCell } from "./DataTable";
import { Diff } from "./Diff";
import { formatUsd } from "../format";
import {
  benchmarkDates, compareStrategyToBenchmark, DEFAULT_TRADE_SIZE, loadStrategyBenchmarks,
  LOOKBACK_OPTIONS, TRADE_SIZE_OPTIONS, overnightSelectionKey, overnightTradeReturn, returnOnAverageDailyCapital,
  selectOvernightExitRows, simulateBook,
  simulateOvernightHold, simulationDay, summarizeOvernightExit, summarizeSelectedOvernightHistory,
  type BenchmarkLoad,
  type LookbackDays, type MoneySummary, type OvernightExitMode, type OvernightExitRule,
  type PickKind, type PickOutcome, type SelectedOvernightHistorySummary, type SimulatedDay, type SimulatedPick,
  type StrategyBenchmark, type StrategyKind, type TradeSizes,
} from "../whatIf";
import { StatStrip, type StatItem } from "./StatStrip";
import {
  addHoldSummaries, holdPickResult, resolveAfterClose, summarizeHold, tieCases, tieKey,
  type AfterCloseMode, type FirstTouch, type HoldBar, type HoldPickResult, type HoldSummary,
} from "../whatIfHold";

const HOLD_DROP_STOPS = [1, 2, 3, 4, 5, 8, 10, 15, 20];
import TogglePill from "./TogglePill";

const EMPTY_OVERNIGHT_ACTUALS = new Map<string, OvernightActualRow>();
const EMPTY_CUSTOM = new Set<string>();
type LoadOvernightActuals = (asOf: string, tickers: string[], force?: boolean) => Promise<void>;

function formatRunLabel(startedAt: string, holdoutAccuracy: number | null): string {
  const when = new Date(startedAt).toLocaleString(undefined, {
    month: "short",
    day: "numeric",
    hour: "numeric",
    minute: "2-digit",
  });
  return holdoutAccuracy === null ? when : `${when} · ${(holdoutAccuracy * 100).toFixed(1)}% holdout`;
}

function formatDay(day: string): string {
  return new Date(`${day}T12:00:00`).toLocaleDateString("en-US", {
    weekday: "long",
    month: "short",
    day: "numeric",
  });
}

function formatMonthDay(day: string): string {
  return new Date(`${day}T12:00:00`).toLocaleDateString("en-US", {
    month: "short",
    day: "numeric",
  });
}

function Pct({ value }: { value: number | null | undefined }) {
  if (value === null || value === undefined) return <span className="muted">—</span>;
  const isUp = value >= 0;
  return (
    <span className={isUp ? "quote-diff-up" : "quote-diff-down"}>
      {isUp ? "▲" : "▼"} {(Math.abs(value) * 100).toFixed(2)}%
    </span>
  );
}

function Dollars({ value }: { value: number | null }) {
  return value === null ? <span className="muted">—</span> : <Diff value={value} pct={null} />;
}

function PickMoney({ outcome, ending = false }: { outcome: PickOutcome; ending?: boolean }) {
  if (ending) {
    if (outcome.endingValue === null) return <span className="muted">Awaiting prices</span>;
    return <span title={outcome.status === "skipped" ? "Hypothetical value; excluded from strategy totals" : undefined}>
      <UsdCell value={outcome.endingValue} />
    </span>;
  }
  if (outcome.status === "skipped") return <span className="muted">Skipped</span>;
  if (outcome.status === "pending") return <span className="muted">Awaiting prices</span>;
  return <Dollars value={outcome.pnl} />;
}

function heldUntilOpen(row: SimulatedPick, mode: OvernightExitMode, custom: ReadonlySet<string>): boolean {
  return mode !== "close" && selectOvernightExitRows([row], mode, custom).length > 0;
}

function exitGapLabel(actual: OvernightActualRow | undefined): string {
  if (!actual) return "Open not loaded";
  if (actual.status === "awaiting_next_open") return "Awaiting open";
  if (actual.status === "corporate_action") return actual.reason ? `No open · ${actual.reason}` : "No open · corporate action";
  if (actual.status === "unavailable") return "Open unavailable";
  if (actual.status === "observed") return "Close mismatch";
  return "Price mismatch";
}

function ScenarioPickMoney({ row, actual, dollarsPerTrade, portion, mode, custom, ending = false }: {
  row: SimulatedPick;
  actual?: OvernightActualRow;
  dollarsPerTrade: number;
  portion: number;
  mode: OvernightExitMode;
  custom: ReadonlySet<string>;
  ending?: boolean;
}) {
  if (!heldUntilOpen(row, mode, custom)) return <PickMoney outcome={row.hypothetical} ending={ending} />;
  const outcome = simulateOvernightHold(row, actual, dollarsPerTrade, portion / 100);
  if (outcome.status !== "observed" || row.hypothetical.pnl === null) {
    return <span className="muted">{exitGapLabel(actual)}</span>;
  }
  if (ending) return <UsdCell value={outcome.endingValue} />;
  const pnl = Math.round((row.hypothetical.pnl + outcome.incrementalPnl!) * 100) / 100;
  return <Dollars value={pnl} />;
}

function moneyItems(money: MoneySummary, closeBaseline = false): StatItem[] {
  return [
    { key: "capital", label: "Completed capital", value: formatUsd(money.invested) },
    { key: "pnl", label: closeBaseline ? "Close P&L" : "P&L", value: <Dollars value={money.pnl} /> },
    { key: "ending", label: closeBaseline ? "Close value" : "End value", value: <UsdCell value={money.endingValue} /> },
    ...(money.pending ? [{ key: "pending", value: `${money.pending} awaiting prices` }] : []),
  ];
}

function statsItems(stats: PaperListStats | undefined): StatItem[] {
  if (!stats || stats.n_scored === 0) return [];
  const hit = stats.hit_rate !== null ? `${(stats.hit_rate * 100).toFixed(0)}% hit` : null;
  return [
    { key: "scored", label: "Scored", value: stats.n_scored },
    { key: "wl", label: "W/L", value: `${stats.wins}W/${stats.losses}L` },
    ...(hit ? [{ key: "hit", label: "Hit", value: hit }] : []),
    { key: "avg", label: "Avg/pick", value: <Pct value={stats.avg} /> },
    ...(stats.n_avoid
      ? [{ key: "skipped", label: "Skipped", value: `${stats.n_avoid} news skips` }]
      : []),
  ];
}

/** Score the complete trade from its entry to the selected exit.
 * A missing open stays out of the win/loss count.
 */
function exitReturn(
  row: SimulatedPick,
  mode: OvernightExitMode,
  actual: OvernightActualRow | undefined,
  custom: ReadonlySet<string>,
  dollarsPerTrade: number,
  portion: number,
): number | null {
  if (mode === "close" || !heldUntilOpen(row, mode, custom)) return row.session_return;
  return overnightTradeReturn(row, actual, dollarsPerTrade, portion / 100);
}

function recordFromReturns(n: number, returns: readonly number[], skipped = 0): PaperListStats {
  const wins = returns.filter((value) => value > 0).length;
  const losses = returns.filter((value) => value < 0).length;
  return {
    n,
    n_scored: returns.length,
    wins,
    losses,
    flats: returns.length - wins - losses,
    hit_rate: returns.length ? wins / returns.length : null,
    avg: returns.length ? returns.reduce((sum, value) => sum + value, 0) / returns.length : null,
    avg_ex_news: null,
    n_avoid: skipped,
  };
}

/** Every scored pick under the selected exit. A held name uses its full trade return.
 * A name the rule sells at the close uses the close. A missing open stays out.
 */
function outcomeStats(
  rows: readonly SimulatedPick[],
  mode: OvernightExitMode,
  actuals: ReadonlyMap<string, OvernightActualRow> | undefined,
  custom: ReadonlySet<string>,
  dollarsPerTrade: number,
  portion: number,
): PaperListStats | undefined {
  const scored = rows.filter((row) => row.hypothetical.status === "scored" || row.hypothetical.status === "held");
  if (scored.length === 0) return undefined;
  const returns = scored.flatMap((row) => {
    const value = exitReturn(row, mode, actuals?.get(row.ticker), custom, dollarsPerTrade, portion);
    return value === null ? [] : [value];
  });
  return recordFromReturns(scored.length, returns);
}

/** Count entry-to-exit outcomes for the trades held to the next open.
 * A name whose next open is already knowable but unusable stays out of the count.
 */
function exitStats(
  rows: readonly SimulatedPick[],
  mode: OvernightExitMode,
  actuals: ReadonlyMap<string, OvernightActualRow> | undefined,
  custom: ReadonlySet<string>,
  dollarsPerTrade: number,
  portion: number,
): PaperListStats | undefined {
  if (mode === "close") return undefined;
  const held = rows.filter((row) => heldUntilOpen(row, mode, custom)
    && (row.hypothetical.status === "scored" || row.hypothetical.status === "held"));
  if (held.length === 0) return undefined;
  const returns = held.flatMap((row) => {
    const value = exitReturn(row, mode, actuals?.get(row.ticker), custom, dollarsPerTrade, portion);
    return value === null ? [] : [value];
  });
  return recordFromReturns(held.length, returns);
}

function SignedDollars({ value, emptyLabel = "—" }: { value: number | null; emptyLabel?: string }) {
  if (value === null) return <span className="muted">{emptyLabel}</span>;
  return <span className={value === 0 ? "muted" : value > 0 ? "quote-diff-up" : "quote-diff-down"}>
    {value > 0 ? "+" : ""}{formatUsd(value)}
  </span>;
}

function SignedPercent({ value, emptyLabel = "—" }: { value: number | null; emptyLabel?: string }) {
  if (value === null) return <span className="muted">{emptyLabel}</span>;
  return <span className={value === 0 ? "muted" : value > 0 ? "quote-diff-up" : "quote-diff-down"}>
    {value > 0 ? "+" : ""}{(value * 100).toFixed(2)}%
  </span>;
}

function ComparisonRow({ label, pnl, gain, detail, hint = "", primary = false, emptyPnl, emptyGain, gainNote }: {
  label: string;
  pnl: number | null;
  gain: number | null;
  detail: string;
  hint?: string;
  primary?: boolean;
  emptyPnl?: string;
  emptyGain?: string;
  gainNote?: string;
}) {
  return (
    <tr className={`row-hover${primary ? " is-strategy" : ""}`} title={hint}>
      <th scope="row"><span>{label}</span><small>{detail}</small></th>
      <td><SignedDollars value={pnl} emptyLabel={emptyPnl} /></td>
      <td><SignedPercent value={gain} emptyLabel={emptyGain} />{gainNote && <small className="comparison-gain-note">{gainNote}</small>}</td>
    </tr>
  );
}

function StrategySummary({ title, topK, days, stats, exitRecord, outcomeRecord, money, benchmark, benchmarkLoading, benchmarkError, scenario, exitMode, holdSummary, holdLabel }: {
  title: string;
  topK: number | undefined;
  days: number;
  stats: PaperListStats | undefined;
  /** Names the rule held overnight. */
  exitRecord?: PaperListStats;
  /** Every pick under the selected exit, including names sold at the close. */
  outcomeRecord?: PaperListStats;
  money: MoneySummary;
  benchmark: StrategyBenchmark;
  benchmarkLoading: boolean;
  benchmarkError: string | null;
  scenario: SelectedOvernightHistorySummary | null;
  exitMode: OvernightExitMode;
  holdSummary?: HoldSummary | null;
  holdLabel?: string;
}) {
  const intradayDetail = benchmark.activeDays
    ? `${benchmark.matchedDays}/${benchmark.activeDays} days priced`
    : "no completed days";
  const holdDetail = benchmark.holdFrom && benchmark.holdThrough
    ? `${formatMonthDay(benchmark.holdFrom)}–${formatMonthDay(benchmark.holdThrough)}`
    : "unavailable";
  const exitLabel = exitMode === "down" ? "Down-day next open"
    : exitMode === "custom" ? "Selected next open" : "Next open";
  const averageDailyCapital = benchmark.holdCapital;
  const strategyPnl = holdSummary ? holdSummary.pnl : scenario ? scenario.pnl : money.pnl;
  const unpricedOpenDays = scenario ? scenario.eligibleDays - scenario.matchedDays : 0;
  const missingSpyDays = benchmark.activeDays - benchmark.matchedDays;
  const strategyGain = returnOnAverageDailyCapital(strategyPnl, averageDailyCapital);
  const intradayGain = benchmark.matchedDays === benchmark.activeDays
    ? returnOnAverageDailyCapital(benchmark.intraday.pnl, averageDailyCapital) : null;
  const holdGain = returnOnAverageDailyCapital(benchmark.hold.pnl, averageDailyCapital);
  const record = outcomeRecord ?? exitRecord ?? stats;
  const heldOvernight = exitRecord && outcomeRecord && exitRecord.n !== outcomeRecord.n;
  const scenarioDetail = holdSummary
    ? `${holdLabel} · ${holdSummary.n} priced${holdSummary.pending ? ` · ${holdSummary.pending} awaiting` : ""}`
    : scenario
    ? `${exitLabel} · ${scenario.matchedDays}/${scenario.eligibleDays} days priced`
    : "Open → close";
  return (
    <div className="view-card slice-card" aria-label={`${title} simulation total`}>
      <div className="slice-card-header">
        <div className="slice-card-heading"><strong>{title}</strong><span>{topK === undefined ? "All picks" : `Top ${topK}`} · {days} days</span></div>
        <span className="slice-card-capital" title="Average capital invested on each active day; the same stake is used for the period-gain comparison.">
          {averageDailyCapital === null ? "No completed capital" : `${formatUsd(averageDailyCapital)} avg/day`}
        </span>
      </div>
      <table className="what-if-comparison" aria-label={`${title} picks and S&P comparison`}>
        <thead><tr><th scope="col">Outcome</th><th scope="col">P&amp;L</th><th scope="col">Period gain</th></tr></thead>
        <tbody>
          <ComparisonRow
            primary label="Stock Pickin'" detail={scenarioDetail}
            pnl={strategyPnl} gain={strategyGain}
            emptyPnl={scenario ? "Awaiting opens" : "Awaiting close"}
            emptyGain={strategyGain === null && unpricedOpenDays ? `${unpricedOpenDays} ${unpricedOpenDays === 1 ? "day" : "days"} unpriced` : "—"}
            gainNote={strategyGain !== null && unpricedOpenDays ? `${unpricedOpenDays} ${unpricedOpenDays === 1 ? "day" : "days"} unpriced · partial` : undefined}
          />
          <ComparisonRow
            label="S&P intraday" detail={intradayDetail}
            pnl={benchmark.intraday.pnl} gain={intradayGain}
            emptyGain={missingSpyDays ? `${missingSpyDays} SPY ${missingSpyDays === 1 ? "day" : "days"} missing` : "—"}
          />
          <ComparisonRow
            label="S&P buy & hold" detail={holdDetail}
            pnl={benchmark.hold.pnl} gain={holdGain}
          />
        </tbody>
      </table>
      <div className="slice-card-metrics">
        <span>{record?.n ?? 0} picks</span>
        <span>{record?.n_scored ?? 0} scored</span>
        {record && record.n_scored > 0 && <>
          <span title="Wins and losses compare the selected exit proceeds with the original purchase cost.">{record.wins} wins / {record.losses} losses</span>
          <span>{record.hit_rate === null ? "—" : `${(record.hit_rate * 100).toFixed(0)}%`} win rate</span>
          <span className="slice-card-avg" title="Arithmetic mean return per scored pick; not the period gain.">Avg/pick <SignedPercent value={record.avg} /></span>
        </>}
        {heldOvernight && exitRecord && exitRecord.n_scored > 0 && (
          <span title="Entry-to-exit results for trades the rule held overnight.">{exitRecord.n_scored} held overnight, {exitRecord.wins} wins / {exitRecord.losses} losses</span>
        )}
        {stats && (stats.n_avoid ?? 0) > 0 && <span>{stats.n_avoid} news skips</span>}
        {money.pending > 0 && <span>{money.pending} awaiting prices</span>}
      </div>
      <div className="what-if-benchmark-help">
        {benchmarkLoading && <span className="muted what-if-benchmark-loading">Loading S&P…</span>}
        {benchmarkError && <span className="error what-if-benchmark-loading" role="status">S&P unavailable: benchmark request failed.</span>}
        {!benchmarkLoading && !benchmarkError && benchmark.activeDays > 0 && benchmark.matchedDays === 0 && (
          <span className="muted what-if-benchmark-loading" role="status">S&P data unavailable for these days.</span>
        )}
      </div>
    </div>
  );
}

const HOLD_REASON_LABELS: Record<HoldPickResult["reason"], string> = {
  target: "Gain",
  stop: "Stop",
  deadline: "Close",
};

function holdEligible(row: SimulatedPick): boolean {
  return row.hypothetical.status === "scored" || row.hypothetical.status === "held";
}

function HoldExitCell({ result, row }: { result: HoldPickResult | null | undefined; row: SimulatedPick }) {
  if (!holdEligible(row)) return <span className="muted">Excluded</span>;
  if (!result) return <span className="muted">Awaiting</span>;
  return <span title={result.date}>
    {HOLD_REASON_LABELS[result.reason]} <UsdCell value={result.price} /> {formatMonthDay(result.date)}
  </span>;
}

function HoldMove({ result, row }: { result: HoldPickResult | null | undefined; row: SimulatedPick }) {
  if (!holdEligible(row) || !result || row.close_price === null) return <span className="muted">—</span>;
  return <Pct value={result.price / row.close_price - 1} />;
}

function HoldMoney({ result, row, ending = false }: { result: HoldPickResult | null | undefined; row: SimulatedPick; ending?: boolean }) {
  if (!holdEligible(row)) return <PickMoney outcome={row.hypothetical} ending={ending} />;
  if (!result) return <span className="muted">Awaiting</span>;
  return ending ? <UsdCell value={result.endingValue} /> : <Diff value={result.pnl} pct={null} />;
}

function ListTable({
  title,
  rows,
  stats,
  isRank,
  money,
  dollarsPerTrade,
  actuals,
  overnightPortion,
  exitMode,
  custom,
  onToggleHold,
  holdRows,
}: {
  title: string;
  rows: SimulatedPick[];
  stats: PaperListStats | undefined;
  isRank: boolean;
  money: MoneySummary;
  dollarsPerTrade: number;
  actuals?: ReadonlyMap<string, OvernightActualRow>;
  overnightPortion: number;
  exitMode: OvernightExitMode;
  custom: ReadonlySet<string>;
  onToggleHold: (ticker: string) => void;
  /** Wait-for-a-gain result per ticker; when set, Exit and the money columns show it. */
  holdRows?: ReadonlyMap<string, HoldPickResult | null> | null;
}) {
  if (rows.length === 0) {
    return <p className="muted">{title}: no list that morning</p>;
  }
  return (
    <div>
      <p className="view-meta" style={{ marginBottom: "var(--space-2)" }}>
        <StatStrip
          items={[
            { key: "title", title: true, align: "start", value: <strong>{title}</strong> },
            { key: "names", align: "start", value: `${rows.length} names` },
            ...statsItems(exitStats(rows, exitMode, actuals, custom, dollarsPerTrade, overnightPortion) ?? stats),
          ]}
        />
      </p>
      <p className="view-meta"><StatStrip items={moneyItems(money, exitMode !== "close")} /></p>
      <div style={{ overflowX: "auto" }}>
        <DataTable
          rows={rows}
          rowKey={(row) => `${title}-${row.rank}-${row.ticker}`}
          columns={[
            { key: "rank", header: "#", cell: (row) => row.rank },
            { key: "ticker", header: "Ticker", cell: (row) => <TickerCell ticker={row.ticker} /> },
            {
              key: "score",
              header: isRank ? "Score" : "Predicted",
              numeric: true,
              cell: (row) => <ScoreCell value={row.predicted} isRank={isRank} />,
            },
            { key: "open", header: "Open", numeric: true, cell: (row) => <UsdCell value={row.open_price} /> },
            { key: "close", header: "Close", numeric: true, cell: (row) => <UsdCell value={row.close_price} /> },
            { key: "session", header: exitMode === "close" && !holdRows ? "Session" : "Exit move", numeric: true,
              cell: (row) => holdRows
                ? <HoldMove result={holdRows.get(row.ticker)} row={row} />
                : <Pct value={exitReturn(row, exitMode, actuals?.get(row.ticker), custom, dollarsPerTrade, overnightPortion)} /> },
            { key: "hold-exit", header: "Exit", when: !!holdRows, cell: (row) => <HoldExitCell result={holdRows?.get(row.ticker)} row={row} /> },
            { key: "exit", header: "Exit", when: exitMode !== "close" && !holdRows, cell: (row) => {
              const eligible = row.hypothetical.status === "scored" || row.hypothetical.status === "held";
              if (!eligible) return <span className="muted">Excluded</span>;
              if (exitMode === "custom") return <label className="what-if-exit-choice">
                <input type="checkbox" checked={custom.has(row.ticker)}
                  aria-label={`Hold ${row.ticker} until next open`}
                  onChange={() => onToggleHold(row.ticker)} />
                <span>Next open</span>
              </label>;
              return <span className={heldUntilOpen(row, exitMode, custom) ? "what-if-exit-open" : "muted"}>
                {heldUntilOpen(row, exitMode, custom) ? "Next open" : "Close"}
              </span>;
            } },
            { key: "pnl", header: <ColumnTitle label={exitMode === "close" && !holdRows ? "P&L" : "Exit P&L"}
              hint={`${formatUsd(dollarsPerTrade)} / trade`} />, numeric: true,
              cell: (row) => holdRows ? <HoldMoney result={holdRows.get(row.ticker)} row={row} /> : <ScenarioPickMoney row={row} actual={actuals?.get(row.ticker)}
                dollarsPerTrade={dollarsPerTrade} portion={overnightPortion} mode={exitMode} custom={custom} /> },
            { key: "ending", header: exitMode === "close" && !holdRows ? "End value" : "Exit value", numeric: true,
              cell: (row) => holdRows ? <HoldMoney result={holdRows.get(row.ticker)} row={row} ending /> : <ScenarioPickMoney row={row} actual={actuals?.get(row.ticker)}
                dollarsPerTrade={dollarsPerTrade} portion={overnightPortion} mode={exitMode} custom={custom} ending /> },
            { key: "news", header: "Saved news check", cell: (row) =>
              <NewsCell flag={row.news_flag} blocks={row.news_blocks} checked={row.news_checked}
                check={row.news_check} skipLabel="saved skip" compactHint /> },
          ]}
        />
      </div>
    </div>
  );
}

function shortfallNote(asked: number | undefined, got: number, label: "Rank" | "Fit"): string | null {
  if (asked === undefined || got === 0 || got >= asked) return null;
  return `${label} had ${got} that morning`;
}

function DayCard({
  day,
  kind,
  rankAsked,
  fitAsked,
  modelLabel,
  tradeSizes,
  hold,
  actuals,
  overnightPortion,
  exitMode,
  custom,
  onToggleHold,
}: {
  day: SimulatedDay;
  kind: PickKind;
  rankAsked: number | undefined;
  fitAsked: number | undefined;
  modelLabel?: string;
  tradeSizes: TradeSizes;
  hold?: {
    gainPct: number; dropPct: number; days: number;
    paths: ReadonlyMap<string, readonly HoldBar[]>; ties: ReadonlyMap<string, FirstTouch>;
  } | null;
  actuals?: ReadonlyMap<string, OvernightActualRow>;
  overnightPortion: number;
  exitMode: OvernightExitMode;
  custom: Record<StrategyKind, ReadonlySet<string>>;
  onToggleHold: (strategy: StrategyKind, ticker: string) => void;
}) {
  const [isOpen, setIsOpen] = useState(false);
  function exitPnl(strategy: StrategyKind): number | null {
    const money = day[`${strategy}_money`];
    if (exitMode === "close") return money.pnl;
    const actualsForDay = actuals ?? EMPTY_OVERNIGHT_ACTUALS;
    const held = selectOvernightExitRows(day[strategy], exitMode, custom[strategy]);
    // The row arrow matches the card: up means the next open beat the close.
    if (held.length > 0 && held.every((row) => {
      const outcome = simulateOvernightHold(row, actualsForDay.get(row.ticker), tradeSizes[strategy], overnightPortion / 100);
      return outcome.status === "observed" && outcome.incrementalPnl !== null;
    })) {
      return held.reduce((sum, row) => sum + simulateOvernightHold(
        row, actualsForDay.get(row.ticker), tradeSizes[strategy], overnightPortion / 100,
      ).incrementalPnl!, 0);
    }
    return summarizeOvernightExit(day[strategy], actualsForDay,
      tradeSizes[strategy], money, overnightPortion / 100, exitMode, custom[strategy]).pnl;
  }
  const rankExitPnl = kind !== "fit" ? exitPnl("rank") : null;
  const fitExitPnl = kind !== "rank" ? exitPnl("fit") : null;
  function holdResult(strategy: StrategyKind) {
    if (!hold) return null;
    return summarizeHold(day[strategy], hold.paths, hold.gainPct, hold.dropPct, hold.days, tradeSizes[strategy], hold.ties);
  }
  function holdRowsFor(strategy: StrategyKind): ReadonlyMap<string, HoldPickResult | null> | null {
    if (!hold) return null;
    return new Map(day[strategy].map((row) => [row.ticker, holdPickResult(
      row, hold.paths.get(row.ticker) ?? [], hold.gainPct, hold.dropPct, hold.days, tradeSizes[strategy], hold.ties,
    )]));
  }
  const rankHold = kind !== "fit" ? holdResult("rank") : null;
  const fitHold = kind !== "rank" ? holdResult("fit") : null;
  const rankNote = kind !== "fit" ? shortfallNote(rankAsked, day.rank.length, "Rank") : null;
  const fitNote = kind !== "rank" ? shortfallNote(fitAsked, day.fit.length, "Fit") : null;
  return (
    <details className="view-card" onToggle={(event) => setIsOpen(event.currentTarget.open)}>
      <summary>
        <StatStrip
          items={[
            {
              key: "title",
              title: true,
              align: "start",
              value: <strong style={{ color: "var(--accent)" }}>{formatDay(day.as_of)}</strong>,
            },
            { key: "kind", align: "start", value: day.scan_id ? "replay" : "live" },
            ...(modelLabel ? [{ key: "model", align: "start" as const, value: modelLabel }] : []),
            ...(kind !== "fit" && day.rank.length > 0
              ? [
                  {
                    key: "rank",
                    label: exitMode === "close" ? "Rank" : "Rank exit",
                    value: (
                      <>
                        {day.rank.length} {exitMode === "close" && <Pct value={day.rank_avg} />} · {rankExitPnl === null && exitMode !== "close"
                          ? <span className="muted">Awaiting open</span> : <Dollars value={rankExitPnl} />}
                      </>
                    ),
                  },
                ]
              : []),
            ...(kind !== "rank" && day.fit.length > 0
              ? [
                  {
                    key: "fit",
                    label: exitMode === "close" ? "Fit" : "Fit exit",
                    value: (
                      <>
                        {day.fit.length} {exitMode === "close" && <Pct value={day.fit_avg} />} · {fitExitPnl === null && exitMode !== "close"
                          ? <span className="muted">Awaiting open</span> : <Dollars value={fitExitPnl} />}
                      </>
                    ),
                  },
                ]
              : []),
            ...(rankHold ? [{
              key: "rank-hold",
              label: `Rank ${hold?.gainPct}% / ${hold?.days}d`,
              value: rankHold.n === 0
                ? <span className="muted">{rankHold.pending} awaiting</span>
                : <><Pct value={rankHold.avg} /> · <Dollars value={rankHold.pnl} /></>,
            }] : []),
            ...(fitHold ? [{
              key: "fit-hold",
              label: `Fit ${hold?.gainPct}% / ${hold?.days}d`,
              value: fitHold.n === 0
                ? <span className="muted">{fitHold.pending} awaiting</span>
                : <><Pct value={fitHold.avg} /> · <Dollars value={fitHold.pnl} /></>,
            }] : []),
            ...(rankNote ? [{ key: "rank-note", align: "start" as const, value: rankNote }] : []),
            ...(fitNote ? [{ key: "fit-note", align: "start" as const, value: fitNote }] : []),
          ]}
        />
      </summary>
      {isOpen && <div className="what-if-day-lists">
        {kind !== "rank" && (
          <ListTable title="Fit" rows={day.fit} stats={day.fit_stats} isRank={false}
            money={day.fit_money} dollarsPerTrade={tradeSizes.fit} actuals={actuals}
            overnightPortion={overnightPortion} exitMode={exitMode} custom={custom.fit}
            onToggleHold={(ticker) => onToggleHold("fit", ticker)} holdRows={holdRowsFor("fit")} />
        )}
        {kind !== "fit" && (
          <ListTable title="Rank" rows={day.rank} stats={day.rank_stats} isRank={true}
            money={day.rank_money} dollarsPerTrade={tradeSizes.rank} actuals={actuals}
            overnightPortion={overnightPortion} exitMode={exitMode} custom={custom.rank}
            onToggleHold={(ticker) => onToggleHold("rank", ticker)} holdRows={holdRowsFor("rank")} />
        )}
      </div>}
    </details>
  );
}

function parseTop(raw: string): number | undefined {
  const trimmed = raw.trim();
  if (trimmed === "") return undefined;
  const n = Number(trimmed);
  if (!Number.isFinite(n) || n <= 0) return undefined;
  return Math.floor(n);
}

function StrategyControl({
  label,
  enabled,
  onEnabled,
  value,
  onValue,
  dollarsPerTrade,
  onTradeSize,
}: {
  label: string;
  enabled: boolean;
  onEnabled: (on: boolean) => void;
  value: string;
  onValue: (value: string) => void;
  dollarsPerTrade: number;
  onTradeSize: (amount: number) => void;
}) {
  return (
    <TogglePill
      on={enabled}
      onToggle={() => onEnabled(!enabled)}
      extra={
        <>
          <input
            className="form-input slice-input"
            type="number"
            min={1}
            step={1}
            inputMode="numeric"
            placeholder="all"
            disabled={!enabled}
            value={value}
            onChange={(event) => onValue(event.target.value)}
            aria-label={`${label} top`}
            title="Number of top picks"
          />
          <select
            className="form-select what-if-trade-size"
            value={dollarsPerTrade}
            disabled={!enabled}
            onChange={(event) => onTradeSize(Number(event.target.value))}
            aria-label={`${label} dollars per trade`}
            title={`${label}: amount per trade`}
          >
            {TRADE_SIZE_OPTIONS.map((amount) => (
              <option key={amount} value={amount}>${amount / 1_000}K / trade</option>
            ))}
          </select>
        </>
      }
    >
      {label}
    </TogglePill>
  );
}

export default function WhatIf() {
  const [rankOn, setRankOn] = useState(true);
  const [fitOn, setFitOn] = useState(true);
  const [rankText, setRankText] = useState("5");
  const [fitText, setFitText] = useState("5");
  const [tradeSizes, setTradeSizes] = useState<TradeSizes>({ rank: DEFAULT_TRADE_SIZE, fit: DEFAULT_TRADE_SIZE });
  const [lookbackDays, setLookbackDays] = useState<LookbackDays>(null);
  const [applyNewsSkips, setApplyNewsSkips] = useState(false);
  const [excludeBlacklisted, setExcludeBlacklisted] = useState(false);
  const [afterClose, setAfterClose] = useState<AfterCloseMode>("close");
  const [holdGainPct, setHoldGainPct] = useState(1);
  const [holdDropPct, setHoldDropPct] = useState(10);
  const [holdDays, setHoldDays] = useState(1);
  const [holdPaths, setHoldPaths] = useState<ReadonlyMap<string, readonly HoldBar[]>>(() => new Map());
  const [tieAnswers, setTieAnswers] = useState<ReadonlyMap<string, FirstTouch>>(() => new Map());
  const [overnightRule, setOvernightRule] = useState<OvernightExitRule>("all");
  const [overnightPortion, setOvernightPortion] = useState(100);
  const [openRefresh, setOpenRefresh] = useState(0);
  const [historyBusy, setHistoryBusy] = useState(false);
  const [historyError, setHistoryError] = useState<string | null>(null);
  const [customSelections, setCustomSelections] = useState<ReadonlyMap<string, ReadonlySet<string>>>(() => new Map());
  const [refresh, setRefresh] = useState(0);
  const [busy, setBusy] = useState(false);
  const [rebuildError, setRebuildError] = useState<string | null>(null);
  const [replayDay, setReplayDay] = useState<string | null>(null);
  const [replayModel, setReplayModel] = useState("live");
  const benchmarkCache = useRef(new Map<string, Promise<BenchmarkLoad>>());
  const actualsCache = useRef(new Map<string, Map<string, OvernightActualRow>>());
  const [, setActualsRevision] = useState(0);
  const { data: raw, error } = useFetchData(() => fetchPaperBook("both"), {
    deps: [refresh], intervalMs: 60_000,
  });
  const selectedReplayDay = replayDay ?? raw?.days[0]?.as_of ?? "";
  const { data: blacklistedTickers, error: blacklistError } = useFetchData(fetchBlacklistedTickers, {
    deps: [refresh], intervalMs: 60_000,
  });
  const { data: trainingRuns } = useFetchData(fetchTrainingRuns, { deps: [] });
  const rankAsked = rankOn ? parseTop(rankText) : undefined;
  const fitAsked = fitOn ? parseTop(fitText) : undefined;
  const kind: PickKind = rankOn && fitOn ? "both" : rankOn ? "rank" : "fit";
  // +0% / 1d resolves to Next morning open, so both settings share one calculation.
  const afterCloseRule = resolveAfterClose({
    mode: afterClose, gainPct: holdGainPct, dropPct: holdDropPct, days: holdDays,
    rule: overnightRule, portion: overnightPortion,
  });
  const holdNextOpen = afterCloseRule.nextOpen !== null;
  const exitRule = afterCloseRule.nextOpen?.rule ?? overnightRule;
  const exitPortion = afterCloseRule.nextOpen?.portion ?? overnightPortion;
  const exitMode: OvernightExitMode = holdNextOpen ? exitRule : "close";
  const excludedTickers = excludeBlacklisted && blacklistedTickers
    ? new Set(blacklistedTickers) : undefined;
  const data = raw ? simulateBook(raw, {
    kind, fitTopK: fitAsked, rankTopK: rankAsked, applyNewsSkips, excludedTickers,
    tradeSizes, lookbackDays, asOf: simulationDay(),
  }) : null;
  const rankDates = data && rankOn ? benchmarkDates(data, "rank") : [];
  const fitDates = data && fitOn ? benchmarkDates(data, "fit") : [];
  const rankKey = rankDates.join(",");
  const fitKey = fitDates.join(",");
  const { data: benchmarkData } = useFetchData(
    () => loadStrategyBenchmarks(rankDates, fitDates, fetchBenchmarkReturns, benchmarkCache.current, simulationDay())
      .then((result) => ({ ...result, revision: refresh })),
    { deps: [rankKey, fitKey, refresh] },
  );

  const loadActuals: LoadOvernightActuals = useCallback(async (asOf, tickers, force = false) => {
    const requested = [...new Set(tickers.map((ticker) => ticker.toUpperCase()))];
    const cached = actualsCache.current.get(asOf);
    const needed = force ? requested : requested.filter((ticker) => !cached?.has(ticker)
      || cached.get(ticker)?.status === "awaiting_next_open");
    if (!needed.length) return;
    const batches: string[][] = [];
    for (let index = 0; index < needed.length; index += 30) batches.push(needed.slice(index, index + 30));
    const responses = await Promise.all(batches.map((batch) => fetchOvernightActuals(asOf, batch)));
    const merged = new Map(actualsCache.current.get(asOf));
    responses.forEach((response) => response.rows.forEach((row) => merged.set(row.ticker, row)));
    actualsCache.current.set(asOf, merged);
    setActualsRevision((value) => value + 1);
  }, []);

  const historyJobs = new Map<string, Set<string>>();
  if (data && holdNextOpen) for (const day of data.days) {
    if (day.scan_id) continue;
    for (const strategy of (["rank", "fit"] as const)) {
      if ((strategy === "rank" && !rankOn) || (strategy === "fit" && !fitOn)
        || day[`${strategy}_money`].invested <= 0) continue;
      const tickers = historyJobs.get(day.as_of) ?? new Set<string>();
      // Load every name. A rule change must not leave yesterday's exit unpriced.
      day[strategy].forEach((row) => tickers.add(row.ticker));
      if (tickers.size) historyJobs.set(day.as_of, tickers);
    }
  }
  const historyJobsKey = JSON.stringify([...historyJobs].map(([asOf, tickers]) => ({ asOf, tickers: [...tickers] })));
  useEffect(() => {
    const jobs = JSON.parse(historyJobsKey) as Array<{ asOf: string; tickers: string[] }>;
    if (!holdNextOpen || jobs.length === 0) {
      setHistoryBusy(false);
      setHistoryError(null);
      return;
    }
    let active = true;
    let cursor = 0;
    let failures = 0;
    setHistoryBusy(true);
    setHistoryError(null);
    async function worker() {
      while (cursor < jobs.length) {
        const job = jobs[cursor++];
        try { await loadActuals(job.asOf, job.tickers, openRefresh > 0); }
        catch { failures += 1; }
      }
    }
    void Promise.all(Array.from({ length: Math.min(4, jobs.length) }, worker)).then(() => {
      if (!active) return;
      setHistoryBusy(false);
      if (failures) setHistoryError(`${failures} ${failures === 1 ? "day" : "days"} could not be checked`);
    });
    return () => { active = false; };
  }, [holdNextOpen, historyJobsKey, loadActuals, openRefresh]);

  const holdJobs = new Map<string, string[]>();
  if (data && afterCloseRule.hold) for (const day of data.days) {
    if (day.scan_id) continue;
    const tickers = [...(rankOn ? day.rank : []), ...(fitOn ? day.fit : [])]
      .filter((row) => row.close_price !== null)
      .map((row) => row.ticker);
    if (tickers.length) holdJobs.set(day.as_of, [...new Set(tickers)]);
  }
  const holdJobsKey = JSON.stringify([...holdJobs]);
  useEffect(() => {
    const jobs = JSON.parse(holdJobsKey) as Array<[string, string[]]>;
    if (jobs.length === 0) return;
    let active = true;
    void Promise.all(jobs.map(([asOf, tickers]) => fetchHoldPath(asOf, tickers, 5))).then((responses) => {
      if (!active) return;
      const merged = new Map<string, readonly HoldBar[]>();
      responses.forEach((response) => response.rows.forEach((row) => {
        merged.set(`${response.as_of}:${row.ticker}`, row.sessions);
      }));
      setHoldPaths(merged);
    }).catch(() => {
      if (active) setHoldPaths(new Map());
    });
    return () => { active = false; };
  }, [holdJobsKey]);

  function holdPathsFor(asOf: string): ReadonlyMap<string, readonly HoldBar[]> {
    return new Map([...holdPaths].flatMap(([key, sessions]) => {
      const [day, ticker] = key.split(":");
      return day === asOf ? [[ticker, sessions] as const] : [];
    }));
  }

  // Sessions where both hold levels printed: ask minute bars which came first.
  const holdRule = afterCloseRule.hold;
  const tieJobs = data && holdRule ? data.days.filter((day) => !day.scan_id).flatMap((day) =>
    (["rank", "fit"] as const).filter((strategy) => (strategy === "rank" ? rankOn : fitOn)).flatMap((strategy) =>
      tieCases(day[strategy], holdPathsFor(day.as_of), holdRule.gainPct, holdRule.dropPct, holdRule.days)))
    .filter((c) => !tieAnswers.has(tieKey(c.ticker, c.session, c.target, c.stop))) : [];
  const tieJobsKey = JSON.stringify(tieJobs.map((c) => tieKey(c.ticker, c.session, c.target, c.stop)).sort());
  useEffect(() => {
    if (tieJobs.length === 0) return;
    let active = true;
    const cases = [...new Map(tieJobs.map((c) => [tieKey(c.ticker, c.session, c.target, c.stop), c])).values()];
    void fetchFirstTouch(cases).then(({ results }) => {
      if (!active) return;
      setTieAnswers((current) => {
        const next = new Map(current);
        results.forEach((r) => next.set(tieKey(r.ticker, r.session, r.target, r.stop), r.first));
        return next;
      });
    }).catch(() => {
      // Unresolved ties keep the stop-first fallback.
    });
    return () => { active = false; };
    // tieJobsKey captures the case list; tieJobs itself is rebuilt every render.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [tieJobsKey]);

  function combinedRecord(parts: Array<PaperListStats | undefined>): PaperListStats | undefined {
    const present = parts.filter((part): part is PaperListStats => part !== undefined && part.n_scored > 0);
    if (present.length === 0) return undefined;
    const n = present.reduce((sum, part) => sum + part.n, 0);
    const scored = present.reduce((sum, part) => sum + part.n_scored, 0);
    const wins = present.reduce((sum, part) => sum + part.wins, 0);
    const losses = present.reduce((sum, part) => sum + part.losses, 0);
    const weighted = present.reduce((sum, part) => sum + (part.avg ?? 0) * part.n_scored, 0);
    return {
      n, n_scored: scored, wins, losses, flats: scored - wins - losses,
      hit_rate: scored ? wins / scored : null,
      avg: scored ? weighted / scored : null,
      avg_ex_news: null, n_avoid: 0,
    };
  }

  function recordsFor(strategy: StrategyKind, count: typeof exitStats): PaperListStats | undefined {
    if (exitMode === "close" || !data) return undefined;
    return combinedRecord(data.days.filter((day) => !day.scan_id).map((day) => count(
      day[strategy], exitMode, actualsCache.current.get(day.as_of),
      customSelections.get(overnightSelectionKey(day, strategy)) ?? EMPTY_CUSTOM,
      tradeSizes[strategy], exitPortion,
    )));
  }

  function toggleCustom(day: SimulatedDay, strategy: StrategyKind, ticker: string) {
    setCustomSelections((current) => {
      const next = new Map(current);
      const key = overnightSelectionKey(day, strategy);
      const selected = new Set(next.get(key));
      if (selected.has(ticker)) selected.delete(ticker);
      else selected.add(ticker);
      next.set(key, selected);
      return next;
    });
  }

  if (error) return <p className="error">{error}</p>;
  if (!raw || !data) return <p className="muted">Loading…</p>;
  const currentBenchmarks = benchmarkData?.revision === refresh && benchmarkData.rankKey === rankKey
    && benchmarkData.fitKey === fitKey ? benchmarkData : null;
  const rankBenchmark = currentBenchmarks?.rank ?? null;
  const fitBenchmark = currentBenchmarks?.fit ?? null;
  function holdFor(strategy: "rank" | "fit"): HoldSummary | null {
    const hold = afterCloseRule.hold;
    if (!hold || !data) return null;
    return addHoldSummaries(data.days.filter((day) => !day.scan_id).map((day) => summarizeHold(
      day[strategy], holdPathsFor(day.as_of), hold.gainPct, hold.dropPct, hold.days, tradeSizes[strategy], tieAnswers,
    )));
  }
  const rankHold = holdFor("rank");
  const fitHold = holdFor("fit");
  const holdLabel = `+${holdGainPct}% / −${holdDropPct}% from close, ${holdDays}d`;
  const rankScenario = holdNextOpen ? summarizeSelectedOvernightHistory(
    data.days, actualsCache.current, "rank", tradeSizes.rank, exitPortion / 100,
    exitRule, customSelections,
  ) : null;
  const fitScenario = holdNextOpen ? summarizeSelectedOvernightHistory(
    data.days, actualsCache.current, "fit", tradeSizes.fit, exitPortion / 100,
    exitRule, customSelections,
  ) : null;

  return (
    <>
      <div className="what-if-controls">
      <div className="slice-bar what-if-toolbar">
        <StrategyControl label="Rank" enabled={rankOn} onEnabled={setRankOn} value={rankText} onValue={setRankText}
          dollarsPerTrade={tradeSizes.rank} onTradeSize={(rank) => setTradeSizes((sizes) => ({ ...sizes, rank }))} />
        <StrategyControl label="Fit" enabled={fitOn} onEnabled={setFitOn} value={fitText} onValue={setFitText}
          dollarsPerTrade={tradeSizes.fit} onTradeSize={(fit) => setTradeSizes((sizes) => ({ ...sizes, fit }))} />
        <select className="form-select what-if-period" value={lookbackDays ?? "all"}
          aria-label="Simulation period"
          onChange={(event) => setLookbackDays(LOOKBACK_OPTIONS.find((option) => String(option.days ?? "all") === event.target.value)?.days ?? null)}>
          {LOOKBACK_OPTIONS.map((option) => <option key={option.label} value={option.days ?? "all"}>{option.label}</option>)}
        </select>
        <label className="what-if-news-toggle" title="Turn off to include priced picks skipped by saved news checks in hypothetical results.">
          <input type="checkbox" checked={applyNewsSkips}
            onChange={(event) => setApplyNewsSkips(event.target.checked)} />
          <span>Apply news skips</span>
        </label>
        <label className="what-if-news-toggle" title="Remove currently blacklisted tickers from historical results without backfilling lower-ranked picks.">
          <input type="checkbox" checked={excludeBlacklisted} disabled={blacklistedTickers === null}
            onChange={(event) => setExcludeBlacklisted(event.target.checked)} />
          <span>Exclude blacklisted</span>
        </label>
        {blacklistError && <span className="error" role="status">
          {blacklistError.includes(": 404") ? "Restart API to enable filter" : "Couldn’t load blacklist"}
        </span>}

        <button
          type="button"
          className="icon-btn what-if-action-btn"
          disabled={busy}
          title="Fill Open→Close after the session"
          onClick={async () => {
            setBusy(true);
            setRebuildError(null);
            try {
              await rebuildPaperBook();
              benchmarkCache.current.clear();
              actualsCache.current.clear();
              setActualsRevision((value) => value + 1);
              setRefresh((n) => n + 1);
            } catch (err) {
              setRebuildError(String(err));
            } finally {
              setBusy(false);
            }
          }}
        >
          {busy ? "Updating…" : "Update closes"}
        </button>
      </div>
      <div className="slice-bar what-if-toolbar">
        <label className="what-if-exit-group">
          <span>After the close</span>
          <select aria-label="After the close" value={afterClose}
            onChange={(event) => {
              const mode = event.target.value as AfterCloseMode;
              setAfterClose(mode);
            }}>
            <option value="close">Sell at the close</option>
            <option value="next-open">Next morning open</option>
            <option value="target">Wait for a gain</option>
          </select>
        </label>
          {afterClose === "target" && <>
            <label className="what-if-slider" title="Sell when the later high reaches this far above the close.">
              <span>Close +{holdGainPct}%</span>
              <input type="range" min={0} max={10} step={1} value={holdGainPct}
                aria-label="Gain target percent" onChange={(event) => setHoldGainPct(Number(event.target.value))} />
            </label>
            <label className="what-if-slider" title="Sell if the later low falls this far below the close.">
              <span>Close −{holdDropPct}%</span>
              <input type="range" min={0} max={HOLD_DROP_STOPS.length - 1} step={1}
                value={HOLD_DROP_STOPS.indexOf(holdDropPct)}
                aria-label="Drop stop percent"
                onChange={(event) => setHoldDropPct(HOLD_DROP_STOPS[Number(event.target.value)])} />
            </label>
            <label className="what-if-slider" title="Sell at the close if neither level prints by this day.">
              <span>Wait {holdDays}d</span>
              <input type="range" min={1} max={5} step={1} value={holdDays}
                aria-label="Days to wait" onChange={(event) => setHoldDays(Number(event.target.value))} />
            </label>
          </>}
        {afterClose === "next-open" && <>
            <select className="form-select" value={overnightRule} aria-label="Stocks held to next open"
              onChange={(event) => setOvernightRule(event.target.value as OvernightExitRule)}>
              <option value="all">All included</option>
              <option value="down">Down today</option>
              <option value="custom">Pick stocks</option>
            </select>
            <select className="form-select what-if-hold-portion" value={overnightPortion} aria-label="Portion held to next open"
              onChange={(event) => setOvernightPortion(Number(event.target.value))}>
              {[100, 50, 30, 10].map((value) => <option key={value} value={value}>{value}% held</option>)}
            </select>
            <button type="button" className="icon-btn what-if-action-btn" title="Refresh actual next opens"
              disabled={historyBusy}
              onClick={() => setOpenRefresh((value) => value + 1)}>Refresh opens</button>
        </>}
        {historyBusy && holdNextOpen && <span className="muted">Checking opens…</span>}
        {historyError && holdNextOpen && <span className="error" role="status">{historyError}</span>}
        {rebuildError && <span className="error">{rebuildError}</span>}
      </div>
      </div>
      <div className="slice-bar">
        <input
          className="form-input"
          type="date"
          value={selectedReplayDay}
          onChange={(event) => setReplayDay(event.target.value)}
          aria-label="Replay day"
        />
        <select
          className="form-select"
          value={replayModel}
          onChange={(event) => setReplayModel(event.target.value)}
          aria-label="Replay model"
        >
          <option value="live">Live Fit pickle</option>
          {(trainingRuns?.runs ?? [])
            .filter((run: TrainingRunRecord) => run.has_archived_model)
            .map((run: TrainingRunRecord) => (
              <option key={run.run_id} value={run.run_id}>
                {formatRunLabel(run.started_at, run.holdout_metrics?.directional_accuracy ?? null)}
              </option>
            ))}
        </select>
        <button
          type="button"
          className="icon-btn"
          disabled={busy || !selectedReplayDay}
          onClick={async () => {
            setBusy(true);
            setRebuildError(null);
            try {
              await replayPaperBook(selectedReplayDay, replayModel === "live" ? null : replayModel);
              setRefresh((n) => n + 1);
            } catch (err) {
              setRebuildError(String(err));
            } finally {
              setBusy(false);
            }
          }}
        >
          {busy ? "Scoring…" : "Run"}
        </button>
      </div>
      {data.days.length === 0 || (!rankOn && !fitOn) ? (
        <p className="muted">{!rankOn && !fitOn ? "Turn on Rank or Fit." : "No morning lists in this period."}</p>
      ) : (
        <>
          <div className="slice-result">
            {rankOn && (
              <StrategySummary title="Rank" topK={rankAsked} days={data.rank_days}
                stats={data.rank_stats} exitRecord={recordsFor("rank", exitStats)} outcomeRecord={recordsFor("rank", outcomeStats)} money={data.rank_money}
                scenario={rankScenario} exitMode={exitMode}
                holdSummary={rankHold} holdLabel={holdLabel}
                benchmark={compareStrategyToBenchmark(data, "rank", rankBenchmark?.response ?? null)}
                benchmarkLoading={rankDates.length > 0 && !rankBenchmark}
                benchmarkError={rankBenchmark?.error ?? null} />
            )}
            {fitOn && (
              <StrategySummary title="Fit" topK={fitAsked} days={data.fit_days}
                stats={data.fit_stats} exitRecord={recordsFor("fit", exitStats)} outcomeRecord={recordsFor("fit", outcomeStats)} money={data.fit_money}
                scenario={fitScenario} exitMode={exitMode}
                holdSummary={fitHold} holdLabel={holdLabel}
                benchmark={compareStrategyToBenchmark(data, "fit", fitBenchmark?.response ?? null)}
                benchmarkLoading={fitDates.length > 0 && !fitBenchmark}
                benchmarkError={fitBenchmark?.error ?? null} />
            )}
          </div>
          {data.days.map((day) => (
            <DayCard
              day={day}
              kind={kind}
              rankAsked={rankAsked}
              fitAsked={fitAsked}
              tradeSizes={tradeSizes}
              hold={afterCloseRule.hold ? {
                gainPct: holdGainPct,
                dropPct: holdDropPct,
                days: holdDays,
                paths: holdPathsFor(day.as_of),
                ties: tieAnswers,
              } : null}
              actuals={actualsCache.current.get(day.as_of)}
              overnightPortion={exitPortion}
              exitMode={exitMode}
              custom={{
                rank: customSelections.get(overnightSelectionKey(day, "rank")) ?? EMPTY_CUSTOM,
                fit: customSelections.get(overnightSelectionKey(day, "fit")) ?? EMPTY_CUSTOM,
              }}
              onToggleHold={(strategy, ticker) => toggleCustom(day, strategy, ticker)}
              modelLabel={
                day.model_run_id
                  ? (() => {
                      const run = (trainingRuns?.runs ?? []).find((item) => item.run_id === day.model_run_id);
                      return run
                        ? formatRunLabel(run.started_at, run.holdout_metrics?.directional_accuracy ?? null)
                        : day.model_run_id.slice(0, 8);
                    })()
                  : day.scan_id
                    ? "Live Fit pickle"
                    : undefined
              }
              key={`${day.as_of}-${day.scan_id || "live"}`}
            />
          ))}
        </>
      )}
    </>
  );
}
