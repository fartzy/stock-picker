import { useState } from "react";
import {
  fetchPaperBook,
  fetchTrainingRuns,
  rebuildPaperBook,
  replayPaperBook,
  type PaperListStats,
  type TrainingRunRecord,
} from "../api";
import { useFetchData } from "../useFetchData";
import { ColumnTitle, DataTable, NewsCell, ScoreCell, TickerCell, UsdCell } from "./DataTable";
import { Diff } from "./Diff";
import { formatUsd } from "../format";
import {
  DEFAULT_TRADE_SIZE, LOOKBACK_OPTIONS, TRADE_SIZE_OPTIONS, simulateBook, simulationDay,
  type LookbackDays, type MoneySummary, type PickKind, type PickOutcome,
  type SimulatedDay, type SimulatedPick, type TradeSizes,
} from "../whatIf";
import { StatStrip, type StatItem } from "./StatStrip";
import TogglePill from "./TogglePill";

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
  if (outcome.status !== "scored") {
    return <span className="muted">{outcome.status === "skipped" ? "Skipped" : "Awaiting close"}</span>;
  }
  return ending ? <UsdCell value={outcome.endingValue} /> : <Dollars value={outcome.pnl} />;
}

function moneyItems(money: MoneySummary): StatItem[] {
  return [
    { key: "capital", label: "Completed capital", value: formatUsd(money.invested) },
    { key: "pnl", label: "P&L", value: <Dollars value={money.pnl} /> },
    { key: "ending", label: "End value", value: <UsdCell value={money.endingValue} /> },
    ...(money.pending ? [{ key: "pending", value: `${money.pending} awaiting close` }] : []),
  ];
}

function statsItems(stats: PaperListStats | undefined): StatItem[] {
  if (!stats || stats.n_scored === 0) return [];
  const hit = stats.hit_rate !== null ? `${(stats.hit_rate * 100).toFixed(0)}% hit` : null;
  return [
    { key: "scored", label: "Scored", value: stats.n_scored },
    { key: "wl", label: "W/L", value: `${stats.wins}W/${stats.losses}L` },
    ...(hit ? [{ key: "hit", label: "Hit", value: hit }] : []),
    { key: "avg", label: "Avg", value: <Pct value={stats.avg} /> },
    ...(stats.n_avoid
      ? [{ key: "skipped", label: "Skipped", value: `${stats.n_avoid} gap-down news` }]
      : []),
  ];
}

function StrategySummary({ title, topK, days, stats, money }: {
  title: string;
  topK: number | undefined;
  days: number;
  stats: PaperListStats | undefined;
  money: MoneySummary;
}) {
  return (
    <div className="view-card slice-card" aria-label={`${title} simulation total`}>
      <div className="slice-card-kicker">
        <StatStrip items={[
          { key: "title", title: true, align: "start", value: title },
          { key: "slice", align: "start", value: topK === undefined ? "all" : `top ${topK}` },
          { key: "days", align: "start", value: `${days}d` },
        ]} />
      </div>
      <div className="slice-card-avg"><Dollars value={money.pnl} /> <span className="muted">P&L</span></div>
      <p className="view-meta">
        <StatStrip items={[
          { key: "names", align: "start", value: `${stats?.n ?? 0} names` },
          ...statsItems(stats),
          ...(money.pending ? [{ key: "pending", value: `${money.pending} awaiting close` }] : []),
        ]} />
      </p>
    </div>
  );
}

function ListTable({
  title,
  rows,
  stats,
  isRank,
  money,
  dollarsPerTrade,
}: {
  title: string;
  rows: SimulatedPick[];
  stats: PaperListStats | undefined;
  isRank: boolean;
  money: MoneySummary;
  dollarsPerTrade: number;
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
            ...statsItems(stats),
          ]}
        />
      </p>
      <p className="view-meta"><StatStrip items={moneyItems(money)} /></p>
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
            { key: "session", header: "Session", numeric: true, cell: (row) => <Pct value={row.session_return} /> },
            { key: "pnl", header: <ColumnTitle label="P&L" hint={`${formatUsd(dollarsPerTrade)} / trade`} />, numeric: true,
              cell: (row) => <PickMoney outcome={row.hypothetical} /> },
            { key: "ending", header: "End value", numeric: true,
              cell: (row) => <PickMoney outcome={row.hypothetical} ending /> },
            { key: "news", header: "News", cell: (row) => <NewsCell flag={row.news_flag} blocks={row.news_blocks} checked={row.news_checked} /> },
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
}: {
  day: SimulatedDay;
  kind: PickKind;
  rankAsked: number | undefined;
  fitAsked: number | undefined;
  modelLabel?: string;
  tradeSizes: TradeSizes;
}) {
  const rankNote = kind !== "fit" ? shortfallNote(rankAsked, day.rank.length, "Rank") : null;
  const fitNote = kind !== "rank" ? shortfallNote(fitAsked, day.fit.length, "Fit") : null;
  return (
    <details className="view-card">
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
                    label: "Rank",
                    value: (
                      <>
                        {day.rank.length} <Pct value={day.rank_avg} /> · <Dollars value={day.rank_money.pnl} />
                      </>
                    ),
                  },
                ]
              : []),
            ...(kind !== "rank" && day.fit.length > 0
              ? [
                  {
                    key: "fit",
                    label: "Fit",
                    value: (
                      <>
                        {day.fit.length} <Pct value={day.fit_avg} /> · <Dollars value={day.fit_money.pnl} />
                      </>
                    ),
                  },
                ]
              : []),
            ...(rankNote ? [{ key: "rank-note", align: "start" as const, value: rankNote }] : []),
            ...(fitNote ? [{ key: "fit-note", align: "start" as const, value: fitNote }] : []),
          ]}
        />
      </summary>
      <div className="what-if-day-lists">
        {kind !== "rank" && (
          <ListTable title="Fit" rows={day.fit} stats={day.fit_stats} isRank={false}
            money={day.fit_money} dollarsPerTrade={tradeSizes.fit} />
        )}
        {kind !== "fit" && (
          <ListTable title="Rank" rows={day.rank} stats={day.rank_stats} isRank={true}
            money={day.rank_money} dollarsPerTrade={tradeSizes.rank} />
        )}
      </div>
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
  const [refresh, setRefresh] = useState(0);
  const [busy, setBusy] = useState(false);
  const [rebuildError, setRebuildError] = useState<string | null>(null);
  const [replayDay, setReplayDay] = useState("");
  const [replayModel, setReplayModel] = useState("live");
  const { data: raw, error } = useFetchData(() => fetchPaperBook("both"), { deps: [refresh] });
  const { data: trainingRuns } = useFetchData(fetchTrainingRuns, { deps: [] });

  if (error) return <p className="error">{error}</p>;
  if (!raw) return <p className="muted">Loading…</p>;

  const rankAsked = rankOn ? parseTop(rankText) : undefined;
  const fitAsked = fitOn ? parseTop(fitText) : undefined;
  const kind: PickKind = rankOn && fitOn ? "both" : rankOn ? "rank" : "fit";
  const data = simulateBook(raw, {
    kind, fitTopK: fitAsked, rankTopK: rankAsked, tradeSizes, lookbackDays, asOf: simulationDay(),
  });

  return (
    <div>
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
        <button
          type="button"
          className="icon-btn"
          disabled={busy}
          title="Fill Open→Close after the session"
          aria-label="Update What if after the close"
          onClick={async () => {
            setBusy(true);
            setRebuildError(null);
            try {
              await rebuildPaperBook();
              setRefresh((n) => n + 1);
            } catch (err) {
              setRebuildError(String(err));
            } finally {
              setBusy(false);
            }
          }}
        >
          {busy ? "Updating…" : "↻"}
        </button>
        <details className="what-if-help">
          <summary aria-label="Simulation details" title="Simulation details">ⓘ</summary>
          <div className="what-if-help-body">
            <p>Each pick uses its strategy’s selected amount every day. Fractional shares; fees and slippage excluded. Totals include live lists only, not replays.</p>
            <p>{data.from ? `${data.from} – ${data.through} (rolling calendar weeks).` : `All recorded days through ${data.through}.`}</p>
          </div>
        </details>
        {rebuildError && <span className="error">{rebuildError}</span>}
      </div>
      <div className="slice-bar">
        <input
          className="form-input"
          type="date"
          value={replayDay || raw.days[0]?.as_of || ""}
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
          disabled={busy || !replayDay}
          onClick={async () => {
            setBusy(true);
            setRebuildError(null);
            try {
              await replayPaperBook(replayDay, replayModel === "live" ? null : replayModel);
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
                stats={data.rank_stats} money={data.rank_money} />
            )}
            {fitOn && (
              <StrategySummary title="Fit" topK={fitAsked} days={data.fit_days}
                stats={data.fit_stats} money={data.fit_money} />
            )}
          </div>
          {data.days.map((day) => (
            <DayCard
              day={day}
              kind={kind}
              rankAsked={rankAsked}
              fitAsked={fitAsked}
              tradeSizes={tradeSizes}
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
    </div>
  );
}
