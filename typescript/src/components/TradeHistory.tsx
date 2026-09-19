import { useMemo, useState } from "react";
import {
  fetchBenchmarkReturns,
  fetchPositions,
  type BenchmarkReturnsResponse,
  type Position,
  type PositionsResponse,
} from "../api";
import AddTradeForm from "./AddTradeForm";
import { Diff } from "./Diff";
import { formatUsd } from "../format";
import { useFetchData } from "../useFetchData";

const TRADE_TIMEZONE = "America/New_York";


function formatTime(executedAt: string): string {
  return new Date(executedAt).toLocaleTimeString("en-US", {
    timeZone: TRADE_TIMEZONE,
    hour: "numeric",
    minute: "2-digit",
  });
}

function formatDay(day: string): string {
  return new Date(`${day}T12:00:00`).toLocaleDateString("en-US", {
    weekday: "long",
    month: "short",
    day: "numeric",
  });
}

function mondayOfDay(day: string): string {
  const [year, month, date] = day.split("-").map(Number);
  const utc = new Date(Date.UTC(year, month - 1, date));
  const weekday = utc.getUTCDay();
  utc.setUTCDate(utc.getUTCDate() - (weekday === 0 ? 6 : weekday - 1));
  return utc.toISOString().slice(0, 10);
}

function addDaysIso(day: string, days: number): string {
  const [year, month, date] = day.split("-").map(Number);
  const utc = new Date(Date.UTC(year, month - 1, date + days));
  return utc.toISOString().slice(0, 10);
}

function formatWeekRange(monday: string): string {
  const friday = addDaysIso(monday, 4);
  const start = new Date(`${monday}T12:00:00`).toLocaleDateString("en-US", {
    month: "short",
    day: "numeric",
  });
  const end = new Date(`${friday}T12:00:00`).toLocaleDateString("en-US", {
    month: "short",
    day: "numeric",
  });
  return `Week ${start}–${end}`;
}

function sessionDay(executedAt: string): string {
  return new Date(executedAt).toLocaleDateString("en-CA", { timeZone: TRADE_TIMEZONE });
}

function formatStamp(executedAt: string, relativeToDay?: string): string {
  const time = `${formatTime(executedAt)} ET`;
  if (!relativeToDay || sessionDay(executedAt) === relativeToDay) return time;
  const when = new Date(executedAt).toLocaleDateString("en-US", {
    timeZone: TRADE_TIMEZONE,
    month: "short",
    day: "numeric",
  });
  return `${when} ${time}`;
}

interface PositionsSummary {
  invested: number;
  pnl: number;
  hasUnknownPnl: boolean;
}

function summarizePositions(positions: Position[]): PositionsSummary {
  return {
    invested: positions.reduce((sum, p) => sum + p.invested, 0),
    pnl: positions.reduce((sum, p) => sum + (p.pnl ?? 0), 0),
    hasUnknownPnl: positions.some((p) => p.pnl === null),
  };
}

function SummaryLine({ label, summary }: { label: string; summary: PositionsSummary }) {
  const { invested, pnl, hasUnknownPnl } = summary;
  return (
    <>
      {label} &middot; {formatUsd(invested)} invested &middot;{" "}
      <Diff value={pnl} pct={invested ? pnl / invested : null} />
      {hasUnknownPnl ? " (partial)" : ""}
    </>
  );
}

function holdToCloseTotal(positions: Position[]): { pnl: number; invested: number } | null {
  const priced = positions.filter((p) => p.hold_close_pnl !== null);
  if (priced.length === 0) return null;
  return {
    pnl: priced.reduce((sum, p) => sum + (p.hold_close_pnl ?? 0), 0),
    invested: priced.reduce((sum, p) => sum + p.invested, 0),
  };
}

function PnlCell({ position, caption }: { position: Position; caption: string }) {
  return (
    <td className="trade-num">
      <div>
        {position.pnl !== null ? (
          <Diff value={position.pnl} pct={position.invested ? position.pnl / position.invested : null} />
        ) : (
          "--"
        )}
      </div>
      <div className="muted">{caption}</div>
    </td>
  );
}

function OpenRow({ position }: { position: Position }) {
  return (
    <tr>
      <td className="trade-ticker">{position.ticker}</td>
      <td className="trade-num">{position.shares}</td>
      <td className="trade-time">{position.buy_time ? `${formatTime(position.buy_time)} ET` : "--"}</td>
      <td className="trade-num">{position.buy_price !== null ? formatUsd(position.buy_price) : "--"}</td>
      <td className="trade-num">{position.current_price !== null ? formatUsd(position.current_price) : "--"}</td>
      <PnlCell position={position} caption="unrealized" />
      <td className="trade-num">{formatUsd(position.invested)}</td>
    </tr>
  );
}

function ClosedRow({ position }: { position: Position }) {
  const holdClosePnl = position.hold_close_pnl ?? null;
  const holdClosePrice = position.hold_close_price ?? null;
  const holdPct =
    holdClosePnl !== null && position.invested ? holdClosePnl / position.invested : null;
  return (
    <tr>
      <td className="trade-ticker">{position.ticker}</td>
      <td className="trade-num">{position.shares}</td>
      <td className="trade-time">{position.buy_time ? formatStamp(position.buy_time, position.day) : "--"}</td>
      <td className="trade-num">{position.buy_price !== null ? formatUsd(position.buy_price) : "--"}</td>
      <td className="trade-time">{position.sell_time ? formatStamp(position.sell_time, position.day) : "--"}</td>
      <td className="trade-num">{position.sell_price !== null ? formatUsd(position.sell_price) : "--"}</td>
      <PnlCell position={position} caption="realized" />
      <td className="trade-num">
        {holdClosePrice !== null ? formatUsd(holdClosePrice) : "--"}
      </td>
      <td className="trade-num">
        {holdClosePnl !== null ? (
          <>
            <Diff value={holdClosePnl} pct={holdPct} />
            <div className="muted">8:40–2:55 CT</div>
          </>
        ) : (
          <span className="muted">session open</span>
        )}
      </td>
      <td className="trade-num">{formatUsd(position.invested)}</td>
    </tr>
  );
}

const OPEN_COLUMNS = ["Ticker", "Shares", "Bought", "Buy Price", "Last", "P&L", "Invested"];
const CLOSED_COLUMNS = [
  "Ticker",
  "Shares",
  "Bought",
  "Buy Price",
  "Sold",
  "Sell Price",
  "P&L",
  "Close",
  "8:40–2:55 CT",
  "Invested",
];

function columnClass(column: string): string | undefined {
  return column === "Ticker" || column === "Bought" || column === "Sold" ? undefined : "trade-num";
}

function HoldToCloseNote({ positions }: { positions: Position[] }) {
  const total = holdToCloseTotal(positions);
  if (total === null) return null;
  return (
    <>
      {" "}
      &middot; 8:40–2:55 CT{" "}
      <Diff value={total.pnl} pct={total.invested ? total.pnl / total.invested : null} />
    </>
  );
}

function SpyPct({ value }: { value: number }) {
  const isUp = value >= 0;
  return (
    <span className={isUp ? "quote-diff-up" : "quote-diff-down"}>
      {isUp ? "▲" : "▼"} {(Math.abs(value) * 100).toFixed(2)}%
    </span>
  );
}

function BenchmarkNote({
  sessionReturn,
  overnightReturn,
}: {
  sessionReturn: number | undefined;
  overnightReturn: number | undefined;
}) {
  if (sessionReturn === undefined && overnightReturn === undefined) return null;
  return (
    <>
      {sessionReturn !== undefined && (
        <>
          {" "}
          &middot; intraday S&amp;P{" "}
          <SpyPct value={sessionReturn} />
        </>
      )}
      {overnightReturn !== undefined && (
        <>
          {" "}
          &middot; buy-and-hold S&amp;P{" "}
          <SpyPct value={overnightReturn} />
        </>
      )}
    </>
  );
}

function ClosedDayGroup({
  day,
  positions,
  sessionReturn,
  overnightReturn,
  expanded,
  onToggle,
}: {
  day: string;
  positions: Position[];
  sessionReturn: number | undefined;
  overnightReturn: number | undefined;
  expanded: boolean;
  onToggle: () => void;
}) {
  const summary = summarizePositions(positions);
  const session = holdToCloseTotal(positions);
  return (
    <details
      className="view-card"
      open={expanded}
      onToggle={(event) => {
        if ((event.currentTarget as HTMLDetailsElement).open !== expanded) onToggle();
      }}
    >
      <summary>
        <strong style={{ color: "var(--accent)" }}>{formatDay(day)}</strong>{" "}
        <span className="view-meta">
          {positions.length} lot{positions.length === 1 ? "" : "s"}
          {" · "}
          {formatUsd(summary.invested)}
          {" · "}
          <Diff value={summary.pnl} pct={summary.invested ? summary.pnl / summary.invested : null} />
          {session !== null && (
            <>
              {" · 8:40–2:55 CT "}
              <Diff
                value={session.pnl}
                pct={session.invested ? session.pnl / session.invested : null}
              />
            </>
          )}
          <BenchmarkNote sessionReturn={sessionReturn} overnightReturn={overnightReturn} />
        </span>
      </summary>
      <div style={{ overflowX: "auto", marginTop: "var(--space-3)" }}>
        <table className="trade-table">
          <thead>
            <tr>
              {CLOSED_COLUMNS.map((column) => (
                <th key={column} className={columnClass(column)}>
                  {column}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {positions.map((position, index) => (
              <ClosedRow position={position} key={`${position.ticker}-${position.day}-${index}`} />
            ))}
          </tbody>
        </table>
      </div>
    </details>
  );
}

function groupClosedByDay(positions: Position[]): [string, Position[]][] {
  const byDay = new Map<string, Position[]>();
  for (const position of positions) {
    const group = byDay.get(position.day) ?? [];
    group.push(position);
    byDay.set(position.day, group);
  }
  return [...byDay.entries()].sort(([a], [b]) => (a < b ? 1 : -1));
}

function groupClosedByWeek(dayGroups: [string, Position[]][]): [string, [string, Position[]][]][] {
  const byWeek = new Map<string, [string, Position[]][]>();
  for (const group of dayGroups) {
    const monday = mondayOfDay(group[0]);
    const week = byWeek.get(monday) ?? [];
    week.push(group);
    byWeek.set(monday, week);
  }
  return [...byWeek.entries()].sort(([a], [b]) => (a < b ? 1 : -1));
}

function isUnmatchedSell(position: Position): boolean {
  return position.closed && position.shares === 0 && position.buy_price === null;
}

function inCalendarMonth(day: string, now: Date): boolean {
  return day.startsWith(`${now.getFullYear()}-${String(now.getMonth() + 1).padStart(2, "0")}`);
}

function inCalendarYear(day: string, now: Date): boolean {
  return day.startsWith(`${now.getFullYear()}-`);
}

function holdPctForDays(days: string[], overnight: Record<string, number>): number | null {
  const unique = [...new Set(days)].sort();
  let wealth = 1;
  let counted = 0;
  for (const day of unique) {
    const pct = overnight[day];
    if (pct === undefined) continue;
    wealth *= 1 + pct;
    counted += 1;
  }
  return counted > 0 ? wealth - 1 : null;
}

function windowSummary(
  lots: Position[],
  spySessionByDay: Record<string, number>,
  peakWorking: Record<string, number> = {},
): {
  lots: Position[];
  sessions: number;
  typicalWorking: number;
  pnl: number;
  spySessionDollars: number;
} {
  const pnl = lots.reduce((sum, p) => sum + (p.pnl ?? 0), 0);
  const byDay = new Map<string, Position[]>();
  for (const lot of lots) {
    const group = byDay.get(lot.day) ?? [];
    group.push(lot);
    byDay.set(lot.day, group);
  }
  let spySessionDollars = 0;
  let workingSum = 0;
  for (const [day, dayLots] of byDay) {
    const stacked = dayLots.reduce((sum, p) => sum + p.invested, 0);
    const dayWorking = peakWorking[day] ?? stacked;
    workingSum += dayWorking;
    const spy = spySessionByDay[day];
    if (spy !== undefined) spySessionDollars += dayWorking * spy;
  }
  const sessions = byDay.size;
  return {
    lots,
    sessions,
    typicalWorking: sessions ? workingSum / sessions : 0,
    pnl,
    spySessionDollars,
  };
}

type PeriodSummary = {
  lots: Position[];
  sessions: number;
  typicalWorking: number;
  pnl: number;
  spySessionDollars: number;
};

function PeriodLine({
  label,
  summary,
  holdPct,
}: {
  label: string;
  summary: PeriodSummary;
  holdPct: number | null;
}) {
  const denom = summary.typicalWorking;
  return (
    <>
      <strong>{label}</strong>
      {` · ${formatUsd(summary.typicalWorking)} avg · ${summary.sessions}d · `}
      <Diff value={summary.pnl} pct={denom ? summary.pnl / denom : null} />
      {" · intraday S&P "}
      <Diff
        value={summary.spySessionDollars}
        pct={denom ? summary.spySessionDollars / denom : null}
      />
      {holdPct !== null && (
        <>
          {" · buy-and-hold S&P "}
          <Diff value={denom * holdPct} pct={holdPct} />
        </>
      )}
      <HoldToCloseNote positions={summary.lots} />
    </>
  );
}

function PeriodRow({
  label,
  summary,
  holdPct,
}: {
  label: string;
  summary: PeriodSummary;
  holdPct: number | null;
}) {
  if (summary.lots.length === 0 || summary.sessions === 0) {
    return (
      <p className="muted" style={{ margin: "0 0 var(--space-2)" }}>
        {label}: no closed lots
      </p>
    );
  }
  return (
    <p className="period-row">
      <PeriodLine label={label} summary={summary} holdPct={holdPct} />
    </p>
  );
}

export default function TradeHistory() {
  const [refreshCount, setRefreshCount] = useState(0);
  const [openNowExpanded, setOpenNowExpanded] = useState(true);
  const [expandedDays, setExpandedDays] = useState<Set<string>>(() => new Set());
  const [weekOpen, setWeekOpen] = useState<Record<string, boolean>>({});
  const { data, error } = useFetchData<PositionsResponse>(fetchPositions, {
    deps: [refreshCount],
  });

  const openPositions = useMemo(
    () => (data?.positions ?? []).filter((position) => !position.closed && position.shares > 0),
    [data],
  );
  const closedLots = useMemo(
    () => (data?.positions ?? []).filter((position) => position.closed && !isUnmatchedSell(position)),
    [data],
  );
  const unmatchedSells = useMemo(
    () => (data?.positions ?? []).filter(isUnmatchedSell),
    [data],
  );
  const dayGroups = useMemo(() => groupClosedByDay(closedLots), [closedLots]);
  const days = useMemo(() => dayGroups.map(([day]) => day), [dayGroups]);
  const { data: benchmarkData } = useFetchData<BenchmarkReturnsResponse>(
    () => (days.length > 0 ? fetchBenchmarkReturns(days) : Promise.resolve({ returns: {} })),
    { deps: [days.join(",")] },
  );

  if (error) return <p className="error">{error}</p>;
  if (!data) return <p className="muted">Loading trade history...</p>;

  const nothing = openPositions.length === 0 && closedLots.length === 0 && unmatchedSells.length === 0;
  const now = new Date();
  const spySessions = benchmarkData?.returns ?? {};
  const overnight = benchmarkData?.overnight ?? {};
  const monthLots = closedLots.filter((p) => inCalendarMonth(p.day, now));
  const yearLots = closedLots.filter((p) => inCalendarYear(p.day, now));
  const peaks = data.peak_working ?? {};
  const month = windowSummary(monthLots, spySessions, peaks);
  const year = windowSummary(yearLots, spySessions, peaks);
  const allTime = windowSummary(closedLots, spySessions, peaks);
  const allHold = benchmarkData?.hold?.pct ?? null;

  return (
    <div>
      {nothing ? (
        <p className="muted">No trades logged yet.</p>
      ) : (
        <>
          <div className="view-card" style={{ marginBottom: "var(--space-3)" }}>
            <PeriodRow
              label="MTD"
              summary={month}
              holdPct={holdPctForDays(monthLots.map((p) => p.day), overnight) ?? allHold}
            />
            <PeriodRow
              label="YTD"
              summary={year}
              holdPct={holdPctForDays(yearLots.map((p) => p.day), overnight) ?? allHold}
            />
            <PeriodRow
              label="All"
              summary={allTime}
              holdPct={holdPctForDays(closedLots.map((p) => p.day), overnight) ?? allHold}
            />
          </div>
          <details
            className="view-card"
            open={openNowExpanded}
            onToggle={(event) => {
              setOpenNowExpanded((event.currentTarget as HTMLDetailsElement).open);
            }}
          >
            <summary>
              <strong style={{ color: "var(--accent)" }}>Open now</strong>{" "}
              <span className="view-meta">
                {openPositions.length === 0 ? (
                  "No open lots"
                ) : (
                  <SummaryLine
                    label={`${openPositions.length} lot${openPositions.length === 1 ? "" : "s"}`}
                    summary={summarizePositions(openPositions)}
                  />
                )}
              </span>
            </summary>
            {openPositions.length === 0 ? (
              <p className="muted" style={{ marginTop: "var(--space-3)" }}>
                Nothing left open.
              </p>
            ) : (
              <div style={{ overflowX: "auto", marginTop: "var(--space-3)" }}>
                <table className="trade-table">
                  <thead>
                    <tr>
                      {OPEN_COLUMNS.map((column) => (
                        <th key={column} className={columnClass(column)}>
                          {column}
                        </th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {openPositions.map((position) => (
                      <OpenRow position={position} key={`${position.ticker}-open`} />
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </details>

          {groupClosedByWeek(dayGroups).map(([monday, weekDays], weekIndex) => {
            const lotsThisWeek = weekDays.flatMap(([, lots]) => lots);
            const weekSummary = windowSummary(lotsThisWeek, spySessions, peaks);
            const weekHold = holdPctForDays(
              lotsThisWeek.map((p) => p.day),
              overnight,
            );
            const isOpen = monday in weekOpen ? weekOpen[monday] : weekIndex === 0;
            return (
              <details
                className="view-card week-block"
                key={monday}
                open={isOpen}
                onToggle={(event) => {
                  const open = (event.currentTarget as HTMLDetailsElement).open;
                  setWeekOpen((current) => ({ ...current, [monday]: open }));
                }}
              >
                <summary>
                  <span className="view-meta">
                    <PeriodLine
                      label={formatWeekRange(monday)}
                      summary={weekSummary}
                      holdPct={weekHold}
                    />
                  </span>
                </summary>
                {weekDays.map(([day, dayPositions]) => (
                  <ClosedDayGroup
                    day={day}
                    positions={dayPositions}
                    sessionReturn={benchmarkData?.returns[day]}
                    overnightReturn={benchmarkData?.overnight?.[day]}
                    expanded={expandedDays.has(day)}
                    onToggle={() =>
                      setExpandedDays((current) => {
                        const next = new Set(current);
                        if (next.has(day)) next.delete(day);
                        else next.add(day);
                        return next;
                      })
                    }
                    key={day}
                  />
                ))}
              </details>
            );
          })}

          {unmatchedSells.length > 0 && (
            <p className="muted" style={{ marginTop: "var(--space-3)" }}>
              Sells with no matching buy in the log:{" "}
              {unmatchedSells
                .map((position) =>
                  position.sell_price !== null
                    ? `${position.ticker} @ ${formatUsd(position.sell_price)}`
                    : position.ticker,
                )
                .join(", ")}
              .
            </p>
          )}
        </>
      )}
      <AddTradeForm onAdded={() => setRefreshCount((c) => c + 1)} />
    </div>
  );
}
