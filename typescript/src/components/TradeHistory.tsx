import { useMemo, useState } from "react";
import {
  fetchBenchmarkReturns,
  fetchFees,
  fetchPositions,
  type BenchmarkReturnsResponse,
  type FeeRecord,
  type FeesResponse,
  type Position,
  type PositionsResponse,
} from "../api";
import AddTradeForm from "./AddTradeForm";
import { formatUsd } from "../format";
import {
  CLOSED_LOT_COLUMNS,
  Diff,
  HOLD_WINDOW_LABEL,
  OPEN_LOT_COLUMNS,
  SignedPct,
  StatBody,
  StatColGroup,
  StatDetail,
  StatExpand,
  StatHead,
  StatRow,
  StatStrip,
  StatTable,
  lotColumnClass,
  periodColumns,
  type StatItem,
  type StatValues,
} from "./stats";
import { useFetchData } from "../useFetchData";

const TRADE_TIMEZONE = "America/New_York";
// Session closes and imported fills can change while this view stays open.
const REFRESH_INTERVAL_MS = 60_000;

function roundUsd(value: number): number {
  return Math.round(value * 100) / 100;
}

function formatTime(executedAt: string): string {
  return new Date(executedAt).toLocaleTimeString("en-US", {
    timeZone: TRADE_TIMEZONE,
    hour: "numeric",
    minute: "2-digit",
  });
}

function formatDay(day: string): string {
  return new Date(`${day}T12:00:00`).toLocaleDateString("en-US", {
    weekday: "short",
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
  manualFees: number;
  hasUnknownPnl: boolean;
}

function summarizePositions(positions: Position[]): PositionsSummary {
  return {
    invested: positions.reduce((sum, p) => sum + p.invested, 0),
    pnl: positions.reduce((sum, p) => sum + (p.pnl ?? 0), 0),
    manualFees: positions.reduce((sum, p) => sum + (p.manual_fee ?? 0), 0),
    hasUnknownPnl: positions.some((p) => p.pnl === null),
  };
}

function openNowItems(count: number, summary: PositionsSummary): StatItem[] {
  const { invested, pnl, manualFees, hasUnknownPnl } = summary;
  const netPnl = roundUsd(pnl - manualFees);
  return [
    { key: "title", title: true, align: "start", value: <strong style={{ color: "var(--accent)" }}>Open now</strong> },
    { key: "lots", align: "start", value: count === 0 ? "No open lots" : `${count} lot${count === 1 ? "" : "s"}` },
    ...(count === 0
      ? []
      : [
          { key: "on-book", label: "On book", value: formatUsd(invested) },
          {
            key: "pnl",
            label: "P&L",
            value: (
              <>
                <Diff value={netPnl} pct={invested ? netPnl / invested : null} />
                {hasUnknownPnl ? " (partial)" : ""}
              </>
            ),
          },
        ]),
  ];
}

function holdToCloseTotal(positions: Position[]): { pnl: number; typicalInvested: number } | null {
  const priced = positions.filter((p) => p.hold_close_pnl !== null);
  if (priced.length === 0) return null;
  return {
    pnl: priced.reduce((sum, p) => sum + (p.hold_close_pnl ?? 0), 0),
    // This comparison has its own capital base: late buys must not dilute it.
    typicalInvested: priced.reduce((sum, p) => sum + p.hold_eligible_invested, 0)
      / new Set(priced.map((p) => p.day)).size,
  };
}

function feeSum(fees: FeeRecord[], days?: Iterable<string>): number {
  const allow = days === undefined ? null : new Set(days);
  return fees.reduce((sum, fee) => (allow === null || allow.has(fee.day) ? sum + fee.amount : sum), 0);
}

/** One ticket per closed lot, same ticker and session. Extra tickets stay leftover. */
function assignLotFees(
  lots: Position[],
  fees: FeeRecord[],
): { lotFees: number[]; leftover: FeeRecord[] } {
  const unused = [...fees];
  const lotFees = lots.map((lot) => {
    const index = unused.findIndex((fee) => fee.day === lot.day && fee.ticker === lot.ticker);
    if (index < 0) return 0;
    const [fee] = unused.splice(index, 1);
    return fee.amount;
  });
  return { lotFees, leftover: unused };
}

function PnlCell({
  pnl,
  invested,
  caption,
}: {
  pnl: number | null;
  invested: number;
  caption?: string;
}) {
  return (
    <td className="trade-num lot-pnl">
      {pnl !== null ? <Diff value={pnl} pct={invested ? pnl / invested : null} /> : "--"}
      {caption ? <div className="muted">{caption}</div> : null}
    </td>
  );
}

function OpenRow({ position, onOpenOvernight }: { position: Position; onOpenOvernight: (ticker: string) => void }) {
  return (
    <tr>
      <td className="trade-ticker"><span className="overnight-ticker-action">{position.ticker}<button type="button" className="overnight-row-link" title={`Open ${position.ticker} overnight quick look`} aria-label={`Open ${position.ticker} overnight quick look`} onClick={() => onOpenOvernight(position.ticker)}>↗</button></span></td>
      <td className="trade-num">{position.shares}</td>
      <td className="trade-time">{position.buy_time ? `${formatTime(position.buy_time)} ET` : "--"}</td>
      <td className="trade-num">{position.buy_price !== null ? formatUsd(position.buy_price) : "--"}</td>
      <td className="trade-num">{position.current_price !== null ? formatUsd(position.current_price) : "--"}</td>
      <PnlCell pnl={position.pnl === null ? null : roundUsd(position.pnl - (position.manual_fee ?? 0))} invested={position.invested} caption="unrealized" />
      <td className="trade-num">{formatUsd(position.invested)}</td>
    </tr>
  );
}

function feeAmount(tickets: number, show: boolean) {
  if (!show) return "";
  return tickets > 0 ? formatUsd(tickets) : "—";
}

function ClosedRow({
  position,
  fee = 0,
  showFees = false,
}: {
  position: Position;
  fee?: number;
  showFees?: boolean;
}) {
  const holdClosePnl = position.hold_close_pnl ?? null;
  const holdClosePrice = position.hold_close_price ?? null;
  const holdPct =
    holdClosePnl !== null && position.hold_eligible_invested
      ? holdClosePnl / position.hold_eligible_invested : null;
  const totalFee = fee + (position.manual_fee ?? 0);
  const netPnl = position.pnl !== null ? roundUsd(position.pnl - totalFee) : null;
  return (
    <tr>
      <td className="trade-ticker">{position.ticker}</td>
      <td className="trade-num">{position.shares}</td>
      <td className="trade-time">{position.buy_time ? formatStamp(position.buy_time, position.day) : "--"}</td>
      <td className="trade-num">{position.buy_price !== null ? formatUsd(position.buy_price) : "--"}</td>
      <td className="trade-time">{position.sell_time ? formatStamp(position.sell_time, position.day) : "--"}</td>
      <td className="trade-num">{position.sell_price !== null ? formatUsd(position.sell_price) : "--"}</td>
      <PnlCell pnl={netPnl} invested={position.invested} />
      <td className="trade-num lot-fees" aria-hidden={!showFees || undefined}>
        {feeAmount(totalFee, showFees)}
      </td>
      <td className="trade-num">
        {holdClosePrice !== null ? formatUsd(holdClosePrice) : "--"}
      </td>
      <td className="trade-num">{formatUsd(position.invested)}</td>
      <td className="trade-num" title={`${position.hold_eligible_shares.toLocaleString()} shares bought before 9:00 AM CT`}>
        {holdClosePnl !== null ? (
          <Diff value={holdClosePnl} pct={holdPct} />
        ) : (
          <span className="muted">{position.hold_eligible_shares === 0 ? "Excluded" : "Awaiting close"}</span>
        )}
      </td>
    </tr>
  );
}

function holdToCloseItem(positions: Position[]): StatItem | null {
  const total = holdToCloseTotal(positions);
  if (total === null) return null;
  const denom = total.typicalInvested;
  return {
    key: "hold",
    label: HOLD_WINDOW_LABEL,
    value: <Diff value={total.pnl} pct={denom ? total.pnl / denom : null} />,
  };
}

function dayValues(
  day: string,
  positions: Position[],
  onBooks: number,
  sessionReturn?: number,
  overnightReturn?: number,
  fees: FeeRecord[] = [],
  showFees = false,
): StatValues {
  const summary = summarizePositions(positions);
  const session = holdToCloseItem(positions);
  const denom = onBooks || summary.invested;
  const tickets = feeSum(fees, [day]) + summary.manualFees;
  const netPnl = roundUsd(summary.pnl - tickets);
  return {
    title: <strong style={{ color: "var(--accent)" }}>{formatDay(day)}</strong>,
    days: `${positions.length} lot${positions.length === 1 ? "" : "s"}`,
    typical: formatUsd(denom),
    pnl: <Diff value={netPnl} pct={denom ? netPnl / denom : null} />,
    fees: feeAmount(tickets, showFees),
    intraday: sessionReturn !== undefined ? <SignedPct value={sessionReturn} /> : "",
    bh: overnightReturn !== undefined ? <SignedPct value={overnightReturn} /> : "",
    hold: session?.value ?? "—",
  };
}

function ClosedDayGroup({
  day,
  positions,
  onBooks,
  sessionReturn,
  overnightReturn,
  expanded,
  onToggle,
  fees = [],
  showFees = false,
}: {
  day: string;
  positions: Position[];
  onBooks: number;
  sessionReturn?: number;
  overnightReturn?: number;
  expanded: boolean;
  onToggle: () => void;
  fees?: FeeRecord[];
  showFees?: boolean;
}) {
  const { lotFees } = assignLotFees(positions, fees.filter((fee) => fee.day === day));
  return (
    <StatExpand
      nested
      open={expanded}
      onToggle={onToggle}
      values={dayValues(day, positions, onBooks, sessionReturn, overnightReturn, fees, showFees)}
    >
      <StatDetail>
        <div className="day-lots">
          <table className="trade-table">
            <StatColGroup columns={CLOSED_LOT_COLUMNS} />
            <thead>
              <tr>
                {CLOSED_LOT_COLUMNS.map((column) => (
                  <th
                    key={column.key}
                    scope="col"
                    data-col={column.key}
                    className={lotColumnClass(column.label)}
                    aria-hidden={(column.key === "fees" && !showFees) || undefined}
                  >
                    {column.key === "fees" && !showFees ? "" : column.label}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {positions.map((position, index) => (
                <ClosedRow
                  position={position}
                  fee={lotFees[index]}
                  showFees={showFees}
                  key={`${position.ticker}-${position.day}-${index}`}
                />
              ))}
            </tbody>
          </table>
        </div>
      </StatDetail>
    </StatExpand>
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
  onBooksByDay: Record<string, number> = {},
): {
  lots: Position[];
  sessions: number;
  typicalOn: number;
  onSum: number;
  pnl: number;
  spyPct: number | null;
} {
  const pnl = lots.reduce((sum, p) => sum + (p.pnl ?? 0), 0);
  const byDay = new Map<string, Position[]>();
  for (const lot of lots) {
    const group = byDay.get(lot.day) ?? [];
    group.push(lot);
    byDay.set(lot.day, group);
  }
  let onSum = 0;
  let spyOn = 0;
  let spySessionDollars = 0;
  for (const [day, dayLots] of byDay) {
    const stacked = dayLots.reduce((sum, p) => sum + p.invested, 0);
    const on = onBooksByDay[day] || stacked;
    onSum += on;
    const spy = spySessionByDay[day];
    if (spy !== undefined) {
      spyOn += on;
      spySessionDollars += on * spy;
    }
  }
  const sessions = byDay.size;
  return {
    lots,
    sessions,
    typicalOn: sessions ? onSum / sessions : 0,
    onSum,
    pnl,
    spyPct: spyOn ? spySessionDollars / spyOn : null,
  };
}

type PeriodSummary = {
  lots: Position[];
  sessions: number;
  typicalOn: number;
  onSum: number;
  pnl: number;
  spyPct: number | null;
};

function periodValues(
  label: string,
  summary: PeriodSummary,
  holdPct: number | null,
  tickets = 0,
  showFees = false,
): StatValues {
  const pctDenom = summary.typicalOn;
  const hold = holdToCloseItem(summary.lots);
  const manualFees = summary.lots.reduce((sum, lot) => sum + (lot.manual_fee ?? 0), 0);
  const totalFees = tickets + manualFees;
  const netPnl = roundUsd(summary.pnl - totalFees);
  return {
    title: <strong>{label}</strong>,
    days: `${summary.sessions}d`,
    typical: `${formatUsd(summary.typicalOn)}/day`,
    pnl: <Diff value={netPnl} pct={pctDenom ? netPnl / pctDenom : null} />,
    fees: feeAmount(totalFees, showFees),
    intraday: summary.spyPct !== null ? <SignedPct value={summary.spyPct} /> : "—",
    bh: holdPct !== null ? <SignedPct value={holdPct} /> : "—",
    hold: hold?.value ?? "—",
  };
}

export default function TradeHistory({ onOpenOvernight }: { onOpenOvernight: (ticker: string) => void }) {
  const [refreshCount, setRefreshCount] = useState(0);
  const [openNowExpanded, setOpenNowExpanded] = useState(true);
  const [expandedDays, setExpandedDays] = useState<Set<string>>(() => new Set());
  const [weekOpen, setWeekOpen] = useState<Record<string, boolean>>({});
  const [showFees, setShowFees] = useState(false);
  const { data, error } = useFetchData<PositionsResponse>(fetchPositions, {
    deps: [refreshCount],
    intervalMs: REFRESH_INTERVAL_MS,
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
    { deps: [days.join(",")], intervalMs: REFRESH_INTERVAL_MS },
  );
  const { data: feesData } = useFetchData<FeesResponse>(fetchFees, {
    deps: [refreshCount],
    intervalMs: REFRESH_INTERVAL_MS,
  });

  if (error) return <p className="error">{error}</p>;
  if (!data) return <p className="muted">Loading trade history...</p>;

  const nothing = openPositions.length === 0 && closedLots.length === 0 && unmatchedSells.length === 0;
  const now = new Date();
  const spySessions = benchmarkData?.returns ?? {};
  const overnight = benchmarkData?.overnight ?? {};
  const monthLots = closedLots.filter((p) => inCalendarMonth(p.day, now));
  const yearLots = closedLots.filter((p) => inCalendarYear(p.day, now));
  const onBooks = data.peak_working ?? {};
  const month = windowSummary(monthLots, spySessions, onBooks);
  const year = windowSummary(yearLots, spySessions, onBooks);
  const allTime = windowSummary(closedLots, spySessions, onBooks);
  const allHold = benchmarkData?.hold?.pct ?? null;
  const fees = feesData?.fees ?? [];

  return (
    <div className="trade-history">
      {nothing ? (
        <p className="muted">No trades logged yet.</p>
      ) : (
        <>
          <details
            className="view-card"
            open={openNowExpanded}
            onToggle={(event) => {
              setOpenNowExpanded((event.currentTarget as HTMLDetailsElement).open);
            }}
          >
            <summary>
              <StatStrip items={openNowItems(openPositions.length, summarizePositions(openPositions))} />
            </summary>
            {openPositions.length === 0 ? (
              <p className="muted" style={{marginTop: "var(--space-3)" }}>
                Nothing left open.
              </p>
            ) : (
              <div style={{ overflowX: "auto",marginTop: "var(--space-3)" }}>
                <table className="trade-table">
                  <thead>
                    <tr>
                      {OPEN_LOT_COLUMNS.map((column) => (
                        <th key={column} className={lotColumnClass(column)}>
                          {column}
                        </th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {openPositions.map((position) => (
                      <OpenRow position={position} onOpenOvernight={onOpenOvernight} key={`${position.ticker}-open`} />
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </details>

          <div
            className="view-card trade-history-scroll"
            role="region"
            aria-label="Closed trade history"
            tabIndex={0}
          >
          <StatTable
            layoutColumns={CLOSED_LOT_COLUMNS}
            columns={periodColumns(
              <button
                type="button"
                className="fees-toggle"
                aria-expanded={showFees}
                aria-label={showFees ? "Hide fees" : "Show fees"}
                onClick={(event) => {
                  event.stopPropagation();
                  setShowFees((open) => !open);
                }}
              >
                Fees
              </button>,
            )}
            className={`trade-history-table ${showFees ? "is-fees-open" : "is-fees-closed"}`}
          >
            <StatHead />
            <StatBody>
              <StatRow
                values={periodValues(
                  "MTD",
                  month,
                  holdPctForDays(monthLots.map((p) => p.day), overnight) ?? allHold,
                  feeSum(fees, monthLots.map((p) => p.day)),
                  showFees,
                )}
              />
              <StatRow
                values={periodValues(
                  "YTD",
                  year,
                  holdPctForDays(yearLots.map((p) => p.day), overnight) ?? allHold,
                  feeSum(fees, yearLots.map((p) => p.day)),
                  showFees,
                )}
              />
              <StatRow
                values={periodValues(
                  "All",
                  allTime,
                  holdPctForDays(closedLots.map((p) => p.day), overnight) ?? allHold,
                  feeSum(fees, closedLots.map((p) => p.day)),
                  showFees,
                )}
              />
            </StatBody>
            {groupClosedByWeek(dayGroups).map(([monday, weekDays], weekIndex) => {
            const lotsThisWeek = weekDays.flatMap(([, lots]) => lots);
            const weekSummary = windowSummary(lotsThisWeek, spySessions, onBooks);
            const weekHold = holdPctForDays(
              lotsThisWeek.map((p) => p.day),
              overnight,
            );
            const isOpen = monday in weekOpen ? weekOpen[monday] : weekIndex === 0;
            return (
              <StatExpand
                key={monday}
                open={isOpen}
                onToggle={() => setWeekOpen((current) => ({ ...current, [monday]: !isOpen }))}
                values={periodValues(
                  formatWeekRange(monday),
                  weekSummary,
                  weekHold,
                  feeSum(fees, lotsThisWeek.map((p) => p.day)),
                  showFees,
                )}
              >
                {weekDays.map(([day, dayPositions]) => (
                  <ClosedDayGroup
                    day={day}
                    positions={dayPositions}
                    onBooks={onBooks[day] ?? 0}
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
                    fees={fees}
                    showFees={showFees}
                    key={day}
                  />
                ))}
              </StatExpand>
            );
         })}
          </StatTable>
          </div>

          {unmatchedSells.length > 0 && (
            <p className="muted" style={{marginTop: "var(--space-3)" }}>
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
