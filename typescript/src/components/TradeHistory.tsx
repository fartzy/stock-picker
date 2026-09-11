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
const POSITIONS_POLL_INTERVAL_MS = 60_000;

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
  return (
    <tr>
      <td className="trade-ticker">{position.ticker}</td>
      <td className="trade-num">{position.shares}</td>
      <td className="trade-time">{position.buy_time ? formatStamp(position.buy_time, position.day) : "--"}</td>
      <td className="trade-num">{position.buy_price !== null ? formatUsd(position.buy_price) : "--"}</td>
      <td className="trade-time">{position.sell_time ? formatStamp(position.sell_time, position.day) : "--"}</td>
      <td className="trade-num">{position.sell_price !== null ? formatUsd(position.sell_price) : "--"}</td>
      <PnlCell position={position} caption="realized" />
      <td className="trade-num">{formatUsd(position.invested)}</td>
    </tr>
  );
}

const OPEN_COLUMNS = ["Ticker", "Shares", "Bought", "Buy Price", "Last", "P&L", "Invested"];
const CLOSED_COLUMNS = ["Ticker", "Shares", "Bought", "Buy Price", "Sold", "Sell Price", "P&L", "Invested"];

function columnClass(column: string): string | undefined {
  return column === "Ticker" || column === "Bought" || column === "Sold" ? undefined : "trade-num";
}

function BenchmarkNote({ benchmarkReturn }: { benchmarkReturn: number | undefined }) {
  if (benchmarkReturn === undefined) return null;
  const isUp = benchmarkReturn >= 0;
  return (
    <>
      {" "}
      &middot; vs S&amp;P{" "}
      <span className={isUp ? "quote-diff-up" : "quote-diff-down"}>
        {isUp ? "▲" : "▼"} {(Math.abs(benchmarkReturn) * 100).toFixed(2)}%
      </span>
    </>
  );
}

function ClosedDayGroup({
  day,
  positions,
  benchmarkReturn,
  expanded,
  onToggle,
}: {
  day: string;
  positions: Position[];
  benchmarkReturn: number | undefined;
  expanded: boolean;
  onToggle: () => void;
}) {
  const summary = summarizePositions(positions);
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
          <SummaryLine
            label={`${positions.length} closed lot${positions.length === 1 ? "" : "s"}`}
            summary={summary}
          />
          <BenchmarkNote benchmarkReturn={benchmarkReturn} />
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

function isUnmatchedSell(position: Position): boolean {
  return position.closed && position.shares === 0 && position.buy_price === null;
}

export default function TradeHistory() {
  const [refreshCount, setRefreshCount] = useState(0);
  const [openNowExpanded, setOpenNowExpanded] = useState(true);
  const [expandedDays, setExpandedDays] = useState<Set<string>>(() => new Set());
  const { data, error } = useFetchData<PositionsResponse>(fetchPositions, {
    deps: [refreshCount],
    intervalMs: POSITIONS_POLL_INTERVAL_MS,
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

  return (
    <div>
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

          {dayGroups.map(([day, dayPositions]) => (
            <ClosedDayGroup
              day={day}
              positions={dayPositions}
              benchmarkReturn={benchmarkData?.returns[day]}
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
