import { useMemo, useState } from "react";
import { fetchPositions, type Position, type PositionsResponse } from "../api";
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
    second: "2-digit",
  });
}

function formatDay(day: string): string {
  return new Date(`${day}T12:00:00`).toLocaleDateString("en-US", {
    weekday: "long",
    month: "short",
    day: "numeric",
  });
}

interface PositionsSummary {
  invested: number;
  pnl: number;
  hasUnknownPnl: boolean;
}

// Shared by the grand total and each day-group total -- same aggregation,
// different slice of positions.
function summarizePositions(positions: Position[]): PositionsSummary {
  return {
    invested: positions.reduce((sum, p) => sum + p.invested, 0),
    pnl: positions.reduce((sum, p) => sum + (p.pnl ?? 0), 0),
    hasUnknownPnl: positions.some((p) => p.pnl === null),
  };
}

// One component for every "X vs Y" diff shown in this table (the gap, and
// P&L) -- same arrow/color/format logic each time, just different values.
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

function DayOpenCell({ position }: { position: Position }) {
  return (
    <td className="trade-num">
      <div>{position.day_open !== null ? formatUsd(position.day_open) : "--"}</div>
      {position.gap !== null && (
        <div className="muted">
          vs close <Diff value={position.gap} pct={position.gap_pct} />
        </div>
      )}
    </td>
  );
}

function ExitCell({ position }: { position: Position }) {
  if (position.closed) {
    return (
      <td>
        <div>Sold {formatTime(position.sell_time!)} ET</div>
        <div className="trade-num">{formatUsd(position.sell_price!)}</div>
      </td>
    );
  }
  return (
    <td>
      <div className="muted">Open position</div>
      <div className="trade-num">{position.current_price !== null ? formatUsd(position.current_price) : "--"}</div>
    </td>
  );
}

function PnlCell({ position }: { position: Position }) {
  return (
    <td className="trade-num">
      <div>
        {position.pnl !== null ? (
          <Diff value={position.pnl} pct={position.invested ? position.pnl / position.invested : null} />
        ) : (
          "--"
        )}
      </div>
      <div className="muted">{position.closed ? "realized" : "if sold now"}</div>
    </td>
  );
}

function PositionRow({ position }: { position: Position }) {
  return (
    <tr>
      <td className="trade-ticker">{position.ticker}</td>
      <td className="trade-num">{position.shares}</td>
      <td className="trade-time">
        {position.buy_time !== null ? `${formatTime(position.buy_time)} ET` : "--"}
      </td>
      <td className="trade-num">
        {position.buy_price !== null ? formatUsd(position.buy_price) : "--"}
      </td>
      <DayOpenCell position={position} />
      <ExitCell position={position} />
      <PnlCell position={position} />
      <td className="trade-num">{formatUsd(position.invested)}</td>
    </tr>
  );
}

const POSITION_TABLE_COLUMNS = ["Ticker", "Shares", "Bought At", "Buy Price", "Day Open", "Exit", "P&L", "Invested"];

function DayGroup({ day, positions }: { day: string; positions: Position[] }) {
  const summary = summarizePositions(positions);

  return (
    <details className="view-card" open>
      <summary>
        <strong style={{ color: "var(--accent)" }}>{formatDay(day)}</strong>{" "}
        <span className="view-meta">
          <SummaryLine label={`${positions.length} position${positions.length === 1 ? "" : "s"}`} summary={summary} />
        </span>
      </summary>
      <div style={{ overflowX: "auto", marginTop: "var(--space-3)" }}>
        <table className="trade-table">
          <thead>
            <tr>
              {POSITION_TABLE_COLUMNS.map((column) => (
                <th key={column} className={column === "Ticker" || column === "Bought At" || column === "Exit" ? undefined : "trade-num"}>
                  {column}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {positions.map((position) => (
              <PositionRow position={position} key={`${position.ticker}-${position.day}`} />
            ))}
          </tbody>
        </table>
      </div>
    </details>
  );
}

function groupByDay(positions: Position[]): [string, Position[]][] {
  const byDay = new Map<string, Position[]>();
  for (const position of positions) {
    const group = byDay.get(position.day) ?? [];
    group.push(position);
    byDay.set(position.day, group);
  }
  return [...byDay.entries()].sort(([a], [b]) => (a < b ? 1 : -1));
}

export default function TradeHistory() {
  const [refreshCount, setRefreshCount] = useState(0);
  const { data, error } = useFetchData<PositionsResponse>(fetchPositions, {
    deps: [refreshCount],
    intervalMs: POSITIONS_POLL_INTERVAL_MS,
  });

  const dayGroups = useMemo(() => groupByDay(data?.positions ?? []), [data]);

  if (error) return <p className="error">{error}</p>;
  if (!data) return <p className="muted">Loading trade history...</p>;

  const positions = data.positions;
  const showGrandTotal = dayGroups.length > 1; // a single day's total would just repeat that day-group's own total

  return (
    <div>
      {positions.length > 0 && showGrandTotal && (
        <div className="summary-line">
          <SummaryLine label={`${positions.length} positions`} summary={summarizePositions(positions)} />
        </div>
      )}
      {positions.length === 0 ? (
        <p className="muted">No trades logged yet.</p>
      ) : (
        dayGroups.map(([day, dayPositions]) => <DayGroup day={day} positions={dayPositions} key={day} />)
      )}
      <AddTradeForm onAdded={() => setRefreshCount((c) => c + 1)} />
    </div>
  );
}
