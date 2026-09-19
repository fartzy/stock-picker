import { useState } from "react";
import {
  fetchPaperBook,
  type PaperBookDay,
  type PaperListStats,
  type PaperPickRow,
} from "../api";
import { useFetchData } from "../useFetchData";

type Kind = "fit" | "rank" | "both";

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

function StatsLine({ stats }: { stats: PaperListStats | undefined }) {
  if (!stats || stats.n_scored === 0) return null;
  const hit = stats.hit_rate !== null ? `${(stats.hit_rate * 100).toFixed(0)}% hit` : null;
  return (
    <>
      {` · ${stats.n_scored} scored · ${stats.wins}W/${stats.losses}L`}
      {hit ? ` · ${hit}` : ""}
      {" · avg "}
      <Pct value={stats.avg} />
      {stats.n_avoid ? ` · avoid ${stats.n_avoid}` : ""}
      {stats.avg_ex_news !== null && stats.avg_ex_news !== undefined && stats.n_avoid ? (
        <>
          {" · ex-news "}
          <Pct value={stats.avg_ex_news} />
        </>
      ) : null}
    </>
  );
}

function Predicted({ value, isRank }: { value: number | null; isRank: boolean }) {
  if (value === null) return <>—</>;
  if (isRank) return <>{value.toFixed(3)}</>;
  return <>{(value * 100).toFixed(2)}%</>;
}

function ListTable({
  title,
  rows,
  stats,
  isRank,
}: {
  title: string;
  rows: PaperPickRow[];
  stats: PaperListStats | undefined;
  isRank: boolean;
}) {
  if (rows.length === 0) {
    return <p className="muted">{title}: no list that morning</p>;
  }
  return (
    <div>
      <p className="view-meta" style={{ marginBottom: "var(--space-2)" }}>
        <strong>{title}</strong>
        {` · ${rows.length} names`}
        <StatsLine stats={stats} />
      </p>
      <div style={{ overflowX: "auto" }}>
        <table className="trade-table">
          <thead>
            <tr>
              <th>#</th>
              <th>Ticker</th>
              <th className="trade-num">{isRank ? "Score" : "Predicted"}</th>
              <th className="trade-num">Open</th>
              <th className="trade-num">Close</th>
              <th className="trade-num">Session</th>
              <th>News</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((row) => (
              <tr key={`${title}-${row.rank}-${row.ticker}`}>
                <td>{row.rank}</td>
                <td className="trade-ticker">{row.ticker}</td>
                <td className="trade-num">
                  <Predicted value={row.predicted} isRank={isRank} />
                </td>
                <td className="trade-num">
                  {row.open_price !== null ? `$${row.open_price.toFixed(2)}` : "—"}
                </td>
                <td className="trade-num">
                  {row.close_price !== null ? `$${row.close_price.toFixed(2)}` : "—"}
                </td>
                <td className="trade-num">
                  <Pct value={row.session_return} />
                </td>
                <td className={row.news_flag ? "quote-diff-down" : "muted"}>
                  {row.news_flag ? `avoid · ${row.news_flag}` : "—"}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function DayCard({ day, kind }: { day: PaperBookDay; kind: Kind }) {
  return (
    <details className="view-card">
      <summary>
        <strong style={{ color: "var(--accent)" }}>{formatDay(day.as_of)}</strong>{" "}
        <span className="view-meta">
          {kind !== "rank" && day.fit.length > 0 && (
            <>
              Fit {day.fit.length} <Pct value={day.fit_avg} />
              <StatsLine stats={day.fit_stats} />
            </>
          )}
          {kind === "both" && day.fit.length > 0 && day.rank.length > 0 && " · "}
          {kind !== "fit" && day.rank.length > 0 && (
            <>
              Rank {day.rank.length} <Pct value={day.rank_avg} />
              <StatsLine stats={day.rank_stats} />
            </>
          )}
        </span>
      </summary>
      <div style={{ display: "grid", gap: "var(--space-4)", marginTop: "var(--space-3)" }}>
        {kind !== "rank" && (
          <ListTable title="Fit" rows={day.fit} stats={day.fit_stats} isRank={false} />
        )}
        {kind !== "fit" && (
          <ListTable title="Rank" rows={day.rank} stats={day.rank_stats} isRank={true} />
        )}
      </div>
    </details>
  );
}

export default function WhatIf() {
  const [kind, setKind] = useState<Kind>("both");
  const [topK, setTopK] = useState<number | undefined>(undefined);
  const { data, error } = useFetchData(() => fetchPaperBook(kind, topK), {
    deps: [kind, topK ?? "all"],
  });

  if (error) return <p className="error">{error}</p>;
  if (!data) return <p className="muted">Loading paper book...</p>;

  return (
    <div>
      <div className="meta-row" style={{ marginBottom: "var(--space-3)" }}>
        <div className="list-toggle" role="group" aria-label="List">
          {(["both", "fit", "rank"] as Kind[]).map((option) => (
            <button
              key={option}
              type="button"
              className={kind === option ? "active" : ""}
              onClick={() => setKind(option)}
            >
              {option === "both" ? "Fit + Rank" : option === "fit" ? "Fit" : "Rank"}
            </button>
          ))}
        </div>
        <label className="meta-label" htmlFor="whatif-topk">
          Top
        </label>
        <select
          id="whatif-topk"
          className="form-select"
          value={topK ?? "all"}
          onChange={(event) => {
            const value = event.target.value;
            setTopK(value === "all" ? undefined : Number(value));
          }}
        >
          <option value="all">all</option>
          {[5, 10, 20, 50].map((n) => (
            <option key={n} value={n}>
              {n}
            </option>
          ))}
        </select>
      </div>
      {data.days.length === 0 ? (
        <p className="muted">No morning lists yet.</p>
      ) : (
        <>
          <div className="view-card">
            {kind !== "rank" && (
              <p className="period-row">
                <strong>Fit</strong>
                {` · ${data.fit_days}d · ${data.fit_stats?.n ?? data.n_picks} names · `}
                <Pct value={data.fit_compound} />
                {" compound"}
                <StatsLine stats={data.fit_stats} />
              </p>
            )}
            {kind !== "fit" && (
              <p className="period-row">
                <strong>Rank</strong>
                {` · ${data.rank_days}d · ${data.rank_stats?.n ?? 0} names · `}
                <Pct value={data.rank_compound} />
                {" compound"}
                <StatsLine stats={data.rank_stats} />
              </p>
            )}
          </div>
          {data.days.map((day) => (
            <DayCard day={day} kind={kind} key={day.as_of} />
          ))}
        </>
      )}
    </div>
  );
}
