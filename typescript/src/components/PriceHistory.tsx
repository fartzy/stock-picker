import { useEffect, useMemo, useRef, useState } from "react";
import {
  fetchFeatureValues,
  fetchPriceHistory,
  fetchRegistry,
  type FeatureValuesResponse,
  type PriceHistoryResponse,
  type RegistryResponse,
} from "../api";
import { Diff } from "./Diff";
import { formatUsd } from "../format";
import { signedMagnitudeColor, themeRgb } from "../theme";
import { useFetchData } from "../useFetchData";

const CHART_HEIGHT_PX = 320;
const TABLE_MAX_HEIGHT_PX = 400;
const MARGIN = { top: 16, right: 16, bottom: 28, left: 64 };
const Y_AXIS_TICKS = 5;
const X_AXIS_TICKS = 6;
const DEFAULT_TICKER = "AAPL";

type Interval = "daily" | "hourly";

function formatAxisDate(isoString: string, interval: Interval): string {
  const d = new Date(isoString);
  return interval === "daily"
    ? d.toLocaleDateString(undefined, { month: "short", day: "numeric" })
    : d.toLocaleString(undefined, { month: "short", day: "numeric", hour: "numeric" });
}

function formatRowDate(isoString: string, interval: Interval): string {
  const d = new Date(isoString);
  return interval === "daily"
    ? d.toLocaleDateString(undefined, { year: "numeric", month: "short", day: "numeric" })
    : d.toLocaleString(undefined, {
        year: "numeric",
        month: "short",
        day: "numeric",
        hour: "numeric",
        minute: "2-digit",
      });
}

function drawLineChart(canvas: HTMLCanvasElement, prices: PriceHistoryResponse["prices"], interval: Interval) {
  const dpr = window.devicePixelRatio || 1;
  const width = canvas.clientWidth;
  const height = CHART_HEIGHT_PX;
  canvas.width = width * dpr;
  canvas.height = height * dpr;
  canvas.style.height = `${height}px`;
  const ctx = canvas.getContext("2d")!;
  ctx.scale(dpr, dpr);
  ctx.clearRect(0, 0, width, height);

  if (prices.length === 0) return;

  const closes = prices.map((p) => p.close);
  const min = Math.min(...closes);
  const max = Math.max(...closes);
  const range = max - min || 1;
  const plotWidth = width - MARGIN.left - MARGIN.right;
  const plotHeight = height - MARGIN.top - MARGIN.bottom;

  const xFor = (i: number) => MARGIN.left + (i / (closes.length - 1 || 1)) * plotWidth;
  const yFor = (v: number) => MARGIN.top + plotHeight - ((v - min) / range) * plotHeight;

  const [lineR, lineG, lineB] = themeRgb("line");
  const [mutedR, mutedG, mutedB] = themeRgb("textMuted");
  const gridColor = `rgb(${lineR},${lineG},${lineB})`;
  const labelColor = `rgb(${mutedR},${mutedG},${mutedB})`;
  ctx.font = "11px ui-monospace, monospace";
  ctx.strokeStyle = gridColor;
  ctx.fillStyle = labelColor;
  ctx.lineWidth = 1;

  // Y-axis: price gridlines + labels.
  for (let i = 0; i < Y_AXIS_TICKS; i++) {
    const value = min + (range * i) / (Y_AXIS_TICKS - 1);
    const y = yFor(value);
    ctx.beginPath();
    ctx.moveTo(MARGIN.left, y);
    ctx.lineTo(width - MARGIN.right, y);
    ctx.stroke();
    ctx.textAlign = "right";
    ctx.textBaseline = "middle";
    ctx.fillText(`$${value.toFixed(2)}`, MARGIN.left - 8, y);
  }

  // X-axis: date labels. First/last are edge-aligned so they don't clip off
  // the canvas; everything in between is centered on its tick.
  ctx.textBaseline = "top";
  for (let i = 0; i < X_AXIS_TICKS; i++) {
    const index = Math.round((i / (X_AXIS_TICKS - 1)) * (prices.length - 1));
    const x = xFor(index);
    ctx.textAlign = i === 0 ? "left" : i === X_AXIS_TICKS - 1 ? "right" : "center";
    ctx.fillText(formatAxisDate(prices[index].date, interval), x, height - MARGIN.bottom + 8);
  }

  const [r, g, b] = themeRgb("accent");
  ctx.strokeStyle = `rgb(${r},${g},${b})`;
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  closes.forEach((close, i) => {
    const x = xFor(i);
    const y = yFor(close);
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  });
  ctx.stroke();
}

function formatFeatureValue(value: number | string | null): string {
  if (value === null) return "--";
  if (typeof value !== "number") return value;
  // Returns/spreads dominate the ~94 columns and live well under 1 -- 4
  // decimals there; RSI/stochastic-scale features run 0-100, where the same
  // 4 decimals is just noise, so 2 is enough to still show real precision.
  return Math.abs(value) < 1 ? value.toFixed(4) : value.toFixed(2);
}

function useFeatureToGroup(): Record<string, string> {
  const { data: registry } = useFetchData<RegistryResponse>(fetchRegistry);
  return useMemo(() => {
    const map: Record<string, string> = {};
    for (const view of registry?.feature_views ?? []) {
      for (const feature of view.features) map[feature] = view.name;
    }
    return map;
  }, [registry]);
}

function FeatureValuesTable({
  ticker,
  interval,
  onNavigateToFeature,
}: {
  ticker: string;
  interval: Interval;
  onNavigateToFeature: (feature: string) => void;
}) {
  const [filter, setFilter] = useState("");
  // Feature tables are computed from daily price history only -- there's no
  // hourly path (see pipeline.py's build_features), so this doesn't depend
  // on the OHLCV table's interval toggle at all.
  const { data: fetched, error: rawError } = useFetchData<FeatureValuesResponse>(
    () =>
      fetchFeatureValues(ticker).catch((err) => {
        throw new Error(`${ticker}::${err.message}`);
      }),
    { deps: [ticker] },
  );
  const data = fetched && fetched.ticker === ticker ? fetched : null;
  const error = rawError?.startsWith(`Error: ${ticker}::`) ? rawError : null;
  const isNotFound = error?.includes("404") ?? false;
  const featureToGroup = useFeatureToGroup();

  // Each column's own |max| -- shading is column-relative (RSI's 0-100 range
  // and a return's ~0.02 range aren't comparable on one shared scale), and
  // this data is already fully loaded client-side, so a single pass over it
  // when it arrives is cheap (~250 rows x ~94 columns).
  const columnMaxAbs = useMemo(() => {
    const maxAbs: Record<string, number> = {};
    for (const row of data?.rows ?? []) {
      for (const col of data?.columns ?? []) {
        const value = row[col];
        if (typeof value === "number") {
          maxAbs[col] = Math.max(maxAbs[col] ?? 0, Math.abs(value));
        }
      }
    }
    return maxAbs;
  }, [data]);

  if (interval === "hourly") {
    return (
      <div className="view-card">
        <h3>Derived features</h3>
        <p className="muted">Derived features are computed from daily history only.</p>
      </div>
    );
  }

  if (isNotFound) {
    return (
      <div className="view-card">
        <h3>Derived features</h3>
        <p className="muted">No derived features for {ticker}.</p>
      </div>
    );
  }
  if (error) return <p className="error">{error}</p>;
  if (!data) return <p className="muted">Loading derived features...</p>;

  const columns = data.columns.filter((c) => c.toLowerCase().includes(filter.toLowerCase()));
  // build_features() (pipeline.py) concatenates one feature-category
  // DataFrame at a time in the same order Registry names its feature views,
  // so `columns` already arrives grouped -- no reordering needed, just a
  // boundary flag wherever the group changes from the previous *visible*
  // column, so a text-filtered-out column doesn't leave a phantom divider.
  let previousGroup: string | undefined;
  const columnsWithGroups = columns.map((c) => {
    const group = featureToGroup[c];
    const isGroupStart = group !== undefined && group !== previousGroup;
    previousGroup = group;
    return { name: c, isGroupStart };
  });

  return (
    <div className="view-card">
      <h3>Derived features ({data.columns.length})</h3>
      <input
        className="form-input"
        value={filter}
        onChange={(e) => setFilter(e.target.value)}
        placeholder={`Filter ${data.columns.length} columns (e.g. "rsi")`}
        style={{ marginBottom: "var(--space-3)", width: "100%" }}
      />
      {/* Horizontal scroll is deliberate, not a layout bug -- with ~90+
          computed columns, a container that visibly scrolls both ways is the
          signal that this table holds a lot of derived data, same spirit as
          the filter box above it. The Date column and header row stay
          pinned via .trade-table-wide's sticky rules so scrolling never
          loses the row you're looking at. */}
      <div style={{ overflow: "auto", maxHeight: TABLE_MAX_HEIGHT_PX }}>
        <table className="trade-table trade-table-wide">
          <thead>
            <tr>
              <th>Date</th>
              {columnsWithGroups.map(({ name, isGroupStart }) => (
                <th
                  key={name}
                  className={isGroupStart ? "col-group-start" : undefined}
                  title={`${featureToGroup[name] ?? ""}: view in Feature Store`}
                >
                  <button type="button" className="feature-header-link" onClick={() => onNavigateToFeature(name)}>
                    {name}
                  </button>
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {[...data.rows].reverse().map((row) => (
              <tr key={row.date as string}>
                <td>{formatRowDate(row.date as string, "daily")}</td>
                {columnsWithGroups.map(({ name, isGroupStart }) => {
                  const value = row[name];
                  const background =
                    typeof value === "number" ? signedMagnitudeColor(value, columnMaxAbs[name] ?? 0) : undefined;
                  return (
                    <td
                      className={`trade-num${isGroupStart ? " col-group-start" : ""}`}
                      key={name}
                      style={{ background }}
                    >
                      {formatFeatureValue(value)}
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

export default function PriceHistory({
  onNavigateToFeature,
}: {
  onNavigateToFeature: (feature: string) => void;
}) {
  const [inputValue, setInputValue] = useState(DEFAULT_TICKER);
  const [ticker, setTicker] = useState(DEFAULT_TICKER);
  const [interval, setInterval_] = useState<Interval>("daily");
  const canvasRef = useRef<HTMLCanvasElement>(null);

  const requestKey = `${ticker}:${interval}`;
  const { data: fetched, error: rawError } = useFetchData<PriceHistoryResponse>(
    // useFetchData doesn't clear stale data/error on a new fetch -- tag the
    // rejection with the request it belongs to so a stale error from a
    // previous ticker/interval doesn't linger after this one resolves.
    () => fetchPriceHistory(ticker, interval).catch((err) => {
      throw new Error(`${requestKey}::${err.message}`);
    }),
    { deps: [ticker, interval] },
  );
  const data = fetched && fetched.ticker === ticker && fetched.interval === interval ? fetched : null;
  const error = rawError?.startsWith(`Error: ${requestKey}::`) ? rawError : null;

  // getJson's error string is "<path> failed: <status>" -- a 404 here just
  // means "not a tracked ticker for this interval," an expected everyday
  // case, not a real error worth an error banner.
  const isNotFound = error?.includes("404") ?? false;

  useEffect(() => {
    if (data && canvasRef.current) drawLineChart(canvasRef.current, data.prices, data.interval);
  }, [data]);

  function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    setTicker(inputValue.trim().toUpperCase());
  }

  return (
    <div>
      <form className="price-history-controls" onSubmit={handleSubmit}>
        <input
          className="form-input"
          value={inputValue}
          onChange={(e) => setInputValue(e.target.value)}
          placeholder="Ticker"
        />
        <button className="btn-primary" type="submit">
          Load
        </button>
        <div className="interval-toggle" style={{ marginLeft: "var(--space-3)" }}>
          {(["daily", "hourly"] as const).map((i) => (
            <button
              key={i}
              type="button"
              className={interval === i ? "active" : ""}
              onClick={() => setInterval_(i)}
            >
              {i === "daily" ? "Daily" : "Hourly"}
            </button>
          ))}
        </div>
      </form>

      {isNotFound && (
        <p className="muted">
          No {interval} price history for {ticker}.{" "}
          {interval === "daily" && "Not in the tracked universe."}
        </p>
      )}
      {error && !isNotFound && <p className="error">{error}</p>}
      {!data && !error && <p className="muted">Loading...</p>}
      {data && data.prices.length > 0 && (
        <>
          <h3>Price history (source data)</h3>
          {(() => {
            const latest = data.prices[data.prices.length - 1];
            const previous = data.prices[data.prices.length - 2];
            return (
              <div className="view-card price-hero">
                <div>
                  <div className="price-hero-ticker">{data.ticker}</div>
                  <div className="price-hero-value">{formatUsd(latest.close)}</div>
                </div>
                {previous && (
                  <Diff value={latest.close - previous.close} pct={(latest.close - previous.close) / previous.close} />
                )}
              </div>
            );
          })()}

          <div className="view-card">
            <canvas ref={canvasRef} style={{ width: "100%", display: "block" }} />
            <p className="muted" style={{ marginTop: "var(--space-2)" }}>
              {data.prices.length} {interval === "daily" ? "days" : "hours"} &middot;{" "}
              {formatRowDate(data.prices[0].date, interval)} to{" "}
              {formatRowDate(data.prices[data.prices.length - 1].date, interval)}
            </p>
          </div>

          <div className="view-card">
            <div style={{ overflowY: "auto", maxHeight: TABLE_MAX_HEIGHT_PX }}>
              <table className="trade-table">
                <thead>
                  <tr>
                    <th>Date</th>
                    <th>Open</th>
                    <th>High</th>
                    <th>Low</th>
                    <th>Close</th>
                    <th>Volume</th>
                  </tr>
                </thead>
                <tbody>
                  {[...data.prices].reverse().map((p) => (
                    <tr key={p.date}>
                      <td>{formatRowDate(p.date, interval)}</td>
                      <td className="trade-num">{formatUsd(p.open)}</td>
                      <td className="trade-num">{formatUsd(p.high)}</td>
                      <td className="trade-num">{formatUsd(p.low)}</td>
                      <td className="trade-num">{formatUsd(p.close)}</td>
                      <td className="trade-num">{Math.round(p.volume).toLocaleString()}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

          <FeatureValuesTable ticker={data.ticker} interval={interval} onNavigateToFeature={onNavigateToFeature} />
        </>
      )}
    </div>
  );
}
