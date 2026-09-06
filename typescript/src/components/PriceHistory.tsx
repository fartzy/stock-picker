import { useEffect, useRef, useState } from "react";
import { fetchPriceHistory, type PriceHistoryResponse } from "../api";
import { Diff } from "./Diff";
import { formatUsd } from "../format";
import { themeRgb } from "../theme";
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

export default function PriceHistory() {
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
        </>
      )}
    </div>
  );
}
