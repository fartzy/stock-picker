import { useState } from "react";
import { fetchBuySignal, type BuySignalResponse } from "../api";
import { formatUsd } from "../format";

// Percent, not fraction -- shown as a plain "%" input; the backend's own
// default (0.005) matches this same 0.5% the Models tab's threshold sweep
// already defaults to.
const DEFAULT_THRESHOLD_PCT = 0.5;

// Sentinel the backend uses for skipped[] when no model is trained yet at
// all, rather than "nothing cleared the bar today" -- see buy_signal.py.
// Deliberately not a valid ticker shape, so it can never collide with a
// real skipped ticker (an earlier "ALL" sentinel collided with Allstate's
// actual ticker symbol).
const NO_MODEL_SENTINEL = "";

export default function BuySignal() {
  const [thresholdPct, setThresholdPct] = useState(DEFAULT_THRESHOLD_PCT);
  const [data, setData] = useState<BuySignalResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function handleCheck() {
    setLoading(true);
    setError(null);
    try {
      setData(await fetchBuySignal(thresholdPct / 100));
    } catch (err) {
      setError(String(err));
    } finally {
      setLoading(false);
    }
  }

  const noModel = data?.skipped.some((s) => s.ticker === NO_MODEL_SENTINEL) ?? false;

  return (
    <div>
      <div className="form-row">
        <label className="muted" style={{ display: "flex", alignItems: "center", gap: 6 }}>
          Threshold
          <input
            className="form-input"
            type="number"
            step="0.1"
            min="0"
            value={thresholdPct}
            onChange={(e) => setThresholdPct(Number(e.target.value))}
            style={{ width: 70 }}
          />
          %
        </label>
        <button className="btn-primary" onClick={handleCheck} disabled={loading}>
          {loading ? "Checking..." : "Check this morning's prices"}
        </button>
      </div>

      {error && (
        <p className="error" style={{ marginTop: 8 }}>
          {error}
        </p>
      )}

      {data && !error && noModel && (
        <p className="muted" style={{ marginTop: 12 }}>
          No trained model yet -- train one on the Models tab first.
        </p>
      )}

      {data && !error && !noModel && (
        <div style={{ marginTop: 12 }}>
          {data.top_drivers.length > 0 && (
            <p className="muted">
              Top drivers: {data.top_drivers.map((d) => `${d.feature} (${d.importance.toFixed(1)}%)`).join(", ")}
            </p>
          )}
          {data.signals.length === 0 ? (
            <p className="muted">No tickers cleared the {thresholdPct}% threshold this morning.</p>
          ) : (
            <table className="trade-table">
              <thead>
                <tr>
                  <th>Ticker</th>
                  <th className="trade-num">Predicted return</th>
                  <th className="trade-num">Open price</th>
                </tr>
              </thead>
              <tbody>
                {data.signals.map((signal) => (
                  <tr key={signal.ticker}>
                    <td className="trade-ticker">{signal.ticker}</td>
                    <td className="trade-num">{(signal.predicted_return * 100).toFixed(2)}%</td>
                    <td className="trade-num">{formatUsd(signal.open_price)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
          <p className="muted" style={{ marginTop: 8 }}>
            Scored {data.scored_count} of {data.scored_count + data.skipped.length} tickers ·{" "}
            <span title={data.skipped.map((s) => `${s.ticker}: ${s.reason}`).join("\n")}>
              {data.skipped.length} skipped
            </span>
          </p>
        </div>
      )}
    </div>
  );
}
