import { useEffect, useState, type FormEvent } from "react";
import {
  fetchOvernightModel, fetchPositions, forecastOvernight,
  type OvernightForecastCase, type OvernightForecastResponse, type OvernightModelResponse,
  type PositionsResponse,
} from "../api";
import { formatUsd } from "../format";
import { OVERNIGHT_FEATURE_DESCRIPTIONS, OVERNIGHT_FEATURE_GROUPS } from "../overnightFeatures";
import { useFetchData } from "../useFetchData";

function percentage(value: number | null): string {
  return value === null ? "—" : `${(value * 100).toFixed(2)}%`;
}

function CaseSummary({ value }: { value: OvernightForecastCase }) {
  return (
    <div className="overnight-case">
      <span>If the close is <strong>{formatUsd(value.assumed_close)}</strong></span>
      <span>Estimated next open <strong>{formatUsd(value.projected_open)}</strong></span>
      <span>{percentage(value.predicted_gap)} gap</span>
    </div>
  );
}

function FeatureSnapshot({ value, columns }: { value: OvernightForecastCase; columns: string[] }) {
  const groups = [...new Set(columns.map((name) => OVERNIGHT_FEATURE_GROUPS[name] ?? "Other"))];
  return (
    <details className="view-card overnight-feature-snapshot">
      <summary>Inputs used for this assumed close · {columns.length} features</summary>
      <p className="muted">Values are model inputs, not explanations of cause. Prior-session values stop before today.</p>
      {groups.map((group) => (
        <div key={group}>
          <h4>{group}</h4>
          <div className="overnight-feature-grid">
            {columns.filter((name) => (OVERNIGHT_FEATURE_GROUPS[name] ?? "Other") === group).map((name) => (
              <div className="overnight-feature-row" key={name}>
                <code>{name}</code>
                <span>{value.features[name]?.toPrecision(5) ?? "—"}</span>
                <small>{OVERNIGHT_FEATURE_DESCRIPTIONS[name] ?? "Model input"}</small>
              </div>
            ))}
          </div>
        </div>
      ))}
    </details>
  );
}

function ForecastResult({ result, model }: { result: OvernightForecastResponse; model: OvernightModelResponse }) {
  const primary = result.cases.find((item) => item.label === "primary");
  if (!primary) return <p className="error">Forecast response is missing its primary scenario.</p>;
  const neighbors = result.cases.filter((item) => item.label !== "primary");
  const losesToUnchanged = primary.oof_model_gap_mae !== null && primary.oof_zero_gap_mae !== null
    && primary.oof_model_gap_mae >= primary.oof_zero_gap_mae;
  return (
    <div className="overnight-results" aria-live="polite">
      <div className="overnight-evidence">
        <strong>{losesToUnchanged ? "No measured accuracy edge over unchanged price." : "Historical accuracy comparison"}</strong>
        <span>Held-out gap error: model {percentage(primary.oof_model_gap_mae)} · unchanged price {percentage(primary.oof_zero_gap_mae)} · {result.evaluated_rows} rows.</span>
        <span>Historical 90th-percentile absolute open error at this price: {primary.oof_abs_open_error_p90_at_assumed_price === null ? "not measured" : formatUsd(primary.oof_abs_open_error_p90_at_assumed_price)}. This is not a guaranteed range.</span>
      </div>
      <div className="overnight-forecast-card">
        <div className="overnight-forecast-flow">
          <div><span>Assumed close · {result.session}</span><strong>{formatUsd(primary.assumed_close)}</strong></div>
          <span aria-hidden="true">→</span>
          <div><span>Estimated open · {primary.next_session ?? "next session unknown"}</span><strong>{formatUsd(primary.projected_open)}</strong></div>
        </div>
        <p>Model gap {percentage(primary.predicted_gap)} · gross difference versus selling at the assumed close {formatUsd(primary.difference_per_share)} per share
          {primary.gross_difference_for_shares !== null ? ` (${formatUsd(primary.gross_difference_for_shares)} for ${result.shares} shares)` : ""}.</p>
        <p>{primary.after_cost_difference_per_share === null
          ? "Execution costs not specified; the difference above is before costs."
          : `After your entered exit costs: ${formatUsd(primary.after_cost_difference_per_share)} per share.`}</p>
      </div>
      <div className="overnight-neighbors">
        <p className="muted">Sensitivity only: if today’s close prints here instead. These are not an uncertainty interval.</p>
        {neighbors.map((item) => <CaseSummary key={item.label} value={item} />)}
      </div>
      <p className="muted">Assumes this price is today’s close; remaining intraday movement is not predicted. Last provider trade: {result.last_trade_at ? new Date(result.last_trade_at).toLocaleString() : "unavailable"}. Model labels observed through {result.model_label_observed_on}; feature contract {model.feature_version}.</p>
      <FeatureSnapshot value={primary} columns={model.feature_columns} />
    </div>
  );
}

export default function Overnight({ initialTicker }: { initialTicker?: string }) {
  const { data: model, error: modelError } = useFetchData<OvernightModelResponse>(fetchOvernightModel);
  const { data: positions } = useFetchData<PositionsResponse>(fetchPositions);
  const [ticker, setTicker] = useState(initialTicker ?? "");
  const [assumedClose, setAssumedClose] = useState("");
  const [step, setStep] = useState("");
  const [shares, setShares] = useState("");
  const [exitTodayCost, setExitTodayCost] = useState("");
  const [exitNextCost, setExitNextCost] = useState("");
  const [result, setResult] = useState<OvernightForecastResponse | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    setTicker(initialTicker ?? "");
    setResult(null);
  }, [initialTicker]);

  const open = (positions?.positions ?? []).filter((position) => !position.closed && position.shares > 0);
  const openByTicker = new Map<string, number>();
  for (const position of open) openByTicker.set(position.ticker, (openByTicker.get(position.ticker) ?? 0) + position.shares);

  async function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setError(null);
    setResult(null);
    const price = Number(assumedClose);
    const parsedStep = step.trim() ? Number(step) : undefined;
    const parsedShares = shares.trim() ? Number(shares) : undefined;
    if (!Number.isFinite(price) || price <= 0 || (parsedStep !== undefined && (!Number.isFinite(parsedStep) || parsedStep <= 0 || parsedStep >= price))) {
      setError("Enter a positive assumed close, and a smaller positive sensitivity step.");
      return;
    }
    if (parsedShares !== undefined && (!Number.isFinite(parsedShares) || parsedShares <= 0)) {
      setError("Shares must be a positive number.");
      return;
    }
    if ((exitTodayCost.trim() === "") !== (exitNextCost.trim() === "")) {
      setError("Enter both exit costs, or leave both blank.");
      return;
    }
    if ([exitTodayCost, exitNextCost].some((value) => value.trim() && (!Number.isFinite(Number(value)) || Number(value) < 0))) {
      setError("Exit costs must be nonnegative numbers.");
      return;
    }
    setBusy(true);
    try {
      const forecast = await forecastOvernight({
        ticker: ticker.trim().toUpperCase(), assumed_close: price,
        ...(parsedStep !== undefined ? { step: parsedStep } : {}),
        ...(parsedShares !== undefined ? { shares: parsedShares } : {}),
        ...(exitTodayCost.trim() && exitNextCost.trim() ? {
          exit_today_cost_per_share: Number(exitTodayCost),
          exit_next_open_cost_per_share: Number(exitNextCost),
        } : {}),
      });
      setResult(forecast);
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "Forecast failed");
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="overnight-page">
      <div className="overnight-intro">
        <div><h3>If it closes at $X, where might it open?</h3><p className="muted">A separate close → next-session-open model. It does not predict the rest of today or tell you to hold.</p></div>
        <span className="overnight-badge">Scenario · not an order</span>
      </div>
      {modelError && <p className="error">Could not load overnight model: {modelError}</p>}
      {model && !model.available && <p className="overnight-evidence">No saved overnight forecast model yet. The 15 inputs are defined, but no number will be invented until a tested artifact is selected.</p>}
      {model && model.available && !model.serving_inputs_pinned && <p className="overnight-evidence">This saved model does not include its pinned morning estimators. Forecasting is disabled until it is retrained with that serving bundle.</p>}
      {openByTicker.size > 0 && <div className="overnight-positions"><span className="muted">Open positions</span>{[...openByTicker].map(([name, count]) => (
        <button type="button" key={name} onClick={() => { setTicker(name); setShares(String(count)); setResult(null); }}>{name} · {count} shares</button>
      ))}</div>}
      <form className="overnight-form" onSubmit={submit}>
        <label>Ticker<input className="form-input" value={ticker} onChange={(event) => setTicker(event.target.value.toUpperCase())} placeholder="WERN" required /></label>
        <label>Assumed close<input className="form-input" type="number" inputMode="decimal" step="any" min="0.0001" value={assumedClose} onChange={(event) => setAssumedClose(event.target.value)} placeholder="Enter your price" required /></label>
        <label>Sensitivity step<input className="form-input" type="number" inputMode="decimal" step="any" min="0.0001" value={step} onChange={(event) => setStep(event.target.value)} placeholder="Default 0.5%" /></label>
        <button className="overnight-nickel" type="button" onClick={() => setStep("0.05")}>Use $0.05</button>
        <label>Shares (optional)<input className="form-input" type="number" inputMode="decimal" step="any" min="0.0001" value={shares} onChange={(event) => setShares(event.target.value)} placeholder="Per-share result" /></label>
        <details className="overnight-costs"><summary>Execution costs (optional)</summary><div><label>Sell-at-close cost / share<input className="form-input" type="number" inputMode="decimal" step="any" min="0" value={exitTodayCost} onChange={(event) => setExitTodayCost(event.target.value)} /></label><label>Sell-at-next-open cost / share<input className="form-input" type="number" inputMode="decimal" step="any" min="0" value={exitNextCost} onChange={(event) => setExitNextCost(event.target.value)} /></label></div></details>
        <button className="btn-primary" type="submit" disabled={busy || !model?.available || !model.serving_inputs_pinned}>{busy ? "Checking…" : "Estimate next open"}</button>
      </form>
      <p className="muted">Enter the price you want to treat as today’s close. We do not silently copy an untimestamped “last” quote into this field.</p>
      {error && <p className="error" role="alert">{error}</p>}
      {result && model && <ForecastResult result={result} model={model} />}
    </div>
  );
}
