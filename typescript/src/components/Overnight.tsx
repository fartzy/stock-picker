import { useEffect, useRef, useState, type FormEvent } from "react";
import {
  fetchOvernightCurrentPrice, fetchOvernightModel, fetchPositions, forecastOvernight,
  type OvernightForecastCase, type OvernightForecastResponse, type OvernightModelResponse,
  type OvernightCurrentPriceResponse, type PositionsResponse,
} from "../api";
import { formatUsd } from "../format";
import { OVERNIGHT_FEATURE_DESCRIPTIONS, OVERNIGHT_FEATURE_GROUPS } from "../overnightFeatures";
import { expireQuotePrefill, isFreshCashSessionQuote, isWithinQuoteSession, type AssumedCloseInput } from "../session";
import { useFetchData } from "../useFetchData";
import OvernightMorningPicks from "./OvernightMorningPicks";

function percentage(value: number | null): string {
  return value === null ? "—" : `${(value * 100).toFixed(2)}%`;
}

function quotePrice(value: number): string {
  return `$${value.toLocaleString("en-US", { minimumFractionDigits: 2, maximumFractionDigits: 6 })}`;
}

const easternTradeTime = new Intl.DateTimeFormat("en-US", {
  timeZone: "America/New_York", year: "numeric", month: "short", day: "numeric",
  hour: "numeric", minute: "2-digit", second: "2-digit", timeZoneName: "short",
});

function isFreshQuote(quote: OvernightCurrentPriceResponse, now: number): boolean {
  return isFreshCashSessionQuote(quote, now);
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
      <p className="muted">Assumes this price is today’s close; remaining intraday movement is not predicted. Last provider trade: {result.last_trade_at ? new Date(result.last_trade_at).toLocaleString() : "unavailable"}. Latest next open observed for training: {result.model_label_observed_on}. This model uses {model.feature_columns.length} inputs.</p>
      <FeatureSnapshot value={primary} columns={model.feature_columns} />
    </div>
  );
}

export default function Overnight({ initialTicker, compact = false }: { initialTicker?: string; compact?: boolean }) {
  const { data: model, error: modelError } = useFetchData<OvernightModelResponse>(fetchOvernightModel);
  const { data: positions } = useFetchData<PositionsResponse>(fetchPositions);
  const [ticker, setTicker] = useState(initialTicker ?? "");
  const [quoteTicker, setQuoteTicker] = useState(initialTicker ?? "");
  const [quoteRefresh, setQuoteRefresh] = useState(0);
  const [quote, setQuote] = useState<OvernightCurrentPriceResponse | null>(null);
  const [quoteBusy, setQuoteBusy] = useState(false);
  const [quoteError, setQuoteError] = useState<string | null>(null);
  const [now, setNow] = useState(() => Date.now());
  const [assumedCloseInput, setAssumedCloseInput] = useState<AssumedCloseInput>({ value: "", quote: null });
  const assumedClose = assumedCloseInput.value;
  const [step, setStep] = useState("");
  const [shares, setShares] = useState("");
  const [exitTodayCost, setExitTodayCost] = useState("");
  const [exitNextCost, setExitNextCost] = useState("");
  const [result, setResult] = useState<OvernightForecastResponse | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const priceInputRef = useRef<HTMLInputElement>(null);
  const priceEditedRef = useRef(false);
  const assumedCloseInputRef = useRef(assumedCloseInput);
  const quoteRequestRef = useRef(0);
  const forecastRequestRef = useRef(0);

  function setCloseInput(input: AssumedCloseInput) {
    assumedCloseInputRef.current = input;
    setAssumedCloseInput(input);
  }

  function invalidateForecast() {
    forecastRequestRef.current += 1;
    setResult(null);
    setError(null);
    setBusy(false);
  }

  function expireAutofilledQuote(timestamp: number) {
    const current = assumedCloseInputRef.current;
    const next = expireQuotePrefill(current, timestamp);
    if (next === current) return;
    setCloseInput(next);
    invalidateForecast();
  }

  useEffect(() => {
    quoteRequestRef.current += 1;
    setTicker(initialTicker ?? "");
    setQuoteTicker(initialTicker ?? "");
    setCloseInput({ value: "", quote: null });
    priceEditedRef.current = false;
    setQuote(null);
    invalidateForecast();
  }, [initialTicker]);

  useEffect(() => () => {
    quoteRequestRef.current += 1;
    forecastRequestRef.current += 1;
  }, []);

  useEffect(() => {
    const activeQuote = assumedCloseInput.quote ?? quote;
    if (!activeQuote) return;
    const tick = () => {
      const timestamp = Date.now();
      setNow(timestamp);
      expireAutofilledQuote(timestamp);
      if (!isFreshCashSessionQuote(activeQuote, timestamp)) window.clearInterval(interval);
    };
    const interval = window.setInterval(() => {
      tick();
    }, 1_000);
    window.addEventListener("focus", tick);
    document.addEventListener("visibilitychange", tick);
    tick();
    return () => {
      window.clearInterval(interval);
      window.removeEventListener("focus", tick);
      document.removeEventListener("visibilitychange", tick);
    };
  }, [quote, assumedCloseInput.quote]);

  useEffect(() => {
    if (!quoteTicker) {
      setQuoteBusy(false);
      setQuote(null);
      setQuoteError(null);
      return;
    }
    let cancelled = false;
    const request = ++quoteRequestRef.current;
    setQuote(null);
    setQuoteError(null);
    setQuoteBusy(true);
    fetchOvernightCurrentPrice(quoteTicker)
      .then((current) => {
        if (cancelled || request !== quoteRequestRef.current) return;
        setQuote(current);
        setNow(Date.now());
        if (!priceEditedRef.current && current.ticker === quoteTicker.trim().toUpperCase() && isFreshQuote(current, Date.now())) {
          setCloseInput({ value: String(current.price), quote: current });
          invalidateForecast();
        }
      })
      .catch((cause) => {
        if (cancelled || request !== quoteRequestRef.current) return;
        setQuoteError(cause instanceof Error ? cause.message : "Current price unavailable");
      })
      .finally(() => {
        if (!cancelled && request === quoteRequestRef.current) setQuoteBusy(false);
      });
    return () => { cancelled = true; };
  }, [quoteTicker, quoteRefresh]);

  const open = (positions?.positions ?? []).filter((position) => !position.closed && position.shares > 0);
  const openByTicker = new Map<string, number>();
  for (const position of open) openByTicker.set(position.ticker, (openByTicker.get(position.ticker) ?? 0) + position.shares);
  const quoteIsFresh = quote !== null && quote.ticker === ticker.trim().toUpperCase() && isFreshQuote(quote, now);
  const quoteStatus = quoteIsFresh
    ? "Fresh for use as an editable assumption"
    : quote !== null && isWithinQuoteSession(quote, now)
      ? "Stale quote; refresh before using this price"
      : "Outside regular cash hours; enter a price manually";

  function refreshQuote() {
    quoteRequestRef.current += 1;
    setQuote(null);
    setQuoteError(null);
    setQuoteTicker(ticker.trim().toUpperCase());
    setQuoteRefresh((value) => value + 1);
  }

  function useCurrentQuote() {
    if (!quote || !isFreshQuote(quote, Date.now()) || quote.ticker !== ticker.trim().toUpperCase()) {
      setNow(Date.now());
      return;
    }
    priceEditedRef.current = false;
    setCloseInput({ value: String(quote.price), quote });
    invalidateForecast();
  }

  function selectTicker(name: string) {
    if (name !== ticker) {
      setTicker(name);
      setCloseInput({ value: "", quote: null });
      priceEditedRef.current = false;
      setShares(openByTicker.has(name) ? String(openByTicker.get(name)) : "");
      invalidateForecast();
    }
    quoteRequestRef.current += 1;
    setQuote(null);
    setQuoteError(null);
    setQuoteTicker(name);
    setQuoteRefresh((value) => value + 1);
    priceInputRef.current?.focus();
  }

  async function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const request = ++forecastRequestRef.current;
    setError(null);
    setResult(null);
    setBusy(false);
    const quotePrefill = assumedCloseInputRef.current.quote;
    if (quotePrefill && !isFreshCashSessionQuote(quotePrefill, Date.now())) {
      expireAutofilledQuote(Date.now());
      setError("Current quote expired; enter an assumed close manually.");
      return;
    }
    const price = Number(assumedCloseInputRef.current.value);
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
      if (request === forecastRequestRef.current) setResult(forecast);
    } catch (cause) {
      if (request === forecastRequestRef.current) setError(cause instanceof Error ? cause.message : "Forecast failed");
    } finally {
      if (request === forecastRequestRef.current) setBusy(false);
    }
  }

  return (
    <div className={`overnight-page ${compact ? "is-quick" : ""}`}>
      {!compact && <>
        <div className="overnight-intro">
          <div><h3>Close → next open</h3><p className="muted">Choose a ticker and a possible closing price.</p></div>
          <span className="overnight-badge">Scenario only</span>
        </div>
        <OvernightMorningPicks selectedTicker={ticker} onSelect={selectTicker} />
      </>}
      {modelError && <p className="error">Could not load overnight model: {modelError}</p>}
      {model && !model.available && <p className="overnight-unavailable">Forecast unavailable — no saved overnight model yet.</p>}
      {model && model.available && !model.serving_inputs_pinned && <p className="overnight-unavailable">Forecast unavailable — this model needs its saved morning inputs.</p>}
      {!compact && openByTicker.size > 0 && <div className="overnight-positions"><span className="muted">Open positions</span>{[...openByTicker].map(([name, count]) => (
        <button type="button" key={name} onClick={() => selectTicker(name)}>{name} · {count} shares</button>
      ))}</div>}
      <form className="overnight-form" onSubmit={submit}>
        <div className="overnight-form-primary">
          <label>Ticker<input className="form-input" value={ticker} onChange={(event) => {
            quoteRequestRef.current += 1;
            setTicker(event.target.value.toUpperCase()); setQuoteTicker(""); setQuote(null); setQuoteError(null);
            setCloseInput({ value: "", quote: null }); priceEditedRef.current = false; invalidateForecast();
          }} placeholder="Ticker" required /></label>
          <div className="overnight-price-field">
            <div className="overnight-price-head"><span>Assumed close</span><span className="overnight-price-actions">
              <button type="button" disabled={quoteBusy || !ticker.trim()} onClick={refreshQuote}>{quoteBusy ? "Loading…" : "Refresh quote"}</button>
              <button type="button" disabled={!quoteIsFresh || quoteBusy} onClick={useCurrentQuote}>Use current</button>
            </span></div>
            <input ref={priceInputRef} name="assumed-close" aria-label="Assumed close" className="form-input" type="number" inputMode="decimal" step="any" min="0.0001" value={assumedClose} onChange={(event) => { priceEditedRef.current = true; setCloseInput({ value: event.target.value, quote: null }); invalidateForecast(); }} placeholder="$ if it closed now" required />
          </div>
          <label><span>Shares <small className="overnight-optional">optional</small></span><input className="form-input" type="number" inputMode="decimal" step="any" min="0.0001" value={shares} onChange={(event) => { setShares(event.target.value); invalidateForecast(); }} placeholder="Per-share" /></label>
          <button className="btn-primary" type="submit" disabled={busy || !model?.available || !model.serving_inputs_pinned}>{busy ? "Checking…" : "Estimate open"}</button>
        </div>
        {(quoteBusy || quote || quoteError) && <p className="overnight-quote-status" role="status">
          {quoteBusy ? "Checking latest trade…" : quote
            ? `Massive last trade ${quotePrice(quote.price)} · ${Number.isFinite(Date.parse(quote.observed_at)) ? easternTradeTime.format(new Date(quote.observed_at)) : "time unavailable"} · ${quoteStatus}`
            : `No fresh trade: ${quoteError}. Enter a price manually.`}
        </p>}
        <details className="overnight-costs"><summary>Adjust sensitivity & costs</summary><div>
          <label>Sensitivity step<input className="form-input" type="number" inputMode="decimal" step="any" min="0.0001" value={step} onChange={(event) => { setStep(event.target.value); invalidateForecast(); }} placeholder="Default 0.5%" /></label>
          <button className="overnight-nickel" type="button" onClick={() => { setStep("0.05"); invalidateForecast(); }}>Use $0.05</button>
          <label>Close exit cost / share<input className="form-input" type="number" inputMode="decimal" step="any" min="0" value={exitTodayCost} onChange={(event) => { setExitTodayCost(event.target.value); invalidateForecast(); }} /></label>
          <label>Next-open exit cost / share<input className="form-input" type="number" inputMode="decimal" step="any" min="0" value={exitNextCost} onChange={(event) => { setExitNextCost(event.target.value); invalidateForecast(); }} /></label>
        </div></details>
      </form>
      {error && <p className="error" role="alert">{error}</p>}
      {result && model && <ForecastResult result={result} model={model} />}
    </div>
  );
}
