import { useEffect, useRef, useState, type FormEvent } from "react";
import {
  fetchOvernightCurrentPrice, fetchOvernightModel, fetchPositions, forecastOvernight,
  type OvernightForecastCase, type OvernightForecastResponse, type OvernightModelResponse,
  type OvernightCurrentPriceResponse, type PositionsResponse,
} from "../api";
import { formatUsd } from "../format";
import { overnightDirection, overnightDollarText, overnightGapText, overnightQuoteStatus } from "../overnightDisplay";
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
const easternShortTime = new Intl.DateTimeFormat("en-US", {
  timeZone: "America/New_York", hour: "numeric", minute: "2-digit",
});

function isFreshQuote(quote: OvernightCurrentPriceResponse, now: number): boolean {
  return isFreshCashSessionQuote(quote, now);
}

function CaseSummary({ value }: { value: OvernightForecastCase }) {
  const direction = overnightDirection(value.predicted_gap, value.difference_per_share);
  const directionLabel = direction === "up" ? "↑ Up" : direction === "down" ? "↓ Down" : value.predicted_gap === 0 ? "→ Flat" : "→ ≈ Flat";
  return (
    <div className="overnight-case">
      <span>Close <strong>{quotePrice(value.assumed_close)}</strong></span>
      <span>Open <strong>{formatUsd(value.projected_open)}</strong></span>
      <span>{directionLabel} {overnightGapText(value.predicted_gap, direction)}</span>
    </div>
  );
}

function FeatureSnapshot({ value, columns }: { value: OvernightForecastCase; columns: string[] }) {
  const groups = [...new Set(columns.map((name) => OVERNIGHT_FEATURE_GROUPS[name] ?? "Other"))];
  return (
    <details className="overnight-result-details overnight-feature-snapshot">
      <summary>Model inputs · {columns.length}</summary>
      <div className="overnight-result-detail-body">
        <p className="muted">Inputs, not explanations of cause.</p>
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
      </div>
    </details>
  );
}

function ForecastResult({ result, model }: { result: OvernightForecastResponse; model: OvernightModelResponse }) {
  const primary = result.cases.find((item) => item.label === "primary");
  if (!primary) return <p className="error">Forecast response is missing its primary scenario.</p>;
  const neighbors = result.cases.filter((item) => item.label !== "primary");
  const losesToUnchanged = primary.oof_model_gap_mae !== null && primary.oof_zero_gap_mae !== null
    && primary.oof_model_gap_mae >= primary.oof_zero_gap_mae;
  const direction = overnightDirection(primary.predicted_gap, primary.difference_per_share);
  const directionLabel = direction === "up" ? "↑ Up" : direction === "down" ? "↓ Down" : primary.predicted_gap === 0 ? "→ Flat" : "→ ≈ Flat";
  const hasComparison = primary.oof_model_gap_mae !== null && primary.oof_zero_gap_mae !== null;
  return (
    <div className="overnight-results" aria-live="polite">
      <div className="overnight-forecast-card">
        <span className="overnight-result-label">Estimated next open · {primary.next_session ?? "next session"}</span>
        <div className="overnight-result-main">
          <strong>{formatUsd(primary.projected_open)}</strong>
          <span className={`overnight-direction is-${direction}`} aria-label={`Model direction: ${direction}, ${overnightGapText(primary.predicted_gap, direction)}`}>
            {directionLabel} {overnightGapText(primary.predicted_gap, direction)}
          </span>
        </div>
        <span className="overnight-result-from">From {quotePrice(primary.assumed_close)} assumed close · {overnightDollarText(primary.difference_per_share)}/share before costs</span>
      </div>
      <p className="overnight-result-caution">
        {hasComparison
          ? `${losesToUnchanged ? "No measured edge" : "Historical error"} · ${percentage(primary.oof_model_gap_mae)} model vs ${percentage(primary.oof_zero_gap_mae)} unchanged`
          : "Experimental estimate · historical error unavailable"}
      </p>
      <details className="overnight-result-details">
        <summary>Accuracy &amp; costs</summary>
        <div className="overnight-result-detail-body">
          <p>Held-out gap error from {result.evaluated_rows} rows. Historical 90th-percentile absolute open error at this price: {primary.oof_abs_open_error_p90_at_assumed_price === null ? "not measured" : formatUsd(primary.oof_abs_open_error_p90_at_assumed_price)}; not a guaranteed range.</p>
          <p>Gross change: {overnightDollarText(primary.difference_per_share)}/share{primary.gross_difference_for_shares !== null ? ` · ${overnightDollarText(primary.gross_difference_for_shares)} for ${result.shares} shares` : ""}. {primary.after_cost_difference_per_share === null
            ? "No execution costs entered."
            : `After entered exit costs: ${overnightDollarText(primary.after_cost_difference_per_share)}/share${primary.after_cost_difference_for_shares !== null ? ` · ${overnightDollarText(primary.after_cost_difference_for_shares)} total` : ""}.`}</p>
          <p>Scenario assumes today closes at the entered price; it does not predict the rest of today. Latest next-open label used: {result.model_label_observed_on}. {model.feature_columns.length} inputs.</p>
        </div>
      </details>
      <details className="overnight-result-details">
        <summary>Other close prices</summary>
        <div className="overnight-result-detail-body">
          <p>What-if sensitivity, not an uncertainty range.</p>
          {neighbors.map((item) => <CaseSummary key={item.label} value={item} />)}
        </div>
      </details>
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
  const quoteStatus = overnightQuoteStatus(quoteIsFresh, quote !== null && isWithinQuoteSession(quote, now));
  const quoteObservedAt = quote && Number.isFinite(Date.parse(quote.observed_at)) ? new Date(quote.observed_at) : null;

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
        {(quoteBusy || quote || quoteError) && <div className="overnight-quote-line">
          <span className="overnight-quote-status" role="status">
            {quoteBusy ? "Checking latest trade…" : quote
              ? `Last ${quotePrice(quote.price)} · ${quoteObservedAt ? `${easternShortTime.format(quoteObservedAt)} ET` : "time unavailable"} · ${quoteStatus}`
              : `No fresh trade: ${quoteError}. Enter a price manually.`}
          </span>
          {quote && <details className="overnight-quote-source">
            <summary>Source</summary>
            <p>Massive last trade · {quoteObservedAt ? easternTradeTime.format(quoteObservedAt) : "time unavailable"}. This is an editable closing-price assumption, not the official close.</p>
          </details>}
        </div>}
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
