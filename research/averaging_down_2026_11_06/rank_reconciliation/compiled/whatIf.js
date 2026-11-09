export const DEFAULT_TRADE_SIZE = 10_000;
export const TRADE_SIZE_RANGE = { min: 1_000, max: 100_000, step: 1_000 };
export const LOOKBACK_OPTIONS = [
    { days: null, label: "All history" },
    { days: 14, label: "Past 2 weeks" },
    { days: 21, label: "Past 3 weeks" },
];
/** Hindsight only. Same-share exposure is sold at the close or next open.
 * The separate verified raw close must agree with the What If close before
 * combining prices from the two sources. Picks excluded by this scenario
 * remain display-only.
 */
export function simulateOvernightHold(row, actual, dollarsPerTrade, holdFraction) {
    if (!Number.isFinite(holdFraction) || holdFraction < 0 || holdFraction > 1) {
        throw new RangeError("Hold fraction must be between 0 and 1");
    }
    const includedInTotals = row.hypothetical.status === "scored" || row.hypothetical.status === "held";
    if (row.open_price === null || !Number.isFinite(row.open_price) || row.open_price <= 0
        || row.close_price === null || !Number.isFinite(row.close_price) || row.close_price <= 0
        || row.hypothetical.endingValue === null) {
        return { status: "awaiting_close", incrementalPnl: null, endingValue: null, includedInTotals };
    }
    if (!actual || actual.status !== "observed" || actual.verified_close === null || actual.next_open === null) {
        return { status: actual?.status === "awaiting_next_open" ? "awaiting_next_open" : "unavailable",
            incrementalPnl: null, endingValue: null, includedInTotals };
    }
    if (Math.abs(actual.verified_close - row.close_price) > Math.max(0.005, row.close_price * 0.001)) {
        return { status: "price_mismatch", incrementalPnl: null, endingValue: null, includedInTotals };
    }
    const shares = dollarsPerTrade / row.open_price;
    const incrementalPnl = cents(shares * holdFraction * (actual.next_open - row.close_price)) / 100;
    return { status: "observed", incrementalPnl,
        endingValue: (cents(row.hypothetical.endingValue) + cents(incrementalPnl)) / 100,
        includedInTotals };
}
/** Entry-to-exit return for the entire trade, including shares sold at the close.
 * An overnight recovery is a win only when the total trade P&L is positive.
 */
export function overnightTradeReturn(row, actual, dollarsPerTrade, holdFraction) {
    if (row.hypothetical.pnl === null)
        return null;
    const outcome = simulateOvernightHold(row, actual, dollarsPerTrade, holdFraction);
    if (outcome.status !== "observed" || outcome.incrementalPnl === null)
        return null;
    return (cents(row.hypothetical.pnl) + cents(outcome.incrementalPnl)) / cents(dollarsPerTrade);
}
export function summarizeOvernightHold(outcomes, baselinePnl) {
    const included = outcomes.filter((outcome) => outcome.includedInTotals);
    const observed = included.filter((outcome) => outcome.status === "observed");
    const incrementalPnl = observed.reduce((sum, outcome) => sum + cents(outcome.incrementalPnl), 0) / 100;
    const complete = included.length > 0 && observed.length === included.length;
    return { included: included.length, observed: observed.length, incrementalPnl,
        scenarioPnl: complete && baselinePnl !== null ? (cents(baselinePnl) + cents(incrementalPnl)) / 100 : null };
}
export function overnightSelectionKey(day, kind) {
    return `${day.as_of}:${day.scan_id || "live"}:${kind}`;
}
/** Select at the close, using only information available by then. */
export function selectOvernightExitRows(rows, rule, customTickers = new Set()) {
    return rows.filter((row) => {
        if (row.hypothetical.status !== "scored" && row.hypothetical.status !== "held")
            return false;
        if (rule === "down")
            return row.session_return !== null && row.session_return < 0;
        if (rule === "custom")
            return customTickers.has(row.ticker);
        return true;
    });
}
/** The existing close result is the baseline; only selected shares change exit. */
export function summarizeOvernightExit(rows, actuals, dollarsPerTrade, baseline, holdFraction, rule, customTickers) {
    if (!Number.isFinite(holdFraction) || holdFraction < 0 || holdFraction > 1) {
        throw new RangeError("Hold fraction must be between 0 and 1");
    }
    const selectedRows = selectOvernightExitRows(rows, rule, customTickers);
    const outcomes = selectedRows.map((row) => simulateOvernightHold(row, actuals.get(row.ticker), dollarsPerTrade, holdFraction));
    const observed = outcomes.filter((outcome) => outcome.status === "observed").length;
    const waiting = outcomes.some((outcome) => outcome.status === "awaiting_next_open" || outcome.status === "awaiting_close");
    const complete = baseline.pending === 0 && baseline.pnl !== null && baseline.endingValue !== null && !waiting;
    if (!complete)
        return { selected: selectedRows.length, observed, pnl: null, endingValue: null, extraPnl: null };
    // A past open that cannot be compared stays at the close. Waiting for this morning blanks the day.
    const extraCents = outcomes.reduce((sum, outcome) => sum + cents(outcome.incrementalPnl ?? 0), 0);
    return {
        selected: selectedRows.length,
        observed,
        pnl: (cents(baseline.pnl) + extraCents) / 100,
        endingValue: (cents(baseline.endingValue) + extraCents) / 100,
        extraPnl: extraCents / 100,
    };
}
/** Aggregate only days whose selected exit has a verified next open. */
export function summarizeSelectedOvernightHistory(days, actualsByDay, kind, dollarsPerTrade, holdFraction, rule, customSelections = new Map()) {
    let eligibleDays = 0;
    let matchedDays = 0;
    let pnlCents = 0;
    let endingCents = 0;
    for (const day of days) {
        if (day.scan_id)
            continue;
        const baseline = day[`${kind}_money`];
        if (baseline.invested <= 0)
            continue;
        eligibleDays += 1;
        const result = summarizeOvernightExit(day[kind], actualsByDay.get(day.as_of) ?? new Map(), dollarsPerTrade, baseline, holdFraction, rule, customSelections.get(overnightSelectionKey(day, kind)));
        if (result.pnl === null || result.endingValue === null)
            continue;
        matchedDays += 1;
        pnlCents += cents(result.pnl);
        endingCents += cents(result.endingValue);
    }
    return {
        eligibleDays, matchedDays,
        pnl: matchedDays ? pnlCents / 100 : null,
        endingValue: matchedDays ? endingCents / 100 : null,
    };
}
/** Use the same complete live days for both repeatable rules so totals compare fairly. */
export function summarizeOvernightHistory(days, actualsByDay, kind, dollarsPerTrade, holdFraction) {
    let eligibleDays = 0;
    let matchedDays = 0;
    let baselineCents = 0;
    let baselineEndingCents = 0;
    let allPnlCents = 0;
    let allEndingCents = 0;
    let allExtraCents = 0;
    let downPnlCents = 0;
    let downEndingCents = 0;
    let downExtraCents = 0;
    let allSelected = 0;
    let allObserved = 0;
    let downSelected = 0;
    let downObserved = 0;
    const noActuals = new Map();
    for (const day of days) {
        if (day.scan_id)
            continue;
        const rows = day[kind];
        const baseline = day[`${kind}_money`];
        if (baseline.invested <= 0)
            continue;
        eligibleDays += 1;
        const actuals = actualsByDay.get(day.as_of) ?? noActuals;
        const all = summarizeOvernightExit(rows, actuals, dollarsPerTrade, baseline, holdFraction, "all");
        const down = summarizeOvernightExit(rows, actuals, dollarsPerTrade, baseline, holdFraction, "down");
        if (all.pnl === null || down.pnl === null)
            continue;
        matchedDays += 1;
        baselineCents += cents(baseline.pnl);
        baselineEndingCents += cents(baseline.endingValue);
        allPnlCents += cents(all.pnl);
        allEndingCents += cents(all.endingValue);
        allExtraCents += cents(all.extraPnl);
        downPnlCents += cents(down.pnl);
        downEndingCents += cents(down.endingValue);
        downExtraCents += cents(down.extraPnl);
        allSelected += all.selected;
        allObserved += all.observed;
        downSelected += down.selected;
        downObserved += down.observed;
    }
    const summary = (selected, observed, pnlCents, endingCents, extraCents) => ({
        selected, observed,
        pnl: matchedDays ? pnlCents / 100 : null,
        endingValue: matchedDays ? endingCents / 100 : null,
        extraPnl: matchedDays ? extraCents / 100 : null,
    });
    return {
        eligibleDays, matchedDays,
        baselinePnl: matchedDays ? baselineCents / 100 : null,
        baselineEndingValue: matchedDays ? baselineEndingCents / 100 : null,
        all: summary(allSelected, allObserved, allPnlCents, allEndingCents, allExtraCents),
        down: summary(downSelected, downObserved, downPnlCents, downEndingCents, downExtraCents),
    };
}
/** Simple period P&L on the average capital put to work each active day.
 * The same denominator makes a reused intraday stake comparable in dollars
 * with one buy-and-hold stake; this is not a compounded return.
 */
export function returnOnAverageDailyCapital(pnl, averageDailyCapital) {
    if (pnl === null || averageDailyCapital === null
        || !Number.isFinite(pnl) || !Number.isFinite(averageDailyCapital)
        || averageDailyCapital <= 0)
        return null;
    return pnl / averageDailyCapital;
}
/** Calendar windows follow the trading dashboard's timezone, not the browser's. */
export function simulationDay(now = new Date()) {
    return new Intl.DateTimeFormat("en-CA", { timeZone: "America/Chicago" }).format(now);
}
const cents = (value) => Math.round(value * 100);
const hasReturn = (row) => row.session_return !== null && Number.isFinite(row.session_return);
const isHardSkip = (row) => Boolean(row.news_blocks && row.news_flag);
const isNewsHold = (row) => Boolean(row.news_blocks && !row.news_flag);
const hasTradePrices = (row) => row.open_price !== null && Number.isFinite(row.open_price) && row.open_price > 0
    && row.close_price !== null && Number.isFinite(row.close_price) && row.close_price > 0;
/** Fixed notional per pick; fractional-share equivalent, before fees/slippage.
 * Round each pick once so displayed line items reconcile with every total.
 */
export function simulatePick(row, dollarsPerTrade, applyNewsSkips = true) {
    if (!Number.isFinite(dollarsPerTrade) || dollarsPerTrade <= 0) {
        throw new RangeError("Dollars per trade must be positive and finite");
    }
    if (!hasReturn(row))
        return { status: isHardSkip(row) && applyNewsSkips ? "skipped" : "pending", pnl: null, endingValue: null };
    if (isHardSkip(row) && !applyNewsSkips && !hasTradePrices(row)) {
        return { status: "pending", pnl: null, endingValue: null };
    }
    const pnlCents = cents(dollarsPerTrade * row.session_return);
    const endingValue = (cents(dollarsPerTrade) + pnlCents) / 100;
    if (isHardSkip(row) && applyNewsSkips) {
        return { status: "skipped", pnl: null, endingValue };
    }
    return { status: isNewsHold(row) ? "held" : "scored", pnl: pnlCents / 100, endingValue };
}
function moneySummary(rows, dollarsPerTrade) {
    const completed = rows.filter((row) => row.hypothetical.status === "scored" || row.hypothetical.status === "held");
    const pnlCents = completed.reduce((sum, row) => sum + cents(row.hypothetical.pnl), 0);
    const investedCents = completed.length * cents(dollarsPerTrade);
    return {
        pnl: completed.length ? pnlCents / 100 : null,
        invested: investedCents / 100,
        endingValue: completed.length ? (investedCents + pnlCents) / 100 : null,
        completed: completed.length,
        pending: rows.filter((row) => row.hypothetical.status === "pending").length,
    };
}
function listStats(rows) {
    const kept = rows.filter((row) => row.hypothetical.status === "scored" || row.hypothetical.status === "held");
    const returns = kept.map((row) => row.session_return);
    const wins = returns.filter((value) => value > 0).length;
    const losses = returns.filter((value) => value < 0).length;
    const avg = returns.length ? returns.reduce((sum, value) => sum + value, 0) / returns.length : null;
    const exNews = rows.filter((row) => !isHardSkip(row) && hasReturn(row)).map((row) => row.session_return);
    return {
        n: rows.length, n_scored: returns.length, wins, losses,
        flats: returns.length - wins - losses,
        hit_rate: returns.length ? wins / returns.length : null,
        avg, avg_ex_news: exNews.length ? exNews.reduce((sum, value) => sum + value, 0) / exNews.length : null,
        n_avoid: rows.filter((row) => row.hypothetical.status === "skipped").length,
    };
}
function compound(avgs) {
    return avgs.length ? avgs.reduce((wealth, avg) => wealth * (1 + avg), 1) - 1 : null;
}
/** One pipeline for filtering, sizing, day results and overall results.
 * Top-K is applied before news skips: skipped picks are not backfilled.
 * Replay rows can be inspected but never double-count a live session in totals.
 * Dollar profits are added, not compounded or multiplied by the compound metric.
 */
export function simulateBook(data, options) {
    const { kind, fitTopK, rankTopK, applyNewsSkips = true, excludedTickers, tradeSizes, lookbackDays, asOf } = options;
    const start = new Date(`${asOf}T12:00:00Z`);
    if (lookbackDays !== null)
        start.setUTCDate(start.getUTCDate() - lookbackDays + 1);
    const from = lookbackDays === null ? null : start.toISOString().slice(0, 10);
    const rowsFor = (rows, dollarsPerTrade, topK) => rows
        .filter((row) => topK === undefined || row.rank <= topK)
        // Exclusions remove original picks; they do not promote a lower-ranked replacement.
        .filter((row) => !excludedTickers?.has(row.ticker.toUpperCase()))
        .map((row) => ({ ...row, hypothetical: simulatePick(row, dollarsPerTrade, applyNewsSkips) }));
    const days = data.days
        .filter((day) => day.as_of <= asOf && (from === null || day.as_of >= from))
        .map((day) => {
        const fit = kind === "rank" ? [] : rowsFor(day.fit, tradeSizes.fit, fitTopK);
        const rank = kind === "fit" ? [] : rowsFor(day.rank, tradeSizes.rank, rankTopK);
        const fitStats = listStats(fit), rankStats = listStats(rank);
        return {
            ...day, fit, rank, fit_avg: fitStats.avg, rank_avg: rankStats.avg,
            fit_stats: fitStats, rank_stats: rankStats,
            fit_money: moneySummary(fit, tradeSizes.fit), rank_money: moneySummary(rank, tradeSizes.rank),
        };
    });
    const liveDays = days.filter((day) => !day.scan_id);
    const fit = liveDays.flatMap((day) => day.fit), rank = liveDays.flatMap((day) => day.rank);
    const fitAvgs = liveDays.map((day) => day.fit_avg).filter((value) => value !== null);
    const rankAvgs = liveDays.map((day) => day.rank_avg).filter((value) => value !== null);
    return {
        ...data, days, from, through: asOf, kind,
        fit_compound: compound(fitAvgs), rank_compound: compound(rankAvgs),
        fit_days: fitAvgs.length, rank_days: rankAvgs.length,
        fit_stats: listStats(fit), rank_stats: listStats(rank),
        fit_money: moneySummary(fit, tradeSizes.fit), rank_money: moneySummary(rank, tradeSizes.rank),
        n_picks: fit.length + rank.length,
    };
}
/** Replays and unscored/skipped picks never create benchmark exposure. A date
 * appears once even if the live book contains multiple rows for that session.
 */
function strategyCapitalByDay(book, kind) {
    const capital = new Map();
    for (const day of book.days) {
        if (day.scan_id)
            continue;
        const invested = kind === "rank" ? day.rank_money.invested : day.fit_money.invested;
        if (invested > 0)
            capital.set(day.as_of, (capital.get(day.as_of) ?? 0) + invested);
    }
    return capital;
}
/** Only completed, active live dates are sent to the shared SPY endpoint. */
export function benchmarkDates(book, kind) {
    return [...strategyCapitalByDay(book, kind).keys()].sort();
}
/** Match SPY's day sessions to the strategy's invested capital on each day.
 * Buy-and-hold uses one fixed stake (average active-day capital) from the
 * first selected day's open to the last selected day's close. The API's
 * first-close to last-close return is chained after the first day's session.
 */
export function compareStrategyToBenchmark(book, kind, benchmark) {
    const capitalByDay = strategyCapitalByDay(book, kind);
    const dates = [...capitalByDay.keys()].sort();
    const activeDays = dates.length;
    const totalCapital = [...capitalByDay.values()].reduce((sum, value) => sum + value, 0);
    const holdCapital = activeDays ? totalCapital / activeDays : null;
    let matchedDays = 0;
    let intradayCapital = 0;
    let intradayPnl = 0;
    for (const [date, capital] of capitalByDay) {
        const sessionReturn = benchmark?.returns[date];
        if (sessionReturn === undefined || !Number.isFinite(sessionReturn))
            continue;
        matchedDays += 1;
        intradayCapital += capital;
        intradayPnl += capital * sessionReturn;
    }
    const intradayDollars = matchedDays ? cents(intradayPnl) / 100 : null;
    const hold = benchmark?.hold;
    const firstSession = benchmark?.returns[dates[0]];
    const hasFullHoldInterval = activeDays > 0
        && hold?.start === dates[0]
        && hold?.end === dates[activeDays - 1]
        && Number.isFinite(hold.pct)
        && firstSession !== undefined && Number.isFinite(firstSession);
    const holdPct = hasFullHoldInterval ? (1 + firstSession) * (1 + hold.pct) - 1 : null;
    const holdDollars = holdPct === null ? null : cents(holdCapital * holdPct) / 100;
    return {
        intraday: {
            pnl: intradayDollars,
            pct: intradayDollars === null ? null : intradayDollars / intradayCapital,
        },
        hold: {
            pnl: holdDollars,
            pct: holdPct,
        },
        matchedDays,
        activeDays,
        intradayCapital,
        holdCapital,
        holdFrom: hasFullHoldInterval ? hold.start : null,
        holdThrough: hasFullHoldInterval ? hold.end : null,
    };
}
/** Independent requests: one failing strategy must not hide the other.
 * Identical date lists share a request; historical lists can be cached by the
 * caller, while current-day lists are always refetched to avoid stale closes.
 */
export async function loadStrategyBenchmarks(rankDates, fitDates, fetcher, historicalCache, today) {
    const requests = new Map();
    const get = (dates) => {
        const key = dates.join(",");
        if (!dates.length)
            return Promise.resolve({ response: null, error: null });
        if (!requests.has(key)) {
            const isHistorical = today !== undefined && dates.every((date) => date < today);
            const cached = isHistorical ? historicalCache?.get(key) : undefined;
            const request = cached ?? fetcher(dates)
                .then((response) => ({ response, error: null }))
                .catch((error) => {
                historicalCache?.delete(key); // A failed request must be retried on the next selection.
                return { response: null, error: String(error) };
            });
            if (isHistorical && historicalCache && !cached)
                historicalCache.set(key, request);
            requests.set(key, request);
        }
        return requests.get(key);
    };
    const [rank, fit] = await Promise.all([get(rankDates), get(fitDates)]);
    return { rankKey: rankDates.join(","), fitKey: fitDates.join(","), rank, fit };
}
