import { useState } from "react";
import { fetchJevEvaluation, refreshJevEvaluation, type JevEvaluation as Evaluation, type JevTargetMetrics } from "../jevApi";
import { JEV_QUESTION_LABELS, jevPercent, jevPriceBasis } from "../jevDisplay";
import { useFetchData } from "../useFetchData";

const errorScore = (value: number | null) => value == null ? "Not available" : value.toFixed(4);

function Target({ target }: { target: JevTargetMetrics }) {
  return <div className="jev-target">
    <h4>{JEV_QUESTION_LABELS[target.question] ?? target.question}</h4>
    <p>{target.n} resolved · {target.unresolved} unresolved · Probability error (Brier): <strong>{errorScore(target.brier)}</strong></p>
    <p className="muted">Lower error is better. This measures probability accuracy, not calibration alone.</p>
    <p className="muted">Returns in each band use {target.question === "hold_overnight" || target.question === "net_positive_overnight" ? "close to next open" : "the saved entry to close"}{target.question === "downside_before_close" ? "; the downside event itself is a touch before close" : ""}.</p>
    <div className="jev-table-scroll"><table className="jev-evaluation-table"><thead><tr><th>Probability band</th><th>Count</th><th>Mean prediction</th><th>Observed frequency</th><th>Mean after costs</th><th>Loss rate</th></tr></thead>
      <tbody>{target.reliability.map((bin) => <tr key={bin.lower}><td>{Math.round(bin.lower * 100)} to {Math.round(bin.upper * 100)}%</td><td>{bin.n}</td><td>{jevPercent(bin.mean_probability)}</td><td>{jevPercent(bin.event_rate)}</td><td>{jevPercent(bin.mean_net_return)}</td><td>{jevPercent(bin.loss_rate)}</td></tr>)}</tbody></table></div>
    <p className="muted">Chronological comparison on {target.baseline_n} matched outcomes: Jev {errorScore(target.matched_jev_brier)} · earlier-session event rate {errorScore(target.prior_day_baseline_brier)}. The first session has no earlier baseline.</p>
  </div>;
}

export default function JevEvaluation() {
  const [tick, setTick] = useState(0);
  const { data, error, loading } = useFetchData(fetchJevEvaluation, { deps: [tick], trackLoading: true });
  const [refreshed, setRefreshed] = useState<Evaluation | null>(null);
  const [busy, setBusy] = useState(false);
  const [refreshError, setRefreshError] = useState<string | null>(null);
  const evaluation = refreshed ?? data;

  async function refresh() {
    setBusy(true);
    setRefreshError(null);
    try { setRefreshed(await refreshJevEvaluation()); }
    catch (cause) { setRefreshError(cause instanceof Error ? cause.message : "Outcome refresh failed"); }
    finally { setBusy(false); }
  }

  return <section className="jev-evaluation">
    <div className="section-title-row"><h3>Does Jev help?</h3><div className="jev-actions">
      <button className="btn-primary" type="button" onClick={() => void refresh()} disabled={busy}>{busy ? "Refreshing…" : "Refresh results"}</button>
    </div></div>
    {(error || refreshError) && <p className="error" role="alert">{refreshError ?? error}</p>}
    {loading && !evaluation && <p className="muted">Loading forecast history…</p>}
    {evaluation && <>
      <p className="jev-status"><strong>{evaluation.counts.completed} completed</strong> · {evaluation.counts.pending} pending{evaluation.counts.unavailable > 0 && <> · {evaluation.counts.unavailable} unavailable</>}</p>
      {evaluation.counts.completed === 0 && <div className="jev-empty">
        <strong>{evaluation.forecast_count === 0 ? "No forecasts yet" : "No completed results yet"}</strong>
        <p className="muted">{evaluation.primary_count === 0 ? "Use “Assess” on a stock to add a forecast." : evaluation.counts.pending > 0 ? "Refresh after the close or next open to check outcomes." : "Open Forecasts for unavailable outcome details."}</p>
      </div>}
      {evaluation.cohorts.map((cohort) => <details className="view-card jev-cohort" key={[cohort.question_set_version, cohort.model, cohort.kind, cohort.entry_basis, cohort.round_trip_cost_per_share, cohort.downside_fraction].join(":")}>
        <summary>{cohort.kind === "rank" ? "Rank" : "Fit"} · {cohort.question_set_version === "entry_v1" ? "Entry to close" : cohort.question_set_version === "morning_v1" ? "Morning questions" : "Close to next open"} · {cohort.completed}/{cohort.records} completed</summary>
        <p className="muted">{cohort.model} · {jevPriceBasis(cohort.entry_basis)}</p>
        <p className="muted">Costs ${cohort.round_trip_cost_per_share ?? 0}/share{cohort.downside_fraction != null ? ` · downside level ${jevPercent(cohort.downside_fraction)}` : ""}. {cohort.question_set_version === "morning_v1" && "Forecasts were generated at their recorded time, usually after the open. Open-to-close returns are descriptive and do not imply an available-at-open trading strategy."}</p>
        <p>{cohort.sessions} completed sessions · Mean after costs {jevPercent(cohort.returns.mean_net_return)} · Win rate {jevPercent(cohort.returns.win_rate)} · Losses {cohort.returns.losses} · Worst {jevPercent(cohort.returns.worst_net_return)}</p>
        {cohort.targets.map((target) => <Target key={target.question} target={target} />)}
        <h4>Same decision cohort</h4><p className="muted">Rank and Fit retain separate units. These are descriptive return comparisons, not a tuned trading rule.</p>
        <div className="jev-table-scroll"><table className="jev-evaluation-table"><thead><tr><th>Group</th><th>Resolved</th><th>Mean after costs</th><th>Win rate</th></tr></thead><tbody>
          {cohort.comparisons.map((group) => <tr key={group.name}><td>{group.name}</td><td>{group.n}</td><td>{jevPercent(group.mean_net_return)}</td><td>{jevPercent(group.win_rate)}</td></tr>)}
        </tbody></table></div>
      </details>)}
      <details className="jev-ledger"><summary>Forecasts ({evaluation.forecast_count})</summary>
        <div className="jev-table-scroll"><table className="jev-evaluation-table"><thead><tr><th>Session</th><th>Ticker</th><th>List / question</th><th>Traded</th><th>Sample</th><th>Outcome</th><th>After costs</th></tr></thead><tbody>
          {[...evaluation.rows].reverse().map((row) => <tr key={row.forecast_id}><td>{row.session}</td><td>{row.ticker}</td><td>{row.kind} / {row.question_set_version}</td><td>{row.traded ? "Yes" : "No"}</td><td>{row.primary ? "Initial" : "Separate"}</td><td title={row.reason ?? undefined}>{row.status}{row.reason ? `: ${row.reason}` : ""}</td><td>{jevPercent(row.net_return)}</td></tr>)}
        </tbody></table></div>
      </details>
    </>}
    <details className="jev-details"><summary>Details</summary>
      <p className="muted">Tracks all saved candidates, including stocks you never traded. Refresh checks the price API without requesting another Jev forecast.</p>
      {evaluation && <p className="muted">{evaluation.primary_count} initial forecasts · {evaluation.duplicate_count} reassessments excluded · {evaluation.legacy_count} records missing snapshots · {evaluation.failed_count} failed requests</p>}
      <button className="section-title-action" type="button" onClick={() => { setRefreshed(null); setTick(tick + 1); }} disabled={busy || loading}>Reload history</button>
      {evaluation && <p className="muted jev-note">{evaluation.policy}</p>}
    </details>
  </section>;
}
