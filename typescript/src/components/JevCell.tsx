import type { JevShadowScores } from "../api";

const ISSUE_LABELS: Record<string, string> = {
  not_configured: "Off",
  request_failed: "Failed",
};

function pct(value: number | null): string {
  return value === null ? "?" : `${Math.round(value * 100)}`;
}

/** Shadow Jev confidences as buy · hold · skip percentages. Display only. */
export default function JevCell({ scores }: { scores?: JevShadowScores | null }) {
  if (!scores) return <span className="muted">Not asked</span>;
  if (scores.issue) {
    const label = ISSUE_LABELS[scores.issue] ?? (scores.issue.startsWith("http_") ? "Failed" : "Unavailable");
    return <span className="muted" title={scores.issue}>{label}</span>;
  }
  return (
    <span className="jev-cell" title={scores.model ?? undefined}>
      {pct(scores.buy_today)} · {pct(scores.hold_overnight)} · {pct(scores.news_skip)}
    </span>
  );
}
