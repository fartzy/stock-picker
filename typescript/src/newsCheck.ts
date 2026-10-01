import type { NewsCheckDetails } from "./api";

const ISSUE_LABELS: Record<string, string> = {
  finnhub_unavailable: "Finnhub unavailable",
  polygon_unavailable: "Backup feed unavailable",
  polygon_truncated: "Backup feed has more articles",
  finnhub_undated_articles: "Finnhub returned articles without dates",
  polygon_undated_articles: "Backup feed returned articles without dates",
  finnhub_invalid_articles: "Finnhub returned incomplete articles",
  polygon_invalid_articles: "Backup feed returned incomplete articles",
  llm_unavailable: "AI review unavailable; local classifier used",
  review_truncated: "AI review covered only part of the articles",
  feeds_unavailable: "News feeds unavailable",
  check_failed: "News check failed",
  ticker_limit: "Outside this batch's 40-name news limit",
};

export function newsCheckPresentation(check?: NewsCheckDetails | null, checked?: boolean | null, flag?: string | null) {
  if (!check) {
    return {
      label: checked === false ? "pending" : flag ? "" : checked === true ? "checked (older scan)" : "details unavailable",
      detail: "This older scan saved the news decision, but not detailed source coverage.",
      warning: false,
    };
  }
  const count = check.article_count;
  const articles = count == null ? "" : ` · ${count} article${count === 1 ? "" : "s"}`;
  const classifierOnly = check.status === "degraded" && check.issues.length === 1 && check.issues[0] === "llm_unavailable";
  const labels: Record<NewsCheckDetails["status"], string> = {
    complete: "reviewed",
    no_news: "no recent articles",
    degraded: classifierOnly ? "AI unavailable" : "limited news check",
    error: "news check failed",
    not_checked: "not checked",
    unknown: "coverage not recorded",
  };
  let coverageLabel = `${labels[check.status] ?? "coverage not recorded"}${articles}`;
  if (check.status === "no_news" || check.status === "error") {
    coverageLabel = labels[check.status];
  } else if (classifierOnly && count != null) {
    coverageLabel = `${labels.degraded} · ${count} article${count === 1 ? "" : "s"} screened`;
  }
  const detail = [
    check.reviewed_count != null && count !== 0
      ? `${check.reviewed_count} of ${count ?? "?"} articles ${check.judge === "classifier" ? "screened by local classifier" : "reviewed"}`
      : "",
    check.sources.length ? `Sources: ${check.sources.join(", ")}` : "",
    ...check.issues.map((issue) => ISSUE_LABELS[issue] ?? "Incomplete news coverage"),
    check.checked_at ? `Checked ${new Date(check.checked_at).toLocaleString()}` : "",
  ].filter(Boolean).join(". ");
  return {
    label: flag ? "" : coverageLabel,
    detail,
    warning: check.status === "degraded" || check.status === "error",
  };
}
