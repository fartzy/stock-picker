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
  const labels: Record<NewsCheckDetails["status"], string> = {
    complete: "reviewed",
    no_news: "no articles found",
    degraded: "limited check",
    error: "check failed",
    not_checked: "not checked",
    unknown: "coverage not recorded",
  };
  const detail = [
    check.reviewed_count != null ? `${check.reviewed_count} of ${count ?? "?"} articles reviewed` : "",
    check.sources.length ? `Sources: ${check.sources.join(", ")}` : "",
    ...check.issues.map((issue) => ISSUE_LABELS[issue] ?? "Incomplete news coverage"),
    check.checked_at ? `Checked ${new Date(check.checked_at).toLocaleString()}` : "",
  ].filter(Boolean).join(". ");
  return {
    label: `${labels[check.status] ?? "coverage not recorded"}${articles}`,
    detail,
    warning: check.status === "degraded" || check.status === "error",
  };
}
