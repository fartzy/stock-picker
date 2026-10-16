/** Same rule as python/stock_picker/news_skip.py. */

const ALWAYS_SKIP_PHRASES = [
  "insider sell",
  "insider sold",
  "insider sale",
  "form 4",
  "cto sells",
  "cto sold",
  "cfo sells",
  "cfo sold",
  "ceo sells",
  "ceo sold",
  "director sells",
  "director sold",
  "officer sells",
  "officer sold",
  "sold shares",
  "sells shares",
  "share sale",
  "stock sale",
  "corporate action:",
  "clinical readout:",
] as const;

const INCOMPLETE_STATUSES = ["degraded", "error", "not_checked", "unknown"] as const;

const HARD_STOP_REVIEW_LABELS: ReadonlyArray<readonly [string, string]> = [
  ["corporate action:", "spinoff-review"],
  ["clinical readout:", "clinical-readout-review"],
];

export function hardStopReviewLabel(newsFlag: string | null | undefined): string | null {
  if (!newsFlag) return null;
  const text = newsFlag.toLowerCase();
  const match = HARD_STOP_REVIEW_LABELS.find(([phrase]) => text.includes(phrase));
  return match ? match[1] : null;
}

export function alwaysSkipNews(newsFlag: string | null | undefined): boolean {
  if (!newsFlag) return false;
  const text = newsFlag.toLowerCase();
  return ALWAYS_SKIP_PHRASES.some((phrase) => text.includes(phrase));
}

export function newsBlocksBuy(
  newsFlag: string | null | undefined,
  openPrice: number | null | undefined,
  prevClose: number | null | undefined,
  checkStatus?: string | null,
): boolean {
  if (INCOMPLETE_STATUSES.some((status) => status === checkStatus)) return true;
  if (!newsFlag) return false;
  if (alwaysSkipNews(newsFlag)) return true;
  if (openPrice == null || prevClose == null || prevClose <= 0) return false;
  return openPrice < prevClose;
}
