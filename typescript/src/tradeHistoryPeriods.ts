export const HISTORY_GROUPINGS = [
  { value: "days", label: "Days" },
  { value: "weeks", label: "Weeks" },
  { value: "months", label: "Months" },
] as const;

export type HistoryGrouping = (typeof HISTORY_GROUPINGS)[number]["value"];
export type DayGroup<T> = [string, T[]];
export interface HistoryPeriod<T> {
  key: string;
  label: string;
  days: DayGroup<T>[];
}

function calendarDate(day: string): Date {
  return new Date(`${day}T12:00:00Z`);
}

export function mondayOfDay(day: string): string {
  const date = calendarDate(day);
  date.setUTCDate(date.getUTCDate() - (date.getUTCDay() + 6) % 7);
  return date.toISOString().slice(0, 10);
}

function periodLabel(start: string, grouping: "weeks" | "months"): string {
  const date = calendarDate(start);
  if (grouping === "months") {
    return date.toLocaleDateString("en-US", { timeZone: "UTC", month: "long", year: "numeric" });
  }
  const format = (value: Date) => value.toLocaleDateString("en-US", {
    timeZone: "UTC", month: "short", day: "numeric",
  });
  const first = format(date);
  date.setUTCDate(date.getUTCDate() + 4);
  return `Week ${first}–${format(date)}`;
}

/** Partition only the supplied days. Passing a month's days back in with
 * "weeks" can later produce partial-week children without pulling in adjacent months. */
export function groupHistoryPeriods<T>(days: DayGroup<T>[], grouping: "weeks" | "months"): HistoryPeriod<T>[] {
  const periods = new Map<string, DayGroup<T>[]>();
  for (const day of days) {
    const start = grouping === "weeks" ? mondayOfDay(day[0]) : `${day[0].slice(0, 7)}-01`;
    const group = periods.get(start) ?? [];
    group.push(day);
    periods.set(start, group);
  }
  return [...periods.entries()]
    .sort(([a], [b]) => b.localeCompare(a))
    .map(([start, entries]) => ({
      key: `${grouping}:${start}`,
      label: periodLabel(start, grouping),
      days: entries.sort(([a], [b]) => b.localeCompare(a)),
    }));
}

/** ISO calendar days supplied by the caller use the trading timezone. */
export function isWeekToDate(day: string, today: string): boolean {
  return day >= mondayOfDay(today) && day <= today;
}
