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

function shiftDay(day: string, amount: number): string {
  const date = calendarDate(day);
  date.setUTCDate(date.getUTCDate() + amount);
  return date.toISOString().slice(0, 10);
}

function weekLabel(start: string, end: string): string {
  const format = (day: string) => calendarDate(day).toLocaleDateString("en-US", {
    timeZone: "UTC", month: "short", day: "numeric",
  });
  return `Week ${format(start)}–${format(end)}`;
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
  return weekLabel(start, shiftDay(start, 4));
}

/** Partition only the supplied days, never pulling in dates outside the parent. */
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

/** Month-local week labels and keys keep boundary weeks and expansion state
 * separate from the adjacent month and from the standalone Weeks view. */
export function groupMonthWeeks<T>(month: HistoryPeriod<T>): HistoryPeriod<T>[] {
  if (!month.days.length) return [];
  const monthStart = `${month.days[0][0].slice(0, 7)}-01`;
  const end = calendarDate(monthStart);
  end.setUTCMonth(end.getUTCMonth() + 1, 0);
  const monthEnd = end.toISOString().slice(0, 10);
  return groupHistoryPeriods(month.days, "weeks").map((week) => {
    const monday = mondayOfDay(week.days[0][0]);
    const friday = shiftDay(monday, 4);
    // Include any recorded weekend execution rather than labeling it outside its range.
    const weekEnd = week.days[0][0] > friday ? week.days[0][0] : friday;
    const start = monday < monthStart ? monthStart : monday;
    const finish = weekEnd > monthEnd ? monthEnd : weekEnd;
    return {
      ...week,
      key: `${month.key}/${week.key}`,
      label: weekLabel(start, finish),
    };
  });
}

/** ISO calendar days supplied by the caller use the trading timezone. */
export function isWeekToDate(day: string, today: string): boolean {
  return day >= mondayOfDay(today) && day <= today;
}
