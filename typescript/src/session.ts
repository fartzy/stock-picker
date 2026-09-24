// US cash session, matching python/stock_picker/ingestion/session.py.
// Keep the hour/settle numbers in lockstep with that module.

const NY = "America/New_York";
export const SESSION_CLOSE_HOUR = 16;
export const SESSION_CLOSE_SETTLE_MINUTES = 15;

function nyParts(now: Date): { year: number; month: number; day: number; hour: number; minute: number; weekday: number } {
  const parts = new Intl.DateTimeFormat("en-US", {
    timeZone: NY,
    weekday: "short",
    year: "numeric",
    month: "2-digit",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
    hourCycle: "h23",
  }).formatToParts(now);
  const get = (type: string) => parts.find((p) => p.type === type)?.value ?? "";
  const weekdayName = get("weekday");
  const weekday = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"].indexOf(weekdayName);
  return {
    year: Number(get("year")),
    month: Number(get("month")),
    day: Number(get("day")),
    hour: Number(get("hour")),
    minute: Number(get("minute")),
    weekday,
  };
}

function isoDate(year: number, month: number, day: number): string {
  return `${year}-${String(month).padStart(2, "0")}-${String(day).padStart(2, "0")}`;
}

function shiftCalendarDay(year: number, month: number, day: number, delta: number): { year: number; month: number; day: number; weekday: number } {
  const utc = new Date(Date.UTC(year, month - 1, day + delta));
  return {
    year: utc.getUTCFullYear(),
    month: utc.getUTCMonth() + 1,
    day: utc.getUTCDate(),
    weekday: utc.getUTCDay(),
  };
}

export function cashSessionDate(now: Date = new Date()): string {
  let { year, month, day, weekday } = nyParts(now);
  while (weekday >= 5) {
    const prev = shiftCalendarDay(year, month, day, -1);
    year = prev.year;
    month = prev.month;
    day = prev.day;
    weekday = prev.weekday;
  }
  return isoDate(year, month, day);
}

export function sessionHasClosed(now: Date = new Date()): boolean {
  const { hour, minute, weekday } = nyParts(now);
  if (weekday >= 5) return true;
  return hour > SESSION_CLOSE_HOUR || (hour === SESSION_CLOSE_HOUR && minute >= SESSION_CLOSE_SETTLE_MINUTES);
}

export function isCashSessionToday(asOf: string | null | undefined, now: Date = new Date()): boolean {
  if (!asOf) return false;
  return asOf === cashSessionDate(now);
}
