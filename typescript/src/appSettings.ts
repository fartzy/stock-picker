import { HISTORY_GROUPINGS, type HistoryGrouping } from "./tradeHistoryPeriods.js";

export interface HistoryDefaults {
  grouping: HistoryGrouping;
  keepRecentWeekOpen: boolean;
  keepRecentMonthOpen: boolean;
}

export const DEFAULT_HISTORY_SETTINGS: HistoryDefaults = {
  grouping: "months",
  keepRecentWeekOpen: false,
  keepRecentMonthOpen: true,
};

const SETTINGS_KEY = "stock-picker.history-defaults.v1";

/** Validate saved preferences independently so an older or damaged value is harmless. */
export function parseHistoryDefaults(raw: string | null): HistoryDefaults {
  try {
    const value: unknown = JSON.parse(raw ?? "null");
    if (!value || typeof value !== "object") return { ...DEFAULT_HISTORY_SETTINGS };
    const saved = value as Partial<HistoryDefaults>;
    return {
      grouping: HISTORY_GROUPINGS.find((option) => option.value === saved.grouping)?.value
        ?? DEFAULT_HISTORY_SETTINGS.grouping,
      keepRecentWeekOpen: typeof saved.keepRecentWeekOpen === "boolean"
        ? saved.keepRecentWeekOpen : DEFAULT_HISTORY_SETTINGS.keepRecentWeekOpen,
      keepRecentMonthOpen: typeof saved.keepRecentMonthOpen === "boolean"
        ? saved.keepRecentMonthOpen : DEFAULT_HISTORY_SETTINGS.keepRecentMonthOpen,
    };
  } catch {
    return { ...DEFAULT_HISTORY_SETTINGS };
  }
}

export function loadHistoryDefaults(): HistoryDefaults {
  try {
    return parseHistoryDefaults(window.localStorage.getItem(SETTINGS_KEY));
  } catch {
    return { ...DEFAULT_HISTORY_SETTINGS };
  }
}

export function saveHistoryDefaults(settings: HistoryDefaults): boolean {
  try {
    window.localStorage.setItem(SETTINGS_KEY, JSON.stringify(settings));
    return true;
  } catch {
    return false;
  }
}
