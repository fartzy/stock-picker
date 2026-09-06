type RGB = [number, number, number];

const CSS_VAR_NAMES = {
  surfaceRaised: "--surface-raised",
  accent: "--accent",
  good: "--good",
  bad: "--bad",
  neutral: "--neutral",
  line: "--line",
  textMuted: "--text-muted",
} as const;

type ThemeKey = keyof typeof CSS_VAR_NAMES;

const HEX_COLOR = /^#?[0-9a-f]{6}$/i;

// Fallback used when a CSS variable is missing or isn't a 6-digit hex color --
// keeps rendering visibly-neutral instead of silently caching NaN components.
const FALLBACK_RGB: RGB = [153, 153, 153];

const rgbCache = new Map<ThemeKey, RGB>();

function hexToRgb(hex: string): RGB | null {
  const clean = hex.trim();
  if (!HEX_COLOR.test(clean)) return null;
  const stripped = clean.replace("#", "");
  return [parseInt(stripped.slice(0, 2), 16), parseInt(stripped.slice(2, 4), 16), parseInt(stripped.slice(4, 6), 16)];
}

export function themeRgb(key: ThemeKey): RGB {
  const cached = rgbCache.get(key);
  if (cached) return cached;

  const raw = getComputedStyle(document.documentElement).getPropertyValue(CSS_VAR_NAMES[key]);
  const rgb = hexToRgb(raw) ?? FALLBACK_RGB;
  if (rgb === FALLBACK_RGB) {
    console.error(`theme.ts: CSS variable ${CSS_VAR_NAMES[key]} is missing or malformed ("${raw}"); using fallback color`);
  }
  rgbCache.set(key, rgb);
  return rgb;
}

export function themeColor(key: ThemeKey): string {
  return `rgb(${themeRgb(key).join(",")})`;
}

// Coverage gradient (bad -> good), interpolated below rather than a hard
// red/green split -- 0.4-1.0 because a well-formed feature is rarely below
// 40% covered even at a long lookback window, so that's the meaningful range.
export const COVERAGE_GRADIENT_MIN_MAX: [number, number] = [0.4, 1.0];

export function coverageColor(pct: number | undefined): string {
  if (pct === undefined) return themeColor("neutral");
  const [gradientMin, gradientMax] = COVERAGE_GRADIENT_MIN_MAX;
  const t = Math.max(0, Math.min(1, (pct - gradientMin) / (gradientMax - gradientMin)));
  const bad = themeRgb("bad");
  const good = themeRgb("good");
  const rgb = bad.map((c, i) => Math.round(c + (good[i] - c) * t));
  return `rgb(${rgb.join(",")})`;
}

// Feature importance is percent-of-total-gain across ~95 features -- most land
// well under 10%, so a fixed 0-15% gradient (rather than coverage's 0.4-1.0) is
// what actually spreads the colors out; a feature above ~15% is already a clear
// standout in a model this wide.
export const IMPORTANCE_GRADIENT_MAX_PCT = 15;

export function importanceColor(pct: number | undefined): string {
  if (pct === undefined || pct === 0) return themeColor("neutral");
  const t = Math.max(0, Math.min(1, pct / IMPORTANCE_GRADIENT_MAX_PCT));
  const neutral = themeRgb("neutral");
  const accent = themeRgb("accent");
  const rgb = neutral.map((c, i) => Math.round(c + (accent[i] - c) * t));
  return `rgb(${rgb.join(",")})`;
}

// The strongest tint a shaded cell can reach, at the column's own extreme
// value -- kept well under 1 so the number underneath always stays legible,
// per the same "color alongside the number, not instead of it" reasoning
// coverageColor/importanceColor already follow with a bounded gradient range.
export const MAGNITUDE_SHADE_MAX_ALPHA = 0.35;

// A diverging tint for a signed numeric cell (e.g. a derived feature's raw
// value), reusing --good/--bad exactly as Diff.tsx already does for price
// moves -- no new color concept, just applied per-cell instead of per-quote.
// Unlike coverageColor/importanceColor (one-directional 0..1 gradients),
// this scales alpha with |value| relative to that column's own max -- a
// column-relative scale, since features span wildly different raw ranges
// (RSI 0-100 vs. returns ~0.02) and a shared absolute scale would wash out
// everything but the widest-range columns.
export function signedMagnitudeColor(value: number, maxAbs: number): string {
  if (maxAbs === 0) return "transparent";
  const t = Math.max(-1, Math.min(1, value / maxAbs));
  const base = themeRgb(t >= 0 ? "good" : "bad");
  const alpha = Math.abs(t) * MAGNITUDE_SHADE_MAX_ALPHA;
  return `rgba(${base.join(",")}, ${alpha.toFixed(2)})`;
}

// A feature below this contributes almost nothing to the trained ensemble --
// with ~95 features, a uniform split would average ~1.05% each, so anything
// under half that is a real standout-low, not just "below average." Below
// this AND not already pruned, Registry flags it as a one-click pruning
// candidate (see Registry.tsx).
export const NEGLIGIBLE_IMPORTANCE_PCT_THRESHOLD = 0.5;
