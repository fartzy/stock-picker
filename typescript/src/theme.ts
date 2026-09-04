type RGB = [number, number, number];

const CSS_VAR_NAMES = {
  surfaceRaised: "--surface-raised",
  accent: "--accent",
  good: "--good",
  bad: "--bad",
  neutral: "--neutral",
  corrNegative: "--corr-negative",
  corrNeutral: "--corr-neutral",
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
