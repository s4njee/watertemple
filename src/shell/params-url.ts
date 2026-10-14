import { defaultParams, type ParamDef, type ParamSchema } from '../effects/types.ts';

// Effect parameters in the query string: `p.<key>=<value>`, written only where a value differs
// from the schema default so links stay short. Pure, so it can be tested without a DOM.
//
//   number   p.warp=0.3         up to 6 significant digits; clamped to min..max, NOT snapped to
//                               step (presets and hand-edited links may sit between steps)
//   boolean  p.flag=1 | 0       `true` / `false` also accepted when reading
//   select   p.mode=choppy      must be one of the options
//   colour   p.tint=0d4a66      sRGB hex without `#`; the stored value is linear RGB

export const PARAM_PREFIX = 'p.';
export const PRESET_KEY = 'preset';

export type Values = Record<string, unknown>;
export type Presets = Readonly<Record<string, Readonly<Record<string, unknown>>>>;

export function srgbToLinear(c: number): number {
  return c <= 0.04045 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4;
}

export function linearToSrgb(c: number): number {
  return c <= 0.0031308 ? c * 12.92 : 1.055 * c ** (1 / 2.4) - 0.055;
}

const clamp01 = (c: number) => Math.min(1, Math.max(0, c));

/** Linear RGB to a six-digit sRGB hex string (lowercase, no `#`). */
export function colorToHex(color: readonly number[]): string {
  return color
    .slice(0, 3)
    .map((c) =>
      Math.round(linearToSrgb(clamp01(c)) * 255)
        .toString(16)
        .padStart(2, '0'),
    )
    .join('');
}

/** `rrggbb` or `rgb`, with or without a leading `#`, to linear RGB. Undefined if malformed. */
export function hexToColor(text: string): [number, number, number] | undefined {
  let hex = text.trim().replace(/^#/, '');
  if (/^[0-9a-f]{3}$/i.test(hex)) hex = [...hex].map((digit) => digit + digit).join('');
  if (!/^[0-9a-f]{6}$/i.test(hex)) return undefined;
  const channel = (at: number) => srgbToLinear(parseInt(hex.slice(at, at + 2), 16) / 255);
  return [channel(0), channel(2), channel(4)];
}

/** Rounds a linear colour to what its URL form can express, so a shared link looks identical. */
export function quantizeColor(color: readonly number[]): [number, number, number] {
  return hexToColor(colorToHex(color))!;
}

/** Six significant digits: far finer than any slider, and free of float noise. */
export function formatNumber(value: number): string {
  return String(Number(value.toPrecision(6)));
}

const NUMBER = /^[+-]?(\d+\.?\d*|\.\d+)(e[+-]?\d+)?$/i;

function parseNumber(text: string): number | undefined {
  const trimmed = text.trim();
  if (!NUMBER.test(trimmed)) return undefined;
  const value = Number(trimmed);
  return Number.isFinite(value) ? value : undefined;
}

/** The value a definition would hold for `raw`, or undefined if `raw` does not fit its type. */
function normalize(def: ParamDef, raw: unknown): unknown {
  switch (def.type) {
    case 'number':
      return typeof raw === 'number' && Number.isFinite(raw)
        ? Math.min(def.max, Math.max(def.min, raw))
        : undefined;
    case 'boolean':
      return typeof raw === 'boolean' ? raw : undefined;
    case 'color':
      return Array.isArray(raw) && raw.length === 3 && raw.every((c) => Number.isFinite(c))
        ? (raw as number[]).map(clamp01)
        : undefined;
    case 'select':
      return typeof raw === 'string' && def.options.includes(raw) ? raw : undefined;
  }
}

function parse(def: ParamDef, text: string): unknown {
  switch (def.type) {
    case 'number': {
      const value = parseNumber(text);
      return value === undefined ? undefined : normalize(def, value);
    }
    case 'boolean':
      if (text === '1' || text === 'true') return true;
      if (text === '0' || text === 'false') return false;
      return undefined;
    case 'color':
      return hexToColor(text);
    case 'select':
      return normalize(def, text);
  }
}

/** The URL text for a valid value; undefined when the value does not fit the definition. */
function format(def: ParamDef, value: unknown): string | undefined {
  const fitted = normalize(def, value);
  if (fitted === undefined) return undefined;
  switch (def.type) {
    case 'number':
      return formatNumber(fitted as number);
    case 'boolean':
      return fitted ? '1' : '0';
    case 'color':
      return colorToHex(fitted as number[]);
    case 'select':
      return fitted as string;
  }
}

/** The `p.*` query entries for the values that differ from the defaults. */
export function encodeParams(
  schema: ParamSchema,
  values: Readonly<Values>,
): Record<string, string> {
  const entries: Record<string, string> = {};
  for (const [key, def] of Object.entries(schema)) {
    const text = format(def, values[key]);
    if (text !== undefined && text !== format(def, def.default)) entries[PARAM_PREFIX + key] = text;
  }
  return entries;
}

/** The valid `p.*` entries of a query. Unknown keys and malformed values are skipped. */
export function decodeParams(schema: ParamSchema, query: URLSearchParams): Values {
  const values: Values = {};
  for (const [key, def] of Object.entries(schema)) {
    const text = query.get(PARAM_PREFIX + key);
    if (text === null) continue;
    const value = parse(def, text);
    if (value !== undefined) values[key] = value;
  }
  return values;
}

/** True when both value sets would produce the same link. */
export function sameValues(schema: ParamSchema, a: Readonly<Values>, b: Readonly<Values>): boolean {
  return JSON.stringify(encodeParams(schema, a)) === JSON.stringify(encodeParams(schema, b));
}

/** Defaults with a preset's overrides on top; overrides that do not fit the schema are skipped. */
export function presetValues(schema: ParamSchema, overrides: Readonly<Values>): Values {
  const values: Values = defaultParams(schema);
  for (const [key, def] of Object.entries(schema)) {
    const value = normalize(def, overrides[key]);
    if (value !== undefined) values[key] = value;
  }
  return values;
}

/**
 * The preset whose values the given values equal, if any. Derived rather than remembered, so a
 * hand edit clears it, and editing back onto a preset restores it.
 */
export function activePreset(
  schema: ParamSchema,
  presets: Presets,
  values: Readonly<Values>,
): string | undefined {
  return Object.keys(presets).find((name) =>
    sameValues(schema, presetValues(schema, presets[name]!), values),
  );
}

/**
 * Everything the query says about parameters: defaults, then `p.*` entries on top. A link with a
 * `preset` but no usable `p.*` entries (a hand-written short link) applies that preset instead.
 */
export function valuesFromQuery(
  schema: ParamSchema,
  presets: Presets,
  query: URLSearchParams,
): Values {
  const decoded = decodeParams(schema, query);
  const name = query.get(PRESET_KEY);
  if (Object.keys(decoded).length === 0 && name !== null && Object.hasOwn(presets, name)) {
    return presetValues(schema, presets[name]!);
  }
  return { ...defaultParams(schema), ...decoded };
}

/**
 * Changes for `updateQuery`: the `p.*` entries for `values` plus `preset` while one matches,
 * with stale `p.*` entries removed. `p.*` stays authoritative so a link keeps working if the
 * preset is later redefined; `preset` is only a readable label for it.
 */
export function paramQueryChanges(
  schema: ParamSchema,
  presets: Presets,
  values: Readonly<Values>,
  current: URLSearchParams,
): Record<string, string | null> {
  const changes: Record<string, string | null> = {};
  for (const key of current.keys()) if (key.startsWith(PARAM_PREFIX)) changes[key] = null;
  Object.assign(changes, encodeParams(schema, values));
  changes[PRESET_KEY] = activePreset(schema, presets, values) ?? null;
  return changes;
}

/** Identifies the parameter part of a query, to tell a real external change from other keys. */
export function paramSignature(query: URLSearchParams): string {
  const entries = [...query.entries()].filter(
    ([key]) => key.startsWith(PARAM_PREFIX) || key === PRESET_KEY,
  );
  return JSON.stringify(entries.sort(([a], [b]) => (a < b ? -1 : a > b ? 1 : 0)));
}
