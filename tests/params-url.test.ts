import { describe, expect, it } from 'vitest';
import type { ParamSchema } from '../src/effects/types.ts';
import {
  activePreset,
  colorToHex,
  decodeParams,
  encodeParams,
  formatNumber,
  hexToColor,
  linearToSrgb,
  paramQueryChanges,
  paramSignature,
  presetValues,
  quantizeColor,
  srgbToLinear,
  valuesFromQuery,
} from '../src/shell/params-url.ts';

const schema = {
  speed: { type: 'number', default: 1, min: 0, max: 3, step: 0.01 },
  warp: { type: 'number', default: 0.6, min: 0, max: 1.5 },
  foam: { type: 'boolean', default: true },
  mode: { type: 'select', default: 'calm', options: ['calm', 'choppy', 'storm'] },
  tint: { type: 'color', default: [0.05, 0.25, 0.4] },
} as const satisfies ParamSchema;

const q = (search: string) => new URLSearchParams(search);

describe('encodeParams', () => {
  it('omits values equal to the defaults', () => {
    expect(
      encodeParams(schema, {
        speed: 1,
        warp: 0.6,
        foam: true,
        mode: 'calm',
        tint: [0.05, 0.25, 0.4],
      }),
    ).toEqual({});
    expect(encodeParams(schema, {})).toEqual({});
  });

  it('writes every type under the p. prefix', () => {
    const entries = encodeParams(schema, {
      speed: 2.5,
      warp: 0,
      foam: false,
      mode: 'storm',
      tint: [1, 0, 0],
    });
    expect(entries).toEqual({
      'p.speed': '2.5',
      'p.warp': '0',
      'p.foam': '0',
      'p.mode': 'storm',
      'p.tint': 'ff0000',
    });
  });

  it('strips float noise from numbers', () => {
    expect(formatNumber(0.1 + 0.2)).toBe('0.3');
    expect(formatNumber(1 / 3)).toBe('0.333333');
    expect(formatNumber(-0)).toBe('0');
    expect(encodeParams(schema, { speed: 0.1 + 0.2 })).toEqual({ 'p.speed': '0.3' });
  });

  it('treats a number that only differs by float noise as the default', () => {
    expect(encodeParams(schema, { warp: 0.6 + 1e-12 })).toEqual({});
  });

  it('clamps out-of-range numbers and skips values of the wrong type', () => {
    expect(encodeParams(schema, { speed: 99 })).toEqual({ 'p.speed': '3' });
    expect(
      encodeParams(schema, { speed: NaN, warp: '1', foam: 0, mode: 'tidal', tint: [1, 2] }),
    ).toEqual({});
  });

  it('ignores keys that are not in the schema', () => {
    expect(encodeParams(schema, { nope: 5, speed: 2 })).toEqual({ 'p.speed': '2' });
  });
});

describe('decodeParams', () => {
  it('reads each type', () => {
    expect(decodeParams(schema, q('p.speed=2.5&p.foam=0&p.mode=storm&p.warp=0.25'))).toEqual({
      speed: 2.5,
      foam: false,
      mode: 'storm',
      warp: 0.25,
    });
  });

  it('returns only entries that are present', () => {
    expect(decodeParams(schema, q('effect=hello&tab=source'))).toEqual({});
  });

  it('clamps numbers to min and max but does not snap them to step', () => {
    expect(decodeParams(schema, q('p.speed=-4'))).toEqual({ speed: 0 });
    expect(decodeParams(schema, q('p.speed=1e9'))).toEqual({ speed: 3 });
    expect(decodeParams(schema, q('p.speed=1.2345'))).toEqual({ speed: 1.2345 });
  });

  it('ignores malformed values and unknown keys without throwing', () => {
    const garbage = [
      'p.speed=',
      'p.speed=abc',
      'p.speed=NaN',
      'p.speed=Infinity',
      'p.speed=0x10',
      'p.speed=1,5',
      'p.speed=1 2',
      'p.foam=yes',
      'p.foam=',
      'p.mode=tidal',
      'p.mode=',
      'p.tint=nothex',
      'p.tint=12345',
      'p.tint=%23gg0000',
      'p.tint=',
      'p.unknown=1',
      'p.=1',
      'p.__proto__=1',
      'p.speed.extra=1',
    ].join('&');
    expect(() => decodeParams(schema, q(garbage))).not.toThrow();
    expect(decodeParams(schema, q(garbage))).toEqual({});
  });

  it('uses the first of a repeated key', () => {
    expect(decodeParams(schema, q('p.speed=2&p.speed=1.5'))).toEqual({ speed: 2 });
  });

  it('accepts true/false and a hash or shorthand on colours', () => {
    expect(decodeParams(schema, q('p.foam=false'))).toEqual({ foam: false });
    const hash = decodeParams(schema, q('p.tint=%23FF0000'));
    expect(hash.tint).toEqual([1, 0, 0]);
    expect(decodeParams(schema, q('p.tint=f00')).tint).toEqual([1, 0, 0]);
  });
});

describe('colour conversion', () => {
  it('converts sRGB and linear properly', () => {
    expect(srgbToLinear(1)).toBeCloseTo(1, 12);
    expect(srgbToLinear(0.5)).toBeCloseTo(0.2140411, 6);
    expect(linearToSrgb(srgbToLinear(0.3))).toBeCloseTo(0.3, 12);
    // 0.5 linear is far brighter than 0x80, which a plain *255 would get wrong.
    expect(colorToHex([0.5, 0.5, 0.5])).toBe('bcbcbc');
    expect(colorToHex([srgbToLinear(0.5), srgbToLinear(0.5), srgbToLinear(0.5)])).toBe('808080');
  });

  it('round-trips every 8-bit level exactly', () => {
    for (let level = 0; level < 256; level++) {
      const hex = level.toString(16).padStart(2, '0');
      expect(colorToHex(hexToColor(hex.repeat(3))!)).toBe(hex.repeat(3));
    }
  });

  it('clamps out-of-range channels', () => {
    expect(colorToHex([2, -1, 0.5])).toBe('ff00' + colorToHex([0.5, 0.5, 0.5]).slice(0, 2));
  });

  it('is stable: decode(encode(x)) does not drift on further round trips', () => {
    const first = decodeParams(schema, q('p.tint=' + colorToHex([0.123, 0.456, 0.789]))).tint;
    expect(first).toBeDefined();
    const second = decodeParams(
      schema,
      new URLSearchParams(encodeParams(schema, { tint: first })),
    ).tint;
    expect(second).toEqual(first);
    expect(quantizeColor(first as number[])).toEqual(first);
  });

  it('keeps a colour within one 8-bit step of the original', () => {
    const original: [number, number, number] = [0.123, 0.456, 0.789];
    const back = quantizeColor(original);
    original.forEach((c, i) => {
      expect(Math.abs(linearToSrgb(back[i]!) - linearToSrgb(c))).toBeLessThanOrEqual(
        0.5 / 255 + 1e-12,
      );
    });
  });

  it('omits a colour whose hex equals the default hex', () => {
    expect(encodeParams(schema, { tint: quantizeColor([0.05, 0.25, 0.4]) })).toEqual({});
  });
});

describe('round trips', () => {
  it('survives encode, URL text, decode for every type', () => {
    const values = {
      speed: 2.25,
      warp: 1.5,
      foam: false,
      mode: 'choppy',
      tint: quantizeColor([0.7, 0.1, 0.3]),
    };
    const search = new URLSearchParams(encodeParams(schema, values)).toString();
    expect(decodeParams(schema, q(search))).toEqual(values);
  });

  it('keeps the canonical form on a second pass', () => {
    const once = encodeParams(schema, { speed: 1 / 3, tint: [0.3, 0.2, 0.1] });
    const decoded = decodeParams(schema, new URLSearchParams(once));
    expect(encodeParams(schema, decoded)).toEqual(once);
  });
});

describe('presets', () => {
  const presets = {
    calm: { speed: 0.4, warp: 0.3 },
    storm: { speed: 2, mode: 'storm', foam: false },
    plain: {},
  };

  it('resolves defaults plus overrides', () => {
    expect(presetValues(schema, presets.calm)).toMatchObject({ speed: 0.4, warp: 0.3, foam: true });
    expect(presetValues(schema, { speed: 99, mode: 'nope' })).toMatchObject({
      speed: 3,
      mode: 'calm',
    });
  });

  it('finds the active preset from the values, and loses it on a hand edit', () => {
    const values = presetValues(schema, presets.calm);
    expect(activePreset(schema, presets, values)).toBe('calm');
    expect(activePreset(schema, presets, { ...values, speed: 0.41 })).toBeUndefined();
    expect(activePreset(schema, presets, presetValues(schema, {}))).toBe('plain');
    expect(activePreset(schema, {}, values)).toBeUndefined();
  });

  it('writes p.* values plus the preset label, and removes stale entries', () => {
    const values = presetValues(schema, presets.storm);
    const current = q('effect=hello&p.warp=0.2&p.gone=1&tab=source&preset=calm');
    expect(paramQueryChanges(schema, presets, values, current)).toEqual({
      'p.warp': null,
      'p.gone': null,
      'p.speed': '2',
      'p.foam': '0',
      'p.mode': 'storm',
      preset: 'storm',
    });
  });

  it('clears the label once the values no longer match', () => {
    const values = { ...presetValues(schema, presets.calm), warp: 0.9 };
    expect(paramQueryChanges(schema, presets, values, q('preset=calm')).preset).toBeNull();
  });

  it('reads a short link that carries only a preset', () => {
    expect(valuesFromQuery(schema, presets, q('preset=calm'))).toMatchObject({
      speed: 0.4,
      warp: 0.3,
    });
  });

  it('lets p.* win over the preset label, and ignores unknown presets', () => {
    expect(valuesFromQuery(schema, presets, q('preset=calm&p.speed=2'))).toMatchObject({
      speed: 2,
      warp: 0.6,
    });
    expect(valuesFromQuery(schema, presets, q('preset=nope'))).toMatchObject({ speed: 1 });
    expect(valuesFromQuery(schema, presets, q('preset=constructor'))).toMatchObject({ speed: 1 });
  });
});

describe('paramSignature', () => {
  it('changes only with p.* and preset entries', () => {
    const base = paramSignature(q('effect=hello&p.speed=2'));
    expect(paramSignature(q('tab=source&hud=1&p.speed=2&effect=hello'))).toBe(base);
    expect(paramSignature(q('effect=hello&p.speed=2.5'))).not.toBe(base);
    expect(paramSignature(q('effect=hello&p.speed=2&preset=calm'))).not.toBe(base);
  });

  it('ignores the order of entries', () => {
    expect(paramSignature(q('p.a=1&p.b=2'))).toBe(paramSignature(q('p.b=2&p.a=1')));
  });
});
