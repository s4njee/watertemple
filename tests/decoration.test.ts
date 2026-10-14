import { afterEach, describe, expect, it, vi } from 'vitest';
import { loadEffect } from '../src/effects/registry.ts';
import { defaultParams } from '../src/effects/types.ts';
import { noProfiler } from '../src/gpu/profiler.ts';
import {
  isInViewport,
  scrollSpeed,
  smoothTowards,
  trackScroll,
  viewportProgress,
} from '../src/gpu/scroll.ts';
import { getTestDevice, readTexture } from './gpu.ts';
import { gpuF32, lane, rowsOf, scalars } from './wgsl-eval.ts';

vi.setConfig({ testTimeout: 60_000 });

// --- Helpers -------------------------------------------------------------------------------------

function halfToFloat(h: number): number {
  const sign = h & 0x8000 ? -1 : 1;
  const exponent = (h >> 10) & 0x1f;
  const fraction = h & 0x3ff;
  if (exponent === 0) return sign * 2 ** -14 * (fraction / 1024);
  if (exponent === 31) return fraction ? Number.NaN : sign * Infinity;
  return sign * 2 ** (exponent - 15) * (1 + fraction / 1024);
}

const srgbToLinear = (c: number) => (c <= 0.04045 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4);
const hexToLinear = (css: string): [number, number, number] => {
  const n = parseInt(css.replace('#', ''), 16);
  return [(n >> 16) & 255, (n >> 8) & 255, n & 255].map((v) => srgbToLinear(v / 255)) as [
    number,
    number,
    number,
  ];
};

/** Linear luminance of an sRGB byte triple, for contrast ratios. */
function luminance(r: number, g: number, b: number): number {
  return (
    0.2126 * srgbToLinear(r / 255) + 0.7152 * srgbToLinear(g / 255) + 0.0722 * srgbToLinear(b / 255)
  );
}
const contrast = (a: number, b: number) => (Math.max(a, b) + 0.05) / (Math.min(a, b) + 0.05);

function fakeCanvas(width: number, height: number): HTMLCanvasElement {
  return Object.assign(new EventTarget(), {
    width,
    height,
    style: {},
    getBoundingClientRect: () => ({ left: 0, top: 0, width, height }),
  }) as unknown as HTMLCanvasElement;
}

interface Rendered {
  width: number;
  height: number;
  /** RGBA, floats (rgba16float target) or bytes / 255 (rgba8unorm). */
  data: Float32Array;
}

interface RenderOptions {
  width?: number;
  height?: number;
  dpr?: number;
  /** Seconds to advance in the first frame. */
  advance?: number;
  /** `float` keeps NaN and out-of-range values visible; `byte` is what a canvas would show. */
  target?: 'float' | 'byte';
  frames?: number;
}

/** Runs a demo effect headlessly with `params` over the defaults and reads the result back. */
async function render(
  id: string,
  overrides: Record<string, unknown> = {},
  preset?: string,
  options: RenderOptions = {},
): Promise<Rendered> {
  const {
    width = 96,
    height = 64,
    dpr = 1,
    advance = 1 / 60,
    target = 'float',
    frames = 2,
  } = options;
  const device = await getTestDevice();
  const effect = await loadEffect(id);
  const format: GPUTextureFormat = target === 'float' ? 'rgba16float' : 'rgba8unorm';
  const texture = device.createTexture({
    size: [width, height],
    format,
    usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC,
  });
  const params = defaultParams(effect.params) as Record<string, unknown>;
  if (preset) Object.assign(params, effect.presets?.[preset]);
  Object.assign(params, overrides);
  const instance = await effect.create({
    device,
    format,
    canvas: fakeCanvas(width, height),
    params: params as never,
    profiler: noProfiler,
  });
  instance.resize(width, height, dpr);
  for (let frame = 0; frame < frames; frame++) {
    const encoder = device.createCommandEncoder();
    instance.frame(
      { time: frame / 60, delta: frame === 0 ? advance : 1 / 60, frame },
      encoder,
      texture.createView(),
    );
    device.queue.submit([encoder.finish()]);
  }
  const bytes = await readTexture(device, texture, target === 'float' ? 8 : 4);
  instance.destroy();
  texture.destroy();
  const data = new Float32Array(width * height * 4);
  if (target === 'float') {
    const halves = new Uint16Array(bytes.buffer, bytes.byteOffset, data.length);
    for (let i = 0; i < data.length; i++) data[i] = halfToFloat(halves[i] ?? 0);
  } else {
    for (let i = 0; i < data.length; i++) data[i] = (bytes[i] ?? 0) / 255;
  }
  return { width, height, data };
}

/** The first pixel of `image` that breaks `ok`, as a message, or undefined. */
function firstBad(
  image: Rendered,
  ok: (r: number, g: number, b: number, a: number) => boolean,
): string | undefined {
  const { data } = image;
  for (let i = 0; i < data.length; i += 4) {
    const r = data[i] ?? Number.NaN;
    const g = data[i + 1] ?? Number.NaN;
    const b = data[i + 2] ?? Number.NaN;
    const a = data[i + 3] ?? Number.NaN;
    if (!ok(r, g, b, a)) return `pixel ${i / 4}: rgba(${r}, ${g}, ${b}, ${a})`;
  }
  return undefined;
}

// --- Scroll helper ---------------------------------------------------------------------------------

describe('scroll arithmetic', () => {
  it('is 0 when the box enters at the bottom and 1 when it leaves at the top', () => {
    // A 100px box in an 800px viewport.
    expect(viewportProgress(800, 100, 0, 800)).toBe(0);
    expect(viewportProgress(-100, 100, 0, 800)).toBe(1);
    expect(viewportProgress(350, 100, 0, 800)).toBeCloseTo(0.5, 12);
  });

  it('is unclamped outside the trip and linear in the box position', () => {
    expect(viewportProgress(900, 100, 0, 800)).toBeLessThan(0);
    expect(viewportProgress(-300, 100, 0, 800)).toBeGreaterThan(1);
    const a = viewportProgress(500, 100, 0, 800);
    const b = viewportProgress(400, 100, 0, 800);
    const c = viewportProgress(300, 100, 0, 800);
    expect(b - a).toBeCloseTo(c - b, 12);
  });

  it('measures against an inner scroller that is not at the top of the page', () => {
    // The scroller starts 200px down and is 400px tall: a box at its bottom edge has just entered.
    expect(viewportProgress(600, 50, 200, 400)).toBe(0);
    expect(viewportProgress(150, 50, 200, 400)).toBe(1);
  });

  it('survives an empty viewport and box', () => {
    expect(viewportProgress(0, 0, 0, 0)).toBe(0.5);
  });

  it('knows when the box overlaps the viewport', () => {
    expect(isInViewport(700, 100, 0, 800)).toBe(true);
    expect(isInViewport(800, 100, 0, 800)).toBe(false);
    expect(isInViewport(-100, 100, 0, 800)).toBe(false);
    expect(isInViewport(-99, 100, 0, 800)).toBe(true);
  });

  it('smooths exponentially, independent of the frame rate', () => {
    // 60 steps of 1/60 s and 1 step of 1 s land on the same value.
    let stepped = 0;
    for (let i = 0; i < 60; i++) stepped = smoothTowards(stepped, 1, 1 / 60, 0.25);
    expect(stepped).toBeCloseTo(smoothTowards(0, 1, 1, 0.25), 10);
    expect(smoothTowards(0, 1, 1, 0.25)).toBeCloseTo(1 - Math.exp(-4), 12);
    // No time constant means no smoothing; no time means no movement.
    expect(smoothTowards(0, 5, 0.016, 0)).toBe(5);
    expect(smoothTowards(2, 5, 0, 0.1)).toBe(2);
  });

  it('turns two scroll offsets into px/s', () => {
    expect(scrollSpeed(100, 110, 0.1)).toBeCloseTo(100, 9);
    expect(scrollSpeed(110, 100, 0.1)).toBeCloseTo(-100, 9);
    expect(scrollSpeed(100, 110, 0)).toBe(0);
  });
});

describe('trackScroll', () => {
  afterEach(() => {
    delete (globalThis as { window?: unknown }).window;
    delete (globalThis as { getComputedStyle?: unknown }).getComputedStyle;
  });

  interface FakeNode {
    parentElement: FakeNode | null;
    overflowY: string;
    scrollTop: number;
    top: number;
    height: number;
    getBoundingClientRect(): { top: number; height: number };
  }
  const node = (parent: FakeNode | null, init: Partial<FakeNode> = {}): FakeNode => {
    const n: FakeNode = {
      parentElement: parent,
      overflowY: 'visible',
      scrollTop: 0,
      top: 0,
      height: 0,
      getBoundingClientRect: () => ({ top: n.top, height: n.height }),
      ...init,
    };
    return n;
  };
  const install = (innerHeight: number) => {
    const win = { innerHeight, scrollY: 0 };
    Object.assign(globalThis, {
      window: win,
      getComputedStyle: (el: FakeNode) => ({ overflowY: el.overflowY }),
    });
    return win;
  };

  it('is inert without a DOM: half way through, not scrolling, never throwing', () => {
    expect(typeof window).toBe('undefined');
    const tracker = trackScroll(fakeCanvas(10, 10));
    const state = tracker.update(1 / 60);
    expect(state.progress).toBe(0.5);
    expect(state.offset).toBe(0);
    expect(state.velocity).toBe(0);
    expect(state.visible).toBe(true);
    expect(() => tracker.update(Number.NaN)).not.toThrow();
    tracker.destroy();
    expect(() => tracker.update(1 / 60)).not.toThrow();
  });

  it('follows a canvas through the window', () => {
    const win = install(800);
    const html = node(null);
    const body = node(html);
    const canvas = node(body, { top: 800, height: 100 });
    const tracker = trackScroll(canvas as unknown as HTMLCanvasElement, { smoothing: 0 });
    expect(tracker.update(1 / 60).progress).toBe(0);
    canvas.top = 350;
    win.scrollY = 450;
    const mid = tracker.update(1 / 60);
    expect(mid.progress).toBeCloseTo(0.5, 9);
    expect(mid.offset).toBe(450);
    expect(mid.velocity).toBeGreaterThan(0);
    canvas.top = -100;
    expect(tracker.update(1 / 60).progress).toBe(1);
    canvas.top = -500;
    const gone = tracker.update(1 / 60);
    expect(gone.progress).toBe(1);
    expect(gone.rawProgress).toBeGreaterThan(1);
    expect(gone.visible).toBe(false);
    tracker.destroy();
  });

  it('uses the nearest scrolling ancestor as the viewport, and its scroll position as the offset', () => {
    install(900);
    const html = node(null);
    const body = node(html);
    // A 400px scroller 100px down the page holding a sticky 400px canvas that never moves on screen.
    const scroller = node(body, { overflowY: 'auto', top: 100, height: 400 });
    const stage = node(scroller);
    const canvas = node(stage, { top: 100, height: 400 });
    const tracker = trackScroll(canvas as unknown as HTMLCanvasElement, { smoothing: 0 });
    const first = tracker.update(1 / 60);
    // Canvas fills the scroller exactly: half way through the scroller's own trip.
    expect(first.progress).toBeCloseTo(0.5, 9);
    expect(first.offset).toBe(0);
    for (let i = 0; i < 120; i++) {
      scroller.scrollTop += 10;
      tracker.update(1 / 60);
    }
    // The canvas sat still on screen, yet the offset kept counting: that is what parallax reads.
    expect(tracker.progress).toBeCloseTo(0.5, 9);
    expect(tracker.offset).toBe(1200);
    // 10 px per 1/60 s is 600 px/s; the smoothed velocity has settled near it.
    expect(tracker.velocity).toBeGreaterThan(550);
    expect(tracker.velocity).toBeLessThan(650);
    // And it dies away when scrolling stops.
    for (let i = 0; i < 120; i++) tracker.update(1 / 60);
    expect(Math.abs(tracker.velocity)).toBeLessThan(1);
    tracker.destroy();
  });

  it('does not start from zero on a page that is already scrolled', () => {
    const win = install(800);
    const html = node(null);
    const canvas = node(node(html), { top: 0, height: 100 });
    win.scrollY = 3000;
    const tracker = trackScroll(canvas as unknown as HTMLCanvasElement);
    const state = tracker.update(1 / 60);
    expect(state.offset).toBe(3000);
    expect(state.velocity).toBe(0);
  });
});

// --- waveline.wgsl ------------------------------------------------------------------------------------

describe('waveline', () => {
  const include = '#include "waveline.wgsl"\n';

  it('covers 0 well above the curve and 1 well below it, with a one-pixel ramp', async () => {
    // Pixel size 1; test x across a few wavelengths, y measured down from the top.
    const rows: number[][] = [];
    for (let i = 0; i < 200; i++) rows.push([i * 7.3, i * 0.37, 3 + (i % 5) * 1.1, 0]);
    const out = await gpuF32(
      `${include}
       fn f(x: vec4f) -> vec4f {
         let c = waveline_curve(x.x, x.y, 240.0, 20.0, 0.5, x.z);
         let line = 100.0 + c.x;
         let above = waveline_coverage(waveline_distance(line - 60.0, line, c.y), 1.0);
         let below = waveline_coverage(waveline_distance(line + 60.0, line, c.y), 1.0);
         let on = waveline_coverage(waveline_distance(line, line, c.y), 1.0);
         return vec4f(above, below, on, 0.0);
       }`,
      rowsOf(rows),
    );
    for (const v of lane(out, 0)) expect(v).toBe(0);
    for (const v of lane(out, 1)) expect(v).toBe(1);
    for (const v of lane(out, 2)) expect(v).toBeCloseTo(0.5, 6);
  });

  it('keeps coverage in [0, 1] for any distance and pixel size', async () => {
    const rows: number[][] = [];
    for (let d = -40; d <= 40; d += 0.37) for (const px of [0.25, 0.5, 1, 2]) rows.push([d, px]);
    const out = await gpuF32(
      `${include}
       fn f(x: vec4f) -> vec4f { return vec4f(waveline_coverage(x.x, x.y), waveline_band(x.x, 0.0, 2.0, x.y), 0.0, 0.0); }`,
      rowsOf(rows),
    );
    for (const v of [...lane(out, 0), ...lane(out, 1)]) {
      expect(v).toBeGreaterThanOrEqual(0);
      expect(v).toBeLessThanOrEqual(1);
    }
  });

  it('has the right sign: negative below the line (the fill), positive above', async () => {
    const out = await scalars(include, [
      'waveline_distance(10.0, 5.0, 0.0)', // point below a flat line at y = 5
      'waveline_distance(0.0, 5.0, 0.0)', // point above it
      'waveline_distance(5.0, 5.0, 0.7)', // on it
    ]);
    expect(out[0]).toBeCloseTo(-5, 6);
    expect(out[1]).toBeCloseTo(5, 6);
    expect(out[2]).toBe(0);
  });

  it('has unit gradient at the line, so one pixel across the edge is one pixel of distance', async () => {
    const h = 0.01;
    const rows: number[][] = [];
    for (let i = 0; i < 300; i++) rows.push([i * 3.7 + 0.5, i * 0.21, (i % 7) * 1.3, 1 + (i % 3)]);
    const out = await gpuF32(
      `${include}
       fn dist(x: f32, y: f32, phase: f32, seed: f32, amp: f32) -> f32 {
         let c = waveline_curve(x, phase, 180.0, amp, 0.7, seed);
         return waveline_distance(y, 60.0 + c.x, c.y);
       }
       fn f(a: vec4f) -> vec4f {
         let amp = 8.0 + a.w * 12.0;
         let c = waveline_curve(a.x, a.y, 180.0, amp, 0.7, a.z);
         let y0 = 60.0 + c.x;     // on the line
         let h = ${h};
         let gx = (dist(a.x + h, y0, a.y, a.z, amp) - dist(a.x - h, y0, a.y, a.z, amp)) / (2.0 * h);
         let gy = (dist(a.x, y0 + h, a.y, a.z, amp) - dist(a.x, y0 - h, a.y, a.z, amp)) / (2.0 * h);
         return vec4f(length(vec2f(gx, gy)), dist(a.x, y0, a.y, a.z, amp), 0.0, 0.0);
       }`,
      rowsOf(rows),
    );
    for (const g of lane(out, 0)) {
      expect(g).toBeGreaterThan(0.97);
      expect(g).toBeLessThan(1.03);
    }
    for (const d of lane(out, 1)) expect(Math.abs(d)).toBeLessThan(1e-3);
  });

  it('returns a slope that matches the curve, and stays inside its amplitude bound', async () => {
    const h = 0.3;
    const rows: number[][] = [];
    for (let i = 0; i < 300; i++) rows.push([i * 1.7, i * 0.19, i * 0.3, (i % 5) * 0.25]);
    const out = await gpuF32(
      `${include}
       fn f(a: vec4f) -> vec4f {
         let c = waveline_curve(a.x, a.y, 300.0, 12.0, a.w + 0.1, a.z);
         let lo = waveline_curve(a.x - ${h}, a.y, 300.0, 12.0, a.w + 0.1, a.z).x;
         let hi = waveline_curve(a.x + ${h}, a.y, 300.0, 12.0, a.w + 0.1, a.z).x;
         return vec4f(c.y, (hi - lo) / (2.0 * ${h}), c.x, 0.0);
       }`,
      rowsOf(rows),
    );
    const analytic = lane(out, 0);
    const numeric = lane(out, 1);
    analytic.forEach((slope, i) =>
      expect(Math.abs(slope - (numeric[i] ?? Number.NaN))).toBeLessThan(0.01),
    );
    for (const y of lane(out, 2)) expect(Math.abs(y)).toBeLessThanOrEqual(1.6 * 12.0 + 1e-3);
  });

  it('is periodic-ish: the slowest sine fixes the base wavelength, so the curve varies on that scale', async () => {
    // Two points a quarter wavelength apart differ; the curve is not constant.
    const out = await scalars(include, [
      'waveline_curve(0.0, 0.0, 100.0, 10.0, 0.0, 1.0).x',
      'waveline_curve(25.0, 0.0, 100.0, 10.0, 0.0, 1.0).x',
    ]);
    expect(Math.abs((out[0] ?? 0) - (out[1] ?? 0))).toBeGreaterThan(0.5);
  });

  it('keeps foam in [0, 1], solid at the crest and gone past its width', async () => {
    const rows: number[][] = [];
    for (let i = 0; i < 300; i++) rows.push([(i % 30) * 0.3, i * 1.7, i * 0.9, (i % 5) * 0.25]);
    const out = await gpuF32(
      `${include}
       fn f(a: vec4f) -> vec4f {
         let p = vec2f(a.y, a.z);
         return vec4f(
           waveline_foam(a.x, p, a.w, 6.0, 1.0, 0.15),
           waveline_foam(8.0, p, a.w, 6.0, 1.0, 0.15),
           waveline_foam(a.x, p, a.w, 6.0, 0.0, 0.15),
           0.0);
       }`,
      rowsOf(rows),
    );
    for (const v of lane(out, 0)) {
      expect(v).toBeGreaterThanOrEqual(0);
      expect(v).toBeLessThanOrEqual(1);
    }
    for (const v of lane(out, 1)) expect(v).toBe(0);
    for (const v of lane(out, 2)) expect(v).toBe(0);
  });
});

// --- wash.wgsl -----------------------------------------------------------------------------------------

describe('wash', () => {
  const include = '#include "wash.wgsl"\n';

  it('produces a smooth, bounded pigment field', async () => {
    const rows: number[][] = [];
    for (let i = 0; i < 400; i++)
      rows.push([(i % 20) * 0.37, Math.floor(i / 20) * 0.41, i * 0.05, (i % 4) * 0.5]);
    const out = await gpuF32(
      `${include}
       fn f(a: vec4f) -> vec4f { return vec4f(wash_field(vec2f(a.x, a.y), a.z, a.w * 1.5), 0.0); }`,
      rowsOf(rows),
    );
    for (let c = 0; c < 3; c++) {
      for (const v of lane(out, c)) {
        expect(Number.isFinite(v)).toBe(true);
        expect(Math.abs(v)).toBeLessThan(1.6);
      }
    }
    // The bloom field really varies (it is not a constant), with a mean near zero.
    const bloom = lane(out, 0);
    const mean = bloom.reduce((s, v) => s + v, 0) / bloom.length;
    const spread = Math.max(...bloom) - Math.min(...bloom);
    expect(Math.abs(mean)).toBeLessThan(0.3);
    expect(spread).toBeGreaterThan(0.5);
  });

  it('shapes blooms: coverage and pooled edge stay in [0, 1], and the rim hugs the inside of the edge', async () => {
    const rows: number[][] = [];
    for (let px = -30; px <= 60; px += 0.5) rows.push([px]);
    const out = await gpuF32(
      `${include}
       fn f(a: vec4f) -> vec4f {
         return vec4f(wash_body(a.x, 3.0), wash_pool(a.x, 8.0, 3.0), wash_granulate(wash_body(a.x, 3.0) * 0.6, 0.2, 0.5), 0.0);
       }`,
      rowsOf(rows),
    );
    const xs = rows.map((r) => r[0] ?? 0);
    const body = lane(out, 0);
    const pool = lane(out, 1);
    body.forEach((v, i) => {
      expect(v).toBeGreaterThanOrEqual(0);
      expect(v).toBeLessThanOrEqual(1);
      expect(pool[i]).toBeGreaterThanOrEqual(0);
      expect(pool[i]).toBeLessThanOrEqual(1);
      const x = xs[i] ?? 0;
      if (x <= -3) expect(v).toBe(0);
      if (x >= 3) expect(v).toBe(1);
      if (x <= -3) expect(pool[i]).toBe(0); // nothing outside the bloom
      if (x >= 3 + 8) expect(pool[i]).toBe(0); // nothing deep inside
    });
    // The rim peaks just inside the outline.
    const peak = pool.indexOf(Math.max(...pool));
    expect(xs[peak]).toBeGreaterThanOrEqual(0);
    expect(xs[peak]).toBeLessThanOrEqual(6);
    for (const v of lane(out, 2)) expect(v).toBeGreaterThanOrEqual(0);
  });

  it('granulation: neutral grain and zero strength change nothing; valleys hold more pigment', async () => {
    const out = await scalars(include, [
      'wash_granulate(0.4, 0.5, 1.0)', // neutral grain
      'wash_granulate(0.4, 0.1, 0.0)', // no strength
      'wash_granulate(0.4, 0.1, 1.0)', // valley (low grain): more
      'wash_granulate(0.4, 0.9, 1.0)', // peak (high grain): less
      'wash_granulate(0.0, 0.1, 1.0)', // no pigment, no granules
    ]);
    expect(out[0]).toBeCloseTo(0.4, 6);
    expect(out[1]).toBeCloseTo(0.4, 6);
    expect(out[2]).toBeGreaterThan(0.4);
    expect(out[3]).toBeLessThan(0.4);
    expect(out[4]).toBe(0);
  });

  it('bakes a seamless paper tile with the right statistics', async () => {
    const rows: number[][] = [];
    for (let i = 0; i < 400; i++) rows.push([(i * 0.137) % 1, (i * 0.0713) % 1]);
    const out = await gpuF32(
      `${include}
       fn f(a: vec4f) -> vec4f {
         let p = wash_paper_tile(vec2f(a.x, a.y));
         let shifted = wash_paper_tile(vec2f(a.x + 1.0, a.y + 1.0));
         return vec4f(p, abs(p.x - shifted.x) + abs(p.y - shifted.y), 0.0);
       }`,
      rowsOf(rows),
    );
    const grain = lane(out, 0);
    const fibre = lane(out, 1);
    for (const v of [...grain, ...fibre]) {
      expect(v).toBeGreaterThanOrEqual(0);
      expect(v).toBeLessThanOrEqual(1);
    }
    for (const e of lane(out, 2)) expect(e).toBeLessThan(1e-3); // the tile repeats
    const mean = grain.reduce((s, v) => s + v, 0) / grain.length;
    expect(mean).toBeGreaterThan(0.4);
    expect(mean).toBeLessThan(0.6);
    // Fibres are rare: mostly zero.
    expect(fibre.filter((v) => v > 0.2).length / fibre.length).toBeLessThan(0.35);
  });

  it('reconstructs a texture with the cubic B-spline exactly where the data is linear', async () => {
    const device = await getTestDevice();
    // 8x8 rgba8unorm ramp along x: texel i holds i * 20 / 255; y has its own ramp in g.
    const data = new Uint8Array(8 * 8 * 4);
    for (let y = 0; y < 8; y++) {
      for (let x = 0; x < 8; x++) data.set([x * 20, y * 20, 100, 255], (y * 8 + x) * 4);
    }
    const texture = device.createTexture({
      size: [8, 8],
      format: 'rgba8unorm',
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
    });
    device.queue.writeTexture({ texture }, data, { bytesPerRow: 32 }, [8, 8]);
    const sampler = device.createSampler({ magFilter: 'linear', minFilter: 'linear' });

    const uvs = [
      [0.3, 0.4],
      [0.5, 0.5],
      [0.62, 0.35],
      [0.45, 0.7],
    ];
    const { createShader } = await import('../src/gpu/shader.ts');
    const module = createShader(device, {
      label: 'bspline-test.wgsl',
      code: `#include "wash.wgsl"
        @group(0) @binding(0) var tex: texture_2d<f32>;
        @group(0) @binding(1) var samp: sampler;
        @group(0) @binding(2) var<storage, read> uvs: array<vec2f>;
        @group(0) @binding(3) var<storage, read_write> out: array<vec4f>;
        @compute @workgroup_size(4) fn main(@builtin(global_invocation_id) id: vec3u) {
          out[id.x] = wash_sample_bspline(tex, samp, uvs[id.x]);
        }`,
    });
    const uvBuffer = device.createBuffer({
      size: 64,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(uvBuffer, 0, new Float32Array(uvs.flat()));
    const outBuffer = device.createBuffer({
      size: 64,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });
    const readBuffer = device.createBuffer({
      size: 64,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    const pipeline = device.createComputePipeline({
      layout: 'auto',
      compute: { module, entryPoint: 'main' },
    });
    const group = device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: texture.createView() },
        { binding: 1, resource: sampler },
        { binding: 2, resource: { buffer: uvBuffer } },
        { binding: 3, resource: { buffer: outBuffer } },
      ],
    });
    const encoder = device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, group);
    pass.dispatchWorkgroups(1);
    pass.end();
    encoder.copyBufferToBuffer(outBuffer, 0, readBuffer, 0, 64);
    device.queue.submit([encoder.finish()]);
    await readBuffer.mapAsync(GPUMapMode.READ);
    const result = new Float32Array(readBuffer.getMappedRange().slice(0));
    readBuffer.unmap();

    uvs.forEach(([u, v], i) => {
      // Linear data: texel index at uv u is u * 8 - 0.5, so the value is that times 20 / 255.
      const expectedR = ((u ?? 0) * 8 - 0.5) * (20 / 255);
      const expectedG = ((v ?? 0) * 8 - 0.5) * (20 / 255);
      expect(result[i * 4]).toBeCloseTo(expectedR, 2);
      expect(result[i * 4 + 1]).toBeCloseTo(expectedG, 2);
      expect(result[i * 4 + 2]).toBeCloseTo(100 / 255, 4); // a constant stays constant
    });
    for (const buffer of [uvBuffer, outBuffer, readBuffer]) buffer.destroy();
    texture.destroy();
  });
});

// --- mist.wgsl -----------------------------------------------------------------------------------------

describe('mist', () => {
  const include = '#include "mist.wgsl"\n';

  it('keeps layer density in [0, 1] and uses the range', async () => {
    const rows: number[][] = [];
    for (let i = 0; i < 600; i++)
      rows.push([(i % 30) * 0.31, Math.floor(i / 30) * 0.29, i * 0.03, (i % 5) * 0.25]);
    const out = await gpuF32(
      `${include}
       fn f(a: vec4f) -> vec4f {
         let p = vec2f(a.x, a.y);
         return vec4f(
           mist_layer(p, vec2f(a.z, 0.0), vec2f(0.0, a.z * 0.6), a.z, a.w * 1.5, 0.55),
           mist_layer(p, vec2f(0.0), vec2f(0.0), a.z, 1.5, 1.0),
           mist_layer(p, vec2f(0.0), vec2f(0.0), a.z, 0.0, 0.0),
           0.0);
       }`,
      rowsOf(rows),
    );
    for (const c of [0, 1, 2]) {
      for (const v of lane(out, c)) {
        expect(v).toBeGreaterThanOrEqual(0);
        expect(v).toBeLessThanOrEqual(1);
      }
    }
    const typical = lane(out, 0);
    expect(Math.max(...typical)).toBeGreaterThan(0.4);
    expect(Math.min(...typical)).toBe(0); // there are clear gaps between wisps
    // More coverage means more mist.
    const mean = (values: number[]) => values.reduce((s, v) => s + v, 0) / values.length;
    expect(mean(lane(out, 1))).toBeGreaterThan(mean(lane(out, 2)));
  });

  it('thins out away from the source edge', async () => {
    const rows: number[][] = [];
    for (let s = 0; s <= 1.0001; s += 0.02) rows.push([s]);
    const out = await gpuF32(
      `${include}
       fn f(a: vec4f) -> vec4f { return vec4f(mist_profile(a.x, 0.5, 0.3), mist_profile(a.x, 0.9, 0.02), 0.0, 0.0); }`,
      rowsOf(rows),
    );
    for (const c of [0, 1]) {
      const profile = lane(out, c);
      profile.forEach((v, i) => {
        expect(v).toBeGreaterThanOrEqual(0);
        expect(v).toBeLessThanOrEqual(1);
        if (i > 0) expect(v).toBeLessThanOrEqual((profile[i - 1] ?? 1) + 1e-6);
      });
      expect(profile[0]).toBe(1);
    }
    expect(lane(out, 0).at(-1)).toBe(0);
  });

  it('never exceeds the opacity cap, however dense, and is monotonic', async () => {
    const rows: number[][] = [];
    for (const cap of [0.1, 0.38, 0.8, 1.5]) {
      for (const d of [-5, 0, 0.05, 0.2, 0.5, 0.9, 1, 3, 100, 1e6]) rows.push([d, cap]);
    }
    const out = await gpuF32(
      `${include}
       fn f(a: vec4f) -> vec4f { return vec4f(mist_opacity(a.x, a.y), MIST_ALPHA_LIMIT, 0.0, 0.0); }`,
      rowsOf(rows),
    );
    const alpha = lane(out, 0);
    rows.forEach((row, i) => {
      const cap = Math.min(row[1] ?? 0, 0.8);
      expect(alpha[i]).toBeGreaterThanOrEqual(0);
      expect(alpha[i]).toBeLessThanOrEqual(cap + 1e-6);
      if ((row[0] ?? 0) <= 0) expect(alpha[i]).toBe(0);
      if ((row[0] ?? 0) >= 1) expect(alpha[i]).toBeCloseTo(cap, 5);
    });
    // Within one cap, opacity never falls as density rises.
    for (let c = 0; c < 4; c++) {
      for (let k = 1; k < 10; k++) {
        expect(alpha[c * 10 + k]).toBeGreaterThanOrEqual((alpha[c * 10 + k - 1] ?? 0) - 1e-7);
      }
    }
    expect(lane(out, 1)[0]).toBeCloseTo(0.8, 6);
  });
});

// --- Demo effects ---------------------------------------------------------------------------------------

const DEMOS = ['wave-divider', 'ink-wash', 'mist'] as const;

describe('decoration demos are premultiplied: rgb never exceeds alpha', () => {
  const EPS = 1.5 / 255;
  const premultiplied = (r: number, g: number, b: number, a: number) =>
    a >= -EPS && a <= 1 + EPS && Math.max(r, g, b) <= a + EPS && Math.min(r, g, b) >= -EPS;

  it.each([
    ['wave-divider', undefined, {}],
    ['wave-divider', 'stormy break', {}],
    ['wave-divider', 'pastel', {}],
    ['wave-divider', 'section cap', {}],
    ['wave-divider', undefined, { backAlpha: 0.2, layers: 5, foam: 1, amplitude: 0.3 }],
    ['mist', undefined, {}],
    ['mist', 'hot spring steam', {}],
    ['mist', 'low fog', {}],
    ['mist', 'top haze', {}],
    ['mist', undefined, { density: 2, maxOpacity: 0.8, coverage: 1, tint: [1, 1, 1] }],
    ['ink-wash', undefined, { transparentPaper: true }],
    ['ink-wash', 'ink over page', {}],
    [
      'ink-wash',
      undefined,
      { transparentPaper: true, contrast: 1.5, edgeDarkening: 1.2, coverage: 0.9 },
    ],
  ] as const)('%s / %s / %j (8-bit)', async (id, preset, overrides) => {
    const image = await render(id, { ...overrides }, preset, {
      target: 'byte',
      width: 160,
      height: 96,
      dpr: 2,
      advance: 3,
    });
    expect(firstBad(image, premultiplied)).toBeUndefined();
  });

  it('the divider is transparent above the waves and solid below them, flipped or not', async () => {
    for (const flip of [false, true]) {
      const image = await render('wave-divider', { flip, matchPage: false }, undefined, {
        target: 'byte',
        width: 128,
        height: 80,
      });
      const alphaAt = (x: number, y: number) =>
        image.data[(y * image.width + x) * 4 + 3] ?? Number.NaN;
      const open = flip ? 79 : 0;
      const solid = flip ? 0 : 79;
      for (let x = 0; x < 128; x += 9) {
        expect(alphaAt(x, open)).toBe(0);
        expect(alphaAt(x, solid)).toBe(1);
      }
    }
  });

  it.each(['#ffffff', '#f3f6f7', '#0b1c29', '#7a3b10', '#00ff7f'])(
    'the front layer is exactly the requested sRGB colour (%s)',
    async (css) => {
      const image = await render(
        'wave-divider',
        { matchPage: false, colorFront: hexToLinear(css) },
        undefined,
        {
          target: 'byte',
          width: 128,
          height: 80,
          dpr: 2,
          advance: 5,
        },
      );
      const want = [1, 3, 5].map((i) => parseInt(css.slice(i, i + 2), 16));
      // The bottom rows are well below the front crest: every pixel is the flat colour, alpha 1.
      for (let y = 70; y < 80; y++) {
        for (let x = 0; x < 128; x += 7) {
          const at = (y * 128 + x) * 4;
          const got = [0, 1, 2].map((c) => Math.round((image.data[at + c] ?? 0) * 255));
          expect(got).toEqual(want);
          expect(image.data[at + 3]).toBe(1);
        }
      }
    },
  );

  it('the mist alpha never exceeds its documented cap, at default settings or the extremes', async () => {
    const maxAlpha = (image: Rendered) => {
      let max = 0;
      for (let i = 3; i < image.data.length; i += 4) max = Math.max(max, image.data[i] ?? 0);
      return max;
    };
    const effect = await loadEffect('mist');
    const cap = Number(defaultParams(effect.params).maxOpacity);
    const base = await render('mist', {}, undefined, {
      target: 'byte',
      width: 200,
      height: 120,
      advance: 4,
    });
    expect(maxAlpha(base)).toBeLessThanOrEqual(cap + 2 / 255);
    expect(maxAlpha(base)).toBeGreaterThan(0.05);
    for (const preset of Object.keys(effect.presets ?? {})) {
      const presetCap = Number(effect.presets?.[preset]?.maxOpacity ?? cap);
      const image = await render('mist', {}, preset, {
        target: 'byte',
        width: 200,
        height: 120,
        advance: 4,
      });
      expect(maxAlpha(image)).toBeLessThanOrEqual(presetCap + 2 / 255);
    }
    const extreme = await render('mist', { density: 2, maxOpacity: 0.8, coverage: 1 }, undefined, {
      target: 'byte',
      width: 200,
      height: 120,
      advance: 4,
    });
    expect(maxAlpha(extreme)).toBeLessThanOrEqual(0.8 + 2 / 255);
    // Even a request above the limit is held to it.
    const over = await render('mist', { density: 2, maxOpacity: 5, coverage: 1 }, undefined, {
      target: 'byte',
      width: 200,
      height: 120,
    });
    expect(maxAlpha(over)).toBeLessThanOrEqual(0.8 + 2 / 255);
  });

  it('the mist is clear away from its source edge by default (it thins out)', async () => {
    const image = await render('mist', { from: 'bottom' }, undefined, {
      target: 'byte',
      width: 160,
      height: 100,
      advance: 4,
    });
    const rowMean = (y: number) => {
      let sum = 0;
      for (let x = 0; x < 160; x++) sum += image.data[(y * 160 + x) * 4 + 3] ?? 0;
      return sum / 160;
    };
    const top = (rowMean(0) + rowMean(1) + rowMean(2)) / 3;
    const bottom = (rowMean(97) + rowMean(98) + rowMean(99)) / 3;
    expect(top).toBeLessThan(bottom);
  });
});

describe('ink wash is a low-contrast background', () => {
  const TEXT = { light: '#0d1b22', dark: '#d8eef7' } as const;

  /** Contrast of the page's text colour against the worst pixel, and the swing from darkest to lightest. */
  async function legibility(overrides: Record<string, unknown>, preset?: string) {
    const params = {
      ...defaultParams((await loadEffect('ink-wash')).params),
      ...((await loadEffect('ink-wash')).presets?.[preset ?? ''] ?? {}),
      ...overrides,
    };
    const image = await render('ink-wash', overrides, preset, {
      target: 'byte',
      width: 240,
      height: 150,
      advance: 30,
    });
    let lo = Infinity;
    let hi = -Infinity;
    for (let i = 0; i < image.data.length; i += 4) {
      const l = luminance(
        (image.data[i] ?? 0) * 255,
        (image.data[i + 1] ?? 0) * 255,
        (image.data[i + 2] ?? 0) * 255,
      );
      lo = Math.min(lo, l);
      hi = Math.max(hi, l);
    }
    // Dark text goes on light paper, light text on dark paper.
    const css = TEXT[params.tone === 'dark' ? 'dark' : 'light'];
    const rgb = [1, 3, 5].map((k) => parseInt(css.slice(k, k + 2), 16)) as [number, number, number];
    const worst = params.tone === 'dark' ? hi : lo;
    return { text: contrast(luminance(...rgb), worst), range: contrast(hi, lo) };
  }

  it.each(['indigo', 'tea', 'storm', 'rose', 'moss'])(
    'default %s keeps body text above 7:1 (WCAG AAA) on both tones',
    async (palette) => {
      for (const tone of ['light', 'dark']) {
        const { text, range } = await legibility({ palette, tone });
        expect(text, `${palette} ${tone}`).toBeGreaterThan(7);
        expect(range, `${palette} ${tone} range`).toBeLessThan(2.1);
      }
    },
  );

  it('every preset keeps body text above 4.5:1 (WCAG AA)', async () => {
    const effect = await loadEffect('ink-wash');
    for (const name of Object.keys(effect.presets ?? {})) {
      // The "over page" preset is transparent: its legibility depends on the page it is laid on.
      if (effect.presets?.[name]?.transparentPaper) continue;
      const { text } = await legibility({}, name);
      expect(text, name).toBeGreaterThan(4.5);
    }
  });
});

describe('decoration demos stay finite across their parameter ranges', () => {
  it.each(DEMOS)('%s', async (id) => {
    const effect = await loadEffect(id);
    const WIDTH = 80;
    const HEIGHT = 56;
    const check = async (label: string, overrides: Record<string, unknown>, preset?: string) => {
      const image = await render(id, overrides, preset, {
        width: WIDTH,
        height: HEIGHT,
        dpr: 2,
        advance: 2,
      });
      const bad = firstBad(image, (r, g, b, a) =>
        [r, g, b, a].every((v) => Number.isFinite(v) && v >= -0.01 && v <= 1.02),
      );
      if (bad) throw new Error(`${id}: ${label}: ${bad}`);
    };

    await check('defaults', {});
    for (const name of Object.keys(effect.presets ?? {})) await check(`preset ${name}`, {}, name);
    for (const [key, def] of Object.entries(effect.params)) {
      const values: unknown[] =
        def.type === 'number'
          ? [def.min, def.max, (def.min + def.max) / 2]
          : def.type === 'boolean'
            ? [true, false]
            : def.type === 'select'
              ? [...def.options]
              : [
                  [0, 0, 0],
                  [1, 1, 1],
                ];
      for (const value of values)
        await check(`${key} = ${JSON.stringify(value)}`, { [key]: value });
    }
    const extremes: Record<string, unknown> = {};
    for (const [key, def] of Object.entries(effect.params)) {
      if (def.type === 'number') extremes[key] = def.max;
    }
    await check('every number at its maximum', extremes);
    const lows: Record<string, unknown> = {};
    for (const [key, def] of Object.entries(effect.params)) {
      if (def.type === 'number') lows[key] = def.min;
    }
    await check('every number at its minimum', lows);
  });

  it('has at least two presets, an epic of 4, and honest scorecards', async () => {
    for (const id of DEMOS) {
      const effect = await loadEffect(id);
      expect(Object.keys(effect.presets ?? {}).length).toBeGreaterThanOrEqual(2);
      expect(effect.meta.epic).toBe('4');
      expect(effect.meta.tier).toBe('A');
      expect(effect.meta.scorecard?.fallback.length).toBeGreaterThan(20);
      expect(effect.meta.scorecard).not.toHaveProperty('gpuMs');
      expect(effect.meta.scorecard).not.toHaveProperty('bundleKb');
    }
  });
});
