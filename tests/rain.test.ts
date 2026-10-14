import { describe, expect, it, vi } from 'vitest';
import { getTestDevice } from './gpu.ts';
import { gpuF32, lane, rowsOf, scalars } from './wgsl-eval.ts';

// Time-series tests run thousands of steps per GPU thread; leave headroom for a busy machine.
vi.setConfig({ testTimeout: 90_000 });

const DROPLETS = '#include "droplets.wgsl"\n';
const RIPPLES = '#include "ripples.wgsl"\n';
const WETNESS = '#include "wetness.wgsl"\n#include "color.wgsl"\n';

/** `count` rows of 4 numbers, each from `make(i)`. */
function rows(count: number, make: (i: number) => number[]): Float32Array {
  return rowsOf(Array.from({ length: count }, (_, i) => make(i)));
}

/** Deterministic pseudo-random sequence in [0, 1) so failures reproduce. */
function sequence(seed: number): () => number {
  let state = seed >>> 0;
  return () => {
    state = (state + 0x6d2b79f5) >>> 0;
    let t = state;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function max(values: number[]): number {
  return values.reduce((a, b) => Math.max(a, b), -Infinity);
}

function min(values: number[]): number {
  return values.reduce((a, b) => Math.min(a, b), Infinity);
}

function allFinite(values: ArrayLike<number>): boolean {
  for (let i = 0; i < values.length; i++)
    if (!Number.isFinite(values[i] ?? Number.NaN)) return false;
  return true;
}

// --- Droplets ---------------------------------------------------------------------------------

describe('droplets', () => {
  // Random point, time, rain, size and layer count from four random numbers per row.
  const random = sequence(11);
  const N = 20_000;
  const inputs = rows(N, () => [random() * 5, random() * 3, random() * 300, random() * 1000]);
  const setup = `
    fn params(x: vec4f) -> vec4f {
      let r = hash_1to2(x.w);
      return vec4f(r.x, mix(0.05, 0.2, r.y), 1.0 + floor(hash_1to1(x.w + 1.0) * 3.0), 0.004);
    }`;

  it('masks stay in [0, 1], the wipe in [0, 1], and drops stay inside their own disc', async () => {
    const out = await gpuF32(
      `${DROPLETS}${setup}
       fn f(x: vec4f) -> vec4f {
         let q = params(x);
         let s = droplet_field(x.xy, x.z, q.x, q.y, i32(q.z), q.w);
         // Beyond the unit disc by more than the anti-aliased rim allows is a bug.
         let over = max(length(s.local) - 1.0, 0.0) * s.radius - 0.5 * q.w;
         return vec4f(s.mask, s.clear, over, s.radius);
       }`,
      inputs,
    );
    expect(allFinite(out)).toBe(true);
    expect(min(lane(out, 0))).toBeGreaterThanOrEqual(0);
    expect(max(lane(out, 0))).toBeLessThanOrEqual(1);
    expect(min(lane(out, 1))).toBeGreaterThanOrEqual(0);
    expect(max(lane(out, 1))).toBeLessThanOrEqual(1);
    expect(max(lane(out, 2))).toBeLessThanOrEqual(1e-4);
    expect(min(lane(out, 3))).toBeGreaterThanOrEqual(0);
    // The sample is not vacuous: plenty of pixels are drops, plenty are not.
    const covered = lane(out, 0).filter((m) => m > 0.5).length / N;
    expect(covered).toBeGreaterThan(0.01);
    expect(covered).toBeLessThan(0.6);
  });

  it('normals are finite and unit length, and outside a drop point straight out', async () => {
    const out = await gpuF32(
      `${DROPLETS}${setup}
       fn f(x: vec4f) -> vec4f {
         let q = params(x);
         let s = droplet_field(x.xy, x.z, q.x, q.y, i32(q.z), q.w);
         let n = droplet_normal(s, mix(0.3, 1.0, fract(x.w)));
         let flat = select(0.0, 1.0, s.mask <= 0.0 && n.z < 0.9999);
         return vec4f(n.xy, length(n), flat);
       }`,
      inputs,
    );
    expect(allFinite(out)).toBe(true);
    for (const length of lane(out, 2)) expect(Math.abs(length - 1)).toBeLessThan(1e-4);
    expect(max(lane(out, 3))).toBe(0);
  });

  it('is deterministic: the same inputs give the same bits', async () => {
    const code = `${DROPLETS}${setup}
      fn f(x: vec4f) -> vec4f {
        let q = params(x);
        let s = droplet_field(x.xy, x.z, q.x, q.y, i32(q.z), q.w);
        return vec4f(s.mask, s.clear, s.radius, s.local.x + 3.0 * s.local.y);
      }`;
    const a = await gpuF32(code, inputs);
    const b = await gpuF32(code, inputs);
    expect(Array.from(b)).toEqual(Array.from(a));
  });

  it('is continuous in time, also across drop resets', async () => {
    // 400 points, 20000 steps of 1/240 s (83 s: several full cycles of every cell, 6-14 s each).
    const place = sequence(5);
    const points = rows(400, () => [place() * 4, place() * 3, place() * 40, 0]);
    const out = await gpuF32(
      `${DROPLETS}
       fn f(x: vec4f) -> vec4f {
         var prev = droplet_field(x.xy, x.z, 1.0, 0.085, 3, 0.004);
         var d_mask = 0.0;
         var d_clear = 0.0;
         var peak = 0.0;
         for (var i = 1; i <= 20000; i++) {
           let s = droplet_field(x.xy, x.z + f32(i) / 240.0, 1.0, 0.085, 3, 0.004);
           d_mask = max(d_mask, abs(s.mask - prev.mask));
           d_clear = max(d_clear, abs(s.clear - prev.clear));
           peak = max(peak, s.mask);
           prev = s;
         }
         return vec4f(d_mask, d_clear, peak, 0.0);
       }`,
      points,
    );
    expect(allFinite(out)).toBe(true);
    // Edge speed is at most ~0.15 image heights per second: 0.15 / 240 / 0.004 (one pixel) = 0.16 of
    // coverage per step. A drop that appeared or vanished in one step would change it by up to 1.
    expect(max(lane(out, 0))).toBeLessThan(0.3);
    expect(max(lane(out, 1))).toBeLessThan(0.3);
    // The points do see drops, so the bound above is not vacuous.
    expect(lane(out, 2).filter((p) => p > 0.9).length).toBeGreaterThan(40);
  });

  it('shows nothing on dry glass and more drops as the rain increases', async () => {
    const place = sequence(9);
    const points = rows(6000, () => [place() * 6, place() * 3, place() * 100, 0]);
    const coverage = async (rain: number) => {
      const out = await gpuF32(
        `${DROPLETS}
         fn f(x: vec4f) -> vec4f {
           let s = droplet_field(x.xy, x.z, ${rain}, 0.085, 3, 0.004);
           return vec4f(s.mask, s.clear, 0.0, 0.0);
         }`,
        points,
      );
      const masks = lane(out, 0).slice(0, points.length / 4);
      const clears = lane(out, 1).slice(0, points.length / 4);
      return {
        drops: masks.filter((m) => m > 0.5).length / masks.length,
        wiped: clears.filter((c) => c > 0.5).length / clears.length,
      };
    };
    const dry = await coverage(0);
    const drizzle = await coverage(0.25);
    const rain = await coverage(0.6);
    const storm = await coverage(1);
    expect(dry.drops).toBe(0);
    expect(dry.wiped).toBe(0);
    expect(drizzle.drops).toBeGreaterThan(0);
    expect(rain.drops).toBeGreaterThan(drizzle.drops);
    expect(storm.drops).toBeGreaterThan(rain.drops);
    expect(storm.wiped).toBeGreaterThan(drizzle.wiped);
  });

  it('the lens flips the picture: the offset points away from where the pixel is in the drop', async () => {
    const values = await scalars(DROPLETS, [
      'droplet_lens(DropletSample(1.0, vec2f(0.5, -0.25), 2.0, 1.0), 3.0).x',
      'droplet_lens(DropletSample(1.0, vec2f(0.5, -0.25), 2.0, 1.0), 3.0).y',
      'droplet_lens(DropletSample(0.0, vec2f(0.0), 0.0, 0.0), 3.0).x',
      'droplet_rim(DropletSample(1.0, vec2f(0.0), 1.0, 1.0))',
      'droplet_rim(DropletSample(1.0, vec2f(1.0, 0.0), 1.0, 1.0))',
      'droplet_highlight(vec3f(0.0, 0.0, 1.0), normalize(vec3f(0.0, 0.0, 1.0)), 50.0)',
      'droplet_highlight(vec3f(0.0, 0.0, 1.0), normalize(vec3f(1.0, 0.0, 0.1)), 50.0)',
    ]);
    expect(values[0]).toBeCloseTo(-3, 5);
    expect(values[1]).toBeCloseTo(1.5, 5);
    expect(values[2]).toBeCloseTo(0, 6);
    expect(values[3]).toBeCloseTo(0, 6);
    expect(values[4]).toBeCloseTo(1, 5);
    expect(values[5]).toBeCloseTo(1, 5);
    expect(values[6]).toBeLessThan(0.01);
  });

  it('the stop-and-go progress of a running drop dwells but never goes backwards', async () => {
    // _droplet_stopgo is the curve a drop's height follows over its run: a line minus two sines.
    const out = await gpuF32(
      `${DROPLETS}
       fn f(x: vec4f) -> vec4f {
         let n1 = 2.0 + floor(x.x * 3.0);
         let n2 = 5.0 + floor(x.y * 3.0);
         var prev = _droplet_stopgo(0.0, n1, x.z, n2, x.w);
         var backwards = 0.0;
         var slowest = 10.0;
         for (var i = 1; i <= 2000; i++) {
           let r = _droplet_stopgo(f32(i) / 2000.0, n1, x.z, n2, x.w);
           backwards = max(backwards, prev.x - r.x);
           slowest = min(slowest, r.y);
           prev = r;
         }
         return vec4f(backwards, prev.x, slowest, _droplet_stopgo(0.0, n1, x.z, n2, x.w).x);
       }`,
      (() => {
        const random = sequence(3);
        return rows(500, () => [
          random(),
          random(),
          random() * Math.PI * 2,
          random() * Math.PI * 2,
        ]);
      })(),
    );
    expect(allFinite(out)).toBe(true);
    expect(max(lane(out, 0))).toBeLessThan(1e-6); // never backwards
    for (const end of lane(out, 1)) expect(Math.abs(end - 1)).toBeLessThan(1e-4); // arrives
    for (const start of lane(out, 3)) expect(Math.abs(start)).toBeLessThan(1e-6); // starts at 0
    expect(min(lane(out, 2))).toBeGreaterThanOrEqual(0.04); // and its speed is never below ~0.05
    expect(min(lane(out, 2))).toBeLessThan(0.3); // yet it does come close to a pause
  });
});

// --- Ripples ----------------------------------------------------------------------------------

describe('ripple rings', () => {
  it('one ring: height within [-1, 1], zero before impact, after fading and beyond the reach', async () => {
    const random = sequence(21);
    const inputs = rows(20_000, () => [
      random() * 0.7,
      random(),
      0.5 + 0.5 * random(),
      random() * Math.PI * 2,
    ]);
    const out = await gpuF32(
      `${RIPPLES}
       fn f(x: vec4f) -> vec4f {
         let off = vec2f(cos(x.w), sin(x.w)) * x.x;
         let r = ripple_ring(off, x.y, x.z);
         return r.xyzz;
       }`,
      inputs,
    );
    expect(allFinite(out)).toBe(true);
    expect(max(lane(out, 0).map(Math.abs))).toBeLessThanOrEqual(1 + 1e-5);
    // Beyond the reach there is exactly nothing.
    const reach = (await scalars(RIPPLES, ['RIPPLE_REACH']))[0]!;
    const rowCount = inputs.length / 4;
    for (let i = 0; i < rowCount; i++) {
      if (inputs[i * 4]! >= reach) {
        expect(out[i * 4]).toBe(0);
        expect(out[i * 4 + 1]).toBe(0);
      }
    }
    // Before impact, at it, after the end, and just before the end.
    const checks = await scalars(RIPPLES, [
      'ripple_ring(vec2f(0.1, 0.0), -0.2, 1.0).x',
      'ripple_ring(vec2f(0.1, 0.0), 0.0, 1.0).x',
      'ripple_ring(vec2f(0.1, 0.0), 1.0, 1.0).x',
      'ripple_ring(vec2f(0.1, 0.0), 1.7, 1.0).x',
      'ripple_ring(vec2f(0.3, 0.0), 0.0, 1.0).x',
      'length(ripple_ring(vec2f(0.3, 0.0), 0.999, 1.0))',
    ]);
    expect(checks.slice(0, 5)).toEqual([0, 0, 0, 0, 0]);
    expect(checks[5]).toBeLessThan(1e-3);
  });

  it('a ring is radially symmetric about its centre and its slope points along the radius', async () => {
    const random = sequence(31);
    const inputs = rows(8000, () => [
      0.02 + random() * 0.45,
      0.02 + random() * 0.96,
      random() * Math.PI * 2,
      random() * Math.PI * 2,
    ]);
    const out = await gpuF32(
      `${RIPPLES}
       fn f(x: vec4f) -> vec4f {
         let a = ripple_ring(vec2f(cos(x.z), sin(x.z)) * x.x, x.y, 0.8);
         let b = ripple_ring(vec2f(cos(x.w), sin(x.w)) * x.x, x.y, 0.8);
         // Slope component perpendicular to the radius, relative to the slope's size.
         let radial = vec2f(cos(x.z), sin(x.z));
         let across = abs(a.y * radial.y - a.z * radial.x);
         return vec4f(abs(a.x - b.x), across, length(a.yz), 0.0);
       }`,
      inputs,
    );
    expect(allFinite(out)).toBe(true);
    expect(max(lane(out, 0))).toBeLessThan(1e-4);
    const across = lane(out, 1);
    const size = lane(out, 2);
    for (let i = 0; i < across.length; i++)
      expect(across[i]!).toBeLessThan(1e-4 * Math.max(1, size[i]!));
  });

  it('the analytic slope matches finite differences of the height', async () => {
    const random = sequence(41);
    const inputs = rows(8000, () => [random() * 4, random() * 4, 0.3 + random() * 8, 0]);
    const out = await gpuF32(
      `${RIPPLES}
       fn f(x: vec4f) -> vec4f {
         let e = 0.0004;
         let r = ripple_rings(x.xy, x.z, 1.0, 1.1, 0);
         let hx = ripple_rings(x.xy + vec2f(e, 0.0), x.z, 1.0, 1.1, 0).x - ripple_rings(x.xy - vec2f(e, 0.0), x.z, 1.0, 1.1, 0).x;
         let hy = ripple_rings(x.xy + vec2f(0.0, e), x.z, 1.0, 1.1, 0).x - ripple_rings(x.xy - vec2f(0.0, e), x.z, 1.0, 1.1, 0).x;
         return vec4f(r.y, r.z, hx / (2.0 * e), hy / (2.0 * e));
       }`,
      inputs,
    );
    expect(allFinite(out)).toBe(true);
    let worst = 0;
    let steep = 0;
    for (let i = 0; i < out.length; i += 4) {
      steep = Math.max(steep, Math.abs(out[i]!), Math.abs(out[i + 1]!));
      worst = Math.max(worst, Math.abs(out[i]! - out[i + 2]!), Math.abs(out[i + 1]! - out[i + 3]!));
    }
    // Most of the error is float rounding in the height differences; a wrong formula is off by tens.
    expect(worst).toBeLessThan(1.5);
    expect(steep).toBeGreaterThan(20);
  });

  it('is continuous across cell borders', async () => {
    // Walk along lines that cross three vertical cell borders in steps of 0.0004 cells and compare the
    // largest step across a border with the largest step anywhere. The steepest slope is ~70 per cell,
    // so a step of 0.0004 changes the height by at most ~0.03; a seam would jump by tenths.
    const out = await gpuF32(
      `${RIPPLES}
       fn f(x: vec4f) -> vec4f {
         var prev = ripple_rings(vec2f(0.7, x.y), x.z, 1.0, 1.1, 0).x;
         var border = 0.0;
         var inside = 0.0;
         var seen = 0.0;
         for (var i = 1; i <= 6000; i++) {
           let px = 0.7 + f32(i) * 0.0004;
           let h = ripple_rings(vec2f(px, x.y), x.z, 1.0, 1.1, 0).x;
           let step_size = abs(h - prev);
           if (floor(px) != floor(px - 0.0004)) { border = max(border, step_size); } else { inside = max(inside, step_size); }
           seen = max(seen, abs(h));
           prev = h;
         }
         return vec4f(border, inside, seen, 0.0);
       }`,
      (() => {
        const random = sequence(51);
        return rows(600, () => [0, random() * 6, random() * 20, 0]);
      })(),
    );
    expect(allFinite(out)).toBe(true);
    expect(max(lane(out, 0))).toBeLessThan(0.05);
    expect(max(lane(out, 1))).toBeLessThan(0.05);
    // Rings reach the border region at all (the test is not vacuous).
    expect(max(lane(out, 2))).toBeGreaterThan(0.3);
  });

  it('is continuous across the borders in y too, and from a cell to its diagonal neighbour', async () => {
    const out = await gpuF32(
      `${RIPPLES}
       fn f(x: vec4f) -> vec4f {
         // A diagonal line through the corner of four cells.
         var prev = ripple_rings(vec2f(0.8 + x.x, 0.8 + x.y), x.z, 1.0, 1.1, 3).x;
         var worst = 0.0;
         for (var i = 1; i <= 2000; i++) {
           let q = 0.8 + f32(i) * 0.0003;
           let h = ripple_rings(vec2f(q + x.x, q + x.y), x.z, 1.0, 1.1, 3).x;
           worst = max(worst, abs(h - prev));
           prev = h;
         }
         return vec4f(worst, 0.0, 0.0, 0.0);
       }`,
      (() => {
        const random = sequence(61);
        return rows(400, () => [
          Math.floor(random() * 4),
          Math.floor(random() * 4),
          random() * 30,
          0,
        ]);
      })(),
    );
    expect(max(lane(out, 0))).toBeLessThan(0.06);
  });

  it('is continuous in time, also across a ring being replaced by the next one', async () => {
    const random = sequence(71);
    const out = await gpuF32(
      `${RIPPLES}
       fn f(x: vec4f) -> vec4f {
         var prev = ripple_rings(x.xy, x.z, 1.0, 1.1, 0);
         var worst = 0.0;
         var peak = 0.0;
         for (var i = 1; i <= 6000; i++) {
           let r = ripple_rings(x.xy, x.z + f32(i) / 240.0, 1.0, 1.1, 0);
           worst = max(worst, abs(r.x - prev.x));
           peak = max(peak, abs(r.x));
           prev = r;
         }
         return vec4f(worst, peak, 0.0, 0.0);
       }`,
      rows(400, () => [random() * 5, random() * 5, random() * 50, 0]),
    );
    expect(allFinite(out)).toBe(true);
    // The front moves at under 0.8 cells per second and the steepest slope is ~70 per cell: about 0.25
    // of height per step at worst for a ring that appeared instantly, 1 for a pop.
    expect(max(lane(out, 0))).toBeLessThan(0.35);
    expect(max(lane(out, 1))).toBeGreaterThan(0.3);
    expect(max(lane(out, 1))).toBeLessThan(4.1);
  });

  it('cells are not in step: at any moment only some of the cells carry a ring', async () => {
    // For 400 cells and 12 moments, count the cells whose centre area shows a ring.
    const out = await gpuF32(
      `${RIPPLES}
       fn f(x: vec4f) -> vec4f {
         var ringing = 0.0;
         for (var c = 0; c < 400; c++) {
           let cell = vec2i(c % 20, c / 20);
           var energy = 0.0;
           for (var k = 0; k < 16; k++) {
             let q = vec2f(cell) + (vec2f(f32(k % 4), f32(k / 4)) + 0.5) * 0.25;
             energy += length(_ripple_cell(q, cell, x.x, 0.5, 1.1, 0));
           }
           if (energy > 0.05) { ringing += 1.0; }
         }
         return vec4f(ringing / 400.0, 0.0, 0.0, 0.0);
       }`,
      rows(12, (i) => [i * 1.37 + 0.2, 0, 0, 0]),
    );
    const shares = lane(out, 0).slice(0, 12);
    for (const share of shares) {
      // A cell rings for about half its period at density 0.5; synchronised cells would give 0 or 1.
      expect(share).toBeGreaterThan(0.1);
      expect(share).toBeLessThan(0.75);
    }
    // And the share is about the same at every moment, not pulsing.
    expect(max(shares) - min(shares)).toBeLessThan(0.25);
  });

  it('normals are unit and finite; the level-of-detail fade runs from 1 to 0', async () => {
    const random = sequence(81);
    const out = await gpuF32(
      `${RIPPLES}
       fn f(x: vec4f) -> vec4f {
         let r = ripple_rings(x.xy, x.z, 1.0, 1.1, 0);
         let n = ripple_normal(r, x.w);
         return vec4f(length(n), n.z, 0.0, 0.0);
       }`,
      rows(5000, () => [random() * 4, random() * 4, random() * 20, random() * 0.05]),
    );
    expect(allFinite(out)).toBe(true);
    for (const length of lane(out, 0)) expect(Math.abs(length - 1)).toBeLessThan(1e-5);
    expect(min(lane(out, 1))).toBeGreaterThan(0);
    const fades = await scalars(RIPPLES, [
      'ripple_lod_fade(0.0)',
      'ripple_lod_fade(0.01)',
      'ripple_lod_fade(0.04)',
      'ripple_lod_fade(0.07)',
      'ripple_lod_fade(1.0)',
    ]);
    expect(fades[0]).toBe(1);
    expect(fades[4]).toBe(0);
    for (let i = 1; i < fades.length; i++)
      expect(fades[i]!).toBeLessThanOrEqual(fades[i - 1]! + 1e-6);
  });
});

// --- Wetness ----------------------------------------------------------------------------------

describe('wetness', () => {
  it('wet albedo stays in [0, 1], is the identity when dry and never gets lighter as it gets wetter', async () => {
    const random = sequence(91);
    const colours = Array.from({ length: 2000 }, () => [random(), random(), random()]);
    const steps = 11;
    const inputs = rows(colours.length * steps, (i) => {
      const c = colours[Math.floor(i / steps)]!;
      return [c[0]!, c[1]!, c[2]!, (i % steps) / (steps - 1)];
    });
    const out = await gpuF32(
      `${WETNESS}
       fn f(x: vec4f) -> vec4f {
         let c = wet_albedo(x.xyz, x.w);
         return vec4f(c.x, c.y, c.z, luminance(c));
       }`,
      inputs,
    );
    expect(allFinite(out)).toBe(true);
    expect(min(Array.from(out))).toBeGreaterThanOrEqual(0);
    expect(max(Array.from(out))).toBeLessThanOrEqual(1 + 1e-6);
    for (let k = 0; k < colours.length; k++) {
      const dry = out.subarray(k * steps * 4, k * steps * 4 + 3);
      const c = colours[k]!;
      for (let j = 0; j < 3; j++) expect(Math.abs(dry[j]! - c[j]!)).toBeLessThan(2e-6);
      for (let s = 1; s < steps; s++) {
        const before = out[(k * steps + s - 1) * 4 + 3]!;
        const after = out[(k * steps + s) * 4 + 3]!;
        expect(after).toBeLessThanOrEqual(before + 1e-5);
      }
    }
  });

  it('wet albedo is more saturated when wet', async () => {
    const out = await gpuF32(
      `${WETNESS}
       fn f(x: vec4f) -> vec4f {
         let c = wet_albedo(x.xyz, x.w);
         let hi = max(c.x, max(c.y, c.z));
         let lo = min(c.x, min(c.y, c.z));
         return vec4f((hi - lo) / max(hi, 1e-4), 0.0, 0.0, 0.0);
       }`,
      rowsOf([
        [0.5, 0.35, 0.25, 0],
        [0.5, 0.35, 0.25, 0.5],
        [0.5, 0.35, 0.25, 1],
        [0.2, 0.25, 0.3, 0],
        [0.2, 0.25, 0.3, 1],
      ]),
    );
    const s = lane(out, 0);
    expect(s[1]!).toBeGreaterThan(s[0]!);
    expect(s[2]!).toBeGreaterThan(s[1]!);
    expect(s[4]!).toBeGreaterThan(s[3]!);
  });

  it('roughness falls and reflectance rises with wetness, all within [0, 1]', async () => {
    const steps = 21;
    const inputs = rows(steps * 12, (i) => {
      const w = (i % steps) / (steps - 1);
      const block = Math.floor(i / steps);
      return [
        w,
        [0.1, 0.5, 0.9][block % 3]!,
        [0.05, 0.4, 0.8, 1][Math.floor(block / 3)]!,
        0.3 + (0.7 * ((block * 7) % 5)) / 4,
      ];
    });
    const out = await gpuF32(
      `${WETNESS}
       fn f(x: vec4f) -> vec4f {
         // x: wetness, roughness, cos(theta), water mask
         let rough = wet_roughness(x.y, x.x);
         let refl = wet_reflectance(x.z, x.x, x.w * 0.0, x.y);
         let flooded = wet_reflectance(x.z, x.x, 1.0, x.y);
         let partial = wet_reflectance(x.z, x.x, x.w, x.y);
         return vec4f(rough, refl, flooded, partial);
       }`,
      inputs,
    );
    expect(allFinite(out)).toBe(true);
    expect(min(Array.from(out))).toBeGreaterThanOrEqual(0);
    expect(max(Array.from(out))).toBeLessThanOrEqual(1);
    for (let i = 1; i < inputs.length / 4; i++) {
      if (i % steps === 0) continue;
      expect(out[i * 4]!).toBeLessThanOrEqual(out[(i - 1) * 4]! + 1e-6); // roughness
      expect(out[i * 4 + 1]!).toBeGreaterThanOrEqual(out[(i - 1) * 4 + 1]! - 1e-6); // damp sheen
      // Over water the wetness of the ground does not matter.
      expect(Math.abs(out[i * 4 + 2]! - out[(i - 1) * 4 + 2]!)).toBeLessThan(1e-6);
    }
    // Flooded surfaces are the Fresnel reflectance of water; damp ground never beats a puddle.
    for (let i = 0; i < inputs.length / 4; i++) {
      expect(out[i * 4 + 1]!).toBeLessThanOrEqual(out[i * 4 + 2]! + 1e-6);
      expect(out[i * 4 + 3]!).toBeLessThanOrEqual(out[i * 4 + 2]! + 1e-6);
    }
    const [wet, dry, puddle] = await scalars(WETNESS, [
      'wet_reflectance(0.5, 1.0, 0.0, 0.2)',
      'wet_reflectance(0.5, 0.0, 0.0, 0.2)',
      'wet_reflectance(0.5, 0.0, 1.0, 0.2) - fresnel_water(0.5)',
    ]);
    expect(dry).toBe(0);
    expect(wet).toBeGreaterThan(0);
    expect(Math.abs(puddle!)).toBeLessThan(1e-6);
  });

  it('flooding follows the height field: depth, shoreline mask and the damp band', async () => {
    const heights = Array.from({ length: 41 }, (_, i) => i / 40); // 0..1
    const bases = [0, 0.3, 0.7];
    const inputs = rows(heights.length * bases.length, (i) => [
      heights[i % heights.length]!,
      0.4,
      0.2,
      bases[Math.floor(i / heights.length)]!,
    ]);
    const out = await gpuF32(
      `${WETNESS}
       fn f(x: vec4f) -> vec4f {
         // x: height, level, reach, base wetness
         let depth = wet_depth(x.x, x.y);
         return vec4f(depth, wet_water_mask(depth, 0.05), wet_damp(x.x, x.y, x.z, x.w), 0.0);
       }`,
      inputs,
    );
    expect(allFinite(out)).toBe(true);
    for (let b = 0; b < bases.length; b++) {
      for (let h = 0; h < heights.length; h++) {
        const i = b * heights.length + h;
        const height = heights[h]!;
        const [depth, mask, damp] = [out[i * 4]!, out[i * 4 + 1]!, out[i * 4 + 2]!];
        expect(depth).toBeCloseTo(Math.max(0.4 - height, 0), 5);
        expect(mask).toBeGreaterThanOrEqual(0);
        expect(mask).toBeLessThanOrEqual(1);
        expect(damp).toBeGreaterThanOrEqual(bases[b]! - 1e-6);
        expect(damp).toBeLessThanOrEqual(1);
        if (height <= 0.4) expect(damp).toBe(1);
        if (height === 0) expect(mask).toBe(1);
        if (height >= 0.4) expect(mask).toBe(0);
        if (height >= 0.4 + 0.2) expect(damp).toBeCloseTo(bases[b]!, 5);
        if (h > 0) {
          // Higher ground: shallower, less covered, less damp.
          expect(depth).toBeLessThanOrEqual(out[(i - 1) * 4]! + 1e-6);
          expect(mask).toBeLessThanOrEqual(out[(i - 1) * 4 + 1]! + 1e-6);
          expect(damp).toBeLessThanOrEqual(out[(i - 1) * 4 + 2]! + 1e-6);
        }
      }
    }
    // A damper ground is never drier anywhere.
    for (let h = 0; h < heights.length; h++) {
      expect(out[(heights.length + h) * 4 + 2]!).toBeGreaterThanOrEqual(out[h * 4 + 2]! - 1e-6);
      expect(out[(2 * heights.length + h) * 4 + 2]!).toBeGreaterThanOrEqual(
        out[(heights.length + h) * 4 + 2]! - 1e-6,
      );
    }
  });

  it('a hard shoreline (edge 0) is a step and seen-through-water colour darkens with depth', async () => {
    const values = await scalars(WETNESS, [
      'wet_water_mask(0.0, 0.0)',
      'wet_water_mask(0.001, 0.0)',
      'wet_water_mask(0.5, 1.0)',
      'wet_through_water(vec3f(0.5), 0.0, vec3f(30.0, 20.0, 10.0)).x',
      'wet_through_water(vec3f(0.5), 0.02, vec3f(30.0, 20.0, 10.0)).x',
      'wet_through_water(vec3f(0.5), 0.1, vec3f(30.0, 20.0, 10.0)).x',
      'wet_through_water(vec3f(0.5), 0.1, vec3f(30.0, 20.0, 10.0)).z',
    ]);
    expect(values[0]).toBe(0);
    expect(values[1]).toBe(1);
    expect(values[2]).toBeCloseTo(0.5, 5);
    expect(values[3]).toBeCloseTo(0.5, 5);
    expect(values[4]!).toBeLessThan(values[3]!);
    expect(values[5]!).toBeLessThan(values[4]!);
    // Red is absorbed faster than blue.
    expect(values[6]!).toBeGreaterThan(values[5]!);
  });
});

// --- Demos: no parameter value may produce NaN, infinity or a negative colour -------------------

function halfToFloat(h: number): number {
  const sign = h & 0x8000 ? -1 : 1;
  const exponent = (h >> 10) & 0x1f;
  const fraction = h & 0x3ff;
  if (exponent === 0) return sign * 2 ** -14 * (fraction / 1024);
  if (exponent === 31) return fraction ? Number.NaN : sign * Infinity;
  return sign * 2 ** (exponent - 15) * (1 + fraction / 1024);
}

async function readFloats(device: GPUDevice, texture: GPUTexture): Promise<Float32Array> {
  const { width, height } = texture;
  const bytesPerRow = Math.ceil((width * 8) / 256) * 256;
  const buffer = device.createBuffer({
    size: bytesPerRow * height,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });
  const encoder = device.createCommandEncoder();
  encoder.copyTextureToBuffer({ texture }, { buffer, bytesPerRow }, { width, height });
  device.queue.submit([encoder.finish()]);
  await buffer.mapAsync(GPUMapMode.READ);
  const padded = new Uint16Array(buffer.getMappedRange().slice(0));
  buffer.unmap();
  buffer.destroy();
  const data = new Float32Array(width * height * 4);
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width * 4; x++) {
      data[y * width * 4 + x] = halfToFloat(padded[(y * bytesPerRow) / 2 + x] ?? 0);
    }
  }
  return data;
}

describe('rain demos stay finite across their parameter ranges', () => {
  const WIDTH = 96;
  const HEIGHT = 64;

  it.each(['rain-glass', 'puddles'])('%s', async (id) => {
    const device = await getTestDevice();
    const { loadEffect } = await import('../src/effects/registry.ts');
    const { defaultParams } = await import('../src/effects/types.ts');
    const { noProfiler } = await import('../src/gpu/profiler.ts');
    const canvas = Object.assign(new EventTarget(), {
      width: WIDTH,
      height: HEIGHT,
      style: {},
      getBoundingClientRect: () => ({ left: 0, top: 0, width: WIDTH, height: HEIGHT }),
    }) as unknown as HTMLCanvasElement;
    const effect = await loadEffect(id);
    // Float target: the sRGB encode and 8-bit clamp would hide a NaN as black.
    const target = device.createTexture({
      size: [WIDTH, HEIGHT],
      format: 'rgba16float',
      usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC,
    });
    const params = defaultParams(effect.params) as Record<string, unknown>;
    const instance = await effect.create({
      device,
      format: 'rgba16float',
      canvas,
      params: params as never,
      profiler: noProfiler,
    });
    instance.resize(WIDTH, HEIGHT, 1);
    expect(Object.keys(effect.presets ?? {}).length).toBeGreaterThanOrEqual(2);

    let frame = 0;
    const check = async (label: string) => {
      // Two frames a long time apart, so drops and rings are in different phases of their cycles.
      for (const delta of [1 / 60, 7.3]) {
        const encoder = device.createCommandEncoder();
        instance.frame({ time: frame / 60, delta, frame }, encoder, target.createView());
        device.queue.submit([encoder.finish()]);
        frame++;
      }
      const data = await readFloats(device, target);
      for (let i = 0; i < data.length; i++) {
        const value = data[i] ?? Number.NaN;
        if (!(value >= -0.01 && value <= 1.5)) {
          throw new Error(
            `${id}: ${label} gave ${value} at channel ${i % 4}, pixel ${(i / 4) | 0}`,
          );
        }
      }
    };

    await check('defaults');
    for (const [name, preset] of Object.entries(effect.presets ?? {})) {
      Object.assign(params, defaultParams(effect.params), preset);
      await check(`preset ${name}`);
    }
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
      for (const value of values) {
        Object.assign(params, defaultParams(effect.params));
        params[key] = value;
        await check(`${key} = ${JSON.stringify(value)}`);
      }
    }
    Object.assign(params, defaultParams(effect.params));
    for (const [key, def] of Object.entries(effect.params)) {
      if (def.type === 'number') params[key] = def.max;
    }
    await check('every number at its maximum');
    for (const [key, def] of Object.entries(effect.params)) {
      if (def.type === 'number') params[key] = def.min;
    }
    await check('every number at its minimum');
    instance.destroy();
    target.destroy();
  });
});
