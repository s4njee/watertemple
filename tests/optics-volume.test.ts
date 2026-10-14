import { describe, expect, it, vi } from 'vitest';
import { createFullscreenPipeline, renderFullscreen } from '../src/gpu/fullscreen.ts';
import { getTestDevice, readTexture } from './gpu.ts';
import { gpuF32, lane, rowsOf, scalars } from './wgsl-eval.ts';

vi.setConfig({ testTimeout: 30_000 });

const REFRACTION = '#include "refraction.wgsl"';
const ABSORPTION = '#include "absorption.wgsl"';
const FOAM = '#include "foam.wgsl"';
const BACKDROP = '#include "backdrop.wgsl"';

/** Small deterministic PRNG (mulberry32) so failures reproduce. */
function makeRandom(seed: number): () => number {
  let state = seed >>> 0;
  return () => {
    state = (state + 0x6d2b79f5) >>> 0;
    let t = state;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function randomRows(count: number, seed: number, lows: number[], highs: number[]): Float32Array {
  const random = makeRandom(seed);
  const rows = new Float32Array(count * 4);
  for (let i = 0; i < count; i++) {
    for (let c = 0; c < 4; c++) {
      const lo = lows[c] ?? 0;
      const hi = highs[c] ?? 1;
      rows[i * 4 + c] = lo + (hi - lo) * random();
    }
  }
  return rows;
}

const degrees = (d: number) => (d * Math.PI) / 180;

// --- Refraction ------------------------------------------------------------------------------------

/** Lateral offset of a ray hitting a surface tilted by `tilt` toward +x, per Snell's law. */
function snellOffset(tiltDeg: number, ior: number, thickness: number): number {
  const theta = degrees(tiltDeg);
  const inside = Math.asin(Math.sin(theta) / ior);
  return -thickness * Math.tan(theta - inside);
}

describe('refraction offsets', () => {
  it('are exactly zero for a flat surface seen head-on, in both forms', async () => {
    const [cheap, physical, cheapY, physicalY] = await scalars(REFRACTION, [
      'refraction_offset_cheap(vec3f(0.0, 0.0, 1.0), 1.33, 0.7).x',
      'refraction_offset_physical(vec3f(0.0, 0.0, 1.0), vec3f(0.0, 0.0, -1.0), 1.33, 0.7).x',
      'refraction_offset_cheap(vec3f(0.0, 0.0, 1.0), 1.33, 0.7).y',
      'refraction_offset_physical(vec3f(0.0, 0.0, 1.0), vec3f(0.0, 0.0, -1.0), 1.33, 0.7).y',
    ]);
    expect(Math.abs(cheap ?? 1)).toBe(0);
    expect(Math.abs(cheapY ?? 1)).toBe(0);
    expect(Math.abs(physical ?? 1)).toBeLessThan(1e-6);
    expect(Math.abs(physicalY ?? 1)).toBeLessThan(1e-6);
  });

  it('are zero when the IOR is 1 (no medium change) or the water has no thickness', async () => {
    const tilted = 'normalize(vec3f(0.4, -0.3, 1.0))';
    const view = 'vec3f(0.0, 0.0, -1.0)';
    const values = await scalars(REFRACTION, [
      `refraction_offset_cheap(${tilted}, 1.0, 0.5).x`,
      `refraction_offset_physical(${tilted}, ${view}, 1.0, 0.5).x`,
      `refraction_offset_cheap(${tilted}, 1.33, 0.0).x`,
      `refraction_offset_physical(${tilted}, ${view}, 1.33, 0.0).y`,
    ]);
    expect(values.map((v) => Math.abs(v))).toEqual([0, expect.any(Number), 0, 0]);
    expect(Math.abs(values[1] ?? 1)).toBeLessThan(1e-6);
  });

  it('grow with the index of refraction and with thickness', async () => {
    const n = 'normalize(vec3f(0.3, 0.0, 1.0))';
    const view = 'vec3f(0.0, 0.0, -1.0)';
    const iors = [1.05, 1.2, 1.33, 1.5, 1.9, 2.4];
    const thicknesses = [0.05, 0.1, 0.2, 0.4];
    const expressions = [
      ...iors.map((ior) => `abs(refraction_offset_cheap(${n}, ${ior}, 0.3).x)`),
      ...iors.map((ior) => `abs(refraction_offset_physical(${n}, ${view}, ${ior}, 0.3).x)`),
      ...thicknesses.map((t) => `abs(refraction_offset_cheap(${n}, 1.33, ${t}).x)`),
      ...thicknesses.map((t) => `abs(refraction_offset_physical(${n}, ${view}, 1.33, ${t}).x)`),
    ];
    const v = await scalars(REFRACTION, expressions);
    const slices = [v.slice(0, 6), v.slice(6, 12), v.slice(12, 16), v.slice(16, 20)];
    for (const slice of slices) {
      for (let i = 1; i < slice.length; i++) expect(slice[i]).toBeGreaterThan(slice[i - 1] ?? 0);
    }
    // Linear in thickness: doubling the thickness doubles the shift.
    expect((v[13] ?? 0) / (v[12] ?? 1)).toBeCloseTo(2, 4);
    expect((v[17] ?? 0) / (v[16] ?? 1)).toBeCloseTo(2, 4);
  });

  it('match Snell’s law for the physical form at known angles', async () => {
    const angles = [5, 15, 30, 45, 60, 75];
    const iors = [1.0003, 1.333, 1.52];
    const cases = angles.flatMap((a) => iors.map((ior) => ({ a, ior })));
    const expressions = cases.map(
      ({ a, ior }) =>
        `refraction_offset_physical(vec3f(${Math.sin(degrees(a))}, 0.0, ${Math.cos(degrees(a))}), vec3f(0.0, 0.0, -1.0), ${ior}, 0.4).x`,
    );
    const got = await scalars(REFRACTION, expressions);
    cases.forEach(({ a, ior }, i) => {
      const expected = snellOffset(a, ior, 0.4);
      expect(got[i], `tilt ${a} deg, ior ${ior}`).toBeCloseTo(expected, 4);
    });
  });

  it('has the direction Snell’s law predicts: toward the surface’s downhill side, never past it', async () => {
    // Normal tilted toward +x: the ray inside bends toward -n, so the bed point seen is at -x.
    // A normal tilted toward -y gives +y.
    const [x, y] = await scalars(REFRACTION, [
      'refraction_offset_physical(normalize(vec3f(0.4, 0.0, 1.0)), vec3f(0.0, 0.0, -1.0), 1.33, 0.5).x',
      'refraction_offset_physical(normalize(vec3f(0.0, -0.4, 1.0)), vec3f(0.0, 0.0, -1.0), 1.33, 0.5).y',
    ]);
    expect(x).toBeLessThan(0);
    expect(y).toBeGreaterThan(0);
  });

  it('handles a tilted view through a flat surface (shift relative to the straight continuation)', async () => {
    const angle = 40;
    const ior = 1.333;
    const view = `vec3f(${Math.sin(degrees(angle))}, 0.0, ${-Math.cos(degrees(angle))})`;
    const [shift] = await scalars(REFRACTION, [
      `refraction_offset_physical(vec3f(0.0, 0.0, 1.0), ${view}, ${ior}, 0.5).x`,
    ]);
    const inside = Math.asin(Math.sin(degrees(angle)) / ior);
    const expected = 0.5 * (Math.tan(inside) - Math.tan(degrees(angle)));
    expect(shift).toBeCloseTo(expected, 4);
  });

  it('cheap form tracks the physical one for gentle slopes', async () => {
    const angles = [3, 8, 15, 25, 30];
    const expressions = angles.flatMap((a) => {
      const n = `vec3f(${Math.sin(degrees(a))}, 0.0, ${Math.cos(degrees(a))})`;
      return [
        `refraction_offset_cheap(${n}, 1.333, 0.4).x`,
        `refraction_offset_physical(${n}, vec3f(0.0, 0.0, -1.0), 1.333, 0.4).x`,
      ];
    });
    const v = await scalars(REFRACTION, expressions);
    angles.forEach((a, i) => {
      const cheap = v[i * 2] ?? 0;
      const physical = v[i * 2 + 1] ?? 1;
      expect(Math.abs(cheap / physical - 1), `${a} deg`).toBeLessThan(0.05);
    });
  });

  it('is finite at grazing views and steep surfaces', async () => {
    const v = await scalars(REFRACTION, [
      'refraction_offset_physical(vec3f(0.0, 0.0, 1.0), vec3f(1.0, 0.0, -0.0001), 1.33, 0.5).x',
      'refraction_offset_physical(normalize(vec3f(1.0, 0.0, 0.001)), vec3f(0.0, 0.0, -1.0), 1.33, 0.5).x',
      'refraction_offset_cheap(normalize(vec3f(1.0, 0.0, 0.0)), 1.33, 0.5).x',
    ]);
    for (const value of v) expect(Number.isFinite(value)).toBe(true);
  });

  it('converts to uv with the aspect ratio', async () => {
    const [x, y] = await scalars(REFRACTION, [
      'refraction_to_uv(vec2f(0.2, 0.2), 2.0).x',
      'refraction_to_uv(vec2f(0.2, 0.2), 2.0).y',
    ]);
    expect(x).toBeCloseTo(0.1, 6);
    expect(y).toBeCloseTo(0.2, 6);
  });
});

describe('refraction edge handling', () => {
  it('never produces a uv outside [0, 1], whatever the offset', async () => {
    // x, y: uv (including outside the image), z, w: offset (up to +-3 images away).
    const input = randomRows(8000, 11, [-0.5, -0.5, -3, -3], [1.5, 1.5, 3, 3]);
    const out = await gpuF32(
      `${REFRACTION}
       fn f(x: vec4f) -> vec4f {
         let a = refraction_safe_uv(x.xy, x.zw, 0.05);
         let b = refraction_safe_uv(x.xy, x.zw, 0.0);
         return vec4f(a, b);
       }`,
      input,
    );
    let bad = 0;
    for (const value of out) if (!(value >= 0 && value <= 1)) bad++;
    expect(bad).toBe(0);
  });

  it('leaves in-range samples alone and never pushes a sample toward the border it is at', async () => {
    const [centre, atLeftOut, atRightOut, atLeftIn] = await scalars(REFRACTION, [
      'refraction_safe_uv(vec2f(0.5, 0.5), vec2f(0.05, 0.0), 0.05).x',
      // On the left border with an offset pointing further left: no movement at all.
      'refraction_safe_uv(vec2f(0.0, 0.5), vec2f(-0.2, 0.0), 0.05).x',
      'refraction_safe_uv(vec2f(1.0, 0.5), vec2f(0.2, 0.0), 0.05).x',
      // On the left border with an offset pointing inward: not suppressed by the fade of the far side.
      'refraction_safe_uv(vec2f(0.02, 0.5), vec2f(0.1, 0.0), 0.05).x',
    ]);
    expect(centre).toBeCloseTo(0.55, 6);
    expect(atLeftOut).toBe(0);
    expect(atRightOut).toBe(1);
    expect(atLeftIn).toBeGreaterThan(0.02);
  });

  it('folds uv by mirroring, not wrapping or clamping', async () => {
    const v = await scalars(REFRACTION, [
      'refraction_mirror_uv(vec2f(-0.2, 0.0)).x',
      'refraction_mirror_uv(vec2f(1.3, 0.0)).x',
      'refraction_mirror_uv(vec2f(2.4, 0.0)).x',
      'refraction_mirror_uv(vec2f(0.37, 0.0)).x',
      'refraction_mirror_uv(vec2f(-1.25, 0.0)).x',
    ]);
    const expected = [0.2, 0.7, 0.4, 0.37, 0.75];
    expected.forEach((e, i) => expect(v[i]).toBeCloseTo(e, 5));
  });

  it('turns a blur radius into a mip level, clamped to the chain', async () => {
    const v = await scalars(REFRACTION, [
      'refraction_blur_lod(0.0, 8.0)',
      'refraction_blur_lod(4.0, 8.0)',
      'refraction_blur_lod(1000.0, 5.0)',
    ]);
    expect(v[0]).toBe(0);
    expect(v[1]).toBeCloseTo(3, 5);
    expect(v[2]).toBe(5);
  });
});

describe('chromatic dispersion', () => {
  it('orders the channel IOR red < green < blue and keeps them >= 1', async () => {
    const out = await gpuF32(
      `${REFRACTION}
       fn f(x: vec4f) -> vec4f { return vec4f(refraction_dispersion_iors(x.x, x.y), 0.0); }`,
      rowsOf([
        [1.333, 0.02],
        [1.52, 0.05],
        [1.0, 0.1],
        [1.05, 0.2],
      ]),
    );
    const r = lane(out, 0);
    const g = lane(out, 1);
    const b = lane(out, 2);
    for (let i = 0; i < 4; i++) {
      expect(r[i]).toBeLessThanOrEqual(g[i] ?? 0);
      expect(g[i]).toBeLessThanOrEqual(b[i] ?? 0);
      expect(r[i]).toBeGreaterThanOrEqual(1);
    }
    expect(r[0]).toBeLessThan(g[0] ?? 0);
    expect(g[0]).toBeCloseTo(1.333, 5);
  });

  it('bends red least and blue most, both in the scale factors and the physical offsets', async () => {
    const n = 'normalize(vec3f(0.35, 0.1, 1.0))';
    const [sr, sg, sb, pr, pg, pb] = await scalars(REFRACTION, [
      'refraction_dispersion_scales(1.333, 0.03).x',
      'refraction_dispersion_scales(1.333, 0.03).y',
      'refraction_dispersion_scales(1.333, 0.03).z',
      `length(refraction_offset_physical(${n}, vec3f(0.0, 0.0, -1.0), 1.333 - 0.03, 0.4))`,
      `length(refraction_offset_physical(${n}, vec3f(0.0, 0.0, -1.0), 1.333, 0.4))`,
      `length(refraction_offset_physical(${n}, vec3f(0.0, 0.0, -1.0), 1.333 + 0.03, 0.4))`,
    ]);
    expect(sg).toBeCloseTo(1, 6);
    expect(sr).toBeLessThan(1);
    expect(sb).toBeGreaterThan(1);
    expect(pr).toBeLessThan(pg ?? 0);
    expect(pg).toBeLessThan(pb ?? 0);
    // The cheap scale factors agree with the exact per-channel offsets to a few percent.
    expect((pr ?? 0) / (pg ?? 1)).toBeCloseTo(sr ?? 0, 1);
    expect((pb ?? 0) / (pg ?? 1)).toBeCloseTo(sb ?? 0, 1);
  });

  it('gives no separation at zero dispersion and builds ordered per-channel offsets', async () => {
    const out = await gpuF32(
      `${REFRACTION}
       fn f(x: vec4f) -> vec4f {
         let o = refraction_rgb_offsets(vec2f(0.1, -0.05), refraction_dispersion_scales(1.333, x.x));
         return vec4f(length(o.r), length(o.g), length(o.b), 0.0);
       }`,
      rowsOf([[0], [0.02], [0.08]]),
    );
    const r = lane(out, 0);
    const g = lane(out, 1);
    const b = lane(out, 2);
    expect(r[0]).toBeCloseTo(g[0] ?? 0, 6);
    expect(b[0]).toBeCloseTo(g[0] ?? 0, 6);
    for (const i of [1, 2]) {
      expect(r[i]).toBeLessThan(g[i] ?? 0);
      expect(g[i]).toBeLessThan(b[i] ?? 0);
    }
    // More dispersion, wider fringe.
    expect((b[2] ?? 0) - (r[2] ?? 0)).toBeGreaterThan((b[1] ?? 0) - (r[1] ?? 0));
  });
});

// --- Sampling through a real texture ---------------------------------------------------------------

/** Renders `cases` (WGSL statements that set `var c: vec3f`) to a row of pixels and returns rgb bytes. */
async function renderCases(cases: string[], texture: GPUTexture): Promise<number[][]> {
  const device = await getTestDevice();
  const branches = cases.map((body, i) => `if (i == ${i}u) { ${body} }`).join('\n  ');
  const pipeline = createFullscreenPipeline(device, {
    fragment: {
      label: 'optics-sampling.wgsl',
      code: `#include "fullscreen.wgsl"
#include "refraction.wgsl"
@group(0) @binding(0) var tex: texture_2d<f32>;
@group(0) @binding(1) var samp: sampler;
@fragment fn fs(in: FullscreenVarying) -> @location(0) vec4f {
  let i = u32(in.position.x);
  var c = vec3f(0.0);
  ${branches}
  return vec4f(c, 1.0);
}`,
    },
    format: 'rgba8unorm',
  });
  const target = device.createTexture({
    size: [cases.length, 1],
    format: 'rgba8unorm',
    usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC,
  });
  const group = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: texture.createView() },
      {
        binding: 1,
        resource: device.createSampler({ magFilter: 'nearest', minFilter: 'nearest' }),
      },
    ],
  });
  device.pushErrorScope('validation');
  const encoder = device.createCommandEncoder();
  renderFullscreen(encoder, target.createView(), pipeline, [group]);
  device.queue.submit([encoder.finish()]);
  const error = await device.popErrorScope();
  if (error) throw new Error(error.message);
  const pixels = await readTexture(device, target);
  target.destroy();
  return cases.map((_, i) => [
    pixels[i * 4] ?? -1,
    pixels[i * 4 + 1] ?? -1,
    pixels[i * 4 + 2] ?? -1,
  ]);
}

function textureOf(device: GPUDevice, size: number, texel: (x: number) => number[]): GPUTexture {
  const data = new Uint8Array(size * 4);
  for (let x = 0; x < size; x++) data.set([...texel(x), 255], x * 4);
  const texture = device.createTexture({
    size: [size, 1],
    format: 'rgba8unorm',
    usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
  });
  device.queue.writeTexture({ texture }, data, { bytesPerRow: size * 4 }, [size, 1]);
  return texture;
}

describe('refraction sampling', () => {
  it('keeps a flat colour exactly (spectral taps are normalised) and bends the channels apart at an edge', async () => {
    const device = await getTestDevice();
    const flat = textureOf(device, 16, () => [128, 64, 192]);
    const flatResults = await renderCases(
      [
        'c = refraction_sample_spectral(tex, samp, vec2f(0.5, 0.5), vec2f(0.07, 0.0), 1.333, 0.05, 0.0, 0.05, 8);',
        'c = refraction_sample_spectral(tex, samp, vec2f(0.5, 0.5), vec2f(-0.07, 0.0), 1.5, 0.08, 0.0, 0.05, 2);',
        'c = refraction_sample_spectral(tex, samp, vec2f(0.9, 0.5), vec2f(0.4, 0.0), 1.333, 0.05, 0.0, 0.05, 16);',
        'c = refraction_sample(tex, samp, vec2f(0.5, 0.5), vec2f(0.2, 0.0), 0.0, 0.05);',
      ],
      flat,
    );
    for (const colour of flatResults) expect(colour).toEqual([128, 64, 192]);

    // Left half black, right half white, edge at 0.5. Green lands just left of it, blue just right.
    const step = textureOf(device, 16, (x) => (x < 8 ? [0, 0, 0] : [255, 255, 255]));
    const results = await renderCases(
      [
        // scales for ior 1.333, dispersion 0.1 are about (0.76, 1, 1.21): r 0.468, g 0.49, b 0.509.
        'c = refraction_sample_rgb(tex, samp, vec2f(0.4, 0.5), refraction_rgb_offsets(vec2f(0.09, 0.0), refraction_dispersion_scales(1.333, 0.1)), 0.0, 0.05);',
        // No dispersion: all three channels agree.
        'c = refraction_sample_rgb(tex, samp, vec2f(0.4, 0.5), refraction_rgb_offsets(vec2f(0.09, 0.0), refraction_dispersion_scales(1.333, 0.0)), 0.0, 0.05);',
        // At the right border with a huge outward offset: must stay on the (white) right side, not wrap to black.
        'c = refraction_sample(tex, samp, vec2f(0.99, 0.5), vec2f(0.5, 0.0), 0.0, 0.05);',
        // At the left border with a huge outward offset: stays black, does not smear in white from the far side.
        'c = refraction_sample(tex, samp, vec2f(0.01, 0.5), vec2f(-0.5, 0.0), 0.0, 0.05);',
      ],
      step,
    );
    expect(results[0]).toEqual([0, 0, 255]);
    expect(results[1]).toEqual([0, 0, 0]);
    expect(results[2]).toEqual([255, 255, 255]);
    expect(results[3]).toEqual([0, 0, 0]);

    // The spectral version smears the same edge into a gradient: not all-or-nothing in any channel.
    const spectral = await renderCases(
      [
        'c = refraction_sample_spectral(tex, samp, vec2f(0.44, 0.5), vec2f(0.0625, 0.0), 1.333, 0.2, 0.0, 0.05, 16);',
      ],
      step,
    );
    const [r, g, b] = spectral[0] ?? [];
    expect(b ?? 0).toBeGreaterThanOrEqual(g ?? 0);
    expect(g ?? 0).toBeGreaterThanOrEqual(r ?? 0);
    expect(b).toBeGreaterThan(r ?? 255);
    flat.destroy();
    step.destroy();
  });
});

// --- Absorption ------------------------------------------------------------------------------------

describe('depth absorption', () => {
  it('transmits everything at zero depth and never leaves [0, 1]', async () => {
    const input = randomRows(6000, 21, [0, 0, 0, -2], [3, 3, 3, 400]);
    const out = await gpuF32(
      `${ABSORPTION}
       fn f(x: vec4f) -> vec4f { return vec4f(absorb_transmittance(x.xyz, x.w), 0.0); }`,
      input,
    );
    for (const value of out) expect(value).toBeGreaterThanOrEqual(0);
    for (const value of out) expect(value).toBeLessThanOrEqual(1);
    const zero = await gpuF32(
      `${ABSORPTION}
       fn f(x: vec4f) -> vec4f { return vec4f(absorb_transmittance(x.xyz, 0.0), 0.0); }`,
      randomRows(64, 22, [0, 0, 0, 0], [5, 5, 5, 0]),
    );
    expect(lane(zero, 0).every((v) => v === 1)).toBe(true);
    expect(lane(zero, 2).every((v) => v === 1)).toBe(true);
  });

  it('decreases monotonically with path length, per channel', async () => {
    const paths = Array.from({ length: 64 }, (_, i) => i * i * 0.05);
    const out = await gpuF32(
      `${ABSORPTION}
       fn f(x: vec4f) -> vec4f { return vec4f(absorb_transmittance(vec3f(0.5, 0.08, 0.02), x.x), 0.0); }`,
      rowsOf(paths.map((p) => [p])),
    );
    for (let c = 0; c < 3; c++) {
      const series = lane(out, c);
      for (let i = 1; i < series.length; i++) {
        expect(series[i]).toBeLessThanOrEqual(series[i - 1] ?? 0);
      }
      expect(series[0]).toBe(1);
      expect(series[series.length - 1]).toBeLessThan(1);
    }
    // Red dies first.
    const last = out.length / 4 - 1;
    expect(out[last * 4]).toBeLessThan(out[last * 4 + 1] ?? 0);
    expect(out[last * 4 + 1]).toBeLessThan(out[last * 4 + 2] ?? 0);
  });

  it('reproduces the specified colour at the specified depth', async () => {
    const cases = [
      [0.06, 0.5, 0.72, 6],
      [0.35, 0.15, 0.04, 1.5],
      [0.9, 0.9, 0.9, 0.2],
      [0.01, 0.2, 0.9, 40],
      [1, 1, 1, 3],
    ];
    const out = await gpuF32(
      `${ABSORPTION}
       fn f(x: vec4f) -> vec4f {
         let sigma = absorb_coefficients_from_color(x.xyz, x.w);
         return vec4f(absorb_transmittance(sigma, x.w), 0.0);
       }`,
      rowsOf(cases),
    );
    cases.forEach(([r, g, b], i) => {
      expect(out[i * 4]).toBeCloseTo(r ?? 0, 5);
      expect(out[i * 4 + 1]).toBeCloseTo(g ?? 0, 5);
      expect(out[i * 4 + 2]).toBeCloseTo(b ?? 0, 5);
    });
    // Impossible inputs stay finite: black asks for the maximum absorption, not infinity.
    const [black] = await scalars(ABSORPTION, [
      'absorb_coefficients_from_color(vec3f(0.0, 0.0, 0.0), 2.0).x',
    ]);
    expect(Number.isFinite(black)).toBe(true);
  });

  it('gives every named water type its documented colour at its reference depth', async () => {
    const types: [number, number[], number][] = [
      [0, [0.06, 0.5, 0.72], 6],
      [1, [0.12, 0.55, 0.5], 3],
      [2, [0.1, 0.5, 0.12], 1.5],
      [3, [0.35, 0.15, 0.04], 1.5],
      [4, [0.25, 0.65, 0.8], 4],
      [5, [0.35, 0.85, 0.95], 3],
      [6, [0.02, 0.45, 0.75], 20],
    ];
    const out = await gpuF32(
      `${ABSORPTION}
       fn f(x: vec4f) -> vec4f {
         let w = absorb_water(i32(x.x));
         return vec4f(absorb_transmittance(w.sigma, x.y), 0.0);
       }`,
      rowsOf(types.map(([kind, , depth]) => [kind, depth])),
    );
    types.forEach(([, colour], i) => {
      for (let c = 0; c < 3; c++) expect(out[i * 4 + c]).toBeCloseTo(colour[c] ?? 0, 4);
    });
  });

  it('absorbs red first in natural water and blue first in tea', async () => {
    const out = await gpuF32(
      `${ABSORPTION}
       fn f(x: vec4f) -> vec4f { return vec4f(absorb_water(i32(x.x)).sigma, 0.0); }`,
      rowsOf([[0], [1], [5], [6], [3]]),
    );
    // Rows: tropical, coastal, pool, ocean. Red goes first in all; green outlasts blue in the clear ones.
    for (const i of [0, 1, 2, 3]) {
      expect(out[i * 4]).toBeGreaterThan(out[i * 4 + 1] ?? 0);
      expect(out[i * 4]).toBeGreaterThan(out[i * 4 + 2] ?? 0);
    }
    for (const i of [0, 2, 3]) expect(out[i * 4 + 1]).toBeGreaterThan(out[i * 4 + 2] ?? 0);
    expect(out[4 * 4 + 2]).toBeGreaterThan(out[4 * 4] ?? 0);
  });

  it('lengthens the path with view angle following Snell’s law', async () => {
    const angles = [0, 20, 45, 60, 80];
    const expressions = angles.map(
      (a) => `absorb_path_length(2.0, ${Math.cos(degrees(a))}, 1.333)`,
    );
    const v = await scalars(ABSORPTION, expressions);
    angles.forEach((a, i) => {
      const inside = Math.asin(Math.sin(degrees(a)) / 1.333);
      expect(v[i]).toBeCloseTo(2 / Math.cos(inside), 3);
    });
    expect(v[0]).toBeCloseTo(2, 6);
    for (let i = 1; i < v.length; i++) expect(v[i]).toBeGreaterThan(v[i - 1] ?? 0);
    const [twoWay, finite] = await scalars(ABSORPTION, [
      'absorb_path_length_two_way(3.0, 1.0, 1.0, 1.333)',
      'absorb_path_length(1.0, 0.0, 1.333)',
    ]);
    expect(twoWay).toBeCloseTo(6, 5);
    expect(Number.isFinite(finite)).toBe(true);
  });

  it('composites: the bed at zero path, a pure tint without turbidity, the deep colour far away', async () => {
    const out = await gpuF32(
      `${ABSORPTION}
       fn f(x: vec4f) -> vec4f {
         let w = absorb_water(i32(x.x));
         let bed = vec3f(0.8, 0.6, 0.4);
         let kind = x.y;
         var c = vec3f(0.0);
         if (kind == 0.0) { c = absorb_composite(bed, w.sigma, w.turbidity, w.scatter, 0.0); }
         if (kind == 1.0) { c = absorb_composite(bed, w.sigma, 0.0, w.scatter, 3.0); }
         if (kind == 2.0) { c = absorb_composite(bed, w.sigma, w.turbidity, w.scatter, 1.0e5); }
         if (kind == 3.0) { c = absorb_deep_color(w.sigma, w.turbidity, w.scatter); }
         if (kind == 4.0) { c = absorb_transmittance(w.sigma, 3.0) * bed; }
         return vec4f(c, 0.0);
       }`,
      rowsOf([
        [1, 0],
        [1, 1],
        [1, 2],
        [1, 3],
        [1, 4],
      ]),
    );
    const row = (i: number) => [out[i * 4] ?? 0, out[i * 4 + 1] ?? 0, out[i * 4 + 2] ?? 0];
    [0.8, 0.6, 0.4].forEach((v, c) => expect(row(0)[c]).toBeCloseTo(v, 6));
    row(1).forEach((v, c) => expect(v).toBeCloseTo(row(4)[c] ?? 0, 6));
    row(2).forEach((v, c) => expect(v).toBeCloseTo(row(3)[c] ?? 0, 5));
    // Deep coastal water is blue-green, not red.
    expect(row(3)[0]).toBeLessThan(row(3)[2] ?? 0);
  });

  it('never exceeds the brighter of bed and scatter colour, and stays non-negative', async () => {
    const input = randomRows(6000, 33, [0, 0, 0, 0], [1, 1, 1, 60]);
    const out = await gpuF32(
      `${ABSORPTION}
       fn f(x: vec4f) -> vec4f {
         let w = absorb_water(i32(x.w) % 7);
         let c = absorb_composite(x.xyz, w.sigma, w.turbidity, w.scatter, x.w);
         let cap = max(x.xyz, w.scatter);
         return vec4f(c, f32(all(c <= cap + vec3f(1e-5)) && all(c >= vec3f(0.0))));
       }`,
      input,
    );
    expect(lane(out, 3).every((v) => v === 1)).toBe(true);
  });

  it('converts visibility to a scattering coefficient that falls as water clears', async () => {
    const v = await scalars(ABSORPTION, [
      'absorb_turbidity_from_visibility(1.0)',
      'absorb_turbidity_from_visibility(10.0)',
      'absorb_turbidity_from_visibility(0.0)',
    ]);
    expect(v[0]).toBeCloseTo(1.7, 5);
    expect(v[1]).toBeCloseTo(0.17, 5);
    expect(Number.isFinite(v[2])).toBe(true);
  });
});

// --- Foam ------------------------------------------------------------------------------------------

describe('foam', () => {
  it('keeps the mask in [0, 1], saturated at zero depth and zero in deep water, for any break-up noise', async () => {
    // x: depth (as a fraction of width 0.5, from -0.2 to 2), y: noise, z: breakup.
    const input = randomRows(8000, 41, [-0.1, 0, 0, 0], [1.0, 1, 1, 0]);
    const out = await gpuF32(
      `${FOAM}
       fn f(x: vec4f) -> vec4f { return vec4f(foam_mask(x.x, 0.5, x.y, x.z), 0.0, 0.0, 0.0); }`,
      input,
    );
    let outOfRange = 0;
    let deepFoam = 0;
    for (let i = 0; i < 8000; i++) {
      const m = out[i * 4] ?? -1;
      const depth = input[i * 4] ?? 0;
      if (!(m >= 0 && m <= 1)) outOfRange++;
      if (depth >= 0.5 && m !== 0) deepFoam++;
    }
    expect(outOfRange).toBe(0);
    expect(deepFoam).toBe(0);

    const v = await scalars(FOAM, [
      'foam_mask(0.0, 0.5, 0.0, 1.0)',
      'foam_mask(0.0, 0.5, 1.0, 1.0)',
      'foam_mask(0.0, 0.5, 0.3, 0.5)',
      'foam_mask(-1.0, 0.5, 0.3, 0.5)',
      'foam_mask(0.5, 0.5, 1.0, 0.0)',
      'foam_mask(3.0, 0.5, 1.0, 1.0)',
    ]);
    expect(v).toEqual([1, 1, 1, 1, 0, 0]);
  });

  it('shrinks monotonically with depth and with break-up noise', async () => {
    const depths = Array.from({ length: 51 }, (_, i) => i / 50);
    const out = await gpuF32(
      `${FOAM}
       fn f(x: vec4f) -> vec4f {
         return vec4f(foam_mask(x.x, 0.6, 1.0, 0.0), foam_mask(x.x, 0.6, 0.5, 1.0), foam_mask(x.x, 0.6, 0.0 + 0.1, 1.0), 0.0);
       }`,
      rowsOf(depths.map((d) => [d])),
    );
    for (let c = 0; c < 3; c++) {
      const series = lane(out, c);
      for (let i = 1; i < series.length; i++)
        expect(series[i]).toBeLessThanOrEqual(series[i - 1] ?? 0);
    }
    // At the same depth less noise means less reach.
    const mid = 20;
    expect(out[mid * 4 + 1]).toBeLessThanOrEqual(out[mid * 4] ?? 0);
    expect(out[mid * 4 + 2]).toBeLessThanOrEqual(out[mid * 4 + 1] ?? 0);
  });

  it('dissolves from nothing to everything as coverage rises, in [0, 1]', async () => {
    const input = randomRows(6000, 43, [0, 0, 0, 0.05], [1, 1, 0, 1]);
    const out = await gpuF32(
      `${FOAM}
       fn f(x: vec4f) -> vec4f {
         return vec4f(foam_dissolve(x.x, x.y, x.w), foam_dissolve(x.x, 1.0, x.w), foam_dissolve(x.x, 0.0, x.w), foam_dissolve(x.x, x.y + 0.05, x.w) - foam_dissolve(x.x, x.y, x.w));
       }`,
      input,
    );
    for (let i = 0; i < 6000; i++) {
      const a = out[i * 4] ?? -1;
      expect(a >= 0 && a <= 1).toBe(true);
      expect(out[i * 4 + 1]).toBe(1);
      expect(out[i * 4 + 2]).toBe(0);
      expect(out[i * 4 + 3]).toBeGreaterThanOrEqual(-1e-6); // more coverage never removes foam
    }
  });

  it('is full foam at the waterline and no foam in deep water for every foam texture value', async () => {
    const out = await gpuF32(
      `${FOAM}
       fn f(x: vec4f) -> vec4f {
         let at_line = foam_dissolve(x.x, foam_mask(0.0, 0.3, x.y, 1.0), 0.3);
         let deep = foam_dissolve(x.x, foam_mask(0.3, 0.3, x.y, 1.0), 0.3);
         return vec4f(at_line, deep, 0.0, 0.0);
       }`,
      randomRows(2000, 44, [0, 0, 0, 0], [1, 1, 0, 0]),
    );
    expect(lane(out, 0).every((v) => v === 1)).toBe(true);
    expect(lane(out, 1).every((v) => v === 0)).toBe(true);
  });

  it('produces a foam texture that is finite, in [0, 1] and has real structure', async () => {
    const input = randomRows(6000, 45, [-30, -30, 0, 0], [30, 30, 40, 0]);
    const out = await gpuF32(
      `${FOAM}
       fn f(x: vec4f) -> vec4f {
         return vec4f(foam_texture(x.xy, x.z), foam_texture_lite(x.xy, x.z), 0.0, 0.0);
       }`,
      input,
    );
    for (const c of [0, 1]) {
      const values = lane(out, c);
      let min = Infinity;
      let max = -Infinity;
      let sum = 0;
      for (const v of values) {
        expect(Number.isFinite(v)).toBe(true);
        min = Math.min(min, v);
        max = Math.max(max, v);
        sum += v;
      }
      const mean = sum / values.length;
      expect(min).toBeGreaterThanOrEqual(0);
      expect(max).toBeLessThanOrEqual(1);
      expect(max - min).toBeGreaterThan(0.6);
      expect(mean).toBeGreaterThan(0.1);
      expect(mean).toBeLessThan(0.7);
    }
  });

  it('animates the foam texture continuously (no popping between nearby times)', async () => {
    const times = Array.from({ length: 200 }, (_, i) => i * 0.01);
    const out = await gpuF32(
      `${FOAM}
       fn f(x: vec4f) -> vec4f { return vec4f(foam_texture(vec2f(3.3, 1.7), x.x), 0.0, 0.0, 0.0); }`,
      rowsOf(times.map((t) => [t])),
    );
    const series = lane(out, 0);
    let biggestStep = 0;
    for (let i = 1; i < series.length; i++) {
      biggestStep = Math.max(biggestStep, Math.abs((series[i] ?? 0) - (series[i - 1] ?? 0)));
    }
    expect(biggestStep).toBeLessThan(0.15);
    expect(Math.max(...series) - Math.min(...series)).toBeGreaterThan(0.01);
  });

  it('has a swash cycle that is periodic, in [0, 1], peaks at the rise point and starts and ends at 0', async () => {
    const phases = Array.from({ length: 201 }, (_, i) => i / 100 - 0.5);
    const out = await gpuF32(
      `${FOAM}
       fn f(x: vec4f) -> vec4f { return vec4f(foam_swash(x.x, 0.25), foam_swash(x.x + 1.0, 0.25), 0.0, 0.0); }`,
      rowsOf(phases.map((p) => [p])),
    );
    for (let i = 0; i < phases.length; i++) {
      const a = out[i * 4] ?? -1;
      expect(a >= 0 && a <= 1).toBe(true);
      expect(out[i * 4 + 1]).toBeCloseTo(a, 5);
    }
    const [zero, peak, nearEnd, quick, slow] = await scalars(FOAM, [
      'foam_swash(0.0, 0.25)',
      'foam_swash(0.25, 0.25)',
      'foam_swash(0.999, 0.25)',
      'foam_swash(0.125, 0.25)',
      'foam_swash(0.625, 0.25)',
    ]);
    expect(zero).toBe(0);
    expect(peak).toBeCloseTo(1, 5);
    expect(nearEnd).toBeLessThan(0.001);
    // Asymmetric: halfway up takes 0.125 of a cycle, halfway down takes 0.375.
    expect(quick).toBeGreaterThan(0.6);
    expect(slow).toBeCloseTo(0.5, 5);
  });

  it('fades a surface in from 0 at the intersection to 1 beyond the fade distance', async () => {
    const v = await scalars(FOAM, [
      'foam_edge_fade(0.0, 0.5)',
      'foam_edge_fade(-3.0, 0.5)',
      'foam_edge_fade(0.25, 0.5)',
      'foam_edge_fade(0.5, 0.5)',
      'foam_edge_fade(7.0, 0.5)',
      'foam_edge_fade(0.0, 0.0)',
    ]);
    expect(v[0]).toBe(0);
    expect(v[1]).toBe(0);
    expect(v[2]).toBeCloseTo(0.5, 6);
    expect(v[3]).toBe(1);
    expect(v[4]).toBe(1);
    expect(Number.isFinite(v[5])).toBe(true);
    const ramp = await gpuF32(
      `${FOAM}
       fn f(x: vec4f) -> vec4f { return vec4f(foam_edge_fade(x.x, 0.5), 0.0, 0.0, 0.0); }`,
      rowsOf(Array.from({ length: 80 }, (_, i) => [i / 100])),
    );
    const series = lane(ramp, 0);
    for (let i = 1; i < series.length; i++)
      expect(series[i]).toBeGreaterThanOrEqual(series[i - 1] ?? 0);
  });

  it('darkens wet ground, leaves dry ground alone, and finds the contact rim', async () => {
    const out = await gpuF32(
      `${FOAM}
       fn f(x: vec4f) -> vec4f {
         let sand = vec3f(0.69, 0.52, 0.26);
         let dry = foam_darken_wet(sand, 0.0, 1.0);
         let wet = foam_darken_wet(sand, 1.0, 1.0);
         return vec4f(dry.x - sand.x, wet.x / sand.x, wet.z / sand.z, foam_wetness(x.x, 0.3));
       }`,
      rowsOf([[0], [0.3], [-1], [0.15]]),
    );
    expect(out[0]).toBe(0);
    expect(out[1]).toBeLessThan(0.9); // darker
    expect(out[2]).toBeLessThan(out[1] ?? 0); // the darker channel darkens more: more saturated
    expect(lane(out, 3)).toEqual([expect.closeTo(1, 6), 0, 1, expect.closeTo(0.5, 6)]);
    const [rimAt0, rimDeep, rimMid] = await scalars(FOAM, [
      'foam_rim(0.0, 0.2)',
      'foam_rim(0.2, 0.2)',
      'foam_rim(0.1, 0.2)',
    ]);
    expect(rimAt0).toBe(1);
    expect(rimDeep).toBe(0);
    expect(rimMid).toBeCloseTo(0.25, 6);
  });
});

// --- Backdrops -------------------------------------------------------------------------------------

describe('backdrops', () => {
  const kinds = [
    ['pool tiles', 0],
    ['test card', 1],
    ['sand', 2],
    ['quiet sand', 3],
  ] as const;

  it.each(kinds)('%s is finite, in [0, 1] and not flat', async (_name, kind) => {
    const side = 64;
    const input = new Float32Array(side * side * 4);
    for (let y = 0; y < side; y++) {
      for (let x = 0; x < side; x++) {
        input.set([(x + 0.5) / side, (y + 0.5) / side, 0, 0], (y * side + x) * 4);
      }
    }
    const out = await gpuF32(
      `${BACKDROP}
       fn f(x: vec4f) -> vec4f { return vec4f(backdrop_sample(${kind}, x.xy, 1.78, 1.0 / 512.0, 1.0), 1.0); }`,
      input,
    );
    let min = Infinity;
    let max = -Infinity;
    const channelSums = [0, 0, 0];
    for (let i = 0; i < side * side; i++) {
      for (let c = 0; c < 3; c++) {
        const v = out[i * 4 + c] ?? Number.NaN;
        expect(Number.isFinite(v)).toBe(true);
        min = Math.min(min, v);
        max = Math.max(max, v);
        channelSums[c] = (channelSums[c] ?? 0) + v;
      }
    }
    expect(min).toBeGreaterThanOrEqual(0);
    expect(max).toBeLessThanOrEqual(1);
    expect(max - min).toBeGreaterThan(0.2);
    expect(channelSums.every((s) => s > 0)).toBe(true);
  });

  it('survives extreme arguments (zero aa, huge and negative positions, tiny density)', async () => {
    const out = await gpuF32(
      `${BACKDROP}
       fn f(x: vec4f) -> vec4f {
         let a = backdrop_sample(0, x.xy, 1.0, 0.0, 0.0);
         let b = backdrop_sample(1, x.xy, 3.0, 0.0, 1.0);
         let c = backdrop_sample(2, x.xy, 1.0, 0.0, 50.0);
         return vec4f(a.x + b.y + c.z, a.y + b.z + c.x, a.z + b.x + c.y, 1.0);
       }`,
      randomRows(2000, 51, [-50, -50, 0, 0], [50, 50, 0, 0]),
    );
    for (const v of out) expect(Number.isFinite(v)).toBe(true);
  });

  it('draws pool tiles with straight grout between lighter, repeating tiles', async () => {
    // Across one tile row: grout on the integer lines, tile colour in between, same pattern every tile.
    const xs = [0.0, 0.5, 1.0, 2.0, 3.5, 5.0, 8.0, 9.5];
    const out = await gpuF32(
      `${BACKDROP}
       fn f(x: vec4f) -> vec4f { return vec4f(backdrop_pool_tiles(vec2f(x.x, 0.5), 0.0), 1.0); }`,
      rowsOf(xs.map((x) => [x])),
    );
    const r = (i: number) => out[i * 4] ?? 0;
    // Red channel: grout (pale grey) is much redder than the aqua tile.
    expect(r(0)).toBeGreaterThan((r(1) ?? 0) * 2);
    expect(r(2)).toBeGreaterThan((r(4) ?? 0) * 2);
    expect(r(3)).toBeCloseTo(r(2), 6);
    expect(r(5)).toBeCloseTo(r(2), 6);
    // Grout is a thin line: 0.04 of a tile either side of the edge is still grout, 0.2 is tile.
    const [near, far] = await scalars(BACKDROP, [
      'backdrop_pool_tiles(vec2f(3.03, 0.5), 0.0).x',
      'backdrop_pool_tiles(vec2f(3.2, 0.5), 0.0).x',
    ]);
    expect(near).toBeGreaterThan((far ?? 0) * 2);
  });

  it('puts straight lines on the test card where a distortion check needs them', async () => {
    // Along the line y = 0.5 (a major grid line): dark at x = 0.2 aspect-independent grid crossings.
    const [onLine, offLine, bar, ramp] = await scalars(BACKDROP, [
      'backdrop_test_card(vec2f(0.1 / 1.5, 0.35), 1.5, 0.0).x',
      'backdrop_test_card(vec2f(0.05 / 1.5, 0.35), 1.5, 0.0).x',
      'backdrop_test_card(vec2f(0.05, 0.03), 1.5, 0.0).x',
      'backdrop_test_card(vec2f(0.9, 0.96), 1.5, 0.0).x',
    ]);
    expect(onLine).toBeLessThan(0.1);
    expect(offLine).toBeGreaterThan(0.2);
    expect(bar).toBeGreaterThan(0.4); // first colour bar is white
    expect(ramp).toBeGreaterThan(0.5); // ramp is light at the right
  });
});
