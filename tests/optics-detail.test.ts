import { describe, expect, it, vi } from 'vitest';
import {
  createFlowMap,
  createFlowSampler,
  createWaterNormalMap,
  createWaterSampler,
} from '../src/gpu/bake.ts';
import { estimateTextureBytes } from '../src/gpu/texture-memory.ts';
import { expectValid, getTestDevice, readTexture } from './gpu.ts';
import { gpuF32, lane, rowsOf, scalars } from './wgsl-eval.ts';

vi.setConfig({ testTimeout: 30_000 });

// --- Helpers -------------------------------------------------------------------------------------

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

/** IEEE half to number. */
function halfToFloat(h: number): number {
  const sign = h & 0x8000 ? -1 : 1;
  const exponent = (h >> 10) & 0x1f;
  const fraction = h & 0x3ff;
  if (exponent === 0) return sign * 2 ** -14 * (fraction / 1024);
  if (exponent === 31) return fraction ? Number.NaN : sign * Infinity;
  return sign * 2 ** (exponent - 15) * (1 + fraction / 1024);
}

/** Reads mip `level` of a 2D texture as tightly packed bytes. */
async function readLevel(
  device: GPUDevice,
  texture: GPUTexture,
  level: number,
  bytesPerPixel: number,
): Promise<Uint8Array> {
  const width = Math.max(1, texture.width >> level);
  const height = Math.max(1, texture.height >> level);
  const bytesPerRow = Math.ceil((width * bytesPerPixel) / 256) * 256;
  const buffer = device.createBuffer({
    size: bytesPerRow * height,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });
  const encoder = device.createCommandEncoder();
  encoder.copyTextureToBuffer({ texture, mipLevel: level }, { buffer, bytesPerRow }, [
    width,
    height,
  ]);
  device.queue.submit([encoder.finish()]);
  await buffer.mapAsync(GPUMapMode.READ);
  const padded = new Uint8Array(buffer.getMappedRange());
  const out = new Uint8Array(width * height * bytesPerPixel);
  for (let y = 0; y < height; y++) {
    out.set(
      padded.subarray(y * bytesPerRow, y * bytesPerRow + width * bytesPerPixel),
      y * width * bytesPerPixel,
    );
  }
  buffer.unmap();
  buffer.destroy();
  return out;
}

/** Channels of a half-float texture level as numbers, `channels` per texel. */
async function readHalf(
  device: GPUDevice,
  texture: GPUTexture,
  level: number,
  channels: number,
): Promise<Float32Array> {
  const bytes = await readLevel(device, texture, level, channels * 2);
  const halves = new Uint16Array(bytes.buffer, bytes.byteOffset, bytes.byteLength / 2);
  return Float32Array.from(halves, halfToFloat);
}

const rms = (values: ArrayLike<number>) => {
  let sum = 0;
  for (let i = 0; i < values.length; i++) sum += (values[i] ?? 0) ** 2;
  return Math.sqrt(sum / values.length);
};

/** Random tangent-space xy pairs inside a disc of the given radius, as two vec2 per row. */
function randomPairs(count: number, seed: number, radius: number): Float32Array {
  const random = makeRandom(seed);
  const rows = new Float32Array(count * 4);
  for (let i = 0; i < count; i++) {
    for (const base of [0, 2]) {
      const angle = random() * Math.PI * 2;
      const r = radius * Math.sqrt(random());
      rows[i * 4 + base] = r * Math.cos(angle);
      rows[i * 4 + base + 1] = r * Math.sin(angle);
    }
  }
  return rows;
}

const NORMALS = '#include "normals.wgsl"\n';
const FLOW = '#include "flow.wgsl"\n';

// --- normals.wgsl --------------------------------------------------------------------------------

describe('normals.wgsl', () => {
  const BLENDS = ['linear', 'udn', 'whiteout', 'pd', 'rnm'] as const;
  const pairCode = (body: string) => `${NORMALS}
    fn f(x: vec4f) -> vec4f {
      let a = normal_from_xy(x.xy);
      let b = normal_from_xy(x.zw);
      ${body}
    }`;

  it('every blend returns a unit normal pointing out of the surface', async () => {
    const input = randomPairs(4000, 11, 0.8);
    for (const radius of [0.3, 0.8]) {
      const rows = randomPairs(2000, 12, radius);
      const out = await gpuF32(
        pairCode(`return vec4f(
          length(normal_blend_linear(a, b)), length(normal_blend_udn(a, b)),
          length(normal_blend_whiteout(a, b)), length(normal_blend_pd(a, b)));`),
        rows,
      );
      const rnm = await gpuF32(
        pairCode(
          `let r = normal_blend_rnm(a, b); return vec4f(length(r), r.z, normal_blend_pd(a, b).z, 0.0);`,
        ),
        rows,
      );
      for (let i = 0; i < out.length; i++) expect(out[i]).toBeCloseTo(1, 5);
      for (const v of lane(rnm, 0)) expect(v).toBeCloseTo(1, 5);
      // Two normals tilted 45 degrees each can sum past the horizon; below that they cannot.
      if (radius < 0.5) for (const v of lane(rnm, 1)) expect(v).toBeGreaterThan(0);
    }
    expect(input.length).toBe(16000);
  });

  it('blending with a flat detail returns the base for UDN, whiteout, PD and RNM; linear halves it', async () => {
    const rows = randomPairs(1000, 21, 0.7);
    // Detail is flat: zero its xy.
    for (let i = 0; i < rows.length; i += 4) {
      rows[i + 2] = 0;
      rows[i + 3] = 0;
    }
    const code = (fn: string) =>
      pairCode(`let r = ${fn}(a, b); return vec4f(r - a, 0.0);`).replace(
        'vec4f(r - a, 0.0)',
        'vec4f(r - a, length(r.xy) / length(a.xy))',
      );
    for (const blend of ['udn', 'whiteout', 'pd', 'rnm'] as const) {
      const out = await gpuF32(code(`normal_blend_${blend}`), rows);
      let worst = 0;
      for (let i = 0; i < out.length; i += 4) {
        worst = Math.max(
          worst,
          Math.abs(out[i] ?? 0),
          Math.abs(out[i + 1] ?? 0),
          Math.abs(out[i + 2] ?? 0),
        );
      }
      expect(worst, blend).toBeLessThan(2e-6);
    }
    // The naive blend: xy stays but z doubles, so every slope is roughly halved.
    const linear = await gpuF32(code('normal_blend_linear'), rows);
    const ratios = lane(linear, 3).filter((v) => Number.isFinite(v));
    const meanRatio = ratios.reduce((s, v) => s + v, 0) / ratios.length;
    expect(meanRatio).toBeGreaterThan(0.4);
    expect(meanRatio).toBeLessThan(0.65);
  });

  it('RNM, whiteout and PD with a flat base return the detail', async () => {
    const rows = randomPairs(1000, 22, 0.7);
    for (let i = 0; i < rows.length; i += 4) {
      rows[i] = 0;
      rows[i + 1] = 0;
    }
    for (const blend of ['rnm', 'whiteout', 'pd'] as const) {
      const out = await gpuF32(
        pairCode(`return vec4f(normal_blend_${blend}(a, b) - b, 0.0);`),
        rows,
      );
      expect(Math.max(...out.map(Math.abs)), blend).toBeLessThan(2e-6);
    }
    // UDN keeps the detail's xy but not its z, so it is only approximately the detail.
    const udn = await gpuF32(pairCode('return vec4f(normal_blend_udn(a, b) - b, 0.0);'), rows);
    expect(Math.max(...udn.map(Math.abs))).toBeLessThan(0.2);
  });

  it('for small slopes every sound blend adds the slopes', async () => {
    const rows = randomPairs(1000, 23, 0.2);
    const out = await gpuF32(
      pairCode(`
        let sum = normalize(vec3f(a.xy + b.xy, 1.0));
        return vec4f(
          length(normal_blend_udn(a, b) - sum), length(normal_blend_whiteout(a, b) - sum),
          length(normal_blend_pd(a, b) - sum), length(normal_blend_rnm(a, b) - sum));`),
      rows,
    );
    // Within 2% of the plain slope sum at slopes up to 0.2 (second-order differences only).
    expect(Math.max(...out)).toBeLessThan(0.03);
    const tiny = await gpuF32(
      pairCode(`
        let sum = normalize(vec3f(a.xy + b.xy, 1.0));
        return vec4f(length(normal_blend_udn(a, b) - sum), length(normal_blend_whiteout(a, b) - sum),
          length(normal_blend_pd(a, b) - sum), length(normal_blend_rnm(a, b) - sum));`),
      randomPairs(1000, 24, 0.05),
    );
    expect(Math.max(...tiny)).toBeLessThan(5e-3);
  });

  it('RNM follows the base tilt where UDN and whiteout do not', async () => {
    // A steep base (slope ~1.2) with a detail tilted along +x: RNM and the true heightfield sum (PD)
    // differ by the rotation, whiteout and PD agree to first order.
    const out = await gpuF32(
      pairCode('return vec4f(normal_blend_rnm(a, b), normal_blend_whiteout(a, b).x);'),
      rowsOf([[0.7, 0, 0.2, 0]]),
    );
    const rnm = [out[0] ?? 0, out[1] ?? 0, out[2] ?? 0];
    expect(Math.hypot(...rnm)).toBeCloseTo(1, 5);
    // Both lean the same way as the base, and the detail adds slope.
    expect(rnm[0]).toBeGreaterThan(0.7);
    expect(out[3] ?? 0).toBeGreaterThan(0.7);
  });

  it('normal_scale: strength 0 is flat, 1 is the identity, 2 doubles the slope', async () => {
    const rows = randomPairs(500, 31, 0.5);
    const out = await gpuF32(
      `${NORMALS}
       fn f(x: vec4f) -> vec4f {
         let n = normal_from_xy(x.xy);
         let flat = normal_scale(n, 0.0);
         let same = normal_scale(n, 1.0);
         let twice = normal_scale(n, 2.0);
         let s2 = twice.xy / twice.z;
         let s1 = n.xy / n.z;
         return vec4f(length(flat - vec3f(0.0, 0.0, 1.0)), length(same - n),
           length(s2 - 2.0 * s1), length(twice));
       }`,
      rows,
    );
    expect(Math.max(...lane(out, 0))).toBeLessThan(1e-6);
    expect(Math.max(...lane(out, 1))).toBeLessThan(1e-6);
    expect(Math.max(...lane(out, 2))).toBeLessThan(1e-5);
    for (const v of lane(out, 3)) expect(v).toBeCloseTo(1, 5);
  });

  it('decodes xy to a unit normal and survives out-of-range data', async () => {
    const [flat, unit, over, signedZ, unormFlat, unormTilt, encoded] = await scalars(NORMALS, [
      'normal_from_xy(vec2f(0.0)).z',
      'length(normal_from_xy(vec2f(0.3, -0.4)))',
      // |xy| = 1.3 is bad data (filter overshoot): it must come back as a finite unit vector.
      'length(normal_from_xy(vec2f(1.3, 0.0)))',
      'normal_decode_signed(vec4f(0.6, 0.0, 0.0, 1.0)).z',
      'normal_decode_unorm(vec4f(0.5, 0.5, 1.0, 1.0)).z',
      'normal_decode_unorm(vec4f(1.0, 0.5, 0.0, 1.0)).z',
      'length(normal_decode_unorm(vec4f(normal_encode_unorm(normalize(vec3f(0.2, -0.3, 0.9))), 1.0)) - normalize(vec3f(0.2, -0.3, 0.9)))',
    ]);
    expect(flat).toBeCloseTo(1, 6);
    expect(unit).toBeCloseTo(1, 6);
    expect(over).toBeCloseTo(1, 6);
    expect(signedZ).toBeCloseTo(0.8, 6);
    expect(unormFlat).toBeCloseTo(1, 6);
    expect(unormTilt).toBeCloseTo(0, 3);
    // The z reconstruction from rg only: the encoded normal (0.2, -0.3, 0.9...) comes back.
    expect(encoded).toBeLessThan(1e-6);
  });

  it('normal_from_gradient and the world mapping describe the same surface', async () => {
    const [sx, sy, len, wx, wy, wz, flatY, yawX, yawY, yawZ] = await scalars(NORMALS, [
      // Slope of the normal is the negated gradient.
      'normal_from_gradient(vec2f(0.3, -0.2), 1.0).x / normal_from_gradient(vec2f(0.3, -0.2), 1.0).z',
      'normal_from_gradient(vec2f(0.3, -0.2), 1.0).y / normal_from_gradient(vec2f(0.3, -0.2), 1.0).z',
      'length(normal_from_gradient(vec2f(3.0, 4.0), 1.0))',
      // The surface y = 0.3 x tilts toward -x: normal (-0.3, 1, 0) normalised.
      'normal_to_world_up(normal_from_gradient(vec2f(0.3, 0.0), 1.0)).x',
      'normal_to_world_up(normal_from_gradient(vec2f(0.3, 0.0), 1.0)).y',
      'normal_to_world_up(normal_from_gradient(vec2f(0.3, 0.0), 1.0)).z',
      'normal_to_world(vec3f(0.0, 0.0, 1.0), vec3f(1.0, 0.0, 0.0), vec3f(0.0, 0.0, -1.0), vec3f(0.0, 1.0, 0.0)).y',
      'normal_yaw(normalize(vec3f(0.5, 0.0, 1.0)), 1.5707963).x',
      'normal_yaw(normalize(vec3f(0.5, 0.0, 1.0)), 1.5707963).y',
      'normal_yaw(normalize(vec3f(0.5, 0.0, 1.0)), 1.5707963).z',
    ]);
    expect(sx).toBeCloseTo(-0.3, 6);
    expect(sy).toBeCloseTo(0.2, 6);
    expect(len).toBeCloseTo(1, 6);
    const norm = Math.hypot(0.3, 1);
    expect(wx).toBeCloseTo(-0.3 / norm, 6);
    expect(wy).toBeCloseTo(1 / norm, 6);
    expect(wz).toBeCloseTo(0, 6);
    expect(flatY).toBeCloseTo(1, 6);
    // Quarter-turn about the normal: x slope moves to y, z unchanged.
    expect(yawX).toBeCloseTo(0, 6);
    expect(yawY).toBeCloseTo(0.5 / Math.hypot(0.5, 1), 6);
    expect(yawZ).toBeCloseTo(1 / Math.hypot(0.5, 1), 6);
    expect(BLENDS).toHaveLength(5);
  });

  it('normal_yaw undoes a uv rotation: a rotated plane gives the same world slope', async () => {
    // Height h(u) = a . u with u = R(angle) q. In q space the gradient is R^T a, so the world normal of
    // the rotated lookup must equal the normal of the plane written directly in q.
    const out = await gpuF32(
      `#include "normals.wgsl"
       #include "math.wgsl"
       fn f(x: vec4f) -> vec4f {
         let a = x.xy;
         let angle = x.z;
         let n_uv = normal_from_gradient(a, 1.0);
         let n_q = normal_yaw(n_uv, -angle);
         let grad_q = rotate_2d(a, -angle);
         let expected = normal_from_gradient(grad_q, 1.0);
         return vec4f(n_q - expected, 0.0);
       }`,
      rowsOf([
        [0.3, -0.2, 0.7, 0],
        [-0.5, 0.4, 2.4, 0],
        [0.1, 0.9, -1.3, 0],
      ]),
    );
    expect(Math.max(...out.map(Math.abs))).toBeLessThan(1e-6);
  });
});

// --- flow.wgsl -----------------------------------------------------------------------------------

describe('flow.wgsl', () => {
  // x = (time, phase offset, flow.x, flow.y); uv and the cycle length are fixed.
  const CYCLE = 1.5;
  const phasesCode = (tail: string) => `${FLOW}
    const CYCLE: f32 = ${CYCLE};
    fn g(uv: vec2f) -> f32 { return sin(7.0 * uv.x + 1.0) * cos(5.0 * uv.y - 0.5); }
    fn f(x: vec4f) -> vec4f {
      let uv = vec2f(0.37, 0.61);
      let ph = flow_phases(uv, x.zw, x.x, CYCLE, x.y);
      ${tail}
    }`;

  function sweep(count: number, seed: number): Float32Array {
    const random = makeRandom(seed);
    const rows = new Float32Array(count * 4);
    for (let i = 0; i < count; i++) {
      rows[i * 4] = random() * 30;
      rows[i * 4 + 1] = random();
      rows[i * 4 + 2] = random() * 1.6 - 0.8;
      rows[i * 4 + 3] = random() * 1.6 - 0.8;
    }
    return rows;
  }

  it('weights are in [0, 1] and sum to 1', async () => {
    const out = await gpuF32(
      phasesCode('return vec4f(ph.weight_a, ph.weight_b, ph.weight_a + ph.weight_b, 0.0);'),
      sweep(4000, 41),
    );
    for (const v of lane(out, 0)) expect(v).toBeGreaterThanOrEqual(0);
    for (const v of lane(out, 0)) expect(v).toBeLessThanOrEqual(1);
    for (const v of lane(out, 1)) expect(v).toBeGreaterThanOrEqual(0);
    for (const v of lane(out, 1)) expect(v).toBeLessThanOrEqual(1);
    for (const v of lane(out, 2)) expect(Math.abs(v - 1)).toBeLessThan(2e-6);
  });

  it('with zero flow both uv sets are exactly the input uv', async () => {
    const rows = sweep(500, 42);
    for (let i = 0; i < rows.length; i += 4) {
      rows[i + 2] = 0;
      rows[i + 3] = 0;
    }
    const out = await gpuF32(phasesCode('return vec4f(ph.uv_a - uv, ph.uv_b - uv);'), rows);
    expect(Math.max(...out.map(Math.abs))).toBe(0);
  });

  it('a phase starts from zero offset at its reset and moves with the flow', async () => {
    // phase_a = fract(t / 1.5 + offset). Reset at t = 1.5 * (1 - offset) + 1.5k; probe just after it.
    const offset = 0.3;
    const delta = 1e-3;
    const flow = [0.5, -0.25];
    const rows: number[][] = [];
    for (const k of [0, 1, 7]) {
      const reset = CYCLE * (1 - offset) + CYCLE * k;
      rows.push([reset + delta, offset, flow[0] ?? 0, flow[1] ?? 0]);
      // Half a cycle later it is the other copy's turn.
      rows.push([reset + CYCLE / 2 + delta, offset, flow[0] ?? 0, flow[1] ?? 0]);
    }
    const out = await gpuF32(
      phasesCode(
        'return vec4f(length(ph.uv_a - uv), ph.weight_a, length(ph.uv_b - uv), ph.weight_b);',
      ),
      rowsOf(rows),
    );
    const flowLength = Math.hypot(flow[0] ?? 0, flow[1] ?? 0);
    for (let r = 0; r < rows.length; r += 2) {
      // Just after A's reset: A is young (tiny offset, tiny weight), B is mid-cycle (full weight).
      expect(out[r * 4]).toBeCloseTo(flowLength * delta, 4);
      expect(out[r * 4 + 1] ?? 1).toBeLessThan((2 * delta) / CYCLE + 1e-4);
      expect(out[r * 4 + 3] ?? 0).toBeGreaterThan(0.99);
      // Half a cycle later the roles swap.
      expect(out[(r + 1) * 4 + 2]).toBeCloseTo(flowLength * delta, 4);
      expect(out[(r + 1) * 4 + 3] ?? 1).toBeLessThan((2 * delta) / CYCLE + 1e-4);
      expect(out[(r + 1) * 4 + 1] ?? 0).toBeGreaterThan(0.99);
    }
    // And a full cycle after the reset an offset has travelled flow * cycle (at weight 0).
    const end = await gpuF32(
      phasesCode('return vec4f(length(ph.uv_a - uv), ph.weight_a, 0.0, 0.0);'),
      rowsOf([[CYCLE * (1 - offset) + CYCLE - delta, offset, flow[0] ?? 0, flow[1] ?? 0]]),
    );
    expect(end[0]).toBeCloseTo(flowLength * CYCLE * (1 - delta / CYCLE), 3);
    expect(end[1] ?? 1).toBeLessThan((2 * delta) / CYCLE + 1e-4);
  });

  it('the blended result is continuous in time, including across resets', async () => {
    // A texture g with |grad g| < 9. The blend of two copies may change per step by at most the drift
    // of each copy (|flow| dt * 9) plus what the weights move (2 dt / cycle each, |g| <= 1).
    const dt = 1 / 240;
    const steps = 1800; // 7.5 s: five cycles, so several resets per pixel
    const flows: [number, number, number][] = [
      [0.4, 0.0, 0.0],
      [-0.3, 0.5, 0.35],
      [0.8, 0.8, 0.9],
    ];
    for (const [fx, fy, offset] of flows) {
      const rows: number[][] = [];
      for (let i = 0; i <= steps; i++) rows.push([i * dt, offset, fx, fy]);
      const out = await gpuF32(
        phasesCode(
          'return vec4f(ph.weight_a * g(ph.uv_a) + ph.weight_b * g(ph.uv_b), 0.0, 0.0, 0.0);',
        ),
        rowsOf(rows),
      );
      const values = lane(out, 0);
      const bound = 9 * Math.hypot(fx, fy) * dt + 2 * 2 * (dt / CYCLE) + 1e-4;
      let worst = 0;
      for (let i = 1; i < values.length; i++) {
        worst = Math.max(worst, Math.abs((values[i] ?? 0) - (values[i - 1] ?? 0)));
      }
      expect(worst, `flow ${fx},${fy}`).toBeLessThan(bound);
    }
  });

  it('per-pixel phase noise stays in [0, 1], varies smoothly and spreads over the range', async () => {
    const random = makeRandom(43);
    const rows = new Float32Array(4000 * 4);
    for (let i = 0; i < rows.length; i++) rows[i] = random() * 40;
    const out = await gpuF32(
      `${FLOW} fn f(x: vec4f) -> vec4f { return vec4f(flow_phase_noise(x.xy), flow_phase_noise(x.xy + vec2f(0.01, 0.0)), 0.0, 0.0); }`,
      rows,
    );
    const a = lane(out, 0);
    const b = lane(out, 1);
    expect(Math.min(...a)).toBeGreaterThanOrEqual(0);
    expect(Math.max(...a)).toBeLessThanOrEqual(1);
    expect(Math.min(...a)).toBeLessThan(0.1);
    expect(Math.max(...a)).toBeGreaterThan(0.9);
    const mean = a.reduce((s, v) => s + v, 0) / a.length;
    expect(mean).toBeGreaterThan(0.45);
    expect(mean).toBeLessThan(0.55);
    // A 0.01 lattice step moves the value by only a little: no hash-like jumps.
    expect(Math.max(...a.map((v, i) => Math.abs(v - (b[i] ?? 0))))).toBeLessThan(0.1);
  });

  it('flow_blend_slope keeps the variance of independent copies and the weights are a partition', async () => {
    const [only, equal, orthogonal] = await scalars(FLOW, [
      'flow_blend_slope(vec2f(0.3, -0.2), vec2f(9.0), 1.0, 0.0).x',
      // Two orthogonal unit inputs at 50/50: plain mixing gives length 0.707, this keeps 1.
      'length(flow_blend_slope(vec2f(1.0, 0.0), vec2f(0.0, 1.0), 0.5, 0.5))',
      'length(0.5 * (vec2f(1.0, 0.0) + vec2f(0.0, 1.0)))',
    ]);
    expect(only).toBeCloseTo(0.3, 6);
    expect(equal).toBeCloseTo(1, 6);
    expect(orthogonal).toBeCloseTo(Math.SQRT1_2, 6);
  });

  it('strength follows the flow speed', async () => {
    const [calm, fast, mid, speed, capped] = await scalars(FLOW, [
      'flow_strength(vec2f(0.0), 0.3, 1.2)',
      'flow_strength(vec2f(1.0, 0.0), 0.3, 1.2)',
      'flow_strength(vec2f(0.0, 0.5), 0.3, 1.2)',
      'flow_speed(vec2f(0.3, 0.4))',
      'flow_speed(vec2f(3.0, 4.0))',
    ]);
    expect(calm).toBeCloseTo(0.3, 6);
    expect(fast).toBeCloseTo(1.2, 6);
    expect(mid).toBeCloseTo(0.75, 6);
    expect(speed).toBeCloseTo(0.5, 6);
    expect(capped).toBe(1);
  });

  it('encodes velocity into bytes and back; zero is exact and the error is at most one level', async () => {
    const random = makeRandom(44);
    const rows = new Float32Array(2000 * 4);
    for (let i = 0; i < rows.length; i += 4) {
      rows[i] = random() * 2 - 1;
      rows[i + 1] = random() * 2 - 1;
    }
    // pack4x8unorm rounds to the nearest byte, exactly what an rgba8unorm render target does.
    const out = await gpuF32(
      `${FLOW}
       fn f(x: vec4f) -> vec4f {
         let bytes = unpack4x8unorm(pack4x8unorm(vec4f(flow_encode(x.xy), 0.0, 0.0)));
         return vec4f(flow_decode(bytes) - x.xy, flow_decode(vec4f(flow_encode(vec2f(0.0)), 0.0, 0.0)));
       }`,
      rows,
    );
    expect(Math.max(...lane(out, 0).map(Math.abs))).toBeLessThanOrEqual(0.5 / 127 + 1e-6);
    expect(Math.max(...lane(out, 1).map(Math.abs))).toBeLessThanOrEqual(0.5 / 127 + 1e-6);
    for (const v of lane(out, 2)) expect(v).toBe(0);
    for (const v of lane(out, 3)) expect(v).toBe(0);
    const [one, minusOne, clamped, naiveZero] = await scalars(FLOW, [
      'flow_decode(vec4f(flow_encode(vec2f(1.0, 0.0)), 0.0, 0.0)).x',
      'flow_decode(vec4f(flow_encode(vec2f(-1.0, 0.0)), 0.0, 0.0)).x',
      'flow_decode(vec4f(flow_encode(vec2f(5.0, 0.0)), 0.0, 0.0)).x',
      // The common encoding: byte 128 is "zero" but decodes to 1/127, a steady drift.
      'flow_decode(vec4f(128.0 / 255.0, 0.0, 0.0, 0.0)).x',
    ]);
    expect(one).toBeCloseTo(1, 6);
    expect(minusOne).toBeCloseTo(-1, 6);
    expect(clamped).toBeCloseTo(1, 6);
    expect(naiveZero).toBeCloseTo(1 / 127, 6);
  });
});

// --- Baked textures ------------------------------------------------------------------------------

describe('createWaterNormalMap', () => {
  const SIZE = 256;

  async function bake(options: Parameters<typeof createWaterNormalMap>[1] = {}) {
    const device = await getTestDevice();
    const map = await expectValid(device, () =>
      createWaterNormalMap(device, { size: SIZE, ...options }),
    );
    const data = await readHalf(device, map.texture, 0, 2);
    return { device, map, data };
  }

  /** Mean absolute difference between texel (x, y) and (x + 1, y), over all rows, for column x. */
  function columnStep(data: Float32Array, x0: number, x1: number): number {
    let sum = 0;
    for (let y = 0; y < SIZE; y++) {
      for (let c = 0; c < 2; c++) {
        sum += Math.abs(
          (data[(y * SIZE + x0) * 2 + c] ?? 0) - (data[(y * SIZE + x1) * 2 + c] ?? 0),
        );
      }
    }
    return sum / (SIZE * 2);
  }
  function rowStep(data: Float32Array, y0: number, y1: number): number {
    let sum = 0;
    for (let x = 0; x < SIZE; x++) {
      for (let c = 0; c < 2; c++) {
        sum += Math.abs(
          (data[(y0 * SIZE + x) * 2 + c] ?? 0) - (data[(y1 * SIZE + x) * 2 + c] ?? 0),
        );
      }
    }
    return sum / (SIZE * 2);
  }

  it('tiles: opposite edges are as close as any two neighbouring columns and rows', async () => {
    const { map, data } = await bake();
    let interiorX = 0;
    let interiorY = 0;
    for (let i = 1; i < SIZE - 2; i++) {
      interiorX += columnStep(data, i, i + 1);
      interiorY += rowStep(data, i, i + 1);
    }
    interiorX /= SIZE - 3;
    interiorY /= SIZE - 3;
    // A map that did not tile would jump by about the size of the signal (~2 * RMS slope) at the edge.
    expect(columnStep(data, SIZE - 1, 0)).toBeLessThan(1.4 * interiorX);
    expect(rowStep(data, SIZE - 1, 0)).toBeLessThan(1.4 * interiorY);
    expect(rowStep(data, SIZE - 1, 0)).toBeLessThan(0.5 * rowStep(data, 0, SIZE / 2));
    expect(columnStep(data, SIZE - 1, 0)).toBeLessThan(0.5 * columnStep(data, 0, SIZE / 2));
    map.destroy();
  });

  it('decodes to finite unit normals with a gentle, zero-mean slope', async () => {
    const { map, data } = await bake();
    let sumX = 0;
    let sumY = 0;
    let maxLength = 0;
    for (let i = 0; i < SIZE * SIZE; i++) {
      const x = data[i * 2] ?? Number.NaN;
      const y = data[i * 2 + 1] ?? Number.NaN;
      expect(Number.isFinite(x) && Number.isFinite(y)).toBe(true);
      sumX += x;
      sumY += y;
      maxLength = Math.max(maxLength, Math.hypot(x, y));
    }
    // |xy| < 1 leaves a real z = sqrt(1 - |xy|^2), so the decoded normal has exactly unit length.
    expect(maxLength).toBeLessThan(0.97);
    expect(Math.abs(sumX / SIZE ** 2)).toBeLessThan(0.02);
    expect(Math.abs(sumY / SIZE ** 2)).toBeLessThan(0.02);
    // Documented calibration: strength 1 is an RMS slope around 0.25.
    const slope = rms(data) * Math.SQRT2;
    expect(slope).toBeGreaterThan(0.15);
    expect(slope).toBeLessThan(0.4);
    map.destroy();
  });

  it('strength scales the slope and a seed gives a different but reproducible pattern', async () => {
    const base = await bake();
    const strong = await bake({ strength: 2 });
    const again = await bake();
    const other = await bake({ seed: 2 });
    const ratio = rms(strong.data) / rms(base.data);
    expect(ratio).toBeGreaterThan(1.6);
    expect(ratio).toBeLessThan(2.1);
    expect(Array.from(again.data.slice(0, 2000))).toEqual(Array.from(base.data.slice(0, 2000)));
    let same = 0;
    for (let i = 0; i < base.data.length; i++) {
      if (Math.abs((base.data[i] ?? 0) - (other.data[i] ?? 0)) < 1e-4) same++;
    }
    expect(same / base.data.length).toBeLessThan(0.05);
    for (const m of [base, strong, again, other]) m.map.destroy();
  });

  it('has a full mip chain that flattens with distance instead of turning to noise', async () => {
    const { device, map } = await bake();
    expect(map.mipLevelCount).toBe(9);
    expect(map.texture.mipLevelCount).toBe(9);
    const slopes: number[] = [];
    for (let level = 0; level < map.mipLevelCount; level++) {
      slopes.push(rms(await readHalf(device, map.texture, level, 2)));
    }
    for (let level = 1; level < slopes.length; level++) {
      expect(slopes[level], `level ${level}`).toBeLessThan((slopes[level - 1] ?? 0) * 1.02);
    }
    expect(slopes[4]).toBeLessThan((slopes[0] ?? 0) * 0.5);
    expect(slopes[8]).toBeLessThan((slopes[0] ?? 0) * 0.1);
    // Level 1 is the 2x2 box average of level 0 (the slope average the filter should produce).
    const l0 = await readHalf(device, map.texture, 0, 2);
    const l1 = await readHalf(device, map.texture, 1, 2);
    let worst = 0;
    for (let y = 0; y < 16; y++) {
      for (let x = 0; x < 16; x++) {
        for (let c = 0; c < 2; c++) {
          const at = (xx: number, yy: number) => l0[(yy * SIZE + xx) * 2 + c] ?? 0;
          const mean =
            (at(2 * x, 2 * y) +
              at(2 * x + 1, 2 * y) +
              at(2 * x, 2 * y + 1) +
              at(2 * x + 1, 2 * y + 1)) /
            4;
          worst = Math.max(worst, Math.abs(mean - (l1[(y * (SIZE / 2) + x) * 2 + c] ?? 0)));
        }
      }
    }
    expect(worst).toBeLessThan(2e-3);
    map.destroy();
  });

  it('can keep a height channel, reports its memory, and rejects bad sizes', async () => {
    const device = await getTestDevice();
    const plain = createWaterNormalMap(device, { size: 64 });
    const tall = createWaterNormalMap(device, { size: 64, height: true });
    expect(plain.format).toBe('rg16float');
    expect(tall.format).toBe('rgba16float');
    for (const map of [plain, tall]) {
      expect(map.bytes).toBe(
        estimateTextureBytes({
          size: [map.size, map.size],
          format: map.format,
          mipLevelCount: map.mipLevelCount,
        }).bytes,
      );
    }
    const heights = await readHalf(device, tall.texture, 0, 4);
    let max = 0;
    for (let i = 2; i < heights.length; i += 4) max = Math.max(max, Math.abs(heights[i] ?? 0));
    expect(max).toBeGreaterThan(0.3);
    expect(max).toBeLessThan(1.5);
    expect(() => createWaterNormalMap(device, { size: 100 })).toThrow(/power of two/);
    await expectValid(device, () => {
      createWaterSampler(device);
      createFlowSampler(device);
    });
    plain.destroy();
    tall.destroy();
  });
});

describe('bake cost', () => {
  it('both bakes finish quickly enough to run at startup', async () => {
    const device = await getTestDevice();
    const time = async (make: () => { destroy(): void }) => {
      const start = performance.now();
      const result = make();
      await device.queue.onSubmittedWorkDone();
      const ms = performance.now() - start;
      result.destroy();
      return ms;
    };
    // Warm-up compiles pipelines the first time; measure the second bake of each.
    await time(() => createWaterNormalMap(device));
    await time(() => createFlowMap(device));
    const normals = await time(() => createWaterNormalMap(device, { size: 512 }));
    const flow = await time(() => createFlowMap(device, { width: 1024 }));
    expect(normals).toBeLessThan(1500);
    expect(flow).toBeLessThan(1500);
  });
});

describe('createFlowMap', () => {
  const WIDTH = 512;
  const HEIGHT = 256;

  interface Decoded {
    vx: Float32Array;
    vy: Float32Array;
    foam: Float32Array;
    depth: Float32Array;
    bytes: Uint8Array;
  }

  async function bake(options: Parameters<typeof createFlowMap>[1] = {}): Promise<Decoded> {
    const device = await getTestDevice();
    const map = await expectValid(device, () =>
      createFlowMap(device, { width: WIDTH, ...options }),
    );
    expect(map.width).toBe(WIDTH);
    expect(map.height).toBe(HEIGHT);
    expect(map.bytes).toBe(WIDTH * HEIGHT * 4);
    const bytes = await readTexture(device, map.texture, 4);
    map.destroy();
    const n = WIDTH * HEIGHT;
    const out: Decoded = {
      vx: new Float32Array(n),
      vy: new Float32Array(n),
      foam: new Float32Array(n),
      depth: new Float32Array(n),
      bytes,
    };
    for (let i = 0; i < n; i++) {
      // The decode flow_decode does: byte 127 is zero.
      out.vx[i] = Math.max(-1, Math.min(1, ((bytes[i * 4] ?? 0) - 127) / 127));
      out.vy[i] = Math.max(-1, Math.min(1, ((bytes[i * 4 + 1] ?? 0) - 127) / 127));
      out.foam[i] = (bytes[i * 4 + 2] ?? 0) / 255;
      out.depth[i] = (bytes[i * 4 + 3] ?? 0) / 255;
    }
    return out;
  }

  it('is finite, bounded, flows left to right, and is dry and exactly still on land', async () => {
    const flow = await bake();
    let speedSum = 0;
    let wet = 0;
    let dry = 0;
    let dryMoving = 0;
    let maxSpeed = 0;
    let signedX = 0;
    for (let i = 0; i < WIDTH * HEIGHT; i++) {
      const speed = Math.hypot(flow.vx[i] ?? 0, flow.vy[i] ?? 0);
      expect(Number.isFinite(speed)).toBe(true);
      maxSpeed = Math.max(maxSpeed, speed);
      if ((flow.depth[i] ?? 0) > 0.3) {
        wet++;
        speedSum += speed;
        signedX += flow.vx[i] ?? 0;
      }
      if (flow.bytes[i * 4 + 3] === 0) {
        dry++;
        if (flow.bytes[i * 4] !== 127 || flow.bytes[i * 4 + 1] !== 127) dryMoving++;
      }
    }
    expect(wet).toBeGreaterThan(0.3 * WIDTH * HEIGHT);
    expect(dry).toBeGreaterThan(0.05 * WIDTH * HEIGHT);
    expect(dryMoving).toBe(0);
    expect(maxSpeed).toBeLessThanOrEqual(1.42);
    expect(speedSum / wet).toBeGreaterThan(0.15);
    expect(speedSum / wet).toBeLessThan(0.7);
    expect(signedX / wet).toBeGreaterThan(0.15);
    // Foam exists, but only on a minority of the map.
    const foamy = flow.foam.filter((v) => v > 0.3).length;
    expect(foamy).toBeGreaterThan(100);
    expect(foamy).toBeLessThan(0.2 * WIDTH * HEIGHT);
  });

  it('is faster in the deep middle than in the shallows', async () => {
    const flow = await bake();
    let deepSum = 0;
    let deep = 0;
    let shallowSum = 0;
    let shallow = 0;
    for (let i = 0; i < WIDTH * HEIGHT; i++) {
      const speed = Math.hypot(flow.vx[i] ?? 0, flow.vy[i] ?? 0);
      const depth = flow.depth[i] ?? 0;
      if (depth > 0.95) {
        deepSum += speed;
        deep++;
      } else if (depth > 0.03 && depth < 0.12) {
        shallowSum += speed;
        shallow++;
      }
    }
    expect(deep).toBeGreaterThan(1000);
    expect(shallow).toBeGreaterThan(1000);
    expect(deepSum / deep).toBeGreaterThan(1.8 * (shallowSum / shallow));
  });

  it('conserves flux: the same amount of water crosses every vertical cut, rocks or not', async () => {
    const flow = await bake();
    const flux: number[] = [];
    for (let x = 0; x < WIDTH; x += 4) {
      let sum = 0;
      for (let y = 0; y < HEIGHT; y++) sum += flow.vx[y * WIDTH + x] ?? 0;
      flux.push(sum / HEIGHT);
    }
    const mean = flux.reduce((s, v) => s + v, 0) / flux.length;
    // 0.13 * 4/3 = the stream function's total rise across the channel.
    expect(mean).toBeGreaterThan(0.11 * (4 / 3) * 0.9);
    expect(mean).toBeLessThan(0.11 * (4 / 3) * 1.1);
    expect((Math.max(...flux) - Math.min(...flux)) / mean).toBeLessThan(0.1);
  });

  it('is divergence-free in open water: no net flux out of boxes of deep water', async () => {
    const flow = await bake();
    const box = 20;
    let tested = 0;
    for (let y0 = 0; y0 + box < HEIGHT; y0 += 10) {
      for (let x0 = 0; x0 + box < WIDTH; x0 += 10) {
        let allDeep = true;
        for (let y = y0; y <= y0 + box && allDeep; y++) {
          for (let x = x0; x <= x0 + box; x++) {
            if ((flow.depth[y * WIDTH + x] ?? 0) < 0.9) {
              allDeep = false;
              break;
            }
          }
        }
        if (!allDeep) continue;
        let net = 0;
        let through = 0;
        for (let t = 0; t <= box; t++) {
          const right = flow.vx[(y0 + t) * WIDTH + x0 + box] ?? 0;
          const left = flow.vx[(y0 + t) * WIDTH + x0] ?? 0;
          const bottom = flow.vy[(y0 + box) * WIDTH + x0 + t] ?? 0;
          const top = flow.vy[y0 * WIDTH + x0 + t] ?? 0;
          net += right - left + bottom - top;
          through += Math.abs(right) + Math.abs(left) + Math.abs(bottom) + Math.abs(top);
        }
        expect(Math.abs(net) / through).toBeLessThan(0.08);
        tested++;
      }
    }
    expect(tested).toBeGreaterThan(30);
  });

  it('is periodic in x and is changed by the seed and the rock count', async () => {
    const flow = await bake();
    let edge = 0;
    let inner = 0;
    for (let y = 0; y < HEIGHT; y++) {
      edge += Math.abs((flow.vx[y * WIDTH + WIDTH - 1] ?? 0) - (flow.vx[y * WIDTH] ?? 0));
      inner += Math.abs(
        (flow.vx[y * WIDTH + WIDTH / 2 - 1] ?? 0) - (flow.vx[y * WIDTH + WIDTH / 2] ?? 0),
      );
    }
    expect(edge).toBeLessThan(1.5 * inner + 0.5);
    const other = await bake({ seed: 5 });
    const none = await bake({ rocks: 0 });
    let differs = 0;
    let rockless = 0;
    for (let i = 0; i < flow.depth.length; i++) {
      if (Math.abs((flow.depth[i] ?? 0) - (other.depth[i] ?? 0)) > 0.1) differs++;
      if (Math.abs((flow.depth[i] ?? 0) - (none.depth[i] ?? 0)) > 0.1) rockless++;
    }
    expect(differs).toBeGreaterThan(200);
    expect(rockless).toBeGreaterThan(500);
    // Without rocks the water is dry only at the banks: the river is wetter.
    const wetness = (d: Decoded) => d.depth.filter((v) => v > 0.3).length;
    expect(wetness(none)).toBeGreaterThan(wetness(flow));
  });

  it('rejects widths it cannot map', async () => {
    const device = await getTestDevice();
    expect(() => createFlowMap(device, { width: 33 })).toThrow(/width/);
  });
});
