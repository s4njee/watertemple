import { describe, expect, it, vi } from 'vitest';
import { checkShader, formatDiagnostic } from '../src/gpu/shader.ts';
import { chunkNames, getChunk } from '../src/wgsl/index.ts';
import { expectValid, getTestDevice, readTexture } from './gpu.ts';
import { gpuF32, gpuU32, lane, rowsOf, scalars } from './wgsl-eval.ts';

// Brute-force comparisons run thousands of samples; leave headroom for a busy machine.
vi.setConfig({ testTimeout: 30_000 });

// --- Test data helpers ---------------------------------------------------------------------------

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

/** `count` rows of 4 floats, each uniform in [lo, hi). */
function randomRows(count: number, seed: number, lo: number, hi: number): Float32Array {
  const random = makeRandom(seed);
  const rows = new Float32Array(count * 4);
  for (let i = 0; i < rows.length; i++) rows[i] = lo + (hi - lo) * random();
  return rows;
}

function summarize(values: number[]) {
  let min = Infinity;
  let max = -Infinity;
  let sum = 0;
  for (const v of values) {
    min = Math.min(min, v);
    max = Math.max(max, v);
    sum += v;
  }
  const mean = sum / values.length;
  let squares = 0;
  for (const v of values) squares += (v - mean) ** 2;
  return { min, max, mean, variance: squares / values.length, count: values.length };
}

function correlation(a: number[], b: number[]): number {
  const sa = summarize(a);
  const sb = summarize(b);
  let covariance = 0;
  for (let i = 0; i < a.length; i++)
    covariance += ((a[i] ?? 0) - sa.mean) * ((b[i] ?? 0) - sb.mean);
  return covariance / a.length / Math.sqrt(sa.variance * sb.variance);
}

/**
 * Collects failed checks so a loop over thousands of samples costs no `expect` calls and a failure
 * reports the first offending rows instead of stopping at one.
 */
function collector() {
  const problems: string[] = [];
  return {
    near(actual: number, expected: number, tolerance: number, label: string) {
      if (!(Math.abs(actual - expected) <= tolerance))
        problems.push(`${label}: got ${actual}, expected ${expected}`);
    },
    check(condition: boolean, label: string) {
      if (!condition) problems.push(label);
    },
    /** The first few problems, for `expect(c.report()).toEqual([])`. */
    report: () => problems.slice(0, 8),
    count: () => problems.length,
  };
}

/** True when both arrays hold exactly the same bits (NaN-safe, distinguishes -0 from 0). */
function sameBits(a: Float32Array, b: Float32Array): boolean {
  const x = new Uint32Array(a.buffer, a.byteOffset, a.length);
  const y = new Uint32Array(b.buffer, b.byteOffset, b.length);
  return x.length === y.length && x.every((word, i) => word === y[i]);
}

function maxAbs(values: number[]): number {
  return values.reduce((m, v) => Math.max(m, Math.abs(v)), 0);
}

// --- Library inventory ---------------------------------------------------------------------------

/** The chunks this library owns; other chunks in src/wgsl belong to other parts of the site. */
const LIBRARY = [
  'color.wgsl',
  'curl.wgsl',
  'fbm.wgsl',
  'hash.wgsl',
  'math.wgsl',
  'noise.wgsl',
  'sdf.wgsl',
  'simplex.wgsl',
  'voronoi.wgsl',
];

describe('library chunks compile', () => {
  it('ships every library chunk', () => {
    expect(chunkNames()).toEqual(expect.arrayContaining(LIBRARY));
  });

  it.each(chunkNames())('%s compiles on its own', async (name) => {
    const device = await getTestDevice();
    const { diagnostics } = await checkShader(device, {
      label: `chunk-${name}`,
      code: `#include "${name}"`,
    });
    const errors = diagnostics.filter((d) => d.severity === 'error');
    expect(errors.map(formatDiagnostic)).toEqual([]);
    expect(diagnostics.map(formatDiagnostic)).toEqual([]);
  });

  it('all chunks compile together (no name collisions)', async () => {
    const device = await getTestDevice();
    const code = chunkNames()
      .map((name) => `#include "${name}"`)
      .join('\n');
    const { diagnostics } = await checkShader(device, { label: 'chunks-all', code });
    expect(diagnostics.map(formatDiagnostic)).toEqual([]);
  });
});

describe('source hygiene', () => {
  // Every chunk is held to the same rules, whoever added it. fullscreen.wgsl is the one chunk
  // that exists to provide an entry point.
  const owned = chunkNames()
    .filter((name) => name !== 'fullscreen.wgsl')
    .map((name) => [name, getChunk(name) ?? ''] as const);

  it.each(owned)(
    '%s is a pure library: no entry points, resources or module-scope var',
    (_name, source) => {
      // Headers may show how to declare a binding; only the code counts.
      const code = source
        .split('\n')
        .filter((line) => !line.trimStart().startsWith('//'))
        .join('\n');
      expect(code).not.toMatch(/@(vertex|fragment|compute)\b/);
      expect(code).not.toMatch(/@(group|binding)\b/);
      expect(code).not.toMatch(/^var\b/m);
    },
  );

  it.each(owned)(
    '%s starts with a header that documents every public function',
    (_name, source) => {
      const header = source
        .split('\n')
        .filter((line) => line.startsWith('//'))
        .join('\n');
      expect(source.startsWith('//')).toBe(true);
      const publicFunctions = [...source.matchAll(/^fn ([a-z]\w*)\(/gm)].map((m) => m[1] ?? '');
      expect(publicFunctions.length).toBeGreaterThan(0);
      const undocumented = publicFunctions.filter((fn) => !header.includes(fn));
      expect(undocumented).toEqual([]);
    },
  );

  it('does not use sin-based hashes', () => {
    for (const [, source] of owned) expect(source).not.toMatch(/43758|fract\s*\(\s*sin/);
  });

  it('every top-level name is declared once across the whole library', () => {
    const seen = new Map<string, string>();
    for (const [name, source] of owned) {
      for (const match of source.matchAll(/^(?:fn|const|struct)\s+(\w+)/gm)) {
        const id = match[1] ?? '';
        const previous = seen.get(id);
        expect(previous, `${id} declared in both ${previous} and ${name}`).toBeUndefined();
        seen.set(id, name);
      }
    }
  });
});

// --- Reference implementations (plain JS, exact integer maths) ------------------------------------

const u32 = (x: number): number => x >>> 0;
const mul = (a: number, b: number): number => Math.imul(a, b) >>> 0;
const LCG_MUL = 1664525;
const LCG_ADD = 1013904223;

function refPcg1(v: number): number {
  const state = u32(mul(v, 747796405) + 2891336453);
  const word = mul(((state >>> ((state >>> 28) + 4)) ^ state) >>> 0, 277803737);
  return (((word >>> 22) ^ word) >>> 0) as number;
}

function refPcg2(x: number, y: number): [number, number] {
  let vx = u32(mul(x, LCG_MUL) + LCG_ADD);
  let vy = u32(mul(y, LCG_MUL) + LCG_ADD);
  vx = u32(vx + mul(vy, LCG_MUL));
  vy = u32(vy + mul(vx, LCG_MUL));
  vx = (vx ^ (vx >>> 16)) >>> 0;
  vy = (vy ^ (vy >>> 16)) >>> 0;
  vx = u32(vx + mul(vy, LCG_MUL));
  vy = u32(vy + mul(vx, LCG_MUL));
  vx = (vx ^ (vx >>> 16)) >>> 0;
  vy = (vy ^ (vy >>> 16)) >>> 0;
  return [vx, vy];
}

function refPcg3(x: number, y: number, z: number): [number, number, number] {
  let vx = u32(mul(x, LCG_MUL) + LCG_ADD);
  let vy = u32(mul(y, LCG_MUL) + LCG_ADD);
  let vz = u32(mul(z, LCG_MUL) + LCG_ADD);
  vx = u32(vx + mul(vy, vz));
  vy = u32(vy + mul(vz, vx));
  vz = u32(vz + mul(vx, vy));
  vx = (vx ^ (vx >>> 16)) >>> 0;
  vy = (vy ^ (vy >>> 16)) >>> 0;
  vz = (vz ^ (vz >>> 16)) >>> 0;
  vx = u32(vx + mul(vy, vz));
  vy = u32(vy + mul(vz, vx));
  vz = u32(vz + mul(vx, vy));
  return [vx, vy, vz];
}

function refPcg4(x: number, y: number, z: number, w: number): [number, number, number, number] {
  let vx = u32(mul(x, LCG_MUL) + LCG_ADD);
  let vy = u32(mul(y, LCG_MUL) + LCG_ADD);
  let vz = u32(mul(z, LCG_MUL) + LCG_ADD);
  let vw = u32(mul(w, LCG_MUL) + LCG_ADD);
  vx = u32(vx + mul(vy, vw));
  vy = u32(vy + mul(vz, vx));
  vz = u32(vz + mul(vx, vy));
  vw = u32(vw + mul(vy, vz));
  vx = (vx ^ (vx >>> 16)) >>> 0;
  vy = (vy ^ (vy >>> 16)) >>> 0;
  vz = (vz ^ (vz >>> 16)) >>> 0;
  vw = (vw ^ (vw >>> 16)) >>> 0;
  vx = u32(vx + mul(vy, vw));
  vy = u32(vy + mul(vz, vx));
  vz = u32(vz + mul(vx, vy));
  vw = u32(vw + mul(vy, vz));
  return [vx, vy, vz, vw];
}

/** u32 -> [0, 1) the way to_unorm_* does it. */
const unorm = (h: number): number => (h >>> 8) / 16777216;

/** Chi-square statistic of `values` in [0, 1) against a uniform distribution over `bins` bins. */
function chiSquareUniform(values: number[], bins: number): number {
  const counts = new Array<number>(bins).fill(0);
  for (const v of values) counts[Math.min(bins - 1, Math.floor(v * bins))]! += 1;
  const expected = values.length / bins;
  return counts.reduce((sum, c) => sum + (c - expected) ** 2 / expected, 0);
}

// 99.99th percentile of chi-square with 15 degrees of freedom is ~ 44; a broken hash scores in the thousands.
const CHI_SQUARE_15_LIMIT = 50;

// --- Hash ----------------------------------------------------------------------------------------

describe('hash.wgsl', () => {
  const header = '#include "hash.wgsl"\n';

  // Words that exercise carries and the top bit, plus a deterministic spread.
  const words = (() => {
    const random = makeRandom(7);
    const input = new Uint32Array(4 * 600);
    input.set([
      0, 0, 0, 0, 1, 2, 3, 4, 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 0x80000000, 1,
      0x7fffffff, 0xdeadbeef,
    ]);
    for (let i = 16; i < input.length; i++) input[i] = Math.floor(random() * 4294967296);
    return input;
  })();

  it('pcg_1d matches the reference bit for bit', async () => {
    const out = await gpuU32(
      `${header} fn f(x: vec4u) -> vec4u { return vec4u(pcg_1d(x.x), pcg_1d(x.y), pcg_1d(x.z), pcg_1d(x.w)); }`,
      words,
    );
    for (let i = 0; i < words.length; i++) expect(out[i]).toBe(refPcg1(words[i] ?? 0));
  });

  it('pcg_2d, pcg_3d and pcg_4d match the reference bit for bit', async () => {
    const out2 = await gpuU32(
      `${header} fn f(x: vec4u) -> vec4u { return vec4u(pcg_2d(x.xy), 0u, 0u); }`,
      words,
    );
    const out3 = await gpuU32(
      `${header} fn f(x: vec4u) -> vec4u { return vec4u(pcg_3d(x.xyz), 0u); }`,
      words,
    );
    const out4 = await gpuU32(`${header} fn f(x: vec4u) -> vec4u { return pcg_4d(x); }`, words);
    for (let row = 0; row < words.length / 4; row++) {
      const [x = 0, y = 0, z = 0, w = 0] = words.subarray(row * 4, row * 4 + 4);
      expect([...out2.subarray(row * 4, row * 4 + 2)]).toEqual(refPcg2(x, y));
      expect([...out3.subarray(row * 4, row * 4 + 3)]).toEqual(refPcg3(x, y, z));
      expect([...out4.subarray(row * 4, row * 4 + 4)]).toEqual(refPcg4(x, y, z, w));
    }
  });

  it('is deterministic: the same inputs give the same bits on every run', async () => {
    const code = `${header} fn f(x: vec4f) -> vec4f { return vec4f(hash_2to1(x.xy), hash_3to1(x.xyz), hash_2to2(x.xy)); }`;
    const input = randomRows(4096, 11, -1000, 1000);
    const a = await gpuF32(code, input);
    const b = await gpuF32(code, input);
    expect(sameBits(a, b)).toBe(true);
  });

  it('to_unorm_* maps u32 onto [0, 1) with 24 bits and never reaches 1', async () => {
    const input = new Uint32Array([
      0, 0, 0, 0, 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 255, 256, 0x80000000, 0x7fffffff,
    ]);
    const mapped = await gpuU32(
      `${header} fn f(x: vec4u) -> vec4u { return bitcast<vec4u>(vec4f(to_unorm_1d(x.x), to_unorm_2d(x.xy).y, to_unorm_3d(x.xyz).z, to_unorm_4d(x).w)); }`,
      input,
    );
    const asFloat = new Float32Array(mapped.buffer);
    expect([...asFloat.subarray(0, 4)]).toEqual([0, 0, 0, 0]);
    // 0xffffffff >> 8 = 2^24 - 1, the largest value: (2^24 - 1) / 2^24 < 1 and exactly representable.
    expect([...asFloat.subarray(4, 8)]).toEqual(new Array<number>(4).fill(16777215 / 16777216));
    expect(asFloat[8]).toBe(0); // 255 >> 8 == 0
    expect(asFloat[9]).toBe(1 / 16777216); // 256 >> 8 == 1
    expect(asFloat[10]).toBe(0.5);
    expect(asFloat[11]).toBe(unorm(0x7fffffff));
  });

  it('float hashes are uniform on [0, 1) over a grid of consecutive integers', async () => {
    // 256 x 256 integer lattice, evaluated together with the three neighbours of each cell.
    const input = new Float32Array(256 * 256 * 4);
    for (let y = 0; y < 256; y++)
      for (let x = 0; x < 256; x++) input.set([x, y], (y * 256 + x) * 4);
    const out = await gpuF32(
      `${header}
       fn f(x: vec4f) -> vec4f {
         return vec4f(hash_2to1(x.xy), hash_2to1(x.xy + vec2f(1.0, 0.0)),
                      hash_2to1(x.xy + vec2f(0.0, 1.0)), hash_2to1(x.xy + vec2f(1.0, 1.0)));
       }`,
      input,
    );
    const here = lane(out, 0);
    const stats = summarize(here);
    expect(stats.min).toBeGreaterThanOrEqual(0);
    expect(stats.max).toBeLessThan(1);
    expect(stats.mean).toBeCloseTo(0.5, 2);
    expect(stats.variance).toBeCloseTo(1 / 12, 2);
    expect(chiSquareUniform(here, 16)).toBeLessThan(CHI_SQUARE_15_LIMIT);
    // No banding or lattice structure: neighbours (in x, in y and diagonally) are uncorrelated.
    for (const neighbour of [1, 2, 3])
      expect(Math.abs(correlation(here, lane(out, neighbour)))).toBeLessThan(0.02);
    // Independent uniforms differ by 1/3 on average; a smooth ramp or repeated values would not.
    const right = lane(out, 1);
    const stepSizes = here.map((v, i) => Math.abs(v - (right[i] ?? 0)));
    expect(summarize(stepSizes).mean).toBeCloseTo(1 / 3, 2);
  });

  it('hash_1to1 and hash_3to1 are uniform over consecutive integers', async () => {
    const line = new Float32Array(65536 * 4);
    for (let i = 0; i < 65536; i++) line[i * 4] = i;
    const out1 = await gpuF32(
      `${header} fn f(x: vec4f) -> vec4f { return vec4f(hash_1to1(x.x), hash_1to1(x.x + 1.0), hash_1to2(x.x)); }`,
      line,
    );
    const a = lane(out1, 0);
    expect(summarize(a).mean).toBeCloseTo(0.5, 2);
    expect(chiSquareUniform(a, 16)).toBeLessThan(CHI_SQUARE_15_LIMIT);
    expect(Math.abs(correlation(a, lane(out1, 1)))).toBeLessThan(0.02);
    expect(Math.abs(correlation(lane(out1, 2), lane(out1, 3)))).toBeLessThan(0.02);

    const cube = new Float32Array(40 * 40 * 40 * 4);
    for (let z = 0; z < 40; z++)
      for (let y = 0; y < 40; y++)
        for (let x = 0; x < 40; x++) cube.set([x, y, z], ((z * 40 + y) * 40 + x) * 4);
    const out3 = await gpuF32(
      `${header} fn f(x: vec4f) -> vec4f { return vec4f(hash_3to3(x.xyz), hash_3to1(x.xyz)); }`,
      cube,
    );
    for (const l of [0, 1, 2, 3]) {
      const values = lane(out3, l);
      expect(summarize(values).mean).toBeCloseTo(0.5, 1);
      expect(chiSquareUniform(values, 16)).toBeLessThan(CHI_SQUARE_15_LIMIT);
    }
    expect(Math.abs(correlation(lane(out3, 0), lane(out3, 1)))).toBeLessThan(0.02);
    expect(Math.abs(correlation(lane(out3, 1), lane(out3, 2)))).toBeLessThan(0.02);
  });

  it('hashes of arbitrary fractional floats and of negative cells are uniform too', async () => {
    const input = randomRows(60000, 3, -500, 500);
    const out = await gpuF32(
      `${header}
       fn f(x: vec4f) -> vec4f {
         return vec4f(hash_2to1(x.xy), ihash_2to1(vec2i(floor(x.xy))), ihash_3to1(vec3i(floor(x.xyz))), hash_2to3(x.xy).z);
       }`,
      input,
    );
    for (const l of [0, 1, 2, 3]) {
      const values = lane(out, l);
      const stats = summarize(values);
      expect(stats.min).toBeGreaterThanOrEqual(0);
      expect(stats.max).toBeLessThan(1);
      expect(stats.mean).toBeCloseTo(0.5, 1);
      expect(chiSquareUniform(values, 16)).toBeLessThan(CHI_SQUARE_15_LIMIT);
    }
  });

  it('ihash_* is the hash of the integer bits (matches the reference)', async () => {
    const cells = [
      [0, 0],
      [1, 0],
      [-1, -1],
      [5, -9],
      [123456, -654321],
      [16777216, -16777216],
    ];
    const out = await gpuF32(
      `${header} fn f(x: vec4f) -> vec4f { return vec4f(ihash_2to1(vec2i(x.xy)), ihash_2to2(vec2i(x.xy)).y, ihash_3to1(vec3i(x.xyz)), ihash_3to3(vec3i(x.xyz)).z); }`,
      rowsOf(cells.map(([x = 0, y = 0]) => [x, y, 0])),
    );
    cells.forEach(([x = 0, y = 0], i) => {
      const [h2x, h2y] = refPcg2(u32(x), u32(y));
      const [, , h3z] = refPcg3(u32(x), u32(y), 0);
      const [h3x] = refPcg3(u32(x), u32(y), 0);
      expect(out[i * 4]).toBe(unorm(h2x));
      expect(out[i * 4 + 1]).toBe(unorm(h2y));
      expect(out[i * 4 + 2]).toBe(unorm(h3x));
      expect(out[i * 4 + 3]).toBe(unorm(h3z));
    });
  });

  it('every output bit is balanced and a one-step input change flips about half the bits', async () => {
    const sequential = new Uint32Array(65536 * 4);
    for (let i = 0; i < 65536; i++) sequential[i * 4] = i;
    const out = await gpuU32(
      `${header}
       fn f(x: vec4u) -> vec4u {
         let a = pcg_1d(x.x);
         let b = pcg_1d(x.x + 1u);
         let c = pcg_2d(vec2u(x.x & 255u, x.x >> 8u));
         let d = pcg_2d(vec2u((x.x & 255u) + 1u, x.x >> 8u));
         return vec4u(a, countOneBits(a ^ b), c.x, countOneBits(c.x ^ d.x));
       }`,
      sequential,
    );
    for (const word of [0, 2]) {
      for (let bit = 0; bit < 32; bit++) {
        let ones = 0;
        for (let i = 0; i < 65536; i++) ones += ((out[i * 4 + word] ?? 0) >>> bit) & 1;
        expect(Math.abs(ones / 65536 - 0.5), `word ${word} bit ${bit}`).toBeLessThan(0.012);
      }
    }
    expect(summarize(lane(out, 1)).mean).toBeGreaterThan(15.5);
    expect(summarize(lane(out, 1)).mean).toBeLessThan(16.5);
    expect(summarize(lane(out, 3)).mean).toBeGreaterThan(15);
    expect(summarize(lane(out, 3)).mean).toBeLessThan(17);
  });

  it('integer-lattice hashes of adjacent cells are uncorrelated, negative cells included', async () => {
    const input = new Float32Array(256 * 256 * 4);
    for (let y = 0; y < 256; y++)
      for (let x = 0; x < 256; x++) input.set([x - 128, y - 128], (y * 256 + x) * 4);
    const out = await gpuF32(
      `${header}
       fn f(x: vec4f) -> vec4f {
         let c = vec2i(x.xy);
         return vec4f(ihash_2to1(c), ihash_2to1(c + vec2i(1, 0)), ihash_2to1(c + vec2i(0, 1)), ihash_3to1(vec3i(c, 7)));
       }`,
      input,
    );
    const here = lane(out, 0);
    expect(chiSquareUniform(here, 16)).toBeLessThan(CHI_SQUARE_15_LIMIT);
    for (const neighbour of [1, 2, 3])
      expect(Math.abs(correlation(here, lane(out, neighbour)))).toBeLessThan(0.02);
  });

  it('hashes the bit pattern: -0.0 and 0.0 differ (as documented), equal bit patterns agree', async () => {
    const out = await gpuF32(
      `${header}
       fn f(x: vec4f) -> vec4f {
         return vec4f(hash_1to1(-abs(x.x)), hash_1to1(abs(x.x)), hash_1to1(x.y), hash_1to1(x.y + 0.0));
       }`,
      rowsOf([[0, 3]]),
    );
    expect(out[0]).not.toBe(out[1]);
    expect(out[2]).toBe(out[3]);
  });

  it('grad_2d picks 8 unit directions and grad_3d the 12 cube-edge directions, evenly', async () => {
    // The table index uses only the low bits of the hash, so high words must wrap around.
    const input = new Float32Array(64 * 4);
    for (let i = 0; i < 64; i++) input[i * 4] = 320 + i;
    const out2d = await gpuF32(
      `${header} fn f(x: vec4f) -> vec4f { return vec4f(grad_2d(u32(x.x)), 0.0, 0.0); }`,
      input,
    );
    const out3d = await gpuF32(
      `${header} fn f(x: vec4f) -> vec4f { return vec4f(grad_3d(u32(x.x)), 0.0); }`,
      input,
    );

    const count = (keys: string[]) => {
      const counts = new Map<string, number>();
      for (const key of keys) counts.set(key, (counts.get(key) ?? 0) + 1);
      return counts;
    };
    const rows = (data: Float32Array, width: number) =>
      Array.from({ length: 64 }, (_, i) => [...data.subarray(i * 4, i * 4 + width)]);

    const dirs2d = rows(out2d, 2);
    for (const [x = 0, y = 0] of dirs2d) expect(Math.hypot(x, y)).toBeCloseTo(1, 6);
    const counts2d = count(dirs2d.map(([x = 0, y = 0]) => `${x.toFixed(3)},${y.toFixed(3)}`));
    expect(counts2d.size).toBe(8);
    expect([...counts2d.values()]).toEqual(new Array<number>(8).fill(8));

    // 12 distinct vectors with two components of +-1 and one 0; four of the 16 slots repeat one.
    const dirs3d = rows(out3d, 3);
    for (const v of dirs3d) {
      expect(v.filter((c) => Math.abs(c) === 1)).toHaveLength(2);
      expect(v.filter((c) => c === 0)).toHaveLength(1);
    }
    const counts3d = count(dirs3d.map((v) => v.join(',')));
    expect(counts3d.size).toBe(12);
    expect([...counts3d.values()].sort()).toEqual(
      [...new Array<number>(8).fill(4), ...new Array<number>(4).fill(8)].sort(),
    );
  });
});

// --- Noise ---------------------------------------------------------------------------------------

/**
 * WGSL for a central-difference check of a function returning (value, gradient) in 2D: reports
 * (error x, error y, |value - plain value|, |gradient|). The step is measured from the rounded
 * sample positions, so float rounding of p +- h does not bias the estimate.
 */
function gradientCheck2d(include: string, analytic: string, plain: string): string {
  return `${include}
    fn f(x: vec4f) -> vec4f {
      let p = x.xy;
      let h = 2e-3;
      let px1 = p + vec2f(h, 0.0);
      let px0 = p - vec2f(h, 0.0);
      let py1 = p + vec2f(0.0, h);
      let py0 = p - vec2f(0.0, h);
      let fx = (${plain}(px1) - ${plain}(px0)) / (px1.x - px0.x);
      let fy = (${plain}(py1) - ${plain}(py0)) / (py1.y - py0.y);
      let d = ${analytic}(p);
      return vec4f(d.y - fx, d.z - fy, abs(d.x - ${plain}(p)), length(d.yz));
    }`;
}

describe('noise: value and gradient (noise.wgsl)', () => {
  const include = '#include "noise.wgsl"\n';

  it('every function stays within its documented range [-1, 1], zero mean, and uses the range', async () => {
    const input = randomRows(200000, 21, -60, 60);
    const out = await gpuF32(
      `${include}
       fn f(x: vec4f) -> vec4f {
         return vec4f(vnoise_2d(x.xy), vnoise_3d(x.xyz), gnoise_2d(x.xy), gnoise_3d(x.xyz));
       }`,
      input,
    );
    const names = ['vnoise_2d', 'vnoise_3d', 'gnoise_2d', 'gnoise_3d'];
    names.forEach((name, l) => {
      const s = summarize(lane(out, l));
      expect(s.min, `${name} min`).toBeGreaterThanOrEqual(-1);
      expect(s.max, `${name} max`).toBeLessThanOrEqual(1);
      expect(s.min, `${name} min uses range`).toBeLessThan(-0.6);
      expect(s.max, `${name} max uses range`).toBeGreaterThan(0.6);
      expect(Math.abs(s.mean), `${name} mean`).toBeLessThan(0.02);
      expect(Math.sqrt(s.variance), `${name} spread`).toBeGreaterThan(0.15);
    });
    // Value noise never reaches +-1 because hashes are in [0, 1).
    for (const l of [0, 1]) {
      const s = summarize(lane(out, l));
      expect(s.min).toBeGreaterThan(-1);
      expect(s.max).toBeLessThan(1);
    }
  });

  it('value noise equals its lattice hash on integer points; gradient noise is zero there', async () => {
    const cells = [
      [0, 0, 0],
      [1, 2, 3],
      [-4, 5, -6],
      [100, -200, 300],
      [-1, -1, -1],
    ];
    const out = await gpuF32(
      `${include}
       fn f(x: vec4f) -> vec4f {
         let c3 = vec3i(x.xyz);
         let a = vnoise_2d(x.xy) - (2.0 * ihash_2to1(c3.xy) - 1.0);
         let b = vnoise_3d(x.xyz) - (2.0 * ihash_3to1(c3) - 1.0);
         return vec4f(a, b, gnoise_2d(x.xy), gnoise_3d(x.xyz));
       }`,
      rowsOf(cells),
    );
    // Value noise at a lattice point is exactly the (hash2 or hash3) value of that point's cell. The 2D
    // and 3D value noise use different hashes, so compare each against its own.
    expect(maxAbs(lane(out, 0))).toBeLessThan(1e-6);
    expect(maxAbs(lane(out, 1))).toBeLessThan(1e-6);
    expect(maxAbs(lane(out, 2))).toBe(0);
    expect(maxAbs(lane(out, 3))).toBe(0);
  });

  it('is deterministic and continuous across cell borders', async () => {
    const input = randomRows(20000, 5, -30, 30);
    const code = `${include}
      fn f(x: vec4f) -> vec4f {
        let p = round(x.xy); // a cell border in both axes
        let e = 1e-4;
        let d = vnoise_2d(p + vec2f(e)) - vnoise_2d(p - vec2f(e));
        let g = gnoise_2d(p + vec2f(e)) - gnoise_2d(p - vec2f(e));
        let g3 = gnoise_3d(vec3f(p, x.z) + vec3f(e)) - gnoise_3d(vec3f(p, x.z) - vec3f(e));
        return vec4f(d, g, g3, vnoise_3d(x.xyz));
      }`;
    const a = await gpuF32(code, input);
    const b = await gpuF32(code, input);
    expect(sameBits(a, b)).toBe(true);
    // A jump across a border would be O(1); a smooth function changes by about gradient * 2e-4.
    for (const l of [0, 1, 2]) expect(maxAbs(lane(a, l))).toBeLessThan(2e-3);
  });

  it('analytic derivatives match finite differences', async () => {
    const input = randomRows(20000, 31, -8, 8);
    const cases: [string, string, string][] = [
      ['vnoise_2d_d', 'vnoise_2d', 'value noise'],
      ['gnoise_2d_d', 'gnoise_2d', 'gradient noise'],
    ];
    for (const [analytic, plain, label] of cases) {
      const out = await gpuF32(gradientCheck2d(include, analytic, plain), input);
      expect(maxAbs(lane(out, 0)), `${label} d/dx`).toBeLessThan(1e-3);
      expect(maxAbs(lane(out, 1)), `${label} d/dy`).toBeLessThan(1e-3);
      expect(maxAbs(lane(out, 2)), `${label} value`).toBeLessThan(1e-6);
      expect(summarize(lane(out, 3)).mean, `${label} gradient is non-trivial`).toBeGreaterThan(0.5);
    }
  });

  it('gnoise_3d_d matches finite differences in all three axes', async () => {
    const input = randomRows(20000, 32, -8, 8);
    const out = await gpuF32(
      `${include}
       fn f(x: vec4f) -> vec4f {
         let p = x.xyz;
         let h = 2e-3;
         let ex = vec3f(h, 0.0, 0.0);
         let ey = vec3f(0.0, h, 0.0);
         let ez = vec3f(0.0, 0.0, h);
         let fx = (gnoise_3d(p + ex) - gnoise_3d(p - ex)) / ((p + ex).x - (p - ex).x);
         let fy = (gnoise_3d(p + ey) - gnoise_3d(p - ey)) / ((p + ey).y - (p - ey).y);
         let fz = (gnoise_3d(p + ez) - gnoise_3d(p - ez)) / ((p + ez).z - (p - ez).z);
         let d = gnoise_3d_d(p);
         return vec4f(length(d.yzw - vec3f(fx, fy, fz)), abs(d.x - gnoise_3d(p)), length(d.yzw), 0.0);
       }`,
      input,
    );
    expect(maxAbs(lane(out, 0))).toBeLessThan(2e-3);
    expect(maxAbs(lane(out, 1))).toBeLessThan(1e-6);
    expect(summarize(lane(out, 2)).mean).toBeGreaterThan(0.5);
  });

  it('tile variants repeat exactly with the period', async () => {
    // lanes: p.x, p.y, period.x, period.y (periods 1..8)
    const random = makeRandom(41);
    const count = 20000;
    const input = new Float32Array(count * 4);
    for (let i = 0; i < count; i++) {
      input.set(
        [
          -10 + 20 * random(),
          -10 + 20 * random(),
          1 + Math.floor(random() * 8),
          1 + Math.floor(random() * 8),
        ],
        i * 4,
      );
    }
    for (const fn of ['vnoise_2d_tile', 'gnoise_2d_tile']) {
      const out = await gpuF32(
        `${include}
         fn f(x: vec4f) -> vec4f {
           let per = vec2i(x.zw);
           let pf = vec2f(per);
           let p = x.xy;
           let a = ${fn}(p, per);
           return vec4f(
             a - ${fn}(p + vec2f(pf.x, 0.0), per),
             a - ${fn}(p + vec2f(0.0, pf.y), per),
             a - ${fn}(p - 2.0 * pf, per),
             abs(a - ${fn}(p + vec2f(1.0, 0.0), per)));
         }`,
        input,
      );
      for (const l of [0, 1, 2]) expect(maxAbs(lane(out, l)), `${fn} lane ${l}`).toBeLessThan(5e-5);
    }
  });

  it('tile variants are real noise, not a constant, and equal the plain noise inside one period', async () => {
    const input = randomRows(20000, 42, 0, 1);
    const out = await gpuF32(
      `${include}
       fn f(x: vec4f) -> vec4f {
         let per = vec2i(6, 5);
         let p = x.xy * vec2f(4.0, 3.0); // stays clear of the seam cells: floor(p) + 1 < period
         let q = x.xy * vec2f(6.0, 5.0);
         return vec4f(
           gnoise_2d_tile(p, per) - gnoise_2d(p),
           vnoise_2d_tile(p, per) - vnoise_2d(p),
           gnoise_2d_tile(q, per),
           abs(gnoise_2d_tile(q, per) - gnoise_2d_tile(q + vec2f(1.0, 0.0), per)));
       }`,
      input,
    );
    expect(maxAbs(lane(out, 0))).toBe(0);
    expect(maxAbs(lane(out, 1))).toBe(0);
    expect(Math.sqrt(summarize(lane(out, 2)).variance)).toBeGreaterThan(0.15);
    // Shifting by one cell (not a full period) must change the value: it is not secretly period 1.
    expect(summarize(lane(out, 3)).mean).toBeGreaterThan(0.1);
  });

  it('gnoise_3d_tile repeats in x, y and z (a seamless time loop)', async () => {
    const random = makeRandom(43);
    const count = 20000;
    const input = new Float32Array(count * 4);
    for (let i = 0; i < count; i++)
      input.set([-8 + 16 * random(), -8 + 16 * random(), -8 + 16 * random(), 0], i * 4);
    const out = await gpuF32(
      `${include}
       fn f(x: vec4f) -> vec4f {
         let per = vec3i(3, 4, 5);
         let p = x.xyz;
         let a = gnoise_3d_tile(p, per);
         return vec4f(
           a - gnoise_3d_tile(p + vec3f(3.0, 0.0, 0.0), per),
           a - gnoise_3d_tile(p + vec3f(0.0, 4.0, 0.0), per),
           a - gnoise_3d_tile(p - vec3f(0.0, 0.0, 5.0), per),
           abs(a - gnoise_3d_tile(p + vec3f(0.0, 0.0, 1.0), per)));
       }`,
      input,
    );
    for (const l of [0, 1, 2]) expect(maxAbs(lane(out, l))).toBeLessThan(5e-5);
    expect(summarize(lane(out, 3)).mean).toBeGreaterThan(0.1);
  });
});

describe('noise: simplex (simplex.wgsl)', () => {
  const include = '#include "simplex.wgsl"\n';

  it('stays within [-1, 1] with zero mean and uses the range', async () => {
    const out = await gpuF32(
      `${include} fn f(x: vec4f) -> vec4f { return vec4f(snoise_2d(x.xy), snoise_3d(x.xyz), 0.0, 0.0); }`,
      randomRows(200000, 51, -60, 60),
    );
    for (const l of [0, 1]) {
      const s = summarize(lane(out, l));
      expect(s.min, `lane ${l} min`).toBeGreaterThanOrEqual(-1);
      expect(s.max, `lane ${l} max`).toBeLessThanOrEqual(1);
      expect(s.min).toBeLessThan(-0.6);
      expect(s.max).toBeGreaterThan(0.6);
      expect(Math.abs(s.mean)).toBeLessThan(0.02);
      expect(Math.sqrt(s.variance)).toBeGreaterThan(0.15);
    }
  });

  it('snoise_2d_d value equals snoise_2d and its gradient matches finite differences', async () => {
    const out = await gpuF32(
      gradientCheck2d(include, 'snoise_2d_d', 'snoise_2d'),
      randomRows(20000, 52, -8, 8),
    );
    expect(maxAbs(lane(out, 0))).toBeLessThan(1e-3);
    expect(maxAbs(lane(out, 1))).toBeLessThan(1e-3);
    expect(maxAbs(lane(out, 2))).toBeLessThan(1e-6);
    expect(summarize(lane(out, 3)).mean).toBeGreaterThan(0.5);
  });

  it('is continuous (no seams between simplex cells) and deterministic', async () => {
    const code = `${include}
      fn f(x: vec4f) -> vec4f {
        let e = 1e-4;
        let a = snoise_2d(x.xy + vec2f(e)) - snoise_2d(x.xy - vec2f(e));
        let b = snoise_3d(x.xyz + vec3f(e)) - snoise_3d(x.xyz - vec3f(e));
        return vec4f(a, b, snoise_2d(x.xy), snoise_3d(x.xyz));
      }`;
    const input = randomRows(100000, 53, -20, 20);
    const a = await gpuF32(code, input);
    const b = await gpuF32(code, input);
    expect(sameBits(a, b)).toBe(true);
    // Over a ~3e-4 step a continuous function with a gradient of up to ~8 moves by ~2e-3 at most;
    // a seam would jump by 0.01 or more.
    expect(maxAbs(lane(a, 0))).toBeLessThan(5e-3);
    expect(maxAbs(lane(a, 1))).toBeLessThan(5e-3);
  });
});

// --- FBM -----------------------------------------------------------------------------------------

describe('fbm.wgsl', () => {
  const include = '#include "fbm.wgsl"\n';
  const configs: [number, number, number][] = [
    [1, 2.0, 0.5],
    [4, 2.0, 0.5],
    [8, 2.0, 0.5],
    [6, 1.7, 0.7],
    [5, 2.3, 1.0],
    [16, 2.0, 0.9],
  ];

  it.each(configs)(
    'stays in range for octaves=%i lacunarity=%f gain=%f',
    async (octaves, lacunarity, gain) => {
      const args = `${octaves}, ${lacunarity.toFixed(2)}, ${gain.toFixed(2)}`;
      const out = await gpuF32(
        `${include}
       fn f(x: vec4f) -> vec4f {
         return vec4f(fbm_2d(x.xy, ${args}), fbm_3d(x.xyz, ${args}), ridged_2d(x.xy, ${args}), fbm_2d_d(x.xy, ${args}).x);
       }`,
        randomRows(60000, 61, -40, 40),
      );
      for (const l of [0, 1, 3]) {
        const s = summarize(lane(out, l));
        expect(s.min, `lane ${l}`).toBeGreaterThanOrEqual(-1);
        expect(s.max, `lane ${l}`).toBeLessThanOrEqual(1);
        expect(Math.abs(s.mean)).toBeLessThan(0.05);
        expect(Math.sqrt(s.variance)).toBeGreaterThan(0.05);
      }
      const ridged = summarize(lane(out, 2));
      expect(ridged.min).toBeGreaterThanOrEqual(0);
      expect(ridged.max).toBeLessThanOrEqual(1);
      expect(ridged.mean).toBeGreaterThan(0.3);
      expect(Math.sqrt(ridged.variance)).toBeGreaterThan(0.02);
    },
  );

  it('one octave is the bare noise; zero octaves is zero; the octave count is capped', async () => {
    const out = await gpuF32(
      `${include}
       fn f(x: vec4f) -> vec4f {
         return vec4f(
           fbm_2d(x.xy, 1, 2.0, 0.5) - gnoise_2d(x.xy),
           fbm_3d(x.xyz, 1, 2.0, 0.5) - gnoise_3d(x.xyz),
           fbm_2d(x.xy, 0, 2.0, 0.5),
           fbm_2d(x.xy, 1000, 2.0, 0.5) - fbm_2d(x.xy, 16, 2.0, 0.5));
       }`,
      randomRows(5000, 62, -20, 20),
    );
    for (const l of [0, 1, 2]) expect(maxAbs(lane(out, l)), `lane ${l}`).toBe(0);
    // Same loop, but a constant 16 lets the compiler unroll and contract it differently from 1000 -> 16.
    expect(maxAbs(lane(out, 3))).toBeLessThan(1e-6);
  });

  it('adds detail with every octave without growing beyond its range', async () => {
    const out = await gpuF32(
      `${include}
       fn f(x: vec4f) -> vec4f {
         return vec4f(fbm_2d(x.xy, 2, 2.0, 0.5), fbm_2d(x.xy, 3, 2.0, 0.5), fbm_2d(x.xy, 8, 2.0, 0.5), 0.0);
       }`,
      randomRows(40000, 63, -40, 40),
    );
    const a = lane(out, 0);
    const b = lane(out, 1);
    const c = lane(out, 2);
    // Each extra octave changes the result by at most its share of the normalised amplitude sum.
    expect(maxAbs(a.map((v, i) => v - (b[i] ?? 0)))).toBeGreaterThan(0.01);
    expect(maxAbs(b.map((v, i) => v - (c[i] ?? 0)))).toBeLessThan(0.6);
  });

  // Octave 0 is exact; later octaves live at coordinates ~lacunarity^i larger, where f32 spacing makes the
  // finite-difference reference itself noisy, hence the looser bound for 4 octaves. A wrong chain rule
  // (e.g. a transposed rotation) is off by O(1) already at 2 octaves.
  it.each([
    ['2, 2.0, 0.5', 2e-3],
    ['4, 2.1, 0.55', 8e-3],
  ] as const)(
    'fbm_2d_d (%s): value matches fbm_2d and the gradient matches finite differences',
    async (args, tolerance) => {
      const out = await gpuF32(
        `${include}
       fn f(x: vec4f) -> vec4f {
         let p = x.xy;
         let h = 1e-3;
         let px1 = p + vec2f(h, 0.0);
         let px0 = p - vec2f(h, 0.0);
         let py1 = p + vec2f(0.0, h);
         let py0 = p - vec2f(0.0, h);
         let fx = (fbm_2d(px1, ${args}) - fbm_2d(px0, ${args})) / (px1.x - px0.x);
         let fy = (fbm_2d(py1, ${args}) - fbm_2d(py0, ${args})) / (py1.y - py0.y);
         let d = fbm_2d_d(p, ${args});
         return vec4f(d.y - fx, d.z - fy, abs(d.x - fbm_2d(p, ${args})), length(d.yz));
       }`,
        randomRows(20000, 64, -2, 2),
      );
      expect(maxAbs(lane(out, 0))).toBeLessThan(tolerance);
      expect(maxAbs(lane(out, 1))).toBeLessThan(tolerance);
      expect(maxAbs(lane(out, 2))).toBeLessThan(1e-5);
      expect(summarize(lane(out, 3)).mean).toBeGreaterThan(0.3);
    },
  );

  it.each([
    ['2, 2.0, 0.5', 4e-3],
    ['4, 2.1, 0.55', 1.5e-2],
  ] as const)(
    'fbm_3d_d (%s): value matches fbm_3d and the gradient matches finite differences',
    async (args, tolerance) => {
      const out3 = await gpuF32(
        `${include}
       fn f(x: vec4f) -> vec4f {
         let p = x.xyz;
         let h = 1e-3;
         let ex = vec3f(h, 0.0, 0.0);
         let ey = vec3f(0.0, h, 0.0);
         let ez = vec3f(0.0, 0.0, h);
         let fx = (fbm_3d(p + ex, ${args}) - fbm_3d(p - ex, ${args})) / ((p + ex).x - (p - ex).x);
         let fy = (fbm_3d(p + ey, ${args}) - fbm_3d(p - ey, ${args})) / ((p + ey).y - (p - ey).y);
         let fz = (fbm_3d(p + ez, ${args}) - fbm_3d(p - ez, ${args})) / ((p + ez).z - (p - ez).z);
         let d = fbm_3d_d(p, ${args});
         return vec4f(length(d.yzw - vec3f(fx, fy, fz)), abs(d.x - fbm_3d(p, ${args})), length(d.yzw), 0.0);
       }`,
        randomRows(20000, 65, -2, 2),
      );
      expect(maxAbs(lane(out3, 0))).toBeLessThan(tolerance);
      expect(maxAbs(lane(out3, 1))).toBeLessThan(1e-5);
      expect(summarize(lane(out3, 2)).mean).toBeGreaterThan(0.3);
    },
  );

  it('fbm_2d_tile repeats with the period at every octave and is not trivially constant', async () => {
    const random = makeRandom(66);
    const count = 20000;
    const input = new Float32Array(count * 4);
    for (let i = 0; i < count; i++) {
      input.set(
        [
          -6 + 12 * random(),
          -6 + 12 * random(),
          1 + Math.floor(random() * 6),
          1 + Math.floor(random() * 6),
        ],
        i * 4,
      );
    }
    const out = await gpuF32(
      `${include}
       fn f(x: vec4f) -> vec4f {
         let per = vec2i(x.zw);
         let pf = vec2f(per);
         let a = fbm_2d_tile(x.xy, per, 5, 0.5);
         return vec4f(
           a - fbm_2d_tile(x.xy + vec2f(pf.x, 0.0), per, 5, 0.5),
           a - fbm_2d_tile(x.xy - vec2f(0.0, 2.0 * pf.y), per, 5, 0.5),
           a,
           abs(a - fbm_2d_tile(x.xy + vec2f(0.5, 0.5), per, 5, 0.5)));
       }`,
      input,
    );
    expect(maxAbs(lane(out, 0))).toBeLessThan(1e-4);
    expect(maxAbs(lane(out, 1))).toBeLessThan(1e-4);
    const s = summarize(lane(out, 2));
    expect(s.min).toBeGreaterThanOrEqual(-1);
    expect(s.max).toBeLessThanOrEqual(1);
    expect(Math.sqrt(s.variance)).toBeGreaterThan(0.05);
    expect(summarize(lane(out, 3)).mean).toBeGreaterThan(0.05);
  });

  it('tiles seamlessly as a texture: the wrap-around neighbour differs no more than an interior neighbour', async () => {
    // Sample a 64x64 texture covering exactly one period; compare wrap seam steps to interior steps.
    const size = 64;
    const input = new Float32Array(size * size * 4);
    for (let y = 0; y < size; y++)
      for (let x = 0; x < size; x++) input.set([x / size, y / size], (y * size + x) * 4);
    const out = await gpuF32(
      `${include}
       fn tex(uv: vec2f) -> f32 { return fbm_2d_tile(uv * 4.0, vec2i(4, 4), 4, 0.5); }
       fn f(x: vec4f) -> vec4f {
         let texel = 1.0 / 64.0;
         return vec4f(tex(x.xy), tex(fract(x.xy + vec2f(texel, 0.0))), tex(fract(x.xy + vec2f(0.0, texel))), 0.0);
       }`,
      input,
    );
    const here = lane(out, 0);
    const right = lane(out, 1);
    let seam = 0;
    let interior = 0;
    for (let y = 0; y < size; y++) {
      seam = Math.max(
        seam,
        Math.abs((here[y * size + size - 1] ?? 0) - (right[y * size + size - 1] ?? 0)),
      );
      for (let x = 0; x < size - 1; x++)
        interior = Math.max(
          interior,
          Math.abs((here[y * size + x] ?? 0) - (right[y * size + x] ?? 0)),
        );
    }
    expect(seam).toBeLessThanOrEqual(interior * 1.01);
  });

  it('domain_warp_2d: zero strength is the identity, displacement is bounded, animation moves it', async () => {
    const out = await gpuF32(
      `${include}
       fn f(x: vec4f) -> vec4f {
         let a = domain_warp_2d(x.xy, 0.0, 4, 2.0, 0.5) - x.xy;
         let b = domain_warp_2d(x.xy, 1.5, 4, 2.0, 0.5) - x.xy;
         let c = domain_warp_2d_anim(x.xy, 0.0, 1.5, 4, 2.0, 0.5);
         let d = domain_warp_2d_anim(x.xy, 3.0, 1.5, 4, 2.0, 0.5);
         return vec4f(length(a), max(abs(b.x), abs(b.y)), length(c - d), length(d - x.xy));
       }`,
      randomRows(20000, 67, -20, 20),
    );
    expect(maxAbs(lane(out, 0))).toBe(0);
    expect(maxAbs(lane(out, 1))).toBeLessThanOrEqual(1.5);
    expect(summarize(lane(out, 1)).mean).toBeGreaterThan(0.1);
    expect(summarize(lane(out, 2)).mean).toBeGreaterThan(0.1);
    expect(maxAbs(lane(out, 3))).toBeLessThanOrEqual(1.5 * Math.SQRT2 + 1e-4);
  });
});

// --- Voronoi -------------------------------------------------------------------------------------

interface VoronoiRef {
  f1: number;
  f2: number;
  id: number;
  cell: [number, number];
}

/**
 * Brute-force Voronoi in double precision over a 5x5 block (so F2 is exact), using the same integer
 * hash as the shader. `time` undefined means static points.
 */
function referenceVoronoi(
  px: number,
  py: number,
  jitter: number,
  time?: number,
  period?: [number, number],
): VoronoiRef {
  const baseX = Math.floor(px);
  const baseY = Math.floor(py);
  const sites: { dist: number; id: number; cell: [number, number] }[] = [];
  for (let dy = -2; dy <= 2; dy++) {
    for (let dx = -2; dx <= 2; dx++) {
      const cx = baseX + dx;
      const cy = baseY + dy;
      const wx = period ? ((cx % period[0]) + period[0]) % period[0] : cx;
      const wy = period ? ((cy % period[1]) + period[1]) % period[1] : cy;
      const [hx, hy, hz] = refPcg3(u32(wx), u32(wy), 0).map(unorm) as [number, number, number];
      const ox =
        time === undefined ? jitter * (hx - 0.5) : 0.4 * jitter * Math.sin(time + 2 * Math.PI * hx);
      const oy =
        time === undefined ? jitter * (hy - 0.5) : 0.4 * jitter * Math.sin(time + 2 * Math.PI * hy);
      const dist = Math.hypot(cx + 0.5 + ox - px, cy + 0.5 + oy - py);
      sites.push({ dist, id: hz, cell: [wx, wy] });
    }
  }
  sites.sort((a, b) => a.dist - b.dist);
  const [first, second] = sites as [(typeof sites)[number], (typeof sites)[number]];
  return { f1: first.dist, f2: second.dist, id: first.id, cell: first.cell };
}

/** Rows of (x, y, 0, 0) for `count` random points in [lo, hi). */
function randomPoints2d(count: number, seed: number, lo: number, hi: number): Float32Array {
  const rows = randomRows(count, seed, lo, hi);
  for (let i = 0; i < count; i++) rows.fill(0, i * 4 + 2, i * 4 + 4);
  return rows;
}

describe('voronoi.wgsl', () => {
  const include = '#include "voronoi.wgsl"\n';
  /** Packs a Voronoi2D result into (f1, f2, id, cell.x) and (cell.y, |site - p| - f1, site cell == cell, 0). */
  const probe = (call: string) => `${include}
    fn f(x: vec4f) -> vec4f {
      let p = x.xy;
      let v = ${call};
      return vec4f(v.f1, v.f2, v.id, f32(v.cell.x) + 4096.0 * f32(v.cell.y));
    }
    fn check(x: vec4f) -> vec4f {
      let p = x.xy;
      let v = ${call};
      let inside = all(vec2i(floor(v.site)) == v.cell);
      return vec4f(length(v.site - p) - v.f1, select(0.0, 1.0, inside), 0.0, 0.0);
    }`;
  const decodeCell = (packed: number): [number, number] => {
    const y = Math.round(packed / 4096);
    return [Math.round(packed - 4096 * y) + 0, y + 0]; // + 0 turns -0 into 0
  };

  it('f1 and id match a brute-force search for every jitter; f2 too when jitter is moderate', async () => {
    const count = 6000;
    const input = randomPoints2d(count, 71, -30, 30);
    for (const jitter of [0.0, 0.5, 1.0]) {
      const out = await gpuF32(probe(`voronoi_2d(p, ${jitter.toFixed(1)})`), input);
      const c = collector();
      let f2Mismatches = 0;
      for (let i = 0; i < count; i++) {
        const ref = referenceVoronoi(input[i * 4]!, input[i * 4 + 1]!, jitter);
        const label = `jitter ${jitter} row ${i}`;
        c.near(out[i * 4]!, ref.f1, 1e-4, `${label} f1`);
        if (ref.f2 - ref.f1 > 1e-3) {
          c.near(out[i * 4 + 2]!, ref.id, 1e-6, `${label} id`);
          c.check(decodeCell(out[i * 4 + 3]!).join() === ref.cell.join(), `${label} cell`);
        }
        if (Math.abs(out[i * 4 + 1]! - ref.f2) > 1e-4) f2Mismatches++;
      }
      expect(c.report()).toEqual([]);
      if (jitter <= 0.5) expect(f2Mismatches, `f2 mismatches at jitter ${jitter}`).toBe(0);
      else expect(f2Mismatches / count, 'f2 mismatch rate at jitter 1').toBeLessThan(0.002);
    }
  });

  it('f2 is exact wherever it matters for edge masks (near borders), even at jitter 1', async () => {
    const count = 20000;
    const input = randomPoints2d(count, 72, -30, 30);
    const out = await gpuF32(probe('voronoi_2d(p, 1.0)'), input);
    const c = collector();
    let nearBorder = 0;
    for (let i = 0; i < count; i++) {
      const ref = referenceVoronoi(input[i * 4]!, input[i * 4 + 1]!, 1);
      if (ref.f2 - ref.f1 < 0.25) {
        nearBorder++;
        c.near(out[i * 4 + 1]!, ref.f2, 1e-4, `row ${i} f2`);
      }
    }
    expect(nearBorder).toBeGreaterThan(count / 10); // a meaningful share of the samples is near a border
    expect(c.report()).toEqual([]);
  });

  it('satisfies its documented invariants: 0 <= f1 <= f2, id in [0, 1), site inside the owning cell', async () => {
    const input = randomPoints2d(50000, 73, -50, 50);
    for (const call of [
      'voronoi_2d(p, 1.0)',
      'voronoi_2d_anim(p, 1.0, 2.5)',
      'voronoi_2d_tile(p, 1.0, vec2i(5, 3))',
      'voronoi_2d_tile_anim(p, 0.7, 1.1, vec2i(4, 4))',
    ]) {
      const out = await gpuF32(probe(call), input);
      const f1 = lane(out, 0);
      const f2 = lane(out, 1);
      const ids = lane(out, 2);
      expect(Math.min(...f1), call).toBeGreaterThanOrEqual(0);
      expect(Math.max(...f1), call).toBeLessThan(1.5);
      expect(
        f1.every((v, i) => v <= (f2[i] ?? 0)),
        call,
      ).toBe(true);
      expect(Math.min(...ids)).toBeGreaterThanOrEqual(0);
      expect(Math.max(...ids)).toBeLessThan(1);
      expect(summarize(ids).mean).toBeCloseTo(0.5, 1);

      const checked = await gpuF32(
        probe(call).replace('fn f(', 'fn unused(').replace('fn check(', 'fn f('),
        input,
      );
      expect(maxAbs(lane(checked, 0)), `${call}: |site - p| == f1`).toBeLessThan(1e-4);
      if (!call.includes('tile'))
        expect(Math.min(...lane(checked, 1)), `${call}: site in its cell`).toBe(1);
    }
  });

  it('jitter 0 is a regular grid: seeds at cell centres', async () => {
    const out = await gpuF32(
      `${include}
       fn f(x: vec4f) -> vec4f {
         let v = voronoi_2d(x.xy, 0.0);
         return vec4f(v.f1, v.f2, length(v.site - (floor(x.xy) + 0.5)), 0.0);
       }`,
      rowsOf([
        [0.5, 0.5],
        [3.5, -2.5],
        [10.5, 10.5],
        [0.0, 0.0],
        [0.5, 0.0],
      ]),
    );
    expect([out[0], out[1], out[2]]).toEqual([0, 1, 0]); // at a seed: F1 = 0, F2 = distance to a neighbour = 1
    expect([out[4], out[5], out[6]]).toEqual([0, 1, 0]);
    expect([out[8], out[9], out[10]]).toEqual([0, 1, 0]);
    // On a corner of the grid all four seeds are equidistant: F1 == F2 == sqrt(0.5).
    expect(out[12]).toBeCloseTo(Math.SQRT1_2, 6);
    expect(out[13]).toBeCloseTo(Math.SQRT1_2, 6);
    // On an edge midpoint two seeds are equidistant: a border (F2 - F1 == 0).
    expect(out[16]).toBeCloseTo(0.5, 6);
    expect(out[17]).toBeCloseTo(0.5, 6);
  });

  it('is deterministic', async () => {
    const input = randomPoints2d(5000, 74, -10, 10);
    const code = probe('voronoi_2d(p, 0.9)');
    const a = await gpuF32(code, input);
    const b = await gpuF32(code, input);
    expect(sameBits(a, b)).toBe(true);
  });

  it('animated points move continuously and match the brute force at the same time', async () => {
    const count = 4000;
    const input = randomPoints2d(count, 75, -20, 20);
    const time = 1.3;
    const out = await gpuF32(probe(`voronoi_2d_anim(p, 1.0, ${time})`), input);
    const c = collector();
    for (let i = 0; i < count; i++) {
      const ref = referenceVoronoi(input[i * 4]!, input[i * 4 + 1]!, 1, time);
      c.near(out[i * 4]!, ref.f1, 1e-4, `row ${i} f1`);
    }
    expect(c.report()).toEqual([]);

    const stepped = await gpuF32(
      `${include}
       fn f(x: vec4f) -> vec4f {
         let a = voronoi_2d_anim(x.xy, 1.0, 1.0);
         let b = voronoi_2d_anim(x.xy, 1.0, 1.01);
         let c = voronoi_2d_anim(x.xy, 1.0, 2.5);
         return vec4f(abs(a.f1 - b.f1), abs(a.f2 - b.f2), abs(a.f1 - c.f1), 0.0);
       }`,
      input,
    );
    // Each seed moves at most 0.4 * sqrt(2) per radian, so distances change by at most that per unit time.
    expect(maxAbs(lane(stepped, 0))).toBeLessThan(0.0072);
    expect(maxAbs(lane(stepped, 1))).toBeLessThan(0.0072);
    expect(summarize(lane(stepped, 2)).mean).toBeGreaterThan(0.05); // it actually animates
  });

  it('tile forms repeat exactly with the period (distances, id and cell)', async () => {
    const random = makeRandom(76);
    const count = 6000;
    const input = new Float32Array(count * 4);
    for (let i = 0; i < count; i++) {
      input.set(
        [
          -12 + 24 * random(),
          -12 + 24 * random(),
          1 + Math.floor(random() * 7),
          1 + Math.floor(random() * 7),
        ],
        i * 4,
      );
    }
    for (const call of ['voronoi_2d_tile(P, 1.0, per)', 'voronoi_2d_tile_anim(P, 0.8, 0.9, per)']) {
      const run = (shift: string) =>
        gpuF32(
          `${include}
           fn f(x: vec4f) -> vec4f {
             let per = vec2i(x.zw);
             let P = x.xy + ${shift};
             let v = ${call};
             return vec4f(v.f1, v.f2, v.id, f32(v.cell.x) + 64.0 * f32(v.cell.y));
           }`,
          input,
        );
      // The shift is a whole number of periods per row, so it has to be expressed per row.
      const base = await run('vec2f(0.0)');
      const shifted = await run('vec2f(x.z, -2.0 * x.w)');
      const c = collector();
      for (let i = 0; i < count; i++) {
        c.near(shifted[i * 4]!, base[i * 4]!, 1e-3, `${call} f1 row ${i}`);
        c.near(shifted[i * 4 + 1]!, base[i * 4 + 1]!, 1e-3, `${call} f2 row ${i}`);
        const ambiguous = base[i * 4 + 1]! - base[i * 4]! < 1e-3;
        if (!ambiguous) {
          c.check(shifted[i * 4 + 2] === base[i * 4 + 2], `${call} id row ${i}`);
          c.check(shifted[i * 4 + 3] === base[i * 4 + 3], `${call} cell row ${i}`);
        }
      }
      expect(c.report()).toEqual([]);
    }
  });

  it('tile forms match the brute-force search with wrapped cell hashes', async () => {
    const count = 4000;
    const period: [number, number] = [5, 3];
    const input = randomPoints2d(count, 77, -15, 15);
    const out = await gpuF32(probe('voronoi_2d_tile(p, 0.6, vec2i(5, 3))'), input);
    const c = collector();
    for (let i = 0; i < count; i++) {
      const ref = referenceVoronoi(input[i * 4]!, input[i * 4 + 1]!, 0.6, undefined, period);
      c.near(out[i * 4]!, ref.f1, 1e-4, `row ${i} f1`);
      c.near(out[i * 4 + 1]!, ref.f2, 1e-4, `row ${i} f2`);
      if (ref.f2 - ref.f1 > 1e-3) {
        c.near(out[i * 4 + 2]!, ref.id, 1e-6, `row ${i} id`);
        c.check(decodeCell(out[i * 4 + 3]!).join() === ref.cell.join(), `row ${i} cell`);
      }
    }
    expect(c.report()).toEqual([]);
  });

  it('a tile form equals the plain form deep inside one period', async () => {
    const out = await gpuF32(
      `${include}
       fn f(x: vec4f) -> vec4f {
         let p = x.xy * 6.0 + 1.0; // cells 1..6 -> all 3x3 neighbours inside [0, 8)
         let a = voronoi_2d(p, 0.9);
         let b = voronoi_2d_tile(p, 0.9, vec2i(8, 8));
         return vec4f(a.f1 - b.f1, a.f2 - b.f2, a.id - b.id, length(a.site - b.site));
       }`,
      randomRows(10000, 78, 0, 0.999),
    );
    // Same arithmetic, but the compiler may contract the tiled and plain paths differently (~1e-8).
    for (const l of [0, 1, 2, 3]) expect(maxAbs(lane(out, l))).toBeLessThan(1e-6);
  });
});

// --- Curl noise ----------------------------------------------------------------------------------

describe('curl.wgsl', () => {
  const include = '#include "curl.wgsl"\n';

  it('curl_2d is divergence-free, while an ordinary gradient field is not (control)', async () => {
    const args = '3, 2.0, 0.5';
    const out = await gpuF32(
      `${include}
       fn f(x: vec4f) -> vec4f {
         let p = x.xy;
         let h = 2e-3;
         let ex = vec2f(h, 0.0);
         let ey = vec2f(0.0, h);
         let dx = (p + ex).x - (p - ex).x;
         let dy = (p + ey).y - (p - ey).y;
         // Jacobian of the curl field
         let vx_x = (curl_2d(p + ex, ${args}).x - curl_2d(p - ex, ${args}).x) / dx;
         let vx_y = (curl_2d(p + ey, ${args}).x - curl_2d(p - ey, ${args}).x) / dy;
         let vy_x = (curl_2d(p + ex, ${args}).y - curl_2d(p - ex, ${args}).y) / dx;
         let vy_y = (curl_2d(p + ey, ${args}).y - curl_2d(p - ey, ${args}).y) / dy;
         // Control: the plain gradient of the same potential
         let gx_x = (fbm_2d_d(p + ex, ${args}).y - fbm_2d_d(p - ex, ${args}).y) / dx;
         let gy_y = (fbm_2d_d(p + ey, ${args}).z - fbm_2d_d(p - ey, ${args}).z) / dy;
         let scale = abs(vx_x) + abs(vx_y) + abs(vy_x) + abs(vy_y);
         return vec4f(vx_x + vy_y, scale, gx_x + gy_y, 0.0);
       }`,
      randomRows(20000, 81, -4, 4),
    );
    const divergence = lane(out, 0);
    const scale = summarize(lane(out, 1)).mean;
    const control = lane(out, 2);
    expect(scale).toBeGreaterThan(1); // the field has real structure
    // Max |div| relative to the typical size of the Jacobian entries: finite-difference noise only.
    expect(maxAbs(divergence) / scale).toBeLessThan(0.02);
    // The control has divergence of the same order as the Jacobian: the test can tell the difference.
    expect(Math.sqrt(summarize(control).variance) / scale).toBeGreaterThan(0.2);
    expect(Math.sqrt(summarize(control).variance)).toBeGreaterThan(
      20 * Math.sqrt(summarize(divergence).variance),
    );
  });

  it('curl_2d_anim is divergence-free at several times and changes over time', async () => {
    const args = '3, 2.0, 0.5';
    const out = await gpuF32(
      `${include}
       fn f(x: vec4f) -> vec4f {
         let p = x.xy;
         let t = x.z * 4.0;
         let h = 2e-3;
         let ex = vec2f(h, 0.0);
         let ey = vec2f(0.0, h);
         let dx = (p + ex).x - (p - ex).x;
         let dy = (p + ey).y - (p - ey).y;
         let vx_x = (curl_2d_anim(p + ex, t, ${args}).x - curl_2d_anim(p - ex, t, ${args}).x) / dx;
         let vy_y = (curl_2d_anim(p + ey, t, ${args}).y - curl_2d_anim(p - ey, t, ${args}).y) / dy;
         let scale = abs(vx_x) + abs(vy_y);
         let change = length(curl_2d_anim(p, t, ${args}) - curl_2d_anim(p, t + 1.0, ${args}));
         return vec4f(vx_x + vy_y, scale, change, length(curl_2d_anim(p, t, ${args})));
       }`,
      randomRows(20000, 82, -4, 4),
    );
    const scale = summarize(lane(out, 1)).mean;
    expect(maxAbs(lane(out, 0)) / scale).toBeLessThan(0.04);
    expect(summarize(lane(out, 2)).mean).toBeGreaterThan(0.3);
    expect(summarize(lane(out, 3)).mean).toBeGreaterThan(0.3);
  });

  it('curl_2d is the potential gradient rotated by 90 degrees: (d psi/dy, -d psi/dx)', async () => {
    const out = await gpuF32(
      `${include}
       fn f(x: vec4f) -> vec4f {
         let p = x.xy;
         let h = 1e-3;
         let ex = vec2f(h, 0.0);
         let ey = vec2f(0.0, h);
         let psi_x = (fbm_2d(p + ex, 2, 2.0, 0.5) - fbm_2d(p - ex, 2, 2.0, 0.5)) / ((p + ex).x - (p - ex).x);
         let psi_y = (fbm_2d(p + ey, 2, 2.0, 0.5) - fbm_2d(p - ey, 2, 2.0, 0.5)) / ((p + ey).y - (p - ey).y);
         let v = curl_2d(p, 2, 2.0, 0.5);
         return vec4f(v.x - psi_y, v.y + psi_x, length(v), 0.0);
       }`,
      randomRows(10000, 83, -3, 3),
    );
    expect(maxAbs(lane(out, 0))).toBeLessThan(3e-3);
    expect(maxAbs(lane(out, 1))).toBeLessThan(3e-3);
    expect(summarize(lane(out, 2)).mean).toBeGreaterThan(0.3);
  });

  it('curl_3d is divergence-free, non-trivial and uses all three components', async () => {
    const out = await gpuF32(
      `${include}
       fn f(x: vec4f) -> vec4f {
         let p = x.xyz;
         let h = 2e-3;
         let ex = vec3f(h, 0.0, 0.0);
         let ey = vec3f(0.0, h, 0.0);
         let ez = vec3f(0.0, 0.0, h);
         let vx_x = (curl_3d(p + ex).x - curl_3d(p - ex).x) / ((p + ex).x - (p - ex).x);
         let vy_y = (curl_3d(p + ey).y - curl_3d(p - ey).y) / ((p + ey).y - (p - ey).y);
         let vz_z = (curl_3d(p + ez).z - curl_3d(p - ez).z) / ((p + ez).z - (p - ez).z);
         let v = curl_3d(p);
         let control = (gnoise_3d_d(p + ex).y - gnoise_3d_d(p - ex).y) / ((p + ex).x - (p - ex).x)
                     + (gnoise_3d_d(p + ey).z - gnoise_3d_d(p - ey).z) / ((p + ey).y - (p - ey).y)
                     + (gnoise_3d_d(p + ez).w - gnoise_3d_d(p - ez).w) / ((p + ez).z - (p - ez).z);
         return vec4f(vx_x + vy_y + vz_z, abs(vx_x) + abs(vy_y) + abs(vz_z), control, length(v));
       }`,
      randomRows(20000, 84, -4, 4),
    );
    const scale = summarize(lane(out, 1)).mean;
    expect(scale).toBeGreaterThan(0.5);
    expect(maxAbs(lane(out, 0)) / scale).toBeLessThan(0.01);
    expect(Math.sqrt(summarize(lane(out, 2)).variance)).toBeGreaterThan(
      10 * Math.sqrt(summarize(lane(out, 0)).variance),
    );
    expect(summarize(lane(out, 3)).mean).toBeGreaterThan(0.3);
  });

  it('curl_3d components are decorrelated (the field is not a 1D swirl)', async () => {
    const out = await gpuF32(
      `${include} fn f(x: vec4f) -> vec4f { return vec4f(curl_3d(x.xyz), 0.0); }`,
      randomRows(20000, 85, -20, 20),
    );
    expect(Math.abs(correlation(lane(out, 0), lane(out, 1)))).toBeLessThan(0.1);
    expect(Math.abs(correlation(lane(out, 1), lane(out, 2)))).toBeLessThan(0.1);
    expect(Math.abs(correlation(lane(out, 0), lane(out, 2)))).toBeLessThan(0.1);
  });
});

// --- SDF -----------------------------------------------------------------------------------------

describe('sdf.wgsl', () => {
  const include = '#include "sdf.wgsl"\n';

  it('2D primitives have exact known values (inside negative, surface 0, outside positive)', async () => {
    const cases: [string, number][] = [
      ['sd_circle(vec2f(0.0), 2.0)', -2],
      ['sd_circle(vec2f(2.0, 0.0), 2.0)', 0],
      ['sd_circle(vec2f(3.0, 4.0), 2.0)', 3],
      ['sd_circle(vec2f(1.0, 0.0), 2.0)', -1],
      ['sd_box_2d(vec2f(0.0), vec2f(1.0, 2.0))', -1],
      ['sd_box_2d(vec2f(1.0, 0.0), vec2f(1.0, 2.0))', 0],
      ['sd_box_2d(vec2f(3.0, 0.0), vec2f(1.0, 2.0))', 2],
      ['sd_box_2d(vec2f(4.0, 6.0), vec2f(1.0, 2.0))', 5],
      ['sd_box_2d(vec2f(0.5, 0.5), vec2f(1.0, 2.0))', -0.5],
      ['sd_box_2d(vec2f(-4.0, -6.0), vec2f(1.0, 2.0))', 5],
      ['sd_rounded_box_2d(vec2f(0.0), vec2f(2.0, 1.0), 0.5)', -1],
      ['sd_rounded_box_2d(vec2f(2.0, 0.0), vec2f(2.0, 1.0), 0.5)', 0],
      ['sd_rounded_box_2d(vec2f(2.5, 0.0), vec2f(2.0, 1.0), 0.5)', 0.5],
      ['sd_rounded_box_2d(vec2f(2.0, 1.0), vec2f(2.0, 1.0), 0.5)', Math.SQRT1_2 - 0.5], // the cut-off corner
      ['sd_rounded_box_2d(vec2f(3.0, 2.0), vec2f(2.0, 1.0), 0.0)', Math.SQRT2],
      ['sd_rounded_box_2d(vec2f(0.0), vec2f(2.0, 1.0), 10.0)', -1], // radius clamps to the short half-size
      ['sd_segment_2d(vec2f(2.0, 3.0), vec2f(0.0), vec2f(4.0, 0.0))', 3],
      ['sd_segment_2d(vec2f(-3.0, 4.0), vec2f(0.0), vec2f(4.0, 0.0))', 5],
      ['sd_segment_2d(vec2f(7.0, 4.0), vec2f(0.0), vec2f(4.0, 0.0))', 5],
      ['sd_segment_2d(vec2f(1.0, 0.0), vec2f(0.0), vec2f(4.0, 0.0))', 0],
      ['sd_segment_2d(vec2f(3.0, 4.0), vec2f(0.0), vec2f(0.0))', 5], // degenerate segment is a point
      ['sd_capsule_2d(vec2f(2.0, 3.0), vec2f(0.0), vec2f(4.0, 0.0), 1.0)', 2],
      ['sd_capsule_2d(vec2f(2.0, 0.0), vec2f(0.0), vec2f(4.0, 0.0), 1.0)', -1],
    ];
    const values = await scalars(
      include,
      cases.map(([e]) => e),
    );
    cases.forEach(([expression, expected], i) =>
      expect(values[i], expression).toBeCloseTo(expected, 5),
    );
  });

  it('3D primitives have exact known values', async () => {
    const cases: [string, number][] = [
      ['sd_sphere(vec3f(0.0), 2.0)', -2],
      ['sd_sphere(vec3f(3.0, 4.0, 0.0), 2.0)', 3],
      ['sd_sphere(vec3f(0.0, 0.0, 2.0), 2.0)', 0],
      ['sd_box_3d(vec3f(0.0), vec3f(1.0, 2.0, 3.0))', -1],
      ['sd_box_3d(vec3f(4.0, 0.0, 0.0), vec3f(1.0, 2.0, 3.0))', 3],
      ['sd_box_3d(vec3f(2.0, 3.0, 4.0), vec3f(1.0, 2.0, 3.0))', Math.sqrt(3)],
      ['sd_box_3d(vec3f(1.0, 2.0, 0.0), vec3f(1.0, 2.0, 3.0))', 0],
      ['sd_rounded_box_3d(vec3f(0.0), vec3f(1.0, 2.0, 3.0), 0.5)', -1],
      ['sd_rounded_box_3d(vec3f(1.0, 0.0, 0.0), vec3f(1.0, 2.0, 3.0), 0.5)', 0],
      ['sd_rounded_box_3d(vec3f(1.0, 2.0, 3.0), vec3f(1.0, 2.0, 3.0), 0.5)', Math.sqrt(0.75) - 0.5],
      ['sd_capsule_3d(vec3f(3.0, 2.0, 0.0), vec3f(0.0), vec3f(0.0, 4.0, 0.0), 1.0)', 2],
      ['sd_capsule_3d(vec3f(0.0, 6.0, 0.0), vec3f(0.0), vec3f(0.0, 4.0, 0.0), 1.0)', 1],
      ['sd_capsule_3d(vec3f(0.0, 2.0, 0.0), vec3f(0.0), vec3f(0.0, 4.0, 0.0), 1.0)', -1],
      ['sd_capsule_3d(vec3f(3.0, 4.0, 0.0), vec3f(0.0), vec3f(0.0), 1.0)', 4], // degenerate capsule is a sphere
      ['sd_torus(vec3f(3.0, 0.0, 0.0), 3.0, 1.0)', -1],
      ['sd_torus(vec3f(4.0, 0.0, 0.0), 3.0, 1.0)', 0],
      ['sd_torus(vec3f(0.0), 3.0, 1.0)', 2],
      ['sd_torus(vec3f(3.0, 1.0, 0.0), 3.0, 1.0)', 0],
      ['sd_torus(vec3f(0.0, 0.0, 5.0), 3.0, 1.0)', 1],
      ['sd_torus(vec3f(0.0, 4.0, 0.0), 3.0, 1.0)', Math.hypot(3, 4) - 1],
    ];
    const values = await scalars(
      include,
      cases.map(([e]) => e),
    );
    cases.forEach(([expression, expected], i) =>
      expect(values[i], expression).toBeCloseTo(expected, 5),
    );
  });

  it('combinators and coverage', async () => {
    const cases: [string, number][] = [
      ['sd_union(1.0, -2.0)', -2],
      ['sd_intersect(1.0, -2.0)', 1],
      ['sd_subtract(-1.0, -2.0)', 2], // a minus b: inside a (-1), inside b (-2) -> outside the result
      ['sd_subtract(-1.0, 3.0)', -1], // far from b: unchanged
      ['sd_round(0.0, 0.25)', -0.25],
      ['sd_onion(-1.0, 0.25)', 0.75],
      ['sd_onion(1.0, 0.25)', 0.75],
      ['sd_onion(0.25, 0.25)', 0],
      ['sd_coverage(0.0, 0.1)', 0.5],
      ['sd_coverage(-1.0, 0.1)', 1],
      ['sd_coverage(1.0, 0.1)', 0],
      ['sd_coverage(-0.025, 0.1)', 0.75],
    ];
    const values = await scalars(
      include,
      cases.map(([e]) => e),
    );
    cases.forEach(([expression, expected], i) =>
      expect(values[i], expression).toBeCloseTo(expected, 6),
    );
  });

  it('every primitive is 1-Lipschitz (a true distance bound) and not trivially flat', async () => {
    const out = await gpuF32(
      `#include "hash.wgsl"
       ${include}
       fn lip2(p: vec2f, q: vec2f) -> f32 {
         let d = length(p - q);
         let h = vec2f(1.0, 2.0);
         let a = vec2f(-1.5, 0.5);
         let b = vec2f(2.0, 1.0);
         var worst = abs(sd_circle(p, 1.5) - sd_circle(q, 1.5));
         worst = max(worst, abs(sd_box_2d(p, h) - sd_box_2d(q, h)));
         worst = max(worst, abs(sd_rounded_box_2d(p, h, 0.7) - sd_rounded_box_2d(q, h, 0.7)));
         worst = max(worst, abs(sd_segment_2d(p, a, b) - sd_segment_2d(q, a, b)));
         worst = max(worst, abs(sd_capsule_2d(p, a, b, 0.4) - sd_capsule_2d(q, a, b, 0.4)));
         return worst / d;
       }
       fn lip3(p: vec3f, q: vec3f) -> f32 {
         let d = length(p - q);
         let h = vec3f(1.0, 2.0, 0.5);
         let a = vec3f(-1.5, 0.5, 0.0);
         let b = vec3f(2.0, 1.0, 1.0);
         var worst = abs(sd_sphere(p, 1.5) - sd_sphere(q, 1.5));
         worst = max(worst, abs(sd_box_3d(p, h) - sd_box_3d(q, h)));
         worst = max(worst, abs(sd_rounded_box_3d(p, h, 0.3) - sd_rounded_box_3d(q, h, 0.3)));
         worst = max(worst, abs(sd_capsule_3d(p, a, b, 0.4) - sd_capsule_3d(q, a, b, 0.4)));
         worst = max(worst, abs(sd_torus(p, 2.0, 0.5) - sd_torus(q, 2.0, 0.5)));
         return worst / d;
       }
       fn f(x: vec4f) -> vec4f {
         let p3 = x.xyz;
         let q3 = p3 + (hash_3to3(p3) - 0.5) * 0.2;
         let p2 = x.xy;
         let q2 = p2 + (hash_2to2(p2) - 0.5) * 0.2;
         return vec4f(lip2(p2, q2), lip3(p3, q3), sd_box_3d(p3, vec3f(1.0)), sd_box_2d(p2, vec2f(1.0)));
       }`,
      randomRows(100000, 91, -4, 4),
    );
    for (const l of [0, 1]) {
      const ratios = lane(out, l);
      expect(Math.max(...ratios), `lane ${l}`).toBeLessThanOrEqual(1 + 1e-3);
      expect(Math.max(...ratios), `lane ${l} reaches the bound`).toBeGreaterThan(0.95);
    }
    // Both signs occur, so the sign convention is exercised.
    expect(Math.min(...lane(out, 2))).toBeLessThan(0);
    expect(Math.max(...lane(out, 2))).toBeGreaterThan(0);
  });

  describe('smooth min / max', () => {
    // rows: a, b in [-2, 2], k in [0.05, 1.5]
    const random = makeRandom(92);
    const count = 50000;
    const rows = new Float32Array(count * 4);
    for (let i = 0; i < count; i++)
      rows.set([-2 + 4 * random(), -2 + 4 * random(), 0.05 + 1.45 * random(), 0], i * 4);

    const kinds = [
      { name: 'poly', deviation: (k: number) => k / 4 },
      { name: 'cubic', deviation: (k: number) => k / 6 },
      { name: 'exp', deviation: (k: number) => k * Math.LN2 },
    ];

    it.each(kinds)(
      'smin_$name: below min, bounded deviation, symmetric, consistent with its blend form',
      async ({ name, deviation }) => {
        const out = await gpuF32(
          `${include}
         fn f(x: vec4f) -> vec4f {
           let a = x.x;
           let b = x.y;
           let k = x.z;
           let s = smin_${name}(a, b, k);
           let sb = smin_${name}_blend(a, b, k);
           return vec4f(min(a, b) - s, s - smin_${name}(b, a, k), sb.x - s, sb.y);
         }`,
          rows,
        );
        const gaps = lane(out, 0); // min - smin, never negative
        const excess = gaps.map((gap, i) => gap - deviation(rows[i * 4 + 2] ?? 0));
        expect(Math.min(...gaps)).toBeGreaterThanOrEqual(-1e-6);
        expect(Math.max(...excess)).toBeLessThan(1e-5); // never deviates more than the documented maximum
        expect(Math.max(...gaps)).toBeGreaterThan(0.05); // and does blend noticeably somewhere
        expect(maxAbs(lane(out, 1))).toBeLessThan(1e-6); // symmetric
        expect(maxAbs(lane(out, 2))).toBeLessThan(1e-6); // blend form returns the same value
        expect(Math.min(...lane(out, 3))).toBeGreaterThanOrEqual(0);
        expect(Math.max(...lane(out, 3))).toBeLessThanOrEqual(1);
      },
    );

    it.each(kinds)(
      'smin_$name: deviation at a == b is exactly the documented maximum, and k -> 0 gives min',
      async ({ name, deviation }) => {
        const ks = [0.1, 0.5, 1.0, 2.0];
        const expressions = [
          ...ks.map((k) => `smin_${name}(0.3, 0.3, ${k})`),
          `smin_${name}(0.3, 0.9, 1e-5)`,
          `smin_${name}(-1.0, 0.25, 1e-5)`,
          `smin_${name}(2.0, -2.0, 1e-9)`,
        ];
        const values = await scalars(include, expressions);
        ks.forEach((k, i) => expect(values[i], `k=${k}`).toBeCloseTo(0.3 - deviation(k), 5));
        expect(values[4]).toBeCloseTo(0.3, 4);
        expect(values[5]).toBeCloseTo(-1, 4);
        expect(values[6]).toBeCloseTo(-2, 5);
      },
    );

    it.each(kinds)(
      'smin_$name: the blend factor is the derivative with respect to b',
      async ({ name }) => {
        const out = await gpuF32(
          `${include}
         fn f(x: vec4f) -> vec4f {
           let h = 1e-3;
           let numeric = (smin_${name}(x.x, x.y + h, x.z) - smin_${name}(x.x, x.y - h, x.z)) / (2.0 * h);
           let w = smin_${name}_blend(x.x, x.y, x.z).y;
           let w_swapped = smin_${name}_blend(x.y, x.x, x.z).y;
           return vec4f(w - numeric, w + w_swapped - 1.0, w, 0.0);
         }`,
          rows,
        );
        expect(maxAbs(lane(out, 0))).toBeLessThan(3e-3);
        expect(maxAbs(lane(out, 1))).toBeLessThan(1e-6);
        const weights = summarize(lane(out, 2));
        expect(weights.min).toBeLessThan(0.05);
        expect(weights.max).toBeGreaterThan(0.95);
      },
    );

    it('poly and cubic are local; far apart the blend goes to the nearer input; w(a == b) is 1/2', async () => {
      const values = await scalars(include, [
        'smin_poly(0.0, 1.0, 1.0)',
        'smin_poly(0.0, 3.0, 1.0)',
        'smin_cubic(0.0, 1.0, 1.0)',
        'smin_cubic(-2.0, 3.0, 1.0)',
        'smin_poly_blend(0.0, 5.0, 1.0).y',
        'smin_poly_blend(5.0, 0.0, 1.0).y',
        'smin_poly_blend(1.0, 1.0, 1.0).y',
        'smin_cubic_blend(1.0, 1.0, 0.5).y',
        'smin_exp_blend(1.0, 1.0, 0.5).y',
        'smin_exp_blend(0.0, 12.0, 0.5).y',
      ]);
      expect(values.slice(0, 4)).toEqual([0, 0, 0, -2]); // exactly min once |a - b| >= k
      expect(values[4]).toBe(0);
      expect(values[5]).toBe(1);
      for (const i of [6, 7, 8]) expect(values[i]).toBeCloseTo(0.5, 6);
      expect(values[9]).toBeLessThan(1e-9);
    });

    it('smin_exp is associative, so merging many blobs does not depend on order', async () => {
      const out = await gpuF32(
        `${include}
         fn f(x: vec4f) -> vec4f {
           let k = 0.4;
           let left = smin_exp(smin_exp(x.x, x.y, k), x.z, k);
           let right = smin_exp(x.x, smin_exp(x.y, x.z, k), k);
           return vec4f(left - right, 0.0, 0.0, 0.0);
         }`,
        randomRows(20000, 93, -3, 3),
      );
      expect(maxAbs(lane(out, 0))).toBeLessThan(2e-6);
    });

    it.each(kinds)(
      'smax_$name is the mirror of smin: never below max, bounded deviation',
      async ({ name, deviation }) => {
        const out = await gpuF32(
          `${include}
         fn f(x: vec4f) -> vec4f {
           let s = smax_${name}(x.x, x.y, x.z);
           return vec4f(s - max(x.x, x.y), s + smin_${name}(-x.x, -x.y, x.z), s, 0.0);
         }`,
          rows,
        );
        const overshoot = lane(out, 0); // smax - max, never negative, at most the deviation
        expect(Math.min(...overshoot)).toBeGreaterThanOrEqual(-1e-6);
        expect(
          Math.max(...overshoot.map((v, i) => v - deviation(rows[i * 4 + 2] ?? 0))),
        ).toBeLessThan(1e-5);
        expect(Math.max(...overshoot)).toBeGreaterThan(0.05);
        expect(maxAbs(lane(out, 1))).toBeLessThan(1e-6);
      },
    );

    it('smooth union of two circles bridges the gap that a plain union leaves', async () => {
      const left = 'sd_circle(vec2f(0.0) - vec2f(-1.1, 0.0), 1.0)';
      const right = 'sd_circle(vec2f(0.0) - vec2f(1.1, 0.0), 1.0)';
      const values = await scalars(include, [
        `sd_union(${left}, ${right})`,
        `smin_poly(${left}, ${right}, 0.6)`,
        `smin_cubic(${left}, ${right}, 0.6)`,
        `smin_exp(${left}, ${right}, 0.1)`,
      ]);
      expect(values[0]).toBeCloseTo(0.1, 5); // plain union: 0.1 outside both circles, the shapes do not touch
      expect(values[1]).toBeCloseTo(0.1 - 0.15, 5); // poly: pulled in by k/4
      expect(values[2]).toBeCloseTo(0.1 - 0.1, 5); // cubic: pulled in by k/6
      expect(values[3]).toBeCloseTo(0.1 - 0.1 * Math.LN2, 5); // exp: pulled in by k ln 2
    });
  });
});

// --- Colour --------------------------------------------------------------------------------------

/** The exact sRGB transfer function in double precision. */
const refLinearToSrgb = (x: number): number =>
  x <= 0.0031308 ? 12.92 * x : 1.055 * x ** (1 / 2.4) - 0.055;
const refSrgbToLinear = (x: number): number =>
  x <= 0.04045 ? x / 12.92 : ((x + 0.055) / 1.055) ** 2.4;

describe('color.wgsl', () => {
  const include = '#include "color.wgsl"\n';

  /** x in [0, 1] sampled finely, with extra density near the linear segment's breakpoints. */
  const ramp = (() => {
    const n = 8192;
    const rows = new Float32Array(n * 4);
    for (let i = 0; i < n; i++) rows[i * 4] = i / (n - 1);
    return rows;
  })();

  it('linear_to_srgb / srgb_to_linear match the exact piecewise curve', async () => {
    const out = await gpuF32(
      `${include}
       fn f(x: vec4f) -> vec4f {
         let c = vec3f(x.x, x.x * 0.5, x.x * x.x);
         let a = linear_to_srgb(c);
         let b = srgb_to_linear(c);
         return vec4f(a.x, b.x, linear_to_srgb_f(x.x), srgb_to_linear_f(x.x)) + vec4f(0.0, 0.0, 0.0, 0.0) * (a.y + a.z + b.y + b.z);
       }`,
      ramp,
    );
    const xs = lane(ramp, 0);
    const check = (l: number, reference: (v: number) => number, label: string) => {
      const errors = lane(out, l).map((v, i) => Math.abs(v - reference(xs[i] ?? 0)));
      expect(Math.max(...errors), label).toBeLessThan(2e-6);
    };
    check(0, refLinearToSrgb, 'linear_to_srgb');
    check(1, refSrgbToLinear, 'srgb_to_linear');
    check(2, refLinearToSrgb, 'linear_to_srgb_f');
    check(3, refSrgbToLinear, 'srgb_to_linear_f');
  });

  it('known values, endpoints and the two breakpoints', async () => {
    const values = await scalars(include, [
      'linear_to_srgb_f(0.0)',
      'linear_to_srgb_f(1.0)',
      'srgb_to_linear_f(0.0)',
      'srgb_to_linear_f(1.0)',
      'linear_to_srgb_f(0.5)',
      'srgb_to_linear_f(0.5)',
      'linear_to_srgb_f(0.18)', // mid grey
      'linear_to_srgb_f(0.0031308)', // top of the linear segment: 12.92 * 0.0031308 = 0.04045
      'srgb_to_linear_f(0.04045)',
      'linear_to_srgb_f(0.0031308 + 1e-6) - linear_to_srgb_f(0.0031308 - 1e-6)', // continuous across the break
      'srgb_to_linear_f(0.04045 + 1e-5) - srgb_to_linear_f(0.04045 - 1e-5)',
      'linear_to_srgb_f(-0.5)', // negatives clamp to 0
      'srgb_to_linear_f(-0.5)',
      'linear_to_srgb_f(4.0)', // HDR passes through above 1
    ]);
    expect(values[0]).toBe(0);
    expect(values[1]).toBeCloseTo(1, 6);
    expect(values[2]).toBe(0);
    expect(values[3]).toBeCloseTo(1, 6);
    expect(values[4]).toBeCloseTo(0.7353569830524495, 6);
    expect(values[5]).toBeCloseTo(0.21404114048223255, 6);
    expect(values[6]).toBeCloseTo(0.4613561295, 6);
    expect(values[7]).toBeCloseTo(0.04045, 5);
    expect(values[8]).toBeCloseTo(0.0031308, 6);
    expect(Math.abs(values[9]!)).toBeLessThan(5e-5); // slope 12.92 over a 2e-6 step
    expect(Math.abs(values[10]!)).toBeLessThan(3e-6); // slope ~0.075 over a 2e-5 step
    expect(values[11]).toBe(0);
    expect(values[12]).toBe(0);
    expect(values[13]).toBeCloseTo(refLinearToSrgb(4), 5);
  });

  it('sRGB round-trips both ways', async () => {
    const out = await gpuF32(
      `${include}
       fn f(x: vec4f) -> vec4f {
         let c = vec3f(x.x, 1.0 - x.x, x.x * x.x);
         let there = srgb_to_linear(linear_to_srgb(c));
         let back = linear_to_srgb(srgb_to_linear(c));
         return vec4f(length(there - c), length(back - c), 0.0, 0.0);
       }`,
      ramp,
    );
    expect(maxAbs(lane(out, 0))).toBeLessThan(5e-6);
    expect(maxAbs(lane(out, 1))).toBeLessThan(5e-6);
  });

  it('luminance uses Rec.709 weights that sum to 1', async () => {
    const values = await scalars(include, [
      'luminance(vec3f(1.0))',
      'luminance(vec3f(1.0, 0.0, 0.0))',
      'luminance(vec3f(0.0, 1.0, 0.0))',
      'luminance(vec3f(0.0, 0.0, 1.0))',
      'luminance(vec3f(0.0))',
      'luminance(vec3f(0.5))',
    ]);
    expect(values[0]).toBeCloseTo(1, 6);
    expect(values[1]).toBeCloseTo(0.2126, 6);
    expect(values[2]).toBeCloseTo(0.7152, 6);
    expect(values[3]).toBeCloseTo(0.0722, 6);
    expect(values[4]).toBe(0);
    expect(values[5]).toBeCloseTo(0.5, 6);
  });

  describe('tonemappers', () => {
    // x from 0 to 64, dense near the knee, in order, so monotonicity can be checked along the rows.
    const n = 8192;
    const sweep = new Float32Array(n * 4);
    for (let i = 0; i < n; i++) sweep[i * 4] = 64 * (i / (n - 1)) ** 2;

    const expressions = [
      ['tonemap_reinhard', 'tonemap_reinhard(c)'],
      ['tonemap_reinhard_white', 'tonemap_reinhard_white(c, 4.0)'],
      ['tonemap_aces', 'tonemap_aces(c)'],
    ] as const;

    it.each(expressions)(
      '%s: maps 0 to 0, stays in [0, 1], and never decreases',
      async (_name, call) => {
        const out = await gpuF32(
          `${include}
         fn f(x: vec4f) -> vec4f {
           let c = vec3f(x.x);
           let t = ${call};
           return vec4f(t, 0.0);
         }`,
          sweep,
        );
        const values = lane(out, 0);
        expect(values[0]).toBe(0);
        expect(Math.min(...values)).toBeGreaterThanOrEqual(0);
        expect(Math.max(...values)).toBeLessThanOrEqual(1);
        for (let i = 1; i < values.length; i++) {
          if ((values[i] ?? 0) < (values[i - 1] ?? 0))
            throw new Error(`${call} decreases at x=${sweep[i * 4]}`);
        }
        expect(values[values.length - 1]).toBeGreaterThan(0.9); // approaches white for big inputs
      },
    );

    it('known values: Reinhard(1) = 1/2, extended Reinhard hits 1 at the white point, ACES(1) ~ 0.80', async () => {
      const values = await scalars(include, [
        'tonemap_reinhard(vec3f(1.0)).x',
        'tonemap_reinhard(vec3f(3.0)).y',
        'tonemap_reinhard_white(vec3f(4.0), 4.0).z',
        'tonemap_reinhard_white(vec3f(0.5), 0.5).x',
        'tonemap_reinhard_white(vec3f(1.0), 4.0).x',
        'tonemap_aces(vec3f(1.0)).x',
        'tonemap_aces(vec3f(0.18)).x',
        'tonemap_aces(vec3f(100.0)).x',
        'tonemap_aces(vec3f(-1.0)).x',
        'tonemap_reinhard(vec3f(-1.0)).x',
        'tonemap_aces(vec3f(0.2, 1.0, 5.0)).y - tonemap_aces(vec3f(1.0)).x', // channels are independent
        'tonemap_aces(vec3f(7.0)).x',
        'tonemap_aces(vec3f(7.5)).x',
      ]);
      expect(values[0]).toBeCloseTo(0.5, 6);
      expect(values[1]).toBeCloseTo(0.75, 6);
      expect(values[2]).toBeCloseTo(1, 6);
      expect(values[3]).toBeCloseTo(1, 6);
      expect(values[4]).toBeCloseTo((1 * (1 + 1 / 16)) / 2, 6);
      expect(values[5]).toBeCloseTo(2.54 / 3.16, 5);
      expect(values[6]).toBeCloseTo(
        (0.18 * (2.51 * 0.18 + 0.03)) / (0.18 * (2.43 * 0.18 + 0.59) + 0.14),
        5,
      );
      expect(values[7]).toBe(1);
      expect(values[8]).toBe(0);
      expect(values[9]).toBe(0);
      expect(values[10]).toBe(0);
      expect(values[11]).toBeLessThan(1); // the documented clip point (~7.2)
      expect(values[11]).toBeGreaterThan(0.99);
      expect(values[12]).toBe(1);
    });

    it('the extended Reinhard keeps more highlight contrast than the plain one below the white point', async () => {
      const values = await scalars(include, [
        'tonemap_reinhard(vec3f(2.0)).x',
        'tonemap_reinhard_white(vec3f(2.0), 4.0).x',
        'tonemap_reinhard_white(vec3f(2.0), 100.0).x - tonemap_reinhard(vec3f(2.0)).x',
      ]);
      expect(values[1]!).toBeGreaterThan(values[0]!);
      expect(Math.abs(values[2]!)).toBeLessThan(1e-3); // a huge white point converges to plain Reinhard
    });
  });

  it('palette_cosine evaluates a + b cos(2 pi (c t + d))', async () => {
    const values = await scalars(include, [
      'palette_cosine(0.0, vec3f(0.5), vec3f(0.5), vec3f(1.0), vec3f(0.0)).x',
      'palette_cosine(0.5, vec3f(0.5), vec3f(0.5), vec3f(1.0), vec3f(0.0)).y',
      'palette_cosine(0.25, vec3f(0.5), vec3f(0.5), vec3f(1.0), vec3f(0.0)).z',
      'palette_cosine(1.0, vec3f(0.5), vec3f(0.5), vec3f(1.0), vec3f(0.0)).x',
      'palette_cosine(0.3, vec3f(0.5), vec3f(0.5), vec3f(1.0, 1.0, 1.0), vec3f(0.0, 0.33, 0.67)).y',
      'palette_cosine(0.7, vec3f(0.8, 0.5, 0.4), vec3f(0.2, 0.4, 0.2), vec3f(2.0, 1.0, 1.0), vec3f(0.0, 0.25, 0.25)).z',
    ]);
    const tau = 2 * Math.PI;
    expect(values[0]).toBeCloseTo(1, 5);
    expect(values[1]).toBeCloseTo(0, 5);
    expect(values[2]).toBeCloseTo(0.5, 5);
    expect(values[3]).toBeCloseTo(1, 5);
    expect(values[4]).toBeCloseTo(0.5 + 0.5 * Math.cos(tau * (0.3 + 0.33)), 5);
    expect(values[5]).toBeCloseTo(0.4 + 0.2 * Math.cos(tau * (1 * 0.7 + 0.25)), 5);
  });
});

// --- Math ----------------------------------------------------------------------------------------

describe('math.wgsl', () => {
  const include = '#include "math.wgsl"\n';

  it('constants, rotation, remap and smootherstep', async () => {
    const values = await scalars(include, [
      'PI',
      'TAU',
      'HALF_PI',
      'rotate_2d(vec2f(1.0, 0.0), HALF_PI).x',
      'rotate_2d(vec2f(1.0, 0.0), HALF_PI).y',
      'rotate_2d(vec2f(2.0, 1.0), 0.7).x - (rotation_2d(0.7) * vec2f(2.0, 1.0)).x',
      'rotate_2d(vec2f(2.0, 1.0), 0.7).y - (rotation_2d(0.7) * vec2f(2.0, 1.0)).y',
      'length(rotate_2d(vec2f(3.0, 4.0), 1.234))',
      'remap(5.0, 0.0, 10.0, 100.0, 200.0)',
      'remap(15.0, 0.0, 10.0, 0.0, 1.0)',
      'remap(0.25, 0.0, 1.0, 1.0, -1.0)',
      'smootherstep(0.0, 1.0, 0.5)',
      'smootherstep(2.0, 4.0, 1.0)',
      'smootherstep(2.0, 4.0, 9.0)',
      'smootherstep(0.0, 1.0, 0.25)',
      'smootherstep(0.0, 1.0, 1e-3) / 1e-3', // flat at the start: derivative 0
    ]);
    expect(values[0]).toBeCloseTo(Math.PI, 6);
    expect(values[1]).toBeCloseTo(2 * Math.PI, 6);
    expect(values[2]).toBeCloseTo(Math.PI / 2, 6);
    expect(values[3]).toBeCloseTo(0, 6);
    expect(values[4]).toBeCloseTo(1, 6);
    expect(Math.abs(values[5]!)).toBeLessThan(1e-6);
    expect(Math.abs(values[6]!)).toBeLessThan(1e-6);
    expect(values[7]).toBeCloseTo(5, 5);
    expect(values[8]).toBe(150);
    expect(values[9]).toBe(1.5);
    expect(values[10]).toBe(0.5);
    expect(values[11]).toBe(0.5);
    expect(values[12]).toBe(0);
    expect(values[13]).toBe(1);
    expect(values[14]).toBeCloseTo(0.25 ** 3 * (0.25 * (0.25 * 6 - 15) + 10), 6);
    expect(values[15]).toBeLessThan(1e-4);
  });

  it('wrap_2d / wrap_3d are floor-modulo (always in [0, period))', async () => {
    const cases: [string, number][] = [
      ['f32(wrap_2d(vec2i(-1, 5), vec2i(4, 3)).x)', 3],
      ['f32(wrap_2d(vec2i(-1, 5), vec2i(4, 3)).y)', 2],
      ['f32(wrap_2d(vec2i(-8, -9), vec2i(4, 3)).x)', 0],
      ['f32(wrap_2d(vec2i(-8, -9), vec2i(4, 3)).y)', 0],
      ['f32(wrap_2d(vec2i(3, 2), vec2i(4, 3)).x)', 3],
      ['f32(wrap_2d(vec2i(4, 3), vec2i(4, 3)).y)', 0],
      ['f32(wrap_2d(vec2i(7, -7), vec2i(1, 1)).x)', 0],
      ['f32(wrap_2d(vec2i(7, -7), vec2i(0, -5)).y)', 0], // invalid periods clamp to 1
      ['f32(wrap_3d(vec3i(-1, -2, -3), vec3i(4, 5, 6)).x)', 3],
      ['f32(wrap_3d(vec3i(-1, -2, -3), vec3i(4, 5, 6)).y)', 3],
      ['f32(wrap_3d(vec3i(-1, -2, -3), vec3i(4, 5, 6)).z)', 3],
      ['f32(wrap_3d(vec3i(13, 14, 15), vec3i(4, 5, 6)).z)', 3],
    ];
    const values = await scalars(
      include,
      cases.map(([e]) => e),
    );
    cases.forEach(([expression, expected], i) => expect(values[i], expression).toBe(expected));
  });
});

// --- A water-effect fragment shader built from the library ---------------------------------------

describe('library in use', () => {
  it('renders an animated, warped, tonemapped, sRGB-encoded water shader', async () => {
    const device = await getTestDevice();
    const size = 64;
    const code = `
      #include "fbm.wgsl"
      #include "color.wgsl"
      struct Params { time: f32, aspect: f32 }
      @group(0) @binding(0) var<uniform> params: Params;
      struct Varying { @builtin(position) position: vec4f, @location(0) uv: vec2f }
      @vertex fn vs(@builtin(vertex_index) index: u32) -> Varying {
        let corner = vec2f(f32((index << 1u) & 2u), f32(index & 2u));
        return Varying(vec4f(corner * 2.0 - 1.0, 0.0, 1.0), vec2f(corner.x, 1.0 - corner.y));
      }
      @fragment fn fs(in: Varying) -> @location(0) vec4f {
        let p = vec2f(in.uv.x * params.aspect, in.uv.y) * 4.0;
        let warped = domain_warp_2d_anim(p, params.time * 0.2, 0.6, 3, 2.0, 0.5);
        let n = 0.5 + 0.5 * fbm_3d(vec3f(warped, params.time * 0.15), 5, 2.0, 0.5);
        let crest = ridged_2d(warped * 1.7 + 3.0, 3, 2.0, 0.5);
        let base = palette_cosine(n, vec3f(0.05, 0.25, 0.4), vec3f(0.1, 0.25, 0.3), vec3f(1.0), vec3f(0.55, 0.6, 0.7));
        let lit = base + vec3f(0.9, 0.95, 1.0) * pow(crest, 6.0) * 0.5;
        return vec4f(linear_to_srgb(tonemap_aces(lit)), 1.0);
      }`;
    const { module, diagnostics } = await checkShader(device, {
      label: 'water-example.wgsl',
      code,
    });
    expect(diagnostics.map(formatDiagnostic)).toEqual([]);

    const uniforms = device.createBuffer({
      size: 16,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    const target = device.createTexture({
      size: [size, size],
      format: 'rgba8unorm',
      usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC,
    });
    const render = async (time: number): Promise<Uint8Array> => {
      device.queue.writeBuffer(uniforms, 0, new Float32Array([time, 1, 0, 0]));
      await expectValid(device, () => {
        const pipeline = device.createRenderPipeline({
          layout: 'auto',
          vertex: { module, entryPoint: 'vs' },
          fragment: { module, entryPoint: 'fs', targets: [{ format: 'rgba8unorm' }] },
        });
        const bindGroup = device.createBindGroup({
          layout: pipeline.getBindGroupLayout(0),
          entries: [{ binding: 0, resource: { buffer: uniforms } }],
        });
        const encoder = device.createCommandEncoder();
        const pass = encoder.beginRenderPass({
          colorAttachments: [{ view: target.createView(), loadOp: 'clear', storeOp: 'store' }],
        });
        pass.setPipeline(pipeline);
        pass.setBindGroup(0, bindGroup);
        pass.draw(3);
        pass.end();
        device.queue.submit([encoder.finish()]);
      });
      return readTexture(device, target);
    };

    const a = await render(0);
    const b = await render(5);
    const distinct = new Set<number>();
    let luma = 0;
    for (let i = 0; i < a.length; i += 4) {
      expect(a[i + 3]).toBe(255);
      distinct.add(((a[i] ?? 0) << 16) | ((a[i + 1] ?? 0) << 8) | (a[i + 2] ?? 0));
      luma += ((a[i] ?? 0) + (a[i + 1] ?? 0) + (a[i + 2] ?? 0)) / 3;
    }
    luma /= size * size;
    expect(distinct.size).toBeGreaterThan(200); // a smooth gradient, not a flat colour or noise bands
    expect(luma).toBeGreaterThan(40); // neither black nor blown out
    expect(luma).toBeLessThan(215);
    // Blue-green water: blue dominates red on average.
    let red = 0;
    let blue = 0;
    for (let i = 0; i < a.length; i += 4) {
      red += a[i] ?? 0;
      blue += a[i + 2] ?? 0;
    }
    expect(blue).toBeGreaterThan(red);
    // The animation moves the pattern.
    let changed = 0;
    for (let i = 0; i < a.length; i++) if (Math.abs((a[i] ?? 0) - (b[i] ?? 0)) > 4) changed++;
    expect(changed).toBeGreaterThan(size * size);
    uniforms.destroy();
    target.destroy();
  });
});
