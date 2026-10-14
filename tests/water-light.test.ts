import { describe, expect, it, vi } from 'vitest';
import { listEffects, loadEffect } from '../src/effects/registry.ts';
import { defaultParams, type ParamSchema } from '../src/effects/types.ts';
import { createCausticsMap, createCausticsSampler } from '../src/gpu/bake-caustics.ts';
import { noProfiler } from '../src/gpu/profiler.ts';
import { checkShader, formatDiagnostic } from '../src/gpu/shader.ts';
import { getChunk } from '../src/wgsl/index.ts';
import { expectValid, getTestDevice } from './gpu.ts';
import { gpuF32, lane, rowsOf } from './wgsl-eval.ts';

// Grids of tens of thousands of samples, and a few dozen headless frames: leave headroom.
vi.setConfig({ testTimeout: 60_000 });

const TAU = Math.PI * 2;

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

const mean = (values: ArrayLike<number>) => {
  let sum = 0;
  for (let i = 0; i < values.length; i++) sum += values[i] ?? 0;
  return sum / values.length;
};

const allFinite = (values: ArrayLike<number>) => {
  for (let i = 0; i < values.length; i++) if (!Number.isFinite(values[i])) return false;
  return true;
};

const minOf = (values: ArrayLike<number>) => {
  let m = Infinity;
  for (let i = 0; i < values.length; i++) m = Math.min(m, values[i] ?? 0);
  return m;
};

const maxOf = (values: ArrayLike<number>) => {
  let m = -Infinity;
  for (let i = 0; i < values.length; i++) m = Math.max(m, values[i] ?? 0);
  return m;
};

/** Rows (u, v, t, s) over an n x n grid of texel centres, at each time. */
function gridRows(n: number, times: number[], sharp: number): Float32Array {
  const rows = new Float32Array(n * n * times.length * 4);
  let k = 0;
  for (const t of times) {
    for (let y = 0; y < n; y++) {
      for (let x = 0; x < n; x++) {
        rows.set([(x + 0.5) / n, (y + 0.5) / n, t, sharp], k);
        k += 4;
      }
    }
  }
  return rows;
}

function sameBits(a: Float32Array, b: Float32Array): boolean {
  const x = new Uint32Array(a.buffer, a.byteOffset, a.length);
  const y = new Uint32Array(b.buffer, b.byteOffset, b.length);
  return x.length === y.length && x.every((word, i) => word === y[i]);
}

function halfToFloat(h: number): number {
  const sign = h & 0x8000 ? -1 : 1;
  const exponent = (h >> 10) & 0x1f;
  const fraction = h & 0x3ff;
  if (exponent === 0) return sign * 2 ** -14 * (fraction / 1024);
  if (exponent === 31) return fraction ? Number.NaN : sign * Infinity;
  return sign * 2 ** (exponent - 15) * (1 + fraction / 1024);
}

/** Mip 0 of an rgba16float texture (needs COPY_SRC) as RGBA floats. */
async function readHalfTexture(device: GPUDevice, texture: GPUTexture): Promise<Float32Array> {
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

/** Runs `expr` (a vec4f expression of `x: vec4f`) over rows, with the baked map bound as `tex`/`samp`. */
async function sampleBaked(
  device: GPUDevice,
  view: GPUTextureView,
  rows: Float32Array,
  expr: string,
): Promise<Float32Array> {
  const count = rows.length / 4;
  const { module, diagnostics } = await checkShader(device, {
    label: 'caustics-baked-harness',
    code: `#include "caustics.wgsl"
@group(0) @binding(0) var tex: texture_2d<f32>;
@group(0) @binding(1) var samp: sampler;
@group(0) @binding(2) var<storage, read> inputs: array<vec4f>;
@group(0) @binding(3) var<storage, read_write> outputs: array<vec4f>;
@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) id: vec3u) {
  if (id.x < ${count}u) { let x = inputs[id.x]; outputs[id.x] = ${expr}; }
}`,
  });
  expect(diagnostics.filter((d) => d.severity === 'error').map(formatDiagnostic)).toEqual([]);
  const buffer = (usage: number) => device.createBuffer({ size: rows.byteLength, usage });
  const input = buffer(GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
  device.queue.writeBuffer(input, 0, rows);
  const output = buffer(GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
  const read = buffer(GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST);
  const pipeline = device.createComputePipeline({
    layout: 'auto',
    compute: { module, entryPoint: 'main' },
  });
  const group = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: view },
      { binding: 1, resource: createCausticsSampler(device) },
      { binding: 2, resource: { buffer: input } },
      { binding: 3, resource: { buffer: output } },
    ],
  });
  const encoder = device.createCommandEncoder();
  const pass = encoder.beginComputePass();
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, group);
  pass.dispatchWorkgroups(Math.ceil(count / 64));
  pass.end();
  encoder.copyBufferToBuffer(output, 0, read, 0, rows.byteLength);
  device.queue.submit([encoder.finish()]);
  await read.mapAsync(GPUMapMode.READ);
  const result = new Float32Array(read.getMappedRange().slice(0));
  read.unmap();
  for (const b of [input, output, read]) b.destroy();
  return result;
}

// --- Chunk hygiene -------------------------------------------------------------------------------

const CHUNKS: Record<string, string> = {
  'caustics.wgsl': 'caustics_',
  'shafts.wgsl': 'shafts_',
  'motes.wgsl': 'motes_',
};

describe('water-light chunks', () => {
  const sources = Object.keys(CHUNKS).map((name) => [name, getChunk(name) ?? ''] as const);

  it.each(sources)('%s is a pure library with a documented header', (_name, source) => {
    expect(source.length).toBeGreaterThan(0);
    expect(source.startsWith('//')).toBe(true);
    const code = source
      .split('\n')
      .filter((line) => !line.trimStart().startsWith('//'))
      .join('\n');
    expect(code).not.toMatch(/@(vertex|fragment|compute)\b/);
    expect(code).not.toMatch(/@(group|binding)\b/);
    expect(code).not.toMatch(/^var\b/m);
    const header = source
      .split('\n')
      .filter((line) => line.startsWith('//'))
      .join('\n');
    const publicFunctions = [...source.matchAll(/^fn ([a-z]\w*)\(/gm)].map((m) => m[1] ?? '');
    expect(publicFunctions.length).toBeGreaterThan(0);
    expect(publicFunctions.filter((fn) => !header.includes(fn))).toEqual([]);
  });

  it('every name carries the file prefix; private helpers start with an underscore', () => {
    for (const [name, source] of sources) {
      const prefix = CHUNKS[name] ?? '';
      for (const match of source.matchAll(/^fn (\w+)\(/gm)) {
        const fn = match[1] ?? '';
        expect(fn.startsWith(prefix) || fn.startsWith('_'), `${name}: ${fn}`).toBe(true);
      }
      for (const match of source.matchAll(/^const (\w+)/gm)) {
        const id = (match[1] ?? '').replace(/^_/, '').toLowerCase();
        expect(id.startsWith(prefix), `${name}: const ${match[1]}`).toBe(true);
      }
    }
  });
});

// --- Caustics: the procedural constructions ---------------------------------------------------------

const METHODS = [
  [0, 'cellular'],
  [1, 'sine'],
] as const;
const CELLS = 12;

const evalCode = (method: number, call = 'eval') =>
  `#include "caustics.wgsl"
   fn f(x: vec4f) -> vec4f {
     return vec4f(caustics_${call}(${method}u, x.xy, ${CELLS}, x.z, x.w), 0.0, 0.0, 0.0);
   }`;

describe.each(METHODS)('caustics_%s (method %s)', (method, _label) => {
  const times = [0, 1.3, 2.9, 4.4, 5.9];

  it('is finite and non-negative over position, time and sharpness', async () => {
    for (const sharp of [0, 0.5, 1]) {
      const out = lane(await gpuF32(evalCode(method), gridRows(96, times, sharp)), 0);
      expect(allFinite(out)).toBe(true);
      expect(minOf(out)).toBeGreaterThanOrEqual(0);
    }
    // Sharpness and cells outside their ranges are clamped, never NaN.
    const wild = lane(
      await gpuF32(
        evalCode(method),
        rowsOf([
          [0.3, 0.7, 2, -3],
          [0.3, 0.7, 2, 9],
          [-4.2, 17.9, -50, 0.5],
        ]),
      ),
      0,
    );
    expect(allFinite(wild)).toBe(true);
    expect(minOf(wild)).toBeGreaterThanOrEqual(0);
  });

  it('repeats exactly every 1.0 in x and in y (and every TAU in time)', async () => {
    const random = makeRandom(11 + method);
    const base: number[][] = [];
    for (let i = 0; i < 1500; i++) {
      // Multiples of 1/1024 stay exact when 1 is added, so only the maths can break the period.
      base.push([
        Math.floor(random() * 1024) / 1024,
        Math.floor(random() * 1024) / 1024,
        Math.round(random() * 600) / 100,
        random(),
      ]);
    }
    const at = async (dx: number, dy: number, dt: number) =>
      lane(
        await gpuF32(
          evalCode(method),
          rowsOf(base.map(([u, v, t, s]) => [(u ?? 0) + dx, (v ?? 0) + dy, (t ?? 0) + dt, s ?? 0])),
        ),
        0,
      );
    const origin = await at(0, 0, 0);
    // A few samples can sit on a cell-site decision boundary and differ by float noise in
    // which cell wins; everything else must match to a thousandth.
    for (const [dx, dy, dt] of [
      [1, 0, 0],
      [0, 1, 0],
      [-1, 2, 0],
      [0, 0, TAU],
    ] as const) {
      const shifted = await at(dx, dy, dt);
      let bad = 0;
      for (let i = 0; i < origin.length; i++) {
        const a = origin[i] ?? 0;
        const b = shifted[i] ?? 0;
        if (Math.abs(a - b) > 2e-3 * (1 + Math.max(a, b))) bad++;
      }
      expect(bad, `shift (${dx}, ${dy}, ${dt})`).toBeLessThanOrEqual(origin.length * 0.003);
    }
  });

  it('is not trivially periodic at a smaller scale: moving by half a tile changes it', async () => {
    const base: number[][] = [];
    const random = makeRandom(5);
    for (let i = 0; i < 400; i++) base.push([random(), random(), 1, 0.6]);
    const a = lane(await gpuF32(evalCode(method), rowsOf(base)), 0);
    const b = lane(
      await gpuF32(
        evalCode(method),
        rowsOf(base.map(([u, v, t, s]) => [(u ?? 0) + 0.5, v ?? 0, t ?? 0, s ?? 0])),
      ),
      0,
    );
    let differing = 0;
    for (let i = 0; i < a.length; i++) if (Math.abs((a[i] ?? 0) - (b[i] ?? 0)) > 0.05) differing++;
    expect(differing).toBeGreaterThan(a.length * 0.5);
  });

  it('keeps its mean near 1 whatever the sharpness (energy is redistributed, not added)', async () => {
    const means: number[] = [];
    const peaks: number[] = [];
    for (const sharp of [0, 0.25, 0.5, 0.75, 1]) {
      const out = lane(await gpuF32(evalCode(method), gridRows(128, times, sharp)), 0);
      means.push(mean(out));
      peaks.push(maxOf(out));
    }
    for (const m of means) {
      expect(m).toBeGreaterThan(0.9);
      expect(m).toBeLessThan(1.1);
    }
    // Same energy, thinner lines: the peaks must grow a lot.
    expect(peaks[4] ?? 0).toBeGreaterThan(3 * (peaks[0] ?? 1));
  });

  it('two stacked layers keep the mean and stay non-negative', async () => {
    for (const sharp of [0.2, 0.8]) {
      const out = lane(await gpuF32(evalCode(method, 'stack'), gridRows(128, times, sharp)), 0);
      expect(allFinite(out)).toBe(true);
      expect(minOf(out)).toBeGreaterThanOrEqual(0);
      expect(mean(out)).toBeGreaterThan(0.88);
      expect(mean(out)).toBeLessThan(1.12);
    }
  });

  it('is deterministic', async () => {
    const rows = gridRows(64, [0.7, 3.1], 0.6);
    const a = await gpuF32(evalCode(method), rows);
    const b = await gpuF32(evalCode(method), rows);
    expect(sameBits(a, b)).toBe(true);
  });

  it('animates: a quarter loop later the pattern is different', async () => {
    const rows0 = gridRows(48, [1], 0.6);
    const rows1 = gridRows(48, [1 + TAU / 4], 0.6);
    const a = lane(await gpuF32(evalCode(method), rows0), 0);
    const b = lane(await gpuF32(evalCode(method), rows1), 0);
    let moved = 0;
    for (let i = 0; i < a.length; i++) if (Math.abs((a[i] ?? 0) - (b[i] ?? 0)) > 0.05) moved++;
    expect(moved).toBeGreaterThan(a.length * 0.3);
  });
});

describe('caustics_rgb and caustics_expose', () => {
  const code = `#include "caustics.wgsl"
    fn f(x: vec4f) -> vec4f {
      let c = caustics_rgb(0u, 2u, x.xy, ${CELLS}, x.z, x.w, 0.01);
      return vec4f(c, 0.0);
    }`;

  it('fringing gives three finite, non-negative channels, each with a mean near 1', async () => {
    const out = await gpuF32(code, gridRows(96, [0.5, 2.5, 4.5], 0.6));
    expect(allFinite(out)).toBe(true);
    for (let c = 0; c < 3; c++) {
      const channel = lane(out, c);
      expect(minOf(channel)).toBeGreaterThanOrEqual(0);
      expect(mean(channel)).toBeGreaterThan(0.85);
      expect(mean(channel)).toBeLessThan(1.15);
    }
    // The channels are shifted copies of each other, not equal.
    const r = lane(out, 0);
    const b = lane(out, 2);
    let differing = 0;
    for (let i = 0; i < r.length; i++) if (Math.abs((r[i] ?? 0) - (b[i] ?? 0)) > 0.02) differing++;
    expect(differing).toBeGreaterThan(r.length * 0.1);
  });

  it('split 0 returns the same grey pattern in every channel', async () => {
    const grey = `#include "caustics.wgsl"
      fn f(x: vec4f) -> vec4f {
        return vec4f(caustics_rgb(1u, 1u, x.xy, ${CELLS}, x.z, x.w, 0.0), 0.0);
      }`;
    const out = await gpuF32(grey, gridRows(32, [1], 0.5));
    expect(sameBits(Float32Array.from(lane(out, 0)), Float32Array.from(lane(out, 1)))).toBe(true);
    expect(sameBits(Float32Array.from(lane(out, 0)), Float32Array.from(lane(out, 2)))).toBe(true);
  });

  it('caustics_expose maps [0, inf) into [0, 1) and is monotonic', async () => {
    const xs = [0, 0.1, 0.5, 1, 2, 10, 100, 1e6];
    const out = lane(
      await gpuF32(
        `#include "caustics.wgsl"
         fn f(x: vec4f) -> vec4f { return vec4f(caustics_expose(x.x, x.y), caustics_expose(-1.0, 1.0), 0.0, 0.0); }`,
        rowsOf(xs.map((c) => [c, 1.5])),
      ),
      0,
    );
    expect(out[0]).toBe(0);
    for (let i = 1; i < out.length; i++) {
      // Strictly rising until f32 rounds 1 - exp(-big) to exactly 1.
      expect(out[i] ?? 0).toBeGreaterThanOrEqual(out[i - 1] ?? 0);
      expect(out[i] ?? 0).toBeLessThanOrEqual(1);
    }
    expect(out[5] ?? 0).toBeGreaterThan(out[4] ?? 1);
    expect(out[3]).toBeCloseTo(1 - Math.exp(-1.5), 5);
  });
});

// --- Caustics: the baked variant ---------------------------------------------------------------------

describe('baked caustic map', () => {
  it('is tileable, finite, non-negative, mean about 1 in every time slice', async () => {
    const device = await getTestDevice();
    const size = 256;
    const map = createCausticsMap(device, { size, label: 'test caustics' });
    expect(map.format).toBe('rgba16float');
    expect(map.mipLevelCount).toBe(9);
    const data = await readHalfTexture(device, map.texture);
    expect(allFinite(data)).toBe(true);
    expect(minOf(data)).toBeGreaterThanOrEqual(0);
    for (let c = 0; c < 4; c++) {
      let sum = 0;
      for (let i = c; i < data.length; i += 4) sum += data[i] ?? 0;
      const m = sum / (size * size);
      expect(m, `slice ${c}`).toBeGreaterThan(0.9);
      expect(m, `slice ${c}`).toBeLessThan(1.1);
    }
    // Seam test: the step across the wrap-around edge must look like any interior step.
    const px = (x: number, y: number, c: number) => data[(y * size + x) * 4 + c] ?? 0;
    let seam = 0;
    let interior = 0;
    for (let y = 0; y < size; y++) {
      seam += Math.abs(px(0, y, 0) - px(size - 1, y, 0));
      interior += Math.abs(px(size / 2, y, 0) - px(size / 2 - 1, y, 0));
    }
    let seamV = 0;
    let interiorV = 0;
    for (let x = 0; x < size; x++) {
      seamV += Math.abs(px(x, 0, 0) - px(x, size - 1, 0));
      interiorV += Math.abs(px(x, size / 2, 0) - px(x, size / 2 - 1, 0));
    }
    expect(seam).toBeLessThan(3 * interior + 1);
    expect(seamV).toBeLessThan(3 * interiorV + 1);
    map.destroy();
  });

  it('rejects sizes that are not a power of two', async () => {
    const device = await getTestDevice();
    expect(() => createCausticsMap(device, { size: 300 })).toThrow(RangeError);
  });

  it('caustics_baked_slice blends the four slices cyclically', async () => {
    const out = await gpuF32(
      `#include "caustics.wgsl"
       fn f(x: vec4f) -> vec4f {
         let s = vec4f(1.0, 2.0, 3.0, 4.0);
         return vec4f(
           caustics_baked_slice(s, x.x),
           caustics_baked_slice(s, x.x + 6.283185307),
           caustics_baked_slice(s, x.x - 12.56637),
           caustics_baked_slice(s, x.x + 0.5 * 6.283185307 / 4.0),
         );
       }`,
      rowsOf([[0], [1.2], [3.3], [6.0], [0.4]]),
    );
    const r = lane(out, 0);
    expect(r[0]).toBeCloseTo(1, 4);
    for (let i = 0; i < 5; i++) {
      expect(lane(out, 1)[i]).toBeCloseTo(r[i] ?? 0, 3);
      expect(lane(out, 2)[i]).toBeCloseTo(r[i] ?? 0, 3);
      expect(r[i] ?? 0).toBeGreaterThanOrEqual(1);
      expect(r[i] ?? 0).toBeLessThanOrEqual(4);
    }
    // Halfway between slices 0 and 1 (smoothstep is 0.5 there) is the average.
    expect(lane(out, 3)[0]).toBeCloseTo(1.5, 3);
  });

  it('caustics_baked: finite, non-negative, mean near 1 for both combines and every lod', async () => {
    const device = await getTestDevice();
    const map = createCausticsMap(device, { size: 512, label: 'test caustics 512' });
    const n = 96;
    for (const mode of [0, 1, 2]) {
      for (const lod of [0, 1, 2, 3]) {
        const rows = gridRows(n, [0, 1.3, 2.9, 4.4, 5.9], lod);
        const out = lane(
          await sampleBaked(
            device,
            map.view,
            rows,
            `vec4f(caustics_baked(tex, samp, x.xy, x.z, x.w, ${mode}u), 0.0, 0.0, 0.0)`,
          ),
          0,
        );
        expect(allFinite(out)).toBe(true);
        expect(minOf(out)).toBeGreaterThanOrEqual(0);
        // The average combine keeps the mean exactly; min is a fitted rescale, good to about 15%.
        const band = 0.15;
        expect(mean(out), `mode ${mode} lod ${lod}`).toBeGreaterThan(1 - band);
        expect(mean(out), `mode ${mode} lod ${lod}`).toBeLessThan(1 + band);
      }
    }
    map.destroy();
  });

  it('caustics_baked_rgb: split 0 is grey, split > 0 fringes', async () => {
    const device = await getTestDevice();
    const map = createCausticsMap(device, { size: 256 });
    const rows = gridRows(48, [1.7], 1);
    const grey = await sampleBaked(
      device,
      map.view,
      rows,
      'vec4f(caustics_baked_rgb(tex, samp, x.xy, x.z, x.w, 0u, 0.0), 0.0)',
    );
    expect(sameBits(Float32Array.from(lane(grey, 0)), Float32Array.from(lane(grey, 2)))).toBe(true);
    const fringed = await sampleBaked(
      device,
      map.view,
      rows,
      'vec4f(caustics_baked_rgb(tex, samp, x.xy, x.z, x.w, 0u, 0.01), 0.0)',
    );
    expect(allFinite(fringed)).toBe(true);
    expect(minOf(fringed)).toBeGreaterThanOrEqual(0);
    expect(sameBits(Float32Array.from(lane(fringed, 0)), Float32Array.from(lane(fringed, 2)))).toBe(
      false,
    );
    map.destroy();
  });
});

// --- Shafts ---------------------------------------------------------------------------------------

describe('shafts.wgsl', () => {
  const code = `#include "shafts.wgsl"
    fn f(x: vec4f) -> vec4f {
      let p = vec2f(x.x, x.y);
      let source = vec2f(0.8, -x.w);
      return vec4f(
        shafts_light(p, source, x.z, 6.0, 0.55, 0.9),
        shafts_beams(shafts_fan(p, source), x.z, 8.0, 1.0),
        shafts_beams(shafts_fan(p, source), x.z, 8.0, 0.0),
        shafts_falloff(p, source, 0.9),
      );
    }`;

  it('every output stays inside [0, 1] and is finite, including right at the source', async () => {
    const random = makeRandom(3);
    const rows: number[][] = [];
    for (let i = 0; i < 20000; i++) {
      rows.push([random() * 2.4 - 0.2, random() * 1.4 - 0.2, random() * 200, 0.02 + random() * 5]);
    }
    rows.push([0.8, -0.5, 1, 0.5], [0.8, 0, 1, 0.5], [0.8, -0.02, 3, 0.02]);
    const out = await gpuF32(code, rowsOf(rows));
    expect(allFinite(out)).toBe(true);
    for (let c = 0; c < 4; c++) {
      const channel = lane(out, c);
      expect(minOf(channel), `lane ${c}`).toBeGreaterThanOrEqual(0);
      expect(maxOf(channel), `lane ${c}`).toBeLessThanOrEqual(1.0001);
    }
  });

  it('the fan coordinate is constant along a ray and the falloff decays with depth', async () => {
    const source = [0.8, -0.3];
    const dir = [0.25, 1]; // a ray leaving the source
    const points = [0.4, 0.8, 1.2, 1.6].map((k) => [
      (source[0] ?? 0) + (dir[0] ?? 0) * k,
      (source[1] ?? 0) + (dir[1] ?? 0) * k,
    ]);
    const out = await gpuF32(
      `#include "shafts.wgsl"
       fn f(x: vec4f) -> vec4f {
         let s = vec2f(0.8, -0.3);
         let p = x.xy;
         return vec4f(shafts_fan(p, s), shafts_falloff(vec2f(x.z, x.w), s, 0.8), 0.0, 0.0);
       }`,
      rowsOf(points.map((p) => [p[0] ?? 0, p[1] ?? 0, 0.5, (p[1] ?? 0) + 0.0])),
    );
    const fan = lane(out, 0);
    for (const value of fan) expect(value).toBeCloseTo(0.25, 4);

    const depths = [0.02, 0.2, 0.5, 0.9];
    const falloff = lane(
      await gpuF32(
        `#include "shafts.wgsl"
         fn f(x: vec4f) -> vec4f { return vec4f(shafts_falloff(vec2f(0.8, x.x), vec2f(0.8, -0.3), 1.0), 0.0, 0.0, 0.0); }`,
        rowsOf(depths.map((d) => [d])),
      ),
      0,
    );
    // Straight below the source the slow ripple is the only wobble; the envelope still decays.
    expect(falloff[3] ?? 1).toBeLessThan(falloff[0] ?? 0);
  });

  it('contrast pushes the beams toward on and off, and the pattern moves with time', async () => {
    const rows: number[][] = [];
    for (let i = 0; i < 4000; i++)
      rows.push([0.1 + (i % 100) / 60, 0.3 + Math.floor(i / 100) / 40, 5, 0.4]);
    const out = await gpuF32(code, rowsOf(rows));
    const variance = (values: number[]) => {
      const m = mean(values);
      return mean(values.map((v) => (v - m) ** 2));
    };
    expect(variance(lane(out, 1))).toBeGreaterThan(variance(lane(out, 2)));
    const later = await gpuF32(
      code,
      rowsOf(rows.map(([a, b, , d]) => [a ?? 0, b ?? 0, 50, d ?? 0])),
    );
    let moved = 0;
    const a = lane(out, 1);
    const b = lane(later, 1);
    for (let i = 0; i < a.length; i++) if (Math.abs((a[i] ?? 0) - (b[i] ?? 0)) > 0.1) moved++;
    expect(moved).toBeGreaterThan(a.length * 0.2);
  });

  it('is deterministic', async () => {
    const rows = rowsOf(Array.from({ length: 500 }, (_, i) => [i / 400, i / 700, i / 9, 0.5]));
    expect(sameBits(await gpuF32(code, rows), await gpuF32(code, rows))).toBe(true);
  });
});

// --- Motes ----------------------------------------------------------------------------------------

describe('motes.wgsl', () => {
  const code = `#include "motes.wgsl"
    fn f(x: vec4f) -> vec4f {
      let p = x.xy;
      return vec4f(
        motes_layer(p, x.z, 0.1, 1.0, 0.12, vec2f(0.01, 0.03), 1.0),
        motes_layer(p, x.z, 0.1, 0.5, 0.12, vec2f(0.01, 0.03), 1.0),
        motes_layer(p, x.z, 0.1, 0.0, 0.12, vec2f(0.01, 0.03), 1.0),
        motes_layer(p, x.z, 0.1, 1.0, 0.25, vec2f(0.01, 0.03), 2.0),
      );
    }`;

  function coverage(values: number[]): number {
    let n = 0;
    for (const v of values) if (v > 0.02) n++;
    return n / values.length;
  }

  it('stays in [0, 1], finite, and density 0 draws nothing', async () => {
    const random = makeRandom(8);
    const rows: number[][] = [];
    for (let i = 0; i < 20000; i++)
      rows.push([random() * 3 - 1, random() * 2 - 0.5, random() * 600, 0]);
    const out = await gpuF32(code, rowsOf(rows));
    expect(allFinite(out)).toBe(true);
    for (let c = 0; c < 4; c++) {
      expect(minOf(lane(out, c))).toBeGreaterThanOrEqual(0);
      expect(maxOf(lane(out, c))).toBeLessThanOrEqual(1);
    }
    expect(maxOf(lane(out, 2))).toBe(0);
    expect(maxOf(lane(out, 0))).toBeGreaterThan(0.5);
  });

  it('density scales how many motes there are; radius scales their size; seed changes them', async () => {
    const rows: number[][] = [];
    for (let y = 0; y < 200; y++) for (let x = 0; x < 200; x++) rows.push([x / 100, y / 100, 3, 0]);
    const out = await gpuF32(code, rowsOf(rows));
    const full = coverage(lane(out, 0));
    const half = coverage(lane(out, 1));
    expect(full).toBeGreaterThan(0.005);
    expect(half / full).toBeGreaterThan(0.35);
    expect(half / full).toBeLessThan(0.65);
    expect(coverage(lane(out, 3))).toBeGreaterThan(2 * full);
    let differing = 0;
    const a = lane(out, 0);
    const b = lane(out, 3);
    for (let i = 0; i < a.length; i++) if (Math.abs((a[i] ?? 0) - (b[i] ?? 0)) > 0.1) differing++;
    expect(differing).toBeGreaterThan(0);
  });

  it('motes drift with the layer velocity and are deterministic', async () => {
    const drifting = `#include "motes.wgsl"
      fn f(x: vec4f) -> vec4f {
        // Light a mote's cell at t = 0, then look where the same mote is at t = 10: shifted by drift * t.
        let v = vec2f(0.02, 0.05);
        return vec4f(
          motes_layer(x.xy, 0.0, 0.1, 1.0, 0.12, v, 1.0),
          motes_layer(x.xy + v * 10.0, 10.0, 0.1, 1.0, 0.12, v, 1.0),
          0.0, 0.0);
      }`;
    const rows: number[][] = [];
    for (let y = 0; y < 100; y++) for (let x = 0; x < 100; x++) rows.push([x / 50, y / 50, 0, 0]);
    const a = await gpuF32(drifting, rowsOf(rows));
    // Same lattice cell and same sub-cell position: the shapes agree up to the sway and twinkle terms.
    const lit0 = lane(a, 0).filter((v) => v > 0.2).length;
    const lit1 = lane(a, 1).filter((v) => v > 0.2).length;
    expect(lit0).toBeGreaterThan(0);
    expect(lit1).toBeGreaterThan(0);
    expect(sameBits(a, await gpuF32(drifting, rowsOf(rows)))).toBe(true);
  });
});

// --- The demos, headlessly, at every preset and parameter extreme -------------------------------------

const WIDTH = 64;
const HEIGHT = 48;

function fakeCanvas(): HTMLCanvasElement {
  return Object.assign(new EventTarget(), {
    width: WIDTH,
    height: HEIGHT,
    style: {},
    getBoundingClientRect: () => ({ left: 0, top: 0, width: WIDTH, height: HEIGHT }),
    setPointerCapture() {},
    releasePointerCapture() {},
  }) as unknown as HTMLCanvasElement;
}

const DEMOS = ['caustics', 'pool-floor', 'underwater'];

describe('water-light demos', () => {
  it('are registered as Epic 4 effects with at least two presets and honest scorecards', async () => {
    const metas = listEffects();
    for (const id of DEMOS) {
      const meta = metas.find((m) => m.id === id);
      expect(meta, id).toBeDefined();
      expect(meta?.epic).toBe('4');
      expect(meta?.scorecard).toBeDefined();
      expect(meta?.scorecard?.gpuMs).toBeUndefined();
      expect(meta?.scorecard?.bundleKb).toBeUndefined();
      const effect = await loadEffect(id);
      expect(Object.keys(effect.presets ?? {}).length).toBeGreaterThanOrEqual(2);
    }
  });

  it.each(DEMOS)(
    '%s: no NaN, Inf or negative colour at any preset or parameter extreme',
    async (id) => {
      const device = await getTestDevice();
      const format: GPUTextureFormat = 'rgba16float';
      const target = device.createTexture({
        size: [WIDTH, HEIGHT],
        format,
        usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC,
      });
      const effect = await loadEffect(id);
      const schema: ParamSchema = effect.params;
      const params = defaultParams(schema) as Record<string, unknown>;

      // Each case is a set of overrides applied to the defaults.
      const cases: [string, Record<string, unknown>][] = [['defaults', {}]];
      for (const [name, preset] of Object.entries(effect.presets ?? {})) {
        cases.push([`preset ${name}`, preset as Record<string, unknown>]);
      }
      for (const [key, def] of Object.entries(schema)) {
        if (def.type === 'number') {
          cases.push([`${key} = min`, { [key]: def.min }], [`${key} = max`, { [key]: def.max }]);
        } else if (def.type === 'select') {
          for (const option of def.options) cases.push([`${key} = ${option}`, { [key]: option }]);
        } else if (def.type === 'boolean') {
          cases.push([`${key} = true`, { [key]: true }], [`${key} = false`, { [key]: false }]);
        } else if (def.type === 'color') {
          cases.push(
            [`${key} = black`, { [key]: [0, 0, 0] }],
            [`${key} = white`, { [key]: [1, 1, 1] }],
          );
        }
      }
      // Every number at its maximum at once, and at its minimum at once.
      for (const bound of ['min', 'max'] as const) {
        const all: Record<string, unknown> = {};
        for (const [key, def] of Object.entries(schema))
          if (def.type === 'number') all[key] = def[bound];
        cases.push([`all numbers = ${bound}`, all]);
      }

      const problems: string[] = [];
      await expectValid(device, async () => {
        const instance = await effect.create({
          device,
          format,
          canvas: fakeCanvas(),
          params: params as never,
          profiler: noProfiler,
          hdr: false,
        });
        instance.resize(WIDTH, HEIGHT, 1);
        let frame = 0;
        for (const [label, overrides] of cases) {
          Object.assign(params, defaultParams(schema), overrides);
          for (let i = 0; i < 2; i++) {
            const encoder = device.createCommandEncoder();
            instance.frame(
              { time: frame / 60, delta: 1 / 60, frame },
              encoder,
              target.createView(),
            );
            device.queue.submit([encoder.finish()]);
            frame++;
          }
          const pixels = await readHalfTexture(device, target);
          let bad = 0;
          let lowest = Infinity;
          for (let i = 0; i < pixels.length; i++) {
            const v = pixels[i] ?? Number.NaN;
            if (!Number.isFinite(v) || v < 0) bad++;
            else lowest = Math.min(lowest, v);
          }
          if (bad > 0) problems.push(`${label}: ${bad} bad channel values (lowest ok ${lowest})`);
          // Premultiplied output can never have a colour channel above alpha.
          for (let i = 0; i < pixels.length; i += 4) {
            const a = pixels[i + 3] ?? 0;
            if (
              a > 1.001 ||
              Math.max(pixels[i] ?? 0, pixels[i + 1] ?? 0, pixels[i + 2] ?? 0) > a + 0.01
            ) {
              problems.push(`${label}: pixel ${i / 4} is not valid premultiplied colour`);
              break;
            }
          }
        }
        instance.destroy();
      });
      target.destroy();
      expect(problems).toEqual([]);
    },
  );
});
