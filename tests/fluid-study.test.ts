import { afterAll, beforeAll, describe, expect, it, vi } from 'vitest';
import {
  FluidUniforms,
  MULTIGRID_COARSEST,
  DEFAULT_FLUID_PARAMS,
  createFluid,
  fluidShaderChunks,
  halfToFloat,
  sorOmega,
  multigridLevelSizes,
  pressureSolveWork,
  type Fluid,
  type FluidParams,
} from '../src/gpu/sim/fluid.ts';
import { createShader } from '../src/gpu/shader.ts';
import { createUniformBuffer } from '../src/gpu/uniforms.ts';
import { expectValid, getTestDevice, readTexture } from './gpu.ts';

vi.setConfig({ testTimeout: 1_800_000 });

// Story 6.8, the solver cost study. Two halves:
//
//  1. Tests of the solver features the study added (a multigrid pressure solve, a cubic reconstruction
//     of the velocity for the dye). They run in the normal suite.
//  2. The study itself, a measurement script, skipped unless FLUID_STUDY names a JSON file to write:
//
//         FLUID_STUDY=/tmp/fluid-study.json npx vitest run tests/fluid-study.test.ts
//
//     (docs/fluid-cost-study.md says what it measures and how to read it).

// --- Helpers -------------------------------------------------------------------------------------

async function readHalf(device: GPUDevice, texture: GPUTexture): Promise<Float32Array> {
  const bytes = await readTexture(device, texture, 8);
  const half = new Uint16Array(bytes.buffer, bytes.byteOffset, texture.width * texture.height * 4);
  const out = new Float32Array(half.length);
  for (let i = 0; i < half.length; i++) out[i] = halfToFloat(half[i]!);
  return out;
}

async function readFloats(device: GPUDevice, texture: GPUTexture): Promise<Float32Array> {
  const bytes = await readTexture(device, texture, 4);
  return new Float32Array(bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.length));
}

interface Velocity {
  width: number;
  height: number;
  u(i: number, j: number): number;
  v(i: number, j: number): number;
}

async function readVelocity(device: GPUDevice, fluid: Fluid): Promise<Velocity> {
  const { width, height, velocityWidth } = fluid.grid;
  const data = await readHalf(device, fluid.velocity.texture);
  return {
    width,
    height,
    u: (i, j) => data[(j * velocityWidth + i) * 4]!,
    v: (i, j) => data[(j * velocityWidth + i) * 4 + 1]!,
  };
}

/** Root-mean-square divergence over the cells of a velocity readback. */
function rmsDivergence(v: Velocity): number {
  let sum = 0;
  for (let j = 0; j < v.height; j++) {
    for (let i = 0; i < v.width; i++) {
      const d = v.u(i + 1, j) - v.u(i, j) + v.v(i, j + 1) - v.v(i, j);
      sum += d * d;
    }
  }
  return Math.sqrt(sum / (v.width * v.height));
}

async function submit(device: GPUDevice, record: (encoder: GPUCommandEncoder) => void) {
  const encoder = device.createCommandEncoder();
  record(encoder);
  device.queue.submit([encoder.finish()]);
  await device.queue.onSubmittedWorkDone();
}

function makeFluid(
  device: GPUDevice,
  params: Partial<FluidParams> = {},
  resolution = 64,
  dyeResolution = 64,
  aspect = 1,
): Fluid {
  return createFluid({
    device,
    resolution,
    dyeResolution,
    width: aspect,
    height: 1,
    params: { velocityDissipation: 0, dyeDissipation: 0, ...params },
  });
}

/** A smooth, strongly divergent field: a Gaussian source. Resolution independent. */
const largeSource = (x: number, y: number): [number, number] => {
  const dx = x - 0.4;
  const dy = y - 0.55;
  const g = Math.exp(-(dx * dx + dy * dy) / 0.02);
  return [dx * g * 4, dy * g * 4];
};

/** The same with a short length scale (sigma 0.03). */
const smallSource = (x: number, y: number): [number, number] => {
  const dx = x - 0.5;
  const dy = y - 0.5;
  const g = Math.exp(-(dx * dx + dy * dy) / 0.0018);
  return [dx * g * 12, dy * g * 12];
};

/**
 * Projects `source` once from a zero pressure and returns the rms divergence afterwards divided by the
 * rms divergence before. `count` is iterations, or V-cycles for the multigrid solver.
 */
async function coldResidual(
  device: GPUDevice,
  fluid: Fluid,
  source: (x: number, y: number) => readonly [number, number],
  count: number,
): Promise<number> {
  fluid.params.warmStart = false;
  fluid.setVelocity(source);
  const before = rmsDivergence(await readVelocity(device, fluid));
  await submit(device, (e) => fluid.project(e, count));
  return rmsDivergence(await readVelocity(device, fluid)) / before;
}

// --- The cubic velocity harness -------------------------------------------------------------------

// Evaluates the solver's own velocity reconstructions (fluid-core.wgsl) at chosen points of a real
// velocity texture, so the numbers below come from the code the dye pass runs.
const HARNESS = `
#include "fluid-core.wgsl"
@group(0) @binding(2) var vtex: texture_2d<f32>;
@group(0) @binding(3) var vsamp: sampler;
@group(0) @binding(4) var<storage, read> pts: array<vec4f>;
@group(0) @binding(5) var<storage, read_write> outv: array<vec4f>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3u) {
  if (id.x >= arrayLength(&pts)) { return; }
  let p = pts[id.x].xy;
  let cells = vec2f(textureDimensions(vtex)) - vec2f(1.0);
  outv[id.x * 4u] = vec4f(vel_at(vtex, p), 0.0, 0.0);
  outv[id.x * 4u + 1u] = vec4f(vel_at_cubic(vtex, vsamp, p, cells, 1u), 0.0, 0.0);
  outv[id.x * 4u + 2u] = vec4f(vel_at_cubic(vtex, vsamp, p, cells, 2u), 0.0, 0.0);
  outv[id.x * 4u + 3u] = vec4f(vel_at_cubic(vtex, vsamp, p, cells, 3u), 0.0, 0.0);
}`;

type InterpolationKind = 'bilinear' | 'bspline' | 'catmull-rom' | 'hardware-bilinear';
type Interpolated = Record<InterpolationKind, [number, number][]>;

/** Velocity (image heights per second) at points p (cells) of a fluid's velocity texture, by each filter. */
async function interpolate(
  device: GPUDevice,
  fluid: Fluid,
  points: readonly (readonly [number, number])[],
): Promise<Interpolated> {
  const module = createShader(device, {
    label: 'fluid-study harness',
    code: HARNESS,
    chunks: fluidShaderChunks,
  });
  const pipeline = device.createComputePipeline({
    layout: 'auto',
    compute: { module, entryPoint: 'main' },
  });
  const uniforms = createUniformBuffer(device, FluidUniforms, 'harness params');
  uniforms.set({ grid: [fluid.grid.width, fluid.grid.height], boundary: 0, solid: 0 });
  uniforms.upload();
  const mask = device.createTexture({
    size: [1, 1],
    format: 'r8unorm',
    usage: GPUTextureUsage.TEXTURE_BINDING,
  });
  const sampler = device.createSampler({ magFilter: 'linear', minFilter: 'linear' });
  const input = new Float32Array(points.length * 4);
  points.forEach((p, i) => input.set([p[0], p[1], 0, 0], i * 4));
  const inputBuffer = device.createBuffer({
    size: input.byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(inputBuffer, 0, input);
  const outBytes = points.length * 4 * 16;
  const outBuffer = device.createBuffer({
    size: outBytes,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
  });
  const readBuffer = device.createBuffer({
    size: outBytes,
    usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
  });
  const group = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: uniforms.buffer } },
      { binding: 1, resource: mask.createView() },
      { binding: 2, resource: fluid.velocity.view },
      { binding: 3, resource: sampler },
      { binding: 4, resource: { buffer: inputBuffer } },
      { binding: 5, resource: { buffer: outBuffer } },
    ],
  });
  await submit(device, (encoder) => {
    const pass = encoder.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, group);
    pass.dispatchWorkgroups(Math.ceil(points.length / 64));
    pass.end();
    encoder.copyBufferToBuffer(outBuffer, 0, readBuffer, 0, outBytes);
  });
  await readBuffer.mapAsync(GPUMapMode.READ);
  const data = new Float32Array(readBuffer.getMappedRange().slice(0));
  readBuffer.unmap();
  for (const buffer of [inputBuffer, outBuffer, readBuffer]) buffer.destroy();
  mask.destroy();
  uniforms.destroy();
  const pick = (k: number) =>
    points.map((_, i): [number, number] => [data[(i * 4 + k) * 4]!, data[(i * 4 + k) * 4 + 1]!]);
  return {
    bilinear: pick(0),
    bspline: pick(1),
    'catmull-rom': pick(2),
    'hardware-bilinear': pick(3),
  };
}

// --- Multigrid pressure solve ---------------------------------------------------------------------

describe('multigrid pressure solve', () => {
  it('builds a pyramid down to one workgroup, any size, with an optional cap', () => {
    expect(multigridLevelSizes(128, 128)).toEqual([
      { width: 128, height: 128 },
      { width: 64, height: 64 },
      { width: 32, height: 32 },
    ]);
    const odd = multigridLevelSizes(100, 75);
    expect(odd.map((s) => `${s.width}x${s.height}`)).toEqual(['100x75', '50x38', '25x19']);
    expect(Math.max(...Object.values(odd.at(-1)!))).toBeLessThanOrEqual(MULTIGRID_COARSEST);
    expect(multigridLevelSizes(512, 512, 3)).toHaveLength(3);
    // The cap never leaves a single-level "pyramid" on a grid that needs one, and a small grid is one level.
    expect(multigridLevelSizes(512, 512, 1)).toHaveLength(2);
    expect(multigridLevelSizes(32, 32)).toHaveLength(1);
    expect(multigridLevelSizes(33, 33)).toHaveLength(2);
  });

  it('counts the work of a solve in dispatches and cell visits', () => {
    expect(pressureSolveWork(128, 128, 'jacobi', 30)).toEqual({
      dispatches: 30,
      cellVisits: 30 * 128 * 128,
    });
    const v = pressureSolveWork(128, 128, 'multigrid', 1, { multigridSmooth: 2 });
    // two levels of (2 + 2 smoothing + restrict + prolong) and one coarse solve
    expect(v.dispatches).toBe(2 * 6 + 1);
    expect(v.cellVisits).toBe(6 * (128 * 128 + 64 * 64) + 32 * 32);
    expect(pressureSolveWork(128, 128, 'multigrid', 2, { multigridSmooth: 2 }).cellVisits).toBe(
      2 * v.cellVisits,
    );
    // One sweep each way is the default.
    expect(pressureSolveWork(128, 128, 'multigrid', 1).dispatches).toBe(2 * 4 + 1);
  });

  it('defaults to Jacobi for pressure and the hardware-filtered bilinear velocity for dye', () => {
    expect(DEFAULT_FLUID_PARAMS.pressureSolver).toBe('jacobi');
    // The study's first recommendation, applied: same function as the loads, a fifth of the cost.
    expect(DEFAULT_FLUID_PARAMS.dyeVelocityFilter).toBe('hardware-bilinear');
    expect(DEFAULT_FLUID_PARAMS.pressureIterations).toBe(30);
  });

  it.each([64, 128])(
    'removes divergence far faster than Jacobi for the same work, at %i cells',
    async (size) => {
      const device = await getTestDevice();
      const multigrid = makeFluid(device, { pressureSolver: 'multigrid' }, size);
      const cycles = 2;
      const work = pressureSolveWork(size, size, 'multigrid', cycles).cellVisits;
      const mg = await coldResidual(device, multigrid, largeSource, cycles);
      multigrid.destroy();
      // The same number of cell visits spent on damped Jacobi.
      const jacobi = makeFluid(device, { pressureSolver: 'jacobi' }, size);
      const sweeps = Math.ceil(work / (size * size));
      const j = await coldResidual(device, jacobi, largeSource, sweeps);
      jacobi.destroy();
      expect(mg).toBeLessThan(0.03);
      expect(j).toBeGreaterThan(0.6);
      expect(mg * 20).toBeLessThan(j);
    },
  );

  it('converges to the same pressure field as a converged relaxation', async () => {
    const device = await getTestDevice();
    const read = async (params: Partial<FluidParams>, count: number) => {
      const fluid = makeFluid(device, { warmStart: false, ...params }, 64);
      fluid.setVelocity(largeSource);
      await submit(device, (e) => fluid.project(e, count));
      const field = await readFloats(device, fluid.pressure.texture);
      fluid.destroy();
      // A pressure is defined up to a constant.
      const mean = field.reduce((a, b) => a + b, 0) / field.length;
      return field.map((x) => x - mean);
    };
    const reference = await read({ pressureSolver: 'sor' }, 1500);
    const mg = await read({ pressureSolver: 'multigrid' }, 6);
    let diff = 0;
    let norm = 0;
    for (let i = 0; i < reference.length; i++) {
      diff += (mg[i]! - reference[i]!) ** 2;
      norm += reference[i]! ** 2;
    }
    expect(norm).toBeGreaterThan(0);
    expect(Math.sqrt(diff / norm)).toBeLessThan(0.02);
  });

  it('leaves a divergence-free field alone and a fluid at rest at rest', async () => {
    const device = await getTestDevice();
    const fluid = makeFluid(device, { pressureSolver: 'multigrid' }, 64);
    await submit(device, (e) => fluid.advance(e, 5));
    const rest = await readVelocity(device, fluid);
    let max = 0;
    for (let j = 0; j < 64; j++)
      for (let i = 0; i <= 64; i++) max = Math.max(max, Math.abs(rest.u(i, j)));
    expect(max).toBe(0);
    fluid.destroy();
  });

  const cases: [string, Partial<FluidParams>, number, number, boolean][] = [
    ['an odd-sized grid', {}, 100, 4 / 3, false],
    ['a 16:9 grid', {}, 192, 16 / 9, false],
    ['an open boundary', { boundary: 'open' }, 96, 1, false],
    ['a periodic boundary', { boundary: 'wrap' }, 96, 1, false],
    ['obstacles', {}, 128, 1, true],
    ['obstacles and an open boundary', { boundary: 'open' }, 128, 1, true],
  ];
  it.each(cases)('converges with %s', async (_name, params, size, aspect, obstacles) => {
    const device = await getTestDevice();
    const fluid = makeFluid(device, { pressureSolver: 'multigrid', ...params }, size, 32, aspect);
    if (obstacles) {
      fluid.obstacles.setRects([
        { x: 0.45, y: 0.3, width: 0.12, height: 0.4 },
        { x: 0.1, y: 0.1, width: 0.05, height: 0.05 },
      ]);
    }
    const mg = await coldResidual(device, fluid, largeSource, 3);
    fluid.destroy();
    const reference = makeFluid(device, { pressureSolver: 'jacobi', ...params }, size, 32, aspect);
    if (obstacles) reference.obstacles.setRects([{ x: 0.45, y: 0.3, width: 0.12, height: 0.4 }]);
    const j = await coldResidual(device, reference, largeSource, 30);
    reference.destroy();
    expect(Number.isFinite(mg)).toBe(true);
    expect(mg).toBeLessThan(0.1);
    expect(mg).toBeLessThan(j);
  });

  it('does better than 30 Jacobi iterations inside a running simulation, with one cycle', async () => {
    const device = await getTestDevice();
    const ratio = async (params: Partial<FluidParams>) => {
      const fluid = makeFluid(device, params, 128);
      const ratios: number[] = [];
      let t = 0;
      for (let step = 0; step < 60; step++) {
        const a = 0.5 + 0.3 * Math.cos(t * 3);
        const b = 0.5 + 0.3 * Math.sin(t * 3);
        t += 1 / 60;
        const a2 = 0.5 + 0.3 * Math.cos(t * 3);
        const b2 = 0.5 + 0.3 * Math.sin(t * 3);
        fluid.addForce([a, b], [a2, b2], 0.05, [(a2 - a) * 120, (b2 - b) * 120]);
        await submit(device, (e) => fluid.advance(e, 1));
        if (step >= 20) {
          const d = await readFloats(device, fluid.divergence);
          let sum = 0;
          for (const x of d) sum += x * x;
          const before = Math.sqrt(sum / d.length);
          ratios.push(rmsDivergence(await readVelocity(device, fluid)) / Math.max(before, 1e-12));
        }
      }
      fluid.destroy();
      return ratios.reduce((a, b) => a + b, 0) / ratios.length;
    };
    const jacobi = await ratio({ pressureSolver: 'jacobi', pressureIterations: 30 });
    const mg = await ratio({ pressureSolver: 'multigrid', multigridCycles: 1 });
    expect(mg).toBeLessThan(0.3);
    expect(mg * 2).toBeLessThan(jacobi);
  });

  it('runs every cycle, smoothing, level and boundary setting without a validation error', async () => {
    const device = await getTestDevice();
    await expectValid(device, async () => {
      for (const params of [
        { multigridCycles: 3, multigridSmooth: 1 },
        { multigridSmooth: 4, multigridLevels: 2 },
        { multigridLevels: 3, multigridCoarse: 8 },
        { multigridLevels: 3, boundary: 'wrap' },
        { dyeScheme: 'bfecc', velocityScheme: 'bfecc', vorticity: 8, viscosity: 0.001 },
      ] as Partial<FluidParams>[]) {
        const fluid = makeFluid(device, { pressureSolver: 'multigrid', ...params }, 64, 128);
        fluid.addForce([0.3, 0.5], [0.7, 0.5], 0.05, [1, 0]);
        fluid.addDye([0.3, 0.5], [0.7, 0.5], 0.05, [1, 1, 1, 1]);
        await submit(device, (e) => fluid.advance(e, 6));
        const v = await readVelocity(device, fluid);
        expect(Number.isFinite(rmsDivergence(v))).toBe(true);
        fluid.destroy();
      }
    });
  });
});

// --- Cubic reconstruction of the velocity for the dye ---------------------------------------------------

describe('cubic velocity reconstruction for the dye', () => {
  const SIZE = 32;

  it('reproduces a linear velocity field exactly, with every filter', async () => {
    const device = await getTestDevice();
    const fluid = makeFluid(device, {}, SIZE);
    const field = (x: number, y: number): [number, number] => [
      0.3 + 0.8 * x - 0.5 * y,
      -0.2 + 0.4 * x + 0.9 * y,
    ];
    fluid.setVelocity(field);
    const points: [number, number][] = [];
    for (let k = 0; k < 200; k++) {
      points.push([
        3 + (SIZE - 6) * ((k * 0.618034) % 1),
        3 + (SIZE - 6) * ((k * 0.381966 + 0.1) % 1),
      ]);
    }
    const result = await interpolate(device, fluid, points);
    for (const kind of ['bilinear', 'hardware-bilinear', 'bspline', 'catmull-rom'] as const) {
      let worst = 0;
      points.forEach((p, i) => {
        const [u, v] = field(p[0] / SIZE, p[1] / SIZE);
        worst = Math.max(
          worst,
          Math.abs(result[kind][i]![0] - u),
          Math.abs(result[kind][i]![1] - v),
        );
      });
      // Half-float storage of the faces is good to about 5e-4 of the value.
      expect(worst, kind).toBeLessThan(2e-3);
    }
    fluid.destroy();
  });

  it('is continuous across cell borders, and has no kink in its slope where bilinear does', async () => {
    const device = await getTestDevice();
    const fluid = makeFluid(device, {}, SIZE);
    const field = (x: number, y: number): [number, number] => [
      0.5 * Math.sin(2 * Math.PI * x + 0.3) * Math.cos(2 * Math.PI * y),
      0.5 * Math.cos(2 * Math.PI * x) * Math.sin(2 * Math.PI * y + 0.7),
    ];
    fluid.setVelocity(field);
    // Along x at the u faces' own rows: kinks sit at integer p.x (the u faces) for u.
    const row = 10.5;
    const borders = [4, 7, 11, 15, 19, 23, 26];
    const offsets = [-0.25, -0.05, -0.001, 0.001, 0.05, 0.25];
    const points: [number, number][] = [];
    for (const b of borders) for (const o of offsets) points.push([b + o, row]);
    const result = await interpolate(device, fluid, points);
    const stats = (kind: InterpolationKind) => {
      let jump = 0;
      let kink = 0;
      borders.forEach((_, bi) => {
        const f = (oi: number) => result[kind][bi * offsets.length + oi]![0];
        jump = Math.max(jump, Math.abs(f(3) - f(2)));
        // slope just right of the border minus slope just left of it, over 0.2 cells
        kink += Math.abs((f(5) - f(4)) / 0.2 - (f(1) - f(0)) / 0.2);
      });
      return { jump, kink: kink / borders.length };
    };
    const bilinear = stats('bilinear');
    for (const kind of ['bilinear', 'hardware-bilinear', 'bspline', 'catmull-rom'] as const) {
      expect(stats(kind).jump, kind).toBeLessThan(2e-3);
    }
    // The filtered bilinear is the same function as the loaded one, kink and all.
    expect(Math.abs(stats('hardware-bilinear').kink - bilinear.kink)).toBeLessThan(
      0.3 * bilinear.kink,
    );
    // Bilinear's slope jumps by the second difference of the field at every border; the cubics do not.
    expect(bilinear.kink).toBeGreaterThan(0.004);
    expect(stats('bspline').kink * 2).toBeLessThan(bilinear.kink);
    expect(stats('catmull-rom').kink * 2).toBeLessThan(bilinear.kink);
    fluid.destroy();
  });

  it('carries dye the same distance whichever filter reads the velocity', async () => {
    const device = await getTestDevice();
    const final = async (filter: FluidParams['dyeVelocityFilter']) => {
      const fluid = makeFluid(device, { dyeOnly: true, dyeVelocityFilter: filter }, 32, 128);
      fluid.setVelocity(() => [0.25, 0]);
      fluid.setDye((x, y) => {
        const v = Math.exp(-((x - 0.3) ** 2 + (y - 0.5) ** 2) / (2 * 0.04 ** 2));
        return [v, v, v, v];
      });
      await submit(device, (e) => fluid.advance(e, 60));
      const data = await readHalf(device, fluid.dye.texture);
      let total = 0;
      let cx = 0;
      for (let y = 0; y < 128; y++) {
        for (let x = 0; x < 128; x++) {
          const v = data[(y * 128 + x) * 4]!;
          total += v;
          cx += v * (x + 0.5);
        }
      }
      fluid.destroy();
      return cx / total / 128;
    };
    // 60 steps at 0.25 image heights per second moves the blob 0.25 across the canvas.
    for (const filter of ['bilinear', 'hardware-bilinear', 'bspline', 'catmull-rom'] as const) {
      expect(await final(filter), filter).toBeCloseTo(0.55, 2);
    }
  });

  it('falls back to bilinear with obstacles and with a periodic boundary', async () => {
    const device = await getTestDevice();
    await expectValid(device, async () => {
      for (const params of [{ boundary: 'wrap' }, { boundary: 'wall' }] as Partial<FluidParams>[]) {
        const fluid = makeFluid(device, { dyeVelocityFilter: 'catmull-rom', ...params }, 32, 64);
        if (params.boundary === 'wall')
          fluid.obstacles.setRects([{ x: 0.4, y: 0.4, width: 0.2, height: 0.2 }]);
        fluid.addForce([0.2, 0.5], [0.8, 0.5], 0.05, [1, 0]);
        fluid.addDye([0.2, 0.5], [0.8, 0.5], 0.05, [1, 1, 1, 1]);
        await submit(device, (e) => fluid.advance(e, 10));
        const dye = await readHalf(device, fluid.dye.texture);
        expect(dye.every(Number.isFinite)).toBe(true);
        fluid.destroy();
      }
    });
  });
});

// --- The study -----------------------------------------------------------------------------------------------

// --- The study -----------------------------------------------------------------------------------------------
//
// Skipped unless FLUID_STUDY names the JSON file to write. A markdown rendering of every table is printed
// and written beside it (<name>.md). Optional knobs:
//   FLUID_STUDY_ONLY=pressure,warm,vortex,resolution,upsample,advection,presets   run only those sections
//   FLUID_STUDY_QUICK=1                                                           fewer repetitions
//   FLUID_STUDY_IMAGES=<dir>                                                      write contact sheets (PNG)
//   FLUID_STUDY_NOTE="..."                                                        what else was using the GPU
// Method (docs/fluid-cost-study.md): GPU time comes from timestamp queries, as the slope of the span from the
// first timed pass starting to the last ending between a small and a large number of repeats of the
// workload, one slope per repetition, reported as median with min and max. Never one pass, never a sum of
// passes. Quality numbers are residual divergence, retained vortex strength and energy, edge width of an
// advected disc, and the error of a stirred dye field against a high-resolution, converged run.

const env =
  (globalThis as { process?: { env: Record<string, string | undefined> } }).process?.env ?? {};
const studyFile = env.FLUID_STUDY;
const imagesDir = env.FLUID_STUDY_IMAGES;
const QUICK = Boolean(env.FLUID_STUDY_QUICK);
const only = (env.FLUID_STUDY_ONLY ?? '').split(',').filter(Boolean);
const wants = (name: string) => only.length === 0 || only.includes(name);
const REPS = QUICK ? 3 : 5;
/** How many times each cost table is measured end to end, minutes apart, and pooled (bursts of other work on the GPU last seconds). */
const PASSES = Number(env.FLUID_STUDY_PASSES ?? (QUICK ? 1 : 2));
/** Aim for a workload at least this long (ms) at the larger repeat count, so jitter is small against it. */
const SPAN_MS = QUICK ? 30 : 60;
/** Switch the (slow) cost measurements off to iterate on the quality ones. */
const COST = !env.FLUID_STUDY_NOCOST;

interface Stat {
  /** Median of the per-repetition slopes. */
  median: number;
  min: number;
  max: number;
  n: number;
  /**
   * The slope between the fastest span seen at each end. Other work on the GPU only ever adds time, so on a
   * busy machine this is nearer the truth than the median; on a quiet one they agree.
   */
  best?: number;
  /** The per-repetition slopes behind `median`. */
  samples?: number[];
}

/** The two sets of repetitions as one: pooled slopes, and the better of the two fastest-span slopes. */
function mergeStats(a: Stat, b: Stat): Stat {
  const samples = [...(a.samples ?? [a.median]), ...(b.samples ?? [b.median])];
  return { ...stat(samples), best: Math.min(a.best ?? a.median, b.best ?? b.median), samples };
}

function stat(samples: number[]): Stat {
  const sorted = [...samples].sort((a, b) => a - b);
  const mid = sorted.length >> 1;
  const median = sorted.length % 2 ? sorted[mid]! : (sorted[mid - 1]! + sorted[mid]!) / 2;
  return { median, min: sorted[0]!, max: sorted[sorted.length - 1]!, n: sorted.length };
}

const fix = (x: number, digits = 3) => (Number.isFinite(x) ? x.toFixed(digits) : 'n/a');
const ms = (s: Stat, digits = 3) =>
  `${fix(s.median, digits)} (${fix(s.min, digits)}-${fix(s.max, digits)})`;

function table(headers: string[], rows: (string | number)[][]): string {
  const line = (cells: (string | number)[]) => `| ${cells.join(' | ')} |`;
  return [line(headers), line(headers.map(() => '---')), ...rows.map(line)].join('\n');
}

/** GPU time of a recorded workload: first timed pass starting to last timed pass ending. */
function createTimer(device: GPUDevice) {
  const COUNT = 64;
  const querySet = device.createQuerySet({ type: 'timestamp', count: COUNT });
  const resolve = device.createBuffer({
    size: COUNT * 8,
    usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC,
  });
  const read = device.createBuffer({
    size: COUNT * 8,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });
  let next = 0;
  const profiler = {
    pass: (): GPURenderPassTimestampWrites | undefined => {
      if (next + 2 > COUNT) return undefined;
      const begin = next;
      next += 2;
      return { querySet, beginningOfPassWriteIndex: begin, endOfPassWriteIndex: begin + 1 };
    },
  };
  async function time(
    record: (encoder: GPUCommandEncoder) => void | (() => void),
  ): Promise<number> {
    next = 0;
    const encoder = device.createCommandEncoder();
    const cleanup = record(encoder);
    const used = next;
    encoder.resolveQuerySet(querySet, 0, COUNT, resolve, 0);
    encoder.copyBufferToBuffer(resolve, 0, read, 0, COUNT * 8);
    device.queue.submit([encoder.finish()]);
    await read.mapAsync(GPUMapMode.READ);
    const stamps = new BigUint64Array(read.getMappedRange().slice(0));
    read.unmap();
    cleanup?.();
    const valid = Array.from(stamps.slice(0, used)).filter((t) => t > 0n);
    if (valid.length < 2) return Number.NaN;
    const first = valid.reduce((a, b) => (a < b ? a : b));
    const last = valid.reduce((a, b) => (a > b ? a : b));
    return Number(last - first) / 1e6;
  }
  /** Milliseconds per repeat: one slope per repetition between `low` and `high` repeats. */
  async function slope(
    record: (n: number) => (encoder: GPUCommandEncoder) => void | (() => void),
    low: number,
    highGuess: number,
    reps = REPS,
    maxHigh = 640,
  ): Promise<Stat> {
    // A workload that is cheap per repeat needs many repeats before the slope stands out of the jitter of a
    // submit (several milliseconds on a busy machine): calibrate the larger count to span about SPAN_MS.
    await time(record(low));
    const rough = (await time(record(highGuess))) - (await time(record(low)));
    const perRepeat = Math.max(rough / (highGuess - low), 1e-4);
    const high = Math.round(Math.min(maxHigh, Math.max(highGuess, SPAN_MS / perRepeat)));
    const slopes: number[] = [];
    const lows: number[] = [];
    const highs: number[] = [];
    for (let i = 0; i < reps; i++) {
      const a = await time(record(low));
      const b = await time(record(high));
      lows.push(a);
      highs.push(b);
      slopes.push((b - a) / (high - low));
    }
    return {
      ...stat(slopes),
      best: (Math.min(...highs) - Math.min(...lows)) / (high - low),
      samples: slopes,
    };
  }
  return {
    profiler,
    time,
    slope,
    destroy: () => [querySet, resolve, read].forEach((x) => x.destroy()),
  };
}

// --- Images (optional) ----------------------------------------------------------------------------------

async function writePng(path: string, width: number, height: number, rgb: Uint8Array) {
  const zlibName = 'node:zlib';
  const fsName = 'node:fs';
  const { deflateSync } = (await import(/* @vite-ignore */ zlibName)) as {
    deflateSync(data: Uint8Array): Uint8Array;
  };
  const { writeFileSync, mkdirSync } = (await import(/* @vite-ignore */ fsName)) as {
    writeFileSync(path: string, data: Uint8Array): void;
    mkdirSync(path: string, options: { recursive: boolean }): void;
  };
  mkdirSync(path.slice(0, path.lastIndexOf('/')), { recursive: true });
  const table = new Uint32Array(256).map((_, n) => {
    let c = n;
    for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
    return c >>> 0;
  });
  const crc = (bytes: Uint8Array) => {
    let c = 0xffffffff;
    for (const b of bytes) c = table[(c ^ b) & 0xff]! ^ (c >>> 8);
    return (c ^ 0xffffffff) >>> 0;
  };
  const chunk = (type: string, data: Uint8Array) => {
    const out = new Uint8Array(12 + data.length);
    const view = new DataView(out.buffer);
    view.setUint32(0, data.length);
    for (let i = 0; i < 4; i++) out[4 + i] = type.charCodeAt(i);
    out.set(data, 8);
    view.setUint32(8 + data.length, crc(out.subarray(4, 8 + data.length)));
    return out;
  };
  const header = new Uint8Array(13);
  const hv = new DataView(header.buffer);
  hv.setUint32(0, width);
  hv.setUint32(4, height);
  header.set([8, 2, 0, 0, 0], 8);
  const raw = new Uint8Array((width * 3 + 1) * height);
  for (let y = 0; y < height; y++)
    raw.set(rgb.subarray(y * width * 3, (y + 1) * width * 3), y * (width * 3 + 1) + 1);
  const parts = [
    new Uint8Array([137, 80, 78, 71, 13, 10, 26, 10]),
    chunk('IHDR', header),
    chunk('IDAT', deflateSync(raw)),
    chunk('IEND', new Uint8Array(0)),
  ];
  const all = new Uint8Array(parts.reduce((n, p) => n + p.length, 0));
  let at = 0;
  for (const p of parts) {
    all.set(p, at);
    at += p.length;
  }
  writeFileSync(path, all);
}

/** Resamples a square scalar field to `size` x `size` (box filter when shrinking by an integer, bilinear otherwise). */
function resample(field: Float32Array, from: number, size: number): Float32Array {
  const out = new Float32Array(size * size);
  const ratio = from / size;
  const box = Number.isInteger(ratio) ? ratio : 1;
  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      if (box > 1) {
        let sum = 0;
        for (let j = 0; j < box; j++)
          for (let i = 0; i < box; i++) sum += field[(y * box + j) * from + x * box + i]!;
        out[y * size + x] = sum / (box * box);
      } else {
        const fx = Math.min(from - 1.001, Math.max(0, (x + 0.5) * ratio - 0.5));
        const fy = Math.min(from - 1.001, Math.max(0, (y + 0.5) * ratio - 0.5));
        const x0 = Math.floor(fx);
        const y0 = Math.floor(fy);
        const tx = fx - x0;
        const ty = fy - y0;
        const at = (i: number, j: number) => field[(y0 + j) * from + x0 + i]!;
        out[y * size + x] =
          (1 - ty) * ((1 - tx) * at(0, 0) + tx * at(1, 0)) +
          ty * ((1 - tx) * at(0, 1) + tx * at(1, 1));
      }
    }
  }
  return out;
}

/** Ink on paper: 0 is white, 1 is black. */
async function writeSheet(
  name: string,
  panels: { field: Float32Array; size: number }[],
  cell = 320,
) {
  if (!imagesDir || panels.length === 0) return;
  const width = cell * panels.length + 4 * (panels.length - 1);
  const rgb = new Uint8Array(width * cell * 3).fill(200);
  panels.forEach((panel, index) => {
    const image = resample(panel.field, panel.size, cell);
    for (let y = 0; y < cell; y++) {
      for (let x = 0; x < cell; x++) {
        const v = Math.round(255 * (1 - Math.min(1, Math.max(0, image[y * cell + x]!))));
        const o = (y * width + index * (cell + 4) + x) * 3;
        rgb[o] = rgb[o + 1] = rgb[o + 2] = v;
      }
    }
  });
  const dir = `${imagesDir}/${name}.png`;
  await writePng(dir, width, cell, rgb);
}

// --- The scenario: a stirred pattern in a closed box -------------------------------------------------------------

const smoothstep = (a: number, b: number, x: number) => {
  const t = Math.min(1, Math.max(0, (x - a) / (b - a)));
  return t * t * (3 - 2 * t);
};

/** Stripes (a smooth, resolvable signal), a disc and a square (hard edges). Values 0 to 1. */
function pattern(x: number, y: number): number {
  const window =
    smoothstep(0.04, 0.06, x) *
    (1 - smoothstep(0.44, 0.46, x)) *
    smoothstep(0.18, 0.2, y) *
    (1 - smoothstep(0.8, 0.82, y));
  const stripes = (0.5 + 0.5 * Math.sin(2 * Math.PI * 6 * x)) * window;
  const disc = Math.hypot(x - 0.72, y - 0.68) < 0.12 ? 1 : 0;
  const square = x > 0.58 && x < 0.7 && y > 0.2 && y < 0.32 ? 1 : 0;
  return Math.max(stripes, disc, square);
}

/** Grid on which every scenario result is compared. */
const COMPARE = 256;
const SCENARIO_STEPS = 120;
/** Speed of the stirring strokes, image heights per second. */
const STIR = Number(env.FLUID_STUDY_STIR ?? 0.45);
/** Coarse grid for the large-scale error. */
const COARSE = 128;

interface ScenarioResult {
  /** The dye (channel 0) box-filtered to COMPARE x COMPARE. */
  field: Float32Array;
  /** Total dye at the end over total dye at the start. */
  mass: number;
  /** Mean of u^2 + v^2 over the faces at the end, (image heights per second)^2. */
  energy: number;
  /** Rms divergence at the end as a rate: area change per second of a fluid parcel, 1 / s. */
  divergence: number;
}

interface ScenarioConfig {
  velocity: number;
  dye: number;
  params?: Partial<FluidParams>;
}

/**
 * Two scripted strokes (a full circle round the middle, then a cross-stroke) stir the pattern for 1 s, the
 * flow then coasts for 1 s. Everything is in canvas units, so any grid gets the same push.
 */
async function runScenario(device: GPUDevice, config: ScenarioConfig): Promise<ScenarioResult> {
  const fluid = createFluid({
    device,
    resolution: config.velocity,
    dyeResolution: config.dye,
    params: {
      velocityDissipation: 0.5,
      dyeDissipation: 0,
      pressureSolver: 'multigrid',
      ...config.params,
    },
  });
  fluid.setDye((x, y) => {
    const v = pattern(x, y);
    return [v, v, v, v];
  });
  const sum = async () => {
    const data = await readHalf(device, fluid.dye.texture);
    let total = 0;
    for (let i = 0; i < data.length; i += 4) total += data[i]!;
    return { data, total };
  };
  const start = await sum();
  for (let step = 0; step < SCENARIO_STEPS; step++) {
    if (step < 60) {
      const a0 = (2 * Math.PI * step) / 60;
      const a1 = (2 * Math.PI * (step + 1)) / 60;
      const am = (a0 + a1) / 2;
      fluid.addForce(
        [0.5 + 0.25 * Math.cos(a0), 0.5 + 0.25 * Math.sin(a0)],
        [0.5 + 0.25 * Math.cos(a1), 0.5 + 0.25 * Math.sin(a1)],
        0.05,
        [-Math.sin(am) * STIR, Math.cos(am) * STIR],
        0.025,
      );
    }
    if (step >= 15 && step < 45) {
      const x0 = 0.2 + (0.6 * (step - 15)) / 30;
      const x1 = 0.2 + (0.6 * (step - 14)) / 30;
      fluid.addForce([x0, 0.5], [x1, 0.5], 0.05, [STIR, 0], 0.025);
    }
    const encoder = device.createCommandEncoder();
    fluid.advance(encoder, 1);
    device.queue.submit([encoder.finish()]);
  }
  await device.queue.onSubmittedWorkDone();
  const end = await sum();
  const velocity = await readVelocity(device, fluid);
  let energy = 0;
  let faces = 0;
  for (let j = 0; j < velocity.height; j++) {
    for (let i = 0; i <= velocity.width; i++) {
      energy += velocity.u(i, j) ** 2;
      faces++;
    }
  }
  for (let j = 0; j <= velocity.height; j++) {
    for (let i = 0; i < velocity.width; i++) {
      energy += velocity.v(i, j) ** 2;
      faces++;
    }
  }
  const dyeSize = fluid.grid.dyeWidth;
  const channel = new Float32Array(dyeSize * dyeSize);
  for (let i = 0; i < channel.length; i++) channel[i] = end.data[i * 4]!;
  const result: ScenarioResult = {
    field: resample(channel, dyeSize, COMPARE),
    mass: end.total / start.total,
    energy: energy / faces,
    divergence: rmsDivergence(velocity) * fluid.grid.cellSize ** -1,
  };
  fluid.destroy();
  return result;
}

const coarsen = (field: Float32Array) => resample(field, COMPARE, COARSE);

/** Large-scale (128²) and fine (256²) relative L2 error of a run against the reference. */
const errorsOf = (run: { field: Float32Array }, ref: { field: Float32Array }) => ({
  coarse: relativeError(coarsen(run.field), coarsen(ref.field)),
  fine: relativeError(run.field, ref.field),
});

/** Relative L2 difference of two fields. */
function relativeError(a: Float32Array, reference: Float32Array): number {
  let diff = 0;
  let norm = 0;
  for (let i = 0; i < a.length; i++) {
    diff += (a[i]! - reference[i]!) ** 2;
    norm += reference[i]! ** 2;
  }
  return Math.sqrt(diff / norm);
}

/** Mean squared gradient of a square field, a measure of how much fine detail it holds. */
function detail(field: Float32Array, size: number): number {
  let sum = 0;
  for (let y = 0; y < size - 1; y++) {
    for (let x = 0; x < size - 1; x++) {
      const gx = field[y * size + x + 1]! - field[y * size + x]!;
      const gy = field[(y + 1) * size + x]! - field[y * size + x]!;
      sum += gx * gx + gy * gy;
    }
  }
  return sum / (size * size);
}

// The best run the solver can make of the scenario, and a second, different one: how far two good runs differ
// is the floor under every error below.
const REFERENCE: ScenarioConfig = {
  velocity: 256,
  dye: 1024,
  params: { pressureSolver: 'multigrid', multigridCycles: 3 },
};
const REFERENCE_2: ScenarioConfig = {
  velocity: 192,
  dye: 768,
  params: { pressureSolver: 'multigrid', multigridCycles: 3 },
};

describe.skipIf(!studyFile)('fluid cost study', () => {
  const results: Record<string, unknown> = {};
  const markdown: string[] = [];
  let device: GPUDevice;
  let timer: ReturnType<typeof createTimer>;
  let timed = false;
  let reference: ScenarioResult | undefined;
  const sheets: Record<string, { field: Float32Array; size: number }[]> = {};

  const note = (text = '') => markdown.push(text);
  const section = (title: string) => {
    note();
    note(`### ${title}`);
    note();
  };

  beforeAll(async () => {
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) throw new Error('No WebGPU adapter');
    timed = adapter.features.has('timestamp-query');
    device = await adapter.requestDevice({
      requiredFeatures: timed ? (['timestamp-query'] as GPUFeatureName[]) : [],
    });
    timer = createTimer(device);
    const osName = 'node:os';
    const os = (await import(/* @vite-ignore */ osName)) as {
      loadavg(): number[];
      cpus(): { model: string }[];
      platform(): string;
    };
    const info = adapter.info;
    results.meta = {
      date: new Date().toISOString(),
      adapter: {
        vendor: info?.vendor,
        architecture: info?.architecture,
        device: info?.device,
        description: info?.description,
      },
      timestampQueries: timed,
      repetitions: REPS,
      quick: QUICK,
      loadAverageAtStart: os.loadavg(),
      cpu: os.cpus()[0]?.model,
      platform: os.platform(),
      note: env.FLUID_STUDY_NOTE ?? '',
    };
    note(
      `Run ${new Date().toISOString()}; timestamp queries ${timed}; ${REPS} repetitions per timing (median, min-max).`,
    );
    if (env.FLUID_STUDY_NOTE) note(`Machine state: ${env.FLUID_STUDY_NOTE}`);
  });

  afterAll(async () => {
    timer?.destroy();
    const fsName = 'node:fs';
    const { writeFileSync } = (await import(/* @vite-ignore */ fsName)) as {
      writeFileSync(path: string, text: string): void;
    };
    writeFileSync(studyFile!, JSON.stringify(results, null, 2) + '\n');
    writeFileSync(studyFile!.replace(/\.json$/, '') + '.md', markdown.join('\n') + '\n');
    console.log(markdown.join('\n'));
  });

  async function getReference() {
    reference ??= await runScenario(device, REFERENCE);
    return reference;
  }

  /** Measures every key once per pass, passes in turn, and pools the passes: a burst of other work only spoils one. */
  async function sweep(
    keys: readonly string[],
    measure: (key: string) => Promise<Stat>,
  ): Promise<Record<string, Stat>> {
    const out: Record<string, Stat> = {};
    for (let pass = 0; pass < PASSES; pass++) {
      for (const key of keys) {
        const next = await measure(key);
        out[key] = out[key] ? mergeStats(out[key], next) : next;
      }
    }
    return out;
  }

  /** Milliseconds per simulation step of a configuration, by the slope method. */
  function stepCost(config: ScenarioConfig, extra: Partial<FluidParams> = {}, reps = REPS) {
    return timer.slope(
      (n) => {
        const fluid = createFluid({
          device,
          resolution: config.velocity,
          dyeResolution: config.dye,
          profiler: timer.profiler,
          params: { ...config.params, ...extra },
        });
        fluid.addForce([0.3, 0.5], [0.7, 0.5], 0.05, [1, 0]);
        fluid.addDye([0.3, 0.5], [0.7, 0.5], 0.05, [1, 1, 1, 1]);
        return (encoder) => {
          fluid.advance(encoder, n);
          return () => fluid.destroy();
        };
      },
      4,
      28,
      reps,
    );
  }

  // 1. Pressure: residual against iterations, cost per iteration, cost to reduce ---------------------------------------

  const SOLVERS: { name: string; params: Partial<FluidParams>; unit: 'iterations' | 'cycles' }[] = [
    { name: 'jacobi', params: { pressureSolver: 'jacobi' }, unit: 'iterations' },
    { name: 'sor', params: { pressureSolver: 'sor' }, unit: 'iterations' },
    {
      name: 'multigrid V(1,1)',
      params: { pressureSolver: 'multigrid', multigridSmooth: 1 },
      unit: 'cycles',
    },
    {
      name: 'multigrid V(2,2)',
      params: { pressureSolver: 'multigrid', multigridSmooth: 2 },
      unit: 'cycles',
    },
    {
      name: 'multigrid V(1,1), 3 levels',
      params: { pressureSolver: 'multigrid', multigridSmooth: 1, multigridLevels: 3 },
      unit: 'cycles',
    },
  ];
  const SIZES = [64, 128, 256, 512];

  /** The count at which a residual curve first falls to `target`, log-log interpolated; null if it never does. */
  function reach(counts: number[], ratios: number[], target: number): number | null {
    for (let i = 0; i < ratios.length; i++) {
      if (ratios[i]! > target) continue;
      if (i === 0) return counts[0]!;
      const a = Math.log(ratios[i - 1]!);
      const b = Math.log(Math.max(ratios[i]!, 1e-9));
      const t = (Math.log(target) - a) / (b - a);
      return Math.exp(
        Math.log(counts[i - 1]!) + t * (Math.log(counts[i]!) - Math.log(counts[i - 1]!)),
      );
    }
    return null;
  }

  it.runIf(wants('pressure'))('pressure solvers: residual, cost and cost to reduce', async () => {
    const sources = { large: largeSource, small: smallSource };
    const curves: Record<string, unknown> = {};
    const costs: Record<string, Record<string, Stat>> = {};
    const reachTable: (string | number)[][] = [];
    const targets = [0.5, 0.1, 0.01];

    // Cost per iteration (or per V-cycle): the slope of the span over repeats of the solve.
    const solverByName = new Map(SOLVERS.map((solver) => [solver.name, solver]));
    const measured = await sweep(
      SIZES.flatMap((size) => SOLVERS.map((solver) => `${size}|${solver.name}`)),
      (key) => {
        const [size, name] = key.split('|') as [string, string];
        const solver = solverByName.get(name)!;
        return timer.slope(
          (n) => {
            const fluid = createFluid({
              device,
              resolution: Number(size),
              dyeResolution: 64,
              profiler: timer.profiler,
              params: { ...solver.params, warmStart: true },
            });
            fluid.setVelocity(largeSource);
            return (encoder) => {
              fluid.project(encoder, n);
              return () => fluid.destroy();
            };
          },
          solver.unit === 'cycles' ? 1 : 10,
          solver.unit === 'cycles' ? 11 : 110,
          REPS,
          solver.unit === 'cycles' ? 120 : 4000,
        );
      },
    );
    for (const size of SIZES) {
      costs[size] = {};
      for (const solver of SOLVERS) costs[size]![solver.name] = measured[`${size}|${solver.name}`]!;
    }

    for (const size of SIZES) {
      for (const [sourceName, source] of Object.entries(sources)) {
        for (const solver of SOLVERS) {
          if (sourceName === 'small' && solver.name.includes('levels')) continue;
          const counts =
            solver.unit === 'cycles' ? [1, 2, 3, 4, 6] : [8, 16, 32, 64, 128, 256, 512, 1024];
          const fluid = makeFluid(device, solver.params, size);
          const ratios: number[] = [];
          for (const count of counts) ratios.push(await coldResidual(device, fluid, source, count));
          fluid.destroy();
          const work = counts.map((c) =>
            pressureSolveWork(
              size,
              size,
              solver.params.pressureSolver ?? 'jacobi',
              c,
              solver.params,
            ),
          );
          curves[`${size}/${sourceName}/${solver.name}`] = { counts, ratios, work };
          const perUnit = costs[size]![solver.name]!.best ?? costs[size]![solver.name]!.median;
          reachTable.push([
            `${size}²`,
            sourceName,
            solver.name,
            ...targets.map((target) => {
              const n = reach(counts, ratios, target);
              return n === null
                ? `> ${counts[counts.length - 1]} ${solver.unit} (> ${fix(counts[counts.length - 1]! * perUnit, 2)} ms)`
                : `${fix(n, 1)} ${solver.unit}, ${fix(n * perUnit, 3)} ms`;
            }),
          ]);
        }
      }
    }
    results.pressure = { curves, costPerUnit: costs, targets };

    section(
      'Pressure solve, cold start: cost of one iteration (Jacobi, SOR) or V-cycle (multigrid), ms',
    );
    note(
      table(
        ['solver', ...SIZES.map((s) => `${s}²`)],
        SOLVERS.map((solver) => [solver.name, ...SIZES.map((s) => ms(costs[s]![solver.name]!, 4))]),
      ),
    );
    section(
      'Pressure solve, cold start: residual divergence (rms after / before) against iterations or cycles, large source (sigma 0.1)',
    );
    for (const size of SIZES) {
      note(`**${size}²**`);
      note();
      const rows: (string | number)[][] = [];
      for (const solver of SOLVERS) {
        const c = curves[`${size}/large/${solver.name}`] as { counts: number[]; ratios: number[] };
        rows.push([
          solver.name,
          c.counts.map((n, i) => `${n}: ${fix(c.ratios[i]!, 3)}`).join(', '),
        ]);
      }
      note(table(['solver', 'count: residual'], rows));
      note();
    }
    section(
      'Pressure solve, cold start: what it takes to reduce the divergence by a factor (time on this GPU)',
    );
    note(table(['grid', 'source', 'solver', 'to 0.5', 'to 0.1', 'to 0.01'], reachTable));
  });

  it.runIf(wants('omega'))('red-black SOR: the optimal omega per grid', async () => {
    const rows: (string | number)[][] = [];
    const out: Record<string, unknown> = {};
    for (const size of [64, 128, 256]) {
      const omegas = [1.5, 1.7, 1.8, 1.9, 1.92, 1.94, 1.95];
      const count = 2 * size;
      const ratios: number[] = [];
      for (const omega of omegas) {
        const fluid = makeFluid(device, { pressureSolver: 'sor', sorOmega: omega }, size);
        ratios.push(await coldResidual(device, fluid, largeSource, count));
        fluid.destroy();
      }
      const formula = sorOmega(size, size);
      const rho = Math.cos(Math.PI / size);
      const young = 2 / (1 + Math.sqrt(1 - rho * rho));
      const fluid = makeFluid(device, { pressureSolver: 'sor', sorOmega: 0 }, size);
      const atFormula = await coldResidual(device, fluid, largeSource, count);
      fluid.destroy();
      const best = ratios.indexOf(Math.min(...ratios));
      out[size] = { omegas, ratios, count, formula, young, atFormula, best: omegas[best] };
      rows.push([
        `${size}²`,
        count,
        `${fix(young, 3)} (used: ${fix(formula, 3)}) -> ${fix(atFormula, 4)}`,
        omegas.map((o, i) => `${o}: ${fix(ratios[i]!, 4)}`).join(', '),
        omegas[best]!,
      ]);
    }
    results.omega = out;
    section('Red-black SOR: residual after 2N iterations for several omega (cold, large source)');
    note(
      table(
        ['grid', 'iterations', 'Young optimum -> residual', 'omega: residual', 'best omega tried'],
        rows,
      ),
    );
  });

  // 2. Warm start in a running simulation --------------------------------------------------------------------------------

  it.runIf(wants('warm'))('warm start in a running simulation', async () => {
    const configs: { name: string; params: Partial<FluidParams> }[] = [
      { name: 'jacobi 15', params: { pressureSolver: 'jacobi', pressureIterations: 15 } },
      { name: 'jacobi 30 (default)', params: { pressureSolver: 'jacobi', pressureIterations: 30 } },
      { name: 'jacobi 60', params: { pressureSolver: 'jacobi', pressureIterations: 60 } },
      { name: 'jacobi 120', params: { pressureSolver: 'jacobi', pressureIterations: 120 } },
      { name: 'sor 30', params: { pressureSolver: 'sor', pressureIterations: 30 } },
      { name: 'sor 60', params: { pressureSolver: 'sor', pressureIterations: 60 } },
      { name: 'sor 120', params: { pressureSolver: 'sor', pressureIterations: 120 } },
      {
        name: 'multigrid V(1,1), 1 cycle',
        params: { pressureSolver: 'multigrid', multigridCycles: 1, multigridSmooth: 1 },
      },
      {
        name: 'multigrid V(2,2), 1 cycle',
        params: { pressureSolver: 'multigrid', multigridCycles: 1, multigridSmooth: 2 },
      },
      {
        name: 'multigrid V(1,1), 2 cycles',
        params: { pressureSolver: 'multigrid', multigridCycles: 2, multigridSmooth: 1 },
      },
    ];
    const sizes = QUICK ? [64, 128] : [64, 128, 256];
    const rows: (string | number)[][] = [];
    const out: Record<string, unknown> = {};
    for (const config of configs) {
      const row: (string | number)[] = [config.name];
      for (const size of sizes) {
        const pair: number[] = [];
        const rate: number[] = [];
        for (const warmStart of [true, false]) {
          const fluid = makeFluid(device, { ...config.params, warmStart }, size);
          const ratios: number[] = [];
          const rates: number[] = [];
          let t = 0;
          for (let step = 0; step < 90; step++) {
            const a = 0.5 + 0.3 * Math.cos(t * 3);
            const b = 0.5 + 0.3 * Math.sin(t * 3);
            t += 1 / 60;
            const a2 = 0.5 + 0.3 * Math.cos(t * 3);
            const b2 = 0.5 + 0.3 * Math.sin(t * 3);
            fluid.addForce([a, b], [a2, b2], 0.05, [(a2 - a) * 120, (b2 - b) * 120]);
            await submit(device, (e) => fluid.advance(e, 1));
            if (step >= 30 && step % 2 === 0) {
              const d = await readFloats(device, fluid.divergence);
              let sum = 0;
              for (const x of d) sum += x * x;
              const before = Math.sqrt(sum / d.length);
              const after = rmsDivergence(await readVelocity(device, fluid));
              ratios.push(after / Math.max(before, 1e-12));
              rates.push(after * size);
            }
          }
          fluid.destroy();
          pair.push(ratios.reduce((x, y) => x + y, 0) / ratios.length);
          rate.push(rates.reduce((x, y) => x + y, 0) / rates.length);
        }
        out[`${config.name}/${size}`] = {
          warm: pair[0],
          cold: pair[1],
          warmRate: rate[0],
          coldRate: rate[1],
        };
        row.push(
          `${fix(pair[0]!, 2)} / ${fix(pair[1]!, 2)}`,
          `${fix(rate[0]!, 3)} / ${fix(rate[1]!, 3)}`,
        );
      }
      rows.push(row);
    }
    results.warm = out;
    section(
      'Warm start in a running simulation (mean over steps 30 to 90 of a force circling a closed box), warm start / cold start',
    );
    note(
      "Each pair of columns: divergence after / before the projection, then the divergence left in the flow as a rate (rms of the divergence times the cell count: how fast a parcel of fluid changes area, 1 / s). The ratio flatters a warm start, whose 'before' still holds the last step's leftovers, so compare the rates.",
    );
    note();
    note(table(['solver', ...sizes.flatMap((s) => [`${s}² ratio`, `${s}² rate`])], rows));
  });

  // 3. Retained vortex ---------------------------------------------------------------------------------------------------------

  it.runIf(wants('vortex'))('vortex retention against grid size and solver', async () => {
    const psi = (x: number, y: number) =>
      0.02 * Math.exp(-((x - 0.5) ** 2 + (y - 0.5) ** 2) / 0.004);
    const e = 1e-4;
    const vortex = (x: number, y: number): [number, number] => [
      -(psi(x, y + e) - psi(x, y - e)) / (2 * e),
      (psi(x + e, y) - psi(x - e, y)) / (2 * e),
    ];
    const configs: { name: string; params: Partial<FluidParams> }[] = [
      { name: 'jacobi 30', params: { pressureSolver: 'jacobi', pressureIterations: 30 } },
      { name: 'multigrid 1', params: { pressureSolver: 'multigrid' } },
      {
        name: 'multigrid 1 + confinement 8',
        params: { pressureSolver: 'multigrid', vorticity: 8 },
      },
    ];
    const rows: (string | number)[][] = [];
    const out: Record<string, unknown> = {};
    for (const size of [64, 128, 256]) {
      for (const config of configs) {
        for (const scheme of ['semi-lagrangian', 'maccormack'] as const) {
          const fluid = makeFluid(device, { ...config.params, velocityScheme: scheme }, size, 64);
          fluid.setVelocity(vortex);
          // Project the analytic vortex properly first, whatever the solver under test is.
          const keep = { ...fluid.params };
          fluid.params.pressureSolver = 'multigrid';
          await submit(device, (enc) => fluid.project(enc, 4));
          Object.assign(fluid.params, keep);
          const measure = async () => {
            await submit(device, (enc) => fluid.resolve(enc));
            const fields = await readHalf(device, fluid.fields);
            let peak = 0;
            for (let i = 0; i < fields.length; i += 4)
              peak = Math.max(peak, Math.abs(fields[i + 2]!));
            const velocity = await readVelocity(device, fluid);
            let energy = 0;
            for (let j = 0; j < size; j++)
              for (let i = 0; i <= size; i++) energy += velocity.u(i, j) ** 2;
            for (let j = 0; j <= size; j++)
              for (let i = 0; i < size; i++) energy += velocity.v(i, j) ** 2;
            return { peak, energy };
          };
          const first = await measure();
          await submit(device, (enc) => fluid.advance(enc, 180));
          const last = await measure();
          fluid.destroy();
          const key = `${size}/${config.name}/${scheme}`;
          out[key] = { peak: last.peak / first.peak, energy: last.energy / first.energy };
          rows.push([
            `${size}²`,
            config.name,
            scheme,
            fix(last.peak / first.peak, 3),
            fix(last.energy / first.energy, 3),
          ]);
        }
      }
    }
    results.vortex = out;
    section(
      'Vortex retention: a Gaussian vortex (sigma 0.045 of the canvas height) after 3 s, no dissipation, no confinement',
    );
    note(
      table(
        [
          'velocity grid',
          'pressure',
          'velocity advection',
          'peak vorticity kept',
          'kinetic energy kept',
        ],
        rows,
      ),
    );
  });

  // 4. Resolution: velocity grid against dye grid ------------------------------------------------------------------------

  it.runIf(wants('resolution'))('velocity grid against dye grid', async () => {
    const ref = await getReference();
    const ref2 = await runScenario(device, REFERENCE_2);
    const floor = errorsOf(ref2, ref);
    const velocities = [64, 96, 128, 192, 256];
    const dyes = [256, 512, 1024];
    const out: Record<string, unknown> = {
      floor,
      reference: { energy: ref.energy, mass: ref.mass, detail: detail(ref.field, COMPARE) },
    };
    const rows: (string | number)[][] = [];
    const costs = COST
      ? await sweep(
          velocities.flatMap((v) => dyes.map((d) => `${v}|${d}`)),
          (key) => {
            const [velocity, dye] = key.split('|').map(Number) as [number, number];
            return stepCost({ velocity, dye, params: { pressureSolver: 'multigrid' } });
          },
        )
      : {};
    for (const velocity of velocities) {
      for (const dye of dyes) {
        const config: ScenarioConfig = { velocity, dye, params: { pressureSolver: 'multigrid' } };
        const run = await runScenario(device, config);
        const cost = costs[`${velocity}|${dye}`] ?? { median: 0, min: 0, max: 0, n: 0 };
        const error = errorsOf(run, ref);
        out[`${velocity}/${dye}`] = {
          error,
          energy: run.energy / ref.energy,
          detail: detail(run.field, COMPARE) / detail(ref.field, COMPARE),
          mass: run.mass,
          cost,
        };
        rows.push([
          `${velocity}²`,
          `${dye}²`,
          ms(cost),
          fix(error.coarse, 3),
          fix(error.fine, 3),
          fix(run.energy / ref.energy, 2),
          fix(detail(run.field, COMPARE) / detail(ref.field, COMPARE), 2),
        ]);
        if (dye === 512 && [64, 128, 256].includes(velocity)) {
          (sheets.resolution ??= []).push({ field: run.field, size: COMPARE });
        }
      }
    }
    sheets.resolution = [{ field: ref.field, size: COMPARE }, ...(sheets.resolution ?? [])];
    results.resolution = out;
    section(
      'Velocity grid against dye grid: cost of a step, and the stirred pattern against the reference run',
    );
    note(
      `Reference: velocity ${REFERENCE.velocity}², dye ${REFERENCE.dye}², multigrid x3. A second good run (velocity ${REFERENCE_2.velocity}², dye ${REFERENCE_2.dye}²) differs from it by ${fix(floor.coarse, 3)} (${COARSE}²) and ${fix(floor.fine, 3)} (${COMPARE}²): errors near that are indistinguishable from it. Pressure: one multigrid V-cycle; MacCormack for velocity and dye. Error: relative L2 of the dye at ${COARSE}² (large-scale placement) and ${COMPARE}² (including fine structure); energy and detail (mean squared gradient at ${COMPARE}²) relative to the reference.`,
    );
    note();
    note(
      table(
        [
          'velocity',
          'dye',
          'ms per step',
          `error ${COARSE}²`,
          `error ${COMPARE}²`,
          'energy',
          'detail',
        ],
        rows,
      ),
    );
  });

  // 5. Upsampling: how the velocity is read for the dye -------------------------------------------------------------------

  it.runIf(wants('upsample'))('upsampling: bilinear against cubic velocity', async () => {
    const kinds = ['bilinear', 'hardware-bilinear', 'bspline', 'catmull-rom'] as const;
    // (a) The reconstruction itself, against an analytic cell flow, at several velocity grid sizes.
    const A = 0.3;
    const flow = (x: number, y: number): [number, number] => [
      A * Math.sin(2 * Math.PI * x) * Math.cos(2 * Math.PI * y),
      -A * Math.cos(2 * Math.PI * x) * Math.sin(2 * Math.PI * y),
    ];
    const interp: Record<string, unknown> = {};
    const interpRows: (string | number)[][] = [];
    for (const size of [16, 32, 64]) {
      const fluid = makeFluid(device, {}, size);
      fluid.setVelocity(flow);
      const points: [number, number][] = [];
      for (let k = 0; k < 2000; k++) {
        points.push([
          2 + (size - 4) * ((k * 0.754877666) % 1),
          2 + (size - 4) * ((k * 0.569840291 + 0.3) % 1),
        ]);
      }
      const result = await interpolate(device, fluid, points);
      // Slope kinks across the u faces (integer x) along a row.
      const row = Math.floor(size / 3) + 0.5;
      const borders = Array.from({ length: size - 6 }, (_, i) => i + 3);
      const offsets = [-0.25, -0.05, 0.05, 0.25];
      const kinkPoints: [number, number][] = [];
      for (const b of borders) for (const o of offsets) kinkPoints.push([b + o, row]);
      const kinked = await interpolate(device, fluid, kinkPoints);
      fluid.destroy();
      for (const kind of kinds) {
        let sq = 0;
        let worst = 0;
        points.forEach((p, i) => {
          const [u, v] = flow(p[0] / size, p[1] / size);
          const eu = result[kind][i]![0] - u;
          const ev = result[kind][i]![1] - v;
          sq += eu * eu + ev * ev;
          worst = Math.max(worst, Math.hypot(eu, ev));
        });
        let kink = 0;
        borders.forEach((_, bi) => {
          const f = (oi: number) => kinked[kind][bi * offsets.length + oi]![0];
          kink += Math.abs((f(3) - f(2)) / 0.2 - (f(1) - f(0)) / 0.2);
        });
        const entry = {
          rms: Math.sqrt(sq / points.length / 2),
          max: worst,
          kink: kink / borders.length,
        };
        interp[`${size}/${kind}`] = entry;
        interpRows.push([
          `${size}²`,
          kind,
          fix(entry.rms * 1000, 3),
          fix(entry.max * 1000, 3),
          fix(entry.kink * 1000, 3),
        ]);
      }
    }
    results.interpolation = interp;
    section(
      'Reading the velocity between its faces: a cell flow of speed 0.3 sampled on a coarse grid',
    );
    note(
      'Error is against the analytic velocity at 2000 interior points, in 1e-3 image heights per second; the kink is the jump of the slope across a face, per cell, in the same unit (a C1 reconstruction has none).',
    );
    note();
    note(table(['velocity grid', 'filter', 'rms error', 'max error', 'slope kink'], interpRows));

    // (b) End to end: a thin dye line carried by that flow, against the exact characteristics.
    const T = 1;
    const STEPS = 60;
    const exactCurve = () => {
      const points: { x: number; y: number }[] = [];
      for (let k = 0; k <= 2000; k++) {
        let x = 0.4;
        let y = k / 2000;
        const dt = 1 / 240;
        for (let s = 0; s < T / dt; s++) {
          const v1 = flow(x, y);
          const v2 = flow(x + 0.5 * dt * v1[0], y + 0.5 * dt * v1[1]);
          const v3 = flow(x + 0.5 * dt * v2[0], y + 0.5 * dt * v2[1]);
          const v4 = flow(x + dt * v3[0], y + dt * v3[1]);
          x += (dt / 6) * (v1[0] + 2 * v2[0] + 2 * v3[0] + v4[0]);
          y += (dt / 6) * (v1[1] + 2 * v2[1] + 2 * v3[1] + v4[1]);
        }
        points.push({ x, y });
      }
      return points;
    };
    const curve = exactCurve();
    const monotone = curve.every((p, i) => i === 0 || p.y >= curve[i - 1]!.y - 1e-9);
    const exactAt = (y: number) => {
      let lo = 0;
      let hi = curve.length - 1;
      while (hi - lo > 1) {
        const mid = (lo + hi) >> 1;
        if (curve[mid]!.y < y) lo = mid;
        else hi = mid;
      }
      const a = curve[lo]!;
      const b = curve[hi]!;
      const t = b.y > a.y ? (y - a.y) / (b.y - a.y) : 0;
      return a.x + (b.x - a.x) * t;
    };
    const streak: Record<string, unknown> = {};
    const streakRows: (string | number)[][] = [];
    const DYE = 512;
    for (const velocity of [16, 32, 64]) {
      for (const kind of kinds) {
        const fluid = makeFluid(
          device,
          { dyeOnly: true, dyeVelocityFilter: kind, dyeScheme: 'maccormack' },
          velocity,
          DYE,
        );
        fluid.setVelocity(flow);
        fluid.setDye((x) => {
          const v = Math.exp(-((x - 0.4) ** 2) / (2 * 0.012 ** 2));
          return [v, v, v, v];
        });
        await submit(device, (e) => fluid.advance(e, STEPS));
        const data = await readHalf(device, fluid.dye.texture);
        fluid.destroy();
        if (velocity <= 32) {
          // A 256 x 256 crop of the middle of the picture, for the contact sheet.
          const crop = new Float32Array(256 * 256);
          for (let y = 0; y < 256; y++) {
            for (let x = 0; x < 256; x++)
              crop[y * 256 + x] = data[((y + 128) * DYE + x + 100) * 4]!;
          }
          (sheets.streakline ??= []).push({ field: crop, size: 256 });
        }
        const residual: number[] = [];
        for (let row = Math.round(0.08 * DYE); row < Math.round(0.92 * DYE); row++) {
          const y = (row + 0.5) / DYE;
          const xe = exactAt(y);
          let sum = 0;
          let weighted = 0;
          for (let col = 0; col < DYE; col++) {
            const x = (col + 0.5) / DYE;
            if (Math.abs(x - xe) > 0.06) continue;
            const d = data[(row * DYE + col) * 4]!;
            sum += d;
            weighted += d * x;
          }
          residual.push(sum > 1e-3 ? (weighted / sum - xe) * DYE : 0);
        }
        // Centreline error, and its high-pass part: what is left after a two-cell moving average.
        const window = Math.round((2 * DYE) / velocity);
        let sq = 0;
        let wiggle = 0;
        residual.forEach((r, i) => {
          let mean = 0;
          let n = 0;
          for (
            let k = Math.max(0, i - window);
            k <= Math.min(residual.length - 1, i + window);
            k++
          ) {
            mean += residual[k]!;
            n++;
          }
          sq += r * r;
          wiggle += (r - mean / n) ** 2;
        });
        const entry = {
          centreline: Math.sqrt(sq / residual.length),
          wiggle: Math.sqrt(wiggle / residual.length),
        };
        streak[`${velocity}/${kind}`] = entry;
        streakRows.push([`${velocity}²`, kind, fix(entry.centreline, 3), fix(entry.wiggle, 3)]);
      }
    }
    results.streakline = { ...streak, exactCurveMonotone: monotone };
    section(
      'Streaklines: a thin dye line (sigma 6 texels at a 512² dye grid) carried through the cell flow for 1 s, against the exact characteristics',
    );
    note(
      `Centreline error is the rms distance of the line's centroid from the exact one, in dye texels; wiggle is the part of it with a period shorter than two velocity cells (the kinks of the interpolation). ${monotone ? '' : 'The exact curve folds: numbers are approximate.'}`,
    );
    note();
    note(
      table(
        ['velocity grid', 'filter', 'centreline error (texels)', 'wiggle (texels)'],
        streakRows,
      ),
    );
  });

  // 6. Advection scheme cost against dye resolution ------------------------------------------------------------------------

  it.runIf(wants('advection'))(
    'dye advection: cost and sharpness by scheme and dye size',
    async () => {
      const schemes = ['semi-lagrangian', 'maccormack', 'bfecc'] as const;
      const dyes = [256, 512, 768, 1024];
      const costRows: (string | number)[][] = [];
      const out: Record<string, unknown> = {};
      const variants: { name: string; params: Partial<FluidParams> }[] = [
        ...schemes.map((scheme) => ({
          name: scheme,
          params: { dyeScheme: scheme } as Partial<FluidParams>,
        })),
        {
          name: 'semi-lagrangian + hardware-filtered bilinear velocity',
          params: { dyeScheme: 'semi-lagrangian', dyeVelocityFilter: 'hardware-bilinear' },
        },
        {
          name: 'maccormack + hardware-filtered bilinear velocity',
          params: { dyeScheme: 'maccormack', dyeVelocityFilter: 'hardware-bilinear' },
        },
        {
          name: 'maccormack + b-spline velocity',
          params: { dyeScheme: 'maccormack', dyeVelocityFilter: 'bspline' },
        },
        {
          name: 'maccormack + catmull-rom velocity',
          params: { dyeScheme: 'maccormack', dyeVelocityFilter: 'catmull-rom' },
        },
        {
          name: 'semi-lagrangian + catmull-rom velocity',
          params: { dyeScheme: 'semi-lagrangian', dyeVelocityFilter: 'catmull-rom' },
        },
      ];
      const costs = await sweep(
        variants.flatMap((variant) => dyes.map((dye) => `${variant.name}|${dye}`)),
        (key) => {
          const [name, dye] = key.split('|') as [string, string];
          const variant = variants.find((v) => v.name === name)!;
          return stepCost({
            velocity: 128,
            dye: Number(dye),
            params: { ...variant.params, dyeOnly: true },
          });
        },
      );
      for (const variant of variants) {
        const row: (string | number)[] = [variant.name];
        for (const dye of dyes) {
          const cost = costs[`${variant.name}|${dye}`]!;
          out[`cost/${variant.name}/${dye}`] = cost;
          row.push(ms(cost));
        }
        costRows.push(row);
      }
      results.advectionCost = out;
      section('Dye advection alone (dyeOnly), ms per step, velocity grid 128²');
      note(table(['scheme', ...dyes.map((d) => `${d}²`)], costRows));

      // Sharpness: a disc and a blob carried once round a vortex, in units that compare across dye sizes.
      const rows: (string | number)[][] = [];
      const rotation = (x: number, y: number): [number, number] => [
        -(y - 0.5) * 2 * Math.PI * 0.25,
        (x - 0.5) * 2 * Math.PI * 0.25,
      ];
      for (const dye of [256, 512, 1024]) {
        for (const scheme of schemes) {
          const disc = makeFluid(device, { dyeOnly: true, dyeScheme: scheme }, 64, dye);
          disc.setVelocity(rotation);
          disc.setDye((x, y) =>
            Math.hypot(x - 0.7, y - 0.5) < 0.05 ? [1, 1, 1, 1] : [0, 0, 0, 0],
          );
          await submit(device, (e) => disc.advance(e, 240));
          const data = await readHalf(device, disc.dye.texture);
          disc.destroy();
          // The disc has come back to where it started; the edge profile runs along the row through its centre.
          const row = Math.round(0.5 * dye - 0.5);
          const profile: number[] = [];
          for (let x = Math.round(0.7 * dye); x < dye; x++)
            profile.push(data[(row * dye + x) * 4]!);
          const first = (level: number) => profile.findIndex((v) => v < level);
          const edge = first(0.1) - first(0.9);
          let peak = 0;
          let total = 0;
          for (let i = 0; i < data.length; i += 4) {
            peak = Math.max(peak, data[i]!);
            total += data[i]!;
          }
          const area = Math.PI * 0.05 * 0.05 * dye * dye;
          out[`edge/${dye}/${scheme}`] = {
            edgeTexels: edge,
            edgeMilliHeights: (1000 * edge) / dye,
            peak,
            mass: total / area,
          };
          rows.push([
            `${dye}²`,
            scheme,
            edge,
            fix((1000 * edge) / dye, 1),
            fix(peak, 3),
            fix(total / area, 3),
          ]);
        }
      }
      results.advectionSharpness = out;
      section(
        'Sharpness: a disc (radius 0.05) carried once round a vortex (240 steps), velocity grid 64²',
      );
      note(
        'Edge is the 10 to 90 percent width of the disc edge, in texels and in thousandths of the canvas height (comparable across dye sizes); peak the largest value; mass the total over the disc area.',
      );
      note();
      note(table(['dye', 'scheme', 'edge (texels)', 'edge (1/1000 height)', 'peak', 'mass'], rows));
    },
  );

  // 7. The shortlist: cost against quality for whole configurations ---------------------------------------------------------

  it.runIf(wants('presets'))('configurations under a fixed budget', async () => {
    const ref = await getReference();
    const velocities = QUICK ? [64, 128, 192] : [64, 96, 128, 192];
    const dyes = QUICK ? [256, 512, 1024] : [256, 512, 768, 1024];
    const advections = [
      { name: 'SL/SL', velocityScheme: 'semi-lagrangian', dyeScheme: 'semi-lagrangian' },
      { name: 'MC/SL', velocityScheme: 'maccormack', dyeScheme: 'semi-lagrangian' },
      { name: 'MC/MC', velocityScheme: 'maccormack', dyeScheme: 'maccormack' },
    ] as const;
    const pressures = [
      { name: 'J30', params: { pressureSolver: 'jacobi', pressureIterations: 30 } },
      { name: 'MG(1,1)', params: { pressureSolver: 'multigrid', multigridSmooth: 1 } },
    ] as const;
    // Today's dye path (loads) against the filtered ones; the Jacobi baseline only with today's.
    const filters = ['bilinear', 'hardware-bilinear', 'catmull-rom'] as const;
    interface Row {
      velocity: number;
      dye: number;
      advection: string;
      pressure: string;
      filter: string;
      cost: Stat;
      error: number;
      errorFine: number;
      energy: number;
    }
    const configs: {
      advection: string;
      pressure: string;
      filter: string;
      config: ScenarioConfig;
    }[] = [];
    for (const velocity of velocities) {
      for (const dye of dyes) {
        for (const advection of advections) {
          for (const pressure of pressures) {
            for (const filter of filters) {
              if (pressure.name === 'J30' && filter !== 'bilinear') continue;
              configs.push({
                advection: advection.name,
                pressure: pressure.name,
                filter,
                config: {
                  velocity,
                  dye,
                  params: {
                    velocityScheme: advection.velocityScheme,
                    dyeScheme: advection.dyeScheme,
                    dyeVelocityFilter: filter,
                    ...pressure.params,
                  },
                },
              });
            }
          }
        }
      }
    }
    const costs = await sweep(
      configs.map((_, i) => String(i)),
      (key) => stepCost(configs[Number(key)]!.config, {}, 3),
    );
    const all: Row[] = [];
    for (const [i, entry] of configs.entries()) {
      const run = await runScenario(device, entry.config);
      const errors = errorsOf(run, ref);
      all.push({
        velocity: entry.config.velocity,
        dye: entry.config.dye,
        advection: entry.advection,
        pressure: entry.pressure,
        filter: entry.filter,
        cost: costs[String(i)]!,
        error: errors.coarse,
        errorFine: errors.fine,
        energy: run.energy / ref.energy,
      });
    }
    results.configurations = all;

    const budgets = [0.5, 1, 2, 4];
    const fastest = (r: Row) => r.cost.best ?? r.cost.median;
    const bestUnder = (budget: number, rows: Row[]) =>
      rows.filter((r) => fastest(r) <= budget).sort((a, b) => a.error - b.error)[0];
    const today = all.filter((r) => r.pressure === 'J30' && r.filter === 'bilinear');
    const picks = budgets.map((budget) => ({
      budget,
      best: bestUnder(budget, all),
      today: bestUnder(budget, today),
    }));
    results.picks = picks;
    const summarise = (r: Row | undefined) =>
      r
        ? [
            `${r.velocity}² / ${r.dye}²`,
            `${r.advection}, ${r.pressure}, ${r.filter}`,
            fix(fastest(r), 2),
            fix(r.error, 3),
            fix(r.energy, 2),
          ]
        : ['none fits', '', '', '', ''];
    section('Configurations: the best-looking one under a GPU budget per simulation step');
    note(
      `Candidates: velocity grid ${velocities.join(', ')}; dye grid ${dyes.join(', ')}; advection velocity/dye (SL semi-Lagrangian, MC MacCormack); pressure J30 (30 Jacobi iterations, today's default) or MG(1,1) (one multigrid V-cycle, one smoothing sweep each way); dye velocity filter bilinear (today's: loads), hardware-bilinear or catmull-rom (J30 only with bilinear). ${all.length} configurations. Cost is the slope per step of the whole step: 'fastest' is the slope between the fastest spans seen over ${PASSES} pass(es) of 3 repetitions (what the picks rank by), the median and range are in the JSON and the frontier table. Error is the relative L2 difference of the stirred pattern at ${COARSE}² from the reference run.`,
    );
    note();
    note(
      table(
        [
          'budget (ms/step)',
          'velocity / dye',
          'advection, pressure, dye velocity',
          'fastest ms',
          'error',
          'energy kept',
        ],
        picks.map(({ budget, best }) => [budget, ...summarise(best)]),
      ),
    );
    note();
    note(
      "The same budgets with today's solver and dye path only (30 Jacobi iterations, bilinear loads):",
    );
    note();
    note(
      table(
        [
          'budget (ms/step)',
          'velocity / dye',
          'advection, pressure, dye velocity',
          'fastest ms',
          'error',
          'energy kept',
        ],
        picks.map(({ budget, today: t }) => [budget, ...summarise(t)]),
      ),
    );
    // The frontier: configurations no other beats on both cost and error.
    const frontier = [...all]
      .sort((a, b) => fastest(a) - fastest(b))
      .filter((row, i, sorted) => sorted.slice(0, i).every((other) => other.error > row.error));
    results.frontier = frontier;
    section('Configurations: the cost / error frontier (no cheaper configuration looks better)');
    note(
      table(
        [
          'velocity',
          'dye',
          'advection',
          'pressure',
          'dye velocity',
          'fastest ms',
          'median ms (min-max)',
          'error',
          'energy kept',
        ],
        frontier.map((r) => [
          `${r.velocity}²`,
          `${r.dye}²`,
          r.advection,
          r.pressure,
          r.filter,
          fix(fastest(r), 2),
          ms(r.cost, 2),
          fix(r.error, 3),
          fix(r.energy, 2),
        ]),
      ),
    );
    // What each change buys at a fixed grid.
    const variants = [
      ['J30', 'bilinear'],
      ['MG(1,1)', 'bilinear'],
      ['MG(1,1)', 'hardware-bilinear'],
      ['MG(1,1)', 'catmull-rom'],
    ] as const;
    const comparison: (string | number)[][] = [];
    for (const velocity of velocities) {
      for (const dye of [dyes[1]!, dyes[dyes.length - 1]!]) {
        const row: (string | number)[] = [`${velocity}² / ${dye}²`];
        for (const [pressure, filter] of variants) {
          const r = all.find(
            (x) =>
              x.velocity === velocity &&
              x.dye === dye &&
              x.advection === 'MC/MC' &&
              x.pressure === pressure &&
              x.filter === filter,
          );
          row.push(r ? `${fix(fastest(r), 2)} ms, error ${fix(r.error, 3)}` : '');
        }
        comparison.push(row);
      }
    }
    section(
      'What each change buys at a fixed grid (MacCormack both): fastest cost per step and error',
    );
    note(
      table(
        [
          'velocity / dye',
          'Jacobi 30, loaded bilinear (today)',
          'multigrid, loaded bilinear',
          'multigrid, hardware bilinear',
          'multigrid, Catmull-Rom',
        ],
        comparison,
      ),
    );
  });

  // 8. What an under-converged solve looks like -------------------------------------------------------------------------------

  it.runIf(wants('convergence'))('what an under-converged solve costs the picture', async () => {
    const ref = await getReference();
    const ref2 = await runScenario(device, REFERENCE_2);
    const floor = errorsOf(ref2, ref);
    const variants: { name: string; params: Partial<FluidParams> }[] = [
      { name: 'jacobi 5', params: { pressureSolver: 'jacobi', pressureIterations: 5 } },
      { name: 'jacobi 10', params: { pressureSolver: 'jacobi', pressureIterations: 10 } },
      { name: 'jacobi 20', params: { pressureSolver: 'jacobi', pressureIterations: 20 } },
      { name: 'jacobi 30 (default)', params: { pressureSolver: 'jacobi', pressureIterations: 30 } },
      { name: 'jacobi 60', params: { pressureSolver: 'jacobi', pressureIterations: 60 } },
      { name: 'jacobi 120', params: { pressureSolver: 'jacobi', pressureIterations: 120 } },
      { name: 'sor 60', params: { pressureSolver: 'sor', pressureIterations: 60 } },
      { name: 'multigrid 1 cycle', params: { pressureSolver: 'multigrid', multigridCycles: 1 } },
      { name: 'multigrid 2 cycles', params: { pressureSolver: 'multigrid', multigridCycles: 2 } },
    ];
    const rows: (string | number)[][] = [];
    const out: Record<string, unknown> = { floor };
    for (const size of [128, 256]) {
      for (const variant of variants) {
        const run = await runScenario(device, { velocity: size, dye: 512, params: variant.params });
        const error = errorsOf(run, ref);
        out[`${size}/${variant.name}`] = {
          error,
          mass: run.mass,
          energy: run.energy / ref.energy,
          divergence: run.divergence,
        };
        rows.push([
          `${size}²`,
          variant.name,
          fix(error.coarse, 3),
          fix(error.fine, 3),
          fix(run.mass, 3),
          fix(run.energy / ref.energy, 2),
          fix(run.divergence, 3),
        ]);
        if (
          size === 128 &&
          ['jacobi 5', 'jacobi 30 (default)', 'multigrid 1 cycle'].includes(variant.name)
        ) {
          (sheets.convergence ??= []).push({ field: run.field, size: COMPARE });
        }
      }
    }
    sheets.convergence = [{ field: ref.field, size: COMPARE }, ...(sheets.convergence ?? [])];
    results.convergence = out;
    section(
      'Under-converged pressure: the stirred pattern after 2 s against the reference run (dye 512²)',
    );
    note(
      `Error: relative L2 difference of the dye from the reference (floor between two good runs: ${fix(floor.coarse, 3)} at ${COARSE}², ${fix(floor.fine, 3)} at ${COMPARE}²). Mass: total dye over its start (1 for an incompressible flow and a conservative scheme; ink piles up or thins where the flow compresses). Energy: kinetic energy relative to the reference. Divergence: rms divergence left in the flow at the end, as the rate at which a parcel's area changes, 1 / s.`,
    );
    note();
    note(
      table(
        [
          'velocity',
          'pressure',
          `error ${COARSE}²`,
          `error ${COMPARE}²`,
          'mass',
          'energy',
          'divergence (1/s)',
        ],
        rows,
      ),
    );
  });

  it.runIf(wants('images'))('contact sheets', async () => {
    for (const [name, panels] of Object.entries(sheets)) await writeSheet(name, panels);
  });
});
