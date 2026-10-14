import { describe, expect, it, vi } from 'vitest';
import {
  COURANT_LIMIT,
  MAX_SOURCES,
  computeShallowWaterGrid,
  createShallowWater,
  shallowWaterStepSeconds,
  type ShallowWater,
  type ShallowWaterKind,
  type ShallowWaterParams,
} from '../src/gpu/sim/shallow-water.ts';
import { loadEffect } from '../src/effects/registry.ts';
import { defaultParams } from '../src/effects/types.ts';
import { noProfiler } from '../src/gpu/profiler.ts';
import { checkShader, formatDiagnostic } from '../src/gpu/shader.ts';
import { expectValid, getTestDevice, readTexture } from './gpu.ts';
import { gpuF32, lane, rowsOf, scalars } from './wgsl-eval.ts';

vi.setConfig({ testTimeout: 120_000 });

// --- Helpers -------------------------------------------------------------------------------------

const KINDS: ShallowWaterKind[] = ['compute', 'render'];
const G = 9.81;

function halfToFloat(bits: number): number {
  const sign = bits & 0x8000 ? -1 : 1;
  const exponent = (bits >> 10) & 0x1f;
  const fraction = bits & 0x3ff;
  if (exponent === 0) return sign * 2 ** -14 * (fraction / 1024);
  if (exponent === 31) return fraction ? Number.NaN : sign * Infinity;
  return sign * 2 ** (exponent - 15) * (1 + fraction / 1024);
}

/** The exact state, read back from the rgba32float textures. */
interface State {
  width: number;
  height: number;
  /** Per cell, row-major, row 0 at the top. */
  h: Float32Array;
  hu: Float32Array;
  hv: Float32Array;
  b: Float32Array;
  dx: number;
  /** Cubic metres of water. */
  volume: number;
  /** Velocity of a cell, m/s (zero where dry). */
  speed(i: number): number;
}

async function readState(device: GPUDevice, solver: ShallowWater): Promise<State> {
  const { width, height, cellSize } = solver.grid;
  const bytes = await readTexture(device, solver.stateTexture, 16);
  const data = new Float32Array(bytes.buffer, bytes.byteOffset, width * height * 4);
  const h = new Float32Array(width * height);
  const hu = new Float32Array(width * height);
  const hv = new Float32Array(width * height);
  const b = new Float32Array(width * height);
  let volume = 0;
  for (let i = 0; i < width * height; i++) {
    h[i] = data[i * 4]!;
    hu[i] = data[i * 4 + 1]!;
    hv[i] = data[i * 4 + 2]!;
    b[i] = data[i * 4 + 3]!;
    volume += h[i]! * cellSize * cellSize;
  }
  return {
    width,
    height,
    h,
    hu,
    hv,
    b,
    dx: cellSize,
    volume,
    speed: (i) => (h[i]! > 1e-4 ? Math.hypot(hu[i]!, hv[i]!) / h[i]! : 0),
  };
}

/** The consumer outputs, read back from the rgba16float textures. */
async function readOutputs(device: GPUDevice, solver: ShallowWater) {
  const { width, height } = solver.grid;
  const decode = async (texture: GPUTexture) => {
    const bytes = await readTexture(device, texture, 8);
    const half = new Uint16Array(bytes.buffer, bytes.byteOffset, width * height * 4);
    return Float32Array.from(half, halfToFloat);
  };
  return {
    width,
    height,
    water: await decode(solver.texture),
    ground: await decode(solver.groundTexture),
  };
}

async function advance(device: GPUDevice, solver: ShallowWater, steps: number): Promise<void> {
  // Several submits keep the command buffers a sensible size.
  let left = steps;
  while (left > 0) {
    const n = Math.min(left, 400);
    const encoder = device.createCommandEncoder();
    solver.advance(encoder, n);
    device.queue.submit([encoder.finish()]);
    left -= n;
  }
  await device.queue.onSubmittedWorkDone();
}

/** Steps for `seconds` of simulated time. */
const stepsFor = (solver: ShallowWater, seconds: number) =>
  Math.round(seconds / solver.stepSeconds);

async function simulate(
  device: GPUDevice,
  solver: ShallowWater,
  seconds: number,
  fps = 60,
  perFrame?: () => void,
): Promise<number> {
  let steps = 0;
  for (let t = 0; t < seconds * fps - 1e-6; t++) {
    perFrame?.();
    const encoder = device.createCommandEncoder();
    steps += solver.update(encoder, 1 / fps);
    device.queue.submit([encoder.finish()]);
  }
  await device.queue.onSubmittedWorkDone();
  return steps;
}

function create(
  device: GPUDevice,
  kind: ShallowWaterKind,
  params: Partial<ShallowWaterParams> = {},
  resolution = 96,
  aspect = 1.5,
): ShallowWater {
  return createShallowWater({
    device,
    kind,
    resolution,
    aspect,
    size: 12,
    params: { friction: 0, ...params },
  });
}

const maxAbs = (a: ArrayLike<number>) =>
  Array.from(a).reduce((m, x) => Math.max(m, Math.abs(x)), 0);
const finiteAll = (a: ArrayLike<number>) => Array.from(a).every(Number.isFinite);
const minOf = (a: ArrayLike<number>) => Array.from(a).reduce((m, x) => Math.min(m, x), Infinity);

/** A small deterministic terrain: rolling hills and a few steep pillars, metres. */
function hills(u: number, v: number): number {
  const smooth =
    0.35 * Math.sin(u * 9.1 + 0.4) * Math.cos(v * 7.3) + 0.25 * Math.sin(u * 17 + v * 11 + 1.3);
  const pillar = (cx: number, cy: number, r: number, h: number) =>
    Math.hypot((u - cx) * 1.5, v - cy) < r ? h : 0;
  return 0.5 + smooth + pillar(0.3, 0.6, 0.06, 1.2) + pillar(0.7, 0.35, 0.05, 1.2);
}

// --- The stability rule, no GPU needed -----------------------------------------------------------

describe('timestep and grid arithmetic', () => {
  it('derives the step from the CFL rule and shrinks it with depth and resolution', () => {
    const base = shallowWaterStepSeconds({ gravity: G, maxDepth: 1, cfl: 0.4 }, 0.05);
    // dt = cfl dx / (2 lambda_cap), lambda_cap = 2 sqrt(g h)
    expect(base).toBeCloseTo((0.4 * 0.05) / (4 * Math.sqrt(G)), 9);
    expect(shallowWaterStepSeconds({ gravity: G, maxDepth: 4, cfl: 0.4 }, 0.05)).toBeCloseTo(
      base / 2,
      9,
    );
    expect(shallowWaterStepSeconds({ gravity: G, maxDepth: 1, cfl: 0.4 }, 0.025)).toBeCloseTo(
      base / 2,
      9,
    );
  });

  it('never lets the CFL number exceed the limit, whatever is asked for', () => {
    const greedy = shallowWaterStepSeconds({ gravity: G, maxDepth: 1, cfl: 50 }, 0.05);
    const limit = shallowWaterStepSeconds({ gravity: G, maxDepth: 1, cfl: COURANT_LIMIT }, 0.05);
    expect(greedy).toBe(limit);
    expect(
      shallowWaterStepSeconds({ gravity: G, maxDepth: 1, cfl: Number.NaN }, 0.05),
    ).toBeGreaterThan(0);
    // The fastest signal the shader allows is lambda_cap, so the CFL number is cfl at most.
    const lambdaCap = 2 * Math.sqrt(G * 1);
    expect((2 * lambdaCap * limit) / 0.05).toBeLessThanOrEqual(COURANT_LIMIT + 1e-9);
  });

  it('fits square cells to the aspect ratio', () => {
    expect(computeShallowWaterGrid(192, 1.5)).toEqual({ width: 192, height: 128 });
    expect(computeShallowWaterGrid(192, 0.5)).toEqual({ width: 96, height: 192 });
    expect(computeShallowWaterGrid(Number.NaN, Number.NaN)).toEqual({ width: 256, height: 256 });
  });
});

// --- The solver, both variants -------------------------------------------------------------------

describe.each(KINDS)('shallow-water solver (%s)', (kind) => {
  it('keeps a lake at rest over uneven terrain at rest', async () => {
    const device = await getTestDevice();
    await expectValid(device, async () => {
      const solver = create(device, kind, { maxDepth: 1.5 });
      const level = 1.1;
      solver.setTerrain(hills);
      solver.setWater((u, v) => Math.max(level - hills(u, v), 0));
      const wet0 = await (async () => {
        await advance(device, solver, 1);
        return readState(device, solver);
      })();
      const volume0 = wet0.volume;
      expect(volume0).toBeGreaterThan(1);
      // Several simulated seconds.
      await advance(device, solver, stepsFor(solver, 8));
      const state = await readState(device, solver);
      let worstSpeed = 0;
      let worstLevel = 0;
      let wet = 0;
      let dry = 0;
      for (let i = 0; i < state.h.length; i++) {
        worstSpeed = Math.max(worstSpeed, state.speed(i));
        if (state.h[i]! > 1e-3) {
          wet++;
          worstLevel = Math.max(worstLevel, Math.abs(state.b[i]! + state.h[i]! - level));
        } else dry++;
      }
      // There is both water and dry ground (the pillars and the hill crests stand clear).
      expect(wet).toBeGreaterThan(500);
      expect(dry).toBeGreaterThan(50);
      expect(worstSpeed).toBeLessThan(2e-3);
      expect(worstLevel).toBeLessThan(2e-4);
      expect(Math.abs(state.volume - volume0) / volume0).toBeLessThan(1e-5);
      // And the published outputs agree: no flow, and a flat surface (zero slope) on the water.
      const out = await readOutputs(device, solver);
      let maxWaterSlope = 0;
      for (let i = 0; i < out.width * out.height; i++) {
        if (out.water[i * 4]! > 0.02) {
          maxWaterSlope = Math.max(
            maxWaterSlope,
            Math.abs(out.ground[i * 4 + 1]!),
            Math.abs(out.ground[i * 4 + 2]!),
          );
        }
      }
      expect(maxWaterSlope).toBeLessThan(5e-3);
      solver.destroy();
    });
  });

  it('conserves volume in a closed basin through a dam break, wetting and drying', async () => {
    const device = await getTestDevice();
    await expectValid(device, async () => {
      const solver = create(device, kind, { maxDepth: 1.3 }, 96);
      // Reservoir 1 m deep on the left, dry basin with pillars on the right, walls all round.
      solver.setTerrain((u, v) => {
        const pillar = (cx: number, cy: number, r: number) =>
          Math.hypot((u - cx) * 1.5, v - cy) < r ? 1.6 : 0;
        return pillar(0.6, 0.35, 0.07) + pillar(0.75, 0.7, 0.08) + 0.3 * u;
      });
      solver.setWater((u) => (u < 0.3 ? 1 - 0.3 * u : 0));
      await advance(device, solver, 1);
      const start = await readState(device, solver);
      const marks: number[] = [];
      let lowest = Infinity;
      for (let k = 0; k < 10; k++) {
        await advance(device, solver, stepsFor(solver, 0.8));
        const s = await readState(device, solver);
        marks.push(Math.abs(s.volume - start.volume) / start.volume);
        lowest = Math.min(lowest, minOf(s.h));
        expect(finiteAll(s.h) && finiteAll(s.hu) && finiteAll(s.hv)).toBe(true);
      }
      // Water has really moved: the reservoir is empty-ish and the far side is wet.
      const end = await readState(device, solver);
      let far = 0;
      for (let y = 0; y < end.height; y++) {
        for (let x = Math.floor(end.width * 0.8); x < end.width; x++)
          far += end.h[y * end.width + x]!;
      }
      expect(far).toBeGreaterThan(1);
      // Tolerance: 1e-5 of the volume over 8 simulated seconds (float32 rounding only; measured ~1e-7).
      expect(Math.max(...marks)).toBeLessThan(1e-5);
      expect(lowest).toBeGreaterThanOrEqual(0);
      solver.destroy();
    });
  });

  it('stays finite and non-negative in a violent dam break, and the clamps hold', async () => {
    const device = await getTestDevice();
    await expectValid(device, async () => {
      // A full-height reservoir released over rough terrain with pillars: the worst case the
      // solver is meant to survive, with no friction to help.
      const maxDepth = 1.5;
      const solver = create(device, kind, { maxDepth, edges: 'wall' }, 96);
      solver.setTerrain((u, v) => 0.25 * hills(u, v) + (u > 0.45 && u < 0.5 ? 1.8 : 0));
      // Deliberately deeper than maxDepth: the cap has to hold it.
      solver.setWater((u, v) => (u < 0.45 ? 3 - 0.25 * hills(u, v) : 0));
      const capped = await (async () => {
        await advance(device, solver, 1);
        return readState(device, solver);
      })();
      expect(Math.max(...capped.h)).toBeLessThanOrEqual(maxDepth + 1e-6);
      // Release: a dam that vanishes.
      solver.setTerrain((u, v) => 0.25 * hills(u, v));
      const lambdaCap = 2 * Math.sqrt(G * maxDepth);
      for (let k = 0; k < 8; k++) {
        await advance(device, solver, stepsFor(solver, 0.5));
        const s = await readState(device, solver);
        expect(finiteAll(s.h) && finiteAll(s.hu) && finiteAll(s.hv) && finiteAll(s.b)).toBe(true);
        expect(minOf(s.h)).toBeGreaterThanOrEqual(0);
        expect(Math.max(...s.h)).toBeLessThanOrEqual(maxDepth + 1e-6);
        // |u| + sqrt(g h) <= lambda_cap in every cell: the CFL guarantee.
        let worst = 0;
        for (let i = 0; i < s.h.length; i++) {
          if (s.h[i]! > 1e-5) worst = Math.max(worst, s.speed(i) + Math.sqrt(G * s.h[i]!));
        }
        expect(worst).toBeLessThanOrEqual(lambdaCap * 1.001);
      }
      solver.destroy();
    });
  });

  it('sends a dry-bed dam break out at the analytic Ritter speed', async () => {
    const device = await getTestDevice();
    await expectValid(device, async () => {
      // A long thin channel, reservoir h0 on the left half, dry on the right, frictionless.
      const h0 = 1;
      const solver = createShallowWater({
        device,
        kind,
        resolution: 192,
        aspect: 6,
        size: 12,
        params: { friction: 0, maxDepth: 1.25 },
      });
      solver.setWater((u) => (u < 0.5 ? h0 : 0));
      await advance(device, solver, 1);
      const c0 = Math.sqrt(G * h0);
      const t = 0.6;
      await advance(device, solver, stepsFor(solver, t) - 1);
      const s = await readState(device, solver);
      const time = solver.simTime;
      const row = Math.floor(s.height / 2);
      const profile = Array.from({ length: s.width }, (_, x) => s.h[row * s.width + x]!);
      // Ritter's solution: h = (2 c0 - x/t)^2 / (9 g) between -c0 t and 2 c0 t, from the dam.
      const ritter = (xx: number) => {
        const q = xx / time;
        if (q <= -c0) return h0;
        if (q >= 2 * c0) return 0;
        return (2 * c0 - q) ** 2 / (9 * G);
      };
      let error = 0;
      let total = 0;
      for (let x = 0; x < s.width; x++) {
        const exact = ritter((x + 0.5) * s.dx - 6);
        error += Math.abs(profile[x]! - exact);
        total += exact;
      }
      // The depth profile matches in the L1 sense to a few percent of the water involved...
      expect(error / total).toBeLessThan(0.04);
      // ...and the front, taken where the depth falls below 1 % of h0, is where the exact solution
      // puts that depth, within 12 % of the distance travelled (a first-order scheme rounds the
      // thin tip off).
      let front = 0;
      for (let x = s.width - 1; x >= 0; x--) {
        if (profile[x]! > 0.01 * h0) {
          front = (x + 1) * s.dx - 6;
          break;
        }
      }
      const exactFront = (2 * c0 - Math.sqrt(9 * G * 0.01 * h0)) * time;
      expect(Math.abs(front - exactFront) / exactFront).toBeLessThan(0.12);
      // Never faster than the tip of the exact solution, 2 c0.
      expect(front).toBeLessThan(2 * c0 * time * 1.02);
      solver.destroy();
    });
  });

  it('drives a bore into a wet bed at the Stoker speed', async () => {
    const device = await getTestDevice();
    await expectValid(device, async () => {
      const h0 = 1;
      const h1 = 0.2;
      const solver = createShallowWater({
        device,
        kind,
        resolution: 192,
        aspect: 6,
        size: 12,
        params: { friction: 0, maxDepth: 1.25 },
      });
      solver.setWater((u) => (u < 0.5 ? h0 : h1));
      await advance(device, solver, 1);
      const c0 = Math.sqrt(G * h0);
      const c1 = Math.sqrt(G * h1);
      // Stoker's middle state: the root of -8 g h1 cm^2 (c0 - cm)^2 + (cm^2 - g h1)^2 (cm^2 + g h1) = 0.
      const f = (cm: number) =>
        -8 * G * h1 * cm ** 2 * (c0 - cm) ** 2 + (cm ** 2 - G * h1) ** 2 * (cm ** 2 + G * h1);
      let lo = c1 * 1.0001;
      let hi = c0;
      for (let i = 0; i < 80; i++) {
        const mid = (lo + hi) / 2;
        if (f(lo) * f(mid) <= 0) hi = mid;
        else lo = mid;
      }
      const cm = (lo + hi) / 2;
      const hm = (cm * cm) / G;
      const shock = (2 * cm * cm * (c0 - cm)) / (cm * cm - c1 * c1);
      await advance(device, solver, stepsFor(solver, 0.9) - 1);
      const s = await readState(device, solver);
      const time = solver.simTime;
      const row = Math.floor(s.height / 2);
      // The bore: the rightmost cell above the mid-level between h1 and the plateau hm.
      let bore = 0;
      for (let x = s.width - 1; x >= 0; x--) {
        if (s.h[row * s.width + x]! > (h1 + hm) / 2) {
          bore = (x + 0.5) * s.dx - 6;
          break;
        }
      }
      expect(Math.abs(bore - shock * time) / (shock * time)).toBeLessThan(0.06);
      // The plateau behind the bore has the predicted depth.
      const plateau = s.h[row * s.width + Math.round(s.width / 2 + (0.5 * shock * time) / s.dx)]!;
      expect(Math.abs(plateau - hm) / hm).toBeLessThan(0.06);
      solver.destroy();
    });
  });

  it('runs water downhill and collects it in the lowest basin, leaving the high ground dry', async () => {
    const device = await getTestDevice();
    await expectValid(device, async () => {
      const solver = create(device, kind, { maxDepth: 0.6, friction: 0.03 }, 96);
      // A plane falling from 1.2 m at the left to 0 at the right, with a pit near the right edge.
      const bed = (u: number, v: number) =>
        1.2 * (1 - u) - 0.45 * Math.exp(-(((u - 0.85) / 0.09) ** 2 + ((v - 0.5) / 0.16) ** 2));
      solver.setTerrain(bed);
      await advance(device, solver, 1);
      // A spring on the high ground, for 4 s.
      const rate = 0.4;
      await simulate(device, solver, 4, 60, () =>
        solver.addSource({ x: 0.12, y: 0.5, radius: 0.5, volume: rate }),
      );
      const sprung = solver.simTime;
      await simulate(device, solver, 60, 60);
      const s = await readState(device, solver);
      const injected = rate * sprung;
      expect(Math.abs(s.volume - injected) / injected).toBeLessThan(0.02);
      let pooled = 0;
      let centroid = 0;
      let wettestUphill = 0;
      let pooledLeft = Infinity;
      for (let y = 0; y < s.height; y++) {
        for (let x = 0; x < s.width; x++) {
          const depth = s.h[y * s.width + x]!;
          const u = (x + 0.5) / s.width;
          centroid += depth * s.dx * s.dx * u;
          if (depth > 0.02) {
            pooled += depth * s.dx * s.dx;
            pooledLeft = Math.min(pooledLeft, u);
          }
          if (u < 0.45) wettestUphill = Math.max(wettestUphill, depth);
        }
      }
      // Nearly all the water is in a pool (deeper than 2 cm), a pool that lies in the low ground at
      // the right; the high ground holds no puddle, only the film that friction leaves behind.
      expect(centroid / s.volume).toBeGreaterThan(0.75);
      expect(pooled / s.volume).toBeGreaterThan(0.9);
      expect(pooledLeft).toBeGreaterThan(0.55);
      expect(wettestUphill).toBeLessThan(0.01);
      // The pit filled to a level surface.
      let low = Infinity;
      let high = -Infinity;
      for (let i = 0; i < s.h.length; i++) {
        if (s.h[i]! > 0.03) {
          low = Math.min(low, s.b[i]! + s.h[i]!);
          high = Math.max(high, s.b[i]! + s.h[i]!);
        }
      }
      expect(high - low).toBeLessThan(0.02);
      solver.destroy();
    });
  });

  it('drains through open edges and holds its water behind walls', async () => {
    const device = await getTestDevice();
    await expectValid(device, async () => {
      const volumes: Record<string, number[]> = {};
      for (const edges of ['wall', 'open'] as const) {
        const solver = create(device, kind, { edges, friction: 0.03, maxDepth: 0.8 }, 96);
        // A pond near the right-hand edge on level ground.
        solver.setWater((u, v) => (Math.hypot((u - 0.8) * 1.5, v - 0.5) < 0.22 ? 0.5 : 0));
        await advance(device, solver, 1);
        const list: number[] = [(await readState(device, solver)).volume];
        for (let k = 0; k < 4; k++) {
          await advance(device, solver, stepsFor(solver, 2));
          list.push((await readState(device, solver)).volume);
        }
        volumes[edges] = list;
        solver.destroy();
      }
      const wall = volumes.wall!;
      const open = volumes.open!;
      expect(Math.abs(wall.at(-1)! - wall[0]!) / wall[0]!).toBeLessThan(1e-4);
      // Open: monotonic loss, most of it gone in 8 simulated seconds.
      for (let i = 1; i < open.length; i++)
        expect(open[i]!).toBeLessThanOrEqual(open[i - 1]! + 1e-6);
      expect(open.at(-1)! / open[0]!).toBeLessThan(0.4);
    });
  });

  it('adds the configured volume per second, removes it, and rains', async () => {
    const device = await getTestDevice();
    await expectValid(device, async () => {
      const solver = create(device, kind, { maxDepth: 1.5 }, 96);
      await advance(device, solver, 1);
      expect((await readState(device, solver)).volume).toBe(0);

      const rate = 0.6;
      await simulate(device, solver, 2, 60, () =>
        solver.addSource({ x: 0.4, y: 0.5, radius: 0.8, volume: rate }),
      );
      const filled = await readState(device, solver);
      // Simulated time is whole steps: compare with the time actually run.
      const expected = rate * solver.simTime;
      expect(solver.simTime).toBeGreaterThan(1.97);
      expect(solver.simTime).toBeLessThan(2.03);
      expect(Math.abs(filled.volume - expected) / expected).toBeLessThan(0.005);

      // A source that overlaps the corner is clipped by the grid but still adds its full volume.
      const before = filled.volume;
      const t0 = solver.simTime;
      await simulate(device, solver, 1, 60, () =>
        solver.addSource({ x: 0.0, y: 0.0, radius: 0.6, volume: 0.3 }),
      );
      const corner = await readState(device, solver);
      expect(Math.abs(corner.volume - before - 0.3 * (solver.simTime - t0))).toBeLessThan(0.004);

      // A drain removes water (limited by what is there): a still 0.6 m lake, 0.5 m^3 / s for a second.
      const lake = create(device, kind, { maxDepth: 1.5 }, 96);
      lake.setWater(() => 0.6);
      await advance(device, lake, 1);
      const full = await readState(device, lake);
      await simulate(device, lake, 1, 60, () =>
        lake.addSource({ x: 0.5, y: 0.5, radius: 1.6, volume: -0.5 }),
      );
      const drained = await readState(device, lake);
      expect(Math.abs(full.volume - drained.volume - 0.5 * lake.simTime)).toBeLessThan(0.01);
      expect(minOf(drained.h)).toBeGreaterThanOrEqual(0);
      lake.destroy();

      // Rain: metres per second over the whole 12 x 8 m grid.
      const rain = create(device, kind, { rain: 0.01, maxDepth: 1.5 }, 96);
      await simulate(device, rain, 1, 60);
      const wet = await readState(device, rain);
      expect(
        Math.abs(wet.volume - 0.01 * rain.simTime * 96) / (0.01 * rain.simTime * 96),
      ).toBeLessThan(0.01);
      rain.destroy();
      solver.destroy();
    });
  });

  it('simulates the same at 60 Hz, 120 Hz and 40 Hz, and survives a long stall', async () => {
    const device = await getTestDevice();
    await expectValid(device, async () => {
      const results: Float32Array[] = [];
      const steps: number[] = [];
      for (const fps of [60, 120, 40]) {
        const solver = create(device, kind, { maxDepth: 1.3 }, 96);
        solver.setTerrain((u, v) => hills(u, v) * 0.3);
        solver.setWater((u, v) => (u < 0.3 ? 1 - hills(u, v) * 0.3 : 0));
        steps.push(await simulate(device, solver, 0.5, fps));
        results.push((await readState(device, solver)).h);
        solver.destroy();
      }
      const reference = results[0]!;
      expect(Math.abs(steps[0]! - steps[1]!)).toBeLessThanOrEqual(1);
      expect(Math.abs(steps[0]! - steps[2]!)).toBeLessThanOrEqual(1);
      let norm = 0;
      for (const x of reference) norm += x;
      for (const other of results.slice(1)) {
        let difference = 0;
        for (let i = 0; i < other.length; i++) difference += Math.abs(other[i]! - reference[i]!);
        expect(difference / norm).toBeLessThan(0.02);
      }

      // A stall delivers one huge delta: capped, not exploded.
      const solver = create(device, kind, { maxDepth: 1.3 }, 96);
      solver.setWater((u) => (u < 0.3 ? 1 : 0));
      const encoder = device.createCommandEncoder();
      const run = solver.update(encoder, 30);
      device.queue.submit([encoder.finish()]);
      expect(run).toBeLessThanOrEqual(48);
      expect(run).toBeGreaterThan(0);
      const after = await readState(device, solver);
      expect(finiteAll(after.h)).toBe(true);
      expect(minOf(after.h)).toBeGreaterThanOrEqual(0);
      solver.destroy();
    });
  });

  it('runs time at the time scale and not faster than it', async () => {
    const device = await getTestDevice();
    await expectValid(device, async () => {
      const solver = create(device, kind, { timeScale: 0.25 }, 64);
      solver.setWater((u) => (u < 0.5 ? 0.5 : 0));
      const steps = await simulate(device, solver, 1, 60);
      expect(steps * solver.stepSeconds).toBeCloseTo(0.25, 1);
      expect(solver.simTime).toBeCloseTo(0.25, 1);
      solver.params.timeScale = 0;
      const none = await simulate(device, solver, 0.5, 60);
      expect(none).toBe(0);
      solver.destroy();
    });
  });
});

// --- Outputs, edits, lifecycle -------------------------------------------------------------------

describe.each(KINDS)('shallow-water outputs and edits (%s)', (kind) => {
  it('publishes depth, velocity, foam, bed, slope and wetness for shading', async () => {
    const device = await getTestDevice();
    await expectValid(device, async () => {
      const solver = create(device, kind, { maxDepth: 1.3, wetLife: 1 }, 96);
      const bed = (u: number) => 0.2 * u;
      solver.setTerrain(bed);
      solver.setWater((u) => (u < 0.3 ? 1 - bed(u) : 0));
      await advance(device, solver, stepsFor(solver, 0.5));
      const out = await readOutputs(device, solver);
      const state = await readState(device, solver);
      const n = out.width * out.height;
      let flowing = 0;
      let foamy = 0;
      let maxFoam = 0;
      let slopeOnDry = 0;
      for (let i = 0; i < n; i++) {
        const depth = out.water[i * 4]!;
        // Depth and bed are the state to half-float precision.
        expect(Math.abs(depth - state.h[i]!)).toBeLessThan(2e-3 * Math.max(1, state.h[i]!));
        expect(Math.abs(out.ground[i * 4]! - state.b[i]!)).toBeLessThan(2e-3);
        if (depth > 0.05 && out.water[i * 4 + 1]! > 0.5) flowing++;
        maxFoam = Math.max(maxFoam, out.water[i * 4 + 3]!);
        if (out.water[i * 4 + 3]! > 0.2) foamy++;
        // Dry ground far from the water: the stored slope is the bed's (0.2 per 12 m = 1/60).
        const u = ((i % out.width) + 0.5) / out.width;
        if (depth === 0 && u > 0.9 && u < 0.96) slopeOnDry = out.ground[i * 4 + 1]!;
        expect(Number.isFinite(out.water[i * 4 + 1]!)).toBe(true);
      }
      expect(flowing).toBeGreaterThan(20); // water runs right: +x velocity
      expect(foamy).toBeGreaterThan(0); // the advancing front is foamy
      expect(maxFoam).toBeLessThanOrEqual(1.001);
      expect(slopeOnDry).toBeCloseTo(0.2 / 12, 3);
      // Ground under water is soaked; far dry ground is not.
      const wetAt = (x: number, y: number) => out.ground[(y * out.width + x) * 4 + 3]!;
      expect(wetAt(10, 32)).toBeGreaterThan(0.99);
      expect(wetAt(out.width - 4, 32)).toBeLessThan(0.05);
      // After the water is gone, the wetness fades over wetLife (1 s here).
      solver.setWater(() => 0);
      await advance(device, solver, 2);
      const gone = await readOutputs(device, solver);
      expect(gone.ground[(32 * out.width + 10) * 4 + 3]!).toBeGreaterThan(0.5);
      await advance(device, solver, stepsFor(solver, 3));
      const later = await readOutputs(device, solver);
      let soaked = 0;
      for (let i = 0; i < n; i++) if (later.ground[i * 4 + 3]! > 0.5) soaked++;
      expect(soaked).toBe(0);
      solver.destroy();
    });
  });

  it('pushes water, raises and digs the bed, and pulls a level zone to its level', async () => {
    const device = await getTestDevice();
    await expectValid(device, async () => {
      const solver = create(device, kind, { maxDepth: 1.5 }, 96);
      solver.setWater((u) => (u < 0.5 ? 0.5 : 0));
      await advance(device, solver, 1);
      const base = await readState(device, solver);

      // A push gives the water under it momentum along the push and leaves dry ground alone.
      const encoder = device.createCommandEncoder();
      solver.addSource({ x: 0.25, y: 0.5, radius: 1.5, pushX: 2, pushY: 0 });
      solver.addSource({ x: 0.85, y: 0.5, radius: 1.5, pushX: 2, pushY: 0 });
      solver.advance(encoder, 50);
      device.queue.submit([encoder.finish()]);
      const pushed = await readState(device, solver);
      const at = (st: State, u: number, v: number) =>
        Math.floor(v * st.height) * st.width + Math.floor(u * st.width);
      expect(pushed.hu[at(pushed, 0.25, 0.5)]!).toBeGreaterThan(0.01);
      expect(pushed.hu[at(pushed, 0.85, 0.5)]!).toBe(0);
      expect(Math.abs(pushed.volume - base.volume) / base.volume).toBeLessThan(1e-4);

      // A bed edit: peak 0.1 m / s for 50 steps on dry ground raises it by 0.1 s; digging goes the other way.
      const edit = create(device, kind, { maxDepth: 1.5, bedMax: 0.12 }, 96);
      const e2 = device.createCommandEncoder();
      edit.addSource({ x: 0.7, y: 0.5, radius: 1, bedRate: 0.1 });
      edit.addSource({ x: 0.2, y: 0.5, radius: 1, bedRate: -0.1 });
      edit.advance(e2, 50);
      device.queue.submit([e2.finish()]);
      const edited = await readState(device, edit);
      const seconds = 50 * edit.stepSeconds;
      const up = edited.b[at(edited, 0.7, 0.5)]!;
      const down = edited.b[at(edited, 0.2, 0.5)]!;
      expect(up).toBeGreaterThan(0.05 * seconds);
      expect(up).toBeLessThanOrEqual(0.12 + 1e-6);
      expect(down).toBeLessThan(-0.05 * seconds);
      expect(edited.b[at(edited, 0.95, 0.05)]!).toBe(0);
      edit.destroy();

      // A level zone on the left: the surface there is pulled toward 0.4 m, ground that is dry
      // stays dry above it and fills below it.
      const sea = create(device, kind, { maxDepth: 1.5, edges: 'wall' }, 96);
      sea.setTerrain((u) => (u < 0.15 ? -0.5 : 0.3 * (u - 0.15) * 4));
      await advance(device, sea, 1);
      for (let k = 0; k < 30; k++) {
        const e3 = device.createCommandEncoder();
        sea.addLevelZone({ x0: 0, y0: 0, x1: 0.08, y1: 1, level: 0.4, rate: 300 });
        sea.advance(e3, 40);
        device.queue.submit([e3.finish()]);
      }
      const filled = await readState(device, sea);
      const i0 = at(filled, 0.04, 0.5);
      expect(filled.b[i0]! + filled.h[i0]!).toBeGreaterThan(0.39);
      expect(filled.b[i0]! + filled.h[i0]!).toBeLessThan(0.41);
      // Water ran up the beach from the sea.
      expect(filled.h[at(filled, 0.2, 0.5)]!).toBeGreaterThan(0.02);
      sea.destroy();
      solver.destroy();
    });
  });

  it('limits the queue of sources and ignores bad ones', async () => {
    const device = await getTestDevice();
    await expectValid(device, async () => {
      const solver = create(device, kind, {}, 64);
      for (let i = 0; i < MAX_SOURCES; i++) {
        expect(solver.addSource({ x: 0.5, y: 0.5, radius: 0.5, volume: 0.01 })).toBe(true);
      }
      expect(solver.addSource({ x: 0.5, y: 0.5, radius: 0.5, volume: 0.01 })).toBe(false);
      expect(solver.pendingSources).toBe(MAX_SOURCES);
      const bad = create(device, kind, {}, 64);
      expect(bad.addSource({ x: Number.NaN, y: 0.5, radius: 0.5, volume: 1 })).toBe(false);
      expect(bad.addSource({ x: 5, y: 5, radius: 0.5, volume: 1 })).toBe(false); // off the grid
      expect(bad.pendingSources).toBe(0);
      expect(() => bad.setTerrain(new Float32Array(10))).toThrow(RangeError);
      // Sources do not carry over to the next update.
      const encoder = device.createCommandEncoder();
      solver.update(encoder, 1 / 60);
      device.queue.submit([encoder.finish()]);
      expect(solver.pendingSources).toBe(0);
      solver.destroy();
      bad.destroy();
    });
  });

  it('re-creates cleanly when the resolution or shape changes, and destroys cleanly', async () => {
    const device = await getTestDevice();
    await expectValid(device, async () => {
      const solver = create(device, kind, { maxDepth: 1.3 }, 64);
      solver.setWater((u) => (u < 0.5 ? 1 : 0));
      await advance(device, solver, 100);
      const g0 = solver.generation;
      const grid0 = solver.grid;
      expect(solver.setResolution(64)).toBe(false);
      expect(solver.generation).toBe(g0);
      expect(solver.setResolution(128)).toBe(true);
      expect(solver.generation).toBe(g0 + 1);
      expect(solver.grid).not.toBe(grid0);
      expect(solver.grid.width).toBe(128);
      expect(solver.grid.height).toBe(85);
      expect(solver.grid.cellSize).toBeCloseTo(12 / 128, 6);
      expect(solver.grid.extent[0]).toBeCloseTo(12, 6);
      expect(solver.grid.extent[1]).toBeCloseTo(12 / (128 / 85), 6);
      // Cleared, and it runs.
      await advance(device, solver, 20);
      const cleared = await readState(device, solver);
      expect(cleared.volume).toBe(0);
      expect(solver.setAspect(1)).toBe(true);
      expect(solver.grid.width).toBe(128);
      expect(solver.grid.height).toBe(128);
      solver.setWater(() => 0.2);
      await advance(device, solver, 10);
      expect((await readState(device, solver)).volume).toBeGreaterThan(1);
      // reset() empties it again.
      const encoder = device.createCommandEncoder();
      solver.reset(encoder);
      device.queue.submit([encoder.finish()]);
      expect((await readState(device, solver)).volume).toBe(0);
      const out = await readOutputs(device, solver);
      expect(maxAbs(out.water)).toBe(0);
      // Destroy twice, and use after destroy, without an error.
      solver.destroy();
      solver.destroy();
      const after = device.createCommandEncoder();
      expect(solver.update(after, 1 / 60)).toBe(0);
      device.queue.submit([after.finish()]);
    });
  });
});

describe('the two implementations', () => {
  it('compute and render give the same water', async () => {
    const device = await getTestDevice();
    await expectValid(device, async () => {
      const states: State[] = [];
      const outputs: Awaited<ReturnType<typeof readOutputs>>[] = [];
      for (const kind of KINDS) {
        const solver = create(device, kind, { maxDepth: 1.3, friction: 0.03 }, 96);
        solver.setTerrain(hills);
        solver.setWater((u, v) => (u < 0.35 ? Math.max(1.2 - hills(u, v), 0) : 0));
        await simulate(device, solver, 1.5, 60, () =>
          solver.addSource({ x: 0.6, y: 0.3, radius: 0.6, volume: 0.2, pushX: 1 }),
        );
        states.push(await readState(device, solver));
        outputs.push(await readOutputs(device, solver));
        solver.destroy();
      }
      const [a, b] = states as [State, State];
      let worst = 0;
      for (let i = 0; i < a.h.length; i++) worst = Math.max(worst, Math.abs(a.h[i]! - b.h[i]!));
      expect(worst).toBeLessThan(1e-4);
      expect(Math.abs(a.volume - b.volume) / a.volume).toBeLessThan(1e-5);
      const [oa, ob] = outputs as [(typeof outputs)[0], (typeof outputs)[0]];
      let outWorst = 0;
      for (let i = 0; i < oa.water.length; i++)
        outWorst = Math.max(outWorst, Math.abs(oa.water[i]! - ob.water[i]!));
      expect(outWorst).toBeLessThan(5e-3);
    });
  });

  it('caches pipelines per device: a second solver compiles nothing new', async () => {
    const device = await getTestDevice();
    await expectValid(device, async () => {
      const a = createShallowWater({ device, kind: 'compute', resolution: 32 });
      const b = createShallowWater({ device, kind: 'compute', resolution: 48 });
      const encoder = device.createCommandEncoder();
      a.advance(encoder, 2);
      b.advance(encoder, 2);
      device.queue.submit([encoder.finish()]);
      await device.queue.onSubmittedWorkDone();
      a.destroy();
      b.destroy();
    });
  });
});

// --- The demo effect ------------------------------------------------------------------------------

describe('the shallow-water effect', () => {
  const WIDTH = 96;
  const HEIGHT = 64;

  function canvasWithEvents() {
    const canvas = Object.assign(new EventTarget(), {
      width: WIDTH,
      height: HEIGHT,
      style: {},
      getBoundingClientRect: () => ({ left: 0, top: 0, width: WIDTH, height: HEIGHT }),
      setPointerCapture() {},
      releasePointerCapture() {},
    }) as unknown as HTMLCanvasElement;
    const send = (type: string, x: number, y: number, extra: Record<string, unknown> = {}) =>
      canvas.dispatchEvent(
        Object.assign(new Event(type), {
          pointerId: 1,
          pointerType: 'mouse',
          clientX: x * WIDTH,
          clientY: y * HEIGHT,
          button: 0,
          shiftKey: false,
          ...extra,
        }),
      );
    return { canvas, send };
  }

  it('restarts on a preset change, a flipped restart flag, a new solver and a new grid, and every tool runs', async () => {
    const device = await getTestDevice();
    const target = device.createTexture({
      size: [WIDTH, HEIGHT],
      format: 'rgba8unorm',
      usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC,
    });
    await expectValid(device, async () => {
      const effect = await loadEffect('shallow-water');
      const params = defaultParams(effect.params);
      const optionsOf = (key: string): readonly string[] =>
        (effect.params[key] as { options: readonly string[] }).options;
      const { canvas, send } = canvasWithEvents();
      const instance = await effect.create({
        device,
        format: 'rgba8unorm',
        canvas,
        params,
        profiler: noProfiler,
        hdr: false,
      });
      instance.resize(WIDTH, HEIGHT, 1);
      let frame = 0;
      const run = (count: number) => {
        for (let i = 0; i < count; i++, frame++) {
          const encoder = device.createCommandEncoder();
          instance.frame({ time: frame / 60, delta: 1 / 60, frame }, encoder, target.createView());
          device.queue.submit([encoder.finish()]);
        }
      };
      run(30);

      // Every pointer tool, with a press, a drag and a shift press.
      for (const tool of optionsOf('tool')) {
        params.tool = tool;
        send('pointerenter', 0.5, 0.6);
        send('pointermove', 0.5, 0.6);
        send('pointerdown', 0.5, 0.6);
        run(3);
        send('pointermove', 0.6, 0.5);
        run(3);
        send('pointerup', 0.6, 0.5);
        send('pointerdown', 0.4, 0.7, { shiftKey: true });
        run(2);
        send('pointerup', 0.4, 0.7);
        run(20);
      }

      // Scenarios, the restart flag, the solver variant and the grid.
      for (const scenario of optionsOf('scenario')) {
        params.scenario = scenario;
        run(40);
      }
      params.restart = !params.restart;
      run(5);
      params.solver = 'render';
      run(5);
      params.gridResolution = '128';
      run(5);
      params.gridResolution = '384';
      run(5);
      params.solver = 'compute';
      params.gridResolution = '192';
      run(5);
      for (const view of optionsOf('view')) {
        params.view = view;
        run(2);
      }
      // Extremes.
      Object.assign(params, { gravity: 25, friction: 0, speed: 2.5, rain: 30, view: 'beauty' });
      run(60);
      Object.assign(params, { gravity: 2, friction: 0.15, speed: 0.1, rain: 0 });
      run(20);
      // A resize re-creates the depth buffer: draw into a target of the new size.
      const big = device.createTexture({
        size: [WIDTH * 2, HEIGHT * 2],
        format: 'rgba8unorm',
        usage: GPUTextureUsage.RENDER_ATTACHMENT,
      });
      instance.resize(WIDTH * 2, HEIGHT * 2, 1);
      for (let i = 0; i < 3; i++, frame++) {
        const encoder = device.createCommandEncoder();
        instance.frame({ time: frame / 60, delta: 1 / 60, frame }, encoder, big.createView());
        device.queue.submit([encoder.finish()]);
      }
      big.destroy();
      instance.destroy();
    });
    const pixels = await readTexture(device, target);
    expect(new Set(pixels).size).toBeGreaterThan(8);
    target.destroy();
  });
});

// --- shallow-water.wgsl --------------------------------------------------------------------------

describe('shallow-water.wgsl decode helpers', () => {
  const include = '#include "shallow-water.wgsl"\n';

  it('compiles on its own', async () => {
    const device = await getTestDevice();
    const { diagnostics } = await checkShader(device, {
      label: 'swe-alone.wgsl',
      code: include,
    });
    expect(diagnostics.map(formatDiagnostic)).toEqual([]);
  });

  it('decodes the water and ground texels', async () => {
    const [depth, vx, vz, speed, foam, froude, dry, bed, sx, sz, wet, surface] = await scalars(
      include,
      [
        'swe_depth(vec4f(-0.1, 0.0, 0.0, 0.0))',
        'swe_velocity(vec4f(0.5, 3.0, -4.0, 0.0)).x',
        'swe_velocity_world(vec4f(0.5, 3.0, -4.0, 0.0)).z',
        'swe_speed(vec4f(0.5, 3.0, -4.0, 0.0))',
        'swe_foam(vec4f(0.5, 0.0, 0.0, 1.7))',
        'swe_froude(vec4f(1.0, 3.13, 0.0, 0.0), 9.81)',
        'swe_froude(vec4f(0.0, 3.13, 0.0, 0.0), 9.81)',
        'swe_bed(vec4f(0.7, 0.1, 0.2, 0.3))',
        'swe_surface_slope(vec4f(0.7, 0.1, 0.2, 0.3)).x',
        'swe_surface_slope(vec4f(0.7, 0.1, 0.2, 0.3)).y',
        'swe_ground_wetness(vec4f(0.7, 0.1, 0.2, 1.4))',
        'swe_surface(vec4f(0.4, 0.0, 0.0, 0.0), vec4f(0.7, 0.0, 0.0, 0.0))',
      ],
    );
    expect(depth).toBe(0);
    expect(vx).toBe(3);
    expect(vz).toBe(-4);
    expect(speed).toBeCloseTo(5, 5);
    expect(foam).toBe(1);
    expect(froude).toBeCloseTo(1, 2);
    expect(dry).toBe(0);
    expect(bed).toBeCloseTo(0.7, 6);
    expect(sx).toBeCloseTo(0.1, 6);
    expect(sz).toBeCloseTo(0.2, 6);
    expect(wet).toBe(1);
    expect(surface).toBeCloseTo(1.1, 6);
  });

  it('has a soft, monotonic wet mask and unit normals that point up on flat water', async () => {
    const depths = [-1, 0, 0.001, 0.005, 0.01, 0.02, 1];
    const out = await scalars(
      include,
      depths.map((d) => `swe_wet_mask(${d}, 0.01)`),
    );
    expect(out[0]).toBe(0);
    expect(out[1]).toBe(0);
    expect(out[4]).toBe(1);
    expect(out[6]).toBe(1);
    for (let i = 1; i < out.length; i++) expect(out[i]!).toBeGreaterThanOrEqual(out[i - 1]! - 1e-6);
    expect(out[3]!).toBeGreaterThan(0.2);
    expect(out[3]!).toBeLessThan(0.8);
    const hard = await scalars(include, ['swe_wet_mask(0.0, 0.0)', 'swe_wet_mask(0.001, 0.0)']);
    expect(hard).toEqual([0, 1]);

    const rows = rowsOf([
      [0, 0, 0, 0],
      [0.1, -0.2, 0, 0],
      [1, 1, 0, 0],
    ]);
    const normals = await gpuF32(
      `${include} fn f(x: vec4f) -> vec4f { let n = swe_surface_normal(vec4f(0.0, x.x, x.y, 0.0), 1.0); return vec4f(n, length(n)); }`,
      rows,
    );
    expect(Math.abs(normals[0]!) + Math.abs(normals[2]!)).toBe(0);
    expect(normals[1]).toBe(1);
    // A surface rising to +x tilts its normal toward -x; uv-down is +z, so a rise along +z tilts to -z.
    expect(normals[4]!).toBeLessThan(0);
    expect(normals[4]!).toBeCloseTo(-0.1 / Math.hypot(0.1, 1, 0.2), 4);
    expect(normals[5]!).toBeCloseTo(1 / Math.hypot(0.1, 1, 0.2), 4);
    expect(normals[6]!).toBeCloseTo(0.2 / Math.hypot(0.1, 1, 0.2), 4);
    for (let r = 0; r < 3; r++) expect(normals[r * 4 + 3]!).toBeCloseTo(1, 5);
    expect(lane(normals, 3)).toHaveLength(3);
  });

  it('derives the bed normal from four heights', async () => {
    const [nx, ny, nz, len] = await scalars(include, [
      'swe_terrain_normal(0.0, 0.2, 0.0, 0.0, 1.0).x',
      'swe_terrain_normal(0.0, 0.2, 0.0, 0.0, 1.0).y',
      'swe_terrain_normal(0.0, 0.2, 0.0, 0.0, 1.0).z',
      'length(swe_terrain_normal(0.0, 0.2, 0.3, 0.0, 1.0))',
    ]);
    // A rise of 0.2 over two metres east: slope 0.1.
    expect(nx).toBeCloseTo(-0.1 / Math.hypot(0.1, 1), 5);
    expect(ny).toBeCloseTo(1 / Math.hypot(0.1, 1), 5);
    // The bed lies flat east-west, so nz is zero either way; software GPUs may return -0.
    expect(nz).toBeCloseTo(0, 5);
    expect(len).toBeCloseTo(1, 5);
  });
});

// --- Cost study (opt in: SHALLOW_WATER_BENCH=1 npx vitest run tests/shallow-water.test.ts -t "cost study") ---

const benchEnabled = Boolean(
  (globalThis as { process?: { env: Record<string, string | undefined> } }).process?.env
    .SHALLOW_WATER_BENCH,
);

describe.skipIf(!benchEnabled)('cost study', () => {
  it('times GPU milliseconds per step for each variant and resolution', async () => {
    const adapter = await navigator.gpu.requestAdapter();
    const timed = Boolean(adapter?.features.has('timestamp-query'));
    const device = timed
      ? await adapter!.requestDevice({ requiredFeatures: ['timestamp-query'] })
      : await getTestDevice();
    const lines: string[] = [`timestamp queries: ${timed}`];

    /** GPU time from the first pass starting to the last pass ending, milliseconds. */
    async function gpuSpan(kind: ShallowWaterKind, cells: number, steps: number): Promise<number> {
      const querySet = device.createQuerySet({ type: 'timestamp', count: 8 });
      let next = 0;
      const profiler = {
        pass: () => {
          const begin = next;
          next += 2;
          return { querySet, beginningOfPassWriteIndex: begin, endOfPassWriteIndex: begin + 1 };
        },
      };
      const solver = createShallowWater({
        device,
        kind,
        resolution: cells,
        aspect: 1.5,
        params: { friction: 0.03 },
        profiler,
      });
      solver.setWater((u) => (u < 0.4 ? 1 : 0));
      const resolve = device.createBuffer({
        size: 64,
        usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC,
      });
      const read = device.createBuffer({
        size: 64,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
      });
      const encoder = device.createCommandEncoder();
      solver.advance(encoder, steps);
      encoder.resolveQuerySet(querySet, 0, 8, resolve, 0);
      encoder.copyBufferToBuffer(resolve, 0, read, 0, 64);
      device.queue.submit([encoder.finish()]);
      await read.mapAsync(GPUMapMode.READ);
      const stamps = new BigUint64Array(read.getMappedRange().slice(0));
      read.unmap();
      solver.destroy();
      const used = Array.from(stamps.slice(0, next)).filter((t) => t > 0n);
      const first = used.reduce((a, b) => (a < b ? a : b));
      const last = used.reduce((a, b) => (a > b ? a : b));
      return Number(last - first) / 1e6;
    }

    for (const cells of [128, 256, 512, 1024]) {
      for (const kind of KINDS) {
        const median = async (steps: number) => {
          const samples: number[] = [];
          for (let i = 0; i < 9; i++) samples.push(timed ? await gpuSpan(kind, cells, steps) : 0);
          samples.sort((a, b) => a - b);
          return samples[4]!;
        };
        const low = await median(10);
        const high = await median(210);
        const perStep = (high - low) / 200;
        const grid = computeShallowWaterGrid(cells, 1.5);
        lines.push(
          `${grid.width}x${grid.height} ${kind}: span(10)=${low.toFixed(3)} ms, span(210)=${high.toFixed(3)} ms, ${perStep.toFixed(4)} ms per step, ${((perStep * 1e6) / (grid.width * grid.height)).toFixed(3)} ns per cell`,
        );
      }
    }
    console.log(`\n${lines.join('\n')}`);
  });
});
