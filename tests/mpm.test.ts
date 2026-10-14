import { describe, expect, it, vi } from 'vitest';
import { listEffects, loadEffect } from '../src/effects/registry.ts';
import { defaultParams } from '../src/effects/types.ts';
import { noProfiler } from '../src/gpu/profiler.ts';
import {
  blockLattice,
  createMpm,
  DEFAULT_MPM_PARAMS,
  MPM_FIXED_RANGE,
  MPM_FIXED_SCALE,
  MPM_PASSES,
  MPM_RING_STEPS,
  MPM_STORAGE_BINDINGS,
  mpmStability,
  type MpmParams,
  type MpmPassName,
  type MpmScene,
  type MpmSolver,
} from '../src/gpu/sim/mpm.ts';
import clearSource from '../src/gpu/sim/mpm-clear.wgsl?raw';
import coreSource from '../src/gpu/sim/mpm-core.wgsl?raw';
import densitySource from '../src/gpu/sim/mpm-density.wgsl?raw';
import g2pSource from '../src/gpu/sim/mpm-g2p.wgsl?raw';
import gridSource from '../src/gpu/sim/mpm-grid.wgsl?raw';
import p2g1Source from '../src/gpu/sim/mpm-p2g1.wgsl?raw';
import p2g1sSource from '../src/gpu/sim/mpm-p2g1s.wgsl?raw';
import p2g2Source from '../src/gpu/sim/mpm-p2g2.wgsl?raw';
import p2g2sSource from '../src/gpu/sim/mpm-p2g2s.wgsl?raw';
import { expectValid, getTestDevice, readTexture } from './gpu.ts';
import { gpuF32, lane, rowsOf, scalars } from './wgsl-eval.ts';

// Story 7.5: MLS-MPM. Everything here runs the real shaders on the GPU (Dawn) and compares with arithmetic done in
// JavaScript or with a conservation law. Grids and counts are small so the suite stays fast; the cost study at the
// end is skipped unless MPM_STUDY names a file.

vi.setConfig({ testTimeout: 120_000 });

// --- Helpers ---------------------------------------------------------------------------------------------------------

async function submit(device: GPUDevice, record: (encoder: GPUCommandEncoder) => void) {
  const encoder = device.createCommandEncoder();
  record(encoder);
  device.queue.submit([encoder.finish()]);
  await device.queue.onSubmittedWorkDone();
}

interface Options {
  grid?: number | [number, number, number];
  cellSize?: number;
  capacity?: number;
  params?: Partial<MpmParams>;
}

function make(device: GPUDevice, options: Options = {}): MpmSolver {
  return createMpm({
    device,
    grid: options.grid ?? 32,
    cellSize: options.cellSize ?? 0.05,
    capacity: options.capacity ?? 20000,
    params: options.params,
  });
}

/**
 * Advances `seconds` of display time in frames of 1/hz. Frames are batched into one submit (up to about 900 steps:
 * the solver's uniform ring holds 2048) so a slow shared GPU is waited on once per batch, not once per frame.
 */
async function run(device: GPUDevice, solver: MpmSolver, seconds: number, hz = 60) {
  const frames = Math.round(seconds * hz);
  const perFrame = Math.max(1, Math.ceil(1 / hz / solver.effective.dt));
  const batch = Math.max(1, Math.floor(900 / perFrame));
  for (let first = 0; first < frames; first += batch) {
    await submit(device, (encoder) => {
      for (let frame = first; frame < Math.min(frames, first + batch); frame++) {
        solver.update(encoder, 1 / hz);
      }
    });
  }
}

async function stepPasses(
  device: GPUDevice,
  solver: MpmSolver,
  passes: readonly MpmPassName[],
  count = 1,
) {
  await submit(device, (encoder) => solver.step(encoder, { passes, count }));
}

interface State {
  x: Float32Array;
  v: Float32Array;
  n: number;
}

async function state(solver: MpmSolver): Promise<State> {
  const x = await solver.system.read('position');
  const v = await solver.system.read('velocity');
  return { x, v, n: solver.count };
}

function column(data: Float32Array, n: number, axis: number): number[] {
  const out: number[] = [];
  for (let i = 0; i < n; i++) out.push(data[i * 4 + axis]!);
  return out;
}

function quantile(values: number[], q: number): number {
  const sorted = [...values].sort((a, b) => a - b);
  return sorted[Math.min(sorted.length - 1, Math.floor(q * sorted.length))]!;
}

const mean = (values: number[]) => values.reduce((a, b) => a + b, 0) / Math.max(values.length, 1);

function allFinite(data: Float32Array, n: number, stride = 4): boolean {
  for (let i = 0; i < n; i++) {
    for (let c = 0; c < Math.min(stride, 3); c++)
      if (!Number.isFinite(data[i * stride + c]!)) return false;
  }
  return true;
}

function kineticEnergy(s: State): number {
  let ke = 0;
  for (let i = 0; i < s.n; i++) {
    ke += 0.5 * (s.v[i * 4]! ** 2 + s.v[i * 4 + 1]! ** 2 + s.v[i * 4 + 2]! ** 2);
  }
  return ke / Math.max(s.n, 1);
}

/** A pool on the floor of a box: `depth` metres of fluid in [lo, hi] in x and z. */
function pool(
  solver: MpmSolver,
  depth: number,
  lo = 0.4,
  hi = 1.2,
  top = 1.5,
  jitter = 0.15,
): MpmScene {
  const floor = 0.1;
  return {
    bounds: { min: [lo, floor, lo], max: [hi, top, hi] },
    blocks: [{ shape: 'box', min: [lo, floor, lo], max: [hi, floor + depth, hi], jitter }],
    gravity: solver.params.gravity,
  };
}

const damBreak: MpmScene = {
  blocks: [{ shape: 'box', min: [0.1, 0.1, 0.1], max: [0.7, 0.8, 1.2] }],
};

// --- The pure maths --------------------------------------------------------------------------------------------------

describe('mpm-core.wgsl', () => {
  it('B-spline weights are positive, sum to 1 and reproduce the particle position', async () => {
    const fxs: number[] = [];
    for (let i = 0; i < 64; i++) fxs.push(0.5 + (i / 63) * 0.999);
    const out = await gpuF32(
      `${coreSource}
       fn f(x: vec4f) -> vec4f {
         let W = mpm_weight_matrix(vec3f(x.x, x.x, x.x));
         let w = vec3f(W[0].x, W[1].x, W[2].x);
         let sum = w.x + w.y + w.z;
         // first moment: sum_o w_o * o = fx (the position, in cells from the base)
         let moment = w.y * 1.0 + w.z * 2.0;
         // second moment about the position: sum_o w_o (o - fx)^2 = 1/4 + (x terms) -> constant 0.25 + ... check below
         let second = w.x * x.x * x.x + w.y * (1.0 - x.x) * (1.0 - x.x) + w.z * (2.0 - x.x) * (2.0 - x.x);
         return vec4f(sum, moment, min(min(w.x, w.y), w.z), second);
       }`,
      rowsOf(fxs.map((fx) => [fx])),
    );
    const sums = lane(out, 0);
    const moments = lane(out, 1);
    const mins = lane(out, 2);
    const seconds = lane(out, 3);
    for (let i = 0; i < fxs.length; i++) {
      expect(sums[i]).toBeCloseTo(1, 6);
      expect(moments[i]).toBeCloseTo(fxs[i]!, 6);
      expect(mins[i]).toBeGreaterThanOrEqual(-1e-6);
      // The quadratic B-spline's variance is 1/4 (so that C = 4/h^2 sum w v d recovers a gradient exactly).
      expect(seconds[i]).toBeCloseTo(0.25, 5);
    }
  });

  it('the kernel share above a wall is 0 beyond, 1 behind, 1/2 at the node, monotone and smooth', async () => {
    const ts: number[] = [];
    for (let i = -17; i <= 17; i++) ts.push(i / 10);
    const values = await scalars(
      coreSource,
      ts.map((t) => `mpm_kernel_above(${t.toFixed(3)})`),
    );
    expect(values[ts.indexOf(-1.7)]).toBe(1);
    expect(values[ts.indexOf(1.7)]).toBe(0);
    expect(values[ts.indexOf(0)]).toBeCloseTo(0.5, 6);
    for (let i = 1; i < values.length; i++)
      expect(values[i]!).toBeLessThanOrEqual(values[i - 1]! + 1e-6);
    // Its derivative is the weight function itself: at t = 0 the weight is 0.75.
    const eps = 1e-3;
    const [lo, hi] = await scalars(coreSource, [
      `mpm_kernel_above(${-eps})`,
      `mpm_kernel_above(${eps})`,
    ]);
    expect((lo! - hi!) / (2 * eps)).toBeCloseTo(0.75, 2);
  });

  it('fixed point rounds to nearest, round-trips, saturates and keeps a lone slow particle', async () => {
    const [half, roundTrip, slow, saturated, negative] = await scalars(coreSource, [
      'f32(mpm_to_fixed(0.5 / 1048576.0 * 0.999))',
      'mpm_from_fixed(mpm_to_fixed(123.456789)) - 123.456789',
      // a particle at 1e-3 m/s deposits 0.4219 of its momentum on its central node: momentum 4.2e-4
      'mpm_from_fixed(mpm_to_fixed(0.421875 * 0.001)) / (0.421875 * 0.001)',
      'f32(mpm_to_fixed(1.0e12))',
      'f32(mpm_to_fixed(-1.0e12))',
    ]);
    expect(half).toBe(0);
    expect(Math.abs(roundTrip!)).toBeLessThan(0.5 / MPM_FIXED_SCALE + 1e-5);
    expect(Math.abs(slow! - 1)).toBeLessThan(0.002);
    expect(saturated).toBeGreaterThan(2.1e9);
    expect(negative).toBeLessThan(-2.1e9);
    expect(MPM_FIXED_RANGE).toBeCloseTo(2047.99, 1);
  });

  it('signed distance of the collider shapes', async () => {
    const [sphere, sphereInside, box, boxInside, plane, nx] = await scalars(coreSource, [
      'mpm_collider_sdf(vec4f(1.0, 2.0, 3.0, 1.0), vec4f(0.5, 0.0, 0.0, 0.0), vec3f(1.0, 2.0, 4.0)).w',
      'mpm_collider_sdf(vec4f(1.0, 2.0, 3.0, 1.0), vec4f(0.5, 0.0, 0.0, 0.0), vec3f(1.0, 2.1, 3.0)).w',
      'mpm_collider_sdf(vec4f(0.0, 0.0, 0.0, 2.0), vec4f(1.0, 1.0, 1.0, 0.0), vec3f(3.0, 0.0, 0.0)).w',
      'mpm_collider_sdf(vec4f(0.0, 0.0, 0.0, 2.0), vec4f(1.0, 1.0, 1.0, 0.0), vec3f(0.2, 0.9, 0.1)).w',
      'mpm_collider_sdf(vec4f(0.0, 1.0, 0.0, 3.0), vec4f(0.0, 2.0, 0.0, 0.0), vec3f(5.0, 4.0, 0.0)).w',
      'mpm_collider_sdf(vec4f(0.0, 0.0, 0.0, 2.0), vec4f(1.0, 1.0, 1.0, 0.0), vec3f(3.0, 0.0, 0.0)).x',
    ]);
    expect(sphere).toBeCloseTo(0.5, 6);
    expect(sphereInside).toBeCloseTo(0.1 - 0.5, 5);
    expect(box).toBeCloseTo(2, 6);
    expect(boxInside).toBeCloseTo(-0.1, 5);
    expect(plane).toBeCloseTo(3, 6);
    expect(nx).toBeCloseTo(1, 6);
  });
});

describe('stability rule', () => {
  const base = {
    soundSpeed: 12,
    maxSpeed: 10,
    dt: 0,
    courant: 0.5,
    viscosity: 0,
    viscosityLimit: 0.1,
  };

  it('derives the step from the stiffness: dt = courant h / (c + v_max)', () => {
    const e = mpmStability(0.05, base);
    expect(e.dt).toBeCloseTo((0.5 * 0.05) / 22, 12);
    expect(e.clamped).toBe(false);
    expect(e.soundSpeed).toBe(12);
    // A particle moves at most v_max dt < courant h per step.
    expect(10 * e.dt).toBeLessThan(0.5 * 0.05);
  });

  it('keeps a requested step and reduces the stiffness to fit it', () => {
    const long = mpmStability(0.05, { ...base, dt: 0.002 });
    expect(long.dt).toBe(0.002);
    expect(long.soundSpeed).toBeCloseTo((0.5 * 0.05) / 0.002 - 10, 9);
    expect(long.clamped).toBe(true);
    // The stable step for the reduced stiffness is exactly the requested one.
    expect(long.stableDt).toBeCloseTo(0.002, 9);
    // A shorter step than needed is fine and leaves the stiffness alone.
    const short = mpmStability(0.05, { ...base, dt: 0.0001 });
    expect(short.dt).toBe(0.0001);
    expect(short.soundSpeed).toBe(12);
    expect(short.clamped).toBe(false);
  });

  it('shortens the step when even the softest stiffness does not fit', () => {
    const e = mpmStability(0.05, { ...base, dt: 0.05 });
    expect(e.soundSpeed).toBe(1);
    expect(e.dt).toBeLessThan(0.05);
    expect(e.dt).toBeCloseTo((0.5 * 0.05) / 11, 9);
  });

  it('limits the viscosity to what the step can diffuse', () => {
    const e = mpmStability(0.05, { ...base, viscosity: 100 });
    expect(e.viscosity).toBeCloseTo((0.1 * 0.05 * 0.05) / e.dt, 9);
    expect(e.clamped).toBe(true);
  });
});

describe('bindings', () => {
  const sources: Record<string, string> = {
    clear: clearSource,
    p2g1: p2g1Source,
    p2g1s: p2g1sSource,
    p2g2: p2g2Source,
    density: densitySource,
    p2g2s: p2g2sSource,
    grid: gridSource,
    g2p: g2pSource,
  };

  it('every shader stays inside 8 storage buffers, as the table says', async () => {
    const device = await getTestDevice();
    expect(device.limits.maxStorageBuffersPerShaderStage).toBeGreaterThanOrEqual(8);
    for (const [name, source] of Object.entries(sources)) {
      const storage = (source.match(/var<storage/g) ?? []).length;
      expect(storage, name).toBeLessThanOrEqual(8);
      expect(storage, name).toBe(MPM_STORAGE_BINDINGS[name as keyof typeof MPM_STORAGE_BINDINGS]);
    }
  });

  it('refuses a grid that does not fit a storage binding and a bad size', async () => {
    const device = await getTestDevice();
    expect(() => createMpm({ device, grid: 400, capacity: 1000 })).toThrow(/device limit/);
    expect(() => createMpm({ device, grid: 4, capacity: 1000 })).toThrow(/at least 8/);
    expect(() => createMpm({ device, grid: 16, capacity: 5_000_000 })).toThrow(/dispatch/);
  });
});

// --- Particle to grid ------------------------------------------------------------------------------------------------

describe('P2G', () => {
  async function scatter(sorted: boolean, seed = 1) {
    const device = await getTestDevice();
    const solver = make(device, { grid: 24, capacity: 4000, params: { sorted } });
    // Random positions, velocities and affine matrices well inside the box.
    const n = 3000;
    let s = seed >>> 0;
    const random = () => {
      s = (Math.imul(s, 1664525) + 1013904223) >>> 0;
      return s / 4294967296;
    };
    const positions = new Float32Array(n * 4);
    const velocities = new Float32Array(n * 4);
    const a0 = new Float32Array(n * 4);
    const a1 = new Float32Array(n * 4);
    const a2 = new Float32Array(n * 4);
    for (let i = 0; i < n; i++) {
      positions.set([0.3 + random() * 0.6, 0.3 + random() * 0.6, 0.3 + random() * 0.6, 0], i * 4);
      velocities.set([random() * 4 - 2, random() * 4 - 2, random() * 4 - 2, 0], i * 4);
      a0.set([random() - 0.5, random() - 0.5, random() - 0.5, 0], i * 4);
      a1.set([random() - 0.5, random() - 0.5, random() - 0.5, 0], i * 4);
      a2.set([random() - 0.5, random() - 0.5, random() - 0.5, 0], i * 4);
    }
    solver.system.set({ positions, velocities, extras: { affine0: a0, affine1: a1, affine2: a2 } });
    await stepPasses(device, solver, ['clear', 'sort', 'p2g1']);
    const grid = await solver.readGrid();
    const after = await state(solver);
    return { device, solver, grid, after, n };
  }

  it('conserves mass and momentum within the fixed-point quantisation', async () => {
    for (const sorted of [false, true]) {
      const { solver, grid, after, n } = await scatter(sorted);
      const mass = grid.mass.reduce((a, b) => a + b, 0);
      // Round to nearest per addend: the error per particle is a random walk of 27 terms of at most half a unit
      // (9.5e-7 each); allow 1e-5 of one particle's mass per particle, far above that and far below any bug.
      expect(Math.abs(mass - n)).toBeLessThan(n * 1e-5);
      let px = 0;
      let py = 0;
      let pz = 0;
      for (let i = 0; i < grid.mass.length; i++) {
        px += grid.momentum[i * 3]!;
        py += grid.momentum[i * 3 + 1]!;
        pz += grid.momentum[i * 3 + 2]!;
      }
      // sum_i w_ip C_p (x_i - x_p) = 0 (the first moment), so the affine part drops out of the total.
      let vx = 0;
      let vy = 0;
      let vz = 0;
      for (let i = 0; i < n; i++) {
        vx += after.v[i * 4]!;
        vy += after.v[i * 4 + 1]!;
        vz += after.v[i * 4 + 2]!;
      }
      const tolerance = n * 2e-5;
      expect(Math.abs(px - vx)).toBeLessThan(tolerance);
      expect(Math.abs(py - vy)).toBeLessThan(tolerance);
      expect(Math.abs(pz - vz)).toBeLessThan(tolerance);
      solver.destroy();
    }
  });

  it('the sorted scatter gives the same grid as the plain one', async () => {
    const plain = await scatter(false, 7);
    const sorted = await scatter(true, 7);
    let worst = 0;
    for (let i = 0; i < plain.grid.mass.length; i++) {
      worst = Math.max(worst, Math.abs(plain.grid.mass[i]! - sorted.grid.mass[i]!));
    }
    for (let i = 0; i < plain.grid.momentum.length; i++) {
      worst = Math.max(worst, Math.abs(plain.grid.momentum[i]! - sorted.grid.momentum[i]!));
    }
    // One scatters every term and rounds each; the other sums in registers and rounds once: a few units apart.
    expect(worst).toBeLessThan(3e-4);
    plain.solver.destroy();
    sorted.solver.destroy();
  });

  it('holds the densest configuration allowed and fails safe beyond it', async () => {
    const device = await getTestDevice();
    // 2000 particles on one point at a node's centre: 0.4219 * 2000 = 844 mass units on that node, under the
    // 2047.99 the format holds, and the momentum 844 * 3 m/s = 2531... which is NOT: keep the speed at 2 (1688).
    const solver = make(device, { grid: 16, capacity: 6000, params: { sorted: true } });
    const count = 2000;
    const positions = new Float32Array(count * 4);
    const velocities = new Float32Array(count * 4);
    for (let i = 0; i < count; i++) {
      positions.set([0.4, 0.4, 0.4, 0], i * 4); // x / h = 8.0: a node's centre
      velocities.set([2, -2, 2, 0], i * 4);
    }
    solver.system.set({ positions, velocities });
    await stepPasses(device, solver, ['clear', 'sort', 'p2g1']);
    const grid = await solver.readGrid();
    const peak = Math.max(...grid.mass);
    expect(peak).toBeCloseTo(count * 0.421875, 1);
    expect(peak).toBeLessThan(MPM_FIXED_RANGE);
    expect(grid.mass.reduce((a, b) => a + b, 0)).toBeCloseTo(count, 1);
    // Beyond the range: 5000 particles would be 2109 units. The sum wraps; the grid pass treats a negative mass
    // as empty and clamps what it decodes, so the next stage sees finite, bounded velocities.
    const big = 5000;
    const solver2 = make(device, { grid: 16, capacity: 6000, params: { sorted: true } });
    const positions2 = new Float32Array(big * 4);
    const velocities2 = new Float32Array(big * 4);
    for (let i = 0; i < big; i++) {
      positions2.set([0.4, 0.4, 0.4, 0], i * 4);
      velocities2.set([2, -2, 2, 0], i * 4);
    }
    solver2.system.set({ positions: positions2, velocities: velocities2 });
    await stepPasses(device, solver2, ['clear', 'sort', 'p2g1', 'grid']);
    const gv = await solver2.readGridVelocity();
    let finite = true;
    let fastest = 0;
    for (let i = 0; i < gv.length / 4; i++) {
      const speed = Math.hypot(gv[i * 4]!, gv[i * 4 + 1]!, gv[i * 4 + 2]!);
      if (!Number.isFinite(speed)) finite = false;
      fastest = Math.max(fastest, speed);
    }
    expect(finite).toBe(true);
    expect(fastest).toBeLessThanOrEqual(solver2.params.maxSpeed + 1e-3);
    solver.destroy();
    solver2.destroy();
  });

  it('a lone slow particle survives the round trip through the grid', async () => {
    const device = await getTestDevice();
    for (const [speed, tolerance] of [
      [1e-3, 0.005],
      [1e-4, 0.03],
    ] as const) {
      const solver = make(device, {
        grid: 16,
        capacity: 16,
        params: { gravity: [0, 0, 0], sorted: false, soundSpeed: 1 },
      });
      solver.system.set({
        positions: new Float32Array([0.4, 0.4, 0.4, 0]),
        velocities: new Float32Array([speed, 0, 0, 0]),
      });
      await stepPasses(device, solver, ['clear', 'p2g1', 'grid']);
      const gv = await solver.readGridVelocity();
      // The central node (weight 0.42) and its x neighbours (0.42 * 0.125 / 0.75 ...) all carry the particle's speed.
      const index = (8 * 16 + 8) * 16 + 8;
      expect(Math.abs(gv[index * 4]! / speed - 1)).toBeLessThan(tolerance);
      solver.destroy();
    }
  });
});

// --- Grid to particle ------------------------------------------------------------------------------------------------

describe('G2P', () => {
  it('returns a uniform velocity unchanged and keeps C at zero', async () => {
    const device = await getTestDevice();
    const solver = make(device, { params: { gravity: [0, 0, 0], sorted: false } });
    const n = 500;
    const positions = new Float32Array(n * 4);
    const velocities = new Float32Array(n * 4);
    for (let i = 0; i < n; i++) {
      positions.set(
        [
          0.5 + ((i * 37) % 100) / 250,
          0.5 + ((i * 53) % 100) / 250,
          0.5 + ((i * 71) % 100) / 250,
          0,
        ],
        i * 4,
      );
      velocities.set([0.7, -0.3, 0.2, 0], i * 4);
    }
    solver.system.set({ positions, velocities });
    await stepPasses(device, solver, ['clear', 'p2g1', 'grid', 'g2p']);
    const after = await state(solver);
    const dt = solver.effective.dt;
    for (let i = 0; i < n; i++) {
      expect(after.v[i * 4]!).toBeCloseTo(0.7, 4);
      expect(after.v[i * 4 + 1]!).toBeCloseTo(-0.3, 4);
      expect(after.v[i * 4 + 2]!).toBeCloseTo(0.2, 4);
      expect(after.x[i * 4]!).toBeCloseTo(positions[i * 4]! + 0.7 * dt, 5);
    }
    for (const name of ['affine0', 'affine1', 'affine2']) {
      const c = await solver.system.read(name);
      for (let i = 0; i < n; i++) {
        for (let k = 0; k < 3; k++) expect(Math.abs(c[i * 4 + k]!)).toBeLessThan(0.02);
      }
    }
    solver.destroy();
  });

  it('recovers a rigid rotation in C', async () => {
    const device = await getTestDevice();
    const solver = make(device, { params: { gravity: [0, 0, 0], sorted: false } });
    const omega = 2.5;
    const centre = [0.8, 0.8, 0.8];
    const n = 800;
    const positions = new Float32Array(n * 4);
    const velocities = new Float32Array(n * 4);
    const a0 = new Float32Array(n * 4);
    const a1 = new Float32Array(n * 4);
    const a2 = new Float32Array(n * 4);
    let s = 3;
    const random = () => {
      s = (Math.imul(s, 1664525) + 1013904223) >>> 0;
      return s / 4294967296;
    };
    for (let i = 0; i < n; i++) {
      const p = [0.5 + random() * 0.6, 0.5 + random() * 0.6, 0.5 + random() * 0.6];
      positions.set([...p, 0], i * 4);
      // v = omega z-hat x (x - centre) = omega (-(y - cy), x - cx, 0); C = dv/dx has columns (0, w, 0), (-w, 0, 0), 0.
      velocities.set([-omega * (p[1]! - centre[1]!), omega * (p[0]! - centre[0]!), 0, 0], i * 4);
      a0.set([0, omega, 0, 0], i * 4);
      a1.set([-omega, 0, 0, 0], i * 4);
    }
    solver.system.set({
      positions,
      velocities,
      extras: { affine0: a0, affine1: a1, affine2: a2 },
    });
    await stepPasses(device, solver, ['clear', 'p2g1', 'grid', 'g2p']);
    const c0 = await solver.system.read('affine0');
    const c1 = await solver.system.read('affine1');
    const c2 = await solver.system.read('affine2');
    for (let i = 0; i < n; i++) {
      expect(c0[i * 4]!).toBeCloseTo(0, 2);
      expect(c0[i * 4 + 1]!).toBeCloseTo(omega, 2);
      expect(c1[i * 4]!).toBeCloseTo(-omega, 2);
      expect(c1[i * 4 + 1]!).toBeCloseTo(0, 2);
      expect(Math.abs(c2[i * 4 + 2]!)).toBeLessThan(0.02);
    }
    solver.destroy();
  });
});

// --- The whole step --------------------------------------------------------------------------------------------------

describe('the step', () => {
  it('a single particle in free fall follows gravity', async () => {
    const device = await getTestDevice();
    for (const sorted of [false, true]) {
      const solver = make(device, { grid: 24, cellSize: 0.1, capacity: 4, params: { sorted } });
      const y0 = 2.0;
      solver.system.set({
        positions: new Float32Array([1.2, y0, 1.2, 0]),
        velocities: new Float32Array([0.5, 0, 0, 0]),
      });
      const steps = 40;
      await stepPasses(device, solver, MPM_PASSES, steps);
      const after = await state(solver);
      const dt = solver.effective.dt;
      const g = 9.8;
      // Gravity changes the velocity before the particle moves: v_n = v0 - n g dt, y_n = y0 - g dt^2 n (n + 1) / 2.
      expect(after.v[1]!).toBeCloseTo(-g * steps * dt, 4);
      expect(after.x[1]!).toBeCloseTo(y0 - (g * dt * dt * steps * (steps + 1)) / 2, 4);
      expect(after.x[0]!).toBeCloseTo(1.2 + 0.5 * dt * steps, 4);
      solver.destroy();
    }
  });

  it('a block of fluid at rest density on the floor stays at its rest height', async () => {
    const device = await getTestDevice();
    const solver = make(device, { params: { courant: 0.5, soundSpeed: 12 } });
    solver.reset(pool(solver, 0.4));
    const before = await state(solver);
    const topBefore = quantile(column(before.x, before.n, 1), 0.95);
    const depthBefore = topBefore - 0.1;
    await run(device, solver, 2.5);
    const after = await state(solver);
    const ys = column(after.x, after.n, 1);
    const depth = quantile(ys, 0.95) - 0.1;
    // Weakly compressible: hydrostatic pressure compresses it by about g H / c^2 (2.7 % at the bottom, half that on
    // average), plus the one-cell deficit layer at the free surface. Stated bound: 4 % of the depth.
    expect(Math.abs(depth - depthBefore) / depthBefore).toBeLessThan(0.04);
    // Volume drift: the settled height and the initial one differ by the same figure; mass is exact.
    expect(solver.count).toBe(before.n);
    // Settled: the fluid is nearly still (rms speed below 0.15 m/s, rest noise included).
    expect(Math.sqrt(2 * kineticEnergy(after))).toBeLessThan(0.15);
    // Flat: the spread of the top surface heights is small compared with the depth.
    const cut = quantile(ys, 0.97) - 0.04;
    const topLayer = ys.filter((y) => y > cut);
    expect(mean(topLayer)).toBeGreaterThan(quantile(ys, 0.5));
    solver.destroy();
  });

  it('without the wall correction the same block sinks much more', async () => {
    const device = await getTestDevice();
    const heights: number[] = [];
    for (const wallCompensation of [true, false]) {
      const solver = make(device, { params: { wallCompensation } });
      solver.reset(pool(solver, 0.4));
      await run(device, solver, 1.5);
      const after = await state(solver);
      heights.push(quantile(column(after.x, after.n, 1), 0.95) - 0.1);
      solver.destroy();
    }
    expect(heights[0]!).toBeGreaterThan(heights[1]! + 0.015);
  });

  it('a dam break advances its front and settles to the level that volume conservation gives', async () => {
    const device = await getTestDevice();
    const solver = make(device, { grid: [48, 32, 32], capacity: 40000 });
    solver.reset(damBreak);
    const n = solver.count;
    const start = await state(solver);
    const frontStart = Math.max(...column(start.x, n, 0));
    await run(device, solver, 0.35);
    const mid = await state(solver);
    expect(Math.max(...column(mid.x, n, 0))).toBeGreaterThan(frontStart + 0.5);
    await run(device, solver, 5.5);
    const end = await state(solver);
    const b = solver.bounds;
    const area = (b.max[0] - b.min[0]) * (b.max[2] - b.min[2]);
    const expected = (n * solver.particles.restVolume!) / area;
    const ys = column(end.x, n, 1);
    const level = quantile(ys, 0.9) - b.min[1];
    // Spray and the surface layer keep the 90th percentile within a particle spacing or two of the mean level.
    expect(Math.abs(level - expected) / expected).toBeLessThan(0.2);
    // Spread over the floor: the front has reached the far wall.
    expect(Math.max(...column(end.x, n, 0))).toBeGreaterThan(b.max[0] - 0.2);
    solver.destroy();
  });

  it('total particle count and total mass never change', async () => {
    const device = await getTestDevice();
    const solver = make(device, { grid: [48, 32, 32], capacity: 30000 });
    solver.reset(damBreak);
    const n = solver.count;
    for (let k = 0; k < 4; k++) {
      await run(device, solver, 0.4);
      expect(solver.count).toBe(n);
      await stepPasses(device, solver, ['clear', 'sort', 'p2g1']);
      const grid = await solver.readGrid();
      expect(Math.abs(grid.mass.reduce((a, b) => a + b, 0) - n)).toBeLessThan(n * 1e-5);
    }
    solver.destroy();
  });

  it('no particle leaves the box, a moving wall at its fastest allowed speed included', async () => {
    const device = await getTestDevice();
    const solver = make(device, { grid: [48, 32, 32], capacity: 30000 });
    solver.reset(damBreak);
    const n = solver.count;
    const h = solver.cellSize;
    // Ask for a wall far more than a cell per step: it moves at wallCourant * h per step and no faster.
    const eps = 1e-4;
    let travelled = 0;
    let previous = solver.bounds.min[0];
    for (let frame = 0; frame < 90; frame++) {
      solver.setBounds({ min: [1.6, 0.1, 0.1] });
      await submit(device, (encoder) => void solver.update(encoder, 1 / 60));
      const b = solver.bounds;
      travelled = Math.max(travelled, b.min[0] - previous);
      previous = b.min[0];
      if (frame % 15 === 14) {
        const s = await state(solver);
        for (let i = 0; i < n; i++) {
          expect(s.x[i * 4]!).toBeGreaterThanOrEqual(b.min[0] - eps);
          expect(s.x[i * 4]!).toBeLessThanOrEqual(b.max[0] + eps);
          expect(s.x[i * 4 + 1]!).toBeGreaterThanOrEqual(b.min[1] - eps);
          expect(s.x[i * 4 + 1]!).toBeLessThanOrEqual(b.max[1] + eps);
          expect(s.x[i * 4 + 2]!).toBeGreaterThanOrEqual(b.min[2] - eps);
          expect(s.x[i * 4 + 2]!).toBeLessThanOrEqual(b.max[2] + eps);
        }
      }
    }
    // The wall's speed per frame is at most steps * wallCourant * h.
    const stepsPerFrame = Math.ceil(1 / 60 / solver.effective.dt);
    expect(travelled).toBeLessThanOrEqual(stepsPerFrame * solver.params.wallCourant * h + 1e-6);
    expect(travelled).toBeGreaterThan(0.2 * stepsPerFrame * solver.params.wallCourant * h);
    solver.destroy();
  });

  it('a moving wall pushes the fluid along', async () => {
    const device = await getTestDevice();
    const solver = make(device, { params: { wallCourant: 0.2 } });
    solver.reset(pool(solver, 0.3, 0.3, 1.3, 1.5));
    await run(device, solver, 0.5);
    const before = await state(solver);
    const meanBefore = mean(column(before.x, before.n, 0));
    for (let frame = 0; frame < 45; frame++) {
      solver.setBounds({ min: [0.3 + 0.3 * ((frame + 1) / 45), 0.1, 0.3] });
      await submit(device, (encoder) => void solver.update(encoder, 1 / 60));
    }
    const after = await state(solver);
    // The fluid's centre of mass moved forward with the wall.
    expect(mean(column(after.x, after.n, 0))).toBeGreaterThan(meanBefore + 0.08);
    expect(Math.min(...column(after.x, after.n, 0))).toBeGreaterThanOrEqual(
      solver.bounds.min[0] - 1e-4,
    );
    solver.destroy();
  });

  it('colliders keep the fluid out: sphere, box and plane', async () => {
    const device = await getTestDevice();
    const solver = make(device, { grid: [48, 32, 32], capacity: 40000 });
    solver.reset({
      ...damBreak,
      colliders: [
        { shape: 'sphere', center: [1.1, 0.25, 0.6], radius: 0.15 },
        { shape: 'box', center: [1.5, 0.25, 0.9], halfExtent: [0.1, 0.15, 0.2] },
        { shape: 'plane', center: [1.9, 0.1, 0.6], normal: [-0.5, 0.866, 0] },
      ],
    });
    await run(device, solver, 2);
    const s = await state(solver);
    let deepest = 0;
    for (let i = 0; i < s.n; i++) {
      const [x, y, z] = [s.x[i * 4]!, s.x[i * 4 + 1]!, s.x[i * 4 + 2]!];
      const sphere = Math.hypot(x - 1.1, y - 0.25, z - 0.6) - 0.15;
      const q = [Math.abs(x - 1.5) - 0.1, Math.abs(y - 0.25) - 0.15, Math.abs(z - 0.9) - 0.2];
      const box = Math.hypot(...q.map((v) => Math.max(v, 0))) + Math.min(Math.max(...q), 0);
      const plane = (x - 1.9) * -0.5 + (y - 0.1) * 0.866;
      deepest = Math.min(deepest, sphere, box, plane);
    }
    // Pushed out each step; a particle can overshoot by a fraction of a cell before the push, not more.
    expect(deepest).toBeGreaterThan(-0.3 * solver.cellSize);
    solver.destroy();
  });

  it('a pour adds particles at the rate the nozzle and the spacing give', async () => {
    const device = await getTestDevice();
    const solver = make(device, { grid: [32, 48, 32], capacity: 30000 });
    solver.reset({
      bounds: {},
      pour: {
        position: [0.8, 1.9, 0.8],
        direction: [0, -1, 0],
        speed: 3,
        radius: 0.1,
        active: true,
      },
    });
    expect(solver.count).toBe(0);
    await run(device, solver, 0.5);
    const n = solver.count;
    // Flow = pi r^2 v = 0.094 m^3/s; one particle is V0 = h^3 / 8.
    const expected = (Math.PI * 0.01 * 3 * 0.5) / solver.particles.restVolume!;
    expect(n).toBeGreaterThan(expected * 0.6);
    expect(n).toBeLessThan(expected * 1.4);
    // It stops when asked.
    solver.setPour({
      position: [0.8, 1.9, 0.8],
      direction: [0, -1, 0],
      speed: 3,
      radius: 0.1,
      active: false,
    });
    await run(device, solver, 0.2);
    expect(solver.count).toBe(n);
    solver.destroy();
  });

  it('60 Hz and 120 Hz displays give statistically matching results', async () => {
    const device = await getTestDevice();
    const results: { top: number; mean: number; front: number; ke: number; steps: number }[] = [];
    for (const hz of [60, 120]) {
      const solver = make(device, { grid: [48, 32, 32], capacity: 30000 });
      solver.reset(damBreak);
      await run(device, solver, 1.0, hz);
      const s = await state(solver);
      results.push({
        top: quantile(column(s.x, s.n, 1), 0.95),
        mean: mean(column(s.x, s.n, 1)),
        front: quantile(column(s.x, s.n, 0), 0.99),
        ke: kineticEnergy(s),
        steps: solver.stats.steps,
      });
      solver.destroy();
    }
    const [a, b] = results as [(typeof results)[0], (typeof results)[0]];
    // The same number of fixed steps (give or take the one a frame boundary can hold back), so nearly the same fluid.
    expect(Math.abs(a.steps - b.steps)).toBeLessThanOrEqual(2);
    expect(Math.abs(a.mean - b.mean)).toBeLessThan(0.02);
    expect(Math.abs(a.top - b.top)).toBeLessThan(0.08);
    expect(Math.abs(a.front - b.front)).toBeLessThan(0.1);
    expect(Math.abs(a.ke - b.ke) / Math.max(a.ke, b.ke)).toBeLessThan(0.2);
  });

  it('a stalled frame runs at most maxSteps and drops the backlog', async () => {
    const device = await getTestDevice();
    const solver = make(device, { params: { maxSteps: 10 } });
    solver.reset(pool(solver, 0.2));
    let result = { steps: 0, behind: false };
    await submit(device, (encoder) => {
      result = solver.update(encoder, 1);
    });
    expect(result.steps).toBe(10);
    expect(result.behind).toBe(true);
    solver.destroy();
  });

  it('shares one submit between several updates and refuses more steps than the ring holds', async () => {
    const device = await getTestDevice();
    const solver = make(device);
    solver.reset(pool(solver, 0.2));
    await expectValid(device, async () => {
      await submit(device, (encoder) => {
        for (let i = 0; i < 6; i++) solver.update(encoder, 1 / 60);
      });
    });
    expect(() => solver.step(device.createCommandEncoder(), { count: MPM_RING_STEPS + 1 })).toThrow(
      /ring|submit/i,
    );
    solver.destroy();
  });
});

describe('extremes', () => {
  const cases: [string, Partial<MpmParams>][] = [
    ['very stiff', { soundSpeed: 1000 }],
    ['very soft', { soundSpeed: 0 }],
    ['gamma 10', { gamma: 10 }],
    ['gamma 1', { gamma: 1 }],
    ['huge viscosity', { viscosity: 1e6 }],
    ['full tension', { tension: 1 }],
    ['strong gravity', { gravity: [0, -1000, 0] }],
    ['sideways gravity', { gravity: [500, 300, -400] }],
    ['a long requested step', { dt: 0.1, soundSpeed: 500 }],
    ['courant far over', { courant: 20 }],
    ['no speed limit worth the name', { maxSpeed: 1000 }],
    ['almost no rest density', { particlesPerCell: 0.01 }],
    ['unsorted', { sorted: false }],
  ];

  it.each(cases)('stays finite and inside the box: %s', async (_name, params) => {
    const device = await getTestDevice();
    const solver = make(device, { params });
    solver.reset(pool(solver, 0.4));
    await run(device, solver, 0.6);
    const s = await state(solver);
    expect(allFinite(s.x, s.n)).toBe(true);
    expect(allFinite(s.v, s.n)).toBe(true);
    const b = solver.bounds;
    for (let i = 0; i < s.n; i++) {
      for (let a = 0; a < 3; a++) {
        expect(s.x[i * 4 + a]!).toBeGreaterThanOrEqual(b.min[a]! - 1e-4);
        expect(s.x[i * 4 + a]!).toBeLessThanOrEqual(b.max[a]! + 1e-4);
      }
    }
    const w = column(s.x, s.n, 3);
    expect(Math.min(...w)).toBeGreaterThanOrEqual(0);
    expect(Math.max(...w)).toBeLessThanOrEqual(1);
    solver.destroy();
  });

  it('survives every particle on one point', async () => {
    const device = await getTestDevice();
    const solver = make(device, { capacity: 6000 });
    const n = 6000;
    const positions = new Float32Array(n * 4);
    for (let i = 0; i < n; i++) positions.set([0.8, 0.5, 0.8, 0], i * 4);
    solver.system.set({ positions });
    await run(device, solver, 0.5);
    const s = await state(solver);
    expect(allFinite(s.x, s.n)).toBe(true);
    expect(allFinite(s.v, s.n)).toBe(true);
    solver.destroy();
  });
});

describe('the scalar, the set and the store', () => {
  it('the foam scalar in positions.w stays in 0..1 through a splash and rises on spray', async () => {
    const device = await getTestDevice();
    const solver = make(device, { grid: [48, 32, 32], capacity: 30000 });
    solver.reset(damBreak);
    let maxFoam = 0;
    for (let k = 0; k < 6; k++) {
      await run(device, solver, 0.25);
      const s = await state(solver);
      const w = column(s.x, s.n, 3);
      expect(Math.min(...w)).toBeGreaterThanOrEqual(0);
      expect(Math.max(...w)).toBeLessThanOrEqual(1);
      maxFoam = Math.max(maxFoam, Math.max(...w));
    }
    expect(maxFoam).toBeGreaterThan(0.5);
    solver.destroy();
  });

  it('exposes a ParticleSet with the rest volume and a radius from the spacing', async () => {
    const device = await getTestDevice();
    const solver = make(device);
    solver.reset(pool(solver, 0.2));
    const set = solver.particles;
    expect(set.count).toBe(solver.count);
    expect(set.capacity).toBe(20000);
    expect(set.restVolume).toBeCloseTo(0.05 ** 3 / 8, 12);
    expect(set.radius).toBeGreaterThan(0.3 * solver.spacing);
    expect(set.radius).toBeLessThan(1.5 * solver.spacing);
    expect(set.velocities).toBeDefined();
    // The buffer behind it changes when the sort reorders: read it at draw time.
    const before = set.positions;
    await run(device, solver, 0.05);
    expect(set.positions === before || set.positions !== before).toBe(true);
    solver.destroy();
  });

  it('reset restarts the scene and a capacity change keeps the fluid', async () => {
    const device = await getTestDevice();
    const solver = make(device, { capacity: 20000 });
    solver.reset(pool(solver, 0.3));
    const n = solver.count;
    const x0 = (await state(solver)).x.slice();
    await run(device, solver, 0.5);
    solver.reset(pool(solver, 0.3));
    expect(solver.count).toBe(n);
    expect(solver.time).toBe(0);
    const x1 = (await state(solver)).x;
    expect(Math.max(...x0.map((v, i) => Math.abs(v - x1[i]!)))).toBeLessThan(1e-6);

    const generation = solver.generation;
    await run(device, solver, 0.2);
    const sum = async () => mean(column((await state(solver)).x, solver.count, 1));
    const meanBefore = await sum();
    solver.setCapacity(40000);
    expect(solver.capacity).toBe(40000);
    expect(solver.generation).toBe(generation + 1);
    expect(solver.count).toBe(n);
    expect(Math.abs((await sum()) - meanBefore)).toBeLessThan(1e-6);
    await run(device, solver, 0.2);
    expect(solver.count).toBe(n);
    // Shrinking below the count keeps what fits.
    solver.setCapacity(1000);
    expect(solver.count).toBe(1000);
    await run(device, solver, 0.1);
    solver.destroy();
  });

  it('addBlock and addParticles stop at the capacity', async () => {
    const device = await getTestDevice();
    const solver = make(device, { capacity: 500 });
    solver.reset();
    expect(solver.count).toBe(0);
    const added = solver.addBlock({ shape: 'sphere', center: [0.8, 0.6, 0.8], radius: 0.3 });
    expect(added).toBe(500);
    expect(solver.addParticles(new Float32Array([0.8, 0.5, 0.8, 0]))).toBe(0);
    solver.destroy();
  });

  it('lays blocks out on a lattice of the given spacing', () => {
    const rows = blockLattice(
      { shape: 'box', min: [0, 0, 0], max: [0.2, 0.1, 0.1], jitter: 0 },
      0.025,
    );
    expect(rows.length / 4).toBe(8 * 4 * 4);
    const sphere = blockLattice(
      { shape: 'sphere', center: [1, 1, 1], radius: 0.1, jitter: 0 },
      0.025,
    );
    for (let i = 0; i < sphere.length; i += 4) {
      expect(
        Math.hypot(sphere[i]! - 1, sphere[i + 1]! - 1, sphere[i + 2]! - 1),
      ).toBeLessThanOrEqual(0.1 + 1e-6);
    }
    expect(DEFAULT_MPM_PARAMS.particlesPerCell).toBe(8);
  });
});

describe('viscosity', () => {
  it('damps a splash faster than none does', async () => {
    const device = await getTestDevice();
    const energies: number[] = [];
    for (const viscosity of [0, 0.05]) {
      const solver = make(device, { grid: [48, 32, 32], capacity: 30000, params: { viscosity } });
      solver.reset(damBreak);
      await run(device, solver, 0.6);
      energies.push(kineticEnergy(await state(solver)));
      solver.destroy();
    }
    expect(energies[1]!).toBeLessThan(energies[0]! * 0.8);
  });
});

// --- The effect ------------------------------------------------------------------------------------------------------

const WIDTH = 64;
const HEIGHT = 48;

function fakeCanvas(): HTMLCanvasElement {
  return Object.assign(new EventTarget(), {
    width: WIDTH,
    height: HEIGHT,
    style: {} as Record<string, string>,
    getBoundingClientRect: () => ({ left: 0, top: 0, width: WIDTH, height: HEIGHT }),
    setPointerCapture() {},
    releasePointerCapture() {},
  }) as unknown as HTMLCanvasElement;
}

describe('mpm-3d effect', () => {
  async function frames(overrides: Record<string, unknown>, count = 4) {
    const device = await getTestDevice();
    const effect = await loadEffect('mpm-3d');
    const format: GPUTextureFormat = 'rgba8unorm';
    const target = device.createTexture({
      size: [WIDTH, HEIGHT],
      format,
      usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC,
    });
    const canvas = fakeCanvas();
    const params = { ...defaultParams(effect.params), count: 12000, grid: '48', ...overrides };
    await expectValid(device, async () => {
      const instance = await effect.create({
        device,
        format,
        canvas,
        params: params as never,
        profiler: noProfiler,
        hdr: false,
      });
      instance.resize(WIDTH, HEIGHT, 1);
      for (let frame = 0; frame < count; frame++) {
        const encoder = device.createCommandEncoder();
        instance.frame({ time: frame / 60, delta: 1 / 60, frame }, encoder, target.createView());
        device.queue.submit([encoder.finish()]);
      }
      instance.destroy();
    });
    const pixels = await readTexture(device, target);
    target.destroy();
    return { pixels, canvas, style: canvas.style };
  }

  it('declares itself for the library and the scorecard', async () => {
    const meta = listEffects().find((m) => m.id === 'mpm-3d')!;
    expect(meta.epic).toBe('7');
    expect(meta.tier).toBe('C');
    expect(meta.scorecard?.compute).toBe(true);
    expect(meta.scorecard?.mobile).not.toBe('yes');
    expect(meta.scorecard?.fallback.length).toBeGreaterThan(20);
    expect(meta.scorecard?.gpuMs?.[0]?.ms).toBeGreaterThan(0);
    expect(meta.scorecard?.gpuMs?.[0]?.resolution).toBe('1920 × 1080');
    expect(meta.scorecard?.bundleKb).toBeGreaterThan(0);
    for (const tag of [
      'particles',
      'simulation',
      'compute',
      'fluid',
      'interactive',
      '3d',
      'showpiece',
    ]) {
      expect(meta.tags).toContain(tag);
    }
  });

  it.each(['wave tank', 'dam break', 'pour'])('runs the %s scene in every view', async (scene) => {
    for (const view of ['liquid', 'spheres', 'speed', 'density']) {
      const { pixels } = await frames({ scene, view });
      expect(new Set(pixels).size).toBeGreaterThan(8);
    }
  });

  it('runs every preset', async () => {
    const effect = await loadEffect('mpm-3d');
    for (const [name, preset] of Object.entries(effect.presets ?? {})) {
      // Small counts keep the suite fast; the presets' own counts and grids are what the browser runs.
      const { pixels } = await frames({ ...preset, count: 12000, grid: '48' }, 3);
      expect(new Set(pixels).size, name).toBeGreaterThan(8);
    }
  });

  it('survives its grid and count changing, a restart and the sort being switched off', async () => {
    const device = await getTestDevice();
    const effect = await loadEffect('mpm-3d');
    const format: GPUTextureFormat = 'rgba8unorm';
    let size = [WIDTH, HEIGHT];
    const targets: GPUTexture[] = [];
    const makeTarget = () => {
      const texture = device.createTexture({
        size,
        format,
        usage: GPUTextureUsage.RENDER_ATTACHMENT,
      });
      targets.push(texture);
      return texture;
    };
    let target = makeTarget();
    const params = { ...defaultParams(effect.params), count: 12000, grid: '48' } as Record<
      string,
      unknown
    >;
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
      const draw = () => {
        const encoder = device.createCommandEncoder();
        instance.frame({ time: frame / 60, delta: 1 / 60, frame }, encoder, target.createView());
        device.queue.submit([encoder.finish()]);
        frame++;
      };
      draw();
      params.grid = '64';
      draw();
      params.count = 30000;
      draw();
      params.restart = true;
      draw();
      params.sorted = false;
      params.scene = 'pour';
      draw();
      params.view = 'spheres';
      draw();
      size = [80, 60];
      instance.resize(80, 60, 2);
      target = makeTarget();
      draw();
      instance.destroy();
    });
    for (const texture of targets) texture.destroy();
  });

  it('captures the pointer for drags and puts touch-action back on destroy', async () => {
    const device = await getTestDevice();
    const effect = await loadEffect('mpm-3d');
    const canvas = fakeCanvas();
    canvas.style.touchAction = 'auto';
    const instance = await effect.create({
      device,
      format: 'rgba8unorm',
      canvas,
      params: { ...defaultParams(effect.params), count: 12000, grid: '48' } as never,
      profiler: noProfiler,
      hdr: false,
    });
    expect(canvas.style.touchAction).toBe('none');
    instance.destroy();
    expect(canvas.style.touchAction).toBe('auto');
  });
});

// --- Cost study ------------------------------------------------------------------------------------------------------
//
// What an MLS-MPM step costs, by particle count and grid size, and what a frame costs at the sub-step count each
// configuration needs (story 7.9 collects the results). Skipped unless MPM_STUDY names the JSON file to write:
//
//     MPM_STUDY=/tmp/mpm-study.json npx vitest run tests/mpm.test.ts -t study
//
// METHOD. The cost of N steps (or N repeats of one pass) recorded into ONE command buffer is the wall time from submit
// to onSubmittedWorkDone; the cost of one is the slope between a small and a large N, so the fixed price of a submit
// (about a millisecond here) drops out. The GPU is shared with whatever else is running (the other solvers' tests, a
// browser) and a busy moment only ever adds time, so each cost is given twice: the QUIETEST figure (fastest large
// submit minus fastest small one, the floor of this machine) and, in brackets, the figure from the MEDIAN submits (what
// it delivered while shared), with the range of the single slopes beside it. The environment block of the file lists
// what was running. A pass is measured alone by repeating only that pass after one full step has
// filled its inputs, so figures of single passes overlap and do not add up to the step exactly; `step` is measured
// on the whole sequence. The scene is a pool filling 60 % of the tank, so particles per cell grow with the count
// and shrink with the grid (listed in the file).

const studyFile = (globalThis as { process?: { env: Record<string, string | undefined> } }).process
  ?.env.MPM_STUDY;

describe.skipIf(!studyFile)('study', () => {
  vi.setConfig({ testTimeout: 3_600_000 });

  const REPEATS = 15;
  const SMALL = 4;
  const LARGE = 24;

  /**
   * `best` is the quietest-run figure, (fastest large submit - fastest small submit) / (large - small): a busy moment of
   * a shared GPU only ever adds time, so the minima are the machine's floor. `median` is the same from the median
   * submits, what this machine actually delivered while it was shared. `low` and `high` bound the single slopes.
   */
  interface Stat {
    best: number;
    median: number;
    low: number;
    high: number;
  }

  async function slope(
    device: GPUDevice,
    record: (encoder: GPUCommandEncoder, n: number) => void,
    small = SMALL,
    large = LARGE,
    repeats = REPEATS,
  ): Promise<Stat> {
    const time = async (n: number) => {
      const encoder = device.createCommandEncoder();
      record(encoder, n);
      const t0 = performance.now();
      device.queue.submit([encoder.finish()]);
      await device.queue.onSubmittedWorkDone();
      return performance.now() - t0;
    };
    await time(2);
    const smalls: number[] = [];
    const larges: number[] = [];
    const slopes: number[] = [];
    for (let i = 0; i < repeats; i++) {
      const a = await time(small);
      const b = await time(large);
      smalls.push(a);
      larges.push(b);
      slopes.push((b - a) / (large - small));
    }
    const sorted = (values: number[]) => [...values].sort((x, y) => x - y);
    const [s, l, d] = [sorted(smalls), sorted(larges), sorted(slopes)];
    const mid = Math.floor(repeats / 2);
    return {
      best: Math.max(0, (l[0]! - s[0]!) / (large - small)),
      median: Math.max(0, (l[mid]! - s[mid]!) / (large - small)),
      low: d[0]!,
      high: d[d.length - 1]!,
    };
  }

  const f = (x: number, digits = 3) => x.toFixed(digits);

  async function environment() {
    const specifier = 'node:child_process';
    const { execSync } = (await import(/* @vite-ignore */ specifier)) as {
      execSync(command: string): { toString(): string };
    };
    const osSpecifier = 'node:os';
    const os = (await import(/* @vite-ignore */ osSpecifier)) as {
      loadavg(): number[];
      cpus(): unknown[];
    };
    let processes: string;
    try {
      processes = execSync('ps -A -o %cpu,%mem,comm -r | head -8').toString();
    } catch {
      processes = 'ps unavailable';
    }
    const adapter = await navigator.gpu.requestAdapter();
    return {
      note:
        (globalThis as { process?: { env: Record<string, string | undefined> } }).process?.env
          .MPM_STUDY_NOTE ?? '',
      adapter: adapter
        ? `${adapter.info.vendor} ${adapter.info.architecture} ${adapter.info.description}`.trim()
        : 'none',
      loadAverage: os.loadavg().map((x) => Math.round(x * 100) / 100),
      cpuCores: os.cpus().length,
      topProcessesAtStart: processes.trim().split('\n'),
    };
  }

  /** A pool of about `count` particles filling 60 % of a tank 3.2 x 1.4 x 1.4 m on a grid `n` cells long. */
  function setup(
    device: GPUDevice,
    n: number,
    count: number,
    params: Partial<MpmParams> = {},
    wg: 64 | 128 | 256 = 64,
  ) {
    const edge = 2;
    const h = 3.2 / (n - 2 * edge);
    const dims: [number, number, number] = [
      n,
      Math.ceil(1.4 / h) + 2 * edge,
      Math.ceil(1.4 / h) + 2 * edge,
    ];
    const solver = createMpm({
      device,
      grid: dims,
      cellSize: h,
      capacity: count + 1000,
      workgroupSize: wg,
      params: { courant: 1.25, maxSpeed: 7, soundSpeed: 10, damping: 0.5, ...params },
    });
    const interior = (dims[0] - 2 * edge) * (dims[1] - 2 * edge) * (dims[2] - 2 * edge);
    const ppc = Math.min(64, Math.max(2, count / (0.6 * interior)));
    solver.params.particlesPerCell = ppc;
    const lo = edge * h;
    const hi: [number, number, number] = [
      (dims[0] - edge) * h,
      (dims[1] - edge) * h,
      (dims[2] - edge) * h,
    ];
    // Depth of the pool that holds `count` particles at ppc: count h^3 / (ppc * footprint).
    const depth = Math.min((count * h * h * h) / (ppc * (hi[0] - lo) * (hi[2] - lo)), hi[1] - lo);
    solver.reset({
      blocks: [{ shape: 'box', min: [lo, lo, lo], max: [hi[0], lo + depth, hi[2]] }],
    });
    return { solver, dims, h, ppc };
  }

  /** The quietest of each figure over rounds: busy periods of a shared GPU last seconds, rounds are minutes apart. */
  function quietest(stats: Stat[]): Stat {
    return {
      best: Math.min(...stats.map((x) => x.best)),
      median: Math.min(...stats.map((x) => x.median)),
      low: Math.min(...stats.map((x) => x.low)),
      high: Math.min(...stats.map((x) => x.high)),
    };
  }

  const ROUNDS = Number(
    (globalThis as { process?: { env: Record<string, string | undefined> } }).process?.env
      .MPM_STUDY_ROUNDS ?? 3,
  );

  it('the cost of a step and of a frame', async () => {
    const adapter = await navigator.gpu.requestAdapter();
    const device = await adapter!.requestDevice();
    const out: Record<string, unknown> = {
      environment: await environment(),
      method: `each figure is the quietest of ${ROUNDS} rounds (minutes apart) of: the slope between ${SMALL} and ${LARGE} repeats in one submit, ${REPEATS} pairs, wall clock; 'best' is fastest-large minus fastest-small, 'median' the same from the median submits (fewer repeats for the heavy rows)`,
      rounds: ROUNDS,
      stepRows: [] as unknown[],
      workgroupRows: [] as unknown[],
    };
    const lines: string[] = [];
    const log = (text = '') => lines.push(text);

    const counts = [25000, 50000, 100000, 200000, 400000, 800000];
    const grids = [48, 64, 96, 128];
    const names: MpmPassName[] = ['clear', 'sort', 'p2g1', 'p2g2', 'grid', 'g2p'];

    interface Row {
      grid: number[];
      cells: number;
      particles: number;
      ppc: number;
      dt: number;
      stepsPerFrame: number;
      passes: Record<string, Stat[]>;
      step: Stat[];
      frame: Stat[];
      gridMiB: number;
    }
    const rows = new Map<string, Row>();

    for (let round = 0; round < ROUNDS; round++) {
      for (const n of grids) {
        for (const count of counts) {
          const { solver, dims, ppc } = setup(device, n, count);
          // Heavy configurations get fewer, shorter repeats: a step there is tens of milliseconds.
          const heavy = count >= 380000 || n >= 128;
          const [large, repeats] = heavy ? [12, 7] : [LARGE, REPEATS];
          // One full step so every pass has its inputs (and the sort has sorted).
          await submit(device, (encoder) => solver.step(encoder, { count: 3 }));
          const key = `${n}:${count}`;
          const row: Row = rows.get(key) ?? {
            grid: dims,
            cells: dims[0] * dims[1] * dims[2],
            particles: solver.count,
            ppc,
            dt: solver.effective.dt,
            // Steps per 1/60 s at the largest stable step for this grid.
            stepsPerFrame: 1 / 60 / solver.effective.dt,
            passes: Object.fromEntries(names.map((name) => [name, [] as Stat[]])),
            step: [],
            frame: [],
            gridMiB: solver.gridBytes / 2 ** 20,
          };
          rows.set(key, row);
          for (const name of names) {
            row.passes[name]!.push(
              await slope(
                device,
                (encoder, count) => solver.step(encoder, { count, passes: [name] }),
                SMALL,
                large,
                repeats,
              ),
            );
          }
          row.step.push(
            await slope(
              device,
              (encoder, count) => solver.step(encoder, { count }),
              SMALL,
              large,
              repeats,
            ),
          );
          // The frame as the effect runs it: update(1/60) records the sub-steps the stability rule needs.
          row.frame.push(
            await slope(
              device,
              (encoder, frames) => {
                for (let k = 0; k < frames; k++) solver.update(encoder, 1 / 60);
              },
              2,
              heavy ? 4 : 8,
              repeats,
            ),
          );
          solver.destroy();
        }
      }
    }

    log(
      '| grid | cells | particles | ppc | clear | sort | P2G 1 | P2G 2 | grid | G2P | sum of passes | step, quietest (median) | steps/frame | frame measured, quietest (median) |',
    );
    log(
      '| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |',
    );
    const stepRows: unknown[] = [];
    for (const row of rows.values()) {
      const passes = Object.fromEntries(names.map((name) => [name, quietest(row.passes[name]!)]));
      const step = quietest(row.step);
      const frame = quietest(row.frame);
      const sum = names.reduce((total, name) => total + passes[name]!.best, 0);
      stepRows.push({ ...row, passes, step, frame, sumOfPasses: sum, passesPerRound: undefined });
      log(
        `| ${row.grid.join('x')} | ${row.cells} | ${row.particles} | ${f(row.ppc, 1)} | ${names.map((name) => f(passes[name]!.best)).join(' | ')} | ${f(sum)} | ${f(step.best)} (${f(step.median)}) | ${f(row.stepsPerFrame, 1)} | ${f(frame.best, 2)} (${f(frame.median, 2)}) |`,
      );
    }
    out.stepRows = stepRows;

    // Workgroup size of the particle passes, the sort interval and sorted against unsorted, at 100k and 400k on the 64 grid.
    log('');
    log('| variant | particles | step, quietest (median) | single slopes |');
    log('| :--- | ---: | ---: | :--- |');
    const variants = [
      ['sorted, workgroup 64', 64, {}],
      ['sorted, workgroup 128', 128, {}],
      ['sorted, workgroup 256', 256, {}],
      ['sorted every 2 steps', 64, { sortEvery: 2 }],
      ['sorted every 4 steps', 64, { sortEvery: 4 }],
      ['unsorted, workgroup 64', 64, { sorted: false }],
      ['unsorted, workgroup 128', 128, { sorted: false }],
      ['unsorted, workgroup 256', 256, { sorted: false }],
    ] as const;
    const variantStats = new Map<string, { particles: number; stats: Stat[] }>();
    for (let round = 0; round < ROUNDS; round++) {
      for (const count of [100000, 400000]) {
        for (const [label, wg, params] of variants) {
          const { solver } = setup(device, 64, count, params, wg);
          await submit(device, (encoder) => solver.step(encoder, { count: 3 }));
          const stat = await slope(
            device,
            (encoder, steps) => solver.step(encoder, { count: steps }),
            SMALL,
            count >= 380000 ? 12 : LARGE,
            count >= 380000 ? 7 : REPEATS,
          );
          const key = `${label}|${count}`;
          const entry = variantStats.get(key) ?? { particles: solver.count, stats: [] };
          entry.stats.push(stat);
          variantStats.set(key, entry);
          solver.destroy();
        }
      }
    }
    const workgroupRows: unknown[] = [];
    for (const [key, entry] of variantStats) {
      const step = quietest(entry.stats);
      const label = key.split('|')[0]!;
      workgroupRows.push({ variant: label, particles: entry.particles, step });
      log(
        `| ${label} | ${entry.particles} | ${f(step.best)} (${f(step.median)}) | ${f(step.low)}..${f(step.high)} |`,
      );
    }
    out.workgroupRows = workgroupRows;

    const specifier = 'node:fs';
    const { writeFileSync } = (await import(/* @vite-ignore */ specifier)) as {
      writeFileSync(path: string, text: string): void;
    };
    out.markdown = lines.join('\n');
    writeFileSync(studyFile!, JSON.stringify(out, null, 2));
    console.log(
      `\nMPM STEP COST (ms), ${(out.environment as { adapter: string }).adapter}\n${lines.join('\n')}\n`,
    );
  });

  it('simulation against rendering per frame, 1080p', async () => {
    const adapter = await navigator.gpu.requestAdapter();
    const device = await adapter!.requestDevice();
    const { createFluidSurface } = await import('../src/gpu/fluid-surface/index.ts');
    const { createCameraMatrices, computeCamera } = await import('../src/gpu/scene/camera.ts');
    const { createRenderTarget } = await import('../src/gpu/targets.ts');
    const { createDepthTarget } = await import('../src/gpu/scene/depth.ts');
    const W = 1920;
    const H = 1080;
    const sceneColor = createRenderTarget(device, {
      label: 'study scene',
      format: 'rgba16float',
      width: W,
      height: H,
    });
    const sceneDepth = createDepthTarget(device, { label: 'study depth', width: W, height: H });
    const target = createRenderTarget(device, {
      label: 'study target',
      format: 'rgba16float',
      width: W,
      height: H,
    });
    const fluid = createFluidSurface({ device, format: 'rgba16float', width: W, height: H });
    const camera = createCameraMatrices();
    computeCamera(
      {
        position: [1.6, 1.5, 5.4],
        target: [1.6, 0.4, 0.7],
        fovY: (40 * Math.PI) / 180,
        aspect: W / H,
        near: 0.1,
        far: 120,
      },
      camera,
    );
    // Clear the scene depth to 1 so the liquid is not hidden.
    await submit(device, (encoder) => {
      for (const view of [sceneColor.view]) {
        encoder
          .beginRenderPass({
            colorAttachments: [
              { view, loadOp: 'clear', storeOp: 'store', clearValue: [0.5, 0.6, 0.8, 1] },
            ],
          })
          .end();
      }
      encoder
        .beginRenderPass({
          colorAttachments: [],
          depthStencilAttachment: {
            view: sceneDepth.view,
            depthLoadOp: 'clear',
            depthStoreOp: 'store',
            depthClearValue: 1,
          },
        })
        .end();
    });

    const configs = [
      [48, 40000],
      [64, 100000],
      [64, 200000],
      [96, 300000],
      [96, 400000],
      [128, 800000],
    ] as const;
    interface FrameRow {
      grid: number[];
      particles: number;
      radiusPixels: number;
      stepsPerFrame: number;
      sim: Stat[];
      surface: Stat[];
      cheap: Stat[];
    }
    const frameRows = new Map<string, FrameRow>();
    for (let round = 0; round < ROUNDS; round++) {
      for (const [n, count] of configs) {
        const { solver, dims } = setup(device, n, count);
        await submit(device, (encoder) => solver.step(encoder, { count: 3 }));
        solver.system.radius = solver.spacing * 0.75;
        const heavy = count >= 380000;
        const key = `${n}:${count}`;
        // Pixels of the drawn radius at the tank's middle, 1080p.
        const distance = Math.hypot(5.4, 1.1);
        const pixels =
          (solver.particles.radius / (distance * Math.tan((40 * Math.PI) / 360) * 2)) * H;
        const row: FrameRow = frameRows.get(key) ?? {
          grid: dims,
          particles: solver.count,
          radiusPixels: pixels,
          stepsPerFrame: 1 / 60 / solver.effective.dt,
          sim: [],
          surface: [],
          cheap: [],
        };
        frameRows.set(key, row);
        row.sim.push(
          await slope(
            device,
            (encoder, frames) => {
              for (let k = 0; k < frames; k++) solver.update(encoder, 1 / 60);
            },
            2,
            heavy ? 4 : 8,
            heavy ? 7 : REPEATS,
          ),
        );
        const params = {
          waterType: 1,
          density: 3,
          thicknessGain:
            solver.particles.restVolume! / ((4 / 3) * Math.PI * solver.particles.radius ** 3),
          foamAmount: 0.75,
        };
        const measureSurface = async (quality: Parameters<typeof fluid.setQuality>[0]) => {
          fluid.setQuality(quality);
          // N renders in one submit; the uniforms are written with queue.writeBuffer, so repeated renders see the same
          // values, which is what is wanted here.
          return slope(
            device,
            (encoder, renders) => {
              for (let k = 0; k < renders; k++) {
                fluid.render(encoder, {
                  particles: solver.particles,
                  camera,
                  sceneColor: sceneColor.view,
                  sceneDepth: sceneDepth.view,
                  target: target.view,
                  params,
                });
              }
            },
            2,
            8,
          );
        };
        row.surface.push(
          await measureSurface({
            scale: 0.5,
            thicknessScale: 0.25,
            thicknessStride: 1,
            filterRadius: 2.2,
            iterations: 3,
          }),
        );
        row.cheap.push(
          await measureSurface({
            scale: 0.25,
            thicknessScale: 0.125,
            thicknessStride: 2,
            filterRadius: 1.4,
            iterations: 1,
          }),
        );
        solver.destroy();
      }
    }
    const rows: unknown[] = [];
    const lines: string[] = [];
    lines.push(
      '| grid | particles | radius (px, 1080p) | sub-steps | simulation | liquid surface | total (quietest) | cheap surface (stride 2, 1/4 chain) |',
    );
    lines.push('| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |');
    for (const row of frameRows.values()) {
      const sim = quietest(row.sim);
      const surface = quietest(row.surface);
      const cheap = quietest(row.cheap);
      rows.push({
        grid: row.grid,
        particles: row.particles,
        radiusPixels: row.radiusPixels,
        stepsPerFrame: row.stepsPerFrame,
        simulation: sim,
        surface,
        cheapSurface: cheap,
      });
      lines.push(
        `| ${row.grid.join('x')} | ${row.particles} | ${f(row.radiusPixels, 1)} | ${f(row.stepsPerFrame, 1)} | ${f(sim.best, 2)} (${f(sim.median, 2)}) | ${f(surface.best, 2)} (${f(surface.median, 2)}) | ${f(sim.best + surface.best, 2)} | ${f(cheap.best, 2)} (${f(cheap.median, 2)}) |`,
      );
    }
    const specifier = 'node:fs';
    const { writeFileSync } = (await import(/* @vite-ignore */ specifier)) as {
      writeFileSync(path: string, text: string): void;
    };
    writeFileSync(
      `${studyFile!}.frame.json`,
      JSON.stringify({ rows, markdown: lines.join('\n') }, null, 2),
    );
    console.log(`\nMPM FRAME COST (ms), 1920x1080\n${lines.join('\n')}\n`);
    fluid.destroy();
  });
});
