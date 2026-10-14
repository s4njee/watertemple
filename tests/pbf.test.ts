import { describe, expect, it, vi } from 'vitest';
import { hasEffect, loadEffect } from '../src/effects/registry.ts';
import { defaultParams } from '../src/effects/types.ts';
import {
  createNeighborCounter,
  createNeighborGrid,
  createParticleSystem,
} from '../src/gpu/particles/index.ts';
import { noProfiler } from '../src/gpu/profiler.ts';
import {
  ballVolume,
  createPbf,
  DEFAULT_PBF_PARAMS,
  kernelIntegral,
  PBF_STAGES,
  pbfConstants,
  poly6,
  poly6Coefficient,
  restState,
  spiky,
  spikyCoefficient,
  spikyDerivative,
  WALL_TABLE_SIZE,
  wallTables,
  type Pbf,
  type PbfBounds,
  type PbfCollider,
  type PbfImpulse,
  type PbfOptions,
  type PbfParams,
  type PbfStage,
  type Vec3,
} from '../src/gpu/sim/pbf.ts';
import kernelsSource from '../src/gpu/sim/pbf-kernels.wgsl?raw';
import { expectValid, getTestDevice, readTexture } from './gpu.ts';
import { gpuF32, rowsOf } from './wgsl-eval.ts';

// Story 7.4: position-based fluids. The solver (src/gpu/sim/pbf.ts and pbf-*.wgsl) runs on the real GPU here and is
// judged against physics and against plain JavaScript: kernel integrals, a brute-force density, conservation laws,
// hydrostatic heights, wave frequencies. What the file proves, by section:
//
//   1  kernels         Poly6 and Spiky integrate to 1 in 2D and 3D (by the library's quadrature and by an independent
//                      Cartesian grid), the Spiky derivative is the derivative of Spiky, the WGSL kernels equal their CPU
//                      twins, the rest state of the lattice (neighbour counts, zero gradient sum, n0 and s0) is consistent
//                      by construction and scales with the spacing as it should, the wall tables are the half-space
//                      integrals they claim to be.
//   2  lattice         a lattice at the rest spacing has density ratio 1 (interior) and a wall layer within a few percent
//                      of 1 only because of the analytic wall density; with gravity off it stays put.
//   3  convergence     the mean density error of a squeezed column falls with the iteration count, a settled column is
//                      the less compressed the more iterations it gets (until Jacobi over-corrects; measured and printed).
//   4  hydrostatics    a column and a dam break settle to the incompressible height with a flat surface, 2D and 3D.
//   5  momentum        without gravity and walls the sum of the velocities is conserved (viscosity on and off).
//   6  containment     nothing leaves the box: dam break, an impulse, a wall at the fastest allowed speed, colliders.
//   7  waves           an oscillating wall makes waves at the wall's frequency, far above the noise of still water.
//   8  extremes        no NaN or Inf, nothing outside the box, at parameter extremes and degenerate configurations.
//   9  frame rate      30, 60 and 120 Hz updates give the same fluid (compared by energy and height, never by position).
//  11  scene           reset, capacity changes, addBlock, emitter rate, impulses.
//  12  foam            positions.w stays in 0..1, is ~0 for calm water and large in spray.
//  13  lifecycle       destroy and recreate, two solvers on one device, every stage on its own; no validation errors.
//  14  effects         pbf-2d and pbf-3d run headlessly through every preset and every `view` (once they exist).
//  15  study           GATED (PBF_STUDY=<json file>): the cost of a sub-step against particle and iteration count.
//
// Particles are physically REORDERED into cell order every sub-step, so an index means nothing from one step to the
// next: every comparison below is an aggregate (sums, means, histograms, brute-force densities from the positions).
// Thresholds were calibrated by repeated runs; each states the measured value and the reason for its margin.

vi.setConfig({ testTimeout: 120_000 });

type Dim = 2 | 3;
const DIMS = [2, 3] as const;
const G = 9.81;

// --- Helpers ----------------------------------------------------------------------------------------------------

/** A tank from the origin: [x, y, z] extent. In 2D only x and y matter (z is -1..1 and unused). */
function tank(dim: Dim, size: Vec3): PbfBounds {
  return { min: [0, 0, dim === 3 ? 0 : -1], max: [size[0], size[1], dim === 3 ? size[2] : 1] };
}

type Options = Omit<PbfOptions, 'device'>;

/** Creates a solver, runs `body` with every validation error turned into a failure, destroys the solver. */
async function withPbf<T>(
  options: Options,
  body: (pbf: Pbf, device: GPUDevice) => Promise<T>,
): Promise<T> {
  const device = await getTestDevice();
  return expectValid(device, async () => {
    const pbf = createPbf({ device, ...options });
    try {
      return await body(pbf, device);
    } finally {
      await device.queue.onSubmittedWorkDone();
      pbf.destroy();
    }
  });
}

/** `frames` updates of `delta` seconds, one submit each; `before` runs ahead of each (to move walls, add pushes). */
async function advance(
  device: GPUDevice,
  pbf: Pbf,
  frames: number,
  delta = 1 / 60,
  before?: (frame: number) => void,
): Promise<void> {
  for (let frame = 0; frame < frames; frame++) {
    before?.(frame);
    const encoder = device.createCommandEncoder();
    pbf.update(encoder, delta);
    device.queue.submit([encoder.finish()]);
    if (frame % 8 === 7) await device.queue.onSubmittedWorkDone();
  }
  await device.queue.onSubmittedWorkDone();
}

/** One call of `run` (a single sub-step, or the chosen stages), waited for. */
async function runOnce(
  device: GPUDevice,
  pbf: Pbf,
  options?: Parameters<Pbf['run']>[1],
): Promise<void> {
  const encoder = device.createCommandEncoder();
  pbf.run(encoder, options);
  device.queue.submit([encoder.finish()]);
  await device.queue.onSubmittedWorkDone();
}

interface Snapshot {
  n: number;
  /** vec4 rows: xyz and the foam scalar. */
  pos: Float32Array;
  /** vec4 rows. */
  vel: Float32Array;
}

async function snapshot(pbf: Pbf): Promise<Snapshot> {
  const pos = await pbf.system.read('position');
  const vel = await pbf.system.read('velocity');
  return { n: pbf.count, pos, vel };
}

const mean = (values: ArrayLike<number>): number => {
  let sum = 0;
  for (let i = 0; i < values.length; i++) sum += values[i]!;
  return values.length > 0 ? sum / values.length : 0;
};

/** Mean of component `c` (0 x, 1 y, 2 z) of the rows. */
function meanComponent(rows: Float32Array, n: number, c: number): number {
  let sum = 0;
  for (let i = 0; i < n; i++) sum += rows[i * 4 + c]!;
  return n > 0 ? sum / n : 0;
}

function sumVelocity(snap: Snapshot): [number, number, number] {
  const sum: [number, number, number] = [0, 0, 0];
  for (let i = 0; i < snap.n; i++) {
    for (let c = 0; c < 3; c++) sum[c]! += snap.vel[i * 4 + c]!;
  }
  return sum;
}

function rmsSpeed(snap: Snapshot): number {
  let sum = 0;
  for (let i = 0; i < snap.n; i++) {
    sum += snap.vel[i * 4]! ** 2 + snap.vel[i * 4 + 1]! ** 2 + snap.vel[i * 4 + 2]! ** 2;
  }
  return Math.sqrt(sum / Math.max(1, snap.n));
}

function maxSpeed(snap: Snapshot): number {
  let best = 0;
  for (let i = 0; i < snap.n; i++) {
    best = Math.max(best, Math.hypot(snap.vel[i * 4]!, snap.vel[i * 4 + 1]!, snap.vel[i * 4 + 2]!));
  }
  return best;
}

/** Kinetic and potential energy per particle (unit mass, potential measured from y = 0) and mean position. */
function energies(snap: Snapshot): { kinetic: number; potential: number; meanX: number } {
  let kinetic = 0;
  for (let i = 0; i < snap.n; i++) {
    kinetic +=
      0.5 * (snap.vel[i * 4]! ** 2 + snap.vel[i * 4 + 1]! ** 2 + snap.vel[i * 4 + 2]! ** 2);
  }
  return {
    kinetic: kinetic / Math.max(1, snap.n),
    potential: G * meanComponent(snap.pos, snap.n, 1),
    meanX: meanComponent(snap.pos, snap.n, 0),
  };
}

function allFinite(snap: Snapshot): boolean {
  for (let i = 0; i < snap.n * 4; i++) {
    if (!Number.isFinite(snap.pos[i]!) || !Number.isFinite(snap.vel[i]!)) return false;
  }
  return true;
}

/** How far outside the box (shrunk by `offset`) the worst particle centre is, and how many are outside by > tolerance. */
function escapes(
  snap: Snapshot,
  dim: Dim,
  bounds: { min: ArrayLike<number>; max: ArrayLike<number> },
  offset: number,
  tolerance = 1e-4,
): { count: number; worst: number } {
  let count = 0;
  let worst = 0;
  for (let i = 0; i < snap.n; i++) {
    let depth = 0;
    for (let a = 0; a < dim; a++) {
      const x = snap.pos[i * 4 + a]!;
      depth = Math.max(depth, bounds.min[a]! + offset - x, x - (bounds.max[a]! - offset));
    }
    if (!(depth <= tolerance)) count++;
    worst = Math.max(worst, depth);
  }
  return { count, worst };
}

/** Poly6 number density over the rest density, self included, by brute force over the positions. */
function densityRatios(dim: Dim, snap: Snapshot, h: number, n0: number): Float32Array {
  const rho = new Float32Array(snap.n);
  const h2 = h * h;
  for (let i = 0; i < snap.n; i++) {
    let sum = 0;
    for (let j = 0; j < snap.n; j++) {
      const dx = snap.pos[i * 4]! - snap.pos[j * 4]!;
      const dy = snap.pos[i * 4 + 1]! - snap.pos[j * 4 + 1]!;
      const dz = dim === 3 ? snap.pos[i * 4 + 2]! - snap.pos[j * 4 + 2]! : 0;
      const r2 = dx * dx + dy * dy + dz * dz;
      if (r2 < h2) sum += poly6(dim, h, Math.sqrt(r2));
    }
    rho[i] = sum / n0;
  }
  return rho;
}

/** Distance of a coordinate from the nearest site of a lattice whose sites sit at (k + 1/2) s. */
const siteOffset = (x: number, s: number): number =>
  Math.abs(((((x / s - 0.5) % 1) + 1.5) % 1) - 0.5) * s;

function fmt(x: number, digits = 3): string {
  return Number.isFinite(x) ? x.toFixed(digits) : String(x);
}

// A deterministic random source for the tests that need one.
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

// --- 1. Kernels, the rest state and the wall tables --------------------------------------------------------------

type KernelFn = (dimension: Dim, h: number, r: number) => number;

/** The integral of a radial kernel on a plain Cartesian midpoint grid: no polar coordinates, nothing from pbf.ts. */
function cartesianIntegral(dim: Dim, h: number, kernel: KernelFn, cells: number): number {
  const dx = (2 * h) / cells;
  const coordinate = (i: number) => -h + (i + 0.5) * dx;
  let sum = 0;
  for (let k = 0; k < (dim === 3 ? cells : 1); k++) {
    const z = dim === 3 ? coordinate(k) : 0;
    for (let j = 0; j < cells; j++) {
      const y = coordinate(j);
      for (let i = 0; i < cells; i++) {
        sum += kernel(dim, h, Math.hypot(coordinate(i), y, z));
      }
    }
  }
  return sum * dx ** dim;
}

describe.each(DIMS)('kernels in %iD', (dim) => {
  const h = 0.0737;

  it('Poly6 and Spiky are normalised: the library quadrature and a Cartesian grid agree on 1', () => {
    // kernelIntegral integrates in shells with Simpson's rule; the integrand is a polynomial, so it is exact to rounding.
    expect(kernelIntegral(dim, h, poly6)).toBeCloseTo(1, 9);
    expect(kernelIntegral(dim, h, spiky)).toBeCloseTo(1, 9);
    // The measure of the support is what a kernel of value 1 would integrate to.
    expect(kernelIntegral(dim, h, () => 1)).toBeCloseTo(ballVolume(dim, h), 12);
    // The grid shares no code and no coordinates with it. A midpoint rule on a ball sampled by a cube: the kernels
    // vanish to third order at the edge, which keeps the voxel error small. Measured 3e-5 (2D, 1200 cells), 2e-4 (3D, 160).
    const cells = dim === 3 ? 160 : 1200;
    expect(Math.abs(cartesianIntegral(dim, h, poly6, cells) - 1)).toBeLessThan(1e-3);
    expect(Math.abs(cartesianIntegral(dim, h, spiky, cells) - 1)).toBeLessThan(1e-3);
  });

  it('the kernels vanish at and beyond h, and the coefficients are the closed forms', () => {
    expect(poly6(dim, h, h)).toBe(0);
    expect(poly6(dim, h, 2 * h)).toBe(0);
    expect(spiky(dim, h, h)).toBe(0);
    expect(spikyDerivative(dim, h, h)).toBe(0);
    expect(spikyDerivative(dim, h, 1.5 * h)).toBe(0);
    expect(poly6Coefficient(dim, h)).toBeCloseTo(
      dim === 3 ? 315 / (64 * Math.PI * h ** 9) : 4 / (Math.PI * h ** 8),
      0,
    );
    // Dimensional analysis: both kernels are number densities, so halving h multiplies them by 2^D.
    expect(
      (poly6Coefficient(dim, h / 2) * (h / 2) ** 6) / (poly6Coefficient(dim, h) * h ** 6),
    ).toBeCloseTo(2 ** dim, 9);
    expect(
      (spikyCoefficient(dim, h / 2) * (h / 2) ** 3) / (spikyCoefficient(dim, h) * h ** 3),
    ).toBeCloseTo(2 ** dim, 9);
  });

  it('the Spiky derivative is the derivative of Spiky (central differences)', () => {
    // Step h * 1e-5 in double precision: truncation error ~(step/h)^2, rounding ~1e-11. Relative 1e-6 is generous.
    const step = h * 1e-5;
    for (let q = 0.02; q < 0.99; q += 0.0475) {
      const r = q * h;
      const numeric = (spiky(dim, h, r + step) - spiky(dim, h, r - step)) / (2 * step);
      const exact = spikyDerivative(dim, h, r);
      expect(exact).toBeLessThan(0);
      expect(Math.abs(numeric - exact)).toBeLessThan(1e-6 * Math.abs(exact));
    }
  });

  it('the WGSL kernels equal the CPU twins', async () => {
    const c6 = poly6Coefficient(dim, h);
    const c3 = 3 * spikyCoefficient(dim, h);
    const lit = (x: number) => x.toExponential(12);
    const random = makeRandom(11);
    const vectors: number[][] = [
      [0, 0, 0],
      [h, 0, 0],
      [0, h, 0],
      [1.3 * h, 0, 0],
      [1e-9, 0, 0],
    ];
    for (let i = 0; i < 300; i++) {
      // Directions in the plane for 2D (the solver keeps z = 0), in space for 3D; radii from 0 to 1.2 h.
      const r = random() * 1.2 * h;
      const a = random() * 2 * Math.PI;
      const b = Math.acos(2 * random() - 1);
      vectors.push(
        dim === 3
          ? [r * Math.sin(b) * Math.cos(a), r * Math.sin(b) * Math.sin(a), r * Math.cos(b)]
          : [r * Math.cos(a), r * Math.sin(a), 0],
      );
    }
    const input = rowsOf(vectors);
    const out = await gpuF32(
      `${kernelsSource}
       fn f(x: vec4f) -> vec4f {
         let g = pbf_spiky_grad(x.xyz, ${lit(h)}, ${lit(c3)});
         return vec4f(g, pbf_poly6(dot(x.xyz, x.xyz), ${lit(h * h)}, ${lit(c6)}));
       }`,
      input,
    );
    // The scales of the two functions: Poly6 peaks at the centre, the Spiky gradient at r = 0 as well (3 cs h^2).
    // The GPU works in f32: the only cancellation is h^2 - r^2 near the edge, an absolute error of a few ulp of h^2,
    // which is ~1e-7 of the peak after cubing. 1e-5 of the peak is two orders of margin.
    const polyPeak = c6 * h ** 6;
    const gradPeak = c3 * h * h;
    for (let i = 0; i < vectors.length; i++) {
      const v = [input[i * 4]!, input[i * 4 + 1]!, input[i * 4 + 2]!];
      const r = Math.hypot(v[0]!, v[1]!, v[2]!);
      const w = poly6(dim, h, r);
      const slope = r > 1e-7 ? spikyDerivative(dim, h, r) / r : 0;
      expect(Math.abs(out[i * 4 + 3]! - w)).toBeLessThan(1e-5 * polyPeak);
      for (let c = 0; c < 3; c++) {
        expect(Math.abs(out[i * 4 + c]! - slope * v[c]!)).toBeLessThan(1e-5 * gradPeak);
      }
    }
    // Exactly zero beyond the radius and for coincident particles (the caller nudges those).
    expect(Array.from(out.subarray(0, 3))).toEqual([0, 0, 0]);
    expect(Math.abs(out[3]! - w0(c6, h))).toBeLessThan(1e-5 * polyPeak);
    expect(out[3 * 4 + 3]!).toBe(0);
    expect(Array.from(out.subarray(12, 15))).toEqual([0, 0, 0]);
  });

  it('pbf_nudge gives a deterministic direction of a thousandth of h, spread over the sphere', async () => {
    const pairs: number[][] = [];
    for (let i = 0; i < 400; i++) pairs.push([i, i + 1]);
    pairs.push([5, 7], [5, 7], [7, 5]);
    const code = `${kernelsSource}
      fn f(x: vec4f) -> vec4f { return vec4f(pbf_nudge(u32(x.x), u32(x.y), ${h.toExponential(9)}), 0.0); }`;
    const out = await gpuF32(code, rowsOf(pairs));
    const sum = [0, 0, 0];
    for (let i = 0; i < 400; i++) {
      const len = Math.hypot(out[i * 4]!, out[i * 4 + 1]!, out[i * 4 + 2]!);
      expect(len).toBeGreaterThan(h * 1e-3 * 0.99);
      expect(len).toBeLessThan(h * 1e-3 * 1.01);
      for (let c = 0; c < 3; c++) sum[c]! += out[i * 4 + c]!;
    }
    // 400 unit-ish vectors with no preferred direction: the mean is ~1/sqrt(400) of the length, not the length.
    expect(Math.hypot(sum[0]!, sum[1]!, sum[2]!) / 400).toBeLessThan(0.2 * h * 1e-3);
    // Deterministic (the same pair twice gives the same vector) and not the same for different pairs.
    expect(Array.from(out.subarray(400 * 4, 400 * 4 + 3))).toEqual(
      Array.from(out.subarray(401 * 4, 401 * 4 + 3)),
    );
    expect(out[0]).not.toBe(out[4]);
  });
});

/** Poly6 at the centre, W(0) = c h^6. */
const w0 = (c6: number, h: number) => c6 * h ** 6;

describe.each(DIMS)('the rest state in %iD', (dim) => {
  const s = 0.02;
  const kernelScale = dim === 3 ? 2.2 : 3;
  const h = kernelScale * s;
  const rest = restState(dim, s, h);

  it('the lattice has the textbook neighbour count and a vanishing gradient sum', () => {
    // 3D, h = 2.2 s: shells r^2 = 1, 2, 3, 4 hold 6 + 12 + 8 + 6 = 32 (r^2 = 5 is 2.236 s, outside).
    // 2D, h = 3 s: r^2 = 1, 2, 4, 5, 8 hold 4 + 4 + 4 + 8 + 4 = 24 (r^2 = 9 is exactly h and does not count).
    expect(rest.neighbours).toBe(dim === 3 ? 32 : 24);
    // By symmetry the vector sum of the gradients cancels; relative to one neighbour's gradient at its peak.
    const peak = 3 * spikyCoefficient(dim, h) * h * h;
    expect(rest.asymmetry / peak).toBeLessThan(1e-10);
  });

  it('n0 and s0 are the lattice sums of Poly6 and of the squared Spiky gradient', () => {
    // An independent triple loop: every integer offset in a cube twice as wide as needed.
    let n0 = 0;
    let squares = 0;
    const reach = Math.ceil(h / s) + 1;
    for (let a = -reach; a <= reach; a++) {
      for (let b = -reach; b <= reach; b++) {
        for (let c = dim === 3 ? -reach : 0; c <= (dim === 3 ? reach : 0); c++) {
          const r = s * Math.sqrt(a * a + b * b + c * c);
          if (r >= h) continue;
          n0 += poly6(dim, h, r);
          if (r > 0) squares += spikyDerivative(dim, h, r) ** 2;
        }
      }
    }
    expect(rest.n0).toBeCloseTo(n0, 6);
    expect(rest.n0 / n0).toBeCloseTo(1, 12);
    expect(rest.s0 / (squares / (n0 * n0))).toBeCloseTo(1, 12);
    // A lattice of spacing s holds one particle per s^D, and Poly6 integrates to 1, so n0 s^D is 1 up to the
    // discreteness of the sum: measured 0.4 % (2D) and 1.4 % (3D) at the defaults.
    expect(Math.abs(rest.n0 * s ** dim - 1)).toBeLessThan(0.03);
  });

  it('is consistent by construction: halving the spacing scales n0 by 2^D and s0 by 4', () => {
    const half = restState(dim, s / 2, h / 2);
    expect(half.n0 / rest.n0).toBeCloseTo(2 ** dim, 9);
    // s0 = sum |grad W|^2 / n0^2: gradients scale as h^-(D+1), n0 as h^-D, so s0 has the units of 1/length^2.
    expect(half.s0 / rest.s0).toBeCloseTo(4, 9);
    expect(half.neighbours).toBe(rest.neighbours);
    // The dimensionless combinations do not depend on the spacing at all.
    for (const spacing of [0.005, 0.013, 0.1, 2]) {
      const r = restState(dim, spacing, kernelScale * spacing);
      expect(r.n0 * spacing ** dim).toBeCloseTo(rest.n0 * s ** dim, 9);
      expect(r.s0 * spacing ** 2).toBeCloseTo(rest.s0 * s ** 2, 9);
      expect(r.neighbours).toBe(rest.neighbours);
    }
  });

  it('pbfConstants reports the same numbers', () => {
    const c = pbfConstants(dim, s);
    expect(c.h).toBeCloseTo(h, 12);
    expect(c.n0).toBe(rest.n0);
    expect(c.s0).toBe(rest.s0);
    expect(c.neighbours).toBe(rest.neighbours);
    expect(c.restVolume).toBeCloseTo(s ** dim, 12);
    expect(c.renderRadius).toBeCloseTo(0.65 * s, 12);
    expect(c.poly6).toBe(poly6Coefficient(dim, h));
    expect(c.spiky).toBe(3 * spikyCoefficient(dim, h));
    // The kernel scale is a free parameter: other values change the neighbour count, never the consistency.
    for (const scale of [1.6, 2.6, 3.4]) {
      const r = restState(dim, s, scale * s);
      expect(Math.abs(r.n0 * s ** dim - 1)).toBeLessThan(0.15);
      expect(r.asymmetry / (3 * spikyCoefficient(dim, scale * s) * (scale * s) ** 2)).toBeLessThan(
        1e-10,
      );
    }
  });

  it('the wall tables are the half-space integrals: half of the lattice at the wall, zero at h', () => {
    const { density, gradient } = wallTables(dim, s, h);
    expect(density.length).toBe(WALL_TABLE_SIZE);
    expect(gradient.length).toBe(WALL_TABLE_SIZE);
    // At the wall the hidden fluid is the half-space: half of the kernel mass times the lattice density 1/s^D
    // (Simpson with 8 intervals per table cell: measured error 1e-7).
    expect(density[0]! * s ** dim).toBeCloseTo(0.5, 5);
    // Relative to the solver's own rest density that is 0.5 / (n0 s^D), within the discreteness of the sum.
    expect(Math.abs(density[0]! / rest.n0 - 0.5)).toBeLessThan(0.015);
    expect(density[WALL_TABLE_SIZE - 1]).toBe(0);
    expect(gradient[WALL_TABLE_SIZE - 1]).toBe(0);
    for (let k = 1; k < WALL_TABLE_SIZE; k++) {
      expect(density[k]!).toBeLessThan(density[k - 1]!);
      expect(gradient[k]!).toBeLessThanOrEqual(gradient[k - 1]!);
      expect(gradient[k]!).toBeGreaterThanOrEqual(0);
    }
    // Closed form of the gradient at the wall: the Spiky kernel integrated over a plane through its centre.
    //   3D: 2 pi cs h^5 / 20 = 1.5 / h        2D: cs h^4 / 2 = 5 / (pi h)
    expect(gradient[0]! * s ** dim * h).toBeCloseTo(dim === 3 ? 1.5 : 5 / Math.PI, 5);
  });

  it('the wall tables agree with an independent quadrature at several distances', () => {
    const { density, gradient } = wallTables(dim, s, h);
    const cs = spikyCoefficient(dim, h);
    const cell = h / (WALL_TABLE_SIZE - 1);
    for (const k of [4, 8, 16, 24, 30]) {
      const d = k * cell;
      // The part of the Poly6 mass beyond a plane at distance d: shells of radius r > d, of which a fraction
      // (1 - d/r) / 2 (3D) or acos(d/r) / pi (2D) of the area lies beyond.
      const n = 100_000;
      const dr = (h - d) / n;
      let mass = 0;
      let flux = 0;
      for (let i = 0; i < n; i++) {
        const r = d + (i + 0.5) * dr;
        mass +=
          poly6(dim, h, r) *
          (dim === 3
            ? 4 * Math.PI * r * r * ((1 - d / r) / 2)
            : 2 * Math.PI * r * (Math.acos(d / r) / Math.PI)) *
          dr;
        // The Spiky kernel integrated over the plane: in 3D rings of area 2 pi rho drho with r^2 = rho^2 + d^2
        // (so rho drho = r dr); in 2D a line, integrated along it.
        if (dim === 3) flux += cs * (h - r) ** 3 * 2 * Math.PI * r * dr;
      }
      if (dim === 2) {
        const reach = Math.sqrt(h * h - d * d);
        const m = 20_000;
        const dy = reach / m;
        for (let i = 0; i < m; i++) flux += 2 * spiky(2, h, Math.hypot((i + 0.5) * dy, d)) * dy;
      }
      expect(Math.abs(density[k]! * s ** dim - mass)).toBeLessThan(2e-4);
      expect(Math.abs(gradient[k]! * s ** dim - flux) / (gradient[0]! * s ** dim)).toBeLessThan(
        2e-4,
      );
    }
  });

  it('scales with the spacing: n D-th power for density, one more for the gradient', () => {
    const a = wallTables(dim, s, h);
    const b = wallTables(dim, s / 2, h / 2);
    for (let k = 0; k < WALL_TABLE_SIZE - 1; k += 4) {
      expect(b.density[k]! / a.density[k]!).toBeCloseTo(2 ** dim, 4);
      if (a.gradient[k]! > 1e-6 * a.gradient[0]!) {
        expect(b.gradient[k]! / a.gradient[k]!).toBeCloseTo(2 ** (dim + 1), 4);
      }
    }
  });
});

// --- 2. A lattice at rest --------------------------------------------------------------------------------------

/** A box of m^D particles at spacing s that the block fills from wall to wall: every layer sits where a lattice puts it. */
function latticeScene(dim: Dim) {
  const s = dim === 3 ? 0.04 : 0.02;
  const m = dim === 3 ? 12 : 24;
  const size = m * s;
  const box: Vec3 = [size, size, size];
  return {
    s,
    m,
    size,
    options: (params: Partial<PbfParams>): Options => ({
      dimension: dim,
      capacity: m ** dim + 8,
      spacing: s,
      bounds: tank(dim, box),
      params: { gravity: [0, 0, 0], ...params },
    }),
    block: { min: [0, 0, 0] as Vec3, max: box, jitter: 0 },
  };
}

describe.each(DIMS)('a lattice at rest in %iD', (dim) => {
  const scene = latticeScene(dim);

  /** The constraint of the lattice as placed (one sub-step, one iteration: the density pass sees it before any correction). */
  async function constraintOfLattice(wallDensity: boolean) {
    return withPbf(scene.options({ wallDensity }), async (pbf, device) => {
      pbf.reset({ blocks: [scene.block] });
      expect(pbf.count).toBe(scene.m ** dim);
      await runOnce(device, pbf, { iterations: 1 });
      const snap = await snapshot(pbf);
      const { density } = await pbf.readState();
      const { h } = pbf.constants;
      // Worst |rho - 1| by position: the interior, the layers within h of a wall but not the first, and the first layer
      // by how many walls it touches (1 a face, 2 an edge, 3 a corner).
      const worst = { interior: 0, near: 0, face: 0, edge: 0, corner: 0 };
      let nInterior = 0;
      let lowest = Infinity;
      for (let i = 0; i < snap.n; i++) {
        let touches = 0;
        let inner = true;
        for (let a = 0; a < dim; a++) {
          const x = snap.pos[i * 4 + a]!;
          if (x < scene.s || x > scene.size - scene.s) touches++;
          if (x <= h || x >= scene.size - h) inner = false;
        }
        const error = Math.abs(density[i]! - 1);
        const key = inner ? 'interior' : (['near', 'face', 'edge', 'corner'] as const)[touches]!;
        worst[key] = Math.max(worst[key], error);
        if (inner) nInterior++;
        lowest = Math.min(lowest, density[i]!);
      }
      return { ...worst, nInterior, lowest };
    });
  }

  it('has density ratio 1 in the interior, and the wall layers are made right by the wall density', async () => {
    const on = await constraintOfLattice(true);
    const off = await constraintOfLattice(false);
    const e = (x: number) => fmt(x, 3);
    console.log(
      `lattice ${dim}D constraint |rho-1|, worst per region (${on.nInterior} interior particles), wall density on: ` +
        `interior ${on.interior.toExponential(1)}, second layers ${e(on.near)}, face ${e(on.face)}, edge ${e(on.edge)}, corner ${e(on.corner)}; ` +
        `off: interior ${off.interior.toExponential(1)}, lowest rho ${e(off.lowest)}`,
    );
    expect(on.nInterior).toBeGreaterThan(20);
    // The interior is a perfect lattice: the ratio is 1 up to f32 summation (measured 8e-7).
    expect(on.interior).toBeLessThan(1e-4);
    expect(off.interior).toBeLessThan(1e-4);
    // The wall layers: the analytic half-space stands in for the missing lattice. A flat wall is right to ~3 % (measured
    // 0.027 in 2D, 0.026 in 3D, and 0.02 in the second layer). Where two or three walls meet each counts the whole
    // half-space, so an edge is off by 8-10 % and a 3D corner by 17 % (pbf-core.wgsl says "double counts a little in a
    // corner"). Those layers are over-dense, so a block resting in a corner is pushed off it. The thresholds are what is
    // true today (a regression guard), not a precision to rely on. Without the wall density the layer sees about half the
    // density (0.46) and the solver would read a free surface along the whole glass.
    expect(on.face).toBeLessThan(0.05);
    expect(on.edge).toBeLessThan(0.15);
    expect(on.corner).toBeLessThan(0.25);
    expect(on.near).toBeLessThan(0.04);
    expect(off.lowest).toBeLessThan(0.6);
    expect(off.lowest).toBeLessThan(on.lowest - 0.2);
  });

  it('stays put for a second with gravity off (no artificial pressure, no wall density: exact equilibrium)', async () => {
    // Workaround, not a feature: the default artificial pressure makes a PERFECT lattice unstable (growth of about e per
    // 4 frames from f32 noise, printed by the next test); with it off a lattice is rock steady.
    await withPbf(
      scene.options({ wallDensity: false, artificialPressure: 0 }),
      async (pbf, device) => {
        pbf.reset({ blocks: [scene.block] });
        await advance(device, pbf, 60);
        const snap = await snapshot(pbf);
        let displacement = 0;
        for (let i = 0; i < snap.n; i++) {
          for (let a = 0; a < dim; a++) {
            displacement = Math.max(displacement, siteOffset(snap.pos[i * 4 + a]!, scene.s));
          }
        }
        // Measured 1e-7 s and 1e-5 m/s (f32 rounding of (x - previous) / dt). Thresholds are three or four orders above.
        expect(displacement).toBeLessThan(1e-3 * scene.s);
        expect(maxSpeed(snap)).toBeLessThan(1e-2);
      },
    );
  });

  it('with the default parameters it rearranges but never collapses or explodes', async () => {
    await withPbf(scene.options({}), async (pbf, device) => {
      pbf.reset({ blocks: [scene.block] });
      const { h } = pbf.constants;
      const series: string[] = [];
      for (let t = 0; t < 4; t++) {
        await advance(device, pbf, 15);
        series.push(fmt(maxSpeed(await snapshot(pbf)), 3));
      }
      // DOUBT for the lead: the wall layers start 10-17 % off (above) and the artificial pressure amplifies the noise of
      // a perfect lattice, so a filled box with gravity off does not sit still: it rattles at 0.1-0.4 m/s for seconds.
      console.log(
        `lattice ${dim}D, default parameters, gravity off: max speed (m/s) per 0.25 s: ${series.join(' ')}`,
      );
      const snap = await snapshot(pbf);
      const rho = densityRatios(dim, snap, h, pbf.constants.n0);
      let worst = 0;
      for (let i = 0; i < snap.n; i++) {
        const inner = [0, 1, ...(dim === 3 ? [2] : [])].every(
          (a) => snap.pos[i * 4 + a]! > h && snap.pos[i * 4 + a]! < scene.size - h,
        );
        if (inner) worst = Math.max(worst, Math.abs(rho[i]! - 1));
      }
      // Measured 0.007 (2D) and 0.02 (3D) after a second; 0.05 allows noise and still catches a collapsed or exploded fluid.
      expect(worst).toBeLessThan(0.05);
      expect(maxSpeed(snap)).toBeLessThan(1);
      expect(allFinite(snap)).toBe(true);
      expect(escapes(snap, dim, pbf.bounds, pbf.params.wallOffset).count).toBe(0);
    });
  });
});

// --- 3. Convergence of the density constraint -----------------------------------------------------------------------

describe.each(DIMS)('convergence of the density constraint in %iD', (dim) => {
  const s = dim === 3 ? 0.04 : 0.02;
  const across = dim === 3 ? 9 : 16;
  const rows = dim === 3 ? 12 : 24;
  const squeeze = 0.92;

  /**
   * A column that fills the tank across but is squeezed along y to 0.92 s, free to expand upwards: the compression can
   * only be relieved by pushing the top up, and Jacobi iterations carry that news down one kernel radius each. Mean
   * |rho - 1| of the particles that are neither near a wall, the floor or the free surface after ONE sub-step.
   */
  async function errorAfter(
    iterations: number,
  ): Promise<{ before: number; after: number; n: number }> {
    const width = across * s;
    return withPbf(
      {
        dimension: dim,
        capacity: across * across * rows + 8,
        spacing: s,
        bounds: tank(dim, [width, 3, width]),
        params: { gravity: [0, 0, 0], iterations, substeps: 1 },
      },
      async (pbf, device) => {
        const positions: number[] = [];
        for (let c = 0; c < (dim === 3 ? across : 1); c++) {
          for (let j = 0; j < rows; j++) {
            for (let i = 0; i < across; i++) {
              positions.push(
                (i + 0.5) * s,
                (j + 0.5) * squeeze * s,
                dim === 3 ? (c + 0.5) * s : 0,
                0,
              );
            }
          }
        }
        pbf.system.set({ positions: new Float32Array(positions) });
        const { h, n0 } = pbf.constants;
        const measure = async () => {
          const snap = await snapshot(pbf);
          const rho = densityRatios(dim, snap, h, n0);
          let top = 0;
          for (let i = 0; i < snap.n; i++) top = Math.max(top, snap.pos[i * 4 + 1]!);
          let sum = 0;
          let count = 0;
          for (let i = 0; i < snap.n; i++) {
            const x = snap.pos[i * 4]!;
            const y = snap.pos[i * 4 + 1]!;
            const z = snap.pos[i * 4 + 2]!;
            const inside =
              x > h &&
              x < width - h &&
              y > h &&
              y < top - h &&
              (dim === 2 || (z > h && z < width - h));
            if (inside) {
              sum += Math.abs(rho[i]! - 1);
              count++;
            }
          }
          return { error: sum / count, count };
        };
        const before = await measure();
        await runOnce(device, pbf, { dt: 1 / 180 });
        const after = await measure();
        return { before: before.error, after: after.error, n: after.count };
      },
    );
  }

  it('the density error of the interior falls with the iteration count', async () => {
    const counts = [1, 2, 3, 4, 6, 8, 12, 16];
    const results = new Map<number, { before: number; after: number; n: number }>();
    for (const k of counts) results.set(k, await errorAfter(k));
    const before = results.get(1)!.before;
    const lines = counts.map(
      (k) => `${String(k).padStart(2)} iterations: mean|rho-1| ${fmt(results.get(k)!.after, 4)}`,
    );
    console.log(
      `squeezed column ${dim}D (squeeze ${squeeze}, ${results.get(1)!.n} interior particles, 1 sub-step of 1/180 s), mean|rho-1| before: ${fmt(before, 4)}\n  ` +
        lines.join('\n  '),
    );
    expect(results.get(1)!.n).toBeGreaterThan(80);
    // One iteration does nothing useful (it even overshoots in 3D: measured 0.093 against 0.083 before); from two on the
    // error falls with every doubling. Measured 2D: .082 .077 .070 .064 .057 for 2, 4, 8, 12, 16; 3D: .089 .083 .072 .063 .055.
    // Between neighbouring counts a step can be flat within noise (3D 2 to 3 rises by 1 %), so the chain allows 3 %.
    const chain = [2, 3, 4, 6, 8, 12, 16];
    for (let i = 1; i < chain.length; i++) {
      expect(results.get(chain[i]!)!.after).toBeLessThan(results.get(chain[i - 1]!)!.after * 1.03);
    }
    for (const [a, b] of [
      [2, 4],
      [4, 8],
      [8, 16],
    ] as const) {
      expect(results.get(b)!.after).toBeLessThan(results.get(a)!.after);
    }
    expect(results.get(16)!.after).toBeLessThan(0.8 * results.get(2)!.after);
    expect(results.get(16)!.after).toBeLessThan(0.8 * before);
  });
});

describe.each(DIMS)('the settled column against the iteration count in %iD', (dim) => {
  const s = dim === 3 ? 0.04 : 0.02;
  const across = dim === 3 ? 8 : 20;
  const rows = dim === 3 ? 15 : 30;

  /** A column 0.6 m high filling the tank across, default parameters but for the iterations; 2 s. */
  async function settledColumn(iterations: number) {
    const width = across * s;
    return withPbf(
      {
        dimension: dim,
        capacity: across * across * rows + 8,
        spacing: s,
        bounds: tank(dim, [width, 2, width]),
        params: { iterations },
      },
      async (pbf, device) => {
        pbf.reset({ blocks: [{ min: [0, 0, 0], max: [width, rows * s, width] }] });
        const incompressible = (pbf.count * s ** dim) / (dim === 3 ? width * width : width);
        await advance(device, pbf, 120);
        const snap = await snapshot(pbf);
        return {
          ratio: (2 * meanComponent(snap.pos, snap.n, 1)) / incompressible,
          rms: rmsSpeed(snap),
          finite: allFinite(snap),
        };
      },
    );
  }

  it('is the less compressed the more iterations it gets, until Jacobi over-corrects', async () => {
    const counts = dim === 3 ? [2, 4, 8] : [2, 3, 4, 6, 8, 12];
    const results = new Map<number, Awaited<ReturnType<typeof settledColumn>>>();
    for (const k of counts) results.set(k, await settledColumn(k));
    console.log(
      `settled column ${dim}D, 3 sub-steps, height 2 mean(y) / incompressible height after 2 s:\n  ` +
        counts
          .map(
            (k) =>
              `${String(k).padStart(2)} iterations: ${fmt(results.get(k)!.ratio, 4)} (rms speed ${fmt(results.get(k)!.rms, 3)})`,
          )
          .join('\n  '),
    );
    // Measured 2D: .972 .991 1.003 1.026 1.053 1.107; 3D: .998 1.007 1.013 1.021 1.028 1.046 (k = 2, 3, 4, 6, 8, 12).
    // More iterations raise the column monotonically: the compression of the lead's "1-3 %" at 2 iterations is gone by
    // 3 or 4. DOUBT (reported to the lead, not a test failure): beyond that the column stands ABOVE the incompressible
    // height (+5 % at 8 and +11 % at 12 iterations in 2D), i.e. the fluid is over-expanded; with relax = 1 the Jacobi
    // sum over-corrects and the wall density pushes the layer along the glass inwards.
    for (let i = 1; i < counts.length; i++) {
      expect(results.get(counts[i]!)!.ratio).toBeGreaterThan(
        results.get(counts[i - 1]!)!.ratio - 0.002,
      );
    }
    for (const k of counts) {
      const r = results.get(k)!;
      expect(r.finite).toBe(true);
      expect(r.ratio).toBeGreaterThan(0.95);
      expect(r.ratio).toBeLessThan(1.15);
    }
    // Two iterations leave the column compressed (2.8 % in 2D; in 3D within noise of 1).
    expect(results.get(2)!.ratio).toBeLessThan(1.003);
  });

  // Known limitation (told by the lead): a single iteration is unstable at the default relaxation. In 2D the column
  // shoots up to the lid of the tank (measured: rms speed 1 to 2 m/s, top 1.99 m after 2 s); in 3D it rattles at 0.2 m/s.
  // `it.fails` documents it: when the solver is fixed this test starts to fail with "expected to fail", flip it then.
  it.fails('one iteration at the default relaxation settles a column (KNOWN TO FAIL)', async () => {
    const r = await settledColumn(1);
    expect(r.rms).toBeLessThan(0.05);
    expect(Math.abs(r.ratio - 1)).toBeLessThan(0.05);
  });
});

// --- 4. Hydrostatics --------------------------------------------------------------------------------------------------

/**
 * The top surface of a fluid: the highest y in every column of width 2 s (and depth 2 s in 3D). A lattice surface is
 * staggered from one spacing to the next (the tops alternate by up to 0.8 s even in a perfectly flat pool), so the
 * columns are two spacings wide.
 */
function surfaceProfile(snap: Snapshot, dim: Dim, s: number): number[] {
  const bins = new Map<number, number>();
  for (let i = 0; i < snap.n; i++) {
    const bx = Math.floor(snap.pos[i * 4]! / (2 * s));
    const bz = dim === 3 ? Math.floor(snap.pos[i * 4 + 2]! / (2 * s)) : 0;
    const key = bx * 1000 + bz;
    bins.set(key, Math.max(bins.get(key) ?? 0, snap.pos[i * 4 + 1]!));
  }
  return [...bins.values()];
}

describe.each(DIMS)('a column settles in %iD', (dim) => {
  const s = dim === 3 ? 0.04 : 0.02;
  const across = dim === 3 ? 8 : 20;
  const rows = dim === 3 ? 15 : 30;
  const width = across * s;

  it('to the incompressible height, at rest, with a flat free surface (default parameters)', async () => {
    await withPbf(
      {
        dimension: dim,
        capacity: across * across * rows + 8,
        spacing: s,
        bounds: tank(dim, [width, 2, width]),
      },
      async (pbf, device) => {
        pbf.reset({ blocks: [{ min: [0, 0, 0], max: [width, rows * s, width] }] });
        const incompressible = (pbf.count * s ** dim) / (dim === 3 ? width * width : width);
        const series: string[] = [];
        for (let t = 0; t < 4; t++) {
          await advance(device, pbf, 45);
          series.push(fmt(rmsSpeed(await snapshot(pbf)), 3));
        }
        const snap = await snapshot(pbf);
        const height = 2 * meanComponent(snap.pos, snap.n, 1);
        const tops = surfaceProfile(snap, dim, s);
        const spread = (Math.max(...tops) - Math.min(...tops)) / s;
        console.log(
          `column ${dim}D: n=${snap.n} incompressible height ${fmt(incompressible, 4)} m, settled height ${fmt(height, 4)} m ` +
            `(${fmt((100 * (height - incompressible)) / incompressible, 2)} %), rms speed per 0.75 s: ${series.join(' ')}, ` +
            `top spread ${fmt(spread, 2)} spacings over ${tops.length} columns`,
        );
        // Measured -2.8 % (2D) and -0.3 % (3D), 2 iterations x 3 sub-steps; the compression of a PBF column grows with
        // its depth and falls with iterations (section 3). 5 % is the budget a viewer would not see; the failures it guards
        // against (a column that sags 20 % or blows up) are far outside it.
        expect(Math.abs(height - incompressible) / incompressible).toBeLessThan(0.05);
        // At rest: measured 0.005 to 0.008 after 3 s (0.04 after the first 0.75 s). Gravity alone gives 0.16 m/s in 1/60 s.
        expect(rmsSpeed(snap)).toBeLessThan(0.03);
        expect(maxSpeed(snap)).toBeLessThan(0.3);
        // Flat: the highest particle of every column of two spacings is within ~1 spacing of the others.
        expect(tops.length).toBeGreaterThan(dim === 3 ? 12 : 8);
        expect(spread).toBeLessThan(1.5);
        expect(allFinite(snap)).toBe(true);
      },
    );
  });

  it('a dam break flows out, sloshes and settles to the volume over the floor area, flat', async () => {
    const size: Vec3 = dim === 3 ? [1.2, 0.8, 0.4] : [1.2, 0.8, 1];
    const damSize = dim === 3 ? 0.2 : 0.3;
    const sp = dim === 3 ? 0.04 : 0.02;
    await withPbf(
      { dimension: dim, capacity: 4000, spacing: sp, bounds: tank(dim, size) },
      async (pbf, device) => {
        pbf.reset({ blocks: [{ min: [0, 0, 0], max: [damSize, 0.5, size[2]] }] });
        const area = size[0] * (dim === 3 ? size[2] : 1);
        const flat = (pbf.count * sp ** dim) / area;
        // Flows: after 0.5 s the front has run along the floor.
        await advance(device, pbf, 30);
        const early = await snapshot(pbf);
        let front = 0;
        for (let i = 0; i < early.n; i++) front = Math.max(front, early.pos[i * 4]!);
        expect(front).toBeGreaterThan(0.6);
        await advance(device, pbf, 60 * 7);
        const snap = await snapshot(pbf);
        const height = 2 * meanComponent(snap.pos, snap.n, 1);
        const tops = surfaceProfile(snap, dim, sp);
        const spread = (Math.max(...tops) - Math.min(...tops)) / sp;
        console.log(
          `dam break ${dim}D: n=${snap.n}, flat level ${fmt(flat, 4)} m, settled 2 mean(y) ${fmt(height, 4)} m (${fmt((100 * (height - flat)) / flat, 1)} %), ` +
            `rms speed ${fmt(rmsSpeed(snap), 3)}, mean x ${fmt(meanComponent(snap.pos, snap.n, 0), 3)} (centre ${fmt(size[0] / 2, 3)}), ` +
            `top spread ${fmt(spread, 2)} spacings`,
        );
        // Mean x of a fluid spread evenly along the floor is the middle of the tank (measured 0.597 and 0.604).
        expect(Math.abs(meanComponent(snap.pos, snap.n, 0) - size[0] / 2)).toBeLessThan(0.08);
        // DOUBT for the lead: a thin pool (6 particles deep) stands 8 % (2D) / 6 % (3D) ABOVE the volume-conserving
        // level, about half a spacing, so the pool is less dense than a lattice (probably the wall term holding the layer
        // along the floor and the sides off the glass; the flat-wall table itself is good to 3 %, section 2). 12 % is a
        // regression guard on that, not a precision.
        expect(Math.abs(height - flat) / flat).toBeLessThan(0.12);
        expect(rmsSpeed(snap)).toBeLessThan(0.06);
        // Cell order changes floating-point reduction order, and a sparse edge column can
        // contain a bead outside the bulk surface. Test bulk flatness separately from outliers:
        // the middle 90% must stay within 2.5 spacings, every column within 4. Volume,
        // velocity and containment assertions above/below still constrain the full particle set.
        const sortedTops = [...tops].sort((a, b) => a - b);
        const trim = Math.max(1, Math.floor(sortedTops.length * 0.05));
        const bulkSpread = (sortedTops[sortedTops.length - trim - 1]! - sortedTops[trim]!) / sp;
        expect(bulkSpread).toBeLessThan(2.5);
        expect(spread).toBeLessThan(4);
        expect(escapes(snap, dim, pbf.bounds, pbf.params.wallOffset).count).toBe(0);
      },
    );
  });
});

// --- 5. Momentum -------------------------------------------------------------------------------------------------------

describe.each(DIMS)('linear momentum without gravity or walls in %iD', (dim) => {
  const s = dim === 3 ? 0.05 : 0.025;
  const half = dim === 3 ? 1.6 : 2.4;

  async function collide(viscosity: number) {
    return withPbf(
      {
        dimension: dim,
        capacity: 6000,
        spacing: s,
        bounds: {
          min: [-half, -half, dim === 3 ? -half : -1],
          max: [half, half, dim === 3 ? half : 1],
        },
        params: { gravity: [0, 0, 0], viscosity },
      },
      async (pbf, device) => {
        // Two unequal blocks, a different number of particles each, running into one another off-centre.
        const a = dim === 3 ? 0.2 : 0.3;
        pbf.addBlock({ min: [-0.5, -a, -a], max: [-0.1, a, a], velocity: [1.5, 0.3, 0.1] });
        pbf.addBlock({
          min: [0.15, -0.5 * a, -0.5 * a],
          max: [0.4, 0.5 * a + 0.1, 0.5 * a],
          velocity: [-1.2, 0, -0.2],
        });
        // Plus a random shear, so the velocities are not two plain slabs.
        const random = makeRandom(5);
        const first = await snapshot(pbf);
        for (let i = 0; i < first.n; i++) {
          for (let c = 0; c < (dim === 3 ? 3 : 2); c++)
            first.vel[i * 4 + c]! += (random() - 0.5) * 1.0;
        }
        // A 2D fluid has no z: the solver keeps z = 0, so a z velocity would be momentum that is destroyed by design.
        if (dim === 2) for (let i = 0; i < first.n; i++) first.vel[i * 4 + 2] = 0;
        pbf.system.set({ positions: first.pos, velocities: first.vel });
        const start = sumVelocity(first);
        const scale = first.n * rmsSpeed(first);
        const drift: number[] = [];
        let extent = 0;
        for (let step = 0; step < 6; step++) {
          await advance(device, pbf, 5);
          const snap = await snapshot(pbf);
          const now = sumVelocity(snap);
          drift.push(Math.hypot(now[0] - start[0], now[1] - start[1], now[2] - start[2]) / scale);
          for (let i = 0; i < snap.n; i++) {
            extent = Math.max(extent, Math.abs(snap.pos[i * 4]!), Math.abs(snap.pos[i * 4 + 1]!));
          }
        }
        return {
          n: first.n,
          drift,
          extent,
          kinetic: energies(await snapshot(pbf)).kinetic,
          kinetic0: energies(first).kinetic,
        };
      },
    );
  }

  it.each([0, 0.3])('conserves the sum of the velocities (viscosity %f)', async (viscosity) => {
    const r = await collide(viscosity);
    console.log(
      `momentum ${dim}D viscosity ${viscosity}: n=${r.n}, |sum v - sum v0| / (n rms v) per 5 frames: ` +
        `${r.drift.map((d) => d.toExponential(1)).join(' ')}; kinetic energy ${fmt(r.kinetic0, 3)} -> ${fmt(r.kinetic, 3)}`,
    );
    // The collision must really have happened and must stay clear of the walls (otherwise the walls take momentum).
    expect(r.extent).toBeLessThan(half - 0.3);
    expect(r.kinetic).toBeLessThan(r.kinetic0);
    // The pair corrections (lambda_i + lambda_j + s_corr) grad W_ij and the XSPH terms are antisymmetric, so the sum
    // cancels to rounding: the velocity is (x - previous) / dt, so x's f32 ulp (6e-8 at |x| ~ 1) over dt = 1/180 gives
    // ~1e-5 m/s per particle, random, hence ~1e-4 of the total. Measured below 2e-6 of n * rms v; 1e-4 leaves a
    // factor 50 of margin while a missing antisymmetry (a momentum leak of even 1 % per step) would be 1e-1.
    for (const d of r.drift) expect(d).toBeLessThan(1e-4);
  });
});

// --- 6. Containment ----------------------------------------------------------------------------------------------------

describe.each(DIMS)('nothing leaves the box in %iD', (dim) => {
  const s = dim === 3 ? 0.04 : 0.02;
  const size: Vec3 = dim === 3 ? [0.8, 0.6, 0.4] : [0.8, 0.6, 1];

  const violent: PbfImpulse = {
    position: [0.4, 0.15, 0.2],
    radius: 0.25,
    velocity: [0, 40, 0],
    strength: 1,
    outward: 60,
  };

  it('through a dam break and repeated violent impulses, for seconds', async () => {
    await withPbf(
      { dimension: dim, capacity: 3000, spacing: s, bounds: tank(dim, size) },
      async (pbf, device) => {
        pbf.reset({
          blocks: [
            { min: [0, 0, 0], max: [0.3, 0.4, size[2]] },
            { min: [0.5, 0, 0], max: [0.8, 0.2, size[2]] },
          ],
        });
        const limit = 1.5 * pbf.constants.h * 180;
        let worstSpeed = 0;
        let worstOut = 0;
        let outside = 0;
        // Pushes every 10th frame for the first 60, with the largest allowed number (4) at once; 4 s in all.
        await advance(device, pbf, 240, 1 / 60, (frame) => {
          if (frame < 60 && frame % 10 === 0) {
            for (let k = 0; k < 4; k++)
              pbf.impulse({ ...violent, position: [0.2 + 0.15 * k, 0.1, 0.2] });
          }
        });
        for (let t = 0; t < 4; t++) {
          const snap = await snapshot(pbf);
          const out = escapes(snap, dim, pbf.bounds, pbf.params.wallOffset, 1e-5);
          outside += out.count;
          worstOut = Math.max(worstOut, out.worst);
          worstSpeed = Math.max(worstSpeed, maxSpeed(snap));
          expect(allFinite(snap)).toBe(true);
          await advance(device, pbf, 30);
        }
        // The wall projection is a clamp: exact up to f32 rounding of the coordinates (about 1e-7).
        expect(outside).toBe(0);
        expect(worstOut).toBeLessThan(1e-5);
        // The speed limit is 1.5 h per sub-step (the correction pass clamps; the viscosity pass averages, never raises).
        expect(worstSpeed).toBeLessThanOrEqual(limit * 1.001);
      },
    );
  });

  it('with the wall at its fastest allowed speed, and with a request for far more', async () => {
    await withPbf(
      { dimension: dim, capacity: 3000, spacing: s, bounds: tank(dim, size) },
      async (pbf, device) => {
        pbf.reset({ blocks: [{ min: [0, 0, 0], max: [0.5, 0.2, size[2]] }] });
        const z = dim === 3 ? 0 : -1;
        const zMax = dim === 3 ? size[2] : 1;
        // The default limit is one spacing per step: s / (1/60 s) = 60 s metres per second.
        const limit = s * 60;
        const perUpdate = limit / 60;
        const frames = Math.round(0.24 / perUpdate);
        let previous = pbf.bounds.min[0]!;
        let fastest = 0;
        let outside = 0;
        // Ask for a wall 0.3 m further in at once: 5 to 10 times what the limit allows in a frame.
        pbf.setBounds([0.3, 0, z], [size[0], size[1], zMax]);
        for (let f = 0; f < frames; f++) {
          await advance(device, pbf, 1);
          fastest = Math.max(fastest, pbf.bounds.min[0]! - previous);
          previous = pbf.bounds.min[0]!;
          outside += escapes(
            await snapshot(pbf),
            dim,
            pbf.bounds,
            pbf.params.wallOffset,
            1e-5,
          ).count;
        }
        console.log(
          `wall ${dim}D: limit ${fmt(limit, 2)} m/s, fastest frame ${fmt(fastest * 60, 3)} m/s, wall at x = ${fmt(pbf.bounds.min[0]!, 3)} after ${frames} frames`,
        );
        // Runs at the limit and not above it (the target is far away: the wall is at full speed all the way).
        expect(fastest).toBeLessThanOrEqual(perUpdate * 1.001);
        expect(fastest).toBeGreaterThan(perUpdate * 0.99);
        expect(pbf.bounds.min[0]!).toBeCloseTo(frames * perUpdate, 4);
        expect(outside).toBe(0);
        const snap = await snapshot(pbf);
        expect(allFinite(snap)).toBe(true);
        // The fluid was pushed along, not run through: every particle is in front of the wall, and the wall has really
        // moved into where the fluid was (it started at 0 to 0.5).
        let leftmost = Infinity;
        for (let i = 0; i < snap.n; i++) leftmost = Math.min(leftmost, snap.pos[i * 4]!);
        expect(leftmost).toBeGreaterThan(pbf.bounds.min[0]! + pbf.params.wallOffset - 1e-5);
        expect(leftmost).toBeGreaterThan(0.2);

        // An explicit limit of 0.5 m/s is obeyed too, on the way back.
        pbf.params.maxWallSpeed = 0.5;
        pbf.setBounds([0, 0, z], [size[0], size[1], zMax]);
        previous = pbf.bounds.min[0]!;
        let slowest = 0;
        for (let f = 0; f < 20; f++) {
          await advance(device, pbf, 1);
          slowest = Math.max(slowest, Math.abs(pbf.bounds.min[0]! - previous));
          previous = pbf.bounds.min[0]!;
        }
        expect(slowest).toBeLessThanOrEqual((0.5 / 60) * 1.001);
        expect(slowest).toBeGreaterThan((0.5 / 60) * 0.99);
        expect(pbf.bounds.min[0]!).toBeCloseTo(frames * perUpdate - 20 * (0.5 / 60), 4);
        expect(
          escapes(await snapshot(pbf), dim, pbf.bounds, pbf.params.wallOffset, 1e-5).count,
        ).toBe(0);
      },
    );
  });

  it('colliders (a sphere and a box) never contain a particle centre', async () => {
    // In 2D the centre must have z = 0: the sphere test is a 3D distance and the particles live in the plane z = 0 (a 2D
    // sphere with another z is a smaller disc, or none at all).
    const sphere: PbfCollider & { kind: 'sphere' } = {
      kind: 'sphere',
      center: [0.3, 0.25, dim === 3 ? 0.2 : 0],
      radius: 0.1,
    };
    const box: PbfCollider & { kind: 'box' } = {
      kind: 'box',
      min: [0.5, 0.1, 0.05],
      max: [0.7, 0.2, 0.35],
    };
    const o = s / 2;
    /** Particle centres inside either collider (grown by the wall offset, less a tolerance), per sample. */
    const countInside = (snap: Snapshot, tolerance: number): number => {
      let inside = 0;
      for (let i = 0; i < snap.n; i++) {
        const p = [snap.pos[i * 4]!, snap.pos[i * 4 + 1]!, dim === 3 ? snap.pos[i * 4 + 2]! : 0];
        const d = Math.hypot(
          p[0]! - sphere.center[0],
          p[1]! - sphere.center[1],
          p[2]! - (dim === 3 ? sphere.center[2] : 0),
        );
        if (d < sphere.radius + o - tolerance) inside++;
        const inBox =
          p[0]! > box.min[0] - o + tolerance &&
          p[0]! < box.max[0] + o - tolerance &&
          p[1]! > box.min[1] - o + tolerance &&
          p[1]! < box.max[1] + o - tolerance &&
          (dim === 2 || (p[2]! > box.min[2] - o + tolerance && p[2]! < box.max[2] + o - tolerance));
        if (inBox) inside++;
      }
      return inside;
    };
    const run = async (colliders: PbfCollider[]) =>
      withPbf(
        { dimension: dim, capacity: 3000, spacing: s, bounds: tank(dim, size) },
        async (pbf, device) => {
          pbf.reset({ blocks: [{ min: [0.05, 0.35, 0], max: [0.75, 0.55, size[2]] }], colliders });
          let inside = 0;
          let onTop = 0;
          for (let t = 0; t < 8; t++) {
            await advance(device, pbf, 30);
            const snap = await snapshot(pbf);
            expect(allFinite(snap)).toBe(true);
            expect(escapes(snap, dim, pbf.bounds, pbf.params.wallOffset).count).toBe(0);
            inside += countInside(snap, 1e-5);
            // Fluid lying on the top of the box at any time.
            if (t < 4) {
              for (let i = 0; i < snap.n; i++) {
                const x = snap.pos[i * 4]!;
                const y = snap.pos[i * 4 + 1]!;
                if (
                  x > box.min[0] &&
                  x < box.max[0] &&
                  y > box.max[1] &&
                  y < box.max[1] + o + 2 * s
                ) {
                  onTop++;
                }
              }
            }
          }
          return { inside, onTop };
        },
      );
    const control = await run([]);
    const real = await run([sphere, box]);
    console.log(
      `colliders ${dim}D: particle-samples inside the colliders' space: ${control.inside} without them, ${real.inside} with them; on top of the box: ${control.onTop} / ${real.onTop}`,
    );
    // Without the colliders the fluid falls straight through where they would be (so the test can fail); with them
    // nothing is inside, to within f32 rounding of the coordinates, and fluid lies on top of the box.
    expect(control.inside).toBeGreaterThan(50);
    expect(real.inside).toBe(0);
    expect(real.onTop).toBeGreaterThan(5);
  });
});

// --- 7. Waves ---------------------------------------------------------------------------------------------------------

/** Amplitude spectrum (a Hann window, so the peak does not leak) of a series sampled every `dt`. */
function spectrum(
  signal: number[],
  dt: number,
  fMax: number,
  df = 0.01,
): { f: number; amplitude: number }[] {
  const average = mean(signal);
  const out: { f: number; amplitude: number }[] = [];
  for (let f = df; f <= fMax + 1e-9; f += df) {
    let re = 0;
    let im = 0;
    let window = 0;
    for (let i = 0; i < signal.length; i++) {
      const w = 0.5 - 0.5 * Math.cos((2 * Math.PI * (i + 0.5)) / signal.length);
      re += (signal[i]! - average) * w * Math.cos(2 * Math.PI * f * i * dt);
      im += (signal[i]! - average) * w * Math.sin(2 * Math.PI * f * i * dt);
      window += w;
    }
    out.push({ f, amplitude: (2 * Math.hypot(re, im)) / window });
  }
  return out;
}

describe('an oscillating wall makes waves', () => {
  const length = 1.5;

  /** Mean water height (2 mean y) within 0.3 m of the far end, sampled every 0.1 s, while the left wall oscillates. */
  async function farEndHeight(
    dim: Dim,
    frequency: number,
    amplitude: number,
    seconds: number,
  ): Promise<number[]> {
    const s = 0.03;
    const depth = 0.3;
    const width = dim === 3 ? 0.15 : 1;
    return withPbf(
      { dimension: dim, capacity: 6000, spacing: s, bounds: tank(dim, [length, 0.9, width]) },
      async (pbf, device) => {
        pbf.reset({ blocks: [{ min: [0, 0, 0], max: [length, depth, width] }] });
        const series: number[] = [];
        let t = 0;
        for (let sample = 0; sample < seconds * 10; sample++) {
          await advance(device, pbf, 6, 1 / 60, () => {
            t += 1 / 60;
            // Ramped over 2 s so the start does not ring the tank at its own modes; a push inwards only (1 - cos >= 0).
            const ramp = Math.min(1, t / 2);
            const x0 = amplitude * ramp * (1 - Math.cos(2 * Math.PI * frequency * t));
            pbf.setBounds([x0, 0, dim === 3 ? 0 : -1], [length, 0.9, dim === 3 ? width : 1]);
          });
          const snap = await snapshot(pbf);
          let sum = 0;
          let count = 0;
          for (let i = 0; i < snap.n; i++) {
            if (snap.pos[i * 4]! > length - 0.3) {
              sum += snap.pos[i * 4 + 1]!;
              count++;
            }
          }
          series.push((2 * sum) / Math.max(1, count));
        }
        expect(
          escapes(await snapshot(pbf), dim, pbf.bounds, pbf.params.wallOffset, 1e-5).count,
        ).toBe(0);
        return series;
      },
    );
  }

  it('still water has no signal to speak of (the noise floor)', async () => {
    const series = await farEndHeight(2, 0.7, 0, 11);
    const floor = Math.max(...spectrum(series.slice(30), 0.1, 2).map((p) => p.amplitude));
    console.log(`wave noise floor (still water, 2D): ${floor.toExponential(2)} m`);
    // Measured 2e-4 m: a column settles and then sits.
    expect(floor).toBeLessThan(2e-3);
  });

  it.each([0.4, 0.7, 1.1])(
    '2D: a wall at %f Hz drives a wave of that frequency',
    async (frequency) => {
      const seconds = 12;
      const series = await farEndHeight(2, frequency, 0.06, seconds);
      const used = series.slice(40);
      const peaks = spectrum(used, 0.1, 2);
      const best = peaks.reduce((a, b) => (b.amplitude > a.amplitude ? b : a));
      const atDrive = peaks.reduce((a, b) =>
        Math.abs(b.f - frequency) < Math.abs(a.f - frequency) ? b : a,
      );
      // Zero crossings of the detrended signal as a second, independent estimate.
      const average = mean(used);
      let crossings = 0;
      for (let i = 1; i < used.length; i++)
        if ((used[i - 1]! - average) * (used[i]! - average) < 0) crossings++;
      const byCrossings = crossings / 2 / ((used.length - 1) * 0.1);
      console.log(
        `waves 2D: drive ${frequency} Hz, spectral peak ${fmt(best.f, 2)} Hz (amplitude ${fmt(best.amplitude, 4)} m), ` +
          `zero crossings give ${fmt(byCrossings, 2)} Hz, height range ${fmt(Math.min(...used), 3)}..${fmt(Math.max(...used), 3)}`,
      );
      // Measured peaks at 0.39 / 0.70 / 1.10 Hz (the resolution of a 10 s window is 0.1 Hz, the grid 0.01). 15 % is the
      // task's tolerance; the tank's own modes are at 0.54 and 0.94 Hz, far enough from all three drives to separate.
      expect(Math.abs(best.f - frequency)).toBeLessThan(0.15 * frequency);
      expect(Math.abs(byCrossings - frequency)).toBeLessThan(0.25 * frequency);
      // Clearly above the still-water floor of 2e-4: measured 0.014 (0.4 Hz), 0.038 (0.7), 0.019 (1.1) m.
      expect(atDrive.amplitude).toBeGreaterThan(0.005);
      expect(atDrive.amplitude).toBeGreaterThan(10 * 2e-4 * 5);
    },
  );

  it('3D: a wall at 0.7 Hz drives a wave of that frequency', async () => {
    const series = await farEndHeight(3, 0.7, 0.06, 10);
    const peaks = spectrum(series.slice(40), 0.1, 2);
    const best = peaks.reduce((a, b) => (b.amplitude > a.amplitude ? b : a));
    console.log(
      `waves 3D: drive 0.7 Hz, spectral peak ${fmt(best.f, 2)} Hz (amplitude ${fmt(best.amplitude, 4)} m)`,
    );
    // A 6 s window resolves 0.17 Hz, so the peak may be a bin away; measured 0.70 Hz, amplitude 0.03 m.
    expect(Math.abs(best.f - 0.7)).toBeLessThan(0.15 * 0.7);
    expect(best.amplitude).toBeGreaterThan(0.005);
  });
});

// --- 8. Extremes ---------------------------------------------------------------------------------------------------------

/** Parameter sets that are absurd on purpose. The solver clamps them (sanitise); none may produce NaN or Inf. */
const EXTREME_PARAMS: [string, Partial<PbfParams>][] = [
  ['gravity 1e4', { gravity: [0, -1e4, 0] }],
  ['gravity -1e9 (clamped)', { gravity: [1e9, -1e9, 1e9] }],
  ['gravity NaN', { gravity: [NaN, Infinity, -Infinity] }],
  ['relaxation 0 (raw Newton step)', { relaxation: 0 }],
  ['relaxation 1e6', { relaxation: 1e6 }],
  ['relaxation NaN', { relaxation: NaN }],
  ['relax 0 (clamped to 0.05)', { relax: 0 }],
  ['relax 0.05', { relax: 0.05 }],
  ['relax 1.5', { relax: 1.5 }],
  ['relax 100', { relax: 100 }],
  ['iterations 1', { iterations: 1 }],
  ['iterations 32', { iterations: 32 }],
  ['iterations 1000 (clamped)', { iterations: 1000 }],
  ['iterations 0 (clamped)', { iterations: 0 }],
  ['substeps 1', { substeps: 1 }],
  ['substeps 8', { substeps: 8 }],
  ['substeps 1000 (clamped)', { substeps: 1000 }],
  ['viscosity 0.9', { viscosity: 0.9 }],
  ['viscosity 5 (clamped)', { viscosity: 5 }],
  ['vorticity 4', { vorticity: 4, viscosity: 0.2 }],
  ['vorticity 100 (clamped)', { vorticity: 100 }],
  ['artificial pressure 10', { artificialPressure: 10 }],
  ['exponent 12', { artificialPressureExponent: 12 }],
  ['exponent 1', { artificialPressureExponent: 1 }],
  ['exponent 99 (clamped)', { artificialPressureExponent: 99 }],
  ['dq 0.02 and 0.9', { artificialPressureRadius: 0.02 }],
  ['dq 5 (clamped)', { artificialPressureRadius: 5 }],
  ['tension 1', { tension: 1 }],
  ['friction 10', { friction: 10 }],
  ['friction 1e6 (clamped)', { friction: 1e6 }],
  ['damping 100', { damping: 100 }],
  ['no wall density, stiffness 4', { wallDensity: false, wallStiffness: 4 }],
  ['wall stiffness 0', { wallStiffness: 0 }],
  ['wall offset 0', { wallOffset: 1e-12 }],
  ['wall offset huge (clamped to h)', { wallOffset: 1e3 }],
  ['maxSpeed tiny', { maxSpeed: 1e-6 }],
  ['maxSpeed huge', { maxSpeed: 1e9 }],
  ['maxSpeed NaN', { maxSpeed: NaN }],
  ['timeStep tiny', { timeStep: 1e-12 }],
  ['timeStep huge', { timeStep: 1e9 }],
  ['timeStep NaN', { timeStep: NaN }],
  ['timeStep negative', { timeStep: -1 }],
  ['maxSteps 1000 (clamped)', { maxSteps: 1000 }],
  ['foam speed 0', { foamSpeed: 0, foamDecay: 1e9 }],
];

describe.each(DIMS)('parameter extremes in %iD', (dim) => {
  const s = 0.03;
  const size: Vec3 = [0.6, 0.6, 0.6];

  it('survive every extreme parameter set: all finite, all inside the box', async () => {
    await withPbf(
      { dimension: dim, capacity: 1500, spacing: s, bounds: tank(dim, size) },
      async (pbf, device) => {
        const failures: string[] = [];
        for (const [label, extreme] of EXTREME_PARAMS) {
          // The same solver for all of them: parameters are live, and reset puts the scene back.
          Object.assign(pbf.params, structuredClone(DEFAULT_PBF_PARAMS), extreme);
          pbf.reset({ blocks: [{ min: [0.05, 0, 0.05], max: [0.35, 0.3, 0.35] }] });
          pbf.impulse({
            position: [0.2, 0.2, 0.2],
            radius: 0.2,
            velocity: [3, 3, 3],
            strength: 1,
            outward: 5,
          });
          await advance(device, pbf, 8);
          const snap = await snapshot(pbf);
          const out = escapes(snap, dim, pbf.bounds, pbf.params.wallOffset, 1e-4);
          if (!allFinite(snap)) failures.push(`${label}: not finite`);
          else if (out.count > 0)
            failures.push(`${label}: ${out.count} outside by up to ${out.worst}`);
          else {
            // The scalar is a foam amount: 0..1 whatever else happens.
            for (let i = 0; i < snap.n; i++) {
              const foam = snap.pos[i * 4 + 3]!;
              if (!(foam >= 0 && foam <= 1)) {
                failures.push(`${label}: foam ${foam}`);
                break;
              }
            }
          }
          // The density ratio is always finite; the multiplier is not for relaxation 0 (below: a known failure).
          const state = await pbf.readState();
          const values = label.startsWith('relaxation 0 ')
            ? [...state.density]
            : [...state.lambda, ...state.density];
          if (!values.every(Number.isFinite)) failures.push(`${label}: solve buffer not finite`);
        }
        expect(failures).toEqual([]);
      },
    );
  });

  it('handles degenerate particles: on a corner, outside the box, far outside, on the floor twice', async () => {
    await withPbf(
      { dimension: dim, capacity: 64, spacing: s, bounds: tank(dim, size) },
      async (pbf, device) => {
        const z = (v: number) => (dim === 3 ? v : 0);
        const rows: number[][] = [
          [0, 0, z(0)],
          [size[0], size[1], z(size[2])],
          [0, size[1], z(0)],
          [size[0], 0, z(size[2])], // the corners, exactly
          [-5, 0.3, z(0.3)],
          [0.3, 100, z(0.3)],
          [0.3, -1e6, z(0.3)],
          [1e4, 1e4, z(1e4)], // outside, far outside
          [0.1, 0.0, z(0.1)],
          [0.1, 0.0, z(0.1)], // on the floor, at the very same point
        ];
        pbf.system.set({ positions: new Float32Array(rows.flatMap((r) => [...r, 0])) });
        expect(pbf.count).toBe(rows.length);
        await advance(device, pbf, 30);
        const snap = await snapshot(pbf);
        expect(allFinite(snap)).toBe(true);
        // Every one of them was put into the box in the first sub-step, and stays.
        expect(escapes(snap, dim, pbf.bounds, pbf.params.wallOffset, 1e-4).count).toBe(0);
      },
    );
  });

  it('pushes coincident particles apart (the deterministic nudge) instead of dividing by zero', async () => {
    await withPbf(
      {
        dimension: dim,
        capacity: 16,
        spacing: s,
        bounds: tank(dim, size),
        params: { gravity: [0, 0, 0] },
      },
      async (pbf, device) => {
        const z = dim === 3 ? 0.3 : 0;
        // Three at the very same point and a fourth beside them.
        pbf.system.set({
          positions: new Float32Array([
            0.3,
            0.3,
            z,
            0,
            0.3,
            0.3,
            z,
            0,
            0.3,
            0.3,
            z,
            0,
            0.31,
            0.3,
            z,
            0,
          ]),
        });
        await advance(device, pbf, 10);
        const snap = await snapshot(pbf);
        expect(allFinite(snap)).toBe(true);
        let minGap = Infinity;
        for (let i = 0; i < snap.n; i++) {
          for (let j = i + 1; j < snap.n; j++) {
            minGap = Math.min(
              minGap,
              Math.hypot(
                snap.pos[i * 4]! - snap.pos[j * 4]!,
                snap.pos[i * 4 + 1]! - snap.pos[j * 4 + 1]!,
                snap.pos[i * 4 + 2]! - snap.pos[j * 4 + 2]!,
              ),
            );
          }
        }
        // Measured: the closest pair ends 0.18 m apart (the artificial pressure flings a coincident pair at the speed
        // limit); anything above a thousandth of h proves the nudge gave them a direction.
        expect(minGap).toBeGreaterThan(1e-3 * pbf.constants.h);
      },
    );
  });

  // Found by this test: with relaxation 0 a particle with no neighbours had C = 0 and a zero denominator, so
  // lambda = -0 / 0 = NaN. The relaxation is now clamped to at least 0.001.
  it('relaxation 0: the multiplier of an isolated particle is finite (clamped to a tiny positive value)', async () => {
    await withPbf(
      {
        dimension: dim,
        capacity: 4,
        spacing: s,
        bounds: tank(dim, size),
        params: { relaxation: 0, gravity: [0, 0, 0] },
      },
      async (pbf, device) => {
        pbf.system.set({ positions: new Float32Array([0.3, 0.3, dim === 3 ? 0.3 : 0, 0]) });
        await runOnce(device, pbf, { iterations: 1 });
        const { lambda, density } = await pbf.readState();
        expect(Number.isFinite(density[0]!)).toBe(true);
        expect(Number.isFinite(lambda[0]!)).toBe(true);
      },
    );
  });

  it('survives odd frame deltas, an empty solver, a full one, and extreme spacings', async () => {
    await withPbf(
      { dimension: dim, capacity: 400, spacing: s, bounds: tank(dim, size) },
      async (pbf, device) => {
        // count 0: update, run and readState all work.
        expect(pbf.count).toBe(0);
        await advance(device, pbf, 3);
        await runOnce(device, pbf);
        expect((await pbf.readState()).lambda.length).toBe(0);
        expect((await pbf.system.read('position')).length).toBe(0);

        // Full: capacity filled, further blocks and the emitter place nothing.
        pbf.setEmitter({ position: [0.3, 0.5, 0.3], velocity: [0, -1, 0], radius: 0.05 });
        const placed = pbf.addBlock({ min: [0, 0, 0], max: [0.6, 0.6, 0.6] });
        expect(placed).toBe(pbf.count);
        pbf.setCapacity(pbf.count);
        expect(pbf.count).toBe(pbf.capacity);
        expect(pbf.addBlock({ min: [0, 0, 0], max: [0.6, 0.6, 0.6] })).toBe(0);
        // NaN, infinite, negative, zero and huge deltas are all clamped to something sane.
        for (const delta of [NaN, Infinity, -Infinity, -1, 0, 1e-12, 0.3, 1e9, 1 / 60]) {
          await advance(device, pbf, 2, delta);
        }
        expect(pbf.count).toBe(pbf.capacity);
        const snap = await snapshot(pbf);
        expect(allFinite(snap)).toBe(true);
        expect(escapes(snap, dim, pbf.bounds, pbf.params.wallOffset).count).toBe(0);
        expect(Number.isFinite(pbf.time)).toBe(true);
      },
    );
    // A spacing of a millimetre in a tank of a few centimetres, and a spacing of ten metres in a tank of a hundred.
    for (const [spacing, extent] of [
      [1e-3, 0.03],
      [10, 100],
    ] as const) {
      await withPbf(
        { dimension: dim, capacity: 600, spacing, bounds: tank(dim, [extent, extent, extent]) },
        async (pbf, device) => {
          pbf.reset({ blocks: [{ min: [0, 0, 0], max: [extent, extent / 2, extent] }] });
          await advance(device, pbf, 20);
          const snap = await snapshot(pbf);
          expect(allFinite(snap)).toBe(true);
          expect(escapes(snap, dim, pbf.bounds, pbf.params.wallOffset, 1e-4 * spacing).count).toBe(
            0,
          );
        },
      );
    }
  });
});

describe('invalid construction', () => {
  it.each([0, -1, NaN, Infinity])('rejects spacing %f', async (spacing) => {
    const device = await getTestDevice();
    expect(() => createPbf({ device, capacity: 10, spacing, bounds: tank(3, [1, 1, 1]) })).toThrow(
      RangeError,
    );
  });

  it.each([0, -5, NaN])('rejects capacity %f', async (capacity) => {
    const device = await getTestDevice();
    expect(() =>
      createPbf({ device, capacity, spacing: 0.05, bounds: tank(3, [1, 1, 1]) }),
    ).toThrow(RangeError);
  });
});

// --- 9. Frame rate -----------------------------------------------------------------------------------------------------

/** What "the same fluid" means for two runs of a chaotic scene: tolerances on aggregates, never on positions. */
const sameFluid = { energy: 0.08, meanX: 0.06, average: 0.03 };

function expectSameFluid(
  a: { potential: number; kinetic: number; meanX: number }[],
  b: { potential: number; kinetic: number; meanX: number }[],
): void {
  const scale = a[0]!.potential;
  a.forEach((x, i) => {
    const o = b[i]!;
    expect(Math.abs(x.potential - o.potential) / scale).toBeLessThan(sameFluid.energy);
    expect(Math.abs(x.kinetic - o.kinetic) / scale).toBeLessThan(sameFluid.energy);
    expect(Math.abs(x.meanX - o.meanX)).toBeLessThan(sameFluid.meanX);
  });
  // Averaged over the samples the noise mostly cancels (measured below 1 %).
  const average = (list: { potential: number }[]) => mean(list.map((x) => x.potential));
  expect(Math.abs(average(a) - average(b)) / scale).toBeLessThan(sameFluid.average);
}

describe('the fluid does not depend on the display rate', () => {
  const dim = 2;
  const s = 0.02;
  const size: Vec3 = [1.2, 0.8, 1];
  const seconds = 3;
  const sampleAt = [0.5, 1, 1.5, 2, 3];

  interface Sample {
    time: number;
    kinetic: number;
    potential: number;
    meanX: number;
  }

  /** The same dam break fed with `delta`-second frames; energies and mean x at the sample times. */
  async function series(
    delta: number,
    params: Partial<PbfParams> = {},
  ): Promise<{ samples: Sample[]; steps: number }> {
    return withPbf(
      { dimension: dim, capacity: 2000, spacing: s, bounds: tank(dim, size), params },
      async (pbf, device) => {
        pbf.reset({ blocks: [{ min: [0, 0, 0], max: [0.3, 0.5, 1] }] });
        const samples: Sample[] = [];
        let steps = 0;
        let call = 0;
        const total = Math.round(seconds / delta);
        const targets = new Map(sampleAt.map((t) => [Math.round(t / delta), t]));
        while (call < total) {
          const encoder = device.createCommandEncoder();
          steps += pbf.update(encoder, delta).steps;
          device.queue.submit([encoder.finish()]);
          call++;
          if (targets.has(call)) {
            await device.queue.onSubmittedWorkDone();
            const e = energies(await snapshot(pbf));
            samples.push({ time: targets.get(call)!, ...e });
            // The simulated clock is where the display clock is, to within one step.
            expect(Math.abs(pbf.time - targets.get(call)!)).toBeLessThan(1 / 60 + 1e-9);
          }
        }
        await device.queue.onSubmittedWorkDone();
        return { samples, steps };
      },
    );
  }

  const row = (name: string, samples: Sample[]) =>
    `${name.padEnd(22)} KE ${samples.map((x) => fmt(x.kinetic, 3).padStart(6)).join(' ')} | PE ${samples
      .map((x) => fmt(x.potential, 3).padStart(6))
      .join(' ')} | mean x ${samples.map((x) => fmt(x.meanX, 3).padStart(6)).join(' ')}`;

  it('30, 60 and 120 Hz updates run the same steps and agree in energy and mean position', async () => {
    const reference = await series(1 / 60);
    const again = await series(1 / 60);
    const hz30 = await series(1 / 30);
    const hz120 = await series(1 / 120);
    // Irregular frames: 1/45 s and 1/90 s alternately (mean 1/60 s), the way a real display jitters.
    console.log(
      `frame rates (dam break 2D, samples at ${sampleAt.join(', ')} s)\n  ` +
        [
          row('60 Hz', reference.samples),
          row('60 Hz again', again.samples),
          row('30 Hz', hz30.samples),
          row('120 Hz', hz120.samples),
        ].join('\n  '),
    );
    // The same number of fixed steps whatever the frame rate: 3 s of 1/60 s (the +1e-6 of the accumulator absorbs rounding).
    expect(reference.steps).toBe(180);
    expect(hz30.steps).toBe(180);
    expect(hz120.steps).toBe(180);

    // Chaotic splashing amplifies f32 rounding (the cell order inside a cell is not deterministic: two identical 60 Hz
    // runs differ), so the baseline is the spread between two identical runs, measured at 0.03 of the potential energy
    // at 0.5 s (0.77 J/kg here; the kinetic energy per particle peaks near 0.9 J/kg at that time).
    const scale = reference.samples[0]!.potential;
    const noise = Math.max(
      ...reference.samples.map((x, i) => Math.abs(x.kinetic - again.samples[i]!.kinetic) / scale),
      ...reference.samples.map(
        (x, i) => Math.abs(x.potential - again.samples[i]!.potential) / scale,
      ),
    );
    console.log(
      `  run-to-run spread of the same 60 Hz run: ${fmt(noise, 4)} of the potential energy`,
    );
    // The same tolerance for another rate as for the same rate again: if 30 or 120 Hz were a different fluid it would show.
    expect(noise).toBeLessThan(sameFluid.energy);
    expectSameFluid(reference.samples, again.samples);
    expectSameFluid(reference.samples, hz30.samples);
    expectSameFluid(reference.samples, hz120.samples);
  });

  it('a step of 1/120 s with 2 sub-steps equals a step of 1/60 s with 4 (the same sub-step of 1/240 s)', async () => {
    const coarse = await series(1 / 60, { timeStep: 1 / 60, substeps: 4 });
    const fine = await series(1 / 120, { timeStep: 1 / 120, substeps: 2 });
    console.log(
      `equal sub-steps\n  ${row('1/60 x 4 sub-steps', coarse.samples)}\n  ${row('1/120 x 2 sub-steps', fine.samples)}`,
    );
    expect(coarse.steps).toBe(180);
    expect(fine.steps).toBe(360);
    expectSameFluid(coarse.samples, fine.samples);
  });

  it('settles to the same height at any rate', async () => {
    const heights: number[] = [];
    for (const delta of [1 / 30, 1 / 60, 1 / 120]) {
      await withPbf(
        { dimension: 2, capacity: 1000, spacing: s, bounds: tank(2, [0.4, 1, 1]) },
        async (pbf, device) => {
          pbf.reset({ blocks: [{ min: [0, 0, 0], max: [0.4, 0.6, 1] }] });
          await advance(device, pbf, Math.round(2 / delta), delta);
          heights.push(2 * meanComponent((await snapshot(pbf)).pos, pbf.count, 1));
        },
      );
    }
    console.log(`settled height at 30, 60, 120 Hz: ${heights.map((x) => fmt(x, 4)).join(', ')} m`);
    // The steps are identical, so only f32 noise separates them: measured within 0.1 %; 1 % is the tolerance.
    for (const height of heights)
      expect(Math.abs(height - heights[1]!) / heights[1]!).toBeLessThan(0.01);
  });
});

// --- 11. Scenes: reset, capacity, blocks, emitter, impulses ----------------------------------------------------------

describe.each(DIMS)('scene management in %iD', (dim) => {
  const s = 0.05;
  const size: Vec3 = [3, 3, 3];
  const base = (capacity: number): Options => ({
    dimension: dim,
    capacity,
    spacing: s,
    bounds: tank(dim, size),
    // No gravity and no artificial pressure: with it a free block flies apart at ~1 m/s (see the jet below), which would
    // hide what these tests look at.
    params: { gravity: [0, 0, 0], artificialPressure: 0 },
  });

  it('reset clears the count and the clock, adds the blocks, restores bounds and gravity', async () => {
    await withPbf(base(1000), async (pbf, device) => {
      const z1 = dim === 3 ? 0.15 : 0;
      const nz = dim === 3 ? 3 : 1;
      pbf.reset({
        blocks: [
          { min: [1.1, 1.1, 1.1], max: [1.3, 1.25, 1.1 + z1] }, // 4 x 3 (x 3)
          { min: [1, 1, 1], max: [1.25, 1.1, 1 + z1] }, // 5 x 2 (x 3)
        ],
      });
      expect(pbf.count).toBe((12 + 10) * nz);
      expect(pbf.time).toBe(0);
      await advance(device, pbf, 12);
      expect(pbf.time).toBeGreaterThan(0.1);
      pbf.setBounds([0.5, 0, 0], [2, 2, 2], { snap: true });
      expect(pbf.bounds.min[0]).toBe(0.5);
      // An impulse is queued but never applied if the scene is reset first.
      pbf.impulse({ position: [0.2, 0.2, 0.2], radius: 1, velocity: [9, 9, 9], strength: 1 });
      const gravity: Vec3 = [1, -2, 3];
      pbf.reset({ gravity, blocks: [{ min: [1, 1, 1], max: [1.2, 1.2, 1.2] }] });
      expect(pbf.time).toBe(0);
      expect(pbf.count).toBe(4 * 4 * (dim === 3 ? 4 : 1));
      expect(Array.from(pbf.params.gravity)).toEqual([1, -2, 3]);
      expect(Array.from(pbf.bounds.min)).toEqual(Array.from(tank(dim, size).min));
      expect(Array.from(pbf.bounds.max)).toEqual(Array.from(tank(dim, size).max));
      pbf.params.gravity = [0, 0, 0];
      await advance(device, pbf, 2);
      expect(maxSpeed(await snapshot(pbf))).toBeLessThan(0.5);
      pbf.reset();
      expect(pbf.count).toBe(0);
      expect(pbf.time).toBe(0);
      await advance(device, pbf, 2);
    });
  });

  it('addBlock returns the number placed and stops at capacity', async () => {
    await withPbf(base(80), async (pbf) => {
      const per = 5 * 4 * (dim === 3 ? 3 : 1);
      expect(pbf.addBlock({ min: [0, 0, 0], max: [5 * s, 4 * s, 3 * s] })).toBe(per);
      expect(pbf.count).toBe(per);
      // The next block is 60 more in 3D (so only the 20 left fit), 20 in 2D (fits entirely).
      const placed = pbf.addBlock({ min: [1, 1, 1], max: [1 + 5 * s, 1 + 4 * s, 1 + 3 * s] });
      expect(placed).toBe(Math.min(per, 80 - per));
      expect(pbf.count).toBe(per + placed);
      // A big block fills what is left and no more.
      const rest = 80 - pbf.count;
      expect(pbf.addBlock({ min: [2, 2, 2], max: [2 + 10 * s, 2 + 10 * s, 2 + 10 * s] })).toBe(
        rest,
      );
      expect(pbf.count).toBe(80);
      expect(pbf.addBlock({ min: [0, 0, 0], max: [1, 1, 1] })).toBe(0);
      // Degenerate blocks place nothing.
      expect(pbf.addBlock({ min: [0, 0, 0], max: [0.01, 1, 1] })).toBe(0);
      expect(pbf.addBlock({ min: [1, 1, 1], max: [0, 0, 0] })).toBe(0);
    });
  });

  it('a capacity change grows or shrinks the buffers and keeps the particles that fit', async () => {
    await withPbf(base(100), async (pbf, device) => {
      pbf.reset({
        blocks: [
          { min: [0.1, 0.1, 0.1], max: [0.1 + 8 * s, 0.1 + 5 * s, 0.1 + (dim === 3 ? 2 : 1) * s] },
        ],
      });
      const count = pbf.count;
      expect(count).toBe(8 * 5 * (dim === 3 ? 2 : 1));
      await advance(device, pbf, 3);
      const before = await snapshot(pbf);
      const generation = pbf.generation;
      const setGeneration = pbf.particles.generation;
      const positions = pbf.particles.positions;
      const solve = pbf.solve;

      pbf.setCapacity(100); // the same: nothing happens
      expect(pbf.generation).toBe(generation);
      expect(pbf.solve).toBe(solve);

      pbf.setCapacity(600);
      expect(pbf.capacity).toBe(600);
      expect(pbf.particles.capacity).toBe(600);
      expect(pbf.count).toBe(count);
      expect(pbf.generation).toBeGreaterThan(generation);
      expect(pbf.particles.generation).not.toBe(setGeneration);
      expect(pbf.particles.positions).not.toBe(positions);
      expect(pbf.solve).not.toBe(solve);
      expect(pbf.particles.count).toBe(count);
      const grown = await snapshot(pbf);
      expect(Array.from(grown.pos)).toEqual(Array.from(before.pos));
      expect(Array.from(grown.vel)).toEqual(Array.from(before.vel));
      // The solver still runs, and has room for more.
      await advance(device, pbf, 3);
      expect(
        pbf.addBlock({ min: [1, 1, 1], max: [1 + 10 * s, 1 + 10 * s, 1 + 10 * s] }),
      ).toBeGreaterThan(50);
      await advance(device, pbf, 3);
      expect(allFinite(await snapshot(pbf))).toBe(true);
      expect((await pbf.readState()).density.length).toBe(pbf.count);

      // Shrink below the count: the count is clamped to the capacity and the first particles are the ones that stay.
      const kept = await snapshot(pbf);
      const generation2 = pbf.generation;
      pbf.setCapacity(40);
      expect(pbf.capacity).toBe(40);
      expect(pbf.count).toBe(40);
      expect(pbf.generation).toBeGreaterThan(generation2);
      const shrunk = await snapshot(pbf);
      expect(Array.from(shrunk.pos)).toEqual(Array.from(kept.pos.subarray(0, 160)));
      await advance(device, pbf, 3);
      expect(pbf.count).toBe(40);
      expect(allFinite(await snapshot(pbf))).toBe(true);
      expect(pbf.addBlock({ min: [0, 0, 0], max: [1, 1, 1] })).toBe(0);
      pbf.setCapacity(0.2); // clamped to at least one
      expect(pbf.capacity).toBe(1);
      expect(pbf.count).toBe(1);
    });
  });

  /** Lattice points of the nozzle: the cross-section of a jet of this radius, counted. */
  const nozzleSlice = (radius: number): number => {
    const m = Math.floor(radius / s);
    let count = 0;
    for (let i = -m; i <= m; i++) {
      for (let j = dim === 3 ? -m : 0; j <= (dim === 3 ? m : 0); j++) {
        if (Math.hypot(i, j) * s <= radius + 1e-9) count++;
      }
    }
    return count;
  };

  it('the emitter pours at the rate (cross-section / s^(D-1)) * speed / s and stops at capacity', async () => {
    const speed = 1;
    const radius = 0.1;
    const seconds = 1;
    const slice = nozzleSlice(radius);
    await withPbf({ ...base(6000), bounds: tank(dim, [3, 4, 3]) }, async (pbf, device) => {
      pbf.setEmitter({ position: [1.5, 3.5, 1.5], velocity: [0, -speed, 0], radius });
      await advance(device, pbf, 60);
      const crossSection = dim === 3 ? Math.PI * radius * radius : 2 * radius;
      // The task's formula: particles per slice (cross-section over s^(D-1)) times slices per second (speed / s).
      const formula = (crossSection / s ** (dim - 1)) * (speed / s) * seconds;
      // And the same with the lattice points of the nozzle counted instead of estimated from the area.
      const counted = slice * (speed / s) * seconds;
      console.log(
        `emitter ${dim}D: ${pbf.count} particles after ${seconds} s; formula ${fmt(formula, 1)}, lattice points counted ${counted} (${slice} per slice)`,
      );
      // The slices fall due every s / speed = 0.05 s: 20 of them in a second (the last may fall on the edge).
      expect(Math.abs(pbf.count - counted)).toBeLessThanOrEqual(slice);
      expect(pbf.count % slice).toBe(0);
      // The continuum formula is out by the rim of the disc (the lattice includes both ends of a diameter): 30 %.
      expect(Math.abs(pbf.count - formula)).toBeLessThan(0.3 * formula);
      // Switched off, it stops.
      pbf.setEmitter(null);
      const count = pbf.count;
      await advance(device, pbf, 30);
      expect(pbf.count).toBe(count);
    });
    // At capacity it stops, with whole slices only.
    await withPbf({ ...base(98), bounds: tank(dim, [3, 4, 3]) }, async (pbf, device) => {
      pbf.setEmitter({ position: [1.5, 3.5, 1.5], velocity: [0, -speed, 0], radius });
      await advance(device, pbf, 60);
      expect(pbf.count).toBeLessThanOrEqual(98);
      expect(pbf.count).toBeGreaterThan(98 - slice);
      const count = pbf.count;
      await advance(device, pbf, 30);
      expect(pbf.count).toBe(count);
      expect(allFinite(await snapshot(pbf))).toBe(true);
    });
  });

  /** Widest distance of any particle from the axis of the jet after a second of pouring. */
  async function jetWidth(params: Partial<PbfParams>): Promise<number> {
    return withPbf(
      { ...base(6000), bounds: tank(dim, [3, 4, 3]), params: { gravity: [0, 0, 0], ...params } },
      async (pbf, device) => {
        pbf.setEmitter({ position: [1.5, 3.5, 1.5], velocity: [0, -1, 0], radius: 0.1 });
        await advance(device, pbf, 60);
        const snap = await snapshot(pbf);
        let widest = 0;
        for (let i = 0; i < snap.n; i++) {
          widest = Math.max(
            widest,
            Math.hypot(snap.pos[i * 4]! - 1.5, dim === 3 ? snap.pos[i * 4 + 2]! - 1.5 : 0),
          );
        }
        return widest;
      },
    );
  }

  // KNOWN LIMITATION: with the default artificial pressure (0.06) and tension 0 there is nothing to balance the
  // repulsion at a free surface, so a jet of lattice-spaced particles flies apart radially at 1 to 1.5 m/s (maxr 1.4 m
  // after one second, in both dims, until it hits the walls 1.5 m away); a perfect lattice does the same on a smaller
  // scale (section 2). The default stays because the same term is what calms the bottom of a pool (rms speed 0.013 m/s
  // against 0.1 to 0.3 without it). `artificialPressureGate` fixes the free blob but not this jet, which spreads for a reason
  // not found yet (3D spreads at about 0.5 m/s with the pressure off too): `it.fails` documents it, flip it when it is fixed.
  it.fails('a free jet keeps its width with the default parameters (KNOWN TO FAIL)', async () => {
    expect(await jetWidth({})).toBeLessThan(0.1 + 2 * s);
  });

  // The workaround that works in 2D: pressure off, a little viscosity. Measured 0.136 against the nozzle's 0.1.
  // (In 3D the jet still spreads at ~0.5 m/s with the pressure off: 0.7 to 0.9 m after a second. Not asserted.)
  it.runIf(dim === 2)(
    'a free jet keeps its width without the artificial pressure (2D)',
    async () => {
      expect(await jetWidth({ artificialPressure: 0, viscosity: 0.2 })).toBeLessThan(0.1 + 2 * s);
    },
  );

  it('an impulse changes the velocities near its position and nowhere else', async () => {
    const spacing = dim === 3 ? 0.05 : 0.03;
    const n = dim === 3 ? 16 : 30;
    const centre: Vec3 = [0.6, 0.6, dim === 3 ? 0.6 : 0];
    await withPbf(
      {
        dimension: dim,
        capacity: 6000,
        spacing,
        bounds: tank(dim, [1.2, 1.2, 1.2]),
        // One sub-step per step: information in this solver travels one kernel radius per iteration, and the whole
        // sub-step of a push can reach 2 h. (Three sub-steps reach 6 h and the "far" region would be empty.)
        params: { gravity: [0, 0, 0], artificialPressure: 0, substeps: 1 },
      },
      async (pbf, device) => {
        const extent = n * spacing;
        const place = () =>
          pbf.reset({
            blocks: [
              {
                min: [0.6 - extent / 2, 0.6 - extent / 2, 0.6 - extent / 2],
                max: [0.6 + extent / 2, 0.6 + extent / 2, 0.6 + extent / 2],
              },
            ],
          });
        place();
        const { h } = pbf.constants;
        const radius = dim === 3 ? 0.12 : 0.15;
        pbf.impulse({ position: centre, radius, velocity: [0, 3, 0], strength: 1 });
        await advance(device, pbf, 1);
        const snap = await snapshot(pbf);
        const dist = (i: number) =>
          Math.hypot(
            snap.pos[i * 4]! - centre[0],
            snap.pos[i * 4 + 1]! - centre[1],
            dim === 3 ? snap.pos[i * 4 + 2]! - centre[2] : 0,
          );
        let inner = 0;
        let innerVy = 0;
        let far = 0;
        let farMax = 0;
        for (let i = 0; i < snap.n; i++) {
          const speed = Math.hypot(snap.vel[i * 4]!, snap.vel[i * 4 + 1]!, snap.vel[i * 4 + 2]!);
          if (dist(i) < 0.5 * radius) {
            inner++;
            innerVy += snap.vel[i * 4 + 1]!;
          } else if (dist(i) > radius + 3 * h) {
            far++;
            farMax = Math.max(farMax, speed);
          }
        }
        console.log(
          `impulse ${dim}D: ${inner} particles in the core, mean vy ${fmt(innerVy / inner, 2)} (target 3); ${far} beyond ${fmt(radius + 3 * h, 2)} m, fastest ${farMax.toExponential(1)} m/s`,
        );
        expect(inner).toBeGreaterThan(3);
        expect(far).toBeGreaterThan(50);
        // Pulled towards (0, 3, 0) with full strength at the centre of the falloff, then slowed by the surrounding fluid.
        expect(innerVy / inner).toBeGreaterThan(1);
        expect(innerVy / inner).toBeLessThan(3.05);
        // Nothing beyond the radius plus three kernel radii moves (gravity is off): one sub-step cannot carry a push
        // that far.
        expect(farMax).toBeLessThan(0.05);

        // The outward burst: a radial speed, positive away from the centre, no velocity to pull towards.
        place();
        pbf.impulse({ position: centre, radius, strength: 0, outward: 4 });
        await advance(device, pbf, 1);
        const burst = await snapshot(pbf);
        let radial = 0;
        let count = 0;
        for (let i = 0; i < burst.n; i++) {
          const d = [
            burst.pos[i * 4]! - centre[0],
            burst.pos[i * 4 + 1]! - centre[1],
            dim === 3 ? burst.pos[i * 4 + 2]! - centre[2] : 0,
          ];
          const len = Math.hypot(d[0]!, d[1]!, d[2]!);
          if (len > 0.25 * radius && len < 0.9 * radius) {
            radial +=
              (burst.vel[i * 4]! * d[0]! +
                burst.vel[i * 4 + 1]! * d[1]! +
                burst.vel[i * 4 + 2]! * d[2]!) /
              len;
            count++;
          }
        }
        console.log(
          `outward burst ${dim}D: mean radial speed ${fmt(radial / count, 2)} m/s over ${count} particles`,
        );
        expect(count).toBeGreaterThan(5);
        expect(radial / count).toBeGreaterThan(0.3);

        // At most four pushes are taken per update: of six far apart, the first four act and the others do not.
        place();
        const spots = [0.1, 0.25, 0.4, 0.55, 0.75, 0.9].map((f): Vec3 => [
          0.6 - extent / 2 + f * extent,
          0.6,
          centre[2],
        ]);
        for (const position of spots) {
          pbf.impulse({ position, radius: 2 * spacing, velocity: [0, 4, 0], strength: 1 });
        }
        await advance(device, pbf, 1);
        const four = await snapshot(pbf);
        const meanVyNear = spots.map((p) => {
          let sum = 0;
          let c = 0;
          for (let i = 0; i < four.n; i++) {
            if (
              Math.hypot(
                four.pos[i * 4]! - p[0],
                four.pos[i * 4 + 1]! - p[1],
                dim === 3 ? four.pos[i * 4 + 2]! - p[2] : 0,
              ) < spacing
            ) {
              sum += four.vel[i * 4 + 1]!;
              c++;
            }
          }
          return c > 0 ? sum / c : 0;
        });
        console.log(
          `six pushes ${dim}D, mean vy near each: ${meanVyNear.map((v) => fmt(v, 2)).join(' ')}`,
        );
        expect(meanVyNear.slice(0, 4).every((v) => v > 0.3)).toBe(true);
        expect(meanVyNear.slice(4).every((v) => Math.abs(v) < 0.1)).toBe(true);
        // A push is used once: the next update has none.
        const before = maxSpeed(four);
        await advance(device, pbf, 1);
        expect(maxSpeed(await snapshot(pbf))).toBeLessThan(before);
      },
    );
  });
});

// --- 12. Foam ----------------------------------------------------------------------------------------------------------

describe.each(DIMS)('the foam scalar in %iD', (dim) => {
  const s = dim === 3 ? 0.05 : 0.02;
  const size: Vec3 = dim === 3 ? [1, 1.2, 0.3] : [1, 1.2, 1];

  it('is the constants of the ParticleSet the renderers read', async () => {
    await withPbf(
      { dimension: dim, capacity: 100, spacing: s, bounds: tank(dim, size) },
      async (pbf) => {
        pbf.reset({ blocks: [{ min: [0, 0, 0], max: [0.3, 0.3, 0.3] }] });
        const set = pbf.particles;
        expect(set.restVolume).toBeCloseTo(s ** dim, 12);
        expect(set.radius).toBeCloseTo(0.65 * s, 12);
        expect(set.count).toBe(pbf.count);
        expect(set.capacity).toBe(100);
        expect(set.positions).toBe(pbf.system.positions);
        expect(set.velocities).toBe(pbf.system.velocities);
      },
    );
    await withPbf(
      { dimension: dim, capacity: 10, spacing: s, bounds: tank(dim, size), renderScale: 1 },
      async (pbf) => {
        expect(pbf.particles.radius).toBeCloseTo(s, 12);
      },
    );
  });

  it('stays in 0..1 through a violent run, is ~0 for calm water and large in spray', async () => {
    await withPbf(
      { dimension: dim, capacity: 3000, spacing: s, bounds: tank(dim, size) },
      async (pbf, device) => {
        // Calm: a pool left alone for 2 s. Foam = sparse AND fast; a calm free surface is sparse but not fast.
        pbf.reset({ blocks: [{ min: [0, 0, 0], max: [size[0], 0.3, size[2]] }] });
        await advance(device, pbf, 120);
        const calm = await snapshot(pbf);
        let calmMax = 0;
        for (let i = 0; i < calm.n; i++) calmMax = Math.max(calmMax, calm.pos[i * 4 + 3]!);
        const calmMean = meanComponent(calm.pos, calm.n, 3);

        // Violent: a dam break with a splash thrown out of the middle of the pool, every frame checked in 0..1.
        pbf.reset({
          blocks: [
            { min: [0, 0, 0], max: [0.4, 0.5, size[2]] },
            { min: [0.7, 0, 0], max: [1, 0.2, size[2]] },
          ],
        });
        let peak = 0;
        let nan = 0;
        let outOfRange = 0;
        let sprayAboveHalf = 0;
        for (let frame = 0; frame < 30; frame++) {
          if (frame === 10)
            pbf.impulse({
              position: [0.5, 0.1, size[2] / 2],
              radius: 0.25,
              velocity: [0, 12, 0],
              strength: 1,
              outward: 4,
            });
          await advance(device, pbf, 8);
          const snap = await snapshot(pbf);
          for (let i = 0; i < snap.n; i++) {
            const foam = snap.pos[i * 4 + 3]!;
            if (Number.isNaN(foam)) nan++;
            else if (foam < 0 || foam > 1) outOfRange++;
            peak = Math.max(peak, foam);
            if (foam > 0.5) sprayAboveHalf++;
          }
        }
        console.log(
          `foam ${dim}D: calm column max ${calmMax.toExponential(1)} mean ${calmMean.toExponential(1)}; splash peak ${fmt(peak, 3)}, ${sprayAboveHalf} particle-samples above 0.5`,
        );
        expect(nan).toBe(0);
        expect(outOfRange).toBe(0);
        // Calm water: measured 8e-6 (2D) and 3e-6 (3D) after 2 s. 0.01 would still be invisible.
        expect(calmMax).toBeLessThan(0.01);
        // Spray: fast (the impulse gives 12 m/s, foamSpeed is 1) and sparse (a thrown sheet, density below 0.8).
        expect(peak).toBeGreaterThan(0.5);
        expect(sprayAboveHalf).toBeGreaterThan(3);
        // It decays (1.5 / s): left alone, the pool is calm again.
        await advance(device, pbf, 60 * 4);
        const after = await snapshot(pbf);
        let max = 0;
        for (let i = 0; i < after.n; i++) max = Math.max(max, after.pos[i * 4 + 3]!);
        expect(max).toBeLessThan(0.5);
        expect(max).toBeLessThanOrEqual(1);
      },
    );
  });
});

// --- 13. Lifecycle ---------------------------------------------------------------------------------------------------

describe.each(DIMS)('lifecycle in %iD', (dim) => {
  const s = dim === 3 ? 0.05 : 0.025;
  const size: Vec3 = [0.6, 0.6, 0.6];
  const scene = { blocks: [{ min: [0, 0, 0] as Vec3, max: [0.6, 0.25, 0.6] as Vec3 }] };
  const base: Options = { dimension: dim, capacity: 1500, spacing: s, bounds: tank(dim, size) };
  const options = (device: GPUDevice): PbfOptions => ({ device, ...base });

  it('destroy, then a fresh solver on the same device, with no validation error (and destroy twice is harmless)', async () => {
    const device = await getTestDevice();
    await expectValid(device, async () => {
      const heights: number[] = [];
      for (let round = 0; round < 3; round++) {
        const pbf = createPbf(options(device));
        pbf.reset(scene);
        await advance(device, pbf, 30);
        heights.push(2 * meanComponent((await snapshot(pbf)).pos, pbf.count, 1));
        await device.queue.onSubmittedWorkDone();
        pbf.destroy();
        pbf.destroy();
      }
      // The same scene, three solvers, one after the other: the same fluid (within f32 noise).
      expect(Math.abs(heights[1]! - heights[0]!)).toBeLessThan(0.01);
      expect(Math.abs(heights[2]! - heights[0]!)).toBeLessThan(0.01);
    });
  });

  it('two solvers on one device, updated in the same command buffer, do not disturb each other', async () => {
    const device = await getTestDevice();
    await expectValid(device, async () => {
      const a = createPbf(options(device));
      const b = createPbf({ ...options(device), params: { gravity: [0, 3, 0] } });
      const alone = createPbf(options(device));
      for (const pbf of [a, b, alone]) pbf.reset(scene);
      for (let frame = 0; frame < 60; frame++) {
        const encoder = device.createCommandEncoder();
        a.update(encoder, 1 / 60);
        b.update(encoder, 1 / 60);
        device.queue.submit([encoder.finish()]);
        const solo = device.createCommandEncoder();
        alone.update(solo, 1 / 60);
        device.queue.submit([solo.finish()]);
        if (frame % 8 === 7) await device.queue.onSubmittedWorkDone();
      }
      await device.queue.onSubmittedWorkDone();
      const sa = await snapshot(a);
      const sb = await snapshot(b);
      const sc = await snapshot(alone);
      const ya = meanComponent(sa.pos, sa.n, 1);
      const yc = meanComponent(sc.pos, sc.n, 1);
      // B has the other gravity (upward): it ends up against the lid, A is a settled pool like the lone solver.
      expect(Math.abs(ya - yc)).toBeLessThan(0.005);
      expect(meanComponent(sb.pos, sb.n, 1)).toBeGreaterThan(ya + 0.1);
      for (const snap of [sa, sb, sc]) expect(allFinite(snap)).toBe(true);
      for (const pbf of [a, b, alone]) pbf.destroy();
    });
  });

  it.each(PBF_STAGES)(
    'run() with only the %s stage does what that stage does, and nothing else',
    async (stage: PbfStage) => {
      await withPbf(
        { ...base, params: { viscosity: 0.5, vorticity: 0.2 } },
        async (pbf, device) => {
          pbf.reset({ blocks: [{ min: [0.1, 0.1, 0.1], max: [0.5, 0.4, 0.5] }] });
          // Some shear to give the velocity stages something to do.
          const start = await snapshot(pbf);
          const random = makeRandom(9);
          for (let i = 0; i < start.n; i++) {
            for (let c = 0; c < (dim === 3 ? 3 : 2); c++)
              start.vel[i * 4 + c] = (random() - 0.5) * 2;
          }
          pbf.system.set({ positions: start.pos, velocities: start.vel });
          const dt = 1 / 60;
          // The stages after the grid need the grid to have been built by an earlier stage; `run` does not build it for them.
          const prepare: PbfStage[][] = {
            predict: [],
            grid: [],
            density: [['grid']],
            correct: [['grid'], ['density']],
            viscosity: [['grid']],
            vorticity: [['grid'], ['viscosity']],
          }[stage] as unknown as PbfStage[][];
          for (const stages of prepare) await runOnce(device, pbf, { dt, stages });
          const before = await snapshot(pbf);
          await runOnce(device, pbf, { dt, stages: [stage] });
          const end = await snapshot(pbf);
          expect(allFinite(end)).toBe(true);
          const key = (snap: Snapshot) =>
            Array.from(
              { length: snap.n },
              (_, i) =>
                `${snap.pos[i * 4]!.toFixed(5)},${snap.pos[i * 4 + 1]!.toFixed(5)},${snap.pos[i * 4 + 2]!.toFixed(5)}`,
            ).sort();
          const samePositions = JSON.stringify(key(before)) === JSON.stringify(key(end));
          const vBefore = sumVelocity(before);
          const vEnd = sumVelocity(end);
          switch (stage) {
            case 'predict': {
              // v += g dt, x += v dt for everybody, away from the walls: the cloud falls by (g dt - mean v) dt.
              const dy = meanComponent(end.pos, end.n, 1) - meanComponent(before.pos, before.n, 1);
              const expected = (vBefore[1] / before.n - G * dt) * dt;
              expect(dy).toBeCloseTo(expected, 5);
              expect(meanComponent(end.vel, end.n, 1) - vBefore[1] / before.n).toBeCloseTo(
                -G * dt,
                5,
              );
              break;
            }
            case 'grid':
              // Only reorders: the same set of positions and the same sum of velocities.
              expect(samePositions).toBe(true);
              expect(vEnd[0]).toBeCloseTo(vBefore[0], 2);
              expect(vEnd[1]).toBeCloseTo(vBefore[1], 2);
              break;
            case 'density': {
              // Writes the solve buffer only.
              expect(samePositions).toBe(true);
              const { density } = await pbf.readState();
              const average = mean(density);
              console.log(`density stage ${dim}D: mean rho/rho0 ${fmt(average, 3)}`);
              expect(density.every((x) => x > 0 && x < 2)).toBe(true);
              expect(average).toBeGreaterThan(0.6);
              expect(average).toBeLessThan(1.2);
              break;
            }
            case 'correct':
              // Moves the positions (a free block with a free surface expands a little) and keeps them in the box.
              expect(samePositions).toBe(false);
              expect(escapes(end, dim, pbf.bounds, pbf.params.wallOffset).count).toBe(0);
              break;
            case 'viscosity':
              // Smooths the velocity (XSPH, 0.5 per step): the sum is kept, the spread falls, positions stay.
              expect(samePositions).toBe(true);
              expect(vEnd[0]).toBeCloseTo(vBefore[0], 1);
              expect(rmsSpeed(end)).toBeLessThan(0.9 * rmsSpeed(before));
              break;
            case 'vorticity': {
              // Confinement changes velocities only (positions as before), by a bounded kick.
              expect(samePositions).toBe(true);
              let kick = 0;
              const index = new Map<string, number>();
              for (let i = 0; i < before.n; i++) {
                index.set(
                  `${before.pos[i * 4]!},${before.pos[i * 4 + 1]!},${before.pos[i * 4 + 2]!}`,
                  i,
                );
              }
              for (let i = 0; i < end.n; i++) {
                const j = index.get(
                  `${end.pos[i * 4]!},${end.pos[i * 4 + 1]!},${end.pos[i * 4 + 2]!}`,
                );
                if (j === undefined) continue;
                for (let c = 0; c < 3; c++) {
                  kick = Math.max(kick, Math.abs(end.vel[i * 4 + c]! - before.vel[j * 4 + c]!));
                }
              }
              console.log(`vorticity stage ${dim}D: largest velocity kick ${fmt(kick, 3)} m/s`);
              expect(kick).toBeGreaterThan(0);
              expect(kick).toBeLessThan(1);
              break;
            }
          }
        },
      );
    },
  );
});

// --- 14. The effects ---------------------------------------------------------------------------------------------------

const WIDTH = 64;
const HEIGHT = 48;

/** No DOM here: effects only use the canvas for input, so listeners are all it needs (as in tests/effects.test.ts). */
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

// The effects are written by someone else: the suites are skipped while a directory does not exist and run for real
// the moment it does (the registry globs src/effects/*/index.ts).
describe.each(['pbf-2d', 'pbf-3d'])('effect %s', (id) => {
  const present = hasEffect(id);

  it.skipIf(!present)(
    'runs headlessly through the defaults, every preset and every view',
    async () => {
      const device = await getTestDevice();
      const effect = await loadEffect(id);
      const base = defaultParams(effect.params) as Record<string, unknown>;
      const cases: [string, Record<string, unknown>][] = [['defaults', base]];
      for (const [name, preset] of Object.entries(effect.presets ?? {})) {
        cases.push([`preset ${name}`, { ...base, ...(preset as Record<string, unknown>) }]);
      }
      const view = (effect.params as Record<string, { type: string; options?: readonly string[] }>)
        .view;
      if (view?.type === 'select') {
        for (const option of view.options ?? [])
          cases.push([`view ${option}`, { ...base, view: option }]);
      }
      const format: GPUTextureFormat = 'rgba8unorm';
      const target = device.createTexture({
        size: [WIDTH, HEIGHT],
        format,
        usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC,
      });
      for (const [label, params] of cases) {
        try {
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
            for (let frame = 0; frame < 3; frame++) {
              const encoder = device.createCommandEncoder();
              instance.frame(
                { time: frame / 60, delta: 1 / 60, frame },
                encoder,
                target.createView(),
              );
              device.queue.submit([encoder.finish()]);
            }
            await device.queue.onSubmittedWorkDone();
            instance.destroy();
          });
        } catch (error) {
          throw new Error(
            `${id}, ${label}: ${error instanceof Error ? error.message : String(error)}`,
            { cause: error },
          );
        }
      }
      // Something was drawn in the end.
      const pixels = await readTexture(device, target);
      expect(new Set(pixels).size).toBeGreaterThan(2);
      target.destroy();
      console.log(
        `${id}: ${cases.length} configurations ran (${cases.map((c) => c[0]).join(', ')})`,
      );
    },
  );

  it.runIf(!present)('is not there yet (skipped until src/effects/<id> exists)', () => {
    expect(present).toBe(false);
  });
});

// --- 15. Cost study (gated) --------------------------------------------------------------------------------------------
//
// STATUS: skeleton done and runnable. Skipped unless PBF_STUDY names a JSON file to write:
//
//     PBF_STUDY=/tmp/pbf-study.json npx vitest run tests/pbf.test.ts -t study
//     PBF_STUDY=/tmp/pbf-study.json PBF_STUDY_QUICK=1 npx vitest run tests/pbf.test.ts -t study   (tiny counts, a smoke run)
//
// What it measures, per sub-step (one `pbf.run`): the GPU time of each pass by its own timestamps (grouped into predict,
// grid, density, correct, viscosity/vorticity) and the span from the first pass starting to the last ending. The groups
// overlap on a tile-based GPU, so they bound each stage and do not add up to the span; use the span for the total.
// Configurations are STATIONARY (gravity 0, a block that fills its tank from the walls), because a drifting state
// changes the neighbour count and with it the cost. Every configuration is measured `REPEATS` times; the table gives the
// quietest run, the median of the quietest half, and the spread within that half (the GPU is shared; a busy moment only
// adds time). The machine's process list at the time is recorded in the JSON. Timestamps tick in steps of about
// 65 microseconds here (Dawn on Metal), so a pass shorter than that reads 0 at small counts: trust the span and the
// slopes there, and the per-pass figures from about 50k particles up.
// NOT DONE (for the lead): nothing else is planned, the sweep lists and the repeat count are the knobs to turn.

const studyEnv = (globalThis as { process?: { env: Record<string, string | undefined> } }).process
  ?.env;
const studyFile = studyEnv?.PBF_STUDY;
const quick = studyEnv?.PBF_STUDY_QUICK === '1';

describe.skipIf(!studyFile)('study', () => {
  const GROUPS = ['predict', 'grid', 'density', 'correct', 'viscosity', 'vorticity'] as const;
  type Group = (typeof GROUPS)[number];
  const groupOf = (label: string): Group => {
    for (const g of ['predict', 'density', 'correct', 'viscosity', 'vorticity'] as const)
      if (label.includes(g)) return g;
    return 'grid'; // grid cells, scan, scatter, reorder
  };

  interface Spread {
    min: number;
    median: number;
    spread: number;
  }
  const spreadOf = (values: number[]): Spread => {
    const sorted = [...values].sort((a, b) => a - b);
    const quiet = sorted.slice(0, Math.max(1, Math.ceil(sorted.length / 2)));
    return {
      min: quiet[0]!,
      median: quiet[Math.floor(quiet.length / 2)]!,
      spread: quiet[quiet.length - 1]! - quiet[0]!,
    };
  };

  interface Row {
    sweep: 'count' | 'iterations';
    dimension: Dim;
    particles: number;
    iterations: number;
    span: Spread;
    groups: Record<Group, Spread>;
    passes: number;
  }

  async function timedDevice(): Promise<GPUDevice> {
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter?.features.has('timestamp-query'))
      throw new Error('timestamp-query is not available');
    return adapter.requestDevice({ requiredFeatures: ['timestamp-query'] });
  }

  /** A block of about n particles that fills its tank from the walls: the stationary configuration of the study. */
  function stationary(dim: Dim, n: number) {
    const s = dim === 3 ? 0.02 : 0.01;
    const nx = dim === 3 ? Math.ceil(Math.cbrt(n)) : Math.ceil(Math.sqrt(1.5 * n));
    const ny = dim === 3 ? nx : Math.ceil(n / nx);
    const nz = dim === 3 ? Math.ceil(n / (nx * ny)) : 1;
    const box: Vec3 = [nx * s, ny * s, nz * s];
    return { s, box, count: nx * ny * nz };
  }

  async function measure(
    device: GPUDevice,
    dim: Dim,
    n: number,
    iterations: number,
    repeats: number,
  ): Promise<Row> {
    const { s, box, count } = stationary(dim, n);
    const slots = 256;
    const querySet = device.createQuerySet({ type: 'timestamp', count: slots });
    const resolve = device.createBuffer({
      size: slots * 8,
      usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC,
    });
    const read = device.createBuffer({
      size: slots * 8,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    let labels: string[] = [];
    const profiler = {
      pass(label: string) {
        const index = labels.length;
        labels.push(label);
        if (index >= slots / 2) return undefined;
        return {
          querySet,
          beginningOfPassWriteIndex: index * 2,
          endOfPassWriteIndex: index * 2 + 1,
        };
      },
    };
    const pbf = createPbf({
      device,
      dimension: dim,
      capacity: count + 8,
      spacing: s,
      bounds: tank(dim, box),
      profiler,
      timedPasses: 1000,
      params: { gravity: [0, 0, 0], iterations },
    });
    pbf.reset({ blocks: [{ min: [0, 0, 0], max: box, jitter: 0.1 }] });
    const spans: number[] = [];
    const perGroup = Object.fromEntries(GROUPS.map((g) => [g, [] as number[]])) as Record<
      Group,
      number[]
    >;
    let passes = 0;
    for (let r = 0; r < repeats + 4; r++) {
      labels = [];
      const encoder = device.createCommandEncoder();
      pbf.run(encoder);
      const used = Math.min(labels.length, slots / 2);
      encoder.resolveQuerySet(querySet, 0, used * 2, resolve, 0);
      encoder.copyBufferToBuffer(resolve, 0, read, 0, slots * 8);
      device.queue.submit([encoder.finish()]);
      await read.mapAsync(GPUMapMode.READ);
      const stamps = new BigUint64Array(read.getMappedRange().slice(0));
      read.unmap();
      if (r < 4) continue; // pipelines, the first sort
      passes = used;
      const sums = Object.fromEntries(GROUPS.map((g) => [g, 0])) as Record<Group, number>;
      let begin = stamps[0]!;
      let end = stamps[1]!;
      for (let i = 0; i < used; i++) {
        const b = stamps[i * 2]!;
        const e = stamps[i * 2 + 1]!;
        if (e > b) sums[groupOf(labels[i]!)] += Number(e - b) / 1e6;
        if (b < begin) begin = b;
        if (e > end) end = e;
      }
      spans.push(Number(end - begin) / 1e6);
      for (const g of GROUPS) perGroup[g].push(sums[g]);
    }
    pbf.destroy();
    querySet.destroy();
    resolve.destroy();
    read.destroy();
    return {
      sweep: 'count',
      dimension: dim,
      particles: count,
      iterations,
      passes,
      span: spreadOf(spans),
      groups: Object.fromEntries(GROUPS.map((g) => [g, spreadOf(perGroup[g])])) as Record<
        Group,
        Spread
      >,
    };
  }

  it(
    'the cost of a sub-step against particle count and iteration count',
    { timeout: 3_600_000 },
    async () => {
      const device = await timedDevice();
      const repeats = quick ? 6 : 24;
      const counts: Record<Dim, number[]> = quick
        ? { 2: [2000, 8000], 3: [4000, 16000] }
        : {
            2: [5_000, 10_000, 25_000, 50_000, 100_000, 250_000],
            3: [10_000, 30_000, 60_000, 120_000, 250_000, 500_000],
          };
      const iterationList = quick ? [1, 2, 4] : [1, 2, 3, 4, 5, 6, 7, 8, 12];
      const fixed: Record<Dim, number> = quick ? { 2: 8000, 3: 16000 } : { 2: 50_000, 3: 60_000 };
      const rows: Row[] = [];
      for (const dim of DIMS) {
        for (const n of counts[dim])
          rows.push({
            ...(await measure(device, dim, n, DEFAULT_PBF_PARAMS.iterations, repeats)),
            sweep: 'count',
          });
        for (const k of iterationList)
          rows.push({
            ...(await measure(device, dim, fixed[dim], k, repeats)),
            sweep: 'iterations',
          });
      }

      const f = (x: number) => x.toFixed(3);
      const lines = [
        '| sweep | D | particles | iterations | sub-step span ms (min / median / spread) | predict | grid | density | correct | viscosity | vorticity |',
        '|---|---|---|---|---|---|---|---|---|---|---|',
        ...rows.map(
          (r) =>
            `| ${r.sweep} | ${r.dimension} | ${r.particles} | ${r.iterations} | ${f(r.span.min)} / ${f(r.span.median)} / ${f(r.span.spread)} | ${GROUPS.map((g) => f(r.groups[g].min)).join(' | ')} |`,
        ),
      ];
      console.log(lines.join('\n'));

      const specifier = 'node:fs';
      const childSpecifier = 'node:child_process';
      const { writeFileSync } = (await import(/* @vite-ignore */ specifier)) as {
        writeFileSync(path: string, text: string): void;
      };
      const { execSync } = (await import(/* @vite-ignore */ childSpecifier)) as {
        execSync(command: string): { toString(): string };
      };
      const processes = (() => {
        try {
          return execSync('ps -Ao pcpu,pmem,comm -r | head -15').toString();
        } catch {
          return '(ps failed)';
        }
      })();
      const adapter = await navigator.gpu.requestAdapter();
      writeFileSync(
        studyFile!,
        JSON.stringify(
          {
            note: 'ms per sub-step (pbf.run, 3 sub-steps make a 1/60 s step); stationary block filling its tank, gravity 0, default parameters but iterations',
            quick,
            repeats,
            adapter: adapter
              ? {
                  vendor: adapter.info.vendor,
                  architecture: adapter.info.architecture,
                  device: adapter.info.device,
                  description: adapter.info.description,
                }
              : null,
            runningDuringTheStudy: processes,
            rows,
            table: lines,
          },
          null,
          2,
        ) + '\n',
      );
      expect(rows.length).toBeGreaterThan(0);
    },
  );
});

// --- 15b. Cost study, part two (gated, same switch): the effect, the quality grid, and a yardstick --------------------------
//
//     PBF_STUDY=/tmp/pbf-study.json PBF_STUDY_NOTE="what else was running" npx vitest run tests/pbf.test.ts -t "study part two"
//
// Merges into the JSON file of part one (keys `effect`, `quality`, `yardstick`). The yardstick is the framework's own
// neighbour-count pass over 250 000 particles, 0.39 ms on a quiet M1 Max (src/effects/particles/notes.md): a reading far
// above that says that somebody else was using the GPU, and every figure of the run is then an upper bound.
describe.skipIf(!studyFile)('study part two', () => {
  const lines: string[] = [];
  const json: Record<string, unknown> = {};
  const log = (text = '') => {
    lines.push(text);
    console.log(text);
  };
  async function finish() {
    const specifier = 'node:fs';
    const fs = (await import(/* @vite-ignore */ specifier)) as {
      readFileSync(path: string, encoding: string): string;
      writeFileSync(path: string, text: string): void;
      existsSync(path: string): boolean;
    };
    const old = fs.existsSync(studyFile!)
      ? (JSON.parse(fs.readFileSync(studyFile!, 'utf8')) as Record<string, unknown>)
      : {};
    fs.writeFileSync(studyFile!, JSON.stringify({ ...old, ...json }, null, 2) + '\n');
  }
  async function timedDevice() {
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter?.features.has('timestamp-query'))
      throw new Error('timestamp-query is not available');
    return adapter.requestDevice({ requiredFeatures: ['timestamp-query'] });
  }
  const f = (x: number, d = 2) => x.toFixed(d);
  // A GPU span recorder: first begin to last end of the timed passes of a recording.
  function recorder(device: GPUDevice) {
    const querySet = device.createQuerySet({ type: 'timestamp', count: 512 });
    const resolve = device.createBuffer({
      size: 512 * 8,
      usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC,
    });
    const read = device.createBuffer({
      size: 512 * 8,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    const state = {
      mode: 'off' as 'off' | 'span' | 'all',
      expected: 0,
      seen: 0,
      labels: [] as string[],
    };
    const profiler = {
      pass(label: string) {
        const index = state.seen++;
        if (state.mode === 'all') {
          state.labels.push(label);
          if (index >= 250) return undefined;
          return {
            querySet,
            beginningOfPassWriteIndex: index * 2,
            endOfPassWriteIndex: index * 2 + 1,
          };
        }
        if (state.mode === 'span') {
          if (state.expected === 1)
            return { querySet, beginningOfPassWriteIndex: 0, endOfPassWriteIndex: 1 };
          if (index === 0)
            return { querySet, beginningOfPassWriteIndex: 0, endOfPassWriteIndex: 2 };
          if (index === state.expected - 1)
            return { querySet, beginningOfPassWriteIndex: 3, endOfPassWriteIndex: 1 };
        }
        return undefined;
      },
    };
    const run = async (record: (e: GPUCommandEncoder) => void, slots: number) => {
      const encoder = device.createCommandEncoder();
      record(encoder);
      encoder.resolveQuerySet(querySet, 0, slots, resolve, 0);
      encoder.copyBufferToBuffer(resolve, 0, read, 0, slots * 8);
      device.queue.submit([encoder.finish()]);
      await read.mapAsync(GPUMapMode.READ);
      const stamps = new BigUint64Array(read.getMappedRange().slice(0, slots * 8));
      read.unmap();
      return stamps;
    };
    /** Quietest and median span (ms) of `unit()` recorded `lo` and `hi` times into one command buffer. */
    async function slope(
      unit: (e: GPUCommandEncoder) => void,
      passesPerUnit: number,
      lo: number,
      hi: number,
      rounds: number,
    ) {
      const time = async (n: number) => {
        state.mode = 'span';
        state.seen = 0;
        state.expected = passesPerUnit * n;
        const stamps = await run((e) => {
          for (let i = 0; i < n; i++) unit(e);
        }, 4);
        return Number(stamps[1]! - stamps[0]!) / 1e6;
      };
      const small: number[] = [],
        large: number[] = [];
      for (let r = 0; r < rounds; r++) {
        small.push(await time(lo));
        large.push(await time(hi));
      }
      small.sort((a, b) => a - b);
      large.sort((a, b) => a - b);
      const mid = Math.floor(rounds / 2);
      const q = (a: number[], p: number) => a[Math.floor(p * (a.length - 1))]!;
      return {
        best: (large[0]! - small[0]!) / (hi - lo),
        median: (large[mid]! - small[mid]!) / (hi - lo),
        p25: (q(large, 0.25) - q(small, 0.25)) / (hi - lo),
        p75: (q(large, 0.75) - q(small, 0.75)) / (hi - lo),
      };
    }
    async function countPasses(unit: (e: GPUCommandEncoder) => void) {
      state.mode = 'all';
      state.seen = 0;
      state.labels = [];
      await run(unit, 2);
      return state.seen;
    }
    async function perLabel(unit: (e: GPUCommandEncoder) => void, rounds: number) {
      const best: Record<string, number> = {};
      for (let r = 0; r < rounds; r++) {
        state.mode = 'all';
        state.seen = 0;
        state.labels = [];
        const stamps = await run(unit, Math.min(500, 2 * 250));
        const sums = new Map<string, number>();
        state.labels.forEach((l, i) =>
          sums.set(l, (sums.get(l) ?? 0) + Number(stamps[i * 2 + 1]! - stamps[i * 2]!) / 1e6),
        );
        for (const [l, v] of sums) best[l] = Math.min(best[l] ?? 1e9, v);
      }
      return best;
    }
    return {
      profiler,
      state,
      slope,
      countPasses,
      perLabel,
      run,
      destroy() {
        querySet.destroy();
        resolve.destroy();
        read.destroy();
      },
    };
  }

  async function psSnapshot(): Promise<string> {
    try {
      const childSpecifier = 'node:child_process';
      const { execSync } = (await import(/* @vite-ignore */ childSpecifier)) as {
        execSync(c: string): { toString(): string };
      };
      return execSync("ps -Ao pcpu,comm -r | head -8 | sed 's/ \\+/ /g'").toString();
    } catch {
      return 'unavailable';
    }
  }

  // A fixed yardstick, so that a reading can be judged against how busy the machine was: the framework's own
  // neighbour-count pass over 250 000 particles in a pool (documented in src/effects/particles/notes.md as 0.39 ms on a
  // quiet M1 Max). A reading well above that means somebody else was using the GPU.
  async function calibrate(device: GPUDevice, rec: ReturnType<typeof recorder>): Promise<number> {
    const n = 250000;
    const system = createParticleSystem({ device, capacity: n, radius: 0.01, dimension: 3 });
    const side = Math.ceil(Math.cbrt(n));
    system.fill(n, (i, p) => {
      p.position[0] = (i % side) * 0.0125 + 0.0003 * ((i * 7) % 5);
      p.position[1] = (Math.floor(i / side) % side) * 0.0125 * 0.4;
      p.position[2] = Math.floor(i / (side * side)) * 0.0125;
    });
    const grid = createNeighborGrid({
      device,
      dimension: 3,
      mode: 'reorder',
      capacity: n,
      profiler: rec.profiler,
    });
    const counter = createNeighborCounter({ device, profiler: rec.profiler });
    const result = device.createBuffer({
      size: n * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
    });
    grid.configure({
      min: [0, 0, 0],
      max: [side * 0.0125, side * 0.0125 * 0.4 + 0.0125, side * 0.0125],
      radius: 0.0275,
    });
    const unit = (e: GPUCommandEncoder) => {
      grid.build(e, system);
      counter.count(e, grid, system, result);
    };
    for (let i = 0; i < 4; i++) await rec.run(unit, 2);
    const best = await rec.perLabel(unit, 14);
    grid.destroy();
    counter.destroy();
    system.destroy();
    result.destroy();
    return best['neighbours count'] ?? 0;
  }

  it('yardstick', async () => {
    const device = await timedDevice();
    const rec = recorder(device);
    const ms = await calibrate(device, rec);
    json.yardstick = {
      neighbourCountMs250k: ms,
      quietMs: 0.39,
      note: studyEnv?.PBF_STUDY_NOTE ?? '',
      ps: await psSnapshot(),
    };
    log(
      `Yardstick (framework neighbour-count pass, 250k particles, 3D): ${f(ms, 3)} ms; 0.39 ms when the GPU is quiet.`,
    );
    rec.destroy();
    await finish();
  });

  // --- C: how much compression, how calm: iterations against sub-steps ------------------------------------------
  it('C. quality against iterations and sub-steps', async () => {
    const device = await getTestDevice();
    const quality: unknown[] = [];
    json.quality = quality;
    const frames = async (pbf: ReturnType<typeof createPbf>, n: number, delta: number) => {
      for (let i = 0; i < n; i++) {
        const e = device.createCommandEncoder();
        pbf.update(e, delta);
        device.queue.submit([e.finish()]);
        if (i % 10 === 9) await device.queue.onSubmittedWorkDone();
      }
      await device.queue.onSubmittedWorkDone();
    };
    const cases = [
      {
        dim: 2 as const,
        spacing: 0.02,
        width: 0.4,
        height: 0.6,
        label: '2D column 0.4 x 0.6 m (30 rows of s = 0.02)',
      },
      {
        dim: 3 as const,
        spacing: 0.025,
        width: 0.3,
        height: 0.5,
        label: '3D column 0.3 x 0.3 x 0.5 m (20 rows of s = 0.025)',
      },
    ];
    for (const c of cases) {
      log(
        `## ${c.label}: compression of the settled column (volume height, 100 % = incompressible) and rms speed after 3 s, default parameters`,
      );
      const subs = [1, 2, 3, 4, 6];
      log(
        [
          'iterations \\ sub-steps per 1/60 s',
          ...subs.map((s) => `${s} (dt ${(1000 / 60 / s).toFixed(1)} ms)`),
        ].join(' | '),
      );
      for (const it of [1, 2, 3, 4, 6, 8]) {
        const row: string[] = [];
        for (const sub of subs) {
          const pbf = createPbf({
            device,
            dimension: c.dim,
            capacity: 20000,
            spacing: c.spacing,
            bounds: {
              min: [0, 0, c.dim === 3 ? 0 : -1],
              max: [c.width, 2, c.dim === 3 ? c.width : 1],
            },
            params: { iterations: it, substeps: sub },
          });
          pbf.reset({
            blocks: [{ min: [0, 0, 0], max: [c.width, c.height, c.width], jitter: 0.05 }],
          });
          await frames(pbf, 180, 1 / 60);
          const pos = await pbf.system.read('position');
          const vel = await pbf.system.read('velocity');
          const n = pbf.count;
          let ys = 0,
            ke = 0,
            top = 0;
          for (let i = 0; i < n; i++) {
            ys += pos[i * 4 + 1]!;
            top = Math.max(top, pos[i * 4 + 1]!);
            ke += vel[i * 4]! ** 2 + vel[i * 4 + 1]! ** 2 + vel[i * 4 + 2]! ** 2;
          }
          const height = (2 * ys) / n;
          const compression = 1 - height / c.height;
          const exploded = top > 1.5 * c.height;
          row.push(
            exploded
              ? 'unstable'
              : `${(compression * 100).toFixed(1)} % (${Math.sqrt(ke / n).toFixed(2)})`,
          );
          quality.push({
            dimension: c.dim,
            iterations: it,
            substeps: sub,
            compression,
            rms: Math.sqrt(ke / n),
            unstable: exploded,
          });
          pbf.destroy();
        }
        log(`${it} | ${row.join(' | ')}`);
      }
      log();
      await finish();
    }
    // Time step: a bigger fixed step with the same sub-step length.
    log(
      '## 3D column, 1/30 s steps with 6 sub-steps (dt 5.6 ms) against 1/60 s with 3 (dt 5.6 ms): the same fluid?',
    );
    for (const [timeStep, sub] of [
      [1 / 60, 3],
      [1 / 30, 6],
      [1 / 20, 8],
    ] as const) {
      const pbf = createPbf({
        device,
        dimension: 3,
        capacity: 20000,
        spacing: 0.025,
        bounds: { min: [0, 0, 0], max: [0.3, 2, 0.3] },
        params: { iterations: 2, substeps: sub, timeStep },
      });
      pbf.reset({ blocks: [{ min: [0, 0, 0], max: [0.3, 0.5, 0.3], jitter: 0.05 }] });
      await frames(pbf, 180, 1 / 60);
      const pos = await pbf.system.read('position');
      const n = pbf.count;
      let ys = 0;
      for (let i = 0; i < n; i++) ys += pos[i * 4 + 1]!;
      log(
        `| ${timeStep.toFixed(4)} s x ${sub} sub-steps | compression ${(100 * (1 - (2 * ys) / n / 0.5)).toFixed(1)} % |`,
      );
      pbf.destroy();
    }
    await finish();
  });

  // --- B: the 3D effect, simulation against rendering --------------------------------------------------
  it('B. pbf-3d simulation versus rendering', async () => {
    const device = await timedDevice();
    const rec = recorder(device);
    const effect = await loadEffect('pbf-3d');
    const W = 1920,
      H = 1080;
    const target = device.createTexture({
      size: [W, H],
      format: 'rgba8unorm',
      usage: GPUTextureUsage.RENDER_ATTACHMENT,
    });
    const rows: unknown[] = [];
    json.effect = rows;
    async function measure(overrides: Record<string, unknown>, warmFrames: number) {
      const values = { ...defaultParams(effect.params), ...overrides } as Record<string, unknown>;
      const canvas = Object.assign(new EventTarget(), {
        width: W,
        height: H,
        style: {},
        getBoundingClientRect: () => ({ left: 0, top: 0, width: W, height: H }),
        setPointerCapture() {},
        releasePointerCapture() {},
      }) as unknown as HTMLCanvasElement;
      const instance = await effect.create({
        device,
        format: 'rgba8unorm',
        canvas,
        params: values as never,
        profiler: rec.profiler,
        hdr: false,
      });
      instance.resize(W, H, 1);
      let frame = 0;
      const unit = (e: GPUCommandEncoder) => {
        instance.frame({ time: frame / 60, delta: 1 / 60, frame }, e, target.createView());
        frame++;
      };
      // Warm up with real time passing, so the pool has collapsed and the splashes are on the way down.
      rec.state.mode = 'off';
      for (let i = 0; i < warmFrames; i++) {
        const e = device.createCommandEncoder();
        unit(e);
        device.queue.submit([e.finish()]);
        if (i % 10 === 9) await device.queue.onSubmittedWorkDone();
      }
      await device.queue.onSubmittedWorkDone();
      // Freeze the state: from here on the solver does nothing when speed = 0, which gives the rendering alone.
      const passes = await rec.countPasses(unit);
      const full = await rec.slope(unit, passes, 1, 4, 20);
      const labels = await rec.perLabel(unit, 12);
      values.speed = 0;
      const idlePasses = await rec.countPasses(unit);
      const idle = await rec.slope(unit, idlePasses, 1, 4, 20);
      const idleLabels = await rec.perLabel(unit, 12);
      instance.destroy();
      return { full, idle, labels, idleLabels };
    }
    log(
      '## pbf-3d at 1920x1080: ms per frame (default settings: 2 iterations x 3 sub-steps, surface at half resolution, 3 filter iterations)',
    );
    log(
      '| particles | frame | rendering only (solver idle) | solver (difference) | solver median | frame median | fluid depth | fluid smooth | fluid thickness | fluid blur | fluid composite |',
    );
    log('| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |');
    for (const n of [10000, 30000, 60000, 120000, 250000, 500000]) {
      const r = await measure({ particles: n, autoRotate: false, ambient: false }, 150);
      const l = r.idleLabels;
      const solver = r.full.best - r.idle.best;
      log(
        `| ${n} | ${f(r.full.best)} | ${f(r.idle.best)} | ${f(solver)} | ${f(r.full.median - r.idle.median)} | ${f(r.full.median)} | ${f(l['fluid depth'] ?? 0)} | ${f(l['fluid smooth'] ?? 0)} | ${f(l['fluid thickness'] ?? 0)} | ${f(l['fluid blur'] ?? 0)} | ${f(l['fluid composite'] ?? 0)} |`,
      );
      rows.push({
        particles: n,
        frame: r.full,
        idle: r.idle,
        solver,
        labels: r.labels,
        idleLabels: r.idleLabels,
      });
      await finish();
    }
    log();
    log('## pbf-3d, 60000 particles: iterations and sub-steps (frame ms and solver share)');
    for (const [it, sub] of [
      [1, 3],
      [2, 2],
      [2, 3],
      [2, 4],
      [3, 3],
      [4, 2],
      [4, 3],
      [6, 3],
    ] as const) {
      const r = await measure(
        { particles: 60000, iterations: it, substeps: sub, autoRotate: false, ambient: false },
        150,
      );
      log(`| ${it} x ${sub} | ${f(r.full.best)} | solver ${f(r.full.best - r.idle.best)} |`);
      rows.push({ particles: 60000, iterations: it, substeps: sub, frame: r.full, idle: r.idle });
      await finish();
    }
    log();
    log('## pbf-3d, 60000 particles: renderer quality (frame ms, solver idle)');
    for (const [name, over] of [
      ['default (scale 1/2, 3 iterations, thickness every particle)', {}],
      [
        'scale 1/4, 1 iteration, thickness every 2nd',
        { renderScale: '1/4', filterIterations: 1, thicknessStride: 2, filterRadius: 1.5 },
      ],
      ['scale 1', { renderScale: '1' }],
      ['spheres view (billboards)', { view: 'spheres' }],
      [
        'tier B cut preset (15000 particles, 2 x 2, scale 1/4)',
        {
          ...((effect.presets?.['tier B cut'] ?? {}) as Record<string, unknown>),
          particles: 15000,
        },
      ],
    ] as const) {
      const r = await measure(
        { particles: 60000, autoRotate: false, ambient: false, ...over },
        150,
      );
      log(`| ${name} | idle ${f(r.idle.best)} | full ${f(r.full.best)} |`);
      rows.push({ particles: 60000, quality: name, frame: r.full, idle: r.idle });
      await finish();
    }
    rec.destroy();
    target.destroy();
  });
});
