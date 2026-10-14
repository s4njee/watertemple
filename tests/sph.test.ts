import { describe, expect, it, vi } from 'vitest';
import {
  blockPoints,
  createSph,
  DEFAULT_SPH_PARAMS,
  deriveSph,
  hexSpacing,
  latticeDensity,
  obstacleDistance,
  sphWallTables,
  WALL_TABLE_SIZE,
  type SphParams,
  type SphSnapshot,
  type SphSolver,
} from '../src/gpu/sim/sph.ts';
import { expectValid, getTestDevice } from './gpu.ts';
import { gpuF32 } from './wgsl-eval.ts';

// Story 7.3: 2D SPH. The kernels are proved by numerical quadrature on the GPU, the solver by what a fluid has to
// do: hold its rest density, conserve momentum, settle hydrostatically, keep every particle in the box, lose energy
// to viscosity and survive absurd parameters. Particle systems are chaotic, so no test compares particle positions
// between two runs; they compare statistics.

vi.setConfig({ testTimeout: 120_000 });

// --- Helpers -----------------------------------------------------------------------------------

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

const KERNELS = '#include "sph-kernels.wgsl"\n';

/** Evaluates `expression` (a WGSL f32 expression in x, y, z) at each point; returns the values. */
async function evaluate(
  expression: string,
  points: Float32Array,
  prelude = '',
): Promise<Float32Array> {
  const out = await gpuF32(
    `${KERNELS}${prelude}
     fn f(p: vec4f) -> vec4f {
       let x = p.x; let y = p.y; let z = p.z;
       return vec4f(${expression}, 0.0, 0.0, 0.0);
     }`,
    points,
  );
  const values = new Float32Array(points.length / 4);
  for (let i = 0; i < values.length; i++) values[i] = out[i * 4]!;
  return values;
}

/** Midpoint-rule integral over [-h, h]^D of a kernel expression in x, y, z (h = 1). */
async function integrate(dimension: 2 | 3, expression: string, n: number): Promise<number> {
  const cell = 2 / n;
  const count = dimension === 2 ? n * n : n * n * n;
  const points = new Float32Array(count * 4);
  let at = 0;
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      for (let k = 0; k < (dimension === 3 ? n : 1); k++) {
        points[at++] = -1 + (i + 0.5) * cell;
        points[at++] = -1 + (j + 0.5) * cell;
        points[at++] = dimension === 3 ? -1 + (k + 0.5) * cell : 0;
        points[at++] = 0;
      }
    }
  }
  const values = await evaluate(expression, points);
  let sum = 0;
  for (const v of values) sum += v;
  return sum * cell ** dimension;
}

// --- Kernels -----------------------------------------------------------------------------------

describe('SPH kernels', () => {
  const N2 = 600;
  const N3 = 90;

  it('every 2D kernel integrates to 1 over its support', async () => {
    const poly6 = await integrate(2, 'sph_poly6_2d(x * x + y * y, 1.0)', N2);
    const spiky = await integrate(2, 'sph_spiky_2d(length(vec2f(x, y)), 1.0)', N2);
    expect(poly6).toBeCloseTo(1, 3);
    expect(spiky).toBeCloseTo(1, 3);
  });

  it('every 3D kernel integrates to 1 over its support', async () => {
    const poly6 = await integrate(3, 'sph_poly6_3d(x * x + y * y + z * z, 1.0)', N3);
    const spiky = await integrate(3, 'sph_spiky_3d(length(vec3f(x, y, z)), 1.0)', N3);
    expect(poly6).toBeCloseTo(1, 2);
    expect(spiky).toBeCloseTo(1, 2);
  });

  it('the 3D constants would be wrong in 2D (the reason there are two sets)', async () => {
    // The 3D poly6 constant used as a 2D kernel integrates to the wrong number: 315 / (64 pi) against 4 / pi.
    const wrong = await integrate(2, 'sph_poly6_3d(x * x + y * y, 1.0)', N2);
    expect(Math.abs(wrong - 1)).toBeGreaterThan(0.2);
  });

  it('the viscosity Laplacian satisfies the consistency condition of a Laplacian in 2D and 3D', async () => {
    // A discrete Laplacian reproduces the Laplacian of |x|^2 / (2 D), which is 1: integral L(r) r^2 / (2 D) = 1.
    const two = await integrate(
      2,
      'sph_viscosity_laplacian_2d(length(vec2f(x, y)), 1.0) * (x * x + y * y) / 4.0',
      N2,
    );
    const three = await integrate(
      3,
      'sph_viscosity_laplacian_3d(length(vec3f(x, y, z)), 1.0) * (x * x + y * y + z * z) / 6.0',
      N3,
    );
    expect(two).toBeCloseTo(1, 3);
    expect(three).toBeCloseTo(1, 2);
  });

  it('is zero at and beyond the support, positive inside, and finite everywhere', async () => {
    const radii = [0, 0.1, 0.5, 0.9, 0.999, 1, 1.5, 100];
    const points = new Float32Array(radii.length * 4);
    radii.forEach((r, i) => (points[i * 4] = r));
    const names = [
      'sph_poly6_2d(x * x, 1.0)',
      'sph_spiky_2d(x, 1.0)',
      'sph_viscosity_laplacian_2d(x, 1.0)',
      'sph_poly6_3d(x * x, 1.0)',
      'sph_spiky_3d(x, 1.0)',
      'sph_viscosity_laplacian_3d(x, 1.0)',
      'sph_cohesion(x, 1.0)',
      'length(sph_spiky_grad_2d(vec2f(x, 0.0), 1.0))',
      'length(sph_poly6_grad_2d(vec2f(x, 0.0), 1.0))',
      'length(sph_spiky_grad_3d(vec3f(x, 0.0, 0.0), 1.0))',
      'length(sph_poly6_grad_3d(vec3f(x, 0.0, 0.0), 1.0))',
    ];
    for (const expression of names) {
      const values = await evaluate(expression, points);
      radii.forEach((r, i) => {
        expect(Number.isFinite(values[i]!), `${expression} at ${r}`).toBe(true);
        if (r >= 1) expect(values[i]!, `${expression} at ${r}`).toBe(0);
        else if (!(r === 0 && expression.includes('grad'))) {
          expect(values[i]!, `${expression} at ${r}`).toBeGreaterThanOrEqual(0);
        }
      });
    }
    // The spiky kernel is a cone: positive slope magnitude at the centre, unlike poly6.
    const centre = await evaluate(
      'length(sph_spiky_grad_2d(vec2f(x, 0.0), 1.0))',
      new Float32Array([1e-4, 0, 0, 0]),
    );
    expect(centre[0]!).toBeCloseTo(30 / Math.PI, 2);
    const poly = await evaluate(
      'length(sph_poly6_grad_2d(vec2f(x, 0.0), 1.0))',
      new Float32Array([1e-4, 0, 0, 0]),
    );
    expect(poly[0]!).toBeLessThan(1e-2);
  });

  it('each gradient matches finite differences of its kernel', async () => {
    const random = makeRandom(11);
    const points = new Float32Array(400 * 4);
    for (let i = 0; i < 400; i++) {
      const r = 0.06 + 0.88 * random();
      const a = random() * Math.PI * 2;
      const b = Math.acos(2 * random() - 1);
      points[i * 4] = r * Math.sin(b) * Math.cos(a);
      points[i * 4 + 1] = r * Math.sin(b) * Math.sin(a);
      points[i * 4 + 2] = r * Math.cos(b);
    }
    const e = 2e-3;
    const check = async (
      label: string,
      kernel: (p: string) => string,
      gradient: string,
      dimension: 2 | 3,
    ) => {
      const vec = (dx: number, dy: number, dz: number) =>
        dimension === 2 ? `vec2f(x + ${dx}, y + ${dy})` : `vec3f(x + ${dx}, y + ${dy}, z + ${dz})`;
      const analytic = await evaluate(`${gradient}.x`, points);
      const analyticY = await evaluate(`${gradient}.y`, points);
      const numeric = await evaluate(
        `(${kernel(vec(e, 0, 0))} - ${kernel(vec(-e, 0, 0))}) / ${2 * e}`,
        points,
      );
      const numericY = await evaluate(
        `(${kernel(vec(0, e, 0))} - ${kernel(vec(0, -e, 0))}) / ${2 * e}`,
        points,
      );
      let scale = 0;
      for (const v of analytic) scale = Math.max(scale, Math.abs(v));
      let worst = 0;
      for (let i = 0; i < analytic.length; i++) {
        worst = Math.max(
          worst,
          Math.abs(analytic[i]! - numeric[i]!),
          Math.abs(analyticY[i]! - numericY[i]!),
        );
      }
      expect(worst / scale, label).toBeLessThan(0.02);
    };
    const p2 = 'vec2f(x, y)';
    const p3 = 'vec3f(x, y, z)';
    await check(
      'poly6 2D',
      (p) => `sph_poly6_2d(dot(${p}, ${p}), 1.0)`,
      `sph_poly6_grad_2d(${p2}, 1.0)`,
      2,
    );
    await check(
      'spiky 2D',
      (p) => `sph_spiky_2d(length(${p}), 1.0)`,
      `sph_spiky_grad_2d(${p2}, 1.0)`,
      2,
    );
    await check(
      'poly6 3D',
      (p) => `sph_poly6_3d(dot(${p}, ${p}), 1.0)`,
      `sph_poly6_grad_3d(${p3}, 1.0)`,
      3,
    );
    await check(
      'spiky 3D',
      (p) => `sph_spiky_3d(length(${p}), 1.0)`,
      `sph_spiky_grad_3d(${p3}, 1.0)`,
      3,
    );
  });

  it('the kernel length scale is h: W(r; h) = W(r / h; 1) / h^D', async () => {
    const points = new Float32Array([0.3, 0, 0, 0, 0.7, 0.1, 0, 0]);
    const a = await evaluate('sph_poly6_2d(x * x + y * y, 1.0)', points);
    const b = await evaluate('sph_poly6_2d(x * x * 4.0 + y * y * 4.0, 2.0) * 4.0', points);
    expect(b[0]!).toBeCloseTo(a[0]!, 4);
    expect(b[1]!).toBeCloseTo(a[1]!, 4);
  });

  it('the coincident direction is a unit vector, equal and opposite for a pair, different for different pairs', async () => {
    const pairs: [number, number][] = [
      [3, 7],
      [7, 3],
      [10, 11],
      [11, 10],
      [4, 9],
    ];
    const points = new Float32Array(pairs.length * 4);
    pairs.forEach(([i, j], k) => {
      points[k * 4] = i;
      points[k * 4 + 1] = j;
    });
    const x = await evaluate('sph_coincident_direction(u32(x), u32(y)).x', points);
    const y = await evaluate('sph_coincident_direction(u32(x), u32(y)).y', points);
    for (let k = 0; k < pairs.length; k++) expect(Math.hypot(x[k]!, y[k]!)).toBeCloseTo(1, 5);
    expect(x[0]! + x[1]!).toBeCloseTo(0, 6);
    expect(y[0]! + y[1]!).toBeCloseTo(0, 6);
    expect(x[2]! + x[3]!).toBeCloseTo(0, 6);
    expect(Math.abs(x[0]! - x[2]!) + Math.abs(y[0]! - y[2]!)).toBeGreaterThan(1e-3);
  });

  it('the cohesion profile peaks at half the support and vanishes at both ends', async () => {
    const q = [0, 0.25, 0.5, 0.75, 1];
    const points = new Float32Array(q.length * 4);
    q.forEach((v, i) => (points[i * 4] = v));
    const v = await evaluate('sph_cohesion(x, 1.0)', points);
    expect(v[0]!).toBe(0);
    expect(v[4]!).toBe(0);
    expect(v[1]!).toBeCloseTo(0.5625, 6);
    expect(v[2]!).toBeCloseTo(1, 6);
    expect(v[1]!).toBeCloseTo(v[3]!, 6);
  });
});

// --- Lattice, wall tables, derived quantities --------------------------------------------------------

describe('rest density and the wall tables', () => {
  it('the lattice sum of poly6 is the kernel integral (1) to within a percent for smoothing ratios of 2.2 and up', () => {
    for (const ratio of [2.2, 2.4, 3]) {
      const s = 0.01;
      const mass = 1000 * s * s;
      const hex = latticeDensity(s, ratio * s, 1000, 'hex') / mass;
      const square = latticeDensity(s, ratio * s, 1000, 'square') / mass;
      // sum_lattice W is ~1 / s^2 times the integral; (1 / s^2) * m is the nominal density.
      expect(hex * s * s).toBeGreaterThan(0.99);
      expect(hex * s * s).toBeLessThan(1.01);
      expect(square * s * s).toBeGreaterThan(0.99);
      expect(square * s * s).toBeLessThan(1.01);
    }
  });

  it('the hexagonal lattice has one particle per spacing squared', () => {
    const { dx, dy } = hexSpacing(0.01);
    expect(dx * dy).toBeCloseTo(0.01 * 0.01, 10);
    const points = blockPoints({ min: [0, 0], max: [1, 1] }, 0.01);
    expect(points.length / 1e4).toBeGreaterThan(0.97);
    expect(points.length / 1e4).toBeLessThan(1.03);
  });

  it('the wall tables match an independent brute-force quadrature over the half-plane', () => {
    const { gamma, force } = sphWallTables();
    expect(gamma.length).toBe(WALL_TABLE_SIZE);
    expect(gamma[0]!).toBeCloseTo(0.5, 6);
    expect(gamma[WALL_TABLE_SIZE - 1]!).toBe(0);
    for (let k = 1; k < WALL_TABLE_SIZE; k++) expect(gamma[k]!).toBeLessThanOrEqual(gamma[k - 1]!);
    // Brute force: sum a fine 2D grid of the kernel over the region x > d (the wall side), h = 1.
    const poly6 = (r2: number) => (r2 >= 1 ? 0 : (4 / Math.PI) * (1 - r2) ** 3);
    const spiky = (r: number) => (r >= 1 ? 0 : (10 / Math.PI) * (1 - r) ** 3);
    const n = 1200;
    const cell = 2 / n;
    for (const k of [0, 4, 8, 16, 24, 31]) {
      const d = k / (WALL_TABLE_SIZE - 1);
      let massBeyond = 0;
      let lineIntegral = 0;
      for (let i = 0; i < n; i++) {
        const x = -1 + (i + 0.5) * cell;
        if (x < d) continue;
        for (let j = 0; j < n; j++) {
          const y = -1 + (j + 0.5) * cell;
          massBeyond += poly6(x * x + y * y) * cell * cell;
        }
      }
      // The force table is the line integral of the spiky kernel along x = d.
      for (let j = 0; j < 4 * n; j++) {
        const y = -1 + (j + 0.5) * (2 / (4 * n));
        lineIntegral += spiky(Math.hypot(d, y)) * (2 / (4 * n));
      }
      expect(gamma[k]!, `gamma(${d})`).toBeCloseTo(massBeyond, 3);
      expect(force[k]!, `force(${d})`).toBeCloseTo(lineIntegral, 3);
    }
  });

  it('obstacleDistance is the signed distance of a circle and a rounded box', () => {
    expect(obstacleDistance({ shape: 'circle', center: [1, 1], radius: 0.5 }, 1, 1.75)).toBeCloseTo(
      0.25,
      6,
    );
    expect(obstacleDistance({ shape: 'circle', center: [1, 1], radius: 0.5 }, 1, 1)).toBeCloseTo(
      -0.5,
      6,
    );
    const box = { shape: 'box', center: [0, 0], halfSize: [1, 0.5], cornerRadius: 0.2 } as const;
    expect(obstacleDistance(box, 1.5, 0)).toBeCloseTo(0.5, 6);
    expect(obstacleDistance(box, 0, 0)).toBeCloseTo(-0.5, 6);
  });

  it('the time step follows the spacing: twice the particles per axis needs twice the sub-steps', () => {
    const bounds = { min: [0, 0], max: [2, 1.2] } as const;
    const p = { ...DEFAULT_SPH_PARAMS };
    const coarse = deriveSph(0.02, 2.4, 1000, bounds, p);
    const fine = deriveSph(0.01, 2.4, 1000, bounds, p);
    expect(fine.dtLimit / coarse.dtLimit).toBeCloseTo(0.5, 6);
    expect(fine.subSteps / coarse.subSteps).toBeGreaterThan(1.7);
    expect(fine.subSteps / coarse.subSteps).toBeLessThan(2.3);
    // dt is exactly a whole number of sub-steps per 1/60 s tick, and never above the CFL limit.
    expect(fine.dt * fine.subSteps).toBeCloseTo(1 / 60, 10);
    expect(fine.dt).toBeLessThanOrEqual(fine.dtLimit + 1e-12);
    // The rest density is the lattice sum, within a percent of the nominal one.
    expect(Math.abs(coarse.restDensity / 1000 - 1)).toBeLessThan(0.01);
    // Capped sub-steps raise dt above the limit (the stability clamps then take over).
    const capped = deriveSph(0.002, 2.4, 1000, bounds, { ...p, maxSubSteps: 10 });
    expect(capped.subSteps).toBe(10);
    expect(capped.dt).toBeGreaterThan(capped.dtLimit);
    // Gravity off still gives a finite sound speed.
    const zero = deriveSph(0.01, 2.4, 1000, bounds, { ...p, gravity: [0, 0] });
    expect(Number.isFinite(zero.soundSpeed) && zero.soundSpeed > 0).toBe(true);
  });
});

// --- The solver ----------------------------------------------------------------------------------

/** Quiet physics for the tests that isolate one thing: no damping, no gravity unless asked. */
const QUIET: Partial<SphParams> = { gravity: [0, 0], damping: 0 };

async function makeSolver(
  options: {
    spacing?: number;
    bounds?: { min: [number, number]; max: [number, number] };
    capacity?: number;
    params?: Partial<SphParams>;
    smoothingRatio?: number;
  } = {},
): Promise<{ device: GPUDevice; sph: SphSolver }> {
  const device = await getTestDevice();
  const sph = createSph({
    device,
    capacity: options.capacity ?? 20_000,
    bounds: options.bounds ?? { min: [0, 0], max: [1, 1] },
    spacing: options.spacing ?? 0.02,
    smoothingRatio: options.smoothingRatio,
    params: options.params,
  });
  return { device, sph };
}

/** Records `frames` updates of `delta` seconds and waits for the GPU. */
async function advance(
  device: GPUDevice,
  sph: SphSolver,
  seconds: number,
  delta = 1 / 60,
): Promise<number> {
  let steps = 0;
  const frames = Math.round(seconds / delta);
  for (let f = 0; f < frames; f++) {
    const encoder = device.createCommandEncoder();
    steps += sph.update(encoder, delta);
    device.queue.submit([encoder.finish()]);
    if (f % 30 === 29) await device.queue.onSubmittedWorkDone();
  }
  await device.queue.onSubmittedWorkDone();
  return steps;
}

/** One step of zero length: refreshes density and pressure for the current positions and moves nothing. */
async function refreshDensity(device: GPUDevice, sph: SphSolver): Promise<SphSnapshot> {
  const encoder = device.createCommandEncoder();
  sph.step(encoder, 1, 0);
  device.queue.submit([encoder.finish()]);
  return sph.read();
}

const sum = (values: ArrayLike<number>): number => {
  let total = 0;
  for (let i = 0; i < values.length; i++) total += values[i]!;
  return total;
};
const mean = (values: ArrayLike<number>): number => sum(values) / values.length;

function kineticEnergy(s: SphSnapshot, mass: number): number {
  let e = 0;
  for (let i = 0; i < s.count; i++) e += 0.5 * mass * (s.vx[i]! ** 2 + s.vy[i]! ** 2);
  return e;
}

describe('SPH solver: rest state', () => {
  it('a block at lattice spacing reads rest density in its interior, and a square lattice within a percent', async () => {
    const { device, sph } = await makeSolver({
      bounds: { min: [0, 0], max: [2, 2] },
      params: QUIET,
    });
    sph.reset({ blocks: [{ min: [0.5, 0.5], max: [1.5, 1.5] }] });
    const s = await expectValid(device, () => refreshDensity(device, sph));
    const { h, restDensity } = sph.derived;
    const interior: number[] = [];
    for (let i = 0; i < s.count; i++) {
      const inside =
        Math.min(s.x[i]! - 0.5, 1.5 - s.x[i]!, s.y[i]! - 0.5, 1.5 - s.y[i]!) > h + 0.01;
      if (inside) interior.push(s.density[i]! / restDensity);
    }
    expect(interior.length).toBeGreaterThan(1000);
    const worst = Math.max(...interior.map((r) => Math.abs(r - 1)));
    // Float32 and the lattice edge of the table: well under a tenth of a percent.
    expect(worst).toBeLessThan(2e-3);
    console.info(
      `interior density error ${worst.toExponential(2)} over ${interior.length} particles`,
    );
    // The pressure there is zero: at rest density the equation of state gives none.
    expect(Math.max(...Array.from(s.pressure))).toBeLessThan(1);

    // The same block on a SQUARE lattice reads a different density (kernel aliasing), within a percent: the
    // reason the rest density is derived from the lattice the particles are spawned on.
    const squarePoints: number[] = [];
    const spacing = sph.derived.spacing;
    for (let i = 0; i < 50; i++) {
      for (let j = 0; j < 50; j++)
        squarePoints.push(0.5 + (i + 0.5) * spacing, 0.5 + (j + 0.5) * spacing, 0, 1);
    }
    sph.reset({ blocks: [], particles: { positions: new Float32Array(squarePoints) } });
    const square = await refreshDensity(device, sph);
    const ratios: number[] = [];
    for (let i = 0; i < square.count; i++) {
      const inside =
        Math.min(square.x[i]! - 0.5, 1.5 - square.x[i]!, square.y[i]! - 0.5, 1.5 - square.y[i]!) >
        h + 0.01;
      if (inside) ratios.push(square.density[i]! / restDensity);
    }
    expect(Math.abs(mean(ratios) - 1)).toBeLessThan(0.01);
  });

  it('a block left alone with no gravity stays put', async () => {
    const { device, sph } = await makeSolver({
      bounds: { min: [0, 0], max: [2, 2] },
      params: QUIET,
    });
    sph.reset({ blocks: [{ min: [0.6, 0.6], max: [1.4, 1.4] }] });
    await advance(device, sph, 0.5);
    const s = await sph.read();
    // The edge particles relax outward a little (the free surface has no pressure holding it); the bulk does not move.
    expect(
      Math.max(...Array.from(s.vx).map(Math.abs), ...Array.from(s.vy).map(Math.abs)),
    ).toBeLessThan(0.1);
  });

  it('the per-particle scalar carries density, pressure or speed as asked', async () => {
    const { device, sph } = await makeSolver({ params: { ...QUIET, scalar: 'density' } });
    sph.reset({ blocks: [{ min: [0.3, 0.3], max: [0.7, 0.7], velocity: [1, 0] }] });
    let s = await refreshDensity(device, sph);
    expect(mean(s.scalar)).toBeGreaterThan(0.85);
    expect(mean(s.scalar)).toBeLessThan(1.05);
    sph.params.scalar = 'speed';
    s = await refreshDensity(device, sph);
    expect(mean(s.scalar)).toBeCloseTo(1, 1);
    sph.params.scalar = 'pressure';
    s = await refreshDensity(device, sph);
    expect(Math.min(...Array.from(s.scalar))).toBeGreaterThanOrEqual(0);
  });
});

describe('SPH solver: conservation', () => {
  it('conserves total linear momentum to round-off with no gravity, damping or walls in reach', async () => {
    const { device, sph } = await makeSolver({
      bounds: { min: [0, 0], max: [4, 4] },
      spacing: 0.03,
      params: { ...QUIET, surfaceTension: 20 },
    });
    sph.reset({ blocks: [{ min: [1.8, 1.8], max: [2.2, 2.2] }] });
    // Random velocities, so the pressure, viscosity, artificial viscosity and cohesion all act.
    const random = makeRandom(3);
    const before = await sph.read();
    const velocities = new Float32Array(before.count * 4);
    const positions = new Float32Array(before.count * 4);
    for (let i = 0; i < before.count; i++) {
      positions.set([before.x[i]!, before.y[i]!, 0, 1], i * 4);
      velocities.set([(random() - 0.5) * 1.2, (random() - 0.5) * 1.2, 0, 0], i * 4);
    }
    sph.reset({ blocks: [], particles: { positions, velocities } });
    const p0 = [0, 0];
    let scale = 0;
    for (let i = 0; i < before.count; i++) {
      p0[0]! += velocities[i * 4]!;
      p0[1]! += velocities[i * 4 + 1]!;
      scale += Math.hypot(velocities[i * 4]!, velocities[i * 4 + 1]!);
    }
    const e0 = kineticEnergy(await sph.read(), 1);
    await expectValid(device, () => advance(device, sph, 0.2));
    const after = await sph.read();
    const p1 = [sum(after.vx), sum(after.vy)];
    // The momentum drift, against the total of |m v| the forces had to push around.
    const drift = Math.hypot(p1[0]! - p0[0]!, p1[1]! - p0[1]!) / scale;
    expect(drift).toBeLessThan(1e-5);
    console.info(
      `momentum drift ${drift.toExponential(2)} of sum |v| over ${after.count} particles, ${sph.stats.steps} sub-steps`,
    );
    // And the forces were not idle: the kinetic energy changed (viscous dissipation, pressure work).
    expect(Math.abs(kineticEnergy(after, 1) / e0 - 1)).toBeGreaterThan(0.05);
  });
});

describe('SPH solver: hydrostatics', () => {
  const WIDTH = 1;
  const COLUMN = 0.5;

  /** Settles a column and returns its statistics. */
  async function settleColumn(
    params: Partial<SphParams>,
    spacing = 0.02,
    seconds = 4,
    delta = 1 / 60,
  ) {
    const { device, sph } = await makeSolver({
      bounds: { min: [0, 0], max: [WIDTH, 1.2] },
      spacing,
      params: { gravityRamp: 0.4, ...params },
    });
    sph.reset({ blocks: [{ min: [0, 0], max: [WIDTH, COLUMN] }] });
    await expectValid(device, () => advance(device, sph, seconds, delta));
    const s = await sph.read();
    return { device, sph, s };
  }

  it('settles to a hydrostatic state: pressure rises linearly with depth at about rho g, height within 3 percent', async () => {
    const { sph, s } = await settleColumn({}, 0.02, 4);
    const d = sph.derived;
    // Pressure against depth below the top, by a least-squares line over the particles away from the surface layer.
    let top = 0;
    for (let i = 0; i < s.count; i++) top = Math.max(top, s.y[i]!);
    const xs: number[] = [];
    const ps: number[] = [];
    for (let i = 0; i < s.count; i++) {
      if (top - s.y[i]! > 3 * d.h) {
        xs.push(s.y[i]!);
        ps.push(s.pressure[i]!);
      }
    }
    const mx = mean(xs);
    const mp = mean(ps);
    let sxy = 0;
    let sxx = 0;
    for (let i = 0; i < xs.length; i++) {
      sxy += (xs[i]! - mx) * (ps[i]! - mp);
      sxx += (xs[i]! - mx) ** 2;
    }
    const slope = -sxy / sxx; // dp / d(depth)
    const expected = d.restDensity * 9.81;
    expect(slope / expected).toBeGreaterThan(0.9);
    expect(slope / expected).toBeLessThan(1.1);
    // Pressure strictly increases with depth: the mean of the bottom quarter exceeds the top quarter, which
    // exceeds the surface layer.
    const band = (lo: number, hi: number) => {
      const v: number[] = [];
      for (let i = 0; i < s.count; i++) if (s.y[i]! >= lo && s.y[i]! < hi) v.push(s.pressure[i]!);
      return mean(v);
    };
    const bandDensity = (lo: number, hi: number) => {
      const v: number[] = [];
      for (let i = 0; i < s.count; i++) if (s.y[i]! >= lo && s.y[i]! < hi) v.push(s.density[i]!);
      return mean(v);
    };
    const bottom = band(0, 0.125);
    const middle = band(0.19, 0.31);
    const upper = band(0.375, 0.5);
    expect(bottom).toBeGreaterThan(middle);
    expect(middle).toBeGreaterThan(upper);
    // The bottom pressure is rho g H (to within the compressible fluid's few percent).
    expect(bottom / (d.restDensity * 9.81 * 0.45)).toBeGreaterThan(0.85);

    // Height: the incompressible height is N s^2 / width. The surface of the particle cloud is where the density
    // has fallen to half the bulk; compare the highest rows' centres plus half a spacing.
    const incompressible = (s.count * d.spacing * d.spacing) / WIDTH;
    const sorted = Array.from(s.y).sort((a, b) => b - a);
    const topRows = mean(sorted.slice(0, Math.floor(WIDTH / d.spacing)));
    const height = topRows + 0.5 * d.spacing * 0.93;
    const error = (height - incompressible) / incompressible;
    expect(Math.abs(error)).toBeLessThan(0.03);
    // Report the compression: the mean density of the settled fluid against rest, and the predicted g H / c^2.
    const compression = mean(s.density) / d.restDensity - 1;
    console.info(
      `column ${s.count} particles, ${d.subSteps} sub-steps/frame: height error ${(error * 100).toFixed(2)}%, mean density ${(compression * 100).toFixed(2)}% of rest, bottom quarter ${(
        (bandDensity(0, 0.125) / d.restDensity - 1) *
        100
      ).toFixed(
        2,
      )}%, predicted g H / c^2 ${(d.hydrostaticCompression * 100).toFixed(2)}% (full domain height)`,
    );
    // Compression stays within a few percent of rest everywhere.
    expect(Math.max(...Array.from(s.density)) / d.restDensity).toBeLessThan(1.05);
  });

  it('the walls hold a column with no crust and no gap: the first layer stands about half a spacing off the floor', async () => {
    const { sph, s } = await settleColumn({}, 0.02, 3);
    const spacing = sph.derived.spacing;
    const floor = Array.from(s.y)
      .sort((a, b) => a - b)
      .slice(0, Math.floor(WIDTH / spacing));
    const gap = mean(floor);
    expect(gap).toBeGreaterThan(0.3 * spacing);
    expect(gap).toBeLessThan(0.75 * spacing);
    // No dense crust at the floor: the density of the bottom layer is within 3 percent of the layer above it.
    let low = 0;
    let lowN = 0;
    let above = 0;
    let aboveN = 0;
    for (let i = 0; i < s.count; i++) {
      if (s.x[i]! < 0.2 || s.x[i]! > 0.8) continue;
      if (s.y[i]! < 1.2 * spacing) {
        low += s.density[i]!;
        lowN++;
      } else if (s.y[i]! < 3 * spacing) {
        above += s.density[i]!;
        aboveN++;
      }
    }
    expect(Math.abs(low / lowN / (above / aboveN) - 1)).toBeLessThan(0.03);
    // Nor a lattice layering at the side wall: the density of the particles next to it matches the interior.
    let wall = 0;
    let wallN = 0;
    let inner = 0;
    let innerN = 0;
    for (let i = 0; i < s.count; i++) {
      if (s.y[i]! < 0.1 || s.y[i]! > 0.3) continue;
      if (s.x[i]! < 1.2 * spacing) {
        wall += s.density[i]!;
        wallN++;
      } else if (s.x[i]! > 0.3 && s.x[i]! < 0.7) {
        inner += s.density[i]!;
        innerN++;
      }
    }
    expect(Math.abs(wall / wallN / (inner / innerN) - 1)).toBeLessThan(0.04);
  });

  it('settles calmly: the residual speed is small against the free-fall speed', async () => {
    const { sph, s } = await settleColumn({}, 0.02, 5);
    const speeds = Array.from(s.vx, (vx, i) => Math.hypot(vx, s.vy[i]!));
    const rms = Math.sqrt(mean(speeds.map((v) => v * v)));
    // Against the speed particles reach falling their own spacing: sqrt(2 g s) = 0.63 m/s at s = 0.02.
    expect(rms).toBeLessThan(0.05);
    console.info(
      `residual speed after 5 s: rms ${rms.toFixed(4)} m/s, max ${Math.max(...speeds).toFixed(3)} m/s (spacing ${sph.derived.spacing}, free fall over a spacing ${Math.sqrt(2 * 9.81 * sph.derived.spacing).toFixed(2)} m/s)`,
    );
  });

  it('60 Hz and 120 Hz stepping settle to the same height and energy', async () => {
    const a = await settleColumn({}, 0.025, 3, 1 / 60);
    const b = await settleColumn({}, 0.025, 3, 1 / 120);
    const heightOf = (s: SphSnapshot) =>
      mean(
        Array.from(s.y)
          .sort((x, y) => y - x)
          .slice(0, 40),
      );
    expect(Math.abs(heightOf(a.s) / heightOf(b.s) - 1)).toBeLessThan(0.01);
    expect(Math.abs(mean(a.s.y) / mean(b.s.y) - 1)).toBeLessThan(0.01);
    // Both ran the same number of simulated sub-steps (the fixed step does not depend on the frame rate).
    expect(Math.abs(a.sph.stats.steps / b.sph.stats.steps - 1)).toBeLessThan(0.01);
    // Residual energy: both are quiet, within the same small number.
    const ea = kineticEnergy(a.s, a.sph.derived.mass) / a.s.count;
    const eb = kineticEnergy(b.s, b.sph.derived.mass) / b.s.count;
    expect(ea).toBeLessThan(0.05);
    expect(eb).toBeLessThan(0.05);
  });

  it('the grid skin keeps the search exact: rebuilding every 8 steps gives the density a brute-force search gives', async () => {
    const { device, sph } = await makeSolver({
      bounds: { min: [0, 0], max: [2, 1] },
      params: { rebuildEvery: 8, gravityRamp: 0 },
    });
    sph.reset({ blocks: [{ min: [0, 0], max: [0.7, 0.8] }] });
    await advance(device, sph, 0.4);
    // A zero-length step on the grid as the 8-step schedule left it, then a CPU brute force on those positions.
    const s = await refreshDensity(device, sph);
    const { h, mass } = sph.derived;
    let checked = 0;
    let worst = 0;
    for (let i = 0; i < s.count; i += 7) {
      let sumW = 0;
      for (let j = 0; j < s.count; j++) {
        const r2 = (s.x[i]! - s.x[j]!) ** 2 + (s.y[i]! - s.y[j]!) ** 2;
        if (r2 < h * h) sumW += (4 / (Math.PI * h ** 8)) * (h * h - r2) ** 3;
      }
      const rho = mass * sumW;
      // The wall term is zero away from the walls; compare only there.
      if (Math.min(s.x[i]!, 2 - s.x[i]!, s.y[i]!, 1 - s.y[i]!) < h) continue;
      worst = Math.max(worst, Math.abs(rho - s.density[i]!) / rho);
      checked++;
    }
    expect(checked).toBeGreaterThan(50);
    expect(worst).toBeLessThan(1e-4);
    expect(sph.stats.builds).toBeLessThan(sph.stats.steps / 4);
  });
});

describe('SPH solver: boundaries and robustness', () => {
  const inside = (s: SphSnapshot, min: [number, number], max: [number, number]) => {
    for (let i = 0; i < s.count; i++) {
      if (!(s.x[i]! >= min[0] && s.x[i]! <= max[0] && s.y[i]! >= min[1] && s.y[i]! <= max[1])) {
        return false;
      }
    }
    return true;
  };

  it('no particle ever leaves the box, even under a violent pointer impulse and a fast fall', async () => {
    const bounds = { min: [0, 0] as [number, number], max: [1, 1] as [number, number] };
    const { device, sph } = await makeSolver({ bounds, params: { gravity: [30, -90] } });
    sph.reset({ blocks: [{ min: [0.2, 0.2], max: [0.8, 0.8], velocity: [20, 20] }] });
    for (let f = 0; f < 60; f++) {
      sph.params.pointers = [
        {
          x: 0.5,
          y: 0.5,
          vx: 5000 * Math.sin(f),
          vy: -5000,
          radius: 0.4,
          strength: 1e5,
          mode: 'push',
        },
        { x: 0.3, y: 0.7, vx: 0, vy: 0, radius: 0.5, strength: 1e5, mode: 'pull' },
      ];
      const encoder = device.createCommandEncoder();
      sph.update(encoder, 1 / 60);
      device.queue.submit([encoder.finish()]);
    }
    await device.queue.onSubmittedWorkDone();
    const s = await sph.read();
    expect(inside(s, bounds.min, bounds.max)).toBe(true);
    expect(Array.from(s.vx).every(Number.isFinite)).toBe(true);
    expect(Math.max(...Array.from(s.vx, (vx, i) => Math.hypot(vx, s.vy[i]!)))).toBeLessThanOrEqual(
      sph.derived.speedMax * 1.001,
    );
  });

  it('kinetic energy decays in a closed box', async () => {
    const { device, sph } = await makeSolver({
      bounds: { min: [0, 0], max: [1, 1] },
      params: { gravity: [0, 0], restitution: 0.9, friction: 0 },
    });
    sph.reset({ blocks: [{ min: [0.1, 0.1], max: [0.9, 0.9], velocity: [1.5, 1] }] });
    const mass = sph.derived.mass;
    const e0 = kineticEnergy(await sph.read(), mass);
    await advance(device, sph, 0.5);
    const e1 = kineticEnergy(await sph.read(), mass);
    await advance(device, sph, 2);
    const e2 = kineticEnergy(await sph.read(), mass);
    expect(e1).toBeLessThan(e0);
    expect(e2).toBeLessThan(0.5 * e1);
  });

  it('viscosity damps a shear flow, and more viscosity damps it faster', async () => {
    const remaining = async (viscosity: number) => {
      const { device, sph } = await makeSolver({
        bounds: { min: [0, 0], max: [3, 3] },
        params: { gravity: [0, 0], damping: 0, viscosity, artificialViscosity: 0 },
      });
      sph.reset({ blocks: [{ min: [1, 1], max: [2, 2] }] });
      const before = await sph.read();
      const positions = new Float32Array(before.count * 4);
      const velocities = new Float32Array(before.count * 4);
      for (let i = 0; i < before.count; i++) {
        positions.set([before.x[i]!, before.y[i]!, 0, 1], i * 4);
        velocities.set([4 * (before.y[i]! - 1.5), 0, 0, 0], i * 4);
      }
      sph.reset({ blocks: [], particles: { positions, velocities } });
      const e0 = kineticEnergy(await sph.read(), sph.derived.mass);
      await advance(device, sph, 0.25);
      return kineticEnergy(await sph.read(), sph.derived.mass) / e0;
    };
    const none = await remaining(0);
    const low = await remaining(0.02);
    const high = await remaining(0.3);
    // Without viscosity the shear (not an equilibrium: pressure acts on the free edges) loses little; with it, a lot.
    expect(low).toBeLessThan(none);
    expect(high).toBeLessThan(low);
    expect(high).toBeLessThan(0.8);
  });

  it('survives parameter extremes without a NaN or an escaped particle', async () => {
    const bounds = { min: [0, 0] as [number, number], max: [1, 1] as [number, number] };
    const extremes: [string, Partial<SphParams>][] = [
      ['stiffness 1', { stiffness: 1 }],
      ['stiffness 200', { stiffness: 200 }],
      ['no viscosity at all', { viscosity: 0, artificialViscosity: 0, damping: 0 }],
      ['viscosity 100', { viscosity: 100, artificialViscosity: 1 }],
      ['gravity 1e4', { gravity: [0, -1e4] }],
      ['one sub-step per tick', { subSteps: 1, maxSubSteps: 1 }],
      ['courant 50', { courant: 50, subSteps: 0, maxSubSteps: 2 }],
      ['surface tension 1e4', { surfaceTension: 1e4 }],
      ['restitution 1, no friction', { restitution: 1, friction: 0 }],
      ['friction 1, restitution 0', { restitution: 0, friction: 1 }],
      ['rebuild every step', { rebuildEvery: 1 }],
    ];
    for (const [label, params] of extremes) {
      const { device, sph } = await makeSolver({ bounds, spacing: 0.04, params });
      sph.reset({ blocks: [{ min: [0.1, 0.1], max: [0.9, 0.6], velocity: [3, -3] }] });
      await expectValid(device, () => advance(device, sph, 0.6));
      const s = await sph.read();
      expect(s.count, label).toBeGreaterThan(100);
      for (const array of [s.x, s.y, s.vx, s.vy, s.density, s.pressure, s.scalar]) {
        expect(Array.from(array).every(Number.isFinite), label).toBe(true);
      }
      expect(inside(s, bounds.min, bounds.max), label).toBe(true);
      sph.destroy();
    }
    // Absurd geometry too: a smoothing ratio at both ends, and a domain much bigger than the fluid.
    for (const ratio of [1.2, 6]) {
      const { device, sph } = await makeSolver({ spacing: 0.04, smoothingRatio: ratio });
      sph.reset({ blocks: [{ min: [0.1, 0.1], max: [0.9, 0.6] }] });
      await expectValid(device, () => advance(device, sph, 0.3));
      const s = await sph.read();
      expect(Array.from(s.density).every(Number.isFinite), `ratio ${ratio}`).toBe(true);
      expect(inside(s, [0, 0], [1, 1]), `ratio ${ratio}`).toBe(true);
    }
  });

  it('keeps fluid out of an obstacle, and stands off it like a wall', async () => {
    const circle = { shape: 'circle', center: [0.5, 0.25], radius: 0.18 } as const;
    const { device, sph } = await makeSolver({
      bounds: { min: [0, 0], max: [1, 1] },
      params: { obstacles: [circle] },
      spacing: 0.02,
    });
    sph.reset({ blocks: [{ min: [0, 0], max: [1, 0.55] }] });
    // The block leaves the obstacle's footprint empty.
    const fresh = await sph.read();
    expect(fresh.count).toBeGreaterThan(1000);
    for (let i = 0; i < fresh.count; i++) {
      expect(obstacleDistance(circle, fresh.x[i]!, fresh.y[i]!)).toBeGreaterThan(0);
    }
    await expectValid(device, () => advance(device, sph, 2));
    const s = await sph.read();
    let nearest = Infinity;
    for (let i = 0; i < s.count; i++) {
      nearest = Math.min(nearest, obstacleDistance(circle, s.x[i]!, s.y[i]!));
    }
    expect(nearest).toBeGreaterThanOrEqual(-1e-4);
    // The ring of particles next to the obstacle has no dense crust either.
    const density: number[] = [];
    const bulk: number[] = [];
    for (let i = 0; i < s.count; i++) {
      const d = obstacleDistance(circle, s.x[i]!, s.y[i]!);
      if (s.y[i]! > 0.4) continue;
      if (d < 1.3 * sph.derived.spacing) density.push(s.density[i]!);
      else if (s.x[i]! < 0.25 && s.y[i]! < 0.25) bulk.push(s.density[i]!);
    }
    expect(density.length).toBeGreaterThan(20);
    expect(Math.abs(mean(density) / mean(bulk) - 1)).toBeLessThan(0.06);
  });
});

describe('SPH solver: scenes, emitter, capacity', () => {
  it('reset replaces the fluid, a block adds to it, and both can be replayed', async () => {
    const { device, sph } = await makeSolver();
    sph.reset({ blocks: [{ min: [0, 0], max: [0.5, 0.5] }] });
    await expectValid(device, () => advance(device, sph, 0.05));
    const first = sph.count;
    expect(first).toBeGreaterThan(500);
    sph.addBlock({ center: [0.7, 0.7], radius: 0.1, velocity: [0, -1] });
    await advance(device, sph, 0.05);
    expect(sph.count).toBeGreaterThan(first + 50);
    sph.reset();
    await advance(device, sph, 0.02);
    expect(sph.count).toBe(first);
    expect(sph.time).toBeLessThan(0.05);
    sph.reset({ blocks: [] });
    await advance(device, sph, 0.02);
    expect(sph.count).toBe(0);
  });

  it('an emitter pours a stream at rest density along it, and stops at capacity', async () => {
    const { device, sph } = await makeSolver({
      bounds: { min: [0, 0], max: [1, 1.5] },
      capacity: 3000,
      params: { gravity: [0, 0], damping: 0 },
    });
    sph.reset({
      blocks: [],
      emitter: { position: [0.5, 1.2], direction: [0, -1], width: 0.2, speed: 1 },
    });
    await expectValid(device, () => advance(device, sph, 0.5));
    const s = await sph.read();
    // 0.5 s at 1 m/s through a 0.2 m nozzle is 0.1 m^2 of fluid: 0.1 / s^2 particles.
    const expected = 0.1 / (sph.derived.spacing * sph.derived.spacing);
    expect(s.count).toBeGreaterThan(expected * 0.8);
    expect(s.count).toBeLessThan(expected * 1.2);
    const heads = Array.from(s.y).filter((y) => y < 1.2 - 0.2);
    expect(heads.length).toBeGreaterThan(100);
    await advance(device, sph, 3);
    expect(sph.count).toBeLessThanOrEqual(3000);
    sph.setEmitter(null);
    const n = sph.count;
    await advance(device, sph, 0.2);
    expect(sph.count).toBe(n);
  });

  it('changing the capacity keeps the fluid and bumps the generation; the ParticleSet view follows', async () => {
    const { device, sph } = await makeSolver({ capacity: 2000 });
    sph.reset({ blocks: [{ min: [0, 0], max: [0.4, 0.4] }] });
    await advance(device, sph, 0.1);
    const n = sph.count;
    expect(n).toBeGreaterThan(300);
    const generation = sph.generation;
    const buffer = sph.particles.positions;
    sph.setCapacity(5000);
    await expectValid(device, () => advance(device, sph, 0.1));
    expect(sph.capacity).toBe(5000);
    expect(sph.particles.capacity).toBe(5000);
    expect(sph.generation).toBeGreaterThan(generation);
    expect(sph.particles.positions).not.toBe(buffer);
    expect(sph.count).toBe(n);
    // Shrinking below the count keeps what fits.
    sph.setCapacity(200);
    await advance(device, sph, 0.05);
    expect(sph.count).toBeLessThanOrEqual(200);
    const view = sph.particles;
    expect(view.restVolume).toBeCloseTo(sph.derived.spacing ** 2, 10);
    expect(view.radius).toBeCloseTo(0.5 * sph.derived.spacing, 10);
    const s = await sph.read();
    expect(Array.from(s.density).every(Number.isFinite)).toBe(true);
  });

  it('a pointer drags the fluid along: the pushed fluid ends up moving with it', async () => {
    const { device, sph } = await makeSolver({
      bounds: { min: [0, 0], max: [1, 1] },
      params: { gravity: [0, 0] },
    });
    sph.reset({ blocks: [{ min: [0.2, 0.2], max: [0.8, 0.8] }] });
    sph.params.pointers = [
      { x: 0.5, y: 0.5, vx: 2, vy: 0, radius: 0.15, strength: 40, mode: 'push' },
    ];
    await advance(device, sph, 0.2);
    const s = await sph.read();
    let near = 0;
    let n = 0;
    for (let i = 0; i < s.count; i++) {
      if (Math.hypot(s.x[i]! - 0.5, s.y[i]! - 0.5) < 0.1) {
        near += s.vx[i]!;
        n++;
      }
    }
    expect(near / n).toBeGreaterThan(0.5);
    // A pull gathers the fluid toward the pointer.
    sph.reset({ blocks: [{ min: [0.2, 0.2], max: [0.8, 0.8] }] });
    sph.params.pointers = [
      { x: 0.3, y: 0.5, vx: 0, vy: 0, radius: 0.5, strength: 60, mode: 'pull' },
    ];
    await advance(device, sph, 0.3);
    const pulled = await sph.read();
    expect(mean(pulled.x)).toBeLessThan(0.49);
  });

  it('every pass is labelled and produces no validation error with a profiler asking for timestamps', async () => {
    const device = await getTestDevice();
    const labels: string[] = [];
    const sph = createSph({
      device,
      capacity: 2000,
      bounds: { min: [0, 0], max: [1, 1] },
      spacing: 0.03,
      profiler: {
        pass(label) {
          labels.push(label);
          return undefined;
        },
      },
    });
    sph.reset({ blocks: [{ min: [0, 0], max: [0.5, 0.5] }] });
    await expectValid(device, () => advance(device, sph, 0.1));
    expect(labels).toEqual(expect.arrayContaining(['sph density', 'sph forces']));
    expect(labels.some((l) => l.startsWith('sph grid'))).toBe(true);
    // Only the first and last sub-step of an update ask the profiler: far fewer than every pass.
    expect(labels.length).toBeLessThan(sph.stats.steps * 2);
    sph.destroy();
    sph.destroy();
  });
});

// --- Cost study --------------------------------------------------------------------------------------
//
// Story 7.9 (the scale study) collects these. Skipped unless SPH_STUDY names the JSON file to write; a markdown table is
// printed as well:
//
//     SPH_STUDY=/tmp/sph-study.json SPH_STUDY_NOTE="what else was running" npx vitest run tests/sph.test.ts -t study
//
// Method (docs/effect-authoring.md, "Reading the HUD honestly"): every figure is the SLOPE of the GPU span between a
// small and a large number of repetitions of the workload recorded into ONE command buffer, so the ~1 ms floor of a pass
// and the submit latency drop out; the span is first-pass start to last-pass end (two timestamps however many passes ran;
// the solver only asks the profiler about the first and last sub-step of an update, which are exactly those two). Small and
// large runs alternate; the figure is the slope between the QUIETEST of each (a busy GPU only adds time), with the slope
// between the MEDIAN runs and the spread (quartiles of the per-pair slopes) alongside, because the machine is shared.
//
// Scene: the dam break of the demo (a 0.55 x 0.95 m column in a 2 x 1.2 m tank), developed for 0.4 s so the particles are
// in their working arrangement; the spacing follows the count.

const studyFile = (globalThis as { process?: { env: Record<string, string | undefined> } }).process
  ?.env.SPH_STUDY;
const studyNote =
  (globalThis as { process?: { env: Record<string, string | undefined> } }).process?.env
    .SPH_STUDY_NOTE ?? '';

describe.skipIf(!studyFile)('study', () => {
  vi.setConfig({ testTimeout: 3_600_000 });

  async function makeTimer() {
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter?.features.has('timestamp-query'))
      throw new Error('timestamp-query is not available');
    const device = await adapter.requestDevice({ requiredFeatures: ['timestamp-query'] });
    const querySet = device.createQuerySet({ type: 'timestamp', count: 4 });
    const resolve = device.createBuffer({
      size: 32,
      usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC,
    });
    const read = device.createBuffer({
      size: 32,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    // Only TWO timestamps matter: the begin of the very first pass of the very first repetition (slot 0) and the end of the
    // last pass of the last repetition (slot 1), so no per-pass timestamp overhead is measured. The workload tells us which
    // repetition it is in (`phase`) and which label its last pass has; Dawn on Metal misbehaves when a pass is given only one
    // of its two timestamps, so the unwanted halves go to slots 2 and 3.
    let active = false;
    let phase: 'first' | 'mid' | 'last' | 'both' = 'mid';
    let firstDone = false;
    let lastLabel = '';
    const profiler = {
      pass(label: string) {
        if (!active) return undefined;
        const wantFirst = (phase === 'first' || phase === 'both') && !firstDone;
        const matchLast = (phase === 'last' || phase === 'both') && label.includes(lastLabel);
        if (wantFirst) {
          firstDone = true;
          return { querySet, beginningOfPassWriteIndex: 0, endOfPassWriteIndex: matchLast ? 1 : 2 };
        }
        if (matchLast) return { querySet, beginningOfPassWriteIndex: 3, endOfPassWriteIndex: 1 };
        return undefined;
      },
    };
    /** GPU milliseconds from the first pass of the first repetition starting to the last pass of the last ending. */
    async function time(
      work: (encoder: GPUCommandEncoder) => void,
      n: number,
      last: string,
    ): Promise<number> {
      lastLabel = last;
      firstDone = false;
      device.pushErrorScope('validation');
      const encoder = device.createCommandEncoder();
      active = true;
      for (let i = 0; i < n; i++) {
        phase = n === 1 ? 'both' : i === 0 ? 'first' : i === n - 1 ? 'last' : 'mid';
        work(encoder);
      }
      active = false;
      encoder.resolveQuerySet(querySet, 0, 4, resolve, 0);
      encoder.copyBufferToBuffer(resolve, 0, read, 0, 32);
      device.queue.submit([encoder.finish()]);
      const problem = await device.popErrorScope();
      if (problem) throw new Error(`validation error while timing: ${problem.message}`);
      await read.mapAsync(GPUMapMode.READ);
      const stamps = new BigUint64Array(read.getMappedRange().slice(0, 32));
      read.unmap();
      return Number(stamps[1]! - stamps[0]!) / 1e6;
    }
    return { device, profiler, time, adapterInfo: adapter.info };
  }
  type Timer = Awaited<ReturnType<typeof makeTimer>>;

  interface Stat {
    best: number;
    median: number;
    /** First and third quartile of the per-pair slopes. */
    q1: number;
    q3: number;
  }

  /**
   * The cost of one repetition of `work`: the slope of the span between `lo` and `hi` repetitions, over `rounds`
   * alternating pairs. `last` is (part of) the profiler label of the last pass the work records.
   */
  async function slope(
    timer: Timer,
    work: (encoder: GPUCommandEncoder) => void,
    last: string,
    options: { rounds?: number; lo?: number; hi?: number } = {},
  ): Promise<Stat> {
    const rounds = options.rounds ?? 24;
    let lo = options.lo;
    let hi = options.hi;
    if (lo === undefined || hi === undefined) {
      let pilot = Infinity;
      for (let i = 0; i < 4; i++) pilot = Math.min(pilot, (await timer.time(work, 4, last)) / 4);
      hi = Math.min(120, Math.max(8, Math.round(3 / Math.max(pilot, 0.005))));
      lo = Math.max(1, Math.round(hi / 4));
    }
    const small: number[] = [];
    const large: number[] = [];
    for (let round = 0; round < rounds; round++) {
      small.push(await timer.time(work, lo, last));
      large.push(await timer.time(work, hi, last));
    }
    const per1 = small.map((s, i) => (large[i]! - s) / (hi! - lo!)).sort((a, b) => a - b);
    small.sort((a, b) => a - b);
    large.sort((a, b) => a - b);
    const m = Math.floor(rounds / 2);
    return {
      best: (large[0]! - small[0]!) / (hi - lo),
      median: (large[m]! - small[m]!) / (hi - lo),
      q1: per1[Math.floor(rounds / 4)]!,
      q3: per1[Math.floor((3 * rounds) / 4)]!,
    };
  }

  const fmt = (s: Stat, digits = 3) =>
    `${s.best.toFixed(digits)} (med ${s.median.toFixed(digits)})`;

  it('per-step cost against particle count, and per-frame cost at the sub-steps each count needs', async () => {
    const timer = await makeTimer();
    const { device } = timer;
    const bounds = { min: [0, 0] as [number, number], max: [2, 1.2] as [number, number] };
    const rows: Record<string, unknown>[] = [];
    const specifier = 'node:fs';
    const { writeFileSync } = (await import(/* @vite-ignore */ specifier)) as {
      writeFileSync(path: string, text: string): void;
    };
    for (const n of [5000, 20_000, 50_000, 100_000, 250_000]) {
      const spacing = Math.sqrt((0.55 * 0.95) / n);
      const sph = createSph({
        device,
        capacity: Math.ceil(n * 1.1),
        bounds,
        spacing,
        profiler: timer.profiler,
        // Uncapped: the sub-step count each resolution actually needs.
        params: { maxSubSteps: 1000 },
      });
      sph.reset({ blocks: [{ min: [0, 0], max: [0.55, 0.95] }] });
      // Develop the flow for 0.4 s of simulated time (a coarse warm-up: the study is about cost, not accuracy).
      for (let f = 0; f < 24; f++) {
        const encoder = device.createCommandEncoder();
        sph.update(encoder, 1 / 60);
        device.queue.submit([encoder.finish()]);
        if (f % 4 === 3) await device.queue.onSubmittedWorkDone();
      }
      await device.queue.onSubmittedWorkDone();
      const d = sph.derived;
      const heavy = n >= 100_000;
      const opts = { rounds: heavy ? 10 : 40 };

      const grid = await slope(
        timer,
        (e) => sph.step(e, 1, d.dt, { grid: true, density: false, forces: false }),
        'reorder',
        opts,
      );
      const density = await slope(
        timer,
        (e) => sph.step(e, 1, d.dt, { grid: false, density: true, forces: false }),
        'density',
        opts,
      );
      // The force pass swaps the position buffers, so repeated steps alternate; both halves are in cell order.
      const forces = await slope(
        timer,
        (e) => sph.step(e, 1, d.dt, { grid: false, density: false, forces: true }),
        'forces',
        opts,
      );
      const step = await slope(timer, (e) => sph.step(e, 1, d.dt, { grid: true }), 'forces', opts);

      // A frame is `subSteps` steps on the solver's own schedule (one rebuild every `rebuildEvery`).
      const frame = await slope(timer, (e) => sph.update(e, 1 / 60), 'forces', {
        rounds: heavy ? 8 : 20,
        lo: 1,
        hi: n >= 100_000 ? 2 : 3,
      });
      // The same frame with the grid rebuilt on every step, to show what the skin buys.
      sph.params.rebuildEvery = 1;
      const frameEveryStep = await slope(timer, (e) => sph.update(e, 1 / 60), 'forces', {
        rounds: heavy ? 8 : 20,
        lo: 1,
        hi: n >= 100_000 ? 2 : 3,
      });
      sph.params.rebuildEvery = 0;
      const row = {
        particles: sph.count,
        spacing,
        h: d.h,
        subStepsPerFrame: d.subSteps,
        dtMs: d.dt * 1000,
        rebuildEvery: d.rebuildEvery,
        gridMs: grid,
        densityMs: density,
        forcesMs: forces,
        stepMs: step,
        frameMs: frame,
        frameRebuildEveryStepMs: frameEveryStep,
      };
      rows.push(row);
      console.info(
        `${sph.count} particles: ${d.subSteps} sub-steps/frame: grid ${fmt(grid)} density ${fmt(density)} forces ${fmt(forces)} step ${fmt(step)} frame ${fmt(frame, 2)} (rebuild every step ${fmt(frameEveryStep, 2)})`,
      );
      sph.destroy();
      writeFileSync(
        studyFile!,
        JSON.stringify(
          { note: studyNote, device: timer.adapterInfo?.description ?? '', rows },
          null,
          2,
        ),
      );
    }
    const header =
      '| particles | sub-steps/frame | rebuild every | grid | density | forces | step (rebuild) | frame | frame (rebuild every step) |';
    const line = (r: Record<string, unknown>) => {
      const s = (key: string, digits = 3) => {
        const v = r[key] as Stat;
        return `${v.best.toFixed(digits)} (${v.q1.toFixed(digits)}-${v.q3.toFixed(digits)})`;
      };
      return `| ${r.particles} | ${r.subStepsPerFrame} | ${r.rebuildEvery} | ${s('gridMs')} | ${s('densityMs')} | ${s('forcesMs')} | ${s('stepMs')} | ${s('frameMs', 2)} | ${s('frameRebuildEveryStepMs', 2)} |`;
    };
    console.info(
      [
        '',
        header,
        '|---|---|---|---|---|---|---|---|---|',
        ...rows.map(line),
        '',
        `ms, quietest-run slope (quartiles of the per-pair slopes). ${studyNote}`,
      ].join('\n'),
    );
  });
});
