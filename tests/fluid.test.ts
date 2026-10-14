import { describe, expect, it, vi } from 'vitest';
import {
  MAX_SPLATS,
  computeFluidGrid,
  createFluid,
  floatToHalf,
  halfToFloat,
  schemePassCount,
  sorOmega,
  type Fluid,
  type FluidParams,
  type FluidScheme,
} from '../src/gpu/sim/fluid.ts';
import { createFluidStirrer, resamplePath, type FluidStroke } from '../src/gpu/sim/fluid-stir.ts';
import type { TrackedPointer } from '../src/gpu/pointer.ts';
import { checkShader, formatDiagnostic } from '../src/gpu/shader.ts';
import { expectValid, getTestDevice, readTexture } from './gpu.ts';
import { scalars } from './wgsl-eval.ts';

vi.setConfig({ testTimeout: 120_000 });

// --- Helpers -------------------------------------------------------------------------------------

const SCHEMES: FluidScheme[] = ['semi-lagrangian', 'maccormack', 'bfecc'];

interface Velocity {
  /** u at face (i, j), i = 0..W, j = 0..H-1; v at face (i, j), i = 0..W-1, j = 0..H. */
  u(i: number, j: number): number;
  v(i: number, j: number): number;
  width: number;
  height: number;
  /** Raw texels, (W + 1) x (H + 1) x 4. */
  data: Float32Array;
}

async function readHalf(device: GPUDevice, texture: GPUTexture): Promise<Float32Array> {
  const bytes = await readTexture(device, texture, 8);
  const half = new Uint16Array(bytes.buffer, bytes.byteOffset, texture.width * texture.height * 4);
  const out = new Float32Array(half.length);
  for (let i = 0; i < half.length; i++) out[i] = halfToFloat(half[i]!);
  return out;
}

async function readVelocity(device: GPUDevice, fluid: Fluid): Promise<Velocity> {
  const { width, height, velocityWidth } = fluid.grid;
  const data = await readHalf(device, fluid.velocity.texture);
  return {
    width,
    height,
    data,
    u: (i, j) => data[(j * velocityWidth + i) * 4]!,
    v: (i, j) => data[(j * velocityWidth + i) * 4 + 1]!,
  };
}

interface Dye {
  width: number;
  height: number;
  data: Float32Array;
  /** Channel c of the texel (x, y). */
  at(x: number, y: number, c?: number): number;
}

async function readDye(device: GPUDevice, fluid: Fluid): Promise<Dye> {
  const { dyeWidth: width, dyeHeight: height } = fluid.grid;
  const data = await readHalf(device, fluid.dye.texture);
  return { width, height, data, at: (x, y, c = 0) => data[(y * width + x) * 4 + c]! };
}

async function submit(device: GPUDevice, record: (encoder: GPUCommandEncoder) => void) {
  const encoder = device.createCommandEncoder();
  record(encoder);
  device.queue.submit([encoder.finish()]);
  await device.queue.onSubmittedWorkDone();
}

function create(
  device: GPUDevice,
  params: Partial<FluidParams> = {},
  resolution = 64,
  dyeResolution = 128,
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

/** Root-mean-square divergence over the cells, and the largest, of a velocity readback. */
function divergence(v: Velocity): { l2: number; max: number } {
  let sum = 0;
  let max = 0;
  for (let j = 0; j < v.height; j++) {
    for (let i = 0; i < v.width; i++) {
      const d = v.u(i + 1, j) - v.u(i, j) + v.v(i, j + 1) - v.v(i, j);
      sum += d * d;
      max = Math.max(max, Math.abs(d));
    }
  }
  return { l2: Math.sqrt(sum / (v.width * v.height)), max };
}

/** Sum of u^2 + v^2 over all real faces: proportional to the kinetic energy. */
function energy(v: Velocity): number {
  let sum = 0;
  for (let j = 0; j < v.height; j++) for (let i = 0; i <= v.width; i++) sum += v.u(i, j) ** 2;
  for (let j = 0; j <= v.height; j++) for (let i = 0; i < v.width; i++) sum += v.v(i, j) ** 2;
  return sum;
}

function allFinite(data: ArrayLike<number>): boolean {
  for (let i = 0; i < data.length; i++) if (!Number.isFinite(data[i])) return false;
  return true;
}

const maxAbs = (data: ArrayLike<number>) => {
  let m = 0;
  for (let i = 0; i < data.length; i++) m = Math.max(m, Math.abs(data[i]!));
  return m;
};

type Rgba = [number, number, number, number];

const gaussian =
  (cx: number, cy: number, sigma: number, peak = 1) =>
  (x: number, y: number): Rgba => {
    const v = peak * Math.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * sigma ** 2));
    return [v, v, v, v];
  };

interface BlobStats {
  total: number;
  peak: number;
  min: number;
  /** Centroid in uv. */
  cx: number;
  cy: number;
  /** Radius of gyration about the centroid divided by sqrt 2, in uv (the sigma of a Gaussian). */
  sigma: number;
}

function blobStats(dye: Dye, channel = 0): BlobStats {
  let total = 0;
  let peak = -Infinity;
  let min = Infinity;
  let mx = 0;
  let my = 0;
  for (let y = 0; y < dye.height; y++) {
    for (let x = 0; x < dye.width; x++) {
      const v = dye.at(x, y, channel);
      total += v;
      peak = Math.max(peak, v);
      min = Math.min(min, v);
      mx += v * (x + 0.5);
      my += v * (y + 0.5);
    }
  }
  const cx = mx / total / dye.width;
  const cy = my / total / dye.height;
  let second = 0;
  for (let y = 0; y < dye.height; y++) {
    for (let x = 0; x < dye.width; x++) {
      const v = dye.at(x, y, channel);
      second += v * (((x + 0.5) / dye.width - cx) ** 2 + ((y + 0.5) / dye.height - cy) ** 2);
    }
  }
  return { total, peak, min, cx, cy, sigma: Math.sqrt(second / total / 2) };
}

/** A rigid rotation about (0.5, 0.5) of `turnsPerSecond` turns per second, in image heights per second. */
const rotation =
  (turnsPerSecond: number) =>
  (x: number, y: number): [number, number] => [
    -(y - 0.5) * 2 * Math.PI * turnsPerSecond,
    (x - 0.5) * 2 * Math.PI * turnsPerSecond,
  ];

/** A divergence-free Gaussian vortex from the stream function psi (velocity = (-dpsi/dy, dpsi/dx)). */
function vortex(amplitude = 0.02): (x: number, y: number) => [number, number] {
  const psi = (x: number, y: number) =>
    amplitude * Math.exp(-((x - 0.5) ** 2 + (y - 0.5) ** 2) / 0.004);
  const e = 1e-4;
  return (x, y) => [
    -(psi(x, y + e) - psi(x, y - e)) / (2 * e),
    (psi(x + e, y) - psi(x - e, y)) / (2 * e),
  ];
}

// --- Pure functions ----------------------------------------------------------------------------------

describe('half floats and grids', () => {
  it('round-trips half floats and keeps their special values', () => {
    for (const x of [0, 1, -1, 0.5, 3.140625, 1024, 65504, -0.0009765625, 6.1e-5]) {
      expect(halfToFloat(floatToHalf(x))).toBeCloseTo(x, 3);
    }
    expect(halfToFloat(floatToHalf(1e9))).toBe(Infinity);
    expect(Number.isNaN(halfToFloat(floatToHalf(Number.NaN)))).toBe(true);
    // The relative rounding error of a half float is at most 2^-11.
    for (let x = 0.01; x < 100; x *= 1.37) {
      expect(Math.abs(halfToFloat(floatToHalf(x)) - x) / x).toBeLessThan(2 ** -10);
    }
  });

  it('sizes square cells to the canvas aspect', () => {
    expect(computeFluidGrid(128, 1)).toEqual({ width: 128, height: 128 });
    expect(computeFluidGrid(128, 2)).toEqual({ width: 128, height: 64 });
    expect(computeFluidGrid(128, 0.5)).toEqual({ width: 64, height: 128 });
    expect(computeFluidGrid(Number.NaN, Number.NaN)).toEqual({ width: 128, height: 128 });
    expect(computeFluidGrid(1e9, 1, 512)).toEqual({ width: 512, height: 512 });
  });

  it('knows the SOR optimum and the cost of a scheme', () => {
    expect(sorOmega(64, 64)).toBeGreaterThan(1.8);
    expect(sorOmega(64, 64)).toBeLessThanOrEqual(1.9);
    expect(sorOmega(8, 8)).toBeLessThan(sorOmega(256, 256));
    expect(SCHEMES.map(schemePassCount)).toEqual([1, 2, 3]);
  });
});

// --- Rest, projection ------------------------------------------------------------------------------------

describe('a fluid at rest', () => {
  it.each(SCHEMES)('stays at rest and keeps its dye (%s)', async (scheme) => {
    const device = await getTestDevice();
    await expectValid(device, async () => {
      const fluid = create(device, { velocityScheme: scheme, dyeScheme: scheme });
      fluid.setDye(gaussian(0.4, 0.5, 0.06));
      const before = await readDye(device, fluid);
      await submit(device, (e) => fluid.advance(e, 30));
      expect(maxAbs((await readVelocity(device, fluid)).data)).toBe(0);
      const after = await readDye(device, fluid);
      let diff = 0;
      for (let i = 0; i < before.data.length; i++) {
        diff = Math.max(diff, Math.abs(before.data[i]! - after.data[i]!));
      }
      expect(diff).toBeLessThan(1e-3);
      fluid.destroy();
    });
  });

  it('has no pressure and no vorticity', async () => {
    const device = await getTestDevice();
    const fluid = create(device);
    await submit(device, (e) => fluid.advance(e, 10));
    expect(maxAbs(await readHalf(device, fluid.fields))).toBe(0);
    fluid.destroy();
  });
});

describe('projection', () => {
  /** A smooth, strongly divergent field: a Gaussian source. */
  const source = (x: number, y: number): [number, number] => {
    const dx = x - 0.4;
    const dy = y - 0.55;
    const g = Math.exp(-(dx * dx + dy * dy) / 0.02);
    return [dx * g * 4, dy * g * 4];
  };

  async function residual(
    device: GPUDevice,
    params: Partial<FluidParams>,
    iterations: number,
  ): Promise<{ before: number; after: number }> {
    const fluid = create(device, { warmStart: false, ...params });
    fluid.setVelocity(source);
    const before = divergence(await readVelocity(device, fluid)).l2;
    await submit(device, (e) => fluid.project(e, iterations));
    const after = divergence(await readVelocity(device, fluid)).l2;
    fluid.destroy();
    return { before, after };
  }

  it('removes the divergence, more of it with more iterations (Jacobi)', async () => {
    const device = await getTestDevice();
    const ratios: number[] = [];
    for (const iterations of [10, 40, 160, 640]) {
      const { before, after } = await residual(device, { pressureSolver: 'jacobi' }, iterations);
      ratios.push(after / before);
    }
    expect(ratios[0]!).toBeLessThan(1);
    for (let i = 1; i < ratios.length; i++) expect(ratios[i]!).toBeLessThan(ratios[i - 1]!);
    expect(ratios[3]!).toBeLessThan(0.08);
  });

  it('red-black SOR gets to a small fraction of the divergence in a few hundred iterations', async () => {
    const device = await getTestDevice();
    const { before, after } = await residual(device, { pressureSolver: 'sor' }, 320);
    expect(after / before).toBeLessThan(0.02);
    // The floor is the half-float rounding of the stored velocity, not the solver.
    const long = await residual(device, { pressureSolver: 'sor' }, 1000);
    expect(long.after / long.before).toBeLessThan(0.005);
  });

  it('removes a local divergence almost completely: the compact stencils match', async () => {
    const device = await getTestDevice();
    const fluid = create(device, { warmStart: false, pressureSolver: 'jacobi' });
    // Small-scale divergence converges fast: a blob of radius ~4 cells.
    fluid.setVelocity((x, y) => {
      const dx = x - 0.5;
      const dy = y - 0.5;
      const g = Math.exp(-(dx * dx + dy * dy) / 0.0035);
      return [dx * g * 6, dy * g * 6];
    });
    const before = divergence(await readVelocity(device, fluid)).l2;
    await submit(device, (e) => fluid.project(e, 200));
    const after = divergence(await readVelocity(device, fluid));
    expect(after.l2 / before).toBeLessThan(0.05);
    fluid.destroy();
  });

  it('leaves a field that is already divergence-free alone', async () => {
    const device = await getTestDevice();
    const fluid = create(device, { warmStart: false });
    // A stream function that vanishes with its gradient at the walls, so the field is also
    // consistent with them.
    const psi = (x: number, y: number) => Math.sin(Math.PI * x) ** 2 * Math.sin(Math.PI * y) ** 2;
    const e = 1e-4;
    fluid.setVelocity((x, y) => [
      -(psi(x, y + e) - psi(x, y - e)) / (2 * e),
      (psi(x + e, y) - psi(x - e, y)) / (2 * e),
    ]);
    const before = await readVelocity(device, fluid);
    await submit(device, (enc) => fluid.project(enc, 100));
    const after = await readVelocity(device, fluid);
    let change = 0;
    let magnitude = 0;
    for (let j = 0; j < before.height; j++) {
      for (let i = 1; i < before.width; i++) {
        change = Math.max(change, Math.abs(after.u(i, j) - before.u(i, j)));
        magnitude = Math.max(magnitude, Math.abs(before.u(i, j)));
      }
    }
    expect(magnitude).toBeGreaterThan(0.5);
    expect(change / magnitude).toBeLessThan(0.1);
    fluid.destroy();
  });

  it('warm starting from the last solution beats starting from zero', async () => {
    const device = await getTestDevice();
    const run = async (warmStart: boolean) => {
      const fluid = create(device, { warmStart, pressureSolver: 'jacobi' });
      fluid.setVelocity(source);
      const first = divergence(await readVelocity(device, fluid)).l2;
      // The same field projected again and again: the warm solver accumulates its solution.
      for (let k = 0; k < 8; k++) {
        fluid.setVelocity(source);
        await submit(device, (e) => fluid.project(e, 20));
      }
      const last = divergence(await readVelocity(device, fluid)).l2 / first;
      fluid.destroy();
      return last;
    };
    expect(await run(true)).toBeLessThan((await run(false)) * 0.7);
  });
});

// --- Advection --------------------------------------------------------------------------------------------------

describe('advection by a prescribed flow', () => {
  it.each(SCHEMES)('carries a dye blob the expected distance (%s)', async (scheme) => {
    const device = await getTestDevice();
    const fluid = create(device, { dyeOnly: true, dyeScheme: scheme });
    fluid.setVelocity(() => [0.4, 0.2]);
    fluid.setDye(gaussian(0.3, 0.35, 0.05));
    const start = blobStats(await readDye(device, fluid));
    await submit(device, (e) => fluid.advance(e, 30));
    const end = blobStats(await readDye(device, fluid));
    // 30 steps of 1/60 s: 0.5 s at (0.4, 0.2) image heights per second on a square canvas.
    expect(end.cx - start.cx).toBeCloseTo(0.2, 2);
    expect(end.cy - start.cy).toBeCloseTo(0.1, 2);
    expect(end.total / start.total).toBeGreaterThan(0.97);
    expect(end.total / start.total).toBeLessThan(1.03);
    fluid.destroy();
  });

  it('is correct on a wide canvas (square cells, distances in image heights)', async () => {
    const device = await getTestDevice();
    const fluid = create(device, { dyeOnly: true, dyeScheme: 'maccormack' }, 96, 192, 2);
    expect(fluid.grid.width).toBe(96);
    expect(fluid.grid.height).toBe(48);
    fluid.setVelocity(() => [0.5, 0]);
    fluid.setDye(gaussian(0.25, 0.5, 0.04));
    const start = blobStats(await readDye(device, fluid));
    await submit(device, (e) => fluid.advance(e, 60));
    const end = blobStats(await readDye(device, fluid));
    // 1 s at 0.5 heights per second on a canvas twice as wide as high: a quarter of the width.
    expect(end.cx - start.cx).toBeCloseTo(0.25, 2);
    fluid.destroy();
  });

  it('wraps around with the periodic boundary', async () => {
    const device = await getTestDevice();
    const fluid = create(device, { dyeOnly: true, boundary: 'wrap', dyeScheme: 'maccormack' });
    fluid.setVelocity(() => [0.5, 0]);
    fluid.setDye(gaussian(0.8, 0.5, 0.04));
    // 1.2 s at 0.5 / s = 0.6: the blob crosses the right edge and comes in on the left at 0.4.
    await submit(device, (e) => fluid.advance(e, 72));
    const dye = await readDye(device, fluid);
    let px = 0;
    let best = -1;
    for (let x = 0; x < dye.width; x++) {
      const v = dye.at(x, Math.floor(dye.height / 2));
      if (v > best) {
        best = v;
        px = (x + 0.5) / dye.width;
      }
    }
    expect(px).toBeCloseTo(0.4, 1);
    expect(best).toBeGreaterThan(0.5);
    fluid.destroy();
  });

  it('lets dye leave through an open boundary', async () => {
    const device = await getTestDevice();
    const fluid = create(device, { dyeOnly: true, boundary: 'open', dyeScheme: 'semi-lagrangian' });
    fluid.setVelocity(() => [0.6, 0]);
    fluid.setDye(gaussian(0.8, 0.5, 0.04));
    const start = blobStats(await readDye(device, fluid)).total;
    await submit(device, (e) => fluid.advance(e, 72));
    expect(blobStats(await readDye(device, fluid)).total / start).toBeLessThan(0.02);
    fluid.destroy();
  });
});

// --- Boundaries and obstacles -----------------------------------------------------------------------------------

describe('walls', () => {
  it.each(SCHEMES)(
    'leave no normal velocity at the edge and keep the dye in (%s)',
    async (scheme) => {
      const device = await getTestDevice();
      const fluid = create(device, { velocityScheme: scheme, dyeScheme: scheme, vorticity: 8 });
      fluid.setDye(gaussian(0.5, 0.5, 0.08, 2));
      const start = blobStats(await readDye(device, fluid)).total;
      for (let k = 0; k < 6; k++) {
        // Shoves at every wall.
        fluid.addForce([0.5, 0.5], [0.95, 0.5], 0.06, [3, 0]);
        fluid.addForce([0.5, 0.5], [0.5, 0.05], 0.06, [0, -3]);
        fluid.addForce([0.5, 0.5], [0.05, 0.95], 0.06, [-2, 2]);
        await submit(device, (e) => fluid.advance(e, 12));
      }
      const v = await readVelocity(device, fluid);
      let worst = 0;
      let interior = 0;
      for (let j = 0; j < v.height; j++) {
        worst = Math.max(worst, Math.abs(v.u(0, j)), Math.abs(v.u(v.width, j)));
        for (let i = 1; i < v.width; i++) interior = Math.max(interior, Math.abs(v.u(i, j)));
      }
      for (let i = 0; i < v.width; i++) {
        worst = Math.max(worst, Math.abs(v.v(i, 0)), Math.abs(v.v(i, v.height)));
      }
      expect(interior).toBeGreaterThan(0.05);
      expect(worst).toBe(0);
      // The dye stays in the box. (Its total is not exactly conserved: the dye is a scalar advected in
      // non-conservative form by an interpolated velocity whose divergence is only zero cell by
      // cell; vigorous stirring moves the total by tens of percent, see notes.md.)
      const end = blobStats(await readDye(device, fluid)).total;
      expect(end / start).toBeGreaterThan(0.6);
      expect(end / start).toBeLessThan(1.6);
      fluid.destroy();
    },
  );
});

describe('obstacles', () => {
  it('make flow go round a block: no velocity into it, no dye in it, and the flow is real', async () => {
    const device = await getTestDevice();
    const fluid = create(device, { vorticity: 6 });
    fluid.obstacles.setRects([{ x: 0.42, y: 0.3, width: 0.16, height: 0.4, radius: 0.03 }]);
    expect(fluid.obstacles.enabled).toBe(true);
    const mask = await readTexture(device, fluid.obstacles.texture, 1);
    const { width, height } = fluid.grid;
    const solid = (i: number, j: number) =>
      i >= 0 && j >= 0 && i < width && j < height && mask[j * width + i]! >= 128;
    fluid.setDye(gaussian(0.25, 0.5, 0.06, 2));
    for (let k = 0; k < 10; k++) {
      fluid.addForce([0.05, 0.5], [0.3, 0.5], 0.08, [1.5, 0]);
      await submit(device, (e) => fluid.advance(e, 12));
    }
    const v = await readVelocity(device, fluid);
    let touching = 0;
    let leak = 0;
    for (let j = 0; j < height; j++) {
      for (let i = 0; i <= width; i++) {
        if (solid(i - 1, j) || solid(i, j)) {
          touching++;
          leak = Math.max(leak, Math.abs(v.u(i, j)));
        }
      }
    }
    for (let j = 0; j <= height; j++) {
      for (let i = 0; i < width; i++) {
        if (solid(i, j - 1) || solid(i, j)) {
          touching++;
          leak = Math.max(leak, Math.abs(v.v(i, j)));
        }
      }
    }
    expect(touching).toBeGreaterThan(50);
    expect(leak).toBe(0);
    // Dye never appears inside the block.
    const dye = await readDye(device, fluid);
    let inside = 0;
    let outside = 0;
    for (let y = 0; y < dye.height; y++) {
      for (let x = 0; x < dye.width; x++) {
        const cell = solid(
          Math.floor((x * width) / dye.width),
          Math.floor((y * height) / dye.height),
        );
        if (cell) inside += Math.abs(dye.at(x, y));
        else outside += dye.at(x, y);
      }
    }
    expect(inside).toBe(0);
    expect(outside).toBeGreaterThan(1);
    // The fluid moved: the flow reaches the far side of the block, so it went around.
    let speed = 0;
    for (let j = 0; j < height; j++) {
      speed = Math.max(speed, Math.abs(v.u(Math.round(width * 0.72), j)));
    }
    expect(speed).toBeGreaterThan(0.02);
    fluid.destroy();
  });

  it('a one-cell wall stops dye and flow completely', async () => {
    const device = await getTestDevice();
    const fluid = create(device, { dyeScheme: 'maccormack', velocityScheme: 'maccormack' });
    const { width, height } = fluid.grid;
    // A full-height wall one cell thick at column 32.
    const wall = new Uint8Array(width * height);
    for (let j = 0; j < height; j++) wall[j * width + 32] = 255;
    fluid.obstacles.set(wall);
    fluid.setDye((x, y) => (x < 0.45 ? gaussian(0.3, 0.5, 0.1, 3)(x, y) : [0, 0, 0, 0]));
    for (let k = 0; k < 16; k++) {
      fluid.addForce([0.1, 0.5], [0.45, 0.5], 0.1, [4, 0]);
      await submit(device, (e) => fluid.advance(e, 10));
    }
    const dye = await readDye(device, fluid);
    let left = 0;
    let right = 0;
    for (let y = 0; y < dye.height; y++) {
      for (let x = 0; x < dye.width; x++) {
        if (x < dye.width / 2 - 2) left += dye.at(x, y);
        else if (x > dye.width / 2 + 2) right += dye.at(x, y);
      }
    }
    expect(left).toBeGreaterThan(10);
    expect(right).toBe(0);
    // The wall is exactly a velocity-free surface for both neighbours.
    const v = await readVelocity(device, fluid);
    for (let j = 0; j < height; j++) {
      expect(v.u(32, j)).toBe(0);
      expect(v.u(33, j)).toBe(0);
    }
    fluid.destroy();
  });

  it('shares the heightfield solver mask type and keeps rectangles across a resize', async () => {
    const device = await getTestDevice();
    const fluid = create(device);
    fluid.obstacles.setRects([{ x: 0.1, y: 0.1, width: 0.2, height: 0.2 }]);
    fluid.resize(2, 1);
    expect(fluid.obstacles.width).toBe(fluid.grid.width);
    expect(fluid.obstacles.enabled).toBe(true);
    fluid.obstacles.clear();
    expect(fluid.obstacles.enabled).toBe(false);
    fluid.destroy();
  });
});

describe('open boundaries (full solver)', () => {
  it('let a pushed blob leave, where a wall keeps it', async () => {
    const device = await getTestDevice();
    const remaining: Record<string, number> = {};
    for (const boundary of ['open', 'wall'] as const) {
      const fluid = create(device, { boundary });
      fluid.setDye(gaussian(0.7, 0.5, 0.05, 2));
      const start = blobStats(await readDye(device, fluid)).total;
      for (let k = 0; k < 20; k++) {
        fluid.addForce([0.5, 0.5], [0.9, 0.5], 0.1, [2.5, 0]);
        await submit(device, (e) => fluid.advance(e, 6));
      }
      remaining[boundary] = blobStats(await readDye(device, fluid)).total / start;
      fluid.destroy();
    }
    expect(remaining.open!).toBeLessThan(0.5);
    expect(remaining.wall!).toBeGreaterThan(0.8);
  });
});

// --- Energy, conservation, detail ------------------------------------------------------------------------------------

describe('energy and forces', () => {
  it.each(SCHEMES)('does not grow without forcing (%s, confinement off)', async (scheme) => {
    const device = await getTestDevice();
    const fluid = create(device, { velocityScheme: scheme, dyeScheme: scheme, vorticity: 0 });
    fluid.addForce([0.3, 0.5], [0.6, 0.55], 0.07, [1.5, 0.3]);
    fluid.addForce([0.7, 0.3], [0.4, 0.6], 0.05, [-1, 0.5]);
    await submit(device, (e) => fluid.advance(e, 2));
    let previous = energy(await readVelocity(device, fluid));
    expect(previous).toBeGreaterThan(0);
    const first = previous;
    for (let k = 0; k < 30; k++) {
      await submit(device, (e) => fluid.advance(e, 4));
      const now = energy(await readVelocity(device, fluid));
      // Slack for half-float rounding of the stored velocity and for the limited MacCormack and
      // BFECC corrections, which are not strictly dissipative: measured wiggles stay under 0.6%.
      expect(now).toBeLessThanOrEqual(previous * 1.015);
      expect(now).toBeLessThanOrEqual(first * 1.01);
      previous = now;
    }
    expect(previous).toBeLessThan(first);
    fluid.destroy();
  });

  it('vorticity confinement keeps a vortex stronger than leaving it alone', async () => {
    const device = await getTestDevice();
    const peaks: number[] = [];
    for (const vorticity of [0, 20]) {
      const fluid = create(device, { vorticity, velocityScheme: 'semi-lagrangian' }, 64, 64);
      fluid.setVelocity(vortex());
      await submit(device, (enc) => {
        fluid.project(enc, 100);
        fluid.advance(enc, 90);
      });
      const fields = await readHalf(device, fluid.fields);
      let peak = 0;
      for (let i = 0; i < fields.length; i += 4) peak = Math.max(peak, Math.abs(fields[i + 2]!));
      peaks.push(peak);
      fluid.destroy();
    }
    expect(peaks[1]!).toBeGreaterThan(peaks[0]! * 1.05);
  });

  it('viscosity smooths a vortex', async () => {
    const device = await getTestDevice();
    const peaks: number[] = [];
    for (const viscosity of [0, 0.002]) {
      const fluid = create(device, { viscosity, velocityScheme: 'maccormack' }, 64, 64);
      fluid.setVelocity(vortex());
      await submit(device, (enc) => {
        fluid.project(enc, 100);
        fluid.advance(enc, 60);
      });
      const fields = await readHalf(device, fluid.fields);
      expect(allFinite(fields)).toBe(true);
      let peak = 0;
      for (let i = 0; i < fields.length; i += 4) peak = Math.max(peak, Math.abs(fields[i + 2]!));
      peaks.push(peak);
      fluid.destroy();
    }
    expect(peaks[1]!).toBeLessThan(peaks[0]! * 0.8);
  });

  it('buoyancy lifts dense dye and sinks it when negative', async () => {
    const device = await getTestDevice();
    const moved: number[] = [];
    for (const buoyancy of [1.5, -1.5]) {
      const fluid = create(device, { buoyancy, dyeScheme: 'semi-lagrangian' }, 64, 64);
      fluid.setDye(gaussian(0.5, 0.5, 0.08));
      const start = blobStats(await readDye(device, fluid));
      await submit(device, (e) => fluid.advance(e, 45));
      moved.push(blobStats(await readDye(device, fluid)).cy - start.cy);
      fluid.destroy();
    }
    // y is down: rising is a negative change.
    expect(moved[0]!).toBeLessThan(-0.02);
    expect(moved[1]!).toBeGreaterThan(0.02);
  });
});

describe('detail: advection schemes carrying a blob round a vortex', () => {
  async function carry(device: GPUDevice, scheme: FluidScheme, steps: number) {
    const fluid = create(device, { dyeOnly: true, dyeScheme: scheme }, 64, 256);
    fluid.setVelocity(rotation(0.25));
    fluid.setDye(gaussian(0.7, 0.5, 0.03));
    await submit(device, (e) => fluid.advance(e, steps));
    const stats = blobStats(await readDye(device, fluid));
    fluid.destroy();
    return stats;
  }

  it('MacCormack and BFECC keep the blob sharper than semi-Lagrangian, and never overshoot', async () => {
    const device = await getTestDevice();
    // One full turn: 4 s at a quarter turn per second.
    const [sl, mc, bfecc] = (await Promise.all(SCHEMES.map((s) => carry(device, s, 240)))) as [
      BlobStats,
      BlobStats,
      BlobStats,
    ];
    expect(sl.peak).toBeLessThan(0.7);
    for (const stats of [mc, bfecc]) {
      expect(stats.peak).toBeGreaterThan(sl.peak + 0.15);
      expect(stats.sigma).toBeLessThan(sl.sigma * 0.85);
    }
    // The limiters: no new extrema, to half-float rounding.
    for (const stats of [sl, mc, bfecc]) {
      expect(stats.min).toBeGreaterThanOrEqual(-2e-3);
      expect(stats.peak).toBeLessThanOrEqual(1 + 2e-3);
    }
    // Dye is conserved to within ten percent per revolution by all three under a divergence-free
    // flow (measured: semi-Lagrangian -8%, MacCormack -4%, BFECC -12%; see notes.md).
    const expected = 2 * Math.PI * 0.03 ** 2 * 256 * 256;
    expect(Math.abs(sl.total / expected - 1)).toBeLessThan(0.1);
    expect(Math.abs(mc.total / expected - 1)).toBeLessThan(0.06);
    expect(Math.abs(bfecc.total / expected - 1)).toBeLessThan(0.15);
  });

  it('the limiter is what stops a sharp step from ringing', async () => {
    const device = await getTestDevice();
    for (const scheme of ['maccormack', 'bfecc'] as const) {
      const fluid = create(device, { dyeOnly: true, dyeScheme: scheme }, 64, 128);
      fluid.setVelocity(() => [0.31, 0.17]);
      // A hard-edged square: the worst case for overshoot.
      fluid.setDye((x, y) =>
        Math.abs(x - 0.35) < 0.1 && Math.abs(y - 0.4) < 0.1 ? [1, 1, 1, 1] : [0, 0, 0, 0],
      );
      await submit(device, (e) => fluid.advance(e, 100));
      const stats = blobStats(await readDye(device, fluid));
      expect(stats.peak).toBeLessThanOrEqual(1 + 1e-3);
      expect(stats.min).toBeGreaterThanOrEqual(-1e-3);
      fluid.destroy();
    }
  });
});

// --- Stability ------------------------------------------------------------------------------------------------------------

describe('robustness', () => {
  it.each(SCHEMES)(
    'survives absurd forces and many sub-steps without NaN or Inf (%s)',
    async (scheme) => {
      const device = await getTestDevice();
      await expectValid(device, async () => {
        const fluid = create(device, {
          velocityScheme: scheme,
          dyeScheme: scheme,
          vorticity: 60,
          viscosity: 0.01,
          buoyancy: 2,
          gravity: [1e6, -1e6],
        });
        for (let k = 0; k < MAX_SPLATS; k++) {
          const a: [number, number] = [Math.random(), Math.random()];
          fluid.addForce(a, [Math.random(), Math.random()], 0.02 + Math.random() * 0.1, [
            (Math.random() - 0.5) * 3e38,
            1e6,
          ]);
          fluid.addDye(a, a, 0.05, [1e30, 1e30, -1e30, 1e30]);
        }
        expect(fluid.addForce([0, 0], [1, 1], 0.1, [1, 1])).toBe(false);
        expect(fluid.pendingSplats).toEqual([MAX_SPLATS, MAX_SPLATS]);
        await submit(device, (e) => fluid.advance(e, 40));
        for (let k = 0; k < 8; k++) await submit(device, (e) => fluid.update(e, 100));
        const v = await readVelocity(device, fluid);
        const dye = await readDye(device, fluid);
        const fields = await readHalf(device, fluid.fields);
        expect(allFinite(v.data)).toBe(true);
        expect(allFinite(dye.data)).toBe(true);
        expect(allFinite(fields)).toBe(true);
        const limit = fluid.params.maxCfl / (fluid.stepSeconds * fluid.grid.height);
        expect(maxAbs(v.data)).toBeLessThanOrEqual(limit * 1.01);
        expect(maxAbs(dye.data)).toBeLessThanOrEqual(fluid.params.dyeLimit);
        fluid.destroy();
      });
    },
  );

  it('ignores splats with non-finite numbers', async () => {
    const device = await getTestDevice();
    const fluid = create(device);
    expect(fluid.addForce([Number.NaN, 0], [0, 0], 0.1, [1, 1])).toBe(false);
    expect(fluid.addForce([0, 0], [0, 0], Infinity, [1, 1])).toBe(false);
    expect(fluid.addDye([0, 0], [0, 0], 0.1, [1, Number.NaN, 0, 0])).toBe(false);
    expect(fluid.pendingSplats).toEqual([0, 0]);
    fluid.destroy();
  });
});

describe('time', () => {
  it('behaves the same at 60 Hz, 120 Hz and 144 Hz', async () => {
    const device = await getTestDevice();
    const results: Velocity[] = [];
    const dyes: BlobStats[] = [];
    for (const fps of [60, 120, 144]) {
      const fluid = create(device, { vorticity: 8 });
      fluid.setDye(gaussian(0.4, 0.5, 0.07));
      fluid.addForce([0.2, 0.5], [0.6, 0.5], 0.08, [1.2, 0.2]);
      let steps = 0;
      for (let frame = 0; frame < fps; frame++) {
        const encoder = device.createCommandEncoder();
        steps += fluid.update(encoder, 1 / fps);
        device.queue.submit([encoder.finish()]);
      }
      await device.queue.onSubmittedWorkDone();
      // One second is sixty steps at any display rate (the last one can land a frame late).
      expect(Math.abs(steps - 60)).toBeLessThanOrEqual(1);
      results.push(await readVelocity(device, fluid));
      dyes.push(blobStats(await readDye(device, fluid)));
      fluid.destroy();
    }
    const peak = maxAbs(results[0]!.data);
    expect(peak).toBeGreaterThan(0.05);
    for (const other of results.slice(1)) {
      let worst = 0;
      for (let i = 0; i < other.data.length; i++) {
        worst = Math.max(worst, Math.abs(other.data[i]! - results[0]!.data[i]!));
      }
      expect(worst / peak).toBeLessThan(0.06);
    }
    for (const other of dyes.slice(1)) {
      expect(other.cx).toBeCloseTo(dyes[0]!.cx, 2);
      expect(other.cy).toBeCloseTo(dyes[0]!.cy, 2);
    }
  });

  it('runs whole steps, carries the remainder and caps a long stall', async () => {
    const device = await getTestDevice();
    const fluid = createFluid({ device, resolution: 32, dyeResolution: 64, maxStepsPerFrame: 4 });
    const counts: number[] = [];
    for (const delta of [1 / 60, 1 / 120, 1 / 120, 1 / 30, 5, 0, Number.NaN, -1]) {
      const encoder = device.createCommandEncoder();
      counts.push(fluid.update(encoder, delta));
      device.queue.submit([encoder.finish()]);
    }
    expect(counts).toEqual([1, 0, 1, 2, 4, 0, 0, 0]);
    expect(fluid.stepSeconds).toBeCloseTo(1 / 60, 8);
    fluid.params.stepRate = 120;
    expect(fluid.stepSeconds).toBeCloseTo(1 / 120, 8);
    await device.queue.onSubmittedWorkDone();
    fluid.destroy();
  });
});

describe('lifecycle', () => {
  it('re-creates its textures cleanly when the canvas changes shape or the resolution does', async () => {
    const device = await getTestDevice();
    await expectValid(device, async () => {
      const fluid = create(device);
      const generation = fluid.generation;
      fluid.setDye(gaussian(0.5, 0.5, 0.1));
      fluid.addForce([0.5, 0.5], [0.6, 0.5], 0.1, [1, 0]);
      await submit(device, (e) => fluid.advance(e, 3));

      expect(fluid.resize(64, 64)).toBe(false);
      expect(fluid.generation).toBe(generation);
      expect(fluid.resize(2, 1)).toBe(true);
      expect(fluid.generation).toBe(generation + 1);
      expect(fluid.grid.width).toBe(64);
      expect(fluid.grid.height).toBe(32);
      expect(fluid.grid.velocityWidth).toBe(65);
      expect(fluid.grid.dyeWidth).toBe(128);
      expect(fluid.pendingSplats).toEqual([0, 0]);
      // The new state is empty, and works.
      expect((await readDye(device, fluid)).data.every((x) => x === 0)).toBe(true);
      await submit(device, (e) => fluid.advance(e, 3));

      expect(fluid.setResolution(96, 192)).toBe(true);
      expect(fluid.grid.width).toBe(96);
      expect(fluid.grid.dyeWidth).toBe(192);
      expect(fluid.setResolution(96, 192)).toBe(false);
      expect(fluid.resize(-1, 5)).toBe(false);
      await submit(device, (e) => fluid.advance(e, 3));
      fluid.destroy();
      fluid.destroy();
      // A destroyed fluid ignores further work.
      await submit(device, (e) => fluid.update(e, 1 / 60));
    });
  });

  it('resets to rest', async () => {
    const device = await getTestDevice();
    const fluid = create(device);
    fluid.setDye(gaussian(0.5, 0.5, 0.1));
    fluid.addForce([0.5, 0.5], [0.7, 0.5], 0.1, [2, 0]);
    await submit(device, (e) => fluid.advance(e, 10));
    await submit(device, (e) => fluid.reset(e));
    expect(maxAbs((await readVelocity(device, fluid)).data)).toBe(0);
    expect(maxAbs((await readDye(device, fluid)).data)).toBe(0);
    fluid.destroy();
  });

  it('runs every boundary, solver and scheme combination without a validation error', async () => {
    const device = await getTestDevice();
    await expectValid(device, async () => {
      for (const boundary of ['wall', 'open', 'wrap'] as const) {
        for (const pressureSolver of ['jacobi', 'sor'] as const) {
          const fluid = create(device, {
            boundary,
            pressureSolver,
            warmStart: boundary !== 'open',
            velocityScheme: boundary === 'wrap' ? 'bfecc' : 'maccormack',
            dyeScheme: boundary === 'open' ? 'bfecc' : 'semi-lagrangian',
            vorticity: 5,
            viscosity: 0.001,
          });
          fluid.obstacles.setRects([{ x: 0.4, y: 0.4, width: 0.1, height: 0.1 }]);
          fluid.addForce([0.2, 0.2], [0.8, 0.7], 0.05, [1, 1]);
          fluid.addDye([0.2, 0.2], [0.8, 0.7], 0.05, [1, 0.5, 0.25, 1], 0.02);
          await submit(device, (e) => fluid.advance(e, 4));
          expect(allFinite((await readVelocity(device, fluid)).data)).toBe(true);
          fluid.destroy();
        }
      }
    });
  });
});

// --- Consumers: hooks, fields, decode helpers ---------------------------------------------------------------------------------

describe('hooks', () => {
  it('lets a consumer add a force field and a dye source', async () => {
    const device = await getTestDevice();
    await expectValid(device, async () => {
      const module = device.createShaderModule({
        code: `
          @group(0) @binding(0) var src: texture_2d<f32>;
          @group(0) @binding(1) var dst: texture_storage_2d<rgba16float, write>;
          @compute @workgroup_size(8, 8)
          fn push(@builtin(global_invocation_id) id: vec3u) {
            let size = textureDimensions(src);
            if (any(id.xy >= size)) { return; }
            let c = vec2i(id.xy);
            textureStore(dst, c, textureLoad(src, c, 0) + vec4f(0.25, 0.0, 0.0, 0.0));
          }
          @compute @workgroup_size(8, 8)
          fn source(@builtin(global_invocation_id) id: vec3u) {
            let size = textureDimensions(src);
            if (any(id.xy >= size)) { return; }
            let c = vec2i(id.xy);
            textureStore(dst, c, textureLoad(src, c, 0) + vec4f(0.0, 0.5, 0.0, 0.5));
          }`,
      });
      const layout = device.createBindGroupLayout({
        entries: [
          { binding: 0, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: 'float' } },
          {
            binding: 1,
            visibility: GPUShaderStage.COMPUTE,
            storageTexture: { access: 'write-only', format: 'rgba16float' },
          },
        ],
      });
      const make = (entryPoint: string) =>
        device.createComputePipeline({
          layout: device.createPipelineLayout({ bindGroupLayouts: [layout] }),
          compute: { module, entryPoint },
        });
      const push = make('push');
      const source = make('source');
      const fluid = create(device, { boundary: 'wrap', velocityScheme: 'semi-lagrangian' });
      let calls = 0;
      const run = (
        pipeline: GPUComputePipeline,
        encoder: GPUCommandEncoder,
        read: GPUTextureView,
        write: GPUTextureView,
        size: [number, number],
      ) => {
        const pass = encoder.beginComputePass();
        pass.setPipeline(pipeline);
        pass.setBindGroup(
          0,
          device.createBindGroup({
            layout,
            entries: [
              { binding: 0, resource: read },
              { binding: 1, resource: write },
            ],
          }),
        );
        pass.dispatchWorkgroups(Math.ceil(size[0] / 8), Math.ceil(size[1] / 8));
        pass.end();
      };
      fluid.hooks.forces = ({ encoder, read, write, grid }) => {
        calls++;
        run(push, encoder, read, write, [grid.velocityWidth, grid.velocityHeight]);
        return true;
      };
      fluid.hooks.dye = ({ encoder, read, write, grid }) => {
        run(source, encoder, read, write, [grid.dyeWidth, grid.dyeHeight]);
        return true;
      };
      await submit(device, (e) => fluid.advance(e, 5));
      expect(calls).toBe(5);
      const v = await readVelocity(device, fluid);
      // Five steps of +0.25 in every u face: a periodic fluid can only keep that as a uniform flow,
      // which the projection leaves alone.
      expect(v.u(10, 10)).toBeGreaterThan(1);
      const dye = await readDye(device, fluid);
      expect(dye.at(20, 20, 1)).toBeGreaterThan(2);
      expect(dye.at(20, 20, 0)).toBe(0);
      fluid.hooks.forces = undefined;
      fluid.hooks.dye = undefined;
      fluid.destroy();
    });
  });
});

describe('the fields texture', () => {
  it('holds the cell-centred velocity and the vorticity', async () => {
    const device = await getTestDevice();
    const fluid = create(device, { dyeOnly: true });
    const turns = 0.2;
    fluid.setVelocity(rotation(turns));
    await submit(device, (e) => fluid.resolve(e));
    const fields = await readHalf(device, fluid.fields);
    const { width, height } = fluid.grid;
    let worstVelocity = 0;
    let worstVorticity = 0;
    for (let j = 8; j < height - 8; j++) {
      for (let i = 8; i < width - 8; i++) {
        const o = (j * width + i) * 4;
        const expected = rotation(turns)((i + 0.5) / width, (j + 0.5) / height);
        worstVelocity = Math.max(
          worstVelocity,
          Math.hypot(fields[o]! - expected[0], fields[o + 1]! - expected[1]),
        );
        // du/dy = -Omega, dv/dx = Omega: the curl is 2 Omega, positive for a rotation that is
        // clockwise on screen (y down), as documented.
        worstVorticity = Math.max(
          worstVorticity,
          Math.abs(fields[o + 2]! - 2 * 2 * Math.PI * turns),
        );
      }
    }
    expect(worstVelocity).toBeLessThan(0.002);
    expect(worstVorticity).toBeLessThan(0.02);
    fluid.destroy();
  });
});

describe('src/wgsl/fluid.wgsl', () => {
  it('compiles', async () => {
    const device = await getTestDevice();
    const { diagnostics } = await checkShader(device, {
      label: 'fluid-chunk',
      code: '#include "fluid.wgsl"',
    });
    expect(diagnostics.map(formatDiagnostic)).toEqual([]);
  });

  it('decodes a fields texel, the colour helpers and the optical density', async () => {
    const r = await scalars('#include "fluid.wgsl"', [
      'fluid_speed(vec4f(3.0, 4.0, 0.0, 0.0))',
      'fluid_vorticity(vec4f(1.0, 2.0, 7.5, -3.0))',
      'fluid_pressure(vec4f(1.0, 2.0, 7.5, -3.0))',
      'fluid_velocity(vec4f(1.0, 2.0, 7.5, -3.0)).y',
      'fluid_dye_density(vec4f(0.1, 0.2, 0.3, 0.9))',
      'fluid_dye_rgb(vec4f(0.1, 0.2, 0.3, 0.9)).b',
      'fluid_transmittance(fluid_absorbance(vec3f(0.25, 0.5, 1.0))).x',
      'fluid_transmittance(fluid_absorbance(vec3f(0.25, 0.5, 1.0))).y',
      'fluid_transmittance(fluid_absorbance(vec3f(0.25, 0.5, 1.0))).z',
      'fluid_transmittance(fluid_absorbance(vec3f(0.0))).x',
      'fluid_diverging(0.0).x + fluid_diverging(0.0).y + fluid_diverging(0.0).z',
      'fluid_diverging(1.0).x',
      'fluid_diverging(-1.0).z',
      'length(fluid_flow_color(vec2f(0.0), 5.0))',
      'length(fluid_flow_color(vec2f(100.0, 0.0), 5.0))',
    ]);
    expect(r[0]).toBeCloseTo(5, 5);
    expect(r[1]).toBeCloseTo(7.5, 5);
    expect(r[2]).toBeCloseTo(-3, 5);
    expect(r[3]).toBeCloseTo(2, 5);
    expect(r[4]).toBeCloseTo(0.9, 5);
    expect(r[5]).toBeCloseTo(0.3, 5);
    expect(r[6]).toBeCloseTo(0.25, 5);
    expect(r[7]).toBeCloseTo(0.5, 5);
    expect(r[8]).toBeCloseTo(1, 5);
    // A colour of 0 is clamped to 0.02, not infinite density.
    expect(r[9]).toBeCloseTo(0.02, 5);
    expect(r[10]).toBeCloseTo(0, 6);
    expect(r[11]).toBeCloseTo(1, 5);
    expect(r[12]).toBeCloseTo(0.9, 5);
    expect(r[13]).toBeCloseTo(0, 6);
    expect(r[14]).toBeGreaterThan(0.5);
  });

  it('fluid_velocity_at reads the staggered texture where the faces are', async () => {
    const device = await getTestDevice();
    const fluid = create(device, { dyeOnly: true }, 32, 64);
    // A linear field is reproduced exactly by the face layout and bilinear filtering.
    const field = (x: number, y: number): [number, number] => [
      0.3 + 0.8 * x - 0.2 * y,
      -0.1 + 0.4 * y + 0.5 * x,
    ];
    fluid.setVelocity(field);
    const { module, diagnostics } = await checkShader(device, {
      label: 'fluid-velocity-at-test',
      code: `
        #include "fluid.wgsl"
        @group(0) @binding(0) var tex: texture_2d<f32>;
        @group(0) @binding(1) var samp: sampler;
        @group(0) @binding(2) var<storage, read_write> out: array<vec2f>;
        @compute @workgroup_size(1)
        fn main(@builtin(global_invocation_id) id: vec3u) {
          let uv = (vec2f(id.xy) + 0.5) / 8.0;
          out[id.y * 8u + id.x] = fluid_velocity_at(tex, samp, uv, vec2f(32.0, 32.0));
        }`,
    });
    expect(diagnostics.filter((d) => d.severity === 'error').map(formatDiagnostic)).toEqual([]);
    const sampler = device.createSampler({ magFilter: 'linear', minFilter: 'linear' });
    const output = device.createBuffer({
      size: 64 * 8,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });
    const read = device.createBuffer({
      size: 64 * 8,
      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    });
    const pipeline = device.createComputePipeline({
      layout: 'auto',
      compute: { module, entryPoint: 'main' },
    });
    const group = device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: fluid.velocity.view },
        { binding: 1, resource: sampler },
        { binding: 2, resource: { buffer: output } },
      ],
    });
    const encoder = device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, group);
    pass.dispatchWorkgroups(8, 8);
    pass.end();
    encoder.copyBufferToBuffer(output, 0, read, 0, 64 * 8);
    device.queue.submit([encoder.finish()]);
    await read.mapAsync(GPUMapMode.READ);
    const values = new Float32Array(read.getMappedRange().slice(0));
    read.unmap();
    let worst = 0;
    for (let y = 1; y < 7; y++) {
      for (let x = 1; x < 7; x++) {
        const expected = field((x + 0.5) / 8, (y + 0.5) / 8);
        worst = Math.max(
          worst,
          Math.abs(values[(y * 8 + x) * 2]! - expected[0]),
          Math.abs(values[(y * 8 + x) * 2 + 1]! - expected[1]),
        );
      }
    }
    // Half-float storage plus 8-bit bilinear weights.
    expect(worst).toBeLessThan(0.006);
    output.destroy();
    read.destroy();
    fluid.destroy();
  });
});

// --- Pointers ----------------------------------------------------------------------------------------------------------------------

describe('pointer stirring', () => {
  function pointer(patch: Partial<TrackedPointer>): TrackedPointer {
    return {
      id: 1,
      type: 'mouse',
      x: 0.5,
      y: 0.5,
      dx: 0,
      dy: 0,
      vx: 0,
      vy: 0,
      speed: 0,
      down: true,
      inside: true,
      pressure: 0.5,
      justPressed: false,
      justReleased: false,
      justEntered: false,
      justLeft: false,
      cancelled: false,
      path: [{ x: 0.5, y: 0.5 }],
      ...patch,
    };
  }

  type Pair = readonly [number, number];

  function recorder() {
    const forces: { from: number[]; to: number[]; velocity: number[]; ramp?: number }[] = [];
    const dyes: { from: number[]; to: number[]; value: number[] }[] = [];
    return {
      forces,
      dyes,
      addForce(from: Pair, to: Pair, _radius: number, velocity: Pair, ramp?: number) {
        forces.push({ from: [...from], to: [...to], velocity: [...velocity], ramp });
        return true;
      },
      addDye(
        from: Pair,
        to: Pair,
        _radius: number,
        value: readonly [number, number, number, number],
      ) {
        dyes.push({ from: [...from], to: [...to], value: [...value] });
        return true;
      },
    };
  }

  const options = {
    mode: 'press only' as const,
    radius: 0.05,
    force: 1,
    dyeAmount: 1,
    colour: (stroke: FluidStroke): Rgba => [stroke.index, 0, 0, 1],
  };

  it('cuts a path into equal pieces and connects them end to start', () => {
    const path = [
      { x: 0, y: 0 },
      { x: 0.5, y: 0 },
      { x: 0.5, y: 0.5 },
    ];
    const { points, length } = resamplePath(path, 1, 4);
    expect(length).toBeCloseTo(1, 6);
    expect(points).toHaveLength(5);
    expect(points[0]).toEqual({ x: 0, y: 0 });
    expect(points[2]!.x).toBeCloseTo(0.5, 6);
    expect(points[2]!.y).toBeCloseTo(0, 6);
    expect(points[4]!.y).toBeCloseTo(0.5, 6);
    // Lengths count image heights: a canvas twice as wide makes the horizontal leg twice as long.
    expect(resamplePath(path, 2, 1).length).toBeCloseTo(1.5, 6);
    expect(resamplePath([{ x: 0.3, y: 0.3 }], 1, 3).length).toBe(0);
  });

  it('pushes along the path with the pointer velocity and lays dye with the stroke colour', () => {
    const stirrer = createFluidStirrer();
    const fluid = recorder();
    const result = stirrer.stir(
      fluid,
      [
        pointer({
          justPressed: true,
          path: [
            { x: 0.2, y: 0.5 },
            { x: 0.4, y: 0.5 },
          ],
        }),
      ],
      1,
      1 / 60,
      options,
    );
    expect(result.active).toBe(true);
    expect(result.speed).toBeCloseTo(0.2 * 60, 4);
    const pushes = fluid.forces.filter((f) => f.ramp !== undefined);
    expect(pushes.length).toBeGreaterThan(0);
    // Pointer speed 12 / s is capped at 3 (maxSpeed), pointing right.
    for (const push of pushes) {
      expect(push.velocity[0]).toBeCloseTo(3, 4);
      expect(Math.abs(push.velocity[1]!)).toBeLessThan(1e-9);
      expect(push.ramp).toBeGreaterThan(0);
    }
    // Chained segments: each ends where the next begins, and the whole spans the path.
    expect(pushes[0]!.from[0]).toBeCloseTo(0.2, 6);
    for (let i = 1; i < pushes.length; i++) {
      expect(pushes[i]!.from[0]).toBeCloseTo(pushes[i - 1]!.to[0]!, 9);
    }
    expect(pushes[pushes.length - 1]!.to[0]).toBeCloseTo(0.4, 6);
    // Dye carries a density of 1 times dyeAmount.
    expect(fluid.dyes.length).toBeGreaterThan(0);
    expect(fluid.dyes[0]!.value[3]).toBeCloseTo(1, 6);
    // A tap puffs: six outward pushes without a ramp.
    expect(fluid.forces.filter((f) => f.ramp === undefined)).toHaveLength(6);
  });

  it('starts a new stroke on every press, and ignores hover unless asked', () => {
    const stirrer = createFluidStirrer();
    const fluid = recorder();
    const moving = (id: number, justPressed: boolean, down = true) =>
      pointer({
        id,
        down,
        justPressed,
        path: [
          { x: 0.2, y: 0.5 },
          { x: 0.25, y: 0.5 },
        ],
      });
    stirrer.stir(fluid, [moving(1, true)], 1, 1 / 60, options);
    stirrer.stir(fluid, [moving(1, false)], 1, 1 / 60, options);
    stirrer.stir(fluid, [moving(1, true), moving(2, true)], 1, 1 / 60, options);
    // The colour callback encodes the stroke index in the first channel; the amount is in the last.
    const indices = fluid.dyes.map((d) => d.value[0]! / d.value[3]!);
    expect(new Set(indices).size).toBe(3);

    const hover = recorder();
    const quiet = stirrer.stir(hover, [moving(3, false, false)], 1, 1 / 60, options);
    expect(quiet.active).toBe(false);
    expect(hover.forces).toHaveLength(0);

    const hovering = recorder();
    const active = stirrer.stir(hovering, [moving(3, false, false)], 1, 1 / 60, {
      ...options,
      mode: 'hover and press',
    });
    expect(active.active).toBe(true);
    // Hover pushes at a fraction of a press and paints a faint dye.
    expect(hovering.forces[0]!.velocity[0]!).toBeLessThan(0.5 * 3);
    expect(hovering.dyes[0]!.value[3]!).toBeLessThan(0.2);
    // A touch has no hover.
    const touch = recorder();
    expect(
      stirrer.stir(touch, [{ ...moving(4, false, false), type: 'touch' }], 1, 1 / 60, {
        ...options,
        mode: 'hover and press',
      }).active,
    ).toBe(false);
    // `off` does nothing at all.
    const off = recorder();
    stirrer.stir(off, [moving(1, true)], 1, 1 / 60, { ...options, mode: 'off' });
    expect(off.forces.length + off.dyes.length).toBe(0);
  });

  it('drips while a pressed pointer stays still', () => {
    const stirrer = createFluidStirrer();
    const fluid = recorder();
    stirrer.stir(fluid, [pointer({ justPressed: false })], 1, 1 / 60, options);
    expect(fluid.forces).toHaveLength(0);
    expect(fluid.dyes).toHaveLength(1);
    expect(fluid.dyes[0]!.from).toEqual(fluid.dyes[0]!.to);
    expect(fluid.dyes[0]!.value[3]).toBeGreaterThan(0);
  });

  it('really paints into a fluid: a drag leaves a smooth stroke of dye', async () => {
    const device = await getTestDevice();
    const fluid = create(device, { vorticity: 6, dyeDissipation: 0 });
    const stirrer = createFluidStirrer();
    let x = 0.15;
    for (let frame = 0; frame < 30; frame++) {
      const nextX = x + 0.02;
      stirrer.stir(
        fluid,
        [
          pointer({
            justPressed: frame === 0,
            x: nextX,
            y: 0.5,
            path: [
              { x, y: 0.5 },
              { x: nextX, y: 0.5 },
            ],
          }),
        ],
        1,
        1 / 60,
        { ...options, force: 0, puff: 0, colour: () => [1, 0.5, 0.2, 1] },
      );
      x = nextX;
      await submit(device, (e) => fluid.update(e, 1 / 60));
    }
    const dye = await readDye(device, fluid);
    const stats = blobStats(dye, 3);
    expect(stats.total).toBeGreaterThan(5);
    // The stroke runs from x = 0.15 to 0.75 along the middle row.
    expect(stats.cx).toBeGreaterThan(0.35);
    expect(stats.cx).toBeLessThan(0.55);
    expect(Math.abs(stats.cy - 0.5)).toBeLessThan(0.05);
    // No beads: along the stroke the density never strays far from its mean.
    const row = Math.floor(dye.height / 2);
    const line: number[] = [];
    for (let px = Math.floor(0.3 * dye.width); px < Math.floor(0.6 * dye.width); px++) {
      line.push(dye.at(px, row, 3));
    }
    const mean = line.reduce((a, b) => a + b, 0) / line.length;
    expect(mean).toBeGreaterThan(0.05);
    expect(Math.max(...line)).toBeLessThan(mean * 1.6);
    expect(Math.min(...line)).toBeGreaterThan(mean * 0.5);
    fluid.destroy();
  });
});

// --- Study ---------------------------------------------------------------------------------------------------------------------------
//
// The numbers in src/effects/fluid-dye/notes.md. Skipped unless FLUID_STUDY names a file to write the
// report to:  FLUID_STUDY=/tmp/fluid-study.txt npx vitest run tests/fluid.test.ts -t study

const studyFile = (globalThis as { process?: { env: Record<string, string | undefined> } }).process
  ?.env.FLUID_STUDY;

describe.skipIf(!studyFile)('study', () => {
  const lines: string[] = [];
  const log = (text = '') => lines.push(text);

  async function finish() {
    // A variable specifier keeps the type checker (which has no Node types here) out of it.
    const specifier = 'node:fs';
    const { writeFileSync } = (await import(/* @vite-ignore */ specifier)) as {
      writeFileSync(path: string, text: string): void;
    };
    writeFileSync(studyFile!, lines.join('\n') + '\n');
  }

  async function timedDevice(): Promise<{ device: GPUDevice; timed: boolean }> {
    const adapter = await navigator.gpu.requestAdapter();
    if (adapter?.features.has('timestamp-query')) {
      return {
        device: await adapter.requestDevice({ requiredFeatures: ['timestamp-query'] }),
        timed: true,
      };
    }
    return { device: await getTestDevice(), timed: false };
  }

  it('residual divergence against pressure iterations', async () => {
    const device = await getTestDevice();
    const sources: Record<string, (x: number, y: number) => [number, number]> = {
      'large source (sigma 0.1)': (x, y) => {
        const dx = x - 0.4;
        const dy = y - 0.55;
        const g = Math.exp(-(dx * dx + dy * dy) / 0.02);
        return [dx * g * 4, dy * g * 4];
      },
      'small source (sigma 0.03)': (x, y) => {
        const dx = x - 0.5;
        const dy = y - 0.5;
        const g = Math.exp(-(dx * dx + dy * dy) / 0.0018);
        return [dx * g * 12, dy * g * 12];
      },
    };
    const counts = [4, 8, 16, 32, 64, 128, 256, 512, 1024];
    log('RESIDUAL DIVERGENCE AFTER ONE PROJECTION FROM ZERO (rms divergence after / before)');
    for (const [name, field] of Object.entries(sources)) {
      for (const size of [64, 128]) {
        for (const solver of ['jacobi', 'sor'] as const) {
          const row: string[] = [];
          for (const iterations of counts) {
            const fluid = create(device, { warmStart: false, pressureSolver: solver }, size, 64);
            fluid.setVelocity(field);
            const before = divergence(await readVelocity(device, fluid)).l2;
            await submit(device, (e) => fluid.project(e, iterations));
            row.push((divergence(await readVelocity(device, fluid)).l2 / before).toFixed(3));
            fluid.destroy();
          }
          log(
            `${name}, ${size}^2, ${solver}: ${counts.map((c, i) => `${c}:${row[i]}`).join('  ')}`,
          );
        }
      }
    }
    log();
    log(
      'RESIDUAL IN A RUNNING SIMULATION (warm start; rms divergence after / before, mean of the last 60 of 90 steps, a force circling the box)',
    );
    for (const size of [64, 128]) {
      for (const [solver, iterations, omega, weight] of [
        ['jacobi', 15, 0, 1],
        ['jacobi', 30, 0, 1],
        ['jacobi', 31, 0, 1],
        ['jacobi', 15, 0, 0.8],
        ['jacobi', 30, 0, 0.8],
        ['jacobi', 31, 0, 0.8],
        ['jacobi', 60, 0, 0.8],
        ['jacobi', 120, 0, 0.8],
        ['sor', 30, 0, 0.8],
        ['sor', 60, 0, 0.8],
        ['sor', 120, 0, 0.8],
        ['sor', 30, 1.5, 0.8],
        ['sor', 60, 1.5, 0.8],
      ] as const) {
        const fluid = create(
          device,
          {
            pressureSolver: solver,
            pressureIterations: iterations,
            sorOmega: omega,
            jacobiWeight: weight,
            vorticity: 0,
          },
          size,
          64,
        );
        const ratios: number[] = [];
        let t = 0;
        for (let step = 0; step < 90; step++) {
          const a = 0.5 + 0.3 * Math.cos(t * 3);
          const b = 0.5 + 0.3 * Math.sin(t * 3);
          t += 1 / 60;
          const a2 = 0.5 + 0.3 * Math.cos(t * 3);
          const b2 = 0.5 + 0.3 * Math.sin(t * 3);
          fluid.addForce([a, b], [a2, b2], 0.05, [(a2 - a) * 120, (b2 - b) * 120]);
          await submit(device, (e) => fluid.advance(e, 1));
          if (step >= 30) {
            const raw = await readTexture(device, fluid.divergence, 4);
            const d = new Float32Array(
              raw.buffer.slice(raw.byteOffset, raw.byteOffset + raw.length),
            );
            let sum = 0;
            for (const x of d) sum += x * x;
            const before = Math.sqrt(sum / d.length);
            ratios.push(divergence(await readVelocity(device, fluid)).l2 / Math.max(before, 1e-12));
          }
        }
        const mean = ratios.reduce((x, y) => x + y, 0) / ratios.length;
        log(
          `${size}^2, ${solver} ${iterations}${solver === 'sor' ? (omega ? ` (omega ${omega})` : ' (omega auto)') : ` (weight ${weight})`}: ${mean.toFixed(2)}`,
        );
        fluid.destroy();
      }
    }
    log();
    await finish();
  });

  it('advection schemes: a blob and a disc carried once round a vortex', async () => {
    const device = await getTestDevice();
    log(
      'ADVECTION SCHEMES, one revolution (240 steps at 60 Hz, 0.25 turn/s), velocity grid 64^2, dye 256^2',
    );
    log(
      'Gaussian blob sigma 0.03 (7.7 texels) at radius 0.2, and a hard disc of radius 0.05. peak/sigma/total relative to the start; edge = 10 to 90 percent width of the disc, in texels; over/under = largest overshoot above 1 and undershoot below 0 of the disc.',
    );
    const variants: { name: string; params: Partial<FluidParams> }[] = [
      { name: 'semi-lagrangian, euler', params: { dyeScheme: 'semi-lagrangian', rk2: false } },
      { name: 'semi-lagrangian, rk2', params: { dyeScheme: 'semi-lagrangian' } },
      { name: 'maccormack, rk2', params: { dyeScheme: 'maccormack' } },
      { name: 'bfecc, rk2', params: { dyeScheme: 'bfecc' } },
      { name: 'maccormack, rk2, NO limiter', params: { dyeScheme: 'maccormack', limiter: false } },
      { name: 'bfecc, rk2, NO limiter', params: { dyeScheme: 'bfecc', limiter: false } },
    ];
    for (const variant of variants) {
      const blob = create(device, { dyeOnly: true, ...variant.params }, 64, 256);
      blob.setVelocity(rotation(0.25));
      blob.setDye(gaussian(0.7, 0.5, 0.03));
      const start = blobStats(await readDye(device, blob));
      await submit(device, (e) => blob.advance(e, 240));
      const end = blobStats(await readDye(device, blob));
      blob.destroy();

      const disc = create(device, { dyeOnly: true, ...variant.params }, 64, 256);
      disc.setVelocity(rotation(0.25));
      disc.setDye((x, y) => (Math.hypot(x - 0.7, y - 0.5) < 0.05 ? [1, 1, 1, 1] : [0, 0, 0, 0]));
      await submit(device, (e) => disc.advance(e, 240));
      const dye = await readDye(device, disc);
      disc.destroy();
      const stats = blobStats(dye);
      // The 10-90 percent edge width on the outer side of the disc, along the row through its centre.
      const row = Math.round(stats.cy * dye.height - 0.5);
      const profile: number[] = [];
      for (let x = Math.round(stats.cx * dye.width); x < dye.width; x++)
        profile.push(dye.at(x, row));
      const first = (level: number) => profile.findIndex((v) => v < level);
      const edge = first(0.1) - first(0.9);
      log(
        `${variant.name.padEnd(30)} blob: peak ${(end.peak / start.peak).toFixed(3)} sigma ${(end.sigma / start.sigma).toFixed(3)} total ${(end.total / start.total).toFixed(3)} min ${end.min.toFixed(4)}   disc: peak ${stats.peak.toFixed(3)} edge ${edge} texels over ${Math.max(0, stats.peak - 1).toFixed(3)} under ${Math.max(0, -stats.min).toFixed(3)}`,
      );
    }
    log();
    await finish();
  });

  it('velocity schemes: how long a vortex stays a vortex', async () => {
    const device = await getTestDevice();
    log(
      'VELOCITY SCHEMES, a Gaussian vortex (sigma 0.045) on a 64^2 grid with the dye ignored, 180 steps (3 s), no dissipation; peak vorticity and enstrophy relative to the start, kinetic energy relative to the start',
    );
    for (const vorticity of [0, 12]) {
      for (const scheme of SCHEMES) {
        const fluid = create(
          device,
          { velocityScheme: scheme, vorticity, pressureIterations: 60 },
          64,
          64,
        );
        fluid.setVelocity(vortex(0.02));
        await submit(device, (e) => fluid.project(e, 200));
        const measure = async () => {
          const fields = await readHalf(device, fluid.fields);
          let peak = 0;
          let enstrophy = 0;
          for (let i = 0; i < fields.length; i += 4) {
            peak = Math.max(peak, Math.abs(fields[i + 2]!));
            enstrophy += fields[i + 2]! ** 2;
          }
          return { peak, enstrophy, energy: energy(await readVelocity(device, fluid)) };
        };
        await submit(device, (e) => fluid.resolve(e));
        const start = await measure();
        await submit(device, (e) => fluid.advance(e, 180));
        const end = await measure();
        log(
          `confinement ${String(vorticity).padStart(2)}  ${scheme.padEnd(16)} peak ${(end.peak / start.peak).toFixed(3)} enstrophy ${(end.enstrophy / start.enstrophy).toFixed(3)} energy ${(end.energy / start.energy).toFixed(3)}`,
        );
        fluid.destroy();
      }
    }
    log();
    await finish();
  });

  it('dye total in a stirred box', async () => {
    const device = await getTestDevice();
    log(
      'DYE TOTAL AFTER 72 STEPS OF STIRRING IN A CLOSED BOX (64^2 / 128^2, dye starts as a Gaussian of peak 2), relative to the start, at 12, 36 and 72 steps',
    );
    for (const [solver, iterations] of [
      ['jacobi', 30],
      ['jacobi', 120],
      ['sor', 400],
    ] as const) {
      for (const scheme of SCHEMES) {
        const fluid = create(device, {
          velocityScheme: scheme,
          dyeScheme: scheme,
          vorticity: 8,
          pressureSolver: solver,
          pressureIterations: iterations,
        });
        fluid.setDye(gaussian(0.5, 0.5, 0.08, 2));
        const start = blobStats(await readDye(device, fluid)).total;
        const series: string[] = [];
        for (let k = 0; k < 6; k++) {
          fluid.addForce([0.5, 0.5], [0.95, 0.5], 0.06, [3, 0]);
          fluid.addForce([0.5, 0.5], [0.5, 0.05], 0.06, [0, -3]);
          fluid.addForce([0.5, 0.5], [0.05, 0.95], 0.06, [-2, 2]);
          await submit(device, (e) => fluid.advance(e, 12));
          if (k === 0 || k === 2 || k === 5) {
            series.push((blobStats(await readDye(device, fluid)).total / start).toFixed(3));
          }
        }
        log(
          `${solver} ${String(iterations).padStart(3)} ${scheme.padEnd(16)} ${series.join('  ')}`,
        );
        fluid.destroy();
      }
    }
    log();
    await finish();
  });

  it('cost per step', async () => {
    const { device, timed } = await timedDevice();
    log(
      `COST (Apple Silicon, Metal through Dawn, timestamp queries: ${timed}); milliseconds, median of 7 submits, slope between 4 and 44 steps unless stated`,
    );

    /** GPU time from the first timed pass starting to the last ending. */
    async function span(
      make: (profiler: { pass(label: string): GPURenderPassTimestampWrites }) => {
        run(e: GPUCommandEncoder): void;
        destroy(): void;
      },
    ): Promise<number> {
      const querySet = device.createQuerySet({ type: 'timestamp', count: 64 });
      let next = 0;
      const profiler = {
        pass: () => {
          const begin = next;
          next += 2;
          return { querySet, beginningOfPassWriteIndex: begin, endOfPassWriteIndex: begin + 1 };
        },
      };
      const thing = make(profiler);
      const resolve = device.createBuffer({
        size: 512,
        usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC,
      });
      const read = device.createBuffer({
        size: 512,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
      });
      const encoder = device.createCommandEncoder();
      thing.run(encoder);
      encoder.resolveQuerySet(querySet, 0, 64, resolve, 0);
      encoder.copyBufferToBuffer(resolve, 0, read, 0, 512);
      device.queue.submit([encoder.finish()]);
      await read.mapAsync(GPUMapMode.READ);
      const stamps = new BigUint64Array(read.getMappedRange().slice(0));
      read.unmap();
      thing.destroy();
      const used = Array.from(stamps.slice(0, next)).filter((t) => t > 0n);
      const first = used.reduce((a, b) => (a < b ? a : b));
      const last = used.reduce((a, b) => (a > b ? a : b));
      return Number(last - first) / 1e6;
    }
    const median = async (f: () => Promise<number>) => {
      const samples: number[] = [];
      for (let i = 0; i < 7; i++) samples.push(await f());
      samples.sort((a, b) => a - b);
      return samples[3]!;
    };
    async function perStep(
      resolution: number,
      dyeResolution: number,
      params: Partial<FluidParams>,
    ) {
      const at = (steps: number) =>
        median(() =>
          span((profiler) => {
            const fluid = createFluid({ device, resolution, dyeResolution, profiler, params });
            return { run: (e) => fluid.advance(e, steps), destroy: () => fluid.destroy() };
          }),
        );
      return ((await at(44)) - (await at(4))) / 40;
    }
    const f = (x: number) => x.toFixed(3);

    log(
      'Whole step (defaults: maccormack both, jacobi 30, confinement off), velocity grid / dye grid:',
    );
    for (const [sim, dye] of [
      [128, 512],
      [128, 1024],
      [256, 512],
      [256, 1024],
    ] as const) {
      log(`  ${sim}^2 / ${dye}^2: ${f(await perStep(sim, dye, {}))} ms per step`);
    }
    log('Dye advection alone (dyeOnly), dye grid 512^2 / 1024^2:');
    for (const scheme of SCHEMES) {
      log(
        `  ${scheme.padEnd(16)} ${f(await perStep(128, 512, { dyeOnly: true, dyeScheme: scheme }))} / ${f(await perStep(128, 1024, { dyeOnly: true, dyeScheme: scheme }))}`,
      );
    }
    log(
      'Velocity advection (whole step minus dye, dyeScheme semi-lagrangian), velocity grid 128^2 / 256^2, dye 64^2:',
    );
    for (const scheme of SCHEMES) {
      log(
        `  ${scheme.padEnd(16)} ${f(await perStep(128, 64, { velocityScheme: scheme, dyeScheme: 'semi-lagrangian', pressureIterations: 1 }))} / ${f(await perStep(256, 64, { velocityScheme: scheme, dyeScheme: 'semi-lagrangian', pressureIterations: 1 }))}`,
      );
    }
    log('Pressure iterations (one dispatch each), velocity grid 128^2 / 256^2 / 512^2:');
    for (const solver of ['jacobi', 'sor'] as const) {
      const row: string[] = [];
      for (const size of [128, 256, 512]) {
        const at = (n: number) =>
          median(() =>
            span((profiler) => {
              const fluid = createFluid({
                device,
                resolution: size,
                dyeResolution: 64,
                profiler,
                params: { pressureSolver: solver },
              });
              return { run: (e) => fluid.project(e, n), destroy: () => fluid.destroy() };
            }),
          );
        row.push(f(((await at(210)) - (await at(10))) / 200));
      }
      log(`  ${solver.padEnd(7)} ${row.join(' / ')} ms per iteration`);
    }
    log('Vorticity confinement and viscosity (extra passes), velocity grid 128^2:');
    log(
      `  confinement off ${f(await perStep(128, 64, { pressureIterations: 1, dyeScheme: 'semi-lagrangian' }))}, on ${f(await perStep(128, 64, { pressureIterations: 1, dyeScheme: 'semi-lagrangian', vorticity: 10 }))}, viscosity (12 iterations) ${f(await perStep(128, 64, { pressureIterations: 1, dyeScheme: 'semi-lagrangian', viscosity: 0.001 }))}`,
    );
    log();
    await finish();
  });
});
