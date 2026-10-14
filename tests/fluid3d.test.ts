import { describe, expect, it, vi } from 'vitest';
import {
  computeFluid3dGrid,
  computeScalarGrid,
  createFluid3d,
  fluid3dTextureBytes,
  pointOnRay,
  rayBox,
  type Fluid3d,
  type Fluid3dParams,
} from '../src/gpu/sim/fluid3d.ts';
import { halfToFloat } from '../src/gpu/sim/fluid.ts';
import { expectValid, getTestDevice, readTexture } from './gpu.ts';
import { scalars } from './wgsl-eval.ts';

vi.setConfig({ testTimeout: 120_000 });

// --- Helpers -------------------------------------------------------------------------------------

/** Reads a whole 3D texture back as a flat array of numbers (`channels` per texel, x fastest). */
async function read3d(
  device: GPUDevice,
  texture: GPUTexture,
  format: 'half4' | 'float1',
): Promise<Float32Array> {
  const { width, height, depthOrArrayLayers: depth } = texture;
  const bytesPerTexel = format === 'half4' ? 8 : 4;
  const rowBytes = width * bytesPerTexel;
  const bytesPerRow = Math.ceil(rowBytes / 256) * 256;
  const buffer = device.createBuffer({
    size: bytesPerRow * height * depth,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });
  const encoder = device.createCommandEncoder();
  encoder.copyTextureToBuffer(
    { texture },
    { buffer, bytesPerRow, rowsPerImage: height },
    { width, height, depthOrArrayLayers: depth },
  );
  device.queue.submit([encoder.finish()]);
  await buffer.mapAsync(GPUMapMode.READ);
  const padded = new Uint8Array(buffer.getMappedRange());
  const channels = format === 'half4' ? 4 : 1;
  const out = new Float32Array(width * height * depth * channels);
  const view = new DataView(padded.buffer, padded.byteOffset, padded.byteLength);
  let o = 0;
  for (let k = 0; k < depth; k++) {
    for (let j = 0; j < height; j++) {
      const row = (k * height + j) * bytesPerRow;
      for (let i = 0; i < width * channels; i++) {
        out[o++] =
          format === 'half4'
            ? halfToFloat(view.getUint16(row + i * 2, true))
            : view.getFloat32(row + i * 4, true);
      }
    }
  }
  buffer.unmap();
  buffer.destroy();
  return out;
}

async function submit(device: GPUDevice, record: (encoder: GPUCommandEncoder) => void) {
  const encoder = device.createCommandEncoder();
  record(encoder);
  device.queue.submit([encoder.finish()]);
  await device.queue.onSubmittedWorkDone();
}

function create(
  device: GPUDevice,
  params: Partial<Fluid3dParams> = {},
  resolution = 16,
  scalarResolution = 32,
  size: readonly [number, number, number] = [1, 1, 1],
): Fluid3d {
  return createFluid3d({
    device,
    resolution,
    scalarResolution,
    size,
    params: { velocityDissipation: 0, scalarDissipation: 0, ...params },
  });
}

interface Velocity {
  nx: number;
  ny: number;
  nz: number;
  /** Faces: u(i, j, k) i = 0..nx; v(i, j, k) j = 0..ny; w(i, j, k) k = 0..nz. */
  u(i: number, j: number, k: number): number;
  v(i: number, j: number, k: number): number;
  w(i: number, j: number, k: number): number;
}

async function readVelocity(device: GPUDevice, fluid: Fluid3d): Promise<Velocity> {
  const { width: nx, height: ny, depth: nz, velocitySize } = fluid.grid;
  const data = await read3d(device, fluid.velocity.texture, 'half4');
  const at = (i: number, j: number, k: number, c: number) =>
    data[((k * velocitySize[1] + j) * velocitySize[0] + i) * 4 + c]!;
  return {
    nx,
    ny,
    nz,
    u: (i, j, k) => at(i, j, k, 0),
    v: (i, j, k) => at(i, j, k, 1),
    w: (i, j, k) => at(i, j, k, 2),
  };
}

/** Root-mean-square divergence over the cells, and the largest. */
function divergence(v: Velocity): { l2: number; max: number } {
  let sum = 0;
  let max = 0;
  for (let k = 0; k < v.nz; k++) {
    for (let j = 0; j < v.ny; j++) {
      for (let i = 0; i < v.nx; i++) {
        const d =
          v.u(i + 1, j, k) -
          v.u(i, j, k) +
          v.v(i, j + 1, k) -
          v.v(i, j, k) +
          v.w(i, j, k + 1) -
          v.w(i, j, k);
        sum += d * d;
        max = Math.max(max, Math.abs(d));
      }
    }
  }
  return { l2: Math.sqrt(sum / (v.nx * v.ny * v.nz)), max };
}

/** Sum of u^2 + v^2 + w^2 over all real faces: proportional to the kinetic energy. */
function energy(v: Velocity): number {
  let sum = 0;
  for (let k = 0; k < v.nz; k++)
    for (let j = 0; j < v.ny; j++) for (let i = 0; i <= v.nx; i++) sum += v.u(i, j, k) ** 2;
  for (let k = 0; k < v.nz; k++)
    for (let j = 0; j <= v.ny; j++) for (let i = 0; i < v.nx; i++) sum += v.v(i, j, k) ** 2;
  for (let k = 0; k <= v.nz; k++)
    for (let j = 0; j < v.ny; j++) for (let i = 0; i < v.nx; i++) sum += v.w(i, j, k) ** 2;
  return sum;
}

function allFinite(data: ArrayLike<number>): boolean {
  for (let i = 0; i < data.length; i++) if (!Number.isFinite(data[i])) return false;
  return true;
}

interface Blob {
  total: number;
  /** Centroid in length units. */
  centre: [number, number, number];
  min: number;
  max: number;
}

/** Statistics of channel `c` of the ink. */
async function readInk(
  device: GPUDevice,
  fluid: Fluid3d,
  c = 3,
): Promise<Blob & { data: Float32Array }> {
  const data = await read3d(device, fluid.density.texture, 'half4');
  const [sx, sy, sz] = fluid.grid.scalarSize;
  const size = fluid.grid.size;
  let total = 0;
  let min = Infinity;
  let max = -Infinity;
  const centre: [number, number, number] = [0, 0, 0];
  for (let k = 0; k < sz; k++) {
    for (let j = 0; j < sy; j++) {
      for (let i = 0; i < sx; i++) {
        const v = data[((k * sy + j) * sx + i) * 4 + c]!;
        total += v;
        min = Math.min(min, v);
        max = Math.max(max, v);
        centre[0] += v * ((i + 0.5) / sx) * size[0];
        centre[1] += v * ((j + 0.5) / sy) * size[1];
        centre[2] += v * ((k + 0.5) / sz) * size[2];
      }
    }
  }
  if (total > 0) for (let n = 0; n < 3; n++) centre[n] = centre[n]! / total;
  return { total, centre, min, max, data };
}

const gaussian =
  (cx: number, cy: number, cz: number, sigma: number, peak = 1) =>
  (x: number, y: number, z: number): [number, number, number, number] => {
    const v = peak * Math.exp(-((x - cx) ** 2 + (y - cy) ** 2 + (z - cz) ** 2) / (2 * sigma ** 2));
    return [v, v, v, v];
  };

describe('grid sizes', () => {
  it('rounds every axis to a multiple of 8 and keeps cells cubic', () => {
    const g = computeFluid3dGrid(64, [1, 1.25, 1]);
    // `resolution` counts the cells along the LONGEST side, which is y here.
    expect([g.width, g.height, g.depth]).toEqual([48, 64, 48]);
    expect(g.cellSize).toBeCloseTo(1 / 64, 9);
    const small = computeFluid3dGrid(20, [1, 1, 1]);
    expect(small.width % 8).toBe(0);
    expect(computeFluid3dGrid(1000, [1, 1, 1]).width).toBe(256);
    expect(computeFluid3dGrid(Number.NaN, [1, 1, 1]).width).toBe(64);
  });

  it('sizes the scalar grid in proportion to the box', () => {
    expect(computeScalarGrid(96, [1, 1.25, 1])).toEqual([77, 96, 77]);
    expect(computeScalarGrid(5, [1, 1, 1])).toEqual([16, 16, 16]);
  });

  it('adds up texture memory', () => {
    const mb = fluid3dTextureBytes([64, 64, 64], [96, 96, 96]) / 2 ** 20;
    expect(mb).toBeGreaterThan(20);
    expect(mb).toBeLessThan(80);
  });
});

describe('smoke', () => {
  it('steps with every scheme without a validation error', async () => {
    const device = await getTestDevice();
    await expectValid(device, async () => {
      for (const params of [
        {},
        { pressureSolver: 'jacobi' as const, pressureIterations: 4 },
        { velocityScheme: 'maccormack' as const, vorticity: 3, buoyancy: 1 },
        {
          scalarScheme: 'semi-lagrangian' as const,
          obstacle: { center: [0.5, 0.5, 0.5] as const, radius: 0.15 },
        },
      ]) {
        const fluid = create(device, params);
        fluid.addDensity([0.5, 0.7, 0.5], [0.5, 0.7, 0.5], 0.1, [1, 2, 3, 1]);
        fluid.addImpulse([0.5, 0.5, 0.5], [0.6, 0.5, 0.5], 0.1, [1, 0, 0]);
        await submit(device, (encoder) => fluid.advance(encoder, 3));
        fluid.destroy();
      }
    });
  });
});

// --- Physics -------------------------------------------------------------------------------------------

/** A divergent velocity: an outward Gaussian push from the middle of the box. */
const outward =
  (sigma: number, amp = 1) =>
  (x: number, y: number, z: number): [number, number, number] => {
    const dx = x - 0.5;
    const dy = y - 0.5;
    const dz = z - 0.5;
    const g = amp * Math.exp(-(dx * dx + dy * dy + dz * dz) / (2 * sigma * sigma));
    return [dx * g, dy * g, dz * g];
  };

describe('at rest', () => {
  it('a fluid at rest stays at rest and keeps its ink', async () => {
    const device = await getTestDevice();
    const fluid = create(device);
    fluid.setDensity(gaussian(0.5, 0.5, 0.5, 0.12));
    const before = await readInk(device, fluid);
    await submit(device, (encoder) => fluid.advance(encoder, 40));
    const v = await readVelocity(device, fluid);
    expect(energy(v)).toBeLessThan(1e-9);
    const after = await readInk(device, fluid);
    expect(after.total).toBeCloseTo(before.total, 1);
    for (let n = 0; n < 3; n++) expect(after.centre[n]).toBeCloseTo(before.centre[n]!, 3);
    fluid.destroy();
  });

  it('keeps a uniform ink at rest too, whatever the gravity (a body force is balanced by pressure)', async () => {
    const device = await getTestDevice();
    const fluid = create(device, { gravity: [0, -3, 0] });
    fluid.setDensity(() => [0, 0, 0, 0]);
    await submit(device, (encoder) => fluid.advance(encoder, 30));
    const v = await readVelocity(device, fluid);
    // The pressure cancels a uniform force to within the solve's accuracy.
    expect(energy(v)).toBeLessThan(1e-3);
    fluid.destroy();
  });
});

describe('projection', () => {
  async function reduction(params: Partial<Fluid3dParams>, iterations: number, resolution = 32) {
    const device = await getTestDevice();
    const fluid = create(device, { warmStart: false, ...params }, resolution);
    fluid.setVelocity(outward(0.12, 6));
    const before = divergence(await readVelocity(device, fluid));
    await submit(device, (encoder) => fluid.project(encoder, iterations));
    const after = divergence(await readVelocity(device, fluid));
    fluid.destroy();
    return { before: before.l2, after: after.l2, ratio: after.l2 / before.l2 };
  }

  it('one multigrid V-cycle removes most of the divergence of a divergent field', async () => {
    const one = await reduction({ pressureSolver: 'multigrid' }, 1);
    expect(one.before).toBeGreaterThan(0.01);
    // Measured: a quarter is left after one V(2,2) cycle on this field from a cold start (the
    // smoothing factor of point Jacobi in 3D is 5/7, so a cycle gains a factor of 4, not 10).
    expect(one.ratio).toBeLessThan(0.35);
  });

  it('more V-cycles reduce it further, and a cycle beats the same work in plain sweeps', async () => {
    const one = await reduction({ pressureSolver: 'multigrid' }, 1);
    const three = await reduction({ pressureSolver: 'multigrid' }, 3);
    // Three cycles leave about a sixtieth (the convergence is geometric).
    expect(three.ratio).toBeLessThan(one.ratio * 0.2);
    expect(three.ratio).toBeLessThan(0.03);
    // A cycle at 32 cells costs about 6 fine-grid sweeps' worth of dispatches.
    const sweeps = await reduction({ pressureSolver: 'jacobi' }, 6);
    expect(one.ratio).toBeLessThan(sweeps.ratio * 0.5);
  });

  it('plain Jacobi reduces the divergence monotonically with more sweeps', async () => {
    const ratios: number[] = [];
    for (const n of [4, 16, 64])
      ratios.push((await reduction({ pressureSolver: 'jacobi' }, n)).ratio);
    expect(ratios[0]!).toBeLessThan(1);
    expect(ratios[1]!).toBeLessThan(ratios[0]!);
    expect(ratios[2]!).toBeLessThan(ratios[1]!);
  });

  it('a warm start continues from where the last solve stopped', async () => {
    const device = await getTestDevice();
    const fluid = create(device, { pressureSolver: 'jacobi' });
    fluid.setVelocity(outward(0.12, 6));
    const d0 = divergence(await readVelocity(device, fluid)).l2;
    // The same field projected twice: the second projection starts from the first one's pressure and
    // works on what is left, so the total only improves.
    await submit(device, (encoder) => fluid.project(encoder, 8));
    const d1 = divergence(await readVelocity(device, fluid)).l2;
    await submit(device, (encoder) => fluid.project(encoder, 8));
    const d2 = divergence(await readVelocity(device, fluid)).l2;
    expect(d1).toBeLessThan(d0);
    expect(d2).toBeLessThan(d1);
    fluid.destroy();
  });

  it('works on a box that is not a cube', async () => {
    const device = await getTestDevice();
    const fluid = create(device, { warmStart: false }, 32, 32, [0.5, 1, 0.75]);
    expect([fluid.grid.width, fluid.grid.height, fluid.grid.depth]).toEqual([16, 32, 24]);
    fluid.setVelocity(outward(0.12, 6));
    const before = divergence(await readVelocity(device, fluid)).l2;
    await submit(device, (encoder) => fluid.project(encoder, 2));
    const after = divergence(await readVelocity(device, fluid)).l2;
    expect(after).toBeLessThan(before * 0.2);
    fluid.destroy();
  });
});

describe('advection', () => {
  const axes = [
    { name: 'x', velocity: [0.3, 0, 0], axis: 0 },
    { name: 'y', velocity: [0, 0.3, 0], axis: 1 },
    { name: 'z', velocity: [0, 0, -0.3], axis: 2 },
  ] as const;

  it.each(axes)(
    'a uniform velocity along $name carries a blob the expected distance',
    async ({ velocity, axis }) => {
      const device = await getTestDevice();
      const fluid = create(device, { scalarOnly: true });
      fluid.setVelocity(() => velocity);
      fluid.setDensity(gaussian(0.5, 0.5, 0.5, 0.08));
      const before = await readInk(device, fluid);
      const steps = 30;
      await submit(device, (encoder) => fluid.advance(encoder, steps));
      const after = await readInk(device, fluid);
      const expected = velocity[axis] * steps * fluid.stepSeconds;
      expect(after.centre[axis]! - before.centre[axis]!).toBeCloseTo(expected, 2);
      for (let n = 0; n < 3; n++) {
        if (n !== axis) expect(after.centre[n]).toBeCloseTo(before.centre[n]!, 3);
      }
      // MacCormack keeps the blob's mass and never overshoots it.
      expect(after.total / before.total).toBeGreaterThan(0.95);
      expect(after.total / before.total).toBeLessThan(1.05);
      expect(after.max).toBeLessThanOrEqual(before.max * 1.001);
      expect(after.min).toBeGreaterThanOrEqual(0);
      fluid.destroy();
    },
  );
});

describe('buoyancy', () => {
  async function drift(buoyancy: number) {
    const device = await getTestDevice();
    const fluid = create(device, { buoyancy }, 24, 32);
    fluid.setDensity(gaussian(0.5, 0.5, 0.5, 0.1, 1));
    const before = await readInk(device, fluid);
    await submit(device, (encoder) => fluid.advance(encoder, 60));
    const after = await readInk(device, fluid);
    fluid.destroy();
    return {
      dy: after.centre[1]! - before.centre[1]!,
      dx: after.centre[0]! - before.centre[0]!,
      dz: after.centre[2]! - before.centre[2]!,
    };
  }

  it('moves a dense blob along gravity and a light one against it', async () => {
    const dense = await drift(8);
    const light = await drift(-8);
    expect(dense.dy).toBeLessThan(-0.02);
    expect(light.dy).toBeGreaterThan(0.02);
    expect(Math.abs(dense.dx)).toBeLessThan(0.01);
    expect(Math.abs(dense.dz)).toBeLessThan(0.01);
    // Symmetric: the same size of drift either way, to within the asymmetry of the wall at the top and bottom.
    expect(Math.abs(dense.dy + light.dy)).toBeLessThan(0.3 * Math.abs(dense.dy));
  });

  it('follows the buoyancy direction', async () => {
    const device = await getTestDevice();
    const fluid = create(device, { buoyancy: 8, buoyancyDirection: [1, 0, 0] }, 24, 32);
    fluid.setDensity(gaussian(0.4, 0.5, 0.5, 0.1, 1));
    const before = await readInk(device, fluid);
    await submit(device, (encoder) => fluid.advance(encoder, 60));
    const after = await readInk(device, fluid);
    expect(after.centre[0]! - before.centre[0]!).toBeGreaterThan(0.02);
    expect(Math.abs(after.centre[1]! - before.centre[1]!)).toBeLessThan(0.01);
    fluid.destroy();
  });
});

describe('walls', () => {
  it('leave zero normal velocity at the six faces and keep the ink in', async () => {
    const device = await getTestDevice();
    const fluid = create(device, { vorticity: 4, buoyancy: 4, scalarDissipation: 0 }, 24, 32);
    fluid.setDensity(gaussian(0.5, 0.3, 0.5, 0.15, 2));
    const before = await readInk(device, fluid);
    // Shove the fluid hard towards every wall.
    const points: [number, number, number][] = [
      [0.9, 0.5, 0.5],
      [0.1, 0.5, 0.5],
      [0.5, 0.9, 0.5],
      [0.5, 0.1, 0.5],
      [0.5, 0.5, 0.9],
      [0.5, 0.5, 0.1],
    ];
    const directions: [number, number, number][] = [
      [3, 0, 0],
      [-3, 0, 0],
      [0, 3, 0],
      [0, -3, 0],
      [0, 0, 3],
      [0, 0, -3],
    ];
    points.forEach((p, i) => fluid.addImpulse(p, p, 0.12, directions[i]!));
    await submit(device, (encoder) => fluid.advance(encoder, 60));
    const v = await readVelocity(device, fluid);
    let worst = 0;
    for (let k = 0; k < v.nz; k++) {
      for (let j = 0; j < v.ny; j++)
        worst = Math.max(worst, Math.abs(v.u(0, j, k)), Math.abs(v.u(v.nx, j, k)));
    }
    for (let k = 0; k < v.nz; k++) {
      for (let i = 0; i < v.nx; i++)
        worst = Math.max(worst, Math.abs(v.v(i, 0, k)), Math.abs(v.v(i, v.ny, k)));
    }
    for (let j = 0; j < v.ny; j++) {
      for (let i = 0; i < v.nx; i++)
        worst = Math.max(worst, Math.abs(v.w(i, j, 0)), Math.abs(v.w(i, j, v.nz)));
    }
    expect(worst).toBe(0);
    expect(energy(v)).toBeGreaterThan(1e-3); // the fluid did move
    const after = await readInk(device, fluid);
    expect(after.total / before.total).toBeGreaterThan(0.8);
    expect(after.total / before.total).toBeLessThan(1.25);
    expect(after.min).toBeGreaterThanOrEqual(0);
    fluid.destroy();
  });

  it('stop ink at a wall: a long push into it leaves the ink inside', async () => {
    const device = await getTestDevice();
    const fluid = create(device, { scalarOnly: true });
    fluid.setVelocity(() => [0.5, 0, 0]);
    fluid.setDensity(gaussian(0.7, 0.5, 0.5, 0.06));
    const before = await readInk(device, fluid);
    await submit(device, (encoder) => fluid.advance(encoder, 120));
    const after = await readInk(device, fluid);
    // The prescribed flow converges on the wall, and a passive scalar is copied, not accumulated, so
    // the total may fall (it cannot rise past the limiter) but the ink is still inside the box.
    expect(after.total / before.total).toBeLessThan(1.05);
    expect(after.total).toBeGreaterThan(0);
    expect(after.centre[0]!).toBeGreaterThan(0.8);
    expect(after.centre[0]!).toBeLessThan(1);
    fluid.destroy();
  });
});

describe('energy and obstacles', () => {
  it('does not gain kinetic energy without forcing (confinement off)', async () => {
    const device = await getTestDevice();
    const fluid = create(device, { pressureIterations: 2 }, 24, 32);
    fluid.setVelocity(outward(0.15, 4));
    // One projection first: the field the solver would actually hold.
    await submit(device, (encoder) => fluid.project(encoder, 3));
    let previous = energy(await readVelocity(device, fluid));
    expect(previous).toBeGreaterThan(0);
    for (let round = 0; round < 6; round++) {
      await submit(device, (encoder) => fluid.advance(encoder, 10));
      const now = energy(await readVelocity(device, fluid));
      expect(now).toBeLessThan(previous * 1.02);
      previous = now;
    }
    fluid.destroy();
  });

  it('flows round a solid sphere: no ink inside it, no velocity through it', async () => {
    const device = await getTestDevice();
    const fluid = create(device, {
      pressureSolver: 'jacobi',
      pressureIterations: 20,
      obstacle: { center: [0.5, 0.5, 0.5], radius: 0.15 },
    });
    fluid.setDensity(gaussian(0.5, 0.5, 0.5, 0.2, 1));
    fluid.addImpulse([0.2, 0.5, 0.5], [0.3, 0.5, 0.5], 0.1, [2, 0, 0]);
    await submit(device, (encoder) => fluid.advance(encoder, 30));
    const ink = await readInk(device, fluid);
    const [sx, sy, sz] = fluid.grid.scalarSize;
    let inside = 0;
    for (let k = 0; k < sz; k++) {
      for (let j = 0; j < sy; j++) {
        for (let i = 0; i < sx; i++) {
          const d = Math.hypot((i + 0.5) / sx - 0.5, (j + 0.5) / sy - 0.5, (k + 0.5) / sz - 0.5);
          if (d < 0.14) inside += ink.data[((k * sy + j) * sx + i) * 4 + 3]!;
        }
      }
    }
    expect(inside).toBe(0);
    const v = await readVelocity(device, fluid);
    // The face just inside the sphere on the -x side of the centre cell carries nothing.
    const mid = fluid.grid.width / 2;
    expect(v.u(mid, fluid.grid.height / 2, fluid.grid.depth / 2)).toBe(0);
    expect(energy(v)).toBeGreaterThan(0);
    fluid.destroy();
  });
});

describe('robustness', () => {
  it('never produces NaN or Inf from extreme impulses and many sub-steps', async () => {
    const device = await getTestDevice();
    const fluid = create(
      device,
      { vorticity: 30, buoyancy: 20, velocityScheme: 'maccormack' },
      16,
      24,
    );
    for (let i = 0; i < 128; i++) {
      const p: [number, number, number] = [Math.random(), Math.random(), Math.random()];
      fluid.addImpulse(p, [p[0], p[1], 1 - p[2]], 0.1, [3e38, -3e38, 1e30]);
      fluid.addDensity(p, p, 0.1, [3e38, 5, -5, 1e30]);
    }
    // Things that are not numbers are refused outright.
    expect(fluid.addImpulse([Number.NaN, 0, 0], [0, 0, 0], 0.1, [1, 0, 0])).toBe(false);
    expect(fluid.addDensity([0, 0, 0], [0, 0, 0], Number.POSITIVE_INFINITY, [1, 1, 1, 1])).toBe(
      false,
    );
    await submit(device, (encoder) => fluid.advance(encoder, 90));
    const v = await read3d(device, fluid.velocity.texture, 'half4');
    const ink = await read3d(device, fluid.density.texture, 'half4');
    expect(allFinite(v)).toBe(true);
    expect(allFinite(ink)).toBe(true);
    const bound = (6 * fluid.grid.cellSize) / fluid.stepSeconds;
    for (let i = 0; i < v.length; i++) expect(Math.abs(v[i]!)).toBeLessThanOrEqual(bound * 1.001);
    fluid.destroy();
  });

  it('refuses splats beyond its queue instead of overflowing', () => {
    return getTestDevice().then((device) => {
      const fluid = create(device);
      let accepted = 0;
      for (let i = 0; i < 400; i++) {
        if (fluid.addImpulse([0.5, 0.5, 0.5], [0.5, 0.5, 0.5], 0.1, [1, 0, 0])) accepted++;
      }
      expect(accepted).toBeLessThan(400);
      expect(fluid.pendingSplats[0]).toBe(accepted);
      fluid.destroy();
    });
  });

  it('looks the same at 60 and 120 Hz and after a stalled frame', async () => {
    const device = await getTestDevice();
    const run = async (frames: number, delta: number) => {
      const fluid = create(device, { buoyancy: 6 }, 16, 24);
      fluid.setDensity(gaussian(0.5, 0.6, 0.5, 0.1, 1));
      for (let i = 0; i < frames; i++)
        await submit(device, (encoder) => fluid.update(encoder, delta));
      const ink = await readInk(device, fluid);
      fluid.destroy();
      return ink.centre[1]!;
    };
    const at60 = await run(60, 1 / 60);
    const at120 = await run(120, 1 / 120);
    expect(at120).toBeCloseTo(at60, 2);
  });

  it('rebuilds cleanly when the resolution, the box or the workgroup changes, and on reset', async () => {
    const device = await getTestDevice();
    await expectValid(device, async () => {
      const fluid = create(device, { vorticity: 2 });
      fluid.addDensity([0.5, 0.5, 0.5], [0.5, 0.5, 0.5], 0.1, [1, 1, 1, 1]);
      await submit(device, (encoder) => fluid.advance(encoder, 2));
      const g0 = fluid.generation;
      expect(fluid.setResolution(16, 32)).toBe(false);
      expect(fluid.generation).toBe(g0);
      expect(fluid.setResolution(24, 40)).toBe(true);
      expect(fluid.generation).toBe(g0 + 1);
      expect(fluid.grid.width).toBe(24);
      expect(fluid.grid.scalarSize[0]).toBe(40);
      // A re-created grid is empty.
      expect((await readInk(device, fluid)).total).toBe(0);
      expect(fluid.setSize([1, 1.5, 1])).toBe(true);
      expect(fluid.grid.height).toBeGreaterThan(fluid.grid.width);
      fluid.addDensity([0.3, 0.3, 0.3], [0.3, 0.3, 0.3], 0.1, [1, 1, 1, 1]);
      await submit(device, (encoder) => fluid.advance(encoder, 2));
      expect((await readInk(device, fluid)).total).toBeGreaterThan(0);
      for (const wg of [
        [8, 4, 4],
        [8, 8, 4],
        [16, 4, 4],
        [4, 4, 4],
      ] as const) {
        fluid.setWorkgroup(wg);
        await submit(device, (encoder) => fluid.advance(encoder, 1));
      }
      expect(() => fluid.setWorkgroup([16, 16, 2])).toThrow(RangeError);
      const g1 = fluid.generation;
      fluid.reset();
      expect(fluid.generation).toBe(g1 + 1);
      expect((await readInk(device, fluid)).total).toBe(0);
      await submit(device, (encoder) => fluid.advance(encoder, 2));
      fluid.destroy();
    });
  });

  it('gives the same answer in every workgroup shape', async () => {
    const device = await getTestDevice();
    const results: Float32Array[] = [];
    for (const wg of [
      [4, 4, 4],
      [8, 4, 4],
      [8, 8, 4],
    ] as const) {
      const fluid = createFluid3d({
        device,
        resolution: 16,
        scalarResolution: 24,
        workgroup: wg,
        params: { buoyancy: 6, vorticity: 3 },
      });
      fluid.setDensity(gaussian(0.5, 0.6, 0.5, 0.12, 1));
      fluid.addImpulse([0.5, 0.5, 0.5], [0.6, 0.5, 0.5], 0.1, [1, 0, 0]);
      await submit(device, (encoder) => fluid.advance(encoder, 10));
      results.push((await readInk(device, fluid)).data);
      fluid.destroy();
    }
    expect(results[1]).toEqual(results[0]);
    expect(results[2]).toEqual(results[0]);
  });
});

describe('resolve', () => {
  it('writes the cell-centred velocity the debug views read', async () => {
    const device = await getTestDevice();
    const fluid = create(device);
    fluid.setVelocity((x, y) => [0.3, 0.1 * x, y * 0]);
    await submit(device, (encoder) => fluid.resolve(encoder));
    const data = await read3d(device, fluid.fields, 'half4');
    const { width, height } = fluid.grid;
    const at = (i: number, j: number, k: number, c: number) =>
      data[((k * height + j) * width + i) * 4 + c]!;
    expect(at(8, 8, 8, 0)).toBeCloseTo(0.3, 2);
    expect(at(8, 8, 8, 1)).toBeCloseTo(0.1 * (8.5 / width), 2);
    expect(at(8, 8, 8, 2)).toBeCloseTo(0, 3);
    fluid.destroy();
  });
});

describe('one command buffer', () => {
  it('still injects the queued splats when resolve and project are encoded after the update', async () => {
    // queue.writeBuffer lands before the whole command buffer: a second parameter upload in the same
    // encoder must not zero the splat counts the injection pass reads.
    const device = await getTestDevice();
    const fluid = create(device, { buoyancy: 4 });
    fluid.addDensity([0.5, 0.5, 0.5], [0.5, 0.5, 0.5], 0.1, [1, 1, 1, 1]);
    fluid.addImpulse([0.5, 0.5, 0.5], [0.5, 0.5, 0.5], 0.1, [0, 1, 0]);
    await submit(device, (encoder) => {
      fluid.advance(encoder, 1);
      fluid.resolve(encoder);
      fluid.project(encoder, 1);
    });
    expect((await readInk(device, fluid)).total).toBeGreaterThan(0.1);
    expect(energy(await readVelocity(device, fluid))).toBeGreaterThan(1e-4);
    fluid.destroy();
  });
});

describe('geometry helpers', () => {
  it('rayBox finds entry and exit, clamps to the origin, and reports a miss', () => {
    expect(rayBox([-1, 0.5, 0.5], [1, 0, 0], [0, 0, 0], [1, 1, 1])).toEqual({ near: 1, far: 2 });
    expect(rayBox([0.5, 0.5, 0.5], [0, 0, 1], [0, 0, 0], [1, 1, 1])).toEqual({ near: 0, far: 0.5 });
    expect(rayBox([-1, 2, 0.5], [1, 0, 0], [0, 0, 0], [1, 1, 1])).toBeNull();
    expect(rayBox([2, 0.5, 0.5], [1, 0, 0], [0, 0, 0], [1, 1, 1])).toBeNull();
    const diagonal = rayBox([-1, -1, -1], [1, 1, 1], [0, 0, 0], [1, 1, 1])!;
    expect(diagonal.near).toBeCloseTo(1, 9);
    expect(diagonal.far).toBeCloseTo(2, 9);
  });

  it('pointOnRay picks a point between the entry and the exit', () => {
    expect(pointOnRay([-1, 0.5, 0.5], [1, 0, 0], [1, 1, 1], 0)).toEqual([0, 0.5, 0.5]);
    expect(pointOnRay([-1, 0.5, 0.5], [1, 0, 0], [1, 1, 1], 0.5)).toEqual([0.5, 0.5, 0.5]);
    expect(pointOnRay([-1, 2, 0.5], [1, 0, 0], [1, 1, 1])).toBeNull();
  });
});

describe('volume.wgsl', () => {
  const header = '#include "volume.wgsl"\n';

  it('volume_ray_box agrees with the CPU twin on known cases and misses cleanly', async () => {
    const cases: [number[], number[]][] = [
      [
        [-1, 0.5, 0.5],
        [1, 0, 0],
      ],
      [
        [0.5, 0.5, 0.5],
        [0, 0, 1],
      ],
      [
        [-1, 2, 0.5],
        [1, 0, 0],
      ],
      [
        [2, 0.5, 0.5],
        [1, 0, 0],
      ],
      [
        [-1, -1, -1],
        [Math.SQRT1_2 * 0.8164965809277261, Math.SQRT1_2 * 0.8164965809277261, 0.5773502691896258],
      ],
      [
        [0.5, 3, 0.5],
        [0, -1, 0],
      ],
    ];
    const expressions = cases.flatMap(([o, d]) => {
      const call = `volume_ray_box(vec3f(${o.join(',')}), vec3f(${d.join(',')}), vec3f(0.0), vec3f(1.0))`;
      return [`${call}.x`, `${call}.y`];
    });
    const out = await scalars(header, expressions);
    cases.forEach(([o, d], i) => {
      const cpu = rayBox(
        o as [number, number, number],
        d as [number, number, number],
        [0, 0, 0],
        [1, 1, 1],
      );
      const near = out[i * 2]!;
      const far = out[i * 2 + 1]!;
      if (cpu) {
        expect(far).toBeGreaterThan(near);
        expect(near).toBeCloseTo(cpu.near, 4);
        expect(far).toBeCloseTo(cpu.far, 4);
      } else {
        expect(far).toBeLessThanOrEqual(near); // volume_hit is false
      }
    });
    const hit = await scalars(header, [
      'select(0.0, 1.0, volume_hit(volume_ray_box(vec3f(-1.0, 0.5, 0.5), vec3f(1.0, 0.0, 0.0), vec3f(0.0), vec3f(1.0))))',
      'select(0.0, 1.0, volume_hit(volume_ray_box(vec3f(-1.0, 2.0, 0.5), vec3f(1.0, 0.0, 0.0), vec3f(0.0), vec3f(1.0))))',
      'select(0.0, 1.0, volume_hit(volume_ray_box(vec3f(-1.0, 0.5, 0.5), vec3f(-1.0, 0.0, 0.0), vec3f(0.0), vec3f(1.0))))',
    ]);
    expect(hit).toEqual([1, 0, 0]); // the third looks away from the box
  });

  it('a uniform slab leaves the analytic transmittance, however finely it is stepped', async () => {
    const sigma = 1.7;
    const thickness = 0.8;
    const analytic = Math.exp(-sigma * thickness);
    const stepped = (n: number) => `
      volume_march_test(${n})`;
    const code = `${header}
      fn volume_march_test(n: i32) -> f32 {
        var acc = volume_start();
        let ds = ${thickness} / f32(n);
        for (var i = 0; i < n; i++) {
          acc = volume_accumulate(acc, vec3f(${sigma}), vec3f(0.0), ds);
        }
        return acc.transmittance.y;
      }`;
    const out = await scalars(code, [
      `volume_slab_transmittance(vec3f(${sigma}), ${thickness}).x`,
      stepped(1),
      stepped(4),
      stepped(64),
      'volume_slab_transmittance(vec3f(1.0), 0.0).x',
      'volume_slab_transmittance(vec3f(1.0), -3.0).x',
    ]);
    expect(out[0]).toBeCloseTo(analytic, 5);
    // The step integral is exact for a constant medium, so the step count does not matter.
    for (const n of [1, 2, 3]) expect(out[n]).toBeCloseTo(analytic, 5);
    expect(out[4]).toBe(1);
    expect(out[5]).toBe(1);
  });

  it('a uniform glowing slab gathers the analytic radiance', async () => {
    // L = S / sigma * (1 - exp(-sigma d)) for in-scatter S and extinction sigma.
    const sigma = 2;
    const source = 0.6;
    const d = 0.9;
    const analytic = (source / sigma) * (1 - Math.exp(-sigma * d));
    const code = `${header}
      fn glow(n: i32) -> f32 {
        var acc = volume_start();
        for (var i = 0; i < n; i++) {
          acc = volume_accumulate(acc, vec3f(${sigma}), vec3f(${source}), ${d} / f32(n));
        }
        return acc.radiance.z;
      }
      fn clear_medium(n: i32) -> f32 {
        var acc = volume_start();
        for (var i = 0; i < n; i++) {
          acc = volume_accumulate(acc, vec3f(0.0), vec3f(${source}), ${d} / f32(n));
        }
        return acc.radiance.x;
      }`;
    const out = await scalars(code, ['glow(1)', 'glow(8)', 'clear_medium(5)']);
    expect(out[0]).toBeCloseTo(analytic, 5);
    expect(out[1]).toBeCloseTo(analytic, 5);
    expect(out[2]).toBeCloseTo(source * d, 5); // no extinction: radiance is just source times length
  });

  it('volume_opaque, the Henyey-Greenstein phase and the jitter behave', async () => {
    const out = await scalars(header, [
      'select(0.0, 1.0, volume_opaque(VolumeAccum(vec3f(0.0), vec3f(0.001, 0.5, 0.0)), 0.01))',
      'select(0.0, 1.0, volume_opaque(VolumeAccum(vec3f(0.0), vec3f(0.001, 0.005, 0.0)), 0.01))',
      'volume_phase_hg(0.3, 0.0) * 12.566370614',
      'volume_phase_hg(1.0, 0.7)',
      'volume_phase_hg(-1.0, 0.7)',
      'volume_jitter(vec2f(13.0, 7.0), 0u)',
      'volume_jitter(vec2f(13.0, 7.0), 5u)',
    ]);
    expect(out.slice(0, 2)).toEqual([0, 1]);
    expect(out[2]).toBeCloseTo(1, 5); // isotropic
    expect(out[3]!).toBeGreaterThan(out[4]! * 20); // forward scattering is strong
    for (const j of out.slice(5)) {
      expect(j).toBeGreaterThanOrEqual(0);
      expect(j).toBeLessThan(1);
    }
    expect(out[5]).not.toBe(out[6]);
  });

  it('the phase function integrates to 1 over the sphere', async () => {
    // Integral over cos from -1 to 1 times 2 pi.
    const n = 4000;
    const code = `${header}
      fn total(g: f32) -> f32 {
        var sum = 0.0;
        for (var i = 0; i < ${n}; i++) {
          let c = -1.0 + (f32(i) + 0.5) * 2.0 / ${n}.0;
          sum += volume_phase_hg(c, g);
        }
        return sum * 2.0 / ${n}.0 * 6.283185307;
      }`;
    const out = await scalars(code, ['total(0.0)', 'total(0.6)', 'total(-0.4)']);
    for (const v of out) expect(v).toBeCloseTo(1, 2);
  });
});

// --- The effect -------------------------------------------------------------------------------------------------

describe('fluid-3d effect', () => {
  const WIDTH = 96;
  const HEIGHT = 72;

  function fakeCanvas(touchAction = 'pan-y') {
    return Object.assign(new EventTarget(), {
      width: WIDTH,
      height: HEIGHT,
      style: { touchAction },
      getBoundingClientRect: () => ({ left: 0, top: 0, width: WIDTH, height: HEIGHT }),
      setPointerCapture() {},
      releasePointerCapture() {},
    }) as unknown as HTMLCanvasElement & { style: { touchAction: string } };
  }

  function pointer(type: string, x: number, y: number, down: boolean) {
    return Object.assign(new Event(type), {
      clientX: x,
      clientY: y,
      pointerId: 1,
      pointerType: 'mouse',
      pressure: down ? 0.5 : 0,
    });
  }

  async function run(
    overrides: Record<string, unknown>,
    script: (frame: number, canvas: ReturnType<typeof fakeCanvas>) => void = () => {},
    frames = 6,
  ) {
    const { loadEffect } = await import('../src/effects/registry.ts');
    const { defaultParams } = await import('../src/effects/types.ts');
    const { noProfiler } = await import('../src/gpu/profiler.ts');
    const device = await getTestDevice();
    const effect = await loadEffect('fluid-3d');
    const target = device.createTexture({
      size: [WIDTH, HEIGHT],
      format: 'rgba8unorm',
      usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC,
    });
    const canvas = fakeCanvas();
    const pixels = await expectValid(device, async () => {
      const instance = await effect.create({
        device,
        format: 'rgba8unorm',
        canvas,
        params: { ...defaultParams(effect.params), ...overrides } as never,
        profiler: noProfiler,
        hdr: false,
      });
      instance.resize(WIDTH, HEIGHT, 1);
      for (let frame = 0; frame < frames; frame++) {
        script(frame, canvas);
        const encoder = device.createCommandEncoder();
        instance.frame({ time: frame / 60, delta: 1 / 60, frame }, encoder, target.createView());
        device.queue.submit([encoder.finish()]);
      }
      const touchDuring = canvas.style.touchAction;
      instance.destroy();
      return { image: await readTexture(device, target), touchDuring };
    });
    return { ...pixels, canvas, device };
  }

  it('every preset runs headlessly through a tap, a drag and an orbit', async () => {
    const { loadEffect } = await import('../src/effects/registry.ts');
    const effect = await loadEffect('fluid-3d');
    const names = Object.keys(effect.presets ?? {});
    expect(names.length).toBeGreaterThanOrEqual(5);
    for (const name of names) {
      const { image } = await run(
        effect.presets![name]! as Record<string, unknown>,
        (frame, canvas) => {
          // Frame 1 taps the tank, 3 to 5 drag across it, and 7 presses the empty corner and orbits.
          if (frame === 1) canvas.dispatchEvent(pointer('pointerdown', 48, 36, true));
          if (frame === 2) canvas.dispatchEvent(pointer('pointerup', 48, 36, false));
          if (frame === 3) canvas.dispatchEvent(pointer('pointerdown', 40, 30, true));
          if (frame === 4) canvas.dispatchEvent(pointer('pointermove', 52, 36, true));
          if (frame === 5) canvas.dispatchEvent(pointer('pointermove', 60, 40, true));
          if (frame === 6) canvas.dispatchEvent(pointer('pointerup', 60, 40, false));
          if (frame === 7) canvas.dispatchEvent(pointer('pointerdown', 4, 4, true));
          if (frame === 8) canvas.dispatchEvent(pointer('pointermove', 20, 10, true));
          if (frame === 9) canvas.dispatchEvent(pointer('pointerup', 20, 10, false));
        },
        11,
      );
      expect(new Set(image).size, `preset ${name} drew something`).toBeGreaterThan(8);
    }
  });

  it('draws the debug views and the obstacle', async () => {
    for (const overrides of [
      { view: 'velocity slice' },
      { view: 'ink slice' },
      { obstacle: 0.12 },
      { lighting: 'per-sample march', skipEmpty: false, jitter: false },
      { lighting: 'off', marchRepeat: 3, renderScale: 1 },
      { drag: 'orbits the camera', pressureSolver: 'jacobi' },
      // The largest grids and the most steps the controls allow.
      {
        resolution: '128',
        scalarResolution: '160',
        steps: 256,
        renderScale: 1,
        pressureIterations: 12,
        vorticity: 24,
        buoyancy: 6,
      },
    ]) {
      // Forty frames: enough for the seeded drop to have set the water moving, which the velocity view shows.
      const { image } = await run(overrides, () => {}, 40);
      expect(new Set(image).size, JSON.stringify(overrides)).toBeGreaterThan(8);
    }
  });

  it('takes the pointer while it lives and gives touch-action back', async () => {
    const { touchDuring, canvas } = await run({});
    expect(touchDuring).toBe('none');
    expect(canvas.style.touchAction).toBe('pan-y');
  });

  it('a tap puts ink in the tank where the pointer ray enters it', async () => {
    // With no ambient drops the picture is the empty tank until the tap lands.
    const quiet = { dropRate: 0 };
    const empty = await run(quiet, () => {}, 12);
    const tapped = await run(
      quiet,
      (frame, canvas) => {
        if (frame === 1) canvas.dispatchEvent(pointer('pointerdown', 48, 36, true));
        if (frame === 2) canvas.dispatchEvent(pointer('pointerup', 48, 36, false));
      },
      12,
    );
    // The first drop is seeded either way, so compare only that the tap changed the picture.
    let differing = 0;
    for (let i = 0; i < empty.image.length; i++)
      if (empty.image[i] !== tapped.image[i]) differing++;
    expect(differing).toBeGreaterThan(20);
  });
});

// --- Study -------------------------------------------------------------------------------------------------------------------------
//
// The numbers in src/effects/fluid-3d/notes.md. Skipped unless FLUID3D_STUDY names a file to write the
// report to:  FLUID3D_STUDY=/tmp/fluid3d.txt npx vitest run tests/fluid3d.test.ts -t study

const studyFile = (globalThis as { process?: { env: Record<string, string | undefined> } }).process
  ?.env.FLUID3D_STUDY;

describe.skipIf(!studyFile)('study', () => {
  const lines: string[] = [];
  const log = (text = '') => lines.push(text);
  const f = (x: number, digits = 3) => x.toFixed(digits);

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

  /** GPU time from the first timed pass starting to the last ending, for whatever `run` encodes. */
  async function span(
    device: GPUDevice,
    run: (
      encoder: GPUCommandEncoder,
      profiler: { pass(label: string): GPURenderPassTimestampWrites },
    ) => void,
  ): Promise<number> {
    const count = 512;
    const querySet = device.createQuerySet({ type: 'timestamp', count });
    let next = 0;
    const profiler = {
      pass: () => {
        const begin = next;
        next += 2;
        return { querySet, beginningOfPassWriteIndex: begin, endOfPassWriteIndex: begin + 1 };
      },
    };
    const resolve = device.createBuffer({
      size: count * 8,
      usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC,
    });
    const read = device.createBuffer({
      size: count * 8,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    const encoder = device.createCommandEncoder();
    run(encoder, profiler);
    encoder.resolveQuerySet(querySet, 0, count, resolve, 0);
    encoder.copyBufferToBuffer(resolve, 0, read, 0, count * 8);
    device.queue.submit([encoder.finish()]);
    await read.mapAsync(GPUMapMode.READ);
    const stamps = new BigUint64Array(read.getMappedRange().slice(0));
    read.unmap();
    const used = Array.from(stamps.slice(0, next)).filter((t) => t > 0n);
    resolve.destroy();
    read.destroy();
    querySet.destroy();
    const first = used.reduce((a, b) => (a < b ? a : b));
    const last = used.reduce((a, b) => (a > b ? a : b));
    return Number(last - first) / 1e6;
  }

  const TANK: [number, number, number] = [1, 1.25, 1];
  /** Round-robin sampling rounds for the cost tables (FLUID3D_ROUNDS; more rounds find quieter moments on a shared GPU). */
  const ROUNDS = Number(
    (globalThis as { process?: { env: Record<string, string | undefined> } }).process?.env
      .FLUID3D_ROUNDS ?? 12,
  );

  it('residual divergence against pressure work (one projection from zero)', async () => {
    const device = await getTestDevice();
    log(
      'RESIDUAL DIVERGENCE AFTER ONE PROJECTION FROM ZERO (rms divergence after / before), Gaussian source sigma 0.12',
    );
    for (const size of [32, 64]) {
      for (const [name, params, counts] of [
        ['multigrid V(2,2)', { pressureSolver: 'multigrid' }, [1, 2, 3, 4, 6]],
        ['multigrid V(3,3)', { pressureSolver: 'multigrid', smoothing: 3 }, [1, 2, 3, 4]],
        ['jacobi', { pressureSolver: 'jacobi' }, [8, 32, 128, 512]],
      ] as [string, Partial<Fluid3dParams>, number[]][]) {
        const row: string[] = [];
        for (const n of counts) {
          const fluid = create(device, { warmStart: false, ...params }, size);
          fluid.setVelocity(outward(0.12, 6));
          const before = divergence(await readVelocity(device, fluid)).l2;
          await submit(device, (e) => fluid.project(e, n));
          row.push(`${n}:${f(divergence(await readVelocity(device, fluid)).l2 / before, 4)}`);
          fluid.destroy();
        }
        log(`  ${size}^3 ${name.padEnd(17)} ${row.join('  ')}`);
      }
    }
    log();

    log(
      'RUNNING SIMULATION: mean ratio of divergence after / before the projection over steps 20 to 60, a falling heavy drop, 64 cells (tank 1 x 1.25 x 1)',
    );
    for (const [name, params] of [
      ['multigrid x1', { pressureSolver: 'multigrid', pressureIterations: 1 }],
      ['multigrid x2', { pressureSolver: 'multigrid', pressureIterations: 2 }],
      ['multigrid x3', { pressureSolver: 'multigrid', pressureIterations: 3 }],
      ['jacobi 8', { pressureSolver: 'jacobi', pressureIterations: 8 }],
      ['jacobi 24', { pressureSolver: 'jacobi', pressureIterations: 24 }],
      ['jacobi 64', { pressureSolver: 'jacobi', pressureIterations: 64 }],
    ] as [string, Partial<Fluid3dParams>][]) {
      const fluid = create(device, { buoyancy: 3, vorticity: 4, ...params }, 64, 64, TANK);
      fluid.setDensity(gaussian(0.4, 0.8, 0.4, 0.07, 2));
      let sum = 0;
      let n = 0;
      for (let step = 1; step <= 60; step++) {
        await submit(device, (e) => fluid.advance(e, 1));
        if (step < 20) continue;
        const after = divergence(await readVelocity(device, fluid)).l2;
        const raw = await read3d(device, fluid.divergence, 'float1');
        let s = 0;
        for (const v of raw) s += v * v;
        const before = Math.sqrt(s / raw.length);
        if (before > 1e-9) {
          sum += after / before;
          n++;
        }
      }
      log(`  ${name.padEnd(14)} ${f(sum / n)}`);
      fluid.destroy();
    }
    log();
    await finish();
  }, 3_600_000);

  type Runner = (
    encoder: GPUCommandEncoder,
    profiler: { pass(label: string): GPURenderPassTimestampWrites },
  ) => void;
  interface SlopeItem {
    name: string;
    low: Runner;
    high: Runner;
    /** Difference in the amount of work between `high` and `low`. */
    per: number;
  }

  /**
   * ms per unit of work for each item: (best span of `high` - best span of `low`) / per. The items are
   * sampled round-robin, many rounds, so a busy moment on the shared GPU hurts every item alike and the
   * minimum over the rounds finds each item's quiet moment.
   */
  async function slopes(device: GPUDevice, items: SlopeItem[], rounds = 12) {
    const best = new Map<string, [number, number]>(
      items.map((i) => [i.name, [Infinity, Infinity]]),
    );
    for (let round = 0; round < rounds; round++) {
      for (const item of items) {
        const b = best.get(item.name)!;
        b[0] = Math.min(b[0], await span(device, item.low));
        b[1] = Math.min(b[1], await span(device, item.high));
      }
    }
    return new Map(
      items.map((i) => [i.name, (best.get(i.name)![1] - best.get(i.name)![0]) / i.per]),
    );
  }

  /** A solver kept alive across samples; `profilerSink` is where its timestamps go. */
  function liveSolver(
    device: GPUDevice,
    resolution: number,
    scalar: number,
    params: Partial<Fluid3dParams>,
    workgroup: [number, number, number] = [4, 4, 4],
  ) {
    let sink: { pass(label: string): GPURenderPassTimestampWrites | undefined } = {
      pass: () => undefined,
    };
    const fluid = createFluid3d({
      device,
      resolution,
      scalarResolution: scalar,
      size: TANK,
      workgroup,
      profiler: { pass: (label: string) => sink.pass(label) },
      params: { vorticity: 4, buoyancy: 1.2, ...params },
    });
    const run =
      (n: number, what: 'step' | 'project' = 'step'): Runner =>
      (encoder, profiler) => {
        sink = profiler;
        if (what === 'step') fluid.advance(encoder, n);
        else fluid.project(encoder, n);
        sink = { pass: () => undefined };
      };
    return { fluid, run };
  }

  it('cost of the solver per step', async () => {
    const { device, timed } = await timedDevice();
    log(
      `SOLVER COST (timestamp queries: ${timed}); milliseconds per step, slope between 8 and 68 steps of the best of FLUID3D_ROUNDS round-robin samples (long samples, so the GPU clocks up); tank 1 x 1.25 x 1 (resolution counts the long side); workgroup 4x4x4 unless stated`,
    );
    const measure = async (
      rows: {
        name: string;
        resolution: number;
        scalar: number;
        params: Partial<Fluid3dParams>;
        workgroup?: [number, number, number];
      }[],
      rounds = ROUNDS,
    ) => {
      const live = rows.map((r) =>
        liveSolver(device, r.resolution, r.scalar, r.params, r.workgroup),
      );
      const result = await slopes(
        device,
        rows.map((r, i) => ({
          name: r.name,
          low: live[i]!.run(4),
          high: live[i]!.run(24),
          per: 20,
        })),
        rounds,
      );
      live.forEach((l) => l.fluid.destroy());
      return result;
    };
    const defaults = { pressureIterations: 2 };

    log(
      'Whole step with the effect defaults (multigrid x2, confinement 4, MacCormack ink, semi-Lagrangian velocity):',
    );
    const sizes = [
      [32, 64],
      [48, 96],
      [64, 96],
      [64, 128],
      [80, 128],
      [96, 128],
      [96, 160],
      [128, 160],
    ] as const;
    const whole = await measure(
      sizes.map(([r, s]) => ({ name: `${r}/${s}`, resolution: r, scalar: s, params: defaults })),
    );
    for (const [r, s] of sizes) {
      const fluid = createFluid3d({ device, resolution: r, scalarResolution: s, size: TANK });
      const g = fluid.grid;
      fluid.destroy();
      const cells = g.width * g.height * g.depth;
      const texels = g.scalarSize[0] * g.scalarSize[1] * g.scalarSize[2];
      log(
        `  velocity ${g.width}x${g.height}x${g.depth} (${(cells / 1e3).toFixed(0)}k cells), ink ${g.scalarSize.join('x')} (${(texels / 1e3).toFixed(0)}k texels): ${f(whole.get(`${r}/${s}`)!)} ms`,
      );
    }

    log('Pieces at velocity 48x64x48 / ink 72x96x72 (resolution 64 / 96), ms per step:');
    const pieces: [string, Partial<Fluid3dParams>][] = [
      ['whole step, defaults', defaults],
      ['no confinement', { ...defaults, vorticity: 0 }],
      ['velocity MacCormack', { ...defaults, velocityScheme: 'maccormack' }],
      ['ink semi-Lagrangian', { ...defaults, scalarScheme: 'semi-lagrangian' }],
      ['ink advection alone, MacCormack (scalarOnly)', { scalarOnly: true }],
      [
        'ink advection alone, semi-Lagrangian (scalarOnly)',
        { scalarOnly: true, scalarScheme: 'semi-lagrangian' },
      ],
      ['multigrid x1', { pressureIterations: 1 }],
      ['multigrid x3', { pressureIterations: 3 }],
      ['jacobi 24 sweeps', { pressureSolver: 'jacobi', pressureIterations: 24 }],
      ['jacobi 64 sweeps', { pressureSolver: 'jacobi', pressureIterations: 64 }],
      [
        'obstacle sphere (jacobi 24)',
        {
          pressureSolver: 'jacobi',
          pressureIterations: 24,
          obstacle: { center: [0.4, 0.4, 0.4], radius: 0.1 },
        },
      ],
    ];
    const pieceCosts = await measure(
      pieces.map(([name, params]) => ({ name, resolution: 64, scalar: 96, params })),
    );
    for (const [name] of pieces) log(`  ${name.padEnd(52)} ${f(pieceCosts.get(name)!)}`);

    log(
      'Pressure alone: ms per Jacobi sweep (slope 20 to 220) and per V(2,2) cycle (slope 4 to 44), project() only:',
    );
    const pressureSizes = [32, 48, 64, 96, 128];
    const pressureLive = pressureSizes.map((r) => ({
      r,
      jac: liveSolver(device, r, 32, { pressureSolver: 'jacobi' }),
      mg: liveSolver(device, r, 32, { pressureSolver: 'multigrid' }),
    }));
    const pressureItems: SlopeItem[] = pressureLive.flatMap(({ r, jac, mg }) => [
      { name: `j${r}`, low: jac.run(20, 'project'), high: jac.run(220, 'project'), per: 200 },
      { name: `m${r}`, low: mg.run(4, 'project'), high: mg.run(44, 'project'), per: 40 },
    ]);
    const pressure = await slopes(device, pressureItems, ROUNDS);
    for (const { r, jac, mg } of pressureLive) {
      const g = jac.fluid.grid;
      const sweep = pressure.get(`j${r}`)!;
      const cycle = pressure.get(`m${r}`)!;
      log(
        `  ${g.width}x${g.height}x${g.depth}: jacobi sweep ${f(sweep, 4)}  V-cycle ${f(cycle, 4)}  (a V-cycle costs ${f(cycle / sweep, 1)} sweeps; ${g.levels} levels)`,
      );
      jac.fluid.destroy();
      mg.fluid.destroy();
    }

    log('Workgroup shape, whole step (defaults) at resolution 64 / 96 and 96 / 160, ms:');
    const shapes: [number, number, number][] = [
      [4, 4, 4],
      [8, 4, 4],
      [4, 4, 8],
      [8, 8, 4],
      [16, 4, 4],
      [8, 8, 2],
      [32, 4, 2],
      [16, 16, 1],
    ];
    const shapeCosts = await measure(
      shapes.flatMap((wg) => [
        {
          name: `${wg.join('x')} small`,
          resolution: 64,
          scalar: 96,
          params: defaults,
          workgroup: wg,
        },
        {
          name: `${wg.join('x')} large`,
          resolution: 96,
          scalar: 160,
          params: defaults,
          workgroup: wg,
        },
      ]),
      ROUNDS,
    );
    for (const wg of shapes) {
      log(
        `  ${wg.join('x').padEnd(7)} ${f(shapeCosts.get(`${wg.join('x')} small`)!)}  ${f(shapeCosts.get(`${wg.join('x')} large`)!)}`,
      );
    }
    log();
    await finish();
  }, 3_600_000);

  it('cost of the render', async () => {
    const { device, timed } = await timedDevice();
    log(
      `RENDER COST at 1920 x 1080 (timestamp queries: ${timed}); the tank in its default view, ink from 150 frames of ambient drops, no solver work in the measured frames; ms, best of FLUID3D_ROUNDS round-robin samples`,
    );
    const { loadEffect } = await import('../src/effects/registry.ts');
    const { defaultParams } = await import('../src/effects/types.ts');
    const W = 1920;
    const H = 1080;
    const canvas = Object.assign(new EventTarget(), {
      width: W,
      height: H,
      style: {},
      getBoundingClientRect: () => ({ left: 0, top: 0, width: W, height: H }),
      setPointerCapture() {},
      releasePointerCapture() {},
    }) as unknown as HTMLCanvasElement;
    const target = device.createTexture({
      size: [W, H],
      format: 'rgba8unorm',
      usage: GPUTextureUsage.RENDER_ATTACHMENT,
    });
    const effect = await loadEffect('fluid-3d');
    const params = { ...defaultParams(effect.params), dropRate: 1.5, autoRotate: false } as Record<
      string,
      unknown
    >;
    let sink: { pass(label: string): GPURenderPassTimestampWrites | undefined } = {
      pass: () => undefined,
    };
    const instance = await effect.create({
      device,
      format: 'rgba8unorm',
      canvas,
      params: params as never,
      profiler: { pass: (label: string) => sink.pass(label) },
      hdr: false,
    });
    instance.resize(W, H, 1);
    for (let frame = 0; frame < 150; frame++) {
      const e = device.createCommandEncoder();
      instance.frame({ time: frame / 60, delta: 1 / 60, frame }, e, target.createView());
      device.queue.submit([e.finish()]);
    }
    await device.queue.onSubmittedWorkDone();
    params.dropRate = 0;

    const base = {
      renderScale: 0.5,
      steps: 64,
      lighting: 'light volume',
      skipEmpty: true,
      shadowSteps: 12,
      marchRepeat: 1,
      jitter: true,
      view: 'ink',
    };
    /** Frames of the effect with these params, delta 0 (no solver work): n of them in one command buffer. */
    const frames =
      (overrides: Record<string, unknown>, n: number): Runner =>
      (encoder, profiler) => {
        Object.assign(params, base, overrides);
        sink = profiler;
        for (let i = 0; i < n; i++) {
          instance.frame({ time: 3 + i, delta: 0, frame: 1000 + i }, encoder, target.createView());
        }
        sink = { pass: () => undefined };
      };
    const frameItem = (name: string, overrides: Record<string, unknown>): SlopeItem => ({
      name,
      low: frames(overrides, 4),
      high: frames(overrides, 20),
      per: 16,
    });
    /** The cost of one extra march: frames with 1 and 6 marches per frame. */
    const marchItem = (name: string, overrides: Record<string, unknown>): SlopeItem => ({
      name,
      low: frames({ ...overrides, marchRepeat: 1 }, 1),
      high: frames({ ...overrides, marchRepeat: 9 }, 1),
      per: 8,
    });

    const scales = [0.25, 0.35, 0.5, 0.75, 1];
    const stepCounts = [32, 64, 96, 128];
    const items: SlopeItem[] = [
      ...scales.map((s) => frameItem(`frame@${s}`, { renderScale: s })),
      ...scales.flatMap((s) =>
        stepCounts.map((n) => marchItem(`march@${s}x${n}`, { renderScale: s, steps: n })),
      ),
      ...[
        ['off', 12],
        ['light volume', 6],
        ['light volume', 12],
        ['light volume', 24],
        ['per-sample march', 6],
        ['per-sample march', 12],
        ['per-sample march', 24],
      ].map(([lighting, shadowSteps]) =>
        frameItem(`light:${lighting}:${shadowSteps}`, { lighting, shadowSteps }),
      ),
      frameItem('skip on', { skipEmpty: true }),
      frameItem('skip off', { skipEmpty: false }),
      marchItem('march skip on', { skipEmpty: true }),
      marchItem('march skip off', { skipEmpty: false }),
      frameItem('jitter on', { jitter: true }),
      frameItem('jitter off', { jitter: false }),
      frameItem('slice view', { view: 'ink slice' }),
    ];
    const cost = await slopes(device, items, ROUNDS);
    const g = (name: string) => f(cost.get(name)!);

    log(
      'Frame (light volume + occupancy, march, composite) at 64 steps, light volume, skip empty, by march resolution scale:',
    );
    for (const s of scales) log(`  scale ${s}: ${g(`frame@${s}`)} ms`);
    log('March alone (one extra march, slope over 1 to 9 per frame), ms, by scale and step count:');
    log(`  ${'scale'.padEnd(6)} ${stepCounts.map((n) => String(n).padStart(7)).join(' ')}`);
    for (const s of scales)
      log(
        `  ${String(s).padEnd(6)} ${stepCounts.map((n) => g(`march@${s}x${n}`).padStart(7)).join(' ')}`,
      );
    log('Lighting at scale 0.5, 64 steps (whole frame, ms):');
    for (const key of [
      'off:12',
      'light volume:6',
      'light volume:12',
      'light volume:24',
      'per-sample march:6',
      'per-sample march:12',
      'per-sample march:24',
    ]) {
      log(`  ${key.replace(':', ', shadow steps ').padEnd(34)} ${g(`light:${key}`)}`);
    }
    log(
      'Empty-space skipping at scale 0.5, 64 steps: frame ' +
        `on ${g('skip on')}, off ${g('skip off')}; march alone on ${g('march skip on')}, off ${g('march skip off')}`,
    );
    log(`Jitter: on ${g('jitter on')}, off ${g('jitter off')}`);
    log(`Debug slice view (composite only, no march or light): ${g('slice view')}`);
    log();
    instance.destroy();
    await finish();
  }, 3_600_000);
  it('cost of the presets (solver and render together)', async () => {
    const { device, timed } = await timedDevice();
    log(
      `WHOLE FRAME BY PRESET at 1920 x 1080 (timestamp queries: ${timed}): one solver step, light volume, march and composite per frame; ms per frame (slope over 4 to 20 frames), best of FLUID3D_ROUNDS samples; ink from 90 warm-up frames of drops`,
    );
    const { loadEffect } = await import('../src/effects/registry.ts');
    const { defaultParams } = await import('../src/effects/types.ts');
    const W = 1920;
    const H = 1080;
    const canvas = Object.assign(new EventTarget(), {
      width: W,
      height: H,
      style: {},
      getBoundingClientRect: () => ({ left: 0, top: 0, width: W, height: H }),
      setPointerCapture() {},
      releasePointerCapture() {},
    }) as unknown as HTMLCanvasElement;
    const target = device.createTexture({
      size: [W, H],
      format: 'rgba8unorm',
      usage: GPUTextureUsage.RENDER_ATTACHMENT,
    });
    const effect = await loadEffect('fluid-3d');
    const defaults = defaultParams(effect.params) as Record<string, unknown>;
    const params = { ...defaults } as Record<string, unknown>;
    let sink: { pass(label: string): GPURenderPassTimestampWrites | undefined } = {
      pass: () => undefined,
    };
    const instance = await effect.create({
      device,
      format: 'rgba8unorm',
      canvas,
      params: params as never,
      profiler: { pass: (label: string) => sink.pass(label) },
      hdr: false,
    });
    instance.resize(W, H, 1);
    const rows: [string, Record<string, unknown>][] = [
      ['ink drop (default)', {}],
      ['low-cost', effect.presets!['low-cost'] as Record<string, unknown>],
      ['high quality', effect.presets!['high quality'] as Record<string, unknown>],
      ['smoke plume', effect.presets!['smoke plume'] as Record<string, unknown>],
      [
        '96 / 128, 96 steps, scale 0.75',
        { resolution: '96', scalarResolution: '128', steps: 96, renderScale: 0.75 },
      ],
      [
        '128 / 160, 128 steps, scale 1',
        { resolution: '128', scalarResolution: '160', steps: 128, renderScale: 1 },
      ],
    ];
    for (const [name, preset] of rows) {
      Object.assign(params, defaults, preset, { dropRate: 1.5, autoRotate: false });
      let frame = 0;
      const frames =
        (n: number): Runner =>
        (encoder, profiler) => {
          sink = profiler;
          for (let i = 0; i < n; i++) {
            instance.frame(
              { time: frame / 60, delta: 1 / 60, frame: frame++ },
              encoder,
              target.createView(),
            );
          }
          sink = { pass: () => undefined };
        };
      for (let i = 0; i < 90; i++) {
        const e = device.createCommandEncoder();
        instance.frame({ time: frame / 60, delta: 1 / 60, frame: frame++ }, e, target.createView());
        device.queue.submit([e.finish()]);
      }
      await device.queue.onSubmittedWorkDone();
      const cost = await slopes(
        device,
        [{ name, low: frames(4), high: frames(20), per: 16 }],
        ROUNDS,
      );
      log(`  ${name.padEnd(34)} ${f(cost.get(name)!)} ms`);
    }
    log();
    instance.destroy();
    await finish();
  }, 3_600_000);
});
