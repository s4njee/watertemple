import { describe, expect, it, vi } from 'vitest';
import {
  COURANT_LIMIT,
  MAX_IMPULSES,
  computeGridSize,
  createHeightfield,
  type Heightfield,
  type HeightfieldKind,
  type WaveParams,
} from '../src/gpu/sim/heightfield.ts';
import { rasterizeRects } from '../src/gpu/sim/obstacles.ts';
import { checkShader, formatDiagnostic } from '../src/gpu/shader.ts';
import { expectValid, getTestDevice, readTexture } from './gpu.ts';

vi.setConfig({ testTimeout: 60_000 });

// --- Helpers -------------------------------------------------------------------------------------

const KINDS: HeightfieldKind[] = ['render', 'compute'];

function halfToFloat(bits: number): number {
  const sign = bits & 0x8000 ? -1 : 1;
  const exponent = (bits >> 10) & 0x1f;
  const fraction = bits & 0x3ff;
  if (exponent === 0) return sign * 2 ** -14 * (fraction / 1024);
  if (exponent === 31) return fraction ? Number.NaN : sign * Infinity;
  return sign * 2 ** (exponent - 15) * (1 + fraction / 1024);
}

interface Field {
  width: number;
  height: number;
  /** Per cell, row-major, row 0 at the top. */
  h: Float32Array;
  v: Float32Array;
  gx: Float32Array;
  gy: Float32Array;
  at(x: number, y: number): number;
}

/** Reads the solver's output texture back. */
async function readField(device: GPUDevice, solver: Heightfield): Promise<Field> {
  const { width, height } = solver.grid;
  const bytes = await readTexture(device, solver.texture, 8);
  const half = new Uint16Array(bytes.buffer, bytes.byteOffset, width * height * 4);
  const h = new Float32Array(width * height);
  const v = new Float32Array(width * height);
  const gx = new Float32Array(width * height);
  const gy = new Float32Array(width * height);
  for (let i = 0; i < width * height; i++) {
    h[i] = halfToFloat(half[i * 4]!);
    v[i] = halfToFloat(half[i * 4 + 1]!);
    gx[i] = halfToFloat(half[i * 4 + 2]!);
    gy[i] = halfToFloat(half[i * 4 + 3]!);
  }
  return { width, height, h, v, gx, gy, at: (x, y) => h[y * width + x]! };
}

async function advance(device: GPUDevice, solver: Heightfield, steps: number): Promise<void> {
  const encoder = device.createCommandEncoder();
  solver.advance(encoder, steps);
  device.queue.submit([encoder.finish()]);
  await device.queue.onSubmittedWorkDone();
}

async function simulate(
  device: GPUDevice,
  solver: Heightfield,
  seconds: number,
  fps = 60,
): Promise<number> {
  let steps = 0;
  for (let t = 0; t < seconds * fps - 1e-6; t++) {
    const encoder = device.createCommandEncoder();
    steps += solver.update(encoder, 1 / fps);
    device.queue.submit([encoder.finish()]);
  }
  await device.queue.onSubmittedWorkDone();
  return steps;
}

function create(
  device: GPUDevice,
  kind: HeightfieldKind,
  params: Partial<WaveParams> = {},
  resolution = 128,
  aspect = 1,
): Heightfield {
  return createHeightfield({
    device,
    kind,
    resolution,
    width: aspect,
    height: 1,
    params: { damping: 0, leak: 0, boundary: 'reflect', ...params },
  });
}

/** The simulation's energy: kinetic plus the gradient energy with the solver's own Courant number. */
function energy(field: Field, courant: number): number {
  let total = 0;
  for (let i = 0; i < field.h.length; i++) {
    total += 0.5 * field.v[i]! ** 2 + 0.5 * courant ** 2 * (field.gx[i]! ** 2 + field.gy[i]! ** 2);
  }
  return total;
}

const maxAbs = (a: ArrayLike<number>) =>
  Array.from(a).reduce((m, x) => Math.max(m, Math.abs(x)), 0);

/** Radius, in cells, at which the angle-averaged height peaks, searched between `from` and `to`. */
function ringRadius(field: Field, cx: number, cy: number, from = 3, to = 60): number {
  const sums = new Float64Array(to + 2);
  const counts = new Float64Array(to + 2);
  for (let y = 0; y < field.height; y++) {
    for (let x = 0; x < field.width; x++) {
      const r = Math.hypot(x + 0.5 - cx, y + 0.5 - cy);
      const bin = Math.round(r);
      if (bin > to) continue;
      sums[bin]! += field.at(x, y);
      counts[bin]! += 1;
    }
  }
  let best = from;
  let bestValue = -Infinity;
  for (let bin = from; bin <= to; bin++) {
    const value = sums[bin]! / Math.max(counts[bin]!, 1);
    if (value > bestValue) {
      bestValue = value;
      best = bin;
    }
  }
  // Parabolic refinement around the peak.
  const a = sums[best - 1]! / Math.max(counts[best - 1]!, 1);
  const b = bestValue;
  const c = sums[best + 1]! / Math.max(counts[best + 1]!, 1);
  const denominator = a - 2 * b + c;
  return best + (denominator < 0 ? (0.5 * (a - c)) / denominator : 0);
}

const DROP = { x: 0.5, y: 0.5, radius: 0.07, strength: 1 };

function drop(solver: Heightfield, strength = DROP.strength) {
  return solver.addDrop(DROP.x, DROP.y, DROP.radius, strength);
}

// --- The solver, both variants -------------------------------------------------------------------

describe.each(KINDS)('heightfield solver (%s)', (kind) => {
  it('keeps a still surface still', async () => {
    const device = await getTestDevice();
    await expectValid(device, async () => {
      const solver = create(device, kind);
      await advance(device, solver, 200);
      const field = await readField(device, solver);
      expect(maxAbs(field.h)).toBe(0);
      expect(maxAbs(field.v)).toBe(0);
      expect(maxAbs(field.gx)).toBe(0);
      solver.destroy();
    });
  });

  it('spreads a centred drop symmetrically in all four axis directions', async () => {
    const device = await getTestDevice();
    await expectValid(device, async () => {
      const solver = create(device, kind);
      drop(solver);
      await advance(device, solver, 40);
      const field = await readField(device, solver);
      const peak = maxAbs(field.h);
      expect(peak).toBeGreaterThan(0.01);
      // The drop is at cell corner (64, 64): cell 64 + i mirrors cell 63 - i.
      let worst = 0;
      for (let i = 0; i < 50; i++) {
        const right = field.at(64 + i, 63);
        const left = field.at(63 - i, 63);
        const down = field.at(63, 64 + i);
        const up = field.at(63, 63 - i);
        worst = Math.max(
          worst,
          Math.abs(right - left),
          Math.abs(right - down),
          Math.abs(right - up),
        );
      }
      expect(worst).toBeLessThan(0.01 * peak);
      solver.destroy();
    });
  });

  it('makes round rings: the wavefront is as far out along a diagonal as along an axis', async () => {
    const device = await getTestDevice();
    await expectValid(device, async () => {
      const solver = create(device, kind);
      drop(solver);
      await advance(device, solver, 60);
      const field = await readField(device, solver);
      // Peak of the ring along an axis and along the diagonal, by sampling thin rays.
      const along = (dx: number, dy: number) => {
        let best = 0;
        let bestValue = -Infinity;
        for (let r = 6; r < 60; r += 0.25) {
          const x = Math.round(63.5 + dx * r - 0.5);
          const y = Math.round(63.5 + dy * r - 0.5);
          // Average a few neighbouring samples across the ray to damp quantisation.
          const value = field.at(x, y);
          if (value > bestValue) {
            bestValue = value;
            best = r;
          }
        }
        return best;
      };
      const axis = (along(1, 0) + along(-1, 0) + along(0, 1) + along(0, -1)) / 4;
      const s = Math.SQRT1_2;
      const diagonal = (along(s, s) + along(-s, s) + along(s, -s) + along(-s, -s)) / 4;
      expect(Math.abs(diagonal - axis) / axis).toBeLessThan(0.05);
      solver.destroy();
    });
  });

  it('sends the wavefront out at the configured speed', async () => {
    const device = await getTestDevice();
    await expectValid(device, async () => {
      const waveSpeed = 0.4;
      const solver = create(device, kind, { waveSpeed });
      drop(solver);
      await simulate(device, solver, 0.3);
      const first = ringRadius(await readField(device, solver), 64, 64);
      await simulate(device, solver, 0.3);
      const second = ringRadius(await readField(device, solver), 64, 64);
      // 0.4 image heights per second on a 128-cell grid is 51.2 cells per second.
      const speed = (second - first) / 0.3;
      const expected = waveSpeed * solver.grid.height;
      expect(Math.abs(speed - expected) / expected).toBeLessThan(0.1);
      solver.destroy();
    });
  });

  it('looks the same at 60 Hz, 120 Hz and after a long stall', async () => {
    const device = await getTestDevice();
    await expectValid(device, async () => {
      const results: Float32Array[] = [];
      const steps: number[] = [];
      for (const fps of [60, 120, 40]) {
        const solver = create(device, kind);
        drop(solver);
        steps.push(await simulate(device, solver, 0.6, fps));
        results.push((await readField(device, solver)).h);
        solver.destroy();
      }
      const reference = results[0]!;
      const peak = maxAbs(reference);
      // Same simulated time, same number of steps to within one.
      expect(Math.abs(steps[0]! - steps[1]!)).toBeLessThanOrEqual(1);
      expect(Math.abs(steps[0]! - steps[2]!)).toBeLessThanOrEqual(1);
      for (const other of results.slice(1)) {
        const difference = other.map((x, i) => x - reference[i]!);
        expect(maxAbs(difference)).toBeLessThan(0.05 * peak);
      }

      // A stall (a tab in the background) delivers one huge delta: capped, not exploded.
      const solver = create(device, kind);
      drop(solver);
      const encoder = device.createCommandEncoder();
      const run = solver.update(encoder, 30);
      device.queue.submit([encoder.finish()]);
      expect(run).toBeLessThanOrEqual(16);
      const after = await readField(device, solver);
      expect(Number.isFinite(maxAbs(after.h))).toBe(true);
      expect(maxAbs(after.h)).toBeLessThan(10);
      solver.destroy();
    });
  });

  it('never gains energy, and loses it at the damping rate', async () => {
    const device = await getTestDevice();
    await expectValid(device, async () => {
      const courant = 0.5;
      const free = create(device, kind, { courant });
      drop(free);
      await advance(device, free, 1);
      const start = energy(await readField(device, free), courant);
      let highest = 0;
      for (let i = 0; i < 12; i++) {
        await advance(device, free, 25);
        highest = Math.max(highest, energy(await readField(device, free), courant));
      }
      // Undamped, reflecting: the energy only wobbles (the leapfrog scheme conserves a shadow
      // energy, not exactly this one), and half-float storage adds a little rounding noise.
      expect(highest).toBeLessThan(start * 1.25);
      free.destroy();

      // Damped: kinetic energy decays at `damping`, and the half of the energy that is kinetic
      // on average makes the total fall as exp(-damping t).
      const damping = 2;
      const damped = create(device, kind, { courant, damping });
      drop(damped);
      await advance(device, damped, 1);
      const before = energy(await readField(device, damped), courant);
      const seconds = 0.5;
      const steps = Math.round(seconds / damped.stepSeconds);
      await advance(device, damped, steps);
      const after = energy(await readField(device, damped), courant);
      const expected = Math.exp(-damping * seconds);
      expect(after).toBeLessThan(before);
      expect(after / before).toBeGreaterThan(expected * 0.7);
      expect(after / before).toBeLessThan(expected * 1.4);
      damped.destroy();
    });
  });

  it('reflects off the grid edge when told to, and swallows the wave when absorbing', async () => {
    const device = await getTestDevice();
    await expectValid(device, async () => {
      const courant = 0.5;
      // 64 cells wide: the front reaches the edge after ~64 steps and the reflection is back
      // at the centre after ~128.
      const results: Record<string, { total: number; centre: number }> = {};
      for (const boundary of ['reflect', 'absorb'] as const) {
        const solver = create(device, kind, { courant, boundary }, 64);
        drop(solver);
        await advance(device, solver, 150);
        const field = await readField(device, solver);
        let centre = 0;
        // Velocity, not height: the drop's net volume leaves a flat raised level behind that is not a wave.
        for (let y = 24; y < 40; y++)
          for (let x = 24; x < 40; x++) centre += Math.abs(field.v[y * 64 + x]!);
        results[boundary] = { total: energy(field, courant), centre };
        solver.destroy();
      }
      const reference = create(device, kind, { courant }, 64);
      drop(reference);
      await advance(device, reference, 1);
      const initial = energy(await readField(device, reference), courant);
      reference.destroy();

      // This energy estimate (kinetic plus a smoothed gradient term) is not the exact invariant of
      // the scheme, so it wanders by a few tens of percent while the true energy is conserved.
      expect(results.reflect!.total).toBeGreaterThan(initial * 0.75);
      expect(results.reflect!.total).toBeLessThan(initial * 1.25);
      // The wave came back to the middle.
      expect(results.reflect!.centre).toBeGreaterThan(0.2);
      // The sponge removed nearly all of it.
      expect(results.absorb!.total).toBeLessThan(initial * 0.06);
      expect(results.absorb!.centre).toBeLessThan(results.reflect!.centre * 0.1);
    });
  });

  it('reflects waves off an obstacle and keeps the far side dry', async () => {
    const device = await getTestDevice();
    await expectValid(device, async () => {
      const solver = create(device, kind, { courant: 0.5 }, 128);
      // A full-height wall, three cells thick, at x = 80..82.
      const wall = new Uint8Array(128 * 128);
      for (let y = 0; y < 128; y++) for (let x = 80; x < 83; x++) wall[y * 128 + x] = 255;
      solver.obstacles.set(wall);
      expect(solver.obstacles.enabled).toBe(true);
      solver.addDrop(48 / 128, 0.5, 0.05, 1);
      await advance(device, solver, 90);
      const field = await readField(device, solver);
      let behind = 0;
      let inWall = 0;
      for (let y = 0; y < 128; y++) {
        for (let x = 83; x < 128; x++) behind = Math.max(behind, Math.abs(field.at(x, y)));
        for (let x = 80; x < 83; x++) inWall = Math.max(inWall, Math.abs(field.at(x, y)));
      }
      // Nothing crosses the wall; the solid cells themselves stay at zero.
      expect(behind).toBe(0);
      expect(inWall).toBe(0);

      // Without the wall the same wave would have crossed that line by now.
      const open = create(device, kind, { courant: 0.5 }, 128);
      open.addDrop(48 / 128, 0.5, 0.05, 1);
      await advance(device, open, 90);
      const openField = await readField(device, open);
      let across = 0;
      for (let y = 0; y < 128; y++) across = Math.max(across, Math.abs(openField.at(90, y)));
      expect(across).toBeGreaterThan(0.01);
      open.destroy();

      // The reflection: the energy stays in the left half instead of leaking.
      const courant = 0.5;
      const kept = energy(field, courant);
      const leaked = energy(openField, courant);
      expect(kept).toBeGreaterThan(0);
      expect(kept).toBeGreaterThan(leaked * 0.9);
      solver.destroy();
    });
  });

  it('never produces NaN or Inf, however hard it is pushed', async () => {
    const device = await getTestDevice();
    await expectValid(device, async () => {
      const solver = create(
        device,
        kind,
        { waveSpeed: 1e6, courant: 50, damping: -5, leak: -1, spongeCells: 1000 },
        96,
      );
      for (let i = 0; i < 16; i++) solver.addDrop(Math.random(), Math.random(), 0.01, 1e6);
      solver.addDrop(Number.NaN, 0.5, 0.05, 1);
      solver.addDrop(0.5, 0.5, 0.05, Number.POSITIVE_INFINITY);
      for (let round = 0; round < 8; round++) {
        await advance(device, solver, 250);
        for (let i = 0; i < 4; i++) solver.displace([0.1, 0.1], [0.9, 0.9], 0.02, 1e5);
      }
      const field = await readField(device, solver);
      for (const channel of [field.h, field.v, field.gx, field.gy]) {
        expect(channel.every(Number.isFinite)).toBe(true);
      }
      expect(maxAbs(field.h)).toBeLessThanOrEqual(512);
      solver.destroy();
    });
  });

  it('clamps the Courant number below the stability limit', async () => {
    const device = await getTestDevice();
    const solver = create(device, kind, { courant: 99, waveSpeed: 0.5 });
    // dt = C * cell / c with C clamped to the limit.
    expect(solver.stepSeconds).toBeCloseTo((COURANT_LIMIT * solver.grid.cellSize) / 0.5, 9);
    solver.params.courant = 0;
    expect(solver.stepSeconds).toBeGreaterThan(0);
    solver.destroy();
  });

  it('accepts a few dozen impulses per frame and refuses more', async () => {
    const device = await getTestDevice();
    await expectValid(device, async () => {
      const solver = create(device, kind);
      let accepted = 0;
      for (let i = 0; i < MAX_IMPULSES + 10; i++) {
        if (solver.addDrop(0.2 + 0.006 * i, 0.5, 0.02, 1)) accepted++;
      }
      expect(accepted).toBe(MAX_IMPULSES);
      expect(solver.pendingImpulses).toBe(MAX_IMPULSES);
      await advance(device, solver, 1);
      expect(solver.pendingImpulses).toBe(0);
      const field = await readField(device, solver);
      expect(maxAbs(field.h)).toBeGreaterThan(0.05);
      solver.destroy();
    });
  });

  it('displaces water around a moving footprint and leaves nothing when it stands still', async () => {
    const device = await getTestDevice();
    await expectValid(device, async () => {
      const still = create(device, kind);
      still.displace([0.4, 0.5], [0.4, 0.5], 0.06, 1);
      await advance(device, still, 1);
      expect(maxAbs((await readField(device, still)).h)).toBe(0);
      still.destroy();

      const solver = create(device, kind);
      solver.displace([0.4, 0.5], [0.6, 0.5], 0.06, 1);
      await advance(device, solver, 1);
      const field = await readField(device, solver);
      // Water rises where the footprint left and drops where it arrived, by the same volume.
      const from = field.at(Math.round(0.4 * 128), 64);
      const to = field.at(Math.round(0.6 * 128), 64);
      expect(from).toBeGreaterThan(0.05);
      expect(to).toBeLessThan(-0.05);
      expect(Math.abs(from + to)).toBeLessThan(0.15 * from);
      // Volume is conserved exactly by construction; allow for half-float rounding of 100 cells.
      const volume = field.h.reduce((sum, x) => sum + x, 0);
      expect(Math.abs(volume)).toBeLessThan(0.02 * from * 100);
      solver.destroy();
    });
  });

  it('re-creates its grid on resize, clears the surface, and does nothing when the grid is unchanged', async () => {
    const device = await getTestDevice();
    await expectValid(device, async () => {
      const solver = create(device, kind, {}, 128, 1);
      drop(solver);
      await advance(device, solver, 20);
      expect(solver.grid.width).toBe(128);
      const generation = solver.generation;
      const texture = solver.texture;

      expect(solver.resize(500, 500)).toBe(false);
      expect(solver.generation).toBe(generation);
      expect(solver.texture).toBe(texture);

      expect(solver.resize(1600, 900)).toBe(true);
      expect(solver.generation).toBe(generation + 1);
      expect(solver.grid.width).toBe(128);
      expect(solver.grid.height).toBe(72);
      expect(solver.grid.cellSize).toBeCloseTo(1 / 72, 12);
      expect(solver.texture.width).toBe(128);
      expect(solver.texture.height).toBe(72);
      // The old waves are gone.
      await advance(device, solver, 10);
      expect(maxAbs((await readField(device, solver)).h)).toBe(0);

      expect(solver.setResolution(64)).toBe(true);
      expect(solver.grid.width).toBe(64);
      expect(solver.setResolution(64)).toBe(false);
      // Still works afterwards, with the new grid.
      solver.addDrop(0.5, 0.5, 0.05, 1);
      await advance(device, solver, 10);
      expect(maxAbs((await readField(device, solver)).h)).toBeGreaterThan(0.01);

      // A portrait canvas makes the height the long side.
      expect(solver.resize(900, 1600)).toBe(true);
      expect(solver.grid.height).toBe(64);
      expect(solver.grid.width).toBe(36);
      solver.destroy();
    });
  });

  it('remembers obstacle rectangles across a resize', async () => {
    const device = await getTestDevice();
    await expectValid(device, async () => {
      const solver = create(device, kind, {}, 100, 1);
      solver.obstacles.setRects([{ x: 0.4, y: 0.4, width: 0.2, height: 0.2 }]);
      expect(solver.obstacles.enabled).toBe(true);
      solver.resize(200, 100);
      expect(solver.obstacles.width).toBe(100);
      expect(solver.obstacles.height).toBe(50);
      expect(solver.obstacles.enabled).toBe(true);
      solver.obstacles.clear();
      expect(solver.obstacles.enabled).toBe(false);
      solver.destroy();
    });
  });

  it('can be reset to a flat surface and destroyed twice without complaint', async () => {
    const device = await getTestDevice();
    await expectValid(device, async () => {
      const solver = create(device, kind);
      drop(solver);
      await advance(device, solver, 30);
      const encoder = device.createCommandEncoder();
      solver.reset(encoder);
      device.queue.submit([encoder.finish()]);
      await advance(device, solver, 10);
      expect(maxAbs((await readField(device, solver)).h)).toBe(0);
      solver.destroy();
      solver.destroy();
    });
  });
});

describe('heightfield solver', () => {
  it('gives the same result from the render and the compute variant', async () => {
    const device = await getTestDevice();
    await expectValid(device, async () => {
      const params = { courant: 0.6, damping: 0.4, leak: 0.05, boundary: 'absorb' as const };
      const fields: Field[] = [];
      for (const kind of KINDS) {
        const solver = create(device, kind, params, 160, 4 / 3);
        solver.addDrop(0.3, 0.4, 0.04, 1);
        solver.obstacles.setRects([{ x: 0.55, y: 0.2, width: 0.08, height: 0.6, radius: 0.02 }]);
        solver.displace([0.7, 0.8], [0.8, 0.7], 0.03, 1);
        await advance(device, solver, 100);
        solver.addDrop(0.2, 0.7, 0.03, -1);
        await advance(device, solver, 150);
        fields.push(await readField(device, solver));
        solver.destroy();
      }
      const [render, compute] = fields as [Field, Field];
      const peak = maxAbs(render.h);
      expect(peak).toBeGreaterThan(0.02);
      // Same maths, different launch: they differ only where float rounding differs (fused
      // multiply-adds, summation order) before being rounded to half precision.
      const worst = maxAbs(render.h.map((x, i) => x - compute.h[i]!));
      const worstSlope = maxAbs(render.gx.map((x, i) => x - compute.gx[i]!));
      expect(worst).toBeLessThan(0.01 * peak);
      expect(worstSlope).toBeLessThan(0.01 * maxAbs(render.gx));
    });
  });

  it('builds its pipelines once per device', async () => {
    const device = await getTestDevice();
    // Warm the cache for both kinds, then count module creation for further solvers.
    for (const kind of KINDS) create(device, kind).destroy();
    const spy = vi.spyOn(device, 'createShaderModule');
    const spyRender = vi.spyOn(device, 'createRenderPipeline');
    const spyCompute = vi.spyOn(device, 'createComputePipeline');
    for (const kind of KINDS) create(device, kind).destroy();
    expect(spy).not.toHaveBeenCalled();
    expect(spyRender).not.toHaveBeenCalled();
    expect(spyCompute).not.toHaveBeenCalled();
    vi.restoreAllMocks();
  });

  it('sizes the grid from the canvas aspect with square cells', () => {
    expect(computeGridSize(256, 16 / 9)).toEqual({ width: 256, height: 144 });
    expect(computeGridSize(512, 1)).toEqual({ width: 512, height: 512 });
    expect(computeGridSize(256, 9 / 16)).toEqual({ width: 144, height: 256 });
    expect(computeGridSize(1e9, 2, 2048)).toEqual({ width: 2048, height: 1024 });
    expect(computeGridSize(Number.NaN, Number.NaN)).toEqual({ width: 256, height: 256 });
  });

  it('rasterises rounded rectangles at cell centres', () => {
    const data = rasterizeRects([{ x: 0.25, y: 0.25, width: 0.5, height: 0.5 }], 8, 8);
    const rows = Array.from({ length: 8 }, (_, y) => Array.from(data.slice(y * 8, y * 8 + 8)));
    expect(rows[3]).toEqual([0, 0, 255, 255, 255, 255, 0, 0]);
    expect(rows[1]).toEqual([0, 0, 0, 0, 0, 0, 0, 0]);
    // A big corner radius shaves the corners off.
    const rounded = rasterizeRects([{ x: 0, y: 0, width: 1, height: 1, radius: 0.5 }], 16, 16);
    expect(rounded[0]).toBe(0);
    expect(rounded[8 * 16 + 8]).toBe(255);
  });
});

// --- The WGSL decode helpers ---------------------------------------------------------------------

describe('heightfield.wgsl', () => {
  const SIZE = 64;

  /** Runs the decode helpers over every cell centre of the solver's output texture. */
  async function decode(device: GPUDevice, solver: Heightfield, strength: number) {
    const code = `
#include "heightfield.wgsl"
@group(0) @binding(0) var tex: texture_2d<f32>;
@group(0) @binding(1) var samp: sampler;
@group(0) @binding(2) var<storage, read_write> out: array<vec4f>;
@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3u) {
  let size = textureDimensions(tex);
  if (any(id.xy >= size)) { return; }
  let uv = (vec2f(id.xy) + 0.5) / vec2f(size);
  let s = heightfield_load(tex, vec2i(id.xy));
  let n = heightfield_sample_normal(tex, samp, uv, ${strength.toFixed(3)});
  let n2 = heightfield_normal(s, ${strength.toFixed(3)});
  let sampled = heightfield_sample(tex, samp, uv);
  // x, y, z: the normal from one filtered tap at the cell centre; w: its length minus the length
  // of the normal built from the exact texel's slopes (0 if the tap is exact).
  out[id.y * size.x + id.x] = vec4f(n, length(n2 - n) + abs(heightfield_height(sampled) - heightfield_height(s)));
}`;
    const { module, diagnostics } = await checkShader(device, {
      label: 'heightfield-decode',
      code,
    });
    const errors = diagnostics.filter((d) => d.severity === 'error');
    if (errors.length > 0) throw new Error(errors.map(formatDiagnostic).join('\n'));
    const pipeline = device.createComputePipeline({
      layout: 'auto',
      compute: { module, entryPoint: 'main' },
    });
    const { width, height } = solver.grid;
    const bytes = width * height * 16;
    const out = device.createBuffer({
      size: bytes,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });
    const read = device.createBuffer({
      size: bytes,
      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    });
    const group = device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: solver.view },
        {
          binding: 1,
          resource: device.createSampler({ magFilter: 'linear', minFilter: 'linear' }),
        },
        { binding: 2, resource: { buffer: out } },
      ],
    });
    const encoder = device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, group);
    pass.dispatchWorkgroups(Math.ceil(width / 8), Math.ceil(height / 8));
    pass.end();
    encoder.copyBufferToBuffer(out, 0, read, 0, bytes);
    device.queue.submit([encoder.finish()]);
    await read.mapAsync(GPUMapMode.READ);
    const result = new Float32Array(read.getMappedRange().slice(0));
    read.unmap();
    out.destroy();
    read.destroy();
    return result;
  }

  it.each(KINDS)(
    'decodes unit normals that agree with the finite-difference height (%s)',
    async (kind) => {
      const device = await getTestDevice();
      await expectValid(device, async () => {
        const solver = create(device, kind, {}, SIZE);
        solver.addDrop(0.4, 0.45, 0.08, 1);
        solver.addDrop(0.7, 0.6, 0.05, -1.5);
        await advance(device, solver, 30);
        const field = await readField(device, solver);
        const strength = 2.5;
        const normals = await decode(device, solver, strength);

        let slopeSeen = 0;
        let worstSlopeError = 0;
        let worstLength = 0;
        let worstTap = 0;
        for (let y = 2; y < SIZE - 2; y++) {
          for (let x = 2; x < SIZE - 2; x++) {
            const i = y * SIZE + x;
            const n = [normals[i * 4]!, normals[i * 4 + 1]!, normals[i * 4 + 2]!] as const;
            worstLength = Math.max(worstLength, Math.abs(Math.hypot(...n) - 1));
            worstTap = Math.max(worstTap, normals[i * 4 + 3]!);
            // The gradient channels are the Sobel gradient of the stored height.
            const sobelX =
              (field.at(x + 1, y - 1) +
                2 * field.at(x + 1, y) +
                field.at(x + 1, y + 1) -
                field.at(x - 1, y - 1) -
                2 * field.at(x - 1, y) -
                field.at(x - 1, y + 1)) /
              8;
            const sobelY =
              (field.at(x - 1, y + 1) +
                2 * field.at(x, y + 1) +
                field.at(x + 1, y + 1) -
                field.at(x - 1, y - 1) -
                2 * field.at(x, y - 1) -
                field.at(x + 1, y - 1)) /
              8;
            slopeSeen = Math.max(slopeSeen, Math.abs(sobelX));
            worstSlopeError = Math.max(
              worstSlopeError,
              Math.abs(field.gx[i]! - sobelX),
              Math.abs(field.gy[i]! - sobelY),
            );
            // The normal is (-strength * gradient, 1) normalised: x/z = -strength * dh/dx.
            const expectedX = -strength * sobelX;
            const expectedY = -strength * sobelY;
            expect(Math.abs(n[0] / n[2] - expectedX)).toBeLessThan(
              0.02 + 0.01 * Math.abs(expectedX),
            );
            expect(Math.abs(n[1] / n[2] - expectedY)).toBeLessThan(
              0.02 + 0.01 * Math.abs(expectedY),
            );
          }
        }
        expect(slopeSeen).toBeGreaterThan(0.03);
        expect(worstLength).toBeLessThan(1e-5);
        expect(worstTap).toBeLessThan(1e-4);
        // Half-float rounding of the stored heights and gradients.
        expect(worstSlopeError).toBeLessThan(0.01 * slopeSeen + 1e-3);
        solver.destroy();
      });
    },
  );
});

// --- Cost study (opt in: HEIGHTFIELD_BENCH=1 npx vitest run tests/heightfield.test.ts -t cost) -------

const benchEnabled = Boolean(
  (globalThis as { process?: { env: Record<string, string | undefined> } }).process?.env
    .HEIGHTFIELD_BENCH,
);

describe.skipIf(!benchEnabled)('cost study', () => {
  /** A device with timestamp queries if this adapter has them, else the shared one. */
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

  it('times GPU milliseconds per step for each variant and resolution', async () => {
    const { device, timed } = await timedDevice();
    const lines: string[] = [`timestamp queries: ${timed}`];

    /** GPU time from the first step starting to the resolve pass ending, milliseconds. */
    async function gpuSpan(solver: Heightfield, steps: number): Promise<number> {
      const querySet = device.createQuerySet({ type: 'timestamp', count: 8 });
      let next = 0;
      const timestamps = new Map<number, GPURenderPassTimestampWrites>();
      // Hands out two query slots per timed pass, in encoding order.
      const profiler = {
        pass: () => {
          const begin = next;
          next += 2;
          return { querySet, beginningOfPassWriteIndex: begin, endOfPassWriteIndex: begin + 1 };
        },
      };
      void timestamps;
      // The solver was created without a profiler, so time through a fresh one with the same state.
      const timedSolver = createHeightfield({
        device,
        kind: solver.kind,
        resolution: solver.grid.width,
        width: 1,
        height: 1,
        profiler,
        params: { damping: 0, leak: 0, boundary: 'reflect' },
      });
      const resolve = device.createBuffer({
        size: 64,
        usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC,
      });
      const read = device.createBuffer({
        size: 64,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
      });
      const encoder = device.createCommandEncoder();
      timedSolver.advance(encoder, steps);
      encoder.resolveQuerySet(querySet, 0, 8, resolve, 0);
      encoder.copyBufferToBuffer(resolve, 0, read, 0, 64);
      device.queue.submit([encoder.finish()]);
      await read.mapAsync(GPUMapMode.READ);
      const stamps = new BigUint64Array(read.getMappedRange().slice(0));
      read.unmap();
      timedSolver.destroy();
      const used = Array.from(stamps.slice(0, next)).filter((t) => t > 0n);
      const first = used.reduce((a, b) => (a < b ? a : b));
      const last = used.reduce((a, b) => (a > b ? a : b));
      return Number(last - first) / 1e6;
    }

    for (const size of [256, 512, 1024]) {
      for (const kind of KINDS) {
        const solver = create(device, kind, {}, size);
        const median = async (steps: number) => {
          const samples: number[] = [];
          for (let i = 0; i < 9; i++) samples.push(timed ? await gpuSpan(solver, steps) : 0);
          samples.sort((a, b) => a - b);
          return samples[4]!;
        };
        const low = await median(10);
        const high = await median(210);
        const perStep = (high - low) / 200;
        lines.push(
          `${size}^2 ${kind}: span(10)=${low.toFixed(3)} ms, span(210)=${high.toFixed(3)} ms, ${perStep.toFixed(4)} ms per step, ${((perStep * 1e6) / (size * size)).toFixed(3)} ns per cell`,
        );
        solver.destroy();
      }
    }
    console.log(`\n${lines.join('\n')}`);
  });
});
