import { describe, expect, it, vi } from 'vitest';
import { effect as washEffect } from '../src/effects/ink-wash/index.ts';
import { effect, INK_COLOURS, absorbance, linear, PAPERS } from '../src/effects/ink-water/index.ts';
import compositeWgsl from '../src/effects/ink-water/composite.wgsl';
import { defaultParams } from '../src/effects/types.ts';
import { noProfiler } from '../src/gpu/profiler.ts';
import { createFluid, floatToHalf, halfToFloat } from '../src/gpu/sim/fluid.ts';
import {
  createPigmentLayer,
  EDGE_CAP,
  W_CAP,
  pigmentDiffusionFraction,
  pigmentSubsteps,
  type PigmentLayer,
  type PigmentParams,
} from '../src/gpu/sim/pigment.ts';
import { expectValid, getTestDevice, readTexture } from './gpu.ts';
import { gpuF32 } from './wgsl-eval.ts';

vi.setConfig({ testTimeout: 120_000 });

// The pigment layer (src/gpu/sim/pigment.ts) is tested on hand-made textures, one rule at a time, with
// the other rules switched off: it is a pass on the dye grid, so every test builds a dye texture,
// steps the layer on it and reads the textures back. Totals through the solver's advection are NOT
// asserted anywhere (MacCormack does not conserve them); the tests that involve the solver only check
// what the pigment passes alone guarantee (a deposited layer that stays where it is).

// --- Helpers ---------------------------------------------------------------------------------------------

const USAGE =
  GPUTextureUsage.TEXTURE_BINDING |
  GPUTextureUsage.STORAGE_BINDING |
  GPUTextureUsage.COPY_SRC |
  GPUTextureUsage.COPY_DST;

/** Everything switched off: only the rule a test turns on does anything. */
const QUIET: Partial<PigmentParams> = {
  diffusion: 0,
  dryDiffusion: 1,
  feather: 0,
  edgeSpeed: 0,
  absorption: 0,
  wetAbsorption: 1,
  granulation: 0,
  lift: 0,
  drying: 0,
  wetSpread: 0,
  wetFloor: 0,
  dryDrag: 0,
  fade: 0,
  maxSubsteps: 1,
};

type Field = (x: number, y: number) => number;

type Format = 'rgba16float' | 'rgba32float';

/** Most tests run the kernels in 32-bit floats, where the algebra is exact; a few run the real 16-bit format. */
function dyeTexture(
  device: GPUDevice,
  width: number,
  height: number,
  format: Format = 'rgba32float',
): GPUTexture {
  return device.createTexture({ size: [width, height], format, usage: USAGE });
}

/** Uploads optical density per channel (and its mean in alpha) from functions of the texel. */
function writeDye(
  device: GPUDevice,
  texture: GPUTexture,
  channels: readonly [Field, Field, Field],
) {
  const { width, height } = texture;
  const wide = texture.format === 'rgba32float';
  const data = wide ? new Float32Array(width * height * 4) : new Uint16Array(width * height * 4);
  const store = (value: number) => (wide ? value : floatToHalf(value));
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const o = (y * width + x) * 4;
      const v = channels.map((f) => f(x, y));
      for (let c = 0; c < 3; c++) data[o + c] = store(v[c]!);
      data[o + 3] = store((v[0]! + v[1]! + v[2]!) / 3);
    }
  }
  device.queue.writeTexture(
    { texture },
    data as Float32Array<ArrayBuffer> | Uint16Array<ArrayBuffer>,
    { bytesPerRow: width * (wide ? 16 : 8) },
    [width, height],
  );
}

function writeWet(device: GPUDevice, layer: PigmentLayer, field: Field) {
  const { width, height } = layer;
  const data = new Float32Array(width * height);
  for (let y = 0; y < height; y++)
    for (let x = 0; x < width; x++) data[y * width + x] = field(x, y);
  device.queue.writeTexture({ texture: layer.wet.texture }, data, { bytesPerRow: width * 4 }, [
    width,
    height,
  ]);
}

/** Uploads the paper tile: r grain height, g fibre, from functions of the texel. */
function paperTexture(device: GPUDevice, size: number, grain: Field, fibre: Field): GPUTextureView {
  const texture = device.createTexture({
    size: [size, size],
    format: 'rgba8unorm',
    usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
  });
  const data = new Uint8Array(size * size * 4);
  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      const o = (y * size + x) * 4;
      data[o] = Math.round(255 * grain(x, y));
      data[o + 1] = Math.round(255 * fibre(x, y));
      data[o + 3] = 255;
    }
  }
  device.queue.writeTexture({ texture }, data, { bytesPerRow: size * 4 }, [size, size]);
  return texture.createView();
}

async function readHalf(device: GPUDevice, texture: GPUTexture): Promise<Float32Array> {
  if (texture.format === 'rgba32float') {
    const bytes = await readTexture(device, texture, 16);
    return new Float32Array(
      bytes.buffer,
      bytes.byteOffset,
      texture.width * texture.height * 4,
    ).slice();
  }
  const bytes = await readTexture(device, texture, 8);
  const half = new Uint16Array(bytes.buffer, bytes.byteOffset, texture.width * texture.height * 4);
  const out = new Float32Array(half.length);
  for (let i = 0; i < half.length; i++) out[i] = halfToFloat(half[i]!);
  return out;
}

async function readFloat(device: GPUDevice, texture: GPUTexture): Promise<Float32Array> {
  const bytes = await readTexture(device, texture, 4);
  return new Float32Array(bytes.buffer, bytes.byteOffset, texture.width * texture.height).slice();
}

/** A pair of dye textures and a function that runs `steps` pigment steps on them, ping-ponging. */
function stepper(device: GPUDevice, layer: PigmentLayer, dt: number) {
  const textures = [
    dyeTexture(device, layer.width, layer.height, layer.format),
    dyeTexture(device, layer.width, layer.height, layer.format),
  ] as const;
  const views = [textures[0].createView(), textures[1].createView()] as const;
  let current = 0;
  return {
    /** The texture holding the suspended pigment now. */
    get texture() {
      return textures[current]!;
    },
    async run(steps: number, perSubmit = 8) {
      for (let done = 0; done < steps; done += perSubmit) {
        const encoder = device.createCommandEncoder();
        for (let i = 0; i < Math.min(perSubmit, steps - done); i++) {
          layer.step(encoder, { dt, read: views[current]!, write: views[1 - current]! });
          current = 1 - current;
        }
        device.queue.submit([encoder.finish()]);
      }
      await device.queue.onSubmittedWorkDone();
    },
    destroy() {
      for (const t of textures) t.destroy();
    },
  };
}

interface Moments {
  total: number;
  cx: number;
  cy: number;
  /** Variance along x and y about the centroid, texels squared. */
  vx: number;
  vy: number;
}

function moments(data: Float32Array, width: number, height: number, channel = 0): Moments {
  let total = 0;
  let sx = 0;
  let sy = 0;
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const v = data[(y * width + x) * 4 + channel]!;
      total += v;
      sx += v * x;
      sy += v * y;
    }
  }
  const cx = sx / total;
  const cy = sy / total;
  let vx = 0;
  let vy = 0;
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const v = data[(y * width + x) * 4 + channel]!;
      vx += v * (x - cx) ** 2;
      vy += v * (y - cy) ** 2;
    }
  }
  return { total, cx, cy, vx: vx / total, vy: vy / total };
}

function sum(data: Float32Array, stride: number, channel: number): number {
  let total = 0;
  for (let i = channel; i < data.length; i += stride) total += data[i]!;
  return total;
}

const finiteAll = (data: ArrayLike<number>) => {
  for (let i = 0; i < data.length; i++) if (!Number.isFinite(data[i])) return false;
  return true;
};

const SIZE = 64;

function makeLayer(
  device: GPUDevice,
  params: Partial<PigmentParams>,
  format: Format = 'rgba32float',
  size = SIZE,
): PigmentLayer {
  return createPigmentLayer({
    device,
    width: size,
    height: size,
    format,
    params: { ...QUIET, ...params },
  });
}

/** The diffusivity that gives a per-face fraction `w` at step `dt` on a grid `size` texels tall. */
const diffusivityFor = (w: number, dt: number, size = SIZE) => w / dt / size / size;

const spike =
  (amplitude: number): Field =>
  (x, y) =>
    x === SIZE / 2 && y === SIZE / 2 ? amplitude : 0;

// --- Diffusion ---------------------------------------------------------------------------------------------

describe('diffusion', () => {
  it('conserves the total pigment and grows the variance of a point linearly with time', async () => {
    const device = await getTestDevice();
    const dt = 1 / 60;
    const w = 0.08;
    const layer = makeLayer(device, { diffusion: diffusivityFor(w, dt) });
    const s = stepper(device, layer, dt);
    writeDye(device, s.texture, [spike(40), spike(20), spike(10)]);

    const samples: { step: number; vx: number; vy: number }[] = [];
    let step = 0;
    for (const target of [10, 20, 40, 80]) {
      await s.run(target - step);
      step = target;
      const data = await readHalf(device, s.texture);
      const m = moments(data, SIZE, SIZE);
      samples.push({ step, vx: m.vx, vy: m.vy });
      // Conserved, in every channel (half floats round, hence the tolerance).
      expect(Math.abs(m.total / 40 - 1)).toBeLessThan(1e-3);
      expect(Math.abs(sum(data, 4, 1) / 20 - 1)).toBeLessThan(1e-3);
      expect(Math.abs(sum(data, 4, 2) / 10 - 1)).toBeLessThan(1e-3);
      // A point stays where it was, and nothing goes negative.
      expect(Math.abs(m.cx - SIZE / 2)).toBeLessThan(0.05);
      expect(Math.min(...data)).toBeGreaterThanOrEqual(0);
    }
    // Variance 2 w per step and axis: the diffusion coefficient says how fast it spreads.
    for (const { step: n, vx, vy } of samples) {
      expect(Math.abs(vx / (2 * w * n) - 1)).toBeLessThan(0.01);
      expect(Math.abs(vy / vx - 1)).toBeLessThan(0.01);
    }
    // Linear: the variance gained in the second half is the first half's.
    const at = (n: number) => samples.find((x) => x.step === n)!.vx;
    expect(Math.abs((at(80) - at(40)) / (at(40) - at(20)) / 2 - 1)).toBeLessThan(0.02);
    layer.destroy();
    s.destroy();
  });

  it('in the real 16-bit format never gains pigment and loses only a rounding sliver per step', async () => {
    const device = await getTestDevice();
    const dt = 1 / 60;
    const w = 0.08;
    const layer = makeLayer(device, { diffusion: diffusivityFor(w, dt) }, 'rgba16float');
    const s = stepper(device, layer, dt);
    writeDye(device, s.texture, [spike(40), spike(40), spike(40)]);
    await s.run(80);
    const m = moments(await readHalf(device, s.texture), SIZE, SIZE);
    // The store rounds (towards zero, on the machine this was written on), about 0.03 percent of a
    // changing texel per step: no mass is ever created, and 80 steps lose a few percent at most.
    expect(m.total / 40).toBeLessThanOrEqual(1.002);
    expect(m.total / 40).toBeGreaterThan(0.95);
    // The spread is not the problem: the variance is the continuum's to within the lost tails.
    expect(Math.abs(m.vx / (2 * w * 80) - 1)).toBeLessThan(0.04);
    layer.destroy();
    s.destroy();
  });

  it('spreads the same in the same time whatever the step size', async () => {
    const device = await getTestDevice();
    const D = diffusivityFor(0.1, 1 / 30);
    const T = 40 / 30;
    const variances: number[] = [];
    for (const dt of [1 / 30, 1 / 60, 1 / 240]) {
      const layer = makeLayer(device, { diffusion: D });
      const s = stepper(device, layer, dt);
      writeDye(device, s.texture, [spike(40), spike(40), spike(40)]);
      await s.run(Math.round(T / dt), 16);
      const m = moments(await readHalf(device, s.texture), SIZE, SIZE);
      variances.push(m.vx);
      expect(Math.abs(m.total / 40 - 1)).toBeLessThan(1e-3);
      layer.destroy();
      s.destroy();
    }
    // Continuum answer: 2 D T in texels squared.
    const expected = 2 * D * T * SIZE * SIZE;
    for (const v of variances) expect(Math.abs(v / expected - 1)).toBeLessThan(0.01);
    expect(Math.max(...variances) / Math.min(...variances)).toBeLessThan(1.01);
  });

  it('splits a large diffusivity into sub-steps instead of going unstable', async () => {
    const device = await getTestDevice();
    const dt = 1 / 60;
    const total = 0.3; // fraction per face and step: needs 3 sub-steps at the 0.12 cap
    const D = diffusivityFor(total, dt);
    expect(pigmentSubsteps({ diffusion: D, maxSubsteps: 4 }, dt, SIZE)).toBe(3);
    expect(pigmentDiffusionFraction({ diffusion: D, maxSubsteps: 4 }, dt, SIZE)).toBeCloseTo(
      0.1,
      6,
    );
    expect(pigmentSubsteps({ diffusion: D, maxSubsteps: 2 }, dt, SIZE)).toBe(2);
    expect(pigmentSubsteps({ diffusion: 0, maxSubsteps: 4 }, dt, SIZE)).toBe(1);
    expect(pigmentSubsteps({ diffusion: Number.NaN, maxSubsteps: 4 }, dt, SIZE)).toBe(1);

    const layer = makeLayer(device, { diffusion: D, maxSubsteps: 4 });
    const s = stepper(device, layer, dt);
    writeDye(device, s.texture, [spike(40), spike(40), spike(40)]);
    await s.run(30);
    expect(layer.lastSubsteps).toBe(3);
    const m = moments(await readHalf(device, s.texture), SIZE, SIZE);
    expect(Math.abs(m.vx / (2 * total * 30) - 1)).toBeLessThan(0.01);
    expect(Math.abs(m.total / 40 - 1)).toBeLessThan(1e-3);
    layer.destroy();
    s.destroy();
  });

  it('is slower on dry paper and stays inside the stability cap when asked for the impossible', async () => {
    const device = await getTestDevice();
    const dt = 1 / 60;
    const D = diffusivityFor(0.1, dt);
    // Dry paper (wetness 0) with dryDiffusion 0.25: a quarter of the variance of a wet sheet.
    const dry = makeLayer(device, { diffusion: D, dryDiffusion: 0.25 });
    const wet = makeLayer(device, { diffusion: D, dryDiffusion: 0.25, wetFloor: 1 });
    writeWet(device, wet, () => 1);
    const out: number[] = [];
    for (const layer of [dry, wet]) {
      const s = stepper(device, layer, dt);
      writeDye(device, s.texture, [spike(40), spike(40), spike(40)]);
      await s.run(40);
      out.push(moments(await readHalf(device, s.texture), SIZE, SIZE).vx);
      s.destroy();
    }
    expect(Math.abs(out[0]! / out[1]! - 0.25)).toBeLessThan(0.01);

    // A diffusivity of 1000 saturates at the cap: no negative pigment, no growth, still conserved.
    const huge = makeLayer(device, { diffusion: 1000, wetFloor: 1, maxSubsteps: 2 });
    writeWet(device, huge, () => 1);
    const s = stepper(device, huge, dt);
    writeDye(device, s.texture, [spike(40), spike(40), spike(40)]);
    await s.run(20);
    const data = await readHalf(device, s.texture);
    expect(finiteAll(data)).toBe(true);
    expect(Math.min(...data)).toBeGreaterThanOrEqual(0);
    expect(Math.max(...data)).toBeLessThanOrEqual(40);
    expect(Math.abs(moments(data, SIZE, SIZE).total / 40 - 1)).toBeLessThan(1e-3);
    expect(W_CAP).toBeLessThan(0.25);
    for (const l of [dry, wet, huge]) l.destroy();
    s.destroy();
  });

  it('feathers along paper fibres: pigment travels further where the paper has a fibre', async () => {
    const device = await getTestDevice();
    const dt = 1 / 60;
    const layer = makeLayer(device, { diffusion: diffusivityFor(0.04, dt), feather: 1 });
    // A horizontal fibre (five rows wide) through the middle; elsewhere plain paper.
    layer.setPaper(
      paperTexture(
        device,
        SIZE,
        () => 0.5,
        (_x, y) => (Math.abs(y - SIZE / 2) <= 2 ? 1 : 0),
      ),
    );
    const s = stepper(device, layer, dt);
    writeDye(device, s.texture, [spike(40), spike(40), spike(40)]);
    await s.run(60);
    const m = moments(await readHalf(device, s.texture), SIZE, SIZE);
    // The fibre is a fast lane along x: the blob is longer along the fibre than across it.
    expect(m.vx / m.vy).toBeGreaterThan(1.5);
    expect(Math.abs(m.total / 40 - 1)).toBeLessThan(1e-3);
    layer.destroy();
    s.destroy();
  });
});

// --- Settling and lifting --------------------------------------------------------------------------------

describe('deposition', () => {
  const blob: Field = (x, y) => 8 * Math.exp(-((x - 32) ** 2 + (y - 32) ** 2) / 50);

  it('moves pigment from the water into the paper without losing any', async () => {
    const device = await getTestDevice();
    const dt = 1 / 60;
    const layer = makeLayer(device, { absorption: 1.2, diffusion: diffusivityFor(0.03, dt) });
    const s = stepper(device, layer, dt);
    writeDye(device, s.texture, [blob, (x, y) => blob(x, y) * 0.5, (x, y) => blob(x, y) * 0.25]);
    const start = await readHalf(device, s.texture);
    const total0 = [0, 1, 2].map((c) => sum(start, 4, c));

    let previousStain = 0;
    for (const steps of [30, 60, 120]) {
      await s.run(steps);
      const dye = await readHalf(device, s.texture);
      const stain = await readHalf(device, layer.stain.texture);
      for (let c = 0; c < 3; c++) {
        const both = sum(dye, 4, c) + sum(stain, 4, c);
        expect(Math.abs(both / total0[c]! - 1)).toBeLessThan(1e-3);
      }
      // More and more of it is in the paper, at about the rate asked for: exp(-1.2 t) stays suspended.
      const stained = sum(stain, 4, 0) / total0[0]!;
      expect(stained).toBeGreaterThan(previousStain);
      previousStain = stained;
    }
    // After 3.5 s at 1.2 / s (dry paper) about 1 - exp(-4.2) = 98.5 percent is deposited.
    const dye = await readHalf(device, s.texture);
    expect(sum(dye, 4, 0) / total0[0]!).toBeLessThan(0.05);
    layer.destroy();
    s.destroy();
  });

  it('deposits at the configured rate (an exponential in time, whatever the step)', async () => {
    const device = await getTestDevice();
    const results: number[] = [];
    for (const dt of [1 / 30, 1 / 120]) {
      const layer = makeLayer(device, { absorption: 0.8 });
      const s = stepper(device, layer, dt);
      writeDye(device, s.texture, [() => 2, () => 2, () => 2]);
      await s.run(Math.round(1 / dt), 12);
      const dye = await readHalf(device, s.texture);
      results.push(sum(dye, 4, 0) / (SIZE * SIZE * 2));
      layer.destroy();
      s.destroy();
    }
    for (const r of results) expect(Math.abs(r - Math.exp(-0.8))).toBeLessThan(2e-3);
  });

  it('settles faster in the valleys of the paper than on its peaks (granulation)', async () => {
    const device = await getTestDevice();
    const dt = 1 / 60;
    const layer = makeLayer(device, { absorption: 1, granulation: 1 });
    // Left half valleys (grain 0), right half peaks (grain 1).
    layer.setPaper(
      paperTexture(
        device,
        SIZE,
        (x) => (x < SIZE / 2 ? 0 : 1),
        () => 0,
      ),
    );
    const s = stepper(device, layer, dt);
    writeDye(device, s.texture, [() => 1, () => 1, () => 1]);
    await s.run(30);
    const stain = await readHalf(device, layer.stain.texture);
    let valley = 0;
    let peak = 0;
    for (let y = 0; y < SIZE; y++) {
      valley += stain[(y * SIZE + 8) * 4]!;
      peak += stain[(y * SIZE + SIZE - 8) * 4]!;
    }
    // gran = 1 + 1.4 (0.5 - grain): 1.7 in a valley, 0.3 on a peak.
    const rate = (g: number) => 1 - Math.exp(-g * 1 * 0.5);
    expect(Math.abs(valley / SIZE / rate(1.7) - 1)).toBeLessThan(0.02);
    expect(Math.abs(peak / SIZE / rate(0.3) - 1)).toBeLessThan(0.05);
    expect(valley / peak).toBeGreaterThan(3.5);
    layer.destroy();
    s.destroy();
  });

  it('takes up less under a wet film than on dry paper, and a wet film lifts stain back (back-runs)', async () => {
    const device = await getTestDevice();
    const dt = 1 / 60;

    // Wet sheet, no absorption left to speak of: the stain dissolves back into the water.
    const layer = makeLayer(device, { absorption: 0, lift: 1.5, wetFloor: 1 });
    writeWet(device, layer, () => 1);
    // Seed the stain through a first, absorbing phase.
    const s = stepper(device, layer, dt);
    layer.params.absorption = 3;
    layer.params.wetAbsorption = 1;
    writeDye(device, s.texture, [() => 1, () => 1, () => 1]);
    await s.run(20);
    layer.params.absorption = 0;
    const before = await readHalf(device, layer.stain.texture);
    const stain0 = sum(before, 4, 0);
    expect(stain0).toBeGreaterThan(0.2 * SIZE * SIZE);
    const dye0 = sum(await readHalf(device, s.texture), 4, 0);
    await s.run(40);
    const after = await readHalf(device, layer.stain.texture);
    const dye1 = sum(await readHalf(device, s.texture), 4, 0);
    // 1 - exp(-1.5 * 40 / 60) = 63 percent of the stain went back into suspension; total conserved.
    expect(sum(after, 4, 0) / stain0).toBeGreaterThan(0.33);
    expect(sum(after, 4, 0) / stain0).toBeLessThan(0.41);
    expect(Math.abs((dye1 + sum(after, 4, 0)) / (dye0 + stain0) - 1)).toBeLessThan(1e-3);

    // The same stain on dry paper stays.
    const dry = makeLayer(device, { absorption: 0, lift: 1.5 });
    const t = stepper(device, dry, dt);
    dry.params.absorption = 3;
    writeDye(device, t.texture, [() => 1, () => 1, () => 1]);
    await t.run(20);
    dry.params.absorption = 0;
    const kept0 = sum(await readHalf(device, dry.stain.texture), 4, 0);
    await t.run(40);
    expect(sum(await readHalf(device, dry.stain.texture), 4, 0) / kept0).toBeGreaterThan(0.99);

    // Wet paper takes up pigment at wetAbsorption times the dry rate.
    const slow = makeLayer(device, { absorption: 1, wetAbsorption: 0.25, wetFloor: 1 });
    writeWet(device, slow, () => 1);
    const u = stepper(device, slow, dt);
    writeDye(device, u.texture, [() => 1, () => 1, () => 1]);
    await u.run(60);
    const rate = sum(await readHalf(device, slow.stain.texture), 4, 0) / (SIZE * SIZE);
    expect(Math.abs(rate - (1 - Math.exp(-0.25)))).toBeLessThan(2e-3);
    for (const l of [layer, dry, slow]) l.destroy();
    s.destroy();
    t.destroy();
    u.destroy();
  });

  it('washes out suspended and deposited pigment alike at the fade rate', async () => {
    const device = await getTestDevice();
    const dt = 1 / 60;
    const layer = makeLayer(device, { absorption: 0.5, fade: 0.7 });
    const s = stepper(device, layer, dt);
    writeDye(device, s.texture, [() => 1, () => 1, () => 1]);
    await s.run(60);
    const both =
      sum(await readHalf(device, s.texture), 4, 0) +
      sum(await readHalf(device, layer.stain.texture), 4, 0);
    expect(Math.abs(both / (SIZE * SIZE) - Math.exp(-0.7))).toBeLessThan(2e-3);
    layer.destroy();
    s.destroy();
  });
});

// --- The deposited layer is not carried by the flow -------------------------------------------------------

describe('the deposited layer under a moving fluid', () => {
  it('does not move while the fluid does, and its pigment stays where it settled', async () => {
    const device = await getTestDevice();
    const fluid = createFluid({
      device,
      resolution: 32,
      dyeResolution: 64,
      params: { velocityDissipation: 0, dyeDissipation: 0, dyeScheme: 'maccormack' },
    });
    const layer = createPigmentLayer({
      device,
      width: fluid.grid.dyeWidth,
      height: fluid.grid.dyeHeight,
      params: { ...QUIET, wetFloor: 1, absorption: 0 },
    });
    layer.attach(fluid);
    writeWet(device, layer, () => 1);

    // A blob on the left becomes stain (absorption on for a moment, no flow yet).
    fluid.setDye((x, y) => {
      const v = 6 * Math.exp(-((x - 0.3) ** 2 + (y - 0.5) ** 2) / 0.004);
      return [v, v * 0.5, v * 0.25, v];
    });
    layer.params.absorption = 4;
    layer.params.wetAbsorption = 1;
    const run = async (steps: number) => {
      const encoder = device.createCommandEncoder();
      layer.beginFrame(encoder);
      fluid.advance(encoder, steps);
      device.queue.submit([encoder.finish()]);
      await device.queue.onSubmittedWorkDone();
    };
    await run(30);
    layer.params.absorption = 0;

    const stain0 = await readHalf(device, layer.stain.texture);
    const dye0 = await readHalf(device, fluid.dye.texture);
    expect(sum(stain0, 4, 0)).toBeGreaterThan(5);
    const centroid0 = moments(dye0, fluid.grid.dyeWidth, fluid.grid.dyeHeight).cx;

    // Now stir hard: a strong steady push to the right across the whole canvas.
    fluid.setDye((x, y) => {
      const v = 6 * Math.exp(-((x - 0.3) ** 2 + (y - 0.5) ** 2) / 0.004);
      return [v, v * 0.5, v * 0.25, v];
    });
    for (let k = 0; k < 6; k++) {
      fluid.addForce([0.1, 0.5], [0.9, 0.5], 0.25, [1.5, 0]);
      await run(10);
    }
    const stain1 = await readHalf(device, layer.stain.texture);
    const dye1 = await readHalf(device, fluid.dye.texture);

    // The suspended pigment was carried (its centroid moved by several texels)...
    const centroid1 = moments(dye1, fluid.grid.dyeWidth, fluid.grid.dyeHeight).cx;
    expect(centroid1 - centroid0).toBeGreaterThan(2);
    // ... and the stain did not budge: with nothing absorbed or lifted it is bit for bit the same.
    expect(stain1.length).toBe(stain0.length);
    let different = 0;
    for (let i = 0; i < stain0.length; i++) if (stain1[i] !== stain0[i]) different++;
    expect(different).toBe(0);
    layer.detach(fluid);
    layer.destroy();
    fluid.destroy();
  });
});

// --- Wetness -------------------------------------------------------------------------------------------------

describe('wetness', () => {
  /** The same recursion in double precision: (w + spread) * decay - linear, floored, capped at 1. */
  function dry(w: number, steps: number, rate: number, dt: number, floor = 0): number {
    const decay = Math.exp(-rate * dt);
    const linear = 0.04 * rate * dt;
    for (let i = 0; i < steps; i++) w = Math.min(1, Math.max(floor, w * decay - linear));
    return w;
  }

  it('recedes at the configured drying rate and reaches zero', async () => {
    const device = await getTestDevice();
    const dt = 1 / 60;
    const layer = makeLayer(device, { drying: 0.6 });
    writeWet(device, layer, () => 0.8);
    const s = stepper(device, layer, dt);
    let steps = 0;
    for (const target of [30, 120, 300, 480]) {
      await s.run(target - steps);
      steps = target;
      const wet = await readFloat(device, layer.wet.texture);
      const expected = dry(0.8, steps, 0.6, dt);
      expect(Math.abs(wet[0]! - expected)).toBeLessThan(2e-5);
      expect(Math.abs(wet[SIZE * SIZE - 1]! - expected)).toBeLessThan(2e-5);
    }
    // 8 seconds of drying at 0.6 / s: (0.8 + 0.04) exp(-4.8) - 0.04 < 0: dry, and exactly zero.
    const wet = await readFloat(device, layer.wet.texture);
    expect(Math.max(...wet)).toBe(0);
    layer.destroy();
    s.destroy();
  });

  it('dries the same in the same time whatever the step size', async () => {
    const device = await getTestDevice();
    const out: number[] = [];
    for (const dt of [1 / 20, 1 / 60, 1 / 240]) {
      const layer = makeLayer(device, { drying: 0.4 });
      writeWet(device, layer, () => 0.9);
      const s = stepper(device, layer, dt);
      await s.run(Math.round(2 / dt), 20);
      out.push((await readFloat(device, layer.wet.texture))[100]!);
      layer.destroy();
      s.destroy();
    }
    // About (0.9 + 0.04) exp(-0.8) - 0.04 = 0.38 in all three.
    for (const w of out) expect(Math.abs(w - 0.38)).toBeLessThan(0.01);
    expect(Math.max(...out) - Math.min(...out)).toBeLessThan(0.005);
  });

  it('never falls under the floor, and a submerged sheet stays submerged', async () => {
    const device = await getTestDevice();
    const layer = makeLayer(device, { drying: 5, wetFloor: 0.4 });
    writeWet(device, layer, (x) => (x < SIZE / 2 ? 1 : 0));
    const s = stepper(device, layer, 1 / 60);
    await s.run(120);
    const wet = await readFloat(device, layer.wet.texture);
    expect(Math.min(...wet)).toBeGreaterThanOrEqual(0.4);
    expect(Math.min(...wet)).toBeLessThan(0.4 + 1e-6);
    layer.destroy();
    s.destroy();
  });

  it('creeps outward by capillary action without creating water', async () => {
    const device = await getTestDevice();
    const dt = 1 / 60;
    const layer = makeLayer(device, { wetSpread: diffusivityFor(0.1, dt) });
    writeWet(device, layer, (x, y) => (x === 32 && y === 32 ? 100 : 0));
    // The brush adds at most 1 per texel; start from a clipped spike instead.
    writeWet(device, layer, (x, y) => (Math.abs(x - 32) < 2 && Math.abs(y - 32) < 2 ? 1 : 0));
    const s = stepper(device, layer, dt);
    const before = (await readFloat(device, layer.wet.texture)).reduce((a, b) => a + b, 0);
    await s.run(30);
    const wet = await readFloat(device, layer.wet.texture);
    expect(Math.abs(wet.reduce((a, b) => a + b, 0) / before - 1)).toBeLessThan(1e-4);
    expect(wet[32 * SIZE + 28]).toBeGreaterThan(0.01);
    layer.destroy();
    s.destroy();
  });

  it('ragged edges change the outline of a wetted patch, never the range, and never the flat top', async () => {
    const device = await getTestDevice();
    const rows: Float32Array[] = [];
    for (const ragged of [0, 0.8]) {
      const layer = makeLayer(device, { ragged });
      layer.addWater([0.5, 0.5], [0.5, 0.5], 0.12, 3);
      const encoder = device.createCommandEncoder();
      layer.beginFrame(encoder);
      device.queue.submit([encoder.finish()]);
      rows.push(await readFloat(device, layer.wet.texture));
      layer.destroy();
    }
    const [plain, rough] = rows as [Float32Array, Float32Array];
    expect(Math.max(...rough)).toBeLessThanOrEqual(1);
    expect(Math.min(...rough)).toBeGreaterThanOrEqual(0);
    let different = 0;
    for (let i = 0; i < plain.length; i++) if (Math.abs(plain[i]! - rough[i]!) > 0.02) different++;
    expect(different).toBeGreaterThan(50);
    // The patch is still roughly where it was: its wetted area stays within a quarter of the plain one.
    const area = (a: Float32Array) => a.reduce((t, v) => t + (v > 0.5 ? 1 : 0), 0);
    expect(Math.abs(area(rough) / area(plain) - 1)).toBeLessThan(0.25);
    expect(rough[32 * SIZE + 32]).toBe(1);
  });

  it('is wetted by the brush, with a flat-topped film for a generous splat', async () => {
    const device = await getTestDevice();
    const layer = makeLayer(device, {});
    layer.addWater([0.5, 0.5], [0.5, 0.5], 0.1, 3);
    layer.addWater([0.1, 0.1], [0.1, 0.1], 0.05, 0.5);
    expect(layer.addWater([Number.NaN, 0], [0, 0], 0.1, 1)).toBe(false);
    const encoder = device.createCommandEncoder();
    layer.beginFrame(encoder);
    device.queue.submit([encoder.finish()]);
    const wet = await readFloat(device, layer.wet.texture);
    expect(wet[32 * SIZE + 32]).toBe(1);
    expect(wet[32 * SIZE + 34]).toBe(1); // flat top
    expect(wet[6 * SIZE + 6]).toBeGreaterThan(0.45);
    expect(wet[6 * SIZE + 6]).toBeLessThan(0.5);
    expect(wet[60 * SIZE + 60]).toBe(0);
    expect(Math.max(...wet)).toBeLessThanOrEqual(1);
    layer.destroy();
  });
});

// --- Edge darkening ---------------------------------------------------------------------------------------------

describe('edge drift', () => {
  it('carries pigment from the wet towards the dry and piles it at the end of the film, conserving it', async () => {
    const device = await getTestDevice();
    const dt = 1 / 60;
    const layer = makeLayer(device, { edgeSpeed: 0.1 * SIZE ** -1 * 60 });
    // A film that thins linearly from 1 at x = 8 to 0 at x = 40, dry beyond.
    writeWet(device, layer, (x) => Math.min(1, Math.max(0, (40 - x) / 32)));
    const s = stepper(device, layer, dt);
    writeDye(device, s.texture, [
      (x) => (x >= 8 && x < 40 ? 1 : 0),
      (x) => (x >= 8 && x < 40 ? 1 : 0),
      (x) => (x >= 8 && x < 40 ? 1 : 0),
    ]);
    const total0 = SIZE * 32;
    await s.run(240);
    const data = await readHalf(device, s.texture);
    const row = (x: number) => data[(32 * SIZE + x) * 4]!;
    expect(Math.abs(sum(data, 4, 0) / total0 - 1)).toBeLessThan(1e-3);
    expect(Math.min(...data)).toBeGreaterThanOrEqual(0);
    // Pigment left the wet end and gathered at the dry end of the film.
    expect(row(10)).toBeLessThan(0.5);
    expect(row(39) + row(40) + row(41)).toBeGreaterThan(3);
    // Nothing crossed into the uniformly dry paper beyond the film.
    expect(row(46)).toBeLessThan(0.01);
    expect(EDGE_CAP).toBeLessThan(0.25);
    layer.destroy();
    s.destroy();
  });

  it('does nothing where the film is uniform', async () => {
    const device = await getTestDevice();
    const layer = makeLayer(device, { edgeSpeed: 5 });
    writeWet(device, layer, () => 0.7);
    const s = stepper(device, layer, 1 / 60);
    writeDye(device, s.texture, [(x) => (x < 32 ? 1 : 0), () => 0, () => 0]);
    await s.run(30);
    const data = await readHalf(device, s.texture);
    // Bit for bit unchanged (a step edge in the pigment, no gradient in the water).
    expect(data[(10 * SIZE + 31) * 4]).toBe(1);
    expect(data[(10 * SIZE + 32) * 4]).toBe(0);
    layer.destroy();
    s.destroy();
  });
});

// --- Dry paper holds the flow -------------------------------------------------------------------------------------

describe('flow through dry paper', () => {
  it('damps the velocity where the paper is dry and leaves a wet film alone', async () => {
    const device = await getTestDevice();
    const cells = 32;
    const layer = createPigmentLayer({
      device,
      width: 64,
      height: 64,
      params: { ...QUIET, dryDrag: 20 },
    });
    // Left half wet, right half dry (wetness in dye texels).
    writeWet(device, layer, (x) => (x < 32 ? 1 : 0));
    const velocity = dyeTexture(device, cells + 1, cells + 1, 'rgba16float');
    const out = dyeTexture(device, cells + 1, cells + 1, 'rgba16float');
    writeDye(device, velocity, [() => 1, () => -0.5, () => 0]);
    const encoder = device.createCommandEncoder();
    const wrote = layer.dampFlow({
      encoder,
      dt: 1 / 60,
      grid: { width: cells, height: cells },
      read: velocity.createView(),
      write: out.createView(),
    });
    device.queue.submit([encoder.finish()]);
    expect(wrote).toBe(true);
    const data = await readHalf(device, out);
    const at = (x: number, y: number, c: number) => data[(y * (cells + 1) + x) * 4 + c]!;
    expect(at(4, 10, 0)).toBeCloseTo(1, 3);
    expect(at(4, 10, 1)).toBeCloseTo(-0.5, 3);
    const factor = Math.exp(-20 / 60);
    expect(at(28, 10, 0)).toBeCloseTo(factor, 2);
    expect(at(28, 10, 1)).toBeCloseTo(-0.5 * factor, 2);
    // Nothing to damp on a submerged sheet or with the drag off.
    layer.params.wetFloor = 1;
    expect(
      layer.dampFlow({
        encoder: device.createCommandEncoder(),
        dt: 1 / 60,
        grid: { width: cells, height: cells },
        read: velocity.createView(),
        write: out.createView(),
      }),
    ).toBe(false);
    layer.destroy();
    velocity.destroy();
    out.destroy();
  });
});

// --- Lifecycle ----------------------------------------------------------------------------------------------------

describe('lifecycle', () => {
  it('re-creates its textures on resize, clears on reset, and survives a solver of another size', async () => {
    const device = await getTestDevice();
    const layer = makeLayer(device, { wetFloor: 0.5 }, 'rgba16float');
    const generation = layer.generation;
    expect(layer.resize(SIZE, SIZE)).toBe(false);
    expect(layer.resize(48, 40)).toBe(true);
    expect(layer.generation).toBe(generation + 1);
    expect([layer.width, layer.height]).toEqual([48, 40]);
    expect(layer.stain.texture.width).toBe(48);
    writeWet(device, layer, () => 0.9);
    const encoder = device.createCommandEncoder();
    layer.reset(encoder);
    device.queue.submit([encoder.finish()]);
    // Reset puts the wetness at its floor, and clears the stain.
    const wet = await readFloat(device, layer.wet.texture);
    expect(Math.min(...wet)).toBe(0.5);
    expect(Math.max(...wet)).toBe(0.5);
    expect(Math.max(...(await readHalf(device, layer.stain.texture)))).toBe(0);

    // A solver whose dye grid is not this layer's grid is left alone: the hooks decline instead of throwing.
    const fluid = createFluid({ device, resolution: 16, dyeResolution: 32 });
    layer.attach(fluid);
    await expectValid(device, async () => {
      const e = device.createCommandEncoder();
      fluid.advance(e, 2);
      device.queue.submit([e.finish()]);
      await device.queue.onSubmittedWorkDone();
    });
    layer.detach(fluid);
    expect(fluid.hooks.dye).toBeUndefined();
    expect(fluid.hooks.forces).toBeUndefined();
    layer.destroy();
    layer.destroy();
    fluid.destroy();
  });
});

// --- Extremes ----------------------------------------------------------------------------------------------------------

describe('parameter extremes', () => {
  const EXTREMES: Record<string, Partial<PigmentParams>> = {
    huge: {
      diffusion: 1e9,
      dryDiffusion: 1e9,
      feather: 1e9,
      edgeSpeed: 1e9,
      absorption: 1e9,
      wetAbsorption: 1e9,
      granulation: 1e9,
      lift: 1e9,
      drying: 1e9,
      wetSpread: 1e9,
      wetFloor: 1e9,
      dryDrag: 1e9,
      fade: 1e9,
      maxSubsteps: 1e9,
      dyeMax: 1e9,
    },
    negative: {
      diffusion: -1,
      dryDiffusion: -1,
      feather: -1,
      edgeSpeed: -1,
      absorption: -1,
      wetAbsorption: -1,
      granulation: -1,
      lift: -1,
      drying: -1,
      wetSpread: -1,
      wetFloor: -1,
      dryDrag: -1,
      fade: -1,
      maxSubsteps: -5,
      dyeMax: -1,
    },
    notANumber: {
      diffusion: Number.NaN,
      dryDiffusion: Number.NaN,
      feather: Number.NaN,
      edgeSpeed: Number.NaN,
      absorption: Number.NaN,
      wetAbsorption: Number.NaN,
      granulation: Number.NaN,
      lift: Number.NaN,
      drying: Number.NaN,
      wetSpread: Number.NaN,
      wetFloor: Number.NaN,
      dryDrag: Number.NaN,
      fade: Number.NaN,
      maxSubsteps: Number.NaN,
      dyeMax: Number.NaN,
    },
    demanding: {
      diffusion: 0.05,
      edgeSpeed: 100,
      absorption: 50,
      lift: 50,
      maxSubsteps: 8,
      wetSpread: 1,
    },
  };

  it.each(Object.entries(EXTREMES))(
    'stays finite, non-negative and bounded with %s parameters',
    async (_name, extreme) => {
      const device = await getTestDevice();
      const layer = makeLayer(device, extreme);
      for (const dt of [1 / 60, 1e-9, 5, Number.NaN]) {
        const s = stepper(device, layer, dt);
        // Big values, a hard edge, and NaN and infinity in the input for good measure.
        writeDye(device, s.texture, [
          (x) => (x < 32 ? 64 : 0),
          (x, y) => (x === 5 && y === 5 ? Number.NaN : 1e4),
          (x, y) => (x === 6 && y === 6 ? Number.POSITIVE_INFINITY : 3),
        ]);
        writeWet(device, layer, (x, y) => ((x + y) % 7) / 6);
        await s.run(12);
        const dye = await readHalf(device, s.texture);
        const stain = await readHalf(device, layer.stain.texture);
        const wet = await readFloat(device, layer.wet.texture);
        expect(finiteAll(dye)).toBe(true);
        expect(finiteAll(stain)).toBe(true);
        expect(finiteAll(wet)).toBe(true);
        expect(Math.min(...dye)).toBeGreaterThanOrEqual(0);
        expect(Math.min(...stain.filter((_, i) => i % 4 < 3))).toBeGreaterThanOrEqual(0);
        expect(Math.max(...dye)).toBeLessThanOrEqual(256);
        expect(Math.max(...stain)).toBeLessThanOrEqual(256);
        expect(Math.min(...wet)).toBeGreaterThanOrEqual(0);
        expect(Math.max(...wet)).toBeLessThanOrEqual(1);
        s.destroy();
      }
      layer.destroy();
    },
  );

  it('survives a solver with absurd forces and a tiny grid', async () => {
    const device = await getTestDevice();
    const fluid = createFluid({ device, resolution: 16, dyeResolution: 32 });
    const layer = createPigmentLayer({
      device,
      width: fluid.grid.dyeWidth,
      height: fluid.grid.dyeHeight,
      params: EXTREMES.demanding,
    });
    layer.attach(fluid);
    await expectValid(device, async () => {
      for (let frame = 0; frame < 4; frame++) {
        fluid.addDye([0.5, 0.5], [0.6, 0.6], 0.1, [5, 5, 5, 5]);
        fluid.addForce([0.2, 0.2], [0.8, 0.8], 0.1, [1e30, -1e30]);
        layer.addWater([0.5, 0.5], [0.5, 0.5], 0.2, 5);
        const encoder = device.createCommandEncoder();
        layer.beginFrame(encoder);
        fluid.update(encoder, 1 / 30);
        device.queue.submit([encoder.finish()]);
      }
      await device.queue.onSubmittedWorkDone();
    });
    expect(finiteAll(await readHalf(device, fluid.dye.texture))).toBe(true);
    expect(finiteAll(await readHalf(device, layer.stain.texture))).toBe(true);
    layer.detach(fluid);
    layer.destroy();
    fluid.destroy();
  });
});

// --- Colour -----------------------------------------------------------------------------------------------------------------

describe('subtractive colour', () => {
  const wgsl = `${compositeWgsl.code}
fn f(x: vec4f) -> vec4f {
  // x.xyz: optical density; the paper is a warm white; x.w selects light (0) or dark (1) paper.
  let base = select(vec3f(0.9, 0.85, 0.7), vec3f(0.02, 0.025, 0.04), x.w > 0.5);
  return vec4f(ink_lay(base, x.xyz, x.w > 0.5), 0.0);
}`;

  const paper = [0.9, 0.85, 0.7];
  const density = (name: string) => absorbance(linear(INK_COLOURS[name]!));
  const compose = async (od: number[], dark = 0) => {
    const out = await gpuF32(wgsl, new Float32Array([...od, dark]));
    return [out[0]!, out[1]!, out[2]!];
  };

  it('puts blue over yellow and gets green, which additive mixing cannot', async () => {
    const blue = density('cobalt');
    const yellow = density('ochre');
    const mixed = await compose(blue.map((b, i) => b * 0.5 + yellow[i]! * 0.5));
    const onlyBlue = await compose(blue);
    const onlyYellow = await compose(yellow);
    // Equal parts: the geometric mean of the two inks' colours over the paper.
    for (let c = 0; c < 3; c++) {
      const expected =
        paper[c]! * Math.sqrt((onlyBlue[c]! / paper[c]!) * (onlyYellow[c]! / paper[c]!));
      expect(mixed[c]!).toBeCloseTo(expected, 4);
    }
    // Equal parts of blue and yellow are green, though neither ink is: more green than red or blue.
    expect(onlyBlue[1]! / paper[1]!).toBeLessThan(onlyBlue[2]! / paper[2]!);
    expect(mixed[1]!).toBeGreaterThan(mixed[0]!);
    expect(mixed[1]!).toBeGreaterThan(mixed[2]!);
    // One over the other (their densities added in full) is darker than either alone, in every channel,
    // and just as green.
    const layered = await compose(blue.map((b, i) => b + yellow[i]!));
    for (let c = 0; c < 3; c++) {
      expect(layered[c]!).toBeLessThanOrEqual(onlyBlue[c]! + 1e-6);
      expect(layered[c]!).toBeLessThanOrEqual(onlyYellow[c]! + 1e-6);
      expect(layered[c]!).toBeLessThanOrEqual(mixed[c]! + 1e-6);
    }
    expect(layered[1]!).toBeGreaterThan(layered[0]!);
    expect(layered[1]!).toBeGreaterThan(layered[2]!);
    // Averaging the two colours instead (what a dye that mixes additively does) is lighter in every
    // channel: the geometric mean of the transmittances is under their arithmetic mean.
    for (let c = 0; c < 3; c++) {
      expect(mixed[c]!).toBeLessThan((onlyBlue[c]! + onlyYellow[c]!) / 2);
    }
  });

  it('never makes the paper brighter, for any pigment, on light paper', async () => {
    const random = (() => {
      let state = 12345;
      return () => (state = (state * 1664525 + 1013904223) >>> 0) / 4294967296;
    })();
    const input = new Float32Array(4 * 2000);
    for (let i = 0; i < 2000; i++) {
      input[i * 4] = random() * 6 * (i % 3 === 0 ? 0 : 1);
      input[i * 4 + 1] = random() * 6 * (i % 5 === 0 ? 0 : 1);
      input[i * 4 + 2] = random() * 6;
    }
    const out = await gpuF32(wgsl, input);
    for (let i = 0; i < 2000; i++) {
      for (let c = 0; c < 3; c++) {
        expect(out[i * 4 + c]!).toBeLessThanOrEqual(paper[c]! + 1e-6);
        expect(out[i * 4 + c]!).toBeGreaterThanOrEqual(0);
      }
    }
    // No pigment, no change; a lot of pigment, near black.
    expect(await compose([0, 0, 0])).toEqual(paper.map((p) => Math.fround(p)));
    expect(Math.max(...(await compose([8, 8, 8])))).toBeLessThan(0.001);
  });

  it('keeps luminous ink on dark paper finite, covering by density and never above 1', async () => {
    const dark = [0.02, 0.025, 0.04];
    expect(await compose([0, 0, 0], 1)).toEqual(dark.map((p) => Math.fround(p)));
    const thin = await compose([0.05, 0.05, 0.05], 1);
    const thick = await compose([4, 3, 0.5], 1);
    // Thin ink barely shows; thick ink covers the paper with its own hue (blue-ish here: blue least absorbed).
    expect(thin[0]!).toBeLessThan(0.2);
    expect(thick[2]!).toBeGreaterThan(thick[0]!);
    expect(Math.max(...thick)).toBeLessThanOrEqual(1);
    const out = await gpuF32(
      wgsl,
      new Float32Array([100, 0, 0, 1, 0, 100, 100, 1, 1e6, 1e6, 1e6, 1]),
    );
    for (const v of out) if (Number.isFinite(v)) expect(v).toBeLessThanOrEqual(1);
  });

  it('defines every ink and paper of the palette', () => {
    for (const name of Object.keys(INK_COLOURS)) {
      const od = density(name);
      expect(od.every((x) => Number.isFinite(x) && x >= 0)).toBe(true);
      expect(Math.max(...od)).toBeGreaterThan(0.5);
    }
    for (const colour of Object.values(PAPERS)) {
      expect(linear(colour).every((x) => x >= 0 && x <= 1)).toBe(true);
    }
  });
});

// --- The effect, headlessly ------------------------------------------------------------------------------------------------

const WIDTH = 64;
const HEIGHT = 48;

type StubCanvas = HTMLCanvasElement & EventTarget;

function stubCanvas(): StubCanvas {
  return Object.assign(new EventTarget(), {
    width: WIDTH,
    height: HEIGHT,
    style: {},
    getBoundingClientRect: () => ({ left: 0, top: 0, width: WIDTH, height: HEIGHT }),
    setPointerCapture() {},
    releasePointerCapture() {},
  }) as unknown as StubCanvas;
}

/** A synthetic pointer event at canvas uv (x, y). Node has `Event`, not `PointerEvent`. */
function pointerEvent(type: string, x: number, y: number, down: boolean, id = 1): Event {
  return Object.assign(new Event(type), {
    pointerId: id,
    pointerType: 'mouse',
    clientX: x * WIDTH,
    clientY: y * HEIGHT,
    pressure: down ? 0.5 : 0,
    buttons: down ? 1 : 0,
  });
}

async function runEffect(
  overrides: Record<string, unknown>,
  frames: number,
  script?: (frame: number, canvas: StubCanvas) => void,
): Promise<Uint8Array> {
  const device = await getTestDevice();
  const format: GPUTextureFormat = 'rgba8unorm';
  const target = device.createTexture({
    size: [WIDTH, HEIGHT],
    format,
    usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC,
  });
  const canvas = stubCanvas();
  await expectValid(device, async () => {
    const instance = await effect.create({
      device,
      format,
      canvas,
      params: { ...defaultParams(effect.params), ...overrides } as ReturnType<
        typeof defaultParams<typeof effect.params>
      >,
      profiler: noProfiler,
      hdr: false,
    });
    instance.resize(WIDTH, HEIGHT, 1);
    for (let frame = 0; frame < frames; frame++) {
      script?.(frame, canvas);
      const encoder = device.createCommandEncoder();
      instance.frame({ time: frame / 60, delta: 1 / 60, frame }, encoder, target.createView());
      device.queue.submit([encoder.finish()]);
    }
    instance.destroy();
    await device.queue.onSubmittedWorkDone();
  });
  const pixels = await readTexture(device, target);
  target.destroy();
  return pixels;
}

const differing = (a: Uint8Array, b: Uint8Array) => {
  let n = 0;
  for (let i = 0; i < a.length; i += 4) {
    if (
      Math.abs(a[i]! - b[i]!) + Math.abs(a[i + 1]! - b[i + 1]!) + Math.abs(a[i + 2]! - b[i + 2]!) >
      12
    )
      n++;
  }
  return n;
};

describe('the effect', () => {
  it.each(Object.entries(effect.presets ?? {}))('draws the "%s" preset', async (_name, preset) => {
    const pixels = await runEffect(preset as Record<string, unknown>, 4);
    expect(new Set(pixels).size).toBeGreaterThan(8);
    // Opaque.
    for (let i = 3; i < pixels.length; i += 4) expect(pixels[i]).toBe(255);
  });

  it('draws every view and every paper', async () => {
    for (const view of ['paint', 'wetness', 'deposited', 'suspended', 'velocity']) {
      const pixels = await runEffect({ view }, 3);
      expect(new Set(pixels).size).toBeGreaterThan(4);
    }
    for (const paper of Object.keys(PAPERS)) {
      const pixels = await runEffect({ paper }, 3);
      expect(new Set(pixels).size).toBeGreaterThan(8);
    }
  });

  it('paints strokes from pointer events, in ink and in clean water, and a stroke is not the same picture as none', async () => {
    const stroke = (frame: number, canvas: StubCanvas) => {
      if (frame === 3) {
        canvas.dispatchEvent(pointerEvent('pointerenter', 0.2, 0.5, false));
        canvas.dispatchEvent(pointerEvent('pointermove', 0.2, 0.5, false));
        canvas.dispatchEvent(pointerEvent('pointerdown', 0.2, 0.5, true));
      } else if (frame > 3 && frame < 20) {
        canvas.dispatchEvent(pointerEvent('pointermove', 0.2 + (frame - 3) * 0.04, 0.5, true));
      } else if (frame === 20) {
        canvas.dispatchEvent(pointerEvent('pointerup', 0.84, 0.5, false));
        canvas.dispatchEvent(pointerEvent('pointerleave', 0.84, 0.5, false));
      }
    };
    const quiet = { ambient: 0, ink: 'crimson' };
    const without = await runEffect(quiet, 40);
    const withInk = await runEffect(quiet, 40, stroke);
    const withWater = await runEffect({ ...quiet, brush: 'clean water' }, 40, stroke);
    expect(differing(without, withInk)).toBeGreaterThan(40);
    expect(differing(without, withWater)).toBeGreaterThan(0);
    expect(differing(withInk, withWater)).toBeGreaterThan(40);
    // Touch, multi-touch and a hover mode do not break anything.
    const touch = (frame: number, canvas: StubCanvas) => {
      for (const id of [3, 4]) {
        const type = frame === 2 ? 'pointerdown' : frame < 10 ? 'pointermove' : 'pointerup';
        canvas.dispatchEvent(
          Object.assign(pointerEvent(type, 0.3 + id * 0.1, 0.2 + frame * 0.03, frame < 10, id), {
            pointerType: 'touch',
          }),
        );
      }
    };
    await runEffect({ touch: 'paints', pointer: 'hover stirs' }, 14, touch);
    await runEffect({ pointer: 'off' }, 6, stroke);
  });

  it('runs a stroke at extreme parameters without a validation error', async () => {
    const stroke = (frame: number, canvas: StubCanvas) => {
      if (frame === 1) canvas.dispatchEvent(pointerEvent('pointerdown', 0.5, 0.5, true));
      if (frame > 1 && frame < 8)
        canvas.dispatchEvent(pointerEvent('pointermove', 0.5 + frame * 0.05, 0.5, true));
    };
    const pixels = await runEffect(
      {
        diffusion: 3e-4,
        viscosity: 0.03,
        swirl: 20,
        settling: 1,
        absorption: 4,
        drying: 0.5,
        dryDrag: 40,
        edgeDarkening: 1,
        granulation: 1,
        lift: 1,
        load: 2,
        water: 2,
        size: 0.1,
        strength: 2.5,
        stepsPerFrame: 5,
      },
      10,
      stroke,
    );
    expect(new Set(pixels).size).toBeGreaterThan(8);
  });
});

// --- Study ---------------------------------------------------------------------------------------------------------------------
//
// The cost numbers in src/effects/ink-water/notes.md. Skipped unless INKWATER_STUDY names a file to write
// the report to:  INKWATER_STUDY=/tmp/inkwater-study.txt npx vitest run tests/ink-water.test.ts -t study
// The method is the one of the solver's study: GPU timestamps, the span from the first timed pass starting to
// the last ending, the slope between a low and a high number of steps (which removes the ~1 ms floor of a
// pass), the median of 7 submits.

const studyFile = (globalThis as { process?: { env: Record<string, string | undefined> } }).process
  ?.env.INKWATER_STUDY;

describe.skipIf(!studyFile)('study', () => {
  it('cost per step and per frame', async () => {
    const adapter = await navigator.gpu.requestAdapter();
    const timed = !!adapter?.features.has('timestamp-query');
    const device = timed
      ? await adapter!.requestDevice({ requiredFeatures: ['timestamp-query'] })
      : await getTestDevice();
    const lines: string[] = [];
    const log = (text = '') => lines.push(text);
    log(
      `COST (timestamp queries: ${timed}); milliseconds, minimum of 9 submits, slope between a low and a high number of steps`,
    );

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
    // The minimum, not the median: other work on the GPU only ever adds time, and the machine is shared.
    const median = async (f: () => Promise<number>) => {
      const samples: number[] = [];
      for (let i = 0; i < 9; i++) samples.push(await f());
      return Math.min(...samples);
    };
    const f = (x: number) => x.toFixed(3);
    const ASPECT = 1662 / 932;
    const cells = (short: number) => Math.round(short * ASPECT);

    /** Whole step of the solver (and, if `layer`, the pigment hooks), velocity `vel` and dye `dye` cells on the short side. */
    async function perStep(
      vel: number,
      dye: number,
      options: { layer?: Partial<PigmentParams>; fluid?: Record<string, number | string> } = {},
    ) {
      const at = (steps: number) =>
        median(() =>
          span((profiler) => {
            const fluid = createFluid({
              device,
              resolution: cells(vel),
              dyeResolution: cells(dye),
              width: 1662,
              height: 932,
              profiler,
              params: { dyeDissipation: 0, pressureIterations: 24, ...options.fluid },
            });
            const layer = options.layer
              ? createPigmentLayer({
                  device,
                  width: fluid.grid.dyeWidth,
                  height: fluid.grid.dyeHeight,
                  profiler,
                  params: options.layer,
                })
              : undefined;
            layer?.attach(fluid);
            return {
              run: (e) => {
                layer?.beginFrame(e);
                fluid.advance(e, steps);
              },
              destroy: () => {
                layer?.destroy();
                fluid.destroy();
              },
            };
          }),
        );
      return ((await at(40)) - (await at(4))) / 36;
    }

    const typical: Partial<PigmentParams> = {
      diffusion: 6e-5,
      feather: 0.6,
      edgeSpeed: 0.012,
      absorption: 0.7,
      granulation: 0.5,
      lift: 0.25,
      drying: 0.15,
      dryDrag: 14,
      maxSubsteps: 2,
    };
    log('Whole step, velocity grid / dye grid on the short side (16:9), MacCormack dye:');
    for (const [v, d] of [
      [64, 384],
      [64, 512],
      [96, 512],
    ] as const) {
      const solver = await perStep(v, d);
      const both = await perStep(v, d, { layer: typical });
      log(
        `  ${v} / ${d}: solver alone ${f(solver)}, with the pigment layer ${f(both)} (the layer adds ${f(both - solver)}), sub-steps ${pigmentSubsteps(typical as PigmentParams, 1 / 60, d)}`,
      );
    }
    log('Semi-Lagrangian dye advection instead of MacCormack, 64 / 384:');
    log(
      `  solver alone ${f(await perStep(64, 384, { fluid: { dyeScheme: 'semi-lagrangian' } }))}, with the layer ${f(await perStep(64, 384, { layer: typical, fluid: { dyeScheme: 'semi-lagrangian' } }))}`,
    );
    log(
      'Diffusion sub-steps (pigment layer, 64 / 384; the layer alone is the difference to the solver alone):',
    );
    const solver = await perStep(64, 384);
    for (const n of [1, 2, 3, 4]) {
      const D = (n * W_CAP * 60) / (384 * 384);
      const total = await perStep(64, 384, {
        layer: { ...typical, diffusion: D, maxSubsteps: n },
      });
      log(`  ${n} sub-step${n > 1 ? 's' : ''}: layer ${f(total - solver)} ms per step`);
    }
    log(
      'Viscosity (8 Jacobi iterations) and vorticity confinement, the solver features the presets use, 64 / 384:',
    );
    log(
      `  viscosity ${f((await perStep(64, 384, { layer: typical, fluid: { viscosity: 0.004, viscosityIterations: 8 } })) - solver)} over solver alone with the layer, vorticity ${f((await perStep(64, 384, { layer: typical, fluid: { vorticity: 8 } })) - solver)}`,
    );

    log(
      'Whole frame of the effect at 1662 x 932 (the browser pane at DPR 2): first timed pass start to last end,',
    );
    log(
      'slope in steps per frame between 1 and 9; the intercept is the one step plus the fixed passes (inject, shade).',
    );
    for (const [name, overrides] of [
      ['default (wet-on-wet)', {}],
      ['ink drop in water', effect.presets!['ink drop in water']!],
      ['sumi-e', effect.presets!['sumi-e']!],
    ] as const) {
      const at = (steps: number) =>
        median(() =>
          span((profiler) => {
            const canvas = Object.assign(new EventTarget(), {
              width: 1662,
              height: 932,
              style: {},
              getBoundingClientRect: () => ({ left: 0, top: 0, width: 1662, height: 932 }),
              setPointerCapture() {},
              releasePointerCapture() {},
            }) as unknown as HTMLCanvasElement;
            const target = device.createTexture({
              size: [1662, 932],
              format: 'rgba8unorm',
              usage: GPUTextureUsage.RENDER_ATTACHMENT,
            });
            let instance: Awaited<ReturnType<typeof effect.create>> | undefined;
            const ready = effect
              .create({
                device,
                format: 'rgba8unorm',
                canvas,
                params: {
                  ...defaultParams(effect.params),
                  ...overrides,
                  stepsPerFrame: steps,
                } as never,
                profiler,
                hdr: false,
              })
              .then((made) => {
                instance = made;
                made.resize(1662, 932, 2);
              });
            return {
              run: (e) => {
                void ready;
                instance!.frame({ time: 0, delta: 1 / 60, frame: 0 }, e, target.createView());
              },
              destroy: () => {
                instance?.destroy();
                target.destroy();
              },
            };
          }),
        );
      void at;
      // `create` is asynchronous, so build the instance first, then time frames of it.
      const frameAt = async (steps: number) => {
        const samples: number[] = [];
        for (let i = 0; i < 9; i++) {
          const querySet = device.createQuerySet({ type: 'timestamp', count: 64 });
          let next = 0;
          const profiler = {
            pass: () => {
              const begin = next;
              next += 2;
              return { querySet, beginningOfPassWriteIndex: begin, endOfPassWriteIndex: begin + 1 };
            },
          };
          const canvas = Object.assign(new EventTarget(), {
            width: 1662,
            height: 932,
            style: {},
            getBoundingClientRect: () => ({ left: 0, top: 0, width: 1662, height: 932 }),
            setPointerCapture() {},
            releasePointerCapture() {},
          }) as unknown as HTMLCanvasElement;
          const target = device.createTexture({
            size: [1662, 932],
            format: 'rgba8unorm',
            usage: GPUTextureUsage.RENDER_ATTACHMENT,
          });
          const instance = await effect.create({
            device,
            format: 'rgba8unorm',
            canvas,
            params: {
              ...defaultParams(effect.params),
              ...overrides,
              ambient: 0,
              stepsPerFrame: steps,
            } as never,
            profiler,
            hdr: false,
          });
          instance.resize(1662, 932, 2);
          // One warm-up frame so pipelines and bind groups exist, then the measured one.
          const warm = device.createCommandEncoder();
          instance.frame({ time: 0, delta: 1 / 60, frame: 0 }, warm, target.createView());
          device.queue.submit([warm.finish()]);
          next = 0;
          const resolve = device.createBuffer({
            size: 512,
            usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC,
          });
          const read = device.createBuffer({
            size: 512,
            usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
          });
          const encoder = device.createCommandEncoder();
          instance.frame({ time: 1 / 60, delta: 1 / 60, frame: 1 }, encoder, target.createView());
          encoder.resolveQuerySet(querySet, 0, 64, resolve, 0);
          encoder.copyBufferToBuffer(resolve, 0, read, 0, 512);
          device.queue.submit([encoder.finish()]);
          await read.mapAsync(GPUMapMode.READ);
          const stamps = new BigUint64Array(read.getMappedRange().slice(0));
          read.unmap();
          const used = Array.from(stamps.slice(0, next)).filter((t) => t > 0n);
          const first = used.reduce((a, b) => (a < b ? a : b));
          const last = used.reduce((a, b) => (a > b ? a : b));
          samples.push(Number(last - first) / 1e6);
          instance.destroy();
          target.destroy();
        }
        return Math.min(...samples);
      };
      const one = await frameAt(1);
      const nine = await frameAt(9);
      const slope = (nine - one) / 8;
      log(
        `  ${name}: ${f(one)} ms at 1 step per frame, ${f(nine)} at 9; ${f(slope)} per step, so ${f(one - slope)} for the rest of the frame`,
      );
    }
    log();

    const specifier = 'node:fs';
    const { writeFileSync } = (await import(/* @vite-ignore */ specifier)) as {
      writeFileSync(path: string, text: string): void;
    };
    writeFileSync(studyFile!, lines.join('\n') + '\n');
  });

  it('the non-simulated ink-wash at the same size, for the comparison', async () => {
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter?.features.has('timestamp-query')) return;
    const device = await adapter.requestDevice({ requiredFeatures: ['timestamp-query'] });
    const samples: number[] = [];
    for (let i = 0; i < 9; i++) {
      const querySet = device.createQuerySet({ type: 'timestamp', count: 16 });
      let next = 0;
      const profiler = {
        pass: () => {
          const begin = next;
          next += 2;
          return { querySet, beginningOfPassWriteIndex: begin, endOfPassWriteIndex: begin + 1 };
        },
      };
      const canvas = Object.assign(new EventTarget(), {
        width: 1662,
        height: 932,
        style: {},
        getBoundingClientRect: () => ({ left: 0, top: 0, width: 1662, height: 932 }),
      }) as unknown as HTMLCanvasElement;
      const target = device.createTexture({
        size: [1662, 932],
        format: 'rgba8unorm',
        usage: GPUTextureUsage.RENDER_ATTACHMENT,
      });
      const instance = await washEffect.create({
        device,
        format: 'rgba8unorm',
        canvas,
        params: defaultParams(washEffect.params) as never,
        profiler,
        hdr: false,
      });
      instance.resize(1662, 932, 2);
      const warm = device.createCommandEncoder();
      instance.frame({ time: 0, delta: 1 / 60, frame: 0 }, warm, target.createView());
      device.queue.submit([warm.finish()]);
      next = 0;
      const resolve = device.createBuffer({
        size: 256,
        usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC,
      });
      const read = device.createBuffer({
        size: 256,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
      });
      const encoder = device.createCommandEncoder();
      instance.frame({ time: 1 / 60, delta: 1 / 60, frame: 1 }, encoder, target.createView());
      encoder.resolveQuerySet(querySet, 0, 16, resolve, 0);
      encoder.copyBufferToBuffer(resolve, 0, read, 0, 256);
      device.queue.submit([encoder.finish()]);
      await read.mapAsync(GPUMapMode.READ);
      const stamps = new BigUint64Array(read.getMappedRange().slice(0));
      read.unmap();
      const used = Array.from(stamps.slice(0, next)).filter((t) => t > 0n);
      samples.push(
        Number(used.reduce((a, b) => (a > b ? a : b)) - used.reduce((a, b) => (a < b ? a : b))) /
          1e6,
      );
      instance.destroy();
      target.destroy();
    }
    const specifier = 'node:fs';
    const { appendFileSync } = (await import(/* @vite-ignore */ specifier)) as {
      appendFileSync(path: string, text: string): void;
    };
    appendFileSync(
      studyFile!,
      `ink-wash (non-simulated, defaults) at 1662 x 932: ${Math.min(...samples).toFixed(3)} ms for the whole frame (minimum of 9)\n`,
    );
  });
});
