import { describe, expect, it, vi } from 'vitest';
import { configureCanvasHdr, displaySupportsHdr } from '../src/gpu/canvas.ts';
import { blit } from '../src/gpu/passes/blit.ts';
import {
  bloomLevelSizes,
  createBloom,
  createOutputPass,
  createPostChain,
  MAX_BLOOM_LEVELS,
  type BloomSettings,
  type OutputSettings,
} from '../src/gpu/post/index.ts';
import { expectValid, getTestDevice, readTexture } from './gpu.ts';
import { gpuF32, lane, rowsOf } from './wgsl-eval.ts';

const USAGE =
  GPUTextureUsage.RENDER_ATTACHMENT |
  GPUTextureUsage.TEXTURE_BINDING |
  GPUTextureUsage.COPY_SRC |
  GPUTextureUsage.COPY_DST;

// --- Half-float helpers (rgba16float has no typed array in Node) ---------------------------------

function halfBits(value: number): number {
  if (value === 0) return 0;
  const bits = new Uint32Array(new Float32Array([value]).buffer)[0] ?? 0;
  return (
    ((bits >>> 16) & 0x8000) | ((((bits >>> 23) & 0xff) - 112) << 10) | ((bits >>> 13) & 0x3ff)
  );
}

function halfValue(bits: number): number {
  const sign = bits & 0x8000 ? -1 : 1;
  const exponent = (bits >> 10) & 0x1f;
  const mantissa = bits & 0x3ff;
  if (exponent === 0) return sign * 2 ** -14 * (mantissa / 1024);
  return sign * 2 ** (exponent - 15) * (1 + mantissa / 1024);
}

/** An rgba16float texture whose texel (x, y) is the rgb triple `colour(x, y)` with alpha 1. */
function halfTexture(
  device: GPUDevice,
  width: number,
  height: number,
  colour: (x: number, y: number) => readonly number[],
): GPUTexture {
  const texture = device.createTexture({
    size: [width, height],
    format: 'rgba16float',
    usage: USAGE,
  });
  const bits = new Uint16Array(width * height * 4);
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const [r = 0, g = 0, b = 0] = colour(x, y);
      bits.set([halfBits(r), halfBits(g), halfBits(b), halfBits(1)], (y * width + x) * 4);
    }
  }
  device.queue.writeTexture({ texture }, bits, { bytesPerRow: width * 8, rowsPerImage: height }, [
    width,
    height,
  ]);
  return texture;
}

async function readHalf(device: GPUDevice, texture: GPUTexture): Promise<number[]> {
  const bits = new Uint16Array((await readTexture(device, texture, 8)).buffer);
  return Array.from(bits, halfValue);
}

function submit(device: GPUDevice, record: (encoder: GPUCommandEncoder) => void) {
  const encoder = device.createCommandEncoder();
  record(encoder);
  device.queue.submit([encoder.finish()]);
}

const sum = (values: number[]) => values.reduce((a, b) => a + b, 0);
const rgb = (texels: number[]) => texels.filter((_, i) => i % 4 < 3);

// --- Bloom ---------------------------------------------------------------------------------------

const TIGHT: Partial<BloomSettings> = {
  intensity: 1,
  threshold: 0,
  knee: 0,
  radius: 0.6,
  levels: 4,
  fireflyFilter: false,
};

/** Runs the bloom chain on a source and returns the finest level as floats. */
async function runBloom(
  device: GPUDevice,
  width: number,
  height: number,
  colour: (x: number, y: number) => readonly number[],
  settings: Partial<BloomSettings>,
) {
  const source = halfTexture(device, width, height, colour);
  const bloom = createBloom(device, { width, height, readable: true });
  let view: GPUTextureView | undefined;
  await expectValid(device, () =>
    submit(device, (encoder) => {
      view = bloom.encode(encoder, source.createView(), settings);
    }),
  );
  const finest = bloom.levels[0];
  const texels = finest && view ? await readHalf(device, finest.texture) : [];
  const size = finest ? ([finest.width, finest.height] as const) : ([0, 0] as const);
  const result = { texels, size, view, bloom };
  source.destroy();
  return result;
}

describe('bloom level sizes', () => {
  it('starts at half size and stops while the smallest level still has a few texels', () => {
    expect(bloomLevelSizes(1920, 1080)).toEqual([
      [960, 540],
      [480, 270],
      [240, 135],
      [120, 68],
      [60, 34],
      [30, 17],
      [15, 9],
      [8, 5],
    ]);
    expect(bloomLevelSizes(1920, 1080).length).toBeLessThanOrEqual(MAX_BLOOM_LEVELS);
    expect(bloomLevelSizes(16, 16)).toEqual([
      [8, 8],
      [4, 4],
    ]);
    expect(bloomLevelSizes(4, 4)).toEqual([[2, 2]]);
    expect(bloomLevelSizes(1, 1)).toEqual([[1, 1]]);
    expect(bloomLevelSizes(64, 64, false)[0]).toEqual([64, 64]);
  });
});

/** A 2x2 block of `value` on the centre of a 64x64 image, black elsewhere. */
const block = (value: number) => (x: number, y: number) =>
  x >= 31 && x <= 32 && y >= 31 && y <= 32 ? [value, value, value] : [0, 0, 0];

describe('bloom', () => {
  it('does nothing at all when intensity is 0', async () => {
    const device = await getTestDevice();
    const source = halfTexture(device, 16, 16, () => [5, 5, 5]);
    const bloom = createBloom(device, { width: 16, height: 16 });
    let view: GPUTextureView | undefined = source.createView();
    await expectValid(device, () =>
      submit(device, (encoder) => {
        view = bloom.encode(encoder, source.createView(), { intensity: 0 });
      }),
    );
    expect(view).toBeUndefined();
    expect(bloom.levelSizes()).toEqual([]);
    bloom.destroy();
    source.destroy();
  });

  it('leaves the image untouched through the post chain at intensity 0', async () => {
    const device = await getTestDevice();
    const colour = (x: number, y: number) => [0.1 + x / 40, 0.4, 3 * (y % 2)];
    const render = async (bloom: Partial<BloomSettings> | false) => {
      const chain = createPostChain(device, { canvasFormat: 'rgba8unorm', width: 16, height: 16 });
      const scene = halfTexture(device, 16, 16, colour);
      const target = device.createTexture({ size: [16, 16], format: 'rgba8unorm', usage: USAGE });
      await expectValid(device, () =>
        submit(device, (encoder) => {
          // Draw the known scene into the chain's target, then run the chain.
          blit(device, encoder, scene.createView(), chain.scene, { filter: 'nearest' });
          chain.encode(encoder, target.createView(), { bloom, output: { dither: 0 } });
        }),
      );
      const pixels = await readTexture(device, target);
      chain.destroy();
      return pixels;
    };
    expect([...(await render({ intensity: 0 }))]).toEqual([...(await render(false))]);
  });

  it('spreads a bright spot symmetrically and conserves its energy', async () => {
    const device = await getTestDevice();
    // Even sizes keep the 2:1 phase the 13-tap filter is built for (see the notes); the 2x2 block
    // sits exactly on the image centre, so every level stays mirror-symmetric.
    const n = 64;
    const { texels, size, bloom } = await runBloom(device, n, n, block(100), TIGHT);
    const [w, h] = size;
    expect([w, h]).toEqual([32, 32]);
    expect(texels.every((v) => Number.isFinite(v) && v >= 0)).toBe(true);

    const at = (x: number, y: number, c = 0) => texels[(y * w + x) * 4 + c] ?? Number.NaN;
    const peak = Math.max(...rgb(texels));
    expect(peak).toBeGreaterThan(1);
    expect(at(15, 15)).toBeGreaterThan(at(12, 15));
    for (let y = 0; y < h; y++) {
      for (let x = 0; x < w; x++) {
        const tolerance = 2e-3 * peak + 1e-6;
        expect(Math.abs(at(x, y) - at(w - 1 - x, y))).toBeLessThanOrEqual(tolerance);
        expect(Math.abs(at(x, y) - at(x, h - 1 - y))).toBeLessThanOrEqual(tolerance);
        expect(Math.abs(at(x, y) - at(y, x))).toBeLessThanOrEqual(tolerance);
        // The three channels were equal in, so they are equal out.
        expect(at(x, y, 1)).toBeCloseTo(at(x, y, 0), 2);
      }
    }
    // Never less energy than went in: each level's mean is preserved and the blend is convex, so
    // the finest level's total (scaled by the pixel-count ratio) matches the input's 4 * 3 * 100.
    const energy = sum(rgb(texels)) * ((n * n) / (w * h));
    expect(energy).toBeGreaterThanOrEqual(1200 * 0.97);
    expect(energy).toBeLessThanOrEqual(1200 * 1.03);
    bloom.destroy();
  });

  it('with the firefly filter on, a lone hot pixel is damped but still glows', async () => {
    const device = await getTestDevice();
    const hot = (x: number, y: number) => (x === 31 && y === 31 ? [100, 100, 100] : [0, 0, 0]);
    const off = await runBloom(device, 64, 64, hot, TIGHT);
    const on = await runBloom(device, 64, 64, hot, { ...TIGHT, fireflyFilter: true });
    const energyOff = sum(rgb(off.texels));
    const energyOn = sum(rgb(on.texels));
    expect(energyOn).toBeGreaterThan(0);
    expect(energyOn).toBeLessThan(energyOff * 0.5);
    // Without the filter the isolated pixel keeps all its energy.
    expect(energyOff * 4).toBeGreaterThanOrEqual(300 * 0.95);
    off.bloom.destroy();
    on.bloom.destroy();
  });

  it('adds nothing for pixels below the threshold, with or without a hot neighbour elsewhere', async () => {
    const device = await getTestDevice();
    const dim = await runBloom(device, 32, 32, () => [0.8, 0.8, 0.8], {
      ...TIGHT,
      threshold: 1,
      knee: 0,
    });
    expect(Math.max(...rgb(dim.texels))).toBe(0);
    dim.bloom.destroy();

    // The knee only starts passing light at threshold - knee, and passes a small, known amount.
    const soft = await runBloom(device, 32, 32, () => [0.9, 0.9, 0.9], {
      ...TIGHT,
      threshold: 1,
      knee: 0.2,
    });
    // soft = (0.9 - 1 + 0.2)^2 / (4 * 0.2) = 0.0125, kept = 0.0125 / 0.9, bloom = 0.9 * kept
    expect(soft.texels[0]).toBeCloseTo(0.0125, 3);
    soft.bloom.destroy();
    const below = await runBloom(device, 32, 32, () => [0.7, 0.7, 0.7], {
      ...TIGHT,
      threshold: 1,
      knee: 0.2,
    });
    expect(Math.max(...rgb(below.texels))).toBe(0);
    below.bloom.destroy();
  });

  it('keeps a constant image above the threshold constant: the excess over the threshold', async () => {
    const device = await getTestDevice();
    const { texels, bloom } = await runBloom(device, 48, 48, () => [2, 2, 2], {
      ...TIGHT,
      threshold: 1,
      knee: 0,
      levels: 5,
      fireflyFilter: true,
    });
    // kept = (2 - 1) / 2, bloom = 2 * kept = 1 everywhere, whatever the number of levels.
    for (const v of rgb(texels)) expect(v).toBeCloseTo(1, 2);
    bloom.destroy();
  });

  it('scales linearly with intensity', async () => {
    const device = await getTestDevice();
    const one = await runBloom(device, 64, 64, block(40), { ...TIGHT, intensity: 1 });
    const half = await runBloom(device, 64, 64, block(40), { ...TIGHT, intensity: 0.5 });
    expect(Math.max(...rgb(one.texels))).toBeGreaterThan(0.5);
    one.texels.forEach((v, i) => expect(half.texels[i]).toBeCloseTo(i % 4 === 3 ? v : v * 0.5, 1));
    one.bloom.destroy();
    half.bloom.destroy();
  });

  it('moves energy to wider levels as radius grows, without adding any', async () => {
    const device = await getTestDevice();
    const tight = await runBloom(device, 64, 64, block(100), { ...TIGHT, radius: 0 });
    const wide = await runBloom(device, 64, 64, block(100), { ...TIGHT, radius: 0.95 });
    expect(Math.max(...rgb(wide.texels))).toBeLessThan(Math.max(...rgb(tight.texels)));
    const total = (texels: number[]) => sum(rgb(texels));
    expect(total(wide.texels) / total(tight.texels)).toBeGreaterThan(0.93);
    expect(total(wide.texels) / total(tight.texels)).toBeLessThan(1.07);
    tight.bloom.destroy();
    wide.bloom.destroy();
  });

  it('recreates its targets on resize and only then', async () => {
    const device = await getTestDevice();
    const source = halfTexture(device, 64, 64, () => [3, 3, 3]);
    const bloom = createBloom(device, { width: 64, height: 64 });
    await expectValid(device, () =>
      submit(device, (e) => bloom.encode(e, source.createView(), TIGHT)),
    );
    expect(bloom.levelSizes()).toEqual([
      [32, 32],
      [16, 16],
      [8, 8],
      [4, 4],
    ]);
    const before = bloom.levels.map((level) => level.texture);

    expect(bloom.resize(64, 64)).toBe(false);
    expect(bloom.levels.map((level) => level.texture)).toEqual(before);

    expect(bloom.resize(40, 24)).toBe(true);
    expect(bloom.levelSizes()).toEqual([
      [20, 12],
      [10, 6],
    ]);
    expect(bloom.levels[0]?.texture).not.toBe(before[0]);

    // Encoding after the resize works and uses the new sizes (a 40x24 source).
    const resized = halfTexture(device, 40, 24, () => [3, 3, 3]);
    await expectValid(device, () =>
      submit(device, (e) => bloom.encode(e, resized.createView(), TIGHT)),
    );
    expect(bloom.levelSizes()[0]).toEqual([20, 12]);
    bloom.destroy();
    expect(bloom.levels.length).toBe(0);
    source.destroy();
    resized.destroy();
  });

  it('clamps the requested levels to what the image allows', async () => {
    const device = await getTestDevice();
    const source = halfTexture(device, 16, 16, () => [3, 3, 3]);
    const bloom = createBloom(device, { width: 16, height: 16 });
    await expectValid(device, () =>
      submit(device, (e) => bloom.encode(e, source.createView(), { ...TIGHT, levels: 99 })),
    );
    expect(bloom.levelSizes()).toEqual([
      [8, 8],
      [4, 4],
    ]);
    bloom.destroy();
    source.destroy();
  });
});

// --- Output pass ---------------------------------------------------------------------------------

type Rgb = readonly [number, number, number];

/** Renders `scene` (and `bloom`) as a width x 1 strip through the output pass. */
async function renderOutput(
  device: GPUDevice,
  scene: readonly Rgb[],
  settings: OutputSettings,
  options: { bloom?: readonly Rgb[]; format?: GPUTextureFormat } = {},
) {
  const format = options.format ?? 'rgba8unorm';
  const width = scene.length;
  const sceneTexture = halfTexture(device, width, 1, (x) => scene[x] ?? [0, 0, 0]);
  const bloomTexture = options.bloom
    ? halfTexture(device, width, 1, (x) => options.bloom?.[x] ?? [0, 0, 0])
    : undefined;
  const target = device.createTexture({ size: [width, 1], format, usage: USAGE });
  const pass = createOutputPass(device, { format });
  await expectValid(device, () =>
    submit(device, (encoder) =>
      pass.encode(
        encoder,
        target.createView(),
        sceneTexture.createView(),
        bloomTexture?.createView(),
        settings,
      ),
    ),
  );
  const output =
    format === 'rgba16float'
      ? await readHalf(device, target)
      : Array.from(await readTexture(device, target));
  pass.destroy();
  for (const texture of [sceneTexture, bloomTexture, target]) texture?.destroy();
  return output;
}

/** What color.wgsl says the pipeline should produce, as 8-bit values, computed on the GPU. */
async function expected8(tonemap: string, exposure: number, inputs: readonly Rgb[]) {
  const mapped =
    tonemap === 'aces'
      ? 'tonemap_aces(c)'
      : tonemap === 'reinhard'
        ? 'tonemap_reinhard(c)'
        : 'clamp(c, vec3f(0.0), vec3f(1.0))';
  const out = await gpuF32(
    `#include "color.wgsl"
     fn f(x: vec4f) -> vec4f {
       let c = x.rgb * ${exposure.toFixed(6)};
       return vec4f(linear_to_srgb(${mapped}), 1.0);
     }`,
    rowsOf(inputs.map((c) => [...c])),
  );
  return [0, 1, 2].map((channel) => lane(out, channel).map((v) => v * 255));
}

const HDR_INPUTS: Rgb[] = [
  [0, 0, 0],
  [0.02, 0.05, 0.1],
  [0.18, 0.18, 0.18],
  [0.5, 0.25, 0.75],
  [1, 1, 1],
  [2, 4, 8],
  [20, 0.5, 100],
];

describe('output pass', () => {
  it.each(['aces', 'reinhard', 'none'] as const)(
    'matches color.wgsl %s followed by sRGB encoding',
    async (tonemap) => {
      const device = await getTestDevice();
      const [reds, greens, blues] = await expected8(tonemap, 1, HDR_INPUTS);
      const pixels = await renderOutput(device, HDR_INPUTS, { tonemap, dither: 0 });
      HDR_INPUTS.forEach((_, i) => {
        expect(Math.abs((pixels[i * 4] ?? -9) - (reds?.[i] ?? 0))).toBeLessThanOrEqual(1);
        expect(Math.abs((pixels[i * 4 + 1] ?? -9) - (greens?.[i] ?? 0))).toBeLessThanOrEqual(1);
        expect(Math.abs((pixels[i * 4 + 2] ?? -9) - (blues?.[i] ?? 0))).toBeLessThanOrEqual(1);
        expect(pixels[i * 4 + 3]).toBe(255);
      });
    },
  );

  it('multiplies by exposure before tonemapping', async () => {
    const device = await getTestDevice();
    const inputs: Rgb[] = [
      [0.05, 0.1, 0.2],
      [0.3, 0.6, 1.5],
    ];
    for (const exposure of [0.25, 2, 4]) {
      const [reds, greens, blues] = await expected8('aces', exposure, inputs);
      const pixels = await renderOutput(device, inputs, { tonemap: 'aces', exposure, dither: 0 });
      inputs.forEach((_, i) => {
        expect(Math.abs((pixels[i * 4] ?? -9) - (reds?.[i] ?? 0))).toBeLessThanOrEqual(1);
        expect(Math.abs((pixels[i * 4 + 1] ?? -9) - (greens?.[i] ?? 0))).toBeLessThanOrEqual(1);
        expect(Math.abs((pixels[i * 4 + 2] ?? -9) - (blues?.[i] ?? 0))).toBeLessThanOrEqual(1);
      });
    }
    // Documented behaviour: with `none`, 0.2 at exposure 2 is exactly 0.4 before encoding.
    const doubled = await renderOutput(device, [[0.2, 0.2, 0.2]], {
      tonemap: 'none',
      exposure: 2,
      dither: 0,
    });
    const direct = await renderOutput(device, [[0.4, 0.4, 0.4]], {
      tonemap: 'none',
      exposure: 1,
      dither: 0,
    });
    expect([...doubled]).toEqual([...direct]);
  });

  it('is the identity up to sRGB encoding for none, exposure 1 and values at most 1', async () => {
    const device = await getTestDevice();
    const levels = [0, 0.001, 0.0031308, 0.1, 0.2140411, 0.5, 0.8, 1];
    const inputs = levels.map((v): Rgb => [v, v, v]);
    const pixels = await renderOutput(device, inputs, { tonemap: 'none', exposure: 1, dither: 0 });
    levels.forEach((v, i) => {
      const encoded = v <= 0.0031308 ? v * 12.92 : 1.055 * v ** (1 / 2.4) - 0.055;
      expect(Math.abs((pixels[i * 4] ?? -9) - encoded * 255)).toBeLessThanOrEqual(0.75);
    });
  });

  it('adds the bloom to the scene before exposure and tonemapping', async () => {
    const device = await getTestDevice();
    const settings: OutputSettings = { tonemap: 'none', exposure: 2, dither: 0 };
    const withBloom = await renderOutput(device, [[0.05, 0.05, 0.05]], settings, {
      bloom: [[0.1, 0.1, 0.1]],
    });
    const summed = await renderOutput(device, [[0.15, 0.15, 0.15]], { ...settings, exposure: 2 });
    expect([...withBloom]).toEqual([...summed]);

    const sceneOnly = await renderOutput(
      device,
      [[0.05, 0.05, 0.05]],
      { ...settings, view: 'scene' },
      {
        bloom: [[0.1, 0.1, 0.1]],
      },
    );
    const plain = await renderOutput(device, [[0.05, 0.05, 0.05]], settings);
    expect([...sceneOnly]).toEqual([...plain]);
    const bloomOnly = await renderOutput(
      device,
      [[0.05, 0.05, 0.05]],
      { ...settings, view: 'bloom' },
      {
        bloom: [[0.1, 0.1, 0.1]],
      },
    );
    const bloomAlone = await renderOutput(device, [[0.1, 0.1, 0.1]], settings);
    expect([...bloomOnly]).toEqual([...bloomAlone]);
  });

  it('only ever brightens when bloom is added', async () => {
    const device = await getTestDevice();
    const scene: Rgb[] = [0, 0.05, 0.2, 0.5, 0.9].map((v) => [v, v, v]);
    const glow: Rgb[] = [0, 0.3, 0.01, 0.2, 0].map((v) => [v, v, v]);
    const settings: OutputSettings = { tonemap: 'aces', dither: 0 };
    const without = await renderOutput(device, scene, settings);
    const withGlow = await renderOutput(device, scene, settings, { bloom: glow });
    without.forEach((v, i) => expect(withGlow[i]).toBeGreaterThanOrEqual(v));
    expect(sum(withGlow)).toBeGreaterThan(sum(without));
  });

  it('dithers with zero-mean noise within a couple of steps and hides banding', async () => {
    const device = await getTestDevice();
    // linear 0.3 encodes to 148.87 in 8 bits: without dither every pixel is 149.
    const flat: Rgb[] = Array.from({ length: 4096 }, () => [0.3, 0.3, 0.3]);
    const ideal = (1.055 * 0.3 ** (1 / 2.4) - 0.055) * 255;
    const plain = await renderOutput(device, flat, { tonemap: 'none', dither: 0 });
    expect(new Set(Array.from({ length: 4096 }, (_, i) => plain[i * 4])).size).toBe(1);

    const dithered = await renderOutput(device, flat, { tonemap: 'none', dither: 1, frame: 3 });
    const reds = Array.from({ length: 4096 }, (_, i) => dithered[i * 4] ?? 0);
    expect(new Set(reds).size).toBeGreaterThan(2);
    for (const v of reds) expect(Math.abs(v - ideal)).toBeLessThanOrEqual(2);
    const mean = sum(reds) / reds.length;
    expect(Math.abs(mean - ideal)).toBeLessThan(0.1);

    // A different frame number gives a different pattern.
    const other = await renderOutput(device, flat, { tonemap: 'none', dither: 1, frame: 4 });
    expect(Array.from(other).join()).not.toBe(Array.from(dithered).join());
  });

  describe('HDR mode', () => {
    const decode = (encoded: number) =>
      encoded <= 0.04045 ? encoded / 12.92 : ((encoded + 0.055) / 1.055) ** 2.4;
    const run = async (values: number[], settings: OutputSettings) =>
      (
        await renderOutput(
          device0,
          values.map((v): Rgb => [v, v, v]),
          { dither: 0, ...settings },
          {
            format: 'rgba16float',
          },
        )
      )
        .filter((_, i) => i % 4 === 0)
        .map(decode);
    let device0: GPUDevice;

    it.each(['aces', 'reinhard', 'none'] as const)(
      'matches SDR up to scene white, then rises past 1 and stays under the headroom (%s)',
      async (tonemap) => {
        device0 = await getTestDevice();
        const dim = [0.01, 0.1, 0.3, 0.6, 0.9, 1];
        const sdr = await run(dim, { tonemap, hdr: false });
        const hdr = await run(dim, { tonemap, hdr: true, headroom: 3 });
        dim.forEach((_, i) => expect(hdr[i]).toBeCloseTo(sdr[i] ?? -1, 2));

        const bright = [1, 1.5, 2, 4, 8, 20, 100, 1000];
        const out = await run(bright, { tonemap, hdr: true, headroom: 3 });
        for (let i = 1; i < out.length; i++) {
          expect(out[i]).toBeGreaterThanOrEqual((out[i - 1] ?? 0) - 1e-3);
        }
        for (const v of out) expect(v).toBeLessThanOrEqual(3 * 1.005);
        if (tonemap === 'none') {
          // No tonemap: scene values pass through up to the shoulder, so 2 is brighter than SDR white.
          expect(out[2]).toBeGreaterThan(1.3);
        }
        // Far above white it approaches the headroom without exceeding it.
        expect(out[out.length - 1]).toBeGreaterThan(2.8);
        // SDR on the same input clips at 1.
        const clipped = await run([20, 100], { tonemap, hdr: false });
        for (const v of clipped) expect(v).toBeLessThanOrEqual(1.001);
      },
    );

    it('a headroom of 1 never goes above SDR white', async () => {
      device0 = await getTestDevice();
      const values = [0.2, 1, 5, 50, 5000];
      for (const tonemap of ['aces', 'none'] as const) {
        const flat = await run(values, { tonemap, hdr: true, headroom: 1 });
        for (const v of flat) expect(v).toBeLessThanOrEqual(1.005);
      }
    });
  });
});

// --- Chain ---------------------------------------------------------------------------------------

describe('post chain', () => {
  it('times each stage under its own label and releases everything', async () => {
    const device = await getTestDevice();
    const labels: string[] = [];
    const profiler = { pass: (label: string) => (labels.push(label), undefined) };
    const chain = createPostChain(device, { canvasFormat: 'rgba8unorm', width: 64, height: 48 });
    expect(chain.scene.format).toBe('rgba16float');
    expect(chain.width).toBe(64);
    const target = device.createTexture({ size: [64, 48], format: 'rgba8unorm', usage: USAGE });

    await expectValid(device, () =>
      submit(device, (encoder) =>
        chain.encode(encoder, target.createView(), { bloom: { levels: 3 } }, profiler),
      ),
    );
    expect(labels.filter((l) => l === 'bloom-down')).toHaveLength(3);
    expect(labels.filter((l) => l === 'bloom-up')).toHaveLength(2);
    expect(labels.filter((l) => l === 'output')).toHaveLength(1);

    labels.length = 0;
    await expectValid(device, () =>
      submit(device, (encoder) =>
        chain.encode(encoder, target.createView(), { bloom: false }, profiler),
      ),
    );
    expect(labels).toEqual(['output']);

    const oldView = chain.scene.view;
    expect(chain.resize(32, 24)).toBe(true);
    expect([chain.width, chain.height]).toEqual([32, 24]);
    expect(chain.scene.view).not.toBe(oldView);
    expect(chain.bloom.maxLevels()).toBe(bloomLevelSizes(32, 24).length);
    await expectValid(device, async () => {
      const small = device.createTexture({ size: [32, 24], format: 'rgba8unorm', usage: USAGE });
      submit(device, (encoder) => chain.encode(encoder, small.createView(), {}));
      small.destroy();
    });
    await expectValid(device, () => chain.destroy());
    target.destroy();
  });
});

// --- HDR canvas opt-in ---------------------------------------------------------------------------

describe('HDR canvas configuration', () => {
  const device = {} as GPUDevice;
  function fakeCanvas(context: object) {
    return { getContext: vi.fn(() => context) } as unknown as HTMLCanvasElement;
  }
  const extended = (format = 'rgba16float') => ({ format, toneMapping: { mode: 'extended' } });

  it('is unavailable under Node, so requesting HDR falls back to the SDR configuration', () => {
    expect(displaySupportsHdr()).toBe(false);
    const context = { configure: vi.fn(), getConfiguration: vi.fn() };
    const result = configureCanvasHdr(fakeCanvas(context), device, 'bgra8unorm');
    expect(result).toMatchObject({ hdr: false, format: 'bgra8unorm', context });
    expect(result.reason).toMatch(/not HDR capable/);
    // Exactly the call an SDR effect gets: no tone mapping member, premultiplied alpha.
    expect(context.configure).toHaveBeenCalledTimes(1);
    expect(context.configure).toHaveBeenCalledWith({
      device,
      format: 'bgra8unorm',
      alphaMode: 'premultiplied',
    });
    expect(context.getConfiguration).not.toHaveBeenCalled();
  });

  it('uses rgba16float with extended tone mapping and premultiplied alpha when confirmed', () => {
    const context = { configure: vi.fn(), getConfiguration: vi.fn(() => extended()) };
    const result = configureCanvasHdr(fakeCanvas(context), device, 'bgra8unorm', {
      displayIsHdr: () => true,
    });
    expect(result).toEqual({ context, format: 'rgba16float', hdr: true });
    expect(context.configure).toHaveBeenCalledWith({
      device,
      format: 'rgba16float',
      alphaMode: 'premultiplied',
      toneMapping: { mode: 'extended' },
    });
  });

  it('falls back when the browser ignores the tone mapping member', () => {
    // Firefox-style: unknown dictionary members are dropped and getConfiguration has no toneMapping.
    const context = {
      configure: vi.fn(),
      getConfiguration: vi.fn(() => ({ format: 'rgba16float' })),
    };
    const result = configureCanvasHdr(fakeCanvas(context), device, 'bgra8unorm', {
      displayIsHdr: () => true,
    });
    expect(result).toMatchObject({ hdr: false, format: 'bgra8unorm' });
    expect(context.configure).toHaveBeenCalledTimes(2);
    expect(context.configure).toHaveBeenLastCalledWith({
      device,
      format: 'bgra8unorm',
      alphaMode: 'premultiplied',
    });
  });

  it('falls back when the mode is only standard, or getConfiguration does not exist', () => {
    const standard = {
      configure: vi.fn(),
      getConfiguration: vi.fn(() => ({ format: 'rgba16float', toneMapping: { mode: 'standard' } })),
    };
    expect(
      configureCanvasHdr(fakeCanvas(standard), device, 'rgba8unorm', { displayIsHdr: () => true })
        .hdr,
    ).toBe(false);
    const bare = { configure: vi.fn() };
    const result = configureCanvasHdr(fakeCanvas(bare), device, 'rgba8unorm', {
      displayIsHdr: () => true,
    });
    expect(result).toMatchObject({ hdr: false, format: 'rgba8unorm' });
  });

  it('falls back instead of throwing when configure rejects the HDR configuration', () => {
    const configure = vi.fn((config: { toneMapping?: unknown }) => {
      if (config.toneMapping) throw new TypeError('toneMapping is not supported');
    });
    const context = { configure, getConfiguration: vi.fn() };
    const result = configureCanvasHdr(fakeCanvas(context), device, 'bgra8unorm', {
      displayIsHdr: () => true,
    });
    expect(result).toMatchObject({ hdr: false, format: 'bgra8unorm' });
    expect(result.reason).toMatch(/configure failed/);
    expect(configure).toHaveBeenLastCalledWith({
      device,
      format: 'bgra8unorm',
      alphaMode: 'premultiplied',
    });
  });
});
