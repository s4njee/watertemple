import { describe, expect, it, vi } from 'vitest';
import { blit } from '../src/gpu/passes/blit.ts';
import { createBlur, MAX_BLUR_RADIUS } from '../src/gpu/passes/blur.ts';
import { perDevice } from '../src/gpu/passes/common.ts';
import { downsample } from '../src/gpu/passes/downsample.ts';
import { expectValid, getTestDevice, readTexture } from './gpu.ts';

const USAGE =
  GPUTextureUsage.RENDER_ATTACHMENT |
  GPUTextureUsage.TEXTURE_BINDING |
  GPUTextureUsage.COPY_SRC |
  GPUTextureUsage.COPY_DST;

function createTexture(device: GPUDevice, width: number, height: number, format: GPUTextureFormat) {
  return device.createTexture({ size: [width, height], format, usage: USAGE });
}

function writePixels(
  device: GPUDevice,
  texture: GPUTexture,
  data: Uint8Array | Uint16Array | Float32Array,
  bytesPerPixel: number,
) {
  device.queue.writeTexture(
    { texture },
    data,
    { bytesPerRow: texture.width * bytesPerPixel, rowsPerImage: texture.height },
    [texture.width, texture.height],
  );
}

function submit(device: GPUDevice, record: (encoder: GPUCommandEncoder) => void) {
  const encoder = device.createCommandEncoder();
  record(encoder);
  device.queue.submit([encoder.finish()]);
}

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

/** An rgba16float texture whose texel (x, y) is `colour(x, y)`. */
function createHalfTexture(
  device: GPUDevice,
  width: number,
  height: number,
  colour: (x: number, y: number) => number[],
) {
  const texture = createTexture(device, width, height, 'rgba16float');
  const bits = new Uint16Array(width * height * 4);
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      colour(x, y).forEach((value, c) => (bits[(y * width + x) * 4 + c] = halfBits(value)));
    }
  }
  writePixels(device, texture, bits, 8);
  return texture;
}

async function readHalf(device: GPUDevice, texture: GPUTexture): Promise<number[]> {
  const bits = new Uint16Array((await readTexture(device, texture, 8)).buffer);
  return Array.from(bits, halfValue);
}

const channel = (texels: number[], c: number) => texels.filter((_, i) => i % 4 === c);

describe('perDevice', () => {
  it('creates once per device and never shares across devices', async () => {
    const device = await getTestDevice();
    const other = await (await navigator.gpu.requestAdapter())!.requestDevice();
    const create = vi.fn(() => ({}));
    const get = perDevice(create);
    expect(get(device)).toBe(get(device));
    expect(get(other)).not.toBe(get(device));
    expect(create).toHaveBeenCalledTimes(2);
    other.destroy();
  });
});

describe('blit', () => {
  it('reproduces a pattern exactly with nearest filtering', async () => {
    const device = await getTestDevice();
    const pattern = new Uint8Array(4 * 4 * 4).map((_, i) => (i * 37 + 11) % 256);
    const source = createTexture(device, 4, 4, 'rgba8unorm');
    const target = createTexture(device, 4, 4, 'rgba8unorm');
    writePixels(device, source, pattern, 4);

    await expectValid(device, () =>
      submit(device, (encoder) =>
        blit(
          device,
          encoder,
          source.createView(),
          { view: target.createView(), format: 'rgba8unorm' },
          { filter: 'nearest' },
        ),
      ),
    );
    expect([...(await readTexture(device, target))]).toEqual([...pattern]);
  });

  it('scales up with nearest filtering by repeating texels', async () => {
    const device = await getTestDevice();
    const source = createTexture(device, 2, 2, 'rgba8unorm');
    const target = createTexture(device, 4, 4, 'rgba8unorm');
    const colours = [
      [255, 0, 0, 255],
      [0, 255, 0, 255],
      [0, 0, 255, 255],
      [255, 255, 0, 255],
    ];
    writePixels(device, source, new Uint8Array(colours.flat()), 4);

    await expectValid(device, () =>
      submit(device, (encoder) =>
        blit(
          device,
          encoder,
          source.createView(),
          { view: target.createView(), format: 'rgba8unorm' },
          { filter: 'nearest' },
        ),
      ),
    );
    const pixels = await readTexture(device, target);
    for (let y = 0; y < 4; y++) {
      for (let x = 0; x < 4; x++) {
        const expected = colours[(y >> 1) * 2 + (x >> 1)];
        expect([...pixels.subarray((y * 4 + x) * 4, (y * 4 + x) * 4 + 4)]).toEqual(expected);
      }
    }
  });

  it('copies unfilterable r32float exactly with nearest filtering', async () => {
    const device = await getTestDevice();
    const values = new Float32Array([0.125, -2, 1000.5, 3.25, 7, 8, 9, 10, 11]);
    const source = createTexture(device, 3, 3, 'r32float');
    const target = createTexture(device, 3, 3, 'r32float');
    writePixels(device, source, values, 4);

    await expectValid(device, () =>
      submit(device, (encoder) =>
        blit(
          device,
          encoder,
          source.createView(),
          { view: target.createView(), format: 'r32float' },
          { filter: 'nearest' },
        ),
      ),
    );
    expect([...new Float32Array((await readTexture(device, target, 4)).buffer)]).toEqual([
      ...values,
    ]);
  });

  it('keeps a constant colour constant with linear filtering, scaled or not', async () => {
    const device = await getTestDevice();
    const source = createTexture(device, 3, 3, 'rgba8unorm');
    writePixels(
      device,
      source,
      new Uint8Array(36).map((_, i) => [20, 120, 220, 255][i % 4]!),
      4,
    );
    for (const size of [3, 8]) {
      const target = createTexture(device, size, size, 'rgba8unorm');
      await expectValid(device, () =>
        submit(device, (encoder) =>
          blit(device, encoder, source.createView(), {
            view: target.createView(),
            format: 'rgba8unorm',
          }),
        ),
      );
      const pixels = await readTexture(device, target);
      pixels.forEach((byte, i) =>
        expect(Math.abs(byte - [20, 120, 220, 255][i % 4]!)).toBeLessThanOrEqual(1),
      );
    }
  });

  it('interpolates when scaling up with linear filtering', async () => {
    const device = await getTestDevice();
    const source = createTexture(device, 2, 1, 'rgba8unorm');
    const target = createTexture(device, 8, 1, 'rgba8unorm');
    writePixels(device, source, new Uint8Array([0, 0, 0, 255, 255, 255, 255, 255]), 4);
    submit(device, (encoder) =>
      blit(
        device,
        encoder,
        source.createView(),
        { view: target.createView(), format: 'rgba8unorm' },
        { filter: 'linear' },
      ),
    );
    const pixels = await readTexture(device, target);
    const red = Array.from({ length: 8 }, (_, x) => pixels[x * 4] ?? -1);
    expect(red[0]).toBe(0);
    expect(red[7]).toBe(255);
    for (let x = 1; x < 8; x++) expect(red[x]!).toBeGreaterThanOrEqual(red[x - 1]!);
    expect(red.slice(2, 6).every((value) => value > 0 && value < 255)).toBe(true);
  });

  it('writes sRGB targets in their encoded form', async () => {
    const device = await getTestDevice();
    const source = createTexture(device, 1, 1, 'rgba8unorm');
    const target = createTexture(device, 1, 1, 'rgba8unorm-srgb');
    writePixels(device, source, new Uint8Array([128, 128, 128, 255]), 4);
    await expectValid(device, () =>
      submit(device, (encoder) =>
        blit(
          device,
          encoder,
          source.createView(),
          { view: target.createView(), format: 'rgba8unorm-srgb' },
          { filter: 'nearest' },
        ),
      ),
    );
    // 128/255 is about 0.5 in linear light, which encodes to about 188.
    expect(Math.abs((await readTexture(device, target))[0]! - 188)).toBeLessThanOrEqual(2);
  });

  it('caches pipelines per device and format, and bind groups per source view', async () => {
    const device = await getTestDevice();
    const createPipeline = vi.spyOn(device, 'createRenderPipeline');
    const createGroup = vi.spyOn(device, 'createBindGroup');
    const source = createTexture(device, 2, 2, 'rgba16float');
    const view = source.createView();
    const target = {
      view: createTexture(device, 2, 2, 'rgba16float').createView(),
      format: 'rgba16float',
    } as const;

    await expectValid(device, () => {
      for (let i = 0; i < 3; i++) submit(device, (encoder) => blit(device, encoder, view, target));
    });
    expect(createPipeline).toHaveBeenCalledTimes(1);
    expect(createGroup).toHaveBeenCalledTimes(1);

    submit(device, (encoder) => blit(device, encoder, view, target, { filter: 'nearest' }));
    expect(createPipeline).toHaveBeenCalledTimes(2);

    const other = await (await navigator.gpu.requestAdapter())!.requestDevice();
    const otherSource = createTexture(other, 2, 2, 'rgba16float');
    const otherTarget = createTexture(other, 2, 2, 'rgba16float');
    const createOther = vi.spyOn(other, 'createRenderPipeline');
    await expectValid(other, () =>
      submit(other, (encoder) =>
        blit(other, encoder, otherSource.createView(), {
          view: otherTarget.createView(),
          format: 'rgba16float',
        }),
      ),
    );
    expect(createOther).toHaveBeenCalledTimes(1);
    createPipeline.mockRestore();
    createGroup.mockRestore();
    other.destroy();
  });
});

describe('downsample', () => {
  it('averages a 2x2 checker into one texel', async () => {
    const device = await getTestDevice();
    const source = createHalfTexture(device, 2, 2, (x, y) =>
      (x + y) % 2 === 0 ? [1, 1, 1, 1] : [0, 0, 0, 1],
    );
    const target = createTexture(device, 1, 1, 'rgba16float');
    await expectValid(device, () =>
      submit(device, (encoder) =>
        downsample(device, encoder, source.createView(), {
          view: target.createView(),
          format: 'rgba16float',
        }),
      ),
    );
    const [r, g, b, a] = await readHalf(device, target);
    for (const value of [r, g, b]) expect(value).toBeCloseTo(0.5, 2);
    expect(a).toBeCloseTo(1, 3);
  });

  it('averages an 8-bit checker too', async () => {
    const device = await getTestDevice();
    const source = createTexture(device, 2, 2, 'rgba8unorm');
    const target = createTexture(device, 1, 1, 'rgba8unorm');
    writePixels(
      device,
      source,
      new Uint8Array([200, 200, 200, 255, 0, 0, 0, 255, 0, 0, 0, 255, 200, 200, 200, 255]),
      4,
    );
    submit(device, (encoder) =>
      downsample(device, encoder, source.createView(), {
        view: target.createView(),
        format: 'rgba8unorm',
      }),
    );
    const pixels = await readTexture(device, target);
    expect(Math.abs(pixels[0]! - 100)).toBeLessThanOrEqual(1);
    expect(pixels[3]).toBe(255);
  });

  it('keeps a constant image constant, edges included', async () => {
    const device = await getTestDevice();
    const source = createHalfTexture(device, 8, 8, () => [0.75, 0.5, 0.25, 1]);
    const target = createTexture(device, 4, 4, 'rgba16float');
    await expectValid(device, () =>
      submit(device, (encoder) =>
        downsample(device, encoder, source.createView(), {
          view: target.createView(),
          format: 'rgba16float',
        }),
      ),
    );
    const texels = await readHalf(device, target);
    texels.forEach((value, i) => expect(value).toBeCloseTo([0.75, 0.5, 0.25, 1][i % 4]!, 2));
  });

  it('gives a moving impulse the same total weight at every phase, so it does not shimmer', async () => {
    const device = await getTestDevice();
    const totals: number[] = [];
    for (const [x, y] of [
      [8, 8],
      [9, 8],
      [8, 9],
      [9, 9],
      [10, 11],
    ] as const) {
      const source = createHalfTexture(device, 16, 16, (px, py) =>
        px === x && py === y ? [1, 0, 0, 1] : [0, 0, 0, 1],
      );
      const target = createTexture(device, 8, 8, 'rgba16float');
      submit(device, (encoder) =>
        downsample(device, encoder, source.createView(), {
          view: target.createView(),
          format: 'rgba16float',
        }),
      );
      totals.push(
        channel(await readHalf(device, target), 0).reduce((sum, value) => sum + value, 0),
      );
    }
    // Four source texels per target texel, so one texel's energy is 1/4 after averaging.
    totals.forEach((total) => expect(total).toBeCloseTo(0.25, 2));
  });

  it('handles non-2:1 sizes and rejects unfilterable targets before drawing', async () => {
    const device = await getTestDevice();
    const source = createTexture(device, 5, 5, 'rgba8unorm');
    const target = createTexture(device, 2, 2, 'rgba8unorm');
    await expectValid(device, () =>
      submit(device, (encoder) =>
        downsample(device, encoder, source.createView(), {
          view: target.createView(),
          format: 'rgba8unorm',
        }),
      ),
    );
    const wide = createTexture(device, 2, 2, 'rgba32float');
    await expectValid(device, () => {
      expect(() =>
        submit(device, (encoder) =>
          downsample(device, encoder, source.createView(), {
            view: wide.createView(),
            format: 'rgba32float',
          }),
        ),
      ).toThrow(/float32-filterable/);
    });
  });
});

describe('blur', () => {
  const SIZE = 33;
  const CENTER = 16;

  async function blurImpulse(
    device: GPUDevice,
    parameters: { radius: number; sigma?: number },
    size = SIZE,
  ): Promise<number[]> {
    const centre = (size - 1) / 2;
    const source = createHalfTexture(device, size, size, (x, y) =>
      x === centre && y === centre ? [1, 0, 0, 1] : [0, 0, 0, 1],
    );
    const target = createTexture(device, size, size, 'rgba16float');
    const blur = createBlur(device, {
      label: 'test blur',
      format: 'rgba16float',
      width: size,
      height: size,
    });
    await expectValid(device, () =>
      submit(device, (encoder) =>
        blur.encode(encoder, source.createView(), target.createView(), parameters),
      ),
    );
    blur.destroy();
    return channel(await readHalf(device, target), 0);
  }

  const gaussian = (radius: number, sigma: number) => {
    const weights = Array.from({ length: radius + 1 }, (_, i) =>
      Math.exp((-i * i) / (2 * sigma * sigma)),
    );
    const total = weights[0]! + 2 * weights.slice(1).reduce((sum, w) => sum + w, 0);
    return weights.map((w) => w / total);
  };

  it('leaves a constant colour unchanged, borders included', async () => {
    const device = await getTestDevice();
    const source = createHalfTexture(device, 16, 16, () => [0.5, 0.25, 0.75, 1]);
    const target = createTexture(device, 16, 16, 'rgba16float');
    const blur = createBlur(device, {
      label: 'constant',
      format: 'rgba16float',
      width: 16,
      height: 16,
    });
    await expectValid(device, () =>
      submit(device, (encoder) =>
        blur.encode(encoder, source.createView(), target.createView(), { radius: 8 }),
      ),
    );
    // Two half-float roundings and filtering in half precision cost a couple of ulps.
    (await readHalf(device, target)).forEach((value, i) =>
      expect(Math.abs(value - [0.5, 0.25, 0.75, 1][i % 4]!)).toBeLessThan(2e-3),
    );
    blur.destroy();
  });

  it('turns a single bright pixel into the analytic Gaussian, for even and odd radii', async () => {
    const device = await getTestDevice();
    for (const radius of [1, 2, 5, 6, 7]) {
      const sigma = (radius + 1) / 3;
      const w = gaussian(radius, sigma);
      const out = await blurImpulse(device, { radius });
      let worst = 0;
      for (let y = 0; y < SIZE; y++) {
        for (let x = 0; x < SIZE; x++) {
          const dx = Math.abs(x - CENTER);
          const dy = Math.abs(y - CENTER);
          const expected = dx <= radius && dy <= radius ? w[dx]! * w[dy]! : 0;
          worst = Math.max(worst, Math.abs(out[y * SIZE + x]! - expected));
        }
      }
      expect(worst).toBeLessThan(1e-3);
    }
  });

  it('is symmetric about the bright pixel and about the diagonal', async () => {
    const device = await getTestDevice();
    const out = await blurImpulse(device, { radius: 6, sigma: 2 });
    const at = (dx: number, dy: number) => out[(CENTER + dy) * SIZE + (CENTER + dx)]!;
    for (let dy = 0; dy <= 7; dy++) {
      for (let dx = 0; dx <= 7; dx++) {
        const value = at(dx, dy);
        for (const mirrored of [at(-dx, dy), at(dx, -dy), at(-dx, -dy), at(dy, dx)]) {
          expect(Math.abs(mirrored - value)).toBeLessThan(2e-4);
        }
      }
    }
    expect(at(0, 0)).toBeGreaterThan(at(1, 0));
    expect(at(1, 0)).toBeGreaterThan(at(3, 0));
  });

  it('conserves energy for any radius and sigma', async () => {
    const device = await getTestDevice();
    const settings = [
      { radius: 1 },
      { radius: 2, sigma: 0.5 },
      { radius: 3 },
      { radius: 4, sigma: 1 },
      { radius: 7 },
      { radius: 8, sigma: 20 },
      { radius: 15, sigma: 4 },
      { radius: 15, sigma: 0.2 },
    ];
    for (const parameters of settings) {
      const out = await blurImpulse(device, parameters, 65);
      const total = out.reduce((sum, value) => sum + value, 0);
      expect(Math.abs(total - 1), JSON.stringify(parameters)).toBeLessThan(0.01);
    }
  });

  it('copies when the radius is 0, and clamps absurd radii', async () => {
    const device = await getTestDevice();
    const none = await blurImpulse(device, { radius: 0 });
    expect(none[CENTER * SIZE + CENTER]).toBeCloseTo(1, 2);
    expect(none.reduce((sum, value) => sum + value, 0)).toBeCloseTo(1, 2);

    const out = await blurImpulse(device, { radius: 100_000 }, 2 * MAX_BLUR_RADIUS + 31);
    expect(Math.abs(out.reduce((sum, value) => sum + value, 0) - 1)).toBeLessThan(0.01);
  });

  it('can blur a texture in place', async () => {
    const device = await getTestDevice();
    const make = () =>
      createHalfTexture(device, SIZE, SIZE, (x, y) =>
        x === CENTER && y === CENTER ? [1, 0, 0, 1] : [0, 0, 0, 1],
      );
    const blur = createBlur(device, {
      label: 'in place',
      format: 'rgba16float',
      width: SIZE,
      height: SIZE,
    });
    const separate = createTexture(device, SIZE, SIZE, 'rgba16float');
    const input = make();
    const inPlace = make();
    await expectValid(device, () =>
      submit(device, (encoder) => {
        blur.encode(encoder, input.createView(), separate.createView(), { radius: 4 });
        const view = inPlace.createView();
        blur.encode(encoder, view, view, { radius: 4 });
      }),
    );
    expect(await readHalf(device, inPlace)).toEqual(await readHalf(device, separate));
    blur.destroy();
  });

  it('applies new parameters on each submitted encode', async () => {
    const device = await getTestDevice();
    const source = createHalfTexture(device, SIZE, SIZE, (x, y) =>
      x === CENTER && y === CENTER ? [1, 0, 0, 1] : [0, 0, 0, 1],
    );
    const blur = createBlur(device, {
      label: 'reused',
      format: 'rgba16float',
      width: SIZE,
      height: SIZE,
    });
    const peaks: number[] = [];
    for (const radius of [2, 8]) {
      const target = createTexture(device, SIZE, SIZE, 'rgba16float');
      submit(device, (encoder) =>
        blur.encode(encoder, source.createView(), target.createView(), { radius }),
      );
      peaks.push(channel(await readHalf(device, target), 0)[CENTER * SIZE + CENTER]!);
    }
    expect(peaks[0]!).toBeGreaterThan(peaks[1]! * 2);
    blur.destroy();
  });

  it('resizes its intermediate and destroys it', async () => {
    const device = await getTestDevice();
    const createdTextures = vi.spyOn(device, 'createTexture');
    const blur = createBlur(device, {
      label: 'resizable',
      format: 'rgba16float',
      width: 8,
      height: 8,
    });
    const intermediate = createdTextures.mock.results[0]?.value as GPUTexture;
    expect(intermediate.label).toBe('resizable intermediate');
    expect(blur.resize(8, 8)).toBe(false);
    const destroyOld = vi.spyOn(intermediate, 'destroy');
    expect(blur.resize(16, 16)).toBe(true);
    expect(destroyOld).toHaveBeenCalledOnce();
    const resized = createdTextures.mock.results.at(-1)?.value as GPUTexture;
    expect(resized.width).toBe(16);

    const source = createHalfTexture(device, 16, 16, () => [0.5, 0.5, 0.5, 1]);
    const target = createTexture(device, 16, 16, 'rgba16float');
    await expectValid(device, () =>
      submit(device, (encoder) =>
        blur.encode(encoder, source.createView(), target.createView(), { radius: 3 }),
      ),
    );
    (await readHalf(device, target)).forEach((value, i) =>
      expect(Math.abs(value - (i % 4 === 3 ? 1 : 0.5))).toBeLessThan(2e-3),
    );

    const destroyCurrent = vi.spyOn(resized, 'destroy');
    blur.destroy();
    expect(destroyCurrent).toHaveBeenCalledOnce();
    createdTextures.mockRestore();
  });

  it('rejects formats it cannot filter, and shares one pipeline per device and format', async () => {
    const device = await getTestDevice();
    expect(() =>
      createBlur(device, { label: 'bad', format: 'rgba32float', width: 4, height: 4 }),
    ).toThrow(/float32-filterable/);

    const createPipeline = vi.spyOn(device, 'createRenderPipeline');
    const first = createBlur(device, { label: 'a', format: 'rg16float', width: 4, height: 4 });
    const second = createBlur(device, { label: 'b', format: 'rg16float', width: 8, height: 8 });
    expect(createPipeline).toHaveBeenCalledTimes(1);

    const other = await (await navigator.gpu.requestAdapter())!.requestDevice();
    const createOther = vi.spyOn(other, 'createRenderPipeline');
    const third = createBlur(other, { label: 'c', format: 'rg16float', width: 4, height: 4 });
    expect(createOther).toHaveBeenCalledTimes(1);
    for (const blur of [first, second, third]) blur.destroy();
    createPipeline.mockRestore();
    other.destroy();
  });
});
