import { describe, expect, it, vi } from 'vitest';
import { generateMipmaps, mipLevelCount } from '../src/gpu/mipmap.ts';
import { expectValid, getTestDevice } from './gpu.ts';

const USAGE =
  GPUTextureUsage.TEXTURE_BINDING |
  GPUTextureUsage.RENDER_ATTACHMENT |
  GPUTextureUsage.COPY_DST |
  GPUTextureUsage.COPY_SRC;

function createTexture(
  device: GPUDevice,
  width: number,
  height: number,
  format: GPUTextureFormat = 'rgba8unorm',
  layers = 1,
  usage = USAGE,
) {
  return device.createTexture({
    size: [width, height, layers],
    format,
    mipLevelCount: mipLevelCount(width, height),
    usage,
  });
}

function upload(
  device: GPUDevice,
  texture: GPUTexture,
  data: Uint8Array | Uint16Array,
  bytesPerPixel = 4,
  layer = 0,
) {
  device.queue.writeTexture(
    { texture, origin: [0, 0, layer] },
    data,
    { bytesPerRow: texture.width * bytesPerPixel, rowsPerImage: texture.height },
    [texture.width, texture.height],
  );
}

async function readLevel(
  device: GPUDevice,
  texture: GPUTexture,
  level: number,
  layer = 0,
  bytesPerPixel = 4,
): Promise<Uint8Array> {
  const width = Math.max(1, texture.width >> level);
  const height = Math.max(1, texture.height >> level);
  const rowBytes = width * bytesPerPixel;
  const bytesPerRow = Math.ceil(rowBytes / 256) * 256;
  const buffer = device.createBuffer({
    size: bytesPerRow * height,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });
  const encoder = device.createCommandEncoder();
  encoder.copyTextureToBuffer(
    { texture, mipLevel: level, origin: [0, 0, layer] },
    { buffer, bytesPerRow },
    { width, height },
  );
  device.queue.submit([encoder.finish()]);
  await buffer.mapAsync(GPUMapMode.READ);
  const padded = new Uint8Array(buffer.getMappedRange());
  const pixels = new Uint8Array(rowBytes * height);
  for (let y = 0; y < height; y++) {
    pixels.set(padded.subarray(y * bytesPerRow, y * bytesPerRow + rowBytes), y * rowBytes);
  }
  buffer.unmap();
  buffer.destroy();
  return pixels;
}

/** Fails unless every byte is within `tolerance` of the expected one. */
function expectBytes(actual: Uint8Array, expected: number[], tolerance = 1) {
  expect(actual.length).toBe(expected.length);
  actual.forEach((byte, i) =>
    expect(Math.abs(byte - (expected[i] ?? NaN))).toBeLessThanOrEqual(tolerance),
  );
}

const solid = (width: number, height: number, rgba: number[]) =>
  new Uint8Array(Array.from({ length: width * height }, () => rgba).flat());

describe('mipLevelCount', () => {
  it('counts floor(log2(max(width, height))) + 1', () => {
    expect(mipLevelCount(1, 1)).toBe(1);
    expect(mipLevelCount(2, 2)).toBe(2);
    expect(mipLevelCount(3, 3)).toBe(2);
    expect(mipLevelCount(4, 1)).toBe(3);
    expect(mipLevelCount(5, 3)).toBe(3);
    expect(mipLevelCount(255, 255)).toBe(8);
    expect(mipLevelCount(256, 256)).toBe(9);
    expect(mipLevelCount(1, 512)).toBe(10);
    expect(mipLevelCount(1920, 1080)).toBe(11);
    expect(mipLevelCount(8192, 8192)).toBe(14);
  });
});

describe('generateMipmaps', () => {
  it('averages 2x2 blocks into each level', async () => {
    const device = await getTestDevice();
    const blocks = [
      [0, 40, 80, 255],
      [100, 140, 180, 255],
      [200, 240, 20, 255],
      [40, 80, 120, 255],
    ];
    const level0 = new Uint8Array(
      Array.from({ length: 16 }, (_, i) => blocks[(i >> 3) * 2 + ((i & 3) >> 1)] ?? []).flat(),
    );
    const texture = createTexture(device, 4, 4);
    upload(device, texture, level0);

    await expectValid(device, () => generateMipmaps(device, texture));

    expect([...(await readLevel(device, texture, 0))]).toEqual([...level0]);
    expectBytes(await readLevel(device, texture, 1), blocks.flat());
    expectBytes(await readLevel(device, texture, 2), [85, 125, 100, 255]);
  });

  it('filters sRGB textures in linear space', async () => {
    const device = await getTestDevice();
    const checker = new Uint8Array([
      0, 0, 0, 255, 255, 255, 255, 255, 255, 255, 255, 255, 0, 0, 0, 255,
    ]);

    const srgb = createTexture(device, 2, 2, 'rgba8unorm-srgb');
    upload(device, srgb, checker);
    await expectValid(device, () => generateMipmaps(device, srgb));
    // Half of linear white, re-encoded: 0.5 -> 0.735 -> 188.
    expectBytes(await readLevel(device, srgb, 1), [188, 188, 188, 255], 2);

    const linear = createTexture(device, 2, 2, 'rgba8unorm');
    upload(device, linear, checker);
    generateMipmaps(device, linear);
    expectBytes(await readLevel(device, linear, 1), [128, 128, 128, 255], 1);
  });

  it('handles non-power-of-two and non-square sizes', async () => {
    const device = await getTestDevice();
    const texture = createTexture(device, 5, 3);
    expect(texture.mipLevelCount).toBe(3);
    upload(device, texture, solid(5, 3, [30, 90, 150, 255]));
    await expectValid(device, () => generateMipmaps(device, texture));
    for (let level = 1; level < 3; level++) {
      const pixels = await readLevel(device, texture, level);
      expectBytes(
        pixels,
        Array.from({ length: pixels.length / 4 }, () => [30, 90, 150, 255]).flat(),
      );
    }
  });

  it('works on a 1xN strip', async () => {
    const device = await getTestDevice();
    const texture = createTexture(device, 1, 8);
    upload(device, texture, solid(1, 8, [200, 100, 50, 255]));
    await expectValid(device, () => generateMipmaps(device, texture));
    expectBytes(await readLevel(device, texture, 3), [200, 100, 50, 255]);
  });

  it('fills every array layer separately', async () => {
    const device = await getTestDevice();
    const texture = createTexture(device, 4, 4, 'rgba8unorm', 3);
    const colours = [
      [255, 0, 0, 255],
      [0, 255, 0, 255],
      [0, 0, 255, 255],
    ];
    colours.forEach((rgba, layer) => upload(device, texture, solid(4, 4, rgba), 4, layer));
    await expectValid(device, () => generateMipmaps(device, texture));
    for (const [layer, rgba] of colours.entries()) {
      expectBytes(await readLevel(device, texture, 2, layer), rgba);
    }
  });

  it('supports rgba16float', async () => {
    const device = await getTestDevice();
    const texture = createTexture(device, 2, 2, 'rgba16float');
    // Half-float bit patterns: 0 and 1.0 (0x3c00).
    const black = [0, 0, 0, 0x3c00];
    const white = [0x3c00, 0x3c00, 0x3c00, 0x3c00];
    upload(device, texture, new Uint16Array([...white, ...black, ...black, ...white]), 8);
    await expectValid(device, () => generateMipmaps(device, texture));
    const half = new Uint16Array((await readLevel(device, texture, 1, 0, 8)).buffer);
    // 0.5 is 0x3800; allow an ulp of filtering error.
    for (const channel of [0, 1, 2])
      expect(Math.abs((half[channel] ?? 0) - 0x3800)).toBeLessThanOrEqual(2);
    expect(half[3]).toBe(0x3c00);
  });

  it('records into a provided encoder instead of submitting', async () => {
    const device = await getTestDevice();
    const texture = createTexture(device, 2, 2);
    upload(device, texture, solid(2, 2, [200, 200, 200, 255]));
    const encoder = device.createCommandEncoder();
    generateMipmaps(device, texture, encoder);
    expect([...(await readLevel(device, texture, 1))]).toEqual([0, 0, 0, 0]);
    device.queue.submit([encoder.finish()]);
    expectBytes(await readLevel(device, texture, 1), [200, 200, 200, 255]);
  });

  it('is a no-op for a 1x1 texture and an error for a larger one without mips', async () => {
    const device = await getTestDevice();
    const single = createTexture(device, 1, 1);
    await expectValid(device, () => generateMipmaps(device, single));

    const flat = device.createTexture({ size: [4, 4], format: 'rgba8unorm', usage: USAGE });
    expect(() => generateMipmaps(device, flat)).toThrow(/single mip level/);
  });

  it('rejects textures it cannot render to or sample from before touching the GPU', async () => {
    const device = await getTestDevice();
    const noRender = createTexture(device, 4, 4, 'rgba8unorm', 1, GPUTextureUsage.TEXTURE_BINDING);
    await expectValid(device, () => {
      expect(() => generateMipmaps(device, noRender)).toThrow(/RENDER_ATTACHMENT/);
    });
  });

  it('throws a readable error for formats that cannot be filtered and rendered', async () => {
    const device = await getTestDevice();
    expect(device.features.has('float32-filterable')).toBe(false);
    const rejected: [GPUTextureFormat, RegExp][] = [
      ['rgba32float', /float32-filterable/],
      ['r32float', /float32-filterable/],
      ['rg32float', /float32-filterable/],
      ['rgba32uint', /cannot be filtered/],
      ['depth24plus', /cannot be filtered/],
    ];
    for (const [format, message] of rejected) {
      const texture = createTexture(device, 4, 4, format);
      await expectValid(device, () => {
        expect(() => generateMipmaps(device, texture)).toThrow(message);
      });
    }
  });

  it('accepts float32 formats once the device has float32-filterable', async () => {
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter?.features.has('float32-filterable')) return;
    const device = await adapter.requestDevice({ requiredFeatures: ['float32-filterable'] });
    const texture = createTexture(device, 2, 2, 'rgba32float');
    const values = new Float32Array([1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1]);
    device.queue.writeTexture({ texture }, values, { bytesPerRow: 32, rowsPerImage: 2 }, [2, 2]);
    await expectValid(device, () => generateMipmaps(device, texture));
    const mean = new Float32Array((await readLevel(device, texture, 1, 0, 16)).buffer);
    expect(Array.from(mean)).toEqual([0.5, 0.5, 0.5, 1]);
    device.destroy();
  });

  it('creates pipelines once per device and format, never sharing them across devices', async () => {
    const device = await getTestDevice();
    const create = vi.spyOn(device, 'createRenderPipeline');
    const run = async (target: GPUDevice) => {
      const texture = createTexture(target, 4, 4, 'bgra8unorm');
      upload(target, texture, solid(4, 4, [10, 20, 30, 255]));
      await expectValid(target, () => generateMipmaps(target, texture));
      expectBytes(await readLevel(target, texture, 2), [10, 20, 30, 255]);
    };

    await run(device);
    const afterFirst = create.mock.calls.length;
    expect(afterFirst).toBe(1);
    await run(device);
    expect(create.mock.calls.length).toBe(afterFirst);

    const other = await (await navigator.gpu.requestAdapter())!.requestDevice();
    const createOther = vi.spyOn(other, 'createRenderPipeline');
    await run(other);
    expect(createOther).toHaveBeenCalledTimes(1);
    other.destroy();
    create.mockRestore();
  });
});
