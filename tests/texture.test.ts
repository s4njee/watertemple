import { afterEach, describe, expect, it, vi } from 'vitest';
import {
  createTextureFromData,
  createTextureFromSource,
  loadTexture,
  updateTextureFromSource,
} from '../src/gpu/texture.ts';
import { expectValid, getTestDevice, readTexture } from './gpu.ts';

afterEach(() => {
  vi.restoreAllMocks();
  vi.unstubAllGlobals();
});

const solid = (width: number, height: number, rgba: number[]) =>
  new Uint8Array(Array.from({ length: width * height }, () => rgba).flat());

async function readLevel(device: GPUDevice, texture: GPUTexture, level: number) {
  const width = Math.max(1, texture.width >> level);
  const height = Math.max(1, texture.height >> level);
  const bytesPerRow = 256;
  const buffer = device.createBuffer({
    size: bytesPerRow * height,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });
  const encoder = device.createCommandEncoder();
  encoder.copyTextureToBuffer(
    { texture, mipLevel: level },
    { buffer, bytesPerRow },
    { width, height },
  );
  device.queue.submit([encoder.finish()]);
  await buffer.mapAsync(GPUMapMode.READ);
  const row = Array.from(new Uint8Array(buffer.getMappedRange()).slice(0, width * 4));
  buffer.unmap();
  buffer.destroy();
  return row;
}

describe('createTextureFromData', () => {
  it('uploads rgba8 pixels', async () => {
    const device = await getTestDevice();
    const data = new Uint8Array([
      255, 0, 0, 255, 0, 255, 0, 255, 0, 0, 255, 255, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110,
      120,
    ]);
    const texture = await expectValid(device, () =>
      createTextureFromData(device, {
        width: 3,
        height: 2,
        data,
        label: 'pattern',
        extraUsage: GPUTextureUsage.COPY_SRC,
      }),
    );
    expect(texture.label).toBe('pattern');
    expect(texture.format).toBe('rgba8unorm');
    expect([texture.width, texture.height, texture.mipLevelCount]).toEqual([3, 2, 1]);
    expect(texture.usage & GPUTextureUsage.TEXTURE_BINDING).toBeTruthy();
    expect([...(await readTexture(device, texture))]).toEqual([...data]);
  });

  it('honours the row stride of narrow and wide formats', async () => {
    const device = await getTestDevice();
    const extraUsage = GPUTextureUsage.COPY_SRC;

    const r8 = createTextureFromData(device, {
      width: 3,
      height: 2,
      format: 'r8unorm',
      data: new Uint8Array([1, 2, 3, 4, 5, 6]),
      extraUsage,
    });
    expect([...(await readTexture(device, r8, 1))]).toEqual([1, 2, 3, 4, 5, 6]);

    const half = createTextureFromData(device, {
      width: 2,
      height: 2,
      format: 'rgba16float',
      data: new Uint16Array(16).fill(0x3c00),
      extraUsage,
    });
    expect(new Uint16Array((await readTexture(device, half, 8)).buffer)[5]).toBe(0x3c00);

    const float = createTextureFromData(device, {
      width: 2,
      height: 1,
      format: 'r32float',
      data: new Float32Array([0.25, -3]),
      extraUsage,
    });
    expect([...new Float32Array((await readTexture(device, float, 4)).buffer)]).toEqual([0.25, -3]);
  });

  it('reads only the view it is given, not the whole underlying buffer', async () => {
    const device = await getTestDevice();
    const backing = new Uint8Array(24).fill(9);
    backing.set([1, 2, 3, 4], 8);
    const texture = createTextureFromData(device, {
      width: 1,
      height: 1,
      data: backing.subarray(8, 12),
      extraUsage: GPUTextureUsage.COPY_SRC,
    });
    expect([...(await readTexture(device, texture))]).toEqual([1, 2, 3, 4]);
  });

  it('builds a mip chain on request', async () => {
    const device = await getTestDevice();
    const texture = await expectValid(device, () =>
      createTextureFromData(device, {
        width: 8,
        height: 4,
        data: solid(8, 4, [40, 80, 120, 255]),
        mipmaps: true,
        extraUsage: GPUTextureUsage.COPY_SRC,
      }),
    );
    expect(texture.mipLevelCount).toBe(4);
    expect(texture.usage & GPUTextureUsage.RENDER_ATTACHMENT).toBeTruthy();
    const last = await readLevel(device, texture, 3);
    last.forEach((byte, i) =>
      expect(Math.abs(byte - [40, 80, 120, 255][i]!)).toBeLessThanOrEqual(1),
    );
  });

  it('keeps a 1x1 texture at one level even with mipmaps requested', async () => {
    const device = await getTestDevice();
    const texture = await expectValid(device, () =>
      createTextureFromData(device, {
        width: 1,
        height: 1,
        data: solid(1, 1, [1, 2, 3, 4]),
        mipmaps: true,
      }),
    );
    expect(texture.mipLevelCount).toBe(1);
  });

  it('rejects bad input before allocating anything', async () => {
    const device = await getTestDevice();
    const createTexture = vi.spyOn(device, 'createTexture');
    const data = solid(2, 2, [0, 0, 0, 0]);

    expect(() => createTextureFromData(device, { width: 2, height: 3, data })).toThrow(
      /needs 24 bytes, got 16/,
    );
    expect(() => createTextureFromData(device, { width: 0, height: 2, data })).toThrow(RangeError);
    expect(() => createTextureFromData(device, { width: 1.5, height: 2, data })).toThrow(
      RangeError,
    );
    expect(() =>
      createTextureFromData(device, { width: 2, height: 2, data, format: 'bc1-rgba-unorm' }),
    ).toThrow(/not supported/);
    const tooWide = device.limits.maxTextureDimension2D + 1;
    expect(() => createTextureFromData(device, { width: tooWide, height: 1, data })).toThrow(
      /maxTextureDimension2D/,
    );
    expect(() =>
      createTextureFromData(device, {
        width: 2,
        height: 2,
        format: 'rgba32float',
        data: new Float32Array(16),
        mipmaps: true,
      }),
    ).toThrow(/float32-filterable/);
    expect(createTexture).not.toHaveBeenCalled();
  });
});

describe('createTextureFromSource', () => {
  function emulateCopy(device: GPUDevice, colour: () => number[] = () => [10, 20, 30, 255]) {
    return vi
      .spyOn(device.queue, 'copyExternalImageToTexture')
      .mockImplementation((_source, destination, size) => {
        const [width, height] = size as [number, number];
        device.queue.writeTexture(
          { texture: destination.texture },
          solid(width, height, colour()),
          { bytesPerRow: width * 4 },
          [width, height],
        );
      });
  }
  const fake = (properties: object) => properties as unknown as GPUCopyExternalImageSource;

  it('sizes the texture from each kind of source', async () => {
    const device = await getTestDevice();
    emulateCopy(device);
    const cases: [object, [number, number]][] = [
      [{ width: 4, height: 2 }, [4, 2]], // ImageBitmap, canvas, OffscreenCanvas, ImageData
      [{ videoWidth: 8, videoHeight: 6, width: 300, height: 150 }, [8, 6]], // HTMLVideoElement
      [{ naturalWidth: 6, naturalHeight: 3, width: 99, height: 99 }, [6, 3]], // HTMLImageElement
      [{ displayWidth: 5, displayHeight: 7 }, [5, 7]], // VideoFrame
    ];
    for (const [source, [width, height]] of cases) {
      const texture = await expectValid(device, () =>
        createTextureFromSource(device, fake(source)),
      );
      expect([texture.width, texture.height]).toEqual([width, height]);
    }
  });

  it('picks sRGB or linear storage and sets the usage copyExternalImageToTexture requires', async () => {
    const device = await getTestDevice();
    emulateCopy(device);
    const colour = createTextureFromSource(device, fake({ width: 2, height: 2 }), {
      label: 'photo',
    });
    expect(colour.format).toBe('rgba8unorm-srgb');
    expect(colour.label).toBe('photo');
    const required =
      GPUTextureUsage.TEXTURE_BINDING |
      GPUTextureUsage.COPY_DST |
      GPUTextureUsage.RENDER_ATTACHMENT;
    expect(colour.usage & required).toBe(required);

    const data = createTextureFromSource(device, fake({ width: 2, height: 2 }), { srgb: false });
    expect(data.format).toBe('rgba8unorm');
  });

  it('forwards flipY and premultipliedAlpha to the copy', async () => {
    const device = await getTestDevice();
    const copy = emulateCopy(device);
    const source = fake({ width: 3, height: 2 });

    const plain = createTextureFromSource(device, source);
    expect(copy).toHaveBeenLastCalledWith(
      { source, flipY: false },
      { texture: plain, premultipliedAlpha: false },
      [3, 2],
    );

    const flipped = createTextureFromSource(device, source, {
      flipY: true,
      premultipliedAlpha: true,
    });
    expect(copy).toHaveBeenLastCalledWith(
      { source, flipY: true },
      { texture: flipped, premultipliedAlpha: true },
      [3, 2],
    );
  });

  it('generates mipmaps after the copy', async () => {
    const device = await getTestDevice();
    emulateCopy(device);
    const texture = await expectValid(device, () =>
      createTextureFromSource(device, fake({ width: 8, height: 8 }), {
        srgb: false,
        mipmaps: true,
        extraUsage: GPUTextureUsage.COPY_SRC,
      }),
    );
    expect(texture.mipLevelCount).toBe(4);
    const top = await readLevel(device, texture, 3);
    top.forEach((byte, i) => expect(Math.abs(byte - [10, 20, 30, 255][i]!)).toBeLessThanOrEqual(1));
  });

  it('refuses sources with no pixels yet or too large for the device', async () => {
    const device = await getTestDevice();
    const copy = emulateCopy(device);
    expect(() =>
      createTextureFromSource(device, fake({ naturalWidth: 0, naturalHeight: 0 })),
    ).toThrow(/no pixels yet/);
    expect(() => createTextureFromSource(device, fake({ videoWidth: 0, videoHeight: 0 }))).toThrow(
      /no pixels yet/,
    );
    const tooWide = device.limits.maxTextureDimension2D + 1;
    expect(() => createTextureFromSource(device, fake({ width: tooWide, height: 1 }))).toThrow(
      /maxTextureDimension2D/,
    );
    expect(copy).not.toHaveBeenCalled();
  });

  it('updateTextureFromSource re-copies and refreshes the mips, and checks the size', async () => {
    const device = await getTestDevice();
    let colour = [10, 20, 30, 255];
    const copy = emulateCopy(device, () => colour);
    const source = fake({ videoWidth: 4, videoHeight: 4 });
    const texture = createTextureFromSource(device, source, {
      srgb: false,
      mipmaps: true,
      extraUsage: GPUTextureUsage.COPY_SRC,
    });

    colour = [200, 100, 50, 255];
    await expectValid(device, () =>
      updateTextureFromSource(device, texture, source, { flipY: true }),
    );
    expect(copy).toHaveBeenLastCalledWith(
      { source, flipY: true },
      { texture, premultipliedAlpha: false },
      [4, 4],
    );
    const top = await readLevel(device, texture, 2);
    top.forEach((byte, i) =>
      expect(Math.abs(byte - [200, 100, 50, 255][i]!)).toBeLessThanOrEqual(1),
    );

    expect(() =>
      updateTextureFromSource(device, texture, fake({ videoWidth: 8, videoHeight: 4 })),
    ).toThrow(RangeError);
  });
});

describe('loadTexture', () => {
  it('fetches, decodes a data texture without colour conversion or premultiplying, and uploads', async () => {
    const device = await getTestDevice();
    const bitmap = { width: 4, height: 2 };
    const fetchMock = vi.fn(async () => ({ ok: true, blob: async () => 'BLOB' }));
    const decode = vi.fn(async () => bitmap);
    vi.stubGlobal('fetch', fetchMock);
    vi.stubGlobal('createImageBitmap', decode);
    const copy = vi.spyOn(device.queue, 'copyExternalImageToTexture').mockImplementation(() => {});

    const texture = await loadTexture(device, '/textures/normal.png', { srgb: false, flipY: true });

    expect(fetchMock).toHaveBeenCalledWith('/textures/normal.png');
    expect(decode).toHaveBeenCalledWith('BLOB', {
      colorSpaceConversion: 'none',
      premultiplyAlpha: 'none',
    });
    expect(texture.label).toBe('/textures/normal.png');
    expect(texture.format).toBe('rgba8unorm');
    expect([texture.width, texture.height]).toEqual([4, 2]);
    expect(copy).toHaveBeenCalledWith(
      { source: bitmap, flipY: true },
      { texture, premultipliedAlpha: false },
      [4, 2],
    );
  });

  it('lets the browser convert colour textures to sRGB, and leaves data textures alone', async () => {
    const device = await getTestDevice();
    const decode = vi.fn(async () => ({ width: 2, height: 2 }));
    vi.stubGlobal(
      'fetch',
      vi.fn(async () => ({ ok: true, blob: async () => 'BLOB' })),
    );
    vi.stubGlobal('createImageBitmap', decode);
    vi.spyOn(device.queue, 'copyExternalImageToTexture').mockImplementation(() => {});

    await loadTexture(device, '/photo.jpg');
    await loadTexture(device, '/mask.png', { srgb: false });

    expect(decode).toHaveBeenNthCalledWith(1, 'BLOB', {
      colorSpaceConversion: 'default',
      premultiplyAlpha: 'none',
    });
    expect(decode).toHaveBeenNthCalledWith(2, 'BLOB', {
      colorSpaceConversion: 'none',
      premultiplyAlpha: 'none',
    });
  });

  it('reports a failed request with the URL and status', async () => {
    const device = await getTestDevice();
    vi.stubGlobal(
      'fetch',
      vi.fn(async () => ({ ok: false, status: 404, statusText: 'Not Found' })),
    );
    await expect(loadTexture(device, '/missing.png')).rejects.toThrow(
      '/missing.png responded 404 Not Found',
    );
  });
});
