import { afterEach, describe, expect, it, vi } from 'vitest';
import {
  HAVE_CURRENT_DATA,
  classifyFile,
  createMediaFromFile,
  createMediaTexture,
  fitScale,
  fitUv,
  limitSize,
  mediaFromTexture,
  videoFrameDue,
  type VideoUploadState,
} from '../src/gpu/media.ts';
import { getTestDevice } from './gpu.ts';
import { gpuF32, lane, rowsOf, scalars } from './wgsl-eval.ts';

// The demo sweeps run hundreds of frames; leave headroom for a busy machine.
vi.setConfig({ testTimeout: 60_000 });

afterEach(() => {
  vi.restoreAllMocks();
});

// --- media.ts: pure logic ------------------------------------------------------------------------

describe('fitScale / fitUv (cover and contain)', () => {
  const aspects = [0.3, 0.6, 1, 1.5, 16 / 9, 2.4, 4];

  it('maps the centre to the centre and equal aspects to the identity', () => {
    for (const canvas of aspects) {
      for (const media of aspects) {
        for (const fit of ['cover', 'contain'] as const) {
          const uv = fitUv([0.5, 0.5], fitScale(canvas, media, fit));
          expect(uv[0]).toBeCloseTo(0.5, 12);
          expect(uv[1]).toBeCloseTo(0.5, 12);
        }
      }
    }
    expect(fitScale(1.7, 1.7, 'cover')).toEqual([1, 1]);
    expect(fitScale(1.7, 1.7, 'contain')).toEqual([1, 1]);
  });

  it('cover never leaves [0, 1] and fills the canvas on one axis', () => {
    for (const canvas of aspects) {
      for (const media of aspects) {
        const scale = fitScale(canvas, media, 'cover');
        for (const corner of [
          [0, 0],
          [1, 0],
          [0, 1],
          [1, 1],
        ] as const) {
          const uv = fitUv(corner, scale);
          expect(uv[0]).toBeGreaterThanOrEqual(-1e-12);
          expect(uv[0]).toBeLessThanOrEqual(1 + 1e-12);
          expect(uv[1]).toBeGreaterThanOrEqual(-1e-12);
          expect(uv[1]).toBeLessThanOrEqual(1 + 1e-12);
        }
        // One axis is shown whole.
        expect(Math.max(scale[0], scale[1])).toBeCloseTo(1, 12);
        // Aspect preserved: the visible part has the canvas's shape.
        expect((scale[0] * media) / scale[1]).toBeCloseTo(canvas, 9);
      }
    }
  });

  it('contain shows the whole media, letterboxed, with its aspect preserved', () => {
    // A wide canvas around a tall picture: bars at the sides, the full height used.
    const scale = fitScale(2, 0.5, 'contain');
    expect(scale[1]).toBeCloseTo(1, 12);
    expect(scale[0]).toBeCloseTo(4, 12);
    // The picture spans canvas x from 0.5 - 1/8 to 0.5 + 1/8.
    expect(fitUv([0.375, 0.5], scale)[0]).toBeCloseTo(0, 12);
    expect(fitUv([0.625, 0.5], scale)[0]).toBeCloseTo(1, 12);
    expect(fitUv([0.1, 0.5], scale)[0]).toBeLessThan(0);
    // A tall canvas around a wide picture: bars above and below.
    const tall = fitScale(0.5, 2, 'contain');
    expect(tall[0]).toBeCloseTo(1, 12);
    expect(tall[1]).toBeCloseTo(4, 12);
    for (const canvas of aspects) {
      for (const media of aspects) {
        const s = fitScale(canvas, media, 'contain');
        expect(Math.min(s[0], s[1])).toBeCloseTo(1, 12);
        expect((s[0] * media) / s[1]).toBeCloseTo(canvas, 9);
      }
    }
  });

  it('survives degenerate aspect ratios', () => {
    for (const bad of [0, -1, Number.MIN_VALUE]) {
      const scale = fitScale(bad, 1.5, 'cover');
      expect(Number.isFinite(scale[0]) && Number.isFinite(scale[1])).toBe(true);
    }
  });
});

describe('limitSize, classifyFile', () => {
  it('shrinks the long side to the limit, keeps the aspect, never upscales', () => {
    expect(limitSize(8000, 4000, 4000)).toEqual({ width: 4000, height: 2000, scaled: true });
    expect(limitSize(3000, 6000, 4000)).toEqual({ width: 2000, height: 4000, scaled: true });
    expect(limitSize(640, 360, 4096)).toEqual({ width: 640, height: 360, scaled: false });
    expect(limitSize(10000, 1, 100).height).toBe(1);
  });

  it('tells images from videos by MIME type, then by extension', () => {
    expect(classifyFile({ name: 'a.bin', type: 'image/webp' })).toBe('image');
    expect(classifyFile({ name: 'a.bin', type: 'video/mp4' })).toBe('video');
    expect(classifyFile({ name: 'Photo.JPG', type: '' })).toBe('image');
    expect(classifyFile({ name: 'clip.mov' })).toBe('video');
    expect(classifyFile({ name: 'notes.txt', type: 'text/plain' })).toBeUndefined();
    expect(classifyFile({})).toBeUndefined();
  });
});

describe('videoFrameDue', () => {
  const base: VideoUploadState = {
    readyState: 4,
    width: 640,
    height: 360,
    currentTime: 1,
    lastUploadedTime: 1,
    frameCallbackSupported: true,
    frameSignalled: false,
  };

  it('waits for a decoded frame and a known size', () => {
    expect(
      videoFrameDue({ ...base, readyState: HAVE_CURRENT_DATA - 1, lastUploadedTime: undefined }),
    ).toBe(false);
    expect(videoFrameDue({ ...base, width: 0, lastUploadedTime: undefined })).toBe(false);
    expect(videoFrameDue({ ...base, height: 0, lastUploadedTime: undefined })).toBe(false);
  });

  it('uploads the first frame immediately', () => {
    expect(videoFrameDue({ ...base, lastUploadedTime: undefined })).toBe(true);
    expect(
      videoFrameDue({ ...base, lastUploadedTime: undefined, frameCallbackSupported: false }),
    ).toBe(true);
  });

  it('with requestVideoFrameCallback uploads only when a new frame was signalled', () => {
    expect(videoFrameDue(base)).toBe(false);
    // currentTime moves between frames on a high refresh display: that alone is not a new frame.
    expect(videoFrameDue({ ...base, currentTime: 1.004 })).toBe(false);
    expect(videoFrameDue({ ...base, frameSignalled: true })).toBe(true);
  });

  it('without it, compares the playback position', () => {
    const plain = { ...base, frameCallbackSupported: false };
    expect(videoFrameDue(plain)).toBe(false);
    expect(videoFrameDue({ ...plain, currentTime: 1.04 })).toBe(true);
  });
});

// --- media.ts: textures, with the browser-only copy stubbed ------------------------------------------

describe('createMediaTexture and the no-DOM guards', () => {
  it('rejects a source with no pixels, and files that are not media', async () => {
    const device = await getTestDevice();
    await expect(
      createMediaTexture(device, { width: 0, height: 0 } as unknown as ImageBitmap),
    ).rejects.toThrow(/no pixels/);
    await expect(
      createMediaFromFile(device, new File(['x'], 'notes.txt', { type: 'text/plain' })),
    ).rejects.toThrow(/not an image or a video/);
  });

  it('needs a browser to decode an image file or play a video file', async () => {
    const device = await getTestDevice();
    vi.stubGlobal('createImageBitmap', undefined);
    await expect(
      createMediaFromFile(device, new File(['x'], 'a.png', { type: 'image/png' })),
    ).rejects.toThrow(/needs a browser/);
    vi.unstubAllGlobals();
    await expect(
      createMediaFromFile(device, new File(['x'], 'a.mp4', { type: 'video/mp4' })),
    ).rejects.toThrow(/needs a browser/);
  });

  it('uploads a still as rgba8unorm-srgb with a mip chain and exposes its shape', async () => {
    const device = await getTestDevice();
    const copy = vi.spyOn(device.queue, 'copyExternalImageToTexture').mockImplementation(() => {});
    const media = await createMediaTexture(device, { width: 300, height: 200 } as ImageBitmap, {
      label: 'test still',
    });
    expect(copy).toHaveBeenCalledTimes(1);
    expect(media.kind).toBe('still');
    expect(media.texture.format).toBe('rgba8unorm-srgb');
    expect([media.width, media.height]).toEqual([300, 200]);
    expect(media.aspect).toBeCloseTo(1.5, 12);
    expect(media.mipLevelCount).toBe(9);
    expect(media.update(device.createCommandEncoder())).toBe(false);
    media.destroy();
    media.destroy(); // idempotent
  });

  it('downscales stills that are larger than the limit while decoding', async () => {
    const device = await getTestDevice();
    vi.spyOn(device.queue, 'copyExternalImageToTexture').mockImplementation(() => {});
    const small = { width: 200, height: 100, close: vi.fn() };
    const decode = vi.fn(async () => small);
    vi.stubGlobal('createImageBitmap', decode);
    const media = await createMediaTexture(device, { width: 800, height: 400 } as ImageBitmap, {
      maxSize: 200,
    });
    expect(decode).toHaveBeenCalledWith(
      expect.anything(),
      expect.objectContaining({
        resizeWidth: 200,
        resizeHeight: 100,
        colorSpaceConversion: 'default',
      }),
    );
    expect(small.close).toHaveBeenCalled();
    expect([media.width, media.height]).toEqual([200, 100]);
    media.destroy();
  });

  it('wraps a texture you rendered yourself', async () => {
    const device = await getTestDevice();
    const texture = device.createTexture({
      size: [8, 4],
      format: 'rgba8unorm-srgb',
      usage: GPUTextureUsage.TEXTURE_BINDING,
    });
    const onDestroy = vi.fn();
    const media = mediaFromTexture(texture, onDestroy);
    expect(media.aspect).toBe(2);
    media.destroy();
    expect(onDestroy).toHaveBeenCalledTimes(1);
  });
});

describe('a video media texture', () => {
  function fakeVideo() {
    let callback: (() => void) | undefined;
    const video = {
      videoWidth: 64,
      videoHeight: 32,
      readyState: 4,
      currentTime: 0,
      requestVideoFrameCallback: vi.fn((fn: () => void) => {
        callback = fn;
        return 7;
      }),
      cancelVideoFrameCallback: vi.fn(),
    };
    return {
      video: video as unknown as HTMLVideoElement,
      mock: video,
      newFrame: () => callback?.(),
    };
  }

  it('uploads once per presented frame, not once per render', async () => {
    const device = await getTestDevice();
    const copy = vi.spyOn(device.queue, 'copyExternalImageToTexture').mockImplementation(() => {});
    const { video, newFrame } = fakeVideo();
    const media = await createMediaTexture(device, video);
    expect(media.kind).toBe('video');
    expect(copy).toHaveBeenCalledTimes(1); // the first frame, at creation
    const encoder = device.createCommandEncoder();
    for (let render = 0; render < 5; render++) expect(media.update(encoder)).toBe(false);
    expect(copy).toHaveBeenCalledTimes(1);

    newFrame();
    expect(media.update(encoder)).toBe(true);
    expect(media.update(encoder)).toBe(false); // the same frame is not copied twice
    expect(copy).toHaveBeenCalledTimes(2);
    // Passes recorded into the caller's encoder; a bad encoder state would throw on finish().
    device.queue.submit([encoder.finish()]);
    media.destroy();
  });

  it('reallocates when the video changes resolution, and says so through `version`', async () => {
    const device = await getTestDevice();
    vi.spyOn(device.queue, 'copyExternalImageToTexture').mockImplementation(() => {});
    const { video, mock, newFrame } = fakeVideo();
    const media = await createMediaTexture(device, video);
    const before = media.view;
    expect(media.version).toBe(0);
    mock.videoWidth = 128;
    mock.videoHeight = 64;
    newFrame();
    expect(media.update(device.createCommandEncoder())).toBe(true);
    expect(media.version).toBe(1);
    expect(media.view).not.toBe(before);
    expect([media.width, media.height]).toEqual([128, 64]);
    media.destroy();
  });

  it('cancels its frame callback and runs onDestroy once', async () => {
    const device = await getTestDevice();
    vi.spyOn(device.queue, 'copyExternalImageToTexture').mockImplementation(() => {});
    const { video, mock } = fakeVideo();
    const onDestroy = vi.fn();
    const media = await createMediaTexture(device, video, { onDestroy });
    media.destroy();
    media.destroy();
    expect(mock.cancelVideoFrameCallback).toHaveBeenCalledWith(7);
    expect(onDestroy).toHaveBeenCalledTimes(1);
    expect(media.update(device.createCommandEncoder())).toBe(false);
  });

  it('refuses a video that has no size yet', async () => {
    const device = await getTestDevice();
    const { video, mock } = fakeVideo();
    mock.videoWidth = 0;
    await expect(createMediaTexture(device, video)).rejects.toThrow(/whenVideoReady/);
  });
});

// --- media.wgsl on the GPU -----------------------------------------------------------------------------

describe('media.wgsl', () => {
  const include = '#include "media.wgsl"';

  it('agrees with the TypeScript fit and maps the centre to the centre', async () => {
    const rows: number[][] = [];
    const aspects = [0.4, 0.75, 1, 1.5, 16 / 9, 3];
    for (const canvas of aspects) {
      for (const media of aspects) rows.push([canvas, media]);
    }
    const out = await gpuF32(
      `${include}
       fn f(x: vec4f) -> vec4f {
         let s = media_fit_scale(x.x, x.y, false);
         let c = media_fit_scale(x.x, x.y, true);
         return vec4f(s.x, s.y, c.x, c.y);
       }`,
      rowsOf(rows),
    );
    rows.forEach(([canvas, media], i) => {
      const cover = fitScale(canvas!, media!, 'cover');
      const contain = fitScale(canvas!, media!, 'contain');
      expect(out[i * 4]).toBeCloseTo(cover[0], 5);
      expect(out[i * 4 + 1]).toBeCloseTo(cover[1], 5);
      expect(out[i * 4 + 2]).toBeCloseTo(contain[0], 5);
      expect(out[i * 4 + 3]).toBeCloseTo(contain[1], 5);
    });

    const centre = await scalars(include, [
      'media_fit_uv(vec2f(0.5), media_fit_scale(2.4, 0.7, false)).x',
      'media_fit_uv(vec2f(0.5), media_fit_scale(2.4, 0.7, false)).y',
      'media_fit_uv(vec2f(0.5), media_fit_scale(0.5, 2.0, true)).x',
      'media_fit_uv(vec2f(0.5), media_fit_scale(0.5, 2.0, true)).y',
    ]);
    for (const value of centre) expect(value).toBeCloseTo(0.5, 6);
  });

  it('cover never leaves [0, 1] anywhere on the canvas', async () => {
    const rows: number[][] = [];
    for (let a = 0; a < 7; a++) {
      for (let m = 0; m < 7; m++) {
        for (let i = 0; i <= 8; i++) {
          for (let j = 0; j <= 8; j++) {
            rows.push([0.3 + a * 0.55, 0.3 + m * 0.55, i / 8, j / 8]);
          }
        }
      }
    }
    // Pack canvas aspect, media aspect and one uv into two rows via the first two lanes and uv packed in z, w.
    const out = await gpuF32(
      `${include}
       fn f(x: vec4f) -> vec4f {
         let uv = media_fit_uv(x.zw, media_fit_scale(x.x, x.y, false));
         return vec4f(uv, media_inside(uv, 0.0), 0.0);
       }`,
      rowsOf(rows),
    );
    let worst = 0;
    for (let i = 0; i < rows.length; i++) {
      const u = out[i * 4] ?? Number.NaN;
      const v = out[i * 4 + 1] ?? Number.NaN;
      worst = Math.max(worst, -u, u - 1, -v, v - 1);
      // Exactly 0.5 only on the border itself (the canvas edge of the longer axis).
      expect(out[i * 4 + 2]!).toBeGreaterThanOrEqual(0.5);
    }
    expect(worst).toBeLessThan(1e-5);
  });

  it('contain letterboxes: bars outside the media, the media untouched inside', async () => {
    // A 2:1 canvas around a 1:2 picture (media_aspect 0.5): the picture covers canvas x in [0.375, 0.625].
    const probes = [0.05, 0.3, 0.38, 0.5, 0.62, 0.7, 0.95];
    const out = await gpuF32(
      `${include}
       fn f(x: vec4f) -> vec4f {
         let uv = media_fit_uv(vec2f(x.x, 0.5), media_fit_scale(2.0, 0.5, true));
         return vec4f(media_inside(uv, 0.0), uv.x, media_inside(vec2f(0.5, x.x), 0.0), 0.0);
       }`,
      rowsOf(probes.map((p) => [p])),
    );
    expect(lane(out, 0)).toEqual([0, 0, 1, 1, 1, 0, 0]);
    expect(lane(out, 1)[3]).toBeCloseTo(0.5, 6);
    // Vertically the picture fills the canvas: no bar above or below.
    expect(lane(out, 2).slice(0, 7)).toEqual([1, 1, 1, 1, 1, 1, 1]);
    // media_inside ramps over `aa` and is 0.5 exactly on the border.
    const ramp = await scalars(include, [
      'media_inside(vec2f(0.0, 0.5), 0.02)',
      'media_inside(vec2f(0.01, 0.5), 0.02)',
      'media_inside(vec2f(-0.01, 0.5), 0.02)',
      'media_inside(vec2f(0.5, 0.995), 0.01)',
    ]);
    expect(ramp[0]).toBeCloseTo(0.5, 6);
    expect(ramp[1]).toBeCloseTo(1, 6);
    expect(ramp[2]).toBeCloseTo(0, 6);
    expect(ramp[3]).toBeCloseTo(1, 6);
  });

  it('media_lod is zero at 1:1 or magnified, and counts halvings when minified', async () => {
    const [same, magnified, half, quarter, cover] = await scalars(include, [
      'media_lod(vec2f(640.0, 360.0), vec2f(1.0), vec2f(640.0, 360.0))',
      'media_lod(vec2f(320.0, 180.0), vec2f(1.0), vec2f(640.0, 360.0))',
      'media_lod(vec2f(1280.0, 720.0), vec2f(1.0), vec2f(640.0, 360.0))',
      'media_lod(vec2f(2560.0, 1440.0), vec2f(1.0), vec2f(640.0, 360.0))',
      // A 4:3 media covering a 16:9 canvas only shows 75% of its height.
      'media_lod(vec2f(1280.0, 960.0), vec2f(1.0, 0.75), vec2f(640.0, 360.0))',
    ]);
    expect(same).toBeCloseTo(0, 6);
    expect(magnified).toBe(0);
    expect(half).toBeCloseTo(1, 6);
    expect(quarter).toBeCloseTo(2, 6);
    expect(cover).toBeCloseTo(1, 6);
  });
});

// --- toon.wgsl on the GPU -----------------------------------------------------------------------------

describe('toon.wgsl', () => {
  const include = '#include "toon.wgsl"';

  it('toon_step is 0.5 on the edge, 0 and 1 half a pixel either side, and monotonic', async () => {
    const [on, below, above, far, hard] = await scalars(include, [
      'toon_step(0.3, 0.3, 0.1)',
      'toon_step(0.3, 0.25, 0.1)',
      'toon_step(0.3, 0.35, 0.1)',
      'toon_step(0.3, 5.0, 0.1)',
      'toon_step(0.3, 0.3001, 0.0)',
    ]);
    expect(on).toBeCloseTo(0.5, 6);
    expect(below).toBeCloseTo(0, 6);
    expect(above).toBeCloseTo(1, 6);
    expect(far).toBe(1);
    expect(hard).toBe(1);

    const xs = Array.from({ length: 400 }, (_, i) => -1 + (2 * i) / 399);
    const values = lane(
      await gpuF32(
        `${include}\nfn f(x: vec4f) -> vec4f { return vec4f(toon_step(0.1, x.x, 0.3)); }`,
        rowsOf(xs.map((x) => [x])),
      ),
      0,
    );
    for (let i = 1; i < values.length; i++)
      expect(values[i]!).toBeGreaterThanOrEqual(values[i - 1]!);
    for (const v of values) expect(v >= 0 && v <= 1).toBe(true);
  });

  it('toon_step integrates to the right coverage wherever the edge falls in a pixel', async () => {
    // Pixel centres are `aa` apart. Summing coverage * aa along a row measures the length of the
    // region beyond the edge, and it must not depend on where in a pixel the edge sits (that is
    // what makes a moving edge look steady instead of crawling).
    const aa = 0.05;
    const length = 2;
    const count = Math.round(length / aa);
    const offsets = [0, 0.1, 0.25, 0.5, 0.77, 0.999].map((f) => 0.6 + f * aa);
    for (const edge of offsets) {
      const centres = Array.from({ length: count }, (_, i) => (i + 0.5) * aa);
      const out = await gpuF32(
        `${include}\nfn f(x: vec4f) -> vec4f { return vec4f(toon_step(${edge}, x.x, ${aa})); }`,
        rowsOf(centres.map((c) => [c])),
      );
      const covered = lane(out, 0)
        .slice(0, count)
        .reduce((sum, v) => sum + v * aa, 0);
      expect(covered).toBeCloseTo(length - edge, 4);
    }
  });

  it('toon_band returns exactly N flat levels away from the edges, in order', async () => {
    for (const bands of [2, 3, 4, 5, 8]) {
      const aa = 0.004;
      const samples = 4000;
      const ts = Array.from({ length: samples }, (_, i) => (i + 0.5) / samples);
      const out = lane(
        await gpuF32(
          `${include}\nfn f(x: vec4f) -> vec4f { return vec4f(toon_band(x.x, ${bands}.0, ${aa})); }`,
          rowsOf(ts.map((t) => [t])),
        ),
        0,
      ).slice(0, samples);

      const levels = Array.from({ length: bands }, (_, k) => k / (bands - 1));
      const flat = new Set<number>();
      out.forEach((value, i) => {
        // Distance to the nearest band edge, in t units.
        const edgeDistance = Math.abs(ts[i]! * bands - Math.round(ts[i]! * bands)) / bands;
        if (edgeDistance > aa) {
          const nearest = levels.reduce((a, b) =>
            Math.abs(b - value) < Math.abs(a - value) ? b : a,
          );
          expect(Math.abs(nearest - value)).toBeLessThan(1e-5);
          flat.add(Math.round(nearest * 1000));
        }
      });
      expect(flat.size).toBe(bands);
      for (let i = 1; i < out.length; i++)
        expect(out[i]!).toBeGreaterThanOrEqual(out[i - 1]! - 1e-6);
      expect(out[0]).toBe(0);
      expect(out[out.length - 1]).toBe(1);
    }
  });

  it('toon_band edges are one pixel wide, and out-of-range input clamps', async () => {
    const bands = 4;
    const aa = 0.02;
    // Across the edge at t = 0.5 the level goes from 1/3 to 2/3 over exactly aa.
    const [before, middle, after, low, high, single] = await scalars(include, [
      `toon_band(${0.5 - aa / 2 - 1e-4}, ${bands}.0, ${aa})`,
      `toon_band(0.5, ${bands}.0, ${aa})`,
      `toon_band(${0.5 + aa / 2 + 1e-4}, ${bands}.0, ${aa})`,
      `toon_band(-3.0, ${bands}.0, ${aa})`,
      `toon_band(9.0, ${bands}.0, ${aa})`,
      `toon_band(0.7, 1.0, ${aa})`,
    ]);
    expect(before).toBeCloseTo(1 / 3, 4);
    expect(middle).toBeCloseTo(0.5, 4);
    expect(after).toBeCloseTo(2 / 3, 4);
    expect(low).toBe(0);
    expect(high).toBe(1);
    expect(single).toBe(0);
  });

  it('toon_posterize and toon_ramp', async () => {
    const [r, g, b, stop0, stop1, stop2, stop3, mid] = await scalars(include, [
      'toon_posterize(vec3f(0.3, 0.6, 0.9), 2.0, 0.0).x',
      'toon_posterize(vec3f(0.3, 0.6, 0.9), 2.0, 0.0).y',
      'toon_posterize(vec3f(0.3, 0.6, 0.9), 3.0, 0.0).z',
      'toon_ramp(0.0, vec3f(1.0), vec3f(2.0), vec3f(3.0), vec3f(4.0)).x',
      'toon_ramp(1.0 / 3.0, vec3f(1.0), vec3f(2.0), vec3f(3.0), vec3f(4.0)).x',
      'toon_ramp(2.0 / 3.0, vec3f(1.0), vec3f(2.0), vec3f(3.0), vec3f(4.0)).x',
      'toon_ramp(1.0, vec3f(1.0), vec3f(2.0), vec3f(3.0), vec3f(4.0)).x',
      'toon_ramp(0.5, vec3f(1.0), vec3f(2.0), vec3f(3.0), vec3f(4.0)).x',
    ]);
    expect([r, g, b]).toEqual([0, 1, 1]);
    expect(stop0).toBeCloseTo(1, 5);
    expect(stop1).toBeCloseTo(2, 5);
    expect(stop2).toBeCloseTo(3, 5);
    expect(stop3).toBeCloseTo(4, 5);
    expect(mid).toBeCloseTo(2.5, 5);
  });

  it('toon_line and toon_ring give coverage of the right size', async () => {
    const [centre, edgeIn, edgeOut, outside, zero, ringIn, ringOut, ringEdge, ringBad] =
      await scalars(include, [
        'toon_line(0.0, 0.2, 0.01)',
        'toon_line(0.1, 0.2, 0.01)',
        'toon_line(-0.1, 0.2, 0.01)',
        'toon_line(0.2, 0.2, 0.01)',
        'toon_line(0.0, 0.0, 0.01)',
        'toon_ring(0.05, 0.0, 0.1, 0.01)',
        'toon_ring(0.2, 0.0, 0.1, 0.01)',
        'toon_ring(0.1, 0.0, 0.1, 0.01)',
        'toon_ring(0.05, 0.1, 0.0, 0.01)',
      ]);
    expect(centre).toBe(1);
    expect(edgeIn).toBeCloseTo(0.5, 5);
    expect(edgeOut).toBeCloseTo(0.5, 5);
    expect(outside).toBe(0);
    expect(zero).toBe(0);
    expect(ringIn).toBe(1);
    expect(ringOut).toBe(0);
    expect(ringEdge).toBeCloseTo(0.5, 5);
    expect(ringBad).toBe(0);

    // A line `width` wide integrates to `width` across its profile, whatever aa is.
    const width = 0.3;
    const aa = 0.07;
    const step = 0.001;
    const ds = Array.from({ length: 1200 }, (_, i) => -0.6 + i * step);
    const profile = lane(
      await gpuF32(
        `${include}\nfn f(x: vec4f) -> vec4f { return vec4f(toon_line(x.x, ${width}, ${aa})); }`,
        rowsOf(ds.map((d) => [d])),
      ),
      0,
    ).slice(0, ds.length);
    expect(profile.reduce((s, v) => s + v * step, 0)).toBeCloseTo(width, 3);
  });

  it('toon_foam_cells is finite, non-negative, and zero on cell borders only', async () => {
    const random = (() => {
      let state = 12345;
      return () => (state = (Math.imul(state, 1664525) + 1013904223) >>> 0) / 4294967296;
    })();
    const rows = Array.from({ length: 4000 }, () => [
      random() * 40 - 20,
      random() * 40 - 20,
      random() * 20,
      0.85,
    ]);
    const out = lane(
      await gpuF32(
        `${include}\nfn f(x: vec4f) -> vec4f { return vec4f(toon_foam_cells(x.xy, x.z, x.w)); }`,
        rowsOf(rows),
      ),
      0,
    );
    const finite = out.every((v) => Number.isFinite(v) && v >= 0 && v < 2);
    expect(finite).toBe(true);
    // Most of the plane is away from borders: the mean is well above zero.
    expect(out.reduce((a, b) => a + b, 0) / out.length).toBeGreaterThan(0.15);
    // The lines are rare: a small share of random points lies within 0.02 of a border.
    expect(out.filter((v) => v < 0.02).length / out.length).toBeLessThan(0.1);
  });

  it('toon_cell_edge masks stay in [0, 1] and keep the requested pixel width', async () => {
    const rows: number[][] = [];
    for (let fi = 0; fi <= 40; fi++) {
      for (const grad of [0, 0.01, 0.3, 1, 2, 5]) {
        for (const width of [-1, 0, 0.5, 2, 6]) rows.push([fi * 0.05, grad, width]);
      }
    }
    const mask = lane(
      await gpuF32(
        `${include}\nfn f(x: vec4f) -> vec4f { return vec4f(toon_cell_edge(x.x, x.y, x.z)); }`,
        rowsOf(rows),
      ),
      0,
    );
    for (const v of mask) expect(v >= 0 && v <= 1).toBe(true);

    // Walking away from a border at gradient g, one pixel is a step of g in F. The covered length in
    // pixels is the width, however steep the gradient.
    for (const [grad, width] of [
      [1.5, 3],
      [0.4, 3],
      [2, 1.5],
    ] as const) {
      const pixels = Array.from({ length: 2400 }, (_, i) => i * 0.005);
      const out = lane(
        await gpuF32(
          `${include}\nfn f(x: vec4f) -> vec4f { return vec4f(toon_cell_edge(x.x * ${grad}, ${grad}, ${width})); }`,
          rowsOf(pixels.map((p) => [p])),
        ),
        0,
      ).slice(0, pixels.length);
      // One side of the border: half the line.
      expect(out.reduce((s, v) => s + v * 0.005, 0)).toBeCloseTo(width / 2, 2);
    }
  });

  it('toon_sparkle is in [0, 1], absent at density 0, present at density 1, and stays in its cell', async () => {
    const random = (() => {
      let state = 99;
      return () => (state = (Math.imul(state, 1664525) + 1013904223) >>> 0) / 4294967296;
    })();
    const rows = Array.from({ length: 6000 }, () => [
      random() * 30 - 15,
      random() * 30 - 15,
      random() * 40,
      random(),
    ]);
    const dense = lane(
      await gpuF32(
        `${include}\nfn f(x: vec4f) -> vec4f { return vec4f(toon_sparkle(x.xy, x.z, 1.0, 0.4, 0.01), toon_sparkle(x.xy, x.z, x.w, 0.4, 0.01), toon_sparkle(x.xy, x.z, 0.0, 0.4, 0.01), 0.0); }`,
        rowsOf(rows),
      ),
      0,
    );
    for (const v of dense) expect(v >= 0 && v <= 1 && Number.isFinite(v)).toBe(true);
    expect(Math.max(...dense)).toBe(1);
    expect(dense.filter((v) => v > 0).length).toBeGreaterThan(100);

    const all = await gpuF32(
      `${include}\nfn f(x: vec4f) -> vec4f { return vec4f(toon_sparkle(x.xy, x.z, 1.0, 0.4, 0.01), toon_sparkle(x.xy, x.z, x.w, 0.4, 0.01), toon_sparkle(x.xy, x.z, 0.0, 0.4, 0.01), 0.0); }`,
      rowsOf(rows),
    );
    expect(lane(all, 2).every((v) => v === 0)).toBe(true);
    // A random density never lights more than density 1 does.
    for (let i = 0; i < rows.length; i++)
      expect(all[i * 4 + 1]!).toBeLessThanOrEqual(all[i * 4]! + 1e-6);

    // Never leaves its cell: on the border of cells it is dark.
    const border = Array.from({ length: 800 }, (_, i) => [
      Math.floor(random() * 20),
      0.3 + (i % 5) * 0.1,
      random() * 40,
      0,
    ]);
    const edges = await gpuF32(
      `${include}\nfn f(x: vec4f) -> vec4f { return vec4f(toon_sparkle(vec2f(x.x, x.y + 0.0), x.z, 1.0, 0.45, 0.0), toon_sparkle(vec2f(x.y, x.x + 1.0 - 1e-4), x.z, 1.0, 0.45, 0.0), 0.0, 0.0); }`,
      rowsOf(border),
    );
    expect(lane(edges, 0).every((v) => v === 0)).toBe(true);
    expect(lane(edges, 1).every((v) => v === 0)).toBe(true);
  });
});

// --- Demo effects, headless -------------------------------------------------------------------------

function halfToFloat(half: number): number {
  const sign = half & 0x8000 ? -1 : 1;
  const exponent = (half >> 10) & 0x1f;
  const fraction = half & 0x3ff;
  if (exponent === 0) return sign * 2 ** -14 * (fraction / 1024);
  if (exponent === 31) return fraction ? Number.NaN : sign * Infinity;
  return sign * 2 ** (exponent - 15) * (1 + fraction / 1024);
}

async function readFloats(device: GPUDevice, texture: GPUTexture): Promise<Float32Array> {
  const { width, height } = texture;
  const bytesPerRow = Math.ceil((width * 8) / 256) * 256;
  const buffer = device.createBuffer({
    size: bytesPerRow * height,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });
  const encoder = device.createCommandEncoder();
  encoder.copyTextureToBuffer({ texture }, { buffer, bytesPerRow }, { width, height });
  device.queue.submit([encoder.finish()]);
  await buffer.mapAsync(GPUMapMode.READ);
  const raw = new Uint16Array(buffer.getMappedRange().slice(0));
  buffer.unmap();
  buffer.destroy();
  const out = new Float32Array(width * height * 4);
  for (let y = 0; y < height; y++) {
    for (let i = 0; i < width * 4; i++) {
      out[y * width * 4 + i] = halfToFloat(raw[(y * bytesPerRow) / 2 + i] ?? 0);
    }
  }
  return out;
}

describe('media and toon demos stay finite across their parameter ranges', () => {
  const WIDTH = 96;
  const HEIGHT = 64;
  const canvas = Object.assign(new EventTarget(), {
    width: WIDTH,
    height: HEIGHT,
    style: {},
    getBoundingClientRect: () => ({ left: 0, top: 0, width: WIDTH, height: HEIGHT }),
  }) as unknown as HTMLCanvasElement;

  it.each(['media-distortion', 'toon-water'])('%s', async (id) => {
    const device = await getTestDevice();
    const { loadEffect } = await import('../src/effects/registry.ts');
    const { defaultParams } = await import('../src/effects/types.ts');
    const { noProfiler } = await import('../src/gpu/profiler.ts');
    const effect = await loadEffect(id);
    expect(effect.presets ? Object.keys(effect.presets).length : 0).toBeGreaterThanOrEqual(2);

    // A float target: the sRGB encode and the 8-bit clamp would hide a NaN as black.
    const target = device.createTexture({
      size: [WIDTH, HEIGHT],
      format: 'rgba16float',
      usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC,
    });
    const params = defaultParams(effect.params) as Record<string, unknown>;
    const instance = await effect.create({
      device,
      format: 'rgba16float',
      canvas,
      params: params as never,
      profiler: noProfiler,
    });
    instance.resize(WIDTH, HEIGHT, 1);

    let frame = 0;
    const check = async (label: string) => {
      for (let i = 0; i < 2; i++) {
        const encoder = device.createCommandEncoder();
        instance.frame({ time: frame / 60, delta: 1 / 60, frame }, encoder, target.createView());
        device.queue.submit([encoder.finish()]);
        frame++;
      }
      const data = await readFloats(device, target);
      for (let i = 0; i < data.length; i++) {
        const value = data[i] ?? Number.NaN;
        if (!(value >= 0 && value <= 1.0001)) {
          throw new Error(
            `${id}: ${label} gave ${value} at channel ${i % 4}, pixel ${(i / 4) | 0}`,
          );
        }
      }
    };

    await check('defaults');
    for (const [name, preset] of Object.entries(effect.presets ?? {})) {
      Object.assign(params, defaultParams(effect.params), preset);
      await check(`preset ${name}`);
    }
    for (const [key, def] of Object.entries(effect.params)) {
      const values: unknown[] =
        def.type === 'number'
          ? [def.min, def.max, (def.min + def.max) / 2]
          : def.type === 'boolean'
            ? [true, false]
            : def.type === 'select'
              ? [...def.options]
              : [
                  [0, 0, 0],
                  [1, 1, 1],
                ];
      for (const value of values) {
        Object.assign(params, defaultParams(effect.params));
        params[key] = value;
        await check(`${key} = ${JSON.stringify(value)}`);
      }
    }
    Object.assign(params, defaultParams(effect.params));
    for (const [key, def] of Object.entries(effect.params)) {
      if (def.type === 'number') params[key] = def.max;
    }
    await check('every number at its maximum');
    Object.assign(params, defaultParams(effect.params));
    for (const [key, def] of Object.entries(effect.params)) {
      if (def.type === 'number') params[key] = def.min;
    }
    await check('every number at its minimum');

    // Odd canvas shapes: very wide and very tall.
    for (const [w, h] of [
      [128, 16],
      [16, 128],
    ] as const) {
      const odd = device.createTexture({
        size: [w, h],
        format: 'rgba16float',
        usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC,
      });
      Object.assign(params, defaultParams(effect.params));
      instance.resize(w, h, 1);
      const encoder = device.createCommandEncoder();
      instance.frame({ time: 1, delta: 1 / 60, frame: frame++ }, encoder, odd.createView());
      device.queue.submit([encoder.finish()]);
      for (const value of await readFloats(device, odd))
        expect(value >= 0 && value <= 1.0001).toBe(true);
      odd.destroy();
    }
    instance.destroy();
    target.destroy();
  });

  it('media-distortion handles the pointer, a press, a dropped file and the video source without a DOM', async () => {
    const device = await getTestDevice();
    const { loadEffect } = await import('../src/effects/registry.ts');
    const { defaultParams } = await import('../src/effects/types.ts');
    const { noProfiler } = await import('../src/gpu/profiler.ts');
    const effect = await loadEffect('media-distortion');
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
    const target = device.createTexture({
      size: [WIDTH, HEIGHT],
      format: 'rgba16float',
      usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC,
    });
    const params = defaultParams(effect.params) as Record<string, unknown>;
    const listeners = Object.assign(new EventTarget(), {
      width: WIDTH,
      height: HEIGHT,
      style: {},
      getBoundingClientRect: () => ({ left: 0, top: 0, width: WIDTH, height: HEIGHT }),
      setPointerCapture() {},
    }) as unknown as HTMLCanvasElement;
    const instance = await effect.create({
      device,
      format: 'rgba16float',
      canvas: listeners,
      params: params as never,
      profiler: noProfiler,
    });
    instance.resize(WIDTH, HEIGHT, 1);
    const fire = (type: string, extra: object = {}) =>
      listeners.dispatchEvent(Object.assign(new Event(type, { cancelable: true }), extra));
    let frame = 0;
    const render = async () => {
      for (let i = 0; i < 3; i++) {
        const encoder = device.createCommandEncoder();
        instance.frame({ time: frame / 60, delta: 1 / 60, frame }, encoder, target.createView());
        device.queue.submit([encoder.finish()]);
        frame++;
        await new Promise((resolve) => setTimeout(resolve, 5));
      }
      for (const value of await readFloats(device, target)) {
        if (!(value >= 0 && value <= 1.0001)) throw new Error(`pointer scenario gave ${value}`);
      }
    };

    for (const mode of ['ambient', 'ripple', 'hover']) {
      params.interaction = mode;
      fire('pointerenter', { clientX: 10, clientY: 10 });
      for (let i = 0; i < 6; i++)
        fire('pointermove', { clientX: 10 + i * 12, clientY: 20 + i * 5 });
      fire('pointerdown', { clientX: 50, clientY: 30, pointerId: 1 });
      await render();
      fire('pointerup', { pointerId: 1 });
      fire('pointerleave');
      await render();
    }

    // A file that is not media is ignored with a warning; an image needs createImageBitmap, which
    // Node lacks, so it fails cleanly and the picture stays.
    const text = new File(['hi'], 'notes.txt', { type: 'text/plain' });
    fire('drop', { dataTransfer: { files: [text], types: ['Files'] } });
    const png = new File(['x'], 'a.png', { type: 'image/png' });
    fire('dragover', { dataTransfer: { files: [png], types: ['Files'] } });
    fire('drop', { dataTransfer: { files: [png], types: ['Files'] } });
    await new Promise((resolve) => setTimeout(resolve, 20));
    await render();
    expect(warn).toHaveBeenCalled();

    // The video source falls back to the procedural picture when there is no DOM.
    params.source = 'video';
    await render();
    await new Promise((resolve) => setTimeout(resolve, 20));
    await render();
    params.source = 'still';
    await render();

    instance.destroy();
    target.destroy();
  });
});
