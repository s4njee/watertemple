import { afterAll, beforeAll, describe, expect, it, vi } from 'vitest';
import {
  createGpuProfiler,
  MAX_TIMED_PASSES,
  READBACK_POOL_SIZE,
  frameSpanFromTimestamps,
  timingsFromTimestamps,
  type GpuProfiler,
} from '../src/gpu/profiler.ts';
import {
  estimateTextureBytes,
  getTextureMemory,
  trackTextureMemory,
  untrackTextureMemory,
} from '../src/gpu/texture-memory.ts';
import { expectValid } from './gpu.ts';

describe('frameSpanFromTimestamps', () => {
  it('spans first begin to last end, which overlapping passes do not sum to', () => {
    // Three passes that overlap, as tile-based GPUs report them: each lasts 1 ms, the frame 1.4.
    const stamps = BigUint64Array.from([
      1_000_000n,
      2_000_000n,
      1_200_000n,
      2_200_000n,
      1_400_000n,
      2_400_000n,
    ]);
    expect(frameSpanFromTimestamps(stamps, 3)).toBeCloseTo(1.4, 6);
    const summed = timingsFromTimestamps(stamps, ['a', 'b', 'c']).reduce((t, p) => t + p.ms, 0);
    expect(summed).toBeCloseTo(3, 6);
  });

  it('ignores the pairs timingsFromTimestamps would drop', () => {
    const stamps = BigUint64Array.from([
      0n,
      5_000_000n,
      9_000_000n,
      8_000_000n,
      3_000_000n,
      4_000_000n,
    ]);
    expect(frameSpanFromTimestamps(stamps, 3)).toBeCloseTo(1, 6);
  });

  it('is undefined when nothing valid was written', () => {
    expect(frameSpanFromTimestamps(BigUint64Array.from([0n, 0n]), 1)).toBeUndefined();
    expect(frameSpanFromTimestamps(new BigUint64Array(0), 0)).toBeUndefined();
  });
});

describe('timingsFromTimestamps', () => {
  it('converts nanosecond pairs to milliseconds per label', () => {
    const stamps = [1_000_000n, 3_500_000n, 4_000_000n, 4_250_000n];
    expect(timingsFromTimestamps(stamps, ['a', 'b'])).toEqual([
      { label: 'a', ms: 2.5 },
      { label: 'b', ms: 0.25 },
    ]);
  });

  it('sums a label used more than once', () => {
    const stamps = [10n, 1_000_010n, 2_000_000n, 3_000_000n];
    expect(timingsFromTimestamps(stamps, ['blur', 'blur'])).toEqual([{ label: 'blur', ms: 2 }]);
  });

  it('drops backwards, unwritten and absurd samples', () => {
    const stamps = [
      5_000n,
      1_000n, // end before begin, e.g. a counter reset
      0n,
      0n, // never written
      1n,
      5_000_000_000n, // five seconds
      100n,
      1_000_100n, // fine
    ];
    expect(timingsFromTimestamps(stamps, ['reset', 'unused', 'stall', 'ok'])).toEqual([
      { label: 'ok', ms: 1 },
    ]);
  });

  it('ignores labels with no timestamps behind them', () => {
    expect(timingsFromTimestamps([], ['a'])).toEqual([]);
  });
});

describe('estimateTextureBytes', () => {
  it('counts uncompressed formats per texel', () => {
    expect(estimateTextureBytes({ size: [16, 8], format: 'rgba8unorm' })).toEqual({
      bytes: 16 * 8 * 4,
      known: true,
    });
    expect(
      estimateTextureBytes({ size: { width: 4, height: 4 }, format: 'rgba16float' }).bytes,
    ).toBe(4 * 4 * 8);
    expect(estimateTextureBytes({ size: [4, 4], format: 'rgba32float' }).bytes).toBe(4 * 4 * 16);
    expect(estimateTextureBytes({ size: [4, 4], format: 'r8unorm' }).bytes).toBe(16);
  });

  it('sums the mip chain', () => {
    // 8x8 + 4x4 + 2x2 + 1x1 texels
    expect(estimateTextureBytes({ size: [8, 8], format: 'r8unorm', mipLevelCount: 4 }).bytes).toBe(
      64 + 16 + 4 + 1,
    );
    // Non-square: 8x2, 4x1, 2x1, 1x1
    expect(estimateTextureBytes({ size: [8, 2], format: 'r8unorm', mipLevelCount: 4 }).bytes).toBe(
      16 + 4 + 2 + 1,
    );
  });

  it('multiplies array layers and samples', () => {
    expect(estimateTextureBytes({ size: [4, 4, 6], format: 'rgba8unorm' }).bytes).toBe(
      4 * 4 * 6 * 4,
    );
    expect(estimateTextureBytes({ size: [4, 4], format: 'rgba8unorm', sampleCount: 4 }).bytes).toBe(
      4 * 4 * 4 * 4,
    );
    // Array layers keep their size down the chain: (4x4 + 2x2 + 1x1) x 3 layers.
    expect(
      estimateTextureBytes({ size: [4, 4, 3], format: 'r8unorm', mipLevelCount: 3 }).bytes,
    ).toBe((16 + 4 + 1) * 3);
  });

  it('shrinks 3D textures in depth with each mip', () => {
    // 4x4x4 + 2x2x2 + 1x1x1
    expect(
      estimateTextureBytes({
        size: [4, 4, 4],
        format: 'r8unorm',
        dimension: '3d',
        mipLevelCount: 3,
      }).bytes,
    ).toBe(64 + 8 + 1);
  });

  it('counts compressed formats per block, rounding partial blocks up', () => {
    expect(estimateTextureBytes({ size: [8, 8], format: 'bc1-rgba-unorm' }).bytes).toBe(4 * 8);
    expect(estimateTextureBytes({ size: [5, 5], format: 'bc7-rgba-unorm' }).bytes).toBe(4 * 16);
    expect(estimateTextureBytes({ size: [12, 12], format: 'astc-6x6-unorm' }).bytes).toBe(4 * 16);
  });

  it('counts unknown formats as 4 bytes per texel and says so', () => {
    expect(estimateTextureBytes({ size: [2, 2], format: 'made-up-format' })).toEqual({
      bytes: 16,
      known: false,
    });
  });

  it('does not let an over-large mip count run past the smallest level', () => {
    expect(estimateTextureBytes({ size: [2, 2], format: 'r8unorm', mipLevelCount: 10 }).bytes).toBe(
      4 + 1,
    );
  });
});

describe('texture memory tracking', () => {
  let device: GPUDevice;
  beforeAll(async () => {
    const adapter = await navigator.gpu.requestAdapter();
    device = await adapter!.requestDevice();
  });
  afterAll(() => device.destroy());

  it('follows createTexture and destroy, once each', async () => {
    trackTextureMemory(device);
    trackTextureMemory(device); // idempotent: no double wrapping, no double counting
    expect(getTextureMemory(device)).toMatchObject({ bytes: 0, textures: 0 });

    const usage = GPUTextureUsage.TEXTURE_BINDING;
    const a = await expectValid(device, () =>
      device.createTexture({ size: [8, 8], format: 'rgba8unorm', usage }),
    );
    const b = device.createTexture({ size: [4, 4], format: 'rgba16float', usage });
    expect(getTextureMemory(device)).toMatchObject({ bytes: 256 + 128, textures: 2 });

    a.destroy();
    a.destroy();
    expect(getTextureMemory(device)).toMatchObject({ bytes: 128, textures: 1 });
    b.destroy();
    expect(getTextureMemory(device)).toMatchObject({ bytes: 0, textures: 0 });
  });

  it('reports unknown formats', () => {
    const texture = device.createTexture({
      size: [4, 4],
      format: 'rgba8unorm',
      usage: GPUTextureUsage.TEXTURE_BINDING,
    });
    expect(getTextureMemory(device)?.unknownFormats).toEqual([]);
    texture.destroy();
  });

  it('keeps separate devices separate and can be removed', async () => {
    const other = await (await navigator.gpu.requestAdapter())!.requestDevice();
    expect(getTextureMemory(other)).toBeUndefined();
    untrackTextureMemory(device);
    expect(getTextureMemory(device)).toBeUndefined();
    expect(Object.prototype.hasOwnProperty.call(device, 'createTexture')).toBe(false);
    // Untracked, the device still works; tracking can start again.
    device
      .createTexture({ size: [2, 2], format: 'rgba8unorm', usage: GPUTextureUsage.TEXTURE_BINDING })
      .destroy();
    trackTextureMemory(device);
    expect(getTextureMemory(device)).toMatchObject({ bytes: 0, textures: 0 });
    other.destroy();
  });
});

describe('createGpuProfiler without timestamp-query', () => {
  const device = { features: new Set<string>() } as unknown as GPUDevice;

  it('is unsupported, stays disabled and never times a pass', () => {
    const profiler = createGpuProfiler(device);
    expect(profiler.supported).toBe(false);
    profiler.enabled = true;
    expect(profiler.enabled).toBe(false);
    profiler.beginFrame();
    expect(profiler.pass('a')).toBeUndefined();
    profiler.endFrame({} as GPUCommandEncoder);
    profiler.afterSubmit();
    expect(profiler.latest).toEqual([]);
    expect(() => profiler.destroy()).not.toThrow();
  });
});

describe('createGpuProfiler', () => {
  let device: GPUDevice;
  let target: GPUTexture;
  let pipeline: GPURenderPipeline;
  let hasFeature = false;

  beforeAll(async () => {
    const adapter = await navigator.gpu.requestAdapter();
    hasFeature = adapter?.features.has('timestamp-query') ?? false;
    if (!hasFeature) {
      console.warn('Dawn has no timestamp-query here: real-GPU profiler tests are skipped.');
      return;
    }
    device = await adapter!.requestDevice({ requiredFeatures: ['timestamp-query'] });
    target = device.createTexture({
      size: [256, 256],
      format: 'rgba8unorm',
      usage: GPUTextureUsage.RENDER_ATTACHMENT,
    });
    // Metal skips the end timestamp of a pass that draws nothing, so the passes must draw.
    const module = device.createShaderModule({
      code: `@vertex fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4f {
        var p = array(vec2f(-1, -1), vec2f(3, -1), vec2f(-1, 3));
        return vec4f(p[i], 0, 1);
      }
      @fragment fn fs() -> @location(0) vec4f { return vec4f(0, 0.5, 1, 1); }`,
    });
    pipeline = device.createRenderPipeline({
      layout: 'auto',
      vertex: { module },
      fragment: { module, targets: [{ format: 'rgba8unorm' }] },
    });
  });
  afterAll(() => device?.destroy());

  /** Encodes `labels.length` clear passes and submits them. */
  function frame(profiler: GpuProfiler, labels: string[]) {
    profiler.beginFrame();
    const encoder = device.createCommandEncoder();
    for (const label of labels) {
      const pass = encoder.beginRenderPass({
        colorAttachments: [
          {
            view: target.createView(),
            loadOp: 'clear',
            storeOp: 'store',
            clearValue: [0, 0, 1, 1],
          },
        ],
        timestampWrites: profiler.pass(label),
      });
      pass.setPipeline(pipeline);
      pass.draw(3);
      pass.end();
    }
    profiler.endFrame(encoder);
    device.queue.submit([encoder.finish()]);
    profiler.afterSubmit();
  }

  async function until(condition: () => boolean, ms = 3000) {
    const start = Date.now();
    while (!condition()) {
      if (Date.now() - start > ms) throw new Error('timed out waiting for the GPU');
      await new Promise((resolve) => setTimeout(resolve, 5));
    }
  }

  it('is supported and off until enabled', () => {
    if (!hasFeature) return;
    const profiler = createGpuProfiler(device);
    expect(profiler.supported).toBe(true);
    expect(profiler.enabled).toBe(false);
    profiler.destroy();
  });

  it('has zero GPU footprint while disabled', async () => {
    if (!hasFeature) return;
    const createQuerySet = vi.spyOn(device, 'createQuerySet');
    const createBuffer = vi.spyOn(device, 'createBuffer');
    const profiler = createGpuProfiler(device);
    const resolveQuerySet = vi.fn();
    profiler.beginFrame();
    expect(profiler.pass('a')).toBeUndefined();
    profiler.endFrame({ resolveQuerySet } as unknown as GPUCommandEncoder);
    profiler.afterSubmit();
    expect(createQuerySet).not.toHaveBeenCalled();
    expect(createBuffer).not.toHaveBeenCalled();
    expect(resolveQuerySet).not.toHaveBeenCalled();
    expect(profiler.latest).toEqual([]);
    profiler.destroy();
    createQuerySet.mockRestore();
    createBuffer.mockRestore();
  });

  it('measures plausible positive durations per pass, with no validation errors', async () => {
    if (!hasFeature) return;
    const profiler = createGpuProfiler(device);
    profiler.enabled = true;
    await expectValid(device, async () => {
      frame(profiler, ['first', 'second']);
      await until(() => profiler.latest.length > 0);
    });
    const { latest } = profiler;
    expect(latest.map((timing) => timing.label)).toEqual(['first', 'second']);
    for (const { ms } of latest) {
      // A pass this small can begin and end on the same timer tick, so zero is a valid reading.
      expect(ms).toBeGreaterThanOrEqual(0);
      expect(ms).toBeLessThan(1000);
    }
    // A fresh array per arrived frame, so consumers can tell samples apart.
    const before = profiler.latest;
    await expectValid(device, async () => {
      frame(profiler, ['first']);
      await until(() => profiler.latest !== before);
    });
    expect(profiler.latest.map((timing) => timing.label)).toEqual(['first']);
    profiler.destroy();
  });

  it('sums a label used twice in one frame', async () => {
    if (!hasFeature) return;
    const profiler = createGpuProfiler(device);
    profiler.enabled = true;
    frame(profiler, ['blur', 'blur', 'composite']);
    await until(() => profiler.latest.length > 0);
    expect(profiler.latest.map((timing) => timing.label)).toEqual(['blur', 'composite']);
    profiler.destroy();
  });

  it('caps passes per frame', async () => {
    if (!hasFeature) return;
    const profiler = createGpuProfiler(device);
    profiler.enabled = true;
    profiler.beginFrame();
    const granted = Array.from({ length: MAX_TIMED_PASSES + 5 }, (_, index) =>
      profiler.pass(`p${index}`),
    );
    expect(granted.filter(Boolean)).toHaveLength(MAX_TIMED_PASSES);
    expect(granted[MAX_TIMED_PASSES]).toBeUndefined();
    // Abandon the frame without encoding it.
    profiler.beginFrame();
    profiler.destroy();
  });

  it('skips timing when every readback buffer is busy, then recovers', async () => {
    if (!hasFeature) return;
    const profiler = createGpuProfiler(device);
    profiler.enabled = true;
    const timed: boolean[] = [];
    await expectValid(device, async () => {
      // No awaiting between frames, so no map can complete and none of the buffers frees up.
      for (let i = 0; i < READBACK_POOL_SIZE + 4; i++) {
        profiler.beginFrame();
        timed.push(profiler.pass('x') !== undefined);
        const encoder = device.createCommandEncoder();
        profiler.endFrame(encoder);
        // Nothing was encoded into a pass here, which is fine for this bookkeeping check; the
        // resolve covers unwritten queries.
        device.queue.submit([encoder.finish()]);
        profiler.afterSubmit();
      }
      await device.queue.onSubmittedWorkDone();
    });
    expect(timed.slice(0, READBACK_POOL_SIZE).every(Boolean)).toBe(true);
    expect(timed.slice(READBACK_POOL_SIZE).some(Boolean)).toBe(false);

    // Once the maps land the buffers are free again.
    await until(() => {
      profiler.beginFrame();
      const granted = profiler.pass('y') !== undefined;
      profiler.beginFrame(); // abandon: frees the slot it took
      return granted;
    });
    // Unwritten timestamps are dropped rather than published as zero-length passes.
    expect(profiler.latest).toEqual([]);
    profiler.destroy();
  });

  it('clears results when disabled and discards late ones', async () => {
    if (!hasFeature) return;
    const profiler = createGpuProfiler(device);
    profiler.enabled = true;
    frame(profiler, ['a']);
    profiler.enabled = false;
    expect(profiler.latest).toEqual([]);
    await device.queue.onSubmittedWorkDone();
    await new Promise((resolve) => setTimeout(resolve, 50));
    expect(profiler.latest).toEqual([]);
    profiler.enabled = true;
    frame(profiler, ['a']);
    await until(() => profiler.latest.length > 0);
    profiler.destroy();
  });

  it('survives destroy with frames in flight', async () => {
    if (!hasFeature) return;
    const profiler = createGpuProfiler(device);
    profiler.enabled = true;
    // An unhandled rejection from an aborted map would fail the whole vitest run.
    await expectValid(device, async () => {
      frame(profiler, ['a']);
      frame(profiler, ['b']);
      profiler.destroy();
      profiler.destroy();
      expect(profiler.enabled).toBe(false);
      expect(profiler.latest).toEqual([]);
      // After destroy the profiler is inert.
      profiler.enabled = true;
      expect(profiler.enabled).toBe(false);
      profiler.beginFrame();
      expect(profiler.pass('c')).toBeUndefined();
      await device.queue.onSubmittedWorkDone();
      await new Promise((resolve) => setTimeout(resolve, 50));
    });
  });
});
