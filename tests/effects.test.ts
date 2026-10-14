import { describe, expect, it, vi } from 'vitest';
import { listEffects, loadEffect } from '../src/effects/registry.ts';
import { defaultParams } from '../src/effects/types.ts';
import { noProfiler } from '../src/gpu/profiler.ts';
import { expectValid, getTestDevice, readTexture } from './gpu.ts';

const WIDTH = 64;
const HEIGHT = 48;

/** There is no DOM here. Effects only use the canvas for input, so listeners are all it needs. */
function fakeCanvas(): HTMLCanvasElement {
  return Object.assign(new EventTarget(), {
    width: WIDTH,
    height: HEIGHT,
    style: {},
    getBoundingClientRect: () => ({ left: 0, top: 0, width: WIDTH, height: HEIGHT }),
    setPointerCapture() {},
    releasePointerCapture() {},
  }) as unknown as HTMLCanvasElement;
}

// Simulations bake textures and compile many pipelines; on a loaded machine that passes 5 seconds.
vi.setConfig({ testTimeout: 30_000 });

describe('registered effects', () => {
  it('finds at least one effect', () => {
    expect(listEffects().length).toBeGreaterThan(0);
  });

  // Every effect must survive its whole lifecycle without a validation error and draw something.
  it.each(listEffects().map((meta) => meta.id))('%s runs headlessly', async (id) => {
    const device = await getTestDevice();
    const format: GPUTextureFormat = 'rgba8unorm';
    const target = device.createTexture({
      size: [WIDTH, HEIGHT],
      format,
      usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC,
    });

    await expectValid(device, async () => {
      const effect = await loadEffect(id);
      expect(effect.meta.id).toBe(id);
      const instance = await effect.create({
        device,
        format,
        canvas: fakeCanvas(),
        params: defaultParams(effect.params),
        profiler: noProfiler,
        hdr: false,
      });
      instance.resize(WIDTH, HEIGHT, 1);
      for (let frame = 0; frame < 3; frame++) {
        const encoder = device.createCommandEncoder();
        instance.frame({ time: frame / 60, delta: 1 / 60, frame }, encoder, target.createView());
        device.queue.submit([encoder.finish()]);
      }
      instance.destroy();
    });

    const pixels = await readTexture(device, target);
    expect(new Set(pixels).size).toBeGreaterThan(8);
  });
});
