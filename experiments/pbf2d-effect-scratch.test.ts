import { describe, expect, it, vi } from 'vitest';
import { loadEffect } from '../src/effects/registry.ts';
import { defaultParams } from '../src/effects/types.ts';
import { noProfiler } from '../src/gpu/profiler.ts';
import { expectValid, getTestDevice, readTexture } from '../tests/gpu.ts';

vi.setConfig({ testTimeout: 600_000 });
const W = 160, H = 96;
function fakeCanvas(): HTMLCanvasElement {
  return Object.assign(new EventTarget(), { width: W, height: H, style: {}, getBoundingClientRect: () => ({ left: 0, top: 0, width: W, height: H }), setPointerCapture() {}, releasePointerCapture() {} }) as unknown as HTMLCanvasElement;
}

describe('pbf-2d scratch', () => {
  it('every scene x view x render', async () => {
    const device = await getTestDevice();
    const target = device.createTexture({ size: [W, H], format: 'rgba8unorm', usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC });
    const effect = await loadEffect('pbf-2d');
    const scenes = (effect.params.scene as { options: string[] }).options;
    const views = (effect.params.view as { options: string[] }).options;
    const renders = (effect.params.render as { options: string[] }).options;
    await expectValid(device, async () => {
      for (const scene of scenes) for (const view of views) for (const render of renders) {
        const params = { ...defaultParams(effect.params), scene, view, render, resolution: '12 mm' } as never;
        const canvas = fakeCanvas();
        const instance = await effect.create({ device, format: 'rgba8unorm', canvas, params, profiler: noProfiler, hdr: false });
        instance.resize(W, H, 1);
        for (let frame = 0; frame < 6; frame++) {
          const e = device.createCommandEncoder();
          instance.frame({ time: frame / 60, delta: 1 / 60, frame }, e, target.createView());
          device.queue.submit([e.finish()]);
        }
        instance.destroy();
      }
    });
    const pixels = await readTexture(device, target);
    expect(new Set(pixels).size).toBeGreaterThan(8);
  });

  it('long run: pour and drops restart, pointer, every resolution', async () => {
    const device = await getTestDevice();
    const target = device.createTexture({ size: [W, H], format: 'rgba8unorm', usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC });
    const effect = await loadEffect('pbf-2d');
    await expectValid(device, async () => {
      for (const [scene, frames] of [['pour', 2200], ['drops', 3600], ['dam break', 700], ['wave-maker', 500]] as const) {
        const params = { ...defaultParams(effect.params), scene, resolution: '12 mm', view: 'neighbours' } as never;
        const canvas = fakeCanvas();
        const instance = await effect.create({ device, format: 'rgba8unorm', canvas, params, profiler: noProfiler, hdr: false });
        instance.resize(W, H, 1);
        const fire = (type: string, x: number, y: number) => canvas.dispatchEvent(Object.assign(new Event(type), { pointerId: 1, pointerType: 'mouse', clientX: x, clientY: y, pressure: 0.5 }));
        for (let frame = 0; frame < frames; frame++) {
          if (frame === 100) { fire('pointerenter', 20, 60); fire('pointerdown', 20, 60); }
          if (frame > 100 && frame < 140) fire('pointermove', 20 + (frame - 100) * 3, 60);
          if (frame === 140) fire('pointerup', 140, 60);
          const e = device.createCommandEncoder();
          instance.frame({ time: frame / 60, delta: 1 / 60, frame }, e, target.createView());
          device.queue.submit([e.finish()]);
          if (frame % 100 === 99) await device.queue.onSubmittedWorkDone();
        }
        instance.destroy();
      }
      for (const resolution of ['8 mm', '6 mm', '4 mm', '3 mm']) {
        const params = { ...defaultParams(effect.params), resolution } as never;
        const instance = await effect.create({ device, format: 'rgba8unorm', canvas: fakeCanvas(), params, profiler: noProfiler, hdr: false });
        instance.resize(W, H, 1);
        for (let frame = 0; frame < 30; frame++) {
          const e = device.createCommandEncoder();
          instance.frame({ time: frame / 60, delta: 1 / 60, frame }, e, target.createView());
          device.queue.submit([e.finish()]);
        }
        instance.destroy();
      }
    });
  });
});
