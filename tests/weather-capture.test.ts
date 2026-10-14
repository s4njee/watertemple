import { mkdirSync, writeFileSync } from 'node:fs';
import { expect, it } from 'vitest';
import { loadEffect } from '../src/effects/registry.ts';
import { defaultParams } from '../src/effects/types.ts';
import { noProfiler } from '../src/gpu/profiler.ts';
import { WEATHER_EFFECT_IDS } from '../src/gpu/weather/sources.ts';
import { expectValid, getTestDevice, readTexture } from './gpu.ts';
const destination = process.env.WEATHER_CAPTURE;
it.skipIf(!destination)(
  'captures actual GPU pixels for six weather fallback posters',
  async () => {
    const device = await getTestDevice();
    await expectValid(device, async () => {
      const width = 640,
        height = 400;
      const target = device.createTexture({
        size: [width, height],
        format: 'rgba8unorm',
        usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC,
      });
      const canvas = Object.assign(new EventTarget(), {
        width,
        height,
        style: {},
        getBoundingClientRect: () => ({ left: 0, top: 0, width, height }),
        setPointerCapture() {},
        releasePointerCapture() {},
      }) as unknown as HTMLCanvasElement;
      mkdirSync(destination!, { recursive: true });
      try {
        for (const id of WEATHER_EFFECT_IDS) {
          const effect = await loadEffect(id),
            params = defaultParams(effect.params);
          if (id === 'frost-melt') {
            params.cycle = 'freeze';
            params.growth = 1;
          }
          const instance = await effect.create({
            device,
            format: 'rgba8unorm',
            canvas,
            params,
            profiler: noProfiler,
          });
          instance.resize(width, height, 1);
          try {
            for (let frame = 0; frame < (id === 'frost-melt' ? 200 : 80); frame++) {
              if (id === 'fogged-glass' && frame === 79) {
                const event = new Event('pointerdown');
                Object.defineProperties(event, {
                  pointerId: { value: 1 },
                  pointerType: { value: 'mouse' },
                  clientX: { value: width * 0.5 },
                  clientY: { value: height * 0.5 },
                });
                canvas.dispatchEvent(event);
              }
              const encoder = device.createCommandEncoder();
              instance.frame(
                { time: frame * 0.05, delta: 0.05, frame },
                encoder,
                target.createView(),
              );
              device.queue.submit([encoder.finish()]);
            }
            const rgba = await readTexture(device, target);
            expect(new Set(rgba).size).toBeGreaterThan(30);
            writeFileSync(`${destination}/${id}.rgba`, rgba);
            writeFileSync(
              `${destination}/${id}.json`,
              JSON.stringify({ width, height, params }, null, 2),
            );
          } finally {
            instance.destroy();
          }
        }
      } finally {
        target.destroy();
      }
    });
  },
  120000,
);
