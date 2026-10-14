import { expect, it } from 'vitest';
import { effect } from '../src/effects/particles/index.ts';
import { defaultParams } from '../src/effects/types.ts';
import { noProfiler } from '../src/gpu/profiler.ts';
import { expectValid, getTestDevice, readTexture } from './gpu.ts';

it('preserves lantern highlights above SDR white and limits them with HDR headroom', async () => {
  const device = await getTestDevice();
  const params = { ...defaultParams(effect.params), ...effect.presets!.lantern, count: 60000 };
  const canvas = Object.assign(new EventTarget(), {
    width: 128,
    height: 128,
    style: {},
    getBoundingClientRect: () => ({ left: 0, top: 0, width: 128, height: 128 }),
    setPointerCapture() {},
    releasePointerCapture() {},
  }) as unknown as HTMLCanvasElement;
  const target = device.createTexture({
    size: [128, 128],
    format: 'rgba16float',
    usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC,
  });
  await expectValid(device, async () => {
    const instance = await effect.create({
      device,
      format: 'rgba16float',
      canvas,
      params,
      profiler: noProfiler,
      hdr: true,
    });
    try {
      instance.resize(128, 128, 1);
      const draw = async (headroom: number) => {
        params.headroom = headroom;
        const encoder = device.createCommandEncoder();
        instance.frame({ delta: 0, time: 0, frame: 0 }, encoder, target.createView());
        device.queue.submit([encoder.finish()]);
        const bytes = await readTexture(device, target, 8);
        const values = new Uint16Array(bytes.buffer, bytes.byteOffset, bytes.byteLength / 2);
        let peak = 0;
        for (let i = 0; i < values.length; i += 4) {
          peak = Math.max(peak, values[i]!, values[i + 1]!, values[i + 2]!);
          // Opaque output prevents accumulated particle alpha from dimming HDR composition.
          expect(values[i + 3]).toBe(0x3c00);
        }
        return peak;
      };
      const sdr = await draw(1);
      const hdr = await draw(3);
      // Positive half-float bit patterns are ordered. 0x3c00 is 1.0, SDR white.
      expect(sdr).toBeLessThanOrEqual(0x3c00);
      expect(hdr).toBeGreaterThan(0x3c00);
      expect(hdr).toBeLessThan(0x4200); // bounded below 3.0, with no infinities.
    } finally {
      instance.destroy();
    }
  });
  target.destroy();
});
