import { mkdirSync, writeFileSync } from 'node:fs';
import { dirname } from 'node:path';
import { describe, expect, it } from 'vitest';
import { loadEffect } from '../src/effects/registry.ts';
import { defaultParams } from '../src/effects/types.ts';
import { WEATHER_EFFECT_IDS } from '../src/gpu/weather/sources.ts';
import type { PassProfiler } from '../src/gpu/profiler.ts';

const destination = process.env.WEATHER_STUDY;
const median = (samples: number[]) =>
  [...samples].sort((a, b) => a - b)[Math.floor(samples.length / 2)]!;

// Explicitly opt in: this is a measurement workload, never a timing assertion in CI.
describe.skipIf(!destination)('weather cost study', () => {
  it('records complete weather frames at 1080p on the available adapter', async () => {
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) throw new Error('No adapter for weather study');
    const timed = adapter.features.has('timestamp-query');
    const device = await adapter.requestDevice({
      requiredFeatures: timed ? ['timestamp-query'] : [],
    });
    const query = timed ? device.createQuerySet({ type: 'timestamp', count: 4096 }) : undefined;
    const resolve = timed
      ? device.createBuffer({
          size: 4096 * 8,
          usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC,
        })
      : undefined;
    const read = timed
      ? device.createBuffer({
          size: 4096 * 8,
          usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
        })
      : undefined;
    let passes = 0;
    const profiler: PassProfiler = {
      pass() {
        const i = passes++;
        if (i >= 2048) throw new Error('Study exceeded timestamp capacity');
        return query
          ? { querySet: query, beginningOfPassWriteIndex: i * 2, endOfPassWriteIndex: i * 2 + 1 }
          : undefined;
      },
    };
    const run = async (unit: (encoder: GPUCommandEncoder) => void, repeats: number) => {
      passes = 0;
      const encoder = device.createCommandEncoder();
      const started = performance.now();
      for (let i = 0; i < repeats; i++) unit(encoder);
      const count = passes * 2;
      if (query && count > 0) {
        encoder.resolveQuerySet(query, 0, count, resolve!, 0);
        encoder.copyBufferToBuffer(resolve!, 0, read!, 0, count * 8);
      }
      device.queue.submit([encoder.finish()]);
      if (read && count > 0) {
        await read.mapAsync(GPUMapMode.READ);
        const stamps = new BigUint64Array(read.getMappedRange());
        const ms = Number(stamps[count - 1]! - stamps[0]!) / 1e6;
        read.unmap();
        return ms;
      }
      await device.queue.onSubmittedWorkDone();
      return performance.now() - started;
    };
    const measure = async (unit: (encoder: GPUCommandEncoder) => void) => {
      for (let i = 0; i < 8; i++) await run(unit, 1);
      const one: number[] = [],
        four: number[] = [];
      for (let i = 0; i < 9; i++) {
        one.push(await run(unit, 1));
        four.push(await run(unit, 4));
      }
      const slopes = one.map((ms, i) => (four[i]! - ms) / 3);
      return {
        spanMedianMs: median(one),
        slopeMedianMs: median(slopes),
        slopeMinMs: Math.min(...slopes),
        samples: { one, four },
        passesPerUnit: passes / 4,
      };
    };
    const rows: unknown[] = [];
    const report = () => {
      mkdirSync(dirname(destination!), { recursive: true });
      writeFileSync(
        destination!,
        JSON.stringify(
          {
            measuredAt: new Date().toISOString(),
            adapter: {
              vendor: adapter.info.vendor,
              architecture: adapter.info.architecture,
              device: adapter.info.device,
              description: adapter.info.description,
            },
            deviceLabel: process.env.WEATHER_STUDY_DEVICE ?? adapter.info.description,
            deviceClass: process.env.WEATHER_STUDY_CLASS ?? 'unclassified',
            metric: timed
              ? 'GPU timestamp span and workload slope'
              : 'CPU encode + queue completion wall time',
            resolution: [1920, 1080],
            rounds: 9,
            repeats: [1, 4],
            warmup: 8,
            note: 'Only this adapter was measured. Device classes and phone thermal behavior must be tested on those devices. A negative workload slope means the timing floor/noise dominates, not negative cost.',
            rows,
          },
          null,
          2,
        ) + '\n',
      );
    };
    device.pushErrorScope('validation');
    try {
      const width = 1920,
        height = 1080;
      const target = device.createTexture({
        size: [width, height],
        format: 'rgba8unorm',
        usage: GPUTextureUsage.RENDER_ATTACHMENT,
      });
      const canvas = Object.assign(new EventTarget(), {
        width,
        height,
        style: {},
        getBoundingClientRect: () => ({ left: 0, top: 0, width, height }),
        setPointerCapture() {},
        releasePointerCapture() {},
      }) as unknown as HTMLCanvasElement;
      const configs: {
        id: string;
        overrides?: Record<string, string | number | boolean>;
        pathSegments?: number;
      }[] = WEATHER_EFFECT_IDS.map((id) => ({ id }));
      configs.push({ id: 'rain-streaks', overrides: { count: 1000, wind: 2 } });
      configs.push({ id: 'bubbles', overrides: { style: 'soap', count: 140, film: 1000 } });
      configs.push({ id: 'splash-burst', overrides: { count: 128 }, pathSegments: 1 });
      configs.push({ id: 'fogged-glass', pathSegments: 16 });
      configs.push({ id: 'frost-melt', overrides: { cycle: 'freeze', growth: 2 } });
      configs.push({ id: 'frost-melt', overrides: { cycle: 'melt' } });
      for (const config of configs) {
        const effect = await loadEffect(config.id);
        const params = { ...defaultParams(effect.params), ...config.overrides };
        const instance = await effect.create({
          device,
          format: 'rgba8unorm',
          canvas,
          params,
          profiler,
        });
        instance.resize(width, height, 1);
        if (config.pathSegments) {
          const enter = new Event('pointerenter');
          Object.defineProperties(enter, {
            pointerId: { value: 1 },
            pointerType: { value: 'mouse' },
            clientX: { value: width * 0.2 },
            clientY: { value: height * 0.5 },
          });
          canvas.dispatchEvent(enter);
          if (config.id === 'splash-burst' || config.id === 'fogged-glass') {
            const press = new Event('pointerdown');
            Object.defineProperties(press, {
              clientX: { value: width * 0.5 },
              clientY: { value: height * 0.5 },
            });
            canvas.dispatchEvent(press);
          }
        }
        let frame = 0;
        const timing = await measure((encoder) => {
          if (config.pathSegments) {
            const event = new Event('pointermove');
            const coalesced = Array.from({ length: config.pathSegments }, (_, i) => ({
              clientX: (0.2 + i * 0.04) * width,
              clientY: (0.5 + 0.1 * Math.sin(i)) * height,
              timeStamp: (frame * 1000) / 60 + i,
              pointerId: 1,
              pointerType: 'mouse',
            }));
            Object.defineProperties(event, {
              pointerId: { value: 1 },
              pointerType: { value: 'mouse' },
              clientX: { value: (config.id === 'splash-burst' ? 0.5 : 0.8) * width },
              clientY: { value: 0.5 * height },
              getCoalescedEvents: { value: () => coalesced },
            });
            canvas.dispatchEvent(event);
          }
          instance.frame(
            { time: frame / 60, delta: 1 / 60, frame: frame++ },
            encoder,
            target.createView(),
          );
        });
        rows.push({
          kind: 'complete frame',
          id: config.id,
          params,
          inputSegments: config.pathSegments ?? 0,
          ...timing,
        });
        console.log(
          config.id,
          config.overrides ?? 'default',
          timing.slopeMedianMs.toFixed(3) + ' ms',
        );
        report();
        instance.destroy();
      }
      target.destroy();
      expect(rows.length).toBe(12);
      const error = await device.popErrorScope();
      device.pushErrorScope('validation');
      if (error) throw new Error(error.message);
    } finally {
      await device.popErrorScope();
      query?.destroy();
      resolve?.destroy();
      read?.destroy();
      device.destroy();
    }
  }, 600000);
});
