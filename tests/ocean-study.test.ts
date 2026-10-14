import { mkdirSync, writeFileSync } from 'node:fs';
import { dirname } from 'node:path';
import { describe, expect, it } from 'vitest';
import { loadEffect } from '../src/effects/registry.ts';
import { defaultParams } from '../src/effects/types.ts';
import { createBuoyancyReadback } from '../src/gpu/ocean/buoyancy.ts';
import { sampleWorldWaves } from '../src/gpu/ocean/waves.ts';
import { oceanMesh } from '../src/gpu/ocean/mesh.ts';
import type { PassProfiler } from '../src/gpu/profiler.ts';

const destination = process.env.OCEAN_STUDY;
const median = (samples: number[]) =>
  [...samples].sort((a, b) => a - b)[Math.floor(samples.length / 2)]!;

// Explicitly opt in: this is a measurement workload, never a timing assertion in CI.
describe.skipIf(!destination)('ocean cost study', () => {
  it('records mesh, FFT, buoyancy and complete 1080p frames on the available adapter', async () => {
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) throw new Error('No adapter for ocean study');
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
            deviceLabel: process.env.OCEAN_STUDY_DEVICE ?? adapter.info.description,
            deviceClass: process.env.OCEAN_STUDY_CLASS ?? 'unclassified',
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
      const configs: { id: string; overrides?: Record<string, string | number> }[] = [];
      for (const id of [
        'gerstner-ocean',
        'fft-ocean',
        'shoreline',
        'river',
        'waterfall',
        'waterline',
        'calm-lake',
      ])
        configs.push({ id });
      for (const mesh of ['projected', 'clipmap'])
        for (const cells of ['48', '96', '160']) {
          if (mesh === 'projected' && cells === '96') continue;
          configs.push({ id: 'gerstner-ocean', overrides: { mesh, cells } });
        }
      for (const fftSize of ['128', '256'])
        configs.push({ id: 'fft-ocean', overrides: { fftSize } });
      configs.push({ id: 'fft-ocean', overrides: { mesh: 'clipmap' } });
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
        let frame = 0;
        const timing = await measure((encoder) =>
          instance.frame(
            { time: frame / 60, delta: 1 / 60, frame: frame++ },
            encoder,
            target.createView(),
          ),
        );
        rows.push({
          kind: 'complete frame',
          id: config.id,
          params,
          triangles: oceanMesh(
            params.mesh === 'clipmap' ? 'clipmap' : 'projected',
            Number(params.cells),
          ).triangles,
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
      for (const count of [6, 96, 768]) {
        const points = new Float32Array(count * 4);
        for (let i = 0; i < count; i++) {
          points[i * 4] = (i % 32) - 16;
          points[i * 4 + 2] = -Math.floor(i / 32) - 5;
        }
        const cpuSamples: number[] = [];
        let checksum = 0;
        for (let r = 0; r < 9; r++) {
          const start = performance.now();
          for (let repeat = 0; repeat < 20; repeat++)
            for (let i = 0; i < count; i++) {
              checksum += sampleWorldWaves(points[i * 4]!, points[i * 4 + 2]!, 1.4, 1, 0.85)
                .position[1];
            }
          cpuSamples.push((performance.now() - start) / 20);
        }
        const query = createBuoyancyReadback(device, profiler, points);
        const timing = await measure((e) => query.encode(e, 1.4, 1, 0.85));
        const staging = device.createBuffer({
          size: points.byteLength,
          usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
        });
        const latencySamples: number[] = [];
        for (let r = 0; r < 9; r++) {
          const start = performance.now();
          passes = 0;
          const e = device.createCommandEncoder();
          query.encode(e, 1.4, 1, 0.85);
          e.copyBufferToBuffer(query.output, 0, staging, 0, points.byteLength);
          device.queue.submit([e.finish()]);
          await staging.mapAsync(GPUMapMode.READ);
          latencySamples.push(performance.now() - start);
          staging.unmap();
        }
        rows.push({
          kind: 'buoyancy query comparison',
          points: count,
          cpuMedianMs: median(cpuSamples),
          cpuSamples,
          gpuCompute: timing,
          readbackWallMedianMs: median(latencySamples),
          latencySamples,
          checksum,
          note: 'Readback wall time includes command encoding, submission, copy and mapAsync; interactive mode uses old samples while the map is pending, never blocks a frame.',
        });
        report();
        staging.destroy();
        query.destroy();
      }
      target.destroy();
      expect(rows.length).toBe(18);
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
