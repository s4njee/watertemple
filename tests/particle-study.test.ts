import { mkdirSync, writeFileSync } from 'node:fs';
import { dirname } from 'node:path';
import { describe, expect, it } from 'vitest';
import { loadEffect } from '../src/effects/registry.ts';
import { defaultParams } from '../src/effects/types.ts';
import { createPbf } from '../src/gpu/sim/pbf.ts';
import type { PassProfiler } from '../src/gpu/profiler.ts';

const destination = process.env.PARTICLE_STUDY;
const median = (samples: number[]) =>
  [...samples].sort((a, b) => a - b)[Math.floor(samples.length / 2)]!;

// Explicitly opt in: this is a measurement workload, never a timing assertion in CI.
describe.skipIf(!destination)('particle scale study', () => {
  it('records solver scaling and complete 1080p frames on the available adapter', async () => {
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) throw new Error('No adapter for particle study');
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
            deviceLabel: process.env.PARTICLE_STUDY_DEVICE ?? adapter.info.description,
            deviceClass: process.env.PARTICLE_STUDY_CLASS ?? 'unclassified',
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
      for (const dimension of [2, 3] as const) {
        for (const target of [4096, 16384, 65536]) {
          const side = Math.floor(target ** (1 / dimension));
          const spacing = 0.8 / side;
          const pbf = createPbf({
            device,
            dimension,
            capacity: target + 64,
            spacing,
            bounds: { min: [0, 0, dimension === 3 ? 0 : -1], max: [1, 1, 1] },
            profiler,
            timedPasses: 10000,
          });
          pbf.reset({
            blocks: [
              {
                min: [0, 0, 0],
                max: [side * spacing, side * spacing, dimension === 3 ? side * spacing : 0],
                jitter: 0,
              },
            ],
          });
          pbf.params.gravity = [0, 0, 0];
          pbf.params.artificialPressure = 0;
          for (const iterations of [2, 4]) {
            pbf.params.iterations = iterations;
            pbf.params.substeps = 1;
            const timing = await measure((encoder) => pbf.run(encoder));
            rows.push({
              kind: 'solver substep',
              id: `pbf-${dimension}d`,
              particles: pbf.count,
              iterations,
              ...timing,
            });
            console.log(
              `PBF ${dimension}D ${pbf.count}, ${iterations} iterations: ${timing.slopeMedianMs.toFixed(3)} ms/substep`,
            );
            report();
          }
          pbf.destroy();
        }
      }
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
      for (const id of ['particles', 'sph-2d', 'pbf-2d', 'pbf-3d', 'mpm-3d', 'fluid-surface']) {
        const effect = await loadEffect(id);
        const counts = id === 'pbf-3d' || id === 'mpm-3d' ? [10000, 40000, 100000] : [undefined];
        for (const count of counts) {
          const params = defaultParams(effect.params);
          const countKey = 'particles' in effect.params ? 'particles' : 'count';
          if (count !== undefined) params[countKey] = count;
          if ('autoRotate' in params) params.autoRotate = false;
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
            id,
            requestedParticles: count ?? params[countKey] ?? null,
            params,
            ...timing,
          });
          console.log(`${id}, ${count ?? 'default'}: ${timing.slopeMedianMs.toFixed(3)} ms/frame`);
          report();
          instance.destroy();
        }
      }
      target.destroy();
      expect(rows.length).toBe(22);
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
