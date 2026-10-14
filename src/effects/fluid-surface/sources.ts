import type { ParticleSet } from '../../gpu/particles/types.ts';
import type { PassProfiler } from '../../gpu/profiler.ts';
import { createShader } from '../../gpu/shader.ts';
import { createUniformBuffer, defineUniformStruct } from '../../gpu/uniforms.ts';
import sourcesShader from './sources.wgsl';

export const SOURCES = ['ball', 'wave', 'stream'] as const;
export type SourceName = (typeof SOURCES)[number];

const SourceParams = defineUniformStruct('SourceParams', {
  kind: 'u32',
  count: 'u32',
  time: 'f32',
  radius: 'f32',
});

/**
 * Liquid volume of each source in cubic metres, which with the particle count fixes the spacing.
 * The stream's figure counts the pool, the jet and the spray together.
 */
const VOLUME: Record<SourceName, number> = { ball: 4.3, wave: 14, stream: 2.6 };

/**
 * Radius of the spheres for a set of `count` particles in the source's volume: the mean spacing.
 * Neighbouring spheres overlap by half their depth, so the union of them has shallow valleys that
 * the filter can remove; a radius near half the spacing leaves holes and deep ones.
 */
export function naturalRadius(source: SourceName, count: number): number {
  return Math.cbrt(VOLUME[source] / Math.max(count, 1));
}

/**
 * The thickness gain that makes the summed chords match the liquid: the volume a particle stands for
 * over the volume of its sphere. Rendered radius changes the sphere, not the liquid.
 */
export function thicknessGain(source: SourceName, count: number, radius: number): number {
  const volumePerParticle = VOLUME[source] / Math.max(count, 1);
  return volumePerParticle / ((4 / 3) * Math.PI * radius ** 3);
}

export interface Sources {
  /** Fills the buffer for `source` at `time` (a compute pass) and returns the set to render. */
  update(
    encoder: GPUCommandEncoder,
    source: SourceName,
    count: number,
    radius: number,
    time: number,
    profiler: PassProfiler,
  ): ParticleSet;
  destroy(): void;
}

/** Analytic particle sources (see sources.wgsl). The buffer grows by powers of two as the count does. */
export function createSources(device: GPUDevice): Sources {
  const module = createShader(device, {
    label: 'fluid-surface/sources.wgsl',
    code: sourcesShader.code,
    chunks: { ...sourcesShader.chunks, 'source-params.wgsl': SourceParams.wgsl },
  });
  const pipeline = device.createComputePipeline({
    label: 'fluid-surface sources',
    layout: 'auto',
    compute: { module, entryPoint: 'cs' },
  });
  const uniforms = createUniformBuffer(device, SourceParams, 'fluid-surface source params');

  let buffer: GPUBuffer | undefined;
  let capacity = 0;
  let generation = 0;
  let group: GPUBindGroup | undefined;

  function ensure(count: number): GPUBuffer {
    if (buffer && count <= capacity) return buffer;
    buffer?.destroy();
    capacity = Math.max(4096, 2 ** Math.ceil(Math.log2(Math.max(count, 1))));
    generation++;
    buffer = device.createBuffer({
      label: 'fluid-surface particles',
      size: capacity * 16,
      usage:
        GPUBufferUsage.STORAGE |
        GPUBufferUsage.VERTEX |
        GPUBufferUsage.COPY_SRC |
        GPUBufferUsage.COPY_DST,
    });
    group = device.createBindGroup({
      label: 'fluid-surface sources',
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: uniforms.buffer } },
        { binding: 1, resource: { buffer } },
      ],
    });
    return buffer;
  }

  return {
    update(encoder, source, count, radius, time, profiler) {
      const live = Math.max(1, Math.floor(count));
      const positions = ensure(live);
      uniforms.set({ kind: SOURCES.indexOf(source), count: live, time, radius });
      uniforms.upload();
      const pass = encoder.beginComputePass({
        label: 'fluid-surface sources',
        timestampWrites: profiler.pass('sources') as GPUComputePassTimestampWrites | undefined,
      });
      pass.setPipeline(pipeline);
      pass.setBindGroup(0, group!);
      pass.dispatchWorkgroups(Math.ceil(live / 64));
      pass.end();
      return { positions, count: live, capacity, radius, generation };
    },
    destroy() {
      buffer?.destroy();
      uniforms.destroy();
      buffer = undefined;
    },
  };
}
