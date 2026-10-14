import { createUniformBuffer, defineUniformStruct } from '../uniforms.ts';
import type { PassProfiler } from '../profiler.ts';
import { createComputePass } from './compute.ts';
import { createNeighborGrid } from './neighbor-grid.ts';
import { createParticleSystem } from './system.ts';
import type { ParticleSet } from './types.ts';
import shader from './whitewater.wgsl?raw';

const STRUCT = defineUniformStruct('WhitewaterParams', {
  gravity: 'vec3f',
  dt: 'f32',
  bounds_min: 'vec3f',
  count: 'u32',
  bounds_max: 'vec3f',
  frame: 'u32',
  emission: 'f32',
  lifetime: 'f32',
  radius: 'f32',
  min_speed: 'f32',
  drag: 'f32',
  buoyancy: 'f32',
  bubble_neighbors: 'u32',
});

export interface WhitewaterParams {
  emission: number;
  lifetime: number;
  minSpeed: number;
  drag: number;
  buoyancy: number;
  bubbleNeighbors: number;
}

/** A bounded secondary store; dead slots are offscreen and get recycled on the GPU. */
export function createWhitewater(options: {
  device: GPUDevice;
  capacity?: number;
  radius: number;
  searchRadius: number;
  bounds: { min: readonly [number, number, number]; max: readonly [number, number, number] };
  profiler?: PassProfiler;
}) {
  const { device, bounds, radius, searchRadius, profiler } = options;
  const system = createParticleSystem({
    device,
    capacity: options.capacity ?? 4096,
    radius,
    label: 'whitewater',
  });
  system.fill(system.capacity, (_i, p) => {
    p.position = [1e6, 1e6, 1e6];
  });
  const grid = createNeighborGrid({ device, dimension: 3, mode: 'sorted', profiler });
  grid.configure({ ...bounds, radius: searchRadius });
  const uniform = createUniformBuffer(device, STRUCT, 'whitewater params');
  const pass = createComputePass(device, {
    label: 'particles/whitewater.wgsl',
    code: shader,
    chunks: { ...grid.chunks({ binding: 5 }), 'whitewater_params.wgsl': STRUCT.wgsl },
    profiler,
    timestamp: 'whitewater update',
  });
  const params: WhitewaterParams = {
    emission: 16,
    lifetime: 2,
    minSpeed: 0.8,
    drag: 8,
    buoyancy: 1.5,
    bubbleNeighbors: 18,
  };
  let frame = 0;
  let destroyed = false;
  return {
    system,
    particles: system.particles,
    params,
    update(
      encoder: GPUCommandEncoder,
      source: ParticleSet,
      delta: number,
      gravity: readonly [number, number, number] = [0, -9.81, 0],
    ) {
      // Primary positions.w is a 0..1 foam/emission indicator; primary buffers are read-only here.
      // One update per submission: queue writes set this submission's uniform values.
      const dt = Number.isFinite(delta) ? Math.min(1 / 30, Math.max(0, delta)) : 0;
      if (dt === 0) return;
      if (!source.velocities) throw new TypeError('Whitewater needs primary velocities');
      grid.build(encoder, source);
      uniform.set({
        gravity,
        dt,
        bounds_min: bounds.min,
        bounds_max: bounds.max,
        count: source.count,
        frame: frame++ >>> 0,
        emission: Math.max(0, params.emission),
        lifetime: Math.max(0.01, params.lifetime),
        radius,
        min_speed: Math.max(0, params.minSpeed),
        drag: Math.max(0, params.drag),
        buoyancy: Math.max(0, params.buoyancy),
        bubble_neighbors: Math.max(2, Math.round(params.bubbleNeighbors)),
      });
      uniform.upload();
      pass.runCount(
        encoder,
        [
          uniform.buffer,
          source.positions,
          source.velocities,
          system.positions,
          system.velocities!,
          ...grid.resources(),
        ],
        system.count,
      );
    },
    reset() {
      system.fill(system.capacity, (_i, p) => {
        p.position = [1e6, 1e6, 1e6];
      });
    },
    destroy() {
      if (destroyed) return;
      destroyed = true;
      pass.destroy();
      uniform.destroy();
      grid.destroy();
      system.destroy();
    },
  };
}
