import { noProfiler, type PassProfiler } from '../profiler.ts';
import { createShader, type ShaderSource } from '../shader.ts';

// A thin helper for "run this compute shader over all particles", used by everything in this folder and
// meant for the solvers built on it. It owns three chores that every compute pass repeats:
//
//  - PIPELINES are created through `createShader` (so errors show in the overlay with the right file and
//    line) and cached per device, keyed by the shader's label, code, chunks and entry point. Two passes made
//    from the same source share one pipeline, and a hot-reloaded edit gets a new one.
//  - BIND GROUPS are cached per buffer identity. A ping-pong or a reorder changes which GPUBuffer sits in a
//    binding every step; a cache keyed by the buffers (and sub-ranges) gives each arrangement its group once.
//  - DISPATCH SIZING: `runCount` turns a particle count into workgroups, and refuses a count that would need
//    more than the 65535 workgroups a single dimension allows (4.19 M particles at workgroup size 64).
//
// The layout is `'auto'`: bind exactly the bindings the entry point uses (a binding the entry point never
// touches is absent from the layout and `run` would fail validation). Everything binds in group 0; pass
// resources as an array whose index is the binding number.

/** What can sit in a binding: a whole buffer, a sub-range, or a sampler/texture view. */
export type BindingResource = GPUBuffer | GPUBufferBinding | GPUSampler | GPUTextureView;

/** The most workgroups one dispatch dimension may have on every conforming device. */
export const MAX_WORKGROUPS_PER_DIMENSION = 65535;

export interface ComputePassOptions {
  /** Shader label, also the file name shown in diagnostics, e.g. "particles/integrate.wgsl". */
  label: string;
  code: string;
  /** Effect-local include targets (see `ShaderSource.chunks`). */
  chunks?: Readonly<Record<string, string>>;
  /** Default `main`. */
  entryPoint?: string;
  /** Must equal the `@workgroup_size` in the shader. Default 64. */
  workgroupSize?: number;
  /** Timestamp slots for the HUD. Default: none. */
  profiler?: PassProfiler;
  /** Profiler label for this pass; passes that share a label are summed by the HUD. Default `label`. */
  timestamp?: string;
}

export interface ComputePass {
  readonly workgroupSize: number;
  /** The pipeline, for tests and for `getBindGroupLayout`. Shared between passes with the same source. */
  readonly pipeline: GPUComputePipeline;
  /** Dispatches `ceil(count / workgroupSize)` workgroups (nothing when `count` is 0). */
  runCount(
    encoder: GPUCommandEncoder,
    resources: readonly (BindingResource | undefined)[],
    count: number,
  ): void;
  /** Dispatches an explicit number of workgroups. */
  runGroups(
    encoder: GPUCommandEncoder,
    resources: readonly (BindingResource | undefined)[],
    groups: number | readonly [number, number?, number?],
  ): void;
  /** Drops cached bind groups (they hold references to the buffers they were made for). */
  clearBindGroups(): void;
  destroy(): void;
}

// --- Caches --------------------------------------------------------------------------------------

const pipelines = new WeakMap<GPUDevice, Map<string, GPUComputePipeline>>();
const modules = new WeakMap<GPUDevice, Map<string, GPUShaderModule>>();

function sourceKey(source: ShaderSource): string {
  return `${source.label}\u0000${source.code}\u0000${JSON.stringify(source.chunks ?? {})}`;
}

/** One module per distinct source and device, so several entry points of a file share a compile. */
export function cachedShaderModule(device: GPUDevice, source: ShaderSource): GPUShaderModule {
  let perDevice = modules.get(device);
  if (!perDevice) modules.set(device, (perDevice = new Map()));
  const key = sourceKey(source);
  let module = perDevice.get(key);
  if (!module) {
    module = createShader(device, source);
    perDevice.set(key, module);
    // A bounded cache: edited shaders during development would otherwise pile up.
    if (perDevice.size > 128) perDevice.delete(perDevice.keys().next().value as string);
  }
  return module;
}

function cachedPipeline(
  device: GPUDevice,
  source: ShaderSource,
  entryPoint: string,
): GPUComputePipeline {
  let perDevice = pipelines.get(device);
  if (!perDevice) pipelines.set(device, (perDevice = new Map()));
  const key = `${sourceKey(source)}\u0000${entryPoint}`;
  let pipeline = perDevice.get(key);
  if (!pipeline) {
    pipeline = device.createComputePipeline({
      label: `${source.label}:${entryPoint}`,
      layout: 'auto',
      compute: { module: cachedShaderModule(device, source), entryPoint },
    });
    perDevice.set(key, pipeline);
    if (perDevice.size > 128) perDevice.delete(perDevice.keys().next().value as string);
  }
  return pipeline;
}

const ids = new WeakMap<object, number>();
let nextId = 1;
const identity = (object: object): number => {
  let id = ids.get(object);
  if (id === undefined) ids.set(object, (id = nextId++));
  return id;
};

function resourceKey(resource: BindingResource | undefined): string {
  if (resource === undefined) return '-';
  if ('buffer' in resource) {
    return `${identity(resource.buffer)}:${resource.offset ?? 0}:${resource.size ?? 'all'}`;
  }
  return String(identity(resource));
}

function toEntry(resource: BindingResource, binding: number): GPUBindGroupEntry {
  if ('buffer' in resource) return { binding, resource };
  if (resource instanceof GPUBuffer) return { binding, resource: { buffer: resource } };
  return { binding, resource };
}

/** Cached bind groups per pass; cleared when it grows past this (stale groups pin destroyed buffers). */
const MAX_BIND_GROUPS = 24;

// --- The pass ------------------------------------------------------------------------------------

export function createComputePass(device: GPUDevice, options: ComputePassOptions): ComputePass {
  const { label, entryPoint = 'main', workgroupSize = 64, profiler = noProfiler } = options;
  const timestamp = options.timestamp ?? label;
  const source: ShaderSource = { label, code: options.code, chunks: options.chunks };
  const pipeline = cachedPipeline(device, source, entryPoint);
  const bindGroups = new Map<string, GPUBindGroup>();

  function bindGroupFor(resources: readonly (BindingResource | undefined)[]): GPUBindGroup {
    const key = resources.map(resourceKey).join('|');
    let group = bindGroups.get(key);
    if (!group) {
      if (bindGroups.size >= MAX_BIND_GROUPS) bindGroups.clear();
      const entries: GPUBindGroupEntry[] = [];
      resources.forEach((resource, binding) => {
        if (resource !== undefined) entries.push(toEntry(resource, binding));
      });
      group = device.createBindGroup({
        label: `${label}:${entryPoint}`,
        layout: pipeline.getBindGroupLayout(0),
        entries,
      });
      bindGroups.set(key, group);
    }
    return group;
  }

  function dispatch(
    encoder: GPUCommandEncoder,
    resources: readonly (BindingResource | undefined)[],
    groups: readonly [number, number, number],
  ): void {
    const pass = encoder.beginComputePass({
      label: timestamp,
      timestampWrites: profiler.pass(timestamp) as GPUComputePassTimestampWrites | undefined,
    });
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroupFor(resources));
    pass.dispatchWorkgroups(groups[0], groups[1], groups[2]);
    pass.end();
  }

  return {
    workgroupSize,
    pipeline,
    runCount(encoder, resources, count) {
      if (!(count >= 0) || !Number.isFinite(count)) {
        throw new RangeError(`${label}: count must be a finite number of at least 0, got ${count}`);
      }
      const groups = Math.ceil(count / workgroupSize);
      if (groups > MAX_WORKGROUPS_PER_DIMENSION) {
        throw new RangeError(
          `${label}: ${count} items need ${groups} workgroups of ${workgroupSize}; one dispatch allows ${MAX_WORKGROUPS_PER_DIMENSION}. Use a larger workgroup size.`,
        );
      }
      if (groups === 0) return;
      dispatch(encoder, resources, [groups, 1, 1]);
    },
    runGroups(encoder, resources, groups) {
      const [x, y = 1, z = 1] = typeof groups === 'number' ? [groups] : groups;
      if (x * y * z === 0) return;
      dispatch(encoder, resources, [x, y, z]);
    },
    clearBindGroups() {
      bindGroups.clear();
    },
    destroy() {
      // The pipeline is shared and cached for the device; only this pass's bind groups are its own.
      bindGroups.clear();
    },
  };
}
