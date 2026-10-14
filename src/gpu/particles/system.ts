import { noProfiler, type PassProfiler } from '../profiler.ts';
import { createComputePass, type ComputePass } from './compute.ts';
import { createRandom, sampleEmitter, type Emitter } from './emitter.ts';
import type { ParticleSet } from './types.ts';

// The particle store: owns the position buffer, the velocity buffer and any extra per-particle arrays a
// solver wants, in the layout `ParticleSet` promises, and everything that is not physics:
//
//  - spawning: `fill` from a function, `set` from typed arrays, `emit` from an Emitter description (this
//    frame's new particles, optionally recycling the oldest), `setCount` to grow or shrink;
//  - a `ParticleSet` view (`system.particles`) that a renderer reads, always pointing at the current buffers;
//  - double buffering and REORDERING: every attribute can have a back buffer. `permute` gathers all of
//    them through one index list and swaps, and `swap(name)` flips one for a pass that cannot update in
//    place. Which GPUBuffer is "current" changes when you do that, so read `system.positions` at encode
//    time and cache bind groups per buffer (createComputePass does);
//  - capacity changes (`resize`), which keep the contents and bump `generation`.
//
// "Dead" particles. The store has no free list: removal is done by shrinking `count`, or by letting the
// integrator respawn particles whose life (`velocities.w`, seconds) ran out. A zero-filled slot has life 0,
// so a system that grows with `setCount` and respawns brings new particles in by itself. See integrator.ts.
//
// All attributes are arrays of f32 vectors with 1, 2 or 4 components. Positions and velocities are 4 (xyz plus
// a scalar, as in types.ts). An extra of 1 component costs 4 bytes per particle, not 16.

export type AttributeComponents = 1 | 2 | 4;

export interface ParticleSystemOptions {
  device: GPUDevice;
  /** Most particles the buffers can hold. Grow it later with `resize`. */
  capacity: number;
  /** World-space radius of a particle (rendering size; solvers also use it as the particle spacing hint). */
  radius: number;
  /** 2 keeps z at 0 everywhere (emitters, integrator, grid); 3 is the default. */
  dimension?: 2 | 3;
  /** Allocate the velocity buffer. Default true. */
  velocities?: boolean;
  /**
   * Extra per-particle arrays, by name -> components. They move with the particles when the system is
   * reordered, which is the reason to keep per-particle state here and not in loose buffers.
   */
  extras?: Readonly<Record<string, AttributeComponents>>;
  profiler?: PassProfiler;
  label?: string;
}

export interface ParticleInit {
  /** xyz in world units. */
  position: [number, number, number];
  /** The per-particle scalar, `positions.w`. */
  scalar: number;
  velocity: [number, number, number];
  /** `velocities.w`: remaining life in seconds when the integrator respawns, otherwise free. */
  velocityW: number;
}

export interface EmitOptions {
  /** When the system is full, overwrite the oldest particles instead of stopping. Default false. */
  ring?: boolean;
  /** Random source; default a generator seeded from the system's running emit counter. */
  random?: () => number;
}

export interface ParticleSystem {
  readonly device: GPUDevice;
  readonly dimension: 2 | 3;
  /** The renderer-facing view. A live object: its buffers and `count` always match the system. */
  readonly particles: ParticleSet;
  readonly capacity: number;
  /** Live particles. Change with fill/set/emit/setCount. */
  readonly count: number;
  /** World-space radius, writable. */
  radius: number;
  readonly generation: number;
  /** Current buffers. Read them every time you encode; they change on `permute` and `swap`. */
  readonly positions: GPUBuffer;
  readonly velocities: GPUBuffer | undefined;
  /** Names of all attributes: 'position', 'velocity' (if any), then the extras. */
  readonly attributeNames: readonly string[];
  attribute(name: string): GPUBuffer;
  /** The other buffer of the attribute, allocated on first use; write it, then `swap(name)`. */
  back(name: string): GPUBuffer;
  /** Exchanges an attribute's current and back buffer. */
  swap(name: string): void;
  /** Number of f32 per particle of an attribute. */
  components(name: string): AttributeComponents;

  /** Replaces the contents with `count` particles from `init`, which fills the object it is given. */
  fill(count: number, init: (index: number, particle: ParticleInit) => void): void;
  /**
   * Replaces the contents from typed arrays of vec4 rows. `positions.length / 4` particles. Extras not
   * mentioned are zeroed.
   */
  set(data: {
    positions: Float32Array;
    velocities?: Float32Array;
    extras?: Readonly<Record<string, Float32Array>>;
  }): void;
  /**
   * Appends up to `n` particles sampled from the emitter (CPU side), or recycles the oldest when `ring`.
   * Returns how many were written.
   */
  emit(emitter: Emitter, n: number, options?: EmitOptions): number;
  /** Sets the live count. Growing zero-fills the new slots (life 0, so a respawning integrator takes them). */
  setCount(count: number): void;
  /** Changes the capacity, keeping the first min(count, capacity) particles; bumps `generation`. */
  resize(capacity: number): void;

  /**
   * Reorders every attribute: afterwards slot k holds what slot `sorted[k]` held. `sorted` is a buffer of
   * `count` u32 that is a permutation of 0..count-1 (the neighbour grid makes one). All attributes move in
   * one call so none can be left behind.
   */
  permute(encoder: GPUCommandEncoder, sorted: GPUBuffer, count?: number): void;

  /** Copies `count` rows of an attribute back to the CPU. For tests and tools: it waits for the GPU. */
  read(name?: string, count?: number): Promise<Float32Array>;
  destroy(): void;
}

const USAGE =
  GPUBufferUsage.STORAGE |
  GPUBufferUsage.VERTEX |
  GPUBufferUsage.COPY_SRC |
  GPUBufferUsage.COPY_DST;

/** Attributes gathered per pass: one binding for the permutation plus a read and a write each. */
const GATHER_PER_PASS = 3;

interface Attribute {
  name: string;
  components: AttributeComponents;
  front: GPUBuffer;
  back: GPUBuffer | undefined;
}

const WGSL_TYPE: Record<AttributeComponents, string> = { 1: 'f32', 2: 'vec2f', 4: 'vec4f' };

function gatherCode(components: readonly AttributeComponents[]): string {
  const declarations = components
    .map((c, i) => {
      const type = WGSL_TYPE[c];
      return (
        `@group(0) @binding(${1 + i * 2}) var<storage, read> src${i}: array<${type}>;\n` +
        `@group(0) @binding(${2 + i * 2}) var<storage, read_write> dst${i}: array<${type}>;`
      );
    })
    .join('\n');
  const body = components.map((_, i) => `  dst${i}[k] = src${i}[j];`).join('\n');
  return `// Generated by system.ts: slot k takes what slot sorted[k] held, for ${components.length} attribute(s).
@group(0) @binding(0) var<storage, read> sorted: array<u32>;
${declarations}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3u) {
  let k = id.x;
  if (k >= arrayLength(&sorted)) { return; }
  let j = sorted[k];
${body}
}
`;
}

export function createParticleSystem(options: ParticleSystemOptions): ParticleSystem {
  const {
    device,
    dimension = 3,
    velocities: hasVelocities = true,
    profiler = noProfiler,
    label = 'particles',
  } = options;
  let capacity = Math.floor(options.capacity);
  if (!(capacity >= 1))
    throw new RangeError(`capacity must be at least 1, got ${options.capacity}`);
  checkSize(capacity);

  function checkSize(n: number): void {
    const bytes = n * 16;
    if (bytes > device.limits.maxStorageBufferBindingSize || bytes > device.limits.maxBufferSize) {
      throw new RangeError(
        `${label}: ${n} particles need ${bytes} bytes per attribute, over the device limit of ${Math.min(device.limits.maxStorageBufferBindingSize, device.limits.maxBufferSize)}`,
      );
    }
  }

  const attributes: Attribute[] = [];
  const byName = new Map<string, Attribute>();
  const define = (name: string, components: AttributeComponents) => {
    const attribute: Attribute = { name, components, front: undefined!, back: undefined };
    attributes.push(attribute);
    byName.set(name, attribute);
  };
  define('position', 4);
  if (hasVelocities) define('velocity', 4);
  for (const [name, components] of Object.entries(options.extras ?? {})) {
    if (byName.has(name)) throw new TypeError(`${label}: duplicate attribute "${name}"`);
    if (![1, 2, 4].includes(components)) {
      throw new TypeError(`${label}: attribute "${name}" must have 1, 2 or 4 components`);
    }
    define(name, components);
  }

  const allocate = (attribute: Attribute, suffix = ''): GPUBuffer =>
    device.createBuffer({
      label: `${label} ${attribute.name}${suffix}`,
      size: capacity * attribute.components * 4,
      usage: USAGE,
    });
  for (const attribute of attributes) attribute.front = allocate(attribute);

  let count = 0;
  let generation = 0;
  let radius = options.radius;
  let cursor = 0;
  let emitted = 0;
  let destroyed = false;
  let zeros = new Uint8Array(0);
  let scratchPositions = new Float32Array(0);
  let scratchVelocities = new Float32Array(0);

  const attribute = (name: string): Attribute => {
    const found = byName.get(name);
    if (!found) throw new TypeError(`${label}: no attribute "${name}"`);
    return found;
  };

  const particles: ParticleSet = {
    get positions() {
      return attributes[0]!.front;
    },
    get velocities() {
      return hasVelocities ? byName.get('velocity')!.front : undefined;
    },
    get count() {
      return count;
    },
    get capacity() {
      return capacity;
    },
    get radius() {
      return radius;
    },
    get generation() {
      return generation;
    },
  };

  function zeroRange(from: number, to: number): void {
    if (to <= from) return;
    for (const a of attributes) {
      const stride = a.components * 4;
      const bytes = (to - from) * stride;
      if (zeros.length < bytes) zeros = new Uint8Array(bytes);
      device.queue.writeBuffer(a.front, from * stride, zeros, 0, bytes);
    }
  }

  function setCountClamped(next: number): void {
    next = Math.max(0, Math.min(capacity, Math.floor(next)));
    zeroRange(count, next);
    count = next;
    cursor = count % capacity;
  }

  function writeRows(name: string, firstRow: number, data: Float32Array): void {
    const a = attribute(name);
    device.queue.writeBuffer(a.front, firstRow * a.components * 4, data);
  }

  const gatherPasses = new Map<string, ComputePass>();
  function gatherPass(components: readonly AttributeComponents[]): ComputePass {
    const key = components.join(',');
    let pass = gatherPasses.get(key);
    if (!pass) {
      pass = createComputePass(device, {
        label: `particles/gather-${key.replaceAll(',', '')}.wgsl`,
        code: gatherCode(components),
        profiler,
        timestamp: `${label} reorder`,
      });
      gatherPasses.set(key, pass);
    }
    return pass;
  }

  const system: ParticleSystem = {
    device,
    dimension,
    particles,
    get capacity() {
      return capacity;
    },
    get count() {
      return count;
    },
    get radius() {
      return radius;
    },
    set radius(value: number) {
      radius = value;
    },
    get generation() {
      return generation;
    },
    get positions() {
      return attributes[0]!.front;
    },
    get velocities() {
      return hasVelocities ? byName.get('velocity')!.front : undefined;
    },
    get attributeNames() {
      return attributes.map((a) => a.name);
    },
    attribute: (name) => attribute(name).front,
    components: (name) => attribute(name).components,
    back(name) {
      const a = attribute(name);
      a.back ??= allocate(a, ' (back)');
      return a.back;
    },
    swap(name) {
      const a = attribute(name);
      a.back ??= allocate(a, ' (back)');
      [a.front, a.back] = [a.back, a.front];
    },

    fill(n, init) {
      n = Math.max(0, Math.min(capacity, Math.floor(n)));
      const positions = new Float32Array(n * 4);
      const velocities = hasVelocities ? new Float32Array(n * 4) : undefined;
      const p: ParticleInit = { position: [0, 0, 0], scalar: 0, velocity: [0, 0, 0], velocityW: 0 };
      for (let i = 0; i < n; i++) {
        p.position[0] = p.position[1] = p.position[2] = 0;
        p.velocity[0] = p.velocity[1] = p.velocity[2] = 0;
        p.scalar = 0;
        p.velocityW = 0;
        init(i, p);
        positions.set(p.position, i * 4);
        positions[i * 4 + 3] = p.scalar;
        if (velocities) {
          velocities.set(p.velocity, i * 4);
          velocities[i * 4 + 3] = p.velocityW;
        }
      }
      if (dimension === 2) for (let i = 0; i < n; i++) positions[i * 4 + 2] = 0;
      system.set({ positions, velocities });
    },

    set({ positions, velocities, extras }) {
      if (positions.length % 4 !== 0)
        throw new RangeError('positions must be vec4 rows (length % 4)');
      const n = Math.min(capacity, positions.length / 4);
      writeRows('position', 0, positions.subarray(0, n * 4));
      if (hasVelocities) {
        if (velocities) writeRows('velocity', 0, velocities.subarray(0, n * 4));
        else writeRows('velocity', 0, new Float32Array(n * 4));
      }
      for (const a of attributes.slice(hasVelocities ? 2 : 1)) {
        const data = extras?.[a.name];
        writeRows(
          a.name,
          0,
          data ? data.subarray(0, n * a.components) : new Float32Array(n * a.components),
        );
      }
      count = n;
      cursor = count % capacity;
    },

    emit(emitter, n, emitOptions = {}) {
      n = Math.floor(n);
      const { ring = false } = emitOptions;
      if (!ring) n = Math.min(n, capacity - count);
      else n = Math.min(n, capacity);
      if (n <= 0) return 0;
      const random = emitOptions.random ?? createRandom(0x51ed + emitted++);

      if (scratchPositions.length < n * 4) scratchPositions = new Float32Array(n * 4);
      if (scratchVelocities.length < n * 4) scratchVelocities = new Float32Array(n * 4);
      const positions = scratchPositions;
      const velocities = hasVelocities ? scratchVelocities : undefined;
      sampleEmitter(emitter, dimension, random, n, positions, velocities);

      // Where the new rows go: after the live ones, wrapping to the oldest when recycling.
      const start = ring ? cursor : count;
      const first = Math.min(n, capacity - start);
      writeRows('position', start, positions.subarray(0, first * 4));
      if (velocities) writeRows('velocity', start, velocities.subarray(0, first * 4));
      if (first < n) {
        writeRows('position', 0, positions.subarray(first * 4, n * 4));
        if (velocities) writeRows('velocity', 0, velocities.subarray(first * 4, n * 4));
      }
      // Extras of a recycled slot restart from zero, like a new particle.
      for (const a of attributes.slice(hasVelocities ? 2 : 1)) {
        const bytes = n * a.components * 4;
        if (zeros.length < bytes) zeros = new Uint8Array(bytes);
        const firstBytes = first * a.components * 4;
        device.queue.writeBuffer(a.front, start * a.components * 4, zeros, 0, firstBytes);
        if (first < n) device.queue.writeBuffer(a.front, 0, zeros, 0, bytes - firstBytes);
      }
      count = Math.min(capacity, count + n);
      cursor = (start + n) % capacity;
      return n;
    },

    setCount: setCountClamped,

    resize(next) {
      next = Math.floor(next);
      if (!(next >= 1)) throw new RangeError(`capacity must be at least 1, got ${next}`);
      if (next === capacity) return;
      checkSize(next);
      const keep = Math.min(count, next);
      const old = attributes.map((a) => ({ front: a.front, back: a.back }));
      capacity = next;
      const encoder = device.createCommandEncoder({ label: `${label} resize` });
      attributes.forEach((a, index) => {
        a.front = allocate(a);
        a.back = undefined;
        if (keep > 0) {
          encoder.copyBufferToBuffer(old[index]!.front, 0, a.front, 0, keep * a.components * 4);
        }
      });
      device.queue.submit([encoder.finish()]);
      for (const o of old) {
        o.front.destroy();
        o.back?.destroy();
      }
      count = keep;
      cursor = count % capacity;
      generation++;
    },

    permute(encoder, sorted, n = count) {
      n = Math.min(n, count);
      if (n <= 0) return;
      for (let first = 0; first < attributes.length; first += GATHER_PER_PASS) {
        const group = attributes.slice(first, first + GATHER_PER_PASS);
        const pass = gatherPass(group.map((a) => a.components));
        const resources: GPUBufferBinding[] = [{ buffer: sorted, size: n * 4 }];
        for (const a of group) {
          a.back ??= allocate(a, ' (back)');
          resources.push({ buffer: a.front }, { buffer: a.back });
        }
        pass.runCount(encoder, resources, n);
      }
      // Only now, after every gather is recorded, does each attribute change which buffer is current.
      for (const a of attributes) [a.front, a.back] = [a.back!, a.front];
    },

    async read(name = 'position', n = count) {
      const a = attribute(name);
      n = Math.min(n, capacity);
      const bytes = n * a.components * 4;
      const readback = device.createBuffer({
        label: `${label} readback`,
        size: Math.max(4, bytes),
        usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
      });
      const encoder = device.createCommandEncoder();
      if (bytes > 0) encoder.copyBufferToBuffer(a.front, 0, readback, 0, bytes);
      device.queue.submit([encoder.finish()]);
      await readback.mapAsync(GPUMapMode.READ);
      const data = new Float32Array(readback.getMappedRange().slice(0, bytes));
      readback.unmap();
      readback.destroy();
      return data;
    },

    destroy() {
      if (destroyed) return;
      destroyed = true;
      for (const a of attributes) {
        a.front.destroy();
        a.back?.destroy();
      }
      for (const pass of gatherPasses.values()) pass.destroy();
    },
  };
  return system;
}
