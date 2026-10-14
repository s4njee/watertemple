import { createFixedStepper } from '../loop.ts';
import { noProfiler, type PassProfiler } from '../profiler.ts';
import { createUniformBuffer, defineUniformStruct } from '../uniforms.ts';
import { createComputePass, type ComputePass } from './compute.ts';
import { EMITTER_SHAPE_CODE, resolveEmitter, type Emitter, type Vec3 } from './emitter.ts';
import integratorSource from './integrator.wgsl';
import type { ParticleSystem } from './system.ts';

// The ready-made integrator for particles that do not interact: fountains, spray, snow, dust, mist, and the
// starting point of every solver that only needs to move particles between its own steps.
//
// SCHEME. Semi-implicit Euler, per sub-step of the fixed step dt:
//     v += (gravity + force_field(p, v, t)) dt;   v *= exp(-damping dt);   p += v dt
// then the box: a particle that left it is put back by reflecting the overshoot and the normal velocity is
// scaled by `restitution` (0 sticks to the wall, 1 is elastic); contact scales the whole velocity by
// `1 - friction`. Velocity is updated before position, which makes the scheme symplectic: for constant
// acceleration it reproduces the discrete ballistic solution
//     v_n = v0 + n a dt,   p_n = p0 + n v0 dt + a dt^2 n (n + 1) / 2
// exactly (tests/particles.test.ts compares against it), and is first-order close to the continuous one.
//
// FIXED TIMESTEP. `advance(encoder, delta)` accumulates real time and runs whole steps of `fixedStep` seconds, at
// most `maxSteps` per frame (createFixedStepper): at 60 Hz or 144 Hz the particles move the same distance per
// second, and a stalled tab skips the backlog instead of replaying it. All sub-steps of a frame run in ONE
// dispatch (the particles are independent), so a frame costs one pass however many steps it takes. The default
// step is 1/240 s, small enough that the leftover fraction of a frame (not simulated, up to one step) is invisible
// and there is no judder when the display rate does not divide the step.
//
// FORCE FIELD. Either a WGSL function chunk,
//     fn force_field(p: vec3f, v: vec3f, time: f32, index: u32) -> vec3f   // an acceleration
// (it may `#include` shared chunks, such as curl.wgsl, and read `params.user0` .. `params.user3`, four vec4f
// the host sets from `integrator.params.user`), or a texture: a 2D or 3D `rgba16float` texture of accelerations
// (xyz) sampled bilinearly over the world rectangle `params.fieldMin` .. `fieldMax` (texel row 0 at min.y),
// scaled by `fieldStrength`. The field is compiled in when the integrator is made. It is the costly part of a
// step, so it is evaluated every `fieldEvery` sub-steps and the acceleration reused in between.
//
// RESPAWN. With `params.respawn` set, `velocities.w` is the particle's remaining life in seconds: it counts down
// by dt, and when it is not positive the particle is re-seeded from the emitter inside the shader (position,
// velocity, and a new life) with a hash of its index and the frame. Zero-filled particles (new slots after
// `setCount`) have no life left, so they enter the stream on the first step; give the emitter a `lifetime`. With
// `floor: 'kill'` a particle that touches the box floor (the bounds' min y) also ends and respawns at once.
//
// ONE SUBMIT. Each `advance`/`step` takes the next of four uniform buffers, so up to four runs can be encoded
// before a submit (a warm-up and a frame, say); the fifth would overwrite the first's parameters.
//
// COST. One read and one write of positions and velocities (64 bytes per particle) plus the field; memory
// bound unless the field is heavy. Binds 2 storage buffers (3 with a texture field: texture and sampler too).

/** Runs of one integrator that can share a submit (each takes the next of this many uniform buffers). */
const UNIFORM_RING = 4;

export const IntegratorUniform = defineUniformStruct('IntegratorParams', {
  gravity: 'vec3f',
  dt: 'f32',
  bounds_min: 'vec3f',
  damping: 'f32',
  bounds_max: 'vec3f',
  restitution: 'f32',
  emit_center: 'vec3f',
  emit_shape: 'u32',
  emit_extent: 'vec3f',
  emit_speed: 'f32',
  emit_direction: 'vec3f',
  emit_spread: 'f32',
  field_min: 'vec3f',
  field_strength: 'f32',
  field_max: 'vec3f',
  time: 'f32',
  emit_jitter: 'f32',
  life_min: 'f32',
  life_max: 'f32',
  friction: 'f32',
  margin: 'f32',
  steps: 'u32',
  field_every: 'u32',
  dimension: 'u32',
  respawn: 'u32',
  kill_floor: 'u32',
  write_speed: 'u32',
  use_bounds: 'u32',
  count: 'u32',
  seed: 'u32',
  user0: 'vec4f',
  user1: 'vec4f',
  user2: 'vec4f',
  user3: 'vec4f',
});

const NO_FIELD = `fn force_field(p: vec3f, v: vec3f, time: f32, index: u32) -> vec3f {
  return vec3f(0.0);
}
`;

const TEXTURE_FIELD_2D = `@group(0) @binding(3) var field_texture: texture_2d<f32>;
@group(0) @binding(4) var field_sampler: sampler;
fn force_field(p: vec3f, v: vec3f, time: f32, index: u32) -> vec3f {
  let uv = (p.xy - params.field_min.xy) / (params.field_max.xy - params.field_min.xy);
  return textureSampleLevel(field_texture, field_sampler, uv, 0.0).xyz * params.field_strength;
}
`;

const TEXTURE_FIELD_3D = `@group(0) @binding(3) var field_texture: texture_3d<f32>;
@group(0) @binding(4) var field_sampler: sampler;
fn force_field(p: vec3f, v: vec3f, time: f32, index: u32) -> vec3f {
  let uvw = (p - params.field_min) / (params.field_max - params.field_min);
  return textureSampleLevel(field_texture, field_sampler, uvw, 0.0).xyz * params.field_strength;
}
`;

type Vec4 = [number, number, number, number];

export type ForceField =
  | {
      /** WGSL defining `fn force_field(p: vec3f, v: vec3f, time: f32, index: u32) -> vec3f`. */
      wgsl: string | { code: string; chunks: Readonly<Record<string, string>> };
    }
  | {
      /** A filterable acceleration texture (rgba16float works everywhere). */
      texture: GPUTextureView;
      dimension: '2d' | '3d';
    };

export interface Box {
  min: Vec3;
  max: Vec3;
}

/** Live parameters: change them any time; they are uploaded every `advance`. */
export interface IntegratorParams {
  gravity: [number, number, number];
  /** Linear drag, 1 / s: the velocity falls as exp(-damping t). */
  damping: number;
  /** Walls; null for none. */
  bounds: Box | null;
  /** Normal velocity kept after a wall hit, 0..1. */
  restitution: number;
  /** Fraction of the whole velocity lost on contact, 0..1. */
  friction: number;
  /** Distance kept between a particle's centre and the wall; default the system's radius. */
  margin: number | undefined;
  /** Seconds. */
  fixedStep: number;
  /** Sub-steps per frame, at most. */
  maxSteps: number;
  /** Evaluate the force field every this many sub-steps (default 1). */
  fieldEvery: number;
  /** Texture fields: the world box the texture covers, and the multiplier. */
  fieldMin: [number, number, number];
  fieldMax: [number, number, number];
  fieldStrength: number;
  /** `speed` writes |v| into positions.w every step (what a palette usually wants); `keep` leaves it. */
  scalar: 'speed' | 'keep';
  /** Re-seed particles whose life ran out. Needs an emitter with `lifetime`. null to disable. */
  respawn: (Emitter & { floor?: 'bounce' | 'kill' }) | null;
  /** Four vec4f the force-field WGSL reads as `params.user0` .. `user3`. */
  user: [Vec4, Vec4, Vec4, Vec4];
}

export interface IntegratorOptions extends Partial<Omit<IntegratorParams, 'user'>> {
  system: ParticleSystem;
  field?: ForceField;
  profiler?: PassProfiler;
  label?: string;
}

export interface Integrator {
  readonly params: IntegratorParams;
  /** Simulation seconds integrated so far. */
  readonly time: number;
  /** Runs the whole steps that `delta` seconds add up to; returns how many (0 on a short frame). */
  advance(encoder: GPUCommandEncoder, delta: number): number;
  /** Runs exactly `steps` steps of `dt` seconds (default `params.fixedStep`), regardless of the clock. */
  step(encoder: GPUCommandEncoder, steps?: number, dt?: number): void;
  setFieldTexture(view: GPUTextureView): void;
  destroy(): void;
}

export function createIntegrator(options: IntegratorOptions): Integrator {
  const { system, field, profiler = noProfiler, label = 'particle integrator' } = options;
  const { device, dimension } = system;
  if (!system.velocities) throw new TypeError(`${label}: the system needs velocities`);

  const params: IntegratorParams = {
    gravity: options.gravity ? [...options.gravity] : [0, -9.81, 0],
    damping: options.damping ?? 0,
    bounds: options.bounds ?? null,
    restitution: options.restitution ?? 0.5,
    friction: options.friction ?? 0,
    margin: options.margin,
    fixedStep: options.fixedStep ?? 1 / 240,
    maxSteps: options.maxSteps ?? 8,
    fieldEvery: options.fieldEvery ?? 1,
    fieldMin: options.fieldMin ?? [0, 0, 0],
    fieldMax: options.fieldMax ?? [1, 1, 1],
    fieldStrength: options.fieldStrength ?? 1,
    scalar: options.scalar ?? 'keep',
    respawn: options.respawn ?? null,
    user: [
      [0, 0, 0, 0],
      [0, 0, 0, 0],
      [0, 0, 0, 0],
      [0, 0, 0, 0],
    ],
  };

  // A ring of uniform buffers: `queue.writeBuffer` lands before the whole command buffer, so two runs in one
  // submit (a warm-up and a frame) would otherwise both see the later run's values.
  const uniformRing = Array.from({ length: UNIFORM_RING }, (_, index) =>
    createUniformBuffer(device, IntegratorUniform, `${label} params ${index}`),
  );
  let ringIndex = 0;
  const fieldWgsl = !field
    ? NO_FIELD
    : 'wgsl' in field
      ? typeof field.wgsl === 'string'
        ? field.wgsl
        : field.wgsl.code
      : field.dimension === '2d'
        ? TEXTURE_FIELD_2D
        : TEXTURE_FIELD_3D;
  const fieldChunks =
    field && 'wgsl' in field && typeof field.wgsl !== 'string' ? field.wgsl.chunks : {};
  let fieldView: GPUTextureView | undefined =
    field && 'texture' in field ? field.texture : undefined;
  const sampler = fieldView
    ? device.createSampler({
        label: `${label} field sampler`,
        magFilter: 'linear',
        minFilter: 'linear',
      })
    : undefined;

  const pass: ComputePass = createComputePass(device, {
    label: 'particles/integrator.wgsl',
    code: integratorSource.code,
    chunks: {
      ...integratorSource.chunks,
      ...fieldChunks,
      'integrator_params.wgsl': IntegratorUniform.wgsl,
      'force_field.wgsl': fieldWgsl,
    },
    profiler,
    timestamp: `${label}`,
  });

  const stepper = () => createFixedStepper(params.fixedStep, { maxSteps: params.maxSteps });
  let fixed = stepper();
  let fixedKey = `${params.fixedStep}|${params.maxSteps}`;
  let time = 0;
  let frame = 0;

  function upload(steps: number, dt: number): GPUBuffer {
    const uniforms = uniformRing[ringIndex]!;
    ringIndex = (ringIndex + 1) % UNIFORM_RING;
    const respawn = params.respawn ? resolveEmitter(params.respawn) : undefined;
    const box = params.bounds;
    const margin = params.margin ?? system.radius;
    uniforms.set({
      gravity: params.gravity,
      dt,
      bounds_min: box?.min ?? [0, 0, 0],
      damping: params.damping,
      bounds_max: box?.max ?? [0, 0, 0],
      restitution: params.restitution,
      emit_center: respawn?.center ?? [0, 0, 0],
      emit_shape: respawn ? EMITTER_SHAPE_CODE[respawn.shape] : 0,
      emit_extent: respawn?.extent ?? [0, 0, 0],
      emit_speed: respawn?.speed ?? 0,
      emit_direction: respawn?.direction ?? [0, 1, 0],
      emit_spread: respawn?.spread ?? 0,
      field_min: params.fieldMin,
      field_strength: params.fieldStrength,
      field_max: params.fieldMax,
      time,
      emit_jitter: respawn?.speedJitter ?? 0,
      life_min: respawn?.lifetime[0] ?? 0,
      life_max: respawn?.lifetime[1] ?? 0,
      friction: params.friction,
      margin,
      steps,
      field_every: Math.max(1, Math.floor(params.fieldEvery)),
      dimension,
      respawn: respawn ? 1 : 0,
      kill_floor: params.respawn?.floor === 'kill' ? 1 : 0,
      write_speed: params.scalar === 'speed' ? 1 : 0,
      use_bounds: box ? 1 : 0,
      count: system.count,
      seed: frame >>> 0,
      user0: params.user[0],
      user1: params.user[1],
      user2: params.user[2],
      user3: params.user[3],
    });
    uniforms.upload();
    return uniforms.buffer;
  }

  function run(encoder: GPUCommandEncoder, steps: number, dt: number): void {
    if (system.count <= 0 || steps <= 0) return;
    const uniformBuffer = upload(steps, dt);
    pass.runCount(
      encoder,
      [uniformBuffer, system.positions, system.velocities!, fieldView, sampler],
      system.count,
    );
    time += steps * dt;
    frame++;
  }

  return {
    params,
    get time() {
      return time;
    },
    advance(encoder, delta) {
      const key = `${params.fixedStep}|${params.maxSteps}`;
      if (key !== fixedKey) {
        fixed = stepper();
        fixedKey = key;
      }
      const { steps } = fixed.advance(delta);
      run(encoder, steps, params.fixedStep);
      return steps;
    },
    step(encoder, steps = 1, dt = params.fixedStep) {
      run(encoder, steps, dt);
    },
    setFieldTexture(view) {
      if (!field || !('texture' in field))
        throw new TypeError(`${label}: not made with a texture field`);
      fieldView = view;
    },
    destroy() {
      for (const uniforms of uniformRing) uniforms.destroy();
      pass.destroy();
    },
  };
}
