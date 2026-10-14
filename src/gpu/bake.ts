import { createFullscreenPipeline, renderFullscreen } from './fullscreen.ts';
import { generateMipmaps, mipLevelCount } from './mipmap.ts';
import { createUniformBuffer, defineUniformStruct, uniformArray } from './uniforms.ts';

/*
 * Procedural textures baked once at startup. A site cannot ship noise per pixel (the `hello` effect
 * shows what per-pixel FBM costs), and this project ships no image assets, so the textures that real
 * water uses are rendered here instead: one fullscreen pass into a texture, then a mip chain.
 * Everything is a few milliseconds of GPU time at startup and nothing per frame.
 *
 *   createWaterNormalMap   seamlessly tileable tangent-space ripple normals     (normals.wgsl)
 *   createFlowMap          divergence-free river current with rocks and banks    (flow.wgsl)
 *   createWaterSampler / createFlowSampler   the samplers these textures want
 */

// --- Water normal map ----------------------------------------------------------------------------

export interface WaterNormalMapOptions {
  /** Texels per side; a power of two from 16 to 2048. Default 512. */
  size?: number;
  /** Different seeds give different, uncorrelated patterns. Default 1. */
  seed?: number;
  /** Noise cells across the tile at the lowest octave (integer, 1..64). Default 6. */
  period?: number;
  /** Noise octaves, each doubling the frequency (1..8). Default 5. */
  octaves?: number;
  /** Amplitude ratio between octaves, 0..1. 0.5 gives equal slope per octave. Default 0.5. */
  gain?: number;
  /**
   * Slope multiplier. At 1 the RMS slope of the default pattern is about 0.25 (|xy| of the stored
   * normal), a gentle ripple; scale it later in the shader with `normal_scale` too. Default 1.
   */
  strength?: number;
  /** 0 = rounded smooth bumps, 1 = sharp crest lines on every octave. Default 0.35. */
  choppiness?: number;
  /** Also store the (mip-averaged) height in the blue channel, for foam or parallax. Default false. */
  height?: boolean;
  label?: string;
}

export interface WaterNormalMap {
  readonly texture: GPUTexture;
  /** All mip levels. */
  readonly view: GPUTextureView;
  /** `rg16float`, or `rgba16float` when `height` was requested. */
  readonly format: GPUTextureFormat;
  readonly size: number;
  readonly mipLevelCount: number;
  /** Estimated GPU memory including the mip chain. */
  readonly bytes: number;
  destroy(): void;
}

// The struct is declared once here and injected into the bake shader, so the CPU packing and the
// WGSL cannot drift apart.
const WaterBake = defineUniformStruct('WaterBake', {
  period: 'i32',
  octaves: 'i32',
  gain: 'f32',
  choppiness: 'f32',
  strength: 'f32',
  seed_z: 'f32',
  texel: 'f32',
  with_height: 'u32',
});

const WATER_BAKE_WGSL = /* wgsl */ `
#include "fullscreen.wgsl"
#include "noise.wgsl"
#include "normals.wgsl"
#include "params.wgsl"

@group(0) @binding(0) var<uniform> bake: WaterBake;

// Converts the unit-less height gradient (per cell of the lowest octave) to a slope. Chosen so that
// strength 1 gives an RMS slope of about 0.25 for the default octaves and gain.
const SLOPE_GAIN: f32 = 0.2;

// Height in roughly [-1, 1]: a fractal sum of seamlessly tileable 3D gradient noise. The third axis is
// the seed (a slice through the noise volume), because shifting a periodic 2D pattern only translates it.
fn height(uv: vec2f) -> f32 {
  var sum = 0.0;
  var norm = 0.0;
  var amp = 1.0;
  for (var i = 0; i < bake.octaves; i++) {
    let per = bake.period * (1 << u32(i));
    let p = uv * f32(per);
    let n = gnoise_3d_tile(vec3f(p, bake.seed_z + 3.7 * f32(i)), vec3i(per, per, 1 << 20));
    // 1 - 2|n| puts a sharp crest line on every zero crossing of the noise.
    let crest = 1.0 - 2.0 * abs(n);
    sum += amp * mix(n, crest, bake.choppiness);
    norm += amp;
    amp *= bake.gain;
  }
  return sum / max(norm, 1e-6);
}

@fragment fn fs(in: FullscreenVarying) -> @location(0) vec4f {
  // uv is the texel centre; the neighbours are one texel away, and the noise is periodic, so the
  // finite difference across the tile edge sees the texels on the other side: no seam.
  let uv = in.uv;
  let dx = vec2f(bake.texel, 0.0);
  let dy = vec2f(0.0, bake.texel);
  let grad_uv = vec2f(
    height(uv + dx) - height(uv - dx),
    height(uv + dy) - height(uv - dy),
  ) / (2.0 * bake.texel);
  let grad_cell = grad_uv / f32(bake.period);
  let n = normal_from_gradient(grad_cell, SLOPE_GAIN * bake.strength);
  let h = select(0.0, height(uv), bake.with_height != 0u);
  return vec4f(n.xy, h, 1.0);
}
`;

function isPowerOfTwo(value: number): boolean {
  return Number.isInteger(value) && value > 0 && (value & (value - 1)) === 0;
}

function clampInt(value: number, lo: number, hi: number): number {
  return Math.min(hi, Math.max(lo, Math.round(value)));
}

/**
 * Bakes a seamlessly tileable water-ripple normal map with a full mip chain.
 *
 * Encoding: tangent-space normal, **signed float, no bias**. `r` = n.x, `g` = n.y (the negated height
 * slopes, in the texture's u and v directions; v grows downwards), z is NOT stored: rebuild it with
 * `normal_decode_signed` from normals.wgsl (z = sqrt(1 - x^2 - y^2)). With `height: true` the format
 * is rgba16float and `b` holds the height (about [-1, 1]); otherwise rg16float. Both are renderable
 * and filterable in WebGPU core, so the mips are made with ordinary linear blits.
 *
 * Mips average xy, which averages the slopes: the map flattens with distance (shorter xy, z -> 1)
 * rather than turning into noise. The cost is that roughness is lost with it; fade strength or add
 * roughness in the shader if distant glints matter. Sample with `createWaterSampler`.
 */
export function createWaterNormalMap(
  device: GPUDevice,
  options: WaterNormalMapOptions = {},
): WaterNormalMap {
  const size = options.size ?? 512;
  if (!isPowerOfTwo(size) || size < 16 || size > 2048) {
    throw new RangeError(
      `createWaterNormalMap: size must be a power of two in 16..2048, got ${size}`,
    );
  }
  const period = clampInt(options.period ?? 6, 1, 64);
  const octaves = clampInt(options.octaves ?? 5, 1, 8);
  const withHeight = options.height ?? false;
  const format: GPUTextureFormat = withHeight ? 'rgba16float' : 'rg16float';
  const levels = mipLevelCount(size, size);
  const label = options.label ?? 'water normal map';

  const texture = device.createTexture({
    label,
    size: [size, size],
    format,
    mipLevelCount: levels,
    // COPY_SRC lets tests and tools read it back.
    usage:
      GPUTextureUsage.TEXTURE_BINDING |
      GPUTextureUsage.RENDER_ATTACHMENT |
      GPUTextureUsage.COPY_SRC,
  });

  const uniforms = createUniformBuffer(device, WaterBake, `${label} params`);
  const seed = options.seed ?? 1;
  uniforms.set({
    period,
    octaves,
    gain: Math.min(1, Math.max(0, options.gain ?? 0.5)),
    choppiness: Math.min(1, Math.max(0, options.choppiness ?? 0.35)),
    strength: Math.max(0, options.strength ?? 1),
    // A slice through the 3D noise; decorrelates seeds that are whole numbers apart.
    seed_z: seed * 13.37,
    texel: 1 / size,
    with_height: withHeight ? 1 : 0,
  });
  uniforms.upload();

  const pipeline = createFullscreenPipeline(device, {
    label: `${label} bake`,
    fragment: {
      label: 'bake/water-normals.wgsl',
      code: WATER_BAKE_WGSL,
      chunks: { 'params.wgsl': WaterBake.wgsl },
    },
    format,
  });
  const bindGroup = device.createBindGroup({
    label: `${label} bake`,
    layout: pipeline.getBindGroupLayout(0),
    entries: [{ binding: 0, resource: { buffer: uniforms.buffer } }],
  });

  const encoder = device.createCommandEncoder({ label: `${label} bake` });
  renderFullscreen(
    encoder,
    texture.createView({ label: `${label} level 0`, baseMipLevel: 0, mipLevelCount: 1 }),
    pipeline,
    [bindGroup],
  );
  generateMipmaps(device, texture, encoder);
  device.queue.submit([encoder.finish()]);
  uniforms.destroy();

  let bytes = 0;
  for (let level = 0; level < levels; level++) {
    const side = Math.max(1, size >> level);
    bytes += side * side * (withHeight ? 8 : 4);
  }
  return {
    texture,
    view: texture.createView({ label: `${label} view` }),
    format,
    size,
    mipLevelCount: levels,
    bytes,
    destroy: () => texture.destroy(),
  };
}

export interface WaterSamplerOptions {
  /** Anisotropic filtering, 1..16. Grazing views of a water plane need it; default 8. */
  anisotropy?: number;
}

/** Trilinear, repeating, anisotropic: what a tiled water normal map should be sampled with. */
export function createWaterSampler(
  device: GPUDevice,
  options: WaterSamplerOptions = {},
): GPUSampler {
  return device.createSampler({
    label: 'water repeat sampler',
    addressModeU: 'repeat',
    addressModeV: 'repeat',
    magFilter: 'linear',
    minFilter: 'linear',
    mipmapFilter: 'linear',
    maxAnisotropy: clampInt(options.anisotropy ?? 8, 1, 16),
  });
}

// --- Flow map ------------------------------------------------------------------------------------

/**
 * The flow map covers a world that is this many times wider than it is tall, and repeats seamlessly
 * along x. World units: the map's height is 1.
 */
export const FLOW_WORLD_ASPECT = 2;

export interface FlowMapOptions {
  /** Texels across (the map is `width x width / 2`). Default 1024. */
  width?: number;
  /** Places the rocks. Default 1. */
  seed?: number;
  /** Rocks in the channel, 0..8. Default 6. */
  rocks?: number;
  /** Strength of the small curl-noise eddies added to the stream function, 0..3. Default 0.6. */
  turbulence?: number;
  label?: string;
}

export interface FlowMap {
  readonly texture: GPUTexture;
  readonly view: GPUTextureView;
  /** Always `rgba8unorm`: filterable and renderable everywhere. */
  readonly format: GPUTextureFormat;
  readonly width: number;
  readonly height: number;
  /** Estimated GPU memory (no mips). */
  readonly bytes: number;
  destroy(): void;
}

const MAX_ROCKS = 8;

const FlowBake = defineUniformStruct('FlowBake', {
  rocks: uniformArray('vec4f', MAX_ROCKS),
  count: 'u32',
  turbulence: 'f32',
  texel: 'f32',
});

const FLOW_BAKE_WGSL = /* wgsl */ `
#include "fullscreen.wgsl"
#include "fbm.wgsl"
#include "flow.wgsl"
#include "math.wgsl"
#include "params.wgsl"

@group(0) @binding(0) var<uniform> bake: FlowBake;

// The world is ${FLOW_WORLD_ASPECT} wide and 1 tall, x periodic. Velocity is v = (dpsi/dy, -dpsi/dx) of
// a stream function psi, so it is divergence-free by construction (up to the finite-difference step).
// Streamlines are the level sets of psi: banks are streamlines (nothing crosses them) and so are the
// rock outlines.
const WORLD_W: f32 = ${FLOW_WORLD_ASPECT}.0;
// Flux of the whole river; the centre speed is U / half-width.
const U_FLOW: f32 = 0.11;

// Meandering centre line and breathing width, both periodic in x with period 2.
fn centre_y(x: f32) -> f32 {
  return 0.5 + 0.07 * sin(PI * x) + 0.025 * sin(TAU * x + 1.3);
}
fn half_width(x: f32) -> f32 {
  return 0.30 + 0.05 * sin(PI * x + 2.1) + 0.02 * sin(TAU * x - 0.7);
}
// -1 at one bank, 0 on the centre line, +1 at the other; unclamped outside the river.
fn across(p: vec2f) -> f32 {
  return (p.y - centre_y(p.x)) / half_width(p.x);
}

// psi = U * F(s) with F(s) = s - s^3/3: the velocity along the channel has the parabolic profile
// F'(s) = 1 - s^2 (fast in the middle, zero at the banks), and narrows -> speeds up through 1 / width.
fn psi_channel(p: vec2f) -> f32 {
  let s = clamp(across(p), -1.0, 1.0);
  return U_FLOW * (s - s * s * s / 3.0);
}

fn channel_velocity(p: vec2f) -> vec2f {
  let e = 1e-3;
  return vec2f(
    psi_channel(p + vec2f(0.0, e)) - psi_channel(p - vec2f(0.0, e)),
    -(psi_channel(p + vec2f(e, 0.0)) - psi_channel(p - vec2f(e, 0.0))),
  ) / (2.0 * e);
}

// Rock i: .x = world x, .y = across-channel position s, .z = radius.
fn rock_centre(i: u32) -> vec2f {
  let x = bake.rocks[i].x;
  return vec2f(x, centre_y(x) + bake.rocks[i].y * half_width(x));
}

// Offset from rock centre c to p, taking the nearer of the periodic images.
fn wrapped(p: vec2f, c: vec2f) -> vec2f {
  var d = p - c;
  d.x -= WORLD_W * round(d.x / WORLD_W);
  return d;
}

fn psi(p: vec2f) -> f32 {
  let s = across(p);
  var total = psi_channel(p);
  // Potential flow around a cylinder of radius R in a uniform stream u: psi = psi_u * (1 - R^2 / r^2).
  // Each rock adds its dipole -R^2 psi_u / r^2, evaluated for the channel's own velocity at the rock.
  // The taper fades the dipoles out toward the banks so no flow leaks across them (flux stays exact);
  // it is 1 over the rocks, which sit at |s| < 0.45.
  let taper = 1.0 - smoothstep(0.7, 1.0, abs(s));
  for (var i = 0u; i < bake.count; i++) {
    let c = rock_centre(i);
    let radius = bake.rocks[i].z;
    let d = wrapped(p, c);
    let u = channel_velocity(c);
    let r2 = max(dot(d, d), radius * radius);
    // The periodic wrap above jumps by a whole period half way round the world, and the dipole does not match
    // across that jump (its u.y * d.x part flips sign), which would draw a seam of spurious velocity
    // there. The window is 1 over and around the rock and 0 well before the jump.
    let window = 1.0 - smoothstep(0.35, 0.9, abs(d.x));
    total -= taper * window * radius * radius * (u.x * d.y - u.y * d.x) / r2;
  }
  // Eddies: curl of a tileable noise potential (Bridson et al.), faded to zero at the banks.
  let env = max(1.0 - s * s, 0.0);
  let q = vec2f(p.x * 4.0, p.y * 4.0);
  total += bake.turbulence * 0.012 * env * env * fbm_2d_tile(q, vec2i(8, 64), 3, 0.5);
  return total;
}

// Distance from p to the nearest rock edge (negative inside a rock); large with no rocks.
fn rock_clearance(p: vec2f) -> f32 {
  var clearance = 1e3;
  for (var i = 0u; i < bake.count; i++) {
    clearance = min(clearance, length(wrapped(p, rock_centre(i))) - bake.rocks[i].z);
  }
  return clearance;
}

// Foam: a ring at each rock, a tapering wake behind it (downstream along the local current), and a
// faint line in the bank shallows.
fn foam_mask(p: vec2f, depth_bank: f32) -> f32 {
  var foam = 0.45 * (1.0 - smoothstep(0.0, 0.14, depth_bank));
  for (var i = 0u; i < bake.count; i++) {
    let c = rock_centre(i);
    let radius = bake.rocks[i].z;
    let d = wrapped(p, c);
    let r = length(d);
    let rim = 1.0 - smoothstep(0.0, 0.03, r - radius);
    let flow_dir = normalize(channel_velocity(c) + vec2f(1e-5, 0.0));
    let along = dot(d, flow_dir);
    let sideways = dot(d, vec2f(-flow_dir.y, flow_dir.x));
    let width = 0.5 * radius + 0.12 * max(along, 0.0);
    let wake = smoothstep(0.0, 0.03, along) * exp(-max(along, 0.0) / (5.0 * radius)) *
      exp(-0.5 * sideways * sideways / (width * width));
    foam = max(foam, max(0.6 * rim, 0.6 * wake));
  }
  return foam;
}

@fragment fn fs(in: FullscreenVarying) -> @location(0) vec4f {
  let p = vec2f(in.uv.x * WORLD_W, in.uv.y);
  let eps = 0.5 * bake.texel;
  let v = vec2f(
    psi(p + vec2f(0.0, eps)) - psi(p - vec2f(0.0, eps)),
    -(psi(p + vec2f(eps, 0.0)) - psi(p - vec2f(eps, 0.0))),
  ) / (2.0 * eps);

  let s = across(p);
  // Shallow ramp at the banks (zero exactly on the bank streamline), 1 in the middle third; the rocks
  // add their own ramp, 0.08 wide.
  let depth_bank = smoothstep(0.0, 0.5, 1.0 - abs(s));
  let clearance = rock_clearance(p);
  let depth = min(depth_bank, smoothstep(0.0, 0.08, clearance));
  // Still water against solid ground. A mask on the velocity only keeps the field divergence-free
  // where the flow runs along the mask's contours: along the bank (the mask depends on s alone) and
  // on the rock outline itself, so the rock mask is thin, and the stream function already makes the
  // flow tangent there.
  let moving = v * smoothstep(0.0, 0.1, depth_bank) * smoothstep(0.004, 0.01, clearance);
  // No foam on dry ground.
  let foam = foam_mask(p, depth_bank) * smoothstep(0.0, 0.004, depth);
  return vec4f(flow_encode(moving), foam, depth);
}
`;

/** Small deterministic PRNG so a seed reproduces the same river. */
function mulberry32(seed: number): () => number {
  let state = seed >>> 0;
  return () => {
    state = (state + 0x6d2b79f5) >>> 0;
    let t = state;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/**
 * Rocks as (x, s, radius) packed into vec4s: evenly spread along the river with jitter, alternating
 * between the two halves of the channel so they never overlap, and kept clear of the banks.
 */
function placeRocks(count: number, seed: number): number[] {
  const random = mulberry32(seed * 2654435761);
  const slot = FLOW_WORLD_ASPECT / Math.max(count, 1);
  const rocks: number[] = [];
  for (let i = 0; i < count; i++) {
    const x = (i + 0.5 + (random() - 0.5) * 0.5) * slot;
    const side = i % 2 === 0 ? 1 : -1;
    const s = side * (0.1 + 0.32 * random());
    const radius = 0.03 + 0.022 * random();
    rocks.push(x % FLOW_WORLD_ASPECT, s, radius, 0);
  }
  return rocks;
}

/**
 * Bakes a flow map of a river running left to right through a handful of rocks.
 *
 * Encoding (`rgba8unorm`, sampled WITHOUT sRGB decoding; no mips):
 *   rg  velocity in the texture's axes (x right, y down). Byte 127 is exactly zero, 0 and 254 are -1
 *       and +1. Decode with `flow_decode`; one unit is the fastest current anywhere in the map (the
 *       gaps beside the rocks), the mid-channel speed is 0.3-0.6. This is the 8-bit convention in
 *       flow.wgsl; the common 0.5 + 0.5 * v cannot represent zero.
 *   b   foam mask in [0, 1]: rings around the rocks, wakes behind them, a trace in the bank shallows
 *   a   water depth in [0, 1]: 0 on land (banks outside the river and the rocks), rising to 1 in the
 *       channel. Treat depth < ~0.05 as dry ground; it is also a ready-made shading input.
 *
 * Construction: velocity is the curl of a stream function, so it is divergence-free and flows around
 * obstacles. The stream function is a meandering channel with a parabolic cross-section (fast in the
 * middle, still at the banks, faster where the channel narrows), plus a potential-flow dipole per rock
 * and a little curl noise. The map is periodic in x (width 2, height 1 world units: FLOW_WORLD_ASPECT),
 * so it can scroll or repeat. Sample it with `createFlowSampler`.
 */
export function createFlowMap(device: GPUDevice, options: FlowMapOptions = {}): FlowMap {
  const width = options.width ?? 1024;
  if (!Number.isInteger(width) || width < 32 || width > 4096 || width % FLOW_WORLD_ASPECT !== 0) {
    throw new RangeError(`createFlowMap: width must be an even integer in 32..4096, got ${width}`);
  }
  const height = width / FLOW_WORLD_ASPECT;
  const count = clampInt(options.rocks ?? 6, 0, MAX_ROCKS);
  const label = options.label ?? 'flow map';
  const format: GPUTextureFormat = 'rgba8unorm';

  const texture = device.createTexture({
    label,
    size: [width, height],
    format,
    usage:
      GPUTextureUsage.TEXTURE_BINDING |
      GPUTextureUsage.RENDER_ATTACHMENT |
      GPUTextureUsage.COPY_SRC,
  });

  const uniforms = createUniformBuffer(device, FlowBake, `${label} params`);
  uniforms.set({
    rocks: placeRocks(count, options.seed ?? 1),
    count,
    turbulence: Math.min(3, Math.max(0, options.turbulence ?? 0.6)),
    // World units per texel; the world is `height` texels tall and 1 unit tall.
    texel: 1 / height,
  });
  uniforms.upload();

  const pipeline = createFullscreenPipeline(device, {
    label: `${label} bake`,
    fragment: {
      label: 'bake/flow.wgsl',
      code: FLOW_BAKE_WGSL,
      chunks: { 'params.wgsl': FlowBake.wgsl },
    },
    format,
  });
  const bindGroup = device.createBindGroup({
    label: `${label} bake`,
    layout: pipeline.getBindGroupLayout(0),
    entries: [{ binding: 0, resource: { buffer: uniforms.buffer } }],
  });
  const encoder = device.createCommandEncoder({ label: `${label} bake` });
  renderFullscreen(encoder, texture.createView({ label: `${label} target` }), pipeline, [
    bindGroup,
  ]);
  device.queue.submit([encoder.finish()]);
  uniforms.destroy();

  return {
    texture,
    view: texture.createView({ label: `${label} view` }),
    format,
    width,
    height,
    bytes: width * height * 4,
    destroy: () => texture.destroy(),
  };
}

/** Bilinear, repeating along x (the river is periodic), clamped along y. */
export function createFlowSampler(device: GPUDevice): GPUSampler {
  return device.createSampler({
    label: 'flow map sampler',
    addressModeU: 'repeat',
    addressModeV: 'clamp-to-edge',
    magFilter: 'linear',
    minFilter: 'linear',
  });
}
