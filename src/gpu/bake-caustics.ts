import { createFullscreenPipeline, renderFullscreen } from './fullscreen.ts';
import { generateMipmaps, mipLevelCount } from './mipmap.ts';
import { createUniformBuffer, defineUniformStruct } from './uniforms.ts';

/*
 * A tileable caustic map, rendered once. Per-pixel caustics cost a cell search or a stack of sines
 * every frame; this costs one texture of four animation frames (the channels of an `rgba16float`
 * texture) and then two texture taps per pixel (`caustics_baked` in caustics.wgsl).
 *
 *   createCausticsMap       the texture: period 1 in uv, four time slices, a full mip chain
 *   createCausticsSampler   repeating, trilinear: what it wants to be sampled with
 *
 * Bake cost is a few tenths of a millisecond of GPU time at 512x512 and happens once at startup.
 */

/** Cells across the baked tile unless asked otherwise: 12 gives a long period at 512 texels. */
export const CAUSTICS_MAP_CELLS = 12;

/** Which construction in caustics.wgsl fills the map. */
export const CAUSTICS_BAKE_METHODS = ['cellular', 'sine'] as const;
export type CausticsBakeMethod = (typeof CAUSTICS_BAKE_METHODS)[number];

export interface CausticsMapOptions {
  /** Texels per side; a power of two from 32 to 2048. Default 512. */
  size?: number;
  /**
   * Pattern cells across the tile (integer, 1..32). Default `CAUSTICS_MAP_CELLS` (12). The bigger it is, the longer before
   * the pattern repeats, and the finer the filaments for a given texture size.
   */
  cells?: number;
  /** Default `cellular`. */
  method?: CausticsBakeMethod;
  /** 0 (soft net) to 1 (thin filaments), fixed at bake time. Default 0.85. */
  sharpness?: number;
  label?: string;
}

export interface CausticsMap {
  readonly texture: GPUTexture;
  /** All mip levels. */
  readonly view: GPUTextureView;
  /** Always `rgba16float`: renderable and filterable in WebGPU core. */
  readonly format: GPUTextureFormat;
  readonly size: number;
  readonly cells: number;
  readonly mipLevelCount: number;
  /** Estimated GPU memory including the mip chain. */
  readonly bytes: number;
  destroy(): void;
}

const CausticsBake = defineUniformStruct('CausticsBake', {
  cells: 'i32',
  method: 'u32',
  sharp: 'f32',
});

const CAUSTICS_BAKE_WGSL = /* wgsl */ `
#include "fullscreen.wgsl"
#include "caustics.wgsl"
#include "params.wgsl"

@group(0) @binding(0) var<uniform> bake: CausticsBake;

// Four slices of one loop, a quarter loop apart: r at 0, g at TAU/4, b at TAU/2, a at 3 TAU/4.
@fragment fn fs(in: FullscreenVarying) -> @location(0) vec4f {
  let uv = in.uv;
  let q = HALF_PI;
  return vec4f(
    caustics_eval(bake.method, uv, bake.cells, 0.0, bake.sharp),
    caustics_eval(bake.method, uv, bake.cells, q, bake.sharp),
    caustics_eval(bake.method, uv, bake.cells, 2.0 * q, bake.sharp),
    caustics_eval(bake.method, uv, bake.cells, 3.0 * q, bake.sharp),
  );
}
`;

function isPowerOfTwo(value: number): boolean {
  return Number.isInteger(value) && value > 0 && (value & (value - 1)) === 0;
}

function clampInt(value: number, lo: number, hi: number): number {
  return Math.min(hi, Math.max(lo, Math.round(value)));
}

/**
 * Bakes a seamlessly tileable caustic map: four time slices of one loop in r, g, b, a, values with a
 * mean of about 1 (so it is NOT clamped to 1: the filaments are several times brighter), and a mip chain
 * made by averaging, which keeps the mean while blurring (use a higher `lod` for softer, deeper water).
 * Sample it with `caustics_baked` from caustics.wgsl and `createCausticsSampler`.
 */
export function createCausticsMap(
  device: GPUDevice,
  options: CausticsMapOptions = {},
): CausticsMap {
  const size = options.size ?? 512;
  if (!isPowerOfTwo(size) || size < 32 || size > 2048) {
    throw new RangeError(`createCausticsMap: size must be a power of two in 32..2048, got ${size}`);
  }
  const cells = clampInt(options.cells ?? CAUSTICS_MAP_CELLS, 1, 32);
  const method = options.method ?? 'cellular';
  const methodIndex = Math.max(0, CAUSTICS_BAKE_METHODS.indexOf(method));
  const label = options.label ?? 'caustics map';
  const levels = mipLevelCount(size, size);
  const format: GPUTextureFormat = 'rgba16float';

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

  const uniforms = createUniformBuffer(device, CausticsBake, `${label} params`);
  uniforms.set({
    cells,
    method: methodIndex,
    sharp: Math.min(1, Math.max(0, options.sharpness ?? 0.85)),
  });
  uniforms.upload();

  const pipeline = createFullscreenPipeline(device, {
    label: `${label} bake`,
    fragment: {
      label: 'bake/caustics.wgsl',
      code: CAUSTICS_BAKE_WGSL,
      chunks: { 'params.wgsl': CausticsBake.wgsl },
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
    bytes += side * side * 8;
  }
  return {
    texture,
    view: texture.createView({ label: `${label} view` }),
    format,
    size,
    cells,
    mipLevelCount: levels,
    bytes,
    destroy: () => texture.destroy(),
  };
}

/** Trilinear and repeating: what a tileable caustic map should be sampled with. */
export function createCausticsSampler(device: GPUDevice): GPUSampler {
  return device.createSampler({
    label: 'caustics repeat sampler',
    addressModeU: 'repeat',
    addressModeV: 'repeat',
    magFilter: 'linear',
    minFilter: 'linear',
    mipmapFilter: 'linear',
  });
}
