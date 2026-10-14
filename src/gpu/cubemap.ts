import { createFullscreenPipeline } from './fullscreen.ts';
import { generateMipmaps, mipLevelCount } from './mipmap.ts';
import type { PassProfiler } from './profiler.ts';
import type { ShaderSource } from './shader.ts';

// --- Face orientation convention -----------------------------------------------------------------
//
// The standard WebGPU / Direct3D / OpenGL cube map layout. Array layers 0..5 are +X, -X, +Y, -Y, +Z, -Z.
// Inside a face, `s` runs left to right and `t` top to bottom (texture coordinates: the first row
// of the face is the top), both in [-1, 1] across the face. The direction a texel looks along:
//
//   face  +X       -X       +Y       -Y       +Z       -Z
//   dir   (1,-t,-s) (-1,-t,s) (s,1,t)  (s,-1,-t) (s,-t,1) (-s,-t,-1)
//
// This is what `textureSample(cube, sampler, dir)` expects, so a cubemap rendered here can be
// sampled with the same direction vector that was used to fill it. tests/optics-surface.test.ts
// reads each face back and samples through the hardware to prove it.

export const CUBE_FACE_NAMES = ['+x', '-x', '+y', '-y', '+z', '-z'] as const;

/** Unnormalised direction of the point (u, v) of a face; u, v in [0, 1] with (0, 0) the top-left. */
export function cubeFaceDirection(face: number, u: number, v: number): [number, number, number] {
  const s = 2 * u - 1;
  const t = 2 * v - 1;
  switch (face) {
    case 0:
      return [1, -t, -s];
    case 1:
      return [-1, -t, s];
    case 2:
      return [s, 1, t];
    case 3:
      return [s, -1, -t];
    case 4:
      return [s, -t, 1];
    case 5:
      return [-s, -t, -1];
    default:
      throw new RangeError(`cubeFaceDirection: face ${face} is not in 0..5`);
  }
}

export interface CubemapOptions {
  /** Edge length of each face in texels. */
  size: number;
  /**
   * WGSL that defines `fn cubemap_source(dir: vec3f) -> vec3f`, the linear RGB radiance seen along
   * the unit vector `dir`. `chunks` supplies effect-local includes, and the source may
   * `#include` anything from src/wgsl (typically "sky.wgsl"). It also sees
   * `cubemap_inputs.user[0..1]` (`array<vec4f, 2>`): eight floats passed to `render`, for a sun
   * direction or other animated parameters.
   */
  source: ShaderSource;
  /** Also build the mip chain (a plain per-face linear downsample; no filtering across face edges). */
  mipmaps?: boolean;
  label?: string;
  /** Added to the usage the helper always sets, e.g. `COPY_SRC` for readback. */
  extraUsage?: GPUTextureUsageFlags;
}

export interface Cubemap {
  /** An `rgba16float` 2D array texture with 6 layers; HDR values are kept (up to 65504). */
  readonly texture: GPUTexture;
  /** A `cube` view over every mip level: bind it as `texture_cube<f32>`. */
  readonly view: GPUTextureView;
  readonly size: number;
  /**
   * Records the six face passes (and the mip chain) into `encoder`. `user` is up to eight floats
   * exposed to the source as `cubemap_inputs.user`. The values are written with the queue, so two
   * `render` calls before one submit leave only the second call's values. Each face pass is timed
   * under `label` when a profiler is given (the HUD sums them).
   */
  render(
    encoder: GPUCommandEncoder,
    user?: ArrayLike<number>,
    timing?: { profiler: PassProfiler; label: string },
  ): void;
  destroy(): void;
}

const INPUT_STRIDE = 256; // minUniformBufferOffsetAlignment is at most 256 on every device
const INPUT_SIZE = 48; // u32 face + 12 bytes padding + array<vec4f, 2>

const FRAGMENT = `#include "fullscreen.wgsl"
#include "SOURCE"

struct CubemapInputs {
  face: u32,
  user: array<vec4f, 2>,
}
@group(0) @binding(0) var<uniform> cubemap_inputs: CubemapInputs;

fn cubemap_face_direction(face: u32, uv: vec2f) -> vec3f {
  let s = 2.0 * uv.x - 1.0;
  let t = 2.0 * uv.y - 1.0;
  switch (face) {
    case 0u: { return vec3f(1.0, -t, -s); }
    case 1u: { return vec3f(-1.0, -t, s); }
    case 2u: { return vec3f(s, 1.0, t); }
    case 3u: { return vec3f(s, -1.0, -t); }
    case 4u: { return vec3f(s, -t, 1.0); }
    default: { return vec3f(-s, -t, -1.0); }
  }
}

@fragment fn fs(in: FullscreenVarying) -> @location(0) vec4f {
  let dir = normalize(cubemap_face_direction(cubemap_inputs.face, in.uv));
  let radiance = cubemap_source(dir);
  // rgba16float tops out at 65504; keep NaN and infinity out of the filter chain.
  return vec4f(clamp(radiance, vec3f(0.0), vec3f(60000.0)), 1.0);
}
`;

/** Allocates the cube texture and the pipeline. Nothing is drawn until `render`. */
export function createCubemap(device: GPUDevice, options: CubemapOptions): Cubemap {
  const { size, source } = options;
  const label = options.label ?? 'cubemap';
  if (!Number.isInteger(size) || size < 1 || size > device.limits.maxTextureDimension2D) {
    throw new RangeError(`createCubemap: size ${size} is not a valid face size`);
  }
  const levels = options.mipmaps ? mipLevelCount(size, size) : 1;
  const format: GPUTextureFormat = 'rgba16float';

  const texture = device.createTexture({
    label,
    size: [size, size, 6],
    format,
    mipLevelCount: levels,
    usage:
      GPUTextureUsage.TEXTURE_BINDING |
      GPUTextureUsage.RENDER_ATTACHMENT |
      (options.extraUsage ?? 0),
  });
  const view = texture.createView({ label: `${label} cube`, dimension: 'cube' });
  const faceViews = Array.from({ length: 6 }, (_, face) =>
    texture.createView({
      label: `${label} ${CUBE_FACE_NAMES[face]}`,
      dimension: '2d',
      baseMipLevel: 0,
      mipLevelCount: 1,
      baseArrayLayer: face,
      arrayLayerCount: 1,
    }),
  );

  const layout = device.createBindGroupLayout({
    label: `${label} layout`,
    entries: [
      {
        binding: 0,
        visibility: GPUShaderStage.FRAGMENT,
        buffer: { type: 'uniform', hasDynamicOffset: true, minBindingSize: INPUT_SIZE },
      },
    ],
  });
  const pipeline = createFullscreenPipeline(device, {
    label,
    // The caller's code goes in as an include so diagnostics point at its own file name.
    fragment: {
      label: `${label} (cubemap pass)`,
      code: FRAGMENT.replace('SOURCE', source.label),
      chunks: { ...source.chunks, [source.label]: source.code },
    },
    format,
    layout: [layout],
  });
  const inputs = device.createBuffer({
    label: `${label} inputs`,
    size: INPUT_STRIDE * 6,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  const bindGroup = device.createBindGroup({
    label: `${label} inputs`,
    layout,
    entries: [{ binding: 0, resource: { buffer: inputs, size: INPUT_SIZE } }],
  });

  return {
    texture,
    view,
    size,
    render(encoder, user = [], timing) {
      const data = new ArrayBuffer(INPUT_STRIDE * 6);
      const words = new Uint32Array(data);
      const floats = new Float32Array(data);
      for (let face = 0; face < 6; face++) {
        const base = (face * INPUT_STRIDE) / 4;
        words[base] = face;
        for (let i = 0; i < Math.min(user.length, 8); i++) floats[base + 4 + i] = user[i] ?? 0;
      }
      device.queue.writeBuffer(inputs, 0, data);

      for (let face = 0; face < 6; face++) {
        // The shared fullscreen helper has no dynamic offsets, so the pass is encoded here.
        const pass = encoder.beginRenderPass({
          label: `${label} ${CUBE_FACE_NAMES[face]}`,
          colorAttachments: [{ view: faceViews[face]!, loadOp: 'clear', storeOp: 'store' }],
          timestampWrites: timing?.profiler.pass(timing.label),
        });
        pass.setPipeline(pipeline);
        pass.setBindGroup(0, bindGroup, [face * INPUT_STRIDE]);
        pass.draw(3);
        pass.end();
      }
      if (levels > 1) generateMipmaps(device, texture, encoder);
    },
    destroy() {
      texture.destroy();
      inputs.destroy();
    },
  };
}
