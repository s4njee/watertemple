import { noProfiler, type PassProfiler } from '../profiler.ts';
import { createDepthTarget } from '../scene/depth.ts';
import { CameraUniform, setCameraUniform, type CameraMatrices } from '../scene/camera.ts';
import type { RenderTarget } from '../targets.ts';
import { createUniformBuffer, defineUniformStruct, uniformArray } from '../uniforms.ts';
import { cachedShaderModule } from './compute.ts';
import billboardSource from './billboards.wgsl';
import type { ParticleSet } from './types.ts';

// The instanced billboard renderer: the cheap, honest way to draw a ParticleSet, and the baseline the
// screen-space fluid surface is compared against. One camera-facing quad per particle, a round sprite sized in
// world units with a floor in pixels, coloured through a palette by the per-particle scalar.
//
// WHY AN INSTANCE-STEP VERTEX BUFFER AND NOT A STORAGE BUFFER. A ParticleSet's positions are created with both
// usages, and either works on a desktop browser. The vertex buffer wins on three counts. (1) Compatibility mode
// gives vertex stages no storage buffers at all (docs/support-matrix.md, 5.1), so a storage read would rule out
// every device that only offers the compat feature level, while a vertex buffer works there unchanged.
// (2) The renderer needs no bind group for the particles, so a solver that ping-pongs or reorders its buffers
// costs nothing here: the buffer is simply set per draw. (3) It is the same speed (the fixed-function vertex
// fetch is built for exactly this stream) and spends none of the 8 storage bindings per stage. The price is that
// per-instance data other than position must come as another vertex buffer; `scalar` below is that.
//
// BLEND MODES. `additive` (glows; order independent, so a reordering sort cannot flicker it), `alpha`
// (premultiplied over; not depth sorted, so overlapping particles blend in buffer order and a particle order
// that changes every frame, as after a reorder, shimmers) and `opaque` (lit spheres with true depth: every
// fragment writes the depth of the sphere surface, so intersecting spheres are correct and no sorting is
// needed). Only `opaque` needs a depth buffer; the others test against one if you pass it.
//
// COLOUR. Output is premultiplied alpha, sRGB-encoded unless the target format is an -srgb format (the canvas is
// not). Additive blending of sRGB-encoded values is not physically additive (it is the usual convention for
// glowing sprites); lower `intensity` for dense clouds.

export type BillboardMode = 'additive' | 'alpha' | 'opaque';

export type Rgb = readonly [number, number, number];

/** Live style: change any field any time. */
export interface BillboardStyle {
  mode: BillboardMode;
  /** Multiplies `ParticleSet.radius`. */
  sizeScale: number;
  /** Sprites are never drawn smaller than this radius in pixels; energy is conserved (fainter) below it. */
  minPixels: number;
  /** 0: hard antialiased disc. 1: smooth falloff from the centre. */
  softness: number;
  /** Additive brightness / alpha opacity. */
  intensity: number;
  /** Sphere shading amount in the opaque mode. */
  shading: number;
  /** The scalar value mapped to the first and last palette stop. */
  scalarRange: readonly [number, number];
  /** 2 to 8 linear-RGB colour stops. */
  palette: readonly Rgb[];
}

export const DEFAULT_BILLBOARD_STYLE: Readonly<BillboardStyle> = {
  mode: 'additive',
  sizeScale: 1,
  minPixels: 0.75,
  softness: 0.35,
  intensity: 1,
  shading: 1,
  scalarRange: [0, 1],
  palette: [
    [0.02, 0.12, 0.32],
    [0.05, 0.45, 0.75],
    [0.55, 0.9, 1],
    [1, 1, 1],
  ],
};

export const BILLBOARD_UNIFORM = defineUniformStruct('BillboardParams', {
  radius: 'f32',
  min_pixels: 'f32',
  softness: 'f32',
  intensity: 'f32',
  scalar_min: 'f32',
  scalar_max: 'f32',
  palette_count: 'u32',
  encode_srgb: 'u32',
  mode: 'u32',
  shading: 'f32',
  palette: uniformArray('vec4f', 8),
});

export interface BillboardOptions {
  device: GPUDevice;
  /** False for a linear floating-point scene that will be encoded by a post/output pass. */
  encodeSrgb?: boolean;
  /** Format of the colour target the renderer draws into. */
  format: GPUTextureFormat;
  /** Format of the depth attachment (when there is one). Default depth24plus. */
  depthFormat?: GPUTextureFormat;
  style?: Partial<BillboardStyle>;
  profiler?: PassProfiler;
  label?: string;
}

export interface DrawOptions {
  /** Size of the target in pixels (for the pixel floor and the antialiasing). Default: the last `resize`. */
  width?: number;
  height?: number;
  /**
   * A depth attachment to test against, shared with other geometry, in `depthFormat`. The opaque mode writes
   * it. Omit to use the renderer's own (sized by `resize`), which opaque rendering needs.
   */
  depth?: GPUTextureView;
  /** Clear the colour target to this colour (premultiplied), or `null` to keep what is there. Default: clear to transparent. */
  clear?: GPUColor | null;
  /** Clear the depth to 1. Default true; pass false to keep a depth buffer you filled before. */
  clearDepth?: boolean;
  /**
   * A buffer of one f32 per particle (VERTEX usage) to colour by instead of `positions.w`. A colour or
   * a heat map computed in a compute pass, e.g. a neighbour count.
   */
  scalar?: GPUBuffer;
  /** Profiler label. Default "particles draw". */
  timestamp?: string;
}

export interface BillboardRenderer {
  readonly style: BillboardStyle;
  /** Sets the size of the colour target (and re-creates the renderer's own depth buffer). */
  resize(width: number, height: number): void;
  draw(
    encoder: GPUCommandEncoder,
    set: ParticleSet,
    camera: CameraMatrices,
    target: GPUTextureView,
    options?: DrawOptions,
  ): void;
  destroy(): void;
}

const MODE_CODE: Record<BillboardMode, number> = { additive: 0, alpha: 1, opaque: 2 };

const BLEND: Record<Exclude<BillboardMode, 'opaque'>, GPUBlendState> = {
  additive: {
    color: { srcFactor: 'one', dstFactor: 'one', operation: 'add' },
    alpha: { srcFactor: 'one', dstFactor: 'one', operation: 'add' },
  },
  alpha: {
    color: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha', operation: 'add' },
    alpha: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha', operation: 'add' },
  },
};

export function createBillboardRenderer(options: BillboardOptions): BillboardRenderer {
  const {
    device,
    format,
    depthFormat = 'depth24plus',
    profiler = noProfiler,
    label = 'billboards',
  } = options;
  const style: BillboardStyle = {
    ...DEFAULT_BILLBOARD_STYLE,
    ...options.style,
    palette: options.style?.palette ?? DEFAULT_BILLBOARD_STYLE.palette,
  };

  const camera = createUniformBuffer(device, CameraUniform, `${label} camera`);
  const params = createUniformBuffer(device, BILLBOARD_UNIFORM, `${label} params`);
  const encodeSrgb = options.encodeSrgb === false || format.endsWith('-srgb') ? 0 : 1;

  let width = 1;
  let height = 1;
  let ownDepth: RenderTarget | undefined;

  const pipelines = new Map<string, { pipeline: GPURenderPipeline; bindGroup: GPUBindGroup }>();
  function pipelineFor(mode: BillboardMode, separateScalar: boolean, depthTest: boolean) {
    const key = `${mode}|${separateScalar}|${depthTest}`;
    let entry = pipelines.get(key);
    if (!entry) {
      const module = cachedShaderModule(device, {
        label: 'particles/billboards.wgsl',
        code: billboardSource.code,
        chunks: { ...billboardSource.chunks, 'billboard_params.wgsl': BILLBOARD_UNIFORM.wgsl },
      });
      const opaque = mode === 'opaque';
      const pipeline = device.createRenderPipeline({
        label: `${label} ${mode}`,
        layout: 'auto',
        vertex: {
          module,
          entryPoint: 'vs',
          buffers: [
            {
              arrayStride: 16,
              stepMode: 'instance',
              attributes: [{ shaderLocation: 0, offset: 0, format: 'float32x4' }],
            },
            {
              // The scalar is either positions.w (stride 16, offset 12) or its own tightly packed array.
              arrayStride: separateScalar ? 4 : 16,
              stepMode: 'instance',
              attributes: [
                { shaderLocation: 1, offset: separateScalar ? 0 : 12, format: 'float32' },
              ],
            },
          ],
        },
        fragment: {
          module,
          entryPoint: opaque ? 'fs_opaque' : 'fs',
          targets: [{ format, blend: opaque ? undefined : BLEND[mode] }],
        },
        primitive: { topology: 'triangle-strip', cullMode: 'none' },
        depthStencil:
          opaque || depthTest
            ? {
                format: depthFormat,
                depthWriteEnabled: opaque,
                depthCompare: 'less',
              }
            : undefined,
      });
      const bindGroup = device.createBindGroup({
        label: `${label} ${mode}`,
        layout: pipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: camera.buffer } },
          { binding: 1, resource: { buffer: params.buffer } },
        ],
      });
      entry = { pipeline, bindGroup };
      pipelines.set(key, entry);
    }
    return entry;
  }

  return {
    style,
    resize(w, h) {
      width = Math.max(1, Math.round(w));
      height = Math.max(1, Math.round(h));
      if (ownDepth) ownDepth.resize(width, height);
    },
    draw(encoder, set, cameraMatrices, target, drawOptions = {}) {
      const w = drawOptions.width ?? width;
      const h = drawOptions.height ?? height;
      const mode = style.mode;
      const separateScalar = drawOptions.scalar !== undefined;
      let depthView = drawOptions.depth;
      if (!depthView && mode === 'opaque') {
        ownDepth ??= createDepthTarget(device, {
          label: `${label} depth`,
          width: w,
          height: h,
          format: depthFormat as 'depth32float' | 'depth24plus',
          sampled: false,
        });
        ownDepth.resize(w, h);
        depthView = ownDepth.view;
      }
      const { pipeline, bindGroup } = pipelineFor(mode, separateScalar, depthView !== undefined);

      setCameraUniform(camera, cameraMatrices, w, h);
      camera.upload();
      const stops = style.palette.slice(0, 8);
      if (stops.length < 2) throw new RangeError(`${label}: the palette needs at least 2 stops`);
      params.set({
        radius: set.radius * style.sizeScale,
        min_pixels: style.minPixels,
        softness: style.softness,
        intensity: style.intensity,
        scalar_min: style.scalarRange[0],
        scalar_max: style.scalarRange[1],
        palette_count: stops.length,
        encode_srgb: encodeSrgb,
        mode: MODE_CODE[mode],
        shading: mode === 'opaque' ? style.shading : 0,
        palette: stops.flatMap(([r, g, b]) => [r, g, b, 1]),
      });
      params.upload();

      const clear =
        drawOptions.clear === undefined ? { r: 0, g: 0, b: 0, a: 0 } : drawOptions.clear;
      const pass = encoder.beginRenderPass({
        label: `${label} draw`,
        colorAttachments: [
          {
            view: target,
            loadOp: clear ? 'clear' : 'load',
            storeOp: 'store',
            clearValue: clear ?? undefined,
          },
        ],
        depthStencilAttachment: depthView
          ? {
              view: depthView,
              depthLoadOp: drawOptions.clearDepth === false ? 'load' : 'clear',
              depthStoreOp: 'store',
              depthClearValue: 1,
            }
          : undefined,
        timestampWrites: profiler.pass(drawOptions.timestamp ?? 'particles draw'),
      });
      if (set.count > 0) {
        pass.setPipeline(pipeline);
        pass.setBindGroup(0, bindGroup);
        pass.setVertexBuffer(0, set.positions);
        pass.setVertexBuffer(1, drawOptions.scalar ?? set.positions);
        pass.draw(4, set.count);
      }
      pass.end();
    },
    destroy() {
      camera.destroy();
      params.destroy();
      ownDepth?.destroy();
      pipelines.clear();
    },
  };
}
