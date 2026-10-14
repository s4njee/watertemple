import { createFullscreenPipeline, renderFullscreen } from '../fullscreen.ts';
import { createBlur, MAX_BLUR_RADIUS, type Blur } from '../passes/blur.ts';
import { linearSampler, perDevice } from '../passes/common.ts';
import {
  createBillboardRenderer,
  type BillboardRenderer,
  type Rgb,
} from '../particles/billboards.ts';
import { orthographicCamera } from '../particles/cameras.ts';
import type { ParticleSet } from '../particles/types.ts';
import { noProfiler, type PassProfiler } from '../profiler.ts';
import { createCameraMatrices } from '../scene/camera.ts';
import { createShader } from '../shader.ts';
import { createRenderTarget, type RenderTarget } from '../targets.ts';
import { createUniformBuffer, defineUniformStruct } from '../uniforms.ts';
import compositeSource from './composite.wgsl';
import splatSource from './splat.wgsl';
import wideSource from './wide.wgsl';

// The 2D liquid-surface renderer: screen-space fluid rendering for a particle fluid seen from the side.
//
// Drawn as discs, a 2D particle fluid looks like ball bearings. So every particle is splatted as a soft
// compact blob into a field texture, the field is blurred a little, and a fullscreen pass thresholds it with
// an anti-aliased edge and shades it as water: one body of liquid with a rim, a dome, a glint, refraction of
// whatever is behind it, and foam. See splat.wgsl and composite.wgsl for the shader side.
//
// FIELD NORMALISATION. The kernel is w(q) = k (1 - q^2)^3 for q = dist / R < 1, R = blobRadius * radius.
// Its integral over the plane is  integral w dA = k * 2 pi R^2 * integral_0^1 q (1-q^2)^3 dq = k pi R^2 / 4.
// A particle stands for `restVolume` (its area at rest, in 2D). Choosing
//
//     k = 4 restVolume / (pi R^2)
//
// gives every particle an integral of exactly restVolume, so a lattice with `restVolume` per particle sums
// to ~1 (the field is the particle density as a fraction of rest density), and the integral of the whole
// field is N * restVolume. restVolume defaults to (2 radius)^2, the cell of a square lattice of touching discs.
//
// WHERE THINGS APPEAR (smoothing 0, ThresholdT = `threshold`). F along a lone particle is k (1 - q^2)^3, so
//   - a lone particle is drawn as a disc of radius  R sqrt(1 - (T/k)^(1/3))  (nothing at all if k <= T:
//     with a big blobRadius isolated particles fall below the density threshold and vanish, like spray
//     thinner than the iso-density);
//   - two particles a distance d apart share a saddle at their midpoint of value 2 k (1 - (d / 2R)^2)^3, so
//     they merge into one region while  d < 2R sqrt(1 - (T / 2k)^(1/3)).
// With the defaults (blobRadius 2.2, T 0.5, restVolume (2 radius)^2: k = 1.05) a lone particle is a disc of
// radius ~1.03 radius and two merge up to a distance of ~2.7 radius, 1.35 lattice spacings.

export type Liquid2dView = 'final' | 'field' | 'discs';

/** The world rectangle shown. Its width follows the target aspect (x right, y up). */
export interface Liquid2dDomain {
  center: readonly [number, number];
  halfHeight: number;
}

/** Live look controls. The caller may pass the same object every frame; missing fields use the defaults. */
export interface Liquid2dParams {
  /** `final` the shaded liquid, `field` the raw field with the iso-line, `discs` plain discs (the baseline). */
  view: Liquid2dView;
  /** Kernel radius in units of `particles.radius`. Larger is smoother and merges farther. */
  blobRadius: number;
  /** Field value of the surface (the half-density iso-line is 0.5). */
  threshold: number;
  /** Width of the antialiased edge in pixels. */
  edgeSoftness: number;
  /** Gaussian radius (sigma) in field texels applied to the field before thresholding. 0 skips the pass. */
  smoothing: number;
  /** Named water colour 0..6 (clear tropical, coastal, murky green, tea lake, glacial milk, pool, open ocean). */
  waterType: number;
  /** Water path length in absorption units (metres) at the deepest point of a body (~60 px from the edge). */
  depthScale: number;
  /** Index of refraction for the backdrop offset. */
  ior: number;
  /** Refraction strength (backdrop offset, in image heights, for a unit slope). 0 disables. */
  refraction: number;
  /** Rim light: fresnel reflection of the sky at steep slopes plus a thin meniscus line inside the edge. */
  rim: number;
  /** Specular glint intensity. 0 disables. */
  highlight: number;
  /** Direction the light comes from, in degrees counter-clockwise from +x on the screen (90 = from above). */
  lightAngle: number;
  /** Foam amount 0..1. 0 disables. */
  foam: number;
  /** `positions.w` at and below which a particle adds no foam. */
  foamLow: number;
  /** `positions.w` at and above which a particle adds full foam. */
  foamHigh: number;
  /** Opacity of the deep body when drawn without a backdrop (0..1). */
  opacity: number;
  /** Fraction of `opacity` kept at zero thickness (thin edges, droplets) without a backdrop. 1 = uniform. */
  thinOpacity: number;
  /** 2 to 8 linear RGB stops for the `discs` view. */
  discPalette: readonly Rgb[];
  /** `positions.w` values mapped to the first and last palette stop in the `discs` view. */
  discRange: readonly [number, number];
}

export const DEFAULT_LIQUID2D_PARAMS: Readonly<Liquid2dParams> = {
  view: 'final',
  blobRadius: 2.6,
  threshold: 0.5,
  edgeSoftness: 1,
  smoothing: 1.5,
  waterType: 0,
  depthScale: 8,
  ior: 1.333,
  refraction: 0.12,
  rim: 1,
  highlight: 40,
  lightAngle: 125,
  foam: 0.6,
  foamLow: 0.3,
  foamHigh: 1,
  opacity: 0.9,
  thinOpacity: 0.6,
  discPalette: [
    [0.02, 0.12, 0.32],
    [0.05, 0.45, 0.75],
    [0.55, 0.9, 1],
    [1, 1, 1],
  ],
  discRange: [0, 1],
};

export interface Liquid2dOptions {
  device: GPUDevice;
  /** Format of the target views passed to `render`. */
  format: GPUTextureFormat;
  /** Target size in pixels. */
  width: number;
  height: number;
  /** Field texels per target pixel. Default 1. */
  fieldScale?: number;
  profiler?: PassProfiler;
  label?: string;
}

export interface Liquid2dRenderInput {
  particles: ParticleSet;
  domain: Liquid2dDomain;
  /**
   * Linear colour covering the whole target in uv (rgba16float, or an -srgb view). With it the output is
   * opaque `mix(backdrop, liquid, coverage)`; without, premultiplied alpha that can sit over page content.
   */
  backdrop?: GPUTextureView;
  target: GPUTextureView;
  params?: Partial<Liquid2dParams>;
  profiler?: PassProfiler;
  /** Keep the target's contents and blend over them. Default false: clear to transparent first. */
  load?: boolean;
}

export interface Liquid2d {
  /**
   * Encodes splat, blur and composite. A Blur object writes its parameters with `queue.writeBuffer`, so
   * call `render` at most once per submit.
   */
  render(encoder: GPUCommandEncoder, input: Liquid2dRenderInput): void;
  /** Resizes the field and blur targets to the new target size (contents are lost). */
  resize(width: number, height: number): void;
  /** The field as of the last `render` (r density, g density * foam). The texture changes on `resize`. */
  readonly field: { texture: GPUTexture; width: number; height: number; format: GPUTextureFormat };
  destroy(): void;
}

/** Field format: renderable, filterable and blendable on every device. */
const FIELD_FORMAT: GPUTextureFormat = 'rgba16float';

export const LIQUID2D_UNIFORM = defineUniformStruct('Liquid2dParams', {
  dom_center: 'vec2f',
  dom_half: 'vec2f',
  out_size: 'vec2f',
  field_size: 'vec2f',
  wide_size: 'vec2f',
  blob_radius: 'f32',
  kernel_k: 'f32',
  foam_low: 'f32',
  foam_high: 'f32',
  threshold: 'f32',
  edge_softness: 'f32',
  depth_scale: 'f32',
  ior: 'f32',
  refraction: 'f32',
  rim: 'f32',
  highlight: 'f32',
  light_angle: 'f32',
  foam: 'f32',
  opacity: 'f32',
  thin_opacity: 'f32',
  res: 'f32',
  wide_sigma: 'f32',
  water_type: 'i32',
  has_backdrop: 'u32',
  encode_srgb: 'u32',
});

/** Kernel constant k = 4 restVolume / (pi R^2): every particle's blob integrates to restVolume. */
export function liquid2dKernelK(restVolume: number, blobWorldRadius: number): number {
  return (4 * restVolume) / (Math.PI * blobWorldRadius * blobWorldRadius);
}

/** Radius (world units) of the disc a lone particle renders as, or 0 if its peak is below `threshold`. */
export function liquid2dLoneRadius(
  radius: number,
  restVolume: number,
  blobRadius: number,
  threshold: number,
): number {
  const blob = blobRadius * radius;
  const k = liquid2dKernelK(restVolume, blob);
  return k <= threshold ? 0 : blob * Math.sqrt(1 - Math.cbrt(threshold / k));
}

/** Centre distance (world units) up to which two particles render as one connected region. */
export function liquid2dMergeDistance(
  radius: number,
  restVolume: number,
  blobRadius: number,
  threshold: number,
): number {
  const blob = blobRadius * radius;
  const k = liquid2dKernelK(restVolume, blob);
  return 2 * k <= threshold ? 0 : 2 * blob * Math.sqrt(1 - Math.cbrt(threshold / (2 * k)));
}

interface DeviceState {
  layout: GPUBindGroupLayout;
  splat: GPURenderPipeline;
  wide: GPURenderPipeline;
  composite: Map<GPUTextureFormat, Pipelines>;
}
interface Pipelines {
  final: GPURenderPipeline;
  field: GPURenderPipeline;
  backdrop: GPURenderPipeline;
}

const PREMULTIPLIED: GPUBlendState = {
  color: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha', operation: 'add' },
  alpha: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha', operation: 'add' },
};
const ADDITIVE: GPUBlendState = {
  color: { srcFactor: 'one', dstFactor: 'one', operation: 'add' },
  alpha: { srcFactor: 'one', dstFactor: 'one', operation: 'add' },
};

const stateFor = perDevice((device): DeviceState => {
  const chunks = { 'liquid2d_params.wgsl': LIQUID2D_UNIFORM.wgsl };
  const module = createShader(device, {
    label: 'liquid-2d/splat.wgsl',
    code: splatSource.code,
    chunks,
  });
  return {
    layout: device.createBindGroupLayout({
      label: 'liquid-2d composite layout',
      entries: [
        { binding: 0, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
        { binding: 1, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float' } },
        { binding: 2, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float' } },
        { binding: 3, visibility: GPUShaderStage.FRAGMENT, sampler: { type: 'filtering' } },
        { binding: 4, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float' } },
      ],
    }),
    splat: device.createRenderPipeline({
      label: 'liquid-2d splat',
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
        ],
      },
      fragment: {
        module,
        entryPoint: 'fs',
        targets: [{ format: FIELD_FORMAT, blend: ADDITIVE }],
      },
      primitive: { topology: 'triangle-strip', cullMode: 'none' },
    }),
    wide: createFullscreenPipeline(device, {
      label: 'liquid-2d wide',
      fragment: { label: 'liquid-2d/wide.wgsl', code: wideSource.code, chunks },
      format: FIELD_FORMAT,
    }),
    composite: new Map(),
  };
});

function compositePipelines(
  device: GPUDevice,
  state: DeviceState,
  format: GPUTextureFormat,
): Pipelines {
  let pipelines = state.composite.get(format);
  if (!pipelines) {
    const make = (entryPoint: string) =>
      createFullscreenPipeline(device, {
        label: `liquid-2d ${entryPoint} ${format}`,
        fragment: {
          label: 'liquid-2d/composite.wgsl',
          code: compositeSource.code,
          chunks: { ...compositeSource.chunks, 'liquid2d_params.wgsl': LIQUID2D_UNIFORM.wgsl },
        },
        entryPoint,
        format,
        blend: PREMULTIPLIED,
        layout: [state.layout],
      });
    pipelines = { final: make('fs'), field: make('fs_field'), backdrop: make('fs_backdrop') };
    state.composite.set(format, pipelines);
  }
  return pipelines;
}

type ParamKey = keyof Liquid2dParams;
const PARAM_KEYS = Object.keys(DEFAULT_LIQUID2D_PARAMS) as ParamKey[];

/** Sigma, in target pixels at the reference height, of the blur that makes the thickness map. */
const WIDE_SIGMA = 24;
/** The thickness map is the field reduced by this factor. */
const WIDE_REDUCTION = 4;

/** Pixel height at which the shader's pixel-sized constants (ring radii, gradient gain) are authored. */
const REFERENCE_HEIGHT = 560;

export function createLiquid2d(options: Liquid2dOptions): Liquid2d {
  const { device, format, fieldScale = 1, label = 'liquid-2d' } = options;
  if (!(fieldScale > 0)) throw new RangeError(`${label}: fieldScale must be positive`);
  const state = stateFor(device);
  const pipelines = compositePipelines(device, state, format);
  const maxDimension = device.limits.maxTextureDimension2D;
  const clampSize = (value: number) => Math.min(maxDimension, Math.max(1, Math.round(value)));

  let width = clampSize(options.width);
  let height = clampSize(options.height);
  let destroyed = false;
  const fieldSize = () => [clampSize(width * fieldScale), clampSize(height * fieldScale)] as const;

  const [fieldWidth, fieldHeight] = fieldSize();
  const fieldTarget: RenderTarget = createRenderTarget(device, {
    label: `${label} field`,
    format: FIELD_FORMAT,
    width: fieldWidth,
    height: fieldHeight,
    usage:
      GPUTextureUsage.RENDER_ATTACHMENT |
      GPUTextureUsage.TEXTURE_BINDING |
      GPUTextureUsage.COPY_SRC,
  });
  const wideSize = () =>
    [
      Math.max(1, Math.ceil(fieldTarget.width / WIDE_REDUCTION)),
      Math.max(1, Math.ceil(fieldTarget.height / WIDE_REDUCTION)),
    ] as const;
  const wideTarget: RenderTarget = createRenderTarget(device, {
    label: `${label} wide`,
    format: FIELD_FORMAT,
    width: wideSize()[0],
    height: wideSize()[1],
  });
  const wideBlur: Blur = createBlur(device, {
    label: `${label} wide blur`,
    format: FIELD_FORMAT,
    width: wideTarget.width,
    height: wideTarget.height,
  });
  let blur: Blur | undefined;
  const ensureBlur = (): Blur =>
    (blur ??= createBlur(device, {
      label: `${label} blur`,
      format: FIELD_FORMAT,
      width: fieldTarget.width,
      height: fieldTarget.height,
    }));

  const uniforms = createUniformBuffer(device, LIQUID2D_UNIFORM, `${label} params`);
  const encodeSrgb = format.endsWith('-srgb') ? 0 : 1;
  const sampler = linearSampler(device);
  const noBackdrop = device.createTexture({
    label: `${label} no backdrop`,
    size: [1, 1],
    format: FIELD_FORMAT,
    usage: GPUTextureUsage.TEXTURE_BINDING,
  });
  const noBackdropView = noBackdrop.createView();

  // Bind groups by (field view, backdrop view). The field view changes only on resize.
  const bindGroups = new WeakMap<GPUTextureView, WeakMap<GPUTextureView, GPUBindGroup>>();
  const bindGroupFor = (
    field: GPUTextureView,
    wide: GPUTextureView,
    backdrop: GPUTextureView,
  ): GPUBindGroup => {
    let byBackdrop = bindGroups.get(field);
    if (!byBackdrop) {
      byBackdrop = new WeakMap();
      bindGroups.set(field, byBackdrop);
    }
    let group = byBackdrop.get(backdrop);
    if (!group) {
      group = device.createBindGroup({
        label: `${label} composite`,
        layout: state.layout,
        entries: [
          { binding: 0, resource: { buffer: uniforms.buffer } },
          { binding: 1, resource: field },
          { binding: 2, resource: backdrop },
          { binding: 3, resource: sampler },
          { binding: 4, resource: wide },
        ],
      });
      byBackdrop.set(backdrop, group);
    }
    return group;
  };
  // The wide pass reads the field, so its bind group follows the field view.
  const wideGroups = new WeakMap<GPUTextureView, GPUBindGroup>();
  const wideGroupFor = (field: GPUTextureView): GPUBindGroup => {
    let group = wideGroups.get(field);
    if (!group) {
      group = device.createBindGroup({
        label: `${label} wide`,
        layout: state.wide.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: uniforms.buffer } },
          { binding: 1, resource: field },
          { binding: 3, resource: sampler },
        ],
      });
      wideGroups.set(field, group);
    }
    return group;
  };
  const splatGroup = device.createBindGroup({
    label: `${label} splat`,
    layout: state.splat.getBindGroupLayout(0),
    entries: [{ binding: 0, resource: { buffer: uniforms.buffer } }],
  });

  let discs: BillboardRenderer | undefined;
  const camera = createCameraMatrices();
  // The billboard renderer keeps the profiler it was created with; this forwards to the current frame's.
  let frameProfiler: PassProfiler = noProfiler;
  const profilerProxy: PassProfiler = { pass: (name) => frameProfiler.pass(name) };
  // The ParticleSet the discs are drawn from: the caller's, with `count` clamped to what is live.
  const discSet = {
    positions: undefined as unknown as GPUBuffer,
    count: 0,
    capacity: 0,
    radius: 0,
    generation: 0,
  };
  // Reused every frame: the resolved parameters.
  const resolved: Liquid2dParams = { ...DEFAULT_LIQUID2D_PARAMS };
  const resolvedRecord = resolved as unknown as Record<ParamKey, unknown>;
  const defaultRecord = DEFAULT_LIQUID2D_PARAMS as unknown as Record<ParamKey, unknown>;

  function resolveParams(given: Partial<Liquid2dParams> | undefined): Liquid2dParams {
    const givenRecord = given as Record<ParamKey, unknown> | undefined;
    for (const key of PARAM_KEYS) {
      const value = givenRecord?.[key];
      resolvedRecord[key] = value === undefined ? defaultRecord[key] : value;
    }
    return resolved;
  }

  return {
    get field() {
      return {
        texture: fieldTarget.texture,
        width: fieldTarget.width,
        height: fieldTarget.height,
        format: FIELD_FORMAT,
      };
    },

    resize(newWidth, newHeight) {
      if (destroyed) return;
      width = clampSize(newWidth);
      height = clampSize(newHeight);
      const [w, h] = fieldSize();
      if (fieldTarget.resize(w, h)) {
        blur?.resize(w, h);
        const [ww, wh] = wideSize();
        wideTarget.resize(ww, wh);
        wideBlur.resize(ww, wh);
      }
      discs?.resize(width, height);
    },

    render(encoder, input) {
      if (destroyed) throw new Error(`${label}: render after destroy`);
      const p = resolveParams(input.params);
      const profiler = input.profiler ?? options.profiler ?? noProfiler;
      frameProfiler = profiler;
      const { particles, domain } = input;
      const aspect = width / height;
      const live = Math.min(
        Math.max(0, Math.floor(particles.count)),
        particles.capacity,
        particles.positions.size >> 4,
      );

      const radius = particles.radius;
      const blobWorld = Math.max(p.blobRadius * radius, 1e-9);
      const restVolume = particles.restVolume ?? (2 * radius) ** 2;
      const hasBackdrop = input.backdrop !== undefined;
      uniforms.set({
        dom_center: domain.center,
        dom_half: [domain.halfHeight * aspect, domain.halfHeight],
        out_size: [width, height],
        field_size: [fieldTarget.width, fieldTarget.height],
        wide_size: [wideTarget.width, wideTarget.height],
        blob_radius: blobWorld,
        kernel_k: liquid2dKernelK(restVolume, blobWorld),
        foam_low: p.foamLow,
        foam_high: Math.max(p.foamHigh, p.foamLow + 1e-6),
        threshold: p.threshold,
        edge_softness: p.edgeSoftness,
        depth_scale: p.depthScale,
        ior: p.ior,
        refraction: p.refraction,
        rim: p.rim,
        highlight: p.highlight,
        light_angle: p.lightAngle,
        foam: p.foam,
        opacity: p.opacity,
        thin_opacity: p.thinOpacity,
        res: height / REFERENCE_HEIGHT,
        wide_sigma: WIDE_SIGMA * (height / REFERENCE_HEIGHT),
        water_type: Math.round(p.waterType),
        has_backdrop: hasBackdrop ? 1 : 0,
        encode_srgb: encodeSrgb,
      });
      uniforms.upload();

      const backdropView = input.backdrop ?? noBackdropView;
      const view = p.view;

      if (view !== 'discs') {
        // 1. Splat.
        const splat = encoder.beginRenderPass({
          label: `${label} splat`,
          colorAttachments: [
            {
              view: fieldTarget.view,
              loadOp: 'clear',
              storeOp: 'store',
              clearValue: { r: 0, g: 0, b: 0, a: 0 },
            },
          ],
          timestampWrites: profiler.pass('liquid splat'),
        });
        if (live > 0 && radius > 0) {
          splat.setPipeline(state.splat);
          splat.setBindGroup(0, splatGroup);
          splat.setVertexBuffer(0, particles.positions);
          splat.draw(4, live);
        }
        splat.end();

        // 2. Blur (optional), in place: the Blur goes through its own intermediate.
        if (p.smoothing > 0) {
          const taps = Math.min(MAX_BLUR_RADIUS, Math.max(1, Math.ceil(2.5 * p.smoothing)));
          ensureBlur().encode(encoder, fieldTarget.view, fieldTarget.view, {
            radius: taps,
            sigma: p.smoothing,
          });
        }

        // 2b. The thickness map: reduce, then blur (sigma in wide texels = target px * fieldScale / reduction).
        if (view === 'final') {
          renderFullscreen(encoder, wideTarget.view, state.wide, [wideGroupFor(fieldTarget.view)], {
            label: `${label} wide`,
            timestampWrites: profiler.pass('liquid wide'),
          });
          const sigma = (WIDE_SIGMA * (height / REFERENCE_HEIGHT) * fieldScale) / WIDE_REDUCTION;
          wideBlur.encode(encoder, wideTarget.view, wideTarget.view, {
            radius: Math.min(MAX_BLUR_RADIUS, Math.max(1, Math.ceil(2.5 * sigma))),
            sigma,
          });
        }
      }

      // 3. Composite.
      const pipeline =
        view === 'final'
          ? pipelines.final
          : view === 'field'
            ? pipelines.field
            : pipelines.backdrop;
      renderFullscreen(
        encoder,
        input.target,
        pipeline,
        [bindGroupFor(fieldTarget.view, wideTarget.view, backdropView)],
        {
          label: `${label} composite`,
          load: input.load,
          timestampWrites: profiler.pass('liquid composite'),
        },
      );

      // 4. The baseline: plain discs over the backdrop.
      if (view === 'discs') {
        discs ??= createBillboardRenderer({
          device,
          format,
          label: `${label} discs`,
          profiler: profilerProxy,
          style: { mode: 'alpha', softness: 0, minPixels: 0.75, intensity: 1 },
        });
        discs.style.palette = p.discPalette;
        discs.style.scalarRange = p.discRange;
        orthographicCamera(
          { center: domain.center, halfHeight: domain.halfHeight, aspect },
          camera,
        );
        discSet.positions = particles.positions;
        discSet.count = live;
        discSet.capacity = particles.capacity;
        discSet.radius = particles.radius;
        discSet.generation = particles.generation;
        discs.draw(encoder, discSet, camera, input.target, {
          width,
          height,
          clear: null,
          timestamp: 'liquid discs',
        });
      }
    },

    destroy() {
      if (destroyed) return;
      destroyed = true;
      fieldTarget.destroy();
      blur?.destroy();
      wideTarget.destroy();
      wideBlur.destroy();
      discs?.destroy();
      uniforms.destroy();
      noBackdrop.destroy();
    },
  };
}
