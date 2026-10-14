import { createFullscreenPipeline, renderFullscreen } from '../fullscreen.ts';
import { fitScale, type MediaFit } from '../media.ts';
import { noProfiler, type PassProfiler } from '../profiler.ts';
import { createUniformBuffer, defineUniformStruct } from '../uniforms.ts';
import advectSource from './fluid-warp-advect.wgsl';
import applySource from './fluid-warp-apply.wgsl';
import coreSource from './fluid-warp-core.wgsl';
import type { Fluid, FluidTexture } from './fluid.ts';

// Fluid as page distortion (story 6.4): the velocity field of the stable-fluids solver pushes CONTENT
// around instead of showing dye. Drag across an image and it smears and swirls like wet paint, then
// (if you want it to) springs back; a headline's letters are dragged along and reassemble.
//
//   const fluid = createFluid({ device, resolution: 128, dyeResolution: 32, ... });
//   const warp = createFluidWarp({ device, fluid, params: { relaxation: 1.2 } });
//   warp.attach();                         // the field is stepped inside every fluid step
//   const view = createFluidWarpPresenter({ device, format });
//   // each frame:
//   stirrer.stir(fluid, ...);  fluid.update(encoder, delta);
//   view.draw(encoder, target, { warp, content, fit: 'cover', canvasSize: [w, h] });
//
// THE TECHNIQUE. The warp keeps a DISPLACEMENT FIELD D: one rgba16float texel stores how far the content
// at that point has been moved from where it rests (image heights, y down). A step is
//     D'(x) = D(x - dt v) + gain dt v,   D' *= exp(-relaxation dt),   |D'| <= maxDisplacement
// i.e. the field is advected by the velocity (the parcel at x carries its displacement with it), gains the
// motion the flow just gave it, and relaxes toward zero at a controllable rate. Drawing then samples
// the content at  uv - D: a BACKWARD map, so there are no holes or overlaps, and D = 0 is the source
// unchanged to the byte. See fluid-warp-advect.wgsl for the derivation and fluid-warp-core.wgsl for
// the sampling functions an effect can use in its own shader.
//
// WHERE THE FIELD LIVES. Options considered:
//   1. The solver's DYE texture, driven through `fluid.hooks.dye` (add v dt after the solver advects
//      it; dye dissipation is the relaxation). Cheapest: no extra target and the solver's MacCormack
//      advection and boundary handling come for free. Rejected because it takes the dye away from the
//      fluid (a consumer could not show ink AND warp content), the relaxation would be tied to the
//      dye's exponential decay and clamp (`dyeLimit` is at least 1), and the grid is the dye's.
//   2. A field of its own, stepped from `hooks.dye` as a per-step callback (what this file does). Costs
//      one small texture pair (about 2 samples per sim cell: 256 x 144 texels at the default) and one
//      fullscreen pass per step, but is independent of everything the solver does with its dye, has
//      its own resolution, relaxation, gain and clamp, and works with any solver configuration.
// It is advected semi-Lagrangian (bilinear): a displacement field is smooth, so the numerical
// diffusion that blurs dye only softens the smear a little, and it cannot overshoot, which a
// MacCormack correction of a field that is allowed to go negative could not promise as cheaply.
//
// The hook runs once per solver step with the velocity that step ended with, so the field sees every
// step, however many the solver ran this frame. `step` is public for solvers that are driven by hand.

/** Format of the displacement field. */
export const WARP_FORMAT: GPUTextureFormat = 'rgba16float';

export interface FluidWarpParams {
  /**
   * Rate at which displacement decays toward zero, 1 / s: D *= exp(-relaxation * dt). 0 never relaxes:
   * a smear is permanent, like paint. About 1 settles in a few seconds (a wet sheet), 4 and up is a
   * spring (jelly, a ripple that is gone in half a second). Default 1.
   */
  relaxation: number;
  /** Largest displacement, image heights. The vector is scaled down to this length. Default 0.4. */
  maxDisplacement: number;
  /** How much of the flow's motion the content takes: 1 follows the fluid exactly. Default 1. */
  gain: number;
}

export const DEFAULT_FLUID_WARP_PARAMS: Readonly<FluidWarpParams> = {
  relaxation: 1,
  maxDisplacement: 0.4,
  gain: 1,
};

export interface FluidWarpOptions {
  device: GPUDevice;
  fluid: Fluid;
  /** Field texels per simulation cell along each axis. Default 2; the field is smooth, 1 to 3 is plenty. */
  scale?: number;
  params?: Partial<FluidWarpParams>;
  /** Receives one timestamp slot per field step (`warp field`). */
  profiler?: PassProfiler;
  label?: string;
}

/** The field's texture and which of the pair it is (cache bind groups by `slot` and `generation`). */
export interface FluidWarpField extends FluidTexture {
  readonly width: number;
  readonly height: number;
}

export interface FluidWarp {
  /** Live parameters. Mutate freely; they are read on every step. */
  readonly params: FluidWarpParams;
  /** The current displacement field (image heights, xy). Changes texture after every step. */
  readonly field: FluidWarpField;
  /** Bumps whenever the field's textures are re-created (the fluid's grid changed, or `setScale`). */
  readonly generation: number;
  /** Field texels per solver cell along each axis. */
  readonly scale: number;
  /**
   * Installs the field step into `fluid.hooks.dye`, keeping any hook that was already there (it runs
   * after the field step). Idempotent.
   */
  attach(): void;
  /** Removes the hook and restores the previous one. */
  detach(): void;
  /**
   * Advances the field one step of `dt` seconds with this velocity texture (the solver's staggered
   * velocity). `attach` calls it for every solver step. Every call in one submitted command buffer must
   * use the same `dt` and parameters: they go through one uniform buffer.
   */
  step(encoder: GPUCommandEncoder, dt: number, velocity: GPUTextureView): void;
  /** Changes the field's resolution relative to the solver grid; re-creates (and clears) the field. */
  setScale(scale: number): void;
  /** Sets the field to zero (the content jumps back to rest). */
  reset(encoder: GPUCommandEncoder): void;
  destroy(): void;
}

// --- Uniforms ----------------------------------------------------------------------------------------

const WarpParams = defineUniformStruct('WarpParams', {
  grid: 'vec2f',
  aspect: 'f32',
  dt: 'f32',
  decay: 'f32',
  gain: 'f32',
  max_disp: 'f32',
  open: 'u32',
});

const WarpApplyParams = defineUniformStruct('WarpApplyParams', {
  bars: 'vec3f',
  aspect: 'f32',
  fit: 'vec2f',
  media_px: 'vec2f',
  canvas_px: 'vec2f',
  field_size: 'vec2f',
  fade: 'f32',
  chroma: 'f32',
  sheen: 'f32',
  debug_gain: 'f32',
  view: 'u32',
  cubic: 'u32',
});

const clamp = (x: number, lo: number, hi: number) => Math.min(hi, Math.max(lo, x));
const finiteOr = (x: number, fallback: number) => (Number.isFinite(x) ? x : fallback);

let nextId = 1;
const ids = new WeakMap<object, number>();
function idOf(object: object): number {
  let id = ids.get(object);
  if (id === undefined) ids.set(object, (id = nextId++));
  return id;
}

/** Size of the field for a solver grid and a scale, within the device's texture limit. */
export function warpFieldSize(
  grid: { width: number; height: number },
  scale: number,
  maxDimension = 4096,
): { width: number; height: number } {
  const s = clamp(finiteOr(scale, 2), 0.5, 8);
  return {
    width: clamp(Math.round(grid.width * s), 8, maxDimension),
    height: clamp(Math.round(grid.height * s), 8, maxDimension),
  };
}

// --- The field ----------------------------------------------------------------------------------------

export function createFluidWarp(options: FluidWarpOptions): FluidWarp {
  const { device, fluid, profiler = noProfiler, label = 'fluid-warp' } = options;
  const params: FluidWarpParams = { ...DEFAULT_FLUID_WARP_PARAMS, ...options.params };
  let scale = clamp(finiteOr(options.scale ?? 2, 2), 0.5, 8);
  const maxDimension = Math.min(device.limits.maxTextureDimension2D, 4096);

  const uniforms = createUniformBuffer(device, WarpParams, `${label} params`);
  const sampler = device.createSampler({
    label: `${label} sampler`,
    magFilter: 'linear',
    minFilter: 'linear',
    addressModeU: 'clamp-to-edge',
    addressModeV: 'clamp-to-edge',
  });
  const pipeline = createFullscreenPipeline(device, {
    fragment: {
      label: 'sim/fluid-warp-advect.wgsl',
      code: advectSource.code,
      chunks: { ...advectSource.chunks, 'warp_params.wgsl': WarpParams.wgsl },
    },
    format: WARP_FORMAT,
  });

  const usage =
    GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC;
  interface Slot {
    texture: GPUTexture;
    view: GPUTextureView;
  }
  let slots: [Slot, Slot] | undefined;
  let size = { width: 0, height: 0 };
  let current = 0;
  let generation = 0;
  let fluidGeneration = -1;
  let groups = new Map<string, GPUBindGroup>();
  let destroyed = false;
  let hookInstalled: Fluid['hooks']['dye'];
  let previousHook: Fluid['hooks']['dye'];

  function release() {
    if (!slots) return;
    for (const slot of slots) slot.texture.destroy();
    slots = undefined;
    groups = new Map();
  }

  function allocate() {
    release();
    size = warpFieldSize(fluid.grid, scale, maxDimension);
    slots = [0, 1].map((i) => {
      const texture = device.createTexture({
        label: `${label} field ${i}`,
        size: [size.width, size.height],
        format: WARP_FORMAT,
        usage,
      });
      return { texture, view: texture.createView({ label: `${label} field ${i} view` }) };
    }) as [Slot, Slot];
    current = 0;
    fluidGeneration = fluid.generation;
    generation++;
  }

  allocate();
  // A fresh texture is zero, which is the rest state; nothing to clear.

  function ensure() {
    const wanted = warpFieldSize(fluid.grid, scale, maxDimension);
    if (
      fluidGeneration !== fluid.generation ||
      wanted.width !== size.width ||
      wanted.height !== size.height
    ) {
      allocate();
    }
  }

  function writeUniforms(dt: number) {
    const grid = fluid.grid;
    uniforms.set({
      grid: [grid.width, grid.height],
      aspect: grid.aspect,
      dt,
      decay: Math.exp(-Math.max(0, finiteOr(params.relaxation, 0)) * dt),
      gain: finiteOr(params.gain, 1),
      max_disp: Math.max(1e-4, finiteOr(params.maxDisplacement, 0.4)),
      open: fluid.params.boundary === 'open' ? 1 : 0,
    });
    uniforms.upload();
  }

  const warp: FluidWarp = {
    params,
    get field() {
      const slot = slots![current]!;
      return {
        texture: slot.texture,
        view: slot.view,
        slot: current,
        width: size.width,
        height: size.height,
      };
    },
    get generation() {
      return generation;
    },
    get scale() {
      return scale;
    },
    attach() {
      if (hookInstalled) return;
      previousHook = fluid.hooks.dye;
      hookInstalled = (context) => {
        warp.step(context.encoder, context.dt, context.velocity);
        return previousHook ? previousHook(context) : false;
      };
      fluid.hooks.dye = hookInstalled;
    },
    detach() {
      if (!hookInstalled) return;
      if (fluid.hooks.dye === hookInstalled) fluid.hooks.dye = previousHook;
      hookInstalled = undefined;
      previousHook = undefined;
    },
    step(encoder, dt, velocity) {
      if (destroyed || !(dt > 0)) return;
      ensure();
      writeUniforms(dt);
      const from = slots![current]!;
      const to = slots![1 - current]!;
      const key = `${current}:${idOf(velocity)}`;
      let group = groups.get(key);
      if (!group) {
        group = device.createBindGroup({
          label: `${label} step`,
          layout: pipeline.getBindGroupLayout(0),
          entries: [
            { binding: 0, resource: { buffer: uniforms.buffer } },
            { binding: 1, resource: from.view },
            { binding: 2, resource: velocity },
            { binding: 3, resource: sampler },
          ],
        });
        groups.set(key, group);
      }
      renderFullscreen(encoder, to.view, pipeline, [group], {
        label: `${label} step`,
        timestampWrites: profiler.pass('warp field'),
      });
      current = 1 - current;
    },
    setScale(next) {
      const value = clamp(finiteOr(next, scale), 0.5, 8);
      if (value === scale) return;
      scale = value;
      allocate();
    },
    reset(encoder) {
      ensure();
      for (const slot of slots!) {
        encoder
          .beginRenderPass({
            label: `${label} clear`,
            colorAttachments: [{ view: slot.view, loadOp: 'clear', storeOp: 'store' }],
          })
          .end();
      }
      current = 0;
    },
    destroy() {
      if (destroyed) return;
      warp.detach();
      destroyed = true;
      release();
      uniforms.destroy();
    },
  };
  return warp;
}

// --- Drawing --------------------------------------------------------------------------------------------

export type WarpView = 'content' | 'displacement' | 'grid' | 'velocity';
const VIEW_INDEX: Record<WarpView, number> = {
  content: 0,
  displacement: 1,
  grid: 2,
  velocity: 3,
};

/** Content to draw: a texture view (with its mip chain) and its size in texels. */
export interface WarpContent {
  readonly view: GPUTextureView;
  readonly width: number;
  readonly height: number;
}

export interface WarpDrawOptions {
  warp: FluidWarp;
  content: WarpContent;
  /** How the content fills the canvas: `cover`, `contain` (CSS object-fit), or `stretch` to take the canvas shape (a text texture rendered at canvas size). Default `cover`. */
  fit?: MediaFit | 'stretch';
  /** The canvas size in pixels (picks the mip level, and the aspect). */
  canvasSize: readonly [number, number];
  /** Colour (linear rgb) of the bars of a `contain` fit. Default black. */
  bars?: readonly [number, number, number];
  /** Border fade in uv, see fwarp_source_uv. Default 0.04. */
  fade?: number;
  /** Channel split in proportion to local stretch. 0 (default) takes a single tap. */
  chroma?: number;
  /** Strength of the liquid film shading. 0 (default) is off. */
  sheen?: number;
  view?: WarpView;
  /** Brightness of the debug views. Default 1. */
  debugGain?: number;
  /** Smooth the field with a cubic B-spline (4 taps) instead of bilinear. Default true. */
  cubic?: boolean;
  /** The solver's `fieldsView`, for the `velocity` view. */
  fields?: GPUTextureView;
  timestampWrites?: GPURenderPassTimestampWrites;
}

export interface FluidWarpPresenter {
  /** Draws the content through the field into `target` (a view in `format`), as sRGB. */
  draw(encoder: GPUCommandEncoder, target: GPUTextureView, options: WarpDrawOptions): void;
  destroy(): void;
}

export interface FluidWarpPresenterOptions {
  device: GPUDevice;
  /** Format of the targets passed to `draw`. */
  format: GPUTextureFormat;
  label?: string;
}

export function createFluidWarpPresenter(options: FluidWarpPresenterOptions): FluidWarpPresenter {
  const { device, format, label = 'fluid-warp view' } = options;
  const uniforms = createUniformBuffer(device, WarpApplyParams, `${label} params`);
  const pipeline = createFullscreenPipeline(device, {
    fragment: {
      label: 'sim/fluid-warp-apply.wgsl',
      code: applySource.code,
      chunks: {
        ...applySource.chunks,
        'warp_apply_params.wgsl': WarpApplyParams.wgsl,
        'fluid-warp-core.wgsl': coreSource.code,
        ...coreSource.chunks,
      },
    },
    format,
  });
  const contentSampler = device.createSampler({
    label: `${label} content sampler`,
    magFilter: 'linear',
    minFilter: 'linear',
    mipmapFilter: 'linear',
    addressModeU: 'clamp-to-edge',
    addressModeV: 'clamp-to-edge',
  });
  const fieldSampler = device.createSampler({
    label: `${label} field sampler`,
    magFilter: 'linear',
    minFilter: 'linear',
    addressModeU: 'clamp-to-edge',
    addressModeV: 'clamp-to-edge',
  });
  const blank = device.createTexture({
    label: `${label} blank`,
    size: [1, 1],
    format: 'rgba16float',
    usage: GPUTextureUsage.TEXTURE_BINDING,
  });
  const blankView = blank.createView();
  let groups = new Map<string, GPUBindGroup>();
  let destroyed = false;

  return {
    draw(encoder, target, o) {
      if (destroyed) return;
      const [cw, ch] = [Math.max(1, o.canvasSize[0]), Math.max(1, o.canvasSize[1])];
      const aspect = cw / ch;
      const fitMode = o.fit ?? 'cover';
      const fit: [number, number] =
        fitMode === 'stretch'
          ? [1, 1]
          : fitScale(aspect, o.content.width / Math.max(1, o.content.height), fitMode);
      const field = o.warp.field;
      uniforms.set({
        bars: o.bars ?? [0, 0, 0],
        aspect,
        fit,
        media_px: [o.content.width, o.content.height],
        canvas_px: [cw, ch],
        field_size: [field.width, field.height],
        fade: o.fade ?? 0.04,
        chroma: Math.max(0, o.chroma ?? 0),
        sheen: Math.max(0, o.sheen ?? 0),
        debug_gain: o.debugGain ?? 1,
        view: VIEW_INDEX[o.view ?? 'content'],
        cubic: o.cubic === false ? 0 : 1,
      });
      uniforms.upload();

      const fieldsView = o.fields ?? blankView;
      const key = `${idOf(o.warp)}:${o.warp.generation}:${field.slot}:${idOf(o.content.view)}:${idOf(fieldsView)}`;
      let group = groups.get(key);
      if (!group) {
        if (groups.size > 64) groups = new Map();
        group = device.createBindGroup({
          label,
          layout: pipeline.getBindGroupLayout(0),
          entries: [
            { binding: 0, resource: { buffer: uniforms.buffer } },
            { binding: 1, resource: o.content.view },
            { binding: 2, resource: contentSampler },
            { binding: 3, resource: field.view },
            { binding: 4, resource: fieldSampler },
            { binding: 5, resource: fieldsView },
          ],
        });
        groups.set(key, group);
      }
      renderFullscreen(encoder, target, pipeline, [group], {
        label,
        timestampWrites: o.timestampWrites,
      });
    },
    destroy() {
      if (destroyed) return;
      destroyed = true;
      groups = new Map();
      uniforms.destroy();
      blank.destroy();
    },
  };
}
