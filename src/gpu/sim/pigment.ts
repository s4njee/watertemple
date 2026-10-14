import { noProfiler, type PassProfiler } from '../profiler.ts';
import { createShader } from '../shader.ts';
import { createUniformBuffer, defineUniformStruct } from '../uniforms.ts';
import type { Fluid } from './fluid.ts';
import flowSource from './pigment-flow.wgsl?raw';
import stepSource from './pigment-step.wgsl?raw';
import waterSource from './pigment-water.wgsl?raw';

// The pigment layer (story 6.6): what ink does in water and on paper, as passes on the dye grid of the
// stable-fluids solver (fluid.ts). The solver advects the pigment; this adds
//
//   diffusion      the pigment spreads with a real diffusivity, wetter paper spreading it faster
//   edge drift     where a film of water thins towards its edge, pigment is drawn out to it (the dark
//                  rim of a dried wash)
//   settling       pigment leaves the water for the paper, into a second layer that is NOT advected
//                  (the stain), faster on dry paper and in the valleys of the paper (granulation),
//                  and dissolves back into water that arrives later (back-runs)
//   wetness        the water itself: a field the brush adds to, that spreads a little and dries
//   dry drag       dry paper holds the flow back (a pass for the solver's force hook)
//
// The pigment is carried as OPTICAL DENSITY (absorbance) per channel in rgb of the solver's dye texture
// (a = its mean, for buoyancy), the stain likewise in rgb of its own texture; densities add, so inks mix
// subtractively once the shading multiplies the paper by exp(-density) (fluid.wgsl's
// fluid_transmittance). pigment-step.wgsl documents every rule and the numerical scheme.
//
// STATE (all dye sized, ping-ponged by this module, never read-write):
//   stain  rgba16float  rgb = deposited density, a = shaped wetness (for shading)
//   wet    r32float     the water film, 0..1; 32 bit because drying moves it by less than a half
//                       float can represent; textureLoad only
//
// USE. `attach(fluid)` installs the two hooks (dye and forces) and the layer then runs inside
// `fluid.update`; call `beginFrame(encoder)` once per frame before it to apply the water the brush
// queued with `addWater`. Or drive it by hand with `step` on any pair of dye textures (the tests do).
// One parameter set per submitted command buffer: the uniform buffer is written with queue.writeBuffer.
//
// UNITS as in the solver: lengths in image heights (the canvas height is 1), time in seconds.
//
// STABILITY. Diffusion and drift are explicit and conservative; per face the diffusion fraction is
// capped at W_CAP and the drift at EDGE_CAP texels per step, so a texel never gives away more than it
// holds. A diffusivity that would need a larger fraction is split into up to `maxSubsteps` equal
// sub-steps (each a dispatch); beyond that it saturates (the cap), it does not blow up.

/** Largest fraction of a texel's pigment one face may exchange per sub-step. Four faces stay < 1 with the drift. */
export const W_CAP = 0.12;
/** Largest drift, texels per sub-step, across one face. */
export const EDGE_CAP = 0.12;
const DEP_FLOOR = 1.5e-3;
const WET_LO = 0.03;
const WET_HI = 0.3;
/** Wetness change per texel at which the edge drift reaches full speed (it starts at 0.3 of this). */
const EDGE_G0 = 0.02;
const WORKGROUP = 16;
/** Water splats accepted between two `beginFrame` calls; more are dropped. */
export const MAX_WATER_SPLATS = 64;
const SPLAT_FLOATS = 12;

export const PIGMENT_FORMAT = 'rgba16float' as const;
export const WET_FORMAT: GPUTextureFormat = 'r32float';

/** Live, mutable parameters: change them any time, they are read on every step. */
export interface PigmentParams {
  /** Diffusivity of the pigment in a wet film, image heights squared per second. 0 is off. */
  diffusion: number;
  /** The fraction of `diffusion` that works on dry paper, 0 to 1. */
  dryDiffusion: number;
  /** How much the paper's fibres and grain modulate diffusion (feathering), 0 to 2. */
  feather: number;
  /** Speed of the pigment drawn out to a thinning film's edge, image heights per second. 0 is off. */
  edgeSpeed: number;
  /** Rate at which suspended pigment is taken up by dry paper, 1 / s. 0 never stains. */
  absorption: number;
  /** The fraction of `absorption` that works under a wet film, 0 to 1. */
  wetAbsorption: number;
  /** How much deeper the paper's valleys take up pigment than its peaks, 0 to 1. */
  granulation: number;
  /** Rate at which stain dissolves back into a wet film, 1 / s. 0 never lifts. */
  lift: number;
  /** Drying rate of the film, 1 / s (an exponential, plus a small linear part so it finishes). */
  drying: number;
  /** Capillary spreading of the film itself, image heights squared per second. */
  wetSpread: number;
  /** Wetness the sheet never falls under: 1 is paper under water (ink in a pool), 0 is dry paper. */
  wetFloor: number;
  /** Rate at which dry paper stops the flow, 1 / s (see pigment-flow.wgsl). 0 lets it flow everywhere. */
  dryDrag: number;
  /** Washing out of suspended and deposited pigment alike, 1 / s. 0 keeps everything. */
  fade: number;
  /** How unevenly the paper takes water, 0 to 1: the outline of a wetted patch is ragged (see pigment-water.wgsl). */
  ragged: number;
  /** Most sub-steps one step may split the diffusion into (1 to 8). */
  maxSubsteps: number;
  /** Largest density in either layer. */
  dyeMax: number;
}

export const DEFAULT_PIGMENT_PARAMS: Readonly<PigmentParams> = {
  diffusion: 6e-5,
  dryDiffusion: 0.03,
  feather: 0.5,
  edgeSpeed: 0.006,
  absorption: 0.6,
  wetAbsorption: 0.2,
  granulation: 0.4,
  lift: 0.2,
  drying: 0.08,
  wetSpread: 2e-5,
  wetFloor: 0,
  dryDrag: 14,
  fade: 0,
  ragged: 0,
  maxSubsteps: 3,
  dyeMax: 64,
};

const PigmentUniforms = defineUniformStruct('PigmentUniforms', {
  size: 'vec2u',
  grid: 'vec2u',
  dt: 'f32',
  w_diff: 'f32',
  w_cap: 'f32',
  feather: 'f32',
  dry_diff: 'f32',
  edge_v: 'f32',
  edge_g0: 'f32',
  absorb: 'f32',
  wet_stick: 'f32',
  gran: 'f32',
  lift: 'f32',
  keep: 'f32',
  dep_floor: 'f32',
  wet_lo: 'f32',
  wet_hi: 'f32',
  wet_spread: 'f32',
  dry_decay: 'f32',
  dry_lin: 'f32',
  wet_floor: 'f32',
  drag: 'f32',
  dye_max: 'f32',
});

const WaterUniforms = defineUniformStruct('WaterUniforms', {
  size: 'vec2u',
  count: 'u32',
  aspect: 'f32',
  ragged: 'f32',
});

const clamp = (x: number, lo: number, hi: number) => Math.min(hi, Math.max(lo, x));
const finiteOr = (x: number, fallback: number) => (Number.isFinite(x) ? x : fallback);
const nonNegative = (x: number) => Math.max(0, finiteOr(x, 0));

/**
 * Sub-steps a step of `dt` seconds splits the diffusion into on a grid `height` texels tall: enough
 * that the diffusion fraction per face, D dt height^2 / n, stays under `W_CAP`, at most `maxSubsteps`.
 */
export function pigmentSubsteps(
  params: Pick<PigmentParams, 'diffusion' | 'maxSubsteps'>,
  dt: number,
  height: number,
): number {
  const fraction = nonNegative(params.diffusion) * nonNegative(dt) * height * height;
  const most = Math.round(clamp(finiteOr(params.maxSubsteps, 3), 1, 8));
  return clamp(Math.ceil(fraction / W_CAP), 1, most);
}

/** The diffusion fraction per face and sub-step (before wetness and paper scale it) a step uses. */
export function pigmentDiffusionFraction(
  params: Pick<PigmentParams, 'diffusion' | 'maxSubsteps'>,
  dt: number,
  height: number,
): number {
  const n = pigmentSubsteps(params, dt, height);
  return (nonNegative(params.diffusion) * nonNegative(dt) * height * height) / n;
}

export interface PigmentTexture {
  readonly texture: GPUTexture;
  readonly view: GPUTextureView;
  /** Which of the two ping-pong textures this is; cache bind groups by it, drop them when `generation` changes. */
  readonly slot: number;
}

export interface PigmentStepContext {
  /** Seconds this step advances. */
  dt: number;
  /** Suspended pigment as it is now (sampled). */
  read: GPUTextureView;
  /** Write-only storage, rgba16float, dye sized: written in full. */
  write: GPUTextureView;
  /** The solver's velocity grid in cells, for the flow pass. Default: the dye grid. */
  grid?: { width: number; height: number };
}

export interface PigmentFlowContext {
  encoder: GPUCommandEncoder;
  dt: number;
  grid: { width: number; height: number };
  /** The staggered velocity (see fluid.ts). */
  read: GPUTextureView;
  /** Write-only storage of the same size. */
  write: GPUTextureView;
}

export interface PigmentLayerOptions {
  device: GPUDevice;
  /** Dye texels across and down. */
  width: number;
  height: number;
  params?: Partial<PigmentParams>;
  /**
   * Storage format of the dye textures this layer is stepped on and of its own stain. The solver's
   * dye is `rgba16float` (the default). `rgba32float` (not filterable: textureLoad only) exists for
   * the tests: 16-bit stores round towards zero on the development machine, which loses about
   * 0.03 percent of a changing texel per step, and only a 32-bit run shows the algebra exactly.
   */
  format?: 'rgba16float' | 'rgba32float';
  profiler?: PassProfiler;
  label?: string;
}

export interface PigmentLayer {
  readonly params: PigmentParams;
  readonly width: number;
  readonly height: number;
  /** Increments when the textures are re-created. */
  readonly generation: number;
  /** Storage format of the dye this layer is stepped on and of its stain. */
  readonly format: 'rgba16float' | 'rgba32float';
  /** The deposited pigment, current. Changes after every step. */
  readonly stain: PigmentTexture;
  /** The water film, current (r32float, textureLoad only). Changes after every step. */
  readonly wet: PigmentTexture;
  /** Sub-steps the last step used. */
  readonly lastSubsteps: number;

  /** The paper (rgba8unorm: r grain, g fibre; repeating, one texel per dye texel). Null restores the flat default. */
  setPaper(view: GPUTextureView | null): void;
  /** Queues water along a segment (uv), Gaussian footprint `radius` image heights, `ramp` as in the solver's splats. */
  addWater(
    from: readonly [number, number],
    to: readonly [number, number],
    radius: number,
    amount: number,
    ramp?: number,
  ): boolean;
  /** Applies the queued water and arms the profiler for this frame's first step. Once per frame, before `fluid.update`. */
  beginFrame(encoder: GPUCommandEncoder): void;
  /** One step on the dye: transport, settle, wetness. `write` receives the new suspended pigment. */
  step(encoder: GPUCommandEncoder, context: PigmentStepContext): void;
  /** Damps a velocity field where the paper is dry. False (and nothing written) when there is nothing to damp. */
  dampFlow(context: PigmentFlowContext): boolean;
  /** Installs `hooks.dye` and `hooks.forces` on the solver. */
  attach(fluid: Fluid): void;
  /** Removes the hooks `attach` installed. */
  detach(fluid: Fluid): void;
  /** Changes the grid; true if the textures were re-created (all state cleared). */
  resize(width: number, height: number): boolean;
  /** Clears stain and wetness (the wetness to its floor). */
  reset(encoder: GPUCommandEncoder): void;
  destroy(): void;
}

// --- Pipelines, cached per device ---------------------------------------------------------------------

type PassName = 'transport' | 'step' | 'water' | 'flow';

interface Pipeline {
  pipeline: GPUComputePipeline;
  layout: GPUBindGroupLayout;
}

const COMPUTE = GPUShaderStage.COMPUTE;
const sampled = (binding: number, sampleType: GPUTextureSampleType): GPUBindGroupLayoutEntry => ({
  binding,
  visibility: COMPUTE,
  texture: { sampleType },
});
const storage = (binding: number, format: GPUTextureFormat): GPUBindGroupLayoutEntry => ({
  binding,
  visibility: COMPUTE,
  storageTexture: { access: 'write-only', format },
});
const uniformEntry = (binding: number): GPUBindGroupLayoutEntry => ({
  binding,
  visibility: COMPUTE,
  buffer: { type: 'uniform' },
});

type Format = 'rgba16float' | 'rgba32float';

function layoutEntries(name: 'step' | 'water' | 'flow', format: Format): GPUBindGroupLayoutEntry[] {
  // rgba32float cannot be filtered without a feature; every pass here only loads, so say so.
  const dye: GPUTextureSampleType = format === 'rgba16float' ? 'float' : 'unfilterable-float';
  switch (name) {
    case 'step':
      return [
        uniformEntry(0),
        sampled(1, dye),
        sampled(2, dye),
        sampled(3, 'unfilterable-float'),
        sampled(4, 'float'),
        storage(5, format),
        storage(6, format),
        storage(7, WET_FORMAT),
      ];
    case 'water':
      return [
        uniformEntry(0),
        sampled(1, 'unfilterable-float'),
        { binding: 2, visibility: COMPUTE, buffer: { type: 'read-only-storage' } },
        storage(3, WET_FORMAT),
      ];
    case 'flow':
      return [
        uniformEntry(0),
        sampled(1, 'float'),
        sampled(2, 'unfilterable-float'),
        storage(3, PIGMENT_FORMAT),
      ];
  }
}

const PASSES: Record<
  PassName,
  { source: 'step' | 'water' | 'flow'; entry: string; layout: 'step' | 'water' | 'flow' }
> = {
  transport: { source: 'step', entry: 'cs_transport', layout: 'step' },
  step: { source: 'step', entry: 'cs_step', layout: 'step' },
  water: { source: 'water', entry: 'cs_water', layout: 'water' },
  flow: { source: 'flow', entry: 'cs_flow', layout: 'flow' },
};

const pipelineCache = new WeakMap<GPUDevice, Map<string, Pipeline>>();
const moduleCache = new WeakMap<GPUDevice, Map<string, GPUShaderModule>>();
const layoutCache = new WeakMap<GPUDevice, Map<string, GPUBindGroupLayout>>();

function getLayout(
  device: GPUDevice,
  name: 'step' | 'water' | 'flow',
  format: Format,
): GPUBindGroupLayout {
  let layouts = layoutCache.get(device);
  if (!layouts) layoutCache.set(device, (layouts = new Map()));
  const key = `${name}:${format}`;
  let layout = layouts.get(key);
  if (!layout) {
    layout = device.createBindGroupLayout({
      label: `pigment ${key} layout`,
      entries: layoutEntries(name, format),
    });
    layouts.set(key, layout);
  }
  return layout;
}

function getModule(
  device: GPUDevice,
  source: 'step' | 'water' | 'flow',
  format: Format,
): GPUShaderModule {
  let modules = moduleCache.get(device);
  if (!modules) moduleCache.set(device, (modules = new Map()));
  const key = `${source}:${format}`;
  let module = modules.get(key);
  if (!module) {
    // The step shader is written for rgba16float; the 32-bit variant is the same text with the format swapped.
    const code = { step: stepSource, water: waterSource, flow: flowSource }[source];
    module = createShader(device, {
      label: `sim/pigment-${source}.wgsl${format === 'rgba16float' ? '' : ` (${format})`}`,
      code: source === 'step' ? code.replaceAll('rgba16float', format) : code,
      chunks: {
        'pigment_uniforms.wgsl': PigmentUniforms.wgsl,
        'water_uniforms.wgsl': WaterUniforms.wgsl,
      },
    });
    modules.set(key, module);
  }
  return module;
}

function getPipeline(device: GPUDevice, name: PassName, format: Format): Pipeline {
  let pipelines = pipelineCache.get(device);
  if (!pipelines) pipelineCache.set(device, (pipelines = new Map()));
  const spec = PASSES[name];
  // Only the step shader depends on the format.
  const fmt: Format = spec.source === 'step' ? format : 'rgba16float';
  const key = `${name}:${fmt}`;
  let found = pipelines.get(key);
  if (!found) {
    const layout = getLayout(device, spec.layout, fmt);
    found = {
      layout,
      pipeline: device.createComputePipeline({
        label: `pigment ${spec.entry} ${fmt}`,
        layout: device.createPipelineLayout({ bindGroupLayouts: [layout] }),
        compute: { module: getModule(device, spec.source, fmt), entryPoint: spec.entry },
      }),
    };
    pipelines.set(key, found);
  }
  return found;
}

let nextId = 1;
const ids = new WeakMap<object, number>();
function idOf(object: object): number {
  let id = ids.get(object);
  if (id === undefined) ids.set(object, (id = nextId++));
  return id;
}

interface Tex {
  texture: GPUTexture;
  view: GPUTextureView;
}

export function createPigmentLayer(options: PigmentLayerOptions): PigmentLayer {
  const { device, profiler = noProfiler, label = 'pigment', format = PIGMENT_FORMAT } = options;
  const params: PigmentParams = { ...DEFAULT_PIGMENT_PARAMS, ...options.params };
  let width = Math.max(1, Math.round(options.width));
  let height = Math.max(1, Math.round(options.height));
  let destroyed = false;
  let generation = 0;
  let timed = false;
  let lastSubsteps = 1;
  let gridCells: [number, number] = [width, height];

  const uniforms = createUniformBuffer(device, PigmentUniforms, `${label} params`);
  const waterUniforms = createUniformBuffer(device, WaterUniforms, `${label} water params`);
  const waterBuffer = device.createBuffer({
    label: `${label} water splats`,
    size: MAX_WATER_SPLATS * SPLAT_FLOATS * 4,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });
  const waterQueue = new Float32Array(MAX_WATER_SPLATS * SPLAT_FLOATS);
  let waterCount = 0;

  const usage =
    GPUTextureUsage.TEXTURE_BINDING |
    GPUTextureUsage.STORAGE_BINDING |
    GPUTextureUsage.RENDER_ATTACHMENT |
    GPUTextureUsage.COPY_SRC |
    GPUTextureUsage.COPY_DST;

  function make(name: string, format: GPUTextureFormat): Tex {
    const texture = device.createTexture({
      label: `${label} ${name}`,
      size: [width, height],
      format,
      usage,
    });
    return { texture, view: texture.createView({ label: `${label} ${name} view` }) };
  }

  // The flat paper used until the effect supplies a baked one: mean grain, no fibre.
  const flatPaper = device.createTexture({
    label: `${label} flat paper`,
    size: [1, 1],
    format: 'rgba8unorm',
    usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
  });
  device.queue.writeTexture({ texture: flatPaper }, new Uint8Array([128, 0, 0, 255]), {}, [1, 1]);
  const flatPaperView = flatPaper.createView();
  let paper: GPUTextureView = flatPaperView;

  let stain: [Tex, Tex];
  let wet: [Tex, Tex];
  let scratch: Tex[] = [];
  let stainCurrent = 0;
  let wetCurrent = 0;
  const groups = new Map<string, GPUBindGroup>();

  function allocate() {
    stain = [make('stain 0', format), make('stain 1', format)];
    wet = [make('wetness 0', WET_FORMAT), make('wetness 1', WET_FORMAT)];
    scratch = [];
    stainCurrent = 0;
    wetCurrent = 0;
    groups.clear();
    generation++;
  }
  function release() {
    for (const t of [...stain, ...wet, ...scratch]) t.texture.destroy();
  }
  allocate();

  const view = (t: Tex) => t.view;

  function ensureScratch(count: number) {
    while (scratch.length < count) scratch.push(make(`scratch ${scratch.length}`, format));
  }

  // --- Uniforms --------------------------------------------------------------------------------------------

  function writeUniforms(dt: number, substeps: number) {
    const D = clamp(nonNegative(params.diffusion), 0, 0.05);
    const cells = height * height;
    const subDt = dt / substeps;
    const drying = nonNegative(params.drying);
    uniforms.set({
      size: [width, height],
      grid: [Math.max(1, Math.round(gridCells[0])), Math.max(1, Math.round(gridCells[1]))],
      dt,
      w_diff: D * subDt * cells,
      w_cap: W_CAP,
      feather: clamp(finiteOr(params.feather, 0), 0, 2),
      dry_diff: clamp(finiteOr(params.dryDiffusion, 0), 0, 1),
      edge_v: Math.min(nonNegative(params.edgeSpeed) * subDt * height, EDGE_CAP),
      edge_g0: EDGE_G0,
      absorb: nonNegative(params.absorption),
      wet_stick: clamp(finiteOr(params.wetAbsorption, 0), 0, 1),
      gran: clamp(finiteOr(params.granulation, 0), 0, 1),
      lift: nonNegative(params.lift),
      keep: Math.exp(-nonNegative(params.fade) * dt),
      dep_floor: DEP_FLOOR,
      wet_lo: WET_LO,
      wet_hi: WET_HI,
      wet_spread: Math.min(nonNegative(params.wetSpread) * dt * cells, 0.2),
      dry_decay: Math.exp(-drying * dt),
      dry_lin: 0.04 * drying * dt,
      wet_floor: clamp(finiteOr(params.wetFloor, 0), 0, 1),
      drag: nonNegative(params.dryDrag),
      dye_max: clamp(finiteOr(params.dyeMax, 64), 1, 256),
    });
    uniforms.upload();
  }

  // --- Bind groups and dispatch --------------------------------------------------------------------------

  function stepGroup(
    name: 'transport' | 'step',
    dyeIn: GPUTextureView,
    dyeOut: GPUTextureView,
  ): GPUBindGroup {
    const sIn = stain[stainCurrent]!;
    const sOut = stain[1 - stainCurrent]!;
    const wIn = wet[wetCurrent]!;
    const wOut = wet[1 - wetCurrent]!;
    const key = [name, idOf(dyeIn), idOf(dyeOut), stainCurrent, wetCurrent, idOf(paper)].join('|');
    let group = groups.get(key);
    if (!group) {
      group = device.createBindGroup({
        label: `pigment ${name}`,
        layout: getPipeline(device, name, format).layout,
        entries: [
          { binding: 0, resource: { buffer: uniforms.buffer } },
          { binding: 1, resource: dyeIn },
          { binding: 2, resource: view(sIn) },
          { binding: 3, resource: view(wIn) },
          { binding: 4, resource: paper },
          { binding: 5, resource: dyeOut },
          { binding: 6, resource: view(sOut) },
          { binding: 7, resource: view(wOut) },
        ],
      });
      groups.set(key, group);
    }
    return group;
  }

  function dispatchOver(
    pass: GPUComputePassEncoder,
    name: PassName,
    group: GPUBindGroup,
    w: number,
    h: number,
  ) {
    pass.setPipeline(getPipeline(device, name, format).pipeline);
    pass.setBindGroup(0, group);
    pass.dispatchWorkgroups(Math.ceil(w / WORKGROUP), Math.ceil(h / WORKGROUP));
  }

  // --- The layer ----------------------------------------------------------------------------------------------

  const layer: PigmentLayer = {
    params,
    get width() {
      return width;
    },
    get height() {
      return height;
    },
    get generation() {
      return generation;
    },
    format,
    get stain() {
      return {
        texture: stain[stainCurrent]!.texture,
        view: stain[stainCurrent]!.view,
        slot: stainCurrent,
      };
    },
    get wet() {
      return { texture: wet[wetCurrent]!.texture, view: wet[wetCurrent]!.view, slot: wetCurrent };
    },
    get lastSubsteps() {
      return lastSubsteps;
    },

    setPaper(next) {
      paper = next ?? flatPaperView;
      groups.clear();
    },

    addWater(from, to, radius, amount, ramp = 0) {
      if (waterCount >= MAX_WATER_SPLATS) return false;
      const numbers = [from[0], from[1], to[0], to[1], radius, ramp, amount];
      if (!numbers.every(Number.isFinite)) return false;
      const o = waterCount * SPLAT_FLOATS;
      waterQueue.set([from[0], from[1], to[0], to[1]], o);
      waterQueue.set([Math.max(radius, 1e-4), Math.max(ramp, 0), 0, 0], o + 4);
      waterQueue.set([amount, 0, 0, 0], o + 8);
      waterCount++;
      return true;
    },

    beginFrame(encoder) {
      timed = true;
      if (destroyed || waterCount === 0) return;
      device.queue.writeBuffer(waterBuffer, 0, waterQueue, 0, waterCount * SPLAT_FLOATS);
      waterUniforms.set({
        size: [width, height],
        count: waterCount,
        aspect: width / height,
        ragged: clamp(finiteOr(params.ragged, 0), 0, 1),
      });
      waterUniforms.upload();
      const group = device.createBindGroup({
        label: 'pigment water',
        layout: getPipeline(device, 'water', format).layout,
        entries: [
          { binding: 0, resource: { buffer: waterUniforms.buffer } },
          { binding: 1, resource: view(wet[wetCurrent]!) },
          { binding: 2, resource: { buffer: waterBuffer } },
          { binding: 3, resource: view(wet[1 - wetCurrent]!) },
        ],
      });
      const pass = encoder.beginComputePass({ label: 'pigment water' });
      dispatchOver(pass, 'water', group, width, height);
      pass.end();
      wetCurrent = 1 - wetCurrent;
      waterCount = 0;
    },

    step(encoder, context) {
      if (destroyed) return;
      const dt = clamp(finiteOr(context.dt, 1 / 60), 1e-4, 0.1);
      if (context.grid) gridCells = [context.grid.width, context.grid.height];
      const n = pigmentSubsteps(params, dt, height);
      lastSubsteps = n;
      writeUniforms(dt, n);
      ensureScratch(n > 1 ? Math.min(2, n - 1) : 0);
      const pass = encoder.beginComputePass({
        label: 'pigment step',
        timestampWrites: timed
          ? (profiler.pass('pigment') as GPUComputePassTimestampWrites | undefined)
          : undefined,
      });
      timed = false;
      let from = context.read;
      for (let i = 0; i < n - 1; i++) {
        const to = view(scratch[i % 2]!);
        dispatchOver(pass, 'transport', stepGroup('transport', from, to), width, height);
        from = to;
      }
      dispatchOver(pass, 'step', stepGroup('step', from, context.write), width, height);
      pass.end();
      stainCurrent = 1 - stainCurrent;
      wetCurrent = 1 - wetCurrent;
    },

    dampFlow(context) {
      if (destroyed || !(params.dryDrag > 0) || params.wetFloor >= 1) return false;
      gridCells = [context.grid.width, context.grid.height];
      const dt = clamp(finiteOr(context.dt, 1 / 60), 1e-4, 0.1);
      writeUniforms(dt, pigmentSubsteps(params, dt, height));
      const key = ['flow', idOf(context.read), idOf(context.write), wetCurrent].join('|');
      let group = groups.get(key);
      if (!group) {
        group = device.createBindGroup({
          label: 'pigment flow',
          layout: getPipeline(device, 'flow', format).layout,
          entries: [
            { binding: 0, resource: { buffer: uniforms.buffer } },
            { binding: 1, resource: context.read },
            { binding: 2, resource: view(wet[wetCurrent]!) },
            { binding: 3, resource: context.write },
          ],
        });
        groups.set(key, group);
      }
      const pass = context.encoder.beginComputePass({ label: 'pigment flow' });
      dispatchOver(pass, 'flow', group, context.grid.width + 1, context.grid.height + 1);
      pass.end();
      return true;
    },

    attach(fluid) {
      fluid.hooks.dye = (context) => {
        if (context.grid.dyeWidth !== width || context.grid.dyeHeight !== height) return false;
        layer.step(context.encoder, {
          dt: context.dt,
          read: context.read,
          write: context.write,
          grid: { width: context.grid.width, height: context.grid.height },
        });
        return true;
      };
      fluid.hooks.forces = (context) =>
        context.grid.dyeWidth === width && context.grid.dyeHeight === height
          ? layer.dampFlow(context)
          : false;
    },

    detach(fluid) {
      fluid.hooks.dye = undefined;
      fluid.hooks.forces = undefined;
    },

    resize(newWidth, newHeight) {
      const w = Math.max(1, Math.round(newWidth));
      const h = Math.max(1, Math.round(newHeight));
      if (destroyed || (w === width && h === height)) return false;
      release();
      width = w;
      height = h;
      waterCount = 0;
      allocate();
      return true;
    },

    reset(encoder) {
      waterCount = 0;
      const floor = clamp(finiteOr(params.wetFloor, 0), 0, 1);
      for (const t of [...stain, ...scratch]) {
        encoder
          .beginRenderPass({
            label: `${label} clear`,
            colorAttachments: [{ view: t.view, loadOp: 'clear', storeOp: 'store' }],
          })
          .end();
      }
      for (const t of wet) {
        encoder
          .beginRenderPass({
            label: `${label} clear wetness`,
            colorAttachments: [
              { view: t.view, loadOp: 'clear', storeOp: 'store', clearValue: [floor, 0, 0, 0] },
            ],
          })
          .end();
      }
      stainCurrent = 0;
      wetCurrent = 0;
    },

    destroy() {
      if (destroyed) return;
      destroyed = true;
      release();
      flatPaper.destroy();
      uniforms.destroy();
      waterUniforms.destroy();
      waterBuffer.destroy();
    },
  };
  return layer;
}
