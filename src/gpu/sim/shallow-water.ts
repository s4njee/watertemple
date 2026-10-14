import { createFixedStepper, type FixedStepper } from '../loop.ts';
import { noProfiler, type PassProfiler } from '../profiler.ts';
import { renderFullscreen } from '../fullscreen.ts';
import { createShader } from '../shader.ts';
import { createUniformBuffer, defineUniformStruct, uniformArray } from '../uniforms.ts';
import computeSource from './shallow-water-compute.wgsl?raw';
import coreSource from './shallow-water-core.wgsl?raw';
import renderSource from './shallow-water-render.wgsl?raw';

// The shallow-water solver (story 6.7): water as a height field WITH velocity, over a terrain. Unlike
// the wave-equation heightfield (heightfield.ts), where a bump only ripples in place, here water is
// mass that MOVES: a dam breaks and the water rushes out, runs down a slope, pools in low ground and
// leaves the high ground dry.
//
// SCHEME. Finite volumes with the HLL Riemann solver and Audusse's hydrostatic reconstruction (see
// shallow-water-core.wgsl for the maths and why this one rather than virtual pipes): well balanced
// (a lake at rest stays at rest over any terrain), positivity preserving (no negative depth at a
// wet/dry front), conservative (water is moved, never made). One pass per step, no flux textures.
//
// STATE. `rgba32float` ping-pong, one texel per cell: (depth h, momentum hu, momentum hv, bed b).
// Why 32 bit when docs/support-matrix.md recommends rgba16float for simulation state: half floats
// cannot conserve mass. A cell's change per step is a small fraction of its depth, below half-float
// resolution (2^-11), so rounding discards it unevenly on the two sides of a face and the volume
// drifts (and a lake at rest over uneven terrain would creep, because b + h is not representable).
// rgba32float is renderable and a write-only storage format in core WebGPU, and the solver only ever
// reads it with textureLoad, so the missing float32-filterable feature does not matter. Consumers
// read the two filterable `rgba16float` outputs below, resolved once per update.
//
// OUTPUTS (both rgba16float, filterable, one texel per cell, uv (0, 0) top-left; decode with
// src/wgsl/shallow-water.wgsl):
//   `view`        (depth m, velocity x m/s, velocity y m/s, foam 0..1)
//   `groundView`  (bed height m, surface slope x, surface slope y, ground wetness 0..1)
//
// UNITS. Metres and seconds, SI gravity by default. Positions given to the API are domain uv (0..1,
// top-left origin, y down, like every texture here); radii are metres. +x is right, +y is down the
// texture, which a 3D consumer maps to world +x and +z.
//
// STABILITY. The timestep is derived, never chosen by the caller: dt = cfl dx / (2 lambda_cap) with
// lambda_cap = 2 sqrt(g maxDepth). The shader clamps every cell to depth <= maxDepth and
// |u| + sqrt(g h) <= lambda_cap, so the fastest signal anywhere is bounded and the CFL condition holds
// whatever happens. The price: water deeper than `maxDepth` is discarded, so set it a little above the
// deepest water you expect (the dam-break demo uses 1.3 x the reservoir depth). A bigger `maxDepth`
// is always safe but costs steps (dt ~ 1 / sqrt(maxDepth)). Hitting the per-frame step cap makes the
// simulation run slower than real time, never unstable.

export type ShallowWaterKind = 'render' | 'compute';
export type EdgeMode = 'wall' | 'open';

/** Format of the solver's state: exact, because mass conservation needs more than half precision. */
export const SHALLOW_WATER_STATE_FORMAT: GPUTextureFormat = 'rgba32float';
/** Format of the two consumer-facing output textures and the foam / wetness history. */
export const SHALLOW_WATER_OUTPUT_FORMAT: GPUTextureFormat = 'rgba16float';
/** Largest CFL number the solver will use. 0.5 is the positivity limit of the 2D HLL update. */
export const COURANT_LIMIT = 0.5;
/** Sources, pushes, sculpts and level zones accepted between two updates; more are dropped. */
export const MAX_SOURCES = 32;
/** Smallest source radius in cells: anything narrower aliases on the grid. */
export const MIN_RADIUS_CELLS = 1.5;
const MAX_RADIUS_CELLS = 64;
// 16 x 16 = 256 invocations, the default per-workgroup maximum.
const WORKGROUP = 16;

/** Live, mutable simulation parameters: change them any time, they are read on every update. */
export interface ShallowWaterParams {
  /** Gravity, m / s^2. Sets the wave speed sqrt(g h) and so the timestep. */
  gravity: number;
  /**
   * Deepest water the solver will hold, metres. Water above it is discarded; the CFL timestep is
   * derived from it (see STABILITY above).
   */
  maxDepth: number;
  /** Manning roughness n (s / m^(1/3)). 0.01 is smooth concrete, 0.03 a natural channel, 0.1 a weedy one. */
  friction: number;
  /** Extra linear drag on the velocity, 1 / s. 0 by default. */
  drag: number;
  /** Rain: metres of water added per second over the whole grid. */
  rain: number;
  /** `wall`: reflecting walls at the grid edge. `open`: water runs off the edge. */
  edges: EdgeMode;
  /** CFL number, clamped to [0.1, COURANT_LIMIT]. Smaller is more accurate and costs more steps. */
  cfl: number;
  /** Simulated seconds per second of wall time. 1 is real time; the step does not change, only how many run. */
  timeScale: number;
  /** Seconds (simulated) for foam to fade to 1/e. */
  foamLife: number;
  /** Seconds (simulated) for wet ground to dry to 1/e. */
  wetLife: number;
  /** Bed height limits for sculpting and uploads, metres. */
  bedMin: number;
  bedMax: number;
}

export const DEFAULT_SHALLOW_WATER_PARAMS: Readonly<ShallowWaterParams> = {
  gravity: 9.81,
  maxDepth: 1.5,
  friction: 0.03,
  drag: 0,
  rain: 0,
  edges: 'wall',
  cfl: 0.45,
  timeScale: 1,
  foamLife: 1.2,
  wetLife: 12,
  bedMin: -8,
  bedMax: 8,
};

/** The grid and what a shading pass needs to know about it. */
export interface ShallowWaterGrid {
  /** Cells across and down. */
  readonly width: number;
  readonly height: number;
  readonly aspect: number;
  /** Side of a cell, metres. */
  readonly cellSize: number;
  /** The domain's size in metres: [width, depth]. */
  readonly extent: readonly [number, number];
  /** One cell in uv. */
  readonly texelSize: readonly [number, number];
}

export interface ShallowWaterOptions {
  device: GPUDevice;
  /** Which implementation to run. Default `compute`. Fixed for the life of the solver. */
  kind?: ShallowWaterKind;
  /** Cells along the longer side. Default 256. */
  resolution?: number;
  /** Width / depth of the domain. Default 1.5. */
  aspect?: number;
  /** Length of the longer side in metres. Default 12. */
  size?: number;
  params?: Partial<ShallowWaterParams>;
  /** Most steps one `update` runs. Default 48. */
  maxStepsPerFrame?: number;
  profiler?: PassProfiler;
  label?: string;
}

/** A field on the grid: row-major `width * height` metres (row 0 at the top), or a function of domain uv. */
export type GridField = Float32Array | ((u: number, v: number) => number);

/** A disc of influence: water in or out, a push, a bed edit, any combination. */
export interface DiscSource {
  /** Centre, domain uv. */
  x: number;
  y: number;
  /** Footprint radius in metres (clamped to 1.5 cells at least). */
  radius: number;
  /** Cubic metres of water added per second of simulated time, spread over the disc. Negative removes. */
  volume?: number;
  /** Acceleration pushed into the water, m / s^2, +x right and +y down the texture. */
  pushX?: number;
  pushY?: number;
  /** Peak bed raise rate, m / s. Negative digs. The water above is kept (it rides up or settles). */
  bedRate?: number;
}

/** A rectangle where the water surface is pulled toward a fixed level: a sea, a tide, an inflow. */
export interface LevelZone {
  /** Corners, domain uv. */
  x0: number;
  y0: number;
  x1: number;
  y1: number;
  /** Target surface height, metres (bed height + depth). */
  level: number;
  /** Relaxation rate, 1 / s. */
  rate: number;
}

export interface ShallowWater {
  readonly kind: ShallowWaterKind;
  /** Live parameters. Mutate freely. */
  readonly params: ShallowWaterParams;
  /** The current grid. A new object after every re-creation. */
  readonly grid: ShallowWaterGrid;
  /** Increments every time the grid is re-created: rebuild bind groups that hold a texture or view. */
  readonly generation: number;
  /** (depth m, velocity x m/s, velocity y m/s, foam), filterable. Stable between updates. */
  readonly texture: GPUTexture;
  readonly view: GPUTextureView;
  /** (bed height m, surface slope x, surface slope y, wetness), filterable. */
  readonly groundTexture: GPUTexture;
  readonly groundView: GPUTextureView;
  /**
   * The exact state (depth, hu, hv, bed) as rgba32float, for tests and tools that need full
   * precision (volume, momentum). Its identity changes on every step: read it, do not hold it.
   */
  readonly stateTexture: GPUTexture;
  /** Simulated seconds per step at the current parameters. */
  readonly stepSeconds: number;
  /** Steps run by the last `update` / `advance`. */
  readonly lastSteps: number;
  /** Simulated seconds since creation or the last reset. */
  readonly simTime: number;
  /** Sources and level zones queued and not yet applied. */
  readonly pendingSources: number;

  /** Re-creates the grid with this many cells on the long side. Returns true if it changed (everything cleared). */
  setResolution(cells: number): boolean;
  /** Re-creates the grid for a new domain shape (width / depth). Returns true if it changed. */
  setAspect(aspect: number): boolean;

  /** Replaces the bed. Water and momentum stay where they are. Applied at the next update. */
  setTerrain(field: GridField): void;
  /** Replaces the water with this depth field, at rest. The bed stays. Applied at the next update. */
  setWater(field: GridField): void;

  /** Queues a disc for the next update only: call it every frame for a continuing spring or push. */
  addSource(source: DiscSource): boolean;
  /** Queues a level zone for the next update only. */
  addLevelZone(zone: LevelZone): boolean;

  /**
   * Advances the simulation by `delta` seconds of wall time (times `params.timeScale`) at the fixed
   * internal timestep and applies queued sources. Call at most once per submitted command buffer.
   * Returns the steps run.
   */
  update(encoder: GPUCommandEncoder, delta: number): number;
  /** Runs exactly `steps` steps, whatever the clock says (benchmarks, tests). */
  advance(encoder: GPUCommandEncoder, steps: number): void;
  /** Empties the grid: no water, a flat bed at 0. */
  reset(encoder: GPUCommandEncoder): void;
  destroy(): void;
}

// --- Uniforms ------------------------------------------------------------------------------------

const SimParams = defineUniformStruct('SimParams', {
  size: 'vec2u',
  dt: 'f32',
  dx: 'f32',
  g: 'f32',
  h_max: 'f32',
  lambda_cap: 'f32',
  manning: 'f32',
  drag: 'f32',
  rain: 'f32',
  bed_min: 'f32',
  bed_max: 'f32',
  resolve_dt: 'f32',
  foam_life: 'f32',
  wet_life: 'f32',
  edge_open: 'u32',
  source_count: 'u32',
  upload_mode: 'u32',
  sources: uniformArray('vec4f', MAX_SOURCES * 2),
});

// --- Pipelines, cached per device -----------------------------------------------------------------

type Pipeline = GPURenderPipeline | GPUComputePipeline;

interface Pipelines {
  upload: Pipeline;
  step: Pipeline;
  resolve: Pipeline;
}

const pipelineCache = new WeakMap<GPUDevice, Partial<Record<ShallowWaterKind, Pipelines>>>();

function createPipelines(device: GPUDevice, kind: ShallowWaterKind): Pipelines {
  const module = createShader(device, {
    label: `sim/shallow-water-${kind}.wgsl`,
    code: kind === 'render' ? renderSource : computeSource,
    chunks: { 'sw_params.wgsl': SimParams.wgsl, 'sw_core.wgsl': coreSource },
  });
  const make = (name: 'upload' | 'step' | 'resolve'): Pipeline =>
    kind === 'render'
      ? device.createRenderPipeline({
          label: `shallow water render ${name}`,
          layout: 'auto',
          vertex: { module, entryPoint: 'fullscreen_vs' },
          fragment: {
            module,
            entryPoint: `fs_${name}`,
            targets:
              name === 'resolve'
                ? [
                    { format: SHALLOW_WATER_OUTPUT_FORMAT },
                    { format: SHALLOW_WATER_OUTPUT_FORMAT },
                    { format: SHALLOW_WATER_OUTPUT_FORMAT },
                  ]
                : [{ format: SHALLOW_WATER_STATE_FORMAT }],
          },
        })
      : device.createComputePipeline({
          label: `shallow water compute ${name}`,
          layout: 'auto',
          compute: { module, entryPoint: `cs_${name}` },
        });
  return { upload: make('upload'), step: make('step'), resolve: make('resolve') };
}

function getPipelines(device: GPUDevice, kind: ShallowWaterKind): Pipelines {
  let perDevice = pipelineCache.get(device);
  if (!perDevice) pipelineCache.set(device, (perDevice = {}));
  return (perDevice[kind] ??= createPipelines(device, kind));
}

// --- Helpers -------------------------------------------------------------------------------------

const clamp = (x: number, lo: number, hi: number) => Math.min(hi, Math.max(lo, x));
const finite = (x: number, fallback: number) => (Number.isFinite(x) ? x : fallback);

/** Grid size for `resolution` cells on the long side of a domain with this aspect (width / depth). */
export function computeShallowWaterGrid(
  resolution: number,
  aspect: number,
  maxDimension = 4096,
): { width: number; height: number } {
  const long = Math.round(clamp(finite(resolution, 256), 16, maxDimension));
  const a = Number.isFinite(aspect) && aspect > 0 ? aspect : 1;
  const width = a >= 1 ? long : Math.max(16, Math.round(long * a));
  const height = a >= 1 ? Math.max(16, Math.round(long / a)) : long;
  return { width: Math.min(width, maxDimension), height: Math.min(height, maxDimension) };
}

/**
 * The timestep for these parameters on a grid with this cell size: cfl dx / (2 lambda_cap),
 * lambda_cap = 2 sqrt(g maxDepth). See STABILITY above.
 */
export function shallowWaterStepSeconds(
  params: Pick<ShallowWaterParams, 'gravity' | 'maxDepth' | 'cfl'>,
  cellSize: number,
): number {
  const g = Math.max(finite(params.gravity, 9.81), 1e-3);
  const depth = Math.max(finite(params.maxDepth, 1.5), 1e-3);
  const cfl = clamp(finite(params.cfl, 0.45), 0.1, COURANT_LIMIT);
  return (cfl * cellSize) / (4 * Math.sqrt(g * depth));
}

/** Sum of the source footprint weights over the grid cells it covers (the normaliser of a source). */
function footprintSum(
  cx: number,
  cy: number,
  radius: number,
  width: number,
  height: number,
): number {
  const x0 = Math.max(0, Math.floor(cx - radius));
  const x1 = Math.min(width - 1, Math.ceil(cx + radius));
  const y0 = Math.max(0, Math.floor(cy - radius));
  const y1 = Math.min(height - 1, Math.ceil(cy + radius));
  let sum = 0;
  for (let y = y0; y <= y1; y++) {
    for (let x = x0; x <= x1; x++) {
      const q = Math.hypot(x + 0.5 - cx, y + 0.5 - cy) / radius;
      if (q < 1) sum += (1 - q * q) ** 2;
    }
  }
  return sum;
}

interface Resources {
  grid: ShallowWaterGrid;
  /** Ping-pong state: (h, hu, hv, bed). */
  state: [GPUTexture, GPUTexture];
  stateViews: [GPUTextureView, GPUTextureView];
  /** What setTerrain / setWater write: (bed, depth). */
  upload: GPUTexture;
  uploadView: GPUTextureView;
  water: GPUTexture;
  waterView: GPUTextureView;
  ground: GPUTexture;
  groundView: GPUTextureView;
  /** Foam and wetness history, ping-pong. */
  history: [GPUTexture, GPUTexture];
  historyViews: [GPUTextureView, GPUTextureView];
  /** Bind groups indexed by the parity of the state being read. */
  uploadGroups: [GPUBindGroup, GPUBindGroup];
  stepGroups: [GPUBindGroup, GPUBindGroup];
  /** Indexed by state parity * 2 + history parity. */
  resolveGroups: [GPUBindGroup, GPUBindGroup, GPUBindGroup, GPUBindGroup];
}

export function createShallowWater(options: ShallowWaterOptions): ShallowWater {
  const {
    device,
    kind = 'compute',
    maxStepsPerFrame = 48,
    profiler = noProfiler,
    label = 'shallow water',
  } = options;
  const extentLong = options.size ?? 12;
  let resolution = options.resolution ?? 256;
  let aspect = options.aspect ?? 1.5;
  const maxDimension = Math.min(device.limits.maxTextureDimension2D, 4096);
  const params: ShallowWaterParams = { ...DEFAULT_SHALLOW_WATER_PARAMS, ...options.params };
  const pipelines = getPipelines(device, kind);
  const uniforms = createUniformBuffer(device, SimParams, `${label} params`);

  const isCompute = kind === 'compute';
  const textureUsage =
    GPUTextureUsage.TEXTURE_BINDING |
    GPUTextureUsage.RENDER_ATTACHMENT |
    GPUTextureUsage.COPY_SRC |
    GPUTextureUsage.COPY_DST |
    (isCompute ? GPUTextureUsage.STORAGE_BINDING : 0);

  let destroyed = false;
  let generation = 0;
  let parity: 0 | 1 = 0;
  let historyParity: 0 | 1 = 0;
  let lastSteps = 0;
  let simTime = 0;
  let stepper: FixedStepper | undefined;
  let stepperSeconds = 0;
  let resources: Resources;
  /** The outputs describe an older state than the state textures (or have never been written). */
  let outputStale = true;
  let resolveSeconds = 0;

  // Pending uploads: the CPU copy of (bed, depth) and which channels changed.
  let staging = new Float32Array(0);
  let pendingBed = false;
  let pendingWater = false;

  const queue = new Float32Array(MAX_SOURCES * 8);
  let queued = 0;

  function allocate(width: number, height: number): Resources {
    const extent: [number, number] =
      width >= height
        ? [extentLong, extentLong / (width / height)]
        : [extentLong * (width / height), extentLong];
    const grid: ShallowWaterGrid = Object.freeze({
      width,
      height,
      aspect: width / height,
      cellSize: extent[0] / width,
      extent: Object.freeze(extent),
      texelSize: Object.freeze([1 / width, 1 / height] as const),
    });
    const make = (name: string, format: GPUTextureFormat) =>
      device.createTexture({
        label: `${label} ${name}`,
        size: [width, height],
        format,
        usage: textureUsage,
      });
    const state: [GPUTexture, GPUTexture] = [
      make('state A', SHALLOW_WATER_STATE_FORMAT),
      make('state B', SHALLOW_WATER_STATE_FORMAT),
    ];
    const upload = make('upload', SHALLOW_WATER_STATE_FORMAT);
    const water = make('water', SHALLOW_WATER_OUTPUT_FORMAT);
    const ground = make('ground', SHALLOW_WATER_OUTPUT_FORMAT);
    const history: [GPUTexture, GPUTexture] = [
      make('history A', SHALLOW_WATER_OUTPUT_FORMAT),
      make('history B', SHALLOW_WATER_OUTPUT_FORMAT),
    ];
    const views = (t: GPUTexture) => t.createView();
    const stateViews: [GPUTextureView, GPUTextureView] = [views(state[0]), views(state[1])];
    const historyViews: [GPUTextureView, GPUTextureView] = [views(history[0]), views(history[1])];
    const uploadView = views(upload);
    const waterView = water.createView({ label: `${label} water view` });
    const groundView = ground.createView({ label: `${label} ground view` });

    const ubo: GPUBindGroupEntry = { binding: 0, resource: { buffer: uniforms.buffer } };
    const group = (pipeline: Pipeline, name: string, entries: GPUBindGroupEntry[]): GPUBindGroup =>
      device.createBindGroup({
        label: `${label} ${name}`,
        layout: pipeline.getBindGroupLayout(0),
        entries: [ubo, ...entries],
      });
    const read = (p: 0 | 1): GPUBindGroupEntry => ({ binding: 1, resource: stateViews[p] });
    const store = (p: 0 | 1): GPUBindGroupEntry[] =>
      isCompute ? [{ binding: 3, resource: stateViews[1 - p]! }] : [];

    const resolveGroup = (p: 0 | 1, hp: 0 | 1) =>
      group(pipelines.resolve, `resolve ${p}${hp}`, [
        read(p),
        { binding: 2, resource: historyViews[hp] },
        ...(isCompute
          ? [
              { binding: 4, resource: waterView },
              { binding: 5, resource: groundView },
              { binding: 6, resource: historyViews[1 - hp]! },
            ]
          : []),
      ]);

    staging = new Float32Array(width * height * 4);
    pendingBed = false;
    pendingWater = false;
    return {
      grid,
      state,
      stateViews,
      upload,
      uploadView,
      water,
      waterView,
      ground,
      groundView,
      history,
      historyViews,
      uploadGroups: [
        group(pipelines.upload, 'upload 0', [
          read(0),
          { binding: 2, resource: uploadView },
          ...store(0),
        ]),
        group(pipelines.upload, 'upload 1', [
          read(1),
          { binding: 2, resource: uploadView },
          ...store(1),
        ]),
      ],
      stepGroups: [
        group(pipelines.step, 'step 0', [read(0), ...store(0)]),
        group(pipelines.step, 'step 1', [read(1), ...store(1)]),
      ],
      resolveGroups: [
        resolveGroup(0, 0),
        resolveGroup(0, 1),
        resolveGroup(1, 0),
        resolveGroup(1, 1),
      ],
    };
  }

  function release(r: Resources) {
    for (const t of [...r.state, r.upload, r.water, r.ground, ...r.history]) t.destroy();
  }

  function rebuild(): boolean {
    const size = computeShallowWaterGrid(resolution, aspect, maxDimension);
    if (resources && size.width === resources.grid.width && size.height === resources.grid.height) {
      return false;
    }
    if (resources) release(resources);
    resources = allocate(size.width, size.height);
    parity = 0;
    historyParity = 0;
    generation++;
    stepper = undefined;
    queued = 0;
    outputStale = true;
    simTime = 0;
    return true;
  }

  rebuild();

  // --- Time and uniforms ---------------------------------------------------------------------------

  const stepSeconds = () => shallowWaterStepSeconds(params, resources.grid.cellSize);

  function writeUniforms(uploadMode: number) {
    const { grid } = resources;
    const dt = stepSeconds();
    const g = Math.max(finite(params.gravity, 9.81), 1e-3);
    const hMax = Math.max(finite(params.maxDepth, 1.5), 1e-3);
    const data = new Float32Array(MAX_SOURCES * 8);
    data.set(queue.subarray(0, queued * 8));
    uniforms.set({
      size: [grid.width, grid.height],
      dt,
      dx: grid.cellSize,
      g,
      h_max: hMax,
      lambda_cap: 2 * Math.sqrt(g * hMax),
      manning: Math.max(0, finite(params.friction, 0)),
      drag: Math.max(0, finite(params.drag, 0)),
      rain: Math.max(0, finite(params.rain, 0)),
      bed_min: finite(params.bedMin, -8),
      bed_max: finite(params.bedMax, 8),
      resolve_dt: resolveSeconds,
      foam_life: Math.max(0.01, finite(params.foamLife, 1.2)),
      wet_life: Math.max(0.01, finite(params.wetLife, 12)),
      edge_open: params.edges === 'open' ? 1 : 0,
      source_count: queued,
      upload_mode: uploadMode,
      sources: data,
    });
    uniforms.upload();
  }

  // --- Passes --------------------------------------------------------------------------------------

  const groupCounts = () =>
    [
      Math.ceil(resources.grid.width / WORKGROUP),
      Math.ceil(resources.grid.height / WORKGROUP),
    ] as const;

  function runUpload(encoder: GPUCommandEncoder) {
    const r = resources;
    device.queue.writeTexture(
      { texture: r.upload },
      staging,
      { bytesPerRow: r.grid.width * 16 },
      { width: r.grid.width, height: r.grid.height },
    );
    const timestampWrites = profiler.pass('sim upload');
    const group = r.uploadGroups[parity];
    if (isCompute) {
      const pass = encoder.beginComputePass({
        label: 'sim upload',
        timestampWrites: timestampWrites as GPUComputePassTimestampWrites | undefined,
      });
      pass.setPipeline(pipelines.upload as GPUComputePipeline);
      pass.setBindGroup(0, group);
      pass.dispatchWorkgroups(...groupCounts());
      pass.end();
    } else {
      renderFullscreen(
        encoder,
        r.stateViews[1 - parity]!,
        pipelines.upload as GPURenderPipeline,
        [group],
        { label: 'sim upload', timestampWrites },
      );
    }
    parity = (1 - parity) as 0 | 1;
    outputStale = true;
  }

  function run(encoder: GPUCommandEncoder, steps: number) {
    if (destroyed) return;
    const count = Math.max(0, Math.floor(steps));
    lastSteps = count;
    const uploading = pendingBed || pendingWater;
    if (count === 0 && !uploading && !outputStale) {
      // Nothing to step and nothing changed. Sources are immediate-mode, so they are dropped.
      queued = 0;
      return;
    }
    const r = resources;

    // One uniform write serves every pass of this run (writes land before the command buffer, so a
    // second write would replace the first for all of them): the sources apply on every step, the
    // upload mode only matters to the upload pass.
    resolveSeconds = count * stepSeconds();
    writeUniforms(pendingBed && pendingWater ? 0 : pendingBed ? 1 : 2);
    if (uploading) {
      runUpload(encoder);
      pendingBed = false;
      pendingWater = false;
    }

    if (count > 0) {
      if (isCompute) {
        // Each dispatch is its own usage scope, so one compute pass can ping-pong the whole run:
        // one pass boundary per update instead of one per step.
        const pass = encoder.beginComputePass({
          label: 'sim steps',
          timestampWrites: profiler.pass('sim steps') as GPUComputePassTimestampWrites | undefined,
        });
        pass.setPipeline(pipelines.step as GPUComputePipeline);
        for (let i = 0; i < count; i++) {
          pass.setBindGroup(0, r.stepGroups[parity]);
          pass.dispatchWorkgroups(...groupCounts());
          parity = (1 - parity) as 0 | 1;
        }
        pass.end();
      } else {
        for (let i = 0; i < count; i++) {
          // Time the first and the last step; the HUD's frame span then covers everything between.
          const timed = i === 0 || i === count - 1 ? profiler.pass('sim step') : undefined;
          renderFullscreen(
            encoder,
            r.stateViews[1 - parity]!,
            pipelines.step as GPURenderPipeline,
            [r.stepGroups[parity]],
            { label: 'sim step', timestampWrites: timed },
          );
          parity = (1 - parity) as 0 | 1;
        }
      }
      simTime += resolveSeconds;
    }
    // Sources are immediate-mode (callers queue them every frame), so none carry over.
    queued = 0;

    // Resolve the outputs and the history.
    const timestampWrites = profiler.pass('sim resolve');
    const group = r.resolveGroups[parity * 2 + historyParity]!;
    if (isCompute) {
      const pass = encoder.beginComputePass({
        label: 'sim resolve',
        timestampWrites: timestampWrites as GPUComputePassTimestampWrites | undefined,
      });
      pass.setPipeline(pipelines.resolve as GPUComputePipeline);
      pass.setBindGroup(0, group);
      pass.dispatchWorkgroups(...groupCounts());
      pass.end();
    } else {
      renderFullscreen(
        encoder,
        [r.waterView, r.groundView, r.historyViews[1 - historyParity]!],
        pipelines.resolve as GPURenderPipeline,
        [group],
        { label: 'sim resolve', timestampWrites },
      );
    }
    historyParity = (1 - historyParity) as 0 | 1;
    outputStale = false;
  }

  function fillStaging(channel: 0 | 1, field: GridField) {
    const { width, height } = resources.grid;
    if (typeof field === 'function') {
      for (let y = 0; y < height; y++) {
        for (let x = 0; x < width; x++) {
          const value = field((x + 0.5) / width, (y + 0.5) / height);
          staging[(y * width + x) * 4 + channel] = Number.isFinite(value) ? value : 0;
        }
      }
    } else {
      if (field.length !== width * height) {
        throw new RangeError(
          `${label}: a grid field needs ${width * height} values (${width} x ${height}), got ${field.length}`,
        );
      }
      for (let i = 0; i < width * height; i++) {
        const value = field[i]!;
        staging[i * 4 + channel] = Number.isFinite(value) ? value : 0;
      }
    }
  }

  function enqueue(a: readonly number[], e: readonly number[]): boolean {
    if (queued >= MAX_SOURCES) return false;
    if (![...a, ...e].every(Number.isFinite)) return false;
    queue.set(a, queued * 8);
    queue.set(e, queued * 8 + 4);
    queued++;
    return true;
  }

  return {
    kind,
    params,
    get grid() {
      return resources.grid;
    },
    get generation() {
      return generation;
    },
    get texture() {
      return resources.water;
    },
    get view() {
      return resources.waterView;
    },
    get groundTexture() {
      return resources.ground;
    },
    get groundView() {
      return resources.groundView;
    },
    get stateTexture() {
      return resources.state[parity];
    },
    get stepSeconds() {
      return stepSeconds();
    },
    get lastSteps() {
      return lastSteps;
    },
    get simTime() {
      return simTime;
    },
    get pendingSources() {
      return queued;
    },

    setResolution(cells) {
      if (!Number.isFinite(cells)) return false;
      resolution = cells;
      return rebuild();
    },
    setAspect(next) {
      if (!(next > 0) || !Number.isFinite(next)) return false;
      aspect = next;
      return rebuild();
    },

    setTerrain(field) {
      fillStaging(0, field);
      pendingBed = true;
    },
    setWater(field) {
      fillStaging(1, field);
      pendingWater = true;
    },

    addSource(source) {
      const { grid } = resources;
      const radius = clamp(
        finite(source.radius, 0) / grid.cellSize,
        MIN_RADIUS_CELLS,
        MAX_RADIUS_CELLS,
      );
      const cx = source.x * grid.width;
      const cy = source.y * grid.height;
      const sum = footprintSum(cx, cy, radius, grid.width, grid.height);
      if (!(sum > 0)) return false;
      // Peak depth rate so that the footprint adds `volume` cubic metres per second in total.
      const depthRate = (source.volume ?? 0) / (sum * grid.cellSize * grid.cellSize);
      return enqueue(
        [cx, cy, radius, depthRate],
        [source.pushX ?? 0, source.pushY ?? 0, source.bedRate ?? 0, 0],
      );
    },
    addLevelZone(zone) {
      const { grid } = resources;
      const x0 = Math.min(zone.x0, zone.x1) * grid.width;
      const x1 = Math.max(zone.x0, zone.x1) * grid.width;
      const y0 = Math.min(zone.y0, zone.y1) * grid.height;
      const y1 = Math.max(zone.y0, zone.y1) * grid.height;
      return enqueue([x0, y0, x1, y1], [zone.level, Math.max(0, zone.rate), 0, 1]);
    },

    update(encoder, delta) {
      if (destroyed) return 0;
      const dt = stepSeconds();
      if (!stepper || Math.abs(dt - stepperSeconds) > dt * 1e-3) {
        stepper = createFixedStepper(dt, { maxSteps: maxStepsPerFrame });
        stepperSeconds = dt;
      }
      const scale = Math.max(0, finite(params.timeScale, 1));
      const { steps } = stepper.advance(finite(delta, 0) * scale);
      run(encoder, steps);
      return steps;
    },
    advance(encoder, steps) {
      run(encoder, steps);
    },
    reset(encoder) {
      queued = 0;
      pendingBed = false;
      pendingWater = false;
      staging.fill(0);
      const r = resources;
      for (const texture of [...r.state, r.water, r.ground, ...r.history]) {
        encoder
          .beginRenderPass({
            label: `${label} clear`,
            colorAttachments: [{ view: texture.createView(), loadOp: 'clear', storeOp: 'store' }],
          })
          .end();
      }
      parity = 0;
      historyParity = 0;
      simTime = 0;
      outputStale = false;
    },
    destroy() {
      if (destroyed) return;
      destroyed = true;
      release(resources);
      uniforms.destroy();
    },
  };
}
