import { createFixedStepper, type FixedStepper } from '../loop.ts';
import { noProfiler, type PassProfiler } from '../profiler.ts';
import { createShader } from '../shader.ts';
import { createUniformBuffer, defineUniformStruct, type UniformBuffer } from '../uniforms.ts';
import advectSource from './fluid-advect.wgsl?raw';
import coreSource from './fluid-core.wgsl?raw';
import forcesSource from './fluid-forces.wgsl?raw';
import multigridSource from './fluid-multigrid.wgsl?raw';
import projectSource from './fluid-project.wgsl?raw';
import { createObstacleMask, type ObstacleMask } from './obstacles.ts';

// The stable-fluids solver (story 6.1): incompressible 2D flow on a grid, in compute shaders, after Jos
// Stam's "Stable Fluids" (1999) and Mark Harris's GPU Gems chapter 38. One step is
//
//     advect velocity -> add forces (vorticity confinement, buoyancy, body force) -> [viscosity]
//       -> divergence -> pressure solve (Jacobi) -> subtract the pressure gradient -> advect dye
//
// plus splats of momentum and dye injected before the steps, and a resolve pass that writes the
// texture consumers read. See fluid-core.wgsl for the grid, fluid-advect.wgsl for the three
// advection schemes, fluid-project.wgsl for the projection, src/wgsl/fluid.wgsl for decoding.
//
// GRID. Staggered (MAC): velocity components live on cell faces, pressure at cell centres. One
// rgba16float texture of (width + 1) x (height + 1) texels holds u in .x and v in .y; see
// fluid-core.wgsl for the exact placement and why (compact differences: an exact projection and
// leak-free solids). Cells are square; the grid has the canvas's aspect ratio. The DYE lives on a
// separate, finer rgba16float grid and is carried by the coarse velocity: the sim runs at 128 to 256
// cells on the long side, the dye at 512 to 1024, which is where the picture gets its detail.
//
// UNITS. Positions are canvas uv (0..1, top-left origin). Lengths (radius, cell size) are IMAGE
// HEIGHTS (canvas height = 1), the unit the heightfield solver and refraction.wgsl use. Velocity is
// image heights per second, vorticity 1 / s, time seconds. A velocity of 1 crosses the canvas
// height in a second.
//
// STORAGE. All velocity and dye state is rgba16float: renderable, filterable and write-only-storage
// in core WebGPU everywhere. Pressure, divergence and vorticity are r32float (storage in core; read
// with textureLoad because it is not filterable without a feature). Nothing relies on read-write
// storage textures: every pass reads a sampled texture and writes a different storage texture, and
// the solver rotates through a small pool of identical textures.
//
// STABILITY. Advection is semi-Lagrangian (unconditionally stable) or a limited correction of it
// (MacCormack, BFECC: the limiter keeps them inside the range of the data they interpolate, which
// is what makes them stable too). Time still advances in fixed steps (`createFixedStepper`) with a
// cap, the trace distance is capped at `maxCfl` cells, velocities and forces are clamped to that
// speed and non-finite values are replaced by zero, so no input blows the field up.
//
// PRESSURE SOLVERS (story 6.8, docs/fluid-cost-study.md). `jacobi` (default, damped) and red-black `sor`
// are relaxations: the error of the longest wavelengths shrinks by a factor of about 1 - c / N^2 (Jacobi) or
// 1 - c / N (SOR) per iteration on an N cell grid, so a larger grid needs ever more of them, and the
// divergence a step leaves is carried over by the warm start. `multigrid` is a V-cycle over a pyramid of
// grids half as wide (fluid-multigrid.wgsl): it removes about nine tenths of the divergence per cycle
// whatever N is, for the price of a handful of relaxation sweeps' worth of dispatches. It is opt-in
// (`pressureSolver: 'multigrid'`, `multigridCycles` per step); the defaults are unchanged.
//
// DYE VELOCITY. The dye grid is finer than the velocity grid, and the dye reads the velocity bilinearly,
// so the field it sees has a kink in its slope at every velocity cell border. By default that read goes
// through the hardware texture filter, which the cost study measured at a quarter to a fifth of the cost
// of the equivalent loads; `dyeVelocityFilter` can ask for a cubic reconstruction ('bspline' or
// 'catmull-rom') instead, or for the loads ('bilinear').

export type FluidBoundary = 'wall' | 'open' | 'wrap';
export type FluidPressureSolver = 'sor' | 'jacobi' | 'multigrid';
/** How the velocity is reconstructed between its faces when it carries the dye; see `FluidParams.dyeVelocityFilter`. */
export type FluidVelocityFilter = 'bilinear' | 'hardware-bilinear' | 'bspline' | 'catmull-rom';
export type FluidScheme = 'semi-lagrangian' | 'maccormack' | 'bfecc';

/** Format of velocity, dye and the resolved fields. */
export const FLUID_FORMAT: GPUTextureFormat = 'rgba16float';
/** Format of pressure, divergence and vorticity. */
export const FLUID_SCALAR_FORMAT: GPUTextureFormat = 'r32float';
/** Splats of each kind (momentum, dye) accepted between two updates; more are dropped. */
export const MAX_SPLATS = 256;
/** Floats per splat in the storage buffer: three vec4f. */
const SPLAT_FLOATS = 12;
const WORKGROUP = 16;
const DYE_LIMIT = 256;

/** Live, mutable parameters: change them any time, they are read on every update. */
export interface FluidParams {
  /** Fixed simulation steps per second. The flow looks the same at any display rate. Default 60. */
  stepRate: number;
  /**
   * Pressure solver: plain `jacobi` (default), red-black `sor`, which needs iterations of the order of
   * the grid size to beat it (see fluid-project.wgsl), or `multigrid`, V-cycles over a pyramid of
   * coarser grids (see fluid-multigrid.wgsl and docs/fluid-cost-study.md): `multigridCycles` per step
   * instead of `pressureIterations`, and a 128 cell grid converges to a few percent in one.
   */
  pressureSolver: FluidPressureSolver;
  /**
   * Damping of the Jacobi iteration, 0.3 to 1. 1 is the textbook iteration, which cannot touch the
   * chequerboard error mode and over-corrects it on every odd iteration; the default 0.8 does neither.
   */
  jacobiWeight: number;
  /** Over-relaxation factor of the `sor` solver, 1 (Gauss-Seidel) to 1.95; 0 picks the optimum for the grid. */
  sorOmega: number;
  /** Iterations of the pressure solve per step; each is one dispatch over the grid (a SOR sweep is two). Default 30. */
  pressureIterations: number;
  /** Start the pressure solve from last step's pressure (a much better guess) instead of zero. Default true. */
  warmStart: boolean;
  /** `multigrid`: V-cycles per step (the unit that `pressureIterations` is for the other solvers), 1 to 8. Default 1. */
  multigridCycles: number;
  /** `multigrid`: damped-Jacobi sweeps before and after the coarse correction on each level, 1 to 8. Default 1. */
  multigridSmooth: number;
  /**
   * `multigrid`: most levels in the pyramid (the finest counts), 2 or more; 0 coarsens until the grid fits one
   * 32 x 32 workgroup tile. Fewer levels leave a coarsest grid that is bigger and solved by relaxation.
   */
  multigridLevels: number;
  /**
   * `multigrid`: sweeps on the coarsest level. When it fits 32 x 32 cells these are red-black SOR sweeps
   * inside a single dispatch (so they are almost free); otherwise damped Jacobi dispatches. Default 24.
   */
  multigridCoarse: number;
  /** Advection of the velocity field. Default `maccormack`. */
  velocityScheme: FluidScheme;
  /** Advection of the dye. Default `maccormack`. */
  dyeScheme: FluidScheme;
  /** Midpoint (second-order) back-tracing instead of Euler. Default true. */
  rk2: boolean;
  /**
   * How the velocity is interpolated when it carries the DYE (velocity advects itself bilinearly
   * whatever this says). `bilinear` (default) is continuous but has a kink in its slope at every
   * cell border, which a dye grid finer than the velocity shows as corners in streaklines, and is
   * computed from eight loads with the obstacle and boundary bookkeeping; `hardware-bilinear` is the
   * same function through the texture filter (two taps); `bspline` (C2, smooths a little) and
   * `catmull-rom` (C1, interpolates) are cubic reconstructions through the filter (four and nine taps
   * per component). The filtered ones are several times cheaper than the loads (docs/fluid-cost-study.md).
   * They are only used without obstacles and without the `wrap` boundary; otherwise the dye is carried
   * by the `bilinear` path.
   */
  dyeVelocityFilter: FluidVelocityFilter;
  /**
   * Clamp the MacCormack and BFECC corrections to the range of the samples they interpolate. Default
   * true and meant to stay so: without it both schemes overshoot, ring and can diverge. It can be
   * switched off to see that.
   */
  limiter: boolean;
  /** Vorticity confinement strength epsilon, 1 / s. 0 is off; 5 to 30 keeps small vortices alive. */
  vorticity: number;
  /** Kinematic viscosity, image heights squared per second. 0 is off (and costs nothing). */
  viscosity: number;
  /** Jacobi iterations of the viscous diffusion per step. Default 12. */
  viscosityIterations: number;
  /** Velocity decay rate, 1 / s: v *= exp(-rate * dt). 0 never loses speed. */
  velocityDissipation: number;
  /** Dye decay rate, 1 / s: dye *= exp(-rate * dt). 0 never fades. */
  dyeDissipation: number;
  /** Uniform acceleration (wind, gravity), image heights per second squared: [x, y], y DOWN. */
  gravity: readonly [number, number];
  /** Acceleration per unit of dye density (the dye's alpha channel) along `buoyancyDirection`. 0 is off. */
  buoyancy: number;
  /** Direction of the buoyancy acceleration, [x, y] with y DOWN (so smoke rises along [0, -1]). */
  buoyancyDirection: readonly [number, number];
  /** What the grid edge does: `wall` closed, free-slip; `open` flow leaves (and clear water enters); `wrap` periodic. */
  boundary: FluidBoundary;
  /** Largest distance a parcel travels in one step, in cells. Also caps velocity at maxCfl / (dt * cells per height). Default 6. */
  maxCfl: number;
  /** Largest dye value (any channel). Default 64. */
  dyeLimit: number;
  /**
   * Only advect the dye (and apply its dissipation) by the velocity field as it is: no velocity
   * advection, forces, pressure or viscosity. For prescribed flows and for the scheme comparison.
   */
  dyeOnly: boolean;
}

/** The `kind` of fluid-core.wgsl's `vel_at_cubic` for each filter other than `bilinear`. */
const DYE_FILTER_KIND: Readonly<Record<FluidVelocityFilter, number>> = {
  bilinear: 0,
  bspline: 1,
  'catmull-rom': 2,
  'hardware-bilinear': 3,
};

export const DEFAULT_FLUID_PARAMS: Readonly<FluidParams> = {
  stepRate: 60,
  pressureSolver: 'jacobi',
  jacobiWeight: 0.8,
  sorOmega: 0,
  pressureIterations: 30,
  warmStart: true,
  multigridCycles: 1,
  multigridSmooth: 1,
  multigridLevels: 0,
  multigridCoarse: 24,
  velocityScheme: 'maccormack',
  dyeScheme: 'maccormack',
  rk2: true,
  dyeVelocityFilter: 'hardware-bilinear',
  limiter: true,
  vorticity: 0,
  viscosity: 0,
  viscosityIterations: 12,
  velocityDissipation: 0.2,
  dyeDissipation: 0.5,
  gravity: [0, 0],
  buoyancy: 0,
  buoyancyDirection: [0, -1],
  boundary: 'wall',
  maxCfl: 6,
  dyeLimit: 64,
  dyeOnly: false,
};

/** The grid and what a shading pass needs to know about it. */
export interface FluidGrid {
  /** Cells across and down. */
  readonly width: number;
  readonly height: number;
  /** width / height of the grid (the canvas aspect, to within one cell). */
  readonly aspect: number;
  /** Side of a cell in image heights: 1 / height. */
  readonly cellSize: number;
  /** One cell in uv: [1 / width, 1 / height]. */
  readonly texelSize: readonly [number, number];
  /** Size of the velocity texture: (width + 1) x (height + 1). */
  readonly velocityWidth: number;
  readonly velocityHeight: number;
  /** Size of the dye texture. */
  readonly dyeWidth: number;
  readonly dyeHeight: number;
}

/** A texture the solver owns, with its default view. */
export interface FluidTexture {
  readonly texture: GPUTexture;
  readonly view: GPUTextureView;
  /**
   * Which of the solver's interchangeable textures this is (0, 1 or 2). Fields that are ping-ponged
   * (velocity, dye) change texture after every step: cache bind groups by `slot`, and drop the
   * cache when `generation` changes.
   */
  readonly slot: number;
}

/** What a custom velocity pass sees; see `FluidHooks.forces`. */
export interface FluidForceContext {
  readonly encoder: GPUCommandEncoder;
  /** Seconds this step advances. */
  readonly dt: number;
  readonly grid: FluidGrid;
  /** The velocity as it is now (sampled; layout in fluid-core.wgsl, decode with fluid.wgsl). */
  readonly read: GPUTextureView;
  /** Write-only storage, rgba16float, same size: write EVERY texel (copy `read` where you add nothing). */
  readonly write: GPUTextureView;
  /** The current dye, for forces that depend on it. */
  readonly dye: GPUTextureView;
}

/** What a custom dye pass sees; see `FluidHooks.dye`. */
export interface FluidDyeContext {
  readonly encoder: GPUCommandEncoder;
  readonly dt: number;
  readonly grid: FluidGrid;
  /** The dye as it is now. */
  readonly read: GPUTextureView;
  /** Write-only storage, rgba16float, dye sized: write every texel. */
  readonly write: GPUTextureView;
  /** The velocity this step ended with. */
  readonly velocity: GPUTextureView;
}

/**
 * Places where a consumer adds its own passes. A hook encodes ONE pass that reads `read` and writes
 * `write`, and returns true if it wrote (the solver then continues from `write`) or false to leave
 * the field untouched. Hooks run once per simulation step.
 */
export interface FluidHooks {
  /** After the built-in forces, before the pressure solve: add a force field, an emitter's momentum, a source/sink. The projection that follows cleans up whatever divergence it makes. */
  forces?: (context: FluidForceContext) => boolean;
  /** After the dye has been advected: add a dye source, a reaction, a colour shift. */
  dye?: (context: FluidDyeContext) => boolean;
}

export interface FluidOptions {
  device: GPUDevice;
  /** Cells along the longer side of the velocity grid. Default 128. */
  resolution?: number;
  /** Texels along the longer side of the dye grid. Default 4 x resolution, capped at 1024. */
  dyeResolution?: number;
  /** Canvas size (any units) that gives the grids their aspect ratio. Default square. */
  width?: number;
  height?: number;
  /** Initial parameters, merged over DEFAULT_FLUID_PARAMS. */
  params?: Partial<FluidParams>;
  /** Most steps one `update` runs. Default 4. */
  maxStepsPerFrame?: number;
  /** Receives timestamp slots for the first step's phases, the last step's dye pass and the resolve pass. */
  profiler?: PassProfiler;
  label?: string;
}

export interface Fluid {
  /** Live parameters. Mutate freely. */
  readonly params: FluidParams;
  /** The current grids. A new object after every re-creation. */
  readonly grid: FluidGrid;
  /** Increments every time the textures are re-created: rebuild bind groups that hold any of them. */
  readonly generation: number;
  /**
   * The resolved consumer texture (width x height cells, rgba16float, filterable, STABLE between
   * updates): x, y the velocity at the cell centre (image heights per second), z the vorticity (1 / s),
   * w the pressure potential. Solid cells are zero. Decode with src/wgsl/fluid.wgsl.
   */
  readonly fields: GPUTexture;
  readonly fieldsView: GPUTextureView;
  /** The dye (dyeWidth x dyeHeight, rgba16float, filterable). Changes texture after every step: cache by `slot`. */
  readonly dye: FluidTexture;
  /** The staggered velocity texture the solver works on ((width + 1) x (height + 1)). Changes after every step. */
  readonly velocity: FluidTexture;
  /** Pressure potential (r32float, textureLoad only), latest solve. Changes with the iteration count. */
  readonly pressure: FluidTexture;
  /** The divergence the last projection removed (r32float, textureLoad only). */
  readonly divergence: GPUTexture;
  /** Solid-versus-fluid mask, grid sized (shared type with the heightfield solver). Flow goes round solids. */
  readonly obstacles: ObstacleMask;
  /** Custom passes. Assign `hooks.forces` / `hooks.dye`. */
  readonly hooks: FluidHooks;
  /** Seconds per simulation step at the current `stepRate`. */
  readonly stepSeconds: number;
  /** Steps run by the last `update` / `advance`. */
  readonly lastSteps: number;
  /** Splats queued and not yet applied: [momentum, dye]. */
  readonly pendingSplats: readonly [number, number];

  /** Re-fits the grids to a canvas of this size (only the aspect matters). True if they were re-created (all state cleared, `generation` bumped). */
  resize(width: number, height: number): boolean;
  /** Changes the cells along the long side of the velocity grid and the dye grid; re-creates if that changes anything. */
  setResolution(cells: number, dyeCells?: number): boolean;

  /**
   * Queues a push: adds `velocity` (image heights per second) to the fluid along the segment from uv
   * `from` to uv `to`, with a Gaussian footprint of `radius` image heights (the distance where it
   * falls to 1/e). For a point, pass the same position twice. `ramp` is the half-width, in image
   * heights, of a smooth start and end along the segment (default 0, a flat capsule): segments laid
   * end to end with the same ramp add up to a uniform stroke with no beads at the joints (see
   * fluid-stir.ts). Returns false (and drops it) if the queue is full or an argument is not finite.
   */
  addForce(
    from: readonly [number, number],
    to: readonly [number, number],
    radius: number,
    velocity: readonly [number, number],
    ramp?: number,
  ): boolean;
  /**
   * Queues dye: adds `value` (rgba; usually colour times amount in rgb and a density in a) along the
   * segment `from` to `to` with the same footprint. Negative values remove dye.
   */
  addDye(
    from: readonly [number, number],
    to: readonly [number, number],
    radius: number,
    value: readonly [number, number, number, number],
    ramp?: number,
  ): boolean;

  /**
   * Advances by `delta` seconds of wall time at the fixed step rate: applies queued splats, runs the
   * whole-number of steps that fit (at most `maxStepsPerFrame`), and resolves `fields`. Call at most
   * once per submitted command buffer. Returns the steps run.
   */
  update(encoder: GPUCommandEncoder, delta: number): number;
  /** Runs exactly `steps` steps whatever the clock says (benchmarks, tests), applying queued splats first. */
  advance(encoder: GPUCommandEncoder, steps: number): void;
  /**
   * Runs the pressure projection on the current velocity now (divergence, `iterations` Jacobi
   * iterations, gradient subtraction), without advecting anything. Defaults to `params.pressureIterations`.
   */
  project(encoder: GPUCommandEncoder, iterations?: number): void;
  /** Writes `fields` from the current velocity and pressure. `update` and `advance` already do. */
  resolve(encoder: GPUCommandEncoder): void;

  /**
   * Replaces the velocity from a function of uv returning [vx, vy] in image heights per second,
   * evaluated at each face's own position. Uploaded immediately (queue order); it is not projected.
   */
  setVelocity(field: (x: number, y: number) => readonly [number, number]): void;
  /** Replaces the dye from a function of uv returning rgba, evaluated at texel centres. */
  setDye(field: (x: number, y: number) => readonly [number, number, number, number]): void;
  /** Clears velocity, pressure, dye and `fields`, and drops queued splats. */
  reset(encoder: GPUCommandEncoder): void;
  destroy(): void;
}

// --- Uniforms --------------------------------------------------------------------------------------

/** The uniform block every fluid kernel reads as `P` (for tests and consumers that write their own kernels against `fluidShaderChunks`). */
export const FluidUniforms = defineUniformStruct('FluidUniforms', {
  grid: 'vec2u',
  dye_size: 'vec2u',
  dt: 'f32',
  cells_per_h: 'f32',
  aspect: 'f32',
  max_disp: 'f32',
  boundary: 'u32',
  solid: 'u32',
  rk2: 'u32',
  limit: 'u32',
  force_count: 'u32',
  dye_count: 'u32',
  vorticity: 'f32',
  buoyancy: 'f32',
  vel_decay: 'f32',
  dye_decay: 'f32',
  diffuse_a: 'f32',
  sor_omega: 'f32',
  jacobi_w: 'f32',
  vmax: 'f32',
  dye_max: 'f32',
  accel: 'vec2f',
  buoy_dir: 'vec2f',
  dye_filter: 'u32',
  mg_sweeps: 'u32',
});

// --- Pipelines, cached per device -------------------------------------------------------------------

/** What a binding is, for building bind group layouts without repeating the WGSL. */
type BindKind =
  | 'uniform'
  | 'mask'
  | 'tex'
  | 'texu'
  | 'storage-rgba'
  | 'storage-rgba8'
  | 'storage-r32'
  | 'buffer'
  | 'sampler';

interface PassSpec {
  source: 'advect' | 'forces' | 'project' | 'multigrid';
  entry: string;
  bindings: Readonly<Record<number, BindKind>>;
}

const BASE = { 0: 'uniform', 1: 'mask' } as const;

function spec(
  source: PassSpec['source'],
  entry: string,
  bindings: Record<number, BindKind>,
): PassSpec {
  return { source, entry, bindings: { ...BASE, ...bindings } };
}

const PASSES = {
  velSl: spec('advect', 'cs_vel_sl', { 2: 'tex', 4: 'storage-rgba' }),
  velPred: spec('advect', 'cs_vel_pred', { 2: 'tex', 4: 'storage-rgba' }),
  velMc: spec('advect', 'cs_vel_mc', { 2: 'tex', 3: 'tex', 4: 'storage-rgba' }),
  velBfeccError: spec('advect', 'cs_vel_bfecc_error', { 2: 'tex', 3: 'tex', 4: 'storage-rgba' }),
  velBfeccFinal: spec('advect', 'cs_vel_bfecc_final', { 2: 'tex', 3: 'tex', 4: 'storage-rgba' }),
  dyeSl: spec('advect', 'cs_dye_sl', { 2: 'tex', 4: 'storage-rgba', 5: 'tex', 6: 'sampler' }),
  dyePred: spec('advect', 'cs_dye_pred', { 2: 'tex', 4: 'storage-rgba', 5: 'tex', 6: 'sampler' }),
  dyeMc: spec('advect', 'cs_dye_mc', {
    2: 'tex',
    3: 'tex',
    4: 'storage-rgba',
    5: 'tex',
    6: 'sampler',
  }),
  dyeBfeccError: spec('advect', 'cs_dye_bfecc_error', {
    2: 'tex',
    3: 'tex',
    4: 'storage-rgba',
    5: 'tex',
    6: 'sampler',
  }),
  dyeBfeccFinal: spec('advect', 'cs_dye_bfecc_final', {
    2: 'tex',
    3: 'tex',
    4: 'storage-rgba',
    5: 'tex',
    6: 'sampler',
  }),
  splatVelocity: spec('forces', 'cs_splat_velocity', { 2: 'tex', 3: 'storage-rgba', 4: 'buffer' }),
  splatDye: spec('forces', 'cs_splat_dye', { 2: 'tex', 3: 'storage-rgba', 4: 'buffer' }),
  curl: spec('forces', 'cs_curl', { 2: 'tex', 8: 'storage-r32' }),
  curlSmooth: spec('forces', 'cs_curl_smooth', { 5: 'texu', 10: 'storage-r32' }),
  forces: spec('forces', 'cs_forces', {
    2: 'tex',
    3: 'storage-rgba',
    5: 'texu',
    6: 'tex',
    7: 'sampler',
    11: 'texu',
  }),
  diffuse: spec('forces', 'cs_diffuse', { 2: 'tex', 3: 'storage-rgba', 9: 'tex' }),
  divergence: spec('project', 'cs_divergence', { 2: 'tex', 6: 'storage-r32' }),
  jacobi: spec('project', 'cs_jacobi', { 3: 'texu', 4: 'texu', 5: 'storage-r32' }),
  jacobiCold: spec('project', 'cs_jacobi_cold', { 3: 'texu', 4: 'texu', 5: 'storage-r32' }),
  sorRed: spec('project', 'cs_sor_red', { 3: 'texu', 4: 'texu', 5: 'storage-r32' }),
  sorRedCold: spec('project', 'cs_sor_red_cold', { 3: 'texu', 4: 'texu', 5: 'storage-r32' }),
  sorBlack: spec('project', 'cs_sor_black', { 3: 'texu', 4: 'texu', 5: 'storage-r32' }),
  project: spec('project', 'cs_project', { 2: 'tex', 4: 'texu', 7: 'storage-rgba' }),
  resolve: spec('project', 'cs_resolve', { 2: 'tex', 4: 'texu', 8: 'storage-rgba' }),
  mgRestrict: spec('multigrid', 'cs_mg_restrict', { 2: 'texu', 3: 'texu', 5: 'storage-r32' }),
  mgMask: spec('multigrid', 'cs_mg_mask', { 7: 'storage-rgba8' }),
  mgProlong: spec('multigrid', 'cs_mg_prolong', {
    3: 'texu',
    4: 'texu',
    5: 'storage-r32',
    6: 'mask',
  }),
  mgCoarse: spec('multigrid', 'cs_mg_coarse', { 2: 'texu', 5: 'storage-r32' }),
} as const satisfies Record<string, PassSpec>;

type PassName = keyof typeof PASSES;

interface Pipeline {
  pipeline: GPUComputePipeline;
  layout: GPUBindGroupLayout;
  spec: PassSpec;
}

const SOURCES = {
  advect: advectSource,
  forces: forcesSource,
  project: projectSource,
  multigrid: multigridSource,
} as const;

function layoutEntry(binding: number, kind: BindKind): GPUBindGroupLayoutEntry {
  const visibility = GPUShaderStage.COMPUTE;
  switch (kind) {
    case 'uniform':
      return { binding, visibility, buffer: { type: 'uniform' } };
    case 'mask':
    case 'tex':
      return { binding, visibility, texture: { sampleType: 'float' } };
    case 'texu':
      return { binding, visibility, texture: { sampleType: 'unfilterable-float' } };
    case 'storage-rgba':
      return {
        binding,
        visibility,
        storageTexture: { access: 'write-only', format: FLUID_FORMAT },
      };
    case 'storage-rgba8':
      return {
        binding,
        visibility,
        storageTexture: { access: 'write-only', format: 'rgba8unorm' },
      };
    case 'storage-r32':
      return {
        binding,
        visibility,
        storageTexture: { access: 'write-only', format: FLUID_SCALAR_FORMAT },
      };
    case 'buffer':
      return { binding, visibility, buffer: { type: 'read-only-storage' } };
    case 'sampler':
      return { binding, visibility, sampler: { type: 'filtering' } };
  }
}

const pipelineCache = new WeakMap<GPUDevice, Map<PassName, Pipeline>>();
const moduleCache = new WeakMap<GPUDevice, Map<string, GPUShaderModule>>();

function getModule(device: GPUDevice, source: PassSpec['source']): GPUShaderModule {
  let modules = moduleCache.get(device);
  if (!modules) moduleCache.set(device, (modules = new Map()));
  let module = modules.get(source);
  if (!module) {
    module = createShader(device, {
      label: `sim/fluid-${source}.wgsl`,
      code: SOURCES[source],
      chunks: { 'fluid_uniforms.wgsl': FluidUniforms.wgsl, 'fluid-core.wgsl': coreSource },
    });
    modules.set(source, module);
  }
  return module;
}

function getPipeline(device: GPUDevice, name: PassName): Pipeline {
  let pipelines = pipelineCache.get(device);
  if (!pipelines) pipelineCache.set(device, (pipelines = new Map()));
  let found = pipelines.get(name);
  if (!found) {
    const passSpec: PassSpec = PASSES[name];
    const layout = device.createBindGroupLayout({
      label: `fluid ${passSpec.entry} layout`,
      entries: Object.entries(passSpec.bindings).map(([binding, kind]) =>
        layoutEntry(Number(binding), kind),
      ),
    });
    const pipeline = device.createComputePipeline({
      label: `fluid ${passSpec.entry}`,
      layout: device.createPipelineLayout({
        label: `fluid ${passSpec.entry} pipeline layout`,
        bindGroupLayouts: [layout],
      }),
      compute: { module: getModule(device, passSpec.source), entryPoint: passSpec.entry },
    });
    found = { pipeline, layout, spec: passSpec };
    pipelines.set(name, found);
  }
  return found;
}

// --- Half floats (for uploads and readback) ------------------------------------------------------------

const scratchF32 = new Float32Array(1);
const scratchU32 = new Uint32Array(scratchF32.buffer);

/** Converts a number to the bits of an IEEE half float (round to nearest, ties away from zero). */
export function floatToHalf(value: number): number {
  scratchF32[0] = value;
  const x = scratchU32[0]!;
  const sign = (x >>> 16) & 0x8000;
  if ((x & 0x7fffffff) > 0x7f800000) return sign | 0x7e00;
  const exponent = ((x >>> 23) & 0xff) - 127 + 15;
  const mantissa = x & 0x7fffff;
  if (exponent >= 31) return sign | 0x7c00;
  if (exponent <= 0) {
    if (exponent < -10) return sign;
    const full = mantissa | 0x800000;
    const shift = 14 - exponent;
    let half = full >> shift;
    if ((full >> (shift - 1)) & 1) half++;
    return sign | half;
  }
  let half = sign | (exponent << 10) | (mantissa >> 13);
  if (mantissa & 0x1000) half++;
  return half;
}

/** The value of a half float given its bits. */
export function halfToFloat(bits: number): number {
  const sign = bits & 0x8000 ? -1 : 1;
  const exponent = (bits >> 10) & 0x1f;
  const fraction = bits & 0x3ff;
  if (exponent === 0) return sign * 2 ** -14 * (fraction / 1024);
  if (exponent === 31) return fraction ? Number.NaN : sign * Infinity;
  return sign * 2 ** (exponent - 15) * (1 + fraction / 1024);
}

// --- Helpers ---------------------------------------------------------------------------------------------

const clamp = (x: number, lo: number, hi: number) => Math.min(hi, Math.max(lo, x));
const finiteOr = (x: number, fallback: number) => (Number.isFinite(x) ? x : fallback);

/** Grid size for `resolution` cells on the long side of a canvas with this aspect ratio. */
export function computeFluidGrid(
  resolution: number,
  aspect: number,
  maxDimension = 4096,
): { width: number; height: number } {
  const long = Math.round(clamp(finiteOr(resolution, 128), 16, maxDimension));
  const a = Number.isFinite(aspect) && aspect > 0 ? aspect : 1;
  const width = a >= 1 ? long : Math.max(16, Math.round(long * a));
  const height = a >= 1 ? Math.max(16, Math.round(long / a)) : long;
  return { width: Math.min(width, maxDimension), height: Math.min(height, maxDimension) };
}

/**
 * Optimal over-relaxation factor of red-black SOR for the 5-point Poisson problem on a w x h grid
 * (Young): 2 / (1 + sqrt(1 - rho^2)) with rho the Jacobi spectral radius. Obstacles and open edges
 * move the optimum a little; the cap keeps those cases comfortably convergent.
 */
export function sorOmega(width: number, height: number): number {
  const rho = (Math.cos(Math.PI / width) + Math.cos(Math.PI / height)) / 2;
  return Math.min(1.9, 2 / (1 + Math.sqrt(1 - rho * rho)));
}

/** Largest side of the coarsest multigrid level that the one-workgroup solve handles. */
export const MULTIGRID_COARSEST = 32;

/**
 * The grids of a multigrid pyramid for a `width` x `height` fine grid: each level halves the one
 * above (rounding up) until the coarsest fits `MULTIGRID_COARSEST` cells on its long side, or until
 * `maxLevels` (2 or more; 0 = no cap) is reached.
 */
export function multigridLevelSizes(
  width: number,
  height: number,
  maxLevels = 0,
): { width: number; height: number }[] {
  const sizes = [{ width, height }];
  const cap = maxLevels > 0 ? Math.max(2, Math.floor(maxLevels)) : 64;
  while (sizes.length < cap) {
    const last = sizes[sizes.length - 1]!;
    if (Math.max(last.width, last.height) <= MULTIGRID_COARSEST) break;
    sizes.push({ width: Math.ceil(last.width / 2), height: Math.ceil(last.height / 2) });
  }
  return sizes;
}

/**
 * The work of one pressure solve in machine-independent units: `dispatches` launched (each costs a
 * fixed latency) and `cellVisits`, the cells they are launched over summed (what the bandwidth-bound
 * part scales with). `count` is iterations for `jacobi` / `sor` and V-cycles for `multigrid`
 * (`params` as in `FluidParams`; only the multigrid fields matter). The divergence and the projection
 * are not included: they are the same for every solver.
 */
export function pressureSolveWork(
  width: number,
  height: number,
  solver: FluidPressureSolver,
  count: number,
  params: Partial<
    Pick<FluidParams, 'multigridSmooth' | 'multigridLevels' | 'multigridCoarse'>
  > = {},
): { dispatches: number; cellVisits: number } {
  if (solver !== 'multigrid') return { dispatches: count, cellVisits: count * width * height };
  const smooth = Math.round(clamp(finiteOr(params.multigridSmooth ?? 1, 1), 1, 8));
  const sizes = multigridLevelSizes(width, height, params.multigridLevels ?? 0);
  let dispatches = 0;
  let cellVisits = 0;
  sizes.forEach((size, level) => {
    const cells = size.width * size.height;
    if (level === sizes.length - 1) {
      if (Math.max(size.width, size.height) <= MULTIGRID_COARSEST) {
        dispatches += 1;
        cellVisits += cells;
      } else {
        const sweeps = Math.round(clamp(finiteOr(params.multigridCoarse ?? 24, 24), 1, 256));
        dispatches += sweeps;
        cellVisits += sweeps * cells;
      }
      return;
    }
    // smoothing before and after, plus the residual-and-restrict and the prolongation
    dispatches += 2 * smooth + 2;
    cellVisits += (2 * smooth + 2) * cells;
  });
  return { dispatches: dispatches * count, cellVisits: cellVisits * count };
}

/** The shader chunks every fluid kernel is built from, for tests and consumers that write their own kernels against the grid. */
export const fluidShaderChunks: Readonly<Record<string, string>> = Object.freeze({
  'fluid_uniforms.wgsl': FluidUniforms.wgsl,
  'fluid-core.wgsl': coreSource,
});

interface Tex {
  texture: GPUTexture;
  view: GPUTextureView;
  id: number;
}

let nextId = 1;
const ids = new WeakMap<object, number>();
function idOf(object: object): number {
  let id = ids.get(object);
  if (id === undefined) ids.set(object, (id = nextId++));
  return id;
}

interface Resources {
  grid: FluidGrid;
  velocity: Tex[];
  dye: Tex[];
  pressure: [Tex, Tex];
  divergence: Tex;
  curl: Tex;
  curlSmooth: Tex;
  fields: Tex;
  groups: Map<string, GPUBindGroup>;
  /** The multigrid pyramid, made the first time that solver runs. */
  multigrid?: Multigrid;
}

/** One level of the multigrid pyramid. Level 0 is the fine grid and owns nothing: it uses the solver's own pressure, divergence, mask and uniforms. */
interface MultigridLevel {
  width: number;
  height: number;
  /** Pressure (the correction, below level 0), ping-pong. */
  pressure: [Tex, Tex] | undefined;
  /** The index of `pressure` that holds the latest iterate. */
  current: number;
  /** Right-hand side. */
  rhs: Tex | undefined;
  /** Solid mask of this level (rgba8unorm), built from the finer one. */
  mask: Tex | undefined;
  uniforms: UniformBuffer<typeof FluidUniforms.fields> | undefined;
}

interface Multigrid {
  levels: MultigridLevel[];
  /** Pyramid shape it was made for. */
  key: string;
}

const SCHEME_PASSES: Record<FluidScheme, number> = {
  'semi-lagrangian': 1,
  maccormack: 2,
  bfecc: 3,
};

export function createFluid(options: FluidOptions): Fluid {
  const { device, maxStepsPerFrame = 4, profiler = noProfiler, label = 'fluid' } = options;
  let resolution = options.resolution ?? 128;
  let dyeResolution = options.dyeResolution ?? Math.min(1024, resolution * 4);
  let aspect = (options.width ?? 1) / (options.height ?? 1);
  const maxDimension = Math.min(device.limits.maxTextureDimension2D - 1, 4096);
  const params: FluidParams = { ...DEFAULT_FLUID_PARAMS, ...options.params };
  const hooks: FluidHooks = {};
  const uniforms = createUniformBuffer(device, FluidUniforms, `${label} params`);
  const sampler = device.createSampler({
    label: `${label} sampler`,
    magFilter: 'linear',
    minFilter: 'linear',
  });
  const forceBuffer = device.createBuffer({
    label: `${label} momentum splats`,
    size: MAX_SPLATS * SPLAT_FLOATS * 4,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });
  const dyeBuffer = device.createBuffer({
    label: `${label} dye splats`,
    size: MAX_SPLATS * SPLAT_FLOATS * 4,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });
  const forceQueue = new Float32Array(MAX_SPLATS * SPLAT_FLOATS);
  const dyeQueue = new Float32Array(MAX_SPLATS * SPLAT_FLOATS);
  let forceCount = 0;
  let dyeCount = 0;
  const mask = createObstacleMask(device, label);

  const usage =
    GPUTextureUsage.TEXTURE_BINDING |
    GPUTextureUsage.STORAGE_BINDING |
    GPUTextureUsage.RENDER_ATTACHMENT |
    GPUTextureUsage.COPY_SRC |
    GPUTextureUsage.COPY_DST;

  let destroyed = false;
  let generation = 0;
  let lastSteps = 0;
  let stepper: FixedStepper | undefined;
  let stepperSeconds = 0;
  let resources: Resources;
  // Which texture of each pool holds the current state.
  let velocityCurrent = 0;
  let dyeCurrent = 0;
  let pressureCurrent = 0;

  function makeTexture(name: string, width: number, height: number, format: GPUTextureFormat): Tex {
    const texture = device.createTexture({
      label: `${label} ${name}`,
      size: [width, height],
      format,
      usage,
    });
    return {
      texture,
      view: texture.createView({ label: `${label} ${name} view` }),
      id: idOf(texture),
    };
  }

  function allocate(width: number, height: number, dyeWidth: number, dyeHeight: number): Resources {
    mask.resize(width, height);
    const grid: FluidGrid = Object.freeze({
      width,
      height,
      aspect: width / height,
      cellSize: 1 / height,
      texelSize: Object.freeze([1 / width, 1 / height] as const),
      velocityWidth: width + 1,
      velocityHeight: height + 1,
      dyeWidth,
      dyeHeight,
    });
    const velocity = [0, 1, 2].map((i) =>
      makeTexture(`velocity ${i}`, width + 1, height + 1, FLUID_FORMAT),
    );
    // The third dye texture only exists once a scheme or a hook needs it.
    const dye = [0, 1].map((i) => makeTexture(`dye ${i}`, dyeWidth, dyeHeight, FLUID_FORMAT));
    return {
      grid,
      velocity,
      dye,
      pressure: [
        makeTexture('pressure 0', width, height, FLUID_SCALAR_FORMAT),
        makeTexture('pressure 1', width, height, FLUID_SCALAR_FORMAT),
      ],
      divergence: makeTexture('divergence', width, height, FLUID_SCALAR_FORMAT),
      curl: makeTexture('vorticity', width, height, FLUID_SCALAR_FORMAT),
      curlSmooth: makeTexture('vorticity blur', width, height, FLUID_SCALAR_FORMAT),
      fields: makeTexture('fields', width, height, FLUID_FORMAT),
      groups: new Map(),
    };
  }

  function release(r: Resources) {
    for (const t of [
      ...r.velocity,
      ...r.dye,
      ...r.pressure,
      r.divergence,
      r.curl,
      r.curlSmooth,
      r.fields,
    ]) {
      t.texture.destroy();
    }
    releaseMultigrid(r);
  }

  function releaseMultigrid(r: Resources) {
    for (const level of r.multigrid?.levels ?? []) {
      for (const t of [...(level.pressure ?? []), level.rhs, level.mask]) t?.texture.destroy();
      level.uniforms?.destroy();
    }
    r.multigrid = undefined;
  }

  function rebuild(): boolean {
    const size = computeFluidGrid(resolution, aspect, maxDimension);
    const dyeSize = computeFluidGrid(
      dyeResolution,
      size.width / size.height,
      device.limits.maxTextureDimension2D,
    );
    const current = resources?.grid;
    if (
      current &&
      current.width === size.width &&
      current.height === size.height &&
      current.dyeWidth === dyeSize.width &&
      current.dyeHeight === dyeSize.height
    ) {
      return false;
    }
    if (resources) release(resources);
    resources = allocate(size.width, size.height, dyeSize.width, dyeSize.height);
    velocityCurrent = 0;
    dyeCurrent = 0;
    pressureCurrent = 0;
    generation++;
    stepper = undefined;
    forceCount = 0;
    dyeCount = 0;
    return true;
  }

  rebuild();

  function ensureDye(count: number) {
    while (resources.dye.length < count) {
      resources.dye.push(
        makeTexture(
          `dye ${resources.dye.length}`,
          resources.grid.dyeWidth,
          resources.grid.dyeHeight,
          FLUID_FORMAT,
        ),
      );
    }
  }

  // --- Bind groups and dispatches -------------------------------------------------------------------

  type Resource = Tex | GPUBuffer | GPUSampler;
  const isTex = (item: Resource): item is Tex => 'view' in item && 'texture' in item;

  /** The uniform block and mask a pass sees as bindings 0 and 1: the solver's own, or a multigrid level's. */
  interface LevelBinding {
    id: number;
    uniform: GPUBuffer;
    mask: GPUTextureView;
  }

  /** The bind group of a pass over these resources (bindings 0 and 1 are filled in), cached. */
  function groupFor(
    name: PassName,
    bound: Readonly<Record<number, Resource>>,
    level?: LevelBinding,
  ): GPUBindGroup {
    const { layout, spec: passSpec } = getPipeline(device, name);
    const all: Record<number, Resource> = { ...bound };
    const key = [
      name,
      level ? `L${level.id}` : '',
      ...Object.keys(passSpec.bindings).map((b) => {
        const binding = Number(b);
        if (binding < 2) return '';
        const item = all[binding];
        if (!item) throw new Error(`fluid pass ${name} is missing binding ${binding}`);
        return `${binding}:${isTex(item) ? item.id : idOf(item)}`;
      }),
    ].join('|');
    let group = resources.groups.get(key);
    if (!group) {
      const entries: GPUBindGroupEntry[] = [
        { binding: 0, resource: { buffer: level ? level.uniform : uniforms.buffer } },
        { binding: 1, resource: level ? level.mask : mask.view },
      ];
      for (const b of Object.keys(passSpec.bindings)) {
        const binding = Number(b);
        if (binding < 2) continue;
        const item = all[binding]!;
        entries.push({
          binding,
          resource: isTex(item) ? item.view : 'size' in item ? { buffer: item } : item,
        });
      }
      group = device.createBindGroup({ label: `fluid ${passSpec.entry}`, layout, entries });
      resources.groups.set(key, group);
    }
    return group;
  }

  function begin(
    encoder: GPUCommandEncoder,
    name: string,
    timestamp?: string,
  ): GPUComputePassEncoder {
    return encoder.beginComputePass({
      label: `fluid ${name}`,
      timestampWrites: timestamp
        ? (profiler.pass(timestamp) as GPUComputePassTimestampWrites | undefined)
        : undefined,
    });
  }

  function dispatch(
    pass: GPUComputePassEncoder,
    name: PassName,
    bound: Readonly<Record<number, Resource>>,
    width: number,
    height: number,
    level?: LevelBinding,
  ) {
    pass.setPipeline(getPipeline(device, name).pipeline);
    pass.setBindGroup(0, groupFor(name, bound, level));
    pass.dispatchWorkgroups(Math.ceil(width / WORKGROUP), Math.ceil(height / WORKGROUP));
  }

  /** The two velocity pool indices that are not `current`. */
  const others = (current: number): [number, number] => {
    const rest = [0, 1, 2].filter((i) => i !== current);
    return [rest[0]!, rest[1]!];
  };
  /** The dye pool indices that are not current: one with two textures, two with three. */
  const freeDye = (): number[] => resources.dye.map((_, i) => i).filter((i) => i !== dyeCurrent);

  // --- Uniforms -------------------------------------------------------------------------------------

  const stepRate = () => clamp(finiteOr(params.stepRate, 60), 10, 240);
  const multigridCoarseSweeps = () =>
    Math.round(clamp(finiteOr(params.multigridCoarse, 24), 1, 256));
  const stepSeconds = () => 1 / stepRate();

  function writeUniforms() {
    const { grid } = resources;
    const dt = stepSeconds();
    const maxCfl = clamp(finiteOr(params.maxCfl, 6), 0.5, 32);
    const boundary = params.boundary === 'open' ? 1 : params.boundary === 'wrap' ? 2 : 0;
    const nonNegative = (x: number) => Math.max(0, finiteOr(x, 0));
    const viscosity = nonNegative(params.viscosity);
    const direction = params.buoyancyDirection;
    const length = Math.hypot(direction[0], direction[1]) || 1;
    uniforms.set({
      grid: [grid.width, grid.height],
      dye_size: [grid.dyeWidth, grid.dyeHeight],
      dt,
      cells_per_h: grid.height,
      aspect: grid.aspect,
      max_disp: maxCfl,
      boundary,
      solid: mask.enabled ? 1 : 0,
      rk2: params.rk2 ? 1 : 0,
      limit: params.limiter ? 1 : 0,
      force_count: forceCount,
      dye_count: dyeCount,
      vorticity: nonNegative(params.vorticity),
      buoyancy: finiteOr(params.buoyancy, 0),
      vel_decay: Math.exp(-nonNegative(params.velocityDissipation) * dt),
      dye_decay: Math.exp(-nonNegative(params.dyeDissipation) * dt),
      diffuse_a: viscosity * dt * grid.height * grid.height,
      jacobi_w: clamp(finiteOr(params.jacobiWeight, 0.8), 0.3, 1),
      sor_omega:
        params.sorOmega > 0 ? clamp(params.sorOmega, 1, 1.95) : sorOmega(grid.width, grid.height),
      vmax: maxCfl / (dt * grid.height),
      dye_max: clamp(finiteOr(params.dyeLimit, 64), 1, DYE_LIMIT),
      accel: [finiteOr(params.gravity[0], 0), finiteOr(params.gravity[1], 0)],
      buoy_dir: [finiteOr(direction[0], 0) / length, finiteOr(direction[1], -1) / length],
      dye_filter: DYE_FILTER_KIND[params.dyeVelocityFilter] ?? 0,
      mg_sweeps: multigridCoarseSweeps(),
    });
    uniforms.upload();
  }

  // --- Passes -------------------------------------------------------------------------------------------

  function advectVelocity(pass: GPUComputePassEncoder) {
    const r = resources;
    const src = r.velocity[velocityCurrent]!;
    const [a, b] = others(velocityCurrent).map((i) => r.velocity[i]!) as [Tex, Tex];
    const size: [number, number] = [r.grid.velocityWidth, r.grid.velocityHeight];
    switch (params.velocityScheme) {
      case 'maccormack':
        dispatch(pass, 'velPred', { 2: src, 4: a }, ...size);
        dispatch(pass, 'velMc', { 2: src, 3: a, 4: b }, ...size);
        velocityCurrent = r.velocity.indexOf(b);
        break;
      case 'bfecc':
        dispatch(pass, 'velPred', { 2: src, 4: a }, ...size);
        dispatch(pass, 'velBfeccError', { 2: src, 3: a, 4: b }, ...size);
        dispatch(pass, 'velBfeccFinal', { 2: src, 3: b, 4: a }, ...size);
        velocityCurrent = r.velocity.indexOf(a);
        break;
      default:
        dispatch(pass, 'velSl', { 2: src, 4: a }, ...size);
        velocityCurrent = r.velocity.indexOf(a);
    }
  }

  function advectDye(pass: GPUComputePassEncoder) {
    const r = resources;
    const scheme = params.dyeScheme;
    if (scheme !== 'semi-lagrangian') ensureDye(3);
    const src = r.dye[dyeCurrent]!;
    const [a, b] = freeDye().map((i) => r.dye[i]!) as [Tex, Tex];
    const flow = r.velocity[velocityCurrent]!;
    const size: [number, number] = [r.grid.dyeWidth, r.grid.dyeHeight];
    if (scheme === 'semi-lagrangian') {
      dispatch(pass, 'dyeSl', { 2: src, 4: a, 5: flow, 6: sampler }, ...size);
      dyeCurrent = r.dye.indexOf(a);
      return;
    }
    if (scheme === 'maccormack') {
      dispatch(pass, 'dyePred', { 2: src, 4: a, 5: flow, 6: sampler }, ...size);
      dispatch(pass, 'dyeMc', { 2: src, 3: a, 4: b, 5: flow, 6: sampler }, ...size);
      dyeCurrent = r.dye.indexOf(b);
    } else {
      dispatch(pass, 'dyePred', { 2: src, 4: a, 5: flow, 6: sampler }, ...size);
      dispatch(pass, 'dyeBfeccError', { 2: src, 3: a, 4: b, 5: flow, 6: sampler }, ...size);
      dispatch(pass, 'dyeBfeccFinal', { 2: src, 3: b, 4: a, 5: flow, 6: sampler }, ...size);
      dyeCurrent = r.dye.indexOf(a);
    }
  }

  function applyForces(pass: GPUComputePassEncoder) {
    const r = resources;
    const src = r.velocity[velocityCurrent]!;
    const size: [number, number] = [r.grid.velocityWidth, r.grid.velocityHeight];
    if (params.vorticity > 0) {
      dispatch(pass, 'curl', { 2: src, 8: r.curl }, r.grid.width, r.grid.height);
      dispatch(pass, 'curlSmooth', { 5: r.curl, 10: r.curlSmooth }, r.grid.width, r.grid.height);
    }
    const [a] = others(velocityCurrent).map((i) => r.velocity[i]!) as [Tex, Tex];
    dispatch(
      pass,
      'forces',
      { 2: src, 3: a, 5: r.curl, 6: r.dye[dyeCurrent]!, 7: sampler, 11: r.curlSmooth },
      ...size,
    );
    velocityCurrent = r.velocity.indexOf(a);

    if (params.viscosity > 0) {
      const iterations = Math.round(clamp(finiteOr(params.viscosityIterations, 12), 1, 64));
      // Jacobi iterations of the implicit diffusion. `base` (the forced velocity) is read by every
      // iteration; the other two textures take turns being the iterate and the output.
      const base = a;
      const [t1, t2] = others(velocityCurrent).map((i) => r.velocity[i]!) as [Tex, Tex];
      let from = base;
      let to = t1;
      for (let i = 0; i < iterations; i++) {
        dispatch(pass, 'diffuse', { 2: from, 3: to, 9: base }, ...size);
        from = to;
        to = to === t1 ? t2 : t1;
      }
      velocityCurrent = r.velocity.indexOf(from);
    }
  }

  /** Makes the pyramid for the current grid and parameters if it is not there, and writes its uniforms. */
  function prepareMultigrid(): Multigrid {
    const r = resources;
    const sizes = multigridLevelSizes(r.grid.width, r.grid.height, params.multigridLevels);
    const key = sizes.map((size) => `${size.width}x${size.height}`).join(',');
    if (r.multigrid?.key !== key) {
      releaseMultigrid(r);
      const levels = sizes.map((size, index): MultigridLevel => {
        if (index === 0) {
          return {
            ...size,
            pressure: undefined,
            current: 0,
            rhs: undefined,
            mask: undefined,
            uniforms: undefined,
          };
        }
        const name = `mg ${index}`;
        return {
          ...size,
          pressure: [
            makeTexture(`${name} pressure 0`, size.width, size.height, FLUID_SCALAR_FORMAT),
            makeTexture(`${name} pressure 1`, size.width, size.height, FLUID_SCALAR_FORMAT),
          ],
          current: 0,
          rhs: makeTexture(`${name} rhs`, size.width, size.height, FLUID_SCALAR_FORMAT),
          mask: makeTexture(`${name} mask`, size.width, size.height, 'rgba8unorm'),
          uniforms: createUniformBuffer(device, FluidUniforms, `${label} ${name} params`),
        };
      });
      r.multigrid = { levels, key };
    }
    const { levels } = r.multigrid!;
    const boundary = params.boundary === 'open' ? 1 : params.boundary === 'wrap' ? 2 : 0;
    for (const level of levels) {
      level.uniforms?.set({
        grid: [level.width, level.height],
        dye_size: [r.grid.dyeWidth, r.grid.dyeHeight],
        boundary,
        solid: mask.enabled ? 1 : 0,
        jacobi_w: clamp(finiteOr(params.jacobiWeight, 0.8), 0.3, 1),
        sor_omega: sorOmega(level.width, level.height),
        mg_sweeps: multigridCoarseSweeps(),
      });
      level.uniforms?.upload();
    }
    return r.multigrid!;
  }

  /**
   * A multigrid solve of the pressure: `cycles` V-cycles from the current pressure (zero for the
   * first when `warmStart` is off). See fluid-multigrid.wgsl for the scheme. Leaves the result in
   * `pressureCurrent`.
   */
  function solveMultigrid(pass: GPUComputePassEncoder, cycles: number) {
    const r = resources;
    const { levels } = prepareMultigrid();
    const last = levels.length - 1;
    const smooth = Math.round(clamp(finiteOr(params.multigridSmooth, 1), 1, 8));
    const coarseSweeps = multigridCoarseSweeps();
    levels[0]!.current = pressureCurrent;

    const binding = (index: number): LevelBinding | undefined => {
      const level = levels[index]!;
      return index === 0
        ? undefined
        : { id: index, uniform: level.uniforms!.buffer, mask: level.mask!.view };
    };
    const current = (index: number): Tex => {
      const level = levels[index]!;
      return index === 0 ? r.pressure[level.current]! : level.pressure![level.current]!;
    };
    const next = (index: number): Tex => {
      const level = levels[index]!;
      return index === 0 ? r.pressure[1 - level.current]! : level.pressure![1 - level.current]!;
    };
    const flip = (index: number) => {
      levels[index]!.current = 1 - levels[index]!.current;
    };
    const rhs = (index: number): Tex => (index === 0 ? r.divergence : levels[index]!.rhs!);

    const relax = (index: number, count: number, cold: boolean) => {
      const level = levels[index]!;
      for (let i = 0; i < count; i++) {
        dispatch(
          pass,
          cold && i === 0 ? 'jacobiCold' : 'jacobi',
          { 3: rhs(index), 4: current(index), 5: next(index) },
          level.width,
          level.height,
          binding(index),
        );
        flip(index);
      }
    };

    const cycle = (index: number, cold: boolean) => {
      const level = levels[index]!;
      if (index === last) {
        if (Math.max(level.width, level.height) <= MULTIGRID_COARSEST) {
          dispatch(
            pass,
            'mgCoarse',
            { 2: rhs(index), 5: next(index) },
            WORKGROUP,
            WORKGROUP,
            binding(index),
          );
          flip(index);
        } else {
          relax(index, coarseSweeps, true);
        }
        return;
      }
      const below = levels[index + 1]!;
      relax(index, smooth, cold);
      dispatch(
        pass,
        'mgRestrict',
        { 2: rhs(index), 3: current(index), 5: rhs(index + 1) },
        below.width,
        below.height,
        binding(index),
      );
      cycle(index + 1, true);
      dispatch(
        pass,
        'mgProlong',
        { 3: current(index), 4: current(index + 1), 5: next(index), 6: levels[index + 1]!.mask! },
        level.width,
        level.height,
        binding(index),
      );
      flip(index);
      relax(index, smooth, false);
    };

    if (mask.enabled) {
      for (let index = 0; index < last; index++) {
        const below = levels[index + 1]!;
        dispatch(pass, 'mgMask', { 7: below.mask! }, below.width, below.height, binding(index));
      }
    }
    for (let i = 0; i < cycles; i++) cycle(0, i === 0 && !params.warmStart);
    pressureCurrent = levels[0]!.current;
  }

  function solvePressure(pass: GPUComputePassEncoder, iterations: number, cycles = 1) {
    const r = resources;
    const width = r.grid.width;
    const height = r.grid.height;
    const velocity = r.velocity[velocityCurrent]!;
    dispatch(pass, 'divergence', { 2: velocity, 6: r.divergence }, width, height);
    if (params.pressureSolver === 'multigrid') {
      solveMultigrid(pass, Math.max(1, Math.round(cycles)));
      const [dst] = others(velocityCurrent).map((i) => r.velocity[i]!) as [Tex, Tex];
      dispatch(
        pass,
        'project',
        { 2: velocity, 4: r.pressure[pressureCurrent]!, 7: dst },
        r.grid.velocityWidth,
        r.grid.velocityHeight,
      );
      velocityCurrent = r.velocity.indexOf(dst);
      return;
    }
    let current = pressureCurrent;
    const count = Math.max(1, Math.round(iterations));
    for (let i = 0; i < count; i++) {
      const from = r.pressure[current]!;
      const to = r.pressure[1 - current]!;
      const cold = i === 0 && !params.warmStart;
      const name: PassName =
        params.pressureSolver === 'jacobi'
          ? cold
            ? 'jacobiCold'
            : 'jacobi'
          : i % 2 === 1
            ? 'sorBlack'
            : cold
              ? 'sorRedCold'
              : 'sorRed';
      dispatch(pass, name, { 3: r.divergence, 4: from, 5: to }, width, height);
      current = 1 - current;
    }
    pressureCurrent = current;
    const [dst] = others(velocityCurrent).map((i) => r.velocity[i]!) as [Tex, Tex];
    dispatch(
      pass,
      'project',
      { 2: velocity, 4: r.pressure[pressureCurrent]!, 7: dst },
      r.grid.velocityWidth,
      r.grid.velocityHeight,
    );
    velocityCurrent = r.velocity.indexOf(dst);
  }

  function inject(encoder: GPUCommandEncoder) {
    if (forceCount === 0 && dyeCount === 0) return;
    const r = resources;
    if (forceCount > 0) {
      device.queue.writeBuffer(forceBuffer, 0, forceQueue, 0, forceCount * SPLAT_FLOATS);
    }
    if (dyeCount > 0) {
      device.queue.writeBuffer(dyeBuffer, 0, dyeQueue, 0, dyeCount * SPLAT_FLOATS);
    }
    const pass = begin(encoder, 'inject', 'inject');
    if (forceCount > 0) {
      const src = r.velocity[velocityCurrent]!;
      const [a] = others(velocityCurrent).map((i) => r.velocity[i]!) as [Tex, Tex];
      dispatch(
        pass,
        'splatVelocity',
        { 2: src, 3: a, 4: forceBuffer },
        r.grid.velocityWidth,
        r.grid.velocityHeight,
      );
      velocityCurrent = r.velocity.indexOf(a);
    }
    if (dyeCount > 0) {
      const src = r.dye[dyeCurrent]!;
      const dst = r.dye[freeDye()[0]!]!;
      dispatch(
        pass,
        'splatDye',
        { 2: src, 3: dst, 4: dyeBuffer },
        r.grid.dyeWidth,
        r.grid.dyeHeight,
      );
      dyeCurrent = r.dye.indexOf(dst);
    }
    pass.end();
    forceCount = 0;
    dyeCount = 0;
  }

  function step(encoder: GPUCommandEncoder, first: boolean, last: boolean) {
    const r = resources;
    const dt = stepSeconds();
    if (!params.dyeOnly) {
      const velocityPass = begin(encoder, 'velocity', first ? 'advect + forces' : undefined);
      advectVelocity(velocityPass);
      applyForces(velocityPass);
      velocityPass.end();

      if (hooks.forces) {
        const [target] = others(velocityCurrent);
        const wrote = hooks.forces({
          encoder,
          dt,
          grid: r.grid,
          read: r.velocity[velocityCurrent]!.view,
          write: r.velocity[target]!.view,
          dye: r.dye[dyeCurrent]!.view,
        });
        if (wrote) velocityCurrent = target;
      }

      const pressurePass = begin(encoder, 'pressure', first ? 'pressure' : undefined);
      solvePressure(
        pressurePass,
        params.pressureIterations,
        Math.round(clamp(finiteOr(params.multigridCycles, 1), 1, 8)),
      );
      pressurePass.end();
    }

    const dyePass = begin(encoder, 'dye', first ? 'dye' : last ? 'dye (last step)' : undefined);
    advectDye(dyePass);
    dyePass.end();

    if (hooks.dye) {
      ensureDye(3);
      const target = (dyeCurrent + 1) % r.dye.length;
      const wrote = hooks.dye({
        encoder,
        dt,
        grid: r.grid,
        read: r.dye[dyeCurrent]!.view,
        write: r.dye[target]!.view,
        velocity: r.velocity[velocityCurrent]!.view,
      });
      if (wrote) dyeCurrent = target;
    }
  }

  function resolveFields(encoder: GPUCommandEncoder, timestamp = 'resolve') {
    const r = resources;
    const pass = begin(encoder, 'resolve', timestamp);
    dispatch(
      pass,
      'resolve',
      { 2: r.velocity[velocityCurrent]!, 4: r.pressure[pressureCurrent]!, 8: r.fields },
      r.grid.width,
      r.grid.height,
    );
    pass.end();
  }

  function run(encoder: GPUCommandEncoder, steps: number) {
    if (destroyed) return;
    const count = Math.max(0, Math.floor(finiteOr(steps, 0)));
    lastSteps = count;
    if (count === 0 && forceCount === 0 && dyeCount === 0) return;
    writeUniforms();
    inject(encoder);
    for (let i = 0; i < count; i++) step(encoder, i === 0, i === count - 1 && count > 1);
    resolveFields(encoder);
  }

  // --- Queues -------------------------------------------------------------------------------------------

  function enqueue(
    queue: Float32Array,
    count: number,
    from: readonly [number, number],
    to: readonly [number, number],
    radius: number,
    ramp: number,
    value: readonly number[],
  ): boolean {
    if (count >= MAX_SPLATS) return false;
    const numbers = [from[0], from[1], to[0], to[1], radius, ramp, ...value];
    if (!numbers.every(Number.isFinite)) return false;
    const o = count * SPLAT_FLOATS;
    queue[o] = from[0];
    queue[o + 1] = from[1];
    queue[o + 2] = to[0];
    queue[o + 3] = to[1];
    queue[o + 4] = Math.max(radius, 1e-4);
    queue[o + 5] = Math.max(ramp, 0);
    queue[o + 6] = 0;
    queue[o + 7] = 0;
    queue[o + 8] = value[0] ?? 0;
    queue[o + 9] = value[1] ?? 0;
    queue[o + 10] = value[2] ?? 0;
    queue[o + 11] = value[3] ?? 0;
    return true;
  }

  // --- Uploads --------------------------------------------------------------------------------------------

  function upload(tex: Tex, width: number, height: number, data: Uint16Array) {
    device.queue.writeTexture(
      { texture: tex.texture },
      data as Uint16Array<ArrayBuffer>,
      { bytesPerRow: width * 8 },
      [width, height],
    );
  }

  const outputOf = (tex: Tex, slot: number): FluidTexture => ({
    texture: tex.texture,
    view: tex.view,
    slot,
  });

  return {
    params,
    hooks,
    get grid() {
      return resources.grid;
    },
    get generation() {
      return generation;
    },
    get fields() {
      return resources.fields.texture;
    },
    get fieldsView() {
      return resources.fields.view;
    },
    get dye() {
      return outputOf(resources.dye[dyeCurrent]!, dyeCurrent);
    },
    get velocity() {
      return outputOf(resources.velocity[velocityCurrent]!, velocityCurrent);
    },
    get pressure() {
      return outputOf(resources.pressure[pressureCurrent]!, pressureCurrent);
    },
    get divergence() {
      return resources.divergence.texture;
    },
    obstacles: mask,
    get stepSeconds() {
      return stepSeconds();
    },
    get lastSteps() {
      return lastSteps;
    },
    get pendingSplats() {
      return [forceCount, dyeCount] as const;
    },

    resize(width, height) {
      if (!(width > 0 && height > 0)) return false;
      aspect = width / height;
      return rebuild();
    },
    setResolution(cells, dyeCells) {
      if (!Number.isFinite(cells)) return false;
      resolution = cells;
      if (dyeCells !== undefined && Number.isFinite(dyeCells)) dyeResolution = dyeCells;
      return rebuild();
    },

    addForce(from, to, radius, velocity, ramp = 0) {
      if (!enqueue(forceQueue, forceCount, from, to, radius, ramp, [velocity[0], velocity[1]])) {
        return false;
      }
      forceCount++;
      return true;
    },
    addDye(from, to, radius, value, ramp = 0) {
      if (!enqueue(dyeQueue, dyeCount, from, to, radius, ramp, value)) return false;
      dyeCount++;
      return true;
    },

    update(encoder, delta) {
      if (destroyed) return 0;
      const dt = stepSeconds();
      if (!stepper || Math.abs(dt - stepperSeconds) > dt * 1e-3) {
        stepper = createFixedStepper(dt, { maxSteps: maxStepsPerFrame });
        stepperSeconds = dt;
      }
      const { steps } = stepper.advance(Number.isFinite(delta) ? delta : 0);
      run(encoder, steps);
      return steps;
    },
    advance(encoder, steps) {
      run(encoder, steps);
    },
    project(encoder, iterations) {
      if (destroyed) return;
      writeUniforms();
      const pass = begin(encoder, 'project', 'project');
      // `iterations` counts V-cycles when the solver is multigrid.
      solvePressure(
        pass,
        iterations ?? params.pressureIterations,
        iterations ?? Math.round(clamp(finiteOr(params.multigridCycles, 1), 1, 8)),
      );
      pass.end();
    },
    resolve(encoder) {
      if (destroyed) return;
      writeUniforms();
      resolveFields(encoder, 'resolve');
    },

    setVelocity(field) {
      const { width, height, velocityWidth, velocityHeight } = resources.grid;
      const data = new Uint16Array(velocityWidth * velocityHeight * 4);
      for (let j = 0; j < velocityHeight; j++) {
        for (let i = 0; i < velocityWidth; i++) {
          const u = field(i / width, (Math.min(j, height - 1) + 0.5) / height)[0];
          const v = field((Math.min(i, width - 1) + 0.5) / width, j / height)[1];
          const o = (j * velocityWidth + i) * 4;
          data[o] = floatToHalf(finiteOr(u, 0));
          data[o + 1] = floatToHalf(finiteOr(v, 0));
        }
      }
      upload(resources.velocity[velocityCurrent]!, velocityWidth, velocityHeight, data);
    },
    setDye(field) {
      const { dyeWidth, dyeHeight } = resources.grid;
      const data = new Uint16Array(dyeWidth * dyeHeight * 4);
      for (let j = 0; j < dyeHeight; j++) {
        for (let i = 0; i < dyeWidth; i++) {
          const value = field((i + 0.5) / dyeWidth, (j + 0.5) / dyeHeight);
          const o = (j * dyeWidth + i) * 4;
          for (let c = 0; c < 4; c++) data[o + c] = floatToHalf(finiteOr(value[c] ?? 0, 0));
        }
      }
      upload(resources.dye[dyeCurrent]!, dyeWidth, dyeHeight, data);
    },
    reset(encoder) {
      forceCount = 0;
      dyeCount = 0;
      const r = resources;
      for (const tex of [
        ...r.velocity,
        ...r.dye,
        ...r.pressure,
        r.divergence,
        r.curl,
        r.curlSmooth,
        r.fields,
      ]) {
        encoder
          .beginRenderPass({
            label: `${label} clear`,
            colorAttachments: [{ view: tex.view, loadOp: 'clear', storeOp: 'store' }],
          })
          .end();
      }
      velocityCurrent = 0;
      dyeCurrent = 0;
      pressureCurrent = 0;
    },
    destroy() {
      if (destroyed) return;
      destroyed = true;
      release(resources);
      mask.destroy();
      uniforms.destroy();
      forceBuffer.destroy();
      dyeBuffer.destroy();
    },
  };
}

/** How many dispatches the advection of one field takes in a scheme (1, 2 or 3). */
export function schemePassCount(scheme: FluidScheme): number {
  return SCHEME_PASSES[scheme];
}
