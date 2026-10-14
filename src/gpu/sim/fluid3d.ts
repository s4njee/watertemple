import { createFixedStepper, type FixedStepper } from '../loop.ts';
import { noProfiler, type PassProfiler } from '../profiler.ts';
import { createShader } from '../shader.ts';
import { createUniformBuffer, defineUniformStruct } from '../uniforms.ts';
import advectSource from './fluid3d-advect.wgsl?raw';
import coreSource from './fluid3d-core.wgsl?raw';
import forcesSource from './fluid3d-forces.wgsl?raw';
import projectSource from './fluid3d-project.wgsl?raw';
import { floatToHalf } from './fluid.ts';

// The 3D stable-fluids solver (story 6.9): incompressible flow in a closed box on a grid, in compute
// shaders, state in 3D textures. The same method as the 2D solver (fluid.ts, read its header first), carried
// into three dimensions:
//
//     advect velocity -> forces (gravity, buoyancy from the ink, vorticity confinement) -> divergence
//       -> pressure solve -> subtract the pressure gradient -> advect the ink
//
// plus splats of momentum and ink injected before the steps (a pointer drag, a falling drop). See
// fluid3d-core.wgsl for the grid and samplers, fluid3d-advect.wgsl, fluid3d-forces.wgsl and
// fluid3d-project.wgsl for the passes, and src/effects/fluid-3d/notes.md for the measurements.
//
// GRID. Staggered (MAC), as in 2D: velocity components on the faces of nx x ny x nz cubic cells in one
// rgba16float 3D texture of (nx+1) x (ny+1) x (nz+1) texels, pressure at the cell centres. A collocated grid
// would read the velocity with one filtered tap instead of three, but its divergence and gradient are wide
// differences that do not match the compact pressure laplacian (a residual at short wavelengths, a leaky
// one-cell wall); at 64 cells that residual is a visible fraction of a vortex, so the exact projection won.
// The cost is measured in notes.md.
//
// TWO RESOLUTIONS, as in 2D: the velocity grid is coarse (48 to 96 cells on the long side), the ink lives on its
// own grid (64 to 160) carried by the coarse velocity, which is where the visible detail comes from.
// Everything is rgba16float (renderable, filterable and write-only storage on every device) except the
// pressure and divergence, which are r32float and read with textureLoad.
//
// UNITS. Positions and lengths in the API are in LENGTH units: the box is [0, size.x] x [0, size.y] x
// [0, size.z] with y UP, and the longest side defaults to 1. Velocity is length per second. The scalar field
// ("density") has the ink's colour as an optical density per unit length in rgb (so two inks add, which
// is subtractive mixing) and its amount in a (what buoyancy and scattering follow).
//
// PRESSURE. Warm-started damped Jacobi, or a multigrid V-cycle built from the same sweeps (see
// fluid3d-project.wgsl). Jacobi needs of the order of N^2 sweeps for the long wavelengths of an N-cell grid; a V-cycle
// costs a handful of fine sweeps and converges per cycle independent of N.
//
// STABILITY. Semi-Lagrangian advection is unconditionally stable and the MacCormack correction is
// limited to the range of the samples it interpolates. Time advances in fixed steps with a cap, the
// trace distance is capped at `maxCfl` cells, velocities and forces are clamped to that speed and
// non-finite values are replaced by zero, so no input blows the field up.

export type Fluid3dScheme = 'semi-lagrangian' | 'maccormack';
export type Fluid3dPressureSolver = 'multigrid' | 'jacobi';

/** Format of velocity, ink and the resolved fields. */
export const FLUID3D_FORMAT: GPUTextureFormat = 'rgba16float';
/** Format of pressure, divergence and the blurred vorticity. */
export const FLUID3D_SCALAR_FORMAT: GPUTextureFormat = 'r32float';
/** Splats of each kind (momentum, ink) accepted between two updates; more are dropped. */
export const MAX_SPLATS_3D = 128;
const SPLAT_FLOATS = 12;
/** Default workgroup shape. 64 invocations; see the workgroup table in notes.md. */
export const DEFAULT_WORKGROUP: readonly [number, number, number] = [4, 4, 4];
const SCALAR_LIMIT = 64;
const MAX_COARSENINGS = 3;

/** Live, mutable parameters: change them any time, they are read on every update. */
export interface Fluid3dParams {
  /** Fixed simulation steps per second. The flow looks the same at any display rate. Default 60. */
  stepRate: number;
  /** `multigrid` (default) or plain `jacobi` sweeps. */
  pressureSolver: Fluid3dPressureSolver;
  /** Damping of the Jacobi sweeps, 0.3 to 1. Default 6/7, the best smoothing weight of the 7-point laplacian; see fluid3d-project.wgsl. */
  jacobiWeight: number;
  /** Pressure work per step: Jacobi sweeps, or V-cycles for `multigrid`. Default 1 V-cycle. */
  pressureIterations: number;
  /** Sweeps before and after the coarse-grid correction on each level of a V-cycle. Default 2. */
  smoothing: number;
  /** Sweeps on the coarsest level. Default 12. */
  coarseIterations: number;
  /** Start the solve from last step's pressure (a much better guess) instead of zero. Default true. */
  warmStart: boolean;
  /** Advection of the velocity. Default `semi-lagrangian`. */
  velocityScheme: Fluid3dScheme;
  /** Advection of the ink. Default `maccormack`. */
  scalarScheme: Fluid3dScheme;
  /** Midpoint (second-order) back-tracing instead of Euler. Default true. */
  rk2: boolean;
  /** Clamp the MacCormack correction to the range of the samples it interpolates. Default true. */
  limiter: boolean;
  /** Vorticity confinement strength epsilon, 1 / s. 0 is off (and costs nothing). */
  vorticity: number;
  /** Velocity decay rate, 1 / s: v *= exp(-rate * dt). */
  velocityDissipation: number;
  /** Ink decay rate, 1 / s: ink *= exp(-rate * dt). */
  scalarDissipation: number;
  /** Uniform acceleration, length per second squared: [x, y, z], y UP. A uniform body force is balanced by pressure; it only matters for what differs from the water. */
  gravity: readonly [number, number, number];
  /** Acceleration per unit of ink amount (the scalar's alpha) along `buoyancyDirection`. Positive sinks ink in water; negative lifts smoke. */
  buoyancy: number;
  /** Direction of the buoyancy acceleration. Default [0, -1, 0], down. */
  buoyancyDirection: readonly [number, number, number];
  /** Largest distance a parcel travels in one step, in cells. Also caps velocity. Default 6. */
  maxCfl: number;
  /** Largest ink value (any channel). Default 64. */
  scalarLimit: number;
  /**
   * Only advect the ink (and apply its dissipation) by the velocity as it is: no velocity advection,
   * forces or pressure. For prescribed flows and tests.
   */
  scalarOnly: boolean;
  /** A solid sphere in the box: centre in length units and radius; radius 0 is none. Multigrid is not used while it is on (Jacobi is). */
  obstacle: { center: readonly [number, number, number]; radius: number };
}

export const DEFAULT_FLUID3D_PARAMS: Readonly<Fluid3dParams> = {
  stepRate: 60,
  pressureSolver: 'multigrid',
  jacobiWeight: 6 / 7,
  pressureIterations: 1,
  smoothing: 2,
  coarseIterations: 12,
  warmStart: true,
  velocityScheme: 'semi-lagrangian',
  scalarScheme: 'maccormack',
  rk2: true,
  limiter: true,
  vorticity: 0,
  velocityDissipation: 0.1,
  scalarDissipation: 0.05,
  gravity: [0, 0, 0],
  buoyancy: 0,
  buoyancyDirection: [0, -1, 0],
  maxCfl: 6,
  scalarLimit: 64,
  scalarOnly: false,
  obstacle: { center: [0.5, 0.5, 0.5], radius: 0 },
};

/** The grids and what a shading pass needs to know about them. */
export interface Fluid3dGrid {
  /** Cells along x, y and z. Each is a multiple of 8. */
  readonly width: number;
  readonly height: number;
  readonly depth: number;
  /** Side of a cell in length units. */
  readonly cellSize: number;
  /** The box in length units: [width, height, depth] * cellSize. */
  readonly size: readonly [number, number, number];
  /** Size of the velocity texture: one more than the cells on each axis. */
  readonly velocitySize: readonly [number, number, number];
  /** Size of the scalar texture. */
  readonly scalarSize: readonly [number, number, number];
  /** Grids in the multigrid hierarchy, the finest included. */
  readonly levels: number;
}

/** A 3D texture the solver owns, with its default view. */
export interface Fluid3dTexture {
  readonly texture: GPUTexture;
  readonly view: GPUTextureView;
  /**
   * Which of the solver's interchangeable textures this is. Fields that are ping-ponged (velocity,
   * scalar) change texture after every step: cache bind groups by `slot`, and drop the cache when
   * `generation` changes.
   */
  readonly slot: number;
}

export interface Fluid3dOptions {
  device: GPUDevice;
  /** Cells along the longest side of the velocity grid (rounded to a multiple of 8, 16 to 256). Default 64. */
  resolution?: number;
  /** Texels along the longest side of the scalar grid (16 to 256). Default 96. */
  scalarResolution?: number;
  /** Relative extents of the box [x, y, z]; the longest becomes `1`. Default [1, 1, 1]. */
  size?: readonly [number, number, number];
  /** Initial parameters, merged over DEFAULT_FLUID3D_PARAMS. */
  params?: Partial<Fluid3dParams>;
  /** Most steps one `update` runs. Default 3. */
  maxStepsPerFrame?: number;
  /** Compute workgroup shape, at most 256 invocations. Default 4 x 4 x 4. */
  workgroup?: readonly [number, number, number];
  /** Receives timestamp slots for the first step's phases, the last step's scalar pass and the injection. */
  profiler?: PassProfiler;
  label?: string;
}

export interface Fluid3d {
  /** Live parameters. Mutate freely. */
  readonly params: Fluid3dParams;
  /** The current grids. A new object after every re-creation. */
  readonly grid: Fluid3dGrid;
  /** Increments every time the textures are re-created: rebuild bind groups that hold any of them. */
  readonly generation: number;
  /** The ink (scalar) texture: rgb optical density per unit length, a amount; rgba16float, filterable. Changes after every step: cache by `slot`. */
  readonly density: Fluid3dTexture;
  /** The staggered velocity texture the solver works on. Changes after every step. */
  readonly velocity: Fluid3dTexture;
  /** Pressure potential (r32float, textureLoad only), latest solve. */
  readonly pressure: Fluid3dTexture;
  /** The divergence the last projection removed (r32float, textureLoad only). */
  readonly divergence: GPUTexture;
  /**
   * Cell-centred velocity (rgb, length per second), cells sized, rgba16float, filterable, STABLE between
   * updates: for debug views. Written by `resolve` and, with confinement on, by every step.
   */
  readonly fields: GPUTexture;
  readonly fieldsView: GPUTextureView;
  /** Seconds per simulation step at the current `stepRate`. */
  readonly stepSeconds: number;
  /** Steps run by the last `update` / `advance`. */
  readonly lastSteps: number;
  /** Splats queued and not yet applied: [momentum, ink]. */
  readonly pendingSplats: readonly [number, number];

  /** Changes the cells of the velocity grid and the texels of the ink grid; re-creates (all state cleared, `generation` bumped) if that changes anything. */
  setResolution(cells: number, scalarCells?: number): boolean;
  /** Changes the relative extents of the box; re-creates if the grid changes. */
  setSize(size: readonly [number, number, number]): boolean;
  /** Changes the workgroup shape; rebuilds pipelines, keeps the state. */
  setWorkgroup(workgroup: readonly [number, number, number]): void;

  /**
   * Queues a push: adds `velocity` (length per second) to the fluid along the segment `from` to `to`
   * (length units, y up) with a Gaussian footprint of `radius`. For a point pass the same position
   * twice. `ramp` is the half-width of a smooth start and end along the segment (see Splat in
   * fluid3d-core.wgsl). Returns false (and drops it) if the queue is full or an argument is not finite.
   */
  addImpulse(
    from: readonly [number, number, number],
    to: readonly [number, number, number],
    radius: number,
    velocity: readonly [number, number, number],
    ramp?: number,
  ): boolean;
  /** Queues ink: adds `value` (rgb optical density per unit length, a amount) along the segment, same footprint. Negative values remove ink. */
  addDensity(
    from: readonly [number, number, number],
    to: readonly [number, number, number],
    radius: number,
    value: readonly [number, number, number, number],
    ramp?: number,
  ): boolean;

  /**
   * Advances by `delta` seconds of wall time at the fixed step rate: applies queued splats, runs the
   * whole number of steps that fit (at most `maxStepsPerFrame`). Call at most once per submitted
   * command buffer. Returns the steps run.
   */
  update(encoder: GPUCommandEncoder, delta: number): number;
  /** Runs exactly `steps` steps whatever the clock says (benchmarks, tests), applying queued splats first. */
  advance(encoder: GPUCommandEncoder, steps: number): void;
  /**
   * Runs the pressure projection on the current velocity now (divergence, `iterations` sweeps or
   * V-cycles, gradient subtraction), without advecting anything. Defaults to `params.pressureIterations`.
   */
  project(encoder: GPUCommandEncoder, iterations?: number): void;
  /** Writes `fields` (cell-centred velocity) from the current velocity. */
  resolve(encoder: GPUCommandEncoder): void;

  /** Replaces the velocity from a function of position (length units) returning [vx, vy, vz], evaluated at each face's own position. Wall faces are zeroed. Not projected. */
  setVelocity(field: (x: number, y: number, z: number) => readonly [number, number, number]): void;
  /** Replaces the ink from a function of position returning rgba, evaluated at texel centres. */
  setDensity(
    field: (x: number, y: number, z: number) => readonly [number, number, number, number],
  ): void;
  /** Re-creates every texture empty (bumps `generation`) and drops queued splats. */
  reset(): void;
  destroy(): void;
}

// --- Uniforms ----------------------------------------------------------------------------------------

const Fluid3dUniforms = defineUniformStruct('Fluid3dUniforms', {
  grid: 'vec3u',
  force_count: 'u32',
  scalar_size: 'vec3u',
  dye_count: 'u32',
  size: 'vec3f',
  h: 'f32',
  accel: 'vec3f',
  inv_h: 'f32',
  buoy_dir: 'vec3f',
  dt: 'f32',
  obstacle: 'vec4f',
  max_disp: 'f32',
  rk2: 'u32',
  limit: 'u32',
  jacobi_w: 'f32',
  vorticity: 'f32',
  buoyancy: 'f32',
  vel_decay: 'f32',
  dye_decay: 'f32',
  vmax: 'f32',
  dye_max: 'f32',
});

// --- Pipelines, cached per device ----------------------------------------------------------------------

/** What a binding is, for building bind group layouts without repeating the WGSL. */
type BindKind = 'uniform' | 'tex' | 'texu' | 'storage-rgba' | 'storage-r32' | 'buffer' | 'sampler';

interface PassSpec {
  source: 'advect' | 'forces' | 'project';
  entry: string;
  bindings: Readonly<Record<number, BindKind>>;
}

function spec(
  source: PassSpec['source'],
  entry: string,
  bindings: Record<number, BindKind>,
): PassSpec {
  return { source, entry, bindings: { 0: 'uniform', ...bindings } };
}

const PASSES = {
  velSl: spec('advect', 'cs_vel_sl', { 2: 'tex', 4: 'storage-rgba', 6: 'sampler' }),
  velMc: spec('advect', 'cs_vel_mc', { 2: 'tex', 3: 'tex', 4: 'storage-rgba', 6: 'sampler' }),
  dyeSl: spec('advect', 'cs_dye_sl', { 2: 'tex', 4: 'storage-rgba', 5: 'tex', 6: 'sampler' }),
  dyePred: spec('advect', 'cs_dye_pred', { 2: 'tex', 4: 'storage-rgba', 5: 'tex', 6: 'sampler' }),
  dyeMc: spec('advect', 'cs_dye_mc', {
    2: 'tex',
    3: 'tex',
    4: 'storage-rgba',
    5: 'tex',
    6: 'sampler',
  }),
  splatVelocity: spec('forces', 'cs_splat_velocity', { 2: 'tex', 3: 'storage-rgba', 4: 'buffer' }),
  splatDye: spec('forces', 'cs_splat_dye', { 2: 'tex', 3: 'storage-rgba', 4: 'buffer' }),
  center: spec('forces', 'cs_center', { 2: 'tex', 3: 'storage-rgba' }),
  curl: spec('forces', 'cs_curl', { 2: 'tex', 12: 'storage-rgba' }),
  blur: spec('forces', 'cs_blur', { 5: 'tex', 8: 'storage-r32' }),
  conf: spec('forces', 'cs_conf', { 5: 'tex', 9: 'texu', 10: 'storage-rgba' }),
  forces: spec('forces', 'cs_forces', {
    2: 'tex',
    3: 'storage-rgba',
    6: 'tex',
    7: 'sampler',
    11: 'tex',
  }),
  divergence: spec('project', 'cs_divergence', { 2: 'tex', 6: 'storage-r32' }),
  jacobi: spec('project', 'cs_jacobi', { 3: 'texu', 4: 'texu', 5: 'storage-r32' }),
  jacobiCold: spec('project', 'cs_jacobi_cold', { 3: 'texu', 4: 'texu', 5: 'storage-r32' }),
  restrict: spec('project', 'cs_restrict', { 3: 'texu', 4: 'texu', 5: 'storage-r32' }),
  prolong: spec('project', 'cs_prolong', { 4: 'texu', 5: 'storage-r32', 8: 'texu' }),
  project: spec('project', 'cs_project', { 2: 'tex', 4: 'texu', 7: 'storage-rgba' }),
} as const satisfies Record<string, PassSpec>;

type PassName = keyof typeof PASSES;

interface Pipeline {
  pipeline: GPUComputePipeline;
  layout: GPUBindGroupLayout;
  spec: PassSpec;
}

const SOURCES = { advect: advectSource, forces: forcesSource, project: projectSource } as const;

function layoutEntry(binding: number, kind: BindKind): GPUBindGroupLayoutEntry {
  const visibility = GPUShaderStage.COMPUTE;
  const viewDimension = '3d';
  switch (kind) {
    case 'uniform':
      return { binding, visibility, buffer: { type: 'uniform' } };
    case 'tex':
      return { binding, visibility, texture: { sampleType: 'float', viewDimension } };
    case 'texu':
      return { binding, visibility, texture: { sampleType: 'unfilterable-float', viewDimension } };
    case 'storage-rgba':
      return {
        binding,
        visibility,
        storageTexture: { access: 'write-only', format: FLUID3D_FORMAT, viewDimension },
      };
    case 'storage-r32':
      return {
        binding,
        visibility,
        storageTexture: { access: 'write-only', format: FLUID3D_SCALAR_FORMAT, viewDimension },
      };
    case 'buffer':
      return { binding, visibility, buffer: { type: 'read-only-storage' } };
    case 'sampler':
      return { binding, visibility, sampler: { type: 'filtering' } };
  }
}

const pipelineCache = new WeakMap<GPUDevice, Map<string, Pipeline>>();
const moduleCache = new WeakMap<GPUDevice, Map<string, GPUShaderModule>>();

function workgroupKey(workgroup: readonly [number, number, number]): string {
  return workgroup.join('x');
}

function getModule(
  device: GPUDevice,
  source: PassSpec['source'],
  workgroup: readonly [number, number, number],
): GPUShaderModule {
  let modules = moduleCache.get(device);
  if (!modules) moduleCache.set(device, (modules = new Map()));
  const key = `${source}|${workgroupKey(workgroup)}`;
  let module = modules.get(key);
  if (!module) {
    const constants = `const WG_X: u32 = ${workgroup[0]}u;\nconst WG_Y: u32 = ${workgroup[1]}u;\nconst WG_Z: u32 = ${workgroup[2]}u;\n`;
    module = createShader(device, {
      label: `sim/fluid3d-${source}.wgsl`,
      code: SOURCES[source],
      chunks: {
        'fluid3d_uniforms.wgsl': Fluid3dUniforms.wgsl + constants,
        'fluid3d-core.wgsl': coreSource,
      },
    });
    modules.set(key, module);
  }
  return module;
}

function getPipeline(
  device: GPUDevice,
  name: PassName,
  workgroup: readonly [number, number, number],
): Pipeline {
  let pipelines = pipelineCache.get(device);
  if (!pipelines) pipelineCache.set(device, (pipelines = new Map()));
  const key = `${name}|${workgroupKey(workgroup)}`;
  let found = pipelines.get(key);
  if (!found) {
    const passSpec: PassSpec = PASSES[name];
    const layout = device.createBindGroupLayout({
      label: `fluid3d ${passSpec.entry} layout`,
      entries: Object.entries(passSpec.bindings).map(([binding, kind]) =>
        layoutEntry(Number(binding), kind),
      ),
    });
    const pipeline = device.createComputePipeline({
      label: `fluid3d ${passSpec.entry}`,
      layout: device.createPipelineLayout({
        label: `fluid3d ${passSpec.entry} pipeline layout`,
        bindGroupLayouts: [layout],
      }),
      compute: {
        module: getModule(device, passSpec.source, workgroup),
        entryPoint: passSpec.entry,
      },
    });
    found = { pipeline, layout, spec: passSpec };
    pipelines.set(key, found);
  }
  return found;
}

// --- Helpers ---------------------------------------------------------------------------------------------

const clamp = (x: number, lo: number, hi: number) => Math.min(hi, Math.max(lo, x));
const finiteOr = (x: number, fallback: number) => (Number.isFinite(x) ? x : fallback);
const roundTo8 = (x: number) => Math.max(8, Math.round(x / 8) * 8);

type Vec3 = readonly [number, number, number];

/**
 * Cells for a box with these relative extents and `resolution` cells along its longest side: cubic
 * cells, every axis a multiple of 8 (so the multigrid hierarchy halves cleanly), at most 256 per axis.
 * `cellSize` is for a longest side of 1 length unit.
 */
export function computeFluid3dGrid(
  resolution: number,
  size: Vec3 = [1, 1, 1],
): { width: number; height: number; depth: number; cellSize: number } {
  const long = roundTo8(clamp(finiteOr(resolution, 64), 16, 256));
  const extents = size.map((s) => (Number.isFinite(s) && s > 0 ? s : 1));
  const longest = Math.max(...extents);
  const cells = extents.map((s) => clamp(roundTo8((s / longest) * long), 8, 256)) as [
    number,
    number,
    number,
  ];
  return { width: cells[0], height: cells[1], depth: cells[2], cellSize: 1 / long };
}

/** Texels of the scalar grid: `resolution` along the longest side, the others in proportion. */
export function computeScalarGrid(
  resolution: number,
  box: Vec3,
  maxDimension = 256,
): [number, number, number] {
  const long = Math.round(clamp(finiteOr(resolution, 96), 16, maxDimension));
  const longest = Math.max(...box);
  return box.map((s) => clamp(Math.round((s / longest) * long), 8, maxDimension)) as [
    number,
    number,
    number,
  ];
}

/** Bytes of GPU memory the solver's textures take at these sizes (rough: drivers pad). */
export function fluid3dTextureBytes(
  cells: Vec3,
  scalar: Vec3,
  options: { confinement?: boolean; scalarTextures?: number } = {},
): number {
  const volume = (d: Vec3, bytes: number) => d[0] * d[1] * d[2] * bytes;
  const vel: Vec3 = [cells[0] + 1, cells[1] + 1, cells[2] + 1];
  let total = 3 * volume(vel, 8) + (options.scalarTextures ?? 3) * volume(scalar, 8);
  total += 2 * volume(cells, 4) + volume(cells, 4); // pressure x2, divergence
  total += volume(cells, 8); // fields
  if (options.confinement !== false) total += 2 * volume(cells, 8) + volume(cells, 4);
  total += (volume(cells, 4) * 3) / 7; // multigrid levels (2 pressures + rhs, 1/8 + 1/64 + ...)
  return total;
}

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

interface Level {
  dims: [number, number, number];
  p: [Tex, Tex];
  rhs: Tex;
  /** Index of the pressure texture that holds the latest iterate. */
  current: number;
}

interface Resources {
  grid: Fluid3dGrid;
  velocity: Tex[];
  scalar: Tex[];
  levels: Level[];
  divergence: Tex;
  curl: Tex;
  blur: Tex;
  conf: Tex;
  fields: Tex;
  groups: Map<string, GPUBindGroup>;
}

/** Ray against the box [min, max]: the distances (t near, t far) along `dir` (not necessarily unit) where it enters and leaves, with t near clamped to 0, or null if it misses. The CPU twin of volume_ray_box in src/wgsl/volume.wgsl. */
export function rayBox(
  origin: Vec3,
  dir: Vec3,
  min: Vec3,
  max: Vec3,
): { near: number; far: number } | null {
  let near = 0;
  let far = Infinity;
  for (let i = 0; i < 3; i++) {
    const o = origin[i]!;
    const d = dir[i]!;
    if (Math.abs(d) < 1e-9) {
      if (o < min[i]! || o > max[i]!) return null;
      continue;
    }
    const t0 = (min[i]! - o) / d;
    const t1 = (max[i]! - o) / d;
    near = Math.max(near, Math.min(t0, t1));
    far = Math.min(far, Math.max(t0, t1));
  }
  return far > near ? { near, far } : null;
}

export function createFluid3d(options: Fluid3dOptions): Fluid3d {
  const { device, maxStepsPerFrame = 3, profiler = noProfiler, label = 'fluid3d' } = options;
  let resolution = options.resolution ?? 64;
  let scalarResolution = options.scalarResolution ?? 96;
  let boxSize: Vec3 = options.size ?? [1, 1, 1];
  let workgroup: [number, number, number] = [...(options.workgroup ?? DEFAULT_WORKGROUP)];
  const params: Fluid3dParams = {
    ...DEFAULT_FLUID3D_PARAMS,
    ...options.params,
    obstacle: { ...DEFAULT_FLUID3D_PARAMS.obstacle, ...options.params?.obstacle },
  };
  const maxDimension = Math.min(device.limits.maxTextureDimension3D - 1, 256);
  const uniforms = createUniformBuffer(device, Fluid3dUniforms, `${label} params`);
  const sampler = device.createSampler({
    label: `${label} sampler`,
    magFilter: 'linear',
    minFilter: 'linear',
    addressModeU: 'clamp-to-edge',
    addressModeV: 'clamp-to-edge',
    addressModeW: 'clamp-to-edge',
  });
  const momentumBuffer = device.createBuffer({
    label: `${label} momentum splats`,
    size: MAX_SPLATS_3D * SPLAT_FLOATS * 4,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });
  const inkBuffer = device.createBuffer({
    label: `${label} ink splats`,
    size: MAX_SPLATS_3D * SPLAT_FLOATS * 4,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });
  const momentumQueue = new Float32Array(MAX_SPLATS_3D * SPLAT_FLOATS);
  const inkQueue = new Float32Array(MAX_SPLATS_3D * SPLAT_FLOATS);
  let momentumCount = 0;
  let inkCount = 0;

  const usage =
    GPUTextureUsage.TEXTURE_BINDING |
    GPUTextureUsage.STORAGE_BINDING |
    GPUTextureUsage.COPY_SRC |
    GPUTextureUsage.COPY_DST;

  let destroyed = false;
  let generation = 0;
  let lastSteps = 0;
  let stepper: FixedStepper | undefined;
  let stepperSeconds = 0;
  let resources: Resources;
  let velocityCurrent = 0;
  let scalarCurrent = 0;

  function makeTexture(
    name: string,
    size: readonly [number, number, number],
    format: GPUTextureFormat,
  ): Tex {
    const texture = device.createTexture({
      label: `${label} ${name}`,
      dimension: '3d',
      size: [size[0], size[1], size[2]],
      format,
      usage,
    });
    return {
      texture,
      view: texture.createView({ label: `${label} ${name} view`, dimension: '3d' }),
      id: idOf(texture),
    };
  }

  function allocate(
    cells: [number, number, number],
    cellSize: number,
    scalarSize: [number, number, number],
  ): Resources {
    const velocitySize: [number, number, number] = [cells[0] + 1, cells[1] + 1, cells[2] + 1];
    // The multigrid hierarchy: halve while every axis stays a multiple of 8 and the coarse grid keeps 4 cells.
    const levels: Level[] = [];
    const dimsAt: [number, number, number][] = [cells];
    while (
      dimsAt.length <= MAX_COARSENINGS &&
      dimsAt[dimsAt.length - 1]!.every((n) => n % 2 === 0 && n >= 8)
    ) {
      dimsAt.push(dimsAt[dimsAt.length - 1]!.map((n) => n / 2) as [number, number, number]);
    }
    const divergence = makeTexture('divergence', cells, FLUID3D_SCALAR_FORMAT);
    dimsAt.forEach((dims, level) => {
      levels.push({
        dims,
        p: [
          makeTexture(`pressure ${level}a`, dims, FLUID3D_SCALAR_FORMAT),
          makeTexture(`pressure ${level}b`, dims, FLUID3D_SCALAR_FORMAT),
        ],
        rhs: level === 0 ? divergence : makeTexture(`rhs ${level}`, dims, FLUID3D_SCALAR_FORMAT),
        current: 0,
      });
    });
    const grid: Fluid3dGrid = Object.freeze({
      width: cells[0],
      height: cells[1],
      depth: cells[2],
      cellSize,
      size: Object.freeze([cells[0] * cellSize, cells[1] * cellSize, cells[2] * cellSize] as const),
      velocitySize: Object.freeze(velocitySize),
      scalarSize: Object.freeze(scalarSize),
      levels: levels.length,
    });
    return {
      grid,
      velocity: [0, 1, 2].map((i) => makeTexture(`velocity ${i}`, velocitySize, FLUID3D_FORMAT)),
      // The third scalar texture only exists once MacCormack needs it.
      scalar: [0, 1].map((i) => makeTexture(`ink ${i}`, scalarSize, FLUID3D_FORMAT)),
      levels,
      divergence,
      curl: makeTexture('vorticity', cells, FLUID3D_FORMAT),
      blur: makeTexture('vorticity blur', cells, FLUID3D_SCALAR_FORMAT),
      conf: makeTexture('confinement', cells, FLUID3D_FORMAT),
      fields: makeTexture('fields', cells, FLUID3D_FORMAT),
      groups: new Map(),
    };
  }

  function release(r: Resources) {
    const all = new Set<Tex>([
      ...r.velocity,
      ...r.scalar,
      ...r.levels.flatMap((l) => [...l.p, l.rhs]),
      r.divergence,
      r.curl,
      r.blur,
      r.conf,
      r.fields,
    ]);
    for (const t of all) t.texture.destroy();
  }

  function rebuild(force = false): boolean {
    const g = computeFluid3dGrid(resolution, boxSize);
    const cells: [number, number, number] = [g.width, g.height, g.depth];
    const box: Vec3 = [g.width * g.cellSize, g.height * g.cellSize, g.depth * g.cellSize];
    const scalarSize = computeScalarGrid(scalarResolution, box, maxDimension);
    const current = resources?.grid;
    if (
      !force &&
      current &&
      current.width === cells[0] &&
      current.height === cells[1] &&
      current.depth === cells[2] &&
      current.scalarSize.every((n, i) => n === scalarSize[i])
    ) {
      return false;
    }
    if (resources) release(resources);
    resources = allocate(cells, g.cellSize, scalarSize);
    velocityCurrent = 0;
    scalarCurrent = 0;
    generation++;
    stepper = undefined;
    momentumCount = 0;
    inkCount = 0;
    return true;
  }

  rebuild();

  function ensureScalar(count: number) {
    while (resources.scalar.length < count) {
      resources.scalar.push(
        makeTexture(`ink ${resources.scalar.length}`, resources.grid.scalarSize, FLUID3D_FORMAT),
      );
    }
  }

  // --- Bind groups and dispatches ----------------------------------------------------------------------

  type Resource = Tex | GPUBuffer | GPUSampler;
  const isTex = (item: Resource): item is Tex => 'view' in item && 'texture' in item;

  /** The bind group of a pass over these resources (binding 0 is filled in), cached. */
  function groupFor(name: PassName, bound: Readonly<Record<number, Resource>>): GPUBindGroup {
    const { layout, spec: passSpec } = getPipeline(device, name, workgroup);
    const key = [
      name,
      ...Object.keys(passSpec.bindings).map((b) => {
        const binding = Number(b);
        if (binding < 1) return '';
        const item = bound[binding];
        if (!item) throw new Error(`fluid3d pass ${name} is missing binding ${binding}`);
        return `${binding}:${isTex(item) ? item.id : idOf(item)}`;
      }),
    ].join('|');
    let group = resources.groups.get(key);
    if (!group) {
      const entries: GPUBindGroupEntry[] = [{ binding: 0, resource: { buffer: uniforms.buffer } }];
      for (const b of Object.keys(passSpec.bindings)) {
        const binding = Number(b);
        if (binding < 1) continue;
        const item = bound[binding]!;
        entries.push({
          binding,
          resource: isTex(item) ? item.view : 'size' in item ? { buffer: item } : item,
        });
      }
      group = device.createBindGroup({ label: `fluid3d ${passSpec.entry}`, layout, entries });
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
      label: `fluid3d ${name}`,
      timestampWrites: timestamp
        ? (profiler.pass(timestamp) as GPUComputePassTimestampWrites | undefined)
        : undefined,
    });
  }

  function dispatch(
    pass: GPUComputePassEncoder,
    name: PassName,
    bound: Readonly<Record<number, Resource>>,
    size: readonly [number, number, number],
  ) {
    pass.setPipeline(getPipeline(device, name, workgroup).pipeline);
    pass.setBindGroup(0, groupFor(name, bound));
    pass.dispatchWorkgroups(
      Math.ceil(size[0] / workgroup[0]),
      Math.ceil(size[1] / workgroup[1]),
      Math.ceil(size[2] / workgroup[2]),
    );
  }

  /** The velocity pool indices that are not `current`. */
  const others = (current: number): [number, number] => {
    const rest = [0, 1, 2].filter((i) => i !== current);
    return [rest[0]!, rest[1]!];
  };
  const freeScalar = (): number[] =>
    resources.scalar.map((_, i) => i).filter((i) => i !== scalarCurrent);
  const cellsOf = (r: Resources): [number, number, number] => [
    r.grid.width,
    r.grid.height,
    r.grid.depth,
  ];

  // --- Uniforms ---------------------------------------------------------------------------------------------

  const stepRate = () => clamp(finiteOr(params.stepRate, 60), 10, 240);
  const stepSeconds = () => 1 / stepRate();

  /** The splat counts the last `run` wrote into the uniforms; `resolve` and `project` repeat them. */
  let writtenCounts: readonly [number, number] = [0, 0];

  /**
   * Uploads the parameters. `queue.writeBuffer` lands before the whole command buffer, so a second write
   * in the same encoder (a `resolve` after an `update`) replaces the first for every pass in it: the
   * splat counts must therefore be the ones the injection pass needs, not whatever is queued now.
   */
  function writeUniforms(counts: readonly [number, number] = writtenCounts) {
    writtenCounts = counts;
    const { grid } = resources;
    const dt = stepSeconds();
    const maxCfl = clamp(finiteOr(params.maxCfl, 6), 0.5, 32);
    const nonNegative = (x: number) => Math.max(0, finiteOr(x, 0));
    const direction = params.buoyancyDirection;
    const length = Math.hypot(direction[0], direction[1], direction[2]) || 1;
    const obstacle = params.obstacle;
    uniforms.set({
      grid: [grid.width, grid.height, grid.depth],
      force_count: counts[0],
      scalar_size: grid.scalarSize,
      dye_count: counts[1],
      size: grid.size,
      h: grid.cellSize,
      accel: [
        finiteOr(params.gravity[0], 0),
        finiteOr(params.gravity[1], 0),
        finiteOr(params.gravity[2], 0),
      ],
      inv_h: 1 / grid.cellSize,
      buoy_dir: [
        finiteOr(direction[0], 0) / length,
        finiteOr(direction[1], -1) / length,
        finiteOr(direction[2], 0) / length,
      ],
      dt,
      obstacle: [
        finiteOr(obstacle.center[0], 0),
        finiteOr(obstacle.center[1], 0),
        finiteOr(obstacle.center[2], 0),
        nonNegative(obstacle.radius),
      ],
      max_disp: maxCfl,
      rk2: params.rk2 ? 1 : 0,
      limit: params.limiter ? 1 : 0,
      jacobi_w: clamp(finiteOr(params.jacobiWeight, 0.8), 0.3, 1),
      vorticity: nonNegative(params.vorticity),
      buoyancy: finiteOr(params.buoyancy, 0),
      vel_decay: Math.exp(-nonNegative(params.velocityDissipation) * dt),
      dye_decay: Math.exp(-nonNegative(params.scalarDissipation) * dt),
      vmax: (maxCfl * grid.cellSize) / dt,
      dye_max: clamp(finiteOr(params.scalarLimit, SCALAR_LIMIT), 1, 256),
    });
    uniforms.upload();
  }

  // --- Passes -------------------------------------------------------------------------------------------------

  function advectVelocity(pass: GPUComputePassEncoder) {
    const r = resources;
    const src = r.velocity[velocityCurrent]!;
    const [a, b] = others(velocityCurrent).map((i) => r.velocity[i]!) as [Tex, Tex];
    const size = r.grid.velocitySize;
    if (params.velocityScheme === 'maccormack') {
      dispatch(pass, 'velSl', { 2: src, 4: a, 6: sampler }, size);
      dispatch(pass, 'velMc', { 2: src, 3: a, 4: b, 6: sampler }, size);
      velocityCurrent = r.velocity.indexOf(b);
    } else {
      dispatch(pass, 'velSl', { 2: src, 4: a, 6: sampler }, size);
      velocityCurrent = r.velocity.indexOf(a);
    }
  }

  function advectScalar(pass: GPUComputePassEncoder) {
    const r = resources;
    const flow = r.velocity[velocityCurrent]!;
    const size = r.grid.scalarSize;
    if (params.scalarScheme === 'maccormack') {
      ensureScalar(3);
      const src = r.scalar[scalarCurrent]!;
      const [a, b] = freeScalar().map((i) => r.scalar[i]!) as [Tex, Tex];
      dispatch(pass, 'dyePred', { 2: src, 4: a, 5: flow, 6: sampler }, size);
      dispatch(pass, 'dyeMc', { 2: src, 3: a, 4: b, 5: flow, 6: sampler }, size);
      scalarCurrent = r.scalar.indexOf(b);
    } else {
      const src = r.scalar[scalarCurrent]!;
      const [a] = freeScalar().map((i) => r.scalar[i]!) as [Tex];
      dispatch(pass, 'dyeSl', { 2: src, 4: a, 5: flow, 6: sampler }, size);
      scalarCurrent = r.scalar.indexOf(a);
    }
  }

  function applyForces(pass: GPUComputePassEncoder) {
    const r = resources;
    const src = r.velocity[velocityCurrent]!;
    const cells = cellsOf(r);
    if (params.vorticity > 0) {
      dispatch(pass, 'center', { 2: src, 3: r.fields }, cells);
      dispatch(pass, 'curl', { 2: r.fields, 12: r.curl }, cells);
      dispatch(pass, 'blur', { 5: r.curl, 8: r.blur }, cells);
      dispatch(pass, 'conf', { 5: r.curl, 9: r.blur, 10: r.conf }, cells);
    }
    const [a] = others(velocityCurrent).map((i) => r.velocity[i]!) as [Tex, Tex];
    dispatch(
      pass,
      'forces',
      { 2: src, 3: a, 6: r.scalar[scalarCurrent]!, 7: sampler, 11: r.conf },
      r.grid.velocitySize,
    );
    velocityCurrent = r.velocity.indexOf(a);
  }

  /** `count` damped-Jacobi sweeps on one level; the first starts from zero when `cold`. */
  function smooth(pass: GPUComputePassEncoder, level: Level, count: number, cold: boolean) {
    for (let i = 0; i < count; i++) {
      const from = level.p[level.current]!;
      const to = level.p[1 - level.current]!;
      dispatch(
        pass,
        i === 0 && cold ? 'jacobiCold' : 'jacobi',
        { 3: level.rhs, 4: from, 5: to },
        level.dims,
      );
      level.current = 1 - level.current;
    }
  }

  /** One V-cycle from level `index`. `cold` starts the finest level from zero instead of from its latest iterate. */
  function vcycle(pass: GPUComputePassEncoder, index: number, fineCold: boolean) {
    const levels = resources.levels;
    const level = levels[index]!;
    const sweeps = Math.round(clamp(finiteOr(params.smoothing, 2), 1, 8));
    // The correction on a coarse level starts from zero; the fine level starts from the latest pressure.
    const cold = index > 0 || fineCold;
    if (index === levels.length - 1) {
      smooth(pass, level, Math.round(clamp(finiteOr(params.coarseIterations, 12), 1, 64)), true);
      return;
    }
    const coarse = levels[index + 1]!;
    smooth(pass, level, sweeps, cold);
    dispatch(
      pass,
      'restrict',
      { 3: level.rhs, 4: level.p[level.current]!, 5: coarse.rhs },
      coarse.dims,
    );
    vcycle(pass, index + 1, false);
    const from = level.p[level.current]!;
    const to = level.p[1 - level.current]!;
    dispatch(pass, 'prolong', { 4: from, 5: to, 8: coarse.p[coarse.current]! }, level.dims);
    level.current = 1 - level.current;
    smooth(pass, level, sweeps, false);
  }

  function solvePressure(pass: GPUComputePassEncoder, iterations: number) {
    const r = resources;
    const velocity = r.velocity[velocityCurrent]!;
    const fine = r.levels[0]!;
    dispatch(pass, 'divergence', { 2: velocity, 6: r.divergence }, cellsOf(r));
    const count = Math.max(1, Math.round(finiteOr(iterations, 1)));
    const multigrid =
      params.pressureSolver === 'multigrid' && r.levels.length > 1 && params.obstacle.radius <= 0;
    if (multigrid) {
      for (let i = 0; i < count; i++) vcycle(pass, 0, i === 0 && !params.warmStart);
    } else {
      smooth(pass, fine, count, !params.warmStart);
    }
    const [dst] = others(velocityCurrent).map((i) => r.velocity[i]!) as [Tex, Tex];
    dispatch(
      pass,
      'project',
      { 2: velocity, 4: fine.p[fine.current]!, 7: dst },
      r.grid.velocitySize,
    );
    velocityCurrent = r.velocity.indexOf(dst);
  }

  function inject(encoder: GPUCommandEncoder) {
    if (momentumCount === 0 && inkCount === 0) return;
    const r = resources;
    if (momentumCount > 0) {
      device.queue.writeBuffer(momentumBuffer, 0, momentumQueue, 0, momentumCount * SPLAT_FLOATS);
    }
    if (inkCount > 0) {
      device.queue.writeBuffer(inkBuffer, 0, inkQueue, 0, inkCount * SPLAT_FLOATS);
    }
    const pass = begin(encoder, 'inject', 'inject');
    if (momentumCount > 0) {
      const src = r.velocity[velocityCurrent]!;
      const [a] = others(velocityCurrent).map((i) => r.velocity[i]!) as [Tex, Tex];
      dispatch(pass, 'splatVelocity', { 2: src, 3: a, 4: momentumBuffer }, r.grid.velocitySize);
      velocityCurrent = r.velocity.indexOf(a);
    }
    if (inkCount > 0) {
      const src = r.scalar[scalarCurrent]!;
      const dst = r.scalar[freeScalar()[0]!]!;
      dispatch(pass, 'splatDye', { 2: src, 3: dst, 4: inkBuffer }, r.grid.scalarSize);
      scalarCurrent = r.scalar.indexOf(dst);
    }
    pass.end();
    momentumCount = 0;
    inkCount = 0;
  }

  function step(encoder: GPUCommandEncoder, first: boolean, last: boolean) {
    if (!params.scalarOnly) {
      const velocityPass = begin(encoder, 'velocity', first ? 'advect + forces' : undefined);
      advectVelocity(velocityPass);
      applyForces(velocityPass);
      velocityPass.end();

      const pressurePass = begin(encoder, 'pressure', first ? 'pressure' : undefined);
      solvePressure(pressurePass, params.pressureIterations);
      pressurePass.end();
    }
    const scalarPass = begin(
      encoder,
      'scalar',
      first ? 'ink' : last ? 'ink (last step)' : undefined,
    );
    advectScalar(scalarPass);
    scalarPass.end();
  }

  function run(encoder: GPUCommandEncoder, steps: number) {
    if (destroyed) return;
    const count = Math.max(0, Math.floor(finiteOr(steps, 0)));
    lastSteps = count;
    if (count === 0 && momentumCount === 0 && inkCount === 0) return;
    writeUniforms([momentumCount, inkCount]);
    inject(encoder);
    for (let i = 0; i < count; i++) step(encoder, i === 0, i === count - 1 && count > 1);
  }

  // --- Queues -----------------------------------------------------------------------------------------------------

  function enqueue(
    queue: Float32Array,
    count: number,
    from: Vec3,
    to: Vec3,
    radius: number,
    ramp: number,
    value: readonly number[],
  ): boolean {
    if (count >= MAX_SPLATS_3D) return false;
    const numbers = [...from, ...to, radius, ramp, ...value];
    if (!numbers.every(Number.isFinite)) return false;
    const o = count * SPLAT_FLOATS;
    queue[o] = from[0];
    queue[o + 1] = from[1];
    queue[o + 2] = from[2];
    queue[o + 3] = Math.max(radius, 1e-4);
    queue[o + 4] = to[0];
    queue[o + 5] = to[1];
    queue[o + 6] = to[2];
    queue[o + 7] = Math.max(ramp, 0);
    queue[o + 8] = value[0] ?? 0;
    queue[o + 9] = value[1] ?? 0;
    queue[o + 10] = value[2] ?? 0;
    queue[o + 11] = value[3] ?? 0;
    return true;
  }

  // --- Uploads ------------------------------------------------------------------------------------------------------

  function upload(tex: Tex, size: readonly [number, number, number], data: Uint16Array) {
    device.queue.writeTexture(
      { texture: tex.texture },
      data as Uint16Array<ArrayBuffer>,
      { bytesPerRow: size[0] * 8, rowsPerImage: size[1] },
      [size[0], size[1], size[2]],
    );
  }

  const outputOf = (tex: Tex, slot: number): Fluid3dTexture => ({
    texture: tex.texture,
    view: tex.view,
    slot,
  });

  return {
    params,
    get grid() {
      return resources.grid;
    },
    get generation() {
      return generation;
    },
    get density() {
      return outputOf(resources.scalar[scalarCurrent]!, scalarCurrent);
    },
    get velocity() {
      return outputOf(resources.velocity[velocityCurrent]!, velocityCurrent);
    },
    get pressure() {
      const fine = resources.levels[0]!;
      return outputOf(fine.p[fine.current]!, fine.current);
    },
    get divergence() {
      return resources.divergence.texture;
    },
    get fields() {
      return resources.fields.texture;
    },
    get fieldsView() {
      return resources.fields.view;
    },
    get stepSeconds() {
      return stepSeconds();
    },
    get lastSteps() {
      return lastSteps;
    },
    get pendingSplats() {
      return [momentumCount, inkCount] as const;
    },

    setResolution(cells, scalarCells) {
      if (!Number.isFinite(cells)) return false;
      resolution = cells;
      if (scalarCells !== undefined && Number.isFinite(scalarCells)) scalarResolution = scalarCells;
      return rebuild();
    },
    setSize(size) {
      boxSize = size;
      return rebuild();
    },
    setWorkgroup(next) {
      const total = next[0] * next[1] * next[2];
      const limits = device.limits;
      if (
        !next.every((n) => Number.isInteger(n) && n >= 1) ||
        total > limits.maxComputeInvocationsPerWorkgroup ||
        next[0] > limits.maxComputeWorkgroupSizeX ||
        next[1] > limits.maxComputeWorkgroupSizeY ||
        next[2] > limits.maxComputeWorkgroupSizeZ
      ) {
        throw new RangeError(`workgroup ${next.join(' x ')} exceeds the device limits`);
      }
      workgroup = [next[0], next[1], next[2]];
      resources.groups.clear();
    },

    addImpulse(from, to, radius, velocity, ramp = 0) {
      if (
        !enqueue(momentumQueue, momentumCount, from, to, radius, ramp, [
          velocity[0],
          velocity[1],
          velocity[2],
        ])
      ) {
        return false;
      }
      momentumCount++;
      return true;
    },
    addDensity(from, to, radius, value, ramp = 0) {
      if (!enqueue(inkQueue, inkCount, from, to, radius, ramp, value)) return false;
      inkCount++;
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
      solvePressure(pass, iterations ?? params.pressureIterations);
      pass.end();
    },
    resolve(encoder) {
      if (destroyed) return;
      writeUniforms();
      const pass = begin(encoder, 'resolve', 'resolve');
      dispatch(
        pass,
        'center',
        { 2: resources.velocity[velocityCurrent]!, 3: resources.fields },
        cellsOf(resources),
      );
      pass.end();
    },

    setVelocity(field) {
      const { velocitySize, cellSize, width, height, depth } = resources.grid;
      const [vx, vy, vz] = velocitySize;
      const h = cellSize;
      const data = new Uint16Array(vx * vy * vz * 4);
      for (let k = 0; k < vz; k++) {
        for (let j = 0; j < vy; j++) {
          for (let i = 0; i < vx; i++) {
            const o = ((k * vy + j) * vx + i) * 4;
            // The padding texels repeat the last real face of the other two axes.
            const ju = Math.min(j, height - 1);
            const ku = Math.min(k, depth - 1);
            const iv = Math.min(i, width - 1);
            const u = i === 0 || i === width ? 0 : field(i * h, (ju + 0.5) * h, (ku + 0.5) * h)[0];
            const v = j === 0 || j === height ? 0 : field((iv + 0.5) * h, j * h, (ku + 0.5) * h)[1];
            const w = k === 0 || k === depth ? 0 : field((iv + 0.5) * h, (ju + 0.5) * h, k * h)[2];
            data[o] = floatToHalf(finiteOr(u, 0));
            data[o + 1] = floatToHalf(finiteOr(v, 0));
            data[o + 2] = floatToHalf(finiteOr(w, 0));
          }
        }
      }
      upload(resources.velocity[velocityCurrent]!, velocitySize, data);
    },
    setDensity(field) {
      const { scalarSize, size } = resources.grid;
      const [sx, sy, sz] = scalarSize;
      const data = new Uint16Array(sx * sy * sz * 4);
      for (let k = 0; k < sz; k++) {
        for (let j = 0; j < sy; j++) {
          for (let i = 0; i < sx; i++) {
            const value = field(
              ((i + 0.5) / sx) * size[0],
              ((j + 0.5) / sy) * size[1],
              ((k + 0.5) / sz) * size[2],
            );
            const o = ((k * sy + j) * sx + i) * 4;
            for (let c = 0; c < 4; c++) data[o + c] = floatToHalf(finiteOr(value[c] ?? 0, 0));
          }
        }
      }
      upload(resources.scalar[scalarCurrent]!, scalarSize, data);
    },
    reset() {
      rebuild(true);
    },
    destroy() {
      if (destroyed) return;
      destroyed = true;
      release(resources);
      uniforms.destroy();
      momentumBuffer.destroy();
      inkBuffer.destroy();
    },
  };
}

/**
 * Turns a pointer ray into a place in the box: the point at fraction `depth` (0 = where the ray
 * enters, 1 = where it leaves) of its path through the box [0, size], or null when it misses.
 * `origin` and `dir` are in the same length units and coordinates as the box (the effect moves the
 * world into box coordinates first).
 */
export function pointOnRay(
  origin: Vec3,
  dir: Vec3,
  size: Vec3,
  depth = 0.5,
): [number, number, number] | null {
  const span = rayBox(origin, dir, [0, 0, 0], size);
  if (!span) return null;
  const t = span.near + (span.far - span.near) * clamp(depth, 0, 1);
  return [origin[0] + dir[0] * t, origin[1] + dir[1] * t, origin[2] + dir[2] * t];
}
