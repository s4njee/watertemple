import { createFixedStepper, type FixedStepper } from '../loop.ts';
import { createComputePass, type ComputePass } from '../particles/compute.ts';
import { createRandom } from '../particles/emitter.ts';
import { createNeighborGrid, type NeighborGrid } from '../particles/neighbor-grid.ts';
import { createParticleSystem, type ParticleSystem } from '../particles/system.ts';
import type { ParticleSet } from '../particles/types.ts';
import { noProfiler, type PassProfiler } from '../profiler.ts';
import { defineUniformStruct, uniformArray } from '../uniforms.ts';
import clearSource from './mpm-clear.wgsl?raw';
import coreSource from './mpm-core.wgsl?raw';
import g2pSource from './mpm-g2p.wgsl?raw';
import gridSource from './mpm-grid.wgsl?raw';
import densitySource from './mpm-density.wgsl?raw';
import p2g1Source from './mpm-p2g1.wgsl?raw';
import p2g1sSource from './mpm-p2g1s.wgsl?raw';
import p2g2Source from './mpm-p2g2.wgsl?raw';
import p2g2sSource from './mpm-p2g2s.wgsl?raw';

// The MLS-MPM solver (story 7.5): a weakly compressible fluid as particles carried over a background grid, in compute
// shaders. The method is Hu et al., "A Moving Least Squares Material Point Method with Displacement Discontinuity and
// Two-Way Rigid Body Coupling" (2018), specialised to a fluid; the passes are in mpm-*.wgsl and each starts with its
// own explanation. There is NO neighbour search: the grid does the neighbouring.
//
//   per step:   clear -> P2G 1 (mass, momentum + APIC affine) -> P2G 2 (density from the grid, pressure and viscous
//               stress, scattered back) -> grid update (v = q / m, gravity, walls, colliders) -> G2P (new v and C,
//               advect, clamp)
//
// STATE. Particles live in a `ParticleSystem` (position + the foam scalar, velocity, the 3x3 affine matrix C as three
// vec4 extras 'affine0..2' whose w components carry the density ratio, and 'aux', one f32 for the colour views).
// The grid is two buffers of 16 bytes per node: the fixed-point sums (mass and momentum, i32 x 4, atomics) and the
// decoded velocity (vec4f). Nodes sit at integer multiples of the cell size h; the domain is
// [0, nx h] x [0, ny h] x [0, nz h], y UP.
//
// UNITS. Metres and seconds. Every particle has mass 1 (mpm-core.wgsl explains the fixed point), so densities appear
// only as ratios r = rho / rho_0 and the rest volume of a particle is V_0 = h^3 / particlesPerCell. Pressure is
// written per rest density, p / rho_0 = (c^2 / gamma)(r^gamma - 1): c is the sound speed, the only stiffness knob.
//
// STABILITY. Explicit stress with a sound speed c needs the time step
//
//     dt <= courant * h / (c + v_max)
//
// derivation. Information moves at most |v| + c per unit time (the fluid's own motion plus a pressure wave), and a
// step of the explicit scheme can only be right if no signal crosses more than a fraction of one cell, because the
// grid couples a particle to nodes within 1.5 h and the discretisation does not see anything beyond them. `courant`
// is that fraction. v_max is the speed limit this solver enforces on grid nodes and particles (`maxSpeed`), so the
// bound holds for every state the solver can reach. Because |v| <= v_max, a particle also moves at most
// courant * h * v_max / (c + v_max) < courant * h per step: it never crosses more than half a cell, which is what the
// quadratic stencil assumes. The viscous term is a diffusion and needs nu dt / h^2 below a constant; nu is clamped to
// viscosityLimit * h^2 / dt. `effective` reports what the clamps decided. Measured limits are in
// src/effects/mpm-3d/notes.md.
//
// FIXED TIMESTEP. `update` accumulates real time and takes whole steps of `effective.dt` (createFixedStepper), at most
// `maxSteps` per call; beyond that the backlog is dropped (the fluid runs slower than real time rather than
// spiralling). A requested `dt` (params.dt, 0 = automatic) is honoured as long as the stiffness allows; otherwise the
// stiffness is clamped to the stable value, not the step.
//
// WALLS AND COLLIDERS. The container is an axis-aligned box that MOVES: `setBounds` sets where the walls should be,
// and over the next `update` they travel there at constant speed, no faster than `wallCourant * h` per step
// (so a particle is never overtaken), with their velocity fed to the grid so they push the fluid. Colliders (sphere,
// box, plane) have a position and a velocity that the solver integrates over the steps of an update.
//
// ONE SUBMIT. Per-step uniforms (the walls and colliders differ from step to step) are written to a ring of
// MPM_RING_STEPS slots with `queue.writeBuffer` and copied into the one uniform buffer between the passes, so any
// number of `update`/`step` calls can share a submit as long as they together use fewer than MPM_RING_STEPS steps.

/** Mass and momentum are summed as i32 with this many units per 1.0 (see mpm-core.wgsl). */
export const MPM_FIXED_SCALE = 1048576;
/** The largest magnitude a node's mass or momentum can hold in the fixed-point grid. */
export const MPM_FIXED_RANGE = 2147483520 / MPM_FIXED_SCALE;
export const MPM_MAX_COLLIDERS = 8;
/** Steps whose uniforms can be pending in one submit. */
export const MPM_RING_STEPS = 2048;
/** Particles within this many cells of the domain edge would read outside the grid; bounds are kept inside it. */
export const MPM_EDGE_CELLS = 2;

export const MpmUniform = defineUniformStruct('MpmParams', {
  dims: 'vec3u',
  count: 'u32',
  h: 'f32',
  inv_h: 'f32',
  dt: 'f32',
  rest_ppc: 'f32',
  gravity: 'vec3f',
  sound_speed: 'f32',
  gamma: 'f32',
  viscosity: 'f32',
  tension: 'f32',
  max_speed: 'f32',
  bounds_min: 'vec3f',
  friction: 'f32',
  bounds_max: 'vec3f',
  r_max: 'f32',
  wall_vmin: 'vec3f',
  collider_count: 'u32',
  wall_vmax: 'vec3f',
  scalar_mode: 'u32',
  foam: 'vec4f',
  foam_decay: 'f32',
  wall_comp: 'u32',
  damping_factor: 'f32',
  // Three vec4f per collider: (centre | point, kind), (radius | half extent | normal, friction), (velocity, 0).
  colliders: uniformArray('vec4f', MPM_MAX_COLLIDERS * 3),
});

export type Vec3 = readonly [number, number, number];

export interface MpmBounds {
  min: Vec3;
  max: Vec3;
}

export type MpmColliderShape = 'sphere' | 'box' | 'plane';

export interface MpmCollider {
  shape: MpmColliderShape;
  /** Sphere and box: the centre. Plane: a point on it. */
  center: Vec3;
  /** Sphere: radius, in metres. */
  radius?: number;
  /** Box: half the size along each axis (axis-aligned). */
  halfExtent?: Vec3;
  /** Plane: unit normal pointing out of the solid, so the solid is where dot(x - center, normal) < 0. */
  normal?: Vec3;
  /** m/s. The solver moves the collider with it over the steps of an update. Default 0. */
  velocity?: Vec3;
  /** 0 slip (default), 1 stick. */
  friction?: number;
}

/** A lattice of particles. `velocity` is m/s. */
export type MpmBlock =
  | { shape: 'box'; min: Vec3; max: Vec3; velocity?: Vec3; jitter?: number }
  | { shape: 'sphere'; center: Vec3; radius: number; velocity?: Vec3; jitter?: number };

/** A stream: layers of particles leave a disc of `radius` metres at `speed` along `direction`, `active` or not. */
export interface MpmPour {
  position: Vec3;
  direction: Vec3;
  speed: number;
  radius: number;
  active: boolean;
}

/** Everything `reset` sets up. Fields left out keep their value, except the particles, which are always replaced. */
export interface MpmScene {
  bounds?: Partial<MpmBounds>;
  gravity?: Vec3;
  blocks?: readonly MpmBlock[];
  colliders?: readonly MpmCollider[];
  pour?: MpmPour | null;
}

/** Live parameters: change any of them any time. */
export interface MpmParams {
  /** m/s^2 along +x, +y, +z. The live gravity vector (tilt a container by tilting this). Default [0, -9.8, 0]. */
  gravity: Vec3;
  /** c, m/s: the sound speed of the equation of state. Stiffer is more incompressible and needs smaller steps. Default 12. */
  soundSpeed: number;
  /** Exponent of the Tait equation, 1 to 10. Default 7 (water). */
  gamma: number;
  /** Kinematic viscosity nu, m^2/s. Water is 1e-6; the grid adds numerical viscosity of its own. Default 0. */
  viscosity: number;
  /** The most negative pressure allowed, as a fraction of c^2 / gamma. 0 (default) means no tension. */
  tension: number;
  /** Ceiling on the density ratio the pressure sees (impulse and fixed-point guard). Default 1.8. */
  densityLimit: number;
  /** Speed limit of grid nodes and particles, m/s. Default 10. */
  maxSpeed: number;
  /** Step length in seconds; 0 (default) takes the largest stable step for the stiffness. */
  dt: number;
  /** Fraction of a cell a signal may cross in one step. Default 0.5. */
  courant: number;
  /** Viscous stability: nu is clamped to this times h^2 / dt. Default 0.1. */
  viscosityLimit: number;
  /** Largest step count of one `update`. Default 64. */
  maxSteps: number;
  /** Walls move at most this fraction of a cell per step. Default 0.4. */
  wallCourant: number;
  /** 0 free slip (default) to 1 no slip, at the container walls. */
  wallFriction: number;
  /** Velocity damping on the grid, 1/s: v *= exp(-damping dt). Calms the acoustic noise of an impulsive start. Default 0. */
  damping: number;
  /** Correct the density estimate near the container walls (see mpm-p2g2.wgsl). Default true. */
  wallCompensation: boolean;
  /** Particles per cell at rest density; set it before `reset` (blocks are laid out with it). Default 8. */
  particlesPerCell: number;
  /** Foam, the 0..1 scalar in positions.w. Density ratios below `sprayLow` count as spray, above `sprayHigh` as water. */
  foam: {
    sprayLow: number;
    sprayHigh: number;
    speedLow: number;
    speedHigh: number;
    decayPerSecond: number;
  };
  /** What `aux` (the colour views' scalar) holds. */
  scalar: 'foam' | 'speed' | 'density';
  /**
   * Bin the particles by stencil cell every step (the framework's counting sort, which also reorders the particle
   * data) and let P2G add per cell instead of per particle: about 8 times fewer atomics. See mpm-p2g1s.wgsl.
   */
  sorted: boolean;
  /**
   * Sort every this many steps (1: every step). Between sorts a particle that has crossed into another stencil cell is
   * added by the slow path (mpm-p2g1s.wgsl), so the result is the same grid and only the cost changes: cheaper
   * sorts, dearer strays. A change of the particle count forces a sort.
   */
  sortEvery: number;
  /** Steps at the start of each update whose passes get timestamps (each costs 5 to 11 of the profiler's slots). Default 1. */
  profileSteps: number;
}

export const DEFAULT_MPM_PARAMS: Readonly<MpmParams> = {
  gravity: [0, -9.8, 0],
  soundSpeed: 12,
  gamma: 7,
  viscosity: 0,
  tension: 0,
  densityLimit: 1.8,
  maxSpeed: 10,
  dt: 0,
  courant: 0.5,
  viscosityLimit: 0.1,
  maxSteps: 64,
  wallCourant: 0.4,
  wallFriction: 0,
  damping: 0,
  wallCompensation: true,
  particlesPerCell: 8,
  foam: { sprayLow: 0.2, sprayHigh: 0.45, speedLow: 3, speedHigh: 7, decayPerSecond: 1.5 },
  scalar: 'foam',
  sorted: true,
  sortEvery: 1,
  profileSteps: 1,
};

/** The passes of a step, in order. `sort` does nothing unless `params.sorted` is on. */
export const MPM_PASSES = ['clear', 'sort', 'p2g1', 'p2g2', 'grid', 'g2p'] as const;
export type MpmPassName = (typeof MPM_PASSES)[number];

/** The shaders behind the passes (p2g1 and p2g2 have a sorted variant). */
export type MpmShaderName =
  'clear' | 'p2g1' | 'p2g1s' | 'p2g2' | 'density' | 'p2g2s' | 'grid' | 'g2p';

/**
 * Storage buffers each shader binds, against the default limit of 8 per shader stage (the uniform is extra).
 * tests/mpm.test.ts counts the `var<storage` declarations of the source files and checks this table.
 */
export const MPM_STORAGE_BINDINGS: Readonly<Record<MpmShaderName, number>> = {
  clear: 1,
  p2g1: 6,
  p2g1s: 7,
  p2g2: 5,
  density: 3,
  p2g2s: 6,
  grid: 2,
  g2p: 7,
};

const PASS_LABELS: Record<MpmPassName, string> = {
  clear: 'mpm clear',
  sort: 'mpm sort',
  p2g1: 'mpm p2g 1',
  p2g2: 'mpm p2g 2',
  grid: 'mpm grid',
  g2p: 'mpm g2p',
};

export interface MpmOptions {
  device: GPUDevice;
  /** Cells along each axis: a number for a cube, or [nx, ny, nz]. Each at least 8. */
  grid: number | readonly [number, number, number];
  /** Edge of a cell in metres. Default 1 / the longest side, a unit domain. */
  cellSize?: number;
  /** Most particles. Grow with `setCapacity`. */
  capacity: number;
  /** Container; default the whole domain less MPM_EDGE_CELLS cells at every edge. */
  bounds?: Partial<MpmBounds>;
  /** The fluid: sound speed (stiffness), exponent, viscosity, tension, particles per cell. Initial values of `params`. */
  material?: Partial<
    Pick<MpmParams, 'soundSpeed' | 'gamma' | 'viscosity' | 'tension' | 'particlesPerCell'>
  >;
  /** Initial live parameters (they win over `material`). */
  params?: Partial<MpmParams>;
  /** Drawn radius as a multiple of the particle spacing. Default 0.7. */
  radiusScale?: number;
  /** Workgroup size of the particle passes: 64, 128 or 256. Default 64. */
  workgroupSize?: 64 | 128 | 256;
  /** Workgroup size of the grid passes. Default 64. */
  gridWorkgroupSize?: 64 | 128 | 256;
  profiler?: PassProfiler;
  label?: string;
}

/** What the clamps made of the requested parameters. */
export interface MpmEffective {
  /** The step actually taken, seconds. */
  dt: number;
  /** The sound speed actually used, m/s (the requested one unless the step was too long for it). */
  soundSpeed: number;
  /** The kinematic viscosity actually used. */
  viscosity: number;
  stepsPerSecond: number;
  /** True when a requested stiffness or viscosity had to be reduced. */
  clamped: boolean;
  /** The limit the CFL rule gives for this stiffness. */
  stableDt: number;
}

export interface MpmUpdateResult {
  /** Steps encoded. */
  steps: number;
  /** The accumulator held more steps than `maxSteps`: the surplus was dropped (slow motion). */
  behind: boolean;
}

export interface MpmStepOptions {
  /** Steps to encode. Default 1. */
  count?: number;
  /** Passes to run in each step, in order. Default all of them. For measurement. */
  passes?: readonly MpmPassName[];
  /** Give the first step's passes timestamps. Default false. */
  timed?: boolean;
}

export interface MpmGridData {
  /** Mass per node in particle units (nx * ny * nz). */
  mass: Float32Array;
  /** Momentum per node, 3 floats each, in particle masses times m/s. */
  momentum: Float32Array;
}

export interface MpmSolver {
  readonly device: GPUDevice;
  /** The particle store: positions (w = foam), velocities, extras. */
  readonly system: ParticleSystem;
  /** What a renderer reads. `restVolume` is the volume of liquid one particle stands for. */
  readonly particles: ParticleSet;
  readonly params: MpmParams;
  readonly effective: MpmEffective;
  readonly dims: readonly [number, number, number];
  readonly cellSize: number;
  /** Particles per cell at rest, the spacing and V_0 follow from it. */
  readonly spacing: number;
  /** World size of the domain. */
  readonly size: Vec3;
  /** The container the solver is using now. */
  readonly bounds: Readonly<MpmBounds>;
  readonly capacity: number;
  readonly count: number;
  /** Bumps when the particle buffers are re-created. */
  readonly generation: number;
  /** Seconds of fluid time simulated since the last reset. */
  readonly time: number;
  readonly stats: { steps: number; updates: number };
  /** Bytes of the grid buffers. */
  readonly gridBytes: number;
  /** Advances by `delta` seconds of real time: whole fixed steps, encoded into `encoder`. */
  update(encoder: GPUCommandEncoder, delta: number): MpmUpdateResult;
  /** Encodes steps regardless of the accumulator (tests and measurement). */
  step(encoder: GPUCommandEncoder, options?: MpmStepOptions): void;
  /** Where the walls should be by the end of the next update; they travel there at a limited speed. */
  setBounds(bounds: Partial<MpmBounds>): void;
  /** Puts the walls there at once, without pushing the fluid (use before particles exist). */
  jumpBounds(bounds: Partial<MpmBounds>): void;
  setColliders(colliders: readonly MpmCollider[]): void;
  /** Adds a lattice of particles; returns how many fitted. */
  addBlock(block: MpmBlock): number;
  /** Adds particles from vec4 rows (xyz, w ignored) and optional velocity rows; returns how many fitted. */
  addParticles(positions: Float32Array, velocities?: Float32Array): number;
  /** Starts, moves or stops the stream. */
  setPour(pour: MpmPour | null): void;
  /** Replaces the particles and applies the scene. With no scene: empties the box. */
  reset(scene?: MpmScene): void;
  setCapacity(capacity: number): void;
  /** The fixed-point grid decoded, after the passes that were encoded have run. Waits for the GPU. */
  readGrid(): Promise<MpmGridData>;
  /** The velocity grid of the last grid pass (vec4: xyz, mass). Waits for the GPU. */
  readGridVelocity(): Promise<Float32Array>;
  destroy(): void;
}

const MAX_PARTICLES_ANY = 1 << 22;

const clamp = (x: number, lo: number, hi: number) => Math.min(hi, Math.max(lo, x));

/** The step and stiffness that satisfy the stability rule; pure so tests and the effect can ask without a solver. */
export function mpmStability(
  cellSize: number,
  params: Pick<
    MpmParams,
    'soundSpeed' | 'maxSpeed' | 'dt' | 'courant' | 'viscosity' | 'viscosityLimit'
  >,
): MpmEffective {
  const { maxSpeed, courant } = params;
  const h = cellSize;
  const floorSpeed = 1;
  const requested = Math.max(params.soundSpeed, floorSpeed);
  const stableDt = (courant * h) / (requested + maxSpeed);
  let dt = params.dt > 0 ? params.dt : stableDt;
  let soundSpeed = requested;
  let clamped = params.soundSpeed < floorSpeed;
  if (dt > stableDt) {
    const limit = (courant * h) / dt - maxSpeed;
    if (limit >= floorSpeed) {
      soundSpeed = Math.min(requested, limit);
    } else {
      soundSpeed = floorSpeed;
      dt = (courant * h) / (floorSpeed + maxSpeed);
    }
    clamped = true;
  }
  const viscosityCap = (params.viscosityLimit * h * h) / dt;
  const viscosity = clamp(params.viscosity, 0, viscosityCap);
  if (viscosity < params.viscosity) clamped = true;
  return {
    dt,
    soundSpeed,
    viscosity,
    stepsPerSecond: 1 / dt,
    clamped,
    stableDt: (courant * h) / (soundSpeed + maxSpeed),
  };
}

// --- Lattices --------------------------------------------------------------------------------------------------------

/**
 * Positions of a jittered lattice of the given spacing filling a block, as vec4 rows (w = 0), centred in the
 * block so no row sits on its faces. Pure and seeded: the same block gives the same particles.
 */
export function blockLattice(block: MpmBlock, spacing: number, seed = 0x5eed): Float32Array {
  const random = createRandom(seed);
  const jitter = (block.jitter ?? 0.15) * spacing;
  const lo: number[] = [];
  const hi: number[] = [];
  if (block.shape === 'box') {
    for (let a = 0; a < 3; a++) {
      lo.push(block.min[a]!);
      hi.push(block.max[a]!);
    }
  } else {
    for (let a = 0; a < 3; a++) {
      lo.push(block.center[a]! - block.radius);
      hi.push(block.center[a]! + block.radius);
    }
  }
  const counts = [0, 1, 2].map((a) => Math.max(0, Math.floor((hi[a]! - lo[a]!) / spacing + 1e-6)));
  const rows: number[] = [];
  for (let k = 0; k < counts[2]!; k++) {
    for (let j = 0; j < counts[1]!; j++) {
      for (let i = 0; i < counts[0]!; i++) {
        const offset = [i, j, k].map(
          (n, a) =>
            lo[a]! +
            (hi[a]! - lo[a]! - counts[a]! * spacing) / 2 +
            (n + 0.5) * spacing +
            (random() * 2 - 1) * jitter,
        );
        if (block.shape === 'sphere') {
          const d = Math.hypot(
            offset[0]! - block.center[0],
            offset[1]! - block.center[1],
            offset[2]! - block.center[2],
          );
          if (d > block.radius) continue;
        }
        rows.push(offset[0]!, offset[1]!, offset[2]!, 0);
      }
    }
  }
  return new Float32Array(rows);
}

/** Lattice points of a disc of `radius` perpendicular to `direction`, centred on the origin, as xyz triples. */
function discLattice(radius: number, spacing: number, direction: Vec3): number[] {
  const [dx, dy, dz] = direction;
  const up: Vec3 = Math.abs(dy) > 0.99 ? [1, 0, 0] : [0, 1, 0];
  let tx = dy * up[2] - dz * up[1];
  let ty = dz * up[0] - dx * up[2];
  let tz = dx * up[1] - dy * up[0];
  const tl = Math.hypot(tx, ty, tz) || 1;
  tx /= tl;
  ty /= tl;
  tz /= tl;
  const bx = dy * tz - dz * ty;
  const by = dz * tx - dx * tz;
  const bz = dx * ty - dy * tx;
  const points: number[] = [];
  const n = Math.ceil(radius / spacing);
  for (let a = -n; a <= n; a++) {
    for (let b = -n; b <= n; b++) {
      const u = (a + 0.5 * (b & 1)) * spacing;
      const v = b * spacing * 0.8660254;
      if (Math.hypot(u, v) > radius) continue;
      points.push(tx * u + bx * v, ty * u + by * v, tz * u + bz * v);
    }
  }
  return points;
}

// --- The solver ------------------------------------------------------------------------------------------------------

type UniformKind = 'f32' | 'i32' | 'u32';

function kindOf(type: unknown): UniformKind {
  if (typeof type !== 'string') return 'f32';
  if (type.endsWith('u') || type === 'u32') return 'u32';
  if (type.endsWith('i') || type === 'i32') return 'i32';
  return 'f32';
}

export function createMpm(options: MpmOptions): MpmSolver {
  const { device, profiler = noProfiler, label = 'mpm' } = options;
  const dims: [number, number, number] =
    typeof options.grid === 'number'
      ? [options.grid, options.grid, options.grid]
      : [options.grid[0], options.grid[1], options.grid[2]];
  for (const n of dims) {
    if (!Number.isInteger(n) || n < 8) {
      throw new RangeError(
        `${label}: grid sizes must be integers of at least 8, got ${dims.join('x')}`,
      );
    }
  }
  const cells = dims[0] * dims[1] * dims[2];
  const gridBytes = cells * 16;
  const limit = Math.min(device.limits.maxStorageBufferBindingSize, device.limits.maxBufferSize);
  if (gridBytes > limit) {
    throw new RangeError(
      `${label}: a ${dims.join('x')} grid needs ${gridBytes} bytes per buffer (16 per node), over the device limit of ${limit}`,
    );
  }
  if (
    device.limits.maxStorageBuffersPerShaderStage < Math.max(...Object.values(MPM_STORAGE_BINDINGS))
  ) {
    throw new RangeError(
      `${label}: the device allows ${device.limits.maxStorageBuffersPerShaderStage} storage buffers per stage, the solver needs ${Math.max(...Object.values(MPM_STORAGE_BINDINGS))}`,
    );
  }
  const h = options.cellSize ?? 1 / Math.max(...dims);
  if (!(h > 0)) throw new RangeError(`${label}: cellSize must be positive, got ${h}`);
  const size: Vec3 = [dims[0] * h, dims[1] * h, dims[2] * h];
  const particleWorkgroup = options.workgroupSize ?? 64;
  const gridWorkgroup = options.gridWorkgroupSize ?? 64;
  const particleLimit = 65535 * particleWorkgroup;

  const params: MpmParams = {
    ...DEFAULT_MPM_PARAMS,
    ...options.material,
    ...options.params,
    foam: { ...DEFAULT_MPM_PARAMS.foam, ...options.params?.foam },
  };
  const radiusScale = options.radiusScale ?? 0.7;

  const spacingNow = () => h / Math.cbrt(Math.max(params.particlesPerCell, 0.5));
  const restVolume = () => (h * h * h) / Math.max(params.particlesPerCell, 0.5);

  // --- Particles -------------------------------------------------------------------------------------------------
  const capacityChecked = Math.floor(options.capacity);
  if (capacityChecked > particleLimit || capacityChecked > MAX_PARTICLES_ANY) {
    throw new RangeError(
      `${label}: ${capacityChecked} particles exceed what one dispatch of workgroup size ${particleWorkgroup} covers (${particleLimit})`,
    );
  }
  const system = createParticleSystem({
    device,
    capacity: capacityChecked,
    radius: radiusScale * spacingNow(),
    extras: { affine0: 4, affine1: 4, affine2: 4, aux: 1 },
    profiler,
    label: `${label} particles`,
  });
  const particles: ParticleSet = {
    get positions() {
      return system.particles.positions;
    },
    get velocities() {
      return system.particles.velocities;
    },
    get count() {
      return system.count;
    },
    get capacity() {
      return system.capacity;
    },
    get radius() {
      return system.radius;
    },
    get restVolume() {
      return restVolume();
    },
    get generation() {
      return system.generation;
    },
  };

  // --- Grid ------------------------------------------------------------------------------------------------------
  const gridBuffer = device.createBuffer({
    label: `${label} grid (fixed point)`,
    size: gridBytes,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
  });
  const gridVelocity = device.createBuffer({
    label: `${label} grid velocity`,
    size: gridBytes,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
  });

  // --- Per-step uniforms -----------------------------------------------------------------------------------------
  const stride = Math.ceil(MpmUniform.size / 256) * 256;
  const ringBytes = stride * MPM_RING_STEPS;
  const ringData = new ArrayBuffer(ringBytes);
  const ringViews = {
    f32: new Float32Array(ringData),
    i32: new Int32Array(ringData),
    u32: new Uint32Array(ringData),
  };
  const ringBuffer = device.createBuffer({
    label: `${label} step ring`,
    size: ringBytes,
    usage: GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
  });
  const uniformBuffer = device.createBuffer({
    label: `${label} params`,
    size: stride,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  let ringCursor = 0;
  const fieldByName = new Map(MpmUniform.layout.map((field) => [field.name, field]));

  function put(slot: number, name: string, value: number | ArrayLike<number>): void {
    const field = fieldByName.get(name);
    if (!field) throw new TypeError(`${label}: no uniform field "${name}"`);
    const kind = kindOf(field.type);
    const view = ringViews[kind];
    const base = (slot * stride + field.offset) >> 2;
    if (typeof value === 'number') {
      view[base] = value;
    } else {
      for (let i = 0; i < value.length; i++) view[base + i] = value[i]!;
    }
  }

  // --- State -----------------------------------------------------------------------------------------------------
  const margin = MPM_EDGE_CELLS * h;
  const edgeMin: Vec3 = [margin, margin, margin];
  const edgeMax: Vec3 = [size[0] - margin, size[1] - margin, size[2] - margin];
  const clampBounds = (min: Vec3, max: Vec3): MpmBounds => {
    const lo: [number, number, number] = [0, 0, 0];
    const hi: [number, number, number] = [0, 0, 0];
    for (let a = 0; a < 3; a++) {
      lo[a] = clamp(min[a]!, edgeMin[a]!, edgeMax[a]! - 4 * h);
      hi[a] = clamp(max[a]!, lo[a]! + 4 * h, edgeMax[a]!);
    }
    return { min: lo, max: hi };
  };
  let current: MpmBounds = clampBounds(
    options.bounds?.min ?? edgeMin,
    options.bounds?.max ?? edgeMax,
  );
  let target: MpmBounds = { min: [...current.min], max: [...current.max] };
  let colliders: MpmCollider[] = [];
  let pour: MpmPour | undefined;
  let pourClock = 0;
  let simTime = 0;
  const stats = { steps: 0, updates: 0 };
  let stepper: FixedStepper | undefined;
  let stepperKey = '';
  let effective = mpmStability(h, params);
  let destroyed = false;
  const random = createRandom(0x4d504d);

  // --- Passes ----------------------------------------------------------------------------------------------------
  // Only the first `profileSteps` steps of an update are timed (the profiler has 32 slots a frame): the passes are
  // made with this gate in place of the profiler and `timing.on` is switched per step.
  const timing = { on: false };
  const gated: PassProfiler = {
    pass: (name) => (timing.on ? profiler.pass(name) : undefined),
  };
  const handles = new Map<MpmShaderName, ComputePass>();
  const sources: Record<
    MpmShaderName,
    { code: string; file: string; workgroup: number; label: string }
  > = {
    clear: {
      code: clearSource,
      file: 'sim/mpm-clear.wgsl',
      workgroup: gridWorkgroup,
      label: PASS_LABELS.clear,
    },
    p2g1: {
      code: p2g1Source,
      file: 'sim/mpm-p2g1.wgsl',
      workgroup: particleWorkgroup,
      label: PASS_LABELS.p2g1,
    },
    p2g1s: {
      code: p2g1sSource,
      file: 'sim/mpm-p2g1s.wgsl',
      workgroup: gridWorkgroup,
      label: PASS_LABELS.p2g1,
    },
    p2g2: {
      code: p2g2Source,
      file: 'sim/mpm-p2g2.wgsl',
      workgroup: particleWorkgroup,
      label: PASS_LABELS.p2g2,
    },
    density: {
      code: densitySource,
      file: 'sim/mpm-density.wgsl',
      workgroup: particleWorkgroup,
      label: PASS_LABELS.p2g2,
    },
    p2g2s: {
      code: p2g2sSource,
      file: 'sim/mpm-p2g2s.wgsl',
      workgroup: gridWorkgroup,
      label: PASS_LABELS.p2g2,
    },
    grid: {
      code: gridSource,
      file: 'sim/mpm-grid.wgsl',
      workgroup: gridWorkgroup,
      label: PASS_LABELS.grid,
    },
    g2p: {
      code: g2pSource,
      file: 'sim/mpm-g2p.wgsl',
      workgroup: particleWorkgroup,
      label: PASS_LABELS.g2p,
    },
  };
  const chunks = { 'mpm-params.wgsl': MpmUniform.wgsl, 'mpm-core.wgsl': coreSource };

  function pass(name: MpmShaderName): ComputePass {
    let found = handles.get(name);
    if (!found) {
      const source = sources[name];
      found = createComputePass(device, {
        label: source.file,
        code: source.code
          .replaceAll('@workgroup_size(64)', `@workgroup_size(${source.workgroup})`)
          .replaceAll('const WG = 64u;', `const WG = ${source.workgroup}u;`),
        chunks,
        workgroupSize: source.workgroup,
        profiler: gated,
        timestamp: source.label,
      });
      handles.set(name, found);
    }
    return found;
  }

  // The sort: the framework's counting sort, binning by the stencil cell floor(x / h - 0.5) of each particle.
  let sorter: NeighborGrid | undefined;
  let sortAge = 0;
  let sortedCount = -1;
  let sortedGeneration = -1;
  function sortParticles(encoder: GPUCommandEncoder): void {
    sorter ??= createNeighborGrid({
      device,
      mode: 'reorder',
      capacity: system.capacity,
      profiler: gated,
      label: 'mpm sort',
    });
    sorter.configure({
      min: [0.5 * h, 0.5 * h, 0.5 * h],
      max: [(dims[0] - 1.5) * h, (dims[1] - 1.5) * h, (dims[2] - 1.5) * h],
      radius: h,
      cellSize: h,
    });
    sorter.build(encoder, system);
  }

  function resources(name: MpmShaderName): GPUBuffer[] {
    const pos = system.attribute('position');
    const vel = system.attribute('velocity');
    const a0 = system.attribute('affine0');
    const a1 = system.attribute('affine1');
    const a2 = system.attribute('affine2');
    switch (name) {
      case 'clear':
        return [uniformBuffer, gridBuffer];
      case 'p2g1':
        return [uniformBuffer, gridBuffer, pos, vel, a0, a1, a2];
      case 'p2g1s':
        return [uniformBuffer, gridBuffer, pos, vel, a0, a1, a2, sorter!.buffers.cellStart];
      case 'p2g2':
        return [uniformBuffer, gridBuffer, pos, a0, a1, a2];
      case 'density':
        return [uniformBuffer, gridBuffer, pos, a0];
      case 'p2g2s':
        return [uniformBuffer, gridBuffer, pos, a0, a1, a2, sorter!.buffers.cellStart];
      case 'grid':
        return [uniformBuffer, gridBuffer, gridVelocity];
      case 'g2p':
        return [uniformBuffer, gridVelocity, pos, vel, a0, a1, a2, system.attribute('aux')];
    }
  }

  function dispatch(encoder: GPUCommandEncoder, name: MpmPassName): void {
    if (name === 'sort') {
      if (!params.sorted || system.count === 0) return;
      const due =
        sorter === undefined ||
        sortAge >= Math.max(1, Math.round(params.sortEvery)) ||
        system.count !== sortedCount ||
        system.generation !== sortedGeneration;
      if (due) {
        sortParticles(encoder);
        sortAge = 0;
        sortedCount = system.count;
        sortedGeneration = system.generation;
      }
      sortAge++;
      return;
    }
    const sorted = params.sorted && sorter !== undefined;
    if (sorted && name === 'p2g2') {
      // Sorted stress: densities per particle, then the per-cell scatter.
      runShader(encoder, 'density');
      runShader(encoder, 'p2g2s');
      return;
    }
    runShader(encoder, name === 'p2g1' && sorted ? 'p2g1s' : (name as MpmShaderName));
  }

  function runShader(encoder: GPUCommandEncoder, shader: MpmShaderName): void {
    const compute = pass(shader);
    if (shader === 'p2g1' || shader === 'p2g2' || shader === 'density' || shader === 'g2p') {
      compute.runCount(encoder, resources(shader), system.count);
      return;
    }
    if (system.count === 0 && (shader === 'p2g1s' || shader === 'p2g2s')) return;
    const items =
      shader === 'p2g1s' || shader === 'p2g2s'
        ? (dims[0] - 2) * (dims[1] - 2) * (dims[2] - 2) * 9
        : cells;
    const groups = Math.ceil(items / sources[shader].workgroup);
    const x = Math.min(groups, 32768);
    const y = Math.ceil(groups / x);
    compute.runGroups(encoder, resources(shader), [x, y, 1]);
  }

  // --- Encoding --------------------------------------------------------------------------------------------------
  function scalarCode(): number {
    return params.scalar === 'speed' ? 1 : params.scalar === 'density' ? 2 : 0;
  }

  function writeStep(
    slot: number,
    dt: number,
    effectiveNow: MpmEffective,
    bounds: MpmBounds,
    wallMin: Vec3,
    wallMax: Vec3,
    colliderOffsets: number,
  ): void {
    put(slot, 'dims', dims);
    put(slot, 'count', system.count);
    put(slot, 'h', h);
    put(slot, 'inv_h', 1 / h);
    put(slot, 'dt', dt);
    put(slot, 'rest_ppc', Math.max(params.particlesPerCell, 0.5));
    put(slot, 'gravity', params.gravity);
    put(slot, 'sound_speed', effectiveNow.soundSpeed);
    put(slot, 'gamma', clamp(params.gamma, 1, 10));
    put(slot, 'viscosity', effectiveNow.viscosity);
    put(slot, 'tension', clamp(params.tension, 0, 1));
    put(slot, 'max_speed', params.maxSpeed);
    put(slot, 'bounds_min', bounds.min);
    put(slot, 'friction', clamp(params.wallFriction, 0, 1));
    put(slot, 'bounds_max', bounds.max);
    put(slot, 'r_max', clamp(params.densityLimit, 1.05, 6));
    put(slot, 'wall_vmin', wallMin);
    put(slot, 'wall_vmax', wallMax);
    put(slot, 'scalar_mode', scalarCode());
    put(slot, 'foam', [
      params.foam.sprayLow,
      params.foam.sprayHigh,
      params.foam.speedLow,
      params.foam.speedHigh,
    ]);
    put(slot, 'foam_decay', Math.exp(-params.foam.decayPerSecond * dt));
    put(slot, 'wall_comp', params.wallCompensation ? 1 : 0);
    put(slot, 'damping_factor', Math.exp(-Math.max(params.damping, 0) * dt));
    const list = colliders.slice(0, MPM_MAX_COLLIDERS);
    put(slot, 'collider_count', list.length);
    const data: number[] = [];
    for (const collider of list) {
      const velocity = collider.velocity ?? [0, 0, 0];
      const friction = clamp(collider.friction ?? 0, 0, 1);
      const [cx, cy, cz] = collider.center;
      const move = colliderOffsets;
      const centre = [cx + velocity[0] * move, cy + velocity[1] * move, cz + velocity[2] * move];
      if (collider.shape === 'sphere') {
        data.push(...centre, 1, collider.radius ?? 0.1, 0, 0, friction);
      } else if (collider.shape === 'box') {
        const e = collider.halfExtent ?? [0.1, 0.1, 0.1];
        data.push(...centre, 2, e[0], e[1], e[2], friction);
      } else {
        const n = collider.normal ?? [0, 1, 0];
        const l = Math.hypot(n[0], n[1], n[2]) || 1;
        data.push(...centre, 3, n[0] / l, n[1] / l, n[2] / l, friction);
      }
      data.push(velocity[0], velocity[1], velocity[2], 0);
    }
    put(slot, 'colliders', data);
  }

  function encodeSteps(
    encoder: GPUCommandEncoder,
    count: number,
    passes: readonly MpmPassName[],
    timedSteps: number,
    dt: number,
  ): void {
    if (count > MPM_RING_STEPS) {
      throw new RangeError(
        `${label}: ${count} steps exceed the ${MPM_RING_STEPS} that one submit can hold`,
      );
    }
    if (ringCursor + count > MPM_RING_STEPS) ringCursor = 0;
    const first = ringCursor;
    ringCursor += count;

    // Where the walls travel this update: constant speed, no faster than wallCourant cells per step.
    const reach = clamp(params.wallCourant, 0.01, 0.9) * h * count;
    const delta = { min: [0, 0, 0], max: [0, 0, 0] };
    for (let a = 0; a < 3; a++) {
      delta.min[a] = clamp(target.min[a]! - current.min[a]!, -reach, reach);
      delta.max[a] = clamp(target.max[a]! - current.max[a]!, -reach, reach);
    }
    const total = count * dt;
    const wallMin: Vec3 = [delta.min[0]! / total, delta.min[1]! / total, delta.min[2]! / total];
    const wallMax: Vec3 = [delta.max[0]! / total, delta.max[1]! / total, delta.max[2]! / total];
    const moving = delta.min.some((x) => x !== 0) || delta.max.some((x) => x !== 0);

    for (let k = 0; k < count; k++) {
      const fraction = (k + 1) / count;
      const bounds: MpmBounds = {
        min: [0, 1, 2].map((a) => current.min[a]! + delta.min[a]! * fraction) as unknown as Vec3,
        max: [0, 1, 2].map((a) => current.max[a]! + delta.max[a]! * fraction) as unknown as Vec3,
      };
      writeStep(first + k, dt, effective, bounds, wallMin, wallMax, (k + 1) * dt);
    }
    device.queue.writeBuffer(ringBuffer, first * stride, ringData, first * stride, count * stride);

    for (let k = 0; k < count; k++) {
      encoder.copyBufferToBuffer(ringBuffer, (first + k) * stride, uniformBuffer, 0, stride);
      timing.on = k < timedSteps;
      for (const name of passes) dispatch(encoder, name);
    }

    timing.on = false;
    if (moving) {
      current = {
        min: [0, 1, 2].map((a) => current.min[a]! + delta.min[a]!) as unknown as Vec3,
        max: [0, 1, 2].map((a) => current.max[a]! + delta.max[a]!) as unknown as Vec3,
      };
    }
    for (const collider of colliders) {
      const v = collider.velocity;
      if (v) {
        collider.center = [
          collider.center[0] + v[0] * total,
          collider.center[1] + v[1] * total,
          collider.center[2] + v[2] * total,
        ];
      }
    }
    simTime += total;
    stats.steps += count;
  }

  function emitPour(seconds: number): void {
    if (!pour || !pour.active || seconds <= 0 || pour.speed <= 0) return;
    const spacing = spacingNow();
    const interval = spacing / pour.speed;
    const [dx, dy, dz] = pour.direction;
    const length = Math.hypot(dx, dy, dz) || 1;
    const direction: Vec3 = [dx / length, dy / length, dz / length];
    pourClock += seconds;
    const disc = discLattice(pour.radius, spacing, direction);
    const positions: number[] = [];
    const velocities: number[] = [];
    let layers = 0;
    while (pourClock >= interval && layers < 64) {
      pourClock -= interval;
      layers++;
      const ahead = pourClock * pour.speed;
      for (let i = 0; i < disc.length; i += 3) {
        const jx = (random() - 0.5) * 0.2 * spacing;
        const jy = (random() - 0.5) * 0.2 * spacing;
        const jz = (random() - 0.5) * 0.2 * spacing;
        positions.push(
          pour.position[0] + disc[i]! + direction[0] * ahead + jx,
          pour.position[1] + disc[i + 1]! + direction[1] * ahead + jy,
          pour.position[2] + disc[i + 2]! + direction[2] * ahead + jz,
          0,
        );
        velocities.push(
          direction[0] * pour.speed,
          direction[1] * pour.speed,
          direction[2] * pour.speed,
          0,
        );
      }
    }
    if (positions.length > 0) append(new Float32Array(positions), new Float32Array(velocities));
  }

  function append(positions: Float32Array, velocities: Float32Array | undefined): number {
    const start = system.count;
    const n = Math.min(positions.length / 4, system.capacity - start);
    if (n <= 0) return 0;
    system.setCount(start + n);
    sortedCount = -1;
    device.queue.writeBuffer(system.attribute('position'), start * 16, positions, 0, n * 4);
    if (velocities) {
      device.queue.writeBuffer(system.attribute('velocity'), start * 16, velocities, 0, n * 4);
    }
    return n;
  }

  function addBlock(block: MpmBlock): number {
    const positions = blockLattice(block, spacingNow(), 0x5eed + system.count);
    const velocities = new Float32Array(positions.length);
    const v = block.velocity;
    if (v && (v[0] !== 0 || v[1] !== 0 || v[2] !== 0)) {
      for (let i = 0; i < positions.length; i += 4) {
        velocities[i] = v[0];
        velocities[i + 1] = v[1];
        velocities[i + 2] = v[2];
      }
    }
    return append(positions, velocities);
  }

  function jumpBounds(bounds: Partial<MpmBounds>): void {
    current = clampBounds(bounds.min ?? current.min, bounds.max ?? current.max);
    target = { min: [...current.min], max: [...current.max] };
  }

  async function readBuffer(buffer: GPUBuffer, bytes: number): Promise<ArrayBuffer> {
    const readback = device.createBuffer({
      label: `${label} readback`,
      size: bytes,
      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    });
    const encoder = device.createCommandEncoder();
    encoder.copyBufferToBuffer(buffer, 0, readback, 0, bytes);
    device.queue.submit([encoder.finish()]);
    await readback.mapAsync(GPUMapMode.READ);
    const copy = readback.getMappedRange().slice(0);
    readback.unmap();
    readback.destroy();
    return copy;
  }

  const solver: MpmSolver = {
    device,
    system,
    particles,
    params,
    get effective() {
      return effective;
    },
    dims,
    cellSize: h,
    get spacing() {
      return spacingNow();
    },
    size,
    get bounds() {
      return current;
    },
    get capacity() {
      return system.capacity;
    },
    get count() {
      return system.count;
    },
    get generation() {
      return system.generation;
    },
    get time() {
      return simTime;
    },
    stats,
    gridBytes,

    update(encoder, delta) {
      effective = mpmStability(h, params);
      const maxSteps = Math.max(1, Math.round(params.maxSteps));
      const key = `${effective.dt}|${maxSteps}`;
      if (!stepper || key !== stepperKey) {
        stepper = createFixedStepper(effective.dt, { maxSteps });
        stepperKey = key;
      }
      stats.updates++;
      const { steps } = stepper.advance(delta);
      const owed = Math.max(0, delta) / effective.dt;
      const behind = owed > maxSteps + 1;
      emitPour(steps * effective.dt);
      if (steps > 0) {
        encodeSteps(
          encoder,
          steps,
          MPM_PASSES,
          Math.max(0, Math.round(params.profileSteps)),
          effective.dt,
        );
      }
      return { steps, behind };
    },

    step(encoder, stepOptions = {}) {
      effective = mpmStability(h, params);
      const count = Math.max(1, Math.round(stepOptions.count ?? 1));
      encodeSteps(
        encoder,
        count,
        stepOptions.passes ?? MPM_PASSES,
        stepOptions.timed ? 1 : 0,
        effective.dt,
      );
    },

    setBounds(bounds) {
      target = clampBounds(bounds.min ?? target.min, bounds.max ?? target.max);
    },

    jumpBounds,

    setColliders(list) {
      colliders = list.slice(0, MPM_MAX_COLLIDERS).map((c) => ({ ...c }));
    },

    addBlock,
    addParticles: append,

    setPour(next) {
      pour = next ? { ...next } : undefined;
      if (!next) pourClock = 0;
    },

    reset(scene = {}) {
      system.setCount(0);
      sortedCount = -1;
      simTime = 0;
      pourClock = 0;
      stepper = undefined;
      jumpBounds(scene.bounds ?? { min: current.min, max: current.max });
      if (scene.gravity) params.gravity = scene.gravity;
      colliders = (scene.colliders ?? []).slice(0, MPM_MAX_COLLIDERS).map((c) => ({ ...c }));
      pour = scene.pour ? { ...scene.pour } : undefined;
      for (const block of scene.blocks ?? []) addBlock(block);
      system.radius = radiusScale * spacingNow();
    },

    setCapacity(capacity) {
      const next = Math.floor(capacity);
      if (next > particleLimit || next > MAX_PARTICLES_ANY) {
        throw new RangeError(
          `${label}: ${next} particles exceed the dispatch limit of ${particleLimit}`,
        );
      }
      system.resize(next);
    },

    async readGrid() {
      const raw = new Int32Array(await readBuffer(gridBuffer, gridBytes));
      const mass = new Float32Array(cells);
      const momentum = new Float32Array(cells * 3);
      for (let i = 0; i < cells; i++) {
        mass[i] = raw[i * 4]! / MPM_FIXED_SCALE;
        momentum[i * 3] = raw[i * 4 + 1]! / MPM_FIXED_SCALE;
        momentum[i * 3 + 1] = raw[i * 4 + 2]! / MPM_FIXED_SCALE;
        momentum[i * 3 + 2] = raw[i * 4 + 3]! / MPM_FIXED_SCALE;
      }
      return { mass, momentum };
    },

    async readGridVelocity() {
      return new Float32Array(await readBuffer(gridVelocity, gridBytes));
    },

    destroy() {
      if (destroyed) return;
      destroyed = true;
      for (const handle of handles.values()) handle.destroy();
      sorter?.destroy();
      system.destroy();
      gridBuffer.destroy();
      gridVelocity.destroy();
      ringBuffer.destroy();
      uniformBuffer.destroy();
    },
  };
  return solver;
}
