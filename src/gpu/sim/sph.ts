import { createFixedStepper, type FixedStepper } from '../loop.ts';
import { noProfiler, type PassProfiler } from '../profiler.ts';
import {
  createComputePass,
  createNeighborGrid,
  createParticleSystem,
  type ComputePass,
  type NeighborGrid,
  type ParticleSet,
  type ParticleSystem,
} from '../particles/index.ts';
import { createUniformBuffer, defineUniformStruct, uniformArray } from '../uniforms.ts';
import coreSource from './sph-core.wgsl';
import densitySource from './sph-density.wgsl';
import forceSource from './sph-force.wgsl';

// Weakly compressible SPH in 2D (story 7.3): smoothed-particle hydrodynamics after Muller, Charypar & Gross,
// "Particle-Based Fluid Simulation for Interactive Applications" (2003), with the standard refinements of the
// weakly compressible literature (Monaghan 1994/2005; Becker & Teschner 2007; Akinci et al. 2013).
//
// ONE SUB-STEP
//   1. grid     the neighbour grid (src/gpu/particles/neighbor-grid.ts, reorder mode) bins the particles and
//               PHYSICALLY REORDERS positions and velocities into cell order. Built every `rebuildEvery`
//               sub-steps, not every one: see "skin" below. (5 passes + the gather.)
//   2. density  rho_i = m sum_j W_poly6(r_ij) + the wall term; p_i from the equation of state.   (1 pass)
//   3. forces   pressure, viscosity, artificial viscosity, cohesion, walls, gravity, pointer; then symplectic
//               Euler, then the hard box and obstacle projection. Writes the NEW positions and velocities to the
//               other half of a double buffer, which is then swapped.                             (1 pass)
//   So a sub-step is 2 dispatches, plus 6 on every `rebuildEvery`-th one. Kernels: src/wgsl/sph-kernels.wgsl.
//
// EQUATION OF STATE. Tait, p = B ((rho / rho0)^7 - 1), B = rho0 c^2 / 7, the standard weakly compressible choice:
// stiff above rest density, so a compressed region is pushed apart hard, and c is the speed of sound at rest.
// The ideal-gas law k (rho - rho0) of the 2003 paper is the same idea with a constant stiffness; Tait's stiffens
// as the fluid compresses, which keeps the density error of an impact small without a huge k everywhere. NEGATIVE
// PRESSURE IS CLAMPED TO ZERO: below rest density a particle gets no pull toward its neighbours. With the pull
// the free surface clumps (the tensile instability: particles snap into pairs and strings); without it a particle
// at the surface simply feels the pressure of the bulk and nothing else. The price is that the free surface is
// not held together by pressure at all; surface tension (below) is separate.
// The density ratio is capped at `pressure_cap` (1.25) when evaluating p: an overlap (a block dropped into water)
// pushes hard but with a finite force.
//
// REST DENSITY, SPACING AND SMOOTHING LENGTH. Particles are spawned on a hexagonal lattice of spacing s: one
// particle per area s^2. The support radius is h = smoothingRatio * s (default 2.4 s, about 18 neighbours in 2D).
// The particle mass is the nominal density times the area, m = rho_nominal s^2 (the same for every particle).
// The rest density rho0 used in the equation of state is NOT assumed equal to rho_nominal: it is the density the
// kernel sum itself reads on that lattice, rho0 = m sum_lattice W_poly6(|x|), computed here in double precision
// (`latticeDensity`). So a block of fluid at lattice spacing reads exactly rho0 in its interior and starts with
// zero pressure: it neither explodes nor collapses. The two differ by under 1 percent for smoothingRatio >= 2.2
// (the sum over a lattice of a smooth kernel is its integral, 1, plus a small aliasing term; a square lattice
// differs from the hexagonal one by a similar amount, tests/sph.test.ts measures both).
// CHANGING RESOLUTION WITHOUT RETUNING: only the spacing changes. h and m follow it (h = ratio s, m = rho s^2),
// every physical parameter (gravity, viscosity in m^2/s, the speed of sound, surface tension in m/s^2, pointer
// strengths) is independent of it, and the time step shrinks in proportion to s (see below). Twice the particles
// per axis means a quarter of the spacing squared, twice the sub-steps.
//
// TIME STEP AND SUB-STEPS. Explicit SPH is stable only if no signal crosses a kernel in one step:
//     dt <= courant * h / c,   c = stiffness * v_ref,   v_ref = sqrt(max(|g|, ...) * H)      (the CFL rule)
// plus the force limit 0.25 sqrt(h / a) (gravity and surface tension), whichever is smallest. The laminar viscosity
// is capped instead of shrinking dt (nu <= 0.12 h^2 / dt). c is chosen, not physical: it sets how compressible the
// fluid is. The hydrostatic compression of a column of height H is about g H / c^2 = 1 / stiffness^2 of the domain
// height (6 percent at the default stiffness 4 for a column as tall as the domain; 1.7 percent measured for a column
// 0.5 m tall in a 1.2 m domain), and the density error of a flow at speed v is about (v / c)^2. v_ref is the free-fall
// speed over the domain height, floored at 2 m/s so a zero-gravity scene still has a finite c. That makes dt
// proportional to the spacing: the sub-steps for one 1/60 s frame are
//     n = ceil((1/60) / dt) = ceil((1/60) * stiffness * v_ref / (courant * h)).
// It is the cost driver of SPH: the demo's tank (2 x 1.2 m, a 0.55 x 0.95 m column) needs 12 sub-steps per frame at
// 2 000 particles, 27 at 10 000, 38 at 20 000, 60 at 50 000, 84 at 100 000 and 132 at 250 000. The stability limit
// was MEASURED on the dam break (stiffness 5, 1 100 particles): courant 0.8 is stable, 1.0 does not (the fluid
// disintegrates: the density falls to a fifth of rest and the kinetic energy rises instead of decaying), so the default 0.5 keeps a margin of 1.6; the textbook 0.125 h / c for a cubic spline is
// five times more cautious than these kernels need. The solver runs the integer n per tick (so dt is exactly
// (1/60)/n, and a 120 Hz display runs n/2 per frame), capped at `maxSubSteps` (default 128); a capped step is larger
// than stable, which the stability clamps absorb at the price of accuracy. The velocity is clamped to
// min(0.4 c, h / (2 dt)): no particle crosses half a kernel in a step.
//
// SKIN. The grid may be rebuilt only every K steps instead of every one. Its cell side is then h + skin with
// skin = (K - 1) speed_max dt: the last step that still uses a grid sees positions K - 1 steps old, and no particle
// moves more than speed_max dt in one step (the velocity clamp), so the 3x3 cell block around a particle still
// contains every pair that is within h NOW. The search stays EXACT, not approximate (tests/sph.test.ts compares a
// K = 8 density with a brute-force search). K defaults to the largest value that keeps skin <= 0.35 h, which is 1 to 2
// at this clamp (a particle may in principle move 0.4 c, much faster than water does): the study table says what
// K = 1 versus 2 buys. Between rebuilds the particles are still almost in cell order, so the loops stay cache friendly.
//
// BOUNDARIES. Penalty forces make a layer of particles stick to the floor and a gap or a dense crust at the wall;
// boundary particles cost an extra particle layer per wall and per obstacle and need their own mass rule. Instead
// the walls are an analytic CONTINUUM (sph-core.wgsl): the missing neighbours behind a wall add their integrated
// kernel to the density and their integrated kernel gradient to the pressure force, tabulated on the CPU
// (`sphWallTables`). The wall acts exactly like fluid at rest behind it: a column settles without a crust, the
// floor feels the hydrostatic pressure, and the particles' first layer stands about half a spacing from the wall.
// Obstacles are SDFs (circle, rounded box) used as locally flat walls. Corners count the two half-planes
// separately, which over-counts the density a little in the very corner. On top of the force a HARD projection
// puts every particle back inside the box (and out of the obstacles) with restitution and friction after every
// step; it is the safety net that makes "no particle leaves the box" true whatever the forces do.
//
// SURFACE TENSION is the cohesion term (a pairwise attraction, Akinci et al. 2013, with their own spline replaced
// by the smooth bump 16 q^2 (1 - q)^2, 2D-normalised by the rest area). It is equal and opposite per pair, so it
// conserves momentum, costs one multiply per neighbour, and makes a free blob round and keeps droplets together.
// It is a cohesion model, not a curvature model: the effective tension depends on the packing.
//
// VISCOSITY. Laminar: nu * sum m_j 2 / (rho_i + rho_j) (v_j - v_i) Laplacian W (kernel constant fixed by the
// consistency condition in sph-kernels.wgsl, so nu is a true kinematic viscosity in m^2/s). Artificial (Monaghan):
// only between approaching particles, it damps the sound waves of the weakly compressible fluid and the particle
// disorder behind an impact. A linear `damping` (1/s) stands for the drag of air and the walls' boundary layer.
//
// WHAT IT CONSERVES. Pressure, viscosity, artificial viscosity and cohesion are all sums of equal and opposite pair
// forces and use symmetric densities, so with no gravity, walls or pointer, total momentum is conserved to
// round-off (tests/sph.test.ts).
//
// UNITS. Metres, kilograms (per unit depth), seconds. A 2 x 1.2 m tank is the default scale: gravity 9.81 looks
// right at that size.
//
// PER-PARTICLE SCALAR. `positions.w` holds, by `params.scalar`: `density` (rho / rho0: 1 in the bulk, below 1 at the
// free surface and in spray, above 1 when compressed), `pressure` (p / (rho0 g H): 1 at the bottom of a full tank)
// or `speed` (m/s). `velocities.w` always holds the speed. It is written by the force pass for free.
//
// IDENTITY. The grid reorders every attribute each K steps: particle index i is NOT stable. Nothing here needs an
// id; a test that follows particles keeps its own.
//
// PROFILER. A frame can run a hundred sub-steps, more passes than the profiler has slots (32), and summing their
// intervals would be wrong anyway. Only the first and the last sub-step of an `update` are timed: the HUD's frame
// span (first timed pass start to last timed pass end) then covers the whole simulation, and the per-pass labels
// ("sph grid", "sph density", "sph forces") show the cost of those two sub-steps together.

// --- Types ---------------------------------------------------------------------------------------

export type Vec2 = readonly [number, number];
export type SphScalar = 'density' | 'pressure' | 'speed';

export interface SphPointer {
  /** World position, metres. */
  x: number;
  y: number;
  /** Pointer velocity, m/s: the drag target of a push. */
  vx: number;
  vy: number;
  /** Influence radius, metres. */
  radius: number;
  /** push: a drag RATE in 1/s (30 is firm); pull: an acceleration in m/s^2. */
  strength: number;
  mode: 'push' | 'pull';
}

export type SphObstacle =
  | { shape: 'circle'; center: Vec2; radius: number }
  | { shape: 'box'; center: Vec2; halfSize: Vec2; cornerRadius?: number };

/** Live parameters: change any of them at any time; they are read on every update. */
export interface SphParams {
  /** Acceleration of gravity, m/s^2 (it may point anywhere: the container-tilt story drives it). */
  gravity: [number, number];
  /** Seconds over which gravity ramps up after a reset (a soft start avoids the hydrostatic compression ringing). */
  gravityRamp: number;
  /** Speed of sound as a multiple of the reference speed sqrt(|g| H). Hydrostatic compression ~ 1 / stiffness^2. */
  stiffness: number;
  /** Kinematic viscosity, m^2/s (water 1e-6; 0.005 to 0.05 reads as water at this scale; 1 is honey). */
  viscosity: number;
  /** Monaghan's alpha, 0 to 1. */
  artificialViscosity: number;
  /** Cohesion, m/s^2: peak pair attraction scale. 0 is off; 10 to 40 gives a round blob. */
  surfaceTension: number;
  /** Linear damping, 1/s. */
  damping: number;
  /** Normal velocity kept at a wall, 0..1. */
  restitution: number;
  /** Fraction of the tangential velocity lost per wall contact step, 0..1. */
  friction: number;
  /** Simulation ticks per second; each tick runs `subSteps` sub-steps. */
  stepRate: number;
  /** Sub-steps per tick; 0 takes the number the CFL rule asks for. */
  subSteps: number;
  /** Cap on the sub-steps per tick. */
  maxSubSteps: number;
  /** The CFL factor: dt <= courant * h / c. */
  courant: number;
  /** Rebuild the grid every this many sub-steps; 0 takes the most the skin allows. */
  rebuildEvery: number;
  /** What `positions.w` carries. */
  scalar: SphScalar;
  /** Up to 4 pointers. */
  pointers: SphPointer[];
  /** Up to 4 obstacles. */
  obstacles: SphObstacle[];
}

export interface SphBounds {
  min: Vec2;
  max: Vec2;
}

export type SphBlock =
  { min: Vec2; max: Vec2; velocity?: Vec2 } | { center: Vec2; radius: number; velocity?: Vec2 };

export interface SphEmitter {
  /** Centre of the nozzle, m. */
  position: Vec2;
  /** Direction of the jet (normalised). */
  direction: Vec2;
  /** Width of the jet, m. */
  width: number;
  /** Speed of the jet, m/s. The flow rate follows: rows of particles are released at speed / row spacing. */
  speed: number;
}

export interface SphScene {
  bounds?: SphBounds;
  spacing?: number;
  blocks?: SphBlock[];
  /** Explicit particles after the blocks: vec4 rows of positions (x, y, 0, scalar) and velocities (vx, vy, 0, 0). */
  particles?: { positions: Float32Array; velocities?: Float32Array };
  emitter?: SphEmitter | null;
  obstacles?: SphObstacle[];
  gravity?: [number, number];
}

export interface SphOptions {
  device: GPUDevice;
  /** Most particles. Change later with `setCapacity`. */
  capacity: number;
  bounds: SphBounds;
  /** Particle spacing s, metres. */
  spacing: number;
  /** h / s. Default 2.4. */
  smoothingRatio?: number;
  /** Nominal density, kg/m^2 (the mass per area): fixes the particle mass. Default 1000. */
  density?: number;
  params?: Partial<SphParams>;
  profiler?: PassProfiler;
  label?: string;
}

/** Everything the solver derives from the spacing, the bounds and the parameters. */
export interface SphDerived {
  spacing: number;
  /** Support radius h. */
  h: number;
  mass: number;
  nominalDensity: number;
  /** The rest density the equation of state uses: the lattice kernel sum. */
  restDensity: number;
  soundSpeed: number;
  referenceSpeed: number;
  /** Seconds. */
  dt: number;
  /** The CFL limit alone, before rounding to a whole number of sub-steps per tick. */
  dtLimit: number;
  /** Sub-steps per tick (1 / stepRate seconds). */
  subSteps: number;
  speedMax: number;
  /** The kinematic viscosity actually used: `params.viscosity`, capped at what the explicit step can carry. */
  viscosity: number;
  skin: number;
  rebuildEvery: number;
  pressureRef: number;
  /** Hydrostatic compression of a column of the domain height: g H / c^2. */
  hydrostaticCompression: number;
}

/** Which parts of a sub-step to record, for cost studies. Omit to run everything on the usual schedule. */
export interface SphStages {
  /** Rebuild the grid on every step (true) or not at all (false). */
  grid?: boolean;
  density?: boolean;
  forces?: boolean;
}

export interface SphSnapshot {
  count: number;
  x: Float32Array;
  y: Float32Array;
  vx: Float32Array;
  vy: Float32Array;
  density: Float32Array;
  pressure: Float32Array;
  scalar: Float32Array;
}

export interface SphSolver {
  readonly system: ParticleSystem;
  /** The renderer-facing view, with `restVolume` (the area per particle, m^2) set. */
  readonly particles: ParticleSet;
  readonly params: SphParams;
  readonly grid: NeighborGrid;
  readonly derived: SphDerived;
  readonly bounds: SphBounds;
  readonly count: number;
  readonly capacity: number;
  /** Simulated seconds since the last reset. */
  readonly time: number;
  /** Sub-steps run so far, and grid rebuilds among them. */
  readonly stats: { steps: number; builds: number };
  /** Advances by the whole fixed sub-steps `delta` seconds add up to; returns how many ran. */
  update(encoder: GPUCommandEncoder, delta: number): number;
  /** Runs exactly `steps` sub-steps of `dt` seconds (default the derived dt), regardless of the clock. */
  step(encoder: GPUCommandEncoder, steps?: number, dt?: number, stages?: SphStages): void;
  /** Replaces the fluid with the scene's blocks (applied at the start of the next update). */
  reset(scene?: SphScene): void;
  /** Appends a block of fluid (applied at the start of the next update). */
  addBlock(block: SphBlock): void;
  setEmitter(emitter: SphEmitter | null): void;
  setCapacity(capacity: number): void;
  /** Copies the particles back to the CPU: for tests and tools. */
  read(): Promise<SphSnapshot>;
  /** Bumps when buffers are re-created. */
  readonly generation: number;
  destroy(): void;
}

// --- Kernel constants and tables (CPU twins of the WGSL) ---------------------------------------------

const poly6 = (r: number, h: number): number =>
  r >= h ? 0 : (4 / (Math.PI * h ** 8)) * (h * h - r * r) ** 3;

/** The hexagonal lattice of unit area per particle: column and row spacing. */
export function hexSpacing(spacing: number): { dx: number; dy: number } {
  const dx = spacing * Math.sqrt(2 / Math.sqrt(3));
  return { dx, dy: (dx * Math.sqrt(3)) / 2 };
}

/**
 * Density read by the kernel sum on an infinite hexagonal lattice of one particle per spacing^2 area, for particles
 * of mass `spacing^2 * nominalDensity`: the rest density of the equation of state. `lattice: 'square'` sums over a
 * square lattice instead (for the comparison in the tests).
 */
export function latticeDensity(
  spacing: number,
  h: number,
  nominalDensity = 1000,
  lattice: 'hex' | 'square' = 'hex',
): number {
  const mass = nominalDensity * spacing * spacing;
  const reach = Math.ceil(h / spacing) + 3;
  const { dx, dy } = hexSpacing(spacing);
  let sum = 0;
  if (lattice === 'square') {
    for (let i = -reach; i <= reach; i++) {
      for (let j = -reach; j <= reach; j++) sum += poly6(Math.hypot(i, j) * spacing, h);
    }
  } else {
    const rows = Math.ceil(h / dy) + 3;
    const cols = Math.ceil(h / dx) + 3;
    for (let j = -rows; j <= rows; j++) {
      for (let i = -cols; i <= cols; i++) {
        const x = (i + (j & 1 ? 0.5 : 0)) * dx;
        sum += poly6(Math.hypot(x, j * dy), h);
      }
    }
  }
  return mass * sum;
}

/** Entries of each wall table (33 samples of d / h from 0 to 1). */
export const WALL_TABLE_SIZE = 33;

function simpson(f: (x: number) => number, a: number, b: number, n = 1024): number {
  const step = (b - a) / n;
  let sum = f(a) + f(b);
  for (let i = 1; i < n; i++) sum += f(a + i * step) * (i % 2 ? 4 : 2);
  return (sum * step) / 3;
}

/**
 * The two tables of the analytic wall (sph-core.wgsl), for h = 1; both are functions of q = d / h, d the distance to
 * the wall:
 *  - `gamma[q]`: the integral of W_poly6 over the half-plane beyond distance d. gamma(0) = 1/2, gamma(1) = 0.
 *  - `force[q]`: the integral of W_spiky along the line at distance d (the half-plane integral of its gradient is
 *    this times the wall normal). It is in units of 1 / h.
 */
export function sphWallTables(): { gamma: Float64Array; force: Float64Array } {
  const gamma = new Float64Array(WALL_TABLE_SIZE);
  const force = new Float64Array(WALL_TABLE_SIZE);
  // The poly6 line integral has the closed form (4 / pi) (32 / 35) (1 - x^2)^(7/2).
  const line = (x: number) => (4 / Math.PI) * (32 / 35) * Math.max(1 - x * x, 0) ** 3.5;
  for (let k = 0; k < WALL_TABLE_SIZE; k++) {
    const d = k / (WALL_TABLE_SIZE - 1);
    gamma[k] = d >= 1 ? 0 : simpson(line, d, 1);
    const half = Math.sqrt(Math.max(1 - d * d, 0));
    force[k] =
      half <= 0
        ? 0
        : 2 *
          simpson((y) => (10 / Math.PI) * Math.max(1 - Math.hypot(d, y), 0) ** 3, 0, half, 1024);
  }
  return { gamma, force };
}

// --- Uniforms ---------------------------------------------------------------------------------------

export const SphUniform = defineUniformStruct('SphParams', {
  gravity: 'vec2f',
  dt: 'f32',
  h: 'f32',
  box_min: 'vec2f',
  box_max: 'vec2f',
  mass: 'f32',
  rest_density: 'f32',
  tait_b: 'f32',
  gamma: 'f32',
  viscosity: 'f32',
  artificial_viscosity: 'f32',
  cohesion: 'f32',
  damping: 'f32',
  speed_max: 'f32',
  margin: 'f32',
  restitution: 'f32',
  friction: 'f32',
  sound_speed: 'f32',
  pressure_cap: 'f32',
  pressure_ref: 'f32',
  count: 'u32',
  scalar_mode: 'u32',
  obstacle_count: 'u32',
  pointer_count: 'u32',
  obstacles: uniformArray('vec4f', 8),
  pointers: uniformArray('vec4f', 8),
  wall_gamma: uniformArray('vec4f', 9),
  wall_force: uniformArray('vec4f', 9),
});

const MAX_OBSTACLES = 4;
const MAX_POINTERS = 4;
const UNIFORM_RING = 4;
const SCALAR_CODE: Record<SphScalar, number> = { density: 0, pressure: 1, speed: 2 };
/** Largest density ratio the equation of state evaluates (see sph-core.wgsl). */
const PRESSURE_CAP = 1.25;
const SKIN = 0.35;
const MAX_REBUILD = 8;

export const DEFAULT_SPH_PARAMS: Readonly<Omit<SphParams, 'pointers' | 'obstacles'>> = {
  gravity: [0, -9.81],
  gravityRamp: 0,
  stiffness: 4,
  viscosity: 0.05,
  artificialViscosity: 0.2,
  surfaceTension: 0,
  damping: 0.3,
  restitution: 0.1,
  friction: 0.05,
  stepRate: 60,
  subSteps: 0,
  maxSubSteps: 128,
  courant: 0.5,
  rebuildEvery: 0,
  scalar: 'density',
};

/**
 * Derives everything from the spacing, the domain and the parameters. Pure, so tests and tools can ask "how many
 * sub-steps does this resolution need" without a GPU.
 */
export function deriveSph(
  spacing: number,
  smoothingRatio: number,
  nominalDensity: number,
  bounds: SphBounds,
  params: Pick<
    SphParams,
    | 'gravity'
    | 'stiffness'
    | 'viscosity'
    | 'surfaceTension'
    | 'courant'
    | 'stepRate'
    | 'subSteps'
    | 'maxSubSteps'
    | 'rebuildEvery'
  >,
): SphDerived {
  const h = smoothingRatio * spacing;
  const mass = nominalDensity * spacing * spacing;
  const restDensity = latticeDensity(spacing, h, nominalDensity);
  const height = bounds.max[1] - bounds.min[1];
  const g = Math.hypot(params.gravity[0], params.gravity[1]);
  const referenceSpeed = Math.max(Math.sqrt(g * height), 2);
  const soundSpeed = Math.max(params.stiffness, 1) * referenceSpeed;
  let dtLimit = (params.courant * h) / soundSpeed;
  if (g > 0) dtLimit = Math.min(dtLimit, 0.25 * Math.sqrt(h / g));
  if (params.surfaceTension > 0) {
    dtLimit = Math.min(dtLimit, 0.25 * Math.sqrt(h / params.surfaceTension));
  }
  const tick = 1 / params.stepRate;
  const asked = params.subSteps > 0 ? Math.round(params.subSteps) : Math.ceil(tick / dtLimit);
  const subSteps = Math.max(1, Math.min(Math.round(params.maxSubSteps), asked));
  const dt = tick / subSteps;
  // The laminar viscosity is an explicit diffusion: stable only while nu dt <= 0.15 h^2 (the Laplacian kernel sums to
  // 40 / (3 h^2)). Rather than shrink dt as h^2 at high resolution, the viscosity that can be carried is capped.
  const viscosity = Math.min(params.viscosity, (0.12 * h * h) / dt);
  const speedMax = Math.min(0.4 * soundSpeed, (0.5 * h) / dt);
  // The grid is rebuilt every K steps; the last step that still uses it sees positions K - 1 steps old, and no
  // particle moves more than speedMax * dt in a step, so a skin of (K - 1) speedMax dt keeps the search exact.
  const stepMove = speedMax * dt;
  const autoRebuild = 1 + Math.floor((SKIN * h) / stepMove);
  const rebuildEvery =
    params.rebuildEvery > 0
      ? Math.floor(params.rebuildEvery)
      : Math.max(1, Math.min(MAX_REBUILD, autoRebuild));
  const skin = (rebuildEvery - 1) * stepMove;
  return {
    spacing,
    h,
    mass,
    nominalDensity,
    restDensity,
    soundSpeed,
    referenceSpeed,
    dt,
    dtLimit,
    subSteps,
    viscosity,
    speedMax,
    skin,
    rebuildEvery,
    pressureRef: restDensity * Math.max(g, 1) * height,
    hydrostaticCompression: (g * height) / (soundSpeed * soundSpeed),
  };
}

// --- Lattices --------------------------------------------------------------------------------------

export function blockPoints(
  block: SphBlock,
  spacing: number,
  obstacles: readonly SphObstacle[] = [],
): [number, number][] {
  const { dx, dy } = hexSpacing(spacing);
  const lo: Vec2 =
    'radius' in block
      ? [block.center[0] - block.radius, block.center[1] - block.radius]
      : block.min;
  const hi: Vec2 =
    'radius' in block
      ? [block.center[0] + block.radius, block.center[1] + block.radius]
      : block.max;
  const points: [number, number][] = [];
  for (let row = 0; ; row++) {
    const y = lo[1] + dy * (row + 0.5);
    if (y > hi[1] - 0.25 * dy) break;
    for (let col = 0; ; col++) {
      const x = lo[0] + dx * (col + 0.5 + (row & 1 ? 0.5 : 0));
      if (x > hi[0] - 0.25 * dx) break;
      if ('radius' in block && Math.hypot(x - block.center[0], y - block.center[1]) > block.radius)
        continue;
      if (obstacles.some((o) => obstacleDistance(o, x, y) < 0.5 * spacing)) continue;
      points.push([x, y]);
    }
  }
  return points;
}

/** Signed distance of a point to an obstacle (negative inside); the CPU twin of sph_obstacle_sd. */
export function obstacleDistance(obstacle: SphObstacle, x: number, y: number): number {
  const qx = x - obstacle.center[0];
  const qy = y - obstacle.center[1];
  if (obstacle.shape === 'circle') return Math.hypot(qx, qy) - obstacle.radius;
  const r = Math.min(obstacle.cornerRadius ?? 0, obstacle.halfSize[0], obstacle.halfSize[1]);
  const ax = Math.abs(qx) - (obstacle.halfSize[0] - r);
  const ay = Math.abs(qy) - (obstacle.halfSize[1] - r);
  return Math.hypot(Math.max(ax, 0), Math.max(ay, 0)) + Math.min(Math.max(ax, ay), 0) - r;
}

// --- The solver ------------------------------------------------------------------------------------

export function createSph(options: SphOptions): SphSolver {
  const { device, profiler = noProfiler, label = 'sph' } = options;
  if (!(options.spacing > 0)) throw new RangeError(`${label}: spacing must be positive`);
  const smoothingRatio = options.smoothingRatio ?? 2.4;
  const nominalDensity = options.density ?? 1000;
  let capacity = Math.max(1, Math.floor(options.capacity));
  let spacing = options.spacing;
  let bounds: SphBounds = {
    min: [...options.bounds.min],
    max: [...options.bounds.max],
  };

  const params: SphParams = {
    ...DEFAULT_SPH_PARAMS,
    gravity: [...DEFAULT_SPH_PARAMS.gravity],
    pointers: [],
    obstacles: [],
    ...options.params,
  };
  params.gravity = [...params.gravity];

  // Only the first and the last sub-step of an update are timed (see the header).
  let timed = true;
  const gate: PassProfiler = { pass: (name) => (timed ? profiler.pass(name) : undefined) };

  const system = createParticleSystem({
    device,
    capacity,
    radius: 0.5 * spacing,
    dimension: 2,
    profiler: gate,
    label: `${label} particles`,
  });
  const grid = createNeighborGrid({
    device,
    dimension: 2,
    mode: 'reorder',
    capacity,
    profiler: gate,
    label: `${label} grid`,
  });
  const uniformRing = Array.from({ length: UNIFORM_RING }, (_, index) =>
    createUniformBuffer(device, SphUniform, `${label} params ${index}`),
  );
  let ringIndex = 0;

  const chunks = {
    'sph_params.wgsl': SphUniform.wgsl,
    'sph-core.wgsl': coreSource.code,
    ...coreSource.chunks,
    ...grid.chunks({ group: 0, binding: 3 }),
  };
  const forceChunks = {
    ...chunks,
    ...grid.chunks({ group: 0, binding: 6 }),
  };
  const densityPass: ComputePass = createComputePass(device, {
    label: 'sim/sph-density.wgsl',
    code: densitySource.code,
    chunks,
    profiler: gate,
    timestamp: 'sph density',
  });
  const forcePass: ComputePass = createComputePass(device, {
    label: 'sim/sph-force.wgsl',
    code: forceSource.code,
    chunks: forceChunks,
    profiler: gate,
    timestamp: 'sph forces',
  });

  let state = device.createBuffer({
    label: `${label} state`,
    size: capacity * 8,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
  });

  const tables = sphWallTables();
  const tableData = (table: Float64Array): number[] => {
    const out = new Array<number>(36).fill(0);
    table.forEach((value, i) => (out[i] = value));
    return out;
  };
  const gammaTable = tableData(tables.gamma);
  const forceTable = tableData(tables.force);

  let generation = 0;
  let simTime = 0;
  let sinceBuild = Number.POSITIVE_INFINITY;
  const stats = { steps: 0, builds: 0 };
  let destroyed = false;

  // Scene state and the queue of structural changes applied at the start of the next update.
  let scene: SphScene = {};
  let emitter: SphEmitter | null = null;
  let emitCarry = 0;
  const pending: (() => void)[] = [];

  let derived = deriveSph(spacing, smoothingRatio, nominalDensity, bounds, params);
  let stepper: FixedStepper | undefined;
  let stepperKey = '';

  const particles: ParticleSet = {
    get positions() {
      return system.positions;
    },
    get velocities() {
      return system.velocities;
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
      return spacing * spacing;
    },
    get generation() {
      return system.generation;
    },
  };

  function refreshDerived(): SphDerived {
    derived = deriveSph(spacing, smoothingRatio, nominalDensity, bounds, params);
    return derived;
  }

  function gravityAt(time: number): [number, number] {
    const ramp = params.gravityRamp > 0 ? Math.min(1, time / params.gravityRamp) : 1;
    const eased = ramp * ramp * (3 - 2 * ramp);
    return [params.gravity[0] * eased, params.gravity[1] * eased];
  }

  function packObstacles(): number[] {
    const data: number[] = [];
    const list = params.obstacles.slice(0, MAX_OBSTACLES);
    for (const o of list) {
      if (o.shape === 'circle') {
        data.push(1, o.center[0], o.center[1], 0, o.radius, 0, 0, 0);
      } else {
        data.push(
          2,
          o.center[0],
          o.center[1],
          0,
          o.halfSize[0],
          o.halfSize[1],
          o.cornerRadius ?? 0,
          0,
        );
      }
    }
    return data;
  }

  function packPointers(): number[] {
    const data: number[] = [];
    for (const p of params.pointers.slice(0, MAX_POINTERS)) {
      data.push(p.x, p.y, p.vx, p.vy, p.radius, p.strength, p.mode === 'pull' ? 1 : 0, 0);
    }
    return data;
  }

  /** Writes the uniforms for the next steps; returns the buffer. */
  function upload(dt: number): GPUBuffer {
    const d = derived;
    const uniforms = uniformRing[ringIndex]!;
    ringIndex = (ringIndex + 1) % UNIFORM_RING;
    const obstacles = params.obstacles.slice(0, MAX_OBSTACLES);
    const pointers = params.pointers.slice(0, MAX_POINTERS);
    const gamma = 7;
    uniforms.set({
      gravity: gravityAt(simTime),
      dt,
      h: d.h,
      box_min: [bounds.min[0], bounds.min[1]],
      box_max: [bounds.max[0], bounds.max[1]],
      mass: d.mass,
      rest_density: d.restDensity,
      tait_b: (d.restDensity * d.soundSpeed * d.soundSpeed) / gamma,
      gamma,
      viscosity: d.viscosity,
      artificial_viscosity: params.artificialViscosity,
      cohesion: (params.surfaceTension * (spacing * spacing)) / (d.h * d.h),
      damping: params.damping,
      speed_max: d.speedMax,
      margin: 0.12 * spacing,
      restitution: params.restitution,
      friction: params.friction,
      sound_speed: d.soundSpeed,
      pressure_cap: PRESSURE_CAP,
      pressure_ref: d.pressureRef,
      count: system.count,
      scalar_mode: SCALAR_CODE[params.scalar],
      obstacle_count: obstacles.length,
      pointer_count: pointers.length,
      obstacles: packObstacles(),
      pointers: packPointers(),
      wall_gamma: gammaTable,
      wall_force: forceTable,
    });
    uniforms.upload();
    return uniforms.buffer;
  }

  function run(
    encoder: GPUCommandEncoder,
    steps: number,
    dt: number,
    stages: SphStages = {},
  ): void {
    const count = system.count;
    if (count <= 0 || steps <= 0) return;
    const d = derived;
    const uniform = upload(dt);
    grid.configure({
      min: [bounds.min[0], bounds.min[1], 0],
      max: [bounds.max[0], bounds.max[1], 0],
      radius: d.h + d.skin,
    });
    // The step cannot move a particle further than `speedMax * dt`: if the velocity limit was lowered below what
    // the particles have, the first step clamps them, and the skin below still holds after that one step.
    for (let s = 0; s < steps; s++) {
      timed = s === 0 || s === steps - 1;
      const build = stages.grid ?? sinceBuild >= d.rebuildEvery;
      if (build) {
        grid.build(encoder, system);
        sinceBuild = 0;
        stats.builds++;
      }
      const gridResources = grid.resources();
      if (stages.density !== false) {
        densityPass.runCount(encoder, [uniform, system.positions, state, ...gridResources], count);
      }
      if (stages.forces !== false) {
        forcePass.runCount(
          encoder,
          [
            uniform,
            system.positions,
            system.velocities,
            state,
            system.back('position'),
            system.back('velocity'),
            ...gridResources,
          ],
          count,
        );
        system.swap('position');
        system.swap('velocity');
      }
      sinceBuild++;
      stats.steps++;
    }
    timed = true;
    simTime += steps * dt;
  }

  // --- Structural changes ------------------------------------------------------------------------

  function fluidFromBlocks(blocks: readonly SphBlock[]): {
    positions: Float32Array;
    velocities: Float32Array;
  } {
    const obstacles = params.obstacles;
    const rows: number[] = [];
    const vel: number[] = [];
    for (const block of blocks) {
      for (const [x, y] of blockPoints(block, spacing, obstacles)) {
        if (rows.length / 4 >= capacity) break;
        rows.push(x, y, 0, 1);
        vel.push(block.velocity?.[0] ?? 0, block.velocity?.[1] ?? 0, 0, 0);
      }
    }
    return { positions: new Float32Array(rows), velocities: new Float32Array(vel) };
  }

  function appendRows(positions: Float32Array, velocities: Float32Array): void {
    const n = Math.min(positions.length / 4, capacity - system.count);
    if (n <= 0) return;
    const start = system.count;
    system.setCount(start + n);
    device.queue.writeBuffer(system.attribute('position'), start * 16, positions, 0, n * 4);
    device.queue.writeBuffer(system.attribute('velocity'), start * 16, velocities, 0, n * 4);
    sinceBuild = Number.POSITIVE_INFINITY;
  }

  function applyReset(next: SphScene): void {
    scene = { ...scene, ...next };
    if (next.bounds) bounds = { min: [...next.bounds.min], max: [...next.bounds.max] };
    if (next.spacing !== undefined && next.spacing > 0) {
      spacing = next.spacing;
      system.radius = 0.5 * spacing;
    }
    if (next.obstacles) params.obstacles = next.obstacles.slice(0, MAX_OBSTACLES);
    if (next.gravity) params.gravity = [...next.gravity];
    emitter = next.emitter === undefined ? emitter : next.emitter;
    emitCarry = 0;
    refreshDerived();
    const fromBlocks = fluidFromBlocks(scene.blocks ?? []);
    let { positions, velocities } = fromBlocks;
    if (scene.particles) {
      const extra = scene.particles;
      const room = Math.max(0, capacity - positions.length / 4);
      const n = Math.min(extra.positions.length / 4, room);
      const joinedPositions = new Float32Array(positions.length + n * 4);
      const joinedVelocities = new Float32Array(positions.length + n * 4);
      joinedPositions.set(positions);
      joinedVelocities.set(velocities);
      joinedPositions.set(extra.positions.subarray(0, n * 4), positions.length);
      if (extra.velocities)
        joinedVelocities.set(extra.velocities.subarray(0, n * 4), positions.length);
      positions = joinedPositions;
      velocities = joinedVelocities;
    }
    system.set({ positions, velocities });
    simTime = 0;
    sinceBuild = Number.POSITIVE_INFINITY;
  }

  function emitRows(delta: number): void {
    if (!emitter || emitter.speed <= 0 || emitter.width <= 0) return;
    const { dx, dy } = hexSpacing(spacing);
    const dir = Math.hypot(emitter.direction[0], emitter.direction[1]) || 1;
    const ux = emitter.direction[0] / dir;
    const uy = emitter.direction[1] / dir;
    const nx = -uy;
    const ny = ux;
    emitCarry += (emitter.speed * delta) / dy;
    const rows = Math.floor(emitCarry);
    if (rows <= 0) return;
    emitCarry -= rows;
    const perRow = Math.max(1, Math.floor(emitter.width / dx));
    const positions: number[] = [];
    const velocities: number[] = [];
    for (let r = 0; r < rows; r++) {
      // Row r left the nozzle (rows - 1 - r) row-times ago: it has travelled that far already.
      const advance = (rows - 1 - r + emitCarry) * dy;
      for (let c = 0; c < perRow; c++) {
        const across = (c - (perRow - 1) / 2 + (r & 1 ? 0.5 : 0)) * dx;
        positions.push(
          emitter.position[0] + nx * across + ux * advance,
          emitter.position[1] + ny * across + uy * advance,
          0,
          1,
        );
        velocities.push(ux * emitter.speed, uy * emitter.speed, 0, 0);
      }
    }
    appendRows(new Float32Array(positions), new Float32Array(velocities));
  }

  function flush(delta: number): void {
    while (pending.length > 0) pending.shift()!();
    emitRows(delta);
  }

  function reallocateState(): void {
    state.destroy();
    state = device.createBuffer({
      label: `${label} state`,
      size: capacity * 8,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });
  }

  function keepStepper(dt: number): FixedStepper {
    const key = `${dt}|${derived.subSteps}`;
    if (!stepper || key !== stepperKey) {
      stepper = createFixedStepper(dt, { maxSteps: Math.max(2 * derived.subSteps, 4) });
      stepperKey = key;
    }
    return stepper;
  }

  refreshDerived();

  const solver: SphSolver = {
    system,
    particles,
    params,
    grid,
    get derived() {
      return derived;
    },
    get bounds() {
      return bounds;
    },
    get count() {
      return system.count;
    },
    get capacity() {
      return capacity;
    },
    get time() {
      return simTime;
    },
    stats,
    get generation() {
      return generation + system.generation;
    },

    update(encoder, delta) {
      flush(Math.min(Math.max(delta, 0), 0.1));
      const d = refreshDerived();
      const { steps } = keepStepper(d.dt).advance(Math.min(Math.max(delta, 0), 0.1));
      run(encoder, steps, d.dt);
      return steps;
    },

    step(encoder, steps = 1, dt, stages) {
      flush(0);
      const d = refreshDerived();
      run(encoder, steps, dt ?? d.dt, stages);
    },

    reset(next = {}) {
      pending.push(() => applyReset(next));
    },

    addBlock(block) {
      pending.push(() => {
        const { positions, velocities } = fluidFromBlocks([block]);
        appendRows(positions, velocities);
      });
    },

    setEmitter(next) {
      emitter = next;
      emitCarry = 0;
    },

    setCapacity(next) {
      next = Math.max(1, Math.floor(next));
      if (next === capacity) return;
      pending.push(() => {
        capacity = next;
        system.resize(next);
        reallocateState();
        generation++;
        sinceBuild = Number.POSITIVE_INFINITY;
      });
    },

    async read() {
      while (pending.length > 0) pending.shift()!();
      const count = system.count;
      const positions = await system.read('position', count);
      const velocities = await system.read('velocity', count);
      const readback = device.createBuffer({
        size: Math.max(8, count * 8),
        usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
      });
      const encoder = device.createCommandEncoder();
      if (count > 0) encoder.copyBufferToBuffer(state, 0, readback, 0, count * 8);
      device.queue.submit([encoder.finish()]);
      await readback.mapAsync(GPUMapMode.READ);
      const sph = new Float32Array(readback.getMappedRange().slice(0, count * 8));
      readback.unmap();
      readback.destroy();
      const out: SphSnapshot = {
        count,
        x: new Float32Array(count),
        y: new Float32Array(count),
        vx: new Float32Array(count),
        vy: new Float32Array(count),
        density: new Float32Array(count),
        pressure: new Float32Array(count),
        scalar: new Float32Array(count),
      };
      for (let i = 0; i < count; i++) {
        out.x[i] = positions[i * 4]!;
        out.y[i] = positions[i * 4 + 1]!;
        out.scalar[i] = positions[i * 4 + 3]!;
        out.vx[i] = velocities[i * 4]!;
        out.vy[i] = velocities[i * 4 + 1]!;
        out.density[i] = sph[i * 2]!;
        out.pressure[i] = sph[i * 2 + 1]!;
      }
      return out;
    },

    destroy() {
      if (destroyed) return;
      destroyed = true;
      densityPass.destroy();
      forcePass.destroy();
      grid.destroy();
      system.destroy();
      state.destroy();
      for (const u of uniformRing) u.destroy();
    },
  };
  return solver;
}
