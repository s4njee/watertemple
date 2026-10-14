import {
  createNeighborGrid,
  createParticleSystem,
  createRandom,
  createComputePass,
  type ComputePass,
  type NeighborGrid,
  type ParticleSet,
  type ParticleSystem,
} from '../particles/index.ts';
import { noProfiler, type PassProfiler } from '../profiler.ts';
import {
  defineUniformStruct,
  uniformArray,
  type UniformFields,
  type UniformStruct,
} from '../uniforms.ts';
import coreSource from './pbf-core.wgsl';
import correctSource from './pbf-correct.wgsl';
import densitySource from './pbf-density.wgsl';
import kernelsSource from './pbf-kernels.wgsl';
import predictSource from './pbf-predict.wgsl';
import viscositySource from './pbf-viscosity.wgsl';
import vorticitySource from './pbf-vorticity.wgsl';

// Position-based fluids (story 7.4): Macklin and Mueller, "Position Based Fluids" (2013), on the GPU, in 2D or 3D from
// the same code. The method treats incompressibility as a constraint on particle positions instead of a force:
//
//   density constraint   C_i = rho_i / rho_0 - 1 = 0           rho_i = sum_j W(|x_i - x_j|, h)
//   one Newton step      dp_i = 1/rho_0 sum_j (lambda_i + lambda_j + s_corr) grad W(x_i - x_j)
//                        lambda_i = -C_i / (sum_k |grad_k C_i|^2 + eps)
//
// and because the correction is a position change, a large time step cannot blow the fluid up: the worst that can
// happen is that it is not corrected enough (it compresses, it looks like jelly), never that it explodes. That is why
// real-time particle water uses it. One SUB-STEP is
//
//   1. predict      v += g dt (+ pointer impulses);  x* = x + v dt, kept inside the container   pbf-predict.wgsl
//   2. grid         counting sort of the PREDICTED positions, every attribute reordered to cell order  (particles/)
//   3. iterate N times, two parallel passes each:
//        density    rho_i, C_i and lambda_i for every particle                               pbf-density.wgsl
//        correct    dp_i from the neighbours' lambda (+ artificial pressure), x* += dp, container, velocity
//                   v = (x* - x) / dt, foam                                                   pbf-correct.wgsl
//   4. viscosity    XSPH smoothing of v; vorticity of v                                      pbf-viscosity.wgsl
//   5. vorticity    confinement force on v (optional)                                        pbf-vorticity.wgsl
//
// An update() is `steps` fixed steps (timeStep, default 1/60 s, capped) of `substeps` sub-steps each.
//
// JACOBI, NOT GAUSS-SEIDEL. The CPU algorithm of the paper updates particles one after another, so each sees its
// neighbours' latest corrections. A GPU cannot, so every particle here reads the positions of the previous iterate and
// all corrections are applied at once. Two things follow. (1) Convergence is slower per iteration: information moves
// one kernel radius per iteration, so a column N kernel radii tall needs about N iterations to carry its weight to the
// bottom, and the deep part stays compressed with fewer (measured in tests/pbf.test.ts and the notes). (2) Summing
// the corrections of about 30 constraints over-corrects, so a plain Jacobi step is unstable unless it is damped: the
// relaxation `eps` in the denominator of lambda (as in the paper) does most of it, and `relax` (an under-relaxation
// factor on dp) is the second knob. Neither is the "averaging" of PBD solvers; the position change is not divided by the
// number of constraints touching a particle, which would make the solver converge far more slowly.
//
// CONSISTENT BY CONSTRUCTION. Everything is derived from one length, the particle `spacing` s, and the kernel
// radius h = kernelScale * s: the rest density is the sum of the kernel over a cubic lattice of spacing s (so a lattice
// at rest has C = 0 by definition), the relaxation and the artificial pressure are expressed in units of the
// gradient sum of that lattice (so `relaxation` 0.1 means the same at any resolution, in 2D and 3D), the viscosity is
// a fraction per 1/60 s, vorticity confinement scales with h. Halve the spacing and the same parameters give the same
// fluid; only the particle count (x8 in 3D, x4 in 2D) and the cost change. `pbfConstants` returns the numbers.
//
// BOUNDARIES AND TUNNELLING. The container is an axis-aligned box whose walls can move (`setBounds`: the walls head
// for the new bounds at no more than `maxWallSpeed`, by default one spacing per fixed step, so a wave-maker drives the
// water rather than teleporting through it). The walls are a clamp of the predicted and of the corrected position, so
// nothing can end a step outside the box whatever the speed or the wall does. Colliders (spheres, boxes) are projected
// out of along the shortest way; a particle can only tunnel through one if it moves more than half the collider's
// thickness in one sub-step, and the speed limit (`maxSpeed`, default 1.5 kernel radii per sub-step = 15.8 m/s at
// 1/240 s and h = 0.044 m) bounds that, so colliders thicker than 3 h are safe at the largest step allowed. A wall
// also removes density: the fluid that would be behind it is added analytically (pbf-core.wgsl), which is what lets
// the solver rest against a wall without the layering of particles stacked on a projection plane.
//
// MOMENTUM. The position correction and the XSPH smoothing are sums of antisymmetric pair terms, so with no walls, no
// gravity and no vorticity the total linear momentum is conserved up to rounding. Two small leaks: the grid is built
// once per sub-step from the predicted positions, and corrections move particles a little before the last iteration, so
// a pair just under a kernel radius apart can be missed by one side of the search (its kernel weight is almost zero);
// and the correction cap and the speed limit clip extreme cases. Vorticity confinement, damping, the walls and gravity
// are the things that put momentum in or take it out.
//
// UNITS. Metres and seconds, gravity -9.81 along y by default; nothing in the solver depends on that, a game can
// work in any unit. The 2D solver is a slice of unit depth: z is 0, kernels are 2D (area), a "volume" is an area.

export type Vec3 = readonly [number, number, number];

// --- Kernels, the rest state and the wall tables (CPU) --------------------------------------------------------------

/** Poly6 normalisation: 315 / (64 pi h^9) in 3D, 4 / (pi h^8) in 2D. W(r) = c (h^2 - r^2)^3 integrates to 1. */
export function poly6Coefficient(dimension: 2 | 3, h: number): number {
  return dimension === 3 ? 315 / (64 * Math.PI * h ** 9) : 4 / (Math.PI * h ** 8);
}

/** Spiky normalisation: 15 / (pi h^6) in 3D, 10 / (pi h^5) in 2D. W(r) = c (h - r)^3 integrates to 1. */
export function spikyCoefficient(dimension: 2 | 3, h: number): number {
  return dimension === 3 ? 15 / (Math.PI * h ** 6) : 10 / (Math.PI * h ** 5);
}

export function poly6(dimension: 2 | 3, h: number, r: number): number {
  return r < h ? poly6Coefficient(dimension, h) * (h * h - r * r) ** 3 : 0;
}

export function spiky(dimension: 2 | 3, h: number, r: number): number {
  return r < h ? spikyCoefficient(dimension, h) * (h - r) ** 3 : 0;
}

/** dW/dr of the Spiky kernel (negative inside the radius): the gradient is this times the unit vector from j to i. */
export function spikyDerivative(dimension: 2 | 3, h: number, r: number): number {
  return r < h ? -3 * spikyCoefficient(dimension, h) * (h - r) ** 2 : 0;
}

/** Volume (area in 2D) of the unit ball of the dimension times h^D: what a kernel integral should come to for W = 1. */
export function ballVolume(dimension: 2 | 3, h: number): number {
  return dimension === 3 ? (4 / 3) * Math.PI * h ** 3 : Math.PI * h * h;
}

/** Simpson's rule on n (even) intervals. */
function simpson(f: (x: number) => number, a: number, b: number, n = 200): number {
  const step = (b - a) / n;
  let sum = f(a) + f(b);
  for (let i = 1; i < n; i++) sum += f(a + i * step) * (i % 2 === 0 ? 2 : 4);
  return (sum * step) / 3;
}

/** Integral of a radial kernel over its whole support, by quadrature in r. 1 for every kernel above. */
export function kernelIntegral(
  dimension: 2 | 3,
  h: number,
  kernel: (dimension: 2 | 3, h: number, r: number) => number,
): number {
  const shell = (r: number) => (dimension === 3 ? 4 * Math.PI * r * r : 2 * Math.PI * r);
  return simpson((r) => shell(r) * kernel(dimension, h, r), 0, h, 400);
}

export interface RestState {
  /** Sum of Poly6 over a cubic lattice of the spacing, the particle itself included: the rest number density. */
  n0: number;
  /** The lattice's sum of |grad W|^2 over n0^2: the denominator of lambda for a particle at rest. */
  s0: number;
  /** Lattice neighbours inside the kernel radius (the particle itself excluded). */
  neighbours: number;
  /** Largest component of the vector sum of the lattice gradients: zero for a symmetric lattice. */
  asymmetry: number;
}

/** The rest state of the cubic lattice with the given spacing, for the kernel radius h. */
export function restState(dimension: 2 | 3, spacing: number, h: number): RestState {
  const reach = Math.ceil(h / spacing);
  let n0 = 0;
  let neighbours = 0;
  let squares = 0;
  const sum = [0, 0, 0];
  const zr = dimension === 3 ? reach : 0;
  for (let a = -reach; a <= reach; a++) {
    for (let b = -reach; b <= reach; b++) {
      for (let c = -zr; c <= zr; c++) {
        const r = Math.hypot(a, b, c) * spacing;
        if (r >= h) continue;
        n0 += poly6(dimension, h, r);
        if (r === 0) continue;
        neighbours++;
        const slope = spikyDerivative(dimension, h, r) / r;
        squares += (slope * r) ** 2;
        sum[0]! += slope * a * spacing;
        sum[1]! += slope * b * spacing;
        sum[2]! += slope * c * spacing;
      }
    }
  }
  return {
    n0,
    s0: squares / (n0 * n0),
    neighbours,
    asymmetry: Math.max(Math.abs(sum[0]!), Math.abs(sum[1]!), Math.abs(sum[2]!)),
  };
}

/** Samples in a wall table: d = 0 .. h in 32 steps. */
export const WALL_TABLE_SIZE = 33;

/**
 * The two tables the analytic wall density uses (pbf-core.wgsl). For a particle at distance d from a flat wall,
 * the fluid that would fill the space behind the wall is a half-space of lattice density 1 / s^D, and
 *   density[k]  = (1 / s^D) * (the part of the Poly6 kernel inside that half-space, 0 .. 1/2)
 *   gradient[k] = (1 / s^D) * (the Spiky kernel integrated over the plane at distance d)
 * the first being the density the wall hides and the second the size of its gradient with respect to the
 * particle's distance (the derivative of the first by d, with Spiky in place of Poly6, as the solver pairs them).
 */
export function wallTables(
  dimension: 2 | 3,
  spacing: number,
  h: number,
): { density: Float32Array; gradient: Float32Array } {
  // The integral of a kernel over the plane at distance z from its centre.
  const plane = (kernel: (dimension: 2 | 3, h: number, r: number) => number, z: number) => {
    if (z >= h) return 0;
    const reach = Math.sqrt(h * h - z * z);
    return dimension === 3
      ? simpson((rho) => 2 * Math.PI * rho * kernel(3, h, Math.hypot(rho, z)), 0, reach, 64)
      : simpson((y) => 2 * kernel(2, h, Math.hypot(y, z)), 0, reach, 64);
  };
  const lattice = 1 / spacing ** dimension;
  const density = new Float32Array(WALL_TABLE_SIZE);
  const gradient = new Float32Array(WALL_TABLE_SIZE);
  const cell = h / (WALL_TABLE_SIZE - 1);
  let beyond = 0;
  for (let k = WALL_TABLE_SIZE - 1; k >= 0; k--) {
    if (k < WALL_TABLE_SIZE - 1) {
      beyond += simpson((z) => plane(poly6, z), k * cell, (k + 1) * cell, 8);
    }
    density[k] = beyond * lattice;
    gradient[k] = plane(spiky, k * cell) * lattice;
  }
  return { density, gradient };
}

// --- Public types -----------------------------------------------------------------------------------------------------

export interface PbfSphere {
  kind: 'sphere';
  center: Vec3;
  radius: number;
}
export interface PbfBox {
  kind: 'box';
  min: Vec3;
  max: Vec3;
}
/** A static obstacle inside the container. At most MAX_COLLIDERS; thinner than 2 * h and a fast particle can pass through. */
export type PbfCollider = PbfSphere | PbfBox;

/** A block of liquid, filled with a lattice of the particle spacing. */
export interface PbfBlock {
  min: Vec3;
  max: Vec3;
  /** Initial velocity. */
  velocity?: Vec3;
  /** Random offset of every particle, in spacings (each axis +-jitter / 2). Default 0.05. 0 gives a perfect lattice. */
  jitter?: number;
}

/** A nozzle: a jet of the particle spacing's lattice, as wide as `radius`, leaving at `velocity`. */
export interface PbfEmitter {
  position: Vec3;
  velocity: Vec3;
  radius: number;
}

export interface PbfImpulse {
  position: Vec3;
  radius: number;
  /** Pull the velocity of the particles inside towards this (a pointer dragged through the water). Default 0. */
  velocity?: Vec3;
  /** How much of the way, 0 to 1 (at the centre of the impulse). Default 0.5. */
  strength?: number;
  /** A radial speed added away from the position (a splash), in world units per second. Default 0. */
  outward?: number;
}

export interface PbfBounds {
  min: Vec3;
  max: Vec3;
}

/** What `reset` sets up. Omitted fields mean empty / unchanged-from-creation. */
export interface PbfScene {
  blocks?: readonly PbfBlock[];
  colliders?: readonly PbfCollider[];
  bounds?: PbfBounds;
  emitter?: PbfEmitter | null;
  gravity?: Vec3;
}

/** Live parameters: change any of them at any time. */
export interface PbfParams {
  /** Acceleration, world units per second squared. Default (0, -9.81, 0). */
  gravity: [number, number, number];
  /** Fixed simulation step, seconds. The fluid looks the same at any display rate. Default 1/60. */
  timeStep: number;
  /** Sub-steps per step (each rebuilds the grid): halves dt, which is the best way to cut compression. Default 2. */
  substeps: number;
  /** Solver iterations per sub-step; each is two passes over the particles. Default 4. */
  iterations: number;
  /** Most steps one update runs: what a slow frame is allowed to catch up. The rest of the backlog is dropped. Default 2. */
  maxSteps: number;
  /** Relaxation eps, in units of the rest lattice gradient sum (at least 0.001, which is the raw Newton step (unstable), 1 halves it. Default 1. */
  relaxation: number;
  /** Under-relaxation of the position correction, 0.1 to 1. Default 1. */
  relax: number;
  /** k of the artificial pressure, in units of lambda at C = 1 (so 0.1 is a repulsion worth 10 % compression). Default 0.06. 0 to 0.3: beyond about 0.4 the fluid is too stiff for the Jacobi step and a column rings itself apart. */
  artificialPressure: number;
  /** n of the artificial pressure, the exponent, a whole number from 1 to 12. Default 4. */
  artificialPressureExponent: number;
  /** dq of the artificial pressure as a fraction of h (the distance at which the term is k). Default 0.2. */
  artificialPressureRadius: number;
  /**
   * How much of the artificial pressure of a lattice at rest spacing is taken away, 0 to 1: at 1 only pairs closer than the rest
   * spacing repel, so a free blob or jet keeps its size; at 0 every pair inside dq..h repels, which calms the bottom of a deep
   * column and swells a free blob. Default 0.
   */
  artificialPressureRest: number;
  /**
   * The artificial pressure fades in with compression: it is zero where the density is at rest and full where the
   * constraint is violated by this much (a fraction, 0.02 = 2 %). Without it a free blob swells and a jet spreads, because
   * the repulsion has nothing to balance it. The price is a noisier pool at rest (the repulsion is what calms the bottom of a column), so 0, which turns the gate off, is the default; try 0.02 for scenes of falling blobs and jets.
   */
  artificialPressureGate: number;
  /** How much of a negative constraint (a density below rest) acts: 0 never pulls particles together, 1 is the paper's. Default 0. */
  tension: number;
  /** XSPH viscosity, the fraction of the velocity difference to the neighbours' mean removed per 1/60 s. 0 to 0.5. Default 0.01. */
  viscosity: number;
  /** Vorticity confinement, 0 to 1 (0.05 to 0.3 is useful). Default 0. */
  vorticity: number;
  /** Velocity damping, 1/s. Default 0. */
  damping: number;
  /** Coulomb friction of the walls and colliders, see pbf-core.wgsl. Default 0.1. */
  friction: number;
  /** Account for the fluid hidden behind the walls in the density (analytic, see pbf-core.wgsl). Default true. */
  wallDensity: boolean;
  /** How hard the walls push back relative to a fluid neighbour (1 would be the plain Newton step, 2 a mirror). Default 2. */
  wallStiffness: number;
  /** Closest distance of a particle centre to a wall, world units. Default spacing / 2. */
  wallOffset: number;
  /** Speed limit, world units per second. 0 picks 1.5 h per sub-step. Default 0. */
  maxSpeed: number;
  /** The container may move at most this fast, world units per second. 0 picks one spacing per step. Default 0. */
  maxWallSpeed: number;
  /** Speed at which fast, sparse particles start to read as foam (positions.w). Default 1. */
  foamSpeed: number;
  /** Foam decay, 1/s. Default 1.5. */
  foamDecay: number;
}

export interface PbfOptions {
  device: GPUDevice;
  /** 2 solves a slice of unit depth with 2D kernels; 3 is the default. */
  dimension?: 2 | 3;
  /** Most particles. `setCapacity` grows it later. */
  capacity: number;
  /** Distance between particles at rest, world units: the one length everything is derived from. */
  spacing: number;
  /** The container (and the domain of the neighbour grid). In 2D only x and y matter. */
  bounds: PbfBounds;
  /** Kernel radius over the spacing. Default 2.2 (3D, 32 lattice neighbours) and 3 (2D, 24). */
  kernelScale?: number;
  /** Rendered particle radius over the spacing. Default 0.65. */
  renderScale?: number;
  params?: Partial<PbfParams>;
  /** At most this many passes of an update get timestamps (the profiler has 32 slots per frame). Default 20. */
  timedPasses?: number;
  profiler?: PassProfiler;
  label?: string;
}

/** The solver's stages, for cost studies (`run`). */
export type PbfStage = 'predict' | 'grid' | 'density' | 'correct' | 'viscosity' | 'vorticity';
export const PBF_STAGES: readonly PbfStage[] = [
  'predict',
  'grid',
  'density',
  'correct',
  'viscosity',
  'vorticity',
];

export interface PbfConstants {
  dimension: 2 | 3;
  spacing: number;
  /** Kernel radius. */
  h: number;
  /** Rest number density (lattice sum). */
  n0: number;
  /** Gradient sum of the rest lattice. */
  s0: number;
  /** Lattice neighbours inside h. */
  neighbours: number;
  /** Volume (area in 2D) one particle stands for: spacing ^ dimension. */
  restVolume: number;
  /** Radius the particles are rendered with. */
  renderRadius: number;
  poly6: number;
  spiky: number;
}

export interface PbfUpdate {
  /** Fixed steps run by this call. */
  steps: number;
  /** Sub-steps run (steps * substeps). */
  substeps: number;
  /** Seconds simulated. */
  seconds: number;
}

export interface Pbf {
  readonly dimension: 2 | 3;
  /** The particles: `position` (xyz, foam 0..1), `velocity`, and the solver's own attributes. Reordered every sub-step. */
  readonly system: ParticleSystem;
  /**
   * What a renderer draws. `positions.w` is a FOAM amount in 0..1: fast particles with few neighbours (spray, splashes,
   * thin sheets) approach 1, calm water is 0. `radius` is the render radius, `restVolume` the volume a particle stands
   * for. Always points at the current buffers.
   */
  readonly particles: ParticleSet;
  /** Live parameters. */
  readonly params: PbfParams;
  readonly constants: PbfConstants;
  /** The neighbour grid, as left by the last sub-step (for neighbour-count views and the like). */
  readonly grid: NeighborGrid;
  /**
   * `array<vec2f>`, per particle in the current (cell) order: the Lagrange multiplier and the density ratio
   * rho / rho_0 of the last solver iteration (1 at rest, below 1 at a free surface, above 1 where compressed).
   * STORAGE | COPY_SRC. Re-created on a capacity change (watch `generation`).
   */
  readonly solve: GPUBuffer;
  /** Reads `solve` back (waits for the GPU): for tests and tools. */
  readState(): Promise<{ lambda: Float32Array; density: Float32Array }>;
  readonly count: number;
  readonly capacity: number;
  /** Bumps when buffers are re-created. */
  readonly generation: number;
  /** The container as it is now (it follows `setBounds` at the limited wall speed). */
  readonly bounds: PbfBounds;
  /** Seconds simulated since the last reset. */
  readonly time: number;
  /** Advances by `delta` seconds of real time in fixed steps. One call per submit. */
  update(encoder: GPUCommandEncoder, delta: number): PbfUpdate;
  /** Runs exactly one sub-step of `dt` seconds (default timeStep / substeps), or only the given stages of it, for studies. */
  run(
    encoder: GPUCommandEncoder,
    options?: { dt?: number; stages?: readonly PbfStage[]; iterations?: number },
  ): void;
  /** Moves the container (the wave-maker): the walls head for these bounds at no more than `maxWallSpeed`. */
  setBounds(min: Vec3, max: Vec3, options?: { snap?: boolean }): void;
  setColliders(colliders: readonly PbfCollider[]): void;
  /** Queues a push for the next update (at most 4 per update; the rest are dropped). */
  impulse(impulse: PbfImpulse): void;
  /** Starts (or, with null, stops) a pour. */
  setEmitter(emitter: PbfEmitter | null): void;
  /** Adds a block of liquid now; returns how many particles fitted. */
  addBlock(block: PbfBlock): number;
  /** Removes everything and sets the scene up. */
  reset(scene?: PbfScene): void;
  /** Grows or shrinks the buffers, keeping the particles that fit. */
  setCapacity(capacity: number): void;
  destroy(): void;
}

// --- Defaults ---------------------------------------------------------------------------------------------------------

export const DEFAULT_PBF_PARAMS: Readonly<PbfParams> = {
  gravity: [0, -9.81, 0],
  timeStep: 1 / 60,
  substeps: 3,
  iterations: 2,
  maxSteps: 2,
  relaxation: 1,
  relax: 1,
  artificialPressure: 0.06,
  artificialPressureExponent: 4,
  artificialPressureRadius: 0.2,
  artificialPressureRest: 0,
  artificialPressureGate: 0,
  tension: 0,
  viscosity: 0.01,
  vorticity: 0,
  damping: 0,
  friction: 0.1,
  wallDensity: true,
  wallStiffness: 2,
  wallOffset: 0,
  maxSpeed: 0,
  maxWallSpeed: 0,
  foamSpeed: 1,
  foamDecay: 1.5,
};

export const MAX_COLLIDERS = 8;
export const MAX_IMPULSES = 4;
/** The most sub-steps one update may run. */
export const MAX_SUBSTEPS_PER_UPDATE = 48;

// --- Uniforms ---------------------------------------------------------------------------------------------------------

/** One block per update. */
export const PBF_PARAMS = defineUniformStruct('PbfParams', {
  gravity: 'vec3f',
  h: 'f32',
  h2: 'f32',
  poly6: 'f32',
  spiky: 'f32',
  inv_n0: 'f32',
  inv_w_dq: 'f32',
  scorr_k: 'f32',
  scorr_n: 'u32',
  scorr_floor: 'f32',
  gate_scale: 'f32',
  eps: 'f32',
  relax: 'f32',
  neg_pressure: 'f32',
  wall_k: 'f32',
  wall_offset: 'f32',
  friction: 'f32',
  max_speed: 'f32',
  max_correction: 'f32',
  damping: 'f32',
  viscosity: 'f32',
  vorticity: 'f32',
  foam_speed: 'f32',
  foam_decay: 'f32',
  count: 'u32',
  dimension: 'u32',
  boundary_density: 'u32',
  collider_count: 'u32',
  impulse_count: 'u32',
  colliders: uniformArray('vec4f', MAX_COLLIDERS * 2),
  impulses: uniformArray('vec4f', MAX_IMPULSES * 3),
  wall_density: uniformArray('vec4f', 9),
  wall_gradient: uniformArray('vec4f', 9),
});

/** One block per sub-step. */
export const PBF_STEP = defineUniformStruct('PbfStep', {
  bounds_min: 'vec3f',
  dt: 'f32',
  bounds_max: 'vec3f',
  inv_dt: 'f32',
  first: 'u32',
});

/** Writes the fields of a uniform struct (f32, u32, vec3f and arrays of vec4f only) into an ArrayBuffer. */
function packUniform<F extends UniformFields>(
  struct: UniformStruct<F>,
  values: Readonly<Record<string, number | ArrayLike<number>>>,
  f32: Float32Array,
  u32: Uint32Array,
  byteOffset: number,
): void {
  for (const field of struct.layout) {
    const value = values[field.name];
    if (value === undefined) continue;
    const base = (byteOffset + field.offset) >> 2;
    const type = field.type;
    if (typeof type !== 'string') {
      const list = value as ArrayLike<number>;
      for (let i = 0; i < list.length; i++) f32[base + i] = list[i]!;
    } else if (type === 'u32') {
      u32[base] = value as number;
    } else if (type === 'f32') {
      f32[base] = value as number;
    } else if (type === 'vec3f') {
      const v = value as ArrayLike<number>;
      f32[base] = v[0]!;
      f32[base + 1] = v[1]!;
      f32[base + 2] = v[2]!;
    } else {
      throw new TypeError(`${struct.name}.${field.name}: ${type} is not packed by pbf.ts`);
    }
  }
}

// --- Constants of a configuration ---------------------------------------------------------------------------------------

export function pbfConstants(
  dimension: 2 | 3,
  spacing: number,
  kernelScale = dimension === 3 ? 2.2 : 3,
  renderScale = 0.65,
): PbfConstants {
  const h = spacing * kernelScale;
  const rest = restState(dimension, spacing, h);
  return {
    dimension,
    spacing,
    h,
    n0: rest.n0,
    s0: rest.s0,
    neighbours: rest.neighbours,
    restVolume: spacing ** dimension,
    renderRadius: spacing * renderScale,
    poly6: poly6Coefficient(dimension, h),
    spiky: 3 * spikyCoefficient(dimension, h),
  };
}

const finite = (x: number, fallback: number) => (Number.isFinite(x) ? x : fallback);
const clamp = (x: number, lo: number, hi: number) => Math.min(hi, Math.max(lo, x));

/** Bytes between two updates' blocks in the staging ring. */
const SLOT_BYTES = 8192;
const SLOTS = 128;

// --- The solver -------------------------------------------------------------------------------------------------------

export function createPbf(options: PbfOptions): Pbf {
  const { device, dimension = 3, profiler: baseProfiler = noProfiler, label = 'pbf' } = options;
  if (!(options.spacing > 0) || !Number.isFinite(options.spacing)) {
    throw new RangeError(`spacing must be positive and finite, got ${options.spacing}`);
  }
  if (!(options.capacity >= 1)) {
    throw new RangeError(`capacity must be at least 1, got ${options.capacity}`);
  }
  const constants = pbfConstants(
    dimension,
    options.spacing,
    options.kernelScale,
    options.renderScale,
  );
  const { h, spacing } = constants;
  const tables = wallTables(dimension, spacing, h);
  const params: PbfParams = {
    ...DEFAULT_PBF_PARAMS,
    ...options.params,
    gravity: [...(options.params?.gravity ?? DEFAULT_PBF_PARAMS.gravity)] as [
      number,
      number,
      number,
    ],
  };
  if (!options.params?.wallOffset) params.wallOffset = spacing / 2;

  // At most `timedPasses` passes per update get timestamps: the profiler has 32 slots a frame and the renderer
  // after the solver needs some. Counting restarts every update.
  const timedLimit = options.timedPasses ?? 20;
  let timed = 0;
  const profiler: PassProfiler = {
    pass: (name) => (timed++ < timedLimit ? baseProfiler.pass(name) : undefined),
  };

  const system = createParticleSystem({
    device,
    dimension,
    capacity: options.capacity,
    radius: constants.renderRadius,
    extras: { previous: 4 },
    profiler,
    label,
  });
  const grid: NeighborGrid = createNeighborGrid({
    device,
    dimension,
    mode: 'reorder',
    capacity: options.capacity,
    profiler,
    label: `${label} grid`,
  });

  // --- Buffers ----------------------------------------------------------------------------------------------------
  const uniformUsage = GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST;
  const paramsBuffer = device.createBuffer({
    label: `${label} params`,
    size: PBF_PARAMS.size,
    usage: uniformUsage,
  });
  const tickBuffer = device.createBuffer({
    label: `${label} tick`,
    size: PBF_STEP.size,
    usage: uniformUsage,
  });
  // Uniforms are copied into place from a ring of staging blocks as the passes are encoded, because a
  // queue.writeBuffer lands before the whole command buffer and the sub-steps of one update differ.
  const staging = device.createBuffer({
    label: `${label} staging`,
    size: SLOT_BYTES * SLOTS,
    usage: GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
  });
  const scratchBytes = new ArrayBuffer(SLOT_BYTES);
  const scratchF32 = new Float32Array(scratchBytes);
  const scratchU32 = new Uint32Array(scratchBytes);
  let slot = 0;

  const storage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST;
  let solve: GPUBuffer | undefined;
  let omega: GPUBuffer | undefined;
  let localGeneration = 0;
  function allocate(capacity: number): void {
    solve?.destroy();
    omega?.destroy();
    solve = device.createBuffer({ label: `${label} solve`, size: capacity * 8, usage: storage });
    omega = undefined;
    localGeneration++;
  }
  allocate(system.capacity);
  const omegaBuffer = (): GPUBuffer =>
    (omega ??= device.createBuffer({
      label: `${label} omega`,
      size: system.capacity * 16,
      usage: storage,
    }));

  // --- Passes -----------------------------------------------------------------------------------------------------
  const chunks = {
    'pbf_params.wgsl': PBF_PARAMS.wgsl + PBF_STEP.wgsl,
    'pbf-core.wgsl': coreSource.code,
    ...coreSource.chunks,
    'pbf-kernels.wgsl': kernelsSource.code,
    ...kernelsSource.chunks,
    ...grid.chunks({ group: 0, binding: 8 }),
  };
  const makePass = (
    file: string,
    source: { code: string; chunks: Readonly<Record<string, string>> },
    stamp: string,
  ): ComputePass =>
    createComputePass(device, {
      label: `sim/${file}`,
      code: source.code,
      chunks: { ...source.chunks, ...chunks },
      profiler,
      timestamp: `${label} ${stamp}`,
    });
  const passes = {
    predict: makePass('pbf-predict.wgsl', predictSource, 'predict'),
    density: makePass('pbf-density.wgsl', densitySource, 'density'),
    correct: makePass('pbf-correct.wgsl', correctSource, 'correct'),
    viscosity: makePass('pbf-viscosity.wgsl', viscositySource, 'viscosity'),
    vorticity: makePass('pbf-vorticity.wgsl', vorticitySource, 'vorticity'),
  };

  // --- State ------------------------------------------------------------------------------------------------------
  const initialBounds: { min: [number, number, number]; max: [number, number, number] } = {
    min: [...options.bounds.min],
    max: [...options.bounds.max],
  };
  const bounds = { min: [...initialBounds.min], max: [...initialBounds.max] } as {
    min: [number, number, number];
    max: [number, number, number];
  };
  const target = { min: [...initialBounds.min], max: [...initialBounds.max] } as typeof bounds;
  const domain = { min: [...initialBounds.min], max: [...initialBounds.max] } as typeof bounds;
  let colliders: PbfCollider[] = [];
  let emitter: PbfEmitter | null = null;
  let emitClock = 0;
  let impulses: PbfImpulse[] = [];
  let accumulator = 0;
  let time = 0;
  let seed = 1;
  let destroyed = false;
  let sliceCache: { key: string; offsets: Float32Array } | undefined;

  const wallSpeed = () =>
    params.maxWallSpeed > 0 ? params.maxWallSpeed : spacing / Math.max(params.timeStep, 1e-4);

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
    restVolume: constants.restVolume,
    get generation() {
      return system.generation + localGeneration;
    },
  };

  function sanitise(): void {
    params.timeStep = clamp(finite(params.timeStep, 1 / 60), 1 / 480, 1 / 15);
    params.substeps = Math.round(clamp(finite(params.substeps, 2), 1, 8));
    params.iterations = Math.round(clamp(finite(params.iterations, 4), 1, 32));
    params.maxSteps = Math.round(clamp(finite(params.maxSteps, 2), 1, 8));
    params.relaxation = clamp(finite(params.relaxation, 1), 1e-3, 100);
    params.relax = clamp(finite(params.relax, 1), 0.05, 1.5);
    params.artificialPressure = clamp(finite(params.artificialPressure, 0), 0, 0.3);
    params.artificialPressureExponent = clamp(finite(params.artificialPressureExponent, 4), 1, 12);
    params.artificialPressureRadius = clamp(
      finite(params.artificialPressureRadius, 0.2),
      0.02,
      0.9,
    );
    params.artificialPressureRest = clamp(finite(params.artificialPressureRest, 0.5), 0, 1);
    params.artificialPressureGate = clamp(finite(params.artificialPressureGate, 0), 0, 10);
    params.tension = clamp(finite(params.tension, 0), 0, 1);
    params.viscosity = clamp(finite(params.viscosity, 0), 0, 0.9);
    params.vorticity = clamp(finite(params.vorticity, 0), 0, 4);
    params.damping = clamp(finite(params.damping, 0), 0, 100);
    params.friction = clamp(finite(params.friction, 0), 0, 10);
    params.wallStiffness = clamp(finite(params.wallStiffness, 2), 0, 4);
    params.wallOffset = clamp(finite(params.wallOffset, spacing / 2), 0, h);
    params.foamSpeed = Math.max(finite(params.foamSpeed, 1), 1e-3);
    params.foamDecay = clamp(finite(params.foamDecay, 1.5), 0, 100);
    for (let k = 0; k < 3; k++) params.gravity[k] = clamp(finite(params.gravity[k]!, 0), -1e4, 1e4);
  }

  function colliderRows(): number[] {
    const rows: number[] = [];
    for (const c of colliders.slice(0, MAX_COLLIDERS)) {
      if (c.kind === 'sphere') {
        rows.push(...c.center, c.radius, 0, 0, 0, 1);
      } else {
        rows.push(...c.min, 0, ...c.max, 2);
      }
    }
    return rows;
  }

  function impulseRows(list: readonly PbfImpulse[]): number[] {
    const rows: number[] = [];
    for (const m of list.slice(0, MAX_IMPULSES)) {
      const v = m.velocity ?? [0, 0, 0];
      rows.push(...m.position, m.radius, ...v, m.strength ?? 0.5, m.outward ?? 0, 0, 0, 0);
    }
    return rows;
  }

  /** Writes this update's params block into the staging ring; returns its byte offset. */
  function stageParams(list: readonly PbfImpulse[]): number {
    sanitise();
    const rest = restState(dimension, spacing, h);
    const dq = params.artificialPressureRadius * h;
    const wdq = poly6(dimension, h, dq);
    const dt = params.timeStep / params.substeps;
    const maxSpeed = params.maxSpeed > 0 ? params.maxSpeed : (1.5 * h) / dt;
    const base = slot * SLOT_BYTES;
    scratchU32.fill(0);
    packUniform(
      PBF_PARAMS,
      {
        gravity: params.gravity,
        h,
        h2: h * h,
        poly6: constants.poly6,
        spiky: constants.spiky,
        inv_n0: 1 / rest.n0,
        inv_w_dq: 1 / wdq,
        scorr_k: params.artificialPressure / rest.s0,
        scorr_n: Math.round(params.artificialPressureExponent),
        gate_scale:
          params.artificialPressureGate > 0
            ? (rest.s0 * (1 + params.relaxation)) / params.artificialPressureGate
            : 0,
        scorr_floor:
          params.artificialPressureRest *
          Math.min(poly6(dimension, h, spacing) / wdq, 1) **
            Math.round(params.artificialPressureExponent),
        eps: params.relaxation * rest.s0,
        relax: params.relax,
        neg_pressure: params.tension,
        wall_k: params.wallStiffness,
        wall_offset: params.wallOffset,
        friction: params.friction,
        max_speed: maxSpeed,
        max_correction: 0.5 * h,
        damping: params.damping,
        viscosity: params.viscosity,
        vorticity: params.vorticity,
        foam_speed: params.foamSpeed,
        foam_decay: params.foamDecay,
        count: system.count,
        dimension,
        boundary_density: params.wallDensity ? 1 : 0,
        collider_count: Math.min(colliders.length, MAX_COLLIDERS),
        impulse_count: Math.min(list.length, MAX_IMPULSES),
        colliders: colliderRows(),
        impulses: impulseRows(list),
        wall_density: Array.from(tables.density),
        wall_gradient: Array.from(tables.gradient),
      },
      scratchF32,
      scratchU32,
      0,
    );
    device.queue.writeBuffer(staging, base, scratchBytes, 0, PBF_PARAMS.size);
    return base;
  }

  /** Writes the sub-step blocks of this update after the params block of `base`; returns the offset of the first. */
  function stageTicks(
    base: number,
    dt: number,
    boundsList: readonly { min: number[]; max: number[] }[],
  ): number {
    const first = base + roundUp(16, PBF_PARAMS.size);
    const blocks = new ArrayBuffer(boundsList.length * PBF_STEP.size);
    const f = new Float32Array(blocks);
    const u = new Uint32Array(blocks);
    boundsList.forEach((b, index) => {
      packUniform(
        PBF_STEP,
        {
          bounds_min: b.min,
          dt,
          bounds_max: b.max,
          inv_dt: 1 / dt,
          first: index === 0 ? 1 : 0,
        },
        f,
        u,
        index * PBF_STEP.size,
      );
    });
    device.queue.writeBuffer(staging, first, blocks);
    return first;
  }

  function configureGrid(): void {
    for (let a = 0; a < 3; a++) {
      domain.min[a] = Math.min(domain.min[a]!, bounds.min[a]!, target.min[a]!);
      domain.max[a] = Math.max(domain.max[a]!, bounds.max[a]!, target.max[a]!);
    }
    grid.configure({ min: domain.min, max: domain.max, radius: h });
  }

  // --- One sub-step -----------------------------------------------------------------------------------------------
  function encodeSubstep(
    encoder: GPUCommandEncoder,
    tickOffset: number,
    stages: ReadonlySet<PbfStage>,
    iterations: number,
  ): void {
    encoder.copyBufferToBuffer(staging, tickOffset, tickBuffer, 0, PBF_STEP.size);
    const count = system.count;
    if (count <= 0) return;
    const common = [paramsBuffer, tickBuffer] as const;

    if (stages.has('predict')) {
      passes.predict.runCount(
        encoder,
        [...common, system.positions, system.velocities, system.attribute('previous')],
        count,
      );
    }
    if (stages.has('grid')) grid.build(encoder, system);
    const gridResources = grid.resources();
    const solveBuffer = solve!;

    for (let i = 0; i < iterations; i++) {
      if (stages.has('density')) {
        passes.density.runCount(
          encoder,
          [
            ...common,
            system.positions,
            undefined,
            undefined,
            solveBuffer,
            undefined,
            undefined,
            ...gridResources,
          ],
          count,
        );
      }
      if (stages.has('correct')) {
        passes.correct.runCount(
          encoder,
          [
            ...common,
            system.positions,
            system.velocities,
            system.attribute('previous'),
            solveBuffer,
            system.back('position'),
            undefined,
            ...gridResources,
          ],
          count,
        );
        system.swap('position');
      }
    }

    const wantVorticity = params.vorticity > 0;
    if (stages.has('viscosity') && (params.viscosity > 0 || wantVorticity)) {
      passes.viscosity.runCount(
        encoder,
        [
          ...common,
          system.positions,
          system.velocities,
          undefined,
          undefined,
          omegaBuffer(),
          system.back('velocity'),
          ...gridResources,
        ],
        count,
      );
      system.swap('velocity');
    }
    if (stages.has('vorticity') && wantVorticity) {
      passes.vorticity.runCount(
        encoder,
        [
          ...common,
          system.positions,
          system.velocities,
          undefined,
          undefined,
          omegaBuffer(),
          undefined,
          ...gridResources,
        ],
        count,
      );
    }
  }

  const ALL = new Set<PbfStage>(PBF_STAGES);

  // --- Spawning ---------------------------------------------------------------------------------------------------
  function append(positions: Float32Array, velocities: Float32Array): number {
    const room = system.capacity - system.count;
    const n = Math.min(room, positions.length / 4);
    if (n <= 0) return 0;
    const start = system.count;
    system.setCount(start + n);
    device.queue.writeBuffer(system.attribute('position'), start * 16, positions, 0, n * 4);
    device.queue.writeBuffer(system.attribute('velocity'), start * 16, velocities, 0, n * 4);
    return n;
  }

  function blockRows(block: PbfBlock): { positions: Float32Array; velocities: Float32Array } {
    const jitter = block.jitter ?? 0.05;
    const velocity = block.velocity ?? [0, 0, 0];
    const counts = [0, 1, 2].map((a) =>
      a === 2 && dimension === 2
        ? 1
        : Math.max(0, Math.floor((block.max[a]! - block.min[a]!) / spacing + 1e-6)),
    );
    const total = counts[0]! * counts[1]! * counts[2]!;
    const positions = new Float32Array(total * 4);
    const velocities = new Float32Array(total * 4);
    const random = createRandom(0x9b1f + seed++);
    let n = 0;
    for (let k = 0; k < counts[2]!; k++) {
      for (let j = 0; j < counts[1]!; j++) {
        for (let i = 0; i < counts[0]!; i++) {
          const jit = () => (random() - 0.5) * jitter * spacing;
          positions[n * 4] = block.min[0]! + (i + 0.5) * spacing + jit();
          positions[n * 4 + 1] = block.min[1]! + (j + 0.5) * spacing + jit();
          positions[n * 4 + 2] = dimension === 2 ? 0 : block.min[2]! + (k + 0.5) * spacing + jit();
          velocities.set(velocity, n * 4);
          n++;
        }
      }
    }
    return { positions, velocities };
  }

  /** Offsets (in the plane across the jet) of one slice of the nozzle, as vec3 rows. */
  function slice(e: PbfEmitter): Float32Array {
    const speed = Math.hypot(...e.velocity) || 1;
    const dir = e.velocity.map((x) => x / speed) as [number, number, number];
    const key = `${e.radius}|${dir.join(',')}`;
    if (sliceCache?.key === key) return sliceCache.offsets;
    // Two unit vectors across the jet.
    let a: [number, number, number];
    let b: [number, number, number];
    if (dimension === 2) {
      a = [-dir[1], dir[0], 0];
      b = [0, 0, 0];
    } else {
      const helper: [number, number, number] = Math.abs(dir[1]) < 0.9 ? [0, 1, 0] : [1, 0, 0];
      a = [
        dir[1] * helper[2] - dir[2] * helper[1],
        dir[2] * helper[0] - dir[0] * helper[2],
        dir[0] * helper[1] - dir[1] * helper[0],
      ];
      const la = Math.hypot(...a);
      a = a.map((x) => x / la) as typeof a;
      b = [
        dir[1] * a[2] - dir[2] * a[1],
        dir[2] * a[0] - dir[0] * a[2],
        dir[0] * a[1] - dir[1] * a[0],
      ];
    }
    const rows: number[] = [];
    const m = Math.floor(e.radius / spacing);
    for (let i = -m; i <= m; i++) {
      for (let j = dimension === 2 ? 0 : -m; j <= (dimension === 2 ? 0 : m); j++) {
        if (Math.hypot(i, j) * spacing > e.radius + 1e-9) continue;
        rows.push(
          a[0] * i * spacing + b[0] * j * spacing,
          a[1] * i * spacing + b[1] * j * spacing,
          a[2] * i * spacing + b[2] * j * spacing,
        );
      }
    }
    sliceCache = { key, offsets: new Float32Array(rows) };
    return sliceCache.offsets;
  }

  /**
   * Spawns the slices of the jet that fall due in the next `seconds`. A slice due `t` seconds into the window starts
   * `t` behind the nozzle, so that after the whole window has been simulated it has flown for exactly `seconds - t`.
   */
  function emitFor(seconds: number): void {
    if (!emitter) return;
    const speed = Math.hypot(...emitter.velocity);
    if (!(speed > 1e-6)) return;
    const interval = spacing / speed;
    const offsets = slice(emitter);
    const per = offsets.length / 3;
    if (per === 0) return;
    const random = createRandom(0x51ab + seed++);
    const rows: number[] = [];
    const velocities: number[] = [];
    let t = interval - emitClock;
    while (t <= seconds + 1e-9 && rows.length / 4 + per <= system.capacity - system.count) {
      for (let k = 0; k < per; k++) {
        const jit = () => (random() - 0.5) * 0.04 * spacing;
        rows.push(
          emitter.position[0] + offsets[k * 3]! - emitter.velocity[0] * t + jit(),
          emitter.position[1] + offsets[k * 3 + 1]! - emitter.velocity[1] * t + jit(),
          dimension === 2
            ? 0
            : emitter.position[2] + offsets[k * 3 + 2]! - emitter.velocity[2] * t + jit(),
          0,
        );
        velocities.push(...emitter.velocity, 0);
      }
      t += interval;
    }
    emitClock = seconds - (t - interval);
    if (rows.length > 0) append(new Float32Array(rows), new Float32Array(velocities));
  }

  // --- Public object ----------------------------------------------------------------------------------------------
  const pbf: Pbf = {
    dimension,
    system,
    particles,
    params,
    constants,
    grid,
    get solve() {
      return solve!;
    },
    async readState() {
      const n = system.count;
      const bytes = Math.max(8, n * 8);
      const readback = device.createBuffer({
        label: `${label} readback`,
        size: bytes,
        usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
      });
      const encoder = device.createCommandEncoder();
      if (n > 0) encoder.copyBufferToBuffer(solve!, 0, readback, 0, n * 8);
      device.queue.submit([encoder.finish()]);
      await readback.mapAsync(GPUMapMode.READ);
      const data = new Float32Array(readback.getMappedRange().slice(0, n * 8));
      readback.unmap();
      readback.destroy();
      const lambda = new Float32Array(n);
      const density = new Float32Array(n);
      for (let i = 0; i < n; i++) {
        lambda[i] = data[i * 2]!;
        density[i] = data[i * 2 + 1]!;
      }
      return { lambda, density };
    },
    get count() {
      return system.count;
    },
    get capacity() {
      return system.capacity;
    },
    get generation() {
      return system.generation + localGeneration;
    },
    get bounds() {
      return bounds;
    },
    get time() {
      return time;
    },

    update(encoder, delta) {
      sanitise();
      const dt = params.timeStep;
      accumulator += clamp(finite(delta, 0), 0, 0.25);
      const whole = Math.floor(accumulator / dt + 1e-6);
      accumulator = Math.max(0, accumulator - whole * dt);
      const steps = Math.min(whole, params.maxSteps);
      const result: PbfUpdate = { steps, substeps: steps * params.substeps, seconds: steps * dt };
      timed = 0;
      if (steps === 0 || system.count === 0) {
        // Still let a pour run, a quiet scene must not hold the jet back.
        if (steps > 0) emitFor(result.seconds);
        return result;
      }

      // The container moves towards its target no faster than the wall speed, once per sub-step.
      const subDt = dt / params.substeps;
      const sequence: { min: number[]; max: number[] }[] = [];
      const limit = wallSpeed() * subDt;
      for (let s = 0; s < result.substeps; s++) {
        for (let a = 0; a < 3; a++) {
          bounds.min[a] = bounds.min[a]! + clamp(target.min[a]! - bounds.min[a]!, -limit, limit);
          bounds.max[a] = bounds.max[a]! + clamp(target.max[a]! - bounds.max[a]!, -limit, limit);
        }
        sequence.push({ min: [...bounds.min], max: [...bounds.max] });
      }
      emitFor(result.seconds);
      configureGrid();

      const used = impulses.splice(0, MAX_IMPULSES);
      const paramsOffset = stageParams(used);
      const tickOffset = stageTicks(paramsOffset, subDt, sequence);
      slot = (slot + 1) % SLOTS;
      encoder.copyBufferToBuffer(staging, paramsOffset, paramsBuffer, 0, PBF_PARAMS.size);
      for (let s = 0; s < result.substeps; s++) {
        encodeSubstep(encoder, tickOffset + s * PBF_STEP.size, ALL, params.iterations);
      }
      time += result.seconds;
      return result;
    },

    run(encoder, runOptions = {}) {
      sanitise();
      timed = 0;
      const dt = runOptions.dt ?? params.timeStep / params.substeps;
      configureGrid();
      const paramsOffset = stageParams([]);
      const tickOffset = stageTicks(paramsOffset, dt, [
        { min: [...bounds.min], max: [...bounds.max] },
      ]);
      slot = (slot + 1) % SLOTS;
      encoder.copyBufferToBuffer(staging, paramsOffset, paramsBuffer, 0, PBF_PARAMS.size);
      encodeSubstep(
        encoder,
        tickOffset,
        runOptions.stages ? new Set(runOptions.stages) : ALL,
        runOptions.iterations ?? params.iterations,
      );
    },

    setBounds(min, max, setOptions = {}) {
      for (let a = 0; a < 3; a++) {
        target.min[a] = finite(min[a]!, target.min[a]!);
        target.max[a] = finite(max[a]!, target.max[a]!);
        if (setOptions.snap) {
          bounds.min[a] = target.min[a]!;
          bounds.max[a] = target.max[a]!;
        }
      }
    },

    setColliders(list) {
      colliders = list.slice(0, MAX_COLLIDERS);
    },

    impulse(impulse) {
      if (impulses.length < MAX_IMPULSES) impulses.push(impulse);
    },

    setEmitter(next) {
      emitter = next ? { ...next } : null;
      emitClock = 0;
    },

    addBlock(block) {
      const rows = blockRows(block);
      return append(rows.positions, rows.velocities);
    },

    reset(scene = {}) {
      system.setCount(0);
      accumulator = 0;
      time = 0;
      impulses = [];
      emitClock = 0;
      emitter = scene.emitter ?? null;
      colliders = (scene.colliders ?? []).slice(0, MAX_COLLIDERS);
      const next = scene.bounds ?? initialBounds;
      for (let a = 0; a < 3; a++) {
        bounds.min[a] = target.min[a] = next.min[a]!;
        bounds.max[a] = target.max[a] = next.max[a]!;
      }
      if (scene.gravity) params.gravity = [scene.gravity[0], scene.gravity[1], scene.gravity[2]];
      for (const block of scene.blocks ?? []) pbf.addBlock(block);
    },

    setCapacity(capacity) {
      capacity = Math.max(1, Math.floor(capacity));
      if (capacity === system.capacity) return;
      system.resize(capacity);
      allocate(capacity);
    },

    destroy() {
      if (destroyed) return;
      destroyed = true;
      for (const pass of Object.values(passes)) pass.destroy();
      grid.destroy();
      system.destroy();
      paramsBuffer.destroy();
      tickBuffer.destroy();
      staging.destroy();
      solve?.destroy();
      omega?.destroy();
    },
  };
  return pbf;
}

function roundUp(multiple: number, value: number): number {
  return Math.ceil(value / multiple) * multiple;
}
