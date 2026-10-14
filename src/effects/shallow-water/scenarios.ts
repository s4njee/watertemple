import type { ShallowWater } from '../../gpu/sim/shallow-water.ts';

// The demo's scenarios: a terrain, some starting water, a few forces that act every frame, and the
// settings the solver needs for that scene. Positions are domain uv; the domain is WORLD_WIDTH x
// WORLD_DEPTH metres. Everything is plain CPU maths so the shapes can be read (and tested) without
// a GPU.

export const WORLD_WIDTH = 12;
export const WORLD_DEPTH = 8;
/** Width / depth of the grid the effect builds. */
export const WORLD_ASPECT = WORLD_WIDTH / WORLD_DEPTH;

export type ScenarioId = 'dam break' | 'hillside spring' | 'tide pool';
export const SCENARIO_IDS: readonly ScenarioId[] = ['dam break', 'hillside spring', 'tide pool'];

export interface BuiltScenario {
  /** Bed height per cell, row-major, row 0 at the top (v = 0), metres. */
  terrain: Float32Array;
  /** Water depth per cell at the start. */
  water: Float32Array;
  /** The bed after the dam goes; undefined when nothing is released. */
  released?: Float32Array;
}

export interface ScenarioInfo {
  id: ScenarioId;
  /** Solver settings the scene needs. */
  maxDepth: number;
  edges: 'wall' | 'open';
  /** Simulated seconds to hold the dam before releasing it; undefined for scenes without one. */
  releaseAfter?: number;
  /** Ground palette: how green the gentle land is, and how much of the high ground is rock. */
  palette: readonly [number, number];
  /** Camera at the start: radians around the vertical, elevation, distance in metres. */
  camera: { yaw: number; pitch: number; distance: number };
  build(width: number, height: number): BuiltScenario;
  /** Called once per frame with the simulated time since the start. */
  forces(sim: ShallowWater, time: number, strength: number): void;
}

// --- Maths ---------------------------------------------------------------------------------------

const clamp = (x: number, lo: number, hi: number) => Math.min(hi, Math.max(lo, x));
const smoothstep = (a: number, b: number, x: number) => {
  const t = clamp((x - a) / (b - a), 0, 1);
  return t * t * (3 - 2 * t);
};

function hash2(ix: number, iy: number): number {
  let h = Math.imul(ix, 374761393) + Math.imul(iy, 668265263);
  h = Math.imul(h ^ (h >>> 13), 1274126177);
  return ((h ^ (h >>> 16)) >>> 0) / 4294967296;
}

/** Smooth value noise in [0, 1]. */
export function valueNoise(x: number, y: number): number {
  const ix = Math.floor(x);
  const iy = Math.floor(y);
  const fx = x - ix;
  const fy = y - iy;
  const ux = fx * fx * (3 - 2 * fx);
  const uy = fy * fy * (3 - 2 * fy);
  const a = hash2(ix, iy);
  const b = hash2(ix + 1, iy);
  const c = hash2(ix, iy + 1);
  const d = hash2(ix + 1, iy + 1);
  return a + (b - a) * ux + (c - a) * uy + (a - b - c + d) * ux * uy;
}

export function fbm(x: number, y: number, octaves = 4): number {
  let sum = 0;
  let amplitude = 0.5;
  let norm = 0;
  for (let i = 0; i < octaves; i++) {
    sum += amplitude * valueNoise(x, y);
    norm += amplitude;
    x = x * 2.03 + 11.3;
    y = y * 2.03 + 7.1;
    amplitude *= 0.5;
  }
  return sum / norm;
}

/** Signed distance to a rounded box centred at (cx, cz) with half sizes (hx, hz). */
function roundedBox(
  x: number,
  z: number,
  cx: number,
  cz: number,
  hx: number,
  hz: number,
  r: number,
) {
  const qx = Math.abs(x - cx) - (hx - r);
  const qz = Math.abs(z - cz) - (hz - r);
  return Math.hypot(Math.max(qx, 0), Math.max(qz, 0)) + Math.min(Math.max(qx, qz), 0) - r;
}

/** 1 inside the shape (negative distance), 0 outside, with a steep but not vertical wall. */
const solid = (distance: number, wall = 0.1) => 1 - smoothstep(-wall, wall, distance);

function fill(
  width: number,
  height: number,
  f: (u: number, v: number, x: number, z: number) => number,
): Float32Array {
  const out = new Float32Array(width * height);
  for (let j = 0; j < height; j++) {
    for (let i = 0; i < width; i++) {
      const u = (i + 0.5) / width;
      const v = (j + 0.5) / height;
      out[j * width + i] = f(u, v, u * WORLD_WIDTH, v * WORLD_DEPTH);
    }
  }
  return out;
}

// --- Dam break -----------------------------------------------------------------------------------

const DAM_X0 = 3.25;
const DAM_X1 = 3.55;

function damBreakBed(x: number, z: number, dam: boolean): number {
  let b = 0.03 * fbm(x * 0.9, z * 0.9);
  // The far shore: a ramp the surge runs up.
  b += 0.55 * smoothstep(9.6, 11.9, x);
  // Obstacles: two round pillars, a block and a low wall, all standing clear of the water.
  b += 1.5 * solid(Math.hypot(x - 6.1, z - 2.4) - 0.55);
  b += 1.5 * solid(Math.hypot(x - 7.0, z - 5.6) - 0.62);
  b += 1.3 * solid(roundedBox(x, z, 8.9, 3.9, 0.6, 0.45, 0.08));
  b += 0.7 * solid(roundedBox(x, z, 5.0, 4.0, 0.12, 0.9, 0.05));
  // The dam: a wall across the whole channel.
  if (dam)
    b +=
      1.7 *
      (smoothstep(DAM_X0 - 0.08, DAM_X0 + 0.08, x) - smoothstep(DAM_X1 - 0.08, DAM_X1 + 0.08, x));
  return b;
}

const damBreak: ScenarioInfo = {
  id: 'dam break',
  maxDepth: 1.3,
  edges: 'wall',
  releaseAfter: 1.4,
  palette: [0, 0],
  camera: { yaw: 0.55, pitch: 0.62, distance: 12.8 },
  build(width, height) {
    const terrain = fill(width, height, (_u, _v, x, z) => damBreakBed(x, z, true));
    const released = fill(width, height, (_u, _v, x, z) => damBreakBed(x, z, false));
    const water = fill(width, height, (_u, _v, x, z) =>
      x < DAM_X0 ? Math.max(1 - damBreakBed(x, z, false), 0) : 0,
    );
    return { terrain, water, released };
  },
  forces() {},
};

// --- Hillside spring -----------------------------------------------------------------------------

function valleyCentre(u: number): number {
  return 0.3 + 0.13 * Math.sin(u * 7 + 0.4) + 0.2 * u;
}

function hillBed(u: number, v: number, x: number, z: number): number {
  // A slope falling away to the right and a little to the bottom, with hills on it...
  let b = 3.0 * Math.pow(1 - u, 1.3) * (0.7 + 0.3 * (1 - v));
  b += 2.2 * (fbm(x * 0.38, z * 0.38) - 0.5) * (0.3 + 0.7 * (1 - u));
  // ...a valley that wanders down it...
  const d = ((v - valleyCentre(u)) * WORLD_DEPTH) / 0.5;
  b -= 0.95 * Math.exp(-d * d) * smoothstep(0.03, 0.22, u) * (1 - 0.45 * u);
  // ...and a basin at the foot.
  const bu = ((u - 0.8) * WORLD_WIDTH) / 2.7;
  const bv = ((v - 0.58) * WORLD_DEPTH) / 2.1;
  b -= 0.95 * Math.exp(-(bu * bu + bv * bv));
  return b;
}

const SPRING_AT = { x: 0.085, y: 0.27 };

const hillsideSpring: ScenarioInfo = {
  id: 'hillside spring',
  maxDepth: 0.8,
  edges: 'open',
  palette: [0.75, 0],
  camera: { yaw: 1.05, pitch: 0.58, distance: 13.4 },
  build(width, height) {
    const terrain = fill(width, height, hillBed);
    // A shallow pond in the basin already, so the first thing you see is water in low ground.
    let low = Infinity;
    for (let j = 0; j < height; j++) {
      for (let i = Math.floor(width * 0.6); i < width; i++)
        low = Math.min(low, terrain[j * width + i]!);
    }
    const level = low + 0.28;
    const water = new Float32Array(width * height);
    for (let j = 0; j < height; j++) {
      for (let i = Math.floor(width * 0.6); i < width; i++) {
        water[j * width + i] = Math.max(level - terrain[j * width + i]!, 0);
      }
    }
    return { terrain, water };
  },
  forces(sim, time, strength) {
    // The spring runs steadily after a moment; a second, weaker one joins it later.
    const ramp = smoothstep(0, 1.5, time);
    sim.addSource({
      x: SPRING_AT.x,
      y: SPRING_AT.y,
      radius: 0.32,
      volume: 0.5 * strength * ramp,
    });
  },
};

// --- Tide pool -----------------------------------------------------------------------------------

const TIDE_PERIOD = 40;
const SEA_EDGE = 0.045;

function tidePoolBed(u: number, v: number, x: number, z: number): number {
  const beach = -1.0 + 1.6 * Math.pow(smoothstep(0.0, 0.32, u), 0.85);
  let b = beach;
  // The rock shelf, uneven.
  const shelf = smoothstep(0.26, 0.4, u);
  b += shelf * (0.34 * (fbm(x * 0.75, z * 0.75) - 0.45) + 0.1 * (fbm(x * 2.4, z * 2.4) - 0.5));
  // Pools in the shelf and the narrow channels that join them to the sea.
  const pool = (cu: number, cv: number, ru: number, rv: number, depth: number) => {
    const a = (u - cu) / ru;
    const c = (v - cv) / rv;
    return depth * Math.exp(-(a * a + c * c) * 1.8);
  };
  b -= pool(0.55, 0.3, 0.12, 0.2, 0.7) * shelf;
  b -= pool(0.73, 0.68, 0.14, 0.21, 0.74) * shelf;
  b -= pool(0.9, 0.3, 0.08, 0.15, 0.66) * shelf;
  b -= pool(0.5, 0.86, 0.09, 0.11, 0.6) * shelf;
  const channel = (
    v0: number,
    v1: number,
    u0: number,
    u1: number,
    width: number,
    depth: number,
  ) => {
    const t = clamp((u - u0) / (u1 - u0), 0, 1);
    const vc = v0 + (v1 - v0) * t;
    const d = ((v - vc) * WORLD_DEPTH) / width;
    return (
      depth *
      Math.exp(-d * d) *
      smoothstep(u0 - 0.02, u0 + 0.02, u) *
      (1 - smoothstep(u1 - 0.02, u1 + 0.02, u))
    );
  };
  b -= channel(0.3, 0.3, 0.3, 0.55, 0.26, 0.4) * shelf;
  b -= channel(0.68, 0.68, 0.3, 0.73, 0.24, 0.36) * shelf;
  return b;
}

const tidePool: ScenarioInfo = {
  id: 'tide pool',
  maxDepth: 1.2,
  edges: 'wall',
  palette: [0, 0.9],
  camera: { yaw: -0.7, pitch: 0.62, distance: 12.4 },
  build(width, height) {
    const terrain = fill(width, height, tidePoolBed);
    const water = new Float32Array(width * height);
    for (let i = 0; i < water.length; i++) water[i] = Math.max(0.12 - terrain[i]!, 0);
    return { terrain, water };
  },
  forces(sim, time) {
    // The tide: the sea's level follows a slow sine plus a small chop that makes the swash.
    const level =
      0.1 +
      0.62 * Math.sin((2 * Math.PI * time) / TIDE_PERIOD) +
      0.025 * Math.sin((2 * Math.PI * time) / 2.6) +
      0.012 * Math.sin((2 * Math.PI * time) / 1.3);
    sim.addLevelZone({ x0: 0, y0: 0, x1: SEA_EDGE, y1: 1, level, rate: 4 });
  },
};

export const SCENARIOS: Readonly<Record<ScenarioId, ScenarioInfo>> = {
  'dam break': damBreak,
  'hillside spring': hillsideSpring,
  'tide pool': tidePool,
};

/** Bed height at domain uv by bilinear lookup in a row-major cell array (for picking). */
export function sampleBed(bed: Float32Array, width: number, height: number, u: number, v: number) {
  const x = clamp(u * width - 0.5, 0, width - 1);
  const y = clamp(v * height - 0.5, 0, height - 1);
  const x0 = Math.floor(x);
  const y0 = Math.floor(y);
  const x1 = Math.min(x0 + 1, width - 1);
  const y1 = Math.min(y0 + 1, height - 1);
  const fx = x - x0;
  const fy = y - y0;
  const a = bed[y0 * width + x0]!;
  const b = bed[y0 * width + x1]!;
  const c = bed[y1 * width + x0]!;
  const d = bed[y1 * width + x1]!;
  return a + (b - a) * fx + (c - a) * fy + (a - b - c + d) * fx * fy;
}
