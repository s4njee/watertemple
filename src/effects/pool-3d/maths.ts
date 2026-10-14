/*
 * The pure maths of the 3D pool: picking and dragging the ball, the ball's buoyancy and how it
 * stirs the water, and the CPU twins of the shader's ray helpers (pool-rays.wgsl), so a test can
 * compare the two. World space is the project's: right-handed, +Y up. The pool is the box
 * |x| <= POOL_HALF, |z| <= POOL_HALF, from the floor at y = -depth up to the rim at y = POOL_RIM;
 * the mean water level is y = 0.
 */

export type Vec3 = [number, number, number];

/** Half the pool's width and length, world units. */
export const POOL_HALF = 1;
/** Height of the rim (and the deck around it) above the water. */
export const POOL_RIM = 0.22;

/** Where the ball and the camera start (the tests aim a drag at the ball from these). */
export const INITIAL_BALL: Vec3 = [0.38, 0.05, -0.2];
export const INITIAL_ORBIT = {
  yaw: 0.7,
  pitch: 0.6,
  distance: 3.7,
  target: [0, -0.3, 0] as Vec3,
};
/** Vertical field of view, radians; widened on a tall canvas so at least MIN_HORIZONTAL_FOV shows. */
export const CAMERA_FOV_Y = (38 * Math.PI) / 180;
export const MIN_HORIZONTAL_FOV = (52 * Math.PI) / 180;

const sub = (a: Vec3, b: Vec3): Vec3 => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const add = (a: Vec3, b: Vec3): Vec3 => [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
const scale = (a: Vec3, s: number): Vec3 => [a[0] * s, a[1] * s, a[2] * s];
const dot = (a: Vec3, b: Vec3) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const length = (a: Vec3) => Math.hypot(a[0], a[1], a[2]);
const normalize = (a: Vec3): Vec3 => {
  const l = length(a) || 1;
  return [a[0] / l, a[1] / l, a[2] / l];
};
export const clamp = (x: number, lo: number, hi: number) => Math.min(hi, Math.max(lo, x));

// --- Rays ----------------------------------------------------------------------------------------

/**
 * Distance along the ray to where it ENTERS the sphere, or null when it misses, the sphere is
 * behind the origin, or the origin is already inside (a ray that starts in the ball does not hit it).
 * `dir` must be a unit vector.
 */
export function raySphere(origin: Vec3, dir: Vec3, center: Vec3, radius: number): number | null {
  const oc = sub(origin, center);
  const b = dot(oc, dir);
  const c = dot(oc, oc) - radius * radius;
  const disc = b * b - c;
  if (disc < 0) return null;
  const t = -b - Math.sqrt(disc);
  return t > 1e-4 ? t : null;
}

export interface BoxExit {
  /** Distance along the ray to the wall it leaves through. */
  t: number;
  /** Unit normal of that wall, pointing into the box. */
  normal: Vec3;
}

/**
 * Where a ray that starts INSIDE the open-topped box |x|, |z| <= POOL_HALF, yLo <= y <= yHi leaves it,
 * and through which face. Rays parallel to a pair of faces never leave through them. The mirror of
 * `pl_exit` in pool-rays.wgsl.
 */
export function rayBoxExit(origin: Vec3, dir: Vec3, yLo: number, yHi: number): BoxExit {
  let best = Infinity;
  let normal: Vec3 = [0, 1, 0];
  const consider = (t: number, n: Vec3) => {
    if (t < best) {
      best = t;
      normal = n;
    }
  };
  if (dir[0] > 1e-6) consider((POOL_HALF - origin[0]) / dir[0], [-1, 0, 0]);
  else if (dir[0] < -1e-6) consider((-POOL_HALF - origin[0]) / dir[0], [1, 0, 0]);
  if (dir[2] > 1e-6) consider((POOL_HALF - origin[2]) / dir[2], [0, 0, -1]);
  else if (dir[2] < -1e-6) consider((-POOL_HALF - origin[2]) / dir[2], [0, 0, 1]);
  if (dir[1] > 1e-6) consider((yHi - origin[1]) / dir[1], [0, -1, 0]);
  else if (dir[1] < -1e-6) consider((yLo - origin[1]) / dir[1], [0, 1, 0]);
  return { t: Math.max(best, 0), normal };
}

/** Distance along the ray to the plane, or null when the ray is parallel to it or the plane is behind. */
export function intersectPlane(origin: Vec3, dir: Vec3, point: Vec3, normal: Vec3): number | null {
  const denominator = dot(dir, normal);
  if (Math.abs(denominator) < 1e-8) return null;
  const t = dot(sub(point, origin), normal) / denominator;
  return t >= 0 ? t : null;
}

/**
 * The world ray through screen uv (origin top-left) from an unprojection (depth 0 is the near plane,
 * 1 the far plane): its origin is the point on the near plane, a hair in front of the eye, and its
 * direction is a unit vector.
 */
export function pointerRay(
  unproject: (u: number, v: number, depth: number) => ArrayLike<number>,
  u: number,
  v: number,
): { origin: Vec3; dir: Vec3 } {
  const near = unproject(u, v, 0);
  const far = unproject(u, v, 1);
  const origin: Vec3 = [near[0]!, near[1]!, near[2]!];
  return { origin, dir: normalize(sub([far[0]!, far[1]!, far[2]!], origin)) };
}

// --- Dragging ------------------------------------------------------------------------------------

export interface Drag {
  /** The plane the ball moves in: through the grab point, facing the camera. */
  point: Vec3;
  normal: Vec3;
  /** Ball centre minus the grab point, so the ball does not jump to the pointer. */
  offset: Vec3;
}

/** Starts a drag: the ray hit the ball at `hit`; the plane through it faces `cameraForward`'s opposite. */
export function beginDrag(hit: Vec3, center: Vec3, cameraForward: Vec3): Drag {
  return { point: hit, normal: normalize(scale(cameraForward, -1)), offset: sub(center, hit) };
}

/** The ball centre for a pointer ray during a drag, or null when the ray never meets the drag plane. */
export function dragOnPlane(origin: Vec3, dir: Vec3, drag: Drag): Vec3 | null {
  const t = intersectPlane(origin, dir, drag.point, drag.normal);
  if (t === null) return null;
  return add(add(origin, scale(dir, t)), drag.offset);
}

/** Keeps a ball of this radius inside the pool and above the floor; `maxY` is how high it may be lifted. */
export function clampBall(center: Vec3, radius: number, floorY: number, maxY: number): Vec3 {
  const limit = POOL_HALF - radius;
  return [
    clamp(center[0], -limit, limit),
    clamp(center[1], floorY + radius, maxY),
    clamp(center[2], -limit, limit),
  ];
}

// --- Buoyancy ------------------------------------------------------------------------------------

/** Fraction of a sphere's volume under the water plane y = 0, for a centre at height `y`. */
export function submergedFraction(y: number, radius: number): number {
  const cap = clamp(radius - y, 0, 2 * radius);
  return (cap * cap * (3 * radius - cap)) / (4 * radius ** 3);
}

export interface BallBody {
  center: Vec3;
  /** Vertical velocity, world units per second. */
  vy: number;
}

export interface BallOptions {
  radius: number;
  /** Ball density relative to water: below 1 floats (submerged fraction settles at the density), above 1 sinks. */
  density: number;
  floorY: number;
  /** Highest the ball may be lifted. */
  maxY: number;
  /** Held by the visitor: no physics. */
  held: boolean;
  gravity?: number;
}

/**
 * One physics step of the ball's height: gravity against buoyancy (acceleration g (f / density - 1),
 * f the submerged fraction), drag that grows with the submerged part, the floor. Horizontal position
 * is the visitor's alone. Sub-stepped for stability, so any `dt` is safe.
 */
export function stepBall(body: BallBody, options: BallOptions, dt: number): BallBody {
  const { radius, density, floorY, maxY, held, gravity = 3.2 } = options;
  if (held) {
    body.vy = 0;
    return body;
  }
  const steps = Math.max(1, Math.ceil(dt / (1 / 120)));
  const h = dt / steps;
  for (let i = 0; i < steps; i++) {
    const f = submergedFraction(body.center[1], radius);
    const accel = gravity * (f / Math.max(density, 0.05) - 1);
    body.vy += accel * h;
    body.vy *= Math.exp(-h * (0.25 + 4 * f));
    body.center[1] += body.vy * h;
    if (body.center[1] < floorY + radius) {
      body.center[1] = floorY + radius;
      body.vy = Math.max(0, body.vy) * 0.3;
    }
    if (body.center[1] > maxY) {
      body.center[1] = maxY;
      body.vy = Math.min(0, body.vy);
    }
  }
  return body;
}

export interface WaterInfluence {
  /** 0..1: how much of the ball's motion reaches the surface. */
  weight: number;
  /** Radius of the footprint it makes, world units. */
  radius: number;
}

/**
 * How a ball at height `y` couples to the surface. Cutting the surface it pushes with the radius of
 * the circle it cuts (never less than 40% of its radius, so a ball just touching still stirs);
 * fully under, its influence fades with depth; above the water it has none.
 */
export function ballWaterInfluence(y: number, radius: number): WaterInfluence {
  if (y >= radius) return { weight: 0, radius };
  if (y > -radius) {
    return { weight: 1, radius: Math.max(Math.sqrt(radius * radius - y * y), 0.4 * radius) };
  }
  return { weight: Math.exp((y + radius) / (0.5 * radius)), radius };
}

// --- Frames --------------------------------------------------------------------------------------

/** Unit vector toward the sun in world space (y up) from elevation and azimuth in radians. */
export function sunDirection(elevation: number, azimuth: number): Vec3 {
  return [
    Math.cos(elevation) * Math.cos(azimuth),
    Math.sin(elevation),
    Math.cos(elevation) * Math.sin(azimuth),
  ];
}

/** Heightfield uv (top-left origin, v along +z) of a world position over the pool. */
export function worldToPoolUv(x: number, z: number): [number, number] {
  return [x / (2 * POOL_HALF) + 0.5, z / (2 * POOL_HALF) + 0.5];
}

/**
 * The sun's light in the heightfield frame the caustics work in (x right, y down the canvas, z up):
 * a world direction (x, y, z) maps to (x, z, y), because uv.v runs along world +z.
 */
export function toCausticsFrame(v: Vec3): Vec3 {
  return [v[0], v[2], v[1]];
}

/** A beam entering flat water: refracted unit direction (world), and the horizontal travel per unit depth. */
export function flatSunRefraction(sun: Vec3, ior: number): { dir: Vec3; shift: [number, number] } {
  const i = scale(sun, -1);
  const eta = 1 / ior;
  // Snell's law at a horizontal surface: refract(i, up, eta) = eta i - (eta dot(up, i) + sqrt(k)) up.
  const d = i[1];
  const k = Math.max(1 - eta * eta * (1 - d * d), 0);
  const dir = normalize([eta * i[0], eta * i[1] - (eta * d + Math.sqrt(k)), eta * i[2]]);
  return { dir, shift: [dir[0] / -dir[1], dir[2] / -dir[1]] };
}
