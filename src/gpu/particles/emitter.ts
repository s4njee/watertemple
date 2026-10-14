// Emitters: a description of where particles appear, how fast they leave, and how long they live. The same
// description drives two paths that produce the same kind of particle:
//
//  - the CPU path, `ParticleSystem.emit(emitter, n)`: samples n particles here and uploads them (a few
//    thousand per frame is nothing; this is how a stream grows toward its count), and
//  - the GPU path, the integrator's `respawn`: a particle whose life runs out is re-seeded in the shader
//    from the same description (src/wgsl/particles.wgsl, `particle_emit_position`/`_velocity`), with no
//    CPU involvement. That is how a fountain keeps running with a million particles.
//
// The two use different random generators, so a given seed does not give the same particle on both.

export type Vec3 = readonly [number, number, number];

export type EmitterShape = 'point' | 'box' | 'sphere' | 'disc';

/** The numbers `particle_emit_position` expects in `shape`; keep in step with src/wgsl/particles.wgsl. */
export const EMITTER_SHAPE_CODE: Readonly<Record<EmitterShape, number>> = {
  point: 0,
  box: 1,
  sphere: 2,
  disc: 3,
};

export interface Emitter {
  /**
   * `point`: at `center`. `box`: uniformly in the box with half-size `extent`. `sphere`: uniformly in the
   * ball of radius `extent[0]` (a disc in 2D). `disc`: uniformly in the horizontal (xz) disc of radius
   * `extent[0]` (a segment along x in 2D). Default `point`.
   */
  shape?: EmitterShape;
  center?: Vec3;
  extent?: Vec3;
  /** Unit vector the particles leave along. Default +y. */
  direction?: Vec3;
  /** Launch speed, world units per second. Default 0. */
  speed?: number;
  /** Half-angle of the cone (a fan in 2D) around `direction`, in radians. Default 0. */
  spread?: number;
  /** Fractional random variation of the speed, 0..1. Default 0. */
  speedJitter?: number;
  /**
   * Life in seconds, uniform between the two. Stored in `velocities.w` (so a system needs velocities). Omit
   * or pass `[0, 0]` for immortal particles (w = 0 is then not "dead": respawn only looks at life when the
   * integrator is given a `respawn` description).
   */
  lifetime?: readonly [number, number];
  /** Initial value of the per-particle scalar (`positions.w`). Default 0. */
  scalar?: number;
}

export interface ResolvedEmitter {
  shape: EmitterShape;
  center: [number, number, number];
  extent: [number, number, number];
  direction: [number, number, number];
  speed: number;
  spread: number;
  speedJitter: number;
  lifetime: [number, number];
  scalar: number;
}

export function resolveEmitter(emitter: Emitter): ResolvedEmitter {
  const [dx = 0, dy = 1, dz = 0] = emitter.direction ?? [0, 1, 0];
  const length = Math.hypot(dx, dy, dz) || 1;
  const [life0 = 0, life1 = life0] = emitter.lifetime ?? [0, 0];
  return {
    shape: emitter.shape ?? 'point',
    center: [...(emitter.center ?? [0, 0, 0])] as [number, number, number],
    extent: [...(emitter.extent ?? [0, 0, 0])] as [number, number, number],
    direction: [dx / length, dy / length, dz / length],
    speed: emitter.speed ?? 0,
    spread: emitter.spread ?? 0,
    speedJitter: emitter.speedJitter ?? 0,
    lifetime: [life0, Math.max(life0, life1)],
    scalar: emitter.scalar ?? 0,
  };
}

/** mulberry32: a small seeded generator, so a given seed always emits the same particles. */
export function createRandom(seed: number): () => number {
  let state = seed >>> 0;
  return () => {
    state = (state + 0x6d2b79f5) >>> 0;
    let t = state;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/**
 * Writes `count` emitted particles as `vec4f` rows into `positions` (xyz, scalar) and `velocities` (xyz,
 * life), starting at row `at`. The CPU twin of `particle_emit_position` / `particle_emit_velocity`.
 */
export function sampleEmitter(
  emitter: Emitter,
  dimension: 2 | 3,
  random: () => number,
  count: number,
  positions: Float32Array,
  velocities: Float32Array | undefined,
  at = 0,
): void {
  const e = resolveEmitter(emitter);
  const [dirX, dirY, dirZ] = e.direction;
  for (let n = 0; n < count; n++) {
    const row = (at + n) * 4;
    let x = 0;
    let y = 0;
    let z = 0;
    if (e.shape === 'box') {
      x = (random() * 2 - 1) * e.extent[0];
      y = (random() * 2 - 1) * e.extent[1];
      z = (random() * 2 - 1) * e.extent[2];
    } else if (e.shape === 'sphere') {
      if (dimension === 2) {
        const a = random() * Math.PI * 2;
        const r = Math.sqrt(random()) * e.extent[0];
        x = Math.cos(a) * r;
        y = Math.sin(a) * r;
      } else {
        const cz = random() * 2 - 1;
        const a = random() * Math.PI * 2;
        const s = Math.sqrt(Math.max(1 - cz * cz, 0));
        const r = Math.cbrt(random()) * e.extent[0];
        x = s * Math.cos(a) * r;
        y = cz * r;
        z = s * Math.sin(a) * r;
      }
    } else if (e.shape === 'disc') {
      if (dimension === 2) {
        x = (random() * 2 - 1) * e.extent[0];
      } else {
        const a = random() * Math.PI * 2;
        const r = Math.sqrt(random()) * e.extent[0];
        x = Math.cos(a) * r;
        z = Math.sin(a) * r;
      }
    }
    positions[row] = e.center[0] + x;
    positions[row + 1] = e.center[1] + y;
    positions[row + 2] = dimension === 2 ? 0 : e.center[2] + z;
    positions[row + 3] = e.scalar;

    if (velocities) {
      const speed = e.speed * (1 + (random() * 2 - 1) * e.speedJitter);
      let vx: number;
      let vy: number;
      let vz = 0;
      if (dimension === 2) {
        const angle = (random() * 2 - 1) * e.spread;
        const c = Math.cos(angle);
        const s = Math.sin(angle);
        vx = (c * dirX - s * dirY) * speed;
        vy = (s * dirX + c * dirY) * speed;
        random(); // keep the stream the same length as in 3D
      } else {
        const cosTheta = 1 + random() * (Math.cos(e.spread) - 1);
        const sinTheta = Math.sqrt(Math.max(1 - cosTheta * cosTheta, 0));
        const phi = random() * Math.PI * 2;
        // An orthonormal basis around the direction.
        const upX = Math.abs(dirY) > 0.99 ? 1 : 0;
        const upY = upX ? 0 : 1;
        let t1x = dirY * 0 - dirZ * upY;
        let t1y = dirZ * upX - dirX * 0;
        let t1z = dirX * upY - dirY * upX;
        const t1 = Math.hypot(t1x, t1y, t1z) || 1;
        t1x /= t1;
        t1y /= t1;
        t1z /= t1;
        const t2x = dirY * t1z - dirZ * t1y;
        const t2y = dirZ * t1x - dirX * t1z;
        const t2z = dirX * t1y - dirY * t1x;
        const cp = Math.cos(phi) * sinTheta;
        const sp = Math.sin(phi) * sinTheta;
        vx = (dirX * cosTheta + t1x * cp + t2x * sp) * speed;
        vy = (dirY * cosTheta + t1y * cp + t2y * sp) * speed;
        vz = (dirZ * cosTheta + t1z * cp + t2z * sp) * speed;
      }
      velocities[row] = vx;
      velocities[row + 1] = vy;
      velocities[row + 2] = vz;
      velocities[row + 3] = e.lifetime[0] + random() * (e.lifetime[1] - e.lifetime[0]);
    }
  }
}
