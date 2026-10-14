import type { TrackedPointer } from '../pointer.ts';
import type { Fluid } from './fluid.ts';

// Turning pointers into flow and ink (story 6.2). Each pointer pushes the fluid along its path and, if
// asked, lays dye behind it; a press starts a stroke, a tap makes a small puff.
//
// The path of a pointer within one frame (from `MultiPointerTracker.poll`, which starts it where the
// previous frame's ended) is cut into a few straight segments and each becomes ONE capsule splat, not a
// dot per frame: a fast drag across the canvas is as continuous as a slow one. The capsules carry a
// smooth ramp at both ends (the `ramp` of `Fluid.addForce`), so the end of one frame's capsule is
// exactly the start of the next frame's and the stroke has no beads at the joins. As a consequence
// the amount of ink per unit LENGTH of stroke is constant, whatever the pointer speed or the frame
// rate; a pointer that stays put while pressed drips (extra dye at a point), so lingering still
// makes a pool.
//
// MOMENTUM. The fluid is given the pointer's own velocity (times `force`), so dragging at 1 canvas
// height per second makes the fluid under the pointer move at about that speed. The projection then
// turns the push into a pair of counter-rotating vortices at the sides of the stroke, which is the
// look of dragging a finger through water.

export type FluidStirMode = 'hover and press' | 'press only' | 'hover only' | 'off';

/** What the colour callback is told about the stroke it is colouring. */
export interface FluidStroke {
  /** The pointer's id. */
  id: number;
  /** Counts strokes since the stirrer was made. A press (or a hover that begins) starts a new one. */
  index: number;
  /** Seconds since the stroke began. */
  age: number;
  /** Length of the stroke so far, in image heights. */
  length: number;
  /** The pointer's speed this frame, image heights per second. */
  speed: number;
  /** The pointer is hovering, not pressed. */
  hover: boolean;
}

export interface FluidStirOptions {
  mode: FluidStirMode;
  /** Radius of the push, image heights (the 1/e distance of the Gaussian). */
  radius: number;
  /** Fraction of the pointer's velocity the fluid takes. 1 matches it; the default 1 is lively. */
  force: number;
  /** Radius of the dye, image heights. Default: `radius`. */
  dyeRadius?: number;
  /** Dye laid per unit length of stroke is `dyeAmount` times the value the colour callback returns. 0 lays none. */
  dyeAmount: number;
  /** The dye value (rgba, per unit amount) of a stroke; null lays no dye. */
  colour: (stroke: FluidStroke) => readonly [number, number, number, number] | null;
  /** Hover pushes and paints at this fraction of a press. Default 0.35 for the push. */
  hoverForce?: number;
  /** Hover lays dye at this fraction of a press. Default 0.12. */
  hoverDye?: number;
  /** Cap on the velocity given to the fluid, image heights per second. Default 3. */
  maxSpeed?: number;
  /** A tap (press without moving) makes a puff of this strength: outward velocity of the ring, image heights per second. Default 0.5. 0 turns the puff off. */
  puff?: number;
  /** Dye dripped per second by a pressed pointer that is not moving. Default 4 x `dyeAmount` x radius. */
  drip?: number;
}

export interface FluidStirResult {
  /** A pointer pushes, hovers or taps this frame: the visitor is interacting. */
  active: boolean;
  /** Fastest pointer this frame, image heights per second. */
  speed: number;
  /** Splats queued this call (momentum plus dye). */
  splats: number;
}

export interface FluidStirrer {
  /**
   * Applies one frame of pointers (from `MultiPointerTracker.poll`) to the fluid. `aspect` is canvas
   * width / height (paths are in uv, lengths in image heights), `delta` the seconds since the last call.
   */
  stir(
    fluid: Pick<Fluid, 'addForce' | 'addDye'>,
    pointers: readonly TrackedPointer[],
    aspect: number,
    delta: number,
    options: FluidStirOptions,
  ): FluidStirResult;
  /** Forgets every stroke (for a new fluid or a mode change). */
  reset(): void;
}

/** Segments one frame's path is cut into at most. */
const MAX_SEGMENTS = 6;

interface Point {
  x: number;
  y: number;
}

/**
 * Cuts a polyline (uv) into `count` pieces of equal length measured in image heights; returns the
 * `count + 1` break points and the total length. A path with a single point has length 0.
 */
export function resamplePath(
  path: readonly Point[],
  aspect: number,
  count: number,
): { points: Point[]; length: number } {
  const first = path[0] ?? { x: 0, y: 0 };
  if (path.length < 2) return { points: [first, first], length: 0 };
  const cumulative = [0];
  for (let i = 1; i < path.length; i++) {
    const a = path[i - 1]!;
    const b = path[i]!;
    cumulative.push(cumulative[i - 1]! + Math.hypot((b.x - a.x) * aspect, b.y - a.y));
  }
  const length = cumulative[cumulative.length - 1]!;
  const pieces = Math.max(1, Math.floor(count));
  const points: Point[] = [];
  let k = 1;
  for (let n = 0; n <= pieces; n++) {
    const target = (length * n) / pieces;
    while (k < path.length - 1 && cumulative[k]! < target) k++;
    const a = path[k - 1]!;
    const b = path[k]!;
    const span = cumulative[k]! - cumulative[k - 1]!;
    const t = span > 1e-12 ? Math.min(1, Math.max(0, (target - cumulative[k - 1]!) / span)) : 0;
    points.push({ x: a.x + (b.x - a.x) * t, y: a.y + (b.y - a.y) * t });
  }
  return { points, length };
}

const clamp = (x: number, lo: number, hi: number) => Math.min(hi, Math.max(lo, x));

export function createFluidStirrer(): FluidStirrer {
  const strokes = new Map<number, FluidStroke>();
  let counter = 0;

  return {
    reset() {
      strokes.clear();
    },
    stir(fluid, pointers, aspect, delta, options) {
      const result: FluidStirResult = { active: false, speed: 0, splats: 0 };
      const dt = clamp(Number.isFinite(delta) ? delta : 1 / 60, 1 / 240, 0.1);
      const {
        radius,
        force,
        dyeAmount,
        colour,
        hoverForce = 0.35,
        hoverDye = 0.12,
        maxSpeed = 3,
        puff = 0.5,
      } = options;
      const dyeRadius = options.dyeRadius ?? radius;
      const drip = options.drip ?? 4 * dyeAmount * dyeRadius;
      const allowHover = options.mode === 'hover and press' || options.mode === 'hover only';
      const allowPress = options.mode === 'hover and press' || options.mode === 'press only';
      const seen = new Set<number>();

      for (const pointer of options.mode === 'off' ? [] : pointers) {
        const pressed = pointer.down;
        // A touch can only press; a mouse or pen hovers when not pressed.
        const hovering = !pressed && pointer.inside && pointer.type !== 'touch';
        const wanted = (pressed && allowPress) || (hovering && allowHover);
        if (!wanted) continue;
        seen.add(pointer.id);
        result.active = true;

        let stroke = strokes.get(pointer.id);
        if (!stroke || stroke.hover === pressed || (pointer.justPressed && allowPress)) {
          stroke = {
            id: pointer.id,
            index: counter++,
            age: 0,
            length: 0,
            speed: 0,
            hover: !pressed,
          };
          strokes.set(pointer.id, stroke);
        }
        stroke.age += dt;
        stroke.hover = !pressed;
        const push = pressed ? 1 : hoverForce;
        const paint = pressed ? 1 : hoverDye;

        const { points, length } = resamplePath(
          pointer.path,
          aspect,
          Math.ceil(length0(pointer.path, aspect) / (radius * 1.5)),
        );
        const pieces = Math.min(MAX_SEGMENTS, points.length - 1);
        const speed = length / dt;
        stroke.speed = speed;
        stroke.length += length;
        result.speed = Math.max(result.speed, speed);
        const dyeValue = dyeAmount > 0 && paint > 0 ? colour(stroke) : null;

        if (length > 1e-5) {
          const ramp = Math.min(radius, dyeRadius) * 0.5;
          for (let i = 0; i < pieces; i++) {
            const a = points[i]!;
            const b = points[i + 1]!;
            const dx = (b.x - a.x) * aspect;
            const dy = b.y - a.y;
            const segment = Math.hypot(dx, dy);
            if (segment < 1e-7) continue;
            // The pointer's velocity along this piece, capped.
            const v = Math.min(speed, maxSpeed);
            const velocity: [number, number] = [
              (dx / segment) * v * force * push,
              (dy / segment) * v * force * push,
            ];
            if (fluid.addForce([a.x, a.y], [b.x, b.y], radius, velocity, ramp)) result.splats++;
            if (dyeValue) {
              const amount = dyeAmount * paint;
              const value: [number, number, number, number] = [
                dyeValue[0] * amount,
                dyeValue[1] * amount,
                dyeValue[2] * amount,
                dyeValue[3] * amount,
              ];
              if (fluid.addDye([a.x, a.y], [b.x, b.y], dyeRadius, value, ramp)) result.splats++;
            }
          }
        }

        // Standing still while pressed: drip, and on the first frame make the puff.
        if (pressed && dyeValue) {
          const linger = clamp(1 - length / dyeRadius, 0, 1);
          const amount = drip * dt * linger;
          if (amount > 0) {
            const value: [number, number, number, number] = [
              dyeValue[0] * amount,
              dyeValue[1] * amount,
              dyeValue[2] * amount,
              dyeValue[3] * amount,
            ];
            if (fluid.addDye([pointer.x, pointer.y], [pointer.x, pointer.y], dyeRadius, value)) {
              result.splats++;
            }
          }
        }
        if (pointer.justPressed && allowPress && puff > 0) {
          // Six small outward pushes round the press: the projection turns them into a ring of
          // vortices, a drop of ink landing in water.
          for (let k = 0; k < 6; k++) {
            const angle = (k / 6) * Math.PI * 2 + stroke.index * 0.7;
            const ox = Math.cos(angle);
            const oy = Math.sin(angle);
            const at: [number, number] = [
              pointer.x + (ox * radius * 0.6) / aspect,
              pointer.y + oy * radius * 0.6,
            ];
            if (fluid.addForce(at, at, radius * 0.7, [ox * puff, oy * puff])) result.splats++;
          }
        }
      }
      for (const id of strokes.keys()) if (!seen.has(id)) strokes.delete(id);
      return result;
    },
  };
}

/** Length of a polyline in image heights. */
function length0(path: readonly Point[], aspect: number): number {
  let total = 0;
  for (let i = 1; i < path.length; i++) {
    total += Math.hypot((path[i]!.x - path[i - 1]!.x) * aspect, path[i]!.y - path[i - 1]!.y);
  }
  return total;
}
