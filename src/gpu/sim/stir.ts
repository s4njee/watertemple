import type { TrackedPointer } from '../pointer.ts';
import type { Heightfield } from './heightfield.ts';

// Turning pointers into water (story 5.2). A tap makes a drop; a drag leaves a trail of dents whose
// depth follows the pointer's speed; a hover is the same trail, faint.
//
// The trail is not a dot per frame. Each frame's path (which starts where the previous frame's ended)
// is walked at a fixed spacing, a fraction of the footprint radius, and a dent is stamped at every
// step, with the distance to the next stamp carried across frames. So a fast drag across the canvas
// is as continuous as a slow one, and the stamps overlap into a smooth trough that radiates a wake.
//
// Why not `Heightfield.displace`, the moving-footprint call? Its two ends telescope: stepping a
// footprint through several positions within one frame cancels everywhere except the first and the
// last, so a path cut into pieces leaves only a dot at each end, per frame. `displace` is for a
// shape that really moves once per frame (a floating object); a pointer is better served by stamps.

export type StirMode = 'hover and press' | 'press only' | 'hover only' | 'off';

export interface StirOptions {
  mode: StirMode;
  /** Footprint radius of a drop, image heights. */
  radius: number;
  /** Strength of a drag at full speed (see Heightfield.addDrop for the unit). */
  strength: number;
  /** A tap's drop is this multiple of `strength`. Default 1.2. */
  tapGain?: number;
  /** A hover's trail is this multiple of a drag's. Default 0.25. */
  hoverGain?: number;
  /** Distance between stamps as a fraction of the radius. Default 0.6. */
  spacing?: number;
  /** Speed (image heights per second) at which a drag reaches full strength. Default 1.2. */
  fullSpeed?: number;
  /** Stamps per pointer per frame, a guard against huge paths. Default 32. */
  maxStamps?: number;
}

export interface StirResult {
  /** Impulses queued this call. */
  impulses: number;
  /** A pointer is pressed, hovering, or tapped this frame: the visitor is interacting. */
  active: boolean;
  /** Strongest pointer speed this frame, image heights per second. */
  speed: number;
}

export interface Stirrer {
  /**
   * Applies one frame of pointers (from `MultiPointerTracker.poll`) to a heightfield. `aspect` is the
   * canvas width / height: distances are in image heights, so a diagonal drag is as strong as a
   * vertical one.
   */
  stir(
    field: Pick<Heightfield, 'addDrop'>,
    pointers: readonly TrackedPointer[],
    aspect: number,
    options: StirOptions,
  ): StirResult;
}

/** Strength curve of a drag: fraction of full strength at a given speed. Slow drags still show. */
export function dragFactor(speed: number, fullSpeed: number): number {
  const s = Math.max(0, speed) / Math.max(fullSpeed, 1e-6);
  return Math.min(1.5, 0.15 + 0.85 * Math.pow(Math.min(s, 1.5), 0.8));
}

export function createStirrer(): Stirrer {
  /** Distance left until each pointer's next stamp, image heights. */
  const carry = new Map<number, number>();

  return {
    stir(field, pointers, aspect, options) {
      const result: StirResult = { impulses: 0, active: false, speed: 0 };
      const seen = new Set<number>();
      const {
        radius,
        strength,
        tapGain = 1.2,
        hoverGain = 0.25,
        spacing = 0.6,
        fullSpeed = 1.2,
        maxStamps = 32,
      } = options;
      const allowHover = options.mode === 'hover and press' || options.mode === 'hover only';
      const allowPress = options.mode === 'hover and press' || options.mode === 'press only';
      const step = Math.max(radius * spacing, 1e-4);

      for (const pointer of options.mode === 'off' ? [] : pointers) {
        const pressed = pointer.down;
        // A touch can only press; a mouse or pen hovers when not pressed.
        const hovering = !pressed && pointer.inside && pointer.type !== 'touch';
        const wanted = (pressed && allowPress) || (hovering && allowHover);
        if (!wanted) {
          carry.delete(pointer.id);
          continue;
        }
        seen.add(pointer.id);
        result.active = true;
        result.speed = Math.max(result.speed, pointer.speed);

        if (pointer.justPressed && allowPress) {
          if (field.addDrop(pointer.x, pointer.y, radius * 1.3, -strength * tapGain)) {
            result.impulses++;
          }
        }

        const gain = pressed ? 1 : hoverGain;
        const stamp = -strength * gain * dragFactor(pointer.speed, fullSpeed) * 0.5;
        let toNext = carry.get(pointer.id) ?? 0;
        let stamps = 0;
        for (let i = 1; i < pointer.path.length && stamps < maxStamps; i++) {
          const a = pointer.path[i - 1]!;
          const b = pointer.path[i]!;
          const length = Math.hypot((b.x - a.x) * aspect, b.y - a.y);
          let travelled = toNext;
          while (travelled <= length && stamps < maxStamps) {
            const t = length > 0 ? travelled / length : 0;
            // The very first stamp of a stroke lands on the press itself, which is also the tap's drop.
            if (!(pointer.justPressed && i === 1 && travelled === 0 && allowPress)) {
              if (field.addDrop(a.x + (b.x - a.x) * t, a.y + (b.y - a.y) * t, radius, stamp)) {
                result.impulses++;
              }
            }
            stamps++;
            travelled += step;
          }
          toNext = travelled - length;
        }
        // If the stamp budget ran out, forget the backlog rather than owing it next frame.
        carry.set(pointer.id, Math.min(toNext, step));
      }
      for (const id of carry.keys()) if (!seen.has(id)) carry.delete(id);
      return result;
    },
  };
}
