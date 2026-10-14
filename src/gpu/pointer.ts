export interface PointerState {
  /** Position over the canvas, 0..1 on each axis with (0, 0) at the top-left, like `uv`. */
  x: number;
  y: number;
  /** A button or finger is pressed (and, with `capture`, has been since it went down here). */
  down: boolean;
  /** The pointer is over the canvas. `x` and `y` keep their last value once it leaves. */
  inside: boolean;
}

export interface PointerTracker {
  /** Live: read it every frame. */
  readonly state: Readonly<PointerState>;
  /** Movement since the last call, in the same 0..1 units. Call once per frame. */
  takeDelta(): { dx: number; dy: number };
  destroy(): void;
}

export interface PointerOptions {
  /**
   * For effects the visitor drags (orbit cameras, stirring a fluid): keeps receiving the pointer
   * after it leaves the canvas mid-drag and stops touch drags from scrolling the page. Leave it
   * off for ambient effects, so a page with a full-width canvas can still be scrolled by touch.
   */
  capture?: boolean;
}

export function trackPointer(
  canvas: HTMLCanvasElement,
  options: PointerOptions = {},
): PointerTracker {
  const state: PointerState = { x: 0.5, y: 0.5, down: false, inside: false };
  let dx = 0;
  let dy = 0;
  let known = false;

  function locate(event: PointerEvent) {
    const rect = canvas.getBoundingClientRect();
    if (rect.width === 0 || rect.height === 0) return;
    const x = (event.clientX - rect.left) / rect.width;
    const y = (event.clientY - rect.top) / rect.height;
    if (known) {
      dx += x - state.x;
      dy += y - state.y;
    }
    known = true;
    state.x = x;
    state.y = y;
  }

  const controller = new AbortController();
  const listen = (type: string, handler: (event: PointerEvent) => void) =>
    canvas.addEventListener(type, handler as EventListener, { signal: controller.signal });

  listen('pointermove', locate);
  listen('pointerenter', (event) => {
    state.inside = true;
    // Entering is a jump, not a drag: do not report the distance travelled outside.
    known = false;
    locate(event);
  });
  listen('pointerleave', () => {
    state.inside = false;
  });
  listen('pointerdown', (event) => {
    state.down = true;
    locate(event);
    if (options.capture) canvas.setPointerCapture(event.pointerId);
  });
  const release = () => {
    state.down = false;
  };
  listen('pointerup', release);
  listen('pointercancel', release);

  const previousTouchAction = canvas.style.touchAction;
  if (options.capture) canvas.style.touchAction = 'none';

  return {
    state,
    takeDelta() {
      const delta = { dx, dy };
      dx = 0;
      dy = 0;
      return delta;
    },
    destroy() {
      controller.abort();
      if (options.capture) canvas.style.touchAction = previousTouchAction;
    },
  };
}

// --- Several pointers at once -----------------------------------------------------------------------
//
// `trackPointer` follows one pointer. `trackPointers` follows every pointer that is over the canvas or
// pressed on it: the mouse, a pen and each finger of a multi-touch, keyed by `pointerId`. It is an
// event accumulator: listeners record what happened, and `poll()` (once per frame) reports it and
// starts a new frame. Nothing here touches the DOM beyond the canvas it is given.

/** A point on a pointer's path, in the canvas's 0..1 space. */
export interface PointerPathPoint {
  x: number;
  y: number;
}

/** One pointer as reported by `poll()`. A fresh object per poll, safe to keep. */
export interface TrackedPointer {
  /** `pointerId`. Stable from `justPressed` to `justReleased` (and for a mouse, for good). */
  id: number;
  /** `'mouse'`, `'pen'` or `'touch'`. */
  type: string;
  /** Position over the canvas, 0..1 on each axis with (0, 0) at the top-left, like `uv`. May leave 0..1 while captured. */
  x: number;
  y: number;
  /** Movement since the previous poll, in the same 0..1 units. Entering the canvas is a jump, not movement. */
  dx: number;
  dy: number;
  /** Velocity in canvas units per second (smoothed over about 40 ms; 0 once still for 100 ms). */
  vx: number;
  vy: number;
  /** Speed in canvas HEIGHTS per second, so horizontal and vertical movement count the same. */
  speed: number;
  /** A button or finger is pressed (and, with `capture`, has been since it went down here). */
  down: boolean;
  /** The pointer is over the canvas. */
  inside: boolean;
  /** Normalised pressure 0..1 as the device reports it (0.5 for a mouse button down, 0 otherwise). */
  pressure: number;
  /** Edge flags, true only in the poll that follows the event. */
  justPressed: boolean;
  justReleased: boolean;
  justEntered: boolean;
  justLeft: boolean;
  /** The release was a cancel (the browser took a touch for scrolling). */
  cancelled: boolean;
  /**
   * Where the pointer went since the previous poll: the position at the end of the previous frame (if
   * it is known and the pointer did not just enter), then one point per move event, coalesced
   * events included. Always at least one point, the last being (x, y). Interpolate along it so a
   * fast drag leaves no dotted gaps.
   */
  path: readonly PointerPathPoint[];
}

export interface MultiPointerOptions extends PointerOptions {
  /** Pointers tracked at once; extra ones are ignored. Default 10. */
  maxPointers?: number;
}

export interface MultiPointerTracker {
  /**
   * Reports every pointer that is over the canvas or pressed, then starts a new frame: movement, edge
   * flags and paths are cleared. Pointers that have left and are not pressed (and touches that
   * have lifted) are reported once more, with their flags, and forgotten. `now` is a
   * `performance.now()`-style clock in ms (default: the real one), used to age the velocity.
   */
  poll(now?: number): TrackedPointer[];
  /** How many pointers are currently pressed. */
  readonly pressedCount: number;
  /**
   * Turns pointer capture on or off while running: with it a pressed pointer keeps reporting after it
   * leaves the canvas and touch drags no longer scroll the page (`touch-action: none`).
   */
  setCapture(capture: boolean): void;
  destroy(): void;
}

interface Tracked {
  id: number;
  type: string;
  x: number;
  y: number;
  /** Position is known (some event has located it). */
  known: boolean;
  /** Where the previous poll left the pointer; undefined right after it entered (a jump). */
  anchor: PointerPathPoint | undefined;
  dx: number;
  dy: number;
  vx: number;
  vy: number;
  lastMove: number;
  aspect: number;
  down: boolean;
  inside: boolean;
  pressure: number;
  justPressed: boolean;
  justReleased: boolean;
  justEntered: boolean;
  justLeft: boolean;
  cancelled: boolean;
  path: PointerPathPoint[];
}

/** Time constant of the velocity smoothing, seconds. */
const VELOCITY_SMOOTHING = 0.04;
/** A pointer that has not moved for this long (ms) is standing still. */
const STILL_AFTER_MS = 100;

export function trackPointers(
  canvas: HTMLCanvasElement,
  options: MultiPointerOptions = {},
): MultiPointerTracker {
  const maxPointers = options.maxPointers ?? 10;
  const pointers = new Map<number, Tracked>();
  let capture = options.capture ?? false;
  const previousTouchAction = canvas.style.touchAction;
  if (capture) canvas.style.touchAction = 'none';

  const stampOf = (event: Event) =>
    Number.isFinite(event.timeStamp) && event.timeStamp > 0 ? event.timeStamp : performance.now();

  function get(event: PointerEvent): Tracked | undefined {
    let pointer = pointers.get(event.pointerId);
    if (!pointer) {
      if (pointers.size >= maxPointers) return undefined;
      pointer = {
        id: event.pointerId,
        type: event.pointerType || 'mouse',
        x: 0.5,
        y: 0.5,
        known: false,
        anchor: undefined,
        dx: 0,
        dy: 0,
        vx: 0,
        vy: 0,
        lastMove: 0,
        aspect: 1,
        down: false,
        inside: false,
        pressure: 0,
        justPressed: false,
        justReleased: false,
        justEntered: false,
        justLeft: false,
        cancelled: false,
        path: [],
      };
      pointers.set(pointer.id, pointer);
    }
    return pointer;
  }

  /** Moves the pointer to the event's position. `jump` means the move is not travel (entering). */
  function locate(pointer: Tracked, event: PointerEvent, jump: boolean) {
    const rect = canvas.getBoundingClientRect();
    if (rect.width === 0 || rect.height === 0) return;
    pointer.aspect = rect.width / rect.height;
    const samples =
      !jump && typeof event.getCoalescedEvents === 'function' ? event.getCoalescedEvents() : [];
    for (const sample of samples.length > 0 ? samples : [event]) {
      const x = (sample.clientX - rect.left) / rect.width;
      const y = (sample.clientY - rect.top) / rect.height;
      const time = stampOf(samples.length > 0 ? sample : event);
      if (jump || !pointer.known) {
        pointer.known = true;
        pointer.anchor = undefined;
        pointer.path = [];
        pointer.dx = 0;
        pointer.dy = 0;
        pointer.vx = 0;
        pointer.vy = 0;
      } else {
        pointer.dx += x - pointer.x;
        pointer.dy += y - pointer.y;
        const seconds = (time - pointer.lastMove) / 1000;
        if (seconds > 0 && pointer.lastMove > 0) {
          const alpha = 1 - Math.exp(-seconds / VELOCITY_SMOOTHING);
          pointer.vx += ((x - pointer.x) / seconds - pointer.vx) * alpha;
          pointer.vy += ((y - pointer.y) / seconds - pointer.vy) * alpha;
        }
      }
      pointer.path.push({ x, y });
      pointer.x = x;
      pointer.y = y;
      pointer.lastMove = time;
    }
  }

  const controller = new AbortController();
  const listen = (type: string, handler: (event: PointerEvent) => void) =>
    canvas.addEventListener(type, handler as EventListener, { signal: controller.signal });

  listen('pointerenter', (event) => {
    const pointer = get(event);
    if (!pointer) return;
    pointer.inside = true;
    pointer.justEntered = true;
    locate(pointer, event, true);
  });
  listen('pointermove', (event) => {
    const pointer = get(event);
    if (!pointer) return;
    // A move can be the first thing heard about a pointer (the page loaded under the cursor).
    if (!pointer.inside && !pointer.down) pointer.justEntered = true;
    pointer.inside = true;
    pointer.pressure = event.pressure ?? pointer.pressure;
    locate(pointer, event, false);
  });
  listen('pointerdown', (event) => {
    const pointer = get(event);
    if (!pointer) return;
    pointer.type = event.pointerType || pointer.type;
    pointer.inside = true;
    pointer.down = true;
    pointer.justPressed = true;
    pointer.cancelled = false;
    pointer.pressure = event.pressure || 0.5;
    // A touch has no hover: its first event is the press, and there is nothing to connect it to.
    locate(pointer, event, !pointer.known || event.pointerType === 'touch');
    if (capture) {
      try {
        canvas.setPointerCapture(event.pointerId);
      } catch {
        // The pointer is no longer active; there is nothing to capture.
      }
    }
  });
  const releasePointer = (pointer: Tracked, cancelled: boolean) => {
    if (pointer.down) {
      pointer.justReleased = true;
      pointer.cancelled = cancelled;
    }
    pointer.down = false;
    pointer.pressure = 0;
  };
  const release = (event: PointerEvent, cancelled: boolean) => {
    const pointer = pointers.get(event.pointerId);
    if (pointer) releasePointer(pointer, cancelled);
  };
  listen('pointerup', (event) => release(event, false));
  listen('pointercancel', (event) => release(event, true));
  listen('pointerleave', (event) => {
    const pointer = pointers.get(event.pointerId);
    if (!pointer) return;
    if (pointer.inside) pointer.justLeft = true;
    pointer.inside = false;
    // Without capture a mouse or pen button released outside the canvas is never heard of, so the
    // press ends here. (Touches are captured implicitly by the browser and end with pointerup.)
    if (!capture && pointer.type !== 'touch') releasePointer(pointer, false);
  });

  return {
    poll(now = performance.now()) {
      const frame: TrackedPointer[] = [];
      for (const pointer of pointers.values()) {
        const still = now - pointer.lastMove > STILL_AFTER_MS;
        const vx = still ? 0 : pointer.vx;
        const vy = still ? 0 : pointer.vy;
        const path: PointerPathPoint[] = pointer.path.length
          ? pointer.path
          : [{ x: pointer.x, y: pointer.y }];
        // Connect with where the last frame ended, so consecutive frames form one unbroken line.
        const full = pointer.anchor && pointer.path.length ? [pointer.anchor, ...path] : path;
        frame.push({
          id: pointer.id,
          type: pointer.type,
          x: pointer.x,
          y: pointer.y,
          dx: pointer.dx,
          dy: pointer.dy,
          vx,
          vy,
          speed: Math.hypot(vx * pointer.aspect, vy),
          down: pointer.down,
          inside: pointer.inside,
          pressure: pointer.pressure,
          justPressed: pointer.justPressed,
          justReleased: pointer.justReleased,
          justEntered: pointer.justEntered,
          justLeft: pointer.justLeft,
          cancelled: pointer.cancelled,
          path: full,
        });
        pointer.anchor = { x: pointer.x, y: pointer.y };
        pointer.path = [];
        pointer.dx = 0;
        pointer.dy = 0;
        pointer.justPressed = false;
        pointer.justReleased = false;
        pointer.justEntered = false;
        pointer.justLeft = false;
        pointer.cancelled = false;
      }
      for (const [id, pointer] of pointers) {
        const gone = !pointer.down && (pointer.type === 'touch' || !pointer.inside);
        if (gone) pointers.delete(id);
      }
      return frame;
    },
    get pressedCount() {
      let count = 0;
      for (const pointer of pointers.values()) if (pointer.down) count++;
      return count;
    },
    setCapture(on) {
      if (on === capture) return;
      capture = on;
      canvas.style.touchAction = on ? 'none' : previousTouchAction;
    },
    destroy() {
      controller.abort();
      pointers.clear();
      canvas.style.touchAction = previousTouchAction;
    },
  };
}
