// Tells an effect where its canvas is in the viewport, for scroll parallax and scroll response.
//
//   const scroll = trackScroll(canvas);
//   // each frame:
//   const { progress, offset, velocity } = scroll.update(time.delta);
//
// The page is read ONCE per frame, from `update`. There are no scroll listeners: a listener would
// have to be attached to every scrollable ancestor (the site's `backdrop` context scrolls an inner
// container, and the window never sees those events), and scroll events arrive at a different rate
// from the frame loop, so the effect would still have to sample once per frame to stay in step with
// what it draws. Reading geometry at the top of `frame`, before the effect writes anything,
// costs one forced layout at most, the same layout the compositor needs anyway, and the reads are
// never interleaved with writes (no layout thrash).
//
// What is read per frame: the canvas rect, the rect of the nearest scrolling ancestor (the
// "viewport" the canvas moves through; the window when there is none), and `scrollTop` of every
// scrolling ancestor plus `window.scrollY`. The ancestor chain is looked up once and again whenever
// the canvas is moved to another parent, or every few seconds in case a style changed.
//
// `progress` comes from the canvas rect, so it is the canvas's place in the viewport. `offset` comes
// from the scroll positions, so it keeps changing for a canvas that does not move on screen (a
// `position: sticky` or `fixed` backdrop), where a rect-based number would sit still. Use `progress`
// to place something along the canvas's trip through the viewport and `offset` to slide layers.
//
// With no DOM (tests, workers, server rendering) or a canvas that has no geometry, the tracker is
// inert: it reports the canvas half way through the viewport, not scrolling, and never throws.

/** Where the canvas is, as of the latest `update`. All numbers are finite. */
export interface ScrollState {
  /** 0 when the canvas's top edge enters at the bottom of the viewport, 1 when its bottom edge leaves at the top. Clamped. */
  progress: number;
  /** The same without the clamp: below 0 before the canvas arrives, above 1 after it has left. */
  rawProgress: number;
  /** Scroll distance in CSS pixels (window plus scrolling ancestors), eased by `smoothing`. 0 at the top of the page. */
  offset: number;
  /** How fast the page is scrolling, in CSS pixels per second, positive when the user scrolls down. Smoothed. */
  velocity: number;
  /** The canvas overlaps the viewport. */
  visible: boolean;
}

export interface ScrollTracker extends Readonly<ScrollState> {
  /** Reads the page and advances the smoothing by `deltaSeconds`. Call once per frame. */
  update(deltaSeconds: number): Readonly<ScrollState>;
  destroy(): void;
}

export interface ScrollOptions {
  /** Time constant in seconds for easing `offset` and `progress` (wheel notches arrive as jumps). Default 0.07; 0 disables. */
  smoothing?: number;
  /** Time constant in seconds for `velocity`. Default 0.15. */
  velocitySmoothing?: number;
}

// --- Pure arithmetic ---------------------------------------------------------------------------

/**
 * How far a box has travelled through a viewport, unclamped: 0 when the box's top edge is at the
 * bottom of the viewport, 1 when the box's bottom edge is at the top. All four arguments share one
 * unit and one origin. Returns 0.5 for a degenerate viewport and box (both zero tall).
 */
export function viewportProgress(
  boxTop: number,
  boxHeight: number,
  viewTop: number,
  viewHeight: number,
): number {
  const travel = viewHeight + boxHeight;
  if (!(travel > 0)) return 0.5;
  return (viewTop + viewHeight - boxTop) / travel;
}

/** Whether the box and the viewport overlap vertically. */
export function isInViewport(
  boxTop: number,
  boxHeight: number,
  viewTop: number,
  viewHeight: number,
): boolean {
  return boxTop < viewTop + viewHeight && boxTop + boxHeight > viewTop;
}

/**
 * Exponential smoothing that does not depend on the frame rate: moves `current` toward `target`
 * by the fraction `1 - exp(-dt / tau)`. `tau <= 0` or `dt` huge returns the target.
 */
export function smoothTowards(current: number, target: number, dt: number, tau: number): number {
  if (!(tau > 0) || !(dt > 0)) return tau > 0 ? current : target;
  return current + (target - current) * (1 - Math.exp(-dt / tau));
}

/** Scroll speed in px/s from two offsets `dt` seconds apart; 0 for a missing or negative interval. */
export function scrollSpeed(previousOffset: number, offset: number, dt: number): number {
  return dt > 1e-4 ? (offset - previousOffset) / dt : 0;
}

const finite = (value: number, fallback = 0) => (Number.isFinite(value) ? value : fallback);
const clamp01 = (value: number) => Math.min(1, Math.max(0, value));

// --- DOM access -----------------------------------------------------------------------------------

interface Box {
  top: number;
  height: number;
}

interface Scroller {
  /** Anything with the three members the tracker reads. */
  scrollTop: number;
  getBoundingClientRect(): { top: number; height: number };
  parentElement: Scroller | null;
}

/** How often the ancestor chain is looked up again, in updates (about 4 s at 60 fps). */
const RESOLVE_EVERY = 240;

function isScrollingOverflow(element: Scroller): boolean {
  try {
    const style = globalThis.getComputedStyle(element as unknown as Element);
    return /(auto|scroll|overlay)/.test(style.overflowY);
  } catch {
    return false;
  }
}

/** Every ancestor that scrolls on its own, nearest first. The root element is the window's job. */
function findScrollers(canvas: Scroller): Scroller[] {
  const found: Scroller[] = [];
  let node = canvas.parentElement;
  while (node && node.parentElement) {
    if (isScrollingOverflow(node)) found.push(node);
    node = node.parentElement;
  }
  return found;
}

function hasDom(canvas: unknown): canvas is Scroller {
  return (
    typeof window !== 'undefined' &&
    typeof canvas === 'object' &&
    canvas !== null &&
    typeof (canvas as Scroller).getBoundingClientRect === 'function'
  );
}

/**
 * Follows `canvas` through the viewport. The tracker never holds on to anything but the canvas
 * and its ancestors, and `destroy` drops them.
 */
export function trackScroll(canvas: HTMLCanvasElement, options: ScrollOptions = {}): ScrollTracker {
  const smoothing = options.smoothing ?? 0.07;
  const velocitySmoothing = options.velocitySmoothing ?? 0.15;
  const state: ScrollState = {
    progress: 0.5,
    rawProgress: 0.5,
    offset: 0,
    velocity: 0,
    visible: true,
  };

  let scrollers: Scroller[] = [];
  let resolvedFor: unknown = undefined;
  let sinceResolve = RESOLVE_EVERY;
  let rawOffset = 0;
  let started = false;
  let destroyed = false;

  const read = (): { box: Box; view: Box; offset: number } | undefined => {
    if (!hasDom(canvas)) return undefined;
    const rect = canvas.getBoundingClientRect();
    if (!(finite(rect.height, -1) >= 0)) return undefined;
    if (resolvedFor !== canvas.parentElement || ++sinceResolve >= RESOLVE_EVERY) {
      scrollers = findScrollers(canvas);
      resolvedFor = canvas.parentElement;
      sinceResolve = 0;
    }
    // The nearest scrolling ancestor is the viewport the canvas moves through; the window otherwise.
    const nearest = scrollers[0]?.getBoundingClientRect();
    const view: Box = nearest
      ? { top: finite(nearest.top), height: finite(nearest.height) }
      : { top: 0, height: finite(window.innerHeight) };
    let offset = finite(window.scrollY);
    for (const scroller of scrollers) offset += finite(scroller.scrollTop);
    return { box: { top: finite(rect.top), height: finite(rect.height) }, view, offset };
  };

  const tracker: ScrollTracker = {
    get progress() {
      return state.progress;
    },
    get rawProgress() {
      return state.rawProgress;
    },
    get offset() {
      return state.offset;
    },
    get velocity() {
      return state.velocity;
    },
    get visible() {
      return state.visible;
    },
    update(deltaSeconds) {
      if (destroyed) return state;
      const dt = Math.max(0, finite(deltaSeconds));
      const sample = read();
      if (!sample) return state;

      const raw = viewportProgress(
        sample.box.top,
        sample.box.height,
        sample.view.top,
        sample.view.height,
      );
      const speed = scrollSpeed(rawOffset, sample.offset, dt);
      state.visible = isInViewport(
        sample.box.top,
        sample.box.height,
        sample.view.top,
        sample.view.height,
      );
      if (!started) {
        // The first reading is the truth: easing in from zero would make a deep-linked page swim.
        started = true;
        state.rawProgress = raw;
        state.offset = sample.offset;
        state.velocity = 0;
      } else {
        state.rawProgress = smoothTowards(state.rawProgress, raw, dt, smoothing);
        state.offset = smoothTowards(state.offset, sample.offset, dt, smoothing);
        state.velocity = smoothTowards(state.velocity, speed, dt, velocitySmoothing);
      }
      rawOffset = sample.offset;
      state.progress = clamp01(state.rawProgress);
      return state;
    },
    destroy() {
      destroyed = true;
      scrollers = [];
      resolvedFor = undefined;
    },
  };
  return tracker;
}
