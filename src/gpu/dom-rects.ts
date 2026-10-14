import type { ObstacleRect } from './sim/obstacles.ts';

// Where the page's own elements are, in the canvas's uv space, so a simulation can treat them as
// solid (story 5.5: ripples that bounce off cards and headings).
//
//   <div class="card" data-wt-obstacle>...</div>          mark an element: it becomes a wall
//   <section data-wt-obstacle-root> <canvas/> ... </section>   optional: scope the search
//
//   const dom = trackDomRects(canvas);
//   // each frame, before the effect writes anything:
//   const { rects, changed } = dom.update(deltaSeconds);
//   if (changed) heightfield.obstacles.setRects(rects);
//
// A WebGPU canvas cannot see the DOM's pixels, but it can know where elements ARE. That is enough
// for a wall: the box, with the corner radius the author gave it. Nothing here reads pixels and
// nothing writes to the page.
//
// THE CONVENTION. An element is an obstacle when it carries `data-wt-obstacle` (any value except
// "off" or "false"; the value is reserved, write it bare). Its border box becomes a rounded
// rectangle, the corner radius is the largest of its four computed `border-radius` values. A
// transformed element is its bounding box (a rotated card is a slightly large rectangle), text is
// the box of its element (mark the inline element, or give a heading `width: fit-content`, if the
// water should hug the words rather than the line), and an element with `display: none`,
// `visibility: hidden`, almost no opacity, or no size contributes nothing.
//
// WHERE IT LOOKS. Not the whole document: several effects can share a page. The search root is
//   1. `options.root`, if given;
//   2. the nearest ancestor of the canvas with `data-wt-obstacle-root`;
//   3. the nearest ancestor of the canvas (stopping before <body>) that contains a marked element;
//   4. the canvas's parent.
// The root is looked up again whenever the canvas is moved to another parent and, with the element
// list, every `requeryEvery` frames. Only elements that overlap the canvas come back as rects, so a
// big root costs a few more `getBoundingClientRect` calls and nothing else.
//
// MEASUREMENT STRATEGY. Per frame, from `update`, like `trackScroll`. The alternatives all fail on
// correctness, not only on cost:
//   - ResizeObserver reports size, not position: it is silent while a card scrolls over a sticky
//     canvas, while a transform animates, or while a sibling above pushes the layout down.
//   - IntersectionObserver reports coarse thresholds of overlap, not boxes.
//   - Scroll listeners must be attached to every scrollable ancestor (the `backdrop` context
//     scrolls an inner container, which the window never hears about), arrive at a different rate
//     from the frame loop, and still miss CSS transitions, animations and JS-driven layout.
//   - A MutationObserver sees DOM edits, not the geometry they cause.
// What is read per frame is one `getBoundingClientRect` for the canvas and one per marked element:
// the same call the browser's own hit testing uses, answered from the layout the browser has just
// finished (a rAF callback runs after style and layout of the previous frame, and this class
// writes nothing, so it never forces a second layout and never interleaves reads with writes).
// Computed style (corner radius, visibility) is the expensive read, so it is cached per element and
// refreshed when the element changes size and every `styleEvery` frames; a transitioning radius is
// therefore late by at most that many frames. A MutationObserver on the root (when the browser has
// one) only marks the element LIST dirty, so a marked element added or removed is picked up on the
// next frame instead of the next requery. Cost: about 0.002 ms per element per frame in a browser
// with a clean layout (not measured on this machine in isolation; see the effect's notes).
//
// WITHOUT A DOM (tests, workers, a canvas stub) the tracker is inert: `update` returns no rects and
// never throws.

/** The attribute that marks an element as solid. */
export const OBSTACLE_ATTRIBUTE = 'data-wt-obstacle';
/** The attribute that scopes the search to an element's subtree. */
export const OBSTACLE_ROOT_ATTRIBUTE = 'data-wt-obstacle-root';
export const OBSTACLE_SELECTOR = `[${OBSTACLE_ATTRIBUTE}]:not([${OBSTACLE_ATTRIBUTE}="off"]):not([${OBSTACLE_ATTRIBUTE}="false"])`;

// --- Pure geometry -----------------------------------------------------------------------------

/** A box in CSS pixels, the shape of a DOMRect. */
export interface Box {
  left: number;
  top: number;
  width: number;
  height: number;
}

/** Four edges in canvas uv. Not clipped: an element half outside the canvas has an edge below 0. */
export interface EdgeBox {
  left: number;
  top: number;
  right: number;
  bottom: number;
}

/** How fast each edge moved, uv per second. Positive is right / down. */
export type EdgeMotion = EdgeBox;

export interface ObstacleOptions {
  /** Corner radius in CSS pixels. Default 0. */
  radiusPx?: number;
  /** Grow (positive) or shrink (negative) the box on every side, CSS pixels. Default 0. */
  outsetPx?: number;
  /** Boxes with a side shorter than this many CSS pixels are skipped. Default 1. */
  minSizePx?: number;
}

export interface Obstacle {
  /** What goes to `Heightfield.obstacles.setRects`: clipped to the canvas, radius in image heights. */
  rect: ObstacleRect;
  /** The same box before clipping, as edges. */
  box: EdgeBox;
  /**
   * The element's own box, clipped the same way, when an `outsetPx` made `rect` bigger or smaller
   * than the element; the same object as `rect` otherwise. For drawing: the wall sits at `rect`, the
   * element at `exact`.
   */
  exact?: ObstacleRect;
  /** Edge speeds since the previous update; undefined for a new, teleporting or first-seen element. */
  motion?: EdgeMotion;
  /** The element the box was measured from (set by `trackDomRects`), for effects that read their own attributes off it. */
  element?: Element;
}

const clamp = (x: number, lo: number, hi: number) => Math.min(hi, Math.max(lo, x));
const finiteBox = (box: Box) =>
  Number.isFinite(box.left) &&
  Number.isFinite(box.top) &&
  Number.isFinite(box.width) &&
  Number.isFinite(box.height);

/**
 * An element's box as an obstacle in the uv space of a canvas whose box is `canvas` (both in the same
 * pixel space, e.g. the viewport). Returns undefined when the box is not finite or has no size, or
 * the canvas has none, or the two do not overlap.
 *
 * Clipping. The rect is clipped to the canvas widened by the corner radius on every side, not to the
 * canvas itself: the rounded corner of a card that runs off the canvas then lies outside the canvas
 * instead of being cut into a bogus rounded corner at the canvas edge. With radius 0 it is exactly
 * the canvas. The radius is in image heights (canvas height = 1) and is limited to half the box's
 * shorter side.
 */
export function boxToObstacle(
  box: Box,
  canvas: Box,
  options: ObstacleOptions = {},
): Obstacle | undefined {
  const { radiusPx = 0, outsetPx = 0, minSizePx = 1 } = options;
  if (!finiteBox(box) || !finiteBox(canvas)) return undefined;
  if (!(canvas.width > 0 && canvas.height > 0)) return undefined;
  const width = box.width + 2 * outsetPx;
  const height = box.height + 2 * outsetPx;
  if (!(width >= minSizePx && height >= minSizePx)) return undefined;

  const left = (box.left - outsetPx - canvas.left) / canvas.width;
  const top = (box.top - outsetPx - canvas.top) / canvas.height;
  const right = left + width / canvas.width;
  const bottom = top + height / canvas.height;
  if (right <= 0 || left >= 1 || bottom <= 0 || top >= 1) return undefined;

  const aspect = canvas.width / canvas.height;
  const half = Math.min(width, height) / 2 / canvas.height;
  const radius = clamp(Math.max(0, radiusPx + Math.max(outsetPx, 0)) / canvas.height, 0, half);
  const marginX = radius / aspect;
  const x0 = Math.max(left, 0 - marginX);
  const x1 = Math.min(right, 1 + marginX);
  const y0 = Math.max(top, 0 - radius);
  const y1 = Math.min(bottom, 1 + radius);
  if (!(x1 > x0 && y1 > y0)) return undefined;
  return {
    rect: { x: x0, y: y0, width: x1 - x0, height: y1 - y0, radius },
    box: { left, top, right, bottom },
  };
}

/** The computed `border-*-radius` properties the radius comes from. */
export interface CornerStyle {
  borderTopLeftRadius?: string;
  borderTopRightRadius?: string;
  borderBottomRightRadius?: string;
  borderBottomLeftRadius?: string;
}

/**
 * One computed corner value ("12px", "50%", "12px 8px" for an elliptical corner) as a circular
 * radius in px: the smaller of the horizontal and vertical radius. Percentages are of the box.
 */
export function parseCornerRadius(
  value: string | undefined,
  width: number,
  height: number,
): number {
  if (!value) return 0;
  const parts = value.trim().split(/\s+/);
  const resolve = (token: string | undefined, basis: number) => {
    if (!token) return 0;
    const n = Number.parseFloat(token);
    if (!Number.isFinite(n) || n < 0) return 0;
    return token.endsWith('%') ? (n / 100) * basis : n;
  };
  const horizontal = resolve(parts[0], width);
  // One value is a circle; for a percentage that means a percentage of each side.
  const vertical = resolve(parts[1] ?? parts[0], height);
  return Math.min(horizontal, vertical);
}

/** The largest corner radius of a box, px, never more than half its shorter side (a pill is 999px). */
export function maxCornerRadiusPx(style: CornerStyle, width: number, height: number): number {
  const radii = [
    style.borderTopLeftRadius,
    style.borderTopRightRadius,
    style.borderBottomRightRadius,
    style.borderBottomLeftRadius,
  ].map((value) => parseCornerRadius(value, width, height));
  return clamp(Math.max(0, ...radii), 0, Math.max(0, Math.min(width, height) / 2));
}

/** Whether computed style says the element can be seen at all. */
export function isVisibleStyle(
  style: { display?: string; visibility?: string; opacity?: string },
  minOpacity = 0.05,
): boolean {
  if (style.display === 'none') return false;
  if (style.visibility === 'hidden' || style.visibility === 'collapse') return false;
  const opacity = style.opacity === undefined || style.opacity === '' ? 1 : Number(style.opacity);
  return !Number.isFinite(opacity) || opacity >= minOpacity;
}

/**
 * Whether two obstacle lists differ by more than `epsilon` in any uv component (the radius is
 * compared in image heights). Same length and order are expected: the tracker keeps DOM order.
 */
export function rectsChanged(
  a: readonly ObstacleRect[],
  b: readonly ObstacleRect[],
  epsilon = 1e-4,
): boolean {
  if (a.length !== b.length) return true;
  for (let i = 0; i < a.length; i++) {
    const p = a[i]!;
    const q = b[i]!;
    if (
      Math.abs(p.x - q.x) > epsilon ||
      Math.abs(p.y - q.y) > epsilon ||
      Math.abs(p.width - q.width) > epsilon ||
      Math.abs(p.height - q.height) > epsilon ||
      Math.abs((p.radius ?? 0) - (q.radius ?? 0)) > epsilon
    ) {
      return true;
    }
  }
  return false;
}

/**
 * Edge speeds between two readings `dt` seconds apart. Undefined when `dt` is not a believable frame
 * interval (0, or a tab that slept) or when an edge jumped farther than `maxJump` uv, which is a
 * layout change (a breakpoint) and not motion.
 */
export function edgeMotion(
  previous: EdgeBox,
  next: EdgeBox,
  dt: number,
  maxJump = 0.3,
  maxDt = 0.25,
): EdgeMotion | undefined {
  if (!(dt > 1e-4 && dt <= maxDt)) return undefined;
  const dl = next.left - previous.left;
  const dt_ = next.top - previous.top;
  const dr = next.right - previous.right;
  const db = next.bottom - previous.bottom;
  if (![dl, dt_, dr, db].every((d) => Math.abs(d) <= maxJump)) return undefined;
  return { left: dl / dt, top: dt_ / dt, right: dr / dt, bottom: db / dt };
}

export interface WakeOptions {
  /** Canvas width / height. */
  aspect: number;
  /** Seconds this reading covers: the impulse is the distance the edge moved, so it is rate independent. */
  dt: number;
  /** Smallest drop radius, image heights. */
  radius: number;
  /** Impulse strength per image height an edge advances (Heightfield.addDrop units). */
  gain: number;
  /** Edge speed (image heights per second) the response saturates at. Default 1.5. */
  maxSpeed?: number;
  /** Slower edges are ignored (image heights per second). Default 0.03. */
  minSpeed?: number;
  /** At most this many drops along one edge. Default 5. */
  maxPerEdge?: number;
}

export interface WakeDrop {
  x: number;
  y: number;
  radius: number;
  strength: number;
}

/**
 * The water a moving box pushes: along each edge that is moving, a few drops just outside it,
 * raised where the edge advances into water and dented where it retreats and the water follows.
 * Edges that lie on or beyond the canvas boundary are not real edges and are skipped. The response
 * saturates (`tanh`) at `maxSpeed`, so a flick of the scroll wheel is a big wave, not a huge one.
 */
export function edgeWake(box: EdgeBox, motion: EdgeMotion, options: WakeOptions): WakeDrop[] {
  const { aspect, dt, radius, gain, maxSpeed = 1.5, minSpeed = 0.03, maxPerEdge = 5 } = options;
  const drops: WakeDrop[] = [];
  if (!(dt > 0 && aspect > 0)) return drops;
  const inside = (v: number) => v > 0.002 && v < 0.998;
  const along = (lo: number, hi: number) => [clamp(lo, 0, 1), clamp(hi, 0, 1)] as const;

  const edges = [
    // [position, outward normal speed in image heights per second, extent along the edge in uv, vertical?]
    {
      at: box.left,
      speed: -motion.left * aspect,
      span: along(box.top, box.bottom),
      vertical: true,
      sign: -1,
    },
    {
      at: box.right,
      speed: motion.right * aspect,
      span: along(box.top, box.bottom),
      vertical: true,
      sign: 1,
    },
    {
      at: box.top,
      speed: -motion.top,
      span: along(box.left, box.right),
      vertical: false,
      sign: -1,
    },
    {
      at: box.bottom,
      speed: motion.bottom,
      span: along(box.left, box.right),
      vertical: false,
      sign: 1,
    },
  ];
  for (const edge of edges) {
    if (!inside(edge.at) || Math.abs(edge.speed) < minSpeed) continue;
    const [lo, hi] = edge.span;
    // The length of the edge in image heights.
    const length = (hi - lo) * (edge.vertical ? 1 : aspect);
    if (!(length > 0)) continue;
    const count = clamp(Math.ceil(length / (radius * 1.6)), 1, maxPerEdge);
    const r = Math.max(radius, length / (count * 1.4));
    const saturated = maxSpeed * Math.tanh(edge.speed / maxSpeed);
    const strength = gain * saturated * dt;
    for (let i = 0; i < count; i++) {
      const t = lo + ((i + 0.5) / count) * (hi - lo);
      // Just outside the wall, so most of the bump lands in water; the solver drops what falls on the solid.
      const offset = r * 0.6;
      drops.push(
        edge.vertical
          ? { x: edge.at + (edge.sign * offset) / aspect, y: t, radius: r, strength }
          : { x: t, y: edge.at + edge.sign * offset, radius: r, strength },
      );
    }
  }
  return drops;
}

// --- The tracker -------------------------------------------------------------------------------------

export interface DomRectsOptions {
  /** Where to look for marked elements. See "WHERE IT LOOKS" above. */
  root?: Element | null;
  /** Re-run the element query every this many updates. Default 60 (about once a second). */
  requeryEvery?: number;
  /** Re-read computed style every this many updates. Default 20. */
  styleEvery?: number;
  /** Grow every box by this many CSS pixels (negative shrinks). Default 0. Read every update: it may be changed live. */
  outsetPx?: number;
  /** Skip boxes with a side shorter than this many CSS pixels. Default 2. Read every update. */
  minSizePx?: number;
  /**
   * Smallest movement (uv) that makes `changed` true. Default 1e-4. Read every update; an effect
   * passes a fraction of a simulation cell so that sub-cell drift costs no re-rasterising.
   */
  epsilon?: number;
  /** Selector for obstacles. Default `OBSTACLE_SELECTOR`. */
  selector?: string;
}

export interface DomRectsResult {
  /** Obstacles overlapping the canvas, in DOM order. The same array object until something changes. */
  readonly rects: readonly ObstacleRect[];
  /** The same obstacles with their unclipped edges and motion (a fresh array each update). */
  readonly obstacles: readonly Obstacle[];
  /** `rects` differs from the previous update (or this is the first one). */
  readonly changed: boolean;
  /** Marked elements found under the root, whether or not they overlap the canvas. */
  readonly found: number;
}

export interface DomRectTracker {
  /** Reads the page. Call once per frame, before writing to it; `deltaSeconds` is the frame time. */
  update(deltaSeconds: number): DomRectsResult;
  /** The element the search is rooted at (undefined without a DOM). */
  readonly root: Element | undefined;
  /** Forces the element list to be rebuilt on the next update. */
  invalidate(): void;
  destroy(): void;
}

const EMPTY: DomRectsResult = Object.freeze({
  rects: Object.freeze([] as ObstacleRect[]),
  obstacles: Object.freeze([] as Obstacle[]),
  changed: false,
  found: 0,
});

interface Entry {
  element: Element;
  radiusPx: number;
  visible: boolean;
  styleAge: number;
  width: number;
  height: number;
  previous: EdgeBox | undefined;
}

function canvasHasDom(canvas: unknown): canvas is HTMLCanvasElement {
  const node = canvas as Partial<HTMLCanvasElement> | null;
  return (
    typeof document !== 'undefined' &&
    typeof globalThis.getComputedStyle === 'function' &&
    typeof node === 'object' &&
    node !== null &&
    typeof node.getBoundingClientRect === 'function' &&
    typeof node.closest === 'function' &&
    'parentElement' in node
  );
}

/** Step 2 to 4 of "WHERE IT LOOKS". Exported for tests that build their own small trees. */
export function findObstacleRoot(canvas: HTMLCanvasElement, selector = OBSTACLE_SELECTOR): Element {
  const scoped = canvas.closest(`[${OBSTACLE_ROOT_ATTRIBUTE}]`);
  if (scoped) return scoped;
  let node = canvas.parentElement;
  while (node && node !== document.body && node !== document.documentElement) {
    if (node.querySelector(selector)) return node;
    node = node.parentElement;
  }
  return canvas.parentElement ?? canvas;
}

export function trackDomRects(
  canvas: HTMLCanvasElement,
  options: DomRectsOptions = {},
): DomRectTracker {
  if (!canvasHasDom(canvas)) {
    return { update: () => EMPTY, root: undefined, invalidate() {}, destroy() {} };
  }
  const requeryEvery = Math.max(1, Math.round(options.requeryEvery ?? 60));
  const styleEvery = Math.max(1, Math.round(options.styleEvery ?? 20));
  const selector = options.selector ?? OBSTACLE_SELECTOR;

  const entries = new Map<Element, Entry>();
  let list: Entry[] = [];
  let root: Element | undefined;
  let observer: MutationObserver | undefined;
  let resolvedParent: Element | null | undefined;
  let sinceQuery = requeryEvery;
  let dirty = true;
  let destroyed = false;
  let lastRects: readonly ObstacleRect[] | undefined;

  function observe(next: Element) {
    if (root === next) return;
    root = next;
    observer?.disconnect();
    if (typeof MutationObserver === 'undefined') return;
    observer ??= new MutationObserver(() => {
      dirty = true;
    });
    observer.observe(next, {
      childList: true,
      subtree: true,
      attributes: true,
      attributeFilter: [OBSTACLE_ATTRIBUTE],
    });
  }

  function requery() {
    observe(options.root ?? findObstacleRoot(canvas, selector));
    resolvedParent = canvas.parentElement;
    const seen = new Set<Element>();
    const next: Entry[] = [];
    for (const element of root ? root.querySelectorAll(selector) : []) {
      // An ancestor of the canvas covers it entirely and would wall the whole pool in.
      if (element === canvas || element.contains(canvas)) continue;
      seen.add(element);
      let entry = entries.get(element);
      if (!entry) {
        entry = {
          element,
          radiusPx: 0,
          visible: true,
          styleAge: styleEvery,
          width: -1,
          height: -1,
          previous: undefined,
        };
        entries.set(element, entry);
      }
      next.push(entry);
    }
    for (const element of entries.keys()) if (!seen.has(element)) entries.delete(element);
    list = next;
    sinceQuery = 0;
    dirty = false;
  }

  function refreshStyle(entry: Entry, box: DOMRect) {
    const style = globalThis.getComputedStyle(entry.element);
    entry.visible = isVisibleStyle(style);
    entry.radiusPx = maxCornerRadiusPx(style, box.width, box.height);
    entry.styleAge = 0;
  }

  return {
    get root() {
      return root;
    },
    invalidate() {
      dirty = true;
    },
    update(deltaSeconds) {
      if (destroyed) return EMPTY;
      if (dirty || ++sinceQuery >= requeryEvery || canvas.parentElement !== resolvedParent) {
        requery();
      }
      const dt = Number.isFinite(deltaSeconds) ? Math.max(0, deltaSeconds) : 0;

      // All reads, no writes: the canvas box, then one box per element, then (rarely) style.
      const canvasBox = canvas.getBoundingClientRect();
      const obstacles: Obstacle[] = [];
      for (const entry of list) {
        const box = entry.element.getBoundingClientRect();
        const resized =
          Math.abs(box.width - entry.width) > 0.5 || Math.abs(box.height - entry.height) > 0.5;
        if (resized || ++entry.styleAge >= styleEvery) refreshStyle(entry, box);
        entry.width = box.width;
        entry.height = box.height;
        if (!entry.visible) {
          entry.previous = undefined;
          continue;
        }
        // Read from `options` every time, so a caller can change the outset while it runs.
        const obstacle = boxToObstacle(box, canvasBox, {
          radiusPx: entry.radiusPx,
          outsetPx: options.outsetPx ?? 0,
          minSizePx: options.minSizePx ?? 2,
        });
        if (!obstacle) {
          entry.previous = undefined;
          continue;
        }
        const motion = entry.previous ? edgeMotion(entry.previous, obstacle.box, dt) : undefined;
        entry.previous = obstacle.box;
        const outset = options.outsetPx ?? 0;
        obstacle.exact =
          outset === 0
            ? obstacle.rect
            : (boxToObstacle(box, canvasBox, {
                radiusPx: entry.radiusPx,
                outsetPx: 0,
                minSizePx: options.minSizePx ?? 2,
              })?.rect ?? obstacle.rect);
        obstacle.element = entry.element;
        obstacles.push(motion ? { ...obstacle, motion } : obstacle);
      }

      const rects = obstacles.map((o) => o.rect);
      const changed = lastRects === undefined || rectsChanged(lastRects, rects, options.epsilon);
      if (changed) lastRects = rects;
      return { rects: lastRects!, obstacles, changed, found: list.length };
    },
    destroy() {
      destroyed = true;
      observer?.disconnect();
      observer = undefined;
      entries.clear();
      list = [];
      root = undefined;
    },
  };
}
