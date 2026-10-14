import { OBSTACLE_ATTRIBUTE } from '../../gpu/dom-rects.ts';

// Demonstration obstacles for a canvas with no page around it (the bare Canvas layout): three small
// real DOM elements laid over the canvas, one of which the visitor can drag. They are ordinary
// elements marked with `data-wt-obstacle`, exactly what a site would write, and the effect treats
// them like any other page content. The container is inserted next to the canvas and removed on
// destroy; it ignores the pointer (the water still hears the mouse through it) except for the
// draggable card.
//
// Accessibility: the two static cards are `aria-hidden` decoration. The draggable one is a focusable
// group labelled for what it does and moves with the arrow keys as well as the pointer.

export interface DemoElements {
  /** The inserted container (a sibling of the canvas). */
  readonly container: HTMLElement;
  /** The elements marked as obstacles. */
  readonly count: number;
  destroy(): void;
}

interface CardSpec {
  label: string;
  /** Position and size as a percentage of the container. */
  left: number;
  top: number;
  width: number;
  height: number;
  radius: string;
  /** Size limits in px: a tiny canvas still shows a card, a huge one does not get a huge card. */
  minWidth: number;
  minHeight: number;
  maxWidth: number;
  maxHeight: number;
}

const CARDS: readonly CardSpec[] = [
  {
    label: 'Card',
    left: 7,
    top: 12,
    width: 30,
    height: 30,
    radius: '14px',
    minWidth: 72,
    minHeight: 28,
    maxWidth: 280,
    maxHeight: 120,
  },
  {
    label: 'Pill',
    left: 60,
    top: 62,
    width: 33,
    height: 14,
    radius: '999px',
    minWidth: 72,
    minHeight: 24,
    maxWidth: 300,
    maxHeight: 56,
  },
];
const DRAGGABLE: CardSpec = {
  label: 'Drag me',
  left: 41,
  top: 34,
  width: 22,
  height: 20,
  radius: '12px',
  minWidth: 64,
  minHeight: 26,
  maxWidth: 200,
  maxHeight: 76,
};

function styleCard(node: HTMLElement, spec: CardSpec) {
  Object.assign(node.style, {
    position: 'absolute',
    left: `${spec.left}%`,
    top: `${spec.top}%`,
    width: `${spec.width}%`,
    height: `${spec.height}%`,
    minWidth: `${spec.minWidth}px`,
    minHeight: `${spec.minHeight}px`,
    maxWidth: `${spec.maxWidth}px`,
    maxHeight: `${spec.maxHeight}px`,
    boxSizing: 'border-box',
    borderRadius: spec.radius,
    border: '1px solid var(--wt-border, rgba(127, 127, 127, 0.5))',
    background: 'color-mix(in srgb, var(--wt-surface, Canvas) 92%, transparent)',
    color: 'var(--wt-text, CanvasText)',
    font: '500 12px/1.2 system-ui, sans-serif',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    overflow: 'hidden',
    pointerEvents: 'none',
    userSelect: 'none',
  } satisfies Partial<CSSStyleDeclaration>);
}

/** Inserts the demonstration elements beside `canvas`. Returns undefined where there is no DOM. */
export function createDemoElements(canvas: HTMLCanvasElement): DemoElements | undefined {
  const parent = canvas.parentElement;
  if (typeof document === 'undefined' || !parent || typeof parent.append !== 'function') {
    return undefined;
  }
  const container = document.createElement('div');
  container.setAttribute('role', 'group');
  container.setAttribute('aria-label', 'Water obstacle demonstration');
  container.dataset.wtObstacleDemo = '';
  Object.assign(container.style, {
    position: 'absolute',
    inset: '0',
    overflow: 'hidden',
    pointerEvents: 'none',
    zIndex: '1',
  } satisfies Partial<CSSStyleDeclaration>);

  for (const spec of CARDS) {
    const node = document.createElement('div');
    node.setAttribute(OBSTACLE_ATTRIBUTE, '');
    node.setAttribute('aria-hidden', 'true');
    node.textContent = spec.label;
    styleCard(node, spec);
    container.append(node);
  }

  const handle = document.createElement('div');
  handle.setAttribute(OBSTACLE_ATTRIBUTE, '');
  handle.setAttribute('role', 'group');
  handle.setAttribute('tabindex', '0');
  handle.setAttribute(
    'aria-label',
    'Draggable card. Drag it, or use the arrow keys, to push the water around.',
  );
  handle.textContent = DRAGGABLE.label;
  styleCard(handle, DRAGGABLE);
  Object.assign(handle.style, {
    pointerEvents: 'auto',
    cursor: 'grab',
    touchAction: 'none',
  } satisfies Partial<CSSStyleDeclaration>);
  container.append(handle);

  // Position as a fraction of the container, so the card keeps its place when the canvas resizes.
  let fx = DRAGGABLE.left / 100;
  let fy = DRAGGABLE.top / 100;
  const place = () => {
    fx = Math.min(1, Math.max(0, fx));
    fy = Math.min(1, Math.max(0, fy));
    // Keep the card inside the container, whatever its size.
    handle.style.left = `min(${fx * 100}%, calc(100% - ${handle.offsetWidth || 0}px))`;
    handle.style.top = `min(${fy * 100}%, calc(100% - ${handle.offsetHeight || 0}px))`;
  };

  const controller = new AbortController();
  const { signal } = controller;
  let grab: { id: number; dx: number; dy: number } | undefined;
  handle.addEventListener(
    'pointerdown',
    (event) => {
      const box = handle.getBoundingClientRect();
      grab = { id: event.pointerId, dx: event.clientX - box.left, dy: event.clientY - box.top };
      try {
        handle.setPointerCapture(event.pointerId);
      } catch {
        // A pointer that is already gone (or a synthetic one) cannot be captured; dragging still works while it is over the card.
      }
      handle.style.cursor = 'grabbing';
      event.preventDefault();
    },
    { signal },
  );
  handle.addEventListener(
    'pointermove',
    (event) => {
      if (!grab || event.pointerId !== grab.id) return;
      const area = container.getBoundingClientRect();
      if (!(area.width > 0 && area.height > 0)) return;
      fx = (event.clientX - grab.dx - area.left) / area.width;
      fy = (event.clientY - grab.dy - area.top) / area.height;
      place();
    },
    { signal },
  );
  const release = (event: PointerEvent) => {
    if (grab && event.pointerId === grab.id) grab = undefined;
    handle.style.cursor = 'grab';
  };
  handle.addEventListener('pointerup', release, { signal });
  handle.addEventListener('pointercancel', release, { signal });
  handle.addEventListener(
    'keydown',
    (event) => {
      const step = event.shiftKey ? 0.1 : 0.03;
      const move: Record<string, [number, number]> = {
        ArrowLeft: [-step, 0],
        ArrowRight: [step, 0],
        ArrowUp: [0, -step],
        ArrowDown: [0, step],
      };
      const delta = move[event.key];
      if (!delta) return;
      event.preventDefault();
      fx += delta[0];
      fy += delta[1];
      place();
    },
    { signal },
  );

  parent.append(container);
  return {
    container,
    count: CARDS.length + 1,
    destroy() {
      controller.abort();
      container.remove();
    },
  };
}
