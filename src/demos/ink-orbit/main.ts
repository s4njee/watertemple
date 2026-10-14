// The "Loose orbit" page (ink-orbit.html): the ink-orbit effect fixed behind a real page, the way a
// site would use it. The page marks its headings, cards and links with `data-wt-obstacle`; the effect
// measures them and does the rest. Everything here is ordinary page code: mount the effect, pass the
// pointer on to it, and wire up the toolbar and the draggable card.

import { createGpuSession } from '../../gpu/device.ts';
import { OBSTACLE_SELECTOR } from '../../gpu/dom-rects.ts';
import { mountEffect, type EffectHost } from '../../shell/host.ts';
import { hexToColor } from '../../shell/params-url.ts';
import { ORBIT_VARIATIONS, orbitVariation } from '../../effects/orbit-variants/catalog.ts';
import './page.css';

const EFFECT = orbitVariation(document.body.dataset.effect);
const variation = ORBIT_VARIATIONS[EFFECT];
if (variation.style > 0) {
  document.documentElement.style.setProperty('--paper', variation.background);
  document.documentElement.style.setProperty('--accent', variation.accent);
}
const ORBIT_ATTRIBUTE = 'data-wt-orbit';
/** Room kept clear at the bottom of the card's section for the toolbar, px. */
const TOOLBAR_CLEARANCE = 140;

function byId<T extends HTMLElement>(id: string): T {
  const node = document.getElementById(id);
  if (!node) throw new Error(`ink-orbit page: #${id} is missing`);
  return node as T;
}

if (import.meta.env.DEV) {
  const { installShaderOverlay } = await import('../../gpu/shader-overlay.ts');
  installShaderOverlay();
}

const stage = byId('stage');
const status = byId('status');
const detail = byId('detail');
const pauseButton = byId<HTMLButtonElement>('pause');
const resetButton = byId<HTMLButtonElement>('reset');
const forcesToggle = byId<HTMLInputElement>('forces');
const boxesButton = byId<HTMLButtonElement>('show-boxes');
const reverseButton = byId<HTMLButtonElement>('reverse');
const paletteSelect = byId<HTMLSelectElement>('palette');
const variationSelect = byId<HTMLSelectElement>('variation');
const lightSlider = byId<HTMLInputElement>('brightness');
const lightValue = byId<HTMLOutputElement>('brightness-value');
const outputSelect = byId<HTMLSelectElement>('output-mode');
const card = byId('floating-card');
const handle = byId<HTMLButtonElement>('drag-handle');
const footnote = byId('card-footnote');
const notes = byId('notes');

const reducedMotion = matchMedia('(prefers-reduced-motion: reduce)').matches;
const paper = hexToColor(getComputedStyle(document.documentElement).getPropertyValue('--paper'));

// --- The effect ------------------------------------------------------------------------------------

/** What the page asks of the effect. Kept here so it survives a Reset (which re-mounts it). */
const settings: Record<string, unknown> = {
  // The page's dark surface; light is emitted by the trails themselves.
  ...(paper ? { paper } : {}),
  vignette: 0,
  ground: 'dark',
  palette: 'rgb cycle',
  gain: 1.4,
  sheen: 0,
  // More, smaller luminous trails across the page.
  heads: 20,
  // Leave room for additive overlaps without washing the colours out.
  ink: 0.55,
  dyeResolution: '768',
  demo: 'never',
  // On an HDR display the glow is this many times SDR white (the "Light" slider).
  paperLight: Number(lightSlider.value),
  ...variation.params,
};
/** The elements' pull on the heads and the slow current round them, while "Forces" is on. */
const FORCES = {
  pull: Number(variation.params.pull ?? 1),
  current: Number(variation.params.current ?? 0.05),
};

variationSelect.replaceChildren(
  ...Object.entries(ORBIT_VARIATIONS).map(
    ([id, item]) => new Option(item.title, id, false, id === EFFECT),
  ),
);
paletteSelect.replaceChildren(
  ...variation.palettes.map(([id, label]) => new Option(label, id, false, id === settings.palette)),
);

const session = await createGpuSession();
let host: EffectHost | undefined;
let playing = !reducedMotion;
/** The canvas is an extended-range one, allowing brighter RGB light. */
let hdr = false;
let lanesInView = 0;

/** Whether the host gave the effect an HDR canvas: it does where the display and browser have one. */
function canvasIsHdr(canvas: HTMLCanvasElement): boolean {
  try {
    return canvas.getContext('webgpu')?.getConfiguration?.()?.toneMapping?.mode === 'extended';
  } catch {
    return false;
  }
}

function mount() {
  host?.destroy();
  host = mountEffect(session, stage, EFFECT, {
    initialParams: settings,
    paused: !playing,
    allowHdr: outputSelect.value !== 'sdr',
  });
  const mounted = host;
  // The effect is rebuilt after a device loss (and on hot reload); the settings are carried over by
  // the host, so there is nothing to re-apply, only the output to look at and the toolbar to enable.
  host.onLoad(() => {
    hdr = canvasIsHdr(mounted.canvas);
    // The shader keeps the measured plates clear of the glow to their exact outlines.
    document.body.dataset.output = hdr ? 'hdr' : 'sdr';
    set('cutout', hdr || variation.style > 0);
    lightSlider.disabled = !hdr;
    pauseButton.disabled = false;
    resetButton.disabled = false;
    render();
  });
}

/** Changes a setting now and for every later re-mount. */
function set(key: string, value: unknown) {
  settings[key] = value;
  const params = host?.loaded?.params;
  if (params) params[key] = value as (typeof params)[string];
  // A paused canvas shows the frame it stopped on; nothing to redraw until it plays again.
}

function render() {
  const ok = session.state.ok;
  document.body.dataset.state = !ok ? 'fallback' : playing ? 'live' : 'paused';
  status.textContent = !ok
    ? 'WebGPU is unavailable'
    : `${variation.title} is ${playing ? 'live' : 'paused'}`;
  pauseButton.textContent = playing ? 'Pause' : 'Play';
  lightValue.textContent = `${Number(lightSlider.value)}×`;
  if (!ok) {
    detail.textContent = 'Showing the page without the fluid';
    pauseButton.disabled = true;
    resetButton.disabled = true;
    return;
  }
  const output = hdr ? 'HDR' : outputSelect.value === 'sdr' ? 'SDR' : 'SDR (no HDR display)';
  detail.textContent = `${output} · ${lanesInView} ${lanesInView === 1 ? 'element' : 'elements'} pulling`;
}

// The effect measures in canvas heights, a page thinks in pixels: keep the orbit a fixed distance from
// the text and the trails a fixed width, whatever the height of the window.
function fit() {
  const height = Math.max(320, innerHeight);
  set('shell', Math.min(0.1, variation.shell / height));
  set('inkRadius', Math.min(0.08, variation.width / height));
  set('radius', Math.min(0.08, 30 / height));
}

session.onChange(render);
fit();
mount();
render();
addEventListener('resize', fit);

// How many elements are in the viewport and pulling: what the heads are working with right now.
function countLanes() {
  if (!session.state.ok) return;
  let lanes = 0;
  for (const node of document.querySelectorAll(OBSTACLE_SELECTOR)) {
    if (node.getAttribute(ORBIT_ATTRIBUTE) === 'off') continue;
    const box = node.getBoundingClientRect();
    if (box.bottom > 0 && box.top < innerHeight && box.right > 0 && box.left < innerWidth) lanes++;
  }
  lanesInView = lanes;
  render();
}
let countQueued = false;
function queueCount() {
  if (countQueued) return;
  countQueued = true;
  requestAnimationFrame(() => {
    countQueued = false;
    countLanes();
  });
}
addEventListener('scroll', queueCount, { passive: true });
addEventListener('resize', queueCount);
queueCount();

// --- The pointer ---------------------------------------------------------------------------------
// The canvas lies behind the page and never receives the pointer itself, and the effect listens only
// on its canvas. So the page passes movement on: hovering anywhere stirs the fluid, and a press on
// open paper paints. Presses on text and controls are left alone, so selecting and clicking work.

function forward(event: PointerEvent, type = event.type) {
  const canvas = host?.canvas;
  if (!canvas || event.target === canvas) return;
  canvas.dispatchEvent(
    new PointerEvent(type, {
      pointerId: event.pointerId,
      pointerType: event.pointerType,
      isPrimary: event.isPrimary,
      clientX: event.clientX,
      clientY: event.clientY,
      pressure: event.pressure,
      button: event.button,
      buttons: event.buttons,
    }),
  );
}

const onPaper = (target: EventTarget | null) =>
  !(target instanceof Element && target.closest('.solid, a, button, input, select, label'));

addEventListener('pointermove', (event) => forward(event), { passive: true });
addEventListener('pointerdown', (event) => {
  if (onPaper(event.target)) forward(event);
});
addEventListener('pointerup', (event) => forward(event));
addEventListener('pointercancel', (event) => forward(event));
document.documentElement.addEventListener('pointerleave', (event) => forward(event));

// --- The toolbar -----------------------------------------------------------------------------------

pauseButton.addEventListener('click', () => {
  playing = !playing;
  if (playing) host?.resume();
  else host?.pause();
  render();
});

// A new fluid: clear water and a fresh set of heads.
resetButton.addEventListener('click', () => {
  mount();
  render();
});

// Off: the heads feel no element and only drift, which is what is underneath the orbits.
forcesToggle.addEventListener('change', () => {
  set('pull', forcesToggle.checked ? FORCES.pull : 0);
  set('current', forcesToggle.checked ? FORCES.current : 0);
});

boxesButton.addEventListener('click', () => {
  const shown = document.body.classList.toggle('show-boxes');
  boxesButton.setAttribute('aria-pressed', String(shown));
});

function describeCard() {
  const way = card.getAttribute(ORBIT_ATTRIBUTE) === 'ccw' ? 'COUNTER-CLOCKWISE' : 'CLOCKWISE';
  footnote.textContent = `TURNING ${way} / DRAG TO MOVE`;
}

// Every element swings its heads the other way: the effect reads the attribute each frame.
reverseButton.addEventListener('click', () => {
  const reversed = reverseButton.getAttribute('aria-pressed') !== 'true';
  reverseButton.setAttribute('aria-pressed', String(reversed));
  for (const node of document.querySelectorAll(`[${ORBIT_ATTRIBUTE}]`)) {
    const way = node.getAttribute(ORBIT_ATTRIBUTE);
    if (way === 'cw') node.setAttribute(ORBIT_ATTRIBUTE, 'ccw');
    else if (way === 'ccw') node.setAttribute(ORBIT_ATTRIBUTE, 'cw');
  }
  describeCard();
});

// Separate pages keep their own layout and defaults. Relative URLs also work under /watertemple/.
variationSelect.addEventListener('change', () => {
  const page = ORBIT_VARIATIONS[orbitVariation(variationSelect.value)].page;
  location.assign(new URL(page, location.href));
});

paletteSelect.addEventListener('change', () => set('palette', paletteSelect.value));

lightSlider.addEventListener('input', () => {
  set('paperLight', Number(lightSlider.value));
  render();
});

// HDR or not is decided when the canvas is configured, so changing it means a new canvas (and a new
// fluid). The same when the window moves to a display with a different range.
outputSelect.addEventListener('change', () => {
  mount();
  render();
});
matchMedia('(dynamic-range: high)').addEventListener('change', () => {
  if (outputSelect.value === 'sdr') return;
  mount();
  render();
});

byId('rearrange').addEventListener('click', () => {
  notes.classList.toggle('rearranged');
  queueCount();
});

// --- The card --------------------------------------------------------------------------------------
// Moved with a transform, by the handle or the arrow keys, and kept inside its section. The effect
// needs no telling: it measures the card every frame and sees it move.

const offset = { x: 0, y: 0 };

function moveCard(x: number, y: number) {
  const section = card.parentElement;
  if (!section) return;
  // Where the card sits with no offset, relative to its section.
  const left = card.offsetLeft;
  const top = card.offsetTop;
  const maxX = section.clientWidth - card.offsetWidth - 12 - left;
  const maxY = section.clientHeight - card.offsetHeight - TOOLBAR_CLEARANCE - top;
  offset.x = Math.max(12 - left, Math.min(maxX, x));
  offset.y = Math.max(12 - top, Math.min(Math.max(12 - top, maxY), y));
  card.style.transform = `translate(${offset.x}px, ${offset.y}px)`;
}

let drag: { id: number; x: number; y: number; fromX: number; fromY: number } | undefined;

handle.addEventListener('pointerdown', (event) => {
  drag = {
    id: event.pointerId,
    x: event.clientX,
    y: event.clientY,
    fromX: offset.x,
    fromY: offset.y,
  };
  try {
    handle.setPointerCapture(event.pointerId);
  } catch {
    // A pointer that is already gone cannot be captured; the drag simply ends when it leaves.
  }
  event.preventDefault();
});
handle.addEventListener('pointermove', (event) => {
  if (!drag || event.pointerId !== drag.id) return;
  moveCard(drag.fromX + event.clientX - drag.x, drag.fromY + event.clientY - drag.y);
});
const endDrag = (event: PointerEvent) => {
  if (drag && event.pointerId === drag.id) drag = undefined;
};
handle.addEventListener('pointerup', endDrag);
handle.addEventListener('pointercancel', endDrag);
handle.addEventListener('keydown', (event) => {
  const step = event.shiftKey ? 40 : 16;
  const move: Record<string, [number, number]> = {
    ArrowLeft: [-step, 0],
    ArrowRight: [step, 0],
    ArrowUp: [0, -step],
    ArrowDown: [0, step],
  };
  const delta = move[event.key];
  if (!delta) return;
  event.preventDefault();
  moveCard(offset.x + delta[0], offset.y + delta[1]);
});
addEventListener('resize', () => moveCard(offset.x, offset.y));

addEventListener('pagehide', (event) => {
  if (event.persisted) return;
  host?.destroy();
  session.destroy();
});
