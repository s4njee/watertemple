import { createGpuSession } from '../../gpu/device.ts';
import { mountEffect, type EffectHost } from '../../shell/host.ts';
import '../ink-orbit/page.css';
import './page.css';
const byId = <T extends HTMLElement>(id: string) => {
  const node = document.getElementById(id);
  if (!node) throw new Error(`Life page: #${id} is missing`);
  return node as T;
};
if (import.meta.env.DEV) {
  const { installShaderOverlay } = await import('../../gpu/shader-overlay.ts');
  installShaderOverlay();
}
const stage = byId('stage'),
  status = byId('status'),
  detail = byId('detail');
const pause = byId<HTMLButtonElement>('pause'),
  step = byId<HTMLButtonElement>('step'),
  reset = byId<HTMLButtonElement>('reset');
const speed = byId<HTMLInputElement>('speed'),
  pattern = byId<HTMLSelectElement>('pattern'),
  size = byId<HTMLSelectElement>('cell-size');
const card = byId('floating-card'),
  handle = byId<HTMLButtonElement>('drag-handle');
const output = byId<HTMLSelectElement>('output-mode'),
  light = byId<HTMLInputElement>('brightness');
const session = await createGpuSession();
let playing = !matchMedia('(prefers-reduced-motion: reduce)').matches,
  hdr = false,
  host: EffectHost | undefined;
const settings: Record<string, string | number | boolean> = {
  running: playing,
  paperLight: 2,
  speed: 12,
  pattern: 'soup',
  cellSize: 7,
  seed: 1,
  singleStep: 0,
  plantRequest: 0,
};
function set(key: string, value: string | number | boolean) {
  settings[key] = value;
  if (host?.loaded) host.loaded.params[key] = value;
}
function render() {
  const ready = session.state.ok && !!host?.loaded;
  document.body.dataset.state = !session.state.ok ? 'fallback' : playing ? 'live' : 'paused';
  const statusText = !session.state.ok
    ? 'WebGPU is unavailable'
    : ready
      ? `Life is ${playing ? 'evolving' : 'paused'}`
      : 'Starting Life…';
  if (status.textContent !== statusText) status.textContent = statusText;
  detail.textContent = ready
    ? `Generation ${host!.loaded!.params.generation} · ${hdr ? 'HDR' : 'SDR'} · seeds every 18s`
    : 'Conway’s Game of Life';
  pause.textContent = playing ? 'Pause' : 'Play';
  pause.disabled = step.disabled = reset.disabled = !ready;
  byId('speed-value').textContent = `${speed.value}/s`;
  byId('brightness-value').textContent = `${light.value}×`;
  light.disabled = !hdr;
}
// Resume for exactly one rendered frame when changing a paused world.
function redraw() {
  if (!playing) host?.resume();
}
function mount() {
  host?.destroy();
  host = mountEffect(session, stage, 'game-of-life', {
    initialParams: settings,
    paused: !playing,
    allowHdr: output.value !== 'sdr',
  });
  const current = host;
  host.onLoad(() => {
    try {
      hdr =
        current.canvas.getContext('webgpu')?.getConfiguration?.()?.toneMapping?.mode === 'extended';
    } catch {
      hdr = false;
    }
    document.body.dataset.output = hdr ? 'hdr' : 'sdr';
    render();
  });
  host.onStats(() => {
    if (!playing) current.pause();
    render();
  });
}
mount();
output.addEventListener('change', () => {
  mount();
  render();
});
light.addEventListener('input', () => {
  set('paperLight', Number(light.value));
  redraw();
  render();
});
matchMedia('(dynamic-range: high)').addEventListener('change', () => {
  if (output.value !== 'sdr') {
    mount();
    render();
  }
});
session.onChange(render);
render();
pause.addEventListener('click', () => {
  playing = !playing;
  set('running', playing);
  if (playing) host?.resume();
  else host?.pause();
  render();
});
step.addEventListener('click', () => {
  playing = false;
  set('running', false);
  set('singleStep', Number(settings.singleStep) + 1);
  host?.resume();
  render();
});
reset.addEventListener('click', () => {
  set('seed', Number(settings.seed) + 1);
  redraw();
});
speed.addEventListener('input', () => {
  set('speed', Number(speed.value));
  render();
});
pattern.addEventListener('change', () => {
  set('pattern', pattern.value);
  redraw();
});
size.addEventListener('change', () => {
  set('cellSize', Number(size.value));
  redraw();
});
addEventListener('pointerdown', (event) => {
  if (
    event.button !== 0 ||
    (event.target instanceof Element &&
      event.target.closest('.solid, a, button, input, select, label'))
  )
    return;
  const rect = host?.canvas.getBoundingClientRect();
  if (!rect) return;
  set('plantX', (event.clientX - rect.left) / rect.width);
  set('plantY', (event.clientY - rect.top) / rect.height);
  set('plantRequest', Number(settings.plantRequest) + 1);
  redraw();
});
const offset = { x: 0, y: 0 };
function moveCard(x: number, y: number) {
  const section = card.parentElement;
  if (!section) return;
  const left = card.offsetLeft,
    top = card.offsetTop;
  offset.x = Math.max(12 - left, Math.min(section.clientWidth - card.offsetWidth - 12 - left, x));
  offset.y = Math.max(
    12 - top,
    Math.min(Math.max(12 - top, section.clientHeight - card.offsetHeight - 180 - top), y),
  );
  card.style.transform = `translate(${offset.x}px,${offset.y}px)`;
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
    /* The drag ends if the pointer has already left. */
  }
  event.preventDefault();
});
handle.addEventListener('pointermove', (event) => {
  if (drag && event.pointerId === drag.id)
    moveCard(drag.fromX + event.clientX - drag.x, drag.fromY + event.clientY - drag.y);
});
const endDrag = (event: PointerEvent) => {
  if (drag?.id === event.pointerId) drag = undefined;
};
handle.addEventListener('pointerup', endDrag);
handle.addEventListener('pointercancel', endDrag);
handle.addEventListener('keydown', (event) => {
  const step = event.shiftKey ? 40 : 16;
  const delta: Record<string, [number, number]> = {
    ArrowLeft: [-step, 0],
    ArrowRight: [step, 0],
    ArrowUp: [0, -step],
    ArrowDown: [0, step],
  };
  const d = delta[event.key];
  if (!d) return;
  event.preventDefault();
  moveCard(offset.x + d[0], offset.y + d[1]);
});
addEventListener('resize', () => moveCard(offset.x, offset.y));
addEventListener('pagehide', (event) => {
  if (!event.persisted) {
    host?.destroy();
    session.destroy();
  }
});
