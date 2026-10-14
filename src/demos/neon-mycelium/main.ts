import { createGpuSession } from '../../gpu/device.ts';
import { mountEffect, type EffectHost } from '../../shell/host.ts';
import '../ink-orbit/page.css';
import './page.css';

const byId = <T extends HTMLElement>(id: string) => {
  const node = document.getElementById(id);
  if (!node) throw new Error(`mycelium page: #${id} is missing`);
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
  reset = byId<HTMLButtonElement>('reset');
const growth = byId<HTMLInputElement>('growth'),
  growthValue = byId('growth-value');
const palette = byId<HTMLSelectElement>('palette'),
  output = byId<HTMLSelectElement>('output-mode');
const light = byId<HTMLInputElement>('brightness'),
  lightValue = byId('brightness-value');
const card = byId('floating-card'),
  handle = byId<HTMLButtonElement>('drag-handle');
const session = await createGpuSession();
const reduced = matchMedia('(prefers-reduced-motion: reduce)').matches;
let playing = !reduced,
  hdr = false,
  host: EffectHost | undefined;
const settings: Record<string, unknown> = { growth: 1, palette: 'electric', paperLight: 1.2 };
function set(key: string, value: string | number) {
  settings[key] = value;
  if (host?.loaded) host.loaded.params[key] = value;
}
function render() {
  const ok = session.state.ok;
  document.body.dataset.state = !ok ? 'fallback' : playing ? 'live' : 'paused';
  status.textContent = ok
    ? `Mycelium is ${playing ? 'growing' : 'paused'}`
    : 'WebGPU is unavailable';
  detail.textContent = ok
    ? `${hdr ? 'HDR' : 'SDR'} · following the page`
    : 'Showing a still neon garden';
  pause.textContent = playing ? 'Pause' : 'Grow';
  pause.disabled = !ok || !host?.loaded;
  reset.disabled = pause.disabled;
  growthValue.textContent = `${Number(growth.value)}×`;
  lightValue.textContent = `${Number(light.value)}×`;
  light.disabled = !hdr;
}
function mount() {
  host?.destroy();
  host = mountEffect(session, stage, 'neon-mycelium', {
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
}
session.onChange(render);
mount();
render();
pause.addEventListener('click', () => {
  playing = !playing;
  if (playing) host?.resume();
  else host?.pause();
  render();
});
reset.addEventListener('click', () => {
  mount();
  render();
});
growth.addEventListener('input', () => {
  set('growth', Number(growth.value));
  render();
});
palette.addEventListener('change', () => {
  set('palette', palette.value);
  mount();
  render();
});
light.addEventListener('input', () => {
  set('paperLight', Number(light.value));
  render();
});
output.addEventListener('change', () => {
  mount();
  render();
});
matchMedia('(dynamic-range: high)').addEventListener('change', () => {
  if (output.value !== 'sdr') {
    mount();
    render();
  }
});
byId('show-boxes').addEventListener('click', () => {
  const shown = document.body.classList.toggle('show-boxes');
  byId('show-boxes').setAttribute('aria-pressed', String(shown));
});
byId('rearrange').addEventListener('click', () => byId('notes').classList.toggle('rearranged'));
// Forward the pointer without capturing touch scroll or making the page text unselectable.
let inside = false;
function forward(event: PointerEvent, type = event.type) {
  host?.canvas.dispatchEvent(
    new PointerEvent(type, {
      pointerId: event.pointerId,
      pointerType: event.pointerType,
      clientX: event.clientX,
      clientY: event.clientY,
      button: event.button,
      buttons: event.buttons,
      pressure: event.pressure,
    }),
  );
}
addEventListener(
  'pointermove',
  (event) => {
    if (!inside) {
      forward(event, 'pointerenter');
      inside = true;
    }
    forward(event);
  },
  { passive: true },
);
addEventListener('pointerdown', (event) => {
  if (!(
    event.target instanceof Element &&
    event.target.closest('.solid, a, button, input, select, label')
  )) {
    if (!inside) {
      forward(event, 'pointerenter');
      inside = true;
    }
    forward(event);
  }
});
addEventListener('pointerup', (event) => forward(event));
addEventListener('pointercancel', (event) => forward(event));
document.documentElement.addEventListener('pointerleave', (event) => {
  inside = false;
  forward(event);
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
