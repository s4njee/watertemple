import { effect } from '../../effects/particles/index.ts';
import { LANTERN_NOZZLE_TOP } from '../../effects/particles/lantern-layout.ts';
import { createGpuSession } from '../../gpu/device.ts';
import { mountEffect, type EffectHost } from '../../shell/host.ts';
import { hexToColor } from '../../shell/params-url.ts';
import './page.css';

function byId<T extends HTMLElement>(id: string): T {
  const element = document.getElementById(id);
  if (!element) throw new Error(`Lantern demo: missing #${id}`);
  return element as T;
}

interface Lantern {
  stage: HTMLElement;
  host?: EffectHost;
  loaded: boolean;
  hdr: boolean;
}
const lanterns: Lantern[] = [
  { stage: byId('stage'), loaded: false, hdr: false },
  { stage: byId('stage-right'), loaded: false, hdr: false },
];
for (const scene of document.querySelectorAll<HTMLElement>('.lantern-scene')) {
  scene.style.setProperty('--nozzle-top', String(LANTERN_NOZZLE_TOP));
}
const status = byId('status');
const pause = byId<HTMLButtonElement>('pause');
const reset = byId<HTMLButtonElement>('reset');
const intensity = byId<HTMLInputElement>('intensity');
const headroom = byId<HTMLInputElement>('headroom');
const headroomValue = byId<HTMLOutputElement>('headroom-value');
const outputMode = byId<HTMLSelectElement>('output-mode');
const display = matchMedia('(dynamic-range: high)');
const motion = matchMedia('(prefers-reduced-motion: reduce)');
let playing = !motion.matches;
const settings: Record<string, unknown> = {
  ...effect.presets!.lantern,
  pageColor: hexToColor('#12130f'),
  headroom: Number(headroom.value),
};
const session = await createGpuSession();

function render() {
  const ready = session.state.ok && lanterns.every((lantern) => lantern.loaded);
  const hdr = ready && lanterns.every((lantern) => lantern.hdr);
  document.body.dataset.state = ready ? (playing ? 'live' : 'paused') : 'fallback';
  document.body.dataset.output = hdr ? 'hdr' : 'sdr';
  const range = hdr ? 'HDR' : outputMode.value === 'sdr' ? 'SDR' : 'SDR fallback';
  status.textContent = ready ? `${range} · ${playing ? 'live' : 'paused'}` : 'Still lanterns';
  status.title = hdr
    ? 'Extended-range HDR canvas is active'
    : 'HDR needs a supported display and browser';
  headroomValue.textContent = `${Number(headroom.value).toFixed(1)}×`;
  pause.textContent = playing ? 'Pause' : 'Play';
  pause.disabled = !ready;
  reset.disabled = !ready;
  intensity.disabled = !ready;
  headroom.disabled = !ready || !hdr;
}

function mount() {
  for (const lantern of lanterns) {
    lantern.host?.destroy();
    lantern.loaded = false;
    lantern.hdr = false;
  }
  render();
  for (const lantern of lanterns) {
    const host = mountEffect(session, lantern.stage, 'particles', {
      initialParams: settings,
      paused: !playing,
      maxDpr: 1.5,
      allowHdr: outputMode.value !== 'sdr',
    });
    lantern.host = host;
    host.onLoad(() => {
      lantern.hdr =
        host.canvas.getContext('webgpu')?.getConfiguration?.()?.toneMapping?.mode === 'extended';
      lantern.loaded = true;
      render();
    });
  }
}

const stopSession = session.onChange((state) => {
  if (!state.ok) for (const lantern of lanterns) lantern.loaded = false;
  render();
});
mount();

pause.addEventListener('click', () => {
  playing = !playing;
  for (const lantern of lanterns) {
    if (playing) lantern.host?.resume();
    else lantern.host?.pause();
  }
  render();
});
reset.addEventListener('click', mount);
function redrawIfPaused() {
  if (playing) return;
  for (const lantern of lanterns) {
    const mounted = lantern.host;
    if (!mounted?.loaded) continue;
    const stop = mounted.onStats(() => {
      stop();
      mounted.pause();
    });
    mounted.resume();
  }
}
function set(key: string, value: number) {
  settings[key] = value;
  for (const lantern of lanterns) {
    if (lantern.host?.loaded) lantern.host.loaded.params[key] = value;
  }
  redrawIfPaused();
}
intensity.addEventListener('input', () => {
  set('brightness', Number(intensity.value));
});
headroom.addEventListener('input', () => {
  set('headroom', Number(headroom.value));
  render();
});
outputMode.addEventListener('change', mount);
const onDisplay = () => {
  if (outputMode.value !== 'sdr') mount();
};
display.addEventListener('change', onDisplay);

const onMotion = () => {
  if (!motion.matches) return;
  playing = false;
  for (const lantern of lanterns) lantern.host?.pause();
  render();
};
motion.addEventListener('change', onMotion);

byId('layer-toggle').addEventListener('click', (event) => {
  const enabled = document.body.classList.toggle('show-layers');
  (event.currentTarget as HTMLButtonElement).setAttribute('aria-pressed', String(enabled));
});
byId('move-card').addEventListener('click', (event) => {
  const button = event.currentTarget as HTMLButtonElement;
  const moved = button.closest('.card')!.classList.toggle('moved');
  button.setAttribute('aria-pressed', String(moved));
});

addEventListener('pagehide', (event) => {
  if (event.persisted) return;
  motion.removeEventListener('change', onMotion);
  display.removeEventListener('change', onDisplay);
  stopSession();
  for (const lantern of lanterns) lantern.host?.destroy();
  session.destroy();
});
