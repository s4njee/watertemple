import { Pane } from 'tweakpane';
import type { ParamDef, ParamSchema } from '../effects/types.ts';
import type { EffectHost, LoadedEffect } from './host.ts';
import {
  activePreset,
  colorToHex,
  hexToColor,
  paramQueryChanges,
  paramSignature,
  presetValues,
  valuesFromQuery,
  type Presets,
  type Values,
} from './params-url.ts';
import { onQueryChange, readQuery, updateQuery } from './router.ts';
import type { Unmount } from './types.ts';
import './controls.css';

/** Trailing delay for URL writes while a slider moves: browsers rate-limit history calls. */
const URL_DELAY_MS = 250;
/** Up to this many presets are buttons; more become a dropdown. */
const MAX_PRESET_BUTTONS = 5;

function el<K extends keyof HTMLElementTagNameMap>(
  tag: K,
  className: string,
  text?: string,
): HTMLElementTagNameMap[K] {
  const node = document.createElement(tag);
  node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

function button(label: string): HTMLButtonElement {
  const node = el('button', 'wt-button', label);
  node.type = 'button';
  return node;
}

/** `waveHeight` and `wave_height` both become "Wave height". */
function humanize(key: string): string {
  const words = key
    .replace(/([a-z\d])([A-Z])/g, '$1 $2')
    .replace(/[_-]+/g, ' ')
    .trim()
    .toLowerCase();
  return words.charAt(0).toUpperCase() + words.slice(1);
}

/** One slider step per 1/200 of the range, rounded to a power of ten, when the schema has none. */
function defaultStep(def: Extract<ParamDef, { type: 'number' }>): number {
  const raw = (def.max - def.min) / 200;
  return raw > 0 ? 10 ** Math.floor(Math.log10(raw)) : 1;
}

/** Writes `values` into the live object. Arrays are updated in place for effects that hold them. */
function assign(params: Values, values: Readonly<Values>) {
  for (const [key, value] of Object.entries(values)) {
    const target = params[key];
    if (Array.isArray(target) && Array.isArray(value)) {
      value.forEach((channel, index) => (target[index] = channel));
    } else {
      params[key] = Array.isArray(value) ? [...value] : value;
    }
  }
}

interface Session {
  effect: LoadedEffect['effect'];
  params: Values;
  schema: ParamSchema;
  presets: Presets;
  pane: Pane | undefined;
  /** What the pane binds to: the UI form of each value (colours in sRGB). */
  view: Values;
  /** Slider tracks, which Tweakpane leaves without a role or value for assistive technology. */
  sliders: Map<string, HTMLElement>;
}

export function mountControls(container: HTMLElement, host: EffectHost): Unmount {
  const root = el('section', 'wt-controls');
  root.setAttribute('aria-labelledby', 'wt-controls-title');
  const head = el('div', 'wt-controls__head');
  const title = el('h2', 'wt-label', 'Parameters');
  title.id = 'wt-controls-title';
  const actions = el('div', 'wt-controls__actions');
  const resetButton = button('Reset');
  const copyButton = button('Copy link');
  actions.append(resetButton, copyButton);
  head.append(title, actions);
  const presetsHost = el('div', 'wt-controls__presets');
  const paneHost = el('div', 'wt-controls__pane');
  const emptyNote = el('p', 'wt-muted', 'This effect has no parameters.');
  emptyNote.hidden = true;
  const linkField = el('input', 'wt-controls__link wt-data');
  linkField.readOnly = true;
  linkField.hidden = true;
  linkField.setAttribute('aria-label', 'Shareable link');
  const status = el('p', 'wt-controls__status wt-muted');
  status.setAttribute('role', 'status');
  root.append(head, presetsHost, paneHost, emptyNote, linkField, status);
  container.append(root);

  let session: Session | undefined;
  let firstLoad = true;
  let syncing = false;
  let writeTimer: number | undefined;
  let statusTimer: number | undefined;
  let copiedTimer: number | undefined;
  // The parameter part of the URL as last written or seen; other keys changing are not our business.
  let seen = paramSignature(readQuery());
  let presetUi: { sync(active: string | undefined): void } | undefined;

  function announce(message: string) {
    status.textContent = '';
    // A frame later, so a repeated message is still announced by screen readers.
    requestAnimationFrame(() => (status.textContent = message));
    clearTimeout(statusTimer);
    statusTimer = window.setTimeout(() => (status.textContent = ''), 4000);
  }

  function writeNow() {
    clearTimeout(writeTimer);
    writeTimer = undefined;
    if (!session) return;
    const { schema, presets, params } = session;
    updateQuery(paramQueryChanges(schema, presets, params, readQuery()), {
      replace: true,
      silent: true,
    });
    seen = paramSignature(readQuery());
  }

  function scheduleWrite() {
    linkField.hidden = true;
    clearTimeout(writeTimer);
    writeTimer = window.setTimeout(writeNow, URL_DELAY_MS);
  }

  function syncPresetUi() {
    if (!session) return;
    presetUi?.sync(activePreset(session.schema, session.presets, session.params));
  }

  /** Pushes the live values into the pane, e.g. after a preset, reset or history navigation. */
  function refreshPane() {
    if (!session) return;
    const { schema, params, view, pane } = session;
    for (const [key, def] of Object.entries(schema)) {
      const value = params[key];
      // Hex strings: the picker's text field then reads `#0d4a66`, and 8-bit rounding is built in.
      view[key] = def.type === 'color' && Array.isArray(value) ? `#${colorToHex(value)}` : value;
    }
    syncing = true;
    pane?.refresh();
    syncing = false;
    syncAria();
    syncPresetUi();
  }

  function syncAria() {
    for (const [key, track] of session?.sliders ?? []) {
      track.setAttribute('aria-valuenow', String(session?.params[key]));
    }
  }

  function setValues(values: Readonly<Values>) {
    if (!session) return;
    linkField.hidden = true;
    assign(session.params, values);
    refreshPane();
  }

  function edited(key: string, def: ParamDef, raw: unknown) {
    if (!session || syncing) return;
    let value = raw;
    if (def.type === 'color') {
      // Already what the link can carry, so the picture matches what a shared link reproduces.
      const color = hexToColor(raw as string);
      if (!color) return;
      value = color;
    } else if (def.type === 'number') {
      value = Math.min(def.max, Math.max(def.min, raw as number));
    }
    assign(session.params, { [key]: value });
    syncAria();
    syncPresetUi();
    scheduleWrite();
  }

  function buildPresets(current: Session) {
    presetsHost.replaceChildren();
    presetUi = undefined;
    const names = Object.keys(current.presets);
    if (names.length === 0) return;
    const apply = (name: string) => {
      setValues(presetValues(current.schema, current.presets[name]!));
      writeNow();
    };

    if (names.length <= MAX_PRESET_BUTTONS) {
      const group = el('div', 'wt-controls__group');
      group.setAttribute('role', 'group');
      group.setAttribute('aria-label', 'Presets');
      const buttons = names.map((name) => {
        const node = button(humanize(name));
        node.addEventListener('click', () => apply(name));
        group.append(node);
        return node;
      });
      presetsHost.append(el('span', 'wt-label', 'Presets'), group);
      presetUi = {
        sync: (active) =>
          buttons.forEach((node, index) =>
            node.setAttribute('aria-pressed', String(names[index] === active)),
          ),
      };
    } else {
      const select = el('select', 'wt-controls__select');
      select.id = 'wt-controls-preset';
      const none = new Option('Custom', '', true, true);
      none.disabled = true;
      select.append(none, ...names.map((name) => new Option(humanize(name), name)));
      select.addEventListener('change', () => apply(select.value));
      const label = el('label', 'wt-label', 'Preset');
      label.htmlFor = select.id;
      presetsHost.append(label, select);
      presetUi = { sync: (active) => (select.value = active ?? '') };
    }
  }

  function build({ effect, params }: LoadedEffect) {
    session?.pane?.dispose();
    paneHost.replaceChildren();
    const schema = effect.params;
    const presets: Presets = effect.presets ?? {};
    const current: Session = {
      effect,
      params,
      schema,
      presets,
      pane: undefined,
      view: {},
      sliders: new Map(),
    };
    session = current;

    // The page decodes only `p.*` before the first frame; a short `?preset=` link lands here.
    if (firstLoad) {
      firstLoad = false;
      assign(params, valuesFromQuery(schema, presets, readQuery()));
    }

    const entries = Object.entries(schema);
    paneHost.hidden = entries.length === 0;
    emptyNote.hidden = entries.length > 0;
    resetButton.disabled = entries.length === 0;

    if (entries.length > 0) {
      const pane = new Pane({ container: paneHost });
      current.pane = pane;
      refreshPane();
      for (const [key, def] of entries) {
        const label = def.label ?? humanize(key);
        const binding =
          def.type === 'number'
            ? pane.addBinding(current.view, key, {
                label,
                min: def.min,
                max: def.max,
                step: def.step ?? defaultStep(def),
              })
            : def.type === 'color'
              ? pane.addBinding(current.view, key, { label, view: 'color' })
              : def.type === 'select'
                ? pane.addBinding(current.view, key, {
                    label,
                    options: def.options.map((option) => ({ text: option, value: option })),
                  })
                : pane.addBinding(current.view, key, { label });
        binding.on('change', (event) => edited(key, def, event.value));
        // Tweakpane's labels are plain text, so name each focusable part after its parameter.
        [...binding.element.querySelectorAll('input, button, [tabindex]')]
          .filter((node) => !node.closest('.tp-popv'))
          .forEach((node, index) => {
            node.setAttribute('aria-label', index === 0 ? label : `${label} value`);
          });
        const track = binding.element.querySelector<HTMLElement>('.tp-sldv_t');
        if (def.type === 'number' && track) {
          track.setAttribute('role', 'slider');
          track.setAttribute('aria-valuemin', String(def.min));
          track.setAttribute('aria-valuemax', String(def.max));
          current.sliders.set(key, track);
        }
      }
    }

    buildPresets(current);
    syncAria();
    syncPresetUi();
    // Hot reloads may change the schema; bring the URL in line with what is now live.
    scheduleWrite();
  }

  resetButton.addEventListener('click', () => {
    if (!session) return;
    setValues(presetValues(session.schema, {}));
    writeNow();
    announce('Parameters reset to defaults');
  });

  copyButton.addEventListener('click', async () => {
    writeNow();
    const url = location.href;
    linkField.hidden = true;
    try {
      await navigator.clipboard.writeText(url);
    } catch {
      showLink(url);
      return;
    }
    copyButton.textContent = 'Copied';
    clearTimeout(copiedTimer);
    copiedTimer = window.setTimeout(() => (copyButton.textContent = 'Copy link'), 2000);
    announce('Link copied to the clipboard');
  });

  // Clipboard access can be missing (insecure origin) or refused; select the link for manual copy.
  function showLink(url: string) {
    linkField.value = url;
    linkField.hidden = false;
    linkField.focus();
    linkField.select();
    announce('Could not copy automatically. The link is selected: press Ctrl+C or Cmd+C.');
  }

  const stopLoad = host.onLoad(build);

  // Back and forward: apply the link's parameters, unless only some other key changed.
  const stopQuery = onQueryChange((query) => {
    const signature = paramSignature(query);
    if (signature === seen || !session) return;
    seen = signature;
    clearTimeout(writeTimer);
    writeTimer = undefined;
    setValues(valuesFromQuery(session.schema, session.presets, query));
  });

  const flush = () => {
    if (writeTimer !== undefined) writeNow();
  };
  addEventListener('pagehide', flush);

  return () => {
    // No flush here: unmounting usually means the URL already moved on to another page.
    clearTimeout(writeTimer);
    clearTimeout(statusTimer);
    clearTimeout(copiedTimer);
    removeEventListener('pagehide', flush);
    stopLoad();
    stopQuery();
    session?.pane?.dispose();
    root.remove();
  };
}
