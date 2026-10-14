import { listEffects, loadEffect } from '../effects/registry.ts';
import type { EffectMeta } from '../effects/types.ts';
import { layouts, type MountedLayout } from './contexts.ts';
import { mountEffect, type EffectHost } from './host.ts';
import { lazyMount } from './lazy.ts';
import { decodeParams } from './params-url.ts';
import { displaySupportsHdr } from '../gpu/canvas.ts';
import { hrefFor, onQueryChange, readQuery, updateQuery } from './router.ts';
import type { ShellContext, Unmount } from './types.ts';
import './effect-page.css';

const TABS = [
  { id: 'notes', title: 'Notes' },
  { id: 'source', title: 'Source' },
  { id: 'scorecard', title: 'Scorecard' },
] as const;
type TabId = (typeof TABS)[number]['id'];

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

function intro(meta: EffectMeta): HTMLElement {
  const header = el('header', 'wt-effect__intro');
  const chips = el('div', 'wt-effect__chips');
  const tier = el('span', 'wt-chip', `Tier ${meta.tier}`);
  tier.dataset.tier = meta.tier;
  chips.append(tier, ...meta.tags.map((tag) => el('span', 'wt-chip', tag)));
  header.append(el('h1', 'wt-effect__title', meta.title), chips);
  if (meta.summary) header.append(el('p', 'wt-muted', meta.summary));
  return header;
}

export function mountEffectPage(container: HTMLElement, shell: ShellContext, id: string): Unmount {
  const meta = listEffects().find((candidate) => candidate.id === id)!;
  document.title = `${meta.title} · Water Temple`;

  const root = el('div', 'wt-effect');
  const main = el('div', 'wt-effect__main');
  const stage = el('div', 'wt-effect__stage');
  const hudSlot = el('div', 'wt-effect__hud');
  const toolbar = el('div', 'wt-effect__toolbar');
  const tabs = el('div', 'wt-tabs');
  tabs.setAttribute('role', 'tablist');
  const tabPanel = el('div', 'wt-effect__tabpanel');
  tabPanel.setAttribute('role', 'tabpanel');
  const side = el('aside', 'wt-effect__side');
  const controlsSlot = el('div', 'wt-effect__controls');
  side.append(intro(meta), controlsSlot);
  main.append(stage, toolbar, tabs, tabPanel);
  root.append(main, side);
  container.append(root);

  // Toolbar: where the effect is shown, then how it is inspected.
  const layoutGroup = el('div', 'wt-effect__group');
  layoutGroup.setAttribute('role', 'group');
  layoutGroup.setAttribute('aria-label', 'Page context');
  const layoutButtons = layouts.map((layout) => {
    const node = button(layout.title);
    node.addEventListener('click', () =>
      updateQuery({ layout: layout === layouts[0] ? null : layout.id }, { replace: true }),
    );
    layoutGroup.append(node);
    return node;
  });
  const hudButton = button('Performance');
  hudButton.addEventListener('click', () =>
    updateQuery({ hud: readQuery().has('hud') ? null : '1' }, { replace: true }),
  );
  const fullscreenButton = button('Fullscreen');
  fullscreenButton.addEventListener('click', () => {
    if (document.fullscreenElement) void document.exitFullscreen();
    else void stage.requestFullscreen();
  });
  toolbar.append(layoutGroup, el('span', 'wt-effect__spacer'), hudButton, fullscreenButton);

  const tabButtons = TABS.map((tab) => {
    const node = el('button', 'wt-tab', tab.title);
    node.type = 'button';
    node.setAttribute('role', 'tab');
    node.addEventListener('click', () =>
      updateQuery({ tab: tab === TABS[0] ? null : tab.id }, { replace: true }),
    );
    tabs.append(node);
    return node;
  });

  let host: EffectHost | undefined;
  let layout: (MountedLayout & { id: string }) | undefined;
  let hud: Unmount | undefined;
  let tab: { id: TabId; unmount: Unmount } | undefined;
  let controls: Unmount | undefined;
  let disposed = false;

  function showLayout(query: URLSearchParams) {
    const wanted = layouts.find((candidate) => candidate.id === query.get('layout')) ?? layouts[0]!;
    layoutButtons.forEach((node, index) =>
      node.setAttribute('aria-pressed', String(layouts[index] === wanted)),
    );
    if (!host || layout?.id === wanted.id) return;
    // The canvas keeps its GPU context when its element moves, so switching context is free.
    host.element.remove();
    layout?.destroy();
    layout = { id: wanted.id, ...wanted.mount(stage) };
    layout.stage.append(host.element);
    stage.append(hudSlot);
  }

  function showHud(query: URLSearchParams) {
    const on = query.has('hud');
    hudButton.setAttribute('aria-pressed', String(on));
    if (!host || on === (hud !== undefined)) return;
    if (on) {
      const mounted = host;
      hud = lazyMount(hudSlot, 'performance', async () =>
        (await import('./hud.ts')).mountHud(hudSlot, mounted, shell),
      );
    } else {
      hud?.();
      hud = undefined;
    }
  }

  function showTab(query: URLSearchParams) {
    const wanted = TABS.find((candidate) => candidate.id === query.get('tab')) ?? TABS[0];
    tabButtons.forEach((node, index) =>
      node.setAttribute('aria-selected', String(TABS[index] === wanted)),
    );
    if (tab?.id === wanted.id) return;
    tab?.unmount();
    tabPanel.replaceChildren();
    const load: Record<TabId, () => Promise<Unmount>> = {
      notes: async () => (await import('./notes.ts')).mountNotes(tabPanel, id),
      source: async () => (await import('./source-viewer.ts')).mountSourceViewer(tabPanel, id),
      scorecard: async () => (await import('./scorecard.ts')).mountScorecard(tabPanel, meta, host),
    };
    tab = { id: wanted.id, unmount: lazyMount(tabPanel, wanted.title, load[wanted.id]) };
  }

  function show(query: URLSearchParams) {
    showLayout(query);
    showHud(query);
    showTab(query);
  }

  const stopListening = onQueryChange(show);
  show(readQuery());

  // The schema is needed to decode parameter values from the URL before the first frame.
  void loadEffect(id).then((effect) => {
    if (disposed) return;
    const wantsHdr = readQuery().has('hdr');
    const mounted = mountEffect(shell.session, stage, id, {
      initialParams: decodeParams(effect.params, readQuery()),
      allowHdr: wantsHdr,
    });
    if (effect.hdr) {
      // The canvas format is fixed when the host is created, so the toggle reloads the page.
      const hdrButton = button('HDR canvas');
      hdrButton.setAttribute('aria-pressed', String(wantsHdr));
      if (displaySupportsHdr()) {
        hdrButton.title = 'Extended-range output. Expect a lower frame rate while it is on.';
        hdrButton.addEventListener('click', () =>
          location.assign(hrefFor({ hdr: wantsHdr ? null : '1' })),
        );
      } else {
        hdrButton.disabled = true;
        hdrButton.title = 'This display does not report high dynamic range.';
      }
      fullscreenButton.before(hdrButton);
    }
    host = mounted;
    controls = lazyMount(controlsSlot, 'controls', async () =>
      (await import('./controls.ts')).mountControls(controlsSlot, mounted),
    );
    // A scorecard opened straight from a link mounted before the host existed; give it the host
    // so it can show this device's live reading.
    if (tab?.id === 'scorecard') {
      tab.unmount();
      tab = undefined;
    }
    show(readQuery());
  });

  return () => {
    disposed = true;
    stopListening();
    controls?.();
    hud?.();
    tab?.unmount();
    layout?.destroy();
    host?.destroy();
    root.remove();
    document.title = 'Water Temple';
  };
}
