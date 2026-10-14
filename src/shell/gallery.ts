import { listEffects, onEffectsChanged } from '../effects/registry.ts';
import type { EffectMeta, Tier } from '../effects/types.ts';
import { mountEffect, type EffectHost } from './host.ts';
import { hrefFor, onQueryChange, readQuery, updateQuery, type QueryChanges } from './router.ts';
import type { ShellContext, Unmount } from './types.ts';
import './gallery.css';

// ---------------------------------------------------------------------------------------------
// Filters. Pure, so they are unit-tested without a DOM.
//
// URL state: `?tier=A` and `?tag=foo`. Both are single-select and combine with AND. Single
// select keeps the URL, the controls, and the empty state easy to reason about; tiers are
// mutually exclusive verdicts, and a multi-tag OR filter is better served by the tag count
// hints than by a second control. An unknown tier value is ignored; an unknown tag matches
// nothing (the page then offers to clear it).
// ---------------------------------------------------------------------------------------------

export interface Filters {
  tier: Tier | null;
  tag: string | null;
}

export const NO_FILTERS: Filters = { tier: null, tag: null };

/** Mirrors plan.md "Practicality tiers"; budgets are GPU time per frame at 1080p. */
export const TIERS: readonly {
  tier: Tier;
  name: string;
  blurb: string;
  budget: string;
}[] = [
  {
    tier: 'A',
    name: 'Drop-in',
    blurb: 'Ambient decoration, safe on mobile.',
    budget: '≤ 2 ms, integrated GPU',
  },
  {
    tier: 'B',
    name: 'Feature',
    blurb: 'Hero sections and interactive set pieces.',
    budget: '≤ 6 ms, integrated GPU',
  },
  {
    tier: 'C',
    name: 'Showpiece',
    blurb: 'Dedicated page, desktop-first.',
    budget: '≤ 12 ms, discrete or Apple Silicon',
  },
];

const isTier = (value: string | null): value is Tier =>
  value === 'A' || value === 'B' || value === 'C';

export function parseFilters(query: URLSearchParams): Filters {
  const tier = query.get('tier');
  return { tier: isTier(tier) ? tier : null, tag: query.get('tag') || null };
}

/** Query changes that set exactly these filters, for `updateQuery` and `hrefFor`. */
export function filterChanges(filters: Filters): QueryChanges {
  return { tier: filters.tier, tag: filters.tag };
}

export function isFiltered(filters: Filters): boolean {
  return filters.tier !== null || filters.tag !== null;
}

export function matchesFilters(meta: EffectMeta, filters: Filters): boolean {
  return (
    (filters.tier === null || meta.tier === filters.tier) &&
    (filters.tag === null || meta.tags.includes(filters.tag))
  );
}

export function filterEffects(metas: readonly EffectMeta[], filters: Filters): EffectMeta[] {
  return metas.filter((meta) => matchesFilters(meta, filters));
}

/**
 * Every tag in use, sorted, each with the number of effects it would show given the tier filter
 * alone. Tags with a zero count stay in the list so the control does not reshuffle.
 */
export function tagCounts(
  metas: readonly EffectMeta[],
  tier: Tier | null,
): { tag: string; count: number }[] {
  const counts = new Map<string, number>();
  for (const meta of metas) {
    for (const tag of meta.tags) {
      counts.set(tag, (counts.get(tag) ?? 0) + (tier === null || meta.tier === tier ? 1 : 0));
    }
  }
  return [...counts]
    .map(([tag, count]) => ({ tag, count }))
    .sort((a, b) => a.tag.localeCompare(b.tag));
}

/** Per tier, the number of effects it would show given the tag filter alone. */
export function tierCounts(metas: readonly EffectMeta[], tag: string | null): Record<Tier, number> {
  const counts: Record<Tier, number> = { A: 0, B: 0, C: 0 };
  for (const meta of metas) {
    if (tag === null || meta.tags.includes(tag)) counts[meta.tier] += 1;
  }
  return counts;
}

// ---------------------------------------------------------------------------------------------
// Live-thumbnail budget. Pure, so it is unit-tested without a DOM.
//
// Every live thumbnail holds an effect instance (pipelines, and for simulations their state
// textures), and every animating one costs GPU time each frame, so both are capped.
//   maxLive 12      About one desktop viewport of cards (4 columns x 2-3 rows) can all show a real
//                   frame, while 100 effects never mean 100 canvases. Live-but-idle canvases cost
//                   memory only, because a paused host draws one frame and stops.
//   maxAnimated 3   A handful of moving cards makes the page feel alive; more than that, at tier B
//                   cost each, risks blowing the frame budget on an integrated GPU.
// Priority: the hovered or focused card, then the cards nearest the viewport centre. A card that
// is already live gets `stickiness` pixels of credit so scrolling does not swap two cards of
// near-equal rank back and forth, which would rebuild an effect every few frames.
// ---------------------------------------------------------------------------------------------

export interface ThumbnailPolicy {
  maxLive: number;
  maxAnimated: number;
  stickiness: number;
}

export const THUMBNAIL_POLICY: ThumbnailPolicy = { maxLive: 12, maxAnimated: 3, stickiness: 120 };

/** A card that is on (or near) screen. `x` and `y` are its centre in viewport pixels. */
export interface CardBox {
  id: string;
  tier: Tier;
  x: number;
  y: number;
}

export interface ThumbnailInput {
  /** Cards currently on screen, in document order. */
  cards: readonly CardBox[];
  viewport: { width: number; height: number };
  /** Ids that have a canvas right now. */
  live: ReadonlySet<string>;
  /** The hovered or keyboard-focused card, if any. */
  focus: string | null;
  /** False when there is no working device: show placeholders only. */
  gpuReady: boolean;
  /** Tab hidden: keep what is mounted but animate nothing. */
  hidden: boolean;
  /** No continuous animation at all, not even on hover. */
  reducedMotion: boolean;
}

export interface ThumbnailPlan {
  /** Cards that should have a canvas. */
  live: ReadonlySet<string>;
  /** The subset of `live` that should animate; the rest hold a single frame. */
  animated: ReadonlySet<string>;
}

export function planThumbnails(
  input: ThumbnailInput,
  policy: ThumbnailPolicy = THUMBNAIL_POLICY,
): ThumbnailPlan {
  if (!input.gpuReady) return { live: new Set(), animated: new Set() };

  const cx = input.viewport.width / 2;
  const cy = input.viewport.height / 2;
  const ranked = input.cards
    .map((card, index) => ({
      card,
      index,
      focused: card.id === input.focus,
      score:
        Math.hypot(card.x - cx, card.y - cy) - (input.live.has(card.id) ? policy.stickiness : 0),
    }))
    .sort((a, b) => Number(b.focused) - Number(a.focused) || a.score - b.score || a.index - b.index)
    .slice(0, policy.maxLive);

  const animated = new Set<string>();
  if (!input.hidden && !input.reducedMotion) {
    for (const { card, focused } of ranked) {
      if (animated.size >= policy.maxAnimated) break;
      // Showpieces are heavy and desktop-first: they only move when the pointer asks.
      if (card.tier !== 'C' || focused) animated.add(card.id);
    }
  }
  return { live: new Set(ranked.map(({ card }) => card.id)), animated };
}

// ---------------------------------------------------------------------------------------------
// DOM
// ---------------------------------------------------------------------------------------------

// Per-effect posters (`src/effects/<id>/poster.webp`) are picked up here as soon as they exist.
// They show on cards that are not live, so a card is never an empty box.
const posterFiles = import.meta.glob<string>('../effects/*/poster.webp', {
  query: '?url',
  import: 'default',
  eager: true,
});
const posterUrls = new Map(
  Object.entries(posterFiles).map(([path, url]) => [path.split('/')[2]!, url]),
);

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

const plural = (count: number) => `${count} ${count === 1 ? 'effect' : 'effects'}`;

const MAX_CARD_TAGS = 3;

interface Card {
  meta: EffectMeta;
  item: HTMLLIElement;
  thumb: HTMLElement;
}

function buildCard(meta: EffectMeta): Card {
  const item = el('li', 'wt-gallery__item');
  const link = el('a', 'wt-gallery__card wt-panel');
  link.href = hrefFor({ effect: meta.id }, '');
  link.dataset.link = '';

  const thumb = el('div', 'wt-gallery__thumb');
  thumb.dataset.tier = meta.tier;
  thumb.setAttribute('aria-hidden', 'true');
  const poster = posterUrls.get(meta.id);
  if (poster) {
    const image = el('img', 'wt-gallery__poster');
    image.src = poster;
    image.alt = '';
    image.loading = 'lazy';
    image.decoding = 'async';
    thumb.append(image);
  }

  const name = el('h2', 'wt-gallery__name', meta.title);
  name.id = `wt-gallery-name-${meta.id}`;
  const tier = el('span', 'wt-chip', `Tier ${meta.tier}`);
  tier.dataset.tier = meta.tier;
  tier.id = `wt-gallery-tier-${meta.id}`;
  const head = el('div', 'wt-gallery__head');
  head.append(name, tier);
  link.setAttribute('aria-labelledby', `${name.id} ${tier.id}`);

  const body = el('div', 'wt-gallery__body');
  body.append(head);
  if (meta.summary) {
    const summary = el('p', 'wt-gallery__summary wt-muted', meta.summary);
    summary.id = `wt-gallery-summary-${meta.id}`;
    link.setAttribute('aria-describedby', summary.id);
    body.append(summary);
  }
  if (meta.tags.length) {
    const tags = el('div', 'wt-gallery__tags');
    tags.append(...meta.tags.slice(0, MAX_CARD_TAGS).map((tag) => el('span', 'wt-chip', tag)));
    const more = meta.tags.length - MAX_CARD_TAGS;
    if (more > 0) tags.append(el('span', 'wt-chip', `+${more}`));
    body.append(tags);
  }

  link.append(thumb, body);
  item.append(link);
  return { meta, item, thumb };
}

export function mountGallery(container: HTMLElement, shell: ShellContext): Unmount {
  document.title = 'Water Temple';
  const root = el('div', 'wt-gallery');
  container.append(root);

  const reducedMotion = matchMedia('(prefers-reduced-motion: reduce)');
  // Torn down and rebuilt together when the effect list changes (dev hot reload).
  let teardown: (() => void) | undefined;

  function render() {
    teardown?.();
    root.replaceChildren();
    const metas = listEffects();
    const cards = new Map(metas.map((meta) => [meta.id, buildCard(meta)]));

    // --- Header -------------------------------------------------------------------------
    const intro = el('header', 'wt-gallery__intro');
    intro.append(
      el('h1', 'wt-gallery__heading', 'Liquid effects'),
      el(
        'p',
        'wt-muted',
        'A research gallery of WebGPU water and liquid effects, each scored for real-world ' +
          'practicality: GPU cost, behaviour on phones, and what to show where WebGPU is missing.',
      ),
    );

    // --- Filters ------------------------------------------------------------------------
    const filterBar = el('section', 'wt-gallery__filters');
    filterBar.setAttribute('aria-label', 'Filter effects');

    const tierGroup = el('div', 'wt-gallery__tiers');
    tierGroup.setAttribute('role', 'group');
    tierGroup.setAttribute('aria-label', 'Practicality tier');
    const tierButtons = new Map<Tier, { button: HTMLButtonElement; count: HTMLElement }>();
    for (const { tier, name, blurb, budget } of TIERS) {
      const button = el('button', 'wt-gallery__tier');
      button.type = 'button';
      const chip = el('span', 'wt-chip', `Tier ${tier}`);
      chip.dataset.tier = tier;
      const count = el('span', 'wt-gallery__tier-count wt-data wt-muted');
      const head = el('span', 'wt-gallery__tier-head');
      head.append(chip, el('span', 'wt-gallery__tier-name', name), count);
      button.append(
        head,
        el('span', 'wt-gallery__tier-blurb wt-muted', blurb),
        el('span', 'wt-gallery__tier-budget wt-data wt-muted', `${budget} @1080p`),
      );
      button.addEventListener('click', () => {
        const pressed = button.getAttribute('aria-pressed') === 'true';
        updateQuery({ tier: pressed ? null : tier }, { replace: true });
      });
      tierButtons.set(tier, { button, count });
      tierGroup.append(button);
    }

    const tagField = el('div', 'wt-gallery__field');
    const tagLabel = el('label', 'wt-label', 'Tag');
    const tagSelect = el('select', 'wt-gallery__select');
    tagSelect.id = 'wt-gallery-tag';
    tagLabel.htmlFor = tagSelect.id;
    tagSelect.addEventListener('change', () => {
      updateQuery({ tag: tagSelect.value || null }, { replace: true });
    });
    tagField.append(tagLabel, tagSelect);

    const count = el('p', 'wt-gallery__count wt-data wt-muted');
    count.setAttribute('role', 'status');
    const clear = el('button', 'wt-button wt-button--quiet', 'Clear filters');
    clear.type = 'button';
    const clearFilters = () => {
      updateQuery(filterChanges(NO_FILTERS), { replace: true });
      // The button disappears once the filters are gone; keep the keyboard in the filters.
      tierButtons.get('A')?.button.focus();
    };
    clear.addEventListener('click', clearFilters);

    const toolbar = el('div', 'wt-gallery__toolbar');
    toolbar.append(tagField, count, clear);
    filterBar.append(tierGroup, toolbar);

    // --- Results ------------------------------------------------------------------------
    const grid = el('ul', 'wt-gallery__grid');
    grid.setAttribute('role', 'list');
    grid.append(...[...cards.values()].map((card) => card.item));

    const empty = el('div', 'wt-gallery__empty wt-panel');
    const emptyText = el('p', '');
    const emptyClear = el('button', 'wt-button', 'Clear filters');
    emptyClear.type = 'button';
    emptyClear.addEventListener('click', clearFilters);
    empty.append(emptyText, emptyClear);

    root.append(intro, filterBar, grid, empty);

    function applyFilters() {
      const filters = parseFilters(readQuery());
      const shown = filterEffects(metas, filters);
      const shownIds = new Set(shown.map((meta) => meta.id));
      for (const [id, card] of cards) card.item.hidden = !shownIds.has(id);

      const perTier = tierCounts(metas, filters.tag);
      for (const [tier, { button, count: badge }] of tierButtons) {
        button.setAttribute('aria-pressed', String(filters.tier === tier));
        badge.textContent = plural(perTier[tier]);
      }

      const tags = tagCounts(metas, filters.tier);
      // A tag from a stale link stays selectable so the select never lies about the URL.
      if (filters.tag !== null && !tags.some(({ tag }) => tag === filters.tag)) {
        tags.push({ tag: filters.tag, count: 0 });
      }
      tagSelect.replaceChildren(
        new Option('All tags', ''),
        ...tags.map(({ tag, count: n }) => {
          const option = new Option(`${tag} (${n})`, tag);
          option.disabled = n === 0 && tag !== filters.tag;
          return option;
        }),
      );
      tagSelect.value = filters.tag ?? '';

      count.textContent = isFiltered(filters)
        ? `${shown.length} of ${plural(metas.length)}`
        : plural(metas.length);
      clear.hidden = !isFiltered(filters);

      grid.hidden = shown.length === 0;
      empty.hidden = shown.length > 0;
      if (shown.length === 0) {
        const parts = [
          filters.tier && `tier ${filters.tier}`,
          filters.tag && `the tag “${filters.tag}”`,
        ].filter(Boolean);
        emptyText.textContent = parts.length
          ? `No effects match ${parts.join(' and ')}.`
          : 'No effects are registered yet.';
        emptyClear.hidden = parts.length === 0;
      }
      schedule();
    }

    // --- Live thumbnails ----------------------------------------------------------------
    const hosts = new Map<string, EffectHost>();
    const visible = new Set<Card>();
    let hovered: string | null = null;
    let focused: string | null = null;
    let frame = 0;

    function reschedule() {
      frame = 0;
      const state = shell.session.state;
      const boxes: CardBox[] = [];
      // Document order, so rank ties resolve top-left first.
      for (const card of cards.values()) {
        if (!visible.has(card)) continue;
        const rect = card.thumb.getBoundingClientRect();
        boxes.push({
          id: card.meta.id,
          tier: card.meta.tier,
          x: rect.left + rect.width / 2,
          y: rect.top + rect.height / 2,
        });
      }

      const plan = planThumbnails({
        cards: boxes,
        viewport: { width: innerWidth, height: innerHeight },
        live: new Set(hosts.keys()),
        focus: hovered ?? focused,
        gpuReady: state.ok,
        hidden: document.hidden,
        reducedMotion: reducedMotion.matches,
      });

      // Free canvases before making new ones so the cap holds at every instant.
      for (const [id, host] of hosts) {
        if (plan.live.has(id)) continue;
        host.destroy();
        hosts.delete(id);
        delete cards.get(id)!.thumb.dataset.live;
      }
      for (const id of plan.live) {
        const card = cards.get(id)!;
        let host = hosts.get(id);
        if (!host) {
          // Small and low-DPR on purpose: a thumbnail is a few tens of thousands of pixels.
          host = mountEffect(shell.session, card.thumb, id, {
            paused: true,
            maxDpr: 1,
          });
          hosts.set(id, host);
        }
        const moving = plan.animated.has(id);
        if (moving && host.paused) host.resume();
        else if (!moving && !host.paused) host.pause();
        card.thumb.dataset.live = moving ? 'animated' : 'still';
      }
    }

    // Coalesces scroll, resize, hover, and observer bursts into one pass per frame. A hidden tab
    // gets no animation frames, so it reschedules synchronously instead.
    function schedule() {
      if (document.hidden) reschedule();
      else frame ||= requestAnimationFrame(reschedule);
    }

    const observer = new IntersectionObserver(
      (entries) => {
        for (const entry of entries) {
          const card = cards.get((entry.target as HTMLElement).dataset.id ?? '');
          if (!card) continue;
          if (entry.isIntersecting) visible.add(card);
          else visible.delete(card);
        }
        schedule();
      },
      // Start building a little before a card scrolls in, so it rarely shows the placeholder.
      { rootMargin: '160px 0px' },
    );

    const pointerOwner = (event: Event) =>
      (event.target as Element | null)?.closest<HTMLElement>('.wt-gallery__item')?.dataset.id ??
      null;
    const onPointerOver = (event: PointerEvent) => {
      // Touch has no hover; a tap would leave the card "hovered" until the next tap.
      if (event.pointerType === 'touch') return;
      hovered = pointerOwner(event);
      schedule();
    };
    const onPointerLeave = () => {
      hovered = null;
      schedule();
    };
    const onFocusIn = (event: FocusEvent) => {
      focused = pointerOwner(event);
      schedule();
    };
    const onFocusOut = () => {
      focused = null;
      schedule();
    };
    grid.addEventListener('pointerover', onPointerOver);
    grid.addEventListener('pointerleave', onPointerLeave);
    grid.addEventListener('focusin', onFocusIn);
    grid.addEventListener('focusout', onFocusOut);
    addEventListener('scroll', schedule, { passive: true });
    addEventListener('resize', schedule);
    document.addEventListener('visibilitychange', schedule);
    reducedMotion.addEventListener('change', schedule);
    const stopSession = shell.session.onChange(schedule);
    const stopQuery = onQueryChange(applyFilters);

    for (const [id, card] of cards) {
      card.item.dataset.id = id;
      observer.observe(card.item);
    }
    applyFilters();

    teardown = () => {
      if (frame) cancelAnimationFrame(frame);
      observer.disconnect();
      visible.clear();
      for (const host of hosts.values()) host.destroy();
      hosts.clear();
      removeEventListener('scroll', schedule);
      removeEventListener('resize', schedule);
      document.removeEventListener('visibilitychange', schedule);
      reducedMotion.removeEventListener('change', schedule);
      stopSession();
      stopQuery();
      teardown = undefined;
    };
  }

  render();
  const stopEffects = onEffectsChanged(render);

  return () => {
    stopEffects();
    teardown?.();
    root.remove();
  };
}
