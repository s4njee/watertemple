import { describe, expect, it } from 'vitest';
import type { EffectMeta, Tier } from '../src/effects/types.ts';
import {
  NO_FILTERS,
  THUMBNAIL_POLICY,
  filterChanges,
  filterEffects,
  isFiltered,
  parseFilters,
  planThumbnails,
  tagCounts,
  tierCounts,
  type CardBox,
  type ThumbnailInput,
} from '../src/shell/gallery.ts';

function meta(id: string, tier: Tier, tags: string[]): EffectMeta {
  return { id, title: id, epic: '4', tier, tags };
}

const METAS = [
  meta('caustics', 'A', ['caustics', 'shader-only']),
  meta('divider', 'A', ['shader-only', 'ui']),
  meta('ripples', 'B', ['heightfield', 'interactive']),
  meta('fluid', 'B', ['compute', 'interactive']),
  meta('ocean', 'C', ['compute', 'ocean']),
];

describe('parseFilters', () => {
  it('reads tier and tag', () => {
    expect(parseFilters(new URLSearchParams('?tier=B&tag=compute'))).toEqual({
      tier: 'B',
      tag: 'compute',
    });
  });

  it('defaults to no filters', () => {
    expect(parseFilters(new URLSearchParams(''))).toEqual(NO_FILTERS);
  });

  it('ignores an invalid tier and an empty tag', () => {
    expect(parseFilters(new URLSearchParams('?tier=Z&tag='))).toEqual(NO_FILTERS);
    expect(parseFilters(new URLSearchParams('?tier=a'))).toEqual(NO_FILTERS);
  });

  it('ignores unrelated keys', () => {
    expect(parseFilters(new URLSearchParams('?effect=hello&hud=1'))).toEqual(NO_FILTERS);
  });
});

describe('filterChanges', () => {
  it('sets present filters and removes absent ones', () => {
    expect(filterChanges({ tier: 'A', tag: null })).toEqual({ tier: 'A', tag: null });
    expect(filterChanges(NO_FILTERS)).toEqual({ tier: null, tag: null });
  });
});

describe('filterEffects', () => {
  const ids = (filters: Parameters<typeof filterEffects>[1]) =>
    filterEffects(METAS, filters).map((m) => m.id);

  it('returns everything without filters', () => {
    expect(ids(NO_FILTERS)).toEqual(['caustics', 'divider', 'ripples', 'fluid', 'ocean']);
  });

  it('filters by tier', () => {
    expect(ids({ tier: 'A', tag: null })).toEqual(['caustics', 'divider']);
  });

  it('filters by tag', () => {
    expect(ids({ tier: null, tag: 'compute' })).toEqual(['fluid', 'ocean']);
  });

  it('combines tier and tag with AND', () => {
    expect(ids({ tier: 'B', tag: 'compute' })).toEqual(['fluid']);
    expect(ids({ tier: 'A', tag: 'compute' })).toEqual([]);
  });

  it('matches nothing for an unknown tag', () => {
    expect(ids({ tier: null, tag: 'nope' })).toEqual([]);
  });

  it('does not match a tag by substring', () => {
    expect(ids({ tier: null, tag: 'comp' })).toEqual([]);
  });
});

describe('isFiltered', () => {
  it('is true when either filter is set', () => {
    expect(isFiltered(NO_FILTERS)).toBe(false);
    expect(isFiltered({ tier: 'C', tag: null })).toBe(true);
    expect(isFiltered({ tier: null, tag: 'ui' })).toBe(true);
  });
});

describe('tagCounts', () => {
  it('lists every tag sorted, counting all effects', () => {
    expect(tagCounts(METAS, null)).toEqual([
      { tag: 'caustics', count: 1 },
      { tag: 'compute', count: 2 },
      { tag: 'heightfield', count: 1 },
      { tag: 'interactive', count: 2 },
      { tag: 'ocean', count: 1 },
      { tag: 'shader-only', count: 2 },
      { tag: 'ui', count: 1 },
    ]);
  });

  it('counts within the chosen tier but keeps every tag listed', () => {
    const counts = new Map(tagCounts(METAS, 'B').map(({ tag, count }) => [tag, count]));
    expect(counts.get('interactive')).toBe(2);
    expect(counts.get('compute')).toBe(1);
    expect(counts.get('ui')).toBe(0);
    expect(counts.size).toBe(7);
  });
});

describe('tierCounts', () => {
  it('counts every tier without a tag', () => {
    expect(tierCounts(METAS, null)).toEqual({ A: 2, B: 2, C: 1 });
  });

  it('counts within the chosen tag', () => {
    expect(tierCounts(METAS, 'compute')).toEqual({ A: 0, B: 1, C: 1 });
    expect(tierCounts(METAS, 'nope')).toEqual({ A: 0, B: 0, C: 0 });
  });
});

describe('planThumbnails', () => {
  const viewport = { width: 1000, height: 800 };

  /** One card per entry, centred on the viewport horizontally, `y` px from the top. */
  const column = (...rows: [id: string, y: number, tier?: Tier][]): CardBox[] =>
    rows.map(([id, y, tier = 'A']) => ({ id, tier, x: 500, y }));

  const input = (overrides: Partial<ThumbnailInput>): ThumbnailInput => ({
    cards: [],
    viewport,
    live: new Set(),
    focus: null,
    gpuReady: true,
    hidden: false,
    reducedMotion: false,
    ...overrides,
  });

  const policy = { maxLive: 2, maxAnimated: 1, stickiness: 0 };

  it('plans nothing for an empty page', () => {
    const plan = planThumbnails(input({}));
    expect(plan.live.size).toBe(0);
    expect(plan.animated.size).toBe(0);
  });

  it('never exceeds the live budget', () => {
    const cards = Array.from({ length: 40 }, (_, i): CardBox => ({
      id: `c${i}`,
      tier: 'A',
      x: 500,
      y: i * 50,
    }));
    const plan = planThumbnails(input({ cards }));
    expect(plan.live.size).toBe(THUMBNAIL_POLICY.maxLive);
    expect(plan.animated.size).toBeLessThanOrEqual(THUMBNAIL_POLICY.maxAnimated);
  });

  it('keeps every visible card live when under budget', () => {
    const plan = planThumbnails(input({ cards: column(['a', 100], ['b', 400]) }));
    expect([...plan.live].sort()).toEqual(['a', 'b']);
  });

  it('prefers cards nearest the viewport centre', () => {
    // Distances from the centre (y = 400): top 350, middle 20, bottom 390.
    const cards = column(['top', 50], ['middle', 380], ['bottom', 790]);
    const plan = planThumbnails(input({ cards }), policy);
    expect([...plan.live].sort()).toEqual(['middle', 'top']);
  });

  it('treats horizontal distance too, so a row is not cut at its right edge', () => {
    const cards: CardBox[] = [
      { id: 'left', tier: 'A', x: 100, y: 400 },
      { id: 'centre', tier: 'A', x: 500, y: 400 },
      { id: 'right', tier: 'A', x: 900, y: 400 },
    ];
    const plan = planThumbnails(input({ cards }), policy);
    expect(plan.live.has('centre')).toBe(true);
    expect(plan.live.size).toBe(2);
  });

  it('breaks exact ties in document order', () => {
    const cards: CardBox[] = [
      { id: 'first', tier: 'A', x: 100, y: 400 },
      { id: 'second', tier: 'A', x: 900, y: 400 },
      { id: 'third', tier: 'A', x: 100, y: 400 },
    ];
    const plan = planThumbnails(input({ cards }), { ...policy, maxLive: 1 });
    expect([...plan.live]).toEqual(['first']);
  });

  it('gives the hovered or focused card a slot first, even far from the centre', () => {
    const cards = column(['far', 0], ['near1', 390], ['near2', 410]);
    const plan = planThumbnails(input({ cards, focus: 'far' }), policy);
    expect(plan.live.has('far')).toBe(true);
    expect(plan.live.size).toBe(2);
    expect(plan.animated.has('far')).toBe(true);
  });

  it('ignores a focus that is not among the visible cards', () => {
    const plan = planThumbnails(input({ cards: column(['a', 400]), focus: 'gone' }), policy);
    expect([...plan.live]).toEqual(['a']);
  });

  it('keeps an already-live card against a marginally closer newcomer (stickiness)', () => {
    const cards = column(['old', 520], ['new', 480]);
    const sticky = { maxLive: 1, maxAnimated: 1, stickiness: 100 };
    const without = planThumbnails(input({ cards }), { ...sticky, stickiness: 0 });
    expect([...without.live]).toEqual(['new']);
    const kept = planThumbnails(input({ cards, live: new Set(['old']) }), sticky);
    expect([...kept.live]).toEqual(['old']);
  });

  it('lets a clearly closer card displace a live one despite stickiness', () => {
    const cards = column(['old', 790], ['new', 400]);
    const plan = planThumbnails(input({ cards, live: new Set(['old']) }), {
      maxLive: 1,
      maxAnimated: 1,
      stickiness: 100,
    });
    expect([...plan.live]).toEqual(['new']);
  });

  it('animates only the best-ranked cards, as many as the animation budget', () => {
    const cards = column(['a', 400], ['b', 300], ['c', 600], ['d', 100]);
    const plan = planThumbnails(input({ cards }), { maxLive: 3, maxAnimated: 2, stickiness: 0 });
    expect([...plan.live].sort()).toEqual(['a', 'b', 'c']);
    expect([...plan.animated].sort()).toEqual(['a', 'b']);
  });

  it('animates tier C only when it is hovered or focused', () => {
    const cards = column(['showpiece', 400, 'C'], ['dropin', 450, 'A']);
    const idle = planThumbnails(input({ cards }), { maxLive: 2, maxAnimated: 1, stickiness: 0 });
    expect(idle.live.has('showpiece')).toBe(true);
    expect([...idle.animated]).toEqual(['dropin']);
    const hovered = planThumbnails(input({ cards, focus: 'showpiece' }), {
      maxLive: 2,
      maxAnimated: 1,
      stickiness: 0,
    });
    expect([...hovered.animated]).toEqual(['showpiece']);
  });

  it('animates nothing under reduced motion, even on hover, but still shows a frame', () => {
    const cards = column(['a', 400], ['b', 450]);
    const plan = planThumbnails(input({ cards, focus: 'a', reducedMotion: true }));
    expect(plan.live.size).toBe(2);
    expect(plan.animated.size).toBe(0);
  });

  it('animates nothing while the tab is hidden, but keeps the canvases', () => {
    const plan = planThumbnails(input({ cards: column(['a', 400]), hidden: true }));
    expect(plan.live.size).toBe(1);
    expect(plan.animated.size).toBe(0);
  });

  it('plans no canvases without a working device', () => {
    const plan = planThumbnails(input({ cards: column(['a', 400]), gpuReady: false }));
    expect(plan.live.size).toBe(0);
    expect(plan.animated.size).toBe(0);
  });

  it('never animates a card that is not live', () => {
    const cards = Array.from({ length: 30 }, (_, i): CardBox => ({
      id: `c${i}`,
      tier: i % 3 === 2 ? 'C' : 'A',
      x: 500,
      y: i * 40,
    }));
    const plan = planThumbnails(input({ cards, focus: 'c29' }));
    for (const id of plan.animated) expect(plan.live.has(id)).toBe(true);
  });
});
