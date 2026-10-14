import { describe, expect, it } from 'vitest';
import { layouts, scrim } from '../src/shell/contexts.ts';

describe('layout registry', () => {
  it('starts with the bare canvas, the default', () => {
    expect(layouts[0]?.id).toBe('bare');
  });

  it('has unique, URL-safe ids', () => {
    const ids = layouts.map((layout) => layout.id);
    expect(new Set(ids).size).toBe(ids.length);
    for (const id of ids) expect(id).toMatch(/^[a-z][a-z0-9-]*$/);
  });

  it('has short titles that fit a toolbar button', () => {
    for (const layout of layouts) {
      expect(layout.title.trim()).toBe(layout.title);
      expect(layout.title.split(/\s+/).length).toBeLessThanOrEqual(2);
      expect(layout.title.length).toBeLessThanOrEqual(12);
    }
    const titles = layouts.map((layout) => layout.title);
    expect(new Set(titles).size).toBe(titles.length);
  });

  it('stays a small set', () => {
    expect(layouts.length).toBeGreaterThanOrEqual(4);
    expect(layouts.length).toBeLessThanOrEqual(7);
  });
});

describe('scrim', () => {
  it('mixes a token with transparency', () => {
    expect(scrim('--wt-bg', 60)).toBe('color-mix(in srgb, var(--wt-bg) 60%, transparent)');
  });

  it('clamps and rounds the strength', () => {
    expect(scrim('--wt-bg', -5)).toContain(' 0%');
    expect(scrim('--wt-bg', 140)).toContain(' 100%');
    expect(scrim('--wt-bg', 61.6)).toContain(' 62%');
  });
});
