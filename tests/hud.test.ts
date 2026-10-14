import { describe, expect, it } from 'vitest';
import {
  budgetStatus,
  createPassWindow,
  createRollingWindow,
  formatBytes,
  formatMs,
  looksQuantised,
  sparklinePoints,
  summarize,
  TIER_BUDGET_MS,
} from '../src/shell/hud.ts';

describe('createRollingWindow', () => {
  it('keeps only samples inside the span', () => {
    const window = createRollingWindow(1000);
    window.push(0, 1);
    window.push(400, 2);
    window.push(900, 3);
    expect(window.values(900)).toEqual([1, 2, 3]);
    expect(window.values(1200)).toEqual([2, 3]);
    expect(window.values(5000)).toEqual([]);
  });

  it('caps the sample count, dropping the oldest', () => {
    const window = createRollingWindow(1e9, 3);
    for (let i = 1; i <= 10; i++) window.push(i, i);
    expect(window.values(10)).toEqual([8, 9, 10]);
  });

  it('stays correct across many pushes and prunes', () => {
    const window = createRollingWindow(100);
    for (let t = 0; t < 5000; t++) window.push(t, t);
    const values = window.values(4999);
    expect(values[0]).toBe(4899);
    expect(values.at(-1)).toBe(4999);
    expect(values).toHaveLength(101);
  });

  it('clears', () => {
    const window = createRollingWindow(100);
    window.push(0, 1);
    window.clear();
    expect(window.values(0)).toEqual([]);
  });
});

describe('summarize', () => {
  it('is undefined for no samples', () => {
    expect(summarize([])).toBeUndefined();
  });

  it('reports mean, extremes and a nearest-rank p95', () => {
    const values = Array.from({ length: 100 }, (_, i) => i + 1);
    expect(summarize(values)).toEqual({ count: 100, mean: 50.5, min: 1, max: 100, p95: 95 });
  });

  it('does not depend on input order and does not mutate it', () => {
    const values = [5, 1, 3];
    expect(summarize(values)).toMatchObject({ min: 1, max: 5, mean: 3, p95: 5 });
    expect(values).toEqual([5, 1, 3]);
  });

  it('lets one slow frame show up in p95 only when it is common enough', () => {
    const smooth = Array.from({ length: 99 }, () => 16);
    expect(summarize([...smooth, 80])?.p95).toBe(16);
    expect(summarize([...smooth.slice(0, 90), ...Array.from({ length: 10 }, () => 80)])?.p95).toBe(
      80,
    );
  });
});

describe('createPassWindow', () => {
  const frame = (noise: number, blur?: number) => [
    { label: 'noise', ms: noise },
    ...(blur === undefined ? [] : [{ label: 'blur', ms: blur }]),
  ];

  it('averages per pass and in total', () => {
    const window = createPassWindow(1000);
    window.push(0, frame(1, 2));
    window.push(100, frame(3, 4));
    const { total, passes } = window.summary(100);
    expect(total?.mean).toBe(5);
    expect(passes).toEqual([
      { label: 'noise', mean: 2, max: 3 },
      { label: 'blur', mean: 3, max: 4 },
    ]);
  });

  it('takes the frame time it is given as the total instead of summing overlapping passes', () => {
    const window = createPassWindow(1000);
    window.push(0, frame(1, 2), 2.2);
    window.push(100, frame(3, 4), 4.4);
    const { total, passes } = window.summary(100);
    expect(total?.mean).toBeCloseTo(3.3, 6);
    expect(passes.map((pass) => pass.mean)).toEqual([2, 3]);
  });

  it('counts a repeated array once: the profiler republishes the last frame until a new one lands', () => {
    const window = createPassWindow(1000);
    const sample = frame(2);
    for (let t = 0; t < 10; t++) window.push(t, sample);
    expect(window.summary(10).total?.count).toBe(1);
    window.push(11, frame(2));
    expect(window.summary(11).total?.count).toBe(2);
  });

  it('ignores empty results and forgets passes that stop reporting', () => {
    const window = createPassWindow(1000);
    window.push(0, []);
    expect(window.summary(0).total).toBeUndefined();
    window.push(0, frame(1, 1));
    window.push(900, frame(1));
    const later = window.summary(1500);
    expect(later.passes.map((pass) => pass.label)).toEqual(['noise']);
    expect(window.summary(5000).passes).toEqual([]);
  });

  it('clears', () => {
    const window = createPassWindow(1000);
    const sample = frame(1);
    window.push(0, sample);
    window.clear();
    expect(window.summary(0).total).toBeUndefined();
    window.push(1, sample); // the same array counts again after a clear
    expect(window.summary(1).total?.count).toBe(1);
  });
});

describe('budgetStatus', () => {
  it('uses the plan.md tier budgets', () => {
    expect(TIER_BUDGET_MS).toEqual({ A: 2, B: 6, C: 12 });
  });

  it('is within, near or over the budget', () => {
    expect(budgetStatus('A', 1).level).toBe('within');
    expect(budgetStatus('A', 1.6).level).toBe('near');
    expect(budgetStatus('A', 2).level).toBe('near');
    expect(budgetStatus('A', 2.01).level).toBe('over');
    expect(budgetStatus('C', 9)).toMatchObject({ level: 'within', budgetMs: 12, ratio: 0.75 });
  });
});

describe('looksQuantised', () => {
  it('recognises durations that all sit on the 0.1 ms grid', () => {
    expect(looksQuantised([0.1, 0.2, 0.3, 0.4, 0.5, 0.1, 0.2, 1.3])).toBe(true);
    expect(looksQuantised([0.1, 0.2, 0.3, 0.4, 0.5, 0.1, 0.2, 1.3000000001])).toBe(true);
  });

  it('rejects real, unquantised timings', () => {
    expect(looksQuantised([0.412, 0.398, 0.405, 0.421, 0.4, 0.39, 0.41, 0.402])).toBe(false);
  });

  it('wants enough evidence and ignores zeros', () => {
    expect(looksQuantised([0.1, 0.2])).toBe(false);
    expect(looksQuantised([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])).toBe(false);
    expect(looksQuantised([0, 0.1, 0.1, 0.2, 0.1, 0.3, 0.1, 0.2, 0.1])).toBe(true);
  });
});

describe('formatting', () => {
  it('keeps a stable number of significant digits', () => {
    expect(formatMs(0.4234)).toBe('0.42');
    expect(formatMs(16.666)).toBe('16.7');
    expect(formatMs(250.4)).toBe('250');
  });

  it('scales bytes', () => {
    expect(formatBytes(512)).toBe('512 B');
    expect(formatBytes(2048)).toBe('2.0 KB');
    expect(formatBytes(8.25 * 1024 * 1024)).toBe('8.3 MB');
    expect(formatBytes(3 * 1024 ** 3)).toBe('3.00 GB');
  });
});

describe('sparklinePoints', () => {
  it('right-aligns the newest sample and flips y', () => {
    expect(sparklinePoints([0, 10], 3, 20, 10, 10)).toBe('10.0,10.0 20.0,0.0');
  });

  it('clips values above the scale', () => {
    expect(sparklinePoints([500], 1, 10, 10, 10)).toBe('0.0,0.0');
  });

  it('is empty without samples', () => {
    expect(sparklinePoints([], 120, 120, 32, 33)).toBe('');
  });
});
