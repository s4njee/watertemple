import { describe, expect, it } from 'vitest';
import {
  boxToObstacle,
  edgeMotion,
  edgeWake,
  isVisibleStyle,
  maxCornerRadiusPx,
  parseCornerRadius,
  rectsChanged,
  trackDomRects,
  type Box,
} from '../src/gpu/dom-rects.ts';
import type { ObstacleRect } from '../src/gpu/sim/obstacles.ts';

// The geometry is pure: plain objects stand in for DOMRects. There is no DOM in these tests.

/** A canvas 400 x 200 CSS pixels with its top-left at (100, 50) in the viewport. */
const canvas: Box = { left: 100, top: 50, width: 400, height: 200 };
const element = (left: number, top: number, width: number, height: number): Box => ({
  left,
  top,
  width,
  height,
});

describe('boxToObstacle', () => {
  it('maps a box inside the canvas to canvas uv', () => {
    const o = boxToObstacle(element(200, 100, 100, 50), canvas)!;
    expect(o.rect.x).toBeCloseTo(0.25);
    expect(o.rect.y).toBeCloseTo(0.25);
    expect(o.rect.width).toBeCloseTo(0.25);
    expect(o.rect.height).toBeCloseTo(0.25);
    expect(o.rect.radius).toBe(0);
    expect(o.box).toEqual({ left: 0.25, top: 0.25, right: 0.5, bottom: 0.5 });
  });

  it('scales the corner radius to image heights (canvas height = 1) and keeps it round', () => {
    const o = boxToObstacle(element(200, 100, 100, 50), canvas, { radiusPx: 10 })!;
    expect(o.rect.radius).toBeCloseTo(10 / 200);
    // The same radius on a canvas twice as big (in CSS px) is the same fraction of its height.
    const big = boxToObstacle(
      element(0, 0, 200, 100),
      { left: 0, top: 0, width: 800, height: 400 },
      {
        radiusPx: 20,
      },
    )!;
    expect(big.rect.radius).toBeCloseTo(20 / 400);
  });

  it('limits the radius to half the shorter side', () => {
    const o = boxToObstacle(element(200, 100, 100, 40), canvas, { radiusPx: 999 })!;
    expect(o.rect.radius).toBeCloseTo(20 / 200);
  });

  it('clips to the canvas, widened by the radius so a rounded corner is not cut at the edge', () => {
    const square = boxToObstacle(element(0, 0, 1000, 1000), canvas)!;
    expect(square.rect).toMatchObject({ x: 0, y: 0, width: 1, height: 1 });
    // The unclipped edges are kept (they say where the real edges are).
    expect(square.box.left).toBeLessThan(0);
    expect(square.box.right).toBeGreaterThan(1);

    const round = boxToObstacle(element(0, 0, 1000, 1000), canvas, { radiusPx: 20 })!;
    expect(round.rect.radius).toBeCloseTo(0.1);
    expect(round.rect.y).toBeCloseTo(-0.1);
    expect(round.rect.x).toBeCloseTo(-0.1 / 2);
    expect(round.rect.x + round.rect.width).toBeCloseTo(1 + 0.1 / 2);
  });

  it('skips boxes that do not overlap the canvas, including ones that only touch it', () => {
    expect(boxToObstacle(element(600, 100, 50, 50), canvas)).toBeUndefined();
    expect(boxToObstacle(element(0, 0, 100, 100), canvas)).toBeUndefined();
    expect(boxToObstacle(element(100, 250, 100, 50), canvas)).toBeUndefined();
    expect(boxToObstacle(element(50, 100, 100, 50), canvas)).toBeDefined();
  });

  it('skips hidden, zero-size and non-finite boxes, and an unusable canvas', () => {
    expect(boxToObstacle(element(200, 100, 0, 50), canvas)).toBeUndefined();
    expect(boxToObstacle(element(200, 100, 50, 0), canvas)).toBeUndefined();
    expect(boxToObstacle(element(200, 100, 0.4, 50), canvas)).toBeUndefined();
    expect(boxToObstacle(element(200, 100, -10, 50), canvas)).toBeUndefined();
    expect(boxToObstacle(element(Number.NaN, 100, 50, 50), canvas)).toBeUndefined();
    expect(boxToObstacle(element(200, 100, Infinity, 50), canvas)).toBeUndefined();
    expect(boxToObstacle(element(200, 100, 50, 50), { ...canvas, width: 0 })).toBeUndefined();
    expect(
      boxToObstacle(element(200, 100, 50, 50), { ...canvas, height: Number.NaN }),
    ).toBeUndefined();
  });

  it('grows and shrinks by the outset, the radius following', () => {
    const grown = boxToObstacle(element(200, 100, 100, 50), canvas, { outsetPx: 10, radiusPx: 8 })!;
    expect(grown.rect.x).toBeCloseTo((190 - 100) / 400);
    expect(grown.rect.width).toBeCloseTo(120 / 400);
    expect(grown.rect.radius).toBeCloseTo(18 / 200);
    const shrunk = boxToObstacle(element(200, 100, 100, 50), canvas, { outsetPx: -30 });
    expect(shrunk).toBeUndefined();
  });
});

describe('corner radius from computed style', () => {
  it('reads px, percent and elliptical corners', () => {
    expect(parseCornerRadius('12px', 200, 100)).toBe(12);
    expect(parseCornerRadius('50%', 200, 100)).toBe(50);
    expect(parseCornerRadius('10px 4px', 200, 100)).toBe(4);
    expect(parseCornerRadius('0px', 200, 100)).toBe(0);
    expect(parseCornerRadius('', 200, 100)).toBe(0);
    expect(parseCornerRadius(undefined, 200, 100)).toBe(0);
    expect(parseCornerRadius('garbage', 200, 100)).toBe(0);
  });

  it('takes the largest corner and never more than half the shorter side', () => {
    const style = {
      borderTopLeftRadius: '4px',
      borderTopRightRadius: '16px',
      borderBottomRightRadius: '0px',
      borderBottomLeftRadius: '8px',
    };
    expect(maxCornerRadiusPx(style, 200, 100)).toBe(16);
    expect(maxCornerRadiusPx({ borderTopLeftRadius: '9999px' }, 200, 100)).toBe(50);
    expect(maxCornerRadiusPx({}, 200, 100)).toBe(0);
  });
});

describe('visibility', () => {
  it('rejects display none, hidden visibility and a nearly transparent element', () => {
    expect(isVisibleStyle({ display: 'block', visibility: 'visible', opacity: '1' })).toBe(true);
    expect(isVisibleStyle({})).toBe(true);
    expect(isVisibleStyle({ display: 'none' })).toBe(false);
    expect(isVisibleStyle({ visibility: 'hidden' })).toBe(false);
    expect(isVisibleStyle({ opacity: '0' })).toBe(false);
    expect(isVisibleStyle({ opacity: '0.5' })).toBe(true);
  });
});

describe('rectsChanged', () => {
  const a: ObstacleRect[] = [{ x: 0.1, y: 0.1, width: 0.2, height: 0.2, radius: 0.02 }];

  it('is false for equal lists and for movement below the epsilon', () => {
    expect(
      rectsChanged(
        a,
        a.map((r) => ({ ...r })),
      ),
    ).toBe(false);
    expect(rectsChanged(a, [{ ...a[0]!, x: 0.10005 }], 1e-3)).toBe(false);
    expect(rectsChanged([], [])).toBe(false);
  });

  it('is true when any component, the radius or the count changes', () => {
    expect(rectsChanged(a, [{ ...a[0]!, x: 0.11 }])).toBe(true);
    expect(rectsChanged(a, [{ ...a[0]!, height: 0.3 }])).toBe(true);
    expect(rectsChanged(a, [{ ...a[0]!, radius: 0.05 }])).toBe(true);
    expect(rectsChanged(a, [])).toBe(true);
    expect(rectsChanged([], a)).toBe(true);
  });

  it('treats a missing radius as 0', () => {
    expect(
      rectsChanged(
        [{ x: 0, y: 0, width: 1, height: 1 }],
        [{ x: 0, y: 0, width: 1, height: 1, radius: 0 }],
      ),
    ).toBe(false);
  });
});

describe('edge motion and wake', () => {
  const still = { left: 0.2, top: 0.2, right: 0.4, bottom: 0.4 };

  it('measures each edge in uv per second', () => {
    const next = { left: 0.21, top: 0.2, right: 0.43, bottom: 0.4 };
    const m = edgeMotion(still, next, 0.1)!;
    expect(m.left).toBeCloseTo(0.1);
    expect(m.right).toBeCloseTo(0.3);
    expect(m.top).toBeCloseTo(0);
  });

  it('refuses a missing interval, a slept tab and a teleport', () => {
    expect(edgeMotion(still, still, 0)).toBeUndefined();
    expect(edgeMotion(still, still, 2)).toBeUndefined();
    expect(edgeMotion(still, { ...still, left: 0.7, right: 0.9 }, 0.016)).toBeUndefined();
  });

  const options = { aspect: 2, dt: 1 / 60, radius: 0.03, gain: 40 };

  it('raises water ahead of an advancing edge and dents it behind a retreating one', () => {
    // The whole box moves right at 0.3 uv/s: the right edge advances, the left retreats.
    const drops = edgeWake(still, { left: 0.3, right: 0.3, top: 0, bottom: 0 }, options);
    const ahead = drops.filter((d) => d.x > 0.4);
    const behind = drops.filter((d) => d.x < 0.2);
    expect(ahead.length).toBeGreaterThan(0);
    expect(behind.length).toBeGreaterThan(0);
    for (const d of ahead) expect(d.strength).toBeGreaterThan(0);
    for (const d of behind) expect(d.strength).toBeLessThan(0);
    // Equal and opposite: the volume the box shoves is the volume that follows it.
    const sum = drops.reduce((s, d) => s + d.strength, 0);
    expect(Math.abs(sum)).toBeLessThan(1e-9);
    // Drops sit within the edge's extent.
    for (const d of drops) {
      expect(d.y).toBeGreaterThanOrEqual(0.2);
      expect(d.y).toBeLessThanOrEqual(0.4);
    }
  });

  it('does nothing for a box at rest, or an edge lying on the canvas boundary', () => {
    expect(edgeWake(still, { left: 0, right: 0, top: 0, bottom: 0 }, options)).toEqual([]);
    const clipped = { left: -0.5, top: 0.2, right: 1.5, bottom: 0.4 };
    expect(edgeWake(clipped, { left: 0.5, right: 0.5, top: 0, bottom: 0 }, options)).toEqual([]);
  });

  it('saturates: a huge speed is not a huge impulse', () => {
    const fast = edgeWake(still, { left: 0, right: 5, top: 0, bottom: 0 }, options);
    const faster = edgeWake(still, { left: 0, right: 500, top: 0, bottom: 0 }, options);
    expect(Math.abs(faster[0]!.strength)).toBeLessThan(Math.abs(fast[0]!.strength) * 1.2);
  });

  it('keeps the impulse per second constant whatever the frame rate', () => {
    const motion = { left: 0, right: 0.3, top: 0, bottom: 0 };
    const total = (dt: number, frames: number) => {
      let sum = 0;
      for (let i = 0; i < frames; i++) {
        sum += edgeWake(still, motion, { ...options, dt }).reduce((s, d) => s + d.strength * 1, 0);
      }
      return sum;
    };
    // The per-drop radius does not depend on dt, so the sums over one second match.
    expect(total(1 / 120, 120)).toBeCloseTo(total(1 / 60, 60), 6);
  });
});

describe('without a DOM', () => {
  it('returns nothing and does not throw', () => {
    const stub = Object.assign(new EventTarget(), {
      getBoundingClientRect: () => ({ left: 0, top: 0, width: 64, height: 48 }),
    }) as unknown as HTMLCanvasElement;
    const tracker = trackDomRects(stub);
    const result = tracker.update(1 / 60);
    expect(result.rects).toEqual([]);
    expect(result.obstacles).toEqual([]);
    expect(result.found).toBe(0);
    expect(result.changed).toBe(false);
    expect(tracker.root).toBeUndefined();
    tracker.invalidate();
    expect(() => tracker.destroy()).not.toThrow();
    expect(tracker.update(1 / 60).rects).toEqual([]);
  });

  it('tolerates a canvas that is not an element at all', () => {
    const tracker = trackDomRects({} as HTMLCanvasElement);
    expect(tracker.update(0.016).rects).toEqual([]);
    tracker.destroy();
  });
});
