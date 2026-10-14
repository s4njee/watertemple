import { describe, expect, it } from 'vitest';
import {
  boxDistance,
  createHead,
  headForce,
  stepHead,
  type Head,
  type HeadBox,
  type HeadField,
} from '../src/effects/ink-orbit/heads.ts';

/** mulberry32, as the effect uses. */
function createRandom(seed: number): () => number {
  let state = seed >>> 0;
  return () => {
    state = (state + 0x6d2b79f5) >>> 0;
    let t = state;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

const card: HeadBox = { cx: 0.8, cy: 0.5, hx: 0.2, hy: 0.1, radius: 0.04, spin: 1 };

function field(overrides: Partial<HeadField> = {}): HeadField {
  return {
    boxes: [card],
    aspect: 1.6,
    time: 0,
    pull: 1,
    swirl: 0.48,
    shell: 0.03,
    wander: 1,
    chaos: 1,
    maxSpeed: 0.5,
    clearance: 0.01,
    ...overrides,
  };
}

/** A head with no character of its own, at rest at (x, y), that never lets go or dies. */
function plainHead(x: number, y: number): Head {
  return {
    x,
    y,
    vx: 0,
    vy: 0,
    age: 0,
    life: 1e9,
    hue: 0,
    colour: 0,
    shell: 1,
    swirl: 1,
    drag: 1.6,
    pace: 1,
    size: 1,
    offset: 0,
    calm: 1e9,
    loose: 0,
  };
}

describe('boxDistance', () => {
  it('is the distance to the edge, negative inside, with the outward normal', () => {
    const right = boxDistance(1.1, 0.5, card);
    expect(right.d).toBeCloseTo(0.1, 6);
    expect([right.nx, right.ny]).toEqual([1, 0]);

    const above = boxDistance(0.8, 0.3, card);
    expect(above.d).toBeCloseTo(0.1, 6);
    expect([above.nx, above.ny]).toEqual([0, -1]);

    expect(boxDistance(0.8, 0.5, card).d).toBeCloseTo(-0.1, 6);
  });

  it('rounds the corners', () => {
    // On the diagonal through the bottom-right corner's centre, a radius away from it.
    const c = Math.SQRT1_2;
    const corner = boxDistance(0.96 + 0.04 * c, 0.56 + 0.04 * c, card);
    expect(corner.d).toBeCloseTo(0, 6);
    expect(corner.nx).toBeCloseTo(c, 6);
    expect(corner.ny).toBeCloseTo(c, 6);
  });
});

describe('headForce', () => {
  const still = { wander: 0, chaos: 0 };

  it('pulls a far head in and pushes a close one out, towards the shell', () => {
    const far = headForce(plainHead(1.2, 0.5), field(still));
    expect(far[0]).toBeLessThan(0);
    const close = headForce(plainHead(1.005, 0.5), field(still));
    expect(close[0]).toBeGreaterThan(0);
  });

  it('swings clockwise on screen for a positive spin and the other way for a negative one', () => {
    // To the right of the card, clockwise (y down) is downwards.
    const cw = headForce(plainHead(1.03, 0.5), field(still));
    expect(cw[1]).toBeGreaterThan(0);
    const ccw = headForce(
      plainHead(1.03, 0.5),
      field({ ...still, boxes: [{ ...card, spin: -1 }] }),
    );
    expect(ccw[1]).toBeLessThan(0);
  });

  it('only pushes away from an element that opted out', () => {
    const force = headForce(
      plainHead(1.03, 0.5),
      field({ ...still, boxes: [{ ...card, spin: 0 }] }),
    );
    expect(force[0]).toBeGreaterThan(0);
    expect(force[1]).toBeCloseTo(0, 6);
  });

  it('feels nothing from the elements when the pull is off', () => {
    const force = headForce(plainHead(1.03, 0.5), field({ ...still, pull: 0 }));
    expect(Math.hypot(force[0], force[1])).toBeLessThan(1e-3);
  });
});

describe('stepHead', () => {
  it('settles a plain head into an orbit round the card, in the direction of its spin', () => {
    const f = field({ wander: 0, chaos: 0 });
    const head = plainHead(1.25, 0.5);
    let turned = 0;
    let angle = Math.atan2(head.y - card.cy, head.x - card.cx);
    const random = createRandom(1);
    for (let i = 0; i < 60 * 20; i++) {
      expect(stepHead(head, f, 1 / 60, random)).toBe(true);
      const next = Math.atan2(head.y - card.cy, head.x - card.cx);
      let delta = next - angle;
      if (delta > Math.PI) delta -= 2 * Math.PI;
      if (delta < -Math.PI) delta += 2 * Math.PI;
      turned += delta;
      angle = next;
    }
    // With y down, a growing atan2 angle is clockwise on screen. More than one full turn in 20 s.
    expect(turned).toBeGreaterThan(2 * Math.PI);
    const { d } = boxDistance(head.x, head.y, card);
    expect(d).toBeGreaterThan(0);
    expect(d).toBeLessThan(0.15);
  });

  it('keeps every head on the canvas, outside the elements and under its top speed', () => {
    const random = createRandom(7);
    const f = field({
      boxes: [card, { cx: 0.3, cy: 0.25, hx: 0.18, hy: 0.06, radius: 0.02, spin: -1 }],
      chaos: 2,
      wander: 2,
    });
    const heads = Array.from({ length: 12 }, (_, i) => createHead(i, f, random));
    for (let frame = 0; frame < 60 * 30; frame++) {
      f.time = frame / 60;
      for (const head of heads) {
        stepHead(head, f, 1 / 60, random);
        expect(head.x).toBeGreaterThan(0);
        expect(head.x).toBeLessThan(f.aspect);
        expect(head.y).toBeGreaterThan(0);
        expect(head.y).toBeLessThan(1);
        expect(Math.hypot(head.vx, head.vy)).toBeLessThanOrEqual(f.maxSpeed + 1e-9);
        for (const box of f.boxes) {
          expect(boxDistance(head.x, head.y, box).d).toBeGreaterThan(0);
        }
      }
    }
  });

  it('reports a rebirth as a jump, not as travel', () => {
    const random = createRandom(3);
    const f = field();
    const head = plainHead(0.3, 0.8);
    head.life = 0.05;
    let jumped = false;
    for (let i = 0; i < 10 && !jumped; i++) jumped = !stepHead(head, f, 1 / 60, random);
    expect(jumped).toBe(true);
    expect(head.age).toBeLessThan(0.05);
  });

  it('gives the same paths for the same seed', () => {
    const run = () => {
      const random = createRandom(11);
      const f = field();
      const heads = Array.from({ length: 4 }, (_, i) => createHead(i, f, random));
      for (let frame = 0; frame < 300; frame++) {
        f.time = frame / 60;
        for (const head of heads) stepHead(head, f, 1 / 60, random);
      }
      return heads.map((head) => [head.x, head.y]);
    };
    expect(run()).toEqual(run());
  });
});
