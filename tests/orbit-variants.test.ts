import { describe, expect, it } from 'vitest';
import {
  biolumePulse,
  stepSpark,
  type EmberSpark,
} from '../src/effects/orbit-variants/dynamics.ts';
import { ORBIT_VARIATIONS } from '../src/effects/orbit-variants/catalog.ts';
import { loadEffect } from '../src/effects/registry.ts';

describe('bioluminescent response', () => {
  it('brightens on sharp turns and near the pointer, but not on straight travel', () => {
    expect(biolumePulse(0, [0.2, 0], [0.2, 0], 1 / 60, 0)).toBe(0);
    expect(biolumePulse(0, [0.2, 0], [0, 0.2], 1 / 60, 0)).toBe(1);
    expect(biolumePulse(0, [0.2, 0], [0.2, 0], 1 / 60, 0.8)).toBe(0.8);
    expect(biolumePulse(0, [0, 0], [0, 0], 1 / 60, 0)).toBe(0);
  });
  it('fades at the same rate across frame rates', () => {
    const decay = (fps: number) => {
      let pulse = 1;
      for (let i = 0; i < fps; i++) pulse = biolumePulse(pulse, [0.2, 0], [0.2, 0], 1 / fps, 0);
      return pulse;
    };
    expect(decay(30)).toBeCloseTo(decay(120), 8);
    expect(decay(60)).toBeLessThan(0.04);
  });
});

describe('detached ember sparks', () => {
  it('coasts away, rises, slows down and expires', () => {
    const spark: EmberSpark = { x: 0.4, y: 0.6, vx: 0.2, vy: -0.1, age: 0, life: 1, radius: 0.002 };
    for (let i = 0; i < 30; i++) expect(stepSpark(spark, 1 / 60)).toBe(true);
    expect(spark.x).toBeGreaterThan(0.4);
    expect(spark.y).toBeLessThan(0.6);
    expect(spark.vx).toBeLessThan(0.2);
    expect(stepSpark(spark, 0.6)).toBe(false);
  });
});

// Verify the catalog and registered effect agree; otherwise standalone and gallery differ silently.
describe('material defaults', () => {
  it.each(Object.entries(ORBIT_VARIATIONS).filter(([, variation]) => variation.style > 0))(
    '%s has the intended palette and motion',
    async (id, variation) => {
      const effect = await loadEffect(id);
      for (const [key, value] of Object.entries(variation.params)) {
        const definition = effect.params[key];
        expect(definition, `${id}.${key} exists`).toBeDefined();
        expect(definition!.default, `${id}.${key}`).toEqual(value);
        if (definition!.type === 'number') {
          expect(value).toBeGreaterThanOrEqual(definition!.min);
          expect(value).toBeLessThanOrEqual(definition!.max);
        }
        if (definition!.type === 'select') expect(definition!.options).toContain(value);
      }
    },
  );
});
