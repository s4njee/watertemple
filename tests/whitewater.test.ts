import { describe, expect, it } from 'vitest';
import { createParticleSystem, createWhitewater } from '../src/gpu/particles/index.ts';
import { expectValid, getTestDevice } from './gpu.ts';

const bounds = { min: [-1, -1, -1] as const, max: [1, 1, 1] as const };

describe('secondary whitewater', () => {
  it('classifies spray, surface foam and submerged bubbles from local occupancy', async () => {
    const device = await getTestDevice();
    await expectValid(device, async () => {
      for (const [neighbors, kind] of [
        [1, 1],
        [10, 2],
        [30, 3],
      ]) {
        const source = createParticleSystem({ device, capacity: neighbors!, radius: 0.01 });
        source.fill(neighbors!, (_i, p) => {
          p.position = [0, 0, 0];
          p.velocity = [0.5, 0, 0];
        });
        const foam = createWhitewater({
          device,
          capacity: 1,
          radius: 0.005,
          searchRadius: 0.1,
          bounds,
        });
        foam.params.emission = 0;
        foam.system.fill(1, (_i, p) => {
          p.position = [0, 0, 0];
          p.velocityW = 1;
        });
        const encoder = device.createCommandEncoder();
        foam.update(encoder, source.particles, 1 / 60);
        device.queue.submit([encoder.finish()]);
        const positions = await foam.system.read();
        const velocities = await foam.system.read('velocity');
        expect(positions[3]).toBe(kind);
        expect(velocities[3]).toBeCloseTo(1 - 1 / 60);
        if (kind === 1) expect(velocities[1]).toBeLessThan(0);
        if (kind === 2) expect(velocities[0]).toBeGreaterThan(0);
        if (kind === 3) expect(velocities[1]).toBeGreaterThan(0);
        foam.destroy();
        foam.destroy();
        source.destroy();
      }
    });
  });

  it('emits only from energetic water and expires without a primary source', async () => {
    const device = await getTestDevice();
    await expectValid(device, async () => {
      const source = createParticleSystem({ device, capacity: 1, radius: 0.01 });
      source.fill(1, (_i, p) => {
        p.scalar = 1;
        p.velocity = [1, 0, 0];
      });
      const foam = createWhitewater({
        device,
        capacity: 64,
        radius: 0.005,
        searchRadius: 0.1,
        bounds,
      });
      foam.params.emission = 10000;
      let encoder = device.createCommandEncoder();
      foam.update(encoder, source.particles, 1 / 60);
      device.queue.submit([encoder.finish()]);
      expect((await foam.system.read()).filter((_v, i) => i % 4 === 3 && _v === 1).length).toBe(64);
      source.setCount(0);
      foam.params.emission = 0;
      // Explicitly near expiry: no count readback or timing-dependent loop is needed.
      foam.system.fill(64, (_i, p) => {
        p.velocityW = 0.001;
      });
      encoder = device.createCommandEncoder();
      foam.update(encoder, source.particles, 1 / 60);
      device.queue.submit([encoder.finish()]);
      expect((await foam.system.read()).filter((_v, i) => i % 4 === 3 && _v !== 0)).toHaveLength(0);
      foam.reset();
      expect((await foam.system.read('velocity')).every((v) => v === 0)).toBe(true);
      foam.destroy();
      source.destroy();
    });
  });
});
