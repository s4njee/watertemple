import { describe, expect, it, vi } from 'vitest';
import {
  createBillboardRenderer,
  type BillboardRenderer,
  type BillboardStyle,
} from '../src/gpu/particles/billboards.ts';
import { orthographicCamera } from '../src/gpu/particles/cameras.ts';
import { createRandom, sampleEmitter } from '../src/gpu/particles/emitter.ts';
import { createIntegrator } from '../src/gpu/particles/integrator.ts';
import { createParticleSystem } from '../src/gpu/particles/system.ts';
import { computeCamera, projectPoint } from '../src/gpu/scene/camera.ts';
import { createDepthTarget } from '../src/gpu/scene/depth.ts';
import { getChunk } from '../src/wgsl/index.ts';
import { expectValid, getTestDevice, readTexture } from './gpu.ts';
import { scalars } from './wgsl-eval.ts';

// Story 7.1: the particle framework. Storage and bookkeeping, the integrator against the analytic solution, and
// the billboard renderer against the camera maths, all on the real GPU.

vi.setConfig({ testTimeout: 60_000 });

async function submit(device: GPUDevice, record: (encoder: GPUCommandEncoder) => void) {
  const encoder = device.createCommandEncoder();
  record(encoder);
  device.queue.submit([encoder.finish()]);
  await device.queue.onSubmittedWorkDone();
}

// --- Storage and bookkeeping -------------------------------------------------------------------------

describe('particle system', () => {
  it('reads back exactly what was spawned, through the ParticleSet view', async () => {
    const device = await getTestDevice();
    const system = createParticleSystem({ device, capacity: 100, radius: 0.05 });
    await expectValid(device, async () => {
      system.fill(40, (i, p) => {
        p.position[0] = i;
        p.position[1] = i * 2;
        p.position[2] = -i;
        p.scalar = i / 10;
        p.velocity[0] = 1 + i;
        p.velocityW = 3;
      });
      const set = system.particles;
      expect(set.count).toBe(40);
      expect(set.capacity).toBe(100);
      expect(set.radius).toBe(0.05);
      expect(set.positions).toBe(system.positions);
      expect(set.velocities).toBe(system.velocities);
      const positions = await system.read('position');
      const velocities = await system.read('velocity');
      expect(positions.length).toBe(160);
      for (let i = 0; i < 40; i++) {
        expect(Array.from(positions.subarray(i * 4, i * 4 + 4))).toEqual([
          i,
          i * 2,
          -i,
          Math.fround(i / 10),
        ]);
        expect(Array.from(velocities.subarray(i * 4, i * 4 + 4))).toEqual([1 + i, 0, 0, 3]);
      }
    });
    // The buffers have the usages the ParticleSet contract promises.
    for (const buffer of [system.positions, system.velocities!]) {
      for (const usage of [
        GPUBufferUsage.STORAGE,
        GPUBufferUsage.VERTEX,
        GPUBufferUsage.COPY_SRC,
        GPUBufferUsage.COPY_DST,
      ]) {
        expect(buffer.usage & usage).toBe(usage);
      }
    }
    system.destroy();
  });

  it('keeps z at zero in two dimensions', async () => {
    const device = await getTestDevice();
    const system = createParticleSystem({ device, capacity: 10, radius: 0.1, dimension: 2 });
    system.fill(5, (i, p) => {
      p.position[0] = i;
      p.position[2] = 9;
    });
    const positions = await system.read('position');
    for (let i = 0; i < 5; i++) expect(positions[i * 4 + 2]).toBe(0);
    expect(system.emit({ shape: 'sphere', extent: [1, 1, 1] }, 5)).toBe(5);
    const emitted = await system.read('position');
    for (let i = 5; i < 10; i++) expect(emitted[i * 4 + 2]).toBe(0);
    system.destroy();
  });

  it('sets from typed arrays and zero-fills what it was not given', async () => {
    const device = await getTestDevice();
    const system = createParticleSystem({
      device,
      capacity: 8,
      radius: 1,
      extras: { age: 1, uv: 2 },
    });
    system.set({
      positions: new Float32Array([1, 2, 3, 4, 5, 6, 7, 8]),
      extras: { uv: new Float32Array([0.5, 0.25, 0.75, 1]) },
    });
    expect(system.count).toBe(2);
    expect(Array.from(await system.read('position'))).toEqual([1, 2, 3, 4, 5, 6, 7, 8]);
    expect(Array.from(await system.read('velocity'))).toEqual(new Array(8).fill(0));
    expect(Array.from(await system.read('age'))).toEqual([0, 0]);
    expect(Array.from(await system.read('uv'))).toEqual([0.5, 0.25, 0.75, 1]);
    expect(system.components('age')).toBe(1);
    expect(system.attributeNames).toEqual(['position', 'velocity', 'age', 'uv']);
    expect(() => system.attribute('nope')).toThrow(/no attribute/);
    system.destroy();
  });

  it('emits from a description, clipping at capacity, then recycles the oldest in a ring', async () => {
    const device = await getTestDevice();
    const system = createParticleSystem({ device, capacity: 10, radius: 0.1 });
    const emitter = {
      center: [1, 2, 3] as const,
      direction: [0, 0, 1] as const,
      speed: 4,
      lifetime: [2, 2] as const,
      scalar: 1,
    };
    expect(system.emit(emitter, 6)).toBe(6);
    expect(system.count).toBe(6);
    expect(system.emit(emitter, 100)).toBe(4);
    expect(system.count).toBe(10);
    expect(system.emit(emitter, 5)).toBe(0); // full, not a ring
    let positions = await system.read('position');
    const velocities = await system.read('velocity');
    for (let i = 0; i < 10; i++) {
      expect(Array.from(positions.subarray(i * 4, i * 4 + 4))).toEqual([1, 2, 3, 1]);
      expect(Array.from(velocities.subarray(i * 4, i * 4 + 4))).toEqual([0, 0, 4, 2]);
    }
    // Ring: the next 3 overwrite slots 0..2, the 4 after that slots 3..6.
    expect(system.emit({ ...emitter, scalar: 2 }, 3, { ring: true })).toBe(3);
    expect(system.emit({ ...emitter, scalar: 3 }, 4, { ring: true })).toBe(4);
    expect(system.count).toBe(10);
    positions = await system.read('position');
    expect(Array.from({ length: 10 }, (_, i) => positions[i * 4 + 3])).toEqual([
      2, 2, 2, 3, 3, 3, 3, 1, 1, 1,
    ]);
    // A ring write that wraps the end of the buffer.
    expect(system.emit({ ...emitter, scalar: 4 }, 6, { ring: true })).toBe(6);
    positions = await system.read('position');
    expect(Array.from({ length: 10 }, (_, i) => positions[i * 4 + 3])).toEqual([
      4, 4, 4, 3, 3, 3, 3, 4, 4, 4,
    ]);
    system.destroy();
  });

  it('samples emitter shapes inside their volumes with the right velocities', () => {
    const random = createRandom(5);
    const n = 3000;
    const positions = new Float32Array(n * 4);
    const velocities = new Float32Array(n * 4);

    sampleEmitter(
      { shape: 'box', center: [1, 2, 3], extent: [0.5, 1, 2] },
      3,
      random,
      n,
      positions,
      velocities,
    );
    for (let i = 0; i < n; i++) {
      expect(Math.abs(positions[i * 4]! - 1)).toBeLessThanOrEqual(0.5);
      expect(Math.abs(positions[i * 4 + 1]! - 2)).toBeLessThanOrEqual(1);
      expect(Math.abs(positions[i * 4 + 2]! - 3)).toBeLessThanOrEqual(2);
    }

    sampleEmitter(
      { shape: 'sphere', center: [0, 0, 0], extent: [2, 0, 0] },
      3,
      random,
      n,
      positions,
      velocities,
    );
    let outsideHalf = 0;
    for (let i = 0; i < n; i++) {
      const r = Math.hypot(positions[i * 4]!, positions[i * 4 + 1]!, positions[i * 4 + 2]!);
      expect(r).toBeLessThanOrEqual(2 + 1e-6);
      if (r > 1) outsideHalf++;
    }
    // Uniform in the ball: 7/8 of the volume lies beyond half the radius.
    expect(outsideHalf / n).toBeGreaterThan(0.84);
    expect(outsideHalf / n).toBeLessThan(0.91);

    sampleEmitter({ shape: 'disc', extent: [1, 0, 0] }, 3, random, n, positions, velocities);
    for (let i = 0; i < n; i++) {
      expect(positions[i * 4 + 1]).toBe(0);
      expect(Math.hypot(positions[i * 4]!, positions[i * 4 + 2]!)).toBeLessThanOrEqual(1 + 1e-6);
    }

    // A cone: every velocity has the requested speed and lies within `spread` of the direction.
    sampleEmitter(
      { direction: [0, 1, 0], speed: 3, spread: 0.3, lifetime: [1, 2] },
      3,
      random,
      n,
      positions,
      velocities,
    );
    for (let i = 0; i < n; i++) {
      const v = velocities.subarray(i * 4, i * 4 + 3);
      expect(Math.hypot(v[0]!, v[1]!, v[2]!)).toBeCloseTo(3, 4);
      expect(v[1]! / 3).toBeGreaterThanOrEqual(Math.cos(0.3) - 1e-5);
      expect(velocities[i * 4 + 3]).toBeGreaterThanOrEqual(1);
      expect(velocities[i * 4 + 3]).toBeLessThanOrEqual(2);
    }

    // The 2D fan stays in the plane.
    sampleEmitter(
      { direction: [1, 0, 0], speed: 2, spread: 0.5 },
      2,
      random,
      n,
      positions,
      velocities,
    );
    for (let i = 0; i < n; i++) {
      expect(velocities[i * 4 + 2]).toBe(0);
      expect(Math.hypot(velocities[i * 4]!, velocities[i * 4 + 1]!)).toBeCloseTo(2, 4);
      expect(Math.abs(Math.atan2(velocities[i * 4 + 1]!, velocities[i * 4]!))).toBeLessThanOrEqual(
        0.5 + 1e-5,
      );
    }
  });

  it('grows with zeroed slots, shrinks, and keeps the data when the capacity changes', async () => {
    const device = await getTestDevice();
    const system = createParticleSystem({ device, capacity: 16, radius: 0.1 });
    system.fill(8, (i, p) => {
      p.position[0] = i + 1;
      p.velocityW = 5;
    });
    system.setCount(4);
    expect(system.count).toBe(4);
    system.setCount(10); // slots 4..9 held older particles; growing must zero them
    expect(system.count).toBe(10);
    let positions = await system.read('position');
    expect(Array.from({ length: 10 }, (_, i) => positions[i * 4])).toEqual([
      1, 2, 3, 4, 0, 0, 0, 0, 0, 0,
    ]);
    const velocities = await system.read('velocity');
    expect(velocities[4 * 4 + 3]).toBe(0);

    const generation = system.generation;
    const before = system.positions;
    system.resize(64);
    expect(system.generation).toBe(generation + 1);
    expect(system.capacity).toBe(64);
    expect(system.particles.capacity).toBe(64);
    expect(system.positions).not.toBe(before);
    expect(system.count).toBe(10);
    positions = await system.read('position');
    expect(Array.from({ length: 10 }, (_, i) => positions[i * 4])).toEqual([
      1, 2, 3, 4, 0, 0, 0, 0, 0, 0,
    ]);

    system.resize(3);
    expect(system.generation).toBe(generation + 2);
    expect(system.count).toBe(3);
    expect(Array.from((await system.read('position')).filter((_, i) => i % 4 === 0))).toEqual([
      1, 2, 3,
    ]);
    system.resize(3); // no change, no new generation
    expect(system.generation).toBe(generation + 2);

    expect(() => createParticleSystem({ device, capacity: 0, radius: 1 })).toThrow(RangeError);
    expect(() => createParticleSystem({ device, capacity: 2 ** 30, radius: 1 })).toThrow(
      /device limit/,
    );
    system.destroy();
  });

  it('reorders every attribute together through a permutation, and swaps buffers', async () => {
    const device = await getTestDevice();
    const system = createParticleSystem({
      device,
      capacity: 300,
      radius: 0.1,
      // Five attributes: more than one gather pass handles.
      extras: { a: 1, b: 2, c: 4 },
    });
    const n = 257;
    const rows = (width: number, salt: number) =>
      Float32Array.from(
        { length: n * width },
        (_, i) => Math.floor(i / width) * 10 + (i % width) + salt,
      );
    system.set({
      positions: rows(4, 0),
      velocities: rows(4, 0.25),
      extras: { a: rows(1, 0.5), b: rows(2, 0.75), c: rows(4, 0.125) },
    });
    // A permutation: reversal.
    const sorted = Uint32Array.from({ length: n }, (_, k) => n - 1 - k);
    const buffer = device.createBuffer({
      size: n * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(buffer, 0, sorted);
    const before = system.positions;
    await expectValid(device, () => submit(device, (e) => system.permute(e, buffer)));
    expect(system.positions).not.toBe(before);
    const check = async (name: string, width: number, salt: number) => {
      const data = await system.read(name);
      const expected = rows(width, salt);
      for (let k = 0; k < n; k++) {
        for (let c = 0; c < width; c++) {
          if (data[k * width + c] !== expected[(n - 1 - k) * width + c]) {
            throw new Error(`${name}: slot ${k} component ${c} did not move with its particle`);
          }
        }
      }
    };
    await check('position', 4, 0);
    await check('velocity', 4, 0.25);
    await check('a', 1, 0.5);
    await check('b', 2, 0.75);
    await check('c', 4, 0.125);
    buffer.destroy();
    system.destroy();
  });
});

// --- The integrator --------------------------------------------------------------------------------------

describe('integrator', () => {
  async function rig(options: { dimension?: 2 | 3 } = {}) {
    const device = await getTestDevice();
    const system = createParticleSystem({ device, capacity: 64, radius: 0.0, ...options });
    return { device, system };
  }

  it('reproduces the discrete ballistic solution of semi-implicit Euler exactly, and the analytic one to O(dt)', async () => {
    const { device, system } = await rig();
    system.fill(3, (i, p) => {
      p.position[0] = i;
      p.position[1] = 10;
      p.velocity[0] = 1;
      p.velocity[1] = 2 + i;
      p.velocity[2] = -1;
    });
    const dt = 1 / 240;
    const integrator = createIntegrator({
      system,
      gravity: [0, -9.81, 0.5],
      fixedStep: dt,
      scalar: 'speed',
    });
    const steps = 180; // 0.75 s
    await expectValid(device, () => submit(device, (e) => integrator.step(e, steps)));
    const positions = await system.read('position');
    const velocities = await system.read('velocity');
    const g = [0, -9.81, 0.5];
    for (let i = 0; i < 3; i++) {
      const p0 = [i, 10, 0];
      const v0 = [1, 2 + i, -1];
      for (let axis = 0; axis < 3; axis++) {
        // v_n = v0 + n g dt;  p_n = p0 + n v0 dt + g dt^2 n (n + 1) / 2
        const vn = v0[axis]! + steps * g[axis]! * dt;
        const pn =
          p0[axis]! + steps * v0[axis]! * dt + (g[axis]! * dt * dt * steps * (steps + 1)) / 2;
        expect(velocities[i * 4 + axis]).toBeCloseTo(vn, 3);
        expect(positions[i * 4 + axis]).toBeCloseTo(pn, 3);
        // The continuous solution differs by g dt t / 2 at most.
        const t = steps * dt;
        const exact = p0[axis]! + v0[axis]! * t + (g[axis]! * t * t) / 2;
        expect(Math.abs(positions[i * 4 + axis]! - exact)).toBeLessThan(
          Math.abs(g[axis]!) * dt * t * 0.51 + 1e-4,
        );
      }
      const speed = Math.hypot(velocities[i * 4]!, velocities[i * 4 + 1]!, velocities[i * 4 + 2]!);
      expect(positions[i * 4 + 3]).toBeCloseTo(speed, 4); // scalar: 'speed'
    }
    expect(integrator.time).toBeCloseTo(steps * dt, 6);
    integrator.destroy();
    system.destroy();
  });

  it('applies linear damping as exp(-k t)', async () => {
    const { device, system } = await rig();
    system.fill(1, (_i, p) => {
      p.velocity[0] = 4;
    });
    const integrator = createIntegrator({
      system,
      gravity: [0, 0, 0],
      damping: 2,
      fixedStep: 1 / 200,
    });
    await submit(device, (e) => integrator.step(e, 200)); // 1 s
    const velocities = await system.read('velocity');
    expect(velocities[0]).toBeCloseTo(4 * Math.exp(-2), 3);
    integrator.destroy();
    system.destroy();
  });

  it('bounces off the box with the configured restitution', async () => {
    for (const restitution of [0.5, 0.8]) {
      const { device, system } = await rig();
      system.fill(1, (_i, p) => {
        p.position[0] = 0.5;
        p.position[1] = 1;
        p.position[2] = 0.5;
      });
      const dt = 1 / 4000;
      const integrator = createIntegrator({
        system,
        gravity: [0, -9.81, 0],
        fixedStep: dt,
        restitution,
        bounds: { min: [0, 0, 0], max: [1, 5, 1] },
        margin: 0,
      });
      // Falls for sqrt(2 y / g), rebounds with restitution times the impact speed, and peaks at
      // restitution^2 times the drop height after a further e v / g.
      const fall = Math.sqrt(2 / 9.81);
      const impact = 9.81 * fall;
      const apexTime = fall + (restitution * impact) / 9.81;
      await submit(device, (e) => integrator.step(e, Math.round(apexTime / dt)));
      const positions = await system.read('position');
      const velocities = await system.read('velocity');
      expect(positions[1]).toBeCloseTo(restitution * restitution, 2);
      expect(Math.abs(velocities[1]!)).toBeLessThan(0.05);
      // Never below the floor.
      expect(positions[1]).toBeGreaterThanOrEqual(0);
      integrator.destroy();
      system.destroy();
    }
  });

  it('bounces off every wall, keeps particles inside, and applies friction on contact', async () => {
    const { device, system } = await rig();
    system.fill(2, (i, p) => {
      p.position[0] = 0.9;
      p.position[1] = 0.5;
      p.position[2] = 0.5;
      p.velocity[0] = i === 0 ? 3 : 0;
      p.velocity[1] = i === 1 ? -3 : 0;
      p.velocity[2] = 1;
    });
    const integrator = createIntegrator({
      system,
      gravity: [0, 0, 0],
      fixedStep: 0.01,
      restitution: 0.5,
      friction: 0.25,
      margin: 0.1,
      bounds: { min: [0, 0, 0], max: [1, 1, 1] },
    });
    await submit(device, (e) => integrator.step(e, 1));
    // Particle 0 passes x = 0.9 (the wall less the margin) and bounces: speed halved, then friction takes a quarter.
    let velocities = await system.read('velocity');
    expect(velocities[0]).toBeCloseTo(-3 * 0.5 * 0.75, 4);
    expect(velocities[2]).toBeCloseTo(1 * 0.75, 4);
    // Run on: nothing ever leaves the shrunk box.
    await submit(device, (e) => integrator.step(e, 600));
    const positions = await system.read('position');
    velocities = await system.read('velocity');
    for (let i = 0; i < 2; i++) {
      for (let axis = 0; axis < 3; axis++) {
        expect(positions[i * 4 + axis]).toBeGreaterThanOrEqual(0.1 - 1e-6);
        expect(positions[i * 4 + axis]).toBeLessThanOrEqual(0.9 + 1e-6);
      }
    }
    expect(Number.isFinite(velocities[0]!)).toBe(true);
    integrator.destroy();
    system.destroy();
  });

  it('runs whole fixed steps from the clock, capped per frame, in one dispatch', async () => {
    const { device, system } = await rig();
    system.fill(1, () => {});
    const integrator = createIntegrator({
      system,
      gravity: [0, 0, 0],
      fixedStep: 0.01,
      maxSteps: 3,
    });
    const encoder = device.createCommandEncoder();
    expect(integrator.advance(encoder, 0.004)).toBe(0);
    expect(integrator.advance(encoder, 0.004)).toBe(0); // 0.008 banked
    expect(integrator.advance(encoder, 0.004)).toBe(1); // 0.012 -> one step
    expect(integrator.advance(encoder, 0.1)).toBe(3); // capped, backlog dropped
    expect(integrator.advance(encoder, 0)).toBe(0);
    encoder.finish();
    integrator.destroy();
    system.destroy();
  });

  it('evaluates a WGSL force field, with host-set parameters', async () => {
    const { device, system } = await rig();
    system.fill(2, (i, p) => {
      p.position[0] = i;
    });
    const integrator = createIntegrator({
      system,
      gravity: [0, 0, 0],
      fixedStep: 0.01,
      field: {
        wgsl: `fn force_field(p: vec3f, v: vec3f, time: f32, index: u32) -> vec3f {
          return vec3f(params.user0.x * (1.0 + f32(index)), params.user1.y, 0.0);
        }`,
      },
    });
    integrator.params.user[0] = [2, 0, 0, 0];
    integrator.params.user[1] = [0, -1, 0, 0];
    await expectValid(device, () => submit(device, (e) => integrator.step(e, 100)));
    const velocities = await system.read('velocity');
    // a = (2 (1 + index), -1) for t = 1 s.
    expect(velocities[0]).toBeCloseTo(2, 3);
    expect(velocities[4]).toBeCloseTo(4, 3);
    expect(velocities[1]).toBeCloseTo(-1, 3);
    integrator.destroy();
    system.destroy();
  });

  it('samples an acceleration texture', async () => {
    const { device, system } = await rig();
    system.fill(1, (_i, p) => {
      p.position[0] = 0.5;
      p.position[1] = 0.5;
    });
    const texture = device.createTexture({
      size: [2, 2],
      format: 'rgba16float',
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
    });
    // Every texel (0, 2, 0, 0) as half floats (2.0 is 0x4000).
    const texel = new Uint16Array([0, 0x4000, 0, 0]);
    device.queue.writeTexture(
      { texture },
      new Uint16Array([...texel, ...texel, ...texel, ...texel]),
      { bytesPerRow: 16 },
      [2, 2],
    );
    const integrator = createIntegrator({
      system,
      gravity: [0, 0, 0],
      fixedStep: 0.01,
      fieldMin: [0, 0, 0],
      fieldMax: [1, 1, 1],
      fieldStrength: 0.5,
      field: { texture: texture.createView(), dimension: '2d' },
    });
    await expectValid(device, () => submit(device, (e) => integrator.step(e, 100)));
    const velocities = await system.read('velocity');
    expect(velocities[1]).toBeCloseTo(1, 2); // 2.0 * 0.5 * 1 s
    expect(velocities[0]).toBeCloseTo(0, 5);
    integrator.destroy();
    texture.destroy();
    system.destroy();
  });

  it('respawns particles whose life ran out, from the emitter, on the GPU', async () => {
    const { device, system } = await rig();
    system.setCount(40); // zero-filled: no life left, so the first step brings every one in
    const integrator = createIntegrator({
      system,
      gravity: [0, 0, 0],
      fixedStep: 0.01,
      respawn: {
        shape: 'box',
        center: [5, 6, 7],
        extent: [1, 0.5, 0.25],
        direction: [0, 1, 0],
        speed: 2,
        lifetime: [0.2, 0.3],
      },
    });
    await expectValid(device, () => submit(device, (e) => integrator.step(e, 1)));
    let positions = await system.read('position');
    let velocities = await system.read('velocity');
    const seen = new Set<number>();
    for (let i = 0; i < 40; i++) {
      expect(Math.abs(positions[i * 4]! - 5)).toBeLessThanOrEqual(1.0001);
      expect(Math.abs(positions[i * 4 + 1]! - 6.02)).toBeLessThanOrEqual(0.5001);
      expect(Math.abs(positions[i * 4 + 2]! - 7)).toBeLessThanOrEqual(0.2501);
      expect(velocities[i * 4 + 1]).toBeCloseTo(2, 4);
      expect(velocities[i * 4 + 3]).toBeGreaterThan(0.18);
      expect(velocities[i * 4 + 3]).toBeLessThanOrEqual(0.3);
      seen.add(positions[i * 4]!);
    }
    expect(seen.size).toBeGreaterThan(30); // distinct random positions
    // 40 more steps (0.4 s) outlive every life, so each has respawned at least once and moved little.
    await submit(device, (e) => integrator.step(e, 40));
    positions = await system.read('position');
    velocities = await system.read('velocity');
    for (let i = 0; i < 40; i++) {
      expect(velocities[i * 4 + 3]).toBeGreaterThan(0);
      expect(positions[i * 4 + 1]).toBeLessThan(6 + 0.5 + 2 * 0.31 + 0.1);
    }
    integrator.destroy();
    system.destroy();
  });

  it('kills particles that touch the floor when asked to', async () => {
    const { device, system } = await rig();
    system.fill(10, (i, p) => {
      p.position[0] = 5;
      p.position[1] = 0.3;
      p.velocity[1] = -5;
      p.velocityW = 100; // plenty of life
      p.scalar = i;
    });
    const integrator = createIntegrator({
      system,
      gravity: [0, 0, 0],
      fixedStep: 0.01,
      bounds: { min: [0, 0, 0], max: [10, 10, 10] },
      margin: 0,
      respawn: { center: [5, 9, 5], speed: 0, lifetime: [50, 50], floor: 'kill' },
    });
    await submit(device, (e) => integrator.step(e, 10)); // reaches the floor after 0.06 s and is re-seeded at y = 9
    const positions = await system.read('position');
    for (let i = 0; i < 10; i++) expect(positions[i * 4 + 1]).toBeGreaterThan(8);
    integrator.destroy();
    system.destroy();
  });

  it('works in 2D: z stays zero whatever the field or emitter does', async () => {
    const { device, system } = await rig({ dimension: 2 });
    system.setCount(20);
    const integrator = createIntegrator({
      system,
      gravity: [0, -1, 5],
      fixedStep: 0.01,
      respawn: { shape: 'sphere', extent: [1, 1, 1], speed: 1, spread: 1, lifetime: [1, 2] },
    });
    await submit(device, (e) => integrator.step(e, 30));
    const positions = await system.read('position');
    const velocities = await system.read('velocity');
    for (let i = 0; i < 20; i++) {
      expect(positions[i * 4 + 2]).toBe(0);
      expect(velocities[i * 4 + 2]).toBe(0);
    }
    integrator.destroy();
    system.destroy();
  });
});

// --- The shared sprite and emitter helpers ---------------------------------------------------------------

describe('particles.wgsl', () => {
  const header = '#include "particles.wgsl"\n';

  it('documents and compiles, and its helpers behave', async () => {
    expect(getChunk('particles.wgsl')).toBeDefined();
    const [
      cornerX,
      cornerY,
      bigRadius,
      bigFade,
      tinyRadius,
      tinyFade,
      hardInside,
      hardOutside,
      softCentre,
      softMid,
      sphereTop,
      sphereRim,
      paletteMid,
      paletteEnd,
    ] = await scalars(header, [
      'particle_corner(3u).x',
      'particle_corner(2u).y',
      'particle_size(1.0, 10.0, 2.0, 100.0, 0.5).x',
      'particle_size(1.0, 10.0, 2.0, 100.0, 0.5).y',
      // radius 0.01 at w 10 with scale 2 and 100 px: 0.1 px, floored to 0.5 px.
      'particle_size(0.01, 10.0, 2.0, 100.0, 0.5).x',
      'particle_size(0.01, 10.0, 2.0, 100.0, 0.5).y',
      'particle_disc_alpha(0.5, 0.0, 0.01)',
      'particle_disc_alpha(1.1, 0.0, 0.01)',
      'particle_disc_alpha(0.0, 1.0, 0.01)',
      'particle_disc_alpha(0.5, 1.0, 0.01)',
      'particle_sphere_z(0.0)',
      'particle_sphere_z(1.0)',
      'particle_palette(array<vec4f, 8>(vec4f(0.0), vec4f(1.0, 0.0, 0.0, 1.0), vec4f(0.0), vec4f(0.0), vec4f(0.0), vec4f(0.0), vec4f(0.0), vec4f(0.0)), 2u, 0.5).x',
      'particle_palette(array<vec4f, 8>(vec4f(0.0), vec4f(1.0, 0.0, 0.0, 1.0), vec4f(0.0), vec4f(0.0), vec4f(0.0), vec4f(0.0), vec4f(0.0), vec4f(0.0)), 2u, 1.0).x',
    ]);
    expect(cornerX).toBe(1);
    expect(cornerY).toBe(1);
    expect(bigRadius).toBeCloseTo(1, 6);
    expect(bigFade).toBe(1);
    // 0.01 world units at 0.1 px grow to 0.5 px: 5x the radius, 1/25 the energy per sprite.
    expect(tinyRadius).toBeCloseTo(0.05, 5);
    expect(tinyFade).toBeCloseTo(1 / 25, 5);
    expect(hardInside).toBe(1);
    expect(hardOutside).toBe(0);
    expect(softCentre).toBe(1);
    expect(softMid).toBeCloseTo(0.5, 1);
    expect(sphereTop).toBe(1);
    expect(sphereRim).toBe(0);
    expect(paletteMid).toBeCloseTo(0.5, 6);
    expect(paletteEnd).toBeCloseTo(1, 6);
  });
});

// --- The billboard renderer --------------------------------------------------------------------------------

describe('billboard renderer', () => {
  const SIZE = 64;
  const FORMAT: GPUTextureFormat = 'rgba8unorm';
  const RED = [1, 0, 0] as const;
  const GREEN = [0, 1, 0] as const;

  const plain: Partial<BillboardStyle> = {
    softness: 0,
    shading: 0,
    intensity: 1,
    scalarRange: [0, 1],
    palette: [RED, GREEN],
  };

  async function drawScene(
    renderer: BillboardRenderer,
    rows: number[],
    camera: ReturnType<typeof computeCamera>,
    options: { radius: number; depth?: boolean; scalar?: Float32Array } = { radius: 0.1 },
  ) {
    const device = await getTestDevice();
    const system = createParticleSystem({
      device,
      capacity: Math.max(1, rows.length / 4),
      radius: options.radius,
    });
    system.set({ positions: Float32Array.from(rows) });
    const target = device.createTexture({
      size: [SIZE, SIZE],
      format: FORMAT,
      usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC,
    });
    let scalar: GPUBuffer | undefined;
    if (options.scalar) {
      scalar = device.createBuffer({
        size: options.scalar.byteLength,
        usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
      });
      device.queue.writeBuffer(scalar, 0, options.scalar);
    }
    renderer.resize(SIZE, SIZE);
    await expectValid(device, () =>
      submit(device, (e) =>
        renderer.draw(e, system.particles, camera, target.createView(), { scalar }),
      ),
    );
    const pixels = await readTexture(device, target);
    target.destroy();
    scalar?.destroy();
    system.destroy();
    return pixels;
  }

  /** Alpha-weighted centroid of the lit pixels, in pixel units with the pixel centre at +0.5. */
  function centroid(pixels: Uint8Array, channel = 3): { x: number; y: number; weight: number } {
    let sx = 0;
    let sy = 0;
    let weight = 0;
    for (let y = 0; y < SIZE; y++) {
      for (let x = 0; x < SIZE; x++) {
        const w = pixels[(y * SIZE + x) * 4 + channel]! / 255;
        sx += w * (x + 0.5);
        sy += w * (y + 0.5);
        weight += w;
      }
    }
    return { x: sx / weight, y: sy / weight, weight };
  }

  const pixel = (pixels: Uint8Array, x: number, y: number) =>
    Array.from(pixels.subarray((y * SIZE + x) * 4, (y * SIZE + x) * 4 + 4));

  it('draws a particle at the pixel a perspective camera predicts, with the predicted size', async () => {
    const device = await getTestDevice();
    const renderer = createBillboardRenderer({
      device,
      format: FORMAT,
      style: { ...plain, mode: 'additive' },
    });
    const camera = computeCamera({
      position: [1, 2, 6],
      target: [0, 0.5, 0],
      fovY: 0.9,
      aspect: 1,
      near: 0.1,
      far: 50,
    });
    const points: [number, number, number][] = [
      [0.3, -0.2, 0.5],
      [-0.6, 0.9, -1],
      [0, 0.5, 0],
    ];
    for (const point of points) {
      const pixels = await drawScene(renderer, [...point, 0], camera, { radius: 0.2 });
      const projected = projectPoint(camera.viewProj, point);
      const c = centroid(pixels);
      expect(c.x).toBeCloseTo(projected.u * SIZE, 0);
      expect(c.y).toBeCloseTo(projected.v * SIZE, 0);
      expect(Math.abs(c.x - projected.u * SIZE)).toBeLessThan(0.35);
      expect(Math.abs(c.y - projected.v * SIZE)).toBeLessThan(0.35);
      // Area of the disc: pi r^2 pixels, with r the world radius times the projection scale over w.
      const radiusPx = ((0.2 * camera.proj[5]!) / projected.w) * (SIZE / 2);
      expect(c.weight).toBeGreaterThan(Math.PI * radiusPx * radiusPx * 0.9);
      expect(c.weight).toBeLessThan(Math.PI * radiusPx * radiusPx * 1.1);
    }
    renderer.destroy();
  });

  it('draws a particle where an orthographic camera predicts, at a size that ignores distance', async () => {
    const device = await getTestDevice();
    const renderer = createBillboardRenderer({
      device,
      format: FORMAT,
      style: { ...plain, mode: 'additive' },
    });
    const camera = orthographicCamera({ center: [1, 2], halfHeight: 2, aspect: 1 });
    const sizes: number[] = [];
    const points: [number, number, number][] = [
      [1.5, 2.5, 0],
      [0.2, 1, 3],
      [2.3, 3.1, -4],
    ];
    for (const point of points) {
      const pixels = await drawScene(renderer, [...point, 0], camera, { radius: 0.3 });
      const projected = projectPoint(camera.viewProj, point);
      const c = centroid(pixels);
      expect(Math.abs(c.x - projected.u * SIZE)).toBeLessThan(0.35);
      expect(Math.abs(c.y - projected.v * SIZE)).toBeLessThan(0.35);
      sizes.push(c.weight);
      // World +x is right and +y is up on screen.
      expect(projected.u).toBeCloseTo((point[0]! - 1) / 4 + 0.5, 5);
      expect(projected.v).toBeCloseTo(0.5 - (point[1]! - 2) / 4, 5);
    }
    // radius 0.3 on a 4-high view of 64 px: 4.8 px, area 72 px, wherever it is.
    for (const area of sizes) expect(area).toBeGreaterThan(Math.PI * 4.8 * 4.8 * 0.9);
    expect(Math.max(...sizes) - Math.min(...sizes)).toBeLessThan(2);
    renderer.destroy();
  });

  it('puts the nearer sphere in front whatever the buffer order, in the opaque mode', async () => {
    const device = await getTestDevice();
    const renderer = createBillboardRenderer({
      device,
      format: FORMAT,
      style: { ...plain, mode: 'opaque' },
    });
    const camera = computeCamera({
      position: [0, 0, 5],
      target: [0, 0, 0],
      fovY: 0.9,
      aspect: 1,
      near: 0.1,
      far: 50,
    });
    // Near particle red (scalar 0) at z = 1, far one green (scalar 1) at z = -1, same ray.
    const near = [0, 0, 1, 0];
    const far = [0, 0, -1, 1];
    for (const order of [
      [near, far],
      [far, near],
    ]) {
      const pixels = await drawScene(renderer, order.flat(), camera, { radius: 0.3 });
      const centre = pixel(pixels, SIZE / 2, SIZE / 2);
      expect(centre[0]).toBe(255);
      expect(centre[1]).toBe(0);
      expect(centre[3]).toBe(255);
      // The far (smaller, farther) sphere is hidden entirely: no green anywhere.
      let green = 0;
      for (let i = 0; i < SIZE * SIZE; i++) if (pixels[i * 4 + 1]! > 0) green++;
      expect(green).toBe(0);
    }
    // A far particle beside the near one is visible: depth testing is not just "draw nothing behind".
    const pixels = await drawScene(renderer, [...near, 0.5, 0, -1, 1], camera, { radius: 0.2 });
    let green = 0;
    for (let i = 0; i < SIZE * SIZE; i++) if (pixels[i * 4 + 1]! > 200) green++;
    expect(green).toBeGreaterThan(8);
    renderer.destroy();
  });

  it('makes spheres intersect correctly: the surface depth, not the centre depth, decides', async () => {
    const device = await getTestDevice();
    const renderer = createBillboardRenderer({
      device,
      format: FORMAT,
      style: { ...plain, mode: 'opaque' },
    });
    const camera = computeCamera({
      position: [0, 0, 5],
      target: [0, 0, 0],
      fovY: 0.9,
      aspect: 1,
      near: 0.1,
      far: 50,
    });
    // Two overlapping spheres side by side and 0.3 apart in depth. A flat sprite would show whichever was drawn
    // last in the overlap; sphere impostors cross along a line, so both colours appear in the overlap region.
    const rows = [-0.15, 0, 0.15, 0, 0.15, 0, -0.15, 1];
    const pixels = await drawScene(renderer, rows, camera, { radius: 0.4 });
    const row = SIZE / 2;
    const reds: number[] = [];
    const greens: number[] = [];
    for (let x = 0; x < SIZE; x++) {
      const [r, g] = pixel(pixels, x, row);
      if (r! > 200) reds.push(x);
      if (g! > 200) greens.push(x);
    }
    // Red (near, left) wins on the left, green (far, right) on the right, and the two meet inside the overlap.
    expect(Math.min(...reds)).toBeLessThan(Math.min(...greens));
    expect(Math.max(...greens)).toBeGreaterThan(Math.max(...reds));
    expect(Math.min(...greens)).toBeLessThan(Math.max(...reds) + 3);
    renderer.destroy();
  });

  it('adds up overlapping particles in the additive mode, and keeps the total with a pixel floor', async () => {
    const device = await getTestDevice();
    const renderer = createBillboardRenderer({
      device,
      format: FORMAT,
      style: { ...plain, mode: 'additive', intensity: 0.3, palette: [GREEN, GREEN] },
    });
    const camera = orthographicCamera({ center: [0, 0], halfHeight: 1, aspect: 1 });
    const one = await drawScene(renderer, [0, 0, 0, 0], camera, { radius: 0.2 });
    const two = await drawScene(renderer, [0, 0, 0, 0, 0, 0, 0, 0], camera, { radius: 0.2 });
    const centre = SIZE / 2;
    expect(pixel(one, centre, centre)[1]).toBeCloseTo(0.3 * 255, -1);
    expect(pixel(two, centre, centre)[1]).toBeCloseTo(0.6 * 255, -1);

    // A particle far smaller than a pixel is still drawn, at a floor of the configured size, and fainter in
    // proportion to the extra area so a cloud of them does not get brighter than the particles it stands for.
    renderer.style.intensity = 1;
    renderer.style.minPixels = 1;
    // Radius 0.005 on a 64 px, 2-high view is 0.16 px: true area 0.08 px^2, drawn one pixel wide at a fraction of the strength.
    const tiny = await drawScene(renderer, [0.25, 0.25, 0, 0], camera, { radius: 0.005 });
    const c = centroid(tiny, 1);
    expect(c.weight).toBeGreaterThan(0.04);
    expect(c.weight).toBeLessThan(0.2);
    expect(Math.abs(c.x - 0.625 * SIZE)).toBeLessThan(0.6);
    renderer.destroy();
  });

  it('colours through the palette by w, or by a separate scalar buffer', async () => {
    const device = await getTestDevice();
    const renderer = createBillboardRenderer({
      device,
      format: FORMAT,
      style: { ...plain, mode: 'alpha' },
    });
    const camera = orthographicCamera({ center: [0, 0], halfHeight: 1, aspect: 1 });
    const rows = [-0.5, 0, 0, 0, 0.5, 0, 0, 1];
    const byW = await drawScene(renderer, rows, camera, { radius: 0.2 });
    expect(pixel(byW, 16, 32)).toEqual([255, 0, 0, 255]);
    expect(pixel(byW, 48, 32)).toEqual([0, 255, 0, 255]);
    // The buffer wins: swap the colours.
    const bySeparate = await drawScene(renderer, rows, camera, {
      radius: 0.2,
      scalar: new Float32Array([1, 0]),
    });
    expect(pixel(bySeparate, 16, 32)).toEqual([0, 255, 0, 255]);
    expect(pixel(bySeparate, 48, 32)).toEqual([255, 0, 0, 255]);
    // Values outside the range clamp to the end colours.
    const clamped = await drawScene(renderer, rows, camera, {
      radius: 0.2,
      scalar: new Float32Array([-5, 9]),
    });
    expect(pixel(clamped, 16, 32)).toEqual([255, 0, 0, 255]);
    expect(pixel(clamped, 48, 32)).toEqual([0, 255, 0, 255]);
    renderer.destroy();
  });

  it('draws nothing, without error, for an empty set or a particle behind the camera', async () => {
    const device = await getTestDevice();
    const renderer = createBillboardRenderer({ device, format: FORMAT, style: plain });
    const camera = computeCamera({
      position: [0, 0, 5],
      target: [0, 0, 0],
      fovY: 0.9,
      aspect: 1,
      near: 0.1,
      far: 50,
    });
    const behind = await drawScene(renderer, [0, 0, 20, 0], camera, { radius: 1 });
    expect(Math.max(...behind)).toBe(0);
    const nan = await drawScene(renderer, [Number.NaN, 0, 0, 0], camera, { radius: 1 });
    expect(Math.max(...nan)).toBe(0);
    renderer.destroy();
  });

  it('shares a caller-supplied depth buffer and can keep what is already drawn', async () => {
    const device = await getTestDevice();
    const renderer = createBillboardRenderer({
      device,
      format: FORMAT,
      style: { ...plain, mode: 'opaque' },
    });
    const camera = computeCamera({
      position: [0, 0, 5],
      target: [0, 0, 0],
      fovY: 0.9,
      aspect: 1,
      near: 0.1,
      far: 50,
    });
    const system = createParticleSystem({ device, capacity: 2, radius: 0.3 });
    system.set({ positions: Float32Array.from([0, 0, 1, 0]) });
    const depth = createDepthTarget(device, {
      label: 'test depth',
      width: SIZE,
      height: SIZE,
      format: 'depth24plus',
      sampled: false,
    });
    const target = device.createTexture({
      size: [SIZE, SIZE],
      format: FORMAT,
      usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC,
    });
    // First the near particle (red), then, keeping colour and depth, a far one (green) that must stay hidden.
    await expectValid(device, async () => {
      await submit(device, (e) =>
        renderer.draw(e, system.particles, camera, target.createView(), {
          depth: depth.view,
          width: SIZE,
          height: SIZE,
        }),
      );
      system.set({ positions: Float32Array.from([0, 0, -1, 1]) });
      await submit(device, (e) =>
        renderer.draw(e, system.particles, camera, target.createView(), {
          depth: depth.view,
          clear: null,
          clearDepth: false,
          width: SIZE,
          height: SIZE,
        }),
      );
    });
    const pixels = await readTexture(device, target);
    expect(pixel(pixels, SIZE / 2, SIZE / 2)).toEqual([255, 0, 0, 255]);
    depth.destroy();
    target.destroy();
    system.destroy();
    renderer.destroy();
  });

  it('shades spheres in the opaque mode: lit on the light side, darker on the far side', async () => {
    const device = await getTestDevice();
    const renderer = createBillboardRenderer({
      device,
      format: FORMAT,
      style: {
        ...plain,
        mode: 'opaque',
        shading: 1,
        palette: [
          [1, 1, 1],
          [1, 1, 1],
        ],
      },
    });
    const camera = orthographicCamera({ center: [0, 0], halfHeight: 1, aspect: 1 });
    const pixels = await drawScene(renderer, [0, 0, 0, 0], camera, { radius: 0.6 });
    // The light comes from the upper left: a pixel up-left of the centre is brighter than one down-right.
    const lit = pixel(pixels, 24, 24)[0]!;
    const dark = pixel(pixels, 40, 40)[0]!;
    expect(lit).toBeGreaterThan(dark + 40);
    renderer.destroy();
  });
});

// --- The demo effect, every view ------------------------------------------------------------------------------

describe('particles effect', () => {
  const WIDTH = 96;
  const HEIGHT = 64;

  const canvas = () =>
    Object.assign(new EventTarget(), {
      width: WIDTH,
      height: HEIGHT,
      style: {},
      getBoundingClientRect: () => ({ left: 0, top: 0, width: WIDTH, height: HEIGHT }),
      setPointerCapture() {},
      releasePointerCapture() {},
    }) as unknown as HTMLCanvasElement;

  const cases: Record<string, string>[] = [
    { scene: 'fountain 3D', view: 'speed', render: 'additive' },
    { scene: 'fountain 3D', view: 'speed', render: 'additive', palette: 'orange', background: '0' },
    {
      scene: 'fountain 3D',
      view: 'speed',
      render: 'additive',
      presentation: 'lantern',
      palette: 'orange',
      background: '0',
    },
    { scene: 'fountain 3D', view: 'neighbour count', render: 'alpha', reorder: '0' },
    { scene: 'fountain 3D', view: 'neighbours of picked', render: 'opaque' },
    { scene: 'fountain 3D', view: 'neighbour count', render: 'additive', alwaysBuild: '1' },
    { scene: 'stream 2D', view: 'speed', render: 'additive' },
    { scene: 'stream 2D', view: 'neighbour count', render: 'opaque' },
    { scene: 'stream 2D', view: 'neighbours of picked', render: 'additive', reorder: '0' },
  ];

  it.each(cases)('draws and survives a count change: %o', async (override) => {
    const { loadEffect } = await import('../src/effects/registry.ts');
    const { defaultParams } = await import('../src/effects/types.ts');
    const { noProfiler } = await import('../src/gpu/profiler.ts');
    const device = await getTestDevice();
    const effect = await loadEffect('particles');
    const values = defaultParams(effect.params) as Record<string, unknown>;
    values.count = 30000;
    for (const [key, value] of Object.entries(override)) {
      values[key] = ['reorder', 'alwaysBuild', 'background'].includes(key) ? value === '1' : value;
    }
    const target = device.createTexture({
      size: [WIDTH, HEIGHT],
      format: 'rgba8unorm',
      usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC,
    });
    await expectValid(device, async () => {
      const instance = await effect.create({
        device,
        format: 'rgba8unorm',
        canvas: canvas(),
        params: values as never,
        profiler: noProfiler,
        hdr: false,
      });
      instance.resize(WIDTH, HEIGHT, 1);
      for (let frame = 0; frame < 6; frame++) {
        if (frame === 2) values.count = 50000; // grows: a capacity change
        if (frame === 4) values.count = 20000; // shrinks
        const encoder = device.createCommandEncoder();
        instance.frame({ time: frame / 60, delta: 1 / 60, frame }, encoder, target.createView());
        device.queue.submit([encoder.finish()]);
      }
      instance.destroy();
    });
    const pixels = await readTexture(device, target);
    expect(new Set(pixels).size).toBeGreaterThan(8);
    if (values.background === false && values.presentation !== 'lantern') {
      // Empty pixels stay transparent, and the live fountain uses orange instead of ocean blue.
      expect(Array.from(pixels.subarray(0, 4))).toEqual([0, 0, 0, 0]);
      let orangePixels = 0;
      for (let i = 0; i < pixels.length; i += 4) {
        if (pixels[i]! > pixels[i + 2]! + 20 && pixels[i + 1]! > 10) orangePixels++;
      }
      expect(orangePixels).toBeGreaterThan(10);
    }
    target.destroy();
  });
});

// --- Cost study of the demo -------------------------------------------------------------------------------------------
//
// What limits the particle count of the `particles` effect (src/effects/particles/notes.md). Skipped unless PARTICLE_STUDY
// names a file to write to:
//
//     PARTICLE_STUDY=/tmp/particles-effect-study.txt npx vitest run tests/particles.test.ts -t study
//
// Same method as the neighbour grid study in tests/neighbors.test.ts: the cost of one frame is the slope of the GPU span
// between a small and a large number of frames recorded into one command buffer (quietest of many alternating runs; the
// GPU is shared and a busy moment only adds time). The per-pass figures are the quietest single pass durations over many
// single frames, and they overlap on this GPU, so they bound each stage and do not add up to the span.

const studyFile = (globalThis as { process?: { env: Record<string, string | undefined> } }).process
  ?.env.PARTICLE_STUDY;

describe.skipIf(!studyFile)('study', () => {
  vi.setConfig({ testTimeout: 3_600_000 });
  const lines: string[] = [];
  const log = (text = '') => lines.push(text);

  async function finish() {
    const specifier = 'node:fs';
    const { writeFileSync } = (await import(/* @vite-ignore */ specifier)) as {
      writeFileSync(path: string, text: string): void;
    };
    writeFileSync(studyFile!, lines.join('\n') + '\n');
  }

  async function timedDevice() {
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter?.features.has('timestamp-query'))
      throw new Error('timestamp-query is not available');
    return adapter.requestDevice({ requiredFeatures: ['timestamp-query'] });
  }

  interface Result {
    /** ms per frame: quietest large run minus quietest small run, over the difference in frames. */
    frame: number;
    /** The same from the median runs. */
    median: number;
    /** Quietest single-pass duration per label over many single frames, in ms. */
    labels: Record<string, number>;
  }

  async function measure(
    device: GPUDevice,
    overrides: Record<string, unknown>,
    width: number,
    height: number,
  ): Promise<Result> {
    const { loadEffect } = await import('../src/effects/registry.ts');
    const { defaultParams } = await import('../src/effects/types.ts');
    const effect = await loadEffect('particles');
    const values = { ...defaultParams(effect.params), ...overrides } as Record<string, unknown>;
    const querySet = device.createQuerySet({ type: 'timestamp', count: 256 });
    const resolve = device.createBuffer({
      size: 256 * 8,
      usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC,
    });
    const read = device.createBuffer({
      size: 256 * 8,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    // Mode 'span': stamps for the first and last pass of the recording. Mode 'all': every pass, by label.
    let mode: 'off' | 'span' | 'all' = 'off';
    let expected = 0;
    let seen = 0;
    let labels: string[] = [];
    const profiler = {
      pass(label: string) {
        const index = seen++;
        if (mode === 'all') {
          labels.push(label);
          if (index >= 127) return undefined;
          return {
            querySet,
            beginningOfPassWriteIndex: index * 2,
            endOfPassWriteIndex: index * 2 + 1,
          };
        }
        if (mode === 'span') {
          if (expected === 1)
            return { querySet, beginningOfPassWriteIndex: 0, endOfPassWriteIndex: 1 };
          if (index === 0)
            return { querySet, beginningOfPassWriteIndex: 0, endOfPassWriteIndex: 2 };
          if (index === expected - 1)
            return { querySet, beginningOfPassWriteIndex: 3, endOfPassWriteIndex: 1 };
        }
        return undefined;
      },
    };
    const canvas = Object.assign(new EventTarget(), {
      width,
      height,
      style: {},
      getBoundingClientRect: () => ({ left: 0, top: 0, width, height }),
      setPointerCapture() {},
      releasePointerCapture() {},
    }) as unknown as HTMLCanvasElement;
    const target = device.createTexture({
      size: [width, height],
      format: 'rgba8unorm',
      usage: GPUTextureUsage.RENDER_ATTACHMENT,
    });
    const instance = await effect.create({
      device,
      format: 'rgba8unorm',
      canvas,
      params: values as never,
      profiler,
      hdr: false,
    });
    instance.resize(width, height, 1);
    let frame = 0;
    const record = (encoder: GPUCommandEncoder, frames: number) => {
      for (let i = 0; i < frames; i++, frame++) {
        instance.frame({ time: frame / 60, delta: 1 / 60, frame }, encoder, target.createView());
      }
    };
    const run = async (frames: number, slots: number) => {
      const encoder = device.createCommandEncoder();
      record(encoder, frames);
      encoder.resolveQuerySet(querySet, 0, slots, resolve, 0);
      encoder.copyBufferToBuffer(resolve, 0, read, 0, slots * 8);
      device.queue.submit([encoder.finish()]);
      await read.mapAsync(GPUMapMode.READ);
      const stamps = new BigUint64Array(read.getMappedRange().slice(0, slots * 8));
      read.unmap();
      return stamps;
    };

    // Warm up (pipelines, the first sort) and count the passes of one frame.
    for (let i = 0; i < 5; i++) {
      mode = 'off';
      await run(1, 2);
    }
    mode = 'all';
    seen = 0;
    labels = [];
    await run(1, 2);
    const perFrame = seen;

    mode = 'span';
    const time = async (frames: number) => {
      seen = 0;
      expected = perFrame * frames;
      const stamps = await run(frames, 4);
      return Number(stamps[1]! - stamps[0]!) / 1e6;
    };
    const small: number[] = [];
    const large: number[] = [];
    const lo = 1;
    const hi = 4;
    for (let round = 0; round < 24; round++) {
      small.push(await time(lo));
      large.push(await time(hi));
    }
    small.sort((a, b) => a - b);
    large.sort((a, b) => a - b);
    const mid = Math.floor(small.length / 2);

    mode = 'all';
    const best: Record<string, number> = {};
    for (let round = 0; round < 24; round++) {
      seen = 0;
      labels = [];
      const stamps = await run(1, perFrame * 2);
      const sums = new Map<string, number>();
      labels.forEach((label, i) => {
        sums.set(label, (sums.get(label) ?? 0) + Number(stamps[i * 2 + 1]! - stamps[i * 2]!) / 1e6);
      });
      for (const [label, ms] of sums) best[label] = Math.min(best[label] ?? Infinity, ms);
    }
    instance.destroy();
    target.destroy();
    querySet.destroy();
    return {
      frame: (large[0]! - small[0]!) / (hi - lo),
      median: (large[mid]! - small[mid]!) / (hi - lo),
      labels: best,
    };
  }

  const f = (x: number, digits = 2) => x.toFixed(digits);

  it('what limits the particle count', async () => {
    const device = await timedDevice();
    log(
      'PARTICLES EFFECT COST. Apple M1 Max, Metal through Dawn (Node), timestamp queries. ms per frame, quietest of 24 alternating runs (median-run figure in brackets).',
    );
    log(
      'Fountain 3D at the default settings unless stated, 1280x720, additive, colour by speed; the integrator runs its usual 2 sub-steps of 1/120 s per 1/60 s frame.',
    );
    log();

    log('A. Whole frame against particle count (speed view, no grid)');
    log(
      [
        'particles',
        'frame, 1280x720',
        'frame, 640x360',
        'integrate pass',
        'draw pass, 1280x720',
      ].join(' | '),
    );
    for (const count of [100_000, 250_000, 500_000, 1_000_000]) {
      const big = await measure(device, { count }, 1280, 720);
      const small = await measure(device, { count }, 640, 360);
      log(
        `${count} | ${f(big.frame)} (med ${f(big.median)}) | ${f(small.frame)} (med ${f(small.median)}) | ${f(big.labels.integrate ?? 0)} | ${f(big.labels['particles draw'] ?? 0)}`,
      );
      await finish();
    }
    log();

    log(
      'B. Fill rate: 1M particles, 1280x720, by particle size (the default radius 0.012 is about 2.4 px at this size) and blend mode (frame ms)',
    );
    log(['size scale', 'additive', 'alpha', 'opaque spheres'].join(' | '));
    for (const size of [0.25, 0.5, 1, 2, 4]) {
      const row: string[] = [];
      for (const render of ['additive', 'alpha', 'opaque']) {
        const r = await measure(device, { count: 1_000_000, size, render }, 1280, 720);
        row.push(`${f(r.frame)} (med ${f(r.median)})`);
      }
      log(`${size} | ${row.join(' | ')}`);
      await finish();
    }
    log();

    log(
      'C. With the neighbour grid and the count view (1280x720, reorder mode, search radius 0.07), frame ms and the passes that matter (quietest single pass)',
    );
    log(
      [
        'particles',
        'frame',
        'grid cells',
        'grid scan',
        'grid scatter',
        'reorder',
        'neighbour count pass',
        'integrate',
        'draw',
      ].join(' | '),
    );
    for (const count of [100_000, 250_000, 500_000, 1_000_000]) {
      const r = await measure(device, { count, view: 'neighbour count', radius: 0.07 }, 1280, 720);
      const l = r.labels;
      log(
        `${count} | ${f(r.frame)} (med ${f(r.median)}) | ${f(l['grid cells'] ?? 0, 3)} | ${f(l['grid scan'] ?? 0, 3)} | ${f(l['grid scatter'] ?? 0, 3)} | ${f(l['particles reorder'] ?? 0, 3)} | ${f(l['neighbours count'] ?? 0)} | ${f(l.integrate ?? 0)} | ${f(l['particles draw'] ?? 0)}`,
      );
      await finish();
    }
    log();

    log(
      'D. The grid every frame with nothing consuming it (alwaysBuild), 1M particles, 1280x720: sorted index vs reorder, and the 2D scene',
    );
    for (const [name, overrides] of [
      ['3D, no grid', { count: 1_000_000 }],
      ['3D, grid built, reorder', { count: 1_000_000, alwaysBuild: true, reorder: true }],
      ['3D, grid built, sorted index', { count: 1_000_000, alwaysBuild: true, reorder: false }],
      ['2D stream, no grid, 250k', { count: 250_000, scene: 'stream 2D' }],
      [
        '2D stream, grid built, reorder, 250k',
        { count: 250_000, scene: 'stream 2D', alwaysBuild: true },
      ],
    ] as const) {
      const r = await measure(device, overrides, 1280, 720);
      log(`${name} | ${f(r.frame)} (med ${f(r.median)})`);
    }
    log();
    await finish();
  });
});
