import { expect, it } from 'vitest';
import { effect } from '../src/effects/game-of-life/index.ts';
import { defaultParams } from '../src/effects/types.ts';
import { noProfiler } from '../src/gpu/profiler.ts';
import { halfToFloat } from '../src/gpu/sim/fluid.ts';
import { createLifeSimulation } from '../src/effects/game-of-life/simulation.ts';
import { GLIDER, makeSeed, type Cell } from '../src/effects/game-of-life/seeds.ts';
import { expectValid, getTestDevice, readTexture } from './gpu.ts';

it('Conway GPU generations preserve a block, oscillate a blinker, move a glider, and wrap the world', async () => {
  const device = await getTestDevice();
  await expectValid(device, async () => {
    const w = 32,
      h = 32,
      sim = createLifeSimulation(device, 'rgba8unorm', w, h);
    const seed = (cells: readonly Cell[]) => {
      const data = new Uint32Array(w * h);
      cells.forEach(([x, y]) => {
        data[y * w + x] = 257;
      });
      sim.seed(data);
    };
    const alive = async () => {
      const pixels = new Uint32Array((await readTexture(device, sim.cells)).buffer);
      const cells: string[] = [];
      pixels.forEach((value, index) => {
        if (value & 1) cells.push(`${index % w},${Math.floor(index / w)}`);
      });
      return cells.sort();
    };
    const step = (count = 1) => {
      const encoder = device.createCommandEncoder();
      sim.step(encoder, count);
      device.queue.submit([encoder.finish()]);
    };
    const names = (cells: readonly Cell[]) => cells.map(([x, y]) => `${x},${y}`).sort();
    try {
      const block: Cell[] = [
        [8, 8],
        [9, 8],
        [8, 9],
        [9, 9],
      ];
      seed(block);
      step(20);
      expect(await alive()).toEqual(names(block));
      seed([
        [7, 8],
        [8, 8],
        [9, 8],
      ]);
      step();
      expect(await alive()).toEqual(
        names([
          [8, 7],
          [8, 8],
          [8, 9],
        ]),
      );
      step();
      expect(await alive()).toEqual(
        names([
          [7, 8],
          [8, 8],
          [9, 8],
        ]),
      );
      seed(GLIDER.map(([x, y]) => [x + 8, y + 8]));
      step(4);
      expect(await alive()).toEqual(names(GLIDER.map(([x, y]) => [x + 9, y + 9])));
      seed([
        [31, 8],
        [0, 8],
        [1, 8],
      ]);
      step();
      expect(await alive()).toEqual(
        names([
          [0, 7],
          [0, 8],
          [0, 9],
        ]),
      );
      seed([[5, 5]]);
      step();
      expect(await alive()).toEqual([]);
      // Echo and age bits never count as live neighbours.
      const dead = new Uint32Array(w * h).fill((8 << 16) | (20 << 8));
      sim.seed(dead);
      step();
      expect(await alive()).toEqual([]);
      const target = device.createTexture({
        size: [128, 96],
        format: 'rgba8unorm',
        usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC,
      });
      try {
        const encoder = device.createCommandEncoder();
        sim.plant(encoder, 0.5, 0.5);
        sim.draw(encoder, target.createView());
        device.queue.submit([encoder.finish()]);
        expect(await alive()).toEqual(names(GLIDER.map(([x, y]) => [x + 16, y + 16])));
        // Exact primary phases stay saturated even where the neon halos overlap.
        for (const [time, channel] of [
          [0, 0],
          [2, 1],
          [4, 2],
        ]) {
          sim.setTime(time!);
          const draw = device.createCommandEncoder();
          sim.draw(draw, target.createView());
          device.queue.submit([draw.finish()]);
          const pixels = await readTexture(device, target);
          const peaks = [0, 0, 0];
          pixels.forEach((value, index) => {
            if (index % 4 < 3) peaks[index % 4] = Math.max(peaks[index % 4]!, value);
          });
          expect(peaks[channel!]!).toBeGreaterThan(200);
          peaks.forEach((peak, index) => {
            if (index !== channel) expect(peak).toBeLessThan(32);
          });
        }
      } finally {
        target.destroy();
      }
    } finally {
      sim.destroy();
    }
  });
}, 30000);

it('seed worlds are reproducible and remain inside their grid', () => {
  for (const pattern of ['garden', 'soup', 'oscillators']) {
    const a = makeSeed(96, 80, pattern, 123),
      b = makeSeed(96, 80, pattern, 123);
    expect(a).toEqual(b);
    expect(a.length).toBe(96 * 80);
    expect(a.some((v) => v & 1)).toBe(true);
  }
  expect(makeSeed(96, 80, 'soup', 123)).not.toEqual(makeSeed(96, 80, 'soup', 124));
});

it('random seed patches add life without clearing surviving cells', async () => {
  const device = await getTestDevice();
  await expectValid(device, async () => {
    const sim = createLifeSimulation(device, 'rgba8unorm', 64, 48);
    try {
      const original = new Uint32Array(64 * 48);
      original[10 * 64 + 10] = 257;
      original[30 * 64 + 40] = 1 | (42 << 8);
      sim.seed(original);
      const encoder = device.createCommandEncoder();
      sim.sprinkle(encoder, 104730);
      device.queue.submit([encoder.finish()]);
      const cells = new Uint32Array((await readTexture(device, sim.cells)).buffer);
      expect(cells[10 * 64 + 10]).toBe(original[10 * 64 + 10]);
      expect(cells[30 * 64 + 40]).toBe(original[30 * 64 + 40]);
      expect(cells.filter((value) => value & 1).length).toBeGreaterThan(30);
    } finally {
      sim.destroy();
    }
  });
});

it('automatic seeding waits for playback time and holds while paused', async () => {
  const device = await getTestDevice();
  await expectValid(device, async () => {
    const params = defaultParams(effect.params);
    Object.assign(params, {
      running: false,
      pattern: 'oscillators',
      reseedInterval: 1,
      cellSize: 4,
    });
    const target = device.createTexture({
      size: [128, 96],
      format: 'rgba8unorm',
      usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC,
    });
    const instance = await effect.create({
      device,
      format: 'rgba8unorm',
      canvas: {} as HTMLCanvasElement,
      params,
      profiler: noProfiler,
    });
    instance.resize(128, 96, 1);
    let frame = 0;
    const draw = () => {
      const encoder = device.createCommandEncoder();
      instance.frame(
        { time: frame * 0.1, delta: 0.1, frame: frame++ },
        encoder,
        target.createView(),
      );
      device.queue.submit([encoder.finish()]);
    };
    try {
      draw();
      const paused = await readTexture(device, target);
      for (let i = 0; i < 20; i++) draw();
      expect(await readTexture(device, target)).toEqual(paused);
      params.running = true;
      // At six seconds the RGB cycle returns to red, and these oscillators return to phase zero.
      for (let i = 0; i < 60; i++) draw();
      const renewed = await readTexture(device, target);
      const livePixels = (pixels: Uint8Array) =>
        pixels.filter((v, i) => i % 4 === 0 && v > 150).length;
      expect(livePixels(renewed)).toBeGreaterThan(livePixels(paused));
      const generation = params.generation;
      params.running = false;
      for (let i = 0; i < 20; i++) draw();
      expect(params.generation).toBe(generation);
      expect(await readTexture(device, target)).toEqual(renewed);
    } finally {
      instance.destroy();
      target.destroy();
    }
  });
}, 30000);

it('HDR neon exceeds SDR white while keeping the primary saturated', async () => {
  const device = await getTestDevice();
  await expectValid(device, async () => {
    const sim = createLifeSimulation(device, 'rgba16float', 32, 24);
    const target = device.createTexture({
      size: [128, 96],
      format: 'rgba16float',
      usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC,
    });
    try {
      const seed = new Uint32Array(32 * 24);
      seed[12 * 32 + 16] = 257;
      seed[12 * 32 + 17] = 257;
      seed[13 * 32 + 16] = 257;
      seed[13 * 32 + 17] = 257;
      sim.seed(seed);
      sim.setTime(0, 2);
      const encoder = device.createCommandEncoder();
      sim.draw(encoder, target.createView());
      device.queue.submit([encoder.finish()]);
      const pixels = new Uint16Array((await readTexture(device, target, 8)).buffer);
      const peaks = [0, 0, 0];
      pixels.forEach((value, index) => {
        if (index % 4 < 3) peaks[index % 4] = Math.max(peaks[index % 4]!, halfToFloat(value));
      });
      expect(peaks[0]).toBeGreaterThan(1.2);
      expect(peaks[1]).toBeLessThan(0.15);
      expect(peaks[2]).toBeLessThan(0.15);
    } finally {
      sim.destroy();
      target.destroy();
    }
  });
});
