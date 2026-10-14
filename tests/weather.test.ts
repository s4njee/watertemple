import { describe, expect, it } from 'vitest';
import { loadEffect } from '../src/effects/registry.ts';
import { defaultParams } from '../src/effects/types.ts';
import { noProfiler } from '../src/gpu/profiler.ts';
import {
  newRain,
  newBubble,
  splashBurst,
  advanceWeather,
  seededRandom,
  MAX_WEATHER_DROPS,
  type WeatherDrop,
} from '../src/gpu/weather/dynamics.ts';
import { createWeatherMemory } from '../src/gpu/weather/memory.ts';
import { WEATHER_EFFECT_IDS, weatherSourceFiles } from '../src/gpu/weather/sources.ts';
import { halfToFloat } from '../src/gpu/sim/fluid.ts';
import { expectValid, getTestDevice, readTexture } from './gpu.ts';
function canvas(width = 128, height = 80) {
  return Object.assign(new EventTarget(), {
    width,
    height,
    style: {},
    getBoundingClientRect: () => ({ left: 0, top: 0, width, height }),
    setPointerCapture() {},
    releasePointerCapture() {},
  }) as unknown as HTMLCanvasElement;
}
function pointer(node: HTMLCanvasElement, type: string, x: number, y: number) {
  const e = new Event(type);
  Object.defineProperties(e, {
    clientX: { value: x * node.width },
    clientY: { value: y * node.height },
    pointerId: { value: 1 },
    pointerType: { value: 'mouse' },
  });
  node.dispatchEvent(e);
}
const difference = (a: Uint8Array, b: Uint8Array) =>
  a.reduce((sum, v, i) => sum + Math.abs(v - b[i]!), 0) / a.length;
async function memoryPixels(device: GPUDevice, texture: GPUTexture) {
  return Float32Array.from(
    new Uint16Array((await readTexture(device, texture, 8)).buffer),
    halfToFloat,
  );
}
describe('weather mechanics', () => {
  it('rain has real depth-dependent size and speed, bends in wind and makes splash particles on impact', () => {
    const random = seededRandom(23),
      drops = Array.from({ length: 100 }, () => newRain(random, 1.5, 1, true));
    expect(new Set(drops.map((p) => p.depth.toFixed(1))).size).toBeGreaterThan(5);
    const near = drops.reduce((a, b) => (a.depth > b.depth ? a : b)),
      far = drops.reduce((a, b) => (a.depth < b.depth ? a : b));
    expect(near.radius).toBeGreaterThan(far.radius);
    expect(near.vy).toBeGreaterThan(far.vy);
    const p = drops[0]!;
    const before = p.x;
    advanceWeather(drops, 0.01, 1.5, 1, 0.7, random);
    expect(p.x).toBeGreaterThan(before);
    p.y = 0.95;
    expect(advanceWeather(drops, 0.01, 1.5, 1, 0.7, random)).toBeGreaterThan(0);
    expect(drops.some((p) => p.kind === 1 && p.vy < 0)).toBe(true);
    expect(drops.some((p) => p.kind === 3)).toBe(true);
  });
  it('ballistic splash motion and bubble wobble are independent of frame partition, and storage is bounded', () => {
    const random = seededRandom(7);
    const original: WeatherDrop[] = [];
    splashBurst(original, 0.5, 0.3, random, 24);
    const a = original.map((p) => ({ ...p })),
      b = original.map((p) => ({ ...p }));
    advanceWeather(a, 0.2, 1.6, 0, 0.55, seededRandom(1));
    for (let i = 0; i < 20; i++) advanceWeather(b, 0.01, 1.6, 0, 0.55, seededRandom(1));
    a.forEach((p, i) => {
      expect(p.x).toBeCloseTo(b[i]!.x, 8);
      expect(p.y).toBeCloseTo(b[i]!.y, 8);
      expect(p.vy).toBeCloseTo(b[i]!.vy, 8);
    });
    const bubble = newBubble(random, 1.6, true),
      c = [{ ...bubble }],
      d = [{ ...bubble }];
    advanceWeather(c, 0.3, 1.6, 0, 0.7, random);
    for (let i = 0; i < 30; i++) advanceWeather(d, 0.01, 1.6, 0, 0.7, random);
    expect(c[0]!.x).toBeCloseTo(d[0]!.x, 8);
    expect(c[0]!.y).toBeCloseTo(d[0]!.y, 8);
    expect(c[0]!.y).toBeLessThan(bubble.y);
    for (let i = 0; i < 100; i++) splashBurst(a, 0.5, 0.5, random, 128);
    expect(a.length).toBe(MAX_WEATHER_DROPS);
    advanceWeather(a, 10, 1.6, 0, 0.55, random);
    expect(a.length).toBe(0);
  });
  it('fog memory wipes a connected stroke, leaves other glass untouched and genuinely refogs', async () => {
    const device = await getTestDevice();
    await expectValid(device, async () => {
      const memory = createWeatherMemory(device, 128, 80);
      const update = (values: Parameters<typeof memory.update>[1]) => {
        const e = device.createCommandEncoder();
        memory.update(e, values);
        device.queue.submit([e.finish()]);
      };
      try {
        const base = {
          mode: 4,
          aspect: 1.6,
          density: 0.9,
          refog: 0.3,
          delta: 0.1,
          brush: 0.07,
          stroke_count: 0,
        };
        update(base);
        const start = await memoryPixels(device, memory.texture);
        expect(start[(40 * 128 + 64) * 4]).toBeCloseTo(0.9, 2);
        update({ ...base, stroke_count: 1, strokes: [0.2, 0.5, 0.8, 0.5] });
        const wiped = await memoryPixels(device, memory.texture);
        for (let x = 28; x < 99; x++) expect(wiped[(40 * 128 + x) * 4]).toBeLessThan(0.02);
        expect(wiped[(8 * 128 + 64) * 4]).toBeGreaterThan(0.85);
        for (let i = 0; i < 20; i++) update(base);
        const recovered = await memoryPixels(device, memory.texture);
        expect(recovered[(40 * 128 + 64) * 4]).toBeGreaterThan(0.35);
      } finally {
        memory.destroy();
      }
    });
  });
  it('frost coverage grows, heat turns it into water, and meltwater travels down the pane', async () => {
    const device = await getTestDevice();
    await expectValid(device, async () => {
      const memory = createWeatherMemory(device, 96, 96);
      const update = (values: Parameters<typeof memory.update>[1]) => {
        const e = device.createCommandEncoder();
        memory.update(e, values);
        device.queue.submit([e.finish()]);
      };
      const sum = (data: Float32Array, channel: number) =>
        data.reduce((n, v, i) => n + (i % 4 === channel ? v : 0), 0);
      try {
        const base = {
          mode: 5,
          aspect: 1,
          delta: 0.1,
          growth: 1,
          heat: 0,
          cycle: 0,
          stroke_count: 0,
        };
        update(base);
        const young = await memoryPixels(device, memory.texture);
        for (let i = 0; i < 60; i++) update({ ...base, cycle: i * 0.1 });
        const frozen = await memoryPixels(device, memory.texture);
        expect(sum(frozen, 1)).toBeGreaterThan(sum(young, 1) + 100);
        for (let i = 0; i < 20; i++) update({ ...base, cycle: 7, heat: 1 });
        const thawed = await memoryPixels(device, memory.texture);
        expect(sum(thawed, 1)).toBeLessThan(sum(frozen, 1) * 0.1);
        expect(sum(thawed, 2)).toBeGreaterThan(100);
        expect([...thawed].every((v) => Number.isFinite(v) && v >= 0 && v <= 1.001)).toBe(true);
        const center = (data: Float32Array) =>
          data.reduce((n, v, i) => n + (i % 4 === 2 ? v * Math.floor(i / 4 / 96) : 0), 0) /
          sum(data, 2);
        const before = center(thawed);
        for (let i = 0; i < 10; i++) update({ ...base, cycle: 7, heat: 1 });
        const drained = await memoryPixels(device, memory.texture);
        expect(center(drained)).toBeGreaterThan(before);
      } finally {
        memory.destroy();
      }
    });
  });
});

it.each(WEATHER_EFFECT_IDS)(
  '%s survives presets, live input, aspect changes and float rendering',
  async (id) => {
    const device = await getTestDevice();
    await expectValid(device, async () => {
      const effect = await loadEffect(id),
        params = defaultParams(effect.params),
        node = canvas();
      const target = device.createTexture({
        size: [128, 80],
        format: 'rgba16float',
        usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC,
      });
      const instance = await effect.create({
        device,
        format: 'rgba16float',
        canvas: node,
        params,
        profiler: noProfiler,
      });
      let frame = 0;
      const draw = () => {
        const e = device.createCommandEncoder();
        instance.frame({ time: frame / 60, delta: 1 / 60, frame: frame++ }, e, target.createView());
        device.queue.submit([e.finish()]);
      };
      try {
        instance.resize(128, 80, 1);
        draw();
        for (const preset of Object.values(effect.presets ?? {})) {
          Object.assign(params, defaultParams(effect.params), preset);
          draw();
        }
        pointer(node, 'pointerdown', 0.2, 0.5);
        pointer(node, 'pointermove', 0.8, 0.5);
        draw();
        pointer(node, 'pointerup', 0.8, 0.5);
        const values = await memoryPixels(device, target);
        expect([...values].every((v) => Number.isFinite(v) && v >= 0 && v <= 1.2)).toBe(true);
        instance.resize(48, 96, 2);
        draw();
        instance.resize(128, 80, 1);
        draw();
        const paths = ['../gpu/weather/effect.ts', '../gpu/weather/particles.wgsl'];
        expect(weatherSourceFiles(paths, id)).toHaveLength(2);
      } finally {
        instance.destroy();
        target.destroy();
      }
    });
  },
  30000,
);

it('splash feedback changes only after a press when ambient bursts are off', async () => {
  const device = await getTestDevice();
  await expectValid(device, async () => {
    const effect = await loadEffect('splash-burst'),
      params = { ...defaultParams(effect.params), automatic: false, speed: 1 },
      node = canvas();
    const target = device.createTexture({
      size: [128, 80],
      format: 'rgba8unorm',
      usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC,
    });
    const instance = await effect.create({
      device,
      format: 'rgba8unorm',
      canvas: node,
      params,
      profiler: noProfiler,
    });
    let frame = 0;
    const draw = () => {
      const e = device.createCommandEncoder();
      instance.frame({ time: frame / 60, delta: 1 / 60, frame: frame++ }, e, target.createView());
      device.queue.submit([e.finish()]);
    };
    try {
      instance.resize(128, 80, 1);
      draw();
      const before = await readTexture(device, target);
      pointer(node, 'pointermove', 0.5, 0.5);
      draw();
      expect(await readTexture(device, target)).toEqual(before);
      pointer(node, 'pointerdown', 0.5, 0.5);
      for (let i = 0; i < 12; i++) draw();
      const after = await readTexture(device, target);
      expect(difference(before, after)).toBeGreaterThan(0.03);
      expect(Math.max(...after.map((value, i) => Math.abs(value - before[i]!)))).toBeGreaterThan(
        32,
      );
    } finally {
      instance.destroy();
      target.destroy();
    }
  });
});

it('soap-film thickness changes iridescence while stationary bubbles keep their positions', async () => {
  const device = await getTestDevice();
  await expectValid(device, async () => {
    const effect = await loadEffect('bubbles'),
      params = { ...defaultParams(effect.params), style: 'soap', speed: 0, film: 160 },
      node = canvas(256, 160);
    const target = device.createTexture({
      size: [256, 160],
      format: 'rgba8unorm',
      usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC,
    });
    const instance = await effect.create({
      device,
      format: 'rgba8unorm',
      canvas: node,
      params,
      profiler: noProfiler,
    });
    const draw = () => {
      const e = device.createCommandEncoder();
      instance.frame({ time: 0, delta: 0, frame: 0 }, e, target.createView());
      device.queue.submit([e.finish()]);
    };
    try {
      instance.resize(256, 160, 1);
      draw();
      const thin = await readTexture(device, target);
      params.film = 800;
      draw();
      const thick = await readTexture(device, target);
      expect(difference(thin, thick)).toBeGreaterThan(0.05);
    } finally {
      instance.destroy();
      target.destroy();
    }
  });
});

it('fog strength changes opacity without clearing its persistent history', async () => {
  const device = await getTestDevice();
  await expectValid(device, async () => {
    const effect = await loadEffect('fogged-glass'),
      params = { ...defaultParams(effect.params), speed: 0, strength: 1 },
      node = canvas();
    const target = device.createTexture({
      size: [128, 80],
      format: 'rgba8unorm',
      usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC,
    });
    const instance = await effect.create({
      device,
      format: 'rgba8unorm',
      canvas: node,
      params,
      profiler: noProfiler,
    });
    const draw = () => {
      const e = device.createCommandEncoder();
      instance.frame({ time: 0, delta: 0, frame: 0 }, e, target.createView());
      device.queue.submit([e.finish()]);
    };
    try {
      instance.resize(128, 80, 1);
      draw();
      const fog = await readTexture(device, target);
      params.strength = 0;
      draw();
      expect(difference(fog, await readTexture(device, target))).toBeGreaterThan(1);
      params.strength = 1;
      draw();
      expect(await readTexture(device, target)).toEqual(fog);
    } finally {
      instance.destroy();
      target.destroy();
    }
  });
});
