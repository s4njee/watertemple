import { describe, expect, it } from 'vitest';
import { loadEffect } from '../src/effects/registry.ts';
import { defaultParams } from '../src/effects/types.ts';
import { noProfiler } from '../src/gpu/profiler.ts';
import { decodeAtlas } from '../src/gpu/ui/atlas.ts';
import {
  advanceSpring,
  mergeDrops,
  advanceDrops,
  segmentDistance,
  seedDrops,
} from '../src/gpu/ui/dynamics.ts';
import { htmlCanvasCaptureSupport } from '../src/gpu/ui/capture.ts';
import { UI_EFFECT_IDS, uiSourceFiles } from '../src/gpu/ui/sources.ts';
import { expectValid, getTestDevice, readTexture } from './gpu.ts';

function canvas(width = 128, height = 80) {
  const attrs = new Map<string, string>();
  return Object.assign(new EventTarget(), {
    width,
    height,
    style: {},
    getBoundingClientRect: () => ({ left: 0, top: 0, width, height }),
    getAttribute: (key: string) => attrs.get(key) ?? null,
    setAttribute: (key: string, value: string) => attrs.set(key, value),
    removeAttribute: (key: string) => attrs.delete(key),
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
    getCoalescedEvents: { value: () => [] },
  });
  node.dispatchEvent(e);
}
const difference = (a: Uint8Array, b: Uint8Array) =>
  a.reduce((sum, v, i) => sum + Math.abs(v - b[i]!), 0) / a.length;
async function fixture(id: string, overrides: Record<string, string | number | boolean> = {}) {
  const device = await getTestDevice(),
    effect = await loadEffect(id),
    params = { ...defaultParams(effect.params), ...overrides },
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
  instance.resize(128, 80, 1);
  let frame = 0;
  const draw = async (count = 1) => {
    for (let i = 0; i < count; i++) {
      const encoder = device.createCommandEncoder();
      instance.frame(
        { time: frame / 60, delta: 1 / 60, frame: frame++ },
        encoder,
        target.createView(),
      );
      device.queue.submit([encoder.finish()]);
    }
    return readTexture(device, target);
  };
  return {
    device,
    effect,
    params,
    node,
    instance,
    draw,
    destroy() {
      instance.destroy();
      target.destroy();
    },
  };
}

describe('liquid UI mechanics', () => {
  it.each([0.12, 1, 1.4])('spring at damping %s is independent of step partition', (damping) => {
    const a = { value: 0, velocity: 2 },
      b = { ...a };
    advanceSpring(a, 1, 0.8, 3, damping);
    for (let i = 0; i < 96; i++) advanceSpring(b, 1, 1 / 120, 3, damping);
    expect(a.value).toBeCloseTo(b.value, 9);
    expect(a.velocity).toBeCloseTo(b.velocity, 9);
  });
  it('spring settles and zero dt preserves a paused state', () => {
    const s = { value: 0, velocity: 0 };
    advanceSpring(s, 1, 0, 3, 0.2);
    expect(s).toEqual({ value: 0, velocity: 0 });
    advanceSpring(s, 1, 10, 3, 0.2);
    expect(s.value).toBeCloseTo(1, 7);
  });
  it('drop coalescence conserves r³, centroid and velocity', () => {
    const a = { x: 0.2, y: 0.3, radius: 0.02, velocity: 0.4 },
      b = { x: 0.25, y: 0.32, radius: 0.03, velocity: 0.1 };
    const c = mergeDrops(a, b);
    expect(c.radius ** 3).toBeCloseTo(a.radius ** 3 + b.radius ** 3, 12);
    expect(c.x * c.radius ** 3).toBeCloseTo(a.x * a.radius ** 3 + b.x * b.radius ** 3, 12);
    expect(c.velocity * c.radius ** 3).toBeCloseTo(
      a.velocity * a.radius ** 3 + b.velocity * b.radius ** 3,
      12,
    );
  });
  it('gravity moves mobile drops and preserves capillary-pinned ones', () => {
    const drops = [
      { x: 0.2, y: 0.3, radius: 0.008, velocity: 0 },
      { x: 0.6, y: 0.3, radius: 0.03, velocity: 0 },
    ];
    advanceDrops(drops, 0.1, 1);
    expect(drops[0]!.y).toBe(0.3);
    expect(drops[1]!.y).toBeGreaterThan(0.3);
  });
  it('swept collision catches a fast drop passing a pinned drop', () => {
    const drops = [
      { x: 0.5, y: 0.22, radius: 0.022, velocity: 5 },
      { x: 0.5, y: 0.29, radius: 0.01, velocity: 0 },
    ];
    advanceDrops(drops, 0.03, 0);
    expect(drops).toHaveLength(1);
    expect(drops[0]!.radius).toBeGreaterThan(0.022);
  });
  it('departed drops drain and do not wrap back into the card', () => {
    const drops = [{ x: 0.5, y: 0.98, radius: 0.03, velocity: 0.3 }];
    advanceDrops(drops, 0.1, 1);
    expect(drops).toHaveLength(0);
  });
  it('seed is bounded, deterministic, and zero-speed drops remain unchanged', () => {
    const drops = seedDrops();
    expect(drops).toHaveLength(32);
    expect(drops).toEqual(seedDrops());
    const before = structuredClone(drops);
    advanceDrops(drops, 0, 1);
    expect(drops).toEqual(before);
  });
  it('streak distance is continuous and handles stationary input', () => {
    expect(segmentDistance(0.5, 0.2, 0, 0, 1, 0)).toBeCloseTo(0.2);
    expect(segmentDistance(0.5, 0.5, 0.5, 0.5, 0.5, 0.5)).toBe(0);
  });
  it('MSDF atlas has independent colour channels and interior/exterior distances', async () => {
    const bytes = await decodeAtlas();
    expect(bytes).toHaveLength(384 * 144 * 4);
    let distinct = 0,
      inside = 0,
      outside = 0;
    for (let i = 0; i < bytes.length; i += 4) {
      const rgb = [bytes[i]!, bytes[i + 1]!, bytes[i + 2]!].sort((a, b) => a - b);
      if (rgb[2]! - rgb[0]! > 20) distinct++;
      if (rgb[1]! > 180) inside++;
      if (rgb[1]! < 60) outside++;
      expect(bytes[i + 3]).toBe(255);
    }
    expect(distinct).toBeGreaterThan(1000);
    expect(inside).toBeGreaterThan(1000);
    expect(outside).toBeGreaterThan(1000);
  });
  it('experimental capture is detected without invoking it', () => {
    expect(htmlCanvasCaptureSupport({})).toBe('unavailable');
    expect(htmlCanvasCaptureSupport({ drawElementImageToTexture() {} })).toBe('draw');
    expect(htmlCanvasCaptureSupport({ copyElementImageToTexture() {} })).toBe('legacy-copy');
  });
  it('source viewer exposes the shared renderer and only the trail shader for trails', () => {
    const paths = ['../gpu/ui/display.wgsl', '../gpu/ui/trail.wgsl', '../gpu/ui/effect.ts'];
    expect(uiSourceFiles(paths, 'water-cursor').map((f) => f.name)).toContain('ui/trail.wgsl');
    expect(uiSourceFiles(paths, 'liquid-glass').map((f) => f.name)).not.toContain('ui/trail.wgsl');
    expect(uiSourceFiles(paths, 'hello')).toEqual([]);
  });
});

describe('liquid UI GPU behavior', () => {
  it.each(UI_EFFECT_IDS)(
    '%s validates every preset, portrait resize and HDR output',
    async (id) => {
      const device = await getTestDevice(),
        effect = await loadEffect(id);
      await expectValid(device, async () => {
        for (const overrides of [{}, ...Object.values(effect.presets ?? {})]) {
          const node = canvas(),
            params = defaultParams(effect.params);
          Object.assign(params, overrides);
          const instance = await effect.create({
            device,
            canvas: node,
            format: 'rgba16float',
            params,
            profiler: noProfiler,
          });
          for (const [width, height, dpr] of [
            [96, 60, 1],
            [48, 96, 2],
            [96, 60, 1],
          ]) {
            const target = device.createTexture({
              size: [width!, height!],
              format: 'rgba16float',
              usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC,
            });
            instance.resize(width!, height!, dpr!);
            const encoder = device.createCommandEncoder();
            instance.frame({ time: 1, delta: 1 / 60, frame: 1 }, encoder, target.createView());
            device.queue.submit([encoder.finish()]);
            const bytes = await readTexture(device, target, 8);
            const halves = new Uint16Array(bytes.buffer);
            expect(halves.every((v) => (v & 0x7c00) !== 0x7c00)).toBe(true);
            target.destroy();
          }
          instance.destroy();
          instance.destroy();
        }
      });
    },
    30000,
  );
  it('wet trail persists after leaving and dries back to the backdrop', async () => {
    const f = await fixture('water-cursor', { dry: 3 });
    try {
      const clean = await f.draw();
      pointer(f.node, 'pointerenter', 0.25, 0.5);
      pointer(f.node, 'pointermove', 0.75, 0.5);
      const wet = await f.draw(3);
      pointer(f.node, 'pointerleave', 0.75, 0.5);
      const lingering = await f.draw(4);
      const dry = await f.draw(180);
      expect(difference(wet, clean)).toBeGreaterThan(0.08);
      expect(difference(lingering, clean)).toBeGreaterThan(0.02);
      expect(difference(dry, clean)).toBeLessThan(difference(lingering, clean) * 0.1);
    } finally {
      f.destroy();
    }
  });
  it('hover, keyboard press and cleanup operate the button', async () => {
    const f = await fixture('liquid-buttons');
    try {
      expect(f.node.getAttribute('role')).toBe('button');
      const before = await f.draw();
      f.node.dispatchEvent(new Event('focus'));
      const hover = await f.draw(12);
      const key = new Event('keydown', { cancelable: true });
      Object.defineProperty(key, 'key', { value: 'Enter' });
      f.node.dispatchEvent(key);
      const press = await f.draw(8);
      expect(key.defaultPrevented).toBe(true);
      expect(difference(hover, before)).toBeGreaterThan(0.01);
      expect(difference(press, hover)).toBeGreaterThan(0.01);
    } finally {
      f.destroy();
      expect(f.node.getAttribute('role')).toBeNull();
      expect(f.node.getAttribute('tabindex')).toBeNull();
    }
  });
  it('all transition styles preserve the same two exact endpoints', async () => {
    const images: Uint8Array[][] = [];
    for (const style of ['drip', 'wave', 'dissolve']) {
      const f = await fixture('liquid-transitions', { automatic: false, style, progress: 0 });
      try {
        const a = await f.draw();
        f.params.progress = 1;
        const b = await f.draw();
        images.push([a, b]);
        expect(difference(a, b)).toBeGreaterThan(10);
      } finally {
        f.destroy();
      }
    }
    for (let i = 1; i < images.length; i++) {
      expect(images[i]![0]).toEqual(images[0]![0]);
      expect(images[i]![1]).toEqual(images[0]![1]);
    }
  });
  it('image ripple responds only when the pointer is inside', async () => {
    const f = await fixture('liquid-image-hover');
    try {
      const a = await f.draw();
      pointer(f.node, 'pointerenter', 0.7, 0.5);
      const b = await f.draw(6);
      expect(difference(a, b)).toBeGreaterThan(0.1);
      pointer(f.node, 'pointerleave', 0.7, 0.5);
      expect(await f.draw()).toEqual(a);
    } finally {
      f.destroy();
    }
  });
  it.each(['tank', 'bar', 'loader'])(
    '%s exposes empty and full fill levels to the GPU and accessibility',
    async (style) => {
      const f = await fixture('fill-level', { style, level: 0 });
      try {
        const a = await f.draw();
        expect(f.node.getAttribute('aria-valuenow')).toBe('0');
        f.params.level = 1;
        const b = await f.draw();
        expect(f.node.getAttribute('aria-valuenow')).toBe('100');
        expect(difference(a, b)).toBeGreaterThan(1);
      } finally {
        f.destroy();
      }
    },
  );
  it('scroll preview reaches both endpoints without a DOM', async () => {
    const f = await fixture('scroll-water', { preview: true, level: 0 });
    try {
      const a = await f.draw();
      f.params.level = 1;
      expect(difference(a, await f.draw())).toBeGreaterThan(3);
    } finally {
      f.destroy();
    }
  });
});
