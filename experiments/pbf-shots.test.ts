import { describe, it, vi } from 'vitest';
import { loadEffect } from '../src/effects/registry.ts';
import { defaultParams } from '../src/effects/types.ts';
import { noProfiler } from '../src/gpu/profiler.ts';
import { getTestDevice, readTexture } from '../tests/gpu.ts';

vi.setConfig({ testTimeout: 1_200_000 });

const env = (globalThis as { process?: { env: Record<string, string | undefined> } }).process!.env;

async function writePng(path: string, width: number, height: number, rgb: Uint8Array) {
  const { deflateSync } = (await import(/* @vite-ignore */ 'node:zlib')) as { deflateSync(d: Uint8Array): Uint8Array };
  const { writeFileSync, mkdirSync } = (await import(/* @vite-ignore */ 'node:fs')) as { writeFileSync(p: string, d: Uint8Array): void; mkdirSync(p: string, o: { recursive: boolean }): void };
  mkdirSync(path.slice(0, path.lastIndexOf('/')), { recursive: true });
  const table = new Uint32Array(256).map((_, n) => { let c = n; for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1; return c >>> 0; });
  const crc = (b: Uint8Array) => { let c = 0xffffffff; for (const x of b) c = table[(c ^ x) & 0xff]! ^ (c >>> 8); return (c ^ 0xffffffff) >>> 0; };
  const chunk = (type: string, data: Uint8Array) => { const out = new Uint8Array(12 + data.length); const v = new DataView(out.buffer); v.setUint32(0, data.length); for (let i = 0; i < 4; i++) out[4 + i] = type.charCodeAt(i); out.set(data, 8); v.setUint32(8 + data.length, crc(out.subarray(4, 8 + data.length))); return out; };
  const header = new Uint8Array(13); const hv = new DataView(header.buffer); hv.setUint32(0, width); hv.setUint32(4, height); header.set([8, 2, 0, 0, 0], 8);
  const raw = new Uint8Array((width * 3 + 1) * height);
  for (let y = 0; y < height; y++) raw.set(rgb.subarray(y * width * 3, (y + 1) * width * 3), y * (width * 3 + 1) + 1);
  const parts = [new Uint8Array([137, 80, 78, 71, 13, 10, 26, 10]), chunk('IHDR', header), chunk('IDAT', deflateSync(raw)), chunk('IEND', new Uint8Array(0))];
  const all = new Uint8Array(parts.reduce((n, p) => n + p.length, 0)); let at = 0; for (const p of parts) { all.set(p, at); at += p.length; }
  writeFileSync(path, all);
}

describe.skipIf(!env.PBF_SHOTS)('shots', () => {
  it('renders', async () => {
    const dir = env.PBF_SHOTS!;
    const id = env.PBF_EFFECT ?? 'pbf-3d';
    const W = Number(env.PBF_W ?? 1280), H = Number(env.PBF_H ?? 720);
    const times = JSON.parse(env.PBF_TIMES ?? '[1,3]') as number[];
    const device = await getTestDevice();
    const effect = await loadEffect(id);
    const preset = env.PBF_PRESET ? (effect.presets?.[env.PBF_PRESET] ?? {}) : {};
    const values = { ...defaultParams(effect.params), ...preset, ...JSON.parse(env.PBF_PARAMS ?? '{}') } as Record<string, unknown>;
    const target = device.createTexture({ size: [W, H], format: 'rgba8unorm', usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC });
    const canvas = Object.assign(new EventTarget(), { width: W, height: H, style: {}, getBoundingClientRect: () => ({ left: 0, top: 0, width: W, height: H }), setPointerCapture() {}, releasePointerCapture() {} }) as unknown as HTMLCanvasElement;
    const instance = await effect.create({ device, format: 'rgba8unorm', canvas, params: values as never, profiler: noProfiler, hdr: false });
    instance.resize(W, H, 1);
    const script = JSON.parse(env.PBF_SCRIPT ?? '[]') as { t: number; type: string; x: number; y: number }[];
    let frame = 0; let t = 0; let si = 0;
    const fire = (type: string, x: number, y: number, id = 1) => {
      const e = new Event(type);
      const props: Record<string, unknown> = { pointerId: id, pointerType: 'mouse', clientX: x * W, clientY: y * H, timeStamp: t * 1000 + 1, pressure: type === 'pointerdown' ? 0.5 : 0, getCoalescedEvents: () => [] };
      for (const [k, v] of Object.entries(props)) Object.defineProperty(e, k, { value: v, configurable: true });
      canvas.dispatchEvent(e);
    };
    for (const shot of times) {
      while (t < shot) {
        while (si < script.length && script[si]!.t <= t) { fire(script[si]!.type, script[si]!.x, script[si]!.y); si++; }
        const enc = device.createCommandEncoder();
        instance.frame({ time: t, delta: 1 / 60, frame: frame++ }, enc, target.createView());
        device.queue.submit([enc.finish()]);
        t += 1 / 60;
        if (frame % 20 === 0) await device.queue.onSubmittedWorkDone();
      }
      await device.queue.onSubmittedWorkDone();
      const px = await readTexture(device, target);
      const rgb = new Uint8Array(W * H * 3);
      for (let i = 0; i < W * H; i++) { rgb[i * 3] = px[i * 4]!; rgb[i * 3 + 1] = px[i * 4 + 1]!; rgb[i * 3 + 2] = px[i * 4 + 2]!; }
      await writePng(`${dir}/${id}-${env.PBF_NAME ?? 'shot'}-t${shot}.png`, W, H, rgb);
    }
    instance.destroy();
  });
});
