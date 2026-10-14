import { describe, it, expect } from 'vitest';
import { getTestDevice, expectValid, readTexture } from './gpu.ts';
import { sampleWaves, sampleWorldWaves } from '../src/gpu/ocean/waves.ts';
import { oceanMesh } from '../src/gpu/ocean/mesh.ts';
import { oceanSpectrum, OCEAN_LENGTHS } from '../src/gpu/ocean/spectrum.ts';
import { createSpectralOcean } from '../src/gpu/ocean/fft.ts';
import { createBuoyancyReadback, BUOY_POINTS } from '../src/gpu/ocean/buoyancy.ts';
import { noProfiler } from '../src/gpu/profiler.ts';
import { riverFlow } from '../src/gpu/ocean/flow.ts';
async function readBuffer(device: GPUDevice, buffer: GPUBuffer, size: number) {
  const staging = device.createBuffer({
    size,
    usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
  });
  const enc = device.createCommandEncoder();
  enc.copyBufferToBuffer(buffer, 0, staging, 0, size);
  device.queue.submit([enc.finish()]);
  await staging.mapAsync(GPUMapMode.READ);
  const out = new Float32Array(staging.getMappedRange().slice(0));
  staging.unmap();
  staging.destroy();
  return out;
}
function half(v: number) {
  const s = v & 0x8000 ? -1 : 1,
    e = (v >> 10) & 31,
    m = v & 1023;
  return s * (e === 0 ? m * 2 ** -24 : (1 + m / 1024) * 2 ** (e - 15));
}
describe('ocean foundations', () => {
  it('analytic tangents agree with displaced finite differences', () => {
    for (const [x, z, t] of [
      [0, 0, 0],
      [3, -9, 1.4],
      [-12, 20, 6],
    ]) {
      const w = sampleWaves(x!, z!, t!),
        eps = 1e-4;
      const a = sampleWaves(x! + eps, z!, t!),
        b = sampleWaves(x!, z! + eps, t!);
      for (let j = 0; j < 3; j++) {
        expect((a.position[j]! - w.position[j]!) / eps).toBeCloseTo(w.dx[j]!, 3);
        expect((b.position[j]! - w.position[j]!) / eps).toBeCloseTo(w.dz[j]!, 3);
      }
      expect(Math.hypot(...w.normal)).toBeCloseTo(1, 8);
      expect(w.normal[1]).toBeGreaterThan(0);
    }
  });
  it('world-space inverse removes the horizontal buoyancy error', () => {
    for (let t = 0; t < 5; t += 0.2) {
      const w = sampleWorldWaves(-4, -8, t, 1, 0.85);
      expect(w.position[0]).toBeCloseTo(-4, 6);
      expect(w.position[2]).toBeCloseTo(-8, 6);
    }
  });
  it.each(['projected', 'clipmap'] as const)(
    '%s meshes are indexed, finite and bounded',
    (strategy) => {
      const m = oceanMesh(strategy, 48);
      expect(m.indices.length % 3).toBe(0);
      expect([...m.vertices].every(Number.isFinite)).toBe(true);
      expect(Math.max(...m.indices)).toBeLessThan(m.vertices.length / 4);
      expect(m.triangles).toBe(strategy === 'projected' ? 4608 : 18432);
    },
  );
  it('stitches ring edges to the coarser lattice', () => {
    const m = oceanMesh('clipmap', 48);
    for (let i = 0; i < 49; i++) {
      const at = i * 4;
      expect(Math.abs(m.vertices[at]! % 1)).toBe(0);
      expect(Math.abs(m.vertices[at + 1]! % 1)).toBe(0);
    }
  });
  it('spectrum is deterministic, band limited and has no DC energy', () => {
    const a = oceanSpectrum({ size: 64 });
    expect(a).toEqual(oceanSpectrum({ size: 64 }));
    expect(a).not.toEqual(oceanSpectrum({ size: 64, seed: 20 }));
    for (let l = 0; l < 3; l++) {
      expect(a[l * 64 * 64 * 4]).toBe(0);
      expect(a[l * 64 * 64 * 4 + 1]).toBe(0);
    }
    expect(() => oceanSpectrum({ size: 63 })).toThrow();
    expect(() => oceanMesh('clipmap', 51)).toThrow();
  });
  it('potential current is tangent to an isolated rock and approaches the far-field flow', () => {
    for (let theta = 0; theta < 6; theta += 0.3) {
      const x = Math.cos(theta),
        z = Math.sin(theta);
      const v = riverFlow(x, z, [[0, 0, 1]]);
      expect(v[0] * x + v[1] * z).toBeCloseTo(0, 7);
    }
    expect(riverFlow(1000, 1000, [[0, 0, 1]])[1]).toBeCloseTo(-1, 4);
  });
  it('GPU buoyancy agrees with the CPU mirror', async () => {
    const device = await getTestDevice();
    await expectValid(device, async () => {
      const query = createBuoyancyReadback(device, noProfiler);
      for (const t of [0, 1.4, 5.1]) {
        const encoder = device.createCommandEncoder();
        query.run(encoder, t, 1, 0.85);
        device.queue.submit([encoder.finish()]);
        const gpu = await readBuffer(device, query.output, BUOY_POINTS.byteLength);
        for (let i = 0; i < BUOY_POINTS.length; i += 4) {
          const w = sampleWorldWaves(BUOY_POINTS[i]!, BUOY_POINTS[i + 2]!, t, 1, 0.85);
          for (let j = 0; j < 3; j++) expect(gpu[i + j]).toBeCloseTo(w.position[j]!, 4);
        } // Start mapping the previous submitted copy so the next query can be encoded.
        const e = device.createCommandEncoder();
        query.run(e, t, 1, 0.85);
        device.queue.submit([e.finish()]);
        await device.queue.onSubmittedWorkDone();
        await new Promise((r) => setTimeout(r, 2));
      }
      query.destroy();
    });
  });
  it('GPU 2D inverse FFT matches a direct complex DFT for all displacement channels', async () => {
    const device = await getTestDevice();
    await expectValid(device, async () => {
      const n = 8,
        area = n * n,
        time = 0.7,
        seeds = oceanSpectrum({ size: n });
      const fft = createSpectralOcean(device, n);
      const enc = device.createCommandEncoder();
      fft.run(enc, time, 1 / 60, { chop: 1, strength: 1 });
      device.queue.submit([enc.finish()]);
      const spatial = await readBuffer(device, fft.spatial, 6 * area * 16);
      for (let l = 0; l < 3; l++)
        for (let y = 0; y < n; y++)
          for (let x = 0; x < n; x++) {
            let h = 0,
              dx = 0,
              dz = 0;
            for (let ky = 0; ky < n; ky++)
              for (let kx = 0; kx < n; kx++) {
                const i = (l * area + ky * n + kx) * 4,
                  k1 = ((kx <= n / 2 ? kx : kx - n) * 2 * Math.PI) / OCEAN_LENGTHS[l]!,
                  k2 = ((ky <= n / 2 ? ky : ky - n) * 2 * Math.PI) / OCEAN_LENGTHS[l]!;
                const length = Math.hypot(k1, k2),
                  phase = Math.sqrt(9.81 * length) * time,
                  c = Math.cos(phase),
                  s = Math.sin(phase);
                const re = (seeds[i]! + seeds[i + 2]!) * c - (seeds[i + 1]! + seeds[i + 3]!) * s;
                const im = (seeds[i]! - seeds[i + 2]!) * s + (seeds[i + 1]! - seeds[i + 3]!) * c;
                const theta = (2 * Math.PI * (kx * x + ky * y)) / n,
                  ct = Math.cos(theta),
                  st = Math.sin(theta);
                h += re * ct - im * st;
                if (length > 0) {
                  if (kx !== n / 2) dx += ((im * ct + re * st) * k1) / length;
                  if (ky !== n / 2) dz += ((im * ct + re * st) * k2) / length;
                }
              }
            const i = (l * area + y * n + x) * 4;
            expect(spatial[i]! / area).toBeCloseTo(h / area, 4);
            expect(spatial[i + 2]! / area).toBeCloseTo(dx / area, 4);
            expect(spatial[3 * area * 4 + i]! / area).toBeCloseTo(dz / area, 4);
            expect(Math.abs(spatial[i + 1]! / area)).toBeLessThan(1e-5);
          }
      fft.destroy();
    });
  });
  it('folding foam accumulates, survives a flat frame, then decays at the chosen lifetime', async () => {
    const device = await getTestDevice();
    await expectValid(device, async () => {
      const fft = createSpectralOcean(device, 64);
      for (let f = 0; f < 30; f++) {
        const e = device.createCommandEncoder();
        fft.run(e, 0.5, 1 / 60, { strength: 2, chop: 1.8, foamThreshold: 1.2 });
        device.queue.submit([e.finish()]);
      }
      const get = async () => {
        const bytes = await readTexture(device, fft.derivativeTexture, 8);
        const values = new Uint16Array(bytes.buffer);
        return [...values].filter((_, i) => i % 4 === 3).map(half);
      };
      const before = await get();
      expect(Math.max(...before)).toBeGreaterThan(0.1);
      const e = device.createCommandEncoder();
      fft.run(e, 0.5, 1 / 60, { strength: 0, foamThreshold: 0.5, foamLifetime: 3 });
      device.queue.submit([e.finish()]);
      const after = await get();
      const i = before.indexOf(Math.max(...before));
      expect(after[i]).toBeGreaterThan(0);
      expect(after[i]! / before[i]!).toBeCloseTo(Math.exp(-1 / 180), 2);
      fft.destroy();
    });
  });
});

describe('ocean scene variants', () => {
  it('shared source files expose the renderer and only the relevant specialized algorithms', async () => {
    const { oceanSourceFiles } = await import('../src/gpu/ocean/sources.ts');
    const paths = [
      'scene.wgsl',
      'effect.ts',
      'mesh.ts',
      'waves.ts',
      'fft.ts',
      'fft.wgsl',
      'spectrum.ts',
      'flow.ts',
      'spray.ts',
      'buoyancy.ts',
    ].map((n) => `../gpu/ocean/${n}`);
    expect(oceanSourceFiles(paths, 'fft-ocean').map((f) => f.name)).toContain('ocean/fft.wgsl');
    expect(oceanSourceFiles(paths, 'river').map((f) => f.name)).not.toContain('ocean/fft.wgsl');
    expect(oceanSourceFiles(paths, 'hello')).toEqual([]);
  });
  it('switches mesh density, FFT allocation, waterline and GPU sampling without leaking invalid bindings', async () => {
    const { loadEffect } = await import('../src/effects/registry.ts');
    const { defaultParams } = await import('../src/effects/types.ts');
    const device = await getTestDevice();
    await expectValid(device, async () => {
      const target = device.createTexture({
        size: [160, 90],
        format: 'rgba8unorm',
        usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC,
      });
      const canvas = Object.assign(new EventTarget(), {
        width: 160,
        height: 90,
        style: {},
        getBoundingClientRect: () => ({ left: 0, top: 0, width: 160, height: 90 }),
      }) as unknown as HTMLCanvasElement;
      for (const id of [
        'gerstner-ocean',
        'fft-ocean',
        'shoreline',
        'river',
        'waterfall',
        'waterline',
        'calm-lake',
      ]) {
        const effect = await loadEffect(id);
        const params = defaultParams(effect.params);
        const instance = await effect.create({
          device,
          format: 'rgba8unorm',
          canvas,
          params,
          profiler: noProfiler,
        });
        instance.resize(160, 90, 1);
        const draw = async (view = target.createView()) => {
          const e = device.createCommandEncoder();
          instance.frame({ time: 1, delta: 1 / 60, frame: 1 }, e, view);
          device.queue.submit([e.finish()]);
          await device.queue.onSubmittedWorkDone();
        };
        for (const preset of Object.values(effect.presets ?? {})) {
          Object.assign(params, defaultParams(effect.params), preset);
          await draw();
        }
        params.mesh = 'clipmap';
        params.cells = '160';
        await draw();
        params.mesh = 'projected';
        params.cells = '48';
        await draw();
        if (id === 'fft-ocean') {
          for (const size of ['128', '256', '64']) {
            params.fftSize = size;
            await draw();
          }
        }
        if (id === 'waterline') {
          for (const cameraHeight of [-1, 0, 0.2, 1]) {
            params.cameraHeight = cameraHeight;
            await draw();
            const pixels = await readTexture(device, target);
            expect(new Set(pixels).size).toBeGreaterThan(16);
          }
        }
        if (id === 'gerstner-ocean') {
          params.buoyancy = 'gpu';
          const e = device.createCommandEncoder();
          for (let j = 0; j < 4; j++)
            instance.frame({ time: 1, delta: 1 / 60, frame: j }, e, target.createView());
          device.queue.submit([e.finish()]);
          await device.queue.onSubmittedWorkDone();
          await draw();
        }
        instance.resize(80, 45, 1);
        const resized = device.createTexture({
          size: [80, 45],
          format: 'rgba8unorm',
          usage: GPUTextureUsage.RENDER_ATTACHMENT,
        });
        await draw(resized.createView());
        resized.destroy();
        instance.destroy();
      }
      target.destroy();
    });
  }, 30000);
});
