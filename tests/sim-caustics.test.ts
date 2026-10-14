import { describe, expect, it, vi } from 'vitest';
import {
  createHeightfieldCaustics,
  flatLandingShift,
  gridIndices,
  lightDirection,
  refractVector,
  type CausticsMethod,
  type CausticsSettings,
  type HeightfieldCaustics,
} from '../src/gpu/sim/caustics.ts';
import { createHeightfield, type Heightfield } from '../src/gpu/sim/heightfield.ts';
import { expectValid, getTestDevice, readTexture } from './gpu.ts';

vi.setConfig({ testTimeout: 60_000 });

const METHODS: CausticsMethod[] = ['mesh', 'jacobian'];

function halfToFloat(bits: number): number {
  const sign = bits & 0x8000 ? -1 : 1;
  const exponent = (bits >> 10) & 0x1f;
  const fraction = bits & 0x3ff;
  if (exponent === 0) return sign * 2 ** -14 * (fraction / 1024);
  if (exponent === 31) return fraction ? Number.NaN : sign * Infinity;
  return sign * 2 ** (exponent - 15) * (1 + fraction / 1024);
}

interface Image {
  width: number;
  height: number;
  /** Channels, row-major, row 0 at the top (uv.y = 0). */
  r: Float32Array;
  g: Float32Array;
  b: Float32Array;
  a: Float32Array;
  at(x: number, y: number): number;
}

async function read(device: GPUDevice, caustics: HeightfieldCaustics): Promise<Image> {
  const { width, height } = caustics;
  const bytes = await readTexture(device, caustics.texture, 8);
  const half = new Uint16Array(bytes.buffer, bytes.byteOffset, width * height * 4);
  const channel = (k: number) => {
    const out = new Float32Array(width * height);
    for (let i = 0; i < out.length; i++) out[i] = halfToFloat(half[i * 4 + k]!);
    return out;
  };
  const g = channel(1);
  return {
    width,
    height,
    r: channel(0),
    g,
    b: channel(2),
    a: channel(3),
    at: (x, y) => g[y * width + x]!,
  };
}

/** Mean of the green channel over a box given in uv. */
function meanOver(image: Image, u0 = 0, v0 = 0, u1 = 1, v1 = 1): number {
  let sum = 0;
  let count = 0;
  for (let y = Math.floor(v0 * image.height); y < Math.ceil(v1 * image.height); y++) {
    for (let x = Math.floor(u0 * image.width); x < Math.ceil(u1 * image.width); x++) {
      sum += image.at(x, y);
      count++;
    }
  }
  return sum / Math.max(count, 1);
}

function stats(image: Image) {
  let min = Infinity;
  let max = -Infinity;
  let sum = 0;
  for (const v of image.g) {
    min = Math.min(min, v);
    max = Math.max(max, v);
    sum += v;
  }
  return { min, max, mean: sum / image.g.length };
}

/** Pearson correlation of the two images' green channels. */
function correlation(a: Image, b: Image): number {
  let ma = 0;
  let mb = 0;
  const n = a.g.length;
  for (let i = 0; i < n; i++) {
    ma += a.g[i]!;
    mb += b.g[i]!;
  }
  ma /= n;
  mb /= n;
  let sab = 0;
  let saa = 0;
  let sbb = 0;
  for (let i = 0; i < n; i++) {
    const da = a.g[i]! - ma;
    const db = b.g[i]! - mb;
    sab += da * db;
    saa += da * da;
    sbb += db * db;
  }
  return sab / Math.sqrt(saa * sbb);
}

async function run(device: GPUDevice, fn: (encoder: GPUCommandEncoder) => void): Promise<void> {
  const encoder = device.createCommandEncoder();
  fn(encoder);
  device.queue.submit([encoder.finish()]);
  await device.queue.onSubmittedWorkDone();
}

function createField(device: GPUDevice, resolution = 128, aspect = 1): Heightfield {
  return createHeightfield({
    device,
    resolution,
    width: aspect,
    height: 1,
    params: { damping: 0, leak: 0, boundary: 'absorb' },
  });
}

const OVERHEAD: [number, number, number] = [0, 0, 1];

describe('caustics helpers', () => {
  it('has no sideways shift for overhead light and the Snell shift for a slanted one', () => {
    const [x0, y0] = flatLandingShift(OVERHEAD, 0.5);
    expect(Math.hypot(x0, y0)).toBeLessThan(1e-6);
    // 45 degrees in the air: sin(theta_w) = sin45 / 1.333, shift = depth * tan(theta_w).
    const sinW = Math.SQRT1_2 / 1.333;
    const expected = 0.5 * (sinW / Math.sqrt(1 - sinW * sinW));
    const [x1, y1] = flatLandingShift(lightDirection(Math.PI / 4, 0), 0.5);
    expect(x1).toBeCloseTo(-expected, 4); // the beam goes away from the sun: light from +x lands at -x
    expect(y1).toBeCloseTo(0, 6);
    // y points down on the canvas, so light from the top (azimuth 90 degrees) lands lower.
    const [x2, y2] = flatLandingShift(lightDirection(Math.PI / 4, Math.PI / 2), 0.5);
    expect(x2).toBeCloseTo(0, 5);
    expect(y2).toBeGreaterThan(0.1);
  });

  it('refracts like Snell', () => {
    const l = lightDirection(Math.PI / 4, 0);
    const r = refractVector([-l[0], -l[1], -l[2]], [0, 0, 1], 1 / 1.333);
    expect(Math.hypot(...r)).toBeCloseTo(1, 5);
    expect(Math.hypot(r[0], r[1]) / 1).toBeCloseTo(Math.SQRT1_2 / 1.333, 5);
  });

  it('builds a grid index list that stays inside the vertex range', () => {
    const indices = gridIndices(7, 5);
    expect(indices.length).toBe(7 * 5 * 6);
    expect(Math.max(...indices)).toBe(8 * 6 - 1);
    expect(new Set(indices).size).toBe(8 * 6);
  });

  it('gives unit light directions', () => {
    for (const [e, a] of [
      [0.2, 1],
      [1.2, 4],
      [Math.PI / 2, 0],
    ] as const) {
      expect(Math.hypot(...lightDirection(e, a))).toBeCloseTo(1, 6);
    }
  });
});

describe.each(METHODS)('caustics from a flat surface (%s)', (method) => {
  it('is uniform intensity 1 under overhead and slanted light', async () => {
    const device = await getTestDevice();
    await expectValid(device, async () => {
      const field = createField(device, 96);
      const caustics = createHeightfieldCaustics({
        device,
        heightfield: field,
        resolution: 128,
        method,
      });
      for (const lightDir of [OVERHEAD, lightDirection(0.9, 0.7), lightDirection(0.4, 3.5)]) {
        await run(device, (e) => caustics.update(e, { lightDir, depth: 0.4 }));
        const image = await read(device, caustics);
        const { min, max, mean } = stats(image);
        expect(min).toBeGreaterThan(0.97);
        expect(max).toBeLessThan(1.03);
        expect(mean).toBeCloseTo(1, 2);
        expect(image.a[0]).toBe(1);
      }
      caustics.destroy();
      field.destroy();
    });
  });

  it('leaves the strip the wall shadows dark when the surface does not extend', async () => {
    const device = await getTestDevice();
    await expectValid(device, async () => {
      const field = createField(device, 96);
      const caustics = createHeightfieldCaustics({
        device,
        heightfield: field,
        resolution: 128,
        method,
      });
      const lightDir = lightDirection(0.6, 0); // from +x: the beam lands toward -x
      const settings: CausticsSettings = { lightDir, depth: 0.4, extend: false };
      await run(device, (e) => caustics.update(e, settings));
      const image = await read(device, caustics);
      const [shift] = flatLandingShift(lightDir, 0.4);
      expect(shift).toBeLessThan(-0.1);
      // The floor within |shift| of the +x edge receives light from beyond the pool: dark.
      expect(meanOver(image, 1 + shift + 0.04, 0.1, 0.99, 0.9)).toBeLessThan(0.05);
      // The rest is untouched.
      expect(meanOver(image, 0.02, 0.1, 1 + shift - 0.04, 0.9)).toBeGreaterThan(0.97);
      caustics.destroy();
      field.destroy();
    });
  });
});

describe.each(METHODS)('caustics from a smooth bump (%s)', (method) => {
  it('focuses light where the lens says and conserves it on average', async () => {
    const device = await getTestDevice();
    await expectValid(device, async () => {
      const field = createField(device, 128);
      // A dome: a converging lens. Sized so that it focuses at about the floor.
      field.addDrop(0.5, 0.5, 0.2, 1.6);
      await run(device, (e) => field.advance(e, 0));
      const caustics = createHeightfieldCaustics({
        device,
        heightfield: field,
        resolution: 256,
        method,
      });
      await run(device, (e) =>
        caustics.update(e, {
          lightDir: OVERHEAD,
          depth: 0.4,
          strength: 1,
          smooth: method === 'mesh' ? 1 : 0,
        }),
      );
      const image = await read(device, caustics);
      const { min, max, mean } = stats(image);
      expect(Number.isFinite(max)).toBe(true);
      expect(mean).toBeGreaterThan(0.93);
      expect(mean).toBeLessThan(1.07);
      // The centre of the lens is much brighter than flat light...
      const centre = meanOver(image, 0.47, 0.47, 0.53, 0.53);
      expect(centre).toBeGreaterThan(1.5);
      // ...the light comes from somewhere: a dimmer annular region exists...
      expect(min).toBeLessThan(0.8);
      // ...and the far corners, outside the bump, are untouched.
      expect(meanOver(image, 0, 0, 0.1, 0.1)).toBeCloseTo(1, 1);
      // The brightest texel is near the centre of the bump (a ring would be too, within its radius).
      let best = 0;
      let bx = 0;
      let by = 0;
      for (let y = 0; y < image.height; y++) {
        for (let x = 0; x < image.width; x++) {
          if (image.at(x, y) > best) {
            best = image.at(x, y);
            bx = x;
            by = y;
          }
        }
      }
      expect(Math.hypot(bx / image.width - 0.5, by / image.height - 0.5)).toBeLessThan(0.25);
      caustics.destroy();
      field.destroy();
    });
  });

  it('shifts the whole lens with the slanted light', async () => {
    const device = await getTestDevice();
    await expectValid(device, async () => {
      const field = createField(device, 128);
      field.addDrop(0.5, 0.5, 0.15, 1.6);
      await run(device, (e) => field.advance(e, 0));
      const caustics = createHeightfieldCaustics({
        device,
        heightfield: field,
        resolution: 256,
        method,
      });
      const lightDir = lightDirection(0.7, 0);
      const depth = 0.4;
      await run(device, (e) => caustics.update(e, { lightDir, depth, smooth: 1 }));
      const image = await read(device, caustics);
      const [dx, dy] = flatLandingShift(lightDir, depth);
      const shifted = meanOver(
        image,
        0.5 + dx - 0.03,
        0.5 + dy - 0.03,
        0.5 + dx + 0.03,
        0.5 + dy + 0.03,
      );
      const unshifted = meanOver(
        image,
        0.5 - dx - 0.03,
        0.5 - dy - 0.03,
        0.5 - dx + 0.03,
        0.5 - dy + 0.03,
      );
      expect(shifted).toBeGreaterThan(unshifted + 0.3);
      caustics.destroy();
      field.destroy();
    });
  });
});

describe('caustics robustness', () => {
  it.each(METHODS)(
    'never goes negative or NaN on a violently stirred surface (%s)',
    async (method) => {
      const device = await getTestDevice();
      await expectValid(device, async () => {
        const field = createField(device, 96);
        for (let i = 0; i < 24; i++) {
          field.addDrop(
            0.1 + 0.8 * ((i * 0.618) % 1),
            0.1 + 0.8 * ((i * 0.377) % 1),
            0.05 + 0.02 * (i % 4),
            i % 2 ? 4 : -4,
          );
        }
        await run(device, (e) => field.advance(e, 30));
        const caustics = createHeightfieldCaustics({
          device,
          heightfield: field,
          resolution: 192,
          method,
        });
        for (const settings of [
          { lightDir: OVERHEAD, depth: 0.5, strength: 4 },
          { lightDir: lightDirection(0.15, 2), depth: 2, strength: 8, dispersion: 0.04, smooth: 1 },
          { lightDir: lightDirection(1.4, 0), depth: 0.01, strength: 0 },
          {
            lightDir: [0.3, 0.2, -0.5] as [number, number, number],
            depth: 1,
            strength: 30,
            extend: false,
          },
          {
            lightDir: [Number.NaN, 0, 1] as [number, number, number],
            depth: Number.NaN,
            strength: 2,
          },
        ] satisfies CausticsSettings[]) {
          await run(device, (e) => caustics.update(e, settings));
          const image = await read(device, caustics);
          for (const channel of [image.r, image.g, image.b, image.a]) {
            for (const v of channel) {
              expect(Number.isFinite(v)).toBe(true);
              expect(v).toBeGreaterThanOrEqual(0);
            }
          }
          // The cap is per layer of light; a surface folded over itself adds layers.
          expect(Math.max(...image.g)).toBeLessThan(200);
        }
        caustics.destroy();
        field.destroy();
      });
    },
  );

  it('gives the channels different light when dispersion is on and the same when off', async () => {
    const device = await getTestDevice();
    await expectValid(device, async () => {
      for (const method of METHODS) {
        const field = createField(device, 96);
        field.addDrop(0.5, 0.5, 0.15, 1.6);
        await run(device, (e) => field.advance(e, 0));
        const caustics = createHeightfieldCaustics({
          device,
          heightfield: field,
          resolution: 128,
          method,
        });
        await run(device, (e) => caustics.update(e, { lightDir: OVERHEAD, depth: 0.5 }));
        const grey = await read(device, caustics);
        expect(grey.r).toEqual(grey.g);
        expect(grey.b).toEqual(grey.g);
        await run(device, (e) =>
          caustics.update(e, { lightDir: OVERHEAD, depth: 0.5, dispersion: 0.05 }),
        );
        const split = await read(device, caustics);
        let differing = 0;
        for (let i = 0; i < split.r.length; i++)
          if (Math.abs(split.r[i]! - split.b[i]!) > 0.1) differing++;
        expect(differing).toBeGreaterThan(20);
        // Each channel still carries the same light on average.
        for (const c of [split.r, split.g, split.b]) {
          const mean = c.reduce((s, v) => s + v, 0) / c.length;
          expect(mean).toBeGreaterThan(0.9);
          expect(mean).toBeLessThan(1.1);
        }
        caustics.destroy();
        field.destroy();
      }
    });
  });
});

describe('the occluder', () => {
  it.each(METHODS)(
    'darkens the floor where the beam is blocked and nowhere else (%s)',
    async (method) => {
      const device = await getTestDevice();
      await expectValid(device, async () => {
        const field = createField(device, 96);
        const caustics = createHeightfieldCaustics({
          device,
          heightfield: field,
          resolution: 192,
          method,
        });
        const depth = 0.5;
        const lightDir = lightDirection(0.9, 0.3);
        // A ball under the surface: centre 0.12 down, radius 0.09.
        const center: [number, number, number] = [0.55, 0.45, -0.12];
        await run(device, (e) =>
          caustics.update(e, { lightDir, depth, occluder: { center, radius: 0.09 } }),
        );
        const image = await read(device, caustics);
        // The beam through the ball's centre (0.12 under the surface) travels the rest of the depth.
        const [ux, uy] = flatLandingShift(lightDir, depth + center[2]);
        const sxu = center[0] + ux;
        const syu = center[1] + uy;
        // Dark at the predicted spot.
        expect(meanOver(image, sxu - 0.02, syu - 0.02, sxu + 0.02, syu + 0.02)).toBeLessThan(0.15);
        // Bright where the shadow would be if the light came straight down (it does not).
        expect(
          meanOver(image, center[0] - 0.02, center[1] - 0.02, center[0] + 0.02, center[1] + 0.02),
        ).toBeGreaterThan(0.8);
        // Away from both, untouched; the whole floor loses about the shadow's area.
        expect(meanOver(image, 0.05, 0.6, 0.35, 0.95)).toBeGreaterThan(0.97);
        const { mean } = stats(image);
        const area = Math.PI * 0.09 ** 2;
        expect(mean).toBeGreaterThan(1 - area * 2.2);
        expect(mean).toBeLessThan(1 - area * 0.5);
        caustics.destroy();
        field.destroy();
      });
    },
  );
});

describe('the two methods', () => {
  it('agree on the pattern of a gentle wave field', async () => {
    const device = await getTestDevice();
    await expectValid(device, async () => {
      const field = createField(device, 128);
      const drops: [number, number, number][] = [
        [0.3, 0.35, 1],
        [0.65, 0.6, -1],
        [0.5, 0.2, 0.8],
      ];
      for (const [x, y, s] of drops) field.addDrop(x, y, 0.06, s);
      await run(device, (e) => field.advance(e, 45));
      const settings: CausticsSettings = {
        lightDir: lightDirection(1.1, 0.5),
        depth: 0.35,
        strength: 1.4,
      };
      const mesh = createHeightfieldCaustics({
        device,
        heightfield: field,
        resolution: 256,
        method: 'mesh',
      });
      const jac = createHeightfieldCaustics({
        device,
        heightfield: field,
        resolution: 256,
        method: 'jacobian',
      });
      await run(device, (e) => {
        mesh.update(e, { ...settings, smooth: 1.2 });
        jac.update(e, { ...settings, smooth: 0.8 });
      });
      const a = await read(device, mesh);
      const b = await read(device, jac);
      const sa = stats(a);
      const sb = stats(b);
      expect(sa.mean).toBeGreaterThan(0.9);
      expect(sa.mean).toBeLessThan(1.1);
      expect(sb.mean).toBeGreaterThan(0.9);
      expect(sb.mean).toBeLessThan(1.1);
      // The pattern has structure for the correlation to mean something.
      expect(sa.max - sa.min).toBeGreaterThan(0.3);
      expect(correlation(a, b)).toBeGreaterThan(0.8);
      mesh.destroy();
      jac.destroy();
      field.destroy();
    });
  });
});

describe('rebuilding', () => {
  it('follows the heightfield when it is re-created, resized or re-gridded', async () => {
    const device = await getTestDevice();
    await expectValid(device, async () => {
      const field = createField(device, 64, 1);
      const caustics = createHeightfieldCaustics({ device, heightfield: field, resolution: 96 });
      expect(caustics.width).toBe(96);
      expect(caustics.height).toBe(96);
      const settings: CausticsSettings = { lightDir: lightDirection(1, 1), depth: 0.4 };
      await run(device, (e) => caustics.update(e, settings));
      const generation = caustics.generation;
      const oldView = caustics.view;

      // The grid is re-created at a new aspect: the texture follows it on the next update.
      expect(field.resize(2, 1)).toBe(true);
      await run(device, (e) => caustics.update(e, settings));
      expect(caustics.generation).toBe(generation + 1);
      expect(caustics.width).toBe(96);
      expect(caustics.height).toBe(48);
      expect(caustics.view).not.toBe(oldView);
      expect(caustics.resize(2, 1)).toBe(false);
      expect(caustics.resize(1, 1)).toBe(true);
      expect(caustics.height).toBe(96);
      expect(caustics.resize(2, 1)).toBe(true);
      expect(caustics.height).toBe(48);

      // Every method runs on the new grid and gives flat light on the flat surface.
      for (const method of METHODS) {
        caustics.method = method;
        await run(device, (e) => caustics.update(e, settings));
        const { mean, min, max } = stats(await read(device, caustics));
        expect(mean).toBeCloseTo(1, 1);
        expect(min).toBeGreaterThan(0.95);
        expect(max).toBeLessThan(1.05);
      }

      expect(caustics.setResolution(160)).toBe(true);
      expect(caustics.width).toBe(160);
      expect(caustics.setResolution(160)).toBe(false);
      caustics.setMeshScale(0.5);
      expect(caustics.meshCells.x).toBeLessThanOrEqual(
        Math.max(field.grid.width, field.grid.height),
      );

      expect(field.setResolution(48)).toBe(true);
      caustics.method = 'mesh';
      await run(device, (e) => caustics.update(e, settings));
      const { mean } = stats(await read(device, caustics));
      expect(mean).toBeCloseTo(1, 1);

      caustics.destroy();
      caustics.destroy();
      field.destroy();
    });
  });

  it('is cheap to update twice in a frame and ignores a destroyed solver', async () => {
    const device = await getTestDevice();
    await expectValid(device, async () => {
      const field = createField(device, 48);
      const caustics = createHeightfieldCaustics({
        device,
        heightfield: field,
        resolution: 64,
        method: 'jacobian',
      });
      await run(device, (e) => {
        caustics.update(e, { lightDir: OVERHEAD, depth: 0.3 });
        caustics.update(e, { lightDir: OVERHEAD, depth: 0.3, smooth: 2 });
      });
      caustics.destroy();
      await run(device, (e) => caustics.update(e, { lightDir: OVERHEAD, depth: 0.3 }));
      field.destroy();
    });
  });
});
