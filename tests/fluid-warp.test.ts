import { describe, expect, it, vi } from 'vitest';
import { createFluid, halfToFloat, type Fluid, type FluidParams } from '../src/gpu/sim/fluid.ts';
import {
  createFluidWarp,
  createFluidWarpPresenter,
  warpFieldSize,
  type FluidWarp,
  type WarpDrawOptions,
} from '../src/gpu/sim/fluid-warp.ts';
import { expectValid, getTestDevice, readTexture } from './gpu.ts';

vi.setConfig({ testTimeout: 120_000 });

// --- Helpers -------------------------------------------------------------------------------------

const SIZE = 128;

async function submit(device: GPUDevice, record: (encoder: GPUCommandEncoder) => void) {
  const encoder = device.createCommandEncoder();
  record(encoder);
  device.queue.submit([encoder.finish()]);
  await device.queue.onSubmittedWorkDone();
}

function createSolver(device: GPUDevice, params: Partial<FluidParams> = {}, aspect = 1): Fluid {
  return createFluid({
    device,
    resolution: 64,
    dyeResolution: 16,
    width: aspect,
    height: 1,
    params: { velocityDissipation: 0, dyeDissipation: 0, dyeScheme: 'semi-lagrangian', ...params },
  });
}

/** A solver running a prescribed flow (nothing but advection of the displacement field). */
function createPrescribed(device: GPUDevice, boundary: 'wrap' | 'wall' | 'open' = 'wrap'): Fluid {
  return createSolver(device, { dyeOnly: true, boundary });
}

/** Linear-light content, one texel per pixel of a SIZE x SIZE canvas. */
function contentTexture(device: GPUDevice, bytes: Uint8Array, size = SIZE) {
  const texture = device.createTexture({
    size: [size, size],
    format: 'rgba8unorm-srgb',
    usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
  });
  device.queue.writeTexture(
    { texture },
    bytes as Uint8Array<ArrayBuffer>,
    { bytesPerRow: size * 4 },
    [size, size],
  );
  return { texture, view: texture.createView(), width: size, height: size };
}

/** A dark ramp with a bright 12 x 12 square whose centre is at (cx, cy) pixels. */
function markerBytes(cx: number, cy: number, size = SIZE): Uint8Array {
  const bytes = new Uint8Array(size * size * 4);
  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      const o = (y * size + x) * 4;
      const inside = Math.abs(x + 0.5 - cx) <= 6 && Math.abs(y + 0.5 - cy) <= 6;
      bytes[o] = inside ? 255 : (20 + (x * 40) / size) | 0;
      bytes[o + 1] = inside ? 255 : 20;
      bytes[o + 2] = inside ? 255 : (20 + (y * 40) / size) | 0;
      bytes[o + 3] = 255;
    }
  }
  return bytes;
}

/** Pseudo-random opaque bytes: every texel different, so any resampling shows. */
function noiseBytes(size = SIZE): Uint8Array {
  const bytes = new Uint8Array(size * size * 4);
  let state = 12345;
  for (let i = 0; i < bytes.length; i++) {
    state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
    bytes[i] = i % 4 === 3 ? 255 : state >>> 24;
  }
  return bytes;
}

async function render(
  device: GPUDevice,
  warp: FluidWarp,
  content: ReturnType<typeof contentTexture>,
  extra: Partial<WarpDrawOptions> = {},
): Promise<Uint8Array> {
  const target = device.createTexture({
    size: [SIZE, SIZE],
    format: 'rgba8unorm',
    usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC,
  });
  const presenter = createFluidWarpPresenter({ device, format: 'rgba8unorm' });
  await submit(device, (encoder) =>
    presenter.draw(encoder, target.createView(), {
      warp,
      content,
      fit: 'stretch',
      canvasSize: [SIZE, SIZE],
      ...extra,
    }),
  );
  const pixels = await readTexture(device, target);
  presenter.destroy();
  target.destroy();
  return pixels;
}

/** Centroid, in pixels, of the pixels brighter than 200 in every channel. */
function markerCentroid(pixels: Uint8Array, size = SIZE): { x: number; y: number; count: number } {
  let sx = 0;
  let sy = 0;
  let count = 0;
  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      const o = (y * size + x) * 4;
      if (pixels[o]! > 200 && pixels[o + 1]! > 200 && pixels[o + 2]! > 200) {
        sx += x + 0.5;
        sy += y + 0.5;
        count++;
      }
    }
  }
  return { x: sx / count, y: sy / count, count };
}

async function readField(device: GPUDevice, warp: FluidWarp) {
  const { width, height, texture } = warp.field;
  const bytes = await readTexture(device, texture, 8);
  const half = new Uint16Array(bytes.buffer, bytes.byteOffset, width * height * 4);
  const out = new Float32Array(width * height * 2);
  for (let i = 0; i < width * height; i++) {
    out[i * 2] = halfToFloat(half[i * 4]!);
    out[i * 2 + 1] = halfToFloat(half[i * 4 + 1]!);
  }
  return {
    width,
    height,
    data: out,
    at: (x: number, y: number): [number, number] => {
      const i =
        Math.min(height - 1, Math.floor(y * height)) * width +
        Math.min(width - 1, Math.floor(x * width));
      return [out[i * 2]!, out[i * 2 + 1]!];
    },
  };
}

function allFinite(data: ArrayLike<number>): boolean {
  for (let i = 0; i < data.length; i++) if (!Number.isFinite(data[i])) return false;
  return true;
}

// --- Tests ----------------------------------------------------------------------------------------

describe('field size', () => {
  it('follows the solver grid and stays inside the limits', () => {
    expect(warpFieldSize({ width: 128, height: 72 }, 2)).toEqual({ width: 256, height: 144 });
    expect(warpFieldSize({ width: 128, height: 72 }, 0.1)).toEqual({ width: 64, height: 36 });
    expect(warpFieldSize({ width: 4000, height: 3000 }, 8, 4096)).toEqual({
      width: 4096,
      height: 4096,
    });
    expect(warpFieldSize({ width: 100, height: 100 }, Number.NaN).width).toBe(200);
  });
});

describe('a field at rest', () => {
  it('renders the source byte for byte', async () => {
    const device = await getTestDevice();
    const fluid = createPrescribed(device);
    const warp = createFluidWarp({ device, fluid });
    const bytes = noiseBytes();
    const content = contentTexture(device, bytes);
    const pixels = await render(device, warp, content);
    let worst = 0;
    for (let i = 0; i < bytes.length; i++)
      worst = Math.max(worst, Math.abs(pixels[i]! - bytes[i]!));
    // sRGB decode in the sampler and encode in the shader: exact up to a rounding of one level.
    expect(worst).toBeLessThanOrEqual(1);
    // With every optional look switched on, rest is still rest.
    const styled = await render(device, warp, content, { chroma: 0.5, sheen: 1, cubic: true });
    let styledWorst = 0;
    for (let i = 0; i < bytes.length; i++) {
      styledWorst = Math.max(styledWorst, Math.abs(styled[i]! - bytes[i]!));
    }
    expect(styledWorst).toBeLessThanOrEqual(1);
    fluid.destroy();
    warp.destroy();
  });

  it('is still the source after the fluid ran with nothing stirred', async () => {
    const device = await getTestDevice();
    const fluid = createSolver(device);
    const warp = createFluidWarp({ device, fluid });
    warp.attach();
    await submit(device, (e) => fluid.advance(e, 20));
    const bytes = noiseBytes();
    const pixels = await render(device, warp, contentTexture(device, bytes), { sheen: 1 });
    for (let i = 0; i < bytes.length; i++) {
      expect(Math.abs(pixels[i]! - bytes[i]!)).toBeLessThanOrEqual(1);
    }
    fluid.destroy();
    warp.destroy();
  });
});

describe('a uniform flow', () => {
  it.each([
    { name: 'to the right', v: [0.25, 0] as const },
    { name: 'up and to the right', v: [0.2, -0.15] as const },
    { name: 'down and to the left', v: [-0.18, 0.12] as const },
  ])('moves the content by v t in the direction of v ($name)', async ({ v }) => {
    const device = await getTestDevice();
    const fluid = createPrescribed(device);
    const warp = createFluidWarp({ device, fluid, params: { relaxation: 0 } });
    warp.attach();
    fluid.setVelocity(() => [v[0], v[1]]);
    const steps = 24;
    const seconds = steps * fluid.stepSeconds;
    await submit(device, (e) => fluid.advance(e, steps));

    // The field itself: D = v t everywhere (image heights; the canvas is square).
    const field = await readField(device, warp);
    for (const [x, y] of [
      [0.3, 0.4],
      [0.5, 0.5],
      [0.7, 0.6],
    ] as const) {
      const d = field.at(x, y);
      expect(d[0]).toBeCloseTo(v[0] * seconds, 2);
      expect(d[1]).toBeCloseTo(v[1] * seconds, 2);
    }

    // And the picture: the marker moved with the flow, by v t, to within a fraction of a pixel.
    const x0 = 50;
    const y0 = 64;
    const pixels = await render(device, warp, contentTexture(device, markerBytes(x0, y0)));
    const found = markerCentroid(pixels);
    expect(found.count).toBeGreaterThan(100);
    expect(found.x).toBeCloseTo(x0 + v[0] * seconds * SIZE, 0);
    expect(found.y).toBeCloseTo(y0 + v[1] * seconds * SIZE, 0);
    fluid.destroy();
    warp.destroy();
  });

  it('scales with the gain', async () => {
    const device = await getTestDevice();
    const fluid = createPrescribed(device);
    const warp = createFluidWarp({ device, fluid, params: { relaxation: 0, gain: 0.5 } });
    warp.attach();
    fluid.setVelocity(() => [0.3, 0]);
    await submit(device, (e) => fluid.advance(e, 30));
    const field = await readField(device, warp);
    expect(field.at(0.5, 0.5)[0]).toBeCloseTo(0.3 * 0.5 * 0.5, 2);
    fluid.destroy();
    warp.destroy();
  });
});

describe('advection of the field', () => {
  it('carries the displacement round a solid-body rotation (|D| = 2 r sin(theta / 2))', async () => {
    const device = await getTestDevice();
    const fluid = createPrescribed(device, 'wall');
    const warp = createFluidWarp({ device, fluid, params: { relaxation: 0 } });
    warp.attach();
    const omega = 1.2;
    // Clockwise on screen with y down: v = omega (-(y - 0.5), x - 0.5), inside a disc.
    fluid.setVelocity((x, y) => {
      const rx = x - 0.5;
      const ry = y - 0.5;
      const r = Math.hypot(rx, ry);
      const taper = 1 - Math.min(1, Math.max(0, (r - 0.38) / 0.06));
      return [-omega * ry * taper, omega * rx * taper];
    });
    const steps = 50;
    await submit(device, (e) => fluid.advance(e, steps));
    const theta = omega * steps * fluid.stepSeconds;
    const field = await readField(device, warp);
    for (const angle of [0.3, 2, 4]) {
      const r = 0.25;
      const px = 0.5 + r * Math.cos(angle);
      const py = 0.5 + r * Math.sin(angle);
      const d = field.at(px, py);
      // The parcel at p was at the point rotated back by theta; D = p - that point.
      const rx = r * Math.cos(angle);
      const ry = r * Math.sin(angle);
      const back = [
        rx * Math.cos(theta) + ry * Math.sin(theta),
        -rx * Math.sin(theta) + ry * Math.cos(theta),
      ];
      const expected = [rx - back[0]!, ry - back[1]!];
      expect(Math.hypot(d[0], d[1])).toBeGreaterThan(0.9 * 2 * r * Math.sin(theta / 2));
      expect(Math.hypot(d[0], d[1])).toBeLessThan(1.05 * 2 * r * Math.sin(theta / 2));
      expect(Math.hypot(d[0] - expected[0]!, d[1] - expected[1]!)).toBeLessThan(0.03);
    }
    fluid.destroy();
    warp.destroy();
  });
});

describe('relaxation', () => {
  it.each([0.7, 2.5])('returns the field to zero at rate %s per second', async (rate) => {
    const device = await getTestDevice();
    const fluid = createPrescribed(device);
    const warp = createFluidWarp({ device, fluid, params: { relaxation: 0 } });
    warp.attach();
    fluid.setVelocity(() => [0.3, 0]);
    await submit(device, (e) => fluid.advance(e, 30));
    const before = (await readField(device, warp)).at(0.5, 0.5)[0];
    expect(before).toBeCloseTo(0.15, 2);

    // Stop the flow; the displacement must decay by exp(-rate t) and nothing else.
    fluid.setVelocity(() => [0, 0]);
    warp.params.relaxation = rate;
    const steps = 36;
    await submit(device, (e) => fluid.advance(e, steps));
    const after = (await readField(device, warp)).at(0.5, 0.5)[0];
    const expected = before * Math.exp(-rate * steps * fluid.stepSeconds);
    expect(Math.abs(after - expected)).toBeLessThan(0.02 * before + 2e-4);

    // Long enough and it is gone, so the content is back (permanent smears are relaxation 0).
    await submit(device, (e) => fluid.advance(e, 600));
    const gone = (await readField(device, warp)).at(0.5, 0.5)[0];
    expect(Math.abs(gone)).toBeLessThan(2e-3);
    fluid.destroy();
    warp.destroy();
  });

  it('keeps a smear for ever at zero (paint)', async () => {
    const device = await getTestDevice();
    const fluid = createPrescribed(device);
    const warp = createFluidWarp({ device, fluid, params: { relaxation: 0 } });
    warp.attach();
    fluid.setVelocity(() => [0.3, 0]);
    await submit(device, (e) => fluid.advance(e, 30));
    fluid.setVelocity(() => [0, 0]);
    await submit(device, (e) => fluid.advance(e, 600));
    expect((await readField(device, warp)).at(0.5, 0.5)[0]).toBeCloseTo(0.15, 2);
    fluid.destroy();
    warp.destroy();
  });
});

describe('bounds', () => {
  it('keeps the displacement within the clamp under violent stirring, finite throughout', async () => {
    const device = await getTestDevice();
    const fluid = createSolver(device, { boundary: 'wall', vorticity: 10 });
    const limit = 0.12;
    const warp = createFluidWarp({
      device,
      fluid,
      params: { relaxation: 0, maxDisplacement: limit, gain: 4 },
    });
    warp.attach();
    let peak = 0;
    for (let k = 0; k < 12; k++) {
      for (let s = 0; s < 40; s++) {
        const a = (k * 40 + s) * 0.7;
        fluid.addForce(
          [0.5 + 0.4 * Math.cos(a), 0.5 + 0.4 * Math.sin(a)],
          [0.5 - 0.4 * Math.cos(a), 0.5 - 0.4 * Math.sin(a)],
          0.08,
          [3 * Math.cos(a * 1.3), 3 * Math.sin(a * 1.3)],
        );
        if (s % 8 === 7) break;
      }
      fluid.addForce([0.5, 0.5], [0.9, 0.5], 0.2, [3e38, -3e38]);
      await submit(device, (e) => fluid.advance(e, 10));
      const field = await readField(device, warp);
      expect(allFinite(field.data)).toBe(true);
      for (let i = 0; i < field.width * field.height; i++) {
        peak = Math.max(peak, Math.hypot(field.data[i * 2]!, field.data[i * 2 + 1]!));
      }
    }
    // The clamp is on the length of the vector; half floats round it by a part in a thousand.
    expect(peak).toBeLessThanOrEqual(limit * 1.003);
    expect(peak).toBeGreaterThan(limit * 0.9);

    // And the picture is still a picture.
    const pixels = await render(device, warp, contentTexture(device, noiseBytes()), {
      chroma: 0.4,
      sheen: 1,
    });
    expect(new Set(pixels).size).toBeGreaterThan(100);
    fluid.destroy();
    warp.destroy();
  });

  it('stays bounded in a long, gentle stir with relaxation, and never goes NaN', async () => {
    const device = await getTestDevice();
    const fluid = createSolver(device, { boundary: 'open', velocityDissipation: 0.3 });
    const warp = createFluidWarp({
      device,
      fluid,
      params: { relaxation: 1.5, maxDisplacement: 0.3 },
    });
    warp.attach();
    for (let k = 0; k < 20; k++) {
      fluid.addForce([0.2, 0.5], [0.8, 0.5], 0.1, [1.5 * (k % 2 ? 1 : -1), 0.4]);
      await submit(device, (e) => fluid.advance(e, 8));
    }
    const field = await readField(device, warp);
    expect(allFinite(field.data)).toBe(true);
    let peak = 0;
    for (let i = 0; i < field.width * field.height; i++) {
      peak = Math.max(peak, Math.hypot(field.data[i * 2]!, field.data[i * 2 + 1]!));
    }
    expect(peak).toBeGreaterThan(0.01);
    expect(peak).toBeLessThanOrEqual(0.3 * 1.003);
    fluid.destroy();
    warp.destroy();
  });
});

describe('borders', () => {
  it('never samples outside the content: a big displacement leaves no smeared or wrapped border', async () => {
    const device = await getTestDevice();
    const fluid = createPrescribed(device, 'open');
    const warp = createFluidWarp({
      device,
      fluid,
      params: { relaxation: 0, maxDisplacement: 0.5 },
    });
    warp.attach();
    fluid.setVelocity(() => [0.9, 0]);
    await submit(device, (e) => fluid.advance(e, 30));
    // Content: red on the left edge column band, blue elsewhere. Nothing may wrap round to the right.
    const bytes = new Uint8Array(SIZE * SIZE * 4);
    for (let y = 0; y < SIZE; y++) {
      for (let x = 0; x < SIZE; x++) {
        const o = (y * SIZE + x) * 4;
        const left = x < 8;
        bytes[o] = left ? 255 : 0;
        bytes[o + 1] = 0;
        bytes[o + 2] = left ? 0 : 255;
        bytes[o + 3] = 255;
      }
    }
    const pixels = await render(device, warp, contentTexture(device, bytes));
    // The content is displaced right by ~0.45: the far-right columns now show the content that
    // started at x ~ 0.55..0.55 (blue), and no red wrapped onto them.
    for (let y = 0; y < SIZE; y += 8) {
      const o = (y * SIZE + (SIZE - 1)) * 4;
      expect(pixels[o]).toBeLessThan(40);
    }
    fluid.destroy();
    warp.destroy();
  });
});

describe('re-creation', () => {
  it('rebuilds the field and the bind groups when the solver grid changes', async () => {
    const device = await getTestDevice();
    await expectValid(device, async () => {
      const fluid = createSolver(device, { boundary: 'wall' });
      const warp = createFluidWarp({ device, fluid });
      warp.attach();
      const presenter = createFluidWarpPresenter({ device, format: 'rgba8unorm' });
      const content = contentTexture(device, noiseBytes());
      const target = device.createTexture({
        size: [SIZE, SIZE],
        format: 'rgba8unorm',
        usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC,
      });
      const draw = (encoder: GPUCommandEncoder) =>
        presenter.draw(encoder, target.createView(), {
          warp,
          content,
          fit: 'cover',
          canvasSize: [SIZE, SIZE],
          chroma: 0.2,
          fields: fluid.fieldsView,
          view: 'velocity',
        });

      fluid.addForce([0.2, 0.5], [0.8, 0.5], 0.1, [2, 0]);
      await submit(device, (e) => {
        fluid.update(e, 1 / 15);
        draw(e);
      });
      const firstGeneration = warp.generation;
      const first = await readField(device, warp);
      expect(first.width).toBe(fluid.grid.width * 2);
      expect(Math.max(...first.data.map(Math.abs))).toBeGreaterThan(0.001);

      // A different shape: the solver re-creates its grids (and clears), the field follows.
      expect(fluid.resize(2, 1)).toBe(true);
      await submit(device, (e) => {
        fluid.update(e, 1 / 60);
        draw(e);
      });
      expect(warp.generation).toBeGreaterThan(firstGeneration);
      const second = await readField(device, warp);
      expect(second.width).toBe(fluid.grid.width * 2);
      expect(second.height).toBe(fluid.grid.height * 2);
      expect(Math.max(...second.data.map(Math.abs))).toBe(0);

      // A different scale re-creates only the field.
      warp.setScale(1);
      await submit(device, (e) => {
        fluid.update(e, 1 / 60);
        draw(e);
      });
      expect(warp.field.width).toBe(fluid.grid.width);

      // Detaching stops the field; the previous (absent) hook is back.
      warp.detach();
      expect(fluid.hooks.dye).toBeUndefined();
      presenter.destroy();
      target.destroy();
      warp.destroy();
      fluid.destroy();
    });
  });

  it('chains with a hook that was already installed, and puts it back on detach', async () => {
    const device = await getTestDevice();
    const fluid = createSolver(device);
    const calls: string[] = [];
    const own = () => {
      calls.push('own');
      return false;
    };
    fluid.hooks.dye = own;
    const warp = createFluidWarp({ device, fluid });
    warp.attach();
    expect(fluid.hooks.dye).not.toBe(own);
    await submit(device, (e) => fluid.advance(e, 3));
    expect(calls).toEqual(['own', 'own', 'own']);
    warp.detach();
    expect(fluid.hooks.dye).toBe(own);
    fluid.destroy();
    warp.destroy();
  });
});

describe('views', () => {
  it.each(['content', 'displacement', 'grid', 'velocity'] as const)(
    'draws the %s view without a validation error',
    async (view) => {
      const device = await getTestDevice();
      const fluid = createSolver(device);
      const warp = createFluidWarp({ device, fluid });
      warp.attach();
      fluid.addForce([0.3, 0.5], [0.7, 0.5], 0.1, [2, 0.5]);
      await submit(device, (e) => fluid.advance(e, 10));
      const pixels = await expectValid(device, () =>
        render(device, warp, contentTexture(device, markerBytes(64, 64)), {
          view,
          fields: fluid.fieldsView,
          sheen: 0.6,
          chroma: 0.3,
        }),
      );
      expect(new Set(pixels).size).toBeGreaterThan(4);
      fluid.destroy();
      warp.destroy();
    },
  );
});
