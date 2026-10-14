import { describe, expect, it, vi } from 'vitest';
import {
  createLiquid2d,
  DEFAULT_LIQUID2D_PARAMS,
  liquid2dKernelK,
  liquid2dLoneRadius,
  liquid2dMergeDistance,
  type Liquid2d,
  type Liquid2dParams,
} from '../src/gpu/liquid-2d/index.ts';
import type { ParticleSet } from '../src/gpu/particles/types.ts';
import { createParticleSystem } from '../src/gpu/particles/system.ts';
import { expectValid, getTestDevice, readTexture } from './gpu.ts';

// The 2D liquid-surface renderer on the real GPU, against independent CPU maths: where a lone particle's
// blob ends, when two blobs merge, what the field integrates to, and the plumbing around them.
//
// The scene is 128 x 96 px showing the world rectangle x in [-1, 1], y in [-0.75, 0.75] (a pixel is 1/64
// world units, x right, y up). The shape tests switch off everything that makes the alpha channel anything
// other than the antialiased coverage of the iso-line (rim light, highlight, foam, thin-edge opacity).

vi.setConfig({ testTimeout: 60_000 });

const W = 128;
const H = 96;
const PX = 64; // pixels per world unit
const DOMAIN = { center: [0, 0] as const, halfHeight: 0.75 };

/** World to continuous pixel coordinates (pixel i spans [i, i + 1]). */
const toPx = (x: number, y: number): [number, number] => [(x + 1) * PX, (0.75 - y) * PX];

/** Plain coverage: alpha is the antialiased iso-line coverage and nothing else. */
const SHAPE: Partial<Liquid2dParams> = {
  smoothing: 0,
  rim: 0,
  highlight: 0,
  foam: 0,
  opacity: 1,
  thinOpacity: 1,
};

const RADIUS = 0.12;
const BLOB = DEFAULT_LIQUID2D_PARAMS.blobRadius;
const THRESHOLD = DEFAULT_LIQUID2D_PARAMS.threshold;

interface Scene {
  device: GPUDevice;
  liquid: Liquid2d;
  target: GPUTexture;
}

async function createScene(width = W, height = H, fieldScale = 1): Promise<Scene> {
  const device = await getTestDevice();
  const liquid = await expectValid(device, () =>
    createLiquid2d({ device, format: 'rgba8unorm', width, height, fieldScale }),
  );
  const target = device.createTexture({
    size: [width, height],
    format: 'rgba8unorm',
    usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC,
  });
  return { device, liquid, target };
}

interface Rows {
  /** x, y, w per particle. */
  list: readonly (readonly [number, number, number?])[];
  radius?: number;
  restVolume?: number;
  /** Buffer size when larger than the list (the tail stays zero unless `tail` fills it). */
  capacity?: number;
  /** Live count when smaller than the list. */
  count?: number;
}

function makeSet(device: GPUDevice, rows: Rows) {
  const capacity = rows.capacity ?? Math.max(rows.list.length, 1);
  const system = createParticleSystem({
    device,
    capacity,
    radius: rows.radius ?? RADIUS,
    dimension: 2,
  });
  const data = new Float32Array(rows.list.length * 4);
  rows.list.forEach(([x, y, w], i) => data.set([x, y, 0, w ?? 0], i * 4));
  system.set({ positions: data });
  if (rows.count !== undefined) system.setCount(rows.count);
  const set: ParticleSet = {
    positions: system.positions,
    count: system.count,
    capacity: system.capacity,
    radius: system.radius,
    restVolume: rows.restVolume,
    generation: system.generation,
  };
  return { system, set };
}

function constantTexture(device: GPUDevice, colour: GPUColor, width = W, height = H) {
  const texture = device.createTexture({
    size: [width, height],
    format: 'rgba16float',
    usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
  });
  const encoder = device.createCommandEncoder();
  encoder
    .beginRenderPass({
      colorAttachments: [
        { view: texture.createView(), loadOp: 'clear', storeOp: 'store', clearValue: colour },
      ],
    })
    .end();
  device.queue.submit([encoder.finish()]);
  return texture;
}

interface DrawOptions {
  params?: Partial<Liquid2dParams>;
  backdrop?: GPUTextureView;
  load?: boolean;
}

/** Renders `set` and reads the rgba8 target back. */
async function draw(scene: Scene, set: ParticleSet, options: DrawOptions = {}) {
  const { device, liquid, target } = scene;
  await expectValid(device, () => {
    const encoder = device.createCommandEncoder();
    liquid.render(encoder, {
      particles: set,
      domain: DOMAIN,
      target: target.createView(),
      params: options.params,
      backdrop: options.backdrop,
      load: options.load,
    });
    device.queue.submit([encoder.finish()]);
  });
  return readTexture(device, target);
}

function halfValue(bits: number): number {
  const sign = bits & 0x8000 ? -1 : 1;
  const exponent = (bits >> 10) & 0x1f;
  const mantissa = bits & 0x3ff;
  if (exponent === 0) return sign * 2 ** -14 * (mantissa / 1024);
  if (exponent === 31) return mantissa ? NaN : sign * Infinity;
  return sign * 2 ** (exponent - 15) * (1 + mantissa / 1024);
}

/** The field as floats: four numbers per texel. */
async function readField(scene: Scene): Promise<Float32Array> {
  const { texture } = scene.liquid.field;
  const bits = new Uint16Array((await readTexture(scene.device, texture, 8)).buffer);
  return Float32Array.from(bits, halfValue);
}

const alphaOf = (pixels: Uint8Array) =>
  Float32Array.from({ length: W * H }, (_, i) => pixels[i * 4 + 3]! / 255);

function sumOf(values: ArrayLike<number>): number {
  let total = 0;
  for (let i = 0; i < values.length; i++) total += values[i]!;
  return total;
}

/** Centroid of an alpha image in continuous pixel coordinates. */
function centroid(alpha: Float32Array): [number, number] {
  let sum = 0;
  let sx = 0;
  let sy = 0;
  for (let y = 0; y < H; y++)
    for (let x = 0; x < W; x++) {
      const a = alpha[y * W + x]!;
      sum += a;
      sx += a * (x + 0.5);
      sy += a * (y + 0.5);
    }
  return [sx / sum, sy / sum];
}

/** Connected regions of alpha > 0.5, by flood fill with 4-neighbours. */
function regions(alpha: Float32Array): number {
  const seen = new Uint8Array(W * H);
  let count = 0;
  for (let start = 0; start < W * H; start++) {
    if (seen[start] || alpha[start]! <= 0.5) continue;
    count++;
    const stack = [start];
    seen[start] = 1;
    while (stack.length) {
      const i = stack.pop()!;
      const x = i % W;
      const y = (i / W) | 0;
      const neighbours = [
        x > 0 ? i - 1 : -1,
        x < W - 1 ? i + 1 : -1,
        y > 0 ? i - W : -1,
        y < H - 1 ? i + W : -1,
      ];
      for (const n of neighbours) {
        if (n >= 0 && !seen[n] && alpha[n]! > 0.5) {
          seen[n] = 1;
          stack.push(n);
        }
      }
    }
  }
  return count;
}

const srgb = (c: number) => (c <= 0.0031308 ? 12.92 * c : 1.055 * c ** (1 / 2.4) - 0.055);

// Independent of the exported helpers: the formulas of the header comment of liquid-2d/index.ts.
function kernelK(restVolume: number, big: number) {
  return (4 * restVolume) / (Math.PI * big * big);
}

describe('liquid-2d kernel', () => {
  it('normalises each blob to integrate to restVolume and derives radius and merge distance', () => {
    const big = BLOB * RADIUS;
    const restVolume = (2 * RADIUS) ** 2;
    const k = kernelK(restVolume, big);
    // Numerically integrate k (1 - q^2)^3 over the disc of radius R.
    let integral = 0;
    const steps = 20000;
    for (let i = 0; i < steps; i++) {
      const r = ((i + 0.5) / steps) * big;
      integral += k * (1 - (r / big) ** 2) ** 3 * 2 * Math.PI * r * (big / steps);
    }
    expect(integral).toBeCloseTo(restVolume, 4);
    expect(liquid2dKernelK(restVolume, big)).toBeCloseTo(k, 12);

    // F(lone radius) = threshold; the saddle at the merge distance = threshold.
    const lone = liquid2dLoneRadius(RADIUS, restVolume, BLOB, THRESHOLD);
    expect(k * (1 - (lone / big) ** 2) ** 3).toBeCloseTo(THRESHOLD, 9);
    const merge = liquid2dMergeDistance(RADIUS, restVolume, BLOB, THRESHOLD);
    expect(2 * k * (1 - (merge / 2 / big) ** 2) ** 3).toBeCloseTo(THRESHOLD, 9);
    // A blob whose peak is under the threshold never shows.
    expect(liquid2dLoneRadius(RADIUS, restVolume * 0.1, BLOB, THRESHOLD)).toBe(0);
  });
});

describe('liquid-2d shape', () => {
  it('draws a lone particle as a round disc of the derived radius, centred on its projection', async () => {
    const scene = await createScene();
    const [wx, wy] = [0.173, -0.081];
    const { system, set } = makeSet(scene.device, { list: [[wx, wy]] });
    const alpha = alphaOf(await draw(scene, set, { params: SHAPE }));

    const restVolume = (2 * RADIUS) ** 2;
    const big = BLOB * RADIUS;
    const k = kernelK(restVolume, big);
    const expected = big * Math.sqrt(1 - Math.cbrt(THRESHOLD / k)) * PX; // pixels
    expect(expected).toBeGreaterThan(5); // a meaningful test: several pixels across

    // Area and radius.
    const area = sumOf(alpha);
    expect(Math.sqrt(area / Math.PI)).toBeCloseTo(expected, 0);
    expect(Math.abs(Math.sqrt(area / Math.PI) / expected - 1)).toBeLessThan(0.025);

    // Position: centroid of coverage within a fraction of a pixel of the projection.
    const [cx, cy] = centroid(alpha);
    const [px, py] = toPx(wx, wy);
    expect(Math.abs(cx - px)).toBeLessThan(0.1);
    expect(Math.abs(cy - py)).toBeLessThan(0.1);

    // Roundness: the 0.5 crossing along eight rays from the true centre is at the expected radius
    // (bilinear lookup of the coverage image), and the second moments are isotropic.
    const bilinear = (x: number, y: number) => {
      const fx = x - 0.5;
      const fy = y - 0.5;
      const x0 = Math.floor(fx);
      const y0 = Math.floor(fy);
      const tx = fx - x0;
      const ty = fy - y0;
      const at = (i: number, j: number) => alpha[j * W + i]!;
      return (
        at(x0, y0) * (1 - tx) * (1 - ty) +
        at(x0 + 1, y0) * tx * (1 - ty) +
        at(x0, y0 + 1) * (1 - tx) * ty +
        at(x0 + 1, y0 + 1) * tx * ty
      );
    };
    for (let direction = 0; direction < 8; direction++) {
      const angle = ((direction + 0.3) / 8) * 2 * Math.PI;
      let t = 0;
      while (bilinear(px + t * Math.cos(angle), py + t * Math.sin(angle)) >= 0.5) t += 0.02;
      expect(Math.abs(t - expected)).toBeLessThan(0.3);
    }
    let ixx = 0;
    let iyy = 0;
    let ixy = 0;
    for (let y = 0; y < H; y++)
      for (let x = 0; x < W; x++) {
        const a = alpha[y * W + x]!;
        ixx += a * (x + 0.5 - cx) ** 2;
        iyy += a * (y + 0.5 - cy) ** 2;
        ixy += a * (x + 0.5 - cx) * (y + 0.5 - cy);
      }
    expect(Math.abs(ixx - iyy) / (ixx + iyy)).toBeLessThan(0.02);
    expect(Math.abs(ixy) / (ixx + iyy)).toBeLessThan(0.02);
    expect(Math.abs((ixx + iyy) / (area * expected * expected * 0.5) - 1)).toBeLessThan(0.05);

    // Nothing beyond the radius (plus the antialiasing ramp).
    for (let y = 0; y < H; y++)
      for (let x = 0; x < W; x++) {
        if (Math.hypot(x + 0.5 - px, y + 0.5 - py) > expected + 1.5) {
          expect(alpha[y * W + x]).toBe(0);
        }
      }
    system.destroy();
    scene.liquid.destroy();
  });

  it('renders two particles as one region inside the merge distance and as two outside it', async () => {
    const scene = await createScene();
    const restVolume = (2 * RADIUS) ** 2;
    const merge = liquid2dMergeDistance(RADIUS, restVolume, BLOB, THRESHOLD);
    // Independent derivation: saddle 2 k (1 - (d / 2R)^2)^3 = threshold.
    const big = BLOB * RADIUS;
    const k = kernelK(restVolume, big);
    const d0 = 2 * big * Math.sqrt(1 - Math.cbrt(THRESHOLD / (2 * k)));
    expect(merge).toBeCloseTo(d0, 9);

    const counts: number[] = [];
    for (const factor of [0.94, 1.06]) {
      const d = factor * d0;
      // The midpoint sits on a pixel centre so the saddle is sampled exactly.
      const mid = [63.5 / PX - 1, 0.75 - 47.5 / PX] as const;
      const { system, set } = makeSet(scene.device, {
        list: [
          [mid[0] - d / 2, mid[1]],
          [mid[0] + d / 2, mid[1]],
        ],
      });
      counts.push(regions(alphaOf(await draw(scene, set, { params: SHAPE }))));
      system.destroy();
    }
    expect(counts).toEqual([1, 2]);
    scene.liquid.destroy();
  });

  it('keeps its field at the integral N * restVolume, whatever the field resolution', async () => {
    for (const [fieldScale, restVolume] of [
      [0.5, undefined],
      [1, 0.01],
    ] as const) {
      const scene = await createScene(W, H, fieldScale);
      const radius = 0.04;
      let seed = 7;
      const random = () => (seed = (seed * 1664525 + 1013904223) >>> 0) / 4294967296;
      // Far enough from the border that no blob is cut off by the domain.
      const list = Array.from(
        { length: 60 },
        () => [(random() - 0.5) * 1.2, (random() - 0.5) * 0.6] as const,
      );
      const { system, set } = makeSet(scene.device, { list, radius, restVolume });
      await draw(scene, set, { params: SHAPE });
      const field = await readField(scene);
      const { width, height } = scene.liquid.field;
      expect(width).toBe(Math.round(W * fieldScale));
      expect(height).toBe(Math.round(H * fieldScale));
      let integral = 0;
      for (let i = 0; i < width * height; i++) integral += field[i * 4]!;
      integral *= (2 * 1.5) / (width * height); // world area of a texel: 2.0 x 1.5 over the field
      const expected = list.length * (restVolume ?? (2 * radius) ** 2);
      expect(Math.abs(integral / expected - 1)).toBeLessThan(0.02);
      system.destroy();
      scene.liquid.destroy();
    }
  });
});

describe('liquid-2d output', () => {
  const lattice = (spacing: number, nx: number, ny: number) =>
    Array.from({ length: nx * ny }, (_, i) => {
      const x = i % nx;
      const y = Math.floor(i / nx);
      return [
        (x - nx / 2) * spacing + (y % 2) * 0.5 * spacing,
        -0.1 + y * spacing * 0.866,
        0.5,
      ] as const;
    });

  it('writes premultiplied alpha in [0, 1] with rgb <= alpha and partly covered pixels', async () => {
    const scene = await createScene();
    const radius = 0.03;
    const { system, set } = makeSet(scene.device, {
      list: lattice(2 * radius, 30, 12),
      radius,
      restVolume: (2 * radius) ** 2,
    });
    const pixels = await draw(scene, set);
    let partial = 0;
    let covered = 0;
    for (let i = 0; i < W * H; i++) {
      const a = pixels[i * 4 + 3]!;
      for (let c = 0; c < 3; c++) expect(pixels[i * 4 + c]!).toBeLessThanOrEqual(a + 1);
      if (a > 0 && a < 255) partial++;
      if (a > 0) covered++;
    }
    expect(covered).toBeGreaterThan(500);
    expect(partial).toBeGreaterThan(30);
    system.destroy();
    scene.liquid.destroy();
  });

  it('writes an opaque image over a backdrop, mixing it in only where there is liquid', async () => {
    const scene = await createScene();
    const radius = 0.03;
    const colour = { r: 0.2, g: 0.5, b: 0.8, a: 1 };
    const backdrop = constantTexture(scene.device, colour);
    const { system, set } = makeSet(scene.device, {
      list: lattice(2 * radius, 30, 12),
      radius,
      restVolume: (2 * radius) ** 2,
    });
    const pixels = await draw(scene, set, { backdrop: backdrop.createView() });
    const plain = [colour.r, colour.g, colour.b].map((c) => Math.round(srgb(c) * 255));
    let changed = 0;
    for (let i = 0; i < W * H; i++) {
      expect(pixels[i * 4 + 3]).toBe(255);
      const same = [0, 1, 2].every((c) => Math.abs(pixels[i * 4 + c]! - plain[c]!) <= 1);
      if (!same) changed++;
    }
    expect(changed).toBeGreaterThan(500);
    expect(changed).toBeLessThan(W * H * 0.6);
    // The corners are far from the liquid: exactly the encoded backdrop.
    for (const i of [0, W - 1, (H - 1) * W, W * H - 1]) {
      for (let c = 0; c < 3; c++)
        expect(Math.abs(pixels[i * 4 + c]! - plain[c]!)).toBeLessThanOrEqual(1);
    }
    system.destroy();
    scene.liquid.destroy();
    backdrop.destroy();
  });

  it('draws only the live particles of a set that has spare capacity', async () => {
    const scene = await createScene();
    // Index 0 is live; 1 and 2 are poison sitting elsewhere in the buffer.
    const { system, set } = makeSet(scene.device, {
      list: [
        [-0.5, 0],
        [0.5, 0],
        [0.5, 0.3],
      ],
      count: 1,
    });
    expect(set.count).toBe(1);
    expect(set.capacity).toBe(3);
    const alpha = alphaOf(await draw(scene, set, { params: SHAPE }));
    const [cx, cy] = centroid(alpha);
    const [px, py] = toPx(-0.5, 0);
    expect(Math.abs(cx - px)).toBeLessThan(0.1);
    expect(Math.abs(cy - py)).toBeLessThan(0.1);
    for (let y = 0; y < H; y++) for (let x = 64; x < W; x++) expect(alpha[y * W + x]).toBe(0); // the whole right half is empty

    // A count larger than the capacity is clamped, not an out-of-range draw.
    const clamped = await draw(scene, { ...set, count: 1000 }, { params: SHAPE });
    const all = alphaOf(clamped);
    expect(sumOf(all)).toBeGreaterThan(sumOf(alpha) * 2.5); // all three particles now
    system.destroy();
    scene.liquid.destroy();
  });

  it('draws the backdrop untouched, or nothing, for an empty set', async () => {
    const scene = await createScene();
    const { system, set } = makeSet(scene.device, { list: [[0, 0]], count: 0 });
    expect(set.count).toBe(0);
    const bare = await draw(scene, set);
    expect(bare.every((v) => v === 0)).toBe(true);

    const colour = { r: 0.04, g: 0.35, b: 0.9, a: 1 };
    const backdrop = constantTexture(scene.device, colour);
    const withBackdrop = await draw(scene, set, { backdrop: backdrop.createView() });
    const expected = [colour.r, colour.g, colour.b].map((c) => srgb(c) * 255);
    for (let i = 0; i < W * H; i++) {
      for (let c = 0; c < 3; c++) {
        expect(Math.abs(withBackdrop[i * 4 + c]! - expected[c]!)).toBeLessThanOrEqual(1.01);
      }
      expect(withBackdrop[i * 4 + 3]).toBe(255);
    }
    // And a `load` render keeps what was in the target.
    await draw(scene, set, { backdrop: backdrop.createView() });
    const kept = await draw(scene, set, { load: true });
    expect(Array.from(kept.subarray(0, 4))).toEqual(Array.from(withBackdrop.subarray(0, 4)));
    system.destroy();
    scene.liquid.destroy();
    backdrop.destroy();
  });

  it('does not let a non-finite particle poison the field', async () => {
    const scene = await createScene();
    const good = makeSet(scene.device, { list: [[0.1, 0.05, 0.5]] });
    await draw(scene, good.set, { params: SHAPE });
    const reference = sumOf((await readField(scene)).filter((_, i) => i % 4 === 0));

    const bad = makeSet(scene.device, {
      list: [
        [0.1, 0.05, 0.5],
        [Number.NaN, 0.1, 0],
        [Infinity, 0, 0],
        [0.3, -Infinity, 0],
        [-0.2, -0.1, Number.NaN],
      ],
    });
    await draw(scene, bad.set, { params: SHAPE });
    const field = await readField(scene);
    expect(field.every(Number.isFinite)).toBe(true);
    // The fourth particle is fine but has a NaN scalar: it still adds density (and no foam).
    const density = sumOf(field.filter((_, i) => i % 4 === 0));
    expect(density).toBeGreaterThan(reference * 1.5);
    expect(density).toBeLessThan(reference * 2.5);
    good.system.destroy();
    bad.system.destroy();
    scene.liquid.destroy();
  });
});

describe('liquid-2d foam', () => {
  it('puts the per-particle scalar into the g channel and makes high-scalar liquid whiter', async () => {
    const scene = await createScene();
    // Two lone blobs, one below foamLow and one above foamHigh.
    const { system, set } = makeSet(scene.device, {
      list: [
        [-0.5, 0, 0],
        [0.5, 0, 1],
      ],
    });
    const params = { ...SHAPE, foam: 1, foamLow: 0.3, foamHigh: 1 };
    const withFoam = await draw(scene, set, { params });
    const field = await readField(scene);
    const { width, height } = scene.liquid.field;
    let sumR = 0;
    let sumG = 0;
    let rightR = 0;
    let rightG = 0;
    for (let y = 0; y < height; y++)
      for (let x = 0; x < width; x++) {
        const i = (y * width + x) * 4;
        sumR += field[i]!;
        sumG += field[i + 1]!;
        if (x >= width / 2) {
          rightR += field[i]!;
          rightG += field[i + 1]!;
        }
      }
    expect(rightG / rightR).toBeCloseTo(1, 2); // the foamy blob: g == r
    expect((sumG - rightG) / (sumR - rightR)).toBeLessThan(1e-3); // the calm blob: g == 0
    expect(sumG / sumR).toBeCloseTo(0.5, 2);

    // Whiteness (red relative to blue; the water is blue, the foam white) at the two centres.
    const pixel = (pixels: Uint8Array, wx: number) => {
      const [px, py] = toPx(wx, 0);
      const i = (Math.floor(py) * W + Math.floor(px)) * 4;
      const a = pixels[i + 3]!;
      return { r: pixels[i]! / a, b: pixels[i + 2]! / a, a };
    };
    const calm = pixel(withFoam, -0.5);
    const foamy = pixel(withFoam, 0.5);
    expect(foamy.r).toBeGreaterThan(calm.r + 0.15);
    expect(foamy.r / foamy.b).toBeGreaterThan(calm.r / calm.b + 0.15);

    // The control: the same two blobs with foam switched off look alike.
    const without = await draw(scene, set, { params: { ...params, foam: 0 } });
    const a = pixel(without, -0.5);
    const b = pixel(without, 0.5);
    expect(Math.abs(a.r - b.r)).toBeLessThan(0.03);
    system.destroy();
    scene.liquid.destroy();
  });
});

describe('liquid-2d views, resize and lifetime', () => {
  it('renders the field and discs views, which differ from the final image and from each other', async () => {
    const scene = await createScene();
    const radius = 0.03;
    const { system, set } = makeSet(scene.device, {
      list: Array.from({ length: 60 }, (_, i) => [
        -0.3 + (i % 10) * radius * 2,
        -0.2 + Math.floor(i / 10) * radius * 2,
        i / 60,
      ]),
      radius,
    });
    const backdrop = constantTexture(scene.device, { r: 0.1, g: 0.3, b: 0.4, a: 1 });
    const views: Record<string, Uint8Array> = {};
    for (const view of ['final', 'field', 'discs'] as const) {
      views[view] = await draw(scene, set, { params: { view }, backdrop: backdrop.createView() });
    }
    const differ = (a: Uint8Array, b: Uint8Array) => a.some((v, i) => v !== b[i]);
    expect(differ(views.final!, views.field!)).toBe(true);
    expect(differ(views.final!, views.discs!)).toBe(true);
    expect(differ(views.field!, views.discs!)).toBe(true);
    // The field view is opaque; the discs sit over the backdrop, which fills everything else.
    expect(views.field!.every((v, i) => i % 4 !== 3 || v === 255)).toBe(true);
    expect(views.discs!.every((v, i) => i % 4 !== 3 || v === 255)).toBe(true);
    // Without a backdrop the discs view is transparent away from the discs.
    const bare = await draw(scene, set, { params: { view: 'discs' } });
    expect(bare[3]).toBe(0);
    expect(bare.some((v, i) => i % 4 === 3 && v > 0)).toBe(true);
    system.destroy();
    scene.liquid.destroy();
    backdrop.destroy();
  });

  it('resizes cleanly, the field following fieldScale, and destroys more than once', async () => {
    const device = await getTestDevice();
    const liquid = createLiquid2d({
      device,
      format: 'rgba8unorm',
      width: 64,
      height: 48,
      fieldScale: 0.5,
    });
    const { system, set } = makeSet(device, { list: [[0, 0, 0.5]] });
    const renderInto = async (width: number, height: number) => {
      const target = device.createTexture({
        size: [width, height],
        format: 'rgba8unorm',
        usage: GPUTextureUsage.RENDER_ATTACHMENT,
      });
      await expectValid(device, async () => {
        const encoder = device.createCommandEncoder();
        liquid.render(encoder, { particles: set, domain: DOMAIN, target: target.createView() });
        device.queue.submit([encoder.finish()]);
        await device.queue.onSubmittedWorkDone();
      });
      target.destroy();
    };
    expect([liquid.field.width, liquid.field.height]).toEqual([32, 24]);
    expect(liquid.field.format).toBe('rgba16float');
    await renderInto(64, 48);

    const before = liquid.field.texture;
    await expectValid(device, () => liquid.resize(201, 99));
    expect([liquid.field.width, liquid.field.height]).toEqual([101, 50]);
    expect(liquid.field.texture).not.toBe(before);
    await renderInto(201, 99);
    await expectValid(device, () => liquid.resize(40, 30));
    expect([liquid.field.width, liquid.field.height]).toEqual([20, 15]);
    await renderInto(40, 30);

    await expectValid(device, () => {
      liquid.destroy();
      liquid.destroy();
    });
    expect(() => liquid.resize(10, 10)).not.toThrow();
    system.destroy();
  });
});
