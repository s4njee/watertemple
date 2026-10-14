import { describe, expect, it, vi } from 'vitest';
import { createFluidSurface, type FluidSurface } from '../src/gpu/fluid-surface/index.ts';
import type { ParticleSet } from '../src/gpu/particles/types.ts';
import { computeCamera, type CameraMatrices } from '../src/gpu/scene/camera.ts';
import { loadEffect } from '../src/effects/registry.ts';
import { defaultParams } from '../src/effects/types.ts';
import { noProfiler } from '../src/gpu/profiler.ts';
import { expectValid, getTestDevice, readTexture } from './gpu.ts';
import { scalars } from './wgsl-eval.ts';

vi.setConfig({ testTimeout: 30_000 });

// --- Harness -------------------------------------------------------------------------------------

const W = 64;
const H = 64;
const FOV = Math.PI / 3;
const TAN = Math.tan(FOV / 2);

function camera(width = W, height = H, distance = 5): CameraMatrices {
  return computeCamera({
    position: [0, 0, distance],
    target: [0, 0, 0],
    fovY: FOV,
    aspect: width / height,
    near: 0.1,
    far: 100,
  });
}

type Particle = readonly [x: number, y: number, z: number, w?: number];

interface TestParticles extends ParticleSet {
  destroy(): void;
}

function makeParticles(
  device: GPUDevice,
  list: readonly Particle[],
  radius: number,
  options: { capacity?: number; count?: number; generation?: number } = {},
): TestParticles {
  const capacity = Math.max(options.capacity ?? list.length, 1);
  const data = new Float32Array(capacity * 4);
  list.forEach((p, i) => data.set([p[0], p[1], p[2], p[3] ?? 0], i * 4));
  const positions = device.createBuffer({
    label: 'test positions',
    size: capacity * 16,
    usage:
      GPUBufferUsage.STORAGE |
      GPUBufferUsage.VERTEX |
      GPUBufferUsage.COPY_SRC |
      GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(positions, 0, data);
  return {
    positions,
    count: options.count ?? list.length,
    capacity,
    radius,
    generation: options.generation ?? 0,
    destroy: () => positions.destroy(),
  };
}

interface Occluder {
  /** Linear depth of the occluding plane. */
  distance: number;
  /** Pixel columns it covers. */
  x0: number;
  x1: number;
}

interface TestScene {
  color: GPUTexture;
  depth: GPUTexture;
  target: GPUTexture;
  /** Created once, as a real consumer holds its views, so the renderer can cache against them. */
  views: { color: GPUTextureView; depth: GPUTextureView; target: GPUTextureView };
  destroy(): void;
}

function createScene(
  device: GPUDevice,
  rgb: readonly [number, number, number],
  options: { width?: number; height?: number; occluder?: Occluder; near?: number } = {},
): TestScene {
  const { width = W, height = H, occluder, near = 0.1 } = options;
  const make = (label: string, format: GPUTextureFormat, usage: GPUTextureUsageFlags) =>
    device.createTexture({ label, size: [width, height], format, usage });
  const color = make(
    'test scene colour',
    'rgba16float',
    GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_SRC,
  );
  const depth = make(
    'test scene depth',
    'depth32float',
    GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
  );
  const target = make(
    'test target',
    'rgba16float',
    GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC | GPUTextureUsage.TEXTURE_BINDING,
  );

  const encoder = device.createCommandEncoder();
  const pass = encoder.beginRenderPass({
    colorAttachments: [
      {
        view: color.createView(),
        loadOp: 'clear',
        storeOp: 'store',
        clearValue: { r: rgb[0], g: rgb[1], b: rgb[2], a: 1 },
      },
    ],
    depthStencilAttachment: {
      view: depth.createView(),
      depthClearValue: 1,
      depthLoadOp: 'clear',
      depthStoreOp: 'store',
    },
  });
  if (occluder) {
    const far = 100;
    const buffer = (far * (occluder.distance - near)) / (occluder.distance * (far - near));
    const module = device.createShaderModule({
      code: `
        @vertex fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4f {
          let c = vec2f(f32((i << 1u) & 2u), f32(i & 2u));
          return vec4f(c * 2.0 - 1.0, 0.0, 1.0);
        }
        struct Out { @location(0) c: vec4f, @builtin(frag_depth) d: f32 }
        @fragment fn fs() -> Out { return Out(vec4f(0.9, 0.1, 0.1, 1.0), ${buffer.toFixed(9)}); }`,
    });
    const pipeline = device.createRenderPipeline({
      layout: 'auto',
      vertex: { module, entryPoint: 'vs' },
      fragment: { module, entryPoint: 'fs', targets: [{ format: 'rgba16float' }] },
      depthStencil: { format: 'depth32float', depthWriteEnabled: true, depthCompare: 'less' },
    });
    pass.setPipeline(pipeline);
    pass.setScissorRect(occluder.x0, 0, occluder.x1 - occluder.x0, height);
    pass.draw(3);
  }
  pass.end();
  device.queue.submit([encoder.finish()]);
  return {
    color,
    depth,
    target,
    views: { color: color.createView(), depth: depth.createView(), target: target.createView() },
    destroy() {
      color.destroy();
      depth.destroy();
      target.destroy();
    },
  };
}

function render(
  device: GPUDevice,
  fluid: FluidSurface,
  scene: TestScene,
  particles: ParticleSet,
  params: Parameters<FluidSurface['render']>[1]['params'] = {},
  cam: CameraMatrices = camera(scene.target.width, scene.target.height),
  stages?: Parameters<FluidSurface['render']>[1]['stages'],
): void {
  const encoder = device.createCommandEncoder();
  fluid.render(encoder, {
    particles,
    camera: cam,
    sceneColor: scene.views.color,
    sceneDepth: scene.views.depth,
    target: scene.views.target,
    params,
    stages,
  });
  device.queue.submit([encoder.finish()]);
}

function half(bits: number): number {
  const sign = bits & 0x8000 ? -1 : 1;
  const exponent = (bits >> 10) & 0x1f;
  const fraction = bits & 0x3ff;
  if (exponent === 0) return sign * 2 ** -14 * (fraction / 1024);
  if (exponent === 31) return fraction ? NaN : sign * Infinity;
  return sign * 2 ** (exponent - 15) * (1 + fraction / 1024);
}

/** An r32float texture as a flat array. */
async function readF32(device: GPUDevice, texture: GPUTexture): Promise<Float32Array> {
  const bytes = await readTexture(device, texture, 4);
  return new Float32Array(bytes.buffer, bytes.byteOffset, bytes.byteLength / 4);
}

/** A 16-bit float texture with `channels` channels as flat numbers. */
async function readHalf(
  device: GPUDevice,
  texture: GPUTexture,
  channels: number,
): Promise<Float32Array> {
  const bytes = await readTexture(device, texture, channels * 2);
  const words = new Uint16Array(bytes.buffer, bytes.byteOffset, bytes.byteLength / 2);
  return Float32Array.from(words, half);
}

/** Writes an r32float texture from the CPU. */
function writeF32(device: GPUDevice, texture: GPUTexture, values: Float32Array): void {
  device.queue.writeTexture(
    { texture },
    values.buffer as ArrayBuffer,
    { offset: values.byteOffset, bytesPerRow: texture.width * 4 },
    [texture.width, texture.height],
  );
}

/** The linear depth of the near surface of a sphere along the ray through pixel (x, y). */
function sphereDepthAt(
  x: number,
  y: number,
  centre: readonly [number, number, number],
  radius: number,
  width = W,
  height = H,
): number {
  const ndcX = ((x + 0.5) / width) * 2 - 1;
  const ndcY = 1 - ((y + 0.5) / height) * 2;
  const dir = [ndcX * TAN * (width / height), ndcY * TAN, -1];
  const length = Math.hypot(...dir);
  const d = dir.map((v) => v / length);
  const c = [centre[0], centre[1], centre[2] - 5];
  const b = d[0]! * c[0]! + d[1]! * c[1]! + d[2]! * c[2]!;
  const disc = b * b - (c[0]! ** 2 + c[1]! ** 2 + c[2]! ** 2) + radius * radius;
  if (disc < 0) return 0;
  return -d[2]! * (b - Math.sqrt(disc));
}

const at = (image: ArrayLike<number>, x: number, y: number, stride = 1, width = W) =>
  image[(y * width + x) * stride] ?? Number.NaN;

// --- Tests ---------------------------------------------------------------------------------------

describe('sphere impostors', () => {
  it('a single particle renders a disc with the depth of a sphere', async () => {
    const device = await getTestDevice();
    await expectValid(device, async () => {
      const fluid = createFluidSurface({
        device,
        format: 'rgba16float',
        width: W,
        height: H,
        scale: 1,
      });
      const scene = createScene(device, [0.2, 0.2, 0.2]);
      const radius = 1;
      const particles = makeParticles(device, [[0, 0, 0]], radius);
      render(device, fluid, scene, particles);
      const depth = await readF32(device, fluid.textures.rawDepth.texture);

      // Centre: the near surface point, one radius in front of the centre (distance 5).
      const centre = at(depth, 32, 32);
      expect(centre).toBeCloseTo(sphereDepthAt(32, 32, [0, 0, 0], radius), 3);
      expect(centre).toBeGreaterThan(3.9);
      expect(centre).toBeLessThan(4.01);

      // Rim: depth rises as the sphere's does. Compare every pixel with the analytic sphere.
      let covered = 0;
      let worst = 0;
      for (let y = 0; y < H; y++) {
        for (let x = 0; x < W; x++) {
          const expected = sphereDepthAt(x, y, [0, 0, 0], radius);
          const actual = at(depth, x, y);
          if (expected === 0) {
            expect(actual).toBe(0);
          } else {
            covered++;
            worst = Math.max(worst, Math.abs(actual - expected));
          }
        }
      }
      expect(covered).toBeGreaterThan(300);
      expect(worst).toBeLessThan(1e-3);
      // Monotone from the centre along a row.
      for (let x = 33; x < 32 + 14; x++) {
        if (at(depth, x, 32) > 0)
          expect(at(depth, x, 32)).toBeGreaterThanOrEqual(at(depth, x - 1, 32));
      }

      particles.destroy();
      scene.destroy();
      fluid.destroy();
    });
  });
});

describe('nearest surface wins', () => {
  it('two particles at different depths resolve with the nearer in front where they overlap', async () => {
    const device = await getTestDevice();
    await expectValid(device, async () => {
      const fluid = createFluidSurface({
        device,
        format: 'rgba16float',
        width: W,
        height: H,
        scale: 1,
      });
      const scene = createScene(device, [0.2, 0.2, 0.2]);
      const radius = 0.8;
      // Same screen position (nearly), far one listed FIRST so a draw-order bug would show.
      const far: Particle = [0.2, 0, -2];
      const near: Particle = [-0.2, 0, 1];
      for (const order of [
        [far, near],
        [near, far],
      ]) {
        const particles = makeParticles(device, order, radius);
        render(device, fluid, scene, particles);
        const depth = await readF32(device, fluid.textures.rawDepth.texture);
        let overlap = 0;
        for (let y = 0; y < H; y++) {
          for (let x = 0; x < W; x++) {
            const a = sphereDepthAt(x, y, [near[0], near[1], near[2]], radius);
            const b = sphereDepthAt(x, y, [far[0], far[1], far[2]], radius);
            if (a > 0 && b > 0) overlap++;
            const expected = Math.min(...[a, b].filter((d) => d > 0), Infinity);
            const actual = at(depth, x, y);
            // Rim pixels are ill-conditioned (the depth slope is vertical there), so the tolerance is loose;
            // the two surfaces are 3 units apart, which is what this is about.
            if (Number.isFinite(expected)) expect(actual).toBeCloseTo(expected, 2);
            else expect(actual).toBe(0);
          }
        }
        expect(overlap).toBeGreaterThan(20);
        particles.destroy();
      }
      scene.destroy();
      fluid.destroy();
    });
  });
});

describe('smoothing', () => {
  async function smoothField(
    filter: 'bilateral' | 'narrow-range',
    field: Float32Array,
    options: { radius?: number; rangeScale?: number; iterations?: number } = {},
  ) {
    const device = await getTestDevice();
    const fluid = createFluidSurface({
      device,
      format: 'rgba16float',
      width: W,
      height: H,
      scale: 1,
      filter,
      filterRadius: 2,
      iterations: options.iterations ?? 1,
      rangeScale: options.rangeScale ?? 0.75,
    });
    const scene = createScene(device, [0, 0, 0]);
    // One particle only to make `live`; the depth stage is skipped so the seeded field is what is filtered.
    const particles = makeParticles(device, [[0, 0, 0]], options.radius ?? 0.5);
    // The projected radius of 0.5 at depth ~6 is ~ 5 px; the cap keeps kernels small.
    writeF32(device, fluid.textures.rawDepth.texture, field);
    render(device, fluid, scene, particles, {}, camera(), ['smooth']);
    const result = await readF32(device, fluid.textures.smoothDepth.texture);
    particles.destroy();
    scene.destroy();
    fluid.destroy();
    return result;
  }

  it('leaves a constant-depth field unchanged', async () => {
    const field = new Float32Array(W * H).fill(6);
    for (const filter of ['bilateral', 'narrow-range'] as const) {
      const result = await smoothField(filter, field);
      let worst = 0;
      for (let i = 0; i < field.length; i++) worst = Math.max(worst, Math.abs(result[i]! - 6));
      expect(worst).toBeLessThan(1e-5);
    }
  });

  it('does not move depth across a silhouette between two blobs at very different depths', async () => {
    // Left half is a blob at depth 3, right half one at depth 9: the silhouette is the column seam.
    const field = new Float32Array(W * H);
    for (let y = 0; y < H; y++) {
      for (let x = 8; x < W - 8; x++) field[y * W + x] = x < W / 2 ? 3 : 9;
    }
    const radius = 0.5;
    const window = 0.75 * radius; // the narrow-range window: rangeScale * particle radius
    const shift = (image: Float32Array) => {
      let worst = 0;
      for (let y = 8; y < H - 8; y++) {
        for (let x = 8; x < W - 8; x++) {
          const before = field[y * W + x]!;
          worst = Math.max(worst, Math.abs(image[y * W + x]! - before));
        }
      }
      return worst;
    };
    const narrow = shift(await smoothField('narrow-range', field, { radius }));
    const bilateral = shift(await smoothField('bilateral', field, { radius }));
    // A foreign surface can pull a pixel by at most the window, however far away it is.
    expect(narrow).toBeLessThanOrEqual(window + 1e-4);
    // At a gap this large the bilateral weight of the other side is e^-32: it keeps the edge too.
    expect(bilateral).toBeLessThan(0.05);
    // Background pixels stay background, and nothing is invented outside the blobs.
    const result = await smoothField('narrow-range', field, { radius });
    expect(result[0]).toBe(0);
    expect(result[(H / 2) * W + W - 1]).toBe(0);
  });

  it('smooths a noisy slanted surface: narrow-range more than bilateral at the same range', async () => {
    // A plane tilted in x with a checker of +-0.3 (a particle radius is 0.5): the bumps are what the
    // filter should remove. Over a slope, a bilateral kernel with a small range only sees its own
    // neighbours at the same height, a narrow-range kernel keeps all of them.
    const field = new Float32Array(W * H);
    const clean = (x: number) => 6 + 0.12 * (x - W / 2);
    for (let y = 0; y < H; y++) {
      for (let x = 0; x < W; x++) {
        field[y * W + x] = clean(x) + ((x + y) % 2 === 0 ? 0.3 : -0.3);
      }
    }
    const error = (image: Float32Array) => {
      let sum = 0;
      let n = 0;
      for (let y = 12; y < H - 12; y++) {
        for (let x = 12; x < W - 12; x++) {
          sum += Math.abs(image[y * W + x]! - clean(x));
          n++;
        }
      }
      return sum / n;
    };
    const noise = error(field);
    const narrow = error(await smoothField('narrow-range', field, { radius: 0.5 }));
    const bilateral = error(await smoothField('bilateral', field, { radius: 0.5 }));
    expect(narrow).toBeLessThan(noise * 0.5);
    expect(bilateral).toBeLessThan(noise);
    expect(narrow).toBeLessThan(bilateral);
  });
});

describe('normals', () => {
  /** The analytic view-space normal of a sphere at (0, 0, 0) (world), through pixel (x, y). */
  function sphereNormal(
    x: number,
    y: number,
    radius: number,
  ): [number, number, number] | undefined {
    const ndcX = ((x + 0.5) / W) * 2 - 1;
    const ndcY = 1 - ((y + 0.5) / H) * 2;
    const depth = sphereDepthAt(x, y, [0, 0, 0], radius);
    if (depth === 0) return undefined;
    // View-space hit point: x and y scale with depth.
    const p = [ndcX * TAN * depth, ndcY * TAN * depth, -depth];
    const n = [p[0]!, p[1]!, p[2]! + 5].map((v) => v / radius) as [number, number, number];
    return n;
  }

  /** Cosine between the normal and the direction to the eye: 0 on the silhouette, 1 facing it. */
  function facing(x: number, y: number, n: readonly [number, number, number]): number {
    const ndcX = ((x + 0.5) / W) * 2 - 1;
    const ndcY = 1 - ((y + 0.5) / H) * 2;
    const dir = [ndcX * TAN, ndcY * TAN, -1];
    const length = Math.hypot(...dir);
    return -(dir[0]! * n[0] + dir[1]! * n[1] + dir[2]! * n[2]) / length;
  }

  it.each(['raw', 'smooth'] as const)(
    'reconstructed normals of a large sphere match the analytic ones away from the silhouette (%s)',
    async (surface) => {
      const device = await getTestDevice();
      await expectValid(device, async () => {
        // A narrow kernel for the smooth case: smoothing a sphere with a kernel as wide as the sphere
        // would legitimately change its shape.
        const fluid = createFluidSurface({
          device,
          format: 'rgba16float',
          width: W,
          height: H,
          scale: 1,
          filterRadius: 0.1,
          iterations: 1,
        });
        const scene = createScene(device, [0.2, 0.2, 0.2]);
        const radius = 2;
        const particles = makeParticles(device, [[0, 0, 0]], radius);
        render(device, fluid, scene, particles, { surface });
        const normals = await readHalf(
          device,
          (surface === 'raw' ? fluid.textures.rawNormals : fluid.textures.smoothNormals).texture,
          4,
        );
        let checked = 0;
        let worst = 0;
        for (let y = 0; y < H; y++) {
          for (let x = 0; x < W; x++) {
            const expected = sphereNormal(x, y, radius);
            if (!expected || facing(x, y, expected) < 0.5) continue;
            const base = (y * W + x) * 4;
            const got = [normals[base]!, normals[base + 1]!, normals[base + 2]!];
            expect(normals[base + 3]).toBe(1);
            expect(Math.hypot(...got)).toBeCloseTo(1, 2);
            const dot = got[0]! * expected[0] + got[1]! * expected[1] + got[2]! * expected[2];
            worst = Math.max(worst, Math.acos(Math.min(1, dot)));
            checked++;
          }
        }
        expect(checked).toBeGreaterThan(400);
        // Under 6 degrees: one-sided finite differences on a sphere 30 pixels in radius.
        expect(worst).toBeLessThan(0.1);
        // Off the sphere the normal image is empty.
        expect(normals[3]).toBe(0);

        particles.destroy();
        scene.destroy();
        fluid.destroy();
      });
    },
  );

  it('takes the normal from the surface side of a silhouette, not from across the gap', async () => {
    // A flat wall at depth 6 in the left half, a second one at depth 3 in the right half: on the
    // wall at the seam the normal must still face the camera (a central difference would tilt it
    // wildly towards the step).
    const device = await getTestDevice();
    const fluid = createFluidSurface({
      device,
      format: 'rgba16float',
      width: W,
      height: H,
      scale: 1,
    });
    const scene = createScene(device, [0, 0, 0]);
    const particles = makeParticles(device, [[0, 0, 0]], 0.5);
    const field = new Float32Array(W * H);
    for (let y = 0; y < H; y++) {
      for (let x = 0; x < W; x++) field[y * W + x] = x < W / 2 ? 6 : 3;
    }
    writeF32(device, fluid.textures.smoothDepth.texture, field);
    render(device, fluid, scene, particles, {}, camera(), ['normals']);
    const normals = await readHalf(device, fluid.textures.smoothNormals.texture, 4);
    for (const x of [W / 2 - 1, W / 2]) {
      const base = ((H / 2) * W + x) * 4;
      expect(normals[base + 2]).toBeGreaterThan(0.95);
    }
    particles.destroy();
    scene.destroy();
    fluid.destroy();
  });
});

describe('thickness', () => {
  /** World direction of the ray through the centre of pixel (x, y) from the camera at z = 5. */
  function pixelRay(x: number, y: number): [number, number, number] {
    const ndcX = ((x + 0.5) / W) * 2 - 1;
    const ndcY = 1 - ((y + 0.5) / H) * 2;
    const v = [ndcX * TAN, ndcY * TAN, -1];
    const length = Math.hypot(...v);
    return [v[0]! / length, v[1]! / length, v[2]! / length];
  }

  async function thicknessImage(list: readonly Particle[], radius: number, gain = 1) {
    const device = await getTestDevice();
    const fluid = createFluidSurface({
      device,
      format: 'rgba16float',
      width: W,
      height: H,
      scale: 1,
      thicknessScale: 1,
      thicknessBlur: 0,
    });
    const scene = createScene(device, [0.2, 0.2, 0.2]);
    const particles = makeParticles(device, list, radius);
    render(device, fluid, scene, particles, { thicknessGain: gain });
    const image = await readHalf(device, fluid.textures.thickness.texture, 2);
    particles.destroy();
    scene.destroy();
    fluid.destroy();
    return image;
  }

  it('one particle gives the chord through its sphere, and integrates to its volume', async () => {
    const device = await getTestDevice();
    await expectValid(device, async () => {
      const radius = 1;
      const image = await thicknessImage([[0, 0, 0]], radius, 1);
      // Along a ray through the centre the chord is the diameter.
      const ray = pixelRay(32, 32);
      const centre: Particle = [ray[0] * 5, ray[1] * 5, 5 - ray[2] * -5];
      const exact = await thicknessImage([centre], radius, 1);
      expect(at(exact, 32, 32, 2)).toBeCloseTo(2 * radius, 2);

      // Volume: sum of chord times the area one pixel covers at the sphere's distance.
      const pixel = (2 * 5 * TAN) / H;
      let sum = 0;
      for (let i = 0; i < W * H; i++) sum += image[i * 2]!;
      const volume = sum * pixel * pixel;
      const expected = (4 / 3) * Math.PI * radius ** 3;
      expect(volume / expected).toBeGreaterThan(0.95);
      expect(volume / expected).toBeLessThan(1.05);
    });
  });

  it('scales linearly with the number of particles along a ray, and with the gain', async () => {
    const ray = pixelRay(32, 32);
    const along = (t: number): Particle => [ray[0] * t, ray[1] * t, 5 + ray[2] * t];
    const radius = 0.3;
    const one = at(await thicknessImage([along(4)], radius), 32, 32, 2);
    const two = at(await thicknessImage([along(4), along(6)], radius), 32, 32, 2);
    const three = at(await thicknessImage([along(3), along(5), along(8)], radius), 32, 32, 2);
    expect(one).toBeCloseTo(0.6, 2);
    expect(two / one).toBeCloseTo(2, 2);
    expect(three / one).toBeCloseTo(3, 2);
    const gained = at(await thicknessImage([along(4)], radius, 2.5), 32, 32, 2);
    expect(gained / one).toBeCloseTo(2.5, 2);
  });

  it('drawing every n-th particle with n times the weight gives the same thickness', async () => {
    const device = await getTestDevice();
    const ray = pixelRay(32, 32);
    // Eight identical particles along the ray; every second one, doubled, must add up to the same.
    const list: Particle[] = Array.from({ length: 8 }, (_, i): Particle => {
      const t = 3 + i * 0.7;
      return [ray[0] * t, ray[1] * t, 5 + ray[2] * t];
    });
    const thickness = async (stride: number) => {
      const fluid = createFluidSurface({
        device,
        format: 'rgba16float',
        width: W,
        height: H,
        scale: 1,
        thicknessScale: 1,
        thicknessBlur: 0,
        thicknessStride: stride,
      });
      const scene = createScene(device, [0.2, 0.2, 0.2]);
      const particles = makeParticles(device, list, 0.25);
      render(device, fluid, scene, particles);
      const image = await readHalf(device, fluid.textures.thickness.texture, 2);
      particles.destroy();
      scene.destroy();
      fluid.destroy();
      return at(image, 32, 32, 2);
    };
    const full = await thickness(1);
    expect(full).toBeCloseTo(8 * 0.5, 2);
    expect(await thickness(2)).toBeCloseTo(full, 2);
    expect(await thickness(4)).toBeCloseTo(full, 2);
    // A stride that does not divide the count still reads inside the buffer: 8 particles, every third.
    expect(await thickness(3)).toBeGreaterThan(0);
  });

  it('carries the particle scalar as thickness times w', async () => {
    const ray = pixelRay(32, 32);
    const image = await thicknessImage(
      [
        [ray[0] * 4, ray[1] * 4, 5 + ray[2] * 4, 1],
        [ray[0] * 6, ray[1] * 6, 5 + ray[2] * 6, 0],
      ],
      0.3,
    );
    // Two equal chords, one with w = 1 and one with 0: the mean w along the ray is 1/2.
    expect(at(image, 32, 32, 2) / 2).toBeCloseTo(0.6, 2);
    expect(image[(32 * W + 32) * 2 + 1]! / image[(32 * W + 32) * 2]!).toBeCloseTo(0.5, 2);
  });
});

describe('composite', () => {
  const BACKGROUND: [number, number, number] = [0.3, 0.3, 0.3];
  const RADIUS = 1.5;

  async function composite(options: {
    occluderDistance?: number;
    fluidZ: number;
    params?: Parameters<typeof render>[4];
  }) {
    const device = await getTestDevice();
    const fluid = createFluidSurface({
      device,
      format: 'rgba16float',
      width: W,
      height: H,
      scale: 1,
    });
    const scene = createScene(device, BACKGROUND, {
      occluder:
        options.occluderDistance === undefined
          ? undefined
          : { distance: options.occluderDistance, x0: 0, x1: W / 2 },
    });
    const particles = makeParticles(device, [[0, 0, options.fluidZ]], RADIUS);
    render(device, fluid, scene, particles, options.params);
    const out = await readHalf(device, scene.target, 4);
    const base = await readHalf(device, scene.color, 4);
    particles.destroy();
    scene.destroy();
    fluid.destroy();
    return { out, base, fluidZ: options.fluidZ };
  }

  /** True when the pixel and its neighbours `margin` pixels away are all inside the sphere's disc. */
  const inside = (x: number, y: number, z: number, margin: number) =>
    [
      [0, 0],
      [margin, 0],
      [-margin, 0],
      [0, margin],
      [0, -margin],
    ].every(
      ([dx, dy]) =>
        sphereDepthAt(
          Math.min(W - 1, Math.max(0, x + dx!)),
          Math.min(H - 1, Math.max(0, y + dy!)),
          [0, 0, z],
          RADIUS,
        ) > 0,
    );

  /** Pixels of the fluid's disc on one side that differ from the scene by more than `eps`. */
  function changed(
    { out, base, fluidZ }: Awaited<ReturnType<typeof composite>>,
    side: 'left' | 'right',
    eps = 0.01,
  ) {
    let covered = 0;
    let different = 0;
    for (let y = 0; y < H; y++) {
      for (let x = side === 'left' ? 0 : W / 2; x < (side === 'left' ? W / 2 : W); x++) {
        if (!inside(x, y, fluidZ, 3)) continue;
        covered++;
        const i = (y * W + x) * 4;
        const d = Math.max(
          Math.abs(out[i]! - base[i]!),
          Math.abs(out[i + 1]! - base[i + 1]!),
          Math.abs(out[i + 2]! - base[i + 2]!),
        );
        if (d > eps) different++;
      }
    }
    return { covered, different };
  }

  it('hides fluid behind an occluder and shows it where nothing is in front', async () => {
    // The fluid sits 7 from the eye (surface 5.5 to 8.5), the occluder covers the left half at 4.
    const hidden = await composite({ occluderDistance: 4, fluidZ: -2 });
    const left = changed(hidden, 'left');
    const right = changed(hidden, 'right');
    expect(left.covered).toBeGreaterThan(100);
    expect(left.different).toBe(0);
    expect(right.covered).toBeGreaterThan(100);
    expect(right.different / right.covered).toBeGreaterThan(0.9);
  });

  it('draws fluid in front of an occluder over it', async () => {
    // Same fluid, occluder moved behind it (12): now the left half is covered by the fluid too.
    const shown = await composite({ occluderDistance: 12, fluidZ: -2 });
    const left = changed(shown, 'left');
    expect(left.covered).toBeGreaterThan(100);
    expect(left.different / left.covered).toBeGreaterThan(0.9);
  });

  it('leaves every pixel away from the fluid exactly as the scene', async () => {
    const result = await composite({ occluderDistance: 4, fluidZ: -2 });
    let checked = 0;
    for (let y = 0; y < H; y++) {
      for (let x = 0; x < W; x++) {
        // Three pixels of margin: the silhouette is blended over a pixel or two.
        let near = false;
        for (let dy = -3; dy <= 3 && !near; dy++) {
          for (let dx = -3; dx <= 3; dx++) {
            const px = Math.min(W - 1, Math.max(0, x + dx));
            const py = Math.min(H - 1, Math.max(0, y + dy));
            if (sphereDepthAt(px, py, [0, 0, result.fluidZ], RADIUS) > 0) {
              near = true;
              break;
            }
          }
        }
        if (near) continue;
        checked++;
        const i = (y * W + x) * 4;
        expect(result.out[i]).toBe(result.base[i]);
        expect(result.out[i + 1]).toBe(result.base[i + 1]);
        expect(result.out[i + 2]).toBe(result.base[i + 2]);
      }
    }
    expect(checked).toBeGreaterThan(1000);
  });
});

describe('particle sets', () => {
  it('an empty set renders the scene untouched', async () => {
    const device = await getTestDevice();
    await expectValid(device, async () => {
      const fluid = createFluidSurface({ device, format: 'rgba16float', width: W, height: H });
      const scene = createScene(device, [0.31, 0.42, 0.53], {
        occluder: { distance: 3, x0: 10, x1: 40 },
      });
      const particles = makeParticles(device, [[0, 0, 0]], 1, { capacity: 16, count: 0 });
      render(device, fluid, scene, particles);
      const out = await readTexture(device, scene.target, 8);
      const base = await readTexture(device, scene.color, 8);
      expect(out.length).toBe(base.length);
      expect(out.every((byte, i) => byte === base[i])).toBe(true);
      expect(fluid.stats.particlesDrawn).toBe(0);
      particles.destroy();
      scene.destroy();
      fluid.destroy();
    });
  });

  it('draws only the live particles when count is smaller than capacity', async () => {
    const device = await getTestDevice();
    await expectValid(device, async () => {
      const fluid = createFluidSurface({
        device,
        format: 'rgba16float',
        width: W,
        height: H,
        scale: 1,
      });
      const scene = createScene(device, [0.2, 0.2, 0.2]);
      // Two particles in the buffer, far apart on screen; only the first is live. Capacity is larger still.
      const list: Particle[] = [
        [-1.5, 0, 0],
        [1.5, 0, 0],
      ];
      const particles = makeParticles(device, list, 0.6, { capacity: 8, count: 1 });
      render(device, fluid, scene, particles);
      let depth = await readF32(device, fluid.textures.rawDepth.texture);
      const sides = (image: Float32Array) => {
        let left = 0;
        let right = 0;
        for (let y = 0; y < H; y++) {
          for (let x = 0; x < W; x++) {
            if (at(image, x, y) <= 0) continue;
            if (x < W / 2) left++;
            else right++;
          }
        }
        return { left, right };
      };
      expect(sides(depth).left).toBeGreaterThan(30);
      expect(sides(depth).right).toBe(0);

      // Growing the count brings the second one in; the same buffer, no re-creation.
      render(device, fluid, scene, { ...particles, count: 2 });
      depth = await readF32(device, fluid.textures.rawDepth.texture);
      expect(sides(depth).right).toBeGreaterThan(30);

      // A count larger than the buffer is clamped, not an out-of-bounds draw.
      render(device, fluid, scene, { ...particles, count: 1000 });
      expect(fluid.stats.particlesDrawn).toBe(8);

      particles.destroy();
      scene.destroy();
      fluid.destroy();
    });
  });

  it('follows a ping-ponged positions buffer without creating bind groups per buffer', async () => {
    const device = await getTestDevice();
    await expectValid(device, async () => {
      const fluid = createFluidSurface({
        device,
        format: 'rgba16float',
        width: W,
        height: H,
        scale: 1,
      });
      const scene = createScene(device, [0.2, 0.2, 0.2]);
      const a = makeParticles(device, [[-1.5, 0, 0]], 0.6);
      const b = makeParticles(device, [[1.5, 0, 0]], 0.6);
      const centroid = async () => {
        const depth = await readF32(device, fluid.textures.rawDepth.texture);
        let sum = 0;
        let n = 0;
        for (let y = 0; y < H; y++) {
          for (let x = 0; x < W; x++) {
            if (at(depth, x, y) > 0) {
              sum += x;
              n++;
            }
          }
        }
        return sum / n;
      };
      render(device, fluid, scene, a);
      const created = fluid.stats.bindGroupsCreated;
      for (let frame = 0; frame < 4; frame++) {
        const current = frame % 2 === 0 ? b : a;
        // Same set, different buffer object, as a solver that swaps its buffers returns it.
        render(device, fluid, scene, { ...current, generation: 0 });
        const x = await centroid();
        if (current === b) expect(x).toBeGreaterThan(40);
        else expect(x).toBeLessThan(24);
      }
      expect(fluid.stats.bindGroupsCreated).toBe(created);

      // A re-created buffer (generation bump) is just another buffer.
      const c = makeParticles(device, [[0, 1.5, 0]], 0.6, { generation: 1 });
      render(device, fluid, scene, c);
      expect(await centroid()).toBeCloseTo(32, -0.5);
      expect(fluid.stats.bindGroupsCreated).toBe(created);
      // Destroying a buffer the renderer has drawn from before does not matter any more.
      a.destroy();
      render(device, fluid, scene, c);

      b.destroy();
      c.destroy();
      scene.destroy();
      fluid.destroy();
    });
  });
});

describe('lifecycle', () => {
  it('resizes cleanly, re-creating targets only when the size changes', async () => {
    const device = await getTestDevice();
    await expectValid(device, async () => {
      const fluid = createFluidSurface({ device, format: 'rgba16float', width: W, height: H });
      const particles = makeParticles(device, [[0, 0, 0]], 1);
      let scene = createScene(device, [0.2, 0.2, 0.2]);
      render(device, fluid, scene, particles);
      const generation = fluid.generation;
      fluid.resize(W, H);
      expect(fluid.generation).toBe(generation);
      expect(fluid.sizes.chain).toEqual([W / 2, H / 2]);

      fluid.resize(96, 40);
      expect(fluid.generation).toBeGreaterThan(generation);
      expect(fluid.sizes.chain).toEqual([48, 20]);
      expect(fluid.sizes.thickness).toEqual([24, 10]);
      scene.destroy();
      scene = createScene(device, [0.2, 0.2, 0.2], { width: 96, height: 40 });
      render(device, fluid, scene, particles, {}, camera(96, 40));
      const depth = await readF32(device, fluid.textures.rawDepth.texture);
      expect(depth.length).toBe(48 * 20);
      expect(depth.some((d) => d > 0)).toBe(true);

      // Changing the quality's resolution takes effect on the next render, with new bind groups.
      fluid.setQuality({ scale: 1, thicknessScale: 0.5 });
      render(device, fluid, scene, particles, {}, camera(96, 40));
      expect(fluid.sizes.chain).toEqual([96, 40]);
      expect(fluid.sizes.thickness).toEqual([48, 20]);

      particles.destroy();
      scene.destroy();
      fluid.destroy();
      fluid.destroy();
    });
  });

  it('shares its pipelines between instances on one device', async () => {
    const device = await getTestDevice();
    await expectValid(device, async () => {
      const one = createFluidSurface({ device, format: 'rgba16float', width: W, height: H });
      const two = createFluidSurface({ device, format: 'rgba16float', width: 32, height: 32 });
      const particles = makeParticles(device, [[0, 0, 0]], 1);
      const scene = createScene(device, [0.2, 0.2, 0.2]);
      render(device, one, scene, particles);
      const small = createScene(device, [0.2, 0.2, 0.2], { width: 32, height: 32 });
      render(device, two, small, particles, {}, camera(32, 32));
      one.destroy();
      render(device, two, small, particles, {}, camera(32, 32));
      two.destroy();
      particles.destroy();
      scene.destroy();
      small.destroy();
    });
  });

  it('renders every surface mode and debug view, at every filter, without a validation error', async () => {
    const device = await getTestDevice();
    await expectValid(device, async () => {
      const scene = createScene(device, [0.2, 0.2, 0.2]);
      const particles = makeParticles(
        device,
        [
          [0, 0, 0, 0.2],
          [0.5, 0.2, 0.3, 0.9],
          [-1, 0.5, -1, 0.5],
        ],
        0.5,
      );
      for (const filter of ['bilateral', 'narrow-range'] as const) {
        const fluid = createFluidSurface({
          device,
          format: 'rgba16float',
          width: W,
          height: H,
          filter,
          iterations: 3,
        });
        for (const surface of ['smooth', 'raw', 'split'] as const) {
          for (const debug of [
            'none',
            'raw-depth',
            'smooth-depth',
            'normals',
            'thickness',
            'foam',
          ] as const) {
            render(device, fluid, scene, particles, {
              surface,
              debug,
              foamAmount: 1,
              waterType: 2,
            });
          }
        }
        fluid.destroy();
      }
      await device.queue.onSubmittedWorkDone();
      particles.destroy();
      scene.destroy();
    });
  });
});

describe('fluid-surface.wgsl helpers', () => {
  const header = '#include "fluid-surface.wgsl"\n';
  // A symmetric perspective camera, 60 degrees vertical, aspect 2, built by hand.
  const camera = `
    fn test_camera() -> Camera {
      var cam: Camera;
      let f = 1.0 / tan(radians(30.0));
      cam.proj = mat4x4f(
        vec4f(f / 2.0, 0.0, 0.0, 0.0),
        vec4f(0.0, f, 0.0, 0.0),
        vec4f(0.0, 0.0, -1.0, -1.0),
        vec4f(0.0, 0.0, -0.2, 0.0));
      cam.near = 0.1;
      cam.far = 100.0;
      return cam;
    }`;

  it('intersects a sphere along the axis and misses beside it', async () => {
    const [front, inside, miss, behind, depthOfOffAxis] = await scalars(header, [
      'fluid_surface_sphere_depth(vec3f(0.0, 0.0, -1.0), vec3f(0.0, 0.0, -5.0), 1.0)',
      'fluid_surface_sphere_span(vec3f(0.0, 0.0, -1.0), vec3f(0.0, 0.0, -0.5), 1.0).x',
      'fluid_surface_sphere_depth(normalize(vec3f(1.0, 0.0, -1.0)), vec3f(0.0, 0.0, -5.0), 1.0)',
      'fluid_surface_sphere_depth(vec3f(0.0, 0.0, -1.0), vec3f(0.0, 0.0, 5.0), 1.0)',
      // A ray 45 degrees off axis through a sphere on that ray: depth is the z of the near point.
      'fluid_surface_sphere_depth(normalize(vec3f(1.0, 0.0, -1.0)), 5.0 * normalize(vec3f(1.0, 0.0, -1.0)), 1.0)',
    ]);
    expect(front).toBeCloseTo(4, 5);
    // The eye is inside: the near intersection is behind it.
    expect(inside).toBeLessThan(0);
    expect(miss).toBe(-1);
    expect(behind).toBe(-1);
    expect(depthOfOffAxis).toBeCloseTo(4 / Math.SQRT2, 5);
  });

  it('keeps a billboard large enough to cover the projected sphere, anywhere on screen', async () => {
    // For centres across the field of view, the tangent rays of the sphere in the radial plane meet
    // the plane of the quad at distance tan(theta +- alpha) * depth from the axis: the quad must reach them.
    const cases: [number, number, number][] = [];
    for (const theta of [0, 0.2, 0.5, 0.8, 1.0]) {
      for (const distance of [2, 5, 20]) cases.push([theta, distance, 0.5]);
    }
    const expressions = cases.map(
      ([theta, distance, radius]) =>
        `fluid_surface_billboard_half_size(${distance}.0 * vec3f(sin(${theta}), 0.0, -cos(${theta})), ${radius}, 0.1)`,
    );
    const results = await scalars(header, expressions);
    cases.forEach(([theta, distance, radius], i) => {
      const alpha = Math.asin(radius / distance);
      const depth = distance * Math.cos(theta);
      const centreX = distance * Math.sin(theta);
      const far = depth * Math.tan(theta + alpha) - centreX;
      const near = centreX - depth * Math.tan(theta - alpha);
      expect(results[i]).toBeGreaterThanOrEqual(Math.max(far, near) - 1e-4);
      // ... and not wasteful: within 15% of the exact extent (a few percent for a small sphere).
      expect(results[i]).toBeLessThan(Math.max(far, near) * 1.15 + 1e-3);
    });
    const [culled] = await scalars(header, [
      'fluid_surface_billboard_half_size(vec3f(0.0, 0.0, -0.3), 0.5, 0.1)',
    ]);
    expect(culled).toBe(-1);
  });

  it('reconstructs the view-space position of a depth, inverting the projection', async () => {
    const [x, y, z] = await scalars(`${header}${camera}`, [
      // uv (0.75, 0.25) is ndc (0.5, 0.5); at depth 4, x = 0.5 * 4 * tan(30) * 2 and y = 0.5 * 4 * tan(30).
      'fluid_surface_view_pos(test_camera(), vec2f(0.75, 0.25), 4.0).x',
      'fluid_surface_view_pos(test_camera(), vec2f(0.75, 0.25), 4.0).y',
      'fluid_surface_view_pos(test_camera(), vec2f(0.75, 0.25), 4.0).z',
    ]);
    const t = Math.tan(Math.PI / 6);
    expect(x).toBeCloseTo(0.5 * 4 * t * 2, 5);
    expect(y).toBeCloseTo(0.5 * 4 * t, 5);
    expect(z).toBe(-4);
    // Projecting it back gives the same uv: clip.xy / w with w = depth.
    const [u, v] = await scalars(`${header}${camera}`, [
      'camera_view_to_uv(test_camera(), fluid_surface_view_pos(test_camera(), vec2f(0.3, 0.8), 7.0)).x',
      'camera_view_to_uv(test_camera(), fluid_surface_view_pos(test_camera(), vec2f(0.3, 0.8), 7.0)).y',
    ]);
    expect(u).toBeCloseTo(0.3, 5);
    expect(v).toBeCloseTo(0.8, 5);
  });

  it('scales the kernel with the projected particle size and caps it', async () => {
    const [near, far, tiny, capped] = await scalars(`${header}${camera}`, [
      'fluid_surface_pixel_radius(test_camera(), 2.0, 0.1, 1000.0)',
      'fluid_surface_pixel_radius(test_camera(), 4.0, 0.1, 1000.0)',
      'fluid_surface_filter_radius(0.2, 2.0, 16.0)',
      'fluid_surface_filter_radius(100.0, 2.0, 16.0)',
    ]);
    expect(near! / far!).toBeCloseTo(2, 5);
    // 0.1 / 2 * f * 500 with f = 1 / tan 30.
    expect(near).toBeCloseTo(((0.1 / 2) * 500) / Math.tan(Math.PI / 6), 3);
    expect(tiny).toBe(1);
    expect(capped).toBe(16);
  });

  it('weights: Gaussian in space, the bilateral range kills far depths, the clamp bounds them', async () => {
    const [w0, w2, bilateralNear, bilateralFar, below, above, inside] = await scalars(header, [
      'fluid_surface_kernel_weight(0.0, 8.0)',
      'fluid_surface_kernel_weight(8.0, 8.0)',
      'fluid_surface_bilateral_weight(1.0, 0.25, 1.0)',
      'fluid_surface_bilateral_weight(1.0, 6.0, 1.0)',
      'fluid_surface_narrow_clamp(-5.0, 1.0, 0.5)',
      'fluid_surface_narrow_clamp(50.0, 1.0, 0.5)',
      'fluid_surface_narrow_clamp(1.2, 1.0, 0.5)',
    ]);
    expect(w0).toBe(1);
    // Two sigma (sigma = radius / 2) is exp(-2).
    expect(w2).toBeCloseTo(Math.exp(-2), 6);
    expect(bilateralNear).toBeCloseTo(Math.exp(-0.5 * 0.0625), 6);
    expect(bilateralFar).toBeLessThan(1e-7);
    expect(below).toBe(0.5);
    expect(above).toBe(1.5);
    expect(inside).toBeCloseTo(1.2, 6);
  });

  it('picks the one-sided difference with the smaller depth change, or the only valid one', async () => {
    const [chosenZ, onlyForward, onlyBackward, none, nx, ny, nz, fallback] = await scalars(header, [
      // forward has a depth step of 5, backward 0.1: take the backward one.
      'fluid_surface_pick_difference(vec3f(1.0, 0.0, -5.0), vec3f(1.0, 0.0, -0.1), true, true).z',
      'fluid_surface_pick_difference(vec3f(1.0, 0.0, -5.0), vec3f(1.0, 0.0, -0.1), true, false).z',
      'fluid_surface_pick_difference(vec3f(1.0, 0.0, -5.0), vec3f(1.0, 0.0, -0.1), false, true).z',
      'length(fluid_surface_pick_difference(vec3f(1.0, 0.0, -5.0), vec3f(1.0, 0.0, -0.1), false, false))',
      'fluid_surface_normal(vec3f(1.0, 0.0, 1.0), vec3f(0.0, 1.0, 0.0)).x',
      'fluid_surface_normal(vec3f(1.0, 0.0, 1.0), vec3f(0.0, 1.0, 0.0)).y',
      'fluid_surface_normal(vec3f(1.0, 0.0, 1.0), vec3f(0.0, 1.0, 0.0)).z',
      'fluid_surface_normal(vec3f(0.0), vec3f(0.0)).z',
    ]);
    expect(chosenZ).toBeCloseTo(-0.1, 6);
    expect(onlyForward).toBeCloseTo(-5, 6);
    expect(onlyBackward).toBeCloseTo(-0.1, 6);
    expect(none).toBe(0);
    // A surface that rises towards +x (z grows with x) tilts its normal towards -x, facing the camera.
    expect(nx).toBeCloseTo(-Math.SQRT1_2, 5);
    expect(ny).toBeCloseTo(0, 5);
    expect(nz).toBeCloseTo(Math.SQRT1_2, 5);
    expect(fallback).toBe(1);
  });

  it('clips a chord at the scene surface', async () => {
    const [full, clipped, hidden, miss] = await scalars(header, [
      'fluid_surface_chord(vec3f(0.0, 0.0, -1.0), vec3f(0.0, 0.0, -5.0), 1.0, 0.1, 100.0)',
      'fluid_surface_chord(vec3f(0.0, 0.0, -1.0), vec3f(0.0, 0.0, -5.0), 1.0, 0.1, 5.0)',
      'fluid_surface_chord(vec3f(0.0, 0.0, -1.0), vec3f(0.0, 0.0, -5.0), 1.0, 0.1, 3.0)',
      'fluid_surface_chord(vec3f(0.0, 0.0, -1.0), vec3f(3.0, 0.0, -5.0), 1.0, 0.1, 100.0)',
    ]);
    expect(full).toBeCloseTo(2, 5);
    expect(clipped).toBeCloseTo(1, 5);
    expect(hidden).toBe(0);
    expect(miss).toBe(0);
  });
});

describe('fluid-surface effect', () => {
  it('runs through every preset and every view without a validation error', async () => {
    const device = await getTestDevice();
    const format: GPUTextureFormat = 'rgba8unorm';
    const canvas = Object.assign(new EventTarget(), {
      width: 96,
      height: 64,
      style: {},
      getBoundingClientRect: () => ({ left: 0, top: 0, width: 96, height: 64 }),
      setPointerCapture() {},
      releasePointerCapture() {},
    }) as unknown as HTMLCanvasElement;
    const target = device.createTexture({
      size: [96, 64],
      format,
      usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC,
    });

    await expectValid(device, async () => {
      const module = await loadEffect('fluid-surface');
      expect(module.meta.epic).toBe('7');
      expect(Object.keys(module.presets ?? {}).length).toBeGreaterThanOrEqual(3);
      const params = defaultParams(module.params);
      const instance = await module.create({
        device,
        format,
        canvas,
        params,
        profiler: noProfiler,
        hdr: false,
      });
      instance.resize(96, 64, 1);
      const views = (module.params.view as { options: readonly string[] }).options;
      const sources = (module.params.source as { options: readonly string[] }).options;
      expect(views.length).toBeGreaterThanOrEqual(7);
      let frame = 0;
      const draw = () => {
        const encoder = device.createCommandEncoder();
        instance.frame({ time: frame / 60, delta: 1 / 60, frame }, encoder, target.createView());
        device.queue.submit([encoder.finish()]);
        frame++;
      };
      // Small counts keep the suite fast; the point is that every path encodes and validates.
      for (const preset of Object.values(module.presets ?? {})) {
        Object.assign(params, defaultParams(module.params), preset, { count: 3000 });
        draw();
      }
      for (const view of views) {
        for (const compare of [false, true]) {
          Object.assign(params, defaultParams(module.params), { view, compare, count: 3000 });
          draw();
        }
      }
      for (const source of sources) {
        Object.assign(params, defaultParams(module.params), { source, count: 3000 });
        draw();
      }
      // The count grows past the first buffer: the particle set is re-created.
      Object.assign(params, defaultParams(module.params), { count: 9000 });
      draw();
      instance.resize(48, 32, 1);
      draw();
      await device.queue.onSubmittedWorkDone();
      instance.destroy();
    });
    target.destroy();
  });
});
