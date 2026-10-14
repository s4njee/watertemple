import { describe, expect, it, vi } from 'vitest';
import {
  CUBE_FACE_NAMES,
  createCubemap,
  cubeFaceDirection,
  type Cubemap,
} from '../src/gpu/cubemap.ts';
import { checkShader, formatDiagnostic } from '../src/gpu/shader.ts';
import { getChunk } from '../src/wgsl/index.ts';
import { getTestDevice } from './gpu.ts';
import { gpuF32, lane, rowsOf, scalars } from './wgsl-eval.ts';

// Brute-force comparisons run thousands of samples; leave headroom for a busy machine.
vi.setConfig({ testTimeout: 60_000 });

// --- Helpers -------------------------------------------------------------------------------------

const CHUNKS = ['fresnel.wgsl', 'sky.wgsl', 'specular.wgsl', 'sss.wgsl', 'waterplane.wgsl'];
const DEG = Math.PI / 180;

/** Rows of `count` consecutive seeds 0.5, 1.5, ... in lane x, so WGSL can derive random inputs. */
function seedRows(count: number, extra: (i: number) => number[] = () => []): Float32Array {
  return rowsOf(Array.from({ length: count }, (_, i) => [i + 0.5, ...extra(i)]));
}

/** Deterministic random unit vector on the sphere, from a float seed, in WGSL. */
const RANDOM_UNIT = `
fn rand_unit(seed: f32) -> vec3f {
  let r = hash_1to2(seed);
  let z = 1.0 - 2.0 * r.x;
  let a = TAU * r.y;
  let s = sqrt(max(1.0 - z * z, 0.0));
  return vec3f(s * cos(a), z, s * sin(a));
}
fn rand_up(seed: f32) -> vec3f {
  var u = rand_unit(seed);
  u.y = abs(u.y);
  return normalize(u + vec3f(0.0, 1e-3, 0.0));
}
`;

function mean(values: number[]): number {
  return values.reduce((a, b) => a + b, 0) / values.length;
}

function halfToFloat(h: number): number {
  const sign = h & 0x8000 ? -1 : 1;
  const exponent = (h >> 10) & 0x1f;
  const fraction = h & 0x3ff;
  if (exponent === 0) return sign * 2 ** -14 * (fraction / 1024);
  if (exponent === 31) return fraction ? Number.NaN : sign * Infinity;
  return sign * 2 ** (exponent - 15) * (1 + fraction / 1024);
}

/** One layer and mip level of an rgba16float texture (needs COPY_SRC) as RGBA floats. */
async function readLayer(
  device: GPUDevice,
  texture: GPUTexture,
  layer: number,
  mip = 0,
): Promise<{ data: Float32Array; width: number; height: number }> {
  const width = Math.max(1, texture.width >> mip);
  const height = Math.max(1, texture.height >> mip);
  const bytesPerRow = Math.ceil((width * 8) / 256) * 256;
  const buffer = device.createBuffer({
    size: bytesPerRow * height,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });
  const encoder = device.createCommandEncoder();
  encoder.copyTextureToBuffer(
    { texture, mipLevel: mip, origin: { x: 0, y: 0, z: layer } },
    { buffer, bytesPerRow },
    { width, height, depthOrArrayLayers: 1 },
  );
  device.queue.submit([encoder.finish()]);
  await buffer.mapAsync(GPUMapMode.READ);
  const padded = new Uint16Array(buffer.getMappedRange().slice(0));
  buffer.unmap();
  buffer.destroy();
  const data = new Float32Array(width * height * 4);
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width * 4; x++) {
      data[y * width * 4 + x] = halfToFloat(padded[(y * bytesPerRow) / 2 + x] ?? 0);
    }
  }
  return { data, width, height };
}

/** Samples the cube texture through the hardware sampler at each direction. */
async function sampleCube(
  device: GPUDevice,
  cube: Cubemap,
  dirs: number[][],
): Promise<Float32Array> {
  const { module, diagnostics } = await checkShader(device, {
    label: 'cube-sample-harness',
    code: `
@group(0) @binding(0) var cube: texture_cube<f32>;
@group(0) @binding(1) var samp: sampler;
@group(0) @binding(2) var<storage, read> dirs: array<vec4f>;
@group(0) @binding(3) var<storage, read_write> outs: array<vec4f>;
@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) id: vec3u) {
  if (id.x < ${dirs.length}u) { outs[id.x] = textureSampleLevel(cube, samp, dirs[id.x].xyz, 0.0); }
}`,
  });
  expect(diagnostics.map(formatDiagnostic)).toEqual([]);
  const input = rowsOf(dirs);
  const inBuffer = device.createBuffer({
    size: input.byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(inBuffer, 0, input);
  const outBuffer = device.createBuffer({
    size: input.byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
  });
  const readBuffer = device.createBuffer({
    size: input.byteLength,
    usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
  });
  const pipeline = device.createComputePipeline({
    layout: 'auto',
    compute: { module, entryPoint: 'main' },
  });
  const group = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: cube.view },
      { binding: 1, resource: device.createSampler({ magFilter: 'linear', minFilter: 'linear' }) },
      { binding: 2, resource: { buffer: inBuffer } },
      { binding: 3, resource: { buffer: outBuffer } },
    ],
  });
  const encoder = device.createCommandEncoder();
  const pass = encoder.beginComputePass();
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, group);
  pass.dispatchWorkgroups(Math.ceil(dirs.length / 64));
  pass.end();
  encoder.copyBufferToBuffer(outBuffer, 0, readBuffer, 0, input.byteLength);
  device.queue.submit([encoder.finish()]);
  await readBuffer.mapAsync(GPUMapMode.READ);
  const result = new Float32Array(readBuffer.getMappedRange().slice(0));
  readBuffer.unmap();
  for (const buffer of [inBuffer, outBuffer, readBuffer]) buffer.destroy();
  return result;
}

async function renderCube(device: GPUDevice, cube: Cubemap, user?: number[]): Promise<void> {
  const encoder = device.createCommandEncoder();
  cube.render(encoder, user);
  device.queue.submit([encoder.finish()]);
  await device.queue.onSubmittedWorkDone();
}

// --- Chunk hygiene -------------------------------------------------------------------------------

describe('optics surface chunks', () => {
  const sources = CHUNKS.map((name) => [name, getChunk(name) ?? ''] as const);

  it.each(sources)('%s is a pure library with a documented header', (_name, source) => {
    expect(source.length).toBeGreaterThan(0);
    expect(source.startsWith('//')).toBe(true);
    expect(source).not.toMatch(/@(vertex|fragment|compute)\b/);
    expect(source).not.toMatch(/@(group|binding)\b/);
    expect(source).not.toMatch(/^var\b/m);
    const header = source
      .split('\n')
      .filter((line) => line.startsWith('//'))
      .join('\n');
    const publicFunctions = [...source.matchAll(/^fn ([a-z]\w*)\(/gm)].map((m) => m[1] ?? '');
    expect(publicFunctions.length).toBeGreaterThan(0);
    expect(publicFunctions.filter((fn) => !header.includes(fn))).toEqual([]);
  });

  it('uses one distinctive prefix per file and private helpers start with an underscore', () => {
    const prefixes: Record<string, string> = {
      'fresnel.wgsl': 'fresnel_',
      'sky.wgsl': 'sky_',
      'specular.wgsl': 'spec_',
      'sss.wgsl': 'sss_',
      'waterplane.wgsl': 'waterplane_',
    };
    for (const [name, source] of sources) {
      const prefix = prefixes[name] ?? '';
      for (const match of source.matchAll(/^fn (\w+)\(/gm)) {
        const fn = match[1] ?? '';
        expect(fn.startsWith(prefix) || fn.startsWith('_'), `${name}: ${fn}`).toBe(true);
      }
      for (const match of source.matchAll(/^const (\w+)/gm)) {
        const id = (match[1] ?? '').replace(/^_/, '').toLowerCase();
        expect(id.startsWith(prefix), `${name}: const ${match[1]}`).toBe(true);
      }
    }
  });
});

// --- Fresnel -------------------------------------------------------------------------------------

describe('fresnel.wgsl', () => {
  const WATER = 1.333;
  const f0Water = ((WATER - 1) / (WATER + 1)) ** 2;

  it('has the right reflectance at normal incidence and at grazing', async () => {
    const [f0, f0Glass, schlick1, schlick0, exact1, exact0, water1, water0, constant] =
      await scalars('#include "fresnel.wgsl"', [
        'fresnel_f0(1.0, 1.333)',
        'fresnel_f0(1.0, 1.5)',
        'fresnel_schlick(1.0, 0.0204)',
        'fresnel_schlick(0.0, 0.0204)',
        'fresnel_exact(1.0, 1.0, 1.333)',
        'fresnel_exact(0.0, 1.0, 1.333)',
        'fresnel_water(1.0)',
        'fresnel_water(0.0)',
        'FRESNEL_F0_WATER',
      ]);
    expect(f0).toBeCloseTo(f0Water, 6);
    expect(f0).toBeGreaterThan(0.019);
    expect(f0).toBeLessThan(0.022);
    expect(f0Glass).toBeCloseTo(0.04, 6);
    expect(schlick1).toBeCloseTo(0.0204, 6);
    expect(schlick0).toBeCloseTo(1, 6);
    expect(exact1).toBeCloseTo(f0Water, 5);
    expect(exact0).toBeCloseTo(1, 6);
    expect(water1).toBeCloseTo(f0Water, 5);
    expect(water0).toBeCloseTo(1, 6);
    expect(constant).toBeCloseTo(f0Water, 5);
  });

  it('Schlick tracks the exact form: 0.01 up to 70 degrees, 0.06 everywhere', async () => {
    const angles = Array.from({ length: 91 }, (_, i) => i);
    const out = await gpuF32(
      `#include "fresnel.wgsl"
       fn f(x: vec4f) -> vec4f {
         let c = cos(x.x * 0.017453292519943295);
         return vec4f(fresnel_schlick(c, FRESNEL_F0_WATER), fresnel_exact(c, 1.0, 1.333), 0.0, 0.0);
       }`,
      rowsOf(angles.map((a) => [a])),
    );
    const schlick = lane(out, 0);
    const exact = lane(out, 1);
    let worst = 0;
    angles.forEach((angle, i) => {
      const error = Math.abs((schlick[i] ?? 0) - (exact[i] ?? 0));
      worst = Math.max(worst, error);
      if (angle <= 70) expect(error, `${angle} degrees`).toBeLessThan(0.01);
      expect(error, `${angle} degrees`).toBeLessThan(0.06);
    });
    expect(worst).toBeGreaterThan(0.04); // the stated bound is honest, not slack
    // Both rise monotonically toward grazing.
    for (let i = 1; i < angles.length; i++) {
      expect(schlick[i] ?? 0).toBeGreaterThanOrEqual((schlick[i - 1] ?? 0) - 1e-6);
      expect(exact[i] ?? 0).toBeGreaterThanOrEqual((exact[i - 1] ?? 0) - 1e-6);
    }
  });

  it('exact form matches an independent evaluation of the Fresnel equations', async () => {
    const angles = [10, 30, 45, 60, 75, 85];
    const out = await gpuF32(
      `#include "fresnel.wgsl"
       fn f(x: vec4f) -> vec4f { return vec4f(fresnel_exact(cos(x.x * 0.017453292519943295), 1.0, 1.333), 0.0, 0.0, 0.0); }`,
      rowsOf(angles.map((a) => [a])),
    );
    angles.forEach((angle, i) => {
      const ci = Math.cos(angle * DEG);
      const st = Math.sin(angle * DEG) / WATER;
      const ct = Math.sqrt(1 - st * st);
      const rs = ((ci - WATER * ct) / (ci + WATER * ct)) ** 2;
      const rp = ((ct - WATER * ci) / (ct + WATER * ci)) ** 2;
      expect(lane(out, 0)[i]).toBeCloseTo((rs + rp) / 2, 5);
    });
  });

  it('clamps out-of-range cosines instead of extrapolating', async () => {
    const [back, over, vecBack] = await scalars('#include "fresnel.wgsl"', [
      'fresnel_schlick(-0.5, 0.02)',
      'fresnel_schlick(1.5, 0.02)',
      'fresnel_schlick_vec(-1.0, vec3f(0.02, 0.5, 0.9)).x',
    ]);
    expect(back).toBeCloseTo(1, 6);
    expect(over).toBeCloseTo(0.02, 6);
    expect(vecBack).toBeCloseTo(1, 6);
  });

  it('roughness lowers the grazing reflectance and zero roughness is plain Schlick', async () => {
    const [zero, plain, rough, roughHalf, edge] = await scalars('#include "fresnel.wgsl"', [
      'fresnel_schlick_roughness(0.2, 0.04, 0.0)',
      'fresnel_schlick(0.2, 0.04)',
      'fresnel_schlick_roughness(0.0, 0.04, 1.0)',
      'fresnel_schlick_roughness(0.0, 0.04, 0.5)',
      'fresnel_schlick_roughness(1.0, 0.04, 0.7)',
    ]);
    expect(zero).toBeCloseTo(plain ?? 0, 6);
    expect(rough).toBeCloseTo(0.04, 6); // fully rough: the edge brightening is gone
    expect(roughHalf).toBeCloseTo(0.5, 6);
    expect(edge).toBeCloseTo(0.04, 6);
  });

  it('handles light leaving the denser medium, including total internal reflection', async () => {
    const critical = Math.asin(1 / WATER);
    const [normal, beyond, justUnder, exactUnder, schlickUnder, sameSide] = await scalars(
      '#include "fresnel.wgsl"',
      [
        'fresnel_schlick_dielectric(1.0, 1.333, 1.0)',
        `fresnel_schlick_dielectric(cos(${critical + 0.05}), 1.333, 1.0)`,
        `fresnel_exact(cos(${critical - 0.05}), 1.333, 1.0)`,
        `fresnel_exact(cos(${0.5}), 1.333, 1.0)`,
        `fresnel_schlick_dielectric(cos(${0.5}), 1.333, 1.0)`,
        'fresnel_schlick_dielectric(0.6, 1.0, 1.333) - fresnel_schlick(0.6, FRESNEL_F0_WATER)',
      ],
    );
    expect(normal).toBeCloseTo(f0Water, 5);
    expect(beyond).toBe(1);
    expect(justUnder).toBeLessThan(1);
    expect(justUnder).toBeGreaterThan(0.1);
    expect(Math.abs((schlickUnder ?? 0) - (exactUnder ?? 0))).toBeLessThan(0.06);
    expect(Math.abs(sameSide ?? 1)).toBeLessThan(1e-6);
    // Total internal reflection from the exact form: everything past the critical angle reflects.
    const [tir] = await scalars('#include "fresnel.wgsl"', [
      `fresnel_exact(cos(${critical + 0.1}), 1.333, 1.0)`,
    ]);
    expect(tir).toBe(1);
  });
});

// --- Sky -----------------------------------------------------------------------------------------

describe('sky.wgsl', () => {
  // Rows: x = (seed, sun elevation in degrees, azimuth of dir relative to the sun in degrees, haze).
  const SKY = `#include "sky.wgsl"
    #include "hash.wgsl"
    #include "math.wgsl"
    ${RANDOM_UNIT}
    fn sun_of(elev_deg: f32) -> vec3f { let e = elev_deg * 0.017453292519943295; return vec3f(0.0, sin(e), cos(e)); }`;

  it('is finite and non-negative for any direction, sun elevation and haze', async () => {
    const count = 6000;
    const out = await gpuF32(
      `${SKY}
       fn f(x: vec4f) -> vec4f {
         let dir = rand_unit(x.x);
         let sun = sun_of(x.y);
         return vec4f(sky_radiance_ex(dir, sun, x.z, x.w), 0.0);
       }`,
      rowsOf(
        Array.from({ length: count }, (_, i) => [
          i + 0.5,
          -60 + ((i * 37) % 151), // -60 .. 90
          (i % 11) / 10,
          (i % 7) / 3, // sun_gain 0 .. 2
        ]),
      ),
    );
    for (const value of out) {
      expect(Number.isFinite(value)).toBe(true);
      expect(value).toBeGreaterThanOrEqual(0);
    }
  });

  it('is brightest toward the sun and falls off away from it', async () => {
    for (const elevation of [5, 20, 45, 75]) {
      const angles = [0, 0.5, 2, 5, 10, 20, 45, 90, 135, 180];
      const out = await gpuF32(
        `${SKY}
         fn f(x: vec4f) -> vec4f {
           let sun = sun_of(x.y);
           // Rotate the sun direction away from itself within the vertical plane.
           let a = (x.y + x.x) * 0.017453292519943295;
           let dir = vec3f(0.0, sin(a), cos(a));
           return vec4f(luminance_of(sky_radiance(dir, sun)), 0.0, 0.0, 0.0);
         }
         fn luminance_of(c: vec3f) -> f32 { return dot(c, vec3f(0.2126, 0.7152, 0.0722)); }`,
        rowsOf(angles.map((a) => [a, elevation])),
      );
      const luminance = lane(out, 0);
      expect(luminance[0]).toBeGreaterThan(5);
      // Moving along the vertical plane away from the sun eventually crosses the zenith, so only
      // compare against the sun itself and against the nearby halo ring.
      expect(luminance[0]).toBeGreaterThan(luminance[3] ?? 0);
      expect(luminance[2]).toBeGreaterThan(luminance[5] ?? 0);
      for (const l of luminance.slice(1)) expect(luminance[0] ?? 0).toBeGreaterThan(l);
    }
  });

  it('the sun is the brightest point of the sphere (random directions)', async () => {
    const count = 5000;
    const out = await gpuF32(
      `${SKY}
       fn f(x: vec4f) -> vec4f {
         let sun = sun_of(30.0);
         let c = sky_radiance(rand_unit(x.x), sun);
         return vec4f(dot(c, vec3f(0.2126, 0.7152, 0.0722)), 0.0, 0.0, 0.0);
       }`,
      seedRows(count),
    );
    const [sunLuminance] = lane(
      await gpuF32(
        `${SKY}
         fn f(x: vec4f) -> vec4f {
           let sun = sun_of(30.0);
           return vec4f(dot(sky_radiance(sun, sun), vec3f(0.2126, 0.7152, 0.0722)), 0.0, 0.0, 0.0);
         }`,
        rowsOf([[0]]),
      ),
      0,
    );
    expect(Math.max(...lane(out, 0))).toBeLessThan(sunLuminance ?? 0);
    expect(sunLuminance).toBeGreaterThan(15);
  });

  it('is continuous across the horizon away from the sun', async () => {
    for (const elevation of [-15, 0, 4, 12, 30, 60]) {
      const out = await gpuF32(
        `${SKY}
         fn f(x: vec4f) -> vec4f {
           // The view azimuth is opposite the sun (dir.z < 0); x.x is the elevation of the view ray.
           let sun = sun_of(x.y);
           let a = x.x;
           let dir = normalize(vec3f(0.0, a, -1.0));
           return vec4f(sky_radiance(dir, sun), 0.0);
         }`,
        rowsOf([
          [-1e-6, elevation],
          [1e-6, elevation],
          [-1e-4, elevation],
          [1e-4, elevation],
        ]),
      );
      const rgb = (i: number) => [out[i * 4] ?? 0, out[i * 4 + 1] ?? 0, out[i * 4 + 2] ?? 0];
      for (let channel = 0; channel < 3; channel++) {
        expect(
          Math.abs((rgb(1)[channel] ?? 0) - (rgb(0)[channel] ?? 0)),
          `elevation ${elevation}, +-1e-6`,
        ).toBeLessThan(0.005);
        expect(
          Math.abs((rgb(3)[channel] ?? 0) - (rgb(2)[channel] ?? 0)),
          `elevation ${elevation}, +-1e-4`,
        ).toBeLessThan(0.02);
      }
    }
  });

  it('keeps the horizon continuous toward the sun too once the sun is above it', async () => {
    const out = await gpuF32(
      `${SKY}
       fn f(x: vec4f) -> vec4f {
         let sun = sun_of(25.0);
         let dir = normalize(vec3f(0.0, x.x, 1.0));
         return vec4f(sky_radiance(dir, sun), 0.0);
       }`,
      rowsOf([[-1e-5], [1e-5]]),
    );
    for (let channel = 0; channel < 3; channel++) {
      expect(Math.abs((out[4 + channel] ?? 0) - (out[channel] ?? 0))).toBeLessThan(0.01);
    }
  });

  it('removes the sun with sun_gain 0 and reddens the sunlight near the horizon', async () => {
    const [withSun, withoutSun, noonRatio, sunsetRatio, nightSun, noonMax] = await scalars(
      `${SKY}`,
      [
        'sky_radiance_ex(sun_of(30.0), sun_of(30.0), 0.3, 1.0).y',
        'sky_radiance_ex(sun_of(30.0), sun_of(30.0), 0.3, 0.0).y',
        'sky_sun_color(sun_of(80.0)).x / sky_sun_color(sun_of(80.0)).z',
        'sky_sun_color(sun_of(2.0)).x / sky_sun_color(sun_of(2.0)).z',
        'length(sky_sun_color(sun_of(-20.0)))',
        'max(sky_sun_color(sun_of(90.0)).x, max(sky_sun_color(sun_of(90.0)).y, sky_sun_color(sun_of(90.0)).z))',
      ],
    );
    expect(withSun).toBeGreaterThan(10);
    expect(withoutSun).toBeLessThan(2);
    expect(noonRatio).toBeLessThan(1.6);
    expect(sunsetRatio).toBeGreaterThan(3);
    expect(nightSun).toBe(0);
    expect(noonMax).toBeLessThanOrEqual(1);
  });

  it('is blue overhead at noon, warm near a low sun, and nearly dark at night', async () => {
    const [noonRed, noonBlue, sunsetGlowRed, sunsetGlowBlue, nightMax] = await scalars(SKY, [
      'sky_radiance(vec3f(0.0, 1.0, 0.0), sun_of(60.0)).x',
      'sky_radiance(vec3f(0.0, 1.0, 0.0), sun_of(60.0)).z',
      'sky_radiance(normalize(vec3f(0.0, 0.08, 1.0)), sun_of(1.0)).x',
      'sky_radiance(normalize(vec3f(0.0, 0.08, 1.0)), sun_of(1.0)).z',
      'max(sky_radiance(normalize(vec3f(0.3, 0.5, -1.0)), sun_of(-30.0)).z, sky_radiance(vec3f(0.0, 1.0, 0.0), sun_of(-30.0)).z)',
    ]);
    expect(noonBlue).toBeGreaterThan(2 * (noonRed ?? 1));
    expect(sunsetGlowRed).toBeGreaterThan(sunsetGlowBlue ?? 1);
    expect(nightMax).toBeLessThan(0.05);
  });

  it('dither is zero-mean triangular noise within one 8-bit step', async () => {
    const count = 4000;
    const out = await gpuF32(
      `${SKY}
       fn f(x: vec4f) -> vec4f { return vec4f(sky_dither(vec2f(x.x, x.y)), 0.0, 0.0, 0.0); }`,
      rowsOf(Array.from({ length: count }, (_, i) => [(i % 64) + 0.5, Math.floor(i / 64) + 0.5])),
    );
    const values = lane(out, 0).slice(0, count);
    expect(Math.min(...values)).toBeGreaterThanOrEqual(-1);
    expect(Math.max(...values)).toBeLessThanOrEqual(1);
    expect(Math.abs(mean(values))).toBeLessThan(0.05);
    expect(Math.max(...values)).toBeGreaterThan(0.8);
  });
});

// --- Water plane ---------------------------------------------------------------------------------

describe('waterplane.wgsl', () => {
  const WP = `#include "waterplane.wgsl"
    #include "hash.wgsl"
    #include "math.wgsl"
    ${RANDOM_UNIT}`;

  it('builds unit camera rays that point where the pitch says', async () => {
    const out = await gpuF32(
      `${WP}
       fn f(x: vec4f) -> vec4f { return vec4f(waterplane_ray_dir(vec2f(x.x, x.y), 1.5, x.z, 1.0), 0.0); }`,
      rowsOf([
        [0.5, 0.5, 0],
        [0.5, 0.5, -0.3],
        [0.5, 0.5, 0.4],
        [0.5, 0.0, 0],
        [0.5, 1.0, 0],
        [0.0, 0.5, 0],
        [1.0, 0.5, 0],
        [0.13, 0.77, 0.2],
      ]),
    );
    const dir = (i: number) => [out[i * 4] ?? 0, out[i * 4 + 1] ?? 0, out[i * 4 + 2] ?? 0];
    expect(dir(0)[0]).toBeCloseTo(0, 6);
    expect(dir(0)[1]).toBeCloseTo(0, 6);
    expect(dir(0)[2]).toBeCloseTo(1, 6);
    expect(dir(1)[1]).toBeCloseTo(Math.sin(-0.3), 5);
    expect(dir(1)[2]).toBeCloseTo(Math.cos(-0.3), 5);
    expect(dir(2)[1]).toBeCloseTo(Math.sin(0.4), 5);
    // The top of the screen looks up by half the vertical field of view (1.0 rad), the bottom down.
    expect(dir(3)[1]).toBeCloseTo(Math.sin(0.5), 5);
    expect(dir(4)[1]).toBeCloseTo(-Math.sin(0.5), 5);
    // Left of screen is -x, right is +x.
    expect(dir(5)[0]).toBeLessThan(-0.3);
    expect(dir(6)[0]).toBeGreaterThan(0.3);
    for (let i = 0; i < 8; i++) {
      expect(Math.hypot(...dir(i))).toBeCloseTo(1, 5);
    }
  });

  it('intersects the plane and reports misses as negative', async () => {
    const [down, slanted, up, level, below, footprint] = await scalars(WP, [
      'waterplane_hit(vec3f(0.0, 3.0, 0.0), vec3f(0.0, -1.0, 0.0))',
      'waterplane_hit(vec3f(0.0, 3.0, 0.0), normalize(vec3f(0.0, -1.0, 1.0)))',
      'waterplane_hit(vec3f(0.0, 3.0, 0.0), normalize(vec3f(0.0, 0.2, 1.0)))',
      'waterplane_hit(vec3f(0.0, 3.0, 0.0), vec3f(0.0, 0.0, 1.0))',
      'waterplane_hit(vec3f(0.0, -1.0, 0.0), vec3f(0.0, -1.0, 0.0))',
      'waterplane_footprint(100.0, -0.1, 0.001)',
    ]);
    expect(down).toBeCloseTo(3, 6);
    expect(slanted).toBeCloseTo(3 * Math.SQRT2, 5);
    expect(up).toBeLessThan(0);
    expect(level).toBeLessThan(0);
    expect(below).toBeLessThan(0);
    expect(footprint).toBeCloseTo(1, 5);
  });

  const waveInputs = (count: number) =>
    seedRows(count, (i) => [(i % 5) * 0.7 + 0.1, [0.5, 1, 2.5][i % 3] ?? 1, [1, 2][i % 2] ?? 1]);
  // x = (seed, time, strength, scale); positions come from the seed in WGSL.
  const WAVE_POINT = `
    fn point_of(seed: f32) -> vec2f { return (hash_1to2(seed + 0.25) - 0.5) * 80.0; }`;

  it('returns unit normals that follow the analytic gradient', async () => {
    const count = 600;
    const out = await gpuF32(
      `${WP}${WAVE_POINT}
       fn f(x: vec4f) -> vec4f {
         let p = point_of(x.x);
         let w = waterplane_wave(p, x.y, x.z, x.w, 0.0);
         let n = waterplane_normal(w.yz);
         return vec4f(length(n), n.y, n.x, w.x);
       }`,
      rowsOf(
        Array.from({ length: count }, (_, i) => [
          i + 0.5,
          (i % 5) * 0.7 + 0.1,
          [0.5, 1, 2.5][i % 3] ?? 1,
          [1, 2][i % 2] ?? 1,
        ]),
      ),
    );
    for (const length of lane(out, 0)) expect(length).toBeCloseTo(1, 5);
    for (const ny of lane(out, 1)) expect(ny).toBeGreaterThan(0.3);
  });

  it('matches finite differences of its own height (value and slope)', async () => {
    const count = 800;
    const step = 4e-3;
    // Each row evaluates height at p, p +- step in x and in z; the harness packs the 4 neighbours in 2 passes.
    const rows = rowsOf(
      Array.from({ length: count }, (_, i) => [
        i + 0.5,
        (i % 5) * 0.7 + 0.1,
        [0.5, 1, 2.5][i % 3] ?? 1,
        [1, 2][i % 2] ?? 1,
      ]),
    );
    const code = (footprint: number) => `${WP}${WAVE_POINT}
      fn f(x: vec4f) -> vec4f {
        let p = point_of(x.x);
        let h = ${step};
        let w = waterplane_wave(p, x.y, x.z, x.w, ${footprint});
        let dx = (waterplane_height(p + vec2f(h, 0.0), x.y, x.z, x.w, ${footprint}) -
                  waterplane_height(p - vec2f(h, 0.0), x.y, x.z, x.w, ${footprint})) / (2.0 * h);
        let dz = (waterplane_height(p + vec2f(0.0, h), x.y, x.z, x.w, ${footprint}) -
                  waterplane_height(p - vec2f(0.0, h), x.y, x.z, x.w, ${footprint})) / (2.0 * h);
        return vec4f(w.y - dx, w.z - dz, max(abs(w.y), abs(w.z)), w.x);
      }`;
    for (const footprint of [0, 0.4]) {
      const out = await gpuF32(code(footprint), rows);
      const ex = lane(out, 0);
      const ez = lane(out, 1);
      const slope = lane(out, 2);
      for (let i = 0; i < count; i++) {
        expect(Math.abs(ex[i] ?? 1), `row ${i} dx, footprint ${footprint}`).toBeLessThan(6e-3);
        expect(Math.abs(ez[i] ?? 1), `row ${i} dz, footprint ${footprint}`).toBeLessThan(6e-3);
      }
      // Not a flat field, and not an absurd one.
      expect(Math.max(...slope)).toBeGreaterThan(0.1);
      expect(Math.max(...slope)).toBeLessThan(4);
    }
  });

  it('is flat at strength 0, bounded by its amplitude, and fades out with distance', async () => {
    const count = 3000;
    const run = async (strength: number, scale: number, footprint: number) =>
      gpuF32(
        `${WP}${WAVE_POINT}
         fn f(x: vec4f) -> vec4f {
           let p = point_of(x.x) * 4.0;
           let w = waterplane_wave(p, x.y, ${strength}, ${scale}, ${footprint});
           return vec4f(w.x, length(w.yz), waterplane_amplitude(${strength}, ${scale}), 0.0);
         }`,
        waveInputs(count),
      );
    const flat = await run(0, 1, 0);
    for (const value of flat) expect(Math.abs(value)).toBe(0);

    for (const [strength, scale] of [
      [1, 1],
      [2.5, 2],
      [0.4, 1],
    ] as const) {
      const out = await run(strength, scale, 0);
      const heights = lane(out, 0).slice(0, count);
      const amplitude = lane(out, 2)[0] ?? 0;
      const ripple = 0.02 * Math.min(strength, 1.5) * scale;
      expect(amplitude).toBeGreaterThan(0.3 * strength * scale);
      for (const h of heights) expect(Math.abs(h)).toBeLessThanOrEqual(amplitude + ripple);
      expect(Math.abs(mean(heights))).toBeLessThan(0.15 * amplitude);
      expect(Math.max(...heights)).toBeGreaterThan(0.2 * amplitude);
    }

    // Beyond the footprint at which every wavelength is under 2.5 pixels, nothing is left.
    const far = await run(1, 1, 30);
    for (const value of [...lane(far, 0), ...lane(far, 1)]) expect(Math.abs(value)).toBe(0);

    // A moderate footprint keeps the swell and drops the ripples: less slope than the full surface.
    const slopeOf = (out: Float32Array) => mean(lane(out, 1).slice(0, count));
    const full = slopeOf(await run(1, 1, 0));
    const mid = slopeOf(await run(1, 1, 0.6));
    const coarse = slopeOf(await run(1, 1, 3));
    expect(mid).toBeLessThan(full);
    expect(mid).toBeGreaterThan(0);
    expect(coarse).toBeLessThan(mid);
  });

  it('animates smoothly (small time steps make small changes) and moves the pattern', async () => {
    const out = await gpuF32(
      `${WP}
       fn f(x: vec4f) -> vec4f {
         let p = vec2f(3.0, 5.0) * x.x;
         let a = waterplane_height(p, 2.0, 1.0, 1.0, 0.0);
         let b = waterplane_height(p, 2.001, 1.0, 1.0, 0.0);
         let c = waterplane_height(p, 4.0, 1.0, 1.0, 0.0);
         return vec4f(a, b, c, 0.0);
       }`,
      seedRows(300),
    );
    let moved = 0;
    for (let i = 0; i < 300; i++) {
      const a = out[i * 4] ?? 0;
      expect(Math.abs((out[i * 4 + 1] ?? 0) - a)).toBeLessThan(0.01);
      if (Math.abs((out[i * 4 + 2] ?? 0) - a) > 0.05) moved++;
    }
    expect(moved).toBeGreaterThan(100);
  });

  it('crest factor is 0 at the mean level, rises with height and reaches 1', async () => {
    const [below, zero, mid, top, over] = await scalars(WP, [
      'waterplane_crest(-0.5, 1.0, 1.0)',
      'waterplane_crest(0.0, 1.0, 1.0)',
      'waterplane_crest(0.2 * waterplane_amplitude(1.0, 1.0), 1.0, 1.0)',
      'waterplane_crest(waterplane_amplitude(1.0, 1.0), 1.0, 1.0)',
      'waterplane_crest(5.0 * waterplane_amplitude(1.0, 1.0), 1.0, 1.0)',
    ]);
    expect(below).toBe(0);
    expect(zero).toBe(0);
    expect(mid).toBeGreaterThan(0);
    expect(mid).toBeLessThan(1);
    expect(top).toBe(1);
    expect(over).toBe(1);
  });

  it('reflects the view ray and folds downward results above the horizon', async () => {
    const out = await gpuF32(
      `${WP}
       fn f(x: vec4f) -> vec4f {
         let dir = rand_unit(x.x);
         let d = vec3f(dir.x, -abs(dir.y), dir.z);
         let n = normalize(rand_up(x.x + 7.0) * vec3f(1.0, 1.0, 1.0));
         let r = waterplane_reflect(normalize(d), n);
         return vec4f(length(r), r.y, 0.0, 0.0);
       }`,
      seedRows(2000),
    );
    for (const length of lane(out, 0)) expect(length).toBeCloseTo(1, 4);
    for (const y of lane(out, 1)) expect(y).toBeGreaterThanOrEqual(0);
    const [mx, my, mz] = (await scalars(WP, [
      'waterplane_reflect(normalize(vec3f(0.6, -0.8, 0.0)), vec3f(0.0, 1.0, 0.0)).x',
      'waterplane_reflect(normalize(vec3f(0.6, -0.8, 0.0)), vec3f(0.0, 1.0, 0.0)).y',
      'waterplane_reflect(normalize(vec3f(0.6, -0.8, 0.0)), vec3f(0.0, 1.0, 0.0)).z',
    ])) as [number, number, number];
    expect(mx).toBeCloseTo(0.6, 6);
    expect(my).toBeCloseTo(0.8, 6);
    expect(mz).toBeCloseTo(0, 6);
  });

  it('refines the hit toward the displaced surface and is exact on flat water', async () => {
    const count = 400;
    const out = await gpuF32(
      `${WP}
       fn f(x: vec4f) -> vec4f {
         let origin = vec3f(0.0, 4.0, 0.0);
         let dir = normalize(vec3f((x.x - 200.0) * 0.003, -0.25 - 0.002 * x.x, 1.0));
         let t0 = waterplane_hit(origin, dir);
         let flat_t = waterplane_hit_waves(origin, dir, t0, 1.0, 0.0, 1.0, 0.0);
         let t = waterplane_hit_waves(origin, dir, t0, 1.0, 1.0, 2.0, 0.0);
         let p0 = origin + dir * t0;
         let p1 = origin + dir * t;
         // Height error of the surface at each point against the ray's height there.
         let e0 = abs(p0.y - waterplane_height(p0.xz, 1.0, 1.0, 2.0, 0.0));
         let e1 = abs(p1.y - waterplane_height(p1.xz, 1.0, 1.0, 2.0, 0.0));
         return vec4f(flat_t - t0, e0, e1, 0.0);
       }`,
      seedRows(count),
    );
    for (const diff of lane(out, 0).slice(0, count)) expect(diff).toBe(0);
    expect(mean(lane(out, 2).slice(0, count))).toBeLessThan(mean(lane(out, 1).slice(0, count)));
  });

  it('places the sun from elevation and azimuth', async () => {
    const [x, y, z, forward, length] = await scalars(WP, [
      `waterplane_sun_dir(${30 * DEG}, ${90 * DEG}).x`,
      `waterplane_sun_dir(${30 * DEG}, ${90 * DEG}).y`,
      `waterplane_sun_dir(${30 * DEG}, 0.0).z`,
      'waterplane_sun_dir(0.0, 0.0).z',
      `length(waterplane_sun_dir(${12 * DEG}, ${-70 * DEG}))`,
    ]);
    expect(x).toBeCloseTo(Math.cos(30 * DEG), 5);
    expect(y).toBeCloseTo(0.5, 5);
    expect(z).toBeCloseTo(Math.cos(30 * DEG), 5);
    expect(forward).toBeCloseTo(1, 6);
    expect(length).toBeCloseTo(1, 5);
  });
});

// --- Specular ------------------------------------------------------------------------------------

describe('specular.wgsl', () => {
  const SPEC = `#include "specular.wgsl"
    #include "sky.wgsl"
    #include "hash.wgsl"
    #include "math.wgsl"
    ${RANDOM_UNIT}
    fn dir_of(zenith: f32, azimuth: f32) -> vec3f {
      return vec3f(sin(zenith) * cos(azimuth), cos(zenith), sin(zenith) * sin(azimuth));
    }`;

  it('GGX distribution has its peak at n.h = 1 and integrates to one', async () => {
    const alpha = 0.3;
    const steps = 4000;
    const out = await gpuF32(
      `${SPEC}
       fn f(x: vec4f) -> vec4f { return vec4f(spec_ggx_d(cos(x.x), x.y), 0.0, 0.0, 0.0); }`,
      rowsOf(Array.from({ length: steps }, (_, i) => [((i + 0.5) / steps) * (Math.PI / 2), alpha])),
    );
    const d = lane(out, 0);
    expect(d[0]).toBeCloseTo(1 / (Math.PI * alpha * alpha), 0);
    expect(d[0]).toBeGreaterThan(d[200] ?? 0);
    let integral = 0;
    for (let i = 0; i < steps; i++) {
      const theta = ((i + 0.5) / steps) * (Math.PI / 2);
      integral +=
        (d[i] ?? 0) * Math.cos(theta) * Math.sin(theta) * (Math.PI / 2 / steps) * 2 * Math.PI;
    }
    expect(integral).toBeCloseTo(1, 2);
    const [peak] = await scalars(SPEC, ['spec_ggx_d(1.0, 0.1)']);
    expect(peak).toBeCloseTo(1 / (Math.PI * 0.01), -1);
  });

  it('the lobe peaks at the mirror direction', async () => {
    // View at 50 degrees from the normal in the xy plane; the mirror light is the opposite side.
    const viewZenith = 50 * DEG;
    for (const alpha of [0.02, 0.08, 0.2]) {
      const candidates: number[][] = [];
      for (let zenith = 0; zenith <= 89; zenith += 0.5) {
        for (let azimuth = 0; azimuth < 360; azimuth += 1) {
          candidates.push([zenith * DEG, azimuth * DEG]);
        }
      }
      const out = await gpuF32(
        `${SPEC}
         fn f(x: vec4f) -> vec4f {
           let n = vec3f(0.0, 1.0, 0.0);
           let v = dir_of(${viewZenith}, 0.0);
           let l = dir_of(x.x, x.y);
           return vec4f(spec_ggx(n, v, l, ${alpha}), 0.0, 0.0, 0.0);
         }`,
        rowsOf(candidates),
      );
      const values = lane(out, 0).slice(0, candidates.length);
      const best = values.indexOf(Math.max(...values));
      const [zenith = 0, azimuth = 0] = candidates[best] ?? [];
      // The mirror of a view at azimuth 0 is at azimuth 180 degrees and the same zenith. Rough lobes
      // peak slightly toward grazing (off-specular peak), so allow that drift.
      expect(azimuth / DEG).toBeCloseTo(180, -1);
      expect(Math.abs(zenith / DEG - 50)).toBeLessThan(alpha < 0.1 ? 2 : 8);
      expect(Math.max(...values)).toBeGreaterThan(0.05);
    }
  });

  it('spec_sun equals the Fresnel reflectance at a perfect mirror and decays away from it', async () => {
    const [mirror, expected, off1, off5, offWide, rough] = await scalars(SPEC, [
      'spec_sun(vec3f(0.0, 1.0, 0.0), dir_of(0.6981317, 0.0), dir_of(0.6981317, 3.14159265), 0.0, SKY_SUN_ANGULAR_RADIUS)',
      'fresnel_water(cos(0.6981317))',
      'spec_sun(vec3f(0.0, 1.0, 0.0), dir_of(0.6981317, 0.0), dir_of(0.6981317 + 0.01, 3.14159265), 0.0, SKY_SUN_ANGULAR_RADIUS)',
      'spec_sun(vec3f(0.0, 1.0, 0.0), dir_of(0.6981317, 0.0), dir_of(0.6981317 + 0.05, 3.14159265), 0.0, SKY_SUN_ANGULAR_RADIUS)',
      'spec_sun(vec3f(0.0, 1.0, 0.0), dir_of(0.6981317, 0.0), dir_of(0.6981317 + 0.3, 3.14159265), 0.0, SKY_SUN_ANGULAR_RADIUS)',
      'spec_sun(vec3f(0.0, 1.0, 0.0), dir_of(0.6981317, 0.0), dir_of(0.6981317, 3.14159265), 0.5, SKY_SUN_ANGULAR_RADIUS)',
    ]);
    expect(mirror).toBeGreaterThan(0.97 * (expected ?? 1));
    expect(mirror).toBeLessThan(1.03 * (expected ?? 1));
    expect(off1).toBeLessThan(mirror ?? 0);
    expect(off5).toBeLessThan(0.25 * (mirror ?? 0));
    expect(offWide).toBeLessThan(0.01 * (mirror ?? 1));
    // A rough surface spreads the glint: the centre is dimmer.
    expect(rough).toBeLessThan(0.2 * (mirror ?? 0));
  });

  it('spec_sun never exceeds one, whatever the geometry and roughness', async () => {
    const count = 20000;
    const out = await gpuF32(
      `${SPEC}
       fn f(x: vec4f) -> vec4f {
         let n = rand_up(x.x);
         let v = rand_unit(x.x + 1000.0);
         let l = rand_unit(x.x + 2000.0);
         let roughness = fract(x.x * 0.0173);
         return vec4f(spec_sun(n, v, l, roughness, SKY_SUN_ANGULAR_RADIUS), spec_ggx(n, v, l, roughness * roughness), 0.0, 0.0);
       }`,
      seedRows(count),
    );
    for (const value of lane(out, 0)) {
      expect(Number.isFinite(value)).toBe(true);
      expect(value).toBeGreaterThanOrEqual(0);
      expect(value).toBeLessThanOrEqual(1);
    }
    for (const value of lane(out, 1)) {
      expect(Number.isFinite(value)).toBe(true);
      expect(value).toBeGreaterThanOrEqual(0);
    }
  });

  it('is energy conserving: the hemispherical integral of the BRDF stays below one', async () => {
    // Integrate over half vectors (importance-friendly), l = reflect(-v, h), dw_l = 4 (v.h) dw_h.
    const thetaSteps = 192;
    const phiSteps = 96;
    for (const [alpha, viewDeg, low, high] of [
      [0.03, 0, 0.019, 0.022], // near a mirror: the albedo is the Fresnel reflectance
      [0.1, 45, 0.015, 0.04],
      [0.3, 45, 0.012, 0.4],
      [0.6, 80, 0.03, 0.9],
      [0.1, 80, 0.1, 0.7],
    ] as const) {
      const rows: number[][] = [];
      for (let i = 0; i < thetaSteps; i++) {
        const u = (i + 0.5) / thetaSteps;
        for (let j = 0; j < phiSteps; j++) rows.push([u, (j + 0.5) / phiSteps]);
      }
      const out = await gpuF32(
        `${SPEC}
         fn f(x: vec4f) -> vec4f {
           let theta_h = 1.5707963 * x.x * x.x;
           let phi_h = TAU * x.y;
           let h = dir_of(theta_h, phi_h);
           let n = vec3f(0.0, 1.0, 0.0);
           let v = dir_of(${viewDeg * DEG}, 0.0);
           let l = reflect(-v, h);
           let weight = 4.0 * max(dot(v, h), 0.0);
           return vec4f(spec_ggx(n, v, l, ${alpha}) * weight, 0.0, 0.0, 0.0);
         }`,
        rowsOf(rows),
      );
      const values = lane(out, 0);
      let albedo = 0;
      for (let i = 0; i < thetaSteps; i++) {
        const u = (i + 0.5) / thetaSteps;
        const theta = (Math.PI / 2) * u * u;
        const dTheta = Math.PI * u * (1 / thetaSteps);
        for (let j = 0; j < phiSteps; j++) {
          albedo +=
            (values[i * phiSteps + j] ?? 0) * Math.sin(theta) * dTheta * ((2 * Math.PI) / phiSteps);
        }
      }
      expect(albedo, `alpha ${alpha}, view ${viewDeg}`).toBeGreaterThan(low);
      expect(albedo, `alpha ${alpha}, view ${viewDeg}`).toBeLessThan(high);
      expect(albedo).toBeLessThanOrEqual(1);
    }
  });

  // --- sparkle ---

  // x = (px, pz, time, footprint). n, v, l chosen so the half vector equals the macro normal.
  // `tilt` moves the half vector away from the macro normal by that slope along x.
  const SPARKLE = (density: number, sharpness = 1, cell = 0.05, flatN = true, tilt = 0) => `${SPEC}
    fn f(x: vec4f) -> vec4f {
      let n = ${flatN ? 'vec3f(0.0, 1.0, 0.0)' : 'normalize(vec3f(0.1, 1.0, -0.05))'};
      let v = normalize(vec3f(0.0, 0.6, -1.0) + ${flatN ? 'vec3f(0.0)' : 'vec3f(0.1, 0.0, -0.05)'});
      let h_axis = normalize(n + vec3f(${tilt}, 0.0, 0.0));
      let l = normalize(2.0 * dot(h_axis, v) * h_axis - v);
      return vec4f(spec_sparkle(vec2f(x.x, x.y), n, v, l, x.z, x.w, ${cell}, ${density}, ${sharpness}), 0.0, 0.0, 0.0);
    }`;
  const grid = (count: number, time = 0, footprint = 0, extent = 40) =>
    rowsOf(
      Array.from({ length: count }, (_, i) => [
        ((i * 0.6180339887) % 1) * extent,
        ((i * 0.7548776662) % 1) * extent,
        time,
        footprint,
      ]),
    );

  it('sparkle is deterministic, within [0, 1], and absent at zero density', async () => {
    const rows = grid(30000, 3.3, 0.02);
    const first = await gpuF32(SPARKLE(0.6), rows);
    const second = await gpuF32(SPARKLE(0.6), rows);
    // Bit for bit, not just close.
    const bitsOf = (a: Float32Array) => new Uint32Array(a.buffer, a.byteOffset, a.length);
    expect(bitsOf(first).every((word, i) => word === bitsOf(second)[i])).toBe(true);
    const values = lane(first, 0);
    for (const value of values) {
      expect(Number.isFinite(value)).toBe(true);
      expect(value).toBeGreaterThanOrEqual(0);
      expect(value).toBeLessThanOrEqual(1);
    }
    expect(Math.max(...values)).toBeGreaterThan(0.5); // glints really exist
    expect(values.filter((v) => v > 0.05).length).toBeGreaterThan(100);
    expect(values.filter((v) => v > 0.05).length).toBeLessThan(values.length / 3); // and are sparse

    const none = await gpuF32(SPARKLE(0), rows);
    for (const value of lane(none, 0)) expect(value).toBe(0);
  });

  it('sparkle is exactly zero outside the glitter path (half vector far from the normal)', async () => {
    const out = await gpuF32(
      `${SPEC}
       fn f(x: vec4f) -> vec4f {
         let n = vec3f(0.0, 1.0, 0.0);
         let h_axis = normalize(vec3f(0.9, 1.0, 0.0));
         let v = normalize(vec3f(0.0, 0.6, -1.0));
         let l = normalize(2.0 * dot(h_axis, v) * h_axis - v);
         return vec4f(spec_sparkle(vec2f(x.x, x.y), n, v, l, x.z, x.w, 0.05, 1.0, 1.0), 0.0, 0.0, 0.0);
       }`,
      grid(5000, 1, 0.02),
    );
    for (const value of lane(out, 0)) expect(value).toBe(0);
  });

  it('density scales how many glints there are, and sharpness narrows each one', async () => {
    const rows = grid(60000, 1.0, 0.02);
    const total = async (code: string) => mean(lane(await gpuF32(code, rows), 0));
    const full = await total(SPARKLE(1));
    const third = await total(SPARKLE(0.3));
    expect(third / full).toBeGreaterThan(0.2);
    expect(third / full).toBeLessThan(0.4);
    // A sharper window catches fewer facets, so the coverage falls.
    const sharp = await total(SPARKLE(1, 4));
    const soft = await total(SPARKLE(1, 0.5));
    expect(sharp).toBeLessThan(full);
    expect(soft).toBeGreaterThan(full);
  });

  it('glints stay put under small camera motion and change smoothly in time', async () => {
    const base = grid(8000, 2.0, 0.03);
    const moved = new Float32Array(base);
    for (let i = 0; i < moved.length; i += 4) moved[i] = (moved[i] ?? 0) + 2e-4;
    const later = new Float32Array(base);
    for (let i = 2; i < later.length; i += 4) later[i] = (later[i] ?? 0) + 2e-3;
    const code = SPARKLE(0.8);
    const a = lane(await gpuF32(code, base), 0);
    const b = lane(await gpuF32(code, moved), 0);
    const c = lane(await gpuF32(code, later), 0);
    let worstSpace = 0;
    let worstTime = 0;
    for (let i = 0; i < 8000; i++) {
      worstSpace = Math.max(worstSpace, Math.abs((a[i] ?? 0) - (b[i] ?? 0)));
      worstTime = Math.max(worstTime, Math.abs((a[i] ?? 0) - (c[i] ?? 0)));
    }
    expect(worstSpace).toBeLessThan(0.05);
    expect(worstTime).toBeLessThan(0.05);
  });

  it('glints twinkle: a point lights up and dims over time', async () => {
    // With the half vector tilted away from the macro normal, a facet is lit only while its slowly
    // circling normal passes through the half vector.
    const points = 1500;
    const times = 240;
    const base = grid(points, 0, 0.02, 10);
    const rows: number[][] = [];
    for (let i = 0; i < points; i++) {
      for (let k = 0; k < times; k++) {
        rows.push([base[i * 4] ?? 0, base[i * 4 + 1] ?? 0, k * 0.1, 0.02]);
      }
    }
    const values = lane(await gpuF32(SPARKLE(1, 1, 0.05, true, 0.12), rowsOf(rows)), 0);
    let twinkling = 0;
    let everLit = 0;
    for (let i = 0; i < points; i++) {
      const series = values.slice(i * times, (i + 1) * times);
      if (Math.max(...series) > 0.5) everLit++;
      if (Math.max(...series) > 0.5 && Math.min(...series) < 0.1) twinkling++;
    }
    expect(everLit).toBeGreaterThan(20);
    expect(twinkling).toBeGreaterThan(10);
  });

  it('anti-aliasing: coarser footprints keep the mean brightness and vary continuously', async () => {
    const mean0 = mean(lane(await gpuF32(SPARKLE(1), grid(120000, 0.5, 0, 200)), 0));
    for (const footprint of [0.1, 0.4, 1.5]) {
      const m = mean(lane(await gpuF32(SPARKLE(1), grid(120000, 0.5, footprint, 400)), 0));
      expect(m / mean0, `footprint ${footprint}`).toBeGreaterThan(0.6);
      expect(m / mean0, `footprint ${footprint}`).toBeLessThan(1.6);
    }
    // Continuity in the footprint (the level blend has no seam where the level changes).
    const fine = grid(6000, 0.5, 0.0333333, 40);
    const finer = grid(6000, 0.5, 0.0333333 * 1.001, 40);
    const a = lane(await gpuF32(SPARKLE(1), fine), 0);
    const b = lane(await gpuF32(SPARKLE(1), finer), 0);
    for (let i = 0; i < 6000; i++) expect(Math.abs((a[i] ?? 0) - (b[i] ?? 0))).toBeLessThan(0.02);
  });

  it('sparkle follows the macro normal: tilted waves still catch glints', async () => {
    const values = lane(await gpuF32(SPARKLE(1, 1, 0.05, false), grid(40000, 0, 0.02, 40)), 0);
    expect(Math.max(...values)).toBeGreaterThan(0.5);
    expect(Math.min(...values)).toBeGreaterThanOrEqual(0);
  });
});

// --- Subsurface scattering -----------------------------------------------------------------------

describe('sss.wgsl', () => {
  const SSS = `#include "sss.wgsl"
    #include "hash.wgsl"
    #include "math.wgsl"
    ${RANDOM_UNIT}`;

  it('is never negative or non-finite, for any geometry and any parameters', async () => {
    const count = 20000;
    const out = await gpuF32(
      `${SSS}
       fn f(x: vec4f) -> vec4f {
         let v = rand_unit(x.x);
         let l = rand_unit(x.x + 5000.0);
         let n = rand_unit(x.x + 9000.0);
         let r = hash_1to3(x.x);
         return vec4f(sss_crest(v, l, n, r.x * 1.4 - 0.2, r.y * 2.0, 0.5 + r.z * 20.0), 0.0, 0.0, 0.0);
       }
       fn hash_1to3(s: f32) -> vec3f { return hash_2to3(vec2f(s, s * 0.37)); }`,
      seedRows(count),
    );
    const values = lane(out, 0);
    for (const value of values) {
      expect(Number.isFinite(value)).toBe(true);
      expect(value).toBeGreaterThanOrEqual(0);
      expect(value).toBeLessThanOrEqual(1);
    }
    expect(values.filter((v) => v > 0.01).length).toBeGreaterThan(200); // not dead everywhere
  });

  it('is exactly zero when the viewer is on the sunny side, looking away from the light', async () => {
    const count = 8000;
    const out = await gpuF32(
      `${SSS}
       fn f(x: vec4f) -> vec4f {
         let l = rand_up(x.x);
         let n = rand_up(x.x + 5000.0);
         // The eye sits where the sun is (view == light) or tilted slightly around it.
         let v = normalize(l + 0.15 * rand_unit(x.x + 9000.0));
         let pass_c = f32(dot(v, l) > 0.0);
         let r = hash_2to3(vec2f(x.x, 1.0));
         return vec4f(sss_crest(v, l, n, r.x, r.y, 1.0 + 15.0 * r.z) * pass_c, 0.0, 0.0, 0.0);
       }`,
      seedRows(count),
    );
    for (const value of lane(out, 0)) expect(value).toBe(0);
  });

  it('is zero with the sun below the horizon and for a viewer on the sun side of any normal', async () => {
    const [below, sunny] = await scalars(SSS, [
      'sss_crest(normalize(vec3f(0.0, 0.3, -1.0)), normalize(vec3f(0.0, -0.2, 1.0)), vec3f(0.0, 1.0, 0.0), 1.0, 0.6, 4.0)',
      'sss_crest(normalize(vec3f(0.0, 0.3, 1.0)), normalize(vec3f(0.0, 0.3, 1.0)), vec3f(0.0, 1.0, 0.0), 1.0, 0.6, 4.0)',
    ]);
    expect(below).toBe(0);
    expect(sunny).toBe(0);
  });

  it('peaks when looking toward the sun through a tall wave and grows with crest height', async () => {
    const lowSun = 'normalize(vec3f(0.0, 0.12, 1.0))';
    const [lookingAt, crestHigh, crestLow, sideways, farAway] = await scalars(SSS, [
      // Eye behind the wave looking toward the sun: its view vector is the direction the light leaves along.
      `sss_crest(normalize(vec3f(0.0, 0.5, 1.0) * vec3f(1.0, 1.0, -1.0)), ${lowSun}, vec3f(0.0, 1.0, 0.0), 1.0, 0.6, 4.0)`,
      `sss_crest(normalize(vec3f(0.0, 0.4, -1.0)), ${lowSun}, normalize(vec3f(0.0, 1.0, -0.4)), 1.0, 0.6, 4.0)`,
      `sss_crest(normalize(vec3f(0.0, 0.4, -1.0)), ${lowSun}, normalize(vec3f(0.0, 1.0, -0.4)), 0.0, 0.6, 4.0)`,
      `sss_crest(normalize(vec3f(1.0, 0.4, -0.3)), ${lowSun}, normalize(vec3f(0.0, 1.0, -0.4)), 1.0, 0.6, 4.0)`,
      `sss_crest(normalize(vec3f(0.0, 0.9, -0.1)), ${lowSun}, normalize(vec3f(0.0, 1.0, -0.4)), 1.0, 0.6, 4.0)`,
    ]);
    expect(crestHigh).toBeGreaterThan(0.2);
    expect(crestHigh).toBeGreaterThan(2 * (crestLow ?? 1));
    expect(crestLow).toBeGreaterThan(0); // a calm surface is not dead
    expect(sideways).toBeLessThan(0.25 * (crestHigh ?? 0));
    expect(farAway).toBeLessThan(0.5 * (crestHigh ?? 0));
    expect(lookingAt).toBeGreaterThan(0);
  });

  it('the colour form multiplies the scatter and light colours', async () => {
    const out = await gpuF32(
      `${SSS}
       fn f(x: vec4f) -> vec4f {
         let v = normalize(vec3f(0.0, 0.4, -1.0));
         let l = normalize(vec3f(0.0, 0.12, 1.0));
         let n = normalize(vec3f(0.0, 1.0, -0.4));
         let s = sss_crest(v, l, n, 1.0, 0.6, 4.0);
         let c = sss_crest_color(v, l, n, 1.0, vec3f(0.1, 0.8, 0.5), vec3f(2.0, 1.0, 0.5), 0.6, 4.0);
         return vec4f(c.x / max(s, 1e-6), c.y / max(s, 1e-6), c.z / max(s, 1e-6), s);
       }`,
      rowsOf([[0]]),
    );
    expect(out[3]).toBeGreaterThan(0.1);
    expect(out[0]).toBeCloseTo(0.2, 4);
    expect(out[1]).toBeCloseTo(0.8, 4);
    expect(out[2]).toBeCloseTo(0.25, 4);
  });
});

// --- Cubemap rendering ---------------------------------------------------------------------------

describe('src/gpu/cubemap.ts', () => {
  const DIRECTION_SOURCE = {
    label: 'test/direction-cube.wgsl',
    code: `fn cubemap_source(dir: vec3f) -> vec3f { return dir * 0.5 + 0.5; }`,
  };

  it('documents the six faces in layer order', () => {
    expect(CUBE_FACE_NAMES).toEqual(['+x', '-x', '+y', '-y', '+z', '-z']);
    // The centre of each face looks along its own axis.
    const axes = [
      [1, 0, 0],
      [-1, 0, 0],
      [0, 1, 0],
      [0, -1, 0],
      [0, 0, 1],
      [0, 0, -1],
    ];
    axes.forEach((axis, face) => {
      const centre = cubeFaceDirection(face, 0.5, 0.5);
      axis.forEach((c, i) => expect(centre[i] ?? 9).toBeCloseTo(c, 9));
    });
    expect(() => cubeFaceDirection(6, 0, 0)).toThrow(RangeError);
  });

  it('writes the right direction into the centre of every face (+-X, +-Y, +-Z)', async () => {
    const device = await getTestDevice();
    const cube = createCubemap(device, {
      size: 9,
      source: DIRECTION_SOURCE,
      extraUsage: GPUTextureUsage.COPY_SRC,
      label: 'test cube',
    });
    await renderCube(device, cube);
    const expected = [
      [1, 0.5, 0.5],
      [0, 0.5, 0.5],
      [0.5, 1, 0.5],
      [0.5, 0, 0.5],
      [0.5, 0.5, 1],
      [0.5, 0.5, 0],
    ];
    for (let face = 0; face < 6; face++) {
      const { data, width } = await readLayer(device, cube.texture, face);
      const centre = (4 * width + 4) * 4;
      for (let c = 0; c < 3; c++) {
        expect(data[centre + c], `${CUBE_FACE_NAMES[face]} channel ${c}`).toBeCloseTo(
          expected[face]?.[c] ?? 0,
          3,
        );
      }
      expect(data[centre + 3]).toBeCloseTo(1, 3);
    }
    cube.destroy();
  });

  it('every texel of every face looks along the documented direction', async () => {
    const device = await getTestDevice();
    const size = 8;
    const cube = createCubemap(device, {
      size,
      source: DIRECTION_SOURCE,
      extraUsage: GPUTextureUsage.COPY_SRC,
    });
    await renderCube(device, cube);
    for (let face = 0; face < 6; face++) {
      const { data } = await readLayer(device, cube.texture, face);
      for (let y = 0; y < size; y++) {
        for (let x = 0; x < size; x++) {
          const dir = cubeFaceDirection(face, (x + 0.5) / size, (y + 0.5) / size);
          const length = Math.hypot(...dir);
          for (let c = 0; c < 3; c++) {
            const expected = (dir[c] ?? 0) / length / 2 + 0.5;
            expect(data[(y * size + x) * 4 + c] ?? 0).toBeCloseTo(expected, 2);
          }
        }
      }
    }
    cube.destroy();
  });

  it('hardware cube sampling returns the colour of the direction it was given', async () => {
    const device = await getTestDevice();
    const cube = createCubemap(device, { size: 64, source: DIRECTION_SOURCE });
    await renderCube(device, cube);
    const axes = [
      [1, 0, 0],
      [-1, 0, 0],
      [0, 1, 0],
      [0, -1, 0],
      [0, 0, 1],
      [0, 0, -1],
    ];
    const random: number[][] = [];
    let seed = 12345;
    const next = () =>
      ((seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0) / 4294967296) * 2 - 1;
    while (random.length < 300) {
      const v = [next(), next(), next()];
      const length = Math.hypot(...(v as [number, number, number]));
      if (length > 0.2 && length <= 1) random.push(v.map((c) => c / length));
    }
    const dirs = [...axes, ...random];
    const out = await sampleCube(device, cube, dirs);
    dirs.forEach((dir, i) => {
      for (let c = 0; c < 3; c++) {
        expect(
          out[i * 4 + c] ?? 0,
          `dir ${dir.map((x) => x.toFixed(2)).join(',')} channel ${c}`,
        ).toBeCloseTo((dir[c] ?? 0) * 0.5 + 0.5, 1);
      }
    });
    // Tight check on a handful of directions inside faces, away from edges.
    const inner = [
      [0.9, 0.2, -0.3],
      [-0.8, -0.3, 0.2],
      [0.1, 0.95, 0.2],
      [0.2, -0.9, -0.1],
      [0.3, -0.2, 0.9],
      [-0.1, 0.25, -0.95],
    ];
    const tight = await sampleCube(device, cube, inner);
    inner.forEach((dir, i) => {
      const length = Math.hypot(...(dir as [number, number, number]));
      for (let c = 0; c < 3; c++)
        expect(tight[i * 4 + c] ?? 0).toBeCloseTo(((dir[c] ?? 0) / length) * 0.5 + 0.5, 2);
    });
    cube.destroy();
  });

  it('keeps HDR values, passes user inputs, and fills the mip chain', async () => {
    const device = await getTestDevice();
    const cube = createCubemap(device, {
      size: 16,
      mipmaps: true,
      extraUsage: GPUTextureUsage.COPY_SRC,
      source: {
        label: 'test/hdr-cube.wgsl',
        code: `fn cubemap_source(dir: vec3f) -> vec3f { return cubemap_inputs.user[0].xyz + cubemap_inputs.user[1].xyz * 0.0; }`,
      },
    });
    expect(cube.texture.mipLevelCount).toBe(5);
    expect(cube.texture.depthOrArrayLayers).toBe(6);
    await renderCube(device, cube, [40, 0.25, 0, 0]);
    for (const mip of [0, 2, 4]) {
      const { data } = await readLayer(device, cube.texture, 3, mip);
      expect(data[0]).toBeCloseTo(40, 0);
      expect(data[1]).toBeCloseTo(0.25, 3);
      expect(data[2]).toBeCloseTo(0, 3);
    }
    // Re-rendering with new inputs changes the result.
    await renderCube(device, cube, [1, 2, 3, 0]);
    const { data } = await readLayer(device, cube.texture, 5, 0);
    expect(data[0]).toBeCloseTo(1, 3);
    expect(data[1]).toBeCloseTo(2, 3);
    expect(data[2]).toBeCloseTo(3, 3);
    cube.destroy();
  });

  it('renders the real sky into a cubemap that agrees with calling the function directly', async () => {
    const device = await getTestDevice();
    const cube = createCubemap(device, {
      size: 32,
      extraUsage: GPUTextureUsage.COPY_SRC,
      source: {
        label: 'test/sky-cube.wgsl',
        code: `#include "sky.wgsl"
          fn cubemap_source(dir: vec3f) -> vec3f { return sky_radiance(dir, normalize(cubemap_inputs.user[0].xyz)); }`,
      },
    });
    const sun = [0.3, 0.45, 0.84];
    await renderCube(device, cube, sun);
    // Face -Y looks down: uniform ground colour, equal to the sky function straight down.
    const { data } = await readLayer(device, cube.texture, 3);
    const [expected] = await scalars(`#include "sky.wgsl"`, [
      `sky_radiance(vec3f(0.0, -1.0, 0.0), normalize(vec3f(${sun.join(', ')}))).y`,
    ]);
    expect(data[(16 * 32 + 16) * 4 + 1]).toBeCloseTo(expected ?? 0, 2);
    // The sun's face (+Z, sun toward +z) holds HDR values above 1 near the sun.
    const sunFace = await readLayer(device, cube.texture, 4);
    let max = 0;
    for (let i = 0; i < sunFace.data.length; i += 4) max = Math.max(max, sunFace.data[i] ?? 0);
    expect(max).toBeGreaterThan(1); // the halo; the 0.0093 rad disc is smaller than a 32-texel face resolves
    cube.destroy();
  });
});

// --- Demo effects: no parameter value may produce NaN, infinity or a negative colour -------------

describe('surface optics demos stay finite across their parameter ranges', () => {
  const WIDTH = 96;
  const HEIGHT = 64;
  const canvas = Object.assign(new EventTarget(), {
    width: WIDTH,
    height: HEIGHT,
    style: {},
    getBoundingClientRect: () => ({ left: 0, top: 0, width: WIDTH, height: HEIGHT }),
  }) as unknown as HTMLCanvasElement;

  it.each(['fresnel-sky', 'sun-glint', 'crest-sss'])('%s', async (id) => {
    const device = await getTestDevice();
    const { loadEffect } = await import('../src/effects/registry.ts');
    const { defaultParams } = await import('../src/effects/types.ts');
    const { noProfiler } = await import('../src/gpu/profiler.ts');
    const effect = await loadEffect(id);
    // Float target: the sRGB encode and 8-bit clamp would hide a NaN as black.
    const target = device.createTexture({
      size: [WIDTH, HEIGHT],
      format: 'rgba16float',
      usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC,
    });
    const params = defaultParams(effect.params) as Record<string, unknown>;
    const instance = await effect.create({
      device,
      format: 'rgba16float',
      canvas,
      params: params as never,
      profiler: noProfiler,
    });
    instance.resize(WIDTH, HEIGHT, 1);

    let frame = 0;
    const check = async (label: string) => {
      for (let i = 0; i < 2; i++) {
        const encoder = device.createCommandEncoder();
        instance.frame({ time: frame / 60, delta: 1 / 60, frame }, encoder, target.createView());
        device.queue.submit([encoder.finish()]);
        frame++;
      }
      const { data } = await readLayer(device, target, 0);
      for (let i = 0; i < data.length; i++) {
        const value = data[i] ?? Number.NaN;
        if (!(value >= -0.01 && value <= 1.5)) {
          throw new Error(
            `${id}: ${label} gave ${value} at channel ${i % 4}, pixel ${(i / 4) | 0}`,
          );
        }
      }
    };

    await check('defaults');
    for (const [name, preset] of Object.entries(effect.presets ?? {})) {
      Object.assign(params, defaultParams(effect.params), preset);
      await check(`preset ${name}`);
    }
    for (const [key, def] of Object.entries(effect.params)) {
      const values: unknown[] =
        def.type === 'number'
          ? [def.min, def.max, (def.min + def.max) / 2]
          : def.type === 'boolean'
            ? [true, false]
            : def.type === 'select'
              ? [...def.options]
              : [
                  [0, 0, 0],
                  [1, 1, 1],
                ];
      for (const value of values) {
        Object.assign(params, defaultParams(effect.params));
        params[key] = value;
        await check(`${key} = ${JSON.stringify(value)}`);
      }
    }
    // The extremes together, and a camera at the water line looking straight down.
    Object.assign(params, defaultParams(effect.params));
    for (const [key, def] of Object.entries(effect.params)) {
      if (def.type === 'number') params[key] = key.startsWith('camera') ? def.min : def.max;
    }
    await check('every number at its extreme');
    instance.destroy();
    target.destroy();
  });
});
